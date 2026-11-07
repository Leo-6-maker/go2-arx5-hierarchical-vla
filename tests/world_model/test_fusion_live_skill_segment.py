"""Hermetic consumption-loop tests; none constructs or qualifies a native world."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_live_skill_segment as segment
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as original
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.next2_episode import state_vector
from scripts.pi0292.eval_contract import matrix10
from test_fusion_reference_qualification_contract import reference
from test_fusion_trajectory_qualification import FakeAdapter, BASE

B = 4103


def eef(tcp, grip=.95):
    return np.r_[tcp[:3, 3], tcp[:3, :3][:2].reshape(-1), grip]


class SegmentAdapter(FakeAdapter):
    """Reuses the old fake's sequence assertions, with real parsing/join helpers."""
    def __init__(self, output):
        super().__init__(dict(phase=segment.PHASE, live_skill_start_step=B,
            physics_dt_s=.002, WBC_changes=False), output)
        self.step = B
        self.trace_sealed = False
        self.reference_sequence = 101
        self.latest['mode'].data = 'WBC'
        self.cancel_calls, self.advances, self.installed_sends = 0, [], []
        self.install_fault = self.effect_fault = None
        self.reference_qualification = {'preserved': 'initializer-report'}

    def prepare_dynamic(self):
        raise AssertionError('MUST_NOT_PREPARE_ANOTHER_WORLD')

    def seal_trace(self, step):
        raise AssertionError('CALLER_OWNS_FINAL_SEAL')

    def cancel_reference(self):
        self.cancel_calls += 1
        self.reference_sequence += 1
        self.latest['mode'].data = 'HOLD'
        return {'kind': 'fake-pre-query-cancel'}

    def measured_task_state(self):
        sample = super().measured_task_state()
        sample['qvel'][:] = (self.step-2503)*1e-6
        sample['ctrl'][:] = (self.step-2503)*1e-7
        sample['tcp'][:3, 3] = [.2+self.step*1e-7, -.1, .4]
        sample['object_pose'][:3, 3] = [2., 3., .2]  # No fake task success.
        if self.advances and self.effect_fault == 'unsafe':
            sample['safety_valid'] = False
        if self.advances and self.effect_fault == 'native-abort':
            sample['native_abort'] = True
        if self.advances and self.effect_fault == 'nonfinite':
            sample['qvel'][0] = np.nan
        return sample

    def reference_events(self):
        extra = ([{'kind': 'lifecycle', 'reason': 'PREFIX_TTL_EXPIRED'}]
                 if self.effect_fault == 'ttl' and self.advances else [])
        return self.installed_sends+extra

    def install_reference(self, action):
        assert not self.trace_sealed
        np.testing.assert_array_equal(action, self.program.preview((self.step-B)//25))
        self.reference_sequence += 1
        source = round(self.step*.002*1e9)/1e9  # Original integer-nanosecond ROS clock.
        ref = reference(source)
        wire = json.dumps(dict(sequence=self.reference_sequence, step=self.step,
                              action=action.tolist()), sort_keys=True)
        wire_sha = hashlib.sha256(wire.encode()).hexdigest()
        ref.update(wire=wire, wire_sha256=wire_sha, action_EEF10=action.tolist())
        for row in ref['receiver_events']:
            row['ref_id'] = wire_sha
            if row['kind'] == 'accept':
                row['raw'] = wire
        installed = ref['receiver_events'][-1]
        installed.update(times=[source+i*.05 for i in range(40)],
            poses=[np.r_[row[:7], BASE].tolist() for row in action],
            inputs=[[0.]]*40, monotonic_s=100.+len(self.installs))
        if self.install_fault == 'missing-ack':
            ref['receiver_events'] = [installed]
        elif self.install_fault == 'changed-suffix':
            ref['action_EEF10'][5][0] += .001
        elif self.install_fault == 'stale-source':
            ref = reference(source-.1)
            ref['action_EEF10'] = action.tolist()
        self.installed_sends.append(copy.deepcopy(installed))
        self.active_reference = ref
        self.installs.append((self.step, action.copy()))
        self.latest['mode'].data = 'WBC'
        return ref

    def advance_to(self, step):
        assert self.installs and self.latest['mode'].data == 'WBC'
        assert self.events[-1]['phase'] == 'live-skill-before-atom'
        assert self.events[-1]['actual_history_before_atom']['actual_history_native_steps'][-1] == self.step
        self.advances.append(step)
        packet = super().advance_to(step)
        installed = self.active_reference['receiver_events'][-1]
        states = copy.deepcopy(installed['poses'])
        if self.effect_fault == 'wrong-target':
            states[-1][13] += .001  # Changing BASE matters to the real join.
        received = installed['monotonic_s']+1.
        if self.effect_fault == 'before-send':
            received = installed['monotonic_s']-.001
        self.ros_evidence.append(dict(topic_key='policy', received_monotonic_s=received,
            message=dict(plan_target_trajectories=dict(time_trajectory=installed['times'],
                state_trajectory=[dict(value=p) for p in states],
                input_trajectory=[dict(value=p) for p in installed['inputs']]))))
        if self.effect_fault == 'gripper-sequence':
            self.grip_memory['values']['sequence'] -= 1
        if self.effect_fault == 'gripper-expiry':
            self.grip_memory['values']['expiry'] -= .01
        if self.effect_fault == 'hold-mode':
            self.latest['mode'].data = 'HOLD'
        if self.effect_fault == 'wrong-pause':
            packet['step'] += 25
        return packet


@pytest.fixture
def flow(tmp_path, monkeypatch):
    from scripts.pi0294 import task_state
    from go2arx5_vla.world_model import next2_contact
    adapter = SegmentAdapter(tmp_path)
    samples = []
    for step in range(2503, B+1, 25):
        adapter.step = step
        samples.append(adapter.measured_task_state())
    raw = np.repeat(eef(samples[-1]['tcp'], 1.)[None], 40, axis=0)
    raw[:, 0] += np.linspace(0., .00039, 40)
    raw[:, 2] += np.linspace(0., -.0002, 40)
    raw[:, 9] = np.linspace(1., .999, 40)
    tcp = samples[-1]['tcp']
    program = TrajectoryAdaptation(raw, tcp, np.eye(4), tcp, tcp, BASE, 1.)
    adapter.program = program
    adapter.spec['max_ticks'] = B+25*(program.end_atom+8)
    windows = []
    for index, (first, last) in enumerate(zip(samples[:-1], samples[1:])):
        row = eef(first['tcp'])
        packet = dict(sequence=index, source_mpc_s=first['time_s'], opening=row[9],
                      expiry=first['time_s']+.4, native_start_step=first['native_step'])
        wrapper = dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY', values=dict(
            initialized=True, has_command=True, steps=last['native_step'], sequence=index,
            opening=row[9], expiry=packet['expiry']))
        windows.append(dict(native_start_step=first['native_step'], native_end_step=last['native_step'],
            actual_ticks=25, solver_policies_with_exact_installed_target=[{'fixture_past_join': True}],
            gripper=dict(submitted=packet, command_consumed=True, consumed_wrapper=wrapper,
                         actual_finger_q_m=last['finger_q'].tolist()),
            actual_installed_consumed_EEF10_BASE7=np.r_[row, BASE].tolist(),
            actual_TCP=last['tcp'].tolist()))
        adapter.grip_packets.append(packet)
        adapter.reference_history.append(dict(end_native_step=last['native_step'], pose=matrix10(row)))
    # A query is permitted only after an actual cancel/HOLD in the future
    # wrapper. Its stale active_reference must never be reused by this helper.
    adapter.active_reference = dict(reference(B*.002-1.), action_EEF10=raw.tolist())
    adapter.cancel_reference()
    # The future wrapper, not this loop or fake install, must perform and ACK
    # the original WBC restoration after cancel/query. This is only a fixture.
    adapter.latest['mode'].data = 'WBC'
    write = task_state.write_gripper
    def record(path, sequence, source, opening):
        assert sequence == len(adapter.grip_packets)
        write(path, sequence, source, opening)
        adapter.grip_packets.append(dict(sequence=sequence, source_mpc_s=source,
                                         opening=opening, expiry=source+.4))
    monkeypatch.setattr(task_state, 'write_gripper', record)
    monkeypatch.setattr(next2_contact, 'gripper_memory', lambda a: a.grip_memory)
    return adapter, program, samples, windows


def run(flow):
    adapter, program, samples, windows = flow
    return segment.run_consumed_segment(adapter, program, start_step=B,
        end_step=adapter.spec['max_ticks'], past_samples=samples, past_windows=windows)


def test_registered_shadow_forecasts_precede_each_original_install(flow,monkeypatch):
    from go2arx5_vla.world_model import fusion_program_forecast_v1
    adapter,program,samples,windows=flow
    adapter.spec['program_response_forecast']={'fixture_only':True}
    calls=[]
    class Shadow:
        def __init__(self,spec,output):
            self.models={'RAW':None,'SCALED':None,'NO_FUTURE_ACTION':None}
            self.records=calls;self.wall_s=0.
        def save(self,history,past,path,*,install_atom):
            assert history['actual_history_native_steps'][-1]==adapter.step==B+25*install_atom
            assert len(adapter.installs)==len(calls)
            assert len(past)==8 and past[-1]['native_end_step']==adapter.step
            np.testing.assert_array_equal(path,program.reference_path)
            calls.append(install_atom)
    monkeypatch.setattr(fusion_program_forecast_v1,'ShadowForecaster',Shadow)
    result,_,_=run(flow)
    assert calls==list(range(0,result['complete_suffix_atoms'],7))
    assert len(adapter.installs)==len(calls)
    assert result['model_reference_changes']==0 and result['learned_models_loaded']==3
    assert result['all_40_knots_consumed'] is True


def test_shadow_failure_preserves_attempt_without_any_new_native_effect(flow,monkeypatch):
    from go2arx5_vla.world_model import fusion_program_forecast_v1
    adapter,program,samples,windows=flow
    adapter.spec['program_response_forecast']={'fixture_only':True}
    class Shadow:
        def __init__(self,spec,output):self.models={};self.records=[]
        def save(self,*args,**kwargs):raise ValueError('FORECAST_SAVE_FAILED')
    monkeypatch.setattr(fusion_program_forecast_v1,'ShadowForecaster',Shadow)
    with pytest.raises(ValueError,match='FORECAST_SAVE_FAILED'):run(flow)
    assert adapter.installs==[] and adapter.advances==[]
    result=json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert result['complete_suffix_atoms']==0 and result['status']=='FAILED_RETAINED_NO_RETRY'


def test_post_query_hold_preserves_raw_proposal_without_claiming_its_execution(flow):
    from dataclasses import replace
    from test_fusion_program_reference_intervention_v1 import document
    adapter,program,samples,windows=flow
    hold=replace(program,reference_intervention=document('HOLD'));adapter.program=hold
    report,_,_=run((adapter,hold,samples,windows))
    assert report['post_query_HOLD_control'] is True
    assert report['complete_suffix_atoms']==(adapter.spec['max_ticks']-B)//25
    assert report['all_40_knots_consumed'] is False
    assert report['raw_knot_consumed']==[False]*40
    np.testing.assert_array_equal(report['original_raw_H40'],program.decoded_H40)
    assert report['command_clips']==0 and report['VLA_inferences']==0


def test_measurement_observer_rejects_before_any_new_install(flow):
    adapter,program,samples,windows=flow
    def reject(sample):
        raise ValueError('DIAGNOSTIC_SCOPE_REJECT')
    with pytest.raises(ValueError,match='DIAGNOSTIC_SCOPE_REJECT'):
        segment.run_consumed_segment(adapter,program,start_step=B,end_step=adapter.spec['max_ticks'],
            past_samples=samples,past_windows=windows,sample_observer=reject)
    assert adapter.installs==[] and adapter.advances==[]


def test_measurement_observer_stops_after_first_actual_effect_and_preserves_failure(flow):
    adapter,program,samples,windows=flow
    seen=[]
    def reject_after_effect(sample):
        seen.append(sample['native_step'])
        if sample['native_step']>B:raise ValueError('CONTACT_SCOPE_REJECT')
    with pytest.raises(ValueError,match='CONTACT_SCOPE_REJECT'):
        segment.run_consumed_segment(adapter,program,start_step=B,end_step=adapter.spec['max_ticks'],
            past_samples=samples,past_windows=windows,sample_observer=reject_after_effect)
    assert seen==[B,B+25] and adapter.advances==[B+25]
    failed=json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert failed['status']=='FAILED_RETAINED_NO_RETRY' and failed['complete_suffix_atoms']==0
    assert failed['last_observed_native_step']==B+25 and not adapter.trace_sealed


def test_fresh_install_after_query_continues_same_history_sequences_and_real_solver_join(flow):
    adapter, program, past_samples, past_windows = flow
    initial_spec = copy.deepcopy(adapter.spec)
    initial_history = copy.deepcopy(adapter.reference_history)
    qualification = adapter.reference_qualification
    history_object = adapter.reference_history
    seq = adapter.reference_sequence
    report, samples, windows = run(flow)
    count = program.end_atom+8
    assert report['complete_suffix_atoms'] == count
    assert report['solver_target_confirmed_atoms'] == report['gripper_confirmed_atoms'] == count
    assert report['all_40_knots_consumed'] and all(report['raw_knot_consumed'])
    assert report['extra_final_hold_atoms'] == 0
    assert adapter.installs[0][0] == B
    assert [step for step, _ in adapter.installs] == list(range(B, adapter.spec['max_ticks'], 175))
    assert adapter.reference_sequence == seq+len(adapter.installs)
    assert adapter.cancel_calls == 1 and not adapter.trace_sealed
    assert adapter.reference_history is history_object
    for actual, expected in zip(adapter.reference_history, initial_history):
        assert actual['end_native_step'] == expected['end_native_step']
        np.testing.assert_array_equal(actual['pose'], expected['pose'])
    assert adapter.reference_qualification is qualification and adapter.spec == initial_spec
    assert len(past_samples) == 65 and len(past_windows) == 64
    assert samples[:65] == past_samples and windows[:64] == past_windows
    assert [p['sequence'] for p in adapter.grip_packets] == list(range(64+count))
    assert len(adapter.reference_history) == len(windows) == 64+count
    assert len(samples) == 65+count
    assert report['full_MPC_hidden_state_restored'] is False
    assert report['controller_selected_reference'] == 'DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED'
    assert report['Processor_decodes'] == report['VLA_inferences'] == report['WM_updates'] == 0
    for elapsed, row in enumerate(report['windows']):
        before = row['actual_history_before_atom']
        np.testing.assert_array_equal(before['actual_state_history'],
            [state_vector(s) for s in samples[56+elapsed:65+elapsed]])
        assert before['actual_state_history_shape'] == [9, 106]
        assert before['past_reference_native_end_steps'][-1] == B+25*elapsed
        assert row['gripper']['submitted']['sequence'] == 64+elapsed
        assert row['gripper']['command_consumed'] is True
        assert row['solver_policies_with_exact_installed_target']
        assert row['clock']['expiry_margin_s'] > 0
        np.testing.assert_array_equal(row['nominal_submitted_EEF10'], program.preview(elapsed)[0])
        np.testing.assert_array_equal(row['actual_TCP'], samples[65+elapsed]['tcp'])
    evidence = json.loads((adapter.output/'LIVE_SKILL_SEGMENT_DECISION.private.json').read_text())
    assert evidence['first_gripper_sequence'] == 64 and evidence['future_response_observed'] is False
    np.testing.assert_array_equal(evidence['raw_H40'], program.decoded_H40)
    assert json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())['all_40_knots_consumed']


def test_fixed_preregistered_excess_budget_only_consumes_final_hold(flow):
    adapter, program, *_ = flow
    adapter.spec['max_ticks'] += 75
    report, samples, windows = run(flow)
    assert report['extra_final_hold_atoms'] == 3
    assert adapter.step == adapter.spec['max_ticks']
    for row in report['windows'][program.end_atom:]:
        np.testing.assert_array_equal(row['nominal_submitted_EEF10'], program.reference_path[-1])
    assert len(windows) == len(samples)-1


@pytest.mark.parametrize('fault', ['phase', 'duck-program', 'sealed', 'seal-marker', 'sealed-summary', 'short-budget',
    'start-grid', 'end-grid', 'unregistered-B', 'unregistered-E', 'dt', 'wbc', 'base',
    'missing-history', 'past-gap', 'past-seq', 'past-solver', 'past-grip',
    'past-tcp', 'past-wrapper', 'current-state', 'desired-anchor', 'reset-reference-history', 'still-HOLD'])
def test_invalid_contract_or_history_rejects_before_install_or_any_effect(flow, fault):
    adapter, program, samples, windows = flow
    start, end = B, adapter.spec['max_ticks']
    if fault == 'phase': adapter.spec['phase'] = 'collect-target-fusion-source-posture-query-v3'
    elif fault == 'duck-program': program = SimpleNamespace(**program.__dict__)
    elif fault == 'sealed': adapter.trace_sealed = True
    elif fault == 'seal-marker': (adapter.output/'trace_seal_request').write_text('{}')
    elif fault == 'sealed-summary': (adapter.output/'init.trace.summary').write_text('{}')
    elif fault == 'short-budget': adapter.spec['max_ticks'] -= 25; end -= 25
    elif fault == 'start-grid': start += 1
    elif fault == 'end-grid': end += 1; adapter.spec['max_ticks'] = end
    elif fault == 'unregistered-B': adapter.spec['live_skill_start_step'] += 25
    elif fault == 'unregistered-E': adapter.spec['max_ticks'] += 25
    elif fault == 'dt': adapter.spec['physics_dt_s'] = .004
    elif fault == 'wbc': adapter.spec['WBC_changes'] = True
    elif fault == 'base': adapter.base_pose_hold[0] += .001
    elif fault == 'missing-history': samples = samples[-8:]
    elif fault == 'past-gap': windows[-1]['native_end_step'] -= 25
    elif fault == 'past-seq': windows[-1]['gripper']['submitted']['sequence'] = 0
    elif fault == 'past-solver': windows[-1]['solver_policies_with_exact_installed_target'] = []
    elif fault == 'past-grip': windows[-1]['gripper']['command_consumed'] = False
    elif fault == 'past-tcp': windows[-1]['actual_TCP'][0][3] += .001
    elif fault == 'past-wrapper': windows[-1]['gripper']['consumed_wrapper']['values']['sequence'] -= 1
    elif fault == 'current-state': samples[-1] = dict(samples[-1], ctrl=np.ones(20))
    elif fault == 'desired-anchor':
        anchor = program.decision_TCP_target.copy(); anchor[0, 3] += .001
        program = TrajectoryAdaptation(program.decoded_H40, program.query_TCP_source,
            program.T_source_from_target, anchor, program.initial_reference_TCP_target, BASE, 1.)
    elif fault == 'reset-reference-history': adapter.reference_history = []
    elif fault == 'still-HOLD': adapter.latest['mode'].data = 'HOLD'
    with pytest.raises(ValueError):
        segment.run_consumed_segment(adapter, program, start_step=start, end_step=end,
            past_samples=samples, past_windows=windows)
    assert not adapter.installs and not adapter.advances
    assert not list(adapter.output.glob('LIVE_SKILL_SEGMENT*'))


@pytest.mark.parametrize('fault', ['missing-ack', 'changed-suffix', 'stale-source'])
def test_first_install_fault_retained_without_gripper_effect_or_retry(flow, fault):
    adapter, *_ = flow
    adapter.install_fault = fault
    with pytest.raises(ValueError): run(flow)
    failure = json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert failure['status'] == 'FAILED_RETAINED_NO_RETRY'
    assert failure['install_attempts'] and failure['complete_suffix_atoms'] == 0
    assert len(adapter.installs) == 1 and not adapter.advances
    assert len(adapter.grip_packets) == 64 and adapter.cancel_calls == 1 and not adapter.trace_sealed
    with pytest.raises(ValueError, match='EXISTING_ATTEMPT'):
        run(flow)
    assert len(adapter.installs) == 1


@pytest.mark.parametrize('fault,expected', [
    ('wrong-target', 'SENDER_TO_SOLVER'), ('before-send', 'SENDER_TO_SOLVER'),
    ('ttl', 'WBC_AND_TTL'), ('hold-mode', 'WBC_AND_TTL'),
    ('gripper-sequence', 'GRIPPER_CONSUMPTION'), ('gripper-expiry', 'GRIPPER_CONSUMPTION'),
    ('wrong-pause', 'EXACT_ORIGINAL_NATIVE_PAUSE'),
    ('unsafe', 'FINITE_ACTUAL_SAFE'), ('native-abort', 'FINITE_ACTUAL_SAFE'),
    ('nonfinite', 'FINITE_ACTUAL_SAFE')])
def test_bad_actual_consumption_retains_effect_and_never_retries_or_seals(flow, fault, expected):
    adapter, *_ = flow
    adapter.effect_fault = fault
    with pytest.raises(ValueError, match=expected): run(flow)
    failure = json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert failure['status'] == 'FAILED_RETAINED_NO_RETRY'
    assert failure['complete_suffix_atoms'] == 0
    assert failure['failure_attempt']['actual_native_step'] == B+25
    assert failure['failure_attempt']['actual_state_vector']
    assert len(adapter.advances) == len(adapter.installs) == 1
    assert len(adapter.grip_packets) == 65 and len(adapter.reference_history) == 64
    assert adapter.cancel_calls == 1 and not adapter.trace_sealed


def test_renewal_ack_failure_preserves_seven_actual_atoms_and_does_not_resubmit(flow, monkeypatch):
    adapter, *_ = flow
    advance = adapter.advance_to
    def reject_next_install(step):
        result = advance(step)
        if step == B+175:
            adapter.install_fault = 'missing-ack'
        return result
    monkeypatch.setattr(adapter, 'advance_to', reject_next_install)
    with pytest.raises(ValueError, match='ACCEPT_AND_INSTALLED'): run(flow)
    failure = json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert failure['complete_suffix_atoms'] == 7
    assert len(adapter.installs) == 2 and len(adapter.advances) == 7
    assert len(adapter.grip_packets) == len(adapter.reference_history) == 71
    assert failure['failure_attempt']['native_start_step'] == B+175
    assert failure['install_attempts'][-1]['stage'] == 'RETURNED_ORIGINAL_INSTALL'


def test_subclass_cannot_smuggle_a_different_preview_or_forecaster(flow):
    adapter, program, samples, windows = flow
    class AlternateProgram(TrajectoryAdaptation):
        def preview(self, elapsed_atoms=0):
            raise AssertionError('must reject before polymorphic preview')
    subclass = AlternateProgram(program.decoded_H40, program.query_TCP_source,
        program.T_source_from_target, program.decision_TCP_target,
        program.initial_reference_TCP_target, program.base_pose_hold, 1.)
    with pytest.raises(ValueError, match='EXACT_TRAJECTORY_ADAPTATION'):
        segment.run_consumed_segment(adapter, subclass, start_step=B,
            end_step=adapter.spec['max_ticks'], past_samples=samples, past_windows=windows)
    assert not adapter.installs


def test_original_ttl_guard_fails_before_effect_and_saved_pre_effect_history_stays_causal(flow, monkeypatch):
    adapter, *_ = flow
    real = original.submit_gripper
    def delayed(*args):
        packet = real(*args)
        adapter.active_reference['source_mpc_s'] -= .4
        return packet
    monkeypatch.setattr(original, 'submit_gripper', delayed)
    # Adapter's native advance guard is represented by the original guard,
    # not a substitute physics loop.
    advance = adapter.advance_to
    def checked(step):
        from go2arx5_vla.world_model.target_reference import require_live_window
        require_live_window(adapter.active_reference['source_mpc_s'], adapter.step*.002, 25)
        return advance(step)
    monkeypatch.setattr(adapter, 'advance_to', checked)
    with pytest.raises(ValueError, match='REFERENCE_DEADLINE'): run(flow)
    failure = json.loads((adapter.output/'LIVE_SKILL_SEGMENT.private.json').read_text())
    assert failure['failure_attempt']['stage'] == 'SUBMITTED_BEFORE_EFFECT'
    assert failure['failure_attempt']['actual_history_before_atom']['actual_history_native_steps'][-1] == B
    assert not adapter.advances
