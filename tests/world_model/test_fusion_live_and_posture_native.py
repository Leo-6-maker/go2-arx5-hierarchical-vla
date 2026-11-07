"""Synthetic native seams only; no model query, physics, or qualification claim."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.trajectory.so3 import geodesic_angle, rotation_matrix_to_rot6d_rows
from go2arx5_vla.world_model import auto1_native
from go2arx5_vla.world_model import fusion_live_proposal_qualification_contract as live
from go2arx5_vla.world_model import fusion_live_proposal_qualification_native as live_native
from go2arx5_vla.world_model import fusion_short_skill as skill
from go2arx5_vla.world_model import fusion_source_posture_qualification_contract as posture
from go2arx5_vla.world_model import fusion_source_posture_qualification_native as posture_native
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as old
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as native
from go2arx5_vla.world_model import target_sequential
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from scripts.pi0292.eval_contract import matrix10
from scripts.wm02 import fusion_live_proposal_qualification_check as live_cli
from scripts.wm02 import fusion_source_posture_qualification as posture_cli
from test_fusion_trajectory_qualification import BASE, FakeAdapter, recipe, setup_flow


PHASES = (live.PHASE, posture.PHASE)


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def row(pose, grip=1.):
    return np.r_[pose[:3, 3], rotation_matrix_to_rot6d_rows(pose[:3, :3]), grip]


class MeasuredAdapter(FakeAdapter):
    """Commands do not move this fixture; actual state is independently supplied."""
    def __init__(self, spec, output, *, after_decision=None):
        super().__init__(spec, output)
        self.after_decision = np.eye(4) if after_decision is None else after_decision
        self.prepare_calls = 0

    def prepare_dynamic(self):
        from go2arx5_vla.world_model.init_trace_reader import RECORD
        self.prepare_calls += 1
        capacity = self.spec.get('trajectory_trace_capacity', 131072)
        (self.output/'init.trace').write_bytes(struct.pack('<8sII', b'WM02IT01', RECORD.size, capacity))

    def measured_task_state(self):
        result = super().measured_task_state()
        if self.step > old.DECISION_STEP:
            result['tcp'] = self.after_decision.copy()
        # This object is deliberately outside the short-skill precontact region.
        result['object_pose'] = np.eye(4)
        result['object_pose'][:3, 3] = [.6, 0., 0.]
        result['robot_object_contact'] = False
        return result

    def query(self, *args, **kwargs):
        raise AssertionError('A frozen replay must not query a model')

    def decode(self, *args, **kwargs):
        raise AssertionError('Already physical H40 must not be decoded again')


def bind_new_flow(recipe, tmp_path, monkeypatch, kind, *, scale=None, actual=None):
    spec, _, _, _, _ = recipe
    spec = copy.deepcopy(spec)
    contract = live if kind == 'live' else posture
    scale = (.25 if kind == 'live' else 1.) if scale is None else scale
    frame = np.eye(4)
    frame[:3, :3] = rz(.3)
    frame[:3, 3] = [.1, -.08, .04]
    goal_target = np.eye(4)
    goal_target[:3, :3] = rz(.08)
    goal_target[:3, 3] = [.03, -.02, .01]
    if kind == 'live':
        frame = np.eye(4)
        raw = np.repeat(row(np.eye(4))[None], 40, axis=0)
        raw[:, 0] = np.linspace(.001, .012, 40)
        inputs = dict(raw=raw, query_tcp=np.eye(4), frame_map=frame,
            archive=dict(external_pins=copy.deepcopy(live.PINS)),
            task=skill.task_document('a'*64), task_sha256=live.PINS['task_sha256'],
            provenance=dict(synthetic_test_fixture=True, historical_read=True,
                            current_native_pause_checked=False, native_execution_authorized=False))
    else:
        goal = frame @ goal_target
        goal_row = row(goal)
        inputs = dict(raw=np.repeat(goal_row[None], 40, axis=0), query_tcp=frame.copy(), frame_map=frame,
            raw_generated_H40=True, raw_is_generated_calibration_reference=True,
            source_observation_state_goal=goal_row, source_goal_TCP_source=goal,
            provenance=dict(synthetic_test_fixture=True, raw_generated_H40=True,
                            source_observed_pose_is_expert_action_label=False))
    probe = TrajectoryAdaptation(inputs['raw'], inputs['query_tcp'], inputs['frame_map'],
        np.eye(4), np.eye(4), BASE, scale, time_stretch_factor=2)
    episode = dict(contract._fixed_episode(), run=str(tmp_path), geometry_scale=scale,
                   planned_end_atom=probe.end_atom)
    spec.update(phase=contract.PHASE, max_ticks=contract.endpoint(episode), trajectory_trace_capacity=524288)
    monkeypatch.setattr(contract, 'episode_config', lambda value: (episode, inputs))
    adapter = MeasuredAdapter(spec, tmp_path, after_decision=actual)
    setup_flow(monkeypatch, adapter)
    return adapter, episode, inputs


def test_factored_default_preserves_original_run_flow(recipe, tmp_path, monkeypatch):
    spec, _, _, episode, inputs = recipe
    results, installs = [], []
    for name, direct in (('original-entry', False), ('factored-defaults', True)):
        output = tmp_path/name; output.mkdir()
        adapter = MeasuredAdapter(spec, output)
        with monkeypatch.context() as patch:
            setup_flow(patch, adapter)
            result = (native.run_program(adapter, episode, inputs) if direct else native.run(adapter))
        results.append(result); installs.append(adapter.installs)
        assert adapter.prepare_calls == 1 and adapter.sealed == old.endpoint(episode)
        assert result['all_40_knots_consumed'] and result['VLA_commands'] == 0
        assert 'short_skill_score' not in result and 'source_posture_diagnostic' not in result
    assert results[0] == results[1]
    assert len(installs[0]) == len(installs[1])
    for (step_a, action_a), (step_b, action_b) in zip(*installs):
        assert step_a == step_b
        np.testing.assert_array_equal(action_a, action_b)


def test_live_wrapper_scores_actual_samples_without_query_or_second_decode(recipe, tmp_path, monkeypatch):
    adapter, episode, inputs = bind_new_flow(recipe, tmp_path, monkeypatch, 'live')
    raw_bytes = inputs['raw'].tobytes()
    scorer_calls = []
    original_score = skill.score
    def record_score(samples, contract):
        scorer_calls.append((samples, contract))
        return original_score(samples, contract)
    monkeypatch.setattr(skill, 'score', record_score)
    report = live_native.run(adapter)
    assert len(scorer_calls) == 1
    samples, contract = scorer_calls[0]
    assert contract == skill.score_contract()
    assert [s['native_step'] for s in samples] == list(range(3703, live.endpoint(episode)+1, 25))
    assert all(s['object_pose'][0, 3] == .6 for s in samples)
    assert report['short_skill_score']['metrics']['signed_progress_m'] == 0.
    assert not report['short_skill_score']['approach_step_success']
    assert adapter.trajectory_program.desired_path[-1, 0] >= .002
    assert report['experiment_role'] == 'VLA_PLUS_EXPLICIT_TRAJECTORY_ADAPTATION'
    assert report['VLA_reference_replays'] == report['actual_VLA_commands'] == report['VLA_commands'] == 1
    assert report['VLA_inferences'] == report['Processor_decodes'] == report['original_Processor_decodes'] == 0
    assert report['online_query'] is report['online_VLA'] is report['full_task_success'] is False
    assert report['all_40_knots_consumed'] and report['actual_trace_binding']['actual_trace_capacity'] == 524288
    assert inputs['raw'].tobytes() == raw_bytes
    decision = json.loads((tmp_path/'TRAJECTORY_QUALIFICATION_DECISION.private.json').read_text())
    np.testing.assert_array_equal(decision['raw_H40'], inputs['raw'])
    assert decision['original_Processor_decodes'] == decision['VLA_inferences'] == 0
    saved = json.loads((tmp_path/'TRAJECTORY_QUALIFICATION.private.json').read_text())
    assert saved['short_skill_score'] == report['short_skill_score']


def test_live_score_contract_mismatch_rejects_before_preparation(recipe, tmp_path, monkeypatch):
    adapter, _, inputs = bind_new_flow(recipe, tmp_path, monkeypatch, 'live')
    inputs['task']['score_contract_sha256'] = '0'*64
    with pytest.raises(ValueError, match='ORIGINAL_SHORT_SKILL_SCORE_REQUIRED'):
        live_native.run(adapter)
    assert adapter.prepare_calls == 0 and not adapter.installs and not adapter.grip_packets


@pytest.mark.parametrize('scale', [.03, 1.])
def test_posture_error_uses_measured_endpoint_in_source_frame(recipe, tmp_path, monkeypatch, scale):
    measured = np.eye(4); measured[:3, :3] = rz(.012); measured[:3, 3] = [.001, -.003, .002]
    adapter, episode, inputs = bind_new_flow(recipe, tmp_path, monkeypatch, 'posture', scale=scale, actual=measured)
    def no_task_scorer(*args, **kwargs):
        raise AssertionError('Calibration must not be relabeled as a scored VLA task')
    monkeypatch.setattr(skill, 'score', no_task_scorer)
    report = posture_native.run(adapter)
    diagnostic = report['source_posture_diagnostic']
    actual_source = inputs['frame_map'] @ measured
    goal = inputs['source_goal_TCP_source']
    difference = actual_source[:3, 3]-goal[:3, 3]
    np.testing.assert_array_equal(report['actual_endpoint_TCP'], measured)
    np.testing.assert_allclose(diagnostic['actual_endpoint_TCP_source'], actual_source, atol=1e-15)
    np.testing.assert_allclose(diagnostic['actual_minus_source_goal_xyz_m'], difference, atol=1e-15)
    assert diagnostic['actual_source_goal_position_error_m'] == pytest.approx(np.linalg.norm(difference))
    assert diagnostic['actual_source_goal_rotation_error_rad'] == pytest.approx(
        float(geodesic_angle(goal[:3, :3], actual_source[:3, :3])))
    if scale == 1.:
        desired_source = inputs['frame_map'] @ matrix10(adapter.trajectory_program.desired_path[-1])
        np.testing.assert_allclose(desired_source, goal, atol=1e-12)
        assert diagnostic['actual_source_goal_position_error_m'] > .01
    assert not diagnostic['full_scale_source_pose_close']
    assert diagnostic['residual_basis'] == 'ACTUAL_MEASURED_ENDPOINT_NOT_REFERENCE_OR_MODEL_PREDICTION'
    for key in ('camera_equivalence', 'full_policy_input_distribution_qualified', 'task_success',
                'native_execution_authorized', 'source_observation_used_as_expert_action_label'):
        assert diagnostic[key] is False
    assert report['experiment_role'] == 'NON_VLA_CALIBRATION'
    assert report['desired_outcome_provenance'] == 'CALIBRATION_INTENT'
    assert report['calibration_goal_provenance'] == 'KNOWN_SOURCE_OBSERVATION_POSE_USED_AS_CALIBRATION_REFERENCE'
    assert not report['generated_reference_is_GR00T_action'] and not report['source_pose_is_expert_action_label']
    assert report['actual_VLA_commands'] == report['VLA_commands'] == report['VLA_reference_replays'] == 0
    assert report['VLA_inferences'] == report['Processor_decodes'] == report['original_Processor_decodes'] == 0
    assert 'short_skill_score' not in report and not report['full_task_success']
    saved = json.loads((tmp_path/'TRAJECTORY_QUALIFICATION.private.json').read_text())
    assert saved['source_posture_diagnostic'] == diagnostic
    decision = json.loads((tmp_path/'TRAJECTORY_QUALIFICATION_DECISION.private.json').read_text())
    assert decision['desired_outcome_provenance'] == 'CALIBRATION_INTENT'
    assert decision['calibration_goal_provenance'] == report['calibration_goal_provenance']
    assert report['all_40_knots_consumed'] and adapter.sealed == posture.endpoint(episode)


@pytest.mark.parametrize('scale', [.03, 1.])
def test_measured_posture_closeness_never_promotes_partial_scale_or_task_success(recipe, tmp_path, monkeypatch, scale):
    measured = np.eye(4)
    measured[:3, :3] = rz(.08)
    measured[:3, 3] = [.03, -.02, .01]
    adapter, _, _ = bind_new_flow(recipe, tmp_path, monkeypatch, 'posture', scale=scale, actual=measured)
    report = posture_native.run(adapter)
    diagnostic = report['source_posture_diagnostic']
    assert diagnostic['actual_source_goal_position_error_m'] == pytest.approx(0., abs=1e-15)
    assert diagnostic['actual_source_goal_rotation_error_rad'] == pytest.approx(0., abs=1e-7)
    assert diagnostic['full_scale_source_pose_close'] is (scale == 1.)
    assert not diagnostic['task_success'] and not diagnostic['camera_equivalence']
    assert not diagnostic['native_execution_authorized'] and not report['target_task_qualification']
    assert not report['full_task_success'] and report['actual_VLA_commands'] == 0


@pytest.mark.parametrize('change', ['generated-missing', 'generated-false', 'goal-shape', 'goal-nan'])
def test_posture_wrapper_rejects_unqualified_goal_before_preparation(recipe, tmp_path, monkeypatch, change):
    adapter, _, inputs = bind_new_flow(recipe, tmp_path, monkeypatch, 'posture')
    if change == 'generated-missing': inputs.pop('raw_is_generated_calibration_reference')
    elif change == 'generated-false': inputs['raw_is_generated_calibration_reference'] = False
    elif change == 'goal-shape': inputs['source_goal_TCP_source'] = np.zeros((3, 3))
    else: inputs['source_goal_TCP_source'][0, 0] = np.nan
    with pytest.raises(ValueError, match='SOURCE_POSTURE_'):
        posture_native.run(adapter)
    assert adapter.prepare_calls == 0 and not adapter.installs and not adapter.grip_packets


@pytest.mark.parametrize('phase', PHASES)
def test_new_native_phase_limits_quota_and_dispatch_are_exact(tmp_path, monkeypatch, phase):
    contract, wrapper = ((live, live_native) if phase == live.PHASE else (posture, posture_native))
    spec, previous = dict(phase=phase), dict(synthetic_previous=True)
    calls = []
    monkeypatch.setattr(contract, 'episode_limit', lambda s, p: calls.append((s, p)) or 7778)
    assert auto1_native.phase_limit(phase, previous, spec) == 7778
    assert calls == [(spec, previous)]
    adapter = target_sequential.TargetSequentialAdapter(spec, tmp_path)
    assert adapter.reference_phase() and adapter.dynamic_phase()
    assert target_sequential.native_task_bucket(phase, True) == '64'
    sentinel = object()
    monkeypatch.setattr(wrapper, 'run', lambda value: sentinel if value is adapter else None)
    assert adapter.dynamic_diagnostic() is sentinel
    assert auto1_native.phase_limit('cold-1', None) == 50
    assert target_sequential.native_task_bucket('cold-1', False) == '0'
    assert target_sequential.native_task_bucket('g2-standing-eef-y-v2', True) == '32'
    with pytest.raises(PermissionError): auto1_native.phase_limit('collect-target-unknown', None)


@pytest.mark.parametrize('phase', PHASES)
def test_new_phases_load_trace_first_only_for_actual_simulator(tmp_path, phase):
    spec = dict(phase=phase, trajectory_trace_capacity_build='/fixture/build03',
        xvfb_run='/fixture/xvfb', native_binary='/fixture/native', trace_library='/fixture/trace',
        outer_library='/fixture/outer', observer_library='/fixture/observer')
    original = dict(LD_PRELOAD='must-not-reach-xvfb', WM02_INIT_TRACE='must-not-reach-xvfb', KEPT='yes')
    argv, environment = target_sequential.native_display_command(spec, tmp_path, original)
    assert 'LD_PRELOAD=/fixture/trace:/fixture/outer:/fixture/observer' in argv
    assert argv[-3:] == ['/fixture/native', 'go2_x5', 'scene']
    assert 'LD_PRELOAD' not in environment and 'WM02_INIT_TRACE' not in environment
    assert environment['KEPT'] == 'yes' and original['LD_PRELOAD'] == 'must-not-reach-xvfb'


@pytest.mark.parametrize('phase', PHASES)
def test_new_phases_require_actual_524_trace_header(recipe, tmp_path, phase):
    spec = dict(recipe[0], phase=phase, trajectory_trace_capacity=524288)
    adapter = MeasuredAdapter(spec, tmp_path); adapter.prepare_dynamic()
    assert native.actual_trace_binding(adapter)['actual_trace_capacity'] == 524288
    from go2arx5_vla.world_model.init_trace_reader import RECORD
    (tmp_path/'init.trace').write_bytes(struct.pack('<8sII', b'WM02IT01', RECORD.size, 262144))
    with pytest.raises(ValueError, match='ACTUAL_NEW_TRACE_HEADER_CAPACITY_524288'):
        native.actual_trace_binding(adapter)


@pytest.mark.parametrize('phase', [*PHASES, 'g2-standing-eef-y-v2'])
def test_install_defers_gripper_only_for_new_program_atom_writer(tmp_path, monkeypatch, phase):
    from scripts.pi0294 import task_state
    writes = []
    monkeypatch.setattr(task_state, 'write_gripper', lambda *args: writes.append(args))
    monkeypatch.setattr(native, 'validate_adapter_install', lambda *args: dict(synthetic_only=True))
    adapter = target_sequential.TargetSequentialAdapter(dict(phase=phase), tmp_path)
    adapter.reference_sequence = -1
    adapter.latest = dict(ee=SimpleNamespace(time=5.006))
    adapter.wait_source = lambda: None
    adapter.observe = lambda: dict(step=2503, time_s=5.006)
    adapter.command_event = lambda event: None
    adapter.refs = object(); events = []
    def publish(topic, wire):
        identity = hashlib.sha256(wire.encode()).hexdigest()
        events.extend(dict(kind=kind, ref_id=identity) for kind in ('accept', 'installed'))
    adapter.publish = publish
    adapter.reference_events = lambda: events
    adapter.spin = lambda seconds, predicate: predicate()
    adapter.install_reference(old.hold_preview(np.eye(4)))
    assert len(writes) == (0 if phase in PHASES else 1)


@pytest.mark.parametrize('contract,cli', [(live, live_cli), (posture, posture_cli)])
def test_new_check_cli_delegates_execution_to_original_worker(tmp_path, monkeypatch, capsys, contract, cli):
    previous = tmp_path/'previous.json'; previous.write_text('{"synthetic_previous":true}')
    path = tmp_path/'spec.json'; path.write_text(json.dumps(dict(phase=contract.PHASE, previous_stage=str(previous))))
    checks, starts = [], []
    monkeypatch.setattr(contract, 'episode_limit', lambda s, p: checks.append((s, p)) or 7778)
    value = cli.check(path)
    assert value['new_integrations'] == value['VLA_queries'] == value['VLA_action_installations'] == 0
    assert not value['native_execution_authorized'] and not value['task_success_claim']
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
        SimpleNamespace(main=lambda argv: starts.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--check-only']) == 0
    assert not starts
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert starts == [['--spec', str(path)]] and len(checks) == 3
    assert json.loads(capsys.readouterr().out)['native_execution_authorized'] is False
