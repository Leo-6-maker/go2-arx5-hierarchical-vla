"""Offline contract/control-flow tests; mocked clocks are never native evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_reference_qualification_contract as c
from go2arx5_vla.world_model import fusion_reference_qualification_native as n
from go2arx5_vla.world_model.auto1_native import phase_limit
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from go2arx5_vla.world_model.storage import digest
from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
from scripts.wm02 import fusion_reference_qualification_check as cli


BASE = [0., 0., .3, 0., 0., 0., 1.]


def episode():
    return dict(schema=c.SCHEMA, phase=c.PHASE, purpose='REFERENCE_QUALIFICATION',
        experiment_role='NON_VLA_CALIBRATION', split='TRAIN', reference_rule=c.RULE,
        task_atoms=64, prehold_atoms=40, screen_atom=48, screen=c.SCREEN.copy(),
        desired_outcome_provenance='CALIBRATION_INTENT',
        rotation_program='HOLD_ANCHOR_ORIENTATION', gripper_program='HOLD_ORIGINAL_WORKING_OPEN',
        base_program='MEASURED_INIT_BASE_POSE_HOLD', physics_dt_s=.002,
        run='/fixture/new-run', scene_id='train-scene', parent='train-parent',
        candidate_id=c.FIRST_NATIVE_CANDIDATE, reference_anchor_mode=c.INIT_ANCHOR)


@pytest.fixture
def recipe(tmp_path):
    def save(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)
    model = tmp_path/'fixture-model'; model.write_bytes(b'not a loadable model: offline fixture')
    binary = tmp_path/'fixture-native'; binary.write_bytes(b'not executable')
    observer = tmp_path/'fixture-observer'; observer.write_bytes(b'not a shared object')
    spec = dict(phase=c.PHASE, profile='wm02-auto1-target', domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        physics=True, pilot1_binding='/fixture/fresh-original-owner-binding',
        source_root=str(tmp_path/'capsule'), source_files={}, WBC_changes=False, physics_dt_s=.002,
        registered_root='/fixture/new-run', campaign_root='/fixture/original-campaign', candidate_index=6,
        max_ticks=c.END_STEP, reference_binding_version='MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
        model_path=str(model), native_binary=str(binary), observer_library=str(observer))
    for name in c.SOURCE_MEMBERS:
        member = Path(spec['source_root'])/name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text('offline fixture member '+name)
        spec['source_files'][name] = digest(member)
    scene = dict(scene_id='train-scene', parent='train-parent', split='TRAIN',
        model_path=str(model), model_sha256=digest(model), static_check='PASSED_NO_INTEGRATION')
    spec['scene_allowlist'] = save('catalog.json', dict(scenes=[scene], native_binary_sha256=digest(binary)))
    spec['scene_allowlist_sha256'] = digest(spec['scene_allowlist'])
    spec['episode_manifest'] = save('episode.json', episode())
    spec['episode_manifest_sha256'] = digest(spec['episode_manifest'])
    previous = dict(status='NATIVE_STAGE_PASSED', campaign_uuid='original-campaign', candidate_index=6,
        shutdown_clean=True, physics_stops=[], actual_target_integrations=4103,
        owned_process_exits=[dict(exit_code=0)], trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS'),
        native_binary_sha256=digest(binary), observer_sha256=digest(observer), scene_sha256=digest(model),
        reference_qualification=dict(actual_native_evidence=True))
    spec['previous_stage'] = save('previous.json', previous)
    spec['previous_stage_sha256'] = digest(spec['previous_stage'])
    spec_path = save('spec.json', spec)
    return spec, previous, Path(spec_path)


def program(anchor=c.INIT_ANCHOR):
    initial = np.eye(4); initial[:3, 3] = [.2, -.1, .4]
    decision = initial.copy(); decision[:3, 3] += [.003, -.002, .001]
    return c.ReferenceProgram(c.FIRST_NATIVE_CANDIDATE, anchor, initial, decision, BASE)


def test_initial_reference_and_decision_desired_are_distinct_explicit_modes():
    fixed, reanchored = program(), program(c.DECISION_ANCHOR)
    np.testing.assert_allclose(fixed.reference_goal[:3, 3]-fixed.desired_goal[:3, 3], [-.003, .002, -.001])
    np.testing.assert_array_equal(reanchored.reference_goal, reanchored.desired_goal)
    assert not fixed.evidence()['reanchor_is_new_explicit_mode']
    assert reanchored.evidence()['reanchor_is_new_explicit_mode']
    assert not fixed.evidence()['native_execution_authorized']
    assert fixed.evidence()['experiment_role'] == 'NON_VLA_CALIBRATION'
    with pytest.raises(ValueError):
        c.ReferenceProgram(c.FIRST_NATIVE_CANDIDATE, 'implicit-repair', np.eye(4), np.eye(4), BASE)


def test_all_catalogue_candidates_are_offline_but_native_v1_is_only_one_z_move():
    for candidate in c.CANDIDATES:
        value = dict(episode(), candidate_id=candidate)
        c.episode_contract(value)
        if candidate == c.FIRST_NATIVE_CANDIDATE:
            c.episode_contract(value, native=True)
        else:
            with pytest.raises(ValueError, match='ONLY_FIRST_Z'):
                c.episode_contract(value, native=True)
    for changed in (dict(split='TEST'), dict(physics_dt_s=.004), dict(gripper=.99),
                    dict(clip=True), dict(native_execution_authorized=True), dict(experiment_role='VLA')):
        with pytest.raises(ValueError):
            c.episode_contract(dict(episode(), **changed))


def test_ramp_and_renewal_keep_frozen_orientation_gripper_and_planned_suffix():
    p = program(); first = p.preview(); renewed = p.preview(7)
    assert first.shape == (40, 10)
    np.testing.assert_allclose(first[:, 9], .042/.044)
    np.testing.assert_allclose(first[:8, 2]-p.initial_tcp[2, 3], np.arange(1, 9)*.0005/8)
    np.testing.assert_array_equal(first[7:], renewed[:33])
    np.testing.assert_array_equal(renewed, p.preview(14))
    np.testing.assert_array_equal(first[:, 3:9], np.tile(first[0, 3:9], (40, 1)))
    with pytest.raises(ValueError): p.preview(8)
    with pytest.raises(ValueError): p.initial_tcp[0, 3] = 99.


def validate(action, p=None, step=c.INIT_STEP, **changes):
    p0 = program()
    args = dict(initial_tcp=p0.initial_tcp, program=p, native_step=step,
                base_pose_hold=BASE, frozen_base_pose_hold=BASE)
    args.update(changes)
    return c.validate_install_preview(action, **args)


def test_every_h40_point_and_raw_mechanical_grip_are_checked_before_certificate():
    p = program()
    assert validate(c.hold_preview(p.initial_tcp))['points_checked'] == 40
    assert validate(p.preview(), p, c.DECISION_STEP)['original_certify_chunk_required']
    for column, value in ((0, .1), (8, .1), (9, 1.), (9, .99)):
        bad = p.preview(); bad[-1, column] = value
        with pytest.raises(ValueError): validate(bad, p, c.DECISION_STEP)
    with pytest.raises(ValueError): validate(p.preview(), p, c.DECISION_STEP, base_pose_hold=[.01]+BASE[1:])
    with pytest.raises(ValueError): validate(p.preview(), p, c.DECISION_STEP+1)
    with pytest.raises(ValueError): validate(p.preview(), None, c.DECISION_STEP)
    with pytest.raises(ValueError): validate(p.preview(), p, c.INIT_STEP)
    with pytest.raises(ValueError): validate(p.preview(), p, c.DECISION_STEP+8*25)
    other = c.ReferenceProgram('X_PLUS_1P25MM', c.INIT_ANCHOR, p.initial_tcp, p.decision_tcp, BASE)
    with pytest.raises(ValueError, match='INSTALL_ONLY_FIRST_Z'):
        validate(other.preview(), other, c.DECISION_STEP)


def test_original_clock_cannot_be_retimed_to_make_eighth_atom_live():
    c.check_atomic_window(7.406, 7.406)
    for native in (7.756, 7.806):
        with pytest.raises(ValueError): c.check_atomic_window(7.406, native)
    with pytest.raises(ValueError): c.check_atomic_window(7.407, 7.406)


def test_recipe_source_scene_and_predecessor_pins(recipe):
    spec, previous, path = recipe
    assert c.episode_limit(spec, previous) == 4103
    assert phase_limit(c.PHASE, previous, spec) == 4103
    assert cli.check(path)['native_execution_authorized'] is False
    for changed in (dict(fusion_binding='/old-fixed-xy'), dict(parallel_batch_spec='/batch'),
                    dict(standby_shutdown_diagnostic=True), dict(reference_readiness_probe=True),
                    dict(diagnostic_debugger='/gdb'), dict(WBC_changes=True), dict(physics_dt_s=.004)):
        with pytest.raises(ValueError): c.episode_config(dict(spec, **changed))
    for changed in (dict(status='FAILED'), dict(shutdown_clean=False), dict(candidate_index=5),
                    dict(campaign_uuid='other'), dict(trace_audit={}), dict(owned_process_exits=[]),
                    dict(owned_process_exits=[dict(exit_code=139)]), dict(scene_sha256='other'),
                    dict(reference_qualification=None)):
        with pytest.raises(ValueError): c.episode_limit(spec, dict(previous, **changed))
    member = Path(spec['source_root'])/c.SOURCE_MEMBERS[0]
    member.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SOURCE_BYTES'): c.episode_config(spec)


def test_static_scene_check_still_required_even_after_rehash(recipe):
    spec, _, _ = recipe
    path = Path(spec['scene_allowlist']); catalog = json.loads(path.read_text())
    catalog['scenes'][0]['static_check'] = 'NOT_RUN'; path.write_text(json.dumps(catalog))
    spec['scene_allowlist_sha256'] = digest(path)
    with pytest.raises(ValueError, match='UNCHANGED_SCENE'): c.episode_config(spec)


def reference(source=7.406):
    wire = '{"fixture":"unchanged receiver wire, not an actual command"}'
    ref = hashlib.sha256(wire.encode()).hexdigest()
    return dict(wire=wire, wire_sha256=ref, source_mpc_s=source, receiver_events=[
        dict(kind='accept', ref_id=ref, raw=wire, source_mpc_s=source, receiver_mpc_s=source,
             deadline=source+.4),
        dict(kind='installed', ref_id=ref, bound_start=source, observation_time_s=source)])


def test_real_receiver_ack_binding_has_no_fabricated_eef_only_shape():
    ref = reference()
    assert n.require_original_install_ack(ref)['installed'] == 1
    for field, value, row in (('raw', 'different', 0), ('deadline', 999., 0),
                              ('receiver_mpc_s', 7.5, 0), ('bound_start', 7., 1)):
        bad = copy.deepcopy(ref); bad['receiver_events'][row][field] = value
        with pytest.raises(ValueError): n.require_original_install_ack(bad)
    for kind in ('accept', 'installed'):
        bad = copy.deepcopy(ref); bad['receiver_events'] = [r for r in bad['receiver_events'] if r['kind'] != kind]
        with pytest.raises(ValueError): n.require_original_install_ack(bad)
    bad = copy.deepcopy(ref); bad['wire'] += ' '
    with pytest.raises(ValueError): n.require_original_install_ack(bad)


def test_solver_join_requires_exact_full_body_target_after_actual_send():
    from go2arx5_vla.world_model.target_reference import trajectory_identity
    times = [7.406, 7.456]
    poses = [[float(i) for i in range(14)]]*2  # Original EEF+BASE target, not an EEF-only ABI.
    inputs = [[0.]]*2
    installed = dict(kind='installed', ref_id='ref', times=times, poses=poses, inputs=inputs,
                     monotonic_s=100.)
    message = dict(plan_target_trajectories=dict(time_trajectory=times,
        state_trajectory=[dict(value=p) for p in poses], input_trajectory=[dict(value=p) for p in inputs]))
    adapter = SimpleNamespace(active_reference=dict(wire_sha256='ref', receiver_events=[installed]),
        reference_events=lambda: [installed], ros_evidence=[
            dict(topic_key='policy', message=message, received_monotonic_s=101.),
            dict(topic_key='arm_request', message={}, received_monotonic_s=101.)])
    row = n.consumed_window(adapter, 0, dict(step=3703), dict(step=3728), {})
    assert row['actual_ticks'] == 25 and row['arm_request_count'] == 1
    assert row['target_trajectory_sha256'] == trajectory_identity(times, poses, inputs)
    assert len(row['solver_policies_with_exact_installed_target']) == 1
    message['plan_target_trajectories']['state_trajectory'][0]['value'] = poses[0][:7]
    assert not n.consumed_window(adapter, 0, dict(step=3703), dict(step=3728), {})[
        'solver_policies_with_exact_installed_target']


def test_import_help_and_default_check_never_import_native_or_torch(recipe):
    _, _, path = recipe
    code = ('import sys;from scripts.wm02.fusion_reference_qualification_check import main;'
            'assert main(["--spec",sys.argv[1]])==0;'
            'assert not {"mujoco","rclpy","torch"}&sys.modules.keys()')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(path)], check=True,
                            capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert 'ACTUAL_ADMISSION_STILL_REQUIRED' in result.stdout
    subprocess.run([sys.executable, '-B', '-m', 'scripts.wm02.fusion_reference_qualification_check', '--help'],
                   check=True, capture_output=True)


def test_execute_delegates_only_to_original_worker_after_recipe_check(recipe, monkeypatch):
    _, _, path = recipe; calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
                        SimpleNamespace(main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]


def test_cli_changed_predecessor_bytes_block_execute_before_original_worker(recipe, monkeypatch):
    spec, previous, path = recipe; calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
                        SimpleNamespace(main=lambda argv: calls.append(argv) or 0))
    previous['status'] = 'FAILED_PRESERVED'
    Path(spec['previous_stage']).write_text(json.dumps(previous))
    with pytest.raises(ValueError, match='PREVIOUS_NATIVE_BYTES_CHANGED'):
        cli.main(['--spec', str(path), '--execute'])
    assert calls == []


def test_actual_adapter_hook_checks_full_h40_before_original_install_chain(tmp_path):
    p = program(); adapter = TargetSequentialAdapter(dict(phase=c.PHASE), tmp_path)
    adapter.qualification_initial_tcp = p.initial_tcp
    adapter.qualification_initial_base_pose_hold = BASE
    adapter.qualification_program = None; adapter.base_pose_hold = BASE
    adapter.observe = lambda: dict(step=c.INIT_STEP)
    events = []; adapter.command_event = events.append
    class OriginalInstallChainReached(Exception): pass
    def stop_before_any_native_import(): raise OriginalInstallChainReached
    adapter.wait_source = stop_before_any_native_import
    action = c.hold_preview(p.initial_tcp)
    bad = action.copy(); bad[-1, 2] += .0001
    with pytest.raises(ValueError, match='ENTIRE_H40'): adapter.install_reference(bad)
    assert not events
    with pytest.raises(OriginalInstallChainReached): adapter.install_reference(action)
    assert events[0]['points_checked'] == 40
    assert events[0]['original_certify_chunk_required']
    old = TargetSequentialAdapter(dict(phase='g2-standing-eef-y-v2'), tmp_path)
    old.wait_source = stop_before_any_native_import
    with pytest.raises(OriginalInstallChainReached): old.install_reference(action)


def test_actual_adapter_dispatch_only_delegates_new_phase(tmp_path, monkeypatch):
    adapter = TargetSequentialAdapter(dict(phase=c.PHASE), tmp_path)
    monkeypatch.setattr(n, 'run', lambda actual: ('new-phase', actual))
    assert adapter.dynamic_diagnostic() == ('new-phase', adapter)


class MockClockAdapter:
    """Control-flow fixture, zero simulator/controller/model construction."""
    def __init__(self, output):
        self.spec = {}; self.output = output; self.step = c.INIT_STEP
        self.base_pose_hold = BASE.copy(); self.reference_history = []; self.ros_evidence = []
        self.latest = dict(mode=SimpleNamespace(data='WBC')); self.installs = []
        self.events = []; self.reference_qualification = None
    def prepare_dynamic(self): pass
    def observe(self): return dict(step=self.step, time_s=self.step*.002)
    def measured_task_state(self):
        return dict(native_step=self.step, time_s=self.step*.002, tcp=np.eye(4),
                    qpos=np.array([0., 0., .3, 1., 0., 0., 0.]), safety_valid=True, native_abort=False)
    def install_reference(self, action):
        n.validate_adapter_install(self, action)
        self.installs.append((self.step, action.copy()))
        # Match the existing ROS publisher's integer-nanosecond source clock.
        self.active_reference = dict(reference(round(self.step*.002*1e9)/1e9), action_EEF10=action.copy())
        return self.active_reference
    def command_event(self, value): self.events.append(value)
    def reference_events(self): return []
    def advance_to(self, step): self.step = step; return self.observe()
    def spin(self, duration): pass
    def capture_loaded_identity(self): pass
    def cancel_reference(self): return dict(kind='mock-cancel-not-native-evidence')
    def seal_trace(self, step): self.sealed = step


def setup_mock_flow(monkeypatch):
    monkeypatch.setattr(n, 'episode_config', lambda spec: episode())
    def mock_window(adapter, event_start, before, after, clock):
        return dict(native_start_step=before['step'], native_end_step=after['step'],
                    actual_ticks=25, solver_policies_with_exact_installed_target=[dict(mock_only=True)])
    monkeypatch.setattr(n, 'consumed_window', mock_window)


def test_mocked_flow_has_exact_original_boundaries_renewals_and_no_reanchor(tmp_path, monkeypatch):
    setup_mock_flow(monkeypatch); adapter = MockClockAdapter(tmp_path)
    report = n.run(adapter)
    assert [s for s, _ in adapter.installs] == [2503, 2678, 2853, 3028, 3203, 3378, 3503, 3678,
                                               3703, 3878, 4053]
    assert adapter.sealed == 4103
    assert len(adapter.reference_history) == 64
    assert report['complete_task_atoms'] == report['solver_target_confirmed_atoms'] == 64
    assert report['native_stage_RESULT_json_required'] and report['safety_STOP_qualification'] == 'NOT_TESTED'
    assert report['experiment_role'] == 'NON_VLA_CALIBRATION' and report['VLA_inferences'] == 0
    assert report['status'] != 'NATIVE_STAGE_PASSED'
    np.testing.assert_array_equal(adapter.installs[-1][1], adapter.installs[-2][1])
    assert report['actual_minus_desired_xyz_m'][2] == -.0005  # No success invented for a static mock.


def test_mocked_prefix_failure_preserves_report_and_never_installs_candidate(tmp_path, monkeypatch):
    setup_mock_flow(monkeypatch)
    from go2arx5_vla.world_model import target_stability
    monkeypatch.setattr(target_stability, 'screen', lambda *args: dict(qualified=False))
    adapter = MockClockAdapter(tmp_path)
    with pytest.raises(ValueError, match='STEADY_PREFIX'): n.run(adapter)
    assert adapter.step == 3703 and max(s for s, _ in adapter.installs) == 3678
    assert (tmp_path/'REFERENCE_QUALIFICATION_PREFIX.private.json').exists()
    assert adapter.reference_qualification is None


def test_mocked_missing_ack_stops_before_first_atomic_advance(tmp_path, monkeypatch):
    setup_mock_flow(monkeypatch); adapter = MockClockAdapter(tmp_path)
    original = adapter.install_reference
    def incomplete(action):
        ref = original(action); ref['receiver_events'] = ref['receiver_events'][:1]; return ref
    adapter.install_reference = incomplete
    with pytest.raises(ValueError, match='ACCEPT_AND_INSTALLED'): n.run(adapter)
    assert adapter.step == 2503 and adapter.reference_history == []


def test_mocked_unconfirmed_solver_retains_failure_and_never_sets_qualification(tmp_path, monkeypatch):
    setup_mock_flow(monkeypatch); adapter = MockClockAdapter(tmp_path)
    original = n.consumed_window
    def unconfirmed(*args):
        row = original(*args); row['solver_policies_with_exact_installed_target'] = []; return row
    monkeypatch.setattr(n, 'consumed_window', unconfirmed)
    with pytest.raises(ValueError, match='CONSUMPTION_REQUIRED'): n.run(adapter)
    assert adapter.reference_qualification is None
    report = json.loads((tmp_path/'REFERENCE_PROGRAM_QUALIFICATION.private.json').read_text())
    assert report['solver_target_confirmed_atoms'] == 0 and report['complete_task_atoms'] == 64


def test_mocked_outer_commission_accepts_new_report_shape_without_old_policy_fields(recipe, monkeypatch):
    """Schema/lifecycle test only: packet and trace audits are explicitly mocked."""
    setup_mock_flow(monkeypatch)
    from scripts.wm02 import commission_target
    from go2arx5_vla.world_model import init_trace_reader, native_replay, target_sequential
    spec, _, path = recipe
    spec.update(source_sha='offline-fixture-source', candidate_source_sha='offline-fixture-native-source')
    output = path.parent/'mock-native-01'; output.mkdir()
    class MockOuterAdapter(MockClockAdapter):
        def __init__(self, actual_spec, actual_output):
            super().__init__(actual_output)
            self.spec = actual_spec; self.node = None; self.exit_records = []; self.trace_sealed = False
        def start(self):
            (self.output/'init.trace').write_bytes(b'offline mocked trace, not native evidence')
            (self.output/'state.bin.journal').write_bytes(b'offline mocked journal')
        def dynamic_phase(self): return True
        def reference_phase(self): return True
        def dynamic_diagnostic(self): return n.run(self)
        def close(self): self.exit_records = [dict(exit_code=0, mock_only=True)]
    monkeypatch.setattr(target_sequential, 'TargetSequentialAdapter', MockOuterAdapter)
    monkeypatch.setattr(init_trace_reader, 'audit_trace', lambda *a, **k:
                        dict(status='RECORDED_TRACE_CHECKS_PASS', mock_only=True))
    monkeypatch.setattr(native_replay, 'packets', lambda path: [None]*4103)
    assert commission_target.run(spec, output) == 0
    result = json.loads((output/'RESULT.json').read_text())
    report = result['reference_qualification']
    assert report['complete_task_atoms'] == report['solver_target_confirmed_atoms'] == 64
    assert result['shutdown_clean'] and result['actual_target_integrations'] == 4103
    assert 'steady_prefix' not in report and 'reference_policy' not in report
    assert report['experiment_role'] == 'NON_VLA_CALIBRATION'


def test_existing_phase_dispatch_is_unchanged():
    assert phase_limit('cold-1', None) == 50
    with pytest.raises(PermissionError): phase_limit('g2-reference-qualification', None)
    with pytest.raises(PermissionError): phase_limit('collect-target-unknown', None)
