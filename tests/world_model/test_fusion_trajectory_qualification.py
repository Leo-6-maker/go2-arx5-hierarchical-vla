"""Offline lineage and lifecycle tests; fake clocks are not native evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as c
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as n
from go2arx5_vla.world_model.auto1_native import phase_limit
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.wm02 import fusion_trajectory_qualification_check as cli
from test_infer_fusion_observation import capture_fixture, put
from test_fusion_reference_qualification_contract import reference

BASE = [0., 0., .3, 0., 0., 0., 1.]


def episode(scale=.03, end_atom=40):
    return dict(schema=c.SCHEMA, phase=c.PHASE, purpose='TRAJECTORY_GEOMETRY_QUALIFICATION',
        experiment_role='NON_VLA_CALIBRATION', split='TRAIN', reference_rule=c.RECIPE,
        reference_anchor_mode='INIT_TCP_2503_REFERENCE_V1', prehold_atoms=40,
        screen_atom=48, screen=c.SCREEN, tail_hold_atoms=8,
        desired_outcome_provenance='CALIBRATION_INTENT', physics_dt_s=.002,
        base_program='MEASURED_INIT_BASE_POSE_HOLD', query_pins={k:v for k,v in c.PINS.items()
                                                            if k != 'first_predecessor'},
        run='/fixture/new-trajectory', scene_id='TRAIN_SCENE', parent='TRAIN_PARENT',
        geometry_scale=scale, planned_end_atom=end_atom)


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    source = tmp_path/'source'
    project = Path(__file__).resolve().parents[2]
    files = {}
    for name in c.SOURCE_MEMBERS:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((project/name).read_bytes()); files[name] = digest(p)
    captured = tmp_path/'capture'; captured.mkdir()
    query = capture_fixture(captured)
    query.update(source_sha=c.QUERY_SOURCE, checkpoint_manifest_sha256=c.PINS['checkpoint'],
        seed=366102, source_files=files)
    root = tmp_path/'query'; (root/'fit-01').mkdir(parents=True)
    put(root/'EXECUTION_SPEC.json', query)
    body = unseal(json.loads(Path(query['recorded_observation']).read_text()))
    state = np.array(body['source_policy_view']['payload']['state_eef10'])
    raw = np.repeat(state[None], 40, axis=0)
    raw[:, 0] += np.linspace(.001, .008, 40)
    raw[:, 2] += .002*np.sin(np.linspace(0, 3, 40))
    raw[:, 9] = np.linspace(1., .95, 40)
    with (root/'fit-01/REAL_VLA_ACTION.npz').open('wb') as f:
        np.savez(f, physical_H40=raw, normalized=np.zeros((1, 40, 132), np.float32))
    result = dict(status='REAL_FROZEN_VLA_OFFLINE_INFERENCE_DONE', source_sha=c.QUERY_SOURCE,
        checkpoint_manifest_sha256=c.PINS['checkpoint'],
        recorded_observation_sha256=query['recorded_observation_sha256'],
        action_sha256=digest(root/'fit-01/REAL_VLA_ACTION.npz'),
        actual_VLA_inferences=1, actual_model_loads=1,
        native_integrations=0, GR00T_updates=0, WM_updates=0, Actor_updates=0)
    put(root/'fit-01/RESULT.private.json', result)
    put(root/'control/LEASE.json', dict(status='CLOSED', worker_exit_code=0,
        spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    put(root/'WORKER_EXIT.private.json', dict(exit_code=0))
    pins = dict(c.PINS, spec=digest(root/'EXECUTION_SPEC.json'),
        result=digest(root/'fit-01/RESULT.private.json'), action=result['action_sha256'],
        observation=query['recorded_observation_sha256'],
        frame_map=body['source_policy_view']['transform_sha256'])
    monkeypatch.setattr(c, 'PINS', pins)
    model, binary, observer = (tmp_path/n for n in ('model', 'binary', 'observer'))
    for p in (model, binary, observer): p.write_bytes(p.name.encode())
    spec = dict(phase=c.PHASE, profile='wm02-auto1-target', domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        physics=True, pilot1_binding='/fixture/original-P1-binding', source_root=str(source), source_files=files,
        WBC_changes=False, physics_dt_s=.002, wall_limit_s=500,
        registered_root='/fixture/new-trajectory', campaign_root='/fixture/campaign', candidate_index=10,
        reference_binding_version='MEASURED_EEF_AND_BASE_POSE_HOLD_V2', trajectory_query_root=str(root),
        model_path=str(model), native_binary=str(binary), observer_library=str(observer))
    trace = tmp_path/'trace-library'; trace.write_bytes(b'not a loadable trace library: offline fixture')
    build = tmp_path/'trace-build.json'
    put(build, dict(status='TRACE_ONLY_131072_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',
        trace_capacity=131072, trace_library_sha256=digest(trace), native_binary_sha256=digest(binary),
        observer_sha256=digest(observer), new_target_integrations=0,
        source_files={name:files[name] for name in (
            'src/go2arx5_vla/target_mujoco/init_trace.cpp', 'src/go2arx5_vla/target_mujoco/init_trace.hpp')}))
    spec.update(trace_library=str(trace), trajectory_trace_capacity_build=str(build),
                trajectory_trace_capacity_build_sha256=digest(build))
    scene = dict(scene_id='TRAIN_SCENE', parent='TRAIN_PARENT', split='TRAIN', model_path=str(model),
                 model_sha256=digest(model), static_check='PASSED_NO_INTEGRATION')
    catalog = tmp_path/'catalog.json'; put(catalog, dict(scenes=[scene], native_binary_sha256=digest(binary)))
    spec.update(scene_allowlist=str(catalog), scene_allowlist_sha256=digest(catalog))
    inputs = c.load_query(spec)
    anchor = np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    program = TrajectoryAdaptation(raw, inputs['query_tcp'], inputs['frame_map'], anchor, anchor, BASE, .03)
    e = episode(end_atom=program.end_atom)
    ep = tmp_path/'episode.json'; put(ep, e)
    spec.update(episode_manifest=str(ep), episode_manifest_sha256=digest(ep), max_ticks=c.endpoint(e))
    prior_root = tmp_path/'previous'; result_path = prior_root/'native-01/RESULT.json'
    previous = dict(status='NATIVE_STAGE_PASSED', phase='collect-target-fusion-reference-qualification-v1',
        campaign_uuid='campaign', candidate_index=10, shutdown_clean=True, failure=None,
        physics_stops=[], original_joint_or_warning_samples=0, reached_exact_pause=True,
        actual_target_integrations=4103, native_binary_sha256=digest(binary),
        observer_sha256=digest(observer), scene_sha256=digest(model),
        owned_process_exits=[dict(name=name, exit_code=0) for name in ('sim','controller','solver','planner')],
        native_outer_guard_rejection_count=0, native_outer_guard_rejections={}, trace_audit_error=None,
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS', errors=[], missing=[], native_step_records=4103),
        reference_qualification=dict(complete_task_atoms=64, solver_target_confirmed_atoms=64,
                                    episode=dict(scene_id='TRAIN_SCENE', parent='TRAIN_PARENT'),
                                    program=dict(candidate_id='Z_PLUS_0P5MM')))
    put(result_path, previous); pins['first_predecessor'] = digest(result_path)
    put(prior_root/'EXECUTION_SPEC.json', dict(phase=previous['phase']))
    put(prior_root/'control/LEASE.json', dict(status='CLOSED', worker_exit_code=0,
        spec_sha256=digest(prior_root/'EXECUTION_SPEC.json')))
    put(prior_root/'WORKER_EXIT.private.json', dict(exit_code=0))
    spec.update(previous_stage=str(result_path), previous_stage_sha256=digest(result_path))
    path = tmp_path/'spec.json'; put(path, spec)
    return spec, previous, path, e, inputs


def test_complete_saved_query_sidecar_and_original_ancestor_recipe(recipe):
    spec, previous, path, e, inputs = recipe
    actual, bound = c.episode_config(spec)
    np.testing.assert_array_equal(bound['raw'], inputs['raw'])
    assert actual == e and c.episode_limit(spec, previous) == c.endpoint(e)
    assert phase_limit(c.PHASE, previous, spec) == c.endpoint(e)
    check = cli.check(path)
    assert check['max_ticks'] == c.endpoint(e)
    assert not check['native_execution_authorized'] and check['new_integrations'] == 0
    assert inputs['provenance']['target_online_images'] is False


@pytest.mark.parametrize('change', ['action', 'result', 'spec', 'query-open', 'worker', 'pixels', 'source'])
def test_query_source_action_and_complete_capture_hashes_cannot_be_dropped(recipe, change):
    spec, *_ = recipe; root = Path(spec['trajectory_query_root'])
    if change in ('action', 'result', 'spec'):
        relative = {'action':'fit-01/REAL_VLA_ACTION.npz', 'result':'fit-01/RESULT.private.json',
                    'spec':'EXECUTION_SPEC.json'}[change]
        with (root/relative).open('ab') as f: f.write(b'changed')
    elif change == 'query-open': put(root/'control/LEASE.json', dict(status='ACTIVE'))
    elif change == 'worker': put(root/'WORKER_EXIT.private.json', dict(exit_code=1))
    elif change == 'pixels':
        query = json.loads((root/'EXECUTION_SPEC.json').read_text())
        image = Path(query['recorded_observation']).parent/'images/000001-body_rgb.png'
        image.write_bytes(b'changed')
    else: (Path(spec['source_root'])/'scripts/wm02/infer_fusion_vla.py').write_text('changed')
    with pytest.raises((ValueError, PermissionError)): c.episode_config(spec)


@pytest.mark.parametrize('change', [dict(physics_dt_s=.004), dict(wall_limit_s=601),
    dict(fusion_binding='/old-FUSION'), dict(WBC_changes=True), dict(parallel_batch_spec='/batch')])
def test_no_shared_safety_clock_or_owner_bypass(recipe, change):
    spec, *_ = recipe
    with pytest.raises(ValueError): c.episode_config(dict(spec, **change))


@pytest.mark.parametrize('change', ['capacity', 'library', 'cpp', 'integrations'])
def test_rehashed_trace_receipt_cannot_replace_actual_build_evidence(recipe, change):
    spec, *_ = recipe; path=Path(spec['trajectory_trace_capacity_build'])
    proof=json.loads(path.read_text())
    if change == 'capacity': proof['trace_capacity']=65536
    elif change == 'library': proof['trace_library_sha256']='0'*64
    elif change == 'integrations': proof['new_target_integrations']=1
    else: proof['source_files']['src/go2arx5_vla/target_mujoco/init_trace.cpp']='0'*64
    put(path,proof); spec['trajectory_trace_capacity_build_sha256']=digest(path)
    with pytest.raises(ValueError,match='ACTUAL_TRACE_ONLY_CAPACITY'): c.episode_config(spec)


@pytest.mark.parametrize('change', [dict(status='FAILED'), dict(shutdown_clean=False),
    dict(owned_process_exits=[]), dict(owned_process_exits=[dict(exit_code=139)]),
    dict(reached_exact_pause=False), dict(actual_target_integrations=4102),
    dict(original_joint_or_warning_samples=1), dict(physics_stops=['STOP'])])
def test_booleans_or_partial_predecessor_never_substitute_native_evidence(recipe, change, monkeypatch):
    spec, previous, *_ = recipe; bad = dict(previous, **change)
    put(Path(spec['previous_stage']), bad)
    spec['previous_stage_sha256'] = digest(spec['previous_stage'])
    monkeypatch.setitem(c.PINS, 'first_predecessor', spec['previous_stage_sha256'])
    with pytest.raises(ValueError): c.episode_limit(spec, bad)


def test_manifest_length_frame_and_explicit_anchor_are_not_adjustable(recipe):
    spec, _, _, e, _ = recipe
    for change in (dict(planned_end_atom=1000), dict(geometry_scale=.031),
                   dict(reference_anchor_mode='DECISION_REANCHOR'), dict(split='TEST'), dict(clip=True)):
        with pytest.raises(ValueError): c.episode_contract(dict(e, **change))
    ep = Path(spec['episode_manifest']); put(ep, dict(e, planned_end_atom=e['planned_end_atom']+1))
    spec['episode_manifest_sha256'] = digest(ep); spec['max_ticks'] += 25
    with pytest.raises(ValueError, match='RETIME_LENGTH'): c.episode_config(spec)


def test_checkonly_import_does_not_load_torch_ros_or_mujoco(recipe, monkeypatch):
    # CLI recipe itself is checked above with a complete synthetic producer.
    # Constants are production pins; the child only exercises import/help.
    code = ('import sys;import scripts.wm02.fusion_trajectory_qualification_check;'
            'import go2arx5_vla.world_model.fusion_trajectory_qualification_contract;'
            'assert not {"torch","rclpy","mujoco"}&sys.modules.keys()')
    subprocess.run([sys.executable, '-B', '-c', code], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    subprocess.run([sys.executable, '-B', '-m', 'scripts.wm02.fusion_trajectory_qualification_check', '--help'],
                   check=True, capture_output=True)
    _, _, path, *_ = recipe; calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
                        SimpleNamespace(main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]


class FakeAdapter:
    """No physics, ROS, model, renderer or command wrapper construction."""
    def __init__(self, spec, output):
        self.spec, self.output, self.step = spec, output, c.INIT_STEP
        self.base_pose_hold = BASE.copy(); self.reference_history = []; self.ros_evidence = []
        self.latest = dict(mode=SimpleNamespace(data='WBC'), ee=SimpleNamespace(time=self.step*.002))
        self.reference_qualification = None; self.events = []; self.installs = []
        self.grip_packets = []; self.grip_memory = None
    def prepare_dynamic(self):
        from go2arx5_vla.world_model.init_trace_reader import RECORD
        (self.output/'init.trace').write_bytes(struct.pack('<8sII', b'WM02IT01', RECORD.size, 131072))
    def observe(self): return dict(step=self.step, time_s=self.step*.002)
    def measured_task_state(self):
        q = np.zeros(34); q[2] = .3; q[3] = 1.
        return dict(native_step=self.step, time_s=self.step*.002, tcp=np.eye(4), object_pose=np.eye(4),
            qpos=q, qvel=np.zeros(32), ctrl=np.zeros(20), finger_q=np.array([.042, .042]),
            gripper_fraction=.042/.044, safety_valid=True, native_abort=False)
    def install_reference(self, action):
        n.validate_adapter_install(self, action)
        self.active_reference = dict(reference(round(self.step*.002*1e9)/1e9), action_EEF10=action.tolist())
        self.installs.append((self.step, action.copy())); return self.active_reference
    def wait_source(self): self.latest['ee'].time = round(self.step*.002*1e9)/1e9
    def reference_events(self): return []
    def command_event(self, value): self.events.append(value)
    def advance_to(self, step):
        assert len(self.grip_packets) == (self.step-c.INIT_STEP)//25+1
        submitted = self.grip_packets[-1]
        self.grip_memory = dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY', values=dict(
            initialized=True, has_command=True, steps=step, sequence=submitted['sequence'],
            opening=submitted['opening'], expiry=submitted['expiry']))
        self.step = step; return self.observe()
    def spin(self, duration): pass
    def capture_loaded_identity(self):
        return dict(sim=dict(loaded_files_sha256={self.spec['trace_library']:digest(self.spec['trace_library'])}))
    def cancel_reference(self): return dict(kind='fake-cancel-no-native-claim')
    def seal_trace(self, step): self.sealed = step


def setup_flow(monkeypatch, adapter):
    from go2arx5_vla.world_model import next2_contact
    from scripts.pi0294 import task_state
    original_write = task_state.write_gripper
    def write(path, sequence, source, opening):
        assert sequence == len(adapter.grip_packets)
        original_write(path, sequence, source, opening)
        adapter.grip_packets.append(dict(sequence=sequence, source_mpc_s=source,
                                         opening=opening, expiry=source+.4))
    monkeypatch.setattr(task_state, 'write_gripper', write)
    monkeypatch.setattr(next2_contact, 'gripper_memory', lambda a: a.grip_memory)
    monkeypatch.setattr(n, 'consumed_window', lambda a, start, before, after, clock: dict(
        native_start_step=before['step'], native_end_step=after['step'], actual_ticks=25, clock=clock,
        solver_policies_with_exact_installed_target=[dict(fake_only=True)]))


def test_full_fake_flow_multiple_suffixes_tail_history_and_contiguous_grip(recipe, tmp_path, monkeypatch):
    spec, _, _, e, _ = recipe
    adapter = FakeAdapter(spec, tmp_path); setup_flow(monkeypatch, adapter)
    report = n.run(adapter)
    count = (c.endpoint(e)-2503)//25
    assert report['complete_task_atoms'] == report['solver_target_confirmed_atoms'] == report['gripper_confirmed_atoms'] == count
    assert report['all_40_knots_consumed'] and adapter.sealed == c.endpoint(e)
    assert len(adapter.grip_packets) == len(adapter.reference_history) == count
    assert [p['sequence'] for p in adapter.grip_packets] == list(range(count))
    assert len(adapter.installs) > 12
    decision = json.loads((tmp_path/'TRAJECTORY_QUALIFICATION_DECISION.private.json').read_text())
    history = decision['actual_history']
    assert history['actual_state_history_shape'] == [9, 106]
    assert history['actual_history_native_steps'] == list(range(3503, 3704, 25))
    assert history['past_reference_native_end_steps'] == list(range(3528, 3704, 25))
    assert np.asarray(history['past_consumed_EEF10_BASE7']).shape == (8, 17)
    program = adapter.trajectory_program
    for step, action in adapter.installs:
        if step >=3703:
            np.testing.assert_array_equal(action, program.preview((step-3703)//25))
    post = [w for w in report['windows'] if w['trajectory_elapsed_atom'] is not None]
    np.testing.assert_array_equal(post[-1]['desired_interval_end_EEF10'], program.desired_at(program.end_atom))
    np.testing.assert_array_equal(post[-1]['nominal_submitted_EEF10'], program.preview(program.end_atom)[0])
    assert all(w['gripper']['command_consumed'] for w in post)
    assert report['VLA_commands'] == 0 and not report['target_task_qualification']
    assert report['status'] != 'NATIVE_STAGE_PASSED'


def test_next_scale_requires_complete_immediate_previous_family_atom_evidence(recipe, tmp_path, monkeypatch):
    spec, previous, _, e, inputs = recipe
    adapter = FakeAdapter(spec, tmp_path); setup_flow(monkeypatch, adapter)
    report = n.run(adapter)
    end = c.endpoint(e)
    previous.update(phase=c.PHASE, actual_target_integrations=end, reference_qualification=report)
    previous['trace_audit']['native_step_records'] = end
    p = Path(spec['previous_stage']); put(p, previous)
    prior_root = p.parent.parent
    put(prior_root/'EXECUTION_SPEC.json', dict(phase=c.PHASE, max_ticks=end))
    put(prior_root/'control/LEASE.json', dict(status='CLOSED', worker_exit_code=0,
        spec_sha256=digest(prior_root/'EXECUTION_SPEC.json')))
    spec['previous_stage_sha256'] = digest(p)
    anchor = np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    program = TrajectoryAdaptation(inputs['raw'], inputs['query_tcp'], inputs['frame_map'], anchor, anchor, BASE, .1)
    next_episode = episode(.1, program.end_atom)
    put(Path(spec['episode_manifest']), next_episode)
    spec.update(episode_manifest_sha256=digest(spec['episode_manifest']), max_ticks=c.endpoint(next_episode))
    assert c.episode_limit(spec, previous) == c.endpoint(next_episode)
    previous['reference_qualification']['windows'][-1]['gripper']['consumed_wrapper']['values']['sequence'] += 1
    put(p, previous); spec['previous_stage_sha256'] = digest(p)
    with pytest.raises(ValueError, match='EACH_ACTUAL_POSE_AND_GRIPPER'): c.episode_limit(spec, previous)


def test_original_outer_commission_accepts_variable_trajectory_endpoint(recipe, tmp_path, monkeypatch):
    from scripts.wm02 import commission_target
    from go2arx5_vla.world_model import init_trace_reader, native_replay, target_sequential
    spec, _, _, e, _ = recipe
    spec.update(source_sha='synthetic-future-source', candidate_source_sha='synthetic-native-source')
    output = tmp_path/'mock-outer'; output.mkdir()
    class OuterAdapter(FakeAdapter):
        def __init__(self, spec, output):
            super().__init__(spec, output); setup_flow(monkeypatch, self)
            self.node=None; self.exit_records=[]; self.trace_sealed=False
        def start(self): (self.output/'state.bin.journal').write_bytes(b'fake-not-native-evidence')
        def dynamic_phase(self): return True
        def reference_phase(self): return True
        def dynamic_diagnostic(self): return n.run(self)
        def close(self): self.exit_records=[dict(name=name, exit_code=0) for name in ('sim','controller','solver','planner')]
    monkeypatch.setattr(target_sequential, 'TargetSequentialAdapter', OuterAdapter)
    monkeypatch.setattr(init_trace_reader, 'audit_trace', lambda *a, **k:dict(status='RECORDED_TRACE_CHECKS_PASS', fake_only=True))
    monkeypatch.setattr(native_replay, 'packets', lambda path:[None]*c.endpoint(e))
    assert commission_target.run(spec, output) == 0
    result = json.loads((output/'RESULT.json').read_text())
    assert result['actual_target_integrations'] == c.endpoint(e) and result['shutdown_clean']
    assert result['reference_qualification']['all_40_knots_consumed']
    assert result['reference_qualification']['complete_task_atoms'] > 64


def test_actual_trace_header_and_mapped_library_are_both_required(recipe, tmp_path):
    spec, *_ = recipe; adapter = FakeAdapter(spec, tmp_path); adapter.prepare_dynamic()
    proof = n.actual_trace_binding(adapter)
    assert proof['actual_trace_capacity'] == 131072
    raw = bytearray((tmp_path/'init.trace').read_bytes()); struct.pack_into('<I', raw, 12, 65536)
    (tmp_path/'init.trace').write_bytes(raw)
    with pytest.raises(ValueError, match='HEADER_CAPACITY'): n.actual_trace_binding(adapter)
    adapter.prepare_dynamic()
    adapter.capture_loaded_identity = lambda:dict(sim=dict(loaded_files_sha256={}))
    with pytest.raises(ValueError, match='LOAD_PINNED'): n.actual_trace_binding(adapter)


def test_new_phase_defers_only_install_time_gripper_writes(tmp_path, monkeypatch):
    from scripts.pi0294 import task_state
    calls=[]; monkeypatch.setattr(task_state, 'write_gripper', lambda *args:calls.append(args))
    monkeypatch.setattr(n, 'validate_adapter_install', lambda *args:dict(checked=True))
    action = c.hold_preview(np.eye(4))
    def prepared(phase):
        adapter = TargetSequentialAdapter(dict(phase=phase), tmp_path)
        adapter.reference_sequence=-1; adapter.latest=dict(ee=SimpleNamespace(time=5.006))
        adapter.wait_source=lambda:None; adapter.observe=lambda:dict(step=2503,time_s=5.006)
        adapter.command_event=lambda event:None; adapter.refs=object(); rows=[]
        def publish(topic, wire):
            import hashlib
            ref=hashlib.sha256(wire.encode()).hexdigest()
            rows.extend([dict(kind=kind,ref_id=ref) for kind in ('accept','installed')])
        adapter.publish=publish; adapter.reference_events=lambda:rows
        adapter.spin=lambda seconds,predicate:predicate()
        return adapter
    prepared(c.PHASE).install_reference(action)
    assert calls == []
    prepared('g2-standing-eef-y-v2').install_reference(action)
    assert len(calls) == 1 and calls[0][1] == 0


def test_wrapper_mismatch_is_preserved_and_never_sets_native_qualification(recipe, tmp_path, monkeypatch):
    spec, *_ = recipe; adapter = FakeAdapter(spec, tmp_path); setup_flow(monkeypatch, adapter)
    from go2arx5_vla.world_model import next2_contact
    def bad_memory(a):
        value = copy.deepcopy(a.grip_memory); value['values']['sequence'] += 1; return value
    monkeypatch.setattr(next2_contact, 'gripper_memory', bad_memory)
    with pytest.raises(ValueError, match='WRAPPER_CONTIGUOUS'): n.run(adapter)
    assert adapter.step == 2528 and adapter.reference_qualification is None


def test_bad_full_suffix_rejected_before_original_install_native_import(tmp_path):
    adapter = TargetSequentialAdapter(dict(phase=c.PHASE, max_ticks=6000), tmp_path)
    adapter.trajectory_initial_tcp = np.eye(4); adapter.trajectory_initial_base = BASE
    adapter.trajectory_program = None; adapter.base_pose_hold = BASE
    adapter.observe = lambda: dict(step=2503)
    adapter.command_event = lambda event: None
    class ReachedOriginal(Exception): pass
    adapter.wait_source = lambda: (_ for _ in ()).throw(ReachedOriginal())
    action = c.hold_preview(np.eye(4)); bad = action.copy(); bad[-1, 0] += .001
    with pytest.raises(ValueError, match='FULL_H40'): adapter.install_reference(bad)
    with pytest.raises(ReachedOriginal): adapter.install_reference(action)
    old = TargetSequentialAdapter(dict(phase='g2-standing-eef-y-v2'), tmp_path)
    old.wait_source = adapter.wait_source
    with pytest.raises(ReachedOriginal): old.install_reference(action)


def test_dispatch_and_existing_phase_limits_unchanged(tmp_path, monkeypatch):
    adapter = TargetSequentialAdapter(dict(phase=c.PHASE), tmp_path)
    monkeypatch.setattr(n, 'run', lambda a: ('new-trajectory-only', a))
    assert adapter.dynamic_diagnostic() == ('new-trajectory-only', adapter)
    assert phase_limit('cold-1', None) == 50
    with pytest.raises(PermissionError): phase_limit('collect-target-unknown', None)


def test_new_trace_first_loader_order_does_not_change_other_phases(tmp_path):
    from go2arx5_vla.world_model.target_sequential import native_display_command
    base = dict(xvfb_run='/fixture/xvfb', native_binary='/fixture/native',
                trace_library='/fixture/trace', outer_library='/fixture/outer',
                observer_library='/fixture/observer')
    env = dict(LD_PRELOAD='unsafe-for-wrapper', WM02_INIT_TRACE='unsafe-for-wrapper')
    for phase, proof, expected in (
        (c.PHASE, dict(trajectory_trace_capacity_build='/proof'), '/fixture/trace:/fixture/outer:/fixture/observer'),
        ('collect-target-next2-contact', dict(extended_trace_build='/proof'), '/fixture/trace:/fixture/outer:/fixture/observer'),
        ('collect-target-fusion-reference-qualification-v1', dict(trajectory_trace_capacity_build='/proof'), '/fixture/outer:/fixture/observer:/fixture/trace'),
        ('cold-1', {}, '/fixture/outer:/fixture/observer:/fixture/trace')):
        argv, wrapper_env = native_display_command(dict(base, phase=phase, **proof), tmp_path, env)
        assert 'LD_PRELOAD='+expected in argv
        assert 'LD_PRELOAD' not in wrapper_env and 'WM02_INIT_TRACE' not in wrapper_env
