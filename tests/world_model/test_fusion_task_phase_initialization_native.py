"""Synthetic lifecycle tests; no native qualification, model query or physics."""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_task_phase_initialization as phase
from go2arx5_vla.world_model import fusion_task_phase_initialization_contract as contract
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as native
from go2arx5_vla.world_model import fusion_source_posture_paused_query_v3 as v3
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_initialization_check as cli
from test_fusion_task_phase_initialization import setup as pure_setup, pack
from test_fusion_live_proposal_qualification_contract import recipe as prior_recipe, prior_files, put
from test_fusion_trajectory_qualification import FakeAdapter, setup_flow
from test_fusion_reference_qualification_contract import reference


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    spec, _, _, _, _ = prior_recipe.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(phase, 'SCENE', digest(spec['model_path']))
    setup = pure_setup.__wrapped__(monkeypatch)
    source = tmp_path / 'SYNTHETIC_ORIGINAL_71558.json'
    source.write_bytes(setup['args']['source_observation_bytes'])
    capsule = Path(spec['source_root'])
    project = Path(__file__).resolve().parents[2]
    for name in contract.SOURCE_MEMBERS:
        path = capsule / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / name).read_bytes())
        spec['source_files'][name] = digest(path)
    for key in list(spec):
        if key.startswith('live_proposal_'):
            spec.pop(key)
    spec.update(phase=contract.PHASE, kind=contract.KIND, source_sha='SYNTHETIC_FUTURE_IMMUTABLE_SOURCE',
        task_phase_source_observation=str(source), task_phase_source_observation_sha256=phase.SOURCE_OBSERVATION,
        task_phase_source_from_target=setup['frame'].tolist(), task_phase_frame_map_sha256=phase.FRAME_MAP)
    episode = phase.make_episode(run=spec['registered_root'], geometry_scale=.03)
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    spec['max_ticks'] = phase.endpoint(episode)
    monkeypatch.setattr(contract, 'TRACE_BUILD', spec['trajectory_trace_capacity_build_sha256'])
    old_root = tmp_path / 'SYNTHETIC_FULL1_V3'
    previous_episode = dict(v3._fixed_episode(), run=str(old_root), geometry_scale=1., planned_end_atom=500)
    previous, path, pin, spec_pin = prior_files(old_root, previous_episode, spec,
                                              source=contract.FIRST_PREDECESSOR_SOURCE)
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR', pin)
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR_SPEC', spec_pin)
    spec.update(previous_stage=str(path), previous_stage_sha256=pin)
    return spec, previous, episode, setup


def rewrite(spec, episode):
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    spec['max_ticks'] = phase.endpoint(episode)


def test_exact_source_static_episode_and_closed_full1_bootstrap_are_read_only(recipe, tmp_path):
    spec, previous, episode, _ = recipe
    root = Path(spec['registered_root'])
    before = set(root.iterdir())
    actual, source = contract.episode_config(spec)
    assert actual == episode and source['source_observation_bytes'] == Path(spec['task_phase_source_observation']).read_bytes()
    assert contract.episode_limit(spec, previous) == auto1_native.phase_limit(contract.PHASE, previous, spec) == 7103
    assert set(root.iterdir()) == before
    path = tmp_path / 'SYNTHETIC_SPEC.json'
    put(path, spec)
    checked = cli.check(path)
    assert checked['max_ticks'] == 7103 and checked['new_integrations'] == checked['VLA_queries'] == 0
    assert checked['runtime_goal_binding'] == 'NOT_CREATED_BY_READ_ONLY_CHECK'
    assert not checked['native_execution_authorized'] and not checked['native_qualification_claim']


@pytest.mark.parametrize('short_teacher',[False,True])
def test_operational_live_v9_entry_checks_the_actual_declared_horizon(recipe,tmp_path,short_teacher):
    from go2arx5_vla.world_model import fusion_task_phase_live_skill_contract_v9 as live
    from go2arx5_vla.world_model import fusion_task_phase_pose_slow_feedback_v9 as slow
    from go2arx5_vla.world_model import fusion_task_reference_diagnostic_v1 as teacher
    from test_fusion_task_reference_diagnostic_v1 import registered_short
    spec,_,_,_=recipe;root=Path(spec['registered_root'])
    if short_teacher:
        reg=registered_short(tmp_path);panel=Path(reg['response_learning_panel']);value=json.loads(panel.read_text())
        value['assignments'][0]['native_run']=root.name;put(panel,value)
        reg.update(registered_root=str(root),response_learning_panel_sha256=digest(panel));reg['response_learning_assignment']['panel_sha256']=digest(panel);spec.update(reg)
    else:spec.update(task_condition=live.MECHANICAL_TASK_CONDITION,max_ticks=66503,live_skill_start_step=34303)
    spec.update(phase=live.PHASE,kind=live.KIND,snapshot_binding_mode='PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V9',paused_query_gpu_spec='/synthetic/gpu/spec',paused_query_pair='/synthetic/pair')
    episode=slow.make_episode(run=str(root),geometry_scale=.5);spec['episode_manifest_sha256']=put(spec['episode_manifest'],episode)
    project=Path(__file__).resolve().parents[2]
    for name in live.SOURCE_MEMBERS:
        path=Path(spec['source_root'])/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((project/name).read_bytes());spec['source_files'][name]=digest(path)
    actual,_=contract.recipe_config(spec,episode,actual_phase=live.PHASE,actual_kind=live.KIND,source_members=live.SOURCE_MEMBERS)
    assert actual==episode
    spec['max_ticks']=66503 if short_teacher else 38703
    with pytest.raises(ValueError):contract.recipe_config(spec,episode,actual_phase=live.PHASE,actual_kind=live.KIND,source_members=live.SOURCE_MEMBERS)


@pytest.mark.parametrize('key,value', [
    ('phase', v3.PHASE), ('kind', 'FUSION_SOURCE_POSTURE_PAUSED_QUERY'),
    ('profile', 'wm02-fusion-vla-shadow'), ('physics', False),
    ('pilot1_binding', None), ('scope', 'WM02_FUSION_REFERENCE_SIM_ONLY'), ('fusion_binding', '/old'),
    ('parallel_batch_spec', '/batch'), ('physics_dt_s', .004), ('WBC_changes', True),
    ('wall_limit_s', 1201), ('wall_limit_s', True), ('trajectory_trace_capacity', 262144),
    ('paused_query_gpu_spec', '/new-query'), ('paused_query_pair', '/pair'),
    ('snapshot_binding_mode', 'SOME_QUERY_MODE')])
def test_initialization_cannot_bypass_original_scope_or_become_a_query(recipe, key, value):
    spec, *_ = recipe
    with pytest.raises(ValueError):
        contract.episode_config(dict(spec, **{key: value}))


@pytest.mark.parametrize('fault', ['source', 'source-pin', 'source-symlink', 'frame', 'scene',
    'capsule', 'clock', 'CF06', 'build-open', 'build-native'])
def test_immutable_recipe_and_native_trace_evidence_are_not_booleans(recipe, fault, tmp_path):
    spec, _, episode, _ = recipe
    if fault == 'source':
        Path(spec['task_phase_source_observation']).write_text('{}')
    elif fault == 'source-pin':
        spec['task_phase_source_observation_sha256'] = 'e' * 64
    elif fault == 'source-symlink':
        link = tmp_path / 'SOURCE_ALIAS'
        link.symlink_to(spec['task_phase_source_observation'])
        spec['task_phase_source_observation'] = str(link)
    elif fault == 'frame':
        spec['task_phase_source_from_target'][0][3] += .001
    elif fault == 'scene':
        Path(spec['model_path']).write_bytes(b'CHANGED')
    elif fault == 'capsule':
        (Path(spec['source_root']) / contract.SOURCE_MEMBERS[-1]).write_bytes(b'CHANGED')
    elif fault == 'clock':
        spec['episode_manifest_sha256'] = put(spec['episode_manifest'], dict(episode, planned_end_atom=129))
        spec['max_ticks'] += 25
    elif fault == 'CF06':
        Path(spec['trace_capacity_failed_result']).write_text('{}')
    elif fault == 'build-open':
        put(Path(spec['trajectory_trace_capacity_build']).parent.parent / 'control/LEASE.json', {'status': 'ACTIVE'})
    else:
        build = Path(spec['trajectory_trace_capacity_build'])
        data = json.loads(build.read_text())
        data['new_target_integrations'] = 1
        put(build, data)
    with pytest.raises((ValueError, PermissionError)):
        contract.episode_config(spec)


@pytest.mark.parametrize('fault', ['result-pin', 'source', 'phase', 'open', 'worker', 'truncated',
    'trace-drop', 'unconsumed-gripper', 'guard', 'physics-stopped'])
def test_first_grade_requires_actual_complete_556_atom_cold_boundary(recipe, fault):
    spec, previous, _, _ = recipe
    root = Path(spec['previous_stage']).parent.parent
    if fault == 'result-pin':
        spec['previous_stage_sha256'] = 'e' * 64
    elif fault == 'source':
        prior = json.loads((root / 'EXECUTION_SPEC.json').read_text())
        prior['source_sha'] = 'UNQUALIFIED_SOURCE'
        put(root / 'EXECUTION_SPEC.json', prior)
    elif fault == 'phase':
        previous['phase'] = 'collect-target-fusion-source-posture-qualification-v1'
    elif fault == 'open':
        put(root / 'control/LEASE.json', {'status': 'ACTIVE'})
    elif fault == 'worker':
        put(root / 'WORKER_EXIT.private.json', {'exit_code': 1, 'failure': None})
    elif fault == 'truncated':
        previous['reference_qualification']['windows'].pop()
    elif fault == 'trace-drop':
        previous['trace_audit']['writer']['dropped'] = 1
    elif fault == 'unconsumed-gripper':
        previous['reference_qualification']['windows'][-1]['gripper']['command_consumed'] = False
    elif fault == 'guard':
        previous['native_outer_guard_rejection_count'] = 1
    else:
        previous['physics_stops'] = [{'kind': 'stop', 'reason': 'SAFETY'}]
    if fault not in ('result-pin', 'source', 'open', 'worker'):
        # Even rehashing an altered result and changing the fixture's independently
        # fixed expected pin cannot turn missing native evidence into a pass.
        spec['previous_stage_sha256'] = put(spec['previous_stage'], previous)
        with pytest.MonkeyPatch.context() as local:
            local.setattr(contract, 'FIRST_PREDECESSOR', spec['previous_stage_sha256'])
            with pytest.raises((ValueError, PermissionError)):
                contract.episode_limit(spec, previous)
    else:
        with pytest.raises((ValueError, PermissionError)):
            contract.episode_limit(spec, previous)


class TaskAdapter(FakeAdapter):
    """An explicit no-physics fixture with independent constant measured TCP."""
    def __init__(self, spec, output, setup):
        super().__init__(spec, output)
        self.setup = setup
        self.actual_state = object()
        self.prepare_calls = 0
        self.measured_steps = []

    def _sample(self):
        result = copy.deepcopy(self.setup['args']['measured_decision_state'])
        result.update(native_step=self.step, time_s=self.step*.002,
            tcp=self.setup['initial'].copy(), finger_q=np.array([.042, .042]),
            gripper_fraction=.042/.044, left_object_contact=False, right_object_contact=False,
            native_grasp=False, contact_valid=True, illegal_contact=False,
            joint_limit_penetration=False, reset=False, takeover=False)
        result['qvel'][0] = self.step * 1e-6
        return result

    def _save_packet(self):
        (self.output / 'state.bin').write_bytes(pack(self._sample()))

    def prepare_dynamic(self):
        from go2arx5_vla.world_model.init_trace_reader import RECORD
        self.prepare_calls += 1
        (self.output / 'init.trace').write_bytes(struct.pack('<8sII', b'WM02IT01', RECORD.size, 524288))
        self._save_packet()

    def measured_task_state(self):
        self.measured_steps.append(self.step)
        return self._sample()

    def install_reference(self, action):
        native.validate_adapter_install(self, action)
        if self.step >= phase.DECISION_STEP:
            saved = self.output / 'TASK_PHASE_GOAL_BINDING.private.json'
            assert saved.is_file()
            assert json.loads(saved.read_text())['frozen_before_generated_reference_install']
        self.active_reference = dict(reference(round(self.step*.002*1e9)/1e9), action_EEF10=action.tolist())
        self.installs.append((self.step, action.copy()))
        return self.active_reference

    def advance_to(self, step):
        result = super().advance_to(step)
        self._save_packet()
        return result

    def query(self, *args, **kwargs):
        raise AssertionError('No model/query is permitted in initialization')


def flow(recipe, tmp_path, monkeypatch):
    spec, _, episode, setup = recipe
    output = tmp_path / 'SYNTHETIC_NATIVE_OUTPUT'
    output.mkdir()
    adapter = TaskAdapter(spec, output, setup)
    setup_flow(monkeypatch, adapter)
    _, source = contract.episode_config(spec)
    monkeypatch.setattr(native, '_verify_actual_fk', lambda *args: {'status': 'SYNTHETIC_FK_TEST_DOUBLE_NOT_NATIVE_EVIDENCE'})
    return adapter, episode, source


def test_shared_native_loop_freezes_actual_goal_before_install_and_saves_nine_actual_samples(recipe, tmp_path, monkeypatch):
    adapter, episode, source = flow(recipe, tmp_path, monkeypatch)
    episode_bytes = Path(adapter.spec['episode_manifest']).read_bytes()
    report = native.run_initialization_program(adapter, episode, source)
    assert adapter.prepare_calls == 1 and adapter.sealed == 7103
    assert report['complete_task_atoms'] == report['solver_target_confirmed_atoms'] == report['gripper_confirmed_atoms'] == 184
    assert report['all_40_knots_consumed'] and len(adapter.grip_packets) == 184
    assert [p['sequence'] for p in adapter.grip_packets] == list(range(184))
    assert report['actual_trace_binding']['actual_trace_capacity'] == 524288
    assert Path(adapter.spec['episode_manifest']).read_bytes() == episode_bytes
    binding = report['task_phase_initialization']
    assert digest(binding['goal_binding_path']) == binding['goal_binding_sha256']
    assert digest(binding['hold_samples_path']) == binding['hold_samples_sha256']
    rows = json.loads(Path(binding['hold_samples_path']).read_text())
    assert [s['native_step'] for s in rows] == list(range(6903, 7104, 25))
    assert len({s['qvel'][0] for s in rows}) == 9
    assert all(s['tcp'] == adapter.setup['initial'].tolist() for s in rows)
    goal = json.loads(Path(binding['goal_binding_path']).read_text())
    reconstructed = phase.load_inputs(**source,
        decision_packet_bytes=Path(goal['saved_packet_path']).read_bytes(),
        measured_decision_state=json.loads(Path(goal['measured_state_path']).read_text()))
    assert reconstructed['frozen'] == goal['frozen_input']
    assert binding['readiness'] == phase.query_readiness(rows, episode, reconstructed)
    assert binding['readiness']['actual_sample_steps'] == [s['native_step'] for s in rows]
    assert not binding['readiness']['query_ready'] and not binding['readiness']['predicates']['full_grade']
    assert binding['readiness']['metrics']['final_actual_TCP'] == adapter.setup['initial'].tolist()
    assert report['actual_VLA_commands'] == report['VLA_commands'] == report['VLA_queries'] == 0
    assert report['original_Processor_decodes'] == report['WM_updates'] == report['Actor_updates'] == 0
    assert not binding['object_and_goal_are_policy_inputs'] and not binding['source_pose_is_expert_action_label']
    assert not binding['query_readiness_is_native_or_GPU_grant'] and not report['full_task_success']
    decision = json.loads((adapter.output / 'TRAJECTORY_QUALIFICATION_DECISION.private.json').read_text())
    assert decision['episode'] == episode and decision['archived_input_provenance']['raw_is_VLA_action'] is False
    assert all(not w['raw_query_identity']['VLA_query_present'] for w in report['windows'])
    saved = json.loads((adapter.output / 'TRAJECTORY_QUALIFICATION.private.json').read_text())
    assert saved == report


@pytest.mark.parametrize('fault', ['existing-binding', 'FK', 'source-bytes', 'decision-packet', 'save-readback'])
def test_failed_binding_never_installs_generated_action_or_promotes_readiness(recipe, tmp_path, monkeypatch, fault):
    adapter, episode, source = flow(recipe, tmp_path, monkeypatch)
    if fault == 'existing-binding':
        (adapter.output / 'TASK_PHASE_GOAL_BINDING.private.json').write_text('{}')
    elif fault == 'FK':
        def bad(*args): raise ValueError('ORIGINAL_FK_WITNESS_REJECTED')
        monkeypatch.setattr(native, '_verify_actual_fk', bad)
    elif fault == 'source-bytes':
        source['source_observation_bytes'] = b'{}'
    elif fault == 'decision-packet':
        original = adapter.measured_task_state
        def bad_packet():
            result = original()
            if adapter.step == phase.DECISION_STEP:
                result['qpos'][0] += .001
            return result
        adapter.measured_task_state = bad_packet
    else:
        from scripts.pi0292 import durable
        original = durable.write
        def replace(path, value):
            original(path, value)
            if Path(path).name == 'TASK_PHASE_GOAL_BINDING.private.json':
                Path(path).write_text('{}')
        monkeypatch.setattr(durable, 'write', replace)
    with pytest.raises((ValueError, RuntimeError)):
        native.run_initialization_program(adapter, episode, source)
    assert not any(step >= phase.DECISION_STEP for step, _ in adapter.installs)
    assert not (adapter.output / 'TASK_PHASE_READINESS.private.json').exists()
    if fault == 'existing-binding':
        assert adapter.prepare_calls == 0 and not adapter.installs


def test_new_phase_route_uses_existing_quota_and_exact_typed_wrapper(tmp_path, monkeypatch):
    spec, previous = dict(phase=contract.PHASE), {'synthetic': True}
    calls = []
    monkeypatch.setattr(contract, 'episode_limit', lambda s, p: calls.append((s, p)) or 7103)
    assert auto1_native.phase_limit(contract.PHASE, previous, spec) == 7103 and calls == [(spec, previous)]
    adapter = target_sequential.TargetSequentialAdapter(spec, tmp_path)
    token = object()
    monkeypatch.setattr(native, 'run', lambda a: token if a is adapter else None)
    assert adapter.dynamic_diagnostic() is token
    assert target_sequential.native_task_bucket(contract.PHASE, True) == '64'
    assert auto1_native.phase_limit('cold-1', None) == 50
    assert target_sequential.native_task_bucket('cold-1', False) == '0'


@pytest.mark.parametrize('native_step', [3703, 10303])
def test_actualstate_fk_witness_uses_one_real_mujoco_packet_without_step_or_reload(tmp_path, monkeypatch, native_step):
    mujoco = pytest.importorskip('mujoco')
    from scripts.pi0294.task_state import ActualState, parse_state
    from go2arx5_vla.world_model.native_replay import forbid_integration
    bodies, motors = [], []
    for i in range(20):
        name = 'task_left_slide' if i == 0 else ('task_right_slide' if i == 1 else 'fixture_joint_' + str(i))
        geom = 'task_left_pad' if i == 0 else ('task_right_pad' if i == 1 else 'fixture_geom_' + str(i))
        bodies.append(f'<body pos="0 0 {i*.02}"><joint name="{name}" type="slide" axis="1 0 0"/>'
                      f'<geom name="{geom}" type="sphere" size=".001" mass=".01"/></body>')
        motors.append(f'<motor joint="{name}"/>')
    xml = ('<mujoco><worldbody><body name="fixture_base" pos=".2 .1 .7"><freejoint/>'
           '<geom type="sphere" size=".01" mass="1"/><site name="x5_ee"/>' + ''.join(bodies) + '</body>'
           '<body name="task_cube" pos=".35 .12 .46"><freejoint/>'
           '<geom name="task_cube_geom" type="box" size=".02 .02 .02" mass=".1"/></body>'
           '</worldbody><actuator>' + ''.join(motors) + '</actuator></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    path = tmp_path / 'SYNTHETIC_FK_ONLY.mjb'
    mujoco.mj_saveModel(model, str(path), None)
    monkeypatch.setattr(phase, 'SCENE', digest(path))
    reader = ActualState(model)
    qpos = model.qpos0.copy()
    qpos[reader.finger_q] = .042
    state = dict(native_step=native_step, time_s=native_step*.002, qpos=qpos, qvel=np.zeros(32), ctrl=np.zeros(20))
    packet = parse_state(pack(state))
    model_qpos0, body_pos = model.qpos0.copy(), model.body_pos.copy()
    with forbid_integration(mujoco):
        measured = reader.update(packet)
    adapter = SimpleNamespace(actual_state=reader, spec=dict(model_path=str(path),
        source_files={'scripts/pi0294/task_state.py': digest(Path(__file__).resolve().parents[2] / 'scripts/pi0294/task_state.py')}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Witness verification cannot forward/step/load a model')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    kwargs = {} if native_step == 3703 else dict(expected_step=native_step)
    proof = native._verify_actual_fk(adapter, packet, measured, **kwargs)
    assert proof['status'] == 'ORIGINAL_ACTUALSTATE_DATA_AND_FK_MATCH_SAVED_PACKET'
    assert proof['wrapper_mj_step_calls'] == proof['wrapper_mj_forward_calls'] == proof['wrapper_model_loads'] == 0
    assert np.array_equal(reader.data.qpos, packet['qpos']) and reader.data.time == native_step*.002
    assert np.array_equal(model.qpos0, model_qpos0) and np.array_equal(model.body_pos, body_pos)
    changed = copy.deepcopy(measured)
    changed['object_pose'][0, 3] += .001
    with pytest.raises(ValueError, match='ACTUAL_FK_MEASURED_object_pose'):
        native._verify_actual_fk(adapter, packet, changed, **kwargs)
    with pytest.raises(ValueError, match='ACTUAL_FK_EXACT_DECISION_CLOCK'):
        native._verify_actual_fk(adapter, packet, measured, expected_step=native_step+25)
    adapter.actual_state = SimpleNamespace(model=reader.model, data=reader.data, pose=reader.pose)
    with pytest.raises(ValueError, match='ORIGINAL_ACTUALSTATE_READER'):
        native._verify_actual_fk(adapter, packet, measured, **kwargs)


def test_check_import_has_no_native_renderer_or_model_and_execute_delegates_original_worker(recipe, tmp_path, monkeypatch):
    code = ('import sys;import scripts.wm02.fusion_task_phase_initialization_check;'
        'import go2arx5_vla.world_model.fusion_task_phase_initialization_native;'
        'assert not {"torch","mujoco","rclpy","transformers","gr00t"}&sys.modules.keys()')
    subprocess.run([sys.executable, '-B', '-c', code], check=True,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    spec, *_ = recipe
    path = tmp_path / 'SPEC.json'
    put(path, spec)
    calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
        SimpleNamespace(main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]
