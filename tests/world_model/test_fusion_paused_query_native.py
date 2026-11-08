"""Synthetic HOLD/query native seams; no world, renderer, model or worker run."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_paused_query_native as n
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as geometry
from go2arx5_vla.world_model import fusion_short_skill as skill
from go2arx5_vla.world_model.auto1_native import phase_limit
from go2arx5_vla.world_model.fusion_observation import transform_hash
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from scripts.wm02 import fusion_paused_query_check as cli
from tests.world_model.test_fusion_trajectory_response_native import response_recipe
from test_fusion_trajectory_qualification import (
    recipe as geometry_recipe, FakeAdapter, setup_flow as geometry_flow,
)
from test_fusion_reference_qualification_contract import reference
from test_infer_fusion_observation import put


@pytest.fixture
def paused_recipe(response_recipe, tmp_path, monkeypatch):
    spec, previous, _, _, _, _ = response_recipe
    project = Path(__file__).resolve().parents[2]
    for name in n.SOURCE_MEMBERS:
        member = Path(spec['source_root']) / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes((project / name).read_bytes())
        spec['source_files'][name] = digest(member)
    root = tmp_path / 'SYNTHETIC_HOLD_ONLY_NATIVE'
    root.mkdir()
    task = skill.task_document(digest(spec['model_path']))
    task_path, score_path = tmp_path / 'TASK.json', tmp_path / 'SCORE.json'
    put(task_path, task); put(score_path, skill.score_contract())
    spec.update(phase=n.PHASE, max_ticks=n.END_STEP, registered_root=str(root),
        paused_target_task=str(task_path), paused_target_task_file_sha256=digest(task_path),
        paused_target_task_sha256=skill.document_hash(task), paused_score_contract=str(score_path),
        paused_score_contract_file_sha256=digest(score_path),
        paused_source_from_target=np.eye(4).tolist(), paused_frame_map_sha256=transform_hash(np.eye(4)),
        paused_query_gpu_spec=str(tmp_path / 'SYNTHETIC_PREREGISTERED_GPU' / 'EXECUTION_SPEC.json'))
    episode = dict(schema=n.SCHEMA, phase=n.PHASE, purpose='LIVE_PAUSED_CAPTURE_QUERY_ONLY',
        run=str(root), experiment_role='NON_ACTION_QUERY_DIAGNOSTIC', split='TRAIN',
        scene_id='F2T06', parent='FUSION_F2_T06_ROOT_XY_V1', decision_native_step=n.END_STEP,
        prehold_atoms=48, task_sha256=skill.document_hash(task), VLA_action_installations=0,
        WM_corrections=0, max_queries=1, physics_dt_s=.002)
    put(Path(spec['episode_manifest']), episode)
    spec['episode_manifest_sha256'] = digest(spec['episode_manifest'])
    catalog = json.loads(Path(spec['scene_allowlist']).read_text())
    catalog['scenes'][0].update(scene_id=episode['scene_id'], parent=episode['parent'])
    put(Path(spec['scene_allowlist']), catalog)
    spec['scene_allowlist_sha256'] = digest(spec['scene_allowlist'])
    previous['reference_qualification']['episode'].update(scene_id=episode['scene_id'], parent=episode['parent'])
    previous['trace_audit']['writer'] = dict(dropped=0)
    put(Path(spec['previous_stage']), previous)
    spec['previous_stage_sha256'] = digest(spec['previous_stage'])
    spec_path = root / 'EXECUTION_SPEC.json'
    put(spec_path, spec)
    preflights = []
    def preflight(path, native_spec):
        assert path == Path(spec['paused_query_gpu_spec']) and native_spec == spec
        preflights.append((path, native_spec))
        return dict(status='SYNTHETIC_ONLY_ALREADY_REGISTERED_PAIR', native_execution_authorized=False)
    # The separate bridge suite authenticates real pair/registry/lease files.
    # Here only its external two-scope admission seam is doubled.
    monkeypatch.setattr(bridge, 'preflight_pair', preflight)
    return spec, previous, spec_path, episode, task, preflights


def test_hold_recipe_binds_exact_task_score_scene_and_original_predecessor(paused_recipe):
    spec, previous, path, episode, task, calls = paused_recipe
    assert n.episode_config(spec) == (episode, task)
    assert n.episode_limit(spec, previous) == 3703
    assert phase_limit(n.PHASE, previous, spec) == 3703
    checked = cli.check(path)
    assert checked['max_ticks'] == 3703
    assert checked['new_integrations'] == checked['VLA_queries'] == checked['VLA_action_installations'] == 0
    assert checked['existing_worker'] == 'scripts.wm02.pilot1_native_worker'
    assert not checked['native_execution_authorized'] and len(calls) == 4


@pytest.mark.parametrize('changes', [
    dict(max_ticks=3728), dict(wall_limit_s=1201), dict(physics_dt_s=.004),
    dict(WBC_changes=True), dict(physics=False), dict(fusion_binding='/old-domain'),
    dict(parallel_batch_spec='/two-worlds'), dict(reference_readiness_probe=True),
])
def test_no_tick_clock_wbc_or_parallel_scope_bypass(paused_recipe, changes):
    spec, _, _, _, _, calls = paused_recipe
    with pytest.raises(ValueError): n.episode_config(dict(spec, **changes))
    assert calls == []


@pytest.mark.parametrize('fault', ['source', 'task', 'score', 'episode', 'frame', 'trace_capacity'])
def test_changed_bytes_or_semantics_fail_before_pair_preflight(paused_recipe, fault):
    spec, _, _, episode, task, calls = paused_recipe
    if fault == 'source':
        member = Path(spec['source_root']) / n.SOURCE_MEMBERS[0]
        member.write_bytes(member.read_bytes() + b'\n# synthetic mutation\n')
    elif fault == 'task':
        task['language'] = 'Touch the object.'
        put(Path(spec['paused_target_task']), task)
        spec['paused_target_task_file_sha256'] = digest(spec['paused_target_task'])
    elif fault == 'score':
        contract = skill.score_contract(); contract['minimum_signed_progress_m'] = 0
        put(Path(spec['paused_score_contract']), contract)
        spec['paused_score_contract_file_sha256'] = digest(spec['paused_score_contract'])
    elif fault == 'episode':
        episode['VLA_action_installations'] = 1
        put(Path(spec['episode_manifest']), episode)
        spec['episode_manifest_sha256'] = digest(spec['episode_manifest'])
    elif fault == 'frame': spec['paused_frame_map_sha256'] = '0' * 64
    else: spec['trajectory_trace_capacity'] = 524288
    with pytest.raises((ValueError, KeyError)): n.episode_config(spec)
    assert calls == []


def _update_predecessor(spec, previous, **spec_changes):
    prior_root = Path(spec['previous_stage']).parent.parent
    prior_path = prior_root / 'EXECUTION_SPEC.json'
    if spec_changes:
        prior = json.loads(prior_path.read_text()); prior.update(spec_changes)
        put(prior_path, prior)
        lease_path = prior_root / 'control/LEASE.json'
        lease = json.loads(lease_path.read_text()); lease['spec_sha256'] = digest(prior_path)
        put(lease_path, lease)
    put(Path(spec['previous_stage']), previous)
    spec['previous_stage_sha256'] = digest(spec['previous_stage'])


@pytest.mark.parametrize('fault', ['failed', 'open', 'dropped', 'window', 'gripper', 'half', 'shortened'])
def test_failed_partial_or_shortened_full1_cannot_admit_hold_query(paused_recipe, fault):
    spec, previous, _, _, _, _ = paused_recipe
    report = previous['reference_qualification']
    if fault == 'failed': previous['status'] = 'NATIVE_STAGE_FAILED_PRESERVED'
    elif fault == 'open':
        lease_path = Path(spec['previous_stage']).parent.parent / 'control/LEASE.json'
        lease = json.loads(lease_path.read_text()); lease['status'] = 'ACTIVE'; put(lease_path, lease)
    elif fault == 'dropped': previous['trace_audit']['writer']['dropped'] = 1
    elif fault == 'window': report['windows'].pop()
    elif fault == 'gripper': report['windows'][-1]['gripper']['consumed_wrapper']['values']['sequence'] += 1
    elif fault == 'half': report['episode']['geometry_scale'] = .5
    else:
        # All internal receipt counters agree, but the frozen full1 program did
        # not reach its declared endpoint. Rehashing it cannot grant admission.
        end = previous['actual_target_integrations'] - 25
        report['windows'].pop()
        for key in ('complete_task_atoms', 'solver_target_confirmed_atoms', 'gripper_confirmed_atoms'):
            report[key] -= 1
        report['native_endpoint_step'] = previous['actual_target_integrations'] = end
        previous['trace_audit']['native_step_records'] = end
        _update_predecessor(spec, previous, max_ticks=end)
    _update_predecessor(spec, previous)
    with pytest.raises(ValueError): n.episode_limit(spec, previous)


class HoldFake(FakeAdapter):
    def __init__(self, spec, output):
        super().__init__(spec, output)
        self.prepared = 0; self.advances = []; self.capture_calls = []
    def prepare_dynamic(self):
        self.prepared += 1
        from go2arx5_vla.world_model.init_trace_reader import RECORD
        (self.output / 'init.trace').write_bytes(struct.pack('<8sII', b'WM02IT01', RECORD.size, 262144))
    def install_reference(self, action):
        n.validate_adapter_install(self, action)
        self.active_reference = dict(reference(round(self.step * .002 * 1e9) / 1e9), action_EEF10=action.tolist())
        self.installs.append((self.step, action.copy())); return self.active_reference
    def advance_to(self, step):
        self.advances.append(step)
        return super().advance_to(step)


def _flow(monkeypatch, adapter, capture=None):
    geometry_flow(monkeypatch, adapter)
    monkeypatch.setattr(n, 'consumed_window', lambda a, start, before, after, clock: dict(
        native_start_step=before['step'], native_end_step=after['step'], actual_ticks=25,
        clock=clock, solver_policies_with_exact_installed_target=[dict(synthetic_only=True)]))
    def captured(actual, **kwargs):
        assert actual is adapter and adapter.step == 3703
        assert adapter.advances == list(range(2528, 3704, 25))
        adapter.capture_calls.append(kwargs)
        if capture is not None: return capture(actual, kwargs)
        return dict(status='SYNTHETIC_QUERY_RETURN', online_action_execution=False)
    monkeypatch.setattr(bridge, 'capture_and_wait_query', captured)


def test_48_atoms_only_initial_hold_then_one_capture_and_no_VLA_install(paused_recipe, tmp_path, monkeypatch):
    spec, _, _, episode, task, preflights = paused_recipe
    output = tmp_path / 'hold-flow'; output.mkdir()
    adapter = HoldFake(spec, output); _flow(monkeypatch, adapter)
    assert n.run(adapter) is None
    report = adapter.reference_qualification
    assert adapter.prepared == 1 and len(preflights) == 1
    assert adapter.step == adapter.sealed == report['native_endpoint_step'] == 3703
    assert report['complete_task_atoms'] == report['solver_target_confirmed_atoms'] == report['gripper_confirmed_atoms'] == 48
    assert len(adapter.capture_calls) == 1 and adapter.capture_calls[0]['capture_index'] == 0
    assert adapter.capture_calls[0]['task'] == task
    assert adapter.capture_calls[0]['task_sha256'] == skill.document_hash(task)
    assert adapter.capture_calls[0]['gpu_spec_path'] == Path(spec['paused_query_gpu_spec'])
    assert [row['sequence'] for row in adapter.grip_packets] == list(range(48))
    assert 3503 in [step for step, _ in adapter.installs]
    for step, action in adapter.installs:
        assert step < 3703
        np.testing.assert_array_equal(action, geometry.hold_preview(np.eye(4)))
    assert report['episode'] == episode and report['prefix']['qualified']
    assert report['VLA_action_installations'] == report['WM_corrections'] == 0
    assert not report['task_executed'] and not report['full_task_success']
    assert json.loads((output / 'PAUSED_QUERY_NATIVE.private.json').read_text()) == report


def test_preregistered_pair_rejection_happens_before_dynamic_prepare(paused_recipe, tmp_path, monkeypatch):
    spec, *_ = paused_recipe
    adapter = HoldFake(spec, tmp_path)
    def denied(*args): raise PermissionError('SYNTHETIC_ORIGINAL_PAIR_LEASE_REJECTED')
    monkeypatch.setattr(bridge, 'preflight_pair', denied)
    with pytest.raises(PermissionError, match='PAIR_LEASE_REJECTED'): n.run(adapter)
    assert adapter.prepared == 0 and adapter.advances == adapter.installs == []
    assert not (tmp_path / 'init.trace').exists()


@pytest.mark.parametrize('fault', ['query_failed', 'advanced_while_querying'])
def test_no_success_or_action_after_failed_or_moving_query(paused_recipe, tmp_path, monkeypatch, fault):
    spec, *_ = paused_recipe
    output = tmp_path / 'failed-flow'; output.mkdir()
    adapter = HoldFake(spec, output)
    def fail(actual, kwargs):
        if fault == 'query_failed': raise RuntimeError('SYNTHETIC_QUERY_FAILED_NO_RETRY')
        actual.step += 25
        return dict(status='SYNTHETIC_INVALID_MOVING_RETURN')
    _flow(monkeypatch, adapter, fail)
    with pytest.raises((RuntimeError, ValueError)): n.run(adapter)
    assert len(adapter.capture_calls) == 1 and adapter.reference_qualification is None
    assert adapter.advances[-1] == 3703 and all(step < 3703 for step, _ in adapter.installs)
    assert not (output / 'PAUSED_QUERY_NATIVE.private.json').exists()


@pytest.mark.parametrize('fault', ['proposal', 'base'])
def test_install_guard_rejects_proposed_motion_or_base_change(paused_recipe, tmp_path, fault):
    spec, *_ = paused_recipe
    adapter = HoldFake(spec, tmp_path)
    adapter.paused_query_initial_tcp = np.eye(4)
    adapter.paused_query_initial_base = list(adapter.base_pose_hold)
    action = geometry.hold_preview(np.eye(4)).copy()
    if fault == 'proposal': action[:, 0] += .001
    else: adapter.base_pose_hold[0] += .001
    with pytest.raises(ValueError, match='ONLY_EXACT_INITIAL_HOLD'): n.validate_adapter_install(adapter, action)


def test_check_only_and_execute_use_original_worker_and_new_adapter_hook(paused_recipe, tmp_path, monkeypatch):
    _, _, path, *_ = paused_recipe
    calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
        SimpleNamespace(main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path)]) == 0 and calls == []
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]
    adapter = TargetSequentialAdapter(dict(phase=n.PHASE), tmp_path)
    monkeypatch.setattr(n, 'run', lambda actual: ('new-paused-query', actual))
    assert adapter.dynamic_diagnostic() == ('new-paused-query', adapter)
    code = ('import sys;import scripts.wm02.fusion_paused_query_check;'
            'import go2arx5_vla.world_model.fusion_paused_query_native;'
            'assert not {"torch","rclpy","mujoco"}&sys.modules.keys()')
    project = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, '-B', '-c', code], check=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(project / 'src'), str(project)))))


def test_original_commission_entry_requires_admitted_target_before_run(tmp_path, monkeypatch):
    from go2arx5_vla.world_model import auto1_native
    from scripts.wm02 import commission_target
    spec_path, output = tmp_path / 'EXECUTION_SPEC.json', tmp_path / 'native-01'
    spec, events = dict(phase=n.PHASE), []
    @contextmanager
    def admitted(path, run_id, actual_output):
        assert (path, run_id, actual_output) == (spec_path, 'native-01', output)
        events.append('admitted_target')
        yield spec
        events.append('closed')
    def run(actual, actual_output):
        assert actual is spec and actual_output == output and events == ['admitted_target']
        events.append('native_run'); return 0
    monkeypatch.setattr(auto1_native, 'admitted_target', admitted)
    monkeypatch.setattr(commission_target, 'run', run)
    assert commission_target.main(['--spec', str(spec_path), '--output', str(output), '--run-id', 'native-01']) == 0
    assert events == ['admitted_target', 'native_run', 'closed']


def test_original_native_worker_keeps_commission_route_and_closed_lease(paused_recipe, monkeypatch):
    from scripts.wm02 import pilot1_native_worker, commission_target
    spec, _, path, *_ = paused_recipe
    root = Path(spec['registered_root']); lease_path = root / 'control/LEASE.json'
    spec.update(lease=str(lease_path), source_sha='b' * 40)
    put(path, spec)
    put(lease_path, dict(status='ACTIVE', run_uuid=root.name, spec_sha256=digest(path)))
    calls = []
    def commission(argv):
        calls.append(argv); return 0
    monkeypatch.setattr(commission_target, 'main', commission)
    monkeypatch.setattr(pilot1_native_worker.signal, 'signal', lambda *args: None)
    assert pilot1_native_worker.main(['--spec', str(path)]) == 0
    assert calls == [['--spec', str(path), '--output', str(root / 'native-01'), '--run-id', 'native-01']]
    lease = json.loads(lease_path.read_text())
    assert lease['status'] == 'CLOSED' and lease['worker_exit_code'] == 0
    assert lease['spec_sha256'] == digest(path)
    assert json.loads((root / 'WORKER_EXIT.private.json').read_text())['failure'] is None
    with pytest.raises(PermissionError): pilot1_native_worker.main(['--spec', str(path)])
    assert len(calls) == 1


def test_paused_phase_install_uses_hold_guard_and_defers_gripper_write(tmp_path, monkeypatch):
    from scripts.pi0294 import task_state
    guarded, grippers = [], []
    monkeypatch.setattr(n, 'validate_adapter_install', lambda a, action: guarded.append(action.copy()) or dict(checked=True))
    monkeypatch.setattr(task_state, 'write_gripper', lambda *args: grippers.append(args))
    adapter = TargetSequentialAdapter(dict(phase=n.PHASE), tmp_path)
    adapter.reference_sequence = -1
    adapter.latest = dict(ee=SimpleNamespace(time=5.006))
    adapter.wait_source = lambda: None
    adapter.observe = lambda: dict(step=2503, time_s=5.006)
    adapter.command_event = lambda event: None
    adapter.refs = object(); rows = []
    def publish(topic, wire):
        ref = hashlib.sha256(wire.encode()).hexdigest()
        rows.extend(dict(kind=kind, ref_id=ref) for kind in ('accept', 'installed'))
    adapter.publish = publish
    adapter.reference_events = lambda: rows
    adapter.spin = lambda seconds, predicate: predicate()
    action = geometry.hold_preview(np.eye(4))
    adapter.install_reference(action)
    assert len(guarded) == 1 and grippers == []
    np.testing.assert_array_equal(guarded[0], action)
