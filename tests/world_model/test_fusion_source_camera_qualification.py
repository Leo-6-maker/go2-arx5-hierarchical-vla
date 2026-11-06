"""Saved-packet CPU qualification contract and explicit hermetic render seam."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import pytest

from scripts.wm02 import qualify_fusion_source_camera as q
from go2arx5_vla.world_model import fusion_scope
from go2arx5_vla.world_model import runtime
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_source_camera_capture import adapter_fixture, recipe, forbid_step


def write(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=digest(path))


def source_fixture(tmp_path, monkeypatch):
    adapter, binding, task, raw = adapter_fixture(tmp_path, monkeypatch)
    r = recipe(monkeypatch)
    source_root = Path(adapter.spec['registered_root'])
    episode = write(source_root/'EPISODE_MANIFEST.private.json', dict(scene_id='F2T06', split='TRAIN', parent='FUSION_F2_T06_ROOT_XY_V1'))
    lease_path = source_root/'control/LEASE.json'; lease_path.parent.mkdir()
    native_spec = dict(adapter.spec, phase='collect-target-fusion-paused-query-v2', source_sha='a'*40,
        lease=str(lease_path), episode_manifest=episode['path'], episode_manifest_sha256=episode['sha256'], max_ticks=3703)
    original_spec = write(source_root/'EXECUTION_SPEC.json', native_spec)
    worker = write(source_root/'WORKER_EXIT.private.json', dict(exit_code=0, failure=None))
    lease = write(lease_path, dict(status='CLOSED', worker_exit_code=0, run_uuid=source_root.name,
        spec_sha256=original_spec['sha256']))
    scene = dict(path=native_spec['model_path'], sha256=digest(native_spec['model_path']))
    monkeypatch.setattr(q, 'SCENE', scene['sha256'])
    native = dict(status='NATIVE_STAGE_PASSED', shutdown_clean=True, physics_stops=[],
        original_joint_or_warning_samples=0, actual_target_integrations=3703, reached_exact_pause=True,
        failure=None, native_outer_guard_rejection_count=0, trace_audit_error=None,
        phase=native_spec['phase'], source_sha='a'*40, scene_sha256=scene['sha256'],
        owned_process_exits=[dict(name=name, exit_code=0) for name in ('sim', 'planner', 'controller', 'solver')],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS', native_step_records=3703,
            writer=dict(events=30000, written=30000, dropped=0)))
    native_result = write(source_root/'native-01/RESULT.json', native)
    journal_path = source_root/'native-01/state.bin.journal'
    with journal_path.open('wb') as stream:
        for step in range(1, 3704):
            row = bytearray(raw); struct.pack_into('<Qd', row, 8, step, step*.002)
            stream.write(struct.pack('<I', len(row))); stream.write(row)
    identity_path = tmp_path/'source_identity.json'; identity_path.write_bytes(r.source_identity_bytes)
    monkeypatch.setattr(q, 'SOURCE_IDENTITY_SHA256', digest(identity_path))
    root = tmp_path/'camera-qual'; root.mkdir()
    output = root/'fit-01'; output.mkdir()
    spec = dict(source_sha='b'*40, registered_root=str(root), campaign_root=str(tmp_path), deadline_epoch=time.time()+180,
        wall_limit_s=180, profile=q.PROFILE, kind=q.KIND, execution_site=q.SITE, recipe=q.RECIPE,
        physics=False, device='cpu', cpus=[0], camera_recipe=r.name, capture_count=1,
        GR00T_updates=0, WM_updates=0, Actor_updates=0, WBC_changes=False,
        source_camera_identity=str(identity_path), source_camera_identity_sha256=digest(identity_path),
        frame_map=dict(T_source_from_target=r.T_source_from_target.tolist(), sha256=q.FRAME_MAP_SHA256),
        saved_native=dict(execution_spec=original_spec, episode_manifest=episode, native_result=native_result,
            worker_exit=worker, lease=lease, scene=scene, journal=dict(path=str(journal_path), sha256=digest(journal_path)),
            native_step=3703, expected_native_ticks=3703, packet_sha256=hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(q, 'local_path', lambda spec, path: Path(path))
    monkeypatch.setattr(q, 'software_renderer_identity', lambda: dict(GL_VENDOR='Mesa TEST DOUBLE',
        GL_RENDERER='llvmpipe TEST DOUBLE', backend='egl_mesa_cpu'))
    return spec, output


def test_one_sealed_actual_packet_is_reconstructed_without_integration_or_queries(tmp_path, monkeypatch):
    spec, output = source_fixture(tmp_path, monkeypatch)
    forbid_step(monkeypatch)
    result = q.run(spec, output)
    assert result['status'] == 'SINGLE_SEALED_TARGET_SOURCE_RIG_CPU_RENDER_PASS'
    assert result['actual_snapshot_render_saves'] == result['bound_observations'] == 1
    assert result['actual_python_forward_calls'] == 3
    assert result['native_integrations'] == result['VLA_inferences'] == result['prohibited_integration_calls'] == 0
    assert len(result['retained_files']) == 8
    assert all(digest(output/name) == pin for name, pin in result['retained_files'].items())
    assert result['online_closed_loop'] is False and result['native_execution_authorized'] is False
    with pytest.raises(FileExistsError): q.run(spec, output)


@pytest.mark.parametrize('fault', ['native_fail', 'open_old_lease', 'worker_fail', 'trace_drop', 'tick', 'journal', 'packet', 'scene', 'wrong_member'])
def test_saved_native_identity_or_completion_failures_reject_before_render(tmp_path, monkeypatch, fault):
    spec, output = source_fixture(tmp_path, monkeypatch)
    saved = spec['saved_native']
    if fault in ('native_fail', 'trace_drop'):
        native = json.loads(Path(saved['native_result']['path']).read_text())
        if fault == 'native_fail': native['status'] = 'FAILED'
        else: native['trace_audit']['writer']['dropped'] = 1
        saved['native_result'] = write(Path(saved['native_result']['path']), native)
    if fault == 'open_old_lease':
        value = json.loads(Path(saved['lease']['path']).read_text()); value['status'] = 'ACTIVE'
        saved['lease'] = write(Path(saved['lease']['path']), value)
    if fault == 'worker_fail': saved['worker_exit'] = write(Path(saved['worker_exit']['path']), dict(exit_code=1, failure='x'))
    if fault == 'tick': saved['native_step'] = 3702
    if fault == 'journal': Path(saved['journal']['path']).write_bytes(b'changed')
    if fault == 'packet': saved['packet_sha256'] = '0'*64
    if fault == 'scene': saved['scene']['sha256'] = '0'*64
    if fault == 'wrong_member':
        wrong = output/'copied-worker.json'; wrong.write_bytes(Path(saved['worker_exit']['path']).read_bytes())
        saved['worker_exit'] = dict(path=str(wrong), sha256=digest(wrong))
    with pytest.raises(ValueError): q.validate_inputs(spec)
    assert not (output/'images').exists()


@pytest.mark.parametrize('fault', ['identity', 'camera', 'updates', 'physics', 'gpu', 'wall', 'cpus', 'capture_count'])
def test_exact_cpu_scope_remains_existing_owner_and_no_training(tmp_path, monkeypatch, fault):
    spec, _ = source_fixture(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(fusion_scope, 'validate', lambda value: called.append(value) or {'existing_owner': True})
    assert fusion_scope.validate_source_camera_cpu(spec) == {'existing_owner': True}
    assert len(called) == 1
    if fault == 'identity': spec['source_camera_identity_sha256'] = '0'*64
    if fault == 'camera': spec['camera_recipe'] = 'ORIGINAL_TARGET_NATIVE_DUAL_CAMERA_V1'
    if fault == 'updates': spec['WM_updates'] = 1
    if fault == 'physics': spec['physics'] = True
    if fault == 'gpu': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    if fault == 'wall': spec['wall_limit_s'] = 181
    if fault == 'cpus': spec['cpus'] = [0, 0]
    if fault == 'capture_count': spec['capture_count'] = 3
    with pytest.raises(PermissionError): fusion_scope.validate_source_camera_cpu(spec)
    assert len(called) == 1


def test_misspelled_cpu_site_cannot_fall_through_generic_runtime(tmp_path):
    root = tmp_path/'job'; root.mkdir()
    source_root = Path(runtime.__file__).resolve().parents[3]
    spec = dict(source_root=str(source_root), registered_root=str(root), profile=q.PROFILE,
        kind=q.KIND, execution_site='MISSPELLED', host=os.uname().nodename, uid=os.getuid(),
        deadline_epoch=time.time()+180, physics=False)
    path = root/'EXECUTION_SPEC.json'; write(path, spec)
    with pytest.raises(PermissionError, match='EXACT_SOURCE_CAMERA_CPU_SITE'):
        with runtime.admitted_run(path, 'fit-01', q.PROFILE, root/'fit-01'):
            raise AssertionError('must reject before workload')
    assert not (root/'fit-01').exists()
