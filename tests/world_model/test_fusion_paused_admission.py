"""Synthetic admission and original worker routing; no CUDA, model or native run."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from go2arx5_vla.world_model import fusion_scope
from go2arx5_vla.world_model.storage import digest


GPU = 'GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
GIB = 1024**3


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _admission_spec(tmp_path, monkeypatch):
    # Import lazily so the fixture and production modules can be collected
    # independently while keeping one real synthetic capture implementation.
    from tests.world_model.test_fusion_paused_inference import make_spec
    spec, _, _, _ = make_spec(tmp_path, monkeypatch)
    spec = copy.deepcopy(spec)
    calibration = tmp_path/'calibration/RESULT.private.json'
    _put(calibration, dict(schema='WM02_FUSION_FIRST_ACTUAL_CALIBRATION_GROUP_V1',
        exit_code=0, actual_method_evaluations=4, actual_native_ticks=5*4103))
    _put(calibration.parent/'control/LEASE.json', dict(status='CLOSED'))
    spec.update(GPU_uuid=GPU, allocator_limit_bytes=13*GIB,
        checkpoint_parameter_bytes=12576064000, calibration_result=str(calibration),
        calibration_result_sha256=digest(calibration))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', GPU)
    scope = Mock(return_value='ORIGINAL_OWNER_SCOPE')
    monkeypatch.setattr(fusion_scope, 'validate', scope)
    return spec, dict(uuid=GPU, free_bytes=17*GIB), scope


def test_live_gpu_identity_keeps_original_scope_and_completed_calibration(tmp_path, monkeypatch):
    spec, actual, scope = _admission_spec(tmp_path, monkeypatch)
    fusion_scope.validate_gpu(spec, actual)
    scope.assert_called_once_with(spec)
    scope.side_effect = PermissionError('ORIGINAL_SCOPE_CLOSED')
    with pytest.raises(PermissionError, match='ORIGINAL_SCOPE_CLOSED'):
        fusion_scope.validate_gpu(spec, actual)


@pytest.mark.parametrize('key,value', [
    ('GPU_uuid', 'GPU-unregistered'),
    ('allocator_limit_bytes', 0),
    ('allocator_limit_bytes', 16*GIB),  # Equality leaves no required 1 GiB margin.
    ('allocator_limit_bytes', 12576064000-1),
    ('checkpoint_parameter_bytes', 0),
    ('checkpoint_parameter_bytes', True),
    ('numeric_mode', 'SILENT_BF16_WEIGHTS'),
    ('max_queries', 2),
    ('max_queries', True),
    ('physics', True),
    ('device', 'cpu'),
    ('GR00T_updates', 1),
    ('WM_updates', 1),
    ('Actor_updates', 1),
    ('profile', 'wm02-pilot1-neural'),
    ('execution_site', 'CORE_PILOT1_NEURAL'),
    ('inference_mode', 'OFFLINE_REAL_VLA_INFERENCE'),
    ('observation_mode', 'SAVED_TARGET_NATIVE_REPLAY_SHADOW'),
])
def test_live_gpu_rejects_changed_identity_before_scope(tmp_path, monkeypatch, key, value):
    spec, actual, scope = _admission_spec(tmp_path, monkeypatch)
    spec[key] = value
    with pytest.raises((PermissionError, ValueError)):
        fusion_scope.validate_gpu(spec, actual)
    scope.assert_not_called()


@pytest.mark.parametrize('fault', ['actual_uuid', 'cuda_other', 'cuda_unset', 'headroom'])
def test_live_gpu_uses_actual_device_environment_and_headroom(tmp_path, monkeypatch, fault):
    spec, actual, scope = _admission_spec(tmp_path, monkeypatch)
    if fault == 'actual_uuid':
        actual['uuid'] = 'GPU-845348f1-1511-dc19-0877-cc97658e3edd'
    elif fault == 'cuda_other':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    elif fault == 'cuda_unset':
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    else:
        actual['free_bytes'] = spec['allocator_limit_bytes']+GIB
    with pytest.raises(PermissionError):
        fusion_scope.validate_gpu(spec, actual)
    scope.assert_not_called()


@pytest.mark.parametrize('fault', ['wrong_hash', 'missing_member'])
def test_live_gpu_checks_source_bytes_before_scope(tmp_path, monkeypatch, fault):
    spec, actual, scope = _admission_spec(tmp_path, monkeypatch)
    from go2arx5_vla.world_model.fusion_paused_inference import SOURCE_MEMBERS
    member = SOURCE_MEMBERS[0]
    if fault == 'wrong_hash':
        spec['source_files'][member] = '0'*64
    else:
        del spec['source_files'][member]
    with pytest.raises(ValueError, match='ORIGINAL_INFERENCE_PRIMITIVE_BYTES'):
        fusion_scope.validate_gpu(spec, actual)
    scope.assert_not_called()


@pytest.mark.parametrize('fault', ['open', 'bad_hash', 'failed', 'incomplete', 'wrong_ticks'])
def test_live_gpu_needs_real_closed_full_four_arm_calibration(tmp_path, monkeypatch, fault):
    spec, actual, scope = _admission_spec(tmp_path, monkeypatch)
    path = Path(spec['calibration_result'])
    result = json.loads(path.read_text())
    if fault == 'open':
        _put(path.parent/'control/LEASE.json', dict(status='ACTIVE'))
    elif fault == 'bad_hash':
        spec['calibration_result_sha256'] = '0'*64
    else:
        key, value = {'failed': ('exit_code', 1),
                      'incomplete': ('actual_method_evaluations', 3),
                      'wrong_ticks': ('actual_native_ticks', 4103)}[fault]
        result[key] = value
        _put(path, result)
        spec['calibration_result_sha256'] = digest(path)
    with pytest.raises(PermissionError, match='ACTUAL_FIRST_NATIVE_CALIBRATION'):
        fusion_scope.validate_gpu(spec, actual)
    scope.assert_called_once_with(spec)


def _worker_spec(tmp_path, kind):
    root = tmp_path/'registered-job'
    spec_path = root/'EXECUTION_SPEC.json'
    lease_path = root/'control/LEASE.json'
    spec = dict(kind=kind, registered_root=str(root), lease=str(lease_path),
                source_sha='a'*40, wall_limit_s=120, deadline_epoch=time.time()+600)
    _put(spec_path, spec)
    _put(lease_path, dict(status='ACTIVE', run_uuid=root.name,
                         spec_sha256=digest(spec_path)))
    return spec_path, spec, root, lease_path


@pytest.mark.parametrize('kind,module', [
    ('FROZEN_VLA_LIVE_PAUSED_QUERY', 'scripts.wm02.infer_fusion_paused_vla'),
    ('FROZEN_VLA_SHADOW', 'scripts.wm02.infer_fusion_vla'),
    ('FUSION_TRAJECTORY_RESPONSE_OFFLINE', 'scripts.wm02.fit_fusion_trajectory_response'),
])
def test_original_worker_routes_new_gpu_old_shadow_and_new_cpu_and_closes_lease(
        tmp_path, monkeypatch, kind, module):
    from scripts.wm02 import pilot1_neural_worker as worker
    spec_path, spec, root, lease_path = _worker_spec(tmp_path, kind)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert (root/'SUPERVISOR.private.json').exists()
        assert kwargs['stdout'].name == str(root/'TRAIN_STDOUT.private.log')
        assert kwargs['stderr'].name == str(root/'TRAIN_STDERR.private.log')
        assert 0 < kwargs['timeout'] <= spec['wall_limit_s']
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, 'run', run)
    assert worker.main(['--spec', str(spec_path)]) == 0
    assert len(calls) == 1
    argv, _ = calls[0]
    assert argv == [worker.sys.executable, '-B', '-m', module,
                    '--spec', str(spec_path), '--output', str(root/'fit-01')]
    lease = json.loads(lease_path.read_text())
    assert lease['status'] == 'CLOSED' and lease['worker_exit_code'] == 0
    assert lease['run_uuid'] == root.name and lease['spec_sha256'] == digest(spec_path)
    exit_record = json.loads((root/'WORKER_EXIT.private.json').read_text())
    assert exit_record['exit_code'] == 0 and exit_record['failure'] is None
    assert exit_record['argv'] == argv
    assert exit_record['wall_s'] >= 0 and exit_record['child_CPU_s'] >= -1e-12
    with pytest.raises(PermissionError, match='EXACT_FRESH_PILOT1_WORKER'):
        worker.main(['--spec', str(spec_path)])
    assert len(calls) == 1


@pytest.mark.parametrize('fault,expected_code', [('child_failure', 23), ('timeout', 1)])
def test_new_query_worker_preserves_failure_and_closes_finite_lease(
        tmp_path, monkeypatch, fault, expected_code):
    from scripts.wm02 import pilot1_neural_worker as worker
    spec_path, _, root, lease_path = _worker_spec(tmp_path, 'FROZEN_VLA_LIVE_PAUSED_QUERY')
    def run(argv, **kwargs):
        if fault == 'timeout':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return SimpleNamespace(returncode=23)
    monkeypatch.setattr(worker.subprocess, 'run', run)
    assert worker.main(['--spec', str(spec_path)]) == expected_code
    lease = json.loads(lease_path.read_text())
    exit_record = json.loads((root/'WORKER_EXIT.private.json').read_text())
    assert lease['status'] == 'CLOSED' and lease['worker_exit_code'] == expected_code
    assert exit_record['exit_code'] == expected_code
    if fault == 'timeout':
        assert exit_record['failure'].startswith('TimeoutExpired:')
    else:
        assert exit_record['failure'] is None


def test_new_query_worker_rejects_unknown_prior_supervisor_without_dispatch(tmp_path, monkeypatch):
    from scripts.wm02 import pilot1_neural_worker as worker
    spec_path, _, root, lease_path = _worker_spec(tmp_path, 'FROZEN_VLA_LIVE_PAUSED_QUERY')
    _put(root/'SUPERVISOR.private.json', dict(pid=123, source_sha='a'*40))
    run = Mock()
    monkeypatch.setattr(worker.subprocess, 'run', run)
    with pytest.raises(FileExistsError, match='DO_NOT_REDISPATCH'):
        worker.main(['--spec', str(spec_path)])
    run.assert_not_called()
    assert json.loads(lease_path.read_text())['status'] == 'ACTIVE'
    assert not (root/'WORKER_EXIT.private.json').exists()


def test_new_query_main_enters_original_gpu_admission_before_inference(tmp_path, monkeypatch):
    from go2arx5_vla.world_model import runtime
    from scripts.wm02 import infer_fusion_paused_vla as entry
    spec_path, output = tmp_path/'EXECUTION_SPEC.json', tmp_path/'fit-01'
    spec, events = {'fixture': 'admitted-only'}, []
    @contextmanager
    def admitted(*args, **kwargs):
        assert args == (spec_path, output.name, 'wm02-fusion-vla-shadow', output)
        assert kwargs == {'gpu': True}
        events.append('admitted')
        yield spec
        events.append('closed')
    def infer(actual_spec, actual_output):
        assert actual_spec is spec and actual_output == output
        assert events == ['admitted']
        events.append('infer')
    monkeypatch.setattr(runtime, 'admitted_run', admitted)
    monkeypatch.setattr(entry, 'infer', infer)
    assert entry.main(['--spec', str(spec_path), '--output', str(output)]) == 0
    assert events == ['admitted', 'infer', 'closed']


def test_new_query_main_does_not_infer_after_admission_rejection(tmp_path, monkeypatch):
    from go2arx5_vla.world_model import runtime
    from scripts.wm02 import infer_fusion_paused_vla as entry
    @contextmanager
    def denied(*args, **kwargs):
        raise PermissionError('ORIGINAL_OWNER_LOCK_BUSY')
        yield  # pragma: no cover; preserve contextmanager protocol.
    infer = Mock()
    monkeypatch.setattr(runtime, 'admitted_run', denied)
    monkeypatch.setattr(entry, 'infer', infer)
    with pytest.raises(PermissionError, match='ORIGINAL_OWNER_LOCK_BUSY'):
        entry.main(['--spec', str(tmp_path/'spec.json'), '--output', str(tmp_path/'fit-01')])
    infer.assert_not_called()
