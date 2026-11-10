import json
import sys
from types import SimpleNamespace

import pytest
from go2arx5_vla.world_model import task_learning_reduction_diagnostic_v1 as diag
from go2arx5_vla.world_model import task_learning_core_inference_v1 as core
from go2arx5_vla.world_model.storage import digest


def spec(tmp_path):
    package = dict(path='private/package', sha256='a' * 64)
    p = tmp_path / 'reference.json'
    p.write_text(json.dumps(dict(status='ACTUAL_CORE_CONTINUOUS_TASK_SEALED_TRAIN40_INFERENCE',
        actual_VLA_queries=40, Processor_decode_calls=40,
        model_identity=dict(candidate_package=package))))
    return dict(profile=core.PROFILE, kind=core.KIND, target_inference_mode=core.MODE,
        physics=False, device='cuda', max_queries=40, GR00T_updates=0, WM_updates=0,
        Actor_updates=0, candidate_package=package,
        numerical_diagnostic=dict(mode=diag.MODE,
            reference_result=dict(path=str(p), sha256=digest(p))))


def test_ordinary_entry_does_not_import_or_configure_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with diag.execution({}, tmp_path):
        pass
    assert not (tmp_path / 'REDUCTION_DIAGNOSTIC_ACTUAL.private.json').exists()


@pytest.mark.parametrize('change', [dict(physics=True), dict(GR00T_updates=1),
    dict(max_queries=4), dict(kind='TASK_LEARNING_LIVE_QUERY_V1')])
def test_diagnostic_cannot_enable_training_native_execution_or_subset(tmp_path, change):
    s = spec(tmp_path); s.update(change)
    with pytest.raises(PermissionError, match='INFERENCE_ONLY_REDUCTION_DIAGNOSIS'):
        diag.validate(s)


def test_reference_must_bind_same_complete_package(tmp_path):
    s = spec(tmp_path); s['candidate_package'] = dict(path='other', sha256='b' * 64)
    with pytest.raises(PermissionError, match='SAME_PACKAGE_AND_COMPLETE_CORE40_REFERENCE'):
        diag.validate(s)


def test_actual_flag_is_restored_and_failure_preserved(tmp_path, monkeypatch):
    backend = SimpleNamespace(allow_bf16_reduced_precision_reduction=True, allow_tf32=False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=backend))))
    with pytest.raises(RuntimeError, match='preserved inference failure'):
        with diag.execution(spec(tmp_path), tmp_path):
            assert backend.allow_bf16_reduced_precision_reduction is False
            assert backend.allow_tf32 is False
            raise RuntimeError('preserved inference failure')
    assert backend.allow_bf16_reduced_precision_reduction is True
    result = json.loads((tmp_path / 'REDUCTION_DIAGNOSTIC_ACTUAL.private.json').read_text())
    assert result['original_flag_restored_on_exit'] is True
    assert result['failure'] == 'RuntimeError: preserved inference failure'
    assert result['package_qualified'] is False and result['tolerances_relaxed'] is False
