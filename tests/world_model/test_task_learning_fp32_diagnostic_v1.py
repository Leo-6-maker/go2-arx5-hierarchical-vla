"""Explicit precision intervention, cleanup and no-admission fixtures."""
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import task_learning_fp32_diagnostic_v1 as fp32
from test_task_learning_tensor_trace_v1 import spec


def configured(tmp_path):
    s=spec(tmp_path);s['inference_precision_diagnostic']={'mode':fp32.MODE};return s


def test_default_path_does_not_import_torch(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',None)
    with fp32.precision({},tmp_path,None):pass
    assert not (tmp_path/'FP32_DIAGNOSTIC_STARTED.private.json').exists()


@pytest.mark.parametrize('change',[dict(diagnostic_only=False),dict(physics=True),
    dict(max_queries=4),dict(GR00T_updates=1),dict(backbone_trace={'mode':'other'}),
    dict(numerical_diagnostic={'mode':'other'})])
def test_precision_intervention_cannot_change_regime_or_combine_ablations(tmp_path,change):
    s=configured(tmp_path);s.update(change)
    with pytest.raises(PermissionError):fp32.validate(s)


def fake(monkeypatch):
    state=SimpleNamespace(autocast=True)
    cuda=SimpleNamespace(allow_tf32=True)
    cudnn=SimpleNamespace(allow_tf32=True)
    @contextmanager
    def autocast(device,*,enabled):
        before=state.autocast;state.autocast=enabled
        try:yield
        finally:state.autocast=before
    torch=SimpleNamespace(float32='torch.float32',autocast=autocast,
        is_autocast_enabled=lambda device:state.autocast,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=cuda),cudnn=cudnn))
    class Model:
        def parameters(self):return [SimpleNamespace(dtype='torch.float32')]
        def get_action(self,fail=False):
            assert not state.autocast and not cuda.allow_tf32 and not cudnn.allow_tf32
            if fail:raise RuntimeError('original forward failure')
            return dict(action_pred=SimpleNamespace(dtype='torch.float32'))
    monkeypatch.setitem(sys.modules,'torch',torch)
    return Model(),state,cuda,cudnn


def test_all40_original_calls_and_original_method_flags_restored(tmp_path,monkeypatch):
    model,state,cuda,cudnn=fake(monkeypatch);original=model.get_action
    # Match the outer original BF16 context; this diagnostic overrides only get_action.
    with fp32.precision(configured(tmp_path),tmp_path,model):
        for _ in range(40):model.get_action()
    assert model.get_action==original and 'get_action'not in model.__dict__
    assert state.autocast and cuda.allow_tf32 and cudnn.allow_tf32
    x=json.loads((tmp_path/'FP32_DIAGNOSTIC_ACTUAL.private.json').read_text())
    assert x['original_queries']==40 and x['original_flags_restored']and x['original_method_restored']
    assert x['precision_changed']and not x['weights_changed']and not x['package_qualified']


def test_original_failure_preserved_and_precision_and_method_restored(tmp_path,monkeypatch):
    model,state,cuda,cudnn=fake(monkeypatch);original=model.get_action
    with pytest.raises(RuntimeError,match='original forward failure'):
        with fp32.precision(configured(tmp_path),tmp_path,model):model.get_action(fail=True)
    assert model.get_action==original and state.autocast and cuda.allow_tf32 and cudnn.allow_tf32
    x=json.loads((tmp_path/'FP32_DIAGNOSTIC_ACTUAL.private.json').read_text())
    assert x['original_queries']==0 and 'original forward failure'in x['failure']
