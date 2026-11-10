"""First-query capture custody tests; no GPU or model qualification."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_backbone_trace_v1 as inner
from test_task_learning_tensor_trace_v1 import Module, Tensor, spec


def test_no_internal_diagnostic_does_not_import_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with inner.trace({}, tmp_path, None): pass
    assert not (tmp_path/'backbone-internal-trace').exists()


@pytest.mark.parametrize('change', [dict(max_queries=1), dict(physics=True),
    dict(diagnostic_only=False), dict(GR00T_updates=1), dict(tensor_trace=None)])
def test_internal_trace_requires_full_original40_and_no_updates(tmp_path, change):
    s=spec(tmp_path);s['backbone_trace']={'mode':inner.MODE};s.update(change)
    with pytest.raises(PermissionError): inner.validate(s)


def setup(tmp_path, monkeypatch):
    s=spec(tmp_path);s['backbone_trace']={'mode':inner.MODE}
    backbone, leaf=Module(),Module()
    model=SimpleNamespace(backbone=backbone)
    flags=SimpleNamespace(allow_tf32=False,allow_bf16_reduced_precision_reduction=True)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor),
        bfloat16='bfloat16',backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flags),
        cudnn=SimpleNamespace(allow_tf32=True))))
    monkeypatch.setattr(inner,'selected_modules',lambda model:dict(backbone=backbone,leaf=leaf))
    monkeypatch.setattr(inner,'constants',lambda model:dict(buffer=Tensor([1.,2.])))
    monkeypatch.setattr(inner,'runtime_metadata',lambda model,torch:dict(
        trainable_backbone_parameters=0,trainable_visual_parameters=0,trainable_language_parameters=0,
        HF_Qwen_source_sha256='685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99'))
    return s,model,leaf


def test_only_first_query_saved_while_all40_execute_and_inputs_remain_unchanged(tmp_path, monkeypatch):
    s,model,leaf=setup(tmp_path,monkeypatch);value=Tensor([3.,4.]);original=value.array.copy()
    with inner.trace(s,tmp_path,model):
        for _ in range(40): model.backbone(value);leaf(value)
    manifest=json.loads((tmp_path/'backbone-internal-trace/TRACE_MANIFEST.private.json').read_text())
    assert manifest['complete'] and manifest['original_query_count']==40 and manifest['traced_query_count']==1
    assert len(manifest['records'])==5
    assert manifest['numeric_flags_changed'] is manifest['tensors_mutated'] is manifest['package_qualified'] is False
    assert np.array_equal(value.array,original) and not model.backbone.pre and not leaf.post


def test_partial_capture_retains_failure_and_removes_every_hook(tmp_path, monkeypatch):
    s,model,leaf=setup(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='original model failure'):
        with inner.trace(s,tmp_path,model):
            model.backbone(Tensor([1.]));raise RuntimeError('original model failure')
    manifest=json.loads((tmp_path/'backbone-internal-trace/TRACE_MANIFEST.private.json').read_text())
    assert manifest['complete'] is False and 'original model failure' in manifest['failure']
    assert not model.backbone.pre and not model.backbone.post and not leaf.pre and not leaf.post


def test_complete40_cannot_hide_a_missing_internal_boundary(tmp_path, monkeypatch):
    s,model,leaf=setup(tmp_path,monkeypatch)
    with pytest.raises(PermissionError,match='COMPLETE_FIRST_QUERY'):
        with inner.trace(s,tmp_path,model):
            for _ in range(40):model.backbone(Tensor([1.]))
    assert not leaf.pre and not leaf.post
