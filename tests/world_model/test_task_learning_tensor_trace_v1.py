"""Trace serialization/lifecycle fixtures are not GPU or policy qualification."""
from collections import UserDict
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_tensor_trace_v1 as trace
from go2arx5_vla.world_model.storage import digest


def spec(tmp_path):
    package = dict(path='private/package', sha256='a' * 64)
    p = tmp_path / 'reference.json'
    p.write_text(json.dumps(dict(actual_VLA_queries=40, Processor_decode_calls=40,
        model_identity=dict(candidate_package=package))))
    return dict(profile='wm02-task-learning-package-core-v1',
        kind='TASK_LEARNING_PACKAGE_CORE_RECONSTRUCTION_V1', execution_site='CORE_FUSION_NEURAL',
        device='cuda', physics=False, max_queries=40, diagnostic_only=True,
        GR00T_updates=0, WM_updates=0, Actor_updates=0, candidate_package=package,
        tensor_trace=dict(mode=trace.MODE, reference_result=dict(path=str(p), sha256=digest(p))))


def test_regular_entry_does_not_import_torch_or_touch_model(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with trace.trace({}, tmp_path, None): pass
    assert not (tmp_path / 'intermediate-trace').exists()


@pytest.mark.parametrize('change', [dict(physics=True), dict(GR00T_updates=1),
    dict(max_queries=4), dict(diagnostic_only=False), dict(execution_site='UNKNOWN'),
    dict(numerical_diagnostic={'mode':'other'})])
def test_trace_rejects_other_execution_regimes(tmp_path, change):
    s = spec(tmp_path); s.update(change)
    with pytest.raises(PermissionError, match='TRACE_ONLY_COMPLETE40'):
        trace.validate(s)


def test_paired_sites_share_package_hash_without_sharing_host_paths(tmp_path):
    s = spec(tmp_path)
    s['candidate_package']['path'] = 'other-site/package'
    assert trace.validate(s)['mode'] == trace.MODE
    s['candidate_package']['sha256'] = 'b' * 64
    with pytest.raises(PermissionError, match='SAME_ACTUAL40_PACKAGE_TRACE_REFERENCE'):
        trace.validate(s)


class Tensor:
    def __init__(self, value):
        self.array = np.asarray(value, dtype=np.float32)
        self.dtype = 'float32'; self.shape = self.array.shape
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class Module:
    def __init__(self): self.pre = []; self.post = []
    def register_forward_pre_hook(self, callback, **kwargs):
        self.pre.append(callback)
        return SimpleNamespace(remove=lambda:self.pre.remove(callback))
    def register_forward_hook(self, callback):
        self.post.append(callback)
        return SimpleNamespace(remove=lambda:self.post.remove(callback))
    def __call__(self, value):
        for callback in self.pre: callback(self, (value,), {})
        for callback in self.post: callback(self, (value,), value)
        return value


def model_and_torch(monkeypatch):
    names = ['vlln', 'state_encoder', 'action_encoder', 'model', 'action_decoder', 'vl_self_attention']
    head = SimpleNamespace(**{n:Module() for n in names})
    model = SimpleNamespace(backbone=Module(), action_head=head)
    backend = SimpleNamespace(allow_bf16_reduced_precision_reduction=True, allow_tf32=False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor),
        bfloat16='bfloat16',backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend))))
    return model, backend


def test_complete_first4_trace_keeps_all40_calls_and_copies_mapping_tensors(tmp_path, monkeypatch):
    model, backend = model_and_torch(monkeypatch)
    tensor = Tensor([[1., 2.]]); original = tensor.array.copy(); h = model.action_head
    with trace.trace(spec(tmp_path), tmp_path, model):
        for query in range(40):
            # BatchFeature/ModelOutput can be UserDict rather than dict.
            assert model.backbone(UserDict(pixel_values=tensor))['pixel_values'] is tensor
            for module in (h.vlln, h.vl_self_attention, h.state_encoder): assert module(tensor) is tensor
            for step in range(4):
                for module in (h.action_encoder, h.model, h.action_decoder): assert module(tensor) is tensor
    assert np.array_equal(tensor.array, original)
    result = json.loads((tmp_path/'intermediate-trace/TRACE_MANIFEST.private.json').read_text())
    assert result['complete'] is True and result['original_query_count'] == 40
    assert result['traced_query_count'] == 4 and result['numeric_flags_changed'] is False
    assert {r['query_index'] for r in result['records']} == {0,1,2,3}
    assert backend.allow_tf32 is False and backend.allow_bf16_reduced_precision_reduction is True
    assert all(not m.pre and not m.post for m in [model.backbone,*vars(h).values()])
    row = next(r for r in result['records'] if r['module']=='backbone' and r['stage']=='output')
    with np.load(tmp_path/'intermediate-trace'/row['file'], allow_pickle=False) as z:
        assert np.array_equal(z['tensor0000'], original)


def test_failed_inference_preserves_partial_trace_and_removes_hooks(tmp_path, monkeypatch):
    model, backend = model_and_torch(monkeypatch)
    with pytest.raises(RuntimeError,match='original failure'):
        with trace.trace(spec(tmp_path), tmp_path, model):
            model.backbone(Tensor([1.]))
            raise RuntimeError('original failure')
    result = json.loads((tmp_path/'intermediate-trace/TRACE_MANIFEST.private.json').read_text())
    assert result['complete'] is False and result['failure']=='RuntimeError: original failure'
    assert not model.backbone.pre and not model.backbone.post
    assert backend.allow_bf16_reduced_precision_reduction is True
