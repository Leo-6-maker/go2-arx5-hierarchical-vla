"""CPU storage seam checks; actual CUDA/restore qualification is separate."""
import copy
from types import SimpleNamespace
import pytest
import torch
from go2arx5_vla.world_model.target_gr00t_backbone_swap_v1 import BackboneOptimizerSwap
from go2arx5_vla.world_model import target_gr00t_core_v1 as core,target_gr00t_profile_v1 as profile


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(3,4);self.action_head=torch.nn.Linear(4,2)
        self.backbone.requires_grad_(False)
    def forward(self,x):return self.action_head(self.backbone(x))


def test_storage_callback_preserves_original_forward_gradients_rng_and_adamw():
    torch.manual_seed(3);a=Tiny();b=copy.deepcopy(a)
    storage=SimpleNamespace(excluded=set(),storage=lambda t:(str(t.device),t.untyped_storage().data_ptr()))
    swap=BackboneOptimizerSwap(b,storage,device='cpu')
    opta=torch.optim.AdamW(a.action_head.parameters(),lr=1e-3,foreach=True)
    optb=torch.optim.AdamW(b.action_head.parameters(),lr=1e-3,foreach=True)
    x=torch.randn(2,3);target=torch.randn(2,2)
    for _ in range(4):
        opta.zero_grad();optb.zero_grad()
        ya=a(x);yb=b(x);assert torch.equal(ya,yb)
        ((ya-target)**2).mean().backward();((yb-target)**2).mean().backward()
        assert all(torch.equal(pa.grad,pb.grad)for pa,pb in zip(a.action_head.parameters(),b.action_head.parameters()))
        rng=torch.get_rng_state().clone();head=[(id(t),t.data_ptr())for t in b.action_head.parameters()]
        swap.before_optimizer()
        assert torch.equal(torch.get_rng_state(),rng)
        assert head==[(id(t),t.data_ptr())for t in b.action_head.parameters()]
        opta.step();optb.step()
        assert all(torch.equal(va,vb)for va,vb in zip(a.state_dict().values(),b.state_dict().values()))
        for sa,sb in zip(opta.state.values(),optb.state.values()):
            assert all(torch.equal(sa[k],sb[k])for k in sa)
    b(x)
    assert swap.to_host_count==swap.to_device_count==4
    assert storage.excluded=={storage.storage(t)for t in b.parameters()}
    assert not swap.audit()['optimizer_changed']and not swap.audit()['feature_cache']


def test_trainable_or_gradient_bearing_backbone_cannot_be_swapped():
    m=Tiny();next(m.backbone.parameters()).requires_grad_(True)
    with pytest.raises(PermissionError,match='ONLY_FROZEN'):BackboneOptimizerSwap(m,None,device='cpu')


def test_optimizer_reload_storage_restores_original_device_even_on_failure():
    model=Tiny();swap=BackboneOptimizerSwap(model,None,device='cpu')
    head=[(id(p),p.data_ptr(),p.detach().clone())for p in model.action_head.parameters()]
    rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='restore failed'):
        with swap.optimizer_restore_storage():
            assert not swap.resident
            raise RuntimeError('restore failed')
    assert swap.resident and swap.audit()['optimizer_state_restores']==1
    assert swap.to_host_count==swap.to_device_count==1
    assert torch.equal(rng,torch.get_rng_state())
    assert all(id(p)==i and p.data_ptr()==ptr and torch.equal(p,v)
        for p,(i,ptr,v)in zip(model.action_head.parameters(),head))
    m=Tiny();next(m.backbone.parameters()).grad=torch.zeros_like(next(m.backbone.parameters()))
    with pytest.raises(PermissionError,match='ONLY_FROZEN'):BackboneOptimizerSwap(m,None,device='cpu')


def test_callback_restores_before_original_input_preparation_not_only_backbone_forward(monkeypatch):
    import sys
    # The CPU seam environment intentionally has no optional HF runtime.
    # Real Core uses its actual TrainerCallback, exercised by the native job.
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(TrainerCallback=object))
    m=Tiny();swap=BackboneOptimizerSwap(m,None,device='cpu');callback=swap.callback()
    callback.on_pre_optimizer_step(None,None,None)
    assert not swap.resident
    callback.on_step_end(None,None,None)
    # Both model.forward and method-based get_action prepare inputs before
    # calling the backbone. They must observe resident weights at entry.
    assert swap.resident and swap.to_device_count==1
    swap.before_optimizer();callback.on_step_begin(None,None,None)
    assert swap.resident and swap.to_device_count==2


@pytest.mark.parametrize('memory',['CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3','CORE_FROZEN_BACKBONE_SWAP_ACTIVATIONS_4G_V4','CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5'])
def test_swap_policy_only_admits_the_explicit_core_full_native_restore(memory):
    p=profile.recipe('a'*64,'b'*64,mode='RESTORE4',variant='FULL_ACTION_SIDE',endpoint=4,
        memory_policy=memory)
    from test_target_gr00t_core_v1 import spec
    s=dict(spec('RESTORE4','FULL_ACTION_SIDE',4),recipe=p,core_native_operation='FULL_ACTION_RESTORE_ONLY')
    assert profile.validate(s)==p
    with pytest.raises(PermissionError,match='ORIGINAL_RESIDENT'):profile.validate(dict(s,execution_site='HOPPER_SINGLE_GPU'))
    with pytest.raises(PermissionError,match='CORE_FIRST'):profile.validate(dict(s,core_native_operation=None))
    assert core.full_restore_only(dict(operation='FULL_ACTION_RESTORE_ONLY',original_Trainer_execution=True,
        formal_training=False,memory_policy=p['memory_policy'],memory_reference=dict(path='/sealed/reference',sha256='a'*64)))


def test_larger_activation_budget_preserves_lossless_storage_and_old_default():
    old,new=core.activation_policy(),core.activation_policy(4*1024**3)
    assert old['per_microbatch_activation_offload_bytes']==2*1024**3
    assert new['per_microbatch_activation_offload_bytes']==4*1024**3
    assert not new['quantization']and not new['parameter_storage_offload']
    assert {k:v for k,v in old.items()if k not in ['schema','per_microbatch_activation_offload_bytes']}=={
        k:v for k,v in new.items()if k not in ['schema','per_microbatch_activation_offload_bytes']}


def test_disabling_cast_weight_cache_preserves_bf16_loss_gradients_and_rng_on_cpu():
    torch.manual_seed(17)
    model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(16,8))
    x=torch.randn(3,8);target=torch.randn(3,8);rows=[]
    previous=torch.is_autocast_cache_enabled()
    try:
        for enabled in [True,False]:
            torch.set_autocast_cache_enabled(enabled);model.zero_grad(set_to_none=True);torch.manual_seed(7)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                assert torch.is_autocast_cache_enabled()is enabled
                output=model(x);assert output.dtype==torch.bfloat16
                loss=(output.float()-target).square().mean()
            loss.backward()
            rows.append((loss.detach().clone(),[p.grad.clone()for p in model.parameters()],torch.get_rng_state().clone()))
        assert torch.equal(rows[0][0],rows[1][0])and torch.equal(rows[0][2],rows[1][2])
        assert all(torch.equal(a,b)for a,b in zip(rows[0][1],rows[1][1]))
    finally:torch.set_autocast_cache_enabled(previous)


def test_weight_cache_qualification_uses_positioned_sampler_in_separate_trace(tmp_path):
    from go2arx5_vla.world_model.target_gr00t_weight_cache_v1 import qualification_inputs
    seen=[]
    class Data:
        trace_root=tmp_path/'real-training-trace'
        processor=None
        def __getitem__(self,index):
            assert index==(0,0)and self.trace_root.is_dir()
            self.processor=SimpleNamespace(collator=lambda rows:{'inputs':rows[0]})
            seen.append((index,self.trace_root))
            return {'processed':'same-original-processed-input'}
    data=Data();original=data.trace_root
    assert qualification_inputs(data,tmp_path/'comparison')=='same-original-processed-input'
    assert data.trace_root==original
    assert data.processor is None
    assert seen==[((0,0),tmp_path/'comparison/weight-cache-qualification-sample-trace')]
