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
    m=Tiny();next(m.backbone.parameters()).grad=torch.zeros_like(next(m.backbone.parameters()))
    with pytest.raises(PermissionError,match='ONLY_FROZEN'):BackboneOptimizerSwap(m,None,device='cpu')


def test_callback_restores_before_original_input_preparation_not_only_backbone_forward():
    m=Tiny();swap=BackboneOptimizerSwap(m,None,device='cpu');callback=swap.callback()
    callback.on_pre_optimizer_step(None,None,None)
    assert not swap.resident
    callback.on_step_end(None,None,None)
    # Both model.forward and method-based get_action prepare inputs before
    # calling the backbone. They must observe resident weights at entry.
    assert swap.resident and swap.to_device_count==1
    swap.before_optimizer();callback.on_step_begin(None,None,None)
    assert swap.resident and swap.to_device_count==2


def test_swap_policy_only_admits_the_explicit_core_full_native_restore():
    p=profile.recipe('a'*64,'b'*64,mode='RESTORE4',variant='FULL_ACTION_SIDE',endpoint=4,
        memory_policy='CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3')
    from test_target_gr00t_core_v1 import spec
    s=dict(spec('RESTORE4','FULL_ACTION_SIDE',4),recipe=p,core_native_operation='FULL_ACTION_RESTORE_ONLY')
    assert profile.validate(s)==p
    with pytest.raises(PermissionError,match='ORIGINAL_RESIDENT'):profile.validate(dict(s,execution_site='HOPPER_SINGLE_GPU'))
    with pytest.raises(PermissionError,match='CORE_FIRST'):profile.validate(dict(s,core_native_operation=None))
    assert core.full_restore_only(dict(operation='FULL_ACTION_RESTORE_ONLY',original_Trainer_execution=True,
        formal_training=False,memory_policy=p['memory_policy'],memory_reference=dict(path='/sealed/reference',sha256='a'*64)))
