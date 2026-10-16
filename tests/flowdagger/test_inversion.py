from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from go2arx5_vla.flowdagger.codec import ModelSpace
from go2arx5_vla.flowdagger.inversion import CountedVelocity, forward_euler, round_trip
from go2arx5_vla.flowdagger.native import FlowContext, native_velocity, require_frozen
from go2arx5_vla.flowdagger.steering import SteeringMLP, observation_features, latent_loss


def test_same_discrete_grid_inverse_and_full_evaluation_accounting():
    target=torch.ones(1, 40, 12)
    velocity=CountedVelocity(lambda x,t: .2*x+t, limit=48)
    result=round_trip(target,velocity,steps=4,fp_per_step=10,active_dim=10)
    torch.testing.assert_close(result['reconstructed'],target,atol=1e-6,rtol=0)
    assert velocity.calls == result['velocity_evaluations'] == result['expected_velocity_evaluations'] == 48
    assert not result['convergence_qualified']
    assert target.eq(1).all()
    with pytest.raises(PermissionError):forward_euler(target,velocity,steps=4)


def test_failed_velocity_is_charged_no_silent_resample():
    velocity=CountedVelocity(lambda x,t: x*float('nan'),limit=1)
    with pytest.raises(ValueError):forward_euler(torch.ones(1,40,12),velocity,steps=4)
    assert velocity.calls == 1


def test_finite_divergent_inverse_is_not_qualified():
    velocity=CountedVelocity(lambda x,t: x*12,limit=48)
    result=round_trip(torch.ones(1,40,12),velocity,steps=4,fp_per_step=10,active_dim=10)
    assert result['active_normalized_MSE'] > 1
    assert not result['convergence_qualified']


@pytest.mark.parametrize('fp', [-1, True, 101])
def test_fixed_point_work_is_explicitly_bounded(fp):
    velocity=CountedVelocity(lambda x,t: x,limit=1000)
    with pytest.raises(ValueError):round_trip(torch.ones(1,40,12),velocity,steps=4,fp_per_step=fp,active_dim=10)
    assert velocity.calls == 0


class ToyHead(torch.nn.Module):
    def __init__(self,alternate,pos_embed=False):
        super().__init__()
        self.config=SimpleNamespace(add_pos_embed=pos_embed,use_alternate_vl_dit=alternate)
        self.action_dim,self.action_horizon,self.num_inference_timesteps,self.num_timestep_buckets=12,40,4,1000
        self.p=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.last_arguments=None
        self.eval()

    def action_encoder(self,x,t,emb):return x
    def action_decoder(self,x,emb):return x
    def position_embedding(self,ids):return torch.ones(len(ids),12)
    def model(self,**kwargs):
        self.last_arguments=kwargs
        return kwargs['hidden_states']*.2+kwargs['timestep'][:,None,None]/1000


@pytest.mark.parametrize('alternate',[False,True])
@pytest.mark.parametrize('pos_embed',[False,True])
def test_native_velocity_masks_grid_and_shape(alternate,pos_embed):
    head=ToyHead(alternate,pos_embed)
    context=FlowContext(torch.zeros(1,2,12),torch.zeros(1,1,12),torch.zeros(1,dtype=torch.long),
        SimpleNamespace(image_mask='image',backbone_attention_mask='attention'),'obs-1')
    velocity=native_velocity(head,context,ModelSpace(40,12,4),limit=1)
    torch.testing.assert_close(velocity(torch.ones(1,40,12),.25),torch.full((1,40,12),.65 if pos_embed else .45))
    assert ('image_mask' in head.last_arguments) == alternate
    if alternate:assert head.last_arguments['backbone_attention_mask']=='attention'
    head.train()
    with pytest.raises(ValueError):require_frozen(head)
    head.eval();head.p.requires_grad_(True)
    with pytest.raises(ValueError):native_velocity(head,context,ModelSpace(40,12,4),limit=1)


def test_steering_only_gradient_no_target_or_base_features_gradient():
    bb=torch.ones(1,2,3,requires_grad=True);state=torch.ones(1,1,2,requires_grad=True)
    context=FlowContext(bb,state,None,None,'obs-1')
    features=observation_features(context)
    steering=SteeringMLP(5,ModelSpace(40,12,4),hidden_dims=(4,))
    target=torch.zeros(1,40,12,requires_grad=True)
    loss=latent_loss(steering(features),target);loss.backward()
    assert all(p.grad is not None for p in steering.parameters())
    assert bb.grad is None and state.grad is None and target.grad is None
