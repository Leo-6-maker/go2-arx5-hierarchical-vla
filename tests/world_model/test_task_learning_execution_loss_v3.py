import numpy as np
import torch

from go2arx5_vla.world_model.task_learning_execution_loss_v3 import (
    PhysicalDecoder,ExecutionLoss,acceleration_loss,rot6d_matrix)


def decoder():
    return PhysicalDecoder(dict(eef_9d=dict(min=[-.02,-.01,-.03,1,-.1,-.1,-.1,1,-.1],
        max=[.02,.01,.03,1,.1,.1,.1,1,.1]),gripper_position=dict(min=[0.],max=[1.])),device='cpu')


def state():
    # A quarter turn makes frame-mixing visible; measured grip is independent.
    return torch.tensor([[.3,.1,.7,0.,-1.,0.,1.,0.,0.,.43]],dtype=torch.float64)


def test_decode_se3_relative_xyz_and_absolute_nominal_grip():
    x=torch.zeros(1,40,132,dtype=torch.float64);x[...,0]=.5;x[...,9]=1.6
    y=decoder()(x,state())
    torch.testing.assert_close(y[0,0,:3],torch.tensor([.3,.11,.7],dtype=torch.float64))
    assert y[0,0,9]==1.
    torch.testing.assert_close(rot6d_matrix(y[...,3:9]),rot6d_matrix(state()[...,3:9])[:,None].expand(1,40,3,3))
    x[...,10:]=500
    torch.testing.assert_close(y,decoder()(x,state()),rtol=0,atol=0)


def test_auxiliary_gradient_finite_difference_no_padding_rotation_or_grip_gradient():
    x=torch.zeros(1,40,132,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[0,17,0]=.1
    expert=decoder().xyz(torch.zeros_like(x),state())
    def loss(v):return acceleration_loss(decoder().xyz(v,state()),expert)[0]
    actual=loss(x);gradient=torch.autograd.grad(actual,x)[0]
    epsilon=1e-6;plus=x.detach().clone();minus=plus.clone()
    plus[0,17,0]+=epsilon;minus[0,17,0]-=epsilon
    finite=(loss(plus)-loss(minus))/(2*epsilon)
    torch.testing.assert_close(gradient[0,17,0],finite,rtol=1e-7,atol=1e-9)
    assert gradient[...,:3].abs().sum()>0
    assert torch.count_nonzero(gradient[...,3:])==0


def test_clean_estimate_respects_noise_to_action_flow_direction():
    noise=torch.randn(1,40,132,dtype=torch.float64)
    action=torch.randn_like(noise);velocity=action-noise
    for t in (0.,.25,.8,1.):
        noisy=(1-t)*noise+t*action
        torch.testing.assert_close(noisy+(1-t)*velocity,action,rtol=1e-12,atol=1e-12)


def test_constant_velocity_expert_is_not_penalized_as_boundary_stop():
    xyz=torch.zeros(1,40,3,dtype=torch.float64)
    xyz[0,:,0]=torch.arange(40,dtype=torch.float64)*.001
    loss,terms=acceleration_loss(xyz,xyz)
    assert loss==0 and terms['acceleration_excess']==0


def test_lambda_zero_returns_original_tensor_and_scoped_hooks_preserve_rng():
    class Head:
        def __init__(self):
            self.action_encoder=torch.nn.Identity();self.action_decoder=torch.nn.Identity()
        def sample_time(self,*args,**kwargs):return torch.rand(1)
    head=Head();original=head.sample_time
    objective=ExecutionLoss(head,decoder(),state(),torch.zeros(1,40,10),[('a',0)],coefficient=0.)
    try:
        rng=torch.get_rng_state().clone();objective.begin(0,{'identity':dict(episode_id='a',anchor=0)})
        x=torch.tensor(2.,requires_grad=True);assert objective.apply(x)is x
        assert torch.equal(torch.get_rng_state(),rng)
    finally:objective.close()
    assert head.sample_time==original


def test_processor_config_uses_enum_names_not_serialized_enum_values():
    from enum import Enum
    from types import SimpleNamespace as NS
    Rep=Enum('Rep',dict(RELATIVE='relative',ABSOLUTE='absolute'))
    Kind=Enum('Kind',dict(EEF='eef',NON_EEF='non_eef'))
    Format=Enum('Format',dict(XYZ_ROT6D='xyz+rot6d',DEFAULT='default'))
    configs=NS(modality_keys=['eef_9d','gripper_position'],mean_std_embedding_keys=None,
        action_configs=[NS(rep=Rep.RELATIVE,type=Kind.EEF,format=Format.XYZ_ROT6D,state_key='eef_9d'),
                        NS(rep=Rep.ABSOLUTE,type=Kind.NON_EEF,format=Format.DEFAULT,state_key='gripper_position')])
    params={k:{n:v.cpu().numpy()for n,v in values.items()}for k,values in decoder().bounds.items()}
    sap=NS(modality_configs={'new_embodiment':{'action':configs}},use_relative_action=True,
        use_percentiles=False,norm_params={'new_embodiment':{'action':params}})
    actual=PhysicalDecoder.from_processor(NS(state_action_processor=sap),device='cpu')
    x=torch.zeros(1,40,132,dtype=torch.float64)
    torch.testing.assert_close(actual(x,state()),decoder()(x,state()),atol=0,rtol=0)
