"""Prospectively fixed R3 box objective; called only after the recurrence trigger."""
import torch

def box_loss(generated,mask):
    if tuple(generated.shape)!=(1,40,132) or mask.shape!=generated.shape:raise ValueError('Original batch1 H40/channel contract required')
    with torch.autocast(device_type=generated.device.type,enabled=False):
        fraction=(generated[...,9].float()+1)/2;m=mask[...,9].float()
        if not torch.isfinite(fraction).all() or not ((m==0)|(m==1)).all() or not m.any():raise ValueError('Nonfinite commands or invalid/empty channel9 mask')
        value=(m*((torch.relu(-fraction)/.05).square()+(torch.relu(fraction-1)/.05).square())).sum()/m.sum()
        return value,fraction

def check():
    raw=torch.zeros(1,40,132,requires_grad=True);mask=torch.zeros_like(raw);mask[...,:10]=1
    assert box_loss(raw,mask)[0].item()==0
    with torch.no_grad():raw[0,0,9]=-1.4;raw[0,39,9]=1.4
    value,fraction=box_loss(raw,mask);value.backward()
    assert value.dtype==torch.float32 and torch.allclose(fraction[0,[0,39]],torch.tensor([-.2,1.2]),atol=1e-6)
    assert abs(value.item()-.8)<1e-5 and torch.allclose(raw.grad[0,[0,39],9],torch.tensor([-2.,2.]),atol=1e-5)
    assert not raw.grad[...,:9].any() and not raw.grad[...,10:].any()
    for i in (0,7,8,39):
        plus=raw.detach().clone();minus=plus.clone();plus[0,i,9]+=.001;minus[0,i,9]-=.001
        fd=(box_loss(plus,mask)[0]-box_loss(minus,mask)[0])/.002
        assert torch.allclose(fd,raw.grad[0,i,9],rtol=.003,atol=1e-4)
    mask[0,0,9]=0;before=box_loss(raw,mask)[0];modified=raw.detach().clone();modified[0,0,9]=-100
    assert torch.equal(before,box_loss(modified,mask)[0])
    for bad in (torch.zeros_like(mask),torch.full_like(mask,.5)):
        try:box_loss(raw,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid mask accepted')
    return dict(status='PASS_R3_BOX_CPU_ANALYTIC_FINITE_DIFFERENCE_MASK_AFFINE',launch_allowed=False,objective_applied=False,FP32=True,uniform_H40=True,coefficient=.1,scale=.05,new_model_queries=0,new_physics_steps=0,real_continuing_gradient='PENDING_TRIGGERED_FORMAL_UPDATE')

if __name__=='__main__':
    import json
    print(json.dumps(check()))
