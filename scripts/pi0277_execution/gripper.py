"""The sole PI0277 loss addition, on the existing final S4 channel 9."""
import torch

def generated_gripper_loss(generated, label, mask):
    if generated.shape != label.shape or mask.shape != generated.shape or tuple(generated.shape) != (1,40,132):
        raise ValueError('Expected original batch1 H40 132-channel tensors')
    with torch.autocast(device_type=generated.device.type, enabled=False):
        g, target = (generated[...,9].float()+1)/2, (label[...,9].float()+1)/2
        m = mask[...,9].float()
        if not torch.isfinite(g).all() or not torch.isfinite(target).all() or not ((m==0)|(m==1)).all() or not m.any():
            raise ValueError('Nonfinite commands or invalid/empty channel-9 mask')
        w = g.new_full((40,),.2/32); w[:8] = .8/8
        weights = w*m
        return (weights*((g-target)/.10).square()).sum()/weights.sum()

def check():
    g=torch.zeros(1,40,132,requires_grad=True);target=torch.zeros_like(g);mask=torch.zeros_like(g);mask[...,:10]=1
    assert generated_gripper_loss(g,target,mask).item()==0
    target[...,9]=.2
    loss=generated_gripper_loss(g,target,mask);loss.backward()
    assert abs(loss.item()-1)<1e-5
    w=torch.full((40,),.2/32);w[:8]=.8/8
    assert torch.allclose(g.grad[0,:,9],-10*w,rtol=1e-5,atol=1e-6)
    assert not g.grad[...,:9].any() and not g.grad[...,10:].any()
    for i in (0,7,8,39):
        plus=g.detach().clone();minus=plus.clone();plus[0,i,9]+=.001;minus[0,i,9]-=.001
        finite=(generated_gripper_loss(plus,target,mask)-generated_gripper_loss(minus,target,mask))/.002
        assert torch.allclose(finite,g.grad[0,i,9],rtol=.003,atol=1e-4)
    early=torch.zeros_like(g);late=early.clone();early[0,0,9]=.2;late[0,39,9]=.2
    ratio=generated_gripper_loss(early,torch.zeros_like(g),mask)/generated_gripper_loss(late,torch.zeros_like(g),mask)
    assert abs(ratio.item()-16)<1e-5
    mask[0,0,9]=0;before=generated_gripper_loss(early,torch.zeros_like(g),mask);early[0,0,9]=1e4
    assert torch.equal(before,generated_gripper_loss(early,torch.zeros_like(g),mask))
    for bad in (torch.zeros_like(mask),torch.full_like(mask,.5)):
        try:generated_gripper_loss(g,target,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid mask accepted')
    return dict(status='PASS_PI0277_ANALYTICAL_GRIPPER_CPU',fp32=True,affine_inverse=True,
                exact_zero=True,nonzero_analytical_gradient=True,finite_difference=True,
                prefix8_weight_mass=.8,suffix32_weight_mass=.2,mask_invariance=True,
                coefficient_in_production_required=.1,model_queries=0,learned_backwards=0,
                real_four_step_gradient_evidence='PENDING_AUTHORIZED_CANARY')

if __name__=='__main__':
    import json
    print(json.dumps(check()))
