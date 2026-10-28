"""Only the approved physical temporal reductions; native and speed untouched."""
import json
import torch

def weighted(errors,valid):
    if errors.ndim!=2 or errors.shape[0]!=1 or errors.shape!=valid.shape or valid.dtype!=torch.bool:
        raise ValueError('Expected production batch1 and unchanged Boolean validity')
    n=errors.shape[1]
    if n not in (39,40) or not torch.isfinite(errors).all() or (errors<0).any() or not valid.any():
        raise ValueError('Invalid physical support/errors')
    prefix=9 if n==40 else 8
    w=errors.new_full((n,),.2/31);w[:prefix]=.8/prefix
    weights=w*valid
    return (errors*weights).sum()/weights.sum()

def position_errors(g,e,m,s,scale,offset,rollout):
    with torch.autocast(device_type=g.device.type,enabled=False):
        gp=rollout.decode_xyz(g[...,:3].float(),s.float(),scale.float(),offset.float())
        ep=rollout.decode_xyz(e[...,:3].float(),s.float(),scale.float(),offset.float())
        residual=gp-ep
        point=(residual/rollout.P_REF_M).square().sum(-1)
        transition=((residual[:,1:]-residual[:,:-1])/rollout.DT_S/rollout.V_REF_MPS).square().sum(-1)
    valid=(m[...,:3]>0).all(-1)
    return point,transition,valid,valid[:,1:]&valid[:,:-1]

def install(arm,output):
    assert arm in ('UNIFORM','PREFIX')
    import pi0247_u_rollout_overlay as rollout
    import pi0248_rotation_cpu as rotation
    original=rollout._physical_losses;original_rotation=rotation.rotation_chordal_loss
    current={}

    def rotational(g,e,m,s,minimum,maximum,**kw):
        uniform,details=original_rotation(g,e,m,s,minimum,maximum,**kw)
        with torch.autocast(device_type=g.device.type,enabled=False):
            pred,_=rotation.decode_relative_rotation(g[...,3:9].float(),s[:,3:9].float(),minimum,maximum)
            target,_=rotation.decode_relative_rotation(e[...,3:9].float(),s[:,3:9].float(),minimum,maximum)
            error=(pred-target).square().sum((-2,-1))/(2*float(kw.get('theta_ref',rotation.THETA_REF))**2)
            prefix=weighted(error,(m[...,3:9]>0).all(-1))
        current.update(uniform_rotation=float(uniform.detach()),weighted_rotation=float(prefix.detach()))
        return (uniform if arm=='UNIFORM' else prefix),details

    def physical(g,e,m,s,scale,offset):
        pos,delta,details=original(g,e,m,s,scale,offset)
        pe,de,pv,dv=position_errors(g,e,m,s,scale,offset,rollout)
        wp,wd=weighted(pe,pv),weighted(de,dv)
        with (output/'temporal_components.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(arm=arm,uniform_position=float(pos.detach()),weighted_position=float(wp.detach()),
                uniform_increment=float(delta.detach()),weighted_increment=float(wd.detach()),**current),sort_keys=True)+'\n')
        # ponytail: reuse the original exact UNIFORM reduction, including its operation order.
        return (pos,delta,details) if arm=='UNIFORM' else (wp,wd,details)

    rotation.rotation_chordal_loss=rotational
    rollout._physical_losses=physical

def self_check():
    import numpy as np
    checks=[]
    for n,p in ((40,9),(39,8)):
        mask=torch.ones((1,n),dtype=torch.bool)
        assert weighted(torch.zeros(1,n),mask)==0
        assert torch.allclose(weighted(torch.ones(1,n)*7,mask),torch.tensor(7.))
        x=torch.linspace(.2,1.2,n,dtype=torch.float64).reshape(1,n).requires_grad_()
        loss=weighted(x.square(),mask);loss.backward()
        w=np.r_[np.full(p,.8/p),np.full(n-p,.2/31)]
        assert np.isclose(w.sum(),1.) and np.isclose(w[:p].sum(),.8)
        assert np.max(abs(x.grad.numpy()[0]-2*x.detach().numpy()[0]*w))<1e-12
        for index in (0,p-1,p,n-1):
            plus=x.detach().clone();minus=plus.clone();plus[0,index]+=1e-6;minus[0,index]-=1e-6
            finite=(weighted(plus.square(),mask)-weighted(minus.square(),mask))/2e-6
            assert abs(float(finite)-float(x.grad[0,index]))<1e-9
        early=torch.zeros(1,n);late=early.clone();early[0,0]=1;late[0,-1]=1
        assert weighted(early,mask)>weighted(late,mask)
        mask[0,0]=False;before=weighted(early,mask);early[0,0]=1e8;assert weighted(early,mask)==before
        for invalid in (torch.zeros_like(mask),mask.float()):
            try:weighted(early,invalid)
            except ValueError:pass
            else:raise AssertionError('invalid support accepted')
        checks.append(dict(horizon=n,positive_unit_mass=True,constant_zero_mask=True,independent_gradient_and_finite_difference=True))
    return dict(status='PASS_TEMPORAL_REDUCTION_CPU_FIXTURES',checks=checks,model_queries=0,learned_backwards=0)

if __name__=='__main__':print(json.dumps(self_check()))
