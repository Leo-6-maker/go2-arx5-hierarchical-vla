"""V2 physical-unit differentiable compiler, separate from the V1 synthetic graph."""
import torch
from .neural import so3_exp

VERSION='WM02_PHYSICAL_WORLD_DELTA_H40_V2'


def actor_primitives(logits,*,target):
    """Single explicit latent-to-physical transform; native compiler never rescales."""
    x=logits.reshape(-1,8,7)
    p=x[...,:3].tanh();r=x[...,3:6].tanh()
    # Construct inside the physical set: saturated FP32 components can otherwise
    # round above the double-precision native norm/stroke bounds. This guard is
    # part of the latent transform, never correction of an issued/raw label.
    guard=8*torch.finfo(x.dtype).eps
    p=(.01*(1-guard))*p/p.norm(dim=-1,keepdim=True).clamp_min(1)
    r=(.05*(1-guard))*r/r.norm(dim=-1,keepdim=True).clamp_min(1)
    lo,hi=(.002/.044,.042/.044)if target else(0.,1.)
    if target:lo,hi=lo+(hi-lo)*guard,hi-(hi-lo)*guard
    g=lo+(hi-lo)*x[...,6:7].sigmoid()
    return torch.cat((p,r,g),-1)


def compile_physical(raw,anchor,time_s,*,target=False):
    """World translation / left-multiplied rotation, row rot6d, absolute timestamps."""
    if raw.shape[-2:]!=(8,7)or anchor.shape[-2:]!=(4,4):raise ValueError('Physical 8x7 and TCP 4x4 required')
    lo,hi=(.002/.044,.042/.044)if target else(0.,1.)
    checked=raw.to(torch.float64) # Match the native double-precision reject decision.
    if (not torch.isfinite(raw).all()or not torch.isfinite(anchor).all()
            or (checked[...,:3].norm(dim=-1)>.01+1e-12).any()
            or (checked[...,3:6].norm(dim=-1)>.05+1e-12).any()
            or (checked[...,6]<lo).any()or(checked[...,6]>hi).any()):
        raise ValueError('Physical primitive outside native bounds; no clipping')
    r=anchor[...,:3,:3];eye=torch.eye(3,dtype=anchor.dtype,device=anchor.device)
    if (not torch.allclose(r.transpose(-1,-2)@r,eye.expand_as(r),atol=1e-7,rtol=0)
            or not torch.allclose(torch.linalg.det(r),torch.ones_like(torch.linalg.det(r)),atol=1e-7,rtol=0)
            or not torch.allclose(anchor[...,3,:],anchor.new_tensor([0,0,0,1]).expand_as(anchor[...,3,:]),atol=1e-12,rtol=0)):
        raise ValueError('Actual SE3 anchor required; no frame repair')
    pos=anchor[...,:3,3];rot=anchor[...,:3,:3];points=[]
    for i in range(8):
        pos=pos+raw[...,i,:3];rot=so3_exp(raw[...,i,3:6])@rot
        points.append(torch.cat((pos,rot[...,:2,:].flatten(-2),raw[...,i,6:7]),-1))
    rows=torch.stack(points+[points[-1]]*32,-2)
    time=torch.as_tensor(time_s,dtype=raw.dtype,device=raw.device)
    if not torch.isfinite(time).all():raise ValueError('Finite absolute command clock required')
    timestamps=time[...,None]+torch.arange(40,dtype=raw.dtype,device=raw.device)*.05
    return rows,timestamps


def tracking_residual(tcp,installed_eef10):
    """Measured/predicted task error to an analytically retained command, never padded zero."""
    rows=installed_eef10[...,3:9].reshape(-1,2,3)
    reference=torch.stack((rows[:,0],rows[:,1],torch.linalg.cross(rows[:,0],rows[:,1])),-2)
    relative=reference@tcp[...,:3,:3].transpose(-1,-2)
    skew=.5*torch.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),-1)
    # sin-angle residual: explicit bounded proxy, not an SO(3) log near pi.
    return torch.cat((installed_eef10[...,:3]-tcp[...,:3,3],skew),-1)
