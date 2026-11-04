"""WM02 neural components; tensor schema V1 is an engineering binding, not native qualification."""
from __future__ import annotations

import torch
from torch import nn

# Explicit floating-base design dimensions: 12 leg + 6 arm joints; gripper separate.
GROUPS = (("base_position",3),("base_rotation",9),("base_velocity",6),
          ("joint_position",18),("joint_velocity",18),("tcp_position",3),
          ("tcp_rotation",9),("tool_velocity",6),("gripper_fraction",1),
          ("gripper_speed",1),("object_position",3),("object_rotation",9),
          ("object_velocity",6),("native_progress",1))
SLICES = {}; offset = 0
for name, size in GROUPS:
    SLICES[name] = slice(offset, offset+size); offset += size
STATE_DIM = offset
CONTEXT_DIM, REFERENCE_DIM, HISTORY_HIDDEN, EVENT_DIM = 9, 400, 256, 4
SCHEMA = {"version":"WM02_NEURAL_ENGINEERING_STATE_V1", "physical":dict(GROUPS),
          "context":["goal_xyz","object_size_xyz","mass_friction_3"],
          "events":["left_contact","right_contact","grasp","support"],
          "restore_only_inputs":[], "native_tensor_binding":"NOT_DYNAMICALLY_QUALIFIED"}


def so3_exp(vector):
    """Smooth exponential, including zero; all future orientations remain on SO(3)."""
    x,y,z=vector.unbind(-1); o=torch.zeros_like(x)
    k=torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*vector.shape[:-1],3,3)
    theta=vector.square().sum(-1).clamp_min(1e-16).sqrt()
    a=torch.sinc(theta/torch.pi)[...,None,None]
    b=(.5*torch.sinc(theta/(2*torch.pi)).square())[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+a*k+b*(k@k)


def part(state,name):return state[...,SLICES[name]]


def compile_reference(state, raw):
    """Differentiable counterpart of bounded world increments; H40 tail holds node7.

    This is a neural graph, never a native receiver or an installation ACK.
    Gripper is the Actor's explicit [0,1] primitive, with no compiler rescale.
    """
    raw=raw.reshape(*raw.shape[:-1],8,7)
    pos=part(state,"tcp_position"); rot=part(state,"tcp_rotation").reshape(-1,3,3)
    points=[]
    for i in range(8):
        dp=raw[...,i,:3]; dr=raw[...,i,3:6]
        dp=.01*dp/dp.norm(dim=-1,keepdim=True).clamp_min(1)
        dr=.05*dr/dr.norm(dim=-1,keepdim=True).clamp_min(1)
        pos=pos+dp; rot=so3_exp(dr)@rot
        points.append(torch.cat((pos,rot[...,:2,:].reshape(-1,6),raw[...,i,6:7]),-1))
    return torch.stack(points+[points[-1]]*32,dim=-2)


class Normalizer(nn.Module):
    def __init__(self,dim=STATE_DIM):
        super().__init__(); self.register_buffer("mean",torch.zeros(dim));self.register_buffer("std",torch.ones(dim))
    def fit(self,train):
        axes=tuple(range(train.ndim-1))
        self.mean.copy_(train.mean(axes));self.std.copy_(train.std(axes).clamp_min(.02))
    def forward(self,state):return (state-self.mean)/self.std


def relative_reference(state,reference,slices):
    """Internal features only; external physical H40 reference stays untouched."""
    tcp=state[...,slices['tcp_position']];rotation=state[...,slices['tcp_rotation']].reshape(-1,3,3)
    xyz=(rotation.transpose(-1,-2)[:,None]@(reference[...,:3]-tcp[:,None])[...,None]).squeeze(-1)/.01
    rows=reference[...,3:9].reshape(-1,40,2,3)
    world=torch.cat((rows,torch.linalg.cross(rows[...,0,:],rows[...,1,:],dim=-1)[...,None,:]),-2)
    local=rotation.transpose(-1,-2)[:,None]@world
    grip=(reference[...,9:10]-state[...,slices['gripper_fraction']][:,None])/(.04/.044)
    return torch.cat((xyz,local[...,:2,:].flatten(-2),grip),-1)


class WorldModel(nn.Module):
    def __init__(self,hidden=(2048,)*5,history_hidden=HISTORY_HIDDEN,groups=GROUPS,context_dim=CONTEXT_DIM,reference_encoding='absolute',interaction_encoding='independent_v1'):
        super().__init__();self.groups=tuple((name,int(n))for name,n in groups)
        if reference_encoding not in('absolute','relative_tcp_v1'):raise ValueError('Versioned internal reference encoding required')
        self.reference_encoding=reference_encoding
        if interaction_encoding not in ('independent_v1','conditional_rigid_residual_v1'):
            raise ValueError('Versioned interaction structure required')
        self.interaction_encoding=interaction_encoding
        self.slices={};offset=0
        for name,n in self.groups:self.slices[name]=slice(offset,offset+n);offset+=n
        self.normalizer=Normalizer(offset);self.history_hidden=history_hidden
        inputs=offset+REFERENCE_DIM+context_dim+2
        self.history=nn.GRUCell(inputs,history_hidden)
        layers=[];width=inputs+history_hidden
        for out in hidden:layers.extend((nn.Linear(width,out),nn.LayerNorm(out),nn.SiLU()));width=out
        self.trunk=nn.Sequential(*layers)
        delta=sum(3 if name.endswith("rotation")else n for name,n in self.groups)
        self.delta=nn.Linear(width,delta);self.events=nn.Linear(width,EVENT_DIM);self.done=nn.Linear(width,1)
        nn.init.normal_(self.delta.weight,std=1e-4);nn.init.zeros_(self.delta.bias);nn.init.constant_(self.done.bias,-3)
    def initial_hidden(self,state):return state.new_zeros((len(state),self.history_hidden))
    def inputs(self,state,reference,context,clock):
        if self.reference_encoding=='relative_tcp_v1':
            reference=relative_reference(state,reference,self.slices)
        return torch.cat((self.normalizer(state),reference.flatten(-2),context,clock),-1)
    def burn(self,states,references,context,clocks=None):
        if states.shape[1]!=8:raise ValueError("Exactly eight causal burn-in atoms")
        h=self.initial_hidden(states[:,0])
        for i in range(8):
            clock=(states.new_tensor([i*.05/32,(i%8)*.05]).expand(len(states),2)
                   if clocks is None else clocks[:,i])
            h=self.history(self.inputs(states[:,i],references[:,i],context,clock),h)
        return h
    def step(self,state,reference,context,clock,hidden):
        x=self.inputs(state,reference,context,clock);hidden=self.history(x,hidden)
        z=self.trunk(torch.cat((x,hidden),-1));d=self.delta(z);parts=[];offset=0
        for name,n in self.groups:
            amount=3 if name.endswith("rotation")else n;v=d[:,offset:offset+amount];offset+=amount
            old=state[...,self.slices[name]]
            if name.endswith("rotation"):
                new=(so3_exp(.05*v)@old.reshape(-1,3,3)).flatten(-2)
            else:
                scale=.01 if name.endswith("position") else .05
                new=old+scale*v
                if name in ("gripper_fraction","native_progress"):new=new.clamp(0,1)
            parts.append(new)
        new=torch.cat(parts,-1)
        if self.interaction_encoding=='conditional_rigid_residual_v1':
            from .structured_interaction import conditional_rigid_residual
            new=conditional_rigid_residual(state,new,reference,clock,context,self.slices)
        return new,self.events(z),self.done(z).squeeze(-1),hidden


def actor_observation(state,context,last_action,time_s):
    """Declared task fields only: excludes base/joints, hidden state, contact oracle and phase."""
    def rot6(name):return part(state,name).reshape(-1,3,3)[...,:2,:].flatten(-2)
    remaining=state.new_full((len(state),1),(32-time_s)/32)
    residual=state.new_zeros((len(state),6)) # Explicit engineering fixture; native tracking binding is pending.
    return torch.cat((part(state,"tcp_position"),rot6("tcp_rotation"),part(state,"gripper_fraction"),
        part(state,"gripper_speed"),part(state,"tool_velocity"),part(state,"object_position"),
        rot6("object_rotation"),part(state,"object_velocity"),context[:,:3],last_action,residual,remaining),-1)


ACTOR_OBS_DIM=98
def mlp(inputs,outputs):
    layers=[]
    for out in (256,256,256):layers.extend((nn.Linear(inputs,out),nn.SiLU()));inputs=out
    return nn.Sequential(*layers,nn.Linear(inputs,outputs))


class Actor(nn.Module):
    def __init__(self):super().__init__();self.net=mlp(4*ACTOR_OBS_DIM,56)
    def forward(self,history):
        values=self.net(history.flatten(-2)).reshape(-1,8,7)
        return torch.cat((values[...,:6].tanh(),values[...,6:7].sigmoid()),-1).flatten(-2)


class Critic(nn.Module):
    def __init__(self):super().__init__();self.net=mlp(4*ACTOR_OBS_DIM,1)
    def forward(self,history):return self.net(history.flatten(-2)).squeeze(-1)


def imagine(model,actor,state,context,hidden,horizon):
    if horizon not in (1,8,16):raise ValueError("Bounded engineering horizon")
    last=state.new_zeros((len(state),56));obs=actor_observation(state,context,last,.4)
    history=obs[:,None].expand(-1,4,-1);alive=state.new_ones(len(state));total=state.new_zeros(len(state))
    actions=[];decisions=[];reference=None
    initial_history=history
    for i in range(horizon):
        if i%8==0:
            obs=actor_observation(state,context,last,.4+i*.05)
            history=torch.cat((history[:,1:],obs[:,None]),1)
            last=actor(history);actions.append(last);decisions.append(i);reference=compile_reference(state,last)
        clock=state.new_tensor([(.4+i*.05)/32,(i%8)*.05]).expand(len(state),2)
        new,events,done,new_hidden=model.step(state,reference,context,clock,hidden)
        new=torch.where(alive[:,None].bool(),new,state)
        hidden=torch.where(alive[:,None].bool(),new_hidden,hidden)
        # Synthetic engineering objective, deliberately not an unbound native reward.
        distance=(part(new,"object_position")-context[:,:3]).square().sum(-1)
        reward=-distance-.01*last.square().mean(-1)
        total=total+(.997**i)*alive*reward
        alive=alive*(done.sigmoid()<.5).to(alive.dtype).detach();state=new
    return total,initial_history,actions,decisions,state
