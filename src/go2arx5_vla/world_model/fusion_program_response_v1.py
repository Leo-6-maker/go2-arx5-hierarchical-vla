"""Small matched response networks; no native executor or Actor/Critic."""
import numpy as np
import os
import torch
from torch import nn
from .fusion_program_response_data_v1 import ARMS,FEATURE_DIM,FUTURE_SLICE

SCHEMA='WM02_PROGRAM_RESPONSE_CHECKPOINT_V1'


class ProgramResponse(nn.Module):
    def __init__(self,arm,hidden=32):
        super().__init__()
        if arm not in ARMS or hidden!=32:raise ValueError('REGISTERED_MATCHED_PROGRAM_MODEL_REQUIRED')
        self.arm=arm
        self.hidden=nn.Linear(FEATURE_DIM,hidden);self.head=nn.Linear(hidden,128)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        for key,shape in [('x_mean',(FEATURE_DIM,)),('x_scale',(FEATURE_DIM,)),('y_mean',(16,8)),('y_scale',(16,8))]:
            self.register_buffer(key,torch.ones(shape)if key.endswith('scale')else torch.zeros(shape))

    def set_normalizers(self,stats):
        for key in ('x_mean','x_scale','y_mean','y_scale'):
            getattr(self,key).copy_(torch.as_tensor(stats[key],device=self.x_mean.device))

    def forward(self,x):
        z=(x-self.x_mean)/self.x_scale
        if self.arm=='NO_FUTURE_ACTION':
            z=z.clone();z[:,FUTURE_SLICE]=0.
        return self.y_mean+self.y_scale*self.head(torch.tanh(self.hidden(z))).reshape(-1,16,8)

    def loss(self,x,y,valid):
        error=self(x)-y
        if self.arm!='RAW':error=error/self.y_scale
        mask=valid[:,:,None].to(error.dtype)
        return (error.square()*mask).sum()/(mask.sum()*8)

    def arrays(self):
        out={k:getattr(self,k).detach().cpu().numpy().copy()for k in ('x_mean','x_scale','y_mean','y_scale')}
        out.update(w0=self.hidden.weight.detach().cpu().numpy().copy(),b0=self.hidden.bias.detach().cpu().numpy().copy(),
            w1=self.head.weight.detach().cpu().numpy().copy(),b1=self.head.bias.detach().cpu().numpy().copy(),
            mask_future=np.array(self.arm=='NO_FUTURE_ACTION'))
        return out


def update(model,optimizer,x,y,valid,index):
    optimizer.zero_grad(set_to_none=True);loss=model.loss(x[index],y[index],valid[index]);loss.backward()
    norm=torch.sqrt(sum(p.grad.square().sum()for p in model.parameters()))
    if not torch.isfinite(loss)or not torch.isfinite(norm):raise FloatingPointError('NONFINITE_PROGRAM_MODEL_NO_CLIP_OR_RETRY')
    optimizer.step()
    return float(loss.detach()),float(norm.detach())


def save_checkpoint(path,model,optimizer,sampler,step,provenance):
    from .neural_training import rng_state
    value=dict(schema=SCHEMA,model_config=dict(arm=model.arm,hidden=32),
        model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
        updates=step,rng=rng_state(),provenance=provenance)
    with open(path,'xb')as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())


def load_checkpoint(path,device,expected):
    from .neural_training import restore_rng
    from .actor_sampling import ExactSampler
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['schema']!=SCHEMA or value['provenance']!=expected:
        raise ValueError('PROGRAM_CHECKPOINT_SOURCE_DATA_OR_CONFIG_DIFFERS')
    model=ProgramResponse(**value['model_config']).to(device);model.load_state_dict(value['model'])
    optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(value['optimizer'])
    s=value['sampler'];sampler=ExactSampler(s['size'],s['mode'],s['seed']);sampler.load_state_dict(s)
    restore_rng(value['rng'])
    return model,optimizer,sampler,value
