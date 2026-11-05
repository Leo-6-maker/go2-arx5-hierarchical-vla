"""Compact HOLD plus XY response; no extrapolation or safety/Markov claim."""
import numpy as np
from .target_arx import CausalARX
from .target_stability import rotation_wxyz
OFFSETS=np.asarray([[0.,0.],[.001,0.],[-.001,0.],[0.,.001]])


def future_choice_mask(data):
    result=dict(data);goal=data['goal'].copy();goal[:,:3]=data['history_goal'][:,0,:3];result['goal']=goal
    return result


def action(data,ids,*,strict=True):
    goal=data['goal'][ids];past=data['history_goal'][ids,0];d=goal[:,:3]-past[:,:3]
    if strict and(not np.allclose(goal[:,3:],past[:,3:],atol=1e-8,rtol=0)or np.any(np.abs(d[:,2])>1e-8)or any(not np.any(np.all(np.isclose(row[:2],OFFSETS,atol=1e-8,rtol=0),axis=1))for row in d)):
        raise ValueError('OUTSIDE_IDENTIFIED_LOCAL_XY_ACTIONS_NO_EXTRAPOLATION')
    return d[:,:2]/.001


def compact(state,meta):
    sl=CausalARX(meta).layout;q0=sl['qpos'].start;r=rotation_wxyz(state[...,q0+3:q0+7]);tcp=state[...,sl['tcp_position_m']];base=state[...,q0:q0+3];relative=np.einsum('...ji,...j->...i',r,tcp-base)
    return np.concatenate((tcp,base,relative),axis=-1)


class HoldResponse:
    def __init__(self,meta,*,baseline='velocity',ridge=.1):self.meta=meta;self.baseline=baseline;self.ridge=ridge
    def features(self,data,ids):
        history=compact(data['state'][ids,:9],self.meta);velocity=(history[:,8]-history[:,7])/.05
        return np.concatenate((velocity,np.ones((len(ids),1))),axis=1)if self.baseline=='velocity'else np.ones((len(ids),1))
    def fit(self,data,ids):
        ids=np.asarray(ids);a=action(data,ids);hold=np.all(np.abs(a)<1e-8,axis=1)
        if hold.sum()<2:raise ValueError('REAL_TRAIN_HOLD_REPEATS_REQUIRED')
        x=self.features(data,ids);truth=compact(data['state'][ids],self.meta);y=truth[:,9:]-truth[:,8,None];self.x_mean=x[hold].mean(0);self.x_scale=np.maximum(x[hold].std(0),1e-5);self.x_mean[-1]=0;self.x_scale[-1]=1;z=(x-self.x_mean)/self.x_scale;penalty=np.eye(z.shape[1])*self.ridge;penalty[-1,-1]=0
        self.baseline_weight=np.linalg.solve(z[hold].T@z[hold]+penalty,z[hold].T@y[hold].reshape(hold.sum(),-1));residual=y.reshape(len(ids),-1)-z@self.baseline_weight;self.response_weight=np.linalg.solve(a.T@a+np.eye(2)*self.ridge,a.T@residual);return self
    def predict(self,data,ids):
        ids=np.asarray(ids);a=action(data,ids);x=self.features(data,ids);z=(x-self.x_mean)/self.x_scale;anchor=compact(data['state'][ids,8],self.meta);delta=z@self.baseline_weight+a@self.response_weight
        return anchor[:,None]+delta.reshape(len(ids),16,9)
    def arrays(self):return {k:getattr(self,k)for k in ['x_mean','x_scale','baseline_weight','response_weight']}

    @classmethod
    def from_arrays(cls,meta,arrays,*,baseline,ridge):
        m=cls(meta,baseline=baseline,ridge=ridge);n=10 if baseline=='velocity'else 1
        for k,shape in dict(x_mean=(n,),x_scale=(n,),baseline_weight=(n,144),response_weight=(2,144)).items():
            value=np.asarray(arrays[k],dtype=float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('FROZEN_RESPONSE_ARRAY_LAYOUT_DIFFERS')
            if k=='x_scale'and np.any(value<=0):raise ValueError('INVALID_RESPONSE_FEATURE_SCALE')
            setattr(m,k,value.copy())
        return m
