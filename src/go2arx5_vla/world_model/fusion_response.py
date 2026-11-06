"""Small supervised TCP response model over an unchanged frozen ARX rollout."""
from collections import Counter
import numpy as np
import torch
from torch import nn

ARMS=('ACTION_CONDITIONED','NO_FUTURE_ACTION')


class ResponseModel(nn.Module):
    def __init__(self, dimension, hidden=32):
        super().__init__()
        self.register_buffer('mean',torch.zeros(dimension))
        self.register_buffer('scale',torch.ones(dimension))
        self.hidden=nn.Linear(dimension,hidden);self.head=nn.Linear(hidden,48)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def fit_normalizer(self, train_x):
        self.mean.copy_(train_x.mean(0));self.scale.copy_(train_x.std(0,unbiased=False).clamp_min(1e-4))

    def forward(self,x,base):
        residual=self.head(torch.tanh(self.hidden((x-self.mean)/self.scale))).reshape(-1,16,3)
        return base+.001*residual

    def arrays(self):
        return {name:t.detach().cpu().numpy().copy() for name,t in dict(mean=self.mean,scale=self.scale,
            w0=self.hidden.weight,b0=self.hidden.bias,w1=self.head.weight,b1=self.head.bias).items()}


def validate_data(data):
    n=len(data['state'])
    if (n!=48 or data['state'].shape!=(n,25,106) or data['goal'].shape!=(n,17)
            or data['history_goal'].shape!=(n,8,17) or data['source_age'].shape!=(n,25)
            or data['source_age_valid'].shape!=(n,25) or not np.isfinite(data['state']).all()
            or len(set(data['run_id']))!=n or data['repeat_index'].shape!=(n,)
            or not np.isfinite(data['goal']).all() or not np.isfinite(data['history_goal']).all()
            or not np.isfinite(data['source_age']).all()
            or data['repeat_index'].dtype.kind not in 'iu' or np.any(data['repeat_index']<0) or np.any(data['repeat_index']>1)
            or not np.array_equal(data['goal'][:,3:],data['history_goal'][:,0,3:])
            or not np.array_equal(data['history_goal'],np.repeat(data['history_goal'][:,:1],8,axis=1))
            or np.any(data['source_age'][:,8]<-1e-8) or np.any(data['source_age'][:,8]>=.4)
            or not np.all(data['source_age_valid'][:,8])
            or np.any(data['source_age_valid'][:,:8]) or np.any(data['source_age_valid'][:,9:])
            or not np.all(data['future_prefix_remaining']==7) or not np.all(data['window_start_s']==0)):
        raise ValueError('COMPLETE_BALANCED_REAL_CAUSAL_RESPONSE_REPLAY_REQUIRED')
    parents={}
    for parent,split in zip(data['parent'],data['split']):
        if parent in parents and parents[parent]!=split:raise ValueError('WHOLE_PARENT_SPLIT_REQUIRED')
        parents[parent]=split
    if Counter(parents.values())!=dict(TRAIN=4,DEV=2):raise ValueError('FOUR_TRAIN_TWO_DEV_NO_TEST_REQUIRED')
    from .fusion_contract import OFFSETS
    actions=data['observed_action']
    if actions.dtype.kind not in 'iu' or np.any(actions<0) or np.any(actions>3):raise ValueError('QUALIFIED_EXECUTED_ACTION_IDS_REQUIRED')
    if not np.allclose(data['goal'][:,:3]-data['history_goal'][:,0,:3],OFFSETS[actions],atol=1e-10,rtol=0):
        raise ValueError('EXECUTED_GOAL_AND_ACTION_ID_DIFFER')
    for parent in parents:
        ids=np.flatnonzero(data['parent']==parent)
        if (Counter(map(int,actions[ids]))!=dict.fromkeys(range(4),2)
                or Counter(zip(map(int,actions[ids]),map(int,data['repeat_index'][ids])))
                !=Counter((a,r)for a in range(4)for r in range(2))):
            raise ValueError('ALL_FOUR_ACTIONS_TWO_PHYSICAL_RESTARTS_PER_PARENT_REQUIRED')
    return {s:np.flatnonzero(data['split']==s) for s in ('TRAIN','DEV')}


def prepare(teacher,data,meta,arm):
    if arm not in ARMS:raise ValueError('MATCHED_RESPONSE_ARM_REQUIRED')
    from .pilot1_residual import inputs
    # This removes the future candidate from BOTH features and the ARX base.
    return inputs(teacher,data,meta,blind_choice=arm=='NO_FUTURE_ACTION')


def action_effect(prediction,hold_prediction,data,ids,layout):
    sl=layout['tcp_position_m'];pairs=[];prefix=[]
    for i in ids:
        if data['observed_action'][i]==0:continue
        holds=ids[(data['parent'][ids]==data['parent'][i])&(data['observed_action'][ids]==0)]
        if len(holds)!=2:raise ValueError('TWO_SAME_PARENT_HOLD_RESTARTS_REQUIRED')
        actual=(data['state'][i,9:,sl]-data['state'][i,8:9,sl])-(
            data['state'][holds,9:,sl]-data['state'][holds,8:9,sl]).mean(0)
        pairs.append(prediction[i]-hold_prediction[i]-actual)
        prefix.extend(np.linalg.norm(data['state'][holds,8,sl]-data['state'][i,8,sl],axis=-1))
    return dict(independent_restart_pairs=len(pairs),exact_forks=False,
        max_prefix_TCP_mismatch_mm=float(max(prefix))*1000,
        horizons={str(h):dict(XYZ_action_delta_RMSE_mm=float(np.sqrt(np.mean(np.asarray(pairs)[:,h-1]**2)))*1000)for h in (1,4,8,16)})
