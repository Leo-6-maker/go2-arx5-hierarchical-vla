"""History-conditioned 50ms task-space contact model; no hidden-state fiction.

Predict only TCP/object displacement and actual per-native-tick contact-any.
Known next reference is an input. Future achieved state, stage and score are not.
This first contact head is not a full-state simulator or a free-rollout model.
"""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def state_feature(row):
    memory=row['actual_gripper_internal_memory']['values']
    return np.r_[row['qpos'],row['qvel'],row['ctrl'],row['TCP_m'],
                 np.asarray(row['object_pose'])[:3,3],row['finger_q'],
                 max(e['max_pad_cube_normal_force_N'] for e in row['native_contact_events']),
                 memory['target'],memory['opening'],memory['expiry']-row['native_end_step']*.002]


def replay(paths):
    episodes=[];parents={}
    for path in paths:
        value=json.loads(gzip.decompress(Path(path).read_bytes()))
        if value.get('schema')!='WM02_PILOT1_ACTUAL_CONTACT_BUNDLE_V1' or value.get('complete_MPC_WBC_hidden_state') is not False:
            raise ValueError('ACTUAL_HISTORY_CONDITIONAL_CONTACT_BUNDLE_REQUIRED')
        for episode in value['episodes']:
            split=episode['split'];parent=episode['parent']
            if split not in ('TRAIN','DEV','TEST') or parent in parents and parents[parent]!=split:
                raise ValueError('WHOLE_PHYSICAL_PARENT_PARTITION_REQUIRED')
            parents[parent]=split;episodes.append(episode)
    if len({e['run_id'] for e in episodes})!=len(episodes):raise ValueError('DUPLICATED_CONTACT_EPISODE')
    x=[];blind=[];truth=[];event=[];current=[];cv=[];partition=[];identities=[]
    for episode in episodes:
        rows=episode['samples']
        if len(rows)!=episode['episode']['task_atoms']:raise ValueError('COMPLETE_ACTUAL_CONTACT_EPISODE_REQUIRED')
        for i,row in enumerate(rows):
            start=2503+25*i
            if row['native_start_step']!=start or row['native_end_step']!=start+25:
                raise ValueError('ORIGINAL_CONTACT_ATOM_CLOCK_REQUIRED')
            events=row['native_contact_events']
            if [e['native_step']for e in events]!=list(range(start+1,start+26)):
                raise ValueError('ALL25_ACTUAL_NATIVE_CONTACT_LABELS_REQUIRED')
            labels=[e['max_pad_cube_normal_force_N']>=.05 for e in events]
            if labels!=[e['contact_positive']for e in events]or bool(row['contact_any'])!=any(labels):
                raise ValueError('ORIGINAL_FORCE_THRESHOLD_NOT_SYNTHETIC_GEOMETRY_REQUIRED')
        for i in range(8,len(rows)-1):
            history=np.concatenate([state_feature(r) for r in rows[i-8:i+1]])
            # These reference values were fixed by the predeclared native
            # program/initial observation before the following integration.
            # No future clock, receiver memory or achieved state is used.
            command=np.asarray(rows[i+1]['executed_reference_EEF10'])
            past=np.asarray(rows[i]['executed_reference_EEF10'])
            now=np.r_[rows[i]['TCP_m'],np.asarray(rows[i]['object_pose'])[:3,3]]
            previous=np.r_[rows[i-1]['TCP_m'],np.asarray(rows[i-1]['object_pose'])[:3,3]]
            after=np.r_[rows[i+1]['TCP_m'],np.asarray(rows[i+1]['object_pose'])[:3,3]]
            x.append(np.r_[history,command]);blind.append(np.r_[history,past]);truth.append((after-now)/.001)
            event.append(float(rows[i+1]['contact_any']));current.append(float(rows[i]['contact_any']))
            cv.append((now-previous)/.001);partition.append(episode['split'])
            identities.append(dict(run_id=episode['run_id'],parent=episode['parent'],anchor_native_step=rows[i]['native_end_step']))
    out=dict(x=np.asarray(x,dtype=np.float32),blind=np.asarray(blind,dtype=np.float32),
             delta_mm=np.asarray(truth,dtype=np.float32),event=np.asarray(event,dtype=np.float32),
             persistence_event=np.asarray(current,dtype=np.float32),cv_mm=np.asarray(cv,dtype=np.float32),
             split=np.asarray(partition),identities=identities)
    if not len(x) or not all(np.isfinite(out[k]).all()for k in ('x','blind','delta_mm','event')):
        raise ValueError('FINITE_ACTUAL_CONTACT_TRANSITIONS_REQUIRED')
    return out


class ContactWM(nn.Module):
    def __init__(self,dimension,hidden=64):
        super().__init__();self.register_buffer('mean',torch.zeros(dimension));self.register_buffer('scale',torch.ones(dimension))
        self.network=nn.Sequential(nn.Linear(dimension,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,7))

    def fit_normalizer(self,x):
        self.mean.copy_(x.mean(0));self.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-4))

    def forward(self,x):return self.network((x-self.mean)/self.scale)


def metrics(delta,probability,data,ids):
    p=np.asarray(probability)[ids];truth=data['event'][ids];d=np.asarray(delta)[ids]-data['delta_mm'][ids]
    positive=p>=.5
    return dict(transitions=len(ids),positive_labels=int(truth.sum()),negative_labels=int(len(ids)-truth.sum()),
                TCP_displacement_RMSE_mm=float(np.sqrt(np.mean(d[:,:3]**2))),
                object_displacement_RMSE_mm=float(np.sqrt(np.mean(d[:,3:]**2))),
                contact_Brier=float(np.mean((p-truth)**2)),contact_accuracy=float(np.mean(positive==truth)),
                true_positive=int(np.sum(positive&(truth==1))),false_positive=int(np.sum(positive&(truth==0))),
                false_negative=int(np.sum(~positive&(truth==1))))
