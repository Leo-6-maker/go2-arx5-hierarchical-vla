"""Current FIT/DEV time-response mechanisms; no optimizer, physics or holdout scoring."""
from pathlib import Path
import itertools,time,json,hashlib
import numpy as np
import torch
from .decision_training import read
from .decision_diagnostics import frozen_model,stable_choice,rule_scores
from .contact_training import select,cv_prediction,pose
from .response_eval_audit import canonical_forecast
from .neural_contract import compile_physical
from .neural import WorldModel
from .replay import SLICES,SCHEMA
from .storage import digest,write_json


def load_model(binding):
    if digest(binding['path'])!=binding['sha256']:raise ValueError('Frozen checkpoint identity differs')
    c=torch.load(binding['path'],map_location='cpu',weights_only=False)
    if c['step']!=200:raise ValueError('Only fixed200 endpoint')
    m=WorldModel(**c['model_config']);m.load_state_dict(c['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def rigid_object(current,pred):
    """Conditional stable-grasp baseline; never assert grasp from privileged labels."""
    r0=current[:,SLICES['tcp_rotation']].reshape(-1,3,3);r=pred[...,SLICES['tcp_rotation']].reshape(len(current),16,3,3)
    local=(r0.transpose(-1,-2)@(current[:,SLICES['object_position']]-current[:,SLICES['tcp_position']])[...,None]).squeeze(-1)
    return pred[...,SLICES['tcp_position']]+(r@local[:,None,:,None]).squeeze(-1)

def rigid_state(current,pred):
    out=pred.clone();out[...,SLICES['object_position']]=rigid_object(current,pred)
    tcp0=current[:,SLICES['tcp_rotation']].reshape(-1,3,3);obj0=current[:,SLICES['object_rotation']].reshape(-1,3,3)
    r=pred[...,SLICES['tcp_rotation']].reshape(len(current),16,3,3)
    out[...,SLICES['object_rotation']]=(r@(tcp0.transpose(-1,-2)@obj0)[:,None]).flatten(-2)
    return out


def ols_curves(data,meta):
    ids=np.flatnonzero(meta['anchor']=='transport');train=ids[(meta['partition'][ids]=='FIT')&(meta['scene'][ids]<4)]
    if len(train)!=20:raise ValueError('Predetermined four FIT parents required')
    raw=data['raw'][...,:3].sum(1).numpy();x=np.c_[np.ones(len(raw)),raw/.004];initial=data['state'][:,8]
    target=torch.cat([data['state'][:,9:25,SLICES[n]]-initial[:,None,SLICES[n]]for n in('tcp_position','object_position')],-1).numpy()
    coef=np.linalg.lstsq(x[train],target[train].reshape(20,-1),rcond=None)[0];delta=(x@coef).reshape(len(raw),16,6)
    return delta,dict(method='OLS same FIT0..3 x5 proposed cumulative xyz/.004; all16 timepoints, one matrix solve',train_parents=[0,1,2,3],train_samples=20,closed_form_solves=1)


def matched_groups(data,meta):
    for part in('FIT','DIAGNOSTIC_DEV'):
        for scene in sorted(set(meta['scene'][meta['partition']==part])):
            for anchor in('preclose','transport','supported_release'):
                ids=np.flatnonzero((meta['scene']==scene)&(meta['anchor']==anchor));b=select(data,torch.tensor(ids))
                if len(ids)!=5:raise ValueError('Five physical candidates required')
                for v in(b['state'][:,:9],b['reference'][:,:8],b['clock'][:,:8],b['context']):
                    if not torch.equal(v,v[:1].expand_as(v)):raise ValueError('Different candidate start/history')
                yield part,int(scene),anchor,ids,b,meta['candidate'][ids].tolist()


def temporal_metrics(pred,event,truth,b,names,*,part,scene,anchor,mode):
    rows=[];transport=anchor=='transport';initial=b['state'][:,8]
    axis=b['context'][0,:2]-initial[0,SLICES['object_position']][:2];axis=axis/axis.norm()
    for k in range(16):
        row=dict(partition=part,scene=scene,anchor=anchor,mode=mode,horizon=k+1)
        for n in('tcp_position','object_position'):
            error=(pred[:,k,SLICES[n]]-truth[:,k,SLICES[n]]).norm(dim=-1)
            v=dict(endpoint_RMSE_m=float(error.square().mean().sqrt()),per_candidate_error_m=error.tolist())
            if transport:
                a=names.index('toward_goal');z=names.index('away');actual=truth[a,k,SLICES[n]]-truth[z,k,SLICES[n]];p=pred[a,k,SLICES[n]]-pred[z,k,SLICES[n]]
                v.update(actual_toward_component_m=float(actual[:2]@axis),predicted_toward_component_m=float(p[:2]@axis),pair_vector_error_m=float((p-actual).norm()),actual_difference_m=actual.tolist(),predicted_difference_m=p.tolist())
            row[n]=v
        actual_rel=truth[:,k,SLICES['object_position']]-truth[:,k,SLICES['tcp_position']];pred_rel=pred[:,k,SLICES['object_position']]-pred[:,k,SLICES['tcp_position']]
        row['relative_object_TCP_error_RMSE_m']=float((pred_rel-actual_rel).square().sum(-1).mean().sqrt())
        def relative(s):return s[:,SLICES['tcp_rotation']].reshape(-1,3,3).transpose(-1,-2)@s[:,SLICES['object_rotation']].reshape(-1,3,3)
        r=relative(pred[:,k]).transpose(-1,-2)@relative(truth[:,k]);cos=(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        sin=torch.stack((r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]),-1).norm(dim=-1)/2
        row['relative_object_TCP_rotation_RMSE_rad']=float(torch.atan2(sin,cos).square().mean().sqrt())
        row['mean_predicted_world_object_minus_TCP_m']=pred_rel.mean(0).tolist();row['mean_actual_world_object_minus_TCP_m']=actual_rel.mean(0).tolist()
        if event is not None:
            row['events']={name:dict(predicted_probability=event[:,k,j].tolist(),actual=b['events'][:,8+k,j].tolist(),Brier=float((event[:,k,j]-b['events'][:,8+k,j]).square().mean()))for j,name in enumerate(SCHEMA['events'])}
        rows.append(row)
    return rows


def run(spec,output):
    start=time.monotonic();cpu=time.process_time();output=Path(output);torch.set_num_threads(len(spec['cpus']))
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Current V5 data identity differs')
    data,meta=read(spec['replay'],'cpu');keep=meta['partition']!='ONCE_HOLDOUT';data=select(data,torch.tensor(np.flatnonzero(keep)));meta={k:v[keep]for k,v in meta.items()}
    ols,ols_info=ols_curves(data,meta);models={};baseline=[];invariants=[]
    for part,scene,anchor,ids,b,names in matched_groups(data,meta):
        truth=b['state'][:,9:25];cv=cv_prediction(b['state'][:,8]);baseline.extend(temporal_metrics(cv,None,truth,b,names,part=part,scene=scene,anchor=anchor,mode='CV'))
        if anchor=='transport':
            p=truth.clone();p[...,SLICES['tcp_position']]=b['state'][:,8,None,SLICES['tcp_position']]+torch.as_tensor(ols[ids,:,:3],dtype=p.dtype);p[...,SLICES['object_position']]=b['state'][:,8,None,SLICES['object_position']]+torch.as_tensor(ols[ids,:,3:],dtype=p.dtype)
            baseline.extend(temporal_metrics(p,None,truth,b,names,part=part,scene=scene,anchor=anchor,mode='OLS_FIT4'))
            # Future true TCP only in this labelled oracle diagnostic, never in WM.
            p=rigid_state(b['state'][:,8],truth)
            baseline.extend(temporal_metrics(p,None,truth,b,names,part=part,scene=scene,anchor=anchor,mode='ORACLE_TCP_RIGID_DIAGNOSTIC'))
    for name,binding in spec['models'].items():
        model=load_model(binding);rows=[];reference_checks=[];decisions=[]
        with torch.no_grad():
            for part,scene,anchor,ids,b,names in matched_groups(data,meta):
                pred,event,hidden=canonical_forecast(model,b)
                if not torch.equal(hidden,hidden[:1].expand_as(hidden)):raise ValueError('Candidate hidden differs')
                truth=b['state'][:,9:25];teacher=[];te=[];h=hidden.clone()
                for k in range(16):
                    p,e,_,h=model.step(b['state'][:,8+k],b['reference'][:,8+k],b['context'],b['clock'][:,8+k],h);teacher.append(p);te.append(e.sigmoid())
                teacher=torch.stack(teacher,1);te=torch.stack(te,1)
                for p,e,mode in[(pred,event,'FREE_CAUSAL'),(teacher,te,'TEACHER_FORCED_PRIVILEGED')]:rows.extend(temporal_metrics(p,e,truth,b,names,part=part,scene=scene,anchor=anchor,mode=mode))
                if anchor=='transport':
                    rigid=rigid_state(b['state'][:,8],pred)
                    rows.extend(temporal_metrics(rigid,None,truth,b,names,part=part,scene=scene,anchor=anchor,mode='PREDICTED_TCP_RIGID_CONDITIONAL'))
                # Stored future reference is only compared, never given to FREE.
                first,_=compile_physical(b['raw'],pose(b['state'][:,8]),b['clock'][:,8,0]*32)
                zero=torch.cat((torch.zeros_like(b['raw'][...,:6]),b['raw'][:,-1:,6:].expand(-1,8,-1)),-1)
                second,_=compile_physical(zero,pose(pred[:,7]),b['clock'][:,8,0]*32+.4)
                reference_checks.append(dict(partition=part,scene=scene,anchor=anchor,first_compiled_vs_stored_max_abs=float((first-b['reference'][:,8]).abs().max()),second_predicted_vs_real_anchor_xyz_RMS_m=float((second[...,:3]-b['reference'][:,16,:,:3]).square().mean().sqrt()),causal_second_block='Recompiled zero EEF hold at model-predicted TCP H8, no actual future reference consumed'))
                raw=meta['physical_raw'][ids]
                if raw.dtype!=np.float64:raise ValueError('Original physical float64 command identity required')
                for k in(0,3,7,15):
                    if anchor=='preclose':q=event[:,k,2];real=b['events'][:,8+k,2]
                    elif anchor=='transport':
                        q=event[:,k,2]-10*(pred[:,k,SLICES['object_position']][:,:2]-b['context'][:,:2]).norm(dim=-1);real=b['events'][:,8+k,2]-10*(truth[:,k,SLICES['object_position']][:,:2]-b['context'][:,:2]).norm(dim=-1)
                    else:continue # No support head/labels in original V5 numeric replay.
                    if anchor=='preclose':rule=rule_scores(b['state'][0,8].numpy(),raw)
                    else:rule=-np.linalg.norm(b['state'][:,8,SLICES['object_position']][:,:2].numpy()+raw[:,:,:2].sum(1)-b['context'][:,:2].numpy(),axis=1)
                    j=stable_choice(q.numpy(),raw);ru=stable_choice(rule,raw);decisions.append(dict(partition=part,scene=scene,anchor=anchor,horizon=k+1,selected=names[j],regret_Q=float(real.max()-real[j]),rule_selected=names[ru],rule_regret_Q=float(real.max()-real[ru]),physical_command_sha256=hashlib.sha256(np.asarray(raw[j],dtype='<f8').tobytes()).hexdigest()))
                if anchor=='transport':
                    order=torch.tensor([4,2,0,3,1]);pp,ee,_=canonical_forecast(model,select(b,order));err=max(float((pp[torch.argsort(order)]-pred).abs().max()),float((ee[torch.argsort(order)]-event).abs().max()))
                    invariants.append(dict(model=name,partition=part,scene=scene,canonical_candidate_permutation_max_abs=err))
                np.savez_compressed(output/f'{name}-{part}-{scene:02d}-{anchor}.npz',free=pred.numpy(),free_events=event.numpy(),teacher_one_step=teacher.numpy(),teacher_events=te.numpy(),truth=truth.numpy(),physical_raw=raw)
        models[name]=dict(checkpoint_sha256=binding['sha256'],rows=rows,reference_checks=reference_checks,decisions=decisions)
    result=dict(source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],models=models,baselines=baseline,OLS=ols_info,invariance=invariants,named_events=SCHEMA['events'],teacher_forcing='Privileged actual states and installed references diagnostic only; never used in FREE or deployable model inputs',rigid_scope='Only actual transport stable-grasp branches; predicted TCP and oracle TCP labelled separately',new_physics=0,WM_updates=0,Actor_updates=0,no_holdout_evaluation=True,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result
