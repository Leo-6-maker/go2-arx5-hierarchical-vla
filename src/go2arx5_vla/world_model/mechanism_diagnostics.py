"""Zero-update causal loss conflicts, current-event observability and auxiliary clocks."""
from pathlib import Path
import json,time,itertools
import numpy as np
import torch
from torch.nn import functional as F
from .storage import digest,write_json
from .temporal_diagnostics import matched_groups,load_model
from .decision_training import read
from .contact_training import select,predict
from .loss_diagnostics import components
from .temporal_training import multitime_effect,fit_scales
from .response_training import first_block_pairs,P
from .neural_training import parameter_hash
from .replay import SLICES,SCHEMA


def forward(model,b):
    return predict(model,b['state'][:,:8],b['state'][:,8],b['reference'][:,:8],b['context'],b['clock'][:,:8],b['clock'][:,8],b['raw'],return_terminal=True,return_logits=True)


def observable_features(state):
    r=state[:,SLICES['tcp_rotation']].reshape(-1,3,3);local=(r.transpose(-1,-2)@(state[:,SLICES['object_position']]-state[:,SLICES['tcp_position']])[...,None]).squeeze(-1)
    rel=(state[:,SLICES['object_velocity']][:,:3]-state[:,SLICES['tool_velocity']][:,:3])
    return torch.cat((torch.ones_like(local[:,:1]),local/.01,rel/.1,state[:,SLICES['gripper_fraction']],state[:,SLICES['gripper_speed']]/.1),-1).numpy()


def event_observability(data,meta):
    # Only actual current measurements; no future event, stage, candidate name or goal.
    x=observable_features(data['state'][:,8]);y=data['initial_event'].numpy()
    if not np.array_equal(y,data['events'][:,7].numpy()):raise ValueError('Current native event label not aligned at anchor')
    fit=meta['partition']=='FIT';unique,ids=np.unique(x[fit],axis=0,return_index=True);target=y[fit][ids];coef=np.linalg.lstsq(unique,target,rcond=None)[0];p=(x@coef).clip(0,1);out={}
    contradictions=0
    groups={}
    for f,e in zip(x[fit],y[fit]):groups.setdefault(f.tobytes(),set()).add(e.tobytes())
    contradictions=sum(len(v)>1 for v in groups.values())
    for name,mask in [('FIT',fit),('DIAGNOSTIC_DEV',meta['partition']=='DIAGNOSTIC_DEV')]:
        out[name]={event:dict(Brier=float(np.mean((p[mask,j]-y[mask,j])**2)),classification_accuracy=float(np.mean((p[mask,j]>.5)==(y[mask,j]>.5))),positive_targets=int(y[mask,j].sum()),examples=int(mask.sum()))for j,event in enumerate(SCHEMA['events'])}
    return dict(method='One FIT-only closed-form linear probability fit, clip0..1; unique current measured feature rows; zero neural optimizer',features='Current local object-minus-TCP xyz/.01, relative linear velocity/.1, measured gripper fraction/speed, intercept; no future/Teacher phase/contact label input',FIT_unique_features=len(unique),FIT_exact_feature_conflicting_labels=contradictions,results=out,interpretation='Predictability on these generated current anchors, not proof that future contacts or all physical contact forces are observable from50D state')


def vector_gradient(value,model):
    params=list(model.named_parameters());grads=torch.autograd.grad(value,[p for _,p in params],retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p)if g is None else g).flatten()for (_,p),g in zip(params,grads)])


def conflicts(model,b,loss_config):
    terms=components(model,b,False);pred,event,_,terminal,logits=forward(model,b);keys=np.array([(int(i//5),'same_stage',0)for i in range(len(b['state']))],dtype=object);pairs=first_block_pairs(keys)
    effect,_=multitime_effect(pred,b,pairs,loss_config);horizons=[0,3,7,15]
    absolute=((pred[:,horizons][:,:,P]-b['state'][:,[9,12,16,24]][:,:,P])/.004).square().mean()
    event16=.1*F.binary_cross_entropy_with_logits(logits['event'][:,horizons],b['events'][:,[8,11,15,23]])
    values=dict(base_continuous=terms[0],base_weighted_event8=terms[1],base_weighted_terminal8=terms[2],M3_multitime_effect=effect,absolute_TCP_object_multitime=absolute,weighted_event16=event16)
    gradients={};numbers={}
    for key,value in values.items():
        gradients[key]=vector_gradient(value,model);numbers[key]=dict(loss=float(value.detach()),gradient_L2=float(gradients[key].norm()))
    angles={}
    for a,c in [('base_continuous','base_weighted_event8'),('M3_multitime_effect','base_weighted_event8'),('M3_multitime_effect','absolute_TCP_object_multitime'),('M3_multitime_effect','weighted_event16'),('absolute_TCP_object_multitime','weighted_event16')]:
        aa=gradients[a];cc=gradients[c];angles[a+'__'+c]=float(aa@cc/(aa.norm()*cc.norm()).clamp_min(1e-30))
    return dict(FIT_parents=[0,1,2,3],windows=len(b['state']),candidate_pairs=len(pairs),loss_gradient=numbers,gradient_cosine=angles,scope='Matched twenty first-block FIT windows in one stage; exact local gradients, not full-FIT causal attribution or optimizer update')


def run(spec,output):
    start=time.monotonic();cpu=time.process_time();output=Path(output);torch.set_num_threads(len(spec['cpus']))
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('V5 data differs')
    data,meta=read(spec['replay'],'cpu');keep=meta['partition']!='ONCE_HOLDOUT';data=select(data,torch.tensor(np.flatnonzero(keep)));meta={k:v[keep]for k,v in meta.items()}
    ids=np.flatnonzero(meta['partition']=='FIT');config=fit_scales(select(data,torch.tensor(ids)),meta['scene'][ids],meta['anchor'][ids]);models={}
    for name,binding in spec['models'].items():
        model=load_model(binding);pin=parameter_hash(model);rows=[]
        with torch.no_grad():
            for part,scene,anchor,index,b,names in matched_groups(data,meta):
                predictions=[forward(model,select(b,slice(i,i+1)))for i in range(5)];pred=torch.cat([x[0]for x in predictions]);event=torch.cat([x[1]for x in predictions]);terminal=torch.cat([x[3]for x in predictions])
                for h in range(1,17):
                    rows.append(dict(partition=part,scene=scene,anchor=anchor,horizon=h,gripper_fraction_RMSE=float((pred[:,h-1,SLICES['gripper_fraction']]-b['state'][:,8+h,SLICES['gripper_fraction']]).square().mean().sqrt()),terminal_mean_probability=float(terminal[:,h-1].mean()),terminal_positive_labels=float(b['done'][:,8+h-1].sum()),terminal_Brier=float((terminal[:,h-1]-b['done'][:,8+h-1]).square().mean()),events={ev:dict(Brier=float((event[:,h-1,j]-b['events'][:,8+h-1,j]).square().mean()),predicted_probability=event[:,h-1,j].tolist(),current_label=b['initial_event'][:,j].tolist(),future_label=b['events'][:,8+h-1,j].tolist(),privileged_current_event_persistence_Brier=float((b['initial_event'][:,j]-b['events'][:,8+h-1,j]).square().mean()))for j,ev in enumerate(SCHEMA['events'])}))
        gradient={}
        if name in('M1','M3'):
            for p in model.parameters():p.requires_grad_(True)
            for anchor in('preclose','transport','supported_release'):
                ids=np.flatnonzero((meta['partition']=='FIT')&(meta['scene']<4)&(meta['anchor']==anchor));gradient[anchor]=conflicts(model,select(data,torch.tensor(ids)),config)
        if parameter_hash(model)!=pin:raise RuntimeError('Offline diagnostics changed model')
        models[name]=dict(checkpoint_sha256=binding['sha256'],gradients=gradient,rows=rows,parameters_unchanged=True)
    result=dict(status='AUTONOMOUS_B0_MECHANISM_DIAG_COMPLETE',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],models=models,current_event_observability=event_observability(data,meta),terminal_qualification='All these local nonterminal branches have no positive terminal targets; cannot establish safety/abort detection',no_holdout_evaluation=True,WM_updates=0,Actor_updates=0,new_physics=0,CUDA_initialized=torch.cuda.is_initialized(),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result
