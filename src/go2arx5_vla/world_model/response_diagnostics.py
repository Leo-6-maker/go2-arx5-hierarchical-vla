"""Current-V5 FIT action pathway diagnosis. No native integration or optimizer."""
from pathlib import Path
import json,time
import numpy as np
import torch
from .replay import SLICES,SCHEMA
from .decision_training import read
from .decision_diagnostics import frozen_model,stable_choice
from .contact_training import predict,select
from .neural_training import parameter_hash
from .storage import digest,write_json

EVENTS={name:i for i,name in enumerate(SCHEMA['events'])}
GRASP=EVENTS['native_grasp']


def groups(data,meta,partition='FIT',anchor='transport'):
    result=[]
    for scene in sorted(set(meta['scene'][meta['partition']==partition])):
        ids=np.flatnonzero((meta['scene']==scene)&(meta['anchor']==anchor))
        if len(ids)!=5:raise ValueError('All five declared real candidates required')
        b=select(data,torch.tensor(ids));
        for key,values in [('history',b['state'][:,:8]),('current',b['state'][:,8]),('history_ref',b['reference'][:,:8]),('context',b['context']),('history_clock',b['clock'][:,:8])]:
            if not torch.equal(values,values[0:1].expand_as(values)):raise ValueError('Candidates do not share '+key)
        names=meta['candidate'][ids].tolist()
        if set(names)!={'hold','toward_goal','away','left','right'}:raise ValueError('Physical transport candidates required')
        result.append((int(scene),ids,b,names))
    return result


def forecast(model,b,raw=None):
    return predict(model,b['state'][:,:8],b['state'][:,8],b['reference'][:,:8],b['context'],b['clock'][:,:8],b['clock'][:,8],b['raw']if raw is None else raw)


def contrast_table(pred,b,names,scene):
    rows=[];a=names.index('toward_goal');z=names.index('away');current=b['state'][:,8]
    direction=current[a,SLICES['object_position']][:2];direction=b['context'][a,:2]-direction;direction=direction/direction.norm()
    for h in (1,8,16):
        row=dict(scene=scene,horizon=h,physical_reference_difference_RMS=float((b['reference'][a,8]-b['reference'][z,8]).square().mean().sqrt()))
        for name in ('tcp_position','object_position'):
            actual=b['state'][:,8+h,SLICES[name]];p=pred[:,h-1,SLICES[name]]
            real=actual[a]-actual[z];estimated=p[a]-p[z]
            row[name]=dict(actual_difference_m=real.tolist(),predicted_difference_m=estimated.detach().tolist(),
                actual_toward_component_m=float(real[:2]@direction),predicted_toward_component_m=float(estimated[:2]@direction),
                contrast_error_m=float((estimated-real).norm()),gain=float(estimated[:2]@direction)/max(1e-12,float(real[:2]@direction)))
        rows.append(row)
    return rows


def sensitivity(model,b,names):
    """FP64 copy of exact FP32 weight values avoids finite-difference cancellation."""
    b={k:v[:1].double()for k,v in b.items()};current=b['state'][:,8];axis=b['context'][:,:2]-current[:,SLICES['object_position']][:,:2];axis=axis/axis.norm(dim=-1,keepdim=True)
    raw=b['raw'].clone();raw[...,:6]=0.;basis=torch.zeros_like(raw);basis[:,:,:2]=axis[:,None]
    alpha=torch.zeros(1,dtype=torch.float64,requires_grad=True);p,_,_=forecast(model,b,raw+alpha[:,None,None]*basis)
    result=[]
    for h in (1,8,16):
        for name in ('tcp_position','object_position'):
            y=(p[:,h-1,SLICES[name]][:,:2]*axis).sum();gradient=float(torch.autograd.grad(y,alpha,retain_graph=True)[0])
            differences={}
            with torch.no_grad():
                for eps in(1e-5,1e-4,5e-4):
                    plus=forecast(model,b,raw+eps*basis)[0][:,h-1,SLICES[name]][:,:2]
                    minus=forecast(model,b,raw-eps*basis)[0][:,h-1,SLICES[name]][:,:2]
                    differences[str(eps)]=float(((plus-minus)*axis).sum()/(2*eps))
            result.append(dict(horizon=h,output=name,autograd_m_per_m_per_atom=gradient,central_differences=differences,
                numerical_relative_error=abs(differences['1e-05']-gradient)/max(1e-12,abs(gradient)),
                interpretation='Derivative per world-metre/atom; nonzero is not physical correctness'))
    return result


def minimum_fit(data,meta):
    """One closed-form diagnostic fit on predetermined FIT parents0..3 only."""
    selected=(meta['anchor']=='transport');fit=selected&(meta['partition']=='FIT')&(meta['scene']<4)
    if int(fit.sum())!=20:raise ValueError('Freeze parents0..3 x5 before fitting')
    # All inputs are proposed commands; the labels are actual measured deltas.
    command=data['raw'][:,:,:3].sum(1).numpy();x=np.c_[np.ones(len(command)),command/.004]
    current=data['state'][:,8];ys=[]
    for h in(1,8,16):ys.extend([(data['state'][:,8+h,SLICES[n]]-current[:,SLICES[n]]).numpy()for n in('tcp_position','object_position')])
    y=np.concatenate(ys,1);coef=np.linalg.lstsq(x[fit],y[fit],rcond=None)[0];prediction=x@coef;results={}
    for part,mask in [('FIT4_training',fit),('FIT20_remainder',selected&(meta['partition']=='FIT')&~fit),('DEV8_diagnostic',selected&(meta['partition']=='DIAGNOSTIC_DEV'))]:
        rows=[]
        for hi,h in enumerate((1,8,16)):
            for ni,name in enumerate(('tcp_position','object_position')):
                s=slice((hi*2+ni)*3,(hi*2+ni+1)*3);error=np.linalg.norm(prediction[mask,s]-y[mask,s],axis=1)
                contrasts=[]
                for scene in sorted(set(meta['scene'][mask])):
                    ids=np.flatnonzero(mask&(meta['scene']==scene));names=meta['candidate'][ids].tolist();a=ids[names.index('toward_goal')];b=ids[names.index('away')]
                    true=y[a,s]-y[b,s];p=prediction[a,s]-prediction[b,s];contrasts.append(dict(scene=int(scene),true_norm_m=float(np.linalg.norm(true)),predicted_norm_m=float(np.linalg.norm(p)),positive_direction=bool(p@true>0)))
                rows.append(dict(horizon=h,output=name,RMSE_m=float(np.sqrt(np.mean(error**2))),contrasts=contrasts))
        results[part]=rows
    return dict(method='OLS, intercept and proposed cumulative world xyz/.004; one solve, no WM optimizer',FIT_parents=[0,1,2,3],training_examples=20,
        partition_results=results,coefficient_shape=list(coef.shape),closed_form_fits=1,formal_WM_updates=0)


def run(spec,output):
    output=Path(output);start=time.monotonic();cpu=time.process_time();torch.set_num_threads(len(spec['cpus']))
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Current40-parent replay hash differs')
    data,meta=read(spec['replay'],'cpu')
    keep=meta['partition']!='ONCE_HOLDOUT';data=select(data,torch.tensor(np.flatnonzero(keep)));meta={k:v[keep]for k,v in meta.items()}
    model=frozen_model(spec['models']['action_conditioned']);pin=parameter_hash(model)
    rows=[];per_candidate=[];repeat_checks=[];coupling=[]
    for scene,ids,b,names in groups(data,meta):
        with torch.no_grad():
            pred,event,hidden=forecast(model,b)
            if not torch.equal(hidden,hidden[:1].expand_as(hidden)):raise ValueError('Initial hidden differs across candidates')
            # Privileged diagnostic one-step errors use actual current state, not
            # next state. They are never used to select a deployable candidate.
            hh=hidden.clone();teacher=[]
            for k in range(8,24):
                new,_,_,hh=model.step(b['state'][:,k],b['reference'][:,k],b['context'],b['clock'][:,k],hh);teacher.append(new)
            teacher=torch.stack(teacher,1)
        np.savez_compressed(output/f'FIT-{scene:02d}-PREDICTIONS.npz',state=pred.numpy(),events=event.numpy(),teacher_one_step=teacher.numpy(),physical_raw=meta['physical_raw'][ids])
        rows.extend(contrast_table(pred,b,names,scene))
        for h in(1,8,16):
            for i,name in enumerate(names):
                values={}
                for field in('tcp_position','object_position'):
                    truth=b['state'][i,8+h,SLICES[field]];base=b['state'][i,8,SLICES[field]]
                    values[field]=dict(actual_delta_m=(truth-base).tolist(),predicted_delta_m=(pred[i,h-1,SLICES[field]]-base).tolist(),
                        endpoint_error_m=float((pred[i,h-1,SLICES[field]]-truth).norm()),teacher_one_step_error_m=float((teacher[i,h-1,SLICES[field]]-truth).norm()))
                actual=b['state'][i,8+h];context=b['context'][i]
                per_candidate.append(dict(scene=scene,candidate=name,horizon=h,outputs=values,predicted_grasp=float(event[i,h-1,GRASP]),actual_grasp=float(b['events'][i,8+h-1,GRASP]),
                    predicted_Q=float(event[i,h-1,GRASP]-10*(pred[i,h-1,SLICES['object_position']][:2]-context[:2]).norm()),
                    actual_Q=float(b['events'][i,8+h-1,GRASP]-10*(actual[SLICES['object_position']][:2]-context[:2]).norm())))
        held=(b['events'][:,8:24,GRASP]>.5).all(1)
        for i in range(len(names)):
            if not held[i]:continue
            relative=b['state'][i,24,SLICES['object_position']]-b['state'][i,24,SLICES['tcp_position']]
            before=b['state'][i,8,SLICES['object_position']]-b['state'][i,8,SLICES['tcp_position']]
            coupling.append(float((relative-before).norm()))
        if scene==0:
            singles=torch.cat([forecast(model,select(b,slice(i,i+1)))[0].detach()for i in range(5)])
            permutation=torch.tensor([4,2,0,3,1]);permuted=forecast(model,select(b,permutation))[0].detach()[torch.argsort(permutation)]
            repeat_checks.append(dict(scene=0,batch1_vs5_state_max_abs=float((singles-pred).abs().max()),permutation_state_max_abs=float((permuted-pred).abs().max())))
    fit=meta['partition']=='FIT';distributions={}
    for name,values in [('normalized_state',model.normalizer(data['state'][torch.tensor(fit),8])),('absolute_reference400',data['reference'][torch.tensor(fit),8].flatten(1)),('context5',data['context'][torch.tensor(fit)]),('clock2',data['clock'][torch.tensor(fit),8])]:
        values=values.detach();distributions[name]=dict(RMS=float(values.square().mean().sqrt()),minimum=float(values.min()),maximum=float(values.max()),feature_std_mean=float(values.std(0).mean()))
    sample=groups(data,meta)[0];double=frozen_model(spec['models']['action_conditioned']).double()
    for p in double.parameters():p.requires_grad_(False)
    gradients=sensitivity(double,sample[2],sample[3]);del double
    from .loss_diagnostics import run as loss_run
    # Current checkpoint/current720-window FIT, not the previous640-window model.
    loss_dir=output/'CURRENT_FIT_LOSS';loss_dir.mkdir();loss=loss_run(spec,loss_dir,reader=read,data_scope='Current V5 FIT720 windows; named3 phases, current fixed200 checkpoints')
    result=dict(status='CURRENT_FIT_RESPONSE_PATH_DIAGNOSED',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],model_sha256=spec['models']['action_conditioned']['sha256'],
        original_checkpoint_sha256=spec['models']['action_conditioned']['parent_checkpoint_sha256'],FIT_parents=24,FIT_transport_branches=120,
        contrast_rows=rows,candidate_rows=per_candidate,input_distributions=distributions,gradient_check=gradients,
        FP64_gradient_check='Exact FP32 parameter values promoted to FP64 for numerical derivative only; no optimizer or replaced checkpoint',
        matched_history_current_hidden=True,batch_checks=repeat_checks,
        stable_grasp_actual_relative_TCP_object=dict(eligible=len(coupling),max_drift_m=max(coupling),RMS_drift_m=float(np.sqrt(np.mean(np.square(coupling))))),
        minimum_fit=minimum_fit(data,meta),current_FIT_loss=str(Path('CURRENT_FIT_LOSS/RESULT.json')),
        WM_updates=0,Actor_updates=0,new_physics=0,holdout_inspected=False,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    if parameter_hash(model)!=pin:raise RuntimeError('Frozen diagnostic changed weights')
    write_json(output/'RESULT.json',result);return result
