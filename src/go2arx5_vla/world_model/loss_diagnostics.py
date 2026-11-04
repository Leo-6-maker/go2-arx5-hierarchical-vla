"""Offline loss/gradient attribution of the exact frozen contact objective, zero optimizer."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from .contact_training import read,training_view,select,errors
from .decision_diagnostics import frozen_model
from .neural_training import parameter_hash
from .storage import digest,write_json


def components(model,b,blind):
    """Continuous +0.1 event BCE +0.1 terminal BCE, matching supervised_loss."""
    h=model.burn(b['state'][:,:8],b['reference'][:,:8],b['context'],b['clock'][:,:8]);state=b['state'][:,8]
    total=state.new_zeros((len(state),3));den=state.new_zeros(len(state));alive=torch.ones(len(state),dtype=torch.bool)
    for i in range(8,16):
        ref=torch.zeros_like(b['reference'][:,i])if blind else b['reference'][:,i]
        new,event,done,hidden=model.step(state,ref,b['context'],b['clock'][:,i],h)
        terms=torch.stack((((new-b['state'][:,i+1])/model.normalizer.std).square().mean(-1),
            .1*F.binary_cross_entropy_with_logits(event,b['events'][:,i],reduction='none').mean(-1),
            .1*F.binary_cross_entropy_with_logits(done,b['done'][:,i],reduction='none')),-1)
        mask=b['valid'][:,i];total=total+mask[:,None]*terms;den=den+mask
        state=torch.where(alive[:,None],new,state);h=torch.where(alive[:,None],hidden,h)
        alive=alive&~((b['done'][:,i]>.5)|(b['timeout'][:,i]>.5))
    return(total/den.clamp_min(1)[:,None]).mean(0)


def attribution(model,b,blind,weight):
    terms=components(model,b,blind);params=list(model.parameters());out={};gradients=[]
    for name,value in zip(('continuous','weighted_event_BCE','weighted_terminal_BCE'),terms):
        grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(p)if g is None else g).flatten()for p,g in zip(params,grads)])
        gradients.append(vector)
        out[name]=dict(mean_loss=float(value.detach()),gradient_L2=float(vector.norm()),population_weighted_gradient_L2=float(vector.norm())*weight)
    all_gradient=sum(gradients);out['summed_gradient_L2']=float(all_gradient.norm())
    out['event_continuous_cosine']=float(gradients[0]@gradients[1]/(gradients[0].norm()*gradients[1].norm()).clamp_min(1e-30))
    return out


def run(spec,output):
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(len(spec['cpus']))
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Original paired replay hash differs')
    data,meta=read(spec['replay'],'cpu');ids=np.flatnonzero(meta['partition']=='FIT');fit=select(data,torch.tensor(ids));train=training_view(fit)
    stages=[]
    for offset in (0,8):
        valid=fit['valid'][:,offset+8:offset+16].sum(-1).numpy()>0;stages.extend(meta['anchor'][ids][valid])
    stages=np.array(stages);result={}
    for arm,binding in spec['models'].items():
        model=frozen_model(binding);pin=parameter_hash(model);blind=arm=='no_future_action';arm_result={}
        for p in model.parameters():p.requires_grad_(True)
        for stage in sorted(set(stages)):
            indices=np.flatnonzero(stages==stage);total=np.zeros(3)
            with torch.no_grad():
                for offset in range(0,len(indices),16):
                    ix=indices[offset:offset+16];total+=components(model,select(train,torch.tensor(ix)),blind).numpy()*len(ix)
            # Fixed hash sample, declared before observing losses. This is an
            # exploratory gradient probe, not the exact full-population gradient.
            probe=sorted(indices,key=lambda i:hashlib.sha256(f'WM02_GRADIENT_V1/{stage}/{i}'.encode()).digest())[:4]
            b=select(train,torch.tensor(indices));mask=b['valid'][:,8:16].bool();events=b['events'][:,8:16,2]
            before=b['events'][:,7:15,2]
            arm_result[str(stage)]=dict(population_windows=len(indices),population_weight=len(indices)/len(stages),
                exact_population_mean_components=(total/len(indices)).tolist(),grasp_positive_atoms=int((events[mask]>.5).sum()),
                valid_atoms=int(mask.sum()),establishments=int(((before<.5)&(events>.5)&mask).sum()),
                releases=int(((before>.5)&(events<.5)&mask).sum()),gradient_probe_indices=[int(i)for i in probe],gradient_probe_windows=len(probe),
                gradient_probe=attribution(model,select(train,torch.tensor(probe)),blind,len(indices)/len(stages)))
        if parameter_hash(model)!=pin:raise RuntimeError('Diagnostic modified frozen weights')
        result[arm]=dict(checkpoint_sha256=binding['sha256'],parameters_unchanged=True,stages=arm_result)
        del model
    result=dict(status='FROZEN_FIT_LOSS_GRADIENT_DIAGNOSTIC',source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],models=result,
        fit_windows=len(train['state']),data_scope='Original FIT only; no DEV/holdout gradients; all FIT windows for mean loss,4 deterministic windows per stage for gradient probe',
        gradient_limit='Norm contributions are not additive or an exact full-FIT gradient; no causal claim that reweighting alone fixes prediction',
        WM_updates=0,Actor_updates=0,new_physics=0,CUDA_initialized=torch.cuda.is_initialized(),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(Path(output)/'RESULT.json',result);return result
