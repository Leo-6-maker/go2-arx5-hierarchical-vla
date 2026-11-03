"""Finite synthetic engineering smoke. No native environment or private replay is imported."""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .neural import WorldModel,Actor,Critic,STATE_DIM,SLICES,GROUPS,SCHEMA,compile_reference,part,imagine


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fixture(path,seed=20261104,count=256):
    """Analytic, deliberately nonphysical fixture, TRAIN-labelled for software testing only."""
    rng=np.random.default_rng(seed);s=torch.zeros(count,STATE_DIM)
    for name,n in GROUPS:
        if name.endswith('rotation'):s[:,SLICES[name]]=torch.eye(3).flatten()
    s[:,SLICES['base_position']]=torch.tensor([0.,0.,.4])
    s[:,SLICES['tcp_position']]=torch.from_numpy(rng.normal([.35,0,.5],.01,(count,3))).float()
    s[:,SLICES['object_position']]=part(s,'tcp_position')+torch.tensor([.02,0.,-.02])
    s[:,SLICES['gripper_fraction']]=.45
    context=torch.zeros(count,9);context[:,:3]=part(s,'object_position')+torch.tensor([.2,0,0]);context[:,3:6]=.04;context[:,6:]=torch.tensor([.05,.7,.7])
    raw=torch.from_numpy(rng.normal(0,.2,(count,2,56))).float()
    raw.reshape(count,2,8,7)[...,-1]=torch.from_numpy(rng.uniform(.35,.55,(count,2,8))).float()
    states=[s.clone()];refs=[];events=[];dones=[];masks=[];active=torch.ones(count,dtype=torch.bool)
    for i in range(16):
        if i%8==0:reference=compile_reference(s,raw[:,i//8])
        refs.append(reference.clone());node=reference[:,i%8];new=s.clone()
        dp=.4*(node[:,:3]-part(s,'tcp_position'))
        new[:,SLICES['tcp_position']]=part(s,'tcp_position')+dp
        r0=node[:,3:6];r1=node[:,6:9]
        new[:,SLICES['tcp_rotation']]=torch.stack((r0,r1,torch.linalg.cross(r0,r1)),1).flatten(-2)
        new[:,SLICES['tool_velocity']][:,:3]=dp/.05
        new[:,SLICES['gripper_fraction']]=node[:,-1:]
        new[:,SLICES['gripper_speed']]=(node[:,-1:]-part(s,'gripper_fraction'))/.05
        held=(part(new,'tcp_position')-part(s,'object_position')).norm(dim=-1)<.08
        held &= node[:,-1]<.6
        new[:,SLICES['object_position']]=part(s,'object_position')+held[:,None]*dp
        new[:,SLICES['object_velocity']][:,:3]=held[:,None]*dp/.05
        new[:,SLICES['native_progress']]=(i%8+1)/8
        new=torch.where(active[:,None],new,s)
        ev=torch.stack((held,held,held,~held),-1).float()
        # 0.8s windows are collector slices, not 32s task timeouts or terminal events.
        done=(torch.arange(count)%7==0)&(i==12)
        masks.append(active.clone());dones.append((done&active).float());events.append(ev)
        active &= ~done;s=new;states.append(s.clone())
    np.savez_compressed(path,state=torch.stack(states,1).numpy(),reference=torch.stack(refs,1).numpy(),
        context=context.numpy(),events=torch.stack(events,1).numpy(),done=torch.stack(dones,1).numpy(),
        valid=torch.stack(masks,1).numpy(),parent_id=np.arange(count),split=np.array('TRAIN'),
        purpose=np.array('SYNTHETIC_ENGINEERING_FIXTURE_NOT_PHYSICAL_DATA'),raw_action=raw.numpy(),
        terminated=torch.stack(dones,1).numpy(),truncated_timeout=np.zeros((count,16),dtype=bool),
        collector_slice=np.broadcast_to(np.arange(16)==15,(count,16)))


def read_fixture(path,device):
    with np.load(path,allow_pickle=False) as f:
        if str(f['purpose'])!='SYNTHETIC_ENGINEERING_FIXTURE_NOT_PHYSICAL_DATA' or str(f['split'])!='TRAIN':raise ValueError('Explicit synthetic-only smoke dataset required')
        return {k:torch.as_tensor(f[k],device=device) for k in ('state','reference','context','events','done','valid')}


def selected(data,cursor,size):
    index=torch.arange(cursor,cursor+size,device=data['state'].device)%len(data['state'])
    return {k:v[index] for k,v in data.items()}


def loss(model,data):
    h=model.burn(data['state'][:,:8],data['reference'][:,:8],data['context'])
    state=data['state'][:,8];per=state.new_zeros(len(state));den=state.new_zeros(len(state))
    for i in range(8,16):
        clock=state.new_tensor([i*.05/32,(i%8)*.05]).expand(len(state),2)
        state,events,done,h=model.step(state,data['reference'][:,i],data['context'],clock,h)
        continuous=((state-data['state'][:,i+1])/model.normalizer.std).square().mean(-1)
        event=F.binary_cross_entropy_with_logits(events,data['events'][:,i],reduction='none').mean(-1)
        terminal=F.binary_cross_entropy_with_logits(done,data['done'][:,i],reduction='none')
        valid=data['valid'][:,i].to(state.dtype)
        per=per+valid*(continuous+.1*(event+terminal));den=den+valid
    return (per/den.clamp_min(1)).mean()


def gradients(model,data,cursor,micro,effective=256):
    if effective%micro:raise ValueError('Exact effective sequence batch required')
    total=0.;model.zero_grad(set_to_none=True)
    for offset in range(0,effective,micro):
        value=loss(model,selected(data,cursor+offset,micro))
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite WM loss')
        (value*(micro/effective)).backward();total+=float(value.detach())*(micro/effective)
    return total


def update(model,opt,data,cursor,micro,effective=256):
    value=gradients(model,data,cursor,micro,effective)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step();return value,float(norm)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else[])


def restore_rng(r):
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


def checkpoint(path,model,opt,step,cursor,provenance,config):
    torch.save(dict(schema='WM02_ENGINEERING_CHECKPOINT_V1',model=model.state_dict(),optimizer=opt.state_dict(),
        normalizer=model.normalizer.state_dict(),rng=rng_state(),step=step,replay_cursor=cursor,
        provenance=provenance,model_config=config),path)


def load_checkpoint(path,device,expected):
    c=torch.load(path,map_location='cpu',weights_only=True)
    if c['schema']!='WM02_ENGINEERING_CHECKPOINT_V1'or c['provenance']!=expected:raise ValueError('Checkpoint input/source/schema identity differs')
    model=WorldModel(**c['model_config']).to(device);model.load_state_dict(c['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5);opt.load_state_dict(c['optimizer'])
    if tree_error(model.normalizer.state_dict(),c['normalizer'])!=0:raise ValueError('Normalizer checkpoint differs')
    restore_rng(c['rng']);return model,opt,c


def tree_error(a,b):
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Restore tensor identity differs')
        return float((a.detach().to('cpu',torch.float64)-b.detach().to('cpu',torch.float64)).abs().max())if a.numel()else 0.
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Restore dictionary differs')
        return max([tree_error(a[k],b[k]) for k in a]or[0.])
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Restore sequence differs')
        return max([tree_error(x,y)for x,y in zip(a,b)]or[0.])
    if a!=b:raise ValueError('Restore scalar differs')
    return 0.


def parameter_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def memory(device):
    if device.type!='cuda':return dict(peak_allocated_bytes=None,peak_reserved_bytes=None)
    torch.cuda.synchronize();return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def actor_smoke(model,data,folder,provenance,wm_checkpoint_sha):
    device=data['state'].device;actor=Actor().to(device);critic=Critic().to(device)
    ao=torch.optim.AdamW(actor.parameters(),lr=3e-5);co=torch.optim.AdamW(critic.parameters(),lr=3e-4)
    model.requires_grad_(False);model.zero_grad(set_to_none=True);before=parameter_hash(model);profiles=[]
    for h in (1,8,16):
        for size in (8,16):
            actor.zero_grad(set_to_none=True)
            b=selected(data,0,size)
            with torch.no_grad():hidden=model.burn(b['state'][:,:8],b['reference'][:,:8],b['context'])
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            start=time.monotonic();ret,hist,actions,decisions,end=imagine(model,actor,b['state'][:,8],b['context'],hidden,h)
            ag=torch.autograd.grad(-ret.mean(),actions,retain_graph=True)
            mg=torch.autograd.grad(part(end,'tcp_position').sum(),actions,retain_graph=True,allow_unused=True)
            (-ret.mean()).backward()
            profiles.append(dict(horizon=h,start_batch=size,actor_decisions=decisions,action_input_gradient_norm=sum(float(x.norm())for x in ag),WM_state_action_gradient_norms=[0. if x is None else float(x.norm())for x in mg],wall_s=time.monotonic()-start,**memory(device)))
            del ret,hist,actions,ag,mg,end,hidden
    actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True);grads=[];objective=0.
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();actor_before=parameter_hash(actor);critic_before=parameter_hash(critic);model_grads=[]
    for offset in range(0,64,16):
        b=selected(data,offset,16)
        with torch.no_grad():hidden=model.burn(b['state'][:,:8],b['reference'][:,:8],b['context'])
        ret,hist,actions,decisions,end=imagine(model,actor,b['state'][:,8],b['context'],hidden,16)
        loss_actor=-ret.mean()/4
        ag=torch.autograd.grad(loss_actor,actions,retain_graph=True);grads.append(sum(float(x.norm())for x in ag))
        mg=torch.autograd.grad(part(end,'tcp_position').sum(),actions,retain_graph=True,allow_unused=True)
        model_grads.append([0. if x is None else float(x.norm())for x in mg])
        loss_actor.backward();(F.mse_loss(critic(hist),ret.detach())/4).backward();objective+=float(ret.mean().detach())/4
    an=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True));cn=float(torch.nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True))
    ao.step();co.step()
    if an<=0 or min(grads)<=0 or min(g[0]for g in model_grads)<=0 or parameter_hash(model)!=before or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Actor input gradient or frozen WM contract failed')
    if parameter_hash(actor)==actor_before or parameter_hash(critic)==critic_before:raise RuntimeError('Real Actor/Critic parameters did not update')
    peak=memory(device)
    cp=folder/'actor-critic-1.pt'
    torch.save(dict(schema='WM02_SYNTHETIC_ACTOR_CRITIC_V1',actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),rng=rng_state(),provenance=provenance,WM_checkpoint_sha256=wm_checkpoint_sha,replay_cursor=51200,Actor_updates=1,critic_updates=1),cp)
    saved=torch.load(cp,map_location='cpu',weights_only=True);a2=Actor().to(device);c2=Critic().to(device);a2.load_state_dict(saved['actor']);c2.load_state_dict(saved['critic'])
    ao2=torch.optim.AdamW(a2.parameters());co2=torch.optim.AdamW(c2.parameters());ao2.load_state_dict(saved['actor_optimizer']);co2.load_state_dict(saved['critic_optimizer']);restore_rng(saved['rng'])
    error=max(tree_error(ao.state_dict(),ao2.state_dict()),tree_error(co.state_dict(),co2.state_dict()),tree_error(actor(hist),a2(hist)),tree_error(critic(hist),c2(hist)))
    result=dict(status='SYNTHETIC_ENGINEERING_ACTOR_H16_COMPLETE',objective='NEGATIVE_SYNTHETIC_OBJECT_GOAL_SQUARED_DISTANCE_MINUS_ACTION_COST_NOT_NATIVE_REWARD',Actor_updates=1,critic_updates=1,WM_updates=0,WM_parameters_bitwise_unchanged=True,WM_parameter_gradients_none=True,Actor_gradient_norm=an,critic_gradient_norm=cn,action_input_gradient_norms=grads,WM_state_action_gradient_norms=model_grads,actor_decisions=[0,8],atomic_dt_s=.05,Actor_dt_s=.4,horizon_s=.8,effective_start_batch=64,microbatch=16,profiles=profiles,checkpoint_sha256=sha(cp),restore_max_abs_error=error,wall_s=time.monotonic()-start,**peak)
    put(folder/'ACTOR_H16_ACTUAL.json',result);return result


def run_smoke(folder,spec,micro=32):
    folder=Path(folder);device=torch.device('cuda:0');started=time.monotonic();CPU=time.process_time()
    torch.set_num_threads(len(spec['cpus']));torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0);total=props.total_memory
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:
        raise PermissionError('CUDA sees a different physical device from the bound GPU0 UUID')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/total)
    fixture(folder/'SYNTHETIC_TRAIN.npz');data=read_fixture(folder/'SYNTHETIC_TRAIN.npz',device)
    provenance=dict(source_sha=spec['source_sha'],data_sha256=sha(folder/'SYNTHETIC_TRAIN.npz'),state_schema=SCHEMA,domain='SYNTHETIC_ENGINEERING_ONLY_NO_NATIVE_TRANSITIONS')
    config=dict(hidden=[2048]*5,history_hidden=256);model=WorldModel(**config).to(device)
    normalizer_mask=torch.cat((torch.ones_like(data['valid'][:,:1]),data['valid']),1)
    model.normalizer.fit(data['state'][normalizer_mask])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5);profiles=[]
    reference_gradient=None
    for mb in (8,16,32):
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();value=gradients(model,data,0,mb)
        g=torch.cat([p.grad.flatten()for p in model.parameters()if p.grad is not None])
        comparison=0. if reference_gradient is None else float((g-reference_gradient).abs().max())
        if reference_gradient is None:reference_gradient=g.detach().clone()
        tolerance=1e-4*max(1.,float(reference_gradient.abs().max()))
        if comparison>tolerance:raise RuntimeError('Microbatch accumulation gradient differs')
        profiles.append(dict(microbatch=mb,effective_sequence_batch=256,loss=value,gradient_max_abs_difference_from_micro8=comparison,optimizer_updates=0,wall_s=time.monotonic()-start,**memory(device)))
        del g
    del reference_gradient;model.zero_grad(set_to_none=True)
    initial=float(loss(model,selected(data,0,32)).detach());trace=[];torch.cuda.reset_peak_memory_stats();train_started=time.monotonic()
    for step in range(1,201):
        value,norm=update(model,opt,data,(step-1)*256,micro)
        trace.append(dict(update=step,loss=value,gradient_norm=norm,cursor=step*256))
        torch.cuda.synchronize()
        with (folder/'WM_OPTIMIZER_STEPS.jsonl').open('a') as stream:
            stream.write(json.dumps(trace[-1],allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        if step==199:checkpoint(folder/'checkpoint-199.pt',model,opt,199,199*256,provenance,config)
        if step%10==0:put(folder/'PROGRESS.json',dict(updates=step,cursor=step*256,loss=value,elapsed_s=time.monotonic()-train_started))
    peak=memory(device);checkpoint(folder/'checkpoint-200.pt',model,opt,200,51200,provenance,config)
    final=float(loss(model,selected(data,0,32)).detach());main_hash=parameter_hash(model)
    clone,clone_opt,c=load_checkpoint(folder/'checkpoint-199.pt',device,provenance)
    update(clone,clone_opt,data,c['replay_cursor'],micro)
    continuation_error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),clone_opt.state_dict()))
    if continuation_error>1e-6:raise RuntimeError('Full optimization continuation after reload differs')
    del clone,clone_opt
    reloaded,reloaded_opt,c=load_checkpoint(folder/'checkpoint-200.pt',device,provenance)
    reload_error=max(tree_error(model.state_dict(),reloaded.state_dict()),tree_error(opt.state_dict(),reloaded_opt.state_dict()))
    if c['step']!=200 or c['replay_cursor']!=51200 or main_hash!=parameter_hash(reloaded):raise RuntimeError('Endpoint identity mismatch')
    del model,opt;torch.cuda.empty_cache()
    result=dict(status='SYNTHETIC_ENGINEERING_WM200_COMPLETE',provenance=provenance,architecture=config,parameters=sum(p.numel()for p in reloaded.parameters()),main_optimizer_updates=200,restore_diagnostic_optimizer_updates=1,actual_WM_optimizer_steps=201,effective_sequence_batch=256,microbatch=micro,burn_in_atoms=8,free_rollout_atoms=8,initial_loss=initial,final_loss=final,loss_finite=bool(np.isfinite(final)),training_loss_decreased=final<initial,profiles=profiles,checkpoint_sha256=sha(folder/'checkpoint-200.pt'),reload_max_abs_error=reload_error,continuation_max_abs_error=continuation_error,normalizer_restored=True,optimizer_restored=True,RNG_restored=True,replay_cursor_restored=51200,training_wall_s=time.monotonic()-train_started,**peak)
    put(folder/'WM02_GPU200_ACTUAL.json',result);put(folder/'OPTIMIZER_TRACE.json',trace)
    actor=actor_smoke(reloaded,data,folder,provenance,result['checkpoint_sha256'])
    summary=dict(status='SYNTHETIC_ENGINEERING_SMOKE_COMPLETE',source_sha=spec['source_sha'],data_sha256=provenance['data_sha256'],native_transitions=0,WM_main_updates=200,WM_restore_probe_updates=1,Actor_updates=1,critic_updates=1,native_tasks=0,PBS_submissions=0,WM=result,Actor=actor,CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,torch=str(torch.__version__),CUDA_build=str(torch.version.cuda),physical_GPU_total_bytes=total,allocator_limit_bytes=spec['allocator_limit_bytes'])
    put(folder/'RESULT.json',summary);return summary
