"""Bounded real S-domain prediction diagnostic; no Actor/reward or target-physics claim."""
from pathlib import Path
import json,os,random,time
import numpy as np
import torch
from torch.nn import functional as F
from .neural import WorldModel
from .replay import SCHEMA,GROUPS,SLICES,STATE_DIM
from .neural_training import checkpoint,load_checkpoint,tree_error,sha,put,memory


def read_numeric(path,device):
    with np.load(path,allow_pickle=False)as f:
        if str(f['purpose'])!='REAL_S_NATIVE_DIAGNOSTIC_REPLAY'or json.loads(str(f['schema_json']))!=json.loads(json.dumps(SCHEMA)):
            raise ValueError('Explicit real S schema required; no synthetic or target padding')
        groups=f['group'].copy();partition=f['partition'].copy();ids=f['record_ids'].copy()
        fit=partition=='FIT';hold=partition=='TRAIN_ORIGIN_HOLDOUT'
        if not fit.any()or not hold.any()or set(groups[fit])&set(groups[hold])or set(ids[fit].flat)&set(ids[hold].flat):raise ValueError('Independent physical-scene holdout required')
        data={k:torch.as_tensor(f[k],device=device,dtype=torch.float32)for k in ('state','reference','context','events','done','timeout','clock','valid')}
        if data['state'].shape[-1]!=STATE_DIM or data['reference'].shape[-2:]!=(40,10):raise ValueError('Actual S state / H40 binding differs')
        return data,fit,hold,dict(fit_groups=sorted({str(g)for g in groups[fit]}),holdout_groups=sorted({str(g)for g in groups[hold]}),manifest_sha256=str(f['manifest_sha256']))


def subset(data,index):return{k:v[index]for k,v in data.items()}


def rollout(model,b,horizon,*,shuffle=False,persistence=False):
    reference=b['reference']
    # Fixed within-partition action trace shift, including burn-in. This is a prediction
    # ablation, not an admissible physical counterfactual or policy reward.
    if shuffle:reference=reference.roll(1,0)
    hidden=model.burn(b['state'][:,:8],reference[:,:8],b['context'],b['clock'][:,:8])
    state=b['state'][:,8];states=[];events=[];done=[];alive=torch.ones(len(state),dtype=torch.bool,device=state.device)
    for i in range(8,8+horizon):
        new,e,d,h=model.step(state,reference[:,i],b['context'],b['clock'][:,i],hidden)
        if persistence:
            e=torch.logit(b['events'][:,7].clamp(1e-6,1-1e-6))
            d=torch.logit(b['done'][:,7].clamp(1e-6,1-1e-6))
        state=torch.where(alive[:,None],state if persistence else new,state)
        hidden=torch.where(alive[:,None],h,hidden)
        states.append(state);events.append(e);done.append(d)
        # Ground-truth masks are loss/continuation masks only; never model features.
        alive=alive&~((b['done'][:,i]>.5)|(b['timeout'][:,i]>.5))
    return torch.stack(states,1),torch.stack(events,1),torch.stack(done,1)


def loss(model,b):
    state,event,done=rollout(model,b,8);valid=b['valid'][:,8:16]
    continuous=((state-b['state'][:,9:17])/model.normalizer.std).square().mean(-1)
    contact=F.binary_cross_entropy_with_logits(event,b['events'][:,8:16],reduction='none').mean(-1)
    terminal=F.binary_cross_entropy_with_logits(done,b['done'][:,8:16],reduction='none')
    return ((continuous+.1*(contact+terminal))*valid).sum()/valid.sum().clamp_min(1)


def update(model,opt,train,cursor,micro=32):
    model.zero_grad(set_to_none=True);total=0.
    for offset in range(0,256,micro):
        idx=torch.arange(cursor+offset,cursor+offset+micro,device=train['state'].device)%len(train['state'])
        value=loss(model,subset(train,idx))
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite real S WM loss')
        (value*(micro/256)).backward();total+=float(value.detach())*(micro/256)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return total,float(norm)


@torch.no_grad()
def metrics(model,b,horizon,*,shuffle=False,persistence=False):
    predicted,event,done=rollout(model,b,horizon,shuffle=shuffle,persistence=persistence)
    truth=b['state'][:,9:9+horizon];mask=b['valid'][:,8:8+horizon].bool()
    out={}
    for name in ('tcp_position','object_position'):
        err=(predicted[...,SLICES[name]]-truth[...,SLICES[name]]).norm(dim=-1)
        out[name+'_RMSE_m']=float(err[mask].square().mean().sqrt())
        out[name+'_endpoint_RMSE_m']=float(err[:,-1].square().mean().sqrt())
    for name in ('tcp_rotation','object_rotation'):
        a=predicted[...,SLICES[name]].reshape(-1,horizon,3,3);c=truth[...,SLICES[name]].reshape(-1,horizon,3,3)
        cos=((a.transpose(-1,-2)@c).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        sin=(a.transpose(-1,-2)@c);v=torch.stack((sin[...,2,1]-sin[...,1,2],sin[...,0,2]-sin[...,2,0],sin[...,1,0]-sin[...,0,1]),-1).norm(dim=-1)/2
        angle=torch.atan2(v,cos.clamp(-1,1));out[name+'_RMSE_rad']=float(angle[mask].square().mean().sqrt())
    target=b['events'][:,8:8+horizon];prob=event.sigmoid()
    out['events']={name:dict(positive_targets=int(target[...,j][mask].sum()),negative_targets=int((1-target[...,j][mask]).sum()),
        brier=float((prob[...,j][mask]-target[...,j][mask]).square().mean()),
        true_positive=int(((prob[...,j]>.5)&(target[...,j]>.5)&mask).sum()),
        false_positive=int(((prob[...,j]>.5)&(target[...,j]<.5)&mask).sum()),
        false_negative=int(((prob[...,j]<.5)&(target[...,j]>.5)&mask).sum()))for j,name in enumerate(SCHEMA['events'])}
    out['terminal_positive_targets']=int(b['done'][:,8:8+horizon][mask].sum())
    out['terminal_brier']=float((done.sigmoid()[mask]-b['done'][:,8:8+horizon][mask]).square().mean())
    out['scored_transition_visits']=int(mask.sum());return out


def run(folder,spec):
    folder=Path(folder);start=time.monotonic();cpu=time.process_time();device=torch.device('cuda:0')
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('Bound GPU1 UUID differs')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
    torch.set_num_threads(len(spec['cpus']));torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('replay','eval_replay'):
        if sha(spec[name])!=spec[name+'_sha256']:raise ValueError('Numeric real replay bytes differ')
    data,fit,hold,identity=read_numeric(spec['replay'],device);evaldata,_,evalhold,evalidentity=read_numeric(spec['eval_replay'],device)
    if identity!=evalidentity:raise ValueError('Evaluation physical groups differ')
    train=subset(data,torch.as_tensor(fit,device=device));dev=subset(data,torch.as_tensor(hold,device=device));dev16=subset(evaldata,torch.as_tensor(evalhold,device=device))
    config=dict(hidden=[2048]*5,history_hidden=256,groups=GROUPS,context_dim=5)
    model=WorldModel(**config).to(device)
    # Unique FIT physical states only; holdout never participates in normalization.
    normalizer_states=torch.unique(train['state'].reshape(-1,STATE_DIM),dim=0);model.normalizer.fit(normalizer_states)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    provenance=dict(source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],eval_data_sha256=spec['eval_replay_sha256'],state_schema=SCHEMA,**identity)
    before=float(loss(model,subset(train,slice(0,32))).detach());trace=[];torch.cuda.reset_peak_memory_stats()
    endpoint=int(spec['endpoint'])
    for step in range(1,endpoint+1):
        if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('Finite real S diagnostic')
        value,norm=update(model,opt,train,(step-1)*256)
        row=dict(update=step,loss=value,gradient_norm=norm,cursor=step*256);trace.append(row)
        with (folder/'WM_OPTIMIZER_STEPS.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        if step in (1,endpoint-1):
            checkpoint(folder/f'checkpoint-{step}.pt',model,opt,step,step*256,provenance,config)
            if step==1:
                first,first_opt,first_c=load_checkpoint(folder/'checkpoint-1.pt',device,provenance)
                if max(tree_error(model.state_dict(),first.state_dict()),tree_error(opt.state_dict(),first_opt.state_dict()))!=0:raise RuntimeError('First-update reload differs')
                del first,first_opt
        if step%10==0:put(folder/'PROGRESS.json',row)
    peak=memory(device);checkpoint(folder/f'checkpoint-{endpoint}.pt',model,opt,endpoint,endpoint*256,provenance,config)
    final=float(loss(model,subset(train,slice(0,32))).detach());reload_errors=[]
    # First-update full reload is independently checked, then continuation 199→200.
    one,one_opt,c=load_checkpoint(folder/'checkpoint-1.pt',device,provenance)
    original=torch.load(folder/'checkpoint-1.pt',map_location='cpu',weights_only=True)
    reload_errors.append(max(tree_error(one.state_dict(),original['model']),tree_error(one_opt.state_dict(),original['optimizer'])));del one,one_opt,original
    clone,clone_opt,c=load_checkpoint(folder/f'checkpoint-{endpoint-1}.pt',device,provenance)
    update(clone,clone_opt,train,c['replay_cursor']);continuation=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),clone_opt.state_dict()));del clone,clone_opt
    reload,reloadopt,c=load_checkpoint(folder/f'checkpoint-{endpoint}.pt',device,provenance)
    reload_errors.append(max(tree_error(model.state_dict(),reload.state_dict()),tree_error(opt.state_dict(),reloadopt.state_dict())))
    if max(*reload_errors,continuation)>1e-6:raise RuntimeError('Real S checkpoint/optimizer continuation differs')
    del model,opt;model=reload
    evaluation={}
    # Only inspect the fixed held-out scene after the finite endpoint; never select an endpoint on it.
    for horizon,b in ((1,dev),(8,dev),(16,dev16)):
        evaluation[str(horizon)]={name:metrics(model,b,horizon,shuffle=name=='action_trace_shift',persistence=name=='state_persistence')for name in ('WM','action_trace_shift','state_persistence')}
    result=dict(status='REAL_S_WM_PREDICTION_DIAGNOSTIC_COMPLETE',provenance=provenance,
        parameters=sum(p.numel()for p in model.parameters()),architecture=config,
        fit_windows=len(train['state']),holdout_windows=len(dev['state']),holdout_H16_windows=len(dev16['state']),normalizer_unique_FIT_states=len(normalizer_states),
        main_optimizer_updates=endpoint,restore_probe_optimizer_updates=1,Actor_updates=0,critic_updates=0,
        initial_same_FIT32_loss=before,final_same_FIT32_loss=final,checkpoint_reload_errors=reload_errors,continuation_error=continuation,
        checkpoint_sha256=sha(folder/f'checkpoint-{endpoint}.pt'),evaluation=evaluation,
        evaluation_scope='One held-out original TRAIN physical scene; overlapping windows, not independent episodes or formal DEV',
        action_ablation='Fixed one-window circular shift within heldout scene, including burn-in; not physical causal qualification',
        reward_validation='NOT_COMPLETE_NO_FORMAL_ACTOR_TRAINING',native_transitions_collected=0,target_transitions=0,
        actual_GPU_uuid=spec['GPU_uuid'],CUDA_total_bytes=props.total_memory,
        torch=str(torch.__version__),CUDA_build=str(torch.version.cuda),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,**peak)
    put(folder/'RESULT.json',result);return result


def repair_legacy_metadata(source,destination,expected_sha256,expected_provenance):
    """Known project checkpoint only: allow numeric NumPy string metadata, never arbitrary pickle."""
    if sha(source)!=expected_sha256:raise ValueError('Original checkpoint SHA differs before metadata repair')
    # V1 saved np.str_ group names. Keep weights_only=True and allow just the
    # three numeric/string constructors needed for those known metadata values.
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.StrDType]):
        original=torch.load(source,map_location='cpu',weights_only=True)
    count=0
    def clean(x):
        nonlocal count
        if isinstance(x,np.generic):count+=1;return x.item()
        if isinstance(x,dict):return{k:clean(v)for k,v in x.items()}
        if isinstance(x,list):return[clean(v)for v in x]
        if isinstance(x,tuple):return tuple(clean(v)for v in x)
        return x
    fixed=clean(original)
    if fixed['provenance']!=expected_provenance:raise ValueError('Repair must retain original source/replay identity')
    if tree_error(original['model'],fixed['model'])or tree_error(original['optimizer'],fixed['optimizer']):raise RuntimeError('Metadata repair changed scientific tensors')
    with Path(destination).open('xb')as stream:torch.save(fixed,stream)
    check=torch.load(destination,map_location='cpu',weights_only=True)
    if tree_error(fixed['model'],check['model'])or tree_error(fixed['optimizer'],check['optimizer']):raise RuntimeError('Repaired checkpoint reload differs')
    return dict(original_sha256=expected_sha256,derived_sha256=sha(destination),numpy_metadata_scalars_normalized=count,
        model_and_optimizer_tensor_max_error=0.,strict_weights_only_reload=True,original_retained=True)


def finish_existing(folder,spec):
    """Finish the existing 200 endpoint after metadata repair; never repeat its main updates."""
    folder=Path(folder);start=time.monotonic();cpu=time.process_time();device=torch.device('cuda:0')
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('Bound GPU1 UUID differs')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
    torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for key in ('replay','eval_replay'):
        if sha(spec[key])!=spec[key+'_sha256']:raise ValueError('Replay differs from original diagnostic')
    data,fit,hold,identity=read_numeric(spec['replay'],device);long,_,longhold,longidentity=read_numeric(spec['eval_replay'],device)
    if identity!=longidentity:raise ValueError('Long evaluation physical groups differ')
    train=subset(data,torch.as_tensor(fit,device=device));dev=subset(data,torch.as_tensor(hold,device=device));dev16=subset(long,torch.as_tensor(longhold,device=device))
    expected=dict(source_sha=spec['original_training_source_sha'],data_sha256=spec['replay_sha256'],eval_data_sha256=spec['eval_replay_sha256'],state_schema=SCHEMA,**identity)
    repairs={}
    for step in (1,199,200):
        original=spec['original_checkpoints'][str(step)]
        repairs[str(step)]=repair_legacy_metadata(original['path'],folder/f'checkpoint-{step}.pt',original['sha256'],expected)
    put(folder/'METADATA_REPAIR.json',repairs)
    old_trace=Path(spec['original_optimizer_trace'])
    if sha(old_trace)!=spec['original_optimizer_trace_sha256']:raise ValueError('Original durable trace differs')
    trace=[json.loads(l)for l in old_trace.read_text().splitlines()]
    if [r['update']for r in trace]!=list(range(1,201)):raise ValueError('Original 200 endpoint was not complete')
    model,opt,c=load_checkpoint(folder/'checkpoint-200.pt',device,expected)
    one,oneopt,first=load_checkpoint(folder/'checkpoint-1.pt',device,expected)
    if first['step']!=1 or first['replay_cursor']!=256:raise ValueError('First checkpoint endpoint differs')
    del one,oneopt
    clone,cloneopt,previous=load_checkpoint(folder/'checkpoint-199.pt',device,expected)
    torch.cuda.reset_peak_memory_stats();update(clone,cloneopt,train,previous['replay_cursor'])
    continuation=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),cloneopt.state_dict()))
    if continuation>1e-6:raise RuntimeError('Restored 199 to 200 continuation differs')
    put(folder/'RESTORE_PROBE.json',dict(optimizer_updates=1,continuation_max_error=continuation))
    del clone,cloneopt
    # No subsequent main or Actor/critic update. All endpoints remain fixed.
    evaluation={}
    for horizon,b in ((1,dev),(8,dev),(16,dev16)):
        evaluation[str(horizon)]={name:metrics(model,b,horizon,shuffle=name=='action_trace_shift',persistence=name=='state_persistence')for name in ('WM','action_trace_shift','state_persistence')}
    result=dict(status='REAL_S_WM200_METADATA_REPAIRED_RELOAD_EVALUATION_COMPLETE',provenance=expected,evaluation_source_sha=spec['source_sha'],
        parameters=sum(p.numel()for p in model.parameters()),architecture=c['model_config'],
        fit_windows=len(train['state']),holdout_windows=len(dev['state']),holdout_H16_windows=len(dev16['state']),
        main_optimizer_updates_in_original_run=200,main_optimizer_updates_in_this_run=0,restore_probe_optimizer_updates=1,Actor_updates=0,critic_updates=0,
        original_first_batch_preupdate_loss=trace[0]['loss'],original_200th_batch_preupdate_loss=trace[-1]['loss'],final_FIT32_loss=float(loss(model,subset(train,slice(0,32))).detach()),
        initial_same_FIT32_loss='NOT_DURABLY_RECORDED_BEFORE_ORIGINAL_METADATA_FAILURE',normalizer_only_original_FIT=True,
        checkpoint_reload_max_error=0.,continuation_max_error=continuation,metadata_repairs=repairs,checkpoint_sha256=sha(folder/'checkpoint-200.pt'),evaluation=evaluation,
        evaluation_scope='One held-out original TRAIN physical scene; overlapping windows, not independent episodes or formal DEV',
        action_ablation='One-window circular reference-trace shift within heldout scene including burn-in; not admissible physical counterfactual',
        reward_validation='NOT_COMPLETE_NO_FORMAL_ACTOR_TRAINING',native_transitions_collected=0,target_transitions=0,
        actual_GPU_uuid=spec['GPU_uuid'],CUDA_total_bytes=props.total_memory,torch=str(torch.__version__),CUDA_build=str(torch.version.cuda),
        CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,**memory(device))
    put(folder/'RESULT.json',result);return result
