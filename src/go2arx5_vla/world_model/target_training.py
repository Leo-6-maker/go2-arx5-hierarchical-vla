"""Finite causal target-native WM. No Actor, simulator or S-domain weights."""
from pathlib import Path
import json,os,random,time,hashlib
import numpy as np
import torch
from torch import nn
from .target_replay import load,SCHEMA
from .storage import digest,write_json
from .neural_training import rng_state,restore_rng,tree_error,memory


def slices(meta):
    offset=0;result={}
    for field in meta['state_fields']:
        result[field['name']]=slice(offset,offset+field['size']);offset+=field['size']
    if offset!=meta['state_dimension']:raise ValueError('ACTUAL_STATE_LAYOUT_MISMATCH')
    return result


class TargetWM(nn.Module):
    def __init__(self,meta,hidden=64):
        super().__init__();self.meta=meta;self.layout=slices(meta);n=meta['state_dimension'];self.hidden=hidden
        self.action_encoding=meta.get('action_encoding','RAW_WORLD_RELATIVE_V1')
        if self.action_encoding not in('RAW_WORLD_RELATIVE_V1','SCALED_WORLD_RELATIVE_1CM_V2','NO_ACTION_V2'):raise ValueError('UNKNOWN_TARGET_ACTION_ENCODING')
        self.register_buffer('mean',torch.zeros(n));self.register_buffer('std',torch.ones(n))
        self.history=nn.GRUCell(n+20,hidden)
        self.delta=nn.Sequential(nn.Linear(n+hidden+20,128),nn.SiLU(),nn.Dropout(.05),nn.Linear(128,n))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def fit_normalizer(self,states):
        self.mean.copy_(states.mean((0,1)));floor=torch.full_like(self.std,.001)
        for name in('qvel','ctrl','tcp_rotation6d','object_rotation6d'):floor[self.layout[name]]=.01
        self.std.copy_(torch.maximum(states.std((0,1),unbiased=False),floor))
    def encode(self,state,goal,elapsed,age,known):
        action=goal.clone();action[:,:3]-=state[:,self.layout['tcp_position_m']]
        # Existing native ABI's root free position; other state fields remain raw.
        action[:,10:13]-=state[:,:3]
        if self.action_encoding=='SCALED_WORLD_RELATIVE_1CM_V2':
            action[:,:3]/=.01;action[:,10:13]/=.01
        elif self.action_encoding=='NO_ACTION_V2':action.zero_()
        return torch.cat((action,elapsed[:,None]/1.2,age[:,None],known[:,None]),-1)
    def predict(self,state,goal,source_age,window_start_s=None,window_start_atom=None):
        batch=len(state);h=state.new_zeros(batch,self.hidden)
        origin=state.new_zeros(batch)if window_start_s is None else window_start_s
        remaining=state.new_full((batch,),6.)if window_start_atom is None else 7-torch.remainder(window_start_atom+8,7)
        for i in range(8):
            elapsed=origin+i*.05
            a=self.encode(state[:,i],goal,elapsed,source_age[:,i],state.new_ones(batch))
            h=self.history(torch.cat(((state[:,i]-self.mean)/self.std,a),-1),h)
        current=state[:,8];pred=[]
        for i in range(8,24):
            # The next renewal is at atom14. Its actual future source age is
            # unknown; an explicit missing mask replaces that input thereafter.
            known=(remaining>(i-8)).to(state.dtype);age=(source_age[:,8]+(i-8)*.05)*known
            a=self.encode(current,goal,origin+i*.05,age,known)
            normalized=(current-self.mean)/self.std
            h=self.history(torch.cat((normalized,a),-1),h)
            current=current+self.delta(torch.cat((normalized,h,a),-1))*self.std
            pred.append(current)
        return torch.stack(pred,1)


def tensors(data,ids,device):
    return {k:torch.as_tensor(data[k][ids],device=device,dtype=torch.float32)for k in('state','goal','source_age','window_start_s','window_start_atom')if k in data}


def update(model,opt,data,cursor,batch=8):
    model.train();ids=torch.randint(len(data['state']),(batch,),device=data['state'].device)
    b={k:v[ids]for k,v in data.items()};opt.zero_grad(set_to_none=True)
    prediction=model.predict(**b)
    loss=((prediction-b['state'][:,9:])/model.std).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_TARGET_LOSS')
    loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),cursor=cursor+batch)


def save(path,model,opt,step,cursor,provenance):
    schema='WM02_TARGET_WM_CHECKPOINT_V2'if model.meta.get('schema')=='WM02_TARGET_NATIVE_REPLAY_V2'else'WM02_TARGET_WM_CHECKPOINT_V1'
    torch.save(dict(schema=schema,model=model.state_dict(),optimizer=opt.state_dict(),normalizer=dict(mean=model.mean,std=model.std),rng=rng_state(),step=step,replay_cursor=cursor,provenance=provenance,metadata=model.meta,hidden=model.hidden),path)


def restore(path,device,expected):
    value=torch.load(path,map_location='cpu',weights_only=True)
    schema='WM02_TARGET_WM_CHECKPOINT_V2'if value['metadata'].get('schema')=='WM02_TARGET_NATIVE_REPLAY_V2'else'WM02_TARGET_WM_CHECKPOINT_V1'
    if value['schema']!=schema or value['provenance']!=expected:raise ValueError('TARGET_CHECKPOINT_IDENTITY_DIFFERS')
    m=TargetWM(value['metadata'],value['hidden']).to(device);m.load_state_dict(value['model'])
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);opt.load_state_dict(value['optimizer'])
    if tree_error(dict(mean=m.mean,std=m.std),value['normalizer']):raise ValueError('TARGET_NORMALIZER_DIFFERS')
    restore_rng(value['rng']);return m,opt,value


def evaluate(model,data,meta):
    model.eval()
    with torch.no_grad():pred=model.predict(**data)
    return prediction_metrics(pred,data,meta)


def rotation6d_angle(value,truth):
    """Gram-Schmidt decoding for offline SO3 diagnostics, never control."""
    def decode(x):
        a=x[:,:3];b=x[:,3:];na=a.norm(dim=-1,keepdim=True);u=a/na.clamp_min(1e-8);w=b-(u*b).sum(-1,keepdim=True)*u;nw=w.norm(dim=-1,keepdim=True);v=w/nw.clamp_min(1e-8)
        return torch.stack((u,v,torch.linalg.cross(u,v,dim=-1)),-1),(na[:,0]>1e-8)&(nw[:,0]>1e-8)
    a,va=decode(value);b,vb=decode(truth);relative=a.transpose(-1,-2)@b
    skew=torch.stack((relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]),-1)/2
    angle=torch.atan2(skew.norm(dim=-1),((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))
    return angle,va&vb


def prediction_metrics(pred,data,meta):
    groups=slices(meta);metrics={}
    q0=groups['qpos'].start;v0=groups['qvel'].start
    for j in meta.get('joints',[]):
        q=q0+j['qpos_address'];v=v0+j['qvel_address'];name=j['name']or str(q)
        if j['type']==0:
            groups[name+'_position_m']=slice(q,q+3);groups[name+'_quaternion_dimensionless']=slice(q+3,q+7);groups[name+'_linear_m_s']=slice(v,v+3);groups[name+'_angular_rad_s']=slice(v+3,v+6)
        elif j['type']in(2,3):
            groups[name+('_m'if j['type']==2 else'_rad')]=slice(q,q+1);groups[name+('_m_s'if j['type']==2 else'_rad_s')]=slice(v,v+1)
    # Honest engineering baselines. CV is local numeric velocity over the two
    # last real states, includes aliased TCP/object; quaternion components are
    # not interpreted as an integrated orientation.
    anchor=data['state'][:,8];velocity=(anchor-data['state'][:,7])/.05
    for h in(1,8,16):
        real=data['state'][:,8+h];persistence=anchor;cv=anchor+velocity*(h*.05)
        metrics[str(h)]={name:dict(model_RMSE=float((pred[:,h-1,sl]-real[:,sl]).square().mean().sqrt()),persistence_RMSE=float((persistence[:,sl]-real[:,sl]).square().mean().sqrt()),local_numeric_velocity_RMSE=float((cv[:,sl]-real[:,sl]).square().mean().sqrt()))for name,sl in groups.items()}
        for name,sl in list(groups.items()):
            if not name.endswith('_quaternion_dimensionless'):continue
            truth=real[:,sl];valid_truth=truth.norm(dim=-1)>1e-8;truth=truth/truth.norm(dim=-1,keepdim=True).clamp_min(1e-8);row={}
            for label,value in(('model',pred[:,h-1,sl]),('persistence',persistence[:,sl]),('local_numeric_velocity',cv[:,sl])):
                valid=valid_truth&(value.norm(dim=-1)>1e-8);unit=value/value.norm(dim=-1,keepdim=True).clamp_min(1e-8);sign=torch.where((unit*truth).sum(-1,keepdim=True)<0,-1.,1.);aligned=truth*sign;angle=4*torch.atan2((unit-aligned).norm(dim=-1),(unit+aligned).norm(dim=-1));row[label+'_RMSE']=float(angle[valid].square().mean().sqrt())if valid.any()else None;row[label+'_invalid_count']=int((~valid).sum())
            metrics[str(h)][name.replace('_quaternion_dimensionless','_SO3_rad')]=row
        for name in('tcp_rotation6d','object_rotation6d'):
            sl=groups[name];row={}
            for label,value in(('model',pred[:,h-1,sl]),('persistence',persistence[:,sl]),('local_numeric_velocity',cv[:,sl])):
                angle,valid=rotation6d_angle(value,real[:,sl]);row[label+'_RMSE']=float(angle[valid].square().mean().sqrt())if valid.any()else None;row[label+'_invalid_count']=int((~valid).sum())
            metrics[str(h)][name.replace('_rotation6d','_SO3_rad')]=row
    return dict(metrics=metrics,evaluation='CAUSAL_FIXED_GOAL_OPEN_LOOP_BURN8_PREDICT16',mixed_unit_aggregates=['qpos','qvel','ctrl'],CV_orientation='Numeric component extrapolation only, not integrated SO3',event_heads='NOT_TRAINED_NO_QUALIFIED_CONTACT_POSITIVES',controller_hidden_state='MISSING_HISTORY_ONLY_NO_MARKOV_CLAIM')


def run(spec,output):
    from .runtime import cuda_uuid
    start=time.monotonic();cpu=time.process_time();out=Path(output);device=torch.device('cuda:0')
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('ACTUAL_TARGET_GPU_UUID_DIFFERS')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('ACTUAL_TARGET_REPLAY_CHANGED')
    data,meta=load(spec['replay']);ids=np.array([i for i,e in enumerate(meta['episodes'])if e['split']=='TRAIN'])
    if not len(ids):raise ValueError('REAL_TRAIN_EPISODES_REQUIRED')
    train=tensors(data,ids,device);provenance=dict(source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],schema=SCHEMA,metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),scope='AUTO1_TARGET_FREE_SPACE_MODEL_ONLY',previous_checkpoint_sha256=spec.get('resume_sha256'))
    write_json(out/'FIRST_BATCH_AUDIT.json',dict(state_shape=list(train['state'].shape),state_dtype=str(train['state'].dtype),goal_shape=list(train['goal'].shape),source_age_shape=list(train['source_age'].shape),state_dimension=meta['state_dimension'],finite=bool(torch.isfinite(train['state']).all()),TRAIN_episodes=len(ids),DEV_episodes=sum(e['split']=='DEV'for e in meta['episodes']),software=dict(torch=str(torch.__version__),CUDA=torch.version.cuda,numpy=np.__version__),device_name=props.name,model_scene_sha256=meta['model_sha256'],controller_state='HISTORY_WITH_EXPLICIT_MISSING_STATE_NOT_COMPLETE_MARKOV'))
    random.seed(20261104);np.random.seed(20261104);torch.manual_seed(20261104)
    m=TargetWM(meta).to(device)
    if not spec.get('resume'):m.fit_normalizer(train['state'])
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);cursor=0
    if spec.get('resume'):
        if digest(spec['resume'])!=spec['resume_sha256']:raise ValueError('FROZEN_FIRST_TARGET_CHECKPOINT_CHANGED')
        old=torch.load(spec['resume'],map_location='cpu',weights_only=True)
        # Dataset identity changes for B; model layout and existing TRAIN-only
        # normalizer stay fixed. Full optimizer/RNG/cursor are carried forward.
        if old['metadata']['state_fields']!=meta['state_fields']or old['metadata']['model_sha256']!=meta['model_sha256']:raise ValueError('INCOMPATIBLE_TARGET_EXECUTOR_LAYOUT')
        m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);restore_rng(old['rng']);cursor=old['replay_cursor']
    if spec.get('baseline_fit_replay'):
        import subprocess
        child_env=dict(os.environ,PYTHONPATH=str(Path(spec['source_root'])/'src')+os.pathsep+spec['source_root'])
        q=subprocess.run([spec['python'],'-B','-m','scripts.wm02.fit_target_baselines','--replay',spec['baseline_fit_replay'],'--eval-replay',spec['baseline_evaluation_replay'],'--output',str(out/'reference-residual-baseline')],cwd=spec['source_root'],env=child_env,capture_output=True,text=True,timeout=30)
        write_json(out/'BASELINE_COMMAND.json',dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr));q.check_returncode()
    steps=spec['endpoint'];initial_eval=evaluate(m,train,meta);write_json(out/'BEFORE_UPDATE.json',initial_eval)
    new_ids=np.array([i for i,e in enumerate(meta['episodes'])if e['episode'].startswith('collect-target-b-')],dtype=int)
    if spec.get('resume'):
        if len(new_ids)!=4:raise ValueError('FOUR_NEW_PREDECLARED_TRAIN_ATTEMPTS_REQUIRED')
        write_json(out/'NEW_BATCH_BEFORE_UPDATE.json',evaluate(m,tensors(data,new_ids,device),meta))
    torch.cuda.reset_peak_memory_stats();event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);event0.record();initial_rng=rng_state();first_reload=None
    for step in range(1,steps+1):
        if time.time()>=spec['deadline_epoch']or any((Path(spec[k])/'STOP.json').exists()for k in('registered_root','campaign_root')):raise TimeoutError('FINITE_TARGET_NEURAL_ENDPOINT')
        row=update(m,opt,train,cursor);cursor=row['cursor']
        with(out/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,**row))+'\n');f.flush()
        if step in(1,steps-1,steps):
            save(out/f'checkpoint-{step}.pt',m,opt,step,cursor,provenance)
            if step==1:
                clone,co,c=restore(out/'checkpoint-1.pt',device,provenance);first_reload=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
    event1.record();torch.cuda.synchronize();main_memory=memory(device)
    # One genuine independent optimizer call, from (endpoint-1) to endpoint.
    expected_rng=rng_state()
    clone,co,c=restore(out/f'checkpoint-{steps-1}.pt',device,provenance)
    if c['step']!=steps-1 or c['replay_cursor']!=cursor-8:raise ValueError('TARGET_REPLAY_CURSOR_DIFFERS_BEFORE_PROBE')
    probe=update(clone,co,train,c['replay_cursor']);continuation=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(expected_rng,rng_state()))
    if probe['cursor']!=cursor:raise ValueError('TARGET_REPLAY_CURSOR_DIFFERS_AFTER_PROBE')
    del clone,co
    clone,co,c=restore(out/f'checkpoint-{steps}.pt',device,provenance)
    if c['step']!=steps or c['replay_cursor']!=cursor:raise ValueError('TARGET_ENDPOINT_CURSOR_DIFFERS')
    endpoint_error=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(expected_rng,rng_state()));del clone,co
    if max(first_reload,continuation,endpoint_error)!=0:raise RuntimeError('TARGET_STRICT_RESTORE_DIFFERS')
    after=evaluate(m,train,meta);write_json(out/'AFTER_UPDATE.json',after)
    result=dict(status='TARGET_WM_ENGINEERING_ENTRY_PASSED',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],checkpoint_sha256=digest(out/f'checkpoint-{steps}.pt'),main_updates=steps,continuation_probe_updates=1,Actor_updates=0,physics_integrations=0,episodes=len(ids),task_atoms=len(ids)*24,parameters=sum(p.numel()for p in m.parameters()),state_dimension=meta['state_dimension'],first_reload_error=first_reload,endpoint_reload_error=endpoint_error,continuation_error=continuation,normalizer='TRAIN_ONLY_FIXED_ON_FIRST_BATCH',held_out_parent_generalization=meta['held_out_parent_generalization'],event_heads='NOT_TRAINED',GPU_uuid=spec['GPU_uuid'],main_cuda_elapsed_ms=event0.elapsed_time(event1),main_update_peak=main_memory,entire_run_peak=memory(device),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,contact_qualification='NOT_QUALIFIED',predictive_value='REPORT_METRICS_NOT_ESTABLISHED_BY_ENTRY_PASS')
    write_json(out/'RESULT.json',result);return result
