"""Matched finite paired S dynamics arms and action-effect evaluation, no Actor optimization."""
from pathlib import Path
import json,os,time,random,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .neural import WorldModel,so3_exp
from .neural_contract import compile_physical
from .replay import GROUPS,SLICES,STATE_DIM,SCHEMA
from .neural_training import checkpoint,load_checkpoint,tree_error,put,sha,memory


def select(data,index):return{k:v[index]for k,v in data.items()}


def read(path,device):
    with np.load(path,allow_pickle=False)as f:
        if str(f['purpose'])!='WM02_S_CONTACT_PAIRED_REPLAY_V1':raise ValueError('Paired native source input required')
        if json.loads(str(f['schema_json']))!=json.loads(json.dumps(SCHEMA)):raise ValueError('S state schema differs')
        data={k:torch.as_tensor(f[k],device=device,dtype=torch.float32)for k in ('state','reference','events','done','timeout','valid','clock','context','raw','initial_event')}
        meta={k:f[k].copy()for k in ('partition','scene','anchor','candidate')}
        expected={'FIT':set(range(16)),'DIAGNOSTIC_DEV':set(range(16,20)),'ONCE_HOLDOUT':set(range(20,24))}
        for p,s in expected.items():
            if set(meta['scene'][meta['partition']==p])!=s:raise ValueError('Predeclared whole-scene partition differs')
        return data,meta


def training_view(b):
    fields=('state','reference','events','done','timeout','valid','clock')
    batches=[]
    for offset in (0,8):
        x={k:v[:,offset:offset+(17 if k=='state' else 16)]if k in fields else v for k,v in b.items()}
        valid=x['valid'][:,8:16].sum(-1)>0;batches.append(select(x,valid))
    return{k:torch.cat([b[k]for b in batches],0)for k in b}


def supervised_loss(model,b,blind,*,return_prediction=False):
    h=model.burn(b['state'][:,:8],b['reference'][:,:8],b['context'],b['clock'][:,:8])
    state=b['state'][:,8];total=state.new_zeros(len(state));den=state.new_zeros(len(state));alive=torch.ones(len(state),dtype=torch.bool,device=state.device);predictions=[]
    for i in range(8,16):
        reference=torch.zeros_like(b['reference'][:,i])if blind else b['reference'][:,i]
        new,event,done,hidden=model.step(state,reference,b['context'],b['clock'][:,i],h)
        continuous=((new-b['state'][:,i+1])/model.normalizer.std).square().mean(-1)
        eventloss=F.binary_cross_entropy_with_logits(event,b['events'][:,i],reduction='none').mean(-1)
        terminal=F.binary_cross_entropy_with_logits(done,b['done'][:,i],reduction='none')
        mask=b['valid'][:,i];total+=mask*(continuous+.1*(eventloss+terminal));den+=mask
        state=torch.where(alive[:,None],new,state);h=torch.where(alive[:,None],hidden,h)
        alive=alive&~((b['done'][:,i]>.5)|(b['timeout'][:,i]>.5))
        if return_prediction:predictions.append(state)
    value=(total/den.clamp_min(1)).mean()
    return(value,torch.stack(predictions,1))if return_prediction else value


def update(model,opt,data,cursor,blind):
    model.zero_grad(set_to_none=True);value=0.
    for offset in range(0,256,32):
        index=torch.arange(cursor+offset,cursor+offset+32,device=data['state'].device)%len(data['state'])
        v=supervised_loss(model,select(data,index),blind)
        if not torch.isfinite(v):raise FloatingPointError('Nonfinite paired source loss')
        (v/8).backward();value+=float(v.detach())/8
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();return value,float(norm)


def pose(state):
    top=torch.cat((state[:,SLICES['tcp_rotation']].reshape(-1,3,3),state[:,SLICES['tcp_position']][:,:,None]),-1)
    return torch.cat((top,state.new_tensor([0,0,0,1]).expand(len(state),1,4)),1)


def predict(model,history,current,history_ref,context,history_clock,first_clock,raw,*,blind=False,tail=False,return_terminal=False,return_logits=False):
    """No future state/event/control-history argument exists. Second block uses predictions."""
    hidden=model.burn(history,history_ref,context,history_clock);initial_hidden=hidden.clone();state=current;pred=[];events=[];terminals=[];event_logits=[];terminal_logits=[]
    for i in range(16):
        if i%8==0:
            action=raw if i==0 else torch.cat((torch.zeros_like(raw[...,:6]),raw[:,-1:,6:7].expand(-1,8,-1)),-1)
            reference,_=compile_physical(action,pose(state),first_clock[:,0]*32+i*.05,target=False)
            if tail:
                reference=reference.clone();reference[:,8:,0]+=.002*torch.sin(torch.linspace(0,torch.pi,32,device=state.device,dtype=state.dtype))
        clock=torch.stack((first_clock[:,0]+i*.05/32,state.new_full((len(state),),(i%8)*.05)),-1)
        state,e,d,hidden=model.step(state,torch.zeros_like(reference)if blind else reference,context,clock,hidden)
        pred.append(state);events.append(e.sigmoid())
        if return_terminal:terminals.append(d.sigmoid())
        if return_logits:event_logits.append(e);terminal_logits.append(d)
    result=(torch.stack(pred,1),torch.stack(events,1),initial_hidden)
    if return_terminal:result=(*result,torch.stack(terminals,1))
    if return_logits:result=(*result,dict(event=torch.stack(event_logits,1),terminal=torch.stack(terminal_logits,1)))
    return result


def cv_prediction(current):
    out=[]
    for i in range(1,17):
        s=current.clone();t=i*.05
        for p,v in (('tcp_position','tool_velocity'),('object_position','object_velocity')):
            s[:,SLICES[p]]=current[:,SLICES[p]]+current[:,SLICES[v]][:,:3]*t
        for r,v in (('tcp_rotation','tool_velocity'),('object_rotation','object_velocity')):
            s[:,SLICES[r]]=(so3_exp(current[:,SLICES[v]][:,3:]*t)@current[:,SLICES[r]].reshape(-1,3,3)).flatten(-2)
        out.append(s)
    return torch.stack(out,1)


def errors(pred,events,b,horizon):
    actual=b['state'][:,9:9+horizon];p=pred[:,:horizon];mask=b['valid'][:,8:8+horizon].bool();out={}
    for name in ('tcp_position','object_position'):
        delta=(p[...,SLICES[name]]-actual[...,SLICES[name]]).norm(dim=-1)
        out[name]={'window_RMSE_m':float(delta[mask].square().mean().sqrt()),'endpoint_RMSE_m':float(delta[:,-1].square().mean().sqrt())}
    for name in ('tcp_rotation','object_rotation'):
        d=p[...,SLICES[name]].reshape(-1,horizon,3,3).transpose(-1,-2)@actual[...,SLICES[name]].reshape(-1,horizon,3,3)
        c=(d.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        v=torch.stack((d[...,2,1]-d[...,1,2],d[...,0,2]-d[...,2,0],d[...,1,0]-d[...,0,1]),-1).norm(dim=-1)/2
        a=torch.atan2(v,c.clamp(-1,1));out[name]={'window_RMSE_rad':float(a[mask].square().mean().sqrt()),'endpoint_RMSE_rad':float(a[:,-1].square().mean().sqrt())}
    if events is not None:
        out['event_Brier']={name:float((events[:,:horizon,j]-b['events'][:,8:8+horizon,j])[mask].square().mean())for j,name in enumerate(SCHEMA['events'])}
    out['trajectories']=len(actual);return out


def score(state,grasp,initial,category):
    # Predeclared diagnostic progress, not formal native success or Actor reward.
    if category in ('precontact','grasp_onset'):return grasp
    if category=='stable_grasp':return grasp+10*(state[...,SLICES['object_position']][...,2]-initial[...,SLICES['object_position']][...,2])
    if category=='supported_release':return(1-grasp)*state[...,SLICES['gripper_fraction']].squeeze(-1).clamp(0,1)
    raise ValueError('Unknown contact class')


def paired_metrics(pred,event,b,meta,all_meta):
    groups={}
    for i,(scene,anchor)in enumerate(zip(meta['scene'],meta['anchor'])):groups.setdefault((int(scene),str(anchor)),[]).append(i)
    pairs=[];choices=[];from_it=__import__('itertools')
    for (scene,anchor),ids in groups.items():
        initial=b['state'][ids[0],8];actual=b['state'][ids,24];realgrasp=b['events'][ids,23,2]
        predicted=pred[ids,-1];prob=event[ids,-1,2]
        rp=score(actual,realgrasp,initial,anchor);pp=score(predicted,prob,initial,anchor)
        chosen=int(torch.argmax(pp));nominal=next((j for j,i in enumerate(ids)if str(meta['candidate'][i])=='nominal'),0)
        random_index=int.from_bytes(hashlib.sha256(f'WM02_ID_V1/{scene}/{anchor}'.encode()).digest()[:4],'big')%len(ids)
        choices.append(dict(scene=scene,anchor=anchor,candidates=len(ids),selected=str(meta['candidate'][ids[chosen]]),
            selected_actual_progress=float(rp[chosen]),nominal_actual_progress=float(rp[nominal]),random_actual_progress=float(rp[random_index]),
            improvement_vs_nominal=float(rp[chosen]-rp[nominal]),improvement_vs_fixed_random=float(rp[chosen]-rp[random_index])))
        for j,k in from_it.combinations(range(len(ids)),2):
            real=actual[j,SLICES['object_position']]-actual[k,SLICES['object_position']];p=predicted[j,SLICES['object_position']]-predicted[k,SLICES['object_position']]
            norm=float(real.norm());pn=float(p.norm());dot=float(real@p)
            pairs.append(dict(scene=scene,anchor=anchor,first=str(meta['candidate'][ids[j]]),second=str(meta['candidate'][ids[k]]),
                true_object_difference_m=norm,predicted_difference_m=pn,difference_vector_error_m=float((p-real).norm()),
                informative_position_pair=norm>.001,direction_positive=bool(dot>0),direction_cosine=dot/max(1e-12,norm*pn),
                true_grasp_difference=float(realgrasp[j]-realgrasp[k]),predicted_grasp_difference=float(prob[j]-prob[k]),
                real_progress_difference=float(rp[j]-rp[k]),predicted_progress_difference=float(pp[j]-pp[k])))
    informative=[r for r in pairs if r['informative_position_pair']];eligible=16 # four heldout scenes x four declared states
    both=sum(c['improvement_vs_nominal']>1e-5 and c['improvement_vs_fixed_random']>1e-5 for c in choices)
    return dict(declared_anchor_denominator=eligible,observed_anchors=len(choices),declared_pair_denominator=160,observed_pairs=len(pairs),
        informative_position_pairs=len(informative),flat_position_pairs=len(pairs)-len(informative),
        positive_direction_informative=sum(r['direction_positive']for r in informative),
        improvement_vs_nominal_and_random_anchors=both,improvement_fraction_declared=both/eligible,
        decision_gate_60percent_pass=both/eligible>=.6,pairs=pairs,choices=choices)


@torch.no_grad()
def evaluate(models,b,meta,folder):
    args=(b['state'][:,:8],b['state'][:,8],b['reference'][:,:8],b['context'],b['clock'][:,:8],b['clock'][:,8],b['raw'])
    conditioned,blind=models;pred,event,h=predict(conditioned,*args)
    pblind,eblind,hblind=predict(blind,*args,blind=True)
    index=np.arange(len(meta['scene']))
    for scene in sorted(set(meta['scene'])):
        for anchor in sorted(set(meta['anchor'])):
            ids=np.flatnonzero((meta['scene']==scene)&(meta['anchor']==anchor))
            if len(ids):index[ids]=np.roll(ids,1)
    shifted_args=(*args[:-1],b['raw'][torch.as_tensor(index,device=b['raw'].device)])
    shifted,eshift,hshift=predict(conditioned,*shifted_args);tail,etail,htail=predict(conditioned,*args,tail=True)
    if not torch.equal(h,hshift)or not torch.equal(h,htail):raise RuntimeError('Future ablation changed real history / initial hidden state')
    # Persist all proposals and predictions before consulting future labels for ranking.
    np.savez_compressed(Path(folder)/'HOLDOUT_PREDICTIONS.npz',conditioned=pred.cpu().numpy(),no_future_action=pblind.cpu().numpy(),shifted_future=shifted.cpu().numpy(),tail_only=tail.cpu().numpy(),events=event.cpu().numpy(),raw=b['raw'].cpu().numpy())
    persist=b['state'][:,8:9].expand(-1,16,-1);cv=cv_prediction(b['state'][:,8])
    privileged=b['initial_event'][:,None].expand(-1,16,-1)
    output={}
    for horizon in (1,8,16):
        output[str(horizon)]={n:errors(p,e,b,horizon)for n,p,e in (
            ('conditioned',pred,event),('no_future_action',pblind,eblind),('future_only_shift',shifted,eshift),('tail_only',tail,etail),
            ('state_persistence',persist,None),('constant_velocity',cv,None),('event_persistence_PRIVILEGED_CURRENT_LABEL',persist,privileged))}
    perclass={}
    for anchor in sorted(set(meta['anchor'])):
        idx=torch.as_tensor(np.flatnonzero(meta['anchor']==anchor),device=pred.device);part=select(b,idx)
        perclass[str(anchor)]={n:errors(p[idx],e[idx],part,16)for n,p,e in (('conditioned',pred,event),('no_future_action',pblind,eblind),('future_only_shift',shifted,eshift))}
    original_ref,_=compile_physical(b['raw'],pose(b['state'][:,8]),b['clock'][:,8,0]*32)
    shifted_ref,_=compile_physical(b['raw'][torch.as_tensor(index,device=pred.device)],pose(b['state'][:,8]),b['clock'][:,8,0]*32)
    curve={n:[float((p[:,i,SLICES['object_position']]-b['state'][:,9+i,SLICES['object_position']]).norm(dim=-1).square().mean().sqrt())for i in range(16)]
        for n,p in (('conditioned',pred),('no_future_action',pblind),('future_only_shift',shifted),('state_persistence',persist),('constant_velocity',cv))}
    return dict(same_start_trajectory_count=len(pred),history_hidden_bitwise_same_in_ablations=True,object_endpoint_RMSE_curve_m=curve,
        horizons=output,contact_strata=perclass,paired_conditioned=paired_metrics(pred,event,b,meta,meta),paired_no_future_action=paired_metrics(pblind,eblind,b,meta,meta),
        future_only_command_contrast=dict(translation_raw_RMS_m=float((b['raw'][index,:,:3]-b['raw'][:,:,:3]).square().mean().sqrt()),rotation_raw_RMS_rad=float((b['raw'][index,:,3:6]-b['raw'][:,:,3:6]).square().mean().sqrt()),gripper_fraction_RMS=float((b['raw'][index,:,6]-b['raw'][:,:,6]).square().mean().sqrt()),
            installed_first8_world_position_vector_RMS_m=float((original_ref[:,:8,:3]-shifted_ref[:,:8,:3]).norm(dim=-1).square().mean().sqrt()),installed_first8_rotation_rows_max_abs_difference=float((original_ref[:,:8,3:9]-shifted_ref[:,:8,3:9]).abs().max())),
        tail_predicted_object_endpoint_change_RMS_m=float((tail[:,-1,SLICES['object_position']]-pred[:,-1,SLICES['object_position']]).norm(dim=-1).square().mean().sqrt()),
        event_baseline_information='Native current-label persistence is privileged relative to both learned models; matched no-future-action arm is the same-observation control',
        Actor_updates=0,formal_reward='NOT_QUALIFIED',native_task_success_claim=False)


def run(folder,spec,*,reader=read,evaluator=evaluate,protocol='WM02_S_CONTACT_IDENTIFICATION_V1',evaluate_dev=False):
    folder=Path(folder);start=time.monotonic();cpu=time.process_time();device=torch.device('cuda:0')
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('Bound physical GPU1 differs')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.set_num_threads(len(spec['cpus']))
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha(spec['replay'])!=spec['replay_sha256']:raise ValueError('Frozen paired replay differs')
    data,meta=reader(spec['replay'],device);fit=torch.as_tensor(meta['partition']=='FIT',device=device);train=training_view(select(data,fit))
    normstates=data['state'][fit];valid=torch.cat((torch.ones_like(data['valid'][fit,:1]),data['valid'][fit]),1).bool();normstates=torch.unique(normstates[valid],dim=0)
    config=dict(hidden=[2048]*5,history_hidden=256,groups=GROUPS,context_dim=5);models=[];arms={}
    for blind in (False,True):
        name='no_future_action'if blind else'action_conditioned';out=folder/name;out.mkdir()
        torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104)
        model=WorldModel(**config).to(device);model.normalizer.fit(normstates);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        provenance=dict(source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],protocol=protocol,arm=name,state_schema=SCHEMA)
        initial=float(supervised_loss(model,select(train,slice(0,32)),blind).detach());torch.cuda.reset_peak_memory_stats()
        for step in range(1,201):
            if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('Finite paired WM endpoint')
            loss,norm=update(model,opt,train,(step-1)*256,blind)
            row=dict(update=step,loss=loss,gradient_norm=norm,cursor=step*256)
            with (out/'OPTIMIZER_STEPS.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            if step in (1,199):
                checkpoint(out/f'checkpoint-{step}.pt',model,opt,step,step*256,provenance,config)
                if step==1:
                    clone,co,c=load_checkpoint(out/'checkpoint-1.pt',device,provenance)
                    if max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict())):raise RuntimeError('First-update complete reload differs')
                    del clone,co
            if step%10==0:put(folder/'PROGRESS.json',dict(arm=name,**row))
        peak=memory(device);checkpoint(out/'checkpoint-200.pt',model,opt,200,51200,provenance,config)
        clone,co,c=load_checkpoint(out/'checkpoint-199.pt',device,provenance);update(clone,co,train,c['replay_cursor'],blind)
        error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()))
        if error>1e-6:raise RuntimeError('Paired WM complete continuation differs')
        del clone,co
        final=float(supervised_loss(model,select(train,slice(0,32)),blind).detach())
        reloaded,reloaded_opt,c=load_checkpoint(out/'checkpoint-200.pt',device,provenance)
        endpoint_error=max(tree_error(model.state_dict(),reloaded.state_dict()),tree_error(opt.state_dict(),reloaded_opt.state_dict()))
        if endpoint_error>1e-6 or c['replay_cursor']!=51200:raise RuntimeError('Paired endpoint200 full reload differs')
        del model,opt,reloaded_opt;model=reloaded
        models.append(model);arms[name]=dict(main_updates=200,restore_probe_updates=1,continuation_error=error,initial_FIT32_loss=initial,final_FIT32_loss=final,
            endpoint_reload_error=endpoint_error,checkpoint_sha256=sha(out/'checkpoint-200.pt'),parameters=sum(p.numel()for p in model.parameters()),**peak)
        put(out/'RESULT.json',arms[name])
    # Both fixed arms are complete before DEV/once-heldout predictions. No
    # endpoint is selected on these labels; the optional V5 DEV is unresampled.
    if evaluate_dev:
        dev=(meta['partition']=='DIAGNOSTIC_DEV')&(data['valid'][:,8:].sum(-1).cpu().numpy()==16)
        dev_folder=folder/'DEV';dev_folder.mkdir()
        put(folder/'DEV_EVALUATION.json',evaluator(models,select(data,torch.as_tensor(dev,device=device)),{k:v[dev]for k,v in meta.items()},dev_folder))
    hold=(meta['partition']=='ONCE_HOLDOUT')&(data['valid'][:,8:].sum(-1).cpu().numpy()==16)
    b=select(data,torch.as_tensor(hold,device=device));m={k:v[hold]for k,v in meta.items()}
    evaluated=evaluator(models,b,m,folder);put(folder/'EVALUATION.json',evaluated)
    result=dict(status='PAIRED_CONTACT_WM_DIAGNOSTIC_COMPLETE',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],arms=arms,
        total_main_updates=400,total_restore_probe_updates=2,Actor_updates=0,critic_updates=0,target_transitions=0,
        evaluation=evaluated,fit_training_windows=len(train['state']),normalizer_unique_FIT_states=len(normstates),
        CUDA_total_bytes=props.total_memory,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    put(folder/'RESULT.json',result);return result
