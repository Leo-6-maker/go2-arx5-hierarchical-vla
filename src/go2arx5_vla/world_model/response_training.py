"""Fixed200 M1/M2 mechanism comparison on unchanged V5 FIT. No Actor optimizer."""
from pathlib import Path
import json,os,time,random,itertools,hashlib
import numpy as np
import torch
from .neural import WorldModel
from .replay import GROUPS,SCHEMA,SLICES
from .decision_training import read
from .response_diagnostics import groups,forecast,contrast_table,EVENTS
from .contact_training import training_view,select,supervised_loss
from .neural_training import checkpoint,load_checkpoint,tree_error,parameter_hash,memory
from .decision_diagnostics import frozen_model,stable_choice
from .storage import digest,write_json

P=list(range(SLICES['tcp_position'].start,SLICES['tcp_position'].stop))+list(range(SLICES['object_position'].start,SLICES['object_position'].stop))


def first_block_pairs(keys):
    groups={}
    for i,(scene,anchor,block)in enumerate(keys):
        if block==0:groups.setdefault((int(scene),str(anchor)),[]).append(i)
    return [(a,b)for ids in groups.values()for a,b in itertools.combinations(ids,2)]


def pair_loss(pred,b,pairs):
    if not pairs:return pred.new_zeros(())
    a=torch.tensor([x[0]for x in pairs],device=pred.device);c=torch.tensor([x[1]for x in pairs],device=pred.device)
    if not torch.equal(b['state'][a,:9],b['state'][c,:9])or not torch.equal(b['reference'][a,:8],b['reference'][c,:8]):
        raise ValueError('Effect supervision requires exact same observed start/history')
    delta=pred[a,-1][:,P]-pred[c,-1][:,P];real=b['state'][a,16][:,P]-b['state'][c,16][:,P]
    return((delta-real)/.004).square().mean()


def update(model,opt,train,keys,cursor,paired):
    model.zero_grad(set_to_none=True);base_total=effect_total=0.;pairs_used=0
    for offset in range(0,256,32):
        ids=np.arange(cursor+offset,cursor+offset+32)%len(train['state']);b=select(train,torch.tensor(ids,device=train['state'].device))
        base,pred=supervised_loss(model,b,False,return_prediction=True);pairs=first_block_pairs(keys[ids])if paired else[];effect=pair_loss(pred,b,pairs)
        loss=base+effect
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite response loss')
        (loss/8).backward();base_total+=float(base.detach())/8;effect_total+=float(effect.detach())/8;pairs_used+=len(pairs)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(base_loss=base_total,effect_loss=effect_total,pairs_used=pairs_used,gradient_norm=float(norm))


def evaluate(models,data,meta,output):
    results={};grasp=EVENTS['native_grasp'];output=Path(output)
    for name,model in models.items():
        records=[];contrasts=[];pin=parameter_hash(model)
        with torch.no_grad():
            for part in('FIT','DIAGNOSTIC_DEV'):
                for scene,ids,b,names in groups(data,meta,part):
                    pred,event,hidden=forecast(model,b)
                    if not torch.equal(hidden,hidden[:1].expand_as(hidden)):raise ValueError('Matched hidden differs')
                    np.savez_compressed(output/f'{name}-{part}-{scene:02d}.npz',state=pred.cpu().numpy(),events=event.cpu().numpy(),physical_raw=meta['physical_raw'][ids])
                    contrasts.extend([dict(partition=part,**r)for r in contrast_table(pred,b,names,scene)])
                    for h in(1,8,16):
                        true_dist=(b['state'][:,8+h,SLICES['object_position']][:,:2]-b['context'][:,:2]).norm(dim=-1);pred_dist=(pred[:,h-1,SLICES['object_position']][:,:2]-b['context'][:,:2]).norm(dim=-1)
                        actual=b['events'][:,8+h-1,grasp]-10*true_dist;scores=event[:,h-1,grasp]-10*pred_dist
                        raw=meta['physical_raw'][ids];index=stable_choice(scores.cpu().numpy(),raw)
                        initial=b['state'][:,8,SLICES['object_position']][:,:2].cpu().numpy();goal=b['context'][:,:2].cpu().numpy();rule=-np.linalg.norm(initial+raw[:,:,:2].sum(1)-goal,axis=1);ru=stable_choice(rule,raw)
                        records.append(dict(scene=scene,partition=part,horizon=h,candidates=names,predicted_Q=scores.cpu().tolist(),actual_Q=actual.cpu().tolist(),
                            selected=names[index],regret=float(actual.max()-actual[index]),rule_regret=float(actual.max()-actual[ru]),rule_selected=names[ru],
                            object_RMSE_m=float((pred[:,h-1,SLICES['object_position']]-b['state'][:,8+h,SLICES['object_position']]).square().sum(-1).mean().sqrt()),
                            tcp_RMSE_m=float((pred[:,h-1,SLICES['tcp_position']]-b['state'][:,8+h,SLICES['tcp_position']]).square().sum(-1).mean().sqrt())))
        if parameter_hash(model)!=pin:raise RuntimeError('Evaluation changed frozen weights')
        # Crossfit abstention: each DEV parent's margin uses only the other7
        # parents' observed pair-score errors. No claimed fresh-test coverage.
        for row in records:
            if row['partition']!='DIAGNOSTIC_DEV'or row['horizon']not in(8,16):continue
            others=[r for r in records if r['partition']=='DIAGNOSTIC_DEV'and r['scene']!=row['scene']and r['horizon']==row['horizon']]
            residuals=[]
            for other in others:
                p=np.array(other['predicted_Q']);a=np.array(other['actual_Q'])
                residuals.extend(abs((p[i]-p[j])-(a[i]-a[j]))for i,j in itertools.combinations(range(5),2))
            threshold=float(np.quantile(residuals,.9));score=np.sort(row['predicted_Q']);margin=float(score[-1]-score[-2]);fallback=margin<=threshold
            row['hybrid']=dict(calibration='Other7 DEV parents, pair score error q90',threshold=threshold,margin=margin,fallback=fallback,
                selected=row['rule_selected']if fallback else row['selected'],regret=row['rule_regret']if fallback else row['regret'])
        results[name]=dict(contrasts=contrasts,decisions=records)
    return dict(scope='Current FIT24 and diagnostic DEV8 only, no old holdout evaluation or endpoint selection',models=results,
        event_map=EVENTS,formal_reward=False,Actor_updates=0,new_physics=0)


def run(spec,output):
    output=Path(output);start=time.monotonic();cpu=time.process_time();device=torch.device('cuda:0')
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('Bound actual GPU differs')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Current V5 source data differs')
    data,meta=read(spec['replay'],device);keep=meta['partition']!='ONCE_HOLDOUT';data=select(data,torch.tensor(np.flatnonzero(keep),device=device));meta={k:v[keep]for k,v in meta.items()}
    ids=np.flatnonzero(meta['partition']=='FIT');fit=select(data,torch.tensor(ids,device=device));train=training_view(fit)
    keys=np.array([(int(s),str(a),block)for block in(0,1)for s,a in zip(meta['scene'][ids],meta['anchor'][ids])],dtype=object)
    states=fit['state'];valid=torch.cat((torch.ones_like(fit['valid'][:,:1]),fit['valid']),1).bool();normalizer=torch.unique(states[valid],dim=0)
    config=dict(hidden=[2048]*5,history_hidden=256,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1');models={};arms={};initial_pins=[]
    for name,paired in [('M1',False),('M2',True)]:
        out=output/name;out.mkdir();torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104)
        model=WorldModel(**config).to(device);model.normalizer.fit(normalizer);initial_pins.append(parameter_hash(model));opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        provenance=dict(source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],protocol='WM02_RESPONSE_M1_M2_V1',arm=name,state_schema=SCHEMA)
        torch.cuda.reset_peak_memory_stats();pairs=0;first_reload=None
        for step in range(1,201):
            if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('Finite response endpoint')
            row=update(model,opt,train,keys,(step-1)*256,paired);pairs+=row['pairs_used'];row.update(update=step,cursor=step*256)
            with(out/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            if step in(1,199,200):
                checkpoint(out/f'checkpoint-{step}.pt',model,opt,step,step*256,provenance,config)
                if step==1:
                    clone,co,c=load_checkpoint(out/'checkpoint-1.pt',device,provenance);first_reload=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
                    if first_reload:raise RuntimeError('First update reload differs')
            if step%10==0:write_json(output/f'PROGRESS-{name}-{step:04d}.json',dict(arm=name,**row))
        peak=memory(device);clone,co,c=load_checkpoint(out/'checkpoint-199.pt',device,provenance);update(clone,co,train,keys,c['replay_cursor'],paired)
        continuation=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
        reload,ro,c=load_checkpoint(out/'checkpoint-200.pt',device,provenance);endpoint=max(tree_error(model.state_dict(),reload.state_dict()),tree_error(opt.state_dict(),ro.state_dict()))
        if max(continuation,endpoint)>1e-6:raise RuntimeError('Complete response checkpoint continuation differs')
        del model,opt,ro;models[name]=reload
        arms[name]=dict(main_updates=200,continuation_probe_updates=1,base_sequence_exposures=51200,effect_pairs=pairs,first_reload_error=first_reload,endpoint_reload_error=endpoint,
            continuation_error=continuation,checkpoint_sha256=digest(out/'checkpoint-200.pt'),initial_parameter_sha256=initial_pins[-1],parameters=sum(p.numel()for p in reload.parameters()),**peak)
        write_json(out/'RESULT.json',arms[name])
    if initial_pins[0]!=initial_pins[1]:raise RuntimeError('Matched initialization differs')
    models['M0']=frozen_model(spec['M0_model'],device=device)
    evaluation=evaluate(models,data,meta,output);write_json(output/'EVALUATION.json',evaluation)
    result=dict(status='FIXED200_RESPONSE_M1_M2_COMPLETE',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],arms=arms,
        frozen_M0_parent_checkpoint=spec['M0_model']['parent_checkpoint_sha256'],same_initial_parameters=True,base_FIT_windows=len(train['state']),normalizer_unique_states=len(normalizer),
        main_WM_updates=400,continuation_probe_updates=2,Actor_updates=0,Critic_updates=0,new_physics=0,target_integrations=0,
        evaluation=evaluation,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result
