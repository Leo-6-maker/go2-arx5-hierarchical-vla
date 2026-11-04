"""One fixed200 M3: same base exposure, multitime causal action-effect loss."""
from pathlib import Path
import json,time,os,random,itertools
import numpy as np
import torch
from .response_training import P,first_block_pairs
from .contact_training import training_view,supervised_loss,select,predict
from .temporal_diagnostics import load_model,run as diagnose
from .neural import WorldModel
from .replay import GROUPS,SCHEMA
from .decision_training import read
from .neural_training import checkpoint,load_checkpoint,tree_error,parameter_hash,memory
from .storage import digest,write_json

HORIZONS=(1,4,8,16)

def fit_scales(fit,scenes,anchors):
    pairs=first_block_pairs(np.array([(int(s),str(a),0)for s,a in zip(scenes,anchors)],dtype=object));a=[v[0]for v in pairs];b=[v[1]for v in pairs]
    scales=[];values=fit['state'].detach().cpu().numpy().astype(np.float64)
    for h in HORIZONS:
        delta=values[a,8+h][:,P]-values[b,8+h][:,P]
        scales.append(max(.0005,float(np.sqrt(np.mean(delta**2)))))
    return dict(horizons=list(HORIZONS),weights=[.25]*4,sigma_m=scales,lambda_effect=1.,sigma_floor_m=.0005,definition='FIT-only RMS over all same-start candidate pairs and six TCP/object position coordinates at each horizon; floor0.5mm; equal horizon weights sum1',FIT_pairs=len(pairs))


def multitime_effect(pred,b,pairs,config):
    if not pairs:return pred.new_zeros(()),[]
    a=torch.tensor([x[0]for x in pairs],device=pred.device);c=torch.tensor([x[1]for x in pairs],device=pred.device)
    for key,values in [('state',b['state'][:,:9]),('history_ref',b['reference'][:,:8]),('history_clock',b['clock'][:,:8]),('context',b['context'])]:
        if not torch.equal(values[a],values[c]):raise ValueError('Effect pair does not share exact start/history:'+key)
    terms=[]
    for h,sigma in zip(config['horizons'],config['sigma_m']):
        delta=pred[a,h-1][:,P]-pred[c,h-1][:,P];real=b['state'][a,8+h][:,P]-b['state'][c,8+h][:,P]
        terms.append(((delta-real)/sigma).square().mean())
    return sum(w*l for w,l in zip(config['weights'],terms))*config['lambda_effect'],[float(v.detach())for v in terms]


def update(model,opt,train,fit,keys,cursor,config):
    model.zero_grad(set_to_none=True);base_total=effect_total=0.;pair_count=0;term_total=[0.]*4
    for offset in range(0,256,32):
        ids=np.arange(cursor+offset,cursor+offset+32)%len(train['state']);b=select(train,torch.tensor(ids,device=train['state'].device));base=supervised_loss(model,b,False)
        first=ids[ids<len(fit['state'])];full=select(fit,torch.tensor(first,device=fit['state'].device));pairs=first_block_pairs(keys[first])
        if pairs:
            # Same deployed compiler as frozen evaluation. H16 reanchors from
            # predicted TCP H8; no actual future installed reference is passed.
            pred,_,_=predict(model,full['state'][:,:8],full['state'][:,8],full['reference'][:,:8],full['context'],full['clock'][:,:8],full['clock'][:,8],full['raw'])
            effect,terms=multitime_effect(pred,full,pairs,config)
            for i,v in enumerate(terms):term_total[i]+=v/8
        else:effect=base.new_zeros(())
        loss=base+effect
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite temporal loss')
        (loss/8).backward();base_total+=float(base.detach())/8;effect_total+=float(effect.detach())/8;pair_count+=len(pairs)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(base_loss=base_total,effect_loss=effect_total,effect_terms=term_total,pairs_used=pair_count,gradient_norm=float(norm))


def run(spec,output,*,update_fn=update):
    start=time.monotonic();cpu=time.process_time();output=Path(output);device=torch.device('cuda:0')
    from .runtime import cuda_uuid
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('Actual GPU UUID differs')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Original V5 differs')
    data,meta=read(spec['replay'],device);ids=np.flatnonzero(meta['partition']=='FIT');fit=select(data,torch.tensor(ids,device=device));train=training_view(fit)
    config_loss=fit_scales(fit,meta['scene'][ids],meta['anchor'][ids])
    if config_loss!=spec['temporal_loss']:raise ValueError('Predeclared FIT-only effect scales differ')
    keys=np.array([(int(s),str(a),0)for s,a in zip(meta['scene'][ids],meta['anchor'][ids])],dtype=object)
    states=fit['state'];valid=torch.cat((torch.ones_like(fit['valid'][:,:1]),fit['valid']),1).bool();normalizer=torch.unique(states[valid],dim=0)
    torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104);config=dict(hidden=[2048]*5,history_hidden=256,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1');m=WorldModel(**config).to(device);m.normalizer.fit(normalizer);initial=parameter_hash(m)
    if initial!=spec['expected_initial_parameter_sha256']:raise ValueError('Matched M1/M2 initialization differs')
    candidate=spec.get('mechanism_candidate','M3');protocol=spec.get('training_protocol','WM02_TEMPORAL_M3_V1')
    opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-5);provenance=dict(source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],protocol=protocol,arm=candidate,state_schema=SCHEMA,temporal_loss=config_loss)
    torch.cuda.reset_peak_memory_stats();pairs=0;first_reload=None
    for step in range(1,201):
        if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('Finite M3 endpoint')
        row=update_fn(m,opt,train,fit,keys,(step-1)*256,config_loss);pairs+=row['pairs_used'];row.update(update=step,cursor=step*256)
        with(output/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        if step in(1,199,200):
            checkpoint(output/f'checkpoint-{step}.pt',m,opt,step,step*256,provenance,config)
            if step==1:
                clone,co,_=load_checkpoint(output/'checkpoint-1.pt',device,provenance);first_reload=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
                if first_reload:raise RuntimeError('First reload differs')
        if step%10==0:write_json(output/f'PROGRESS-{step:04d}.json',row)
    main_peak=memory(device);clone,co,c=load_checkpoint(output/'checkpoint-199.pt',device,provenance);update_fn(clone,co,train,fit,keys,c['replay_cursor'],config_loss);continuation=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
    clone,co,_=load_checkpoint(output/'checkpoint-200.pt',device,provenance);endpoint=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
    if max(continuation,endpoint)>1e-6:raise RuntimeError('M3 continuation differs')
    result=dict(status=candidate+'_FIXED200_COMPLETE',candidate=candidate,source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],checkpoint_sha256=digest(output/'checkpoint-200.pt'),main_updates=200,continuation_probe_updates=1,base_sequence_exposures=51200,FIT_windows=len(train['state']),temporal_loss=config_loss,effect_pairs=pairs,first_reload_error=first_reload,endpoint_reload_error=endpoint,continuation_error=continuation,initial_parameter_sha256=initial,parameters=sum(p.numel()for p in m.parameters()),normalizer_unique_states=len(normalizer),new_physics=0,Actor_updates=0,Critic_updates=0,target_integrations=0,main_update_peak=main_peak,entire_run_peak=memory(device),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result
