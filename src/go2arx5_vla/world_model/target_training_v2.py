"""Finite variable-window target WM refresh; strict TRAIN/DEV and full recovery."""
from pathlib import Path
import json,time,random,hashlib,os
import numpy as np
import torch
from .target_training import TargetWM,tensors,update,save,restore,evaluate,tree_error,rng_state,restore_rng,memory,checkpoint_schema
from .target_replay import load
from .target_replay_v2 import SCHEMA,select_samples
from .target_late_decision import SCHEMA as DECISION
from .storage import digest,write_json
MIGRATION='V1_FIXED_GOAL_TO_V2_ACTUAL_WINDOW_CLOCK_NO_WEIGHT_CHANGE'


def compatible(old,new,*,migration=None):
    for k in('state_fields','joints','actuator_gear','actuator_gain'):
        if old[k]!=new[k]:raise ValueError('TARGET_STATE_ACTUATOR_LAYOUT_CHANGED:'+k)
    if old.get('schema')in(SCHEMA,DECISION)and new.get('schema')in(SCHEMA,DECISION):
        if old.get('collection_protocol')!=new.get('collection_protocol'):raise ValueError('COLLECTION_PREFIX_PROTOCOL_CHANGED_REQUIRE_FRESH_MODEL')
        if old['schema']!=new['schema']or old.get('action')!=new.get('action'):raise ValueError('ACTION_SEMANTICS_CHANGED_REQUIRE_FRESH_MODEL')
        if old['robot_fingerprint']!=new['robot_fingerprint']:raise ValueError('FROZEN_ROBOT_STRUCTURE_CHANGED')
        if old.get('executor_sha256')!=new.get('executor_sha256'):raise ValueError('NATIVE_EXECUTOR_CHANGED_REQUIRES_EXPLICIT_SEPARATE_MODEL')
    elif new.get('schema')==SCHEMA:
        if migration!=MIGRATION or old['model_sha256']!=new['original_model_sha256']:raise ValueError('EXPLICIT_FROZEN_V1_TO_V2_CLOCK_EVALUATION_MIGRATION_REQUIRED')
    elif old['model_sha256']!=new['model_sha256']:raise ValueError('LEGACY_MODEL_ASSET_CHANGED')


def partition_metrics(model,data,meta,device,*,batch=None):
    result={}
    for split in('TRAIN','DEV'):
        ids=select_samples(data,meta,split,batch=batch)
        if not len(ids):continue
        parents=sorted({meta['episodes'][int(data['episode_index'][i])]['parent']for i in ids})
        by_parent={}
        for parent in parents:
            selected=np.asarray([i for i in ids if meta['episodes'][int(data['episode_index'][i])]['parent']==parent])
            by_parent[parent]=dict(windows=len(selected),**evaluate(model,tensors(data,selected,device),meta))
        result[split]=dict(windows=len(ids),parents=len(parents),by_parent=by_parent,**evaluate(model,tensors(data,ids,device),meta))
    return result


def run(spec,output):
    from .runtime import cuda_uuid
    out=Path(output);start=time.monotonic();cpu=time.process_time();device=torch.device('cuda:0');props=torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('ACTUAL_SINGLE_TARGET_GPU_REQUIRED')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('SEALED_NIGHT1_REPLAY_CHANGED')
    data,meta=load(spec['replay'])
    if meta['schema']not in(SCHEMA,DECISION):raise ValueError('VERSIONED_WINDOW_DATA_REQUIRED')
    ids=select_samples(data,meta,'TRAIN')
    if not len(ids):raise ValueError('REAL_TRAIN_WINDOWS_REQUIRED')
    train=tensors(data,ids,device);random.seed(20261105);np.random.seed(20261105);torch.manual_seed(20261105)
    prior=None;model_meta=dict(meta)
    if spec.get('resume'):
        if digest(spec['resume'])!=spec['resume_sha256']:raise ValueError('FROZEN_REFRESH_CHECKPOINT_CHANGED')
        prior=torch.load(spec['resume'],map_location='cpu',weights_only=True)
    encoding=spec.get('action_encoding',(prior['metadata'].get('action_encoding','RAW_WORLD_RELATIVE_V1')if prior else'RAW_WORLD_RELATIVE_V1'))
    if prior and encoding!=prior['metadata'].get('action_encoding','RAW_WORLD_RELATIVE_V1'):raise ValueError('STRICT_RESUME_CANNOT_CHANGE_ACTION_FEATURES')
    if encoding!='RAW_WORLD_RELATIVE_V1':model_meta['action_encoding']=encoding
    m=TargetWM(model_meta).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);cursor=0;prior_steps=0
    recipe='FRESH_V2_MODEL_TRAIN_ONLY_NORMALIZER'
    if spec.get('resume'):
        if digest(spec['resume'])!=spec['resume_sha256']:raise ValueError('FROZEN_REFRESH_CHECKPOINT_CHANGED')
        c=prior
        if c['schema']!=checkpoint_schema(meta):raise ValueError('NO_SILENT_TRAINING_SCHEMA_MIGRATION')
        compatible(c['metadata'],meta);m.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);restore_rng(c['rng']);cursor=c['replay_cursor'];prior_steps=c['provenance'].get('prior_main_updates',0)+c['step'];recipe='STRICT_V2_CONTINUATION_FIXED_FIRST_NORMALIZER'
    else:m.fit_normalizer(train['state'])
    provenance=dict(source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],schema=meta['schema'],metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),model_metadata_sha256=hashlib.sha256(json.dumps(model_meta,sort_keys=True).encode()).hexdigest(),action_encoding=encoding,robot_fingerprint=meta['robot_fingerprint'],recipe=recipe,previous_checkpoint_sha256=spec.get('resume_sha256'),prior_main_updates=prior_steps,elapsed_feature_scale_s=1.2)
    write_json(out/'FIRST_BATCH_AUDIT.json',dict(state_shape=list(train['state'].shape),TRAIN_windows=len(ids),DEV_windows=len(select_samples(data,meta,'DEV')),state_dimension=meta['state_dimension'],TRAIN_parents=len({e['parent']for e in meta['episodes']if e['split']=='TRAIN'}),DEV_parents=len({e['parent']for e in meta['episodes']if e['split']=='DEV'}),dtype=str(train['state'].dtype),finite=bool(torch.isfinite(train['state']).all()),normalizer=recipe,software=dict(torch=str(torch.__version__),CUDA=torch.version.cuda,numpy=np.__version__),GPU_name=props.name,robot_fingerprint=meta['robot_fingerprint'],overlapping_windows_independent=False))
    write_json(out/'BEFORE_UPDATE.json',partition_metrics(m,data,meta,device))
    if spec.get('new_batch_tag'):
        if not len(select_samples(data,meta,'TRAIN',batch=spec['new_batch_tag'])):raise ValueError('PREDECLARED_NEW_TRAIN_BATCH_REQUIRED')
        write_json(out/'NEW_BATCH_BEFORE_UPDATE.json',partition_metrics(m,data,meta,device,batch=spec['new_batch_tag']))
    steps=spec['endpoint'];torch.cuda.reset_peak_memory_stats();event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);event0.record();reload_first=None
    for step in range(1,steps+1):
        if time.time()>=spec['deadline_epoch']or any((Path(spec[k])/'STOP.json').exists()for k in('registered_root','campaign_root')):raise TimeoutError('FINITE_NIGHT1_NEURAL_ENDPOINT')
        row=update(m,opt,train,cursor);cursor=row['cursor']
        with(out/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,**row))+'\n');f.flush()
        if step in(1,steps-1,steps):
            save(out/f'checkpoint-{step}.pt',m,opt,step,cursor,provenance)
            if step==1:
                clone,co,c=restore(out/'checkpoint-1.pt',device,provenance);reload_first=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()));del clone,co
    event1.record();torch.cuda.synchronize();peak=memory(device);expected_rng=rng_state()
    clone,co,c=restore(out/f'checkpoint-{steps-1}.pt',device,provenance)
    if c['step']!=steps-1 or c['replay_cursor']!=cursor-8:raise ValueError('CURSOR_BEFORE_PROBE_DIFFERS')
    probe=update(clone,co,train,c['replay_cursor']);continuation=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(expected_rng,rng_state()))
    if probe['cursor']!=cursor:raise ValueError('CURSOR_AFTER_PROBE_DIFFERS')
    del clone,co
    clone,co,c=restore(out/f'checkpoint-{steps}.pt',device,provenance);endpoint_error=max(tree_error(m.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(expected_rng,rng_state()))
    if c['replay_cursor']!=cursor or c['step']!=steps:raise ValueError('ENDPOINT_CURSOR_OR_STEP_DIFFERS')
    del clone,co
    if max(reload_first,continuation,endpoint_error)!=0:raise RuntimeError('STRICT_NIGHT1_RESTORE_DIFFERS')
    write_json(out/'AFTER_UPDATE.json',partition_metrics(m,data,meta,device))
    if spec.get('new_batch_tag'):write_json(out/'NEW_BATCH_AFTER_UPDATE.json',partition_metrics(m,data,meta,device,batch=spec['new_batch_tag']))
    result=dict(status='TARGET_WM_ENGINEERING_ENTRY_PASSED',schema=meta['schema'],source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],checkpoint_sha256=digest(out/f'checkpoint-{steps}.pt'),main_updates=steps,prior_main_updates=prior_steps,continuation_probe_updates=1,optimizer_calls=steps+1,Actor_updates=0,physics_integrations=0,replay_cursor=cursor,episodes=len(meta['episodes']),task_atoms=meta['task_atoms'],TRAIN_windows=len(ids),DEV_windows=len(select_samples(data,meta,'DEV')),parameters=sum(p.numel()for p in m.parameters()),state_dimension=meta['state_dimension'],first_reload_error=reload_first,endpoint_reload_error=endpoint_error,continuation_error=continuation,recipe=recipe,held_out_parent_generalization=meta['held_out_parent_generalization'],contact_qualification=meta['contact_collection'],GPU_uuid=spec['GPU_uuid'],main_cuda_elapsed_ms=event0.elapsed_time(event1),main_update_peak=peak,entire_run_peak=memory(device),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,predictive_value='REQUIRES_RECORDED_DEV_BASELINE_COMPARISON',control_value='NOT_TESTED')
    result['action_encoding']=encoding
    write_json(out/'RESULT.json',result);return result
