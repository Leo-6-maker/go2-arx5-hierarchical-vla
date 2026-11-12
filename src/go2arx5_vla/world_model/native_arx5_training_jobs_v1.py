"""Finite scientific operations inside the existing Core owner, never a scheduler."""
from __future__ import annotations
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

from . import native_arx5_training_profile_v1 as profile
from . import native_arx5_training_runtime_v1 as rt
from . import task_learning_features_v1 as features
from .storage import digest


def visit_identity(visits):
    return hashlib.sha256(visits.astype('<i8').tobytes()).hexdigest()


def disk_rows(pin):
    import torch
    path=Path(pin['path']);rt.require(digest(path)==pin['sha256'],'NEW_CACHE_PIN')
    manifest=json.loads(path.read_bytes());admitted={}
    rt.require(manifest['schema']==rt.CACHE_SCHEMA and manifest['parent_files_identity']==profile.PARENT_ID,
               'NEW_OFFICIAL_CACHE')
    def get(index):
        if index not in admitted:
            row=manifest['rows'][index];p=path.parent/row['file']
            rt.require(p.resolve().is_relative_to(path.parent.resolve())and digest(p)==row['sha256'],'EXACT_FEATURE_ROW')
            admitted[index]=torch.load(p,map_location='cpu',weights_only=False)
            rt.require(admitted[index]['identity']==row['identity'],'FEATURE_ROW_OBSERVATION_IDENTITY')
        return admitted[index]
    return get


def generation_equivalence(model,data,rows,indices,root,guard):
    import torch
    import numpy as np
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import tree_identity
    records=[]
    for ordinal,index in enumerate(indices):
        guard();inputs=data.processor.collator([data[index]['processed']])['inputs'];row=rows(index)
        seed=2026111200+ordinal;outputs=[]
        for route in ('ORDINARY_FULL_RGB','CACHED_BACKBONE'):
            name=f'query-{ordinal:02d}-{route}'
            write(root/(name+'-INTENT.private.json'),dict(index=index,route=route,seed=seed,
                actual_model_invocation_requested=1,input_identity=tree_identity(inputs)))
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(seed)
                value=(model.get_action(features.inference_inputs(inputs))if route=='ORDINARY_FULL_RGB'else
                    model.action_head.get_action(features.feature_packet(row),
                        features.action_packet(model,row,model.dtype,generation=True)))['action_pred']
                outputs.append(value.detach().cpu())
            rt.require(outputs[-1].shape==(1,40,132)and bool(torch.isfinite(outputs[-1]).all()),
                       'FINITE_COMPLETE_H40_GENERATION')
            np.save(root/(name+'.npy'),outputs[-1].float().numpy(),allow_pickle=False)
        exact=torch.equal(*outputs);records.append(dict(index=index,seed=seed,exact=exact))
        rt.require(exact,'FULL_RGB_AND_FROZEN_BACKBONE_H40_EXACT')
    return dict(records=records,actual_full_RGB_generations=len(records),
                actual_cached_generations=len(records),passes=True)


def trace_writer(root):
    def record(value):
        with (root/'ACTUAL_CALLS.jsonl').open('a')as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
    return record


def full_cache_update_qualification(s,model,contract,data,frozen,root,guard):
    """Compare a COMPLETE original batch32 update against the CUDA row bank."""
    import torch
    from transformers import set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    rows=disk_rows(s['feature_cache']);visits=profile.target_visit_indices(data.manifest)
    probes=[next(i for i,(ep,anchor)in enumerate(data.rows)
                 if ep['parent']=='TASK_LEARNING_UNDER_HAND_'+layout and anchor==chosen)
            for layout in ('TRAIN01','TRAIN07','TRAIN13','TRAIN25')for chosen in(0,9)]
    predictions=generation_equivalence(model,data,rows,probes,root,guard)
    initial={n:model.get_parameter(n).detach().cpu().clone()for n in contract.trainable_names}
    results={};bank=None
    for route in ('ORDINARY_FULL_RGB','CUDA_FROZEN_BACKBONE_BANK'):
        guard();folder=root/route;folder.mkdir()
        with torch.no_grad():
            for n,value in initial.items():model.get_parameter(n).copy_(value)
        model.zero_grad(set_to_none=True)
        if route=='CUDA_FROZEN_BACKBONE_BANK':
            model.backbone.to('cpu');gc.collect();torch.cuda.empty_cache()
            bank=rt.ResidentRows(s['feature_cache'],dataset_sha256=data.manifest_sha,
                statistics_sha256=data.statistics_sha,maximum_bytes=s['GPU_row_bank_limit_bytes'],guard=guard)
        optimizer,scheduler=rt.optimizer_and_scheduler(model,contract,s['recipe'])
        set_seed(s['recipe']['seed']);audit=NativeNoiseAudit(model);torch.cuda.reset_peak_memory_stats()
        try:
            result=rt.update(model,contract,optimizer,scheduler,visits,0,bank or rows,
                full_data=data if route=='ORDINARY_FULL_RGB'else None,audit=audit,guard=guard,
                qualifying=True,trace=trace_writer(folder))
            results[route]=result;write(folder/'UPDATE_ACTUAL.private.json',result)
        finally:
            audit.close();optimizer.state.clear();del optimizer,scheduler
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    left,right=[results[k]['equivalence']for k in results]
    exact=left==right
    unchanged=frozen_fingerprint(model)==frozen
    result=dict(schema='NATIVE_ARX5_COMPLETE_BATCH32_CACHE_UPDATE_QUALIFICATION_V1',
        passes=exact and unchanged,parent_files_identity=profile.PARENT_ID,
        dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
        cache=s['feature_cache'],parameter_contract=contract.describe(),
        complete_32_microbatch_update_exact=exact,frozen_weights_unchanged=unchanged,
        predictions=predictions,actual_non_candidate_optimizer_updates=2,
        actual_training_head_calls=64,actual_full_model_training_calls=32,
        candidate_optimizer_updates=0,bank=bank.describe(),executions=results,
        CPU_training_activation_offload=False,physics_integrations=0)
    write(root/'QUALIFICATION_ACTUAL.private.json',result)
    rt.require(result['passes'],'COMPLETE_BATCH32_FULL_CACHE_UPDATE_EQUIVALENCE')
    return result


def install_source_endpoint(pin,model,contract):
    """Carry ONLY source500 whitelist values; never source Adam, RNG or stats."""
    import torch
    p=Path(pin['path']);rt.require(digest(p/'COMPLETE.json')==pin['sha256'],'SOURCE500_COMPLETE_PIN')
    cp=json.loads((p/'COMPLETE.json').read_bytes());r=cp['recipe']
    rt.require(cp['schema']==rt.CHECKPOINT_SCHEMA and cp['global_step']==500
        and cp['parent_files_identity']==profile.PARENT_ID and r==profile.recipe('B_SOURCE',r['dataset_sha256'],r['statistics_sha256'])
        and cp['parameter_contract']==contract.describe()
        and all(digest(p/n)==sha for n,sha in cp['files'].items()),'ACTUAL_OWN_B_SOURCE500_ONLY')
    values=torch.load(p/'trainable.pt',map_location='cpu',weights_only=True)
    rt.require(set(values)==set(contract.trainable_names),'SOURCE_SAME_CATEGORY10_AND_LAST2')
    with torch.no_grad():
        for n,value in values.items():
            target=model.get_parameter(n)
            rt.require(value.shape==target.shape and value.dtype==target.dtype,'SOURCE_WHITELIST_LAYOUT')
            target.copy_(value)


def training_admission(s,contract):
    """Bind new candidate updates to actual new-parent/cache/restore evidence."""
    pin=s['training_admission'];p=Path(pin['path'])
    rt.require(digest(p)==pin['sha256'],'ACTUAL_TRAINING_ADMISSION_PIN')
    a=json.loads(p.read_bytes())
    rt.require(a.get('schema')=='NATIVE_ARX5_OFFICIAL_PARENT_TRAINING_ADMISSION_V1'
        and a.get('passes') is True and a.get('parent_files_identity')==profile.PARENT_ID
        and a.get('dataset_sha256')==s['dataset_sha256']
        and a.get('statistics_sha256')==s['statistics_sha256']
        and a.get('feature_cache')==s['feature_cache']
        and a.get('parameter_contract')==contract.describe(), 'NEW_PARENT_AND_IMAGE_TRAINING_QUALIFICATION')
    proofs={}
    for kind in ('complete_batch32_cache_update','fresh_process_restore4','ordinary_RGB_reconstruction'):
        q=a[kind];file=Path(q['path']);rt.require(digest(file)==q['sha256'],'QUALIFICATION_PROOF_'+kind)
        proofs[kind]=json.loads(file.read_bytes())
    cache=proofs['complete_batch32_cache_update'];restore=proofs['fresh_process_restore4']
    ordinary=proofs['ordinary_RGB_reconstruction']
    rt.require(cache.get('schema')=='NATIVE_ARX5_COMPLETE_BATCH32_CACHE_UPDATE_QUALIFICATION_V1'
        and cache.get('passes') is True and cache.get('complete_32_microbatch_update_exact') is True
        and cache.get('actual_non_candidate_optimizer_updates')==2
        and cache.get('cache')==s['feature_cache']
        and cache.get('CPU_training_activation_offload') is False
        and restore.get('schema')=='NATIVE_ARX5_FRESH_PROCESS_4_VS_2_PLUS_2_ACTUAL_V1'
        and restore.get('passes') is True and restore.get('actual_non_candidate_optimizer_updates')==8
        and restore.get('complete_next_update_and_final_state_exact') is True
        and restore.get('feature_cache')==s['feature_cache']
        and restore.get('independent_process_model_loads')==3
        and ordinary.get('schema')=='NATIVE_ARX5_ORDINARY_RGB_ENDPOINT_RECONSTRUCTION_ACTUAL_V1'
        and ordinary.get('passes') is True and ordinary.get('global_step')==4
        and ordinary.get('normalized_and_physical_H40_bit_exact') is True
        and ordinary.get('original_graph_without_wrappers') is True
        and ordinary.get('training_feature_cache_used') is False,
        'ACTUAL_COMPLETE_BATCH32_AND_FRESH_PROCESS_RESTORE_NOT_LEGACY_FIT4')
    return a


def compare_restore_jobs(continuous,split,resumed,output):
    """Reconcile three actually finished fresh processes, without model calls."""
    from scripts.pi0292.durable import write
    roots=[Path(p)for p in(continuous,split,resumed)]
    actual=[json.loads((p/'SCIENCE_ACTUAL.private.json').read_bytes())for p in roots]
    rt.require(all(a.get('schema')=='NATIVE_ARX5_ACTUAL_CACHED_STAGE_V1'
        and a.get('qualification_only') is True and a.get('actual_model_loads')==1
        and a.get('frozen_weights_unchanged') is True for a in actual),'THREE_ACTUAL_RESTORE_QUALIFICATION_JOBS')
    c,s,r=actual
    rt.require([a['global_step']for a in actual]==[4,2,4]
        and [a['actual_optimizer_updates']for a in actual]==[4,2,2]
        and r['resume']==s['checkpoint'] and not c['resume'] and not s['resume']
        and c['recipe']==s['recipe']==r['recipe']
        and c['feature_cache']==s['feature_cache']==r['feature_cache']
        and len({(a['process_identity']['pid'],a['process_identity']['start_ticks'])for a in actual})==3,
        'ACTUAL_FRESH_PROCESS_4_VS_2_PLUS_2_NOT_SESSION_REUSE')
    traces=[[json.loads(line)['equivalence']for line in(p/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]for p in roots]
    states=[]
    for a in(c,r):
        cp=Path(a['checkpoint']['path'])
        rt.require(digest(cp/'COMPLETE.json')==a['checkpoint']['sha256'],'RESTORE_FINAL_COMPLETE_PIN')
        complete=json.loads((cp/'COMPLETE.json').read_bytes())
        rt.require(complete.get('Processor_saved') is True
            and all(digest(cp/n)==sha for n,sha in complete['files'].items()),'ALL_FRESH_FINAL_STATE_MEMBERS')
        states.append(json.loads((cp/'STATE_IDENTITY.json').read_bytes()))
    exact=traces[0]==traces[1]+traces[2] and states[0]==states[1]
    result=dict(schema='NATIVE_ARX5_FRESH_PROCESS_4_VS_2_PLUS_2_ACTUAL_V1',passes=exact,
        complete_next_update_and_final_state_exact=exact,independent_process_model_loads=3,
        actual_non_candidate_optimizer_updates=8,candidate_optimizer_updates=0,
        actual_head_training_calls=256,feature_cache=c['feature_cache'],recipe=c['recipe'],
        dataset_sha256=c['dataset_sha256'],statistics_sha256=c['statistics_sha256'],
        parent_files_identity=profile.PARENT_ID,actual_finished_job_result_sha256=[digest(p/'SCIENCE_ACTUAL.private.json')for p in roots],
        physics_integrations=0)
    write(Path(output),result)
    rt.require(exact,'FRESH_PROCESS_RESTORE_ALL_LOSSES_GRADIENTS_ADAM_RNG_AND_CURSOR')
    return result


def cached_updates(s,model,contract,data,frozen,root,guard):
    import torch
    from transformers import set_seed
    from scripts.pi0292.durable import write
    from scripts.pi0294.native_train import NativeNoiseAudit
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    qualification=s['operation']=='QUALIFY_RESTORE'
    recipe=s['recipe'];stage=recipe['stage'];stop=s['stop_step'];resume=s.get('resume')
    rt.require(qualification or s['operation']=='CANDIDATE','EXPLICIT_UPDATE_PURPOSE')
    rt.require(recipe==profile.recipe(stage,data.manifest_sha,data.statistics_sha),'EXACT_500_25_STAGE')
    rt.require(stage!='B_SOURCE','SOURCE_READER_REQUIRES_SEPARATE_FEEDBACK_CLOCK_ADMISSION')
    rt.require((qualification and stage=='A_TARGET'and stop in(2,4))or(not qualification and stop in(250,500)),
               'DECLARED_QUALIFICATION_OR_CANDIDATE_ENDPOINT')
    if not qualification:
        rt.require(data.for_optimization,'ACTUAL_NEW_TRAINING_ADMISSION')
        training_admission(s,contract)
    if not resume and stage=='B_TARGET':install_source_endpoint(s['source_endpoint'],model,contract)
    model.backbone.to('cpu');model.action_head.to('cuda');model.eval();gc.collect();torch.cuda.empty_cache()
    bank=rt.ResidentRows(s['feature_cache'],dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
                        maximum_bytes=s['GPU_row_bank_limit_bytes'],guard=guard)
    visits=profile.target_visit_indices(data.manifest);visits_sha=visit_identity(visits)
    optimizer,scheduler=rt.optimizer_and_scheduler(model,contract,recipe)
    set_seed(recipe['seed'])
    start=0
    if resume:
        cp=Path(resume['path']);rt.require(digest(cp/'COMPLETE.json')==resume['sha256'],'FULL_RESUME_PIN')
        start=rt.restore_checkpoint(cp,model,contract,optimizer,scheduler,recipe,s['feature_cache'],frozen,visits_sha)
    rt.require(0<=start<stop and (not qualification or(start,stop)in((0,2),(0,4),(2,4))), 'FINITE_NEXT_UPDATE_RANGE')
    write(root/'START_ACTUAL.private.json',dict(restored=bool(resume),start_step=start,stop_step=stop,
        visit_sha256=visits_sha,bank=bank.describe(),qualification_only=qualification,
        source_whitelist_carried=stage=='B_TARGET'and not resume,actual_optimizer_updates=0))
    contract._training_session_used=True
    audit=NativeNoiseAudit(model);checkpoint=None;torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start,stop):
            result=rt.update(model,contract,optimizer,scheduler,visits,step,bank,audit=audit,guard=guard,
                             qualifying=qualification,trace=trace_writer(root))
            with(root/'OPTIMIZER_STEPS.jsonl').open('a')as stream:stream.write(json.dumps(result)+'\n')
            write(root/'PROGRESS.private.json',result)
            if step+1 in(250,500)or step+1==stop:
                checkpoint=root/f'checkpoint-{step+1}'
                rt.save_checkpoint(checkpoint,model,contract,optimizer,scheduler,step+1,recipe,
                                   s['feature_cache'],frozen,visits_sha,processor=data.processor,
                                   statistics_file=data.root/'statistics.json')
        unchanged=frozen_fingerprint(model)==frozen
        rt.require(unchanged,'FROZEN_OFFICIAL_PARENT_UNCHANGED')
        return dict(schema='NATIVE_ARX5_ACTUAL_CACHED_STAGE_V1',qualification_only=qualification,
            stage=stage,actual_optimizer_updates=stop-start,actual_head_training_calls=32*(stop-start),
            candidate_optimizer_updates=0 if qualification else stop-start,
            global_step=stop,cursor=32*stop,checkpoint=dict(path=str(checkpoint),sha256=digest(checkpoint/'COMPLETE.json')),
            parent_files_identity=profile.PARENT_ID,dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
            frozen_weights_unchanged=unchanged,bank=bank.describe(),actual_model_loads=1,
            recipe=recipe,feature_cache=s['feature_cache'],resume=resume,
            max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
            physics_integrations=0,original_DEV_TEST_queries=0)
    finally:audit.close()


def run(s,root,guard):
    """Guard is the already registered finite owner; no allocation is minted here."""
    from scripts.pi0294.numerics import MODE,prepare_environment,configure_torch
    prepare_environment(MODE)
    import os
    rt.require(os.uname().nodename=='corelab-4080s'and os.getuid()==2009 and not os.getenv('PBS_JOBID')
        and s['science_role']==profile.SCOPE,'EXISTING_REGISTERED_CORE_OPERATION_ONLY')
    rt.require(s['operation'] in('BUILD_CACHE','QUALIFY_FULL_CACHE','QUALIFY_RESTORE','CANDIDATE','RECONSTRUCT')
        and s['recipe']==profile.recipe(s['recipe']['stage'],s['dataset_sha256'],s['statistics_sha256']),
        'EXPLICIT_NEW_OPERATION_AND_STAGE_BEFORE_MODEL_LOAD')
    guard()
    import torch
    configure_torch(torch,MODE);torch.set_num_threads(4)
    from .target_gr00t_training_v1 import gpu_identity
    gpu_identity(torch,dict(s,execution_site='CORELAB_4080S'))
    from scripts.pi0292.durable import write
    root=Path(root)
    model,contract,frozen=rt.load_official(s,root,guard)
    data=rt.build_target_data(s,for_optimization=s['operation']=='CANDIDATE')
    if s['operation']=='BUILD_CACHE':
        pin=rt.cache_backbone(model,data,root/'cache',guard)
        result=dict(schema=rt.CACHE_SCHEMA,actual_model_loads=1,actual_backbone_calls=len(data),feature_cache=pin,
            actual_optimizer_updates=0,physics_integrations=0)
    elif s['operation']=='QUALIFY_FULL_CACHE':
        result=full_cache_update_qualification(s,model,contract,data,frozen,root,guard)
    elif s['operation']=='RECONSTRUCT':
        from .native_arx5_endpoint_v1 import run as reconstruct
        result=reconstruct(s,model,contract,data,frozen,root,guard)
    else:
        result=cached_updates(s,model,contract,data,frozen,root,guard)
    result['actual_model_loads']=1
    result['process_identity']=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    guard();write(root/'SCIENCE_ACTUAL.private.json',result)
    return result
