"""Bounded native smoke/main entry; refuses to load outside a verified own PBS job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import time

from .durable import OwnerLock,read,write
from .train_adapter import NativeDataset,ProcessedCollator,official_import,sha

RELEASE='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65'
BASE_REVISION='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
COSMOS_REVISION='9ce19a195e423419c349abfc86fd07178b230561'


def verify_allocation(root,mode):
    adoption=read(root/'ADOPTION.json');authorization=read(root/'NEURAL_AUTHORIZATION.json')
    if adoption['release']!=RELEASE or time.time()>=adoption['new_work_cutoff_epoch']:raise PermissionError('AUTHORITY_EXPIRED_OR_CHANGED')
    job=os.environ.get('PBS_JOBID')
    if not job or authorization['job_id']!=job or mode not in authorization['allowed_modes']:
        raise PermissionError('EXPLICIT_OWN_PBS_ALLOCATION_REQUIRED')
    if time.time()>=authorization['allocation_end_epoch'] or authorization['prior_GPU_allocated_s']+authorization['allocation_walltime_s']>8*3600:
        raise PermissionError('GPU_TIME_BUDGET')
    remaining=authorization['predicted_remaining_peak_bytes']
    owned=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    if remaining<=0 or owned+remaining>180*1024**3 or shutil.disk_usage(root).free<remaining+20*1024**3:
        raise PermissionError('ALLOCATION_SPACE_PREFLIGHT_FAILED')
    info=json.loads(subprocess.check_output(['qstat','-f','-F','json',job],text=True,timeout=20))['Jobs'][job]
    hostname=socket.gethostname().split('.')[0]
    nodes={part.split('/')[0].split('.')[0] for part in info['exec_host'].split('+')}
    if info['job_state']!='R' or info['Job_Owner'].split('@')[0]!=os.environ['USER'] or hostname not in nodes:
        raise PermissionError('NOT_ON_RUNNING_OWN_COMPUTE_NODE')
    if int(info['Resource_List']['ngpus'])!=1:raise PermissionError('ONE_ALLOCATED_GPU_REQUIRED')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible or visible=='-1':raise PermissionError('SCHEDULER_SINGLE_GPU_VISIBILITY_REQUIRED')
    import torch
    if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):raise PermissionError('ONE_H200_REQUIRED')
    return dict(job_id=job,hostname=hostname,visible=visible,gpu=torch.cuda.get_device_name(0),
        allocation_end_epoch=authorization['allocation_end_epoch'],verified_epoch=time.time())


def recipe_for(mode,microbatch):
    if microbatch not in (1,2):raise ValueError('ONLY_PRE_MAIN_OOM_BATCH_FALLBACK_ALLOWED')
    return dict(protocol_release=RELEASE,kind='main' if mode=='main' else 'smoke',
        pretrained_repo='nvidia/GR00T-N1.7-3B',pretrained_revision=BASE_REVISION,Cosmos_revision=COSMOS_REVISION,
        max_steps=4096 if mode=='main' else 4,seed=29200,data_seed=29200,
        effective_batch=32,per_device_train_batch_size=microbatch,gradient_accumulation_steps=32//microbatch,
        world_size=1,workers=2,learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,
        max_grad_norm=1.,scheduler='cosine',warmup_ratio=.05,bf16=True,tf32=True,
        optimizer='adamw_torch_fused',optimizer_source='Pinned official training_config default AdamW implementation',
        gradient_checkpointing=False,eval_strategy='no',save_total_limit=None,skip_weight_loading=False,
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        load_bf16=False,backbone_trainable_params_fp32=True,state_dropout_prob=0.,
        action_loss='UNCHANGED_NATIVE_MASKED_FLOW_MATCHING',statistics_fit=False,
        samples='ALL_LEGAL_TRAIN_WINDOWS_CYCLIC_PCG64_29200_PLUS_CYCLE',num_inference_timesteps=4)


def verify_inputs(root,bindings,mode):
    gate=read(root/'evidence/HOPPER_DATA_QUALIFICATION.json')
    if gate['status']!='PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE':raise PermissionError('DATA_AND_ADAPTER_GATE_REQUIRED')
    dataset=Path(bindings['dataset'])
    if sha(dataset/'DATASET_MANIFEST.json')!=gate['manifest_sha256'] or sha(dataset/'statistics.json')!=gate['statistics_sha256']:
        raise ValueError('QUALIFIED_DATASET_CHANGED')
    base=Path(bindings['base'])
    identity=read(root/'evidence/PRETRAINED_FILES.private.json')
    if identity['revision']!=BASE_REVISION:raise ValueError('WRONG_BASE_REVISION')
    for row in identity['files']:
        if sha(base/row['name'])!=row['sha256']:raise ValueError('BASE_FILE_CHANGED_BEFORE_LOAD')
    cache=read(root/'evidence/COSMOS_IDENTITY.private.json')
    if cache['revision']!=COSMOS_REVISION:raise ValueError('WRONG_COSMOS_REVISION')
    for row in cache['files']:
        if sha(Path(bindings['cosmos'])/row['file'])!=row['sha256']:raise ValueError('COSMOS_FILE_CHANGED_BEFORE_LOAD')
    if mode=='main':
        if read(root/'evidence/MODEL_SMOKE.json')['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED':
            raise PermissionError('ACTUAL_MODEL_SMOKE_GATE_REQUIRED')
        if read(root/'evidence/OPEN_LOOP_PROBE_SEAL.json')['status']!='SEALED_BEFORE_MAIN':
            raise PermissionError('PRE_REGISTERED_PROBE_SEAL_REQUIRED')
        if read(root/'evidence/MAIN_CAPACITY_PREFLIGHT.json')['status']!='PASS_FULL_4096_WITHIN_REMAINING_BUDGET':
            raise PermissionError('ACTUAL_THROUGHPUT_AND_STORAGE_GATE_REQUIRED')


def run(args):
    root=Path(args.root);lock=OwnerLock(root/'NEURAL_OWNER.lock')
    mode=args.mode;allocation=verify_allocation(root,mode)
    bindings=read(args.bindings);verify_inputs(root,bindings,mode)
    official_import(bindings['source'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    import torch
    from transformers import AutoModel,TrainingArguments,TrainerCallback,set_seed
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # Registers the official class, after allocation verification.
    from .trainer_adapter import make_trainer_class,parameter_inventory,tensor_identity,rng_identity
    torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    recipe=recipe_for(mode,args.microbatch)
    ledger_path=root/'HOPPER_RESOURCE_LEDGER.json';ledger=read(ledger_path)
    if args.microbatch==1 and not read(root/'evidence/PRE_MAIN_OOM_FALLBACK.json')['authorized_same_effective_batch']:
        raise PermissionError('MICROBATCH_FALLBACK_NEEDS_ACTUAL_PRE_MAIN_OOM_EVIDENCE')
    output=root/'training'/('main' if mode=='main' else ('smoke_continuous' if mode=='smoke-continuous' else 'smoke_split'))
    session='session-resume' if mode=='smoke-resume' else 'session-start'
    if mode=='main' and args.resume:session='session-resume-'+str(read(Path(args.resume)/'trainer_state.json')['global_step'])
    evidence=output/session
    evidence.mkdir(parents=True,exist_ok=False)
    resume=None
    if mode=='smoke-resume':resume=output/'checkpoint-2'
    if mode=='main' and args.resume:resume=Path(args.resume)
    if mode=='main' and ledger['main_updates']>0 and resume is None:raise PermissionError('NO_SECOND_PRISTINE_MAIN_START')
    if mode!='main' and ledger['model_smoke_updates']+(2 if mode!='smoke-continuous' else 4)>16:
        raise PermissionError('SMOKE_UPDATE_BUDGET')
    if resume and not (resume/'CHECKPOINT_COMPLETE.json').exists():raise ValueError('FULL_CHECKPOINT_REQUIRED')
    write(evidence/'ALLOCATION.json',allocation);write(evidence/'RECIPE.json',recipe)
    set_seed(29200)
    loading=dict(local_files_only=True,trust_remote_code=True,cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION)
    ledger['model_loads']+=1
    write(ledger_path,ledger)
    model,loading_info=AutoModel.from_pretrained(bindings['base'],local_files_only=True,trust_remote_code=True,
        output_loading_info=True,transformers_loading_kwargs=loading,
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    missing=loading_info.get('missing_keys',[])
    if any('mask_token' not in k for k in missing) or loading_info.get('unexpected_keys') or loading_info.get('mismatched_keys'):
        raise ValueError('OFFICIAL_CHECKPOINT_LOADING_MISMATCH')
    if any('mask_token' in k for k in missing) and model.action_head.mask_token is not None:
        with torch.no_grad():model.action_head.mask_token.copy_(.02*torch.randn_like(model.action_head.mask_token))
    if model.action_head.num_inference_timesteps!=4:raise ValueError('NATIVE_FOUR_STEP_SAMPLER_REQUIRED')
    initial=parameter_inventory(model)
    row10={n:tensor_identity(p[10]) for n,p in model.named_parameters() if n.startswith('action_head.') and p.ndim and p.shape[0]==32}
    write(evidence/'PRETRAINED_LOAD.json',dict(status='ACTUAL_OFFICIAL_PRISTINE_LOAD',loading_info=loading_info,
        revision=BASE_REVISION,config=model.config.to_filtered_dict(),parameters=initial,embodiment_10_rows=row10,
        row_source='Official checkpoint unless explicitly listed in missing mask_token initialization',skip_weight_loading=False))
    raw_bindings=read(bindings['raw_bindings'])
    dataset=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],for_optimization=True,
        raw_bindings=raw_bindings,trace_root=evidence/'worker_trace')
    dataset.initialize()
    training_args=TrainingArguments(output_dir=str(output),max_steps=recipe['max_steps'],
        per_device_train_batch_size=args.microbatch,gradient_accumulation_steps=32//args.microbatch,
        learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,max_grad_norm=1.,
        lr_scheduler_type='cosine',warmup_ratio=.05,bf16=True,tf32=True,optim='adamw_torch_fused',
        dataloader_num_workers=2,dataloader_pin_memory=True,remove_unused_columns=False,
        seed=29200,data_seed=29200,eval_strategy='no',logging_strategy='steps',logging_steps=1,
        save_strategy='steps',save_steps=2048 if mode=='main' else 4,save_total_limit=None,
        save_only_model=False,gradient_checkpointing=False,report_to=[],disable_tqdm=True,
        ignore_data_skip=True,load_best_model_at_end=False)
    trainer=None

    class AuditCallback(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            if time.time()>=min(read(root/'ADOPTION.json')['new_work_cutoff_epoch'],allocation['allocation_end_epoch']-60):
                raise TimeoutError('NEURAL_NEW_WORK_OR_ALLOCATION_CUTOFF')
            item=read(ledger_path);key='main_updates' if mode=='main' else 'model_smoke_updates'
            if item[key]+item.get('pending_updates',0)>= (4096 if mode=='main' else 16):raise PermissionError('UPDATE_BUDGET')
            item.update(pending_updates=1,pending_kind=key,pending_global_step=state.global_step)
            write(ledger_path,item)

        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            if mode=='main':return
            groups={'projector':0.,'diffusion':0.}
            for name,p in model.named_parameters():
                if p.grad is None:continue
                if not bool(torch.isfinite(p.grad).all()):raise ValueError('NONFINITE_NATIVE_GRADIENT')
                if name.startswith('action_head.'):
                    key='diffusion' if name.startswith('action_head.model.') else 'projector'
                    groups[key]+=float(p.grad.float().square().sum())
            if not all(value>0 for value in groups.values()):raise ValueError('MISSING_PROJECTOR_OR_DIFFUSION_GRADIENT')
            with (evidence/'GRADIENTS.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=state.global_step,gradient_squared_norms=groups))+'\n')

        def on_step_end(self,arguments,state,control,**kwargs):
            item=read(ledger_path);key='main_updates' if mode=='main' else 'model_smoke_updates'
            item[key]+=1;item['pending_updates']=0;write(ledger_path,item)
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=state.global_step,cursor=trainer.consumed,
                    scheduler=trainer.lr_scheduler.state_dict(),rng=rng_identity(),epoch=time.time()),sort_keys=True)+'\n')
            if mode=='smoke-first' and state.global_step==2:
                control.should_save=True;control.should_training_stop=True

    trainer=make_trainer_class()(model=model,args=training_args,train_dataset=dataset,
        data_collator=ProcessedCollator(dataset.processor),callbacks=[AuditCallback()],
        evidence_dir=evidence,recipe=recipe,smoke=mode!='main')
    started=time.time()
    result=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    final=parameter_inventory(model)
    frozen=[name for name in initial if name.startswith('backbone.')]
    unchanged=all(initial[name]['sha256']==final[name]['sha256'] for name in frozen)
    changed=[name for name in initial if initial[name]['requires_grad'] and initial[name]['sha256']!=final[name]['sha256']]
    if not unchanged or not changed:raise ValueError('FROZEN_VLM_CHANGED_OR_NO_ACTION_WEIGHT_UPDATED')
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(evidence/'TRAINING_RUN.json',dict(status='RUN_COMPLETE_COMPARISON_PENDING' if mode!='main' else 'MAIN_COMPLETED_FIXED_S4096',
        mode=mode,global_step=trainer.state.global_step,consumed_samples=trainer.consumed,
        frozen_backbone_unchanged=unchanged,changed_trainable_parameters=changed,final_parameters=final,
        native_metrics=result.metrics,elapsed_training_s=time.time()-started,
        CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime,recipe=recipe,
        selected='NONE',deployment='NONE'))
    if trainer.noise_audit:trainer.noise_audit.close()
    lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--bindings',required=True)
    parser.add_argument('--mode',choices=['smoke-continuous','smoke-first','smoke-resume','main'],required=True)
    parser.add_argument('--microbatch',type=int,default=2);parser.add_argument('--resume')
    run(parser.parse_args())
