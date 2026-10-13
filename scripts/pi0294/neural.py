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

from scripts.pi0292.durable import OwnerLock,read,write
from .native_dataset import NativeDataset,ProcessedCollator,official_import,sha

T0=1791901497
CUTOFF=1791939600
DEADLINE=1791943200
BASE_REVISION='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
COSMOS_REVISION='9ce19a195e423419c349abfc86fd07178b230561'


def verify_allocation(root,mode,request_path):
    from .allocation_admission import admit
    allocation=admit(root,request_path)
    authorization=read(root/'NEURAL_AUTHORIZATION.json')
    if authorization['job_id']!=os.environ.get('PBS_JOBID') or mode not in authorization['allowed_modes'] or authorization['request_sha256']!=sha(request_path):
        raise PermissionError('EXACT_REQUEST_MODE_ALLOCATION_REQUIRED')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible or visible=='-1':raise PermissionError('SCHEDULER_SINGLE_GPU_VISIBILITY_REQUIRED')
    import torch
    if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):raise PermissionError('ONE_H200_REQUIRED')
    allocation.update(allocation_end_epoch=authorization['allocation_end_epoch'],gpu=torch.cuda.get_device_name(0))
    return allocation


def recipe_for(mode,microbatch,workers=2,max_steps=4096):
    if microbatch not in (2,4,8) or workers not in (0,2,4):raise ValueError('PRESEALED_BENCHMARK_OPTIONS_ONLY')
    if max_steps not in (4096,6144):raise ValueError('FIXED_MAIN_HORIZONS_ONLY')
    return dict(kind='main' if mode=='main' else 'smoke',pretrained_repo='nvidia/GR00T-N1.7-3B',
        pretrained_revision=BASE_REVISION,Cosmos_revision=COSMOS_REVISION,
        max_steps=6144 if mode=='main' else (2 if mode=='benchmark' else 4),
        scheduler_horizon_steps=6144 if mode=='main' else (2 if mode=='benchmark' else 4),
        warmup_steps=308 if mode=='main' else 0,extension_from_step=None,
        seed=29400,data_seed=29400,effective_batch=32,per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=32//microbatch,world_size=1,workers=workers,
        learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,
        max_grad_norm=1.,scheduler='cosine',warmup_ratio=.05,bf16=True,tf32=True,
        optimizer='adamw_torch_fused',gradient_checkpointing=False,save_total_limit=None,
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        load_bf16=False,backbone_trainable_params_fp32=True,state_dropout_prob=0.,
        action_loss='UNCHANGED_NATIVE_MASKED_FLOW_MATCHING',statistics_fit=False,
        samples='PI0294_TASK50_FAMILY_EPISODE_STAGE_20_25_25_20_10',num_inference_timesteps=4)


def dev_panel(records):
    """All eight fixed DEV attempts remain in denominator; packets are not evidence."""
    if len(records)!=8 or len({r['layout_id'] for r in records})!=8:
        raise ValueError('EXACT_FIXED_EIGHT_DEV_ATTEMPTS_REQUIRED')
    for row in records:
        if (row['task'] not in ('T1','T2') or row['attempted'] is not True
            or row['proof_kind']!='REAL_CORELAB_MODEL_CLOSED_LOOP'
            or any(len(row[k])!=64 for k in ('result_sha256','physics_sha256','checkpoint_identity_sha256'))
            or type(row['persistent_success']) is not bool or type(row['safety_valid']) is not bool
            or not 0<=row['rejected_prefixes']<=row['total_prefixes']):
            raise ValueError('ACTUAL_HASH_BOUND_MODEL_DEV_RESULTS_REQUIRED')
    counts={task:sum(r['persistent_success'] for r in records if r['task']==task) for task in ('T1','T2')}
    if any(sum(r['task']==task for r in records)!=4 for task in ('T1','T2')):
        raise ValueError('FIXED_FOUR_DEV_PER_TASK_REQUIRED')
    if len({r['checkpoint_identity_sha256'] for r in records})!=1:
        raise ValueError('ONE_SHARED_CHECKPOINT_PER_DEV_PANEL')
    total=sum(r['total_prefixes'] for r in records)
    return dict(successes=counts,total_successes=sum(counts.values()),
        safety_violations=sum(not r['safety_valid'] for r in records),
        rejection_rate=sum(r['rejected_prefixes'] for r in records)/total if total else 1.,
        minimum_task_success=min(counts.values())/4,macro_success=sum(counts.values())/8)


def extension_gate(panels):
    earlier,later=dev_panel(panels['M2048']),dev_panel(panels['M4096'])
    if {(r['task'],r['layout_id']) for r in panels['M2048']}!={(r['task'],r['layout_id']) for r in panels['M4096']}:
        raise ValueError('SAME_FIXED_DEV_SUBSET_REQUIRED')
    return (later['total_successes']>=earlier['total_successes']+2
        and all(n>=1 for n in later['successes'].values())
        and earlier['safety_violations']==later['safety_violations']==0)


def choose_candidate(panels):
    ranked=[]
    for step,records in panels.items():
        if int(step) not in (2048,4096,6144):raise ValueError('ONLY_PREREGISTERED_DEV_CANDIDATES')
        value=dev_panel(records)
        if value['safety_violations']:continue
        ranked.append((value['minimum_task_success'],value['macro_success'],-value['rejection_rate'],-int(step),int(step)))
    if not ranked:raise PermissionError('NO_SAFE_ACTUAL_DEV_CANDIDATE')
    return max(ranked)[-1]


def verify_extension(root,resume,allocation,ledger):
    gate=read(root/'evidence/EXTENSION_ADMISSION.json')
    if gate['status']!='AUTHORIZED_PREREGISTERED_M6144_EXTENSION' or not extension_gate(gate['actual_DEV_panels']):
        raise PermissionError('ACTUAL_FIXED_DEV_IMPROVEMENT_GATE_REQUIRED')
    if (not resume or read(Path(resume)/'trainer_state.json')['global_step']<4096
        or ledger['main_updates']>6144 or time.time()+gate['predicted_remaining_training_s']>=min(CUTOFF,allocation['allocation_end_epoch']-180)):
        raise PermissionError('FULL_RESUME_AND_MEASURED_EXTENSION_TIME_REQUIRED')
    if gate['predicted_remaining_training_s']<=0 or gate['remaining_gpu_allocation_s']<gate['predicted_remaining_training_s']:
        raise PermissionError('ACTUAL_REMAINING_EXTENSION_GPU_BUDGET_REQUIRED')


def verify_inputs(root,bindings,mode):
    gate=read(root/'evidence/HOPPER_INPUT_QUALIFICATION.json')
    if gate['status']!='PASS_FIXED_INPUTS_CPU_INTERFACE' or gate['bindings_sha256']!=sha(root/'NEURAL_BINDINGS.private.json'):
        raise PermissionError('ACTUAL_FIXED_INPUT_QUALIFICATION_REQUIRED')
    dataset=Path(bindings['dataset'])
    if sha(dataset/'DATASET_MANIFEST.json')!=gate['manifest_sha256'] or sha(dataset/'statistics.json')!=gate['statistics_sha256']:
        raise ValueError('QUALIFIED_DATASET_CHANGED')
    identity=read(root/'evidence/PRETRAINED_FILES.private.json')
    if identity['revision']!=BASE_REVISION:raise ValueError('WRONG_BASE_REVISION')
    for row in identity['files']:
        if sha(Path(bindings['base'])/row['name'])!=row['sha256']:raise ValueError('BASE_FILE_CHANGED')
    cache=read(root/'evidence/COSMOS_IDENTITY.private.json')
    if cache['revision']!=COSMOS_REVISION:raise ValueError('WRONG_COSMOS_REVISION')
    for row in cache['files']:
        if sha(Path(bindings['cosmos'])/row['file'])!=row['sha256']:raise ValueError('COSMOS_FILE_CHANGED')
    if mode=='main':
        if gate['mode']!='multitask_v3' or read(root/'evidence/MODEL_SMOKE.json')['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED':
            raise PermissionError('ACTUAL_SMOKE_AND_MULTITASK_DATA_REQUIRED')
        if read(root/'evidence/TRAINING_PROFILE_SEAL.json')['status']!='SEALED_ACTUAL_MICROBATCH_AND_WORKER_BENCHMARK':
            raise PermissionError('MEASURED_TRAINING_PROFILE_REQUIRED')


def benchmark_plan(qualification):
    if qualification['status']!='PASS_PI0294_ACTUAL_NATIVE_CPU_LOADER_0_2_4':
        raise PermissionError('ACTUAL_NEW_ADAPTER_LOADER_QUALIFICATION_REQUIRED')
    rates=qualification['actual_loader_throughput']
    for workers in (0,2,4):
        if any(not rates[f'{workers}/{step}']['natural_iterator_exhaustion']
            or not 0<rates[f'{workers}/{step}']['consumed_samples_per_s']<float('inf') for step in (0,1)):
            raise PermissionError('ACTUAL_FINITE_CPU_LOADER_RATES_REQUIRED')
    # Only steady initial consumption is compared; resume is an identity gate.
    workers=max((0,2,4),key=lambda w:(rates[f'{w}/0']['consumed_samples_per_s'],-w))
    return [dict(microbatch=m,gradient_accumulation_steps=32//m,workers=workers,updates=2) for m in (2,4,8)]


def run_benchmarks(args):
    """Three sealed candidates in one own allocation; never wait for new data."""
    import sys
    root=Path(args.root);allocation=verify_allocation(root,'benchmark-all',args.request)
    bindings=read(args.bindings);verify_inputs(root,bindings,'benchmark')
    request=read(args.request);qualification_path=Path(request['cpu_qualification'])
    if sha(qualification_path)!=request['cpu_qualification_sha256']:
        raise PermissionError('SEALED_ACTUAL_CPU_QUALIFICATION_REQUIRED')
    qualification=read(qualification_path)
    gate=read(root/'evidence/HOPPER_INPUT_QUALIFICATION.json')
    if any(qualification[k]!=gate[k] for k in ('manifest_sha256','statistics_sha256')):
        raise PermissionError('BENCHMARK_QUALIFIED_DATA_BINDING_MISMATCH')
    if read(root/'evidence/MODEL_SMOKE.json')['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED':
        raise PermissionError('ACTUAL_EIGHT_UPDATE_SMOKE_REQUIRED')
    initial=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if initial['model_smoke_updates']!=8 or initial.get('benchmark_updates',0)!=0 or initial.get('pending_updates',0):
        raise PermissionError('EXACT_FRESH_SIX_UPDATE_BENCHMARK_RESERVATION')
    profile_path=root/'evidence/TRAINING_PROFILE_SEAL.json'
    if profile_path.exists():raise FileExistsError('Preserve prior measured profile')
    plan=benchmark_plan(qualification);receipts=[]
    write(root/'evidence/BENCHMARK_PLAN.json',dict(plan=plan,
        CPU_qualification_sha256=sha(qualification_path),total_optimizer_updates=6,
        request_sha256=sha(args.request),job_id=allocation['job_id']))
    for item in plan:
        if time.time()>=min(CUTOFF,allocation['allocation_end_epoch']-180):
            raise TimeoutError('NO_TIME_FOR_NEXT_REAL_BENCHMARK')
        command=[sys.executable,'-m','scripts.pi0294.neural','--root',str(root),
            '--bindings',args.bindings,'--request',args.request,'--mode','benchmark',
            '--microbatch',str(item['microbatch']),'--workers',str(item['workers'])]
        result=subprocess.run(command,check=False,timeout=max(1,min(CUTOFF,allocation['allocation_end_epoch']-120)-time.time()))
        if result.returncode:raise RuntimeError('BENCHMARK_CANDIDATE_FAILED_NO_AUTOMATIC_RETRY')
        receipt=root/'training'/f"benchmark-{item['microbatch']}-{item['workers']}"/'session-start/TRAINING_RUN.json'
        actual=read(receipt)
        if (actual['mode']!='benchmark' or actual['global_step']!=2
            or len(actual['actual_synchronized_optimizer_step_s'])!=2
            or not actual['frozen_backbone_unchanged'] or not actual['changed_trainable_parameters']
            or any(actual[k]!=gate[k] for k in ('manifest_sha256','statistics_sha256'))):
            raise ValueError('REAL_TWO_UPDATE_BENCHMARK_RECEIPT_REQUIRED')
        receipts.append(dict(**item,receipt=str(receipt.relative_to(root)),receipt_sha256=sha(receipt),
            steps_per_s=actual['actual_optimizer_steps_per_s'],
            peak_gpu_allocated_bytes=actual['peak_gpu_allocated_bytes'],peak_gpu_reserved_bytes=actual['peak_gpu_reserved_bytes']))
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if ledger['model_smoke_updates']!=14 or ledger.get('benchmark_updates')!=6 or ledger.get('pending_updates',0):
        raise ValueError('ACTUAL_BENCHMARK_UPDATE_ACCOUNTING_MISMATCH')
    if any(not 0<r['steps_per_s']<float('inf') for r in receipts):raise ValueError('FINITE_ACTUAL_GPU_RATES_REQUIRED')
    selected=max(receipts,key=lambda r:(r['steps_per_s'],-r['microbatch']))
    write(profile_path,dict(status='SEALED_ACTUAL_MICROBATCH_AND_WORKER_BENCHMARK',
        microbatch=selected['microbatch'],workers=selected['workers'],actual_candidates=receipts,
        CPU_qualification_sha256=sha(qualification_path),manifest_sha256=gate['manifest_sha256'],
        statistics_sha256=gate['statistics_sha256'],selection='HIGHEST_REAL_TWO_UPDATE_RATE_THEN_SMALLER_MICROBATCH',
        model_smoke_updates=14,benchmark_updates=6,model_candidate=False,
        scheduler_horizon_steps=6144,warmup_steps=308,normal_stop=4096,
        extension_requires_fixed_actual_DEV_gate=True,selected='NONE',deployment='NONE'))


def smoke_plan():
    return [('smoke-continuous',4),('smoke-first',2),('smoke-resume',2)]


def run_smoke(args):
    """One PBS allocation, three fresh model processes, one actual comparison."""
    import sys
    root=Path(args.root);allocation=verify_allocation(root,'smoke-all',args.request)
    bindings=read(args.bindings);verify_inputs(root,bindings,'smoke-continuous')
    request=read(args.request)
    if not {'smoke-all',*(mode for mode,_ in smoke_plan())}.issubset(set(request['allowed_modes'])):
        raise PermissionError('EXACT_COMPARISON_SUBMODES_MUST_BE_AUTHORIZED')
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if ledger['model_smoke_updates']!=0 or ledger.get('pending_updates',0) or ledger.get('benchmark_updates',0):
        raise PermissionError('SMOKE_FIRST_ATTEMPT_ONLY_NO_AUTOMATIC_RETRY')
    if any((root/name).exists() for name in ('training/smoke_continuous','training/smoke_split',
        'evidence/MODEL_SMOKE.json','evidence/RESUME_EQUIVALENCE.json','evidence/SMOKE_DRIVER.json')):
        raise FileExistsError('Preserve any earlier actual smoke evidence')
    started=time.monotonic();status='FAILED_PRESERVED';completed=[]
    before=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    try:
        for mode,updates in smoke_plan():
            remaining=min(CUTOFF,allocation['allocation_end_epoch']-120)-time.time()
            if remaining<=0:raise TimeoutError('NO_TIME_FOR_NEXT_SMOKE_SUBPROCESS')
            command=[sys.executable,'-m','scripts.pi0294.neural','--root',str(root),
                '--request',args.request,'--bindings',args.bindings,'--mode',mode,
                '--microbatch',str(args.microbatch),'--workers',str(args.workers)]
            result=subprocess.run(command,check=False,timeout=remaining)
            if result.returncode:raise RuntimeError('ACTUAL_SMOKE_SUBPROCESS_FAILED_NO_RETRY:'+mode)
            completed.append(dict(mode=mode,updates=updates))
        actual=read(root/'HOPPER_RESOURCE_LEDGER.json')
        if actual['model_smoke_updates']!=8 or actual.get('pending_updates',0):
            raise ValueError('ACTUAL_EIGHT_UPDATE_LEDGER_REQUIRED')
        remaining=min(CUTOFF,allocation['allocation_end_epoch']-60)-time.time()
        if remaining<=0:raise TimeoutError('NO_TIME_FOR_FULL_NATIVE_CHECKPOINT_COMPARISON')
        # The sealed comparator reads this adapter's actual saved sampler state;
        # it does not invoke the PI0292 sampler or inherit any PI0292 PASS.
        result=subprocess.run([sys.executable,'-m','scripts.pi0292.compare_smoke','--root',str(root)],
            check=False,timeout=remaining)
        comparison=root/'evidence/RESUME_EQUIVALENCE.json'
        if (result.returncode or not comparison.exists()
            or read(comparison)['status']!='PASS_FIXED_RESUME_EQUIVALENCE'
            or read(root/'evidence/MODEL_SMOKE.json')['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED'):
            raise ValueError('ACTUAL_FULL_RESUME_COMPARISON_DID_NOT_PASS')
        status='PASS_ACTUAL_EIGHT_UPDATE_NATIVE_SMOKE'
    finally:
        after=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
        cpu=[a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime for a,b in zip(after,before)]
        write(root/'evidence/SMOKE_DRIVER.json',dict(status=status,completed_subprocesses=completed,
            job_id=allocation['job_id'],request_sha256=sha(args.request),wall_s=time.monotonic()-started,
            self_CPU_s=cpu[0],reaped_children_CPU_s=cpu[1],CPU_s=sum(cpu),
            accounting_scope='WHOLE_DRIVER_INCLUDES_SUBPROCESS_CPU_NOT_ADDITIVE_TO_CHILD_RECEIPTS',
            selected='NONE',deployment='NONE'))


def run(args):
    if args.mode=='smoke-all':return run_smoke(args)
    if args.mode=='benchmark-all':return run_benchmarks(args)
    root=Path(args.root);lock=OwnerLock(root/'NEURAL_OWNER.lock')
    mode=args.mode;allocation=verify_allocation(root,mode,args.request)
    bindings=read(args.bindings);verify_inputs(root,bindings,mode)
    official_import(bindings['source'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    import torch
    from transformers import AutoModel,TrainingArguments,TrainerCallback,set_seed
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # Registers the official class, after allocation verification.
    from .native_train import make_trainer_class,parameter_inventory,tensor_identity,rng_identity
    torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    recipe=recipe_for(mode,args.microbatch,args.workers,getattr(args,'max_steps',4096))
    if mode=='main':
        profile=read(root/'evidence/TRAINING_PROFILE_SEAL.json')
        if profile['microbatch']!=args.microbatch or profile['workers']!=args.workers:
            raise PermissionError('FIXED_MEASURED_TRAINING_PROFILE_REQUIRED')
        if args.stop_at>args.max_steps:raise PermissionError('STOP_EXCEEDS_AUTHORIZED_UPDATE_LIMIT')
    ledger_path=root/'HOPPER_RESOURCE_LEDGER.json';ledger=read(ledger_path)
    if mode=='main' and args.max_steps==6144:
        verify_extension(root,args.resume,allocation,ledger)
    if mode=='benchmark' and ledger.get('benchmark_updates',0)+2>6:
        raise PermissionError('EXACT_THREE_TWO_UPDATE_BENCHMARKS_ONLY')
    output=root/'training'/('main' if mode=='main' else ('benchmark-'+str(args.microbatch)+'-'+str(args.workers) if mode=='benchmark' else ('smoke_continuous' if mode=='smoke-continuous' else 'smoke_split')))
    session='session-resume' if mode=='smoke-resume' else 'session-start'
    if mode=='main' and args.resume:session='session-resume-'+str(read(Path(args.resume)/'trainer_state.json')['global_step'])
    evidence=output/session
    evidence.mkdir(parents=True,exist_ok=False)
    resume=None
    if mode=='smoke-resume':resume=output/'checkpoint-2'
    if mode=='main' and args.resume:resume=Path(args.resume)
    if mode=='main' and ledger['main_updates']>0 and resume is None:raise PermissionError('NO_SECOND_PRISTINE_MAIN_START')
    if mode!='main' and ledger['model_smoke_updates']+(4 if mode=='smoke-continuous' else 2)>16:
        raise PermissionError('SMOKE_UPDATE_BUDGET')
    if resume and not (resume/'CHECKPOINT_COMPLETE.json').exists():raise ValueError('FULL_CHECKPOINT_REQUIRED')
    write(evidence/'ALLOCATION.json',allocation);write(evidence/'RECIPE.json',recipe)
    set_seed(29400)
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
    dataset.sampling_smoke=mode!='main'
    dataset.initialize()
    training_args=TrainingArguments(output_dir=str(output),max_steps=recipe['max_steps'],
        per_device_train_batch_size=args.microbatch,gradient_accumulation_steps=32//args.microbatch,
        learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,max_grad_norm=1.,
        lr_scheduler_type='cosine',warmup_ratio=.05 if mode!='main' else 0.,
        warmup_steps=308 if mode=='main' else 0,bf16=True,tf32=True,optim='adamw_torch_fused',
        dataloader_num_workers=args.workers,dataloader_pin_memory=True,remove_unused_columns=False,
        seed=29400,data_seed=29400,eval_strategy='no',logging_strategy='steps',logging_steps=1,
        save_strategy='no' if mode in ('main','benchmark') else 'steps',save_steps=2,save_total_limit=None,
        save_only_model=False,gradient_checkpointing=False,report_to=[],disable_tqdm=True,
        ignore_data_skip=True,load_best_model_at_end=False)
    trainer=None
    step_times=[];step_started=None
    torch.cuda.reset_peak_memory_stats()

    class AuditCallback(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            nonlocal step_started
            torch.cuda.synchronize();step_started=time.monotonic()
            if time.time()>=min(read(root/'ADOPTION.json')['new_work_cutoff_epoch'],allocation['allocation_end_epoch']-60):
                raise TimeoutError('NEURAL_NEW_WORK_OR_ALLOCATION_CUTOFF')
            item=read(ledger_path);key='main_updates' if mode=='main' else 'model_smoke_updates'
            if item[key]+item.get('pending_updates',0)>= (args.max_steps if mode=='main' else 16):raise PermissionError('UPDATE_BUDGET')
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
            if mode=='benchmark':
                item['benchmark_updates']=item.get('benchmark_updates',0)+1;write(ledger_path,item)
            torch.cuda.synchronize();step_times.append(time.monotonic()-step_started)
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=state.global_step,cursor=trainer.consumed,
                    scheduler=trainer.lr_scheduler.state_dict(),rng=rng_identity(),epoch=time.time()),sort_keys=True)+'\n')
            if mode=='main' and state.global_step==args.stop_at:
                control.should_save=True;control.should_training_stop=True
            if mode=='main' and state.global_step in (512,2048,4096,6144):control.should_save=True
            if mode=='smoke-first' and state.global_step==2:
                control.should_save=mode!='benchmark';control.should_training_stop=True
            if time.time()>=min(read(root/'ADOPTION.json')['new_work_cutoff_epoch'],allocation['allocation_end_epoch'])-180:
                control.should_save=True;control.should_training_stop=True

    trainer=make_trainer_class()(model=model,args=training_args,train_dataset=dataset,
        data_collator=ProcessedCollator(dataset.processor),callbacks=[AuditCallback()],
        evidence_dir=evidence,recipe=recipe,smoke=mode not in ('main','benchmark'))
    started=time.time()
    result=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    final=parameter_inventory(model)
    frozen=[name for name in initial if name.startswith('backbone.')]
    unchanged=all(initial[name]['sha256']==final[name]['sha256'] for name in frozen)
    changed=[name for name in initial if initial[name]['requires_grad'] and initial[name]['sha256']!=final[name]['sha256']]
    if not unchanged or not changed:raise ValueError('FROZEN_VLM_CHANGED_OR_NO_ACTION_WEIGHT_UPDATED')
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    final_status='RUN_COMPLETE_COMPARISON_PENDING'
    if mode=='main':final_status='FORMAL_NATIVE_CHECKPOINT_STAGE_COMPLETE' if trainer.state.global_step==args.stop_at else 'FORMAL_PARTIAL_FULL_STATE_PRESERVED'
    write(evidence/'TRAINING_RUN.json',dict(status=final_status,
        mode=mode,global_step=trainer.state.global_step,consumed_samples=trainer.consumed,
        frozen_backbone_unchanged=unchanged,changed_trainable_parameters=changed,final_parameters=final,
        native_metrics=result.metrics,elapsed_training_s=time.time()-started,
        CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime,recipe=recipe,
        actual_synchronized_optimizer_step_s=step_times,
        actual_parent_dataloader_wait_s=trainer.measured_loader.wait_s,
        actual_optimizer_steps_per_s=len(step_times)/(sum(step_times) or float('inf')),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        benchmark_weights_are_candidate=False,manifest_sha256=dataset.manifest_sha,statistics_sha256=dataset.statistics_sha,
        selected='NONE',deployment='NONE'))
    if trainer.noise_audit:trainer.noise_audit.close()
    lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--bindings',required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--mode',choices=['smoke-all','smoke-continuous','smoke-first','smoke-resume','benchmark','benchmark-all','main'],required=True)
    parser.add_argument('--microbatch',type=int,default=2);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--stop-at',type=int,choices=[512,2048,4096,6144],default=4096);parser.add_argument('--resume')
    parser.add_argument('--max-steps',type=int,choices=[4096,6144],default=4096)
    run(parser.parse_args())

