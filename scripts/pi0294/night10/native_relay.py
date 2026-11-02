"""Bind existing finite native workers to Night10; no model or physics implementation.

Private configuration fixes producer/data/device paths. The existing Night10 DAG
provides scheduling; original registry/onceIO, user units and GPU locks retain
custody. This adapter never modifies M13, its checkout, or a historical actor.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def raw(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as f:f.write(raw(value));f.flush();os.fsync(f.fileno())


def update(path,value):
    from .supervisor import atomic
    atomic(Path(path),value)


def once(path,value,key):
    from scripts.pi0294.dec1.source_cpu_once_io import commit
    path=Path(path)
    return commit(path,expected_before_sha256=sha(path),proposed_bytes=raw(value),request_id=key,
        intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),
        lock_path=path.with_suffix('.owner.lock'))


def observe(root):
    text=subprocess.check_output(['systemctl','--user','show',Path(root).name+'.service',
        '-p','MainPID','-p','SubState','-p','InvocationID','-p','ControlGroup',
        '-p','CPUUsageNSec','-p','ExecMainStatus','-p','ExecMainStartTimestampMonotonic',
        '-p','ExecMainExitTimestampMonotonic'],text=True,timeout=8)
    return dict(line.split('=',1)for line in text.splitlines()if'='in line)


def owner(c):
    f=Path(c['existing_neural_owner_lock']).open('a+b')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:f.close();raise
    return f


def admission(c):
    if (os.uname().nodename!='corelab-4080s' or os.getuid()!=2009
            or c.get('schema')!='CORE_NATIVE_RELAY_BINDING_V1'
            or c.get('candidate_order')!=['Phase4096','D0','D1','D2','PB_CONTROL','PB_BOUNDARY']
            or c.get('GPU_index')!=0 or c.get('GPU_uuid')!=c.get('GPU0_uuid')
            or not Path(c['root']).resolve().is_relative_to(Path(c['source']).resolve().parent.parent)):
        raise PermissionError('EXACT_EXISTING_CORE_OWNER_AND_FROZEN_RELAY')
    for name,pin in c['input_pins'].items():
        if sha(name)!=pin:raise PermissionError('SEALED_RELAY_INPUT_CHANGED:'+name)
    if sha(Path(c['source'])/'scripts/pi0294/night10/native_relay.py')!=c['adapter_sha256']:
        raise PermissionError('SEALED_NATIVE_ADAPTER_CHANGED')


def checkpoint(producer):
    """Unavailable is not a fabricated zero or a substitute qualification point."""
    root=Path(producer)
    if not(root/'control/LEASE.json').is_file():return None
    state=observe(root)
    if state['MainPID']!='0' or read(root/'control/LEASE.json').get('status')!='CLOSED':
        return None
    if not(root/'RESULT.json').exists():return None
    result=read(root/'RESULT.json')
    if (result.get('optimizer_updates')!=4096 or result.get('error')is not None
            or result.get('status')!='COMPLETE_BOOK5_FINITE_TARGET_LEARNING'):
        return None
    cp=root/'training/checkpoint-4096';proof=root/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json'
    if not(cp/'CHECKPOINT_COMPLETE.json').is_file()or not proof.is_file():return None
    value=read(proof)
    if (value.get('checkpoint_manifest_sha256')!=sha(cp/'CHECKPOINT_COMPLETE.json')
            or any(value.get(k)!=v for k,v in dict(optimizer_updates=4096,sampler_cursor=131072,
                optimizer_state_entries=42,model_reconstruction=True,scheduler_restored=True,
                RNG_restored=True,frozen_parent_unchanged=True).items())):
        raise PermissionError('ACTUAL_FULL_CLOSED_MODEL_RESTORE_REQUIRED')
    return cp


def stage(c,name,module,manifest,files):
    root=Path(c['run_parent'])/name
    if root.exists()or root.is_symlink():raise PermissionError('NATIVE_IDENTITY_EXISTS_NO_REPLAY')
    c.setdefault('_actors',[]).append(str(root))
    root.mkdir(mode=0o700)
    for n in('control','tmp','home','cache','model_inputs/nvidia','entry/infrastructure','entry/code/vla'):
        (root/n).mkdir(parents=True,exist_ok=True)
    bindings=files.pop('NEURAL_BINDINGS.private.json')
    (root/'model_inputs/nvidia/Cosmos-Reason2-2B').symlink_to(bindings['cosmos'],target_is_directory=True)
    files['NEURAL_BINDINGS.private.json']=bindings
    source=Path(c['source']);worker=(source/module).read_bytes()
    for n,value in files.items():exclusive(root/n,value)
    for n in('enter.sh','doctor.py','ENVIRONMENTS.json'):
        (root/'entry/infrastructure'/n).write_bytes((Path(c['entry'])/n).read_bytes())
    (root/'worker.py').write_bytes(worker)
    manifest.update(run_uuid=root.name,owner_uid=2009,role='V7_NATIVE_RELAY__'+root.name,
        code_source_sha=c['source_commit'],dependency_code=c['source'],GPU_uuid=c['GPU_uuid'],
        protected_remaining_bytes=c['protected_remaining_bytes'],free_margin_bytes=30*1024**3,
        deadline_epoch=min(c['dispatch_deadline_epoch'],time.time()+c['native_seconds'])
            if manifest.get('recipe',{}).get('max_steps')!=8192 else time.time()+c['formal_seconds'])
    manifest['sealed_inputs']={n:sha(root/n)for n in files}|{'worker.py':sha(root/'worker.py')}
    exclusive(root/'RUN_MANIFEST.json',manifest)
    registry=read(c['registry']);account=read(c['LOCAL'])
    if any(account.get(n)!=[]for n in('unknown_effects','pending_updates','pending_queries')):
        raise PermissionError('CANONICAL_UNKNOWN_EFFECTS_BLOCK_ADMISSION')
    if any(e.get('path')==str(root)or e.get('role')==manifest['role']for e in registry['entries']):
        raise PermissionError('DUPLICATE_CANONICAL_RUN_IDENTITY')
    registry['entries'].append(dict(role=manifest['role'],path=str(root),run_uuid=root.name,
        executor_node='corelab-4080s',owner='EXISTING_LOCAL_PC',scope=manifest['kind'],
        source_sha=c['source_commit'],local_lease=str(root/'control/LEASE.json')))
    account.setdefault('V7_book_learning_runs',{})[root.name]=dict(status='REGISTERED_NOT_STARTED',
        root=str(root),role=manifest['role'],manifest_sha256=sha(root/'RUN_MANIFEST.json'),
        finite_limits=manifest['limits'],original_campaign_costs_and_protected_forecast_unchanged=True)
    once(c['registry'],registry,root.name+'-REGISTER');once(c['LOCAL'],account,root.name+'-ADMIT')
    exclusive(root/'control/REGISTERED_PATHS.json',read(c['registry']))
    exclusive(root/'control/LEASE.json',dict(status='ACTIVE',run_uuid=root.name,
        manifest_sha256=sha(root/'RUN_MANIFEST.json'),deadline_epoch=manifest['deadline_epoch']))
    exclusive(root/'RELAY_CUSTODY.json',dict(config_sha256=c['_config_sha256'],module=module,
        source_commit=c['source_commit'],source_seal_sha256=c['source_seal_sha256']))
    return root


def audit_terminal(root,manifest,result):
    limits=manifest['limits']
    counts=dict(model_loads=result.get('model_load_attempts'),optimizer_updates=result.get('optimizer_updates',0),
        generation_queries=result.get('generation_attempts',result.get('model_queries',0)),
        physics_integrations=result.get('physics_integrations',result.get('integrations',0)),
        RGB_pairs=result.get('actual_RGB_pairs',result.get('RGB_pairs',0)))
    if any(type(n)is not int or n<0 or n>limits[k]for k,n in counts.items()):
        raise PermissionError('UNKNOWN_OR_EXCESS_NATIVE_COUNTS')
    for row in result.get('conditions',[]):
        folder=root/'conditions'/row['id']
        if (sha(folder/'ACTUAL_RECORDS.jsonl')!=row['raw_sha256']
                or sha(folder/'INDEPENDENT_RAW_AUDIT.json')!=row['audit_sha256']
                or sha(folder/'TERMINAL.json')!=row['terminal_sha256']):
            raise PermissionError('ACTUAL_NATIVE_RECORDS_OR_SCORE_CHANGED')
        from scripts.pi0294.multitask.night_book_audit import audit
        independent=audit(folder,execution_prefix=8)
        if independent!=read(folder/'INDEPENDENT_RAW_AUDIT.json'):
            raise PermissionError('INDEPENDENT_ORIGINAL_BOOK_SCORE_MISMATCH')
        for name,pin in row.get('reset_witness_files',{}).items():
            if sha(folder/name)!=pin:raise PermissionError('RESET_WITNESS_CHANGED')
    return counts


def execute(c,root):
    """One finite native unit, independently observed and onceIO closed."""
    from scripts.pi0294.multitask.book_reset import registered_writer_scope,live_writers
    root=Path(root);m=read(root/'RUN_MANIFEST.json')
    if time.time()>=m['deadline_epoch']or observe(root)['MainPID']!='0':
        raise PermissionError('COLD_FINITE_NATIVE_ENTRY_REQUIRED')
    exclusive(root/'LAUNCH_INTENT.json',dict(manifest_sha256=sha(root/'RUN_MANIFEST.json'),epoch=time.time(),automatic_retry=False))
    env=dict(HOME=str(root/'home'),TMPDIR=str(root/'tmp'),HF_HOME=str(root/'cache/hf'),
        XDG_CACHE_HOME=str(root/'cache/xdg'),CUDA_CACHE_PATH=str(root/'cache/cuda'),
        CUDA_VISIBLE_DEVICES=c['GPU_uuid'],MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    duration=max(1,int(m['deadline_epoch']-time.time()))
    args=['systemd-run','--user','--unit='+root.name,'--property=Type=exec','--property=RemainAfterExit=yes',
        '--property=RuntimeMaxSec='+str(duration+60),'--property=TimeoutStopSec=30','--property=KillMode=control-group',
        '--property=CPUQuota=400%','--property=AllowedCPUs=0-3','--property=MemoryMax=48G','--property=TasksMax=256',
        '--property=StandardOutput=append:'+str(root/'stdout.log'),'--property=StandardError=append:'+str(root/'stderr.log'),
        '--working-directory='+str(root),'/usr/bin/env',*[k+'='+v for k,v in env.items()],
        '/bin/bash',str(root/'entry/infrastructure/enter.sh'),'4080-neural-evaluation',c['python'],'-B',
        str(root/'worker.py'),'--root',str(root),'--manifest-sha256',sha(root/'RUN_MANIFEST.json')]
    subprocess.run(args,capture_output=True,text=True,check=True,timeout=12)
    state=observe(root)
    if int(state['MainPID'])<=0:raise PermissionError('UNKNOWN_NATIVE_LAUNCH_IDENTITY')
    scope=registered_writer_scope(root,root.name+'.service',state,extra_pids=(os.getpid(),))
    exclusive(root/'ACTUAL_STARTED.json',dict(unit=state,process_scope=scope,epoch=time.time()))
    while True:
        state=observe(root)
        update(Path(c['root'])/('PROGRESS_'+root.name+'.json'),dict(unit=state,epoch=time.time(),
            progress=read(root/'FIT_PROGRESS.json')if(root/'FIT_PROGRESS.json').exists()else None))
        if state['MainPID']=='0':break
        if time.time()>m['deadline_epoch']+180:raise PermissionError('UNKNOWN_NATIVE_FINITE_CLOSURE')
        time.sleep(15)
    # A missing result is UNKNOWN; never synthesize scientific zero or retry.
    result=read(root/'RESULT.json');counts=audit_terminal(root,m,result)
    group=Path('/sys/fs/cgroup')/scope['ControlGroup'].lstrip('/')
    if group.exists()and any(f.read_text().strip()for f in group.rglob('cgroup.procs')):
        raise PermissionError('UNKNOWN_NATIVE_CHILD_CGROUP')
    if live_writers(root,scope=scope):raise PermissionError('UNKNOWN_NATIVE_RESIDUAL_WRITER')
    account=read(c['LOCAL']);row=account['V7_book_learning_runs'][root.name]
    if row['status']!='REGISTERED_NOT_STARTED'or row['manifest_sha256']!=sha(root/'RUN_MANIFEST.json'):
        raise PermissionError('UNKNOWN_CANONICAL_RUN_TRANSITION')
    counts.update(result_sha256=sha(root/'RESULT.json'),CPU_s=int(state['CPUUsageNSec'])/1e9,
        GPU_allocated_s_upper=math.ceil((int(state['ExecMainExitTimestampMonotonic'])
            -int(state['ExecMainStartTimestampMonotonic']))/1e6),
        TRAIN_window_visits=result.get('TRAIN_visits',0),
        all_costs_retained=True,scientific_success_not_inferred_from_exit=True)
    row.update(status='NATIVE_TERMINAL',actual=counts,closed_epoch=time.time(),all_old_campaign_costs_retained=True)
    once(c['LOCAL'],account,root.name+'-TERMINAL')
    lease=read(root/'control/LEASE.json');lease.update(status='CLOSED',result_sha256=sha(root/'RESULT.json'),closed_epoch=time.time())
    once(root/'control/LEASE.json',lease,root.name+'-LEASE-CLOSE')
    exclusive(root/'RELAY_CLOSED.json',dict(lifecycle='TERMINAL',effects_closed=True,actual=counts,
        native_status=result['status'],scientific_success_not_inferred_from_exit=True,process_scope=scope))
    return result


def predecessor(root):
    root=Path(root)
    return dict(root=str(root),unit=root.name+'.service',
        process_scope=read(root/'ACTUAL_STARTED.json')['process_scope'],
        files={n:sha(root/n)for n in('RUN_MANIFEST.json','RESULT.json','control/LEASE.json')})


def shared(c,steps):
    from scripts.pi0294.multitask.shared_training import recipe,KIND,qualified_entry
    from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
    from scripts.pi0294.multitask.book_gpu_capacity import TRAINING_REQUEST
    p=recipe(**c['shared_recipe_pins'],steps=steps)
    files={'TRAIN_FIT_PLAN.json':p,'RAW_PARENT_AUDIT.json':read(c['raw_audit']),
        'NEURAL_BINDINGS.private.json':dict(c['neural_bindings'],dataset=c['shared_data']),
        'M32_MEMORY_EXECUTION.json':LOW_RESIDENCY_POLICY}
    m=dict(kind=KIND,recipe=p,raw_parent_files_identity=p['raw_parent_files_identity'],raw_parent=c['raw_parent'],
        memory_execution_sha256=hashlib.sha256(raw(LOW_RESIDENCY_POLICY)).hexdigest(),training_GPU_entry_MiB=24576,
        GPU_training_qualification=TRAINING_REQUEST if steps==10 else None,
        GPU_training_admission=None if steps==10 else shared_training_certificate(c),
        limits=dict(model_loads=1,optimizer_updates=steps,physics_integrations=0,
            generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0))
    preview=dict(m,GPU_uuid=c['GPU_uuid'],sealed_inputs={'M32_MEMORY_EXECUTION.json':m['memory_execution_sha256']})
    qualified_entry(preview)
    return stage(c,c['shared_native10_name']if steps==10 else c['shared_formal_name'],
        'scripts/pi0294/multitask/shared_learning.py',m,files)


def shared_training_certificate(c):
    root=Path(c['run_parent'])/c['shared_native10_name']
    return dict(schema='SHARED4_RAW_SAME_GPU_NATIVE10_TRAINING_V1',qualification_root=str(root),
        files={n:sha(root/n)for n in('RUN_MANIFEST.json','RESULT.json','M32_CAPACITY_RESULT.json',
            'ACTUAL_DELTA_CHECKPOINT_RESTORE.json','ACTUAL_ACTIVATION_OFFLOAD.json','RAW_ZERO_STEP_PARAMETERS.json','control/LEASE.json')})


def shared_inference(c):
    from scripts.pi0294.multitask.shared_training import recipe
    from scripts.pi0294.multitask.shared_evaluation import INFERENCE_KIND,INFERENCE_LIMITS
    from scripts.pi0294.multitask.raw_parent_delivery import PARENT_ID
    p=recipe(**c['shared_recipe_pins'])
    return stage(c,c['shared_inference_name'],'scripts/pi0294/multitask/shared_policy.py',
        dict(kind=INFERENCE_KIND,raw_parent=c['raw_parent'],raw_zero_step=True,
            evaluation_checkpoint_manifest_sha256=PARENT_ID,raw_parent_files_identity=PARENT_ID,
            initialization_recipe=p,limits=INFERENCE_LIMITS),
        {'RAW_PARENT_AUDIT.json':read(c['raw_audit']),
         'NEURAL_BINDINGS.private.json':dict(c['neural_bindings'],dataset=c['shared_data'])})


def book(c,model):
    from scripts.pi0294.multitask import book_reset as reset
    producer=c['book_producers'][model];cp=checkpoint(producer)
    if cp is None:return dict(status='BLOCKED_NOT_RUN',reason='SPECIFIC_FULL_CHECKPOINT_UNAVAILABLE',valid=0,unrun=10)
    candidates={name:sha(value/'CHECKPOINT_COMPLETE.json')for name,path in c['book_producers'].items()
        if(value:=checkpoint(path))is not None}
    panel=reset.plan(candidates);base=dict(c['book_base_manifest']);base['delta_checkpoint']=str(cp)
    base.update(delta_checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),delta_producer_root=producer,
        delta_producer_closed_lease_sha256=sha(Path(producer)/'control/LEASE.json'),
        delta_native_qualification_file='ACTUAL_DELTA_CHECKPOINT_RESTORE.json',
        delta_native_qualification_sha256=sha(Path(producer)/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json'))
    from scripts.pi0294.multitask.book_phase_diagnostic import KIND as DIAGNOSTIC_KIND
    from scripts.pi0294.multitask.book_gpu_capacity import INFERENCE_REQUEST,qualified_dev_entry
    diagnostic=dict(base,kind=DIAGNOSTIC_KIND,GPU_inference_admission=None,GPU_inference_qualification=INFERENCE_REQUEST,
        limits=dict(model_loads=1,optimizer_updates=0,physics_integrations=0,generation_queries=96,RGB_pairs=0,source_constructors=0))
    d=stage(c,'night10-'+c['binding_id']+'-'+model.lower()+'-native96','scripts/pi0294/multitask/book_policy.py',diagnostic,
        {'NEURAL_BINDINGS.private.json':c['book_bindings'],'TRAIN32_DIAGNOSTIC_PLAN.json':c['book_diagnostic_plan']})
    result=execute(c,d)
    cert=dict(schema='N17_NATIVE96_CERTIFIED_DEV10_INFERENCE_V1',qualification_root=str(d),
        files={n:sha(d/n)for n in('RUN_MANIFEST.json','RESULT.json','GPU_INFERENCE_CAPACITY.json','control/LEASE.json')})
    preview=dict(base,kind=reset.KIND,GPU_uuid=c['GPU_uuid'],limits=reset.LIMITS,GPU_inference_qualification=None)
    qualified_dev_entry(preview,cert)
    previous=read(c['reset_first_predecessor']);rows=[]
    for index in range(10):
        if time.time()+c['native_seconds']>c['dispatch_deadline_epoch']:break
        reset.verify_predecessor(previous,current_root=Path(c['run_parent'])/reset.condition_run_name(panel,model,index),
            expected_previous=(model,index-1,reset.digest(panel))if index else None)
        m=dict(base,kind=reset.KIND,GPU_inference_admission=cert,GPU_inference_qualification=None,
            reset_model=model,reset_condition_index=index,limits=reset.LIMITS,
            reset_source_files=read(c['reset_qualification'])['source_files'])
        files={'NEURAL_BINDINGS.private.json':c['book_bindings'],'BOOK_RESET_DEV10_PLAN.json':panel,
            'RESET_NATIVE_QUALIFICATION.json':read(c['reset_qualification']),'RESET_PREDECESSOR.json':previous}
        actor=stage(c,reset.condition_run_name(panel,model,index),'scripts/pi0294/multitask/book_policy.py',m,files)
        result=execute(c,actor);rows.extend(result.get('conditions',[]))
        if result['status']!='COMPLETE_BOOK_RESET_DEV_CONDITION':break
        previous=predecessor(actor)
    return dict(status='CLOSED_BOOK_RESET_PANEL',valid=sum(r['valid_model_condition']for r in rows),
        successes=sum(r['success']for r in rows),unrun=10-len(rows),conditions=rows,checkpoint_sha256=candidates[model])


def start_formal_watch(c,config_path,root):
    """Transfer a staged finite job to another existing user unit, not to the PC.

    The caller releases the neural owner lock before invoking this function.
    It only acknowledges adoption after the new controller holds that lock.
    """
    root=Path(root);logs=Path(c['root'])/'watchers';logs.mkdir(exist_ok=True)
    name=root.name+'-continuation';duration=c['formal_seconds']+600
    exclusive(root/'CONTINUATION_INTENT.json',dict(unit=name+'.service',
        config_sha256=sha(config_path),epoch=time.time(),automatic_retry=False))
    argv=['systemd-run','--user','--unit='+name,'--property=Type=exec',
        '--property=RuntimeMaxSec='+str(duration),'--property=Restart=no',
        '--property=CPUQuota=100%','--property=AllowedCPUs=7','--property=MemoryMax=2G',
        '--property=StandardOutput=append:'+str(logs/(name+'.stdout')),
        '--property=StandardError=append:'+str(logs/(name+'.stderr')),
        '--working-directory='+c['root'],'/usr/bin/env','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+c['source']+':'+str(Path(c['source'])/'src'),
        c['python'],'-B','-m','scripts.pi0294.night10.native_relay','watch',
        '--config',str(config_path),'--actor-root',str(root),'--receipt',str(root/'CONTINUATION_FINAL.json')]
    subprocess.run(argv,capture_output=True,text=True,check=True,timeout=12)
    stop=time.monotonic()+30
    while not(root/'WATCHER_READY.json').exists():
        if time.monotonic()>stop:raise PermissionError('UNKNOWN_FORMAL_RESIDENT_ADOPTION')
        time.sleep(.2)
    ready=read(root/'WATCHER_READY.json');unit=observe(root.with_name(name))
    if unit['MainPID']!=str(ready['PID'])or unit['InvocationID']!=ready['InvocationID']:
        raise PermissionError('FORMAL_RESIDENT_START_IDENTITY_CHANGED')
    return dict(status='ADOPTED_FINITE_RESIDENT_SHARED8192',actor_root=str(root),
        checkpoint_complete=False,controller=ready,own_finite_seconds=c['formal_seconds'])


def effects_closed(c):
    roots=c.get('_actors',[])
    return bool(roots)and all((Path(r)/'control/LEASE.json').is_file()
        and read(Path(r)/'control/LEASE.json').get('status')=='CLOSED'for r in roots)


def audit_inputs(c):
    """Bounded readonly data/census checks; no Processor, model or rendering."""
    import numpy as np
    from scripts.pi0294.native_dataset import NativeDataset
    from scripts.pi0294.generalization.minimal_model import unchanged_raw_parent
    from scripts.pi0294.multitask.shared_task_sampling import TaskBalancedSampler,validate_fixed59
    from scripts.pi0294.multitask.shared_evaluation import checked_population
    from scripts.pi0294.multitask import book_phase_boundary as pb
    started=time.process_time();b=c['neural_bindings']
    unchanged_raw_parent(c['raw_parent'],read(c['raw_audit']))
    shared=NativeDataset(c['shared_data'],b['source'],b['cosmos'],for_optimization=True)
    bank=validate_fixed59(shared.rows);pairs=checked_population(read(Path(c['shared_data'])/'TASK_DEV_POPULATION.json'))
    files={r['file']:r for r,a in shared.rows}
    for name,row in files.items():
        path=Path(c['shared_data'])/name
        if sha(path)!=row['sha256']:raise PermissionError('NUMERIC_EXPERT_INPUT_CHANGED')
        with np.load(path,allow_pickle=False)as packet:
            if packet['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10':
                raise PermissionError('ACTUAL_EXECUTED_EXPERT_COMMANDS_ONLY')
    sampler=TaskBalancedSampler(shared)
    if sampler.sequence_sha256!=c['shared_sequence_sha256']:raise PermissionError('SHARED_SAMPLER_CHANGED')
    arm=c['D2_data'];data=NativeDataset(arm['root'],b['source'],b['cosmos'],for_optimization=True)
    index=read(c['phase_index']);pins=dict(index_sha256=sha(c['phase_index']),
        dataset_sha256=arm['manifest_sha256'],statistics_sha256=arm['statistics_sha256'],
        population_sha256=arm['population_sha256'])
    n=len(data.rows);seed=c['D2_seed']
    uniform=np.concatenate([np.random.Generator(np.random.PCG64(seed+i)).permutation(n)
        for i in range((131072+n-1)//n)])[:131072]
    sequences=dict(D2_UNIFORM4096_PLANNED=uniform,
        PB_CONTROL4096_PLANNED=pb.sequence(data.rows,index,boundary=False,seed=seed,**pins),
        PB_BOUNDARY4096_PLANNED=pb.sequence(data.rows,index,boundary=True,seed=seed,**pins))
    census={}
    for name,values in sequences.items():
        pin=hashlib.sha256(values.astype('<i8').tobytes()).hexdigest()
        if pin!=c['exposure']['sequences'][name]['sequence_int64_le_sha256']:
            raise PermissionError('FROZEN131072_DRAW_SEQUENCE_CHANGED')
        late=sum(data.rows[int(i)][0]['id'].startswith('BOOK_LATE_RECOVERY_')
            and data.rows[int(i)][1]<16 for i in values)
        census[name]=dict(visits=len(values),unique_windows=len(set(map(int,values))),
            late_first16_visits=late,sequence_sha256=pin)
    joints={}
    for name,value in c['controller_replay'].items():
        folder=Path(c['run_parent'])/value['run_id']/'conditions'/value['condition']
        if sha(folder/'ACTUAL_RECORDS.jsonl')!=value['raw_sha256']or sha(folder/'NATIVE_SCENE.private.xml')!=value['XML_sha256']:
            raise PermissionError('SAVED_CONTROLLER_EVIDENCE_CHANGED')
        joints[name]=dict(commanded_margin_rad=value['stop']['commanded_margin'][3],
            actual_margin_rad=value['stop']['actual_margin'][3],
            actual_minus_reconstructed_command_rad=value['joint4_actual_minus_command_rad'],
            command_is_reconstruction_not_direct_record=True,Teacher_or_WBC_cause_not_proven=True)
    if 'torch' in sys.modules:raise PermissionError('CPU_AUDIT_MUST_NOT_IMPORT_TORCH')
    return dict(status='PASS_CPU_FROZEN_INPUTS_AND_SAMPLING',effects_closed=True,
        common59=dict(full=46,suffixes=13,windows=len(shared.rows),task_windows={t:sum(map(len,v.values()))for t,v in bank.items()}),
        original_DEV_counts={t:len(rows)for t,rows in pairs['tasks'].items()},
        raw_seven_file_stat_matches_trusted_full_hash=True,census=census,joint4=joints,
        optimizer_updates=0,model_loads=0,physics_integrations=0,CPU_s=time.process_time()-started)


def monitor(c):
    """Read live custody into Night10 readiness files; never dispatch a worker."""
    root=Path(c['root']);m13=Path(c['existing_M13']);started=time.process_time()
    while time.time()<c['dispatch_deadline_epoch']-120:
        slot=False;reason='EXISTING_NEURAL_OWNER_BUSY';lock=None
        try:
            lock=owner(c);slot=True
            for f in m13.glob('jobs/*/*_STAGED.private.json'):
                actor=Path(read(f)['root'])
                if observe(actor)['MainPID']!='0'or read(actor/'control/LEASE.json')['status']!='CLOSED':
                    slot=False;reason='EXISTING_NATIVE_EFFECT_NOT_CLOSED';break
            account=read(c['LOCAL'])
            if any(account.get(k)!=[]for k in('unknown_effects','pending_updates','pending_queries')):
                slot=False;reason='CANONICAL_UNKNOWN_EFFECT'
            for row in account.get('V7_book_learning_runs',{}).values():
                if not row.get('role','').startswith('V7_NATIVE_RELAY__'):continue
                actor=Path(row['root']);lease=actor/'control/LEASE.json'
                if (not lease.is_file()or read(lease).get('status')!='CLOSED'
                        or observe(actor)['MainPID']!='0'):
                    slot=False;reason='NEW_NATIVE_EFFECT_NOT_CLOSED';break
        except BlockingIOError:pass
        finally:
            if lock:lock.close()
        update(root/'NEURAL_SLOT.json',dict(available=slot,reason='COLD_NEURAL_SLOT_AVAILABLE'if slot else reason,epoch=time.time()))
        for name,producer in c['book_producers'].items():
            update(root/('MODEL_'+name+'.json'),dict(available=checkpoint(producer)is not None,producer=producer,epoch=time.time()))
        state=read(m13/'STATE.json');progress=Path(c['book_producers']['D2'])/'FIT_PROGRESS.json'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
        fs=os.statvfs(root)
        update(root/'NIGHT_SUMMARY.json',dict(epoch=time.time(),dispatch_deadline_epoch=c['dispatch_deadline_epoch'],
            current_M13=state,current_D2=read(progress)if progress.exists()else None,
            new_tasks=read(root/'queue/STATE.json')if(root/'queue/STATE.json').exists()else None,
            task_receipts={f.name:read(f)for f in(root/'receipts').glob('*.json')},
            real_resources=dict(GPUs=gpu,available_RAM_bytes=available,free_disk_bytes=fs.f_bavail*fs.f_frsize),
            new_policy_success_not_inferred_from_process_exit=True,monitor_CPU_s=time.process_time()-started))
        time.sleep(15)
    return dict(status='CLOSED_FINITE_READONLY_MONITOR',effects_closed=True,CPU_s=time.process_time()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','audit','monitor','shared10','shared16','shared8192','watch','book'))
    parser.add_argument('--config',required=True);parser.add_argument('--model');parser.add_argument('--receipt')
    parser.add_argument('--actor-root')
    a=parser.parse_args();c=read(a.config);c['_config_sha256']=sha(a.config);admission(c)
    if a.action=='preflight':
        if time.time()+c['native_seconds']>=c['dispatch_deadline_epoch']:
            raise PermissionError('FINITE_NEW_DISPATCH_WINDOW_REQUIRED')
        if a.model:
            from scripts.pi0294.multitask.book_reset import source_files
            qualification=read(c['reset_qualification'])
            if(not qualification['passed']or qualification['source_files']!=source_files()
                    or checkpoint(c['book_producers'][a.model])is None):
                raise PermissionError('OWN_FULL_MODEL_AND_ACTUAL_RESET_QUALIFICATION_REQUIRED')
        print(json.dumps(dict(status='PASS_COLD_RELAY_METADATA_NO_GPU',models=0,physics_integrations=0)))
        return 0
    if not a.receipt:raise ValueError('EXACT_TASK_RECEIPT_PATH_REQUIRED')
    if a.action in('audit','monitor'):
        value=audit_inputs(c)if a.action=='audit'else monitor(c)
        exclusive(a.receipt,dict(value,epoch=time.time()));return 0
    lock=owner(c);failure=None;value=None
    update(Path(c['root'])/'NEURAL_SLOT.json',dict(available=False,epoch=time.time(),owner_PID=os.getpid()))
    try:
        if a.action=='watch':
            root=Path(a.actor_root)
            if (root.parent!=Path(c['run_parent'])or root.name!=c['shared_formal_name']
                    or read(root/'RELAY_CUSTODY.json')['config_sha256']!=c['_config_sha256']):
                raise PermissionError('EXACT_PREVIOUSLY_STAGED_FORMAL_HANDOFF')
            c['_actors']=[str(root)];unit=observe(root.with_name(root.name+'-continuation'))
            exclusive(root/'WATCHER_READY.json',dict(PID=os.getpid(),
                proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                InvocationID=unit['InvocationID'],epoch=time.time(),neural_owner_lock_held=True,
                PC_or_Vanda_dependency=False))
            result=execute(c,root)
            value=dict(status='CLOSED_FINITE_SHARED8192',native_status=result['status'],
                optimizer_updates=result['optimizer_updates'],actor_root=str(root),
                checkpoint_manifest_sha256=sha(root/'training/checkpoint-8192/CHECKPOINT_COMPLETE.json')
                    if(root/'training/checkpoint-8192/CHECKPOINT_COMPLETE.json').is_file()else None,
                restore_sha256=sha(root/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json')
                    if(root/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json').is_file()else None)
        elif a.action=='shared8192':
            if time.time()>=c['dispatch_deadline_epoch']:raise PermissionError('TEN_HOUR_DISPATCH_WINDOW_CLOSED')
            root=shared(c,8192);lock.close()
            value=start_formal_watch(c,a.config,root)
        elif a.action=='book':value=book(c,a.model)
        else:
            actor=shared(c,10)if a.action=='shared10'else shared_inference(c)
            result=execute(c,actor)
            if a.action=='shared10':
                # Validate the same-device full-state certificate against the
                # actual formal recipe; never treat native10 as a policy.
                from scripts.pi0294.multitask.shared_training import recipe,qualified_entry
                from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
                m=read(actor/'RUN_MANIFEST.json');p=recipe(**c['shared_recipe_pins'])
                m.update(recipe=p,GPU_training_qualification=None,GPU_training_admission=shared_training_certificate(c),
                    limits=dict(model_loads=1,optimizer_updates=8192,physics_integrations=0,
                        generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0))
                qualified_entry(m)
            else:
                from scripts.pi0294.multitask.shared_evaluation import KIND,LIMITS
                from scripts.pi0294.multitask.shared_policy import qualified_inference_entry
                cert=dict(schema='SHARED4_SAME_GPU_NATIVE16_INFERENCE_V1',qualification_root=str(actor),
                    files={n:sha(actor/n)for n in('RUN_MANIFEST.json','RESULT.json',
                        'SHARED_GPU_INFERENCE_CAPACITY.json','ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json','control/LEASE.json')})
                m=read(actor/'RUN_MANIFEST.json');m.update(kind=KIND,limits=LIMITS,shared_inference_admission=cert)
                qualified_inference_entry(m)
            value=dict(status='PASS_OWN_RAW_QUALIFICATION',root=str(actor),result_sha256=sha(actor/'RESULT.json'))
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        closed=effects_closed(c)
        value=dict(status='CLOSED_KNOWN_FAILURE'if closed else'UNKNOWN_EFFECT',error=failure,effects_closed=closed)
    finally:
        lock.close();exclusive(a.receipt,dict(value,effects_closed=value.get('effects_closed',True),epoch=time.time()))
    return 0 if failure is None else 1


if __name__=='__main__':raise SystemExit(main())
