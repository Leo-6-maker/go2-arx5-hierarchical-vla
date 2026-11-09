"""Thin Core single-GPU adapter to the actual target GR00T Trainer.

Uses the existing neural lock, registry, Journal and durable writer. The PBS
producer stays independent. This finite local diagnostic cannot run physics,
change frozen controls, reopen a lease or deliver a formal2000 candidate.
"""
import json,os,subprocess,time,signal
from pathlib import Path
from contextlib import closing
from . import target_gr00t_profile_v1 as profile
from .storage import digest

SCOPE='TARGET_GR00T_CORE_V1'


def full_restore_only(c):
    operation=c.get('operation')
    profile.need(operation in(None,'FULL_ACTION_RESTORE_ONLY'),'DECLARED_CORE_NATIVE_OPERATION')
    if operation is None:return False
    profile.need(c.get('original_Trainer_execution')is True
        and c.get('formal_training')is False
        and c.get('memory_policy')in('CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2','CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3','CORE_FROZEN_BACKBONE_SWAP_ACTIVATIONS_4G_V4','CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5')
        and isinstance(c.get('memory_reference'),dict)
        and set(c['memory_reference'])=={'path','sha256'},
        'CORE_FULL_ACTION_NATIVE_RESTORE_NOT_CACHE_OR_FORMAL')
    return True


def activation_policy(maximum=2*1024**3):
    from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
    profile.need(maximum in(2*1024**3,4*1024**3),'DECLARED_LOSSLESS_CORE_ACTIVATION_CAPACITY')
    return dict(LOW_RESIDENCY_POLICY,schema='CORE_LOSSLESS_CPU_ACTIVATIONS_'+str(maximum//1024**3)+'G_V1',
        per_microbatch_activation_offload_bytes=maximum)


def trim_unused_logits(model):
    """Pinned Qwen backbone consumes hidden_states only, never language logits.

    Qwen's official logits_to_keep option changes only its final unused vocab
    projection. No weights, hidden tokens, masks, precision or GR00T loss move.
    Actual matched full-H40 prediction equality is required before optimization.
    """
    import inspect
    language=model.backbone.model
    profile.need('logits_to_keep'in inspect.signature(language.forward).parameters,
        'OFFICIAL_QWEN_UNUSED_LOGITS_OPTION')
    def select_one(module,args,kwargs):
        profile.need(kwargs.get('labels')is None and kwargs.get('output_hidden_states')is True,
            'FROZEN_FEATURES_ONLY_NOT_LANGUAGE_TRAINING')
        profile.need(kwargs.get('logits_to_keep',0)in(0,1),'NO_OTHER_LOGITS_SELECTION')
        return args,dict(kwargs,logits_to_keep=1)
    return language.register_forward_pre_hook(select_one,with_kwargs=True)


def verify_memory_predictions(reference,current):
    import numpy as np
    path=Path(reference['path']);profile.need(digest(path)==reference['sha256'],'SEALED_ORIGINAL_MEMORY_REFERENCE')
    original=json.loads(path.read_bytes())['records'];folder=Path(current)
    actual=json.loads((folder/'RESULT.json').read_bytes())['records']
    profile.need(len(original)==len(actual)==4,'FOUR_REAL_MEMORY_EXECUTION_COMPARISONS')
    for a,b in zip(original,actual):
        profile.need(all(a[k]==b[k]for k in ('episode','anchor','seed','noise','command')),
            'EXACT_MEMORY_REFERENCE_INPUTS_NOISE_COMMAND')
        first=path.parent/(a['episode']+'.npz');second=folder/(b['episode']+'.npz')
        profile.need(digest(first)==a['sha256']and digest(second)==b['sha256'],'PINNED_REAL_MEMORY_PREDICTIONS')
        with np.load(first,allow_pickle=False)as x,np.load(second,allow_pickle=False)as y:
            profile.need(all(np.array_equal(x[k],y[k])for k in ('physical_H40','normalized','actual_state10','expert_policy_H40')),
                'ACTUAL_MEMORY_EXECUTION_CHANGED_MODEL_OUTPUT')
    return dict(status='PASS_ACTUAL_FOUR_MATCHED_FULL_H40_BIT_EXACT',observations=4,
        reference_sha256=reference['sha256'],tolerances_relaxed=False,weights_or_dtype_changed=False)


def gpu(spec):
    from .pilot1_scope import GPU_UUIDS
    profile.need(spec['GPU_uuid']in GPU_UUIDS,'KNOWN_CORE_GPU_UUID')
    q=subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],
        '--query-gpu=uuid,name,memory.free,memory.total','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=15)
    uuid,name,free,total=[s.strip()for s in q.stdout.split(',')]
    profile.need(uuid==spec['GPU_uuid']and name=='NVIDIA GeForce RTX 4080 SUPER'
        and os.environ.get('CUDA_VISIBLE_DEVICES')==uuid,'ACTUAL_CORE_SINGLE_SHARED_DEVICE')
    return dict(uuid=uuid,name=name,free_bytes=int(free)*1024**2,total_bytes=int(total)*1024**2)


def scope(c):
    from .pilot1_storage import local_path
    profile.need(os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
        and not os.environ.get('PBS_JOBID'),'ACTUAL_CORE_OWNER_NOT_PBS')
    root=Path(c['root']);lease=json.loads((root/'control/LEASE.json').read_bytes())
    full_restore_only(c)
    profile.need(c['scope']==SCOPE and c['physics']is False and c['WBC_changes']is False
        and c['baseline_changed']is False and c['allow_internal_GR00T_updates']is True
        and lease.get('status')=='ACTIVE'and lease.get('config_sha256')==digest(root/'ALLOCATION_CONFIG.json')
        and time.time()+1200<c['deadline_epoch']and not(root/'STOP.json').exists(),
        'FRESH_FINITE_REGISTERED_CORE_TRAINING_SCOPE')
    reg=json.loads(Path(c['registry']).read_bytes())
    profile.need(any(e.get('path')==str(root)and e.get('scope')==SCOPE
        and e.get('local_lease')==str(root/'control/LEASE.json')for e in reg['entries']),
        'ORIGINAL_CORE_REGISTRY_REQUIRED')
    for path in (root,c['source_root'],c['bindings']['dataset'],c['bindings']['source'],
                 c['bindings']['cosmos'],c['parent']['path'],c['python']):local_path(c,path)
    return root,reg


def site(s):
    """Actual ownership/storage admission before importing Torch or a model."""
    p=profile.validate(s);root=Path(s['registered_root'])
    cpath=Path(s['core_scope_root'])/'ALLOCATION_CONFIG.json'
    profile.need(digest(cpath)==s['core_scope_sha256'],'PINNED_NEW_CORE_SCOPE')
    c=json.loads(cpath.read_bytes());master,reg=scope(c)
    profile.need(spec_operation(s)==spec_operation(c), 'CORE_NATIVE_OPERATION_MATCHES_ALLOCATION')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    profile.need(root.parent==master and lease.get('status')=='ACTIVE'
        and lease.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
        and lease.get('job_id')==c['generation_id']
        and s['deadline_epoch']==c['deadline_epoch']and s['GPU_uuid']==c['GPU_uuid']
        and all(s[k]==c[k]for k in ('source_root','source_sha','source_files','bindings','parent'))
        and p['dataset_sha256']==c['dataset_sha256']and p['statistics_sha256']==c['statistics_sha256']
        and p.get('memory_policy','CUDA_RESIDENT')==c.get('memory_policy','CUDA_RESIDENT')
        and any(e.get('path')==str(root)and e.get('scope')==SCOPE
            and e.get('source_sha')==s['source_sha']for e in reg['entries']),
        'ACTUAL_REGISTERED_CORE_CHILD_LEASE')
    for name,pin in s['source_files'].items():
        profile.need(digest(Path(s['source_root'])/name)==pin,'CORE_SEALED_SOURCE:'+name)
    for name,pin in c['dataset_files'].items():
        profile.need(digest(Path(s['bindings']['dataset'])/name)==pin,'CORE_SEALED_DATA:'+name)
    actual=gpu(s)
    profile.need(0<s['allocator_limit_bytes']<=actual['free_bytes']-2*1024**3,
        'CORE_SHARED_DEVICE_TWO_GIB_MARGIN')
    return p


def spec_operation(value):
    return value.get('core_native_operation',value.get('operation'))


def run(c):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .target_gr00t_allocation_v1 import compare_next_update,fit4_gate
    root,reg=scope(c);full=full_restore_only(c)
    stop=[];children=[];started=time.monotonic()
    signal.signal(signal.SIGTERM,lambda sig,frame:stop.append(sig))
    signal.signal(signal.SIGUSR1,lambda sig,frame:stop.append(sig))
    journal=Journal(root/'control/journal',root.name)
    with closing(OwnerLock(Path(c['owner_root'])/'NIGHT1_NEURAL_OWNER.lock')):
        profile.need(not(root/'ALLOCATION_STARTED.json').exists(),'NEVER_REPLAY_CORE_ALLOCATION')
        write(root/'ALLOCATION_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=c['source_sha']))
        def child(name,mode,endpoint,*,resume=None,start_step=0,generation=None):
            scope(c);profile.need(not stop,'CORE_STOP_REQUESTED')
            r=root/name
            def register():
                profile.need(not r.exists(),'FRESH_CORE_CHILD')
                actual=gpu(c);limit=min(c['allocator_limit_bytes'],actual['free_bytes']-2*1024**3)
                profile.need(limit>8*1024**3,'CORE_HEADROOM_BEFORE_MODEL_START')
                r.mkdir()
                for n in ('control','tmp','cache'):(r/n).mkdir()
                alias=r/'model_inputs/nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True)
                alias.symlink_to(c['bindings']['cosmos'],target_is_directory=True)
                recipe=profile.recipe(c['dataset_sha256'],c['statistics_sha256'],mode=mode,
                    variant='FULL_ACTION_SIDE'if full else'INTERFACES_LAST2_DIT',endpoint=endpoint,microbatch=1,
                    memory_policy=c.get('memory_policy','CUDA_RESIDENT'))
                s=dict(registered_root=str(r),source_root=c['source_root'],source_sha=c['source_sha'],
                    source_files=c['source_files'],bindings=c['bindings'],parent=c['parent'],
                    profile=profile.PROFILE,kind=profile.KIND,execution_site='CORELAB_4080S',recipe=recipe,
                    allocation_request_id=c['generation_id'],generation_id=generation or c['generation_id']+'-'+name,
                    start_step=start_step,resume=resume,GPU_uuid=c['GPU_uuid'],allocator_limit_bytes=limit,
                    GR00T_updates=endpoint-start_step,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
                    physics=False,WBC_changes=False,parent_manifest_sha256=profile.PARENT,role='TARGET-CORE-'+name,
                    registry=c['registry'],deadline_epoch=c['deadline_epoch'],closeout_reserve_s=1200,
                    core_scope_root=str(root),core_scope_sha256=digest(root/'ALLOCATION_CONFIG.json'))
                if c.get('memory_reference'):s['memory_reference']=c['memory_reference']
                if full:s['core_native_operation']='FULL_ACTION_RESTORE_ONLY'
                profile.validate(s);write(r/'EXECUTION_SPEC.json',s)
                with closing(OwnerLock(Path(c['owner_root'])/'OWNER.lock')):
                    registry,pin=snapshot(c['registry'])
                    profile.need(not any(e.get('path')==str(r)for e in registry['entries']),'FRESH_CORE_REGISTRY')
                    registry['entries'].append(dict(path=str(r),role=s['role'],scope=SCOPE,
                        local_lease=str(r/'control/LEASE.json'),owner='EXISTING_LOCAL_PC',source_sha=c['source_sha']))
                    once(Path(c['registry']),registry,root.name+'-'+name+'-REGISTER',expected_before_sha256=pin)
                write(r/'control/LEASE.json',dict(status='ACTIVE',job_id=c['generation_id'],
                    spec_sha256=digest(r/'EXECUTION_SPEC.json'),deadline_epoch=c['deadline_epoch']))
                return dict(spec_sha256=digest(r/'EXECUTION_SPEC.json'))
            registered=journal.once(name+'-REGISTER',dict(mode=mode,endpoint=endpoint,resume=resume),register)
            def execute():
                env=dict(os.environ,PYTHONPATH=c['source_root']+'/src:'+c['source_root'],
                    TMPDIR=str(r/'tmp'),XDG_CACHE_HOME=str(r/'cache'),HF_HOME=str(r/'cache/hf'),
                    HF_MODULES_CACHE=str(r/'cache/hf-modules'),TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',
                    NO_ALBUMENTATIONS_UPDATE='1')
                with (r/'stdout.log').open('xb')as out,(r/'stderr.log').open('xb')as err:
                    proc=subprocess.Popen([c['python'],'-B','-m','scripts.wm02.train_target_gr00t_v1',
                        '--spec',str(r/'EXECUTION_SPEC.json')],cwd=c['source_root'],env=env,
                        stdout=out,stderr=err,start_new_session=True)
                    write(r/'ACTUAL_STARTED.json',dict(pid=proc.pid,epoch=time.time()))
                    sent=False
                    while proc.poll()is None:
                        if (stop or time.time()+1200>c['deadline_epoch'])and not sent:
                            os.killpg(proc.pid,signal.SIGTERM);sent=True
                        if time.time()>c['deadline_epoch']:
                            os.killpg(proc.pid,signal.SIGKILL)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                value=json.loads((r/'RESULT.json').read_bytes());lease=json.loads((r/'control/LEASE.json').read_bytes())
                profile.need(lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==proc.returncode,
                    'CORE_CHILD_CLOSED_WITH_ACTUAL_EXIT')
                return dict(name=name,root=str(r),exit_code=proc.returncode,result_sha256=digest(r/'RESULT.json'),
                    GR00T_updates=value.get('GR00T_updates',0),global_step=value.get('global_step'),
                    checkpoint=value.get('checkpoint'),wall_s=value.get('wall_s'),
                    endpoint_complete=value.get('global_step')==endpoint)
            row=journal.once(name+'-EXECUTE',registered,execute);children.append(row)
            write(root/'CHILDREN_ACTUAL.json',dict(children=children));return row
        if full:
            generation=c['generation_id']+'-restore4'
            continuous=child('continuous4','RESTORE4',4,generation=generation)
            profile.need(continuous['exit_code']==0 and continuous['endpoint_complete'],
                'CORE_FULL_NATIVE_CONTINUOUS4_FAILURE_PRESERVED')
            split=child('split2','RESTORE4',2,generation=generation)
            profile.need(split['exit_code']==0 and split['endpoint_complete'],
                'CORE_FULL_NATIVE_SPLIT2_FAILURE_PRESERVED')
            cp=Path(split['checkpoint'])
            resumed=child('resume4','RESTORE4',4,start_step=2,generation=generation,
                resume=dict(path=str(cp),sha256=digest(cp/'CHECKPOINT_COMPLETE.json')))
            profile.need(resumed['exit_code']==0 and resumed['endpoint_complete'],
                'CORE_FULL_NATIVE_RESUME4_FAILURE_PRESERVED')
            proof=compare_next_update(continuous['checkpoint'],resumed['checkpoint'])
            profile.need(json.loads((Path(continuous['checkpoint'])/'SAMPLER_STATE.json').read_text())['cursor']==128,
                'CORE_FULL_NATIVE_FOUR_STEP_CURSOR128')
            write(root/'FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(proof,
                comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'))
            return dict(status='CORE_ORIGINAL_TRAINER_FULL_ACTION_RESTORE_ONLY_SEALED',
                children=children,proof=proof,optimizer_calls=8,original_Trainer_execution=True,
                formal_training=False,physics_integrations=0,wall_s=time.monotonic()-started)
        cap=child('capacity2','CAPACITY',2)
        profile.need(cap['exit_code']==0 and cap['endpoint_complete'],'CORE_CAPACITY_FAILURE_PRESERVED')
        cp=Path(cap['checkpoint']);s=json.loads((Path(cap['root'])/'EXECUTION_SPEC.json').read_bytes())
        restored=child('capacity-resume1','CAPACITY',2,start_step=1,
            resume=dict(path=str(cp.parent/'checkpoint-1'),sha256=digest(cp.parent/'checkpoint-1/CHECKPOINT_COMPLETE.json')),
            generation=s['generation_id'])
        profile.need(restored['exit_code']==0 and restored['endpoint_complete'],'CORE_INDEPENDENT_RESTORE_FAILED')
        proof=compare_next_update(cp,restored['checkpoint']);write(root/'NEXT_UPDATE_RESTORE_ACTUAL.json',proof)
        steps=[json.loads(x)for x in (Path(cap['root'])/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]
        forecast=cap['wall_s']+200*max(x['optimizer_update_wall_s']for x in steps)*1.25+600
        write(root/'FIT4_TIME_ADMISSION.json',dict(predicted_wall_s=forecast,remaining_s=c['deadline_epoch']-time.time()))
        if time.time()+forecast+1200>=c['deadline_epoch']:
            return dict(status='CORE_CAPACITY_RESTORE_PASSED_WAIT_FRESH_FIT4_WINDOW',children=children)
        fit=child('fit4-200','FIT4',200)
        profile.need(fit['exit_code']==0 and fit['endpoint_complete'],'CORE_FIT4_PARTIAL_OR_FAILURE_PRESERVED')
        gate=fit4_gate(fit['root']);write(root/'FIT4_GATE_200.json',gate)
        return dict(status='CORE_FIT4_CHAIN_SEALED',children=children,fit4_gate=gate,
            actual_GR00T_updates=sum(r['GR00T_updates']for r in children),
            wall_s=time.monotonic()-started,formal2000_started=False,physics=0)
