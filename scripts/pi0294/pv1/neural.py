"""PV1 baseline, primary/local-replica blocks and terminal DEV primitives."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.neural import checkpoint_identity,serve
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.native_train import parameter_inventory
from scripts.pi0294.numerics import prepare_environment,configure_torch
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from .data_delivery import consume
from .dataset import PV1Dataset,POPULATION
from .evaluation_plan import session_spec,final_session_spec,validate_registry,remaining_panel
from .evaluation_plan import development,final_confirmation,POPULATION_FILE,FINAL_PLAN
from .decision import AUTHORITY
from .sampler import ARMS
from .native_admission import admit
from .native_transport_qualification import run as qualify_transport
from .neural_ledger import Ledger
from .evaluation_ledger import for_request as ledger_for_request
from .pre_model_qualification import require_current
from .probes import run as probes
from .recipe import PARENT,validate_primary,run_key
from .training import run as train,checked_resume
from .visibility import refresh
from .baseline import admission as dev_admission
from . import evaluation_platform as evaluation


TRAIN_STAGES=('P2_PRIMARY','P4_REPLICA')
STAGES=('B0_DEV','P2_PRIMARY','P4_DEV','P4_REPLICA','P5_FINAL')


def require_native_device(request,allocation,torch):
    """Physical device only; admit/qualification remain mandatory before this."""
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    count=torch.cuda.device_count() if visible else 0
    name=torch.cuda.get_device_name(0) if count==1 else None
    if any('platform_evaluation_version' in v for v in (request,allocation)):
        evaluation.require_device(request,allocation,visible=visible,count=count,name=name)
    elif not visible or count!=1 or 'H200' not in name:
        raise PermissionError('PV1_ONE_VISIBLE_NATIVE_H200_ONLY')


def replica_selection(root,input_,stage):
    """Recompute the P2 winner from sealed actual DEV evidence, not summaries."""
    root=Path(root);selection=input_('plans/PV1_P4_REPLICA_SELECTION.json')
    if selection.get('local_recipe_stage')!='P2':
        raise PermissionError('PV1_P3_REPLICATION_2048_NOT_IMPLEMENTED')
    input_(POPULATION_FILE)
    if (sha(root/POPULATION_FILE)!=POPULATION
            or selection.get('schema')!='PI0294_PV1_P2_REPLICA_SELECTION_V1'
            or selection.get('authority_release')!=AUTHORITY
            or selection.get('population_plan_sha256')!=POPULATION
            or selection.get('route')!=stage['route']
            or selection.get('new_bank_manifest_sha256')!=stage['new_bank_manifest_sha256']
            or selection.get('LOCKBOX_opened') is not False
            or selection.get('replica_seed')!=20261020
            or selection.get('replica_steps')!=4096
            or set(selection.get('candidates',{}))!={'B0_CORR',*ARMS[stage['route']]}):
        raise PermissionError('PV1_EXACT_P2_PAIR_AND_B0_ONLY_UNOPENED_LOCKBOX_SELECTION')
    summaries={}
    for candidate,declared in selection['candidates'].items():
        checkpoint=declared.get('checkpoint_manifest_sha256','')
        if (not re.fullmatch('[0-9a-f]{64}',checkpoint)
                or declared.get('global_step')!=(2048 if candidate=='B0_CORR' else 4096)
                or candidate=='B0_CORR' and checkpoint!=PARENT):
            raise PermissionError('PV1_SELECTION_EXACT_TERMINAL_CHECKPOINT_IDENTITY')
        evidence=[]
        for item in declared['evidence']:
            actual={}
            for kind,name in item.items():
                input_(name)
                actual[kind]=(root/name).read_bytes()
            evidence.append(actual)
        verified=dev_admission((root/POPULATION_FILE).read_bytes(),route=stage['route'],
            new_bank_manifest_sha256=stage['new_bank_manifest_sha256'],evidence=evidence,
            candidate=candidate,checkpoint_manifest_sha256=checkpoint)
        valid=[json.loads(item['adjudication']) for item in evidence
            if json.loads(item['adjudication'])['status'] in ('SUCCESS','VALID_TASK_FAIL')]
        summaries[candidate]=dict(successes=verified['successes'],
            S1_rejections=sum(r['S1_rejections'] for r in valid))
    baseline=summaries['B0_CORR']['successes']
    # P2 arms have equal intervention complexity; their predeclared IDs break ties.
    ranked=sorted(ARMS[stage['route']],key=lambda c:(-min(summaries[c]['successes'].values()),
        -sum(summaries[c]['successes'].values()),summaries[c]['S1_rejections'],c))
    winner=ranked[0];scores=summaries[winner]['successes']
    if (selection.get('winner')!=winner or min(scores.values())<18
            or any(scores[t]<baseline[t]-1 for t in scores)
            or min(scores.values())<=min(baseline.values())):
        raise PermissionError('PV1_REPLICA_ONLY_ACTUAL_DEV_QUALIFIED_PREDECLARED_WINNER')
    return winner


def stage_work(root,request,*,allow_pending_segment_copy=False):
    if request['stage'] not in STAGES:
        raise PermissionError('PV1_ONLY_IMPLEMENTED_PREDECLARED_PHASES')
    root=Path(root);m=read(root/'RUN_MANIFEST.json');sealed=m['execution']['sealed_inputs']
    def input_(name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise PermissionError('PV1_ONLY_OWN_SEALED_PHASE_INPUT_PATHS')
        p=root/name
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_mode&0o222 or sealed.get(name)!=sha(p)):
            raise PermissionError('PV1_PRESEALED_NATIVE_PHASE_INPUT:'+name)
        return read(p)
    stage=input_('PV1_STAGE_BINDING.json')
    from scripts.pi0294.dec1.source_admission import admitted as dec1_admitted
    dec1_source_training=dec1_admitted(root,m,request,stage,input_)
    evaluation=input_(FINAL_PLAN if request['stage']=='P5_FINAL' else 'plans/SOURCE_EVALUATION_PLAN.json')
    prior=input_('plans/PRIOR_EVALUATION_OUTCOMES.json')
    if request['stage']=='P5_FINAL':
        checkpoint=input_('plans/P5_CANDIDATE_CHECKPOINT.private.json')
        freeze=evaluation['candidate_freeze'];expected=final_confirmation(input_(POPULATION_FILE)['groups'],freeze)
        if (evaluation!=expected or request.get('training_block')
                or request.get('evaluation_candidate')!=freeze['candidate']
                or checkpoint.get('candidate')!=freeze['candidate'] or checkpoint.get('global_step')!=4096
                or checkpoint.get('checkpoint_manifest_sha256')!=freeze['checkpoint_manifest_sha256']):
            raise PermissionError('PV1_P5_ONLY_EXACT_UNIQUE_FROZEN_CANDIDATE_AND_LOCKBOX80')
        p=Path(checkpoint['checkpoint']);complete=read(p/'CHECKPOINT_COMPLETE.json')
        if (p.stat().st_uid!=os.getuid() or sha(p/'CHECKPOINT_COMPLETE.json')!=freeze['checkpoint_manifest_sha256']
                or complete.get('global_step')!=4096 or complete.get('full_resume') is not True):
            raise PermissionError('PV1_P5_ACTUAL_COMPLETE_NATIVE_FROZEN_CHECKPOINT_REQUIRED')
        count=sum(len(remaining_panel(prior,final_session_spec(root,label))) for label in evaluation['panels'])
        if not count:raise PermissionError('PV1_P5_NO_REMAINING_LOCKBOX_CONDITIONS_NO_NEW_MODEL_LOAD')
        return
    if request['stage']=='P4_DEV':
        checkpoint=input_('plans/EVALUATION_CHECKPOINT.private.json')
        allowed=ARMS[stage['route']]+tuple('REPLICA_'+a for a in ARMS[stage['route']])
        if (evaluation['candidate'] not in allowed
                or request['evaluation_candidate']!=evaluation['candidate'] or request.get('training_block')
                or checkpoint['candidate']!=evaluation['candidate'] or checkpoint['global_step']!=4096):
            raise PermissionError('PV1_ONLY_FROZEN_TERMINAL_PRIMARY_CHECKPOINT_DEVELOPMENT')
        p=Path(checkpoint['checkpoint']);complete=read(p/'CHECKPOINT_COMPLETE.json')
        if (p.stat().st_uid!=os.getuid() or sha(p/'CHECKPOINT_COMPLETE.json')!=checkpoint['checkpoint_manifest_sha256']
                or complete['global_step']!=4096 or complete['full_resume'] is not True):
            raise PermissionError('PV1_ACTUAL_COMPLETE_NATIVE_4096_CHECKPOINT_REQUIRED')
        if evaluation['candidate'].startswith('REPLICA_'):
            arm=evaluation['candidate'].removeprefix('REPLICA_')
            if replica_selection(root,input_,stage)!=arm:
                raise PermissionError('PV1_REPLICA_DEV_ONLY_ACTUAL_SELECTED_LOCAL_RECIPE')
            saved=read(p/'SAMPLER_STATE.json');recipe=input_('plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][evaluation['candidate']]
            validate_primary(recipe,input_('plans/FROZEN_SOURCE_PAIR_RECIPE.json'))
            value=read(Ledger(root).path)
            if (run_key(recipe)!=evaluation['candidate'] or recipe['replication_of_local_recipe'] is not True
                    or recipe['seed']!=20261020 or saved.get('recipe')!=recipe
                    or complete.get('files',{}).get('SAMPLER_STATE.json')!=sha(p/'SAMPLER_STATE.json')
                    or value['pending_updates'] or value['unknown_effects']
                    or value.get('training_runs',{}).get(evaluation['candidate'],{}).get('last_native_step')!=4096):
                raise PermissionError('PV1_REPLICA_DEV_EXACT_TRAINED_SECOND_SEED_AND_CHARGED_4096')
        count=sum(len(remaining_panel(prior,session_spec(root,label))) for label in evaluation['panels'])
        if not count:raise PermissionError('PV1_NO_REMAINING_DEV_CONDITIONS_NO_NEW_MODEL_LOAD')
        input_('plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
        return
    if request['stage']=='B0_DEV':
        if evaluation['candidate']!='B0_CORR' or request.get('training_block'):
            raise PermissionError('PV1_B0_DEV_ONLY_NO_OPTIMIZATION')
        count=0
        for label in evaluation['panels']:count+=len(remaining_panel(prior,session_spec(root,label)))
        if not count:raise PermissionError('PV1_NO_REMAINING_BASELINE_CONDITIONS_NO_NEW_MODEL_LOAD')
        return
    block=input_('plans/PRIMARY_TRAIN_BLOCK.json')
    from scripts.pi0294.dec1.segmented_training import enabled as segment_enabled
    segment=segment_enabled(request)
    replica=request['stage']=='P4_REPLICA'
    if (block!=request['training_block'] or block['arm'] not in ARMS[stage['route']]
            or block['replica'] is not replica or block['stop_after_step'] not in ((1024,2048,4096) if segment else (1024,4096))):
        raise PermissionError('PV1_EXACT_PRIMARY_ARM_AND_STOPS')
    key=('REPLICA_' if replica else '')+block['arm']
    if replica:
        if replica_selection(root,input_,stage)!=block['arm']:
            raise PermissionError('PV1_REPLICA_ARM_MUST_MATCH_ACTUAL_DEV_WINNER')
        recipe=input_('plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][key]
        validate_primary(recipe,input_('plans/FROZEN_SOURCE_PAIR_RECIPE.json'))
        if run_key(recipe)!=key or recipe['seed']!=20261020:
            raise PermissionError('PV1_REPLICA_SEPARATE_IDENTITY_AND_FIXED_SECOND_SEED')
    population=input_(POPULATION_FILE)
    if sealed[POPULATION_FILE]!=POPULATION:raise PermissionError('PV1_ACTUAL_PREFROZEN_SCENE_POPULATION_SHA')
    expected=development(population['groups'],route=stage['route'],candidate='B0_CORR',
        new_bank_manifest_sha256=stage['new_bank_manifest_sha256'])
    conditions={c['condition_id'] for p in expected['panels'].values() for c in p['cards']}
    if not dec1_source_training:
        baseline=input_('plans/PV1_B0_DEV_ADMISSION.json')
        if (baseline['schema']!='PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1'
            or baseline['status']!='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE'
            or baseline['authority_release']!=AUTHORITY or baseline['population_plan_sha256']!=POPULATION
            or baseline['checkpoint_manifest_sha256']!=PARENT
            or baseline['new_bank_manifest_sha256']!=stage['new_bank_manifest_sha256']
            or set(baseline['condition_result_sha256'])!=conditions
            or set(baseline['independent_audit_sha256'])!=conditions
            or any(not re.fullmatch('[0-9a-f]{64}',v)
                for table in ('condition_result_sha256','independent_audit_sha256') for v in baseline[table].values())
            or baseline['real_native_faults'] or baseline['Teacher_takeover']
            or baseline['all_condition_evidence_audits_PASS'] is not True):
            raise PermissionError('PV1_COMPLETE_REAL_B0_BEFORE_NEW_TRAINING_NO_SUCCESS_RATE_GATE')
    input_('plans/INITIAL_CORR_PARAMETER_INVENTORY.json');input_('plans/PV1_FIXED_TRAIN_PROBES.json')
    relative=block.get('resume_checkpoint_relative')
    probe_only=block.get('probe_only',False)
    from scripts.pi0294.dec1.long_training import admitted as long_admitted
    long_training=long_admitted(root,m,request)
    if type(probe_only) is not bool or probe_only and (not relative or block['stop_after_step']!=1024):
        raise PermissionError('PV1_PROBE_ONLY_REQUIRES_EXACT_COMPLETE_1024_NO_UPDATES')
    if not relative:
        if block['stop_after_step']!=1024 and not long_training:raise PermissionError('PV1_1024_FIXED_PROBES_BEFORE_LATER_PRIMARY_UPDATES')
    else:
        p=Path(relative)
        if (p.is_absolute() or '..' in p.parts or p.parent!=Path('training')/key
                or (root/p).is_symlink() or not (root/p).resolve().is_relative_to(root.resolve())):
            raise PermissionError('PV1_ONLY_OWN_NATIVE_PRIMARY_FULL_RESUME')
        if allow_pending_segment_copy and segment and request['expected_native_start_step']==32:
            proof=input_('plans/RECOVERY32_FULL_STATE_IMPORT.private.json')
            source=input_('plans/RECOVERY32_SOURCE_BINDING.private.json')
            from scripts.pi0294.dec1.recovery32_intake import EXPECTED
            if ((root/p).exists() or proof.get('source')!=EXPECTED or proof.get('global_step')!=32
                    or source.get('checkpoint_complete_sha256')!=EXPECTED['complete_sha256']):
                raise PermissionError('DEC1_ONLY_COLD_PENDING_STATE32_COPY_NOT_MODEL_READINESS')
            return
        if allow_pending_segment_copy and segment and request['expected_native_start_step']==1024:
            from scripts.pi0294.dec1.b_recovery_intake import cold_binding
            cold_binding(root,request,input_)
            return
        if allow_pending_segment_copy and segment and request['expected_native_start_step']==2048:
            from scripts.pi0294.dec1.segment_intake import BINDING,PROOF,valid_binding
            binding=input_(BINDING);proof=input_(PROOF);valid_binding(binding,block['arm'])
            if ((root/p).exists() or proof.get('source')!=binding or proof.get('global_step')!=2048
                    or proof.get('actual_consumed_cursor')!=65536):
                raise PermissionError('DEC1_ONLY_COLD_PENDING2048_FULL_STATE_AND_ORIGINAL_PROBES')
            return
        complete=read(root/p/'CHECKPOINT_COMPLETE.json')
        step=complete['global_step']
        if (complete['full_resume'] is not True or not 0<step<4096 or p.name!='checkpoint-'+str(step)
                or block['stop_after_step']!=((2048 if step<1024 or request.get('B_heartbeat_recovery_release') is not None and step==1024 else 4096) if segment else (4096 if long_training else (1024 if step<1024 or probe_only else 4096)))):
            raise PermissionError('PV1_EXACT_COMPLETE_PARTIAL_PHASE_CONTINUATION')
        if probe_only:
            value=read(Ledger(root).path)
            if (step!=1024 or value['pending_updates'] or value['unknown_effects']
                    or value.get('training_runs',{}).get(key,{}).get('last_native_step')!=1024):
                raise PermissionError('PV1_PROBE_ONLY_EXACT_CHARGED_1024_NO_UNKNOWN_OPTIMIZER_EFFECTS')
            completed=root/'evidence'/(key+'-1024')/'TRAIN_PROBES/RESULT.json'
            if ((completed.exists() and read(completed).get('complete') is True)
                    or 'plans/COMPLETED_1024_TRAIN_PROBES.json' in sealed
                    and input_('plans/COMPLETED_1024_TRAIN_PROBES.json').get('complete') is True):
                raise PermissionError('PV1_NO_REMAINING_1024_PROBES_NO_NEW_MODEL_LOAD')
        elif step>=1024:
            result=input_('plans/COMPLETED_1024_TRAIN_PROBES.json')
            if (result['status']!='ACTUAL_FIXED_TRAIN_PROBES_NOT_TASK_SUCCESS' or result['complete'] is not True
                    or len(result['records'])!=12 or result['probe_plan_sha256']!=sealed['plans/PV1_FIXED_TRAIN_PROBES.json']
                    or result['new_physics'] or result['physical_success_gate']):
                raise PermissionError('PV1_ACTUAL_FIXED_1024_PROBES_REQUIRED_NOT_TASK_SUCCESS')


def parent_checkpoint(root):
    root=Path(root);inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    p=Path(inputs['CORR_parent_checkpoint']);complete=read(p/'CHECKPOINT_COMPLETE.json')
    if (sha(p/'CHECKPOINT_COMPLETE.json')!=PARENT or complete['global_step']!=2048
            or complete['full_resume'] is not True):
        raise PermissionError('PV1_ONLY_ORIGINAL_COMPLETE_CORR2048_PARENT')
    for name,digest in complete['files'].items():
        member=p/name
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or not member.resolve().is_relative_to(p.resolve()) or sha(member)!=digest):
            raise ValueError('PV1_NATIVE_FULL_PARENT_FILE_CHANGED:'+name)
    return p


def train_dataset(root):
    root=Path(root);b=read(root/'NEURAL_BINDINGS.private.json');inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    stage=read(root/'PV1_STAGE_BINDING.json');block=read(root/'plans/PRIMARY_TRAIN_BLOCK.json')
    return PV1Dataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,
        raw_bindings=read(b['raw_bindings']),correction_root=inputs['C1_dataset'],
        correction_bindings=read(inputs['C1_raw_bindings']),new_root=inputs['new_dataset'],
        new_raw_bindings=read(inputs['new_raw_bindings']),route=stage['route'],arm=block['arm'],
        seed=20261020 if block['replica'] else 20261019)


def record_candidate(ledger,candidate,identity):
    with ledger.transaction() as v:
        candidates=v.setdefault('development_candidates',{})
        proof=dict(checkpoint_manifest_sha256=identity['checkpoint_manifest_sha256'],global_step=identity['global_step'])
        if candidate in candidates:
            if candidates[candidate]!=proof:raise PermissionError('PV1_ONE_IMMUTABLE_CHECKPOINT_PER_DEV_CANDIDATE')
        else:
            if len(candidates)>=6:raise PermissionError('PV1_AT_MOST_SIX_DEV_CHECKPOINTS_INCLUDING_B0')
            candidates[candidate]=proof


def run(root,request_path):
    root,request_path=Path(root),Path(request_path);r=read(request_path)
    if r.get('stage') not in STAGES:
        raise PermissionError('PV1_THIS_DISPATCHER_ONLY_B0_PRIMARY_TERMINAL_DEV_AND_P5_PHASES')
    allocation=admit(root,request_path)
    ledger=(ledger_for_request(root,request_path,final=r['stage']=='P5_FINAL')
        if 'platform_evaluation_version' in r else
        Ledger(root,final=True) if r['stage']=='P5_FINAL' else Ledger(root))
    lock=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    directory=request_path.parent;status='FAILED_PRESERVED';phase='NATIVE_ENTRY';instance=None;records=[];error=None
    if (directory/'PBS_PROCESS_STARTED.json').exists():
        lock.close();raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_PROCESS_NO_BLIND_RESTART')
    write(directory/'PBS_PROCESS_STARTED.json',dict(job_id=allocation['job_id'],hostname=allocation['hostname'],
        pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],started_epoch=time.time(),models_loaded=0))
    def guard():
        ledger.guard()
        if r.get('scientific_admission_version')=='DEC1_SOURCE_TRAIN_STAGE_V1':
            limit=resource.getrlimit(resource.RLIMIT_CPU)[0]
            if limit!=resource.RLIM_INFINITY and time.process_time()>=limit-300:
                raise TimeoutError('DEC1_PHASE_CPU_BOUND_PRESERVE_KNOWN_STEP_AND_CHECKPOINT_RESERVE')
        if time.time()>=allocation['allocation_end_epoch']-2700:
            raise TimeoutError('PV1_ORIGINAL_NATIVE_45_MIN_CLOSEOUT_RESERVE')
        if r['stage'] in TRAIN_STAGES:
            a=read(root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
            if time.time()>=a['stop_new_training_epoch']:raise TimeoutError('PV1_ORIGINAL_TRAINING_CUTOFF')
        try:read_live_owner(root,root.name)
        except BaseException as exc:
            if getattr(exc,'diagnostics',None) is not None:
                p=directory/'OWNER_HEARTBEAT_LOSS.json'
                if not p.exists():write(p,dict(error=type(exc).__name__,diagnostics=exc.diagnostics,heartbeat_written=False))
            raise
    def unload():
        nonlocal instance
        if instance is not None:model.unload(instance);instance=None
    def load(checkpoint,label,*,trainable):
        nonlocal instance
        guard();require_current(root,allocation)
        identity=checkpoint_identity(checkpoint,label,r['source_sha'])
        ledger.charge('model_loads',1)
        intent=directory/'MODEL_LOAD_INTENTS'/(label+'.json');intent.parent.mkdir(exist_ok=True)
        if intent.exists():raise FileExistsError('PV1_PRESERVE_EACH_MODEL_LOAD_INTENT_NO_RETRY')
        write(intent,dict(job_id=allocation['job_id'],label=label,
            checkpoint_manifest_sha256=identity['checkpoint_manifest_sha256'],epoch=time.time()))
        instance=model.load(root,checkpoint,identity,trainable=trainable)
        return identity
    try:
        prepare_environment(r['numerical_mode'])
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
        remaining=int(160*3600-read(ledger.path)['CPU_s'])
        from scripts.pi0294.dec1.future_admission import bounded_process_cpu
        remaining=bounded_process_cpu(r,remaining)
        if 'platform_evaluation_version' in r:
            remaining=evaluation.process_cpu_limit(r,remaining)
        if remaining<=30:raise PermissionError('PV1_HOPPER_ACTUAL_CPU_BUDGET_EXHAUSTED')
        resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        stage_work(root,r)
        guard();phase='ACTUAL_NATIVE_PREMODEL_QUALIFICATION'
        proof=qualify_transport(root,directory,allocation,guard)
        write(directory/'PV1_PREMODEL_QUALIFICATION_READBACK.json',proof)
        phase='NEW_BANK_FULL_FIXED_REVISION_CONSUMPTION';records.append(consume(root,allocation,guard))
        parent=parent_checkpoint(root)
        write(directory/'FULL_CORR_PARENT_CHECK.json',dict(status='PASS_ALL_NATIVE_CORR_PARENT_FILES',
            checkpoint_manifest_sha256=PARENT,new_model_loads=0))
        guard()
        import torch
        require_native_device(r,allocation,torch)
        configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        if r['stage'] in ('B0_DEV','P4_DEV','P5_FINAL'):
            final=r['stage']=='P5_FINAL';plan=read(root/(FINAL_PLAN if final else 'plans/SOURCE_EVALUATION_PLAN.json'))
            candidate=plan['candidate_freeze']['candidate'] if final else plan['candidate'];phase=candidate+'-LOAD'
            checkpoint=(parent if r['stage']=='B0_DEV' else
                Path(read(root/'plans/P5_CANDIDATE_CHECKPOINT.private.json' if final else 'plans/EVALUATION_CHECKPOINT.private.json')['checkpoint']))
            identity=load(checkpoint,candidate,trainable=r['stage']=='B0_DEV')
            record_candidate(ledger,candidate,identity)
            inventory=parameter_inventory(instance.model)
            if r['stage']=='B0_DEV':write(root/'evidence/INITIAL_CORR_PARAMETER_INVENTORY.json',inventory)
            else:
                original=read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
                if set(original)!=set(inventory) or any(v['sha256']!=inventory[n]['sha256'] for n,v in original.items() if not v['requires_grad']):
                    raise PermissionError('PV1_ORIGINAL_FROZEN_VLM_CHANGED_BEFORE_DEVELOPMENT')
                write(directory/'FROZEN_VLM_PROOF.json',dict(status='PASS_ORIGINAL_FROZEN_VLM_NATIVE_PARAMETER_HASHES'))
                if not final:
                    from .future_admission import wait_native_release
                    wait_native_release(root,directory,allocation,guard)
            for label in plan['panels']:
                spec=(final_session_spec if final else session_spec)(root,label)
                if spec['candidate']!=candidate:raise PermissionError('PV1_ONE_SHARED_CHECKPOINT_ACROSS_BOTH_TASKS_AND_NOISE_BANKS')
                prior=read(root/'plans/PRIOR_EVALUATION_OUTCOMES.json')
                if not remaining_panel(prior,spec):
                    records.append(dict(panel=label,status='KNOWN_VALID_CONDITIONS_SKIPPED_NO_REEVALUATION'));continue
                phase=label
                fixture=root/'plans'/('PV1-PREMODEL-ALLOCATION-'+str(r['submit_count']).zfill(2))/'REGISTRY.json'
                peer=read(fixture);prior=read(root/'plans/PRIOR_EVALUATION_OUTCOMES.json');registry_path=root/'plans'/label/'REGISTRY.json'
                sealed_registry=[None]
                def panel_guard():
                    guard()
                    if registry_path.exists():
                        current=sha(registry_path)
                        if sealed_registry[0] is None:
                            validate_registry(read(registry_path),spec,identity,run_uuid=root.name,
                                job_id=allocation['job_id'],peer=peer,prior_rows=prior)
                            sealed_registry[0]=current
                        elif sealed_registry[0]!=current:raise PermissionError('PV1_MODEL_REGISTRY_CHANGED_NO_HOT_REBIND')
                serve(root,label,instance,allocation,ledger,panel_guard,max_queries=len(spec['cards'])*80,
                    ready_status='PV1_NATIVE_MODEL_READY',noise_namespace=spec['noise_namespace'],
                    terminal_status='PV1_NATIVE_PANEL_TERMINAL_OUTCOMES_AUDITED_SEPARATELY',
                    native_pre_model_qualification=require_current(root,allocation),visibility_refresh=refresh)
                records.append(dict(panel=label,status='NATIVE_PANEL_TERMINAL_NOT_TASK_SUCCESS'))
            status=('PV1_P5_LOCKBOX80_NATIVE_PANELS_TERMINAL_ACTUAL_OUTCOMES_AUDITED_SEPARATELY'
                if final else 'PV1_DEV_NATIVE_PANELS_TERMINAL_ACTUAL_OUTCOMES_AUDITED_SEPARATELY')
        else:
            block=r['training_block'];replica=r['stage']=='P4_REPLICA'
            key=('REPLICA_' if replica else '')+block['arm'];phase=key+'-TRAIN-INPUT'
            if block!=read(root/'plans/PRIMARY_TRAIN_BLOCK.json') or block['replica'] is not replica:
                raise PermissionError('PV1_EXACT_PREFROZEN_PRIMARY_TRAINING_BLOCK')
            relative=block.get('resume_checkpoint_relative');checkpoint=root/relative if relative else parent
            if relative and (Path(relative).is_absolute() or '..' in Path(relative).parts):
                raise PermissionError('PV1_ONLY_OWN_EXACT_NATIVE_PRIMARY_RESUME')
            probe_only=block.get('probe_only',False)
            identity=load(checkpoint,key+('-1024' if probe_only else '-TRAIN-INPUT'),trainable=not probe_only)
            phase=key+'-TRAIN-TO'+str(block['stop_after_step'])
            if probe_only:
                dataset=train_dataset(root);dataset.initialize()
                recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][key]
                validate_primary(recipe,read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json'))
                if run_key(recipe)!=key or checked_resume(root,key,checkpoint,recipe,dataset)!=1024:
                    raise PermissionError('PV1_PROBE_ONLY_EXACT_FULL_RECIPE_OPTIMIZER_RNG_SAMPLER_CHECKPOINT')
                phase=key+'-1024-FIXED-TRAIN-PROBES'
                records.append(probes(root,key+'-1024',instance,dataset,ledger,guard))
                status='PV1_EXACT_1024_PROBES_RECOVERED_ZERO_OPTIMIZER_UPDATES'
            else:
                from scripts.pi0294.dec1 import long_training
                start=read(checkpoint/'CHECKPOINT_COMPLETE.json')['global_step'] if relative else 0
                long_run=long_training.admitted(root,read(root/'RUN_MANIFEST.json'),r) and start<1024
                if long_run:
                    def train_phase(*,stop,phase,checkpoint):
                        return train(root,instance,allocation,guard,ledger,arm=block['arm'],request_id=r['request_id'],
                            resume_checkpoint=checkpoint or (root/relative if relative else None),
                            stop_after_step=stop,replica=False,long_phase=phase)
                    def fixed_probes():
                        dataset=train_dataset(root);dataset.initialize()
                        value=probes(root,key+'-1024',instance,dataset,ledger,guard)
                        long_training.probe_receipt(root,key)
                        return value
                    result,phases=long_training.execute(start_step=start,train=train_phase,probe=fixed_probes,
                        unload=unload,load=lambda p,trainable:load(p,key+'-LONG-'+('TRAIN' if trainable else 'PROBES'),trainable=trainable),
                        guard=guard,checkpoint=root/'training'/key/'checkpoint-1024')
                    records.extend({k:v[k] for k in ('status','global_step','actual_updates_this_attempt','actual_consumed_cursor') if k in v}
                        if 'global_step' in v else v for v in phases)
                else:
                    result=train(root,instance,allocation,guard,ledger,arm=block['arm'],request_id=r['request_id'],
                        resume_checkpoint=checkpoint if relative else None,stop_after_step=block['stop_after_step'],replica=replica)
                    records.append({k:result[k] for k in ('status','global_step','actual_updates_this_attempt','actual_consumed_cursor')})
                step=result['global_step'];unload()
                if step==1024 and not long_run:
                    phase=key+'-1024-FIXED-TRAIN-PROBES'
                    checkpoint=root/'training'/key/'checkpoint-1024'
                    load(checkpoint,key+'-1024',trainable=False)
                    dataset=train_dataset(root);dataset.initialize()
                    records.append(probes(root,key+'-1024',instance,dataset,ledger,guard))
                prefix='PV1_REPLICA' if replica else 'PV1_PRIMARY'
                status=prefix+'_4096_NATIVE_CHECKPOINT_TERMINAL_DEV_PENDING' if step==4096 else prefix+'_EXACT_PARTIAL_STATE_PRESERVED'
        return dict(status=status,records=records)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc);raise
    finally:
        try:unload()
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=time.process_time()-started+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
            try:
                ledger.cpu(cpu)
                write(directory/'PBS_PROCESS_FINAL.json',dict(status=status,phase=phase,records=records,error=error,
                    job_id=allocation['job_id'],CPU_s=cpu,completed_epoch=time.time(),selected='NONE',real_robot_deployment='NONE'))
            finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    args=parser.parse_args();print(json.dumps(run(args.root,args.request)),flush=True)
