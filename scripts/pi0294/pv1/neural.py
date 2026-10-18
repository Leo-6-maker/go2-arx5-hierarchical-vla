"""PV1 baseline, primary blocks and terminal DEV through mature primitives."""
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
from .evaluation_plan import session_spec,validate_registry,remaining_panel
from .evaluation_plan import development,POPULATION_FILE
from .decision import AUTHORITY
from .sampler import ARMS
from .native_admission import admit
from .native_transport_qualification import run as qualify_transport
from .neural_ledger import Ledger
from .pre_model_qualification import require_current
from .probes import run as probes
from .recipe import PARENT
from .training import run as train


def stage_work(root,request):
    if request['stage'] not in ('B0_DEV','P2_PRIMARY','P4_DEV'):
        raise PermissionError('PV1_ONLY_IMPLEMENTED_PREDECLARED_PHASES')
    root=Path(root);m=read(root/'RUN_MANIFEST.json');sealed=m['execution']['sealed_inputs']
    def input_(name):
        p=root/name
        if p.is_symlink() or p.stat().st_mode&0o222 or sealed.get(name)!=sha(p):
            raise PermissionError('PV1_PRESEALED_NATIVE_PHASE_INPUT:'+name)
        return read(p)
    stage=input_('PV1_STAGE_BINDING.json');evaluation=input_('plans/SOURCE_EVALUATION_PLAN.json')
    prior=input_('plans/PRIOR_EVALUATION_OUTCOMES.json')
    if request['stage']=='P4_DEV':
        checkpoint=input_('plans/EVALUATION_CHECKPOINT.private.json')
        if (evaluation['candidate'] not in ARMS[stage['route']]
                or request['evaluation_candidate']!=evaluation['candidate'] or request.get('training_block')
                or checkpoint['candidate']!=evaluation['candidate'] or checkpoint['global_step']!=4096):
            raise PermissionError('PV1_ONLY_FROZEN_TERMINAL_PRIMARY_CHECKPOINT_DEVELOPMENT')
        p=Path(checkpoint['checkpoint']);complete=read(p/'CHECKPOINT_COMPLETE.json')
        if (p.stat().st_uid!=os.getuid() or sha(p/'CHECKPOINT_COMPLETE.json')!=checkpoint['checkpoint_manifest_sha256']
                or complete['global_step']!=4096 or complete['full_resume'] is not True):
            raise PermissionError('PV1_ACTUAL_COMPLETE_NATIVE_4096_CHECKPOINT_REQUIRED')
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
    if (block!=request['training_block'] or block['arm'] not in ARMS[stage['route']]
            or block['replica'] is not False or block['stop_after_step'] not in (1024,4096)):
        raise PermissionError('PV1_EXACT_PRIMARY_ARM_AND_STOPS')
    baseline=input_('plans/PV1_B0_DEV_ADMISSION.json');population=input_(POPULATION_FILE)
    if sealed[POPULATION_FILE]!=POPULATION:raise PermissionError('PV1_ACTUAL_PREFROZEN_SCENE_POPULATION_SHA')
    expected=development(population['groups'],route=stage['route'],candidate='B0_CORR',
        new_bank_manifest_sha256=stage['new_bank_manifest_sha256'])
    conditions={c['condition_id'] for p in expected['panels'].values() for c in p['cards']}
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
    if not relative:
        if block['stop_after_step']!=1024:raise PermissionError('PV1_1024_FIXED_PROBES_BEFORE_LATER_PRIMARY_UPDATES')
    else:
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or p.parent!=Path('training')/block['arm']:
            raise PermissionError('PV1_ONLY_OWN_NATIVE_PRIMARY_FULL_RESUME')
        complete=read(root/p/'CHECKPOINT_COMPLETE.json')
        step=complete['global_step']
        if (complete['full_resume'] is not True or not 0<step<4096 or p.name!='checkpoint-'+str(step)
                or block['stop_after_step']!=(1024 if step<1024 else 4096)):
            raise PermissionError('PV1_EXACT_COMPLETE_PARTIAL_PHASE_CONTINUATION')
        if step>=1024:
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
    if r.get('stage') not in ('B0_DEV','P2_PRIMARY','P4_DEV'):
        raise PermissionError('PV1_THIS_DISPATCHER_ONLY_B0_PRIMARY_AND_TERMINAL_DEV_PHASES')
    allocation=admit(root,request_path);ledger=Ledger(root)
    lock=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    directory=request_path.parent;status='FAILED_PRESERVED';phase='NATIVE_ENTRY';instance=None;records=[];error=None
    if (directory/'PBS_PROCESS_STARTED.json').exists():
        lock.close();raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_PROCESS_NO_BLIND_RESTART')
    write(directory/'PBS_PROCESS_STARTED.json',dict(job_id=allocation['job_id'],hostname=allocation['hostname'],
        pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],started_epoch=time.time(),models_loaded=0))
    def guard():
        ledger.guard()
        if time.time()>=allocation['allocation_end_epoch']-2700:
            raise TimeoutError('PV1_ORIGINAL_NATIVE_45_MIN_CLOSEOUT_RESERVE')
        if r['stage']=='P2_PRIMARY':
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
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
            raise PermissionError('PV1_ONE_VISIBLE_NATIVE_H200_ONLY')
        configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        if r['stage'] in ('B0_DEV','P4_DEV'):
            plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json')
            candidate=plan['candidate'];phase=candidate+'-LOAD'
            checkpoint=parent if r['stage']=='B0_DEV' else Path(read(root/'plans/EVALUATION_CHECKPOINT.private.json')['checkpoint'])
            identity=load(checkpoint,candidate,trainable=r['stage']=='B0_DEV')
            record_candidate(ledger,candidate,identity)
            inventory=parameter_inventory(instance.model)
            if r['stage']=='B0_DEV':write(root/'evidence/INITIAL_CORR_PARAMETER_INVENTORY.json',inventory)
            else:
                original=read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
                if set(original)!=set(inventory) or any(v['sha256']!=inventory[n]['sha256'] for n,v in original.items() if not v['requires_grad']):
                    raise PermissionError('PV1_ORIGINAL_FROZEN_VLM_CHANGED_BEFORE_DEVELOPMENT')
                write(directory/'FROZEN_VLM_PROOF.json',dict(status='PASS_ORIGINAL_FROZEN_VLM_NATIVE_PARAMETER_HASHES'))
            for label in plan['panels']:
                spec=session_spec(root,label)
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
                serve(root,label,instance,allocation,ledger,panel_guard,max_queries=24*80,
                    ready_status='PV1_NATIVE_MODEL_READY',noise_namespace=spec['noise_namespace'],
                    terminal_status='PV1_NATIVE_PANEL_TERMINAL_OUTCOMES_AUDITED_SEPARATELY')
                records.append(dict(panel=label,status='NATIVE_PANEL_TERMINAL_NOT_TASK_SUCCESS'))
            status='PV1_DEV_NATIVE_PANELS_TERMINAL_ACTUAL_OUTCOMES_AUDITED_SEPARATELY'
        else:
            block=r['training_block'];phase='PRIMARY_TRAIN_INPUT'
            if block!=read(root/'plans/PRIMARY_TRAIN_BLOCK.json') or block['replica'] is not False:
                raise PermissionError('PV1_EXACT_PREFROZEN_PRIMARY_TRAINING_BLOCK')
            relative=block.get('resume_checkpoint_relative');checkpoint=root/relative if relative else parent
            if relative and (Path(relative).is_absolute() or '..' in Path(relative).parts):
                raise PermissionError('PV1_ONLY_OWN_EXACT_NATIVE_PRIMARY_RESUME')
            identity=load(checkpoint,block['arm']+'-TRAIN-INPUT',trainable=True)
            phase=block['arm']+'-TRAIN-TO'+str(block['stop_after_step'])
            result=train(root,instance,allocation,guard,ledger,arm=block['arm'],request_id=r['request_id'],
                resume_checkpoint=checkpoint if relative else None,stop_after_step=block['stop_after_step'])
            records.append({k:result[k] for k in ('status','global_step','actual_updates_this_attempt','actual_consumed_cursor')})
            step=result['global_step'];unload()
            if step==1024:
                phase=block['arm']+'-1024-FIXED-TRAIN-PROBES'
                checkpoint=root/'training'/block['arm']/'checkpoint-1024'
                load(checkpoint,block['arm']+'-1024',trainable=False)
                dataset=train_dataset(root);dataset.initialize()
                records.append(probes(root,block['arm']+'-1024',instance,dataset,ledger,guard))
            status='PV1_PRIMARY_4096_NATIVE_CHECKPOINT_TERMINAL_DEV_PENDING' if step==4096 else 'PV1_PRIMARY_EXACT_PARTIAL_STATE_PRESERVED'
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
