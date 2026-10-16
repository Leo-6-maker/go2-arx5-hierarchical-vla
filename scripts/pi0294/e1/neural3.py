"""Last bounded allocation: only sealed unstarted/incomplete fixed M1/M2 work."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.native_train import parameter_inventory
from scripts.pi0294.numerics import prepare_environment,configure_torch
from .control_read import read
from .allocation import admit
from .scope import Ledger
from .neural import checkpoint_identity,serve


def stopped(root):
    p=root/'SOURCE_SAFETY_STOP_FROM_CORE.json'
    if not p.exists():return False
    value=read(p)
    if value['run_uuid']!=root.name or value['core_marker']['status']!='STOP_ALL_NEW_E1_SOURCE_PHYSICS':
        raise PermissionError('E1_EXACT_SOURCE_SAFETY_STOP_MESSAGE_REQUIRED')
    return True


def frozen_proof(root,instance):
    original=read(root/'training/M1/session-start/INITIAL_PARAMETERS.json');current=parameter_inventory(instance.model)
    if set(original)!=set(current) or any(current[n]['sha256']!=p['sha256'] for n,p in original.items() if not p['requires_grad']):
        raise ValueError('E1_REMAINDER_ORIGINAL_FROZEN_PARAMETER_HASH_MISMATCH')
    return dict(status='PASS_ORIGINAL_FROZEN_PARAMETER_HASHES',fixed_parameters=sum(not p['requires_grad'] for p in original.values()))


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);request=read(request_path);allocation=admit(root,request_path)
    if request['mode']!='E1_FIXED_REMAINDER_RECOVERY' or request['submit_count']!=3:raise PermissionError('E1_LAST_ALLOCATION_REMAINDER_ONLY')
    plan=read(root/'plans/REMAINDER_FIXED_SCOPE.json')
    if plan['run_uuid']!=root.name or [s['base_label'] for s in plan['models']]!=['M1-1024-CANONICAL','M2-BC','M2-CORR']:
        raise PermissionError('E1_FIXED_REMAINDER_ORDER_NO_NEW_MODELS')
    ledger=Ledger(root,'hopper');start_ledger=read(ledger.path)
    if (start_ledger['optimizer_updates']!=plan['optimizer_updates_at_binding']
            or start_ledger['pending_updates']!=plan['pending_updates_at_binding']
            or start_ledger['optimizer_updates']+start_ledger['pending_updates']>2048):
        raise PermissionError('E1_SEALED_ACTUAL_UPDATE_ACCOUNTING')
    parent=root/'training/M1/checkpoint-1024'
    if sha(parent/'CHECKPOINT_COMPLETE.json')!=plan['fixed_M1_parent_manifest_sha256'] or read(parent/'CHECKPOINT_COMPLETE.json')['global_step']!=1024:
        raise PermissionError('E1_REMAINDER_FIXED_M1_1024_PARENT')
    if start_ledger['pending_updates'] and any(s['training_action'] in ['NEW','RESUME'] for s in plan['models']):
        raise PermissionError('E1_UNKNOWN_OPTIMIZER_INTENT_NO_TRAINING_RESTART')
    prepare_environment(request['numerical_mode'])
    import torch
    if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0) or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise PermissionError('E1_OWN_SINGLE_H200_REQUIRED')
    configure_torch(torch,request['numerical_mode']);torch.set_num_threads(4);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_CPU,(int(115200-read(ledger.path)['CPU_s']),)*2)
    owner=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time();stage='INITIAL';status='FAILED_PRESERVED';records=[]
    def guard():
        ledger.guard()
        if time.time()>=allocation['allocation_end_epoch']-2700:raise TimeoutError('E1_LAST_ALLOCATION_45_MIN_CLOSEOUT')
        h=read(root/'OWNER_HEARTBEAT.json')
        if h['run_uuid']!=root.name or not 0<=time.time()-h['observed_epoch']<=300:raise TimeoutError('E1_OWNER_HEARTBEAT_LOST')
        if stopped(root) and stage!='M2-BC-TRAIN':raise PermissionError('E1_ACTUAL_SOURCE_STOP_NO_MORE_CORRECTION_OR_PHYSICS')
    try:
        from . import model
        from .dataset import SmallDataset
        from .train import run as train
        b=read(root/'NEURAL_BINDINGS.private.json');selection=read(root/'plans/SMALL8_BINDING.json')
        dataset=SmallDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=False,raw_bindings=read(b['raw_bindings']),selected_ids=[x['id'] for x in selection['episodes']]);dataset.initialize()
        for spec in plan['models']:
            base=spec['base_label'];label=spec['label'];branch=spec['branch'];action=spec['training_action'];stage=base+'-TRAIN'
            if label!=base+'-REMAINDER-03' or branch!={'M1-1024-CANONICAL':'M1','M2-BC':'M2-BC','M2-CORR':'M2-CORR'}[base]:raise PermissionError('E1_NO_UNREGISTERED_REMAINDER_LABEL')
            if action=='BLOCKED':records.append(dict(label=label,status='BLOCKED_NATIVE_STATE_PRESERVED',reason=spec['blocked_reason']));continue
            if (plan['source_physics_stopped'] or stopped(root)) and branch=='M2-CORR':
                records.append(dict(label=label,status='NOT_RUN_SOURCE_SAFETY_STOP'));continue
            if action in ['NEW','RESUME']:
                if branch=='M1':raise PermissionError('E1_M1_NEVER_RETRAINED_IN_REMAINDER')
                guard();checkpoint=root/spec['training_checkpoint_relative']
                if action=='NEW' and checkpoint!=parent:raise PermissionError('E1_BOTH_NEW_M2_BRANCHES_ONLY_FIXED_M1_PARENT')
                if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=spec['training_checkpoint_manifest_sha256']:raise PermissionError('E1_EXACT_SEALED_TRAINING_STATE')
                identity=checkpoint_identity(checkpoint,'M1-1024' if action=='NEW' else branch+'-PARTIAL',request['source_sha'])
                with ledger.transaction() as l:l['model_loads']+=1
                instance=model.load(root,checkpoint,identity,trainable=True);proof=frozen_proof(root,instance)
                result=train(root,instance,allocation,guard,branch=branch,resume_checkpoint=checkpoint if action=='RESUME' else None);model.unload(instance)
                if result['global_step']!=512:
                    status='PARTIAL_NATIVE_REMAINDER_STATE_PRESERVED';return dict(status=status,records=records)
            elif action!='COMPLETE':raise PermissionError('E1_KNOWN_NATIVE_TRAINING_ACTION_REQUIRED')
            checkpoint=root/'training'/branch/('checkpoint-1024' if branch=='M1' else 'checkpoint-512')
            if action=='COMPLETE' and sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=spec['fixed_checkpoint_manifest_sha256']:
                raise PermissionError('E1_ALREADY_COMPLETED_CHECKPOINT_MAY_NOT_BE_REPLACED')
            if (plan['source_physics_stopped'] or stopped(root)) or not spec['remaining_conditions']:
                records.append(dict(label=label,status='NO_PHYSICAL_REPLAY_SOURCE_STOP_OR_ALL_CONDITIONS_ALREADY_VALID'));continue
            guard();stage=label;identity=checkpoint_identity(checkpoint,label,request['source_sha'])
            expected=1024 if branch=='M1' else 512
            if identity['global_step']!=expected:raise PermissionError('E1_NO_PARTIAL_CHECKPOINT_POLICY_EVALUATION')
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False);proof=frozen_proof(root,instance)
            folder=root/'evidence'/label;folder.mkdir(parents=True,exist_ok=True)
            prior=root/'evidence'/base/'probes'
            if prior.exists():
                expert=read(prior/'ON_EXPERT_RESULTS.json') if (prior/'ON_EXPERT_RESULTS.json').exists() else {}
                student=read(prior/'ON_STUDENT_RESULTS.private.json') if (prior/'ON_STUDENT_RESULTS.private.json').exists() else {}
                write(folder/'PROBE_REUSE_BINDING.json',dict(status='REUSED_COMPLETE_FIXED_PROBES' if expert.get('complete') and len(student.get('records',[]))==12 else 'PARTIAL_OR_UNKNOWN_OLD_PROBES_PRESERVED_NO_RESAMPLING',
                    original_label=base,original_expert_records=len(expert.get('records',[])),original_student_records=len(student.get('records',[])),new_queries=0))
            else:model.probes(root,label,instance,dataset,ledger,guard)
            write(folder/'FROZEN_PARAMETER_PROOF.json',proof)
            serve(root,label,instance,allocation,ledger,guard);model.unload(instance)
            records.append(dict(label=label,status='SEALED_REMAINDER_STAGE_TERMINAL'))
        status='FINITE_FIXED_REMAINDER_STAGES_TERMINAL';return dict(status=status,records=records)
    finally:
        cpu=time.process_time()-started;ledger.cpu(cpu)
        write(request_path.parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,records=records,CPU_s=cpu,job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),selected='NONE',deployment='NONE'));owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.request)),flush=True)
