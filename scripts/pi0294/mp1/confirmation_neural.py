"""Only the one frozen source candidate; no trainer, probes or second selection."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.neural import checkpoint_identity,serve
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.numerics import prepare_environment,configure_torch
from . import confirmation_config as config
from .allocation import admit
from .native_transport_qualification import run as qualify
from .owner_heartbeat import read_live_owner
from .plans import remaining_panel
from .science import Ledger


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);request=read(request_path)
    if request['mode']!=config.MODE or not config.is_confirmation(read(root/'MP1_STAGE_BINDING.json')):
        raise PermissionError('ONE_FROZEN_SOURCE_CONFIRMATION_ONLY')
    allocation=admit(root,request_path);ledger=Ledger(root,'hopper');plan=config.validate_plan(root)
    prior=config.prior_closure(root,request,read(ledger.path));checkpoint=config.candidate_checkpoint(root)
    prepare_environment(request['numerical_mode'])
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
        raise PermissionError('CONFIRMATION_ONLY_IN_OWN_RUNNING_SINGLE_H200_PBS')
    configure_torch(torch,request['numerical_mode']);torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    remaining=int(ledger.cpu_caps['hopper']-read(ledger.path)['CPU_s'])
    if remaining<=0:raise PermissionError('FINITE_CONFIRMATION_HOPPER_CPU_REMAINING')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));owner=OwnerLock(root/'NEURAL_OWNER.lock')
    started=time.process_time();instance=None;status='FAILED_PRESERVED';records=[];stage='PREMODEL_QUALIFICATION'
    active_end=allocation['allocation_end_epoch']-2700
    def guard():
        ledger.guard()
        if time.time()>=active_end:raise TimeoutError('ORIGINAL_2700S_ALLOCATION_CLOSEOUT')
        for name in ['RELAY_ENGINEERING_STOP.json','SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json']:
            if (root/name).exists():raise PermissionError('CONFIRMATION_STOP_PRESERVED_NO_NEXT_CASE_OR_MODEL')
        read_live_owner(root,root.name)
        l=read(ledger.path)
        if l['optimizer_updates'] or l['pending_updates']:
            raise PermissionError('CONFIRMATION_HAS_ZERO_TRAINING_AUTHORITY')
    try:
        qualification=qualify(root,request_path.parent,allocation,guard)
        write(request_path.parent/'N1_PREMODEL_QUALIFICATION_READBACK.json',qualification)
        for label,spec in plan['panels'].items():
            remaining_cases=remaining_panel(prior['rows'],spec)
            if not remaining_cases:
                records.append(dict(panel=label,status='PRIOR_TERMINAL_PANEL_SKIPPED_BEFORE_MODEL_LOAD',new_queries=0))
                continue
            if time.time()+900>=active_end:
                records.append(dict(panel=label,status='NOT_RUN_KEEP_ORIGINAL_ALLOCATION_CLOSEOUT',new_queries=0));break
            stage=label+'-ALLOCATION-'+str(request['submit_count']).zfill(2);guard()
            if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=spec['checkpoint_manifest_sha256']:
                raise PermissionError('SINGLE_CONFIRMATION_CANDIDATE_CHANGED')
            identity=checkpoint_identity(checkpoint,stage,request['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False)
            if any(p.requires_grad for p in instance.model.parameters()):raise PermissionError('CONFIRMATION_PARAMETERS_ALL_FROZEN')
            folder=root/'evidence'/stage;folder.mkdir(mode=0o700)
            write(folder/'CANDIDATE_BINDING.json',dict(candidate=config.plan.CANDIDATE,
                checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],global_step=2048,
                all_parameters_frozen=True,noise_namespace=spec['noise_namespace'],new_training_updates=0,new_fixed_probes=0))
            serve(root,stage,instance,allocation,ledger,guard,max_queries=80*len(remaining_cases),
                ready_status='MP1_NATIVE_MODEL_READY',terminal_status='MP1_NATIVE_SESSION_TERMINAL_NOT_TASK_SUCCESS',
                noise_namespace=spec['noise_namespace'])
            model.unload(instance);instance=None
            records.append(dict(panel=label,session=stage,status='NATIVE_SESSION_TERMINAL_OUTCOMES_INDEPENDENT_AUDIT_REQUIRED'))
        status='SINGLE_SOURCE_CONFIRMATION_ALLOCATION_TERMINAL_NOT_AGGREGATE_SUCCESS'
        return dict(status=status,records=records)
    finally:
        try:
            if instance is not None:model.unload(instance)
        finally:
            cpu=time.process_time()-started;ledger.cpu(cpu)
            write(request_path.parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,records=records,CPU_s=cpu,
                job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),optimizer_updates=0,selected='NONE',deployment='NONE'))
            owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    args=parser.parse_args();print(json.dumps(run(args.root,args.request)),flush=True)
