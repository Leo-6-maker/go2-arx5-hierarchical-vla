"""Evaluate the two completed E1 models; no optimizer or new probe generation."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_io import read_available
from scripts.pi0294.numerics import prepare_environment,configure_torch
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.neural import checkpoint_identity,serve
from .allocation import admit
from .science import Ledger
from .recovery_config import validate_checkpoint_bindings,is_recovery,MODE,CHECKPOINTS


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);request=read(request_path)
    if request['mode']!=MODE or not is_recovery(read(root/'MP1_STAGE_BINDING.json')):raise PermissionError('FIXED_EXISTING_MODEL_EVALUATION_ONLY')
    allocation=admit(root,request_path);ledger=Ledger(root,'hopper');bindings=read(root/'plans/RECOVERY_CHECKPOINTS.private.json')
    validate_checkpoint_bindings(bindings);prepare_environment(request['numerical_mode'])
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
        raise PermissionError('RECOVERY_IN_OWN_SINGLE_H200_ALLOCATION_ONLY')
    configure_torch(torch,request['numerical_mode']);torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(28800,28800))
    owner=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time();instance=None;status='FAILED_PRESERVED';records=[];stage='INITIAL'
    def guard():
        ledger.guard()
        if read_available(root/'RELAY_ENGINEERING_STOP.json') is not None:raise PermissionError('RELAY_ENGINEERING_STOP_NO_NEXT_MODEL_NOT_NATIVE_SAFETY_FAIL')
        if time.time()>=allocation['allocation_end_epoch']-2700:raise TimeoutError('ORIGINAL_45_MIN_ALLOCATION_CLOSEOUT')
        heartbeat=read(root/'OWNER_HEARTBEAT.json')
        if heartbeat['run_uuid']!=root.name or not 0<=time.time()-heartbeat['observed_epoch']<=300:raise TimeoutError('SINGLE_MP1_OWNER_HEARTBEAT_LOST')
        if (root/'SOURCE_SAFETY_STOP_FROM_CORE.json').exists() or (root/'SOURCE_PHYSICS_STOP.json').exists():raise PermissionError('SOURCE_SAFETY_STOP_PRESERVED')
        l=read(ledger.path)
        if l['optimizer_updates'] or l['pending_updates']:raise PermissionError('RECOVERY_HAS_NO_TRAINING_AUTHORITY')
    try:
        plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json')
        for branch in CHECKPOINTS:
            guard();base='E1REC-'+branch+'-512-TRAIN4';stage=base+'-ALLOCATION-'+str(request['submit_count']).zfill(2)
            checkpoint=Path(bindings[branch]['checkpoint'])
            if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=CHECKPOINTS[branch]:raise PermissionError('FIXED_E1_CHECKPOINT_CHANGED')
            identity=checkpoint_identity(checkpoint,stage,request['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False)
            if any(p.requires_grad for p in instance.model.parameters()):raise PermissionError('INFERENCE_ONLY_ALL_PARAMETERS_FROZEN')
            folder=root/'evidence'/stage;folder.mkdir(parents=True,exist_ok=False)
            write(folder/'EXISTING_WEIGHT_BINDING.json',dict(branch=branch,checkpoint_manifest_sha256=CHECKPOINTS[branch],
                global_step=512,optimizer_updates=0,new_probes=0,old_E1_not_reopened=True,canonical_precision=True,all_parameters_frozen=True))
            serve(root,stage,instance,allocation,ledger,guard,max_queries=80*len(plan['panels'][base]['cards']),
                ready_status='MP1_NATIVE_MODEL_READY',terminal_status='MP1_NATIVE_SESSION_TERMINAL_NOT_TASK_SUCCESS')
            model.unload(instance);instance=None
            records.append(dict(panel=base,session=stage,status='NATIVE_SESSION_TERMINAL_OUTCOMES_AUDITED_SEPARATELY'))
        status='EXISTING_E1_M2_NATIVE_EVALUATION_PHASES_TERMINAL';return dict(status=status,records=records)
    finally:
        try:
            if instance is not None:model.unload(instance)
        finally:
            cpu=time.process_time()-started;ledger.cpu(cpu)
            write(request_path.parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,records=records,CPU_s=cpu,
                job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),optimizer_updates=0,selected='NONE',deployment='NONE'))
            owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.request)),flush=True)
