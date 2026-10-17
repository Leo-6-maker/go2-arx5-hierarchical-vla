"""Measured official source learning and fixed, teacher-off development panels."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.neural import checkpoint_identity,serve
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_io import read_available
from scripts.pi0294.native_train import parameter_inventory
from scripts.pi0294.numerics import prepare_environment,configure_torch
from .allocation import admit
from .dataset import SourceDataset
from .science import Ledger
from .train import run as train
from .cold_boundary import Guard as ColdGuard
from .owner_heartbeat import read_live_owner


def checkpoint_state(root,branch):
    entries=[]
    for p in (Path(root)/'training'/branch).glob('checkpoint-*/CHECKPOINT_COMPLETE.json'):
        v=read(p)
        if p.parent.name!='checkpoint-'+str(v['global_step']) or not v['full_resume'] or not 0<v['global_step']<=2048:
            raise PermissionError('MP1_COMPLETE_REGISTERED_NATIVE_SOURCE_STATE_ONLY')
        entries.append((v['global_step'],p.parent))
    return max(entries,key=lambda x:x[0]) if entries else (0,None)


def frozen_proof(root,instance):
    original=read(root/'plans/PARENT_PARAMETER_INVENTORY.json');current=parameter_inventory(instance.model)
    if set(original)!=set(current) or any(p['sha256']!=current[n]['sha256'] for n,p in original.items() if not p['requires_grad']):
        raise PermissionError('MP1_ORIGINAL_FROZEN_VLM_PARAMETER_HASH_MISMATCH')
    return dict(status='PASS_FIXED_PARENT_FROZEN_PARAMETERS',fixed_parameters=sum(not p['requires_grad'] for p in original.values()))


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);r=read(request_path);allocation=admit(root,request_path);ledger=Ledger(root,'hopper')
    prepare_environment(r['numerical_mode'])
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
        raise PermissionError('MP1_OWN_NATIVE_SINGLE_VISIBLE_H200_ONLY')
    configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    remaining_CPU=int(172800-read(ledger.path)['CPU_s'])
    if remaining_CPU<=0:raise PermissionError('MP1_SOURCE_STUDY_CPU_EXHAUSTED')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining_CPU,remaining_CPU));lock=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time()
    stage='INITIAL';status='FAILED_PRESERVED';records=[];instance=None;heartbeats=[];last_heartbeat_log=0
    cold_guard=ColdGuard(root,r['submit_count']);cold=cold_guard.binding
    def source_stopped():
        return (root/'SOURCE_SAFETY_STOP_FROM_CORE.json').exists() or (root/'SOURCE_PHYSICS_STOP.json').exists()
    def guard():
        nonlocal last_heartbeat_log
        ledger.guard()
        if cold_guard.blocked():raise PermissionError('RELAY_ENGINEERING_STOP_NO_NEXT_MODEL_NOT_NATIVE_SAFETY_FAIL')
        if time.time()>=allocation['allocation_end_epoch']-2700:raise TimeoutError('MP1_ORIGINAL_45_MIN_ALLOCATION_CLOSEOUT')
        if cold:
            try:proof=read_live_owner(root,root.name)
            except BaseException as exc:
                heartbeats.append(dict(epoch=time.time(),error=type(exc).__name__,diagnostics=getattr(exc,'diagnostics',None)));raise
            if proof['server_revalidation'] or time.time()-last_heartbeat_log>=30:
                heartbeats.append(dict(epoch=time.time(),proof=proof));last_heartbeat_log=time.time()
        else:
            h=read(root/'OWNER_HEARTBEAT.json')
            if h['run_uuid']!=root.name or not 0<=time.time()-h['observed_epoch']<=300:raise TimeoutError('MP1_SINGLE_OWNER_HEARTBEAT_LOST')
        if source_stopped() and not stage.startswith('S-BC-TRAIN-'):
            raise PermissionError('MP1_SOURCE_DOMAIN_SAFETY_STOP_PRESERVED')
    def load(checkpoint,label,trainable):
        nonlocal instance
        guard();identity=checkpoint_identity(checkpoint,label,r['source_sha'])
        with ledger.transaction() as l:l['model_loads']=l.get('model_loads',0)+1
        instance=model.load(root,checkpoint,identity,trainable=trainable);proof=frozen_proof(root,instance)
        return proof
    def unload():
        nonlocal instance
        if instance is not None:model.unload(instance);instance=None
    def evaluate(branch,step,checkpoint):
        nonlocal stage
        labels=[branch+'-'+str(step)+'-TRAIN4']+([branch+'-2048-DEV8'] if step==2048 else [])
        for base in labels:
            if source_stopped():
                records.append(dict(panel=base,status='NOT_RUN_SOURCE_DOMAIN_SAFETY_STOP_PRESERVED'));continue
            guard();stage=base;session=base+'-ALLOCATION-'+str(r['submit_count']).zfill(2)
            folder=root/'evidence'/session;folder.mkdir(parents=True,exist_ok=False)
            proof=load(checkpoint,session,False);write(folder/'FROZEN_PARAMETER_PROOF.json',proof)
            probe=root/'evidence'/base/'probes'
            if base.endswith('TRAIN4'):
                if probe.exists():
                    old=read(probe/'ON_EXPERT_RESULTS.json') if (probe/'ON_EXPERT_RESULTS.json').exists() else {}
                    write(folder/'PROBE_REUSE_BINDING.json',dict(status='PRIOR_FIXED_PROBES_PRESERVED_NO_RESAMPLING',
                        prior_expert_complete=old.get('complete',False),prior_expert_records=len(old.get('records',[])),new_queries=0))
                else:
                    b=read(root/'NEURAL_BINDINGS.private.json');d=SourceDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=False,raw_bindings=read(b['raw_bindings']));d.initialize()
                    model.probes(root,base,instance,d,ledger,guard)
            spec=read(root/'plans/SOURCE_EVALUATION_PLAN.json')['panels'][base]
            serve(root,session,instance,allocation,ledger,guard,max_queries=80*len(spec['cards']),ready_status='MP1_NATIVE_MODEL_READY',terminal_status='MP1_NATIVE_SESSION_TERMINAL_NOT_TASK_SUCCESS')
            unload();records.append(dict(panel=base,session=session,status='NATIVE_SESSION_TERMINAL_ACTUAL_OUTCOMES_SEPARATELY_AUDITED'))
    try:
        if cold:
            qualification=read_live_owner(root,root.name,force_revalidate=True)
            write(request_path.parent/'OWNER_HEARTBEAT_SERVER_QUALIFICATION.json',dict(status='PASS_FORCED_NATIVE_COMPUTE_OWNER_READ_BEFORE_MODEL_LOAD',job_id=os.environ['PBS_JOBID'],epoch=time.time(),proof=qualification,models_loaded=0,new_queries=0,new_physics=0))
        parent=read(root/'plans/SOURCE_PARENT_CHECKPOINT.json');parent_path=Path(parent['checkpoint'])
        if sha(parent_path/'CHECKPOINT_COMPLETE.json')!=parent['checkpoint_manifest_sha256']:raise PermissionError('MP1_AUDITED_M1_1024_PARENT_CHANGED')
        for branch in ['S-BC','S-CORR']:
            if branch=='S-CORR' and source_stopped():
                records.append(dict(branch=branch,status='NOT_RUN_CORRECTION_SOURCE_SAFETY_STOP'));continue
            step,checkpoint=checkpoint_state(root,branch)
            if step<512:
                stage=branch+'-TRAIN-TO512';load(checkpoint or parent_path,branch+'-TRAIN-INPUT',True)
                result=train(root,instance,allocation,guard,ledger,branch=branch,resume_checkpoint=checkpoint,stop_after_step=512);unload()
                step,checkpoint=checkpoint_state(root,branch)
                if step<512:status='PARTIAL_NATIVE_SOURCE_STATE_PRESERVED';return dict(status=status,records=records)
            fixed512=root/'training'/branch/'checkpoint-512'
            if not (fixed512/'CHECKPOINT_COMPLETE.json').is_file():raise PermissionError('MP1_FIXED_512_LEARNING_CURVE_CHECKPOINT_REQUIRED')
            evaluate(branch,512,fixed512)
            if step<2048:
                stage=branch+'-TRAIN-TO2048';load(checkpoint,branch+'-TRAIN-INPUT',True)
                result=train(root,instance,allocation,guard,ledger,branch=branch,resume_checkpoint=checkpoint,stop_after_step=2048);unload()
                step,checkpoint=checkpoint_state(root,branch)
                if step<2048:status='PARTIAL_NATIVE_SOURCE_STATE_PRESERVED';return dict(status=status,records=records)
            evaluate(branch,2048,checkpoint)
        status='FIXED_SOURCE_PAIR_NATIVE_PHASES_TERMINAL';return dict(status=status,records=records)
    finally:
        try:unload()
        finally:
            cpu=time.process_time()-started
            try:
                ledger.cpu(cpu)
                if cold:write(request_path.parent/'OWNER_HEARTBEAT_DIAGNOSTICS.json',dict(watchdog_s=300,heartbeat_written=False,rows=heartbeats))
                write(request_path.parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,records=records,CPU_s=cpu,job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),selected='NONE',real_robot_deployment='NONE'))
            finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.request)),flush=True)
