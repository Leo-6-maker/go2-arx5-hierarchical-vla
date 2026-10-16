"""Qualified T1 intervention, then fixed same-parent BC/correction comparison."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.numerics import prepare_environment,configure_torch
from .allocation import admit
from .scope import Ledger
from .neural import checkpoint_identity,serve


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);r=read(request_path);allocation=admit(root,request_path)
    if r['mode']!='E1_T1_COLLECTION_MATCHED_M2':raise PermissionError('E1_FIXED_SECOND_WORKLOAD_ONLY')
    ledger=Ledger(root,'hopper')
    if read(ledger.path)['optimizer_updates']!=1024 or read(ledger.path)['pending_updates']:
        raise PermissionError('ACTUAL_COMPLETE_M1_PARENT_REQUIRED')
    prepare_environment(r['numerical_mode'])
    import torch
    if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0) or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise PermissionError('E1_OWN_SINGLE_H200_ONLY')
    configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_CPU,(int(115200-read(ledger.path)['CPU_s']),)*2)
    owner=OwnerLock(root/'NEURAL_OWNER.lock');start=time.process_time();stage='INITIAL';status='FAILED_PRESERVED'
    def guard():
        ledger.guard()
        if time.time()>=allocation['allocation_end_epoch']-2700:raise TimeoutError('E1_PRESERVE_45_MIN_CLOSEOUT')
        heartbeat=read(root/'OWNER_HEARTBEAT.json')
        if heartbeat['run_uuid']!=root.name or not 0<=time.time()-heartbeat['observed_epoch']<=300:raise TimeoutError('E1_OWNER_HEARTBEAT_LOST')
    try:
        from . import model
        from .dataset import SmallDataset
        from .train import run as train
        parent=root/'training/M1/checkpoint-1024'
        if read(parent/'CHECKPOINT_COMPLETE.json')['global_step']!=1024:raise PermissionError('E1_FIXED_M1_1024_PARENT_ONLY')
        b=read(root/'NEURAL_BINDINGS.private.json');selection=read(root/'plans/SMALL8_BINDING.json')
        dataset=SmallDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=False,raw_bindings=read(b['raw_bindings']),selected_ids=[x['id'] for x in selection['episodes']]);dataset.initialize()
        repair=read(root/'plans/INFERENCE_PRECISION_REPAIR.json')
        if repair['status']!='FIXED_INPUT_ENGINEERING_REPAIR_NO_RETRAIN' or repair['fixed_steps']!=[256,1024]:
            raise PermissionError('E1_PRESEALED_PRECISION_REPAIR_REQUIRED')
        for step in repair['fixed_steps']:
            guard();stage='M1-'+str(step)+'-CANONICAL';checkpoint=root/'training/M1'/('checkpoint-'+str(step))
            identity=checkpoint_identity(checkpoint,stage,r['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False)
            model.probes(root,stage,instance,dataset,ledger,guard)
            serve(root,stage,instance,allocation,ledger,guard);model.unload(instance)
        stage='D-M1-1024';identity=checkpoint_identity(parent,stage,r['source_sha'])
        with ledger.transaction() as l:l['model_loads']+=1
        instance=model.load(root,parent,identity,trainable=False)
        serve(root,stage,instance,allocation,ledger,guard);model.unload(instance)
        end=time.time()+300
        while not (root/'plans/FINAL_CORRECTION_BINDING.private.json').exists():
            guard()
            if time.time()>=end:raise TimeoutError('E1_FINAL_DATA_BINDING_UNSET_NO_UNBOUNDED_GPU_WAIT')
            time.sleep(.2)
        final_binding=read(root/'plans/FINAL_CORRECTION_BINDING.private.json')
        if final_binding['status']!='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER':raise PermissionError('E1_ACTUAL_FINAL_DATA_CONSUMPTION_REQUIRED')
        write(root/'plans/CORRECTION_BINDING.private.json',final_binding)
        for branch in ['M2-BC','M2-CORR']:
            if branch=='M2-CORR' and final_binding.get('source_physics_stopped'):
                status='STOP_ACTUAL_SAFETY_OFFLINE_BC_ONLY_COMPLETED';return dict(status=status)
            guard();stage=branch+'-TRAIN';identity=checkpoint_identity(parent,'M1-1024',r['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,parent,identity,trainable=True)
            result=train(root,instance,allocation,guard,branch=branch);model.unload(instance)
            if result['global_step']!=512:raise TimeoutError('E1_PARTIAL_M2_PRESERVED_NO_RESELECT')
            stage=branch;checkpoint=root/'training'/branch/'checkpoint-512';identity=checkpoint_identity(checkpoint,branch,r['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False)
            model.probes(root,branch,instance,dataset,ledger,guard)
            serve(root,branch,instance,allocation,ledger,guard);model.unload(instance)
        if read(ledger.path)['optimizer_updates']!=2048:raise ValueError('E1_EXACT_MATCHED_TOTAL_UPDATES')
        status='ACTUAL_D_AND_MATCHED_M2_STAGES_TERMINAL';return dict(status=status)
    finally:
        cpu=time.process_time()-start;ledger.cpu(cpu)
        write(request_path.parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,CPU_s=cpu,job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),selected='NONE',deployment='NONE'));owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.request)),flush=True)
