"""Durable bounded CPU bank qualification, only after the same supplement ends."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from .durable import OwnerLock,read,write
from .resource_guard import owned_bytes
from .train_adapter import sha


def live(pid,start):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return False
    fields=path.read_text().split()
    return fields[21]==str(start) and fields[2]!='Z'


def run(args):
    root=Path(args.root);operations=Path(args.operations)
    statefile=root/'evidence/BANK_PIPELINE.json'
    owner=OwnerLock(root/'BANK_PIPELINE.lock')
    physics=OwnerLock(root/'PHYSICS_OWNER.lock')
    if statefile.exists():raise FileExistsError('Preserve prior pipeline; diagnose before explicit repair')
    prior=read(root/'evidence/DISPATCH_supplement.json')
    if prior['status']!='TERMINAL_SUCCESS' or len(prior['records'])!=24:raise ValueError('ALL_24_TERMINAL_REQUIRED')
    if live(prior['pid'],prior['proc_start']):raise ValueError('WAIT_FOR_DISPATCHER_PROCESS_EXIT')
    ledger=read(root/'RESOURCE_LEDGER.json')
    if any(r['status']=='STARTED' for r in ledger['starts']):raise ValueError('UNRECONCILED_PHYSICS_ATTEMPT')
    adoption=read(root/'ADOPTION.json')
    deadline=min(time.time()+3600,adoption['new_work_cutoff_epoch'])
    if deadline-time.time()<300:raise TimeoutError('INSUFFICIENT_CPU_STAGE_WINDOW')
    reserve=4*(deadline-time.time())
    if ledger['CPU_s']+ledger['CPU_upper_s']+reserve>=96*3600:raise ValueError('CPU_STAGE_RESERVATION')
    if owned_bytes(root)+4*1024**3>=40*1024**3 or shutil.disk_usage(root).free<14*1024**3:raise ValueError('CPU_STAGE_SPACE_MARGIN')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    ledger['CPU_upper_s']+=reserve
    ledger['events'].append(dict(stage='BANK_PIPELINE_RESERVED',CPU_upper_s=reserve,physics_steps=0))
    write(root/'RESOURCE_LEDGER.json',ledger)
    state=dict(status='RUNNING',pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        started=time.time(),deadline=deadline,steps=[],current=None,child_pid=None,CPU_reserved_s=reserve)
    write(statefile,state)
    base_env=dict(os.environ,PYTHONPATH=str(operations)+':'+str(root/'code/src'),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    initial=resource.getrusage(resource.RUSAGE_SELF)
    cpu0=initial.ru_utime+initial.ru_stime

    def execute(name,command,receipt,expected):
        if time.time()>=deadline:raise TimeoutError('CPU_STAGE_CUTOFF')
        state.update(current=name,updated=time.time());write(statefile,state)
        with (root/'logs'/('bank_'+name+'.stdout')).open('xb') as out,(root/'logs'/('bank_'+name+'.stderr')).open('xb') as err:
            child=subprocess.Popen(command,env=base_env,cwd=operations,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            state.update(child_pid=child.pid,child_start=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]);write(statefile,state)
            try:code=child.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('CPU_CHILD_CUTOFF')
        value=read(receipt) if receipt.is_file() else {}
        record=dict(stage=name,exit=code,receipt=str(receipt.relative_to(root)),status=value.get('status'),
            sha256=sha(receipt) if receipt.is_file() else None)
        state['steps'].append(record);write(statefile,state)
        if code or value.get('status')!=expected:raise RuntimeError('CPU_STAGE_GATE_FAILED:'+name)
        if owned_bytes(root)>=40*1024**3 or shutil.disk_usage(root).free<10*1024**3:raise ValueError('CPU_OUTPUT_SPACE_GUARD')
        return value

    try:
        execute('export',[args.python,'-m','scripts.pi0292.build_bank','--root',str(root),'--old',args.old],
            root/'BANK_V2_ADMISSION.json','ARX5_BANK_V2_QUALIFIED')
        native=root/'bank_v2_native'
        execute('native_all_windows',[args.python,str(root/'code/scripts/pi0291_arx5_data/qualify_loader.py'),
            '--dataset-root',str(native),'--official-source',args.source,'--cosmos-processor-cache',args.cosmos,
            '--output',str(root/'evidence/DATALOADER_SMOKE.json')],root/'evidence/DATALOADER_SMOKE.json','NATIVE_DATALOADER_PASS')
        results={}
        for split in ('TRAIN','DEV'):
            output=root/'evidence/adapter_bank_v2'/(split+'_COMPARISON.json')
            results[split]=execute('adapter_'+split,[args.python,'-m','scripts.pi0292.qualify_adapter',
                '--dataset',str(native),'--source',args.source,'--cosmos',args.cosmos,'--frozen-code',str(root/'code'),
                '--split',split,'--output',str(output)],output,'PASS_REAL_SPLIT_TENSOR_EQUIVALENCE')
        directory=root/'evidence/trainer_loader_bank_v2';directory.mkdir()
        output=directory/'LOADER_COMPARISON.json'
        loader=execute('trainer_loader',[args.python,'-m','scripts.pi0292.qualify_trainer_loader',
            '--dataset',str(native),'--source',args.source,'--cosmos',args.cosmos,'--output',str(output)],
            output,'PASS_REAL_TRAINER_LOADER_CPU_ONLY')
        manifest=read(native/'DATASET_MANIFEST.json')
        codec=read(root/'evidence/DATALOADER_SMOKE.json')
        for split in ('TRAIN','DEV'):
            if codec['codec'][split]['full_H40_windows_actually_checked']!=manifest[split+'_windows'] or manifest[split+'_windows']==0:
                raise ValueError('FULL_NONEMPTY_SPLIT_CODEC_REQUIRED')
        write(root/'evidence/TRAINER_ADAPTER_CPU_QUALIFICATION.json',dict(status='PASS_BANK_V2_TRAIN_DEV_AND_TRAINER_LOADER_CPU',
            manifest_sha256=sha(native/'DATASET_MANIFEST.json'),statistics_sha256=sha(native/'statistics.json'),
            source_operations=read(operations/'OPERATIONS_SEAL.json')['source_sha'],
            splits={split:dict(status=results[split]['status'],windows=manifest[split+'_windows']) for split in results},
            loader_status=loader['status'],no_legacy_guard_in_training=True,upstream_modified=False,
            full_codec_receipt_sha256=sha(root/'evidence/DATALOADER_SMOKE.json'),model_qualification=False))
        execute('probe_seal',[args.python,'-m','scripts.pi0292.seal_probe','--root',str(root),'--dataset',str(native),
            '--source',args.source,'--cosmos',args.cosmos,'--frozen-code',str(root/'code')],
            root/'evidence/OPEN_LOOP_PROBE_SEAL.json','SEALED_BEFORE_MAIN')
        state.update(status='BANK_V2_CPU_AND_EXPERT_PROBE_QUALIFIED',current=None,child_pid=None)
    except Exception as error:
        state.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        used=own.ru_utime+own.ru_stime-cpu0+children.ru_utime+children.ru_stime
        state.update(CPU_s=used,finished=time.time());write(statefile,state)
        ledger=read(root/'RESOURCE_LEDGER.json');ledger['CPU_s']+=used;ledger['CPU_upper_s']-=reserve
        ledger['events'].append(dict(stage='BANK_PIPELINE',CPU_s=used,status=state['status'],physics_steps=0))
        ledger['owned_bytes']=owned_bytes(root)
        write(root/'RESOURCE_LEDGER.json',ledger)
        physics.close();owner.close()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','old','operations','python','source','cosmos'):p.add_argument('--'+name,required=True)
    result=run(p.parse_args());print(result['status'])
