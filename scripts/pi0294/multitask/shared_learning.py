"""Registered finite shared training entry using the existing native fit loop.

The existing owner stages/leases/launches/closes this worker. It does not submit
jobs, allocate a second owner, contact Vanda or modify a running capsule.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
    entry=parser.parse_args();body=(Path(entry.root)/'RUN_MANIFEST.json').read_bytes()
    if hashlib.sha256(body).hexdigest()!=entry.manifest_sha256:
        raise PermissionError('SHARED_ENTRY_SEALED_MANIFEST_BEFORE_PROJECT_IMPORT')
    dependency=json.loads(body)['dependency_code'];sys.path[:0]=[dependency,dependency+'/src']


def run(root,manifest_sha256):
    from scripts.pi0292.durable import read,write,OwnerLock
    from scripts.pi0294.e1.smoke import node_local,put
    from scripts.pi0294.native_dataset import NativeDataset,sha,official_import
    from scripts.pi0294.generalization.minimal_model import unchanged_raw_parent
    from scripts.pi0294.multitask.shared_training import validate_manifest,validate_data,qualified_entry
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    root=node_local(root);m=read(root/'RUN_MANIFEST.json');p=read(root/'TRAIN_FIT_PLAN.json')
    validate_manifest(m,p)
    if (sha(root/'RUN_MANIFEST.json')!=manifest_sha256 or m.get('owner_uid')!=os.getuid()
            or os.uname().nodename!='corelab-4080s'):
        raise PermissionError('SHARED_EXISTING_NATIVE_OWNER_AND_SEALED_MANIFEST')
    lease=read(root/'control/LEASE.json')
    if lease.get('status')!='ACTIVE'or lease.get('manifest_sha256')!=manifest_sha256:
        raise PermissionError('SHARED_EXISTING_ACTIVE_OWNER_LEASE')
    if not any(r.get('path')==str(root)and r.get('role')==m['role']
               for r in read(root/'control/REGISTERED_PATHS.json')['entries']):
        raise PermissionError('SHARED_EXISTING_PATH_REGISTRY')
    for name,pin in m['sealed_inputs'].items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve())or sha(path)!=pin:
            raise PermissionError('SHARED_SEALED_INPUT_CHANGED:'+name)
    bindings=read(root/'NEURAL_BINDINGS.private.json')
    for name in ('source','python','cosmos','dataset','cosmos_hub'):node_local(bindings[name])
    raw=node_local(m['raw_parent']);audit=read(root/'RAW_PARENT_AUDIT.json')
    unchanged_raw_parent(raw,audit)
    if m.get('sealed_inputs',{}).get('RAW_PARENT_AUDIT.json')!=sha(root/'RAW_PARENT_AUDIT.json'):
        raise PermissionError('SHARED_TRANSFER_AUDIT_IS_SEALED')
    data=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],for_optimization=True)
    validate_data(data,p)
    if m.get('GPU_training_admission')is not None:node_local(m['GPU_training_admission']['qualification_root'])
    required=qualified_entry(m)
    lock=OwnerLock(root.parent/'global-native-GPU-locks'/(m['GPU_uuid']+'.lock'))
    started=time.monotonic();CPU=time.process_time();loads=0;failure=None;fit=None;model=instance=contract=None
    def guard():
        if time.time()>=m['deadline_epoch']or(root/'STOP.json').exists():
            raise PermissionError('SHARED_FINITE_OWNER_DEADLINE_OR_STOP')
        fs=os.statvfs(root)
        if fs.f_bavail*fs.f_frsize<m['protected_remaining_bytes']+m['free_margin_bytes']:
            raise PermissionError('SHARED_PROTECTED_REMAINING_WORK_AND_DISK_MARGIN')
    def stop(sig,frame):raise TimeoutError('SHARED_NATIVE_FINITE_END')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
    try:
        guard();prepare_environment(MODE);official_import(bindings['source'])
        import torch
        from transformers import set_seed
        from scripts.pi0294.multitask.shared_raw_model import load as load_raw
        from scripts.pi0294.multitask.book_fit import run as run_fit
        configure_torch(torch,MODE);torch.set_num_threads(4);set_seed(p['seed'])
        count=torch.cuda.device_count();free,total=torch.cuda.mem_get_info()if count==1 else(0,0)
        actual_uuid=str(torch.cuda.get_device_properties(0).uuid)if count==1 else None
        put(root/'GPU_ENTRY_OBSERVATION.json',dict(device_count=count,free_bytes=free,total_bytes=total,
            required_free_bytes=required,actual_GPU_uuid=actual_uuid,epoch=time.time()))
        if count!=1 or free<required or actual_uuid!=m['GPU_uuid']:
            raise PermissionError('SHARED_ACTUAL_SINGLE_QUALIFIED_DEVICE_AND_CAPACITY')
        put(root/'MODEL_LOAD_INTENT.json',dict(raw_parent_files_identity=p['raw_parent_files_identity'],epoch=time.time()));loads+=1
        instance,contract=load_raw(root,m,bindings,p,guard);model=instance.model
        original_generate=instance.generate
        def timed_generate(request,seed):
            signal.alarm(max(1,min(60,int(m['deadline_epoch']-time.time()))))
            try:return original_generate(request,seed)
            finally:signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
        instance.generate=timed_generate
        write(root/'MODEL_LOADED.json',dict(identity=instance.identity,contract=contract.describe(),
            raw_parameter_initialization_verified=True,observed_epoch=time.time()))
        fit=run_fit(root,instance,contract,guard);write(root/'TRAIN_FIT_RESULT.json',fit)
    except BaseException as error:
        failure=type(error).__name__+': '+str(error);raise
    finally:
        signal.alarm(0)
        progress=read(root/'FIT_PROGRESS.json')if(root/'FIT_PROGRESS.json').exists()else dict(step=0,cursor=0)
        write(root/'RESULT.json',dict(status='COMPLETE_SHARED_RAW_FINITE_LEARNING'if fit is not None else'FAILED_SHARED_RAW_FINITE_LEARNING',
            error=failure,scientific_kind=m['kind'],model_load_attempts=loads,
            generation_attempts=len(list(root.glob('*/*.INTENT.json'))),completed_generations=len(list(root.glob('*/*.RESULT.json'))),
            optimizer_updates=progress['step'],TRAIN_visits=progress['cursor'],physics_integrations=0,DEV_conditions=0,
            CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,completed_epoch=time.time()))
        data=model=instance=contract=None;gc.collect();lock.close()


if __name__=='__main__':run(entry.root,entry.manifest_sha256)
