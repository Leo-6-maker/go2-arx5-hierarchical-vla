"""One recorded-observation draw through the existing E1 loader and Processor.

This diagnostic has no simulator, relay server or formal evaluation outcomes.
The management entry registers the output, pins inputs and grants the finite
local run before launch. No remote accounting or queue is consulted here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def node_local(path):
    path=Path(path).resolve(strict=True)
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=left.split()[4]
        if str(path)==mount or str(path).startswith(mount.rstrip('/')+'/'):
            mounts.append((len(mount),right.split()[0]))
    if not mounts or max(mounts)[1]not in('ext4','xfs','btrfs','zfs'):
        raise PermissionError('SMOKE_INPUT_MUST_BE_NODE_LOCAL:'+str(path))
    return path


def run(root, manifest_sha256):
    root=Path(root).resolve(strict=True);manifest=root/'RUN_MANIFEST.json'
    if sha(manifest)!=manifest_sha256:
        raise PermissionError('SMOKE_MANIFEST_CHANGED')
    m=read(manifest)
    if (m['kind']!='ONE_E1_RECORDED_OBSERVATION_INFERENCE_DEBUG'
        or m['run_uuid']!=root.name or m['owner_uid']!=os.getuid()
        or socket.gethostname()!='corelab-4080s' or os.getuid()!=2009
        or m['limits']['model_loads']!=1 or m['limits']['generation_queries']!=1
        or m['limits']['physics_integrations']!=0 or m['limits']['optimizer_updates']!=0
        or not 0< m['limits']['wall_seconds']<=900
        or not time.time()<m['deadline_epoch']):
        raise PermissionError('SMOKE_FINITE_REGISTERED_DEBUG_IDENTITY')
    lease=read(root/'control/LEASE.json')
    if lease['status']!='ACTIVE'or lease['run_uuid']!=root.name or lease['manifest_sha256']!=manifest_sha256:
        raise PermissionError('SMOKE_OWN_ACTIVE_LEASE')
    registry=read(root/'control/REGISTERED_PATHS.json')
    if not any(x.get('path')==str(root)and x.get('role')==m['role']and x.get('run_uuid')==root.name for x in registry['entries']):
        raise PermissionError('SMOKE_EXISTING_REGISTRY_OUTPUT')
    for name,digest in m['sealed_inputs'].items():
        p=root/name
        if not p.resolve().is_relative_to(root)or sha(p)!=digest:
            raise PermissionError('SMOKE_SEALED_INPUT_CHANGED:'+name)
    node_local(root);bindings=read(root/'NEURAL_BINDINGS.private.json')
    for name in('source','python','cosmos','dataset','cosmos_hub'):
        node_local(bindings[name])
    checkpoint=node_local(m['checkpoint'])
    dependency=node_local(m['dependency_code'])
    import sys
    sys.path[:0]=[str(dependency),str(dependency/'src')]
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0292.relay_protocol import unseal,validate_observation
    from scripts.pi0292.eval_contract import gate_action
    from scripts.pi0294.numerics import prepare_environment,configure_torch
    from scripts.pi0294.e1 import model
    from scripts.pi0294.e1.neural import checkpoint_identity
    from scripts.pi0294.e1.artifacts import atomic_npz
    lock=OwnerLock(root.parent/'global-native-GPU-locks'/(m['GPU_uuid']+'.lock'))
    instance=None;loads=queries=0;phase='LOCAL_ADMISSION';failure=None
    began=time.monotonic();cpu=time.process_time()
    def stop(sig,frame):
        raise TimeoutError('SMOKE_FINITE_WALL_OR_OWN_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(min(m['limits']['wall_seconds'],max(1,int(m['deadline_epoch']-time.time()))))
    def guard():
        if time.time()>=m['deadline_epoch']or(root/'STOP.json').exists():
            raise PermissionError('SMOKE_DEADLINE_OR_OWN_STOP')
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=8,check=True)
        row=next(x.split(', ')for x in q.stdout.splitlines()if x.split(', ')[0]==m['GPU_uuid'])
        if '4080'not in row[1]or int(row[2])*1024**2<30*1024**3 or int(row[3])*1024**2<24*1024**3:
            raise PermissionError('SMOKE_REAL_TOTAL30_FREE24_GIB')
        if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<m['limits']['output_bytes']+1024**3:
            raise PermissionError('SMOKE_REAL_OUTPUT_SPACE')
        put(root/'NATIVE_STARTED.json',dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),observed_epoch=time.time(),GPU=row))
        prepare_environment(m['numerical_mode'])
        import torch
        configure_torch(torch,m['numerical_mode']);torch.set_num_threads(4)
        if torch.cuda.device_count()!=1:raise PermissionError('SMOKE_ONE_VISIBLE_DEVICE')
        identity=checkpoint_identity(checkpoint,'E1_DEBUG',m['code_source_sha'])
        if identity['checkpoint_manifest_sha256']!=m['checkpoint_manifest_sha256']or identity['global_step']!=512:
            raise PermissionError('SMOKE_EXACT_EXISTING_E1_512')
        observation=unseal(read(root/'RECORDED_OBSERVATION.json'))['payload'];validate_observation(observation)
        guard();phase='MODEL_LOAD';loads=1
        put(root/'MODEL_LOAD_INTENT.json',dict(checkpoint_manifest_sha256=m['checkpoint_manifest_sha256'],observed_epoch=time.time()))
        instance=model.load(root,checkpoint,identity,trainable=False)
        put(root/'MODEL_LOADED.json',dict(identity=identity,observed_epoch=time.time(),elapsed_seconds=time.monotonic()-began))
        guard();phase='FREE_GENERATION';queries=1
        put(root/'GENERATION_INTENT.json',dict(noise_seed=m['noise_seed'],observed_epoch=time.time()))
        value=instance.generate(observation,m['noise_seed'])
        atomic_npz(root/'DRAW.npz',physical_H40=value['physical_H40'],normalized=value['normalized'])
        gate=gate_action(value['physical_H40'],observation['state_eef10'],0.,root.name)
        put(root/'DRAW.json',dict(noise=value['noise'],inference_precision=value['inference_precision'],gate=gate,
            physical_shape=list(value['physical_H40'].shape),arrays_sha256=sha(root/'DRAW.npz'),
            recorded_observation_sha256=m['sealed_inputs']['RECORDED_OBSERVATION.json'],observed_epoch=time.time()))
        phase='COMPLETE_ONE_REAL_DRAW_NOT_FORMAL_EVAL'
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        raise
    finally:
        signal.alarm(0)
        try:
            if instance is not None:model.unload(instance)
        finally:
            put(root/'RESULT.json',dict(status='COMPLETE'if failure is None else'FAILED',phase=phase,error=failure,
                model_load_attempts=loads,generation_attempts=queries,completed_generations=int((root/'DRAW.json').exists()),
                CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began,completed_epoch=time.time(),
                physics_integrations=0,optimizer_updates=0,formal_valid_conditions=0,task_success_claimed=False))
            lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest-sha256',required=True)
    args=p.parse_args();run(args.root,args.manifest_sha256)
