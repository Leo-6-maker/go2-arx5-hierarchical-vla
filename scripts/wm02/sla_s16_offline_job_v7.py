"""Finite user-authorized offline S16 service, existing lock and child workers.

This does not renew/rebuild a simulation campaign or admit any native task.
The run's own finite lease backs the existing local-user-service site binding.
"""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

_SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_SOURCE/'src'), str(_SOURCE)]

from scripts.pi0292.durable import OwnerLock, write
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.sla_adapter.s16_offline_v7 import OPERATION


def identity(spec):
    root=Path(spec['registered_root'])
    if (spec['scope']!='SLA_OFFLINE_S16_ONLY_V7' or spec['operation']!=OPERATION
            or spec['physics'] is not False or spec['native_execution_admitted'] is not False
            or spec['campaign_root']!=str(root) or not spec['user_authorization']
            or os.getuid()!=2009 or os.uname().nodename!='corelab-4080s'
            or time.time()>=spec['deadline_epoch']):
        raise PermissionError('EXPLICIT_FINITE_OFFLINE_S16_USER_SERVICE_ONLY')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    registry=json.loads(Path(spec['registry']).read_bytes())
    if (lease['status']!='ACTIVE' or lease['spec_sha256']!=digest(root/'EXECUTION_SPEC.json')
            or lease['owner']!='EXISTING_LOCAL_PC' or lease['scope']!=spec['scope']
            or not any(e.get('path')==str(root) and e.get('source_sha')==spec['source_sha']
                       and e.get('scope')==spec['scope'] for e in registry['entries'])):
        raise PermissionError('ORIGINAL_REGISTRY_AND_OWN_FINITE_LEASE_REQUIRED')
    pid=subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,
        '--property=MainPID','--value'],text=True,timeout=15).strip()
    if pid!=str(os.getpid()):raise PermissionError('ACTUAL_OWN_USER_SERVICE_PID_REQUIRED')


def run(spec):
    root=Path(spec['registered_root']);source=Path(spec['source_root'])
    start=time.time();held=None;child=None;code=1;owned=None
    def stop(signum,frame):
        if child and child.poll()is None:child.send_signal(signum)
        raise KeyboardInterrupt('FINITE_OWN_S16_SERVICE_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(source/'src')+':'+str(source)+':'+spec['upstream_root']+'/src:'+
        spec['upstream_root']+'/packages/openpi-client/src',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    env.pop('LD_PRELOAD',None)
    def call(folder,replay=False):
        nonlocal child
        argv=[spec['python'],'-B','-m','scripts.sla.train_s16_offline_v7',str(folder/'EXECUTION_SPEC.json')]
        if replay:argv.append('--replay')
        write(folder/'LAUNCH_INTENT.private.json',dict(argv=argv,epoch=time.time(),automatic_retry=False))
        with(root/'logs'/(folder.name+'.log')).open('xb')as stream:
            child=subprocess.Popen(argv,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            return child.wait(timeout=max(1,spec['deadline_epoch']-time.time()))
    try:
        identity(spec)
        if (root/'STARTED.private.json').exists():raise FileExistsError('S16_NO_RESTART_OR_REPLAY')
        write(root/'STARTED.private.json',dict(epoch=start,pid=os.getpid(),physical_requests=0))
        while time.time()<spec['deadline_epoch']-3600:
            identity(spec)
            if(root/'STOP.json').exists():raise TimeoutError('OFFLINE_S16_STOP_BEFORE_LOAD')
            free=int(subprocess.check_output(['nvidia-smi','-i',spec['GPU_uuid'],'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip())*1024**2
            ram=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))
            if free>spec['allocator_limit_bytes']+1024**3 and ram>spec['frame_cache_bytes']+24*1024**3:
                try:held=OwnerLock(spec['neural_owner_lock'])
                except BlockingIOError:held=None
                if held:break
            write(root/'PROGRESS.private.json',dict(state='WAIT_SHARED_GPU_CAPACITY_NO_MODEL',epoch=time.time(),
                free_GPU_bytes=free,available_host_bytes=ram,optimizer_updates=0,physical_requests=0))
            time.sleep(60)
        else:raise TimeoutError('FINITE_S16_CAPACITY_WAIT_EXPIRED')
        owned=time.time()
        write(root/'GPU_OWNER_ACTUAL.private.json',dict(epoch=owned,GPU_uuid=spec['GPU_uuid'],
            neural_owner_lock=spec['neural_owner_lock'],shared_GPU=True,other_user_processes_modified=False))
        binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),
            GPU_uuid=spec['GPU_uuid'],GPU_name=spec['GPU_name'],GPU_headroom_bytes=1024**3,
            allocator_limit_bytes=spec['allocator_limit_bytes'],host_headroom_bytes=24*1024**3,
            allocation_deadline_epoch=spec['deadline_epoch'],
            paths={k:str(Path(spec[k]).resolve(strict=True))for k in('parent_params','dataset','compilation_cache_dir')},
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock']))
        write(root/'SITE_BINDING.private.json',binding)
        env.update(CUDA_VISIBLE_DEVICES=spec['GPU_uuid'],JAX_PLATFORMS='cuda',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION=str(spec['JAX_memory_fraction']),
            XLA_FLAGS=json.loads(Path(spec['initial_function_rule']['path']).read_bytes())['XLA_FLAGS'])
        train=root/'candidate500';train.mkdir()
        q=dict(spec,registered_root=str(train),deadline_epoch=spec['deadline_epoch']-180,
            site_binding=dict(path=str(root/'SITE_BINDING.private.json'),sha256=digest(root/'SITE_BINDING.private.json')))
        write(train/'EXECUTION_SPEC.json',q)
        write(root/'PROGRESS.private.json',dict(state='S16_ACTUAL_TRAINING_WORKER_STARTED',epoch=time.time(),physical_requests=0))
        rc=call(train)
        if rc not in(0,75):raise RuntimeError('S16_TRAINING_CHILD_FAILED:'+str(rc))
        trained=json.loads((train/'RESULT.private.json').read_bytes())
        if rc==75:
            write(root/'RESULT.private.json',dict(trained,complete500=False,automatic_retry=False));code=75
        else:
            assert trained['step']==500 and trained['target_visits']==16000
            replay=root/'fresh_RGB';replay.mkdir()
            t=dict(q,registered_root=str(replay),package=trained['package'],
                reference_probes=str(train/'work/step500-RGB/MANIFEST.private.json'))
            write(replay/'EXECUTION_SPEC.json',t)
            write(root/'PROGRESS.private.json',dict(state='S16_FRESH_ORDINARY_RGB32_REPLAY',epoch=time.time(),optimizer_updates=500,physical_requests=0))
            if call(replay,True)!=0:raise RuntimeError('S16_FRESH_RGB_CHILD_FAILED')
            rgb=json.loads((replay/'REPLAY_RESULT.private.json').read_bytes())
            assert rgb['all_bit_exact']and rgb['ordinary_RGB_generations']==32
            write(root/'RESULT.private.json',dict(trained,
                status='S16_OFFLINE500_AND_RGB32_CLOSED_FIT_RESULTS_RETAINED',
                complete500=True,fresh_RGB32_bit_exact=True,
                fresh_RGB_result=dict(path=str(replay/'REPLAY_RESULT.private.json'),sha256=digest(replay/'REPLAY_RESULT.private.json')),
                total_model_loads=trained['actual']['actual_parent_loads']+rgb['actual_parent_loads'],
                total_ordinary_RGB_generations=trained['actual']['ordinary_RGB_generations']+rgb['ordinary_RGB_generations'],
                wait_wall_s=owned-start,GPU_owned_wall_s=time.time()-owned,wall_s=time.time()-start,
                automatic_retry=False,automatic_native_dispatch=False,goal_complete=False))
            code=0
    except BaseException:
        write(root/'FAILED.private.json',dict(error=traceback.format_exc(),physical_requests=0,automatic_retry=False))
        traceback.print_exc()
    finally:
        if child and child.poll()is None:
            child.terminate()
            try:child.wait(timeout=180)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if held:held.close()
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.time()-start,
            GPU_owned_wall_s=0 if owned is None else time.time()-owned,physical_requests=0))
        lease=json.loads((root/'control/LEASE.json').read_bytes())
        write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path)
    args=p.parse_args();raise SystemExit(run(json.loads(args.spec.read_bytes())))
