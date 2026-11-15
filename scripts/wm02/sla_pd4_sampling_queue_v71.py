"""Run the fixed64 sampling diagnostic after the current P0 resource window.

This is a finite registered user service, not a new scheduler. It reuses the
canonical neural owner and never opens, stops or changes the P0/native jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

STOP = False


def request_stop(*args):
    global STOP
    STOP = True


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_cold(spec):
    """A closed file alone is insufficient to pass the resource dependency."""
    pin=spec['P0_dependency'];root=Path(pin['root'])
    if sha(root/'EXECUTION_SPEC.json') != pin['spec_sha256']:
        raise PermissionError('EXACT_EXISTING_P0_DEPENDENCY_REQUIRED')
    state=dict(line.split('=',1)for line in subprocess.check_output(['systemctl','--user','show',
        'wm02-'+root.name,'--property=MainPID,SubState'],text=True,timeout=15).splitlines())
    try:lease=read(root/'control/LEASE.json')
    except FileNotFoundError:return None
    if state['MainPID']!='0' or lease['status']!='CLOSED':return None
    try:
        worker=read(root/'WORKER_EXIT.private.json');result=read(root/'RESULT.private.json')
    except FileNotFoundError:
        return None
    if worker['exit_code']!=lease['worker_exit_code'] or result['exit_code']!=worker['exit_code']:
        raise PermissionError('P0_TERMINAL_RECEIPTS_MUST_AGREE')
    for actor in (root/'runs').iterdir():
        lease=read(actor/'control/LEASE.json')
        if lease['status']!='CLOSED':return None
        if (actor/'SUPERVISOR.private.json').exists():
            try:exited=read(actor/'WORKER_EXIT.private.json')
            except FileNotFoundError:return None
            if exited['exit_code']!=lease['worker_exit_code']:
                raise PermissionError('P0_ACTOR_TERMINAL_RECEIPTS_MUST_AGREE')
            pid=read(actor/'SUPERVISOR.private.json')['pid']
            try:command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
            except FileNotFoundError:command=b''
            if str(actor/'EXECUTION_SPEC.json').encode() in command:return None
    return dict(unit_main_pid=0,result_sha256=sha(root/'RESULT.private.json'),
        worker_exit_code=worker['exit_code'],scope_closed=True,
        science_completion_not_used_as_diagnostic_qualification=True)


def run(spec_path):
    spec_path=Path(spec_path);s=read(spec_path);R=Path(s['registered_root']);S=Path(s['source_root'])
    sys.path[:0]=[str(S/'src'),str(S)]
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    assert subprocess.check_output(['systemctl','--user','show','wm02-'+R.name,
        '--property=MainPID','--value'],text=True,timeout=15).strip()==str(os.getpid())
    lease=read(R/'control/LEASE.json')
    assert lease['status']=='ACTIVE'and lease['spec_sha256']==sha(spec_path)
    # A restarted parent must reconcile its prior child, never close or replay it.
    if (R/'SUPERVISOR.private.json').exists():
        raise PermissionError('RECONCILE_PREVIOUS_DIAGNOSTIC_PARENT_NOT_RESTART')
    started=time.time();held=child=None;code=1;error=None;gpu_started=None;usage=None;result=None
    try:
        if (s['scope']!='SLA_PD4_FIXED500_SAMPLER20_DIAGNOSTIC_V71'
                or s['planned_generations']!=64 or s['physics']is not False
                or s['planned_optimizer_updates']!=0):
            raise PermissionError('REGISTERED_FIXED64_GPU_DIAGNOSTIC_ONLY')
        for name,pin in s['implementation_files'].items():
            assert sha(R/name)==pin
        assert sha(R/'DECLARATION.private.json')==s['declaration_sha256']
        write(R/'SUPERVISOR.private.json',dict(pid=os.getpid(),started_epoch=started,
            implementation_source_commit=s['implementation_source_commit']))
        j=Journal(R/'control/journal',R.name)
        while time.time()<s['deadline_epoch']-3720:
            if STOP or (R/'STOP.json').exists():raise InterruptedError('OWNED_SAMPLING_STOP')
            prior=dependency_cold(s)
            if prior is None:
                write(R/'PROGRESS.private.json',dict(state='WAIT_CURRENT_P0_WINDOW_NO_MODEL',epoch=time.time(),
                    optimizer_updates=0,model_generations=0,physical_requests=0))
                time.sleep(30);continue
            free=int(subprocess.check_output(['nvidia-smi','--id='+s['GPU_uuid'],
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())*1024**2
            ram=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))
            if free>s['allocator_limit_bytes']+1024**3 and ram>s['frame_cache_bytes']+24*1024**3:
                try:held=OwnerLock(s['neural_owner_lock'])
                except BlockingIOError:held=None
                if held:break
            write(R/'PROGRESS.private.json',dict(state='WAIT_SHARED_CAPACITY_AFTER_P0_NO_MODEL',epoch=time.time(),
                optimizer_updates=0,model_generations=0,physical_requests=0))
            time.sleep(30)
        else:
            result=dict(status='UNSTARTED_FINITE_SITE_WINDOW',exit_code=75,actual_SLA_parent_loads=0,
                actual_generations=0,optimizer_updates=0,training_visits=0,physical_requests=0,automatic_retry=False)
            code=75;return result
        assert dependency_cold(s) is not None
        gpu_started=time.time();end=min(gpu_started+3600,s['deadline_epoch']-120)
        write(R/'GPU_OWNER_ACTUAL.private.json',dict(epoch=gpu_started,GPU_uuid=s['GPU_uuid'],
            shared_GPU=True,other_jobs_modified=False,prior_P0=prior))
        b=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),
            GPU_uuid=s['GPU_uuid'],GPU_name=s['GPU_name'],GPU_headroom_bytes=1024**3,
            allocator_limit_bytes=s['allocator_limit_bytes'],host_headroom_bytes=24*1024**3,
            allocation_deadline_epoch=s['deadline_epoch'],paths={k:str(Path(s[k]).resolve(strict=True))for k in(
                'parent_params','dataset','compilation_cache_dir')},allocation=dict(kind='EXISTING_USER_SERVICE',
                campaign_root=str(R),neural_owner_lock=s['neural_owner_lock']))
        write(R/'SITE_BINDING.private.json',b);folder=R/'inference';folder.mkdir()
        q=dict(s['inference_template'],registered_root=str(folder),deadline_epoch=end,max_walltime_s=3600,
            max_gpu_hours=1,site_binding=dict(path=str(R/'SITE_BINDING.private.json'),sha256=sha(R/'SITE_BINDING.private.json')))
        write(folder/'EXECUTION_SPEC.json',q)
        env=dict(os.environ,PYTHONPATH=str(S/'src')+':'+str(S)+':'+s['upstream_root']+'/src:'
            +s['upstream_root']+'/packages/openpi-client/src',CUDA_VISIBLE_DEVICES=s['GPU_uuid'],JAX_PLATFORMS='cuda',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION=str(s['JAX_memory_fraction']),
            XLA_FLAGS=read(s['initial_function_rule']['path'])['XLA_FLAGS'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('LD_PRELOAD',None)
        argv=[s['python'],'-u','-B',str(R/'diagnose_pd4_sampling_v71.py'),str(folder/'EXECUTION_SPEC.json')]
        def launch():
            nonlocal child,usage
            write(R/'INFERENCE_INTENT.private.json',dict(argv=argv,automatic_retry=False,planned_generations=64))
            usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            log=(R/'logs/inference.log').open('xb')
            try:child=subprocess.Popen(argv,cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            return dict(pid=child.pid,epoch=time.time(),source_commit=s['implementation_source_commit'])
        j.once('INFERENCE_LAUNCH',dict(spec_sha256=sha(folder/'EXECUTION_SPEC.json')),launch)
        assert child is not None,'RECONCILE_PREVIOUS_OR_UNKNOWN_INFERENCE_NOT_REPLAY'
        write(R/'PROGRESS.private.json',dict(state='FIXED64_RGB20_RUNNING',epoch=time.time(),optimizer_updates=0,
            planned_generations=64,physical_requests=0))
        while child.poll()is None:
            if STOP or (R/'STOP.json').exists():raise InterruptedError('OWNED_SAMPLING_STOP')
            if time.time()>=end:raise TimeoutError('FINITE_FIXED64_INFERENCE_ENDPOINT')
            time.sleep(1)
        assert child.returncode==0,child.returncode
        actual=read(folder/'RESULT.private.json')
        assert(actual['actual_generations'],actual['model_executions_completed'],actual['generation_attempts'])==(64,64,64)
        assert actual['optimizer_updates']==actual['physical_requests']==0
        result=dict(status='FIXED64_RGB20_CLOSED_SUCCESS_DIAGNOSTIC_ONLY',exit_code=0,
            inference_result_sha256=sha(folder/'RESULT.private.json'),
            actual_SLA_parent_loads=actual['actual_SLA_parent_loads'],actual_generations=64,
            summaries=actual['summaries'],optimizer_updates=0,training_visits=0,physical_requests=0,
            GPU_owned_wall_s=time.time()-gpu_started,automatic_retry=False)
        code=0;return result
    except BaseException:
        error=traceback.format_exc();write(R/'FAILED.private.json',dict(error=error,automatic_retry=False,
            optimizer_updates=0,physical_requests=0));raise
    finally:
        if child and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if held:held.close()
        if result is not None:write(R/'RESULT.private.json',dict(result,wall_s=time.time()-started))
        counts=R/'inference/COUNTERS.private.json'
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu=(after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime)if usage else 0.
        write(R/'WORKER_EXIT.private.json',dict(exit_code=code,error=error,wall_s=time.time()-started,
            inference_child_CPU_s=child_cpu,GPU_owned_wall_s=time.time()-gpu_started if gpu_started else 0.,
            counters=read(counts)if counts.exists()else None,optimizer_updates=0,physical_requests=0))
        lease=read(R/'control/LEASE.json')
        write(R/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))


if __name__=='__main__':
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path);a=p.parse_args()
    result=run(a.spec);print(json.dumps(result));raise SystemExit(result['exit_code'])
