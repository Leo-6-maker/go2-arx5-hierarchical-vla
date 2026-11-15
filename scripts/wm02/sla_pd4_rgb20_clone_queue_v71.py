"""One finite CPU follower of the already registered RGB20 job.

No GPU ownership, model execution or physical dispatch is implemented here.
An upstream failure is retained and never causes a replacement inference job.
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


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def completed_sampling(pin):
    root=Path(pin['root'])
    if sha(root/'EXECUTION_SPEC.json')!=pin['spec_sha256']:
        raise PermissionError('PINNED_EXISTING_RGB20_JOB_REQUIRED')
    pid=subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,
        '--property=MainPID','--value'],text=True,timeout=15).strip()
    lease=read(root/'control/LEASE.json')
    if pid!='0'or lease['status']!='CLOSED':return None
    try:worker=read(root/'WORKER_EXIT.private.json')
    except FileNotFoundError:return None
    if worker['exit_code']!=lease['worker_exit_code']:
        raise PermissionError('UPSTREAM_TERMINAL_RECEIPTS_DISAGREE')
    if worker['exit_code']!=0:
        raise RuntimeError('UPSTREAM_RGB20_FAILURE_RETAINED_NO_REPLACEMENT')
    result=read(root/'RESULT.private.json');path=root/'inference/RESULT.private.json'
    if result['exit_code']!=0 or result['actual_generations']!=64:
        raise PermissionError('ACTUAL_FIXED64_COMPLETE_REQUIRED')
    if sha(path)!=result['inference_result_sha256']:
        raise PermissionError('COMPLETED_RGB20_BYTES_REQUIRED')
    return dict(path=str(path),sha256=sha(path),parent_result_sha256=sha(root/'RESULT.private.json'))


def run(path):
    path=Path(path);s=read(path);R=Path(s['registered_root']);S=Path(s['source_root'])
    sys.path[:0]=[str(S/'src'),str(S)]
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.journal import Journal
    assert subprocess.check_output(['systemctl','--user','show','wm02-'+R.name,
        '--property=MainPID','--value'],text=True,timeout=15).strip()==str(os.getpid())
    lease=read(R/'control/LEASE.json');assert lease['status']=='ACTIVE'and lease['spec_sha256']==sha(path)
    if(R/'SUPERVISOR.private.json').exists():raise PermissionError('RECONCILE_PREVIOUS_CPU_FOLLOWER_NOT_RESTART')
    child=None;result=None;code=1;started=time.time();usage=None
    try:
        assert s['maximum_clone_calls']==64 and s['physics']is False and s['optimizer_updates']==0
        for name,pin in s['implementation_files'].items():assert sha(R/name)==pin
        assert sha(R/'DECLARATION.private.json')==s['declaration_sha256']
        baseline=Path(s['baseline81']);assert sha(baseline)==s['baseline81_sha256']
        assert read(baseline.parent.parent/'control/LEASE.json')['status']=='CLOSED'
        write(R/'SUPERVISOR.private.json',dict(pid=os.getpid(),started_epoch=started))
        while time.time()<s['deadline_epoch']-660:
            if(R/'STOP.json').exists():raise InterruptedError('OWNED_CPU_FOLLOWER_STOP')
            sampling=completed_sampling(s['sampling_dependency'])
            available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
            if sampling is not None and available>3*1024**3:break
            state='WAIT_EXISTING_RGB20_NO_MODELS_OR_CLONES'if sampling is None else'WAIT_CPU_HEADROOM_NO_CLONES'
            write(R/'PROGRESS.private.json',dict(state=state,epoch=time.time(),
                clone_certifier_calls=0,new_model_generations=0,optimizer_updates=0,physical_requests=0))
            time.sleep(30)
        else:
            result=dict(status='UNSTARTED_FINITE_CPU_WINDOW',exit_code=75,clone_certifier_calls=0,
                new_model_generations=0,optimizer_updates=0,physical_requests=0,automatic_retry=False)
            code=75;return result
        q=dict(s['analysis_template'],sampling_result=sampling)
        write(R/'INPUT_SPEC.private.json',q)
        env=dict(os.environ,**s['native_environment']);env.update(PYTHONPATH=str(S/'src')+':'+str(S)+':'
            +s['native_environment'].get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_PRELOAD',None)
        j=Journal(R/'control/journal',R.name)
        def launch():
            nonlocal child,usage
            argv=[s['python'],'-u','-B',str(R/'diagnose_pd4_rgb20_joints_v71.py'),str(R/'INPUT_SPEC.private.json'),str(R/'analysis')]
            write(R/'CLONE_WORKER_INTENT.private.json',dict(argv=argv,maximum_clone_calls=64,automatic_retry=False))
            usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            with(R/'logs/analysis.log').open('xb')as f:
                child=subprocess.Popen(argv,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            return dict(pid=child.pid,epoch=time.time())
        j.once('CLONE_WORKER',dict(input_sha256=sha(R/'INPUT_SPEC.private.json')),launch)
        assert child is not None,'RECONCILE_UNKNOWN_CHILD_NEVER_REPLAY'
        write(R/'PROGRESS.private.json',dict(state='SAVED_RGB20_CLONE_ANALYSIS_RUNNING',epoch=time.time(),
            planned_variants=64,new_model_generations=0,optimizer_updates=0,physical_requests=0))
        assert child.wait(timeout=540)==0
        actual=read(R/'analysis/RESULT.private.json')
        assert actual['variants']==64 and actual['clone_certifier_attempts']==actual['clone_certifier_calls_completed']<=64
        assert actual['new_model_generations']==actual['optimizer_updates']==actual['physical_requests']==0
        result=dict(status='RGB20_RECORDED_CONTEXT_DIAGNOSTIC_CLOSED',exit_code=0,
            analysis_sha256=sha(R/'analysis/RESULT.private.json'),summaries=actual['summaries'],
            clone_certifier_calls=actual['clone_certifier_calls_completed'],new_model_generations=0,
            optimizer_updates=0,physical_requests=0,automatic_retry=False)
        code=0;return result
    except BaseException:
        write(R/'FAILED.private.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if child and child.poll()is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if result is not None:write(R/'RESULT.private.json',dict(result,wall_s=time.time()-started))
        after=resource.getrusage(resource.RUSAGE_CHILDREN);counts=R/'analysis/COUNTERS.private.json'
        write(R/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.time()-started,
            child_CPU_s=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime if usage else 0.,
            counters=read(counts)if counts.exists()else None,optimizer_updates=0,physical_requests=0))
        write(R/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))


if __name__=='__main__':
    def interrupted(*args):raise InterruptedError('FINITE_OWNED_CPU_FOLLOWER_STOP')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);a=p.parse_args()
    result=run(a.spec);raise SystemExit(result['exit_code'])
