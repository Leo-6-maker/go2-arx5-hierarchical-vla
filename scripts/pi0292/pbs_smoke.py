"""One bounded own-allocation smoke chain; never submits or starts main training."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import time

from .durable import OwnerLock,read,write
from .data_consumer import owned_bytes
from .train_adapter import sha


def seconds(value):
    h,m,s=map(int,value.split(':'));return h*3600+m*60+s


def pbs_epoch(value):
    if isinstance(value,(int,float)) or str(value).isdigit():return float(value)
    return time.mktime(time.strptime(value,'%a %b %d %H:%M:%S %Y'))


def run(args):
    root=Path(args.root);job=os.environ.get('PBS_JOBID')
    if not job:raise PermissionError('PBS_JOBID_REQUIRED')
    request=read(args.request);owner=OwnerLock(root/'PBS_OWNER.lock')
    statefile=root/'evidence'/('PBS_SMOKE_'+request['request_id']+'.json')
    if statefile.exists():raise FileExistsError('Never rerun an allocation wrapper')
    adoption=read(root/'ADOPTION.json')
    if adoption['release']!='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65':raise PermissionError('FIXED_AUTHORITY_REQUIRED')
    if request['stage']!='smoke' or request['walltime_s']!=1800:raise PermissionError('S3_ONLY_30_MINUTE_ALLOCATION')
    os.environ['PATH']='/cm/shared/apps/pbspro/current/bin:'+os.environ.get('PATH','')
    info=json.loads(subprocess.check_output(['qstat','-f','-F','json',job],text=True,timeout=20))['Jobs'][job]
    host=socket.gethostname().split('.')[0]
    if info['job_state']!='R' or info['Job_Owner'].split('@')[0]!=os.environ['USER']:
        raise PermissionError('OWN_RUNNING_JOB_REQUIRED')
    if host not in {part.split('/')[0].split('.')[0] for part in info['exec_host'].split('+')}:
        raise PermissionError('ALLOCATED_COMPUTE_NODE_REQUIRED')
    if int(info['Resource_List']['ngpus'])!=1 or seconds(info['Resource_List']['walltime'])!=1800:
        raise PermissionError('ONE_GPU_30_MINUTE_RESOURCE_REQUEST_REQUIRED')
    start=pbs_epoch(info['stime']);end=start+1800
    if time.time()>=min(adoption['new_work_cutoff_epoch'],end-300):raise TimeoutError('SMOKE_ALLOCATION_TOO_LATE')
    for _ in range(20):
        ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
        entry=next(r for r in ledger['PBS_submissions'] if r['request_id']==request['request_id'])
        if entry.get('job_id')==job:break
        time.sleep(.5)
    else:raise PermissionError('SUBMISSION_RECEIPT_MUST_BIND_EXACT_JOB_ID')
    if entry['status']!='SUBMITTED' or len(ledger['PBS_submissions'])>3:raise PermissionError('SUBMISSION_STATE_OR_BUDGET')
    prior=ledger['GPU_allocated_s']
    if prior+1800>8*3600:raise PermissionError('TOTAL_ALLOCATION_BUDGET')
    if read(root/'evidence/HOPPER_DATA_QUALIFICATION.json')['status']!='PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE':
        raise PermissionError('DATA_GATE_REQUIRED_BEFORE_MODEL')
    operations=Path(request['operations'])
    seal=read(operations/'OPERATIONS_SEAL.json')
    if seal['source_sha']!=request['source_sha']:raise ValueError('REQUEST_SOURCE_CHANGED')
    for name,digest in seal['files'].items():
        if sha(operations/name)!=digest:raise ValueError('COMMITTED_OPERATIONS_CHANGED')
    if owned_bytes(root)+request['predicted_remaining_peak_bytes']>180*1024**3:
        raise PermissionError('ALLOCATION_PEAK_SPACE_BUDGET')
    if shutil.disk_usage(root).free<request['predicted_remaining_peak_bytes']+20*1024**3:
        raise PermissionError('ALLOCATION_FREE_SPACE_MARGIN')
    write(root/'NEURAL_AUTHORIZATION.json',dict(job_id=job,allowed_modes=['smoke-continuous','smoke-first','smoke-resume'],
        allocation_end_epoch=end,allocation_walltime_s=1800,prior_GPU_allocated_s=prior,
        predicted_remaining_peak_bytes=request['predicted_remaining_peak_bytes'],source_sha=request['source_sha']))
    entry.update(status='RUNNING',allocation_start_epoch=start,allocation_end_epoch=end,host=host)
    write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
    state=dict(status='RUNNING',job_id=job,request_id=request['request_id'],allocation_start_epoch=start,
        allocation_end_epoch=end,steps=[],current=None,pid=os.getpid(),source_sha=request['source_sha'])
    write(statefile,state)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    env=dict(os.environ,PYTHONPATH=str(operations)+':'+str(root/'data/bank_v2/frozen_source/src'),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    child=None
    def terminate(signum,frame):raise TimeoutError('OWN_PBS_CLOSEOUT_SIGNAL')
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGALRM,terminate)
    signal.alarm(max(1,int(min(end-30,adoption['new_work_cutoff_epoch'])-time.time())))
    try:
        for mode in ('smoke-continuous','smoke-first','smoke-resume','compare'):
            if min(end,adoption['new_work_cutoff_epoch'])-time.time()<180:raise TimeoutError('RESERVE_SMOKE_CLOSEOUT')
            state['current']=mode;write(statefile,state)
            command=[request['python'],'-m','scripts.pi0292.compare_smoke','--root',str(root)] if mode=='compare' else [
                request['python'],'-m','scripts.pi0292.neural_train','--root',str(root),'--bindings',request['bindings'],'--mode',mode]
            began=time.time()
            with (root/'logs'/(request['request_id']+'_'+mode+'.stdout')).open('xb') as out,(root/'logs'/(request['request_id']+'_'+mode+'.stderr')).open('xb') as err:
                child=subprocess.Popen(command,cwd=operations,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(child_pid=child.pid,child_start=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]);write(statefile,state)
                code=child.wait(timeout=max(1,min(end-45,adoption['new_work_cutoff_epoch'])-time.time()))
            child=None
            state['steps'].append(dict(mode=mode,exit=code,wall_s=time.time()-began));write(statefile,state)
            if code:raise RuntimeError('SMOKE_CHILD_FAILED:'+mode)
        gate=read(root/'evidence/MODEL_SMOKE.json')
        if gate['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED':raise ValueError('ACTUAL_MODEL_GATE_NOT_PASSED')
        state.update(status='SMOKE_COMPLETE_NO_MAIN_STARTED',current=None,child_pid=None)
    except BaseException as error:
        state.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
        state.update(finished=time.time(),CPU_s=cpu,allocated_observed_s=time.time()-start)
        write(statefile,state)
        ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');entry=next(r for r in ledger['PBS_submissions'] if r['request_id']==request['request_id'])
        entry.update(status='WRAPPER_TERMINAL_SCHEDULER_RECONCILIATION_PENDING',wrapper_status=state['status'],
            CPU_s=cpu,CPU_reserved_s=0,allocated_observed_s=state['allocated_observed_s'],
            GPU_reserved_s=1800)
        ledger['CPU_s']+=cpu;write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
        owner.close()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    result=run(parser.parse_args());print(json.dumps({k:result[k] for k in ('status','job_id','CPU_s')}))
