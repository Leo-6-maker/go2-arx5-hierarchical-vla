"""One finite PBS allocation: raw native10, raw inference, fresh shared8192.

Uses the existing native relay registration/onceIO/account and workers. PBS is
the scheduler. This adapter never submits another job or contacts Core, changes
credentials, performs physics, or restarts a failed scientific producer.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import statistics

from scripts.pi0294.night10 import native_relay as relay
from .shared_site import project_quota,validate_vanda,PROJECT_SCRATCH


def prepare_inputs(c,guard):
    """Extract this fixed private input archive on the allocated compute node."""
    destination=Path(c['input_root']);ready=destination/'TRANSFER_COMPLETE.json'
    if ready.exists():
        v=relay.read(ready)
        if v.get('archive_sha256')!=c['input_archive_sha256']:
            raise PermissionError('OTHER_FIXED_SHARED_INPUTS_ALREADY_PRESENT')
        return v
    destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    archive=Path(c['input_archive']);digest=hashlib.sha256();count=total=0
    class Reader:
        def __init__(self,stream):self.stream=stream
        def read(self,n=-1):
            body=self.stream.read(n);digest.update(body);return body
    with archive.open('rb')as stream:
        reader=Reader(stream)
        with tarfile.open(fileobj=reader,mode='r|')as tar:
            for member in tar:
                guard();path=destination/member.name
                if (not member.isfile()or Path(member.name).is_absolute()
                        or not path.resolve().is_relative_to(destination.resolve())):
                    raise PermissionError('ONLY_EXPECTED_PRIVATE_DATA_REGULAR_MEMBERS')
                path.parent.mkdir(parents=True,exist_ok=True)
                source=tar.extractfile(member);remaining=member.size
                with path.open('xb')as out:
                    while remaining:
                        chunk=source.read(min(1024**2,remaining))
                        if not chunk:raise EOFError('INCOMPLETE_PRIVATE_INPUT_MEMBER')
                        out.write(chunk);remaining-=len(chunk)
                count+=1;total+=member.size
        while reader.read(4*1024**2):guard()
    if digest.hexdigest()!=c['input_archive_sha256']or archive.stat().st_size!=c['input_archive_bytes']:
        raise PermissionError('PRIVATE_INPUT_TRANSFER_ARCHIVE_CHANGED')
    pins=dict(zip(('DATASET_MANIFEST.json','statistics.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json'),
        (c['shared_recipe_pins'][k]for k in('manifest_sha256','statistics_sha256','population_sha256','task_DEV_sha256'))))
    if any(relay.sha(destination/'data'/n)!=pin for n,pin in pins.items()):
        raise PermissionError('FROZEN59_NUMERIC_STATISTICS_AND_DEV_BYTES_CHANGED')
    v=dict(archive_sha256=digest.hexdigest(),files=count,payload_bytes=total,
        dataset_pins=pins,epoch=time.time(),new_data_or_statistics=False)
    relay.exclusive(ready,v)
    return v


def job():
    key=os.environ.get('PBS_JOBID')
    if not key:raise PermissionError('VANDA_ALLOCATED_JOB_REQUIRED')
    q=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',key],
        capture_output=True,text=True,check=True,timeout=20)
    rows=[v for k,v in json.loads(q.stdout)['Jobs'].items()if k.split('.')[0]==key.split('.')[0]]
    if len(rows)!=1:raise PermissionError('EXACT_SINGLE_CURRENT_PBS_ID')
    return key,rows[0]


def seconds(value):
    hours,minutes,secs=map(int,value.split(':'))
    return hours*3600+minutes*60+secs


def context(c):
    key,j=job();now=time.time()
    quota_path=Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
    quota=project_quota(quota_path.read_text(),user=os.environ['USER'],
        report_epoch=quota_path.stat().st_mtime,observed_epoch=now)
    duration=seconds(j['Resource_List']['walltime'])
    used=seconds(j.get('resources_used',{}).get('walltime','00:00:00'))
    v=dict(schema='SHARED_RAW_VANDA_A40_EXECUTION_V1',job_id=key,owner_uid=os.getuid(),
        owner_user=os.environ['USER'],science_run_id=c['science_run_id'],
        submission_epoch=c['submission_epoch'],submission_nonce=c['submission_nonce'],
        authorization_mode='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION',
        project_receipt=c['project_receipt'],quota_receipt=quota,
        predicted_new_peak_bytes=c['predicted_new_peak_bytes'],
        allocation_deadline_epoch=min(c['allocation_deadline_epoch'],now+duration-used-120),
        storage_roots=c['storage_roots'])
    validate_vanda(v,j,now=now,uid=os.getuid(),user=os.environ['USER'],
        hostname=os.uname().nodename,job_id=key)
    return v


def admission(c):
    if (c.get('schema')!='SHARED_M32_VANDA_SINGLE_ALLOCATION_V1'
            or c.get('execution_site')!='VANDA_A40' or os.getuid()!=22182
            or time.time()>=c['dispatch_deadline_epoch']
            or not str(Path(c['root']).resolve()).startswith(PROJECT_SCRATCH+'/')
            or c.get('resume_checkpoint') is not None):
        raise PermissionError('FRESH_FINITE_SHARED_M32_VANDA_CHAIN')
    handoff=Path(c['owner_handoff'])
    if relay.sha(handoff)!=c['owner_handoff_sha256']:
        raise PermissionError('PINNED_CORE_COLD_HANDOFF')
    h=relay.read(handoff)
    owner=relay.read(c['LOCAL']).get('V7_shared_raw_m32_owner',{})
    if (h.get('science_run_id')!=c['science_run_id'] or h.get('Core_shared_producer_closed')is not True
            or h.get('Core_shared_nodes_disabled')!=['RAW_NATIVE10','RAW_NATIVE16','SHARED8192_ADOPT']
            or owner.get('site')!='VANDA_A40' or owner.get('science_run_id')!=c['science_run_id']
            or owner.get('status')!='REGISTERED_UNIQUE_BASELINE' or owner.get('root')!=c['root']
            or owner.get('submission_nonce')!=c['submission_nonce']
            or owner.get('owner_handoff_sha256')!=c['owner_handoff_sha256']):
        raise PermissionError('ONE_REGISTERED_SCIENTIFIC_PRODUCER_AFTER_CORE_ACK')
    for name,pin in c['source_files'].items():
        if relay.sha(Path(c['source'])/name)!=pin:
            raise PermissionError('PINNED_VANDA_RUNTIME_SOURCE:'+name)
    return context(c)


def environment(root,c):
    env=os.environ.copy()
    env.update(HOME=str(root/'home'),TMPDIR=str(root/'tmp'),HF_HOME=str(root/'cache/hf'),
        XDG_CACHE_HOME=str(root/'cache/xdg'),CUDA_CACHE_PATH=str(root/'cache/cuda'),
        CUDA_VISIBLE_DEVICES=c['GPU_uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    return env


def formal_time_admission(c,native_root,*,now=None):
    """Fit the complete finite job using measured native10, never fewer steps."""
    now=time.time()if now is None else now
    native_root=Path(native_root)
    rows=[json.loads(s)for s in(native_root/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]
    if [r['step']for r in rows]!=list(range(1,11)):
        raise PermissionError('ACTUAL_TEN_CONTINUOUS_NATIVE_UPDATES_REQUIRED')
    intervals=[b['epoch']-a['epoch']for a,b in zip(rows[1:],rows[2:])]
    if any(not 0<v<3600 for v in intervals):
        raise PermissionError('FINITE_NATIVE_UPDATE_MEASUREMENTS')
    result=relay.read(native_root/'RESULT.json')
    # The native result includes raw load, both diagnostics, checkpoint and
    # full reload. Count that entire overhead for each of four formal saves.
    overhead=max(0,result['wall_s']-(rows[-1]['epoch']-rows[0]['epoch']))
    predicted=8192*max(intervals)*1.25+4*overhead+c['export_and_close_reserve_s']
    remaining=c['site_context']['allocation_deadline_epoch']-now
    report=dict(schema='SHARED8192_NATIVE10_MEASURED_TIME_ADMISSION_V1',
        stable_median_s_per_update=statistics.median(intervals),
        stable_max_s_per_update=max(intervals),native_full_overhead_s=overhead,
        forecast_complete8192_s=predicted,remaining_allocation_s=remaining,
        four_checkpoint_save_reload_pairs=True,export_and_close_reserve_s=c['export_and_close_reserve_s'],
        passes=remaining>=predicted and c['formal_seconds']>=predicted,
        qualification_weights_or_optimizer_inherited=False)
    relay.exclusive(Path(c['root'])/'FORMAL_TIME_ADMISSION.json',report)
    if not report['passes']:raise PermissionError('PBS_REMAINING_TIME_CANNOT_CONTAIN_FULL8192_AND_CLOSE')
    return report


def execute(c,root):
    """No retry: persist one launch, wait for its real result, onceIO close."""
    root=Path(root);manifest=relay.read(root/'RUN_MANIFEST.json')
    relay.exclusive(root/'LAUNCH_INTENT.json',dict(
        manifest_sha256=relay.sha(root/'RUN_MANIFEST.json'),PBS_JOBID=os.environ['PBS_JOBID'],
        epoch=time.time(),automatic_retry=False))
    start=time.monotonic()
    with (root/'stdout.log').open('xb')as out,(root/'stderr.log').open('xb')as err:
        child=subprocess.Popen([c['python'],'-B',str(root/'worker.py'),'--root',str(root),
            '--manifest-sha256',relay.sha(root/'RUN_MANIFEST.json')],
            env=environment(root,c),cwd=root,stdout=out,stderr=err,start_new_session=True)
        identity=dict(PID=child.pid,proc_start=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            PBS_JOBID=os.environ['PBS_JOBID'],epoch=time.time())
        relay.exclusive(root/'ACTUAL_STARTED.json',identity)
        while child.poll()is None:
            relay.update(Path(c['root'])/'STATE.json',dict(
                status='RUNNING',actor=str(root),**identity,heartbeat_epoch=time.time(),
                progress=relay.read(root/'FIT_PROGRESS.json')if(root/'FIT_PROGRESS.json').exists()else None))
            if time.time()>manifest['deadline_epoch']+30:
                os.killpg(child.pid,signal.SIGTERM)
                child.wait(timeout=30)
                break
            time.sleep(10)
    processes=subprocess.check_output(['ps','-eo','pid,pgid'],text=True,timeout=10)
    if any(len(parts:=line.split())==2 and parts[1]==str(child.pid)for line in processes.splitlines()[1:]):
        raise PermissionError('UNKNOWN_REMAINING_NATIVE_PROCESS_GROUP')
    result=relay.read(root/'RESULT.json');counts=relay.audit_terminal(root,manifest,result)
    counts.update(result_sha256=relay.sha(root/'RESULT.json'),CPU_s=result['CPU_s'],
        GPU_allocated_s_upper=int(time.monotonic()-start)+1,TRAIN_window_visits=result.get('TRAIN_visits',0),
        OS_returncode=child.returncode,all_costs_retained=True)
    account,before=relay.snapshot(c['LOCAL'])
    row=account['V7_book_learning_runs'][root.name]
    if row['status']!='REGISTERED_NOT_STARTED'or row['manifest_sha256']!=relay.sha(root/'RUN_MANIFEST.json'):
        raise PermissionError('UNKNOWN_CANONICAL_VANDA_NATIVE_TRANSITION')
    row.update(status='NATIVE_TERMINAL',actual=counts,closed_epoch=time.time())
    relay.once(c['LOCAL'],account,root.name+'-TERMINAL',expected_before_sha256=before)
    lease,before=relay.snapshot(root/'control/LEASE.json')
    lease.update(status='CLOSED',result_sha256=relay.sha(root/'RESULT.json'),closed_epoch=time.time())
    relay.once(root/'control/LEASE.json',lease,root.name+'-LEASE-CLOSE',expected_before_sha256=before)
    relay.exclusive(root/'RELAY_CLOSED.json',dict(
        effects_closed=True,native_status=result['status'],actual=counts,producer_run_uuid=root.name))
    if child.returncode!=0 or result.get('error')is not None:
        raise RuntimeError('CLOSED_NATIVE_WORKER_FAILED:'+str(result.get('error')))
    return result


def run(c):
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0294.generalization.minimal_model import audit_raw_parent
    from .shared_training import qualified_entry
    from .shared_policy import qualified_inference_entry
    from .shared_evaluation import KIND,LIMITS
    root=Path(c['root']);lock=OwnerLock(root/'OWNER.lock')
    start=time.monotonic();cpu=time.process_time();error=None;status='UNKNOWN'
    try:
        c['site_context']=admission(c)
        relay.exclusive(root/'ACTUAL_ALLOCATION_STARTED.json',dict(
            PBS_JOBID=os.environ['PBS_JOBID'],PID=os.getpid(),
            proc_start=Path('/proc/self/stat').read_text().split()[21],
            hostname=os.uname().nodename,site_context=c['site_context'],epoch=time.time()))
        def guard():
            if time.time()>=c['site_context']['allocation_deadline_epoch']:
                raise TimeoutError('FINITE_SAME_PBS_ALLOCATION')
            fs=os.statvfs(root)
            if fs.f_bavail*fs.f_frsize<c['protected_remaining_bytes']+c['predicted_new_peak_bytes']+30*1024**3:
                raise PermissionError('VANDA_ACTUAL_PROJECT_STORAGE_AND_EXISTING_PROTECTION')
        guard()
        prepare_inputs(c,guard)
        # The probe is a short subprocess: its CUDA context exits before any
        # native worker. The original PBS device visibility is retained.
        probe="import json,torch;print(json.dumps(dict(count=torch.cuda.device_count(),name=torch.cuda.get_device_name(0),uuid=str(torch.cuda.get_device_properties(0).uuid))))"
        q=subprocess.run([c['python'],'-c',probe],capture_output=True,text=True,check=True,timeout=60)
        gpu=json.loads(q.stdout)
        if gpu['count']!=1 or gpu['name']!='NVIDIA A40':
            raise PermissionError('EXACT_PBS_VISIBLE_SINGLE_A40')
        c['GPU_uuid']=gpu['uuid']
        relay.exclusive(root/'ACTUAL_GPU.json',gpu)
        # Full immutable input validation happens on the allocated compute
        # node, not on the login host.
        audit=audit_raw_parent(c['raw_parent'],guard=guard)
        relay.exclusive(root/'RAW_PARENT_AUDIT.json',audit)
        c['raw_audit']=str(root/'RAW_PARENT_AUDIT.json')
        for action in ('native10','native16','formal8192'):
            if time.time()>=c['dispatch_deadline_epoch']:
                raise TimeoutError('NO_NEW_WORK_AFTER_APPROVED_DISPATCH_CUTOFF')
            c['site_context']=context(c);guard()
            actor=relay.shared_inference(c)if action=='native16'else relay.shared(c,10 if action=='native10'else 8192)
            execute(c,actor)
            if action=='native10':
                from .shared_training import recipe
                preview=relay.read(actor/'RUN_MANIFEST.json')
                p=recipe(**c['shared_recipe_pins'])
                preview.update(recipe=p,GPU_training_qualification=None,
                    GPU_training_admission=relay.shared_training_certificate(c),
                    limits=dict(model_loads=1,optimizer_updates=8192,physics_integrations=0,
                        generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0))
                qualified_entry(preview)
            elif action=='native16':
                names=('RUN_MANIFEST.json','RESULT.json','SHARED_GPU_INFERENCE_CAPACITY.json',
                    'ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json','control/LEASE.json')
                preview=relay.read(actor/'RUN_MANIFEST.json')
                preview.update(kind=KIND,limits=LIMITS,shared_inference_admission=dict(
                    schema='SHARED4_SAME_GPU_NATIVE16_INFERENCE_V1',qualification_root=str(actor),
                    files={n:relay.sha(actor/n)for n in names}))
                qualified_inference_entry(preview)
                formal_time_admission(c,Path(c['run_parent'])/c['shared_native10_name'])
        from .shared_delivery import export
        delivered=export(actor,Path(c['root'])/'inference8192',guard=guard)
        relay.exclusive(root/'INFERENCE_EXPORT_COMPLETE.json',delivered)
        status='COMPLETE_SHARED8192_AND_NATIVE_QUALIFICATIONS'
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        status='FAILED_KNOWN_CLOSED'if all(
            (Path(a)/'control/LEASE.json').is_file()
            and relay.read(Path(a)/'control/LEASE.json').get('status')=='CLOSED'
            for a in c.get('_actors',[]))else'UNKNOWN_NATIVE_EFFECT'
        raise
    finally:
        formal=Path(c['run_parent'])/c['shared_formal_name']
        progress=relay.read(formal/'FIT_PROGRESS.json')if(formal/'FIT_PROGRESS.json').is_file()else None
        final=dict(status=status,error=error,science_run_id=c['science_run_id'],
            PBS_JOBID=os.environ.get('PBS_JOBID'),actors=c.get('_actors',[]),formal_progress=progress,
            allocated_GPU_wall_s_upper=int(time.monotonic()-start)+1,
            controller_CPU_s=time.process_time()-cpu,ended_epoch=time.time(),automatic_retry=False,
            actual_PBS_terminal_independent_confirmation_pending=True)
        relay.exclusive(root/'ALLOCATION_NATIVE_FINAL.json',final)
        relay.update(root/'STATE.json',final)
        account,before=relay.snapshot(c['LOCAL'])
        owner=account['V7_shared_raw_m32_owner']
        if owner.get('root')!=c['root']or owner.get('submission_nonce')!=c['submission_nonce']:
            raise PermissionError('PRESERVE_ANOTHER_CANONICAL_OWNER')
        owner.update(status='UNKNOWN_NATIVE_EFFECT'if status=='UNKNOWN_NATIVE_EFFECT'else'NATIVE_CLOSED_PENDING_PBS_TERMINAL',
            native_final_sha256=relay.sha(root/'ALLOCATION_NATIVE_FINAL.json'),native_status=status)
        relay.once(c['LOCAL'],account,root.name+'-NATIVE-FINAL',expected_before_sha256=before)
        lock.close()
    # No GPU/model runtime depends on another station or authentication.
    # Output publication is a separate CPU boundary after native close.
    from .shared_delivery import publish_hf
    publish_hf(c,delivered)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    a=p.parse_args();path=Path(a.config)
    if relay.sha(path)!=a.config_sha256:raise PermissionError('SEALED_VANDA_RUN_CONFIGURATION')
    c=relay.read(path);c['_config_sha256']=a.config_sha256
    run(c)


if __name__=='__main__':main()
