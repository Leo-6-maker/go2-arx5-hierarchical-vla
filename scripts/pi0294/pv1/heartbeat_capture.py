"""Same-job PV1 endurance using the unchanged post-read heartbeat collector."""
import json
import math
import os
from pathlib import Path
import signal
import socket
import threading
import time

from scripts.pi0292.allocation_admission import owned_jobs
from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1.native_heartbeat_capture import collect
from .allocation import single_h200_select,native_request_binding
from .decision import AUTHORITY
from .transport_fixture import immutable,namespace,BASE
from . import evaluation_platform as evaluation
from . import evaluation_core_identity as Core_native

FILENAME='PV1_NATIVE_HEARTBEAT_CAPTURE.json'
CLOCK='CLOCK_REALTIME_AFTER_EACH_READ_AND_CLOCK_MONOTONIC'


def identity(root,directory,allocation):
    root,directory=Path(root),Path(directory)
    if (root.is_symlink() or root.stat().st_uid!=os.getuid() or directory.is_symlink()
            or directory.parent.resolve()!=(root/'pbs').resolve()):
        raise PermissionError('PV1_EXACT_OWNED_NATIVE_CAPTURE_PATH')
    _,number,_=namespace(root,BASE+'-ALLOCATION-'+directory.name.removeprefix('alloc-'))
    m=immutable(root/'RUN_MANIFEST.json');r=immutable(directory/'REQUEST.json')
    if Core_native.cold_profile(m):
        if r.get('allocation_number')!=number or immutable(directory/'ALLOCATION_ADMISSION.json')!=allocation:
            raise PermissionError('PV1_CORE_SAME_SEALED_CAPTURE_ALLOCATION')
        native=Core_native.current_identity(root,m,r,allocation)
        return dict(job_id=native['native_handle'],hostname=native['hostname'],
            request_sha256=sha(directory/'REQUEST.json'),source_sha=m['code']['vla_commit'],
            manifest_sha256=sha(root/'RUN_MANIFEST.json'))
    if allocation.get('status')!=evaluation.STATUS and any('platform_evaluation_version' in x for x in (m,r,allocation)):
        raise PermissionError('PV1_EVALUATION_IDENTITY_CANNOT_FALL_BACK_TO_H200')
    if (directory.name!='alloc-'+str(number).zfill(2) or r['submit_count']!=number
            or r['run_uuid']!=root.name or r['source_sha']!=m['code']['vla_commit']
            or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or allocation['request_sha256']!=sha(directory/'REQUEST.json')
            or immutable(directory/'ALLOCATION_ADMISSION.json')!=allocation
            or allocation['status'] not in ('PASS_PV1_OWN_NATIVE_H200_ALLOCATION',evaluation.STATUS)
            or allocation['run_uuid']!=root.name or allocation['project']!=r['project']):
        raise PermissionError('PV1_SAME_SEALED_REQUEST_MANIFEST_NATIVE_ADMISSION')
    host=socket.gethostname();job_id=os.environ.get('PBS_JOBID');user=os.environ.get('USER')
    if (not job_id or job_id!=allocation['job_id'] or host!=allocation['hostname']
            or not allocation['allocation_start_epoch']<=time.time()<allocation['allocation_end_epoch']):
        raise PermissionError('PV1_PREMODEL_ONLY_ON_CURRENT_NATIVE_COMPUTE_ALLOCATION')
    for p in (directory/'PBS_PROCESS_FINAL.json',root/'pbs/MODEL_STATE.json'):
        if p.exists() and read(p).get('job_id')==job_id:
            raise PermissionError('PV1_ENDURANCE_MUST_PRECEDE_MODEL_AND_TERMINAL')
    if allocation['status']==evaluation.STATUS:
        evaluation.current_identity(m,r,allocation,run_uuid=root.name,
            manifest_sha256=sha(root/'RUN_MANIFEST.json'),request_sha256=sha(directory/'REQUEST.json'),
            user=user,hostname=host,job_id=job_id)
    else:
        j=owned_jobs(user).get(job_id)
        if (not j or j['job_state']!='R' or j['Job_Owner'].split('@')[0]!=user
            or j['project']!=r['project'] or int(j['Resource_List']['ngpus'])!=1
            or not single_h200_select(j['Resource_List'],r['ncpus'])
            or not native_request_binding(j,r)
            or j['Job_Name']!=r['job_name']
            or host.split('.')[0] not in {x.split('/')[0].split('.')[0] for x in j['exec_host'].split('+')}):
            raise PermissionError('PV1_FRESH_OWN_RUNNING_SELECTED_PROJECT_H200_REQUIRED')
    return dict(job_id=job_id,hostname=host,request_sha256=sha(directory/'REQUEST.json'),
        source_sha=m['code']['vla_commit'],manifest_sha256=sha(root/'RUN_MANIFEST.json'))


def capture(root,directory,allocation,guard,*,service=None):
    root,directory=Path(root),Path(directory);native=identity(root,directory,allocation)
    cutoff=allocation['allocation_end_epoch']-2700
    if time.time()+630>=cutoff:raise PermissionError('PV1_KEEP_NATIVE_CLOSEOUT_AFTER_600S_ENDURANCE')
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise PermissionError('PV1_NATIVE_CAPTURE_REQUIRES_OWN_MAIN_THREAD_TIMER')
    folder=directory/'heartbeat-capture';folder.mkdir(mode=0o700)
    owner=OwnerLock(folder/'CAPTURE.lock');started=time.process_time()
    def expired(signum,frame):raise TimeoutError('PV1_BOUNDED_NATIVE_CAPTURE_NO_RESTART')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,min(900,cutoff-time.time()))
    journal=folder/'READS.jsonl'
    def check():
        guard()
        if time.time()>=cutoff or sha(root/'RUN_MANIFEST.json')!=native['manifest_sha256']:
            raise PermissionError('PV1_ORIGINAL_NATIVE_CAPTURE_DEADLINE_AND_SOURCE')
    try:
        with journal.open('xb') as output:
            os.chmod(journal,0o600)
            def emit(row):
                output.write((json.dumps(row,allow_nan=False)+'\n').encode());output.flush();os.fsync(output.fileno())
            rows=collect(root,root.name,check,emit,duration_s=600,interval_s=7.5,service=service)
        identity(root,directory,allocation)
        v=dict(schema='PI0294_PV1_NATIVE_HEARTBEAT_CAPTURE_V1',status='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY',
            authority_release=AUTHORITY,run_uuid=root.name,**native,clock_source=CLOCK,rows=rows,
            raw_journal_relative=str(journal.relative_to(root)),raw_journal_sha256=sha(journal),
            CPU_s=time.process_time()-started,models_loaded=0,new_model_queries=0,new_physics_integrations=0,
            heartbeat_written=False,transport_qualified=False)
        path=directory/FILENAME
        if path.exists():raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_CAPTURE')
        write(path,v);path.chmod(0o400);journal.chmod(0o400)
        verify(root,directory,allocation)
        write(folder/'TERMINAL.json',dict(status=v['status'],fragment_sha256=sha(path),CPU_s=v['CPU_s']))
        return v
    except BaseException as exc:
        write(folder/'TERMINAL.json',dict(status='FAILED_NATIVE_CAPTURE_PRESERVED',
            error_type=type(exc).__name__,error=str(exc),CPU_s=time.process_time()-started,
            raw_journal_sha256=sha(journal) if journal.exists() else None,heartbeat_written=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);owner.close()


def verify(root,directory,allocation):
    root,directory=Path(root),Path(directory);path=directory/FILENAME
    if directory.is_symlink() or directory.parent.resolve()!=(root/'pbs').resolve():
        raise PermissionError('PV1_IMMUTABLE_SAME_JOB_CAPTURE_DIRECTORY')
    raw=path.read_bytes();v=immutable(path)
    from .mix_core_identity import STATUS as CD_CORE_STATUS
    if allocation.get('status') in (Core_native.STATUS, CD_CORE_STATUS):
        Core_native.sealed_identity(immutable(root/'RUN_MANIFEST.json'),immutable(directory/'REQUEST.json'),
            allocation,run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),
            request_sha256=sha(directory/'REQUEST.json'))
    elif allocation.get('status')==evaluation.STATUS:
        evaluation.sealed_identity(immutable(root/'RUN_MANIFEST.json'),immutable(directory/'REQUEST.json'),
            allocation,run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),
            request_sha256=sha(directory/'REQUEST.json'))
    if (v['schema']!='PI0294_PV1_NATIVE_HEARTBEAT_CAPTURE_V1'
            or v['status']!='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY' or v['authority_release']!=AUTHORITY
            or v['run_uuid']!=root.name or v['job_id']!=allocation['job_id'] or v['hostname']!=allocation['hostname']
            or v['manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or v['request_sha256']!=allocation['request_sha256']
            or v['source_sha']!=read(root/'RUN_MANIFEST.json')['code']['vla_commit']
            or v['models_loaded'] or v['new_model_queries'] or v['new_physics_integrations']
            or v['heartbeat_written'] is not False or v['transport_qualified'] is not False):
        raise PermissionError('PV1_REAL_IMMUTABLE_CURRENT_JOB_HEARTBEAT_FRAGMENT')
    journal=root/v['raw_journal_relative']
    if (journal.is_symlink() or journal.resolve()!=(directory/'heartbeat-capture/READS.jsonl').resolve()
            or journal.stat().st_mode&0o222 or sha(journal)!=v['raw_journal_sha256']
            or [json.loads(line) for line in journal.read_text().splitlines()]!=v['rows']):
        raise PermissionError('PV1_RAW_HEARTBEAT_JOURNAL_NOT_INLINE_ASSERTION')
    rows=v['rows']
    if (v['clock_source']!=CLOCK or len(rows)<81 or [r['sequence'] for r in rows]!=list(range(len(rows)))
            or rows[-1]['monotonic_s']-rows[0]['monotonic_s']<600
            or len({r['proof']['heartbeat']['observed_epoch'] for r in rows})<2):
        raise PermissionError('PV1_ACTUAL_600S_MULTIPLE_OWNER_UPDATES_REQUIRED')
    previous=None
    for r in rows:
        p=r['proof'];sampled=p['revalidated_clock_epoch'] if p['server_revalidation'] else p['cached_clock_epoch']
        t=r['monotonic_s']
        if (type(t) not in (int,float) or not math.isfinite(t)
                or p['clock_sampled_after_read'] is not True or p['heartbeat_written'] is not False
                or p['watchdog_s']!=300 or p['heartbeat']['run_uuid']!=root.name
                or not 0<=sampled-p['heartbeat']['observed_epoch']<=300
                or not allocation['allocation_start_epoch']<=sampled<=allocation['allocation_end_epoch']
                or previous is not None and not 0<t-previous<=300):
            raise PermissionError('PV1_UNCHANGED_POST_READ_WATCHDOG_AND_REAL_CLOCK')
        previous=t
    if path.read_bytes()!=raw:raise PermissionError('PV1_CAPTURE_CHANGED_DURING_READ')
    return v
