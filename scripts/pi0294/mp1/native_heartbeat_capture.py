"""Capture native heartbeat evidence, never manufacture the full N1 admission."""
import json
import math
import os
from pathlib import Path
import socket
import signal
import threading
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from .n1_limits import RELEASE,resolve
from .owner_heartbeat import read_live_owner

FILENAME='N1_NATIVE_HEARTBEAT_CAPTURE.json'
DURATION_S=600
INTERVAL_S=7.5
MAXIMUM_WALL_S=900


def collect(root,run_uuid,guard,emit,*,duration_s=DURATION_S,interval_s=INTERVAL_S,
            reader=read_live_owner,clock=time.monotonic,sleep=time.sleep):
    """The injectable clock is for CPU tests; the native entry fixes both limits."""
    if (type(duration_s) not in [int,float] or not math.isfinite(duration_s) or duration_s<=0
            or type(interval_s) not in [int,float] or not math.isfinite(interval_s)
            or not 0<interval_s<=7.5):
        raise ValueError('FINITE_HEARTBEAT_CAPTURE_SCHEDULE')
    rows=[];first=None
    while True:
        guard()
        before=clock()
        proof=reader(root,run_uuid,force_revalidate=len(rows)%8==0)
        after=clock()
        if not all(type(x) in [int,float] and math.isfinite(x) for x in [before,after]):
            raise ValueError('FINITE_REAL_MONOTONIC_HEARTBEAT_CLOCK')
        if after<before or rows and not 0<after-rows[-1]['monotonic_s']<=300:
            raise PermissionError('MONOTONIC_CLOCK_OR_HEARTBEAT_SAMPLING_GAP')
        sampled=proof['revalidated_clock_epoch'] if proof['server_revalidation'] else proof['cached_clock_epoch']
        if (proof['clock_sampled_after_read'] is not True or proof['heartbeat_written'] is not False
                or proof['watchdog_s']!=300 or proof['heartbeat']['run_uuid']!=run_uuid
                or not 0<=sampled-proof['heartbeat']['observed_epoch']<=300):
            raise PermissionError('ORIGINAL_POST_READ_OWNER_HEARTBEAT_REQUIRED')
        guard()
        row=dict(sequence=len(rows),monotonic_s=after,read_duration_s=after-before,proof=proof)
        emit(row);rows.append(row)
        if first is None:first=after
        if after-first>=duration_s:break
        sleep(min(interval_s,duration_s-(after-first)))
    return rows


def native_identity(root,directory,allocation,manifest):
    """A supplied allocation dictionary alone is not proof of native execution."""
    from scripts.pi0292.allocation_admission import owned_jobs
    root=Path(root);directory=Path(directory)
    if (not directory.is_dir() or directory.is_symlink() or root.is_symlink()
            or directory.parent.resolve()!=(root/'pbs').resolve()
            or directory.name not in ['alloc-'+str(i).zfill(2) for i in range(1,13)]
            or root.stat().st_uid!=os.getuid()):
        raise PermissionError('EXACT_OWNED_NATIVE_ALLOCATION_CAPTURE_PATH')
    request=read(directory/'REQUEST.json')
    number=int(directory.name.removeprefix('alloc-'))
    if (request['submit_count']!=number or request['run_uuid']!=root.name
            or request['source_sha']!=manifest['code']['vla_commit']
            or request['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or allocation['request_sha256']!=sha(directory/'REQUEST.json')
            or read(directory/'ALLOCATION_ADMISSION.json')!=allocation):
        raise PermissionError('EXACT_SAME_REQUEST_MANIFEST_AND_NATIVE_ADMISSION')
    hostname=socket.gethostname();job_id=os.environ.get('PBS_JOBID');user=os.environ.get('USER')
    if not job_id or job_id!=allocation['job_id'] or hostname!=allocation['hostname']:
        raise PermissionError('NO_LOGIN_OR_FOREIGN_NATIVE_HEARTBEAT_CAPTURE')
    if (allocation['status']!='PASS_MP1_OWN_SOURCE_PBS'
            or not allocation['allocation_start_epoch']<=time.time()<allocation['allocation_end_epoch']):
        raise PermissionError('UNEXPIRED_ORIGINAL_NATIVE_ALLOCATION_ONLY')
    for path in [directory/'PBS_PROCESS_FINAL.json',root/'pbs/MODEL_STATE.json']:
        if path.exists() and read(path).get('job_id')==job_id:
            raise PermissionError('NATIVE_CAPTURE_MUST_PRECEDE_MODEL_AND_TERMINAL')
    job=owned_jobs(user).get(job_id)
    if (not job or job['job_state']!='R' or job['project']!='CFP04-CF-007'
            or job['Job_Owner'].split('@')[0]!=user or int(job['Resource_List']['ngpus'])!=1
            or hostname.split('.')[0] not in {x.split('/')[0].split('.')[0] for x in job['exec_host'].split('+')}):
        raise PermissionError('FRESH_OWN_NATIVE_H200_PROJECT_JOB_REQUIRED')
    return dict(job_id=job_id,hostname=hostname,request_sha256=sha(directory/'REQUEST.json'),
                manifest_sha256=sha(root/'RUN_MANIFEST.json'),source_sha=manifest['code']['vla_commit'])


def capture(root,directory,allocation,guard):
    """Called only before a model load; no heartbeat writer or model imports."""
    root=Path(root);directory=Path(directory);manifest=read(root/'RUN_MANIFEST.json')
    n1=resolve(root,manifest)
    if n1 is None:raise PermissionError('EXPLICIT_SEALED_N1_CAPTURE_ONLY')
    identity=native_identity(root,directory,allocation,manifest)
    cutoff=min(allocation['allocation_end_epoch']-2700,n1['new_experiment_cutoff_epoch'])
    if time.time()+DURATION_S+30>=cutoff:
        raise PermissionError('KEEP_ORIGINAL_ALLOCATION_CLOSEOUT_AFTER_ENDURANCE')
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise PermissionError('NATIVE_CAPTURE_REQUIRES_OWN_MAIN_THREAD_TIMER')
    # The enclosing neural worker owns CPU accounting. This extra lock owns only
    # this evidence fragment; it cannot take over the scientific owner lease.
    folder=directory/'heartbeat-capture';folder.mkdir(mode=0o700)
    owner=OwnerLock(folder/'CAPTURE.lock');started=time.process_time()
    def expire(signum,frame):raise TimeoutError('BOUNDED_NATIVE_HEARTBEAT_CAPTURE_NO_AUTOMATIC_RESTART')
    previous_handler=signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,min(MAXIMUM_WALL_S,max(.001,cutoff-time.time())))
    journal=folder/'READS.jsonl';terminal=folder/'TERMINAL.json'
    def check():
        guard()
        if time.time()>=cutoff:raise TimeoutError('N1_CAPTURE_ORIGINAL_CLOSEOUT')
        if sha(root/'RUN_MANIFEST.json')!=identity['manifest_sha256']:
            raise PermissionError('N1_CAPTURE_MANIFEST_CHANGED_NO_HOT_REBIND')
    try:
        with journal.open('xb') as output:
            os.chmod(journal,0o600)
            def emit(row):
                output.write((json.dumps(row,allow_nan=False)+'\n').encode());output.flush();os.fsync(output.fileno())
            rows=collect(root,root.name,check,emit,duration_s=DURATION_S,interval_s=INTERVAL_S)
        native_identity(root,directory,allocation,manifest)
        if len(rows)<81 or len({r['proof']['heartbeat']['observed_epoch'] for r in rows})<2:
            raise PermissionError('REAL_600S_OWNER_UPDATES_NOT_REPEATED_SINGLE_SMOKE')
        value=dict(schema='MP1_N1_NATIVE_HEARTBEAT_CAPTURE_V1',status='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY',
            authority_release=RELEASE,run_uuid=root.name,**identity,
            clock_source='CLOCK_REALTIME_AFTER_EACH_READ_AND_CLOCK_MONOTONIC',
            rows=rows,raw_journal_relative=str(journal.relative_to(root)),raw_journal_sha256=sha(journal),
            CPU_s=time.process_time()-started,models_loaded=0,new_model_queries=0,new_physics_integrations=0,
            heartbeat_written=False,transport_qualified=False,full_N1_qualification=False)
        path=directory/FILENAME
        if path.exists():raise FileExistsError('PRESERVE_EXISTING_NATIVE_HEARTBEAT_CAPTURE')
        write(path,value);path.chmod(0o400);journal.chmod(0o400)
        write(terminal,dict(status=value['status'],fragment_sha256=sha(path),CPU_s=value['CPU_s']))
        terminal.chmod(0o400)
        return value
    except BaseException as exc:
        write(terminal,dict(status='FAILED_NATIVE_CAPTURE_PRESERVED_NO_AUTOMATIC_RESTART',
            error_type=type(exc).__name__,error=str(exc),CPU_s=time.process_time()-started,
            raw_journal_sha256=sha(journal) if journal.exists() else None,heartbeat_written=False,
            models_loaded=0,new_model_queries=0,new_physics_integrations=0))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler);owner.close()


def verify(root,directory,allocation):
    """Only join an immutable, same-job fragment with its actual raw journal."""
    root=Path(root);directory=Path(directory);path=directory/FILENAME
    if (path.is_symlink() or directory.is_symlink() or path.stat().st_mode&0o222
            or directory.parent.resolve()!=(root/'pbs').resolve()):
        raise PermissionError('IMMUTABLE_SAME_JOB_NATIVE_HEARTBEAT_FRAGMENT_ONLY')
    raw=path.read_bytes();v=read(path)
    if (v['schema']!='MP1_N1_NATIVE_HEARTBEAT_CAPTURE_V1'
            or v['status']!='PASS_NATIVE_HEARTBEAT_FRAGMENT_ONLY' or v['authority_release']!=RELEASE
            or v['run_uuid']!=root.name or v['job_id']!=allocation['job_id'] or v['hostname']!=allocation['hostname']
            or v['manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or v['request_sha256']!=allocation['request_sha256']
            or v['source_sha']!=read(root/'RUN_MANIFEST.json')['code']['vla_commit']
            or v['models_loaded'] or v['new_model_queries'] or v['new_physics_integrations']
            or v['heartbeat_written'] is not False or v['transport_qualified'] is not False
            or v['full_N1_qualification'] is not False):
        raise PermissionError('IMMUTABLE_SAME_JOB_NATIVE_HEARTBEAT_FRAGMENT_ONLY')
    journal=root/v['raw_journal_relative']
    if (journal.is_symlink() or journal.resolve()!=(directory/'heartbeat-capture/READS.jsonl').resolve()
            or journal.stat().st_mode&0o222 or sha(journal)!=v['raw_journal_sha256']
            or [json.loads(line) for line in journal.read_text().splitlines()]!=v['rows']):
        raise PermissionError('NATIVE_HEARTBEAT_ROWS_MUST_EQUAL_SEALED_RAW_JOURNAL')
    rows=v['rows']
    if (v['clock_source']!='CLOCK_REALTIME_AFTER_EACH_READ_AND_CLOCK_MONOTONIC'
            or len(rows)<81 or [row['sequence'] for row in rows]!=list(range(len(rows)))
            or rows[-1]['monotonic_s']-rows[0]['monotonic_s']<600
            or len({row['proof']['heartbeat']['observed_epoch'] for row in rows})<2):
        raise PermissionError('NATIVE_600S_RAW_HEARTBEAT_SAMPLING_REQUIRED')
    previous=None
    for row in rows:
        monotonic=row['monotonic_s'];proof=row['proof']
        sampled=proof['revalidated_clock_epoch'] if proof['server_revalidation'] else proof['cached_clock_epoch']
        if (type(monotonic) not in [int,float] or not math.isfinite(monotonic)
                or proof['clock_sampled_after_read'] is not True or proof['heartbeat_written'] is not False
                or proof['watchdog_s']!=300 or proof['heartbeat']['run_uuid']!=root.name
                or not 0<=sampled-proof['heartbeat']['observed_epoch']<=300
                or not allocation['allocation_start_epoch']<=sampled<=allocation['allocation_end_epoch']
                or previous is not None and not 0<monotonic-previous<=300):
            raise PermissionError('RAW_NATIVE_HEARTBEAT_CLOCK_RUN_AND_ORIGINAL_WATCHDOG')
        previous=monotonic
    if path.read_bytes()!=raw:raise PermissionError('NATIVE_HEARTBEAT_FRAGMENT_CHANGED_DURING_READ')
    return v
