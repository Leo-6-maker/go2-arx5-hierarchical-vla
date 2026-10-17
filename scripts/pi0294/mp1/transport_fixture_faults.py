"""Native isolated faults, never writes the production owner heartbeat."""
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.rpc_metrics import SessionMetrics
from scripts.pi0294.relay_endpoint import inside
from .owner_heartbeat import HeartbeatLost,read_live_owner
from .read_only_metadata import read_only


def start_exit_heartbeat(folder,run):
    parent=inside(Path(folder),'faults');parent.mkdir(mode=0o700)
    path=parent/'actual_expired_heartbeat';path.mkdir(mode=0o700)
    code='import json,sys,time;from pathlib import Path;p=Path(sys.argv[1]);p.write_text(json.dumps(dict(run_uuid=sys.argv[2],observed_epoch=time.time())))'
    p=subprocess.run([sys.executable,'-c',code,str(path/'OWNER_HEARTBEAT.json'),run],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20)
    if p.returncode:raise RuntimeError('ISOLATED_EXIT_HEARTBEAT_WRITER_FAILED')
    value=read(path/'OWNER_HEARTBEAT.json')
    if value['run_uuid']!=run or not 0<=time.time()-value['observed_epoch']<20:
        raise PermissionError('ACTUAL_ISOLATED_PROCESS_EXIT_AND_FRESH_INITIAL_HEARTBEAT')
    write(path/'WRITER_TERMINAL.json',dict(returncode=0,heartbeat_sha256=sha(path/'OWNER_HEARTBEAT.json'),
        observed_epoch=time.time(),production_heartbeat_written=False))
    return path


def heartbeat_fault(path,run,expected):
    before=sha(path/'OWNER_HEARTBEAT.json')
    try:read_live_owner(path,run,force_revalidate=True)
    except expected as exc:
        value=dict(status='PASS',failed_closed=True,unknown_effects_resent=0,expected_error=type(exc).__name__,
            heartbeat_sha256=before,diagnostics=getattr(exc,'diagnostics',None),production_heartbeat_written=False)
    else:raise PermissionError('TRUE_ISOLATED_HEARTBEAT_LOSS_WAS_NOT_REJECTED')
    if sha(path/'OWNER_HEARTBEAT.json')!=before:raise PermissionError('FAULT_READER_MUST_NOT_RENEW_HEARTBEAT')
    return value


def concurrent_replace(folder,run):
    path=Path(folder)/'concurrent_replace';path.mkdir(mode=0o700)
    write(path/'OWNER_HEARTBEAT.json',dict(run_uuid=run,observed_epoch=time.time()))
    stop=threading.Event();errors=[];writes=[]
    def writer():
        try:
            while not stop.is_set():
                now=time.time();write(path/'OWNER_HEARTBEAT.json',dict(run_uuid=run,observed_epoch=now));writes.append(now)
                stop.wait(.003)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=writer);thread.start();rows=[]
    try:
        for _ in range(40):
            rows.append(read_live_owner(path,run,force_revalidate=True));time.sleep(.005)
    finally:stop.set();thread.join(timeout=20)
    if thread.is_alive() or errors or len(writes)<2 or len({r['heartbeat']['observed_epoch'] for r in rows})<2:
        raise PermissionError('REAL_CONCURRENT_ATOMIC_REPLACE_NOT_OBSERVED')
    write(path/'READS.json',dict(rows=rows,writes=writes))
    return dict(status='PASS',failed_closed=False,unknown_effects_resent=0,actual_replaces=len(writes),
        post_read_samples=len(rows),rows_sha256=sha(path/'READS.json'),production_heartbeat_written=False)


def metadata_failure(folder):
    path=Path(folder)/'metadata_read_failure';path.mkdir(mode=0o700);write(path/'RESOURCE.json',dict(fixture=True))
    attempts=[]
    def operation():
        attempts.append(time.monotonic())
        if len(attempts)==1:raise OSError(errno.ESTALE,'ISOLATED_INJECTED_STALE_METADATA_READ')
        return read(path/'RESOURCE.json')
    before=time.monotonic();value=read_only(operation,kind='registered_resource_status',sleep=lambda seconds:time.sleep(.01))
    if value!={'fixture':True} or len(attempts)!=2:raise PermissionError('BOUNDED_CLASSIFIED_FIXTURE_METADATA_RECOVERY')
    return dict(status='PASS',failed_closed=False,unknown_effects_resent=0,attempts=len(attempts),wall_s=time.monotonic()-before,
        injected_errno=errno.ESTALE,recovered_only_declared_idempotent_read=True,production_read_fault_injected=False)


def statistics_lock(folder):
    started_CPU=time.process_time();path=Path(folder)/'statistics_lock';path.mkdir(mode=0o700);lock=path/'STATS.lock'
    code='import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX);print("LOCK_HELD",flush=True);sys.stdin.read(1)'
    child=subprocess.Popen([sys.executable,'-u','-c',code,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    try:
        if not select.select([child.stdout],[],[],20)[0] or child.stdout.readline()!=b'LOCK_HELD\n':
            raise TimeoutError('ACTUAL_ISOLATED_STATISTICS_LOCK_HOLDER')
        metrics=SessionMetrics();emitted=[];before=time.monotonic();metrics.reply(dict(fixture=True),emitted.append)
        elapsed=time.monotonic()-before
        if child.poll() is not None or emitted!=[{'fixture':True}] or elapsed>=20:
            raise PermissionError('PRODUCTION_METRIC_REPLY_MUST_NOT_WAIT_FOR_STATISTICS_LOCK')
    finally:
        if child.stdin and not child.stdin.closed:child.stdin.close()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise
        if child.stdout:child.stdout.close()
        if child.stderr:child.stderr.close()
    if child.returncode:raise PermissionError('ISOLATED_STATS_HOLDER_DID_NOT_CLOSE')
    class FixtureLedger:
        role='hopper'
        @contextmanager
        def transaction(self):
            with lock.open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);value={};yield value;write(path/'COUNTERS.json',value)
    settled=metrics.settle(FixtureLedger(),time.process_time()-started_CPU)
    return dict(status='PASS',failed_closed=False,unknown_effects_resent=0,actual_lock_holder_exit=0,
        reply_while_lock_held=True,reply_wall_s=elapsed,settlement_after_unlock=settled,
        production_session_metrics_primitive=True,production_ledger_lock_injected=False)


def finish(folder,run):
    folder=Path(folder);faults=folder/'faults';result={}
    expired=faults/'actual_expired_heartbeat'
    if time.time()-read(expired/'OWNER_HEARTBEAT.json')['observed_epoch']<=300:
        raise PermissionError('ACTUAL_EXITED_WRITER_MUST_TRULY_AGE_PAST_ORIGINAL_WATCHDOG')
    result['actual_expired_heartbeat']=heartbeat_fault(expired,run,HeartbeatLost)
    for name,wrong in [('actual_future_heartbeat',False),('wrong_run_heartbeat',True)]:
        path=faults/name;path.mkdir(mode=0o700)
        write(path/'OWNER_HEARTBEAT.json',dict(run_uuid='ISOLATED_WRONG_RUN' if wrong else run,
            observed_epoch=time.time()+(0 if wrong else 30)))
        result[name]=heartbeat_fault(path,run,PermissionError if wrong else HeartbeatLost)
    result['concurrent_replace']=concurrent_replace(faults,run)
    result['metadata_read_failure']=metadata_failure(faults)
    result['statistics_lock']=statistics_lock(faults)
    for name,value in result.items():
        p=faults/name/'RESULT.json';write(p,value);p.chmod(0o400)
    return result
