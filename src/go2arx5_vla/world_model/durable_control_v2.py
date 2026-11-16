"""Bounded contention and missing-control-file waits, never scientific retries."""
from contextlib import contextmanager
import json,time
from pathlib import Path
from scripts.pi0292.durable import OwnerLock

@contextmanager
def short_lock(path,timeout=10):
    until=time.monotonic()+timeout;delay=.01
    while True:
        try:lock=OwnerLock(path);break
        except BlockingIOError:
            if time.monotonic()>=until:raise
            time.sleep(min(delay,max(0,until-time.monotonic())));delay=min(.25,delay*1.5)
    try:yield lock
    finally:lock.close()

def read_control(path,timeout=10):
    until=time.monotonic()+timeout
    while True:
        try:return json.loads(Path(path).read_bytes())
        except FileNotFoundError:
            if time.monotonic()>=until:raise
            time.sleep(.02)

def valid_dispatch(path,generation,*,now=None,timeout=10):
    value=read_control(path,timeout);now=time.time()if now is None else now
    if (value.get('schema')!='MFREE_RECOVERY_DISPATCH_V2'or value.get('generation')!=generation
            or not isinstance(value.get('collectors'),list)or type(value.get('ready'))is not bool):
        raise ValueError('UNKNOWN_OR_DAMAGED_DISPATCH_VERSION_NO_DISPATCH')
    if not value['ready']or now-value['epoch']>180:return dict(value,collectors=[],wait='WAIT_DISPATCH')
    return value

def launch_phase(case,journal_record=None,*,parent_cgroup_empty=False):
    """Evidence classification only. A launch INTENT is never replayable."""
    case=Path(case)
    if(case/'RESULT.private.json').exists():return 'TERMINAL_IMPORT_ONLY'
    if journal_record is not None:return 'UNKNOWN_LAUNCH_INTENT_NO_REPLAY'
    if(case/'EXECUTION_SPEC.json').exists():return 'REGISTERED_NOT_LAUNCHED'if parent_cgroup_empty else'WAIT_OWNER_RECONCILIATION'
    return 'CONFIRMED_NOT_STARTED'if parent_cgroup_empty else'WAIT_OWNER_RECONCILIATION'
