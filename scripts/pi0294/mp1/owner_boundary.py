"""Bounded retry for the owner's idempotent metadata accounting boundary only."""
import subprocess
import time


def sync_metadata(sync,heartbeat,*,attempts=3,maximum_wait_s=120,sleep=time.sleep,clock=time.monotonic):
    if type(attempts) is not int or not 1<=attempts<=3 or not 0<maximum_wait_s<=120:
        raise ValueError('FINITE_METADATA_READ_RECOVERY_ONLY')
    deadline=clock()+maximum_wait_s;last=None
    for index in range(attempts):
        if index and clock()>=deadline:
            raise last
        heartbeat()
        try:
            return sync()
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            last=exc
            if index+1==attempts or clock()>=deadline:
                raise
            heartbeat()
            sleep(min(5,max(0,deadline-clock())))
