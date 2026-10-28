"""Retry only classified read failures; never act as a second heartbeat owner."""
import errno
import signal
import subprocess
import threading
import time

from .owner_boundary import sync_metadata

KINDS={'native_job_status','registered_resource_status','closed_episode_metadata'}


def read_only(operation,*,kind,**bounds):
    if kind not in KINDS:
        raise PermissionError('DECLARED_READONLY_METADATA_ONLY_NO_ACTION_QUERY_START_OR_QSUB')
    maximum=bounds.get('maximum_wait_s',120)
    attempts=bounds.get('attempts',3)
    if type(attempts) is not int or not 1<=attempts<=3 or not 0<maximum<=120:
        raise ValueError('FINITE_METADATA_READ_RECOVERY_ONLY')
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise PermissionError('READONLY_METADATA_REQUIRES_OWN_MAIN_THREAD_WITHOUT_EXISTING_ALARM')
    clock=bounds.get('clock',time.monotonic);deadline=clock()+maximum
    def expired(signum,frame):
        raise subprocess.TimeoutExpired('DECLARED_READONLY_METADATA_NO_MUTATION',maximum)
    def checked():
        remaining=deadline-clock()
        if remaining<=0:raise subprocess.TimeoutExpired('DECLARED_READONLY_METADATA_NO_MUTATION',maximum)
        previous=signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        try:return operation()
        except RuntimeError as exc:
            message=str(exc)
            integrity=['PermissionError','AssertionError','ValueError','JSONDecodeError','MUTATION_OUTCOME_UNKNOWN']
            if (any(word in message for word in integrity)
                    or 'OSError: [Errno 116] Stale file handle' not in message):
                raise PermissionError('UNCLASSIFIED_REMOTE_READ_FAILURE_NO_RETRY') from exc
            raise
        except OSError as exc:
            if exc.errno!=errno.ESTALE:raise
            raise RuntimeError('KNOWN_READONLY_ESTALE') from exc
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
    return sync_metadata(checked,lambda:None,**bounds)
