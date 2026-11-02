"""Retire only an identified idle project manager at a verified cold boundary.

This is not worker cancellation. The caller must verify the sealed manager,
completed metadata steps, all native workers/leases, and its exact wait state
while the manager is briefly paused. A failed cold check resumes it unchanged.
"""
import os
from pathlib import Path
import select
import signal
import time


def process_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(proc_start=fields[19], state=fields[0],
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split())


def stop_idle_manager(identity, verify_frozen, record_intent):
    """No PID reuse; no child/worker kill; no stop when cold proof is missing."""
    pid = identity['PID']
    if type(pid) is not int or pid <= 1 or identity['proc_start'] is None:
        raise PermissionError('EXACT_PROJECT_MANAGER_IDENTITY_REQUIRED')
    if str(process_identity(pid)['proc_start']) != str(identity['proc_start']):
        raise PermissionError('PROJECT_MANAGER_PID_REUSED')
    fd = os.pidfd_open(pid)
    paused = terminating = False
    try:
        if str(process_identity(pid)['proc_start']) != str(identity['proc_start']):
            raise PermissionError('PROJECT_MANAGER_IDENTITY_CHANGED')
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        paused = True
        until = time.monotonic()+3
        while process_identity(pid)['state'] not in ('T', 't'):
            if time.monotonic() >= until:
                raise PermissionError('PROJECT_MANAGER_PAUSE_NOT_CONFIRMED')
            time.sleep(.02)
        frozen = process_identity(pid)
        if frozen['children'] or not verify_frozen():
            return False
        # Evidence is durable before the one known idle-manager termination.
        record_intent(dict(PID=pid, proc_start=str(identity['proc_start']),
            verified_cold_wait=True, child_processes=0, epoch=time.time()))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        terminating = True
        signal.pidfd_send_signal(fd, signal.SIGCONT)
        paused = False
        if not select.select([fd], [], [], 10)[0]:
            raise RuntimeError('IDLE_MANAGER_TERMINATION_UNKNOWN_NO_SUCCESSOR_START')
        return True
    finally:
        if paused and not terminating:
            signal.pidfd_send_signal(fd, signal.SIGCONT)
        os.close(fd)


def inference_choice(gpu0_ready, gpu1_ready):
    """GPU1 accelerates an otherwise ready task; its absence cannot block GPU0."""
    return 1 if gpu1_ready else 0 if gpu0_ready else None
