"""Select the existing SFT preparation path from an actual scan readback.

This is no compute grant. The chosen original preparer still verifies the
complete selected-microbatch restore, or the independent microbatch1 runtime.
"""

WAIT_RUNNING_SCAN = 'WAIT_RUNNING_SCAN'
USE_SCAN_PATH = 'USE_COMPLETED_SCAN_PREPARER'
USE_QUEUE_MICRO1 = 'USE_QUEUE_BLOCKED_MICRO1_SUCCESSOR'


def choose(snapshot):
    pbs = snapshot.get('PBS') or {}
    state = pbs.get('job_state')
    lease = snapshot.get('lease') or {}
    worker = snapshot.get('worker') or {}
    result = snapshot.get('result')
    if state == 'R':
        return WAIT_RUNNING_SCAN
    if state in ('Q', 'H'):
        if not (lease.get('status') == 'PBS_SUBMITTED'
                and not worker and result is None):
            raise ValueError('SCAN_QUEUE_STATE_NEEDS_ACTUAL_UNSTARTED_RECONCILIATION')
        return USE_QUEUE_MICRO1
    if state == 'F':
        if not (pbs.get('Exit_status') == 0 and lease.get('status') == 'CLOSED'
                and lease.get('worker_exit_code') == worker.get('exit_code') == 0
                and isinstance(result, dict) and result
                and result.get('exit_code', 0) == 0):
            raise ValueError('SCAN_FINISHED_WITHOUT_CLOSED0_PROOF_NEEDS_RECONCILIATION')
        return USE_SCAN_PATH
    raise ValueError('SCAN_UNKNOWN_OR_UNSUPPORTED_PBS_STATE_NO_SFT_DISPATCH')
