"""Bounded read retries; scheduler uncertainty is not Core process termination.

Only a live heartbeat may consume a separately fresh raw campaign ledger after
three typed full-queue read failures. That result explicitly retains UNKNOWN
queue status and cannot be used for phase release/native model admission.
Mutation/heartbeat unknowns are never caught or retried here.
"""
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure

def observe(full_read,raw_read,*,heartbeat_only=False,record_failure=lambda value:None):
    if type(heartbeat_only) is not bool:raise ValueError('EXPLICIT_HEARTBEAT_PHASE')
    last=None
    for attempt in range(1,4):
        try:
            result=full_read()
        except ReadOnlyFailure as error:
            last=error;record_failure(dict(attempt=attempt,maximum_attempts=3,kind='READ_ONLY_FULL_QUEUE_UNKNOWN',error=str(error),queue_is_empty=False,native_termination_inferred=False))
        else:
            return dict(result,full_queue_status='ACTUAL_CURRENT_COMPLETE_QUEUE_READ',heartbeat_only_raw_ledger=False)
    if not heartbeat_only:raise last
    # raw_read must independently enforce source, exact fresh ledger and native
    # owner/lease/process guards. No previous bytes or empty-queue substitute.
    result=raw_read()
    return dict(result,full_queue_status='UNKNOWN_THREE_TYPED_READ_FAILURES',heartbeat_only_raw_ledger=True,native_admission_or_phase_release=False)
