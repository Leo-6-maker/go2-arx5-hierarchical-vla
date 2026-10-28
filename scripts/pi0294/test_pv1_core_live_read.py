import pytest
from scripts.pi0294.pv1.evaluation_core_live_read import observe
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure,UnknownMutation

def test_complete_read_needs_no_retry_or_raw_credit():
 calls=[];out=observe(lambda:dict(global_base64='ACTUAL'),lambda:calls.append(1))
 assert out['full_queue_status']=='ACTUAL_CURRENT_COMPLETE_QUEUE_READ' and not out['heartbeat_only_raw_ledger'] and not calls
@pytest.mark.parametrize('failures',[1,2])
def test_only_typed_read_failures_can_retry_inside_three_existing_bounds(failures):
 calls=[]
 def full():
  calls.append(1)
  if len(calls)<=failures:raise ReadOnlyFailure('typed')
  return dict(global_base64='ACTUAL')
 out=observe(full,lambda:pytest.fail('raw fallback not allowed'));assert len(calls)==failures+1 and not out['heartbeat_only_raw_ledger']
def test_three_failures_do_not_claim_empty_queue_or_terminate_core():
 calls=[];events=[]
 def full():calls.append(1);raise ReadOnlyFailure('scheduler unavailable')
 out=observe(full,lambda:dict(global_base64='FRESH_RAW'),heartbeat_only=True,record_failure=events.append)
 assert len(calls)==len(events)==3 and out['heartbeat_only_raw_ledger'] and out['full_queue_status'].startswith('UNKNOWN') and out['native_admission_or_phase_release'] is False
 assert all(e['native_termination_inferred'] is False and e['queue_is_empty'] is False for e in events)
def test_no_model_phase_release_from_unknown_queue():
 calls=[]
 def fail():raise ReadOnlyFailure('typed')
 with pytest.raises(ReadOnlyFailure):observe(fail,lambda:calls.append(1),heartbeat_only=False)
 assert not calls
@pytest.mark.parametrize('error',[UnknownMutation('unknown write'),PermissionError('identity failure'),TimeoutError('not typed')])
def test_unknown_writes_or_identity_errors_are_not_retried(error):
 calls=[]
 def fail():calls.append(1);raise error
 with pytest.raises(type(error)):observe(fail,lambda:pytest.fail('not allowed'),heartbeat_only=True)
 assert calls==[1]
def test_raw_read_failure_cannot_publish_freshness():
 def full():raise ReadOnlyFailure('typed')
 def raw():raise ReadOnlyFailure('raw not fresh')
 with pytest.raises(ReadOnlyFailure):observe(full,raw,heartbeat_only=True)
