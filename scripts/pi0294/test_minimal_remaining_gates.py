"""Actual prefix bytes, future typed request/PBS/material explicitly fixtures."""
from copy import deepcopy
import pytest
from scripts.pi0294.test_minimal_remaining_effects import remaining,actual_prefix
from scripts.pi0294.generalization import minimal_remaining_contract as r,minimal_session_gate as g
from scripts.pi0294.generalization import minimal_execution as e


def packet(ledger,index):
    request=ledger.request;d=g.plan(request)[index]
    counts={'MODEL_RAW':(0,0,0),'FIRST_UPDATE':(0,0,1),'RESTORE_FULL':(2,1,1),'OFFICIAL_EXPORT':(512,36,1)}[d['phase']]
    counts=dict(zip(r.KEYS,counts));effect=ledger.guard()
    effect.update(active_session=d['session_id']);effect['sessions'][d['session_id']].update(counts,status='ACTIVE')
    ready=dict(schema=g.READY,ticket_version=g.REMAINING_VERSION,**g.prefix_binding(request),**d,
        job_id=ledger.allocation['job_id'],request_id=request['request_id'],submission_nonce=request['submission_nonce'],
        request_sha256=ledger.allocation['request_sha256'],source_sha=request['source_sha'],
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        session_counters=counts,observed_epoch=100.)
    return ready,effect


@pytest.mark.parametrize('index',range(9))
def test_remaining_nine_tickets_use_own_debits_and_cannot_replay_original_prefix(remaining,index):
    ledger,_=remaining;ready,effect=packet(ledger,index)
    assert g.check_ready(ready,ledger.request,request_sha256=ledger.allocation['request_sha256'],job_id=ledger.allocation['job_id'],effect_value=effect)==g.plan(ledger.request)[index]
    assert len(g.plan())==13 and len(g.plan(ledger.request))==9
    for sid in(e.native_sessions()[0]['id'],e.native_sessions()[1]['id']):
        with pytest.raises(PermissionError):g.descriptor(sid,'MODEL_RAW',request=ledger.request)
    bad=deepcopy(ready);bad['ticket_version']=g.VERSION
    with pytest.raises(PermissionError):g.check_ready(bad,ledger.request,request_sha256=ledger.allocation['request_sha256'],job_id=ledger.allocation['job_id'],effect_value=effect)


@pytest.mark.parametrize('change',['carried_as_new','prefix','unknown','pending'])
def test_saved_zero_credit_does_not_bypass_own_effects_or_pending_fences(remaining,change):
    ledger,_=remaining;ready,effect=packet(ledger,1)
    if change=='carried_as_new':ready['session_counters']['generation_queries']=effect['sessions'][ready['session_id']]['generation_queries']=15
    elif change=='prefix':effect['audited_saved_prefix']['result_sha256']='a'*64
    elif change=='unknown':effect['unknown_effects']=['LOST_REPLY_KEEP']
    else:effect['pending_updates']=[{'request_id':ledger.request['request_id']}]
    with pytest.raises(PermissionError):g.check_ready(ready,ledger.request,request_sha256=ledger.allocation['request_sha256'],job_id=ledger.allocation['job_id'],effect_value=effect)


@pytest.mark.parametrize('change',[None,'old_ticket','prefix','stale','quota'])
def test_remaining_release_binds_same_saved_prefix_fresh_native_and_full_budget(remaining,change):
    ledger,_=remaining;ready,_=packet(ledger,1);body=e.raw(ready)
    value=dict(schema=g.RELEASE,ticket_version=g.REMAINING_VERSION,**g.prefix_binding(ledger.request),
        status='PASS_MINIMAL_SESSION_FIRST_UPDATE_NATIVE',ready_sha256=e.sha(body),
        session_id=ready['session_id'],phase=ready['phase'],job_id=ready['job_id'],request_sha256=ready['request_sha256'],
        scope_grant_sha256=e.authority.GRANT_SHA,observed_epoch=101.,native_proof=dict(
            whole_remaining_package_shortfalls={},complete_queue_verified=True,project_eligible=True,
            actual_project_quota_covers_remaining_output=True,global_unknown_effects=[],global_pending_updates=[],native_own_job_id=ready['job_id']))
    if change=='old_ticket':value['ticket_version']=g.VERSION
    elif change=='prefix':value['saved_prefix_result_sha256']='a'*64
    elif change=='stale':value['observed_epoch']=-1000
    elif change=='quota':value['native_proof']['actual_project_quota_covers_remaining_output']=False
    if change:
        with pytest.raises(PermissionError):g.check_release(value,body,ledger.request,job_id=ready['job_id'],now_epoch=102.)
    else:assert g.check_release(value,body,ledger.request,job_id=ready['job_id'],now_epoch=102.)==value
