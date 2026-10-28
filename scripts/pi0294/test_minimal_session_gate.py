"""Finite ticket/release CPU fixtures; no registered native processes or GPU."""
from copy import deepcopy
import pytest
from scripts.pi0294.test_minimal_execution import actual,request
from scripts.pi0294.generalization import minimal_session_gate as n,minimal_effects as f,minimal_execution as e


def fixture(g,cap,index):
    r=request(g,cap);d=n.plan()[index];spec=next(x for x in e.native_sessions()if x['id']==d['session_id'])
    counts=dict(optimizer_updates=0,generation_queries=0,model_loads=0)
    if d['phase']=='FIRST_UPDATE':counts.update(generation_queries=0 if spec['formal_candidate']else 15,model_loads=1)
    elif d['phase']=='RESTORE_FULL':counts.update(optimizer_updates=2,generation_queries=16,model_loads=1)
    elif d['phase']=='OFFICIAL_EXPORT':counts.update(optimizer_updates=512,generation_queries=36,model_loads=1)
    ready=dict(schema=n.READY,ticket_version=n.VERSION,**d,job_id='CPU_NATIVE_HANDLE_FIXTURE',request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],request_sha256=e.sha(e.raw(r)),source_sha=r['source_sha'],
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,session_counters=counts,observed_epoch=100.)
    effects=dict(schema=f.SCHEMA,request_sha256=ready['request_sha256'],job_id=ready['job_id'],source_sha=r['source_sha'],
        active_session=d['session_id'],pending_updates=[],unknown_effects=[],sessions={d['session_id']:counts})
    return r,ready,effects


@pytest.mark.parametrize('index',range(13))
def test_all_original_finite_phases_bind_exact_session_and_actual_effect_boundary(actual,index):
    g,cap=actual;r,ready,effects=fixture(g,cap,index)
    assert n.check_ready(ready,r,request_sha256=ready['request_sha256'],job_id=ready['job_id'],effect_value=effects)==n.plan()[index]
    bad=deepcopy(ready);bad['session_counters']['model_loads']+=1
    with pytest.raises(PermissionError):n.check_ready(bad,r,request_sha256=ready['request_sha256'],job_id=ready['job_id'],effect_value=effects)


@pytest.mark.parametrize('change',[None,'stale','wrong_job','wrong_ready','unknown','pending','quota','future'])
def test_session_release_is_fresh_and_preserves_original_resource_and_unknown_fences(actual,change):
    g,cap=actual;r,ready,_=fixture(g,cap,2);body=e.raw(ready)
    value=dict(schema=n.RELEASE,ticket_version=n.VERSION,status='PASS_MINIMAL_SESSION_FIRST_UPDATE_NATIVE',
        ready_sha256=e.sha(body),session_id=ready['session_id'],phase=ready['phase'],job_id=ready['job_id'],
        request_sha256=ready['request_sha256'],scope_grant_sha256=e.authority.GRANT_SHA,observed_epoch=101.,
        native_proof=dict(whole_remaining_package_shortfalls={},complete_queue_verified=True,project_eligible=True,
            actual_project_quota_covers_remaining_output=True,global_unknown_effects=[],global_pending_updates=[],native_own_job_id=ready['job_id']))
    if change=='stale':value['observed_epoch']=-1000.
    elif change=='wrong_job':value['job_id']='OTHER_JOB'
    elif change=='wrong_ready':value['ready_sha256']='a'*64
    elif change=='unknown':value['native_proof']['global_unknown_effects']=['PRESERVED']
    elif change=='pending':value['native_proof']['global_pending_updates']=[{}]
    elif change=='quota':value['native_proof']['actual_project_quota_covers_remaining_output']=False
    elif change=='future':value['observed_epoch']=1000.
    before=deepcopy(value)
    if change:
        with pytest.raises(PermissionError):n.check_release(value,body,r,job_id=ready['job_id'],now_epoch=102.)
    else:assert n.check_release(value,body,r,job_id=ready['job_id'],now_epoch=102.)==value
    assert value==before
