"""Raw synthetic native-site fixtures only; no allocation/model/physics claims."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import hashlib,json
from unittest.mock import patch
import pytest
from scripts.pi0294.pv1 import evaluation_native as n,evaluation_scope as s,evaluation_account_jobs as account
from scripts.pi0294.pv1 import evaluation_admission as a,evaluation_platform as e
from scripts.pi0294.test_pv1_a40_evaluation_identity import case,proof
from scripts.pi0294.test_pv1_v4_evaluation_admission import inputs,native,site,raw,rebuild_snapshot
from scripts.pi0294.test_pv1_v4_evaluation_work import synthetic_population_only

REPO=Path(__file__).resolve().parents[2]
GRANT=(REPO/'reports/PI0294/design/parallel-science-20261022/V4_S2_GLOBAL_RESOURCE_AND_SAVED_B4096_REUSE_AUTHORIZATION_20261024.json').read_bytes()
OTHER=(REPO/'reports/PI0294/design/parallel-science-20261022/V4_INDEPENDENT_ACCOUNT_JOB1481491_HUMAN_IDENTITY_20261024.json').read_bytes()


@pytest.fixture
def current(case):
    c=case;i=inputs(c);r=json.loads(i['request_bytes']);m=json.loads(i['manifest_bytes'])
    q=site(i)['effective_queue_limit'];q['observed_epoch']=r['created_epoch']-1;qb=raw(q);qs=hashlib.sha256(qb).hexdigest()
    r.update(queue_limit_relative='plans/QUEUE_LIMIT.json',queue_limit_sha256=qs,sealed_inputs={'plans/QUEUE_LIMIT.json':qs})
    m['execution']['sealed_inputs']={'plans/QUEUE_LIMIT.json':qs};i['manifest_bytes']=raw(m)
    r['run_manifest_sha256']=hashlib.sha256(i['manifest_bytes']).hexdigest();i['request_bytes']=raw(r)
    i=native(c,i);r=json.loads(i['request_bytes']);site_=site(i)
    run=dict(site_,schema='PI0294_V4_NATIVE_RUNNING_SITE_ATTESTATION_V1',source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
        run_uuid=r['run_uuid'],campaign_run_uuid=r['campaign_run_uuid'],job_id=c['a']['job_id'],source_sha=r['source_sha'],
        phase_started_epoch=c['a']['allocation_start_epoch'])
    kwargs=dict(recompute_inputs=i,queue_proof=proof(c),site_bytes=raw(run),effective_queue_bytes=qb,
        user='e1192548',hostname=c['a']['hostname'],job_id=c['a']['job_id'],grant_bytes=GRANT)
    # Native authority adoption itself is separately tested from actual captured
    # bytes. Only this synthetic fixture's authority is isolated, not resources.
    with patch.object(s,'validate_pair',return_value=dict(adopted_epoch=c['now']-1000)):
        yield c,kwargs


def test_typed_native_raw_inputs_recompute_with_partial_B0_without_model_permission(current):
    c,k=current;receipt=n.validate(**k);i=k['recompute_inputs'];r=json.loads(i['request_bytes']);m=json.loads(i['manifest_bytes'])
    assert receipt['status']==e.STATUS and receipt['resource_recomputation']['all_remaining_costs_fit'] is True
    assert receipt['physics_authorized'] is False and receipt['new_optimizer_updates']==0 and receipt['model_loaded'] is False
    assert receipt['native_allocated_ncpus']==4 and receipt['independent_outside_jobs']==[]
    e.sealed_identity(m,r,receipt,run_uuid=r['run_uuid'],manifest_sha256=hashlib.sha256(i['manifest_bytes']).hexdigest(),request_sha256=hashlib.sha256(i['request_bytes']).hexdigest())


@pytest.mark.parametrize('fault',['site_stale','prequeue_site','job','nonce','request','source','owner_project','membership','date','balance_GPU','balance_CPU',
    'quota','queue_cap','wrong_host','wrong_user','not_R','rerun','snapshot_before_phase','resource_shortfall','wrong_grant','missing_scope'])
def test_native_phase_rejects_stale_wrong_or_underfunded_raw_inputs(current,fault):
    c,k=current;k=deepcopy(k);site_=json.loads(k['site_bytes']);i=k['recompute_inputs']
    if fault=='site_stale':site_['observed_epoch']=c['now']-121
    elif fault=='prequeue_site':site_['phase_started_epoch']-=1
    elif fault in ('job','nonce','request','source'):site_[dict(job='job_id',nonce='submission_nonce',request='request_sha256',source='source_sha')[fault]]='wrong'
    elif fault=='owner_project':site_['project']['name']='CFP04-CF-031'
    elif fault=='membership':site_['project']['users']=[]
    elif fault=='date':site_['project']['end_date']='2020-01-01'
    elif fault=='balance_GPU':site_['balance_GPU_hours']=0
    elif fault=='balance_CPU':site_['balance_CPU_hours']=0
    elif fault=='quota':site_['quota']['available_bytes']=0
    elif fault=='queue_cap':k['effective_queue_bytes']=raw(dict(json.loads(k['effective_queue_bytes']),maximum_wall_s=1))
    elif fault=='wrong_host':k['hostname']='GN-A40-999'
    elif fault=='wrong_user':k['user']='other'
    elif fault in ('not_R','rerun'):
        q=k['queue_proof'];j=json.loads(q['qstat']['stdout']);row=j['Jobs'][c['a']['job_id']];row['job_state']='Q' if fault=='not_R' else 'R';row['run_count']=2 if fault=='rerun' else 1;q['qstat']['stdout']=json.dumps(j)
    elif fault=='snapshot_before_phase':
        snap=json.loads(i['snapshot_bytes']);snap['observed_epoch']=c['a']['allocation_start_epoch']-1;i['snapshot_bytes']=raw(snap)
    elif fault=='resource_shortfall':
        snap=json.loads(i['snapshot_bytes']);snap['used']['Core_CPU_s']=99999999;i['snapshot_bytes']=raw(snap)
    elif fault=='wrong_grant':k['grant_bytes']+=b' '
    else:
        with patch.object(s,'validate_pair',side_effect=PermissionError('NO_ADOPTION')):
            with pytest.raises(PermissionError):n.validate(**k)
        return
    k['site_bytes']=raw(site_)
    with pytest.raises((PermissionError,KeyError,ValueError)):n.validate(**k)


def outside():
    return dict(platform=e.PLATFORM,job_id='1481491.stdct-mgmt-02',name='l08b-1-e1192548-1',project='CFP04-CF-031',ngpus=2,state='R')


def test_exact_human_other_job_kept_separate_never_charged_or_mutated():
    row=outside();before=deepcopy(row)
    result=account.checked_outside([row],[row],OTHER)
    assert row==before and result[0]['may_use_modify_stop'] is False
    assert result[0]['human_confirmation_sha256']==account.SHA
    with pytest.raises(PermissionError):account.checked_outside([row],[row])
    with pytest.raises(PermissionError):account.checked_outside([row],[row],OTHER+b' ')


@pytest.mark.parametrize('key,value',[('job_id','1481492.stdct-mgmt-02'),('name','another'),('project','CFP04-CF-007'),('ngpus',1),('state','H')])
def test_confirmation_is_not_an_exception_for_another_or_changed_job(key,value):
    row=outside();row[key]=value
    with pytest.raises(PermissionError):account.checked_outside([row],[row],OTHER)


def test_second_unknown_outside_GPU_still_rejects():
    rows=[outside(),dict(outside(),job_id='999.stdct-mgmt-02')]
    with pytest.raises(PermissionError):account.checked_outside(rows,rows,OTHER)


def release_case(current):
    import base64
    c,k=current;phase=c['now']-5;site_=json.loads(k['site_bytes']);site_['phase_started_epoch']=phase
    i=k['recompute_inputs'];r=json.loads(i['request_bytes'])
    owner=dict(schema='PI0294_V4_FRESH_DEV_QUERY_RELEASE_V1',owner='EXISTING_LOCAL_PC',job_id=k['job_id'],
        run_uuid=r['run_uuid'],request_sha256=hashlib.sha256(i['request_bytes']).hexdigest(),request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],phase=n.PHASE,phase_started_epoch=phase,observed_epoch=c['now'],
        global_ledger_sha256=hashlib.sha256(i['global_bytes']).hexdigest(),LOCAL_ledger_sha256=hashlib.sha256(i['local_bytes']).hexdigest(),
        snapshot_sha256=hashlib.sha256(i['snapshot_bytes']).hexdigest(),running_site_base64=base64.b64encode(raw(site_)).decode())
    allocation=n.validate(**k)
    args=dict(recompute_inputs=i,queue_proof=k['queue_proof'],owner_bytes=raw(owner),allocation=allocation,
        phase_started_epoch=phase,extra={name:k[name] for name in ('grant_bytes','effective_queue_bytes')})
    return c,k,owner,args


def test_same_post_load_phase_requires_fresh_raw_budget_and_native_site_again(current):
    c,k,owner,args=release_case(current)
    with patch.dict(n.os.environ,dict(USER='e1192548',PBS_JOBID=k['job_id'])),patch.object(n.socket,'gethostname',return_value=k['hostname']):
        result=n.verify_release(**args)
    assert result['physics_authorized'] is False and result['new_optimizer_updates']==0
    assert result['resource_recomputation']['all_remaining_costs_fit'] is True


@pytest.mark.parametrize('fault',['old_phase','other_job','other_nonce','stale_owner','old_GLOBAL','old_LOCAL','old_snapshot','preload_site','changed_allocation'])
def test_prequeue_or_previous_owner_release_cannot_be_reused_after_loading(current,fault):
    c,k,owner,args=release_case(current)
    if fault=='old_phase':owner['phase_started_epoch']-=1
    elif fault=='other_job':owner['job_id']='999.stdct-mgmt-02'
    elif fault=='other_nonce':owner['submission_nonce']='wrong'
    elif fault=='stale_owner':owner['observed_epoch']=c['now']-121
    elif fault in ('old_GLOBAL','old_LOCAL','old_snapshot'):owner[dict(old_GLOBAL='global_ledger_sha256',old_LOCAL='LOCAL_ledger_sha256',old_snapshot='snapshot_sha256')[fault]]='a'*64
    elif fault=='preload_site':
        import base64
        site_=json.loads(base64.b64decode(owner['running_site_base64']));site_['phase_started_epoch']=c['a']['allocation_start_epoch'];owner['running_site_base64']=base64.b64encode(raw(site_)).decode()
    else:args['allocation']=dict(args['allocation'],allocation_end_epoch=args['allocation']['allocation_end_epoch']+1)
    args['owner_bytes']=raw(owner)
    with patch.dict(n.os.environ,dict(USER='e1192548',PBS_JOBID=k['job_id'])),patch.object(n.socket,'gethostname',return_value=k['hostname']):
        with pytest.raises(PermissionError):n.verify_release(**args)
