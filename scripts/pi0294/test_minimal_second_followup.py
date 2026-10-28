"""Captured old F/closed bytes; prospective request/allocation remain CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_retry_execution as retry
from scripts.pi0294.generalization import minimal_followup_scope as s,minimal_portable_inventory as portable
from scripts.pi0294.generalization import minimal_execution_adapter as adapters,minimal_retry_runtime_guard as guard
from scripts.pi0294.pv1.neural_ledger import check

@pytest.fixture
def actual():
    name=os.environ.get('ARX5_FOLLOWUP_CAPTURE')
    if not name:pytest.skip('complete actual second failure/portable metadata required')
    v=json.loads(Path(name).read_bytes());gb=gzip.decompress(base64.b64decode(v['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(v['LOCAL_gzip_base64']));scope=Path(v['scope_path']).read_bytes()
    return v,gb,lb,scope

@pytest.fixture
def adopted(actual):
    v,gb,lb,scope=actual;out=s.propose(gb,lb,scope,closed_readback=v['closed_readback'],now_epoch=v['closed_readback']['observed_epoch'])
    return v,out['proposed_global'],lb


def inputs(adopted):
    v,g,lb=adopted;old=g[retry.FIELD];r=json.loads(base64.b64decode(old['request_base64']));now=g[s.FIELD]['adopted_epoch']+2
    r.update(retry.fixed_request_fields(2),request_id='88888888-8888-4888-8888-888888888888',submission_nonce='99999999-9999-4999-8999-999999999999',source_sha='f'*40,run_manifest_sha256='d'*64,cold_input_audit_sha256='e'*64,submit_count=37,created_epoch=now,
        previous_retry_request_id=s.FAILED,previous_retry_row_sha256=g[s.FIELD]['old_retry_row_sha256'],previous_retry_marker_sha256=g[s.FIELD]['old_retry_marker_sha256'],previous_retry_closed_lease_sha256=s.LEASE,followup_scope_sha256=s.GRANT_SHA)
    pi=deepcopy(v['portable']['portable']);closed=v['portable']['closed'];names=['arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02','arx5-minimal-raw-parent-first512-retry-01']
    physical=dict(schema=guard.PORTABLE_SCHEMA,request_id=r['request_id'],submission_nonce=r['submission_nonce'],run_uuid=s.RUN,scope_sha256=s.GRANT_SHA,old_metadata_ceiling=192*1024**2,paid_output_ceiling=e.native_resources()['Hopper_storage_bytes'],engineering_metadata_allowance=64*1024**2,
        portable_inventory=pi,old_inventory_sha256=pi['portable_sha256'],old_roots=[dict(run_uuid=n,manifest_sha256=closed[n]['manifest_sha256'],closed_lease_sha256=closed[n]['lease_sha256'])for n in names])
    r['physical_custody_sha256']=e.sha(e.raw(physical));root=str(Path(pi['value']['roots'][names[0]]['path']).with_name(s.RUN))
    proof=dict(status='ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION',old_inventory_sha256=pi['portable_sha256'],portable_inventory_sha256=pi['portable_sha256'],portable_identity_sha256=portable.identity_digest(pi['value']),current_portable_observation_sha256=pi['portable_sha256'],old_allocated_bytes=pi['value']['allocated_bytes'],new_allocated_bytes=8192,next_output_upper_bytes=64*1024**2,old_metadata_ceiling=physical['old_metadata_ceiling'],paid_output_ceiling=physical['paid_output_ceiling'],old_roots=[pi['value']['roots'][n]['path']for n in names],new_root=root,ledger_changed=False,native_permission=False)
    return dict(global_bytes=e.raw(g),local_bytes=lb,request_bytes=e.raw(r),followup_closed_readback=v['closed_readback'],physical_custody=physical,physical_proof=proof,physical_observed_epoch=now,now_epoch=now)


def test_scope_adds_only_one_count_and_one_control_and_preserves_every_old_row(actual):
    v,gb,lb,scope=actual;g=json.loads(gb);p=s.propose(gb,lb,scope,closed_readback=v['closed_readback'],now_epoch=v['closed_readback']['observed_epoch']);out=p['proposed_global'];check(out)
    assert out['PBS_submissions']==g['PBS_submissions'] and out[retry.FIELD]==g[retry.FIELD]
    assert out['CPU_s']==g['CPU_s']+3600 and out['new_storage_current_bytes']==g['new_storage_current_bytes']+64*1024**2
    assert p['full_future_proof']['dimensions']['PBS_submissions']['projected']==40
    assert p['full_future_proof']['cap_changes_required']=={} and not p['native_admission']
    with pytest.raises(PermissionError):s.propose(e.raw(out),lb,scope,closed_readback=v['closed_readback'],now_epoch=v['closed_readback']['observed_epoch'])


def test_three_rows_new_request_bind_and_full_check(adopted):
    v,g,lb=adopted;args=inputs(adopted);new=retry.register_followup(**args)['proposed_global'];check(new)
    assert new[retry.FIELD]==g[retry.FIELD] and new['PBS_submissions'][:-1]==g['PBS_submissions']
    assert len(retry.rows(new))==3 and new['CPU_s']==g['CPU_s'] and new['new_storage_current_bytes']==g['new_storage_current_bytes']
    rid=json.loads(args['request_bytes'])['request_id'];assert adapters.current(new)[1]['request_id']==rid
    bound=retry.bind(new,rid,'123456.stdct-mgmt-02',now_epoch=args['now_epoch']+1);check(bound)
    assert adapters.current(bound)[1]['job_id']=='123456.stdct-mgmt-02'
    with pytest.raises(PermissionError):retry.bind(bound,rid,'123457.stdct-mgmt-02',now_epoch=args['now_epoch']+2)
    assert retry.validate(bound,generation=1)==g[retry.FIELD]

@pytest.mark.parametrize('field',['unknown','pending','old_row','old_marker','new_scope','lease','payload','portable'])
def test_no_relabelled_failure_or_missing_portable_scope(adopted,field):
    args=inputs(adopted);g=json.loads(args['global_bytes']);r=json.loads(args['request_bytes'])
    if field=='unknown':g['unknown_effects']=['UNKNOWN_PRESERVED']
    elif field=='pending':g['pending_updates']=[{'PENDING':True}]
    elif field=='old_row':g['PBS_submissions'][-1]['model_load_attempts']=1
    elif field=='old_marker':g[retry.FIELD]['before']['CPU_s']+=1
    elif field=='new_scope':r['followup_scope_sha256']='0'*64
    elif field=='lease':r['previous_retry_closed_lease_sha256']='0'*64
    elif field=='payload':r['formal_terminal_steps']=1024
    else:args['physical_proof']['portable_inventory_sha256']='0'*64
    args.update(global_bytes=e.raw(g),request_bytes=e.raw(r))
    with pytest.raises((PermissionError,KeyError,ValueError,AssertionError)):retry.register_followup(**args)


def test_old_native_request_keeps_old_literal_contract(adopted):
    _,g,_=adopted;old=json.loads(base64.b64decode(g[retry.FIELD]['request_base64']));assert 'continuation_generation'not in old
    assert retry.request(old)==old and retry.fixed_request_fields()['run_uuid']=='arx5-minimal-raw-parent-first512-retry-01'

@pytest.mark.parametrize('success',[False,True])
def test_followup_terminal_keeps_two_closed_rows_and_old_paid_gross(adopted,success):
    from scripts.pi0294.test_minimal_terminal import captured_CPU_terminal
    args=inputs(adopted);g=retry.register_followup(**args)['proposed_global'];rid=json.loads(args['request_bytes'])['request_id']
    g=retry.bind(g,rid,'123456.stdct-mgmt-02',now_epoch=args['now_epoch']+1);_,row=retry._selected(g)
    t=captured_CPU_terminal(g,success=success)
    for kind in ('native','effects','audit'):t[kind].update({k:row[k]for k in('request_id','request_sha256','job_id','source_sha')})
    t['native'].update(submission_nonce=row['submission_nonce'],prior_readonly_preparation_CPU_upper_s=0,conservative_CPU_s=1000,new_native_conservative_CPU_s=1000,finished_epoch=args['now_epoch']+5,observed_epoch=args['now_epoch']+6)
    t['audit']['effect_summary_sha256']=e.sha(e.raw(t['effects']));t['now_epoch']=args['now_epoch']+7;t.update(physical_proof=deepcopy(g[retry.FOLLOWUP_FIELD]['physical_preflight']),physical_observed_epoch=args['now_epoch']+6)
    after=retry.settle(g,rid,**t);check(after)
    assert after['PBS_submissions'][:-1]==g['PBS_submissions'][:-1] and after[retry.FIELD]==g[retry.FIELD]
    assert after['new_storage_current_bytes']==g['new_storage_current_bytes'] and after['new_storage_peak_bytes']==g['new_storage_peak_bytes']
    assert after['CPU_s']==g['CPU_s']+1000 and after['optimizer_updates']==g['optimizer_updates']+(1028 if success else 0)
    assert retry._selected(after)[1]['scientific_endpoint_complete']is success
    with pytest.raises(PermissionError):retry.settle(after,rid,**t)


def test_second_control_owner_leaves_actual_cold_preparation_within_paid3600(adopted):
    from scripts.pi0294.generalization import minimal_native_owner as owner
    request=json.loads(inputs(adopted)['request_bytes'])
    assert owner.control_CPU_limits(request)==(2990,3000)
    assert owner.control_CPU_limits(request)[1]+600==retry.CONTROL['Hopper_CPU_s']
    prior=dict(request,**retry.fixed_request_fields())
    prior.pop('continuation_generation')
    assert owner.control_CPU_limits(prior)==(3590,3600)
    changed=dict(request,profile=retry.PROFILE)
    with pytest.raises(PermissionError):owner.control_CPU_limits(changed)
