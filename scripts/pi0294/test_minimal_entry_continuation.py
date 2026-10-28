"""Actual closed third failed job; every prospective handle is a CPU fixture."""
import os,json,base64,gzip
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_retry_execution as retry,minimal_entry_continuation_scope as s,minimal_portable_inventory as portable,minimal_retry_runtime_guard as guard,minimal_execution_adapter as adapters
from scripts.pi0294.dec1.a40_training_context import minimal_training_adapter

@pytest.fixture
def actual():
 p=Path(os.environ.get('ARX5_ENTRY_CONTINUATION_CAPTURE',''))
 if not p.is_file():pytest.skip('actual closed1503589 full canonical capture required')
 v=json.loads(p.read_bytes());return v,gzip.decompress(base64.b64decode(v['GLOBAL_gzip_base64'])),gzip.decompress(base64.b64decode(v['LOCAL_gzip_base64'])),Path(v['scope_path']).read_bytes()

@pytest.fixture
def adopted(actual):
 v,gb,lb,body=actual;out=s.propose(gb,lb,body,closed_readback=v['closed_readback'],now_epoch=v['closed_readback']['observed_epoch']);return v,out['proposed_global'],lb


def test_exact_cap41_and_eight_MiB_preserve_all_old_costs_and_rows(actual):
 v,gb,lb,body=actual;g=json.loads(gb);out=s.propose(gb,lb,body,closed_readback=v['closed_readback'],now_epoch=v['closed_readback']['observed_epoch']);n=out['proposed_global'];f=out['full_future_proof']
 assert f['cap_changes_required']=={}and f['dimensions']['PBS_submissions']['projected']==41
 assert f['dimensions']['transfer_bytes']['cap']-f['dimensions']['transfer_bytes']['projected']==435935
 assert n['PBS_submissions']==g['PBS_submissions']and n[retry.FOLLOWUP_FIELD]==g[retry.FOLLOWUP_FIELD]and n[retry.FIELD]==g[retry.FIELD]
 assert n['CPU_s']==g['CPU_s']+3600 and n['new_storage_current_bytes']==g['new_storage_current_bytes']+64*1024**2
 assert s.CONTROL['transfer_bytes']==8*1024**2


def inputs(adopted):
 v,g,lb=adopted;previous=g[retry.FOLLOWUP_FIELD];r=json.loads(base64.b64decode(previous['request_base64']));now=g[s.FIELD]['adopted_epoch']+2
 r.update(retry.fixed_request_fields(3),source_sha='f'*40,request_id='44444444-4444-4444-8444-444444444444',submission_nonce='55555555-5555-4555-8555-555555555555',run_manifest_sha256='a'*64,cold_input_audit_sha256='b'*64,submit_count=38,created_epoch=now,
  previous_entry_request_id=s.FAILED,previous_entry_row_sha256=g[s.FIELD]['old_retry_row_sha256'],previous_entry_marker_sha256=g[s.FIELD]['old_retry_marker_sha256'],previous_entry_closed_lease_sha256=s.LEASE,entry_scope_sha256=s.GRANT_SHA)
 # Explicit future portable and process fixtures. All actual closed GLOBAL/
 # source/terminal input above is captured; this fixture is not native custody.
 pi=deepcopy(previous['physical_custody']['portable_inventory']);names=list(pi['value']['roots']);extra='arx5-minimal-raw-parent-first512-retry-02';pi['value']['roots'][extra]=dict(path='/CPU_FIXTURE/'+extra,filesystem=next(iter(pi['value']['roots'].values()))['filesystem'],input_symlinks={});pi['value']['metadata_ceiling']=256*1024**2;pi['portable_sha256']=portable.digest(pi['value'])
 physical=dict(previous['physical_custody'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],run_uuid=s.RUN,scope_sha256=s.GRANT_SHA,old_metadata_ceiling=256*1024**2,portable_inventory=pi,old_inventory_sha256=pi['portable_sha256'])
 physical['old_roots']=deepcopy(previous['physical_custody']['old_roots'])+[dict(run_uuid=extra,manifest_sha256=previous['run_manifest_sha256']if 'run_manifest_sha256'in previous else json.loads(base64.b64decode(previous['request_base64']))['run_manifest_sha256'],closed_lease_sha256=s.LEASE)]
 r['physical_custody_sha256']=e.sha(e.raw(physical));proof=deepcopy(previous['physical_preflight']);proof.update(old_inventory_sha256=pi['portable_sha256'],portable_inventory_sha256=pi['portable_sha256'],portable_identity_sha256=portable.identity_digest(pi['value']),current_portable_observation_sha256=pi['portable_sha256'],old_allocated_bytes=pi['value']['allocated_bytes'],old_metadata_ceiling=256*1024**2,old_roots=[pi['value']['roots'][n]['path']for n in names+[extra]],new_root='/CPU_FIXTURE/'+s.RUN,new_allocated_bytes=4096)
 return dict(global_bytes=e.raw(g),local_bytes=lb,request_bytes=e.raw(r),followup_closed_readback=v['closed_readback'],physical_custody=physical,physical_proof=proof,physical_observed_epoch=now,now_epoch=now)


def test_original_entry_and_actual_four_row_register_bind(adopted):
 args=inputs(adopted);g=retry.register_entry_followup(**args)['proposed_global'];from scripts.pi0294.pv1.neural_ledger import check
 check(g);r=json.loads(args['request_bytes']);assert minimal_training_adapter(retry.VERSION,s.PROFILE)is retry
 assert len(retry.rows(g))==4 and adapters.current(g)[1]['request_id']==r['request_id']
 bound=retry.bind(g,r['request_id'],'123456.stdct-mgmt-02',now_epoch=args['now_epoch']+1);check(bound)
 assert bound['PBS_submissions'][:-1]==g['PBS_submissions'][:-1]and retry.validate(bound,generation=2)==g[retry.FOLLOWUP_FIELD]
 assert g[retry.ENTRY_FIELD]['control_paid_once']['transfer_bytes']==8*1024**2

@pytest.mark.parametrize('field',['old_marker','old_row','scope','lease','effect','pending','physical'])
def test_exact_current_failure_scope_and_protected_history_cannot_be_bypassed(adopted,field):
 args=inputs(adopted);g=json.loads(args['global_bytes']);r=json.loads(args['request_bytes'])
 if field=='old_marker':g[retry.FOLLOWUP_FIELD]['before']['CPU_s']+=1
 elif field=='old_row':g['PBS_submissions'][-1]['committed_optimizer_updates']=1
 elif field=='scope':r['entry_scope_sha256']='0'*64
 elif field=='lease':r['previous_entry_closed_lease_sha256']='0'*64
 elif field=='effect':g['unknown_effects']=['UNKNOWN_RETAINED']
 elif field=='pending':g['pending_updates']=['PENDING_RETAINED']
 else:args['physical_proof']['portable_identity_sha256']='0'*64
 args.update(global_bytes=e.raw(g),request_bytes=e.raw(r))
 with pytest.raises((PermissionError,KeyError,AssertionError,ValueError)):retry.register_entry_followup(**args)

@pytest.mark.parametrize('success',[False,True])
def test_fourth_row_original_terminal_retains_all_prior_rows(adopted,success):
    from scripts.pi0294.test_minimal_terminal import captured_CPU_terminal
    args=inputs(adopted);g=retry.register_entry_followup(**args)['proposed_global'];rid=json.loads(args['request_bytes'])['request_id'];g=retry.bind(g,rid,'123456.stdct-mgmt-02',now_epoch=args['now_epoch']+1);_,row=retry._selected(g)
    t=captured_CPU_terminal(g,success=success)
    for kind in('native','effects','audit'):t[kind].update({k:row[k]for k in('request_id','request_sha256','job_id','source_sha')})
    t['native'].update(submission_nonce=row['submission_nonce'],prior_readonly_preparation_CPU_upper_s=0,conservative_CPU_s=1000,new_native_conservative_CPU_s=1000,finished_epoch=args['now_epoch']+5,observed_epoch=args['now_epoch']+6)
    t['audit']['effect_summary_sha256']=e.sha(e.raw(t['effects']));t['now_epoch']=args['now_epoch']+7;t.update(physical_proof=deepcopy(g[retry.ENTRY_FIELD]['physical_preflight']),physical_observed_epoch=args['now_epoch']+6)
    after=retry.settle(g,rid,**t);from scripts.pi0294.pv1.neural_ledger import check
    check(after);assert after['PBS_submissions'][:-1]==g['PBS_submissions'][:-1]and after[retry.FOLLOWUP_FIELD]==g[retry.FOLLOWUP_FIELD]
    assert after['new_storage_current_bytes']==g['new_storage_current_bytes']and after['new_storage_peak_bytes']==g['new_storage_peak_bytes']
    assert after['CPU_s']==g['CPU_s']+1000 and after['optimizer_updates']==g['optimizer_updates']+(1028 if success else 0)
