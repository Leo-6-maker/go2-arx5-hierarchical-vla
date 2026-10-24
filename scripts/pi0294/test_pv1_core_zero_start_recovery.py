"""Synthetic CPU receipts: one exact zero-effect replacement, no retry pool."""
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import evaluation_core_zero_start_recovery as z,evaluation_core_accounting as c
from scripts.pi0294.test_pv1_core_evaluation_accounting import row
NEW='abac1234-1234-4234-8234-123456789abc';NONCE='bbac1234-1234-4234-8234-123456789abc';CONTROL='cbac1234-1234-4234-8234-123456789abc'
def fixture():
 a=row();a.update(request_id=z.PREVIOUS_A,status=c.STATES[3],native_exit_status=0,terminal_native_sha256='1'*64,conservative_GPU_s=14092)
 b=row();b.update(request_id=z.FAILED,request_sha256=z.FAILED_REQUEST_SHA,status=c.STATES[3],native_exit_status=1,evaluation_candidate='B_COVERAGE',conservative_GPU_s=70,conservative_CPU_s=1,query_attempts=0,model_load_attempts=0,unknown_query_intents=[],reserved_neural_queries=0,native_handle='FIXTURE_HANDLE')
 proof=dict(request_id=z.FAILED,native_handle=b['native_handle'],actual_NATIVE_PROCESS_FINAL_sha256=z.FAILED_FINAL_SHA)
 effects=dict(request_id=z.FAILED,native_handle=b['native_handle'],effect_journal_complete=True,unknown_query_intents=[],optimizer_updates_committed=0,new_physics=0,new_episodes=0,query_attempts=0,model_load_attempts=0)
 b.update(terminal_native_sha256=c.digest(proof),terminal_effect_summary_sha256=c.digest(effects))
 closed=dict(request_id=z.FAILED,terminal_native_sha256=b['terminal_native_sha256'],native_exit_status=1,GPU_lease_status='CLOSED',physical_lease_status='CLOSED',active_native_actors=0,remaining_holds=0,actual_valid_conditions=0,actual_model_loads=0,actual_queries=0,actual_physics=0,actual_training_updates=0)
 ac=dict(request_id=z.PREVIOUS_A,terminal_native_sha256=a['terminal_native_sha256'],GPU_lease_status='CLOSED',physical_lease_status='CLOSED',actual_valid_conditions=24,active_native_actors=0,remaining_holds=0)
 marker=dict(schema=z.SCHEMA,failed_request_id=z.FAILED,failed_request_sha256=z.FAILED_REQUEST_SHA,failed_terminal_native=proof,failed_effects=effects,failed_dual_closure=closed,independent_closure_readback_sha256='a'*64,A_dual_closure=ac,replacement_request_id=NEW,replacement_submission_nonce=NONCE,replacement_run_uuid='pv1-v4a-b4096-n294101-zero-start-recovery-01',replacement_source_sha='c'*40,control_id=CONTROL,original_stage_GPU_upper_s=57720,automatic_retry=False,maximum_replacements=1,same_original_B24_only=True,created_epoch=c.old.STOP_NEW-3600)
 return {c.FIELD:{z.PREVIOUS_A:a,z.FAILED:b},z.FIELD:marker}
def replacement(g):
 r=row();r.update(request_id=NEW,submission_nonce=NONCE,child_run_uuid=g[z.FIELD]['replacement_run_uuid'],evaluation_candidate='B_COVERAGE',source_sha='c'*40,startup_recovery_sha256=c.digest(g[z.FIELD]));return r
def test_retains_all_old_paid_rows_and_one_scientific_panel():
 g=fixture();before=deepcopy(g);assert z.validated(g)==g[z.FIELD];assert len(z.effective_rows(g))==1
 g[c.FIELD][NEW]=replacement(g);assert z.validated(g)==g[z.FIELD] and len(z.effective_rows(g))==2
 for k,r in before[c.FIELD].items():assert g[c.FIELD][k]==r
@pytest.mark.parametrize('fault',['missing_marker','missing_old_B','old_request_hash','query','load','unknown','not_terminal','exit0','erase_cost','wrong_final','not_complete','new_physics','wrong_effects','active_lease','active_actor','holds','fake_valid_B','A_partial','A_not_closed','reuse_id','reuse_nonce','other_run','late','more_replacements','auto_retry','GPU_raise','fourth_row','wrong_new_source','queued_new_request','wrong_new_arm','wrong_new_bank'])
def test_rejects_effects_unclosed_partial_reuse_or_expansion(fault):
 g=fixture();m=g[z.FIELD];old=g[c.FIELD][z.FAILED];r=replacement(g)
 if fault=='missing_marker':g.pop(z.FIELD);g[c.FIELD][NEW]=r
 if fault=='missing_old_B':g[c.FIELD].pop(z.FAILED)
 if fault=='old_request_hash':old['request_sha256']='f'*64
 if fault=='query':old['query_attempts']=1
 if fault=='load':old['model_load_attempts']=1
 if fault=='unknown':old['unknown_query_intents']=[0]
 if fault=='not_terminal':old['status']=c.STATES[1]
 if fault=='exit0':old['native_exit_status']=0
 if fault=='erase_cost':old['conservative_GPU_s']=0
 if fault=='wrong_final':m['failed_terminal_native']['actual_NATIVE_PROCESS_FINAL_sha256']='f'*64
 if fault=='not_complete':m['failed_effects']['effect_journal_complete']=False
 if fault=='new_physics':m['failed_effects']['new_physics']=1
 if fault=='wrong_effects':old['terminal_effect_summary_sha256']='f'*64
 if fault=='active_lease':m['failed_dual_closure']['physical_lease_status']='ACTIVE'
 if fault=='active_actor':m['failed_dual_closure']['active_native_actors']=1
 if fault=='holds':m['failed_dual_closure']['remaining_holds']=1
 if fault=='fake_valid_B':m['failed_dual_closure']['actual_valid_conditions']=1
 if fault=='A_partial':m['A_dual_closure']['actual_valid_conditions']=23
 if fault=='A_not_closed':m['A_dual_closure']['GPU_lease_status']='ACTIVE'
 if fault=='reuse_id':m['replacement_request_id']=z.FAILED
 if fault=='reuse_nonce':m['replacement_submission_nonce']=old['submission_nonce']
 if fault=='other_run':m['replacement_run_uuid']='old-B'
 if fault=='late':m['created_epoch']=c.old.STOP_NEW
 if fault=='more_replacements':m['maximum_replacements']=2
 if fault=='auto_retry':m['automatic_retry']=True
 if fault=='GPU_raise':m['original_stage_GPU_upper_s']+=1
 if fault=='fourth_row':g[c.FIELD]['other']=row()
 if fault in ('wrong_new_source','queued_new_request','wrong_new_arm','wrong_new_bank'):
  if fault=='wrong_new_source':r['source_sha']='f'*40
  if fault=='queued_new_request':r['queue_id']=CONTROL
  if fault=='wrong_new_arm':r['evaluation_candidate']='A_CONTROL'
  if fault=='wrong_new_bank':r['noise_bank']=294102
  g[c.FIELD][NEW]=r
 with pytest.raises(PermissionError):z.validated(g)
def test_unmarked_original_behavior_stays_strict():
 g=fixture();g.pop(z.FIELD);before=deepcopy(g);assert z.validated(g) is None and z.effective_rows(g)==list(g[c.FIELD].values());assert g==before
