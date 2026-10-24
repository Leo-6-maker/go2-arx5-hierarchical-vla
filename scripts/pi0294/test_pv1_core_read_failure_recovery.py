"""Synthetic CPU proof graph; never native qualification or new dispatch."""
import hashlib,json
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import evaluation_core_read_failure_recovery as r,evaluation_core_zero_start_recovery as z,evaluation_core_accounting as c
from scripts.pi0294.test_pv1_core_zero_start_recovery import fixture
from scripts.pi0294.test_pv1_core_evaluation_accounting import row
NEW='deed1234-1234-4234-8234-123456789abc';NONCE='eeed1234-1234-4234-8234-123456789abc';CONTROL='feed1234-1234-4234-8234-123456789abc'
def graph():
 g=fixture();m1=g[z.FIELD];m1['replacement_request_id']=r.FAILED;m1['replacement_source_sha']='a6991fca99f82e8dc413816e8bde6c8f0e1552ea'
 b=row();b.update(request_id=r.FAILED,request_sha256=r.REQUEST_SHA,source_sha=m1['replacement_source_sha'],status=c.STATES[3],native_exit_status=1,query_attempts=0,model_load_attempts=0,unknown_query_intents=[],conservative_GPU_s=678,conservative_CPU_s=3,charged_CPU_s=1.667072924,charged_effects={'fixture-CPU':{'kind':'CPU_s','count':1.667072924}},evaluation_candidate='B_COVERAGE',child_run_uuid='pv1-v4a-b4096-n294101-zero-start-recovery-01',reserved_neural_queries=0)
 p=dict(request_id=r.FAILED,actual_NATIVE_PROCESS_FINAL_sha256=r.FINAL_SHA,new_resident_allocated_upper_bytes=64*1024**2)
 e=dict(request_id=r.FAILED,effect_journal_complete=True,unknown_query_intents=[],optimizer_updates_committed=0,new_physics=0,new_episodes=0,query_attempts=0,model_load_attempts=0)
 b.update(terminal_native_sha256=c.digest(p),terminal_effect_summary_sha256=c.digest(e));g[c.FIELD][r.FAILED]=b
 close=dict(request_id=r.FAILED,terminal_native_sha256=c.digest(p),native_exit_status=1,GPU_lease_status='CLOSED',physical_lease_status='CLOSED',active_native_actors=0,remaining_holds=0,actual_valid_conditions=0,actual_model_loads=0,actual_queries=0,actual_physics=0,actual_training_updates=0)
 storage=dict(schema='PV1_KNOWN_ZERO_CLOSED_UNIQUE_ALLOCATED_OUTPUT_V1',request_id=r.FAILED,no_historical_charge_refunded=True,no_future_upper_deleted=True,neural=dict(conservative_new_allocated_upper_bytes=64*1024**2),physical=dict(conservative_new_allocated_upper_bytes=64*1024**2))
 g['Core_B_cold_read_failure_storage']=dict(proof_sha256=hashlib.sha256((json.dumps(storage,indent=2,allow_nan=False)+'\n').encode()).hexdigest())
 m=dict(schema=r.SCHEMA,previous_boundary_sha256=c.digest(m1),failed_request_id=r.FAILED,failed_terminal_native=p,failed_effects=e,failed_dual_closure=close,known_zero_storage=storage,exact_read_failure_diagnostic_sha256='48062b4b1175e0a582be055747b55e1596728fe116cb1df1302500b39c094eae',independent_closure_readback_sha256='a'*64,replacement_request_id=NEW,replacement_submission_nonce=NONCE,replacement_run_uuid=r.RUN,replacement_source_sha='f'*40,control_id=CONTROL,created_epoch=c.old.STOP_NEW-100,automatic_retry=False,maximum_additional_replacements=1)
 g[r.FIELD]=m;return g,m1,m

def test_exact_model_zero_read_failure_can_be_reviewed_without_erasing_cost():
 g,first,m=graph();before=deepcopy(g);assert r.validate(g,first)==m and g==before
@pytest.mark.parametrize('fault',['model','query','unknown','cost','native_not_terminal','open_lease','physics','false_cause','other_boundary','two_more','auto','late','other_run','output_refund','missing_proof'])
def test_only_this_known_closed_zero_effect_cause_is_eligible(fault):
 g,first,m=graph();b=g[c.FIELD][r.FAILED]
 if fault=='model':b['model_load_attempts']=1
 if fault=='query':b['query_attempts']=1
 if fault=='unknown':b['unknown_query_intents']=[0]
 if fault=='cost':b['conservative_GPU_s']=0
 if fault=='native_not_terminal':b['status']=c.STATES[1]
 if fault=='open_lease':m['failed_dual_closure']['physical_lease_status']='ACTIVE'
 if fault=='physics':m['failed_effects']['new_physics']=1
 if fault=='false_cause':m['exact_read_failure_diagnostic_sha256']='f'*64
 if fault=='other_boundary':m['previous_boundary_sha256']='f'*64
 if fault=='two_more':m['maximum_additional_replacements']=2
 if fault=='auto':m['automatic_retry']=True
 if fault=='late':m['created_epoch']=c.old.STOP_NEW
 if fault=='other_run':m['replacement_run_uuid']='old-run'
 if fault=='output_refund':m['known_zero_storage']['no_historical_charge_refunded']=False
 if fault=='missing_proof':g['Core_B_cold_read_failure_storage']['proof_sha256']='f'*64
 with pytest.raises(PermissionError):r.validate(g,first)
