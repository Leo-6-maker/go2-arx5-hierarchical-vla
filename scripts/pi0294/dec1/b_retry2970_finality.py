"""Accounting-only composition of a real prior PBS F and fresh native files.

No timeout is a terminal observation. This exact failed job was already
independently observed F/Exit1. PBS finished-job history is immutable (Altair
User Guide9.1.2.2); a new filesystem attestation can join its retained bytes to
the completed original-state audit. The old PBS timestamp is never restamped.
Fresh queue/project/quota admission remains mandatory for any future job.
"""
from copy import deepcopy
import hashlib,json,math
from . import b_retry2970_closeout as close

SCHEMA='FAILED2970_KNOWN_PBS_FINAL_NATIVE_FILES_ATTESTATION_V1'
ACCOUNTING_SCHEMA='VAP1_A40_KNOWN_FINAL_NATIVE_FILES_ACCOUNTING_V1'
BASIS='PRIOR_REAL_PBS_F1_PLUS_CURRENT_NATIVE_TERMINAL_ARTIFACTS_ACCOUNTING_ONLY'
HISTORY_SHA='e4d4c3b99ae080c8ec1888d0f9d8a779a43cc43ab4ab2787a1f669a0ff46b2b0'
AUDIT_SHA='3f48052b44f3097658381bdaef70a8585700b3581664d2e3082d658b016c49eb'
COMPLETED_SHA='afeb97399fb0c068bdbbbd755a0c90f3870a4a0f98571064075d3fdd29f102db'
AUDIT_PROGRAM='9ab558783e283377a29e8fd3d00148d7ee8cfde0120426130c24d6752cf88e4c'

def need(ok,why):
    if not ok:raise PermissionError('FAILED2970_KNOWN_FINALITY_'+why)

def decode(raw,expected):
    need(isinstance(raw,bytes) and hashlib.sha256(raw).hexdigest()==expected,'EXACT_RETAINED_ACTUAL_BYTES')
    return json.loads(raw)

def accounting_proof(value,readback,historical_epoch):
    proof=deepcopy(value['native_proof'])
    proof.update(schema=ACCOUNTING_SCHEMA,observed_epoch=readback['observed_epoch'],observation_basis=BASIS,
        original_PBS_observed_epoch=historical_epoch,fresh_native_scheduler_check=False,
        fresh_native_terminal_artifact_check=True,original_PBS_readback_sha256=HISTORY_SHA,
        full_state_audit_result_sha256=AUDIT_SHA,full_state_audit_completed_sha256=COMPLETED_SHA,
        terminal_artifact_readback=deepcopy(readback),new_training_admission=False)
    return proof

def validate_accounting_proof(proof,*,row,effects,now_epoch):
    """Exact opt-in kind at the mature accounting boundary, never generic F."""
    need(proof.get('schema')==ACCOUNTING_SCHEMA and proof.get('observation_basis')==BASIS
         and row.get('job_id')==close.failed.JOB and row.get('request_id')==close.failed.REQUEST
         and row.get('request_sha256')==close.failed.REQUEST_SHA and row.get('source_sha')==close.failed.SOURCE,
         'ONLY_EXACT1480817_FAILED_REQUEST_AT_ACCOUNTING_BOUNDARY')
    need(proof.get('original_PBS_readback_sha256')==HISTORY_SHA
         and proof.get('full_state_audit_result_sha256')==AUDIT_SHA
         and proof.get('full_state_audit_completed_sha256')==COMPLETED_SHA
         and proof.get('original_PBS_observed_epoch')==1792756412.9734638
         and proof.get('fresh_native_scheduler_check') is False
         and proof.get('fresh_native_terminal_artifact_check') is True
         and proof.get('new_training_admission') is False,'RETAINED_PBS_PROVENANCE_AND_EXPLICIT_FRESH_FILES_ONLY')
    rb=proof.get('terminal_artifact_readback',{})
    need(rb.get('schema')==SCHEMA and rb.get('native_scheduler_query_used') is False
         and type(rb.get('observed_epoch')) in (int,float) and rb['observed_epoch']==proof.get('observed_epoch')
         and 0<=now_epoch-rb['observed_epoch']<=120
         and rb.get('request_sha256')==close.failed.REQUEST_SHA
         and rb.get('run_manifest_sha256')==row.get('run_manifest_sha256')
         and rb.get('process_final_sha256')==proof.get('process_final_sha256')=='ccb665e0b2041930f3b46b8dcb3c9245cfe34a48cdd9ba60bcd8eb1cd53e1bec'
         and rb.get('wrapper_final_sha256')==proof.get('wrapper_final_sha256')=='899da277bacb96b476775bca19811282b237a4ca9fdf6fc46492a847d0e77916'
         and rb.get('native_effect_ledger_sha256')=='d96731acc5e9ea9a70e77438a8911db1a368aaf800f3892ed4933c8ac4cd71a6'
         and rb.get('old_owner_state')==dict(MainPID='0',ActiveState='failed',ExecMainStatus='1')
         and rb.get('audit_actor_handle_absent') is True,'SAME_CURRENT_TERMINAL_FILES_AND_INACTIVE_ACTORS')
    want=dict(job_id=close.failed.JOB,request_id=close.failed.REQUEST,request_sha256=close.failed.REQUEST_SHA,
        source_sha=close.failed.SOURCE,arm='B_COVERAGE',start_native_step=2048,last_native_step=2970,
        optimizer_updates_committed=922,query_attempts=17,model_load_attempts=2,pending_updates=[],
        unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
    need(all(effects.get(k)==v for k,v in want.items())
         and {k:v for k,v in rb.get('effects',{}).items() if k!='finished_epoch'}=={k:v for k,v in effects.items() if k!='finished_epoch'}
         and proof.get('conservative_GPU_s')==8481 and proof.get('conservative_CPU_s')==11023
         and proof.get('new_resident_allocated_upper_bytes')==70000000000 and proof.get('Exit_status')==1,
         'EXACT_KNOWN_COSTS_AND922_EFFECTS_NO_REFUND_OR_SUCCESS_RELABEL')
    return proof

def evidence(*,saved_bytes,completion_bytes,historical_native_bytes,readback,expected,
             readonly_allocation,budget_before,now_epoch):
    need(type(now_epoch) in (int,float) and math.isfinite(now_epoch)
         and 1792348196<=now_epoch<1792952996,'ORIGINAL_REAL_CLOCK')
    saved=decode(saved_bytes,AUDIT_SHA);done=decode(completion_bytes,COMPLETED_SHA)
    history=decode(historical_native_bytes,HISTORY_SHA)
    close.failed.audit(saved,expected);job=history['native_job'];close.failed.native(job,expected)
    need(job==saved['raw_native'] and job.get('Rerunable') in (False,'False')
         and int(job.get('run_count',1))==1,'PRIOR_REAL_FINAL_F1_NONRERUNNABLE_ONE_ATTEMPT')
    need(done.get('status')=='COMPLETE_INDEPENDENT_SAVED_STATE_AUDIT_NO_SETTLEMENT'
         and done.get('returncode')==0 and done.get('result_sha256')==AUDIT_SHA
         and done.get('program_sha256')==AUDIT_PROGRAM
         and type(done.get('CPU_actual_s')) in (int,float) and 0<=done['CPU_actual_s']<600,
         'REAL_COMPLETED_FINITE_ORIGINAL_STATE_AUDIT')
    need(saved.get('financial_settlement_ready') is False
         and saved['native_proof'].get('fresh_native_scheduler_check') is False
         and saved['native_proof']['observed_epoch']==history['observed_epoch']
         and saved['streaming_witnesses'].get('all_unique_inodes_independently_rehashed') is True
         and saved['streaming_witnesses'].get('all_paths_rechecked') is True,'UNMODIFIED_HISTORICAL_PBS_AND_FULL_SECOND_READ')
    need(readback.get('schema')==SCHEMA and readback.get('native_scheduler_query_used') is False
         and type(readback.get('observed_epoch')) in (int,float)
         and 0<=now_epoch-readback['observed_epoch']<=120,'ACTUAL_FRESH_FILES_READ_NOT_A_NEW_PBS_OBSERVATION')
    need(readback.get('request_sha256')==close.failed.REQUEST_SHA
         and readback.get('run_manifest_sha256')==expected['request']['run_manifest_sha256']
         and readback.get('process_final_sha256')==history['receipts']['PBS_PROCESS_FINAL.json']['sha256']==saved['native_proof']['process_final_sha256']
         and readback.get('wrapper_final_sha256')==history['receipts']['PBS_WRAPPER_FINAL.json']['sha256']==saved['native_proof']['wrapper_final_sha256']
         and readback.get('native_effect_ledger_sha256')==history['receipts']['A40_NATIVE_EFFECT_LEDGER.json']['sha256'],
         'SAME_CURRENT_REQUEST_MANIFEST_FINAL_WRAPPER_AND_EFFECT_LEDGER_BYTES')
    current=readback['effects'];close.failed.effects(current,expected)
    need({k:v for k,v in current.items() if k!='finished_epoch'}=={k:v for k,v in saved['effects'].items() if k!='finished_epoch'},
         'COMPLETE_ACTUAL922_17_2_EFFECT_JOURNAL_WITHOUT_NEW_OR_UNKNOWN_EFFECT')
    need(readback.get('old_owner_state')==dict(MainPID='0',ActiveState='failed',ExecMainStatus='1')
         and readback.get('audit_actor_handle_absent') is True,'FAILED_OWNER_AND_COMPLETED_AUDITOR_NOT_RESTARTED')
    need(isinstance(budget_before,tuple) and len(budget_before)==2,'CURRENT_PAIRED_BUDGET')
    total=close.readonly_cpu_allocation(readonly_allocation,*budget_before,expected)
    need(saved.get('prior_failed_audit_CPU_upper_s')==480 and saved['native_proof']['conservative_CPU_s']==total
         and saved['native_proof']['conservative_GPU_s']==8481,'ALL_KNOWN_FAILED_ATTEMPT_AND_READ_COSTS_RETAINED')
    value=deepcopy(saved);value['CPU_audit_actual_s']=done['CPU_actual_s']
    value.update(financial_settlement_ready=True,financial_observation_basis=BASIS,
        readonly_audit_allocation=deepcopy(readonly_allocation),terminal_finality_readback=deepcopy(readback),
        historical_native_observed_epoch=history['observed_epoch'])
    value['accounting_native_proof']=accounting_proof(saved,readback,history['observed_epoch'])
    return value

def settle_once(h,d,*,value,expected,now_epoch):
    close.failed.audit(value,expected);close.failed.prior.retry.validate_adopted(h,d)
    need(value.get('financial_settlement_ready') is True and value.get('financial_observation_basis')==BASIS,
         'EXPLICIT_ACCOUNTING_ONLY_FINALITY_COMPOSITION')
    proof=value['accounting_native_proof'];rb=value['terminal_finality_readback']
    need(proof==accounting_proof(value,rb,value['historical_native_observed_epoch'])
         and proof['fresh_native_scheduler_check'] is False and proof['new_training_admission'] is False
         and 0<=now_epoch-rb['observed_epoch']<=120,'COMPOSITE_FRESH_FILES_PROOF_NEVER_RESTAMP_OR_NEW_ADMISSION')
    need(close.readonly_cpu_allocation(value['readonly_audit_allocation'],h,d,expected)==proof['conservative_CPU_s'],
         'SAME_EXISTING_RESERVED_CPU_AT_ONCE_SETTLEMENT')
    from .a40_accounting import settle
    nh,nd=settle(h,d,request_id=close.failed.REQUEST,native_proof=proof,effects=value['effects'],now_epoch=now_epoch)
    close.validate_after(nh,nd,expected)
    return nh,nd
