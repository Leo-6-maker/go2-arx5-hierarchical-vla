"""Exact final B retry audit contracts; no IO, dispatch or completion inference.

The caller must stream all three15-member states through the original resume
validator, preserving the different1024/2048 origins. These contracts require
that evidence and the actual current native process terminal and owner exit.
"""
import base64
import hashlib
import re
from . import b_heartbeat_recovery as b,b_saved_state_reuse as saved,b_system_startup_retry as retry
from .b_recovery_completion import identity

SOURCE='36b5ed27d41f71a74767c17f0e028149c34ac8a1'
JOB='1480817.stdct-mgmt-02'
REQUEST='28efd26a-f579-454b-b14a-a4c9847cbfa2'
REQUEST_SHA='8e437f152edf4c2f3f28c11795d31594b65fca9f6fb91666e8017edb13e3be24'
NONCE='29952496-9cc0-4d33-a188-282dfa2d38c0'

def need(ok,why):
 if not ok:raise PermissionError('B_RETRY4096_COMPLETION_'+why)

def expected(value):
 r=value['request'];need(value['job_id']==JOB and value['request_sha256']==REQUEST_SHA
  and r.get('request_id')==REQUEST and r.get('source_sha')==SOURCE and r.get('submission_nonce')==NONCE
  and r.get('expected_native_start_step')==2048 and r.get('training_block',{}).get('stop_after_step')==4096
  and r.get('CPU_upper_s')==54000 and r.get('additional_output_upper_bytes')==70000000000
  and retry.request(r),'EXACT_CURRENT_UNIQUE_B2048_TO4096_REQUEST')
 return r

def owner(state):
 need(state==dict(MainPID='0',ActiveState='inactive',ExecMainStatus='0'),'ACTUAL_NATURAL_OWNER_SUCCESSFUL_EXIT')

def native(job,value):
 expected(value);identity(job,value)
 need(job.get('job_state')=='F' and job.get('Exit_status')==0,'ACTUAL_NATIVE_F_EXIT0_NOT_OBSERVATION_TIMEOUT')

def effects(e,value):
 r=expected(value)
 want=dict(job_id=JOB,request_id=REQUEST,request_sha256=REQUEST_SHA,source_sha=SOURCE,arm='B_COVERAGE',
  start_native_step=2048,last_native_step=4096,optimizer_updates_committed=2048,query_attempts=17,
  model_load_attempts=2,recipe_sha256=b.RECIPE,pending_updates=[],unknown_query_intents=[],
  effect_journal_complete=True,new_physics=0,new_episodes=0)
 need(all(e.get(k)==v for k,v in want.items()),'ACTUAL2048_UPDATES_SAVED17_COMPAT_QUERIES2_LOADS_NOT_RESERVED29')

def origin(step,complete,value):
 expected(value)
 origins={1024:(b.FAILED_JOB,b.FAILED_SOURCE),2048:(saved.JOB,saved.SOURCE),4096:(JOB,SOURCE)}
 need(step in origins and complete.get('global_step')==step and complete.get('full_resume') is True
  and len(complete.get('files',{}))==15 and (complete.get('origin_job_id'),complete.get('origin_source_sha'))==origins[step],
  'ORIGINAL_FAILED1024_RECOVERED2048_CURRENT4096_ORIGINS_SEPARATE')

def audit(v,value):
 r=expected(value);native(v['raw_native'],value);effects(v['effects'],value)
 f=v['process_final'];p=v['native_proof'];wrapper=v['wrapper_final']
 need(p.get('job_id')==JOB and p.get('request_id')==REQUEST and p.get('job_state')=='F' and p.get('Exit_status')==0
  and p.get('recorded_process_terminal') is True and p.get('new_resident_allocated_upper_bytes')==r['additional_output_upper_bytes'],
  'REAL_PROCESS_FINAL_FULL_OUTPUT_BOUND_NO_RETIREMENT_CREDIT')
 need(f.get('status')=='A40_SOURCE_4096_COMPLETE' and f.get('global_step')==f.get('last_returned_training_step')==4096
  and f.get('error') is None and f.get('request_sha256')==REQUEST_SHA and f.get('source_sha')==SOURCE
  and f.get('job_id')==wrapper.get('job_id')==JOB and wrapper.get('returncode')==0
  and {k:z for k,z in f.get('effects',{}).items() if k!='finished_epoch'}=={k:z for k,z in v['effects'].items() if k!='finished_epoch'},
  'EXACT_RETURNED4096_WRAPPER_AND_ACTUAL_EFFECTS')
 need(set(v['checkpoints'])==set(v['checkpoint_members'])=={'1024','2048','4096'}
  and v.get('probe_count')==12 and len(v.get('probe_files',{}))==13
  and v.get('new_1024_probe_queries')==0 and v.get('imported1024_and2048_origin_preserved') is True
  and all(v.get(k)==0 for k in ('models_loaded','new_queries','new_updates','new_physics')),'THREE_FULL_STATES_ORIGINAL_PROBES_READONLY')
 required={'optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json','trainer_state.json','statistics.json','config.json'}
 for step in (1024,2048,4096):
  row=v['checkpoints'][str(step)];members=v['checkpoint_members'][str(step)]
  need(row.get('members')==len(members)==15 and row.get('actual_consumed_cursor')==step*32
   and required<=set(members) and any(n.startswith('processor/') for n in members)
   and any(n.endswith('.safetensors') for n in members)
   and all(re.fullmatch('[0-9a-f]{64}',s) for s in members.values())
   and re.fullmatch('[0-9a-f]{64}',row.get('complete_sha256','')) is not None,'FULL_OPTIMIZER_SCHEDULER_RNG_SAMPLER_PROCESSOR_MODEL_MEMBERS')
 need(v['checkpoints']['1024']['complete_sha256']==b.SEED_SHA and v['checkpoints']['2048']['complete_sha256']==saved.CHECKPOINT
  and v.get('probe_result_sha256')==b.PROBE_SHA,'ORIGINAL_B1024_B2048_AND_PROBE_IDENTITIES')
 raw=base64.b64decode(v['original_probe_result_base64'],validate=True)
 need(hashlib.sha256(raw).hexdigest()==b.PROBE_SHA and v['probe_files'].get('evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json')==b.PROBE_SHA
  and all(n.startswith('evidence/B_COVERAGE-1024/TRAIN_PROBES/') and re.fullmatch('[0-9a-f]{64}',s) for n,s in v['probe_files'].items()),'ORIGINAL12_REAL_PROBE_ARRAYS_AND_RESULT_BYTES')
 return v

def settled(v,value):
 expected(value);h,d=v['ledger'],v['DEC1'];retry.validate_adopted(h,d)
 need(v['job_id']==JOB and v['request_id']==REQUEST and v['updates']==h['optimizer_updates']==d['optimizer_updates']==10792
  and v['unknown_effects']==h['unknown_effects']==h['pending_updates']==d['unknown_effects']==[], 'ACTUAL_CUMULATIVE8744_PLUS2048_NOT_OLD9987')
 need(h['training_runs'][b.LINEAGE]['last_native_step']==4096 and h['training_runs']['B_COVERAGE']['last_native_step']==1829
  and h['training_runs'][b.zero.HR.LINEAGE]['last_native_step']==4096,'ORIGINAL_B1829_LOST805_AND_A4096_PRESERVED')
 rows=[r for r in h['PBS_submissions'] if r.get('request_id')==REQUEST]
 need(len(rows)==1 and rows[0]['job_id']==JOB and rows[0]['request_sha256']==REQUEST_SHA and rows[0]['source_sha']==SOURCE
  and rows[0]['status']=='NATIVE_TERMINAL' and rows[0]['native_exit_status']==0
  and rows[0]['committed_optimizer_updates']==2048 and rows[0]['query_attempts']==17
  and rows[0]['reserved_optimizer_updates']==rows[0]['reserved_neural_queries']==0,'EXACT_THIS_TERMINAL_EFFECT_ROW')
 need(not any(h['reserved'].values()) and h['CPU_reserved_s']==h['storage_reserved_bytes']==0
  and all(d.get(k,0)==0 for k in ('GPU_reserved_s','CPU_reserved_s','optimizer_updates_reserved','neural_queries_reserved','GPU_site_output_reserved_bytes')),'TRUE_ALL_RELEASED_RESERVATIONS')
 return rows[0]
