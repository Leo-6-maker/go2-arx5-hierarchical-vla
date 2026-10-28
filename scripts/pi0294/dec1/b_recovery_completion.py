"""Pure B1024-to2048 completion contracts; no dispatch or ledger IO.

A checkpoint audit remains responsible for streaming every original member and
running the mature optimizer/RNG/sampler validator. Metadata alone is not proof
that the job has completed or that its files have actually been read.
"""
import base64
import hashlib
import re
from . import b_heartbeat_recovery as b


def need(ok,why):
    if not ok:raise PermissionError('B_RECOVERY_COMPLETION_'+why)


def native(job,expected):
    need(job.get('job_state')=='F' and job.get('Exit_status')==0,'REAL_SUCCESSFUL_NATIVE_TERMINAL')
    identity(job,expected)


def identity(job,expected):
    req=expected['request']
    need(job.get('Job_Owner','').split('@')[0]=='e1192548'
         and job.get('project')==req['project']=='CFP04-CF-007'
         and job.get('Job_Name')==req['job_name']=='pi0294_dec1_a40_train'
         and job.get('Variable_List',{}).get('PV1_REQUEST_ID')==req['request_id']
         and job.get('Variable_List',{}).get('PV1_SUBMISSION_NONCE')==req['submission_nonce']
         and job.get('Rerunable') in (False,'False') and int(job.get('run_count',1))==1
         and int(job.get('Resource_List',{}).get('ngpus',0))==1,'EXACT_BOUND_NATIVE_IDENTITY')


def effects(value,expected):
    req=expected['request'];b.lineage(req)
    want=dict(job_id=expected['job_id'],request_id=req['request_id'],request_sha256=expected['request_sha256'],
        source_sha=req['source_sha'],arm='B_COVERAGE',start_native_step=1024,last_native_step=2048,
        optimizer_updates_committed=1024,query_attempts=17,model_load_attempts=2,recipe_sha256=b.RECIPE,
        pending_updates=[],unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
    need(req['expected_native_start_step']==1024 and req['training_block']['stop_after_step']==2048
         and all(value.get(k)==v for k,v in want.items()),'EXACT_NEW1024_UPDATES17_QUERIES2_LOADS_NO_PROBE_REPLAY')


def origin(step,complete,expected):
    need(step in (1024,2048) and complete.get('full_resume') is True
         and complete.get('global_step')==step and len(complete.get('files',{}))==15,'BOTH_FULL_NATIVE_RESUME_STATES')
    wanted=(b.FAILED_JOB,b.FAILED_SOURCE) if step==1024 else (expected['job_id'],expected['request']['source_sha'])
    need((complete.get('origin_job_id'),complete.get('origin_source_sha'))==wanted,
         'ORIGINAL_FAILED_B1024_AND_NEW_RECOVERY_B2048_ORIGINS')


def audit(value,expected):
    native(value['raw_native'],expected);effects(value['effects'],expected)
    req=expected['request'];final=value['process_final'];proof=value['native_proof']
    need(proof.get('job_id')==expected['job_id'] and proof.get('job_state')=='F'
         and proof.get('Exit_status')==0 and proof.get('request_id')==req['request_id'], 'SAME_NATIVE_PROOF')
    need(final.get('status')=='A40_KNOWN_PARTIAL_FULL_STATE_PRESERVED'
         and final.get('global_step')==final.get('last_returned_training_step')==2048
         and final.get('error') is None and final.get('source_sha')==req['source_sha']
         and final.get('request_sha256')==expected['request_sha256']
         and {k:v for k,v in final.get('effects',{}).items() if k!='finished_epoch'}==
             {k:v for k,v in value['effects'].items() if k!='finished_epoch'},'FINAL_RETURNED_STATE_AND_EFFECT_JOURNAL')
    complete_states(value)
    return value


def failed_native(job,expected):
    need(job.get('job_state')=='F' and job.get('Exit_status')==1,'REAL_FAILED_NATIVE_TERMINAL')
    identity(job,expected)


def failed_audit(value,expected):
    """Saved full state can be retained after a failed process, without success.

    This contract has no continuation or admission path. F/Exit1 and the last
    returned step remain failure evidence even when all saved members validate.
    """
    failed_native(value['raw_native'],expected);effects(value['effects'],expected)
    f=value['process_final'];p=value['native_proof'];req=expected['request']
    need(p.get('job_id')==expected['job_id'] and p.get('request_id')==req['request_id']
         and p.get('job_state')=='F' and p.get('Exit_status')==1,'FAILED_PROOF_NOT_SUCCESS')
    need(f.get('status')=='FAILED_PRESERVED' and f.get('global_step')==2048
         and f.get('last_returned_training_step')==1024
         and f.get('error')==dict(type='HeartbeatLost',message='MP1_SINGLE_OWNER_HEARTBEAT_LOST_AFTER_SERVER_REVALIDATION')
         and f.get('source_sha')==req['source_sha'] and f.get('request_sha256')==expected['request_sha256']
         and {k:v for k,v in f.get('effects',{}).items() if k!='finished_epoch'}==
             {k:v for k,v in value['effects'].items() if k!='finished_epoch'}
         and value['wrapper_final'].get('returncode')==1,'EXACT_HEARTBEAT_FAILURE_AFTER_SAVE')
    need(p.get('new_resident_allocated_upper_bytes')==req['additional_output_upper_bytes'],
         'FAILED_ORIGINAL_STORAGE_BOUND_NO_REFUND')
    complete_states(value)
    return value


def complete_states(value):
    need(set(value['checkpoints'])==set(value['checkpoint_members'])=={'1024','2048'}
         and value.get('probe_count')==12 and len(value.get('probe_files',{}))==13
         and all(value.get(k)==0 for k in ('models_loaded','new_queries','new_updates','new_physics')),
         'FULL_STATES_AND_ORIGINAL_PROBES_WITHOUT_NEW_NEURAL_WORK')
    required={'optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json','trainer_state.json','statistics.json','config.json'}
    for step in (1024,2048):
        row=value['checkpoints'][str(step)];members=value['checkpoint_members'][str(step)]
        need(row.get('members')==len(members)==15 and row.get('actual_consumed_cursor')==step*32
             and required<=set(members) and any(n.startswith('processor/') for n in members)
             and any(n.endswith('.safetensors') for n in members)
             and re.fullmatch('[0-9a-f]{64}',row.get('complete_sha256','')) is not None,'COMPLETE_MEMBERS_AND_CONSUMED_CURSOR')
    need(value['checkpoints']['1024']['complete_sha256']==b.SEED_SHA
         and value['probe_result_sha256']==b.PROBE_SHA,'UNCHANGED_ORIGINAL_SEED_AND_PROBES')
    raw=base64.b64decode(value['original_probe_result_base64'],validate=True)
    need(hashlib.sha256(raw).hexdigest()==b.PROBE_SHA
         and value['probe_files'].get('evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json')==b.PROBE_SHA,
         'ACTUAL_ORIGINAL_PROBE_RESULT_BYTES')
    need(all(n.startswith('evidence/B_COVERAGE-1024/TRAIN_PROBES/')
             and re.fullmatch('[0-9a-f]{64}',v) for n,v in value['probe_files'].items()),'ORIGINAL_ARRAY_FILE_IDENTITIES')


def settled(value,expected):
    h,d=value['ledger'],value['DEC1'];b.validate_adopted(h,d)
    need(value['job_id']==expected['job_id'] and value['request_id']==expected['request']['request_id']
         and value['updates']==h['optimizer_updates']==d['optimizer_updates']==8744
         and value['unknown_effects']==h['unknown_effects']==h['pending_updates']==d['unknown_effects']==[],
         'ONE_ACTUAL_SETTLEMENT_PRESERVES_7720_PLUS1024')
    need(h['training_runs'][b.LINEAGE]['last_native_step']==2048
         and h['training_runs']['B_COVERAGE']['last_native_step']==1829
         and h['training_runs'][b.zero.HR.LINEAGE]['last_native_step']==4096,
         'NEW_B2048_WITH_OLD_PAID_B1829_AND_A4096_PRESERVED')
    rows=[x for x in h['PBS_submissions'] if x.get('request_id')==expected['request']['request_id']]
    need(len(rows)==1 and rows[0]['job_id']==expected['job_id'] and rows[0]['status']=='NATIVE_TERMINAL'
         and rows[0]['committed_optimizer_updates']==1024 and rows[0]['query_attempts']==17
         and rows[0]['reserved_optimizer_updates']==rows[0]['reserved_neural_queries']==0,
         'THIS_TERMINAL_ROW_ONLY_NOT_OLD_B1_OR_B2')
    return rows[0]
