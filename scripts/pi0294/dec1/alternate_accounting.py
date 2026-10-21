"""One explicitly typed A40 compatibility allocation inside the PV1 ledger.

H200_s is the historical field name. The alternate allocation is projected
conservatively 1:1 into that SAME total, not reported as physical H200 usage.
No native commands or file writes occur in this module.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
import uuid

from scripts.pi0294.pv1.resources import admit_resources
from .authority import AUTHORITY, PARENT_AUTHORITY, PARENT, R0, HARD_CLOSE

VERSION='DEC1_VANDA_CORR_SAVED_TRAIN_COMPAT_V1'
PLATFORM='VANDA_A40'
WALL=1800
PAD=60
CPU_UPPER=4*(WALL+PAD)
OUTPUT_UPPER=256*1024**2
QUERIES=17
OLD_PBS=11
OLD_GPU=20494
OLD_QUERIES=1054
TERMINAL=('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')


def fail(reason):
    raise PermissionError('DEC1_ALTERNATE_'+reason)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(v, *, integer=False):
    if type(v) not in (int,float) or not math.isfinite(v) or v<0 or integer and type(v) is not int:
        fail('FINITE_NONNEGATIVE_EXACT_COUNTER')
    return v


def is_alternate(item):
    return item.get('native_platform')==PLATFORM


def require_hopper_active_chain(value):
    if any(s.get('native_platform') is not None and s['status'] not in TERMINAL for s in value['PBS_submissions']):
        fail('HOPPER_DISPATCH_OR_SETTLE_CANNOT_CONTROL_ACTIVE_FOREIGN_SERVER_JOB')


def validate_item(s):
    if (s.get('native_platform')!=PLATFORM or s.get('scientific_admission_version')!=VERSION
            or s.get('stage')!='DEC1_SAVED_TRAIN_COMPATIBILITY'
            or s.get('authority_amendment')!=AUTHORITY
            or s.get('parent_checkpoint_manifest_sha256')!=PARENT or s.get('statistics_sha256')!=R0
            or s.get('reserved_H200_s')!=WALL or s.get('accounting_pad_s')!=PAD
            or s.get('CPU_upper_s')!=CPU_UPPER or s.get('query_limit')!=QUERIES
            or s.get('additional_output_upper_bytes')!=OUTPUT_UPPER
            or s.get('physical_GPU_kind')!='NVIDIA_A40' or s.get('physical_GPU_count')!=1
            or s.get('legacy_GPU_seconds_field_is_projection') is not True
            or s.get('project')!='CFP04-CF-007' or s.get('job_name')!='pi0294_dec1_a40'
            or re.fullmatch('[0-9a-f]{40}',s.get('source_sha','')) is None
            or re.fullmatch('[0-9a-f]{64}',s.get('request_sha256','')) is None):
        fail('EXACT_TYPED_SINGLE_MODEL_PLATFORM_AND_FIXED_SCOPE')
    for key in ('run_manifest_sha256','cold_input_audit_sha256'):
        if re.fullmatch('[0-9a-f]{64}',s.get(key,'')) is None:fail('COLD_MANIFEST_AND_AUDIT_BOUND_BEFORE_QSUB')
    for key in ('request_id','submission_nonce'):
        try: canonical=str(uuid.UUID(s[key]))
        except (ValueError,KeyError,TypeError,AttributeError):fail('CANONICAL_REQUEST_UUID')
        if canonical!=s[key]:fail('CANONICAL_REQUEST_UUID')
    job=s.get('job_id')
    if job is None:
        if s.get('status')!='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY':fail('NO_JOB_ID_ONLY_UNRESOLVED_QSUB_INTENT')
    elif re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job) is None:
        fail('EXACT_NATIVE_VANDA_SERVER_NOT_HOPPER_ID')
    if s.get('status') not in ('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE',
                              'NATIVE_RUNNING_QUALIFICATION',*TERMINAL):
        fail('KNOWN_ALTERNATE_LIFECYCLE')
    if s.get('status')=='NATIVE_QSUB_REJECTED_NO_JOB':
        fail('REJECTION_REQUIRES_SEPARATE_NATIVE_NO_JOB_RECONCILIATION_NOT_AUTOMATIC')
    if s['status']=='NATIVE_TERMINAL':
        for key in ('terminal_native_sha256','terminal_effect_summary_sha256'):
            if re.fullmatch('[0-9a-f]{64}',s.get(key,'')) is None:fail('EXACT_TERMINAL_EVIDENCE_BYTES')
        if (finite(s['conservative_H200_s'])>WALL+PAD
                or finite(s['conservative_CPU_s'])>CPU_UPPER
                or not 0<=finite(s['query_attempts'],integer=True)<=QUERIES
                or s.get('reserved_neural_queries')!=0):
            fail('TERMINAL_ALLOCATION_AND_QUERY_CHARGE')
    elif s.get('reserved_neural_queries')!=QUERIES:
        fail('ALL_SEVENTEEN_QUERIES_RESERVED_UNTIL_NATIVE_TERMINAL')
    return s


def validate_items(value):
    typed=[s for s in value['PBS_submissions'] if s.get('native_platform') is not None]
    compatibility=[s for s in typed if s.get('platform_training_version') is None]
    source=[s for s in typed if s.get('platform_training_version') is not None]
    if len(compatibility)>1:fail('ONE_COMPATIBILITY_ALLOCATION_NOT_AUTOMATIC_RETRIES')
    for s in compatibility:validate_item(s)
    if source:
        from .a40_accounting import validate_item as source_item
        if len(source)>7 or sum(s['status'] not in TERMINAL for s in source)>1:
            fail('BOUNDED_SOURCE_JOBS_AND_ONE_ACTIVE_A40_CHAIN')
        for s in source:source_item(s)
        if sum(s.get('reserved_optimizer_updates',0) for s in source)>value['reserved'].get('optimizer_updates',0):
            fail('NO_DROPPED_ALTERNATE_SOURCE_UPDATE_RESERVATION')
    if sum(s.get('reserved_neural_queries',0) for s in typed if s['status'] not in TERMINAL)>value['reserved'].get('neural_queries',0):
        fail('NO_DROPPED_EXTERNAL_QUERY_RESERVATION')
    return value


def reserve(value, request, *, now_epoch, fresh_native):
    """Return a changed copy; the caller must durably store it BEFORE qsub."""
    v=deepcopy(value);validate_items(v)
    if (v.get('schema')!='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
            or v.get('authority_release')!=PARENT_AUTHORITY or v.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or v.get('unknown_effects')!=[] or v.get('pending_updates')!=[]
            or len(v['PBS_submissions'])!=OLD_PBS or v['used']['PBS_submissions']!=OLD_PBS
            or v['used']['H200_s']!=OLD_GPU or v['generation_queries']!=OLD_QUERIES
            or v['optimizer_updates']!=0 or any(finite(n) for n in v['reserved'].values())
            or any(s['status'] not in TERMINAL for s in v['PBS_submissions'])):
        fail('EXACT_RECONCILED_PREQUALIFICATION_BASELINE_NOT_A_SECOND_OWNER')
    if (request.get('schema')!='DEC1_ALTERNATE_QUALIFICATION_REQUEST_V1'
            or request.get('scientific_admission_version')!=VERSION
            or request.get('authority_release')!=PARENT_AUTHORITY
            or request.get('authority_amendment')!=AUTHORITY
            or request.get('campaign_run_uuid')!=v['campaign_run_uuid']
            or request.get('original_T0_epoch')!=1792348196 or request.get('hard_close_epoch')!=HARD_CLOSE
            or request.get('submit_count')!=OLD_PBS+1
            or request.get('walltime_s')!=WALL or request.get('ncpus')!=4 or request.get('memory_gb')!=32
            or not 0<=now_epoch-request['created_epoch']<=1200
            or not now_epoch+WALL+PAD<1792866596
            or request.get('model_service_or_training_or_physics') is not False):
        fail('ONE_FRESH_BOUNDED_NEW_REQUEST_ORIGINAL_CLOCK')
    if (fresh_native.get('schema')!='DEC1_DUAL_SITE_EXCLUSIVE_PREQUEUE_PROOF_V1'
            or not 0<=now_epoch-fresh_native['observed_epoch']<=120
            or fresh_native.get('Hopper_owned_jobs')!=[] or fresh_native.get('Vanda_owned_jobs')!=[]
            or fresh_native.get('old_679029_terminal') is not True
            or fresh_native.get('original_parent_STOP') is not False
            or fresh_native.get('original_unknown_effects')!=[]
            or fresh_native.get('Vanda_project_eligible_and_covers_request') is not True
            or fresh_native.get('single_existing_local_owner') is not True
            or fresh_native.get('ledger_sha256')!=digest(value)
            or fresh_native.get('request_sha256')!=digest(request)):
        fail('FRESH_TWO_SITE_NATIVE_EXCLUSION_AND_SITE_PERMISSION')
    admit_resources(v['used'],v['reserved'],dict(H200_s=WALL+PAD,PBS_submissions=1,neural_queries=QUERIES),
        active_GPUs=0,requested_GPUs=1,job_wall_s=WALL,site_wall_limit_s=WALL)
    if (finite(v.get('new_storage_current_bytes',v.get('new_storage_peak_bytes',0)),integer=True)
            +finite(v.get('storage_reserved_bytes',0),integer=True)+OUTPUT_UPPER>64*1024**3):
        fail('COMBINED_GPU_SITE_OUTPUT_CAP_INCLUDES_DELIVERED_ALTERNATE_ASSETS')
    # These are predeclared ceilings, not A40 training or target timing data.
    # Preserve space for the unchanged four source phases and matrix envelope.
    if (request.get('remaining_declared_source_and_matrix_GPU_s')!=155220
            or request.get('remaining_declared_source_and_matrix_PBS')!=7
            or WALL+PAD+155220>48*3600
            or OLD_PBS+1+7>24 or 1+7>8
            or finite(request.get('DEC1_CPU_used_s'))+CPU_UPPER>96*3600
            or finite(v['CPU_s'])+CPU_UPPER>160*3600):
        fail('NONADDITIVE_48H_8PBS_CPU_AND_REMAINING_CORE_CEILINGS')
    item=dict(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',native_platform=PLATFORM,
        scientific_admission_version=VERSION,authority_amendment=AUTHORITY,
        stage='DEC1_SAVED_TRAIN_COMPATIBILITY',request_id=request['request_id'],
        submission_nonce=request['submission_nonce'],request_sha256=digest(request),source_sha=request['source_sha'],
        run_uuid=request['run_uuid'],project='CFP04-CF-007',job_name='pi0294_dec1_a40',job_id=None,
        parent_checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,
        reserved_H200_s=WALL,accounting_pad_s=PAD,CPU_upper_s=CPU_UPPER,
        query_limit=QUERIES,reserved_neural_queries=QUERIES,additional_output_upper_bytes=OUTPUT_UPPER,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,
        run_manifest_sha256=fresh_native['run_manifest_sha256'],cold_input_audit_sha256=fresh_native['cold_input_audit_sha256'],
        submitted_intent_epoch=now_epoch,global_submit_count=OLD_PBS+1)
    validate_item(item);v['PBS_submissions'].append(item);v['used']['PBS_submissions']+=1
    v['reserved']['H200_s']=WALL+PAD;v['reserved']['neural_queries']=QUERIES
    v['CPU_reserved_s']=finite(v.get('CPU_reserved_s',0))+CPU_UPPER
    v['storage_reserved_bytes']=v.get('storage_reserved_bytes',0)+OUTPUT_UPPER
    validate_items(v);return v


def bind_job(value,request_id,job_id,*,now_epoch):
    v=deepcopy(value);validate_items(v)
    row=next(s for s in v['PBS_submissions'] if s.get('request_id')==request_id)
    if (not is_alternate(row) or row['status']!='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY'
            or row.get('job_id') is not None or re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id) is None):
        fail('ONE_EXACT_NATIVE_REPLY_NO_SECOND_SUBMIT')
    row.update(job_id=job_id,status='SUBMITTED_PENDING_NATIVE',native_reply_epoch=now_epoch)
    validate_items(v);return v


def settle(value,request_id,*,native_proof,effects,now_epoch):
    """Terminal native evidence and durable event journal, never estimated Q time."""
    v=deepcopy(value);validate_items(v)
    row=next(s for s in v['PBS_submissions'] if s.get('request_id')==request_id)
    if row['status']=='NATIVE_TERMINAL':fail('ALREADY_SETTLED_NO_SECOND_CHARGE')
    if (not is_alternate(row) or native_proof.get('schema')!='DEC1_NATIVE_VANDA_TERMINAL_V1'
            or native_proof.get('job_id')!=row.get('job_id') or native_proof.get('job_state')!='F'
            or native_proof.get('project')!=row['project'] or native_proof.get('job_name')!=row['job_name']
            or native_proof.get('owner')!='e1192548' or native_proof.get('ngpus')!=1
            or native_proof.get('request_id')!=request_id or native_proof.get('submission_nonce')!=row['submission_nonce']
            or native_proof.get('recorded_process_terminal') is not True
            or not 0<=now_epoch-native_proof['observed_epoch']<=120):
        fail('EXACT_OWN_TERMINAL_NATIVE_REQUEST_JOB_AND_PROCESS')
    gpu=finite(native_proof['conservative_GPU_s']);cpu=finite(native_proof['conservative_CPU_s'])
    if gpu>WALL+PAD or cpu>CPU_UPPER:fail('ALLOCATION_BOUND_VIOLATION_REQUIRES_EXPLICIT_UNKNOWN_RECONCILIATION')
    if (effects.get('schema')!='DEC1_SAVED_TRAIN_EFFECT_SUMMARY_V1'
            or effects.get('request_id')!=request_id or effects.get('job_id')!=row['job_id']
            or effects.get('request_sha256')!=row['request_sha256']
            or effects.get('physical_or_optimizer_effects')!=0
            or type(effects.get('query_attempts')) is not int or not 0<=effects['query_attempts']<=QUERIES
            or type(effects.get('model_load_attempts')) is not int or not 0<=effects['model_load_attempts']<=1
            or not isinstance(effects.get('unknown_query_intents'),list)
            or len(set(effects['unknown_query_intents']))!=len(effects['unknown_query_intents'])
            or any(type(i) is not int for i in effects['unknown_query_intents'])
            or not set(effects['unknown_query_intents'])<=set(range(effects['query_attempts']))
            or effects['query_attempts'] and effects['model_load_attempts']!=1
            or effects['model_load_attempts'] and gpu<=0):
        fail('ACTUAL_BOUND_EVENT_JOURNAL_NO_PHYSICAL_EFFECTS')
    v['used']['H200_s']+=gpu;v['reserved']['H200_s']-=WALL+PAD
    v['reserved']['neural_queries']-=QUERIES
    v['generation_queries']+=effects['query_attempts'];v['used']['neural_queries']=v['generation_queries']
    v['model_loads']=v.get('model_loads',0)+effects['model_load_attempts']
    v['CPU_s']+=cpu;v['CPU_reserved_s']-=CPU_UPPER;v['storage_reserved_bytes']-=OUTPUT_UPPER
    if effects['unknown_query_intents']:
        v['unknown_effects'].append(dict(kind='ALTERNATE_UNSETTLED_QUERY_INTENTS_NO_RETRY',
            request_id=request_id,indices=effects['unknown_query_intents']))
    row.update(status='NATIVE_TERMINAL',conservative_H200_s=gpu,conservative_CPU_s=cpu,
        physical_A40_allocated_s=gpu,query_attempts=effects['query_attempts'],reserved_neural_queries=0,
        terminal_native_sha256=digest(native_proof),terminal_effect_summary_sha256=digest(effects),
        native_exit_status=native_proof['Exit_status'],settled_epoch=now_epoch)
    validate_items(v);admit_resources(v['used'],v['reserved'],{})
    return v
