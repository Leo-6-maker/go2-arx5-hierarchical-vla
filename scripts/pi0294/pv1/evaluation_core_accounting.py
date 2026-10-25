"""Corelab GPU effects in the original campaign; no PBS identity is invented.

Pure proposals only. Native process/GPU qualification, complete fresh remaining
resources and original once IO must independently precede any dispatch/effect.
The first 25 PBS rows and all failed training costs remain unchanged.
"""
from copy import deepcopy
from . import evaluation_core_B24_1000 as late
import math
import re
import uuid

from scripts.pi0294.dec1.alternate_accounting import finite,digest
from . import evaluation_scope as scope
from . import evaluation_stage as stage
from . import evaluation_platform as old
from .evaluation_accounting import progress,output_upper
from .decision import AUTHORITY

FIELD='Core_native_GPU_allocations'
VERSION='PV1_CORELAB_RTX4080_SOURCE_DEV_V1'
ROLE='PI0294_PV1_SHARED_POLICY_CORELAB_EVALUATION_V1'
PROFILE='4080-neural-evaluation'
STATES=('PRESTART_INTENT_UNKNOWN_NO_RETRY','BOUND_QUALIFICATION','NATIVE_RUNNING','NATIVE_TERMINAL',
    'STAGE_QUEUE_INTENT_UNKNOWN_NO_RETRY','NATIVE_QUEUED_NOT_RUNNING')
CPU_UPPER=80000
GPU_UPPER=28860
MODEL_BYTES=12576324408


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_GPU_'+why)


def canonical_uuid(value):
    try:return type(value) is str and str(uuid.UUID(value))==value and uuid.UUID(value).version==4
    except (ValueError,TypeError,AttributeError):return False


def process_handle(boot_id,pid,proc_start):
    need(canonical_uuid(boot_id) and type(pid) is int and pid>0
         and type(proc_start) is int and proc_start>0,'REAL_BOOT_PID_AND_PROCESS_START_IDENTITY')
    return 'coregpu:'+boot_id+':'+str(pid)+':'+str(proc_start)


def row(value):
    r=value;candidate=r.get('evaluation_candidate')
    need(candidate in stage.CANDIDATES,'ORIGINAL_AB_ONLY')
    expected=dict(native_platform='CORELAB_RTX4080',platform_evaluation_version=VERSION,
        authority_release=AUTHORITY,campaign_run_uuid=scope.CAMPAIGN,
        evaluation_execution_stage=stage.VERSION,evaluation_stage_plan_sha256=stage.PLAN_SHA,
        evaluation_stage_card_sha256=stage.CARD_SHA,owner_user='liuyu-nfs',owner_uid=2009,
        hostname='corelab-4080s',physical_GPU_count=1,physical_GPU_kind='RTX_4080_SUPER',
        stage='P4_DEV',noise_bank=294101,remaining_conditions=24,
        checkpoint_manifest_sha256=old.ENDPOINTS[candidate],statistics_sha256=old.R0,
        parent_checkpoint_manifest_sha256=old.PARENT,optimizer_update_limit=0,
        reserved_optimizer_updates=0,charges_DEC1_subcaps=False,new_robot_physics=0,
        walltime_s=28800,accounting_pad_s=60,reserved_GPU_s=GPU_UPPER,
        CPU_upper_s=CPU_UPPER,declared_process_CPU_limit_s=79760,active_cpu_limit=4,
        requested_memory_bytes=48*1024**3,minimum_GPU_memory_bytes=30*1024**3,
        query_limit=1937,additional_output_upper_bytes=output_upper(24)+MODEL_BYTES,
        original_T0_epoch=old.T0,hard_close_epoch=old.HARD_CLOSE,
        PBS_submissions_consumed=0,legacy_GPU_seconds_field_is_projection=True)
    need(all(r.get(k)==v and type(r[k]) is type(v) for k,v in expected.items()),
         'EXACT_ORIGINAL_STAGE_AND_NON_PBS_CORE_ENVELOPE')
    need(not any(k in r for k in ('project','submit_count','job_name','PBS_JOBID',
         'platform_training_version','training_block')),'NO_FAKE_PBS_OR_TRAINING_IDENTITY')
    need(canonical_uuid(r.get('request_id')) and canonical_uuid(r.get('submission_nonce')),
         'UNIQUE_CANONICAL_REQUEST_AND_NONCE')
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',str(r.get('child_run_uuid',''))) is not None
         and r['child_run_uuid']!=scope.CAMPAIGN,'DISTINCT_REGISTERED_CHILD')
    for key,n in (('source_sha',40),('request_sha256',64),('run_manifest_sha256',64),
            ('planned_panel_sha256',64),('prior_outcomes_sha256',64),
            ('cold_input_audit_sha256',64),('prequeue_future_admission_sha256',64)):
        need(re.fullmatch('[0-9a-f]{'+str(n)+'}',str(r.get(key,''))) is not None,
             'SEALED_PROVENANCE:'+key)
    from .evaluation_core_B24_1000 import start_limit
    need(old.T0<=finite(r.get('created_epoch'))<start_limit(r)
         and re.fullmatch('GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(r.get('GPU_uuid',''))) is not None,
         'ORIGINAL_CLOCK_AND_REAL_DEVICE_UUID')
    need(r.get('status') in STATES,'KNOWN_PROCESS_LIFECYCLE')
    handle=r.get('native_handle')
    if r['status'] in (STATES[0],STATES[4],STATES[5]):
        need(handle is None and not any(k in r for k in ('pid','proc_start','boot_id')),
             'UNBOUND_INTENT_IS_UNKNOWN_NOT_A_JOB')
    elif r['status']==STATES[3] and r.get('native_start_rejected_before_binding') is True:
        from .evaluation_core_prebind_failure import validate_row as validate_prebind_failure
        validate_prebind_failure(r)
    else:
        need(handle==process_handle(r.get('boot_id'),r.get('pid'),r.get('proc_start'))
             and finite(r.get('process_started_epoch'))>=r['created_epoch'],
             'BOUND_ACTUAL_NATIVE_PROCESS_NOT_PBS')
    q,loads,cpu=progress(r)
    if r['status']==STATES[3]:
        unknown=r.get('unknown_query_intents')
        need(type(unknown) is list and len(unknown)==len(set(unknown))
             and all(type(i) is int and 0<=i<r.get('query_attempts',0) for i in unknown),
             'EXPLICIT_UNRESOLVED_QUERIES')
        need(q<=finite(r.get('query_attempts'),integer=True)<=r['query_limit']
             and loads<=finite(r.get('model_load_attempts'),integer=True)<=2
             and cpu<=finite(r.get('conservative_CPU_s'))<=CPU_UPPER
             and finite(r.get('conservative_GPU_s'))<=GPU_UPPER
             and type(r.get('native_exit_status')) is int
             and r.get('reserved_neural_queries')==len(unknown),
             'KNOWN_FINAL_EFFECTS_AND_NO_REFUND_OF_UNKNOWN')
        for k in ('terminal_native_sha256','terminal_effect_summary_sha256'):
            need(re.fullmatch('[0-9a-f]{64}',str(r.get(k,''))) is not None,'FULL_TERMINAL_EVIDENCE')
    else:
        need(type(r.get('reserved_neural_queries')) is int
             and r['reserved_neural_queries']==r['query_limit']-q,'ALL_QUERY_ESCROW_PRESENT')
    return r


def validate_global(value):
    from .evaluation_core_B24_1000 import validate as late_validate
    late_validate(value)
    from .evaluation_core_shared_B24 import validate as shared_validate
    shared_validate(value)
    from .evaluation_core_shared_repair import validate as repair_validate
    repair_validate(value)
    if FIELD not in value:
        need('Core_native_GPU_stage_queue' not in value,'NO_ORPHAN_QUEUE_WITHOUT_REAL_ESCROW_ROWS')
        return value
    scope.validate_global(value)
    rows=value[FIELD]
    from .evaluation_core_zero_start_recovery import validated,effective_rows
    recovery=validated(value)
    scientific=effective_rows(value)
    need(type(rows) is dict and 1<=len(rows)<=(6 if 'Core_B24_shared_GPU0_owner_facts_cold_repair' in value else 5 if 'Core_B24_shared_GPU0_human_boundary' in value else 4 if 'Core_B_known_zero_scheduler_read_recovery' in value else (3 if recovery else 2)),'BOUNDED_ORIGINAL_STAGE_CORE_INDEX')
    for key,r in rows.items():
        need(key==row(r)['request_id'],'EXACT_REQUEST_INDEX')
    paired=scientific+[r for r in value['PBS_submissions']
        if r.get('platform_evaluation_version')==old.VERSION]
    need(len(paired)<=2 and all(sum(r['evaluation_candidate']==c for r in paired)<=1
         for c in stage.CANDIDATES),'ONE_PANEL_PER_ARM_ACROSS_BOTH_PLATFORMS')
    active=[r for r in rows.values() if r['status']!=STATES[3]]
    computing=[r for r in active if r['status'] not in (STATES[4],STATES[5])]
    pbs_active=sum(r['status'] not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')
         for r in value['PBS_submissions'])
    need(len(computing)+pbs_active<=1 and (not active or not pbs_active),
         'SINGLE_CAMPAIGN_GPU_ACROSS_PLATFORMS')
    from .evaluation_core_queue import validate_global as queue_check
    queue_check(value)
    for key in ('request_id','submission_nonce'):
        identities=[r.get(key) for r in [*rows.values(),*value['PBS_submissions']] if r.get(key) is not None]
        need(len(identities)==len(set(identities)),'NO_REUSED_IDENTITY:'+key)
    need(sum(r['reserved_GPU_s'] for r in active)<=value['reserved']['H200_s']
         and sum(CPU_UPPER-progress(r)[2] for r in active)<=value['CPU_reserved_s']
         and sum(r['additional_output_upper_bytes'] for r in active)<=value['storage_reserved_bytes']
         and sum(r['reserved_neural_queries'] for r in rows.values())<=value['reserved']['neural_queries'],
         'ALL_REAL_GPU_CPU_OUTPUT_QUERY_ESCROWS_PRESENT')
    need(sum(r.get('conservative_GPU_s',0) for r in rows.values() if r['status']==STATES[3])<=value['used']['H200_s']
         and sum(r.get('query_attempts',progress(r)[0]) for r in rows.values())<=value['generation_queries']
         and sum(r.get('model_load_attempts',progress(r)[1]) for r in rows.values())<=value['model_loads']
         and sum(r.get('conservative_CPU_s',progress(r)[2]) for r in rows.values())<=value['CPU_s'],
         'ALL_CORE_EFFECTS_EXIST_IN_ORIGINAL_GLOBAL_TOTALS')
    return value


def checked(value):
    from .neural_ledger import check
    return check(value,allow_unknown=True)


def reserve(value,record):
    """Escrow proposal only; complete fresh all-site admission is separate."""
    v=deepcopy(checked(value));scope.validate_global(v);r=deepcopy(row(record))
    need(r['status']==STATES[0] and r['request_id'] not in v.get(FIELD,{})
         and not any(k.startswith('queue_') for k in r)
         and not v['unknown_effects'] and not v['pending_updates'],'ONE_NEW_INTENT_NO_UNKNOWN_REPLAY')
    need(progress(r)==(0,0,0),'NO_PRETEND_PREEXISTING_EFFECTS')
    v.setdefault(FIELD,{})[r['request_id']]=r
    v['reserved']['H200_s']+=GPU_UPPER;v['reserved']['neural_queries']+=r['query_limit']
    v['CPU_reserved_s']+=CPU_UPPER;v['storage_reserved_bytes']+=r['additional_output_upper_bytes']
    return checked(v)


def bind(value,*,request_id,process,now_epoch,queue_dispatch_proof=None):
    v=deepcopy(checked(value));r=v[FIELD][request_id]
    queued=r['status']==STATES[5]
    need(r['status'] in (STATES[0],STATES[5]) and r['native_handle'] is None,'ONCE_NATIVE_BIND_NO_RESTART')
    if queued:
        from .evaluation_core_queue import dispatch_check
        dispatch_check(v,r,process,queue_dispatch_proof,now_epoch=now_epoch)
    else:need(queue_dispatch_proof is None,'NO_QUEUE_PROOF_ON_UNQUEUED_REQUEST')
    exact=dict(hostname=r['hostname'],uid=2009,GPU_uuid=r['GPU_uuid'],request_id=request_id,
        request_sha256=r['request_sha256'],submission_nonce=r['submission_nonce'],
        run_manifest_sha256=r['run_manifest_sha256'],source_sha=r['source_sha'])
    need(all(process.get(k)==x and type(process[k]) is type(x) for k,x in exact.items())
         and process.get('schema')=='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1'
         and process.get('process_state')=='RUNNING'
         and 0<=finite(now_epoch)-finite(process.get('observed_epoch'))<=120
         and r['created_epoch']<=finite(process.get('started_epoch'))
         and (queued or process['started_epoch']<late.start_limit(r))
         and process['started_epoch']<=now_epoch<old.STOP_PHYSICS,
         'CURRENT_EXACT_ACTUAL_BOOT_PID_NOT_A_PREPARED_HANDLE')
    handle=process_handle(process.get('boot_id'),process.get('pid'),process.get('proc_start'))
    r.update(status=STATES[1],native_handle=handle,boot_id=process['boot_id'],pid=process['pid'],
        proc_start=process['proc_start'],process_started_epoch=process['started_epoch'],
        native_bind_proof_sha256=digest(process))
    return checked(v)


def charge(value,*,request_id,native_handle,kind,count,effect_id):
    v=deepcopy(checked(value));r=v[FIELD][request_id]
    need(r['status'] in STATES[1:3] and r['native_handle']==native_handle,'LIVE_EXACT_NATIVE_PROCESS')
    need(kind in ('generation_queries','model_loads','CPU_s')
         and type(effect_id) is str and re.fullmatch(r'[A-Za-z0-9._:/-]{1,200}',effect_id)
         and effect_id not in r.get('charged_effects',{}),'ONE_UNIQUE_PRE_EFFECT_DEBIT')
    amount=finite(count,integer=kind!='CPU_s');q,loads,cpu=progress(r)
    need(kind=='CPU_s' or amount==1,'ONE_MODEL_OR_QUERY_ATTEMPT')
    if kind!='CPU_s':need(not v['unknown_effects'] and not v['pending_updates'],'UNKNOWN_EFFECT_FENCE')
    if kind=='generation_queries':
        need(loads>0 and q+amount<=r['query_limit'],'QUERY_AFTER_MODEL_WITHIN_ESCROW')
        r['charged_query_attempts']=q+amount;r['reserved_neural_queries']-=amount
        v['generation_queries']+=amount;v['used']['neural_queries']+=amount;v['reserved']['neural_queries']-=amount
    elif kind=='model_loads':
        need(loads+amount<=2,'BOUNDED_LOADS');r['charged_model_loads']=loads+amount;v['model_loads']+=amount
    else:
        need(cpu+amount<=CPU_UPPER,'BOUNDED_CPU');r['charged_CPU_s']=cpu+amount
        v['CPU_s']+=amount;v['CPU_reserved_s']-=amount
    r.setdefault('charged_effects',{})[effect_id]=dict(kind=kind,count=amount);r['status']=STATES[2]
    return checked(v)


def settle(value,*,request_id,native_proof,effects,now_epoch):
    v=deepcopy(checked(value));r=v[FIELD][request_id]
    need(r['status'] in STATES[1:3],'KNOWN_BOUND_TERMINAL_NOT_SETTLED_TWICE')
    p=native_proof
    exact=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_TERMINAL_V1',request_id=request_id,
        native_handle=r['native_handle'],hostname=r['hostname'],uid=2009,GPU_uuid=r['GPU_uuid'],
        boot_id=r['boot_id'],pid=r['pid'],proc_start=r['proc_start'],process_state='TERMINAL',
        request_sha256=r['request_sha256'],submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,terminal_inventory_stable=True)
    need(all(p.get(k)==x and type(p[k]) is type(x) for k,x in exact.items())
         and p.get('exit_evidence_kind') in ('SYSTEMD_EXEC_MAIN_EXIT','NATIVE_PROCESS_FINAL')
         and type(p.get('native_exit_status')) is int
         and r['process_started_epoch']<=finite(p.get('finished_epoch'))<=now_epoch
         and 0<=finite(now_epoch)-finite(p.get('observed_epoch'))<=120,
         'NATIVE_EXIT_EVIDENCE_NOT_PID_ABSENCE_OR_TIMEOUT')
    gpu=finite(p.get('conservative_GPU_s'));cpu=finite(p.get('conservative_CPU_s'))
    output=finite(p.get('new_resident_allocated_upper_bytes'),integer=True)
    need(math.ceil(p['finished_epoch']-r['process_started_epoch'])+60<=gpu<=GPU_UPPER
         and cpu<=CPU_UPPER and output<=r['additional_output_upper_bytes'],
         'FULL_NATIVE_RESERVATION_WALL_AND_CPU_OUTPUT_BOUND')
    expected=dict(schema='PV1_CORE_GPU_NATIVE_DEV_EFFECT_SUMMARY_V1',request_id=request_id,
        native_handle=r['native_handle'],request_sha256=r['request_sha256'],source_sha=r['source_sha'],
        run_manifest_sha256=r['run_manifest_sha256'],evaluation_candidate=r['evaluation_candidate'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=old.R0,
        optimizer_updates_committed=0,new_physics=0,new_episodes=0,effect_journal_complete=True)
    need(all(effects.get(k)==x and type(effects[k]) is type(x) for k,x in expected.items()),
         'FULL_SAME_NATIVE_EFFECT_JOURNAL')
    q=finite(effects.get('query_attempts'),integer=True);loads=finite(effects.get('model_load_attempts'),integer=True)
    charged,paid_loads,paid_cpu=progress(r);unknown=effects.get('unknown_query_intents')
    need(charged<=q<=r['query_limit'] and paid_loads<=loads<=2 and paid_cpu<=cpu
         and (not q or loads) and type(unknown) is list and len(unknown)==len(set(unknown))
         and all(type(i) is int and 0<=i<q for i in unknown),'NO_LOST_PAID_OR_UNKNOWN_EFFECTS')
    v['used']['H200_s']+=gpu;v['CPU_s']+=cpu-paid_cpu
    v['generation_queries']+=q-charged;v['used']['neural_queries']+=q-charged;v['model_loads']+=loads-paid_loads
    v['reserved']['H200_s']-=GPU_UPPER;v['CPU_reserved_s']-=CPU_UPPER-paid_cpu
    v['reserved']['neural_queries']-=r['reserved_neural_queries']-len(unknown)
    v['storage_reserved_bytes']-=r['additional_output_upper_bytes']
    v['new_storage_current_bytes']+=output;v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    if unknown:v['unknown_effects'].append(dict(kind='CORE_GPU_UNKNOWN_QUERY_INTENTS_NO_RETRY',
        request_id=request_id,native_handle=r['native_handle'],unknown_query_intents=deepcopy(unknown)))
    r.update(status=STATES[3],native_exit_status=p['native_exit_status'],conservative_GPU_s=gpu,
        conservative_CPU_s=cpu,query_attempts=q,model_load_attempts=loads,
        unknown_query_intents=deepcopy(unknown),reserved_neural_queries=len(unknown),
        terminal_native_sha256=digest(p),terminal_effect_summary_sha256=digest(effects),settled_epoch=now_epoch)
    return checked(v)
