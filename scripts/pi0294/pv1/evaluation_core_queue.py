"""The two original panels may share one finite resident serial queue.

Pure lifecycle proposals. A queue intent is UNKNOWN, not a native job. Only a
live server owner acknowledgement accepts both exact requests before stop-new.
No GPU effects are allowed until a separate real process bind and all original
native qualification gates. Persistence must use the original once IO.
"""
from copy import deepcopy
import re

from . import evaluation_core_accounting as c

MODE='CORE_RESIDENT_SERIAL_AB_STAGE_V1'
FIELD='Core_native_GPU_stage_queue'


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_QUEUE_'+why)


def clock(work,*,now_epoch):
    """Both requests are accepted now; only execution, not submission, waits."""
    jobs=work['jobs'];wall=sum(j['walltime_s']+j['accounting_pad_s'] for j in jobs)
    need(len(jobs)==2 and [j['candidate'] for j in jobs]==list(c.stage.CANDIDATES)
         and all(j['noise_bank']==294101 and j['conditions']==24 for j in jobs),
         'EXACT_ORIGINAL_TWO_WHOLE_PANELS')
    return dict(first_stage_remaining_wall_upper_s=wall,
        both_stage_panels_fit_original_physics_window=now_epoch+wall<=c.old.STOP_PHYSICS,
        last_stage_submission_fits_original_stop_new=c.old.T0<=now_epoch<c.old.STOP_NEW,
        dispatch_mode=MODE,queued_requests_are_not_running=True,
        original_matrix_and_final_resources_protected=True,
        all_protected_future_jobs_wall_completion_promised=False)


def plan(records):
    need(len(records)==2 and [r['evaluation_candidate'] for r in records]==list(c.stage.CANDIDATES),
         'EXACT_AB_ORDER_NO_OTHER_PANEL_OR_CANDIDATE')
    first=records[0];queue_id=first.get('queue_id')
    need(c.canonical_uuid(queue_id) and all(r.get('queue_id')==queue_id
         and r.get('queue_position')==i and type(r['queue_position']) is int
         and r.get('queue_dispatch_mode')==MODE and r['source_sha']==first['source_sha']
         and r['GPU_uuid']==first['GPU_uuid'] for i,r in enumerate(records)),
         'ONE_EXACT_SERVER_QUEUE_SOURCE_AND_SINGLE_DEVICE')
    # Final request hashes are in the native ACK. This fixed plan contains the
    # pre-existing identities so it can itself be sealed into both requests.
    keys=('request_id','submission_nonce','child_run_uuid',
        'evaluation_candidate','checkpoint_manifest_sha256','planned_panel_sha256','created_epoch')
    return dict(schema=MODE,queue_id=queue_id,source_sha=first['source_sha'],GPU_uuid=first['GPU_uuid'],
        campaign_run_uuid=c.scope.CAMPAIGN,original_T0_epoch=c.old.T0,stop_new_epoch=c.old.STOP_NEW,
        stop_physics_epoch=c.old.STOP_PHYSICS,hard_close_epoch=c.old.HARD_CLOSE,
        maximum_running_GPU_processes=1,automatic_retry=False,maximum_tasks=2,
        requests=[{k:r[k] for k in keys} for r in records])


def owner(proof,queue,*,now_epoch):
    need(type(proof) is dict and proof.get('hostname')=='corelab-4080s'
         and type(proof.get('uid')) is int and proof['uid']==2009
         and proof.get('source_sha')==queue['plan']['source_sha']
         and proof.get('queue_id')==queue['plan']['queue_id']
         and proof.get('queue_plan_sha256')==c.digest(queue['plan'])
         and proof.get('process_state')=='RUNNING'
         and 0<=c.finite(now_epoch)-c.finite(proof.get('observed_epoch'))<=120
         and c.old.T0<=c.finite(proof.get('started_epoch'))<c.old.STOP_NEW
         and proof['started_epoch']<=proof['observed_epoch']<c.old.STOP_PHYSICS,
         'LIVE_ACTUAL_RESIDENT_OWNER_NOT_PREPARED_FILES_OR_EXPIRED_POLL')
    handle=c.process_handle(proof.get('boot_id'),proof.get('pid'),proof.get('proc_start'))
    need(proof.get('owner_handle')==handle
         and proof.get('native_unit')=='pv1-core-'+queue['plan']['queue_id']+'.service'
         and proof.get('native_unit_MainPID')==proof['pid']
         and proof.get('native_unit_ActiveState')=='active'
         and proof.get('native_queue_bytes_sha256')==c.digest(queue['plan']),
         'ACTUAL_OWN_UNIT_PROCESS_AND_IMMUTABLE_QUEUE_BYTES')
    return handle


def validate_global(value):
    rows=list(value.get(c.FIELD,{}).values())
    queued=[r for r in rows if any(k.startswith('queue_') for k in r)]
    if FIELD not in value:
        need(not queued and not any(r['status'] in c.STATES[4:] for r in rows),
             'NO_UNREGISTERED_QUEUE_STATUS_OR_PARTIAL_QUEUE_METADATA')
        return value
    queue=value[FIELD]
    if len(rows)>2:
        from .evaluation_core_zero_start_recovery import validated
        need(validated(value) is not None,'ONE_EXACT_KNOWN_ZERO_COLD_REPLACEMENT_ONLY')
        rows=queued
    need(len(queued)==len(rows)==2 and queue.get('phase') in ('INTENT_UNKNOWN_NO_RETRY','NATIVE_ACCEPTED'),
         'ONE_FINITE_PAIR_NO_REPLACEMENT_POOL')
    ordered=sorted(queued,key=lambda r:r['queue_position']);p=plan(ordered)
    need(queue.get('plan')==p and queue.get('plan_sha256')==c.digest(p)
         and all(r.get('queue_plan_sha256')==c.digest(p) for r in rows)
         and len({r['request_id'] for r in rows})==2,'SAME_DURABLE_TWO_REQUEST_PLAN')
    if queue['phase']=='INTENT_UNKNOWN_NO_RETRY':
        need(all(r['status']==c.STATES[4] and c.progress(r)==(0,0,0) for r in rows)
             and not any(k in queue for k in ('accepted_epoch','native_ack','owner_handle')),
             'QUEUE_INTENT_IS_NOT_NATIVE_ACCEPTANCE_OR_EFFECT')
    else:
        ack=queue['native_ack'];t=queue['accepted_epoch']
        handle=owner(ack,queue,now_epoch=ack['observed_epoch'])
        need(queue['owner_handle']==handle and ack['schema']=='PV1_CORE_RESIDENT_QUEUE_ACCEPTED_V1'
             and ack['accepted_epoch']==t and ack['started_epoch']<=t<=ack['observed_epoch']<=t+120
             and c.old.T0<=t<c.old.STOP_NEW
             and all(c.old.T0<=r['created_epoch']<=t for r in ordered)
             and ack['requests']==[{k:r[k] for k in ('request_id','request_sha256','submission_nonce')}
                 for r in ordered]
             and all(type(ack.get(k)) is int and ack[k]==0 for k in ('native_GPU_processes_started',
                 'model_loads','neural_queries','physics_integrations'))
             and all(r['status'] in (c.STATES[5],*c.STATES[1:4]) for r in rows),
             'EXACT_BEFORE_STOP_NEW_NATIVE_ACK_ZERO_EFFECTS')
    return value


def reserve_two(value,records):
    """One complete two-row escrow proposal; not a native queue receipt."""
    v=deepcopy(c.checked(value));records=deepcopy(records)
    need(c.FIELD not in v and FIELD not in v and not v['unknown_effects'] and not v['pending_updates'],
         'ONE_NEW_QUEUE_NO_RETRY_OR_PRIOR_PARTIAL_INTENT')
    p=plan(records);pin=c.digest(p)
    for r in records:
        c.row(r)
        need(r['status']==c.STATES[4] and r.get('queue_plan_sha256')==pin
             and c.progress(r)==(0,0,0),'FULL_ESCROW_BEFORE_ANY_GPU_EFFECT')
        v.setdefault(c.FIELD,{})[r['request_id']]=r
        v['reserved']['H200_s']+=c.GPU_UPPER;v['reserved']['neural_queries']+=r['query_limit']
        v['CPU_reserved_s']+=c.CPU_UPPER;v['storage_reserved_bytes']+=r['additional_output_upper_bytes']
    v[FIELD]=dict(phase='INTENT_UNKNOWN_NO_RETRY',plan=p,plan_sha256=pin)
    return c.checked(v)


def accept_two(value,*,native_ack,now_epoch):
    v=deepcopy(c.checked(value));q=v[FIELD];ack=deepcopy(native_ack)
    need(q['phase']=='INTENT_UNKNOWN_NO_RETRY','ONCE_ACCEPTANCE_NO_REDELIVERY')
    handle=owner(ack,q,now_epoch=now_epoch)
    need(ack.get('schema')=='PV1_CORE_RESIDENT_QUEUE_ACCEPTED_V1'
         and c.old.T0<=c.finite(ack.get('accepted_epoch'))<c.old.STOP_NEW
         and ack['accepted_epoch']<=now_epoch<c.old.STOP_PHYSICS,
         'FRESH_BEFORE_STOP_NEW_ACCEPTANCE')
    q.update(phase='NATIVE_ACCEPTED',native_ack=ack,accepted_epoch=ack['accepted_epoch'],owner_handle=handle)
    for r in v[c.FIELD].values():r['status']=c.STATES[5]
    return c.checked(v)


def dispatch_check(value,record,process,proof,*,now_epoch):
    q=value[FIELD]
    need(q['phase']=='NATIVE_ACCEPTED' and type(proof) is dict,'NATIVE_ACCEPTED_QUEUE_REQUIRED')
    handle=owner(proof,q,now_epoch=now_epoch)
    need(handle==q['owner_handle'] and proof.get('schema')=='PV1_CORE_RESIDENT_QUEUE_DISPATCH_V1'
         and proof.get('request_id')==record['request_id']
         and proof.get('request_sha256')==record['request_sha256']
         and proof.get('GPU_process_native_handle')==c.process_handle(process.get('boot_id'),process.get('pid'),process.get('proc_start'))
         and q['accepted_epoch']<=process['started_epoch']<=now_epoch
         and now_epoch+sum(r['walltime_s']+r['accounting_pad_s'] for r in value[c.FIELD].values()
             if r['status']!=c.STATES[3])<=c.old.STOP_PHYSICS,
         'SAME_LIVE_OWNER_ACTUAL_NEW_PROCESS_AND_REMAINING_ORIGINAL_CLOCK')
    ordered=sorted(value[c.FIELD].values(),key=lambda r:r['queue_position'])
    need(not any(r['status'] in c.STATES[1:3] for r in ordered),'SINGLE_RUNNING_GPU_PROCESS')
    if record['queue_position']==1:
        previous=ordered[0]
        need(previous['status']==c.STATES[3] and previous['native_exit_status']==0
             and proof.get('previous_terminal_native_sha256')==previous['terminal_native_sha256']
             and all(re.fullmatch('[0-9a-f]{64}',str(proof.get(k,''))) is not None
                 for k in ('previous_independent_panel_audit_sha256','previous_dual_closure_sha256')),
             'PREDECESSOR_NATURALLY_TERMINAL_INDEPENDENT_AUDIT_AND_CLOSED_RESOURCES')
    else:need(all(r['status']==c.STATES[5] for r in ordered),'FIRST_REQUEST_ONCE_IN_ORIGINAL_ORDER')
    return dict(status='PASS_CORE_SERIAL_DISPATCH_IDENTITY_ONLY',native_model_permission=False,physics_authorized=False)


def accepted_request(request,ack,plan_value,*,request_sha256):
    """An immutable native ACK joins a request; no process is inferred alive."""
    q=dict(plan=plan_value);t=c.finite(ack.get('accepted_epoch'))
    owner(ack,q,now_epoch=ack['observed_epoch'])
    fixed=dict(schema=MODE,campaign_run_uuid=c.scope.CAMPAIGN,original_T0_epoch=c.old.T0,
        stop_new_epoch=c.old.STOP_NEW,stop_physics_epoch=c.old.STOP_PHYSICS,hard_close_epoch=c.old.HARD_CLOSE,
        maximum_running_GPU_processes=1,automatic_retry=False,maximum_tasks=2,
        source_sha=request['source_sha'],GPU_uuid=request['GPU_uuid'])
    need(all(plan_value.get(k)==v and type(plan_value[k]) is type(v) for k,v in fixed.items())
         and [r['evaluation_candidate'] for r in plan_value['requests']]==list(c.stage.CANDIDATES),
         'ORIGINAL_FINITE_AB_QUEUE_AND_CLOCKS')
    i=request.get('queue_position')
    need(request.get('queue_dispatch_mode')==MODE and request.get('queue_id')==plan_value['queue_id']
         and request.get('queue_plan_sha256')==c.digest(plan_value)
         and type(i) is int and i in (0,1) and len(plan_value['requests'])==2
         and len(ack['requests'])==2 and ack['schema']=='PV1_CORE_RESIDENT_QUEUE_ACCEPTED_V1'
         and c.old.T0<=t<c.old.STOP_NEW and ack['started_epoch']<=t<=ack['observed_epoch']<=t+120
         and plan_value['requests'][i]['request_id']==request['request_id']
         and plan_value['requests'][i]['child_run_uuid']==request['run_uuid']
         and plan_value['requests'][i]['submission_nonce']==request['submission_nonce']
         and plan_value['requests'][i]['created_epoch']==request['created_epoch']<=t
         and plan_value['requests'][i]['planned_panel_sha256']==request['planned_panel_sha256']
         and plan_value['requests'][i]['evaluation_candidate']==request['evaluation_candidate']
         and plan_value['requests'][i]['checkpoint_manifest_sha256']==request['checkpoint_manifest_sha256']
         and ack['requests'][i]=={k:v for k,v in dict(request_id=request['request_id'],
             request_sha256=request_sha256,submission_nonce=request['submission_nonce']).items()}
         and all(type(ack.get(k)) is int and ack[k]==0 for k in ('native_GPU_processes_started',
             'model_loads','neural_queries','physics_integrations')),
         'EXACT_ACCEPTED_BEFORE_CUTOFF_REQUEST_NOT_LATE_NEW_WORK')
    return t
