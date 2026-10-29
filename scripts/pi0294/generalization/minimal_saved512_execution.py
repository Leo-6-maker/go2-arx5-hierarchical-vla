"""One new original-GLOBAL row, with old paid states retained as inputs."""
import base64,json,math,re
from copy import deepcopy
from . import minimal_execution as e,minimal_saved512_contract as c,minimal_saved512_recovery_scope as authority
from . import minimal_remaining_execution as previous
from .minimal_validation_cache import exact_ledger_validation
from .pbs_count_authority import history_upper

FIELD='ARX5_minimal_saved512_inference_and_raw_E2_execution'
VERSION=c.VERSION;PROFILE=c.PROFILE;RUN=c.RUN
SCHEMA='ARX5_SAVED512_INFERENCE_AND_RAW_E2_ORIGINAL_GLOBAL_INDEX_V1'
PHYSICAL='plans/MINIMAL_SAVED512_PHYSICAL_CUSTODY.private.json'
PHYSICAL_SCHEMA='ARX5_SAVED512_ALL_PAID_ROOTS_AND_NEW70GB_CUSTODY_V1'
PAID_OUTPUT=144294967296
need=e.need


def native_resources():return deepcopy(authority.NATIVE)


def fixed_request_fields():
    v=e.fixed_request_fields();v.update(schema='ARX5_SAVED512_INFERENCE_AND_RAW_E2_REQUEST_V1',
        platform_training_version=VERSION,profile=PROFILE,run_uuid=RUN,walltime_s=28800,accounting_pad_s=120,
        reserved_H200_s=28920,CPU_upper_s=78000,prior_readonly_preparation_CPU_upper_s=0,
        additional_output_upper_bytes=0,paid_output_allowance_bytes=PAID_OUTPUT,new_output_bound_bytes=70000000000,
        optimizer_update_limit=512,query_limit=62,model_load_limit=5,session_plan=c.sessions(),
        native_session_plan_version='SAVED_E1_512_INFERENCE_ONLY_THEN_RAW_E2_512_V1',
        completion_scope_grant_sha256=authority.GRANT_SHA,closed_predecessor_request_id=authority.REQUEST,
        closed_predecessor_row_sha256=authority.ROW,closed_predecessor_marker_sha256=authority.MARKER,
        saved512_result_sha256=c.RESULT,saved512_OS_exit_sha256=c.OS_EXIT,retained_old_native=deepcopy(c.OLD_PAID),
        physical_guard_policy=PHYSICAL_SCHEMA,old_sessions_never_reexecuted=True)
    return v


def request(v,*,central=False):
    need(all(v.get(k)==x and type(v.get(k))is type(x)for k,x in fixed_request_fields().items()),
        'SAVED512_EXACT_FIXED0_PLUS512_AND62_5_REQUEST')
    old=deepcopy(v);old.update(e.fixed_request_fields());e.request(old,central=central)
    need(v['source_sha']!=c.SOURCE and v['request_id']!=authority.REQUEST,
        'SAVED512_DISTINCT_COLD_SOURCE_AND_NEW_REQUEST_NOT_REPLAY')
    for k in('physical_custody_sha256','effective_queue_proof_sha256'):
        need(type(v.get(k))is str and re.fullmatch('[0-9a-f]{64}',v[k]),'SAVED512_EXACT_SEALED_'+k)
    return v


def select_profile(limits,r):
    request(r);need(limits.get('walltime')=='08:00:00','SAVED512_EXACT_EIGHT_HOUR_NATIVE_PROFILE')
    old=deepcopy(r);old.update(e.fixed_request_fields());view=dict(limits,walltime='16:30:00')
    # All capacity/chunk/device checks are the original strict native parser.
    return e.select_profile(view,old)


def project_envelope(r):
    request(r);return dict(GPU_hours=r['reserved_H200_s']/3600,
        CPU_hours=36*r['reserved_H200_s']/3600,reserved_CPU_including_prior_preparation_s=r['CPU_upper_s'],
        actual_process_CPU_upper_s=r['CPU_upper_s'],active_CPU_limit=8,project_hold_is_not_experiment_CPU_spend=True)


def process_CPU_limits(r):
    request(r);return dict(parent_RLIMIT_CPU_s=77160,isolated_child_CPU_upper_s=600,
        terminal_observer_CPU_upper_s=240,prior_readonly_preparation_CPU_upper_s=0,whole_reserved_CPU_upper_s=78000,new_CPU_budget=0)


def basic(g):
    rows=[r for r in g['PBS_submissions']if r.get('platform_training_version')==VERSION];m=g.get(FIELD)
    if m is None:
        need(not rows,'SAVED512_ROW_REQUIRES_EXPLICIT_ORIGINAL_GLOBAL_INDEX');return None
    need(len(rows)==1 and m.get('schema')==SCHEMA and m.get('version')==VERSION,
        'SAVED512_ONE_EXACT_BOUNDED_RECOVERY_ROW')
    row=rows[0];request(row,central=True);rb=base64.b64decode(m['request_base64'],validate=True);saved=json.loads(rb);request(saved)
    need(e.sha(rb)==m['request_sha256']==row['request_sha256']
        and all(row.get(k)==x and type(row.get(k))is type(x)for k,x in saved.items())
        and set(row)-set(saved)<=previous.TERMINAL_KEYS|{'request_sha256','status','job_id','submitted_epoch'},
        'SAVED512_ORIGINAL_REQUEST_BYTES_NOT_RESTAMPED')
    b=c.binding(base64.b64decode(m['saved_result_base64'],validate=True),base64.b64decode(m['saved_exit_base64'],validate=True))
    need(m['saved_binding']==b and m['request_id']==row['request_id']and m['source_sha']==row['source_sha']
        and m['run_manifest_sha256']==row['run_manifest_sha256']and m['scope_grant_sha256']==authority.GRANT_SHA
        and m['physical_custody_sha256']==row['physical_custody_sha256']==e.sha(e.raw(m['physical_custody']))
        and m['native_hold']==native_resources()and m['control_paid_here']==e._vector({})
        and m['old_paid_gross_and_history_unchanged']is True,
        'SAVED512_REAL_AUDITED_INPUTS_AND_SEPARATE_OWN_HOLD')
    need(m['before']['PBS_submissions']==39 and row['submit_count']==40
        and g['PBS_submissions'][39]==row and g['used']['PBS_submissions']==len(g['PBS_submissions'])
        and 40<=len(g['PBS_submissions'])<=history_upper(g,43) and(row['status']==e.STATES[2]or len(g['PBS_submissions'])==40),
        'SAVED512_ONE_PBS40_PLUS_MATRIX3_WITHOUT_REWRITING_OLD_ROWS')
    clock=e.authority.validate(g)
    need(all(row[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=row['created_epoch']<=m['registered_epoch']==row['submitted_epoch']
        and m['registered_epoch']+row['reserved_H200_s']<clock['stop_new_training_epoch'],
        'SAVED512_ORIGINAL_CLOCK_AND_FULL_SAVE_WINDOW')
    return m,row


@exact_ledger_validation
def validate(g,*,generation=None):
    need(generation is None,'SAVED512_EXPLICIT_RECOVERY_NOT_ZERO_EFFECT_GENERATION')
    selected=basic(g)
    if selected is None:return None
    m,row=selected;authority.validate(g);authority.predecessor(g)
    need(m['scope_adoption_sha256']==e.sha(e.raw(g[authority.FIELD]))
        and m['original_package_sha256']==e.sha(e.raw(g[e.package.FIELD])),
        'SAVED512_EXACT_ADOPTED_CONTROL_AND_OLD_PACKAGE_BYTES')
    from . import minimal_saved512_physical as physical
    physical.validate_custody(m['physical_custody'],row,saved_binding=m['saved_binding'])
    physical.check_proof(m['physical_preflight'],m['physical_custody'],row,
        now_epoch=m['registered_epoch'],observed_epoch=m['physical_observed_epoch'])
    need(all(g[k]>=m['before'][k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads'))
        and g['used']['H200_s']>=m['before']['H200_s'],'SAVED512_NO_COST_REFUND_OR_ENDPOINT_ERASURE')
    if row['status']==e.STATES[2]:
        from .minimal_saved512_terminal import validate_terminal
        validate_terminal(g,row,m)
    else:need(m['status']=='REGISTERED'and m['completed_sessions']==[]and 'terminal'not in m,'SAVED512_PREPARED_IS_NOT_COMPLETE')
    held=forecast_without_validate(g,m,row);old=g[e.original.PACKAGE]['remaining_resources']
    need(all(g['reserved'][k]>=old[k]+held[k]for k in e.package.TRACKED)
        and g['CPU_reserved_s']>=old['Hopper_CPU_s']+held['Hopper_CPU_s']
        and g['storage_reserved_bytes']>=old['Hopper_storage_bytes']+held['Hopper_storage_bytes'],
        'SAVED512_RETAIN_ALL_OLD_EVAL_ESCROW_AND_NEW_COMPUTE')
    return m


def forecast_without_validate(g,m,row):
    v=previous.forecast_remaining(g)
    if row['status']!=e.STATES[2]:
        for k,x in native_resources().items():
            if k!='PBS_submissions':v[k]+=x
    return v


def forecast_remaining(g):
    return previous.forecast_remaining(g)if validate(g)is None else forecast_without_validate(g,*basic(g))


def rows(g):return []if validate(g)is None else[basic(g)[1]]
def current_id(g):return basic(g)[1]['request_id']
def terminal_CPU_prior(g,row):
    need(basic(g)[1]==row,'SAVED512_ONLY_OWN_NEW_CPU');return 0


def register(global_bytes,local_bytes,*,request_bytes,result_bytes,exit_bytes,physical_custody,
             physical_proof,physical_observed_epoch,fresh_readback,now_epoch):
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.dec1 import vanda_campaign as queue
    from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
    from .minimal_remaining_concurrency import cold_Core_rows
    from . import minimal_saved512_physical as physical
    g=json.loads(global_bytes);r=json.loads(request_bytes);request(r);check(g);authority.predecessor(g)
    need(authority.validate(g)is not None and FIELD not in g
        and g['pending_updates']==g['unknown_effects']==[]
        and all(x['status']in e.original.TERMINAL for x in g['PBS_submissions']),
        'SAVED512_ONLY_AFTER_TRUE_CLOSED_PREDECESSOR_NO_UNKNOWN_CLEAR')
    b=c.binding(result_bytes,exit_bytes);cold_Core_rows(g)
    actual=queue.queue_rows(fresh_readback['complete_queue'],now_epoch=now_epoch);own,outside,unknown=queue.classify(g,actual)
    checked_outside(actual,outside,base64.b64decode(fresh_readback['independent_confirmation_base64'],validate=True))
    need(own==unknown==[]and 0<=now_epoch-fresh_readback['observed_epoch']<=120
        and fresh_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh_readback['complete_queue_verified']is True and fresh_readback['old_workers_terminal']is True
        and fresh_readback['old_closed_science_lease_sha256']==authority.LEASE,
        'SAVED512_FRESH_COMPLETE_NATIVE_QUEUE_CANONICAL_AND_REAL_OLD_CLOSE')
    need(all(x.get('request_id')!=r['request_id']and x.get('submission_nonce')!=r['submission_nonce']
        for x in[*g['PBS_submissions'],*g.get(e.original.CORE_ROWS,{}).values()]),'SAVED512_NEW_UNIQUE_BOTH_SITE_IDENTITY')
    physical.validate_custody(physical_custody,r,saved_binding=b)
    physical.check_proof(physical_proof,physical_custody,r,now_epoch=now_epoch,observed_epoch=physical_observed_epoch)
    v=deepcopy(g);v['PBS_submissions'].append(dict(r,request_sha256=e.sha(request_bytes),status=e.STATES[0],job_id=None,submitted_epoch=now_epoch));v['used']['PBS_submissions']+=1
    for k in e.package.TRACKED:v['reserved'][k]+=native_resources()[k]
    v['CPU_reserved_s']+=78000
    before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads')}
    before.update(H200_s=g['used']['H200_s'],PBS_submissions=g['used']['PBS_submissions'])
    v[FIELD]=dict(schema=SCHEMA,version=VERSION,request_id=r['request_id'],request_sha256=e.sha(request_bytes),request_base64=base64.b64encode(request_bytes).decode(),
        source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],saved_result_base64=base64.b64encode(result_bytes).decode(),saved_exit_base64=base64.b64encode(exit_bytes).decode(),saved_binding=b,
        scope_grant_sha256=authority.GRANT_SHA,scope_adoption_sha256=e.sha(e.raw(g[authority.FIELD])),original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),
        native_hold=native_resources(),control_paid_here=e._vector({}),physical_custody=deepcopy(physical_custody),physical_custody_sha256=r['physical_custody_sha256'],
        physical_preflight=deepcopy(physical_proof),physical_observed_epoch=physical_observed_epoch,before=before,registered_epoch=now_epoch,status='REGISTERED',completed_sessions=[],old_paid_gross_and_history_unchanged=True)
    v['revision']+=1;validate(v);check(v);future=e.scope.first512_forecast(e.raw(v),local_bytes)
    need(future['cap_changes_required']=={},'SAVED512_COMPLETE_ALL14_MATRIX_FINAL_BEFORE_REGISTER')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,native_admission=False)


def bind(g,request_id,job_id,*,now_epoch):
    validate(g);v=deepcopy(g);_,r=basic(v)
    need(r['request_id']==request_id and r['status']==e.STATES[0]and r['job_id']is None,'SAVED512_BIND_ONCE_ONLY')
    r.update(status=e.STATES[1],job_id=job_id,native_reply_epoch=now_epoch);v['revision']+=1
    from scripts.pi0294.pv1.neural_ledger import check
    validate(v);check(v);return v


def settle(g,request_id,**kwargs):
    from .minimal_saved512_terminal import settle as once
    return once(g,request_id,**kwargs)
