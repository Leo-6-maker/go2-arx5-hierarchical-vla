"""One typed completion row in the original GLOBAL, after the audited prefix.

Pure proposals only. The old package/rows/costs stay unchanged. The planned
remaining hold is consumed once by registration, not added to the forecast a
second time. Native registration, own leases, original owner and onceIO remain.
"""
import base64,json,math,re
from copy import deepcopy
from . import minimal_execution as e,minimal_remaining_contract as r
from . import minimal_partial_completion_scope as authority,minimal_retry_execution as retry
from .minimal_validation_cache import exact_ledger_validation

FIELD='ARX5_minimal_first512_actual_saved_prefix_completion'
VERSION=r.VERSION
PROFILE=r.PROFILE
RUN='arx5-minimal-raw-parent-first512-remainder-01'
SCHEMA='ARX5_MINIMAL_REAL_SAVED_PREFIX_COMPLETION_INDEX_V1'
REQUEST_SCHEMA='ARX5_MINIMAL_REAL_SAVED_PREFIX_COMPLETION_REQUEST_V1'
PHYSICAL='plans/MINIMAL_REMAINING_PHYSICAL_CUSTODY.private.json'
PREFIX_STORAGE='plans/MINIMAL_SAVED_PREFIX_STORAGE_INVENTORY.private.json'
PHYSICAL_SCHEMA='ARX5_MINIMAL_RETAINED_NONZERO_PREFIX_AND_NEW_OUTPUT_CUSTODY_V1'
PAID_OUTPUT=e.native_resources()['Hopper_storage_bytes']
TERMINAL_KEYS=retry.TERMINAL_KEYS
CONTROL=e._vector({})
need=e.need


def native_resources():return deepcopy(authority.NATIVE)


def fixed_request_fields():
    v=e.fixed_request_fields();sessions=deepcopy(r.sessions())
    for s in sessions:s['query_limit']=s['own_query_limit']
    v.update(schema=REQUEST_SCHEMA,platform_training_version=VERSION,profile=PROFILE,run_uuid=RUN,
        optimizer_update_limit=1026,query_limit=76,model_load_limit=7,
        prior_readonly_preparation_CPU_upper_s=0,additional_output_upper_bytes=0,
        paid_output_allowance_bytes=PAID_OUTPUT,session_plan=sessions,
        native_session_plan_version='AUDITED_PREFIX_FIFTEEN_E2_ZEROS_PLUS_THREE_OWN_SESSIONS_V1',
        completion_scope_grant_sha256=authority.GRANT_SHA,closed_predecessor_request_id=authority.REQUEST,
        closed_predecessor_row_sha256=authority.ROW,closed_predecessor_marker_sha256=authority.MARKER,
        saved_prefix_result_sha256=r.RESULT,saved_prefix_closure_sha256=r.CLOSED,
        saved_prefix_audit_lease_sha256=r.AUDIT_LEASE,retained_old_native=deepcopy(r.OLD_PAID),
        physical_guard_policy=PHYSICAL_SCHEMA,old_sessions_never_reexecuted=True)
    return v


def _original_identity_view(value):
    # Validation view only: never persisted or used for resource proposals.
    v=deepcopy(value);v.update(e.fixed_request_fields());return v


def request(value,*,central=False):
    fixed=fixed_request_fields()
    need(all(value.get(k)==v and type(value.get(k))is type(v)for k,v in fixed.items()),
        'REMAINING_EXACT_SAME_SCIENCE_AND_OWN1026_76_7')
    for k in('physical_custody_sha256','effective_queue_proof_sha256'):
        need(type(value.get(k))is str and re.fullmatch('[0-9a-f]{64}',value[k]),'REMAINING_SEALED_'+k)
    e.request(_original_identity_view(value),central=central)
    need(value['request_id']!=authority.REQUEST and value['source_sha']!=r.OLD_SOURCE,
        'REMAINING_NEW_REQUEST_AND_COLD_SOURCE_NOT_OLD_PROCESS_RESTART')
    return value


def select_profile(limits,value):request(value);return e.select_profile(limits,_original_identity_view(value))


def project_envelope(value):
    request(value)
    return dict(GPU_hours=value['reserved_H200_s']/3600,
        CPU_hours=max(c for c,_ in e.CAPACITY_PROFILES)*value['reserved_H200_s']/3600,
        reserved_CPU_including_prior_preparation_s=value['CPU_upper_s'],actual_process_CPU_upper_s=value['CPU_upper_s'],
        active_CPU_limit=8,project_hold_is_not_experiment_CPU_spend=True)


def process_CPU_limits(value):
    request(value)
    return dict(parent_RLIMIT_CPU_s=value['CPU_upper_s']-600-240,isolated_child_CPU_upper_s=600,
        terminal_observer_CPU_upper_s=240,prior_readonly_preparation_CPU_upper_s=0,
        whole_reserved_CPU_upper_s=value['CPU_upper_s'],new_CPU_budget=0)


def basic(g):
    """Acyclic exact index identity, also used by the original cap reader."""
    rows=[v for v in g['PBS_submissions']if v.get('platform_training_version')==VERSION]
    m=g.get(FIELD)
    if m is None:
        need(not rows,'REMAINING_ROW_REQUIRES_EXPLICIT_ADOPTED_INDEX');return None
    need(len(rows)==1 and m.get('schema')==SCHEMA and m.get('version')==VERSION,
        'REMAINING_EXACT_ONE_NEW_ROW_NOT_UNBOUNDED_RETRIES')
    row=rows[0];request(row,central=True)
    rb=base64.b64decode(m['request_base64'],validate=True);saved=json.loads(rb);request(saved)
    need(e.sha(rb)==m['request_sha256']==row['request_sha256']
        and all(row.get(k)==v and type(row.get(k))is type(v)for k,v in saved.items())
        and set(row)-set(saved)<=TERMINAL_KEYS|{'request_sha256','status','job_id','submitted_epoch'},
        'REMAINING_ORIGINAL_EXACT_REQUEST_BYTES_NO_RESTAMP_OR_ALIAS')
    prefix=r.saved_prefix(base64.b64decode(m['prefix_result_base64'],validate=True),base64.b64decode(m['prefix_closure_base64'],validate=True))
    old=[v for v in g['PBS_submissions']if v.get('request_id')==authority.REQUEST]
    need(len(old)==1 and e.sha(e.raw(old[0]))==authority.ROW
        and e.sha(e.raw(g[retry.ENTRY_FIELD]))==authority.MARKER
        and m['prefix_binding']==prefix and m['completion_scope_grant_sha256']==authority.GRANT_SHA
        and m['completion_authority_sha256']==e.sha(e.raw(g[authority.FIELD]))
        and m['original_package_sha256']==e.sha(e.raw(g[e.package.FIELD]))
        and m['native_hold']==native_resources()and m['control_paid_here']==e._vector({})
        and m['old_paid_gross_and_history_unchanged']is True,
        'REMAINING_REAL_NONZERO_CLOSED_PREFIX_AND_ALREADY_PAID_CONTROL')
    cap=e.authority.validate(g);before=m['before']
    need(cap is not None and all(row[k]==cap[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and type(m['registered_epoch'])in(int,float)and math.isfinite(m['registered_epoch'])
        and cap['T_CD']<=m['registered_epoch']and m['registered_epoch']+row['reserved_H200_s']<cap['stop_new_training_epoch']
        and row['submitted_epoch']==m['registered_epoch']and before['PBS_submissions']==38
        and type(row['submit_count'])is int and row['submit_count']==39
        and type(row['created_epoch'])in(int,float)and math.isfinite(row['created_epoch'])
        and cap['T_CD']<=row['created_epoch']<=m['registered_epoch']
        and g['used']['PBS_submissions']==len(g['PBS_submissions'])
        and 39<=len(g['PBS_submissions'])<=authority.AFTER
        and(row['status']==e.STATES[2]or len(g['PBS_submissions'])==39)
        and g['PBS_submissions'][38]==row,
        'REMAINING_ONE_PBS_WITH_ORIGINAL_WINDOW_MATRIX_AND_FINAL_PROTECTION')
    need(m['request_id']==row['request_id']and m['source_sha']==row['source_sha']
        and m['run_manifest_sha256']==row['run_manifest_sha256']
        and m['physical_custody_sha256']==row['physical_custody_sha256']==e.sha(e.raw(m['physical_custody']))
        and m['physical_custody']['schema']==PHYSICAL_SCHEMA,
        'REMAINING_EXACT_COLD_REQUEST_MANIFEST_AND_RETAINED_NONZERO_PHYSICAL_CUSTODY')
    need(m['physical_custody']['prior_physical_custody_sha256']==old[0]['physical_custody_sha256']
        and m['physical_custody']['old_metadata_roots']==g[retry.ENTRY_FIELD]['physical_custody']['old_roots']
        and m['physical_custody']['old_metadata_portable_sha256']==g[retry.ENTRY_FIELD]['physical_custody']['portable_inventory']['portable_sha256']
        and m['physical_custody']['old_metadata_identity_sha256']==g[retry.ENTRY_FIELD]['terminal']['physical']['portable_identity_sha256']
        and row['Core_measurement_capture_sha256']==g[e.FIELD]['Core_measurement_proof']['capture_sha256'],
        'REMAINING_OLD_PHYSICAL_CUSTODY_AND_REAL_MEASUREMENT_GATE_RETAINED')
    return m,row


@exact_ledger_validation
def validate(g,*,generation=None):
    need(generation is None,'REMAINING_NOT_A_ZERO_EFFECT_RETRY_GENERATION')
    selected=basic(g)
    if selected is None:return None
    m,row=selected
    # Reject an incomplete new input before the expensive full old-history
    # join. Every valid path still performs that join below, before admission.
    from . import minimal_remaining_physical as physical
    physical.validate_custody(m['physical_custody'],row,prefix_binding=m['prefix_binding'])
    physical.check_proof(m['physical_preflight'],m['physical_custody'],row,
        now_epoch=m['registered_epoch'],observed_epoch=m['physical_observed_epoch'])
    authority.validate(g);authority.predecessor(g)
    before=m['before']
    need(all(g[k]>=before[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads'))
        and g['used']['H200_s']>=before['H200_s']and g['new_storage_peak_bytes']>=g['new_storage_current_bytes'],
        'REMAINING_NO_REFUND_OF_ANY_OLD_COST_CURRENT_OR_PEAK')
    if row['status']==e.STATES[2]:
        from . import minimal_remaining_terminal as terminal
        terminal.validate_terminal(g,row,m)
    else:
        need(m['status']=='REGISTERED'and m['completed_sessions']==[]and 'terminal'not in m,
            'REMAINING_PREPARED_OR_RUNNING_FLAGS_ARE_NOT_COMPLETE')
    remaining=forecast_without_validate(g,m,row);old=g[e.original.PACKAGE]['remaining_resources']
    for k in e.package.TRACKED:need(g['reserved'][k]>=old[k]+remaining[k],'REMAINING_BOTH_SCOPES_FULL_HOLD:'+k)
    need(g['CPU_reserved_s']>=old['Hopper_CPU_s']+remaining['Hopper_CPU_s']
        and g['storage_reserved_bytes']>=old['Hopper_storage_bytes']+remaining['Hopper_storage_bytes'],
        'REMAINING_OLD_EVAL_AND_NEW_NATIVE_FULL_HOLDS')
    return m


def forecast_without_validate(g,m,row):
    value=retry._base_remaining(g)
    if row['status']!=e.STATES[2]:
        for k,n in native_resources().items():
            if k!='PBS_submissions':value[k]+=n
    return value


def forecast_remaining(g):
    m=validate(g)
    if m is None:return retry.forecast_remaining(g)
    return forecast_without_validate(g,*basic(g))


def rows(g):
    if validate(g)is None:return []
    return [v for v in g['PBS_submissions']if v.get('platform_training_version')==VERSION]


def current_id(g):return basic(g)[1]['request_id']


def budget_LOCAL(g,local):
    # Original control and completion control were paid before this request.
    return authority.budget_LOCAL(g,retry.budget_LOCAL(g,local))


def register(global_bytes,local_bytes,*,request_bytes,prefix_result_bytes,prefix_closure_bytes,
             physical_custody,physical_proof,physical_observed_epoch,fresh_readback,now_epoch):
    from scripts.pi0294.pv1.neural_ledger import check
    g=json.loads(global_bytes);value=json.loads(request_bytes);request(value);check(g)
    need(FIELD not in g and authority.validate(g)is not None
        and g['pending_updates']==g['unknown_effects']==[]
        and all(x['status']in e.original.TERMINAL for x in g['PBS_submissions']),
        'REMAINING_ONCE_KNOWN_CLOSED_TRAINERS_NOT_RESTART_OR_UNKNOWN_CLEAR')
    prefix=r.saved_prefix(prefix_result_bytes,prefix_closure_bytes)
    from . import minimal_remaining_physical as physical,minimal_remaining_concurrency as concurrency
    concurrency.cold_Core_rows(g)
    from scripts.pi0294.dec1 import vanda_campaign as queue
    from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
    rows=queue.queue_rows(fresh_readback['complete_queue'],now_epoch=now_epoch)
    campaign,outside,unknown=queue.classify(g,rows)
    checked_outside(rows,outside,base64.b64decode(fresh_readback['independent_confirmation_base64'],validate=True))
    need(campaign==unknown==[],'REMAINING_REAL_UNFILTERED_NATIVE_QUEUE_NO_ACTIVE_OR_UNKNOWN_TRAINER')
    physical.validate_custody(physical_custody,value,prefix_binding=prefix)
    physical.check_proof(physical_proof,physical_custody,value,now_epoch=now_epoch,observed_epoch=physical_observed_epoch)
    need(0<=now_epoch-fresh_readback['observed_epoch']<=120
        and fresh_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh_readback['complete_queue_verified']is True and fresh_readback['unknown_campaign_jobs']==[]
        and fresh_readback['active_experiment_Vanda_jobs']==campaign and fresh_readback['old_workers_terminal']is True
        and fresh_readback['old_closed_science_lease_sha256']==authority.LEASES['science'],
        'REMAINING_FRESH_ACTUAL_CANONICAL_COMPLETE_QUEUE_AND_CLOSED_PREDECESSOR')
    for old in[*g['PBS_submissions'],*g.get(e.original.CORE_ROWS,{}).values()]:
        need(old.get('request_id')!=value['request_id']and old.get('submission_nonce')!=value['submission_nonce'],
            'REMAINING_DISTINCT_BOTH_SITES_REQUEST_AND_NONCE')
    v=deepcopy(g);v['PBS_submissions'].append(dict(value,request_sha256=e.sha(request_bytes),status=e.STATES[0],job_id=None,submitted_epoch=now_epoch))
    v['used']['PBS_submissions']+=1
    held=native_resources()
    for k in e.package.TRACKED:v['reserved'][k]+=held[k]
    v['CPU_reserved_s']+=held['Hopper_CPU_s'];v['storage_reserved_bytes']+=held['Hopper_storage_bytes']
    before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads')}
    before.update(H200_s=g['used']['H200_s'],PBS_submissions=g['used']['PBS_submissions'])
    v[FIELD]=dict(schema=SCHEMA,version=VERSION,request_id=value['request_id'],request_sha256=e.sha(request_bytes),
        request_base64=base64.b64encode(request_bytes).decode(),source_sha=value['source_sha'],run_manifest_sha256=value['run_manifest_sha256'],
        prefix_result_base64=base64.b64encode(prefix_result_bytes).decode(),prefix_closure_base64=base64.b64encode(prefix_closure_bytes).decode(),
        prefix_binding=prefix,completion_scope_grant_sha256=authority.GRANT_SHA,completion_authority_sha256=e.sha(e.raw(g[authority.FIELD])),
        original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),native_hold=held,control_paid_here=e._vector({}),
        physical_custody=deepcopy(physical_custody),physical_custody_sha256=value['physical_custody_sha256'],
        physical_preflight=deepcopy(physical_proof),physical_observed_epoch=physical_observed_epoch,
        before=before,registered_epoch=now_epoch,status='REGISTERED',completed_sessions=[],old_paid_gross_and_history_unchanged=True)
    v['revision']+=1;validate(v);check(v)
    future=e.scope.first512_forecast(e.raw(v),local_bytes)
    need(future['cap_changes_required']=={},'REMAINING_FULL14_FUTURE_AND_ORIGINAL_PROTECTED_RESOURCES')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,native_admission=False)


def bind(g,request_id,job_id,*,now_epoch):
    validate(g);v=deepcopy(g);_,row=basic(v)
    need(row['request_id']==request_id and row['status']==e.STATES[0]and row['job_id']is None,
        'REMAINING_BIND_EXACT_NEW_ONCE_ONLY_REQUEST')
    row.update(status=e.STATES[1],job_id=job_id,native_reply_epoch=now_epoch)
    from scripts.pi0294.pv1.neural_ledger import check
    v['revision']+=1;validate(v);check(v);return v


def terminal_CPU_prior(g,row):
    need(basic(g)[1]==row and row['prior_readonly_preparation_CPU_upper_s']==0,
        'REMAINING_CPU_OLD900_AND_ALL_OLD_COSTS_REMAIN_IN_PREDECESSOR')
    authority.predecessor(g);return 0


def settle(g,request_id,**kwargs):
    from .minimal_remaining_terminal import settle as once_settle
    return once_settle(g,request_id,**kwargs)
