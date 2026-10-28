"""One bounded completion of the original partly executed first512 package.

This is control/cold-audit accounting, not another zero-effect replacement.
The actual two qualification calls and all71 queries remain paid history.
No request, model, optimizer, scheduler or native permission is created here.
"""
import base64,json,math
from copy import deepcopy
from . import minimal_execution as e,minimal_retry_execution as retry
from .minimal_validation_cache import exact_ledger_validation

FIELD='ARX5_first512_saved_prefix_one_completion_authority'
SCHEMA='ARX5_FIRST512_SAVED_PREFIX_COMPLETION_CONTROL_ADOPTION_V1'
GRANT_SHA='c2733caa01de7a85419d9fd46f2e315f8bea89d892e43b6b1945f6af319bc20c'
REQUEST='a6b3936e-f273-4491-b43b-b8168a6d0f3b'
JOB='1504095.stdct-mgmt-02'
ROW='0d2db1622d2be252325d0ade59795ddffcb899868fce8a6b21d94c153b2ee5c9'
MARKER='723b0ff3bfa54be04fae78acd1c39b0f0efd283c28f0278734dfe8681319a477'
CAPTURE='8856f85d5d8c905905b7b135d1ede84e012b47ad8efa92fece16e4524de1a8ba'
LEASES=dict(science='1758991a278c5e131a4e9aaeceeac1bab518a9be9f5b11a9375065d196d40621',
            metadata='07af7cfbb0515b0aac3c0eade942ac51bc5ac1aa40511a62b738ff47012fbcfc')
BEFORE=41;AFTER=42
CONTROL=e._vector(dict(Hopper_CPU_s=3600,local_CPU_s=600,
    Hopper_storage_bytes=64*1024**2,transfer_bytes=256*1024))
WIRE=256*1024
NATIVE=e._vector(dict(e.native_resources(),optimizer_updates=1026,neural_queries=76,Hopper_storage_bytes=0))
MODEL_LOAD_LIMIT=7


def grant(body):
    e.need(type(body)is bytes and e.sha(body)==GRANT_SHA,'COMPLETION_EXACT_BOUNDED_SCOPE_BYTES')
    v=json.loads(body)
    e.need(v['closed_request_id']==REQUEST and v['closed_job_id']==JOB
        and v['closed_row_sha256']==ROW and v['closed_marker_sha256']==MARKER
        and v['independent_closed_capture_sha256']==CAPTURE
        and v['actual_CLOSED_science_lease_sha256']==LEASES['science']
        and v['actual_CLOSED_metadata_lease_sha256']==LEASES['metadata']
        and v['caps_delta']=={'PBS_submissions':{'before':BEFORE,'after':AFTER}}
        and v['maximum_additional_GPU_jobs']==v['maximum_experiment_Vanda_GPU_concurrency']==1
        and v['preserved_actual_effects']==dict(optimizer_updates=2,neural_queries=71,model_loads=5)
        and v['remaining_native_effect_upper']['optimizer_updates']==1026
        and v['remaining_native_effect_upper']['neural_queries']==76
        and v['remaining_native_effect_upper']['model_loads']==MODEL_LOAD_LIMIT
        and v['full_independent_saved_prefix_audit_required_before_completion_credit']is True
        and v['formal_training_starts_from_original_raw_parent_not_qualification_weights']is True
        and v['authority_basis']['direct_human_messages'][-1]=='继续推进，无需批准'
        and v['other_caps_changed']is v['clocks_reset']is v['native_admission']is False,
        'COMPLETION_ORIGINAL_SUFFIX_NOT_NEW_ARM_OR_ZERO_EFFECT_REPLAY')
    return v


def predecessor(g):
    marker=retry.validate(g,generation=3);old,row=retry._selected(g,generation=3)
    e.need(marker==old and e.sha(e.raw(row))==ROW and e.sha(e.raw(marker))==MARKER
        and row['request_id']==REQUEST and row['job_id']==JOB
        and row['status']==e.STATES[2] and row['native_exit_status']==1
        and row['scientific_endpoint_complete']is False
        and (row['committed_optimizer_updates'],row['query_attempts'],row['model_load_attempts'])==(2,71,5),
        'COMPLETION_EXACT_REAL_NONZERO_CLOSED_PREDECESSOR_NOT_OLD_ZERO_ROW')
    t=marker['terminal'];sessions=t['effects']['sessions'];ids=[s['id']for s in e.native_sessions()]
    e.need([sessions[k]['status']for k in ids]==['COMPLETE','COMPLETE','ACTIVE','NOT_STARTED','NOT_STARTED']
        and t['effects']['pending_updates']==t['effects']['unknown_query_intents']==[]
        and t['native']['conservative_GPU_s']==1917 and t['native']['conservative_CPU_s']==1222,
        'COMPLETION_PRESERVED_TWO_COMPLETED_PREFIXES_AND_KNOWN_UNOPTIMIZED_E2')
    return marker,row


@exact_ledger_validation
def validate(g,*,generation=None):
    e.need(generation is None,'COMPLETION_NO_OTHER_GENERATION_SELECTOR')
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True));predecessor(g)
    clock=e.authority.validate(g)
    e.need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['closed_row_sha256']==ROW and m['closed_marker_sha256']==MARKER
        and m['closed_capture_sha256']==CAPTURE and m['closed_lease_sha256']==LEASES
        and m['control_paid_once']==CONTROL and m['planned_remaining_native']==NATIVE
        and m['model_load_upper']==MODEL_LOAD_LIMIT and m['after_PBS_cap']==AFTER
        and m['phase']=='CONTROL_AND_SAVED_PREFIX_AUDIT_ONLY'
        and m['prefix_independent_audit_complete_at_adoption']is False
        and m['native_admission']is False and m['old_paid_gross_unchanged']is True
        and all(m[k]==h[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=m['adopted_epoch']<clock['stop_new_training_epoch']
        and m['adopted_epoch']+NATIVE['H200_s']<clock['stop_new_training_epoch'],
        'COMPLETION_EXACT_PAID_CONTROL_AND_UNRESET_WINDOW_NOT_NATIVE_GO')
    before=m['before']
    e.need(g['CPU_s']>=before['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=before['new_storage_current_bytes']+CONTROL['Hopper_storage_bytes']
        and g['new_storage_peak_bytes']>=max(before['new_storage_peak_bytes'],g['new_storage_current_bytes'])
        and g['used']['PBS_submissions']==before['PBS_submissions']==38,
        'COMPLETION_OLD_COSTS_CURRENT_PEAK_AND_NO_NEW_REQUEST_AT_CONTROL_STAGE')
    return m


def planned_unregistered(g):
    return deepcopy(NATIVE)if validate(g)is not None else e._vector({})


def budget_LOCAL(g,l):
    """Conservative paid control projection, never canonical Core adoption."""
    if validate(g)is None:return l
    key='ARX5_first512_saved_prefix_completion_control_projection'
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False)
    if key in l:
        e.need(l[key]==proof,'COMPLETION_ONCE_CONSERVATIVE_LOCAL_COST_PROJECTION');return l
    v=deepcopy(l);v['CPU_s']['local_PC']+=CONTROL['local_CPU_s']
    v['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes'];v[key]=proof
    return v


def propose(global_bytes,local_bytes,scope_bytes,*,closed_capture_bytes,fresh_readback,now_epoch):
    h=grant(scope_bytes);g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g);predecessor(g);closed=json.loads(closed_capture_bytes)
    e.need(FIELD not in g and g['unknown_effects']==g['pending_updates']==[]
        and all(r['status']in e.original.TERMINAL for r in g['PBS_submissions'])
        and all(r['status']=='NATIVE_TERMINAL'for r in g.get(e.original.CORE_ROWS,{}).values()),
        'COMPLETION_ONCE_ALL_ORIGINAL_NATIVE_ACTORS_TERMINAL_BOUNDARY')
    e.need(e.sha(closed_capture_bytes)==CAPTURE and closed['GLOBAL_sha256']==e.sha(global_bytes)
        and closed['old_UNKNOWN_permanent']is True and closed['all_own_processes_absent']is True
        and closed['independent_full_qualification_audit_complete']is False
        and all(closed['leases'][k]['sha256']==v and closed['leases'][k]['value']['status']=='CLOSED'
                for k,v in LEASES.items()),'COMPLETION_REAL_INDEPENDENT_NONZERO_TERMINAL_AND_BOTH_CLOSED_LEASES')
    e.need(0<=now_epoch-fresh_readback['observed_epoch']<=120
        and fresh_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh_readback['closed_lease_sha256']==LEASES
        and fresh_readback['all_original_actors_absent']is True
        and fresh_readback['original_SETTLE_UNKNOWN_preserved']is True
        and fresh_readback['new_mutations']==0,'COMPLETION_FRESH_CANONICAL_JOIN_NOT_AN_OLD_CAPTURE_PASS')
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')}
    before['PBS_submissions']=g['used']['PBS_submissions']
    v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),
        closed_row_sha256=ROW,closed_marker_sha256=MARKER,closed_capture_sha256=CAPTURE,
        closed_lease_sha256=deepcopy(LEASES),before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(local_bytes),
        fresh_readback_sha256=e.sha(e.raw(fresh_readback)),control_paid_once=deepcopy(CONTROL),
        planned_remaining_native=deepcopy(NATIVE),model_load_upper=MODEL_LOAD_LIMIT,after_PBS_cap=AFTER,
        phase='CONTROL_AND_SAVED_PREFIX_AUDIT_ONLY',prefix_independent_audit_complete_at_adoption=False,
        old_paid_gross_unchanged=True,native_admission=False,before=before,adopted_epoch=now_epoch,
        **{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    v['revision']+=1;validate(v);check(v)
    future=e.scope.first512_forecast(e.raw(v),local_bytes)
    e.need(future['cap_changes_required']=={},'COMPLETION_COMPLETE_ALL14_FUTURE_AND_MATRIX_FINAL_PROTECTION')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,
        new_requests=0,new_PBS=0,new_model_queries_updates=0,native_admission=False)
