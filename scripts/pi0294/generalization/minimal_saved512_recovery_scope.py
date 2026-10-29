"""Bounded control for saved E1 own512 and untouched E2 fresh raw512.

Pure original GLOBAL proposals only. This module grants no request, loader,
optimizer, scheduler, checkpoint admission or native GPU permission.
"""
import base64,json,math
from copy import deepcopy
from . import minimal_execution as e,minimal_remaining_execution as previous
from .minimal_validation_cache import exact_ledger_validation

FIELD='ARX5_saved_E1_512_E2_raw512_recovery_control'
LOCAL_FIELD=FIELD+'_historical_LOCAL_projection'
SCHEMA='ARX5_SAVED_E1_512_E2_RAW512_RECOVERY_CONTROL_ADOPTION_V1'
GRANT_SHA='eaeadf6d27d130587ced2c8714ab1d4c798e1fecb6be889604c8f81553fdfaf9'
ROW='7258ce8fcb73ab8458e4f4c54312f0516304ce706d3c2a2a2ec11affec0f687d'
MARKER='d696911fa65cf6f4d2204b3e387edc469111c2aceaf89402bf8e7053bcd53aa3'
LEASE='1ddde92683d4b8e9dcb68e2f6ba913de5f756da1bb8d081533def129e501fa40'
REQUEST='80aeab00-91c9-455d-96bc-587f716f227f'
JOB='1505362.stdct-mgmt-02'
BEFORE=42;AFTER=43
CONTROL=e._vector(dict(Hopper_CPU_s=3600,local_CPU_s=600,Hopper_storage_bytes=8*1024**2))
NATIVE=e._vector(dict(H200_s=28920,Hopper_CPU_s=78000,optimizer_updates=512,neural_queries=62,PBS_submissions=1))
need=e.need


def grant(body):
    need(type(body)is bytes and e.sha(body)==GRANT_SHA,'SAVED512_RECOVERY_EXACT_SCOPE_BYTES')
    h=json.loads(body)
    need(h['closed_job_id']==JOB and h['closed_request_id']==REQUEST
        and h['closed_row_sha256']==ROW and h['closed_marker_sha256']==MARKER
        and h['closed_lease_sha256']==LEASE
        and h['caps_delta']=={'PBS_submissions':{'before':BEFORE,'after':AFTER}}
        and h['control']==dict(site_CPU_s=3600,local_CPU_s=600,site_metadata_output_bytes=8*1024**2,transfer_bytes=0)
        and h['native_effect_upper']==dict(optimizer_updates=512,neural_queries=62,model_loads=5)
        and h['planned_native']==dict(GPU_s=28920,site_CPU_s=78000,PBS_submissions=1,additional_site_output_charge_bytes=0)
        and h['other_caps_changed']is h['clocks_reset']is h['native_admission']is False
        and all(z is True for z in h['preserved'].values()),
        'SAVED512_RECOVERY_ONLY_UNFINISHED_SAME_SCIENCE_NOT_EXTRA_E1_STEPS')
    return h


def count_upper(g,old):
    """Exact later authority only; frozen earlier grant bytes stay unchanged."""
    m=g.get(FIELD)
    if m is None:return old
    grant(base64.b64decode(m['grant_base64'],validate=True))
    need(m['grant_sha256']==GRANT_SHA and m['closed_row_sha256']==ROW
        and m['closed_marker_sha256']==MARKER and m['after_PBS_cap']==AFTER,
        'SAVED512_RECOVERY_EXACT_COUNT_OVERLAY')
    return max(old,AFTER)


def predecessor(g):
    previous.validate(g);m,row=previous.basic(g)
    need(e.sha(e.raw(row))==ROW and e.sha(e.raw(m))==MARKER
        and row['request_id']==REQUEST and row['job_id']==JOB and row['status']==e.STATES[2]
        and row['native_exit_status']==1 and row['scientific_endpoint_complete']is False
        and (row['committed_optimizer_updates'],row['query_attempts'],row['model_load_attempts'])==(514,39,5)
        and m['status']=='INCOMPLETE'and m['completed_sessions']==['E2_SHARED_ACTION_HEAD_QUALIFY2'],
        'SAVED512_RECOVERY_REAL_NONZERO_CLOSED_SAME_PREDECESSOR')
    t=m['terminal'];s=t['effects']['sessions']
    need(s['E1_ADAPTER_ONLY_FORMAL512']['optimizer_updates']==512
        and s['E1_ADAPTER_ONLY_FORMAL512']['status']=='ACTIVE'
        and s['E2_SHARED_ACTION_HEAD_FORMAL512']['status']=='NOT_STARTED'
        and t['effects']['pending_updates']==t['effects']['unknown_query_intents']==[]
        and t['native']['conservative_GPU_s']==4162 and t['native']['conservative_CPU_s']==3585,
        'SAVED512_RECOVERY_NO_DISCARD_OF_PAID_UPDATES_QUERIES_OR_FAILURES')
    physical=t['physical']
    need(physical['paid_output_ceiling']==144294967296
        and physical['prefix_allocated_bytes']+physical['old_metadata_ceiling']
            +physical['new_allocated_bytes']+70000000000<=physical['paid_output_ceiling'],
        'SAVED512_RECOVERY_OLD_AND_NEW70GB_MUST_FIT_THE_SAME_ALREADY_PAID_POOL')
    return m,row


@exact_ledger_validation
def validate(g,*,generation=None):
    need(generation is None,'SAVED512_RECOVERY_NO_OTHER_GENERATION')
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True));predecessor(g);clock=e.authority.validate(g)
    need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['closed_row_sha256']==ROW and m['closed_marker_sha256']==MARKER
        and m['closed_lease_sha256']==LEASE and m['control_paid_once']==CONTROL
        and m['planned_remaining_native']==NATIVE and m['model_load_upper']==5
        and m['after_PBS_cap']==AFTER and m['phase']=='CONTROL_AND_INDEPENDENT_SAVED_STATE_AUDIT_ONLY'
        and m['saved_state_audit_complete_at_adoption']is m['native_admission']is False
        and all(m[k]==h[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=m['adopted_epoch']<clock['stop_new_training_epoch']
        and m['adopted_epoch']+NATIVE['H200_s']<clock['stop_new_training_epoch'],
        'SAVED512_RECOVERY_PAID_CONTROL_IS_NOT_NATIVE_REGISTER_OR_SUCCESS')
    b=m['before']
    need(g['CPU_s']>=b['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=b['new_storage_current_bytes']+CONTROL['Hopper_storage_bytes']
        and g['new_storage_peak_bytes']>=max(b['new_storage_peak_bytes'],g['new_storage_current_bytes'])
        and b['PBS_submissions']==39 and 39<=g['used']['PBS_submissions']<=AFTER,
        'SAVED512_RECOVERY_NO_REFUND_OF_OLD_CURRENT_PEAK_OR_COST')
    return m


def budget_LOCAL(g,local):
    if validate(g)is None:return local
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False)
    if LOCAL_FIELD in local:
        need(local[LOCAL_FIELD]==proof,'SAVED512_RECOVERY_ONCE_EXPLICIT_NONCANONICAL_LOCAL_PROJECTION');return local
    v=deepcopy(local);v['CPU_s']['local_PC']+=CONTROL['local_CPU_s'];v[LOCAL_FIELD]=proof
    return v


def planned_unregistered(g):
    return deepcopy(NATIVE)if validate(g)is not None else e._vector({})


def propose(global_bytes,historical_LOCAL_bytes,scope_bytes,*,closed_readback,now_epoch):
    h=grant(scope_bytes);g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g);predecessor(g)
    need(FIELD not in g and g['unknown_effects']==g['pending_updates']==[]
        and 0<=now_epoch-closed_readback['observed_epoch']<=120
        and closed_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and closed_readback['lease_sha256']==LEASE
        and closed_readback['all_complete_canonical_bytes_match']is True
        and closed_readback['finite_terminal_OS_exit']['returncode']==0,
        'SAVED512_RECOVERY_FRESH_EXACT_NATIVE_CLOSED_CONTEXT_NOT_CAPTURED_GO')
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')};before['PBS_submissions']=g['used']['PBS_submissions']
    v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes'];v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes']);v['revision']+=1
    v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),closed_row_sha256=ROW,closed_marker_sha256=MARKER,closed_lease_sha256=LEASE,before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(historical_LOCAL_bytes),fresh_closed_readback_sha256=e.sha(e.raw(closed_readback)),control_paid_once=deepcopy(CONTROL),planned_remaining_native=deepcopy(NATIVE),model_load_upper=5,after_PBS_cap=AFTER,phase='CONTROL_AND_INDEPENDENT_SAVED_STATE_AUDIT_ONLY',saved_state_audit_complete_at_adoption=False,native_admission=False,before=before,adopted_epoch=now_epoch,**{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    validate(v);check(v);future=e.scope.first512_forecast(e.raw(v),historical_LOCAL_bytes)
    need(future['cap_changes_required']=={},'SAVED512_RECOVERY_COMPLETE_ALL14_MATRIX_FINAL_AND_OLD_EVALS')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,new_requests=0,new_PBS=0,new_model_queries_updates=0,native_admission=False)
