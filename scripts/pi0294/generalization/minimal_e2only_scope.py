"""Exact human PBS44 continuation: keep E1 and the failed E2 costs intact.

Pure proposals in the original GLOBAL only. Native source/identity, actual
queue/quota, immutable inputs and the existing detached owner remain required.
"""
import base64,json
from copy import deepcopy
from . import minimal_execution as e,minimal_saved512_execution as previous
from .minimal_validation_cache import exact_ledger_validation

FIELD='ARX5_E2_only_raw512_PBS44_authority'
LOCAL_FIELD=FIELD+'_historical_LOCAL_projection'
SCHEMA='ARX5_E2_ONLY_RAW512_PBS44_CONTROL_ADOPTION_V1'
GRANT_SHA='1cc87ade52dc5e266376c56bfd3fb57c5eaa63648789c963b596601107b931cc'
REQUEST='586b6601-daf5-4823-950b-605896e611c0'
JOB='1506193.stdct-mgmt-02'
ROW='bca2e93f38bdcca3ec4d10dc79abe418d8a5fbcc20f9b627b06bb40fcd172f28'
MARKER='31d445f0f334be1603c3f9ebbfb834381e7fb6cb81b77e4c903370009fafd3a4'
LEASE='d422ff437ad7409b0042c7761816963bc5bf5eae8d0c9d6c188d90293d44fabb'
BEFORE=43;AFTER=44
CONTROL=e._vector(dict(Hopper_CPU_s=3600,local_CPU_s=600))
NATIVE=e._vector(dict(H200_s=22560,Hopper_CPU_s=54000,optimizer_updates=512,neural_queries=37,PBS_submissions=1))
PAID_OUTPUT=144294967296
need=e.need


def grant(body):
    need(type(body)is bytes and e.sha(body)==GRANT_SHA,'E2ONLY_EXACT_DIRECT_PBS44_GRANT')
    h=json.loads(body)
    need(h['schema']=='ARX5_E2_ONLY_RAW512_PBS44_DIRECT_AUTHORITY_V1'
        and h['closed_job_id']==JOB and h['closed_request_id']==REQUEST
        and h['closed_row_sha256']==ROW and h['closed_marker_sha256']==MARKER and h['closed_lease_sha256']==LEASE
        and h['caps_delta']=={'PBS_submissions':{'before':BEFORE,'after':AFTER}}
        and h['native_effect_upper']==dict(GPU_s=22560,site_CPU_s=54000,optimizer_updates=512,queries=37,
            model_loads=2,PBS_submissions=1,additional_output_charge_bytes=0)
        and h['control']==dict(site_CPU_s=3600,PC_CPU_s=600,site_metadata_bytes=8388608,new_financial_storage_charge_bytes=0)
        and h['E1_new_training_or_inference']==0 and h['E2_initialization']=='ORIGINAL_RAW_PARENT_FRESH_OPTIMIZER'
        and h['E2_target_updates']==512 and h['E2_scheduler_horizon']==1024
        and h['other_caps_changed']is h['native_adoption']is False and all(h['preserved'].values()),
        'E2ONLY_NO_EXTRA_TRAIN_OR_REFUND_OR_OTHER_CAP_CHANGE')
    return h


def predecessor(g):
    previous.validate(g);m,row=previous.basic(g)
    need(e.sha(e.raw(row))==ROW and e.sha(e.raw(m))==MARKER
        and row['request_id']==REQUEST and row['job_id']==JOB and row['status']==e.STATES[2]
        and row['native_exit_status']==1 and row['scientific_endpoint_complete']is False
        and (row['committed_optimizer_updates'],row['query_attempts'],row['model_load_attempts'])==(122,25,4)
        and m['status']=='INCOMPLETE'and m['completed_sessions']==['E1_ADAPTER_ONLY_FORMAL512'],
        'E2ONLY_REAL_CLOSED_PARTIAL122_NOT_NEW_GO_OR_SAVED_CHECKPOINT')
    t=m['terminal'];s=t['effects']['sessions']
    need(s['E1_ADAPTER_ONLY_FORMAL512']['status']=='COMPLETE'
        and s['E1_ADAPTER_ONLY_FORMAL512']['optimizer_updates']==0
        and s['E2_SHARED_ACTION_HEAD_FORMAL512']['status']=='ACTIVE'
        and s['E2_SHARED_ACTION_HEAD_FORMAL512']['optimizer_updates']==122
        and t['native']['conservative_GPU_s']==2526 and t['native']['conservative_CPU_s']==1825
        and t['effects']['pending_updates']==t['effects']['unknown_query_intents']==[],
        'E2ONLY_PRESERVE_E1_QUALIFICATION_AND_ALL_FAILED_NATIVE_EFFECTS')
    return m,row


@exact_ledger_validation
def validate(g,*,generation=None):
    need(generation is None,'E2ONLY_NO_OTHER_GENERATION')
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True));predecessor(g);clock=e.authority.validate(g)
    need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['closed_row_sha256']==ROW and m['closed_marker_sha256']==MARKER and m['closed_lease_sha256']==LEASE
        and m['control_paid_once']==CONTROL and m['planned_remaining_native']==NATIVE
        and m['model_load_upper']==2 and m['after_PBS_cap']==AFTER
        and m['site_metadata_in_existing70GB_root']==8388608 and m['new_storage_charge_bytes']==0
        and m['native_admission']is False and m['actual_LOCAL_written']is False
        and all(m[k]==h[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=m['adopted_epoch']<m['adopted_epoch']+NATIVE['H200_s']<clock['stop_new_training_epoch'],
        'E2ONLY_EXACT_COUNT_OVERLAY_AND_ALREADY_PAID_OUTPUT_POOL')
    b=m['before']
    need(b['PBS_submissions']==40 and 40<=g['used']['PBS_submissions']<=AFTER
        and g['CPU_s']>=b['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=b['new_storage_current_bytes']
        and g['new_storage_peak_bytes']>=b['new_storage_peak_bytes'],
        'E2ONLY_OLD_PAID_HISTORY_AND_PEAK_NEVER_REFUNDED')
    return m


def budget_LOCAL(g,local):
    if validate(g)is None:return local
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False)
    if LOCAL_FIELD in local:
        need(local[LOCAL_FIELD]==proof,'E2ONLY_ONCE_HISTORICAL_LOCAL_PROJECTION');return local
    v=deepcopy(local);v['CPU_s']['local_PC']+=CONTROL['local_CPU_s'];v[LOCAL_FIELD]=proof
    return v


def planned_unregistered(g):
    from . import minimal_e2only_execution as execution
    return deepcopy(NATIVE)if validate(g)is not None and execution.basic(g)is None else e._vector({})


def propose(global_bytes,historical_LOCAL_bytes,scope_bytes,*,closed_readback,now_epoch):
    h=grant(scope_bytes);g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g);predecessor(g)
    need(FIELD not in g and g['unknown_effects']==g['pending_updates']==[]
        and all(r['status']==e.STATES[2]for r in g['PBS_submissions'])
        and 0<=now_epoch-closed_readback['observed_epoch']<=120
        and closed_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and closed_readback['lease_sha256']==LEASE and closed_readback['all_complete_canonical_bytes_match']is True
        and closed_readback['finite_terminal_OS_exit']['returncode']==0,
        'E2ONLY_FRESH_CANONICAL_TERMINAL_AND_OWN_CLOSED_LEASE_NO_REPLAY')
    v=deepcopy(g);v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['revision']+=1
    before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')}
    before['PBS_submissions']=g['used']['PBS_submissions']
    v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),
        closed_row_sha256=ROW,closed_marker_sha256=MARKER,closed_lease_sha256=LEASE,before_GLOBAL_sha256=e.sha(global_bytes),
        historical_LOCAL_sha256=e.sha(historical_LOCAL_bytes),fresh_closed_readback_sha256=e.sha(e.raw(closed_readback)),
        control_paid_once=deepcopy(CONTROL),planned_remaining_native=deepcopy(NATIVE),model_load_upper=2,after_PBS_cap=AFTER,
        site_metadata_in_existing70GB_root=8388608,new_storage_charge_bytes=0,native_admission=False,actual_LOCAL_written=False,
        before=before,adopted_epoch=now_epoch,**{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    validate(v);check(v);future=e.scope.first512_forecast(e.raw(v),historical_LOCAL_bytes)
    need(future['cap_changes_required']=={},'E2ONLY_ALL14_OLD_EVAL_MATRIX3_FINAL_PROTECTED')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,
        new_requests=0,new_PBS=0,new_model_queries_updates=0,native_admission=False)
