"""Finite same-remainder cold repair, preserving the original campaign caps.

The earlier diagnostic liability is included once. This is not another GPU
request, new science arm, effect credit, or permission to replay an UNKNOWN.
"""
import base64,json,math
from copy import deepcopy
from . import minimal_execution as e,minimal_remaining_install_control as original

FIELD='ARX5_same_remainder_finite_cold_recovery'
SCHEMA='ARX5_SAME_REMAINDER_FINITE_COLD_RECOVERY_ADOPTION_V1'
GRANT_SHA='983f894b771c14f73dd89cae2e899d681232977965808c7503e49c4743c066e9'
CONTROL=e._vector(dict(Hopper_CPU_s=180,local_CPU_s=120,transfer_bytes=49152,
                       Hopper_storage_bytes=16*1024**2))
LOCAL_FIELD=FIELD+'_historical_LOCAL_projection'


def grant(body):
    e.need(type(body)is bytes and e.sha(body)==GRANT_SHA,'COLD_RECOVERY_EXACT_ENGINEERING_SCOPE')
    h=json.loads(body)
    e.need(h['predecessor_control_grant_sha256']==original.GRANT_SHA
        and h['control']==dict(site_CPU_s=180,local_CPU_s=120,transfer_bytes=49152,retained_metadata_bytes=16*1024**2)
        and sum(h['wire_plan'].values())==CONTROL['transfer_bytes']
        and h['campaign_caps_changed']=={}and all(n==0 for n in h['limits'].values())
        and all(v is True for v in h['preserved'].values()),'COLD_RECOVERY_NO_CAP_SCIENCE_OR_RESTART_CREDIT')
    return h


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    old=original.validate(g);e.need(old is not None,'COLD_RECOVERY_REQUIRES_REAL_OLD_CONTROL')
    h=grant(base64.b64decode(m['grant_base64'],validate=True));b=m['before']
    e.need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['old_control_marker_sha256']==e.sha(e.raw(old))
        and m['old_package_sha256']==e.sha(e.raw(g[e.package.FIELD]))
        and m['control_paid_once']==CONTROL and m['diagnostic_liability_included_once']==h['diagnostic_liability_included_once']
        and m['new_GPU_or_scientific_effect_permission']is False
        and all(m[k]==h[k]==old[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and type(m['adopted_epoch'])in(int,float)and math.isfinite(m['adopted_epoch'])
        and m['T_CD']<=m['adopted_epoch']<m['stop_new_training_epoch']
        and g['CPU_s']>=b['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=b['new_storage_current_bytes']+CONTROL['Hopper_storage_bytes']
        and g['new_storage_peak_bytes']>=max(b['new_storage_peak_bytes'],g['new_storage_current_bytes'])
        and b['PBS_submissions']==38 and g['used']['PBS_submissions']>=38,
        'COLD_RECOVERY_ORIGINAL_COSTS_CURRENT_PEAK_CLOCK_AND_ONCE_DIAGNOSTIC_CLASSIFICATION')
    return m


def budget_LOCAL(g,l):
    if validate(g)is None:return l
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False)
    if LOCAL_FIELD in l:
        e.need(l[LOCAL_FIELD]==proof,'COLD_RECOVERY_ONCE_HISTORICAL_PROJECTION');v=l
    else:
        v=deepcopy(l);v['CPU_s']['local_PC']+=CONTROL['local_CPU_s']
        v['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes'];v[LOCAL_FIELD]=proof
    from . import minimal_registered_unsubmitted_resume as resume
    return resume.budget_LOCAL(g,v)


def propose(global_bytes,historical_LOCAL_bytes,scope_bytes,*,now_epoch):
    h=grant(scope_bytes);g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    from . import minimal_remaining_execution as remaining,minimal_remaining_concurrency as concurrency
    check(g);old=original.validate(g)
    e.need(old is not None and FIELD not in g and remaining.FIELD not in g
        and g['used']['PBS_submissions']==38 and g['pending_updates']==g['unknown_effects']==[]
        and all(r['status']==e.STATES[2]for r in g['PBS_submissions']),
        'COLD_RECOVERY_ONCE_NO_REGISTERED_REMAINDER_OR_LIVE_TRAINER')
    concurrency.cold_Core_rows(g)
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')}
    before['PBS_submissions']=g['used']['PBS_submissions']
    v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes']);v['revision']+=1
    v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),
        old_control_marker_sha256=e.sha(e.raw(old)),old_package_sha256=e.sha(e.raw(g[e.package.FIELD])),
        before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(historical_LOCAL_bytes),before=before,
        control_paid_once=deepcopy(CONTROL),diagnostic_liability_included_once=h['diagnostic_liability_included_once'],
        adopted_epoch=now_epoch,new_GPU_or_scientific_effect_permission=False,
        **{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    validate(v);check(v);future=e.scope.first512_forecast(e.raw(v),historical_LOCAL_bytes)
    e.need(future['cap_changes_required']=={},'COLD_RECOVERY_ALL14_AND_MATRIX_FINAL_PROTECTED')
    return dict(proposed_global=v,full_future_proof=future,native_admission=False,original_onceIO_required=True)
