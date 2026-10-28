"""One finite ordinary-engineering supplement inside unchanged campaign caps.

Old completion control, native effects, prices and reservations remain intact.
Pure proposals are not adoption or permission. The original onceIO owner must
compare current canonical bytes and independently reconcile any unknown reply.
"""
import base64,json,math
from copy import deepcopy
from . import minimal_execution as e,minimal_partial_completion_scope as original

FIELD='ARX5_remaining_same_scope_finite_install_control'
SCHEMA='ARX5_REMAINING_SAME_SCOPE_FINITE_INSTALL_CONTROL_ADOPTION_V1'
GRANT_SHA='d5b8e6f6e40128e5c5a8ba25eb5a20e0bec04f8312bf9b6398e122715591cdea'
CONTROL=e._vector(dict(Hopper_CPU_s=240,local_CPU_s=90,transfer_bytes=65536))
LOCAL_FIELD=FIELD+'_historical_LOCAL_projection'


def grant(body):
    e.need(type(body)is bytes and e.sha(body)==GRANT_SHA,'INSTALL_CONTROL_EXACT_BOUNDED_ENGINEERING_BYTES')
    v=json.loads(body)
    e.need(v['old_completion_scope_sha256']==original.GRANT_SHA
        and v['control']==dict(site_CPU_s=240,local_CPU_s=90,transfer_bytes=65536,additional_output_bytes=0)
        and sum(v['wire_plan'].values())==CONTROL['transfer_bytes']
        and v['campaign_caps_changed']=={} and all(n==0 for n in v['limits'].values())
        and all(x is True for x in v['preserved'].values()),'INSTALL_CONTROL_NO_NEW_SCIENCE_CAP_OR_OLD_COST_CREDIT')
    return v


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True));old=g.get(original.FIELD,{})
    e.need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['control_paid_once']==CONTROL and old.get('grant_sha256')==original.GRANT_SHA
        and m['old_completion_marker_sha256']==e.sha(e.raw(old))
        and m['original_package_sha256']==e.sha(e.raw(g[e.package.FIELD]))
        and m['old_predecessor_row_sha256']==original.ROW
        and m['actual_prefix_result_sha256']==h['actual_prefix_result_sha256']
        and m['installed_source_seal_sha256']==h['already_installed_source_seal_sha256']
        and m['new_GPU_or_native_effect_permission']is False,
        'INSTALL_CONTROL_ONE_SAME_REMAINDER_NO_OLD_MARKER_OR_PACKAGE_REWRITE')
    b=m['before']
    e.need(all(type(m[k])is type(h[k])and m[k]==h[k]==old[k] for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and type(m['adopted_epoch'])in(int,float)and math.isfinite(m['adopted_epoch'])
        and m['T_CD']<=m['adopted_epoch']<m['stop_new_training_epoch']
        and b['PBS_submissions']==38 and g['used']['PBS_submissions']>=38
        and g['CPU_s']>=b['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=b['new_storage_current_bytes']
        and g['new_storage_peak_bytes']>=max(b['new_storage_peak_bytes'],g['new_storage_current_bytes']),
        'INSTALL_CONTROL_OLD_CLOCK_COST_AND_STORAGE_PEAK_RETAINED')
    return m


def budget_LOCAL(g,l):
    if validate(g)is None:return l
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False)
    if LOCAL_FIELD in l:
        e.need(l[LOCAL_FIELD]==proof,'INSTALL_CONTROL_ONCE_HISTORICAL_PROJECTION');return l
    v=deepcopy(l);v['CPU_s']['local_PC']+=CONTROL['local_CPU_s']
    v['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes'];v[LOCAL_FIELD]=proof
    return v


def propose(global_bytes,historical_LOCAL_bytes,scope_bytes,*,now_epoch):
    h=grant(scope_bytes);g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g);original.validate(g);original.predecessor(g)
    from . import minimal_remaining_execution as remaining,minimal_remaining_concurrency as concurrency
    e.need(FIELD not in g and remaining.FIELD not in g
        and g['used']['PBS_submissions']==38 and g['pending_updates']==g['unknown_effects']==[]
        and all(r['status']==e.STATES[2]for r in g['PBS_submissions']),
        'INSTALL_CONTROL_ONCE_BEFORE_ONE_REMAINING_REQUEST_AND_NO_LIVE_TRAINER')
    concurrency.cold_Core_rows(g)
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')}
    before['PBS_submissions']=g['used']['PBS_submissions'];v['CPU_s']+=CONTROL['Hopper_CPU_s']
    v[FIELD]=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),
        control_paid_once=deepcopy(CONTROL),old_completion_marker_sha256=e.sha(e.raw(g[original.FIELD])),
        original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),old_predecessor_row_sha256=original.ROW,
        actual_prefix_result_sha256=h['actual_prefix_result_sha256'],installed_source_seal_sha256=h['already_installed_source_seal_sha256'],
        before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(historical_LOCAL_bytes),
        adopted_epoch=now_epoch,before=before,new_GPU_or_native_effect_permission=False,
        **{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    v['revision']+=1;validate(v);check(v)
    future=e.scope.first512_forecast(e.raw(v),historical_LOCAL_bytes)
    e.need(future['cap_changes_required']=={},'INSTALL_CONTROL_COMPLETE_ALL14_FUTURE_AND_PROTECTED_MATRIX_FINAL')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,native_admission=False)
