"""One first512 escrow in the existing GLOBAL, not a second budget or owner.

These are pure transitions. Original onceIO and actual standard registration
remain mandatory. Core's node-local mirror is explicitly pending; no model,
PBS, physical lease, measurement qualification or fresh LOCAL is invented.
"""
from copy import deepcopy
import json
import math

from . import minimal_adoption as authority, minimal_scope as scope
from .minimal_dataset import PANEL_SHA, CURSOR_SHA
from .minimal_training import RAW_PARENT_FILES_ID
from .minimal_study import R0
from scripts.pi0294.pv1.future_budget import _vector
from scripts.pi0294.pv1 import mix_accounting as original

FIELD = 'ARX5_minimal_first512_execution_package'
SCHEMA = 'ARX5_MINIMAL_FIRST512_ORIGINAL_GLOBAL_PACKAGE_V1'
TRACKED = ('H200_s','optimizer_updates','neural_queries')


def sessions():
    """Separate local counters even when one original owner runs a single PBS."""
    rows = [dict(id='E0_RAW_DIAGNOSTIC',arm='E0_RAW',optimizer_limit=0,
                 query_limit=41,sample_visits=0,fresh_raw_parent=True,formal_candidate=False)]
    for arm in ('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
        rows.append(dict(id=arm+'_QUALIFY2',arm=arm,optimizer_limit=2,
            query_limit=17,sample_visits=64,fresh_raw_parent=True,formal_candidate=False))
    for arm in ('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
        rows.append(dict(id=arm+'_FORMAL512',arm=arm,optimizer_limit=512,
            query_limit=36,sample_visits=16384,fresh_raw_parent=True,formal_candidate=True))
    return rows


def resources():
    return _vector({k:sum(r[k] for r in scope.components().values()) for k in _vector({})})


def validate(g):
    from . import minimal_execution as execution, minimal_retry_execution as retry,minimal_remaining_execution as remaining_execution
    has_retry=(retry.FIELD in g or any(r.get('platform_training_version')==retry.VERSION
                                    for r in g['PBS_submissions']))
    if FIELD not in g:
        authority.need(not has_retry,'RETRY_REQUIRES_ORIGINAL_RESERVED_PACKAGE')
        return None
    m=authority.validate(g);p=g[FIELD]
    authority.need(m is not None and p['schema']==SCHEMA and p['scope']==scope.SCOPE
        and p['grant_sha256']==authority.GRANT_SHA
        and p['cap_authority_sha256']==authority.sha(authority.raw(m))
        and all(p[k]==m[k]for k in ('T_CD','deadline_epoch','stop_new_training_epoch'))
        and type(p['reserved_epoch'])in(int,float) and math.isfinite(p['reserved_epoch'])
        and m['T_CD']<=p['reserved_epoch']<m['stop_new_training_epoch'],
        'PACKAGE_EXACT_APPROVED_CAPS_AND_ORIGINAL_CLOCK')
    authority.need(p['original_resources']==resources()
        and p['session_plan']==sessions() and p['scheduler_horizon']==1024
        and p['raw_parent_files_identity']==RAW_PARENT_FILES_ID
        and p['statistics_sha256']==R0 and p['panel_sha256']==PANEL_SHA
        and p['full_cursor_sha256']==CURSOR_SHA
        and p['Core_LOCAL_sync']=='PENDING_CANONICAL_4080_NODE_LOCAL_NO_COPY'
        and p['actual_LOCAL_written']is False and p['native_admission']is False
        and p['Core_measurement_gate_for_formal_training']is True,
        'PACKAGE_FROZEN_SCIENCE_SESSIONS_AND_PENDING_CORE_MIRROR')
    remaining=_vector(p['remaining_resources'])
    authority.need(all(remaining[k]<=p['original_resources'][k]for k in remaining),
                   'PACKAGE_NONNEGATIVE_ORIGINAL_REMAINING_RESOURCES')
    if remaining_execution.FIELD in g:
        remaining_execution.validate(g)
        remaining=remaining_execution.forecast_remaining(g)
    elif has_retry:
        retry.validate(g)
        remaining=retry.forecast_remaining(g)
    elif execution.FIELD in g:
        execution.validate(g)
        remaining=execution.forecast_remaining(g)
    else:
        authority.need(p['completed_sessions']==[] and p['registered_requests']==[]
            and p['remaining_resources']==p['original_resources']
            and p['status']=='GLOBAL_RESERVED_CORE_LOCAL_PENDING_NO_DISPATCH',
            'PACKAGE_RESERVATION_ONLY_NOT_FAKE_SESSION_COMPLETION')
    old=g[original.PACKAGE]['remaining_resources']
    for key in TRACKED:
        authority.need(g['reserved'][key]>=remaining[key]+old[key],
                       'PACKAGE_BOTH_OLD_AND_NEW_NEURAL_HOLDS_RETAINED:'+key)
    authority.need(g['CPU_reserved_s']>=remaining['Hopper_CPU_s']+old['Hopper_CPU_s']
        and g['storage_reserved_bytes']>=remaining['Hopper_storage_bytes']+old['Hopper_storage_bytes'],
        'PACKAGE_BOTH_OLD_AND_NEW_SITE_CPU_AND_STORAGE_HOLDS_RETAINED')
    from scripts.pi0294.generalization.amd_training_accounting import effective_original_eval_PBS
    from scripts.pi0294.pv1.evaluation_work import MATRIX
    old_pbs=effective_original_eval_PBS(g,old['PBS_submissions'])
    effective=authority.effective_caps(g,m['caps'])
    authority.need(g['used']['PBS_submissions']+old_pbs+remaining['PBS_submissions']
        +MATRIX['PBS_submissions']<=effective['PBS_submissions'],
        'PACKAGE_TWO_PBS_AND_ORIGINAL_MATRIX_THREE_PROTECTED')
    return p


def reservation_credit(g, historical_LOCAL_bytes, baseline):
    """Exact projection delta; retain the old max(GLOBAL, historical LOCAL).

    A full new hold can partly overlap that old max. Crediting its nominal
    size would erase old Core escrow. Compute the *actual* floor delta and
    original-package credit delta instead, without writing a stripped ledger.
    """
    p=validate(g)
    if p is None:return _vector({})
    from scripts.pi0294.pv1.mix_vanda_first import budget_LOCAL
    from scripts.pi0294.pv1.evaluation_admission import resource_floor
    projected_LOCAL=budget_LOCAL(g,json.loads(historical_LOCAL_bytes))
    _,held=resource_floor(g,projected_LOCAL)
    from . import minimal_execution as execution, minimal_retry_execution as retry,minimal_remaining_execution as remaining_execution
    adapter=remaining_execution if remaining_execution.FIELD in g else retry if retry.FIELD in g else execution
    view=deepcopy(g);v=adapter.forecast_remaining(g)
    for key in TRACKED:view['reserved'][key]-=v[key]
    view['CPU_reserved_s']-=v['Hopper_CPU_s']
    view['storage_reserved_bytes']-=v['Hopper_storage_bytes']
    _,without=resource_floor(view,projected_LOCAL)
    credit=_vector({});old=g[original.PACKAGE]['remaining_resources']
    for key in (*TRACKED,'Hopper_CPU_s','Hopper_storage_bytes'):
        authority.need(baseline['dimensions'][key]['reserved']==held[key],
                       'EXACT_CURRENT_BUDGET_FLOOR_FOR_OWN_HOLD:'+key)
        delta=(held[key]-min(held[key],old[key]))-(without[key]-min(without[key],old[key]))
        authority.need(0<=delta<=v[key],'NO_NOMINAL_CREDIT_OVER_HISTORICAL_LOCAL_FLOOR:'+key)
        credit[key]=delta
    return credit


def reserve(global_bytes,historical_LOCAL_bytes,*,now_epoch):
    authority.need(type(global_bytes)is bytes and type(historical_LOCAL_bytes)is bytes,
                   'PACKAGE_COMPLETE_LEDGER_INPUT_BYTES')
    g=json.loads(global_bytes);m=authority.validate(g)
    authority.need(m is not None and FIELD not in g,'PACKAGE_ONCE_AFTER_REAL_CAP_ADOPTION')
    authority.need(authority.sha(historical_LOCAL_bytes)==m['historical_LOCAL_sha256']
        and type(now_epoch)in(int,float) and math.isfinite(now_epoch)
        and m['T_CD']<=now_epoch<m['stop_new_training_epoch']
        and now_epoch+59520<m['deadline_epoch'],
        'PACKAGE_ORIGINAL_WINDOW_AND_EXACT_HISTORICAL_LOCAL')
    original.validate_global(g)
    authority.need(g['pending_updates']==g['unknown_effects']==[]
        and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
        and all(r['status']=='NATIVE_TERMINAL'for r in g.get(original.CORE_ROWS,{}).values()),
        'PACKAGE_NO_ACTIVE_GPU_OR_UNKNOWN_EFFECTS')
    before=scope.first512_forecast(global_bytes,historical_LOCAL_bytes)
    authority.need(not before['cap_changes_required'],'PACKAGE_FULL_FUTURE_FITS_ADOPTED_CAPS')
    v=deepcopy(g);amount=resources()
    p=dict(schema=SCHEMA,scope=scope.SCOPE,grant_sha256=authority.GRANT_SHA,
        cap_authority_sha256=authority.sha(authority.raw(m)),
        T_CD=m['T_CD'],deadline_epoch=m['deadline_epoch'],stop_new_training_epoch=m['stop_new_training_epoch'],
        reserved_epoch=now_epoch,before_GLOBAL_sha256=authority.sha(global_bytes),
        historical_LOCAL_sha256=authority.sha(historical_LOCAL_bytes),
        original_resources=amount,remaining_resources=deepcopy(amount),session_plan=sessions(),
        completed_sessions=[],registered_requests=[],scheduler_horizon=1024,
        raw_parent_files_identity=RAW_PARENT_FILES_ID,statistics_sha256=R0,
        panel_sha256=PANEL_SHA,full_cursor_sha256=CURSOR_SHA,
        Core_LOCAL_sync='PENDING_CANONICAL_4080_NODE_LOCAL_NO_COPY',actual_LOCAL_written=False,
        native_admission=False,Core_measurement_gate_for_formal_training=True,
        status='GLOBAL_RESERVED_CORE_LOCAL_PENDING_NO_DISPATCH')
    v[FIELD]=p
    for key in TRACKED:v['reserved'][key]+=amount[key]
    v['CPU_reserved_s']+=amount['Hopper_CPU_s'];v['storage_reserved_bytes']+=amount['Hopper_storage_bytes']
    v['revision']+=1;validate(v);original.validate_global(v)
    after=scope.first512_forecast(authority.raw(v),historical_LOCAL_bytes)
    authority.need(not after['cap_changes_required'] and all(
        after['dimensions'][k]['projected']==before['dimensions'][k]['projected']
        for k in before['dimensions']), 'PACKAGE_RESERVATION_IS_NOT_NEW_SPEND_OR_DOUBLE_FUTURE')
    authority.need(all(v[k]==x for k,x in g.items()if k not in
        ('revision','reserved','CPU_reserved_s','storage_reserved_bytes')),
        'PACKAGE_ALL_PAID_COSTS_OLD_ROWS_SCOPE_AND_HISTORY_UNCHANGED')
    return dict(proposed_global=v,full_future_proof=after,actual_LOCAL_changed=False,
                native_admission=False,new_requests=0,new_PBS=0,new_GPU=0)
