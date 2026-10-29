"""S1 addition to the original complete fourteen-dimension future forecast.

No shared-cap rewrite, refund, reservation or dispatch. The original forecast
retains old E1/E2/evaluation work, matrix and final protection. The human's new
S1 allowance is a named additional scope, usable only by these four attempts.
"""
from copy import deepcopy
import hashlib
import json

from scripts.pi0294.pv1.future_budget import _vector
from scripts.pi0294.generalization.minimal_scope import first512_forecast
from .task_path_entry import authority, need

GiB = 1024**3
# Partition the approved 14,400 core-seconds across Core and PC; do not reserve
# 14,400 on Core and then silently add another PC allowance.
REQUESTED = _vector(dict(Core_CPU_s=13800, local_CPU_s=600,
    episode_starts=4, physics_ticks=62012, engine_starts=96,
    Core_storage_bytes=8*GiB, local_storage_bytes=GiB, transfer_bytes=8*GiB))
# Exact newly approved resources, not a reusable increase for old experiments.
# The internal native constructor forecast needs no shared-cap increase.
SCOPED_ALLOWANCE = dict(REQUESTED, engine_starts=0)


def forecast(approval_bytes, global_bytes, local_bytes, *, observation, now_epoch):
    """Join actual full-byte identities and the original remaining-work model.

    A projection is not native permission: source, current process/queue,
    registry/lease, node-local dependency and task-safety guards still apply.
    """
    authority(approval_bytes, now_epoch=now_epoch)
    need(type(global_bytes) is bytes and type(local_bytes) is bytes,
         'COMPLETE_CANONICAL_BUDGET_BYTES')
    for key, raw in (('GLOBAL', global_bytes), ('LOCAL', local_bytes)):
        value = observation[key]
        need(value['sha256'] == hashlib.sha256(raw).hexdigest()
             and value['bytes'] == len(raw), 'ACTUAL_FULL_'+key+'_IDENTITY')
        need(type(value['observed_epoch']) in (int, float)
             and 0 <= now_epoch-value['observed_epoch'] <= 120,
             'FRESH_CANONICAL_'+key+'_OBSERVATION')
    g, l = json.loads(global_bytes), json.loads(local_bytes)
    need(g['pending_updates'] == g['unknown_effects'] == []
         and l['pending_updates'] == l['pending_queries'] == l['unknown_effects'] == []
         and l.get('Hopper_unknown_effects', []) == [], 'ORIGINAL_PENDING_OR_UNKNOWN_EFFECTS')
    original = first512_forecast(global_bytes, local_bytes)
    return add_to_original(original)


def add_to_original(original):
    """Finite arithmetic only, useful for isolated budget boundary tests.

    Native use must call forecast with full canonical bytes; this function's
    caller-supplied table cannot authorize a run or replace the original model.
    """
    need(set(original['dimensions']) == set(REQUESTED), 'ALL_ORIGINAL_FOURTEEN_DIMENSIONS')
    rows, shortfalls = {}, {}
    for name, amount in REQUESTED.items():
        old = original['dimensions'][name]
        value = old['projected']+amount
        cap = old['cap']
        supplement = SCOPED_ALLOWANCE[name]
        if cap is None:
            need(name == 'PBS_submissions' and old.get('count_still_accounted') is True,
                 'ONLY_ALREADY_ADOPTED_PBS_COUNT_UNLIMITED')
            missing = 0
        else:
            # S1 cannot pay an existing old-work shortfall. Its separate grant
            # covers only the S1 addition; every original obligation stays
            # inside its unchanged original cap.
            missing = max(0, old['projected']-cap, value-cap-supplement)
        rows[name] = dict(cap=cap, original_all_work_projected=old['projected'],
            S1_additional=amount, projected_with_S1=value, shortfall=missing,
            original_scope_shortfall=0 if cap is None else max(0, old['projected']-cap),
            separately_approved_S1_allowance=supplement,
            shared_cap_increase_needed_without_S1_grant=0 if cap is None else max(0, value-cap),
            combined_named_scope_ceiling=None if cap is None else cap+supplement,
            headroom=None if cap is None else cap+supplement-value)
        if missing:
            shortfalls[name] = missing
    return dict(schema='S1_APPROVED_TASK_PATH_ORIGINAL_FULL14_FORECAST_V1',
        status='ORIGINAL_SCOPE_AND_APPROVED_S1_ALLOWANCE_FIT_NOT_NATIVE_ADMISSION' if not shortfalls else 'RESOURCE_FORECAST_SHORTFALL',
        original_forecast=deepcopy(original), requested=dict(REQUESTED), dimensions=rows,
        shortfalls=shortfalls, approved_RGB_pairs_max=2084,
        total_new_Core_and_PC_CPU_s=REQUESTED['Core_CPU_s']+REQUESTED['local_CPU_s'],
        constructor_upper_is_native_WBC_and_planner_only=True,
        new_GR00T_queries=0, new_optimizer_updates=0, new_PBS_submissions=0,
        old_work_matrix_and_final_protection_preserved=True,
        shared_original_resource_caps_changed=False,
        S1_named_allowance=dict(SCOPED_ALLOWANCE),
        S1_allowance_spendable_by_old_work=False, S1_allowance_native_adopted=False,
        canonical_ledger_changed=False,
        resources_reserved=False, native_admission=False, launch_allowed=False)
