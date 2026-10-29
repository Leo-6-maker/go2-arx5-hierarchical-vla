"""Explicit human-approved D25-first execution, inside the existing package.

The unavailable Core ledger is never represented as freshly read or written.
Vanda remains the existing neural allocation authority. Its full future proof
uses a conservative, explicit projection of the last confirmed Core ledger.
Core execution stays blocked until that ledger is actually reconciled.
"""
from copy import deepcopy
import json
from . import mix_accounting as a, successor_cd as s

FIELD = 'V5_CD_D25_Vanda_first'
AUTH_SHA = '142887779054722881f4fe982d049a41f507624a3e86f6ca49ff87cc87770c0e'
ORDER = ('TRAIN:C50:1024', 'TRAIN:D25:1024',
         'EVAL:C50:N294101', 'EVAL:D25:N294101')
LOCAL_SHA = '7cea556c4c80765582644ce1ed47d71f3c0c3084b445d02dd16a55854b386f1a'
CORE_STORAGE = 335556053016 + 34051160888
LOCAL_STORAGE = 9126805504


def need(ok, why):
    if not ok:
        raise PermissionError('V5_VANDA_FIRST_' + why)


def validated(g):
    m = g.get(FIELD)
    if m is None:
        return None
    original = s.validated(g)
    need(m.get('schema') == 'V5_CD_EXPLICIT_D25_FIRST_EXISTING_PACKAGE_V1'
         and m.get('authority_sha256') == AUTH_SHA
         and m.get('physical_order') == list(ORDER)
         and m.get('original_scope_order') == list(s.ORDER)
         and m.get('confirmed_LOCAL_sha256') == LOCAL_SHA
         and m.get('T_CD') == original['T_CD']
         and m.get('deadline_epoch') == original['deadline_epoch']
         and original['T_CD'] <= a.number(m.get('adopted_epoch'))
             < original['stop_new_training_epoch']
         and m.get('LOCAL_sync') in ('PENDING', 'SYNCED')
         and m.get('LOCAL_preparation_upper_s') == 60
         and m.get('no_Core_admission_from_snapshot') is True
         and m.get('old_UNKNOWN_unchanged') is True,
         'EXACT_APPROVED_ORDER_CLOCK_AND_CONFIRMED_SNAPSHOT')
    from . import mix_core_premodel_repair as repair
    need(repair.validated(g) is not None, 'ORIGINAL_C50_EVAL_COMPUTE_RETAINED')
    return m


def budget_LOCAL(g, local):
    """Pure budget projection; never a native read, write or Core permission."""
    m = validated(g)
    from . import mix_core_recovery as recovered
    if recovered.FIELD in g:
        value=recovered.budget_LOCAL(g,local)
        from scripts.pi0294.generalization import minimal_execution, minimal_retry_execution
        if (minimal_retry_execution.FIELD in g or any(
                r.get('platform_training_version')==minimal_retry_execution.VERSION
                for r in g['PBS_submissions'])):
            from scripts.pi0294.generalization import minimal_partial_completion_scope as completion
            from scripts.pi0294.generalization import minimal_saved512_recovery_scope as saved512
            from scripts.pi0294.generalization import minimal_e2only_execution as e2only
            return e2only.budget_LOCAL(g,saved512.budget_LOCAL(g,completion.budget_LOCAL(g,minimal_retry_execution.budget_LOCAL(g,value))))
        if minimal_execution.FIELD in g:return minimal_execution.budget_LOCAL(g,value)
        return value
    if m is None or m['LOCAL_sync'] == 'SYNCED':
        return local
    need(a.sha(a.raw(local)) == LOCAL_SHA, 'EXACT_LAST_CONFIRMED_LOCAL_BYTES')
    need(local['unknown_effects'] == local['pending_updates']
         == local['pending_queries'] == [], 'NO_UNKNOWN_LOCAL_LEDGER_EFFECT')
    from . import mix_core_premodel_repair as repair
    restored = repair.allocation(g, 'C50')
    w = deepcopy(local)
    w['GPU_reserved_s'] += restored['H200_s']
    for key, site in (('Hopper_CPU_s', 'Hopper'), ('Core_CPU_s', 'CoreLab'),
                      ('local_CPU_s', 'local_PC')):
        w['CPU_reserved_s'][site] += restored[key]
    w['CPU_reserved_s']['local_PC'] -= m['LOCAL_preparation_upper_s']
    w['CPU_s']['local_PC'] += m['LOCAL_preparation_upper_s']
    control = g[a.PACKAGE].get('prepaid_control', {}).get('TRAIN:D25:1024')
    if control is not None:
        costs = control['costs']
        need(costs == dict(Hopper_CPU_s=1800, Hopper_storage_bytes=64*1024**2,
                          transfer_bytes=256*1024**2, local_CPU_s=15),
             'ORIGINAL_D25_CONTROL_ONLY')
        for key, site in (('Hopper_CPU_s', 'Hopper'), ('local_CPU_s', 'local_PC')):
            w['CPU_reserved_s'][site] -= costs[key]
            w['CPU_s'][site] += costs[key]
        w['network_transfer_accounted_bytes'] += costs['transfer_bytes']
    for row in g['PBS_submissions']:
        if row.get('platform_training_version') != a.VERSION or row.get('arm') != 'D25':
            continue
        need(not any(x.get('request_id') == row['request_id']
                     for x in local['PBS_submissions']), 'NO_DUPLICATE_D25_MIRROR')
        if row['status'] in a.TERMINAL:
            need(row['status'] == 'NATIVE_TERMINAL', 'ACTUAL_TERMINAL_EFFECTS')
            w['GPU_reserved_s'] -= 22560
            w['CPU_reserved_s']['Hopper'] -= 54000
            w['GPU_allocated_s'] += row['conservative_H200_s']
            w['CPU_s']['Hopper'] += row['conservative_CPU_s']
            w['optimizer_updates'] += row['committed_optimizer_updates']
            w['neural_queries'] += row['query_attempts']
        w['PBS_submissions'].append(deepcopy(row))
    need(w['GPU_reserved_s'] >= 0 and all(x >= 0 for x in w['CPU_reserved_s'].values()),
         'NO_PROJECTION_RESERVATION_UNDERFLOW')
    return w


def propose(global_bytes, local_bytes, *, authority_bytes, closure_capture_bytes,
            metadata_capture_bytes, replacement_request_id,
            replacement_submission_nonce, now_epoch):
    """One GLOBAL proposal, retaining the exact original LOCAL catch-up plan."""
    need(a.sha(authority_bytes) == AUTH_SHA, 'DIRECT_HUMAN_START_NOW_BYTES')
    authority = json.loads(authority_bytes)
    need(authority['direct_human_answer'] == 'start now'
         and authority['physical_execution_order'] == list(ORDER),
         'ONLY_ORIGINAL_D25_THEN_TWO_PANELS')
    g, l = json.loads(global_bytes), json.loads(local_bytes)
    need(FIELD not in g and FIELD not in l and a.sha(local_bytes) == LOCAL_SHA,
         'ONCE_NEW_BOUNDARY_EXACT_LAST_CONFIRMED_LOCAL')
    need(all(r['status'] in a.TERMINAL for r in g['PBS_submissions'])
         and all(r['status'] == 'NATIVE_TERMINAL'
                 for r in g.get(a.CORE_ROWS, {}).values()), 'NO_ACTIVE_TASK_GPU')
    need(not any(r.get('arm') == 'D25' for r in g['PBS_submissions']),
         'NO_ALREADY_DISPATCHED_D25')
    from . import mix_core_premodel_repair as repair
    base = repair.propose(global_bytes, local_bytes,
        closure_capture_bytes=closure_capture_bytes,
        metadata_capture_bytes=metadata_capture_bytes,
        replacement_request_id=replacement_request_id,
        replacement_submission_nonce=replacement_submission_nonce,
        now_epoch=now_epoch)
    v, w = base['proposed_global'], base['proposed_LOCAL']
    marker = dict(schema='V5_CD_EXPLICIT_D25_FIRST_EXISTING_PACKAGE_V1',
        authority_sha256=AUTH_SHA, physical_order=list(ORDER),
        original_scope_order=list(s.ORDER), confirmed_LOCAL_sha256=LOCAL_SHA,
        before_GLOBAL_sha256=a.sha(global_bytes),
        T_CD=s.validated(g)['T_CD'], deadline_epoch=s.validated(g)['deadline_epoch'],
        adopted_epoch=now_epoch, LOCAL_sync='PENDING',
        LOCAL_preparation_upper_s=60,
        no_Core_admission_from_snapshot=True, old_UNKNOWN_unchanged=True)
    v[FIELD] = marker
    v[a.PACKAGE]['order'] = list(ORDER)
    v[a.PACKAGE]['remaining_resources']['local_CPU_s'] -= 60
    # w is a prospective once-reconciliation target, never called adopted.
    w[FIELD] = deepcopy(marker)
    w[a.PACKAGE]['order'] = list(ORDER)
    w[a.PACKAGE]['remaining_resources']['local_CPU_s'] -= 60
    w['CPU_reserved_s']['local_PC'] -= 60
    w['CPU_s']['local_PC'] += 60
    need(budget_LOCAL(v, l)['CPU_reserved_s'] == w['CPU_reserved_s']
         and budget_LOCAL(v, l)['GPU_reserved_s'] == w['GPU_reserved_s'],
         'PROJECTION_EQUALS_ORIGINAL_ONCE_REPAIR_ARITHMETIC')
    a.validate_global(v)
    full = s.recompute(v, l, Core_storage_upper_bytes=CORE_STORAGE,
                       local_storage_upper_bytes=LOCAL_STORAGE)
    need(not full['shortfalls'], 'ALL_ORIGINAL_REMAINING_WORK_MATRIX_FINAL_PAYABLE')
    return dict(proposed_global=v, proposed_LOCAL_when_reachable=w,
        actual_LOCAL_changed=False, full_remaining_proof=full,
        marker=marker, original_caps_and_clocks_unchanged=True,
        new_GPU_models_queries_or_physics=0)


def pending(g):
    m = validated(g)
    from . import mix_core_recovery as recovered
    return m is not None and m['LOCAL_sync'] == 'PENDING' and recovered.validate(g)is None


def reconcile_LOCAL(global_bytes, local_bytes, *, D25_terminal_capture):
    """Existing two-ledger onceIO catch-up, only after real D25 full closure.

    Caller must read canonical LOCAL on its actual host and persist both
    proposals with fresh CAS/readback. This pure function grants no execution.
    """
    from . import mix_stage_handoff as handoff
    from . import mix_core_premodel_repair as repair, mix_core_accounting as core
    g, l = json.loads(global_bytes), json.loads(local_bytes)
    from .mix_vanda_cohost import FIELD as cohost_field
    need(cohost_field not in g, 'COLOCATED_EXECUTION_REQUIRES_ITS_ACTUAL_TERMINAL_COST_MIRROR_NOT_OLD_D25_ONLY_CATCHUP')
    need(pending(g) and a.sha(local_bytes) == LOCAL_SHA,
         'ACTUAL_CANONICAL_LOCAL_STILL_EXACT_CONFIRMED_BEFORE_BYTES')
    proof = handoff.closed_endpoint(global_bytes=global_bytes, arm='D25',
                                    **D25_terminal_capture)
    need(g[a.PACKAGE]['completed_stages'] == list(ORDER[:2]),
         'BOTH_TRAINED_NO_EARLY_CORE_EVAL')
    v, w = deepcopy(g), budget_LOCAL(g, l)
    v[FIELD]['LOCAL_sync'] = 'SYNCED'
    w[FIELD] = deepcopy(v[FIELD])
    w[repair.FIELD] = deepcopy(v[repair.FIELD])
    w[a.PACKAGE] = deepcopy(v[a.PACKAGE])
    proof = handoff.closed_endpoint(global_bytes=a.raw(v), arm='D25',
                                    **D25_terminal_capture)
    row = next(x for x in v['PBS_submissions']
               if x.get('request_id') == proof['request_id'])
    w.setdefault('V5_CD_terminal_training_mirrors', {})[proof['request_id']] = dict(
        handoff=proof, actual_GPU_s=row['conservative_H200_s'],
        actual_CPU_s=row['conservative_CPU_s'],
        actual_updates=row['committed_optimizer_updates'],
        actual_queries=row['query_attempts'], old_COSTS_unchanged=True,
        new_GLOBAL_or_PBS_writes=0)
    core.checked(v)
    core.paired(v, w)
    need(not s.recompute(v, w, Core_storage_upper_bytes=CORE_STORAGE,
                         local_storage_upper_bytes=LOCAL_STORAGE)['shortfalls'],
         'ACTUAL_CATCHUP_RETAINS_ALL_FUTURE_WORK')
    return dict(proposed_global=v, proposed_LOCAL=w,
        LOCAL_before_sha256=a.sha(local_bytes),
        GLOBAL_before_sha256=a.sha(global_bytes),
        no_new_fee_or_scope=True, original_onceIO_required=True,
        native_IO=False, model_or_physics_admission=False)
