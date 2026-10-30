"""One full-cost continuation of the closed, zero-model D25 original panel.

Uses the existing paired package and onceIO. No D25 output credit, old cost
refund, changed science, new cap, or change to other replacement limits.
"""
from copy import deepcopy
import json
import re
from . import mix_accounting as a, successor_cd as s

FIELD = 'V5_CD_D25_known_zero_original_panel_continuation'
FAILED = '9dd25c79-9a67-482a-bce7-d88a0fc922bc'
SOURCE = 'c446323ae71ec69761f743f5d68c76a2078ecdad'
RUN = 'v5-cd-d25-n294101-02'


def need(ok, why):
    if not ok:
        raise PermissionError('V5_CD_D25_CONTINUATION_' + why)


def validate(g):
    m = g.get(FIELD)
    if m is None:
        return None
    from . import mix_core_accounting as c
    r = g[c.FIELD][FAILED]
    need(m.get('schema') == 'V5_CD_D25_EXACT_ZERO_FULL_COST_CONTINUATION_V1'
         and m['failed_request_id'] == FAILED and m['failed_source_sha'] == SOURCE
         and m['scope_grant_sha256'] == s.GRANT_SHA
         and m['T_CD'] == s.validated(g)['T_CD']
         and m['deadline_epoch'] == s.validated(g)['deadline_epoch'], 'SAME_SCOPE_AND_CLOCK')
    need(a.sha(a.raw(r)) == m['failed_row_sha256']
         and r['source_sha'] == SOURCE and r['evaluation_candidate'] == 'D25'
         and r['status'] == 'NATIVE_TERMINAL' and r['native_exit_status'] == 1
         and r['query_attempts'] == r['model_load_attempts'] == 0
         and r['unknown_query_intents'] == []
         and c.digest(m['native']) == r['terminal_native_sha256']
         and c.digest(m['effects']) == r['terminal_effect_summary_sha256'], 'UNCHANGED_KNOWN_ZERO_TERMINAL')
    from .evaluation_core_accounting import terminal_values
    # Replay the original validation at its actual settlement time. A current
    # process/canonical-ledger read is checked separately before adoption.
    values = terminal_values(r, request_id=FAILED, native_proof=m['native'],
                             effects=m['effects'], now_epoch=r['settled_epoch'])
    need(values['queries'] == values['loads'] == 0 and not values['unknown']
         and values['gpu'] == r['conservative_GPU_s']
         and values['cpu'] == r['conservative_CPU_s'], 'REAL_ORIGINAL_TERMINAL_VALUES')
    h = m['history_audit']['history']
    need(h['valid_conditions'] == 0 and h['complete_DEV24'] is False
         and m['settlement']['physical']['episodes'] == 0
         and m['settlement']['physical']['constructors'] == 0
         and m['settlement']['physical']['live'] == m['settlement']['physical']['shadow'] == 0
         and m['settlement']['terminal_native_sha256'] == r['terminal_native_sha256']
         and m['settlement']['request_id'] == FAILED, 'NO_BEHAVIOR_OR_PHYSICS_TO_REPLAY')
    need(set(m['closed_leases']) == {'GPU', 'physical', 'control'}, 'THREE_CLOSED_LEASES')
    for role, lease in m['closed_leases'].items():
        need(lease['status'] == 'CLOSED' and lease['all_actual_workers_terminal'] is True,
             'REAL_CLOSED_' + role)
        if role != 'control':
            need(lease['terminal_native_sha256'] == r['terminal_native_sha256']
                 and lease['run_uuid'] == r['child_run_uuid'], 'SAME_CLOSED_' + role)
        else:
            need(lease['run_uuid'] == 'v5-cd-d25-eval-owner-01'
                 and lease['original_owner_OS_exit_status'] == 0
                 and lease['terminal_result_sha256'] == a.sha(a.raw(m['owner_result'])), 'NATURAL_OWNER_EXIT')
    need(m['resources'] == a.stage_vector('EVAL', Core_GPU_projection=True)
         and m['prepaid_control'] == c.CONTROL and m['output_credit_bytes'] == 0
         and m['all_old_costs_and_history_retained'] is True
         and g[a.PACKAGE]['engineering_replacements_used'] == 4, 'FULL_COST_NO_OTHER_LIMIT_CHANGE')
    need(c.canonical_uuid(m['replacement_request_id']) and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id'] != FAILED and m['replacement_run_uuid'] == RUN
         and re.fullmatch('[0-9a-f]{40}', m['source_sha']), 'ONE_NEW_COLD_IDENTITY')
    replacement = g[c.FIELD].get(m['replacement_request_id'])
    from . import mix_core_d25_local_successor as successor
    next_marker = successor.validated(g)
    permitted = (FAILED, m['replacement_request_id']) + ((next_marker['replacement_request_id'],) if next_marker else ())
    from .mix_core_d25_admission_repair import validate as admission_repair
    repaired05=admission_repair(g,next_marker)if next_marker else None
    if repaired05:permitted+=(repaired05['replacement_request_id'],)
    need(all(row['request_id'] in permitted
             for row in g[c.FIELD].values() if row['evaluation_candidate'] == 'D25'), 'ONLY_EXACT_REPLACEMENT')
    if replacement is not None:
        need(replacement['evaluation_candidate'] == 'D25'
             and replacement['submission_nonce'] == m['replacement_submission_nonce']
             and replacement['child_run_uuid'] == RUN and replacement['source_sha'] == m['source_sha']
             and replacement['checkpoint_manifest_sha256'] == r['checkpoint_manifest_sha256']
             and replacement['planned_panel_sha256'] == r['planned_panel_sha256'], 'ORIGINAL_MODEL_AND_FULL_PANEL')
    return m


def propose(global_bytes, local_bytes, *, closed_readback, new_request_id, new_nonce, source_sha, now_epoch):
    from . import mix_core_accounting as c
    g, l = json.loads(global_bytes), json.loads(local_bytes)
    c.checked(g); c.paired(g, l)
    need(FIELD not in g and FIELD not in l and not g['pending_updates'] and not g['unknown_effects']
         and all(r['status'] == c.STATES[3] for r in g[c.FIELD].values())
         and all(r['status'] in a.TERMINAL for r in g['PBS_submissions']), 'ONCE_COLD_BOUNDARY')
    cap = closed_readback
    need(cap['status'] == 'INDEPENDENT_ACTUAL_D25_ZERO_BEHAVIOR_NATIVE_TERMINAL_READBACK'
         and cap['request_id'] == FAILED and cap['LOCAL_sha256'] == a.sha(local_bytes)
         and 0 <= now_epoch - cap['observed_epoch'] <= 120
         and now_epoch < s.validated(g)['deadline_epoch'] - c.GPU_UPPER, 'FRESH_CANONICAL_TERMINAL')
    stage = c.key('D25'); p = g[a.PACKAGE]
    need(p['completed_stages'] == c.before_stage('D25', g)
         and l['V5_CD_Core_CPU_substages'][stage]['request_id'] == FAILED
         and l['V5_CD_Core_CPU_substages'][stage]['status'] == 'SETTLED_KNOWN_NATIVE_TERMINAL', 'ORIGINAL_D25_NEXT')
    identities = [*g[c.FIELD].values(), *g['PBS_submissions'], *g.get('Core_native_GPU_allocations', {}).values()]
    need(not any(r.get('request_id') == new_request_id or r.get('submission_nonce') == new_nonce for r in identities), 'DISTINCT_IDENTITY')
    f = cap['files']; intent = f['resident/DUAL_FINANCIAL_TERMINAL_INTENT.json']['value']
    m = dict(schema='V5_CD_D25_EXACT_ZERO_FULL_COST_CONTINUATION_V1',
        failed_request_id=FAILED, failed_source_sha=SOURCE, failed_row_sha256=a.sha(a.raw(g[c.FIELD][FAILED])),
        scope_grant_sha256=s.GRANT_SHA, T_CD=s.validated(g)['T_CD'], deadline_epoch=s.validated(g)['deadline_epoch'],
        adopted_epoch=now_epoch, source_sha=source_sha, replacement_request_id=new_request_id,
        replacement_submission_nonce=new_nonce, replacement_run_uuid=RUN,
        native=deepcopy(intent['native']), effects=deepcopy(intent['effects']),
        history_audit=deepcopy(f['resident/INDEPENDENT_PANEL_HISTORY_AUDIT.json']['value']),
        owner_result=deepcopy(f['OWNER_RESULT.json']['value']),
        settlement=deepcopy(l['V5_CD_Core_terminal_settlements'][FAILED]),
        closed_leases={k:deepcopy(v['value']) for k,v in cap['leases'].items()},
        resources=a.stage_vector('EVAL',Core_GPU_projection=True), prepaid_control=deepcopy(c.CONTROL),
        original_prepaid_control=deepcopy(p['prepaid_control'][stage]), output_credit_bytes=0,
        all_old_costs_and_history_retained=True)
    v,w = deepcopy(g),deepcopy(l)
    active = {k:cost-c.CONTROL.get(k,0) for k,cost in m['resources'].items()}
    for val in (v,w):
        val[FIELD] = deepcopy(m)
        for k,cost in active.items(): val[a.PACKAGE]['remaining_resources'][k] += cost
        val[a.PACKAGE]['prepaid_control'][stage] = dict(costs=deepcopy(c.CONTROL), epoch=now_epoch,
            financial_upper_not_actual_process_CPU=True, continuation_sha256=a.sha(a.raw(m)),
            failed_request_id=FAILED, replacement_request_id=new_request_id)
    for k in ('H200_s','neural_queries','optimizer_updates'): v['reserved'][k] += active[k]
    v['CPU_reserved_s'] += active['Hopper_CPU_s']; v['CPU_s'] += c.CONTROL['Hopper_CPU_s']
    v['storage_reserved_bytes'] += active['Hopper_storage_bytes']
    v['new_storage_current_bytes'] += c.CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes'] = max(v['new_storage_peak_bytes'], v['new_storage_current_bytes'])
    w['GPU_reserved_s'] += active['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:
        w['CPU_reserved_s'][site] += active[k]; w['CPU_s'][site] += c.CONTROL.get(k,0)
    w['network_transfer_accounted_bytes'] += c.CONTROL['transfer_bytes']
    w['Core_native_GPU_current_storage_bytes'] += c.CONTROL['Core_storage_bytes']
    v['revision'] += 1; c.checked(v); c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full = first512_forecast(a.raw(v),a.raw(w))
    need(not full['cap_changes_required'], 'ALL_REMAINING_WORK_MATRIX_FINAL:' + str(full['cap_changes_required']))
    return dict(proposed_global=v, proposed_LOCAL=w, full_future_proof=full,
                original_onceIO_required=True, native_registration=False, model_or_physics_permission=False)
