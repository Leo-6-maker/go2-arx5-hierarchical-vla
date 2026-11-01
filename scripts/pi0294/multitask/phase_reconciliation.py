"""Additive read-only reconciliation of the closed Phase4096 chain.

The sealed manager assumes every event has ``value``.  Its existing failure
writer instead emits terminal_failure directly.  Normalize only these two
known representations; never suppress malformed records or replace scores.
"""
import hashlib
import json
import math
from pathlib import Path

VERSION = 'PHASE4096_ADDITIVE_RECONCILIATION_V1'
KINDS = frozenset(('qualification', 'warmup_tick', 'warmup_complete',
    'observation', 'planned_H40', 'submitted_interval', 'native_tick',
    'completed_interval', 'native_tick_failure', 'terminal_failure'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError('OBJECT_REQUIRED:' + str(path))
    return value


def event_value(record):
    if not isinstance(record, dict) or record.get('kind') not in KINDS:
        raise ValueError('UNKNOWN_EVENT_SCHEMA')
    kind = record['kind']
    if set(record) == {'kind', 'value'} and isinstance(record['value'], dict):
        value = record['value']
    elif kind == 'terminal_failure' and set(record) == {
            'kind', 'error', 'actual_step', 'actual_state'}:
        value = {k: v for k, v in record.items() if k != 'kind'}
    else:
        raise ValueError('MALFORMED_EVENT:' + kind)
    if kind in ('native_tick', 'native_tick_failure', 'terminal_failure', 'observation'):
        state = value.get('actual_state')
        if not isinstance(state, dict):
            raise ValueError('ACTUAL_STATE_REQUIRED:' + kind)
        if not isinstance(state.get('time_s'), (int, float)) or not math.isfinite(state['time_s']):
            raise ValueError('FINITE_NATIVE_TIME_REQUIRED')
    if kind in ('native_tick_failure', 'terminal_failure'):
        if not isinstance(value.get('error'), str) or not value['error']:
            raise ValueError('FAILURE_REASON_REQUIRED')
    return kind, value


def summarize_case(records, condition, audit):
    for key in ('task_success', 'complete_episode', 'original_native_safety_pass',
                'native_LIBERO_goal_final', 'additional_support_release_withdraw_hold_pass'):
        if type(audit.get(key)) is not bool:
            raise ValueError('BOOLEAN_NATIVE_SCORE_REQUIRED:' + key)
    if (type(condition.get('success')) is not bool
            or type(condition.get('valid_model_condition')) is not bool
            or condition['success'] != audit['task_success']
            or condition['hard_native_safety'] != (not audit['original_native_safety_pass'])):
        raise ValueError('ORIGINAL_SCORE_MISMATCH')
    if audit['task_success'] and (not all(audit[k] for k in (
            'complete_episode', 'original_native_safety_pass', 'native_LIBERO_goal_final',
            'additional_support_release_withdraw_hold_pass')) or audit['integrity_problems']):
        raise ValueError('SUCCESS_WITHOUT_COMPLETE_NATIVE_PROOF')
    first = {}; last_grasp = None; first_loss = None; previous = False
    initial_z = None; maximum_rise = 0.; failures = []; submitted = completed = 0
    for record in records:
        kind, value = event_value(record)
        if kind == 'submitted_interval': submitted += 1
        if kind == 'completed_interval': completed += 1
        if kind == 'observation' and value['index'] == 0:
            initial_z = float(value['actual_state']['object_pose'][2][3])
        if kind not in ('native_tick', 'native_tick_failure', 'terminal_failure'):
            continue
        s = value['actual_state']; t = float(s['time_s']) - 5.
        if kind == 'terminal_failure':
            failures.append(dict(task_s=t, reason=value['error'], event=kind))
            continue
        if type(s.get('native_grasp')) is not bool:
            raise ValueError('BOOLEAN_NATIVE_GRASP_REQUIRED')
        grasp = s['native_grasp']
        if initial_z is not None:
            maximum_rise = max(maximum_rise, (float(s['object_pose'][2][3])-initial_z)*1000)
        if grasp:
            first.setdefault('native_grasp_s', t); last_grasp = t
        if previous and not grasp and first_loss is None: first_loss = t
        previous = grasp
        if s.get('object_supported_by_shelf'): first.setdefault('shelf_support_s', t)
        if s.get('native_libero_goal'): first.setdefault('native_goal_s', t)
        if kind == 'native_tick_failure':
            failures.append(dict(task_s=t, reason=value['error'], event=kind))
    if completed != audit['completed_intervals'] or submitted-completed not in (0, 1):
        raise ValueError('ACTUAL_EXECUTED_COMMAND_COUNT_MISMATCH')
    if condition['hard_native_safety']:
        if not failures or audit['terminal_error'] != failures[-1]['reason']:
            raise ValueError('SAFETY_FAILURE_MUST_HAVE_ACTUAL_RECORD')
        category = 'NATIVE_SAFETY_STOP'
    elif condition.get('gate_rejected'): category = 'ORIGINAL_REFERENCE_GATE_REJECTION'
    elif condition['success']: category = 'FULL_STUDENT_TASK_SUCCESS'
    elif audit['native_LIBERO_goal_final']: category = 'FINAL_GEOMETRY_TRUE_BUT_SUPPORT_RELEASE_WITHDRAW_HOLD_FAILED'
    else: category = 'FULL_DURATION_TASK_GOAL_NOT_REACHED'
    return dict(id=condition['id'], valid=condition['valid_model_condition'],
        success=condition['success'], classification=category,
        submitted_commands=submitted, completed_commands=completed,
        first_events=first, last_native_grasp_s=last_grasp,
        first_native_grasp_loss_s=first_loss, maximum_book_rise_mm=maximum_rise,
        first_native_failure=failures[0] if failures else None,
        original_final_native_goal=audit['native_LIBERO_goal_final'],
        original_support_release_withdraw_hold=audit['additional_support_release_withdraw_hold_pass'],
        original_integrity_problems=audit['integrity_problems'],
        legal_release_can_also_end_grasp=True, student_DEV_not_TRAIN=True)


def reconcile(root, *, expected_seal_sha256, unit_reader, old_source_root):
    """No writes, model calls, integration, rendering, or changes to originals."""
    root = Path(root); anchors = {}
    def pinned(path):
        path = Path(path); anchors[str(path)] = sha(path); return read(path)
    if sha(root/'CAPSULE_SEAL.json') != expected_seal_sha256:
        raise PermissionError('EXACT_OLD_CAPSULE_REQUIRED')
    seal = pinned(root/'CAPSULE_SEAL.json')
    for name, pin in seal['files'].items():
        if sha(root/name) != pin: raise PermissionError('OLD_CAPSULE_CHANGED:' + name)
    config = pinned(root/'QUEUE.json'); old_state = pinned(root/'STATE.json')
    if old_state['TRAIN'] != dict(status='CLOSED_AUDITED', step=4096, cursor=131072):
        raise PermissionError('TRAIN_NOT_CLOSED')
    if any(old_state['diagnostics'][a] != 'CLOSED_AUDITED_96_GENERATIONS' for a in ('uniform', 'phase')):
        raise PermissionError('DIAGNOSTICS_NOT_CLOSED')
    for path in sorted((root/'steps').glob('*.json')):
        if pinned(path)['status'] != 'COMPLETE': raise PermissionError('UNKNOWN_OLD_STEP')
    actors = [root, Path(old_source_root)]; runs = {}
    for prefix in ('BOOK_PHASE_B2_S4096', 'BOOK_PHASE_TRAIN32_UNIFORM4096',
                   'BOOK_PHASE_TRAIN32_PHASE4096', 'BOOK_PHASE4096_DEV10'):
        staged = pinned(root/(prefix+'_STAGED.private.json'))
        closed = pinned(root/(prefix+'_COMPLETED_CLOSED.private.json'))
        final = pinned(root/(prefix+'_FINAL_INDEPENDENT.private.json'))
        actor = Path(staged['root']); actors.append(actor)
        manifest = actor/'RUN_MANIFEST.json'
        if sha(manifest) != staged['manifest_sha256']: raise PermissionError('OLD_RUN_IDENTITY')
        pinned(manifest); lease = actor/'control/LEASE.json'
        if (sha(lease) != closed['lease_sha256'] or sha(lease) != final['lease_sha256']
                or pinned(lease)['status'] != 'CLOSED' or closed['actual'] != final['actual']):
            raise PermissionError('EXACT_CLOSED_LEASE_AND_ACTUAL_COST_REQUIRED')
        result = actor/'RESULT.json'
        if sha(result) != closed['actual']['result_sha256']: raise PermissionError('CLOSED_RESULT_CHANGED')
        runs[prefix] = dict(root=str(actor), result=pinned(result), closed=closed)
        if (actor/'STOP.json').exists(): pinned(actor/'STOP.json')
    units = {str(p): unit_reader(p.name+'.service') for p in actors}
    if any(int(u['MainPID']) != 0 for u in units.values()): raise PermissionError('OLD_ACTOR_STILL_RUNNING')
    source = pinned(Path(old_source_root)/'STATE.json')
    if source['source_attempts'] or source['recoveries'] or source['exports']:
        raise PermissionError('OLD_SOURCE_EFFECT_REQUIRES_RECONCILIATION')
    source_calls = []
    for path in sorted((Path(old_source_root)/'calls').glob('*/INTENT.json')):
        intent = pinned(path); reply = path.with_name('RESULT.json')
        if intent['effect'] == 'MUTATION' and not reply.exists():
            raise PermissionError('UNKNOWN_OLD_SOURCE_EFFECT')
        if reply.exists(): pinned(reply)
        source_calls.append(dict(operation=intent['operation'], effect=intent['effect'],
            exact_reply_exists=reply.exists()))
    prepared = pinned(root/'BOOK_PHASE_B2_S4096_PREPARED.private.json')
    local_path = Path(prepared['LOCAL']); local = read(local_path)
    for key in ('unknown_effects', 'pending_updates', 'pending_queries'):
        if local[key]: raise PermissionError('UNKNOWN_CANONICAL_EFFECT:' + key)
    dev = runs['BOOK_PHASE4096_DEV10']; result = dev['result']; folder = Path(dev['root'])
    audit = pinned(root/'BOOK_PHASE4096_DEV10_INDEPENDENT_AUDIT.private.json')
    if audit['status'] != 'PASS_INDEPENDENT_ORIGINAL_DEV10_ACTUAL_MODEL_INPUTS_COMMANDS_AND_NATIVE_SCORE':
        raise PermissionError('INDEPENDENT_DEV_AUDIT_REQUIRED')
    cases = []
    for condition in result['conditions']:
        raw = folder/'conditions'/condition['id']; original_audit = pinned(raw/'INDEPENDENT_RAW_AUDIT.json')
        if anchors[str(raw/'INDEPENDENT_RAW_AUDIT.json')] != condition['audit_sha256']:
            raise ValueError('ORIGINAL_AUDIT_CHANGED')
        if sha(raw/'TERMINAL.json') != condition['terminal_sha256']: raise ValueError('TERMINAL_CHANGED')
        pinned(raw/'TERMINAL.json'); records = raw/'ACTUAL_RECORDS.jsonl'
        raw_pin = sha(records); anchors[str(records)] = raw_pin
        if raw_pin != condition['raw_sha256'] or original_audit['raw_sha256'] != raw_pin:
            raise ValueError('RAW_SOURCE_CHANGED')
        with records.open() as stream:
            cases.append(summarize_case((json.loads(line) for line in stream), condition, original_audit))
    counts = dict(valid=sum(c['valid'] for c in cases), successes=sum(c['success'] for c in cases),
        unrun=10-len(cases), hard_native_safety=sum(c['classification']=='NATIVE_SAFETY_STOP' for c in cases))
    if (counts['valid'] != result['valid_conditions'] or counts['successes'] != result['successes']
            or counts['valid'] != audit['valid_conditions'] or counts['successes'] != audit['successes']
            or counts['unrun'] != audit['remaining_unrun']): raise ValueError('ORIGINAL_COUNTS_DISAGREE')
    return dict(schema=VERSION, status='OLD_CHAIN_ACTUALLY_CLOSED_ADDITIVE_RECONCILIATION_PASS',
        counts=counts, conditions=cases, original_root=str(root), original_capsule_sha256=expected_seal_sha256,
        original_STATE_status=old_state['status'], original_summary_error=old_state['first_blocker'],
        original_source_root=str(old_source_root), actors=units, source_sha256=anchors,
        old_source_calls=source_calls, canonical_LOCAL_no_unknown_or_pending=True,
        canonical_LOCAL_observed_sha256=sha(local_path),
        original_STATE_RESULT_STOP_and_capsules_unchanged=True,
        new_models_integrations_rendering=0, new_native_safety_qualification_still_required=True,
        permission_scope='NEW_COLD_SUCCESSOR_ONLY_NOT_REOPENING_OLD_DEV')


def require_reconciled_closed(path, *, expected_sha256, unit_reader):
    path = Path(path)
    if sha(path) != expected_sha256: raise PermissionError('EXACT_RECONCILIATION_REQUIRED')
    result = read(path)
    if result.get('schema') != VERSION or result.get('status') != 'OLD_CHAIN_ACTUALLY_CLOSED_ADDITIVE_RECONCILIATION_PASS':
        raise PermissionError('ACTUAL_CLOSED_RECONCILIATION_REQUIRED')
    for name, pin in result['source_sha256'].items():
        if sha(name) != pin: raise PermissionError('ORIGINAL_CLOSED_SOURCE_CHANGED')
    for name in result['actors']:
        if int(unit_reader(Path(name).name+'.service')['MainPID']) != 0:
            raise PermissionError('OLD_ACTOR_STILL_RUNNING')
    return result
