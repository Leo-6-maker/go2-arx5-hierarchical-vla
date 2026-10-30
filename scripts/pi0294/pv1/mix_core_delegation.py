"""Finite Core effect custody inside an existing, fully reserved allocation.

Pure proposals only: no IO, credentials, dispatch, heartbeat or scientific GO.
The original owner/onceIO must adopt both ledgers, persist every local debit
BEFORE its effect, and reconcile unknown replies without replay. The local
view is explicitly delegated; its original GLOBAL observation never becomes
fresh merely because Core observes a live process. Original qualification,
model release, physical accounting, safety and independent audit still apply.
"""
from copy import deepcopy
import json
import re

from . import mix_accounting as a, mix_core_accounting as c
from . import mix_core_admission as admission, mix_core_identity as identity
from . import mix_core_settlement as terminal

MODE = 'V5_CD_CORE_FINITE_RESERVED_LOCAL_EFFECTS_V1'
FIELD = 'V5_CD_Core_finite_delegations'
GRANT = 'V5_CD_CORE_FINITE_RESERVED_GRANT_V1'
VIEW = 'V5_CD_CORE_DELEGATED_EFFECT_VIEW_V1'
HELD = 'DELEGATED_HELD_UNTIL_CANONICAL_TERMINAL'
MERGED = 'CANONICAL_TERMINAL_MERGED_NO_REUSE'
_INPUTS = ('manifest_bytes', 'request_bytes', 'global_bytes', 'local_bytes',
           'snapshot_bytes', 'training_capture_bytes', 'population_bytes',
           'plan_bytes', 'prior_bytes', 'native_facts_bytes')


def need(ok, message):
    if not ok:
        raise PermissionError('V5_CD_CORE_DELEGATION_' + message)


def load(raw):
    need(type(raw) is bytes, 'ACTUAL_BYTES_REQUIRED')
    value = json.loads(raw, parse_constant=lambda _: need(False, 'FINITE_JSON'))
    need(type(value) is dict, 'OBJECT_REQUIRED')
    return value


def _digest(value):
    return a.sha(a.raw(value))


def _body(grant_bytes):
    grant = load(grant_bytes)
    from scripts.pi0294.generalization import minimal_core_delegation as first512
    if grant.get('schema')==first512.GRANT:return first512.body(grant_bytes)
    need(set(grant) == {'schema', 'body', 'body_sha256'}
         and grant['schema'] == GRANT and type(grant['body']) is dict
         and grant['body_sha256'] == _digest(grant['body']), 'GRANT_BODY_HASH')
    b = grant['body']; r = b['request']; m = b['manifest']; row = b['issued_row']
    need(set(b) == {'mode', 'request', 'manifest', 'issued_row', 'admission',
         'source_inputs', 'issued_epoch', 'execution_not_after_epoch',
         'accounting_not_after_epoch', 'limits', 'original_GLOBAL_revision'},
         'EXACT_GRANT_BODY')
    identity.request_identity(m, r, _digest(m)); c.validate_row(row)
    need(b['mode'] == r.get('global_accounting_mode')
         == m['execution'].get('global_accounting_mode')
         == row.get('global_accounting_mode') == MODE, 'EXPLICIT_COLD_SELECTOR')
    need(b['source_inputs']['manifest_bytes'] == _digest(m)
         and b['source_inputs']['request_bytes'] == _digest(r)
         and row['run_manifest_sha256'] == _digest(m)
         and row['request_sha256'] == _digest(r)
         and row['status'] == c.STATES[1] and c.progress(row) == (0, 0, 0),
         'EXACT_ZERO_EFFECT_BOUND_START_NOT_A_LATE_GRANT')
    proof = b['admission']; allocation = proof['allocation']
    need(proof['schema'] == 'V5_CD_CORE_FULL_REMAINING_NATIVE_RESOURCE_RECOMPUTATION_V1'
         and proof['request_id'] == r['request_id']
         and proof['request_sha256'] == _digest(r)
         and proof['GLOBAL_sha256'] == b['source_inputs']['global_bytes']
         and proof['LOCAL_sha256'] == b['source_inputs']['local_bytes']
         and proof['snapshot_sha256'] == b['source_inputs']['snapshot_bytes']
         and proof['shortfalls'] == {} and proof['all_remaining_costs_fit'] is True
         and proof['original_matrix_and_final_protected'] is True
         and proof['model_permission'] is False and proof['physics_authorized'] is False,
         'COMPLETE_ORIGINAL_RECOMPUTATION_NOT_A_GO_FLAG')
    identity.sealed_identity(m, r, allocation, run_uuid=r['run_uuid'],
        manifest_sha256=_digest(m), request_sha256=_digest(r), same_bound_process=row)
    now = a.number(b['issued_epoch'])
    need(0 <= now - allocation['native_process']['observed_epoch'] <= 120
         and b['execution_not_after_epoch'] == allocation['allocation_end_epoch'] - 2700
         and b['accounting_not_after_epoch'] == allocation['allocation_end_epoch']
         and now < b['execution_not_after_epoch'] < b['accounting_not_after_epoch']
         <= r['deadline_epoch'], 'ORIGINAL_CLOCK_AND_CLOSEOUT_RESERVE')
    expected = dict(generation_queries=row['reserved_neural_queries'], model_loads=2,
                    CPU_s=row['CPU_upper_s'], GPU_s=row['reserved_GPU_s'],
                    output_bytes=row['additional_output_upper_bytes'], optimizer_updates=0)
    need(b['limits'] == expected, 'ONLY_THIS_EXISTING_RESERVED_ENVELOPE')
    return b


def _marker(grant_bytes):
    b = _body(grant_bytes); row = b['issued_row']
    return dict(schema='V5_CD_CORE_FINITE_DELEGATION_CUSTODY_V1', status=HELD,
        request_id=row['request_id'], grant_sha256=a.sha(grant_bytes),
        issued_row_sha256=_digest(row), source_sha=row['source_sha'],
        native_handle=row['native_handle'], submission_nonce=row['submission_nonce'],
        issued_epoch=b['issued_epoch'], execution_not_after_epoch=b['execution_not_after_epoch'],
        original_GLOBAL_sha256=b['source_inputs']['global_bytes'],
        original_LOCAL_sha256=b['source_inputs']['local_bytes'])


def assert_remote_writer_allowed(g, request_id):
    """Required in the original remote writer before *any* non-STATE request.

    Never remove this marker to resume a legacy writer. Terminal merging uses
    the original pure charge function under one canonical owner/onceIO instead.
    """
    need(request_id not in g.get(FIELD, {}), 'REMOTE_WRITER_CANNOT_DOUBLE_SPEND_DELEGATED_REQUEST')
    return True


def issue(**inputs):
    """Recompute all original raw gates, then mark already-held resources.

    The returned grant is a proposal until both exact onceIO receipts and
    independent canonical readbacks are accepted by ``activate``. Issuance
    changes neither resource totals nor any row, package, cap or clock.
    """
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(load(inputs['request_bytes'])):
        from scripts.pi0294.generalization.minimal_core_delegation import issue as issue512
        return issue512(**inputs)
    need(set(inputs) == set(_INPUTS) | {'now_epoch'}, 'COMPLETE_ORIGINAL_ADMISSION_INPUTS')
    proof = admission.recompute(**inputs)
    g, l, m, r = (load(inputs[k]) for k in
                  ('global_bytes', 'local_bytes', 'manifest_bytes', 'request_bytes'))
    # Existing onceIO writes use this exact serializer. Reconstruction below
    # binds the entire before state, not only one request or a claimed PASS.
    for name, value in (('global_bytes', g), ('local_bytes', l),
                        ('manifest_bytes', m), ('request_bytes', r)):
        need(inputs[name] == a.raw(value), 'CANONICAL_INPUT_BYTES:' + name)
    row = deepcopy(g[c.FIELD][r['request_id']]); c.paired(g, l)
    need(r['request_id'] not in g.get(FIELD, {})
         and g.get(FIELD, {}) == l.get(FIELD, {})
         and all(v['status'] == MERGED for v in g.get(FIELD, {}).values()),
         'ONE_FRESH_DELEGATION_NO_OVERLAPPING_OWNER')
    body = dict(mode=MODE, request=r, manifest=m, issued_row=row, admission=proof,
        source_inputs={k:a.sha(inputs[k]) for k in _INPUTS},
        issued_epoch=inputs['now_epoch'],
        execution_not_after_epoch=proof['allocation']['allocation_end_epoch']-2700,
        accounting_not_after_epoch=proof['allocation']['allocation_end_epoch'],
        limits=dict(generation_queries=row['reserved_neural_queries'], model_loads=2,
                    CPU_s=row['CPU_upper_s'], GPU_s=row['reserved_GPU_s'],
                    output_bytes=row['additional_output_upper_bytes'], optimizer_updates=0),
        original_GLOBAL_revision=g['revision'])
    grant = dict(schema=GRANT, body=body, body_sha256=_digest(body))
    marker = _marker(a.raw(grant)); v, w = deepcopy(g), deepcopy(l)
    for value in (v, w):
        value.setdefault(FIELD, {})[r['request_id']] = deepcopy(marker)
    v['revision'] += 1
    c.checked(v); c.paired(v, w)
    return dict(grant=grant, proposed_global=v, proposed_LOCAL=w,
                original_onceIO_required=True, native_IO_performed=False,
                model_permission=False, physics_authorized=False)


def _receipt(intent_bytes, receipt_bytes, *, before, after):
    intent, receipt = load(intent_bytes), load(receipt_bytes)
    need(intent.get('schema') == 'DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
         and receipt.get('schema') == 'DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
         and intent.get('request_id') == receipt.get('request_id')
         and type(intent.get('target')) is str and intent['target'].startswith('/')
         and receipt.get('target') == intent['target']
         and intent.get('before_sha256') == receipt.get('before_sha256') == before
         and intent.get('proposed_sha256') == receipt.get('after_sha256') == after
         and receipt.get('intent_sha256') == a.sha(intent_bytes)
         and receipt.get('status') == 'CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
         and intent.get('native_dispatch_admitted') is receipt.get('native_dispatch_admitted') is False,
         'ACTUAL_ORIGINAL_ONCE_INTENT_RECEIPT_AND_READBACK')


def _initial(grant_bytes, activation):
    b = _body(grant_bytes)
    return dict(schema=VIEW, mode=MODE, grant_sha256=a.sha(grant_bytes),
        activation=deepcopy(activation),
        canonical_GLOBAL_observed_epoch=b['issued_epoch'],
        canonical_GLOBAL_is_fresh=False, canonical_GLOBAL_is_writable=False,
        canonical_GLOBAL_sha256=b['source_inputs']['global_bytes'],
        row=deepcopy(b['issued_row']), events=[], sequence=0,
        event_chain_head_sha256=_digest(dict(grant_sha256=a.sha(grant_bytes), activation=activation)),
        local_observed_epoch=activation['epoch'], unknown_effects=[],
        pending_effects=[], pending_updates=[], model_permission=False,
        physics_authorized=False, original_onceIO_required_before_effect=True)


def activate(grant_bytes, *, global_bytes, local_bytes, global_intent_bytes,
             global_receipt_bytes, local_intent_bytes, local_receipt_bytes, now_epoch):
    """Accept actual two-site adoption; no process/model release is granted."""
    from scripts.pi0294.generalization import minimal_core_delegation as first512
    if load(grant_bytes).get('schema')==first512.GRANT:
        return first512.activate(grant_bytes,global_bytes=global_bytes,local_bytes=local_bytes,
            global_intent_bytes=global_intent_bytes,global_receipt_bytes=global_receipt_bytes,
            local_intent_bytes=local_intent_bytes,local_receipt_bytes=local_receipt_bytes,now_epoch=now_epoch)
    b = _body(grant_bytes); marker = _marker(grant_bytes)
    g, l = load(global_bytes), load(local_bytes); c.checked(g); c.paired(g, l)
    rid = b['request']['request_id']; now = a.number(now_epoch)
    need(0 <= now-b['issued_epoch'] <= 120, 'FRESH_INDEPENDENT_DUAL_ADOPTION_READBACK')
    for value, label in ((g, 'global_bytes'), (l, 'local_bytes')):
        need(value.get(FIELD, {}).get(rid) == marker, 'SAME_CANONICAL_DELEGATION_MARKER')
        before = deepcopy(value); del before[FIELD][rid]
        if not before[FIELD]:
            del before[FIELD]
        if label == 'global_bytes':
            need(before['revision'] == b['original_GLOBAL_revision'] + 1, 'ONE_ADOPTION_REVISION')
            before['revision'] -= 1
        need(_digest(before) == b['source_inputs'][label], 'EXACT_ADOPTION_ONLY_NO_OTHER_CHANGE')
    _receipt(global_intent_bytes, global_receipt_bytes,
             before=b['source_inputs']['global_bytes'], after=a.sha(global_bytes))
    _receipt(local_intent_bytes, local_receipt_bytes,
             before=b['source_inputs']['local_bytes'], after=a.sha(local_bytes))
    need(load(global_intent_bytes)['target'] != load(local_intent_bytes)['target'],
         'DISTINCT_CANONICAL_LEDGER_TARGETS')
    activation = dict(epoch=now, GLOBAL_sha256=a.sha(global_bytes), LOCAL_sha256=a.sha(local_bytes),
        GLOBAL_intent_sha256=a.sha(global_intent_bytes), GLOBAL_receipt_sha256=a.sha(global_receipt_bytes),
        LOCAL_intent_sha256=a.sha(local_intent_bytes), LOCAL_receipt_sha256=a.sha(local_receipt_bytes))
    return _initial(grant_bytes, activation)


def _fold_event(r, event, totals):
    """Fold into one PRIVATE replay copy; never copy each growing prefix."""
    kind = event['kind']; amount = event['count']
    need(kind in ('model_loads', 'generation_queries', 'CPU_s'), 'NO_TRAINING_OR_UNREGISTERED_EFFECT')
    amount = a.number(amount, integer=kind != 'CPU_s')
    need(kind == 'CPU_s' or amount == 1, 'ONE_PRE_EFFECT_ATTEMPT')
    q, loads, cpu = totals
    need(event['effect_id'] not in r.get('charged_effects', {}), 'TOKEN_ALREADY_USED_NO_REPLAY')
    if kind == 'model_loads':
        need(loads + amount <= 2, 'MODEL_ATTEMPT_LIMIT')
        loads += amount; r['charged_model_loads'] = loads
    elif kind == 'generation_queries':
        need(loads > 0 and q + amount <= r['query_limit'], 'QUERY_AFTER_MODEL_INSIDE_OWN_ESCROW')
        q += amount; r['charged_query_attempts'] = q; r['reserved_neural_queries'] -= amount
    else:
        need(cpu + amount <= r['CPU_upper_s'], 'CPU_INSIDE_OWN_ESCROW')
        cpu += amount; r['charged_CPU_s'] = cpu
    r.setdefault('charged_effects', {})[event['effect_id']] = dict(kind=kind, count=amount)
    r['status'] = c.STATES[2]
    return q, loads, cpu


def _validate_event(state, event, b):
    need(set(event) == {'effect_id', 'kind', 'count', 'sequence', 'epoch',
         'native_observed_epoch', 'native_facts_sha256', 'previous_state_sha256',
         'previous_event_sha256'}, 'EXACT_EFFECT_RECORD')
    need(type(event['sequence']) is int and event['sequence'] == state['sequence']
         and event['effect_id'] == '{}:{}:{}:DEBIT'.format(
             b['issued_row']['pid'], b['issued_row']['proc_start'], state['sequence'])
         and event['previous_event_sha256'] == state['event_chain_head_sha256']
         and type(event['previous_state_sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', event['previous_state_sha256']),
         'ONCE_ORDERED_TOKEN_AND_EVENT_CHAIN')
    now, seen = a.number(event['epoch']), a.number(event['native_observed_epoch'])
    end = b['accounting_not_after_epoch'] if event['kind'] == 'CPU_s' else b['execution_not_after_epoch']
    need(state['local_observed_epoch'] <= now < end and 0 <= now-seen <= 120
         and seen >= b['issued_row']['process_started_epoch']
         and type(event['native_facts_sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', event['native_facts_sha256']), 'FRESH_LOCAL_PROCESS_AND_UNCHANGED_CLOCK')
    return now


def _event_step(state, event, b):
    now = _validate_event(state, event, b)
    need(event['previous_state_sha256'] == _digest(state), 'EXACT_CURRENT_STATE_BEFORE_DEBIT')
    new = deepcopy(state)
    _fold_event(new['row'], event, c.progress(state['row'])); c.validate_row(new['row'])
    new['events'].append(deepcopy(event)); new['sequence'] += 1; new['local_observed_epoch'] = now
    new['event_chain_head_sha256'] = _digest(event)
    return new


def validate_state(grant_bytes, state_bytes):
    """Linear event-chain/counter verification from one immutable grant.

    Each new debit binds its exact complete before-state hash in consume.
    Durable ``mix_core_delegation_store`` additionally verifies ALL original
    onceIO before/after receipts, including those hashes. Do not substitute
    this pure validator for that native-file reconciliation before an effect
    or terminal merge. Rehashing every complete prefix here would make a
    full 2097-query panel cubic overall without adding receipt evidence.
    """
    b, state = _body(grant_bytes), load(state_bytes)
    need(state.get('grant_sha256') == a.sha(grant_bytes)
         and state.get('schema') == VIEW and state.get('mode') == MODE,
         'EXACT_DELEGATED_VIEW_NOT_A_CANONICAL_GLOBAL')
    need(state.get('unknown_effects') == state.get('pending_effects') == state.get('pending_updates') == [],
         'UNKNOWN_OR_PENDING_EFFECT_MUST_BE_RECONCILED_NO_REPLAY')
    activation = state['activation']
    need(set(activation) == {'epoch', 'GLOBAL_sha256', 'LOCAL_sha256', 'GLOBAL_intent_sha256',
         'GLOBAL_receipt_sha256', 'LOCAL_intent_sha256', 'LOCAL_receipt_sha256'}
         and 0 <= a.number(activation['epoch'])-b['issued_epoch'] <= 120
         and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
                 for k, v in activation.items() if k != 'epoch'), 'ACTIVATION_RECEIPT_BINDING')
    need(type(state.get('events')) is list and len(state['events']) <= b['issued_row']['query_limit']+10,
         'ORIGINAL_BOUNDED_EFFECT_INDEX')
    rebuilt = _initial(grant_bytes, activation)
    totals = c.progress(rebuilt['row'])
    for event in state['events']:
        now = _validate_event(rebuilt, event, b)
        if rebuilt['sequence'] == 0:
            need(event['previous_state_sha256'] == _digest(rebuilt), 'EXACT_INITIAL_STATE_HASH')
        totals = _fold_event(rebuilt['row'], event, totals)
        rebuilt['events'].append(event); rebuilt['sequence'] += 1
        rebuilt['local_observed_epoch'] = now; rebuilt['event_chain_head_sha256'] = _digest(event)
    c.validate_row(rebuilt['row'])
    need(rebuilt == state, 'NO_COUNTER_RESET_MUTABLE_GRANT_OR_FAKE_GLOBAL_FRESHNESS')
    return state


def _live_process(b, row, p):
    """Fresh runtime identity, not a repeated premodel free-memory test.

    The grant already contains original full admission. After a recorded
    model-load debit the SAME neural PID may own CUDA memory. Every other
    compute PID remains subject to the original exact desktop allowance.
    """
    from . import evaluation_core_identity as original, mix_core_desktop as desktop
    r = b['request']; initial = b['admission']['allocation']['native_process']
    fixed = ('schema', 'hostname', 'uid', 'GPU_uuid', 'request_id', 'request_sha256',
             'submission_nonce', 'run_manifest_sha256', 'source_sha', 'process_state',
             'native_handle', 'native_model_permission', 'physics_authorized',
             'started_epoch', 'boot_id', 'pid', 'proc_start')
    need(all(p.get(k) == initial[k] and type(p[k]) is type(initial[k]) for k in fixed),
         'SAME_ACTUAL_KERNEL_PROCESS_NOT_REBOUND_OR_RESTARTED')
    need(p['process_state'] == 'RUNNING', 'ACTUAL_LIVE_BOUND_PROCESS')
    identity.check_clock(r, started_epoch=p['started_epoch'], now_epoch=p['observed_epoch'])
    limits = p['kernel_limits']
    need(original.kernel_limits(str(limits['CPU_quota'])+' '+str(limits['CPU_period']),
         str(limits['memory_max_bytes']), str(limits['pids_max'])) == limits,
         'ORIGINAL_NATIVE_KERNEL_ENVELOPE')
    props = p['native_unit_properties']
    need(props.get('MainPID') == str(row['pid']) and props.get('ActiveState') == 'active'
         and props.get('RuntimeMaxUSec') == initial['native_unit_properties']['RuntimeMaxUSec']
         and props.get('ControlGroup') == initial['native_unit_properties']['ControlGroup'],
         'ACTUAL_SAME_UNIT_AND_CGROUP')
    raw = p['native_GPU_raw']
    need(set(raw) == {'devices', 'processes'} and all(type(v.get('returncode')) is int
         and v['returncode'] == 0 for v in raw.values()), 'COMPLETE_CURRENT_GPU_READS')
    _, loads, _ = c.progress(row); process_raw = raw['processes']['stdout']; receipt = None
    if desktop.FIELD in r:
        proof = p['device']['exact_desktop_coexistence']['kernel_read']
        need(proof['boot_id'] == p['boot_id'] and proof['observed_epoch'] <= p['observed_epoch'],
             'SAME_CURRENT_DESKTOP_KERNEL_OBSERVATION')
        process_raw, receipt = desktop.filter_processes(process_raw, r,
            proof=proof, now_epoch=p['observed_epoch'])
    device = original.parse_GPU_rows(raw['devices']['stdout'], process_raw,
        GPU_uuid=r['GPU_uuid'], own_pid=row['pid'] if loads else None)
    if receipt is not None:
        if not loads:
            need(device['total_bytes']-device['used_bytes'] >= 30*1024**3,
                 'ORIGINAL_DESKTOP_PREMODEL_FREE_MEMORY')
        device['exact_desktop_coexistence'] = receipt
    need(device == p['device'], 'CURRENT_RAW_DEVICE_EQUALS_RECORDED_READ')


def consume(grant_bytes, state_bytes, *, effect_id, kind, count,
            native_facts_bytes, now_epoch, stopped=False):
    """Propose a debit; the caller must confirm its onceIO write before use.

    ``stopped`` is the original owner's combined source/parent/safety STOP
    observation. It never authorizes clearing a STOP or an unknown intent.
    The owner must also retain its current lease, output and physical guards.
    """
    need(type(stopped) is bool and not stopped, 'LOCAL_STOP_OR_SAFETY_FENCE')
    b = _body(grant_bytes); state = validate_state(grant_bytes, state_bytes)
    need(state_bytes == a.raw(state), 'CANONICAL_LOCAL_STATE_BYTES')
    p = load(native_facts_bytes)
    _live_process(b, state['row'], p)
    event = dict(effect_id=effect_id, kind=kind, count=count, sequence=state['sequence'],
        epoch=now_epoch, native_observed_epoch=p['observed_epoch'],
        native_facts_sha256=a.sha(native_facts_bytes), previous_state_sha256=a.sha(state_bytes),
        previous_event_sha256=state['event_chain_head_sha256'])
    return _event_step(state, event, b)


def merge_terminal(grant_bytes, state_bytes, global_bytes, local_bytes, *,
                   native_proof, effects, physical_bytes, delivery_receipt_bytes, now_epoch):
    """Apply only this request's saved debits to fresh canonical bytes, then
    use the ORIGINAL terminal accounting. Caller must independently verify
    all actual local onceIO receipts through the durable store before this
    pure proposal; an event hash is not a native receipt. Other scopes' changes
    remain in the supplied canonical state. There is no remote write here.

    Exact current own-row equality forbids mixed remote/local charging. The
    marker remains permanently, updated to terminal custody; lease closure
    and scientific stage advancement still require their original evidence.
    """
    from scripts.pi0294.generalization import minimal_core_delegation as first512
    if load(grant_bytes).get('schema')==first512.GRANT:
        from scripts.pi0294.generalization.minimal_core_settlement import merge_terminal as merge512
        return merge512(grant_bytes,state_bytes,global_bytes,local_bytes,native_proof=native_proof,effects=effects,
            physical_bytes=physical_bytes,delivery_receipt_bytes=delivery_receipt_bytes,now_epoch=now_epoch)
    b = _body(grant_bytes); state = validate_state(grant_bytes, state_bytes)
    g, l = load(global_bytes), load(local_bytes); c.checked(g); c.paired(g, l)
    rid = b['request']['request_id']; marker = _marker(grant_bytes)
    need(g.get(FIELD, {}).get(rid) == l.get(FIELD, {}).get(rid) == marker,
         'EXACT_STILL_HELD_CANONICAL_DELEGATION_NOT_PREVIOUSLY_MERGED')
    need(g[c.FIELD][rid] == l[c.FIELD][rid] == b['issued_row'],
         'NO_OTHER_WRITER_BIND_DEBIT_SETTLEMENT_OR_ROW_REPLACEMENT')
    need(a.number(now_epoch) >= state['local_observed_epoch'], 'TERMINAL_AFTER_SAVED_EFFECTS')
    need(state['local_observed_epoch'] <= a.number(native_proof.get('finished_epoch')),
         'NO_NEW_DEBIT_AFTER_ACTUAL_NATIVE_TERMINAL')
    v = deepcopy(g)
    # Exactly c.charge's scalar mutations, in original event order (including
    # binary CPU additions), but validate/copy the full campaign only once.
    # validate_state already replayed every per-row limit and unique token.
    # Calling c.charge repeatedly here would copy and validate a growing
    # full campaign for each of 2097 terminal query debits.
    for event in state['events']:
        amount = event['count']
        if event['kind'] == 'generation_queries':
            v['generation_queries'] += amount; v['used']['neural_queries'] += amount
            v['reserved']['neural_queries'] -= amount
        elif event['kind'] == 'model_loads':
            v['model_loads'] += amount
        else:
            v['CPU_s'] += amount; v['CPU_reserved_s'] -= amount
    v[c.FIELD][rid] = deepcopy(state['row']); c.checked(v)
    result = terminal.propose(a.raw(v), local_bytes, request_id=rid,
        native_proof=native_proof, effects=effects, physical_bytes=physical_bytes,
        delivery_receipt_bytes=delivery_receipt_bytes, now_epoch=now_epoch)
    closed = dict(marker, status=MERGED, terminal_native_sha256=c.digest(native_proof),
                  terminal_effect_view_sha256=a.sha(state_bytes), merged_epoch=now_epoch,
                  effects=len(state['events']), actual_global_before_sha256=a.sha(global_bytes),
                  actual_LOCAL_before_sha256=a.sha(local_bytes))
    for value in (result['proposed_global'], result['proposed_LOCAL']):
        value[FIELD][rid] = deepcopy(closed)
    c.checked(result['proposed_global']); c.paired(result['proposed_global'], result['proposed_LOCAL'])
    result.update(original_onceIO_required=True, native_IO_performed=False,
                  leases_closed=False, scientific_stage_advanced=False)
    return result
