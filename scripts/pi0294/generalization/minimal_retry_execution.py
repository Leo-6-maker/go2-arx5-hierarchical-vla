"""Cold, single known-zero first512 replacement in the original GLOBAL.

This is a pure two-row accounting adapter, not an owner or dispatcher. The
original failed row, execution marker and package remain byte-for-byte intact.
The replacement has a distinct version and request, so selecting ``rows[0]``
can never silently select it. Native consumers must explicitly opt into this
adapter and the sealed periodic physical guard before using its proposals.
"""
import base64
import json
import math
import re
from copy import deepcopy
from pathlib import PurePosixPath

from . import minimal_execution as e, minimal_retry_history as history
from . import minimal_retry_paid_io as paid, minimal_retry_runtime_guard as guard
from . import minimal_scope as scope, minimal_terminal as terminal

FIELD = 'ARX5_minimal_first512_known_zero_replacement'
VERSION = 'ARX5_MINIMAL_RAW_PARENT_FIRST512_KNOWN_ZERO_RETRY_V1'
SCHEMA = 'ARX5_MINIMAL_FIRST512_KNOWN_ZERO_REPLACEMENT_INDEX_V1'
REQUEST = 'ARX5_MINIMAL_FIRST512_KNOWN_ZERO_RETRY_REQUEST_V1'
PROFILE = 'vanda-arx5-minimal-raw-parent-first512-known-zero-retry'
RUN = 'arx5-minimal-raw-parent-first512-retry-01'
PHYSICAL_POLICY = 'SEALED_30S_AND_BEFORE_OUTPUT_ORIGINAL_PAID_GROSS_V1'
REPLACEMENT = scope.components()['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT']
# Core metadata remains unused future escrow. All actual control components
# remain charged/projected; subtracting the whole replacement is not a refund.
CONSUMED = dict(REPLACEMENT, Core_storage_bytes=0)
CONTROL = e._vector({k: v for k, v in REPLACEMENT.items()
                     if k not in ('H200_s', 'PBS_submissions', 'Core_storage_bytes')})
TERMINAL_KEYS = {'native_reply_epoch', 'native_exit_status', 'conservative_H200_s',
    'conservative_CPU_s', 'committed_optimizer_updates', 'query_attempts',
    'model_load_attempts', 'terminal_native_sha256', 'terminal_effect_summary_sha256',
    'independent_audit_sha256', 'settled_epoch', 'scientific_endpoint_complete'}
need = e.need


def _number(v):
    need(type(v) in (int, float) and math.isfinite(v) and v >= 0, 'RETRY_FINITE_COUNTER')
    return v


def _digest(v, length=64):
    need(type(v) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', v),
         'RETRY_FULL_IMMUTABLE_IDENTITY')
    return v


def native_resources():
    """Full original compute; output is bounded by the already-paid gross."""
    v = e.native_resources()
    v['Hopper_storage_bytes'] = 0
    return v


def fixed_request_fields():
    v = e.fixed_request_fields()
    v.update(schema=REQUEST, platform_training_version=VERSION, profile=PROFILE,
        prior_readonly_preparation_CPU_upper_s=0, additional_output_upper_bytes=0,
        paid_output_allowance_bytes=e.native_resources()['Hopper_storage_bytes'],
        physical_guard_policy=PHYSICAL_POLICY, run_uuid=RUN,
        original_scientific_execution_version=e.VERSION)
    return v


def request(r, *, central=False):
    fixed = fixed_request_fields()
    need(all(r.get(k) == v and type(r.get(k)) is type(v) for k, v in fixed.items()),
         'RETRY_EXACT_SCIENCE_FULL_COMPUTE_AND_PAID_OUTPUT')
    for key in ('closed_predecessor_row_sha256', 'closed_predecessor_marker_sha256',
                'retry_history_sha256', 'physical_custody_sha256'):
        _digest(r.get(key))
    need(e._uuid(r.get('closed_predecessor_request_id'))
         and r['closed_predecessor_request_id'] != r.get('request_id'),
         'RETRY_DISTINCT_PREDECESSOR_REQUEST')
    _digest(r.get('effective_queue_proof_sha256'))
    # Reuse the original identity, scientific-plan and state checks only.
    # This transient validation view is never written, charged or admitted.
    original_view = deepcopy(r)
    original_view.update({k: e.fixed_request_fields()[k] for k in
        ('schema', 'platform_training_version', 'profile',
         'prior_readonly_preparation_CPU_upper_s', 'additional_output_upper_bytes')})
    e.request(original_view, central=central)
    return r


def select_profile(limits, r):
    request(r)
    v = deepcopy(r)
    v.update({k: e.fixed_request_fields()[k] for k in ('schema', 'platform_training_version',
        'profile', 'prior_readonly_preparation_CPU_upper_s', 'additional_output_upper_bytes')})
    return e.select_profile(limits, v)


def project_envelope(r):
    request(r)
    return dict(GPU_hours=r['reserved_H200_s'] / 3600,
        CPU_hours=max(c for c, _ in e.CAPACITY_PROFILES) * r['reserved_H200_s'] / 3600,
        reserved_CPU_including_prior_preparation_s=r['CPU_upper_s'],
        actual_process_CPU_upper_s=r['CPU_upper_s'], active_CPU_limit=8,
        project_hold_is_not_experiment_CPU_spend=True)


def process_CPU_limits(r):
    request(r)
    return dict(parent_RLIMIT_CPU_s=r['CPU_upper_s'] - 600 - 240,
        isolated_child_CPU_upper_s=600, terminal_observer_CPU_upper_s=240,
        prior_readonly_preparation_CPU_upper_s=0, whole_reserved_CPU_upper_s=r['CPU_upper_s'],
        new_CPU_budget=0)


def _base_remaining(g):
    """Original package bytes stay historical; this exact overlay consumes it."""
    value = deepcopy(g[e.package.FIELD]['remaining_resources'])
    e.subtract(value, CONSUMED)
    return value


def _selected(g):
    m = g[FIELD]
    items = [r for r in g['PBS_submissions'] if r.get('platform_training_version') == VERSION]
    need(len(items) == 1 and items[0]['request_id'] == m['request_id'],
         'RETRY_ONE_NEW_ROW_NOT_ORIGINAL_OR_ANOTHER_RETRY')
    return m, items[0]


def _history_and_request(g):
    """Acyclic identity/provenance check also used by the terminal CPU hook."""
    m, row = _selected(g)
    need(m.get('schema') == SCHEMA and m.get('version') == VERSION
         and m.get('scope_grant_sha256') == e.authority.GRANT_SHA,
         'RETRY_EXPLICIT_INDEX_AND_ORIGINAL_AUTHORITY')
    h = m['history']
    history.verify(g, h)
    # e.validate sees exactly its own original-version row. It never selects
    # this differently-versioned replacement or rewrites the old marker.
    old = e.validate(g)
    need(old == h['original_marker'] and g[e.FIELD] == h['original_marker']
         and g[e.package.FIELD] == m['original_package'],
         'RETRY_OLD_MARKER_PACKAGE_AND_FAILED_ROW_UNCHANGED')
    request(row, central=True)
    rb = base64.b64decode(m['request_base64'], validate=True)
    saved = json.loads(rb)
    request(saved)
    need(e.sha(rb) == row['request_sha256'] == m['request_sha256']
         and all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in saved.items())
         and set(row) - set(saved) <= TERMINAL_KEYS | {
             'request_sha256', 'status', 'job_id', 'submitted_epoch'},
         'RETRY_EXACT_NEW_REQUEST_BYTES_NOT_RESTAMPED_OLD_ROW')
    need(row['closed_predecessor_request_id'] == h['original_row']['request_id']
         and row['closed_predecessor_row_sha256'] == h['original_row_sha256']
         and row['closed_predecessor_marker_sha256'] == h['original_marker_sha256']
         and row['retry_history_sha256'] == e.sha(e.raw(h))
         and row['paid_output_allowance_bytes'] == h['original_paid_output_bytes']
         and row['source_sha'] != h['original_row']['source_sha']
         and row['run_manifest_sha256'] != h['original_row']['run_manifest_sha256']
         and row['submission_nonce'] != h['original_row']['submission_nonce'],
         'RETRY_NEW_COLD_IDENTITY_AND_TRUE_CLOSED_PAID_PREDECESSOR')
    cap = e.authority.validate(g)
    need(all(row[k] == cap[k] for k in ('T_CD', 'deadline_epoch', 'stop_new_training_epoch'))
         and cap['T_CD'] <= _number(m['registered_epoch'])
         and m['registered_epoch'] + row['reserved_H200_s'] < cap['stop_new_training_epoch']
         and row['submitted_epoch'] == m['registered_epoch'],
         'RETRY_UNRESET_WINDOW_AND_COMPLETE_ORIGINAL_MAIN')
    physical = m['physical_custody']
    need(e.sha(e.raw(physical)) == row['physical_custody_sha256']
         and physical.get('schema') == guard.SCHEMA
         and physical.get('request_id') == row['request_id']
         and physical.get('submission_nonce') == row['submission_nonce']
         and physical.get('run_uuid') == RUN
         and physical.get('old_metadata_ceiling') == e.CONTROL['Hopper_storage_bytes']
         and physical.get('paid_output_ceiling') == row['paid_output_allowance_bytes']
         and physical.get('engineering_metadata_allowance') == CONTROL['Hopper_storage_bytes'],
         'RETRY_SEALED_PHYSICAL_GUARD_REQUIRED_WITHOUT_NEW_OUTPUT_CREDIT')
    _digest(physical.get('old_inventory_sha256'))
    need([x.get('run_uuid') for x in physical.get('old_roots', [])] == [
        'arx5-minimal-raw-parent-first512-01', 'arx5-minimal-raw-parent-first512-02'],
        'RETRY_BOTH_RETAINED_CLOSED_ROOTS')
    for old_root in physical['old_roots']:
        _digest(old_root.get('manifest_sha256')); _digest(old_root.get('closed_lease_sha256'))
    need(physical['old_roots'][1]['closed_lease_sha256'] == h['original_closed_lease_sha256'],
         'RETRY_EXACT_FAILED_ROOT_CLOSED_LEASE')
    return m, row


def terminal_CPU_prior(g, row):
    """The original900 is already in the immutable failed row, never paid twice."""
    need(type(g) is dict and FIELD in g, 'RETRY_CPU_ACTUAL_ADOPTED_GLOBAL_CONTEXT_REQUIRED')
    m, selected = _history_and_request(g)
    need(row == selected and row['prior_readonly_preparation_CPU_upper_s'] == 0,
         'RETRY_CPU_ONLY_EXACT_NEW_ROW_CAN_OMIT_PRIOR900')
    old = m['history']['original_marker']['terminal']['native']
    need(old['prior_readonly_preparation_CPU_upper_s'] == e.PREPARATION_CPU_UPPER
         and old['conservative_CPU_s'] == old['new_native_conservative_CPU_s'] + e.PREPARATION_CPU_UPPER,
         'RETRY_CPU_ORIGINAL900_REMAINS_ACTUALLY_SETTLED')
    return 0


def _terminal_fields(native, effects, audit, amounts):
    return dict(native_exit_status=native['Exit_status'], conservative_H200_s=amounts['GPU_s'],
        conservative_CPU_s=amounts['CPU_s'], committed_optimizer_updates=amounts['updates'],
        query_attempts=amounts['queries'], model_load_attempts=amounts['loads'],
        terminal_native_sha256=e.sha(e.raw(native)), terminal_effect_summary_sha256=e.sha(e.raw(effects)),
        independent_audit_sha256=e.sha(e.raw(audit)), scientific_endpoint_complete=amounts['success'])


def _terminal_physical(m, proof, *, observed_epoch, finished_epoch, settled_epoch):
    p = m['physical_custody']; before = m['physical_preflight']
    need(_number(finished_epoch) <= _number(observed_epoch) <= _number(settled_epoch)
         and settled_epoch - observed_epoch <= 120
         and proof.get('status') == 'ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION'
         and proof.get('ledger_changed') is False and proof.get('native_permission') is False
         and proof.get('old_inventory_sha256') == p['old_inventory_sha256']
         and proof.get('old_allocated_bytes') == before['old_allocated_bytes']
         and proof.get('old_roots') == before['old_roots'] and proof.get('new_root') == before['new_root']
         and proof.get('old_metadata_ceiling') == p['old_metadata_ceiling']
         and proof.get('paid_output_ceiling') == p['paid_output_ceiling']
         and _number(proof['next_output_upper_bytes']) >= guard.CONTROL_GROWTH_RESERVE
         and _number(proof['new_allocated_bytes']) + proof['next_output_upper_bytes'] <= p['paid_output_ceiling'],
         'RETRY_ACTUAL_POST_F_PHYSICAL_LIMIT_AND_CLOSURE_MARGIN')


def validate(g):
    if FIELD not in g:
        need(not any(r.get('platform_training_version') == VERSION for r in g['PBS_submissions']),
             'RETRY_ROW_REQUIRES_ITS_EXPLICIT_INDEX')
        return None
    m, row = _history_and_request(g)
    need(m['control_paid_once'] == CONTROL and m['native_hold'] == native_resources()
         and m['consumed_original_replacement'] == CONSUMED
         and m['remaining_resources'] == _base_remaining(g)
         and e.sha(e.raw(m['physical_preflight'])) == m['physical_preflight_sha256']
         and m['original_gross_output_charge_preserved'] is True
         and m['original_onceIO_required'] is True,
         'RETRY_ONLY_ORIGINAL_REPLACEMENT_AND_FULL_NATIVE_HOLD')
    b = m['before']
    need(g['used']['PBS_submissions'] == len(g['PBS_submissions'])
         and g['used']['PBS_submissions'] >= b['PBS_submissions'] + 1
         and g['CPU_s'] >= b['CPU_s'] + CONTROL['Hopper_CPU_s']
         and g['new_storage_current_bytes'] >= b['new_storage_current_bytes'] + CONTROL['Hopper_storage_bytes']
         and g['new_storage_peak_bytes'] >= max(b['new_storage_peak_bytes'], g['new_storage_current_bytes'])
         and all(g[k] >= b[k] for k in ('optimizer_updates', 'generation_queries', 'model_loads'))
         and g['used']['H200_s'] >= b['H200_s'], 'RETRY_ALL_OLD_COSTS_CURRENT_AND_PEAK_RETAINED')
    if row['status'] == e.STATES[2]:
        t = m['terminal']
        need(t.get('schema') == terminal.SCHEMA, 'RETRY_REAL_TERMINAL_REQUIRED')
        amounts = terminal.terminal_values(row, t['native'], t['effects'], t['audit'], retry_global=g)
        fields = _terminal_fields(t['native'], t['effects'], t['audit'], amounts)
        need(all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in fields.items())
             and t['native']['observed_epoch'] <= _number(row['settled_epoch'])
             and m['completed_sessions'] == amounts['complete_sessions']
             and m['status'] == ('COMPLETE' if amounts['success'] else 'INCOMPLETE'),
             'RETRY_TERMINAL_FULL_ORIGINAL_SCIENTIFIC_PROOF')
        _terminal_physical(m, t['physical'], observed_epoch=t['physical_observed_epoch'],
            finished_epoch=t['native']['finished_epoch'], settled_epoch=row['settled_epoch'])
        need(g['used']['H200_s'] >= b['H200_s'] + amounts['GPU_s']
             and g['CPU_s'] >= b['CPU_s'] + CONTROL['Hopper_CPU_s'] + amounts['CPU_s']
             and g['optimizer_updates'] >= b['optimizer_updates'] + amounts['updates']
             and g['generation_queries'] >= b['generation_queries'] + amounts['queries']
             and g['model_loads'] >= b['model_loads'] + amounts['loads'],
             'RETRY_TERMINAL_ALL_EFFECTS_REMAIN_PAID')
    else:
        need('terminal' not in m and m['status'] == 'REGISTERED' and m['completed_sessions'] == [],
             'RETRY_NO_PREPARED_OR_RUNTIME_FLAGS_AS_TERMINAL')
    remaining = _base_remaining(g)
    if row['status'] != e.STATES[2]:
        for k, v in native_resources().items():
            if k != 'PBS_submissions': remaining[k] += v
    old = g[e.original.PACKAGE]['remaining_resources']
    for k in e.package.TRACKED:
        need(_number(g['reserved'][k]) >= old[k] + remaining[k], 'RETRY_BOTH_PACKAGES_HELD:' + k)
    need(_number(g['CPU_reserved_s']) >= old['Hopper_CPU_s'] + remaining['Hopper_CPU_s']
         and _number(g['storage_reserved_bytes']) >= old['Hopper_storage_bytes'] + remaining['Hopper_storage_bytes'],
         'RETRY_BOTH_PACKAGES_CPU_AND_STORAGE_HELD')
    return m


def rows(g):
    """Both real rows, preserved order; consumers must select by request_id."""
    if validate(g) is None: return e.rows(g)
    return [r for r in g['PBS_submissions'] if r.get('platform_training_version') in (e.VERSION, VERSION)]


def selected(g, request_id):
    items = [r for r in rows(g) if r['request_id'] == request_id]
    need(len(items) == 1, 'RETRY_EXACT_REQUEST_SELECTION_NOT_ROWS_ZERO')
    return items[0]


def forecast_remaining(g):
    if validate(g) is None: return e.forecast_remaining(g)
    _, row = _selected(g); v = _base_remaining(g)
    if row['status'] != e.STATES[2]:
        for k, n in native_resources().items():
            if k != 'PBS_submissions': v[k] += n
    return e._vector(v)


def budget_LOCAL(g, local):
    """Conservative paid-control projection, never a fresh canonical write."""
    v = e.budget_LOCAL(g, local)
    m = validate(g)
    if m is None: return v
    key = 'ARX5_minimal_first512_retry_control_projection'
    proof = dict(request_sha256=m['request_sha256'], costs=CONTROL,
                 canonical_LOCAL_write_proven=False, original_once_control_only=True)
    if key in v:
        need(v[key] == proof, 'RETRY_LOCAL_PROJECTION_IDENTITY_UNCHANGED'); return v
    v = deepcopy(v); v['CPU_s']['local_PC'] += CONTROL['local_CPU_s']
    v['network_transfer_accounted_bytes'] += CONTROL['transfer_bytes']
    v[key] = proof
    return v


def register(global_bytes, local_bytes, *, request_bytes, native_readback, closed_readback,
             old_cold_close_readback, root_inventory, physical_custody, physical_proof,
             physical_observed_epoch, now_epoch):
    """Propose one cold retry; actual onceIO/native owner qualification follow.

    Recompute from supplied full originals. The saved candidate is only an
    identity proposal, never a reusable budget PASS or adopted paid-IO credit.
    """
    g = json.loads(global_bytes); r = json.loads(request_bytes); request(r)
    need(FIELD not in g and g['unknown_effects'] == g['pending_updates'] == [],
         'RETRY_ONCE_NO_EXISTING_INDEX_OR_UNKNOWN_CLEAR')
    need(all(x['status'] in e.original.TERMINAL for x in g['PBS_submissions'])
         and all(x['status'] == 'NATIVE_TERMINAL' for x in g.get(e.original.CORE_ROWS, {}).values()),
         'RETRY_COLD_SERIAL_BOUNDARY_NO_UNQUALIFIED_CONCURRENCY')
    new_root = physical_proof['new_root']
    candidate = paid.candidate(global_bytes, local_bytes, native_readback=native_readback,
        closed_readback=closed_readback, old_cold_close_readback=old_cold_close_readback,
        root_inventory=root_inventory, new_request_id=r['request_id'], new_nonce=r['submission_nonce'],
        new_root=new_root, now_epoch=now_epoch)
    h = history.capture(global_bytes, local_bytes, native_readback=native_readback, closed_readback=closed_readback)
    need(r['retry_history_sha256'] == e.sha(e.raw(h))
         and r['physical_custody_sha256'] == e.sha(e.raw(physical_custody)),
         'RETRY_EXACT_CAPTURED_HISTORY_AND_SEALED_PHYSICAL_CONTEXT')
    need(_number(physical_observed_epoch) <= _number(now_epoch)
         and now_epoch - physical_observed_epoch <= 120
         and physical_proof.get('status') == 'ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION'
         and physical_proof.get('ledger_changed') is False and physical_proof.get('native_permission') is False
         and physical_proof['old_inventory_sha256'] == physical_custody['old_inventory_sha256']
         and physical_proof['old_metadata_ceiling'] == e.CONTROL['Hopper_storage_bytes']
         and physical_proof['paid_output_ceiling'] == candidate['old_paid_output_allowance']
         and _number(physical_proof['old_allocated_bytes']) == candidate['old_actual_control_allocated']
         and physical_proof['old_allocated_bytes'] <= e.CONTROL['Hopper_storage_bytes']
         and _number(physical_proof['next_output_upper_bytes']) >= guard.CONTROL_GROWTH_RESERVE
         and _number(physical_proof['new_allocated_bytes']) + physical_proof['next_output_upper_bytes'] <= candidate['old_paid_output_allowance']
         and physical_proof['old_roots'] == [str(PurePosixPath(new_root).parent / name) for name in (
             'arx5-minimal-raw-parent-first512-01', 'arx5-minimal-raw-parent-first512-02')],
         'RETRY_FRESH_ACTUAL_PHYSICAL_CAP_AND_PREWRITE_RESERVE')
    # The former candidate omitted a second control debit in several axes.
    # Add all replacement control components explicitly, retaining Core meta.
    full = deepcopy(candidate['full_dimensions'])
    for k, dim in full.items():
        extra = CONTROL[k] - (CONTROL[k] if k == 'Hopper_storage_bytes' else 0)
        extra += REPLACEMENT[k] - CONSUMED[k]
        dim['projected'] += extra; dim['shortfall'] = max(0, dim['projected'] - dim['cap'])
    need(not any(d['shortfall'] for d in full.values()), 'RETRY_COMPLETE_FUTURE_WITH_PAID_CONTROL_HAS_NO_GAP')
    v = deepcopy(g)
    row = dict(r, request_sha256=e.sha(request_bytes), status=e.STATES[0], job_id=None, submitted_epoch=now_epoch)
    v['PBS_submissions'].append(row); v['used']['PBS_submissions'] += 1
    held = native_resources()
    for k in e.package.TRACKED: v['reserved'][k] += held[k] - REPLACEMENT[k]
    v['CPU_reserved_s'] += held['Hopper_CPU_s'] - REPLACEMENT['Hopper_CPU_s']
    v['storage_reserved_bytes'] += held['Hopper_storage_bytes'] - REPLACEMENT['Hopper_storage_bytes']
    v['CPU_s'] += CONTROL['Hopper_CPU_s']
    v['new_storage_current_bytes'] += CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes'] = max(v['new_storage_peak_bytes'], v['new_storage_current_bytes'])
    before = {k: g[k] for k in ('CPU_s', 'new_storage_current_bytes', 'new_storage_peak_bytes',
                               'optimizer_updates', 'generation_queries', 'model_loads')}
    before.update(H200_s=g['used']['H200_s'], PBS_submissions=g['used']['PBS_submissions'])
    v[FIELD] = dict(schema=SCHEMA, version=VERSION, scope_grant_sha256=e.authority.GRANT_SHA,
        request_id=r['request_id'], request_sha256=e.sha(request_bytes), request_base64=base64.b64encode(request_bytes).decode(),
        history=h, original_package=deepcopy(g[e.package.FIELD]), physical_custody=deepcopy(physical_custody),
        physical_preflight=deepcopy(physical_proof), physical_preflight_sha256=e.sha(e.raw(physical_proof)),
        before_GLOBAL_sha256=e.sha(global_bytes),
        LOCAL_input_sha256=e.sha(local_bytes), complete_future_dimensions=full,
        control_paid_once=deepcopy(CONTROL), native_hold=held, consumed_original_replacement=deepcopy(CONSUMED),
        remaining_resources=_base_remaining(g), before=before, registered_epoch=now_epoch,
        original_gross_output_charge_preserved=True, original_onceIO_required=True,
        status='REGISTERED', completed_sessions=[])
    v['revision'] += 1; validate(v)
    return dict(proposed_global=v, complete_future_dimensions=full, original_onceIO_required=True,
        original_global_reader_hooks_required=True, native_admission=False, new_PBS_intents=1,
        new_model_queries_updates_physics=0)


def bind(g, request_id, job_id, *, now_epoch):
    validate(g); _, row = _selected(g)
    need(row['request_id'] == request_id and row['status'] == e.STATES[0] and row['job_id'] is None,
         'RETRY_EXACT_UNBOUND_NEW_ROW_ONCE')
    need(g['unknown_effects'] == g['pending_updates'] == [], 'RETRY_BIND_NEVER_CLEAR_GLOBAL_LATCHES')
    need(_number(now_epoch) >= row['submitted_epoch'] and now_epoch < row['stop_new_training_epoch'],
         'RETRY_REAL_BIND_INSIDE_ORIGINAL_WINDOW')
    need(type(job_id) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02', job_id)
         and all(x.get('job_id') != job_id for x in g['PBS_submissions']), 'RETRY_NEW_UNIQUE_NATIVE_PBS_HANDLE')
    v = deepcopy(g); _, current = _selected(v)
    current.update(status=e.STATES[1], job_id=job_id, native_reply_epoch=now_epoch)
    v['revision'] += 1; validate(v); return v


def settle(g, request_id, *, native, effects, audit, physical_proof, physical_observed_epoch, now_epoch):
    """Once actual costs; no second output charge or return of old paid costs."""
    validate(g); mark, row = _selected(g)
    need(row['request_id'] == request_id and row['status'] == e.STATES[1]
         and g['unknown_effects'] == g['pending_updates'] == []
         and _number(now_epoch) >= native['observed_epoch'], 'RETRY_EXACT_KNOWN_TERMINAL_ONCE')
    amounts = terminal.terminal_values(row, native, effects, audit, retry_global=g)
    _terminal_physical(mark, physical_proof, observed_epoch=physical_observed_epoch,
        finished_epoch=native['finished_epoch'], settled_epoch=now_epoch)
    v = deepcopy(g); m, current = _selected(v); held = native_resources()
    for k in e.package.TRACKED: v['reserved'][k] -= held[k]
    v['CPU_reserved_s'] -= held['Hopper_CPU_s']
    v['storage_reserved_bytes'] -= held['Hopper_storage_bytes']
    v['used']['H200_s'] += amounts['GPU_s']; v['CPU_s'] += amounts['CPU_s']
    v['optimizer_updates'] += amounts['updates']; v['used']['optimizer_updates'] = v['optimizer_updates']
    v['generation_queries'] += amounts['queries']; v['used']['neural_queries'] = v['generation_queries']
    v['model_loads'] += amounts['loads']
    current.update(status=e.STATES[2], settled_epoch=now_epoch, **_terminal_fields(native, effects, audit, amounts))
    m.update(status='COMPLETE' if amounts['success'] else 'INCOMPLETE', completed_sessions=amounts['complete_sessions'],
        terminal=dict(schema=terminal.SCHEMA, native=deepcopy(native), effects=deepcopy(effects), audit=deepcopy(audit),
                      physical=deepcopy(physical_proof), physical_observed_epoch=physical_observed_epoch))
    v['revision'] += 1; validate(v); return v
