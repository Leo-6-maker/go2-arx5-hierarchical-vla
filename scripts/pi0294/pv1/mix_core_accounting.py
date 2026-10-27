"""C/D Core stage proposals within the already reserved successor package.

No IO or native admission. The existing owner/Journal/onceIO must persist both
proposals and independently verify the actual process before model effects.
Old A/B allocations, clocks, sharing exceptions and paid history are inputs.
"""
from copy import deepcopy
import math
import re

from . import mix_accounting as a, successor_cd as s, mix_stage_handoff as handoff
from .evaluation_core_accounting import canonical_uuid, process_handle, digest
from .evaluation_accounting import progress
from .successor_core_context import EVAL_ROLE

VERSION = a.CORE_VERSION
ROLE = 'V5_CD_CORE_NEURAL_EVALUATION_V1'
PROFILE = '4080-neural-evaluation'
FIELD = a.CORE_ROWS
STATES = ('PRESTART_INTENT_UNKNOWN_NO_RETRY', 'BOUND_QUALIFICATION',
          'NATIVE_RUNNING', 'NATIVE_TERMINAL')
CPU_UPPER = 80000
GPU_UPPER = 28860
CONTROL = dict(Hopper_CPU_s=7200, local_CPU_s=7200,
               Hopper_storage_bytes=64*1024**2, Core_storage_bytes=128*1024**2,
               transfer_bytes=1024**3)
CANDIDATES = 'V5_CD_development_candidates'


def checked(value):
    from .neural_ledger import check
    return check(value, allow_unknown=True)


def need(ok, message):
    if not ok:
        raise PermissionError('V5_CD_CORE_ACCOUNTING_' + message)


def key(arm):
    need(arm in ('C50', 'D25'), 'ONLY_TWO_FIXED_ARMS')
    return 'EVAL:' + arm + ':N294101'


def before_stage(arm, g=None):
    order=s.execution_order(g) if g is not None else s.ORDER
    return list(order[:order.index(key(arm))])


def remaining_stage():
    value = a.stage_vector('EVAL',Core_GPU_projection=True)
    for name, amount in CONTROL.items():
        value[name] -= amount
    return value


def validate_row(r):
    arm = r.get('evaluation_candidate')
    key(arm)
    from . import mix_core_desktop as desktop
    desktop.allowed(r)
    from .evaluation_scope import CAMPAIGN
    expected = dict(schema='V5_CD_CORE_NATIVE_GPU_ALLOCATION_V1', campaign_run_uuid=CAMPAIGN,
        native_platform='CORELAB_RTX4080', platform_evaluation_version=VERSION, owner_user='liuyu-nfs',
        authority_release=s.TASK_SOURCE, scope_grant_sha256=s.GRANT_SHA,
        stage='V5_MATCHED_DEV24', noise_bank=294101, remaining_conditions=24,
        checkpoint_step=1024, statistics_sha256=a.STATS,
        parent_checkpoint_manifest_sha256=a.PARENT, owner_uid=2009,
        hostname='corelab-4080s', physical_GPU_count=1,
        physical_GPU_kind='RTX_4080_SUPER', walltime_s=28800,
        accounting_pad_s=60, reserved_GPU_s=GPU_UPPER, CPU_upper_s=CPU_UPPER,
        declared_process_CPU_limit_s=79760, active_cpu_limit=4,
        minimum_GPU_memory_bytes=30*1024**3, requested_memory_bytes=48*1024**3,
        optimizer_update_limit=0, reserved_optimizer_updates=0,
        PBS_submissions_consumed=0, query_limit=2097, charges_DEC1_subcaps=False,
        new_robot_physics=0, legacy_GPU_seconds_field_is_projection=True,
        additional_output_upper_bytes=remaining_stage()['Hopper_storage_bytes'])
    need(all(r.get(n) == v and type(r[n]) is type(v) for n, v in expected.items()),
         'EXACT_NEW_SCOPE_ZERO_TRAINING_ENVELOPE')
    need(not any(n in r for n in ('project', 'submit_count', 'PBS_JOBID', 'job_name',
        'platform_training_version', 'training_block', 'shared_GPU_grant_sha256',
        'queue_dispatch_mode', 'startup_recovery_sha256')), 'NO_OLD_PBS_OR_B24_EXCEPTION')
    need(canonical_uuid(r.get('request_id')) and canonical_uuid(r.get('submission_nonce')),
         'FRESH_CANONICAL_REQUEST_AND_NONCE')
    for name, length in (('source_sha', 40), ('request_sha256', 64),
            ('run_manifest_sha256', 64), ('checkpoint_manifest_sha256', 64),
            ('training_handoff_sha256', 64), ('planned_panel_sha256', 64),
            ('cold_input_audit_sha256', 64)):
        need(type(r.get(name)) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', r[name]),
             'FULL_IMMUTABLE_PROVENANCE:' + name)
    need(r.get('status') in STATES and type(r.get('created_epoch')) in (int, float)
         and math.isfinite(r['created_epoch']) and type(r.get('T_CD')) in (int, float)
         and type(r.get('deadline_epoch')) in (int, float)
         and r['T_CD'] <= r['created_epoch'] < r['deadline_epoch'] - 28800 - 60,
         'ACTUAL_CLOCK_AND_COMPLETE_PANEL_WINDOW')
    need(type(r.get('child_run_uuid')) is str and re.fullmatch(
         r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', r['child_run_uuid'])
         and type(r.get('GPU_uuid')) is str and re.fullmatch(
         r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', r['GPU_uuid']),
         'DISTINCT_REGISTERED_CHILD_AND_REAL_DEVICE')
    if r['status'] == STATES[0]:
        need(r.get('native_handle') is None and not any(n in r for n in
             ('boot_id', 'pid', 'proc_start', 'process_started_epoch')), 'UNKNOWN_IS_NOT_RUNNING')
    else:
        need(r.get('native_handle') == process_handle(r.get('boot_id'), r.get('pid'), r.get('proc_start'))
             and r['created_epoch'] <= r['process_started_epoch'] < r['deadline_epoch'] - 28860,
             'REAL_BOUND_BOOT_PID_AND_START')
    q, loads, cpu = progress(r)
    need(type(r.get('reserved_neural_queries')) is int and
         r['reserved_neural_queries'] == (len(r.get('unknown_query_intents',[])) if r['status']==STATES[3] else r['query_limit']-q),
         'EVERY_QUERY_CHARGED_OR_HELD')
    return r


def validate_rows(g):
    """Called by the existing global validator; never recursively checks it."""
    if FIELD not in g:
        return g
    marker = s.validated(g)
    rows = g[FIELD]
    from . import mix_core_premodel_repair as repair
    repaired=repair.validated(g)
    need(type(rows) is dict and 1 <= len(rows) <= (3 if repaired else 2), 'ONLY_ORIGINAL_PAIRED_PANELS')
    arms = []
    for request_id, r in rows.items():
        validate_row(r)
        need(request_id == r['request_id'] and r['T_CD'] == marker['T_CD']
             and r['deadline_epoch'] == marker['deadline_epoch'], 'ONE_ADOPTED_CLOCK')
        repair.output_credit(g,r)
        if not repaired or request_id!=repair.FAILED:arms.append(r['evaluation_candidate'])
        endpoint = g[a.PACKAGE]['endpoints'][r['evaluation_candidate']]
        need(endpoint['checkpoint_manifest_sha256'] == r['checkpoint_manifest_sha256'],
             'ACTUALLY_CLOSED_1024_ENDPOINT_NOT_OLD_AB')
    need(len(arms) == len(set(arms)), 'ONE_PANEL_EACH_ARM')
    for name in ('request_id', 'submission_nonce'):
        identities = [r.get(name) for r in [*rows.values(), *g['PBS_submissions'],
            *g.get('Core_native_GPU_allocations', {}).values()] if r.get(name) is not None]
        need(len(identities) == len(set(identities)), 'NO_CROSS_SCOPE_IDENTITY_REUSE:' + name)
    active = [r for r in rows.values() if r['status'] != STATES[3]]
    p = a.package_view(g)['remaining_resources']
    need(g['reserved']['H200_s'] >= p['H200_s'] + sum(r['reserved_GPU_s'] for r in active)
         and g['reserved']['neural_queries'] >= p['neural_queries'] + sum(r['reserved_neural_queries'] for r in active)
         and g['CPU_reserved_s'] >= p['Hopper_CPU_s'] + sum(CPU_UPPER-progress(r)[2] for r in active)
         and g['storage_reserved_bytes'] >= p['Hopper_storage_bytes'] + sum(r['additional_output_upper_bytes']-repair.output_credit(g,r) for r in active),
         'ACTIVE_AND_ALL_FUTURE_ESCROWS_RETAINED')
    return g


def paired(g, l):
    from .mix_vanda_first import pending
    need(not pending(g), 'ACTUAL_LOCAL_RECONCILIATION_REQUIRED_BEFORE_CORE_ADMISSION')
    a.validate_global(g)
    from . import mix_core_recovery as recovered
    recovered.paired(g,l)
    need(s.validated(l) == s.validated(g) and l[a.PACKAGE] == g[a.PACKAGE],
         'EXACT_REAL_BOTH_LEDGER_PACKAGE_NOT_STALE_CACHED_LOCAL')
    need(g.get('unknown_effects') == g.get('pending_updates') == []
         and l['unknown_effects'] == l['pending_updates'] == l['pending_queries'] == [],
         'NO_UNKNOWN_EFFECT_OR_PENDING_UPDATE')
    need(l.get(FIELD, {}) == g.get(FIELD, {}), 'SAME_ACTUAL_CORE_STAGE_INDEX')


def closed_training(g, l, terminal_capture, arm):
    """Require all real terminal bytes again, even after LOCAL paid projection."""
    paired(g, l)
    proof = handoff.closed_endpoint(global_bytes=a.raw(g), arm=arm, **terminal_capture)
    mirror = l.get('V5_CD_terminal_training_mirrors', {}).get(proof['request_id'])
    need(type(mirror) is dict and mirror['handoff']['settled_sha256'] == proof['settled_sha256']
         and mirror['handoff']['closed_lease_sha256'] == proof['closed_lease_sha256']
         and mirror['handoff']['OS_exit_sha256'] == proof['OS_exit_sha256'],
         'TRAINING_COST_ALREADY_MIRRORED_ONCE_TO_CANONICAL_LOCAL')
    need(a.package_view(g)['completed_stages'] == before_stage(arm, g), 'NO_SKIP_OR_D_INHERITANCE')
    return proof


def prepay_control(g, l, *, terminal_capture, arm, now_epoch):
    proof = closed_training(g, l, terminal_capture, arm)
    from .mix_core_cpu_projection import validate as actual_Core_CPU
    actual_Core_CPU(g,l)
    marker = s.validated(g)
    need(marker['T_CD'] <= a.number(now_epoch) < marker['deadline_epoch'] - 28860,
         'COMPLETE_NATIVE_PANEL_AND_CLOSEOUT_WINDOW')
    stage = key(arm)
    need(stage not in g[a.PACKAGE].get('prepaid_control', {}), 'CONTROL_ONCE_NOT_TWO_NEW_BUDGETS')
    v, w = deepcopy(g), deepcopy(l)
    for value in (v, w):
        p = value[a.PACKAGE]
        for n, cost in CONTROL.items():
            need(p['remaining_resources'][n] >= cost, 'EXISTING_CONTROL_ESCROW:' + n)
            p['remaining_resources'][n] -= cost
        p.setdefault('prepaid_control', {})[stage] = dict(costs=deepcopy(CONTROL), epoch=now_epoch,
            financial_upper_not_actual_process_CPU=True, closed_training_handoff_sha256=a.sha(a.raw(proof)))
    v['CPU_reserved_s'] -= 7200; v['CPU_s'] += 7200
    v['storage_reserved_bytes'] -= CONTROL['Hopper_storage_bytes']
    v['new_storage_current_bytes'] += CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes'] = max(v['new_storage_peak_bytes'], v['new_storage_current_bytes'])
    for site in ('Hopper', 'local_PC'):
        w['CPU_reserved_s'][site] -= 7200; w['CPU_s'][site] += 7200
    w['network_transfer_accounted_bytes'] += CONTROL['transfer_bytes']
    w['Core_native_GPU_current_storage_bytes'] = l.get('Core_native_GPU_current_storage_bytes', 0) + CONTROL['Core_storage_bytes']
    paired(v, w)
    return dict(proposed_global=v, proposed_LOCAL=w, original_onceIO_required=True,
                native_IO_performed=False, model_or_physics_admission=False)


def register(g, l, record, physical_manifest_bytes, *, terminal_capture, now_epoch):
    """Move an escrow slice, never reserve or charge the complete package twice."""
    arm = record.get('evaluation_candidate'); proof = closed_training(g, l, terminal_capture, arm)
    r = deepcopy(validate_row(record)); p = a.package_view(g); stage = key(arm)
    need(stage in p.get('prepaid_control', {}) and r['status'] == STATES[0], 'REAL_PAID_CONTROL_BEFORE_UNIQUE_INTENT')
    need(r['request_id'] not in g.get(FIELD, {}) and
         not any(x['status'] not in a.TERMINAL for x in g['PBS_submissions']) and
         not any(x['status'] != STATES[3] for x in g.get(FIELD, {}).values()) and
         not any(x['status'] != 'NATIVE_TERMINAL' for x in g.get('Core_native_GPU_allocations', {}).values()),
         'ONE_TASK_GPU_AND_NO_UNKNOWN_RETRY')
    need(r['T_CD'] == s.validated(g)['T_CD'] and r['deadline_epoch'] == s.validated(g)['deadline_epoch']
         and 0 <= a.number(now_epoch)-r['created_epoch'] <= 120
         and now_epoch < r['deadline_epoch']-28860
         and r['checkpoint_manifest_sha256'] == proof['checkpoint_manifest_sha256']
         and r['training_handoff_sha256'] == a.sha(a.raw(proof)), 'CURRENT_CLOSED_TRAINING_EXACT_REQUEST')
    from .successor_core_context import SCHEMA
    m = handoff.load(physical_manifest_bytes)
    need(m.get('schema') == SCHEMA and m.get('scope') == EVAL_ROLE
         and m.get('candidate') == arm and m['run_uuid'] == r['child_run_uuid']
         and m['code']['vla_commit'] == r['source_sha']
         and m['authority']['protocol_ref'] == s.TASK_SOURCE
         and m['authority']['scope_grant_sha256'] == s.GRANT_SHA
         and m['execution']['profile'] == '4080-simulation'
         and m['limits']['operation_deadline_epoch'] <= r['deadline_epoch'], 'EXPLICIT_PHYSICAL_PEER_NOT_OLD_ROLE')
    expected = dict(CPU_s=93900, physics_ticks=832000, policy_episodes=26, engine_starts=2108)
    need(all(type(m['limits'].get(n)) is int and m['limits'][n] == cost for n, cost in expected.items()),
         'ORIGINAL_FULL_PHYSICAL_UPPER_NO_NEW_SCIENCE')
    from . import mix_core_premodel_repair as repair
    repair.output_credit(g,r)
    v, w = deepcopy(g), deepcopy(l); allocation = repair.allocation(g,arm)
    for value in (v, w):
        for n, cost in allocation.items():
            need(value[a.PACKAGE]['remaining_resources'][n] >= cost, 'STAGE_INSIDE_ORIGINAL_POOL:' + n)
            value[a.PACKAGE]['remaining_resources'][n] -= cost
        value.setdefault(FIELD, {})[r['request_id']] = deepcopy(r)
    if stage in w.get('V5_CD_Core_CPU_substages', {}):
        repair_marker=repair.validated(g)
        from . import mix_core_recovery as recovered
        repair_marker=recovered.validate(g)or repair_marker
        need(repair_marker is not None and r['request_id']==repair_marker['replacement_request_id']
             and w['V5_CD_Core_CPU_substages'][stage]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL', 'EXACT_CLOSED_PREMODEL_PHYSICAL_PREDECESSOR')
        w.setdefault('V5_CD_historical_closed_Core_CPU_substages',{})[repair.FAILED]=deepcopy(w['V5_CD_Core_CPU_substages'][stage])
    w.setdefault('V5_CD_Core_CPU_substages', {})[stage] = dict(expected,
        status='RESERVED_NOT_SPENT', run_uuid=m['run_uuid'], role=EVAL_ROLE,
        source_sha=r['source_sha'], scope_grant_sha256=s.GRANT_SHA,
        run_manifest_sha256=a.sha(physical_manifest_bytes), request_id=r['request_id'])
    held = w.setdefault('Core_native_GPU_reserved', {})
    for n, cost in dict(episode_starts=26, physics_ticks=832000, constructors=2108,
            CoreLab_output_bytes=allocation['Core_storage_bytes'], transfer_bytes=allocation['transfer_bytes']).items():
        held[n] = held.get(n, 0) + cost
    paired(v, w)
    return dict(proposed_global=v, proposed_LOCAL=w, original_onceIO_required=True,
                native_IO_performed=False, native_process_or_model_admission=False)


def bind(g, l, *, request_id, process, now_epoch):
    paired(g, l); r = g[FIELD][request_id]
    need(r['status'] == STATES[0] and r['native_handle'] is None, 'ONE_PROCESS_BIND_NEVER_RESTART_UNKNOWN')
    expected = dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1', hostname='corelab-4080s', uid=2009,
        GPU_uuid=r['GPU_uuid'], request_id=request_id, request_sha256=r['request_sha256'],
        submission_nonce=r['submission_nonce'], run_manifest_sha256=r['run_manifest_sha256'],
        source_sha=r['source_sha'], process_state='RUNNING', native_model_permission=False, physics_authorized=False)
    need(all(process.get(n) == val and type(process[n]) is type(val) for n, val in expected.items())
         and r['created_epoch'] <= process['started_epoch'] <= process['observed_epoch'] <= now_epoch
         and now_epoch-process['observed_epoch'] <= 120
         and now_epoch < r['deadline_epoch']-28860, 'FRESH_ACTUAL_PROCESS_BEFORE_MODEL')
    handle = process_handle(process.get('boot_id'), process.get('pid'), process.get('proc_start'))
    need(process.get('native_handle') == handle, 'NO_PREPARED_OR_PBS_HANDLE')
    v, w = deepcopy(g), deepcopy(l)
    for value in (v, w):
        value[FIELD][request_id].update(status=STATES[1], native_handle=handle,
            boot_id=process['boot_id'], pid=process['pid'], proc_start=process['proc_start'],
            process_started_epoch=process['started_epoch'], native_bind_proof_sha256=digest(process))
    paired(v, w)
    return dict(proposed_global=v, proposed_LOCAL=w, original_onceIO_required=True,
                model_or_physics_admission=False)


def charge(g, *, request_id, native_handle, kind, count, effect_id):
    """Original pre-effect neural debit; LOCAL receives the exact terminal delta."""
    a.validate_global(g); v = deepcopy(g); r = v[FIELD][request_id]
    need(r['status'] in STATES[1:3] and r['native_handle'] == native_handle, 'SAME_LIVE_BOUND_NATIVE_PROCESS')
    need(not v['unknown_effects'] and not v['pending_updates'], 'UNKNOWN_FENCE_NO_NEW_EFFECT')
    need(kind in ('generation_queries', 'model_loads', 'CPU_s') and type(effect_id) is str
         and re.fullmatch(r'[A-Za-z0-9._:/-]{1,200}', effect_id)
         and effect_id not in r.get('charged_effects', {}), 'UNIQUE_ONCE_PRE_EFFECT_DEBIT')
    amount = a.number(count, integer=kind != 'CPU_s'); q, loads, cpu = progress(r)
    need(kind == 'CPU_s' or amount == 1, 'ONE_MODEL_OR_QUERY_ATTEMPT')
    if kind == 'generation_queries':
        need(loads > 0 and q + amount <= r['query_limit'], 'QUERY_AFTER_MODEL_INSIDE_ESCROW')
        r['charged_query_attempts'] = q + amount; r['reserved_neural_queries'] -= amount
        v['generation_queries'] += amount; v['used']['neural_queries'] += amount; v['reserved']['neural_queries'] -= amount
    elif kind == 'model_loads':
        need(loads + amount <= 2, 'BOUNDED_MODEL_ATTEMPTS'); r['charged_model_loads'] = loads + amount; v['model_loads'] += amount
    else:
        need(cpu + amount <= CPU_UPPER, 'ACTUAL_NEURAL_CPU_INSIDE_ESCROW')
        r['charged_CPU_s'] = cpu + amount; v['CPU_s'] += amount; v['CPU_reserved_s'] -= amount
    r.setdefault('charged_effects', {})[effect_id] = dict(kind=kind, count=amount)
    r['status'] = STATES[2]; a.validate_global(v)
    return v
