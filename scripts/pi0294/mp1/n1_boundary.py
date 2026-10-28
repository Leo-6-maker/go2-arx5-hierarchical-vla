"""One contiguous N1 cold-boundary format, with the full prior chain retained."""
from pathlib import Path
import re

from scripts.pi0292.durable import read
from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.native_dataset import sha
from . import n1_limits as n1

SCHEMA = 'MP1_N1_SAME_ROOT_COLD_BOUNDARY_V1'
TRANSPORT_PLAN = 'plans/N1_TRANSPORT_FIXTURE.json'
NAMES = ['COLD_BOUNDARY.json', 'PRIOR_RUN_MANIFEST.json', 'PRIOR_HOPPER_LEDGER.json',
         'PRIOR_ADJUDICATIONS.json', 'NATIVE_TERMINALS.json', 'EFFECT_CLOSURE.json',
         'PROCESS_CLOSURE.json']


def paths(number):
    if type(number) is not int or not 6 <= number <= 12:
        raise PermissionError('N1_FUTURE_CONTIGUOUS_COLD_ALLOCATION_06_TO_12')
    return ['plans/N1/alloc-' + str(number).zfill(2) + '/' + name for name in NAMES]


def target(relative):
    match = re.fullmatch(r'plans/N1/alloc-([0-9]{2})/COLD_BOUNDARY\.json', relative or '')
    if match is None:
        return None
    number = int(match[1])
    paths(number)
    return number


def native_history(past, native, number, cap):
    rows = past['PBS_submissions']
    if (past['run_uuid'] != 'source-pair-01' or len(rows) != number - 1
            or past['GPU_reserved_s'] or past['pending_updates'] or past.get('CPU_reserved_s',0)
            or any(type(past[k]) is not int for k in ['GPU_allocated_s','optimizer_updates','generation_queries'])
            or not 2560 <= past['optimizer_updates'] <= 4096
            or not 0 <= past['generation_queries'] <= cap['generation_queries']):
        raise PermissionError('N1_ALL_PRIOR_ALLOCATIONS_SETTLED_NO_UNKNOWN_UPDATE')
    if len(native) != len(rows):
        raise PermissionError('N1_EVERY_NATIVE_ATTEMPT_MUST_HAVE_A_RECEIPT')
    jobs = set()
    total = 0
    for index, (row, proof) in enumerate(zip(rows, native), 1):
        if row['request_id'] != 'alloc-' + str(index).zfill(2):
            raise PermissionError('N1_CONTINUOUS_PRIOR_ALLOCATION_INTENTS')
        if row['status'] == 'NATIVE_QSUB_REJECTED_NO_JOB':
            if row.get('job_id') is not None or proof.get('job_id') is not None or proof.get('qsub_returncode', 0) == 0:
                raise PermissionError('N1_REJECTED_QSUB_NOT_UNKNOWN_OR_A_JOB')
            continue
        job = row.get('job_id')
        if not job or job in jobs or proof['job_id'] != job:
            raise PermissionError('N1_UNIQUE_PRIOR_NATIVE_JOB_IDENTITIES')
        jobs.add(job)
        value = proof['native']
        if (row['status'] != 'NATIVE_TERMINAL' or value['job_state'] != 'F'
                or value['project'] != 'CFP04-CF-007'
                or value['Job_Owner'].split('@')[0] != 'e1192548'):
            raise PermissionError('N1_ONLY_OWN_PROJECT_NATIVE_TERMINALS')
        wall = sum(int(v) * k for v, k in zip(value.get('resources_used', {}).get('walltime', '00:00:00').split(':'), (3600, 60, 1)))
        if value.get('stime'):
            if not value.get('obittime'):
                raise PermissionError('N1_TERMINAL_ALLOCATION_END_REQUIRED')
            upper = max(wall, pbs_epoch(value['obittime']) - pbs_epoch(value['stime']))
        else:
            if value.get('exec_host') or value.get('run_count', 0) or wall:
                raise PermissionError('N1_UNOBSERVED_GPU_COST_NOT_ZERO')
            upper = 0
        if row['GPU_conservative_charge_s'] != upper:
            raise PermissionError('N1_NATIVE_PRIOR_COST_CANNOT_RESET_OR_DOUBLE_COUNT')
        total += upper
    if past['GPU_allocated_s'] != total or total > cap['GPU_s']:
        raise PermissionError('N1_CUMULATIVE_NATIVE_SOURCE_BUDGET')


def preserved_stops(root,number,stops,descendant_hashes):
    """An outer epoch owns newer STOPs; every older sealed hash still applies."""
    from . import cold_boundary as prior
    from . import cold_boundary05 as fifth
    root=Path(root);descendant_hashes={} if descendant_hashes is None else descendant_hashes
    for name,digest in descendant_hashes.items():
        match=re.fullmatch(r'RELAY_ENGINEERING_STOP_ALLOCATION([0-9]{2})\.json',name)
        if (match is None or not number<int(match[1])<=12 or (root/name).is_symlink()
                or sha(root/name)!=digest):
            raise PermissionError('N1_EXACT_OUTER_EPOCH_STOP_CONTEXT')
    current='RELAY_ENGINEERING_STOP_ALLOCATION'+str(number).zfill(2)+'.json'
    actual={p.name for p in root.glob('RELAY_ENGINEERING_STOP*.json') if p.name!=current and p.name not in descendant_hashes}
    if (stops.get('RELAY_ENGINEERING_STOP.json')!=prior.STOP_SHA
            or stops.get('RELAY_ENGINEERING_STOP_ALLOCATION04.json')!=fifth.STOP04_SHA
            or any(not re.fullmatch(r'RELAY_ENGINEERING_STOP(?:_ALLOCATION[0-9]{2})?\.json',n) for n in stops)
            or actual!=set(stops)
            or any((root/name).is_symlink() or sha(root/name)!=digest for name,digest in stops.items())):
        raise PermissionError('N1_ORIGINAL_ENGINEERING_STOPS_MUST_REMAIN_BYTE_IDENTICAL')
    result=dict(descendant_hashes)
    if (root/current).exists():
        if (root/current).is_symlink():raise PermissionError('N1_REAL_CURRENT_EPOCH_STOP')
        result[current]=sha(root/current)
    return result


def validate(root, worker_manifest=None, *, proposed_manifest=None,_descendant_stop_hashes=None):
    from . import cold_boundary as prior
    from . import cold_boundary05 as fifth
    from .plans import remaining_panel
    root = Path(root)
    m = read(root / 'RUN_MANIFEST.json') if proposed_manifest is None else proposed_manifest
    number = target(m['execution'].get('cold_engineering_binding_relative'))
    if number is None:
        raise PermissionError('N1_EXPLICIT_COLD_BOUNDARY_REQUIRED')
    effective = n1.resolve(root, m)
    if effective is None:
        raise PermissionError('N1_COMPLETE_AUTHORIZATION_LAYER_REQUIRED')
    names = paths(number)
    for name in names + [TRANSPORT_PLAN]:
        p = root / name
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_mode & 0o222 or m['execution']['sealed_inputs'].get(name) != sha(p)):
            raise PermissionError('N1_SEALED_PRIOR_COLD_BOUNDARY_EVIDENCE')
    q, old, past, rows, native, effects, processes = [read(root / name) for name in names]
    if (q['schema'] != SCHEMA or q['authority_release'] != n1.RELEASE
            or q['parent_authority_release'] != prior.AUTH or q['run_uuid'] != root.name
            or q['parent_run_uuid'] != root.parent.parent.name
            or q['original_T0_epoch'] != n1.ORIGINAL_T0
            or q['target_allocation_number'] != number or q['effective_source_PBS_cap'] != 12
            or q['new_source_sha'] != m['code']['vla_commit']
            or q['prior_manifest_sha256'] != sha(root / names[1])
            or q['adoption_delta_sha256'] != sha(root / n1.DELTA)
            or q['new_attempt_rights'] is not False or q['old_STOP_removed'] is not False):
        raise PermissionError('N1_SINGLE_AUTHORITY_EPOCH_AND_OLD_OUTCOMES')
    # Recurse over at most seven immutable boundaries, never replace earlier proof.
    old_number = target(old['execution'].get('cold_engineering_binding_relative'))
    if number == 6:
        if old['execution'].get('cold_engineering_binding_relative') != fifth.BINDING:
            raise PermissionError('N1_FIRST_BOUNDARY_MUST_RETAIN_FIFTH')
    elif old_number != number - 1:
        raise PermissionError('N1_NO_SKIPPED_COLD_BOUNDARY_HISTORY')
    stops=q['preserved_STOP_hashes']
    descendants=preserved_stops(root,number,stops,_descendant_stop_hashes)
    if old_number is None:prior.validate(root,proposed_manifest=old)
    else:prior.validate(root,proposed_manifest=old,_descendant_stop_hashes=descendants)
    if ({k:v for k,v in m.items() if k not in ['code', 'execution']}
            != {k:v for k,v in old.items() if k not in ['code', 'execution']}):
        raise PermissionError('N1_CANNOT_CHANGE_BIRTH_SCIENCE_OR_LIMITS')
    variable = {'sealed_inputs', 'cold_engineering_binding_relative', 'n1_authority_binding'}
    if ({k:v for k,v in m['execution'].items() if k not in variable}
            != {k:v for k,v in old['execution'].items() if k not in variable}):
        raise PermissionError('N1_CANNOT_CHANGE_ENTRY_IDENTITY')
    variable = {'vla_commit', 'capsule_relative', 'source_seal_relative', 'source_seal_sha256', 'unchanged_learning_files'}
    if ({k:v for k,v in m['code'].items() if k not in variable}
            != {k:v for k,v in old['code'].items() if k not in variable}):
        raise PermissionError('N1_CANNOT_CHANGE_UPSTREAM_OR_NUMERICAL_BINDINGS')
    from . import endpoint_anchor_ni as ni
    extra=ni.binding_files(q)
    expected = dict(old['execution']['sealed_inputs'],
                    **{n:sha(root / n) for n in names + n1.FILES + list(stops) + [TRANSPORT_PLAN]+extra})
    if (number > 6 and old['execution']['sealed_inputs'].get(TRANSPORT_PLAN)
            != expected[TRANSPORT_PLAN]):
        raise PermissionError('N1_RECORDED_TRANSPORT_REFERENCE_CANNOT_CHANGE_ON_RESUME')
    if m['execution']['sealed_inputs'] != expected:
        raise PermissionError('N1_NO_REMOVED_OR_UNDECLARED_INPUTS')
    if extra:
        ni.validate_binding(root,q,m)
        ni.checked_source_boundary(root,past,rows)
    native_history(past, native, number, effective['source'])
    previous_ledger = read(root / (fifth.LEDGER if number == 6 else paths(number - 1)[2]))
    for key in ['GPU_allocated_s', 'generation_queries', 'optimizer_updates', 'CPU_s']:
        if key in previous_ledger and past.get(key, -1) < previous_ledger[key]:
            raise PermissionError('N1_PRIOR_CUMULATIVE_COUNTERS_CANNOT_RESET')
    current_path = root / 'HOPPER_RESOURCE_LEDGER.json'
    if current_path.is_file():
        current = read(current_path)
        for key in ['GPU_allocated_s', 'generation_queries', 'optimizer_updates', 'CPU_s']:
            if key in past and current.get(key, -1) < past[key]:
                raise PermissionError('N1_CURRENT_LEDGER_CANNOT_LOSE_BOUND_PRIOR_COST')
    if (len(rows) < 22 or len({v['attempt_id'] for v in rows}) != len(rows)
            or any(v['live_native_failures'] for v in rows)):
        raise PermissionError('N1_ALL_PREVIOUS_OUTCOMES_NO_NATIVE_SAFETY_BYPASS')
    plan = read(root / 'plans/SOURCE_EVALUATION_PLAN.json')
    from .behavior_matrix import fixed_cards
    cards = fixed_cards(plan)
    for row in rows:
        base = row['attempt_id'].removesuffix('-REPLACEMENT-01')
        card = cards.get(base)
        if (card is None or row['model'] != card['mp1_model_label']
                or any(row[k] != card[k] for k in ['scene_group', 'task', 'split'])
                or row['teacher_takeover'] is not False):
            raise PermissionError('N1_ONLY_ORIGINAL_FIXED_32_CONDITIONS')
    for spec in plan['panels'].values():
        remaining_panel(rows, spec)
    previous = read(root / fifth.OUTCOMES) if number == 6 else read(root / paths(number - 1)[3])
    indexed = {v['attempt_id']:v for v in rows}
    if any(indexed.get(v['attempt_id']) != v for v in previous):
        raise PermissionError('N1_PRIOR_POLICY_AND_ENGINEERING_OUTCOMES_ARE_IMMUTABLE')
    if (effects['status'] != 'PASS_ALL_PRIOR_SOURCE_EFFECTS_CLOSED'
            or effects['pending_queries'] or effects['pending_installs'] or effects['pending_starts']
            or effects['pending_optimizer_updates']
            or effects['core_reservations_closed'] is not True or effects['core_CPU_reserved_s']
            or effects['mutation_journal_count'] != effects['completed_mutation_journal_count']
            or effects['mutation_journal_count'] < 1970
            or effects['cumulative_native_generation_queries'] != past['generation_queries']
            or effects['native_draws_all_classified'] is not True
            or effects['all_action_arrays_verified'] is not True
            or effects['prior_outcome_ids'] != sorted(indexed)
            or effects['new_model_queries'] or effects['new_physics']):
        raise PermissionError('N1_UNKNOWN_EFFECTS_NOT_RETRIED_OR_IMPUTED')
    if processes['status'] != 'PASS_PRIOR_OWNERS_AND_CONSUMERS_TERMINAL':
        raise PermissionError('N1_OLD_PROCESS_IDENTITY_CLOSURE')
    role = 'hopper' if m['execution']['profile'] == 'hopper-neural' else 'core'
    if {v['role'] for v in processes['rows']} != {'core', 'hopper', 'local_owner'}:
        raise PermissionError('N1_BOTH_HOSTS_AND_OLD_LOCAL_CONTROLLER_CLOSED')
    for entry in processes['rows']:
        if entry['terminal_verified'] is not True:
            raise PermissionError('N1_LIVE_OR_UNKNOWN_OLD_CONSUMER')
        if entry['role'] == role:
            stat = Path('/proc', str(entry['pid']), 'stat')
            if stat.exists():
                row = stat.read_text().split()
                if row[2] != 'Z' and row[21] == str(entry['proc_start']):
                    raise PermissionError('N1_OLD_CONSUMER_STILL_LIVE_ON_NATIVE_HOST')
    for name in ['SOURCE_PHYSICS_STOP.json', 'SOURCE_SAFETY_STOP_FROM_CORE.json']:
        if (root / name).exists():
            raise PermissionError('N1_NEVER_RESOLVES_REAL_NATIVE_SAFETY_STOP')
    for manifest in [m] + ([worker_manifest] if worker_manifest is not None else []):
        seal = root / manifest['code']['source_seal_relative']
        hashes = old['code']['unchanged_learning_files']
        if (sha(seal) != manifest['code']['source_seal_sha256']
                or manifest['code'].get('unchanged_learning_files') != hashes
                or any(read(seal)['files'].get(k) != v for k,v in hashes.items())):
            raise PermissionError('N1_ORIGINAL_LEARNING_DATA_MODEL_AND_PLANT_BYTES')
    return q
