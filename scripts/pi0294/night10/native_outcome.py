"""Read native effects separately from scientific success; reuse existing close IO."""
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_outcome(result, capacity=None, *, closed):
    """A closed resource refusal is not a successful 96-generation qualification."""
    if not closed:
        raise PermissionError('DIAGNOSTIC_NATIVE_EFFECTS_NOT_CLOSED')
    if result.get('optimizer_updates') != 0 or result.get('physics_integrations') != 0:
        raise PermissionError('DIAGNOSTIC_UNEXPECTED_TRAINING_OR_PHYSICS')
    complete = (result.get('status') == 'COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC'
        and result.get('error') is None and result.get('model_load_attempts') == 1
        and result.get('generation_attempts') == result.get('completed_generations') == 96)
    capacity_pass = (capacity is not None and capacity.get('capacity_pass') is True
        and capacity.get('completed_generations') == 96
        and capacity.get('minimum_free_bytes', 0) >= 1024**3
        and len(capacity.get('measurements', [])) == 97)
    success = complete and capacity_pass
    return dict(lifecycle='TERMINAL', native_effects_closed=True,
        scientific_result='COMPLETE_NATIVE96' if success else
            'RESOURCE_ENTRY_REFUSAL' if result.get('model_load_attempts') == 0 and
                result.get('generation_attempts') == 0 and 'CAPACITY' in (result.get('error') or '')
                else 'INCOMPLETE_DIAGNOSTIC',
        scientific_success=success, inference_capacity_pass=success,
        completed_generations=result.get('completed_generations', 0), training_admitted=False)


def terminal_usage(journal, unit):
    cpu = [int(r['CPU_USAGE_NSEC'])/1e9 for r in journal if 'CPU_USAGE_NSEC' in r]
    begin = [int(r['__REALTIME_TIMESTAMP']) for r in journal
        if r.get('MESSAGE', '').startswith('Started '+unit)]
    end = [int(r['__REALTIME_TIMESTAMP']) for r in journal if 'CPU_USAGE_NSEC' in r]
    if (len(cpu) != 1 or len(begin) != 1 or len(end) != 1 or end[0] < begin[0]
            or not math.isfinite(cpu[0]) or cpu[0] < 0):
        raise PermissionError('EXACT_TERMINAL_NATIVE_USAGE_REQUIRED')
    return dict(CPU_s=cpu[0], GPU_allocated_s_upper=math.ceil((end[0]-begin[0])/1e6))


def known_failure(root, staged, *, main_pid, journal, canonical):
    """Prove a finite model-only failure, or zero-effect pre-physics DEV failure.

    Actual physics records still require the original condition/prefix/native-score
    audit. Missing terminal records, live workers and ambiguous writes remain UNKNOWN.
    This function does not write a ledger, lease, resource grant or scientific score.
    """
    root = Path(root)
    if main_pid != 0 or sha(root/'RUN_MANIFEST.json') != staged['manifest_sha256']:
        raise PermissionError('EXACT_TERMINAL_NATIVE_IDENTITY_REQUIRED')
    manifest, result, lease = (read(root/n) for n in
        ('RUN_MANIFEST.json', 'RESULT.json', 'control/LEASE.json'))
    if (lease.get('status') != 'ACTIVE' or lease.get('manifest_sha256') != staged['manifest_sha256']
            or root.name != manifest.get('run_uuid')
            or any(canonical.get(k) != [] for k in ('unknown_effects', 'pending_updates', 'pending_queries'))):
        raise PermissionError('ACTIVE_EXACT_LEASE_AND_KNOWN_CANONICAL_EFFECTS_REQUIRED')
    status = result.get('status')
    fit = status == 'FAILED_BOOK_FINITE_OPTIMIZATION' and manifest.get('kind') == 'V7_BOOK_M32_FINITE_LEARNING'
    capacity_failed = (status == 'COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC'
        and (root/'GPU_INFERENCE_CAPACITY.json').exists()
        and read(root/'GPU_INFERENCE_CAPACITY.json').get('capacity_pass') is False)
    diagnostic = (status in ('FAILED_MATCHED_TRAIN32_PHASE_DIAGNOSTIC',
        'COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC')
        and manifest.get('kind') == 'V7_BOOK_PHASE_MATCHED_TRAIN32_DIAGNOSTIC'
        and (result.get('error') or capacity_failed))
    dev = (status == 'STOPPED_PARTIAL_DEV10' and manifest.get('kind') == 'V7_BOOK_FROZEN_DEV10'
        and result.get('conditions') == [] and result.get('valid_conditions') == 0
        and result.get('model_queries') == 0 and result.get('integrations') == 0
        and result.get('RGB_pairs') == 0)
    if not (fit or diagnostic or dev) or not (result.get('error') or capacity_failed):
        raise PermissionError('KNOWN_FAILED_NATIVE_SCHEMA_REQUIRED')
    if result.get('physics_integrations', result.get('integrations')) != 0:
        raise PermissionError('PHYSICS_REQUIRES_ORIGINAL_NATIVE_CONDITION_AUDIT')
    loads = result.get('model_load_attempts')
    attempts = result.get('generation_attempts', result.get('model_queries', 0))
    completed = result.get('completed_generations', 0)
    updates = result.get('optimizer_updates', 0)
    if any(type(x) is not int or x < 0 for x in (loads, attempts, completed, updates)):
        raise PermissionError('ACTUAL_NATIVE_COUNTS_REQUIRED')
    if (loads > 1 or completed > attempts or attempts > manifest['limits']['generation_queries']
            or updates > manifest['limits']['optimizer_updates'] or (not fit and updates != 0)):
        raise PermissionError('ACTUAL_NATIVE_COUNTS_EXCEED_SEALED_SCOPE')
    if diagnostic and (attempts != len(list(root.glob('*.INTENT.json')))
            or completed != len(list(root.glob('*.RESULT.json')))):
        raise PermissionError('ACTUAL_PARTIAL_DIAGNOSTIC_FILES_MUST_MATCH_COUNTS')
    files = {n: sha(root/n) for n in ('RESULT.json', 'FIT_PROGRESS.json', 'STOP.json',
        'DIAGNOSTIC_PROGRESS.json', 'GPU_INFERENCE_CAPACITY.json', 'GPU_ENTRY_OBSERVATION.json',
        'ACTUAL_FULL_CHECKPOINT_RESTORE.json', 'ACTUAL_PARENT_DELTA_INFERENCE_RECONSTRUCTION.json')
        if (root/n).exists()}
    actual = dict(model_loads=loads, optimizer_updates=updates,
        TRAIN_window_visits=result.get('TRAIN_visits', 0), generation_queries=attempts,
        completed_generations=completed, physics_integrations=0, formal_DEV_conditions=0,
        source_episode_starts=0, new_RGB_pairs=0, result_sha256=files['RESULT.json'],
        error=result.get('error') or 'NATIVE96_CAPACITY_NOT_QUALIFIED',
        all_costs_retained=True, scientific_success=False,
        **terminal_usage(journal, root.name+'.service'))
    if dev:
        actual.update(valid_DEV_conditions=0, DEV_successes=0, unrun_DEV_conditions=10)
    return dict(actual=actual, files=files, result_status=status,
        scientific_result='FAILED_OPTIMIZATION' if fit else
            'INCOMPLETE_DIAGNOSTIC' if diagnostic else 'PRE_PHYSICS_ENGINEERING_OR_RESOURCE_FAILURE')
