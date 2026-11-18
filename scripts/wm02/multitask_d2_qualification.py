"""Independent cold evidence gates for D2 development and target TRAIN."""

from collections import Counter
from pathlib import Path

from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.multitask_scope_d2 import PLAN, SCOPE
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read


def frozen_candidate(spec):
    return spec['source_sha'], spec['source_seal'], spec['d2_profile']


def verified_campaign(binding, *, stage, population, candidate):
    spec = read_pin(binding['spec'])
    result = read_pin(binding['result'])
    root = Path(spec['registered_root'])
    lease = read(root/'control/LEASE.json')
    unit = unit_state('wm02-'+root.name)
    stage_entry = read_pin(spec['d2_science_admission'])
    if (spec['plan_commit'] != PLAN or spec['scope'] != SCOPE or spec['d2_stage'] != stage
            or frozen_candidate(spec) != candidate or stage_entry['compile_population'] != population
            or result['source_sha'] != spec['source_sha'] or result['unresolved_case_launches']
            or result['new_verified_native_requests'] != len(result['records'])
            or lease['status']!='CLOSED' or lease['spec_sha256']!=binding['spec']['sha256']
            or unit['pid'] or unit['cgroup_pids'] or Path(binding['result']['path'])!=root/'RESULT.private.json'):
        raise ValueError('D2_EXACT_COLD_MATCHED_CAMPAIGN_REQUIRED')
    paths = {Path(path).parent.name:Path(path) for path in spec['cases']}
    accepted, rejected = [], []
    for row in result['records']:
        cid = row['id']
        case_path = paths[cid]
        case = read(case_path)
        case_root = case_path.parent
        terminal = dict(path=str(case_root/'RESULT.private.json'),sha256=row['result_sha256'])
        native = read_pin(terminal)
        owned = read(case_root/'control/LEASE.json')
        if (case_root.parent!=root/'cases' or owned['status']!='CLOSED'
                or owned['spec_sha256']!=digest(case_path) or case['panel_entry_id']!=cid
                or native['source_sha']!=spec['source_sha'] or native['physical_requests']!=1
                or native['optimizer_updates'] or native['training_visits'] or native['model_generations']
                or native['actual_native_integrations']!=row['native_ticks']
                or row['integrations_after_STOP'] or row['native_ticks']>case['max_ticks']):
            raise ValueError('D2_NATIVE_CASE_MUST_RECONCILE_BEFORE_GATE')
        if row['task_and_mobile_success'] and row['completed'] and row.get('G7_pass'):
            g7 = read_pin(dict(path=str(case_root/'D2_G7_ACTUAL.private.json'),sha256=row['G7']['sha256']))
            if (not g7['passed'] or g7['native_result_sha256']!=terminal['sha256']
                    or g7['statistics_fitted'] or native['technical_failure'] or native['unsafe']):
                raise ValueError('D2_G7_RECEIPT_AND_CLOSED_NATIVE_CASE_REQUIRED')
            accepted.append((row, case, terminal, native))
        else:
            rejected.append(dict(id=cid,native_ticks=row['native_ticks'], first_failure=row['first_failure'],
                hard_STOP=row['hard_STOP'],technical_failure=row['technical_failure'],result=terminal))
    return accepted, rejected, [r['id'] for r in result['records']], result['native_ticks']


def readiness(bindings, population_pin, candidate):
    population = read_pin(population_pin)
    development = {row['id']:row for row in population['conditions']['DEVELOPMENT']}
    successful = Counter()
    attempted = set()
    for bound in bindings:
        passed, _, ids, _ = verified_campaign(bound,stage='D',population=population_pin,candidate=candidate)
        if not ids or (set(ids)&attempted) or not set(ids)<=development.keys():
            raise ValueError('D2_DISTINCT_PREREGISTERED_DEVELOPMENT_CASES_REQUIRED')
        if len(ids)!=12 or len({development[cid]['block'] for cid in ids})!=1:
            raise ValueError('D2_SEALED_FULL_DEVELOPMENT_BLOCK_REQUIRED')
        attempted.update(ids)
        successful.update((development[row['id']]['family'],development[row['id']]['instruction'])
                          for row, *_ in passed)
    coverage = {f'{f}/{task}':successful[f,task] for f in ('F1','F2') for task in INSTRUCTIONS}
    return dict(passed=all(coverage.values()),cases=len(attempted),margin_and_capture_passes=sum(successful.values()),
                coverage=coverage,candidate=list(candidate),population_sha256=population_pin['sha256'])


def export(spec, destination):
    """Use only the new frozen QUAL population for the new target component."""
    from scripts.sla.export_multitask_g7_v6 import export_qualified_population

    if spec['physics'] or spec['plan_commit']!=PLAN or spec['new_physical_requests_budget']:
        raise PermissionError('D2_EXPORT_IS_A_SEPARATE_NONINTEGRATING_STAGE')
    population_pin = spec['compiled_population']
    candidate = tuple(spec['science_candidate'])
    if not readiness(spec['development_campaigns'],population_pin,candidate)['passed']:
        raise PermissionError('D2_COMPLETE_DEVELOPMENT_G7_COVERAGE_REQUIRED')
    planned = read_pin(population_pin)['conditions']['QUAL']
    planned_ids = [row['id'] for row in planned]
    if len(spec['teacher_campaigns'])!=1 or len(planned)!=48:
        raise PermissionError('D2_EXACT_FROZEN_QUALIFICATION_POPULATION_REQUIRED')
    cases, failed, resolved, ticks = verified_campaign(spec['teacher_campaigns'][0],
                                                       stage='Q',population=population_pin,candidate=candidate)
    if resolved!=planned_ids:
        raise PermissionError('D2_EXACT_FROZEN_QUALIFICATION_POPULATION_REQUIRED')
    by_id = {row['id']:row for row in planned}
    selected = []
    for row,case,terminal,native in cases:
        condition = by_id[row['id']]
        if (case['task_instruction_id']!=condition['instruction']
                or case['mobile_condition']['family']!=condition['family']
                or case['mobile_condition']['layout']!=condition['cluster']):
            raise ValueError('D2_QUAL_BOUND_NATIVE_CONDITION_REQUIRED')
        selected.append((condition,case,terminal,native))
    return export_qualified_population(spec,destination,selected,failed,ticks)
