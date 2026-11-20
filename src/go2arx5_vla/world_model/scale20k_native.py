"""Fixed Go2 native evaluation identities for the three SCALE20K snapshots."""
from collections import Counter

from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, OFFLINE_PER_DOMAIN, RESTORE_ORDINALS
from go2arx5_vla.sla_adapter.scale20k_package import ENDPOINTS, PLAN, SCOPE, read_package
from .multitask_score_v6 import INSTRUCTIONS


AUTHORITY_SOURCE = 'CURRENT_USER_SCALE20K_BOUNDED_NATIVE_EVALUATION'
POPULATION_SHA = '86ad22180a0e1fd937fdff28302e190fef65928c0d8b724de282b6c8e0eb0c4a'
POPULATION_SCHEMA = 'SLA_SCALE20K_FROZEN_TARGET_EVALUATION_V1'
RGB_SCHEMA = 'SLA_SCALE20K_CORE_TARGET_RGB16_ACTUAL_V1'
DEV, CONFIRM = 'SCALE_DEV', 'SCALE_CONFIRM'
CAPS = {stage: (24, 1096872, 3840) for stage in (DEV, CONFIRM)}
BLOCK_LIMIT = 6


def next_block(spec, ordered, carried):
    """Select the next contiguous block of the already frozen endpoint population."""
    if 'condition_stop' not in spec:
        return ordered[carried:]
    end = spec['condition_stop']
    if type(carried) is not int or type(end) is not int or not 0 <= carried < end <= min(len(ordered),carried+BLOCK_LIMIT):
        raise ValueError('SCALE20K_REQUIRES_THE_NEXT_FIXED_BLOCK_OF_AT_MOST_SIX')
    return ordered[carried:end]


def stage_arm(stage, arms):
    if (stage not in CAPS or len(arms) != 1 or not set(arms).issubset(ENDPOINTS)
            or (stage == CONFIRM and set(arms) != {'SCALE-MIX20K'})):
        raise ValueError('SCALE20K_STAGE_REQUIRES_ONE_FROZEN_ENDPOINT')
    return next(iter(arms))


def conditions(population, stage, arms):
    arm = stage_arm(stage, arms)
    if (population['schema'] != POPULATION_SCHEMA or population['plan'] != PLAN
            or not population['scene_instruction_independent'] or population['old_TEST_reused'] is not False):
        raise ValueError('SCALE20K_FROZEN_DISJOINT_DEV_AND_CONFIRMATION_REQUIRED')
    clusters = {r['id']:r for r in population['clusters'] if r['stage'] == stage}
    rows = population['conditions'][stage]
    scenes = {r['scene']['sha256'] for r in population['clusters']}
    if (len(population['clusters']) != 8 or len(scenes) != 8 or len(clusters) != 4
            or Counter(r['family'] for r in clusters.values()) != Counter(F1=2,F2=2)
            or len({r['reset'] for r in population['clusters']}) != 8
            or len(rows) != 24 or len({r['id'] for r in rows}) != 24
            or Counter((r['cluster'],r['instruction']) for r in rows) !=
                Counter((cluster,instruction) for cluster in clusters for instruction in INSTRUCTIONS)
            or any((r['family'],r['reset']) != (clusters[r['cluster']]['family'],clusters[r['cluster']]['reset']) for r in rows)):
        raise ValueError('SCALE20K_REQUIRES_THE_ORIGINAL_FOUR_LAYOUT_SIX_INSTRUCTION_PANEL')
    return [(r,arm) for r in rows]


def arms_for(spec):
    arm = stage_arm(spec['evaluation_stage'], spec['packages'])
    if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN or spec['physics'] is not True
            or spec['compiled_population']['sha256'] != POPULATION_SHA or spec['new_optimizer_updates_budget']
            or spec['new_training_visits_budget']):
        raise PermissionError('SCALE20K_FROZEN_NATIVE_EVALUATION_SCOPE_REQUIRED')
    return {arm}


def endpoint(arm, package, evaluation, core_restore):
    from .multitask_scope_v6 import read_pin

    manifest = read_package(package['path'],package['sha256'])
    actual, core = (read_pin(pin) for pin in (evaluation,core_restore))
    if ((manifest['arm'],manifest['step']) != ENDPOINTS[arm]
            or evaluation['sha256'] != manifest['offline_evaluation_sha256']
            or actual['exit_code'] or actual['status'] != 'ENDPOINT_PREDICTIONS_SEALED'
            or (actual['arm'],actual['step']) != ENDPOINTS[arm] or actual['optimization_updates']
            or actual['checkpoint']['sha256'] != manifest['checkpoint_sha256']
            or actual['panel']['sha256'] != manifest['panel_sha256']
            or actual['new_model_generations'] != len(DOMAINS)*OFFLINE_PER_DOMAIN
            or len(actual['rows']['target']) != OFFLINE_PER_DOMAIN or actual['old_TEST_access'] or actual['new_native_requests']
            or core['schema'] != RGB_SCHEMA or core['arm'] != arm or core['exit_code'] or not core['passed']
            or core['package_sha256'] != package['sha256'] or core['evaluation_sha256'] != evaluation['sha256']
            or core['site']['host'] != 'corelab-4080s' or core['new_model_generations'] != 16
            or core['new_native_requests'] or core['new_native_ticks'] or core['optimizer_updates']
            or tuple(r['ordinal'] for r in core['rows']) != RESTORE_ORDINALS):
        raise ValueError('SCALE20K_SEALED_ENDPOINT_AND_ACTUAL_CORE_RGB16_REQUIRED')
    for row in core['rows']:
        recorded = actual['rows']['target'][row['ordinal']]
        if (any(recorded[k] != row[k] for k in ('index','noise_id','normalized_sha256','decoded_sha256'))
                or not row['same_original_H40_and_seam_decision']):
            raise ValueError('SCALE20K_CORE_RGB16_OUTPUT_OR_ORIGINAL_GUARD_CHANGED')
    return manifest,actual
