"""Bind ADAPT's sealed package to its original single-world native population."""
from pathlib import Path

from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.adapt5k_package import Provider
from go2arx5_vla.sla_adapter.checkpoint_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import RESTORE_ORDINALS

SCOPE = 'SLA_ADAPT5K_NATIVE_G7'
GOAL = '1b2a25b7c1720051a2627d30831fb18130462374'
AUTHORITY_SOURCE = 'CURRENT_USER_ADAPT5K_GOAL_V2_SINGLE_WORLD'
POPULATION_SHA = 'cdcd114a712739315d06ee923fe8221142d911ab6b60a64f30eac1b619eec2e0'
DEPLOY_SCHEMA = 'SLA_LONGRUN_DEPLOY_COMPAT_V2'
DEPLOY_CHECKS = {'xyz_m':.001,'so3_rad':.01,'gripper_m':.0005,'closing_s':.050}
DEV, CONFIRM = 'ADAPT_DEV','ADAPT_CONFIRM'
ENDPOINTS = {'ADAPT0':0,'ADAPT2K':2000,'ADAPT5K':5000}
CAPS = dict.fromkeys((DEV,CONFIRM),(24,1096872,3840))


def arms_for(spec):
    stage, arms = spec['evaluation_stage'], set(spec['packages'])
    if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN or spec['goal_plan_sha'] != GOAL
            or spec['compiled_population']['sha256'] != POPULATION_SHA or spec['physics'] is not True
            or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']
            or stage not in CAPS or len(arms) != 1 or not arms <= ENDPOINTS.keys()
            or (stage == CONFIRM and arms != {'ADAPT5K'})):
        raise PermissionError('ADAPT_NATIVE_REQUIRES_FIXED_POPULATION_AND_ONE_SEALED_ENDPOINT')
    return arms


def conditions(population, stage, arms):
    from scripts.sla.prepare_adapt5k_native_population import validate

    validate(population)
    arm, = arms
    return [(row,arm) for row in population['conditions'][stage]]


def next_block(spec, expected, consumed):
    end = spec['condition_stop']
    if type(end) is not int or not consumed < end <= min(consumed+6,len(expected)):
        raise ValueError('ADAPT_NATIVE_REQUIRES_THE_NEXT_FIXED_BLOCK_OF_AT_MOST_SIX')
    return expected[consumed:end]


def package_manifest(arm, package):
    if arm != 'ADAPT0':
        raise PermissionError('ADAPT_NATIVE_ENDPOINT_PACKAGE_NOT_YET_ADMITTED')
    return Provider.read_manifest(Path(package['path']),package['sha256'])


def endpoint(arm, package, evaluation, deployment, *, gpu_uuid):
    manifest = package_manifest(arm,package)
    formal, actual = read_pin(evaluation),read_pin(deployment)
    proof = actual['proof']
    declaration = read_pin(proof['declaration'])
    ordinary, restore = (read_pin(proof[k]) for k in ('ordinary','restore'))
    if (manifest['evaluation_sha256'] != evaluation['sha256'] or proof['A40_formal'] != evaluation
            or formal['checkpoint']['sha256'] != manifest['ADAPT0_sha256'] or formal['phase'] != ENDPOINTS[arm]
            or formal['panel']['sha256'] != manifest['panel_sha256'] or formal['model_generations'] != 376
            or formal['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or formal['role'] != 'formal'
            or formal['exit_code'] or formal['old_TEST_access'] or formal['new_updates'] or formal['native_requests']
            or actual['schema'] != DEPLOY_SCHEMA or declaration['schema'] != DEPLOY_SCHEMA
            or actual['passed'] is not True or actual['strict_failure_preserved'] is not True
            or actual['model_generations'] != 32 or actual['denominator'] != 16 or actual['native_requests']
            or actual['optimizer_updates'] or declaration['limits'] != DEPLOY_CHECKS
            or declaration['checkpoint_sha256'] != manifest['ADAPT0_sha256']
            or declaration['panel_sha256'] != manifest['panel_sha256']
            or declaration['ordinary_A40_formal_sha256'] != evaluation['sha256']):
        raise PermissionError('ADAPT_NATIVE_REQUIRES_THE_ACTUAL_FIXED_DEPLOYMENT_V2_PASS')
    for role,row in (('ordinary',ordinary),('restore',restore)):
        if (row['compat_role'] != role or row['status'] != 'CORE_ADAPT0_COMPAT_V2_ARRAYS_SEALED'
                or row['package'] != package['sha256'] or row['phase'] != ENDPOINTS[arm] or row['exit_code']
                or row['site']['host'] != 'corelab-4080s' or row['site']['GPU_uuid'] != gpu_uuid
                or row['deployment_compat_v2'] != proof['declaration'] or row['model_generations'] != 16
                or row['generation_attempts'] != 16 or row['optimizer_updates'] or row['native_requests']):
            raise PermissionError('ADAPT_NATIVE_REPLAY_PACKAGE_GPU_OR_ROLE_CHANGED')
    if (ordinary['environment'] != restore['environment'] or ordinary['process_identity'] == restore['process_identity']
            or tuple(r['ordinal'] for r in actual['rows']) != RESTORE_ORDINALS
            or any(not (r['passed'] and r['cross_GPU_guard_exact'] and r['same_Core_restore_bitwise']
                        and r['closing_presence_equal'] and r['P10_closing_equal'] and r['inside_limits'])
                   for r in actual['rows'])):
        raise PermissionError('ADAPT_NATIVE_REQUIRES_ALL_SIXTEEN_GUARD_AND_RESTORE_PAIRS')
    return manifest,formal
