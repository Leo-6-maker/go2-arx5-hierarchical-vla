"""Native follow-up identities over the existing G7 action and guard path."""
from pathlib import Path

from . import multisource_d4_native as d4


SCOPE = 'SLA_D416_GRASP_AND_JOINT_NATIVE'
PLAN = d4.E0_PLAN
AUTHORITY_SOURCE = 'CURRENT_USER_D416_BOUNDED_NATIVE_FOLLOWUP'
GRASP, CONFIRM, JOINT = 'D5_GRASP24', 'D5_CONFIRM32', 'J_SCREEN48'
CONTROL = 'D4-B'
CANDIDATES = ('D5-W', 'D5-T')
JOINT_ARMS = ('J0', 'J1')
ARMS = (CONTROL, *CANDIDATES, *JOINT_ARMS)
POPULATION_SHA = 'bc2f4bddf256bf957489a096335f2a541ab6d390715289214420c1cfb0631a50'
CAPS = {GRASP:(24, 856872, 2880), CONFIRM:(32, 1142496, 3840), JOINT:(48, 2193744, 7680)}


def manifest_contract(manifest):
    from go2arx5_vla.sla_adapter import grasp_supervision_d5, joint_source_d4, multisource_d4

    arm = manifest['bindings']['arm']
    if arm == CONTROL:
        multisource_d4.package_contract(manifest)
    elif arm in CANDIDATES:
        grasp_supervision_d5.package_contract(manifest)
    elif arm in JOINT_ARMS:
        joint_source_d4.package_contract(manifest)
    else:
        raise ValueError('PREREGISTERED_D416_G7_PACKAGE_REQUIRED')


def endpoint(arm, package, result, restore, *, core_replay=None, source_restore=None):
    from .multitask_scope_v6 import read_pin

    if arm == CONTROL:
        return d4.endpoint(arm, package, result, restore)
    if arm not in (*CANDIDATES, *JOINT_ARMS):
        raise ValueError('PREREGISTERED_D416_ENDPOINT_REQUIRED')
    manifest = read_pin(dict(path=str(Path(package['path'])/'MANIFEST.private.json'), sha256=package['sha256']))
    manifest_contract(manifest)
    actual, rgb, core = (read_pin(pin) for pin in (result, restore, core_replay))
    visits = 4000 if arm == 'J1' else 0
    if (actual['arm'] != arm or actual['exit_code'] or actual['package']['sha256'] != package['sha256']
            or manifest['bindings']['arm'] != arm or actual['last_checkpoint']['sha256'] != manifest['original_complete_sha256']
            or (actual['new_updates'],actual['new_visits'],actual['new_source_visits']) != (500,16000,visits)
            or actual['actual']['training_full_generations'] != 2000 or actual['new_native_requests']
            or not actual['full_restore']['bit_exact'] or actual['new_integrations']
            or rgb['package']['sha256'] != package['sha256'] or rgb['exit_code'] or not rgb['passed']
            or rgb['actual_generations'] != 16 or len(rgb['observations']) != 16
            or not all(r['normalized_exact'] and r['EEF10_exact'] for r in rgb['observations'])
            or core['arm'] != arm or core['package_sha256'] != package['sha256']
            or core['actual_generations'] != 16 or core['normalized_bit_exact'] != 16
            or not core['original_H40_seam_decisions_match']
            or core['native_requests'] or core['native_integrations']):
        raise ValueError('D416_FIXED500_AND_BOTH_ORDINARY_RGB_RESTORES_REQUIRED')
    if arm in JOINT_ARMS:
        native_source = read_pin(source_restore)
        if (native_source['exit_code'] or not native_source['passed'] or not native_source['full_restore']['bit_exact']
                or native_source['actual_generations'] != 16 or len(native_source['observations']) != 16
                or not all(r['normalized_exact'] and r['reference_exact'] for r in native_source['observations'])
                or not actual['source_private_bit_exact']
                or actual['source_actual']['ordinary_RGB_generations'] != 96
                or actual['source_actual']['completed_training_visits'] != visits):
            raise ValueError('D416_JOINT_REQUIRES_SEALED_SOURCE_DEV_AND_INDEPENDENT_RESTORE')
    return manifest, actual


def arms_for(spec):
    from .multitask_scope_v6 import read_pin

    stage = spec['evaluation_stage']
    if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN or spec['physics'] is not True
            or spec['compiled_population']['sha256'] != POPULATION_SHA
            or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']):
        raise PermissionError('STRICT_SEPARATE_D416_NATIVE_FOLLOWUP_SCOPE_REQUIRED')
    if stage == GRASP:
        arms = (CONTROL, *CANDIDATES)
    elif stage == JOINT:
        arms = JOINT_ARMS
    elif stage == CONFIRM:
        from .mfree_unattended_runtime_v1 import unit_state
        from .storage import digest

        screen_pin = spec['screen_decision']
        screen = read_pin(screen_pin)
        result = read_pin(screen['campaign_result'])
        root = Path(screen['campaign_result']['path']).parent
        lease_path = root/'control/LEASE.json'
        lease = read_pin(dict(path=str(lease_path), sha256=digest(lease_path)))
        state = unit_state('wm02-'+root.name)
        chosen = screen['selected_arm']
        if (screen['stage'] != GRASP or not screen['complete_comparable'] or chosen not in CANDIDATES
                or result['exit_code'] or result['new_verified_native_requests'] != CAPS[GRASP][0]
                or screen['population_sha256'] != POPULATION_SHA or lease['status'] != 'CLOSED'
                or state['pid'] or state['cgroup_pids']
                or screen['selected_package_sha256'] != spec['packages'][chosen]['sha256']):
            raise PermissionError('COMPARABLE_24_AND_FROZEN_CANDIDATE_REQUIRED_FOR_CONFIRM32')
        arms = (CONTROL, chosen)
    else:
        raise ValueError('UNKNOWN_D416_FOLLOWUP_PANEL')
    if set(arms) != set(spec['packages']):
        raise ValueError('D416_REQUIRES_ALL_PREDECLARED_ARMS')
    return arms


def conditions(population, stage, arms):
    from scripts.wm02.sla_followup_population_v1 import validate_frozen

    validate_frozen(population)
    picked = []
    for row in population['conditions'][stage]:
        for arm in row['arm_order']:
            if arm == 'SELECTED_D5':
                chosen, = set(arms).intersection(CANDIDATES)
                arm = chosen
            if arm not in arms:
                raise ValueError('EXACT_SEALED_D416_ARM_ORDER_REQUIRED')
            picked.append((row, arm))
    if len(picked) != CAPS[stage][0]:
        raise ValueError('D416_FIXED_REQUEST_CAP_CHANGED')
    return picked
