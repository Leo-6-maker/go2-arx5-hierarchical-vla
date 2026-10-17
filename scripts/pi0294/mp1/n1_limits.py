"""A sealed N1 overlay never rewrites a run's original authority or costs."""
from copy import deepcopy
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha

RELEASE = '926f9ef69042bb9f3b7b336a076d2be8fa0afbb9'
PARENT_AUTHORITY = 'c8cdf60e41e31da5796b44a064ef2f22d594a491'
PARENT_RUN = '20261016T191445Z-localpc-mp1'
ORIGINAL_T0 = 1792178085
BUNDLE = {
    'docs/handover/2026-10-18/MP1_N1_OVERNIGHT_EXECUTE.md':
        '855521cb1263f58519103dedec672d5f3beb67be996102cc5f347d38b741070b',
    'docs/protocols/PI0294_MP1_N1_OVERNIGHT_AUTONOMY_20261018.md':
        '1889b12a2dea570f5ebbf10727821ad34a53f121f4da1a0c2cfe426f41dc04aa',
    'reports/PI0294/design/MP1_N1_OVERNIGHT_AUTONOMY_20261018.json':
        '6a822bf37f123f3106cfbe736aff7395f4ad960409599eb43ac49bfbfa2d6d58',
}
CARD = 'plans/N1/MP1_N1_OVERNIGHT_AUTONOMY_20261018.json'
DELTA = 'plans/N1/ADOPTION_DELTA.json'
PROTOCOL = 'plans/N1/PROTOCOL.md'
HANDOFF = 'plans/N1/HANDOFF.md'
FILES = [HANDOFF, PROTOCOL, CARD, DELTA]
FIRST_ADOPTION = 'authority/N1/ADOPTION_DELTA.json'


def validated_delta(delta, original):
    """Check one first adoption, not an allocation-local clock extension."""
    if (delta['schema'] != 'MP1_N1_ADOPTION_DELTA_V1'
            or delta['id'] != 'MP1_N1_SINGLE_AUTHORITY_ADOPTION_01'
            or delta['authority_release'] != RELEASE
            or delta['parent_authority_release'] != PARENT_AUTHORITY
            or delta['parent_run_uuid'] != PARENT_RUN
            or delta['source_child'] != 'source-pair-01'
            or delta['owner'] != 'EXISTING_SINGLE_LOCAL_PC'
            or delta['original_T0_epoch'] != ORIGINAL_T0
            or delta['bundle_sha256'] != BUNDLE
            or delta['reset_on_resume'] is not False
            or delta['new_competing_owner'] is not False
            or delta['original_birth_manifests_cards_and_costs_preserved'] is not True
            or delta['old_scope_costs_reclassified'] is not False):
        raise PermissionError('EXACT_SINGLE_N1_AUTHORITY_AND_ORIGINAL_IDENTITY')
    if (original['authority_release'] != PARENT_AUTHORITY
            or original['run_uuid'] != PARENT_RUN
            or original['T0_epoch'] != ORIGINAL_T0):
        raise PermissionError('N1_CANNOT_CREATE_ANOTHER_MP1_PARENT_OR_T0')
    began = delta['autonomy_started_epoch']
    published = delta['publication_epoch']
    if (type(began) is not int or type(published) is not int
            or published != 1792261823 or not published <= began < published + 12 * 3600):
        raise PermissionError('N1_ONE_FIRST_ADOPTION_WITHIN_PUBLICATION_WINDOW')
    for key, hours in [('new_submission_and_candidate_cutoff_epoch', 30),
                       ('new_experiment_cutoff_epoch', 34), ('hard_closeout_epoch', 36)]:
        if delta[key] != min(began + hours * 3600, original[key]):
            raise PermissionError('N1_BOUNDED_WINDOW_CANNOT_RESET_OR_EXTEND_MP1')
    return delta


def effective(original, card, delta):
    """Return cumulative ceilings, leaving all caller-owned birth data untouched."""
    validated_delta(delta, original)
    if card['schema'] != 'MP1_N1_OVERNIGHT_ENGINEERING_AUTONOMY_V1':
        raise PermissionError('EXACT_N1_CARD_SCHEMA')
    change = card['cumulative_amendments']
    result = dict(authority_release=RELEASE, original_T0_epoch=ORIGINAL_T0,
                  parent=deepcopy(original['incremental_budget']),
                  source=dict(GPU_s=change['source_H200_GPU_seconds']['new'],
                              PBS=change['source_PBS_attempts']['new'],
                              CPU_s=deepcopy(change['source_CPU_core_seconds']['new']),
                              generation_queries=change['source_generation_queries']['new'],
                              job_wall_s_max=change['source_new_job_wall_seconds']['new_max']),
                  parent_H200_s=change['parent_H200_GPU_hours']['new'] * 3600,
                  parent_A40_s=change['parent_A40_GPU_hours']['new'] * 3600,
                  parent_GPU_PBS=dict(Hopper=change['parent_Hopper_GPU_PBS_attempts']['new'],
                                      Vanda=change['parent_Vanda_GPU_PBS_attempts']['new']),
                  parent_Vanda_CPU_build_PBS=change['parent_Vanda_CPU_build_attempts']['new'],
                  parent_CPU_core_hours=deepcopy(change['parent_CPU_core_hours']['new']),
                  parent_network_GiB=change['parent_network_transfer_GiB']['new'],
                  final_GPU_reserve_s=dict(Hopper=21600, Vanda=14400),
                  unchanged=deepcopy(card['unchanged_resource_limits']))
    result['parent'].update(
        semantics='CUMULATIVE_MP1_CEILINGS_INCLUDE_PRIOR_MP1_USED_AND_RESERVED_NOT_E0_E1_RECLASSIFICATION',
        H200_allocated_GPU_hours=change['parent_H200_GPU_hours']['new'],
        A40_allocated_GPU_hours=change['parent_A40_GPU_hours']['new'],
        Hopper_new_GPU_PBS_submissions=result['parent_GPU_PBS']['Hopper'],
        Vanda_new_GPU_PBS_submissions=result['parent_GPU_PBS']['Vanda'],
        Vanda_new_CPU_build_submissions=result['parent_Vanda_CPU_build_PBS'],
        CPU_core_hours=deepcopy(result['parent_CPU_core_hours']),
        new_network_transfer_GiB_max=result['parent_network_GiB'],
        final_evaluation_GPU_reserve_hours=dict(Hopper=6, Vanda=4))
    for key in ['autonomy_started_epoch', 'new_submission_and_candidate_cutoff_epoch',
                'new_experiment_cutoff_epoch', 'hard_closeout_epoch']:
        result[key] = delta[key]
    return result


def resolve(root, manifest=None):
    """Only a future immutable consumer explicitly sealed to N1 can use it."""
    root = Path(root)
    manifest = read(root / 'RUN_MANIFEST.json') if manifest is None else manifest
    binding = manifest.get('execution', {}).get('n1_authority_binding')
    if binding is None:
        return None
    if root.name != 'source-pair-01' or root.parent.parent.name != PARENT_RUN:
        raise PermissionError('N1_EXACT_EXISTING_SOURCE_CHILD')
    if binding.get('release') != RELEASE or set(binding) != {'release', 'adoption_delta_sha256'}:
        raise PermissionError('EXACT_MANIFEST_BOUND_N1_LAYER')
    sealed = manifest['execution']['sealed_inputs']
    for name in FILES:
        path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_mode & 0o222 or sealed.get(name) != sha(path)):
            raise PermissionError('N1_IMMUTABLE_COMPLETE_BUNDLE_AND_DELTA')
    for name, expected in zip(FILES[:3], BUNDLE.values()):
        if sha(root / name) != expected:
            raise PermissionError('N1_FIXED_RELEASE_BYTES')
    if sha(root / DELTA) != binding['adoption_delta_sha256']:
        raise PermissionError('N1_SAME_FIRST_ADOPTION_NO_ALLOCATION_RESET')
    parent = root.parent.parent
    first = parent / FIRST_ADOPTION
    if (first.is_symlink() or not first.resolve().is_relative_to(parent.resolve())
            or first.stat().st_mode & 0o222 or sha(first) != binding['adoption_delta_sha256']):
        raise PermissionError('N1_PARENT_FIRST_ADOPTION_MUST_MATCH_EVERY_COLD_CONSUMER')
    original = read(parent / 'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    return effective(original, read(root / CARD), read(root / DELTA))


def allocation_number(label, maximum=5):
    """Use one contiguous range; extended labels alone never grant permission."""
    if type(maximum) is not int or not 1 <= maximum <= 12:
        raise ValueError('EXPLICIT_BOUNDED_ALLOCATION_CAP')
    base, separator, suffix = label.rpartition('-ALLOCATION-')
    if (not base or separator != '-ALLOCATION-' or len(suffix) != 2
            or not suffix.isascii() or not suffix.isdecimal()
            or not 1 <= int(suffix) <= maximum):
        raise PermissionError('EXACT_FIXED_PANEL_AND_ADOPTED_ALLOCATION_RANGE')
    return base, int(suffix)


def require_remaining(used, reserved, amount, ceiling):
    values = [used, reserved, amount, ceiling]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('EXACT_NONNEGATIVE_CUMULATIVE_RESOURCE_COUNTS')
    if used + reserved + amount > ceiling:
        raise PermissionError('N1_OLD_USED_AND_CURRENT_RESERVATIONS_COUNT_ONCE')
