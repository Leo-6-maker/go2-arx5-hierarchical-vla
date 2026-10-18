"""Finite original-distribution population, fixed before any policy results."""
from copy import deepcopy

from scripts.pi0294.scene_cards import digest, normalise_layouts, scene_group, task_card


COUNTS = dict(TRAIN=dict(F0=24, F1=36, F2=36),
    DEV=dict(F0=2, F1=5, F2=5), LOCKBOX=dict(F0=4, F1=8, F2=8))
OFFSETS = dict(TRAIN=2000, DEV=3000, LOCKBOX=4000)
CANDIDATES_PER_SLOT = 32
BANKS = (294101, 294102)


def slots():
    return [dict(split=split, family=family, index=index)
        for split, counts in COUNTS.items() for family, count in counts.items() for index in range(count)]


def original_group(split, family, index, candidate):
    if (split not in COUNTS or family not in COUNTS[split]
            or not 0 <= index < COUNTS[split][family]
            or not 0 <= candidate < CANDIDATES_PER_SLOT):
        raise PermissionError('PV1_ONLY_FIXED_FINITE_ORIGINAL_DISTRIBUTION_POOL')
    group = scene_group('CONFIRM_CORE', family, OFFSETS[split]+index, candidate_index=candidate)
    original = group['group_id']
    identity = f'PI0294/PV1/{split}/{family}/{index:03d}'
    group.update(group_id=identity, scene_group=identity, split=split,
        original_frozen_generator_identity=original,
        original_frozen_generator_index=OFFSETS[split]+index, layout_candidate_index=candidate,
        direction='+Y' if group['goal_xy'][1] > group['source_xy'][1] else '-Y')
    return group


def candidate_pool(exclusions):
    excluded = normalise_layouts(exclusions)
    population = [dict(**slot, candidates=[original_group(**slot, candidate=i)
        for i in range(CANDIDATES_PER_SLOT)]) for slot in slots()]
    return dict(schema='PI0294_PV1_FINITE_PREFREEZE_POOL_V1', slots=population,
        counts=deepcopy(COUNTS), candidates_per_slot=CANDIDATES_PER_SLOT,
        exclusion_population_sha256=digest(excluded), exclusion_records=len(excluded),
        original_spatial_visual_distributions=True, generator_split_namespace='CONFIRM_CORE',
        source_generator_offsets=dict(OFFSETS), fixed_noise_banks=list(BANKS),
        allowed_rejection_reasons=['GEOMETRIC_NEAR_DUPLICATE', 'ORIGINAL_STATIC_IK_COLLISION_OR_FOV_FAIL'],
        policy_or_Teacher_success_screening=False, new_physics=0, new_queries=0,
        status='POOL_ONLY_NOT_STATIC_QUALIFIED_NOT_CAPTURE_AUTHORITY')


def evaluation_cards(groups, split):
    if split not in ('DEV', 'LOCKBOX'):
        raise PermissionError('PV1_ONLY_FROZEN_EVALUATION_SPLITS')
    selected = [g for g in groups if g['split'] == split]
    wanted = {(family, n) for family, count in COUNTS[split].items() for n in range(count)}
    if (len(selected) != sum(COUNTS[split].values())
            or {(g['family'], int(g['group_id'].rsplit('/', 1)[1])) for g in selected} != wanted):
        raise PermissionError('PV1_COMPLETE_PRESPECIFIED_EVALUATION_POPULATION')
    cards = []
    for group in selected:
        for task in ('T1', 'T2'):
            for bank in BANKS:
                card = task_card(group, task)
                card.update(id=card['id']+'-N'+str(bank), noise_bank=bank,
                    condition_id=group['group_id']+'/'+task+'/N'+str(bank),
                    noise_namespace='PI0294/PV1/'+split+'/N'+str(bank),
                    teacher_takeover=False, policy_candidate='PENDING_FROZEN_CANDIDATE_BINDING')
                cards.append(card)
    return cards
