"""New deterministic TRAIN population in the unchanged CR2 geometric envelope."""
from copy import deepcopy
import hashlib
import itertools

import numpy as np

from scripts.pi0294.cr2 import plan as cr2
from scripts.pi0294.scene_cards import digest, task_card

AUTH = '43f7fe6f86c683f2e38e5c8802eb8cde208e575c'
CARD = 'reports/PI0294/design/VAP1_VANDA_POLICY_DATA_TEACHER_SCOPE_20261021.json'
NAMESPACE = 'PI0294/VAP1_DATA/TRAIN'
AP, DIRECTIONS, DISTANCES, R1 = cr2.AP, cr2.DIRECTIONS, cr2.DISTANCES, cr2.R1


def slots():
    return list(itertools.product(range(2), AP, DIRECTIONS, DISTANCES))


def identity(replicate, appearance, direction, distance):
    if (type(replicate) is not int or replicate not in (0, 1)
            or appearance not in AP or direction not in DIRECTIONS or distance not in DISTANCES):
        raise PermissionError('VAP1_EXACT_24_TRAIN_SLOTS')
    return f'{NAMESPACE}/{appearance}/{direction}/{distance}/{replicate:03d}'


def candidate(replicate, appearance, direction, distance, index):
    name = identity(replicate, appearance, direction, distance)
    if type(index) is not int or not 0 <= index < 32:
        raise PermissionError('VAP1_MAX_32_STATIC_CANDIDATES')
    seed_name = f'VAP1_FIXED_R1/V1/{name}/{index}'
    seed = int.from_bytes(hashlib.sha256(seed_name.encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    # Retain CR2's exact declared sampling distributions, not its sampled layouts.
    x, center = float(rng.uniform(.425, .475)), float(rng.uniform(-.025, .025))
    dx, separation = float(rng.uniform(-.012, .012)), float(rng.uniform(*DISTANCES[distance]))
    dy = (1 if direction == '+Y' else -1) * np.sqrt(separation ** 2 - dx ** 2)
    template = cr2.candidate(appearance, direction, distance, 0)
    visual = deepcopy(template['visual'])
    if appearance == 'AP2':
        visual.update(table_rgba=[*rng.uniform(.50, .76, 3).tolist(), 1.],
                      floor_rgba=[*rng.uniform(.54, .78, 3).tolist(), 1.],
                      light_scale=float(rng.uniform(.8, 1.2)))
    group = dict(template, group_id=name, scene_group=name, family='VAP1_DATA',
        replicate=replicate, seed=seed, seed_identity=seed_name, layout_candidate_index=index,
        source_xy=[x, float(center - dy / 2)], goal_xy=[x + dx, float(center + dy / 2)],
        cube_yaw=float(rng.uniform(-.16, .16)), visual=visual, euclidean_distance_m=separation)
    validate_group(group)
    return group


def validate_group(group):
    name = identity(*(group[k] for k in ('replicate', 'appearance_profile', 'direction', 'distance_bin')))
    if group['group_id'] != name or group['scene_group'] != name or group['family'] != 'VAP1_DATA':
        raise PermissionError('VAP1_NEW_NAMESPACE_NOT_CR2_REOPEN')
    # Only the identity changes. CR2 still enforces all geometry/camera/scorer rules.
    prior = dict(group, group_id=cr2.identity(group['appearance_profile'], group['direction'], group['distance_bin']),
                 scene_group=cr2.identity(group['appearance_profile'], group['direction'], group['distance_bin']),
                 family='CR2_FIXED_R1', replicate=0)
    cr2.validate_group(prior)


def pool(exclusions):
    return dict(schema='PI0294_VAP1_FINITE_POOL_V1', authority_release=AUTH,
        namespace=NAMESPACE, slots=[dict(slot=identity(*slot),
            candidates=[candidate(*slot, i) for i in range(32)]) for slot in slots()],
        exclusion_sha256=digest(exclusions), task_results_seen=False, physics=0, new_range_fits=0)


def cards(groups):
    if len(groups) != 24 or {g['group_id'] for g in groups} != {identity(*s) for s in slots()}:
        raise PermissionError('VAP1_ALL_24_PAIRED_GROUPS_SEALED_BEFORE_RESULTS')
    result = []
    for group in sorted(groups, key=lambda g: (g['replicate'], g['group_id'])):
        validate_group(group)
        for task in ('T1', 'T2'):
            card = task_card(group, task)
            card.update(id=card['id'] + '-C04', nominal_clearance_m=.004,
                        origin='VAP1_NEW_TRAIN_ORIGINAL_CR2_TEACHER_NOT_POLICY')
            result.append(card)
    return result


def bank_gate(rows, groups, *, cumulative=False):
    planned = [c for c in cards(groups) if cumulative or c['replicate'] == 0]
    indexed = {c['id']: c for c in planned}
    if len({r['id'] for r in rows}) != len(rows) or any(r['id'] not in indexed for r in rows):
        raise PermissionError('VAP1_FIRST_ATTEMPTS_ONLY_NO_REPLACEMENT_OR_BATCH_LEAK')
    by_task = lambda key: {t: sum(r.get(key) is True for r in rows if indexed[r['id']]['task'] == t)
                           for t in ('T1', 'T2')}
    success, admitted = by_task('task_success'), by_task('admitted')
    used_groups = [g for g in groups if cumulative or g['replicate'] == 0]
    good = {r['id'] for r in rows if r.get('admitted') is True}
    pairs = sum(all(c['id'] in good for c in planned if c['group_id'] == g['group_id']) for g in used_groups)
    coverage = {f'{ap}/{direction}/{task}': sum(c['id'] in good for c in planned
        if (c['appearance_profile'], c['direction'], c['task']) == (ap, direction, task))
        for ap, direction, task in itertools.product(AP, DIRECTIONS, ('T1', 'T2'))}
    distance = {f'{d}/{t}': sum(c['id'] in good for c in planned if (c['distance_bin'], c['task']) == (d, t))
                for d, t in itertools.product(DISTANCES, ('T1', 'T2'))}
    safe = all(r.get('native_faults') == 0 and r.get('unknown_effects') == []
               and r.get('R0_PASS_to_R1_FAIL') is False and r.get('evidence_qualified') is True for r in rows)
    passed = (len(rows) == len(planned) and min(success.values()) >= (20 if cumulative else 10)
              and min(admitted.values()) >= (20 if cumulative else 10) and pairs >= (18 if cumulative else 9)
              and min(coverage.values()) >= (2 if cumulative else 1)
              and (not cumulative or min(distance.values()) >= 9) and safe)
    return dict(status='PASS_VAP1_DATA_BANK_GATE' if passed else 'NOT_READY',
        successes=success, admitted=admitted, paired_groups=pairs, coverage=coverage, distance=distance,
        planned=len(planned), accounted=len(rows), safety_and_no_regression=safe,
        cumulative=cumulative, policy_success_claim=False)
