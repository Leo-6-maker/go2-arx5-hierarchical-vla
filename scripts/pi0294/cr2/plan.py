"""Twelve pre-result TRAIN slots using the existing COV1-v2 geometric envelope."""
from copy import deepcopy
import hashlib
import itertools

import numpy as np

from scripts.pi0294.data_cov1.plan import preflight_score_geometry
from scripts.pi0294.scene_cards import digest, scene_group, task_card

AUTH = '5d1393154054b977134b7d8c456f66ca478d0bf9'
GO = 'HUMAN_REPLY_call_656187cffb8b43519e678e7561b78575_FULL_CR2_FIXED_R1'
CARD = 'reports/PI0294/design/CR2_FIXED_R1_BANK_PILOT_SCOPE_20261020.json'
NAMESPACE = 'PI0294/CR2_FIXED_R1/TRAIN'
R1 = '50d9672397fd0b5353da2011b9f9f568dc9c65c26fb271458f87d60b2d497fa8'
AP = ('AP0', 'AP1', 'AP2')
DIRECTIONS = ('+Y', '-Y')
DISTANCES = dict(NEAR=(.15, .18), FAR=(.19, .22))


def identity(appearance, direction, distance):
    if appearance not in AP or direction not in DIRECTIONS or distance not in DISTANCES:
        raise PermissionError('CR2_ONLY_FIXED_TWELVE_TRAIN_SLOTS')
    return f'{NAMESPACE}/{appearance}/{direction}/{distance}/000'


def candidate(appearance, direction, distance, index):
    name = identity(appearance, direction, distance)
    if type(index) is not int or not 0 <= index < 32:
        raise PermissionError('CR2_MAX_32_STATIC_CANDIDATES_NO_RESULT_REDRAWS')
    seed_name = f'CR2_FIXED_R1/V1/{name}/{index}'
    seed = int.from_bytes(hashlib.sha256(seed_name.encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    x, center = float(rng.uniform(.425, .475)), float(rng.uniform(-.025, .025))
    dx, separation = float(rng.uniform(-.012, .012)), float(rng.uniform(*DISTANCES[distance]))
    dy = (1 if direction == '+Y' else -1) * np.sqrt(separation ** 2 - dx ** 2)
    visual = deepcopy(scene_group('PILOT', 'F0', 0)['visual'])
    if appearance == 'AP1':
        visual.update(table_rgba=[.60, .62, .66, 1.], floor_rgba=[.70, .68, .65, 1.], light_scale=1.)
    elif appearance == 'AP2':
        visual.update(table_rgba=[*rng.uniform(.50, .76, 3).tolist(), 1.],
                      floor_rgba=[*rng.uniform(.54, .78, 3).tolist(), 1.], light_scale=float(rng.uniform(.8, 1.2)))
    group = dict(group_id=name, scene_group=name, split='TRAIN', family='CR2_FIXED_R1',
        appearance_profile=appearance, direction=direction, distance_bin=distance, replicate=0,
        seed=seed, seed_identity=seed_name, layout_candidate_index=index,
        source_xy=[x, float(center - dy / 2)], goal_xy=[x + dx, float(center + dy / 2)],
        cube_yaw=float(rng.uniform(-.16, .16)), table_top_m=scene_group('PILOT', 'F0', 0)['table_top_m'],
        visual=visual, euclidean_distance_m=separation, qualification='NOT_STATIC_OR_DYNAMIC_QUALIFIED')
    validate_group(group)
    return group


def validate_group(group):
    keys = ('appearance_profile', 'direction', 'distance_bin')
    name = identity(*(group[k] for k in keys))
    index = group['layout_candidate_index']
    if (group['group_id'] != name or group['scene_group'] != name or group['split'] != 'TRAIN'
            or group['family'] != 'CR2_FIXED_R1' or group['replicate'] != 0
            or type(index) is not int or not 0 <= index < 32):
        raise PermissionError('CR2_NEW_EXACT_TRAIN_NAMESPACE_NOT_COV1_OR_PV1')
    source, goal = np.asarray(group['source_xy']), np.asarray(group['goal_xy'])
    if source.shape != (2,) or goal.shape != (2,) or not np.isfinite(np.r_[source, goal, group['cube_yaw']]).all():
        raise ValueError('CR2_FINITE_PLANAR_GEOMETRY')
    distance = float(np.linalg.norm(goal - source)); low, high = DISTANCES[group['distance_bin']]
    if (not low <= distance <= high or abs(distance - group['euclidean_distance_m']) > 1e-12
            or not .425 <= source[0] <= .475 or abs(goal[0] - source[0]) > .012 + 1e-12
            or abs((source[1] + goal[1]) / 2) > .025 + 1e-12 or abs(group['cube_yaw']) > .16
            or (goal[1] > source[1]) != (group['direction'] == '+Y')):
        raise ValueError('CR2_FROZEN_COV1_V2_EUCLIDEAN_ENVELOPE')
    for camera, suffix in itertools.product(('body', 'wrist'), ('_camera_translation', '_camera_rpy')):
        if group['visual'][camera + suffix] != [0., 0., 0.]:
            raise PermissionError('CR2_FROZEN_CAMERA_CALIBRATION')
    preflight_score_geometry(group)


def slots():
    return list(itertools.product(AP, DIRECTIONS, DISTANCES))


def pool(exclusions):
    return dict(schema='PI0294_CR2_FINITE_POOL_V1', authority_release=AUTH,
        namespace=NAMESPACE, slots=[dict(slot=identity(*slot), candidates=[candidate(*slot, i) for i in range(32)])
                                   for slot in slots()], exclusion_sha256=digest(exclusions),
        task_results_seen=False, physics=0, new_range_fits=0)


def cards(groups):
    expected = {identity(*slot) for slot in slots()}
    if len(groups) != 12 or {g['group_id'] for g in groups} != expected:
        raise PermissionError('CR2_ALL_TWELVE_PAIRED_GROUPS_REQUIRED')
    first = [identity('AP0', '-Y', 'NEAR'), identity('AP2', '-Y', 'FAR')]
    ordered = sorted(groups, key=lambda g: (first.index(g['group_id']) if g['group_id'] in first else 2, g['group_id']))
    result = []
    for group in ordered:
        validate_group(group)
        for task in ('T1', 'T2'):
            card = task_card(group, task)
            card.update(id=card['id'] + '-C04', nominal_clearance_m=.004,
                        origin='CR2_FIXED_R1_NEW_TRAIN_ORIGINAL_TEACHER_NOT_POLICY')
            result.append(card)
    return result


def bank_gate(rows, groups):
    planned = cards(groups); indexed = {c['id']: c for c in planned}
    if len({r['id'] for r in rows}) != len(rows) or any(r['id'] not in indexed for r in rows):
        raise PermissionError('CR2_EACH_ATTEMPT_ONCE_NO_REPLACEMENTS')
    success = {task: sum(bool(r.get('task_success')) for r in rows if indexed[r['id']]['task'] == task) for task in ('T1', 'T2')}
    admitted = {task: sum(bool(r.get('admitted')) for r in rows if indexed[r['id']]['task'] == task) for task in ('T1', 'T2')}
    pairs = sum(all(any(r['id'] == c['id'] and r.get('admitted') for r in rows)
                    for c in planned if c['group_id'] == g['group_id']) for g in groups)
    coverage = {f'{ap}/{direction}/{task}': sum(bool(r.get('admitted')) for r in rows
        if (indexed[r['id']]['appearance_profile'], indexed[r['id']]['direction'], indexed[r['id']]['task']) == (ap, direction, task))
        for ap, direction, task in itertools.product(AP, DIRECTIONS, ('T1', 'T2'))}
    safe = all(not r.get('native_faults') and not r.get('unknown_effects') and not r.get('R0_PASS_to_R1_FAIL') for r in rows)
    passed = (len(rows) == 24 and min(success.values()) >= 10 and min(admitted.values()) >= 10
              and pairs >= 9 and min(coverage.values()) >= 1 and safe
              and all(r.get('evidence_qualified') for r in rows))
    return dict(status='PASS_CR2_DATA_BANK_GATE' if passed else 'NOT_READY', successes=success,
                admitted=admitted, paired_groups=pairs, coverage=coverage, planned=24, accounted=len(rows),
                safety_and_no_regression=safe, policy_success_claim=False)
