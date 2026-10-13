"""Pre-register paired layouts; generation alone is not qualification."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.pi0291_arx5_data.config import PARAMETERS as P

INSTRUCTIONS = {
    'T1': 'Pick up the red cube, place it inside the blue square, release it, and move the gripper away.',
    'T2': 'Push the red cube along the table into the blue square without lifting it, then move the gripper away.',
}
FIRST = {'TRAIN': (8, 12, 12, 0), 'DEV': (2, 3, 3, 0)}
EXTENDED = {'TRAIN': (16, 32, 32, 16), 'DEV': (4, 4, 4, 4)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def scene_group(split, family, index, *, candidate_index=0):
    if split not in ('TRAIN', 'DEV', 'COMMISSIONING', 'PILOT', 'CONFIRM_CORE', 'CONFIRM_CHALLENGE'):
        raise ValueError('Unregistered split')
    if family not in ('F0', 'F1', 'F2', 'F3') or index < 0:
        raise ValueError('Invalid scene identity')
    identity = f'PI0294/{split}/{family}/{index:03d}'
    if not isinstance(candidate_index, int) or not 0 <= candidate_index < 1024:
        raise ValueError('Invalid pre-freeze layout candidate index')
    seed_identity = identity if candidate_index == 0 else identity+f'/candidate/{candidate_index}'
    seed = int.from_bytes(hashlib.sha256(seed_identity.encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    spatial = family in ('F1', 'F3')
    x = float(rng.uniform(.435, .465) if spatial else rng.uniform(.442, .458))
    gap = float(rng.uniform(.16, .20) if spatial else rng.uniform(.155, .17))
    center_y = float(rng.uniform(-.012, .012))
    sign = -1 if spatial and index % 2 else 1
    source = [x, center_y-sign*gap/2]
    goal = [float(x+rng.uniform(-.006, .006)), center_y+sign*gap/2]
    visual = dict(table_rgba=[.69, .70, .66, 1.], floor_rgba=[.72, .74, .76, 1.], light_scale=1.,
                  body_camera_translation=[0., 0., 0.], body_camera_rpy=[0., 0., 0.],
                  wrist_camera_translation=[0., 0., 0.], wrist_camera_rpy=[0., 0., 0.])
    if family in ('F2', 'F3'):
        visual.update(table_rgba=[*rng.uniform(.50, .76, 3).tolist(), 1.],
                      floor_rgba=[*rng.uniform(.54, .78, 3).tolist(), 1.], light_scale=float(rng.uniform(.8, 1.2)))
    if family == 'F3':
        for camera in ('body', 'wrist'):
            visual[camera+'_camera_translation'] = rng.uniform(-.01, .01, 3).tolist()
            visual[camera+'_camera_rpy'] = rng.uniform(-np.pi/60, np.pi/60, 3).tolist()
    return dict(group_id=identity, scene_group=identity, seed=seed, split=split, family=family,
                source_xy=source, goal_xy=goal, cube_yaw=float(rng.uniform(-.08, .08)),
                table_top_m=P['table_top'], visual=visual, qualification='NOT_STATIC_OR_CONTACT_QUALIFIED')


def task_card(group, task):
    if task not in INSTRUCTIONS:
        raise ValueError('Only T1/T2 are registered')
    result = copy.deepcopy(group)
    result.update(id=group['group_id'].replace('/', '-')+'-'+task, task=task,
                  task_id='ARX5_FIXED_BASE_'+('PICK_PLACE' if task == 'T1' else 'PUSH')+'_V1',
                  instruction=INSTRUCTIONS[task], mode='normal', domain='FIXED_BASE_ARX5_TASK_SIM')
    result['initial_state_sha256'] = digest(group)
    return result


def legacy_layouts():
    """Read fixed legacy design cards, including the never-training confirm set."""
    repository = Path(__file__).resolve().parents[2]
    sources = ('reports/PI0291/20261013T023717-localpc-arx5/TEACHER_AND_LAYOUT_SEAL.json',
               'reports/PI0292/design/CONFIRM_LAYOUTS.json')
    rows, hashes = [], {}
    for relative in sources:
        raw = (repository/relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        layouts = json.loads(raw)['layouts']
        rows.extend(normalise_layouts(layouts))
    return rows, hashes


def normalise_layouts(population):
    """Accept frozen layouts, actual CARDs and reservation/attempt card wrappers."""
    if isinstance(population, dict):
        if 'source_xy' in population or 'card' in population:
            population = [population]
        else:
            population = list(population.values())
    result = []
    for row in population:
        card = row.get('card', row)
        if not isinstance(card, dict) or not all(k in card for k in ('source_xy', 'goal_xy', 'cube_yaw', 'split')):
            raise ValueError('Legacy comparison requires exact saved geometric CARD fields')
        identity = card.get('group_id') or card.get('scene_group') or card.get('id')
        if not identity:
            raise ValueError('Legacy layout identity missing')
        result.append(dict(group_id=identity, split=card['split'], source_xy=list(card['source_xy']),
                           goal_xy=list(card['goal_xy']), cube_yaw=float(card['cube_yaw'])))
    return result


def build_scene_plan(version='v3.0', *, legacy_groups=None, preserved_groups=()):
    counts = FIRST if version == 'v3.0' else EXTENDED if version == 'v3.1' else None
    if counts is None:
        raise ValueError('Unsupported dataset version')
    if legacy_groups is None:
        legacy_groups, legacy_hashes = legacy_layouts()
    else:
        legacy_groups = normalise_layouts(legacy_groups)
        legacy_hashes = {'explicit_layout_population': digest(legacy_groups)}
    preserved_groups = normalise_layouts(preserved_groups)
    exclusions = [*legacy_groups, *preserved_groups]
    # Generate the same master population for both versions before any rollout.
    identities = [(split, f'F{family}', i) for split, families in FIRST.items()
                  for family, count in enumerate(families) for i in range(count)]
    identities.extend((split, f'F{family}', i) for split, families in EXTENDED.items()
                      for family, count in enumerate(families) for i in range(FIRST[split][family], count))
    identities.extend((split, family, i) for split, families in
                      (('CONFIRM_CORE', (('F0', 2), ('F1', 3), ('F2', 3))),
                       ('CONFIRM_CHALLENGE', (('F3', 8),))) for family, count in families for i in range(count))
    accepted, rejected = [], []
    for split, family, index in identities:
        for candidate in range(1024):
            group = scene_group(split, family, index, candidate_index=candidate)
            report = near_duplicate_report([group], [*exclusions, *accepted])
            if report['conflicts']:
                rejected.append(dict(group_id=group['group_id'], candidate_index=candidate,
                                     seed=group['seed'], conflicts=report['conflicts']))
                continue
            group['layout_candidate_index'] = candidate
            accepted.append(group)
            break
        else:
            raise RuntimeError('Pre-freeze candidate generation exhausted; no physical retry authorized')
    groups = [g for g in accepted if g['split'] in counts
              and int(g['group_id'].rsplit('/', 1)[1]) < counts[g['split']][int(g['family'][1])]]
    confirmation = [g for g in accepted if g['split'].startswith('CONFIRM_')]
    uniqueness = validate_groups(accepted, exclusions)
    return dict(schema='PI0294_PAIRED_SCENE_PLAN_V1', version=version, groups=groups,
                cards=[task_card(g, t) for g in groups for t in ('T1', 'T2')],
                confirmation_groups=confirmation,
                confirmation_cards=[task_card(g, t) for g in confirmation for t in ('T1', 'T2')],
                legacy_sources=legacy_hashes, preserved_groups_sha256=digest(list(preserved_groups)),
                prefreeze_rejected_candidates=rejected, geometric_uniqueness=uniqueness,
                generation='DETERMINISTIC_MASTER_POPULATION_PREFREEZE_EXCLUSIONS_NOT_PHYSICAL_ATTEMPT_REPLACEMENT',
                qualification='GEOMETRIC_UNIQUENESS_CHECKED_NOT_STATIC_IK_OR_CONTACT_QUALIFIED')


def near_duplicate_report(groups, legacy_groups=(), position_tolerance_m=.002, yaw_tolerance_rad=np.pi/90):
    """Visual changes never hide matching underlying manipulation layouts."""
    if position_tolerance_m <= 1e-8 or yaw_tolerance_rad <= 0:
        raise ValueError('Explicit physical near-duplicate bounds required')
    prior = [(c, True) for c in legacy_groups]
    conflicts = []
    for card in groups:
        for old, legacy in prior:
            if card['group_id'] == old.get('group_id') and not legacy:
                continue
            source_distance = float(np.linalg.norm(np.asarray(card['source_xy'])-old['source_xy']))
            goal_distance = float(np.linalg.norm(np.asarray(card['goal_xy'])-old['goal_xy']))
            yaw_known = 'cube_yaw' in old and 'cube_yaw' in card
            yaw_distance = float(abs((card.get('cube_yaw', 0.)-old.get('cube_yaw', 0.)+np.pi) % (2*np.pi)-np.pi)) if yaw_known else None
            if (max(source_distance, goal_distance) <= position_tolerance_m
                    and (not yaw_known or yaw_distance <= yaw_tolerance_rad)):
                conflicts.append(dict(group_id=card['group_id'], other_group_id=old.get('group_id', old.get('id')),
                    split=card['split'], other_split=old.get('split'), source_distance_m=source_distance,
                    goal_distance_m=goal_distance, yaw_distance_rad=yaw_distance, legacy=legacy,
                    disposition='CROSS_SPLIT_LEAKAGE' if card['split'] != old.get('split') else 'SAME_SPLIT_REQUIRES_ONE_GROUP'))
        prior.append((card, False))
    return dict(position_tolerance_m=position_tolerance_m, yaw_tolerance_rad=float(yaw_tolerance_rad),
                comparison='SOURCE_AND_TARGET_XY_AND_CUBE_YAW_IGNORING_VISUAL_AND_FAMILY',
                conflicts=conflicts, legacy_layouts_checked=len(legacy_groups), integration_steps=0)


def validate_groups(groups, legacy_groups=(), duplicate_tolerance_m=.002):
    """Reject reused identities and physical near duplicates across group IDs."""
    seen = set()
    for card in groups:
        if card['group_id'] in seen:
            raise ValueError('Duplicate group identity')
        seen.add(card['group_id'])
        source, goal = np.asarray(card['source_xy']), np.asarray(card['goal_xy'])
        if (not np.isfinite(np.r_[source, goal]).all() or not .15 <= np.linalg.norm(goal-source) <= .22
                or not (.42 <= source[0] <= .48 and .42 <= goal[0] <= .48)
                or max(abs(source[1]), abs(goal[1])) > .14):
            raise ValueError('Candidate outside conservative tabletop bounds')
    report = near_duplicate_report(groups, legacy_groups, position_tolerance_m=duplicate_tolerance_m)
    if report['conflicts']:
        conflict = report['conflicts'][0]
        raise ValueError(f"Near duplicate physical layouts: {conflict['group_id']} and {conflict['other_group_id']}; {conflict['disposition']}")
    return {'groups': len(seen), 'legacy_layouts_checked': len(legacy_groups), 'integration_steps': 0,
            'near_duplicates': report, 'ik_and_sweep': 'NOT_RUN', 'camera_visibility': 'NOT_RUN'}
