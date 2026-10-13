"""Pre-register paired layouts; generation alone is not qualification."""
from __future__ import annotations

import copy
import hashlib
import json
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


def scene_group(split, family, index):
    if split not in ('TRAIN', 'DEV', 'COMMISSIONING', 'PILOT', 'CONFIRM_CORE', 'CONFIRM_CHALLENGE'):
        raise ValueError('Unregistered split')
    if family not in ('F0', 'F1', 'F2', 'F3') or index < 0:
        raise ValueError('Invalid scene identity')
    identity = f'PI0294/{split}/{family}/{index:03d}'
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')
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


def build_scene_plan(version='v3.0'):
    counts = FIRST if version == 'v3.0' else EXTENDED if version == 'v3.1' else None
    if counts is None:
        raise ValueError('Unsupported dataset version')
    groups = [scene_group(split, f'F{family}', i) for split, families in counts.items()
              for family, count in enumerate(families) for i in range(count)]
    return dict(schema='PI0294_PAIRED_SCENE_PLAN_V1', version=version, groups=groups,
                cards=[task_card(g, t) for g in groups for t in ('T1', 'T2')],
                qualification='NOT_STATIC_OR_CONTACT_QUALIFIED')


def validate_groups(groups, legacy_groups=(), duplicate_tolerance_m=1e-8):
    """Reject reused identities and exact physical layout/split conflicts."""
    seen = set()
    prior = list(legacy_groups)
    for card in groups:
        if card['group_id'] in seen:
            raise ValueError('Duplicate group identity')
        seen.add(card['group_id'])
        source, goal = np.asarray(card['source_xy']), np.asarray(card['goal_xy'])
        if (not np.isfinite(np.r_[source, goal]).all() or not .15 <= np.linalg.norm(goal-source) <= .22
                or not (.42 <= source[0] <= .48 and .42 <= goal[0] <= .48)
                or max(abs(source[1]), abs(goal[1])) > .14):
            raise ValueError('Candidate outside conservative tabletop bounds')
        for old in prior:
            if (np.linalg.norm(source-np.asarray(old['source_xy'])) <= duplicate_tolerance_m
                    and np.linalg.norm(goal-np.asarray(old['goal_xy'])) <= duplicate_tolerance_m):
                raise ValueError('Duplicate underlying physical layout; keep variants within one group')
        prior.append(card)
    return {'groups': len(seen), 'legacy_layouts_checked': len(legacy_groups), 'integration_steps': 0,
            'ik_and_sweep': 'NOT_RUN', 'camera_visibility': 'NOT_RUN'}
