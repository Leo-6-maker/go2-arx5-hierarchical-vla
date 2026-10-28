"""New ARX5 domain identities, budgets and deterministic layout registration."""
from __future__ import annotations

import hashlib
import numpy as np
from go2arx5_vla.task_data.config import canonical_hash, INSTRUCTION

AUTHORITY = '15cc7957488fb879f5f050e3602327592da7036b'
DOMAIN = 'FIXED_BASE_ARX5_TASK_SIM'
TASK = 'ARX5_FIXED_BASE_T1_V1'
COMMISSIONING = tuple(f'C291{i:02d}' for i in range(12))
PILOTS = tuple(f'P291{i:02d}' for i in range(8))
TRAIN = tuple(f'T291{i:03d}' for i in range(64))
DEV = tuple(f'D291{i:03d}' for i in range(16))
PARAMETERS = {
    'version': 'PI0291_REAL_CHAIN_V1', 'dt': .002, 'duration': 32.,
    'sensor_hz': 20, 'H': 40, 'prefix': 8, 'pixels': 256,
    'table_top': .44, 'cube_edge': .04, 'cube_mass': .05,
    'finger_travel': .044, 'closed_inner_gap': .001048,
    'finger_force_max_N': 8., 'finger_target_speed_m_s': .04,
    'finger_kp': 400., 'finger_kd': 3., 'pad_friction': 1.2,
    'cube_friction': .8, 'table_friction': .8,
    'arm_torque_max_Nm': [27., 27., 27., 7., 7., 7.],
    'arm_kp': [90., 120., 100., 40., 30., 25.],
    'arm_joint_damping': 3.,
    'arm_target_speed_rad_s': .7,
    'arm_speed_stop_rad_s': [5.5, 5.5, 5.5, 20., 20., 20.],
    'reference_speed_m_s': .06, 'reference_angular_speed_rad_s': .3,
    'penetration_stop_m': .004, 'tool_speed_stop_m_s': .4,
    'finger_speed_stop_m_s': .25, 'relative_grasp_speed_max_m_s': .02,
    'opposing_contact_dot_max': -.5, 'minimum_contact_force_N': .05,
    'lift_clearance': .08, 'withdraw_clearance': .13,
    'position_stage_tolerance_m': .004,
}


def layout(identifier):
    if identifier not in COMMISSIONING + PILOTS + TRAIN + DEV:
        raise ValueError('Unregistered ARX5 episode identifier')
    seed = int.from_bytes(hashlib.sha256(('PI0291/' + identifier).encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    split = 'TRAIN' if identifier in TRAIN else 'DEV' if identifier in DEV else 'PILOT' if identifier in PILOTS else 'COMMISSIONING'
    novel = identifier in PILOTS[4:]
    source = np.array([.45, -.08]) + rng.uniform(-.008, .008, 2)
    goal = np.array([.45, .08]) + rng.uniform(-.008, .008, 2)
    if np.linalg.norm(goal-source) < .15:
        goal[1] = source[1] + .155
    if novel:
        source[1], goal[1] = -source[1], -goal[1]
    return dict(id=identifier, seed=seed, group_id='PI0291/' + identifier,
                family='NEW_QUALIFICATION_REVERSED_Y' if novel else 'ARX5_FORWARD_TABLE',
                split=split, source_xy=source.tolist(), goal_xy=goal.tolist(),
                cube_yaw=float(rng.uniform(-.10, .10)), instruction=INSTRUCTION,
                table_top_m=PARAMETERS['table_top'])
