from __future__ import annotations

import hashlib
import json
import numpy as np

AUTHORITY = 'e9918c68d03cc9a1049aa33e2bd2b0ef4eaea398'
CANONICAL = 6127997973
TASK = 'INDEP_MJ_PLACE_V1'
DOMAIN = 'INDEPENDENT_DYNAMIC_TOOL_SIM_ONLY_NON_REALTIME'
INSTRUCTION = 'Pick up the red cube, place it inside the blue square, release it, and move the gripper away.'
COMMISSIONING = tuple(f'C290{i}' for i in range(4))
PILOTS = tuple(f'P290{i}' for i in range(8))
PARAMETERS = {
    'version': 'PI0290_DYNAMIC_TOOL_V1', 'dt': .002, 'duration': 32.,
    'sensor_hz': 20, 'H': 40, 'prefix': 8, 'pixels': 256,
    'table_top': .4, 'cube_edge': .04, 'cube_mass': .05,
    'finger_travel': .044, 'closed_inner_gap': .007,
    'finger_mass': .06, 'tool_mass': .36,
    'finger_force_max_N': 8., 'finger_target_speed_m_s': .04,
    'finger_kp': 400., 'finger_kd': 3.,
    'tool_force_max_N': 40., 'tool_torque_max_Nm': 2.,
    'tool_position_kp': 650., 'tool_position_kd': 36.,
    'tool_rotation_kp': 15., 'tool_rotation_kd': 1.2,
    'reference_speed_m_s': .06, 'reference_angular_speed_rad_s': .3,
    'pad_friction': 1.2, 'cube_friction': .8, 'table_friction': .8,
    'penetration_stop_m': .004, 'tool_speed_stop_m_s': .4,
    'finger_speed_stop_m_s': .25, 'relative_grasp_speed_max_m_s': .02,
    'opposing_contact_dot_max': -.5, 'minimum_contact_force_N': .05,
    'lift_clearance': .08, 'withdraw_clearance': .13,
    'position_stage_tolerance_m': .004,
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def layout(identifier):
    if identifier not in COMMISSIONING + PILOTS:
        raise ValueError('Unregistered episode identifier')
    seed = int.from_bytes(hashlib.sha256(identifier.encode('ascii')).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    heldout = identifier in PILOTS[4:]
    source = np.array([-.11, -.07]) + rng.uniform(-.025, .025, 2)
    goal = np.array([.11, .07]) + rng.uniform(-.025, .025, 2)
    yaw = float(rng.uniform(-.12, .12))
    if heldout:
        source[1] *= -1
        goal[1] *= -1
        yaw = float(rng.uniform(.18, .30)) * (-1 if int(identifier[-1]) % 2 else 1)
    value = dict(id=identifier, seed=seed, family='HELDOUT_REVERSED_Y_AND_YAW' if heldout else 'SEEN_DIAGONAL',
                 split='HELDOUT' if heldout else ('SEEN' if identifier in PILOTS else 'COMMISSIONING'),
                 source_xy=source.tolist(), goal_xy=goal.tolist(), cube_yaw=yaw, instruction=INSTRUCTION)
    if np.linalg.norm(goal-source) < .15 or np.max(np.abs(np.r_[source, goal])) > .25:
        raise ValueError('Frozen layout violates geometric precheck')
    return value
