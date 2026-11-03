"""Typed privileged WM state. Actor observations use a separate whitelist."""
from __future__ import annotations
import numpy as np

FIELDS = (('tcp_position_m', 3, 'position'), ('object_position_m', 3, 'position'),
    ('tcp_rotation', 9, 'rotation'), ('object_rotation', 9, 'rotation'),
    ('arm_q_rad', 6, 'joint'), ('finger_q_m', 2, 'joint'),
    ('qvel', 14, 'velocity'), ('tool_velocity', 6, 'velocity'),
    ('object_velocity', 6, 'velocity'), ('arm_targets_rad', 6, 'controller'),
    ('finger_targets_m', 2, 'controller'), ('qacc_warmstart', 14, 'solver'),
    ('ctrl', 8, 'solver'), ('ik_target_position_m', 3, 'controller'),
    ('ik_target_rotation', 9, 'rotation'), ('cached_ik_solution', 6, 'controller'),
    ('cached_ik_error', 2, 'controller'), ('ik_cache_valid', 1, 'controller'),
    ('time_s', 1, 'time'))
SLICES = {}
_cursor = 0
for _name, _width, _group in FIELDS:
    SLICES[_name] = slice(_cursor, _cursor+_width)
    _cursor += _width
STATE_DIM = _cursor
ROTATIONS = tuple(SLICES[n] for n, _, g in FIELDS if g == 'rotation')
EVENTS = ('left_object_contact', 'right_object_contact', 'native_grasp', 'terminal')


def encode(state, memory, native):
    """Native integration layout explicitly checked against independent arrays.

    Bound to MuJoCo3.2.7, nq15/nv14/na0: time,qpos,qvel,warmstart,ctrl,
    then applied forces and other integration inputs, all zero in this driver.
    Object orientation uses SO(3); the freejoint quaternion is never an MSE target.
    """
    integration = np.asarray(native['integration'])
    if (integration.shape != (234,) or np.asarray(state['qpos']).shape != (15,)
            or np.asarray(state['qvel']).shape != (14,)
            or not np.array_equal(integration[1:16], state['qpos'])
            or not np.array_equal(integration[16:30], state['qvel'])
            or abs(integration[0]-state['time_s']) > 1e-10
            or np.any(integration[52:] != 0)):
        raise ValueError('Unbound native integration layout/nonzero exogenous state')
    valid = memory['_ik_target'] is not None
    target = np.asarray(memory['_ik_target']) if valid else np.eye(4)
    values = dict(tcp_position_m=state['tcp'][:3, 3], object_position_m=state['object_pose'][:3, 3],
        tcp_rotation=state['tcp'][:3, :3].ravel(), object_rotation=state['object_pose'][:3, :3].ravel(),
        arm_q_rad=state['qpos'][:6], finger_q_m=state['qpos'][6:8], qvel=state['qvel'],
        tool_velocity=np.r_[state['tool_linear_velocity'], state['tool_angular_velocity']],
        object_velocity=np.r_[state['object_linear_velocity'], state['object_angular_velocity']],
        arm_targets_rad=memory['arm_targets'], finger_targets_m=memory['finger_targets'],
        qacc_warmstart=integration[30:44], ctrl=integration[44:52],
        ik_target_position_m=target[:3, 3], ik_target_rotation=target[:3, :3].ravel(),
        cached_ik_solution=memory['_cached_ik_solution'] if valid else np.zeros(6),
        cached_ik_error=memory['_cached_ik_error'] if valid else np.zeros(2),
        ik_cache_valid=[float(valid)], time_s=[state['time_s']])
    value = np.concatenate([np.asarray(values[n]).ravel() for n, _, _ in FIELDS]).astype(np.float32)
    if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
        raise ValueError('Invalid privileged state')
    return value


def macro_arrays(macro):
    if not macro['complete_state_target']:
        raise ValueError('Partial safety event cannot become a full-state target')
    first, last = macro['atoms'][0], macro['atoms'][-1]
    x = encode(first['state_before'], first['controller_memory'], first['native_before'])
    y = encode(last['state_after'], last['controller_memory_after'], last['native_after'])
    action = np.stack([a['action_raw'] for a in macro['atoms']]).astype(np.float32).ravel()
    if action.shape != (56,):
        raise ValueError('Exactly eight actual commands required')
    events = np.asarray([last['state_after'].get(k, macro['terminal']) if k != 'terminal' else macro['terminal']
                         for k in EVENTS], dtype=np.float32)
    return x, action, y, events
