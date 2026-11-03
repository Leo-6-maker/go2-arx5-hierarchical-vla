"""WM01 task-frame primitive and causal transition checks."""
from __future__ import annotations

import numpy as np
from ..task_data.codec import pose10
from ..target_sim.scorer import pose_matrix
from ..trajectory.so3 import geodesic_angle

FRAME = 'PI0291_MUJOCO_WORLD'
TCP = 'ARX5_NATIVE_SITE_x5_ee'
DT = .05
PREFIX = 8


def exp_rotation(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    cross = np.array([[0., -vector[2], vector[1]], [vector[2], 0., -vector[0]],
                      [-vector[1], vector[0], 0.]])
    a = np.sinc(angle/np.pi)
    b = .5*np.sinc(angle/(2*np.pi))**2
    return np.eye(3)+a*cross+b*cross@cross


def compile_action(raw, anchor, time_s, *, frame=FRAME, tcp=TCP):
    """World-frame increments, consumed at t+i*50ms (native first-point ABI)."""
    raw = np.asarray(raw, dtype=float)
    anchor = pose_matrix(anchor)
    if (raw.shape != (8, 7) or not np.isfinite(raw).all() or not np.isfinite(time_s)
            or frame != FRAME or tcp != TCP):
        raise ValueError('Expected finite 8x7 action and exact native frame/TCP/time')
    if (np.any(np.linalg.norm(raw[:, :3], axis=-1) > .01+1e-12)
            or np.any(np.linalg.norm(raw[:, 3:6], axis=-1) > .05+1e-12)
            or np.any((raw[:, 6] < 0) | (raw[:, 6] > 1))):
        raise ValueError('Primitive bounds exceeded; no label clipping')
    current = anchor.copy()
    poses = []
    for point in raw:
        current[:3, 3] += point[:3]
        current[:3, :3] = exp_rotation(point[3:6])@current[:3, :3]
        poses.append(current.copy())
    poses = np.asarray(poses+[poses[-1].copy() for _ in range(32)])
    opening = np.r_[raw[:, 6], np.full(32, raw[-1, 6])]
    return dict(poses=poses, opening=opening, offset_s=np.arange(40)*DT,
                timestamps_s=float(time_s)+np.arange(40)*DT,
                rows10=np.stack([pose10(p, g) for p, g in zip(poses, opening)]),
                frame=FRAME, tcp=TCP, action_raw=raw.copy(), consumed_prefix=8)


def command_labels(poses, openings, anchor):
    """Invert *consumed commands*, never achieved states or a preview suffix."""
    from scipy.spatial.transform import Rotation
    poses = pose_matrix(np.asarray(poses))
    openings = np.asarray(openings, dtype=float)
    previous = pose_matrix(anchor)
    if poses.shape != (8, 4, 4) or openings.shape != (8,):
        raise ValueError('Exactly eight consumed commands required')
    rows = []
    for pose, opening in zip(poses, openings):
        vector = Rotation.from_matrix(pose[:3, :3]@previous[:3, :3].T).as_rotvec()
        rows.append(np.r_[pose[:3, 3]-previous[:3, 3], vector, opening])
        previous = pose
    raw = np.stack(rows)
    # Return representability separately; retain original commands on failure.
    representable = bool(np.isfinite(raw).all() and
        (np.linalg.norm(raw[:, :3], axis=-1) <= .01+1e-12).all() and
        (np.linalg.norm(raw[:, 3:6], axis=-1) <= .05+1e-12).all() and
        ((raw[:, 6] >= 0) & (raw[:, 6] <= 1)).all())
    return raw, representable


def validate_atom(row):
    required = ('episode_id', 'scene_id', 'parent_id', 'split', 'executor_id',
        'segment_id', 't0', 't1', 'physics_ticks', 'state_before', 'state_after',
        'action_raw', 'reference_installed', 'reference_consumed', 'controller_memory',
        'context', 'terminal', 'truncated', 'safety_reason', 'planned_dt', 'actual_dt', 'frame', 'tcp')
    if any(key not in row for key in required):
        raise ValueError('Missing transition provenance')
    if row['frame'] != FRAME or row['tcp'] != TCP or row['split'] not in ('TRAIN', 'CALIB', 'DEV', 'TEST'):
        raise ValueError('Invalid frame/TCP/split')
    t0, t1 = float(row['t0']), float(row['t1'])
    ticks = row['physics_ticks']
    if (not np.isfinite([t0, t1, row['actual_dt'], row['planned_dt']]).all()
            or type(ticks) is not int or ticks < 0 or ticks > 25 or t1 < t0
            or abs(row['planned_dt']-DT) > 1e-10
            or abs((t1-t0)-row['actual_dt']) > 1e-9
            or abs((t1-t0)-ticks*.002) > 1e-8):
        raise ValueError('Invalid physical clock or tick count')
    if ticks != 25 and not (row['terminal'] or row['truncated']):
        raise ValueError('Incomplete nonterminal interval')
    before, after = row['state_before'], row['state_after']
    if abs(before['time_s']-t0) > 1e-8 or abs(after['time_s']-t1) > 1e-8:
        raise ValueError('State and command clocks differ')
    if before.get('reset') or after.get('reset') or before.get('takeover') or after.get('takeover'):
        raise ValueError('Reset/takeover cannot enter a transition')
    consumed = row['reference_consumed']
    if len(consumed) != ticks:
        raise ValueError('Every actual physics tick needs a consumed command')
    installed = row['reference_installed']
    for i, command in enumerate(consumed):
        if (abs(command['time_s']-(t0+i*.002)) > 1e-8
                or not np.array_equal(command['eef10'], installed)):
            raise ValueError('Consumed command differs from installed command')
    return row


def macro_transition(atoms):
    if not 1 <= len(atoms) <= 8:
        raise ValueError('A macro has one to eight actual atomic intervals')
    for row in atoms:
        validate_atom(row)
    identity = ('episode_id', 'scene_id', 'parent_id', 'split', 'executor_id', 'segment_id', 'frame', 'tcp')
    for left, right in zip(atoms, atoms[1:]):
        if any(left[k] != right[k] for k in identity) or left['terminal'] or left['truncated']:
            raise ValueError('Cross-episode/reset/executor/terminal macro')
        if abs(left['t1']-right['t0']) > 1e-8:
            raise ValueError('Missing physical interval')
        for key in ('qpos', 'qvel', 'tcp', 'object_pose'):
            if not np.array_equal(left['state_after'][key], right['state_before'][key]):
                raise ValueError('State discontinuity')
    complete = len(atoms) == 8 and all(r['physics_ticks'] == 25 for r in atoms)
    if not complete and not (atoms[-1]['terminal'] or atoms[-1]['truncated']):
        raise ValueError('Incomplete macro must retain its terminal/truncation event')
    return dict(**{k: atoms[0][k] for k in identity}, t0=atoms[0]['t0'], t1=atoms[-1]['t1'],
        atoms=atoms, state_before=atoms[0]['state_before'], state_after=atoms[-1]['state_after'],
        complete_state_target=complete, terminal=atoms[-1]['terminal'], truncated=atoms[-1]['truncated'],
        physics_ticks=sum(r['physics_ticks'] for r in atoms))


ACTOR_FIELDS = ('tcp', 'gripper_fraction', 'gripper_speed', 'tool_linear_velocity',
    'tool_angular_velocity', 'object_pose', 'object_linear_velocity', 'object_angular_velocity')


def actor_state(state, goal, remaining_s):
    if not 0 <= remaining_s <= 32 or np.asarray(goal).shape != (3,):
        raise ValueError('Invalid current task context')
    return {**{k: np.asarray(state[k]).copy() for k in ACTOR_FIELDS},
            'goal': np.asarray(goal).copy(), 'remaining_s': float(remaining_s), 'task': 'T1'}
