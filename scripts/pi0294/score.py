"""Independent saved-state scoring; no teacher phase or command inputs."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from go2arx5_vla.target_sim.scorer import (
    BOOL_FIELDS, boolean_array, finite_array, physical_grasp, pose_matrix, score_episode, sustained,
)
from go2arx5_vla.task_data.io import file_hash, write


def score(data, card, complete=True, table_top=None):
    table = P['table_top'] if table_top is None else table_top
    if table != P['table_top']:
        raise ValueError('Frozen table height changed')
    if card['task'] == 'T1':
        result = score_episode(data, goal_center=[*card['goal_xy'], table+.02],
                               initial_object_pose=data['object_pose'][0], table_height_m=table,
                               complete=complete, evidence_source='REAL_MUJOCO_FIXED_BASE_ARX5')
    elif card['task'] == 'T2':
        result = score_push(data, card, complete=complete)
    else:
        raise ValueError('Unregistered task')
    result.update(task=card['task'], task_id=card['task_id'], domain='FIXED_BASE_ARX5_TASK_SIM',
                  maximum_actual_lift_m=float(np.max(data['object_pose'][:, 2, 3]-data['object_pose'][0, 2, 3])),
                  maximum_actual_carry_xy_m=float(np.max(np.linalg.norm(data['object_pose'][:, :2, 3]-data['object_pose'][0, :2, 3], axis=1))))
    return result


def score_push(data, card, complete=True):
    times = np.asarray(data['time_s'], dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid physical clock')
    n = len(times)
    obj = pose_matrix(finite_array(data['object_pose'], (n, 4, 4), 'object_pose'))
    tcp = pose_matrix(finite_array(data['tcp_world'], (n, 4, 4), 'tcp_world'))
    linear = finite_array(data['object_linear_velocity'], (n, 3), 'object_linear_velocity')
    angular = finite_array(data['object_angular_velocity'], (n, 3), 'object_angular_velocity')
    flags = {k: boolean_array(data[k], (n,), k) for k in BOOL_FIELDS}
    contact = boolean_array(data['robot_object_contact'], (n,), 'robot_object_contact')
    grasp = physical_grasp(flags)
    xyz = obj[:, :3, 3]
    displacement = np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1)
    rise = xyz[:, 2]-xyz[0, 2]
    corners = np.einsum('nij,kj->nki', obj[:, :3, :3], np.asarray(list(itertools.product((-.02, .02), repeat=3))))+xyz[:, None, :]
    predicates = dict(zone_corners=(np.abs(corners[:, :, :2]-np.asarray(card['goal_xy'])) <= .07).all(axis=(1, 2)),
                      table_height=np.abs(xyz[:, 2]-(P['table_top']+.02)) <= .005,
                      linear_speed=np.linalg.norm(linear, axis=1) <= .02,
                      angular_speed=np.linalg.norm(angular, axis=1) <= .10,
                      released=~grasp & ~flags['native_grasp'], robot_separated=~contact,
                      tcp_clear=np.linalg.norm(tcp[:, :3, 3]-xyz, axis=1) >= .08,
                      object_and_contacts_valid=flags['object_present'] & flags['contact_valid'] & flags['gripper_state_valid'])
    hold_valid = np.logical_and.reduce(list(predicates.values()))
    hold_indices = [int(m[0]) for t in np.arange(560, 601)/20 for m in [np.flatnonzero(np.abs(times-t) <= 1e-8)] if len(m) == 1]
    clock = bool(n == 641 and np.all(np.abs(times-np.arange(641)/20) <= 1e-8))
    safe = bool(flags['safety_valid'].all() and not any(flags[k].any() for k in
                ('native_abort', 'illegal_contact', 'joint_limit_penetration', 'reset', 'takeover')))
    # High-rate evidence catches brief grasp/lift events between camera frames.
    contact_clock = np.asarray(data.get('contact_time_s', []), dtype=float)
    micro = np.asarray(data.get('contact_audit', []), dtype=float)
    highrate_valid = bool(contact_clock.shape == (16000,) and micro.shape == (16000, 5)
                          and np.isfinite(micro).all() and np.all(np.abs(contact_clock-np.arange(1, 16001)*.002) <= 1e-8))
    if highrate_valid:
        max_rise = max(float(rise.max()), float(micro[:, 3].max()-xyz[0, 2]))
        actual_contact = bool(contact.any() or (micro[:, 0] > 0).any())
        max_grasp_s = maximum_run_duration(contact_clock, micro[:, 1].astype(bool), .002)
        highrate_safe = bool((micro[:, 4] == 0).all())
    else:
        max_rise, actual_contact, max_grasp_s, highrate_safe = float(rise.max()), bool(contact.any()), 0., False
    history = dict(actual_robot_contact=actual_contact, horizontal_push=bool(displacement.max() >= .10),
                   no_lift=bool(max_rise <= .005+1e-8), no_sustained_bilateral_grasp=bool(max_grasp_s < .3-1e-8 and not sustained(times, grasp)),
                   highrate_contact_complete=highrate_valid)
    full = bool(complete and clock and safe and highrate_safe)
    passed = int(hold_valid[hold_indices].sum())
    return dict(evidence_source='REAL_MUJOCO_FIXED_BASE_ARX5', complete_32s=full, clock_valid=clock,
                safety_valid=safe and highrate_safe, history=history, hold_observed=len(hold_indices), hold_pass=passed,
                hold_subpredicates={k: int(v[hold_indices].sum()) for k, v in predicates.items()},
                maximum_center_height_rise_m=max_rise, maximum_bilateral_grasp_s=max_grasp_s,
                contact_height_numerical_tolerance_m=1e-8,
                persistent_success=bool(full and all(history.values()) and len(hold_indices) == 41 and passed == 41),
                selected='NONE', deployment='NONE')


def maximum_run_duration(times, flags, dt):
    longest, start, previous = 0., None, None
    for t, flag in zip(times, flags):
        if not flag or previous is not None and t-previous > dt+1e-8:
            start = None
        if flag:
            start = t if start is None else start
            longest = max(longest, float(t-start+dt))
        previous = t
    return longest


def saved_score(folder):
    root = Path(folder)
    card = json.loads((root/'CARD.json').read_text())
    with np.load(root/'physics.npz', allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    result = score(arrays, card, complete=len(arrays['time_s']) == 641)
    result.update(physics_sha256=file_hash(root/'physics.npz'), raw_sample_count=len(arrays['time_s']))
    write(root/'SCORE.json', result)
    return result
