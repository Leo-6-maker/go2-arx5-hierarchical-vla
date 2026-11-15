"""New mobile consumer identity: accepted receiver knot at the actual clock.

This reads the original receiver/solver witnesses. It changes no controller,
preview length, certification threshold, plan or gripper command.
"""
import numpy as np
from ..trajectory.so3 import quat_to_matrix
from .target_reference import trajectory_identity, installed_target_identities

VERSION = 'MOBILE_ACCEPTED_BOUNDARY_050_V1'


def accepted_boundary(events, wire, native_time, policies, *, initial_hold=False):
    sent = installed_target_identities(events, wire)
    witnesses = []
    for event in policies:
        if event['topic_key'] != 'policy':
            continue
        target = event['message']['plan_target_trajectories']
        key = trajectory_identity(target['time_trajectory'],
            [s['value'] for s in target['state_trajectory']],
            [s['value'] for s in target['input_trajectory']])
        if key in sent and sent[key]['monotonic_s'] <= event['received_monotonic_s']:
            witnesses.append(key)
    if not witnesses:
        raise ValueError('ACCEPTED_REFERENCE_REQUIRES_EXACT_SOLVER_TARGET_WITNESS')
    # Rolling publications have the same accepted reference. Select an exact
    # knot in an actual publication, rather than guessing controller interpolation.
    for key in reversed(witnesses):
        event = sent[key]
        indices = np.flatnonzero(np.abs(np.asarray(event['times'])-native_time) <= 1e-8)
        if len(indices) == 1:
            index = int(indices[0]); row = np.asarray(event['poses'][index], float)
            if row.shape != ((14,) if initial_hold else (7,)) or not np.isfinite(row).all():
                raise ValueError('M_FREE_ACCEPTED_EEF_ONLY_POSE7_REQUIRED')
            pose = np.eye(4); pose[:3, 3] = row[:3]; pose[:3, :3] = quat_to_matrix(row[3:7])
            return pose, dict(version=VERSION, native_time_s=float(native_time),
                receiver_knot_time_s=event['times'][index], receiver_knot_index=index,
                accepted_wire_sha256=wire, solver_target_identity=key,
                receiver_bound_start_s=event['bound_start'],
                receiver_observation_time_s=event['observation_time_s'],
                accepted_pose7=row[:7].tolist(), initial_constant_BASE_hold=initial_hold, exact_receiver_knot=True,
                controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED')
    raise ValueError('NO_EXACT_ACCEPTED_RECEIVER_KNOT_AT_ACTUAL_NATIVE_BOUNDARY')


def boundary(adapter):
    packet = adapter.observe()
    pose, evidence = accepted_boundary(adapter.reference_events(),
        adapter.active_reference['wire_sha256'], packet['time_s'], adapter.ros_evidence,
        initial_hold=packet['step']==5703)
    evidence['native_step'] = packet['step']
    adapter.command_event(dict(phase='mobile-accepted-boundary-050', **evidence))
    return pose, evidence
