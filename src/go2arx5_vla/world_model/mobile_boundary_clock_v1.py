"""New mobile consumer identity: accepted receiver knot at the actual clock.

This reads the original receiver/solver witnesses. It changes no controller,
preview length, certification threshold, plan or gripper command.
"""
import numpy as np
from ..trajectory.so3 import quat_to_matrix
from .target_reference import trajectory_identity, installed_target_identities

VERSION = 'MOBILE_ACCEPTED_BOUNDARY_050_V1'


def _policy_identity(event):
    target = event['message']['plan_target_trajectories']
    return trajectory_identity(target['time_trajectory'],
        [s['value'] for s in target['state_trajectory']],
        [s['value'] for s in target['input_trajectory']])


def accepted_boundary(events, wire, native_time, policies, *, initial_hold=False, policy_identity_cache=None):
    sent = installed_target_identities(events, wire)
    witnesses = []
    for event in policies:
        if event['topic_key'] != 'policy':
            continue
        # ROS callbacks append immutable event dictionaries. Hash each actual
        # target once; rescanning and serializing the entire growing policy
        # history every50ms made wall cost grow with episode length.
        identity=id(event)
        key=None if policy_identity_cache is None else policy_identity_cache.get(identity)
        if key is None:
            key = _policy_identity(event)
            if policy_identity_cache is not None:policy_identity_cache[identity]=key
        if key in sent and sent[key]['monotonic_s'] <= event['received_monotonic_s']:
            witnesses.append(key)
    return _select_boundary(sent, reversed(witnesses), wire, native_time, initial_hold)


def _select_boundary(sent, latest_first, wire, native_time, initial_hold):
    found = False
    # Rolling publications have the same accepted reference. Select an exact
    # knot in an actual publication, rather than guessing controller interpolation.
    for key in latest_first:
        found = True
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
    if not found:
        raise ValueError('ACCEPTED_REFERENCE_REQUIRES_EXACT_SOLVER_TARGET_WITNESS')
    raise ValueError('NO_EXACT_ACCEPTED_RECEIVER_KNOT_AT_ACTUAL_NATIVE_BOUNDARY')


class AcceptedBoundaryIndex:
    """Index append-only receiver and callback evidence by exact identity."""

    def __init__(self):
        self.planner_cursor = self.policy_cursor = (0, None, None)
        self.by_wire = {}
        self.policy_observations = {}

    @staticmethod
    def offset(rows, cursor):
        size, first, last = cursor
        if size and (len(rows) < size or rows[0] is not first or rows[size-1] is not last):
            return 0
        return size

    @staticmethod
    def cursor(rows):
        return (len(rows), rows[0], rows[-1]) if rows else (0, None, None)

    def update(self, events, policies):
        first_event = self.offset(events, self.planner_cursor)
        first_policy = self.offset(policies, self.policy_cursor)
        if not first_event:
            self.by_wire.clear()
        if not first_policy:
            self.policy_observations.clear()
        for event in events[first_event:]:
            if event.get('kind') == 'installed':
                key = trajectory_identity(event['times'], event['poses'], event['inputs'])
                self.by_wire.setdefault(event['ref_id'], {})[key] = event
        for sequence in range(first_policy, len(policies)):
            policy = policies[sequence]
            if policy['topic_key'] == 'policy':
                key = _policy_identity(policy)
                self.policy_observations.setdefault(key, []).append((sequence, policy['received_monotonic_s']))
        self.planner_cursor, self.policy_cursor = self.cursor(events), self.cursor(policies)

    def boundary(self, wire, native_time, *, initial_hold=False):
        sent = self.by_wire.get(wire, {})
        matches = []
        for key, event in sent.items():
            for sequence, stamp in reversed(self.policy_observations.get(key, ())):
                if stamp >= event['monotonic_s']:
                    matches.append((sequence, key))
                    break
        matches.sort(reverse=True)
        return _select_boundary(sent, (key for _,key in matches), wire, native_time, initial_hold)


def boundary(adapter):
    packet = adapter.observe()
    wire = adapter.active_reference['wire_sha256']
    index = getattr(adapter, 'mobile_boundary_index', None)
    if index is not None:
        index.update(adapter.reference_events(), adapter.ros_evidence)
        pose, evidence = index.boundary(wire, packet['time_s'], initial_hold=packet['step']==5703)
    else:
        if not hasattr(adapter,'_mobile_boundary_policy_ids'):adapter._mobile_boundary_policy_ids={}
        pose, evidence = accepted_boundary(adapter.reference_events(), wire, packet['time_s'],
            adapter.ros_evidence,initial_hold=packet['step']==5703,
            policy_identity_cache=adapter._mobile_boundary_policy_ids)
    evidence['native_step'] = packet['step']
    adapter.command_event(dict(phase='mobile-accepted-boundary-050', **evidence))
    return pose, evidence
