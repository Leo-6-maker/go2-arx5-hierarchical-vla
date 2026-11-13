"""Native H40 previews for the 0.5 second expert feedback comparison."""
import numpy as np

from ..trajectory.so3 import (
    matrix_to_quat, quat_slerp, quat_to_matrix, rotation_matrix_to_rot6d_rows, rot6d_rows_to_matrix,
)
from .task_learning_grasp_v1 import FiftyMsProgram, need

VERSION = 'EEF_STREAM_V2'
PREFIX_ATOMS = 10
HORIZON = 40


def _smooth_fractions(limit, distance, blend):
    increment = limit / max(distance, limit)
    boundaries = np.minimum(np.arange(HORIZON // PREFIX_ATOMS + 1) * increment, 1.)
    return (boundaries[:-1, None] + np.diff(boundaries)[:, None] * blend).reshape(HORIZON)


def configuration(spec):
    from .corelab_expert_semantics_v3 import recorded_tail
    value = spec.get('expert_eef_stream_v2')
    if value is None:
        return False
    need(value == dict(version=VERSION, prefix_atoms=PREFIX_ATOMS)
         and recorded_tail(spec), 'EEF_STREAM_FRESH_DENSE_EXPERT_ONLY')
    return True


def expert_program(proposal, previous_nominal_grip):
    """Follow the expert goal in bounded 0.5s segments through the same WBC."""
    requested = proposal.decoded_H40
    first = requested[0]
    goal = requested[-1]
    start_rotation = rot6d_rows_to_matrix(first[3:9])
    goal_rotation = rot6d_rows_to_matrix(goal[3:9])
    start_quaternion = matrix_to_quat(start_rotation)
    goal_quaternion = matrix_to_quat(goal_rotation)
    rotation_angle = 2. * np.arccos(np.clip(abs(start_quaternion @ goal_quaternion), 0., 1.))
    displacement = goal[:3] - first[:3]
    distance = np.linalg.norm(displacement)
    grip_travel = abs(goal[9] - previous_nominal_grip)

    # Each prospective segment arrives at zero velocity. A replacement query
    # therefore starts without replaying the initial 0.5s of a 2s slow ramp.
    u = np.linspace(0., 1., PREFIX_ATOMS)
    blend = u**3 * (10. + u * (-15. + 6. * u))
    # Interpolate each segment between its own bounded endpoint fractions.
    # This keeps a short final segment smooth when it reaches the expert goal.
    position_fraction = _smooth_fractions(.005, distance, blend)
    rotation_fraction = _smooth_fractions(.025, rotation_angle, blend)
    grip_fraction = _smooth_fractions(.18, grip_travel, blend)
    raw = np.empty_like(requested)
    raw[:, :3] = first[:3] + position_fraction[:, None] * displacement
    raw[:, 3:9] = rotation_matrix_to_rot6d_rows(quat_to_matrix(
        quat_slerp(start_quaternion, goal_quaternion, rotation_fraction[:, None])))
    raw[:, 9] = previous_nominal_grip + grip_fraction * (goal[9] - previous_nominal_grip)
    return FiftyMsProgram(raw, proposal.decision_TCP_target, proposal.initial_reference_TCP_target,
                          proposal.base_pose_hold)
