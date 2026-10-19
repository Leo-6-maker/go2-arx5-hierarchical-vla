"""Read-only H40/prefix kinematic attribution, never a shorter safety gate."""
import numpy as np

from scripts.pi0292.eval_contract import error_metrics, gate_action, matrix10
from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateConfig
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix


REGIONS = ('PREFIX8_INTERNAL', 'PREFIX_TAIL_BOUNDARY', 'TAIL_INTERNAL')


def analyze(action, state, observation_time, reference_uuid, *, expert=None):
    """Locate existing violations without changing arrays, time or thresholds.

    Edge 8->9 is kept separate: its first endpoint belongs to the executed
    prefix but its second endpoint does not. A prefix report is not permission
    to execute a chunk rejected by the original full-H40 gate.
    """
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if action.shape != (40, 10) or state.shape != (10,):
        raise ValueError('ORIGINAL_FULL_H40_AND_ACTUAL_STATE_REQUIRED')
    if (not np.isfinite(action).all() or not np.isfinite(state).all()
            or not np.isfinite(observation_time) or observation_time<0):
        raise ValueError('NONFINITE_ARRAY_NOT_A_FINITE_HORIZON_ATTRIBUTION')
    gate = gate_action(action, state, observation_time, reference_uuid)
    config = TrajectorySafetyGateConfig()
    dt = np.diff(float(observation_time) + np.arange(40) / 20)
    rotation = quat_to_matrix(matrix_to_quat(matrix10(action)[:, :3, :3]))
    values = dict(linear_speed_mps=np.linalg.norm(np.diff(action[:, :3], axis=0), axis=-1) / dt,
        angular_speed_radps=geodesic_angle(rotation[:-1], rotation[1:]) / dt)
    thresholds = dict(linear_speed_mps=config.linear_speed_reject_mps,
        angular_speed_radps=config.angular_speed_reject_radps)
    regions = [REGIONS[0] if i < 7 else REGIONS[1] if i == 7 else REGIONS[2] for i in range(39)]
    speeds = {}
    for name, array in values.items():
        threshold = thresholds[name]
        index = int(np.argmax(array))
        speeds[name] = dict(reject_threshold=threshold, max=float(array[index]),
            max_edge_points_1based=[index + 1, index + 2], max_region=regions[index],
            regions={region:dict(edges=sum(r == region for r in regions),
                max=float(np.max(array[np.asarray(regions) == region])),
                reject_edges_points_1based=[[i + 1, i + 2] for i, value in enumerate(array)
                    if regions[i] == region and value > threshold]) for region in REGIONS})
    errors = None
    if expert is not None:
        expert = np.asarray(expert, dtype=np.float64)
        if expert.shape != (40, 10) or not np.isfinite(expert).all():
            raise ValueError('EXPLICIT_FINITE_MATCHED_H40_EXPERT_REQUIRED')
        original = error_metrics(action, expert, state)
        errors = dict(prefix8=original['prefix8'], H40=original['H40'],
            tail32={name:float(np.mean(original['pointwise'][key][8:])) for name, key in
                [('XYZ_mean_m', 'XYZ_euclidean_m'), ('SO3_mean_rad', 'SO3_rad'), ('gripper_MAE', 'gripper_absolute')]})
    return dict(schema='PI0294_PV1_READ_ONLY_HORIZON_ATTRIBUTION_V1',
        original_full_H40_gate=gate, speed_attribution=speeds,
        first_point_position_jump_m=float(np.linalg.norm(action[0, :3] - state[:3])),
        gripper_outside_original_range_points_1based=[int(i + 1) for i in np.flatnonzero((action[:, 9] < 0) | (action[:, 9] > 1))],
        matched_expert_errors=errors, matched_expert_available=expert is not None,
        prefix_is_not_an_alternative_admission_gate=True, actions_modified=False,
        new_queries=0, new_physics=0)
