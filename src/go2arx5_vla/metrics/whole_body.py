"""Whole-body metrics for native WBC rollout artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.so3 import geodesic_angle, quat_to_matrix


def _array(data: Mapping[str, Any], key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
    if key not in data:
        if default is None:
            raise KeyError(key)
        return np.asarray(default)
    return np.asarray(data[key])


def metric_summary(values: Any) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": float("nan"), "rmse": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _valid_mask(data: Mapping[str, Any]) -> np.ndarray:
    for key in ("tracking_valid", "metric_valid"):
        if key in data:
            return np.asarray(data[key], dtype=bool).reshape(-1)
    return np.ones(len(_array(data, "desired_pos")), dtype=bool)


def _yaw_from_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = [q[..., i] for i in range(4)]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _slice_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return metric_summary(values[mask]) if np.any(mask) else metric_summary(np.empty(0))


def compute_whole_body_metrics(
    data: Mapping[str, Any], *, boundary_time_s: Optional[float] = None
) -> dict[str, Any]:
    """Compute tracking and whole-body metrics on the valid reference prefix."""

    mask = _valid_mask(data)
    desired_pos = _array(data, "desired_pos")
    achieved_pos = _array(data, "achieved_pos")
    desired_quat = _array(data, "desired_quat")
    achieved_quat = _array(data, "achieved_quat")
    position_error = np.linalg.norm(desired_pos - achieved_pos, axis=-1)
    rotation_error = geodesic_angle(quat_to_matrix(desired_quat), quat_to_matrix(achieved_quat))
    sim_time = _array(data, "sim_time", np.arange(len(mask), dtype=np.float64) * 0.02).reshape(-1)
    result: dict[str, Any] = {
        "sample_count": int(np.sum(mask)),
        "position_m": _slice_metrics(position_error, mask),
        "rotation_rad": _slice_metrics(rotation_error, mask),
        "reference_completion": bool(np.asarray(data.get("reference_complete", [False])).any()),
        "tracking_valid_sample_count": int(np.sum(mask)),
    }
    if "traj_s_ref" in data and "traj_L" in data:
        s_ref = np.asarray(data["traj_s_ref"], dtype=np.float64).reshape(-1)
        length = np.asarray(data["traj_L"], dtype=np.float64).reshape(-1)
        valid_length = np.maximum(length, 1e-9)
        result["path_completion"] = float(np.clip(s_ref[mask][-1] / valid_length[mask][-1], 0.0, 1.0)) if np.any(mask) else float("nan")
    else:
        result["path_completion"] = None
    if "traj_timing_err" in data:
        timing = np.asarray(data["traj_timing_err"], dtype=np.float64).reshape(-1)
        result["lag_arc_m"] = _slice_metrics(np.abs(timing), mask)
        result["signed_timing_error_m"] = _slice_metrics(timing, mask)
    else:
        result["lag_arc_m"] = metric_summary(np.empty(0))
        result["signed_timing_error_m"] = metric_summary(np.empty(0))
    result["time_in_tolerance_fraction_0.05m_0.20rad"] = (
        float(np.mean((position_error[mask] <= 0.05) & (rotation_error[mask] <= 0.20)))
        if np.any(mask)
        else float("nan")
    )

    base_pos = _array(data, "base_pos")
    base_quat = _array(data, "base_quat")
    base_yaw = np.unwrap(_yaw_from_quaternion(base_quat))
    base_delta = np.diff(base_pos, axis=0)
    result["base"] = {
        "displacement_m": float(np.linalg.norm(base_pos[mask][-1] - base_pos[mask][0])) if np.any(mask) else float("nan"),
        "path_length_m": float(np.sum(np.linalg.norm(base_delta, axis=-1)[mask[1:]])) if len(base_delta) else 0.0,
        "yaw_change_rad": float(abs(base_yaw[mask][-1] - base_yaw[mask][0])) if np.any(mask) else float("nan"),
        "linear_velocity_mps": _slice_metrics(
            np.linalg.norm(_array(data, "base_lin_vel"), axis=-1), mask
        ) if "base_lin_vel" in data else metric_summary(np.empty(0)),
        "yaw_rate_radps": _slice_metrics(
            np.abs(np.gradient(base_yaw, sim_time, edge_order=1)), mask
        ) if len(base_yaw) > 1 else metric_summary(np.empty(0)),
    }
    if "arm_q" in data:
        arm_q = np.asarray(data["arm_q"], dtype=np.float64)
    else:
        dof_pos = _array(data, "dof_pos")
        arm_q = dof_pos[:, 12:18]
    arm_step = np.linalg.norm(np.diff(arm_q, axis=0), axis=-1) if len(arm_q) > 1 else np.zeros(0)
    result["arm"] = {
        "joint_path_length_rad": float(np.sum(arm_step[mask[1:]])) if len(arm_step) else 0.0,
        "joint_step_rad": _slice_metrics(arm_step, mask[1:]) if len(arm_step) else metric_summary(np.empty(0)),
    }
    if "joint_limit_margin" in data:
        margin = np.asarray(data["joint_limit_margin"], dtype=np.float64)
        # The launcher logs margins for all native DOFs.  Keep this arm field
        # scoped to the six arm joints instead of accidentally reporting a
        # dog/gripper margin as an arm safety metric.
        arm_margin = margin[:, 12:18] if margin.ndim == 2 and margin.shape[1] >= 18 else margin
        result["arm"]["minimum_joint_limit_margin"] = float(np.min(arm_margin[mask])) if np.any(mask) else float("nan")
        result["arm"]["joint_limit_margin_p01"] = float(np.percentile(arm_margin[mask], 1)) if np.any(mask) else float("nan")
    else:
        result["arm"]["minimum_joint_limit_margin"] = None

    for flag_name in ("fall_flag", "reset_buf", "invalid_state", "joint_limit_violation"):
        if flag_name in data:
            result[flag_name] = int(np.asarray(data[flag_name], dtype=bool).reshape(-1)[mask].sum())
        else:
            result[flag_name] = 0

    if boundary_time_s is not None:
        before = mask & (sim_time <= float(boundary_time_s))
        after = mask & (sim_time > float(boundary_time_s))
        result["progressive_boundary"] = {
            "boundary_time_s": float(boundary_time_s),
            "before_sample_count": int(before.sum()),
            "after_sample_count": int(after.sum()),
            "base_motion_before_m": _path_length(base_pos, before),
            "base_motion_after_m": _path_length(base_pos, after),
            "eef_position_error_before_m": _slice_metrics(position_error, before),
            "eef_position_error_after_m": _slice_metrics(position_error, after),
            "eef_rotation_error_before_rad": _slice_metrics(rotation_error, before),
            "eef_rotation_error_after_rad": _slice_metrics(rotation_error, after),
        }
    return result


def _path_length(values: np.ndarray, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(values[indices], axis=0), axis=-1)))


def classify_progressive_behavior(
    metrics: Mapping[str, Any], *, boundary_time_s: Optional[float] = None
) -> str:
    """Apply an auditable coarse label to a progressive rollout."""

    if metrics.get("termination_code") in {"TRACKING_ABORT", "FALL", "INVALID_STATE", "JOINT_LIMIT"}:
        return "TRACKING_FAILURE"
    boundary = metrics.get("progressive_boundary")
    if boundary is None and boundary_time_s is not None:
        boundary = {}
    if boundary:
        before = float(boundary.get("base_motion_before_m", 0.0))
        after = float(boundary.get("base_motion_after_m", 0.0))
        if after > max(0.02, 1.25 * before):
            return "CONTINUOUS_BASE_COORDINATION"
        if after > 0.02:
            return "BASE_RELOCATION_LIKE"
    base = metrics.get("base", {})
    if float(base.get("path_length_m", 0.0)) < 0.05:
        return "ARM_ONLY_LIKE"
    if metrics.get("rotation_rad", {}).get("p95", 0.0) > 0.8:
        return "OSCILLATORY_REDUNDANCY_RESOLUTION"
    return "OTHER"
