"""Separate reference, controller, and end-to-end task-space metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..trajectory.so3 import geodesic_angle, quat_to_matrix

REFERENCE_RECONSTRUCTION_ERROR = "REFERENCE_RECONSTRUCTION_ERROR"
CONTROLLER_TRACKING_ERROR = "CONTROLLER_TRACKING_ERROR"
END_TO_END_TASK_SPACE_ERROR = "END_TO_END_TASK_SPACE_ERROR"


def error_summary(values: Any) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("metric values must be non-empty and finite")
    return {
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def task_space_error(
    reference_position: Any,
    achieved_position: Any,
    reference_rotation: Any,
    achieved_rotation: Any,
    *,
    metric_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p_ref = np.asarray(reference_position, dtype=np.float64)
    p_ach = np.asarray(achieved_position, dtype=np.float64)
    r_ref = np.asarray(reference_rotation, dtype=np.float64)
    r_ach = np.asarray(achieved_rotation, dtype=np.float64)
    if p_ref.shape != p_ach.shape or p_ref.shape[-1] != 3:
        raise ValueError("position arrays must have equal shape [...,3]")
    if r_ref.shape != r_ach.shape or r_ref.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must have equal shape [...,3,3]")
    pos = np.linalg.norm(p_ref - p_ach, axis=-1)
    rot = geodesic_angle(r_ref, r_ach)
    return {
        "metric_name": metric_name,
        "position_m": error_summary(pos),
        "rotation_rad": error_summary(rot),
        "sample_count": int(pos.size),
        "metadata": {} if metadata is None else dict(metadata),
    }


def _rotation_matrix_or_quaternion(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-2:] == (3, 3):
        return array
    if array.shape[-1:] == (4,):
        return quat_to_matrix(array)
    raise ValueError("rotation arrays must be [...,3,3] or [...,4] xyzw quaternions")


def _named_task_space_error(
    reference_position: Any,
    achieved_position: Any,
    reference_rotation: Any,
    achieved_rotation: Any,
    *,
    metric_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return task_space_error(
        reference_position,
        achieved_position,
        _rotation_matrix_or_quaternion(reference_rotation),
        _rotation_matrix_or_quaternion(achieved_rotation),
        metric_name=metric_name,
        metadata=metadata,
    )


def reference_reconstruction_error(
    continuous_position: Any,
    reconstructed_position: Any,
    continuous_rotation: Any,
    reconstructed_rotation: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Error introduced by representing a continuous reference through a chunk."""
    return _named_task_space_error(
        continuous_position,
        reconstructed_position,
        continuous_rotation,
        reconstructed_rotation,
        metric_name=REFERENCE_RECONSTRUCTION_ERROR,
        metadata=metadata,
    )


def controller_tracking_error(
    consumed_position: Any,
    achieved_position: Any,
    consumed_rotation: Any,
    achieved_rotation: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Error between the reference consumed by WBC and achieved EEF pose."""
    return _named_task_space_error(
        consumed_position,
        achieved_position,
        consumed_rotation,
        achieved_rotation,
        metric_name=CONTROLLER_TRACKING_ERROR,
        metadata=metadata,
    )


def end_to_end_task_space_error(
    continuous_position: Any,
    achieved_position: Any,
    continuous_rotation: Any,
    achieved_rotation: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end error from the original continuous target to achieved EEF."""
    return _named_task_space_error(
        continuous_position,
        achieved_position,
        continuous_rotation,
        achieved_rotation,
        metric_name=END_TO_END_TASK_SPACE_ERROR,
        metadata=metadata,
    )
