"""Canonical target/progress and decoded-vs-expert metrics.

This is a direct extraction of the accepted Stage44R semantics.  In
particular, progress is *distance reduction* (``d_start - d_end``), not a
thresholded task-success label, and the displacement cosine is undefined
when either vector has effectively zero length.  All positions are world
frame metres and all action comparisons retain the existing mixed-unit
diagnostic names.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np


ACTION_HORIZON = 40
ACTION_WIDTH = 10
TARGET_PROGRESS_TOLERANCE_M = 1.0e-9


class EvaluationContractError(ValueError):
    """Raised when an evaluation input violates the frozen CPU contract."""


def _finite_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(f"{name} is not numeric") from exc
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise EvaluationContractError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise EvaluationContractError(f"{name} contains non-finite values")
    return array


def array_sha256(value: Any) -> str:
    """Hash a contiguous array without changing dtype or values."""

    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def trajectory_metrics(
    positions_world: Any,
    target_world: Any,
    *,
    prefix: str = "trajectory",
    progress_tolerance_m: float = TARGET_PROGRESS_TOLERANCE_M,
) -> dict[str, Any]:
    """Return the accepted Stage44R target-direction metrics.

    ``positions_world`` is an ``[N,3]`` world-frame EEF path.  ``N=1`` is
    valid for a raw metric record (path length is then zero), while an empty
    path is rejected.  The output key names intentionally match Stage44R.
    """

    positions = _finite_array(positions_world, name=f"{prefix}.positions_world")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise EvaluationContractError(
            f"{prefix}.positions_world must be finite [N,3] with N>=1, got {positions.shape}"
        )
    target = _finite_array(target_world, name=f"{prefix}.target_world", shape=(3,))
    tolerance = float(progress_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise EvaluationContractError("progress_tolerance_m must be finite and non-negative")

    distances = np.linalg.norm(positions - target[None, :], axis=1)
    if not np.isfinite(distances).all():  # defensive after finite-input validation
        raise EvaluationContractError(f"{prefix} distances are non-finite")
    displacement = positions[-1] - positions[0]
    target_vector = target - positions[0]
    displacement_norm = float(np.linalg.norm(displacement))
    target_norm = float(np.linalg.norm(target_vector))
    cosine = (
        None
        if displacement_norm <= 1.0e-12 or target_norm <= 1.0e-12
        else float(np.dot(displacement, target_vector) / (displacement_norm * target_norm))
    )
    progress = float(distances[0] - distances[-1])
    best_progress = float(distances[0] - np.min(distances))
    sign = (
        "POSITIVE"
        if progress > tolerance
        else ("ZERO" if abs(progress) <= tolerance else "NEGATIVE")
    )
    return {
        "initial_target_distance_m": float(distances[0]),
        "minimum_target_distance_m": float(np.min(distances)),
        "minimum_target_index": int(np.argmin(distances)),
        "final_target_distance_m": float(distances[-1]),
        "absolute_progress_m": progress,
        "best_progress_m": best_progress,
        "progress_sign": sign,
        "endpoint_displacement_m": displacement.tolist(),
        "target_displacement_m": target_vector.tolist(),
        "endpoint_target_vector_cosine": cosine,
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "roughness_position_second_difference_m": optional_numeric_summary(
            np.linalg.norm(np.diff(positions, n=2, axis=0), axis=1)
        ),
        "sample_count": int(positions.shape[0]),
    }


def comparison_metrics(
    decoded: Any,
    expert: Any,
    anchor_world: Any,
    target_world: Any,
    *,
    prefix: str = "prediction",
) -> dict[str, Any]:
    """Compare one decoded ``[40,10]`` action with its expert action.

    This preserves Stage44R's position/rot6d/gripper diagnostics and computes
    target metrics on the decoded position path after applying the world-frame
    anchor.  It does not call a model or a SafetyGate.
    """

    decoded_array = _finite_array(
        decoded, name=f"{prefix}.decoded", shape=(ACTION_HORIZON, ACTION_WIDTH)
    )
    expert_array = _finite_array(
        expert, name=f"{prefix}.expert", shape=(ACTION_HORIZON, ACTION_WIDTH)
    )
    anchor = _finite_array(anchor_world, name=f"{prefix}.anchor_world", shape=(3,))
    target = _finite_array(target_world, name=f"{prefix}.target_world", shape=(3,))
    delta = decoded_array - expert_array
    prediction_world = decoded_array[:, :3] + anchor[None, :]
    target_result = trajectory_metrics(prediction_world, target, prefix=f"{prefix}.target")
    prediction_roughness = target_result["roughness_position_second_difference_m"]
    expert_roughness = optional_numeric_summary(
        np.linalg.norm(np.diff(expert_array[:, :3], n=2, axis=0), axis=1)
    )
    expert_mean = expert_roughness["mean"]
    prediction_mean = prediction_roughness["mean"]
    return {
        "target": target_result,
        "position_rmse_m": float(np.sqrt(np.mean(delta[:, :3] * delta[:, :3]))),
        "position_max_abs_m": float(np.max(np.abs(delta[:, :3]))),
        "endpoint_error_m": float(np.linalg.norm(decoded_array[-1, :3] - expert_array[-1, :3])),
        "prediction_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(decoded_array[:, :3], axis=0), axis=1))
        ),
        "expert_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(expert_array[:, :3], axis=0), axis=1))
        ),
        "prediction_roughness_position_second_difference_m": prediction_roughness,
        "expert_roughness_position_second_difference_m": expert_roughness,
        "roughness_ratio": (
            None
            if expert_mean is None or float(expert_mean) <= 1.0e-12
            else float(prediction_mean) / float(expert_mean)
        ),
        "eef_rot6d_rmse": float(np.sqrt(np.mean(delta[:, 3:9] * delta[:, 3:9]))),
        "eef_rot6d_max_abs": float(np.max(np.abs(delta[:, 3:9]))),
        "gripper_rmse": float(np.sqrt(np.mean(delta[:, 9:10] * delta[:, 9:10]))),
        "gripper_max_abs": float(np.max(np.abs(delta[:, 9:10]))),
        "full_action_rmse_mixed_units": float(np.sqrt(np.mean(delta * delta))),
    }


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    """Return deterministic scalar summary statistics for aggregate evidence."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise EvaluationContractError("metric sample is empty or non-finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def optional_numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    """Match the existing action-profile empty-second-difference convention."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return numeric_summary(array)


def aggregate_trajectory_metrics(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate target metric dictionaries without introducing thresholds."""

    rows = [dict(item) for item in metrics]
    if not rows:
        raise EvaluationContractError("cannot aggregate an empty metric cohort")
    scalar_keys = (
        "initial_target_distance_m",
        "minimum_target_distance_m",
        "final_target_distance_m",
        "absolute_progress_m",
        "best_progress_m",
        "path_length_m",
    )
    result: dict[str, Any] = {
        "count": len(rows),
        "progress_sign_counts": {
            sign: int(sum(row.get("progress_sign") == sign for row in rows))
            for sign in ("POSITIVE", "ZERO", "NEGATIVE")
        },
    }
    for key in scalar_keys:
        result[key] = numeric_summary(float(row[key]) for row in rows)
    cosines = [row["endpoint_target_vector_cosine"] for row in rows if row.get("endpoint_target_vector_cosine") is not None]
    result["endpoint_target_vector_cosine"] = numeric_summary(float(value) for value in cosines) if cosines else None
    roughness_means = [
        row["roughness_position_second_difference_m"].get("mean")
        for row in rows
        if isinstance(row.get("roughness_position_second_difference_m"), Mapping)
        and row["roughness_position_second_difference_m"].get("mean") is not None
    ]
    result["roughness_position_second_difference_m"] = (
        numeric_summary(float(value) for value in roughness_means)
        if roughness_means
        else None
    )
    return result


__all__ = [
    "ACTION_HORIZON",
    "ACTION_WIDTH",
    "TARGET_PROGRESS_TOLERANCE_M",
    "EvaluationContractError",
    "aggregate_trajectory_metrics",
    "array_sha256",
    "comparison_metrics",
    "numeric_summary",
    "optional_numeric_summary",
    "trajectory_metrics",
]
