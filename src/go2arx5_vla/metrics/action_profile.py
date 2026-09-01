"""Non-corrective action-profile and chunk-overlap metrics for Stage22.

The functions in this module audit a predicted or expert task-space chunk. No
clipping, smoothing, retiming, or other trajectory modification is performed.
All timestamps remain part of the chunk contract.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.so3 import geodesic_angle, quat_slerp, quat_to_matrix


def _stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        return {"count": int(array.size), "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _rotation_steps(quaternion: np.ndarray) -> np.ndarray:
    if len(quaternion) < 2:
        return np.empty(0, dtype=np.float64)
    dots = np.clip(np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def trajectory_profile(
    position: Any,
    quaternion: Any,
    timestamps: Any,
    *,
    achieved_position: Optional[Any] = None,
    achieved_quaternion: Optional[Any] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return descriptive profile metrics for one timestamped pose sequence."""

    pos = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1 or pos.shape != (ts.size, 3) or quat.shape != (ts.size, 4):
        raise ValueError("profile pose arrays must have shapes [N], [N,3], and [N,4]")
    if ts.size < 2 or not np.isfinite(ts).all() or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("profile inputs must be finite and contain at least two points")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("profile timestamps must be strictly increasing")
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("profile contains a zero quaternion")
    quat = quat / norms
    dt = np.diff(ts)
    translation_step = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    rotation_step = _rotation_steps(quat)
    linear_speed = translation_step / dt
    angular_speed = rotation_step / dt
    second_difference = np.diff(pos, n=2, axis=0)
    roughness = np.linalg.norm(second_difference, axis=-1)
    result: dict[str, Any] = {
        "point_count": int(ts.size),
        "duration_s": float(ts[-1] - ts[0]),
        "path_length_m": float(np.sum(translation_step)),
        "translation_step_m": _stats(translation_step),
        "rotation_step_rad": _stats(rotation_step),
        "linear_speed_mps": _stats(linear_speed),
        "angular_speed_radps": _stats(angular_speed),
        "roughness_position_second_difference_m": _stats(roughness),
        "finite": True,
        "timestamp_period_s": _stats(dt),
        "provenance": {} if provenance is None else dict(provenance),
    }
    if achieved_position is not None:
        achieved = np.asarray(achieved_position, dtype=np.float64).reshape(3)
        if not np.isfinite(achieved).all():
            raise ValueError("achieved_position must be finite [3]")
        result["initial_position_discontinuity_m"] = float(np.linalg.norm(pos[0] - achieved))
    if achieved_quaternion is not None:
        achieved_q = np.asarray(achieved_quaternion, dtype=np.float64).reshape(4)
        if not np.isfinite(achieved_q).all() or np.linalg.norm(achieved_q) < 1e-10:
            raise ValueError("achieved_quaternion must be a finite non-zero [4]")
        achieved_q = achieved_q / np.linalg.norm(achieved_q)
        result["initial_rotation_discontinuity_rad"] = float(
            geodesic_angle(quat_to_matrix(achieved_q), quat_to_matrix(quat[0]))
    )
    return result

def chunk_profile(
    chunk: TrajectoryChunkV1,
    *,
    achieved_position: Optional[Any] = None,
    achieved_quaternion: Optional[Any] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Profile a canonical chunk without changing its values."""

    return trajectory_profile(
        chunk.eef_position,
        chunk.eef_orientation,
        chunk.point_timestamps,
        achieved_position=achieved_position,
        achieved_quaternion=achieved_quaternion,
        provenance={"chunk_id": chunk.chunk_id, **({} if provenance is None else dict(provenance))},
    )


def _sample_pose(chunk: TrajectoryChunkV1, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a chunk at absolute timestamps using linear position/SLERP pose."""

    ts = np.asarray(chunk.point_timestamps, dtype=np.float64)
    if np.any(query < ts[0] - 1e-10) or np.any(query > ts[-1] + 1e-10):
        raise ValueError("overlap query is outside the chunk interval")
    pos = np.empty((query.size, 3), dtype=np.float64)
    quat = np.empty((query.size, 4), dtype=np.float64)
    for index, value in enumerate(query):
        right = int(np.searchsorted(ts, value, side="left"))
        right = min(max(right, 0), len(ts) - 1)
        if right == 0 or np.isclose(value, ts[right], atol=1e-12):
            left = right
            alpha = 0.0
        else:
            left = right - 1
            alpha = float((value - ts[left]) / (ts[right] - ts[left]))
        pos[index] = (1.0 - alpha) * chunk.eef_position[left] + alpha * chunk.eef_position[right]
        quat[index] = quat_slerp(chunk.eef_orientation[left], chunk.eef_orientation[right], alpha)
    return pos, quat


def chunk_overlap_metrics(
    previous: TrajectoryChunkV1,
    current: TrajectoryChunkV1,
    *,
    overlap_sample_timestamps: Optional[Any] = None,
) -> dict[str, Any]:
    """Compare two predictions on their common absolute future interval."""

    start = max(float(previous.point_timestamps[0]), float(current.point_timestamps[0]))
    end = min(float(previous.point_timestamps[-1]), float(current.point_timestamps[-1]))
    result: dict[str, Any] = {
        "previous_chunk_id": previous.chunk_id,
        "current_chunk_id": current.chunk_id,
        "common_interval_s": None,
        "overlap_point_count": 0,
        "overlap_available": bool(end >= start),
    }
    if end < start:
        return result
    if overlap_sample_timestamps is None:
        grid = np.unique(
            np.concatenate(
                [
                    previous.point_timestamps[
                        (previous.point_timestamps >= start) & (previous.point_timestamps <= end)
                    ],
                    current.point_timestamps[
                        (current.point_timestamps >= start) & (current.point_timestamps <= end)
                    ],
                    np.asarray([start, end], dtype=np.float64),
                ]
            )
        )
    else:
        grid = np.asarray(overlap_sample_timestamps, dtype=np.float64).reshape(-1)
        if grid.size == 0 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("overlap_sample_timestamps must be finite and strictly increasing")
        if grid[0] < start - 1e-10 or grid[-1] > end + 1e-10:
            raise ValueError("overlap_sample_timestamps must lie inside the common interval")
    old_pos, old_q = _sample_pose(previous, grid)
    new_pos, new_q = _sample_pose(current, grid)
    position_error = np.linalg.norm(old_pos - new_pos, axis=-1)
    rotation_error = np.asarray(
        geodesic_angle(quat_to_matrix(old_q), quat_to_matrix(new_q)), dtype=np.float64
    ).reshape(-1)
    if grid.size >= 2:
        old_velocity = np.diff(old_pos, axis=0) / np.diff(grid)[:, None]
        new_velocity = np.diff(new_pos, axis=0) / np.diff(grid)[:, None]
        tangent_error = np.linalg.norm(old_velocity - new_velocity, axis=-1)
    else:
        tangent_error = np.empty(0, dtype=np.float64)
    result.update(
        {
            "common_interval_s": [float(grid[0]), float(grid[-1])],
            "overlap_point_count": int(grid.size),
            "position_error_m": _stats(position_error),
            "rotation_error_rad": _stats(rotation_error),
            "translational_tangent_error_mps": _stats(tangent_error),
            "timestamp_grid_s": grid.tolist(),
            "no_trajectory_correction": True,
        }
    )
    new_first_time = float(current.point_timestamps[0])
    old_at_new_first, old_q_at_new_first = _sample_pose(
        previous, np.asarray([new_first_time], dtype=np.float64)
    )
    result["old_prediction_at_new_observation_vs_new_first"] = {
        "timestamp_s": new_first_time,
        "position_error_m": float(np.linalg.norm(old_at_new_first[0] - current.eef_position[0])),
        "rotation_error_rad": float(
            geodesic_angle(
                quat_to_matrix(old_q_at_new_first[0]),
                quat_to_matrix(current.eef_orientation[0]),
            )
        ),
    }
    return result
