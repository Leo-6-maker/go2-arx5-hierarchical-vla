"""CPU-only geometry/timing qualification for the Stage34 Motion API.

This module intentionally has no model, simulator, CUDA, or scheduler
dependency.  It implements the exact project preview semantics on timestamped
EEF trajectories so that Stage34-A/B/C can be audited without constructing
GR00T or invoking the delivered WBC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..contracts.motion_intent import (
    DEFAULT_MOTION_INTENT_V2_LAMBDA_M_PER_RAD,
    MotionIntentV2,
)
from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.preview import (
    DEFAULT_WBC_PREVIEW_OFFSETS_M,
    se3_arc_length_preview,
)
from ..trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rot6d_rows_to_matrix,
)


DEFAULT_SE3_LAMBDA_M_PER_RAD = DEFAULT_MOTION_INTENT_V2_LAMBDA_M_PER_RAD
DEFAULT_PREVIEW_POSITION_P95_LIMIT_M = 0.006767
DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD = 0.038215
DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN = 0.99
GEOMETRY_K_VALUES = (4, 6, 9, 12)
TIMING_MODES = ("T1", "T3", "T5", "T9")
TIMING_KNOTS = {
    "T1": np.asarray([0.0, 1.0], dtype=np.float64),
    "T3": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    "T5": np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64),
    "T9": np.asarray([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float64),
}


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _arc_map(
    positions: np.ndarray,
    orientations: np.ndarray,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    if lambda_m_per_rad <= 0.0 or not np.isfinite(lambda_m_per_rad):
        raise ValueError("lambda_m_per_rad must be finite and positive")
    translation = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    rotation = geodesic_angle(
        quat_to_matrix(orientations[:-1]), quat_to_matrix(orientations[1:])
    )
    segments = np.sqrt(translation * translation + (lambda_m_per_rad * rotation) ** 2)
    return segments, np.concatenate(([0.0], np.cumsum(segments)))


def _arc_progress(
    chunk: TrajectoryChunkV1,
    times: np.ndarray,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> np.ndarray:
    segments, cumulative = _arc_map(chunk.eef_position, chunk.eef_orientation, lambda_m_per_rad)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.zeros_like(times, dtype=np.float64)
    out = np.empty_like(times, dtype=np.float64)
    source_times = _as_float_array(chunk.point_timestamps)
    for index, timestamp in enumerate(times):
        if timestamp <= source_times[0]:
            out[index] = 0.0
            continue
        if timestamp >= source_times[-1]:
            out[index] = 1.0
            continue
        i = int(np.searchsorted(source_times, timestamp, side="right") - 1)
        dt = source_times[i + 1] - source_times[i]
        fraction = 0.0 if dt <= 0.0 else (timestamp - source_times[i]) / dt
        out[index] = (cumulative[i] + fraction * segments[i]) / total
    return out


def _pose_time_at_arc(
    chunk: TrajectoryChunkV1,
    target_arc: float,
    *,
    segments: np.ndarray | None = None,
    cumulative: np.ndarray | None = None,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Interpolate a pose and its original timestamp at an arc-length value."""

    positions = _as_float_array(chunk.eef_position)
    orientations = _as_float_array(chunk.eef_orientation)
    timestamps = _as_float_array(chunk.point_timestamps)
    if segments is None or cumulative is None:
        segments, cumulative = _arc_map(positions, orientations, lambda_m_per_rad)
    total = float(cumulative[-1])
    if target_arc <= 1e-12:
        return positions[0].copy(), orientations[0].copy(), float(timestamps[0]), False
    if target_arc >= total - 1e-12:
        return positions[-1].copy(), orientations[-1].copy(), float(timestamps[-1]), True
    i = int(np.searchsorted(cumulative, target_arc, side="right") - 1)
    i = max(0, min(i, len(segments) - 1))
    while i < len(segments) - 1 and segments[i] <= 1e-12:
        i += 1
    if segments[i] <= 1e-12:
        return positions[-1].copy(), orientations[-1].copy(), float(timestamps[-1]), True
    fraction = float((target_arc - cumulative[i]) / segments[i])
    fraction = float(np.clip(fraction, 0.0, 1.0))
    position = (1.0 - fraction) * positions[i] + fraction * positions[i + 1]
    orientation = quat_slerp(orientations[i], orientations[i + 1], fraction)
    timestamp = (1.0 - fraction) * timestamps[i] + fraction * timestamps[i + 1]
    return position, orientation, float(timestamp), False


def _piecewise_linear(x: np.ndarray, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    return np.interp(x, knots_x, knots_y)


def _gripper_at(chunk: TrajectoryChunkV1, timestamps: np.ndarray) -> np.ndarray | None:
    if chunk.gripper is None:
        return None
    source = _as_float_array(chunk.gripper)
    source_flat = source.reshape(len(chunk.point_timestamps), -1)
    if source_flat.shape[1] != 1:
        raise ValueError("Stage34 expects one scalar gripper channel")
    values = np.interp(
        timestamps,
        _as_float_array(chunk.point_timestamps),
        source_flat[:, 0],
    )
    return values if source.ndim == 1 else values[:, None]


def _intent_from_control_poses(
    dense: TrajectoryChunkV1,
    *,
    intent_id: str,
    timestamps: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    arc_fractions: np.ndarray,
    timing_mode: str,
    timing_timestamps: np.ndarray,
    provenance: Mapping[str, Any],
) -> MotionIntentV2:
    return MotionIntentV2(
        intent_id=intent_id,
        observation_timestamp=float(dense.observation_timestamp),
        point_timestamps=timestamps,
        eef_position=positions,
        eef_orientation=orientations,
        source_frame=dense.source_frame,
        tcp_semantic_id=dense.tcp_semantic_id,
        validity_start=float(dense.validity_start),
        validity_end=float(dense.validity_end),
        timing_mode=timing_mode,
        timing_arc_fraction=arc_fractions,
        timing_timestamps=timing_timestamps,
        gripper=_gripper_at(dense, timestamps),
        provenance=dict(provenance),
    )


def load_snapshot_chunk(path: Path) -> TrajectoryChunkV1:
    """Load an already-unlocked Stage22 train snapshot without model code."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    timestamps = _as_float_array(payload["future_timestamps"])
    action = _as_float_array(payload["future_desired_eef_9d"])
    if timestamps.ndim != 1 or action.shape != (len(timestamps), 9):
        raise ValueError("snapshot must contain [N] timestamps and [N,9] EEF targets")
    _finite("snapshot timestamps", timestamps)
    _finite("snapshot EEF targets", action)
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("snapshot timestamps must be strictly increasing")
    orientations = matrix_to_quat(rot6d_rows_to_matrix(action[:, 3:]))
    provenance = dict(payload.get("provenance", {}))
    provenance.update(
        {
            "snapshot_path": str(Path(path).resolve()),
            "snapshot_schema": payload.get("schema"),
            "source_representation": "future_desired_eef_9d_native_row_packed_rot6d",
            "diagnostic_only": True,
        }
    )
    gripper = payload.get("future_gripper")
    return TrajectoryChunkV1(
        chunk_id=str(payload.get("example_id") or payload.get("target_id") or Path(path).stem),
        source="Stage22_train_snapshot",
        observation_timestamp=float(payload.get("observation_timestamp", timestamps[0])),
        point_timestamps=timestamps,
        eef_position=action[:, :3],
        eef_orientation=orientations,
        source_frame=str(payload["source_frame"]),
        tcp_semantic_id=str(payload["tcp_semantic_id"]),
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=None if gripper is None else _as_float_array(gripper),
        provenance=provenance,
    )


def build_geometry_candidate(
    dense: TrajectoryChunkV1,
    k: int,
    *,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> MotionIntentV2:
    """Build the frozen uniform normalized-SE(3)-arc geometry candidate."""

    if int(k) not in GEOMETRY_K_VALUES:
        raise ValueError(f"k must be one of {GEOMETRY_K_VALUES}")
    if dense.point_count < int(k):
        raise ValueError("dense trajectory has fewer points than requested controls")
    segments, cumulative = _arc_map(dense.eef_position, dense.eef_orientation, lambda_m_per_rad)
    total = float(cumulative[-1])
    if total <= 1e-12:
        fractions = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
        timestamps = np.linspace(dense.validity_start, dense.validity_end, int(k), dtype=np.float64)
        positions = np.repeat(dense.eef_position[:1], int(k), axis=0)
        orientations = np.repeat(dense.eef_orientation[:1], int(k), axis=0)
    else:
        fractions = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
        records = [
            _pose_time_at_arc(
                dense,
                float(fraction * total),
                segments=segments,
                cumulative=cumulative,
                lambda_m_per_rad=lambda_m_per_rad,
            )
            for fraction in fractions
        ]
        positions = np.asarray([row[0] for row in records], dtype=np.float64)
        orientations = np.asarray([row[1] for row in records], dtype=np.float64)
        timestamps = np.asarray([row[2] for row in records], dtype=np.float64)
        positions[0] = dense.eef_position[0]
        positions[-1] = dense.eef_position[-1]
        orientations[0] = dense.eef_orientation[0]
        orientations[-1] = dense.eef_orientation[-1]
        timestamps[0] = dense.point_timestamps[0]
        timestamps[-1] = dense.point_timestamps[-1]
    provenance = dict(dense.provenance)
    provenance.update(
        {
            "stage34_candidate": "UNIFORM_NORMALIZED_SE3_ARC",
            "geometry_k": int(k),
            "geometry_arc_fractions": fractions.tolist(),
            "se3_lambda_m_per_rad": float(lambda_m_per_rad),
            "interpolation": "LINEAR_XYZ_QUATERNION_SLERP",
            "knot_policy": "UNIFORM_NORMALIZED_ARC_INCLUDING_ENDPOINTS",
            "smoothing": False,
            "spline_fitting": False,
            "trajectory_optimization": False,
            "manual_knot_placement": False,
        }
    )
    return _intent_from_control_poses(
        dense,
        intent_id=f"{dense.chunk_id}:K{int(k)}",
        timestamps=timestamps,
        positions=positions,
        orientations=orientations,
        arc_fractions=fractions,
        timing_mode="T0",
        timing_timestamps=timestamps,
        provenance=provenance,
    )


def build_timing_candidate(
    dense: TrajectoryChunkV1,
    geometry: MotionIntentV2,
    timing_mode: str,
    *,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> MotionIntentV2:
    """Apply one preregistered timing law while keeping geometry unchanged."""

    if timing_mode not in TIMING_MODES:
        raise ValueError(f"timing_mode must be one of {TIMING_MODES}")
    control_u = np.linspace(0.0, 1.0, geometry.point_count, dtype=np.float64)
    source_segments, source_cumulative = _arc_map(
        dense.eef_position, dense.eef_orientation, lambda_m_per_rad
    )
    source_total = float(source_cumulative[-1])
    source_times = np.asarray(
        [
            _pose_time_at_arc(
                dense,
                float(u * source_total),
                segments=source_segments,
                cumulative=source_cumulative,
                lambda_m_per_rad=lambda_m_per_rad,
            )[2]
            for u in TIMING_KNOTS[timing_mode]
        ],
        dtype=np.float64,
    )
    if timing_mode == "T1":
        source_times = np.asarray(
            [float(dense.validity_start), float(dense.validity_end)], dtype=np.float64
        )
    control_times = _piecewise_linear(control_u, TIMING_KNOTS[timing_mode], source_times)
    control_times[0] = float(dense.validity_start)
    control_times[-1] = float(dense.validity_end)
    provenance = dict(geometry.provenance)
    provenance.update(
        {
            "stage34_candidate": "FIXED_GEOMETRY_COMPACT_TIMING",
            "timing_mode": timing_mode,
            "timing_arc_knots": TIMING_KNOTS[timing_mode].tolist(),
            "timing_knot_timestamps": source_times.tolist(),
            "timing_interpolation": "MONOTONIC_PIECEWISE_LINEAR",
            "geometry_unchanged": True,
        }
    )
    return _intent_from_control_poses(
        dense,
        intent_id=f"{geometry.intent_id}:{timing_mode}",
        timestamps=control_times,
        positions=geometry.eef_position.copy(),
        orientations=geometry.eef_orientation.copy(),
        arc_fractions=TIMING_KNOTS[timing_mode],
        timing_mode=timing_mode,
        timing_timestamps=source_times,
        provenance=provenance,
    )


def _chunk(value: MotionIntentV2 | TrajectoryChunkV1) -> TrajectoryChunkV1:
    if isinstance(value, MotionIntentV2):
        return value.to_trajectory_chunk()
    if isinstance(value, TrajectoryChunkV1):
        return value
    raise TypeError(f"expected MotionIntentV2 or TrajectoryChunkV1, got {type(value)!r}")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _speed_summary(chunk: TrajectoryChunkV1) -> dict[str, Any]:
    timestamps = _as_float_array(chunk.point_timestamps)
    dt = np.diff(timestamps)
    if np.any(dt <= 0.0):
        return {"finite": False, "monotonic_timestamps": False}
    translation_speed = np.linalg.norm(np.diff(chunk.eef_position, axis=0), axis=-1) / dt
    rotation_speed = geodesic_angle(
        quat_to_matrix(chunk.eef_orientation[:-1]), quat_to_matrix(chunk.eef_orientation[1:])
    ) / dt
    return {
        "finite": bool(np.isfinite(translation_speed).all() and np.isfinite(rotation_speed).all()),
        "monotonic_timestamps": True,
        "translational_mps": _summary(translation_speed),
        "angular_radps": _summary(rotation_speed),
    }


def evaluate_preview_equivalence(
    dense: TrajectoryChunkV1,
    candidate: MotionIntentV2 | TrajectoryChunkV1,
    *,
    case_id: str | None = None,
    representation: str | None = None,
    anchor_count: int = 101,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> dict[str, Any]:
    """Compare dense and candidate previews on a fixed full-duration grid."""

    if int(anchor_count) < 2:
        raise ValueError("anchor_count must be at least two")
    candidate_chunk = _chunk(candidate)
    anchors = np.linspace(
        float(dense.validity_start), float(dense.validity_end), int(anchor_count), dtype=np.float64
    )
    dense_positions = []
    candidate_positions = []
    dense_orientations = []
    candidate_orientations = []
    dense_clamp = []
    candidate_clamp = []
    for anchor in anchors:
        dense_preview = se3_arc_length_preview(
            dense,
            anchor_timestamp=float(anchor),
            offsets_m=DEFAULT_WBC_PREVIEW_OFFSETS_M,
            lambda_m_per_rad=lambda_m_per_rad,
        )
        candidate_preview = se3_arc_length_preview(
            candidate_chunk,
            anchor_timestamp=float(anchor),
            offsets_m=DEFAULT_WBC_PREVIEW_OFFSETS_M,
            lambda_m_per_rad=lambda_m_per_rad,
        )
        dense_positions.append(dense_preview.eef_position)
        candidate_positions.append(candidate_preview.eef_position)
        dense_orientations.append(dense_preview.eef_orientation)
        candidate_orientations.append(candidate_preview.eef_orientation)
        dense_clamp.append(dense_preview.endpoint_clamped)
        candidate_clamp.append(candidate_preview.endpoint_clamped)
    dense_positions_arr = np.asarray(dense_positions, dtype=np.float64)
    candidate_positions_arr = np.asarray(candidate_positions, dtype=np.float64)
    dense_orientations_arr = np.asarray(dense_orientations, dtype=np.float64)
    candidate_orientations_arr = np.asarray(candidate_orientations, dtype=np.float64)
    dense_clamp_arr = np.asarray(dense_clamp, dtype=bool)
    candidate_clamp_arr = np.asarray(candidate_clamp, dtype=bool)
    position_errors = np.linalg.norm(dense_positions_arr - candidate_positions_arr, axis=-1)
    rotation_errors = geodesic_angle(
        quat_to_matrix(dense_orientations_arr), quat_to_matrix(candidate_orientations_arr)
    )
    dense_progress = _arc_progress(dense, anchors, lambda_m_per_rad)
    candidate_progress = _arc_progress(candidate_chunk, anchors, lambda_m_per_rad)
    progress_errors = np.abs(dense_progress - candidate_progress)
    dense_segments, dense_cumulative = _arc_map(
        dense.eef_position, dense.eef_orientation, lambda_m_per_rad
    )
    candidate_segments, candidate_cumulative = _arc_map(
        candidate_chunk.eef_position, candidate_chunk.eef_orientation, lambda_m_per_rad
    )
    dense_total = float(dense_cumulative[-1])
    candidate_total = float(candidate_cumulative[-1])
    arc_ratio = candidate_total / dense_total if dense_total > 1e-12 else 1.0
    arc_relative_error = abs(candidate_total - dense_total) / max(dense_total, 1e-12)
    start_position_error = float(np.linalg.norm(dense.eef_position[0] - candidate_chunk.eef_position[0]))
    end_position_error = float(np.linalg.norm(dense.eef_position[-1] - candidate_chunk.eef_position[-1]))
    start_rotation_error = float(
        geodesic_angle(
            quat_to_matrix(dense.eef_orientation[0]),
            quat_to_matrix(candidate_chunk.eef_orientation[0]),
        )
    )
    end_rotation_error = float(
        geodesic_angle(
            quat_to_matrix(dense.eef_orientation[-1]),
            quat_to_matrix(candidate_chunk.eef_orientation[-1]),
        )
    )
    duration_error = abs(
        float(candidate_chunk.validity_end - candidate_chunk.validity_start)
        - float(dense.validity_end - dense.validity_start)
    )
    finite = bool(
        np.isfinite(position_errors).all()
        and np.isfinite(rotation_errors).all()
        and np.isfinite(progress_errors).all()
    )
    monotonic = bool(
        np.all(np.diff(dense.point_timestamps) > 0.0)
        and np.all(np.diff(candidate_chunk.point_timestamps) > 0.0)
    )
    clamp_agreement = float(np.mean(dense_clamp_arr == candidate_clamp_arr))
    start_end_exact = bool(
        start_position_error <= 1e-10
        and end_position_error <= 1e-10
        and start_rotation_error <= 1e-10
        and end_rotation_error <= 1e-10
    )
    duration_exact = bool(duration_error <= 1e-10)
    passes_limits = bool(
        finite
        and monotonic
        and _summary(position_errors)["p95"] < DEFAULT_PREVIEW_POSITION_P95_LIMIT_M
        and _summary(rotation_errors)["p95"] < DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD
        and clamp_agreement >= DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN
        and start_end_exact
        and duration_exact
    )
    result: dict[str, Any] = {
        "case_id": case_id or dense.chunk_id,
        "representation": representation or candidate_chunk.chunk_id,
        "anchor_count": int(anchor_count),
        "preview_position_error_m": _summary(position_errors),
        "preview_rotation_error_rad": _summary(rotation_errors),
        "endpoint_clamp_agreement": clamp_agreement,
        "se3_arc_length_m": {"dense": dense_total, "candidate": candidate_total},
        "se3_arc_length_ratio": float(arc_ratio),
        "se3_arc_length_relative_error": float(arc_relative_error),
        "time_to_arc_progress_error": _summary(progress_errors),
        "start_end_pose_exact": start_end_exact,
        "start_position_error_m": start_position_error,
        "end_position_error_m": end_position_error,
        "start_rotation_error_rad": start_rotation_error,
        "end_rotation_error_rad": end_rotation_error,
        "duration_exact": duration_exact,
        "duration_error_s": float(duration_error),
        "finite": finite,
        "monotonic_timestamps": monotonic,
        "dense_speed_profile": _speed_summary(dense),
        "candidate_speed_profile": _speed_summary(candidate_chunk),
        "passes_limits": passes_limits,
        "_position_errors": position_errors,
        "_rotation_errors": rotation_errors,
        "_progress_errors": progress_errors,
        "_endpoint_clamp_agreement": clamp_agreement,
        "_arc_relative_error": arc_relative_error,
    }
    return result


def public_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop internal arrays and convert NumPy scalars for JSON output."""

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items() if not str(key).startswith("_")}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return convert({key: value for key, value in result.items() if not str(key).startswith("_")})


def aggregate_evaluations(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    representation: str | None = None,
) -> dict[str, Any]:
    """Aggregate case-level diagnostics without discarding any candidate row."""

    rows = list(evaluations)
    if not rows:
        raise ValueError("at least one evaluation is required")
    position_errors = np.concatenate([_as_float_array(row["_position_errors"]).reshape(-1) for row in rows])
    rotation_errors = np.concatenate([_as_float_array(row["_rotation_errors"]).reshape(-1) for row in rows])
    progress_errors = np.concatenate([_as_float_array(row["_progress_errors"]).reshape(-1) for row in rows])
    arc_errors = np.asarray([float(row["_arc_relative_error"]) for row in rows], dtype=np.float64)
    clamp = np.asarray([float(row["_endpoint_clamp_agreement"]) for row in rows], dtype=np.float64)
    finite = bool(all(bool(row["finite"]) for row in rows))
    monotonic = bool(all(bool(row["monotonic_timestamps"]) for row in rows))
    start_end_exact = bool(all(bool(row["start_end_pose_exact"]) for row in rows))
    duration_exact = bool(all(bool(row["duration_exact"]) for row in rows))
    position_summary = _summary(position_errors)
    rotation_summary = _summary(rotation_errors)
    pass_limits = bool(
        finite
        and monotonic
        and position_summary["p95"] < DEFAULT_PREVIEW_POSITION_P95_LIMIT_M
        and rotation_summary["p95"] < DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD
        and float(np.min(clamp)) >= DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN
        and start_end_exact
        and duration_exact
        and all(bool(row["passes_limits"]) for row in rows)
    )
    return {
        "representation": representation or rows[0].get("representation"),
        "case_count": len(rows),
        "anchor_count": sorted({int(row["anchor_count"]) for row in rows}),
        "preview_position_error_m": position_summary,
        "preview_rotation_error_rad": rotation_summary,
        "endpoint_clamp_agreement": {
            "mean": float(np.mean(clamp)),
            "min": float(np.min(clamp)),
            "max": float(np.max(clamp)),
        },
        "se3_arc_length_relative_error": _summary(arc_errors),
        "se3_arc_length_ratio": _summary(
            np.asarray([float(row["se3_arc_length_ratio"]) for row in rows], dtype=np.float64)
        ),
        "time_to_arc_progress_error": _summary(progress_errors),
        "finite": finite,
        "monotonic_timestamps": monotonic,
        "start_end_pose_exact": start_end_exact,
        "duration_exact": duration_exact,
        "passes_limits": pass_limits,
        "per_case": [public_evaluation(row) for row in rows],
    }


__all__ = [
    "DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN",
    "DEFAULT_PREVIEW_POSITION_P95_LIMIT_M",
    "DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD",
    "DEFAULT_SE3_LAMBDA_M_PER_RAD",
    "DEFAULT_WBC_PREVIEW_OFFSETS_M",
    "GEOMETRY_K_VALUES",
    "TIMING_KNOTS",
    "TIMING_MODES",
    "aggregate_evaluations",
    "build_geometry_candidate",
    "build_timing_candidate",
    "evaluate_preview_equivalence",
    "load_snapshot_chunk",
    "public_evaluation",
]
