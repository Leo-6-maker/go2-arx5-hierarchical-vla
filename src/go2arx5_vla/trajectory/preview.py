"""Delivered WBC spatial SE(3) preview semantics."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1, WBCPreviewV1
from .so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rotation_matrix_to_rot6d_rows,
)

DEFAULT_WBC_PREVIEW_OFFSETS_M = np.asarray(
    [0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500], dtype=np.float64
)
DEFAULT_SE3_LAMBDA_M_PER_RAD = 0.15


def _arc_map(positions: np.ndarray, quats: np.ndarray, lambda_m_per_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    r = quat_to_matrix(quats)
    translation = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    rotation = geodesic_angle(r[:-1], r[1:])
    # This is the metric used by the delivered native implementation
    # (modules/trajectory.py::to_canonical): translation and the scaled
    # rotational increment are orthogonal components of one SE(3) increment.
    segments = np.sqrt(translation * translation + (float(lambda_m_per_rad) * rotation) ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    return segments, cumulative


def _time_to_arc(times: np.ndarray, segments: np.ndarray, cumulative: np.ndarray, t: float) -> float:
    if t <= times[0]:
        return 0.0
    if t >= times[-1]:
        return float(cumulative[-1])
    i = int(np.searchsorted(times, t, side="right") - 1)
    dt = times[i + 1] - times[i]
    fraction = 0.0 if dt <= 0.0 else float((t - times[i]) / dt)
    return float(cumulative[i] + fraction * segments[i])


def _pose_at_arc(
    positions: np.ndarray,
    quats: np.ndarray,
    segments: np.ndarray,
    cumulative: np.ndarray,
    s: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    total = float(cumulative[-1])
    clamped = s >= total - eps
    if s <= eps:
        return positions[0].copy(), quats[0].copy(), False
    if s >= total:
        return positions[-1].copy(), quats[-1].copy(), True
    # Skip zero-length segments. This keeps repeated points and stationary
    # holds well-defined without pretending they have spatial progress.
    i = int(np.searchsorted(cumulative, s, side="right") - 1)
    while i < segments.size and segments[i] <= eps:
        i += 1
    if i >= segments.size:
        return positions[-1].copy(), quats[-1].copy(), True
    fraction = float((s - cumulative[i]) / segments[i])
    fraction = float(np.clip(fraction, 0.0, 1.0))
    p = (1.0 - fraction) * positions[i] + fraction * positions[i + 1]
    q = quat_slerp(quats[i], quats[i + 1], fraction)
    return p, q, clamped


def se3_arc_length_preview(
    chunk: TrajectoryChunkV1,
    *,
    anchor_timestamp: float | None = None,
    offsets_m: np.ndarray = DEFAULT_WBC_PREVIEW_OFFSETS_M,
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD,
) -> WBCPreviewV1:
    """Extract the delivered nine-point preview from a timestamped chunk.

    The offsets are metres in the composite SE(3) arc metric used by the
    delivered artifact: ``sqrt(translation_m**2 + (lambda * rotation_rad)**2)``.
    A purely
    rotational path therefore remains supported; a completely stationary path
    returns the held pose and marks all points as endpoint-clamped.
    """
    times = np.asarray(chunk.point_timestamps, dtype=np.float64)
    pos = np.asarray(chunk.eef_position, dtype=np.float64)
    q = np.asarray(chunk.eef_orientation, dtype=np.float64)
    offsets = np.asarray(offsets_m, dtype=np.float64)
    if offsets.ndim != 1 or np.any(offsets < 0.0) or np.any(np.diff(offsets) < 0.0):
        raise ValueError("offsets_m must be sorted and non-negative")
    if len(offsets) > 9:
        raise ValueError("native WBC preview supports at most nine offsets")
    if lambda_m_per_rad <= 0.0:
        raise ValueError("lambda_m_per_rad must be positive")
    segments, cumulative = _arc_map(pos, q, lambda_m_per_rad)
    anchor = float(chunk.observation_timestamp if anchor_timestamp is None else anchor_timestamp)
    s0 = _time_to_arc(times, segments, cumulative, anchor)
    preview_pos = []
    preview_q = []
    clamped = []
    for offset in offsets:
        p, qq, was_clamped = _pose_at_arc(pos, q, segments, cumulative, s0 + float(offset))
        preview_pos.append(p)
        preview_q.append(qq)
        clamped.append(was_clamped)
    preview_pos_arr = np.asarray(preview_pos)
    preview_q_arr = np.asarray(preview_q)
    provenance = dict(chunk.provenance)
    provenance.update(
        {
            "preview_parameterization": "SE3_ARC_LENGTH",
            "lambda_m_per_rad": float(lambda_m_per_rad),
            "endpoint_policy": "CLAMP_FINAL_POSE",
            "stationary_policy": "HOLD_POSE",
        }
    )
    return WBCPreviewV1(
        anchor_timestamp=anchor,
        offsets_m=offsets.copy(),
        eef_position=preview_pos_arr,
        # The delivered native controller uses the archived PyTorch3D
        # compatibility convention: first two matrix rows, flattened.  The
        # canonical chunk keeps SO(3)/quaternion semantics and is converted
        # here at the WBC wire boundary.
        eef_rot6d=rotation_matrix_to_rot6d_rows(quat_to_matrix(preview_q_arr)),
        source_frame=chunk.source_frame,
        tcp_semantic_id=chunk.tcp_semantic_id,
        endpoint_clamped=np.asarray(clamped, dtype=bool),
        provenance=provenance,
    )


def arc_length_preview_from_grid(
    s_grid: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    *,
    s_current: float,
    horizon_m: float = 0.5,
    offsets_m: np.ndarray = DEFAULT_WBC_PREVIEW_OFFSETS_M,
    source_frame: str = "NATIVE_WORLD_ALIGNED",
    tcp_semantic_id: str = "x5_link6_gripper_center",
    provenance: dict | None = None,
) -> WBCPreviewV1:
    """Match ``TrajectoryBatch.sample_preview`` on a canonical Gamma grid.

    The delivered native runtime first resamples a trajectory to an ascending
    arc-length ``Gamma`` grid and then samples preview points by linear
    position interpolation and quaternion SLERP in that grid.  This helper is
    intentionally separate from :func:`se3_arc_length_preview`, which builds
    an arc map from a timestamped chunk.  Using this grid-level form avoids
    accidentally comparing two different discretization procedures during
    the Stage 15 native self-consistency audit.
    """
    s = np.asarray(s_grid, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    q = np.asarray(quaternions, dtype=np.float64)
    offsets = np.asarray(offsets_m, dtype=np.float64)
    if s.ndim != 1 or len(s) < 2 or p.shape != (len(s), 3) or q.shape != (len(s), 4):
        raise ValueError("grid preview expects s [M], position [M,3], quaternion [M,4]")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("grid preview inputs must be finite")
    if np.any(np.diff(s) <= 0.0):
        raise ValueError("s_grid must be strictly increasing")
    if offsets.ndim != 1 or np.any(offsets < 0.0) or np.any(np.diff(offsets) < 0.0):
        raise ValueError("offsets_m must be sorted and non-negative")
    if len(offsets) > 9:
        raise ValueError("native WBC preview supports at most nine offsets")
    if horizon_m < 0.0:
        raise ValueError("horizon_m must be non-negative")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    total = float(s[-1])
    query = np.clip(float(s_current) + float(horizon_m) *
                    np.asarray([0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.85, 1.0])[:len(offsets)],
                    s[0], total)
    out_p = np.empty((len(query), 3), dtype=np.float64)
    out_q = np.empty((len(query), 4), dtype=np.float64)
    endpoint = query >= total - 1e-12
    for j, value in enumerate(query):
        if value <= s[0]:
            out_p[j] = p[0]
            out_q[j] = q[0]
            continue
        if value >= total:
            out_p[j] = p[-1]
            out_q[j] = q[-1]
            continue
        i = int(np.searchsorted(s, value, side="right") - 1)
        i = max(0, min(i, len(s) - 2))
        alpha = float((value - s[i]) / max(s[i + 1] - s[i], 1e-12))
        out_p[j] = (1.0 - alpha) * p[i] + alpha * p[i + 1]
        out_q[j] = quat_slerp(q[i], q[i + 1], alpha)
    metadata = {} if provenance is None else dict(provenance)
    metadata.update({
        "preview_parameterization": "NATIVE_GAMMA_ARC_LENGTH",
        "preview_fractions": [0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.85, 1.0][:len(offsets)],
        "endpoint_policy": "CLAMP_FINAL_POSE",
    })
    return WBCPreviewV1(
        anchor_timestamp=float(s_current),
        offsets_m=offsets.copy(),
        eef_position=out_p,
        eef_rot6d=rotation_matrix_to_rot6d_rows(quat_to_matrix(out_q)),
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        endpoint_clamped=endpoint,
        provenance=metadata,
    )
