"""Deterministic transport from timestamped motion to the native WBC API.

The delivered WBC trajectory canonicalizer accepts one scalar sample interval.
Stage35, however, deliberately stores a timestamped continuous reference whose
sample times are not uniform.  This module is the sole Stage36 binding between
those two interfaces: it evaluates the source pose in source-time coordinates
on the frozen 121-point, 60 Hz grid and returns the resulting pose sequence
with the native scalar ``dt``.  It does not smooth, retime, or alter the source
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .so3 import quat_slerp


NATIVE_SAMPLE_COUNT = 121
NATIVE_DURATION_S = 2.0
NATIVE_DT_S = NATIVE_DURATION_S / float(NATIVE_SAMPLE_COUNT - 1)


@dataclass(frozen=True)
class NativeTransportResult:
    """Pose samples and audit metadata produced by native materialization."""

    timestamps: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    source_timestamps: np.ndarray
    source_position: np.ndarray
    source_quaternion: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.size)

    def audit(self) -> dict[str, Any]:
        source_dt = np.diff(self.source_timestamps)
        native_dt = np.diff(self.timestamps)
        return {
            "transport": "TIMESTAMP_PRESERVING_UNIFORM_NATIVE_GRID_V1",
            "source_sample_count": int(self.source_timestamps.size),
            "source_start_s": float(self.source_timestamps[0]),
            "source_end_s": float(self.source_timestamps[-1]),
            "source_dt_median_s": float(np.median(source_dt)),
            "source_dt_min_s": float(np.min(source_dt)),
            "source_dt_max_s": float(np.max(source_dt)),
            "native_sample_count": int(self.timestamps.size),
            "native_start_s": float(self.timestamps[0]),
            "native_end_s": float(self.timestamps[-1]),
            "native_dt_s": float(NATIVE_DT_S),
            "native_dt_uniform": bool(
                np.allclose(native_dt, NATIVE_DT_S, rtol=0.0, atol=1e-12)
            ),
            "native_grid_formula": "tau_j=2.0*j/120, j=0..120",
            "position_interpolation": "PIECEWISE_LINEAR_IN_ORIGINAL_SOURCE_TIME",
            "quaternion_interpolation": "SHORTEST_PATH_SLERP_IN_ORIGINAL_SOURCE_TIME",
        }


def _validated_inputs(
    timestamps: Any,
    position: Any,
    quaternion: Any,
    *,
    duration_s: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    if ts.ndim != 1 or ts.size < 2:
        raise ValueError("source timestamps must be a one-dimensional array with at least two points")
    if pos.shape != (ts.size, 3):
        raise ValueError("source position must have shape [N,3]")
    if quat.shape != (ts.size, 4):
        raise ValueError("source quaternion must have shape [N,4] in xyzw order")
    if not np.isfinite(ts).all() or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("native transport inputs must be finite")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("source timestamps must be strictly increasing")
    if not np.isclose(ts[0], 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("source timestamps must start at 0.0 s")
    if not np.isclose(ts[-1], float(duration_s), rtol=0.0, atol=1e-10):
        raise ValueError("source timestamps must end at the frozen duration")
    if int(sample_count) < 2 or not np.isfinite(float(duration_s)) or float(duration_s) <= 0.0:
        raise ValueError("native duration and sample count must be finite and valid")
    norms = np.linalg.norm(quat, axis=-1)
    if np.any(norms < 1e-10):
        raise ValueError("source quaternion contains a zero quaternion")
    quat = quat / norms[:, None]
    return ts.copy(), pos.copy(), quat


def materialize_timestamped_pose_to_native(
    timestamps: Any,
    position: Any,
    quaternion: Any,
    *,
    duration_s: float = NATIVE_DURATION_S,
    sample_count: int = NATIVE_SAMPLE_COUNT,
) -> NativeTransportResult:
    """Materialize one timestamped pose sequence onto the frozen native grid.

    Position interpolation is piecewise linear in the original timestamp
    domain.  Quaternion interpolation is shortest-path SLERP in that same
    domain.  The output endpoints are copied from the normalized source
    endpoints explicitly so endpoint identity is not an interpolation side
    effect.
    """

    ts, pos, quat = _validated_inputs(
        timestamps,
        position,
        quaternion,
        duration_s=float(duration_s),
        sample_count=int(sample_count),
    )
    duration = float(duration_s)
    count = int(sample_count)
    native_ts = duration * np.arange(count, dtype=np.float64) / float(count - 1)
    native_pos = np.stack(
        [np.interp(native_ts, ts, pos[:, axis]) for axis in range(3)], axis=-1
    )
    native_quat = np.empty((count, 4), dtype=np.float64)
    for index, timestamp in enumerate(native_ts):
        if timestamp <= ts[0]:
            source_index = 0
            fraction = 0.0
        elif timestamp >= ts[-1]:
            source_index = ts.size - 2
            fraction = 1.0
        else:
            source_index = int(np.searchsorted(ts, timestamp, side="right") - 1)
            source_index = max(0, min(source_index, ts.size - 2))
            source_interval = float(ts[source_index + 1] - ts[source_index])
            fraction = float((timestamp - ts[source_index]) / source_interval)
        native_quat[index] = quat_slerp(
            quat[source_index], quat[source_index + 1], fraction
        )

    native_pos[0] = pos[0]
    native_pos[-1] = pos[-1]
    native_quat[0] = quat[0]
    native_quat[-1] = quat[-1]
    if not np.isfinite(native_pos).all() or not np.isfinite(native_quat).all():
        raise ValueError("native materialization produced non-finite values")
    return NativeTransportResult(
        timestamps=native_ts,
        position=native_pos,
        quaternion=native_quat,
        source_timestamps=ts,
        source_position=pos,
        source_quaternion=quat,
    )


__all__ = [
    "NATIVE_DT_S",
    "NATIVE_DURATION_S",
    "NATIVE_SAMPLE_COUNT",
    "NativeTransportResult",
    "materialize_timestamped_pose_to_native",
]
