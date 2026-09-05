"""Side-by-side experimental task-space motion contract for Stage34.

``MotionIntentV2`` is deliberately independent of the production/V1 action
ABI.  It carries a compact, timestamped EEF trajectory plus an explicit
normalized arc-fraction/time law.  The native WBC row-packed rot6d wire format
is still produced only by the existing preview adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix
from .trajectory import TrajectoryChunkV1


MOTION_INTENT_V2_TIMING_MODES = frozenset({"T0", "T1", "T3", "T5", "T9"})
DEFAULT_MOTION_INTENT_V2_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
DEFAULT_MOTION_INTENT_V2_TCP = "x5_link6_gripper_center"
DEFAULT_MOTION_INTENT_V2_ANCHOR = "RUNTIME_MEASURED_CURRENT_EEF"
DEFAULT_MOTION_INTENT_V2_LAMBDA_M_PER_RAD = 0.15


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalized_arc_fraction(
    positions: np.ndarray,
    orientations: np.ndarray,
    lambda_m_per_rad: float = DEFAULT_MOTION_INTENT_V2_LAMBDA_M_PER_RAD,
) -> np.ndarray:
    """Return the canonical SE(3) arc fraction of each supplied pose."""

    if lambda_m_per_rad <= 0.0 or not np.isfinite(lambda_m_per_rad):
        raise ValueError("lambda_m_per_rad must be finite and positive")
    translation = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    rotation = geodesic_angle(
        quat_to_matrix(orientations[:-1]), quat_to_matrix(orientations[1:])
    )
    segments = np.sqrt(translation * translation + (lambda_m_per_rad * rotation) ** 2)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(positions), dtype=np.float64)
    return cumulative / total


@dataclass(frozen=True)
class MotionIntentV2:
    """Experimental timestamped EEF motion intent.

    The arrays in ``eef_position``/``eef_orientation`` are the delivered
    control poses, not a model output tensor.  ``timing_arc_fraction`` and
    ``timing_timestamps`` describe the monotonic progress law used to place
    those poses in time.  A runtime-measured current EEF anchor is represented
    by metadata and is never inferred from a VLA target here.
    """

    intent_id: str
    observation_timestamp: float
    point_timestamps: np.ndarray
    eef_position: np.ndarray
    eef_orientation: np.ndarray
    source_frame: str
    tcp_semantic_id: str
    validity_start: float
    validity_end: float
    timing_mode: str
    timing_arc_fraction: Optional[np.ndarray] = None
    timing_timestamps: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    anchor_semantics: str = DEFAULT_MOTION_INTENT_V2_ANCHOR
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.intent_id).strip():
            raise ValueError("intent_id must be non-empty")
        if not str(self.source_frame).strip() or not str(self.tcp_semantic_id).strip():
            raise ValueError("source_frame and tcp_semantic_id must be non-empty")
        if not str(self.anchor_semantics).strip():
            raise ValueError("anchor_semantics must be non-empty")
        if self.timing_mode not in MOTION_INTENT_V2_TIMING_MODES:
            raise ValueError(f"unsupported timing_mode: {self.timing_mode}")

        ts = np.asarray(self.point_timestamps, dtype=np.float64)
        pos = np.asarray(self.eef_position, dtype=np.float64)
        ori = np.asarray(self.eef_orientation, dtype=np.float64)
        if ts.ndim != 1 or ts.size < 2:
            raise ValueError("point_timestamps must contain at least two points")
        if pos.shape != (ts.size, 3):
            raise ValueError(f"eef_position must be [{ts.size},3]")
        if ori.shape != (ts.size, 4):
            raise ValueError(f"eef_orientation must be [{ts.size},4] in xyzw order")
        if np.any(np.diff(ts) <= 0.0):
            raise ValueError("MotionIntentV2 point_timestamps must be strictly increasing")
        for name, value in (("point_timestamps", ts), ("eef_position", pos), ("eef_orientation", ori)):
            _finite(name, value)

        observation = float(self.observation_timestamp)
        start = float(self.validity_start)
        end = float(self.validity_end)
        if not np.isfinite(observation) or not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("observation and validity timestamps must be finite")
        if end <= start:
            raise ValueError("validity_end must be greater than validity_start")
        if not np.isclose(ts[0], start, rtol=0.0, atol=1e-10) or not np.isclose(
            ts[-1], end, rtol=0.0, atol=1e-10
        ):
            raise ValueError("control timestamps must span the validity interval exactly")

        norms = np.linalg.norm(ori, axis=-1)
        if np.any(norms < 1e-10):
            raise ValueError("eef_orientation contains a zero quaternion")
        ori = ori / norms[:, None]

        grip = None if self.gripper is None else np.asarray(self.gripper, dtype=np.float64)
        if grip is not None:
            if grip.shape not in {(ts.size,), (ts.size, 1)}:
                raise ValueError(f"gripper must be [{ts.size}] or [{ts.size},1]")
            _finite("gripper", grip)

        arc = None if self.timing_arc_fraction is None else np.asarray(
            self.timing_arc_fraction, dtype=np.float64
        )
        law_ts = None if self.timing_timestamps is None else np.asarray(
            self.timing_timestamps, dtype=np.float64
        )
        if (arc is None) != (law_ts is None):
            raise ValueError("timing_arc_fraction and timing_timestamps must be supplied together")
        if arc is None:
            arc = np.asarray([0.0, 1.0], dtype=np.float64)
            law_ts = np.asarray([start, end], dtype=np.float64)
        if arc.ndim != 1 or law_ts.ndim != 1 or arc.size < 2 or arc.size != law_ts.size:
            raise ValueError("timing law arrays must be one-dimensional and have equal length")
        _finite("timing_arc_fraction", arc)
        _finite("timing_timestamps", law_ts)
        if not np.isclose(arc[0], 0.0, rtol=0.0, atol=1e-10) or not np.isclose(
            arc[-1], 1.0, rtol=0.0, atol=1e-10
        ):
            raise ValueError("timing arc fraction must start at 0 and end at 1")
        if np.any(np.diff(arc) <= 0.0) or np.any(arc < 0.0) or np.any(arc > 1.0):
            raise ValueError("timing arc fraction must be strictly increasing in [0,1]")
        if np.any(np.diff(law_ts) <= 0.0):
            raise ValueError("timing timestamps must be strictly increasing")
        if not np.isclose(law_ts[0], start, rtol=0.0, atol=1e-10) or not np.isclose(
            law_ts[-1], end, rtol=0.0, atol=1e-10
        ):
            raise ValueError("timing law must span the validity interval exactly")
        if law_ts[0] < ts[0] - 1e-10 or law_ts[-1] > ts[-1] + 1e-10:
            raise ValueError("timing law lies outside the control timestamp interval")

        object.__setattr__(self, "point_timestamps", ts)
        object.__setattr__(self, "eef_position", pos)
        object.__setattr__(self, "eef_orientation", ori)
        object.__setattr__(self, "timing_arc_fraction", arc)
        object.__setattr__(self, "timing_timestamps", law_ts)
        object.__setattr__(self, "gripper", grip)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def point_count(self) -> int:
        return int(self.point_timestamps.size)

    @classmethod
    def from_trajectory_chunk(
        cls,
        chunk: TrajectoryChunkV1,
        *,
        intent_id: Optional[str] = None,
        timing_mode: str = "T0",
        timing_arc_fraction: Optional[np.ndarray] = None,
        timing_timestamps: Optional[np.ndarray] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "MotionIntentV2":
        arc = (
            _normalized_arc_fraction(chunk.eef_position, chunk.eef_orientation)
            if timing_arc_fraction is None
            else np.asarray(timing_arc_fraction, dtype=np.float64)
        )
        law_ts = (
            np.asarray(chunk.point_timestamps, dtype=np.float64)
            if timing_timestamps is None
            else np.asarray(timing_timestamps, dtype=np.float64)
        )
        metadata = dict(chunk.provenance)
        metadata.update({"contract": "MotionIntentV2", "source_chunk_id": chunk.chunk_id})
        if provenance is not None:
            metadata.update(dict(provenance))
        return cls(
            intent_id=str(intent_id or chunk.chunk_id),
            observation_timestamp=float(chunk.observation_timestamp),
            point_timestamps=np.asarray(chunk.point_timestamps, dtype=np.float64),
            eef_position=np.asarray(chunk.eef_position, dtype=np.float64),
            eef_orientation=np.asarray(chunk.eef_orientation, dtype=np.float64),
            source_frame=chunk.source_frame,
            tcp_semantic_id=chunk.tcp_semantic_id,
            validity_start=float(chunk.validity_start),
            validity_end=float(chunk.validity_end),
            timing_mode=timing_mode,
            timing_arc_fraction=arc,
            timing_timestamps=law_ts,
            gripper=None if chunk.gripper is None else np.asarray(chunk.gripper, dtype=np.float64),
            anchor_semantics=DEFAULT_MOTION_INTENT_V2_ANCHOR,
            provenance=metadata,
        )

    def to_trajectory_chunk(
        self,
        *,
        expected_source_frame: Optional[str] = None,
        expected_tcp_semantic_id: Optional[str] = None,
    ) -> TrajectoryChunkV1:
        """Materialize a V1-compatible timestamped chunk with fail-closed guards."""

        if expected_source_frame is not None and self.source_frame != expected_source_frame:
            raise ValueError(
                f"source_frame mismatch: expected {expected_source_frame}, got {self.source_frame}"
            )
        if expected_tcp_semantic_id is not None and self.tcp_semantic_id != expected_tcp_semantic_id:
            raise ValueError(
                "tcp_semantic_id mismatch: expected "
                f"{expected_tcp_semantic_id}, got {self.tcp_semantic_id}"
            )
        metadata = dict(self.provenance)
        metadata.update(
            {
                "contract": "MotionIntentV2",
                "intent_id": self.intent_id,
                "timing_mode": self.timing_mode,
                "anchor_semantics": self.anchor_semantics,
            }
        )
        return TrajectoryChunkV1(
            chunk_id=self.intent_id,
            source="MotionIntentV2",
            observation_timestamp=self.observation_timestamp,
            point_timestamps=self.point_timestamps.copy(),
            eef_position=self.eef_position.copy(),
            eef_orientation=self.eef_orientation.copy(),
            source_frame=self.source_frame,
            tcp_semantic_id=self.tcp_semantic_id,
            validity_start=self.validity_start,
            validity_end=self.validity_end,
            gripper=None if self.gripper is None else self.gripper.copy(),
            provenance=metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable canonical representation."""

        return {
            "schema": "MotionIntentV2",
            "intent_id": self.intent_id,
            "observation_timestamp": float(self.observation_timestamp),
            "point_timestamps": self.point_timestamps.tolist(),
            "eef_position": self.eef_position.tolist(),
            "eef_orientation": self.eef_orientation.tolist(),
            "source_frame": self.source_frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "validity_start": float(self.validity_start),
            "validity_end": float(self.validity_end),
            "timing_mode": self.timing_mode,
            "timing_arc_fraction": self.timing_arc_fraction.tolist(),
            "timing_timestamps": self.timing_timestamps.tolist(),
            "gripper": None if self.gripper is None else self.gripper.tolist(),
            "anchor_semantics": self.anchor_semantics,
            "provenance": dict(self.provenance),
        }


__all__ = [
    "DEFAULT_MOTION_INTENT_V2_ANCHOR",
    "DEFAULT_MOTION_INTENT_V2_FRAME",
    "DEFAULT_MOTION_INTENT_V2_LAMBDA_M_PER_RAD",
    "DEFAULT_MOTION_INTENT_V2_TCP",
    "MOTION_INTENT_V2_TIMING_MODES",
    "MotionIntentV2",
]
