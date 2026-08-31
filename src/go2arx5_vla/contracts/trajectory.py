"""Simulator-independent trajectory contracts.

The canonical internal orientation is a unit quaternion in ``xyzw`` order.
This is deliberately explicit: WBC rot6d is a derived wire representation,
not the storage representation used by the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.so3 import matrix_to_quat, quat_to_matrix, rot6d_to_matrix, rotation_matrix_to_rot6d


def _array(value: Any, *, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class TrajectoryChunkV1:
    """A timestamped EEF trajectory chunk.

    ``point_timestamps`` are absolute monotonic timestamps in the same clock
    domain as ``observation_timestamp``.  ``source_frame`` and
    ``tcp_semantic_id`` are required even for synthetic benchmarks so that a
    later adapter cannot silently mix calibration domains.
    """

    chunk_id: str
    source: str
    observation_timestamp: float
    point_timestamps: np.ndarray
    eef_position: np.ndarray
    eef_orientation: np.ndarray
    source_frame: str
    tcp_semantic_id: str
    validity_start: float
    validity_end: float
    gripper: Optional[np.ndarray] = None
    eef_rot6d: Optional[np.ndarray] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ts = _array(self.point_timestamps)
        pos = _array(self.eef_position)
        ori = _array(self.eef_orientation)
        grip = None if self.gripper is None else _array(self.gripper)
        rot6d = None if self.eef_rot6d is None else _array(self.eef_rot6d)
        if ts.ndim != 1 or ts.size == 0:
            raise ValueError("point_timestamps must be non-empty [N]")
        if pos.shape != (ts.size, 3):
            raise ValueError(f"eef_position must be [{ts.size},3]")
        if ori.shape != (ts.size, 4):
            raise ValueError(f"eef_orientation must be [{ts.size},4] in xyzw order")
        if grip is not None and grip.shape not in {(ts.size,), (ts.size, 1)}:
            raise ValueError(f"gripper must be [{ts.size}] or [{ts.size},1]")
        if rot6d is not None and rot6d.shape != (ts.size, 6):
            raise ValueError(f"eef_rot6d must be [{ts.size},6]")
        if np.any(np.diff(ts) < 0.0):
            raise ValueError("point_timestamps must be non-decreasing")
        if float(self.validity_end) < float(self.validity_start):
            raise ValueError("validity_end must be >= validity_start")
        for name, value in (("point_timestamps", ts), ("eef_position", pos), ("eef_orientation", ori)):
            _finite(name, value)
        if grip is not None:
            _finite("gripper", grip)
        if rot6d is not None:
            _finite("eef_rot6d", rot6d)
            # Validate the supplied representation without silently using it
            # as the canonical orientation.
            rot6d_to_matrix(rot6d)
        norms = np.linalg.norm(ori, axis=-1)
        if np.any(norms < 1e-10):
            raise ValueError("eef_orientation contains a zero quaternion")
        ori = ori / norms[:, None]
        object.__setattr__(self, "point_timestamps", ts)
        object.__setattr__(self, "eef_position", pos)
        object.__setattr__(self, "eef_orientation", ori)
        object.__setattr__(self, "gripper", grip)
        object.__setattr__(self, "eef_rot6d", rot6d)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def point_count(self) -> int:
        return int(self.point_timestamps.size)

    @property
    def rot6d(self) -> np.ndarray:
        """Return WBC/GR00T-compatible first-two-column rot6d."""
        return rotation_matrix_to_rot6d(quat_to_matrix(self.eef_orientation))

    @classmethod
    def from_rot6d(
        cls,
        *,
        chunk_id: str,
        source: str,
        observation_timestamp: float,
        point_timestamps: Any,
        eef_position: Any,
        eef_rot6d: Any,
        source_frame: str,
        tcp_semantic_id: str,
        validity_start: float,
        validity_end: float,
        gripper: Optional[Any] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TrajectoryChunkV1":
        matrices = rot6d_to_matrix(_array(eef_rot6d))
        return cls(
            chunk_id=chunk_id,
            source=source,
            observation_timestamp=observation_timestamp,
            point_timestamps=point_timestamps,
            eef_position=eef_position,
            eef_orientation=matrix_to_quat(matrices),
            source_frame=source_frame,
            tcp_semantic_id=tcp_semantic_id,
            validity_start=validity_start,
            validity_end=validity_end,
            gripper=gripper,
            eef_rot6d=_array(eef_rot6d),
            provenance={} if provenance is None else provenance,
        )

    def transformed(self, source_to_target: np.ndarray, *, target_frame: str) -> "TrajectoryChunkV1":
        """Apply a rigid frame transform to every EEF pose."""
        tf = _array(source_to_target)
        if tf.shape != (4, 4):
            raise ValueError("source_to_target must be [4,4]")
        if not np.allclose(tf[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("source_to_target must be homogeneous")
        r = tf[:3, :3]
        p = (r @ self.eef_position.T).T + tf[:3, 3]
        q = matrix_to_quat(np.einsum("ij,njk->nik", r, quat_to_matrix(self.eef_orientation)))
        provenance = dict(self.provenance)
        provenance["frame_transform"] = tf.tolist()
        return TrajectoryChunkV1(
            chunk_id=self.chunk_id,
            source=self.source,
            observation_timestamp=self.observation_timestamp,
            point_timestamps=self.point_timestamps.copy(),
            eef_position=p,
            eef_orientation=q,
            source_frame=target_frame,
            tcp_semantic_id=self.tcp_semantic_id,
            validity_start=self.validity_start,
            validity_end=self.validity_end,
            gripper=None if self.gripper is None else self.gripper.copy(),
            provenance=provenance,
        )


@dataclass(frozen=True)
class WBCPreviewV1:
    """The delivered WBC's external nine-point XYZ+rot6d preview."""

    anchor_timestamp: float
    offsets_m: np.ndarray
    eef_position: np.ndarray
    eef_rot6d: np.ndarray
    source_frame: str
    tcp_semantic_id: str
    endpoint_clamped: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        offsets = _array(self.offsets_m)
        pos = _array(self.eef_position)
        rot = _array(self.eef_rot6d)
        clamp = np.asarray(self.endpoint_clamped, dtype=bool)
        if offsets.ndim != 1 or pos.shape != (offsets.size, 3) or rot.shape != (offsets.size, 6):
            raise ValueError("WBCPreviewV1 shapes must be [K], [K,3], [K,6]")
        if clamp.shape != (offsets.size,):
            raise ValueError("endpoint_clamped must be [K]")
        if np.any(np.diff(offsets) < 0.0) or np.any(offsets < 0.0):
            raise ValueError("preview offsets must be sorted and non-negative")
        _finite("preview position", pos)
        _finite("preview rot6d", rot)
        rot6d_to_matrix(rot)
        object.__setattr__(self, "offsets_m", offsets)
        object.__setattr__(self, "eef_position", pos)
        object.__setattr__(self, "eef_rot6d", rot)
        object.__setattr__(self, "endpoint_clamped", clamp)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def xyz_rot6d(self) -> np.ndarray:
        return np.concatenate([self.eef_position, self.eef_rot6d], axis=-1)


@dataclass(frozen=True)
class GripperCommandV1:
    """Separate gripper channel; the delivered 18D WBC does not own it."""

    timestamp: float
    normalized_command: float
    stale: bool
    raw_source_convention: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp) or not np.isfinite(self.normalized_command):
            raise ValueError("gripper timestamp and command must be finite")
        object.__setattr__(self, "provenance", dict(self.provenance))
