"""Embodiment-neutral trajectory-shape contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.relative_se3 import _as_pose, pose_from_position_quaternion, relative_poses
from ..trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix
from .trajectory import TrajectoryChunkV1


@dataclass(frozen=True)
class TrajectoryShapeV1:
    """A derived SE(3) motion expressed in an explicit task-frame origin.

    Ordinary benchmark chunks use the initial EEF pose as the task frame, so
    ``task_poses[0]`` is identity. Native-generator traces may instead use a
    reset-world-aligned trajectory-origin frame; that mode is explicit in
    ``task_frame_semantics`` and preserves the native loader's translation-only
    anchor operation.
    """

    trajectory_id: str
    parent_id: str
    point_timestamps: np.ndarray
    task_poses: np.ndarray
    task_frame_semantics: str
    family: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    original_provenance: Mapping[str, Any] = field(default_factory=dict)
    path_parameter: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        ts = np.asarray(self.point_timestamps, dtype=np.float64)
        poses = _as_pose(self.task_poses)
        if ts.ndim != 1 or len(ts) < 2 or poses.shape != (len(ts), 4, 4):
            raise ValueError("TrajectoryShapeV1 requires [N] timestamps and [N,4,4] poses")
        if np.any(np.diff(ts) <= 0.0):
            raise ValueError("shape timestamps must be strictly increasing")
        if abs(float(ts[0])) > 1e-8:
            raise ValueError("shape timestamps must start at zero")
        if not np.isfinite(ts).all():
            raise ValueError("shape timestamps must be finite")
        object.__setattr__(self, "point_timestamps", ts)
        object.__setattr__(self, "task_poses", poses)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "original_provenance", dict(self.original_provenance))
        if self.path_parameter is None:
            dp = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)
            dtheta = geodesic_angle(
                poses[:-1, :3, :3], poses[1:, :3, :3]
            )
            path = np.concatenate([[0.0], np.cumsum(np.sqrt(dp * dp + (0.15 * dtheta) ** 2))])
        else:
            path = np.asarray(self.path_parameter, dtype=np.float64)
        if path.shape != (len(ts),) or not np.all(np.isfinite(path)):
            raise ValueError("path_parameter must be finite with shape [N]")
        if abs(float(path[0])) > 1e-8 or np.any(np.diff(path) < -1e-10):
            raise ValueError("path_parameter must start at zero and be nondecreasing")
        object.__setattr__(self, "path_parameter", path)

    @property
    def point_count(self) -> int:
        return int(self.point_timestamps.size)

    @property
    def duration_s(self) -> float:
        return float(self.point_timestamps[-1])

    @property
    def position(self) -> np.ndarray:
        return self.task_poses[..., :3, 3]

    @property
    def quaternion(self) -> np.ndarray:
        return matrix_to_quat(self.task_poses[..., :3, :3])

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly contract summary.

        Arrays are intentionally serialized as lists here; large trajectory
        payloads should remain in the external NPZ/object store and be bound
        through ``original_provenance`` instead.
        """
        return {
            "schema": "TrajectoryShapeV1",
            "trajectory_id": self.trajectory_id,
            "parent_id": self.parent_id,
            "point_count": self.point_count,
            "duration_s": self.duration_s,
            "point_timestamps": self.point_timestamps.tolist(),
            "task_frame_semantics": self.task_frame_semantics,
            "family": self.family,
            "metadata": dict(self.metadata),
            "original_provenance": dict(self.original_provenance),
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: TrajectoryChunkV1,
        *,
        parent_id: Optional[str] = None,
        family: str = "UNKNOWN",
        metadata: Optional[Mapping[str, Any]] = None,
        task_frame_semantics: str = "INITIAL_EEF_POSE_FRAME",
    ) -> "TrajectoryShapeV1":
        absolute = pose_from_position_quaternion(chunk.eef_position, chunk.eef_orientation)
        relative = relative_poses(absolute)
        return cls(
            trajectory_id=chunk.chunk_id,
            parent_id=chunk.chunk_id if parent_id is None else parent_id,
            point_timestamps=chunk.point_timestamps - float(chunk.point_timestamps[0]),
            task_poses=relative,
            task_frame_semantics=task_frame_semantics,
            family=family,
            metadata={} if metadata is None else metadata,
            original_provenance={
                "chunk_source": chunk.source,
                "source_frame": chunk.source_frame,
                "tcp_semantic_id": chunk.tcp_semantic_id,
                "validity_start": chunk.validity_start,
                "validity_end": chunk.validity_end,
                **dict(chunk.provenance),
            },
        )

    @classmethod
    def from_task_poses(
        cls,
        *,
        trajectory_id: str,
        parent_id: str,
        point_timestamps: Any,
        task_poses: Any,
        task_frame_semantics: str,
        family: str,
        metadata: Optional[Mapping[str, Any]] = None,
        original_provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TrajectoryShapeV1":
        timestamps = np.asarray(point_timestamps, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("point_timestamps must be [N] with N>=2")
        timestamps = timestamps - timestamps[0]
        return cls(
            trajectory_id=trajectory_id,
            parent_id=parent_id,
            point_timestamps=timestamps,
            task_poses=task_poses,
            task_frame_semantics=task_frame_semantics,
            family=family,
            metadata={} if metadata is None else metadata,
            original_provenance={} if original_provenance is None else original_provenance,
        )
