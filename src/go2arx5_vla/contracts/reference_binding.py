"""Explicit native reset/anchor reference binding contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..trajectory.relative_se3 import _as_pose, apply_binding, pose_position, pose_quaternion
from .placement import PlacementProfileV1
from .trajectory import TrajectoryChunkV1
from .trajectory_shape import TrajectoryShapeV1


@dataclass(frozen=True)
class EpisodeReferenceBindingV1:
    """Bind a shape to one native reset without hiding frame semantics.

    Homogeneous convention: ``T_A_B`` maps coordinates in frame B into frame
    A. The resulting pose is explicitly
    ``T_world_eef(t) = T_world_anchor * T_anchor_task * T_task_eef(t)``.
    """

    episode_id: str
    reset_timestamp: float
    native_base_pose: np.ndarray
    native_shoulder_pose: np.ndarray
    native_reach_anchor_pose: np.ndarray
    native_eef_semantic_id: str
    tcp_semantic_id: str
    T_world_anchor: np.ndarray
    T_anchor_task: np.ndarray
    T_link_tcp: np.ndarray
    source_shape: TrajectoryShapeV1
    placement_profile: PlacementProfileV1
    native_generator_reference_semantics: str
    policy_reference_frame: str
    binding_semantics: str = "HOMOGENEOUS_SE3"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "native_base_pose",
            "native_shoulder_pose",
            "native_reach_anchor_pose",
            "T_world_anchor",
            "T_anchor_task",
            "T_link_tcp",
        ):
            object.__setattr__(self, name, _as_pose(getattr(self, name)))
        if not np.isfinite(float(self.reset_timestamp)):
            raise ValueError("reset_timestamp must be finite")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def T_task_eef(self) -> np.ndarray:
        return self.source_shape.task_poses

    @property
    def T_world_eef(self) -> np.ndarray:
        if self.binding_semantics == "NATIVE_TRANSLATION_ONLY_WORLD_ALIGNED":
            # The delivered native loader adds the task translation to the
            # world-aligned Gamma positions but does not rotate those
            # positions.  Rotation still composes explicitly at the pose
            # level.  This is intentionally not silently represented as a
            # generic rigid transform.
            anchor = self.T_world_anchor
            task = self.T_anchor_task
            shape = self.T_task_eef
            out = np.zeros_like(shape)
            out[..., :3, :3] = np.matmul(
                np.matmul(anchor[:3, :3], task[:3, :3]), shape[..., :3, :3]
            )
            out[..., :3, 3] = (
                anchor[:3, 3][None, :]
                + task[:3, 3][None, :]
                + shape[..., :3, 3]
            )
            out[..., 3, 3] = 1.0
            return out
        return apply_binding(self.T_world_anchor, self.T_anchor_task, self.T_task_eef)

    @property
    def point_timestamps(self) -> np.ndarray:
        return self.reset_timestamp + self.source_shape.point_timestamps

    def bound_chunk(self, *, source: str = "EpisodeReferenceBindingV1") -> TrajectoryChunkV1:
        poses = self.T_world_eef
        return TrajectoryChunkV1(
            chunk_id=self.source_shape.trajectory_id,
            source=source,
            observation_timestamp=float(self.reset_timestamp),
            point_timestamps=self.point_timestamps,
            eef_position=pose_position(poses),
            eef_orientation=pose_quaternion(poses),
            source_frame=self.policy_reference_frame,
            tcp_semantic_id=self.tcp_semantic_id,
            validity_start=float(self.point_timestamps[0]),
            validity_end=float(self.point_timestamps[-1]),
            provenance={
                **dict(self.source_shape.original_provenance),
                **dict(self.provenance),
                "binding_contract": "EpisodeReferenceBindingV1",
                "episode_id": self.episode_id,
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "reset_timestamp": float(self.reset_timestamp),
            "native_eef_semantic_id": self.native_eef_semantic_id,
            "tcp_semantic_id": self.tcp_semantic_id,
            "native_generator_reference_semantics": self.native_generator_reference_semantics,
            "policy_reference_frame": self.policy_reference_frame,
            "binding_semantics": self.binding_semantics,
            "trajectory_id": self.source_shape.trajectory_id,
            "placement_profile": self.placement_profile.profile_id,
            "T_world_anchor": self.T_world_anchor.tolist(),
            "T_anchor_task": self.T_anchor_task.tolist(),
            "T_task_eef": self.T_task_eef.tolist(),
            "T_world_eef": self.T_world_eef.tolist(),
        }
