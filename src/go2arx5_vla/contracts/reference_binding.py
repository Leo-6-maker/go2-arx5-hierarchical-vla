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

    @classmethod
    def from_external_chunk(
        cls,
        chunk: TrajectoryChunkV1,
        *,
        episode_id: str,
        placement_profile: PlacementProfileV1 | None = None,
    ) -> "EpisodeReferenceBindingV1":
        """Create an explicit identity binding for an absolute external chunk.

        A GR00T chunk is already expressed in the declared policy frame, so it
        must not be re-anchored using the native shape-placement operation.
        This helper still gives the offline interface the same named episode
        binding boundary as benchmark references.  ``bind_external_chunk``
        preserves the original absolute poses and parallel gripper channel;
        the identity transforms here are provenance, not a claim about a
        physical reset state.
        """

        identity = np.eye(4, dtype=np.float64)
        shape = TrajectoryShapeV1.from_chunk(
            chunk,
            parent_id=chunk.chunk_id,
            family="EXTERNAL_GR00T_ABSOLUTE_EEF",
            task_frame_semantics="EXTERNAL_ABSOLUTE_IDENTITY",
            metadata={"external_absolute_chunk": True},
        )
        profile = placement_profile or PlacementProfileV1(
            profile_id=f"{chunk.chunk_id}-external-identity",
            intent="CENTERED",
            status="EXTERNAL_REFERENCE_IDENTITY_BINDING",
            provenance={"not_a_native_placement": True},
        )
        return cls(
            episode_id=str(episode_id),
            reset_timestamp=float(chunk.observation_timestamp),
            native_base_pose=identity,
            native_shoulder_pose=identity,
            native_reach_anchor_pose=identity,
            native_eef_semantic_id=chunk.tcp_semantic_id,
            tcp_semantic_id=chunk.tcp_semantic_id,
            T_world_anchor=identity,
            T_anchor_task=identity,
            T_link_tcp=identity,
            source_shape=shape,
            placement_profile=profile,
            native_generator_reference_semantics="EXTERNAL_ABSOLUTE_EEF_NO_REANCHOR",
            policy_reference_frame=chunk.source_frame,
            binding_semantics="EXTERNAL_ABSOLUTE_IDENTITY",
            provenance={
                "binding_role": "offline_interface_seal",
                "source_chunk_id": chunk.chunk_id,
                "source_chunk_source": chunk.source,
            },
        )

    def bind_external_chunk(
        self,
        chunk: TrajectoryChunkV1,
        *,
        source: str = "EpisodeReferenceBindingV1",
    ) -> TrajectoryChunkV1:
        """Attach this binding's identity to an already-absolute chunk.

        This method intentionally performs no pose correction, clipping, or
        native reset anchoring.  It is the safe path for the Stage17 actual
        model offline seal.
        """

        if chunk.source_frame != self.policy_reference_frame:
            raise ValueError(
                f"external chunk frame {chunk.source_frame!r} does not match binding frame "
                f"{self.policy_reference_frame!r}"
            )
        if chunk.tcp_semantic_id != self.tcp_semantic_id:
            raise ValueError(
                f"external chunk TCP {chunk.tcp_semantic_id!r} does not match binding TCP "
                f"{self.tcp_semantic_id!r}"
            )
        if not np.isclose(chunk.observation_timestamp, self.reset_timestamp, atol=1e-8):
            raise ValueError("external chunk observation timestamp does not match binding reset timestamp")
        provenance = {
            **dict(chunk.provenance),
            **dict(self.provenance),
            "binding_contract": "EpisodeReferenceBindingV1",
            "episode_id": self.episode_id,
            "binding_semantics": self.binding_semantics,
            "external_absolute_no_reanchor": True,
        }
        return TrajectoryChunkV1(
            chunk_id=chunk.chunk_id,
            source=source,
            observation_timestamp=chunk.observation_timestamp,
            point_timestamps=chunk.point_timestamps.copy(),
            eef_position=chunk.eef_position.copy(),
            eef_orientation=chunk.eef_orientation.copy(),
            source_frame=chunk.source_frame,
            tcp_semantic_id=chunk.tcp_semantic_id,
            validity_start=chunk.validity_start,
            validity_end=chunk.validity_end,
            gripper=None if chunk.gripper is None else chunk.gripper.copy(),
            eef_rot6d=None if chunk.eef_rot6d is None else chunk.eef_rot6d.copy(),
            provenance=provenance,
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
