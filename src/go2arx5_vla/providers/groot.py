"""Canonical, model-agnostic boundary for decoded GR00T actions.

The provider deliberately accepts *decoded physical* actions rather than a
Torch tensor or a GR00T ``BatchFeature``.  That keeps model/runtime details on
the provider side of the thin waist.  The public provider method returns only
``TrajectoryChunkV1``; the optional gripper values travel in the chunk's
parallel channel and are separated by the WBC adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..contracts.trajectory import GripperCommandV1, TrajectoryChunkV1
from ..trajectory.so3 import (
    matrix_to_quat,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d,
)
from .base import TrajectoryProvider


@dataclass(frozen=True)
class GR00TProviderConfig:
    """Explicit physical/time semantics for one decoded GR00T action.

    ``action_sample_period_s`` is intentionally optional.  A production
    dataset must provide point timestamps; a period is acceptable for a
    labelled runtime test only.  In particular, this config does not assume
    the synthetic Stage10 20 Hz value.
    """

    source_frame: str
    tcp_semantic_id: str
    action_sample_period_s: Optional[float] = None
    timestamp_provenance: str = "TBD_BY_ADAPTATION_DATASET"
    eef_rot6d_layout: str = "GR00T_FIRST_TWO_ROWS"
    gripper_source_convention: str = "decoded_physical_gripper"
    source: str = "GR00T_N1.7"

    def __post_init__(self) -> None:
        if not self.source_frame or not self.tcp_semantic_id:
            raise ValueError("source_frame and tcp_semantic_id are required")
        if self.eef_rot6d_layout != "GR00T_FIRST_TWO_ROWS":
            raise ValueError("only the validated GR00T first-two-rows rot6d layout is supported")
        if self.action_sample_period_s is not None:
            period = float(self.action_sample_period_s)
            if not np.isfinite(period) or period <= 0.0:
                raise ValueError("action_sample_period_s must be positive and finite")
            if self.timestamp_provenance == "TBD_BY_ADAPTATION_DATASET":
                raise ValueError(
                    "a supplied action_sample_period_s needs an explicit provenance label"
                )


class GR00TTrajectoryProvider(TrajectoryProvider):
    """Decode one physical ``[H,10]`` GR00T action sequence into a chunk.

    The Stage10 custom embodiment emits ``eef_9d`` as XYZ plus the GR00T
    first-two-rows rot6d representation, followed by one gripper value.  The
    provider converts that representation to the repository's canonical
    quaternion/first-two-columns storage.  No raw universal ``[H,132]``
    tensor is exposed downstream.
    """

    def __init__(
        self,
        decoded_action: Any,
        *,
        config: GR00TProviderConfig,
        point_timestamps: Any | None = None,
        chunk_id: str = "groot-chunk-0",
        observation_timestamp: float | None = None,
        provenance: Optional[Mapping[str, Any]] = None,
        batch_index: int = 0,
    ) -> None:
        action = np.asarray(decoded_action, dtype=np.float64)
        if action.ndim == 3:
            if not 0 <= int(batch_index) < action.shape[0]:
                raise ValueError("batch_index is outside decoded_action batch")
            action = action[int(batch_index)]
        if action.ndim != 2 or action.shape[1] != 10 or action.shape[0] == 0:
            raise ValueError("decoded_action must have shape [H,10] or [B,H,10]")
        if not np.isfinite(action).all():
            raise ValueError("decoded_action contains non-finite values")
        if not chunk_id:
            raise ValueError("chunk_id is required")

        self.config = config
        self.decoded_action = action
        self.chunk_id = str(chunk_id)
        self._point_timestamps = None if point_timestamps is None else np.asarray(point_timestamps, dtype=np.float64)
        self._observation_timestamp = None if observation_timestamp is None else float(observation_timestamp)
        self._provenance = {} if provenance is None else dict(provenance)
        self._emitted = False

        if self._point_timestamps is not None:
            if self._point_timestamps.shape != (action.shape[0],):
                raise ValueError("point_timestamps must have shape [H]")
            if not np.isfinite(self._point_timestamps).all():
                raise ValueError("point_timestamps contains non-finite values")
            if action.shape[0] > 1 and np.any(np.diff(self._point_timestamps) <= 0.0):
                raise ValueError("point_timestamps must be strictly increasing")
        if self._observation_timestamp is not None and not np.isfinite(self._observation_timestamp):
            raise ValueError("observation_timestamp must be finite")
        if self._point_timestamps is None and config.action_sample_period_s is None:
            raise ValueError(
                "point_timestamps are required when action_sample_period_s is not configured"
            )

    @property
    def point_count(self) -> int:
        return int(self.decoded_action.shape[0])

    @staticmethod
    def _timestamps(
        count: int,
        *,
        observation_timestamp: float,
        point_timestamps: np.ndarray | None,
        period_s: float | None,
    ) -> np.ndarray:
        if point_timestamps is not None:
            return point_timestamps.copy()
        if period_s is None:
            raise ValueError("an explicit timestamp source is required")
        return float(observation_timestamp) + np.arange(count, dtype=np.float64) * float(period_s)

    def build_chunk(self, *, observation_timestamp: float | None = None) -> TrajectoryChunkV1:
        """Build the canonical chunk without invoking a model or controller."""

        if observation_timestamp is None:
            if self._observation_timestamp is None:
                raise ValueError("observation_timestamp is required for this provider")
            obs = self._observation_timestamp
        else:
            obs = float(observation_timestamp)
        if not np.isfinite(obs):
            raise ValueError("observation_timestamp must be finite")

        timestamps = self._timestamps(
            self.point_count,
            observation_timestamp=obs,
            point_timestamps=self._point_timestamps,
            period_s=self.config.action_sample_period_s,
        )
        if timestamps[0] < obs - 1e-9:
            raise ValueError("point_timestamps cannot precede observation_timestamp")
        eef = self.decoded_action[:, :9]
        gripper = self.decoded_action[:, 9].copy()
        matrices = rot6d_rows_to_matrix(eef[:, 3:9])
        quaternions = matrix_to_quat(matrices)
        canonical_rot6d = rotation_matrix_to_rot6d(matrices)
        provenance = {
            **self._provenance,
            "provider": "GR00TTrajectoryProvider",
            "output_contract": "TrajectoryChunkV1",
            "decoded_action_shape": list(self.decoded_action.shape),
            "eef_slice": [0, 9],
            "gripper_slice": [9, 10],
            "input_eef_rot6d_layout": self.config.eef_rot6d_layout,
            "canonical_eef_orientation": "unit_quaternion_xyzw",
            "canonical_chunk_rot6d_layout": "FIRST_TWO_COLUMNS",
            "source_frame": self.config.source_frame,
            "tcp_semantic_id": self.config.tcp_semantic_id,
            "timestamp_provenance": self.config.timestamp_provenance,
            "action_sample_period_s": self.config.action_sample_period_s,
            "gripper_source_convention": self.config.gripper_source_convention,
        }
        return TrajectoryChunkV1(
            chunk_id=self.chunk_id,
            source=self.config.source,
            observation_timestamp=obs,
            point_timestamps=timestamps,
            eef_position=eef[:, :3],
            eef_orientation=quaternions,
            eef_rot6d=canonical_rot6d,
            source_frame=self.config.source_frame,
            tcp_semantic_id=self.config.tcp_semantic_id,
            validity_start=float(timestamps[0]),
            validity_end=float(timestamps[-1]),
            gripper=gripper,
            provenance=provenance,
        )

    def gripper_command(
        self,
        chunk: TrajectoryChunkV1,
        *,
        timestamp: float | None = None,
        stale: bool = False,
    ) -> Optional[GripperCommandV1]:
        """Read the parallel gripper channel without adding it to WBC input."""

        if chunk.gripper is None:
            return None
        now = float(chunk.observation_timestamp if timestamp is None else timestamp)
        values = np.asarray(chunk.gripper).reshape(-1)
        index = int(np.clip(np.searchsorted(chunk.point_timestamps, now, side="right") - 1, 0, len(values) - 1))
        return GripperCommandV1(
            timestamp=now,
            normalized_command=float(values[index]),
            stale=bool(stale),
            raw_source_convention=self.config.gripper_source_convention,
            provenance={"chunk_id": chunk.chunk_id, "parallel_channel": True},
        )

    def next_chunk(self, observation_timestamp: float) -> Optional[TrajectoryChunkV1]:
        if self._emitted:
            return None
        chunk = self.build_chunk(observation_timestamp=observation_timestamp)
        self._emitted = True
        return chunk
