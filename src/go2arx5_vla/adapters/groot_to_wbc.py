"""Offline GR00T -> delivered-WBC reference adapter.

This module owns reference conversion only. It never loads a policy, starts a
simulator, or emits a joint command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..contracts.trajectory import GripperCommandV1, TrajectoryChunkV1, WBCPreviewV1
from ..trajectory.preview import (
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    DEFAULT_WBC_PREVIEW_OFFSETS_M,
    se3_arc_length_preview,
)


@dataclass(frozen=True)
class ReferenceAdapterConfig:
    expected_source_frame: str
    wbc_frame: str
    tcp_semantic_id: str
    preview_offsets_m: np.ndarray = field(default_factory=lambda: DEFAULT_WBC_PREVIEW_OFFSETS_M.copy())
    lambda_m_per_rad: float = DEFAULT_SE3_LAMBDA_M_PER_RAD
    stale_timeout_s: Optional[float] = None
    frame_transform: Optional[np.ndarray] = None


class GR00TToWBCAdapter:
    """Convert a timestamped canonical chunk to the exact WBC preview ABI."""

    def __init__(self, config: ReferenceAdapterConfig):
        self.config = config

    def _prepare_chunk(self, chunk: TrajectoryChunkV1) -> TrajectoryChunkV1:
        if chunk.tcp_semantic_id != self.config.tcp_semantic_id:
            raise ValueError(
                f"TCP mismatch: chunk={chunk.tcp_semantic_id!r}, expected={self.config.tcp_semantic_id!r}"
            )
        if chunk.source_frame != self.config.expected_source_frame:
            if self.config.frame_transform is None:
                raise ValueError(
                    f"frame mismatch: chunk={chunk.source_frame!r}, expected={self.config.expected_source_frame!r}; "
                    "provide an explicit frame_transform"
                )
            chunk = chunk.transformed(self.config.frame_transform, target_frame=self.config.wbc_frame)
        elif chunk.source_frame != self.config.wbc_frame:
            # The names may be aliases only when the caller explicitly says so.
            if self.config.frame_transform is None:
                raise ValueError(
                    f"frame mismatch: source frame {chunk.source_frame!r} is not the WBC frame "
                    f"{self.config.wbc_frame!r}; provide an explicit frame_transform"
                )
            chunk = chunk.transformed(self.config.frame_transform, target_frame=self.config.wbc_frame)
        return chunk

    def adapt(
        self,
        chunk: TrajectoryChunkV1,
        *,
        controller_timestamp: Optional[float] = None,
    ) -> tuple[WBCPreviewV1, Optional[GripperCommandV1]]:
        prepared = self._prepare_chunk(chunk)
        now = float(prepared.observation_timestamp if controller_timestamp is None else controller_timestamp)
        age = max(0.0, now - float(prepared.observation_timestamp))
        stale = self.config.stale_timeout_s is not None and age > self.config.stale_timeout_s
        preview = se3_arc_length_preview(
            prepared,
            anchor_timestamp=now,
            offsets_m=np.asarray(self.config.preview_offsets_m, dtype=np.float64),
            lambda_m_per_rad=self.config.lambda_m_per_rad,
        )
        provenance = dict(preview.provenance)
        provenance.update(
            {
                "adapter": "GR00TToWBCAdapter",
                "command_age_s": age,
                "stale": bool(stale),
                "external_reference": "EEF_XYZ_ROT6D",
            }
        )
        preview = WBCPreviewV1(
            anchor_timestamp=preview.anchor_timestamp,
            offsets_m=preview.offsets_m,
            eef_position=preview.eef_position,
            eef_rot6d=preview.eef_rot6d,
            source_frame=preview.source_frame,
            tcp_semantic_id=preview.tcp_semantic_id,
            endpoint_clamped=preview.endpoint_clamped,
            provenance=provenance,
        )
        grip = None
        if prepared.gripper is not None:
            values = np.asarray(prepared.gripper).reshape(-1)
            idx = int(np.clip(np.searchsorted(prepared.point_timestamps, now, side="right") - 1, 0, values.size - 1))
            grip = GripperCommandV1(
                timestamp=now,
                normalized_command=float(values[idx]),
                stale=bool(stale),
                raw_source_convention="canonical_normalized",
                provenance={"chunk_id": prepared.chunk_id},
            )
        return preview, grip

    def adapt_xyz_rot6d(
        self,
        *,
        chunk_id: str,
        observation_timestamp: float,
        point_timestamps: Any,
        eef_xyz: Any,
        eef_rot6d: Any,
        gripper: Optional[Any] = None,
        validity_start: Optional[float] = None,
        validity_end: Optional[float] = None,
        source: str = "GR00T",
        controller_timestamp: Optional[float] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> tuple[WBCPreviewV1, Optional[GripperCommandV1]]:
        ts = np.asarray(point_timestamps, dtype=np.float64)
        start = float(ts[0] if validity_start is None else validity_start)
        end = float(ts[-1] if validity_end is None else validity_end)
        chunk = TrajectoryChunkV1.from_rot6d(
            chunk_id=chunk_id,
            source=source,
            observation_timestamp=observation_timestamp,
            point_timestamps=ts,
            eef_position=eef_xyz,
            eef_rot6d=eef_rot6d,
            source_frame=self.config.expected_source_frame,
            tcp_semantic_id=self.config.tcp_semantic_id,
            validity_start=start,
            validity_end=end,
            gripper=gripper,
            provenance={} if provenance is None else provenance,
        )
        return self.adapt(chunk, controller_timestamp=controller_timestamp)
