"""Deterministic same-scene expert labels for the Stage22 visual-reach task."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.so3 import quat_slerp


@dataclass(frozen=True)
class SameSceneExpertConfigV1:
    """Test-only task-space expert configuration."""

    horizon: int = 40
    action_sample_period_s: float = 0.05
    motion_duration_s: float = 1.95
    gripper_value: float = 0.5
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {
            "expert": "deterministic_smoothstep_position_hold",
            "orientation": "constant_current_orientation",
            "timing": "TEST_ONLY_STAGE22_SIMULATION",
        }
    )

    def __post_init__(self) -> None:
        if int(self.horizon) < 2:
            raise ValueError("horizon must be at least 2")
        for name in ("action_sample_period_s", "motion_duration_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(float(self.gripper_value)):
            raise ValueError("gripper_value must be finite")


def _smoothstep(alpha: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def expert_pose_sequence(
    start_position: Any,
    target_position: Any,
    start_quaternion: Any,
    timestamps_from_start: Any,
    *,
    motion_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth position-to-target/hold pose sequence."""

    start = np.asarray(start_position, dtype=np.float64).reshape(3)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    start_q = np.asarray(start_quaternion, dtype=np.float64).reshape(4)
    times = np.asarray(timestamps_from_start, dtype=np.float64).reshape(-1)
    if not np.isfinite(start).all() or not np.isfinite(target).all() or not np.isfinite(start_q).all():
        raise ValueError("expert pose inputs must be finite")
    if np.linalg.norm(start_q) < 1e-10 or not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("expert quaternion/timestamps are invalid")
    if not np.isfinite(float(motion_duration_s)) or float(motion_duration_s) <= 0.0:
        raise ValueError("motion_duration_s must be positive and finite")
    start_q = start_q / np.linalg.norm(start_q)
    alpha = _smoothstep(times / float(motion_duration_s))
    positions = start[None, :] + alpha[:, None] * (target - start)[None, :]
    quaternions = np.broadcast_to(start_q, (times.size, 4)).copy()
    # Keep the SO(3) operation explicit even though v0 holds orientation.
    if times.size:
        quaternions[0] = quat_slerp(start_q, start_q, 0.0)
    return positions, quaternions


def build_same_scene_expert_chunk(
    *,
    chunk_id: str,
    observation_timestamp: float,
    current_position: Any,
    target_position: Any,
    current_quaternion: Any,
    config: SameSceneExpertConfigV1 | None = None,
    source_frame: str,
    tcp_semantic_id: str,
) -> TrajectoryChunkV1:
    """Create one state-conditioned expert chunk for a native observation."""

    cfg = SameSceneExpertConfigV1() if config is None else config
    timestamps = float(observation_timestamp) + np.arange(int(cfg.horizon), dtype=np.float64) * float(
        cfg.action_sample_period_s
    )
    positions, quaternions = expert_pose_sequence(
        current_position,
        target_position,
        current_quaternion,
        timestamps - float(observation_timestamp),
        motion_duration_s=cfg.motion_duration_s,
    )
    provenance = {
        **dict(cfg.provenance),
        "stage": "Stage22",
        "label_source": "same_scene_deterministic_expert",
        "target_excluded_from_model_input": True,
        "action_sample_period_s": float(cfg.action_sample_period_s),
    }
    return TrajectoryChunkV1(
        chunk_id=str(chunk_id),
        source="STAGE22_SAME_SCENE_EXPERT",
        observation_timestamp=float(observation_timestamp),
        point_timestamps=timestamps,
        eef_position=positions,
        eef_orientation=quaternions,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=np.full(int(cfg.horizon), float(cfg.gripper_value), dtype=np.float64),
        provenance=provenance,
    )


def write_rgb_ppm(path: str | Path, image: Any) -> str:
    """Write a dependency-free RGB PPM frame and return its SHA256."""

    output = Path(path)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("image must have shape [H,W,3]")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(f"P6\n{array.shape[1]} {array.shape[0]}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(array).tobytes())
    return hashlib.sha256(output.read_bytes()).hexdigest()
