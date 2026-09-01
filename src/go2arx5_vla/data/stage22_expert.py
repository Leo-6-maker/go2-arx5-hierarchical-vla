"""Same-scene expert snapshot records and offline materialization helpers.

Stage22 records one model-facing observation at each native expert replanning
point. The 40-point label is generated from the current achieved EEF to the
same-scene target. A snapshot is expanded to a one-window LeRobot episode by
repeating the observation frame/state; this is intentional and explicitly
marked as a snapshot fixture, not a fabricated continuous camera recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.so3 import rot6d_rows_to_matrix
from .readiness import (
    CameraStreamV1,
    CalibrationBundleV1,
    GR00TTrainingExampleV1,
    RawEpisodeV1,
    StateV1,
)


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


@dataclass(frozen=True)
class SameSceneExpertSnapshotV1:
    """One exact native-camera/state observation with an expert future label."""

    example_id: str
    raw_episode_id: str
    simulator_instance_id: str
    observation_timestamp: float
    body_frame_path: str
    wrist_frame_path: str
    state_v1: StateV1
    future_timestamps: np.ndarray
    future_desired_eef_9d: np.ndarray
    future_gripper: np.ndarray
    target_position_world_m: np.ndarray
    source_frame: str
    tcp_semantic_id: str
    calibration_bundle_id: str
    target_id: str
    target_excluded_from_model_input: bool = True
    native_control_dt_s: Optional[float] = None
    raw_telemetry: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.raw_episode_id or not self.simulator_instance_id:
            raise ValueError("example, episode, and simulator identifiers are required")
        if not np.isfinite(float(self.observation_timestamp)):
            raise ValueError("observation_timestamp must be finite")
        for name in (
            "body_frame_path",
            "wrist_frame_path",
            "source_frame",
            "tcp_semantic_id",
            "calibration_bundle_id",
            "target_id",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} is required")
        raw_timestamps = np.asarray(self.future_timestamps, dtype=np.float64)
        timestamps = _finite_array("future_timestamps", raw_timestamps, raw_timestamps.shape)
        if timestamps.ndim != 1 or timestamps.size < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("future_timestamps must be strictly increasing with at least two points")
        desired = _finite_array("future_desired_eef_9d", self.future_desired_eef_9d, (timestamps.size, 9))
        gripper = _finite_array("future_gripper", self.future_gripper, (timestamps.size, 1))
        rot6d_rows_to_matrix(desired[:, 3:9])
        target = _finite_array("target_position_world_m", self.target_position_world_m, (3,))
        object.__setattr__(self, "future_timestamps", timestamps)
        object.__setattr__(self, "future_desired_eef_9d", desired)
        object.__setattr__(self, "future_gripper", gripper)
        object.__setattr__(self, "target_position_world_m", target)
        if self.native_control_dt_s is not None and (
            not np.isfinite(float(self.native_control_dt_s)) or float(self.native_control_dt_s) <= 0.0
        ):
            raise ValueError("native_control_dt_s must be positive and finite when present")
        object.__setattr__(self, "raw_telemetry", dict(self.raw_telemetry))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def horizon(self) -> int:
        return int(self.future_timestamps.size)

    @property
    def physical_action(self) -> np.ndarray:
        return np.concatenate([self.future_desired_eef_9d, self.future_gripper], axis=-1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "SameSceneExpertSnapshotV1",
            "example_id": self.example_id,
            "raw_episode_id": self.raw_episode_id,
            "simulator_instance_id": self.simulator_instance_id,
            "observation_timestamp": float(self.observation_timestamp),
            "body_frame_path": self.body_frame_path,
            "wrist_frame_path": self.wrist_frame_path,
            "state_v1": self.state_v1.as_dict(),
            "future_timestamps": self.future_timestamps.tolist(),
            "future_desired_eef_9d": self.future_desired_eef_9d.tolist(),
            "future_gripper": self.future_gripper.tolist(),
            "target_position_world_m": self.target_position_world_m.tolist(),
            "source_frame": self.source_frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "calibration_bundle_id": self.calibration_bundle_id,
            "target_id": self.target_id,
            "target_excluded_from_model_input": bool(self.target_excluded_from_model_input),
            "native_control_dt_s": self.native_control_dt_s,
            "raw_telemetry": dict(self.raw_telemetry),
            "provenance": dict(self.provenance),
        }


def snapshot_to_raw_episode(
    snapshot: SameSceneExpertSnapshotV1,
    *,
    calibration: CalibrationBundleV1,
) -> RawEpisodeV1:
    """Expand one snapshot into a transparent one-window ``RawEpisodeV1``."""

    timestamps = snapshot.future_timestamps
    state = snapshot.state_v1.as_array()
    state_eef = np.broadcast_to(state[:9], (snapshot.horizon, 9)).copy()
    state_gripper = np.broadcast_to(state[9:], (snapshot.horizon, 1)).copy()
    desired = snapshot.future_desired_eef_9d.copy()
    desired_gripper = snapshot.future_gripper.copy()
    achieved = state_eef.copy()
    achieved_gripper = state_gripper.copy()
    camera_provenance = {
        "source": "Stage22_native_same_scene_snapshot",
        "simulator_instance_id": snapshot.simulator_instance_id,
        "snapshot_expanded": True,
    }
    cameras = {
        "body": CameraStreamV1(
            name="body",
            frame_paths=tuple(snapshot.body_frame_path for _ in timestamps),
            timestamps=timestamps,
            declared_fps=None,
            frame=snapshot.source_frame,
            provenance=camera_provenance,
        ),
        "wrist": CameraStreamV1(
            name="wrist",
            frame_paths=tuple(snapshot.wrist_frame_path for _ in timestamps),
            timestamps=timestamps,
            declared_fps=None,
            frame=snapshot.source_frame,
            provenance=camera_provenance,
        ),
    }
    return RawEpisodeV1(
        episode_id=snapshot.example_id,
        trajectory_id=snapshot.example_id,
        source="STAGE22_NATIVE_SAME_SCENE_EXPERT_SNAPSHOT",
        sim_level="NATIVE_ISAACGYM_SAME_SCENE",
        task={
            "task_id": "SIM_VISUAL_REACH_V0",
            "instruction": "move the end effector to the red target",
            "target_id": snapshot.target_id,
            "target_position_world_m": snapshot.target_position_world_m.tolist(),
            "target_excluded_from_model_input": True,
        },
        native_timestamps=timestamps,
        state_eef_9d=state_eef,
        state_gripper=state_gripper,
        desired_eef_9d=desired,
        desired_gripper=desired_gripper,
        achieved_eef_9d=achieved,
        achieved_gripper=achieved_gripper,
        camera_streams=cameras,
        calibration_bundle_id=calibration.bundle_id,
        frame_graph={
            "pose_frame": snapshot.source_frame,
            "tcp": snapshot.tcp_semantic_id,
            "camera_frame": "NATIVE_SAME_SCENE_CAMERA",
            "simulator_instance_id": snapshot.simulator_instance_id,
        },
        lifecycle_events=(
            {"type": "snapshot_observation", "timestamp": float(snapshot.observation_timestamp)},
            {"type": "expert_label_window", "timestamp": float(timestamps[0])},
        ),
        raw_whole_body_streams={
            "native_observation_telemetry": dict(snapshot.raw_telemetry),
            "target_position_world_m": snapshot.target_position_world_m.tolist(),
            "replanning_snapshot": True,
        },
        clock_domains={"native": "NATIVE_SIM_CLOCK", "body": "NATIVE_SIM_CLOCK", "wrist": "NATIVE_SIM_CLOCK"},
        metadata={
            **dict(snapshot.provenance),
            "stage": "Stage22",
            "simulator_instance_id": snapshot.simulator_instance_id,
            "target_excluded_from_model_input": True,
            "snapshot_expanded_horizon": True,
            "snapshot_expansion_reason": "one_model_observation_to_one_40_point_expert_window",
            "source_timestamps_authority": True,
            "desired_label_only": True,
            "achieved_stream_role": "same_snapshot_state_audit_only_not_training_label",
            "hardware_control": False,
            "native_wbc_camera": True,
        },
    )


def snapshot_to_training_example(snapshot: SameSceneExpertSnapshotV1) -> GR00TTrainingExampleV1:
    """Build the logical training contract without importing LeRobot/GR00T."""

    return GR00TTrainingExampleV1(
        example_id=snapshot.example_id,
        raw_episode_id=snapshot.raw_episode_id,
        observation_timestamp=snapshot.observation_timestamp,
        observation_images={"body": snapshot.body_frame_path, "wrist": snapshot.wrist_frame_path},
        task_instruction="move the end effector to the red target",
        state_v1=snapshot.state_v1,
        future_timestamps=snapshot.future_timestamps,
        future_desired_eef_9d=snapshot.future_desired_eef_9d,
        future_gripper=snapshot.future_gripper,
        calibration_bundle_id=snapshot.calibration_bundle_id,
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        provenance={
            **dict(snapshot.provenance),
            "target_excluded_from_model_input": True,
            "label_source": "same_scene_deterministic_expert",
        },
    )


def load_snapshot(path: str | Path) -> SameSceneExpertSnapshotV1:
    """Load one JSON snapshot descriptor emitted by the native collector."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = StateV1.from_array(payload["state_v1"]["values"])
    return SameSceneExpertSnapshotV1(
        example_id=str(payload["example_id"]),
        raw_episode_id=str(payload["raw_episode_id"]),
        simulator_instance_id=str(payload["simulator_instance_id"]),
        observation_timestamp=float(payload["observation_timestamp"]),
        body_frame_path=str(payload["body_frame_path"]),
        wrist_frame_path=str(payload["wrist_frame_path"]),
        state_v1=state,
        future_timestamps=payload["future_timestamps"],
        future_desired_eef_9d=payload["future_desired_eef_9d"],
        future_gripper=payload["future_gripper"],
        target_position_world_m=payload["target_position_world_m"],
        source_frame=str(payload["source_frame"]),
        tcp_semantic_id=str(payload["tcp_semantic_id"]),
        calibration_bundle_id=str(payload["calibration_bundle_id"]),
        target_id=str(payload["target_id"]),
        target_excluded_from_model_input=bool(payload.get("target_excluded_from_model_input", True)),
        native_control_dt_s=payload.get("native_control_dt_s"),
        raw_telemetry=payload.get("raw_telemetry", {}),
        provenance=payload.get("provenance", {}),
    )
