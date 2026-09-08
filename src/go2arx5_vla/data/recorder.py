"""Read-only recorder abstraction and replay qualification backend.

``RecorderV1`` accepts asynchronous streams but has no method that sends a
command to a robot.  The replay helper is the only Stage19 backend and copies
D3 source samples into a node-local/in-memory recording envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .readiness import RawEpisodeV1, _jsonable
from .teleop import TeleopCommandV1


def _copy_finite(value: Any, name: str) -> Any:
    if value is None:
        return None
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _timestamp(value: Any, name: str) -> float:
    timestamp = float(value)
    if not np.isfinite(timestamp):
        raise ValueError(f"{name} must be finite")
    return timestamp


@dataclass(frozen=True)
class RecorderConfigV1:
    episode_id: str
    source: str
    calibration_bundle_id: str
    frame_graph: Mapping[str, Any]
    require_monotonic_streams: bool = True

    def __post_init__(self) -> None:
        if not self.episode_id or not self.source or not self.calibration_bundle_id:
            raise ValueError("episode_id, source, and calibration_bundle_id are required")
        object.__setattr__(self, "frame_graph", MappingProxyType(dict(self.frame_graph)))


@dataclass(frozen=True)
class RecordedEpisodeV1:
    config: RecorderConfigV1
    state_samples: tuple[Mapping[str, Any], ...]
    camera_samples: Mapping[str, tuple[Mapping[str, Any], ...]]
    desired_commands: tuple[TeleopCommandV1, ...]
    lifecycle_events: tuple[Mapping[str, Any], ...]
    safety_events: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_samples",
            MappingProxyType({str(k): tuple(v) for k, v in self.camera_samples.items()}),
        )

    @property
    def state_count(self) -> int:
        return len(self.state_samples)

    def as_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        state_summary: dict[str, Any] = {"count": self.state_count}
        if self.state_samples:
            timestamps = [float(row["timestamp"]) for row in self.state_samples]
            state_summary.update({"first_timestamp": timestamps[0], "last_timestamp": timestamps[-1]})
        result = {
            "schema": "RecordedEpisodeV1",
            "episode_id": self.config.episode_id,
            "source": self.config.source,
            "calibration_bundle_id": self.config.calibration_bundle_id,
            "frame_graph": _jsonable(self.config.frame_graph),
            "state": state_summary,
            "camera_counts": {name: len(rows) for name, rows in self.camera_samples.items()},
            "desired_command_count": len(self.desired_commands),
            "lifecycle_event_count": len(self.lifecycle_events),
            "safety_event_count": len(self.safety_events),
            "actuation_backend": "NONE_READ_ONLY_RECORDING_ONLY",
        }
        if include_values:
            result["state_samples"] = _jsonable(self.state_samples)
            result["camera_samples"] = _jsonable(self.camera_samples)
            result["desired_commands"] = [command.as_dict() for command in self.desired_commands]
            result["lifecycle_events"] = _jsonable(self.lifecycle_events)
            result["safety_events"] = _jsonable(self.safety_events)
        return result


class RecorderV1:
    """Collect asynchronous recording streams without a physical backend."""

    def __init__(self, config: RecorderConfigV1) -> None:
        self.config = config
        self._state_samples: list[Mapping[str, Any]] = []
        self._camera_samples: dict[str, list[Mapping[str, Any]]] = {}
        self._desired_commands: list[TeleopCommandV1] = []
        self._lifecycle_events: list[Mapping[str, Any]] = []
        self._safety_events: list[Mapping[str, Any]] = []

    def _check_next_timestamp(self, rows: list[Mapping[str, Any]], timestamp: float, name: str) -> None:
        if self.config.require_monotonic_streams and rows and timestamp <= float(rows[-1]["timestamp"]):
            raise ValueError(f"{name} timestamps must be strictly increasing")

    def append_state(
        self,
        timestamp: float,
        *,
        go2_state: Any = None,
        arx5_state: Any = None,
        achieved_eef_9d: Any = None,
        gripper_state: Any = None,
        base_pose: Any = None,
    ) -> None:
        timestamp = _timestamp(timestamp, "state timestamp")
        self._check_next_timestamp(self._state_samples, timestamp, "state")
        row = {
            "timestamp": timestamp,
            "go2_state": _copy_finite(go2_state, "go2_state"),
            "arx5_state": _copy_finite(arx5_state, "arx5_state"),
            "achieved_eef_9d": _copy_finite(achieved_eef_9d, "achieved_eef_9d"),
            "gripper_state": _copy_finite(gripper_state, "gripper_state"),
            "base_pose": _copy_finite(base_pose, "base_pose"),
        }
        self._state_samples.append(MappingProxyType(row))

    def append_camera(self, stream_name: str, timestamp: float, frame_path: str) -> None:
        if not stream_name or not frame_path:
            raise ValueError("camera stream name and frame path are required")
        timestamp = _timestamp(timestamp, f"camera[{stream_name}] timestamp")
        rows = self._camera_samples.setdefault(stream_name, [])
        self._check_next_timestamp(rows, timestamp, f"camera[{stream_name}]")
        rows.append(MappingProxyType({"timestamp": timestamp, "frame_path": str(frame_path)}))

    def append_command(self, command: TeleopCommandV1) -> None:
        if not isinstance(command, TeleopCommandV1):
            raise TypeError("desired command must be TeleopCommandV1")
        if self.config.require_monotonic_streams and self._desired_commands:
            if command.timestamp <= self._desired_commands[-1].timestamp:
                raise ValueError("desired command timestamps must be strictly increasing")
        self._desired_commands.append(command)

    def append_event(self, event: Mapping[str, Any], *, safety: bool = False) -> None:
        if not isinstance(event, Mapping) or not event.get("type"):
            raise ValueError("event must be a mapping with a type")
        copied = MappingProxyType({str(key): value for key, value in event.items()})
        (self._safety_events if safety else self._lifecycle_events).append(copied)

    def finalize(self) -> RecordedEpisodeV1:
        return RecordedEpisodeV1(
            config=self.config,
            state_samples=tuple(self._state_samples),
            camera_samples={name: tuple(rows) for name, rows in self._camera_samples.items()},
            desired_commands=tuple(self._desired_commands),
            lifecycle_events=tuple(self._lifecycle_events),
            safety_events=tuple(self._safety_events),
        )


def replay_raw_episode(raw: RawEpisodeV1) -> RecordedEpisodeV1:
    """Replay D3/native streams into ``RecorderV1`` without any actuation."""

    recorder = RecorderV1(
        RecorderConfigV1(
            episode_id=raw.episode_id,
            source=f"REPLAY:{raw.source}",
            calibration_bundle_id=raw.calibration_bundle_id,
            frame_graph=dict(raw.frame_graph),
        )
    )
    achieved_gripper = raw.achieved_gripper
    for index, timestamp in enumerate(raw.native_timestamps):
        recorder.append_state(
            float(timestamp),
            arx5_state=None if raw.arx5_joint_state is None else raw.arx5_joint_state[index],
            achieved_eef_9d=raw.achieved_eef_9d[index],
            gripper_state=None if achieved_gripper is None else achieved_gripper[index],
            base_pose=None if raw.go2_base_pose is None else raw.go2_base_pose[index],
        )
        recorder.append_command(
            TeleopCommandV1(
                command_id=f"{raw.episode_id}:command_{index:06d}",
                timestamp=float(timestamp),
                eef_9d=raw.desired_eef_9d[index],
                gripper=raw.desired_gripper[index],
                frame=raw.source_frame,
                tcp_semantic_id=raw.tcp_semantic_id,
                source="D3_REPLAY_DESIRED_LABEL",
            )
        )
    for name, stream in raw.camera_streams.items():
        for timestamp, frame_path in zip(stream.timestamps, stream.frame_paths):
            recorder.append_camera(name, float(timestamp), frame_path)
    for event in raw.lifecycle_events:
        recorder.append_event(dict(event))
    for event in raw.safety_events:
        recorder.append_event(dict(event), safety=True)
    return recorder.finalize()
