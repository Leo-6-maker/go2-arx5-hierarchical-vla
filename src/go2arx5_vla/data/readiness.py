"""Stage18 raw-recording and training-example contracts.

This module deliberately contains no simulator, camera, robot, or model
runtime.  It is a NumPy-only boundary for turning an immutable native-rate
episode into a synchronized model-facing example.

Important conventions:

* ``eef_9d`` is ``XYZ + GR00T_FIRST_TWO_ROWS`` rot6d.
* source timestamps are authoritative; declared FPS is metadata to audit.
* desired and achieved EEF streams are always stored separately.
* no resampling, clipping, retiming, or calibration inference happens here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..trajectory.so3 import (
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


GROOT_EEF_ROT6D_LAYOUT = "GR00T_FIRST_TWO_ROWS"
CAMERA_TIMESTAMP_LAYOUT = "SOURCE_TIMESTAMPS_PRESERVED"


def _readonly_array(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze(value: Any) -> Any:
    """Copy nested metadata without leaving mutable array references."""

    if isinstance(value, np.ndarray):
        return _readonly_array(value, name="metadata", dtype=value.dtype)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _strict_timestamps(name: str, value: Any) -> np.ndarray:
    timestamps = _readonly_array(value, name=name)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    _finite(name, timestamps)
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return timestamps


def _shape(name: str, value: Any, expected: tuple[int, ...]) -> np.ndarray:
    array = _readonly_array(value, name=name)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    _finite(name, array)
    return array


def _validate_rot6d_rows(name: str, value: np.ndarray) -> np.ndarray:
    try:
        matrices = rot6d_rows_to_matrix(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid {GROOT_EEF_ROT6D_LAYOUT} representation") from exc
    identity = np.einsum("...ji,...jk->...ik", matrices, matrices)
    determinant = np.linalg.det(matrices)
    if not np.allclose(identity, np.eye(3), atol=2e-6) or not np.allclose(
        determinant, 1.0, atol=2e-6
    ):
        raise ValueError(f"{name} does not decode to valid SO(3)")
    return matrices


def _stats(value: Any) -> dict[str, float | int | None]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    if not np.isfinite(array).all():
        return {"count": int(array.size), "min": None, "max": None, "mean": None, "p95": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CameraStreamV1:
    """One camera stream with source timestamps and declared metadata."""

    name: str
    frame_paths: tuple[str, ...]
    timestamps: np.ndarray
    declared_fps: Optional[float]
    frame: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    intrinsics: Mapping[str, Any] = field(default_factory=dict)
    extrinsics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera stream name is required")
        paths = tuple(str(path) for path in self.frame_paths)
        if not paths or any(not path for path in paths):
            raise ValueError("camera stream must contain non-empty frame paths")
        timestamps = _strict_timestamps(f"camera[{self.name}].timestamps", self.timestamps)
        if len(paths) != timestamps.size:
            raise ValueError("camera frame_paths and timestamps must have equal length")
        fps = None if self.declared_fps is None else float(self.declared_fps)
        if fps is not None and (not np.isfinite(fps) or fps <= 0.0):
            raise ValueError("declared_fps must be positive and finite")
        if not self.frame:
            raise ValueError("camera frame is required")
        object.__setattr__(self, "frame_paths", paths)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "declared_fps", fps)
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "intrinsics", _freeze(self.intrinsics))
        object.__setattr__(self, "extrinsics", _freeze(self.extrinsics))

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.size)

    @property
    def effective_rate_hz(self) -> Optional[float]:
        if self.timestamps.size < 2:
            return None
        duration = float(self.timestamps[-1] - self.timestamps[0])
        return None if duration <= 0.0 else float((self.timestamps.size - 1) / duration)

    @property
    def effective_period_s(self) -> Optional[float]:
        if self.timestamps.size < 2:
            return None
        return float(np.median(np.diff(self.timestamps)))

    def as_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        return {
            "schema": "CameraStreamV1",
            "name": self.name,
            "frame_paths": list(self.frame_paths) if include_paths else {"count": self.frame_count},
            "timestamps": self.timestamps.tolist(),
            "declared_fps": self.declared_fps,
            "effective_rate_hz": self.effective_rate_hz,
            "effective_period_s": self.effective_period_s,
            "frame": self.frame,
            "provenance": _jsonable(self.provenance),
            "intrinsics": _jsonable(self.intrinsics),
            "extrinsics": _jsonable(self.extrinsics),
        }


@dataclass(frozen=True)
class CalibrationBundleV1:
    """Versioned calibration metadata; unknown physical values stay unknown."""

    bundle_id: str
    version: str
    status: str
    T_go2_arx5: Optional[np.ndarray] = None
    T_flange_tcp: Optional[np.ndarray] = None
    body_intrinsics: Mapping[str, Any] = field(default_factory=dict)
    body_extrinsics: Mapping[str, Any] = field(default_factory=dict)
    wrist_intrinsics: Mapping[str, Any] = field(default_factory=dict)
    wrist_extrinsics: Mapping[str, Any] = field(default_factory=dict)
    arx5_joint_zero_version: str = "REAL_TBD"
    gripper_mapping: Mapping[str, Any] = field(default_factory=dict)
    clock_sync_version: str = "REAL_TBD"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    real_robot_calibration: bool = False

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.version or not self.status:
            raise ValueError("calibration bundle_id, version, and status are required")
        for name in ("T_go2_arx5", "T_flange_tcp"):
            value = getattr(self, name)
            if value is not None:
                matrix = _shape(name, value, (4, 4))
                if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
                    raise ValueError(f"{name} must be homogeneous")
                object.__setattr__(self, name, matrix)
        for name in (
            "body_intrinsics",
            "body_extrinsics",
            "wrist_intrinsics",
            "wrist_extrinsics",
            "gripper_mapping",
            "provenance",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if not isinstance(self.real_robot_calibration, bool):
            raise ValueError("real_robot_calibration must be bool")

    @classmethod
    def from_d3_episode(cls, episode: Mapping[str, Any]) -> "CalibrationBundleV1":
        frame_graph = dict(episode.get("frame_graph", {}))
        mount = str(frame_graph.get("mount", "SIM_REFERENCE_UMI_MOUNT"))
        tcp = str(frame_graph.get("tcp", "SIM_PROVISIONAL_TCP"))
        return cls(
            bundle_id="D3_SIMULATION_CALIBRATION_BUNDLE_V1",
            version="D3_V0",
            status="SIMULATION_ONLY",
            body_intrinsics=dict(episode.get("camera_calibration", {}).get("body", {}).get("intrinsics", {})),
            body_extrinsics=dict(episode.get("camera_calibration", {}).get("body", {}).get("T_parent_camera", {})),
            wrist_intrinsics=dict(episode.get("camera_calibration", {}).get("wrist", {}).get("intrinsics", {})),
            wrist_extrinsics=dict(episode.get("camera_calibration", {}).get("wrist", {}).get("T_parent_camera", {})),
            arx5_joint_zero_version="SIM_REFERENCE_URDF_ZERO_V0",
            gripper_mapping={"status": "SIMULATION_ONLY", "raw_to_normalized": "D3_SOURCE_CONVENTION"},
            clock_sync_version="D3_SOURCE_TIMESTAMP_FIELDS_V0",
            provenance={
                "source": "D3 simulated Isaac render fixture",
                "mount": mount,
                "tcp": tcp,
                "frame_graph": frame_graph,
                "real_robot_calibration": False,
                "unknown_real_fields": [
                    "T_Go2_ARX5",
                    "T_flange_TCP",
                    "ARX5_joint_zero_version",
                    "clock_sync_calibration",
                ],
            },
            real_robot_calibration=False,
        )

    @property
    def T_Go2_ARX5(self) -> Optional[np.ndarray]:
        """Contract-name alias retained for documentation/API audits."""

        return self.T_go2_arx5

    @property
    def T_flange_TCP(self) -> Optional[np.ndarray]:
        """Contract-name alias retained for documentation/API audits."""

        return self.T_flange_tcp

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "CalibrationBundleV1",
            "bundle_id": self.bundle_id,
            "version": self.version,
            "status": self.status,
            "T_go2_arx5": None if self.T_go2_arx5 is None else self.T_go2_arx5.tolist(),
            "T_flange_tcp": None if self.T_flange_tcp is None else self.T_flange_tcp.tolist(),
            "body_intrinsics": _jsonable(self.body_intrinsics),
            "body_extrinsics": _jsonable(self.body_extrinsics),
            "wrist_intrinsics": _jsonable(self.wrist_intrinsics),
            "wrist_extrinsics": _jsonable(self.wrist_extrinsics),
            "arx5_joint_zero_version": self.arx5_joint_zero_version,
            "gripper_mapping": _jsonable(self.gripper_mapping),
            "clock_sync_version": self.clock_sync_version,
            "provenance": _jsonable(self.provenance),
            "real_robot_calibration": self.real_robot_calibration,
        }


@dataclass(frozen=True)
class RawEpisodeV1:
    """Immutable source-rate recording and audit state."""

    episode_id: str
    trajectory_id: str
    source: str
    sim_level: str
    task: Mapping[str, Any]
    native_timestamps: np.ndarray
    state_eef_9d: np.ndarray
    state_gripper: np.ndarray
    desired_eef_9d: np.ndarray
    desired_gripper: np.ndarray
    achieved_eef_9d: np.ndarray
    camera_streams: Mapping[str, CameraStreamV1]
    calibration_bundle_id: str
    frame_graph: Mapping[str, Any]
    lifecycle_events: Sequence[Mapping[str, Any]]
    achieved_gripper: Optional[np.ndarray] = None
    arx5_joint_state: Optional[np.ndarray] = None
    go2_base_pose: Optional[np.ndarray] = None
    raw_whole_body_streams: Mapping[str, Any] = field(default_factory=dict)
    safety_events: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    clock_domains: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("episode_id", "trajectory_id", "source", "calibration_bundle_id"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} is required")
        timestamps = _strict_timestamps("native_timestamps", self.native_timestamps)
        count = int(timestamps.size)
        state = _shape("state_eef_9d", self.state_eef_9d, (count, 9))
        desired = _shape("desired_eef_9d", self.desired_eef_9d, (count, 9))
        achieved = _shape("achieved_eef_9d", self.achieved_eef_9d, (count, 9))
        state_gripper = _shape("state_gripper", self.state_gripper, (count, 1))
        desired_gripper = _shape("desired_gripper", self.desired_gripper, (count, 1))
        achieved_gripper = None
        if self.achieved_gripper is not None:
            achieved_gripper = _shape("achieved_gripper", self.achieved_gripper, (count, 1))
        _validate_rot6d_rows("state_eef_9d.rot6d", state[:, 3:9])
        _validate_rot6d_rows("desired_eef_9d.rot6d", desired[:, 3:9])
        _validate_rot6d_rows("achieved_eef_9d.rot6d", achieved[:, 3:9])

        arx5 = None
        if self.arx5_joint_state is not None:
            arx5 = _readonly_array(self.arx5_joint_state, name="arx5_joint_state")
            if arx5.ndim != 2 or arx5.shape[0] != count:
                raise ValueError("arx5_joint_state must have shape [T,D]")
            _finite("arx5_joint_state", arx5)
        base = None
        if self.go2_base_pose is not None:
            base = _shape("go2_base_pose", self.go2_base_pose, (count, 4, 4))
            if not np.allclose(base[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
                raise ValueError("go2_base_pose must contain homogeneous transforms")

        streams = dict(self.camera_streams)
        if any(not isinstance(stream, CameraStreamV1) for stream in streams.values()):
            raise TypeError("camera_streams values must be CameraStreamV1")
        object.__setattr__(self, "native_timestamps", timestamps)
        object.__setattr__(self, "state_eef_9d", state)
        object.__setattr__(self, "state_gripper", state_gripper)
        object.__setattr__(self, "desired_eef_9d", desired)
        object.__setattr__(self, "desired_gripper", desired_gripper)
        object.__setattr__(self, "achieved_eef_9d", achieved)
        object.__setattr__(self, "achieved_gripper", achieved_gripper)
        object.__setattr__(self, "arx5_joint_state", arx5)
        object.__setattr__(self, "go2_base_pose", base)
        object.__setattr__(self, "camera_streams", _freeze(streams))
        object.__setattr__(self, "task", _freeze(self.task))
        object.__setattr__(self, "frame_graph", _freeze(self.frame_graph))
        object.__setattr__(self, "lifecycle_events", tuple(_freeze(event) for event in self.lifecycle_events))
        object.__setattr__(self, "safety_events", tuple(_freeze(event) for event in self.safety_events))
        object.__setattr__(self, "raw_whole_body_streams", _freeze(self.raw_whole_body_streams))
        object.__setattr__(self, "clock_domains", _freeze(self.clock_domains))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def frame_count(self) -> int:
        return int(self.native_timestamps.size)

    @property
    def timestamps(self) -> np.ndarray:
        """Short alias for callers that use the source episode field name."""

        return self.native_timestamps

    @property
    def source_frame(self) -> str:
        return str(self.frame_graph.get("pose_frame", "REAL_TBD"))

    @property
    def tcp_semantic_id(self) -> str:
        return str(self.frame_graph.get("tcp", "REAL_TBD"))

    @classmethod
    def from_d3_json(cls, path: str | Path) -> "RawEpisodeV1":
        source_path = Path(path)
        payload = json.loads(source_path.read_text())
        camera_streams: dict[str, CameraStreamV1] = {}
        camera_calibration = payload.get("camera_calibration", {})
        for name, camera in payload.get("camera", {}).items():
            calibration = camera_calibration.get(name, {})
            camera_streams[str(name)] = CameraStreamV1(
                name=str(name),
                frame_paths=tuple(camera.get("frames", [])),
                timestamps=camera.get("timestamps", []),
                declared_fps=camera.get("fps"),
                frame=str(camera.get("frame", payload.get("frame_graph", {}).get("pose_frame", "REAL_TBD"))),
                provenance={"source_episode_provenance": camera.get("provenance")},
                intrinsics=calibration.get("intrinsics", {}),
                extrinsics=calibration.get("T_parent_camera", {}),
            )
        state = payload.get("state", {})
        target = payload.get("action_target", {})
        achieved_action = payload.get("action_achieved", {})
        desired_eef = target.get("eef_9d", payload.get("eef_target"))
        desired_gripper = target.get("gripper_position", payload.get("gripper"))
        achieved_eef = achieved_action.get("eef_9d", payload.get("eef_achieved"))
        achieved_gripper = achieved_action.get("gripper_position", payload.get("gripper"))
        if desired_eef is None or desired_gripper is None or achieved_eef is None:
            raise ValueError(f"D3 episode {source_path} does not contain target/achieved EEF streams")
        calibration = CalibrationBundleV1.from_d3_episode(payload)
        metadata = {
            "source_json": str(source_path),
            "source_json_sha256": sha256_file(source_path),
            "source_episode": payload.get("source_episode"),
            "robot_id": payload.get("robot_id"),
            "visual_variant": payload.get("visual_variant"),
            "visual_profile": payload.get("visual_profile"),
            "randomization": payload.get("randomization", {}),
            "provenance": payload.get("provenance", {}),
            "quality": payload.get("quality", {}),
            "target_achieved_separate": True,
            "representation": "SYNTHETIC/ISAAC_RENDER_ONLY",
        }
        optional_streams: dict[str, Any] = {}
        for key in (
            "imu",
            "contact",
            "go2_joint_state",
            "go2_joint_velocity",
            "arx5_joint_velocity",
            "arx5_flange_pose",
            "arm_command",
            "gripper_measured",
        ):
            if key in payload:
                optional_streams[key] = payload[key]
        if isinstance(payload.get("raw_state"), Mapping):
            optional_streams.update({f"raw_state.{key}": value for key, value in payload["raw_state"].items()})
        if isinstance(payload.get("raw_commands"), Mapping):
            optional_streams.update({f"raw_commands.{key}": value for key, value in payload["raw_commands"].items()})
        return cls(
            episode_id=str(payload["episode_id"]),
            trajectory_id=str(payload.get("trajectory_id", payload["episode_id"])),
            source=str(payload.get("provenance", {}).get("source", "D3")),
            sim_level=str(payload.get("sim_level", "UNKNOWN")),
            task=dict(payload.get("task", {})),
            native_timestamps=payload["timestamps"],
            state_eef_9d=state["eef_9d"],
            state_gripper=state.get("gripper_position", payload.get("gripper")),
            desired_eef_9d=desired_eef,
            desired_gripper=desired_gripper,
            achieved_eef_9d=achieved_eef,
            achieved_gripper=achieved_gripper,
            arx5_joint_state=payload.get("arx5_joint_state"),
            go2_base_pose=payload.get("go2_base_pose"),
            camera_streams=camera_streams,
            calibration_bundle_id=calibration.bundle_id,
            frame_graph=dict(payload.get("frame_graph", {})),
            lifecycle_events=tuple(payload.get("events", [])),
            safety_events=tuple(payload.get("safety_events", [])),
            clock_domains=dict(payload.get("clock_domains", {})),
            raw_whole_body_streams=optional_streams,
            metadata=metadata,
        )

    def as_dict(self, *, include_arrays: bool = True, include_camera_paths: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": "RawEpisodeV1",
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "source": self.source,
            "sim_level": self.sim_level,
            "task": _jsonable(self.task),
            "calibration_bundle_id": self.calibration_bundle_id,
            "frame_graph": _jsonable(self.frame_graph),
            "camera_streams": {
                name: stream.as_dict(include_paths=include_camera_paths)
                for name, stream in self.camera_streams.items()
            },
            "lifecycle_events": _jsonable(self.lifecycle_events),
            "safety_events": _jsonable(self.safety_events),
            "clock_domains": _jsonable(self.clock_domains),
            "metadata": _jsonable(self.metadata),
        }
        if include_arrays:
            result.update(
                {
                    "native_timestamps": self.native_timestamps.tolist(),
                    "state_eef_9d": self.state_eef_9d.tolist(),
                    "state_gripper": self.state_gripper.tolist(),
                    "desired_eef_9d": self.desired_eef_9d.tolist(),
                    "desired_gripper": self.desired_gripper.tolist(),
                    "achieved_eef_9d": self.achieved_eef_9d.tolist(),
                    "achieved_gripper": None if self.achieved_gripper is None else self.achieved_gripper.tolist(),
                    "arx5_joint_state": None if self.arx5_joint_state is None else self.arx5_joint_state.tolist(),
                    "go2_base_pose": None if self.go2_base_pose is None else self.go2_base_pose.tolist(),
                    "raw_whole_body_streams": _jsonable(self.raw_whole_body_streams),
                }
            )
        else:
            result["array_shapes"] = {
                "native_timestamps": list(self.native_timestamps.shape),
                "state_eef_9d": list(self.state_eef_9d.shape),
                "state_gripper": list(self.state_gripper.shape),
                "desired_eef_9d": list(self.desired_eef_9d.shape),
                "desired_gripper": list(self.desired_gripper.shape),
                "achieved_eef_9d": list(self.achieved_eef_9d.shape),
                "achieved_gripper": None if self.achieved_gripper is None else list(self.achieved_gripper.shape),
                "arx5_joint_state": None if self.arx5_joint_state is None else list(self.arx5_joint_state.shape),
                "go2_base_pose": None if self.go2_base_pose is None else list(self.go2_base_pose.shape),
            }
        return result


@dataclass(frozen=True)
class SynchronizerConfigV1:
    """Synchronization policy with no production-rate assumption."""

    sample_stream: str = "body"
    max_skew_warn_s: float = 0.010
    max_skew_reject_s: float = 0.050
    gap_factor: float = 1.5
    declared_rate_tolerance: float = 0.01
    expected_declared_rate_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.sample_stream:
            raise ValueError("sample_stream is required")
        if not (0.0 <= self.max_skew_warn_s <= self.max_skew_reject_s):
            raise ValueError("skew thresholds must satisfy 0 <= warn <= reject")
        if not np.isfinite(self.gap_factor) or self.gap_factor <= 1.0:
            raise ValueError("gap_factor must be > 1")
        if not np.isfinite(self.declared_rate_tolerance) or self.declared_rate_tolerance < 0.0:
            raise ValueError("declared_rate_tolerance must be non-negative")
        if self.expected_declared_rate_hz is not None and (
            not np.isfinite(self.expected_declared_rate_hz) or self.expected_declared_rate_hz <= 0.0
        ):
            raise ValueError("expected_declared_rate_hz must be positive and finite")


@dataclass(frozen=True)
class SynchronizationResultV1:
    episode_id: str
    sample_stream: str
    sample_timestamps: np.ndarray
    indices: Mapping[str, np.ndarray]
    skew_seconds: Mapping[str, np.ndarray]
    effective_rates_hz: Mapping[str, Optional[float]]
    declared_rates_hz: Mapping[str, Optional[float]]
    dropped_frames: Mapping[str, int]
    non_monotonic_counts: Mapping[str, int]
    clock_discontinuities: Mapping[str, tuple[int, ...]]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample = _strict_timestamps("sample_timestamps", self.sample_timestamps)
        normalized_indices = {}
        for key, value in self.indices.items():
            normalized_indices[str(key)] = _readonly_array(value, name=f"indices[{key}]", dtype=np.int64)
        normalized_skew = {}
        for key, value in self.skew_seconds.items():
            normalized_skew[str(key)] = _readonly_array(value, name=f"skew_seconds[{key}]")
        if any(value.shape != sample.shape for value in normalized_indices.values()):
            raise ValueError("synchronization indices must align with sample_timestamps")
        if any(value.shape != sample.shape for value in normalized_skew.values()):
            raise ValueError("synchronization skew arrays must align with sample_timestamps")
        object.__setattr__(self, "sample_timestamps", sample)
        object.__setattr__(self, "indices", _freeze(normalized_indices))
        object.__setattr__(self, "skew_seconds", _freeze(normalized_skew))
        object.__setattr__(self, "effective_rates_hz", _freeze(self.effective_rates_hz))
        object.__setattr__(self, "declared_rates_hz", _freeze(self.declared_rates_hz))
        object.__setattr__(self, "dropped_frames", _freeze(self.dropped_frames))
        object.__setattr__(self, "non_monotonic_counts", _freeze(self.non_monotonic_counts))
        object.__setattr__(self, "clock_discontinuities", _freeze(self.clock_discontinuities))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def sample_count(self) -> int:
        return int(self.sample_timestamps.size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "SynchronizationResultV1",
            "episode_id": self.episode_id,
            "sample_stream": self.sample_stream,
            "sample_timestamps": self.sample_timestamps.tolist(),
            "indices": _jsonable(self.indices),
            "skew_seconds": _jsonable(self.skew_seconds),
            "effective_rates_hz": _jsonable(self.effective_rates_hz),
            "declared_rates_hz": _jsonable(self.declared_rates_hz),
            "dropped_frames": _jsonable(self.dropped_frames),
            "non_monotonic_counts": _jsonable(self.non_monotonic_counts),
            "clock_discontinuities": _jsonable(self.clock_discontinuities),
            "warnings": list(self.warnings),
            "metrics": _jsonable(self.metrics),
            "provenance": _jsonable(self.provenance),
        }


def _nearest_indices(source_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    if source_timestamps.ndim != 1 or source_timestamps.size == 0:
        raise ValueError("source timestamps must be non-empty")
    indices = np.searchsorted(source_timestamps, query_timestamps, side="left")
    indices = np.clip(indices, 0, source_timestamps.size - 1)
    left = np.maximum(indices - 1, 0)
    choose_left = np.abs(query_timestamps - source_timestamps[left]) <= np.abs(
        source_timestamps[indices] - query_timestamps
    )
    return np.where(choose_left, left, indices).astype(np.int64)


def _timestamp_issues(timestamps: Any, *, gap_factor: float) -> tuple[int, tuple[int, ...], Optional[float]]:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return (0 if values.ndim == 1 else 1, tuple(), None)
    differences = np.diff(values)
    non_monotonic = int(np.count_nonzero(differences <= 0.0))
    positive = differences[differences > 0.0]
    if positive.size == 0:
        return non_monotonic, tuple(), None
    nominal = float(np.median(positive))
    discontinuities = tuple(int(index) for index, delta in enumerate(differences) if delta > nominal * gap_factor)
    dropped = int(sum(max(0, int(round(delta / nominal)) - 1) for delta in differences if delta > nominal * gap_factor))
    return non_monotonic, discontinuities, float(dropped)


class SynchronizerV1:
    """Align observations to a chosen stream while retaining source times."""

    def __init__(self, config: SynchronizerConfigV1 | None = None) -> None:
        self.config = SynchronizerConfigV1() if config is None else config

    def synchronize(
        self,
        raw: RawEpisodeV1,
        *,
        sample_timestamps: Any | None = None,
    ) -> SynchronizationResultV1:
        streams = raw.camera_streams
        if sample_timestamps is None:
            if self.config.sample_stream not in streams:
                raise ValueError(f"sample stream {self.config.sample_stream!r} is absent")
            sample = streams[self.config.sample_stream].timestamps.copy()
        else:
            sample = _strict_timestamps("sample_timestamps", sample_timestamps)
        native = _strict_timestamps("raw.native_timestamps", raw.native_timestamps)

        indices: dict[str, np.ndarray] = {}
        skew: dict[str, np.ndarray] = {}
        native_indices = _nearest_indices(native, sample)
        indices["state"] = native_indices
        indices["desired"] = native_indices.copy()
        indices["achieved"] = native_indices.copy()
        indices["command"] = native_indices.copy()
        native_skew = native[native_indices] - sample
        skew["state"] = native_skew
        skew["desired"] = native_skew.copy()
        skew["achieved"] = native_skew.copy()
        skew["command"] = native_skew.copy()

        effective_rates: dict[str, Optional[float]] = {"native": None}
        declared_rates: dict[str, Optional[float]] = {"native": None}
        dropped: dict[str, int] = {}
        non_monotonic: dict[str, int] = {}
        discontinuities: dict[str, tuple[int, ...]] = {}
        warnings: list[str] = []
        for name, stream in streams.items():
            stream_indices = _nearest_indices(stream.timestamps, sample)
            indices[f"camera:{name}"] = stream_indices
            stream_skew = stream.timestamps[stream_indices] - sample
            skew[f"camera:{name}"] = stream_skew
            effective_rates[name] = stream.effective_rate_hz
            declared_rates[name] = stream.declared_fps
            bad_count, gaps, dropped_count = _timestamp_issues(
                stream.timestamps, gap_factor=self.config.gap_factor
            )
            non_monotonic[name] = bad_count
            discontinuities[name] = gaps
            dropped[name] = int(dropped_count or 0)
            if stream.declared_fps is not None and stream.effective_rate_hz is not None:
                relative_error = abs(stream.effective_rate_hz - stream.declared_fps) / stream.declared_fps
                if relative_error > self.config.declared_rate_tolerance:
                    warnings.append("DECLARED_EFFECTIVE_RATE_MISMATCH")

        native_bad_count, native_gaps, native_dropped = _timestamp_issues(
            native, gap_factor=self.config.gap_factor
        )
        non_monotonic["native"] = native_bad_count
        discontinuities["native"] = native_gaps
        dropped["native"] = int(native_dropped or 0)
        max_skew = {
            key: _stats(np.abs(value)) for key, value in skew.items()
        }
        metrics = {
            "camera_state_skew_s": max_skew.get(f"camera:{self.config.sample_stream}", _stats([])),
            "command_state_skew_s": max_skew.get("command", _stats([])),
            "all_skew_s": max_skew,
            "max_abs_skew_s": float(max((np.max(np.abs(value)) for value in skew.values()), default=0.0)),
            "declared_rate_mismatch": [
                name
                for name, rate in effective_rates.items()
                if name in streams
                and rate is not None
                and streams[name].declared_fps is not None
                and abs(rate - streams[name].declared_fps) / streams[name].declared_fps
                > self.config.declared_rate_tolerance
            ],
        }
        return SynchronizationResultV1(
            episode_id=raw.episode_id,
            sample_stream=self.config.sample_stream,
            sample_timestamps=sample,
            indices=indices,
            skew_seconds=skew,
            effective_rates_hz=effective_rates,
            declared_rates_hz=declared_rates,
            dropped_frames=dropped,
            non_monotonic_counts=non_monotonic,
            clock_discontinuities=discontinuities,
            warnings=tuple(dict.fromkeys(warnings)),
            metrics=metrics,
            provenance={
                "synchronizer": "SynchronizerV1",
                "source_timestamps_preserved": True,
                "state_alignment": "nearest_source_sample",
                "rotation_interpolation": "SLERP_ONLY_IF_EXPLICITLY_REQUESTED",
                "production_action_rate": "TBD_BY_ADAPTATION_DATASET",
            },
        )


def interpolate_pose_rot6d_rows(
    source_timestamps: Any,
    source_pose_9d: Any,
    query_timestamps: Any,
) -> np.ndarray:
    """Interpolate XYZ linearly and orientation with quaternion SLERP.

    Rot6d is decoded to SO(3) before interpolation and repacked afterward;
    this function never linearly interpolates the six rotation numbers.
    Queries must lie within the source interval.
    """

    source_t = _strict_timestamps("source_timestamps", source_timestamps)
    pose = _readonly_array(source_pose_9d, name="source_pose_9d")
    if pose.shape != (source_t.size, 9):
        raise ValueError("source_pose_9d must have shape [N,9]")
    _finite("source_pose_9d", pose)
    matrices = _validate_rot6d_rows("source_pose_9d.rot6d", pose[:, 3:9])
    quaternions = matrix_to_quat(matrices)
    query = _readonly_array(query_timestamps, name="query_timestamps")
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query_timestamps must be finite [M]")
    if np.any(query < source_t[0] - 1e-10) or np.any(query > source_t[-1] + 1e-10):
        raise ValueError("query_timestamps must lie inside source timestamp interval")
    result = np.empty((query.size, 9), dtype=np.float64)
    for output_index, value in enumerate(query):
        right = int(np.searchsorted(source_t, value, side="left"))
        right = min(max(right, 0), source_t.size - 1)
        if right == 0 or np.isclose(value, source_t[right], atol=1e-12):
            left = right
        else:
            left = right - 1
        if left == right:
            alpha = 0.0
        else:
            alpha = float((value - source_t[left]) / (source_t[right] - source_t[left]))
        position = (1.0 - alpha) * pose[left, :3] + alpha * pose[right, :3]
        quaternion = quat_slerp(quaternions[left], quaternions[right], alpha)
        rotation = quat_to_matrix(quaternion)
        result[output_index] = np.concatenate([position, rotation_matrix_to_rot6d_rows(rotation)])
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StateV1:
    """Primary model-facing state ABI: EEF XYZ + row-rot6d + gripper."""

    eef_xyz: np.ndarray
    eef_rot6d_rows: np.ndarray
    gripper: np.ndarray
    rot6d_layout: str = GROOT_EEF_ROT6D_LAYOUT

    def __post_init__(self) -> None:
        xyz = _shape("StateV1.eef_xyz", self.eef_xyz, (3,))
        rot = _shape("StateV1.eef_rot6d_rows", self.eef_rot6d_rows, (6,))
        grip = _shape("StateV1.gripper", self.gripper, (1,))
        if self.rot6d_layout != GROOT_EEF_ROT6D_LAYOUT:
            raise ValueError(f"StateV1 requires {GROOT_EEF_ROT6D_LAYOUT}")
        _validate_rot6d_rows("StateV1.eef_rot6d_rows", rot)
        object.__setattr__(self, "eef_xyz", xyz)
        object.__setattr__(self, "eef_rot6d_rows", rot)
        object.__setattr__(self, "gripper", grip)

    @classmethod
    def from_array(cls, value: Any) -> "StateV1":
        array = _shape("StateV1.array", value, (10,))
        return cls(array[:3], array[3:9], array[9:10])

    def as_array(self) -> np.ndarray:
        value = np.concatenate([self.eef_xyz, self.eef_rot6d_rows, self.gripper])
        value.setflags(write=False)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "StateV1",
            "layout": ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"],
            "rot6d_layout": self.rot6d_layout,
            "values": self.as_array().tolist(),
        }


@dataclass(frozen=True)
class GR00TTrainingExampleV1:
    """Derived model-facing sample with desired, timestamped future labels."""

    example_id: str
    raw_episode_id: str
    observation_timestamp: float
    observation_images: Mapping[str, str]
    task_instruction: str
    state_v1: StateV1
    future_timestamps: np.ndarray
    future_desired_eef_9d: np.ndarray
    future_gripper: np.ndarray
    calibration_bundle_id: str
    source_frame: str
    tcp_semantic_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.raw_episode_id:
            raise ValueError("example_id and raw_episode_id are required")
        if not np.isfinite(float(self.observation_timestamp)):
            raise ValueError("observation_timestamp must be finite")
        images = {str(key): str(value) for key, value in self.observation_images.items()}
        if any(not key or not value for key, value in images.items()):
            raise ValueError("observation image paths must be non-empty")
        timestamps = _strict_timestamps("future_timestamps", self.future_timestamps)
        desired = _shape("future_desired_eef_9d", self.future_desired_eef_9d, (timestamps.size, 9))
        gripper = _shape("future_gripper", self.future_gripper, (timestamps.size, 1))
        _validate_rot6d_rows("future_desired_eef_9d.rot6d", desired[:, 3:9])
        if not self.calibration_bundle_id or not self.source_frame or not self.tcp_semantic_id:
            raise ValueError("calibration, frame, and TCP semantics are required")
        object.__setattr__(self, "observation_images", _freeze(images))
        object.__setattr__(self, "future_timestamps", timestamps)
        object.__setattr__(self, "future_desired_eef_9d", desired)
        object.__setattr__(self, "future_gripper", gripper)
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def horizon(self) -> int:
        return int(self.future_timestamps.size)

    @property
    def physical_action(self) -> np.ndarray:
        value = np.concatenate([self.future_desired_eef_9d, self.future_gripper], axis=-1)
        value.setflags(write=False)
        return value

    @classmethod
    def from_raw(
        cls,
        raw: RawEpisodeV1,
        synchronization: SynchronizationResultV1,
        *,
        sample_index: int = 0,
        horizon: int = 40,
    ) -> "GR00TTrainingExampleV1":
        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= synchronization.sample_count:
            raise IndexError("sample_index is outside synchronization result")
        native_start = int(synchronization.indices["state"][sample_index])
        native_end = native_start + int(horizon)
        if native_end > raw.frame_count:
            raise ValueError("episode does not contain the requested future horizon")
        image_paths: dict[str, str] = {}
        for name in ("body", "wrist"):
            if name in raw.camera_streams:
                camera_index = int(synchronization.indices[f"camera:{name}"][sample_index])
                image_paths[name] = raw.camera_streams[name].frame_paths[camera_index]
        if not image_paths:
            raise ValueError("at least one camera stream is required for a training example")
        state_index = native_start
        return cls(
            example_id=f"{raw.episode_id}:sample_{sample_index:06d}:h{int(horizon)}",
            raw_episode_id=raw.episode_id,
            observation_timestamp=float(synchronization.sample_timestamps[sample_index]),
            observation_images=image_paths,
            task_instruction=str(raw.task.get("instruction", "")),
            state_v1=StateV1.from_array(
                np.concatenate([raw.state_eef_9d[state_index], raw.state_gripper[state_index]])
            ),
            future_timestamps=raw.native_timestamps[native_start:native_end],
            future_desired_eef_9d=raw.desired_eef_9d[native_start:native_end],
            future_gripper=raw.desired_gripper[native_start:native_end],
            calibration_bundle_id=raw.calibration_bundle_id,
            source_frame=raw.source_frame,
            tcp_semantic_id=raw.tcp_semantic_id,
            provenance={
                "builder": "GR00TTrainingExampleV1.from_raw",
                "raw_episode_id": raw.episode_id,
                "sample_index": sample_index,
                "native_start_index": native_start,
                "horizon": int(horizon),
                "label_semantics": "future_desired_EEF_XYZ_rot6d_rows_plus_gripper",
                "achieved_eef_role": "audit_only_never_training_target",
                "timestamp_semantics": "native_source_timestamps",
            },
        )

    def to_lerobot_record(self) -> dict[str, Any]:
        """Return a dependency-free LeRobot-compatible logical record.

        The per-frame ``action`` is the first desired command.  The full
        future action trajectory remains explicitly available under
        ``future_action`` and ``action_timestamps`` for the official GR00T
        processor/chunking layer.
        """

        first_action = self.physical_action[0]
        return {
            "timestamp": float(self.observation_timestamp),
            "observation.state": self.state_v1.as_array().tolist(),
            "action": first_action.tolist(),
            "future_action": self.physical_action.tolist(),
            "action_timestamps": self.future_timestamps.tolist(),
            "observation.images.external_rgb": self.observation_images.get("body"),
            "observation.images.wrist_rgb": self.observation_images.get("wrist"),
            "task": self.task_instruction,
            "metadata": {
                "schema": "GR00TTrainingExampleV1",
                "example_id": self.example_id,
                "calibration_bundle_id": self.calibration_bundle_id,
                "source_frame": self.source_frame,
                "tcp_semantic_id": self.tcp_semantic_id,
                "rot6d_layout": GROOT_EEF_ROT6D_LAYOUT,
                "future_label": "desired",
                "achieved_stream_in_raw_episode": True,
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "GR00TTrainingExampleV1",
            "example_id": self.example_id,
            "raw_episode_id": self.raw_episode_id,
            "observation_timestamp": float(self.observation_timestamp),
            "observation_images": _jsonable(self.observation_images),
            "task_instruction": self.task_instruction,
            "state_v1": self.state_v1.as_dict(),
            "future_timestamps": self.future_timestamps.tolist(),
            "future_desired_eef_9d": self.future_desired_eef_9d.tolist(),
            "future_gripper": self.future_gripper.tolist(),
            "physical_action_shape": list(self.physical_action.shape),
            "calibration_bundle_id": self.calibration_bundle_id,
            "source_frame": self.source_frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "provenance": _jsonable(self.provenance),
        }
