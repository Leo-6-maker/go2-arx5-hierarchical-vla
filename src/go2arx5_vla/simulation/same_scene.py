"""Simulator-independent same-scene camera observation contracts.

Native IsaacGym code supplies the actual RGB arrays.  These contracts make
the provenance claim explicit: camera pixels, target marker, robot state, and
WBC episode must carry one simulator-instance identity.  They do not provide
physical calibration or a real-robot camera limit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..data.readiness import StateV1


def _image(name: str, value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"{name} must have shape [H,W,3]")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} must be numeric RGB data")
    copy = image.copy()
    copy.setflags(write=False)
    return copy


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


@dataclass(frozen=True)
class SameSceneTargetV1:
    target_id: str
    simulator_instance_id: str
    position_world_m: np.ndarray
    radius_m: float = 0.035
    color_rgb: tuple[int, int, int] = (230, 24, 20)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id or not self.simulator_instance_id:
            raise ValueError("target_id and simulator_instance_id are required")
        position = np.asarray(self.position_world_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_world_m must be finite [3]")
        if not np.isfinite(float(self.radius_m)) or float(self.radius_m) <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        color = tuple(int(value) for value in self.color_rgb)
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError("color_rgb must contain three values in [0,255]")
        object.__setattr__(self, "position_world_m", position.copy())
        object.__setattr__(self, "color_rgb", color)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "SameSceneTargetV1",
            "target_id": self.target_id,
            "simulator_instance_id": self.simulator_instance_id,
            "position_world_m": self.position_world_m.tolist(),
            "radius_m": float(self.radius_m),
            "color_rgb": list(self.color_rgb),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class NativeSceneObservationV1:
    """One same-scene observation supplied to the VLA boundary."""

    observation_id: str
    simulator_instance_id: str
    simulation_time_s: float
    body_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: StateV1
    language: str
    target: SameSceneTargetV1
    source_frame: str
    tcp_semantic_id: str
    native_wbc_active: bool = True
    camera_provenance: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id or not self.simulator_instance_id or not self.language:
            raise ValueError("observation_id, simulator_instance_id, and language are required")
        if self.target.simulator_instance_id != self.simulator_instance_id:
            raise ValueError("target and observation must share simulator_instance_id")
        if not np.isfinite(float(self.simulation_time_s)) or float(self.simulation_time_s) < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        body = _image("body_rgb", self.body_rgb)
        wrist = _image("wrist_rgb", self.wrist_rgb)
        if not self.source_frame or not self.tcp_semantic_id:
            raise ValueError("source_frame and tcp_semantic_id are required")
        object.__setattr__(self, "body_rgb", body)
        object.__setattr__(self, "wrist_rgb", wrist)
        object.__setattr__(self, "camera_provenance", dict(self.camera_provenance))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def same_scene(self) -> bool:
        return bool(self.native_wbc_active)

    @property
    def image_hashes(self) -> dict[str, str]:
        return {"body": image_sha256(self.body_rgb), "wrist": image_sha256(self.wrist_rgb)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "NativeSceneObservationV1",
            "observation_id": self.observation_id,
            "simulator_instance_id": self.simulator_instance_id,
            "simulation_time_s": float(self.simulation_time_s),
            "body_rgb_shape": list(self.body_rgb.shape),
            "wrist_rgb_shape": list(self.wrist_rgb.shape),
            "image_sha256": self.image_hashes,
            "state": self.state.as_array().tolist(),
            "state_abi": "StateV1=EEF_XYZ_3+GR00T_FIRST_TWO_ROWS_ROT6D_6+GRIPPER_1",
            "language": self.language,
            "target": self.target.as_dict(),
            "source_frame": self.source_frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "native_wbc_active": bool(self.native_wbc_active),
            "same_scene": self.same_scene,
            "camera_provenance": dict(self.camera_provenance),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SameSceneCounterfactualV1:
    """Audit result for image/location counterfactual construction."""

    observation_a_id: str
    observation_b_id: str
    simulator_instance_id: str
    same_state: bool
    same_language: bool
    different_target_location: bool
    different_image: bool
    state_max_abs_error: float
    target_translation_m: float
    body_image_changed: bool
    wrist_image_changed: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "SameSceneCounterfactualV1",
            "observation_a_id": self.observation_a_id,
            "observation_b_id": self.observation_b_id,
            "simulator_instance_id": self.simulator_instance_id,
            "same_state": bool(self.same_state),
            "same_language": bool(self.same_language),
            "different_target_location": bool(self.different_target_location),
            "different_image": bool(self.different_image),
            "state_max_abs_error": float(self.state_max_abs_error),
            "target_translation_m": float(self.target_translation_m),
            "body_image_changed": bool(self.body_image_changed),
            "wrist_image_changed": bool(self.wrist_image_changed),
            "provenance": dict(self.provenance),
        }


def compare_counterfactuals(
    observation_a: NativeSceneObservationV1,
    observation_b: NativeSceneObservationV1,
    *,
    state_atol: float = 1e-7,
) -> SameSceneCounterfactualV1:
    if observation_a.simulator_instance_id != observation_b.simulator_instance_id:
        raise ValueError("counterfactual observations must share simulator_instance_id")
    a_state = observation_a.state.as_array()
    b_state = observation_b.state.as_array()
    state_error = float(np.max(np.abs(a_state - b_state)))
    target_translation = float(
        np.linalg.norm(observation_a.target.position_world_m - observation_b.target.position_world_m)
    )
    body_changed = image_sha256(observation_a.body_rgb) != image_sha256(observation_b.body_rgb)
    wrist_changed = image_sha256(observation_a.wrist_rgb) != image_sha256(observation_b.wrist_rgb)
    return SameSceneCounterfactualV1(
        observation_a_id=observation_a.observation_id,
        observation_b_id=observation_b.observation_id,
        simulator_instance_id=observation_a.simulator_instance_id,
        same_state=bool(state_error <= float(state_atol)),
        same_language=observation_a.language == observation_b.language,
        different_target_location=bool(target_translation > 1e-6),
        different_image=bool(body_changed or wrist_changed),
        state_max_abs_error=state_error,
        target_translation_m=target_translation,
        body_image_changed=body_changed,
        wrist_image_changed=wrist_changed,
        provenance={
            "comparison": "same_robot_state_different_marker_pose_same_simulator_instance",
            "state_atol": float(state_atol),
        },
    )
