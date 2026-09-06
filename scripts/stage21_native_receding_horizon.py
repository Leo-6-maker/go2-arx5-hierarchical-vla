#!/usr/bin/env python3
"""Stage21 native same-scene camera and GT-first receding-horizon runner.

The script is deliberately an integration harness around the delivered native
IsaacGym WBC.  It adds only project-owned observation and reference-update
bookkeeping:

* a fixed red target actor and body/wrist camera sensors in the same scene;
* a reference-only chunk replacement path; and
* a predeclared fixed-prefix schedule for the GT desired EEF trajectory.

No reset, physics teleport, policy-history clear, trajectory clipping, or
transport is performed at a reference update.  Actual GR00T inference is a
separate optional mode and is never forwarded to WBC by this runner without a
successful pre-execution safety-gate decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.data import StateV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.runtime import RecedingHorizonExecutorV1
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1
from go2arx5_vla.simulation import (
    NativeSceneObservationV1,
    SameSceneTargetV1,
    compare_counterfactuals,
)
from go2arx5_vla.trajectory.so3 import (
    quat_slerp,
    quat_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
LANGUAGE = "move the end effector to the red target"
TEST_ACTION_PERIOD_S = 0.05
SIMULATOR = "IsaacGym_native_WBC_same_scene"


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _tensor_hash(value: Any) -> str:
    array = value.detach().cpu().numpy()
    return _sha256_array(np.asarray(array))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _load_pose_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("timestamp", "position", "quaternion")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("trajectory is missing: " + ", ".join(missing))
        timestamp = np.asarray(data["timestamp"], dtype=np.float64)
        position = np.asarray(data["position"], dtype=np.float64)
        quaternion = np.asarray(data["quaternion"], dtype=np.float64)
    if timestamp.ndim != 1 or len(timestamp) < 2:
        raise ValueError("trajectory timestamp must be a one-dimensional sequence with at least 2 points")
    if position.shape != (len(timestamp), 3) or quaternion.shape != (len(timestamp), 4):
        raise ValueError("trajectory position/quaternion shapes must be [T,3]/[T,4]")
    if not np.isfinite(timestamp).all() or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    return {"timestamp": timestamp, "position": position, "quaternion": quaternion}


def _pose_window(
    trajectory: Mapping[str, np.ndarray],
    *,
    start_time_s: float,
    period_s: float = TEST_ACTION_PERIOD_S,
    points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a future chunk without changing the parent trajectory."""

    if not np.isfinite(float(start_time_s)):
        raise ValueError("start_time_s must be finite")
    if float(period_s) <= 0.0 or int(points) < 2:
        raise ValueError("period_s must be positive and points must be >= 2")
    source_t = trajectory["timestamp"] - float(trajectory["timestamp"][0])
    query_t = float(start_time_s) + np.arange(int(points), dtype=np.float64) * float(period_s)
    clipped = np.clip(query_t, source_t[0], source_t[-1])
    position = np.stack(
        [np.interp(clipped, source_t, trajectory["position"][:, axis]) for axis in range(3)],
        axis=-1,
    )
    quaternion = np.empty((len(clipped), 4), dtype=np.float64)
    for index, value in enumerate(clipped):
        if value <= source_t[0]:
            quaternion[index] = trajectory["quaternion"][0]
            continue
        if value >= source_t[-1]:
            quaternion[index] = trajectory["quaternion"][-1]
            continue
        right = int(np.searchsorted(source_t, value, side="right"))
        left = max(0, min(right - 1, len(source_t) - 2))
        alpha = float((value - source_t[left]) / (source_t[left + 1] - source_t[left]))
        quaternion[index] = quat_slerp(
            trajectory["quaternion"][left], trajectory["quaternion"][left + 1], alpha
        )
    point_timestamps = float(start_time_s) + np.arange(int(points), dtype=np.float64) * float(period_s)
    return point_timestamps, position, quaternion


def _chunk_payload_equivalent(left: Any, right: Any) -> bool:
    return (
        np.array_equal(np.asarray(left.point_timestamps), np.asarray(right.point_timestamps))
        and np.array_equal(np.asarray(left.eef_position), np.asarray(right.eef_position))
        and np.array_equal(np.asarray(left.eef_orientation), np.asarray(right.eef_orientation))
        and np.array_equal(np.asarray(left.gripper), np.asarray(right.gripper))
        and left.source_frame == right.source_frame
        and left.tcp_semantic_id == right.tcp_semantic_id
    )


def _make_chunk(
    trajectory: Mapping[str, np.ndarray],
    *,
    chunk_id: str,
    observation_timestamp: float,
    start_time_s: float,
    source: str = "Stage16_GT_DESIRED_EEF_REPLAY",
    period_s: float = TEST_ACTION_PERIOD_S,
) -> TrajectoryChunkV1:
    timestamps, position, quaternion = _pose_window(
        trajectory, start_time_s=start_time_s, period_s=period_s, points=40
    )
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source=source,
        observation_timestamp=float(observation_timestamp),
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=quaternion,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=np.full(40, 0.5, dtype=np.float64),
        provenance={
            "stage": "Stage21",
            "reference_kind": "GT_DESIRED_EEF_REPLAY",
            "parent_trajectory": "Stage16_E1_W0_00",
            "point_timestamp_semantics": "absolute_test_sim_clock",
            "action_sample_period_s": float(period_s),
            "action_sample_period_provenance": "TEST_ONLY_STAGE21_SIMULATION",
            "no_spatial_correction": True,
        },
    )


def _chunk_profile(chunk: TrajectoryChunkV1, achieved_position: np.ndarray | None = None) -> dict[str, Any]:
    positions = np.asarray(chunk.eef_position, dtype=np.float64)
    rotations = quat_to_matrix(np.asarray(chunk.eef_orientation, dtype=np.float64))
    dt = np.diff(np.asarray(chunk.point_timestamps, dtype=np.float64))
    translation_step = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    angular_step = np.asarray(
        2.0
        * np.arccos(
            np.clip(
                np.abs(np.sum(chunk.eef_orientation[:-1] * chunk.eef_orientation[1:], axis=-1)),
                -1.0,
                1.0,
            )
        ),
        dtype=np.float64,
    )
    linear_speed = translation_step / np.maximum(dt, 1e-12)
    angular_speed = angular_step / np.maximum(dt, 1e-12)
    second_difference = np.diff(positions, n=2, axis=0)
    result: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "point_count": chunk.point_count,
        "source": chunk.source,
        "path_length_m": float(translation_step.sum()),
        "duration_s": float(chunk.validity_end - chunk.validity_start),
        "translation_step_m": {
            "max": float(translation_step.max()) if len(translation_step) else 0.0,
            "p95": float(np.percentile(translation_step, 95)) if len(translation_step) else 0.0,
        },
        "rotation_step_rad": {
            "max": float(angular_step.max()) if len(angular_step) else 0.0,
            "p95": float(np.percentile(angular_step, 95)) if len(angular_step) else 0.0,
        },
        "linear_speed_mps": {
            "mean": float(linear_speed.mean()) if len(linear_speed) else 0.0,
            "max": float(linear_speed.max()) if len(linear_speed) else 0.0,
        },
        "angular_speed_radps": {
            "mean": float(angular_speed.mean()) if len(angular_speed) else 0.0,
            "max": float(angular_speed.max()) if len(angular_speed) else 0.0,
        },
        "roughness_position_second_difference_m": {
            "mean": float(np.linalg.norm(second_difference, axis=-1).mean())
            if len(second_difference)
            else 0.0,
            "max": float(np.linalg.norm(second_difference, axis=-1).max())
            if len(second_difference)
            else 0.0,
        },
        "finite": bool(np.isfinite(positions).all() and np.isfinite(chunk.eef_orientation).all()),
        "trajectory_hash": _sha256_array(
            np.concatenate(
                [
                    np.asarray(chunk.point_timestamps, dtype=np.float64)[:, None],
                    positions,
                    np.asarray(chunk.eef_orientation, dtype=np.float64),
                ],
                axis=-1,
            )
        ),
    }
    if achieved_position is not None:
        result["initial_position_discontinuity_m"] = float(
            np.linalg.norm(positions[0] - np.asarray(achieved_position, dtype=np.float64))
        )
    return result


def _current_eef(native: Any) -> tuple[np.ndarray, np.ndarray]:
    state = native.end_effector_state[0].detach().cpu().numpy().copy()
    return state[:3], state[3:7]


def _current_state_v1(native: Any) -> StateV1:
    position, quaternion = _current_eef(native)
    anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
    rotation_rows = rotation_matrix_to_rot6d_rows(quat_to_matrix(quaternion))
    return StateV1(
        eef_xyz=position - anchor,
        eef_rot6d_rows=rotation_rows,
        gripper=np.asarray([0.5], dtype=np.float64),
    )


def _native_runtime_telemetry(native: Any) -> dict[str, Any]:
    """Return non-invasive whole-body telemetry for a post-step row.

    Stage21 only replaces the external reference. This helper reads the
    delivered native state after the policy step so reports can separate EEF
    tracking from base/joint behavior without changing controller state.
    Missing optional fields are omitted rather than fabricated.
    """

    def row_array(name: str) -> np.ndarray | None:
        value = getattr(native, name, None)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value[0].detach().cpu().numpy()
        else:
            value = np.asarray(value)
            if value.ndim > 1:
                value = value[0]
        return np.asarray(value).copy()

    telemetry: dict[str, Any] = {}
    for name in ("base_quat", "dof_pos", "dof_vel", "joint_pos_target"):
        value = row_array(name)
        if value is not None:
            telemetry[name] = value.tolist()
    dof_pos = np.asarray(telemetry["dof_pos"], dtype=np.float64) if "dof_pos" in telemetry else None
    dof_vel = np.asarray(telemetry["dof_vel"], dtype=np.float64) if "dof_vel" in telemetry else None
    if dof_pos is not None and dof_pos.ndim == 1 and dof_pos.size >= 18:
        telemetry["arm_q"] = dof_pos[12:18].tolist()
        telemetry["dog_q"] = dof_pos[:12].tolist()
    if dof_vel is not None and dof_vel.ndim == 1 and dof_vel.size >= 18:
        telemetry["arm_dq"] = dof_vel[12:18].tolist()
        telemetry["dog_dq"] = dof_vel[:12].tolist()
    limits = getattr(native, "dof_pos_limits", None)
    if limits is not None and dof_pos is not None:
        bounds = limits.detach().cpu().numpy() if hasattr(limits, "detach") else np.asarray(limits)
        bounds = np.asarray(bounds)
        if bounds.ndim == 3:
            bounds = bounds[0]
        if bounds.shape == (dof_pos.size, 2):
            margin = np.minimum(dof_pos - bounds[:, 0], bounds[:, 1] - dof_pos)
            telemetry["joint_limit_margin"] = margin.astype(np.float64).tolist()
            telemetry["joint_limit_violation"] = bool(np.any(margin < 0.0))
    state_arrays = [value for value in (dof_pos, dof_vel) if value is not None]
    telemetry["invalid_state"] = bool(state_arrays and not all(np.isfinite(value).all() for value in state_arrays))
    body_height = getattr(native, "body_height_buf", None)
    if body_height is not None:
        if hasattr(body_height, "detach"):
            body_height = bool(body_height[0].detach().cpu().item())
        else:
            body_height = bool(np.asarray(body_height).reshape(-1)[0])
        telemetry["fall_flag"] = body_height
    joint_term = getattr(native, "joint_limit_termination_buf", None)
    if joint_term is not None:
        if hasattr(joint_term, "detach"):
            joint_term = bool(joint_term[0].detach().cpu().item())
        else:
            joint_term = bool(np.asarray(joint_term).reshape(-1)[0])
        telemetry["joint_limit_termination"] = joint_term
    return telemetry


def _camera_image(native: Any, handle: Any) -> np.ndarray:
    if native.device != "cpu":
        native.gym.fetch_results(native.sim, True)
    native.gym.step_graphics(native.sim)
    native.gym.render_all_camera_sensors(native.sim)
    raw = np.asarray(
        native.gym.get_camera_image(native.sim, native.envs[0], handle, native._stage21_gymapi.IMAGE_COLOR),
        dtype=np.uint8,
    )
    expected = int(native.camera_props.height) * int(native.camera_props.width) * 4
    if raw.size != expected:
        raise RuntimeError(f"native camera returned {raw.size} bytes, expected {expected}")
    return raw.reshape(int(native.camera_props.height), int(native.camera_props.width), 4)[..., :3].copy()


def _capture_scene(native: Any) -> tuple[np.ndarray, np.ndarray]:
    if native.device != "cpu":
        native.gym.fetch_results(native.sim, True)
    native.gym.step_graphics(native.sim)
    native.gym.render_all_camera_sensors(native.sim)
    images = []
    for name in ("body", "wrist"):
        raw = np.asarray(
            native.gym.get_camera_image(
                native.sim,
                native.envs[0],
                native.stage21_camera_handles[name],
                native._stage21_gymapi.IMAGE_COLOR,
            ),
            dtype=np.uint8,
        )
        expected = int(native.camera_props.height) * int(native.camera_props.width) * 4
        if raw.size != expected:
            raise RuntimeError(f"{name} camera returned {raw.size} bytes, expected {expected}")
        images.append(raw.reshape(int(native.camera_props.height), int(native.camera_props.width), 4)[..., :3].copy())
    return images[0], images[1]


def _red_pixel_count(image: np.ndarray) -> int:
    rgb = np.asarray(image, dtype=np.uint8)
    red = (rgb[..., 0] > 100) & (rgb[..., 0] > rgb[..., 1] * 1.35) & (rgb[..., 0] > rgb[..., 2] * 1.35)
    return int(np.count_nonzero(red))


def _observation(
    native: Any,
    *,
    observation_id: str,
    target_id: str,
    target_position: np.ndarray,
    body_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    simulator_instance_id: str,
) -> NativeSceneObservationV1:
    return NativeSceneObservationV1(
        observation_id=observation_id,
        simulator_instance_id=simulator_instance_id,
        simulation_time_s=float(native.traj_sim_time[0].detach().cpu().item()),
        body_rgb=body_rgb,
        wrist_rgb=wrist_rgb,
        state=_current_state_v1(native),
        language=LANGUAGE,
        target=SameSceneTargetV1(
            target_id=target_id,
            simulator_instance_id=simulator_instance_id,
            position_world_m=np.asarray(target_position, dtype=np.float64),
            color_rgb=(230, 24, 20),
            provenance={"actor": "stage21_red_target", "same_simulator_world": True},
        ),
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        native_wbc_active=True,
        camera_provenance={
            "simulator": SIMULATOR,
            "body_camera": "native_attached_to_base",
            "wrist_camera": "native_attached_to_x5_link6",
            "image_color_format": "RGB_uint8_HWC",
        },
        provenance={
            "stage": "Stage21",
            "observation_contract": "NativeSceneObservationV1",
            "hardware_control": False,
        },
    )


def _write_observation(output_dir: Path, observation: NativeSceneObservationV1) -> None:
    stem = output_dir / observation.observation_id
    np.savez_compressed(
        stem.with_suffix(".npz"),
        body_rgb=observation.body_rgb,
        wrist_rgb=observation.wrist_rgb,
        state=observation.state.as_array(),
        target_position_world_m=observation.target.position_world_m,
    )
    stem.with_suffix(".json").write_text(json.dumps(observation.as_dict(), indent=2) + "\n")


class _Stage21NativeEnvFactory:
    """Build the subclass after IsaacGym/WBC imports are available."""

    @staticmethod
    def make(WBCEnv: Any, initial_chunk: TrajectoryChunkV1, torch: Any, gymapi: Any, gymtorch: Any):
        from native_wbc_benchmark import install_external_env

        ExternalReferenceWBCEnv = install_external_env(
            WBCEnv,
            Path("stage21_initial_reference.npz"),
            torch,
            "BENCHMARK_CONTINUOUS",
        )

        class Stage21NativeWBCEnv(ExternalReferenceWBCEnv):
            def __init__(self, *args, **kwargs):
                self._stage21_initial_chunk = initial_chunk
                self._stage21_current_chunk = None
                self._stage21_anchor_world = None
                self._stage21_load_events: list[dict[str, Any]] = []
                self._stage21_reference_phase = "STARTUP"
                self._stage21_wbc_forward_count = 0
                self._stage21_gymapi = gymapi
                self._stage21_gymtorch = gymtorch
                self._stage21_simulator_instance_id = "native-isaacgym-pid-%d" % os.getpid()
                super().__init__(*args, **kwargs)

            def _create_envs(self):
                super()._create_envs()
                if not hasattr(self, "rendering_camera"):
                    raise RuntimeError(
                        "ONE_SPECIFIC_NATIVE_CAMERA_BLOCKER_REMAINS: delivered runtime did not create body camera"
                    )

                marker_options = gymapi.AssetOptions()
                marker_options.fix_base_link = True
                marker_options.disable_gravity = True
                marker_options.collapse_fixed_joints = True
                marker_asset = self.gym.create_sphere(self.sim, 0.035, marker_options)
                self._stage21_target_position = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
                marker_pose = gymapi.Transform()
                marker_pose.p = gymapi.Vec3(*self._stage21_target_position.tolist())
                self.stage21_target_actor = self.gym.create_actor(
                    self.envs[0], marker_asset, marker_pose, "stage21_red_target", self.num_envs, 0, 0
                )
                self.gym.set_rigid_body_color(
                    self.envs[0],
                    self.stage21_target_actor,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.90, 0.02, 0.01),
                )
                self.stage21_target_root_index = int(
                    self.gym.get_actor_index(self.envs[0], self.stage21_target_actor, gymapi.DOMAIN_SIM)
                )

                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], self.body_names[0]
                )
                wrist_body_name = self.body_names[self.ee_idx]
                wrist_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], wrist_body_name
                )
                body_transform = gymapi.Transform()
                body_transform.p = gymapi.Vec3(1.15, -1.15, 0.95)
                body_transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0.0, 1.0, 0.0), math.radians(135.0)
                )
                self.gym.attach_camera_to_body(
                    self.rendering_camera,
                    self.envs[0],
                    body_handle,
                    body_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
                wrist_props = gymapi.CameraProperties()
                wrist_props.width = int(self.camera_props.width)
                wrist_props.height = int(self.camera_props.height)
                self.stage21_wrist_camera = self.gym.create_camera_sensor(self.envs[0], wrist_props)
                wrist_transform = gymapi.Transform()
                wrist_transform.p = gymapi.Vec3(0.28, -0.10, 0.12)
                wrist_transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0.0, 1.0, 0.0), math.radians(135.0)
                )
                self.gym.attach_camera_to_body(
                    self.stage21_wrist_camera,
                    self.envs[0],
                    wrist_handle,
                    wrist_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
                self.stage21_camera_handles = {
                    "body": self.rendering_camera,
                    "wrist": self.stage21_wrist_camera,
                }

            def _stage21_reset_reference_progress(self, env_ids):
                names = (
                    "traj_s",
                    "traj_s_prev",
                    "traj_sim_time",
                    "traj_d_lat",
                    "traj_sdot_meas",
                    "traj_sdot_ref",
                    "traj_timing_err",
                    "traj_twist_err",
                    "traj_dlat_sum",
                    "traj_timing_abs_sum",
                    "traj_ik_jump_max",
                    "traj_samples",
                    "traj_dlat_sq_sum",
                    "traj_twist_err_sum",
                    "traj_rho_sum",
                    "traj_rho_max",
                    "traj_rho_above_hi_count",
                    "traj_rho_valid_count",
                    "traj_base_util_sum",
                    "traj_base_util_count",
                    "traj_v_ff_sum",
                    "traj_v_base_sum",
                    "traj_motor_power_sum",
                    "traj_manipulability_sum",
                )
                for name in names:
                    value = getattr(self, name, None)
                    if value is not None:
                        value[env_ids] = 0.0
                if hasattr(self, "traj_term_cause"):
                    self.traj_term_cause[env_ids] = False
                if hasattr(self, "traj_early_term"):
                    self.traj_early_term[env_ids] = False
                if hasattr(self, "reverse_buf"):
                    self.reverse_buf[env_ids] = False
                zero = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
                self.arm_goal_pos_world[env_ids] = self.traj_batch.p_at(zero)
                self.arm_goal_quat_world[env_ids] = self.traj_batch.quat_at(zero)

            def _stage21_load_chunk(self, chunk: TrajectoryChunkV1, env_ids, *, initial: bool) -> dict[str, Any]:
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
                if len(env_ids) != 1:
                    raise ValueError("Stage21 runner supports exactly one environment")
                position = torch.as_tensor(chunk.eef_position, dtype=torch.float32)
                quaternion = torch.as_tensor(chunk.eef_orientation, dtype=torch.float32)
                from modules.trajectory import quat_to_mat, to_canonical

                relative_timestamps = np.asarray(chunk.point_timestamps, dtype=np.float64)
                relative_timestamps = relative_timestamps - relative_timestamps[0]
                dt = float(np.median(np.diff(relative_timestamps)))
                gamma, time_law = to_canonical(position, quat_to_mat(quaternion), dt)
                if initial:
                    before_anchor = self.traj_batch.gamma_p[env_ids, 0].clone()
                    self.traj_batch.load(env_ids, [gamma], [time_law])
                    self._anchor_external_reference(env_ids)
                    anchor = self.traj_batch.gamma_p[env_ids, 0] - before_anchor
                    self._stage21_anchor_world = anchor[0].detach().clone()
                else:
                    if self._stage21_anchor_world is None:
                        raise RuntimeError("initial reference anchor has not been established")
                    self.traj_batch.load(env_ids, [gamma], [time_law])
                    self.traj_batch.gamma_p[env_ids] += self._stage21_anchor_world.view(1, 1, 3)
                    anchor = self._stage21_anchor_world.view(1, 3)
                self._stage21_reset_reference_progress(env_ids)
                self._stage21_current_chunk = chunk
                event = {
                    "chunk_id": chunk.chunk_id,
                    "initial": bool(initial),
                    "point_count": chunk.point_count,
                    "dt_s": dt,
                    "anchor_world_m": anchor[0].detach().cpu().tolist(),
                    "trajectory_hash": _sha256_array(
                        np.concatenate(
                            [
                                chunk.point_timestamps[:, None],
                                chunk.eef_position,
                                chunk.eef_orientation,
                            ],
                            axis=-1,
                        )
                    ),
                    "episode_length_after": int(self.episode_length_buf[env_ids[0]].detach().cpu().item()),
                }
                self._stage21_load_events.append(event)
                return event

            def _load_trajectory_for(self, env_ids):
                if not self._traj_tracking_enabled():
                    return super()._load_trajectory_for(env_ids)
                self._stage21_load_chunk(self._stage21_initial_chunk, env_ids, initial=True)

            def stage21_mark_wbc_forward(self) -> None:
                """Record a real policy forward without changing native state."""

                self._stage21_wbc_forward_count += 1
                self._stage21_reference_phase = "RUNNING"

            def initial_bind_reference(self, chunk: TrajectoryChunkV1) -> dict[str, Any]:
                """Bind a fresh pre-forward reference without resetting native state."""

                if self._stage21_wbc_forward_count != 0:
                    raise RuntimeError(
                        "INITIAL_BIND_AFTER_FIRST_WBC_FORWARD: use load_reference_chunk"
                    )
                env_ids = torch.arange(1, device=self.device, dtype=torch.long)
                before_root = _tensor_hash(self.root_states)
                before_dof = _tensor_hash(self.dof_state)
                before_episode = int(self.episode_length_buf[0].detach().cpu().item())
                before_reset = bool(self.reset_buf[0].detach().cpu().item())
                event = self._stage21_load_chunk(chunk, env_ids, initial=False)
                after_root = _tensor_hash(self.root_states)
                after_dof = _tensor_hash(self.dof_state)
                after_episode = int(self.episode_length_buf[0].detach().cpu().item())
                after_reset = bool(self.reset_buf[0].detach().cpu().item())
                event.update(
                    {
                        "lifecycle_phase": "INITIAL_BIND",
                        "reset_buf_before": before_reset,
                        "reset_buf_after": after_reset,
                        "physics_root_state_unchanged": before_root == after_root,
                        "physics_dof_state_unchanged": before_dof == after_dof,
                        "episode_length_unchanged": before_episode == after_episode,
                        "physics_reset": False,
                        "physics_teleport": False,
                        "policy_history_reset": False,
                    }
                )
                self._stage21_reference_phase = "INITIAL_BOUND"
                return event

            def load_reference_chunk(self, chunk: TrajectoryChunkV1) -> dict[str, Any]:
                if self._stage21_wbc_forward_count == 0:
                    raise RuntimeError(
                        "REFERENCE_UPDATE_BEFORE_FIRST_WBC_FORWARD: use initial_bind_reference"
                    )
                if bool(self.reset_buf[0].detach().cpu().item()):
                    raise RuntimeError("ONE_SPECIFIC_REFERENCE_HOT_SWAP_BLOCKER_REMAINS: reset already requested")
                env_ids = torch.arange(1, device=self.device, dtype=torch.long)
                before_root = _tensor_hash(self.root_states)
                before_dof = _tensor_hash(self.dof_state)
                before_episode = int(self.episode_length_buf[0].detach().cpu().item())
                event = self._stage21_load_chunk(chunk, env_ids, initial=False)
                after_root = _tensor_hash(self.root_states)
                after_dof = _tensor_hash(self.dof_state)
                after_episode = int(self.episode_length_buf[0].detach().cpu().item())
                event.update(
                    {
                        "lifecycle_phase": "RUNNING_UPDATE",
                        "physics_root_state_unchanged": before_root == after_root,
                        "physics_dof_state_unchanged": before_dof == after_dof,
                        "episode_length_unchanged": before_episode == after_episode,
                        "physics_reset": False,
                        "physics_teleport": False,
                        "policy_history_reset": False,
                    }
                )
                self._stage21_reference_phase = "RUNNING"
                return event

            @property
            def stage21_anchor_world(self):
                if self._stage21_anchor_world is None:
                    raise RuntimeError("Stage21 anchor is not available before native reset")
                return self._stage21_anchor_world

            @property
            def stage21_simulator_instance_id(self) -> str:
                return self._stage21_simulator_instance_id

            @property
            def stage21_current_chunk(self) -> TrajectoryChunkV1:
                if self._stage21_current_chunk is None:
                    raise RuntimeError("no Stage21 chunk has been loaded")
                return self._stage21_current_chunk

            @property
            def stage21_reference_phase(self) -> str:
                return self._stage21_reference_phase

            @property
            def stage21_wbc_forward_count(self) -> int:
                return int(self._stage21_wbc_forward_count)

        return Stage21NativeWBCEnv


def _configure_stage21(cfg: Any) -> None:
    from native_wbc_benchmark import configure_eval

    configure_eval(cfg)
    cfg.env.record_video = True
    cfg.env.num_recording_envs = 1
    cfg.env.recording_width_px = 96
    cfg.env.recording_height_px = 96
    cfg.env.recording_frame_stride = 1
    cfg.env.episode_length_s = 10000
    cfg.terrain.teleport_robots = False
    cfg.terrain.center_robots = True


def _build_native(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]):
    # The native runtime requires IsaacGym to be imported before torch.
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import isaacgym  # noqa: F401
    import torch
    from isaacgym import gymapi, gymtorch
    from go1_gym.envs.config import RoboDuetRuntimeOptions, build_roboduet_config
    from go1_gym.envs.roboduet.wbc_env import WBCEnv
    from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
    from go1_gym.utils.global_switch import global_switch
    from native_wbc_benchmark import actor_model_classes, safe_load_actor

    torch.set_num_threads(1)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    options = RoboDuetRuntimeOptions(
        num_envs=1,
        robot="go2_x5",
        use_rot6d=True,
        dyna_gait=True,
        goal_reaching=True,
        traj_tracking=True,
        arm_action_mode="end_to_end",
        reach_table=True,
    )
    cfg = build_roboduet_config(options=options, debug=True)
    _configure_stage21(cfg)
    initial_chunk = _make_chunk(
        trajectory,
        chunk_id="gt-0000",
        observation_timestamp=0.0,
        start_time_s=0.0,
    )
    EnvClass = _Stage21NativeEnvFactory.make(WBCEnv, initial_chunk, torch, gymapi, gymtorch)
    global_switch.open_switch()
    native = EnvClass(
        sim_device=args.sim_device,
        headless=True,
        num_envs=1,
        cfg=cfg,
        graphics_device_id=0,
    )
    env = HistoryWrapper(native)
    env.reset()
    # Camera creation is enabled for native sensors, but automatic video
    # capture remains disabled; Stage21 captures explicit observation frames.
    native.pause_recording()

    actors = None
    loads = []
    if args.mode in ("gt_receding", "model_receding"):
        if args.arm is None or args.dog is None:
            raise ValueError("--arm and --dog are required for --mode gt_receding")
        ArmActor, DogActor = actor_model_classes()
        arm = ArmActor()
        dog = DogActor()
        loads = [safe_load_actor("arm", args.arm, arm, torch), safe_load_actor("dog", args.dog, dog, torch)]
        device = torch.device(args.sim_device)
        actors = (arm.to(device), dog.to(device), torch, device)
    return native, env, actors, loads, cfg


def _set_target(native: Any, position: np.ndarray) -> None:
    if not hasattr(native, "root_states"):
        raise RuntimeError("native root state tensor is unavailable")
    index = int(native.stage21_target_root_index)
    native.root_states[index, :3] = native.root_states.new_tensor(np.asarray(position, dtype=np.float32))
    native.root_states[index, 3:7] = native.root_states.new_tensor([0.0, 0.0, 0.0, 1.0])
    native.root_states[index, 7:13] = 0.0
    import torch

    actor_index = torch.as_tensor([index], device=native.device, dtype=torch.int32)
    native.gym.set_actor_root_state_tensor_indexed(
        native.sim,
        native._stage21_gymtorch.unwrap_tensor(native.root_states),
        native._stage21_gymtorch.unwrap_tensor(actor_index),
        1,
    )
    native.gym.refresh_actor_root_state_tensor(native.sim)
    native.gym.refresh_rigid_body_state_tensor(native.sim)
    native._stage21_target_position = np.asarray(position, dtype=np.float64).copy()


def _camera_smoke(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    native = env = None
    result: dict[str, Any] = {
        "schema": "STAGE21_NATIVE_SAME_SCENE_CAMERA_SMOKE_V1",
        "stage": "Stage21",
        "simulation_only": True,
        "hardware_control": False,
        "fixed_primary_reset_seed": int(args.seed),
        "simulator": SIMULATOR,
    }
    try:
        native, env, _, loads, cfg = _build_native(args, trajectory)
        a_position = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
        b_position = np.asarray([0.30, -0.20, 0.58], dtype=np.float64)
        _set_target(native, a_position)
        body_a, wrist_a = _capture_scene(native)
        obs_a = _observation(
            native,
            observation_id="stage21_observation_target_a",
            target_id="red_target_a",
            target_position=a_position,
            body_rgb=body_a,
            wrist_rgb=wrist_a,
            simulator_instance_id=native.stage21_simulator_instance_id,
        )
        _write_observation(output_dir, obs_a)

        # Moving only the fixed-base marker changes the same-scene image. No
        # physics step is inserted between the two observations.
        robot_root_before = _tensor_hash(native.root_states[: native.num_envs])
        robot_dof_before = _tensor_hash(native.dof_state)
        _set_target(native, b_position)
        body_b, wrist_b = _capture_scene(native)
        robot_root_after = _tensor_hash(native.root_states[: native.num_envs])
        robot_dof_after = _tensor_hash(native.dof_state)
        obs_b = _observation(
            native,
            observation_id="stage21_observation_target_b",
            target_id="red_target_b",
            target_position=b_position,
            body_rgb=body_b,
            wrist_rgb=wrist_b,
            simulator_instance_id=native.stage21_simulator_instance_id,
        )
        _write_observation(output_dir, obs_b)

        occluded_body = body_a.copy()
        h, w = occluded_body.shape[:2]
        occluded_body[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
        np.savez_compressed(output_dir / "stage21_observation_target_a_occluded.npz", body_rgb=occluded_body, wrist_rgb=wrist_a, state=obs_a.state.as_array())
        counterfactual = compare_counterfactuals(obs_a, obs_b)
        result.update(
            {
                "status": "PASS" if counterfactual.same_state and counterfactual.different_image else "PASS_WITH_VISIBILITY_DEVIATION",
                "native_camera": {
                    "body_attached": True,
                    "wrist_attached": True,
                    "body_handle": str(native.stage21_camera_handles["body"]),
                    "wrist_handle": str(native.stage21_camera_handles["wrist"]),
                    "resolution": [int(cfg.env.recording_width_px), int(cfg.env.recording_height_px)],
                },
                "same_scene": {
                    "simulator_instance_id": native.stage21_simulator_instance_id,
                    "native_wbc_active": True,
                    "target_actor": "stage21_red_target",
                    "robot_root_unchanged": robot_root_before == robot_root_after,
                    "robot_dof_unchanged": robot_dof_before == robot_dof_after,
                },
                "counterfactual": counterfactual.as_dict(),
                "image_observations": {
                    "target_a": obs_a.as_dict(),
                    "target_b": obs_b.as_dict(),
                    "red_pixel_count": {
                        "body_a": _red_pixel_count(body_a),
                        "body_b": _red_pixel_count(body_b),
                        "wrist_a": _red_pixel_count(wrist_a),
                        "wrist_b": _red_pixel_count(wrist_b),
                    },
                    "occlusion_diagnostic": {
                        "kind": "synthetic_image_occlusion_only",
                        "input": "target_a_body_rgb",
                        "output": "stage21_observation_target_a_occluded.npz",
                    },
                },
                "claim_boundary": "RGB_and_state_are_same_scene; model_visual_grounding_not_yet_evaluated",
            }
        )
        (output_dir / "camera_smoke.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    except Exception as exc:
        result.update(
            {
                "status": "BLOCKED",
                "blocker": {"type": type(exc).__name__, "message": str(exc)},
                "gate": "ONE_SPECIFIC_NATIVE_CAMERA_BLOCKER_REMAINS",
            }
        )
        (output_dir / "camera_smoke.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return result


def _policy_step(native: Any, env: Any, actors: Any) -> tuple[bool, Any]:
    arm, dog, torch, _ = actors
    arm_obs = env.get_arm_observations()
    actions_arm = arm(arm_obs["obs_history"])
    if not torch.isfinite(actions_arm).all():
        raise FloatingPointError("non-finite arm policy output")
    if env.num_plan_actions > 0:
        env.plan(actions_arm)
    dog_obs = env.get_dog_observations()
    actions_dog = dog(dog_obs["obs_history"])
    if not torch.isfinite(actions_dog).all():
        raise FloatingPointError("non-finite dog policy output")
    if hasattr(native, "stage21_mark_wbc_forward"):
        native.stage21_mark_wbc_forward()
    _, _, done, info = env.step(actions_dog, actions_arm[..., : native.num_actions_arm])
    return bool(done.detach().cpu().any().item()), {
        "arm_raw_output": actions_arm[0].detach().cpu().tolist(),
        "arm_action_6d": actions_arm[0, : native.num_actions_arm].detach().cpu().tolist(),
        "internal_body_plan_9d": actions_arm[0, native.num_actions_arm :].detach().cpu().tolist(),
        "dog_action_12d": actions_dog[0].detach().cpu().tolist(),
        "info_keys": sorted(str(key) for key in info.keys()) if hasattr(info, "keys") else [],
    }


def _gt_receding(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    native = env = None
    result: dict[str, Any] = {
        "schema": "STAGE21_GT_FIRST_RECEDING_HORIZON_V1",
        "stage": "Stage21",
        "mode": "SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON",
        "simulation_only": True,
        "hardware_control": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "fixed_primary_reset_seed": int(args.seed),
        "prefix_steps_declared": [4, 6, 8, 10],
        "prefix_steps_frozen": int(args.prefix_steps),
        "action_sample_period_s": TEST_ACTION_PERIOD_S,
        "action_sample_period_provenance": "TEST_ONLY_STAGE21_SIMULATION",
        "simulator": SIMULATOR,
        "trajectory_source": str(args.trajectory),
    }
    rows: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    termination = "SCHEDULE_END"
    try:
        native, env, actors, loads, cfg = _build_native(args, trajectory)
        gate = TrajectorySafetyGateV1()
        native_control_dt_s = float(native.dt)
        if not np.isfinite(native_control_dt_s) or native_control_dt_s <= 0.0:
            raise ValueError(f"native control dt must be positive and finite, got {native_control_dt_s}")
        initial_position, initial_quaternion = _current_eef(native)
        initial_anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
        initial_local_position = initial_position - initial_anchor
        initial_chunk = native.stage21_current_chunk
        initial_gate = gate.evaluate(
            initial_chunk,
            now=float(native.traj_sim_time[0].detach().cpu().item()),
            achieved_position=initial_local_position,
            achieved_orientation=initial_quaternion,
        )
        gate_results.append({"chunk_id": initial_chunk.chunk_id, "result": initial_gate.as_dict()})
        if initial_gate.decision is GateDecision.REJECT:
            raise RuntimeError("GT initial chunk was rejected by TrajectorySafetyGateV1")
        executor = RecedingHorizonExecutorV1(
            prefix_steps=int(args.prefix_steps),
            provenance={
                "stage": "Stage21",
                "mode": "GT_FIRST",
                "schedule": "SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON",
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
            },
        )
        # Native WBC progress is intentionally reference-local and resets when
        # a new chunk is loaded.  Lifecycle timestamps must instead use one
        # monotonic simulator observation clock across the whole episode.
        sim_time = 0.0
        executor.submit_initial(
            initial_chunk,
            receive_timestamp=sim_time,
            execute_timestamp=sim_time,
            current_achieved_position=initial_local_position,
            current_achieved_orientation=initial_quaternion,
        )
        max_steps = (int(args.max_prefix_updates) + 1) * int(args.prefix_steps)
        update_count = 0
        reset_events = 0
        for step in range(max_steps):
            done, policy = _policy_step(native, env, actors)
            sim_time += native_control_dt_s
            event = native.pop_stage16_termination()
            current_position, current_quaternion = _current_eef(native)
            native_reference_time = float(native.traj_sim_time[0].detach().cpu().item())
            row = {
                "step": int(step),
                "sim_time": float(native.traj_sim_time[0].detach().cpu().item()),
                "observation_timestamp": float(sim_time),
                "native_reference_local_time": native_reference_time,
                "chunk_id": native.stage21_current_chunk.chunk_id,
                "update_count": int(update_count),
                "done": bool(done),
                "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
                "episode_length": int(native.episode_length_buf[0].detach().cpu().item()),
                "traj_s": float(native.traj_s[0].detach().cpu().item()),
                "traj_s_ref": float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item()),
                "traj_d_lat": float(native.traj_d_lat[0].detach().cpu().item()),
                "traj_timing_err": float(native.traj_timing_err[0].detach().cpu().item()),
                "traj_L": float(native.traj_batch.L[0].detach().cpu().item()),
                "desired_pos": native.traj_batch.p_at(native.traj_batch.s_ref(native.traj_sim_time))[0].detach().cpu().tolist(),
                "desired_quat": native.traj_batch.quat_at(native.traj_batch.s_ref(native.traj_sim_time))[0].detach().cpu().tolist(),
                "achieved_pos": current_position.tolist(),
                "achieved_quat": current_quaternion.tolist(),
                "base_pos": native.base_pos[0].detach().cpu().tolist(),
                "base_lin_vel": native.base_lin_vel[0].detach().cpu().tolist(),
                "base_ang_vel": native.base_ang_vel[0].detach().cpu().tolist(),
                "internal_body_plan_9d": policy["internal_body_plan_9d"],
                "arm_action_6d": policy["arm_action_6d"],
                "dog_action_12d": policy["dog_action_12d"],
                "termination_code": "NONE",
            }
            row.update(_native_runtime_telemetry(native))
            if event is not None:
                row["termination_event"] = event
                row["termination_code"] = event.get("termination_code", "UNKNOWN")
                termination = row["termination_code"]
                reset_events += 1
            elif done:
                row["termination_code"] = "NATIVE_TASK_TERMINATION"
                termination = "NATIVE_TASK_TERMINATION"
                reset_events += 1
            rows.append(row)
            if done:
                break

            if (step + 1) % int(args.prefix_steps) != 0 or update_count >= int(args.max_prefix_updates):
                continue

            current_chunk = native.stage21_current_chunk
            next_chunk = _make_chunk(
                trajectory,
                chunk_id=f"gt-{update_count + 1:04d}",
                observation_timestamp=sim_time,
                start_time_s=sim_time,
            )
            local_position = current_position - native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
            next_gate = gate.evaluate(
                next_chunk,
                now=sim_time,
                achieved_position=local_position,
                achieved_orientation=current_quaternion,
            )
            gate_results.append({"chunk_id": next_chunk.chunk_id, "result": next_gate.as_dict()})
            if next_gate.decision is GateDecision.REJECT:
                termination = "GT_REFERENCE_REJECTED_BY_GATE"
                break
            previous_index = min(int(args.prefix_steps), current_chunk.point_count - 1)
            previous_position = current_chunk.eef_position[previous_index]
            previous_quaternion = current_chunk.eef_orientation[previous_index]
            arm_history_before = _tensor_hash(env.arm_obs_history)
            dog_history_before = _tensor_hash(env.dog_obs_history)
            update = executor.submit_update(
                next_chunk,
                receive_timestamp=sim_time,
                execute_timestamp=sim_time,
                current_achieved_position=local_position,
                current_achieved_orientation=current_quaternion,
                previous_predicted_position=previous_position,
                previous_predicted_orientation=previous_quaternion,
            )
            native_update = native.load_reference_chunk(next_chunk)
            arm_history_after = _tensor_hash(env.arm_obs_history)
            dog_history_after = _tensor_hash(env.dog_obs_history)
            native_update.update(
                {
                    "chunk_handoff": update.handoff.as_dict(),
                    "lifecycle_status": update.lifecycle_status.value,
                    "arm_policy_history_unchanged": arm_history_before == arm_history_after,
                    "dog_policy_history_unchanged": dog_history_before == dog_history_after,
                    "executor_update": update.as_dict(),
                }
            )
            handoffs.append(native_update)
            update_count += 1

        valid_rows = [row for row in rows if row.get("termination_code") == "NONE"]
        result.update(
            {
                "status": (
                    "GT_RECEDING_HORIZON_INFRASTRUCTURE_PASS"
                    if update_count > 0
                    and reset_events == 0
                    and all(
                        item.get("physics_root_state_unchanged", False)
                        and item.get("physics_dof_state_unchanged", False)
                        and item.get("episode_length_unchanged", False)
                        and item.get("arm_policy_history_unchanged", False)
                        and item.get("dog_policy_history_unchanged", False)
                        for item in handoffs
                    )
                    else "GT_RECEDING_HORIZON_NATIVE_TERMINATION"
                ),
                "termination": termination,
                "steps": len(rows),
                "valid_steps": len(valid_rows),
                "reference_updates": int(update_count),
                "reset_events": int(reset_events),
                "camera_same_scene_available": True,
                "safety_gate": gate_results,
                "handoffs": handoffs,
                "actor_loads": loads,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": actors[2].__version__,
                    "device": str(actors[3]),
                    "native_control_dt_s": native_control_dt_s,
                    "prefix_duration_native_s": native_control_dt_s * int(args.prefix_steps),
                    "chunk_action_sample_period_s": TEST_ACTION_PERIOD_S,
                    "chunk_to_control_rate_ratio": TEST_ACTION_PERIOD_S / native_control_dt_s,
                    "native_anchor_world_m": native.stage21_anchor_world.detach().cpu().numpy().tolist(),
                },
                "claim_boundary": "GT_first_reference_hot_swap_qualification; not actual_GR00T_task_performance",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "BLOCKED" if not rows else "GT_RECEDING_HORIZON_PARTIAL",
                "blocker": {"type": type(exc).__name__, "message": str(exc)},
                "gate": "ONE_SPECIFIC_REFERENCE_HOT_SWAP_BLOCKER_REMAINS",
                "safety_gate": gate_results,
                "handoffs": handoffs,
            }
        )
    finally:
        if native is not None:
            try:
                result.setdefault("native_load_events", native._stage21_load_events)
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    with (output_dir / "gt_receding_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row)) + "\n")
    result = _jsonable(result)
    (output_dir / "gt_receding_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("camera_smoke", "gt_receding"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path)
    parser.add_argument("--dog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefix-steps", type=int, choices=(4, 6, 8, 10), default=4)
    parser.add_argument("--max-prefix-updates", type=int, default=5)
    args = parser.parse_args()
    if args.max_prefix_updates < 1:
        parser.error("--max-prefix-updates must be positive")
    trajectory = _load_pose_file(args.trajectory)
    if args.mode == "camera_smoke":
        result = _camera_smoke(args, trajectory)
    else:
        result = _gt_receding(args, trajectory)
    return 0 if result.get("status", "").endswith("PASS") or result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
