#!/usr/bin/env python3
"""Native IsaacGym WBC benchmark launcher.

This is an integration launcher, not a copy of the private lab source. The
source root, checkpoints, and trajectories are external inputs. The only
reference change is a subclass hook that loads one project-owned benchmark
trajectory into the delivered runtime's existing ``TrajectoryBatch``; policy
classes, history, normalization, action postprocessing, physics, and native
robot asset are unchanged.

The launcher intentionally has no P1M, GR00T, camera, writer, ROS, ZMQ, or
hardware path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np


def actor_model_classes():
    import torch.nn as nn

    class ArmActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_history_encoder = nn.Sequential(
                nn.Linear(12060 - 201, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
            )
            self.actor_body = nn.Sequential(
                nn.Linear(201 + 128, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 15),
            )

        def forward(self, history):
            obs = history[..., -201:]
            latent = self.actor_history_encoder(history[..., :-201])
            out = self.actor_body(torch.cat((obs, latent), dim=-1))
            out = out.clone()
            out[..., 6:] = torch.tanh(out[..., 6:])
            return out

    class DogActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_body = nn.Sequential(
                nn.Linear(2700, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 12),
            )

        def forward(self, history):
            return self.actor_body(history)

    import torch

    return ArmActor, DogActor


def safe_load_actor(role, path, model, torch):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("%s checkpoint payload is %s, expected dict" % (role, type(payload).__name__))
    prefixes = ("actor_body.", "actor_history_encoder.") if role == "arm" else ("actor_body.",)
    expected = set(model.state_dict())
    actual = {key for key in payload if any(key.startswith(prefix) for prefix in prefixes)}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    mismatch = [
        {
            "key": key,
            "model": list(model.state_dict()[key].shape),
            "checkpoint": list(payload[key].shape),
        }
        for key in sorted(expected & actual)
        if tuple(model.state_dict()[key].shape) != tuple(payload[key].shape)
    ]
    if missing or unexpected or mismatch:
        raise RuntimeError(
            json.dumps(
                {
                    "role": role,
                    "missing": missing,
                    "unexpected": unexpected,
                    "shape_mismatch": mismatch,
                }
            )
        )
    model.load_state_dict({key: payload[key] for key in expected}, strict=True)
    model.eval()
    return {
        "role": role,
        "safe_load": "PASS",
        "checkpoint": str(path),
        "payload_keys": len(payload),
        "inference_keys": len(actual),
    }


def quat_geodesic_xyzw(q0, q1):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=-1)), -1.0, 1.0))


def scalar_tensor(obj, name, default=np.nan):
    value = getattr(obj, name, None)
    if value is None:
        return float(default)
    try:
        return float(value[0].detach().cpu().item())
    except Exception:
        return float(default)


def array_tensor(obj, name, width=None):
    value = getattr(obj, name, None)
    if value is None:
        return np.empty((0, 0 if width is None else width), dtype=np.float32)
    arr = value.detach().cpu().numpy().copy()
    if arr.ndim == 1:
        arr = arr[None, :]
    if width is not None:
        arr = arr[:, :width]
    return arr


def configure_eval(cfg):
    cfg.env.arm_policy_enabled = True
    cfg.env.record_video = False
    cfg.env.num_recording_envs = 0
    cfg.env.num_envs = 1
    cfg.env.episode_length_s = 10000
    cfg.commands.resampling_time = 10000
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.asset.render_sphere = False
    cfg.domain_rand.push_robots = False
    for name in (
        "randomize_friction",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
        "randomize_com_displacement",
        "randomize_end_effector_force",
        "randomize_action_delay",
    ):
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)
    for group in (cfg.rewards, cfg.wbc.rewards):
        for name in ("use_terminal_body_height", "use_terminal_roll", "use_terminal_pitch"):
            if hasattr(group, name):
                setattr(group, name, False)


def _slerp_xyzw(q0, q1, fraction):
    """Small NumPy SLERP used only to construct the 40-point input mode."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + float(fraction) * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta = max(math.sin(theta), 1e-12)
    a = math.sin((1.0 - float(fraction)) * theta) / sin_theta
    b = math.sin(float(fraction) * theta) / sin_theta
    out = a * q0 + b * q1
    return out / max(float(np.linalg.norm(out)), 1e-12)


def _resample_to_40_points(timestamps, position, quaternion):
    """Make the exact timestamped 40-point interface input, without a model."""
    target = np.linspace(float(timestamps[0]), float(timestamps[-1]), 40)
    out_p = np.stack([np.interp(target, timestamps, position[:, i]) for i in range(3)], axis=-1)
    out_q = np.empty((len(target), 4), dtype=np.float64)
    for j, t in enumerate(target):
        i = int(np.searchsorted(timestamps, t, side="right") - 1)
        i = max(0, min(i, len(timestamps) - 2))
        dt = float(timestamps[i + 1] - timestamps[i])
        alpha = 0.0 if dt <= 0.0 else float((t - timestamps[i]) / dt)
        out_q[j] = _slerp_xyzw(quaternion[i], quaternion[i + 1], alpha)
    return target, out_p, out_q


def install_external_env(WBCEnv, trajectory_path, torch, reference_mode):
    """Return a WBCEnv subclass that replaces only trajectory loading."""

    class ExternalReferenceWBCEnv(WBCEnv):
        def __init__(self, *args, **kwargs):
            self._stage15_trajectory_path = Path(trajectory_path)
            self._stage15_load_count = 0
            self._stage16_pending_termination = None
            self._stage36_native_transport = None
            super().__init__(*args, **kwargs)

        def _stage16_joint_margin(self):
            limits = getattr(self, "dof_pos_limits", None)
            if limits is None:
                return np.empty(0, dtype=np.float32), False
            q = array_tensor(self, "dof_pos")[0]
            bounds = limits.detach().cpu().numpy()
            margin = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
            return margin.astype(np.float32), bool(np.any(margin < 0.0))

        def _stage16_snapshot(self):
            """Capture the pre-reset state; reset_idx otherwise overwrites it."""
            from go2arx5_vla.benchmark.lifecycle import classify_termination

            s_ref = self.traj_batch.s_ref(self.traj_sim_time)
            desired_pos = self.traj_batch.p_at(s_ref)
            desired_quat = self.traj_batch.quat_at(s_ref)
            s_k, preview_pos, preview_quat, preview_sdot = self.traj_batch.sample_preview(
                self.traj_s,
                float(self.cfg.wbc.goal_reaching.trajectory.preview_horizon),
                int(self.cfg.wbc.goal_reaching.trajectory.preview_points),
            )
            del s_k, preview_sdot
            ee = array_tensor(self, "end_effector_state")
            dof_pos = array_tensor(self, "dof_pos")[0]
            dof_vel = array_tensor(self, "dof_vel")[0]
            margin, joint_violation = self._stage16_joint_margin()
            joint_limit_termination = bool(
                getattr(self, "joint_limit_termination_buf", torch.zeros(1, dtype=torch.bool, device=self.device))[0]
                .detach()
                .cpu()
                .item()
            )
            state_arrays = (ee[0], array_tensor(self, "base_pos")[0], dof_pos, dof_vel)
            invalid = not all(np.isfinite(value).all() for value in state_arrays)
            body_height = bool(getattr(self, "body_height_buf", torch.zeros(1, dtype=torch.bool, device=self.device))[0].detach().cpu().item())
            event = {
                "reset_buf": bool(self.reset_buf[0].detach().cpu().item()),
                "time_out_buf": bool(self.time_out_buf[0].detach().cpu().item()),
                "body_height_buf": body_height,
                "fall_flag": body_height,
                "traj_early_term": bool(self.traj_early_term[0].detach().cpu().item()),
                "termination_cause": self.traj_term_cause[0].detach().cpu().tolist(),
                "invalid_state": bool(invalid),
                "joint_limit_violation": bool(joint_violation),
                "joint_limit_termination": joint_limit_termination,
                "sim_time": scalar_tensor(self, "traj_sim_time"),
                "traj_s": scalar_tensor(self, "traj_s"),
                "traj_s_ref": float(s_ref[0].detach().cpu().item()),
                "traj_d_lat": scalar_tensor(self, "traj_d_lat"),
                "traj_timing_err": scalar_tensor(self, "traj_timing_err"),
                "traj_L": scalar_tensor(self.traj_batch, "L"),
                "reference_duration": scalar_tensor(self.traj_batch, "T"),
                "desired_pos": desired_pos[0].detach().cpu().tolist(),
                "desired_quat": desired_quat[0].detach().cpu().tolist(),
                "preview_pos": preview_pos[0].detach().cpu().tolist(),
                "preview_quat": preview_quat[0].detach().cpu().tolist(),
                "achieved_pos": ee[0, :3].tolist(),
                "achieved_quat": ee[0, 3:7].tolist(),
                "eef_linear_velocity": ee[0, 7:10].tolist(),
                "eef_angular_velocity": ee[0, 10:13].tolist(),
                "base_pos": array_tensor(self, "base_pos")[0].tolist(),
                "base_quat": array_tensor(self, "base_quat")[0].tolist(),
                "base_lin_vel": array_tensor(self, "base_lin_vel")[0].tolist(),
                "base_feedforward": array_tensor(self, "base_feedforward_cmd")[0].tolist(),
                "delta_velocity": array_tensor(self, "delta_velocity_cmd")[0].tolist(),
                "commands_dog": array_tensor(self, "commands_dog")[0].tolist(),
                "dof_pos": dof_pos.tolist(),
                "dof_vel": dof_vel.tolist(),
                "joint_pos_target": array_tensor(self, "joint_pos_target")[0].tolist(),
                "arm_q": dof_pos[12:18].tolist(),
                "arm_dq": dof_vel[12:18].tolist(),
                "dog_q": dof_pos[:12].tolist(),
                "dog_dq": dof_vel[:12].tolist(),
                "joint_limit_margin": margin.tolist(),
            }
            event["termination_code"] = classify_termination(event)
            return event

        def check_termination(self):
            super().check_termination()
            if bool(self.reset_buf[0].detach().cpu().item()):
                self._stage16_pending_termination = self._stage16_snapshot()

        def pop_stage16_termination(self):
            event = self._stage16_pending_termination
            self._stage16_pending_termination = None
            return event

        def _reset_external_progress(self, env_ids):
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

        def _anchor_external_reference(self, env_ids):
            from isaacgym.torch_utils import quat_apply, quat_conjugate

            n = len(env_ids)
            sh_pos, sh_quat = self._shoulder_frame()
            sh_pos, sh_quat = sh_pos[env_ids], sh_quat[env_ids]
            direction_body = torch.nn.functional.normalize(self._traj_anchor_offset, dim=-1).expand(n, -1)
            direction_world = quat_apply(self.base_quat[env_ids], direction_body)
            if self.reach_table is None:
                r_max = torch.full(
                    (n,), float(self.cfg.wbc.goal_reaching.reach_radius), device=self.device
                )
            else:
                u_shoulder = quat_apply(quat_conjugate(sh_quat), direction_world)
                r_max = self.reach_table.query(u_shoulder).clamp_min(1e-3)
            distance = float(self.cfg.wbc.goal_reaching.rho_star) * r_max
            anchor = sh_pos + direction_world * distance.unsqueeze(-1)
            self.traj_batch.gamma_p[env_ids] += anchor.unsqueeze(1)

        def _load_trajectory_for(self, env_ids):
            if not self._traj_tracking_enabled():
                return super()._load_trajectory_for(env_ids)
            data = np.load(str(self._stage15_trajectory_path), allow_pickle=False)
            if "timestamp" not in data or "position" not in data or "quaternion" not in data:
                raise ValueError("benchmark trajectory must contain timestamp, position, quaternion")
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
            position = np.asarray(data["position"], dtype=np.float64)
            quaternion = np.asarray(data["quaternion"], dtype=np.float64)
            if timestamps.ndim != 1 or position.shape != (len(timestamps), 3) or quaternion.shape != (len(timestamps), 4):
                raise ValueError("benchmark trajectory shape is not [T], [T,3], [T,4]")
            if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
                raise ValueError("benchmark trajectory timestamps must be strictly increasing")
            from modules.trajectory import quat_to_mat, to_canonical

            # The native TrajectoryBatch time law starts at zero. Benchmark
            # files are required to use the same relative episode clock.
            if reference_mode == "BENCHMARK_40POINT_RECONSTRUCTED":
                timestamps, position, quaternion = _resample_to_40_points(
                    timestamps, position, quaternion
                )
            else:
                from go2arx5_vla.trajectory.native_transport import (
                    NATIVE_DT_S,
                    materialize_timestamped_pose_to_native,
                )

                transport = materialize_timestamped_pose_to_native(
                    timestamps, position, quaternion
                )
                timestamps = transport.timestamps
                position = transport.position
                quaternion = transport.quaternion
                self._stage36_native_transport = transport.audit()
            timestamps = timestamps - timestamps[0]
            legacy_dt = float((timestamps[-1] - timestamps[0]) / float(len(timestamps) - 1))
            dt = float(NATIVE_DT_S) if reference_mode == "BENCHMARK_CONTINUOUS" else legacy_dt
            gamma, time_law = to_canonical(
                torch.as_tensor(position, dtype=torch.float32),
                quat_to_mat(torch.as_tensor(quaternion, dtype=torch.float32)),
                dt,
            )
            count = len(env_ids)
            self.traj_batch.load(env_ids, [gamma] * count, [time_law] * count)
            self._anchor_external_reference(env_ids)
            self._reset_external_progress(env_ids)
            self._stage15_load_count += count

    return ExternalReferenceWBCEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument(
        "--reference-mode",
        choices=("NATIVE_GENERATOR", "BENCHMARK_CONTINUOUS", "BENCHMARK_40POINT_RECONSTRUCTED"),
        default="BENCHMARK_CONTINUOUS",
    )
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument(
        "--no-stop-at-reference-completion",
        action="store_true",
        help="diagnostic only: continue to the requested step count after the time-law endpoint",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="optional project-owned case metadata copied into the summary",
    )
    parser.add_argument(
        "--disable-trajectory-termination",
        action="store_true",
        help="diagnostic only: keep policy/physics unchanged while disabling trajectory early-stop gates",
    )
    args = parser.parse_args()
    if args.reference_mode != "NATIVE_GENERATOR" and args.trajectory is None:
        parser.error("--trajectory is required for an external reference mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))
    from go2arx5_vla.benchmark.lifecycle import reference_is_complete

    case_metadata = {}
    if args.metadata_json is not None:
        case_metadata = json.loads(args.metadata_json.read_text())

    # The delivered native config reserves 512 time-law points.  That is
    # sufficient for its default 8 s bank only because bank trajectories are
    # sampled more sparsely; a project-owned 100 Hz external reference can
    # legitimately contain more points.  Increase only the storage capacity
    # for this benchmark input.  This does not alter the controller, policy,
    # time law, termination thresholds, or physics semantics; without it the
    # delivered TrajectoryBatch.load() would silently truncate the reference.
    reference_capacity_points = None
    configured_reference_capacity_points = None
    reference_capacity_resized = False
    if args.reference_mode != "NATIVE_GENERATOR":
        with np.load(str(args.trajectory), allow_pickle=False) as reference_data:
            if "timestamp" not in reference_data:
                raise ValueError("benchmark trajectory must contain timestamp")
            reference_capacity_points = 40 if args.reference_mode == "BENCHMARK_40POINT_RECONSTRUCTED" else int(
                len(reference_data["timestamp"])
            )

    # IsaacGym must be imported before torch. The source package is private,
    # external, and supplied through --source-root.
    import isaacgym  # noqa: F401
    import torch
    from isaacgym.torch_utils import quat_apply, quat_conjugate

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from go1_gym.envs.config import RoboDuetRuntimeOptions, build_roboduet_config
    from go1_gym.envs.roboduet.wbc_env import WBCEnv
    from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
    from go1_gym.utils.global_switch import global_switch

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
    configure_eval(cfg)
    configured_reference_capacity_points = int(cfg.wbc.goal_reaching.trajectory.max_tl_points)
    if reference_capacity_points is not None and reference_capacity_points > configured_reference_capacity_points:
        cfg.wbc.goal_reaching.trajectory.max_tl_points = int(reference_capacity_points)
        reference_capacity_resized = True
    if args.disable_trajectory_termination:
        cfg.wbc.goal_reaching.trajectory.terminate_d_lat = 0.0
        cfg.wbc.goal_reaching.trajectory.terminate_timing_frac = 0.0
    ArmActor, DogActor = actor_model_classes()
    arm = ArmActor()
    dog = DogActor()
    arm_load = safe_load_actor("arm", args.arm, arm, torch)
    dog_load = safe_load_actor("dog", args.dog, dog, torch)
    device = torch.device(args.sim_device)
    arm = arm.to(device)
    dog = dog.to(device)

    RuntimeWBCEnv = (
        WBCEnv
        if args.reference_mode == "NATIVE_GENERATOR"
        else install_external_env(WBCEnv, args.trajectory, torch, args.reference_mode)
    )
    native = None
    env = None
    rows = []
    status = "UNKNOWN"
    termination = "NATURAL_COMPLETION"
    error = None
    episode_ended = False
    reference_completed = False

    def on_term(signum, frame):
        del frame
        raise TimeoutError("received signal %s" % signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        global_switch.open_switch()
        native = RuntimeWBCEnv(
            sim_device=args.sim_device,
            headless=True,
            num_envs=1,
            cfg=cfg,
            graphics_device_id=0,
        )
        env = HistoryWrapper(native)
        env.reset()
        max_steps = max(1, int(round(args.seconds * 50.0)))
        terminated_early = False
        with torch.inference_mode():
            for step in range(max_steps):
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
                rew_dog, rew_arm, done, info = env.step(
                    actions_dog, actions_arm[..., : native.num_actions_arm]
                )
                del rew_dog, rew_arm, info
                done_bool = bool(done.detach().cpu().any().item())
                s_ref = native.traj_batch.s_ref(native.traj_sim_time)
                desired_pos = native.traj_batch.p_at(s_ref)
                desired_quat = native.traj_batch.quat_at(s_ref)
                s_k, preview_pos, preview_quat, preview_sdot = native.traj_batch.sample_preview(
                    native.traj_s,
                    float(cfg.wbc.goal_reaching.trajectory.preview_horizon),
                    int(cfg.wbc.goal_reaching.trajectory.preview_points),
                )
                del s_k, preview_sdot
                ee = array_tensor(native, "end_effector_state")
                ee_pos = ee[:, :3]
                ee_quat = ee[:, 3:7]
                dof_pos_np = array_tensor(native, "dof_pos")[0]
                dof_vel_np = array_tensor(native, "dof_vel")[0]
                limits = getattr(native, "dof_pos_limits", None)
                if limits is None:
                    joint_margin_np = np.empty(0, dtype=np.float32)
                    joint_violation = False
                else:
                    bounds_np = limits.detach().cpu().numpy()
                    joint_margin_np = np.minimum(
                        dof_pos_np - bounds_np[:, 0], bounds_np[:, 1] - dof_pos_np
                    ).astype(np.float32)
                    joint_violation = bool(np.any(joint_margin_np < 0.0))
                invalid_state = not all(
                    np.isfinite(value).all()
                    for value in (ee[0], dof_pos_np, dof_vel_np)
                )
                row = {
                    "step": step,
                    "sim_time": scalar_tensor(native, "traj_sim_time"),
                    "done": done_bool,
                    "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
                    "time_out_buf": bool(native.time_out_buf[0].detach().cpu().item()),
                    "traj_early_term": bool(native.traj_early_term[0].detach().cpu().item()),
                    "termination_cause": native.traj_term_cause[0].detach().cpu().tolist(),
                    "traj_s": scalar_tensor(native, "traj_s"),
                    "traj_s_ref": float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item()),
                    "traj_d_lat": scalar_tensor(native, "traj_d_lat"),
                    "traj_timing_err": scalar_tensor(native, "traj_timing_err"),
                    "traj_L": scalar_tensor(native.traj_batch, "L"),
                    "reference_duration": scalar_tensor(native.traj_batch, "T"),
                    "desired_pos": desired_pos[0].detach().cpu().tolist(),
                    "desired_quat": desired_quat[0].detach().cpu().tolist(),
                    "preview_pos": preview_pos[0].detach().cpu().tolist(),
                    "preview_quat": preview_quat[0].detach().cpu().tolist(),
                    "achieved_pos": ee_pos[0].tolist(),
                    "achieved_quat": ee_quat[0].tolist(),
                    "eef_linear_velocity": ee[0, 7:10].tolist(),
                    "eef_angular_velocity": ee[0, 10:13].tolist(),
                    "base_pos": array_tensor(native, "base_pos")[0].tolist(),
                    "base_quat": array_tensor(native, "base_quat")[0].tolist(),
                    "base_lin_vel": array_tensor(native, "base_lin_vel")[0].tolist(),
                    "base_feedforward": array_tensor(native, "base_feedforward_cmd")[0].tolist(),
                    "delta_velocity": array_tensor(native, "delta_velocity_cmd")[0].tolist(),
                    "commands_dog": array_tensor(native, "commands_dog")[0].tolist(),
                    "arm_raw_output": actions_arm[0].detach().cpu().tolist(),
                    "arm_action_6d": actions_arm[0, : native.num_actions_arm].detach().cpu().tolist(),
                    "internal_body_plan_9d": actions_arm[0, native.num_actions_arm :].detach().cpu().tolist(),
                    "dog_action_12d": actions_dog[0].detach().cpu().tolist(),
                    "dof_pos": dof_pos_np.tolist(),
                    "dof_vel": dof_vel_np.tolist(),
                    "joint_pos_target": array_tensor(native, "joint_pos_target")[0].tolist(),
                    "arm_q": array_tensor(native, "dof_pos")[0, 12:18].tolist(),
                    "arm_dq": array_tensor(native, "dof_vel")[0, 12:18].tolist(),
                    "dog_q": array_tensor(native, "dof_pos")[0, :12].tolist(),
                    "dog_dq": array_tensor(native, "dof_vel")[0, :12].tolist(),
                    "joint_limit_margin": joint_margin_np.tolist(),
                    "invalid_state": bool(invalid_state),
                    "joint_limit_violation": bool(joint_violation),
                    "joint_limit_termination": False,
                    "fall_flag": False,
                    "termination_code": "NONE",
                }
                event = native.pop_stage16_termination()
                if event is not None:
                    # env.step() calls reset_idx before returning.  Replace
                    # the post-reset observation with the captured pre-reset
                    # state so the final tracking sample is not a reset sample.
                    row.update(event)
                    row["step"] = step
                    row["done"] = done_bool
                row["episode_done"] = False
                row["reference_complete"] = False
                row["tracking_valid"] = True
                if event is not None:
                    row["episode_done"] = True
                    episode_ended = True
                    termination = row["termination_code"]
                elif args.reference_mode != "NATIVE_GENERATOR":
                    reference_completed = reference_is_complete(
                        sim_time=float(row["sim_time"]),
                        reference_duration=float(row["reference_duration"]),
                        s_ref=float(row["traj_s_ref"]),
                        reference_length=float(row["traj_L"]),
                        dt=float(native.dt),
                    )
                    if reference_completed and not args.no_stop_at_reference_completion:
                        row["reference_complete"] = True
                        row["episode_done"] = True
                        row["termination_code"] = "REFERENCE_COMPLETE"
                        episode_ended = True
                        termination = "REFERENCE_COMPLETE"
                # Derived timing telemetry is written after the pre-reset
                # event is merged. It is observational only and never feeds
                # back into the delivered termination/controller semantics.
                traj_length = float(row.get("traj_L", np.nan))
                timing_error = float(row.get("traj_timing_err", np.nan))
                row["traj_s_ref_minus_traj_s"] = float(row["traj_s_ref"] - row["traj_s"])
                row["traj_timing_err_abs"] = float(abs(timing_error))
                row["traj_timing_error_fraction"] = (
                    float(abs(timing_error) / traj_length)
                    if np.isfinite(traj_length) and traj_length > 0.0
                    else float("nan")
                )
                row["timing_margin_m"] = (
                    float(0.70 * traj_length - abs(timing_error))
                    if np.isfinite(traj_length)
                    else float("nan")
                )
                row["timing_margin_fraction"] = (
                    float(0.70 - abs(timing_error) / traj_length)
                    if np.isfinite(traj_length) and traj_length > 0.0
                    else float("nan")
                )
                if not row.get("reference_complete", False):
                    reference_completed = reference_completed or bool(row.get("reference_complete", False))
                rows.append(row)
                if step % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "sim_time": row["sim_time"],
                                "traj_s": row["traj_s"],
                                "traj_L": row["traj_L"],
                                "done": done_bool,
                            }
                        ),
                        flush=True,
                    )
                if done_bool:
                    if event is not None:
                        termination = row["termination_code"]
                    terminated_early = True
                    break
                if episode_ended:
                    break
        status = (
            "NATIVE_%s_PASS" % args.reference_mode
            if (episode_ended and termination == "REFERENCE_COMPLETE") or not terminated_early
            else "NATIVE_%s_PARTIAL" % args.reference_mode
        )
    except Exception as exc:
        status = (
            "NATIVE_%s_PARTIAL" % args.reference_mode
            if rows
            else "NATIVE_%s_BLOCKED" % args.reference_mode
        )
        error = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps({"status": status, "error": error}), file=sys.stderr, flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                termination = "TEARDOWN_ERROR:%s:%s" % (type(exc).__name__, exc)

    def stack(name, width=None):
        values = [row[name] for row in rows]
        if not values:
            return np.empty((0, 0 if width is None else width), dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        return arr.reshape(len(values), width) if width is not None else arr

    output_npz = args.output_dir / "native_external_rollout.npz"
    termination_code_index = {
        "NONE": 0,
        "REFERENCE_COMPLETE": 1,
        "TIME_LIMIT": 2,
        "FALL": 3,
        "INVALID_STATE": 4,
        "JOINT_LIMIT": 5,
        "TRACKING_ABORT": 6,
        "NATIVE_TASK_TERMINATION": 7,
        "UNKNOWN": 8,
    }
    np.savez_compressed(
        output_npz,
        step=np.asarray([row["step"] for row in rows], dtype=np.int32),
        sim_time=np.asarray([row["sim_time"] for row in rows], dtype=np.float32),
        desired_pos=stack("desired_pos", 3),
        desired_quat=stack("desired_quat", 4),
        preview_pos=stack("preview_pos", 9 * 3),
        preview_quat=stack("preview_quat", 9 * 4),
        eef_linear_velocity=stack("eef_linear_velocity", 3),
        eef_angular_velocity=stack("eef_angular_velocity", 3),
        traj_s=stack("traj_s"),
        traj_s_ref=stack("traj_s_ref"),
        traj_d_lat=stack("traj_d_lat"),
        traj_timing_err=stack("traj_timing_err"),
        traj_s_ref_minus_traj_s=stack("traj_s_ref_minus_traj_s"),
        traj_timing_err_abs=stack("traj_timing_err_abs"),
        traj_timing_error_fraction=stack("traj_timing_error_fraction"),
        timing_margin_m=stack("timing_margin_m"),
        timing_margin_fraction=stack("timing_margin_fraction"),
        traj_L=stack("traj_L"),
        achieved_pos=stack("achieved_pos", 3),
        achieved_quat=stack("achieved_quat", 4),
        base_pos=stack("base_pos", 3),
        base_quat=stack("base_quat", 4),
        base_lin_vel=stack("base_lin_vel", 3),
        base_feedforward=stack("base_feedforward", 3),
        delta_velocity=stack("delta_velocity", 3),
        commands_dog=stack("commands_dog"),
        arm_raw_output=stack("arm_raw_output", 15),
        arm_action_6d=stack("arm_action_6d", 6),
        internal_body_plan_9d=stack("internal_body_plan_9d", 9),
        dog_action_12d=stack("dog_action_12d", 12),
        dof_pos=stack("dof_pos"),
        dof_vel=stack("dof_vel"),
        joint_pos_target=stack("joint_pos_target"),
        arm_q=stack("arm_q", 6),
        arm_dq=stack("arm_dq", 6),
        dog_q=stack("dog_q", 12),
        dog_dq=stack("dog_dq", 12),
        joint_limit_margin=stack("joint_limit_margin"),
        episode_done=np.asarray([row.get("episode_done", False) for row in rows], dtype=bool),
        reference_complete=np.asarray([row.get("reference_complete", False) for row in rows], dtype=bool),
        tracking_valid=np.asarray([row.get("tracking_valid", True) for row in rows], dtype=bool),
        reset_buf=np.asarray([row.get("reset_buf", False) for row in rows], dtype=bool),
        time_out_buf=np.asarray([row.get("time_out_buf", False) for row in rows], dtype=bool),
        fall_flag=np.asarray([row.get("fall_flag", row.get("body_height_buf", False)) for row in rows], dtype=bool),
        invalid_state=np.asarray([row.get("invalid_state", False) for row in rows], dtype=bool),
        joint_limit_violation=np.asarray([row.get("joint_limit_violation", False) for row in rows], dtype=bool),
        termination_code_index=np.asarray(
            [termination_code_index.get(row.get("termination_code", "UNKNOWN"), 8) for row in rows],
            dtype=np.int8,
        ),
        reference_duration=np.asarray([row.get("reference_duration", np.nan) for row in rows], dtype=np.float32),
    )
    if rows:
        desired = stack("desired_pos", 3)
        achieved = stack("achieved_pos", 3)
        dq = np.linalg.norm(desired - achieved, axis=-1)
        rot = quat_geodesic_xyzw(stack("desired_quat", 4), stack("achieved_quat", 4))
        base = stack("base_pos", 3)
        base_q = stack("base_quat", 4)
        metrics = {
            "position_m": {
                "mean": float(np.mean(dq)),
                "rmse": float(np.sqrt(np.mean(dq * dq))),
                "p95": float(np.percentile(dq, 95)),
                "max": float(np.max(dq)),
            },
            "rotation_rad": {
                "mean": float(np.mean(rot)),
                "rmse": float(np.sqrt(np.mean(rot * rot))),
                "p95": float(np.percentile(rot, 95)),
                "max": float(np.max(rot)),
            },
            "time_in_tolerance_fraction_0.05m_0.20rad": float(np.mean((dq <= 0.05) & (rot <= 0.20))),
            "path_completion": float(stack("traj_s", None)[-1] / max(stack("traj_L", None)[-1], 1e-9))
            if "traj_s" in rows[0] and rows[0].get("traj_L") is not None
            else None,
            "base_displacement_m": float(np.linalg.norm(base[-1] - base[0])),
            "base_path_length_m": float(np.linalg.norm(np.diff(base, axis=0), axis=-1).sum()) if len(base) > 1 else 0.0,
            "base_yaw_change_rad_quaternion": float(quat_geodesic_xyzw(base_q[0], base_q[-1])),
            "steps": len(rows),
        }
    else:
        metrics = {}
    summary = {
        "status": status,
        "termination": termination,
        "steps": len(rows),
        "requested_steps": int(round(args.seconds * 50.0)),
        "reference_mode": args.reference_mode,
        "trajectory": None if args.trajectory is None else str(args.trajectory),
        "output_npz": str(output_npz),
        "safe_load": [arm_load, dog_load],
        "metrics": metrics,
        "error": error,
        "episode_ended": bool(episode_ended),
        "reference_completed": bool(reference_completed or any(row.get("reference_complete", False) for row in rows)),
        "termination_code": termination,
        "termination_codes_by_step": {
            code: sum(1 for row in rows if row.get("termination_code") == code)
            for code in sorted({row.get("termination_code", "UNKNOWN") for row in rows})
        },
        "reference_capacity_points": reference_capacity_points,
        "configured_reference_capacity_points": configured_reference_capacity_points,
        "reference_capacity_resized": reference_capacity_resized,
        "native_transport": None if native is None else getattr(native, "_stage36_native_transport", None),
        "case_metadata": case_metadata,
    }
    (args.output_dir / "native_external_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "native_external_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "NATIVE_%s_PASS" % args.reference_mode else 2


if __name__ == "__main__":
    raise SystemExit(main())
