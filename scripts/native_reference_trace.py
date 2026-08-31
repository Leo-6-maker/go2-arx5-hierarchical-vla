#!/usr/bin/env python3
"""Record the delivered native trajectory binding without loading a policy.

This diagnostic deliberately constructs the lab environment only to exercise
its reset and reference-loader code.  It never loads a checkpoint and never
calls a policy.  The output is a compact per-reset trace used by Stage 15's
independent reference-binding audit.

The script is expected to run on a compute node with the delivered IsaacGym
runtime on ``PYTHONPATH``.  The source package and its runtime are explicit
inputs; this repository does not vendor the lab package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _scalar(value):
    arr = _array(value)
    return float(np.asarray(arr).reshape(-1)[0])


def _valid_prefix(values, *, atol=1e-7):
    """Find the prefix before TrajectoryBatch's repeated tail padding."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2:
        return int(values.size)
    changes = np.flatnonzero(np.abs(np.diff(values)) > atol)
    return int(changes[-1] + 2) if changes.size else 2


def _pose(position, quaternion):
    from go2arx5_vla.trajectory.relative_se3 import pose_from_position_quaternion

    return pose_from_position_quaternion(position, quaternion)


def _record(native, reset_index):
    from isaacgym.torch_utils import quat_apply, quat_conjugate
    import torch

    tb = native.traj_batch
    gamma_s_full = _array(tb.gamma_s[0])
    n_gamma = _valid_prefix(gamma_s_full)
    gamma_s = gamma_s_full[:n_gamma]
    gamma_p = _array(tb.gamma_p[0, :n_gamma])
    gamma_q = _array(tb.gamma_quat[0, :n_gamma])
    gamma_tangent = _array(tb.gamma_tangent[0, :n_gamma])

    tl_t_full = _array(tb.tl_t[0])
    n_tl = _valid_prefix(tl_t_full)
    tl_t = tl_t_full[:n_tl]
    tl_s = _array(tb.tl_s[0, :n_tl])
    tl_sdot = _array(tb.tl_sdot[0, :n_tl])

    base_pos = _array(native.base_pos[0])
    base_quat = _array(native.base_quat[0])
    shoulder_pos_t, shoulder_quat_t = native._shoulder_frame()
    shoulder_pos = _array(shoulder_pos_t[0])
    shoulder_quat = _array(shoulder_quat_t[0])
    mount_tf = _array(native.arm_mount_tfs[0])
    mount_quat = _array(native.arm_mount_quat_body[0])
    anchor_offset = _array(native._traj_anchor_offset[0])
    direction_body = anchor_offset / max(float(np.linalg.norm(anchor_offset)), 1e-12)
    direction_world = _array(
        quat_apply(
            native.base_quat[0:1],
            native._traj_anchor_offset / native._traj_anchor_offset.norm(dim=-1, keepdim=True),
        )[0]
    )
    if native.reach_table is None:
        reach_max = float(native.cfg.wbc.goal_reaching.reach_radius)
    else:
        u_shoulder = quat_apply(
            quat_conjugate(shoulder_quat_t[0:1]),
            torch.as_tensor(direction_world, dtype=torch.float32, device=native.device)[None],
        )
        reach_max = _scalar(native.reach_table.query(u_shoulder))
    rho_star = float(native.cfg.wbc.goal_reaching.rho_star)
    anchor_position = shoulder_pos + direction_world * (rho_star * reach_max)

    s_current = _scalar(native.traj_s)
    preview_s, preview_p, preview_q, preview_sdot = native.traj_batch.sample_preview(
        native.traj_s,
        float(native.cfg.wbc.goal_reaching.trajectory.preview_horizon),
        int(native.cfg.wbc.goal_reaching.trajectory.preview_points),
    )
    arm_terms = _array(native._arm_traj_obs_terms()[0])
    arm_obs, arm_priv = native.get_arm_observations()
    eef = _array(native.end_effector_state[0])
    arm_q = _array(native.dof_pos[0, native.num_actions_loco:native.num_actions_loco + native.num_actions_arm])
    rigid = native.rigid_body_state.view(native.num_envs, native.num_bodies, 13)
    link6_state = _array(rigid[0, native.ee_idx])
    root = _array(native.root_states[0])

    return {
        "reset_index": int(reset_index),
        "root_state": root,
        "base_position": base_pos,
        "base_quaternion_xyzw": base_quat,
        "mount_tf_xyzrpy": mount_tf,
        "mount_quaternion_xyzw": mount_quat,
        "shoulder_position": shoulder_pos,
        "shoulder_quaternion_xyzw": shoulder_quat,
        "anchor_offset_body": anchor_offset,
        "direction_body": direction_body,
        "direction_world": direction_world,
        "reach_max_m": np.asarray(reach_max, dtype=np.float64),
        "rho_star": np.asarray(rho_star, dtype=np.float64),
        "anchor_position": anchor_position,
        "anchor_rotation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "ee_local_offset_link6": _array(native.ee_local_offset[0]),
        "arm_q": arm_q,
        "link6_state": link6_state,
        "ee_idx": np.asarray(int(native.ee_idx), dtype=np.int64),
        "gamma_s": gamma_s,
        "gamma_p_bound_world": gamma_p,
        "gamma_q_bound_world_xyzw": gamma_q,
        "gamma_tangent_bound_world": gamma_tangent,
        "time_law_t": tl_t,
        "time_law_s": tl_s,
        "time_law_sdot": tl_sdot,
        "trajectory_length_m": np.asarray(_scalar(tb.L)),
        "trajectory_duration_s": np.asarray(_scalar(tb.T)),
        "traj_s_current_m": np.asarray(s_current),
        "traj_sim_time_s": np.asarray(_scalar(native.traj_sim_time)),
        "preview_s_m": _array(preview_s[0]),
        "preview_position_bound_world": _array(preview_p[0]),
        "preview_quaternion_bound_world_xyzw": _array(preview_q[0]),
        "preview_sdot_mps": _array(preview_sdot[0]),
        "arm_traj_terms_134": arm_terms,
        "arm_observation_frame_201": _array(arm_obs[0]),
        "arm_privileged_observation": _array(arm_priv[0]) if arm_priv is not None else np.empty(0),
        "arm_goal_position_world": _array(native.arm_goal_pos_world[0]),
        "arm_goal_quaternion_world_xyzw": _array(native.arm_goal_quat_world[0]),
        "eef_state_tcp_position_world": eef[:3],
        "eef_state_link6_quaternion_xyzw": eef[3:7],
        "heading_quaternion_xyzw": _array(native._heading_quat()[0]),
        "traj_term_cause": _array(native.traj_term_cause[0]),
        "traj_early_term": np.asarray(bool(native.traj_early_term[0].item())),
        "body_names": np.asarray(list(native.body_names)),
        "dof_names": np.asarray(list(native.dof_names)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--sim-device", default="cuda:0")
    args = parser.parse_args()
    if args.resets < 1:
        parser.error("--resets must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))

    # IsaacGym must be imported before torch in the delivered runtime.
    import isaacgym  # noqa: F401
    import torch

    from go1_gym.envs.config import RoboDuetRuntimeOptions, build_roboduet_config
    from go1_gym.envs.roboduet.wbc_env import WBCEnv
    from go1_gym.utils.global_switch import global_switch
    from native_wbc_benchmark import configure_eval

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    native = None
    traces = []
    error = None
    try:
        global_switch.open_switch()
        native = WBCEnv(
            sim_device=args.sim_device,
            headless=True,
            num_envs=1,
            cfg=cfg,
            graphics_device_id=0,
        )
        for reset_index in range(args.resets):
            native.reset()
            trace = _record(native, reset_index)
            trace_path = args.output / ("reset_%03d.npz" % reset_index)
            np.savez_compressed(trace_path, **trace)
            traces.append(
                {
                    "reset_index": reset_index,
                    "path": str(trace_path),
                    "gamma_points": int(trace["gamma_s"].size),
                    "time_law_points": int(trace["time_law_t"].size),
                    "trajectory_length_m": float(trace["trajectory_length_m"]),
                    "trajectory_duration_s": float(trace["trajectory_duration_s"]),
                    "anchor_position": trace["anchor_position"].tolist(),
                    "base_position": trace["base_position"].tolist(),
                    "arm_observation_shape": list(trace["arm_observation_frame_201"].shape),
                    "arm_traj_terms_shape": list(trace["arm_traj_terms_134"].shape),
                }
            )
    except Exception as exc:  # diagnostic report must preserve exact blocker
        error = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps({"error": error}), file=sys.stderr, flush=True)
    finally:
        if native is not None:
            try:
                native.close()
            except Exception as exc:
                error = error or {"type": type(exc).__name__, "message": str(exc)}

    summary = {
        "status": "PASS" if error is None and len(traces) == args.resets else "PARTIAL_OR_BLOCKED",
        "requested_resets": args.resets,
        "completed_resets": len(traces),
        "seed": args.seed,
        "sim_device": args.sim_device,
        "source_root": str(args.source_root),
        "error": error,
        "traces": traces,
    }
    (args.output / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
