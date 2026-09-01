#!/usr/bin/env python3
"""Compare the source-backed native ARX5 FK model with IsaacGym state.

This is a geometry-only audit.  It creates one native simulator environment,
never loads a checkpoint, never calls a policy, and never changes the
delivered archive or any canonical target asset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _quat_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose(position, quaternion):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quat_to_matrix(quaternion)
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def _geodesic(r0, r1):
    relative = np.asarray(r0)[:3, :3].T @ np.asarray(r1)[:3, :3]
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))


def _world_pose(base_pos, base_quat, local_pose):
    return _pose(base_pos, base_quat) @ local_pose


def _set_arm_state(native, indices, q):
    native.dof_pos[0, indices] = native.dof_pos.new_tensor(q)
    native.dof_vel[0, indices] = 0.0
    from isaacgym import gymtorch

    native.gym.set_dof_state_tensor(native.sim, gymtorch.unwrap_tensor(native.dof_state))
    native.gym.refresh_rigid_body_state_tensor(native.sim)


def _advance_one_physics_step(native):
    """Commit a reset state so GPU rigid-body tensors reflect its kinematics."""
    native.gym.simulate(native.sim)
    native.gym.fetch_results(native.sim, True)
    native.gym.refresh_dof_state_tensor(native.sim)
    native.gym.refresh_actor_root_state_tensor(native.sim)
    native.gym.refresh_rigid_body_state_tensor(native.sim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=15015)
    parser.add_argument("--sim-device", default="cuda:0")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))

    # IsaacGym must be imported before torch in the supplied runtime.
    import isaacgym  # noqa: F401
    import torch

    from go1_gym.envs.config import RoboDuetRuntimeOptions, build_roboduet_config
    from go1_gym.envs.roboduet.wbc_env import WBCEnv
    from go1_gym.utils.global_switch import global_switch
    from native_wbc_benchmark import configure_eval

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    kin = NativeArmKinematics.from_urdf(args.urdf)
    cfg = build_roboduet_config(
        options=RoboDuetRuntimeOptions(
            num_envs=1,
            robot="go2_x5",
            use_rot6d=True,
            dyna_gait=True,
            goal_reaching=True,
            traj_tracking=True,
            arm_action_mode="end_to_end",
            reach_table=True,
        ),
        debug=True,
    )
    configure_eval(cfg)
    native = None
    result = {
        "status": "BLOCKED",
        "samples_requested": int(args.samples),
        "seed": int(args.seed),
        "urdf": str(args.urdf),
        "joint_names": list(kin.joint_names),
    }
    try:
        global_switch.open_switch()
        native = WBCEnv(
            sim_device=args.sim_device,
            headless=True,
            num_envs=1,
            cfg=cfg,
            graphics_device_id=0,
        )
        native.reset()
        names = list(native.dof_names)
        indices = [names.index(name) for name in kin.joint_names]
        body_names = list(native.body_names)
        link6_index = body_names.index("x5_link6")
        rng = np.random.default_rng(args.seed)
        limits = kin.joint_limits
        margin = np.minimum(0.05, 0.2 * (limits[:, 1] - limits[:, 0]))
        low, high = limits[:, 0] + margin, limits[:, 1] - margin
        pos_errors = []
        tcp_pos_errors = []
        rot_errors = []
        tcp_rot_errors = []
        samples = []
        for sample_index in range(args.samples):
            # The native reset hook supplies a source-backed, bounded random
            # arm state and refreshes the articulation before observation.
            # Repeated resets avoid relying on an undocumented direct-state
            # refresh behavior in the GPU pipeline.
            native.reset()
            _advance_one_physics_step(native)
            q = _array(native.dof_pos[0, indices])
            base_pos = _array(native.base_pos[0])
            base_quat = _array(native.base_quat[0])
            base_tf = _pose(base_pos, base_quat)
            rigid_view = native.rigid_body_state.view(native.num_envs, native.num_bodies, 13)
            rigid = _array(rigid_view[0, link6_index])
            sim_link6 = _pose(rigid[:3], rigid[3:7])
            sim_tcp = sim_link6.copy()
            sim_tcp[:3, 3] = sim_link6[:3, 3] + sim_link6[:3, :3] @ kin.tcp_offset_xyz
            expected_link6 = base_tf @ kin.fk(q)["__link6__"]
            expected_tcp = base_tf @ kin.fk(q)["__tcp__"]
            pos_error = float(np.linalg.norm(sim_link6[:3, 3] - expected_link6[:3, 3]))
            tcp_error = float(np.linalg.norm(sim_tcp[:3, 3] - expected_tcp[:3, 3]))
            rot_error = _geodesic(sim_link6, expected_link6)
            tcp_rot_error = _geodesic(sim_tcp, expected_tcp)
            pos_errors.append(pos_error)
            tcp_pos_errors.append(tcp_error)
            rot_errors.append(rot_error)
            tcp_rot_errors.append(tcp_rot_error)
            if sample_index < 3:
                samples.append({
                    "q": q.tolist(),
                    "sim_link6_position": sim_link6[:3, 3].tolist(),
                    "expected_link6_position": expected_link6[:3, 3].tolist(),
                    "sim_tcp_position": sim_tcp[:3, 3].tolist(),
                    "expected_tcp_position": expected_tcp[:3, 3].tolist(),
                })

        # Joint-wise differential directions at the center of the safe box.
        native.reset()
        _advance_one_physics_step(native)
        q0 = _array(native.dof_pos[0, indices])
        base_pos = _array(native.base_pos[0])
        base_quat = _array(native.base_quat[0])
        base_tf = _pose(base_pos, base_quat)
        native.gym.refresh_jacobian_tensors(native.sim)
        native_jacobian, _ = native._arm_jacobian()
        native_jacobian = _array(native_jacobian[0])
        rigid_view = native.rigid_body_state.view(native.num_envs, native.num_bodies, 13)
        current_link6 = _pose(
            _array(rigid_view[0, link6_index])[:3],
            _array(rigid_view[0, link6_index])[3:7],
        )
        tcp_offset_world = current_link6[:3, :3] @ kin.tcp_offset_xyz
        delta = 1e-6
        differential = []
        for joint_i, _index in enumerate(indices):
            qp, qm = q0.copy(), q0.copy()
            qp[joint_i] += delta
            qm[joint_i] -= delta
            model_plus = kin.fk(qp)["__tcp__"]
            model_minus = kin.fk(qm)["__tcp__"]
            model_p = (model_plus[:3, 3] - model_minus[:3, 3]) / (2 * delta)
            model_dp = base_tf[:3, :3] @ model_p
            relative = model_plus[:3, :3] @ model_minus[:3, :3].T
            skew = (relative - relative.T) / 2.0
            model_dr = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]]) / (2 * delta)
            native_dp = native_jacobian[:3, joint_i] + np.cross(
                native_jacobian[3:, joint_i], tcp_offset_world
            )
            native_dr = native_jacobian[3:, joint_i]
            dot = float(np.dot(native_dp, model_dp))
            differential.append({
                "joint": kin.joint_names[joint_i],
                "native_tcp_translation_derivative": native_dp.tolist(),
                "model_translation_derivative": model_dp.tolist(),
                "direction_cosine": dot / max(float(np.linalg.norm(native_dp) * np.linalg.norm(model_dp)), 1e-12),
                "norm_ratio": float(np.linalg.norm(native_dp) / max(float(np.linalg.norm(model_dp)), 1e-12)),
                "native_rotation_derivative": native_dr.tolist(),
                "model_rotation_derivative": (base_tf[:3, :3] @ model_dr).tolist(),
            })
        result.update({
            "status": "PASS",
            "native_dof_indices": indices,
            "native_link6_body_index": link6_index,
            "native_body_names": body_names,
            "native_base_position": base_pos.tolist(),
            "native_base_quaternion_xyzw": base_quat.tolist(),
            "position_error_m": {
                "max": float(np.max(pos_errors)),
                "p95": float(np.percentile(pos_errors, 95)),
                "mean": float(np.mean(pos_errors)),
            },
            "tcp_position_error_m": {
                "max": float(np.max(tcp_pos_errors)),
                "p95": float(np.percentile(tcp_pos_errors, 95)),
                "mean": float(np.mean(tcp_pos_errors)),
            },
            "link6_rotation_error_rad": {
                "max": float(np.max(rot_errors)),
                "p95": float(np.percentile(rot_errors, 95)),
                "mean": float(np.mean(rot_errors)),
            },
            "tcp_rotation_error_rad": {
                "max": float(np.max(tcp_rot_errors)),
                "p95": float(np.percentile(tcp_rot_errors, 95)),
                "mean": float(np.mean(tcp_rot_errors)),
            },
            "differential_tests": differential,
            "samples": samples,
            "note": "geometry-only; no policy/checkpoint execution",
        })
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(result, indent=2), file=sys.stderr, flush=True)
    finally:
        if native is not None:
            try:
                native.close()
            except Exception as exc:
                result.setdefault("teardown_error", {"type": type(exc).__name__, "message": str(exc)})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
