#!/usr/bin/env python3
"""Audit Stage 15's independent native-reference reconstruction.

The input traces are produced by ``native_reference_trace.py``.  This script
does not import IsaacGym or the private lab package.  It reconstructs the
translation-only native anchor, the Gamma-grid preview, and the dynamic
heading-frame policy reference from the recorded native state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.contracts import EpisodeReferenceBindingV1, PlacementProfileV1, TrajectoryShapeV1
from go2arx5_vla.trajectory.preview import arc_length_preview_from_grid
from go2arx5_vla.trajectory.relative_se3 import pose_from_position_quaternion
from go2arx5_vla.trajectory.so3 import (
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_rot6d_rows,
    rot6d_to_matrix,
)
from go2arx5_vla.wbc.native.reference_binding import NativeAnchorInputs, native_translation_anchor


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack((aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :3] *= -1.0
    return out


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    pure = np.concatenate((v, np.zeros(v.shape[:-1] + (1,))), axis=-1)
    return quat_mul(quat_mul(q, pure), quat_conj(q))[..., :3]


def quat_geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0))


def _table_query(table: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Pure NumPy copy of ReachabilityTable.query for the 2-D table."""
    u = np.asarray(u, dtype=np.float64)
    az = np.arctan2(u[..., 1], u[..., 0])
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    n_az, n_el = table.shape

    def indices(value, lo, span, n, wrap):
        x = (value - lo) / span * n - 0.5
        i0_float = np.floor(x)
        w = np.clip(x - i0_float, 0.0, 1.0)
        i0 = i0_float.astype(np.int64)
        i1 = i0 + 1
        if wrap:
            i0 %= n
            i1 %= n
        else:
            i0 = np.clip(i0, 0, n - 1)
            i1 = np.clip(i1, 0, n - 1)
        return i0, i1, w

    a0, a1, wa = indices(az, -np.pi, 2.0 * np.pi, n_az, True)
    e0, e1, we = indices(el, -np.pi / 2.0, np.pi, n_el, False)
    return (table[a0, e0] * (1.0 - wa) * (1.0 - we)
            + table[a1, e0] * wa * (1.0 - we)
            + table[a0, e1] * (1.0 - wa) * we
            + table[a1, e1] * wa * we)


def _policy_reference(trace: dict[str, np.ndarray], table: np.ndarray | None) -> np.ndarray:
    p = trace["preview_position_bound_world"]
    q = trace["preview_quaternion_bound_world_xyzw"]
    tangent = trace["gamma_tangent_bound_world"]
    s_k = trace["preview_s_m"]
    s_grid = trace["gamma_s"]
    # Native _heading_quat() sets x/y to zero and renormalizes the remaining
    # z/w components.  This is intentionally not a generic yaw extractor.
    heading = trace["base_quaternion_xyzw"].copy()
    heading[:2] = 0.0
    heading /= max(float(np.linalg.norm(heading)), 1e-12)
    heading_inv = quat_conj(heading)
    p_heading = quat_apply(heading_inv, p - trace["base_position"])
    q_heading = quat_mul(np.broadcast_to(heading_inv, q.shape), q)
    rot6d = rotation_matrix_to_rot6d_rows(quat_to_matrix(q_heading))

    # Reproduce the native tangent_at(s_k) interpolation on the Gamma grid.
    tangent_k = np.empty_like(p)
    for j, value in enumerate(s_k):
        if value <= s_grid[0]:
            tangent_k[j] = tangent[0]
        elif value >= s_grid[-1]:
            tangent_k[j] = tangent[-1]
        else:
            i = int(np.searchsorted(s_grid, value, side="right") - 1)
            alpha = float((value - s_grid[i]) / (s_grid[i + 1] - s_grid[i]))
            v = (1.0 - alpha) * tangent[i] + alpha * tangent[i + 1]
            tangent_k[j] = v / max(float(np.linalg.norm(v)), 1e-12)
    tangent_heading = quat_apply(np.broadcast_to(heading_inv, (len(tangent_k), 4)), tangent_k)
    sdot = trace["preview_sdot_mps"]
    if table is None:
        rho = np.full(len(p), np.nan, dtype=np.float64)
    else:
        d = p - trace["shoulder_position"]
        r = np.linalg.norm(d, axis=-1)
        direction = d / np.maximum(r[:, None], 1e-12)
        shoulder_r = quat_to_matrix(trace["shoulder_quaternion_xyzw"])
        u_shoulder = (shoulder_r.T @ direction.T).T
        rho = r / np.maximum(_table_query(table, u_shoulder), 1e-3)
    return np.concatenate((p_heading, rot6d, tangent_heading, sdot[:, None], rho[:, None]), axis=-1).reshape(-1)


def _grid_interpolate(s_grid: np.ndarray, positions: np.ndarray, quaternions: np.ndarray, query: np.ndarray):
    """Interpolate the native Gamma grid at the native time-law samples."""
    s_grid = np.asarray(s_grid, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    out_p = np.empty((len(query), 3), dtype=np.float64)
    out_q = np.empty((len(query), 4), dtype=np.float64)
    for j, value in enumerate(query):
        value = float(np.clip(value, s_grid[0], s_grid[-1]))
        if value <= s_grid[0]:
            out_p[j], out_q[j] = positions[0], quaternions[0]
            continue
        if value >= s_grid[-1]:
            out_p[j], out_q[j] = positions[-1], quaternions[-1]
            continue
        i = int(np.searchsorted(s_grid, value, side="right") - 1)
        alpha = float((value - s_grid[i]) / (s_grid[i + 1] - s_grid[i]))
        out_p[j] = (1.0 - alpha) * positions[i] + alpha * positions[i + 1]
        out_q[j] = quat_slerp(quaternions[i], quaternions[i + 1], alpha)
    return out_p, out_q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reach-table", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.trace_dir / "trace_summary.json").read_text())
    table = None
    if args.reach_table is not None:
        # The reach table is a data artifact, not a policy/checkpoint.  This
        # audit accepts only the weights-only payload and never executes it.
        import torch

        payload = torch.load(str(args.reach_table), map_location="cpu", weights_only=True)
        table = np.asarray(payload["table"], dtype=np.float64)

    rows = []
    max_anchor = 0.0
    max_bound_position = 0.0
    max_bound_rotation = 0.0
    max_preview_position = 0.0
    max_preview_rotation = 0.0
    max_policy_reference = 0.0
    for item in summary["traces"]:
        data = dict(np.load(item["path"], allow_pickle=False))
        anchor = native_translation_anchor(
            NativeAnchorInputs(
                base_position=data["base_position"],
                base_quaternion_xyzw=data["base_quaternion_xyzw"],
                arm_mount_translation_body=data["mount_tf_xyzrpy"][:3],
                anchor_offset_body=data["anchor_offset_body"],
                rho_star=float(data["rho_star"]),
                reach_max_m=float(data["reach_max_m"]),
            )
        )
        anchor_err = float(np.linalg.norm(anchor["anchor_position"] - data["anchor_position"]))
        max_anchor = max(max_anchor, anchor_err)
        local_p = data["gamma_p_bound_world"] - data["anchor_position"]
        time_p_bound, time_q_bound = _grid_interpolate(
            data["gamma_s"], data["gamma_p_bound_world"],
            data["gamma_q_bound_world_xyzw"], data["time_law_s"],
        )
        time_p_local = time_p_bound - data["anchor_position"]
        task_poses = pose_from_position_quaternion(time_p_local, time_q_bound)
        shape = TrajectoryShapeV1.from_task_poses(
            trajectory_id="native-reset-%03d" % int(data["reset_index"]),
            parent_id="native-generator-trace",
            point_timestamps=data["time_law_t"],
            task_poses=task_poses,
            task_frame_semantics="NATIVE_WORLD_ALIGNED_TRANSLATION_ONLY",
            family="HISTORICAL_ROBODUET_NATIVE_GENERATOR",
        )
        identity = np.eye(4, dtype=np.float64)
        binding = EpisodeReferenceBindingV1(
            episode_id="native-reset-%03d" % int(data["reset_index"]),
            reset_timestamp=0.0,
            native_base_pose=pose_from_position_quaternion(data["base_position"], data["base_quaternion_xyzw"]),
            native_shoulder_pose=pose_from_position_quaternion(data["shoulder_position"], data["shoulder_quaternion_xyzw"]),
            native_reach_anchor_pose=pose_from_position_quaternion(data["anchor_position"], data["anchor_rotation_xyzw"]),
            native_eef_semantic_id="x5_link6",
            tcp_semantic_id="gripper_center",
            T_world_anchor=pose_from_position_quaternion(data["anchor_position"], data["anchor_rotation_xyzw"]),
            T_anchor_task=identity,
            T_link_tcp=pose_from_position_quaternion(data["ee_local_offset_link6"], [0.0, 0.0, 0.0, 1.0]),
            source_shape=shape,
            placement_profile=PlacementProfileV1("native-generator", "CENTERED"),
            native_generator_reference_semantics="NATIVE_CANONICAL_LOCAL_TRANSLATION_ONLY",
            policy_reference_frame="NATIVE_DYNAMIC_HEADING_FRAME",
        )
        bound = binding.T_world_eef
        bound_p = bound[..., :3, 3]
        bound_q = binding.bound_chunk().eef_orientation
        pos_err = np.linalg.norm(bound_p - time_p_bound, axis=-1)
        rot_err = quat_geodesic(bound_q, time_q_bound)
        max_bound_position = max(max_bound_position, float(np.max(pos_err)))
        max_bound_rotation = max(max_bound_rotation, float(np.max(rot_err)))

        preview = arc_length_preview_from_grid(
            data["gamma_s"], data["gamma_p_bound_world"], data["gamma_q_bound_world_xyzw"],
            s_current=float(data["traj_s_current_m"]),
            horizon_m=0.5,
            source_frame="NATIVE_WORLD_ALIGNED",
            tcp_semantic_id="gripper_center",
        )
        # WBCPreviewV1 intentionally stores rot6d; decode it for the geodesic
        # comparison against native sample_preview's quaternion.
        preview_q = matrix_to_quat(rot6d_rows_to_matrix(preview.eef_rot6d))
        preview_pos_err = np.linalg.norm(preview.eef_position - data["preview_position_bound_world"], axis=-1)
        preview_rot_err = quat_geodesic(preview_q, data["preview_quaternion_bound_world_xyzw"])
        max_preview_position = max(max_preview_position, float(np.max(preview_pos_err)))
        max_preview_rotation = max(max_preview_rotation, float(np.max(preview_rot_err)))

        policy_ref = _policy_reference(data, table)
        native_terms = data["arm_traj_terms_134"]
        policy_err = float(np.max(np.abs(policy_ref - native_terms[8:]))) if table is not None else float("nan")
        if np.isfinite(policy_err):
            max_policy_reference = max(max_policy_reference, policy_err)
        rows.append({
            "reset_index": int(data["reset_index"]),
            "gamma_points": int(len(data["gamma_s"])),
            "anchor_position_error_m": anchor_err,
            "bound_position_max_m": float(np.max(pos_err)),
            "bound_rotation_max_rad": float(np.max(rot_err)),
            "preview_position_max_m": float(np.max(preview_pos_err)),
            "preview_rotation_max_rad": float(np.max(preview_rot_err)),
            "policy_reference_max_abs_error": policy_err,
            "shape_semantics": shape.task_frame_semantics,
        })

    result = {
        "status": "PASS",
        "episode_count": len(rows),
        "trajectory_count": len(rows),
        "max_anchor_position_error_m": max_anchor,
        "max_bound_position_error_m": max_bound_position,
        "max_bound_rotation_error_rad": max_bound_rotation,
        "max_preview_position_error_m": max_preview_position,
        "max_preview_rotation_error_rad": max_preview_rotation,
        "max_policy_reference_abs_error": None if table is None else max_policy_reference,
        "policy_reference_table_used": table is not None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
