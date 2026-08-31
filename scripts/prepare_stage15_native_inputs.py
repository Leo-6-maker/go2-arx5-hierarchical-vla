#!/usr/bin/env python3
"""Materialize reset-bound native WBC inputs from Stage 15 shape objects.

The generated NPZ files are external run artifacts and are normally stored
under the node-local Stage 15 root.  This script is kept in the integration
repository so the binding operation is reproducible without importing the
private WBC package or loading a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.contracts import EpisodeReferenceBindingV1, PlacementProfileV1, TrajectoryShapeV1
from go2arx5_vla.trajectory.relative_se3 import pose_from_position_quaternion
from go2arx5_vla.trajectory.so3 import matrix_to_quat
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics


DEFAULT_IDS = (
    "E1_W0_00",
    "E1_W0_01",
    "E1_ang_0_00",
    "E1_curvature_1_00",
    "E1_coupling_1_00",
    "E2_static_0_0",
    "E2_cont_0_0",
)


def _quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = np.asarray(q0, dtype=np.float64)
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
    return np.asarray(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ],
        dtype=np.float64,
    )


def _pose_from_trace(trace: dict, position_key: str, quaternion_key: str) -> np.ndarray:
    return pose_from_position_quaternion(trace[position_key], trace[quaternion_key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=list(DEFAULT_IDS))
    parser.add_argument(
        "--center-all",
        action="store_true",
        help=(
            "diagnostic only: bind every shape's first pose to the reset TCP; "
            "does not preserve the placement challenge"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entries = {row["trajectory_id"]: row for row in manifest["entries"]}
    trace = dict(np.load(args.trace, allow_pickle=False))
    anchor_position = np.asarray(trace["anchor_position"], dtype=np.float64)
    reach_max = float(np.asarray(trace["reach_max_m"]))
    direction_world = np.asarray(trace["direction_world"], dtype=np.float64)
    native_anchor = _pose_from_trace(trace, "anchor_position", "anchor_rotation_xyzw")
    base_pose = _pose_from_trace(trace, "base_position", "base_quaternion_xyzw")
    shoulder_pose = _pose_from_trace(
        trace, "shoulder_position", "shoulder_quaternion_xyzw"
    )
    tcp_tf = pose_from_position_quaternion(
        np.asarray(trace["ee_local_offset_link6"], dtype=np.float64),
        [0.0, 0.0, 0.0, 1.0],
    )
    native_kin = NativeArmKinematics.from_urdf(args.urdf)
    reset_tcp_pose = pose_from_position_quaternion(
        trace["base_position"], trace["base_quaternion_xyzw"]
    ) @ native_kin.tcp_pose(trace["arm_q"])
    reset_tcp_position = reset_tcp_pose[:3, 3]
    reset_tcp_quaternion = matrix_to_quat(reset_tcp_pose[:3, :3])
    outputs = []
    for trajectory_id in args.ids:
        if trajectory_id not in entries:
            raise KeyError("trajectory is absent from native benchmark manifest: %s" % trajectory_id)
        row = entries[trajectory_id]
        source = dict(np.load(row["shape_file"], allow_pickle=False))
        timestamps = np.asarray(source["timestamp"], dtype=np.float64)
        position = np.asarray(source["position"], dtype=np.float64)
        relative_q = np.asarray(source["quaternion"], dtype=np.float64)
        profile = row["placement_profile"]
        scale = float(profile["initial_reach_scale"])
        # The native external loader anchors positions at rho_star=0.60.  For
        # a centered benchmark, bind the shape's initial pose to the actual
        # reset TCP so a short diagnostic path is not silently turned into a
        # large initial-error test.  Deliberate edge/static/infeasible intents
        # retain an explicit reach-scaled displacement from the native anchor.
        if args.center_all or profile["intent"] in ("CENTERED", "COMFORTABLE"):
            target_start = reset_tcp_position
        else:
            target_start = anchor_position + direction_world * ((scale - 0.60) * reach_max)
        offset = target_start - anchor_position
        task_to_anchor = pose_from_position_quaternion(offset, reset_tcp_quaternion)
        profile_contract = PlacementProfileV1(
            profile_id=profile["profile_id"],
            intent=profile["intent"],
            parameters=profile,
            original_class=row.get("original_class"),
            native_class=row["native_classification"]["native_class"],
            status="CLASSIFIED_FOR_NATIVE_EMBODIMENT",
            provenance={"manifest": str(args.manifest)},
        )
        shape = TrajectoryShapeV1.from_task_poses(
            trajectory_id=trajectory_id,
            parent_id=row["parent_id"],
            point_timestamps=timestamps,
            task_poses=np.asarray(
                [
                    pose_from_position_quaternion(p, q)
                    for p, q in zip(position, relative_q)
                ]
            ),
            task_frame_semantics="INITIAL_EEF_POSE_FRAME",
            family=row["family"],
            metadata={"native_classification": row["native_classification"]},
            original_provenance={"row": row},
        )
        binding = EpisodeReferenceBindingV1(
            episode_id="stage15-reset-000",
            reset_timestamp=0.0,
            native_base_pose=base_pose,
            native_shoulder_pose=shoulder_pose,
            native_reach_anchor_pose=native_anchor,
            native_eef_semantic_id="x5_link6_gripper_center",
            tcp_semantic_id="x5_link6_gripper_center",
            T_world_anchor=native_anchor,
            T_anchor_task=task_to_anchor,
            T_link_tcp=tcp_tf,
            source_shape=shape,
            placement_profile=profile_contract,
            native_generator_reference_semantics="NATIVE_GAMMA_SE3_ARC_LENGTH_TRANSLATION_ANCHORED",
            policy_reference_frame="NATIVE_WORLD_ALIGNED_DYNAMIC_HEADING_PREVIEW",
            binding_semantics="NATIVE_TRANSLATION_ONLY_WORLD_ALIGNED",
            provenance={
                "native_trace": str(args.trace),
                "anchor_direction_world": direction_world.tolist(),
                "reach_max_m": reach_max,
                "native_anchor_position": anchor_position.tolist(),
                "center_all_override": bool(args.center_all),
            },
        )
        bound = binding.bound_chunk(source="STAGE15_NATIVE_BOUND_REFERENCE")
        path = args.output / (trajectory_id + ".npz")
        np.savez_compressed(
            path,
            timestamp=bound.point_timestamps - bound.point_timestamps[0],
            # The delivered native loader adds its reset anchor after it
            # reads the external Gamma positions.  Store the exact loader
            # input (task offset + shape), while the absolute bound trajectory
            # remains in native_bound_inputs.json for auditability.
            position=bound.eef_position - native_anchor[:3, 3][None, :],
            quaternion=bound.eef_orientation,
        )
        outputs.append(
            {
                "trajectory_id": trajectory_id,
                "path": str(path),
                "placement_intent": profile["intent"],
                "native_class": row["native_classification"]["native_class"],
                "task_to_anchor_translation_world": offset.tolist(),
                "task_to_anchor_quaternion_xyzw": reset_tcp_quaternion.tolist(),
                "reset_tcp_position_world": reset_tcp_position.tolist(),
                "reset_tcp_quaternion_xyzw": reset_tcp_quaternion.tolist(),
                "binding": binding.as_dict(),
            }
        )
    result = {
        "schema": "STAGE15_NATIVE_BOUND_INPUTS_V1",
        "manifest": str(args.manifest),
        "trace": str(args.trace),
        "reset_semantics": "single deterministic reset trace; loader anchor is translation-only and task orientation is explicit",
        "entries": outputs,
    }
    (args.output / "native_bound_inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"count": len(outputs), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
