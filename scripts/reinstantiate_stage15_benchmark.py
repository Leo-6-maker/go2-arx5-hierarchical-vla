#!/usr/bin/env python3
"""Reinstantiate E1/E2 trajectories as native-embodiment benchmarks.

The original E1/E2 files are retained as immutable source artifacts.  This
script creates a derived shape view (initial-E​​EF-relative SE(3) motion) and a
separate native placement profile.  The positional classification is a
bounded kinematic analysis using the delivered native reach table; it is not a
dynamic WBC result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.contracts import TrajectoryChunkV1, TrajectoryShapeV1
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _derived_hash(shape: TrajectoryShapeV1) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(shape.point_timestamps, dtype="<f8").tobytes())
    h.update(np.asarray(shape.position, dtype="<f8").tobytes())
    h.update(np.asarray(shape.quaternion, dtype="<f8").tobytes())
    h.update(shape.task_frame_semantics.encode("utf-8"))
    return h.hexdigest()


def _table_query(table: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    az = np.arctan2(u[..., 1], u[..., 0])
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    n_az, n_el = table.shape

    def idx(value, lo, span, n, wrap):
        x = (value - lo) / span * n - 0.5
        i0f = np.floor(x)
        w = np.clip(x - i0f, 0.0, 1.0)
        i0 = i0f.astype(np.int64)
        i1 = i0 + 1
        if wrap:
            i0 %= n
            i1 %= n
        else:
            i0 = np.clip(i0, 0, n - 1)
            i1 = np.clip(i1, 0, n - 1)
        return i0, i1, w

    a0, a1, wa = idx(az, -np.pi, 2.0 * np.pi, n_az, True)
    e0, e1, we = idx(el, -np.pi / 2.0, np.pi, n_el, False)
    return (table[a0, e0] * (1.0 - wa) * (1.0 - we)
            + table[a1, e0] * wa * (1.0 - we)
            + table[a0, e1] * (1.0 - wa) * we
            + table[a1, e1] * wa * we)


def _load_native_state(trace_path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(trace_path, allow_pickle=False))


def _native_profile(row: dict, source_class: str, source_id: str) -> tuple[str, dict]:
    if source_class == "STATIC":
        intent = "STATIC_BASE_RELOCATION_TARGET"
        scale = 1.12
    elif source_class == "CONTINUOUS":
        intent = "CONTINUOUS_BASE_COORDINATION_TARGET"
        scale = 0.72
    elif source_class == "INFEASIBLE":
        intent = "EXPECTED_INFEASIBLE"
        scale = 1.80
    elif source_class == "BOUNDARY":
        intent = "BOUNDARY"
        scale = 0.95
    else:
        intent = "CENTERED"
        scale = 0.60
    return intent, {
        "profile_id": "%s-native-placement" % source_id,
        "intent": intent,
        "initial_reach_scale": scale,
        "anchor_direction": "native_reset_anchor_offset_body",
        "classification_basis": "native_2d_reach_table_position_only",
    }


def _classify_native(
    shape_position: np.ndarray,
    *,
    native: dict[str, np.ndarray],
    reach_table: np.ndarray,
    profile: dict,
    grid_extent: float = 1.0,
    grid_step: float = 0.05,
) -> dict:
    base = native["base_position"].astype(np.float64)
    base_r = quat_to_matrix(native["base_quaternion_xyzw"])
    mount = native["mount_tf_xyzrpy"][:3].astype(np.float64)
    shoulder0 = native["shoulder_position"].astype(np.float64)
    shoulder_r = quat_to_matrix(native["shoulder_quaternion_xyzw"])
    direction = native["direction_world"].astype(np.float64)
    r_anchor = float(native["reach_max_m"])
    scale = float(profile["initial_reach_scale"])
    start = shoulder0 + direction * (scale * r_anchor)
    target = start[None, :] + np.asarray(shape_position, dtype=np.float64)

    def ratios(shoulders: np.ndarray) -> np.ndarray:
        delta = target[None, :, :] - shoulders[:, None, :]
        radius = np.linalg.norm(delta, axis=-1)
        bearing_world = delta / np.maximum(radius[..., None], 1e-12)
        bearing_shoulder = np.einsum("ij,ctj->cti", shoulder_r.T, bearing_world)
        rmax = _table_query(reach_table, bearing_shoulder)
        return radius / np.maximum(rmax, 1e-3)

    current = ratios(shoulder0[None])[0]
    offsets = np.arange(-grid_extent, grid_extent + grid_step * 0.5, grid_step)
    xy = np.asarray([(x, y) for x in offsets for y in offsets], dtype=np.float64)
    xy3 = np.concatenate((xy, np.zeros((len(xy), 1), dtype=np.float64)), axis=-1)
    shoulders = shoulder0[None, :] + xy3
    static_rho = ratios(shoulders)
    static_score = np.max(static_rho, axis=1)
    best_static_index = int(np.argmin(static_score))
    best_static = float(static_score[best_static_index])
    pointwise = np.min(static_rho, axis=0)
    best_pointwise = float(np.max(pointwise))
    fixed_score = float(np.max(current))
    if fixed_score <= 0.85:
        native_class = "NATIVE_ARM_EASY"
    elif fixed_score <= 1.0:
        native_class = "NATIVE_BOUNDARY"
    elif best_static <= 1.0:
        native_class = "NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT"
    elif best_pointwise <= 1.0:
        native_class = "NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED"
    else:
        native_class = "NATIVE_GEOMETRICALLY_INFEASIBLE"
    return {
        "native_class": native_class,
        "position_only": True,
        "fixed_base_max_rho": fixed_score,
        "best_static_max_rho": best_static,
        "best_static_base_xy_offset": xy[best_static_index].tolist(),
        "best_pointwise_max_rho": best_pointwise,
        "native_anchor_start": start.tolist(),
        "target_position_min": target.min(axis=0).tolist(),
        "target_position_max": target.max(axis=0).tolist(),
        "grid_extent_m": grid_extent,
        "grid_step_m": grid_step,
        "reachability_table_range_m": [float(reach_table.min()), float(reach_table.max())],
        "dynamic_feasibility": "NOT_EVALUATED",
        "orientation_feasibility": "NOT_EVALUATED",
    }


def _descriptor(shape: TrajectoryShapeV1) -> dict:
    p = shape.position
    q = shape.quaternion
    dt = np.diff(shape.point_timestamps)
    dp = np.diff(p, axis=0)
    step = np.linalg.norm(dp, axis=-1)
    speed = step / np.maximum(dt, 1e-12)
    angular = geodesic_angle(quat_to_matrix(q[:-1]), quat_to_matrix(q[1:])) / np.maximum(dt, 1e-12)
    if len(p) > 2:
        tangent_change = np.linalg.norm(np.diff(dp, axis=0), axis=-1)
        curvature = tangent_change / np.maximum(step[:-1] * step[1:], 1e-9)
    else:
        curvature = np.zeros(0)
    orientation_range = geodesic_angle(
        np.broadcast_to(quat_to_matrix(q[0]), (len(q), 3, 3)), quat_to_matrix(q)
    )
    return {
        "duration_s": float(shape.duration_s),
        "path_length_m": float(step.sum()),
        "net_displacement_m": float(np.linalg.norm(p[-1] - p[0])),
        "translation_speed_mps": {
            "p01": float(np.percentile(speed, 1)), "median": float(np.median(speed)),
            "p95": float(np.percentile(speed, 95)), "max": float(np.max(speed)),
        },
        "angular_speed_radps": {
            "p01": float(np.percentile(angular, 1)), "median": float(np.median(angular)),
            "p95": float(np.percentile(angular, 95)), "max": float(np.max(angular)),
        },
        "workspace_min": p.min(axis=0).tolist(),
        "workspace_max": p.max(axis=0).tolist(),
        "orientation_range_rad_max": float(np.max(orientation_range)),
        "curvature_proxy_max": float(np.max(curvature)) if len(curvature) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--e2-root", type=Path, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--reach-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shape_dir = args.output / "shapes"
    shape_dir.mkdir(parents=True, exist_ok=True)
    native = _load_native_state(args.native_trace)
    import torch

    reach_payload = torch.load(str(args.reach_table), map_location="cpu", weights_only=True)
    reach_table = np.asarray(reach_payload["table"], dtype=np.float64)
    e1_manifest_path = args.e1_root / "benchmark" / "E1_BENCHMARK_MANIFEST.json"
    e2_manifest_path = args.e2_root / "benchmark" / "E2_WHOLE_BODY_EXTENSION.json"
    e1_manifest = json.loads(e1_manifest_path.read_text())
    e2_manifest = json.loads(e2_manifest_path.read_text())
    entries = []

    sources = [("E1", e1_manifest.get("trajectories", []), args.e1_root, "E1"),
               ("E2", e2_manifest.get("rows", []), args.e2_root, "E2")]
    for source_name, manifest_rows, root, kind in sources:
        for row in manifest_rows:
            trajectory_id = row["trajectory_id"]
            raw_path = root / row["path"]
            data = np.load(raw_path, allow_pickle=False)
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
            chunk = TrajectoryChunkV1(
                chunk_id=trajectory_id,
                source="%s_SIM_REFERENCE_ONLY" % source_name,
                observation_timestamp=float(timestamps[0]),
                point_timestamps=timestamps,
                eef_position=np.asarray(data["position"], dtype=np.float64),
                eef_orientation=np.asarray(data["quaternion"], dtype=np.float64),
                source_frame="SIM_REFERENCE_ONLY",
                tcp_semantic_id="SIM_PROVISIONAL_TCP",
                validity_start=float(timestamps[0]),
                validity_end=float(timestamps[-1]),
                gripper=np.asarray(data["gripper"], dtype=np.float64) if "gripper" in data else None,
                provenance={"original_file_sha256": _sha256(raw_path), "source_manifest": str(raw_path)},
            )
            parent_id = row.get("parent_group", row.get("parent_e1", trajectory_id)) or trajectory_id
            family = row.get("family", "UNKNOWN")
            shape = TrajectoryShapeV1.from_chunk(
                chunk,
                parent_id=str(parent_id),
                family=family,
                metadata={
                    "derived_view": "INITIAL_EEF_RELATIVE_SE3_SHAPE",
                    "original_source": source_name,
                    "original_class": row.get("bucket", row.get("generator_kind", "UNKNOWN")),
                },
                task_frame_semantics="INITIAL_EEF_POSE_FRAME",
            )
            original_class = row.get("bucket", row.get("generator_kind", "UNKNOWN"))
            if source_name == "E1":
                if row.get("parent_group") == "boundary_set":
                    source_class = "BOUNDARY"
                elif row.get("parent_group") == "stress_set":
                    source_class = "INFEASIBLE"
                elif row.get("parent_group") == "whole_body_families":
                    source_class = "CONTINUOUS"
                else:
                    source_class = "CENTERED"
            else:
                source_class = str(row.get("generator_kind", "UNKNOWN"))
            intent, profile = _native_profile(row, source_class, trajectory_id)
            classification = _classify_native(shape.position, native=native, reach_table=reach_table, profile=profile)
            shape_file = shape_dir / (trajectory_id + ".npz")
            np.savez_compressed(
                shape_file,
                timestamp=shape.point_timestamps,
                position=shape.position,
                quaternion=shape.quaternion,
            )
            entry = {
                "trajectory_id": trajectory_id,
                "source": source_name,
                "parent_id": str(parent_id),
                "family": family,
                "shape_file": str(shape_file),
                "shape_sha256": _derived_hash(shape),
                "original_file": str(raw_path),
                "original_file_sha256": _sha256(raw_path),
                "original_class": original_class,
                "original_placement_intent": intent,
                "placement_profile": profile,
                "descriptor": _descriptor(shape),
                "native_classification": classification,
            }
            entries.append(entry)

    counts = {}
    for entry in entries:
        label = entry["native_classification"]["native_class"]
        counts[label] = counts.get(label, 0) + 1
    result = {
        "schema": "NATIVE_BENCHMARK_V1",
        "source_semantics": "E1/E2 absolute placement is retained only as provenance; shape is initial-EEF-relative",
        "native_reference_reset_trace": str(args.native_trace),
        "native_classification_basis": "bounded native reach-table positional analysis; no dynamics",
        "trajectory_count": len(entries),
        "native_class_counts": counts,
        "entries": entries,
    }
    (args.output / "native_benchmark_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"trajectory_count": len(entries), "native_class_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
