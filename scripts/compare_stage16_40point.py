#!/usr/bin/env python3
"""Decompose the matched continuous-vs-40-point Stage16 ablation.

The 40-point mode is an interface reconstruction, not an actual GR00T call.
This script keeps three quantities separate:

* reference reconstruction error: continuous source pose vs its 40-point
  piecewise interpolation;
* controller tracking error: the native WBC error reported for each mode;
* end-to-end task-space error: achieved pose vs the original continuous source
  pose at the rollout timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from go2arx5_vla.benchmark.progressive_relocation import load_npz, trajectory_descriptor
from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_slerp, quat_to_matrix
from go2arx5_vla.metrics.whole_body import metric_summary


def _slerp_at(query: np.ndarray, grid: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    result = np.empty((len(query), 4), dtype=np.float64)
    for index, value in enumerate(np.clip(query, grid[0], grid[-1])):
        right = int(np.searchsorted(grid, value, side="right"))
        left = max(0, min(right - 1, len(grid) - 2))
        span = float(grid[left + 1] - grid[left])
        fraction = 0.0 if span <= 0.0 else (float(value) - float(grid[left])) / span
        result[index] = quat_slerp(quaternion[left], quaternion[left + 1], fraction)
    return result


def _pose_at(
    query: np.ndarray,
    grid: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    position = np.stack(
        [np.interp(np.clip(query, grid[0], grid[-1]), grid, position[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return position, _slerp_at(query, grid, quaternion)


def _summary_from_pose_error(
    achieved_position: np.ndarray,
    achieved_quaternion: np.ndarray,
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
) -> dict[str, Any]:
    position_error = np.linalg.norm(achieved_position - reference_position, axis=-1)
    rotation_error = geodesic_angle(
        quat_to_matrix(achieved_quaternion), quat_to_matrix(reference_quaternion)
    )
    return {
        "position_m": metric_summary(position_error),
        "rotation_rad": metric_summary(rotation_error),
        "time_in_tolerance_fraction_0.05m_0.20rad": float(
            np.mean((position_error <= 0.05) & (rotation_error <= 0.20))
        ),
        "sample_count": int(len(position_error)),
    }


def _load_metrics(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        str(row["trajectory_id"]): row.get("metrics", {})
        for row in payload.get("cases", [])
        if "trajectory_id" in row
    }


def _load_rollout_path(metrics: Mapping[str, Any], root: Path, trajectory_id: str) -> Path:
    path = metrics.get("npz_path")
    if path:
        candidate = Path(str(path))
        if candidate.exists():
            return candidate
    return root / trajectory_id / "native_external_rollout.npz"


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return None
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _copy_tracking_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "position_m": metrics.get("position_m", {}),
        "rotation_rad": metrics.get("rotation_rad", {}),
        "lag_arc_m": metrics.get("lag_arc_m", {}),
        "path_completion": metrics.get("path_completion"),
        "reference_completion": metrics.get("reference_completion"),
        "time_in_tolerance_fraction_0.05m_0.20rad": metrics.get(
            "time_in_tolerance_fraction_0.05m_0.20rad"
        ),
        "termination_code": metrics.get("termination_code"),
        "status": metrics.get("status"),
    }


def _delta_summary(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pose_key in ("position_m", "rotation_rad"):
        result[pose_key] = {}
        for statistic in ("mean", "rmse", "median", "p95", "max"):
            a = first.get(pose_key, {}).get(statistic)
            b = second.get(pose_key, {}).get(statistic)
            result[pose_key][statistic] = None if a is None or b is None else float(b - a)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-manifest", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--continuous-metrics", type=Path, required=True)
    parser.add_argument("--forty-metrics", type=Path, required=True)
    parser.add_argument("--continuous-rollout-root", type=Path, required=True)
    parser.add_argument("--forty-rollout-root", type=Path, required=True)
    parser.add_argument("--reset-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    continuous_manifest = json.loads(args.continuous_manifest.read_text())
    ablation_manifest = json.loads(args.ablation_manifest.read_text())
    continuous_cases = {
        str(case["trajectory_id"]): case for case in continuous_manifest.get("cases", [])
    }
    ablation_cases = {
        str(case["trajectory_id"]): case for case in ablation_manifest.get("cases", [])
    }
    if set(continuous_cases) != set(ablation_cases):
        raise ValueError("continuous and 40-point manifests are not matched")
    if any(
        case.get("reference_mode") == "GR00T40_RECONSTRUCTED_REFERENCE"
        for case in continuous_cases.values()
    ):
        raise ValueError("continuous manifest contains the 40-point mode")
    if not all(
        case.get("reference_mode") == "GR00T40_RECONSTRUCTED_REFERENCE"
        for case in ablation_cases.values()
    ):
        raise ValueError("ablation manifest is not marked as GR00T40_RECONSTRUCTED_REFERENCE")

    continuous_metrics = _load_metrics(args.continuous_metrics)
    forty_metrics = _load_metrics(args.forty_metrics)
    trace = load_npz(args.reset_trace)
    anchor = np.asarray(trace["anchor_position"], dtype=np.float64)
    cases = []
    descriptor_rows = []

    for trajectory_id in continuous_cases:
        case = continuous_cases[trajectory_id]
        source = load_npz(Path(str(case["trajectory_path"])))
        timestamp = np.asarray(source["timestamp"], dtype=np.float64)
        source_position = np.asarray(source["position"], dtype=np.float64) + anchor[None, :]
        source_quaternion = np.asarray(source["quaternion"], dtype=np.float64)
        points_40 = np.linspace(float(timestamp[0]), float(timestamp[-1]), 40)
        point_position, point_quaternion = _pose_at(
            points_40, timestamp, source_position, source_quaternion
        )
        reconstructed_position, reconstructed_quaternion = _pose_at(
            timestamp, points_40, point_position, point_quaternion
        )
        reconstruction = _summary_from_pose_error(
            reconstructed_position,
            reconstructed_quaternion,
            source_position,
            source_quaternion,
        )

        a_metrics = continuous_metrics.get(trajectory_id, {})
        b_metrics = forty_metrics.get(trajectory_id, {})
        a_rollout = load_npz(_load_rollout_path(a_metrics, args.continuous_rollout_root, trajectory_id))
        b_rollout = load_npz(_load_rollout_path(b_metrics, args.forty_rollout_root, trajectory_id))
        a_time = np.asarray(a_rollout["sim_time"], dtype=np.float64)
        b_time = np.asarray(b_rollout["sim_time"], dtype=np.float64)
        a_ref_position, a_ref_quaternion = _pose_at(
            a_time, timestamp, source_position, source_quaternion
        )
        b_ref_position, b_ref_quaternion = _pose_at(
            b_time, timestamp, source_position, source_quaternion
        )
        a_end_to_end = _summary_from_pose_error(
            np.asarray(a_rollout["achieved_pos"], dtype=np.float64),
            np.asarray(a_rollout["achieved_quat"], dtype=np.float64),
            a_ref_position,
            a_ref_quaternion,
        )
        b_end_to_end = _summary_from_pose_error(
            np.asarray(b_rollout["achieved_pos"], dtype=np.float64),
            np.asarray(b_rollout["achieved_quat"], dtype=np.float64),
            b_ref_position,
            b_ref_quaternion,
        )
        descriptor = trajectory_descriptor(
            timestamp,
            np.asarray(source["position"], dtype=np.float64),
            source_quaternion,
        )
        descriptor_rows.append((descriptor, reconstruction, b_end_to_end))
        cases.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_hash": case.get("trajectory_hash"),
                "reference_reconstruction_error": reconstruction,
                "controller_tracking_error": {
                    "continuous": _copy_tracking_metrics(a_metrics),
                    "gr00t40_reconstructed": _copy_tracking_metrics(b_metrics),
                    "penalty_gr00t40_minus_continuous": _delta_summary(a_metrics, b_metrics),
                },
                "end_to_end_task_space_error": {
                    "continuous": a_end_to_end,
                    "gr00t40_reconstructed": b_end_to_end,
                    "penalty_gr00t40_minus_continuous": _delta_summary(
                        a_end_to_end, b_end_to_end
                    ),
                },
                "descriptor": descriptor,
                "matched": {
                    "seed_continuous": continuous_cases[trajectory_id].get("reset_seed"),
                    "seed_gr00t40": ablation_cases[trajectory_id].get("reset_seed"),
                    "trajectory_path_same": continuous_cases[trajectory_id].get("trajectory_path")
                    == ablation_cases[trajectory_id].get("trajectory_path"),
                    "checkpoint_hashes_same": all(
                        continuous_cases[trajectory_id].get(key)
                        == ablation_cases[trajectory_id].get(key)
                        for key in (
                            "archive_sha256",
                            "arm_checkpoint_sha256",
                            "dog_checkpoint_sha256",
                            "config_sha256",
                        )
                    ),
                    "actual_gr00t_called": bool(
                        ablation_cases[trajectory_id].get("actual_gr00t_called", False)
                    ),
                },
            }
        )

    feature_values = {
        "curvature_proxy_rad_per_m": [],
        "translation_speed_p95_mps": [],
        "translation_speed_max_mps": [],
        "angular_speed_p95_radps": [],
        "path_length_m": [],
        "orientation_range_rad": [],
    }
    metric_values = {
        "reconstruction_position_mean_m": [],
        "reconstruction_rotation_mean_rad": [],
        "end_to_end_40_position_mean_m": [],
        "end_to_end_40_rotation_mean_rad": [],
    }
    for descriptor, reconstruction, end_to_end in descriptor_rows:
        feature_values["curvature_proxy_rad_per_m"].append(
            descriptor.get("curvature_proxy_rad_per_m", descriptor.get("curvature_proxy_max"))
        )
        feature_values["translation_speed_p95_mps"].append(
            descriptor["translation_speed_mps"]["p95"]
        )
        feature_values["translation_speed_max_mps"].append(
            descriptor["translation_speed_mps"]["max"]
        )
        feature_values["angular_speed_p95_radps"].append(
            descriptor["angular_speed_radps"]["p95"]
        )
        feature_values["path_length_m"].append(descriptor["path_length_m"])
        feature_values["orientation_range_rad"].append(descriptor["orientation_range_rad_max"])
        metric_values["reconstruction_position_mean_m"].append(
            reconstruction["position_m"]["mean"]
        )
        metric_values["reconstruction_rotation_mean_rad"].append(
            reconstruction["rotation_rad"]["mean"]
        )
        metric_values["end_to_end_40_position_mean_m"].append(
            end_to_end["position_m"]["mean"]
        )
        metric_values["end_to_end_40_rotation_mean_rad"].append(
            end_to_end["rotation_rad"]["mean"]
        )

    correlations = {
        metric_name: {
            feature_name: _pearson(metric_series, feature_values[feature_name])
            for feature_name in feature_values
        }
        for metric_name, metric_series in metric_values.items()
    }
    mode_counts = {
        "continuous": {},
        "gr00t40_reconstructed": {},
    }
    for mode, metrics in mode_counts.items():
        source = continuous_metrics if mode == "continuous" else forty_metrics
        for row in source.values():
            code = str(row.get("termination_code", "UNKNOWN"))
            metrics[code] = metrics.get(code, 0) + 1

    payload = {
        "schema": "STAGE16_40POINT_ABLATION_COMPARISON_V1",
        "continuous_manifest": str(args.continuous_manifest),
        "ablation_manifest": str(args.ablation_manifest),
        "continuous_metrics": str(args.continuous_metrics),
        "forty_metrics": str(args.forty_metrics),
        "interface_chain": [
            "continuous trajectory",
            "40 timestamped EEF points",
            "TrajectoryChunkV1",
            "canonical GR00T-to-WBC adapter",
            "native 9-point preview",
            "native WBC",
        ],
        "actual_gr00t_called": False,
        "case_count": len(cases),
        "termination_counts": mode_counts,
        "cases": cases,
        "dependence_correlations": correlations,
        "interpretation": (
            "Reference reconstruction, native controller tracking, and end-to-end task-space "
            "error are reported separately. This is an interface ablation and does not change "
            "the GR00T horizon or establish visual-policy performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "termination_counts": mode_counts,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
