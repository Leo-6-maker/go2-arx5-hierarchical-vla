#!/usr/bin/env python3
"""Analyze a Stage21 receding-horizon trace without model dependencies.

The native runner writes JSONL so the raw per-step evidence remains easy to
inspect.  This companion computes task-space, whole-body, internal-plan,
handoff, latency, and termination summaries.  It intentionally does not
smooth, clip, retime, or reinterpret an actual model trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.metrics import analyze_internal_plan, compute_whole_body_metrics
from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix


def _summary(values: Any) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {key: None for key in ("mean", "rmse", "median", "p95", "max")}
    if not np.isfinite(array).all():
        raise ValueError("analysis values must be finite")
    return {
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _rows_data(rows: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    keys = (
        "sim_time",
        "desired_pos",
        "desired_quat",
        "achieved_pos",
        "achieved_quat",
        "base_pos",
        "base_quat",
        "base_lin_vel",
        "base_ang_vel",
        "arm_q",
        "arm_dq",
        "dog_q",
        "dog_dq",
        "dof_pos",
        "dof_vel",
        "joint_limit_margin",
        "internal_body_plan_9d",
        "traj_s_ref",
        "traj_L",
        "traj_timing_err",
        "reset_buf",
        "fall_flag",
        "invalid_state",
        "joint_limit_violation",
        "joint_limit_termination",
    )
    data: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        if len(values) != len(rows):
            continue
        data[key] = np.asarray(values)
    return data


def _profile_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(summary.get("model_cases", []))
    profiles = [case["profile"] for case in cases if "profile" in case]
    def values(key: str) -> list[float]:
        return [float(profile[key]) for profile in profiles if key in profile]

    return {
        "chunk_count": len(profiles),
        "finite_chunk_count": sum(bool(profile.get("finite", False)) for profile in profiles),
        "duration_s": _summary(values("duration_s")),
        "path_length_m": _summary(values("translation_path_length_m")),
        "initial_position_discontinuity_m": _summary(values("initial_position_discontinuity_m")),
        "initial_rotation_discontinuity_rad": _summary(values("initial_rotation_discontinuity_rad")),
        "max_linear_speed_mps": _summary(
            [float(profile["linear_speed_mps"]["max"]) for profile in profiles if "linear_speed_mps" in profile]
        ),
        "roughness_second_difference_m": _summary(
            [
                float(profile["roughness_position_second_difference_m"]["mean"])
                for profile in profiles
                if "roughness_position_second_difference_m" in profile
            ]
        ),
        "chunk_profiles": [
            {
                "chunk_id": case.get("chunk_id"),
                "decoded_sha256": case.get("decoded_sha256"),
                "path_length_m": profile.get("translation_path_length_m"),
                "duration_s": profile.get("duration_s"),
                "max_linear_speed_mps": profile.get("linear_speed_mps", {}).get("max"),
                "roughness_mean_m": profile.get("roughness_position_second_difference_m", {}).get("mean"),
                "initial_position_discontinuity_m": profile.get("initial_position_discontinuity_m"),
                "initial_rotation_discontinuity_rad": profile.get("initial_rotation_discontinuity_rad"),
            }
            for case, profile in zip(cases, profiles)
        ],
    }


def _handoff_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    handoffs = [item.get("chunk_handoff", item) for item in summary.get("handoffs", [])]
    position = [item["previous_prediction_to_new_first_position_m"] for item in handoffs if item.get("previous_prediction_to_new_first_position_m") is not None]
    rotation = [item["previous_prediction_to_new_first_rotation_rad"] for item in handoffs if item.get("previous_prediction_to_new_first_rotation_rad") is not None]
    achieved_position = [item["achieved_to_new_first_position_m"] for item in handoffs if item.get("achieved_to_new_first_position_m") is not None]
    achieved_rotation = [item["achieved_to_new_first_rotation_rad"] for item in handoffs if item.get("achieved_to_new_first_rotation_rad") is not None]
    statuses = [item.get("lifecycle_status") for item in summary.get("handoffs", [])]
    return {
        "handoff_count": len(handoffs),
        "lifecycle_status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "previous_prediction_to_new_first_position_m": _summary(position),
        "previous_prediction_to_new_first_rotation_rad": _summary(rotation),
        "achieved_to_new_first_position_m": _summary(achieved_position),
        "achieved_to_new_first_rotation_rad": _summary(achieved_rotation),
        "raw_no_correction": True,
        "interpretation": "prediction_handoff_and_achieved_handoff_are_separate_metrics",
    }


def analyze(summary_path: Path, rows_path: Path, *, target_position: np.ndarray | None = None) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("trace contains no rows")
    data = _rows_data(rows)
    required = ("desired_pos", "desired_quat", "achieved_pos", "achieved_quat", "base_pos", "base_quat")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"trace is missing required telemetry: {', '.join(missing)}")
    tracking = compute_whole_body_metrics(data)
    desired_pos = data["desired_pos"].astype(np.float64)
    achieved_pos = data["achieved_pos"].astype(np.float64)
    desired_quat = data["desired_quat"].astype(np.float64)
    achieved_quat = data["achieved_quat"].astype(np.float64)
    position_error = np.linalg.norm(desired_pos - achieved_pos, axis=-1)
    rotation_error = geodesic_angle(quat_to_matrix(desired_quat), quat_to_matrix(achieved_quat))
    timing_margin = 0.70 * data["traj_L"].astype(np.float64) - data["traj_timing_err"].astype(np.float64)
    result: dict[str, Any] = {
        "schema": "STAGE21_RECEDING_HORIZON_METRICS_V1",
        "source_summary": str(summary_path),
        "source_rows": str(rows_path),
        "status": summary.get("status"),
        "termination": summary.get("termination"),
        "sample_count": len(rows),
        "reference_updates": summary.get("reference_updates"),
        "tracking": {
            "metric_name": "CONTROLLER_TRACKING_ERROR",
            "position_m": _summary(position_error),
            "rotation_rad": _summary(rotation_error),
            "time_in_tolerance_fraction_0.05m_0.20rad": float(np.mean((position_error <= 0.05) & (rotation_error <= 0.20))),
            "finite": bool(np.isfinite(position_error).all() and np.isfinite(rotation_error).all()),
        },
        "whole_body": tracking,
        "timing": {
            "timing_margin_m": _summary(timing_margin),
            "timing_error_m": _summary(data["traj_timing_err"]),
            "final_traj_s_ref_m": float(data["traj_s_ref"][-1]),
            "final_traj_L_m": float(data["traj_L"][-1]),
            "last_chunk_path_completion": float(np.clip(data["traj_s_ref"][-1] / max(data["traj_L"][-1], 1e-9), 0.0, 1.0)),
            "interpretation": "reference progress is reset at every hot-swap; last-chunk completion is not global task completion",
        },
        "internal_plan": analyze_internal_plan(data, burden=np.abs(data["traj_timing_err"])),
        "termination_breakdown": {
            "row_termination_codes": {
                code: sum(row.get("termination_code") == code for row in rows)
                for code in sorted({row.get("termination_code", "UNKNOWN") for row in rows})
            },
            "reset_buf_count": int(np.asarray(data.get("reset_buf", np.zeros(len(rows), dtype=bool)), dtype=bool).sum()),
            "fall_flag_count": int(np.asarray(data.get("fall_flag", np.zeros(len(rows), dtype=bool)), dtype=bool).sum()),
            "invalid_state_count": int(np.asarray(data.get("invalid_state", np.zeros(len(rows), dtype=bool)), dtype=bool).sum()),
            "joint_limit_violation_row_count": int(np.asarray(data.get("joint_limit_violation", np.zeros(len(rows), dtype=bool)), dtype=bool).sum()),
            "joint_limit_termination_row_count": int(np.asarray(data.get("joint_limit_termination", np.zeros(len(rows), dtype=bool)), dtype=bool).sum()),
            "interpretation": "margin violation is diagnostic; only explicit native termination is a joint-limit termination",
        },
        "model_profile": _profile_summary(summary),
        "handoff": _handoff_summary(summary),
        "latency": {
            "model_forward_s": _summary([case["response"]["latency_s"] for case in summary.get("model_cases", [])]),
            "offline_mailbox_wait_s": _summary([case["response_wait_wall_s"] for case in summary.get("model_cases", [])]),
            "chunk_age_at_execution_s": _summary([case["chunk_age_at_execution_s"] for case in summary.get("model_cases", [])]),
            "transport": "shared NFS offline experiment harness, not production transport",
        },
        "decomposition": {
            "VLA_PREDICTION_ERROR": "NOT_COMPUTABLE_WITHOUT_SAME_SCENE_DESIRED_LABEL_TRAJECTORY",
            "REFERENCE_RECONSTRUCTION_ERROR": "SEALED_SEPARATELY_BY_PROVIDER_INTERFACE_AUDIT",
            "CONTROLLER_TRACKING_ERROR": "COMPUTED_FROM_NATIVE_CONSUMED_REFERENCE_AND_ACHIEVED_EEF",
            "END_TO_END_TASK_SPACE_ERROR": "NOT_COMPUTED_AS_A_TASK_SUCCESS_LABEL_IS_NOT_A_CONTINUOUS_REFERENCE",
        },
        "task_marker": {
            "target_position_world_m": None if target_position is None else target_position.tolist(),
            "final_achieved_position_world_m": achieved_pos[-1].tolist(),
            "final_marker_distance_m": None if target_position is None else float(np.linalg.norm(achieved_pos[-1] - target_position)),
            "task_success_evaluated": False,
            "interpretation": "descriptive endpoint distance only; no task-performance claim",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-position", type=float, nargs=3, default=None)
    args = parser.parse_args()
    target = None if args.target_position is None else np.asarray(args.target_position, dtype=np.float64)
    result = analyze(args.summary, args.rows, target_position=target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
