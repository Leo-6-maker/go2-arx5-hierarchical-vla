#!/usr/bin/env python3
"""Evaluate one simulation-only GR00T -> native WBC closed-loop run.

The four reported quantities have deliberately different reference pairs:

* VLA_PREDICTION_ERROR: decoded model action vs desired task label;
* REFERENCE_RECONSTRUCTION_ERROR: model path vs the native consumed path;
* CONTROLLER_TRACKING_ERROR: native consumed path vs achieved EEF;
* END_TO_END_TASK_SPACE_ERROR: desired task label vs achieved EEF.

The native launcher adds a reset-derived position anchor to external
references.  This evaluator estimates that additive, simulation-only offset
from the logged consumed reference before comparing the task label with the
achieved world-frame EEF.  No trajectory values are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from go2arx5_vla.metrics import (
    controller_tracking_error,
    end_to_end_task_space_error,
    reference_reconstruction_error,
    vla_prediction_error,
)
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rot6d_rows_to_matrix,
)


LAMBDA_M_PER_RAD = 0.15


def _arc_map(position: np.ndarray, quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = quat_to_matrix(quaternion)
    translation = np.linalg.norm(np.diff(position, axis=0), axis=-1)
    angular = geodesic_angle(rotation[:-1], rotation[1:])
    segments = np.sqrt(translation * translation + (LAMBDA_M_PER_RAD * angular) ** 2)
    return segments, np.concatenate([[0.0], np.cumsum(segments)])


def _pose_at_arc(
    query: np.ndarray, position: np.ndarray, quaternion: np.ndarray, cumulative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(query, dtype=np.float64).reshape(-1)
    clipped = np.clip(values, 0.0, float(cumulative[-1]))
    out_position = np.empty((len(clipped), 3), dtype=np.float64)
    out_quaternion = np.empty((len(clipped), 4), dtype=np.float64)
    for index, value in enumerate(clipped):
        if value <= cumulative[0]:
            out_position[index] = position[0]
            out_quaternion[index] = quaternion[0]
            continue
        if value >= cumulative[-1]:
            out_position[index] = position[-1]
            out_quaternion[index] = quaternion[-1]
            continue
        segment = int(np.searchsorted(cumulative, value, side="right") - 1)
        segment = max(0, min(segment, len(position) - 2))
        fraction = float((value - cumulative[segment]) / max(cumulative[segment + 1] - cumulative[segment], 1e-12))
        out_position[index] = (1.0 - fraction) * position[segment] + fraction * position[segment + 1]
        out_quaternion[index] = quat_slerp(quaternion[segment], quaternion[segment + 1], fraction)
    return out_position, out_quaternion


def _valid_mask(data: Any) -> np.ndarray:
    if "tracking_valid" in data:
        return np.asarray(data["tracking_valid"], dtype=bool).reshape(-1)
    return np.ones(len(data["achieved_pos"]), dtype=bool)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--native-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-period-s", type=float, default=0.05)
    args = parser.parse_args()

    with np.load(args.model_output, allow_pickle=False) as model_data:
        decoded = np.asarray(model_data["decoded_physical_action"], dtype=np.float64)
        desired = np.asarray(model_data["desired_action"], dtype=np.float64)
    if decoded.shape != (1, 40, 10) or desired.shape != (40, 10):
        raise ValueError(f"unexpected model artifact shapes decoded={decoded.shape} desired={desired.shape}")
    decoded = decoded[0]
    pred_position = decoded[:, :3]
    pred_quaternion = matrix_to_quat(rot6d_rows_to_matrix(decoded[:, 3:9]))
    label_position = desired[:, :3]
    label_quaternion = matrix_to_quat(rot6d_rows_to_matrix(desired[:, 3:9]))

    with np.load(args.native_rollout, allow_pickle=False) as native_data:
        native = {key: np.asarray(native_data[key]) for key in native_data.files}
    required = ("desired_pos", "desired_quat", "achieved_pos", "achieved_quat", "traj_s_ref", "traj_L")
    missing = [key for key in required if key not in native]
    if missing:
        raise ValueError("native rollout missing: " + ", ".join(missing))
    mask = _valid_mask(native)
    if not np.any(mask):
        raise ValueError("native rollout has no valid tracking samples")

    _, pred_cumulative = _arc_map(pred_position, pred_quaternion)
    _, label_cumulative = _arc_map(label_position, label_quaternion)

    native_s = np.asarray(native["traj_s_ref"], dtype=np.float64).reshape(-1)
    native_l = np.asarray(native["traj_L"], dtype=np.float64).reshape(-1)
    pred_at_s, pred_q_at_s = _pose_at_arc(native_s, pred_position, pred_quaternion, pred_cumulative)
    native_desired_pos = np.asarray(native["desired_pos"], dtype=np.float64)
    native_desired_q = np.asarray(native["desired_quat"], dtype=np.float64)
    # The delivered external-reference loader adds an additive reset anchor
    # to positions and leaves the declared orientation convention unchanged.
    # Estimate only that logged coordinate offset; never alter the candidate.
    position_offset_samples = native_desired_pos[mask] - pred_at_s[mask]
    position_offset = np.median(position_offset_samples, axis=0)
    residual_position_offset = np.linalg.norm(
        position_offset_samples - position_offset[None, :], axis=-1
    )
    native_rotation = quat_to_matrix(native_desired_q)
    pred_rotation_at_s = quat_to_matrix(pred_q_at_s)
    first_valid = int(np.flatnonzero(mask)[0])
    rotation_offset = native_rotation[first_valid] @ pred_rotation_at_s[first_valid].T
    pred_world_pos = pred_at_s + position_offset[None, :]
    pred_world_q = matrix_to_quat(
        np.einsum("ij,njk->nik", rotation_offset, pred_rotation_at_s)
    )

    controller_position = np.asarray(native["achieved_pos"], dtype=np.float64)[mask]
    controller_quaternion = np.asarray(native["achieved_quat"], dtype=np.float64)[mask]
    controller = controller_tracking_error(
        native_desired_pos[mask], controller_position,
        native_desired_q[mask], controller_quaternion,
        metadata={"reference": "native_logged_consumed_reference", "post_reference_idle_excluded": True},
    )
    reconstruction = reference_reconstruction_error(
        pred_world_pos[mask], native_desired_pos[mask],
        pred_world_q[mask], native_desired_q[mask],
        metadata={
            "reference": "decoded_model_to_native_consumed_reference",
            "position_anchor": "logged_additive_reset_anchor_estimate",
            "post_reference_idle_excluded": True,
        },
    )

    progress_fraction = np.clip(native_s / np.maximum(native_l, 1e-12), 0.0, 1.0)
    label_at_progress, label_q_at_progress = _pose_at_arc(
        progress_fraction * float(label_cumulative[-1]),
        label_position,
        label_quaternion,
        label_cumulative,
    )
    label_world_pos = label_at_progress + position_offset[None, :]
    label_world_q = matrix_to_quat(
        np.einsum("ij,njk->nik", rotation_offset, quat_to_matrix(label_q_at_progress))
    )
    end_to_end = end_to_end_task_space_error(
        label_world_pos[mask], controller_position,
        label_world_q[mask], controller_quaternion,
        metadata={
            "label": "future_desired_eef",
            "progress_alignment": "native_traj_s_ref_over_traj_L",
            "post_reference_idle_excluded": True,
        },
    )
    vla = vla_prediction_error(
        label_position,
        pred_position,
        quat_to_matrix(label_quaternion),
        quat_to_matrix(pred_quaternion),
        metadata={"label": "future_desired_eef", "sample_count": 40},
    )

    result: dict[str, Any] = {
        "schema": "STAGE20S_CLOSED_LOOP_ERROR_DECOMPOSITION_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "actual_model_closed_loop": True,
        "model_output": str(args.model_output),
        "native_rollout": str(args.native_rollout),
        "action_sample_period_s": float(args.action_period_s),
        "valid_tracking_samples": int(mask.sum()),
        "four_errors": {
            "VLA_PREDICTION_ERROR": vla,
            "REFERENCE_RECONSTRUCTION_ERROR": reconstruction,
            "CONTROLLER_TRACKING_ERROR": controller,
            "END_TO_END_TASK_SPACE_ERROR": end_to_end,
        },
        "anchor_alignment": {
            "position_offset_world_minus_candidate_m": position_offset.tolist(),
            "position_offset_residual_stats_m": {
                "mean": float(np.mean(residual_position_offset)),
                "p95": float(np.percentile(residual_position_offset, 95)),
                "max": float(np.max(residual_position_offset)),
            },
            "orientation_offset_world_minus_candidate": rotation_offset.tolist(),
            "method": "median logged native desired minus candidate at native arc progress",
        },
        "termination": {
            "reference_complete": bool(np.asarray(native.get("reference_complete", [False]), dtype=bool).any()),
            "termination_code": None,
        },
        "whole_body": None,
        "task_target": {
            "label_final_position_local_m": label_position[-1].tolist(),
            "achieved_final_position_world_m": controller_position[-1].tolist(),
            "label_final_position_world_m": label_world_pos[mask][-1].tolist(),
            "final_position_error_m": float(np.linalg.norm(controller_position[-1] - label_world_pos[mask][-1])),
            "final_rotation_error_rad": float(
                geodesic_angle(
                    quat_to_matrix(controller_quaternion[-1]),
                    quat_to_matrix(label_world_q[mask][-1]),
                )
            ),
        },
    }
    summary_path = args.native_rollout.parent / "native_external_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        result["termination"]["termination_code"] = summary.get("termination_code")
        result["termination"]["native_summary"] = {
            "status": summary.get("status"),
            "steps": summary.get("steps"),
            "reference_mode": summary.get("reference_mode"),
        }
    try:
        from go2arx5_vla.metrics import compute_whole_body_metrics

        result["whole_body"] = compute_whole_body_metrics(native)
    except Exception as exc:  # pragma: no cover - preserve primary decomposition
        result["whole_body"] = {"status": "UNAVAILABLE", "error": str(exc)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
