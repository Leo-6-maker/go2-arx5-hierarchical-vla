#!/usr/bin/env python3
"""Seal one actual Stage20S GR00T output at the simulation interface.

This script is intentionally a non-corrective boundary test.  It decodes no
model and does not run IsaacGym.  It performs:

    decoded [40,10] -> GR00TTrajectoryProvider -> TrajectoryChunkV1
    -> TrajectorySafetyGateV1 -> EpisodeReferenceBindingV1 -> WBC preview

If the safety gate rejects the model output, the script records the rejection
and stops before the preview.  A rejected base-model trajectory is evidence
about that trajectory, not an excuse to clip or re-time it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1
from go2arx5_vla.metrics import reference_reconstruction_error, vla_prediction_error
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
)


SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stats(value: Any) -> dict[str, float]:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _trajectory_audit(
    decoded: np.ndarray, timestamps: np.ndarray, state: np.ndarray
) -> dict[str, Any]:
    position = decoded[:, :3]
    rotation = rot6d_rows_to_matrix(decoded[:, 3:9])
    initial_rotation = rot6d_rows_to_matrix(state[3:9][None, :])[0]
    translation_steps = np.linalg.norm(np.diff(position, axis=0), axis=-1)
    rotation_steps = geodesic_angle(rotation[:-1], rotation[1:])
    dt = np.diff(timestamps)
    linear_speed = translation_steps / dt
    angular_speed = rotation_steps / dt
    tangent = np.diff(position, axis=0)
    curvature = np.zeros(max(0, len(position) - 2), dtype=np.float64)
    if len(position) >= 3:
        a = tangent[:-1]
        b = tangent[1:]
        denominator = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
        curvature = np.linalg.norm(b / np.maximum(dt[1:, None], 1e-12) - a / np.maximum(dt[:-1, None], 1e-12), axis=-1) / denominator
    return {
        "finite": bool(np.isfinite(decoded).all() and np.isfinite(timestamps).all()),
        "point_count": int(len(decoded)),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "initial_position_m": position[0].tolist(),
        "initial_position_discontinuity_m": float(np.linalg.norm(position[0] - state[:3])),
        "initial_rotation_discontinuity_rad": float(
            geodesic_angle(initial_rotation, rotation[0])
        ),
        "translation_path_length_m": float(np.sum(translation_steps)),
        "translation_step_m": _stats(translation_steps),
        "rotation_step_rad": _stats(rotation_steps),
        "linear_speed_mps": _stats(linear_speed),
        "angular_speed_radps": _stats(angular_speed),
        "curvature_proxy": _stats(curvature) if curvature.size else None,
        "workspace_min_m": np.min(position, axis=0).tolist(),
        "workspace_max_m": np.max(position, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--action-period-provenance", default="TEST_ONLY_SIM_VISUAL_REACH_V0")
    parser.add_argument("--episode-id", default="stage20s-actual-groot-visual-reach")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.actual_output, allow_pickle=False) as data:
        decoded_batch = np.asarray(data["decoded_physical_action"], dtype=np.float64)
        state = np.asarray(data["input_state"], dtype=np.float64).reshape(-1)
        desired = np.asarray(data["desired_action"], dtype=np.float64) if "desired_action" in data else None
    if decoded_batch.shape != (1, 40, 10):
        raise ValueError(f"expected decoded_physical_action [1,40,10], got {decoded_batch.shape}")
    if state.shape != (10,):
        raise ValueError(f"expected input_state [10], got {state.shape}")
    decoded = decoded_batch[0]
    timestamps = np.arange(len(decoded), dtype=np.float64) * float(args.action_period_s)
    config = GR00TProviderConfig(
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_ID,
        action_sample_period_s=float(args.action_period_s),
        timestamp_provenance=args.action_period_provenance,
    )
    provider = GR00TTrajectoryProvider(
        decoded,
        config=config,
        observation_timestamp=0.0,
        chunk_id=args.episode_id,
        provenance={
            "stage": "STAGE20S",
            "simulation_only": True,
            "actual_gr00t_output": True,
        },
    )
    chunk = provider.next_chunk(0.0)
    if chunk is None:  # pragma: no cover - defensive
        raise RuntimeError("provider emitted no chunk")
    achieved_orientation = matrix_to_quat(rot6d_rows_to_matrix(state[3:9][None, :])[0])
    gate = TrajectorySafetyGateV1().evaluate(
        chunk,
        now=0.0,
        achieved_position=state[:3],
        achieved_orientation=achieved_orientation,
    )
    bound = None
    preview = None
    gripper = None
    binding = None
    if gate.accepted:
        binding = EpisodeReferenceBindingV1.from_external_chunk(
            chunk, episode_id=args.episode_id
        )
        bound = binding.bind_external_chunk(chunk)
        preview, gripper = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=SOURCE_FRAME,
                wbc_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_ID,
                stale_timeout_s=0.75,
            )
        ).adapt(bound, controller_timestamp=0.0)

    np.savez_compressed(
        args.output_dir / "groot_output_trajectory_candidate.npz",
        timestamp=timestamps,
        position=chunk.eef_position,
        quaternion=chunk.eef_orientation,
        gripper=np.asarray(chunk.gripper, dtype=np.float64),
    )
    audit: dict[str, Any] = {
        "schema": "STAGE20S_ACTUAL_MODEL_INTERFACE_SEAL_V1",
        "status": "PASS" if gate.accepted else "ACTUAL_GR00T_TRAJECTORY_REJECTED_BY_GATE",
        "simulation_only": True,
        "hardware_control": False,
        "actual_wbc_execution": False,
        "source_artifact": str(args.actual_output),
        "input": {
            "decoded_shape": list(decoded.shape),
            "decoded_sha256": _sha(decoded),
            "state_shape": list(state.shape),
            "state": state.tolist(),
            "desired_available": desired is not None,
        },
        "trajectory": _trajectory_audit(decoded, timestamps, state),
        "chain": [
            "actual GR00T decoded physical [40,10]",
            "GR00TTrajectoryProvider",
            "TrajectoryChunkV1",
            "TrajectorySafetyGateV1",
            "EpisodeReferenceBindingV1",
            "GR00TToWBCAdapter",
            "WBCPreviewV1" if gate.accepted else "STOPPED_AT_TRAJECTORY_SAFETY_GATE",
        ],
        "provider": {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "point_timestamps": chunk.point_timestamps.tolist(),
            "source_frame": chunk.source_frame,
            "tcp_semantic_id": chunk.tcp_semantic_id,
            "output_shape": [chunk.point_count, 9],
            "gripper_shape": None if chunk.gripper is None else list(np.asarray(chunk.gripper).shape),
            "timestamp_provenance": args.action_period_provenance,
        },
        "safety_gate": gate.as_dict(),
        "binding": None if binding is None else {
            "episode_id": binding.episode_id,
            "semantics": binding.binding_semantics,
            "pose_preservation_m": float(np.max(np.linalg.norm(bound.eef_position - chunk.eef_position, axis=-1))),
            "rotation_preservation_rad": float(np.max(geodesic_angle(quat_to_matrix(bound.eef_orientation), quat_to_matrix(chunk.eef_orientation)))),
        },
        "preview": None if preview is None else {
            "xyz_rot6d_shape": list(preview.xyz_rot6d.shape),
            "finite": bool(np.isfinite(preview.xyz_rot6d).all()),
            "source_frame": preview.source_frame,
            "tcp_semantic_id": preview.tcp_semantic_id,
            "gripper_parallel_channel": gripper is not None,
            "endpoint_clamped": preview.endpoint_clamped.tolist(),
        },
        "model_vs_label": None,
        "candidate_trajectory": str(args.output_dir / "groot_output_trajectory_candidate.npz"),
        "provenance": {
            "action_sample_period_s": float(args.action_period_s),
            "action_sample_period_provenance": args.action_period_provenance,
            "source_frame": SOURCE_FRAME,
            "tcp_semantic_id": TCP_ID,
            "no_correction_applied": True,
        },
    }
    if desired is not None and desired.shape == (40, 10):
        desired_r = rot6d_rows_to_matrix(desired[:, 3:9])
        pred_r = rot6d_rows_to_matrix(decoded[:, 3:9])
        audit["model_vs_label"] = vla_prediction_error(
            desired[:, :3], decoded[:, :3], desired_r, pred_r,
            metadata={"label": "future_desired_eef", "gripper_separate": True},
        )
        audit["model_vs_label"]["gripper_abs_error"] = _stats(np.abs(desired[:, 9] - decoded[:, 9]))
    identity_r = quat_to_matrix(chunk.eef_orientation)
    audit["reference_reconstruction"] = reference_reconstruction_error(
        chunk.eef_position, chunk.eef_position, identity_r, identity_r,
        metadata={
            "definition": "provider_chunk_round_trip_at_model_grid",
            "expected": "numerical_zero",
            "does_not_measure_unobserved_continuous_truth": True,
        },
    )
    path = args.output_dir / "stage20s_actual_model_interface_seal.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
