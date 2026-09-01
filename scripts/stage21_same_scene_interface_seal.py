#!/usr/bin/env python3
"""Seal actual same-scene GR00T chunks before any native WBC execution.

This is a non-corrective audit.  It preserves the model output byte-for-byte,
builds the canonical chunk, evaluates the existing safety gate, and only for
ACCEPT/WARN chunks constructs the ordinary binding and WBC preview.  It never
steps IsaacGym and never changes timing, poses, or gripper values.
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
from go2arx5_vla.data import StateV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix, rot6d_rows_to_matrix


SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stats(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(array)) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "p95": float(np.percentile(array, 95.0)) if array.size else 0.0,
        "max": float(np.max(array)) if array.size else 0.0,
    }


def _profile(decoded: np.ndarray, timestamps: np.ndarray, state: StateV1) -> dict[str, Any]:
    positions = np.asarray(decoded[:, :3], dtype=np.float64)
    matrices = rot6d_rows_to_matrix(np.asarray(decoded[:, 3:9], dtype=np.float64))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    rotation_steps = geodesic_angle(matrices[:-1], matrices[1:])
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    linear_speed = steps / np.maximum(dt, 1e-12)
    angular_speed = rotation_steps / np.maximum(dt, 1e-12)
    second_difference = np.diff(positions, n=2, axis=0)
    return {
        "finite": bool(np.isfinite(decoded).all() and np.isfinite(timestamps).all()),
        "point_count": int(len(decoded)),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "initial_position_discontinuity_m": float(np.linalg.norm(positions[0] - state.eef_xyz)),
        "initial_rotation_discontinuity_rad": float(
            geodesic_angle(rot6d_rows_to_matrix(state.eef_rot6d_rows[None, :])[0], matrices[0])
        ),
        "translation_path_length_m": float(np.sum(steps)),
        "translation_step_m": _stats(steps),
        "rotation_step_rad": _stats(rotation_steps),
        "linear_speed_mps": _stats(linear_speed),
        "angular_speed_radps": _stats(angular_speed),
        "roughness_position_second_difference_m": _stats(np.linalg.norm(second_difference, axis=-1)),
        "workspace_min_m": np.min(positions, axis=0).tolist(),
        "workspace_max_m": np.max(positions, axis=0).tolist(),
        "gt_comparison": "NOT_AVAILABLE_FOR_SAME_SCENE_CAMERA_SMOKE",
    }


def _load_model_case(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(item["artifact"], allow_pickle=False) as data:
        decoded_batch = np.asarray(data["decoded_physical_action"], dtype=np.float64)
        state_array = np.asarray(data["input_state"], dtype=np.float64).reshape(-1)
    if decoded_batch.shape != (1, 40, 10):
        raise ValueError(f"{item['input_id']} decoded output must be [1,40,10], got {decoded_batch.shape}")
    state = StateV1.from_array(state_array)
    decoded = decoded_batch[0]
    if not np.isfinite(decoded).all():
        raise ValueError(f"{item['input_id']} decoded output is not finite")
    timestamps = np.arange(40, dtype=np.float64) * 0.05
    target = None if item.get("target_position_world_m") is None else np.asarray(item["target_position_world_m"], dtype=np.float64)
    return decoded, state_array, timestamps, state, target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--action-period-provenance", default="TEST_ONLY_STAGE21_SIMULATION")
    args = parser.parse_args()
    if not np.isfinite(args.action_period_s) or args.action_period_s <= 0.0:
        parser.error("--action-period-s must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.inference_manifest.read_text())
    if manifest.get("schema") != "STAGE21_SAME_SCENE_GR00T_INFERENCE_V1":
        raise ValueError("unexpected Stage21 inference manifest schema")

    gate = TrajectorySafetyGateV1()
    adapter = GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame=SOURCE_FRAME,
            wbc_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_ID,
            stale_timeout_s=0.75,
        )
    )
    cases: list[dict[str, Any]] = []
    for item in manifest["inputs"]:
        decoded, state_array, _, state, target = _load_model_case(item)
        timestamps = np.arange(len(decoded), dtype=np.float64) * float(args.action_period_s)
        provider = GR00TTrajectoryProvider(
            decoded,
            config=GR00TProviderConfig(
                source_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_ID,
                action_sample_period_s=float(args.action_period_s),
                timestamp_provenance=args.action_period_provenance,
            ),
            observation_timestamp=0.0,
            chunk_id=item["input_id"],
            provenance={
                "stage": "Stage21",
                "same_scene_observation": item["input_path"],
                "actual_gr00t_output": True,
                "no_correction_applied": True,
            },
        )
        chunk = provider.next_chunk(0.0)
        if chunk is None:  # pragma: no cover
            raise RuntimeError("provider emitted no chunk")
        achieved_q = StateV1.from_array(state_array)
        achieved_orientation = rot6d_rows_to_matrix(achieved_q.eef_rot6d_rows[None, :])[0]
        from go2arx5_vla.trajectory.so3 import matrix_to_quat

        achieved_quaternion = matrix_to_quat(achieved_orientation)
        decision = gate.evaluate(
            chunk,
            now=0.0,
            achieved_position=state.eef_xyz,
            achieved_orientation=achieved_quaternion,
        )
        binding = None
        preview = None
        gripper = None
        if decision.accepted:
            binding = EpisodeReferenceBindingV1.from_external_chunk(chunk, episode_id=f"stage21-{item['input_id']}")
            bound = binding.bind_external_chunk(chunk)
            preview, gripper = adapter.adapt(bound, controller_timestamp=0.0)
        candidate_path = args.output_dir / f"{item['input_id']}_trajectory_candidate.npz"
        np.savez_compressed(
            candidate_path,
            timestamp=timestamps,
            position=chunk.eef_position,
            quaternion=chunk.eef_orientation,
            gripper=np.asarray(chunk.gripper, dtype=np.float64),
        )
        cases.append(
            {
                "input_id": item["input_id"],
                "input_path": item["input_path"],
                "target_position_world_m": None if target is None else target.tolist(),
                "input_state": state_array.tolist(),
                "decoded_sha256": _sha(decoded),
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "point_count": chunk.point_count,
                    "eef_shape": list(chunk.eef_position.shape),
                    "gripper_shape": None if chunk.gripper is None else list(np.asarray(chunk.gripper).shape),
                    "point_timestamps": chunk.point_timestamps.tolist(),
                    "source_frame": chunk.source_frame,
                    "tcp_semantic_id": chunk.tcp_semantic_id,
                    "finite": bool(np.isfinite(chunk.eef_position).all() and np.isfinite(chunk.eef_orientation).all()),
                    "no_correction_applied": True,
                },
                "profile": _profile(decoded, timestamps, state),
                "safety_gate": decision.as_dict(),
                "binding": None
                if binding is None
                else {
                    "episode_id": binding.episode_id,
                    "semantics": binding.binding_semantics,
                    "pose_preservation_position_m": float(np.max(np.linalg.norm(bound.eef_position - chunk.eef_position, axis=-1))),
                    "pose_preservation_rotation_rad": float(
                        np.max(geodesic_angle(quat_to_matrix(bound.eef_orientation), quat_to_matrix(chunk.eef_orientation)))
                    ),
                },
                "preview": None
                if preview is None
                else {
                    "shape": list(preview.xyz_rot6d.shape),
                    "finite": bool(np.isfinite(preview.xyz_rot6d).all()),
                    "source_frame": preview.source_frame,
                    "tcp_semantic_id": preview.tcp_semantic_id,
                    "gripper_parallel_channel": gripper is not None,
                },
                "candidate_trajectory": str(candidate_path),
                "execution": "NOT_EXECUTED_IN_INTERFACE_SEAL",
            }
        )

    rejected = [case["input_id"] for case in cases if case["safety_gate"]["decision"] == "REJECT"]
    warnings = [case["input_id"] for case in cases if case["safety_gate"]["decision"] == "WARN"]
    result = {
        "schema": "STAGE21_SAME_SCENE_MODEL_INTERFACE_SEAL_V1",
        "status": "PASS" if not rejected else "ACTUAL_GR00T_TRAJECTORY_REJECTED_BY_GATE",
        "simulation_only": True,
        "hardware_control": False,
        "actual_wbc_execution": False,
        "chain": [
            "native same-scene RGB + StateV1",
            "actual GR00T [1,40,132]",
            "GR00TTrajectoryProvider",
            "TrajectoryChunkV1",
            "TrajectorySafetyGateV1",
            "EpisodeReferenceBindingV1",
            "GR00TToWBCAdapter",
            "WBCPreviewV1",
        ],
        "gate_summary": {"case_count": len(cases), "rejected": rejected, "warned": warnings},
        "action_sample_period_s": float(args.action_period_s),
        "action_sample_period_provenance": args.action_period_provenance,
        "production_action_period": "TBD_BY_ADAPTATION_DATASET",
        "cases": cases,
        "claim_boundary": "full_chunk_interface_and_profile_audit_only; no actual_model_task_success_claim",
    }
    output = args.output_dir / "same_scene_model_interface_seal.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
