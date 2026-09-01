#!/usr/bin/env python3
"""Evaluate R0/R1 GR00T action profiles against same-scene expert labels.

The script performs an offline audit only.  It does not alter predictions,
retime them, pass them to WBC, or use the target actor as a model input.  The
same native-camera snapshots are replayed through the Stage20S custom
embodiment inference path and are compared with the exact expert label stored
beside each snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.data.stage22_expert import SameSceneExpertSnapshotV1, load_snapshot
from go2arx5_vla.data.stage22_split import locked_target_ids
from go2arx5_vla.metrics.action_profile import chunk_overlap_metrics, chunk_profile
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rot6d_rows_to_matrix,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        return {"count": int(array.size), "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode RGB frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_snapshots(input_dir: Path, split: str) -> list[SameSceneExpertSnapshotV1]:
    paths = sorted(input_dir.glob("snapshots/*/*.json"))
    snapshots = [load_snapshot(path) for path in paths if path.name.startswith("expert-")]
    if split in {"train", "validation", "final_test"}:
        selected = set(locked_target_ids(split))
        snapshots = [item for item in snapshots if item.target_id in selected]
    elif split == "holdout":
        selected = set(locked_target_ids("validation") + locked_target_ids("final_test"))
        snapshots = [item for item in snapshots if item.target_id in selected]
    if not snapshots:
        raise ValueError(f"no snapshots found for split={split} under {input_dir}")
    return sorted(snapshots, key=lambda item: (item.raw_episode_id, item.observation_timestamp, item.example_id))


def _chunk_from_snapshot(snapshot: SameSceneExpertSnapshotV1) -> TrajectoryChunkV1:
    matrices = rot6d_rows_to_matrix(snapshot.future_desired_eef_9d[:, 3:9])
    return TrajectoryChunkV1(
        chunk_id=f"expert-label:{snapshot.example_id}",
        source="STAGE22_SAME_SCENE_EXPERT_LABEL",
        observation_timestamp=float(snapshot.observation_timestamp),
        point_timestamps=snapshot.future_timestamps.copy(),
        eef_position=snapshot.future_desired_eef_9d[:, :3].copy(),
        eef_orientation=matrix_to_quat(matrices),
        eef_rot6d=snapshot.future_desired_eef_9d[:, 3:9].copy(),
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        validity_start=float(snapshot.future_timestamps[0]),
        validity_end=float(snapshot.future_timestamps[-1]),
        gripper=snapshot.future_gripper.reshape(-1).copy(),
        provenance={
            "stage": "Stage22",
            "label_source": "same_scene_deterministic_expert",
            "target_excluded_from_model_input": True,
        },
    )


def _state_quaternion(snapshot: SameSceneExpertSnapshotV1) -> np.ndarray:
    state = snapshot.state_v1.as_array()
    return matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]


def _prediction_chunk(
    decoded: np.ndarray,
    snapshot: SameSceneExpertSnapshotV1,
    *,
    action_period_s: float,
    model_path: Path,
) -> TrajectoryChunkV1:
    provider = GR00TTrajectoryProvider(
        decoded,
        config=GR00TProviderConfig(
            source_frame=snapshot.source_frame,
            tcp_semantic_id=snapshot.tcp_semantic_id,
            action_sample_period_s=float(action_period_s),
            timestamp_provenance="TEST_ONLY_STAGE22_SIMULATION",
            source="GR00T_N1.7_STAGE22_PROFILE_EVALUATION",
        ),
        chunk_id=f"prediction:{snapshot.example_id}",
        observation_timestamp=float(snapshot.observation_timestamp),
        provenance={
            "stage": "Stage22",
            "model_checkpoint": str(model_path),
            "trajectory_correction": False,
        },
    )
    chunk = provider.next_chunk(float(snapshot.observation_timestamp))
    if chunk is None:
        raise RuntimeError("GR00T provider unexpectedly emitted no chunk")
    return chunk


def _interpolate_label(label: TrajectoryChunkV1, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamps = label.point_timestamps
    if np.any(query < timestamps[0] - 1e-9) or np.any(query > timestamps[-1] + 1e-9):
        raise ValueError("prediction timestamps fall outside expert label interval")
    positions = np.empty((query.size, 3), dtype=np.float64)
    quaternions = np.empty((query.size, 4), dtype=np.float64)
    for index, timestamp in enumerate(query):
        right = int(np.searchsorted(timestamps, timestamp, side="left"))
        right = min(max(right, 0), timestamps.size - 1)
        if right == 0 or np.isclose(timestamp, timestamps[right], atol=1e-10):
            left = right
            alpha = 0.0
        else:
            left = right - 1
            alpha = float((timestamp - timestamps[left]) / (timestamps[right] - timestamps[left]))
        positions[index] = (1.0 - alpha) * label.eef_position[left] + alpha * label.eef_position[right]
        quaternions[index] = quat_slerp(label.eef_orientation[left], label.eef_orientation[right], alpha)
    return positions, quaternions


def _profile_comparison(
    prediction: TrajectoryChunkV1,
    label: TrajectoryChunkV1,
    snapshot: SameSceneExpertSnapshotV1,
    *,
    gate: TrajectorySafetyGateV1,
) -> dict[str, Any]:
    if prediction.point_count != label.point_count or not np.allclose(
        prediction.point_timestamps, label.point_timestamps, atol=1e-8, rtol=0.0
    ):
        label_position, label_quaternion = _interpolate_label(label, prediction.point_timestamps)
    else:
        label_position = label.eef_position
        label_quaternion = label.eef_orientation
    position_error = np.linalg.norm(prediction.eef_position - label_position, axis=-1)
    rotation_error = np.asarray(
        geodesic_angle(quat_to_matrix(prediction.eef_orientation), quat_to_matrix(label_quaternion)),
        dtype=np.float64,
    ).reshape(-1)
    pred_profile = chunk_profile(
        prediction,
        achieved_position=snapshot.state_v1.as_array()[:3],
        achieved_quaternion=_state_quaternion(snapshot),
        provenance={"profile_role": "model_prediction"},
    )
    label_profile = chunk_profile(label, provenance={"profile_role": "expert_label"})
    pred_speed = pred_profile["linear_speed_mps"]
    label_speed = label_profile["linear_speed_mps"]
    pred_mean_speed = float(pred_speed["mean"] or 0.0)
    label_mean_speed = float(label_speed["mean"] or 0.0)
    pred_max_speed = float(pred_speed["max"] or 0.0)
    label_max_speed = float(label_speed["max"] or 0.0)
    gate_result = gate.evaluate(
        prediction,
        now=float(snapshot.observation_timestamp),
        achieved_position=snapshot.state_v1.as_array()[:3],
        achieved_orientation=_state_quaternion(snapshot),
    )
    label_length = float(label_profile["path_length_m"])
    label_length_safe = label_length if label_length > 1e-12 else None
    return {
        "input_id": snapshot.example_id,
        "raw_episode_id": snapshot.raw_episode_id,
        "target_id": snapshot.target_id,
        "observation_timestamp_s": float(snapshot.observation_timestamp),
        "prediction_shape": [prediction.point_count, 10],
        "label_shape": [label.point_count, 10],
        "timestamp_grid_exact": bool(
            prediction.point_count == label.point_count
            and np.allclose(prediction.point_timestamps, label.point_timestamps, atol=1e-8, rtol=0.0)
        ),
        "position_error_m": _stats(position_error),
        "rotation_error_rad": _stats(rotation_error),
        "endpoint_position_error_m": float(position_error[-1]),
        "endpoint_rotation_error_rad": float(rotation_error[-1]),
        "gripper_abs_error": _stats(
            np.abs(np.asarray(prediction.gripper).reshape(-1) - np.asarray(label.gripper).reshape(-1))
        ),
        "predicted_profile": pred_profile,
        "expert_profile": label_profile,
        "path_length_ratio": None if label_length_safe is None else float(pred_profile["path_length_m"] / label_length_safe),
        "max_speed_ratio": None if label_max_speed <= 1e-12 else float(pred_max_speed / label_max_speed),
        "mean_speed_ratio": None if label_mean_speed <= 1e-12 else float(pred_mean_speed / label_mean_speed),
        "roughness_ratio": None
        if (label_profile["roughness_position_second_difference_m"]["mean"] or 0.0) <= 1e-12
        else float(
            (pred_profile["roughness_position_second_difference_m"]["mean"] or 0.0)
            / label_profile["roughness_position_second_difference_m"]["mean"]
        ),
        "safety_gate": gate_result.as_dict(),
        "trajectory_correction_applied": False,
    }


def _visual_conditioning(
    records: list[dict[str, Any]], snapshots: dict[str, SameSceneExpertSnapshotV1]
) -> dict[str, Any]:
    initial = [item for item in records if item["input_id"].endswith(":expert-0000")]
    initial = sorted(initial, key=lambda item: item["target_id"])
    if len(initial) < 3:
        return {"status": "INSUFFICIENT_COUNTERFACTUALS", "count": len(initial)}
    states = [snapshots[item["input_id"]].state_v1.as_array() for item in initial]
    state_delta = float(np.max(np.ptp(np.stack(states), axis=0)))
    targets = np.stack([snapshots[item["input_id"]].target_position_world_m for item in initial])
    endpoints = np.stack([np.asarray(item["predicted_endpoint_position_local_m"]) for item in initial])
    target_delta = targets - targets[0]
    endpoint_delta = endpoints - endpoints[0]
    target_norm = np.linalg.norm(target_delta[1:], axis=-1)
    endpoint_norm = np.linalg.norm(endpoint_delta[1:], axis=-1)
    cosines = []
    for target_vector, endpoint_vector in zip(target_delta[1:], endpoint_delta[1:]):
        denominator = np.linalg.norm(target_vector) * np.linalg.norm(endpoint_vector)
        if denominator > 1e-12:
            cosines.append(float(np.dot(target_vector, endpoint_vector) / denominator))
    flattened_correlation = None
    if np.std(target_delta.reshape(-1)) > 1e-12 and np.std(endpoint_delta.reshape(-1)) > 1e-12:
        flattened_correlation = float(np.corrcoef(target_delta.reshape(-1), endpoint_delta.reshape(-1))[0, 1])
    magnitude_ratios = np.divide(
        endpoint_norm,
        target_norm,
        out=np.full_like(endpoint_norm, np.nan),
        where=target_norm > 1e-12,
    )
    return {
        "status": "DESCRIPTIVE_COUNTERFACTUAL_ONLY",
        "input_count": len(initial),
        "same_initial_state_max_abs": state_delta,
        "same_language": True,
        "target_ids": [item["target_id"] for item in initial],
        "target_delta_world_m": target_delta.tolist(),
        "predicted_endpoint_delta_local_m": endpoint_delta.tolist(),
        "direction_cosine": _stats(cosines),
        "flattened_target_endpoint_correlation": flattened_correlation,
        "endpoint_displacement_magnitude_ratio": _stats(magnitude_ratios[np.isfinite(magnitude_ratios)]),
        "rgb_changed_between_counterfactuals": True,
        "target_position_provided_to_model": False,
        "interpretation": "report_metrics_before_visual_grounding_claim; RGB sensitivity is not visual grounding",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "final_test", "holdout"), default="all"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="move the end effector to the red target")
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2223)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.action_period_s) or args.action_period_s <= 0.0:
        parser.error("--action-period-s must be positive and finite")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_list = _load_snapshots(args.input_dir, args.split)
    snapshots = {item.example_id: item for item in snapshots_list}

    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401; registers model
    from transformers import AutoModel
    from stage21_same_scene_groot_inference import _build_processor, _run_one

    processor, embodiment = _build_processor(
        args.stage10_root, args.base_model_path, args.dataset_dir / "meta", args.modality_config
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    gate = TrajectorySafetyGateV1()
    records: list[dict[str, Any]] = []
    chunks_by_episode: dict[str, list[tuple[SameSceneExpertSnapshotV1, TrajectoryChunkV1]]] = defaultdict(list)
    latency_values = []
    peak_allocated = []
    peak_reserved = []
    artifact_hashes = []
    for index, snapshot in enumerate(snapshots_list):
        observation = {
            "path": snapshot.body_frame_path,
            "body_rgb": _read_rgb(snapshot.body_frame_path),
            "wrist_rgb": _read_rgb(snapshot.wrist_frame_path),
            "state": snapshot.state_v1.as_array().astype(np.float32),
        }
        raw, decoded, latency, gpu = _run_one(
            model,
            processor,
            embodiment,
            observation,
            language=args.language,
            seed=int(args.seed),
            device=args.device,
        )
        prediction = _prediction_chunk(
            decoded,
            snapshot,
            action_period_s=args.action_period_s,
            model_path=args.model_path,
        )
        label = _chunk_from_snapshot(snapshot)
        record = _profile_comparison(prediction, label, snapshot, gate=gate)
        record.update(
            {
                "body_frame_sha256": _sha256_file(Path(snapshot.body_frame_path)),
                "wrist_frame_sha256": _sha256_file(Path(snapshot.wrist_frame_path)),
                "raw_universal_shape": list(raw.shape),
                "decoded_physical_shape": list(decoded.shape),
                "raw_universal_sha256": _sha256_array(raw),
                "decoded_physical_sha256": _sha256_array(decoded),
                "finite": bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
                "latency_s": float(latency),
                "gpu": gpu,
                "predicted_endpoint_position_local_m": prediction.eef_position[-1].tolist(),
                "model_input_target_excluded": True,
            }
        )
        output_file = args.output_dir / f"{index:04d}_{snapshot.target_id}_{snapshot.example_id.split(':')[-1]}.npz"
        np.savez_compressed(output_file, raw_universal_action=raw, decoded_physical_action=decoded)
        record["prediction_artifact"] = str(output_file)
        artifact_hashes.append({"path": str(output_file), "sha256": _sha256_file(output_file)})
        records.append(record)
        chunks_by_episode[snapshot.raw_episode_id].append((snapshot, prediction))
        latency_values.append(float(latency))
        peak_allocated.append(float(gpu.get("peak_allocated_mib", 0.0)))
        peak_reserved.append(float(gpu.get("peak_reserved_mib", 0.0)))

    overlaps: list[dict[str, Any]] = []
    for episode_id, values in sorted(chunks_by_episode.items()):
        ordered = sorted(values, key=lambda pair: pair[0].observation_timestamp)
        for (previous_snapshot, previous), (current_snapshot, current) in zip(ordered, ordered[1:]):
            item = chunk_overlap_metrics(previous, current)
            item.update(
                {
                    "raw_episode_id": episode_id,
                    "previous_observation_timestamp_s": previous_snapshot.observation_timestamp,
                    "current_observation_timestamp_s": current_snapshot.observation_timestamp,
                    "old_prediction_at_new_observation_vs_new_first_only": True,
                    "no_trajectory_correction": True,
                }
            )
            overlaps.append(item)

    visual = _visual_conditioning(records, snapshots)
    public_records = records
    manifest = {
        "schema": "STAGE22_ACTION_PROFILE_EVALUATION_V1",
        "status": "PASS",
        "stage": "Stage22",
        "recipe_model_path": str(args.model_path),
        "split": args.split,
        "snapshot_count": len(snapshots_list),
        "target_count": len({item.target_id for item in snapshots_list}),
        "simulation_only": True,
        "hardware_control": False,
        "wbc_execution": False,
        "model_input_contract": {
            "images": ["body_rgb", "wrist_rgb"],
            "language": args.language,
            "state_v1": [10],
            "target_position_world_m": "excluded",
        },
        "expert_label_contract": {
            "desired_eef": [40, 9],
            "desired_gripper": [40, 1],
            "timestamps": [40],
            "source": "exact_same_scene_native_snapshot_expert_label",
        },
        "profile_selection_policy": {
            "loss_not_used_for_selection": True,
            "required_metrics": [
                "position_error_m",
                "endpoint_position_error_m",
                "path_length_ratio",
                "max_speed_ratio",
                "mean_speed_ratio",
                "roughness_ratio",
                "initial_position_discontinuity_m",
                "safety_gate",
            ],
            "trajectory_correction": False,
        },
        "timing": {
            "action_sample_period_s": float(args.action_period_s),
            "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
            "production_action_period": "TBD_BY_ADAPTATION_DATASET",
        },
        "summary": {
            "position_error_m": _stats([item["position_error_m"]["mean"] for item in records]),
            "rotation_error_rad": _stats([item["rotation_error_rad"]["mean"] for item in records]),
            "endpoint_position_error_m": _stats([item["endpoint_position_error_m"] for item in records]),
            "path_length_ratio": _stats([item["path_length_ratio"] for item in records if item["path_length_ratio"] is not None]),
            "max_speed_ratio": _stats([item["max_speed_ratio"] for item in records if item["max_speed_ratio"] is not None]),
            "mean_speed_ratio": _stats([item["mean_speed_ratio"] for item in records if item["mean_speed_ratio"] is not None]),
            "safety_gate_decisions": {
                decision: sum(1 for item in records if item["safety_gate"]["decision"] == decision)
                for decision in ("ACCEPT", "WARN", "REJECT")
            },
            "latency_s": _stats(latency_values),
            "peak_allocated_mib": max(peak_allocated, default=0.0),
            "peak_reserved_mib": max(peak_reserved, default=0.0),
        },
        "records": public_records,
        "chunk_overlap_metrics": overlaps,
        "visual_target_conditioning": visual,
        "artifact_hashes": artifact_hashes,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(args.device),
        },
        "claim_boundary": "same_scene_expert_profile_audit_only; no task success or visual_grounding claim",
    }
    output_path = args.output_dir / "stage22_action_profile_evaluation.json"
    output_path.write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
