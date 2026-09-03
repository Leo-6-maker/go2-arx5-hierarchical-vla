#!/usr/bin/env python3
"""Recompute sealed BASE/O1 temporal-prefix metrics without model execution.

The script consumes only the immutable Stage23 Phase-A prediction artifacts
and the frozen TINY1 expert label.  It truncates each existing H=40 output to
H in {4, 8, 12, 16, 24, 32, 40}; it does not smooth, clip, retime, or alter
any prediction.  No GR00T model, CUDA device, WBC, or training path is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HORIZONS = (4, 8, 12, 16, 24, 32, 40)
SEEDS = (2223, 2224, 2225)
LINEAR_SPEED_REJECT_MPS = 2.0
EXPECTED_ARCHIVE_SHA256 = "720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd"
EXPECTED_PAYLOAD_MANIFEST_SHA256 = "dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return {"count": int(array.size), "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"rot6d must have shape [N,6], got {value.shape}")
    first = value[:, :3]
    second = value[:, 3:]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm < 1e-10):
        raise ValueError("rot6d has a degenerate first row")
    first = first / first_norm
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm < 1e-10):
        raise ValueError("rot6d has a degenerate second row")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _geodesic_rotation_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(predicted, -1, -2), target)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _profile(position: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    if len(position) < 2:
        raise ValueError("prefix requires at least two points")
    dt = np.diff(timestamps)
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    speeds = steps / dt
    roughness = np.linalg.norm(np.diff(position, n=2, axis=0), axis=1)
    path_length = float(np.sum(steps))
    return {
        "path_length_m": path_length,
        "translation_step_m": _stats(steps),
        "linear_speed_mps": _stats(speeds),
        "roughness_position_second_difference_m": _stats(roughness),
        "max_linear_speed_mps": float(np.max(speeds)),
        "duration_s": float(timestamps[-1] - timestamps[0]),
    }


def _prefix_metrics(
    predicted: np.ndarray,
    expert: np.ndarray,
    timestamps: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    pred = np.asarray(predicted[:horizon], dtype=np.float64)
    label = np.asarray(expert[:horizon], dtype=np.float64)
    ts = np.asarray(timestamps[:horizon], dtype=np.float64)
    pred_position = pred[:, :3]
    label_position = label[:, :3]
    pred_rotation = _rot6d_to_matrix(pred[:, 3:9])
    label_rotation = _rot6d_to_matrix(label[:, 3:9])
    position_error = np.linalg.norm(pred_position - label_position, axis=1)
    rotation_error = _geodesic_rotation_error(pred_rotation, label_rotation)
    pred_profile = _profile(pred_position, ts)
    label_profile = _profile(label_position, ts)
    pred_path = float(pred_profile["path_length_m"])
    label_path = float(label_profile["path_length_m"])
    pred_roughness = float(pred_profile["roughness_position_second_difference_m"]["mean"] or 0.0)
    label_roughness = float(label_profile["roughness_position_second_difference_m"]["mean"] or 0.0)
    label_max_speed = float(label_profile["max_linear_speed_mps"])
    max_speed_ratio = float(pred_profile["max_linear_speed_mps"] / label_max_speed) if label_max_speed > 0.0 else None
    return {
        "horizon": horizon,
        "point_count": int(horizon),
        "finite": bool(np.isfinite(pred).all() and np.isfinite(label).all()),
        "position_error_m": _stats(position_error),
        "rotation_error_rad": _stats(rotation_error),
        "endpoint_position_error_m": float(position_error[-1]),
        "endpoint_rotation_error_rad": float(rotation_error[-1]),
        "initial_position_discontinuity_m": float(np.linalg.norm(pred_position[0] - label_position[0])),
        "initial_rotation_discontinuity_rad": float(rotation_error[0]),
        "predicted_profile": pred_profile,
        "expert_profile": label_profile,
        "path_length_ratio": float(pred_path / label_path) if label_path > 1e-12 else None,
        "mean_speed_ratio": float(pred_path / label_path) if label_path > 1e-12 else None,
        "max_speed_ratio": max_speed_ratio,
        "roughness_ratio": float(pred_roughness / label_roughness) if label_roughness > 1e-12 else None,
        "linear_speed_gate": {
            "decision": "REJECT" if pred_profile["max_linear_speed_mps"] > LINEAR_SPEED_REJECT_MPS else "ACCEPT",
            "observed_max_mps": pred_profile["max_linear_speed_mps"],
            "threshold_mps": LINEAR_SPEED_REJECT_MPS,
            "threshold_provenance": "TEST_ONLY",
        },
        "trajectory_correction_applied": False,
    }


def _read_expert(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(path).to_pydict()
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    actions = np.asarray(table["action"], dtype=np.float64)
    if timestamps.shape != (40,) or actions.shape != (40, 10):
        raise ValueError(f"unexpected expert shapes: {timestamps.shape}, {actions.shape}")
    if np.any(~np.isfinite(timestamps)) or np.any(~np.isfinite(actions)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("expert timestamps/actions are invalid")
    return timestamps, actions


def _load_expected_hashes(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = root / "evaluations" / "prediction_tensor_hashes.tsv"
    result: dict[tuple[str, int], dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            result[(row["evaluation_label"], int(row["seed"]))] = row
    return result


def _load_artifact(path: Path, expected: dict[str, str] | None) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    raw = np.asarray(data["raw_universal_action"])
    decoded = np.asarray(data["decoded_physical_action"])
    if raw.shape != (1, 40, 132) or decoded.shape != (1, 40, 10):
        raise ValueError(f"unexpected artifact shapes in {path}: {raw.shape}, {decoded.shape}")
    raw_hash = sha256_array(raw)
    decoded_hash = sha256_array(decoded)
    npz_hash = sha256_file(path)
    if expected is not None:
        for key, actual in (("raw_tensor_sha256", raw_hash), ("decoded_tensor_sha256", decoded_hash), ("npz_sha256", npz_hash)):
            if expected.get(key) != actual:
                raise ValueError(f"sealed {key} mismatch for {path}: {actual} != {expected.get(key)}")
    return decoded[0].astype(np.float64), {
        "path": str(path),
        "npz_sha256": npz_hash,
        "raw_tensor_sha256": raw_hash,
        "decoded_tensor_sha256": decoded_hash,
        "raw_shape": list(raw.shape),
        "decoded_shape": list(decoded.shape),
        "finite": bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "endpoint_position_error_m",
        "endpoint_rotation_error_rad",
        "path_length_ratio",
        "mean_speed_ratio",
        "max_speed_ratio",
        "roughness_ratio",
        "initial_position_discontinuity_m",
        "initial_rotation_discontinuity_rad",
    )
    aggregate: dict[str, Any] = {name: _stats([record[name] for record in records if record[name] is not None]) for name in scalar_names}
    aggregate["position_error_m"] = _stats([value for record in records for value in [record["position_error_m"]["mean"]]])
    aggregate["rotation_error_rad"] = _stats([value for record in records for value in [record["rotation_error_rad"]["mean"]]])
    aggregate["linear_speed_gate_counts"] = {
        decision: sum(record["linear_speed_gate"]["decision"] == decision for record in records)
        for decision in ("ACCEPT", "REJECT")
    }
    aggregate["all_finite"] = bool(all(record["finite"] for record in records))
    return aggregate


def audit(evidence_root: Path, dataset_dir: Path, bundle_archive: Path | None) -> dict[str, Any]:
    timestamps, expert = _read_expert(dataset_dir)
    expected_hashes = _load_expected_hashes(evidence_root)
    experiments: dict[str, Path] = {
        "BASE": evidence_root / "base_evaluations" / "597965.base_raw",
        "O1_10": evidence_root / "evaluations" / "candidate_10",
        "O1_25": evidence_root / "evaluations" / "candidate_25",
        "O1_50": evidence_root / "evaluations" / "candidate_50",
        "O1_100": evidence_root / "evaluations" / "candidate_100",
        "O1_200": evidence_root / "evaluations" / "candidate_200",
    }
    records: list[dict[str, Any]] = []
    for experiment, directory in experiments.items():
        label = "BASE" if experiment == "BASE" else f"candidate_{experiment.split('_')[1]}"
        for seed in SEEDS:
            path = directory / f"seed_{seed}.npz"
            decoded, artifact = _load_artifact(path, expected_hashes.get((label, seed)))
            for horizon in HORIZONS:
                metrics = _prefix_metrics(decoded, expert, timestamps, horizon)
                records.append({"experiment": experiment, "seed": seed, "artifact": artifact, **metrics})
    by_experiment: dict[str, Any] = {}
    for experiment in experiments:
        experiment_records = [record for record in records if record["experiment"] == experiment]
        by_experiment[experiment] = {
            "seed_count": len({record["seed"] for record in experiment_records}),
            "horizons": {
                str(horizon): _aggregate([record for record in experiment_records if record["horizon"] == horizon])
                for horizon in HORIZONS
            },
        }

    # The second BASE evaluation in the sealed bundle is an intentional
    # duplicate provenance path.  Verify its raw files are byte-identical to
    # the canonical BASE path before making any prefix comparison.
    base_duplicate_checks = []
    for seed in SEEDS:
        first = evidence_root / "base_evaluations" / "597965.base_raw" / f"seed_{seed}.npz"
        second = evidence_root / "base_evaluations" / "597966.base_raw" / f"seed_{seed}.npz"
        first_hash = sha256_file(first)
        second_hash = sha256_file(second)
        base_duplicate_checks.append({"seed": seed, "597965_sha256": first_hash, "597966_sha256": second_hash, "identical": first_hash == second_hash})

    payload_manifest = evidence_root / "payload_file_manifest.tsv"
    return {
        "schema": "STAGE24_PREFIX_AUDIT_V1",
        "stage": "Stage24-A2 sealed O1 temporal-prefix audit",
        "source": {
            "evidence_root": str(evidence_root),
            "bundle_archive": str(bundle_archive) if bundle_archive else None,
            "bundle_archive_sha256": sha256_file(bundle_archive) if bundle_archive else None,
            "expected_bundle_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "payload_manifest_sha256": sha256_file(payload_manifest),
            "expected_payload_manifest_sha256": EXPECTED_PAYLOAD_MANIFEST_SHA256,
            "timestamps_source": str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"),
            "timestamp_period_s": _stats(np.diff(timestamps)),
            "expert_shape": list(expert.shape),
            "expert_action_label": "desired EEF XYZ+rot6d+gripper; achieved EEF not used",
        },
        "contract": {
            "horizons": list(HORIZONS),
            "seeds": list(SEEDS),
            "full_prediction_shape": [1, 40, 132],
            "decoded_shape": [1, 40, 10],
            "action_period_s": 0.05,
            "action_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
            "linear_speed_reject_mps": LINEAR_SPEED_REJECT_MPS,
            "correction": "NONE",
        },
        "base_duplicate_checks": base_duplicate_checks,
        "experiments": by_experiment,
        "records": records,
        "interpretation": {
            "prefixes_are_truncated_views_of_sealed_outputs": True,
            "no_new_inference": True,
            "no_model_execution": True,
            "no_wbc": True,
            "no_claim_of_causal_horizon_effect": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--bundle-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.evidence_root, args.dataset_dir, args.bundle_archive)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "records": len(payload["records"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
