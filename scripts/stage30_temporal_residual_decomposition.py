#!/usr/bin/env python3
"""Zero-model Stage30 decomposition of the six sealed Stage29 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "STAGE30_TEMPORAL_RESIDUAL_DECOMPOSITION_V1"
PI_GATE = "PI_GATE_STATE_0010"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE30_TEMPORAL_RESIDUAL_RESULT"
STAGE29_EXECUTION_SHA = "07290d042c24996b93d046edf514f91a786f52ba"
STAGE29_CLOSURE_SHA = "ae10dbd44442bafd02bbe927f95ae271526e4970"
STAGE29_SEAL_SHA256 = "9c9c25e9106c4a3538d160e354a013fdf6fd4a5300b519ab8de6d98a8865247b"
STAGE29_RESULT_SHA256 = "1e800b69604a97f16cfc704b591ff2c710c14430282cbd1b0cc6e6da0c5dfd48"
STAGE29_PARTIAL_SHA256 = "3835e4a4728be6c8eba147c4299bf59883e9ac42aefe6c3a77f370d8e4fde6bf"
SEEDS = (2223, 2224, 2225)
CHECKPOINTS = ("BASE", "H2@2000")
HORIZON = 40
ACTIVE_DIM = 10
UNIVERSAL_DIM = 132
ACTION_PERIOD_S = 0.05
NEAR_CLIP_BOUND = 0.99
AXIS_NAMES = ("x", "y", "z")
ACTIVE_DIM_NAMES = (
    "x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5", "gripper",
)
EXPECTED_NPZ_NAMES = {
    (2223, "BASE"): "seed_2223_base_sampler_output.npz",
    (2223, "H2@2000"): "seed_2223_h2_2000_sampler_output.npz",
    (2224, "BASE"): "seed_2224_base_sampler_output.npz",
    (2224, "H2@2000"): "seed_2224_h2_2000_sampler_output.npz",
    (2225, "BASE"): "seed_2225_base_sampler_output.npz",
    (2225, "H2@2000"): "seed_2225_h2_2000_sampler_output.npz",
}
EXPECTED_ARRAY_KEYS = {
    "initial_noise_float32",
    "raw_universal_action",
    "decoded_physical_action",
    "expert_physical_action",
}


class Stage30Error(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise Stage30Error(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _git_bytes(repo: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative_path}"])


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(rows),
        "min": min(rows),
        "median": median(rows),
        "mean": mean(rows),
        "max": max(rows),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def error_metrics(error: np.ndarray) -> dict[str, Any]:
    array = np.asarray(error, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise Stage30Error(f"invalid XYZ error array: {array.shape}")
    mse = float(np.mean(np.square(array)))
    per_axis = []
    for axis, name in enumerate(AXIS_NAMES):
        axis_mse = float(np.mean(np.square(array[..., axis])))
        per_axis.append({"axis": name, "mse": axis_mse, "rms": math.sqrt(axis_mse)})
    return {"mse": mse, "rms": math.sqrt(mse), "per_axis": per_axis}


def per_horizon_error(error: np.ndarray, start_index: int) -> list[dict[str, Any]]:
    array = np.asarray(error, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != len(SEEDS) or array.shape[-1] != 3:
        raise Stage30Error(f"invalid aggregate horizon error array: {array.shape}")
    rows = []
    for local_index in range(array.shape[1]):
        values = array[:, local_index, :]
        rows.append({
            "horizon_index": local_index + start_index,
            "xyz_rmse": float(np.sqrt(np.mean(np.square(values)))),
            "per_axis_rmse": {
                name: float(np.sqrt(np.mean(np.square(values[:, axis]))))
                for axis, name in enumerate(AXIS_NAMES)
            },
            "mean_residual_xyz": np.mean(values, axis=0).astype(float).tolist(),
        })
    return rows


def _band_payload(energy: np.ndarray, period_s: float) -> dict[str, Any]:
    if energy.ndim != 2:
        raise Stage30Error("DFT energy must be [frequency, axis]")
    n = energy.shape[0]
    bins = np.arange(n)
    folded = np.minimum(bins, n - bins)
    very_low_cut = int(math.ceil(n / 8.0))
    high_cut = int(math.ceil(n / 4.0))
    masks = {
        "dc": folded == 0,
        "very_low_non_dc": (folded >= 1) & (folded < very_low_cut),
        "mid_low": (folded >= very_low_cut) & (folded < high_cut),
        "low_including_dc": folded < high_cut,
        "high": folded >= high_cut,
    }

    def describe(values: np.ndarray) -> dict[str, Any]:
        total = float(np.sum(values))
        bands = {name: float(np.sum(values[mask])) for name, mask in masks.items()}
        return {
            "total_energy": total,
            "dc_energy": bands["dc"],
            "very_low_non_dc_energy": bands["very_low_non_dc"],
            "mid_low_energy": bands["mid_low"],
            "low_including_dc_energy": bands["low_including_dc"],
            "high_energy": bands["high"],
            "dc_fraction": None if total <= 0.0 else bands["dc"] / total,
            "very_low_non_dc_fraction": None if total <= 0.0 else bands["very_low_non_dc"] / total,
            "low_including_dc_fraction": None if total <= 0.0 else bands["low_including_dc"] / total,
            "high_fraction": None if total <= 0.0 else bands["high"] / total,
            "high_over_low_energy": (
                None if bands["low_including_dc"] <= 0.0
                else bands["high"] / bands["low_including_dc"]
            ),
        }

    aggregate = describe(np.sum(energy, axis=1))
    per_axis = []
    for axis, name in enumerate(AXIS_NAMES):
        per_axis.append({"axis": name, **describe(energy[:, axis])})
    return {
        "sample_count": n,
        "period_s": period_s,
        "very_low_cut_bin": very_low_cut,
        "very_low_cut_hz": very_low_cut / (n * period_s),
        "high_cut_bin": high_cut,
        "high_cut_hz": high_cut / (n * period_s),
        "aggregate_xyz": aggregate,
        "per_axis": per_axis,
    }


def spectral_decomposition(series: np.ndarray, period_s: float = ACTION_PERIOD_S) -> dict[str, Any]:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise Stage30Error(f"invalid spectral series: {array.shape}")
    energy = np.square(np.abs(np.fft.fft(array, axis=0)))
    return _band_payload(energy, period_s)


def aggregate_spectral_decomposition(
    series_by_seed: Sequence[np.ndarray], period_s: float = ACTION_PERIOD_S
) -> dict[str, Any]:
    arrays = [np.asarray(series, dtype=np.float64) for series in series_by_seed]
    if len(arrays) != len(SEEDS) or len({array.shape for array in arrays}) != 1:
        raise Stage30Error("aggregate spectral inputs must be three equal-shaped seed arrays")
    energy = sum((np.square(np.abs(np.fft.fft(array, axis=0))) for array in arrays), np.zeros_like(arrays[0]))
    return {"aggregation": "sum_unscaled_dft_energy_across_three_seeds", **_band_payload(energy, period_s)}


def _stage23_high_fraction(series: np.ndarray) -> float | None:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise Stage30Error("Stage23 DFT input must be [N,3]")
    centered = array - np.mean(array, axis=0, keepdims=True)
    energy = np.sum(np.square(np.abs(np.fft.fft(centered, axis=0))), axis=1)
    n = array.shape[0]
    folded = np.minimum(np.arange(n), n - np.arange(n))
    high = float(np.sum(energy[folded >= int(math.ceil(n / 4.0))]))
    total = float(np.sum(energy))
    return None if total <= 0.0 else high / total


def _adjacent_cosines(vectors: np.ndarray, epsilon: float = 1e-15) -> list[float]:
    rows = np.asarray(vectors, dtype=np.float64)
    values = []
    for left, right in zip(rows[:-1], rows[1:]):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > epsilon:
            values.append(float(np.dot(left, right) / denominator))
    return values


def trajectory_profile(xyz: np.ndarray) -> dict[str, Any]:
    positions = np.asarray(xyz, dtype=np.float64)
    if positions.shape != (HORIZON, 3) or not np.isfinite(positions).all():
        raise Stage30Error(f"invalid trajectory profile shape: {positions.shape}")
    first = np.diff(positions, axis=0)
    second = np.diff(positions, n=2, axis=0)
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    cosines = _adjacent_cosines(first)
    return {
        "path_length_m": float(np.sum(first_norm)),
        "linear_speed_mps": _summary(first_norm / ACTION_PERIOD_S),
        "roughness_second_difference_m": _summary(second_norm),
        "adjacent_increment_cosine": _summary(cosines),
        "back_and_forth_rate": None if not cosines else sum(value < 0.0 for value in cosines) / len(cosines),
        "first_difference_stage23_high_fraction": _stage23_high_fraction(first),
    }


def common_seed_decomposition(predictions: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    values = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(expert, dtype=np.float64)
    if values.shape != (len(SEEDS), HORIZON, 3) or target.shape != (HORIZON, 3):
        raise Stage30Error(f"common/seed decomposition shape mismatch: {values.shape}, {target.shape}")
    ensemble = np.mean(values, axis=0)
    common_error = ensemble - target
    seed_component = values - ensemble[None, :, :]
    total_error = values - target[None, :, :]
    total_mse = float(np.mean(np.square(total_error)))
    common_mse = float(np.mean(np.square(common_error)))
    seed_variance = float(np.mean(np.square(seed_component)))
    return {
        "total_residual_mse": total_mse,
        "common_mode_residual_mse": common_mse,
        "seed_dependent_variance": seed_variance,
        "common_fraction_of_total_mse": None if total_mse <= 0.0 else common_mse / total_mse,
        "seed_dependent_fraction_of_total_mse": None if total_mse <= 0.0 else seed_variance / total_mse,
        "decomposition_absolute_error": abs(total_mse - common_mse - seed_variance),
        "per_axis": [
            {
                "axis": name,
                "total_residual_mse": float(np.mean(np.square(total_error[..., axis]))),
                "common_mode_residual_mse": float(np.mean(np.square(common_error[..., axis]))),
                "seed_dependent_variance": float(np.mean(np.square(seed_component[..., axis]))),
            }
            for axis, name in enumerate(AXIS_NAMES)
        ],
        "between_seed_variance_per_horizon_xyz": np.mean(np.square(seed_component), axis=(0, 2)).astype(float).tolist(),
        "between_seed_variance_per_horizon_per_axis": np.mean(np.square(seed_component), axis=0).astype(float).tolist(),
        "diagnostic_ensemble_mean_profile": trajectory_profile(ensemble),
        "diagnostic_ensemble_mean_error": {
            "position": error_metrics(common_error),
            "first_difference": error_metrics(np.diff(common_error, axis=0)),
            "second_difference": error_metrics(np.diff(common_error, n=2, axis=0)),
        },
        "diagnostic_only_not_controller": True,
    }


def clipping_stats(raw_active: np.ndarray) -> dict[str, Any]:
    array = np.asarray(raw_active, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (HORIZON, ACTIVE_DIM) or not np.isfinite(array).all():
        raise Stage30Error(f"invalid raw active normalized array: {array.shape}")
    clipped = np.clip(array, -1.0, 1.0)
    exceeded = np.abs(array) > 1.0
    at_bound = np.abs(clipped) >= 1.0
    near_bound = np.abs(clipped) >= NEAR_CLIP_BOUND

    def fraction(mask: np.ndarray) -> float:
        return float(np.mean(mask))

    return {
        "sample_trajectories": int(array.shape[0]),
        "raw_max_abs": float(np.max(np.abs(array))),
        "raw_fraction_abs_gt_1": fraction(exceeded),
        "raw_positive_gt_1_fraction": fraction(array > 1.0),
        "raw_negative_lt_minus_1_fraction": fraction(array < -1.0),
        "raw_mean_positive_excess_over_1": float(np.mean(np.maximum(np.abs(array) - 1.0, 0.0))),
        "after_official_clip_fraction_at_bound": fraction(at_bound),
        "after_official_clip_fraction_near_bound": fraction(near_bound),
        "xyz": {
            "raw_fraction_abs_gt_1": fraction(exceeded[..., :3]),
            "after_official_clip_fraction_at_bound": fraction(at_bound[..., :3]),
            "after_official_clip_fraction_near_bound": fraction(near_bound[..., :3]),
        },
        "per_dimension": [
            {
                "dimension": index,
                "name": ACTIVE_DIM_NAMES[index],
                "raw_fraction_abs_gt_1": fraction(exceeded[..., index]),
                "after_official_clip_fraction_at_bound": fraction(at_bound[..., index]),
                "after_official_clip_fraction_near_bound": fraction(near_bound[..., index]),
                "raw_max_abs": float(np.max(np.abs(array[..., index]))),
            }
            for index in range(ACTIVE_DIM)
        ],
        "per_horizon_index": [
            {
                "horizon_index": index,
                "raw_fraction_abs_gt_1_active_10d": fraction(exceeded[:, index, :]),
                "raw_fraction_abs_gt_1_xyz": fraction(exceeded[:, index, :3]),
                "after_official_clip_fraction_at_bound_active_10d": fraction(at_bound[:, index, :]),
            }
            for index in range(HORIZON)
        ],
    }


def _lag1(values: np.ndarray) -> float | None:
    rows = np.asarray(values, dtype=np.float64)
    left = rows[:-1]
    right = rows[1:]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 1e-15 else float(np.dot(left, right) / denominator)


def _sign_flip(values: np.ndarray) -> tuple[int, int, float | None]:
    rows = np.asarray(values, dtype=np.float64)
    left = rows[:-1]
    right = rows[1:]
    valid = (np.abs(left) > 1e-15) & (np.abs(right) > 1e-15)
    eligible = int(np.sum(valid))
    flips = int(np.sum((left * right < 0.0) & valid))
    return flips, eligible, None if eligible == 0 else flips / eligible


def _cosine_structure(vectors: np.ndarray) -> dict[str, Any]:
    cosines = _adjacent_cosines(vectors)
    return {
        "cosine": _summary(cosines),
        "negative_fraction": None if not cosines else sum(value < 0.0 for value in cosines) / len(cosines),
    }


def direction_structure(predicted: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    trajectory = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(expert, dtype=np.float64)
    residual = trajectory - target
    per_axis = []
    total_flips = 0
    total_eligible = 0
    lag_values = []
    for axis, name in enumerate(AXIS_NAMES):
        flips, eligible, rate = _sign_flip(residual[:, axis])
        lag = _lag1(residual[:, axis])
        total_flips += flips
        total_eligible += eligible
        if lag is not None:
            lag_values.append(lag)
        per_axis.append({
            "axis": name,
            "sign_flips": flips,
            "eligible_adjacent_pairs": eligible,
            "sign_flip_rate": rate,
            "lag1_autocorrelation": lag,
        })
    return {
        "residual_per_axis": per_axis,
        "residual_sign_flip_rate_all_axes": None if total_eligible == 0 else total_flips / total_eligible,
        "residual_lag1_autocorrelation_axis_mean": None if not lag_values else mean(lag_values),
        "adjacent_residual_vector": _cosine_structure(residual),
        "predicted_increment_back_and_forth": _cosine_structure(np.diff(trajectory, axis=0)),
        "residual_increment_back_and_forth": _cosine_structure(np.diff(residual, axis=0)),
    }


def _artifact_entry(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def validate_stage29_seal(seal_path: Path, project_repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _sha256_file(seal_path) != STAGE29_SEAL_SHA256:
        raise Stage30Error("Stage29 artifact seal SHA mismatch")
    seal = _read_json(seal_path)
    if (
        seal.get("status") != "PASS"
        or seal.get("artifact_count") != 16
        or seal.get("execution_project_sha") != STAGE29_EXECUTION_SHA
        or seal.get("posthoc_repair_sha") != STAGE29_CLOSURE_SHA
    ):
        raise Stage30Error("Stage29 artifact seal contract drifted")
    verified = []
    for entry in seal.get("artifacts", []):
        expected_sha = entry["sha256"]
        expected_bytes = int(entry["bytes"])
        location = str(entry["path"])
        if location.startswith("git:"):
            _, revision, relative_path = location.split(":", 2)
            content = _git_bytes(project_repo, revision, relative_path)
            observed_sha = hashlib.sha256(content).hexdigest()
            observed_bytes = len(content)
        else:
            path = Path(location)
            observed_sha = _sha256_file(path)
            observed_bytes = path.stat().st_size
        if observed_sha != expected_sha or observed_bytes != expected_bytes:
            raise Stage30Error(f"Stage29 sealed artifact drifted: {location}")
        verified.append({"path": location, "bytes": observed_bytes, "sha256": observed_sha})
    if len(verified) != 16:
        raise Stage30Error("Stage29 artifact verification count mismatch")
    return seal, verified


def _load_stage29_inputs(
    stage29_root: Path,
    stage29_result: Mapping[str, Any],
    stage29_seal: Mapping[str, Any],
) -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], list[dict[str, Any]], np.ndarray]:
    records = stage29_result.get("records", [])
    lookup = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    if len(records) != 6 or set(lookup) != set(EXPECTED_NPZ_NAMES):
        raise Stage30Error("Stage29 result does not contain the exact six frozen records")
    sealed_paths = {entry["path"]: entry for entry in stage29_seal["artifacts"] if not str(entry["path"]).startswith("git:")}
    arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    manifest_rows = []
    expert_reference = None
    for key, filename in EXPECTED_NPZ_NAMES.items():
        seed, checkpoint = key
        path = (stage29_root / filename).resolve(strict=True)
        row = lookup[key]
        if row["artifact"]["path"] != str(path) or row["artifact"]["sha256"] != _sha256_file(path):
            raise Stage30Error(f"Stage29 record/artifact mismatch: {path}")
        sealed_entry = sealed_paths.get(str(path))
        if sealed_entry is None or sealed_entry["sha256"] != row["artifact"]["sha256"]:
            raise Stage30Error(f"Stage29 NPZ is not bound by seal: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_ARRAY_KEYS:
                raise Stage30Error(f"Stage29 NPZ key drift: {path}")
            payload = {name: np.array(archive[name], copy=True) for name in archive.files}
        expected_shapes = {
            "initial_noise_float32": (1, HORIZON, UNIVERSAL_DIM),
            "raw_universal_action": (1, HORIZON, UNIVERSAL_DIM),
            "decoded_physical_action": (1, HORIZON, ACTIVE_DIM),
            "expert_physical_action": (1, HORIZON, ACTIVE_DIM),
        }
        for name, array in payload.items():
            if array.shape != expected_shapes[name] or array.dtype != np.float32 or not np.isfinite(array).all():
                raise Stage30Error(f"Stage29 array shape/dtype/finite drift: {path}:{name}")
        if _array_sha256(payload["initial_noise_float32"]) != row["initial_noise"]["float32_sha256"]:
            raise Stage30Error(f"saved initial noise hash mismatch: {path}")
        if _array_sha256(payload["raw_universal_action"]) != row["raw_universal_action"]["sha256"]:
            raise Stage30Error(f"saved raw action hash mismatch: {path}")
        if _array_sha256(payload["decoded_physical_action"]) != row["decoded_physical_action"]["sha256"]:
            raise Stage30Error(f"saved decoded action hash mismatch: {path}")
        if expert_reference is None:
            expert_reference = payload["expert_physical_action"]
        elif not np.array_equal(expert_reference, payload["expert_physical_action"]):
            raise Stage30Error("expert physical trajectory differs across Stage29 NPZ artifacts")
        arrays[key] = payload
        manifest_rows.append({
            "seed": seed,
            "checkpoint": checkpoint,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "array_sha256": {name: _array_sha256(array) for name, array in payload.items()},
        })
    for seed in SEEDS:
        if not np.array_equal(
            arrays[(seed, "BASE")]["initial_noise_float32"],
            arrays[(seed, "H2@2000")]["initial_noise_float32"],
        ):
            raise Stage30Error(f"paired initial noise no longer matches for seed {seed}")
    assert expert_reference is not None
    return arrays, manifest_rows, expert_reference


def _validate_stage29_profile_reproduction(
    arrays: Mapping[tuple[int, str], Mapping[str, np.ndarray]],
    stage29_result: Mapping[str, Any],
) -> dict[str, Any]:
    records = {
        (int(row["seed"]), str(row["checkpoint"])): row
        for row in stage29_result["records"]
    }
    comparisons = []
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            observed = trajectory_profile(arrays[(seed, checkpoint)]["decoded_physical_action"][0, :, :3])
            expected = records[(seed, checkpoint)]["metrics"]["predicted_profile"]
            fields = {
                "path_length_m": (observed["path_length_m"], expected["path_length_m"]),
                "linear_speed_mean_mps": (observed["linear_speed_mps"]["mean"], expected["linear_speed_mps"]["mean"]),
                "linear_speed_max_mps": (observed["linear_speed_mps"]["max"], expected["linear_speed_mps"]["max"]),
                "roughness_mean_m": (observed["roughness_second_difference_m"]["mean"], expected["second_difference_m"]["mean"]),
                "back_and_forth_rate": (observed["back_and_forth_rate"], expected["back_and_forth_rate"]),
                "first_difference_high_fraction": (
                    observed["first_difference_stage23_high_fraction"],
                    expected["first_difference_dft"]["high_fraction"],
                ),
            }
            differences = {name: abs(float(left) - float(right)) for name, (left, right) in fields.items()}
            if max(differences.values()) > 1e-12:
                raise Stage30Error(f"Stage29 profile metric reproduction drifted for {checkpoint} seed {seed}")
            comparisons.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "maximum_absolute_difference": max(differences.values()),
            })
    return {"status": "MATCH", "record_count": len(comparisons), "tolerance": 1e-12, "comparisons": comparisons}


def _aggregate_direction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "residual_sign_flip_rate_all_axes": lambda row: row["residual_sign_flip_rate_all_axes"],
        "residual_lag1_autocorrelation_axis_mean": lambda row: row["residual_lag1_autocorrelation_axis_mean"],
        "adjacent_residual_cosine_mean": lambda row: row["adjacent_residual_vector"]["cosine"]["mean"],
        "adjacent_residual_negative_fraction": lambda row: row["adjacent_residual_vector"]["negative_fraction"],
        "predicted_increment_back_and_forth_rate": lambda row: row["predicted_increment_back_and_forth"]["negative_fraction"],
        "residual_increment_back_and_forth_rate": lambda row: row["residual_increment_back_and_forth"]["negative_fraction"],
    }
    return {
        field: _summary(value for row in rows if (value := accessor(row)) is not None)
        for field, accessor in fields.items()
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization_token != PI_GATE:
        raise Stage30Error("PI gate token mismatch")
    head = _git(args.project_repo, "rev-parse", "HEAD")
    if head != args.expected_project_sha or _git(args.project_repo, "status", "--short"):
        raise Stage30Error("Stage30 checkout SHA/cleanliness mismatch")
    subprocess.check_call(["git", "-C", str(args.project_repo), "merge-base", "--is-ancestor", STAGE29_CLOSURE_SHA, head])

    seal, verified_stage29 = validate_stage29_seal(args.stage29_seal, args.project_repo)
    if _sha256_file(args.stage29_result) != STAGE29_RESULT_SHA256:
        raise Stage30Error("Stage29 result SHA mismatch")
    stage29_result = _read_json(args.stage29_result)
    if stage29_result.get("status") != "PASS_SCIENTIFIC_EXECUTION_POSTHOC_CLOSURE":
        raise Stage30Error("Stage29 result status drifted")
    p0_sha = _sha256_file(args.p0_evidence)
    if p0_sha != args.expected_p0_sha256:
        raise Stage30Error("P0 evidence SHA mismatch")
    p0 = _read_json(args.p0_evidence)
    if p0.get("status") not in {"PASS", "PASS_WITH_PROVENANCE_NOTE"}:
        raise Stage30Error(f"P0 did not close import provenance: {p0.get('status')}")
    if p0.get("model_forward") is not False or p0.get("sampler_calls") != 0:
        raise Stage30Error("P0 execution boundary drifted")

    arrays, input_manifest, expert_physical = _load_stage29_inputs(
        args.stage29_root, stage29_result, seal,
    )
    stage29_metric_reproduction = _validate_stage29_profile_reproduction(arrays, stage29_result)
    expert_xyz = expert_physical[0, :, :3].astype(np.float64)
    predictions = {
        checkpoint: np.stack([
            arrays[(seed, checkpoint)]["decoded_physical_action"][0, :, :3]
            for seed in SEEDS
        ]).astype(np.float64)
        for checkpoint in CHECKPOINTS
    }
    raw_active = {
        checkpoint: np.stack([
            arrays[(seed, checkpoint)]["raw_universal_action"][0, :, :ACTIVE_DIM]
            for seed in SEEDS
        ]).astype(np.float64)
        for checkpoint in CHECKPOINTS
    }

    error_arrays: dict[str, dict[str, list[np.ndarray]]] = {}
    pointwise_per_seed = []
    frequency_per_seed = []
    direction_per_seed = []
    clipping_per_seed = []
    for checkpoint in CHECKPOINTS:
        error_arrays[checkpoint] = {"position": [], "first_difference": [], "second_difference": []}
        for seed_index, seed in enumerate(SEEDS):
            predicted = predictions[checkpoint][seed_index]
            position = predicted - expert_xyz
            first = np.diff(position, axis=0)
            second = np.diff(position, n=2, axis=0)
            signals = {"position": position, "first_difference": first, "second_difference": second}
            for name, signal in signals.items():
                error_arrays[checkpoint][name].append(signal)
            pointwise_per_seed.append({
                "checkpoint": checkpoint,
                "seed": seed,
                **{name: error_metrics(signal) for name, signal in signals.items()},
            })
            frequency_per_seed.append({
                "checkpoint": checkpoint,
                "seed": seed,
                "signals": {name: spectral_decomposition(signal) for name, signal in signals.items()},
            })
            structure = direction_structure(predicted, expert_xyz)
            direction_per_seed.append({"checkpoint": checkpoint, "seed": seed, **structure})
            clipping_per_seed.append({
                "checkpoint": checkpoint,
                "seed": seed,
                **clipping_stats(raw_active[checkpoint][seed_index : seed_index + 1]),
            })

    pointwise_aggregate = {}
    per_horizon = {}
    frequency_aggregate = {}
    common_mode = {}
    direction_aggregate = {}
    clipping_aggregate = {}
    for checkpoint in CHECKPOINTS:
        pointwise_aggregate[checkpoint] = {}
        per_horizon[checkpoint] = {}
        frequency_aggregate[checkpoint] = {}
        for name, start_index in (("position", 0), ("first_difference", 1), ("second_difference", 2)):
            stacked = np.stack(error_arrays[checkpoint][name])
            pointwise_aggregate[checkpoint][name] = error_metrics(stacked)
            per_horizon[checkpoint][name] = per_horizon_error(stacked, start_index)
            frequency_aggregate[checkpoint][name] = aggregate_spectral_decomposition(
                error_arrays[checkpoint][name]
            )
        common_mode[checkpoint] = common_seed_decomposition(predictions[checkpoint], expert_xyz)
        direction_rows = [row for row in direction_per_seed if row["checkpoint"] == checkpoint]
        direction_aggregate[checkpoint] = _aggregate_direction(direction_rows)
        clipping_aggregate[checkpoint] = clipping_stats(raw_active[checkpoint])

    official_utils = args.pinned_groot_source / "gr00t/data/utils.py"
    clipping_source_sha = _sha256_file(official_utils)
    if clipping_source_sha != "fa765b240295d9884ab62326c8a21465ae21e6f10133a4b68ac321e8101e4ab7":
        raise Stage30Error("official clipping source SHA mismatch")

    arrays_path = args.output_root / "stage30_temporal_decomposition_arrays.npz"
    np.savez_compressed(
        arrays_path,
        expert_xyz=expert_xyz.astype(np.float64),
        base_predictions_xyz=predictions["BASE"].astype(np.float64),
        h2_predictions_xyz=predictions["H2@2000"].astype(np.float64),
        base_position_residual=predictions["BASE"] - expert_xyz[None, :, :],
        h2_position_residual=predictions["H2@2000"] - expert_xyz[None, :, :],
        base_ensemble_mean_xyz=np.mean(predictions["BASE"], axis=0),
        h2_ensemble_mean_xyz=np.mean(predictions["H2@2000"], axis=0),
        base_raw_normalized_active_10d=raw_active["BASE"],
        h2_raw_normalized_active_10d=raw_active["H2@2000"],
    )

    result = {
        "schema": SCHEMA,
        "status": "PASS_ZERO_MODEL_DECOMPOSITION",
        "pi_gate": PI_GATE,
        "project_sha": head,
        "p0_runtime_import_origin": {
            "status": p0["status"],
            "evidence_path": str(args.p0_evidence.resolve()),
            "evidence_sha256": p0_sha,
            "module_count": p0["module_count"],
            "inside_pinned_count": p0["inside_pinned_count"],
            "byte_exact_count": p0["byte_exact_count"],
        },
        "stage29_artifact_seal_revalidated": {
            "status": "PASS",
            "path": str(args.stage29_seal.resolve()),
            "sha256": STAGE29_SEAL_SHA256,
            "artifact_count": len(verified_stage29),
            "stage29_execution_sha": STAGE29_EXECUTION_SHA,
            "stage29_posthoc_closure_sha": STAGE29_CLOSURE_SHA,
        },
        "input_contract": {
            "seeds": list(SEEDS),
            "checkpoints": list(CHECKPOINTS),
            "trajectory_shape_per_record": [1, HORIZON, ACTIVE_DIM],
            "raw_shape_per_record": [1, HORIZON, UNIVERSAL_DIM],
            "expert_xyz_sha256": _array_sha256(expert_xyz.astype(np.float32)),
            "paired_initial_noise_revalidated": True,
        },
        "stage29_profile_metric_reproduction": stage29_metric_reproduction,
        "pointwise_error": {
            "definition": "EEF XYZ residual p_pred(t)-p_expert(t)",
            "per_seed": pointwise_per_seed,
            "aggregate": pointwise_aggregate,
            "per_horizon_index": per_horizon,
        },
        "frequency_decomposition": {
            "convention": {
                "transform": "unscaled all-bin numpy FFT along time, energy=abs(FFT)^2",
                "dc": "folded_bin=0",
                "very_low_non_dc": "1 <= folded_bin < ceil(N/8)",
                "mid_low": "ceil(N/8) <= folded_bin < ceil(N/4)",
                "low_including_dc": "folded_bin < ceil(N/4)",
                "high_stage23_compatible": "folded_bin >= ceil(N/4)",
                "thresholds_declared_before_comparison": True,
            },
            "per_seed": frequency_per_seed,
            "aggregate": frequency_aggregate,
        },
        "common_mode_vs_seed_variance": common_mode,
        "diagnostic_ensemble_mean": {
            "BASE": common_mode["BASE"]["diagnostic_ensemble_mean_profile"],
            "H2@2000": common_mode["H2@2000"]["diagnostic_ensemble_mean_profile"],
            "EXPERT": trajectory_profile(expert_xyz),
            "diagnostic_only_not_controller": True,
        },
        "raw_normalized_clipping_audit": {
            "official_decode_contract": "np.clip(normalized_values,-1,1) before min-max unnormalization",
            "official_source_path": str(official_utils.resolve()),
            "official_source_sha256": clipping_source_sha,
            "near_bound_threshold": NEAR_CLIP_BOUND,
            "per_seed": clipping_per_seed,
            "aggregate": clipping_aggregate,
            "normalization_or_clipping_changed": False,
        },
        "error_direction_structure": {
            "definitions": {
                "sign_flip": "adjacent scalar residuals have opposite signs; exact zeros excluded",
                "lag1_autocorrelation": "Pearson cosine of separately centered adjacent scalar residual sequences",
                "adjacent_residual_cosine": "cosine between e_t and e_(t+1)",
                "back_and_forth": "negative cosine between adjacent XYZ increments",
            },
            "per_seed": direction_per_seed,
            "aggregate": direction_aggregate,
        },
        "derived_arrays": _artifact_entry(arrays_path, "derived_numeric_arrays"),
        "execution": {
            "posthoc_model_calls": 0,
            "new_sampler_calls": 0,
            "checkpoint_loads": 0,
            "gpu_scientific_execution": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_executed": False,
            "hardware_executed": False,
        },
        "all_finite": True,
        "scientific_contract_changed": False,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "claim_boundary": "deterministic decomposition of six sealed Stage29 trajectories on one locked TINY1 training sample",
        "next_gate": NEXT_GATE,
    }
    if not _all_finite(result):
        raise Stage30Error("Stage30 result contains a non-finite number")

    result_path = args.output_root / "stage30_temporal_residual_result.json"
    _write_json_atomic(result_path, result)
    raw_manifest_path = args.output_root / "stage30_input_output_hash_manifest.json"
    _write_json_atomic(raw_manifest_path, {
        "schema": SCHEMA,
        "stage29_seal": _artifact_entry(args.stage29_seal, "stage29_artifact_seal"),
        "stage29_result": _artifact_entry(args.stage29_result, "stage29_result"),
        "p0_evidence": _artifact_entry(args.p0_evidence, "p0_runtime_import_origin"),
        "stage29_npz_inputs": input_manifest,
        "stage30_outputs": [
            _artifact_entry(arrays_path, "derived_numeric_arrays"),
            _artifact_entry(result_path, "result"),
        ],
    })

    seal_paths = [
        args.stage29_seal,
        args.stage29_result,
        args.stage29_root / "stage29_c2_partial_records.json",
        *[args.stage29_root / name for name in EXPECTED_NPZ_NAMES.values()],
        args.p0_evidence,
        arrays_path,
        result_path,
        raw_manifest_path,
        Path(__file__).resolve(),
        args.p0_runner,
        args.p0_wrapper,
        args.analysis_test,
        args.p0_test,
    ]
    entries = []
    seen = set()
    for path in seal_paths:
        resolved = path.resolve(strict=True)
        if str(resolved) not in seen:
            seen.add(str(resolved))
            entries.append(_artifact_entry(resolved, "stage30_evidence"))
    partial_entry = next(entry for entry in entries if entry["path"].endswith("stage29_c2_partial_records.json"))
    if partial_entry["sha256"] != STAGE29_PARTIAL_SHA256:
        raise Stage30Error("Stage29 partial records SHA mismatch")
    seal_payload = {
        "schema": "STAGE30_TEMPORAL_RESIDUAL_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "pi_gate": PI_GATE,
        "project_sha": head,
        "stage29_artifact_seal_sha256": STAGE29_SEAL_SHA256,
        "artifact_count": len(entries),
        "artifacts": entries,
        "posthoc_model_calls": 0,
        "new_sampler_calls": 0,
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    seal_path = args.output_root / "stage30_temporal_residual_artifact_seal.json"
    _write_json_atomic(seal_path, seal_payload)
    return {
        "result": _artifact_entry(result_path, "result"),
        "raw_manifest": _artifact_entry(raw_manifest_path, "input_output_hash_manifest"),
        "derived_arrays": _artifact_entry(arrays_path, "derived_numeric_arrays"),
        "artifact_seal": {**_artifact_entry(seal_path, "artifact_seal"), "artifact_count": len(entries)},
        "p0_status": p0["status"],
        "posthoc_model_calls": 0,
        "new_sampler_calls": 0,
        "next_gate": NEXT_GATE,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, required=True)
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--stage29-root", type=Path, required=True)
    parser.add_argument("--stage29-seal", type=Path, required=True)
    parser.add_argument("--stage29-result", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--expected-p0-sha256", required=True)
    parser.add_argument("--pinned-groot-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p0-runner", type=Path, required=True)
    parser.add_argument("--p0-wrapper", type=Path, required=True)
    parser.add_argument("--analysis-test", type=Path, required=True)
    parser.add_argument("--p0-test", type=Path, required=True)
    parser.add_argument("--authorization-token", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = analyze(_parser().parse_args(argv))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "posthoc_model_calls": 0,
            "new_sampler_calls": 0,
        }, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
