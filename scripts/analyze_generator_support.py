#!/usr/bin/env python3
"""Compare Stage16 descriptors with the historical generator descriptor set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from go2arx5_vla.benchmark.progressive_relocation import load_npz, trajectory_descriptor


FEATURE_NAMES = (
    "duration_s",
    "translation_path_length_m",
    "net_displacement_m",
    "translation_speed_mean_mps",
    "translation_speed_p95_mps",
    "translation_speed_max_mps",
    "angular_speed_mean_radps",
    "angular_speed_p95_radps",
    "angular_speed_max_radps",
    "orientation_range_rad",
    "curvature_proxy_rad_per_m",
    "se3_path_length_m",
)

HISTORICAL_COLUMNS = {
    "duration_s": "duration_s",
    "translation_path_length_m": "position_path_length_m",
    "net_displacement_m": "net_displacement_m",
    "translation_speed_mean_mps": "translation_speed_mps.mean",
    "translation_speed_p95_mps": "translation_speed_mps.p95",
    "translation_speed_max_mps": "translation_speed_mps.max",
    "angular_speed_mean_radps": "angular_speed_radps.mean",
    "angular_speed_p95_radps": "angular_speed_radps.p95",
    "angular_speed_max_radps": "angular_speed_radps.max",
    "orientation_range_rad": "orientation_range_rad",
    "curvature_proxy_rad_per_m": "curvature_proxy_rad_per_m",
    "se3_path_length_m": "se3_arc_length_m",
}


def _load_historical(path: Path) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("historical descriptor CSV is empty")
    matrix = np.asarray(
        [[float(row[HISTORICAL_COLUMNS[name]]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    return matrix, rows


def _benchmark_features(descriptor: Mapping[str, Any]) -> np.ndarray:
    translation = descriptor["translation_speed_mps"]
    angular = descriptor["angular_speed_radps"]
    translation_mean = translation.get("mean", translation.get("median", 0.0))
    angular_mean = angular.get("mean", angular.get("median", 0.0))
    return np.asarray(
        [
            descriptor["duration_s"],
            descriptor["translation_path_length_m"],
            descriptor["net_displacement_m"],
            translation_mean,
            translation["p95"],
            translation["max"],
            angular_mean,
            angular["p95"],
            angular["max"],
            descriptor["orientation_range_rad_max"],
            descriptor.get("curvature_proxy_rad_per_m", descriptor["curvature_proxy_max"]),
            descriptor["path_length_m"],
        ],
        dtype=np.float64,
    )


def _nearest(matrix: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(matrix)
        distance, index = tree.query(queries, k=1)
        return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)
    except ImportError:
        distances = np.empty(len(queries), dtype=np.float64)
        indices = np.empty(len(queries), dtype=np.int64)
        for start in range(0, len(queries), 256):
            chunk = queries[start : start + 256]
            delta = chunk[:, None, :] - matrix[None, :, :]
            squared = np.sum(delta * delta, axis=-1)
            indices[start : start + len(chunk)] = np.argmin(squared, axis=1)
            distances[start : start + len(chunk)] = np.sqrt(
                squared[np.arange(len(chunk)), indices[start : start + len(chunk)]]
            )
        return distances, indices


def _leave_one_out_distance(matrix: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(matrix)
        distance, _ = tree.query(matrix, k=2)
        return np.asarray(distance[:, 1], dtype=np.float64)
    except ImportError:
        values = []
        for start in range(0, len(matrix), 128):
            chunk = matrix[start : start + 128]
            delta = chunk[:, None, :] - matrix[None, :, :]
            squared = np.sum(delta * delta, axis=-1)
            local = np.arange(len(chunk))[:, None]
            local_indices = np.arange(start, min(start + len(chunk), len(matrix)))[None, :]
            squared[local, local_indices] = np.inf
            values.extend(np.sqrt(np.min(squared, axis=1)).tolist())
        return np.asarray(values, dtype=np.float64)


def _robust_normalize(samples: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(samples, axis=0)
    q25 = np.percentile(samples, 25, axis=0)
    q75 = np.percentile(samples, 75, axis=0)
    # Several reconstructed generator columns are intentionally fixed (for
    # example duration).  A 1e-6 floor would turn a six-second benchmark
    # duration difference into a meaningless multi-million distance.  Use a
    # relative robust fallback for zero-IQR dimensions instead.
    fallback = np.maximum(np.abs(median) * 0.05, 1e-3)
    scale = np.maximum(q75 - q25, fallback)
    return (samples - median) / scale, (queries - median) / scale, {"median": median, "iqr": scale}


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return None
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _band(value: float, low: float, high: float) -> str:
    if value <= low:
        return "LOW"
    if value <= high:
        return "MEDIUM"
    return "HIGH"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    historical_raw, historical_rows = _load_historical(args.historical_csv)
    metrics_payload = json.loads(args.metrics.read_text())
    metrics_by_id = {
        str(row["trajectory_id"]): row.get("metrics", {})
        for row in metrics_payload.get("cases", [])
    }
    manifest = json.loads(args.manifest.read_text())
    benchmark_descriptors = []
    case_rows = []
    for case in manifest.get("cases", []):
        data = load_npz(Path(str(case["trajectory_path"])))
        descriptor = trajectory_descriptor(
            data["timestamp"], data["position"], data["quaternion"]
        )
        benchmark_descriptors.append(_benchmark_features(descriptor))
        case_rows.append((case, descriptor))

    normalized_samples, normalized_queries, normalization = _robust_normalize(
        historical_raw, np.asarray(benchmark_descriptors, dtype=np.float64)
    )
    loo = _leave_one_out_distance(normalized_samples)
    query_distance, query_index = _nearest(
        normalized_samples, normalized_queries
    )
    p95 = float(np.percentile(loo, 95))
    p99 = float(np.percentile(loo, 99))

    cases = []
    position_errors = []
    rotation_errors = []
    distances = []
    for index, ((case, descriptor), distance, nearest_index) in enumerate(
        zip(case_rows, query_distance, query_index)
    ):
        if distance <= p95:
            support_status = "WITHIN_EMPIRICAL_SUPPORT_RANGE"
        elif distance <= p99:
            support_status = "NEAR_SUPPORT_BOUNDARY"
        else:
            support_status = "OUTSIDE_EMPIRICAL_SUPPORT_RANGE"
        metrics = metrics_by_id.get(str(case["trajectory_id"]), {})
        position_mean = metrics.get("position_m", {}).get("mean")
        rotation_mean = metrics.get("rotation_rad", {}).get("mean")
        if position_mean is not None:
            position_errors.append(float(position_mean))
            rotation_errors.append(float(rotation_mean))
            distances.append(float(distance))
        cases.append(
            {
                "trajectory_id": str(case["trajectory_id"]),
                "cohort": case.get("cohort"),
                "native_class": case.get("native_class"),
                "descriptor": descriptor,
                "generator_descriptor_distance": float(distance),
                "nearest_historical_sample_index": int(nearest_index),
                "support_status": support_status,
                "termination_code": metrics.get("termination_code"),
                "position_error_mean_m": position_mean,
                "rotation_error_mean_rad": rotation_mean,
            }
        )

    def sample_threshold(column: str, quantile: float) -> float:
        index = list(HISTORICAL_COLUMNS.values()).index(column)
        return float(np.percentile(historical_raw[:, index], quantile))

    thresholds = {
        "translation_speed_p95_mps": (
            sample_threshold("translation_speed_mps.p95", 33),
            sample_threshold("translation_speed_mps.p95", 67),
        ),
        "angular_speed_p95_radps": (
            sample_threshold("angular_speed_radps.p95", 33),
            sample_threshold("angular_speed_radps.p95", 67),
        ),
        "curvature_proxy_rad_per_m": (
            sample_threshold("curvature_proxy_rad_per_m", 33),
            sample_threshold("curvature_proxy_rad_per_m", 67),
        ),
        "orientation_range_rad": (
            sample_threshold("orientation_range_rad", 33),
            sample_threshold("orientation_range_rad", 67),
        ),
        "se3_path_length_m": (
            sample_threshold("se3_arc_length_m", 33),
            sample_threshold("se3_arc_length_m", 67),
        ),
    }
    stratified = defaultdict(lambda: {"count": 0, "position_mean_m": [], "rotation_mean_rad": [], "distance": []})
    for row in cases:
        descriptor = row["descriptor"]
        metrics = metrics_by_id.get(row["trajectory_id"], {})
        stratification = {
            "speed": _band(descriptor["translation_speed_mps"]["p95"], *thresholds["translation_speed_p95_mps"]),
            "angular_speed": _band(descriptor["angular_speed_radps"]["p95"], *thresholds["angular_speed_p95_radps"]),
            "curvature": _band(
                descriptor.get("curvature_proxy_rad_per_m", descriptor["curvature_proxy_max"]),
                *thresholds["curvature_proxy_rad_per_m"],
            ),
            "orientation_demand": _band(descriptor["orientation_range_rad_max"], *thresholds["orientation_range_rad"]),
            "workspace_burden": _band(descriptor["path_length_m"], *thresholds["se3_path_length_m"]),
            "progressive_relocation_burden": (
                "PROGRESSIVE" if row.get("cohort") == "PROGRESSIVE_BASE_RELOCATION" else "NON_PROGRESSIVE"
            ),
        }
        row["stratification"] = stratification
        for dimension, band in stratification.items():
            key = "%s=%s" % (dimension, band)
            bucket = stratified[key]
            bucket["count"] += 1
            if row.get("position_error_mean_m") is not None:
                bucket["position_mean_m"].append(row["position_error_mean_m"])
                bucket["rotation_mean_rad"].append(row["rotation_error_mean_rad"])
                bucket["distance"].append(row["generator_descriptor_distance"])

    stratified_output = {}
    for key, value in sorted(stratified.items()):
        stratified_output[key] = {
            "count": value["count"],
            "position_error_mean_m": float(np.mean(value["position_mean_m"])) if value["position_mean_m"] else None,
            "rotation_error_mean_rad": float(np.mean(value["rotation_mean_rad"])) if value["rotation_mean_rad"] else None,
            "generator_descriptor_distance_mean": float(np.mean(value["distance"])) if value["distance"] else None,
            "termination_counts": dict(
                sorted(
                    Counter(
                        row.get("termination_code", "UNKNOWN")
                        for row in cases
                        if key in {
                            "%s=%s" % (dimension, row["stratification"][dimension])
                            for dimension in row["stratification"]
                        }
                    ).items()
                )
            ),
        }

    support_counts = Counter(row["support_status"] for row in cases)
    termination_by_support = defaultdict(Counter)
    for row in cases:
        termination_by_support[row["support_status"]][row.get("termination_code", "UNKNOWN")] += 1
    payload = {
        "schema": "STAGE16_GENERATOR_DESCRIPTOR_SUPPORT_V1",
        "historical_descriptor_file": str(args.historical_csv),
        "historical_sample_count": len(historical_rows),
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "method": "median_and_IQR",
            "median": normalization["median"].tolist(),
            "iqr": normalization["iqr"].tolist(),
        },
        "support_calibration": {
            "method": "leave_one_out_nearest_neighbor_distance",
            "p95": p95,
            "p99": p99,
            "distance_semantics": "GENERATOR_DESCRIPTOR_DISTANCE",
        },
        "support_status_counts": dict(sorted(support_counts.items())),
        "termination_by_support_status": {
            key: dict(sorted(value.items())) for key, value in sorted(termination_by_support.items())
        },
        "correlation": {
            "position_error_mean_m_vs_generator_descriptor_distance": _pearson(distances, position_errors),
            "rotation_error_mean_rad_vs_generator_descriptor_distance": _pearson(distances, rotation_errors),
        },
        "stratified_summary": stratified_output,
        "cases": cases,
        "interpretation": (
            "This is empirical descriptor support relative to 10000 reconstructed generator samples; "
            "it is not TRUE_TRAINING_OOD because training visitation frequencies are unknown."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "historical_sample_count": len(historical_rows),
        "benchmark_case_count": len(cases),
        "support_status_counts": payload["support_status_counts"],
        "termination_by_support_status": payload["termination_by_support_status"],
        "correlation": payload["correlation"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
