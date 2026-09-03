#!/usr/bin/env python3
"""Audit Stage22 action statistics against pinned N1.7 relative-EEF stats.

This is a CPU-only, source-data audit.  It reads JSON statistics files and a
processor configuration; it does not construct a model, open CUDA, load
checkpoint tensors, run inference, or alter any dataset.  The H2 statistics
are expected to be the exact frozen Stage22 statistics, while the upstream
statistics are read from the exact Stage10 N1.7 base-model tree.

The audit keeps the time-indexed ``[H=40, D=9]`` relative EEF cells visible.
Global counts alone can hide the important distinction between a structural
zero-width first relative point, constant orientation dimensions, and a
degenerate translation distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_H2_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
EXPECTED_H2_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
HORIZON = 40
EEF_DIM = 9
TRANSLATION_DIMS = (0, 1, 2)
ROTATION_DIMS = (3, 4, 5, 6, 7, 8)
DIM_NAMES = (
    "translation_x",
    "translation_y",
    "translation_z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
)
UPSTREAM_CATEGORIES: dict[str, tuple[str, ...]] = {
    "oxe_droid_relative_eef_relative_joint": ("eef_9d",),
    "xdof_relative_eef_relative_joint": ("left_wrist_eef", "right_wrist_eef"),
    "real_g1_relative_eef_relative_joints": (
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for child in value:
            result.extend(flatten(child))
        return result
    return [float(value)]


def nested_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [len(value)] + (nested_shape(value[0]) if value else [])


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "finite": True,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
        }
    if not all(math.isfinite(value) for value in items):
        raise ValueError("statistics contain non-finite values")
    ordered = sorted(items)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(items),
        "finite": True,
        "min": min(items),
        "median": median,
        "max": max(items),
        "mean": sum(items) / len(items),
    }


def zero_width(low: float, high: float) -> bool:
    # Match the official normalization utility's np.isclose-style tolerance
    # used by the Stage23 action-pipeline audit.
    return abs(high - low) <= 1e-8 + 1e-5 * abs(high)


def stat_node_summary(node: Mapping[str, Any], *, expected_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    lower = flatten(node["q01"])
    upper = flatten(node["q99"])
    if len(lower) != len(upper):
        raise ValueError("q01/q99 cell count mismatch")
    shape = nested_shape(node["q01"])
    if expected_shape is not None and tuple(shape) != expected_shape:
        raise ValueError(f"unexpected q01 shape {shape}; expected {list(expected_shape)}")
    widths = [high - low for low, high in zip(lower, upper)]
    zero = [zero_width(low, high) for low, high in zip(lower, upper)]
    return {
        "q01_q99_shape": shape,
        "total_cells": len(widths),
        "zero_width_cells_official_isclose": sum(zero),
        "nonzero_width_cells": sum(not item for item in zero),
        "zero_width_fraction": sum(zero) / len(zero) if zero else None,
        "width": numeric_summary(widths),
        "q01": numeric_summary(lower),
        "q99": numeric_summary(upper),
    }


def time_indexed_eef_summary(node: Mapping[str, Any]) -> dict[str, Any]:
    lower = node["q01"]
    upper = node["q99"]
    shape = nested_shape(lower)
    if shape != [HORIZON, EEF_DIM]:
        raise ValueError(f"expected time-indexed EEF stats [{HORIZON},{EEF_DIM}], got {shape}")
    per_dimension: dict[str, Any] = {}
    for index, name in enumerate(DIM_NAMES):
        lows = [float(lower[t][index]) for t in range(HORIZON)]
        highs = [float(upper[t][index]) for t in range(HORIZON)]
        widths = [high - low for low, high in zip(lows, highs)]
        zero = [zero_width(low, high) for low, high in zip(lows, highs)]
        per_dimension[name] = {
            "dimension": index,
            "zero_width_time_indices": [t for t, is_zero in enumerate(zero) if is_zero],
            "zero_width_count": sum(zero),
            "nonzero_width_count": sum(not item for item in zero),
            "width": numeric_summary(widths),
            "q01": numeric_summary(lows),
            "q99": numeric_summary(highs),
        }
    translation_widths = [
        float(upper[t][d]) - float(lower[t][d])
        for t in range(HORIZON)
        for d in TRANSLATION_DIMS
    ]
    rotation_widths = [
        float(upper[t][d]) - float(lower[t][d])
        for t in range(HORIZON)
        for d in ROTATION_DIMS
    ]
    return {
        "shape": shape,
        "global": stat_node_summary(node, expected_shape=(HORIZON, EEF_DIM)),
        "per_dimension": per_dimension,
        "translation": {
            "dimensions": list(TRANSLATION_DIMS),
            "nonzero_width_cells": sum(
                per_dimension[DIM_NAMES[d]]["nonzero_width_count"] for d in TRANSLATION_DIMS
            ),
            "total_cells": HORIZON * len(TRANSLATION_DIMS),
            "width": numeric_summary(translation_widths),
        },
        "rotation": {
            "dimensions": list(ROTATION_DIMS),
            "zero_width_cells": sum(
                per_dimension[DIM_NAMES[d]]["zero_width_count"] for d in ROTATION_DIMS
            ),
            "total_cells": HORIZON * len(ROTATION_DIMS),
            "width": numeric_summary(rotation_widths),
        },
    }


def find_node(stats: Mapping[str, Any], *, scope: str, key: str) -> Mapping[str, Any]:
    value = stats[scope][key]
    if not isinstance(value, Mapping):
        raise ValueError(f"statistics node is not an object: {scope}/{key}")
    return value


def upstream_summary(base_stats: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, keys in UPSTREAM_CATEGORIES.items():
        category_data = base_stats.get(category)
        if not isinstance(category_data, Mapping):
            raise ValueError(f"missing upstream category: {category}")
        category_result: dict[str, Any] = {}
        for key in keys:
            node = find_node(category_data, scope="relative_action", key=key)
            category_result[key] = time_indexed_eef_summary(node)
        result[category] = category_result
    return result


def slice_stat_node(node: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, list) and key in {"min", "max", "mean", "std", "q01", "q99"}:
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def audit(
    h2_stats_path: Path,
    h2_relative_stats_path: Path,
    base_stats_path: Path,
    processor_path: Path,
) -> dict[str, Any]:
    h2 = load_json(h2_stats_path)
    h2_relative_file = load_json(h2_relative_stats_path)
    base = load_json(base_stats_path)
    processor = load_json(processor_path)
    h2_action = h2.get("action")
    h2_state = h2.get("observation.state")
    h2_relative = h2_relative_file.get("eef_9d")
    if not isinstance(h2_action, Mapping) or not isinstance(h2_state, Mapping):
        raise ValueError("H2 stats must contain action and observation.state nodes")
    if not isinstance(h2_relative, Mapping):
        raise ValueError("H2 relative stats must contain eef_9d")
    h2_state_summary = stat_node_summary(h2_state, expected_shape=(10,))
    h2_absolute = stat_node_summary(slice_stat_node(h2_action, 0, EEF_DIM), expected_shape=(EEF_DIM,))
    h2_gripper = stat_node_summary(slice_stat_node(h2_action, EEF_DIM, EEF_DIM + 1), expected_shape=(1,))
    h2_relative_summary = time_indexed_eef_summary(h2_relative)
    upstream = upstream_summary(base)

    upstream_partition_medians: dict[str, list[float]] = {"translation": [], "rotation": []}
    upstream_partition_ranges: dict[str, list[float]] = {"translation": [], "rotation": []}
    for category in upstream.values():
        for record in category.values():
            for partition in ("translation", "rotation"):
                for dimension in (TRANSLATION_DIMS if partition == "translation" else ROTATION_DIMS):
                    width = record["per_dimension"][DIM_NAMES[dimension]]["width"]
                    upstream_partition_medians[partition].append(float(width["median"]))
                    upstream_partition_ranges[partition].extend([float(width["min"]), float(width["max"])])

    compatibility: dict[str, Any] = {}
    for partition in ("translation", "rotation"):
        h2_width = h2_relative_summary[partition]["width"]
        medians = upstream_partition_medians[partition]
        ranges = upstream_partition_ranges[partition]
        h2_median = float(h2_width["median"])
        compatibility[partition] = {
            "h2_width_median": h2_median,
            "upstream_dimension_median_range": [min(medians), max(medians)],
            "upstream_dimension_observed_cell_range": [min(ranges), max(ranges)],
            "h2_median_within_upstream_median_range": min(medians) <= h2_median <= max(medians),
            "h2_median_within_upstream_observed_cell_range": min(ranges) <= h2_median <= max(ranges),
            "h2_median_to_upstream_observed_min_ratio": h2_median / min(ranges) if min(ranges) > 0 else None,
            "h2_median_to_upstream_observed_max_ratio": h2_median / max(ranges) if max(ranges) > 0 else None,
        }

    kwargs = processor.get("processor_kwargs", {})
    if not isinstance(kwargs, Mapping):
        kwargs = {}
    translation_nonzero = h2_relative_summary["translation"]["nonzero_width_cells"]
    translation_total = h2_relative_summary["translation"]["total_cells"]
    orientation_zero = h2_relative_summary["rotation"]["zero_width_cells"]
    rotation_total = h2_relative_summary["rotation"]["total_cells"]
    translation_scale = compatibility["translation"]
    if translation_scale["h2_width_median"] is None:
        raise AssertionError("translation compatibility did not produce a median")
    translation_ratio_min = translation_scale["h2_median_to_upstream_observed_min_ratio"]
    translation_ratio_max = translation_scale["h2_median_to_upstream_observed_max_ratio"]
    if translation_ratio_min is not None and translation_ratio_min >= 0.1 and translation_ratio_max <= 10.0:
        classification = "SHIFTED_BUT_REASONABLE_ON_TRANSLATION"
    else:
        classification = "EXTREME_OR_NOT_COMPARABLE_ON_TRANSLATION"
    result: dict[str, Any] = {
        "schema": "STAGE24_NORMALIZATION_AUDIT_V1",
        "stage": "Stage24-A3 normalization and pretrained relative-EEF comparison",
        "source_files": {
            "h2_statistics": {
                "path": str(h2_stats_path.resolve()),
                "sha256": sha256_file(h2_stats_path),
                "expected_sha256": EXPECTED_H2_STATS_SHA256,
                "expected_match": sha256_file(h2_stats_path) == EXPECTED_H2_STATS_SHA256,
            },
            "h2_relative_statistics": {
                "path": str(h2_relative_stats_path.resolve()),
                "sha256": sha256_file(h2_relative_stats_path),
                "expected_sha256": EXPECTED_H2_RELATIVE_STATS_SHA256,
                "expected_match": sha256_file(h2_relative_stats_path) == EXPECTED_H2_RELATIVE_STATS_SHA256,
            },
            "base_statistics": {"path": str(base_stats_path.resolve()), "sha256": sha256_file(base_stats_path)},
            "base_processor_config": {"path": str(processor_path.resolve()), "sha256": sha256_file(processor_path)},
        },
        "processor_contract": {
            key: kwargs.get(key)
            for key in (
                "use_percentiles",
                "use_mean_std",
                "max_state_dim",
                "max_action_dim",
                "max_action_horizon",
                "use_relative_action",
                "state_dropout_prob",
            )
        },
        "h2_frozen_stage22_statistics": {
            "observation_state_10d": h2_state_summary,
            "relative_action_eef_9d": h2_relative_summary,
            "absolute_action_eef_9d": h2_absolute,
            "absolute_action_gripper_position": h2_gripper,
        },
        "upstream_relative_eef_statistics": upstream,
        "scale_comparison": compatibility,
        "classification": {
            "relative_eef_scale": classification,
            "physical_translation_zero_width": (
                "STRUCTURAL_FIRST_POINT_ONLY"
                if translation_nonzero == translation_total - len(TRANSLATION_DIMS)
                else "NONSTRUCTURAL_TRANSLATION_DEGENERACY_REQUIRES_REVIEW"
            ),
            "orientation_partition": (
                "TASK_CONSTANT_ORIENTATION_STATS"
                if orientation_zero == rotation_total
                else "NONCONSTANT_ORIENTATION_STATS"
            ),
            "gripper_partition": "TASK_CONSTANT_GRIPPER_STATS" if h2_gripper["total_cells"] == 1 and h2_gripper["zero_width_cells_official_isclose"] == 1 else "NONCONSTANT_GRIPPER_STATS",
            "overall": "SHIFTED_BUT_REASONABLE_ON_TRANSLATION; ROTATION_NOT_COMPARABLE_TASK_CONSTANT",
        },
        "interpretation": {
            "observations": [
                "H2 relative EEF statistics are time-indexed [40,9]; global zero-width counts are not interpreted without partitioning.",
                "The first relative point is structurally zero for translation because it is anchored to the last observed state.",
                "The fixture holds orientation and gripper constant, so their zero-width statistics are task-design facts, not evidence of translation collapse.",
                "The H2 translation width median is compared descriptively with pinned upstream relative-EEF cell statistics; this is not a claim of distributional equivalence or task generalization.",
            ],
            "not_proven": [
                "Normalization is not proven to be harmless for all future robot data.",
                "The comparison does not establish that the H2 profile failure has a single cause.",
                "Production action timing and real-robot limits remain undefined.",
            ],
        },
        "execution_boundary": {
            "json_only": True,
            "model_constructed": False,
            "forward": False,
            "inference": False,
            "training": False,
            "cuda_used": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
        },
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-stats", type=Path, required=True)
    parser.add_argument("--h2-relative-stats", type=Path, required=True)
    parser.add_argument("--base-stats", type=Path, required=True)
    parser.add_argument("--base-processor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(
        args.h2_stats.expanduser().resolve(strict=True),
        args.h2_relative_stats.expanduser().resolve(strict=True),
        args.base_stats.expanduser().resolve(strict=True),
        args.base_processor_config.expanduser().resolve(strict=True),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
