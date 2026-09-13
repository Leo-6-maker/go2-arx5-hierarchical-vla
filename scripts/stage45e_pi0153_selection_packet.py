#!/usr/bin/env python3
"""Build the PI0153 CPU-only route/tier/seed selection packet.

The input is the sealed PI0152 campaign.  This postprocessor reads existing
E4 rows and D1/D2A derived records, validates their source bindings, and
emits small summaries plus a 432-row physical table.  It never imports a
model, opens a checkpoint, invokes a sampler, or changes the sealed source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


PI_GATE = "PI_GATE_STATE_0153"
ISSUE_COMMENT = 5649870843
PR_NUMBER = 66
EXECUTION_HEAD = "eab15ec185203d69c03ab82b7843bcaea9200281"
EXPECTED_ANALYSIS_HEAD = "c967a8cf8e129652754bd5e86dc3fd091a13a4be"
EXPECTED_WINDOWS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in (
        "red_target_train_a",
        "red_target_train_b",
        "red_target_train_c",
        "red_target_train_d",
    )
    for index in range(6)
)
TARGET_IDS = tuple(window.split(":", 1)[0].removeprefix("stage22_expert_") for window in EXPECTED_WINDOWS[::6])
ROUTES = ("legacy", "n2", "l1")
SEEDS = (2222, 2223, 2224)
TIERS = (1000, 2000)

PHYSICAL_METRICS = (
    "final_target_distance_m",
    "absolute_progress_m",
    "minimum_target_distance_m",
    "prediction_path_length_m",
    "roughness_ratio",
    "endpoint_target_vector_cosine",
    "endpoint_error_m",
    "position_rmse_m",
    "best_progress_m",
)

METRIC_DIRECTIONS = {
    "final_target_distance_m": "lower_is_better_descriptively",
    "absolute_progress_m": "higher_is_better_descriptively",
    "minimum_target_distance_m": "lower_is_better_descriptively",
    "prediction_path_length_m": "descriptive_no_quality_threshold",
    "roughness_ratio": "lower_is_better_descriptively",
    "endpoint_target_vector_cosine": "higher_is_better_descriptively",
    "endpoint_error_m": "lower_is_better_descriptively",
    "position_rmse_m": "lower_is_better_descriptively",
    "best_progress_m": "higher_is_better_descriptively",
}

EXPECTED_DIAGNOSTIC_ROWS = {
    "D1": {"legacy": 864, "l1": 1296, "n2": 1296},
    "D2A": {"legacy": 384, "l1": 576, "n2": 576},
}


class SelectionPacketError(RuntimeError):
    """Raised when sealed evidence cannot support the requested packet."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionPacketError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionPacketError(f"cannot read JSON: {path}") from exc


def sha256_file(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is non-finite")
    return result


def label_for(route: str, seed: int, tier: int) -> str:
    return f"{route.upper()}(seed{seed})@{tier}"


def route_tier_key(route: str, tier: int) -> str:
    return f"{route.upper()}@{tier}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing row source: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # The retained historical D2B source is JSONL rather than a JSON
        # array. Accept that immutable storage format without broadening the
        # row schema or silently skipping malformed lines.
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SelectionPacketError(f"malformed JSONL row: {path}:{line_number}") from exc
        value = values
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = [dict(value)]
    else:
        raise SelectionPacketError(f"row source is not a list/object: {path}")
    require(all(isinstance(row, Mapping) for row in rows), f"row source contains non-object: {path}")
    return [dict(row) for row in rows]


def _canonical_target_id(window_id: str, target_id: Any, label: str) -> str:
    expected = window_id.split(":", 1)[0].removeprefix("stage22_expert_")
    actual = str(target_id)
    allowed = {expected, f"stage22_expert_{expected}"}
    require(actual in allowed, f"target identity drift: {label}/{window_id}: {actual}")
    return expected


def _source_row_target_id(row: Mapping[str, Any], window_id: str) -> Any:
    if row.get("target_id") is not None:
        return row.get("target_id")
    identity = row.get("identity")
    if isinstance(identity, Mapping) and identity.get("target_id") is not None:
        return identity.get("target_id")
    # Historical L1 rows intentionally carry the target only in window_id.
    return window_id.split(":", 1)[0]


def _source_target_exclusion(row: Mapping[str, Any]) -> Any:
    if "target_xyz_in_model_input" in row:
        return row.get("target_xyz_in_model_input")
    identity = row.get("identity")
    if isinstance(identity, Mapping):
        return identity.get("target_xyz_in_model_input")
    return None


def _historical_legacy_e4_rows(rows: Sequence[Mapping[str, Any]], tier: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            continue
        if (
            identity.get("solver") == "E4"
            and identity.get("checkpoint") == f"FULL24@{tier}"
            and identity.get("final_stage") is True
        ):
            selected.append(dict(row))
    return selected


def _source_row_label(row: Mapping[str, Any]) -> str | None:
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("label")
        if isinstance(value, str):
            return value
    identity = row.get("identity")
    if isinstance(identity, Mapping):
        value = identity.get("checkpoint")
        if isinstance(value, str):
            return value
    return None


def _source_row_window(row: Mapping[str, Any]) -> str | None:
    value = row.get("window_id")
    if isinstance(value, str):
        return value
    identity = row.get("identity")
    if isinstance(identity, Mapping) and isinstance(identity.get("window_id"), str):
        return str(identity["window_id"])
    return None


def _source_physical_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        # Retained D2B Legacy E4 rows store the same physical trajectory
        # metrics under physical.free_running_endpoint.trajectory_metrics.
        physical = row.get("physical")
        endpoint = physical.get("free_running_endpoint") if isinstance(physical, Mapping) else None
        metrics = endpoint.get("trajectory_metrics") if isinstance(endpoint, Mapping) else None
    require(isinstance(metrics, Mapping), "source row has no metrics namespace")
    target = metrics.get("target")
    require(isinstance(target, Mapping), "source row has no target metric namespace")
    cosine_value = target.get("endpoint_target_vector_cosine")
    cosine = None if cosine_value is None else finite_number(cosine_value, "endpoint_target_vector_cosine")
    return {
        "final_target_distance_m": finite_number(target.get("final_target_distance_m"), "final_target_distance_m"),
        "absolute_progress_m": finite_number(target.get("absolute_progress_m"), "absolute_progress_m"),
        "minimum_target_distance_m": finite_number(target.get("minimum_target_distance_m"), "minimum_target_distance_m"),
        "prediction_path_length_m": finite_number(metrics.get("prediction_path_length_m"), "prediction_path_length_m"),
        "roughness_ratio": finite_number(metrics.get("roughness_ratio"), "roughness_ratio"),
        "endpoint_target_vector_cosine": cosine,
        "endpoint_error_m": finite_number(metrics.get("endpoint_error_m"), "endpoint_error_m"),
        "position_rmse_m": finite_number(metrics.get("position_rmse_m"), "position_rmse_m"),
        "best_progress_m": finite_number(target.get("best_progress_m"), "best_progress_m"),
    }


def _source_progress_sign(row: Mapping[str, Any]) -> str:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        physical = row.get("physical")
        endpoint = physical.get("free_running_endpoint") if isinstance(physical, Mapping) else None
        metrics = endpoint.get("trajectory_metrics") if isinstance(endpoint, Mapping) else None
    require(isinstance(metrics, Mapping), "source row has no progress metrics namespace")
    target = metrics.get("target")
    require(isinstance(target, Mapping), "source row has no progress target namespace")
    value = target.get("progress_sign")
    require(isinstance(value, str) and value, "source row has no progress sign")
    return value


def _report_physical_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    physical = row.get("physical_metrics")
    require(isinstance(physical, Mapping), "physical report row has no physical_metrics")
    target = physical.get("target")
    require(isinstance(target, Mapping), "physical report row has no target metrics")
    return {
        "final_target_distance_m": finite_number(target.get("final_target_distance_m"), "report final_target_distance_m"),
        "absolute_progress_m": finite_number(target.get("absolute_progress_m"), "report absolute_progress_m"),
        "minimum_target_distance_m": finite_number(target.get("minimum_target_distance_m"), "report minimum_target_distance_m"),
        "prediction_path_length_m": finite_number(physical.get("prediction_path_length_m"), "report prediction_path_length_m"),
        "roughness_ratio": finite_number(physical.get("roughness_ratio"), "report roughness_ratio"),
        "endpoint_target_vector_cosine": None,
        "endpoint_error_m": finite_number(physical.get("endpoint_error_m"), "report endpoint_error_m"),
        "position_rmse_m": finite_number(physical.get("position_rmse_m"), "report position_rmse_m"),
        "best_progress_m": finite_number(target.get("best_progress_m"), "report best_progress_m"),
    }


def _assert_same_metric(left: float | None, right: float | None, label: str) -> None:
    if left is None or right is None:
        require(left is None and right is None, f"{label} nullability drift")
        return
    require(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12), f"{label} source/report drift")


def _summary(values: Iterable[float]) -> dict[str, Any]:
    data = [finite_number(value, "summary value") for value in values]
    require(data, "cannot summarize an empty metric set")
    return {
        "count": len(data),
        "mean": float(statistics.fmean(data)),
        "median": float(statistics.median(data)),
        "min": float(min(data)),
        "max": float(max(data)),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(rows, "cannot summarize empty physical rows")
    metrics: dict[str, Any] = {}
    for name in PHYSICAL_METRICS:
        values = [row["metrics"][name] for row in rows if row["metrics"].get(name) is not None]
        metrics[name] = _summary(values) if values else None
    signs = Counter(str(row["progress_sign"]) for row in rows)
    return {
        "row_count": len(rows),
        "metrics": metrics,
        "progress_sign_counts": dict(sorted(signs.items())),
    }


def _git_identity(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, f"git command failed: git {' '.join(args)}")
        return result.stdout.strip()

    return {
        "repository": str(repo),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "working_tree_clean": git("status", "--porcelain") == "",
    }


def _source_rows_for_arm(
    arm: Mapping[str, Any],
    source_cache: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = Path(str(arm.get("rows_path"))).resolve()
    require(path.is_file() and not path.is_symlink(), f"arm source rows unavailable: {path}")
    cache_key = str(path)
    if cache_key not in source_cache:
        source_cache[cache_key] = _read_rows(path)
    rows = source_cache[cache_key]
    source_kind = str(arm.get("source_kind"))
    tier = int(arm["checkpoint_step"])
    if source_kind.startswith("HISTORICAL_LEGACY_D2B_E4"):
        rows = _historical_legacy_e4_rows(rows, tier)
    require(len(rows) == 24, f"source row cardinality drift for {arm.get('label')}: {len(rows)}")
    by_window: dict[str, dict[str, Any]] = {}
    for row in rows:
        window = _source_row_window(row)
        require(window is not None, f"source row has no window id: {path}")
        require(window not in by_window, f"duplicate source window {window}: {path}")
        by_window[window] = row
    return by_window, {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_kind": source_kind,
        "selected_row_count": len(rows),
        "selected_window_ids": list(by_window),
    }


def build_physical_rows(campaign_root: Path, report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arms = report.get("arm_summaries")
    require(isinstance(arms, list), "physical report arm_summaries missing")
    source_cache: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    physical_rows: list[dict[str, Any]] = []
    expected_labels = {
        label_for(route, seed, tier)
        for route in ROUTES
        for seed in SEEDS
        for tier in TIERS
    }
    observed_labels: set[str] = set()
    for arm in arms:
        require(isinstance(arm, Mapping), "physical report arm is not an object")
        label = str(arm.get("label"))
        require(label not in observed_labels, f"duplicate arm in physical report: {label}")
        observed_labels.add(label)
        route = str(arm.get("route"))
        seed = int(arm.get("seed"))
        tier = int(arm.get("checkpoint_step"))
        require(route in ROUTES and seed in SEEDS and tier in TIERS, f"unexpected arm identity: {label}")
        require(label == label_for(route, seed, tier), f"arm label identity drift: {label}")
        source_by_window, source_info = _source_rows_for_arm(arm, source_cache)
        source_files[source_info["path"]] = source_info
        report_rows = arm.get("per_window")
        require(isinstance(report_rows, list) and len(report_rows) == 24, f"report row count drift: {label}")
        ordered = sorted(report_rows, key=lambda row: int(row.get("window_index", -1)))
        require([row.get("window_id") for row in ordered] == list(EXPECTED_WINDOWS), f"window order drift: {label}")
        for index, report_row in enumerate(ordered):
            require(isinstance(report_row, Mapping), f"report row is not an object: {label}/{index}")
            window = str(report_row.get("window_id"))
            source_row = source_by_window.get(window)
            require(source_row is not None, f"source window missing for {label}: {window}")
            source_sha = canonical_json_sha256(source_row)
            require(source_sha == report_row.get("source_row_sha256"), f"source row SHA drift: {label}/{window}")
            canonical_target = _canonical_target_id(window, _source_row_target_id(source_row, window), label)
            _canonical_target_id(window, report_row.get("target_id"), label)
            require(_source_target_exclusion(source_row) is False, f"target exclusion drift: {label}/{window}")
            source_metrics = _source_physical_metrics(source_row)
            reported_metrics = _report_physical_metrics(report_row)
            report_target = report_row["physical_metrics"]["target"]
            for metric in PHYSICAL_METRICS:
                # The accepted PI0152 report predates endpoint-cosine
                # aggregation. Preserve and expose the source metric rather
                # than treating that legacy omission as a source/report null.
                if metric == "endpoint_target_vector_cosine" and "endpoint_target_vector_cosine" not in report_target:
                    continue
                _assert_same_metric(source_metrics[metric], reported_metrics[metric], f"{label}/{window}/{metric}")
            physical_rows.append(
                {
                    "arm": label,
                    "route": route,
                    "seed": seed,
                    "tier": tier,
                    "window_index": index,
                    "window_id": window,
                    "target_id": canonical_target,
                    "source_target_id": str(source_row.get("target_id")),
                    "report_target_id": str(report_row.get("target_id")),
                    "report_omits_endpoint_target_vector_cosine": "endpoint_target_vector_cosine" not in report_target,
                    "source_kind": str(report_row.get("source_kind")),
                    "source_rows_path": source_info["path"],
                    "source_rows_sha256": source_info["sha256"],
                    "source_row_sha256": source_sha,
                    "raw_path": str(arm.get("raw_path")),
                    "raw_sha256": str(arm.get("raw_sha256")),
                    "stats_route": report_row.get("stats_route"),
                    "checkpoint_tree_sha256": report_row.get("checkpoint_tree_sha256"),
                    "physical_metrics_namespace": report_row.get("physical_metrics_namespace"),
                    "progress_sign": _source_progress_sign(source_row),
                    "metrics": source_metrics,
                }
            )
    require(observed_labels == expected_labels, f"cohort mismatch: missing={sorted(expected_labels-observed_labels)} extra={sorted(observed_labels-expected_labels)}")
    require(len(physical_rows) == 432, f"physical row count drift: {len(physical_rows)}")
    keys = [(row["arm"], row["window_id"]) for row in physical_rows]
    require(len(keys) == len(set(keys)), "duplicate physical row keys")
    source_row_hashes = [row["source_row_sha256"] for row in physical_rows]
    require(len(source_row_hashes) == len(set(source_row_hashes)), "duplicate source row hashes across physical rows")
    return physical_rows, {
        "source_files": sorted(source_files.values(), key=lambda item: item["path"]),
        "arm_count": len(observed_labels),
        "physical_row_count": len(physical_rows),
        "unique_row_key_count": len(set(keys)),
        "unique_source_row_hash_count": len(set(source_row_hashes)),
    }


def _rows_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["arm"])].append(row)
    for label, group in result.items():
        require(len(group) == 24, f"arm row count drift in summaries: {label}")
        require([row["window_id"] for row in sorted(group, key=lambda item: item["window_index"])] == list(EXPECTED_WINDOWS), f"arm window order drift in summaries: {label}")
    return dict(result)


def build_stratified_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    arm_summaries = []
    for route in ROUTES:
        for tier in TIERS:
            for seed in SEEDS:
                label = label_for(route, seed, tier)
                group = by_arm[label]
                arm_summaries.append(
                    {
                        "arm": label,
                        "route": route,
                        "tier": tier,
                        "seed": seed,
                        "stats_route": group[0]["stats_route"],
                        "checkpoint_tree_sha256": group[0]["checkpoint_tree_sha256"],
                        "source_kind_counts": dict(sorted(Counter(row["source_kind"] for row in group).items())),
                        **_summarize_rows(group),
                    }
                )
    route_tier_summaries = []
    for route in ROUTES:
        for tier in TIERS:
            seed_summaries = [
                next(item for item in arm_summaries if item["arm"] == label_for(route, seed, tier))
                for seed in SEEDS
            ]
            group = [row for row in rows if row["route"] == route and row["tier"] == tier]
            route_tier_summaries.append(
                {
                    "route": route,
                    "tier": tier,
                    "route_tier": route_tier_key(route, tier),
                    "row_count": len(group),
                    "seed_count": len(seed_summaries),
                    "seed_summaries": seed_summaries,
                    "three_seed_summary": _summarize_rows(group),
                }
            )
    return {
        "schema": "VANDA_STAGE45E_PI0153_STRATIFIED_PHYSICAL_SUMMARIES_V1",
        "status": "PASS_READONLY_FROM_SOURCE_BOUND_E4_ROWS",
        "arm_summaries": arm_summaries,
        "route_tier_summaries": route_tier_summaries,
        "tier_mixing": False,
        "seed_mixing_within_arm": False,
        "winner_selection": False,
    }


def _pair_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for route in ("l1", "n2"):
        for tier in TIERS:
            for seed in SEEDS:
                specs.append(
                    {
                        "comparison_class": "intervention_vs_legacy",
                        "left_route": route,
                        "right_route": "legacy",
                        "left_tier": tier,
                        "right_tier": tier,
                        "left_seed": seed,
                        "right_seed": seed,
                    }
                )
    for route in ROUTES:
        for seed in SEEDS:
            specs.append(
                {
                    "comparison_class": "exposure",
                    "left_route": route,
                    "right_route": route,
                    "left_tier": 2000,
                    "right_tier": 1000,
                    "left_seed": seed,
                    "right_seed": seed,
                }
            )
    for tier in TIERS:
        for seed in SEEDS:
            specs.append(
                {
                    "comparison_class": "l1_vs_n2",
                    "left_route": "l1",
                    "right_route": "n2",
                    "left_tier": tier,
                    "right_tier": tier,
                    "left_seed": seed,
                    "right_seed": seed,
                }
            )
    require(len(specs) == 27, f"pair specification count drift: {len(specs)}")
    require(Counter(item["comparison_class"] for item in specs) == Counter({"intervention_vs_legacy": 12, "exposure": 9, "l1_vs_n2": 6}), "pair class count drift")
    return specs


def _pair_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in PHYSICAL_METRICS:
        values = [finite_number(row[metric], f"pair delta {metric}") for row in rows if row.get(metric) is not None]
        result[metric] = _summary(values) if values else None
    return result


def build_paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    comparisons = []
    for spec in _pair_specs():
        left_label = label_for(spec["left_route"], spec["left_seed"], spec["left_tier"])
        right_label = label_for(spec["right_route"], spec["right_seed"], spec["right_tier"])
        left = sorted(by_arm[left_label], key=lambda item: item["window_index"])
        right = sorted(by_arm[right_label], key=lambda item: item["window_index"])
        require([row["window_id"] for row in left] == [row["window_id"] for row in right], f"pair window mismatch: {left_label} vs {right_label}")
        deltas: list[dict[str, Any]] = []
        for left_row, right_row in zip(left, right):
            delta = {
                metric: (
                    None
                    if left_row["metrics"][metric] is None or right_row["metrics"][metric] is None
                    else float(left_row["metrics"][metric]) - float(right_row["metrics"][metric])
                )
                for metric in PHYSICAL_METRICS
            }
            deltas.append(
                {
                    "window_id": left_row["window_id"],
                    "target_id": left_row["target_id"],
                    "left_arm": left_label,
                    "right_arm": right_label,
                    "delta_left_minus_right": delta,
                }
            )
        by_target = {}
        for target in TARGET_IDS:
            target_deltas = [row["delta_left_minus_right"] for row in deltas if row["target_id"] == target]
            by_target[target] = {"row_count": len(target_deltas), "delta_left_minus_right": _pair_summary(target_deltas)}
        comparison_id = f"{spec['comparison_class']}__{left_label}_minus_{right_label}"
        comparisons.append(
            {
                "comparison_id": comparison_id,
                **spec,
                "left_arm": left_label,
                "right_arm": right_label,
                "pairing_key": ["window_id", "target_id", "seed", "tier"],
                "window_count": len(deltas),
                "delta_convention": "left_minus_right",
                "overall": {"row_count": len(deltas), "delta_left_minus_right": _pair_summary([row["delta_left_minus_right"] for row in deltas])},
                "by_target": by_target,
            }
        )
    return {
        "schema": "VANDA_STAGE45E_PI0153_PAIRED_COMPARISONS_V1",
        "status": "PASS_DESCRIPTIVE_SAME_WINDOW_PAIRS",
        "pairing_contract": {
            "intervention_vs_legacy": 12,
            "exposure": 9,
            "l1_vs_n2": 6,
            "total": 27,
            "same_seed_window_tier_required": True,
            "tiers_mixed_within_comparison": False,
            "new_weighted_score": False,
            "significance_test": False,
        },
        "metric_directions": dict(METRIC_DIRECTIONS),
        "comparisons": comparisons,
    }


def _diagnostic_metric_summary(records: Sequence[Mapping[str, Any]], namespace: str) -> dict[str, Any]:
    names = ("rmse", "cosine", "l2", "mae", "max_abs")
    result: dict[str, Any] = {}
    for name in names:
        values = [
            finite_number(record[namespace].get(name), f"{namespace}.{name}")
            for record in records
            if isinstance(record.get(namespace), Mapping) and record[namespace].get(name) is not None
        ]
        result[name] = _summary(values) if values else None
    return result


def build_phase_diagnostic_summary(campaign_root: Path) -> dict[str, Any]:
    units: dict[str, Any] = {}
    for phase in ("D1", "D2A"):
        phase_units: dict[str, Any] = {}
        for route in ROUTES:
            unit_path = campaign_root / "evidence_units" / f"{phase}_{route}.json"
            unit = read_json(unit_path)
            require(unit.get("phase") == phase and unit.get("route") == route, f"diagnostic unit identity drift: {phase}:{route}")
            attempt_root = Path(str(unit.get("attempt_root"))).resolve()
            require(attempt_root.is_dir(), f"diagnostic attempt root unavailable: {attempt_root}")
            derived_paths = sorted(attempt_root.glob("probes/*/w*/seed*/t*/derived_metrics.json"))
            require(len(derived_paths) == EXPECTED_DIAGNOSTIC_ROWS[phase][route], f"{phase}:{route} derived row count drift: {len(derived_paths)}")
            records: list[dict[str, Any]] = []
            key_set: set[tuple[Any, ...]] = set()
            by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for path in derived_paths:
                derived = read_json(path)
                require(derived.get("status") == "DERIVED_AFTER_RAW_SEAL", f"derived status drift: {path}")
                require(derived.get("raw_before_derived") is True, f"raw-before-derived drift: {path}")
                raw_manifest_path = path.parent / "raw_manifest.json"
                raw_manifest = read_json(raw_manifest_path)
                require(raw_manifest.get("status") == "RAW_SEALED_BEFORE_DERIVED", f"raw seal status drift: {raw_manifest_path}")
                identity = raw_manifest.get("identity")
                require(isinstance(identity, Mapping), f"raw identity missing: {raw_manifest_path}")
                require(identity.get("target_xyz_in_model_input") is False, f"target exclusion drift: {raw_manifest_path}")
                key = (identity.get("checkpoint"), identity.get("window_id"), identity.get("noise_seed"), identity.get("t_bucket"))
                require(key not in key_set, f"duplicate diagnostic probe key: {key}")
                key_set.add(key)
                record = {
                    "checkpoint": str(identity.get("checkpoint")),
                    "window_id": str(identity.get("window_id")),
                    "target_id": str(identity.get("target_id")),
                    "noise_seed": identity.get("noise_seed"),
                    "t": identity.get("t"),
                    "t_bucket": identity.get("t_bucket"),
                    "derived_path": str(path),
                    "derived_sha256": sha256_file(path),
                    "raw_manifest_path": str(raw_manifest_path),
                    "raw_manifest_sha256": sha256_file(raw_manifest_path),
                    "raw_arrays_path": str(path.parent / "raw_arrays.npz"),
                    "status": derived.get("status"),
                    "raw_before_derived": derived.get("raw_before_derived"),
                }
                for namespace in (
                    "normalized_local_flow",
                    "state_path_divergence",
                    "field_sensitivity",
                    "teacher_bridge_local_error",
                    "off_path_diagnostic",
                ):
                    if namespace in derived:
                        record[namespace] = dict(derived[namespace])
                records.append(record)
                by_arm[str(identity.get("checkpoint"))].append(record)
            require(records, f"empty diagnostic records: {phase}:{route}")
            phase_metrics = {}
            namespaces = ("normalized_local_flow",) if phase == "D1" else (
                "state_path_divergence",
                "field_sensitivity",
                "teacher_bridge_local_error",
                "off_path_diagnostic",
            )
            for namespace in namespaces:
                phase_metrics[namespace] = _diagnostic_metric_summary(records, namespace)
            arm_metrics = {}
            for arm, arm_records in sorted(by_arm.items()):
                arm_metrics[arm] = {
                    "row_count": len(arm_records),
                    "metric_summary": {namespace: _diagnostic_metric_summary(arm_records, namespace) for namespace in namespaces},
                }
            phase_units[route] = {
                "unit_path": str(unit_path),
                "unit_sha256": sha256_file(unit_path),
                "attempt_root": str(attempt_root),
                "derived_row_count": len(records),
                "checkpoint_arm_count": len(by_arm),
                "phase_metrics_overall": phase_metrics,
                "by_arm": arm_metrics,
                "raw_before_derived_all": all(record["raw_before_derived"] is True for record in records),
                "unique_probe_key_count": len(key_set),
                "inner_seal": dict(unit.get("inner_seal", {})) if isinstance(unit.get("inner_seal"), Mapping) else None,
                "unified_seal": dict(unit.get("unified_seal", {})) if isinstance(unit.get("unified_seal"), Mapping) else None,
            }
        units[phase] = phase_units
    return {
        "schema": "VANDA_STAGE45E_PI0153_D1_D2A_DIAGNOSTIC_SUMMARY_V1",
        "status": "PASS_READONLY_FROM_SEALED_DERIVED_RECORDS",
        "phases": units,
        "model_execution": False,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), f"artifact unavailable: {path}")
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = None
    return {"path": str(path), "relative_to_campaign_root": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_source_inventory(campaign_root: Path, report: Mapping[str, Any], physical_source_info: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "final_result.json",
        "cpu_summary.json",
        "CPU_EVIDENCE_INVENTORY.json",
        "SHA256SUMS",
        "physical_metrics_report.json",
        "campaign_admission.json",
        "canonical_row_index.json",
        "C10_HANDOFF.json",
        "external_evidence_inventory.json",
        "PROVENANCE_REPAIR.json",
        "historical_diagnostic_evidence_audit.json",
        "lineage_plan.json",
    )
    artifacts = [_artifact_record(campaign_root / name, campaign_root) for name in names]
    resource_canary = campaign_root / "resource_canary" / "legacy_resource_canary.json"
    if resource_canary.is_file():
        artifacts.append(_artifact_record(resource_canary, campaign_root))
    corrections = campaign_root.parent.parent / "provenance" / "STAGE45E_H200_ALLOCATION_LEDGER_PI0143.jsonl.corrections.jsonl"
    if corrections.is_file():
        artifacts.append(_artifact_record(corrections, campaign_root))
    unit_records = []
    for phase in ("E4", "D1", "D2A"):
        for route in ROUTES:
            path = campaign_root / "evidence_units" / f"{phase}_{route}.json"
            unit = _artifact_record(path, campaign_root)
            unit["phase"] = phase
            unit["route"] = route
            unit_records.append(unit)
    return {
        "schema": "VANDA_STAGE45E_PI0153_SOURCE_INVENTORY_V1",
        "status": "PASS_SOURCE_BOUND_READONLY",
        "campaign_root": str(campaign_root),
        "sealed_campaign_artifacts": artifacts,
        "physical_row_source_files": list(physical_source_info["source_files"]),
        "evidence_unit_files": unit_records,
        "canonical_row_index_semantics": "PHASE_REFERENCE_INDEX_NOT_PHYSICAL_METRICS",
        "raw_arrays_included": False,
        "checkpoint_payloads_included": False,
        "corelab_or_receiver_dependency": False,
    }


def build_row_call_accounting(campaign_root: Path, report: Mapping[str, Any], final_result: Mapping[str, Any], cpu_summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_kinds = Counter(str(row["source_kind"]) for row in rows)
    current_direct = source_kinds.get("CURRENT_H200_SEALED_E4", 0)
    recovery = source_kinds.get("CURRENT_H200_RECOVERY_SOURCE", 0)
    historical = sum(count for kind, count in source_kinds.items() if kind.startswith("HISTORICAL_"))
    cohort = report.get("cohort")
    require(isinstance(cohort, Mapping), "physical report cohort missing")
    require(current_direct == 312, f"current direct row count drift: {current_direct}")
    require(recovery == 24, f"recovery row count drift: {recovery}")
    require(historical == 96, f"historical row count drift: {historical}")
    require(current_direct + recovery == cohort.get("current_phase_rows") == 336, "current phase/recovery row accounting drift")
    require(cohort.get("current_recovery_source_rows") == recovery, "recovery row declaration drift")
    require(cohort.get("historical_readonly_rows") == historical, "historical row declaration drift")
    require(len(rows) == current_direct + recovery + historical == 432, "physical row accounting total drift")
    canonical_index = read_json(campaign_root / "canonical_row_index.json")
    require(canonical_index.get("entry_count") == 1296, "canonical index entry count drift")
    require(canonical_index.get("index_kind") == "PHASE_REFERENCE_INDEX_NOT_PHYSICAL_METRICS", "canonical index semantics drift")
    require(canonical_index.get("physical_metric_rows_in_index") == 0, "canonical index physical-row claim drift")
    accounting = cpu_summary.get("diagnostic_accounting")
    require(isinstance(accounting, Mapping), "diagnostic accounting missing")
    accepted = accounting.get("accepted_unique_scientific_calls")
    all_attempted = accounting.get("all_attempted_model_calls")
    failed = accounting.get("failed_attempt_model_calls")
    overlap = accounting.get("overlapped_recovery_model_calls")
    require(accepted == 8976 and all_attempted == 8976 and failed == 120 and overlap == 120, "call accounting drift")
    require(final_result.get("training") is False and final_result.get("fresh_training_lineages") == 0, "training boundary drift")
    return {
        "schema": "VANDA_STAGE45E_PI0153_ROW_AND_CALL_ACCOUNTING_V1",
        "status": "PASS_DEDUPLICATED_SOURCE_BOUND_ACCOUNTING",
        "physical_rows": {
            "physical_metric_rows": len(rows),
            "source_kind_counts": dict(sorted(source_kinds.items())),
            "current_direct_rows": current_direct,
            "current_recovery_rows": recovery,
            "current_phase_rows_including_recovery": current_direct + recovery,
            "historical_readonly_rows": historical,
            "additive_formula": "312 current-direct + 24 recovery + 96 historical = 432 unique physical rows",
            "non_additive_warning": "336 current-phase rows already include the 24 recovery rows; do not add 336+24+96",
            "unique_physical_row_keys": len({(row["arm"], row["window_id"]) for row in rows}),
            "canonical_phase_reference_entries": canonical_index.get("entry_count"),
            "canonical_index_is_not_physical_metric_rows": True,
        },
        "model_calls": {
            "accepted_unique_scientific_calls": accepted,
            "all_attempted_model_calls_deduplicated_by_scientific_identity": all_attempted,
            "new_current_allocation_calls": 8856,
            "reused_recovery_source_calls": 120,
            "failed_attempt_model_calls": failed,
            "overlapped_recovery_model_calls": overlap,
            "formula": "8856 new current calls + 120 retained recovery-source calls = 8976 accepted unique calls",
            "recovery_warning": "The 120 failed-attempt calls are the same retained recovery-source identities; do not count them again as new calls.",
        },
    }


def _format(value: Any, digits: int = 9) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_readme(packet: Mapping[str, Any], stratified: Mapping[str, Any], paired: Mapping[str, Any], accounting: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> str:
    lines = [
        "# PI0153 tier/seed selection packet",
        "",
        "CPU/read-only material derived from the sealed PI0152 H200 campaign. No model, checkpoint payload, sampler, GPU, PBS, training, Validation-E, Holdout-F, WBC, simulator, or hardware execution occurred while producing this packet.",
        "",
        f"- Gate: `{PI_GATE}` / Issue comment `{ISSUE_COMMENT}`",
        f"- Science execution HEAD: `{packet['execution_head']}`",
        f"- Analysis HEAD: `{packet['analysis_identity']['head']}`",
        f"- Physical rows: `{accounting['physical_rows']['physical_metric_rows']}` unique rows",
        f"- Pair comparisons: `{paired['pairing_contract']['total']}` ({paired['pairing_contract']['intervention_vs_legacy']} intervention-vs-Legacy, {paired['pairing_contract']['exposure']} exposure, {paired['pairing_contract']['l1_vs_n2']} L1-vs-N2)",
        "",
        "## Row and recovery accounting",
        "",
        "`432 = 312 current-direct + 24 recovery + 96 historical`. The reported `336 current-phase` count includes the 24 recovery rows, so `336 + 24 + 96` is intentionally not an additive formula. The 1296-entry canonical index is a phase-reference index and contains zero physical metric rows.",
        "",
        "The accepted unique call identity count is `8976`: `8856` new current-allocation calls plus `120` retained recovery-source calls. The `120` failed-attempt calls and `120` recovery overlap refer to those same identities and are not added again.",
        "",
        "## 18 arm summaries",
        "",
        "| arm | rows | progress m | final target distance m | minimum distance m | path length m | roughness | endpoint cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in stratified["arm_summaries"]:
        metrics = arm["metrics"]
        get = lambda name: metrics[name]["mean"] if metrics[name] else None
        lines.append(
            "| {arm} | {rows} | {progress} | {final} | {minimum} | {path} | {rough} | {cosine} |".format(
                arm=arm["arm"],
                rows=arm["row_count"],
                progress=_format(get("absolute_progress_m")),
                final=_format(get("final_target_distance_m")),
                minimum=_format(get("minimum_target_distance_m")),
                path=_format(get("prediction_path_length_m")),
                rough=_format(get("roughness_ratio")),
                cosine=_format(get("endpoint_target_vector_cosine")),
            )
        )
    lines.extend(
        [
            "",
            "The table is descriptive and keeps every seed/tier arm separate; it is not a winner selection.",
            "",
            "## D1/D2A read-only diagnostic closure",
            "",
        ]
    )
    for phase in ("D1", "D2A"):
        lines.append(f"### {phase}")
        lines.append("")
        for route in ROUTES:
            item = diagnostic["phases"][phase][route]
            lines.append(f"- `{route}`: {item['derived_row_count']} derived rows, {item['checkpoint_arm_count']} arms, raw-before-derived={item['raw_before_derived_all']}")
            for namespace, summary in item["phase_metrics_overall"].items():
                rmse = summary["rmse"]["mean"] if summary.get("rmse") else None
                cosine = summary["cosine"]["mean"] if summary.get("cosine") else None
                lines.append(f"  - `{namespace}`: rmse mean `{_format(rmse)}`, cosine mean `{_format(cosine)}`")
        lines.append("")
    lines.extend(
        [
            "## Metric/frame definitions",
            "",
            "- `final_target_distance_m`: Euclidean distance from the final decoded EEF position to the target, in world-frame metres.",
            "- `absolute_progress_m`: initial target distance minus final target distance; positive means distance reduction, not a success threshold.",
            "- `minimum_target_distance_m`: minimum decoded-path distance to the target, in metres.",
            "- `prediction_path_length_m`: sum of successive decoded EEF XYZ segment lengths, in metres.",
            "- `roughness_ratio`: mean norm of decoded XYZ second differences divided by the expert XYZ second-difference mean; dimensionless.",
            "- `endpoint_target_vector_cosine`: cosine between decoded endpoint displacement and the target vector from the decoded start; dimensionless and distinct from decoded-vs-expert error.",
            "- `endpoint_error_m`: decoded final XYZ minus expert final XYZ norm, in the existing physical/control frame; it is not final target distance.",
            "- All paired deltas use `left - right` on the same TRAIN24 window; no weighted score or significance threshold is used.",
            "",
            "## Recommendation and limitations",
            "",
            "Recommendation: make the decision unit one complete route×tier block with all three seeds, review the 24-window paired evidence and D1/D2A summaries, then have PI pre-register exactly one route/tier for the first Validation. Do not select from a single seed, pool tiers, or infer a winner from route means.",
            "",
            "Limitations: this is TRAIN24 evidence only; 24 windows are repeated structured probes rather than 24 IID samples, and the physical E4 sources include both current/recovery and historical read-only provenance. D1/D2A summaries are descriptive and do not establish a new scientific intervention.",
            "",
            "Files: `selection_packet.json`, `per_window_physical_metrics.json`, `stratified_summaries.json`, `paired_comparisons.json`, `phase_diagnostic_summary.json`, `metric_frame_definitions.json`, `row_and_call_accounting.json`, `source_inventory.json`, and `PACKET_SHA256SUMS`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_fresh(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite packet output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_packet(campaign_root: Path, output_dir: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).resolve()
    output_dir = Path(output_dir).resolve()
    require(campaign_root.is_dir(), f"campaign root unavailable: {campaign_root}")
    report_path = campaign_root / "physical_metrics_report.json"
    report = read_json(report_path)
    require(report.get("status") == "PASS_READONLY_DERIVED_FROM_SEALED_E4_ROWS", "physical report is not the accepted sealed readout")
    require(report.get("pi_gate") == "PI_GATE_STATE_0152", "physical report gate drift")
    require(report.get("execution_head") == EXECUTION_HEAD, "physical report execution HEAD drift")
    require(report.get("raw_before_derived") is True and report.get("source_rows_untouched") is True, "physical report provenance flags drift")
    final_result = read_json(campaign_root / "final_result.json")
    cpu_summary = read_json(campaign_root / "cpu_summary.json")
    require(final_result.get("token") == "STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_COMPLETE_FOR_PI_REVIEW", "final result token drift")
    require(cpu_summary.get("status") == "PASS_READONLY_FROM_SEALED_GPU_EVIDENCE", "CPU summary status drift")
    rows, physical_source_info = build_physical_rows(campaign_root, report)
    stratified = build_stratified_summaries(rows)
    paired = build_paired_comparisons(rows)
    diagnostic = build_phase_diagnostic_summary(campaign_root)
    accounting = build_row_call_accounting(campaign_root, report, final_result, cpu_summary, rows)
    analysis_identity = _git_identity(Path(__file__).resolve().parents[1])
    require(analysis_identity["head"] == EXPECTED_ANALYSIS_HEAD, f"analysis HEAD drift: {analysis_identity['head']}")
    require(analysis_identity["working_tree_clean"], "analysis worktree must be clean when packet is built")
    source_inventory = build_source_inventory(campaign_root, report, physical_source_info)
    artifact_hashes = {
        "final_result": sha256_file(campaign_root / "final_result.json"),
        "cpu_summary": sha256_file(campaign_root / "cpu_summary.json"),
        "physical_metrics_report": sha256_file(campaign_root / "physical_metrics_report.json"),
        "canonical_row_index": sha256_file(campaign_root / "canonical_row_index.json"),
        "sha256sums": sha256_file(campaign_root / "SHA256SUMS"),
    }
    return {
        "schema": "VANDA_STAGE45E_PI0153_TIER_SEED_SELECTION_PACKET_V1",
        "status": "PASS_CPU_READONLY_SELECTION_MATERIAL_READY_FOR_PI_REVIEW",
        "return_token": "VANDA_PI0153_TIER_SEED_SELECTION_PACKET_FOR_PI_REVIEW",
        "pi_gate": PI_GATE,
        "issue_comment": ISSUE_COMMENT,
        "pr_number": PR_NUMBER,
        "campaign_root": str(campaign_root),
        "execution_head": report.get("execution_head"),
        "analysis_identity": analysis_identity,
        "input_contract": {
            "split": "TRAIN24",
            "routes": list(ROUTES),
            "seeds": list(SEEDS),
            "tiers": list(TIERS),
            "window_count": 24,
            "window_order": list(EXPECTED_WINDOWS),
            "target_xyz_in_model_input": False,
            "model_execution_during_analysis": False,
            "training_during_analysis": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "winner_selection": False,
        },
        "artifact_hashes": artifact_hashes,
        "row_accounting_file": "row_and_call_accounting.json",
        "physical_rows_file": "per_window_physical_metrics.json",
        "stratified_summaries_file": "stratified_summaries.json",
        "paired_comparisons_file": "paired_comparisons.json",
        "diagnostic_summary_file": "phase_diagnostic_summary.json",
        "metric_definitions_file": "metric_frame_definitions.json",
        "source_inventory_file": "source_inventory.json",
        "recommendation": "Use one complete route×tier block with all three seeds as the PI decision unit, then pre-register one route/tier for first Validation; do not mix tiers or select from one seed or pooled means.",
        "limitations": [
            "TRAIN24 only; no Validation-E/Holdout-F.",
            "Windows and seeds are structured repeated probes, not IID samples.",
            "Physical E4 table contains current, recovery-source, and historical read-only provenance with explicit de-duplication.",
            "D1/D2A summaries are descriptive and do not choose a scientific winner.",
        ],
        "source_inventory": source_inventory,
        "row_accounting": accounting,
        "pairing_contract": paired["pairing_contract"],
        "diagnostic_overview": {
            phase: {
                route: {
                    "derived_row_count": diagnostic["phases"][phase][route]["derived_row_count"],
                    "checkpoint_arm_count": diagnostic["phases"][phase][route]["checkpoint_arm_count"],
                    "raw_before_derived_all": diagnostic["phases"][phase][route]["raw_before_derived_all"],
                }
                for route in ROUTES
            }
            for phase in ("D1", "D2A")
        },
        "descriptive_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        packet = build_packet(args.campaign_root, args.output_dir)
        campaign_root = Path(packet["campaign_root"])
        report = read_json(campaign_root / "physical_metrics_report.json")
        rows, source_info = build_physical_rows(campaign_root, report)
        stratified = build_stratified_summaries(rows)
        paired = build_paired_comparisons(rows)
        diagnostic = build_phase_diagnostic_summary(campaign_root)
        accounting = build_row_call_accounting(campaign_root, report, read_json(campaign_root / "final_result.json"), read_json(campaign_root / "cpu_summary.json"), rows)
        definitions = {
            "schema": "VANDA_STAGE45E_PI0153_METRIC_FRAME_DEFINITIONS_V1",
            "status": "PASS_EXPLICIT_PHYSICAL_METRIC_SEMANTICS",
            "frame": {
                "source_action_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
                "decoded_target_metric_frame": "WORLD_FRAME_METRES",
                "target_vector": "target_world - decoded_start_world",
                "anchor_translation": "existing sealed materialization; no re-anchoring or target-dependent transform",
            },
            "metrics": {
                "final_target_distance_m": "Euclidean decoded final EEF position to target; target distance, not expert error.",
                "absolute_progress_m": "initial target distance minus final target distance; positive is distance reduction.",
                "minimum_target_distance_m": "minimum target distance over decoded trajectory.",
                "prediction_path_length_m": "sum of decoded successive XYZ segment norms.",
                "roughness_ratio": "mean decoded XYZ second-difference norm divided by expert XYZ second-difference mean; dimensionless.",
                "endpoint_target_vector_cosine": "cosine(decoded endpoint displacement, target vector from decoded start).",
                "endpoint_error_m": "norm(decoded final XYZ - expert final XYZ); not target distance.",
                "position_rmse_m": "RMSE of decoded-vs-expert XYZ over H40.",
                "best_progress_m": "initial target distance minus minimum target distance.",
            },
            "paired": {
                "delta": "left minus right on the same window_id/target/seed/tier key",
                "intervention_vs_legacy": "L1 or N2 versus Legacy at the same tier and seed: 12 comparisons.",
                "exposure": "@2000 versus @1000 within the same route and seed: 9 comparisons.",
                "l1_vs_n2": "L1 versus N2 at the same tier and seed: 6 comparisons.",
                "no_iid_or_significance_claim": True,
                "no_weighted_score": True,
            },
        }
        output_dir = Path(args.output_dir).resolve()
        require(not output_dir.exists() and not output_dir.is_symlink(), f"output directory must be fresh: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        write_fresh(output_dir / "selection_packet.json", packet)
        write_fresh(output_dir / "per_window_physical_metrics.json", {"schema": "VANDA_STAGE45E_PI0153_PER_WINDOW_PHYSICAL_METRICS_V1", "row_count": len(rows), "rows": rows})
        write_fresh(output_dir / "stratified_summaries.json", stratified)
        write_fresh(output_dir / "paired_comparisons.json", paired)
        write_fresh(output_dir / "phase_diagnostic_summary.json", diagnostic)
        write_fresh(output_dir / "metric_frame_definitions.json", definitions)
        write_fresh(output_dir / "row_and_call_accounting.json", accounting)
        write_fresh(output_dir / "source_inventory.json", source_info | {"sealed_artifacts": packet["source_inventory"]})
        write_fresh(output_dir / "README.md", build_readme(packet, stratified, paired, accounting, diagnostic))
        entries = []
        for path in sorted(output_dir.iterdir()):
            if path.name in {"PACKET_SHA256SUMS", "selection_packet_manifest.json"} or path.is_dir():
                continue
            entries.append(f"{sha256_file(path)}  {path.name}")
        write_fresh(output_dir / "PACKET_SHA256SUMS", "\n".join(entries) + "\n")
        manifest = {
            "schema": "VANDA_STAGE45E_PI0153_PACKET_MANIFEST_V1",
            "status": "PASS_HASHED_CPU_SELECTION_PACKET",
            "packet_root": str(output_dir),
            "file_count": len(entries),
            "entries_exclude_manifest": True,
            "entries": [{"sha256": line.split("  ", 1)[0], "name": line.split("  ", 1)[1]} for line in entries],
            "packet_sha256sums": sha256_file(output_dir / "PACKET_SHA256SUMS"),
            "return_token": packet["return_token"],
        }
        write_fresh(output_dir / "selection_packet_manifest.json", manifest)
        print(json.dumps({"status": packet["status"], "return_token": packet["return_token"], "output_dir": str(output_dir), "file_count": len(entries), "packet_sha256sums": manifest["packet_sha256sums"], "physical_rows": len(rows), "paired_comparisons": len(paired["comparisons"])}, sort_keys=True))
        return 0
    except (SelectionPacketError, OSError, ValueError) as exc:
        print(f"STAGE45E_PI0153_SELECTION_PACKET_BLOCKED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
