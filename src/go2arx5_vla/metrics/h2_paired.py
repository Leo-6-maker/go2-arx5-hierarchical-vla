"""Strict paired metrics for the Stage22 R0/R1 attribution experiment.

This module compares already-produced offline profile manifests.  It never
loads a model, changes a trajectory, selects a checkpoint, or executes WBC.
The pairing key is ``(snapshot input_id, inference_seed)`` and overlap rows
are paired by episode and absolute observation timestamps.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EVALUATION_SCHEMA = "STAGE22_ACTION_PROFILE_EVALUATION_V1"
SUPPORTED_SPLITS = {"validation", "final_test"}


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _record_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    path_ratio = record.get("path_length_ratio")
    mean_speed_ratio = record.get("mean_speed_ratio")
    max_speed_ratio = record.get("max_speed_ratio")
    roughness_ratio = record.get("roughness_ratio")

    def deviation(value: Any) -> float | None:
        return None if value is None else abs(float(value) - 1.0)

    return {
        "position_mean_error_m": _nested(record, "position_error_m", "mean"),
        "position_p95_error_m": _nested(record, "position_error_m", "p95"),
        "position_max_error_m": _nested(record, "position_error_m", "max"),
        "rotation_mean_error_rad": _nested(record, "rotation_error_rad", "mean"),
        "rotation_p95_error_rad": _nested(record, "rotation_error_rad", "p95"),
        "rotation_max_error_rad": _nested(record, "rotation_error_rad", "max"),
        "endpoint_position_error_m": record.get("endpoint_position_error_m"),
        "endpoint_rotation_error_rad": record.get("endpoint_rotation_error_rad"),
        "path_length_ratio": path_ratio,
        "path_length_ratio_abs_deviation": deviation(path_ratio),
        "mean_speed_ratio": mean_speed_ratio,
        "mean_speed_ratio_abs_deviation": deviation(mean_speed_ratio),
        "max_speed_ratio": max_speed_ratio,
        "max_speed_ratio_abs_deviation": deviation(max_speed_ratio),
        "roughness_ratio": roughness_ratio,
        "roughness_ratio_abs_deviation": deviation(roughness_ratio),
        "initial_position_discontinuity_m": _nested(
            record, "predicted_profile", "initial_position_discontinuity_m"
        ),
        "initial_rotation_discontinuity_rad": _nested(
            record, "predicted_profile", "initial_rotation_discontinuity_rad"
        ),
    }


def _index_records(
    payloads: Sequence[Mapping[str, Any]], *, expected_split: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    if expected_split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported H2 comparison split: {expected_split}")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for payload in payloads:
        if payload.get("schema") != EVALUATION_SCHEMA:
            raise ValueError("H2 paired comparison requires action-profile evaluation manifests")
        if payload.get("split") != expected_split:
            raise ValueError("all H2 paired inputs must use the explicitly requested split")
        seed = payload.get("inference_seed")
        if seed is None:
            raise ValueError("each profile evaluation must record inference_seed")
        seed = int(seed)
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("each profile evaluation must contain non-empty records")
        local_ids: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping) or not record.get("input_id"):
                raise ValueError("every profile record must contain input_id")
            input_id = str(record["input_id"])
            if input_id in local_ids:
                raise ValueError(f"duplicate input_id in one seed manifest: {input_id}")
            local_ids.add(input_id)
            key = (input_id, seed)
            if key in indexed:
                raise ValueError(f"duplicate H2 pairing key: {key}")
            indexed[key] = record
    return indexed


def _overlap_key(item: Mapping[str, Any]) -> tuple[str, float, float]:
    episode = item.get("raw_episode_id")
    previous = item.get("previous_observation_timestamp_s")
    current = item.get("current_observation_timestamp_s")
    if episode is None or previous is None or current is None:
        raise ValueError("overlap rows need episode and absolute observation timestamps")
    return str(episode), float(previous), float(current)


def _index_overlaps(
    payloads: Sequence[Mapping[str, Any]], *, expected_split: str
) -> dict[tuple[str, float, float, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, float, float, int], Mapping[str, Any]] = {}
    for payload in payloads:
        if payload.get("schema") != EVALUATION_SCHEMA or payload.get("split") != expected_split:
            raise ValueError("overlap inputs must match the paired profile manifests")
        seed = int(payload["inference_seed"])
        overlaps = payload.get("chunk_overlap_metrics", [])
        if not isinstance(overlaps, list):
            raise ValueError("chunk_overlap_metrics must be a list")
        for item in overlaps:
            if not isinstance(item, Mapping):
                raise ValueError("overlap rows must be objects")
            episode, previous, current = _overlap_key(item)
            key = (episode, previous, current, seed)
            if key in indexed:
                raise ValueError(f"duplicate H2 overlap pairing key: {key}")
            indexed[key] = item
    return indexed


def _overlap_metrics(item: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "overlap_position_mean_error_m": _nested(item, "position_error_m", "mean"),
        "overlap_position_p95_error_m": _nested(item, "position_error_m", "p95"),
        "overlap_position_max_error_m": _nested(item, "position_error_m", "max"),
        "overlap_rotation_mean_error_rad": _nested(item, "rotation_error_rad", "mean"),
        "overlap_rotation_p95_error_rad": _nested(item, "rotation_error_rad", "p95"),
        "overlap_rotation_max_error_rad": _nested(item, "rotation_error_rad", "max"),
        "overlap_tangent_mean_error_mps": _nested(
            item, "translational_tangent_error_mps", "mean"
        ),
        "overlap_tangent_p95_error_mps": _nested(
            item, "translational_tangent_error_mps", "p95"
        ),
        "overlap_tangent_max_error_mps": _nested(
            item, "translational_tangent_error_mps", "max"
        ),
        "old_vs_new_first_position_error_m": _nested(
            item, "old_prediction_at_new_observation_vs_new_first", "position_error_m"
        ),
        "old_vs_new_first_rotation_error_rad": _nested(
            item, "old_prediction_at_new_observation_vs_new_first", "rotation_error_rad"
        ),
    }


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _decision_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        decision = _nested(record, "safety_gate", "decision")
        if decision is not None:
            counts[str(decision)] += 1
    return {decision: int(counts.get(decision, 0)) for decision in ("ACCEPT", "WARN", "REJECT")}


def _summary(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, Any]:
    return _stats(row.get(value_key) for row in rows)


def compare_h2_profile_payloads(
    r0_payloads: Sequence[Mapping[str, Any]],
    r1_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
    source_files: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build a strict paired R0/R1 comparison for one locked split."""

    r0_index = _index_records(r0_payloads, expected_split=expected_split)
    r1_index = _index_records(r1_payloads, expected_split=expected_split)
    if set(r0_index) != set(r1_index):
        missing_in_r0 = sorted(set(r1_index) - set(r0_index))
        missing_in_r1 = sorted(set(r0_index) - set(r1_index))
        raise ValueError(
            "R0/R1 snapshot-seed pairing mismatch: "
            f"missing_in_r0={missing_in_r0[:5]}, missing_in_r1={missing_in_r1[:5]}"
        )

    paired_records: list[dict[str, Any]] = []
    metric_names: list[str] | None = None
    for input_id, seed in sorted(r0_index, key=lambda value: (value[1], value[0])):
        r0_record = r0_index[(input_id, seed)]
        r1_record = r1_index[(input_id, seed)]
        r0_values = _record_metrics(r0_record)
        r1_values = _record_metrics(r1_record)
        if metric_names is None:
            metric_names = list(r0_values)
        paired_records.append(
            {
                "pair_key": {"input_id": input_id, "inference_seed": seed},
                "raw_episode_id": r0_record.get("raw_episode_id"),
                "target_id": r0_record.get("target_id"),
                "r0": {
                    "metrics": r0_values,
                    "finite": bool(r0_record.get("finite", True)),
                    "safety_gate_decision": _nested(r0_record, "safety_gate", "decision"),
                },
                "r1": {
                    "metrics": r1_values,
                    "finite": bool(r1_record.get("finite", True)),
                    "safety_gate_decision": _nested(r1_record, "safety_gate", "decision"),
                },
                "delta_r1_minus_r0": {
                    name: _delta(r0_values.get(name), r1_values.get(name))
                    for name in r0_values
                },
            }
        )

    r0_overlaps = _index_overlaps(r0_payloads, expected_split=expected_split)
    r1_overlaps = _index_overlaps(r1_payloads, expected_split=expected_split)
    if set(r0_overlaps) != set(r1_overlaps):
        raise ValueError("R0/R1 common-future overlap pairing mismatch")
    paired_overlaps: list[dict[str, Any]] = []
    overlap_metric_names: list[str] | None = None
    for key in sorted(r0_overlaps, key=lambda value: (value[3], value[0], value[1], value[2])):
        r0_values = _overlap_metrics(r0_overlaps[key])
        r1_values = _overlap_metrics(r1_overlaps[key])
        if overlap_metric_names is None:
            overlap_metric_names = list(r0_values)
        paired_overlaps.append(
            {
                "pair_key": {
                    "raw_episode_id": key[0],
                    "previous_observation_timestamp_s": key[1],
                    "current_observation_timestamp_s": key[2],
                    "inference_seed": key[3],
                },
                "r0": r0_values,
                "r1": r1_values,
                "delta_r1_minus_r0": {
                    name: _delta(r0_values.get(name), r1_values.get(name))
                    for name in r0_values
                },
            }
        )

    def rows_for(recipe: str, name: str) -> list[float | None]:
        return [
            _nested(row, recipe, "metrics", name)
            for row in paired_records
        ]

    def overlap_rows_for(recipe: str, name: str) -> list[float | None]:
        return [_nested(row, recipe, name) for row in paired_overlaps]

    record_summary = {
        "r0": {name: _stats(rows_for("r0", name)) for name in (metric_names or [])},
        "r1": {name: _stats(rows_for("r1", name)) for name in (metric_names or [])},
        "delta_r1_minus_r0": {
            name: _stats(_nested(row, "delta_r1_minus_r0", name) for row in paired_records)
            for name in (metric_names or [])
        },
    }
    overlap_summary = {
        "r0": {
            name: _stats(overlap_rows_for("r0", name)) for name in (overlap_metric_names or [])
        },
        "r1": {
            name: _stats(overlap_rows_for("r1", name)) for name in (overlap_metric_names or [])
        },
        "delta_r1_minus_r0": {
            name: _stats(_nested(row, "delta_r1_minus_r0", name) for row in paired_overlaps)
            for name in (overlap_metric_names or [])
        },
    }
    return {
        "schema": "STAGE22_H2_PAIRED_PROFILE_COMPARISON_V1",
        "status": "PASS",
        "expected_split": expected_split,
        "pairing_key": ["input_id", "inference_seed"],
        "overlap_pairing_key": [
            "raw_episode_id",
            "previous_observation_timestamp_s",
            "current_observation_timestamp_s",
            "inference_seed",
        ],
        "paired_alignment": {
            "status": "PASS",
            "r0_pair_count": len(r0_index),
            "r1_pair_count": len(r1_index),
            "paired_record_count": len(paired_records),
            "r0_overlap_count": len(r0_overlaps),
            "r1_overlap_count": len(r1_overlaps),
            "paired_overlap_count": len(paired_overlaps),
        },
        "inference_seeds": sorted({seed for _, seed in r0_index}),
        "source_files": {} if source_files is None else {key: list(value) for key, value in source_files.items()},
        "trajectory_correction": False,
        "training_loss_used_for_selection": False,
        "delta_definition": "R1 metric minus R0 metric; negative means lower-is-better error improved for that metric",
        "safety_gate_decisions": {
            "r0": _decision_counts(r0_index.values()),
            "r1": _decision_counts(r1_index.values()),
        },
        "finite_outputs": {
            "r0_all_finite": all(bool(record.get("finite", True)) for record in r0_index.values()),
            "r1_all_finite": all(bool(record.get("finite", True)) for record in r1_index.values()),
        },
        "summary": {"trajectory": record_summary, "overlap": overlap_summary},
        "paired_records": paired_records,
        "paired_overlaps": paired_overlaps,
        "claim_boundary": "offline_matched_profile_attribution_only; no task-success or WBC-execution claim",
    }
