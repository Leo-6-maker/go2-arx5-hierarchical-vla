#!/usr/bin/env python3
"""Aggregate matched Stage22 profile evaluations over diffusion seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    middle = len(finite) // 2
    if len(finite) % 2:
        median = finite[middle]
    else:
        median = 0.5 * (finite[middle - 1] + finite[middle])
    p95_index = min(len(finite) - 1, max(0, int(round(0.95 * (len(finite) - 1)))))
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": median,
        "p95": finite[p95_index],
        "max": finite[-1],
    }


def _values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _nested_values(records: list[dict[str, Any]], outer: str, inner: str) -> list[float]:
    values = []
    for record in records:
        item = record.get(outer, {})
        value = item.get(inner) if isinstance(item, dict) else None
        if value is not None:
            values.append(float(value))
    return values


def _aggregate(inputs: list[Path], recipe: str) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads:
        raise ValueError("at least one profile evaluation is required")
    split = payloads[0].get("split")
    model_path = payloads[0].get("recipe_model_path")
    input_ids: set[str] | None = None
    records: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    seed_rows = []
    for path, payload in zip(inputs, payloads):
        if payload.get("split") != split or payload.get("recipe_model_path") != model_path:
            raise ValueError("seed evaluations must use the same model and split")
        current_ids = {str(record.get("input_id")) for record in payload.get("records", [])}
        if input_ids is None:
            input_ids = current_ids
        elif current_ids != input_ids:
            raise ValueError("seed evaluations do not cover the same input snapshots")
        current_records = payload.get("records", [])
        current_overlaps = payload.get("chunk_overlap_metrics", [])
        records.extend(current_records)
        overlaps.extend(current_overlaps)
        seed_rows.append(
            {
                "path": str(path),
                "inference_seed": payload.get("inference_seed"),
                "snapshot_count": payload.get("snapshot_count"),
                "safety_gate_decisions": payload.get("summary", {}).get("safety_gate_decisions"),
            }
        )
    decisions = {decision: 0 for decision in ("ACCEPT", "WARN", "REJECT")}
    for record in records:
        decision = record.get("safety_gate", {}).get("decision")
        if decision in decisions:
            decisions[decision] += 1
    summary = {
        "position_mean_error_m": _stats(_nested_values(records, "position_error_m", "mean")),
        "position_p95_error_m": _stats(_nested_values(records, "position_error_m", "p95")),
        "rotation_mean_error_rad": _stats(_nested_values(records, "rotation_error_rad", "mean")),
        "endpoint_position_error_m": _stats(_values(records, "endpoint_position_error_m")),
        "path_length_ratio": _stats(_values(records, "path_length_ratio")),
        "mean_speed_ratio": _stats(_values(records, "mean_speed_ratio")),
        "max_speed_ratio": _stats(_values(records, "max_speed_ratio")),
        "roughness_ratio": _stats(_values(records, "roughness_ratio")),
        "initial_position_discontinuity_m": _stats(
            _nested_values(records, "predicted_profile", "initial_position_discontinuity_m")
        ),
        "safety_gate_decisions": decisions,
        "overlap_position_error_m": _stats(_nested_values(overlaps, "position_error_m", "p95")),
        "overlap_position_mean_error_m": _stats(_nested_values(overlaps, "position_error_m", "mean")),
        "overlap_rotation_error_rad": _stats(_nested_values(overlaps, "rotation_error_rad", "p95")),
        "overlap_tangent_error_mps": _stats(
            _nested_values(overlaps, "translational_tangent_error_mps", "p95")
        ),
    }
    return {
        "schema": "STAGE22_PROFILE_SEED_AGGREGATE_V1",
        "status": "PASS",
        "recipe": recipe,
        "model_path": model_path,
        "split": split,
        "seed_count": len(inputs),
        "seeds": seed_rows,
        "snapshot_count_per_seed": len(input_ids or []),
        "aggregate_record_count": len(records),
        "aggregate_overlap_count": len(overlaps),
        "summary": summary,
        "selection_basis": "matched_diffusion_seeds_and_held_out_profile_metrics; training_loss_not_used",
        "trajectory_correction": False,
        "records": records,
        "chunk_overlap_metrics": overlaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = _aggregate(args.inputs, args.recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
