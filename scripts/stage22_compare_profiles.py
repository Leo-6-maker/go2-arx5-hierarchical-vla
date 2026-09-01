#!/usr/bin/env python3
"""Compare two Stage22 profile-evaluation manifests without selecting by loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "p95": ordered[index],
        "max": ordered[-1],
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    def values(key: str) -> list[float]:
        result = []
        for record in records:
            value = record.get(key)
            if value is not None:
                result.append(float(value))
        return result

    decisions = {decision: 0 for decision in ("ACCEPT", "WARN", "REJECT")}
    for record in records:
        decision = record.get("safety_gate", {}).get("decision")
        if decision in decisions:
            decisions[decision] += 1
    overlap = payload.get("chunk_overlap_metrics", [])
    return {
        "model_path": payload.get("recipe_model_path"),
        "split": payload.get("split"),
        "snapshot_count": payload.get("snapshot_count"),
        "position_mean_error_m": _stats(
            [float(record["position_error_m"]["mean"]) for record in records]
        ),
        "rotation_mean_error_rad": _stats(
            [float(record["rotation_error_rad"]["mean"]) for record in records]
        ),
        "endpoint_position_error_m": _stats(values("endpoint_position_error_m")),
        "path_length_ratio": _stats(values("path_length_ratio")),
        "max_speed_ratio": _stats(values("max_speed_ratio")),
        "mean_speed_ratio": _stats(values("mean_speed_ratio")),
        "roughness_ratio": _stats(values("roughness_ratio")),
        "initial_position_discontinuity_m": _stats(
            [float(record["predicted_profile"]["initial_position_discontinuity_m"]) for record in records]
        ),
        "safety_gate_decisions": decisions,
        "overlap_count": len(overlap),
        "overlap_position_error_m": _stats(
            [float(item["position_error_m"]["mean"]) for item in overlap if item.get("position_error_m")]
        ),
        "overlap_rotation_error_rad": _stats(
            [float(item["rotation_error_rad"]["mean"]) for item in overlap if item.get("rotation_error_rad")]
        ),
        "overlap_tangent_error_mps": _stats(
            [
                float(item["translational_tangent_error_mps"]["mean"])
                for item in overlap
                if item.get("translational_tangent_error_mps")
            ]
        ),
        "visual_target_conditioning": payload.get("visual_target_conditioning", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r0 = json.loads(args.r0.read_text(encoding="utf-8"))
    r1 = json.loads(args.r1.read_text(encoding="utf-8"))
    payload = {
        "schema": "STAGE22_R0_R1_PROFILE_COMPARISON_V1",
        "status": "PASS",
        "selection_basis": "held_out_geometry_and_profile_metrics_not_training_loss",
        "trajectory_correction": False,
        "r0": _summary(r0),
        "r1": _summary(r1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
