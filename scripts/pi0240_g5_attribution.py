#!/usr/bin/env python3
"""Aggregate the frozen PI0240 G4 records without new model queries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": None if not array.size else float(array.mean()),
        "median": None if not array.size else float(np.median(array)),
        "p95": None if not array.size else float(np.percentile(array, 95)),
        "max": None if not array.size else float(array.max()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "gate": dict(Counter(row["model_gate"]["decision"] for row in rows)),
        "expert_gate": dict(Counter(row["expert_gate"]["decision"] for row in rows)),
        "model_extra_issues": dict(Counter(issue for row in rows for issue in row["model_extra_issues"])),
        "xyz_mean_m": stats(row["fit"]["xyz_m"]["mean"] for row in rows),
        "xyz_first_m": stats(row["fit"]["xyz_m"]["first"] for row in rows),
        "xyz_endpoint_m": stats(row["fit"]["xyz_m"]["endpoint"] for row in rows),
        "so3_mean_rad": stats(row["fit"]["so3_rad"]["mean"] for row in rows),
        "linear_speed_max_mps": stats(row["fit"]["velocity"]["model_linear_mps"]["max"] for row in rows),
        "angular_speed_max_radps": stats(row["fit"]["velocity"]["model_angular_radps"]["max"] for row in rows),
        "target_min_distance_m": stats(row["target_min_distance_m"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_jsonl(args.g4_root / "baseline1000" / "records.jsonl")
    fresh = load_jsonl(args.g4_root / "fresh10000" / "records.jsonl")
    if len(baseline) != 128 or len(fresh) != 128:
        raise RuntimeError(f"G4 record count drift: baseline={len(baseline)} fresh={len(fresh)}")
    key = lambda row: (row["split"], int(row["episode_index"]), row["source_episode_id"])
    base_map = {key(row): row for row in baseline}; fresh_map = {key(row): row for row in fresh}
    if set(base_map) != set(fresh_map) or len(base_map) != 128:
        raise RuntimeError("baseline/fresh fixed cohort identity mismatch")

    metric_paths = {
        "xyz_mean_m": ("fit", "xyz_m", "mean"),
        "xyz_first_m": ("fit", "xyz_m", "first"),
        "xyz_endpoint_m": ("fit", "xyz_m", "endpoint"),
        "so3_mean_rad": ("fit", "so3_rad", "mean"),
        "linear_speed_max_mps": ("fit", "velocity", "model_linear_mps", "max"),
        "target_min_distance_m": ("target_min_distance_m",),
    }
    def value(row: dict[str, Any], path: tuple[str, ...]) -> float:
        current: Any = row
        for name in path:
            current = current[name]
        return float(current)
    paired = {}
    paired_rows = []
    for name, path in metric_paths.items():
        deltas = [value(fresh_map[cell], path) - value(base_map[cell], path) for cell in sorted(base_map)]
        paired[name] = {"fresh_minus_baseline": stats(deltas), "fresh_better_fraction": float(np.mean(np.asarray(deltas) < 0.0))}
    for cell in sorted(base_map):
        left, right = base_map[cell], fresh_map[cell]
        paired_rows.append({
            "split": cell[0], "episode_index": cell[1], "source_episode_id": cell[2],
            "target_family": right["target_family"], "phase": right["phase"], "jitter_family": right["jitter_family"],
            **{f"{name}_baseline": value(left, path) for name, path in metric_paths.items()},
            **{f"{name}_fresh": value(right, path) for name, path in metric_paths.items()},
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fresh:
        grouped[f"{row['split']}|{row['target_family']}|phase{row['phase']}|{row['jitter_family']}"].append(row)
    train = [row for row in fresh if row["split"] == "TRAIN"]
    calib = [row for row in fresh if row["split"] == "CALIB"]
    train_linear_reject = sum("LINEAR_SPEED_REJECT" in row["model_extra_issues"] for row in train)
    calib_linear_reject = sum("LINEAR_SPEED_REJECT" in row["model_extra_issues"] for row in calib)
    report = {
        "schema": "PI0240_G5_FAILURE_ATTRIBUTION_V1",
        "status": "SCIENCE_FAIL_TRAIN_FIT",
        "selected_checkpoint": "NONE",
        "coverage": {"models": ["baseline1000", "fresh10000"], "train_per_model": 64, "calib_per_model": 64, "unique_scientific_cells": 256, "get_action_calls": 256},
        "baseline": {"all": summarize(baseline), "TRAIN": summarize([row for row in baseline if row["split"] == "TRAIN"]), "CALIB": summarize([row for row in baseline if row["split"] == "CALIB"])},
        "fresh10000": {"all": summarize(fresh), "TRAIN": summarize(train), "CALIB": summarize(calib)},
        "paired": paired,
        "fresh10000_strata": {name: summarize(rows) for name, rows in sorted(grouped.items())},
        "attribution": {
            "primary": "TRAIN_FIT_DEFICIT_EVIDENCE",
            "evidence": [
                f"fresh10000 TRAIN model-extra LINEAR_SPEED_REJECT {train_linear_reject}/64",
                f"fresh10000 CALIB model-extra LINEAR_SPEED_REJECT {calib_linear_reject}/64",
                "G3 real dataset/collator [B,40,132] to independent decoder physical parity passed",
                "fresh10000 improves strongly over baseline but remains far from expert actions on TRAIN",
            ],
            "not_supported": ["TRAIN_CALIB_GAP_ON_FIXED_COHORT", "WBC_OR_TRACKING_CAUSE", "TARGET_SCORER_CAUSE"],
            "limitations": ["No WBC or simulator execution was authorized after science fail", "No DEV-E/F read", "The 12 known expert REJECT rows are outside the fixed 128 G4 cohort and remain separately preserved in G3"],
        },
        "recommended_mainline_run_card_not_authorized": {
            "name": "FIXED16_FREE_RUNNING_MEMORIZATION_FEASIBILITY",
            "initializer": "PI0226 adapted checkpoint-1000",
            "sample_rule": "From frozen 64 TRAIN, PI0238 salt-hash one row per A-D x phase stratum; 16 total; no metric-based replacement",
            "trainable": ["projector", "diffusion/action_head"],
            "frozen": ["visual", "LLM", "VLLN"],
            "contract": "official dense-relative H40, StateV1, dt=0.05 s, target XYZ excluded",
            "optimizer_family": "AdamW", "learning_rate": 1e-4, "warmup_steps": 10, "seed": 22301,
            "max_optimizer_steps": 1000, "checkpoints": [250, 500, 1000],
            "free_running_screen": "step0/250/500/1000 x fixed16 = 64 policy queries",
            "proposed_success_before_run": ["zero SafetyGate REJECT on fixed16 at step1000", "median per-row XYZ RMSE <=0.020 m", "p95 per-row XYZ RMSE <=0.035 m", "median SO(3) RMSE <=0.050 rad"],
            "stop": "If step1000 misses any criterion, stop and diagnose objective/sampler/train-infer linkage; do not add steps",
            "execution": "NOT_AUTHORIZED_IN_PI0240",
        },
        "backup_not_authorized": "If fixed16 cannot be memorized, audit diffusion target/sampler/free-running linkage and parameter updates before any large training.",
        "forbidden_counts": {"training_backward": 0, "optimizer_steps": 0, "wbc_steps": 0, "sim_steps": 0, "dev_e_reads": 0, "final_f_reads": 0},
        "terminal_token": "SCIENCE_FAIL_TRAIN_FIT — STOP_FOR_PI_RECIPE_DECISION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".paired.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader(); writer.writerows(paired_rows)
    print(json.dumps({"status": report["status"], "output": str(args.output), "paired_csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
