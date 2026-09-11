#!/usr/bin/env python3
"""CPU-only validator for future optimizer-step-bound exposure traces.

Dataset fetch traces are useful but can be affected by prefetch/workers and are
not optimizer-step evidence. A future separately gated trainer hook should emit
one event *per optimizer step per rank* after the optimizer step has actually
been applied. Each event contains the TRAIN window IDs that contributed to that
rank's accumulated gradient since the previous optimizer step.

This tool only validates and summarizes an already-produced JSONL trace. It does
not patch the trainer or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


EVENT_SCHEMA = "STAGE45F_OPTIMIZER_STEP_EXPOSURE_EVENT_V1"


class OptimizerExposureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OptimizerExposureError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing trace: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"line {line_number} is not object")
        require(value.get("schema") == EVENT_SCHEMA, f"line {line_number} schema drift")
        require(value.get("split") == "TRAIN", f"line {line_number} is not TRAIN")
        require(value.get("validation_e") is False and value.get("holdout_f") is False, f"line {line_number} locked split flag")
        require(value.get("trainer_boundary") == "AFTER_SUCCESSFUL_OPTIMIZER_STEP", f"line {line_number} trainer boundary drift")
        require(value.get("optimizer_step_applied") is True, f"line {line_number} does not bind a successful optimizer step")
        step = value.get("optimizer_step")
        rank = value.get("rank")
        world_size = value.get("world_size")
        grad_accum = value.get("gradient_accumulation_steps")
        sample_ids = value.get("sample_window_ids")
        require(isinstance(step, int) and not isinstance(step, bool) and step >= 1, f"line {line_number} optimizer_step invalid")
        require(isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0, f"line {line_number} rank invalid")
        require(isinstance(world_size, int) and not isinstance(world_size, bool) and world_size >= 1, f"line {line_number} world_size invalid")
        require(rank < world_size, f"line {line_number} rank outside world_size")
        require(isinstance(grad_accum, int) and not isinstance(grad_accum, bool) and grad_accum >= 1, f"line {line_number} gradient_accumulation_steps invalid")
        require(isinstance(sample_ids, list) and sample_ids, f"line {line_number} sample_window_ids missing")
        require(all(isinstance(item, str) and item for item in sample_ids), f"line {line_number} invalid sample window id")
        rows.append({
            "optimizer_step": step,
            "rank": rank,
            "world_size": world_size,
            "gradient_accumulation_steps": grad_accum,
            "sample_window_ids": list(sample_ids),
        })
    require(rows, "empty optimizer exposure trace")
    return rows


def analyze(rows: Sequence[Mapping[str, Any]], trace_sha256: str) -> dict[str, Any]:
    world_sizes = {int(row["world_size"]) for row in rows}
    grad_accums = {int(row["gradient_accumulation_steps"]) for row in rows}
    require(len(world_sizes) == 1, "world_size changed within trace")
    require(len(grad_accums) == 1, "gradient accumulation changed within trace")
    world_size = next(iter(world_sizes))
    grad_accum = next(iter(grad_accums))

    by_step: dict[int, dict[int, list[str]]] = {}
    for row in rows:
        step = int(row["optimizer_step"])
        rank = int(row["rank"])
        rank_map = by_step.setdefault(step, {})
        require(rank not in rank_map, f"duplicate rank event at optimizer step {step}, rank {rank}")
        rank_map[rank] = list(row["sample_window_ids"])

    steps = sorted(by_step)
    require(steps == list(range(1, max(steps) + 1)), "optimizer-step sequence has gaps")
    expected_ranks = set(range(world_size))
    for step in steps:
        require(set(by_step[step]) == expected_ranks, f"optimizer step {step} does not contain exactly one event per rank")

    per_window: dict[str, int] = {}
    per_step_global_samples: list[int] = []
    rank_sample_counts: dict[int, int] = {rank: 0 for rank in expected_ranks}
    for step in steps:
        count = 0
        for rank in range(world_size):
            ids = by_step[step][rank]
            count += len(ids)
            rank_sample_counts[rank] += len(ids)
            for window_id in ids:
                per_window[window_id] = per_window.get(window_id, 0) + 1
        per_step_global_samples.append(count)

    counts = list(per_window.values())
    total_samples = sum(per_step_global_samples)
    proportions = [value / total_samples for value in counts]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0.0)
    max_entropy = math.log(len(counts)) if counts else 0.0
    return {
        "schema": "STAGE45F_OPTIMIZER_STEP_EXPOSURE_ACCOUNTING_V1",
        "status": "PASS_EXACT_OPTIMIZER_STEP_TRACE_ANALYSIS",
        "trace_sha256": trace_sha256,
        "optimizer_step_count": len(steps),
        "world_size": world_size,
        "gradient_accumulation_steps": grad_accum,
        "global_sample_exposure_count": total_samples,
        "per_step_global_sample_count": {
            "min": min(per_step_global_samples),
            "max": max(per_step_global_samples),
            "mean": statistics.fmean(per_step_global_samples),
        },
        "rank_sample_counts": {str(key): value for key, value in sorted(rank_sample_counts.items())},
        "unique_window_count": len(per_window),
        "per_window_counts": dict(sorted(per_window.items())),
        "per_window_count_summary": {
            "min": min(counts),
            "max": max(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "range": max(counts) - min(counts),
        },
        "normalized_window_exposure_entropy": entropy / max_entropy if max_entropy > 0.0 else 1.0,
        "exact_optimizer_step_exposure_available": True,
        "epoch_claim": "NOT_DERIVED_AUTOMATICALLY",
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Counts are exact for samples serialized by the trainer after successful optimizer steps. They do not establish iid samples, independent windows, or effective epochs.",
        "training_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--trace", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rows = read_events(args.trace)
    print(json.dumps(analyze(rows, sha256_file(args.trace)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OptimizerExposureError as exc:
        print(f"STAGE45F_OPTIMIZER_EXPOSURE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
