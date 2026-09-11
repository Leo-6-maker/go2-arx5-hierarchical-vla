#!/usr/bin/env python3
"""CPU-only exact exposure accounting for future training traces.

A future trainer may emit one JSONL row per optimizer micro-batch/sample with
its TRAIN window identity.  This tool converts that trace into exact counts.
It intentionally refuses to infer epochs or per-window exposure from max_steps
alone.
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


class ExposureAccountingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExposureAccountingError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trace(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing trace: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"line {line_number} is not object")
        require(value.get("schema") == "STAGE45F_TRAIN_EXPOSURE_EVENT_V1", f"line {line_number} schema drift")
        step = value.get("optimizer_step")
        window_id = value.get("window_id")
        require(isinstance(step, int) and step >= 1, f"line {line_number} optimizer_step invalid")
        require(isinstance(window_id, str) and window_id, f"line {line_number} window_id missing")
        require(value.get("split") == "TRAIN", f"line {line_number} is not TRAIN")
        require(value.get("validation_e") is False and value.get("holdout_f") is False, f"line {line_number} locked split flag")
        rows.append({"optimizer_step": step, "window_id": window_id})
    require(rows, "empty exposure trace")
    return rows


def analyze(rows: Sequence[Mapping[str, Any]], trace_sha256: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    step_counts: dict[int, int] = {}
    for row in rows:
        counts[row["window_id"]] = counts.get(row["window_id"], 0) + 1
        step_counts[row["optimizer_step"]] = step_counts.get(row["optimizer_step"], 0) + 1
    ordered_steps = sorted(step_counts)
    require(ordered_steps == list(range(1, max(ordered_steps) + 1)), "optimizer-step sequence has gaps")
    values = list(counts.values())
    total = len(rows)
    proportions = [count / total for count in values]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0.0)
    max_entropy = math.log(len(values)) if values else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 1.0
    return {
        "schema": "STAGE45F_EXACT_EXPOSURE_ACCOUNTING_V1",
        "status": "PASS_EXACT_TRACE_ANALYSIS",
        "trace_sha256": trace_sha256,
        "optimizer_step_count": max(ordered_steps),
        "exposure_event_count": total,
        "microbatch_or_sample_events_per_step": {
            "min": min(step_counts.values()),
            "max": max(step_counts.values()),
            "mean": statistics.fmean(step_counts.values()),
        },
        "unique_window_count": len(counts),
        "per_window_counts": dict(sorted(counts.items())),
        "per_window_count_summary": {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "range": max(values) - min(values),
        },
        "normalized_window_exposure_entropy": normalized_entropy,
        "epoch_claim": "NOT_DERIVED_AUTOMATICALLY",
        "exact_per_window_exposure_available": True,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "This is exact trace accounting only. It does not imply iid samples, independent windows, or effective epochs.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rows = read_trace(args.trace)
    print(json.dumps(analyze(rows, sha256_file(args.trace)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExposureAccountingError as exc:
        print(f"STAGE45F_EXPOSURE_ACCOUNTING_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
