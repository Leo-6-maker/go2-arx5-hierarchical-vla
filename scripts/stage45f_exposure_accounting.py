#!/usr/bin/env python3
"""CPU-only exact exposure accounting for future training traces.

Two trace strengths are supported:
- `STAGE45F_TRAIN_EXPOSURE_EVENT_V1`: sample/microbatch events explicitly
  bound to optimizer steps by future trainer instrumentation;
- `STAGE45F_TRAIN_SAMPLE_FETCH_EVENT_V1`: exact dataset fetch identities from
  a non-invasive side channel, without claiming optimizer-step correspondence.

Both provide exact per-window event counts. Neither is automatically called an
epoch count or iid exposure.
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


STRONG_SCHEMA = "STAGE45F_TRAIN_EXPOSURE_EVENT_V1"
FETCH_SCHEMA = "STAGE45F_TRAIN_SAMPLE_FETCH_EVENT_V1"


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
    observed_schema: str | None = None
    previous_fetch_ordinal = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"line {line_number} is not object")
        schema = value.get("schema")
        require(schema in {STRONG_SCHEMA, FETCH_SCHEMA}, f"line {line_number} schema drift")
        if observed_schema is None:
            observed_schema = schema
        require(schema == observed_schema, "mixed exposure trace schemas are forbidden")
        window_id = value.get("window_id")
        require(isinstance(window_id, str) and window_id, f"line {line_number} window_id missing")
        require(value.get("split") == "TRAIN", f"line {line_number} is not TRAIN")
        require(value.get("validation_e") is False and value.get("holdout_f") is False, f"line {line_number} locked split flag")
        if schema == STRONG_SCHEMA:
            step = value.get("optimizer_step")
            require(isinstance(step, int) and not isinstance(step, bool) and step >= 1, f"line {line_number} optimizer_step invalid")
            rows.append({"schema": schema, "optimizer_step": step, "window_id": window_id})
        else:
            ordinal = value.get("fetch_ordinal")
            require(isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal == previous_fetch_ordinal + 1, f"line {line_number} fetch_ordinal is not contiguous")
            previous_fetch_ordinal = ordinal
            require(value.get("optimizer_step") is None, f"line {line_number} fetch-only trace must not invent optimizer_step")
            require(value.get("optimizer_step_binding") == "NOT_CAPTURED_BY_DATASET_SIDECHANNEL", f"line {line_number} fetch-only binding drift")
            rows.append({"schema": schema, "fetch_ordinal": ordinal, "window_id": window_id})
    require(rows, "empty exposure trace")
    return rows


def _count_summary(counts: Mapping[str, int], total: int) -> dict[str, Any]:
    values = list(counts.values())
    proportions = [count / total for count in values]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0.0)
    max_entropy = math.log(len(values)) if values else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 1.0
    return {
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
    }


def analyze(rows: Sequence[Mapping[str, Any]], trace_sha256: str) -> dict[str, Any]:
    schema = str(rows[0]["schema"])
    require(all(row["schema"] == schema for row in rows), "mixed trace schemas")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["window_id"]] = counts.get(row["window_id"], 0) + 1
    total = len(rows)
    common = {
        "schema": "STAGE45F_EXACT_EXPOSURE_ACCOUNTING_V2",
        "status": "PASS_EXACT_TRACE_ANALYSIS",
        "trace_source_schema": schema,
        "trace_sha256": trace_sha256,
        "exposure_event_count": total,
        **_count_summary(counts, total),
        "epoch_claim": "NOT_DERIVED_AUTOMATICALLY",
        "exact_per_window_event_exposure_available": True,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if schema == STRONG_SCHEMA:
        step_counts: dict[int, int] = {}
        for row in rows:
            step_counts[int(row["optimizer_step"])] = step_counts.get(int(row["optimizer_step"]), 0) + 1
        ordered_steps = sorted(step_counts)
        require(ordered_steps == list(range(1, max(ordered_steps) + 1)), "optimizer-step sequence has gaps")
        common.update(
            {
                "optimizer_step_binding": "EXPLICIT_IN_TRACE",
                "optimizer_step_count": max(ordered_steps),
                "microbatch_or_sample_events_per_step": {
                    "min": min(step_counts.values()),
                    "max": max(step_counts.values()),
                    "mean": statistics.fmean(step_counts.values()),
                },
                "exact_optimizer_step_bound_exposure_available": True,
                "claim_boundary": "Exact per-window events and explicit optimizer-step binding are available, but this still does not imply iid samples, independent windows, or effective epochs.",
            }
        )
    else:
        common.update(
            {
                "optimizer_step_binding": "NOT_CAPTURED_BY_DATASET_SIDECHANNEL",
                "optimizer_step_count": None,
                "microbatch_or_sample_events_per_step": None,
                "exact_optimizer_step_bound_exposure_available": False,
                "claim_boundary": "Exact dataset fetch counts are available. Optimizer-step correspondence is intentionally unclaimed until separately instrumented at the trainer boundary.",
            }
        )
    return common


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
