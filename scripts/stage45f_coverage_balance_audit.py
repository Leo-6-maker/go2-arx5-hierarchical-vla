#!/usr/bin/env python3
"""Manifest-only descriptive coverage audit for hypothetical expanded TRAIN data.

The tool summarizes target, source-episode and caller-defined initial-state bins
without declaring an acceptable balance threshold. It is intended to reveal
whether 'more data' is mostly more copies from one region rather than broader
coverage. It never opens model/data payloads or authorizes collection/training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


class CoverageBalanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageBalanceError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _records(value: Any, name: str) -> list[dict[str, str]]:
    require(isinstance(value, list) and value, f"{name} must be non-empty list")
    result = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        require(isinstance(item, Mapping), f"{name}[{index}] must be object")
        fields = {}
        for field in ("window_id", "target_id", "source_episode_id", "initial_state_bin"):
            observed = item.get(field)
            require(isinstance(observed, str) and observed, f"{name}[{index}].{field} missing")
            fields[field] = observed
        require(fields["window_id"] not in ids, f"duplicate {name} window id")
        ids.add(fields["window_id"])
        if name == "added_windows":
            require(item.get("validation_or_holdout_origin") is False, f"{name}[{index}] locked split origin")
        result.append(fields)
    return result


def _summary(records: Sequence[Mapping[str, str]], field: str) -> dict[str, Any]:
    counts = Counter(str(item[field]) for item in records)
    values = list(counts.values())
    total = sum(values)
    probabilities = [count / total for count in values]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    max_entropy = math.log(len(values)) if len(values) > 1 else 0.0
    return {
        "category_count": len(counts),
        "counts": dict(sorted(counts.items())),
        "min_count": min(values),
        "max_count": max(values),
        "mean_count": statistics.fmean(values),
        "max_to_min_ratio": max(values) / min(values),
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 1.0,
    }


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_COVERAGE_BALANCE_REQUEST_V1", "schema drift")
    require(value.get("source_split") == "TRAIN", "coverage audit is TRAIN-only")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    base = _records(value.get("base_windows"), "base_windows")
    added = _records(value.get("added_windows"), "added_windows")
    require(not {row["window_id"] for row in base}.intersection(row["window_id"] for row in added), "added window IDs overlap base")
    combined = [*base, *added]
    dimensions = ("target_id", "source_episode_id", "initial_state_bin")
    return {
        "schema": "STAGE45F_COVERAGE_BALANCE_AUDIT_V1",
        "status": "PASS_DESCRIPTIVE_MANIFEST_ANALYSIS",
        "decision_state": "PI_DECISION_REQUIRED",
        "base_window_count": len(base),
        "added_window_count": len(added),
        "combined_window_count": len(combined),
        "base": {field: _summary(base, field) for field in dimensions},
        "added": {field: _summary(added, field) for field in dimensions},
        "combined": {field: _summary(combined, field) for field in dimensions},
        "automatic_balance_threshold_used": False,
        "automatic_resampling_authorized": False,
        "data_collection_authorized": False,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Initial-state bins are caller-defined descriptive strata, not learned clusters or causal states. This tool reports concentration/coverage only and does not choose collection or sampling weights.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(analyze(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CoverageBalanceError as exc:
        print(f"STAGE45F_COVERAGE_BALANCE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
