#!/usr/bin/env python3
"""Preparation-only retained-artifact storage planner for future campaigns.

The planner uses observed full-state checkpoint sizes and a caller-specified
safety margin to estimate scratch required when *all scientifically valid runs*
are retained. It distinguishes independent fresh run tiers from a resume-chain
campaign because those designs retain different numbers of run roots/checkpoints.
It never deletes artifacts, chooses a storage system, or authorizes training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


class StoragePlanError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StoragePlanError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _nonnegative_int(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{field} must be non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    require(result > 0, f"{field} must be positive int")
    return result


def _tiers(value: Any) -> list[dict[str, Any]]:
    require(isinstance(value, list) and value, "run_tiers must be non-empty list")
    result: list[dict[str, Any]] = []
    previous = 0
    for index, item in enumerate(value):
        require(isinstance(item, Mapping), f"run_tiers[{index}] must be object")
        max_steps = _positive_int(item.get("max_steps"), f"run_tiers[{index}].max_steps")
        require(max_steps > previous, "run_tiers must be strictly increasing by max_steps")
        previous = max_steps
        checkpoints = item.get("checkpoint_steps")
        require(isinstance(checkpoints, list) and checkpoints, f"run_tiers[{index}].checkpoint_steps missing")
        require(checkpoints == sorted(set(checkpoints)), f"run_tiers[{index}].checkpoint_steps must be increasing unique")
        require(all(isinstance(step, int) and not isinstance(step, bool) and 0 < step <= max_steps for step in checkpoints), f"run_tiers[{index}] invalid checkpoint step")
        require(checkpoints[-1] == max_steps, f"run tier {max_steps} terminal checkpoint must be retained")
        result.append({"max_steps": max_steps, "checkpoint_steps": list(checkpoints)})
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_STORAGE_RETENTION_REQUEST_V2", "schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    require(value.get("retain_all_valid_runs") is True, "all scientifically valid runs must be retained")
    observed = value.get("observed_full_state_checkpoint_bytes")
    require(isinstance(observed, list) and observed, "observed checkpoint sizes missing")
    require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in observed), "invalid checkpoint size")
    seeds = value.get("predeclared_seeds")
    require(isinstance(seeds, list) and seeds and len(seeds) == len(set(seeds)), "predeclared seeds invalid")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds), "invalid seed")
    tiers = _tiers(value.get("run_tiers"))
    execution_mode = value.get("tier_execution_mode")
    require(execution_mode in {"INDEPENDENT_FRESH_RUNS", "RESUME_CHAIN"}, "tier_execution_mode invalid")
    non_checkpoint_per_run = _nonnegative_int(value.get("estimated_non_checkpoint_bytes_per_run"), "estimated_non_checkpoint_bytes_per_run")
    available = _positive_int(value.get("available_bytes_verified"), "available_bytes_verified")
    margin = value.get("safety_margin_fraction")
    require(isinstance(margin, (int, float)) and not isinstance(margin, bool), "safety_margin_fraction must be numeric")
    margin = float(margin)
    require(math.isfinite(margin) and margin >= 0.0, "safety_margin_fraction invalid")

    if execution_mode == "INDEPENDENT_FRESH_RUNS":
        checkpoint_count_per_seed = sum(len(tier["checkpoint_steps"]) for tier in tiers)
        run_count = len(seeds) * len(tiers)
    else:
        checkpoint_count_per_seed = len(sorted({step for tier in tiers for step in tier["checkpoint_steps"]}))
        run_count = len(seeds)
    checkpoint_count = len(seeds) * checkpoint_count_per_seed
    conservative_checkpoint_bytes = max(observed)
    raw_required = conservative_checkpoint_bytes * checkpoint_count + non_checkpoint_per_run * run_count
    required_with_margin = math.ceil(raw_required * (1.0 + margin))
    return {
        "schema": "STAGE45F_STORAGE_RETENTION_PLAN_V2",
        "status": "PASS_ESTIMATE_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "observed_checkpoint_bytes": {
            "count": len(observed),
            "min": min(observed),
            "median": statistics.median(observed),
            "max": max(observed),
            "conservative_size_used": conservative_checkpoint_bytes,
        },
        "tier_execution_mode": execution_mode,
        "run_tiers": tiers,
        "predeclared_seed_count": len(seeds),
        "retained_run_root_count": run_count,
        "retained_checkpoint_count": checkpoint_count,
        "estimated_non_checkpoint_bytes_per_run": non_checkpoint_per_run,
        "raw_required_bytes": raw_required,
        "safety_margin_fraction": margin,
        "required_bytes_estimate": required_with_margin,
        "available_bytes_verified": available,
        "descriptive_capacity_sufficient": available >= required_with_margin,
        "retain_all_valid_runs": True,
        "automatic_deletion_allowed": False,
        "delete_worse_valid_runs_allowed": False,
        "training_authorized": False,
        "storage_selected": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "This is a conservative byte estimate from observed checkpoint sizes. Independent fresh tiers and resume chains have different retention footprints. The plan does not authorize deleting evidence or selecting a storage target.",
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
    except StoragePlanError as exc:
        print(f"STAGE45F_STORAGE_PLAN_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
