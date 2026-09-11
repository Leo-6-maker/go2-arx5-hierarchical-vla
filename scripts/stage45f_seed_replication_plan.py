#!/usr/bin/env python3
"""Preparation-only seed replication planner for a frozen candidate recipe.

The canonical Stage45E replication family is 2222/2223/2224.  This tool only
reports missing seeds from that predeclared family; it never launches training
and never replaces a valid completed seed based on performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CANONICAL_SEEDS = (2222, 2223, 2224)


class SeedReplicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedReplicationError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plan(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_SEED_REPLICATION_REQUEST_V1", "request schema drift")
    require(value.get("recipe_frozen") is True, "replication requires frozen recipe")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    recipe_identity = value.get("recipe_identity")
    require(isinstance(recipe_identity, str) and recipe_identity, "recipe_identity missing")
    completed = value.get("completed_valid_seeds")
    require(isinstance(completed, list), "completed_valid_seeds must be list")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in completed), "completed seed invalid")
    require(len(set(completed)) == len(completed), "completed seeds duplicate")
    require(set(completed).issubset(set(CANONICAL_SEEDS)), "completed seed outside canonical family")
    invalid = value.get("scientifically_invalid_attempts", [])
    require(isinstance(invalid, list), "scientifically_invalid_attempts must be list")
    for index, attempt in enumerate(invalid):
        require(isinstance(attempt, Mapping), f"invalid attempt {index} must be object")
        require(isinstance(attempt.get("seed"), int), f"invalid attempt {index} seed missing")
        require(isinstance(attempt.get("reason"), str) and attempt["reason"], f"invalid attempt {index} reason missing")
    missing = [seed for seed in CANONICAL_SEEDS if seed not in completed]
    return {
        "schema": "STAGE45F_SEED_REPLICATION_PLAN_V1",
        "status": "PASS_PLAN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "recipe_identity": recipe_identity,
        "canonical_seed_family": list(CANONICAL_SEEDS),
        "completed_valid_seeds": list(completed),
        "missing_canonical_seeds": missing,
        "scientifically_invalid_attempts_preserved": [dict(item) for item in invalid],
        "replacement_by_metric_allowed": False,
        "valid_completed_seed_discard_allowed": False,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Missing canonical seeds are reported from a predeclared family; no seed is chosen because of observed performance.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(plan(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeedReplicationError as exc:
        print(f"STAGE45F_SEED_REPLICATION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
