#!/usr/bin/env python3
"""Preparation-only 2x2 N2/L1 interaction design.

If N2 and L1 each show partial effects, a future combined arm may be useful,
but it must be treated as a separately gated interaction experiment rather
than an automatic composition. This planner freezes the four conceptual cells
and matched-seed requirements without authorizing N2+L1 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CELLS = {
    "A_LEGACY": {"normalization": "FULL24", "loss": "LEGACY"},
    "B_N2": {"normalization": "N2_BALANCED_XYZ", "loss": "LEGACY"},
    "C_L1": {"normalization": "FULL24", "loss": "L1_PHYSICAL_AWARE"},
    "D_N2_PLUS_L1": {"normalization": "N2_BALANCED_XYZ", "loss": "L1_PHYSICAL_AWARE"},
}
CANONICAL_SEEDS = (2222, 2223, 2224)


class FactorialPlanError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FactorialPlanError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plan(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_N2_L1_FACTORIAL_REQUEST_V1", "schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    evidence = value.get("existing_cells")
    require(isinstance(evidence, Mapping), "existing_cells missing")
    normalized: dict[str, Any] = {}
    for cell in ("A_LEGACY", "B_N2", "C_L1"):
        record = evidence.get(cell)
        require(isinstance(record, Mapping), f"existing cell {cell} missing")
        seeds = record.get("valid_training_seeds")
        require(isinstance(seeds, list), f"{cell}.valid_training_seeds must be list")
        require(all(seed in CANONICAL_SEEDS for seed in seeds), f"{cell} seed outside canonical family")
        require(len(set(seeds)) == len(seeds), f"{cell} duplicate seed")
        recipe_identity = record.get("recipe_identity")
        require(isinstance(recipe_identity, str) and recipe_identity, f"{cell}.recipe_identity missing")
        normalized[cell] = {"valid_training_seeds": sorted(seeds), "recipe_identity": recipe_identity}

    common_all_existing_cells = sorted(
        set(normalized["A_LEGACY"]["valid_training_seeds"])
        .intersection(normalized["B_N2"]["valid_training_seeds"])
        .intersection(normalized["C_L1"]["valid_training_seeds"])
    )
    missing_for_full_matched = [seed for seed in CANONICAL_SEEDS if seed not in common_all_existing_cells]
    per_cell_missing = {
        cell: [seed for seed in CANONICAL_SEEDS if seed not in normalized[cell]["valid_training_seeds"]]
        for cell in ("A_LEGACY", "B_N2", "C_L1")
    }
    return {
        "schema": "STAGE45F_N2_L1_FACTORIAL_PLAN_V2",
        "status": "PASS_PLAN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "cells": CELLS,
        "existing_cells": normalized,
        "future_combined_cell": "D_N2_PLUS_L1",
        "combined_training_authorized": False,
        "canonical_seed_family": list(CANONICAL_SEEDS),
        "common_matched_seeds_across_A_B_C": common_all_existing_cells,
        "missing_seeds_for_fully_matched_A_B_C_context": missing_for_full_matched,
        "missing_canonical_seeds_by_existing_cell": per_cell_missing,
        "future_combined_arm_requirement": {
            "fresh_base": True,
            "same_dataset": True,
            "same_optimizer_scheduler_precision_batch": True,
            "same_canonical_seed_family": True,
            "matched_seed_interaction_requires_A_B_C_D_same_seed": True,
            "normalization_change": "N2 only",
            "loss_change": "L1 only",
            "all_other_science_variables_frozen": True,
        },
        "descriptive_interaction_contrast": "D_N2_PLUS_L1 - B_N2 - C_L1 + A_LEGACY",
        "interaction_claim_boundary": "Difference-of-differences is descriptive only on the same training seed and matched TRAIN windows across all four cells. Nonlinear training means additivity/causality is not assumed.",
        "automatic_combination_allowed": False,
        "validation_e_read": False,
        "holdout_f_read": False,
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
    except FactorialPlanError as exc:
        print(f"STAGE45F_N2_L1_FACTORIAL_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
