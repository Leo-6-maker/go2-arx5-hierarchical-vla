#!/usr/bin/env python3
"""Preparation-only planner for data-cardinality vs optimizer-step scaling.

When a TRAIN set grows, keeping optimizer steps fixed reduces nominal exposure
per window under uniform one-window-per-step sampling; scaling steps with
cardinality changes optimization budget. This tool reports both views under
explicit assumptions and never calls either an epoch-equivalent experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class DataComputeScalingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataComputeScalingError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{field} must be positive int")
    return value


def plan(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_DATA_COMPUTE_SCALING_REQUEST_V1", "schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    require(value.get("sampling_assumption") == "UNIFORM_ONE_WINDOW_EVENT_PER_OPTIMIZER_STEP", "unsupported sampling assumption")
    base_windows = _positive_int(value.get("base_window_count"), "base_window_count")
    expanded_windows = _positive_int(value.get("expanded_window_count"), "expanded_window_count")
    base_steps = _positive_int(value.get("base_optimizer_steps"), "base_optimizer_steps")
    proposed_fixed_steps = _positive_int(value.get("proposed_fixed_optimizer_steps"), "proposed_fixed_optimizer_steps")
    require(expanded_windows >= base_windows, "expanded_window_count must not shrink base")

    base_nominal = base_steps / base_windows
    fixed_nominal = proposed_fixed_steps / expanded_windows
    cardinality_matched_steps_float = base_nominal * expanded_windows
    cardinality_matched_steps_integer = round(cardinality_matched_steps_float)
    exact_integer_match = abs(cardinality_matched_steps_integer - cardinality_matched_steps_float) < 1e-12
    return {
        "schema": "STAGE45F_DATA_COMPUTE_SCALING_PLAN_V1",
        "status": "PASS_ASSUMPTION_BOUND_PLAN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "sampling_assumption": "UNIFORM_ONE_WINDOW_EVENT_PER_OPTIMIZER_STEP",
        "base": {
            "window_count": base_windows,
            "optimizer_steps": base_steps,
            "nominal_expected_events_per_window": base_nominal,
        },
        "fixed_step_expanded_data": {
            "window_count": expanded_windows,
            "optimizer_steps": proposed_fixed_steps,
            "nominal_expected_events_per_window": fixed_nominal,
            "relative_nominal_per_window_exposure_vs_base": fixed_nominal / base_nominal,
        },
        "cardinality_scaled_hypothetical": {
            "window_count": expanded_windows,
            "optimizer_steps_float_for_same_nominal_expected_events_per_window": cardinality_matched_steps_float,
            "nearest_integer_optimizer_steps": cardinality_matched_steps_integer,
            "exact_integer_match": exact_integer_match,
            "execution_authorized": False,
        },
        "epoch_claim_allowed": False,
        "exact_realized_exposure_claim_allowed": False,
        "exact_realized_exposure_requires_trace": True,
        "scheduler_lr_history_must_be_audited_separately": True,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "These are nominal expectation calculations under an explicit uniform-sampling assumption. They are not realized exposure, epoch equivalence, or authorization to scale optimizer steps.",
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
    except DataComputeScalingError as exc:
        print(f"STAGE45F_DATA_COMPUTE_SCALING_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
