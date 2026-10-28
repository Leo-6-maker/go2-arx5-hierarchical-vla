"""Constructed 1-D liveness probe, NOT a replay or identified model of the WBC.

Specializes the PI0266 translational teacher to a straight 100-mm path,
a six-second quintic clock, one-second residual blend and 0.4-s replanning.
No model, simulator, robot data, private runtime or external dependency is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DISTANCE = 0.1
REPLAN = 0.4


def quintic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 10*x**3 - 15*x**4 + 6*x**5


def path(t: float) -> float:
    return DISTANCE * quintic(t / 6.0)


def phase(x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= DISTANCE:
        return 6.0
    lo, hi = 0.0, 6.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if path(mid) < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def proposal(x: float, variant: str) -> float:
    if variant == "PI0266_POSITION_SPECIALIZATION":
        s = phase(x)
        return path(s + REPLAN) + (1 - quintic(REPLAN)) * (x - path(s))
    if variant == "CONSTANT_ARCLENGTH_EXPONENTIAL_CORRECTION":
        projected = max(0.0, min(DISTANCE, x))
        return (min(DISTANCE, projected + 0.02 * REPLAN)
                + math.exp(-2 * REPLAN) * (x - projected))
    raise ValueError(variant)


def run(variant: str, error: float) -> dict:
    if not math.isfinite(error) or error < 0:
        raise ValueError("Tracking error must be finite and nonnegative")
    x = 0.0
    first = None
    near = None
    for i in range(100):
        reference = proposal(x, variant)
        # Explicit toy executor: end-of-prefix position is reference minus error.
        x = reference - error
        if i == 0:
            first = x
        if abs(x - DISTANCE) <= 0.005 and near is None:
            near = (i + 1) * REPLAN
    return dict(variant=variant, endpoint_tracking_error_m=error,
                first_executed_progress_m=first, endpoint_m=x,
                target_error_m=abs(x-DISTANCE), time_within_5mm_s=near,
                elapsed_s=40.0)


def evaluate() -> dict:
    variants = ("PI0266_POSITION_SPECIALIZATION",
                "CONSTANT_ARCLENGTH_EXPONENTIAL_CORRECTION")
    rows = [run(v, b) for v in variants for b in (0.0, 0.0001, 0.0005)]
    expected = (path(REPLAN) - 0.0005) / quintic(REPLAN)
    observed = next(r for r in rows if r["variant"] == variants[0]
                    and r["endpoint_tracking_error_m"] == 0.0005)
    assert abs(observed["endpoint_m"] - expected) < 1e-12
    assert observed["time_within_5mm_s"] is None
    assert all(r["target_error_m"] < 0.001 for r in rows
               if r["variant"] == variants[1])
    return dict(
        scope="CONSTRUCTED_1D_COUNTEREXAMPLE_NOT_WBC_REPLAY",
        source_commit="bd250056efe058452981c7460dd3792aa2b23eca",
        source_files=["scripts/pi0266_expert.py", "scripts/pi0259_kinematic.py"],
        first_04s_nominal_m=path(REPLAN),
        blend_removed_fraction=quintic(REPLAN),
        analytic_stalled_position_m=expected,
        rows=rows,
        caveats=[
            "Endpoint bias is a chosen toy executor, not an identified WBC.",
            "SO3, preview, collision, measurement and native time binding are omitted.",
            "The alternative is a candidate task-space rule, not qualified robot control.",
            "No historical experimental verdict changes."],
        model_queries=0, training_updates=0, robot_physics_steps=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(evaluate(), indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite existing evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
