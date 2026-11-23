"""Post-registration identifiability check. A bounded continuation with the same in-support predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis import d3, fit, pinned, read, sha


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("registration", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    cfg = read(args.registration)
    source = args.root / "v15/d3-verified01/D3_VERIFIED.json"
    pinned(source, cfg["source_D3_sha256"])
    cards = d3(args.root, read(source))
    cases = []
    for card in cards:
        training = [d for d in cards if d is not card]
        _, lag, m, _ = fit(training)
        u = np.concatenate([d["u"] for d in training])
        # Projection onto the closed convex hull fixes every observed input. The norm
        # of a linear image of any convex combination is bounded by the largest vertex norm.
        max_effect = float(np.max(np.linalg.norm(u @ m[:, :3], axis=1)))
        initial = float(1000 * np.linalg.norm(card["actual"] - card["old"]))
        cases.append(
            {
                "id": card["summary"]["id"],
                "lag_ms": 2 * lag,
                "initial_position_error_mm": initial,
                "largest_possible_position_response_mm": max_effect,
                "position_error_lower_bound_mm": max(0.0, initial - max_effect),
                "maximum_training_command_translation_mm": float(
                    np.linalg.norm(u[:, :2], axis=1).max()
                ),
            }
        )
    with args.output.open("x") as f:
        json.dump(
            {
                "scope": "EXPLORATORY_IDENTIFIABILITY_COUNTEREXAMPLE",
                "audit_sha256": sha(Path(__file__)),
                "method": "Project observed-input columns onto the training convex hull before applying the fitted matrix. Set three unidentified columns to zero. Exactly the unbounded zero-diagonal model on training support. A mathematical uncertainty witness; not a measured saturation or a native controller.",
                "records": cases,
                "native_requests": 0,
                "new_physics_ticks": 0,
            },
            f,
            allow_nan=False,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
