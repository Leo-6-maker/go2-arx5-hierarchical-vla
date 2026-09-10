#!/usr/bin/env python3
"""Validate and plan a manifest-driven adaptation-training handoff.

This generic entry point intentionally stops at the backend boundary.  It
cannot silently choose a recipe, resume a checkpoint, construct a model, or
start training.  A stage-specific wrapper may execute the explicit plan after
its own runtime admission and must seal the resulting full trainer state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.training.reusable import (
    ReusableTrainingError,
    load_training_manifest,
    write_training_plan,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write plan JSON here")
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    try:
        manifest = load_training_manifest(args.manifest)
        if args.output is not None:
            plan = write_training_plan(
                manifest,
                output=args.output,
                manifest_path=args.manifest,
                verify_artifacts=args.verify_artifacts,
            )
        else:
            # Use the same library path as file output, but retain a pure
            # stdout mode for CI and login-node admission checks.
            from go2arx5_vla.training.reusable import (
                build_training_plan,
                verify_training_artifacts,
            )

            plan = build_training_plan(manifest)
            plan["verified_artifacts"] = (
                verify_training_artifacts(manifest, manifest_path=args.manifest)
                if args.verify_artifacts
                else {"skipped": True}
            )
        _print_json(plan)
        return 0
    except (ReusableTrainingError, OSError, TypeError, ValueError) as exc:
        _print_json(
            {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
