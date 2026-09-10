#!/usr/bin/env python3
"""Generic manifest-driven checkpoint cohort evaluator.

The command has two deliberately separate modes:

* ``--plan`` validates a future experiment without model/runtime access.
* ``--raw-predictions`` consumes a stage-specific raw-prediction handoff,
  persists each raw ``[1,40,132]`` array, then uses the accepted CPU decoder
  and metric library.  It never constructs or trains a model.

Historical Stage45B/Stage45C wrappers remain frozen; this is the canonical
future-facing entry point for new manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.reusable import (
    ReusableEvaluationError,
    build_evaluation_plan,
    load_evaluation_manifest,
    load_prediction_records,
    run_evaluation,
)
from go2arx5_vla.experiments.reusable import write_json


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan", action="store_true", help="validate and print a zero-inference plan"
    )
    mode.add_argument(
        "--raw-predictions", type=Path, help="JSON/JSONL stage-specific raw handoff"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="fresh output root for execution mode"
    )
    parser.add_argument(
        "--output", type=Path, help="write the plan/summary JSON here instead of stdout"
    )
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()

    try:
        manifest = load_evaluation_manifest(args.manifest)
        if args.plan:
            payload = build_evaluation_plan(manifest)
            if args.output is not None:
                write_json(args.output, payload)
            else:
                _print_json(payload)
            return 0

        if args.output_dir is None:
            parser.error("execution mode requires --output-dir")
        records = load_prediction_records(args.raw_predictions)
        summary = run_evaluation(
            manifest,
            records,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            records_base_dir=args.raw_predictions.resolve().parent,
            verify_artifacts=args.verify_artifacts,
        )
        if args.output is not None:
            write_json(args.output, summary)
        else:
            _print_json(summary)
        return 0
    except (ReusableEvaluationError, OSError, TypeError, ValueError) as exc:
        _print_json(
            {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
