#!/usr/bin/env python3
"""Stage45C checkpoint-evaluation adapter skeleton.

The current Corelab lane supports protocol admission and CPU evaluation of raw
predictions supplied by a future Golden Release runtime.  It deliberately has
no GR00T/Torch import and cannot create a model.  ``--plan-only`` is the only
mode valid for the unbound PI0088 manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.checkpoint_eval import (  # noqa: E402
    aggregate_evaluation_rows,
    evaluate_raw_prediction,
    load_evaluation_manifest,
    require_runtime_bound,
    runtime_is_bound,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    manifest = load_evaluation_manifest(args.manifest)
    plan = {
        "schema": "CORELAB_STAGE45C_EVALUATION_RUNNER_PLAN_V1",
        "manifest": str(args.manifest.resolve()),
        "manifest_status": manifest["status"],
        "runtime_bound": runtime_is_bound(manifest),
        "model_construction": False,
        "model_forward": False,
        "gpu_h200": False,
        "pbs_submitted": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if args.plan_only:
        payload = plan
    else:
        # Runtime identity admission is intentionally fail-closed.  A future
        # H200 adapter may produce the JSON raw-prediction handoff, but this
        # script will never manufacture that handoff or load a model itself.
        require_runtime_bound(manifest)
        if args.raw_predictions is None or args.stats_path is None or args.output is None:
            parser.error("bound evaluation requires --raw-predictions, --stats-path, and --output")
        raw_document = json.loads(args.raw_predictions.read_text(encoding="utf-8"))
        records = raw_document.get("records") if isinstance(raw_document, dict) else raw_document
        if not isinstance(records, list):
            raise ValueError("raw predictions must be a JSON list or {records:[...]}")
        rows = [evaluate_raw_prediction(record, stats_path=args.stats_path) for record in records]
        payload = aggregate_evaluation_rows(rows)
        payload["runner_plan"] = plan | {"model_forward": "external_runtime_only"}
    if args.output is not None:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
