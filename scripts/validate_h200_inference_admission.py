#!/usr/bin/env python3
"""Run the static H200 inference-admission cross-binding preflight.

This command verifies a resolved stage-specific admission document before any
future model adapter is allowed to import or construct a model.  It performs
no H200, GPU, PBS, model-forward, or training work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.h200_admission import (
    H200AdmissionError,
    validate_h200_inference_admission,
)
from go2arx5_vla.evaluation.reusable import load_evaluation_manifest
from go2arx5_vla.experiments.reusable import load_json, write_json


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_evaluation_manifest(args.manifest)
        admission = load_json(args.admission)
        result = validate_h200_inference_admission(
            manifest,
            admission,
            manifest_path=args.manifest,
            require_adapter_binding=True,
        )
        if args.output is not None:
            write_json(args.output, result)
        else:
            _print_json(result)
        return 0
    except (H200AdmissionError, OSError, TypeError, ValueError) as exc:
        _print_json(
            {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
