#!/usr/bin/env python3
"""Analyze existing checkpoint-evaluation rows without inference or mutation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.analyzer import (
    CheckpointAnalysisError,
    analyze_rows,
    load_rows,
    write_analysis,
)
from go2arx5_vla.experiments.reusable import load_json


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def _manifest_pairs(path: Path) -> list[dict[str, str]] | None:
    if path is None:
        return None
    document = load_json(path)
    analysis = document.get("analysis", {})
    if analysis is None:
        analysis = {}
    if not isinstance(analysis, dict):
        raise CheckpointAnalysisError("analysis manifest has no analysis section")
    pairs = analysis.get("paired_comparisons", [])
    if pairs is None:
        return []
    if not isinstance(pairs, list):
        raise CheckpointAnalysisError("analysis.paired_comparisons must be a list")
    return [dict(pair) for pair in pairs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="per_window.jsonl/JSON or evaluator output root",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help="override/add a paired comparison; repeatable",
    )
    args = parser.parse_args()
    try:
        rows, source_path = load_rows(args.input)
        if args.pair:
            pairs = [
                {
                    "name": f"pair_{index}_{left}_vs_{right}",
                    "left": left,
                    "right": right,
                }
                for index, (left, right) in enumerate(args.pair, start=1)
            ]
        else:
            pairs = _manifest_pairs(args.manifest) if args.manifest else None
        result = analyze_rows(rows, paired_comparisons=pairs)
        summary = write_analysis(
            result, output_dir=args.output_dir, source_path=source_path
        )
        _print_json(summary)
        return 0
    except (CheckpointAnalysisError, OSError, TypeError, ValueError) as exc:
        _print_json(
            {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
