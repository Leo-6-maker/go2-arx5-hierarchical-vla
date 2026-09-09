#!/usr/bin/env python3
"""Aggregate already materialized Stage45C per-window CPU evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.checkpoint_eval import (  # noqa: E402
    aggregate_evaluation_rows,
    load_evaluation_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_evaluation_manifest(args.manifest)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    rows = document.get("rows") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON list or {rows:[...]}")
    aggregate = aggregate_evaluation_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "window_count": aggregate["window_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
