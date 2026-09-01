#!/usr/bin/env python3
"""Aggregate the per-target Stage22 native expert oracle manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("episodes/*/expert_episode_summary.json"))
    if not paths:
        parser.error(f"no per-episode summaries found under {args.input_dir}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    statuses = Counter(item.get("status", "UNKNOWN") for item in summaries)
    terminations = Counter(item.get("termination", "UNKNOWN") for item in summaries)
    payload: dict[str, Any] = {
        "schema": "STAGE22_NATIVE_SAME_SCENE_EXPERT_COLLECTION_V1",
        "status": "PASS"
        if all(item.get("status") == "EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS" for item in summaries)
        else "DEVIATION",
        "simulation_only": True,
        "hardware_control": False,
        "episode_count": len(summaries),
        "target_ids": [item.get("target_id") for item in summaries],
        "snapshot_count": sum(int(item.get("snapshot_count", 0)) for item in summaries),
        "reference_updates": sum(int(item.get("reference_updates", 0)) for item in summaries),
        "steps": sum(int(item.get("steps", 0)) for item in summaries),
        "reset_events": sum(int(item.get("reset_events", 0)) for item in summaries),
        "status_breakdown": dict(statuses),
        "termination_breakdown": dict(terminations),
        "fixed_reset_seeds": sorted({int(item["fixed_reset_seed"]) for item in summaries if "fixed_reset_seed" in item}),
        "prefix_steps_frozen": sorted({int(item["prefix_steps_frozen"]) for item in summaries if "prefix_steps_frozen" in item}),
        "target_excluded_from_model_input": all(
            bool(item.get("target_excluded_from_model_input")) for item in summaries
        ),
        "per_episode": summaries,
        "source_manifests": [str(path) for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
