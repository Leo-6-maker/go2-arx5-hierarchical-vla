#!/usr/bin/env python3
"""Run controlled synchronization/quality-gate stress cases on one D3 episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2arx5_vla.data import RawEpisodeV1, run_synchronization_stress


DEFAULT_D3_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-data-d3")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/synchronization_stress")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-id", default="d3_A_d2_ep_00_V0")
    args = parser.parse_args()
    source = args.d3_root / "raw" / "episodes" / args.episode_id / "episode.json"
    raw = RawEpisodeV1.from_d3_json(source)
    envelope = run_synchronization_stress(raw)[0]
    result = {
        "schema": "STAGE19_SYNCHRONIZATION_STRESS_RUN_V1",
        "status": "PASS" if envelope["source_unchanged"] else "FAIL_SOURCE_MUTATED",
        "source_episode": str(source),
        "source_episode_id": raw.episode_id,
        "stress": envelope,
        "no_hardware": True,
        "no_actuation": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "stage19_sync_stress.json"
    result["output_path"] = str(output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
