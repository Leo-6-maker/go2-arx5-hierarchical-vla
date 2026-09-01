#!/usr/bin/env python3
"""Qualify RecorderV1 with a D3 replay backend; never opens a robot link."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from go2arx5_vla.data import RawEpisodeV1, replay_raw_episode


DEFAULT_D3_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-data-d3")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/recorder_replay")


def _fingerprint(raw: RawEpisodeV1) -> str:
    payload = json.dumps(
        raw.as_dict(include_arrays=True, include_camera_paths=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-id", default="d3_A_d2_ep_00_V0")
    args = parser.parse_args()
    source = args.d3_root / "raw" / "episodes" / args.episode_id / "episode.json"
    raw = RawEpisodeV1.from_d3_json(source)
    before = _fingerprint(raw)
    recorded = replay_raw_episode(raw)
    after = _fingerprint(raw)
    result = {
        "schema": "STAGE19_RECORDER_REPLAY_RUN_V1",
        "status": "PASS" if before == after else "FAIL_SOURCE_MUTATED",
        "source_episode": str(source),
        "source_episode_id": raw.episode_id,
        "source_fingerprint_before": before,
        "source_fingerprint_after": after,
        "recorded": recorded.as_dict(),
        "native_state_count_expected": raw.frame_count,
        "native_state_count_recorded": recorded.state_count,
        "desired_command_count_expected": raw.frame_count,
        "desired_command_count_recorded": len(recorded.desired_commands),
        "camera_counts_expected": {name: stream.frame_count for name, stream in raw.camera_streams.items()},
        "camera_counts_recorded": {name: len(rows) for name, rows in recorded.camera_samples.items()},
        "desired_label_preserved_separately": True,
        "actuation_backend": "NONE_READ_ONLY_RECORDING_ONLY",
    }
    if recorded.state_count != raw.frame_count or len(recorded.desired_commands) != raw.frame_count:
        raise RuntimeError(json.dumps(result))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "stage19_recorder_replay.json"
    result["output_path"] = str(output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
