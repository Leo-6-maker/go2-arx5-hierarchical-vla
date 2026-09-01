#!/usr/bin/env python3
"""Build a small, node-local Stage18 D3 data-readiness fixture.

The source D3 tree is never modified.  This script records paths, hashes,
schemas, synchronization metrics, and derived examples under a separate
output root; it does not copy images or private/model files into Git.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from go2arx5_vla.data import (
    CalibrationBundleV1,
    EpisodeQualityGateConfig,
    EpisodeQualityGateV1,
    GR00TTrainingExampleV1,
    RawEpisodeV1,
    SynchronizerConfigV1,
    SynchronizerV1,
)
from go2arx5_vla.data.processor import dataset_statistics, examples_to_lerobot_records
from go2arx5_vla.data.readiness import sha256_file


DEFAULT_D3_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-data-d3")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage18/d3_fixture")


def select_episode_files(d3_root: Path, limit: int) -> list[Path]:
    candidates = sorted((d3_root / "raw" / "episodes").glob("*/episode.json"))
    v0 = [path for path in candidates if path.parent.name.endswith("_V0")]
    selected = v0 if len(v0) >= limit else candidates
    if len(selected) < limit:
        raise RuntimeError(f"D3 contains only {len(selected)} usable episodes; need {limit}")
    return selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-limit", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=40, help="fixture horizon; not a production timing assumption")
    args = parser.parse_args()
    if args.episode_limit < 1 or args.horizon < 1:
        parser.error("episode-limit and horizon must be positive")
    if not args.d3_root.is_dir():
        raise FileNotFoundError(args.d3_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    episode_files = select_episode_files(args.d3_root, args.episode_limit)
    examples: list[GR00TTrainingExampleV1] = []
    episode_records: list[dict] = []

    synchronizer = SynchronizerV1(
        SynchronizerConfigV1(
            sample_stream="body",
            # These are pipeline fixture thresholds, not robot safety limits.
            max_skew_warn_s=0.010,
            max_skew_reject_s=0.050,
        )
    )
    quality_gate = EpisodeQualityGateV1(
        EpisodeQualityGateConfig(
            check_camera_files=True,
            simulation_fixture_allowed=True,
        )
    )

    for source_path in episode_files:
        raw = RawEpisodeV1.from_d3_json(source_path)
        payload = json.loads(source_path.read_text())
        calibration = CalibrationBundleV1.from_d3_episode(payload)
        sync = synchronizer.synchronize(raw)
        quality = quality_gate.evaluate(raw, synchronization=sync, calibration=calibration)
        if not quality.accepted:
            raise RuntimeError(
                f"D3 fixture episode {raw.episode_id} rejected: {json.dumps(quality.as_dict())}"
            )
        example = GR00TTrainingExampleV1.from_raw(raw, sync, sample_index=0, horizon=args.horizon)
        examples.append(example)
        (examples_dir / f"{raw.episode_id}.json").write_text(
            json.dumps(example.as_dict(), indent=2) + "\n"
        )
        episode_records.append(
            {
                "episode_id": raw.episode_id,
                "trajectory_id": raw.trajectory_id,
                "source_json": str(source_path),
                "source_json_sha256": sha256_file(source_path),
                "raw_summary": raw.as_dict(include_arrays=False, include_camera_paths=False),
                "calibration": calibration.as_dict(),
                "synchronization": sync.as_dict(),
                "quality": quality.as_dict(),
                "derived_example": example.example_id,
                "derived_action_label": "desired",
                "achieved_eef_retained_in_raw_only": True,
            }
        )

    records = examples_to_lerobot_records(examples)
    (args.output_dir / "lerobot_compatible_records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    stats = dataset_statistics(examples)
    (args.output_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    decisions = Counter(record["quality"]["decision"] for record in episode_records)
    first_sync = episode_records[0]["synchronization"]
    manifest = {
        "schema": "STAGE18_D3_FIXTURE_MANIFEST_V1",
        "status": "SIMULATION_PROXY_ONLY",
        "d3_root": str(args.d3_root),
        "source_episodes_unchanged": True,
        "selected_episode_count": len(episode_records),
        "selected_episode_ids": [record["episode_id"] for record in episode_records],
        "episode_files": [record["source_json"] for record in episode_records],
        "episode_records": episode_records,
        "quality_decisions": dict(decisions),
        "derived_example_count": len(examples),
        "fixture_horizon": args.horizon,
        "fixture_horizon_provenance": "TEST_ONLY_D3_PIPELINE_FIXTURE",
        "state_abi": "StateV1 = EEF XYZ 3 + GR00T_FIRST_TWO_ROWS rot6d 6 + gripper 1",
        "action_abi": "future desired EEF XYZ+rot6d rows [H,9] + gripper [H,1]",
        "target_achieved_separate": True,
        "achieved_eef_used_as_training_label": False,
        "camera_rate_audit": {
            "declared_rates_hz": first_sync["declared_rates_hz"],
            "effective_rates_hz": first_sync["effective_rates_hz"],
            "warnings": first_sync["warnings"],
            "source_timestamp_truth": True,
            "known_issue": "D3_CAMERA_FPS_METADATA_MISMATCH",
        },
        "lerobot_compatibility": {
            "logical_records": str(args.output_dir / "lerobot_compatible_records.jsonl"),
            "statistics": str(args.output_dir / "statistics.json"),
            "official_processor_model_forward": False,
            "official_processor_smoke": "separate_stage18_processor_smoke.py",
        },
        "real_robot_calibration": False,
        "wbc_closed_loop": False,
        "groot_training_approved": False,
        "no_image_copy": True,
    }
    (args.output_dir / "stage18_d3_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
