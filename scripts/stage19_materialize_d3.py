#!/usr/bin/env python3
"""Materialize the six D3 episodes as a node-local GR00T/LeRobot v2 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2arx5_vla.data import (
    CalibrationBundleV1,
    EpisodeQualityGateConfig,
    EpisodeQualityGateV1,
    MaterializationConfigV1,
    MaterializationInputV1,
    RawEpisodeV1,
    SynchronizerConfigV1,
    SynchronizerV1,
    materialize_lerobot_v2,
)


DEFAULT_D3_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-data-d3")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/lerobot_fixture")


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
    parser.add_argument("--horizon", type=int, default=40, help="fixture horizon; not production timing")
    parser.add_argument("--train-episode-count", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.d3_root.is_dir():
        raise FileNotFoundError(args.d3_root)
    if args.episode_limit < 1 or args.horizon < 1:
        parser.error("episode-limit and horizon must be positive")

    synchronizer = SynchronizerV1(
        SynchronizerConfigV1(
            sample_stream="body",
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
    inputs: list[MaterializationInputV1] = []
    selected = select_episode_files(args.d3_root, args.episode_limit)
    for source_path in selected:
        raw = RawEpisodeV1.from_d3_json(source_path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        calibration = CalibrationBundleV1.from_d3_episode(payload)
        synchronization = synchronizer.synchronize(raw)
        quality = quality_gate.evaluate(
            raw,
            synchronization=synchronization,
            calibration=calibration,
        )
        if not quality.accepted:
            raise RuntimeError(
                f"D3 episode {raw.episode_id} rejected: {json.dumps(quality.as_dict())}"
            )
        inputs.append(
            MaterializationInputV1(
                raw=raw,
                synchronization=synchronization,
                calibration=calibration,
                quality=quality,
            )
        )

    materialized = materialize_lerobot_v2(
        inputs,
        MaterializationConfigV1(
            output_dir=args.output_dir,
            horizon=args.horizon,
            sample_stream="body",
            dataset_name="go2_arx5_stage19_d3_target",
            overwrite=args.overwrite,
            train_episode_count=args.train_episode_count,
        ),
    )
    result = {
        "schema": "STAGE19_D3_MATERIALIZATION_RUN_V1",
        "status": "PASS",
        "source_root": str(args.d3_root),
        "selected_episode_ids": [item.raw.episode_id for item in inputs],
        "quality_decisions": {
            item.raw.episode_id: item.quality.decision.value
            for item in inputs
            if item.quality is not None
        },
        "materialized": materialized.as_dict(),
        "source_episodes_unchanged": True,
        "hardware_control": False,
        "training_or_finetuning": False,
    }
    run_path = args.output_dir.parent / "stage19_materialization_run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    result["run_manifest"] = str(run_path)
    run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
