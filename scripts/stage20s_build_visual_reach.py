#!/usr/bin/env python3
"""Build the node-local SIM_VISUAL_REACH_V0 LeRobot/GR00T fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2arx5_vla.data import (
    CalibrationBundleV1,
    MaterializationConfigV1,
    MaterializationInputV1,
    SynchronizerV1,
    materialize_lerobot_v2,
)
from go2arx5_vla.simulation import build_visual_reach_raw, default_visual_reach_specs, write_visual_reach_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--family", choices=("EASY_VISUAL_REACH", "PROGRESSIVE_BASE_RELOCATION"), default="EASY_VISUAL_REACH")
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--train-episode-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.train_episode_count is not None and not 1 <= args.train_episode_count <= args.episodes:
        parser.error("--train-episode-count must be within the generated episode count")

    output = args.output_dir.expanduser().resolve()
    if output == Path("/") or len(output.parts) < 4:
        parser.error(f"refusing broad output path: {output}")
    frame_root = output / "source_frames"
    specs = default_visual_reach_specs(args.episodes, family=args.family, seed=args.seed)
    inputs = []
    synchronizer = SynchronizerV1()
    for spec in specs:
        raw = build_visual_reach_raw(spec, frame_root)
        sync = synchronizer.synchronize(raw)
        inputs.append(MaterializationInputV1(raw=raw, synchronization=sync, calibration=CalibrationBundleV1(
            bundle_id=raw.calibration_bundle_id,
            version="SIM_VISUAL_REACH_V0",
            status="SIMULATION_ONLY",
            arx5_joint_zero_version="SIM_REFERENCE_URDF_ZERO_V0",
            gripper_mapping={"status": "SIMULATION_ONLY", "normalized_center": 0.5},
            clock_sync_version="SIM_EXPLICIT_SHARED_TIMESTAMPS_V0",
            provenance={"source": "Stage20S procedural fixture"},
            real_robot_calibration=False,
        )))
    materialized = materialize_lerobot_v2(
        inputs,
        MaterializationConfigV1(
            output_dir=output / "lerobot",
            horizon=40,
            sample_stream="body",
            dataset_name=f"sim_visual_reach_v0_{args.family.lower()}",
            overwrite=args.overwrite,
            video_codec="mp4v",
            train_episode_count=args.train_episode_count,
            split_seed=args.seed,
        ),
    )
    manifest_path = output / "stage20s_sim_visual_reach_manifest.json"
    write_visual_reach_manifest(manifest_path, specs, dataset_path=materialized.output_dir)
    summary = {
        "schema": "STAGE20S_SIM_VISUAL_REACH_BUILD_RESULT_V1",
        "status": "PASS",
        "manifest": str(manifest_path),
        "materialized": materialized.as_dict(),
        "source_frames": str(frame_root),
        "hardware_control": False,
        "native_wbc_execution": False,
    }
    (output / "build_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
