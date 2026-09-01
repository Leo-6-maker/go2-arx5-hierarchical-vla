#!/usr/bin/env python3
"""Materialize Stage22 native expert snapshots as an official-loader fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2arx5_vla.data import (
    CalibrationBundleV1,
    MaterializationConfigV1,
    MaterializationInputV1,
    SynchronizerV1,
    dataset_statistics,
    load_snapshot,
    materialize_lerobot_v2,
    snapshot_to_training_example,
    snapshot_to_raw_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-target-count", type=int, default=4)
    parser.add_argument("--train-target-ids", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    snapshot_paths = sorted(input_dir.glob("snapshots/*/*.json"))
    if not snapshot_paths:
        parser.error(f"no snapshot descriptors found below {input_dir / 'snapshots'}")
    snapshots = [load_snapshot(path) for path in snapshot_paths]
    target_ids = []
    for snapshot in snapshots:
        if snapshot.target_id not in target_ids:
            target_ids.append(snapshot.target_id)
    if args.train_target_ids is None:
        train_target_ids = [target for target in target_ids if "train" in target.lower()]
        if not train_target_ids:
            train_target_ids = target_ids[: int(args.train_target_count)]
    else:
        train_target_ids = [str(target) for target in args.train_target_ids]
        if any(target not in target_ids for target in train_target_ids):
            parser.error("--train-target-ids must refer to targets present in the snapshots")
    train_targets = set(train_target_ids)
    if not 1 <= len(train_targets) < len(target_ids):
        parser.error("training target selection must leave at least one held-out target")
    ordered_target_ids = list(train_target_ids) + [target for target in target_ids if target not in train_targets]
    snapshots = [
        snapshot
        for target_id in ordered_target_ids
        for snapshot in snapshots
        if snapshot.target_id == target_id
    ]
    calibration_cache: dict[str, CalibrationBundleV1] = {}
    inputs = []
    logical_examples = []
    synchronization = SynchronizerV1()
    for snapshot in snapshots:
        calibration = calibration_cache.setdefault(
            snapshot.calibration_bundle_id,
            CalibrationBundleV1(
                bundle_id=snapshot.calibration_bundle_id,
                version="STAGE22_NATIVE_SIM_V1",
                status="SIMULATION_ONLY",
                arx5_joint_zero_version="SIM_REFERENCE_URDF_ZERO_V0",
                gripper_mapping={"status": "SIMULATION_ONLY", "normalized_center": 0.5},
                clock_sync_version="NATIVE_SIM_EXPLICIT_TIMESTAMP_V1",
                provenance={
                    "source": "Stage22 native same-scene expert collector",
                    "physical_calibration": False,
                },
                real_robot_calibration=False,
            ),
        )
        raw = snapshot_to_raw_episode(snapshot, calibration=calibration)
        sync = synchronization.synchronize(raw)
        inputs.append(MaterializationInputV1(raw=raw, synchronization=sync, calibration=calibration))
        logical_examples.append(snapshot_to_training_example(snapshot))

    materialized = materialize_lerobot_v2(
        inputs,
        MaterializationConfigV1(
            output_dir=output_dir,
            horizon=40,
            sample_stream="body",
            dataset_name="stage22_native_same_scene_expert_v1",
            overwrite=bool(args.overwrite),
            video_codec="mp4v",
            train_episode_count=sum(snapshot.target_id in train_targets for snapshot in snapshots),
            split_seed=2201,
        ),
    )
    payload = {
        "schema": "STAGE22_NATIVE_EXPERT_MATERIALIZATION_V1",
        "status": "PASS",
        "input_dir": str(input_dir),
        "output_dir": str(materialized.output_dir),
        "snapshot_count": len(snapshots),
        "target_count": len(target_ids),
        "train_target_ids": list(train_target_ids),
        "validation_target_ids": [target for target in ordered_target_ids if target not in train_targets],
        "train_example_count": sum(snapshot.target_id in train_targets for snapshot in snapshots),
        "validation_example_count": sum(snapshot.target_id not in train_targets for snapshot in snapshots),
        "horizon": 40,
        "state_abi": "StateV1 = EEF XYZ[3] + EEF rot6d[6] + gripper[1]",
        "action_abi": "future desired EEF[40,9] + future desired gripper[40,1]",
        "target_excluded_from_model_input": True,
        "snapshot_expanded_horizon": True,
        "source_timestamp_authority": True,
        "logical_statistics": dataset_statistics(logical_examples),
        "materialized": materialized.as_dict(),
        "hardware_control": False,
        "training": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "stage22_materialization_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
