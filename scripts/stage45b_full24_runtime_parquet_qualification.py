#!/usr/bin/env python3
"""Qualify the Stage45B runtime-parquet provenance V2 contract.

This is a login-node, non-GPU qualification.  It independently checks the
sealed FULL24 TRAIN parquet against the deterministic Stage22 smoothstep law,
uses the same label-only anchor reconciliation and StructuredActionV1 codec as
the H200 runner, and computes the V2 q01/q99 statistics artifact.  It never
loads BASE weights, imports GR00T, reads Validation-E/Holdout-F, or mutates the
canonical parquet tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


PI_GATE = "PI_GATE_STATE_0083"
PROTOCOL_REVISION = "RUNTIME_PARQUET_PROVENANCE_V2"
STAGE = "Stage45B"
BRANCH = "stage45b/full24-fresh-base-execution"
EXPECTED_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
EXPECTED_FILE_COUNT = 85
EXPECTED_TOTAL_BYTES = 919661
EXPECTED_B0_SHA256 = "49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d"
EXPECTED_B0_PAYLOAD_SHA256 = "c31c818121ac01a402fc70e577db225ffb6f5d110649aef9dada94338c8a5032"
EXPECTED_STAGE22_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
EXPECTED_STAGE22_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
EXPECTED_SPLIT_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
EXPECTED_TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
EXPECTED_LANGUAGE = "move the end effector to the red target"
EXPECTED_SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
EXPECTED_TCP = "x5_link6_gripper_center"
EXPECTED_HORIZON = 40
EXPECTED_STATE_DIM = 10
EXPECTED_PHYSICAL_DIM = 10
EXPECTED_ACTION_DIM = 132
EXPECTED_TIMING_MODE = "T16"
EXPECTED_RETURN_TOKEN = "VANDA_STAGE45B_FULL24_RUNTIME_DATASET_QUALIFIED_V2"
EXPECTED_PROVENANCE_TOKEN = "VANDA_STAGE45B_FULL24_ALGORITHMIC_PROVENANCE_CLOSED"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not child.is_dir():
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            if child.is_symlink() or not child.is_file():
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _float_rows(frame: Any, column: str, width: int) -> np.ndarray:
    if column not in frame.columns:
        raise RuntimeError(f"missing parquet column {column!r}")
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[column]]
    if len(rows) != EXPECTED_HORIZON or any(row.size != width for row in rows):
        raise RuntimeError(
            f"{column} must contain {EXPECTED_HORIZON} rows of width {width}"
        )
    result = np.stack(rows, axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError(f"{column} contains non-finite values")
    return result


def _timestamps(frame: Any) -> np.ndarray:
    if "timestamp" not in frame.columns:
        raise RuntimeError("parquet timestamp column is missing")
    result = np.asarray(frame["timestamp"].to_numpy(), dtype=np.float64)
    if result.shape != (EXPECTED_HORIZON,) or not np.isfinite(result).all():
        raise RuntimeError("parquet timestamp vector is not finite H40")
    return result


def _verify_dataset_metadata(dataset: Path, repo: Path) -> dict[str, Any]:
    split_path = dataset / "meta" / "stage22_split_lock.json"
    split = read_json(split_path)
    expected_split = {
        "schema": "STAGE22_MATERIALIZED_SPLIT_LOCK_V1",
        "source_lock": "manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json",
        "selected_split": "train",
        "dataset_role": "TRAIN",
        "selected_target_ids": list(EXPECTED_TARGETS),
        "train_target_ids": list(EXPECTED_TARGETS),
        "validation_target_ids": [],
        "final_test_target_ids": [],
        "training_allowed": True,
        "final_test_in_training": False,
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise RuntimeError(f"split lock mismatch for {key}: {split.get(key)!r}")
    split_source = repo / "manifests" / "STAGE22_TARGET_SPLIT_LOCK_V1.json"
    if sha256_file(split_source) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("repository split-lock SHA256 mismatch")

    episodes_path = dataset / "meta" / "episodes.jsonl"
    episode_rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episode_rows) != 24:
        raise RuntimeError(f"expected 24 episode records, got {len(episode_rows)}")
    episode_ids: list[str] = []
    for index, row in enumerate(episode_rows):
        target = EXPECTED_TARGETS[index // 6]
        source_id = f"stage22_expert_{target}:expert-{index % 6:04d}"
        if (
            row.get("episode_index") != index
            or row.get("length") != EXPECTED_HORIZON
            or row.get("split") != "train"
            or row.get("tasks") != [EXPECTED_LANGUAGE]
            or row.get("source_episode_id") != source_id
        ):
            raise RuntimeError(f"episode mapping mismatch at index {index}")
        episode_ids.append(source_id)

    info_path = dataset / "meta" / "info.json"
    info = read_json(info_path)
    if info.get("total_episodes") != 24 or info.get("total_frames") != 960:
        raise RuntimeError("FULL24 info.json is not 24 episodes / 960 frames")
    stats_path = dataset / "meta" / "stats.json"
    relative_stats_path = dataset / "meta" / "relative_stats.json"
    metadata = {
        "info_json": {"path": str(info_path), "sha256": sha256_file(info_path)},
        "episodes_jsonl": {
            "path": str(episodes_path),
            "sha256": sha256_file(episodes_path),
        },
        "stage22_split_lock_json": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
        },
        "stats_json": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
        "relative_stats_json": {
            "path": str(relative_stats_path),
            "sha256": sha256_file(relative_stats_path),
        },
    }
    if metadata["stats_json"]["sha256"] != EXPECTED_STAGE22_STATS_SHA256:
        raise RuntimeError("Stage22 stats.json identity mismatch")
    if metadata["relative_stats_json"]["sha256"] != EXPECTED_STAGE22_RELATIVE_STATS_SHA256:
        raise RuntimeError("Stage22 relative_stats.json identity mismatch")
    return {
        "split_lock": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "source_lock_sha256": EXPECTED_SPLIT_SHA256,
            "selected_target_ids": list(EXPECTED_TARGETS),
            "validation_target_ids": [],
            "final_test_target_ids": [],
        },
        "episodes": {
            "path": str(episodes_path),
            "sha256": sha256_file(episodes_path),
            "count": len(episode_rows),
            "ordered_source_example_ids": episode_ids,
        },
        "metadata_files": metadata,
    }


def _verify_b0(b0_path: Path, dataset_tree_sha256: str) -> dict[str, Any]:
    if b0_path.is_symlink() or not b0_path.is_file():
        raise RuntimeError(f"historical B0 stats is not a regular file: {b0_path}")
    observed_sha = sha256_file(b0_path)
    if observed_sha != EXPECTED_B0_SHA256:
        raise RuntimeError(f"historical B0 SHA mismatch: {observed_sha}")
    document = read_json(b0_path)
    if document.get("statistics_payload_sha256") != EXPECTED_B0_PAYLOAD_SHA256:
        raise RuntimeError("historical B0 payload SHA mismatch")
    if document.get("dataset", {}).get("tree_sha256") != dataset_tree_sha256:
        raise RuntimeError("historical B0 FULL24 binding mismatch")
    ordered = document.get("source_snapshots", {}).get("ordered_records")
    if not isinstance(ordered, list) or len(ordered) != 24:
        raise RuntimeError("historical B0 source mapping is not FULL24")
    return {
        "path": str(b0_path),
        "sha256": observed_sha,
        "statistics_payload_sha256": document["statistics_payload_sha256"],
        "source_mapping_count": len(ordered),
        "retained_as_historical_only": True,
        "execution_stats_authority": False,
    }


def _algorithmic_window(
    *, index: int, target: str, snapshot_index: int, timestamps: np.ndarray,
    physical_action: np.ndarray, states: np.ndarray, source_example_id: str,
) -> tuple[dict[str, Any], bool]:
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix

    u = np.linspace(0.0, 1.0, EXPECTED_HORIZON, dtype=np.float64)
    smooth = u * u * (3.0 - 2.0 * u)
    expected_position = physical_action[0, :3][None, :] + smooth[:, None] * (
        physical_action[-1, :3] - physical_action[0, :3]
    )[None, :]
    position_abs = np.abs(physical_action[:, :3] - expected_position)
    position_l2 = np.linalg.norm(physical_action[:, :3] - expected_position, axis=1)
    rotations = rot6d_rows_to_matrix(physical_action[:, 3:9])
    orientation_errors = np.asarray(
        [float(geodesic_angle(rotation, rotations[0])) for rotation in rotations],
        dtype=np.float64,
    )
    state_rotation = rot6d_rows_to_matrix(states[0, 3:9])
    anchor_rotation_errors = np.asarray(
        [float(geodesic_angle(state_rotation, rotations[0]))], dtype=np.float64
    )
    dt = np.diff(timestamps)
    metrics = {
        "window_index": index,
        "target_id": target,
        "snapshot_index": snapshot_index,
        "source_example_id": source_example_id,
        "row_count": int(physical_action.shape[0]),
        "position_formula_max_abs_error_m": float(np.max(position_abs)),
        "position_formula_max_l2_error_m": float(np.max(position_l2)),
        "orientation_constant_max_geodesic_error_rad": float(np.max(orientation_errors)),
        "orientation_constant_max_raw_rot6d_abs_change": float(
            np.max(np.abs(physical_action[:, 3:9] - physical_action[0, 3:9]))
        ),
        "action_gripper_max_abs_error_from_0p5": float(
            np.max(np.abs(physical_action[:, 9] - 0.5))
        ),
        "state_gripper_max_abs_error_from_0p5": float(
            np.max(np.abs(states[:, 9] - 0.5))
        ),
        "state_eef_action_pose0_position_error_m": float(
            np.linalg.norm(states[0, :3] - physical_action[0, :3])
        ),
        "state_eef_action_pose0_rotation_error_rad": float(np.max(anchor_rotation_errors)),
        "state_repeat_max_abs_error": float(np.max(np.abs(states - states[0]))),
        "timestamps_dt_max_abs_error_s": float(np.max(np.abs(dt - 0.05))),
        "duration_error_s": float(abs((timestamps[-1] - timestamps[0]) - 1.95)),
        "state_repeated_exact": bool(np.array_equal(states, np.broadcast_to(states[0], states.shape))),
        "cohort_order_entry": f"{target} x snapshot-{snapshot_index}",
    }
    passed = bool(
        metrics["row_count"] == EXPECTED_HORIZON
        and metrics["position_formula_max_abs_error_m"] <= 1e-6
        and metrics["position_formula_max_l2_error_m"] <= 1e-6
        and metrics["orientation_constant_max_geodesic_error_rad"] <= 1e-6
        and metrics["action_gripper_max_abs_error_from_0p5"] <= 1e-6
        and metrics["state_gripper_max_abs_error_from_0p5"] <= 1e-6
        and metrics["state_eef_action_pose0_position_error_m"] <= 1e-6
        and metrics["state_eef_action_pose0_rotation_error_rad"] <= 1e-6
        and metrics["state_repeated_exact"]
        and metrics["timestamps_dt_max_abs_error_s"] <= 1e-9
        and metrics["duration_error_s"] <= 1e-9
    )
    metrics["status"] = "PASS" if passed else "BLOCKED"
    return metrics, passed


def _source_bridge(
    *, snapshot_root: Path, dataset_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]], b0: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the surviving train_a source snapshots bridge to parquet float32."""

    records = b0.get("source_snapshots", {}).get("ordered_records", [])
    if len(records) != 24:
        raise RuntimeError("B0 source mapping is not available for bridge proof")
    bridge_records: list[dict[str, Any]] = []
    for snapshot_index in range(6):
        target = EXPECTED_TARGETS[0]
        path = snapshot_root / f"stage22_expert_{target}" / f"expert-{snapshot_index:04d}.json"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"surviving train_a source snapshot is unavailable: {path}")
        expected_record = records[snapshot_index]
        expected_sha = expected_record.get("json_sha256") or expected_record.get(
            "snapshot_json_sha256"
        )
        if not isinstance(expected_sha, str):
            raise RuntimeError(f"B0 source SHA is missing at train_a index {snapshot_index}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise RuntimeError(f"train_a snapshot SHA mismatch at index {snapshot_index}")
        payload = read_json(path)
        expected_id = f"stage22_expert_{target}:expert-{snapshot_index:04d}"
        if payload.get("example_id") != expected_id or payload.get("target_id") != target:
            raise RuntimeError(f"train_a snapshot identity mismatch at index {snapshot_index}")
        desired = np.asarray(payload.get("future_desired_eef_9d"), dtype=np.float32).astype(np.float64)
        gripper = np.asarray(payload.get("future_gripper"), dtype=np.float32).reshape(-1, 1).astype(np.float64)
        state = np.asarray(payload.get("state_v1", {}).get("values"), dtype=np.float32).astype(np.float64)
        timestamps = np.asarray(payload.get("future_timestamps"), dtype=np.float32).astype(np.float64)
        parquet_action, parquet_state, parquet_timestamps = dataset_frames[snapshot_index]
        if desired.shape != (40, 9) or gripper.shape != (40, 1) or state.shape != (10,):
            raise RuntimeError(f"train_a source shape mismatch at index {snapshot_index}")
        desired_error = float(np.max(np.abs(desired - parquet_action[:, :9])))
        gripper_error = float(np.max(np.abs(gripper - parquet_action[:, 9:10])))
        state_error = float(np.max(np.abs(state - parquet_state[0])))
        timestamp_error = float(np.max(np.abs(timestamps - parquet_timestamps)))
        record = {
            "snapshot_index": snapshot_index,
            "source_example_id": expected_id,
            "path": str(path),
            "json_sha256": observed_sha,
            "float32_bridge_max_abs_error": {
                "desired_eef_9d": desired_error,
                "gripper": gripper_error,
                "state_v1": state_error,
                "timestamps": timestamp_error,
            },
            "status": "PASS"
            if max(desired_error, gripper_error, state_error, timestamp_error) == 0.0
            else "BLOCKED",
        }
        if record["status"] != "PASS":
            raise RuntimeError(f"train_a source-to-parquet float32 bridge failed at index {snapshot_index}")
        bridge_records.append(record)
    return {
        "status": "PASS",
        "surviving_source_snapshot_count": 6,
        "historical_source_snapshot_count": 24,
        "unavailable_historical_replica_count": 18,
        "missing_replicas_are_not_execution_authority": True,
        "records": bridge_records,
    }


def qualify(
    *, repo: Path, dataset: Path, snapshot_root: Path, b0_path: Path,
    expected_head: str, stats_output: Path, result_output: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    dataset = dataset.resolve()
    snapshot_root = snapshot_root.resolve()
    stats_output = stats_output.resolve()
    result_output = result_output.resolve()
    if stats_output.exists() or result_output.exists():
        raise RuntimeError("V2 evidence outputs must be new; refusing overwrite")
    if run_git(repo, "branch", "--show-current") != BRANCH:
        raise RuntimeError("execution branch mismatch")
    if run_git(repo, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("execution HEAD mismatch")
    if run_git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("execution checkout is dirty")
    if len(expected_head) != 40:
        raise RuntimeError("expected_head must be a full commit SHA")

    entries = tree_entries(dataset, "payload/lerobot_train")
    dataset_tree_sha256 = tree_digest(entries)
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    if (
        dataset_tree_sha256 != EXPECTED_TREE_SHA256
        or len(entries) != EXPECTED_FILE_COUNT
        or total_bytes != EXPECTED_TOTAL_BYTES
    ):
        raise RuntimeError(
            f"FULL24 identity mismatch: {dataset_tree_sha256}, {len(entries)}, {total_bytes}"
        )
    metadata = _verify_dataset_metadata(dataset, repo)
    b0_identity = _verify_b0(b0_path.resolve(), dataset_tree_sha256)

    data_paths = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if len(data_paths) != 24 or [int(path.stem.split("_")[-1]) for path in data_paths] != list(range(24)):
        raise RuntimeError("FULL24 parquet file order is not exactly episode 0..23")
    import pandas as pd

    dataset_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    window_metrics: list[dict[str, Any]] = []
    labels: list[Any] = []
    reconciliation_records: list[dict[str, Any]] = []
    from go2arx5_vla.structured_action import (
        encode_structured_action,
        reconcile_stage45b_label_anchor,
    )

    for index, path in enumerate(data_paths):
        frame = pd.read_parquet(path, columns=["timestamp", "action", "observation.state"])
        timestamps = _timestamps(frame)
        action = _float_rows(frame, "action", EXPECTED_PHYSICAL_DIM)
        states = _float_rows(frame, "observation.state", EXPECTED_STATE_DIM)
        dataset_frames.append((action, states, timestamps))
        target = EXPECTED_TARGETS[index // 6]
        snapshot_index = index % 6
        source_example_id = f"stage22_expert_{target}:expert-{snapshot_index:04d}"
        metrics, passed = _algorithmic_window(
            index=index,
            target=target,
            snapshot_index=snapshot_index,
            timestamps=timestamps,
            physical_action=action,
            states=states,
            source_example_id=source_example_id,
        )
        if not passed:
            raise RuntimeError(f"algorithmic Stage22 provenance failed at window {index}")
        window_metrics.append(metrics)
        reconciled_state, reconciliation = reconcile_stage45b_label_anchor(action, states[0])
        label = encode_structured_action(
            action,
            timestamps,
            reconciled_state,
            timing_mode=EXPECTED_TIMING_MODE,
            chunk_id=source_example_id,
            source_frame=EXPECTED_SOURCE_FRAME,
            tcp_semantic_id=EXPECTED_TCP,
            observation_timestamp=float(timestamps[0]),
            provenance={
                "stage": STAGE,
                "pi_gate": PI_GATE,
                "protocol_revision": PROTOCOL_REVISION,
                "canonical_label_source": "SEALED_FULL24_TRAIN_PARQUET",
                "source_example_id": source_example_id,
                "target_id": target,
                "target_excluded_from_model_input": True,
                "stage45b_label_anchor_reconciliation": reconciliation,
            },
        )
        labels.append(label)
        reconciliation_records.append(
            {
                "window_index": index,
                "source_example_id": source_example_id,
                "status": "PASS",
                **reconciliation,
            }
        )
    if len(labels) != 24:
        raise RuntimeError(f"runtime dataset produced {len(labels)} labels instead of 24")

    bridge = _source_bridge(
        snapshot_root=snapshot_root,
        dataset_frames=dataset_frames,
        b0=read_json(b0_path.resolve()),
    )
    cohort = [f"{target} x snapshot-{snapshot}" for target in EXPECTED_TARGETS for snapshot in range(6)]
    if [record["cohort_order_entry"] for record in window_metrics] != cohort:
        raise RuntimeError("FULL24 cohort order is not target-major then snapshot-major")

    from go2arx5_vla.stage38_structured_training import _structured_stats

    statistics = _structured_stats(labels, use_percentiles=True)
    if (
        statistics.get("schema") != "STAGE38_STRUCTURED_ACTION_STATISTICS_V1"
        or statistics.get("label_count") != 24
        or statistics.get("timing_mode") != EXPECTED_TIMING_MODE
        or statistics.get("universal_action_dim") != EXPECTED_ACTION_DIM
        or statistics.get("use_percentiles") is not True
        or statistics.get("selected_lower_bound") != "q01"
        or statistics.get("selected_upper_bound") != "q99"
    ):
        raise RuntimeError("runtime-parquet statistics contract mismatch")
    payload_sha256 = canonical_sha256(statistics)
    evidence = {
        "schema": "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2",
        "status": EXPECTED_RETURN_TOKEN,
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "protocol_revision": PROTOCOL_REVISION,
        "repository": {
            "branch": BRANCH,
            "head": expected_head,
            "working_tree_required_clean": True,
        },
        "canonical_label_source": {
            "kind": "SEALED_FULL24_TRAIN_PARQUET",
            "root": str(dataset),
            "tree_sha256": dataset_tree_sha256,
            "parquet_files_read": len(data_paths),
            "parquet_tree_untouched": True,
        },
        "historical_b0": b0_identity,
        "algorithmic_provenance": {
            "status": EXPECTED_PROVENANCE_TOKEN,
            "law": "p_expected[k] = p0 + (k/39)^2 * (3 - 2*k/39) * (p39 - p0)",
            "u_grid": "k/39 for k=0..39",
            "orientation": "constant native WBC row-packed rot6d",
            "gripper": 0.5,
            "timestamp_policy": "40 points, dt=0.05 s, duration=1.95 s",
            "cohort_order": cohort,
            "window_count": len(window_metrics),
            "window_metrics": window_metrics,
            "source_to_parquet_bridge": bridge,
        },
        "anchor_reconciliation": {
            "codec_anchor_invariant_unchanged": True,
            "normal_position_tolerance_m": 1e-9,
            "normal_rotation_tolerance_rad": 1e-9,
            "provenance_admission_tolerance_m": 1e-6,
            "provenance_admission_tolerance_rad": 1e-6,
            "physical_action_untouched": True,
            "observation_state_untouched": True,
            "label_only_state_copy": True,
            "records": reconciliation_records,
        },
        "label_contract": {
            "codec": "go2arx5_vla.structured_action.encode_structured_action",
            "reconciliation_helper": "go2arx5_vla.structured_action.reconcile_stage45b_label_anchor",
            "statistics_implementation": "go2arx5_vla.stage38_structured_training._structured_stats",
            "timing_mode": EXPECTED_TIMING_MODE,
            "state_v1_shape": [EXPECTED_STATE_DIM],
            "physical_action_shape": [EXPECTED_HORIZON, EXPECTED_PHYSICAL_DIM],
            "structured_action_shape": [EXPECTED_HORIZON, EXPECTED_ACTION_DIM],
            "source_frame": EXPECTED_SOURCE_FRAME,
            "tcp_semantic_id": EXPECTED_TCP,
            "inactive_dimensions_excluded": True,
            "padding_excluded": True,
        },
        "statistics_payload_sha256": payload_sha256,
        "statistics": statistics,
        "scientific_semantics": {
            "scientific_question_changed": False,
            "normalization_policy_changed": False,
            "execution_artifact_identity_changed": True,
            "normalization": "MINMAX_TO_NEG1_POS1 with q01/q99",
            "revision": PROTOCOL_REVISION,
        },
        "execution_boundary": {
            "gpu_or_simulator_execution": False,
            "model_constructed": False,
            "gr00t_forwards": 0,
            "wbc_steps": 0,
            "reference_updates": 0,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        "return_token": EXPECTED_RETURN_TOKEN,
        "next_gate": "STAGE45B_H200_REPAIRED_REFREEZE_ADMISSION",
    }
    write_json(stats_output, evidence)
    result = {
        "schema": "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_QUALIFICATION_RESULT_V2",
        "status": EXPECTED_RETURN_TOKEN,
        "pi_gate": PI_GATE,
        "protocol_revision": PROTOCOL_REVISION,
        "repository": evidence["repository"],
        "stats_artifact": {
            "path": str(stats_output),
            "bytes": int(stats_output.stat().st_size),
            "sha256": sha256_file(stats_output),
        },
        "statistics_payload_sha256": payload_sha256,
        "dataset_tree": {
            "sha256": dataset_tree_sha256,
            "regular_file_count": len(entries),
            "total_bytes": total_bytes,
        },
        "algorithmic_provenance_token": EXPECTED_PROVENANCE_TOKEN,
        "runtime_dataset_label_count": len(labels),
        "cohort_order": cohort,
        "source_bridge": bridge,
        "anchor_reconciliation_record_count": len(reconciliation_records),
        "scientific_semantics": evidence["scientific_semantics"],
        "execution_boundary": evidence["execution_boundary"],
        "return_token": EXPECTED_RETURN_TOKEN,
    }
    write_json(result_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--b0-stats", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = qualify(
            repo=args.repo,
            dataset=args.dataset,
            snapshot_root=args.snapshot_root,
            b0_path=args.b0_stats,
            expected_head=args.expected_head,
            stats_output=args.stats_output,
            result_output=args.result_output,
        )
    except Exception as exc:
        print(f"VANDA_STAGE45B_RUNTIME_PARQUET_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
