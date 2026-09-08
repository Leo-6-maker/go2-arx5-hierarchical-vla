#!/usr/bin/env python3
"""Close the Stage45B FULL24 StructuredAction statistics gate.

This is a CPU-only, fail-closed preflight.  It validates the already
materialized Stage22 TRAIN fixture and its 24 exact expert snapshots, builds
the same ``StructuredActionV1`` labels used by the Stage38 adapter, and calls
the frozen Stage38 ``_structured_stats`` implementation.  It never imports a
model runtime, opens a simulator, reads Validation-E/Holdout-F, or writes to
the source fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_STAGE = "Stage45B"
EXPECTED_GATE = "PI_GATE_STATE_0073"
EXPECTED_TREE_SHA256 = (
    "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
)
EXPECTED_FILE_COUNT = 85
EXPECTED_TOTAL_BYTES = 919661
EXPECTED_SPLIT_SHA256 = (
    "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
)
EXPECTED_TRAIN_TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
EXPECTED_TARGET_POSITIONS = {
    "red_target_train_a": [0.30, 0.10, 0.58],
    "red_target_train_b": [0.30, -0.10, 0.58],
    "red_target_train_c": [0.30, 0.05, 0.64],
    "red_target_train_d": [0.30, -0.05, 0.63],
}
EXPECTED_LANGUAGE = "move the end effector to the red target"
EXPECTED_SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
EXPECTED_TCP = "x5_link6_gripper_center"
EXPECTED_HORIZON = 40
EXPECTED_STATE_DIM = 10
EXPECTED_PHYSICAL_DIM = 10
EXPECTED_ACTION_DIM = 132
EXPECTED_TIMING_MODE = "T16"
EXPECTED_STATS_SCHEMA = "STAGE38_STRUCTURED_ACTION_STATISTICS_V1"
EXPECTED_RETURN_TOKEN = "CORELAB_STAGE45B_FULL24_STRUCTURED_STATS_CLOSED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read JSON: {path}: {exc}") from exc


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"dataset root is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink found in immutable fixture: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": f"{prefix}/{relative}",
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSONL is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _verify_split(dataset_root: Path, repo_root: Path) -> dict[str, Any]:
    split_path = dataset_root / "meta" / "stage22_split_lock.json"
    split = _read_json(split_path)
    expected = {
        "schema": "STAGE22_MATERIALIZED_SPLIT_LOCK_V1",
        "source_lock": "manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json",
        "selected_split": "train",
        "dataset_role": "TRAIN",
        "selected_target_ids": list(EXPECTED_TRAIN_TARGETS),
        "train_target_ids": list(EXPECTED_TRAIN_TARGETS),
        "validation_target_ids": [],
        "final_test_target_ids": [],
        "training_allowed": True,
        "final_test_in_training": False,
    }
    for key, value in expected.items():
        if split.get(key) != value:
            raise RuntimeError(f"materialized split lock mismatch for {key}: {split.get(key)!r}")
    source_lock = repo_root / "manifests" / "STAGE22_TARGET_SPLIT_LOCK_V1.json"
    if _sha256_file(source_lock) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("repository Stage22 target split lock SHA256 mismatch")
    return {
        "path": str(split_path),
        "sha256": _sha256_file(split_path),
        "source_lock_path": str(source_lock),
        "source_lock_sha256": _sha256_file(source_lock),
        "selected_target_ids": list(EXPECTED_TRAIN_TARGETS),
        "validation_target_ids": [],
        "final_test_target_ids": [],
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def _verify_episodes(dataset_root: Path) -> dict[str, Any]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    rows = _load_jsonl(episodes_path)
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 materialized episodes, got {len(rows)}")
    ordered_ids: list[str] = []
    for index, row in enumerate(rows):
        target = EXPECTED_TRAIN_TARGETS[index // 6]
        snapshot = index % 6
        expected_id = f"stage22_expert_{target}:expert-{snapshot:04d}"
        if row.get("episode_index") != index or row.get("length") != EXPECTED_HORIZON:
            raise RuntimeError(f"episode metadata mismatch at index {index}")
        if row.get("split") != "train" or row.get("tasks") != [EXPECTED_LANGUAGE]:
            raise RuntimeError(f"episode split/task mismatch at index {index}")
        if row.get("source_episode_id") != expected_id:
            raise RuntimeError(
                f"episode source mapping mismatch at index {index}: {row.get('source_episode_id')!r}"
            )
        ordered_ids.append(expected_id)
    return {
        "path": str(episodes_path),
        "sha256": _sha256_file(episodes_path),
        "count": len(rows),
        "horizon": EXPECTED_HORIZON,
        "ordered_source_example_ids": ordered_ids,
    }


def _snapshot_path(snapshot_root: Path, target: str, snapshot_index: int) -> Path:
    return snapshot_root / f"stage22_expert_{target}" / f"expert-{snapshot_index:04d}.json"


def _load_label(snapshot_path: Path, *, target: str, snapshot_index: int) -> Any:
    payload = _read_json(snapshot_path)
    expected_id = f"stage22_expert_{target}:expert-{snapshot_index:04d}"
    if payload.get("schema") != "SameSceneExpertSnapshotV1":
        raise RuntimeError(f"snapshot schema mismatch: {snapshot_path}")
    if payload.get("example_id") != expected_id or payload.get("target_id") != target:
        raise RuntimeError(f"snapshot identity mismatch: {snapshot_path}")
    if payload.get("raw_episode_id") != f"stage22_expert_{target}":
        raise RuntimeError(f"snapshot raw episode mismatch: {snapshot_path}")
    if payload.get("target_excluded_from_model_input") is not True:
        raise RuntimeError(f"target exclusion flag is not true: {snapshot_path}")
    if payload.get("source_frame") != EXPECTED_SOURCE_FRAME:
        raise RuntimeError(f"source frame mismatch: {snapshot_path}")
    if payload.get("tcp_semantic_id") != EXPECTED_TCP:
        raise RuntimeError(f"TCP semantic mismatch: {snapshot_path}")
    if payload.get("target_position_world_m") != EXPECTED_TARGET_POSITIONS[target]:
        raise RuntimeError(f"target position provenance mismatch: {snapshot_path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("language") != EXPECTED_LANGUAGE:
        raise RuntimeError(f"snapshot language provenance mismatch: {snapshot_path}")

    state_payload = payload.get("state_v1")
    if not isinstance(state_payload, dict) or state_payload.get("schema") != "StateV1":
        raise RuntimeError(f"StateV1 payload missing: {snapshot_path}")
    state = np.asarray(state_payload.get("values"), dtype=np.float64)
    timestamps = np.asarray(payload.get("future_timestamps"), dtype=np.float64)
    desired = np.asarray(payload.get("future_desired_eef_9d"), dtype=np.float64)
    gripper = np.asarray(payload.get("future_gripper"), dtype=np.float64)
    if state.shape != (EXPECTED_STATE_DIM,):
        raise RuntimeError(f"StateV1 shape mismatch at {snapshot_path}: {state.shape}")
    if timestamps.shape != (EXPECTED_HORIZON,):
        raise RuntimeError(f"timestamp shape mismatch at {snapshot_path}: {timestamps.shape}")
    if desired.shape != (EXPECTED_HORIZON, 9):
        raise RuntimeError(f"EEF label shape mismatch at {snapshot_path}: {desired.shape}")
    if gripper.shape != (EXPECTED_HORIZON, 1):
        raise RuntimeError(f"gripper label shape mismatch at {snapshot_path}: {gripper.shape}")
    if not np.isfinite(state).all() or not np.isfinite(timestamps).all():
        raise RuntimeError(f"non-finite state/timestamps at {snapshot_path}")
    if not np.isfinite(desired).all() or not np.isfinite(gripper).all():
        raise RuntimeError(f"non-finite expert labels at {snapshot_path}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError(f"timestamps are not strictly increasing at {snapshot_path}")
    if not np.allclose(state[:9], desired[0], rtol=0.0, atol=1e-10):
        raise RuntimeError(f"snapshot anchor state does not match first desired pose: {snapshot_path}")

    body_path = Path(str(payload.get("body_frame_path", "")))
    wrist_path = Path(str(payload.get("wrist_frame_path", "")))
    for frame_path in (body_path, wrist_path):
        if not frame_path.is_file() or frame_path.is_symlink():
            raise RuntimeError(f"snapshot camera frame is not a regular file: {frame_path}")

    # Import only the CPU codec and the frozen Stage38 statistics implementation.
    from go2arx5_vla.structured_action import encode_structured_action

    physical = np.concatenate([desired, gripper], axis=-1)
    return encode_structured_action(
        physical,
        timestamps,
        state,
        timing_mode=EXPECTED_TIMING_MODE,
        chunk_id=expected_id,
        source_frame=EXPECTED_SOURCE_FRAME,
        tcp_semantic_id=EXPECTED_TCP,
        observation_timestamp=float(payload["observation_timestamp"]),
        provenance={
            "stage": "Stage45B-B0",
            "source_example_id": expected_id,
            "target_id": target,
            "target_excluded_from_model_input": True,
        },
    )


def close_stats(
    *,
    repo_root: Path,
    dataset_root: Path,
    snapshot_root: Path,
    stats_output: Path,
    result_output: Path,
    expected_git_head: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    snapshot_root = snapshot_root.resolve()
    stats_output = stats_output.resolve()
    result_output = result_output.resolve()
    if not repo_root.is_dir() or not dataset_root.is_dir() or not snapshot_root.is_dir():
        raise RuntimeError("repository, dataset, and snapshot roots must be directories")
    if stats_output.exists() or result_output.exists():
        raise RuntimeError("B0 output paths must be new; refusing overwrite")
    if not expected_git_head or len(expected_git_head) != 40:
        raise RuntimeError("expected_git_head must be a full commit SHA")

    entries = _tree_entries(dataset_root, "payload/lerobot_train")
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    tree_sha256 = _tree_digest(entries)
    if len(entries) != EXPECTED_FILE_COUNT or total_bytes != EXPECTED_TOTAL_BYTES:
        raise RuntimeError(
            f"FULL24 tree size mismatch: files={len(entries)} bytes={total_bytes}"
        )
    if tree_sha256 != EXPECTED_TREE_SHA256:
        raise RuntimeError(
            f"FULL24 tree SHA256 mismatch: expected {EXPECTED_TREE_SHA256}, got {tree_sha256}"
        )

    split = _verify_split(dataset_root, repo_root)
    episodes = _verify_episodes(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    relative_stats_path = dataset_root / "meta" / "relative_stats.json"
    info = _read_json(info_path)
    if info.get("total_episodes") != 24 or info.get("total_frames") != 960:
        raise RuntimeError("FULL24 info.json episode/frame counts are not 24/960")
    if info.get("fps") is None or not np.isfinite(float(info["fps"])):
        raise RuntimeError("FULL24 info.json fps is not finite")
    metadata_files = {
        "info_json": {"path": str(info_path), "sha256": _sha256_file(info_path)},
        "stats_json": {"path": str(stats_path), "sha256": _sha256_file(stats_path)},
        "relative_stats_json": {
            "path": str(relative_stats_path),
            "sha256": _sha256_file(relative_stats_path),
        },
    }

    labels: list[Any] = []
    snapshot_records: list[dict[str, Any]] = []
    for target in EXPECTED_TRAIN_TARGETS:
        for snapshot_index in range(6):
            path = _snapshot_path(snapshot_root, target, snapshot_index)
            label = _load_label(path, target=target, snapshot_index=snapshot_index)
            labels.append(label)
            snapshot_records.append(
                {
                    "window_index": len(labels) - 1,
                    "target_id": target,
                    "snapshot_index": snapshot_index,
                    "example_id": f"stage22_expert_{target}:expert-{snapshot_index:04d}",
                    "json_path": str(path),
                    "json_bytes": int(path.stat().st_size),
                    "json_sha256": _sha256_file(path),
                }
            )
    if len(labels) != 24:
        raise RuntimeError(f"expected 24 labels, got {len(labels)}")

    from go2arx5_vla.stage38_structured_training import _structured_stats

    statistics = _structured_stats(labels, use_percentiles=True)
    if statistics.get("schema") != EXPECTED_STATS_SCHEMA:
        raise RuntimeError("Stage38 statistics schema changed")
    if statistics.get("label_count") != 24:
        raise RuntimeError("Stage38 statistics label count is not 24")
    if statistics.get("timing_mode") != EXPECTED_TIMING_MODE:
        raise RuntimeError("Stage38 statistics timing mode changed")
    if statistics.get("universal_action_dim") != EXPECTED_ACTION_DIM:
        raise RuntimeError("Stage38 statistics action dimension changed")
    if statistics.get("selected_lower_bound") != "q01" or statistics.get("selected_upper_bound") != "q99":
        raise RuntimeError("Stage45B requires q01/q99 statistics")
    if statistics.get("use_percentiles") is not True:
        raise RuntimeError("Stage45B requires use_percentiles=true")
    if len(statistics.get("active_observation_count_by_dim", [])) != EXPECTED_ACTION_DIM:
        raise RuntimeError("Stage38 statistics active-count vector is not 132-dimensional")
    if not all(int(value) > 0 for value in statistics["active_observation_count_by_dim"]):
        raise RuntimeError("at least one structured action dimension has no observations")

    stats_payload_sha256 = _canonical_sha256(statistics)
    stats_document = {
        "schema": "CORELAB_STAGE45B_FULL24_STRUCTURED_STATS_V1",
        "status": EXPECTED_RETURN_TOKEN,
        "stage": EXPECTED_STAGE,
        "pi_gate": EXPECTED_GATE,
        "repository": {
            "head": expected_git_head,
            "working_tree_required_clean": True,
        },
        "dataset": {
            "role": "TRAIN",
            "root": str(dataset_root),
            "canonical_tree_method": "STAGE22_PREPARE_H200_HANDOFF_TREE_DIGEST_V1",
            "tree_sha256": tree_sha256,
            "regular_file_count": len(entries),
            "total_bytes": total_bytes,
            "episode_count": 24,
            "frame_count": 960,
            "window_count": 24,
            "new_data_generation": False,
            "simulator_read": False,
            "metadata_files": metadata_files,
        },
        "split_lock": split,
        "episodes": episodes,
        "source_snapshots": {
            "root": str(snapshot_root),
            "count": len(snapshot_records),
            "ordered_records": snapshot_records,
        },
        "label_contract": {
            "source": "Stage22 SameSceneExpertSnapshotV1 labels",
            "codec": "go2arx5_vla.structured_action.encode_structured_action",
            "statistics_implementation": "go2arx5_vla.stage38_structured_training._structured_stats",
            "timing_mode": EXPECTED_TIMING_MODE,
            "state_v1_shape": [EXPECTED_STATE_DIM],
            "physical_action_shape": [EXPECTED_HORIZON, EXPECTED_PHYSICAL_DIM],
            "structured_action_shape": [EXPECTED_HORIZON, EXPECTED_ACTION_DIM],
            "source_frame": EXPECTED_SOURCE_FRAME,
            "tcp_semantic_id": EXPECTED_TCP,
            "target_xyz_in_model_input": False,
            "inactive_dimensions_excluded": True,
            "padding_excluded": True,
        },
        "statistics_payload_sha256": stats_payload_sha256,
        "statistics": statistics,
        "execution_boundary": {
            "gr00t_forwards": 0,
            "wbc_steps": 0,
            "reference_updates": 0,
            "new_data_generation": 0,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "gpu_or_simulator_execution": False,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        "return_token": EXPECTED_RETURN_TOKEN,
        "next_gate": "CORELAB_STAGE45B_H200_ADMISSION_READY",
    }
    _write_json(stats_output, stats_document)
    stats_artifact = {
        "path": str(stats_output),
        "bytes": int(stats_output.stat().st_size),
        "sha256": _sha256_file(stats_output),
    }
    result_document = {
        "schema": "CORELAB_STAGE45B_FULL24_STRUCTURED_STATS_RESULT_V1",
        "status": EXPECTED_RETURN_TOKEN,
        "stage": EXPECTED_STAGE,
        "pi_gate": EXPECTED_GATE,
        "repository_head": expected_git_head,
        "stats_artifact": stats_artifact,
        "stats_payload_sha256": stats_payload_sha256,
        "dataset_tree": {
            "sha256": tree_sha256,
            "regular_file_count": len(entries),
            "total_bytes": total_bytes,
        },
        "label_count": len(labels),
        "ordered_example_ids": [record["example_id"] for record in snapshot_records],
        "source_snapshot_json_sha256": [record["json_sha256"] for record in snapshot_records],
        "normalization": {
            "mode": "MINMAX_TO_NEG1_POS1",
            "use_percentiles": True,
            "lower": "q01",
            "upper": "q99",
            "inactive_dimensions_excluded": True,
            "padding_excluded": True,
        },
        "execution_boundary": stats_document["execution_boundary"],
        "return_token": EXPECTED_RETURN_TOKEN,
    }
    _write_json(result_output, result_document)
    return result_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--expected-git-head", required=True)
    args = parser.parse_args()
    try:
        result = close_stats(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            snapshot_root=args.snapshot_root,
            stats_output=args.stats_output,
            result_output=args.result_output,
            expected_git_head=args.expected_git_head,
        )
    except Exception as exc:  # pragma: no cover - command-line fail-closed surface
        print(f"BLOCKED_PROVENANCE_OR_STATS: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
