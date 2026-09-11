#!/usr/bin/env python3
"""Build and verify the Stage45E N2 XYZ-normalisation artifact.

This module is deliberately CPU-only.  It consumes the accepted Stage45C
runtime-V2 statistics document and applies exactly one intervention: the
selected lower/upper bounds for the first three StructuredActionV1 scalars
(the XYZ channels) are replaced by the accepted FULL24 robust centre plus or
minus the common half-range.  Every other statistic, label, frame, mask and
codec rule is copied byte-for-byte at the value level.

No model, GR00T, torch, simulator, WBC or GPU import is allowed here.  Paths
are supplied by the caller; no historical scratch path is embedded in the
N2 artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SOURCE_STATS_SHA256 = (
    "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
)
SOURCE_PAYLOAD_SHA256 = (
    "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
)
SOURCE_SCHEMA = "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2"
SOURCE_PROTOCOL = "RUNTIME_PARQUET_PROVENANCE_V2"
N2_SCHEMA = "CORELAB_STAGE45E_N2_STRUCTURED_STATS_V1"
N2_STATUS = "CPU_STATIC_N2_ARTIFACT_READY"
HALF_RANGE_M = 0.107349471
XYZ_DIMS = (0, 1, 2)
ACTION_HORIZON = 40
ACTION_DIM = 132
GEOMETRY_FUTURE_COUNT = 15
ACTIVE_SCALAR_COUNT = 190
LABEL_COUNT = 24
FRAME_COUNT = 960
EXPECTED_DATASET_TREE_SHA256 = (
    "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
)
EXPECTED_DATASET_FILE_COUNT = 85
EXPECTED_DATASET_TOTAL_BYTES = 919661
TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
TARGET_POSITIONS = {
    "red_target_train_a": [0.30, 0.10, 0.58],
    "red_target_train_b": [0.30, -0.10, 0.58],
    "red_target_train_c": [0.30, 0.05, 0.64],
    "red_target_train_d": [0.30, -0.05, 0.63],
}
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP = "x5_link6_gripper_center"


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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing N2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_finite_vector(value: Any, *, name: str, length: int = ACTION_DIM) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise RuntimeError(f"{name} must have shape [{length}], got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return array


def _validate_mask(value: Any) -> list[list[float]]:
    from go2arx5_vla.structured_action import structured_action_mask

    observed = np.asarray(value, dtype=np.float64)
    expected = structured_action_mask("T16")
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise RuntimeError("N2 requires the unchanged StructuredActionV1 T16 mask")
    if int(np.count_nonzero(observed)) != ACTIVE_SCALAR_COUNT:
        raise RuntimeError("StructuredActionV1 T16 active scalar count is not 190")
    return expected.tolist()


def validate_source_document(source: Mapping[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
    """Validate the accepted runtime-V2 identity before reading tensors."""

    if source.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError(f"runtime stats schema mismatch: {source.get('schema')!r}")
    if source.get("protocol_revision") != SOURCE_PROTOCOL:
        raise RuntimeError("runtime stats protocol revision mismatch")
    if source.get("statistics_payload_sha256") != SOURCE_PAYLOAD_SHA256:
        raise RuntimeError("runtime stats payload SHA256 mismatch")
    stats = source.get("statistics")
    if not isinstance(stats, Mapping):
        raise RuntimeError("runtime stats statistics object is missing")
    if stats.get("schema") != "STAGE38_STRUCTURED_ACTION_STATISTICS_V1":
        raise RuntimeError("Stage38 statistics schema mismatch")
    if stats.get("normalization") != "MINMAX_TO_NEG1_POS1":
        raise RuntimeError("normalization law mismatch")
    if stats.get("use_percentiles") is not True:
        raise RuntimeError("N2 requires use_percentiles=true")
    if stats.get("selected_lower_bound") != "q01" or stats.get("selected_upper_bound") != "q99":
        raise RuntimeError("N2 requires q01/q99 selected bounds")
    if stats.get("timing_mode") != "T16" or stats.get("action_horizon") != ACTION_HORIZON:
        raise RuntimeError("timing/horizon identity mismatch")
    if stats.get("universal_action_dim") != ACTION_DIM or stats.get("active_scalar_count") != ACTIVE_SCALAR_COUNT:
        raise RuntimeError("StructuredActionV1 dimension identity mismatch")
    if stats.get("label_count") != LABEL_COUNT:
        raise RuntimeError("FULL24 label count mismatch")
    mask = _validate_mask(stats.get("active_mask"))
    selected_lower = _require_finite_vector(stats.get("selected_lower"), name="selected_lower")
    selected_upper = _require_finite_vector(stats.get("selected_upper"), name="selected_upper")
    if np.any(selected_upper < selected_lower):
        raise RuntimeError("runtime statistics have inverted selected bounds")
    raw_stats = stats.get("statistics")
    if not isinstance(raw_stats, Mapping):
        raise RuntimeError("per-dimension runtime statistics are missing")
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        _require_finite_vector(raw_stats.get(name), name=f"statistics.{name}")
    outer_hash = sha256_file(source_path) if source_path is not None else None
    if outer_hash is not None and outer_hash != SOURCE_STATS_SHA256:
        raise RuntimeError(
            f"runtime stats outer SHA256 mismatch: expected {SOURCE_STATS_SHA256}, got {outer_hash}"
        )
    canonical_payload = _canonical_sha256(stats)
    if canonical_payload != SOURCE_PAYLOAD_SHA256:
        raise RuntimeError(
            "runtime stats canonical payload differs from accepted payload SHA256: "
            f"expected {SOURCE_PAYLOAD_SHA256}, got {canonical_payload}"
        )
    return {
        "outer_sha256": outer_hash,
        "payload_sha256": canonical_payload,
        "schema": SOURCE_SCHEMA,
        "protocol_revision": SOURCE_PROTOCOL,
        "active_mask": mask,
        "selected_lower": selected_lower,
        "selected_upper": selected_upper,
    }


def _regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode if root.exists() else 0
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise RuntimeError(f"dataset root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def validate_dataset_identity(dataset_root: Path) -> dict[str, Any]:
    entries = _regular_tree_entries(dataset_root, "payload/lerobot_train")
    total_bytes = sum(int(item["bytes"]) for item in entries)
    tree_sha = _tree_digest(entries)
    observed = {
        "tree_sha256": tree_sha,
        "regular_file_count": len(entries),
        "total_bytes": total_bytes,
        "episode_count": LABEL_COUNT,
        "frame_count": FRAME_COUNT,
        "role": "TRAIN_ONLY",
    }
    if (
        tree_sha != EXPECTED_DATASET_TREE_SHA256
        or len(entries) != EXPECTED_DATASET_FILE_COUNT
        or total_bytes != EXPECTED_DATASET_TOTAL_BYTES
    ):
        raise RuntimeError(f"FULL24 dataset identity mismatch: {observed}")
    return observed


def _snapshot_json(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"snapshot is not an object: {path}")
    return payload


def _load_snapshot_label(path: Path, *, target: str, snapshot_index: int) -> Any:
    """Rebuild the exact Stage38 label without requiring camera files."""

    payload = _snapshot_json(path)
    expected_id = f"stage22_expert_{target}:expert-{snapshot_index:04d}"
    if payload.get("schema") != "SameSceneExpertSnapshotV1":
        raise RuntimeError(f"snapshot schema mismatch: {path}")
    if payload.get("example_id") != expected_id or payload.get("target_id") != target:
        raise RuntimeError(f"snapshot identity mismatch: {path}")
    if payload.get("raw_episode_id") != f"stage22_expert_{target}":
        raise RuntimeError(f"snapshot raw episode mismatch: {path}")
    if payload.get("target_excluded_from_model_input") is not True:
        raise RuntimeError(f"target exclusion flag is not true: {path}")
    if payload.get("source_frame") != SOURCE_FRAME or payload.get("tcp_semantic_id") != TCP:
        raise RuntimeError(f"snapshot frame/TCP contract mismatch: {path}")
    if payload.get("target_position_world_m") != TARGET_POSITIONS[target]:
        raise RuntimeError(f"snapshot target metadata mismatch: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("language") != LANGUAGE:
        raise RuntimeError(f"snapshot language provenance mismatch: {path}")
    state_payload = payload.get("state_v1")
    if not isinstance(state_payload, Mapping) or state_payload.get("schema") != "StateV1":
        raise RuntimeError(f"StateV1 payload missing: {path}")
    state = np.asarray(state_payload.get("values"), dtype=np.float64)
    timestamps = np.asarray(payload.get("future_timestamps"), dtype=np.float64)
    desired = np.asarray(payload.get("future_desired_eef_9d"), dtype=np.float64)
    gripper = np.asarray(payload.get("future_gripper"), dtype=np.float64)
    if state.shape != (10,) or timestamps.shape != (ACTION_HORIZON,):
        raise RuntimeError(f"snapshot state/timestamp shape mismatch: {path}")
    if desired.shape != (ACTION_HORIZON, 9) or gripper.shape != (ACTION_HORIZON, 1):
        raise RuntimeError(f"snapshot physical label shape mismatch: {path}")
    if not np.isfinite(state).all() or not np.isfinite(timestamps).all():
        raise RuntimeError(f"snapshot state/timestamp contains non-finite values: {path}")
    if not np.isfinite(desired).all() or not np.isfinite(gripper).all():
        raise RuntimeError(f"snapshot physical label contains non-finite values: {path}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError(f"snapshot timestamps are not strictly increasing: {path}")
    if not np.allclose(state[:9], desired[0], rtol=0.0, atol=1e-10):
        raise RuntimeError(f"snapshot anchor mismatch: {path}")
    from go2arx5_vla.structured_action import encode_structured_action

    return encode_structured_action(
        np.concatenate([desired, gripper], axis=-1),
        timestamps,
        state,
        timing_mode="T16",
        chunk_id=expected_id,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP,
        observation_timestamp=float(payload["observation_timestamp"]),
        provenance={
            "stage": "Stage45E-N2",
            "source_example_id": expected_id,
            "target_id": target,
            "target_excluded_from_model_input": True,
        },
    )


def load_full24_labels(snapshot_root: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    labels: list[Any] = []
    records: list[dict[str, Any]] = []
    for target in TARGETS:
        for snapshot_index in range(6):
            path = Path(snapshot_root) / f"stage22_expert_{target}" / f"expert-{snapshot_index:04d}.json"
            label = _load_snapshot_label(path, target=target, snapshot_index=snapshot_index)
            labels.append(label)
            records.append(
                {
                    "window_index": len(labels) - 1,
                    "target_id": target,
                    "snapshot_index": snapshot_index,
                    "example_id": f"stage22_expert_{target}:expert-{snapshot_index:04d}",
                    "json_sha256": sha256_file(path),
                    "json_bytes": int(path.stat().st_size),
                }
            )
    if len(labels) != LABEL_COUNT:
        raise RuntimeError(f"expected {LABEL_COUNT} FULL24 labels, got {len(labels)}")
    return labels, records


def load_full24_labels_from_dataset(dataset_root: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    """Rebuild the 24 labels from the sealed materialized TRAIN parquet.

    The Stage45C snapshot-witness relay is intentionally partial on Corelab;
    the exact 85-file materialized TRAIN tree is the canonical source for the
    full clipping census.  This path reads only parquet metadata/arrays and
    never opens video or constructs a simulator.
    """

    import pandas as pd
    from go2arx5_vla.structured_action import encode_structured_action

    dataset_root = Path(dataset_root)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    rows = [
        item
        for item in (
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    if len(rows) != LABEL_COUNT:
        raise RuntimeError(f"FULL24 episodes metadata has {len(rows)} rows, expected {LABEL_COUNT}")
    labels: list[Any] = []
    records: list[dict[str, Any]] = []
    for index, episode in enumerate(rows):
        expected_target = TARGETS[index // 6]
        expected_snapshot = index % 6
        expected_source_id = f"stage22_expert_{expected_target}:expert-{expected_snapshot:04d}"
        if episode.get("episode_index") != index or episode.get("length") != ACTION_HORIZON:
            raise RuntimeError(f"FULL24 episode metadata mismatch at index {index}")
        if episode.get("split") != "train" or episode.get("tasks") != [LANGUAGE]:
            raise RuntimeError(f"FULL24 split/task metadata mismatch at index {index}")
        if episode.get("source_episode_id") != expected_source_id:
            raise RuntimeError(f"FULL24 window order mismatch at index {index}")
        parquet = dataset_root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        if parquet.is_symlink() or not parquet.is_file():
            raise RuntimeError(f"missing canonical FULL24 parquet: {parquet}")
        frame = pd.read_parquet(parquet, columns=["timestamp", "observation.state", "action"])
        timestamps = np.asarray(frame["timestamp"].tolist(), dtype=np.float64).reshape(-1)
        state_rows = np.stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in frame["observation.state"]])
        action_rows = np.stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in frame["action"]])
        if timestamps.shape != (ACTION_HORIZON,) or state_rows.shape != (ACTION_HORIZON, 10) or action_rows.shape != (ACTION_HORIZON, 10):
            raise RuntimeError(f"FULL24 parquet shape mismatch at episode {index}")
        if not np.isfinite(timestamps).all() or not np.isfinite(state_rows).all() or not np.isfinite(action_rows).all():
            raise RuntimeError(f"FULL24 parquet non-finite values at episode {index}")
        if np.any(np.diff(timestamps) <= 0.0):
            raise RuntimeError(f"FULL24 timestamps are not strictly increasing at episode {index}")
        if not np.allclose(state_rows[0], action_rows[0], rtol=0.0, atol=1e-6):
            raise RuntimeError(f"FULL24 state/action anchor mismatch at episode {index}")
        labels.append(
            encode_structured_action(
                action_rows,
                timestamps,
                state_rows[0],
                timing_mode="T16",
                chunk_id=expected_source_id,
                source_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP,
                observation_timestamp=float(timestamps[0]),
                provenance={
                    "stage": "Stage45E-N2",
                    "source_example_id": expected_source_id,
                    "target_id": expected_target,
                    "target_excluded_from_model_input": True,
                    "materialized_parquet_read_only": True,
                },
            )
        )
        records.append(
            {
                "window_index": index,
                "target_id": expected_target,
                "snapshot_index": expected_snapshot,
                "example_id": expected_source_id,
                "parquet_path_relative": f"data/chunk-000/episode_{index:06d}.parquet",
                "parquet_sha256": sha256_file(parquet),
                "parquet_bytes": int(parquet.stat().st_size),
            }
        )
    return labels, records


def _clip_metrics(labels: Sequence[Any], lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    if len(labels) != LABEL_COUNT:
        raise RuntimeError(f"expected {LABEL_COUNT} labels for clipping audit")
    values = np.stack([np.asarray(label.values, dtype=np.float64) for label in labels], axis=0)
    masks = np.stack([np.asarray(label.mask, dtype=np.float64) for label in labels], axis=0).astype(bool)
    if values.shape != (LABEL_COUNT, ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError(f"unexpected structured label tensor shape: {values.shape}")
    active_values: list[float] = []
    xyz_values: list[float] = []
    active_deltas: list[float] = []
    xyz_deltas: list[float] = []
    active_norm_deltas: list[float] = []
    xyz_norm_deltas: list[float] = []
    active_outside = 0
    xyz_outside = 0
    for dim in range(ACTION_DIM):
        observed = values[:, :, dim][masks[:, :, dim]]
        if observed.size == 0:
            continue
        clipped = np.clip(observed, lower[dim], upper[dim])
        delta = np.abs(clipped - observed)
        scale = upper[dim] - lower[dim]
        norm_delta = np.zeros_like(delta) if scale == 0.0 else np.abs(
            np.clip(2.0 * (observed - lower[dim]) / scale - 1.0, -1.0, 1.0)
            - (2.0 * (observed - lower[dim]) / scale - 1.0)
        )
        active_values.extend(observed.tolist())
        active_deltas.extend(delta.tolist())
        active_norm_deltas.extend(norm_delta.tolist())
        active_outside += int(np.count_nonzero(delta > 0.0))
        if dim in XYZ_DIMS:
            xyz_values.extend(observed.tolist())
            xyz_deltas.extend(delta.tolist())
            xyz_norm_deltas.extend(norm_delta.tolist())
            xyz_outside += int(np.count_nonzero(delta > 0.0))
    if len(active_values) != LABEL_COUNT * ACTIVE_SCALAR_COUNT:
        raise RuntimeError(
            "FULL24 active scalar audit count mismatch: "
            f"{len(active_values)} != {LABEL_COUNT * ACTIVE_SCALAR_COUNT}"
        )
    if len(xyz_values) != LABEL_COUNT * GEOMETRY_FUTURE_COUNT * len(XYZ_DIMS):
        raise RuntimeError(f"FULL24 XYZ scalar audit count mismatch: {len(xyz_values)}")

    def summarize(deltas: Sequence[float]) -> dict[str, float]:
        array = np.asarray(deltas, dtype=np.float64)
        return {
            "max_abs": float(np.max(array)) if array.size else 0.0,
            "mean_abs": float(np.mean(array)) if array.size else 0.0,
            "l2": float(np.linalg.norm(array)) if array.size else 0.0,
            "nonzero_count": int(np.count_nonzero(array > 0.0)),
        }

    return {
        "active_scalar_count_observed": len(active_values),
        "xyz_scalar_count_observed": len(xyz_values),
        "n2_outside_count_all_active": active_outside,
        "n2_outside_count_xyz": xyz_outside,
        "physical_clipping_delta_all_active": summarize(active_deltas),
        "physical_clipping_delta_xyz": summarize(xyz_deltas),
        "normalized_clipping_delta_all_active": summarize(active_norm_deltas),
        "normalized_clipping_delta_xyz": summarize(xyz_norm_deltas),
    }


def _compare_non_xyz(source_stats: Mapping[str, Any], n2_stats: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that the intervention is limited to selected XYZ bound entries."""

    changed: list[str] = []
    for key in source_stats:
        if key not in n2_stats:
            changed.append(key)
            continue
        if key not in {"selected_lower", "selected_upper"}:
            if source_stats[key] != n2_stats[key]:
                changed.append(key)
            continue
        before = source_stats[key]
        after = n2_stats[key]
        if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
            changed.append(key)
            continue
        for index, (left, right) in enumerate(zip(before, after)):
            if index not in XYZ_DIMS and left != right:
                changed.append(f"{key}[{index}]")
    if changed:
        raise RuntimeError(f"N2 changed fields outside selected XYZ bounds: {changed}")
    return {
        "all_non_xyz_statistics_exact": True,
        "changed_fields": ["selected_lower[0:3]", "selected_upper[0:3]"],
        "source_statistics_sha256": _canonical_sha256(source_stats),
        "n2_statistics_sha256": _canonical_sha256(n2_stats),
    }


def build_n2_document(
    source: Mapping[str, Any],
    *,
    source_sha256: str = SOURCE_STATS_SHA256,
    payload_sha256: str = SOURCE_PAYLOAD_SHA256,
    labels: Sequence[Any] | None = None,
    snapshot_records: Sequence[Mapping[str, Any]] | None = None,
    dataset_identity: Mapping[str, Any] | None = None,
    repository_head: str | None = None,
) -> dict[str, Any]:
    source_check = validate_source_document(source)
    if source_sha256 != SOURCE_STATS_SHA256 or payload_sha256 != SOURCE_PAYLOAD_SHA256:
        raise RuntimeError("N2 source identity constants do not match accepted runtime-V2 authority")
    source_stats = source["statistics"]
    lower = source_check["selected_lower"]
    upper = source_check["selected_upper"]
    centers = (lower + upper) / 2.0
    n2_lower = lower.copy()
    n2_upper = upper.copy()
    for dim in XYZ_DIMS:
        n2_lower[dim] = centers[dim] - HALF_RANGE_M
        n2_upper[dim] = centers[dim] + HALF_RANGE_M
    n2_stats = copy.deepcopy(source_stats)
    n2_stats["selected_lower"] = n2_lower.tolist()
    n2_stats["selected_upper"] = n2_upper.tolist()
    non_xyz = _compare_non_xyz(source_stats, n2_stats)
    clipping = _clip_metrics(labels, n2_lower, n2_upper) if labels is not None else {
        "status": "PENDING_LABEL_AUDIT",
        "active_scalar_count_expected": LABEL_COUNT * ACTIVE_SCALAR_COUNT,
        "xyz_scalar_count_expected": LABEL_COUNT * GEOMETRY_FUTURE_COUNT * len(XYZ_DIMS),
    }
    if labels is not None and len(labels) != LABEL_COUNT:
        raise RuntimeError("N2 label audit requires exactly 24 labels")
    execution_boundary = {
        "gpu": False,
        "model_constructed": False,
        "gr00t_forwards": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "wbc_steps": 0,
        "simulator_steps": 0,
        "pbs_submissions": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "new_data_generation": False,
    }
    artifact = {
        "schema": N2_SCHEMA,
        "status": N2_STATUS,
        "stage": "Stage45E",
        "authority": {
            "pi_gate": "PI_GATE_STATE_0121",
            "issue_comment": 5633708741,
            "branch": "stage45e/n2-balanced-xyz-normalization",
            "base_head": "a843294780c1392369bfc987e468857c63fc13a5",
        },
        "repository": {
            "head": repository_head,
            "working_tree_required_clean": True,
        },
        "intervention": {
            "kind": "ONE_FACTOR_XYZ_SELECTED_BOUND_REPLACEMENT",
            "normalization": "MINMAX_TO_NEG1_POS1",
            "use_percentiles": True,
            "robust_centers_unchanged": True,
            "center_definition": "0.5*(accepted_FULL24_selected_lower+accepted_FULL24_selected_upper)",
            "common_half_range_m": HALF_RANGE_M,
            "xyz_universal_dimensions": list(XYZ_DIMS),
            "center_xyz": centers[list(XYZ_DIMS)].tolist(),
            "selected_lower_xyz": n2_lower[list(XYZ_DIMS)].tolist(),
            "selected_upper_xyz": n2_upper[list(XYZ_DIMS)].tolist(),
            "non_xyz_selected_bounds_unchanged": True,
        },
        "source_runtime_v2": {
            "outer_sha256": source_sha256,
            "payload_sha256": payload_sha256,
            "schema": SOURCE_SCHEMA,
            "protocol_revision": SOURCE_PROTOCOL,
            "source_kind": source.get("canonical_label_source", {}).get("kind"),
            "source_tree_sha256": source.get("canonical_label_source", {}).get("tree_sha256"),
            "materialization_path_embedded": False,
            "historical_scratch_path_embedded": False,
        },
        "statistics": n2_stats,
        "statistics_payload_sha256": _canonical_sha256(n2_stats),
        "statistics_change_audit": non_xyz,
        "dataset": dict(dataset_identity or {
            "id": "FULL24_EXISTING_TRAIN",
            "role": "TRAIN_ONLY",
            "tree_sha256": EXPECTED_DATASET_TREE_SHA256,
            "regular_file_count": EXPECTED_DATASET_FILE_COUNT,
            "total_bytes": EXPECTED_DATASET_TOTAL_BYTES,
            "episode_count": LABEL_COUNT,
            "frame_count": FRAME_COUNT,
            "window_count": LABEL_COUNT,
            "new_data_generation": False,
            "validation_e_read": False,
            "holdout_f_read": False,
        }),
        "label_and_window_audit": {
            "label_contract": "Stage38 StructuredActionV1 encode_structured_action",
            "window_order": "target-major then snapshot-major",
            "window_count": LABEL_COUNT,
            "frame_count": FRAME_COUNT,
            "all_windows_unchanged": True,
            "snapshot_records": list(snapshot_records or []),
            "snapshot_record_count": len(snapshot_records or []),
            "target_xyz_in_model_input": False,
            "mask": {
                "timing_mode": "T16",
                "shape": [ACTION_HORIZON, ACTION_DIM],
                "active_scalar_count": ACTIVE_SCALAR_COUNT,
                "unchanged": True,
            },
        },
        "clipping_audit": clipping,
        "training_recipe": {
            "starting_checkpoint": "BASE",
            "base_tree_sha256": "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4",
            "seed": 2222,
            "max_steps": 2000,
            "save_steps": [1000, 2000],
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw_torch",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": "cosine",
            "warmup_steps": 10,
            "warmup_ratio": 0.0,
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "workers": 0,
            "eval": "no",
            "resume": False,
            "save_only_model": False,
            "tune_projector": True,
            "tune_diffusion": True,
            "tune_visual": False,
            "tune_llm": False,
            "tune_vlln": False,
            "relative": True,
            "percentiles": True,
        },
        "preserved_contracts": {
            "structured_action_schema": "StructuredActionV1",
            "structured_action_shape": [1, ACTION_HORIZON, ACTION_DIM],
            "decoded_action_shape": [ACTION_HORIZON, 10],
            "geometry": "K16",
            "timing": "T16",
            "source_frame": SOURCE_FRAME,
            "tcp_semantic_id": TCP,
            "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
            "target_xyz_in_model_input": False,
            "d1_decoder_changed": False,
            "wbc_changed": False,
        },
        "execution_boundary": execution_boundary,
        "next_gate": "STAGE45E_CORELAB_GPU_ADMISSION_AFTER_SCIENCE_COMMIT",
        "return_token": "STAGE45E_N2_CPU_STATIC_ARTIFACT_READY",
    }
    return artifact


def verify_n2_document(document: Mapping[str, Any], *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if document.get("schema") != N2_SCHEMA or document.get("status") != N2_STATUS:
        raise RuntimeError("N2 artifact schema/status mismatch")
    if document.get("intervention", {}).get("common_half_range_m") != HALF_RANGE_M:
        raise RuntimeError("N2 half-range drift")
    stats = document.get("statistics")
    if not isinstance(stats, Mapping):
        raise RuntimeError("N2 statistics missing")
    _validate_mask(stats.get("active_mask"))
    _require_finite_vector(stats.get("selected_lower"), name="N2.selected_lower")
    _require_finite_vector(stats.get("selected_upper"), name="N2.selected_upper")
    if source is not None:
        validate_source_document(source)
        _compare_non_xyz(source["statistics"], stats)
    boundary = document.get("execution_boundary", {})
    for key in ("gpu", "model_constructed", "gr00t_forwards", "training_steps", "backward_steps", "optimizer_steps", "scheduler_steps", "checkpoint_writes", "wbc_steps", "simulator_steps", "pbs_submissions", "new_data_generation"):
        if boundary.get(key) not in (False, 0):
            raise RuntimeError(f"N2 artifact execution boundary is not zero for {key}")
    if boundary.get("validation_e_read") is not False or boundary.get("holdout_f_read") is not False:
        raise RuntimeError("N2 artifact touched a locked evaluation split")
    if document.get("source_runtime_v2", {}).get("materialization_path_embedded") is not False:
        raise RuntimeError("N2 artifact embeds a runtime materialization path")
    return {
        "schema": N2_SCHEMA,
        "statistics_payload_sha256": _canonical_sha256(stats),
        "all_non_xyz_statistics_exact": True,
        "execution_boundary_zero": True,
        "historical_scratch_path_embedded": False,
    }


def build_from_paths(
    *,
    source_path: Path,
    output_path: Path,
    snapshot_root: Path | None = None,
    dataset_root: Path | None = None,
    repository_head: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    source = _read_json(source_path)
    source_identity = validate_source_document(source, source_path=source_path)
    labels = None
    snapshot_records: list[dict[str, Any]] = []
    dataset_identity = validate_dataset_identity(Path(dataset_root)) if dataset_root is not None else None
    if dataset_root is not None:
        labels, snapshot_records = load_full24_labels_from_dataset(Path(dataset_root))
    elif snapshot_root is not None:
        labels, snapshot_records = load_full24_labels(Path(snapshot_root))
    document = build_n2_document(
        source,
        source_sha256=source_identity["outer_sha256"] or SOURCE_STATS_SHA256,
        payload_sha256=source_identity["payload_sha256"],
        labels=labels,
        snapshot_records=snapshot_records,
        dataset_identity=dataset_identity,
        repository_head=repository_head,
    )
    verify_n2_document(document, source=source)
    _write_json(Path(output_path), document)
    artifact_sha = sha256_file(Path(output_path))
    return {
        "schema": "CORELAB_STAGE45E_N2_STATS_RESULT_V1",
        "status": "PASS",
        "artifact": {"path": str(Path(output_path)), "sha256": artifact_sha, "bytes": Path(output_path).stat().st_size},
        "statistics_payload_sha256": document["statistics_payload_sha256"],
        "source_runtime_v2": document["source_runtime_v2"],
        "intervention": document["intervention"],
        "clipping_audit": document["clipping_audit"],
        "dataset": document["dataset"],
        "execution_boundary": document["execution_boundary"],
        "return_token": document["return_token"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repository-head")
    args = parser.parse_args()
    try:
        result = build_from_paths(
            source_path=args.source_stats,
            output_path=args.output,
            snapshot_root=args.snapshot_root,
            dataset_root=args.dataset_root,
            repository_head=args.repository_head,
        )
    except Exception as exc:  # pragma: no cover - command-line fail-closed surface
        print(f"STAGE45E_N2_STATS_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
