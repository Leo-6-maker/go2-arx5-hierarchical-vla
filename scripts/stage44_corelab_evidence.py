#!/usr/bin/env python3
"""Stage44 Corelab training-lineage and behavior diagnosis harness.

The Stage44 gate deliberately separates CPU evidence from the only model
runtime operation allowed in this stage:

* ``audit-pre`` performs A0--A2 and freezes an execution bundle;
* :mod:`stage44_corelab_train_forwards` performs exactly 24 TRAIN-only GR00T
  forwards (A3), with no target XYZ in the model-facing input;
* ``audit-post`` performs A4--A5 and emits the PI stop package.

No function in this module imports IsaacGym, CUDA, GR00T, a checkpoint, or a
WBC source.  NumPy is used only for finite-array and metric checks on the
node-local evidence package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PI_GATE = "PI_GATE_STATE_0069"
ISSUE_COMMENT = None
STAGE_ID = "Stage44"
BRANCH = "stage44-dev/corelab-training-audit"
STAGE43_HEAD = "b2a70dcec0765cf69819bb76c180e5c92ae3eac3"
STAGE43_BUNDLE_SHA256 = "127d95a2a5b9a6a40e864e029d85fd09a9e4dd55441aae58071502aa76700da6"
STAGE43_RESULT_SHA256 = "e635ecaa321bde38903be45781e9b47063ba6b81c2bc810b1a1ae64925ecfcdb"
STAGE43_SEAL_SHA256 = "668bee50dc38997403a065d5f4831662731715cf28c289af3982e4f9abbf7283"
STAGE43_COHORT_MANIFEST_SHA256 = "62e95728ea6da7a0d90613e52adabc0737f3b116f63d4f01ebb680387ea2de48"
STAGE43_TOKEN = "CORELAB_STAGE43_CONTROLLED_INITIAL_CONDITION_COHORT_CHARACTERIZED"
STAGE22_TRAIN_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
TINY1_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
TINY1_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
GROOT_SOURCE_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
STRUCTURED_CHECKPOINT_TREE_SHA256 = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
STRUCTURED_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
DEV_SEED = 4242
ACTION_DT_S = 0.05
TARGET_POSITION_WORLD_M = [0.30, 0.10, 0.58]
TRAIN_TARGETS = {
    "red_target_train_a": [0.30, 0.10, 0.58],
    "red_target_train_b": [0.30, -0.10, 0.58],
    "red_target_train_c": [0.30, 0.05, 0.64],
    "red_target_train_d": [0.30, -0.05, 0.63],
}
TRAIN_TARGET_ORDER = tuple(TRAIN_TARGETS)
SNAPSHOT_COUNT = 24
SNAPSHOT_HORIZON = 40
MODEL_INPUT_KEYS = ("body_rgb", "wrist_rgb", "state")

FORBIDDEN_FIELDS = (
    "training", "fine_tuning", "backward", "optimizer_step", "checkpoint_write",
    "validation_e_read", "holdout_f_read", "new_data_read", "hardware_executed",
    "h200_used", "vanda_used", "wbc_imported", "wbc_executed", "reference_update",
    "physics_reset", "physics_teleport", "policy_history_clear", "target_in_model_input",
    "checkpoint_substitution", "model_substitution", "processor_substitution",
    "precision_change", "quantization", "cpu_offload", "clipping", "smoothing",
    "blending", "adaptive_retiming", "target_relocation", "joint_perturbation",
    "orientation_perturbation", "velocity_perturbation", "settling_steps",
)


class Stage44EvidenceError(ValueError):
    """Raised when the Stage44 evidence is not admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage44EvidenceError(message)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value)).tobytes())


def read_json(path: Path, *, canonical: bool = False) -> Dict[str, Any]:
    path = Path(path).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "JSON is not a regular file: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required: %s" % path)
    if canonical:
        _require(path.read_bytes() == canonical_json_bytes(value), "JSON is not canonical: %s" % path)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    destination.write_bytes(payload)
    return sha256_bytes(payload)


def forbidden_operations() -> Dict[str, bool]:
    return {key: False for key in FORBIDDEN_FIELDS}


def _finite(value: Any, shape: Tuple[int, ...], label: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    _require(tuple(array.shape) == shape and bool(np.isfinite(array).all()), "%s must be finite %s" % (label, shape))
    return array


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve(strict=True).relative_to(Path(root).resolve(strict=True)).as_posix()


def tree_digest(root: Path, *, prefix: str = "") -> Dict[str, Any]:
    """Hash regular files using the Stage22 handoff ordering."""

    root = Path(root).resolve(strict=True)
    rows: List[Dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Stage44EvidenceError("symlink in evidence tree: %s" % path)
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": (prefix + relative) if prefix else relative, "bytes": size, "sha256": digest})
        total += size
    payload = "".join("%s\0%s\0%s\n" % (row["path"], row["bytes"], row["sha256"]) for row in rows).encode("utf-8")
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": total,
        "tree_sha256": sha256_bytes(payload),
        "symlink_count": 0,
        "entries": rows,
    }


def _expected_snapshot_ids() -> List[str]:
    return [
        "stage22_expert_%s:expert-%04d" % (target, index)
        for target in TRAIN_TARGET_ORDER
        for index in range(6)
    ]


def _snapshot_path(expert_root: Path, target: str, index: int) -> Path:
    path = Path(expert_root) / "snapshots" / ("stage22_expert_%s" % target) / ("expert-%04d.json" % index)
    _require(path.is_file() and not path.is_symlink(), "missing TRAIN snapshot: %s" % path)
    return path.resolve(strict=True)


def _ppm_rgb(path: Path) -> np.ndarray:
    """Read a dependency-free binary PPM (P6) frame."""

    data = Path(path).read_bytes()
    cursor = 0

    def token() -> bytes:
        nonlocal cursor
        while cursor < len(data) and data[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            return token()
        start = cursor
        while cursor < len(data) and data[cursor] not in b" \t\r\n":
            cursor += 1
        return data[start:cursor]

    _require(token() == b"P6", "PPM is not P6: %s" % path)
    width = int(token()); height = int(token()); max_value = int(token())
    _require(width == 96 and height == 96 and max_value == 255, "PPM ABI mismatch: %s" % path)
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
    payload = data[cursor:]
    _require(len(payload) == width * height * 3, "PPM payload length mismatch: %s" % path)
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(bool(array.size) and bool(np.isfinite(array).all()), "metric sample is empty/nonfinite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _source_sha(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "source file is missing: %s" % path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _stage43_predecessor(stage43_root: Path) -> Dict[str, Any]:
    root = Path(stage43_root).resolve(strict=True)
    bundle_path = root / "stage43_execution_bundle.json"
    result_path = root / "stage43_result.json"
    seal_path = root / "stage43_result.seal.json"
    cohort_path = root / "stage43_cohort_manifest.json"
    bundle = read_json(bundle_path, canonical=True)
    result = read_json(result_path, canonical=True)
    seal = read_json(seal_path, canonical=True)
    cohort = read_json(cohort_path, canonical=True)
    _require(sha256_file(bundle_path) == STAGE43_BUNDLE_SHA256, "Stage43 bundle hash mismatch")
    _require(sha256_file(result_path) == STAGE43_RESULT_SHA256, "Stage43 result hash mismatch")
    _require(sha256_file(seal_path) == STAGE43_SEAL_SHA256, "Stage43 result seal hash mismatch")
    _require(sha256_file(cohort_path) == STAGE43_COHORT_MANIFEST_SHA256, "Stage43 cohort manifest hash mismatch")
    _require(result.get("status") == "PASS" and result.get("token") == STAGE43_TOKEN, "Stage43 result is not accepted")
    _require(seal.get("status") == "PASS" and seal.get("result_sha256") == STAGE43_RESULT_SHA256, "Stage43 seal does not bind result")
    _require(bundle.get("stage_id") == "Stage43" and bundle.get("budgets", {}).get("episodes") == 5, "Stage43 bundle identity drift")
    _require(bundle.get("artifacts", {}).get("structured_checkpoint", {}).get("tree_sha256") == STRUCTURED_CHECKPOINT_TREE_SHA256, "structured checkpoint identity drift")
    _require(bundle.get("artifacts", {}).get("tiny1_dataset", {}).get("tree_sha256") == TINY1_TREE_SHA256, "TINY1 identity drift")
    _require(bundle.get("artifacts", {}).get("structured_stats", {}).get("sha256") == STRUCTURED_STATS_SHA256, "structured stats identity drift")
    _require(bundle.get("artifacts", {}).get("cosmos_snapshot", {}).get("revision") == COSMOS_REVISION, "Cosmos revision drift")
    _require(bundle.get("artifacts", {}).get("cosmos_snapshot", {}).get("tree_sha256") == COSMOS_TREE_SHA256, "Cosmos tree identity drift")
    return {
        "root": str(root),
        "bundle": bundle,
        "result": result,
        "seal": seal,
        "cohort_manifest": cohort,
        "bundle_sha256": STAGE43_BUNDLE_SHA256,
        "result_sha256": STAGE43_RESULT_SHA256,
        "seal_sha256": STAGE43_SEAL_SHA256,
        "cohort_manifest_sha256": STAGE43_COHORT_MANIFEST_SHA256,
    }


def audit_a0(repo: Path, stage43_root: Path, expert_root: Path, train_dataset: Path) -> Dict[str, Any]:
    """A0: prove accepted lineage and an exact TRAIN-only split."""

    predecessor = _stage43_predecessor(stage43_root)
    expert_root = Path(expert_root).resolve(strict=True)
    train_dataset = Path(train_dataset).resolve(strict=True)
    split = read_json(train_dataset / "meta" / "stage22_split_lock.json")
    _require(split.get("selected_split") == "train" and split.get("dataset_role") == "TRAIN", "Stage22 split is not TRAIN")
    _require(split.get("training_allowed") is True and split.get("final_test_in_training") is False, "TRAIN split lock is unsafe")
    _require(split.get("validation_target_ids") == [] and split.get("final_test_target_ids") == [], "validation/final-test target leaked into TRAIN")
    _require(split.get("selected_target_ids") == list(TRAIN_TARGET_ORDER), "TRAIN target order drift")

    episode_rows: List[Dict[str, Any]] = []
    episodes_path = train_dataset / "meta" / "episodes.jsonl"
    _require(episodes_path.is_file(), "TRAIN episodes metadata is missing")
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            _require(isinstance(row, dict), "TRAIN episode row is not an object")
            episode_rows.append(row)
    expected_ids = _expected_snapshot_ids()
    observed_ids = [str(row.get("source_episode_id")) for row in episode_rows]
    _require(observed_ids == expected_ids, "TRAIN source episode order/count drift")
    _require(all(row.get("split") == "train" and row.get("length") == 40 for row in episode_rows), "TRAIN episode metadata drift")
    _require(all("holdout" not in source.lower() and "validation" not in source.lower() for source in observed_ids), "holdout/validation source entered TRAIN lineage")

    train_tree = tree_digest(train_dataset, prefix="payload/lerobot_train/")
    _require(train_tree["tree_sha256"] == STAGE22_TRAIN_TREE_SHA256, "Stage22 TRAIN tree hash mismatch")
    _require(train_tree["file_count"] == 85 and train_tree["total_bytes"] == 919661, "Stage22 TRAIN tree size drift")

    snapshot_records: List[Dict[str, Any]] = []
    for target in TRAIN_TARGET_ORDER:
        summary_path = expert_root / "episodes" / ("stage22_expert_%s" % target) / "expert_episode_summary.json"
        summary = read_json(summary_path)
        _require(summary.get("status") == "EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS", "expert episode is not PASS: %s" % target)
        _require(summary.get("target_id") == target and summary.get("target_position_world_m") == TRAIN_TARGETS[target], "expert target drift: %s" % target)
        _require(summary.get("fixed_reset_seed") == 13 and summary.get("snapshot_count") == 6 and summary.get("valid_steps") == 24, "expert episode contract drift: %s" % target)
        _require(summary.get("target_excluded_from_model_input") is True, "expert target exclusion drift: %s" % target)
        _require(len(summary.get("snapshots", [])) == 6, "expert snapshot count drift: %s" % target)
        for index in range(6):
            snapshot = _snapshot_path(expert_root, target, index)
            payload = read_json(snapshot)
            expected_id = "stage22_expert_%s:expert-%04d" % (target, index)
            _require(payload.get("example_id") == expected_id and payload.get("target_id") == target, "snapshot lineage drift: %s" % expected_id)
            _require("holdout" not in str(snapshot).lower(), "holdout snapshot path entered A0")
            snapshot_records.append({
                "example_id": expected_id,
                "target_id": target,
                "snapshot_path": str(snapshot),
                "snapshot_sha256": sha256_file(snapshot),
                "body_frame_path": str(Path(str(payload.get("body_frame_path"))).resolve(strict=True)),
                "wrist_frame_path": str(Path(str(payload.get("wrist_frame_path"))).resolve(strict=True)),
                "summary_path": str(summary_path.resolve(strict=True)),
                "summary_sha256": sha256_file(summary_path),
            })

    source_files = [
        _source_sha(repo / "scripts" / "stage44_corelab_evidence.py"),
        _source_sha(repo / "scripts" / "stage43_corelab_evidence.py"),
        _source_sha(repo / "scripts" / "stage43_corelab_finalize.py"),
    ]
    return {
        "schema": "CORELAB_STAGE44_A0_TRAINING_LINEAGE_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44_A0_TRAINING_LINEAGE_PASS",
        "repository": {"branch": BRANCH, "stage43_accepted_head": STAGE43_HEAD},
        "predecessor": {
            "stage43_bundle_sha256": predecessor["bundle_sha256"],
            "stage43_result_sha256": predecessor["result_sha256"],
            "stage43_result_seal_sha256": predecessor["seal_sha256"],
            "stage43_cohort_manifest_sha256": predecessor["cohort_manifest_sha256"],
            "stage43_token": STAGE43_TOKEN,
        },
        "artifacts": {
            "stage43_root": str(Path(stage43_root).resolve()),
            "expert_root": str(expert_root),
            "train_dataset": str(train_dataset),
            "train_split_lock": _source_sha(train_dataset / "meta" / "stage22_split_lock.json"),
            "train_episodes": _source_sha(episodes_path),
        },
        "train_split_lock": split,
        "train_dataset_tree": train_tree,
        "expected_snapshot_count": SNAPSHOT_COUNT,
        "snapshot_records": snapshot_records,
        "accepted_model_lineage": {
            "groot_source_sha": GROOT_SOURCE_SHA,
            "structured_checkpoint_tree_sha256": STRUCTURED_CHECKPOINT_TREE_SHA256,
            "structured_stats_sha256": STRUCTURED_STATS_SHA256,
            "tiny1_tree_sha256": TINY1_TREE_SHA256,
            "tiny1_stats_sha256": TINY1_STATS_SHA256,
            "tiny1_relative_stats_sha256": TINY1_RELATIVE_STATS_SHA256,
            "modality_sha256": MODALITY_SHA256,
            "cosmos_revision": COSMOS_REVISION,
            "cosmos_tree_sha256": COSMOS_TREE_SHA256,
        },
        "source_files": source_files,
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": "accepted TRAIN lineage only; no model forward, WBC, validation, or holdout read",
    }


def _audit_snapshot(snapshot_path: Path, target: str, index: int) -> Dict[str, Any]:
    payload = read_json(snapshot_path)
    expected_id = "stage22_expert_%s:expert-%04d" % (target, index)
    _require(payload.get("schema") == "SameSceneExpertSnapshotV1", "wrong snapshot schema: %s" % expected_id)
    _require(payload.get("example_id") == expected_id and payload.get("raw_episode_id") == "stage22_expert_%s" % target, "snapshot ID drift: %s" % expected_id)
    _require(payload.get("target_id") == target and payload.get("target_position_world_m") == TRAIN_TARGETS[target], "snapshot target drift: %s" % expected_id)
    _require(payload.get("source_frame") == SOURCE_FRAME and payload.get("tcp_semantic_id") == TCP_SEMANTIC_ID, "snapshot frame drift: %s" % expected_id)
    _require(payload.get("target_excluded_from_model_input") is True, "target was not excluded from model input: %s" % expected_id)
    provenance = payload.get("provenance", {})
    _require(provenance.get("language") == LANGUAGE and provenance.get("target_excluded_from_model_input") is True, "snapshot model boundary drift: %s" % expected_id)
    _require(provenance.get("label_source") == "deterministic_smoothstep_position_hold", "expert label source drift: %s" % expected_id)
    state_payload = payload.get("state_v1", {})
    _require(state_payload.get("schema") == "StateV1" and state_payload.get("layout") == ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"], "StateV1 ABI drift: %s" % expected_id)
    state = _finite(state_payload.get("values"), (10,), expected_id + ".state")
    timestamps = _finite(payload.get("future_timestamps"), (SNAPSHOT_HORIZON,), expected_id + ".timestamps")
    desired = _finite(payload.get("future_desired_eef_9d"), (SNAPSHOT_HORIZON, 9), expected_id + ".desired")
    gripper = _finite(payload.get("future_gripper"), (SNAPSHOT_HORIZON, 1), expected_id + ".gripper")
    observation_timestamp = float(payload.get("observation_timestamp"))
    _require(np.isfinite(observation_timestamp), "observation timestamp is nonfinite: %s" % expected_id)
    _require(abs(float(timestamps[0]) - observation_timestamp) <= 1e-7, "label timestamp does not start at observation: %s" % expected_id)
    _require(np.allclose(np.diff(timestamps), ACTION_DT_S, rtol=0.0, atol=1e-8), "label timestamp spacing drift: %s" % expected_id)
    _require(np.max(np.abs(desired[0, :3] - state[:3])) <= 1e-7, "label does not bind current EEF position: %s" % expected_id)
    _require(np.max(np.abs(desired[0, 3:9] - state[3:9])) <= 1e-7, "label does not bind current EEF orientation: %s" % expected_id)
    body_path = Path(str(payload.get("body_frame_path"))).resolve(strict=True)
    wrist_path = Path(str(payload.get("wrist_frame_path"))).resolve(strict=True)
    body = _ppm_rgb(body_path); wrist = _ppm_rgb(wrist_path)
    # Stage22 provenance binds the encoded P6 files, while the model runner
    # separately records hashes of the decoded uint8 arrays.  Keep those
    # domains distinct so an encoding/header change cannot be mistaken for an
    # identical model-facing tensor.
    body_file_sha = sha256_file(body_path); wrist_file_sha = sha256_file(wrist_path)
    body_array_sha = sha256_array(body); wrist_array_sha = sha256_array(wrist)
    _require(body_file_sha == provenance.get("body_frame_sha256"), "body frame file hash mismatch: %s" % expected_id)
    _require(wrist_file_sha == provenance.get("wrist_frame_sha256"), "wrist frame file hash mismatch: %s" % expected_id)
    dt = np.diff(timestamps)
    displacement = np.diff(desired[:, :3], axis=0)
    speed = np.linalg.norm(displacement, axis=1) / dt
    path_length = float(np.sum(np.linalg.norm(displacement, axis=1)))
    label_data = {
        "timestamps": timestamps.tolist(),
        "desired": desired.tolist(),
        "gripper": gripper.tolist(),
    }
    return {
        "example_id": expected_id,
        "target_id": target,
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": sha256_file(snapshot_path),
        "observation_timestamp_s": observation_timestamp,
        "body_frame_path": str(body_path),
        "wrist_frame_path": str(wrist_path),
        "body_frame_sha256": body_file_sha,
        "wrist_frame_sha256": wrist_file_sha,
        "body_array_sha256": body_array_sha,
        "wrist_array_sha256": wrist_array_sha,
        "body_red_pixel_count": int(np.count_nonzero((body[..., 0] > 100) & (body[..., 0] > body[..., 1] * 1.35) & (body[..., 0] > body[..., 2] * 1.35))),
        "wrist_red_pixel_count": int(np.count_nonzero((wrist[..., 0] > 100) & (wrist[..., 0] > wrist[..., 1] * 1.35) & (wrist[..., 0] > wrist[..., 2] * 1.35))),
        "state_v1": state.tolist(),
        "state_v1_sha256": sha256_array(state),
        "target_position_world_m": list(TRAIN_TARGETS[target]),
        "label_sha256": sha256_array(np.concatenate([desired, gripper], axis=1)),
        "label_initial_position_max_abs_error_m": float(np.max(np.abs(desired[0, :3] - state[:3]))),
        "label_initial_rotation_max_abs_error": float(np.max(np.abs(desired[0, 3:9] - state[3:9]))),
        "label_path_length_m": path_length,
        "label_speed_mps": _stats(speed.tolist()),
        "label_endpoint_local_m": desired[-1, :3].tolist(),
        "label_data": label_data,
        "target_world_is_metadata_only": True,
    }


def audit_a1(a0: Mapping[str, Any]) -> Dict[str, Any]:
    """A1: audit all 24 TRAIN expert labels without a model call."""

    rows: List[Dict[str, Any]] = []
    for target in TRAIN_TARGET_ORDER:
        for index in range(6):
            rows.append(_audit_snapshot(Path(_snapshot_path(Path(a0["artifacts"]["expert_root"]), target, index)), target, index))
    _require(len(rows) == SNAPSHOT_COUNT, "A1 snapshot count mismatch")
    _require(all(row["target_world_is_metadata_only"] for row in rows), "target metadata entered label/model boundary")
    return {
        "schema": "CORELAB_STAGE44_A1_EXPERT_LABEL_AUDIT_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44_A1_24_EXPERT_LABELS_PASS",
        "snapshot_count": len(rows),
        "target_count": len(TRAIN_TARGET_ORDER),
        "target_ids": list(TRAIN_TARGET_ORDER),
        "examples_per_target": {target: sum(row["target_id"] == target for row in rows) for target in TRAIN_TARGET_ORDER},
        "horizon": SNAPSHOT_HORIZON,
        "action_sample_period_s": ACTION_DT_S,
        "language": LANGUAGE,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "state_abi": "StateV1=EEF_XYZ_3+GR00T_FIRST_TWO_ROWS_ROT6D_6+GRIPPER_1",
        "body_frame_shape": [96, 96, 3],
        "wrist_frame_shape": [96, 96, 3],
        "label_shape": [40, 10],
        "initial_label_position_error_m": _stats([row["label_initial_position_max_abs_error_m"] for row in rows]),
        "initial_label_rotation_error": _stats([row["label_initial_rotation_max_abs_error"] for row in rows]),
        "label_path_length_m": _stats([row["label_path_length_m"] for row in rows]),
        "label_max_speed_mps": _stats([row["label_speed_mps"]["max"] for row in rows]),
        "unique_body_frames": len({row["body_array_sha256"] for row in rows}),
        "unique_wrist_frames": len({row["wrist_array_sha256"] for row in rows}),
        "unique_state_v1": len({row["state_v1_sha256"] for row in rows}),
        "rows": rows,
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": "24 accepted TRAIN expert labels and frame/state integrity; no learned-policy or task-success claim",
    }


def _source_marker_lines(path: Path, patterns: Sequence[str]) -> Dict[str, List[int]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {pattern: [index + 1 for index, line in enumerate(text) if re.search(pattern, line)] for pattern in patterns}


def audit_a2(repo: Path, stage43: Mapping[str, Any]) -> Dict[str, Any]:
    """A2: attribute Stage43 references to actual GR00T outputs."""

    root = Path(stage43["root"])
    result = stage43["result"]
    bundle = stage43["bundle"]
    conditions = result.get("conditions", [])
    _require(len(conditions) == 5, "Stage43 condition count drift")
    rows: List[Dict[str, Any]] = []
    model_refs = 0
    expert_refs = 0
    for condition in conditions:
        condition_id = str(condition.get("condition_id"))
        croot = root / "conditions" / condition_id
        state_payload = read_json(croot / "state_v1.json", canonical=True)
        observations = state_payload.get("observations", [])
        responses_payload = read_json(croot / "worker_response.json", canonical=True)
        responses = responses_payload.get("responses", [])
        _require(len(observations) == 4 and len(responses) == 4, "Stage43 reference count drift: %s" % condition_id)
        qrows: List[Dict[str, Any]] = []
        for index, obs in enumerate(observations):
            qid = "%s-q%d" % (condition_id, index)
            _require(obs.get("request_id") == qid and obs.get("response", {}).get("status") == "PASS", "Stage43 observation is not a model result: %s" % qid)
            response = responses[index]
            _require(response.get("request_id") == qid and response.get("status") == "PASS", "Stage43 worker result is not PASS: %s" % qid)
            _require(response.get("action_contract") == "STRUCTURED_ACTION_V1" and response.get("decode_contract") == "STRUCTURED_ACTION_V1_DEFERRED_TO_NATIVE", "Stage43 action attribution drift: %s" % qid)
            _require(obs.get("raw_sha256") == response.get("raw_sha256") and response.get("worker_forward_count_delta") == 1, "Stage43 raw action is not bound to one GR00T forward: %s" % qid)
            qrows.append({
                "request_id": qid,
                "reference_source": "GR00T_FORWARD_OUTPUT",
                "raw_action_sha256": response.get("raw_sha256"),
                "worker_forward_count_delta": response.get("worker_forward_count_delta"),
                "action_contract": response.get("action_contract"),
                "target_in_model_forward": False,
            })
            model_refs += 1
        bind = read_json(croot / "initial_bind.json", canonical=True)
        handoff = read_json(croot / "handoff.json", canonical=True)
        _require(len(bind.get("events", [])) == 1 and len(handoff.get("events", [])) == 3, "Stage43 binding/update count drift: %s" % condition_id)
        _require(bind["events"][0].get("chunk_id") == "%s-q0" % condition_id, "Stage43 initial reference source drift: %s" % condition_id)
        _require([item.get("chunk_id") for item in handoff["events"]] == ["%s-q%d" % (condition_id, i) for i in (1, 2, 3)], "Stage43 handoff source drift: %s" % condition_id)
        _require(all(item.get("reference_only") is True and item.get("reference_update_delta_observed") == 1 for item in handoff["events"]), "Stage43 reference-only update drift: %s" % condition_id)
        rows.append({
            "condition_id": condition_id,
            "gr00t_forward_count": condition.get("gr00t_forward_count"),
            "reference_update_count": condition.get("reference_update_count"),
            "execution_integrity": condition.get("execution_integrity"),
            "references": qrows,
            "reference_updates_are_reference_only": True,
            "expert_label_reference_count": 0,
        })
    # The Stage43 frozen bundle has no expert-label artifact and binds each
    # reference through the worker's raw StructuredActionV1 response.
    _require("expert_label" not in json.dumps(bundle.get("artifacts", {}), sort_keys=True).lower(), "Stage43 bundle unexpectedly names an expert reference artifact")
    _require(model_refs == 20 and expert_refs == 0, "Stage43 reference attribution total drift")
    source_paths = [
        repo / "scripts" / "stage21_same_scene_groot_inference.py",
        repo / "scripts" / "stage21_same_scene_groot_worker.py",
        repo / "scripts" / "stage21_native_model_receding_horizon.py",
        repo / "scripts" / "stage43_corelab_initial_condition.py",
    ]
    for path in source_paths:
        _require(path.is_file(), "Stage43 attribution source missing: %s" % path)
    return {
        "schema": "CORELAB_STAGE44_A2_STAGE43_REFERENCE_ATTRIBUTION_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44_A2_STAGE43_REFERENCE_ATTRIBUTION_PASS",
        "stage43_predecessor": {"result_sha256": stage43["result_sha256"], "bundle_sha256": stage43["bundle_sha256"], "seal_sha256": stage43["seal_sha256"]},
        "condition_count": len(rows),
        "model_reference_count": model_refs,
        "expert_label_reference_count": expert_refs,
        "reference_source": "actual GR00T worker StructuredActionV1 raw output",
        "reference_update_semantics": "initial bind plus reference-only q1/q2/q3 updates; no expert label trajectory substituted",
        "conditions": rows,
        "source_files": [{"path": str(path.resolve()), "sha256": sha256_file(path), "markers": _source_marker_lines(path, (r"get_action", r"STRUCTURED_ACTION_V1", r"reference_only"))} for path in source_paths],
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": "Stage43 reference provenance only; not a task-success result",
    }


def prepare_bundle(repo: Path, a0_path: Path, a1_path: Path, a2_path: Path, output: Path) -> Dict[str, Any]:
    a0 = read_json(a0_path, canonical=True); a1 = read_json(a1_path, canonical=True); a2 = read_json(a2_path, canonical=True)
    _require(a0.get("status") == "PASS" and a1.get("status") == "PASS" and a2.get("status") == "PASS", "A0/A1/A2 must PASS before bundle freeze")
    stage43_bundle = read_json(Path(a0["artifacts"]["stage43_root"]) / "stage43_execution_bundle.json", canonical=True)
    artifacts = stage43_bundle.get("artifacts", {})
    required = {"stage10_root", "base_model_path", "structured_checkpoint", "structured_stats", "tiny1_dataset", "modality", "cosmos_snapshot"}
    _require(required.issubset(artifacts), "Stage43 bundle lacks model runtime paths")
    runner = repo / "scripts" / "stage44_corelab_train_forwards.py"
    _require(runner.is_file(), "Stage44 A3 runner is missing")
    bundle = {
        "schema": "CORELAB_STAGE44_EXECUTION_BUNDLE_V1",
        "stage_id": STAGE_ID,
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "repository": {"branch": BRANCH, "head": _git_head(repo)},
        "predecessor": {
            "stage43_head": STAGE43_HEAD,
            "stage43_bundle_sha256": STAGE43_BUNDLE_SHA256,
            "stage43_result_sha256": STAGE43_RESULT_SHA256,
            "stage43_seal_sha256": STAGE43_SEAL_SHA256,
            "a0_sha256": sha256_file(a0_path),
            "a1_sha256": sha256_file(a1_path),
            "a2_sha256": sha256_file(a2_path),
        },
        "artifacts": {
            "stage10_root": artifacts["stage10_root"],
            "base_model_path": artifacts["base_model_path"],
            "structured_checkpoint": artifacts["structured_checkpoint"],
            "structured_stats": artifacts["structured_stats"],
            "tiny1_dataset": artifacts["tiny1_dataset"],
            "modality": artifacts["modality"],
            "cosmos_snapshot": artifacts["cosmos_snapshot"],
            "expert_root": a0["artifacts"]["expert_root"],
            "train_dataset": a0["artifacts"]["train_dataset"],
        },
        "train_only_inputs": {"snapshot_count": SNAPSHOT_COUNT, "target_ids": list(TRAIN_TARGET_ORDER), "expected_ids": _expected_snapshot_ids(), "language": LANGUAGE, "seed": DEV_SEED, "model_input_keys": list(MODEL_INPUT_KEYS), "target_xyz_in_model_input": False},
        "budgets": {"a3_gr00t_forward_count": SNAPSHOT_COUNT, "wbc_forward_count": 0, "reference_update_count": 0, "training_steps": 0, "backward_steps": 0, "optimizer_steps": 0},
        "execution_class": "CORELAB_STAGE44_TRAIN_ONLY_GR00T_FORWARD_COHORT_V1",
        "runner": {"relative_path": "scripts/stage44_corelab_train_forwards.py", "sha256": sha256_file(runner)},
        "forbidden_operations": forbidden_operations(),
        "classification": "NON_SCIENTIFIC_TRAINING_LINEAGE_AND_BEHAVIOR_DIAGNOSTIC",
    }
    write_json(output, bundle)
    return bundle


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _delta_stats(delta: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(delta, dtype=np.float64)
    _require(bool(np.isfinite(value).all()), "delta contains nonfinite values")
    flat = value.reshape(-1)
    return {"l2": float(np.linalg.norm(flat)), "max_abs": float(np.max(np.abs(flat))), "mean_abs": float(np.mean(np.abs(flat))), "rms": float(np.sqrt(np.mean(flat * flat)))}


def _load_a3_artifact(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require(path.is_file() and not path.is_symlink(), "A3 artifact is missing: %s" % path)
    with np.load(path, allow_pickle=False) as data:
        _require(set(data.files) == {"raw_universal_action", "decoded_physical_action", "input_state", "body_rgb", "wrist_rgb"}, "A3 artifact input/output key drift")
        raw = _finite(data["raw_universal_action"], (1, 40, 132), "A3 raw", np.float32)
        decoded = _finite(data["decoded_physical_action"], (40, 10), "A3 decoded", np.float64)
        state = _finite(data["input_state"], (10,), "A3 state", np.float32)
        body = _finite(data["body_rgb"], (96, 96, 3), "A3 body", np.uint8)
        wrist = _finite(data["wrist_rgb"], (96, 96, 3), "A3 wrist", np.uint8)
    return raw, decoded, state, body, wrist


def audit_a4_a5(repo: Path, pre_root: Path, a3_root: Path, output_root: Path) -> Dict[str, Any]:
    """A4 coverage census and A5 causal diagnosis after the 24 forwards."""

    pre_root = Path(pre_root).resolve(strict=True)
    a3_root = Path(a3_root).resolve(strict=True)
    a0 = read_json(pre_root / "a0_training_lineage.json", canonical=True)
    a1 = read_json(pre_root / "a1_expert_label_audit.json", canonical=True)
    a2 = read_json(pre_root / "a2_stage43_reference_attribution.json", canonical=True)
    bundle = read_json(pre_root / "stage44_execution_bundle.json", canonical=True)
    a3_manifest_path = a3_root / "a3_train_forwards_manifest.json"
    a3 = read_json(a3_manifest_path, canonical=True)
    _require(a3.get("status") == "PASS" and a3.get("gr00t_forward_count_observed") == SNAPSHOT_COUNT, "A3 is not an exact 24-forward PASS")
    _require(a3.get("bundle_sha256") == sha256_file(pre_root / "stage44_execution_bundle.json"), "A3 bundle hash mismatch")
    _require(a3.get("target_xyz_in_model_input") is False and a3.get("model_input_keys") == list(MODEL_INPUT_KEYS), "A3 target/input boundary drift")
    rows_by_id = {row["example_id"]: row for row in a1["rows"]}
    forward_rows = a3.get("rows", [])
    _require(len(forward_rows) == SNAPSHOT_COUNT, "A3 row count mismatch")
    model_rows: List[Dict[str, Any]] = []
    raw_hashes: List[str] = []; decoded_hashes: List[str] = []
    body_hashes: List[str] = []; wrist_hashes: List[str] = []; state_hashes: List[str] = []
    errors: List[float] = []; output_vs_label: List[Dict[str, Any]] = []
    for item in forward_rows:
        example_id = str(item.get("example_id"))
        _require(example_id in rows_by_id, "A3 example is not a sealed TRAIN label: %s" % example_id)
        _require(item.get("status") == "PASS" and item.get("forward_count_delta_observed") == 1, "A3 forward counter drift: %s" % example_id)
        _require(item.get("target_xyz_in_model_input") is False and item.get("model_input_keys") == list(MODEL_INPUT_KEYS), "A3 target boundary drift: %s" % example_id)
        artifact = Path(str(item.get("artifact_path"))).resolve(strict=True)
        raw, decoded, state, body, wrist = _load_a3_artifact(artifact)
        _require(sha256_file(artifact) == item.get("artifact_sha256"), "A3 artifact hash mismatch: %s" % example_id)
        _require(sha256_array(raw) == item.get("raw_action_sha256") and sha256_array(decoded) == item.get("decoded_action_sha256"), "A3 action hash mismatch: %s" % example_id)
        _require(sha256_array(state) == item.get("state_v1_sha256") and sha256_array(body) == item.get("body_sha256") and sha256_array(wrist) == item.get("wrist_sha256"), "A3 input hash mismatch: %s" % example_id)
        label = np.asarray(rows_by_id[example_id]["label_data"]["desired"], dtype=np.float64)
        label_full = np.concatenate([label, np.asarray(rows_by_id[example_id]["label_data"]["gripper"], dtype=np.float64)], axis=1)
        delta = decoded - label_full
        metrics = {
            "eef_position": _delta_stats(delta[:, :3]),
            "eef_rot6d": _delta_stats(delta[:, 3:9]),
            "gripper": _delta_stats(delta[:, 9:10]),
            "full_physical": _delta_stats(delta),
            "prediction_endpoint_vs_label_m": float(np.linalg.norm(decoded[-1, :3] - label_full[-1, :3])),
            "prediction_path_length_m": float(np.sum(np.linalg.norm(np.diff(decoded[:, :3], axis=0), axis=1))),
            "label_path_length_m": float(np.sum(np.linalg.norm(np.diff(label_full[:, :3], axis=0), axis=1))),
        }
        model_rows.append({
            "example_id": example_id,
            "target_id": rows_by_id[example_id]["target_id"],
            "artifact_path": str(artifact),
            "raw_action_sha256": item["raw_action_sha256"],
            "decoded_action_sha256": item["decoded_action_sha256"],
            "input_state_sha256": item["state_v1_sha256"],
            "metrics_vs_expert_label": metrics,
            "finite": True,
        })
        raw_hashes.append(item["raw_action_sha256"]); decoded_hashes.append(item["decoded_action_sha256"])
        body_hashes.append(item["body_sha256"]); wrist_hashes.append(item["wrist_sha256"]); state_hashes.append(item["state_v1_sha256"])
        errors.append(metrics["full_physical"]["rms"])
        output_vs_label.append(metrics)

    coverage = {
        "schema": "CORELAB_STAGE44_A4_DATA_COVERAGE_CENSUS_V1",
        "status": "PASS",
        "train_dataset_tree_sha256": a0["train_dataset_tree"]["tree_sha256"],
        "train_dataset_total_episodes": 24,
        "train_dataset_total_frames": 960,
        "snapshot_label_count": SNAPSHOT_COUNT,
        "unique_train_target_ids": list(TRAIN_TARGET_ORDER),
        "unique_train_target_count": len(TRAIN_TARGET_ORDER),
        "target_positions_world_m": TRAIN_TARGETS,
        "examples_per_target": a1["examples_per_target"],
        "unique_body_frames": len(set(body_hashes)),
        "unique_wrist_frames": len(set(wrist_hashes)),
        "unique_state_v1": len(set(state_hashes)),
        "unique_model_raw_actions": len(set(raw_hashes)),
        "unique_model_decoded_actions": len(set(decoded_hashes)),
        "snapshot_expansion": {"materialized_frames": 960, "source_snapshots": 24, "frames_per_snapshot": 40, "same_observation_repeated_within_materialized_window": True},
        "coverage_limits": [
            "one language task",
            "four TRAIN target coordinates",
            "no validation/final-test episodes",
            "snapshot-expanded windows rather than 24 independent continuous camera trajectories",
        ],
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": "coverage census only; not generalization or task-success evidence",
    }
    write_json(output_root / "a4_data_coverage_census.json", coverage)

    stage43_result = read_json(Path(a0["artifacts"]["stage43_root"]) / "stage43_result.json", canonical=True)
    progress = [item.get("progress_sign") for item in stage43_result.get("conditions", [])]
    stage43_metrics = [{"condition_id": item.get("condition_id"), "progress_sign": item.get("progress_sign"), "absolute_progress_m": item.get("absolute_progress_m"), "final_target_error_m": item.get("final_target_error_m")} for item in stage43_result.get("conditions", [])]
    diagnosis = {
        "schema": "CORELAB_STAGE44_A5_CAUSAL_DIAGNOSIS_V1",
        "status": "PASS",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "token": "CORELAB_STAGE44_A5_CAUSAL_DIAGNOSIS_PASS",
        "a3_runtime": {"forward_count": a3.get("gr00t_forward_count_observed"), "all_finite": a3.get("all_outputs_finite"), "target_xyz_in_model_input": a3.get("target_xyz_in_model_input"), "model_input_keys": a3.get("model_input_keys"), "seed": a3.get("seed")},
        "prediction_vs_expert": {"full_physical_rms_mixed_units": _stats(errors), "position_rms_m": _stats([item["eef_position"]["rms"] for item in output_vs_label]), "position_max_abs_m": _stats([item["eef_position"]["max_abs"] for item in output_vs_label]), "endpoint_error_m": _stats([item["prediction_endpoint_vs_label_m"] for item in output_vs_label]), "per_example": model_rows},
        "output_diversity": {"unique_raw_actions": len(set(raw_hashes)), "unique_decoded_actions": len(set(decoded_hashes)), "unique_states": len(set(state_hashes)), "unique_body_frames": len(set(body_hashes)), "unique_wrist_frames": len(set(wrist_hashes)), "classification": "OUTPUT_DIVERSITY_PRESENT" if len(set(decoded_hashes)) > 1 else "OUTPUTS_BYTE_IDENTICAL"},
        "stage43_task_behavior_context": {"condition_count": len(progress), "progress_signs": progress, "all_progress_negative": bool(progress) and all(value == "NEGATIVE" for value in progress), "conditions": stage43_metrics, "interpretation": "Stage43 endpoint completion is reference completion; all five actual EEF target-progress signs were NEGATIVE."},
        "causal_interpretation": {
            "model_uses_only_declared_modalities": True,
            "model_output_changes_across_train_observations": len(set(decoded_hashes)) > 1,
            "current_evidence_proves_task_success": False,
            "current_evidence_supports_data_expansion": True,
            "current_evidence_supports_h200_fine_tuning_planning": True,
            "h200_fine_tuning_executed": False,
            "recommendation": "DATA_EXPANSION_BEFORE_H200_FINE_TUNING",
            "reason": "The accepted plumbing is exercised, but the current single-task/four-target TRAIN cohort is snapshot-expanded and Stage43 actual EEF target progress is negative across all five controlled initial conditions. Add broader independent trajectories/target coverage before authorizing a fine-tuning run; this stage does not choose a success threshold.",
        },
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": "bounded 24-sample TRAIN-only model behavior diagnosis; no generalization, task-success, or fine-tuning result",
    }
    write_json(output_root / "a5_causal_diagnosis.json", diagnosis)

    result = {
        "schema": "CORELAB_STAGE44_TRAINING_LINEAGE_BEHAVIOR_DIAGNOSIS_RESULT_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44_TRAINING_LINEAGE_AND_BEHAVIOR_DIAGNOSIS_CHARACTERIZED",
        "return_token": "GPT_PI_REVIEW_AFTER_STAGE44_TRAINING_AUDIT",
        "repository": {"branch": BRANCH, "head": _git_head(repo)},
        "predecessors": {"stage43_result_sha256": STAGE43_RESULT_SHA256, "stage44_bundle_sha256": sha256_file(pre_root / "stage44_execution_bundle.json"), "a0_sha256": sha256_file(pre_root / "a0_training_lineage.json"), "a1_sha256": sha256_file(pre_root / "a1_expert_label_audit.json"), "a2_sha256": sha256_file(pre_root / "a2_stage43_reference_attribution.json"), "a3_sha256": sha256_file(a3_manifest_path), "a4_sha256": sha256_file(output_root / "a4_data_coverage_census.json"), "a5_sha256": sha256_file(output_root / "a5_causal_diagnosis.json")},
        "observed_counters": {"train_snapshots": SNAPSHOT_COUNT, "gr00t_forward_count": int(a3.get("gr00t_forward_count_observed")), "wbc_forward_count": 0, "reference_update_count": 0, "training_steps": 0, "backward_steps": 0, "optimizer_steps": 0},
        "stages": {"A0": {"status": a0.get("status"), "token": a0.get("token")}, "A1": {"status": a1.get("status"), "token": a1.get("token")}, "A2": {"status": a2.get("status"), "token": a2.get("token")}, "A3": {"status": a3.get("status"), "forward_count": a3.get("gr00t_forward_count_observed")}, "A4": {"status": coverage.get("status"), "token": "CORELAB_STAGE44_A4_DATA_COVERAGE_PASS"}, "A5": {"status": diagnosis.get("status"), "token": diagnosis.get("token")}},
        "decision_context": diagnosis["causal_interpretation"],
        "stop_gate": {"status": "STOP_PI", "next_action": "PI_DECISION_DATA_EXPANSION_AND_H200_FINE_TUNING", "do_not_start": ["WBC", "A40/Vanda", "H200 fine-tuning", "validation/final-test reads", "success-threshold selection"]},
        "forbidden_operations": forbidden_operations(),
        "claim_boundary": diagnosis["claim_boundary"],
    }
    result_path = output_root / "stage44_result.json"
    write_json(result_path, result)
    seal = {
        "schema": "CORELAB_STAGE44_RESULT_SEAL_V1",
        "stage": STAGE_ID,
        "status": "PASS",
        "result_sha256": sha256_file(result_path),
        "bundle_sha256": sha256_file(pre_root / "stage44_execution_bundle.json"),
        "a0_sha256": sha256_file(pre_root / "a0_training_lineage.json"),
        "a1_sha256": sha256_file(pre_root / "a1_expert_label_audit.json"),
        "a2_sha256": sha256_file(pre_root / "a2_stage43_reference_attribution.json"),
        "a3_sha256": sha256_file(a3_manifest_path),
        "a4_sha256": sha256_file(output_root / "a4_data_coverage_census.json"),
        "a5_sha256": sha256_file(output_root / "a5_causal_diagnosis.json"),
    }
    write_json(output_root / "stage44_result.seal.json", seal)
    report = _render_report(result, a0, a1, a2, a3, coverage, diagnosis, seal)
    (output_root / "STAGE44_CORELAB_TRAINING_AUDIT.md").write_text(report, encoding="utf-8")
    return result


def _render_report(result: Mapping[str, Any], a0: Mapping[str, Any], a1: Mapping[str, Any], a2: Mapping[str, Any], a3: Mapping[str, Any], a4: Mapping[str, Any], a5: Mapping[str, Any], seal: Mapping[str, Any]) -> str:
    context = a5["causal_interpretation"]
    return "\n".join([
        "# Stage44 Corelab training lineage and behavior diagnosis",
        "",
        "Status: `%s`" % result["token"],
        "",
        "This is a bounded A0–A5 audit. It performs no WBC execution, no A40 wait, no validation/final-test read, no backward pass, and no fine-tuning.",
        "",
        "## Evidence chain",
        "",
        "- Stage43 result: `%s`; bundle: `%s`; result seal: `%s`." % (a0["predecessor"]["stage43_result_sha256"], a0["predecessor"]["stage43_bundle_sha256"], a0["predecessor"]["stage43_result_seal_sha256"]),
        "- A0/A1/A2: PASS; Stage43 references attributed to `%s` (%d model forwards, %d expert-label references)." % (a2["reference_source"], a2["model_reference_count"], a2["expert_label_reference_count"]),
        "- A3: `%s` with observed GR00T forwards `%s`; model input keys `%s`; target XYZ in model input `%s`." % (a3["status"], a3["gr00t_forward_count_observed"], ", ".join(a3["model_input_keys"]), a3["target_xyz_in_model_input"]),
        "- A4/A5: PASS; result seal binds `%s`." % seal["result_sha256"],
        "",
        "## TRAIN coverage",
        "",
        "- 24 expert snapshots, four target IDs, six snapshots per target; materialized TRAIN fixture: 24 episodes / 960 frames.",
        "- Unique body/wrist/state inputs: %d / %d / %d; unique decoded model outputs: %d." % (a4["unique_body_frames"], a4["unique_wrist_frames"], a4["unique_state_v1"], a4["unique_model_decoded_actions"]),
        "- This remains one language task and snapshot-expanded windows; no validation or final-test examples were read.",
        "",
        "## Diagnosis",
        "",
        "- Stage43 actual EEF target-progress signs: `%s` (all negative: `%s`)." % (", ".join(a5["stage43_task_behavior_context"]["progress_signs"]), a5["stage43_task_behavior_context"]["all_progress_negative"]),
        "- Model output diversity across the 24 TRAIN observations: `%s`; prediction-vs-label metrics remain continuous characterization, without a chosen success threshold." % a5["output_diversity"]["classification"],
        "- Recommendation: `%s`." % context["recommendation"],
        "- H200 fine-tuning executed: `%s`; this package requests PI direction only." % context["h200_fine_tuning_executed"],
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("audit-pre")
    pre.add_argument("--repo", type=Path, required=True)
    pre.add_argument("--stage43-root", type=Path, required=True)
    pre.add_argument("--expert-root", type=Path, required=True)
    pre.add_argument("--train-dataset", type=Path, required=True)
    pre.add_argument("--output-root", type=Path, required=True)
    post = sub.add_parser("audit-post")
    post.add_argument("--repo", type=Path, required=True)
    post.add_argument("--pre-root", type=Path, required=True)
    post.add_argument("--a3-root", type=Path, required=True)
    post.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-pre":
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        a0 = audit_a0(args.repo.resolve(strict=True), args.stage43_root, args.expert_root, args.train_dataset)
        a0_path = output_root / "a0_training_lineage.json"; write_json(a0_path, a0)
        a1 = audit_a1(a0)
        a1_path = output_root / "a1_expert_label_audit.json"; write_json(a1_path, a1)
        a2 = audit_a2(args.repo.resolve(strict=True), _stage43_predecessor(args.stage43_root))
        a2_path = output_root / "a2_stage43_reference_attribution.json"; write_json(a2_path, a2)
        bundle = prepare_bundle(args.repo.resolve(strict=True), a0_path, a1_path, a2_path, output_root / "stage44_execution_bundle.json")
        print(json.dumps({"stage": STAGE_ID, "status": "PASS", "A0": a0["token"], "A1": a1["token"], "A2": a2["token"], "bundle_sha256": sha256_file(output_root / "stage44_execution_bundle.json")}, sort_keys=True))
        return 0
    result = audit_a4_a5(args.repo.resolve(strict=True), args.pre_root, args.a3_root, args.output_root)
    print(json.dumps({"stage": STAGE_ID, "status": result["status"], "token": result["token"], "result_sha256": sha256_file(Path(args.output_root) / "stage44_result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
