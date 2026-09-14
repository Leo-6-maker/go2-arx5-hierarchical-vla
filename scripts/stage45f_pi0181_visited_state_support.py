#!/usr/bin/env python3
"""PI0181 CPU-only visited-state support and historical replay preparation.

This module consumes only the accepted, sealed Stage43 evidence bundle and the
sealed TRAIN-only FULL24 parquet dataset.  It does not import a policy runtime,
load a checkpoint, contact a GPU, run a simulator/WBC, or perform an optimizer
step.  The Stage22 expert implementation is used as a deterministic offline
relabeler for the exact observed EEF state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from go2arx5_vla.simulation import SameSceneExpertConfigV1, build_same_scene_expert_chunk, write_rgb_ppm
from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows


PI_GATE = 5664712814
PR_POINTER = 5664717125
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"

STAGE43_ACCEPTED_HEAD = "b2a70dcec0765cf69819bb76c180e5c92ae3eac3"
STAGE43_RESULT_SHA256 = "e635ecaa321bde38903be45781e9b47063ba6b81c2bc810b1a1ae64925ecfcdb"
STAGE43_SEAL_SHA256 = "668bee50dc38997403a065d5f4831662731715cf28c289af3982e4f9abbf7283"
STAGE43_COHORT_MANIFEST_SHA256 = "62e95728ea6da7a0d90613e52adabc0737f3b116f63d4f01ebb680387ea2de48"
STAGE43_BUNDLE_SHA256 = "127d95a2a5b9a6a40e864e029d85fd09a9e4dd55441aae58071502aa76700da6"

STAGE43_ROOT_DEFAULT = Path("/var/cache/users/liuyu-nfs/stage43_corelab_controlled_cohort_20260908")
FULL24_ROOT_DEFAULT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/native_expert/lerobot_train")

TARGET_WORLD = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
TARGET_ID = "red_target_train_a"
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"
STAGE43_SEED = 4242
TRAIN_SEED = 6164
UPDATE_COUNT = 256
FULL24_SLOT_COUNT = 24
VISITED_ROW_COUNT = 20
HORIZON = 40
STATE_DIM = 10
ACTION_DIM = 10

CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)

CONDITION_SEALS = {
    "C0_NOMINAL": {
        "manifest": "1ff3075637e8c54cdbeff4ca9d892949b0f68f0b474815b26ce8afe6e51f619a",
        "result": "bd826d4e8c92e357201efa39c556b513fbb378b76b9dedfb25ca8f2850eece78",
        "seal": "d4287e903b74ed2c677f103a559d30cced7a1b4b91f4046d06069d849a30e013",
    },
    "C1_X_PLUS_5CM": {
        "manifest": "8489736dce236c472a46baad5bfecc5db4a33a8e4f2364508a677fc5a29346d2",
        "result": "b4f5ff0590801e2011dd01c0d37e8f21774210d9d652996051aa89d55f86e57a",
        "seal": "ddbabd76d87ce4aca50a2e955e3819a40b36e549e68148d94573021bc0316e4c",
    },
    "C2_X_MINUS_5CM": {
        "manifest": "ad2197a887828487eb1c93fcbde8f9fc3abe789276e6e8e8591b636aeb1c36a7",
        "result": "61d6d2ee9e3bd640f1403d9b9987fc7a4eb09ab2b51921c96e91c4d46a38093c",
        "seal": "ac01335f48ee792f2dadf40c4170a3547dab0d3fd0207bfee6d0c820a11d7212",
    },
    "C3_Y_PLUS_5CM": {
        "manifest": "6598c2b6c927aed464190e84db86c3c8c91f4432c53b7d204d9e14876689784a",
        "result": "d09c34333c99f2d31f475b15a60c7414370783d6604d164ea3d30920214e957b",
        "seal": "8a871e9e03b7873aab723e0428ecf7e4b6d25d1bbc62326218d7d772127e70ca",
    },
    "C4_Y_MINUS_5CM": {
        "manifest": "789e113fbf10e860fbdd11430f1895f1afee48737fab9e78d5e044118438112a",
        "result": "635a64aaf9c8af5cc40f2bb28cd097772521f33175faabea95a80bf150baf403",
        "seal": "3587faa6bb3b52934eb3824631cfef8ba671c56d140d2a6df2ec9dd8ecf7390a",
    },
}


class PI0181Error(RuntimeError):
    """Fail-closed error for a provenance or contract mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0181Error(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"required JSON is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = root.resolve()
    require(root.is_dir() and not root.is_symlink(), f"tree root is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            require(not child.is_symlink() and child.is_dir(), f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            require(not child.is_symlink() and child.is_file(), f"unsafe file entry: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_sha256(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    entries = regular_tree_entries(root, prefix)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest(), entries


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    require(array.shape == shape, f"{label} shape {array.shape} != {shape}")
    require(np.isfinite(array).all(), f"{label} contains non-finite values")
    return array


def verify_stage43(root: Path) -> dict[str, Any]:
    """Verify the accepted Stage43 seal and all five exact condition payloads."""

    root = root.resolve()
    require(root.is_dir() and not root.is_symlink(), f"Stage43 root is invalid: {root}")
    result_path = root / "stage43_result.json"
    seal_path = root / "stage43_result.seal.json"
    cohort_path = root / "stage43_cohort_manifest.json"
    bundle_path = root / "stage43_execution_bundle.json"
    require(sha256_file(result_path) == STAGE43_RESULT_SHA256, "Stage43 result SHA256 mismatch")
    require(sha256_file(seal_path) == STAGE43_SEAL_SHA256, "Stage43 seal SHA256 mismatch")
    require(sha256_file(cohort_path) == STAGE43_COHORT_MANIFEST_SHA256, "Stage43 cohort manifest SHA256 mismatch")
    result = read_json(result_path)
    seal = read_json(seal_path)
    cohort = read_json(cohort_path)
    bundle = read_json(bundle_path)
    require(result.get("stage") == "Stage43" and result.get("condition_order") == list(CONDITION_ORDER), "Stage43 result identity/order mismatch")
    require(seal.get("schema") == "CORELAB_STAGE43_RESULT_SEAL_V1" and seal.get("status") == "PASS", "Stage43 seal is not PASS")
    require(cohort.get("schema") == "CORELAB_STAGE43_COHORT_EVIDENCE_MANIFEST_V1", "Stage43 cohort schema mismatch")
    require(cohort.get("condition_order") == list(CONDITION_ORDER), "Stage43 cohort order mismatch")
    require(seal.get("result_sha256") == STAGE43_RESULT_SHA256, "Stage43 seal result reference mismatch")
    require(seal.get("cohort_manifest_sha256") == STAGE43_COHORT_MANIFEST_SHA256, "Stage43 seal cohort reference mismatch")
    require(seal.get("bundle_sha256") == STAGE43_BUNDLE_SHA256, "Stage43 seal bundle reference mismatch")
    require(result.get("bundle_sha256") == STAGE43_BUNDLE_SHA256, "Stage43 result bundle reference mismatch")
    require(cohort.get("bundle_sha256") == STAGE43_BUNDLE_SHA256, "Stage43 cohort bundle reference mismatch")
    require(bundle.get("schema") or bundle.get("stage") or bundle.get("bundle_sha256"), "Stage43 execution bundle is empty/unrecognized")

    cohort_by_id = {item["condition_id"]: item for item in cohort.get("conditions", [])}
    require(tuple(cohort_by_id) == CONDITION_ORDER, "Stage43 cohort condition list/order mismatch")
    condition_audits: list[dict[str, Any]] = []
    for condition_id in CONDITION_ORDER:
        condition_root = root / "conditions" / condition_id
        require(condition_root.is_dir() and not condition_root.is_symlink(), f"accepted condition directory missing: {condition_root}")
        expected = CONDITION_SEALS[condition_id]
        condition_manifest_path = condition_root / "condition_manifest.json"
        condition_result_path = condition_root / "condition_result.json"
        condition_seal_path = condition_root / "condition_seal.json"
        observed_hashes = {
            "manifest": sha256_file(condition_manifest_path),
            "result": sha256_file(condition_result_path),
            "seal": sha256_file(condition_seal_path),
        }
        require(observed_hashes == expected, f"{condition_id} condition seal hashes mismatch: {observed_hashes}")
        require(cohort_by_id[condition_id]["condition_manifest_sha256"] == expected["manifest"], f"{condition_id} cohort manifest reference mismatch")
        require(cohort_by_id[condition_id]["condition_result_sha256"] == expected["result"], f"{condition_id} cohort result reference mismatch")
        require(cohort_by_id[condition_id]["condition_seal_sha256"] == expected["seal"], f"{condition_id} cohort seal reference mismatch")
        condition_manifest = read_json(condition_manifest_path)
        condition_result = read_json(condition_result_path)
        condition_seal = read_json(condition_seal_path)
        require(condition_manifest.get("schema") == "CORELAB_STAGE43_CONDITION_EVIDENCE_MANIFEST_V1", f"{condition_id} manifest schema mismatch")
        require(condition_result.get("condition_id") == condition_id and condition_result.get("cases") == 4, f"{condition_id} result identity/case count mismatch")
        require(condition_result.get("gr00t_forward_count") == 4, f"{condition_id} forward count mismatch")
        require(condition_seal.get("status") == "PASS" and condition_seal.get("stage") == "Stage43", f"{condition_id} seal status mismatch")
        forbidden = condition_result.get("forbidden_operations", {})
        for key in ("validation_e_read", "holdout_f_read", "training", "h200_used", "hardware_executed", "new_data_read"):
            require(forbidden.get(key) is False, f"{condition_id} forbidden operation is true: {key}")
        records = condition_manifest.get("records", [])
        require(len(records) == 22, f"{condition_id} record count drift: {len(records)}")
        for record in records:
            relative = record["relative_path"]
            path = condition_root / relative
            require(path.is_file() and not path.is_symlink(), f"{condition_id} manifest record missing: {relative}")
            require(sha256_file(path) == record["sha256"], f"{condition_id} manifest record hash mismatch: {relative}")
        observations = read_json(condition_root / "state_v1.json").get("observations", [])
        require(len(observations) == 4, f"{condition_id} observation row count is not four")
        expected_ids = [f"{condition_id}-q{index}" for index in range(4)]
        require([row.get("request_id") for row in observations] == expected_ids, f"{condition_id} observation order mismatch")
        condition_audits.append({
            "condition_id": condition_id,
            "condition_manifest_sha256": expected["manifest"],
            "condition_result_sha256": expected["result"],
            "condition_seal_sha256": expected["seal"],
            "observation_count": len(observations),
            "gr00t_forward_count": condition_result["gr00t_forward_count"],
            "forbidden_operations": forbidden,
        })
    return {
        "root": str(root),
        "accepted_head": STAGE43_ACCEPTED_HEAD,
        "result_sha256": STAGE43_RESULT_SHA256,
        "seal_sha256": STAGE43_SEAL_SHA256,
        "cohort_manifest_sha256": STAGE43_COHORT_MANIFEST_SHA256,
        "bundle_sha256": STAGE43_BUNDLE_SHA256,
        "condition_order": list(CONDITION_ORDER),
        "conditions": condition_audits,
        "rows_expected": VISITED_ROW_COUNT,
        "validation_e_read": False,
        "holdout_f_read": False,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
        "training": 0,
    }


def _load_npz_array(path: Path, key: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        require(key in archive.files, f"{label} missing key {key}")
        value = np.asarray(archive[key])
    return _finite_array(value, shape, label)


def _current_observation_pose(condition_root: Path, q: int, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup = read_json(condition_root / "condition_setup.json")
    executor = read_json(condition_root / "executor_bookkeeping.json")
    handoff = read_json(condition_root / "handoff.json")
    if q == 0:
        event = executor["events"][0]
        current_eef = np.asarray(setup["event"]["s0"]["eef_world_m"], dtype=np.float64)
        current_quat = np.asarray(event["handoff"]["current_achieved_orientation"], dtype=np.float64)
    else:
        event = handoff["events"][q - 1]
        update = event["executor_update"]
        current_eef = np.asarray(event["observation_state_after_segment"]["eef_world_m"], dtype=np.float64)
        current_quat = np.asarray(update["handoff"]["current_achieved_orientation"], dtype=np.float64)
    _finite_array(current_eef, (3,), f"{condition_root.name} q{q} current EEF")
    _finite_array(current_quat, (4,), f"{condition_root.name} q{q} quaternion")
    require(np.linalg.norm(current_quat) > 1e-10, f"{condition_root.name} q{q} quaternion is zero")
    anchor = current_eef - state[:3].astype(np.float64)
    return current_eef, current_quat, anchor


def _segment_source(condition_root: Path, q: int) -> tuple[Path, list[dict[str, Any]]]:
    filename = "rows_before_handoff.json" if q < 3 else "rows_after_handoff.json"
    segment_index = q if q < 3 else 0
    payload = read_json(condition_root / filename)
    segments = payload.get("segments", [])
    require(segment_index < len(segments), f"{condition_root.name} q{q} segment missing")
    rows = segments[segment_index]
    require(isinstance(rows, list) and rows, f"{condition_root.name} q{q} WBC row segment is empty")
    require(all(row.get("chunk_id") == f"{condition_root.name}-q{q}" for row in rows), f"{condition_root.name} q{q} WBC chunk provenance mismatch")
    return condition_root / filename, rows


def _wbc_summary(condition_root: Path, q: int) -> dict[str, Any]:
    source, rows = _segment_source(condition_root, q)
    achieved = _finite_array([row["achieved_pos"] for row in rows], (len(rows), 3), f"{condition_root.name} q{q} achieved WBC positions")
    desired = _finite_array([row["desired_pos"] for row in rows], (len(rows), 3), f"{condition_root.name} q{q} desired WBC positions")
    error = achieved - desired
    rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return {
        "source_file": source.name,
        "source_file_sha256": sha256_file(source),
        "segment_row_count": len(rows),
        "first_achieved_pos": achieved[0].tolist(),
        "last_achieved_pos": achieved[-1].tolist(),
        "tracking_rms_position_m": rms,
        "tracking_max_position_m": float(np.max(np.linalg.norm(error, axis=1))),
    }


def _expert_action(request_id: str, timestamp: float, state: np.ndarray, anchor: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    target_local = TARGET_WORLD - anchor
    chunk = build_same_scene_expert_chunk(
        chunk_id=request_id,
        observation_timestamp=float(timestamp),
        current_position=state[:3],
        target_position=target_local,
        current_quaternion=current_quat,
        config=SameSceneExpertConfigV1(),
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_ID,
    )
    rot6d = rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation))
    action = np.concatenate(
        [chunk.eef_position, rot6d, np.asarray(chunk.gripper).reshape(HORIZON, 1)], axis=1
    )
    return _finite_array(action, (HORIZON, ACTION_DIM), f"{request_id} expert action")


def extract_rows(stage43_root: Path, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}
    visited_dir = output_root / "visited_rows"
    expert_dir = output_root / "expert_labels"
    frame_dir = output_root / "source_frames"
    visited_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for condition_index, condition_id in enumerate(CONDITION_ORDER):
        condition_root = stage43_root / "conditions" / condition_id
        state_payload = read_json(condition_root / "state_v1.json")
        observations = state_payload["observations"]
        metrics = read_json(condition_root / "condition_metrics.json")["metrics"]
        initial_distance = float(metrics["initial_target_error_m"])
        for q, observation in enumerate(observations):
            request_id = f"{condition_id}-q{q}"
            require(observation["request_id"] == request_id, f"row id mismatch: {request_id}")
            state = _finite_array(observation["state_v1"], (STATE_DIM,), f"{request_id} StateV1").astype(np.float64)
            require(sha256_array(state.astype(np.float32)) == observation["state_v1_sha256"], f"{request_id} StateV1 hash mismatch")
            body = _load_npz_array(condition_root / "observation_body.npz", request_id, (96, 96, 3), f"{request_id} body RGB")
            wrist = _load_npz_array(condition_root / "observation_wrist.npz", request_id, (96, 96, 3), f"{request_id} wrist RGB")
            raw = _load_npz_array(condition_root / "raw_action_d132.npz", request_id, (1, HORIZON, 132), f"{request_id} raw action")
            decoded = _load_npz_array(condition_root / "decoded_action_d10_h40.npz", request_id, (HORIZON, ACTION_DIM), f"{request_id} decoded action")
            require(sha256_array(body) == observation["body_sha256"], f"{request_id} body hash mismatch")
            require(sha256_array(wrist) == observation["wrist_sha256"], f"{request_id} wrist hash mismatch")
            require(sha256_array(raw) == observation["raw_sha256"], f"{request_id} raw hash mismatch")
            require(sha256_array(decoded) == observation["decoded_sha256"], f"{request_id} decoded hash mismatch")
            current_eef, current_quat, anchor = _current_observation_pose(condition_root, q, state)
            require(np.allclose(anchor + state[:3], current_eef, atol=2e-6), f"{request_id} StateV1/EEF anchor mismatch")
            expert = _expert_action(request_id, float(observation["observation_timestamp"]), state, anchor, current_quat)
            wbc = _wbc_summary(condition_root, q)
            current_distance = float(np.linalg.norm(current_eef - TARGET_WORLD))
            progress = initial_distance - current_distance
            row_slug = f"{condition_index:02d}_{q:02d}_{condition_id}"
            body_path = frame_dir / f"{row_slug}_body.ppm"
            wrist_path = frame_dir / f"{row_slug}_wrist.ppm"
            body_sha = write_rgb_ppm(body_path, body)
            wrist_sha = write_rgb_ppm(wrist_path, wrist)
            visited_payload = {
                "body_rgb": body,
                "wrist_rgb": wrist,
                "state_v1": state.astype(np.float32),
                "historical_raw": raw.astype(np.float32),
                "historical_decoded": decoded.astype(np.float64),
                "current_eef_world_m": current_eef,
                "anchor_world_m": anchor,
                "target_world_m": TARGET_WORLD,
            }
            expert_payload = {
                "expert_action_structured_v1": expert.astype(np.float64),
                "expert_eef_world_xyz": (anchor[None, :] + expert[:, :3]).astype(np.float64),
                "source_timestamps": float(observation["observation_timestamp"]) + np.arange(HORIZON) * 0.05,
            }
            visited_path = visited_dir / f"{row_slug}.npz"
            expert_path = expert_dir / f"{row_slug}.npz"
            np.savez_compressed(visited_path, **visited_payload)
            np.savez_compressed(expert_path, **expert_payload)
            required_names = (
                "state_v1.json", "observation_body.npz", "observation_wrist.npz", "raw_action_d132.npz",
                "decoded_action_d10_h40.npz", "condition_setup.json", "executor_bookkeeping.json",
                "handoff.json", "condition_metrics.json", "safety_gate.json",
            )
            for name in required_names:
                source_files[f"{condition_id}/{name}"] = sha256_file(condition_root / name)
            rows.append({
                "row_id": request_id,
                "row_slug": row_slug,
                "condition_id": condition_id,
                "condition_order_index": condition_index,
                "q_index": q,
                "observation_timestamp_s": float(observation["observation_timestamp"]),
                "target_id": TARGET_ID,
                "target_world_m": TARGET_WORLD.tolist(),
                "language": LANGUAGE,
                "model_input_fields": ["body_rgb", "wrist_rgb", "state_v1", "language"],
                "model_input_excludes": ["target_world_m", "expert_action_structured_v1", "historical_raw", "historical_decoded", "target_distance_m", "progress_m"],
                "state_v1": state.tolist(),
                "state_v1_sha256": observation["state_v1_sha256"],
                "body_shape": list(body.shape),
                "body_dtype": str(body.dtype),
                "body_sha256": observation["body_sha256"],
                "wrist_shape": list(wrist.shape),
                "wrist_dtype": str(wrist.dtype),
                "wrist_sha256": observation["wrist_sha256"],
                "raw_shape": list(raw.shape),
                "raw_sha256": observation["raw_sha256"],
                "decoded_shape": list(decoded.shape),
                "decoded_sha256": observation["decoded_sha256"],
                "expert_action_shape": list(expert.shape),
                "expert_action_sha256": sha256_array(expert),
                "expert_provenance": {
                    "source": "STAGE22_SAME_SCENE_EXPERT",
                    "target_world_m": TARGET_WORLD.tolist(),
                    "start_current_eef_world_m": current_eef.tolist(),
                    "start_current_eef_local_m": state[:3].tolist(),
                    "anchor_world_m": anchor.tolist(),
                    "source_frame": SOURCE_FRAME,
                    "tcp_semantic_id": TCP_ID,
                    "position_law": "smoothstep",
                    "orientation_law": "constant_current_orientation",
                    "horizon": HORIZON,
                    "dt_s": 0.05,
                    "duration_s": 1.95,
                    "target_excluded_from_model_input": True,
                    "new_model_call": False,
                    "new_simulator_step": False,
                },
                "current_eef_world_m": current_eef.tolist(),
                "anchor_world_m": anchor.tolist(),
                "target_distance_m": current_distance,
                "initial_target_distance_m": initial_distance,
                "progress_m": progress,
                "historical_policy_safety_gate": observation["safety_gate"],
                "historical_policy_action_sha256": observation["decoded_sha256"],
                "wbc_segment": wbc,
                "payload_files": {
                    "visited_row_npz": str(visited_path.relative_to(output_root)),
                    "expert_label_npz": str(expert_path.relative_to(output_root)),
                    "body_frame_ppm": str(body_path.relative_to(output_root)),
                    "wrist_frame_ppm": str(wrist_path.relative_to(output_root)),
                    "body_frame_sha256": body_sha,
                    "wrist_frame_sha256": wrist_sha,
                },
                "source_files": {name: source_files[f"{condition_id}/{name}"] for name in required_names},
                "source_stage43": {
                    "accepted_head": STAGE43_ACCEPTED_HEAD,
                    "result_sha256": STAGE43_RESULT_SHA256,
                    "seal_sha256": STAGE43_SEAL_SHA256,
                    "cohort_manifest_sha256": STAGE43_COHORT_MANIFEST_SHA256,
                },
            })
    require(len(rows) == VISITED_ROW_COUNT, f"materialized row count mismatch: {len(rows)}")
    rows_path = output_root / "visited_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    visited_tree_sha, visited_entries = tree_sha256(visited_dir, "visited_rows")
    expert_tree_sha, expert_entries = tree_sha256(expert_dir, "expert_labels")
    visited_manifest = {
        "schema": "CORELAB_PI0181_VISITED_STATE_PACK_V1",
        "status": "READY",
        "target_id": TARGET_ID,
        "row_count": len(rows),
        "condition_order": list(CONDITION_ORDER),
        "q_order": [0, 1, 2, 3],
        "model_input_fields": ["body_rgb", "wrist_rgb", "state_v1", "language"],
        "target_excluded_from_model_input": True,
        "stage43_result_sha256": STAGE43_RESULT_SHA256,
        "stage43_seal_sha256": STAGE43_SEAL_SHA256,
        "stage43_cohort_manifest_sha256": STAGE43_COHORT_MANIFEST_SHA256,
        "tree_sha256": visited_tree_sha,
        "file_count": len(visited_entries),
        "entries": visited_entries,
        "source_files": source_files,
        "validation_e_read": False,
        "holdout_f_read": False,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
    }
    expert_manifest = {
        "schema": "CORELAB_PI0181_EXPERT_LABEL_PACK_V1",
        "status": "READY",
        "label_source": "STAGE22_SAME_SCENE_EXPERT",
        "target_id": TARGET_ID,
        "row_count": len(rows),
        "action_shape": [HORIZON, ACTION_DIM],
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_ID,
        "tree_sha256": expert_tree_sha,
        "file_count": len(expert_entries),
        "entries": expert_entries,
        "target_excluded_from_model_input": True,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
    }
    write_json(output_root / "visited_pack_manifest.json", visited_manifest)
    write_json(output_root / "expert_label_manifest.json", expert_manifest)
    return rows, {
        "visited_pack_tree_sha256": visited_tree_sha,
        "expert_label_pack_sha256": expert_tree_sha,
        "visited_pack_manifest_sha256": sha256_file(output_root / "visited_pack_manifest.json"),
        "expert_label_manifest_sha256": sha256_file(output_root / "expert_label_manifest.json"),
        "rows_jsonl_sha256": sha256_file(rows_path),
        "source_files": source_files,
    }


def _full24_states(full24_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta_info = read_json(full24_root / "meta" / "info.json")
    split_lock = read_json(full24_root / "meta" / "stage22_split_lock.json")
    splits = meta_info.get("splits", {})
    require(meta_info.get("total_episodes") == 24 and meta_info.get("total_frames") == 960, "FULL24 meta count mismatch")
    require(splits.get("train") == "0:24" and splits.get("validation") == "24:24", "FULL24 train/empty-validation split mismatch")
    require(split_lock.get("selected_split") == "train" and split_lock.get("dataset_role") == "TRAIN", "FULL24 split lock is not TRAIN")
    require(split_lock.get("validation_target_ids") == [] and split_lock.get("final_test_target_ids") == [], "FULL24 split lock contains non-train targets")
    tree_sha, entries = tree_sha256(full24_root, "payload/lerobot_train")
    require(tree_sha == "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789", f"FULL24 tree SHA mismatch: {tree_sha}")
    require(len(entries) == 85 and sum(item["bytes"] for item in entries) == 919661, "FULL24 tree count/bytes mismatch")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise PI0181Error("pyarrow is required for the CPU-only FULL24 coverage audit") from exc
    parquet_paths = sorted((full24_root / "data").glob("chunk-*/episode_*.parquet"))
    require(len(parquet_paths) == 24, f"FULL24 parquet episode count mismatch: {len(parquet_paths)}")
    state_rows: list[np.ndarray] = []
    for path in parquet_paths:
        table = parquet.read_table(path, columns=["observation.state"])
        values = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        require(values.shape == (40, STATE_DIM), f"FULL24 state shape mismatch in {path}: {values.shape}")
        require(np.isfinite(values).all(), f"FULL24 non-finite state in {path}")
        state_rows.append(values)
    states = np.concatenate(state_rows, axis=0)
    require(states.shape == (960, STATE_DIM), f"FULL24 concatenated state shape mismatch: {states.shape}")
    return states, {
        "root": str(full24_root.resolve()),
        "tree_sha256": tree_sha,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "meta_info_sha256": sha256_file(full24_root / "meta" / "info.json"),
        "split_lock_sha256": sha256_file(full24_root / "meta" / "stage22_split_lock.json"),
        "split": splits,
        "state_shape": list(states.shape),
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def coverage_report(rows: Sequence[Mapping[str, Any]], full24_root: Path, output_root: Path) -> dict[str, Any]:
    states, full24_audit = _full24_states(full24_root)
    state_matrix = np.asarray([row["state_v1"] for row in rows], dtype=np.float64)
    nearest_state_distance = np.linalg.norm(state_matrix[:, None, :] - states[None, :, :], axis=2).min(axis=1)
    nearest_eef_distance = np.linalg.norm(state_matrix[:, None, :3] - states[None, :, :3], axis=2).min(axis=1)
    policy_l2: list[float] = []
    policy_rmse: list[float] = []
    roughness_discrepancy: list[float] = []
    first_difference_discrepancy: list[float] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_slug = row["row_slug"]
        with np.load(output_root / "visited_rows" / f"{row_slug}.npz", allow_pickle=False) as visited, np.load(output_root / "expert_labels" / f"{row_slug}.npz", allow_pickle=False) as expert:
            policy = np.asarray(visited["historical_decoded"], dtype=np.float64)
            oracle = np.asarray(expert["expert_action_structured_v1"], dtype=np.float64)
        delta = policy - oracle
        policy_l2.append(float(np.linalg.norm(delta)))
        policy_rmse.append(float(np.sqrt(np.mean(delta * delta))))
        policy_second = np.diff(policy, n=2, axis=0)
        oracle_second = np.diff(oracle, n=2, axis=0)
        first_difference_discrepancy.append(float(np.mean(np.linalg.norm(np.diff(policy, axis=0) - np.diff(oracle, axis=0), axis=1))))
        roughness_discrepancy.append(float(np.mean(np.linalg.norm(policy_second - oracle_second, axis=1))))
        rows_by_id[row["row_id"]] = {
            "condition_id": row["condition_id"],
            "q_index": row["q_index"],
            "nearest_full24_state_l2": float(nearest_state_distance[len(rows_by_id)]),
            "nearest_full24_eef_xyz_l2": float(nearest_eef_distance[len(rows_by_id)]),
            "policy_vs_expert_active10d_l2": policy_l2[-1],
            "policy_vs_expert_active10d_rmse": policy_rmse[-1],
            "first_difference_discrepancy": first_difference_discrepancy[-1],
            "roughness_discrepancy": roughness_discrepancy[-1],
            "target_distance_m": row["target_distance_m"],
            "progress_m": row["progress_m"],
            "safety_gate": row["historical_policy_safety_gate"],
            "wbc_tracking_rms_position_m": row["wbc_segment"]["tracking_rms_position_m"],
        }
    pairwise = np.linalg.norm(state_matrix[:, None, :] - state_matrix[None, :, :], axis=2)
    pairwise[np.diag_indices_from(pairwise)] = np.inf
    nearest_other = np.min(pairwise, axis=1)
    safety_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row["historical_policy_safety_gate"]["decision"])
        safety_counts[decision] = safety_counts.get(decision, 0) + 1
    report = {
        "schema": "CORELAB_PI0181_VISITED_STATE_COVERAGE_REPORT_V1",
        "status": "READY",
        "target_id": TARGET_ID,
        "target_world_m": TARGET_WORLD.tolist(),
        "row_count": len(rows),
        "condition_order": list(CONDITION_ORDER),
        "q_order": [0, 1, 2, 3],
        "full24": full24_audit,
        "per_row": rows_by_id,
        "summary": {
            "nearest_full24_state_l2_mean": float(np.mean(nearest_state_distance)),
            "nearest_full24_state_l2_max": float(np.max(nearest_state_distance)),
            "nearest_full24_eef_xyz_l2_mean": float(np.mean(nearest_eef_distance)),
            "nearest_full24_eef_xyz_l2_max": float(np.max(nearest_eef_distance)),
            "policy_vs_expert_active10d_l2_mean": float(np.mean(policy_l2)),
            "policy_vs_expert_active10d_l2_max": float(np.max(policy_l2)),
            "policy_vs_expert_active10d_rmse_mean": float(np.mean(policy_rmse)),
            "first_difference_discrepancy_mean": float(np.mean(first_difference_discrepancy)),
            "roughness_discrepancy_mean": float(np.mean(roughness_discrepancy)),
            "state_diversity_pairwise_l2_min": float(np.min(pairwise[np.isfinite(pairwise)])),
            "state_diversity_nearest_other_l2_mean": float(np.mean(nearest_other)),
            "state_diversity_nearest_other_l2_max": float(np.max(nearest_other)),
            "target_distance_m_by_row": [float(row["target_distance_m"]) for row in rows],
            "progress_m_by_row": [float(row["progress_m"]) for row in rows],
            "safety_gate_distribution": safety_counts,
            "q_progression": {
                str(q): [float(row["progress_m"]) for row in rows if int(row["q_index"]) == q]
                for q in range(4)
            },
        },
        "active_action_dimensions": 10,
        "policy_vs_expert_comparison": "same observed state, active StructuredActionV1 physical 10D, descriptive only",
        "target_exclusion_boundary": "target XYZ and expert labels are metadata/supervision only; model input is body RGB + wrist RGB + StateV1 + language",
        "validation_e_read": False,
        "holdout_f_read": False,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
    }
    write_json(output_root / "coverage_report.json", report)
    report["sha256"] = sha256_file(output_root / "coverage_report.json")
    return report


def full24_sequence(seed: int = TRAIN_SEED, update_count: int = UPDATE_COUNT) -> list[int]:
    rng = random.Random(int(seed))
    sequence: list[int] = []
    while len(sequence) < int(update_count):
        cycle = list(range(FULL24_SLOT_COUNT))
        rng.shuffle(cycle)
        sequence.extend(cycle)
    return sequence[: int(update_count)]


def multi_visited_schedule(seed: int = TRAIN_SEED, update_count: int = UPDATE_COUNT) -> dict[str, Any]:
    """Build deterministic S20 slots without changing the existing Q0 law."""

    rng = random.Random(int(seed))
    full_sequence: list[int] = []
    while len(full_sequence) < int(update_count):
        cycle = list(range(FULL24_SLOT_COUNT))
        rng.shuffle(cycle)
        full_sequence.extend(cycle)
    full_sequence = full_sequence[: int(update_count)]
    visited_cycle = [f"{condition_id}-q{q}" for condition_id in CONDITION_ORDER for q in range(4)]
    rng.shuffle(visited_cycle)
    e_slots = [
        {"position": index, "source": "FULL24", "full24_slot": int(slot), "visited_row_id": None}
        for index, slot in enumerate(full_sequence)
    ]
    s_slots = []
    for index, slot in enumerate(full_sequence):
        if index % 2 == 0:
            s_slots.append({"position": index, "source": "FULL24", "full24_slot": int(slot), "visited_row_id": None})
        else:
            s_slots.append({"position": index, "source": "VISITED20", "full24_slot": None, "visited_row_id": visited_cycle[(index // 2) % VISITED_ROW_COUNT]})
    require(len(e_slots) == UPDATE_COUNT and len(s_slots) == UPDATE_COUNT, "S20 schedule update count drift")
    require(sum(slot["source"] == "FULL24" for slot in s_slots) == 128, "S20 FULL24 slot count drift")
    require(sum(slot["source"] == "VISITED20" for slot in s_slots) == 128, "S20 visited slot count drift")
    require(set(slot["visited_row_id"] for slot in s_slots if slot["visited_row_id"] is not None) == set(visited_cycle), "S20 visited cycle coverage drift")
    return {
        "schema": "CORELAB_PI0181_S20_DETERMINISTIC_SCHEDULE_V1",
        "seed": int(seed),
        "update_count": int(update_count),
        "full24_slot_count": FULL24_SLOT_COUNT,
        "visited_row_count": VISITED_ROW_COUNT,
        "full24_sequence": full_sequence,
        "full24_sequence_sha256": canonical_sha(full_sequence),
        "visited_cycle": visited_cycle,
        "visited_cycle_sha256": canonical_sha(visited_cycle),
        "rng_rule": "random.Random(seed), FULL24 shuffle stream first, then one deterministic visited-row shuffle",
        "E": e_slots,
        "S20": s_slots,
        "counts": {"E_FULL24": 256, "S20_FULL24": 128, "S20_VISITED20": 128},
        "pairing_rule": "E and S20 share the same FULL24 slot at every even/paired position; S20 replaces odd positions with VISITED20",
        "validation_e_read": False,
        "holdout_f_read": False,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
        "training_executed": False,
    }


def write_training_package(output_root: Path, schedule: Mapping[str, Any], pack_hashes: Mapping[str, Any]) -> dict[str, Any]:
    package = {
        "schema": "CORELAB_PI0181_MULTI_VISITED20_TRAINING_PACKAGE_V1",
        "package_id": "MULTI_VISITED20_V1",
        "status": "PREPARED_NOT_EXECUTED",
        "base_checkpoint": "LATE_BOUND_PI_SELECTION_LEGACY_OR_L1",
        "arms": {
            "E": {"name": "FULL24_ONLY", "updates": 256, "full24_slots": 256, "visited_slots": 0},
            "S20": {"name": "128_FULL24_PLUS_128_DETERMINISTIC_VISITED20", "updates": 256, "full24_slots": 128, "visited_slots": 128},
        },
        "runspec": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "schedule": "cosine",
            "warmup_updates": 10,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "precision": "bf16",
            "trainable_modules": ["projector", "diffusion"],
            "frozen_modules": ["visual", "llm", "vlln"],
            "loss": "masked StructuredActionV1 flow matching",
            "T16": "UNCHANGED_FROM_PI0178",
            "rng_seed": TRAIN_SEED,
            "target_id": TARGET_ID,
            "target_excluded_from_model_input": True,
        },
        "source_packs": dict(pack_hashes),
        "schedule_sha256": sha256_file(output_root / "s20_schedule.json"),
        "training_executed": False,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "gpu_execution": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    package_path = output_root / "training_package" / "MULTI_VISITED20_V1_RUNSPEC.json"
    write_json(package_path, package)
    package["runspec_sha256"] = sha256_file(package_path)
    return package


def historical_replay(rows: Sequence[Mapping[str, Any]], output_root: Path, recorder_script: Path) -> dict[str, Any]:
    """Render real Stage43 camera frames with historical and expert H40 overlays."""

    replay_root = output_root / "historical_native_replay"
    frame_rows: list[dict[str, Any]] = []
    for row in rows:
        row_slug = row["row_slug"]
        with np.load(output_root / "visited_rows" / f"{row_slug}.npz", allow_pickle=False) as visited, np.load(output_root / "expert_labels" / f"{row_slug}.npz", allow_pickle=False) as expert:
            decoded = np.asarray(visited["historical_decoded"], dtype=np.float64)
            anchor = np.asarray(visited["anchor_world_m"], dtype=np.float64)
            expert_action = np.asarray(expert["expert_action_structured_v1"], dtype=np.float64)
            historical_world = anchor[None, :] + decoded[:, :3]
            expert_world = anchor[None, :] + expert_action[:, :3]
        source_body = output_root / row["payload_files"]["body_frame_ppm"]
        source_wrist = output_root / row["payload_files"]["wrist_frame_ppm"]
        frame_rows.append({
            "timestamp_s": float(row["condition_order_index"]) * 10.0 + float(row["observation_timestamp_s"]),
            "source_observation_timestamp_s": float(row["observation_timestamp_s"]),
            "body_frame": str(source_body),
            "wrist_frame": str(source_wrist),
            "achieved_eef_xyz": row["current_eef_world_m"],
            "target_xyz": row["target_world_m"],
            "predicted_h40_xyz": historical_world.tolist(),
            "expert_h40_xyz": expert_world.tolist(),
            "model_update_index": int(row["q_index"]),
            "safety_gate": row["historical_policy_safety_gate"]["decision"],
            "safety_gate_issue_codes": row["historical_policy_safety_gate"].get("issues", []),
            "wbc_tracking_rms_position_m": row["wbc_segment"]["tracking_rms_position_m"],
            "lifecycle_state": f"HISTORICAL_STAGE43_{row['condition_id']}_Q{row['q_index']}",
            "abort_reason": None,
            "target_distance_m": row["target_distance_m"],
            "initial_target_distance_m": row["initial_target_distance_m"],
            "progress_m": row["progress_m"],
            "timeout": False,
            "all_finite": True,
        })
    frames_path = output_root / "historical_replay_frames.json"
    portable_rows = []
    for row in frame_rows:
        portable = dict(row)
        portable["body_frame"] = str(Path(row["body_frame"]).relative_to(output_root))
        portable["wrist_frame"] = str(Path(row["wrist_frame"]).relative_to(output_root))
        portable_rows.append(portable)
    write_json(frames_path, {
        "schema": "CORELAB_PI0181_HISTORICAL_REPLAY_FRAMES_V1",
        "source_root": ".",
        "source_frame_paths_relative": True,
        "frames": portable_rows,
    })
    require(recorder_script.is_file(), f"replay recorder script missing: {recorder_script}")
    scripts_dir = str(recorder_script.resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from stage45f_pi0179_replay_recorder import ReplayRecorderConfigV1, ReplayRecorderV1, ReplayFrameV1

    frames = tuple(ReplayFrameV1.from_dict(row) for row in frame_rows)
    manifest = ReplayRecorderV1(ReplayRecorderConfigV1(
        output_dir=replay_root,
        source_identity={
            "kind": "SEALED_STAGE43_HISTORICAL_NATIVE_EVIDENCE",
            "stage43_result_sha256": STAGE43_RESULT_SHA256,
            "stage43_seal_sha256": STAGE43_SEAL_SHA256,
            "stage43_cohort_manifest_sha256": STAGE43_COHORT_MANIFEST_SHA256,
            "new_model_calls": 0,
            "new_simulator_steps": 0,
        },
        classification="PI0181_HISTORICAL_NATIVE_REPLAY",
        overwrite=True,
        native_third_person_source_available=False,
        allow_missing_ffmpeg=True,
    )).render(frames)
    manifest_path = replay_root / "manifest.json"
    manifest["frames_json_sha256"] = sha256_file(frames_path)
    manifest["historical_source_note"] = "Stage43 accepted body/wrist RGB and native WBC rows; no native third-person camera payload exists, so third_person.mp4 is a body-derived annotated proxy."
    write_json(manifest_path, manifest)
    return {
        "status": "PASS" if manifest.get("render_status") == "PASS" and manifest.get("native_third_person_source_available") else "PARTIAL",
        "reason": manifest.get("historical_native_replay_status"),
        "frame_count": len(frames),
        "frames_json_sha256": sha256_file(frames_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(replay_root),
        "native_third_person_source_available": False,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage43_root = Path(args.stage43_root).resolve()
    full24_root = Path(args.full24_root).resolve()
    output_root = Path(args.output_root).resolve()
    require(not output_root.exists() or not any(output_root.iterdir()), f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    stage43_audit = verify_stage43(stage43_root)
    rows, pack_hashes = extract_rows(stage43_root, output_root)
    coverage = coverage_report(rows, full24_root, output_root)
    schedule = multi_visited_schedule()
    write_json(output_root / "s20_schedule.json", schedule)
    schedule_sha = sha256_file(output_root / "s20_schedule.json")
    package = write_training_package(output_root, schedule, pack_hashes)
    replay = historical_replay(rows, output_root, Path(args.recorder_script).resolve())
    result = {
        "schema": "CORELAB_PI0181_VISITED_STATE_SUPPORT_RESULT_V1",
        "token": "CORELAB_PI0181_VISITED_STATE_SUPPORT_FAST_LANE_FOR_PI_REVIEW",
        "status": "READY",
        "pi_gate": PI_GATE,
        "pr_pointer": PR_POINTER,
        "source_head": SOURCE_HEAD,
        "code_script_sha256": sha256_file(Path(__file__).resolve()),
        "stage43": stage43_audit,
        "stage43_seal_revalidated": True,
        "visited_rows_expected": VISITED_ROW_COUNT,
        "visited_rows_materialized": len(rows),
        "expert_relabeled_rows": len(rows),
        "target_id": TARGET_ID,
        "target_world_m": TARGET_WORLD.tolist(),
        "visited_pack_tree_sha256": pack_hashes["visited_pack_tree_sha256"],
        "expert_label_pack_sha256": pack_hashes["expert_label_pack_sha256"],
        "coverage_report_sha256": coverage["sha256"],
        "s20_schedule_sha256": schedule_sha,
        "multi_visited20_package": {
            "package_id": package["package_id"],
            "status": package["status"],
            "runspec_sha256": package["runspec_sha256"],
        },
        "historical_native_replay": replay,
        "counters": {
            "validation_e_reads": 0,
            "holdout_f_reads": 0,
            "new_model_calls": 0,
            "new_simulator_steps": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "gpu_execution": 0,
            "h200_pbs": 0,
            "hardware": 0,
            "wbc": 0,
            "sim": 0,
        },
        "preserved_pi0178_support": [
            "SINGLE_Q0_V1 package remains unchanged",
            "candidate scorecard remains READY/HOLD",
            "replay recorder remains reach-first/grasp-later",
        ],
        "execution": {
            "new_model_calls": 0,
            "new_simulator_steps": 0,
            "training_executed": False,
            "gpu_execution": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
    }
    write_json(output_root / "pi0181_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage43-root", type=Path, default=STAGE43_ROOT_DEFAULT)
    parser.add_argument("--full24-root", type=Path, default=FULL24_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recorder-script", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except (PI0181Error, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "CORELAB_PI0181_VISITED_STATE_SUPPORT_RESULT_V1", "status": "BLOCKED", "error": str(exc)}, indent=2))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
