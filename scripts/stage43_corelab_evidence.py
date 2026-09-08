#!/usr/bin/env python3
"""CPU-only Stage43 G0/G1 evidence and fail-closed cohort contract.

Stage43 starts by checking that the task target is a real same-scene visual
fixture and that the ``achieved_pos`` stream is a world-frame EEF measurement.
Only sealed Stage41 LH10 evidence is read.  This module deliberately does
not import IsaacGym, CUDA, GR00T, or a checkpoint.  The GPU runner is kept in
``stage43_corelab_initial_condition.py`` and consumes a frozen bundle emitted
by this module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import stage40_corelab_evidence as stage40
import stage41_corelab_evidence as stage41


PI_GATE = "PI_GATE_STATE_0068"
ISSUE_COMMENT = 5580348321
STAGE_ID = "Stage43"
BRANCH = "stage43-dev/corelab-controlled-cohort"
STAGE42_HEAD = "6461240cfa9615ea5757a5db50dd03416a021bf0"
STAGE41_HEAD = "95ec9ee0ebdaac8ea09b427cde94005591ac37fd"
STAGE41_BUNDLE_SHA256 = "e9891d04b75ed50bd405052fdcd730fe4a7bfbf75224d68ca48d3026afc37aee"
STAGE41_A_SEAL = "3882e5178551384a41685533944b7388788499fcd2b978178c8eddc2f8b6e02c"
STAGE41_B_SEAL = "4a90f0cd86fa53c3b04cc83ca1afb388836e57250dc849b3b02a4d8cb0936cac"
M2_SEAL = "ad178ac49da1ced987f60224987ccab8dc1d3343674bb511b7c3fe13a3b8dcc5"
DEV_SEED = 4242
LANGUAGE = "move the end effector to the red target"
TARGET_POSITION_WORLD_M = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
HANDOFF_S = 0.9750000000000001
ACTION_DT_S = 0.05
Q_IDS = tuple("stage41-q%d" % index for index in range(11))
COHORT_IDS = ("C0_NOMINAL", "C1_X_PLUS_5CM", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM")
COHORT_DELTAS_M = {
    "C0_NOMINAL": (0.0, 0.0, 0.0),
    "C1_X_PLUS_5CM": (0.05, 0.0, 0.0),
    "C2_X_MINUS_5CM": (-0.05, 0.0, 0.0),
    "C3_Y_PLUS_5CM": (0.0, 0.05, 0.0),
    "C4_Y_MINUS_5CM": (0.0, -0.05, 0.0),
}
COHORT_ORDER = tuple(COHORT_IDS)
COHORT_SEGMENT_LABELS = ("q0_to_q1", "q1_to_q2", "q2_to_q3", "q3_completion")
COHORT_FORWARD_COUNT = 4
COHORT_REFERENCE_UPDATE_COUNT = 3
COHORT_TOTAL_COUNT = 5
COHORT_TOTAL_FORWARDS = COHORT_TOTAL_COUNT * COHORT_FORWARD_COUNT
COHORT_TOTAL_UPDATES = COHORT_TOTAL_COUNT * COHORT_REFERENCE_UPDATE_COUNT
COHORT_SEGMENT_SCHEDULE = (49, 49, 49, 98)

# Stage43 owns its own bundle/manifest/seal namespace.  The accepted Stage40
# and Stage41 serializers are intentionally not reused for the cohort because
# their stage IDs and counter contracts describe a different experiment.
BUNDLE_SCHEMA = "CORELAB_STAGE43_EXECUTION_BUNDLE_V1"
CONDITION_MANIFEST_SCHEMA = "CORELAB_STAGE43_CONDITION_EVIDENCE_MANIFEST_V1"
CONDITION_SEAL_SCHEMA = "CORELAB_STAGE43_CONDITION_SEAL_V1"
RESULT_SCHEMA = "CORELAB_STAGE43_CONTROLLED_COHORT_RESULT_V1"
RETURN_TOKEN = "GPT_PI_REVIEW_AFTER_STAGE43_CONTROLLED_COHORT"
STRUCTURED_CHECKPOINT_TREE_SHA256 = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
STRUCTURED_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
STAGE22_MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
GROOT_SOURCE_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
WBC_TREE_SHA256 = "9d653c39c73d5755a29d46f4dff91b69120244f660f41a977db471d58ba8b193"
WBC_ARM_SHA256 = "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a"
WBC_DOG_SHA256 = "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670"
TRAJECTORY_SHA256 = "e09d03e4f837234a1ed7504f6d16f202a40d7fab2c3cbbc0f2e2fa17b0b33722"

FORBIDDEN_FIELDS = (
    "training", "fine_tuning", "formal_seed_used", "validation_e_read", "holdout_f_read",
    "new_data_read", "hardware_executed", "h200_used", "vanda_used", "checkpoint_substitution",
    "model_substitution", "processor_substitution", "precision_change", "quantization", "cpu_offload",
    "clipping", "smoothing", "blending", "adaptive_retiming", "physics_reset", "physics_teleport",
    "policy_history_clear", "target_relocation", "orientation_perturbation", "joint_perturbation",
    "velocity_perturbation", "settling_steps", "handoff_teleport",
)


class Stage43EvidenceError(ValueError):
    """Raised when a Stage43 evidence or contract check fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage43EvidenceError(message)


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


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required: %s" % path)
    return value


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    destination.write_bytes(payload)
    return sha256_bytes(payload)


def expected_forbidden_operations() -> Dict[str, bool]:
    return {name: False for name in FORBIDDEN_FIELDS}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.call(["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant]) == 0


def _source_lines(path: Path, patterns: Sequence[str]) -> Dict[str, List[int]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {pattern: [index + 1 for index, line in enumerate(lines) if re.search(pattern, line)] for pattern in patterns}


def _source_marker(path: Path, pattern: str) -> int:
    matches = _source_lines(path, (pattern,))[pattern]
    _require(matches, "source marker not found: %s in %s" % (pattern, path))
    return matches[0]


def _resolved_child(root: Path, relative: str) -> Path:
    relative_path = Path(str(relative))
    _require(not relative_path.is_absolute() and relative_path.parts, "evidence path must be relative")
    _require(all(part not in {"", ".", ".."} for part in relative_path.parts), "non-canonical evidence path")
    base = Path(root).resolve(strict=True)
    cursor = base
    for part in relative_path.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "symlink evidence is forbidden")
    path = (base / relative_path).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise Stage43EvidenceError("evidence path escapes root") from exc
    _require(path.is_file() and not path.is_symlink(), "evidence is not a regular file: %s" % path)
    return path


def _role_path(manifest: Mapping[str, Any], root: Path, role: str) -> Path:
    records = [item for item in manifest.get("records", []) if item.get("role") == role]
    _require(len(records) == 1, "role is not unique: %s" % role)
    path = _resolved_child(root, str(records[0].get("relative_path")))
    _require(sha256_file(path) == str(records[0].get("sha256")), "role hash mismatch: %s" % role)
    return path


def _finite_vector(value: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(tuple(array.shape) == shape and bool(np.isfinite(array).all()), "%s must be finite %s" % (label, shape))
    return array


def _red_pixel_count(image: np.ndarray) -> int:
    rgb = np.asarray(image, dtype=np.uint8)
    mask = (rgb[..., 0] > 100) & (rgb[..., 0] > rgb[..., 1] * 1.35) & (rgb[..., 0] > rgb[..., 2] * 1.35)
    return int(np.count_nonzero(mask))


def _flatten_rows(run_root: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for name in ("rows_before_handoff.json", "rows_after_handoff.json"):
        payload = read_json(run_root / name)
        segments = payload.get("segments")
        _require(isinstance(segments, list), "row segments missing: %s" % name)
        for segment in segments:
            _require(isinstance(segment, list), "row segment is not a list: %s" % name)
            rows.extend(segment)
    _require(rows, "sealed run has no WBC rows")
    return rows


def _load_run(run_root: Path, seal_path: Path, predecessor_seal: Path, expected_seal: str) -> Dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    seal = stage41.verify_stage_seal(seal_path, root, expected_head=STAGE41_HEAD, predecessor_seal=predecessor_seal)
    _require(seal["seal_sha256"] == expected_seal, "unexpected Stage41 seal identity")
    manifest = read_json(root / "evidence_manifest.json")
    stage41.verify_evidence_manifest(manifest, root)
    state = read_json(_role_path(manifest, root, "state_v1"))
    observations = state.get("observations")
    _require(isinstance(observations, list) and len(observations) == len(Q_IDS), "Stage41 observation count mismatch")
    body_path = _role_path(manifest, root, "observation_body")
    wrist_path = _role_path(manifest, root, "observation_wrist")
    body_npz: Dict[str, np.ndarray] = {}
    wrist_npz: Dict[str, np.ndarray] = {}
    with np.load(body_path, allow_pickle=False) as body_data, np.load(wrist_path, allow_pickle=False) as wrist_data:
        for index, request_id in enumerate(Q_IDS):
            _require(request_id in body_data.files and request_id in wrist_data.files, "missing sealed image: %s" % request_id)
            body = np.asarray(body_data[request_id], dtype=np.uint8)
            wrist = np.asarray(wrist_data[request_id], dtype=np.uint8)
            _require(body.shape == (96, 96, 3) and wrist.shape == (96, 96, 3), "image ABI mismatch: %s" % request_id)
            body_npz[request_id] = body
            wrist_npz[request_id] = wrist
            obs = observations[index]
            _require(obs.get("request_id") == request_id, "observation order mismatch")
            _require(obs.get("body_sha256") == sha256_array(body), "sealed body hash mismatch: %s" % request_id)
            _require(obs.get("wrist_sha256") == sha256_array(wrist), "sealed wrist hash mismatch: %s" % request_id)
    requests: Dict[str, Dict[str, Any]] = {}
    for index, request_id in enumerate(Q_IDS):
        request_json = read_json(root / "mailbox" / "requests" / (request_id + ".json"))
        request_npz_path = root / "mailbox" / "requests" / (request_id + ".npz")
        _require(request_json.get("request_id") == request_id, "request order mismatch: %s" % request_id)
        _require(request_json.get("language") == LANGUAGE, "language drift: %s" % request_id)
        _require(request_json.get("seed") == DEV_SEED, "seed drift: %s" % request_id)
        _require(request_json.get("source_frame") == SOURCE_FRAME and request_json.get("tcp_semantic_id") == TCP_SEMANTIC_ID, "frame contract drift")
        target = _finite_vector(request_json.get("target_position_world_m"), (3,), request_id + ".target_json")
        _require(np.array_equal(target, TARGET_POSITION_WORLD_M), "evaluation target drift: %s" % request_id)
        _require(request_npz_path.is_file() and not request_npz_path.is_symlink(), "request NPZ missing: %s" % request_id)
        with np.load(request_npz_path, allow_pickle=False) as request_npz:
            required = {"body_rgb", "wrist_rgb", "state", "target_position_world_m"}
            _require(required.issubset(set(request_npz.files)), "request ABI mismatch: %s" % request_id)
            _require(np.array_equal(np.asarray(request_npz["body_rgb"], dtype=np.uint8), body_npz[request_id]), "request/body mismatch: %s" % request_id)
            _require(np.array_equal(np.asarray(request_npz["wrist_rgb"], dtype=np.uint8), wrist_npz[request_id]), "request/wrist mismatch: %s" % request_id)
            target_npz = _finite_vector(request_npz["target_position_world_m"], (3,), request_id + ".target_npz")
            _require(np.array_equal(target_npz, TARGET_POSITION_WORLD_M), "request target drift: %s" % request_id)
        requests[request_id] = request_json
        _require(observations[index].get("request", {}).get("target_position_world_m") == TARGET_POSITION_WORLD_M.tolist(), "sealed target metadata drift")
    rows = _flatten_rows(root)
    _require(all(str(row.get("termination_code", "NONE")) == "NONE" and row.get("done") is False and row.get("reset_buf") is False for row in rows), "sealed lifecycle rows are not clean")
    return {"root": root, "seal": seal, "manifest": manifest, "observations": observations, "requests": requests, "body": body_npz, "wrist": wrist_npz, "rows": rows}


def _target_metrics(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    rows = snapshot["rows"]
    initial_event = read_json(Path(snapshot["root"]) / "initial_bind.json")["events"][0]
    anchor = _finite_vector(initial_event.get("anchor_world_m"), (3,), "anchor_world_m")
    current_local = _finite_vector(initial_event.get("executor_initial", {}).get("handoff", {}).get("current_achieved_position"), (3,), "initial achieved local")
    initial_world = anchor + current_local
    row_positions = np.asarray([_finite_vector(row.get("achieved_pos"), (3,), "achieved_pos") for row in rows], dtype=np.float64)
    row_desired = np.asarray([_finite_vector(row.get("desired_pos"), (3,), "desired_pos") for row in rows], dtype=np.float64)
    positions = np.vstack((initial_world, row_positions))
    target = TARGET_POSITION_WORLD_M
    distances = np.linalg.norm(positions - target[None, :], axis=1)
    tracking = np.linalg.norm(row_positions - row_desired, axis=1)
    _require(bool(np.isfinite(distances).all() and np.isfinite(tracking).all()), "target metrics are non-finite")
    minimum_index = int(np.argmin(distances))
    d0 = float(distances[0])
    dmin = float(distances[minimum_index])
    dfinal = float(distances[-1])
    return {
        "sample_count_including_S0": int(len(distances)),
        "initial_target_error_m": d0,
        "minimum_target_error_m": dmin,
        "minimum_error_sample_index": minimum_index,
        "final_target_error_m": dfinal,
        "absolute_progress_m": float(d0 - dfinal),
        "best_progress_m": float(d0 - dmin),
        "normalized_progress": float((d0 - dfinal) / d0) if d0 > 0.0 else None,
        "initial_eef_world_m": initial_world.tolist(),
        "final_eef_world_m": positions[-1].tolist(),
        "tracking_rms_m": float(math.sqrt(float(np.mean(tracking * tracking)))),
        "tracking_max_m": float(np.max(tracking)),
        "progress_sign": "POSITIVE" if d0 > dfinal else ("ZERO" if math.isclose(d0, dfinal, abs_tol=1e-12) else "NEGATIVE"),
        "threshold_used": None,
        "definition": "S0 is the sealed INITIAL_BIND current achieved world position; subsequent samples are sealed achieved_pos rows; target is fixed world [0.30,0.10,0.58]",
    }


def _ast_model_input_audit(repo_root: Path) -> Dict[str, Any]:
    inference_path = Path(repo_root) / "scripts" / "stage21_same_scene_groot_inference.py"
    worker_path = Path(repo_root) / "scripts" / "stage21_same_scene_groot_worker.py"
    request_path = Path(repo_root) / "scripts" / "stage21_native_model_receding_horizon.py"
    for path in (inference_path, worker_path, request_path):
        _require(path.is_file(), "model boundary source missing: %s" % path)
    inference_text = inference_path.read_text(encoding="utf-8")
    tree = ast.parse(inference_text, filename=str(inference_path))
    run_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_one_raw"]
    _require(len(run_nodes) == 1, "_run_one_raw definition is not unique")
    run_node = run_nodes[0]
    calls = [node for node in ast.walk(run_node) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "VLAStepData"]
    _require(len(calls) == 1, "VLAStepData call is not unique")
    call = calls[0]
    keywords = [item.arg for item in call.keywords if item.arg is not None]
    _require(keywords == ["images", "states", "actions", "text", "embodiment"], "unexpected VLAStepData input ABI")
    call_dump = ast.dump(call, include_attributes=False)
    _require("target_position_world_m" not in call_dump, "numeric target entered VLAStepData")
    run_segment = ast.get_source_segment(inference_text, run_node) or ""
    _require("target_position_world_m" not in run_segment, "numeric target referenced in model-forward function")
    worker_text = worker_path.read_text(encoding="utf-8")
    _require("_load_observation" in worker_text and "_run_one_raw" in worker_text, "worker boundary markers missing")
    worker_calls = [node for node in ast.walk(ast.parse(worker_text, filename=str(worker_path))) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_run_one_raw"]
    _require(worker_calls, "worker does not call _run_one_raw")
    worker_keywords = sorted(item.arg for item in worker_calls[0].keywords if item.arg is not None)
    _require(worker_keywords == ["device", "language", "seed"], "worker forwards an unexpected numeric target")
    _require(len(worker_calls[0].args) == 4 and "target_position_world_m" not in ast.dump(worker_calls[0], include_attributes=False), "worker call carries a numeric target")
    request_text = request_path.read_text(encoding="utf-8")
    _require("target_position_world_m=np.asarray(target_position_world_m" in request_text, "request target storage marker missing")
    return {
        "status": "PASS",
        "target_excluded_from_model_forward": True,
        "model_input_keys": keywords,
        "worker_forward_keywords": worker_keywords,
        "request_target_role": "archived observation/evaluation metadata only",
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in (inference_path, worker_path, request_path)],
        "source_marker_lines": {
            str(inference_path): {
                "load_required_inputs": _source_marker(inference_path, r"required\s*=\s*\(\"body_rgb\",\s*\"wrist_rgb\",\s*\"state\"\)"),
                "vla_step_data": _source_marker(inference_path, r"content\s*=\s*VLAStepData\("),
                "model_forward": _source_marker(inference_path, r"prediction\s*=\s*model\.get_action"),
            },
            str(worker_path): {"load_observation": _source_marker(worker_path, r"observation\s*=\s*inference\._load_observation")},
            str(request_path): {"target_metadata_write": _source_marker(request_path, r"target_position_world_m=np\.asarray\(target_position_world_m")},
        },
        "claim_boundary": "the target XYZ is persisted for provenance/evaluation but is not a numeric GR00T input; visual/language grounding remains an empirical model question",
    }


def _source_semantics_audit(repo_root: Path, wbc_source_root: Path) -> Dict[str, Any]:
    stage21 = Path(repo_root) / "scripts" / "stage21_native_receding_horizon.py"
    stage40 = Path(repo_root) / "scripts" / "stage40_live_one_handoff.py"
    stage41_runner = Path(repo_root) / "scripts" / "stage41_corelab_long_horizon.py"
    wrapper = Path(repo_root) / "scripts" / "native_wbc_benchmark.py"
    for path in (stage21, stage40, stage41_runner, wrapper):
        _require(path.is_file(), "local semantic source missing: %s" % path)
    stage21_text = stage21.read_text(encoding="utf-8")
    target_literal = r"np\.asarray\(\[0\.30,\s*0\.10,\s*0\.58\]"
    target_literal_count = len(re.findall(target_literal, stage21_text))
    _require(target_literal_count >= 1, "native target actor literal is missing")
    required_markers = {
        "target_actor_creation": _source_marker(stage21, r"self\.stage21_target_actor\s*=\s*self\.gym\.create_actor"),
        "target_actor_name": _source_marker(stage21, r"stage21_red_target"),
        "target_marker_pose": _source_marker(stage21, r"marker_pose\.p\s*=\s*gymapi\.Vec3"),
        "target_color": _source_marker(stage21, r"set_rigid_body_color"),
        "native_capture": _source_marker(stage21, r"def _capture_scene"),
        "native_image_color": _source_marker(stage21, r"IMAGE_COLOR"),
        "red_mask": _source_marker(stage21, r"def _red_pixel_count"),
        "target_contract": _source_marker(stage21, r"position_world_m=np\.asarray\(target_position"),
        "body_camera_attach": _source_marker(stage21, r"body_camera"),
    }
    _require("target_position_world_m=np.asarray([0.30, 0.10, 0.58]" in stage40.read_text(encoding="utf-8"), "Stage40 evaluation target literal drift")
    _require("TARGET_POSITION_WORLD_M = np.asarray([0.30, 0.10, 0.58]" in stage41_runner.read_text(encoding="utf-8"), "Stage41 evaluation target literal drift")

    remote_wbc = Path(wbc_source_root) / "go1_gym" / "envs" / "roboduet" / "wbc_env.py"
    remote_legged = Path(wbc_source_root) / "go1_gym" / "envs" / "roboduet" / "legged_robot.py"
    for path in (remote_wbc, remote_legged):
        _require(path.is_file(), "WBC source missing: %s" % path)
    wbc_markers = {
        "world_to_body_conversion": _source_marker(remote_wbc, r"return self\._pose_world_to_body_9d\(self\.end_effector_state\[:, :3\]"),
        "world_target_doc": _source_marker(remote_wbc, r"World-frame position of the current target"),
        "eef_from_rigid_body": _source_marker(remote_wbc, r"self\.end_effector_state\s*=\s*self\.rigid_body_state\.view"),
        "eef_grasp_offset": _source_marker(remote_wbc, r"self\.end_effector_state\[:, :3\]\s*\+=\s*quat_apply"),
        "world_error": _source_marker(remote_wbc, r"self\.ee_pos_err\[:\]\s*=\s*target_pos_world\s*-\s*self\.end_effector_state\[:, :3\]"),
        "post_physics_eef_refresh": _source_marker(remote_wbc, r"def _arm_post_physics_hook"),
    }
    legged_markers = {
        "acquire_rigid_body_tensor": _source_marker(remote_legged, r"acquire_rigid_body_state_tensor"),
        "wrap_rigid_body_tensor": _source_marker(remote_legged, r"self\.rigid_body_state\s*=\s*gymtorch\.wrap_tensor"),
        "world_base_position": _source_marker(remote_legged, r"self\.base_pos\s*=\s*self\.root_states\[: self\.num_envs, 0:3\]"),
    }
    return {
        "status": "PASS",
        "target_literal_count_in_native_fixture": target_literal_count,
        "evaluation_target_world_m": TARGET_POSITION_WORLD_M.tolist(),
        "target_actor": {"name": "stage21_red_target", "radius_m": 0.035, "color_source_rgb": [230, 24, 20], "pose_source": TARGET_POSITION_WORLD_M.tolist()},
        "native_camera_capture": {"body_and_wrist": True, "image_format": "RGB_uint8_HWC", "resolution": [96, 96]},
        "achieved_pos_semantics": {
            "frame": "world",
            "eef_point": TCP_SEMANTIC_ID,
            "source_chain": "IsaacGym rigid_body_state -> WBC end_effector_state -> native row achieved_pos",
            "claim": "achieved_pos is the actual grasp-point EEF world position, not desired/reference trajectory position",
        },
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in (stage21, stage40, stage41_runner, wrapper, remote_wbc, remote_legged)],
        "source_marker_lines": {str(stage21): required_markers, str(remote_wbc): wbc_markers, str(remote_legged): legged_markers},
    }


def _camera_smoke_audit(camera_root: Path) -> Dict[str, Any]:
    root = Path(camera_root).resolve(strict=True)
    report_path = root / "camera_smoke.json"
    report = read_json(report_path)
    _require(report.get("schema") == "STAGE21_NATIVE_SAME_SCENE_CAMERA_SMOKE_V1", "camera smoke schema mismatch")
    _require(report.get("status") == "PASS", "camera smoke was not PASS")
    same_scene = report.get("same_scene", {})
    _require(same_scene.get("native_wbc_active") is True and same_scene.get("target_actor") == "stage21_red_target", "camera smoke is not native same-scene")
    camera = report.get("native_camera", {})
    _require(camera.get("body_attached") is True and camera.get("wrist_attached") is True and camera.get("resolution") == [96, 96], "native camera attachment evidence missing")
    images: Dict[str, Dict[str, Any]] = {}
    for target_id, expected_position in (("target_a", TARGET_POSITION_WORLD_M), ("target_b", np.asarray([0.30, -0.20, 0.58]))):
        observation = report.get("image_observations", {}).get(target_id, {})
        target = _finite_vector(observation.get("target", {}).get("position_world_m"), (3,), target_id + ".target")
        _require(np.array_equal(target, expected_position), "camera smoke target pose drift: %s" % target_id)
        for camera_name in ("body", "wrist"):
            npz = root / ("stage21_observation_%s.npz" % target_id)
            with np.load(npz, allow_pickle=False) as data:
                image = np.asarray(data[camera_name + "_rgb"], dtype=np.uint8)
            count = _red_pixel_count(image)
            expected_count = int(report["image_observations"]["red_pixel_count"][camera_name + "_" + target_id[-1]])
            _require(count == expected_count, "camera smoke red-pixel evidence mismatch: %s/%s" % (target_id, camera_name))
            images.setdefault(target_id, {})[camera_name] = {"red_pixel_count": count, "sha256": sha256_array(image)}
    visibility = report.get("image_observations", {}).get("red_pixel_count", {})
    _require(any(int(visibility.get(key, 0)) > 0 for key in ("body_a", "wrist_a", "body_b", "wrist_b")), "camera smoke never rendered the red target")
    _require(report.get("counterfactual", {}).get("same_state") is True and report.get("counterfactual", {}).get("different_image") is True, "camera smoke did not isolate target pose in same scene")
    return {
        "status": "PASS",
        "path": str(report_path),
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in (report_path, root / "stage21_observation_target_a.npz", root / "stage21_observation_target_b.npz")],
        "same_scene": same_scene,
        "target_a_world_m": TARGET_POSITION_WORLD_M.tolist(),
        "target_b_world_m": [0.30, -0.20, 0.58],
        "red_pixel_counts": visibility,
        "recomputed_images": images,
        "claim_boundary": "native renderer places and visibly renders the actor in the same body/wrist camera scene; visibility depends on camera pose and robot state (not every Stage41 frame must contain a visible pixel)",
    }


def audit_g0(
    *,
    repo_root: Path,
    wbc_source_root: Path,
    run_a: Path,
    seal_a: Path,
    run_b: Path,
    seal_b: Path,
    predecessor_seal: Path,
    camera_root: Path,
) -> Dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    _require(_git(repo, "branch", "--show-current") == BRANCH, "G0 must run on authorized Stage43 branch")
    _require(_is_ancestor(repo, STAGE42_HEAD, _git(repo, "rev-parse", "HEAD")), "Stage43 branch must descend from accepted Stage42 HEAD")
    _require(not _git(repo, "status", "--porcelain", "--untracked-files=all"), "G0 repository must be clean")
    pred_info = stage40.verify_stage_seal(Path(predecessor_seal), Path(predecessor_seal).parent, expected_stage="M2", _admit_without_predecessor=True)
    _require(pred_info["seal_sha256"] == M2_SEAL, "M2 predecessor seal drift")
    snapshot_a = _load_run(run_a, seal_a, predecessor_seal, STAGE41_A_SEAL)
    snapshot_b = _load_run(run_b, seal_b, predecessor_seal, STAGE41_B_SEAL)
    _require(snapshot_a["seal"]["seal_sha256"] == STAGE41_A_SEAL and snapshot_b["seal"]["seal_sha256"] == STAGE41_B_SEAL, "Stage41 predecessor identities drift")
    metrics_a = _target_metrics(snapshot_a)
    metrics_b = _target_metrics(snapshot_b)
    _require(metrics_a == metrics_b, "sealed LH10 A/B target metrics disagree")
    source = _source_semantics_audit(repo, Path(wbc_source_root).resolve(strict=True))
    model_input = _ast_model_input_audit(repo)
    camera = _camera_smoke_audit(camera_root)
    target_checks = {
        "rendered_actor_pose_equals_evaluation_target": source["target_actor"]["pose_source"] == TARGET_POSITION_WORLD_M.tolist(),
        "camera_smoke_target_a_equals_evaluation_target": camera["target_a_world_m"] == TARGET_POSITION_WORLD_M.tolist(),
        "stage41_request_targets_equal_evaluation_target": all(item.get("target_position_world_m") == TARGET_POSITION_WORLD_M.tolist() for item in snapshot_a["requests"].values()) and all(item.get("target_position_world_m") == TARGET_POSITION_WORLD_M.tolist() for item in snapshot_b["requests"].values()),
        "target_excluded_from_model_forward": model_input["target_excluded_from_model_forward"],
        "achieved_pos_world_frame": source["achieved_pos_semantics"]["frame"] == "world",
        "same_scene_native_camera_evidence": camera["status"] == "PASS",
    }
    _require(all(target_checks.values()), "CORELAB_STAGE43_TARGET_SEMANTICS_BLOCKED")
    return {
        "schema": "CORELAB_STAGE43_G0_TARGET_AUDIT_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "issue_comment": ISSUE_COMMENT,
        "status": "PASS",
        "token": "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED",
        "repository": {"branch": BRANCH, "head": _git(repo, "rev-parse", "HEAD"), "stage42_head": STAGE42_HEAD},
        "predecessor": {"stage41_head": STAGE41_HEAD, "stage41_bundle_sha256": STAGE41_BUNDLE_SHA256, "lh10_a_seal_sha256": STAGE41_A_SEAL, "lh10_b_seal_sha256": STAGE41_B_SEAL, "m2_seal_sha256": M2_SEAL},
        "target_checks": target_checks,
        "target": {"evaluation_world_m": TARGET_POSITION_WORLD_M.tolist(), "language": LANGUAGE, "source_frame": SOURCE_FRAME, "tcp_semantic_id": TCP_SEMANTIC_ID},
        "source_semantics": source,
        "model_input_audit": model_input,
        "camera_smoke": camera,
        "sealed_stage41": {
            "LH10-A": {"run_root": str(snapshot_a["root"]), "seal": snapshot_a["seal"], "metrics": metrics_a, "request_count": len(snapshot_a["requests"]), "stage41_body_hashes": [sha256_array(snapshot_a["body"][key]) for key in Q_IDS]},
            "LH10-B": {"run_root": str(snapshot_b["root"]), "seal": snapshot_b["seal"], "metrics": metrics_b, "request_count": len(snapshot_b["requests"]), "stage41_body_hashes": [sha256_array(snapshot_b["body"][key]) for key in Q_IDS]},
            "corresponding_image_identity": all(np.array_equal(snapshot_a["body"][key], snapshot_b["body"][key]) and np.array_equal(snapshot_a["wrist"][key], snapshot_b["wrist"][key]) for key in Q_IDS),
        },
        "evaluation_policy": {"binary_success_threshold_m": None, "continuous_metrics_only": True, "target_progress_is_descriptive": True},
        "forbidden_operations": expected_forbidden_operations(),
        "claim_boundary": "the target is a real native same-scene red actor at the evaluation coordinate and is excluded from numeric GR00T input; Stage41 task progress is negative in this frozen replay and is reported descriptively, not re-labeled as success/failure",
    }


def expected_native_schedule(native_dt_s: float) -> Tuple[int, int, int, int]:
    dt = float(native_dt_s)
    _require(math.isfinite(dt) and dt > 0.0, "native dt must be positive finite")
    handoff = int(round(HANDOFF_S / dt))
    _require(handoff > 0, "native handoff schedule produced no steps")
    return (handoff, handoff, handoff, 2 * handoff)


def validate_g1_static_contract(*, g0: Mapping[str, Any], repo_root: Path, expected_native_dt_s: Optional[float] = None) -> Dict[str, Any]:
    _require(g0.get("schema") == "CORELAB_STAGE43_G0_TARGET_AUDIT_V1" and g0.get("status") == "PASS", "G0 is not PASS")
    _require(g0.get("token") == "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED", "wrong G0 token")
    repo = Path(repo_root).resolve(strict=True)
    _require(_git(repo, "branch", "--show-current") == BRANCH, "G1 branch drift")
    _require(_is_ancestor(repo, STAGE42_HEAD, _git(repo, "rev-parse", "HEAD")), "G1 predecessor HEAD drift")
    _require(tuple(g0.get("target", {}).get("evaluation_world_m", ())) == tuple(TARGET_POSITION_WORLD_M.tolist()), "G1 target drift")
    schedule = expected_native_schedule(expected_native_dt_s) if expected_native_dt_s is not None else COHORT_SEGMENT_SCHEDULE
    _require(len(schedule) == 4 and all(int(item) > 0 for item in schedule), "G1 segment schedule invalid")
    _require(tuple(COHORT_ORDER) == ("C0_NOMINAL", "C1_X_PLUS_5CM", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM"), "condition order drift")
    for condition in COHORT_ORDER:
        delta = np.asarray(COHORT_DELTAS_M[condition], dtype=np.float64)
        _require(delta.shape == (3,) and np.isfinite(delta).all(), "condition delta invalid")
        _require(float(delta[2]) == 0.0, "condition changes base Z")
        _require(condition == "C0_NOMINAL" or np.isclose(np.linalg.norm(delta[:2]), 0.05, rtol=0.0, atol=1e-12), "condition magnitude drift")
    return {
        "schema": "CORELAB_STAGE43_G1_STATIC_CONTRACT_V1",
        "stage": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE43_COHORT_HARNESS_READY",
        "condition_order": list(COHORT_ORDER),
        "conditions": {key: {"delta_base_xy_m": list(COHORT_DELTAS_M[key]), "variable_changed": "pre_episode_robot_base_xy_only"} for key in COHORT_ORDER},
        "schedule": {"segments": list(COHORT_SEGMENT_LABELS), "native_ticks": list(schedule), "gr00t_forwards_per_condition": COHORT_FORWARD_COUNT, "reference_updates_per_condition": COHORT_REFERENCE_UPDATE_COUNT, "nominal_total_wbc_ticks": int(sum(schedule)), "native_schedule_derivation": "ROUND_HANDOFF_S_OVER_OBSERVED_NATIVE_DT"},
        "total_budget": {"episodes": COHORT_TOTAL_COUNT, "gr00t_forwards": COHORT_TOTAL_FORWARDS, "reference_updates": COHORT_TOTAL_UPDATES, "wbc_ticks": int(COHORT_TOTAL_COUNT * sum(schedule))},
        "invariants": {"target_fixed_world_m": TARGET_POSITION_WORLD_M.tolist(), "language": LANGUAGE, "seed": DEV_SEED, "no_orientation_velocity_joint_change": True, "no_settling_steps": True, "base_set_once_before_S0": True, "physics_only_after_S0": True, "fresh_observation_at_each_q": True, "stop_remaining_cohort_on_first_failure": True},
        "forbidden_operations": expected_forbidden_operations(),
    }


def _regular_file_identity(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Return a non-symlink file identity for a runnable input."""

    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), "runnable input is not a regular file: %s" % candidate)
    digest = sha256_file(candidate)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "runnable input hash mismatch: %s" % candidate)
    return {"path": str(candidate.resolve()), "sha256": digest, "size_bytes": int(candidate.stat().st_size)}


def validate_stage43_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the immutable Stage43 bundle without importing runtime code."""

    _require(bundle.get("schema") == BUNDLE_SCHEMA, "wrong Stage43 execution bundle schema")
    _require(bundle.get("stage_id") == STAGE_ID, "wrong Stage43 bundle stage")
    _require(bundle.get("pi_gate") == PI_GATE and bundle.get("issue_comment") == ISSUE_COMMENT, "wrong Stage43 PI gate")
    _require(bundle.get("classification") == "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE", "wrong Stage43 classification")
    _require(bundle.get("execution_class") == "CORELAB_STAGE43_CONTROLLED_INITIAL_CONDITION_COHORT_V1", "wrong Stage43 execution class")
    _require(bundle.get("dev_seed") == DEV_SEED, "Stage43 seed drift")
    repository = bundle.get("repository")
    _require(isinstance(repository, dict) and repository.get("branch") == BRANCH, "Stage43 repository branch drift")
    _require(isinstance(repository.get("head"), str) and len(repository["head"]) == 40, "invalid Stage43 repository HEAD")
    _require(repository.get("stage42_head") == STAGE42_HEAD, "Stage42 predecessor HEAD drift")
    target = bundle.get("target")
    _require(isinstance(target, dict), "Stage43 target contract missing")
    _require(target.get("evaluation_world_m") == TARGET_POSITION_WORLD_M.tolist(), "Stage43 target drift")
    _require(target.get("language") == LANGUAGE and target.get("source_frame") == SOURCE_FRAME and target.get("tcp_semantic_id") == TCP_SEMANTIC_ID, "Stage43 task contract drift")
    _require(target.get("numeric_xyz_in_model_input") is False, "Stage43 hidden numeric target is not excluded")
    protocol = bundle.get("protocol")
    _require(isinstance(protocol, dict), "Stage43 protocol missing")
    _require(protocol.get("handoff_s") == HANDOFF_S and protocol.get("action_dt_s") == ACTION_DT_S, "Stage43 timing contract drift")
    _require(protocol.get("segment_native_ticks") == list(COHORT_SEGMENT_SCHEDULE), "Stage43 segment schedule drift")
    _require(protocol.get("gr00t_forward_count_per_condition") == COHORT_FORWARD_COUNT and protocol.get("reference_update_count_per_condition") == COHORT_REFERENCE_UPDATE_COUNT, "Stage43 per-condition budget drift")
    conditions = bundle.get("conditions")
    _require(isinstance(conditions, list) and [item.get("condition_id") for item in conditions] == list(COHORT_ORDER), "Stage43 condition order drift")
    for item in conditions:
        _require(isinstance(item, dict), "Stage43 condition record is not an object")
        condition_id = str(item.get("condition_id"))
        _require(condition_id in COHORT_DELTAS_M, "unknown Stage43 condition")
        delta = item.get("delta_base_xy_m")
        _require(delta == list(COHORT_DELTAS_M[condition_id]), "Stage43 condition delta drift")
        _require(item.get("variable_changed") == "pre_episode_robot_base_xy_only", "Stage43 condition variable drift")
    budgets = bundle.get("budgets")
    _require(isinstance(budgets, dict), "Stage43 budgets missing")
    expected_budgets = {"episodes": COHORT_TOTAL_COUNT, "gr00t_forward_count": COHORT_TOTAL_FORWARDS, "reference_update_count": COHORT_TOTAL_UPDATES, "wbc_step_count": COHORT_TOTAL_COUNT * sum(COHORT_SEGMENT_SCHEDULE)}
    _require(budgets == expected_budgets, "Stage43 total budget drift")
    lineage = bundle.get("executable_lineage")
    _require(isinstance(lineage, dict), "Stage43 executable lineage missing")
    for key in ("runner_relative_path", "runner_sha256", "source_relative_path", "source_sha256"):
        _require(isinstance(lineage.get(key), str) and lineage[key], "Stage43 executable lineage field missing: %s" % key)
    for key in ("runner_sha256", "source_sha256"):
        _require(len(lineage[key]) == 64, "Stage43 executable lineage hash invalid: %s" % key)
    predecessors = bundle.get("predecessor")
    _require(isinstance(predecessors, dict), "Stage43 predecessor binding missing")
    _require(predecessors.get("stage42_head") == STAGE42_HEAD and predecessors.get("stage41_head") == STAGE41_HEAD, "Stage43 predecessor lineage drift")
    _require(predecessors.get("m2_seal_sha256") == M2_SEAL, "Stage43 M2 predecessor seal drift")
    g0 = bundle.get("g0")
    g1 = bundle.get("g1")
    _require(isinstance(g0, dict) and g0.get("status") == "PASS" and g0.get("token") == "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED", "Stage43 G0 binding is not PASS")
    _require(isinstance(g1, dict) and g1.get("status") == "PASS" and g1.get("token") == "CORELAB_STAGE43_COHORT_HARNESS_READY", "Stage43 G1 binding is not PASS")
    for item in (g0, g1):
        _require(isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64 and isinstance(item.get("path"), str), "Stage43 audit binding is incomplete")
    artifacts = bundle.get("artifacts")
    _require(isinstance(artifacts, dict), "Stage43 frozen artifact identities missing")
    expected_artifact_keys = {"structured_checkpoint", "structured_stats", "tiny1_dataset", "modality", "cosmos_snapshot", "trajectory", "wbc_source", "wbc_arm_checkpoint", "wbc_dog_checkpoint", "stage10_root", "base_model_path", "model_path", "dataset_dir", "modality_config", "cosmos_snapshot_path"}
    _require(expected_artifact_keys.issubset(set(artifacts)), "Stage43 frozen artifact identity set is incomplete")
    _require(artifacts["structured_checkpoint"].get("tree_sha256") == STRUCTURED_CHECKPOINT_TREE_SHA256, "STRUCTURED@1000 tree identity drift")
    _require(artifacts["structured_stats"].get("sha256") == STRUCTURED_STATS_SHA256, "structured stats identity drift")
    _require(artifacts["tiny1_dataset"].get("tree_sha256") == TINY1_TREE_SHA256, "TINY1 tree identity drift")
    _require(artifacts["modality"].get("sha256") == STAGE22_MODALITY_SHA256, "modality identity drift")
    _require(artifacts["cosmos_snapshot"].get("revision") == COSMOS_REVISION and artifacts["cosmos_snapshot"].get("tree_sha256") == COSMOS_TREE_SHA256, "Cosmos identity drift")
    _require(artifacts["trajectory"].get("sha256") == TRAJECTORY_SHA256, "trajectory identity drift")
    _require(artifacts["wbc_source"].get("tree_sha256") == WBC_TREE_SHA256, "WBC source identity drift")
    _require(artifacts["wbc_arm_checkpoint"].get("sha256") == WBC_ARM_SHA256 and artifacts["wbc_dog_checkpoint"].get("sha256") == WBC_DOG_SHA256, "WBC checkpoint identity drift")
    forbidden = bundle.get("forbidden_operations")
    _require(isinstance(forbidden, dict), "Stage43 forbidden-operation map missing")
    for name in FORBIDDEN_FIELDS:
        _require(forbidden.get(name) is False, "Stage43 forbidden operation is enabled: %s" % name)
    return {"stage_id": STAGE_ID, "status": "PASS", "schema": BUNDLE_SCHEMA}


def verify_stage43_bundle_files(bundle: Mapping[str, Any], *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify G0/G1 and executable hashes referenced by a frozen bundle."""

    validate_stage43_bundle(bundle)
    for key, expected_status, expected_token in (("g0", "PASS", "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED"), ("g1", "PASS", "CORELAB_STAGE43_COHORT_HARNESS_READY")):
        item = bundle[key]
        path = Path(str(item["path"]))
        _require(path.is_file() and not path.is_symlink(), "Stage43 audit file is missing: %s" % path)
        _require(sha256_file(path) == item["sha256"], "Stage43 audit hash mismatch: %s" % key)
        payload = read_json(path)
        _require(payload.get("status") == expected_status and payload.get("token") == expected_token, "Stage43 audit token mismatch: %s" % key)
    if repo_root is not None:
        root = Path(repo_root).resolve(strict=True)
        runner = root / str(bundle["executable_lineage"]["runner_relative_path"])
        source = root / str(bundle["executable_lineage"]["source_relative_path"])
        _require(sha256_file(runner) == bundle["executable_lineage"]["runner_sha256"], "Stage43 runner hash drift")
        _require(sha256_file(source) == bundle["executable_lineage"]["source_sha256"], "Stage43 source hash drift")
    return {"status": "PASS", "g0_sha256": bundle["g0"]["sha256"], "g1_sha256": bundle["g1"]["sha256"]}


def build_condition_manifest(evidence_root: Path, role_paths: Mapping[str, str]) -> Dict[str, Any]:
    """Hash the complete per-condition evidence directory."""

    root = Path(evidence_root).resolve(strict=True)
    roles = {str(key): str(value) for key, value in role_paths.items()}
    _require(len(roles) >= 18, "Stage43 condition evidence is incomplete")
    _require(len(set(roles.values())) == len(roles), "Stage43 condition roles reuse one file")
    records = []
    for role, relative in sorted(roles.items()):
        path = _resolved_child(root, relative)
        records.append({"role": role, "relative_path": Path(relative).as_posix(), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return {"schema": CONDITION_MANIFEST_SCHEMA, "stage_id": STAGE_ID, "records": records, "record_count": len(records)}


def verify_condition_manifest(manifest: Mapping[str, Any], evidence_root: Path) -> Dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    _require(manifest.get("schema") == CONDITION_MANIFEST_SCHEMA and manifest.get("stage_id") == STAGE_ID, "wrong Stage43 condition manifest")
    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == int(manifest.get("record_count", -1)), "Stage43 condition record count mismatch")
    roles = []
    paths = []
    for record in records:
        _require(isinstance(record, dict), "Stage43 condition manifest record is not an object")
        role = str(record.get("role", "")); relative = str(record.get("relative_path", ""))
        path = _resolved_child(root, relative)
        _require(int(record.get("size_bytes", -1)) == path.stat().st_size and str(record.get("sha256")) == sha256_file(path), "Stage43 condition evidence hash mismatch: %s" % role)
        roles.append(role); paths.append(relative)
    _require(len(roles) == len(set(roles)) and len(paths) == len(set(paths)), "Stage43 condition manifest has duplicate roles/paths")
    _require(len(roles) >= 18, "Stage43 condition manifest is incomplete")
    return {"status": "PASS", "record_count": len(records)}


def prepare_stage43_bundle(
    *,
    repo: Path,
    g0_path: Path,
    g1_path: Path,
    runner: Path,
    output: Path,
    structured_checkpoint: Path,
    structured_stats: Path,
    tiny1_dataset: Path,
    modality: Path,
    cosmos_snapshot: Path,
    trajectory: Path,
    wbc_source: Path,
    arm_checkpoint: Path,
    dog_checkpoint: Path,
    stage10_root: Path,
    base_model_path: Path,
    model_path: Path,
    dataset_dir: Path,
    modality_config: Path,
    require_remote: bool = True,
) -> Dict[str, Any]:
    """Freeze the Stage43 bundle after G0/G1 and before any GPU import."""

    repo = Path(repo).resolve(strict=True)
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    _require(branch == BRANCH, "Stage43 bundle must use the authorized branch")
    _require(len(head) == 40 and _is_ancestor(repo, STAGE42_HEAD, head), "Stage43 repository lineage is invalid")
    _require(not _git(repo, "status", "--porcelain", "--untracked-files=all"), "Stage43 repository is not clean")
    if require_remote:
        remote_line = _git(repo, "ls-remote", "--heads", "origin", branch)
        _require(remote_line.split()[0] == head, "Stage43 HEAD is not pushed")
    g0_path = Path(g0_path).resolve(strict=True); g1_path = Path(g1_path).resolve(strict=True)
    g0 = read_json(g0_path); g1 = read_json(g1_path)
    _require(g0.get("status") == "PASS" and g0.get("token") == "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED", "G0 is not PASS")
    _require(g1.get("status") == "PASS" and g1.get("token") == "CORELAB_STAGE43_COHORT_HARNESS_READY", "G1 is not PASS")
    _require(g0.get("target", {}).get("evaluation_world_m") == TARGET_POSITION_WORLD_M.tolist(), "G0 target drift")
    _require(g0.get("repository", {}).get("stage42_head") == STAGE42_HEAD, "G0 Stage42 lineage drift")
    runner = Path(runner).resolve(strict=True)
    source = Path(__file__).resolve(strict=True)
    _git(repo, "ls-files", "--error-unmatch", runner.relative_to(repo).as_posix())
    _git(repo, "ls-files", "--error-unmatch", source.relative_to(repo).as_posix())
    def directory_identity(path: Path, **values: Any) -> Dict[str, Any]:
        candidate = Path(path).resolve(strict=True)
        _require(candidate.is_dir() and not candidate.is_symlink(), "runnable directory is invalid: %s" % candidate)
        result = {"path": str(candidate)}; result.update(values); return result
    artifacts: Dict[str, Any] = {
        "structured_checkpoint": directory_identity(structured_checkpoint, tree_sha256=STRUCTURED_CHECKPOINT_TREE_SHA256, label="STRUCTURED@1000"),
        "structured_stats": _regular_file_identity(structured_stats, STRUCTURED_STATS_SHA256),
        "tiny1_dataset": directory_identity(tiny1_dataset, tree_sha256=TINY1_TREE_SHA256, label="TINY1"),
        "modality": _regular_file_identity(modality, STAGE22_MODALITY_SHA256),
        "cosmos_snapshot": directory_identity(cosmos_snapshot, revision=COSMOS_REVISION, tree_sha256=COSMOS_TREE_SHA256, label="Cosmos-Reason2-2B"),
        "trajectory": _regular_file_identity(trajectory, TRAJECTORY_SHA256),
        "wbc_source": directory_identity(wbc_source, tree_sha256=WBC_TREE_SHA256),
        "wbc_arm_checkpoint": _regular_file_identity(arm_checkpoint, WBC_ARM_SHA256),
        "wbc_dog_checkpoint": _regular_file_identity(dog_checkpoint, WBC_DOG_SHA256),
        "stage10_root": directory_identity(stage10_root),
        "base_model_path": directory_identity(base_model_path),
        "model_path": directory_identity(model_path, tree_sha256=STRUCTURED_CHECKPOINT_TREE_SHA256, label="STRUCTURED@1000"),
        "dataset_dir": directory_identity(dataset_dir, tree_sha256=TINY1_TREE_SHA256, label="TINY1"),
        "modality_config": _regular_file_identity(modality_config, STAGE22_MODALITY_SHA256),
        "cosmos_snapshot_path": directory_identity(cosmos_snapshot, revision=COSMOS_REVISION),
    }
    bundle: Dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "stage_id": STAGE_ID,
        "pi_gate": PI_GATE,
        "issue_comment": ISSUE_COMMENT,
        "classification": "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE",
        "execution_class": "CORELAB_STAGE43_CONTROLLED_INITIAL_CONDITION_COHORT_V1",
        "dev_seed": DEV_SEED,
        "repository": {"branch": branch, "head": head, "stage42_head": STAGE42_HEAD},
        "executable_lineage": {
            "runner_relative_path": runner.relative_to(repo).as_posix(), "runner_sha256": sha256_file(runner),
            "source_relative_path": source.relative_to(repo).as_posix(), "source_sha256": sha256_file(source),
        },
        "predecessor": {"stage42_head": STAGE42_HEAD, "stage41_head": STAGE41_HEAD, "stage41_bundle_sha256": STAGE41_BUNDLE_SHA256, "lh10_a_seal_sha256": STAGE41_A_SEAL, "lh10_b_seal_sha256": STAGE41_B_SEAL, "m2_seal_sha256": M2_SEAL},
        "g0": {"path": str(g0_path), "sha256": sha256_file(g0_path), "status": g0["status"], "token": g0["token"]},
        "g1": {"path": str(g1_path), "sha256": sha256_file(g1_path), "status": g1["status"], "token": g1["token"]},
        "target": {"evaluation_world_m": TARGET_POSITION_WORLD_M.tolist(), "language": LANGUAGE, "source_frame": SOURCE_FRAME, "tcp_semantic_id": TCP_SEMANTIC_ID, "numeric_xyz_in_model_input": False},
        "protocol": {"handoff_s": HANDOFF_S, "action_dt_s": ACTION_DT_S, "segment_native_ticks": list(COHORT_SEGMENT_SCHEDULE), "gr00t_forward_count_per_condition": COHORT_FORWARD_COUNT, "reference_update_count_per_condition": COHORT_REFERENCE_UPDATE_COUNT, "wbc_ticks_per_condition": int(sum(COHORT_SEGMENT_SCHEDULE)), "execution_order": list(COHORT_ORDER), "stop_on_first_failure": True, "initial_condition_mutation": "set base XY exactly once before S0; no settling or post-S0 state mutation"},
        "conditions": [{"condition_id": key, "delta_base_xy_m": list(COHORT_DELTAS_M[key]), "variable_changed": "pre_episode_robot_base_xy_only"} for key in COHORT_ORDER],
        "budgets": {"episodes": COHORT_TOTAL_COUNT, "gr00t_forward_count": COHORT_TOTAL_FORWARDS, "reference_update_count": COHORT_TOTAL_UPDATES, "wbc_step_count": COHORT_TOTAL_COUNT * sum(COHORT_SEGMENT_SCHEDULE)},
        "artifacts": artifacts,
        "forbidden_operations": expected_forbidden_operations(),
    }
    validate_stage43_bundle(bundle)
    digest = write_canonical_json(Path(output), bundle)
    return {"status": "PASS", "stage": STAGE_ID, "bundle_sha256": digest, "path": str(output), "repository_head": head}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("g0")
    audit.add_argument("--repo", type=Path, required=True)
    audit.add_argument("--wbc-source-root", type=Path, required=True)
    audit.add_argument("--run-a", type=Path, required=True)
    audit.add_argument("--seal-a", type=Path, required=True)
    audit.add_argument("--run-b", type=Path, required=True)
    audit.add_argument("--seal-b", type=Path, required=True)
    audit.add_argument("--predecessor-seal", type=Path, required=True)
    audit.add_argument("--camera-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    g1 = sub.add_parser("g1")
    g1.add_argument("--repo", type=Path, required=True)
    g1.add_argument("--g0", type=Path, required=True)
    g1.add_argument("--output", type=Path, required=True)
    g1.add_argument("--native-dt-s", type=float)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--repo", type=Path, required=True)
    bundle.add_argument("--g0", type=Path, required=True)
    bundle.add_argument("--g1", type=Path, required=True)
    bundle.add_argument("--runner", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--structured-checkpoint", type=Path, required=True)
    bundle.add_argument("--structured-stats", type=Path, required=True)
    bundle.add_argument("--tiny1-dataset", type=Path, required=True)
    bundle.add_argument("--modality", type=Path, required=True)
    bundle.add_argument("--cosmos-snapshot", type=Path, required=True)
    bundle.add_argument("--trajectory", type=Path, required=True)
    bundle.add_argument("--wbc-source", type=Path, required=True)
    bundle.add_argument("--arm-checkpoint", type=Path, required=True)
    bundle.add_argument("--dog-checkpoint", type=Path, required=True)
    bundle.add_argument("--stage10-root", type=Path, required=True)
    bundle.add_argument("--base-model-path", type=Path, required=True)
    bundle.add_argument("--model-path", type=Path, required=True)
    bundle.add_argument("--dataset-dir", type=Path, required=True)
    bundle.add_argument("--modality-config", type=Path, required=True)
    bundle.add_argument("--no-remote-check", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "g0":
        result = audit_g0(repo_root=args.repo, wbc_source_root=args.wbc_source_root, run_a=args.run_a, seal_a=args.seal_a, run_b=args.run_b, seal_b=args.seal_b, predecessor_seal=args.predecessor_seal, camera_root=args.camera_root)
    elif args.command == "g1":
        result = validate_g1_static_contract(g0=read_json(args.g0), repo_root=args.repo, expected_native_dt_s=args.native_dt_s)
    else:
        result = prepare_stage43_bundle(repo=args.repo, g0_path=args.g0, g1_path=args.g1, runner=args.runner, output=args.output, structured_checkpoint=args.structured_checkpoint, structured_stats=args.structured_stats, tiny1_dataset=args.tiny1_dataset, modality=args.modality, cosmos_snapshot=args.cosmos_snapshot, trajectory=args.trajectory, wbc_source=args.wbc_source, arm_checkpoint=args.arm_checkpoint, dog_checkpoint=args.dog_checkpoint, stage10_root=args.stage10_root, base_model_path=args.base_model_path, model_path=args.model_path, dataset_dir=args.dataset_dir, modality_config=args.modality_config, require_remote=not args.no_remote_check)
    if args.command == "bundle":
        print(json.dumps(result, sort_keys=True))
        return 0
    digest = write_canonical_json(args.output, result)
    print(json.dumps({"status": result["status"], "token": result["token"], "path": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
