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
    args = parser.parse_args(argv)
    if args.command == "g0":
        result = audit_g0(repo_root=args.repo, wbc_source_root=args.wbc_source_root, run_a=args.run_a, seal_a=args.seal_a, run_b=args.run_b, seal_b=args.seal_b, predecessor_seal=args.predecessor_seal, camera_root=args.camera_root)
    else:
        result = validate_g1_static_contract(g0=read_json(args.g0), repo_root=args.repo, expected_native_dt_s=args.native_dt_s)
    digest = write_canonical_json(args.output, result)
    print(json.dumps({"status": result["status"], "token": result["token"], "path": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
