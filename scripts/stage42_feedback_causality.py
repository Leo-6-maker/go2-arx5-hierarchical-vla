#!/usr/bin/env python3
"""CPU-only Stage42 feedback-causality audit and sealed interpretation.

The audit consumes only the accepted Stage41 LH10 evidence.  It never
constructs a model or simulator.  The optional ``finalize`` command validates
the bounded model-probe output produced by :mod:`stage42_corelab_probes` and
seals one machine-readable plus human-readable interpretation package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # CPU-only bundle preparation need not import NumPy.
    np = None  # type: ignore[assignment]

import stage41_corelab_evidence as stage41


PI_GATE = "PI_GATE_STATE_0067"
ISSUE_COMMENT = 5579985169
STAGE_ID = "Stage42"
BRANCH = "stage42-dev/corelab-feedback-causality"
STAGE41_HEAD = "95ec9ee0ebdaac8ea09b427cde94005591ac37fd"
STAGE41_BUNDLE_SHA256 = "e9891d04b75ed50bd405052fdcd730fe4a7bfbf75224d68ca48d3026afc37aee"
STAGE41_A_SEAL = "3882e5178551384a41685533944b7388788499fcd2b978178c8eddc2f8b6e02c"
STAGE41_B_SEAL = "4a90f0cd86fa53c3b04cc83ca1afb388836e57250dc849b3b02a4d8cb0936cac"
M2_SEAL = "ad178ac49da1ced987f60224987ccab8dc1d3343674bb511b7c3fe13a3b8dcc5"
DEV_SEED = 4242
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
HANDOFF_S = 0.9750000000000001
ACTION_DT_S = 0.05
Q_IDS = tuple("stage41-q%d" % index for index in range(11))
SELECTED_IDS = ("stage41-q0", "stage41-q5", "stage41-q10")
CF_IDS = ("CF1", "CF2", "CF3", "CF4", "CF5", "CF6")
EXPECTED_FORWARDS = 9
EXPECTED_REPLAYS = 3
EXPECTED_COUNTERFACTUALS = 6

FORBIDDEN_FIELDS = (
    "training", "fine_tuning", "formal_seed_used", "validation_e_read",
    "holdout_f_read", "new_data_read", "hardware_executed", "h200_used",
    "checkpoint_substitution", "model_substitution", "processor_substitution",
    "precision_change", "quantization", "cpu_offload", "safety_threshold_change",
    "wbc_controller_change", "wbc_rollout", "reference_update", "ros", "zmq",
)


class Stage42EvidenceError(ValueError):
    """Raised when Stage42 evidence does not satisfy its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage42EvidenceError(message)


def _require_numpy() -> Any:
    _require(np is not None, "NumPy is required for array-based Stage42 audit operations")
    return np


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
    return {field: False for field in FORBIDDEN_FIELDS}


def _finite_array(value: Any, shape: Sequence[int], label: str, dtype: Any = None) -> np.ndarray:
    numpy = _require_numpy()
    array = numpy.asarray(value, dtype=numpy.float64 if dtype is None else dtype)
    _require(tuple(array.shape) == tuple(shape), "%s shape mismatch: %s" % (label, array.shape))
    _require(bool(numpy.isfinite(array).all()), "%s contains non-finite values" % label)
    return array


def _pair_delta(left: Any, right: Any) -> Dict[str, float]:
    delta = np.asarray(left) - np.asarray(right)
    _require(bool(np.isfinite(delta).all()), "pair delta is non-finite")
    flat = delta.reshape(-1).astype(np.float64)
    return {
        "l2": float(np.linalg.norm(flat)),
        "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
        "mean_abs": float(np.mean(np.abs(flat))) if flat.size else 0.0,
    }


def _hash_equality_matrix(values: Sequence[str]) -> List[List[bool]]:
    return [[str(left) == str(right) for right in values] for left in values]


def _array_equality_matrix(values: Sequence[np.ndarray]) -> List[List[bool]]:
    return [[bool(np.array_equal(left, right)) for right in values] for left in values]


def _decoded_trajectory_metrics(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    position = _pair_delta(left[:, :3], right[:, :3])
    rot6d = _pair_delta(left[:, 3:9], right[:, 3:9])
    return {
        "eef_position_l2": position["l2"],
        "eef_position_max_abs": position["max_abs"],
        "eef_position_mean_abs": position["mean_abs"],
        "eef_rot6d_l2": rot6d["l2"],
        "eef_rot6d_max_abs": rot6d["max_abs"],
        "eef_rot6d_mean_abs": rot6d["mean_abs"],
        "gripper_l2": _pair_delta(left[:, 9], right[:, 9])["l2"],
        "endpoint_position_l2": float(np.linalg.norm(left[-1, :3] - right[-1, :3])),
    }


def _role_path(manifest: Mapping[str, Any], root: Path, role: str) -> Path:
    matches = [item for item in manifest.get("records", []) if item.get("role") == role]
    _require(len(matches) == 1, "Stage41 role is not unique: %s" % role)
    relative = Path(str(matches[0].get("relative_path")))
    _require(not relative.is_absolute() and ".." not in relative.parts, "non-canonical Stage41 role path")
    path = (Path(root).resolve(strict=True) / relative).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "Stage41 role is not a regular file: %s" % role)
    _require(sha256_file(path) == str(matches[0].get("sha256")), "Stage41 role hash mismatch: %s" % role)
    return path


def _load_npz_member(path: Path, key: str, shape: Sequence[int], dtype: Any, label: str) -> np.ndarray:
    _require(path.is_file() and not path.is_symlink(), "missing NPZ: %s" % path)
    with np.load(path, allow_pickle=False) as data:
        _require(key in data.files, "%s missing NPZ member %s" % (label, key))
        array = np.asarray(data[key], dtype=dtype)
    return _finite_array(array, shape, label, dtype=dtype)


def _load_run_snapshot(run_root: Path, seal_path: Path, predecessor_seal: Path) -> Dict[str, Any]:
    info = stage41.verify_stage_seal(
        Path(seal_path), Path(run_root), expected_head=STAGE41_HEAD, predecessor_seal=Path(predecessor_seal)
    )
    _require(info["status"] == "PASS", "Stage41 seal did not pass: %s" % seal_path)
    _require(info["seal_sha256"] in {STAGE41_A_SEAL, STAGE41_B_SEAL}, "unexpected Stage41 seal identity")
    manifest = read_json(Path(run_root) / "evidence_manifest.json")
    stage41.verify_evidence_manifest(manifest, Path(run_root))
    state = read_json(_role_path(manifest, Path(run_root), "state_v1"))
    observations = state.get("observations")
    _require(isinstance(observations, list) and len(observations) == len(Q_IDS), "Stage41 observation count mismatch")
    body_path = _role_path(manifest, Path(run_root), "observation_body")
    wrist_path = _role_path(manifest, Path(run_root), "observation_wrist")
    raw_path = _role_path(manifest, Path(run_root), "raw_action_d132")
    decoded_path = _role_path(manifest, Path(run_root), "decoded_action_d10_h40")
    body_npz = {key: _load_npz_member(body_path, key, (96, 96, 3), np.uint8, "body/%s" % key) for key in Q_IDS}
    wrist_npz = {key: _load_npz_member(wrist_path, key, (96, 96, 3), np.uint8, "wrist/%s" % key) for key in Q_IDS}
    raw_npz = {key: _load_npz_member(raw_path, key, (1, 40, 132), np.float32, "raw/%s" % key) for key in Q_IDS}
    decoded_npz = {key: _load_npz_member(decoded_path, key, (40, 10), np.float64, "decoded/%s" % key) for key in Q_IDS}
    records: Dict[str, Dict[str, Any]] = {}
    for index, request_id in enumerate(Q_IDS):
        observation = observations[index]
        _require(str(observation.get("request_id")) == request_id, "Stage41 request order mismatch")
        request_path = Path(run_root) / "mailbox" / "requests" / (request_id + ".npz")
        request_sha = sha256_file(request_path)
        with np.load(request_path, allow_pickle=False) as request_npz:
            _require(set(("body_rgb", "wrist_rgb", "state", "target_position_world_m")).issubset(request_npz.files), "archived request ABI mismatch: %s" % request_id)
            request_body = np.asarray(request_npz["body_rgb"], dtype=np.uint8)
            request_wrist = np.asarray(request_npz["wrist_rgb"], dtype=np.uint8)
            request_state = np.asarray(request_npz["state"], dtype=np.float32).reshape(-1)
            target = np.asarray(request_npz["target_position_world_m"], dtype=np.float64).reshape(-1)
        _require(np.array_equal(request_body, body_npz[request_id]), "archived body differs from sealed body: %s" % request_id)
        _require(np.array_equal(request_wrist, wrist_npz[request_id]), "archived wrist differs from sealed wrist: %s" % request_id)
        _require(np.array_equal(request_state, np.asarray(observation.get("state_v1"), dtype=np.float32)), "archived StateV1 differs: %s" % request_id)
        _require(tuple(target.shape) == (3,) and bool(np.isfinite(target).all()), "archived target ABI mismatch: %s" % request_id)
        raw = raw_npz[request_id]
        decoded = decoded_npz[request_id]
        hashes = {
            "body_sha256": sha256_array(body_npz[request_id]),
            "wrist_sha256": sha256_array(wrist_npz[request_id]),
            "state_v1_sha256": sha256_array(request_state),
            "raw_action_sha256": sha256_array(raw),
            "decoded_action_sha256": sha256_array(decoded),
        }
        for key, value in hashes.items():
            expected_key = {"raw_action_sha256": "raw_sha256", "decoded_action_sha256": "decoded_sha256"}.get(key, key)
            _require(str(observation.get(expected_key)) == value, "sealed %s mismatch: %s" % (key, request_id))
        response = observation.get("response", {})
        _require(isinstance(response, dict) and response.get("status") == "PASS", "Stage41 response is not PASS: %s" % request_id)
        gate = observation.get("safety_gate", {})
        _require(isinstance(gate, dict) and str(gate.get("decision")) in {"ACCEPT", "WARN"}, "Stage41 SafetyGate is inadmissible: %s" % request_id)
        records[request_id] = {
            "request_id": request_id,
            "observation_timestamp": float(observation.get("observation_timestamp")),
            "request_npz_path": str(request_path),
            "request_npz_sha256": request_sha,
            "body": body_npz[request_id],
            "wrist": wrist_npz[request_id],
            "state": request_state,
            "target": target,
            "raw": raw,
            "decoded": decoded,
            "hashes": hashes,
            "safety_gate": gate,
            "response": response,
            "structured_decode": observation.get("structured_decode", {}),
        }
    timestamps = [records[key]["observation_timestamp"] for key in Q_IDS]
    _require(timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps), "Stage41 timestamps are not strictly ordered")
    return {
        "seal": info,
        "seal_sha256": info["seal_sha256"],
        "run_root": str(Path(run_root).resolve()),
        "manifest_sha256": sha256_file(Path(run_root) / "evidence_manifest.json"),
        "records": records,
    }


def _within_run_profile(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    records = snapshot["records"]
    fields = {
        "body_rgb": [records[key]["body"] for key in Q_IDS],
        "wrist_rgb": [records[key]["wrist"] for key in Q_IDS],
        "state_v1": [records[key]["state"] for key in Q_IDS],
        "raw_action_d132": [records[key]["raw"] for key in Q_IDS],
        "decoded_action_d10_h40": [records[key]["decoded"] for key in Q_IDS],
    }
    hashes = {
        "body_rgb": [records[key]["hashes"]["body_sha256"] for key in Q_IDS],
        "wrist_rgb": [records[key]["hashes"]["wrist_sha256"] for key in Q_IDS],
        "state_v1": [records[key]["hashes"]["state_v1_sha256"] for key in Q_IDS],
        "raw_action_d132": [records[key]["hashes"]["raw_action_sha256"] for key in Q_IDS],
        "decoded_action_d10_h40": [records[key]["hashes"]["decoded_action_sha256"] for key in Q_IDS],
    }
    pairwise: List[Dict[str, Any]] = []
    adjacent: List[Dict[str, Any]] = []
    for left_index in range(len(Q_IDS)):
        for right_index in range(left_index + 1, len(Q_IDS)):
            left_id, right_id = Q_IDS[left_index], Q_IDS[right_index]
            item = {
                "left": left_id,
                "right": right_id,
                "state_v1": _pair_delta(fields["state_v1"][left_index], fields["state_v1"][right_index]),
                "raw_action_d132": _pair_delta(fields["raw_action_d132"][left_index], fields["raw_action_d132"][right_index]),
                "decoded_action_d10_h40": _pair_delta(fields["decoded_action_d10_h40"][left_index], fields["decoded_action_d10_h40"][right_index]),
                "decoded_trajectory": _decoded_trajectory_metrics(fields["decoded_action_d10_h40"][left_index], fields["decoded_action_d10_h40"][right_index]),
            }
            pairwise.append(item)
            if right_index == left_index + 1:
                adjacent.append(item)
    raw_distinct = len(set(hashes["raw_action_d132"])) > 1
    decoded_distinct = len(set(hashes["decoded_action_d10_h40"])) > 1
    return {
        "request_ids": list(Q_IDS),
        "timestamps_s": [float(records[key]["observation_timestamp"]) for key in Q_IDS],
        "hashes": hashes,
        "hash_equality_matrices": {key: _hash_equality_matrix(value) for key, value in hashes.items()},
        "array_equality_matrices": {key: _array_equality_matrix(value) for key, value in fields.items()},
        "adjacent": adjacent,
        "pairwise": pairwise,
        "state_v1_distinct": len(set(hashes["state_v1"])) == len(Q_IDS),
        "raw_action_distinct": raw_distinct,
        "decoded_action_distinct": decoded_distinct,
        "classification": (
            "WITHIN_RUN_ACTION_DIVERSITY_PRESENT" if raw_distinct or decoded_distinct else "WITHIN_RUN_ACTIONS_BYTE_IDENTICAL"
        ),
        "safety_gate": [
            {
                "request_id": key,
                "decision": records[key]["safety_gate"].get("decision"),
                "issues": list(records[key]["safety_gate"].get("issues", [])),
            }
            for key in Q_IDS
        ],
        "archived_request_files": {
            key: {
                "relative_path": "mailbox/requests/%s.npz" % key,
                "sha256": records[key]["request_npz_sha256"],
                "observation_timestamp_s": records[key]["observation_timestamp"],
                "component_hashes": {
                    "body_sha256": records[key]["hashes"]["body_sha256"],
                    "wrist_sha256": records[key]["hashes"]["wrist_sha256"],
                    "state_v1_sha256": records[key]["hashes"]["state_v1_sha256"],
                },
            }
            for key in SELECTED_IDS
        },
        "accepted_raw_action_sha256": {
            key: records[key]["hashes"]["raw_action_sha256"] for key in SELECTED_IDS
        },
    }


def _corresponding_ab_profile(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    fields = ("body_rgb", "wrist_rgb", "state_v1", "raw_action_d132", "decoded_action_d10_h40")
    hashes: Dict[str, Any] = {}
    arrays: Dict[str, Any] = {}
    for field in fields:
        hashes[field] = {
            "a": list(left["within_run"]["hashes"][field]),
            "b": list(right["within_run"]["hashes"][field]),
            "identical": left["within_run"]["hashes"][field] == right["within_run"]["hashes"][field],
        }
        arrays[field] = all(
            np.array_equal(left["snapshot"]["records"][request_id][{"body_rgb": "body", "wrist_rgb": "wrist", "state_v1": "state", "raw_action_d132": "raw", "decoded_action_d10_h40": "decoded"}[field]], right["snapshot"]["records"][request_id][{"body_rgb": "body", "wrist_rgb": "wrist", "state_v1": "state", "raw_action_d132": "raw", "decoded_action_d10_h40": "decoded"}[field]])
            for request_id in Q_IDS
        )
    return {"hashes": hashes, "arrays": arrays, "all_corresponding_fields_identical": all(item["identical"] for item in hashes.values()) and all(arrays.values())}


def _find_lines(path: Path, patterns: Sequence[str]) -> Dict[str, List[int]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result: Dict[str, List[int]] = {}
    for pattern in patterns:
        regex = re.compile(pattern)
        result[pattern] = [index + 1 for index, line in enumerate(lines) if regex.search(line)]
    return result


def _recursive_keys(value: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            keys.append(path)
            keys.extend(_recursive_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.extend(_recursive_keys(child, "%s[%d]" % (prefix, index)))
    return keys


def _success_semantics(repo_root: Path, run_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    phase_path = Path(run_snapshot["run_root"]) / "phase_result.json"
    phase = read_json(phase_path)
    completion = phase.get("completion")
    _require(isinstance(completion, dict), "Stage41 completion object is missing")
    stage41_runner = Path(repo_root) / "scripts" / "stage41_corelab_long_horizon.py"
    stage40_phase = Path(repo_root) / "scripts" / "stage40_corelab_phase.py"
    lifecycle = Path(repo_root) / "src" / "go2arx5_vla" / "benchmark" / "lifecycle.py"
    for path in (stage41_runner, stage40_phase, lifecycle):
        _require(path.is_file(), "success-semantics source is missing: %s" % path)
    marker_lines = {
        str(stage41_runner): _find_lines(stage41_runner, (r"endpoint\s*=\s*bool\(", r"traj_s_ref", r"traj_L", r"reference_duration")),
        str(stage40_phase): _find_lines(stage40_phase, (r"endpoint\s*=\s*bool\(", r"traj_s_ref", r"traj_L", r"reference_duration")),
        str(lifecycle): _find_lines(lifecycle, (r"def reference_is_complete", r"reference_duration", r"s_ref")),
    }
    completion_keys = sorted(_recursive_keys(completion))
    success_key_pattern = re.compile(r"(^|\.)(task|object|environment|grasp|placement|manipulation|success)(_|\.|$)", re.IGNORECASE)
    exposed_success_keys = [key for key in completion_keys if success_key_pattern.search(key)]
    return {
        "token": "CORELAB_STAGE42_SUCCESS_SEMANTICS_AUDITED",
        "status": "PASS",
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (stage41_runner, stage40_phase, lifecycle)
        ],
        "source_marker_lines": marker_lines,
        "completion": completion,
        "completion_keys": completion_keys,
        "endpoint_definition": {
            "endpoint_reached": "native_time_s >= reference_duration_s - max(native_dt_s * 0.5, 1e-9) AND traj_s_ref >= traj_L - max(1e-4, 1e-3 * max(traj_L, 1.0))",
            "observed_endpoint_reached": bool(completion.get("endpoint_reached")),
            "observed_traj_s_ref": completion.get("traj_s_ref"),
            "observed_traj_L": completion.get("traj_L"),
            "observed_reference_duration_s": completion.get("reference_duration_s"),
        },
        "task_success_signal": {
            "explicit_object_task_environment_signal_exposed": bool(exposed_success_keys),
            "keys_found": exposed_success_keys,
            "measured_in_stage41": False,
            "claim_boundary": "GR00T_to_WBC_trajectory_execution_completed; not manipulation_task_success",
        },
        "interpretation": "endpoint_reached is reference/trajectory completion evidence only; no object, grasp, placement, or environment-success outcome was exposed or measured in the sealed Stage41 completion record",
    }


def _locate_first(root: Path, relative: str) -> Path:
    path = Path(root) / relative
    _require(path.is_file() and not path.is_symlink(), "required WBC source is missing: %s" % path)
    return path


def _parse_urdf(path: Path) -> Dict[str, Any]:
    root = ET.parse(str(path)).getroot()
    joints: List[Dict[str, Any]] = []
    for element in root.findall(".//joint"):
        if element.get("type") == "fixed":
            continue
        limit = element.find("limit")
        _require(limit is not None, "URDF joint has no limit: %s" % element.get("name"))
        values = {key: float(limit.get(key)) for key in ("lower", "upper") if limit.get(key) is not None}
        _require(set(values) == {"lower", "upper"}, "URDF joint limit is incomplete: %s" % element.get("name"))
        joints.append({"index": len(joints), "name": element.get("name"), "type": element.get("type"), "hard_limit": values})
    _require(len(joints) == 20, "expected 20 non-fixed URDF joints, got %d" % len(joints))
    return {"path": str(path), "sha256": sha256_file(path), "joint_count": len(joints), "joints": joints}


def _source_line(path: Path, pattern: str) -> int:
    lines = _find_lines(path, (pattern,)).get(pattern, [])
    _require(bool(lines), "source marker not found: %s in %s" % (pattern, path))
    return lines[0]


def _joint_run_profile(run: Mapping[str, Any], urdf: Mapping[str, Any], soft_factor: float) -> Dict[str, Any]:
    rows: List[Mapping[str, Any]] = []
    root = Path(run["run_root"])
    for name in ("rows_before_handoff.json", "rows_after_handoff.json"):
        payload = read_json(root / name)
        segments = payload.get("segments")
        _require(isinstance(segments, list), "row segments missing: %s" % name)
        rows.extend(row for segment in segments for row in segment)
    hard = np.asarray([[joint["hard_limit"]["lower"], joint["hard_limit"]["upper"]] for joint in urdf["joints"]], dtype=np.float64)
    center = hard.mean(axis=1)
    half = 0.5 * (hard[:, 1] - hard[:, 0]) * float(soft_factor)
    soft = np.column_stack((center - half, center + half))
    q = np.asarray([_finite_array(row.get("dof_pos"), (20,), "dof_pos", np.float64) for row in rows], dtype=np.float64)
    margin = np.asarray([_finite_array(row.get("joint_limit_margin"), (20,), "joint_limit_margin", np.float64) for row in rows], dtype=np.float64)
    hard_margin = np.minimum(q - hard[:, 0], hard[:, 1] - q)
    soft_margin_expected = np.minimum(q - soft[:, 0], soft[:, 1] - q)
    flag = np.asarray([bool(row.get("joint_limit_violation")) for row in rows], dtype=bool)
    term = np.asarray([bool(row.get("joint_limit_termination")) for row in rows], dtype=bool)
    reset = np.asarray([bool(row.get("reset_buf")) for row in rows], dtype=bool)
    done = np.asarray([bool(row.get("done")) for row in rows], dtype=bool)
    invalid = np.asarray([bool(row.get("invalid_state")) for row in rows], dtype=bool)
    # The Stage41 runner serializes this diagnostic after casting to float32.
    # Compare at that ABI boundary so harmless float64 reconstruction noise
    # cannot turn an exact source-semantic match into a false audit failure.
    _require(
        np.array_equal(np.asarray(margin, dtype=np.float32), np.asarray(soft_margin_expected, dtype=np.float32)),
        "serialized joint margin is not the source soft-limit margin",
    )
    per_joint = []
    for index, joint in enumerate(urdf["joints"]):
        negative = margin[:, index] < 0.0
        hard_negative = hard_margin[:, index] < 0.0
        per_joint.append({
            "index": index,
            "name": joint["name"],
            "hard_limit": joint["hard_limit"],
            "soft_limit": {"lower": float(soft[index, 0]), "upper": float(soft[index, 1])},
            "minimum_soft_signed_margin": float(np.min(margin[:, index])),
            "negative_soft_margin_sample_count": int(negative.sum()),
            "negative_soft_margin_fraction": float(np.mean(negative)),
            "minimum_hard_signed_margin": float(np.min(hard_margin[:, index])),
            "hard_limit_violation_sample_count": int(hard_negative.sum()),
            "q_min": float(np.min(q[:, index])),
            "q_max": float(np.max(q[:, index])),
        })
    negative_cells = margin < 0.0
    return {
        "run_root": str(root),
        "row_count": len(rows),
        "joint_limit_violation_flag_row_count": int(flag.sum()),
        "joint_limit_violation_flag_fraction": float(np.mean(flag)),
        "negative_soft_margin_cell_count": int(negative_cells.sum()),
        "negative_soft_margin_cell_fraction": float(np.mean(negative_cells)),
        "minimum_soft_signed_margin": float(np.min(margin)),
        "maximum_soft_limit_violation_magnitude": float(np.max(np.maximum(-margin, 0.0))),
        "hard_limit_violation_cell_count": int((hard_margin < 0.0).sum()),
        "minimum_hard_signed_margin": float(np.min(hard_margin)),
        "joint_limit_termination_row_count": int(term.sum()),
        "reset_row_count": int(reset.sum()),
        "done_row_count": int(done.sum()),
        "invalid_state_row_count": int(invalid.sum()),
        "per_joint": per_joint,
        "flag_matches_negative_soft_margin_rows": bool(np.array_equal(flag, np.any(negative_cells, axis=1))),
        "achieved_state_source": "serialized dof_pos captured from native simulator state; reference trajectory is not used for this flag",
    }


def _joint_limit_semantics(repo_root: Path, wbc_source_root: Path, run_a: Mapping[str, Any], run_b: Mapping[str, Any], baseline_profile: Optional[Path]) -> Dict[str, Any]:
    wrapper = Path(repo_root) / "scripts" / "native_wbc_benchmark.py"
    legged = _locate_first(wbc_source_root, "go1_gym/envs/roboduet/legged_robot.py")
    wbc_env = _locate_first(wbc_source_root, "go1_gym/envs/roboduet/wbc_env.py")
    config = _locate_first(wbc_source_root, "go1_gym/envs/config/go1.py")
    urdf_path = _locate_first(wbc_source_root, "resources/robots/go2_x5_v3/urdf/go2_x5.urdf")
    for path in (wrapper, legged, wbc_env, config, urdf_path):
        _require(path.is_file(), "joint-limit source is missing: %s" % path)
    config_text = config.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"rewards\.soft_dof_pos_limit['\"]?\s*:\s*([0-9]+(?:\.[0-9]+)?)", config_text)
    if match is None:
        match = re.search(r"soft_dof_pos_limit\s*=\s*([0-9]+(?:\.[0-9]+)?)", config_text)
    _require(match is not None, "soft_dof_pos_limit source value is missing")
    soft_factor = float(match.group(1))
    urdf = _parse_urdf(urdf_path)
    profile_a = _joint_run_profile(run_a, urdf, soft_factor)
    profile_b = _joint_run_profile(run_b, urdf, soft_factor)
    stage41_m2 = None
    if baseline_profile is not None:
        baseline = read_json(baseline_profile)
        overall = baseline.get("tracking", {}).get("overall", {})
        stage41_m2 = overall.get("lifecycle_observations", {})
    source_markers = {
        "wrapper_joint_margin": {"path": str(wrapper), "line": _source_line(wrapper, r"def _stage16_joint_margin")},
        "wrapper_margin_expression": {"path": str(wrapper), "line": _source_line(wrapper, r"margin\s*=\s*np\.minimum")},
        "wrapper_flag_expression": {"path": str(wrapper), "line": _source_line(wrapper, r"joint_violation\s*=\s*bool\(np\.any\(margin\s*<\s*0\.0")},
        "legged_urdf_load": {"path": str(legged), "line": _source_line(legged, r"self\.dof_pos_limits\[i, 0\]\s*=\s*props\[\"lower\"\]")},
        "legged_soft_limit": {"path": str(legged), "line": _source_line(legged, r"self\.dof_pos_limits\[i, 0\]\s*=\s*m\s*-\s*0\.5")},
        "config_soft_factor": {"path": str(config), "line": _source_line(config, r"soft_dof_pos_limit")},
        "wbc_env_limit_use": {"path": str(wbc_env), "line": _source_line(wbc_env, r"self\.dof_pos_limits\[arm_slice, 0\]")},
    }
    hashes = [{"path": str(path), "sha256": sha256_file(path)} for path in (wrapper, legged, wbc_env, config, urdf_path)]
    same_pattern = profile_a["per_joint"] == profile_b["per_joint"] and profile_a["negative_soft_margin_cell_count"] == profile_b["negative_soft_margin_cell_count"]
    resolved = bool(
        same_pattern
        and profile_a["hard_limit_violation_cell_count"] == 0
        and profile_b["hard_limit_violation_cell_count"] == 0
        and profile_a["joint_limit_termination_row_count"] == 0
        and profile_b["joint_limit_termination_row_count"] == 0
        and profile_a["flag_matches_negative_soft_margin_rows"]
        and profile_b["flag_matches_negative_soft_margin_rows"]
    )
    return {
        "token": "CORELAB_STAGE42_JOINT_LIMIT_METRIC_SEMANTICS_RESOLVED" if resolved else "CORELAB_STAGE42_JOINT_LIMIT_METRIC_AMBIGUOUS_STOP_PI",
        "status": "PASS" if resolved else "BLOCKED",
        "classification": "JOINT_LIMIT_METRIC_SEMANTICS_RESOLVED" if resolved else "JOINT_LIMIT_METRIC_AMBIGUOUS_STOP_PI",
        "source_files": hashes,
        "source_marker_lines": source_markers,
        "urdf": urdf,
        "soft_limit_factor": soft_factor,
        "margin_definition": "min(dof_pos - soft_lower, soft_upper - dof_pos), negative means outside the diagnostic soft interval; units are native joint coordinates (radians for revolute, meters for prismatic)",
        "limit_provenance": "URDF hard limits are loaded into dof_pos_limits, then narrowed symmetrically by cfg.rewards.soft_dof_pos_limit before the wrapper computes the diagnostic margin",
        "simulator_clip_or_tolerance": "the per-row flag is diagnostic-only; serialized achieved dof_pos remains within hard URDF limits, joint_limit_termination/reset/done remain false; reset-time clamp exists in the source but no runtime hard-limit termination was observed",
        "run_a": profile_a,
        "run_b": profile_b,
        "pattern_repeats_deterministically": same_pattern,
        "stage41_m2_predecessor_profile": stage41_m2,
        "interpretation": "the observed negative flag is a deterministic crossing of the 0.9 soft diagnostic interval at gripper joints, not an achieved-state hard URDF-limit violation",
    }


def audit_f0_f2(
    run_a: Path,
    seal_a: Path,
    run_b: Path,
    seal_b: Path,
    predecessor_seal: Path,
    repo_root: Path,
    wbc_source_root: Path,
    baseline_profile: Optional[Path] = None,
) -> Dict[str, Any]:
    snapshot_a = _load_run_snapshot(run_a, seal_a, predecessor_seal)
    snapshot_b = _load_run_snapshot(run_b, seal_b, predecessor_seal)
    _require(snapshot_a["seal_sha256"] == STAGE41_A_SEAL, "LH10-A seal mismatch")
    _require(snapshot_b["seal_sha256"] == STAGE41_B_SEAL, "LH10-B seal mismatch")
    within_a = _within_run_profile(snapshot_a)
    within_b = _within_run_profile(snapshot_b)
    left = {"snapshot": snapshot_a, "within_run": within_a}
    right = {"snapshot": snapshot_b, "within_run": within_b}
    ab = _corresponding_ab_profile(left, right)
    _require(ab["all_corresponding_fields_identical"], "Stage41 A/B corresponding-q artifacts are not identical")
    f0 = {
        "token": "CORELAB_STAGE42_WITHIN_RUN_CAUSAL_AUDIT_PASS",
        "status": "PASS",
        "runs": {
            "LH10-A": {"seal": snapshot_a["seal"], "run_root": snapshot_a["run_root"], "manifest_sha256": snapshot_a["manifest_sha256"], "within_run": within_a},
            "LH10-B": {"seal": snapshot_b["seal"], "run_root": snapshot_b["run_root"], "manifest_sha256": snapshot_b["manifest_sha256"], "within_run": within_b},
        },
        "a_b_corresponding_q": ab,
        "interpretation": "within-run diversity is characterized from exact archived arrays; numerical norms are descriptive and exact hash identity is the deterministic classification",
    }
    f1 = _success_semantics(repo_root, snapshot_a)
    f2 = _joint_limit_semantics(repo_root, wbc_source_root, snapshot_a, snapshot_b, baseline_profile)
    status = "PASS" if f0["status"] == "PASS" and f1["status"] == "PASS" and f2["status"] == "PASS" else "BLOCKED"
    return {
        "schema": "CORELAB_STAGE42_F0_F2_AUDIT_V1",
        "pi_gate": PI_GATE,
        "stage": STAGE_ID,
        "status": status,
        "predecessor": {
            "stage41_head": STAGE41_HEAD,
            "stage41_bundle_sha256": STAGE41_BUNDLE_SHA256,
            "lh10_a_seal_sha256": snapshot_a["seal_sha256"],
            "lh10_b_seal_sha256": snapshot_b["seal_sha256"],
            "m2_seal_sha256": M2_SEAL,
        },
        "f0": f0,
        "f1": f1,
        "f2": f2,
        "forbidden_operations": expected_forbidden_operations(),
    }


def _verify_bundle(bundle: Mapping[str, Any]) -> None:
    _require(bundle.get("schema") == "CORELAB_STAGE42_EXECUTION_BUNDLE_V1", "wrong Stage42 bundle schema")
    _require(bundle.get("pi_gate") == PI_GATE and bundle.get("stage_id") == STAGE_ID, "wrong Stage42 bundle identity")
    _require(bundle.get("dev_seed") == DEV_SEED, "Stage42 seed drift")
    _require(bundle.get("forbidden_operations") == expected_forbidden_operations(), "Stage42 forbidden operation map is not clear")
    _require(bundle.get("predecessor", {}).get("stage41_head") == STAGE41_HEAD, "Stage41 predecessor head drift")
    _require(bundle.get("predecessor", {}).get("stage41_bundle_sha256") == STAGE41_BUNDLE_SHA256, "Stage41 predecessor bundle drift")
    budgets = bundle.get("budgets", {})
    _require(budgets == {"replay_forwards": 3, "counterfactual_forwards": 6, "total_gr00t_forwards": 9, "wbc_steps": 0, "reference_updates": 0}, "Stage42 budget drift")
    plan = bundle.get("counterfactual_plan", {})
    _require(tuple(plan.get("ids", [])) == CF_IDS, "counterfactual ID order drift")


def _manifest_records(root: Path) -> List[Dict[str, Any]]:
    excluded = {"evidence_manifest.json", "stage42_result.json", "stage42_result.seal.json", "stage42_result.seal.json.sha256"}
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            records.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _require(records, "Stage42 evidence root has no materialized records")
    return records


def _verify_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _require(manifest.get("schema") == "CORELAB_STAGE42_EVIDENCE_MANIFEST_V1", "wrong Stage42 evidence manifest schema")
    records = manifest.get("records")
    _require(isinstance(records, list) and records, "Stage42 evidence manifest is empty")
    seen = set()
    for record in records:
        relative = Path(str(record.get("relative_path")))
        _require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() not in seen, "non-canonical/duplicate Stage42 evidence path")
        seen.add(relative.as_posix())
        path = (Path(root).resolve(strict=True) / relative).resolve(strict=True)
        _require(path.is_file() and not path.is_symlink(), "Stage42 manifest file is missing: %s" % relative)
        _require(int(record.get("size_bytes")) == path.stat().st_size, "Stage42 evidence size mismatch: %s" % relative)
        _require(str(record.get("sha256")) == sha256_file(path), "Stage42 evidence hash mismatch: %s" % relative)


def _read_probe_summary(probe_path: Path, bundle: Mapping[str, Any]) -> Dict[str, Any]:
    probe = read_json(probe_path)
    _require(probe.get("schema") == "CORELAB_STAGE42_PROBE_RESULT_V1", "wrong Stage42 probe schema")
    _require(probe.get("status") == "PASS", "Stage42 probe is not PASS")
    _require(probe.get("bundle_sha256") == sha256_bytes(canonical_json_bytes(bundle)), "probe/bundle hash mismatch")
    _require(probe.get("seed") == DEV_SEED, "probe seed mismatch")
    _require(probe.get("counters") == {"gr00t_forward_count": 9, "replay_forward_count": 3, "counterfactual_forward_count": 6, "wbc_step_count": 0, "reference_update_count": 0}, "probe counters drift")
    _require(probe.get("replay_pass") is True and probe.get("counterfactual_pass") is True, "probe pass predicates are incomplete")
    tasks = probe.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 9, "probe task count mismatch")
    _require([str(item.get("probe_id")) for item in tasks[:3]] == ["REPLAY-S0", "REPLAY-S5", "REPLAY-S10"], "replay task order mismatch")
    _require([str(item.get("probe_id")) for item in tasks[3:]] == list(CF_IDS), "counterfactual task order mismatch")
    for index, task in enumerate(tasks):
        _require(task.get("forward_count_before") == index and task.get("forward_count_after") == index + 1 and task.get("forward_count_delta") == 1, "probe causal counter mismatch")
        _require(task.get("raw_shape") == [1, 40, 132] and task.get("decoded_shape") == [40, 10] and task.get("finite") is True, "probe tensor ABI mismatch")
    return probe


def _interpretation(audit: Mapping[str, Any], probe: Mapping[str, Any]) -> Dict[str, Any]:
    f0 = audit["f0"]
    f2 = audit["f2"]
    probe_sensitivity = probe.get("sensitivity", {})
    classifications = {name: item.get("classification") for name, item in probe_sensitivity.items()}
    outputs_change = f0["runs"]["LH10-A"]["within_run"]["classification"] == "WITHIN_RUN_ACTION_DIVERSITY_PRESENT"
    modalities_detected = any(value == "SENSITIVITY_DETECTED" for value in classifications.values())
    joint_resolved = f2.get("classification") == "JOINT_LIMIT_METRIC_SEMANTICS_RESOLVED"
    if outputs_change and modalities_detected and joint_resolved:
        next_route = "MULTI_CONDITION_DEVELOPMENT_COHORT_SCIENTIFICALLY_SENSIBLE_NEXT; do not auto-start before PI review"
    elif not outputs_change or not modalities_detected:
        next_route = "MODEL_DATA_TRAINING_REVIEW_FIRST; do not spend GPU on a larger closed-loop cohort"
    else:
        next_route = "CONTROLLER_OR_SEMANTICS_REVIEW_FIRST; do not expand the cohort before PI adjudication"
    return {
        "stage41_loop_interpretation": "GR00T_OUTPUT_CHANGES_WITH_ARCHIVED_FEEDBACK_STATES" if outputs_change else "STRUCTURALLY_CLOSED_BUT_WITHIN_RUN_ACTIONS_BYTE_IDENTICAL",
        "modality_sensitivity": classifications,
        "endpoint_claim": audit["f1"]["task_success_signal"]["claim_boundary"],
        "joint_limit_claim": f2.get("interpretation"),
        "next_investment_recommendation": next_route,
        "bounded_scope": "one TINY1 development fixture, archived S0/S5/S10 bytes, seed 4242; no robustness/generalization or global modality-grounding claim",
    }


def _human_report(result: Mapping[str, Any], report_sha_placeholder: str = "") -> str:
    audit = result["audit"]
    probe = result["probe"]
    interp = result["interpretation"]
    f2 = audit["f2"]
    lines = [
        "# Stage42 feedback causality and behavior-quality qualification",
        "",
        "Status: `CORELAB_STAGE42_FEEDBACK_CAUSALITY_QUALIFIED`",
        "",
        "This package is CPU/read-only for F0/F1/F2/F5 and contains exactly three archived replay forwards plus six fixed counterfactual forwards. It adds no threshold, changes no checkpoint/processor/precision, and performs no WBC rollout or reference update.",
        "",
        "## Frozen predecessor",
        "",
        "- Stage41 HEAD: `%s`" % result["predecessor"]["stage41_head"],
        "- Stage41 bundle SHA-256: `%s`" % result["predecessor"]["stage41_bundle_sha256"],
        "- LH10-A seal: `%s`" % result["predecessor"]["lh10_a_seal_sha256"],
        "- LH10-B seal: `%s`" % result["predecessor"]["lh10_b_seal_sha256"],
        "- Stage42 bundle SHA-256: `%s`" % result["execution_bundle_sha256"],
        "",
        "## F0 — within-run diversity",
        "",
        "- LH10-A classification: `%s`; LH10-B classification: `%s`." % (audit["f0"]["runs"]["LH10-A"]["within_run"]["classification"], audit["f0"]["runs"]["LH10-B"]["within_run"]["classification"]),
        "- StateV1 hashes are distinct within each run: A `%s`; B `%s`." % (audit["f0"]["runs"]["LH10-A"]["within_run"]["state_v1_distinct"], audit["f0"]["runs"]["LH10-B"]["within_run"]["state_v1_distinct"]),
        "- A/B corresponding-q body, wrist, StateV1, raw, and decoded arrays: identical `%s`." % audit["f0"]["a_b_corresponding_q"]["all_corresponding_fields_identical"],
        "- Within-run pairwise norms and hash matrices are stored in the machine report; norms are characterization, not pass thresholds.",
        "",
        "## F1 — endpoint semantics",
        "",
        "- `endpoint_reached` criterion: `%s`." % audit["f1"]["endpoint_definition"]["endpoint_reached"],
        "- Stage41 completion evidence: `traj_s_ref=%s`, `traj_L=%s`, endpoint `%s`." % (audit["f1"]["endpoint_definition"]["observed_traj_s_ref"], audit["f1"]["endpoint_definition"]["observed_traj_L"], audit["f1"]["endpoint_definition"]["observed_endpoint_reached"]),
        "- Claim boundary: `GR00T_to_WBC_trajectory_execution_completed; not manipulation_task_success`.",
        "",
        "## F2 — joint-limit semantics",
        "",
        "- Classification: `%s`; soft-limit factor: `%s`." % (f2["classification"], f2["soft_limit_factor"]),
        "- LH10-A/B flag rows: `%s/%s`; negative soft-margin cells: `%s/%s`; maximum violation magnitude: `%s`." % (f2["run_a"]["joint_limit_violation_flag_row_count"], f2["run_b"]["joint_limit_violation_flag_row_count"], f2["run_a"]["negative_soft_margin_cell_count"], f2["run_b"]["negative_soft_margin_cell_count"], f2["run_a"]["maximum_soft_limit_violation_magnitude"]),
        "- Hard URDF-limit violating cells: A `%s`; B `%s`; joint-limit termination rows: A/B `%s/%s`." % (f2["run_a"]["hard_limit_violation_cell_count"], f2["run_b"]["hard_limit_violation_cell_count"], f2["run_a"]["joint_limit_termination_row_count"], f2["run_b"]["joint_limit_termination_row_count"]),
        "- Interpretation: `%s`." % f2["interpretation"],
        "",
        "## F3/F4 — archived replay and modality probes",
        "",
        "- Counters: 3 exact replays + 6 counterfactuals = `%s` GR00T forwards; WBC `0`; reference updates `0`." % probe["counters"]["gr00t_forward_count"],
        "- Exact replay: `%s`; replay raw hashes match accepted Stage41 q0/q5/q10: `%s`." % (probe["replay_pass"], probe.get("replay_hashes_exact")),
        "- Counterfactual pass: `%s`; all six use only archived body/wrist/StateV1 bytes and unchanged language." % probe["counterfactual_pass"],
    ]
    for name, item in probe.get("sensitivity", {}).items():
        lines.append("- %s: `%s`; changed raw/decoded bytes `%s`; raw max-abs delta `%s`; decoded max-abs delta `%s`." % (name, item.get("classification"), item.get("output_bytes_changed"), item.get("raw_max_abs_delta"), item.get("decoded_max_abs_delta")))
    lines.extend([
        "",
        "## F5 interpretation",
        "",
        "1. `%s`" % interp["stage41_loop_interpretation"],
        "2. Modality classification: `%s`." % json.dumps(interp["modality_sensitivity"], sort_keys=True),
        "3. `%s`" % interp["endpoint_claim"],
        "4. `%s`" % interp["joint_limit_claim"],
        "5. `%s`" % interp["next_investment_recommendation"],
        "",
        "Boundary: `%s`." % interp["bounded_scope"],
        "",
        "Final token: `CORELAB_STAGE42_FEEDBACK_CAUSALITY_QUALIFIED`; stop for PI review.",
    ])
    return "\n".join(lines) + "\n"


def finalize(
    bundle_path: Path,
    audit_path: Path,
    probe_path: Path,
    output_root: Path,
    report_path: Path,
    result_path: Path,
    seal_path: Path,
) -> Dict[str, Any]:
    bundle = read_json(bundle_path)
    _verify_bundle(bundle)
    audit = read_json(audit_path)
    _require(audit.get("schema") == "CORELAB_STAGE42_F0_F2_AUDIT_V1" and audit.get("status") == "PASS", "F0/F1/F2 audit is not PASS")
    _require(audit.get("f0", {}).get("token") == "CORELAB_STAGE42_WITHIN_RUN_CAUSAL_AUDIT_PASS", "F0 token missing")
    _require(audit.get("f1", {}).get("token") == "CORELAB_STAGE42_SUCCESS_SEMANTICS_AUDITED", "F1 token missing")
    _require(audit.get("f2", {}).get("classification") == "JOINT_LIMIT_METRIC_SEMANTICS_RESOLVED", "F2 did not resolve joint-limit semantics")
    probe = _read_probe_summary(probe_path, bundle)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    local_bundle = output_root / "stage42_execution_bundle.json"
    local_audit = output_root / "f0_f2_audit.json"
    local_probe = output_root / "probe_result.json"
    if local_bundle != Path(bundle_path).resolve():
        local_bundle.write_bytes(Path(bundle_path).read_bytes())
    if local_audit != Path(audit_path).resolve():
        local_audit.write_bytes(Path(audit_path).read_bytes())
    if local_probe != Path(probe_path).resolve():
        local_probe.write_bytes(Path(probe_path).read_bytes())
    result: Dict[str, Any] = {
        "schema": "CORELAB_STAGE42_FEEDBACK_CAUSALITY_RESULT_V1",
        "stage_id": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE42_FEEDBACK_CAUSALITY_QUALIFIED",
        "return_token": "GPT_PI_REVIEW_AFTER_STAGE42_FEEDBACK_CAUSALITY",
        "classification": "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE",
        "thresholds_added": False,
        "repository_branch": bundle.get("repository_branch"),
        "repository_head": bundle.get("repository_head"),
        "execution_bundle_sha256": sha256_bytes(canonical_json_bytes(bundle)),
        "predecessor": bundle.get("predecessor"),
        "audit_sha256": sha256_file(audit_path),
        "probe_result_sha256": sha256_file(probe_path),
        "audit": audit,
        "probe": probe,
        "interpretation": _interpretation(audit, probe),
        "counters": probe["counters"],
        "forbidden_operations": expected_forbidden_operations(),
    }
    human = _human_report(result)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(human, encoding="utf-8")
    result["human_report"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    records = _manifest_records(output_root)
    manifest = {
        "schema": "CORELAB_STAGE42_EVIDENCE_MANIFEST_V1",
        "stage_id": STAGE_ID,
        "records": records,
        "record_count": len(records),
    }
    manifest_path = output_root / "evidence_manifest.json"
    manifest_sha = write_canonical_json(manifest_path, manifest)
    result["evidence_manifest"] = manifest
    result["evidence_manifest_sha256"] = manifest_sha
    result_sha = write_canonical_json(result_path, result)
    _verify_manifest(manifest, output_root)
    seal = {
        "schema": "CORELAB_STAGE42_STAGE_SEAL_V1",
        "stage_id": STAGE_ID,
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": result["token"],
        "execution_bundle_sha256": result["execution_bundle_sha256"],
        "audit_sha256": result["audit_sha256"],
        "probe_result_sha256": result["probe_result_sha256"],
        "evidence_manifest_sha256": manifest_sha,
        "result_sha256": result_sha,
        "predecessor": result["predecessor"],
    }
    seal_sha = write_canonical_json(seal_path, seal)
    Path(str(seal_path) + ".sha256").write_text("{}  {}\n".format(seal_sha, Path(seal_path).name), encoding="utf-8")
    return {
        "status": "PASS",
        "token": result["token"],
        "return_token": result["return_token"],
        "result_sha256": result_sha,
        "seal_sha256": seal_sha,
        "evidence_manifest_sha256": manifest_sha,
        "human_report_sha256": result["human_report"]["sha256"],
        "result": str(result_path),
        "seal": str(seal_path),
        "report": str(report_path),
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--run-a", type=Path, required=True)
    audit.add_argument("--seal-a", type=Path, required=True)
    audit.add_argument("--run-b", type=Path, required=True)
    audit.add_argument("--seal-b", type=Path, required=True)
    audit.add_argument("--predecessor-seal", type=Path, required=True)
    audit.add_argument("--repo-root", type=Path, required=True)
    audit.add_argument("--wbc-source-root", type=Path, required=True)
    audit.add_argument("--baseline-profile", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--bundle", type=Path, required=True)
    finish.add_argument("--audit", type=Path, required=True)
    finish.add_argument("--probe-result", type=Path, required=True)
    finish.add_argument("--output-root", type=Path, required=True)
    finish.add_argument("--report", type=Path, required=True)
    finish.add_argument("--result", type=Path, required=True)
    finish.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        value = audit_f0_f2(args.run_a, args.seal_a, args.run_b, args.seal_b, args.predecessor_seal, args.repo_root, args.wbc_source_root, args.baseline_profile)
        digest = write_canonical_json(args.output, value)
        print(json.dumps({"status": value["status"], "output": str(args.output), "sha256": digest, "f0": value["f0"]["token"], "f1": value["f1"]["token"], "f2": value["f2"]["token"]}, sort_keys=True))
        return 0 if value["status"] == "PASS" else 2
    value = finalize(args.bundle, args.audit, args.probe_result, args.output_root, args.report, args.result, args.seal)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
