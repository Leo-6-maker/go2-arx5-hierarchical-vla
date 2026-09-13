#!/usr/bin/env python3
"""CPU-only PI0154 recipe and Validation-E admission.

The static mode is deliberately usable before any Validation-E payload is
opened.  The ``--admit`` mode is the first authorized payload reader: it
rehashes the six locked snapshot descriptors and their referenced frames,
records the access ledger, and emits a late-bound admission for the H200
runner.  This module never imports Torch, GR00T, CUDA, a simulator, or a
model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json"
EXPECTED_SCHEMA = "STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1"
EXPECTED_STATUS = "FROZEN_BEFORE_VALIDATION_E_PAYLOAD_READ"
EXPECTED_GATE = 5650096011
EXPECTED_PR_POINTER = 5650106991
EXPECTED_BRANCH = "stage45f/vanda-validation-e-gen1-l1-2000"
EXPECTED_BASE_HEAD = "191120699b8219cbf62be8ffe60dc19755dec218"
TARGET_ID = "red_target_holdout_e"
EXPECTED_SNAPSHOT_IDS = tuple(
    f"stage22_expert_{TARGET_ID}:expert-{index:04d}" for index in range(6)
)
EXPECTED_SNAPSHOT_LOCK_SHA = "23c35e2204b9555c8431a3f86fd7e1767740bf7196341972e847bc43c141aec8"
EXPECTED_SPLIT_LOCK_SHA = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
EXPECTED_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
EXPECTED_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
EXPECTED_CALLS = {"conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36}
EXPECTED_T = [0.0, 0.25, 0.5, 0.75]
EXPECTED_BUCKETS = [0, 250, 500, 750]
EXPECTED_NOISE_SEED = 4242
EXPECTED_NOISE_SHAPE = [1, 40, 132]
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "cuda", "cv2", "numpy"}
FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "transformers",
    "gr00t",
    "numpy",
    "cv2",
    "isaacgym",
    "go2arx5_vla",
)
FRAME_ORIGIN_SPREAD_TOLERANCE_M = 1e-7
FRAME_STATE_TOLERANCE_M = 1e-7
EXPECTED_ACTION_PERIOD_S = 0.05
EXPECTED_LANGUAGE = "move the end effector to the red target"


class AdmissionError(RuntimeError):
    """Raised when PI0154 cannot be admitted exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_durable(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if not overwrite:
        require(not path.exists() and not path.is_symlink(), f"record must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _top_level_static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL))
    return {
        "schema": "STAGE45F_PI0154_ADMISSION_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "FAIL",
        "top_level_forbidden_imports": forbidden,
        "pbs": 0,
        "h200": 0,
        "model_import": 0,
        "model_forward": 0,
        "training": 0,
        "holdout_f_read": 0,
    }


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    """Return model/runtime modules already present before model admission."""

    modules = sys.modules if imported_modules is None else imported_modules
    return sorted(
        name
        for name in modules
        if isinstance(name, str)
        and any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    )


def _validate_freeze(freeze: Mapping[str, Any], *, expected_code_commit: str | None = None) -> dict[str, Any]:
    require(freeze.get("schema") == EXPECTED_SCHEMA, "recipe freeze schema drift")
    require(freeze.get("status") == EXPECTED_STATUS, "recipe freeze status drift")
    authority = freeze.get("authority")
    require(isinstance(authority, Mapping) and authority.get("issue_11_comment") == EXPECTED_GATE and authority.get("pr_66_comment") == EXPECTED_PR_POINTER, "PI0154 authority drift")
    support = freeze.get("support_checkout")
    require(isinstance(support, Mapping), "support checkout is missing")
    require(support.get("base_execution_head") == EXPECTED_BASE_HEAD, "support base HEAD drift")
    require(support.get("branch") == EXPECTED_BRANCH, "support branch drift")
    paths = support.get("evaluation_code_paths")
    require(paths == [
        "scripts/stage45f_vanda_validation_e_admission.py",
        "scripts/stage45f_vanda_validation_e_gen1.py",
        "hpc/pbs/stage45f_validation_e_gen1_h200.pbs",
    ], "evaluation code surface drift")
    if expected_code_commit is not None:
        require(support.get("evaluation_code_commit") == expected_code_commit, "evaluation code commit is not frozen")
    arms = freeze.get("checkpoint_arms")
    require(isinstance(arms, list) and len(arms) == 6, "six checkpoint arms are required")
    require([(item.get("route"), item.get("seed"), item.get("tier")) for item in arms] == [
        ("Legacy", 2222, 2000), ("Legacy", 2223, 2000), ("Legacy", 2224, 2000),
        ("L1", 2222, 2000), ("L1", 2223, 2000), ("L1", 2224, 2000),
    ], "checkpoint matrix/order drift")
    runtime = freeze.get("shared_runtime_identity")
    require(isinstance(runtime, Mapping), "shared runtime identity missing")
    require(runtime.get("full24_runtime_stats_sha256") == EXPECTED_STATS_SHA and runtime.get("full24_runtime_stats_payload_sha256") == EXPECTED_STATS_PAYLOAD_SHA, "stats identity drift")
    inputs = freeze.get("validation_input_lock")
    require(isinstance(inputs, Mapping), "validation input lock missing")
    require(inputs.get("target_id") == TARGET_ID and inputs.get("snapshot_count") == 6 and inputs.get("snapshot_ids") == list(EXPECTED_SNAPSHOT_IDS), "validation snapshot lock drift")
    require(inputs.get("snapshot_lock_sha256") == EXPECTED_SNAPSHOT_LOCK_SHA and inputs.get("split_lock_sha256") == EXPECTED_SPLIT_LOCK_SHA, "split/snapshot lock SHA drift")
    protocol = freeze.get("evaluation_protocol")
    require(isinstance(protocol, Mapping), "evaluation protocol missing")
    require(protocol.get("arms") == 6 and protocol.get("windows_per_arm") == 6 and protocol.get("unique_queries") == 36, "validation cardinality drift")
    require(protocol.get("noise_seed") == EXPECTED_NOISE_SEED and protocol.get("noise_shape") == EXPECTED_NOISE_SHAPE, "noise identity drift")
    require(protocol.get("t_values") == EXPECTED_T and protocol.get("timestep_ids") == EXPECTED_BUCKETS and protocol.get("dt") == 0.25, "E4 schedule drift")
    require(protocol.get("nominal_calls") == EXPECTED_CALLS, "nominal call budget drift")
    recovery = protocol.get("recovery_ceiling")
    require(isinstance(recovery, Mapping) and recovery.get("max_allocations") == 2 and recovery.get("max_model_side_calls") == 210 and recovery.get("max_model_loads") == 7, "recovery ceiling drift")
    boundaries = freeze.get("locked_boundaries")
    require(isinstance(boundaries, Mapping) and all(value is False for value in boundaries.values()), "locked boundary opened")
    guard = _top_level_static_guard()
    require(guard["status"] == "PASS", "admission module static guard failed")
    return {
        "schema": "STAGE45F_PI0154_RECIPE_FREEZE_VALIDATION_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "freeze_path": str(FREEZE_PATH),
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "authority": dict(authority),
        "support_checkout": dict(support),
        "checkpoint_count": len(arms),
        "snapshot_ids": list(EXPECTED_SNAPSHOT_IDS),
        "nominal_calls": dict(EXPECTED_CALLS),
        "static_guard": guard,
        "payload_read": False,
        "holdout_f_read": False,
        "model_import": False,
        "return_token": "VANDA_PI0154_VALIDATION_E_ADMISSION_RECORDED",
    }


def static_plan(freeze_path: Path = FREEZE_PATH) -> dict[str, Any]:
    freeze = load_json(freeze_path)
    require(isinstance(freeze, Mapping), "recipe freeze must be an object")
    return _validate_freeze(freeze)


def _regular_file(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def _finite_numbers(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} must have length {length}")
    result: list[float] = []
    for item in value:
        require(isinstance(item, (int, float)) and not isinstance(item, bool), f"{label} contains non-number")
        number = float(item)
        require(math.isfinite(number), f"{label} contains non-finite value")
        result.append(number)
    return result


def _frame_origin_audit(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Bind the native relative origin without target-based recentering.

    The retained Stage22 descriptor does not serialize the native anchor as a
    separate field.  Its raw telemetry does serialize the same observed EEF
    in world and native-relative coordinates.  Their difference is the frame
    transform origin; it is audited against StateV1 and shared across all six
    snapshots.  It is never inferred from the target or from an output.
    """

    state_payload = payload.get("state_v1")
    require(isinstance(state_payload, Mapping), f"StateV1 is missing: {path}")
    require(state_payload.get("schema") == "StateV1", f"StateV1 schema drift: {path}")
    require(state_payload.get("layout") == ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"], f"StateV1 layout drift: {path}")
    state = _finite_numbers(state_payload.get("values"), 10, f"StateV1 values {path}")
    telemetry = payload.get("raw_telemetry")
    require(isinstance(telemetry, Mapping), f"raw telemetry is missing: {path}")
    local = _finite_numbers(telemetry.get("achieved_position_local_m"), 3, f"achieved local EEF {path}")
    world = _finite_numbers(telemetry.get("achieved_position_world_m"), 3, f"achieved world EEF {path}")
    state_error = max(abs(local[index] - state[index]) for index in range(3))
    require(state_error <= FRAME_STATE_TOLERANCE_M, f"StateV1/local EEF mismatch: {path}: {state_error}")
    anchor = [world[index] - local[index] for index in range(3)]
    require(all(math.isfinite(item) for item in anchor), f"frame origin is non-finite: {path}")
    explicit_values: list[dict[str, Any]] = []
    for container_name in ("raw_telemetry", "provenance", None):
        container = payload if container_name is None else payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key in ("anchor_world_m", "native_anchor_world_m"):
            if key not in container:
                continue
            explicit = _finite_numbers(container.get(key), 3, f"{key} {path}")
            error = max(abs(explicit[index] - anchor[index]) for index in range(3))
            require(error <= FRAME_STATE_TOLERANCE_M, f"explicit frame origin disagrees with world/local telemetry: {path}: {error}")
            explicit_values.append({"container": container_name or "top_level", "key": key, "max_abs_error_m": error})
    return {
        "anchor_world_m": anchor,
        "source": "raw_telemetry.achieved_position_world_m-minus-achieved_position_local_m",
        "state_v1_local_max_abs_error_m": state_error,
        "explicit_anchor_checks": explicit_values,
        "target_recentered": False,
        "measured_start_recentered": False,
    }


def _read_snapshot_descriptor(path: Path, expected_id: str) -> dict[str, Any]:
    path = _regular_file(path, "snapshot descriptor")
    payload = load_json(path)
    require(isinstance(payload, Mapping), f"snapshot descriptor is not an object: {path}")
    require(payload.get("schema") == "SameSceneExpertSnapshotV1", f"snapshot schema drift: {path}")
    require(payload.get("example_id") == expected_id, f"snapshot ID drift: {path}")
    require(payload.get("target_id") == TARGET_ID, f"snapshot target drift: {path}")
    require(payload.get("target_excluded_from_model_input") is True, f"target exclusion is not explicit: {path}")
    timestamps = payload.get("future_timestamps")
    desired = payload.get("future_desired_eef_9d")
    gripper = payload.get("future_gripper")
    require(isinstance(timestamps, list) and len(timestamps) == 40, f"E snapshot timestamps are not H40: {path}")
    require(isinstance(desired, list) and len(desired) == 40 and all(isinstance(row, list) and len(row) == 9 for row in desired), f"E snapshot EEF labels are not H40x9: {path}")
    require(isinstance(gripper, list) and len(gripper) == 40 and all(isinstance(row, list) and len(row) == 1 for row in gripper), f"E snapshot gripper labels are not H40x1: {path}")
    require(payload.get("source_frame") == "NATIVE_WBC_RELATIVE_REFERENCE", f"E source frame drift: {path}")
    require(payload.get("tcp_semantic_id") == "x5_link6_gripper_center", f"E TCP identity drift: {path}")
    provenance = payload.get("provenance")
    require(isinstance(provenance, Mapping), f"E provenance is missing: {path}")
    require(provenance.get("language") == EXPECTED_LANGUAGE, f"E language provenance drift: {path}")
    require(provenance.get("target_excluded_from_model_input") is True, f"E provenance target exclusion drift: {path}")
    require(provenance.get("label_source") == "deterministic_smoothstep_position_hold", f"E label source drift: {path}")
    observation = payload.get("observation_timestamp")
    require(isinstance(observation, (int, float)) and not isinstance(observation, bool) and math.isfinite(float(observation)), f"E observation timestamp drift: {path}")
    timestamp_values = _finite_numbers(timestamps, 40, f"E timestamps {path}")
    require(abs(timestamp_values[0] - float(observation)) <= 1e-7, f"E timestamp origin drift: {path}")
    require(all(abs(timestamp_values[index + 1] - timestamp_values[index] - EXPECTED_ACTION_PERIOD_S) <= 1e-8 for index in range(39)), f"E timestamp spacing drift: {path}")
    state_payload = payload.get("state_v1")
    require(isinstance(state_payload, Mapping), f"E StateV1 is missing: {path}")
    state_values = _finite_numbers(state_payload.get("values"), 10, f"E StateV1 values {path}")
    desired_values = [[float(item) for item in row] for row in desired]
    gripper_values = [[float(item) for item in row] for row in gripper]
    require(all(math.isfinite(item) for row in desired_values for item in row), f"E desired label is non-finite: {path}")
    require(all(math.isfinite(item) for row in gripper_values for item in row), f"E gripper label is non-finite: {path}")
    require(max(abs(desired_values[0][index] - state_values[index]) for index in range(9)) <= FRAME_STATE_TOLERANCE_M, f"E label does not bind StateV1 pose: {path}")
    frame_origin = _frame_origin_audit(payload, path)
    body = _regular_file(Path(str(payload.get("body_frame_path"))), "E body frame")
    wrist = _regular_file(Path(str(payload.get("wrist_frame_path"))), "E wrist frame")
    require(provenance.get("body_frame_sha256") == sha256_file(body), f"E body frame provenance SHA drift: {path}")
    require(provenance.get("wrist_frame_sha256") == sha256_file(wrist), f"E wrist frame provenance SHA drift: {path}")
    return {
        "snapshot_id": expected_id,
        "descriptor_path": str(path),
        "descriptor_sha256": sha256_file(path),
        "descriptor_bytes": int(path.stat().st_size),
        "body_frame_path": str(body),
        "body_frame_sha256": sha256_file(body),
        "body_frame_bytes": int(body.stat().st_size),
        "wrist_frame_path": str(wrist),
        "wrist_frame_sha256": sha256_file(wrist),
        "wrist_frame_bytes": int(wrist.stat().st_size),
        "source_frame": payload.get("source_frame"),
        "tcp_semantic_id": payload.get("tcp_semantic_id"),
        "target_id": payload.get("target_id"),
        "observation_timestamp": payload.get("observation_timestamp"),
        "frame_origin": frame_origin,
        "state_shape": [10],
        "label_shape": [40, 10],
        "payload": dict(payload),
    }


def _find_snapshot_paths(source_root: Path) -> list[Path]:
    root = Path(source_root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"E snapshot root is unavailable: {root}")
    expected = {f"expert-{index:04d}.json" for index in range(6)}
    candidates: dict[str, Path] = {}
    for path in sorted(root.glob("snapshots/*/*.json")):
        if path.name in expected and path.parent.name == f"stage22_expert_{TARGET_ID}":
            candidates[path.name] = path
    if not candidates:
        for path in sorted(root.rglob("*.json")):
            if path.name in expected and path.parent.name == f"stage22_expert_{TARGET_ID}":
                candidates[path.name] = path
    require(set(candidates) == expected, f"E snapshot set is not the exact six-window cohort: {sorted(candidates)}")
    return [candidates[f"expert-{index:04d}.json"] for index in range(6)]


def admit(
    source_root: Path,
    output_root: Path,
    *,
    expected_code_commit: str,
    stats_path: Path,
    prior_use_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    freeze = load_json(FREEZE_PATH)
    require(isinstance(freeze, Mapping), "recipe freeze must be an object")
    freeze_validation = _validate_freeze(freeze, expected_code_commit=expected_code_commit)
    require(not output_root.exists() and not output_root.is_symlink(), f"admission output must be fresh: {output_root}")
    stats_path = _regular_file(stats_path, "runtime stats")
    require(sha256_file(stats_path) == EXPECTED_STATS_SHA, "runtime stats file SHA mismatch")
    started = now_utc()
    snapshot_paths = _find_snapshot_paths(source_root)
    records = []
    for index, path in enumerate(snapshot_paths):
        records.append(_read_snapshot_descriptor(path, EXPECTED_SNAPSHOT_IDS[index]))
    origins = [record["frame_origin"] for record in records]
    canonical_origin = list(origins[0]["anchor_world_m"])
    origin_errors = [
        max(abs(float(origin["anchor_world_m"][axis]) - canonical_origin[axis]) for axis in range(3))
        for origin in origins
    ]
    max_origin_spread = max(origin_errors)
    require(max_origin_spread <= FRAME_ORIGIN_SPREAD_TOLERANCE_M, f"E frame origin is not shared across six windows: {max_origin_spread}")
    frame_origin = {
        "schema": "STAGE45F_NATIVE_FRAME_ORIGIN_BINDING_V1",
        "anchor_world_m": canonical_origin,
        "source": "raw_telemetry.achieved_position_world_m-minus-achieved_position_local_m",
        "max_cross_window_spread_m": max_origin_spread,
        "spread_tolerance_m": FRAME_ORIGIN_SPREAD_TOLERANCE_M,
        "target_recentered": False,
        "measured_start_recentered": False,
        "state_v1_and_world_transform_audited": True,
    }
    finished = now_utc()
    output_root.mkdir(parents=True, exist_ok=False)
    access_ledger = {
        "schema": "STAGE45F_PI0154_VALIDATION_E_ACCESS_LEDGER_V1",
        "status": "PAYLOAD_READ_AND_BOUND",
        "gate": EXPECTED_GATE,
        "target_id": TARGET_ID,
        "source_root": str(Path(source_root).resolve()),
        "snapshot_ids": list(EXPECTED_SNAPSHOT_IDS),
        "first_payload_read_at_utc": started,
        "finished_at_utc": finished,
        "descriptor_reads": 6,
        "frame_reads": 12,
        "holdout_f_read": False,
        "simulator_collection": False,
        "new_data_generated": False,
        "records": [{key: value for key, value in record.items() if key != "payload"} for record in records],
        "frame_origin": frame_origin,
        "prior_e_use_check": dict(prior_use_report or {"status": "NOT_PROVIDED", "absence_is_not_evidence": True}),
    }
    write_json_durable(output_root / "validation_e_access_ledger.json", access_ledger)
    binding = {
        "schema": "STAGE45F_PI0154_VALIDATION_E_ADMISSION_V1",
        "status": "ADMITTED_BEFORE_MODEL_IMPORT",
        "pi_gate": EXPECTED_GATE,
        "pr_pointer": EXPECTED_PR_POINTER,
        "authority_token": "PI_GATE_STATE_0154",
        "execution_branch": EXPECTED_BRANCH,
        "execution_head": expected_code_commit,
        "recipe_freeze": {
            "path": str(FREEZE_PATH),
            "sha256": freeze_validation["freeze_sha256"],
            "status": EXPECTED_STATUS,
        },
        "checkpoint_arms": freeze["checkpoint_arms"],
        "stats": {"path": str(stats_path), "sha256": sha256_file(stats_path), "payload_sha256": EXPECTED_STATS_PAYLOAD_SHA},
        "validation_input": {
            "target_id": TARGET_ID,
            "source_root": str(Path(source_root).resolve()),
            "snapshot_lock_sha256": EXPECTED_SNAPSHOT_LOCK_SHA,
            "split_lock_sha256": EXPECTED_SPLIT_LOCK_SHA,
            "snapshot_ids": list(EXPECTED_SNAPSHOT_IDS),
            "snapshot_records": [{key: value for key, value in record.items() if key != "payload"} for record in records],
            "frame_origin": frame_origin,
            "access_ledger": str((output_root / "validation_e_access_ledger.json").resolve()),
            "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"),
            "target_xyz_in_model_input": False,
            "holdout_f_read": False,
        },
        "protocol": freeze["evaluation_protocol"],
        "frame_decoder_metrics": {
            "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
            "tcp_semantic_id": "x5_link6_gripper_center",
            "timing": "T16",
            "decoder": "Stage40_inverse_normalization_and_materialization",
            "metric_source_sha256": freeze["shared_runtime_identity"]["metrics_source_sha256"],
            "decoder_source_sha256": freeze["shared_runtime_identity"]["decoder_source_sha256"],
            "recenter_target": False,
            "frame_origin_binding": frame_origin,
        },
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "pbs_submissions": 0,
        "holdout_f_read": False,
        "output_root": str(Path(output_root).resolve()),
    }
    binding_path = output_root / "validation_e_admission.json"
    write_json_durable(binding_path, binding)
    result = {
        "schema": "STAGE45F_PI0154_VALIDATION_E_ADMISSION_RESULT_V1",
        "status": "PASS_PRE_MODEL",
        "token": "VANDA_PI0154_VALIDATION_E_ADMISSION_RECORDED",
        "execution_head": expected_code_commit,
        "freeze_sha256": freeze_validation["freeze_sha256"],
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "access_ledger_path": str(output_root / "validation_e_access_ledger.json"),
        "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"),
        "snapshot_count": 6,
        "descriptor_reads": 6,
        "frame_reads": 12,
        "payload_read": True,
        "holdout_f_read": False,
        "model_import": False,
        "model_forward": False,
        "training": False,
        "pbs_submissions": 0,
    }
    write_json_durable(output_root / "validation_e_admission_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static", action="store_true")
    modes.add_argument("--admit", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--prior-use-report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.static:
            result = static_plan()
        else:
            require(args.source_root is not None and args.output_root is not None and args.expected_code_commit and args.stats is not None, "--admit requires source/output/code-commit/stats")
            prior_use_report = None
            if args.prior_use_report is not None:
                loaded_prior_use_report = load_json(args.prior_use_report.resolve(strict=True))
                require(isinstance(loaded_prior_use_report, Mapping), "prior-use report must be a JSON object")
                prior_use_report = loaded_prior_use_report
            result = admit(
                args.source_root.resolve(),
                args.output_root.resolve(),
                expected_code_commit=args.expected_code_commit,
                stats_path=args.stats.resolve(),
                prior_use_report=prior_use_report,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"VANDA_PI0154_ADMISSION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
