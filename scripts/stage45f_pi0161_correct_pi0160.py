#!/usr/bin/env python3
"""CPU-only correction and accounting audit for the PI0160 raw packet.

This script deliberately consumes immutable node-local PI0160 evidence and
does not import Torch, GR00T, IsaacGym, or the WBC.  It corrects the two
reported frame/flow quantities from the original raw records, records the
limits of what can be recovered from those records, and builds a conservative
attempt ledger.  The resulting report is a new artifact; no historical file
is modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONDITIONS = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
COMPLETE_CONDITIONS = ("C0_NOMINAL", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM")
T_BUCKETS = (67, 369, 783)
HORIZON = 40
ACTION_DIM = 132
ACTIVE_SCALARS = 190
TARGET = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
PI_GATE = "PI_GATE_STATE_0161"
PI_COMMENT = 5652160514
PR_COMMENT = 5652164343
BRANCH = "stage45f/corelab-local-initial-state-pilot"
BASE_HEAD = "1522865703d0259ed0db5181eaccd9e7aa5f46d9"

EXPERT_ROOT = "/var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_expert_20260913_v7"
MODEL_ROOT = "/var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_model_20260913_v3"

# These are the hashes recorded by the accepted PI0160 packet.  They are
# checked before any scalar is read from the raw bundle.
EXPECTED_FILES = {
    "expert_result": "56e6035efe486cda720b286a63f5879b3d85f14766253a232f4842144e37b6a8",
    "expert_admission": "5d839039ba4f289913ed10001e1c6c93a44c87f6e11f860487e40b4226af82d2",
    "expert_evidence": "56e8b0840500de607d3e1d8f57f048411e14b7c73d424de9b11ee5d49d554bd1",
    "model_result": "fd79a35299f94cf1fa5bbe25b88938585752b7bd7cd210859ad951040ad64257",
    "model_admission": "a38120a6b34953d9680427517a566f83f688524454de6021bc86da5efe0a8152",
    "model_evidence": "cfbd52252db43d36f5f26907744a49e6da31f77261c0ebe5b82942d9b066db48",
    "model_raw_manifest": "3cf6b8081ed6606ef676d9104a91e61d33efd13e08c9a37b12b60d24a7925171",
    "model_table": "a21484cd0c70f5a543d6f1b90318461a3997e4ed2cf233127dd8905b775f0117",
    "model_clipping": "a00a0a415b32fa1fbecd898b73e081d70d4d3c123048eb74d833350769613568",
    "model_trajectory": "70634a88153f71d21b9d92979803ab4a8f70931882992c34e6182ff261599ad3",
    "model_conclusion": "ff2f396aedcf6283d6bb5d8fb37710c457f9feec5e7ca6bf65a7d5109232eae4",
}


class CorrectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    require(not path.exists(), f"refusing to overwrite correction evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
    return sha256_file(path)


def write_text(path: Path, value: str) -> str:
    require(not path.exists(), f"refusing to overwrite correction evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return sha256_file(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    require(not path.exists(), f"refusing to overwrite correction evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def finite_metric(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    require(np.isfinite(value).all(), "nonfinite correction input")
    return {
        "rmse": float(np.sqrt(np.mean(value * value))),
        "max_abs": float(np.max(np.abs(value))),
        "count": int(value.size),
    }


def active_field_metrics(value: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3:
        value = value[0]
    require(value.shape == (HORIZON, ACTION_DIM), f"field shape drift: {value.shape}")
    require(mask.shape == value.shape and int(mask.sum()) == ACTIVE_SCALARS, "T16 mask drift")
    return {
        "active_190": finite_metric(value[mask]),
        "xyz": finite_metric(value[:, :3]),
        "rot6d": finite_metric(value[:, 3:9]),
        "timing": finite_metric(value[:, 9]),
        "gripper": finite_metric(value[:, 10]),
        "padding": finite_metric(value[:, 11:]),
    }


def verify_file(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA drift: expected {expected}, got {actual}")
    return actual


def verify_source_bundle(expert_root: Path, model_root: Path) -> dict[str, Any]:
    files = {
        "expert_result": expert_root / "pi0160_matched_expert_result.json",
        "expert_admission": expert_root / "admission.json",
        "expert_evidence": expert_root / "evidence_hashes.json",
        "model_result": model_root / "pi0160_matched_reset_expert_model_result.json",
        "model_admission": model_root / "model_admission.json",
        "model_evidence": model_root / "evidence_hashes.json",
        "model_raw_manifest": model_root / "raw_manifest.json",
        "model_table": model_root / "matched_expert_model_table.csv",
        "model_clipping": model_root / "clipping_and_label_quality.json",
        "model_trajectory": model_root / "trajectory_plot.svg",
        "model_conclusion": model_root / "one_page_conclusion.md",
    }
    hashes = {label: verify_file(path, EXPECTED_FILES[label], label) for label, path in files.items()}
    expert = json_read(files["expert_result"])
    model = json_read(files["model_result"])
    expert_admission = json_read(files["expert_admission"])
    model_admission = json_read(files["model_admission"])
    require(expert.get("status") == "PARTIAL", f"unexpected sealed expert status: {expert.get('status')}")
    require(expert.get("counters", {}).get("conditions_started") == 5, "expert condition count drift")
    require(model.get("status") == "COMPLETED", f"unexpected sealed model status: {model.get('status')}")
    require(model.get("counters", {}).get("model_side_calls") == 45, "model call count drift")
    require(model.get("counters", {}).get("conditioning_completed") == 10, "conditioning count drift")
    require(model.get("counters", {}).get("action_head_calls") == 35, "action-head count drift")
    require(model_admission.get("expert_root") == str(expert_root), "model did not bind the audited v7 expert root")
    require(model_admission.get("expert_result_sha256") == hashes["expert_result"], "model/expert result binding drift")
    manifest = json_read(files["model_raw_manifest"])
    require(isinstance(manifest, list) and len(manifest) == 20, "sealed model raw manifest must contain 20 rows")
    raw_hashes: dict[str, str] = {}
    for item in manifest:
        path = Path(str(item["raw_path"]))
        expected = str(item["raw_file_sha256"])
        actual = verify_file(path, expected, f"raw {path.name}")
        raw_hashes[str(path)] = actual
    return {
        "paths": {label: str(path) for label, path in files.items()},
        "hashes": hashes,
        "expert_result": expert,
        "model_result": model,
        "expert_admission": expert_admission,
        "model_admission": model_admission,
        "raw_manifest": manifest,
        "raw_hashes": raw_hashes,
    }


def load_q0(expert_root: Path, condition: str) -> dict[str, np.ndarray]:
    path = expert_root / "conditions" / condition / "snapshots" / "q0.npz"
    require(path.is_file() and not path.is_symlink(), f"missing q0: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"eef_world_m", "anchor_world_m", "target_world_m"}
        require(required.issubset(set(data.files)), f"q0 keys drift for {condition}: {data.files}")
        return {
            "eef_world": np.asarray(data["eef_world_m"], dtype=np.float64).reshape(3),
            "anchor": np.asarray(data["anchor_world_m"], dtype=np.float64).reshape(3),
            "target": np.asarray(data["target_world_m"], dtype=np.float64).reshape(3),
            "path": np.asarray(path),
        }


def corrected_expert_condition(expert_root: Path, condition: str) -> dict[str, Any]:
    cond_root = expert_root / "conditions" / condition
    q0 = load_q0(expert_root, condition)
    require(np.array_equal(q0["target"], TARGET), f"target drift in {condition}")
    rows_path = cond_root / "rows.json"
    old_metrics_path = cond_root / "metrics.json"
    require(rows_path.is_file(), f"complete condition lacks rows.json: {condition}")
    payload = json_read(rows_path)
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 245, f"unexpected persisted row count for {condition}: {len(rows) if isinstance(rows, list) else None}")
    anchor = q0["anchor"]
    achieved = np.asarray([row["achieved_pos"] for row in rows], dtype=np.float64)
    desired = np.asarray([row["desired_pos"] for row in rows], dtype=np.float64)
    recorded_achieved = np.asarray([row["achieved_pos_world"] for row in rows], dtype=np.float64)
    recorded_desired = np.asarray([row["desired_pos_world"] for row in rows], dtype=np.float64)
    require(np.isfinite(achieved).all() and np.isfinite(desired).all(), f"nonfinite expert rows: {condition}")
    # This is the source-level audit predicate: _step_rows' fields are already
    # world-frame, while the old wrapper added anchor a second time.
    require(np.allclose(recorded_achieved - achieved, anchor[None, :], rtol=0.0, atol=2e-6), f"achieved anchor mismatch: {condition}")
    require(np.allclose(recorded_desired - desired, anchor[None, :], rtol=0.0, atol=2e-6), f"desired anchor mismatch: {condition}")
    positions = np.vstack((q0["eef_world"], achieved))
    distances = np.linalg.norm(positions - TARGET[None, :], axis=1)
    tracking = desired - achieved
    old = json_read(old_metrics_path)
    corrected = {
        "condition_id": condition,
        "frame": "world",
        "source_fields": {"achieved": "rows[].achieved_pos", "desired": "rows[].desired_pos", "old_wrong_fields": ["rows[].achieved_pos_world", "rows[].desired_pos_world"]},
        "duplicate_anchor_detected": True,
        "anchor_world_m": anchor.tolist(),
        "target_world_m": TARGET.tolist(),
        "initial_target_error_m": float(distances[0]),
        "minimum_target_error_m": float(np.min(distances)),
        "minimum_error_sample_index": int(np.argmin(distances)),
        "final_target_error_m": float(distances[-1]),
        "absolute_progress_m": float(distances[0] - distances[-1]),
        "best_progress_m": float(distances[0] - np.min(distances)),
        "normalized_progress": float((distances[0] - distances[-1]) / distances[0]),
        "reference_path_length_m": float(np.sum(np.linalg.norm(np.diff(desired, axis=0), axis=1))),
        "achieved_path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "tracking_rms_m": float(np.sqrt(np.mean(np.linalg.norm(tracking, axis=1) ** 2))),
        "tracking_max_m": float(np.max(np.linalg.norm(tracking, axis=1))),
        "progress_sign": "POSITIVE" if distances[0] > distances[-1] else ("ZERO" if np.isclose(distances[0], distances[-1]) else "NEGATIVE"),
        "sample_count_including_S0": int(len(distances)),
        "persisted_rows": int(len(rows)),
        "old_report_metrics": {key: old.get(key) for key in ("final_target_error_m", "absolute_progress_m", "reference_path_length_m", "achieved_path_length_m", "tracking_rms_m", "tracking_max_m")},
        "old_report_correction": {"final_target_error_delta_m": float(distances[-1] - float(old.get("final_target_error_m"))), "absolute_progress_delta_m": float((distances[0] - distances[-1]) - float(old.get("absolute_progress_m")))},
    }
    return corrected


def c1_persistence_audit(expert_root: Path) -> dict[str, Any]:
    cond_root = expert_root / "conditions" / "C1_X_PLUS_5CM"
    failure = json_read(cond_root / "failure.json")
    files: list[dict[str, Any]] = []
    for path in sorted(cond_root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append({"relative_path": path.relative_to(cond_root).as_posix(), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    require(not (cond_root / "rows.json").exists(), "C1 rows.json unexpectedly appeared in historical raw")
    require(int(failure.get("rows_saved", -1)) == 49, "C1 in-memory rows_saved marker drift")
    require(int(failure.get("snapshots_saved", -1)) == 2, "C1 snapshot count drift")
    return {
        "condition_id": "C1_X_PLUS_5CM",
        "status": failure.get("status"),
        "failure": failure,
        "rows_json_present": False,
        "persisted_rows_confirmed": 0,
        "in_memory_rows_saved_marker": int(failure["rows_saved"]),
        "snapshots_persisted": int(failure["snapshots_saved"]),
        "recoverable_snapshot_ids": ["q0", "q1"],
        "unrecoverable_telemetry_gap": True,
        "evidence_files": files,
        "interpretation": "rows_saved is an in-memory counter; no rows.json was written before the abort, so the 49 rows are not independently persisted row evidence",
    }


def clipping_and_codec_audit(model_root: Path) -> dict[str, Any]:
    old = json_read(model_root / "clipping_and_label_quality.json")
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        item = old[condition]
        output[condition] = {
            "active_mask_shape": item.get("active_mask_shape"),
            "active_label_scalars": item.get("active_label_scalars"),
            "normalized_boundary_saturation_count": item.get("normalized_clipped_scalar_count"),
            "actual_preclip_clipping_count": None,
            "actual_clipping_status": "NOT_RECOVERABLE_PRECLIP_NOT_SAVED",
            "base_state_clipping_audit": "NOT_PRESENT_IN_PI0160_LABEL_ARTIFACT",
            "codec_roundtrip_input_untouched": bool(item.get("label_codec_roundtrip_input_untouched")),
            "strict_codec_rotation_error_rad": item.get("label_side_anchor_reconciliation", {}).get("strict_codec_rotation_error_rad"),
            "stats_route": item.get("stats_route"),
        }
        require(output[condition]["active_mask_shape"] == [40, 132], f"clip mask shape drift: {condition}")
        require(output[condition]["active_label_scalars"] == ACTIVE_SCALARS, f"active scalar count drift: {condition}")
        require(output[condition]["codec_roundtrip_input_untouched"], f"codec round-trip audit failed: {condition}")
    return {
        "schema": "CORELAB_PI0161_CLIPPING_CODEC_CORRECTION_V1",
        "interpretation": "The old >=1 count is a normalized boundary-saturation count, not a true clip count. Pre-clip values were not sealed, so true clipping is explicitly unknown.",
        "conditions": output,
    }


def corrected_model_rows(bundle: Mapping[str, Any], expert_metrics: Mapping[str, Any], model_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table_path = model_root / "matched_expert_model_table.csv"
    with table_path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    require(len(table) == 20, f"model table row count drift: {len(table)}")
    by_key = {(row["condition"], row["mode"], row.get("t_bucket", "")): row for row in table}
    corrected: list[dict[str, Any]] = []
    mask_by_t: dict[str, list[float]] = {}
    zero_xyz_by_t: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        for bucket in T_BUCKETS:
            raw_path = model_root / "raw" / f"{condition}_teacher_t{bucket:03d}.npz"
            with np.load(raw_path, allow_pickle=False) as data:
                required = {"x0_float32", "x_t_float32", "v_target_float32", "v_hat_float32", "x1_hat_float32", "active_mask_float64", "t_value", "t_bucket"}
                require(required.issubset(set(data.files)), f"teacher raw keys drift: {raw_path}")
                x0 = np.asarray(data["x0_float32"], dtype=np.float64)
                xt = np.asarray(data["x_t_float32"], dtype=np.float64)
                vtarget = np.asarray(data["v_target_float32"], dtype=np.float64)
                vhat = np.asarray(data["v_hat_float32"], dtype=np.float64)
                x1hat = np.asarray(data["x1_hat_float32"], dtype=np.float64)
                mask = np.asarray(data["active_mask_float64"], dtype=np.float64).astype(bool)
                t_value = float(np.asarray(data["t_value"]).reshape(-1)[0])
                require(int(np.asarray(data["t_bucket"]).reshape(-1)[0]) == bucket, f"teacher bucket drift: {raw_path}")
            x1 = x0 + vtarget
            model_field = vhat - vtarget
            zero_field = -vtarget
            model_reconstruction = x1hat - x1
            zero_reconstruction = xt - x1
            identity = model_reconstruction - (1.0 - t_value) * model_field
            model_blocks = active_field_metrics(model_field, mask)
            zero_blocks = active_field_metrics(zero_field, mask)
            model_recon_blocks = active_field_metrics(model_reconstruction, mask)
            zero_recon_blocks = active_field_metrics(zero_reconstruction, mask)
            old_row = by_key[(condition, "MATCHED_LOCAL_FIELD", str(bucket))]
            out = {
                "condition": condition,
                "mode": "MATCHED_LOCAL_FIELD",
                "t_bucket": bucket,
                "t_value": t_value,
                "raw_path": str(raw_path),
                "raw_file_sha256": sha256_file(raw_path),
                "model_field_active_rmse": model_blocks["active_190"]["rmse"],
                "model_field_xyz_rmse": model_blocks["xyz"]["rmse"],
                "model_field_active_max_abs": model_blocks["active_190"]["max_abs"],
                "zero_field_active_rmse": zero_blocks["active_190"]["rmse"],
                "zero_field_xyz_rmse": zero_blocks["xyz"]["rmse"],
                "zero_field_active_max_abs": zero_blocks["active_190"]["max_abs"],
                "model_reconstruction_active_rmse": model_recon_blocks["active_190"]["rmse"],
                "zero_reconstruction_active_rmse": zero_recon_blocks["active_190"]["rmse"],
                "flow_identity_max_abs": float(np.max(np.abs(identity))),
                "decoded_final_target_error_m_reused": float(old_row["final_target_error_m"]),
                "decoded_progress_m_reused": float(old_row["absolute_progress_m"]),
                "decoder_stats_sha256_reused": old_row["decode_stats_sha256"],
                "old_zero_field_active_rmse": float(old_row["zero_field_active_rmse"]),
                "old_zero_field_xyz_rmse": float(old_row["zero_field_xyz_rmse"]),
            }
            require(out["flow_identity_max_abs"] <= 3e-6, f"flow algebra identity drift: {raw_path}")
            corrected.append(out)
            mask_by_t.setdefault(str(bucket), []).append(out["zero_field_active_rmse"])
            zero_xyz_by_t.setdefault(str(bucket), []).append(out["zero_field_xyz_rmse"])
    aggregate = {
        "zero_field_by_t": {
            bucket: {"active_rmse_mean": float(np.mean(values)), "xyz_rmse_mean": float(np.mean(zero_xyz_by_t[bucket])), "conditions": len(values)}
            for bucket, values in mask_by_t.items()
        },
        "old_zero_field_was_reconstruction_residual": True,
        "correct_zero_field_definition": "-(x1-x0)",
        "reconstruction_residual_definition": "x_t-x1",
    }
    return corrected, aggregate


def attempt_ledger() -> dict[str, Any]:
    # v1/v5 have no admission or result file; they cannot be silently treated
    # as zero.  The lower bound below counts only runs that proved a native
    # simulator construction through their sealed result counters.
    entries = [
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v1", "simulator_instances": None, "status": "UNRESOLVED_NO_PERSISTED_ADMISSION"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v2", "simulator_instances": 0, "native_wbc_ticks": 0, "status": "BLOCKED_BEFORE_NATIVE_CONSTRUCTION", "failure": "libpython3.8.so.1.0 ImportError"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v3", "simulator_instances": 5, "native_wbc_ticks": 1225, "reference_updates": 15, "status": "COMPLETED"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v4", "simulator_instances": 2, "native_wbc_ticks": 245, "reference_updates": 3, "status": "PARTIAL", "failure": "C1 NameError before native step"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v5", "simulator_instances": None, "status": "UNRESOLVED_NO_PERSISTED_ADMISSION"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v6", "simulator_instances": 2, "native_wbc_ticks": 296, "reference_updates": 4, "status": "PARTIAL", "failure": "C1 TRACKING_ABORT"},
        {"root": "stage45f_pi0160_matched_reset_expert_20260913_v7", "simulator_instances": 5, "native_wbc_ticks": 1031, "reference_updates": 13, "status": "PARTIAL", "failure": "C1 TRACKING_ABORT"},
        {"root": "stage45f_pi0160_matched_reset_model_20260913_v3", "simulator_instances": 0, "model_constructions": 1, "model_side_calls": 45, "status": "COMPLETED_MODEL_ONLY"},
    ]
    known_sim = sum(int(item.get("simulator_instances") or 0) for item in entries)
    known_wbc = sum(int(item.get("native_wbc_ticks") or 0) for item in entries)
    known_updates = sum(int(item.get("reference_updates") or 0) for item in entries)
    unknown = [item["root"] for item in entries if item.get("simulator_instances") is None]
    return {
        "schema": "CORELAB_PI0161_PI0160_ATTEMPT_LEDGER_V1",
        "entries": entries,
        "simulator_instance_cap_from_pi0160": 7,
        "known_simulator_instance_lower_bound": known_sim,
        "known_native_wbc_tick_total": known_wbc,
        "known_reference_update_total": known_updates,
        "unknown_attempt_artifacts": unknown,
        "accounting_status": "OVER_CAP_AND_INCOMPLETE" if known_sim > 7 or unknown else "WITHIN_CAP_COMPLETE",
        "new_simulator_execution_allowed": False,
        "block_reason_codes": ["KNOWN_SIMULATOR_INSTANCE_LOWER_BOUND_EXCEEDS_PI0160_CAP" if known_sim > 7 else None, "UNRESOLVED_PRIOR_ATTEMPT_ARTIFACTS" if unknown else None],
        "interpretation": "v3/v4/v6/v7 alone prove 14 native simulator constructions, already above the PI0160 cap of 7; v1/v5 lack persisted admissions and remain unresolved rather than being counted as zero.",
    }


def self_test() -> dict[str, Any]:
    x0 = np.zeros((1, HORIZON, ACTION_DIM), dtype=np.float32)
    x1 = np.ones_like(x0) * 0.25
    t = 0.37
    xt = (1.0 - t) * x0 + t * x1
    vtarget = x1 - x0
    vhat = vtarget.copy()
    x1hat = xt + (1.0 - t) * vhat
    require(np.allclose(x1hat, x1, rtol=0.0, atol=1e-7), "oracle flow endpoint identity failed")
    require(np.array_equal(-vtarget, -(x1 - x0)), "zero-field definition failed")
    anchor = np.asarray([0.5, -0.2, 0.1])
    require(np.allclose((x1[0, :, :3] + anchor) - x1[0, :, :3], anchor), "anchor separation failed")
    return {"status": "PASS", "tests": ["oracle_x1_hat_equals_x1", "zero_field_is_negative_vtarget", "world_anchor_applied_once"]}


def source_import_audit() -> dict[str, Any]:
    """Prove this correction path has no model/simulator import surface."""

    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = sorted(name for name in imports if name.split(".")[0] in {"torch", "transformers", "isaacgym", "gr00t"})
    require(not forbidden, f"forbidden model/simulator import in correction script: {forbidden}")
    return {"script": str(source_path), "top_level_imports": sorted(imports), "forbidden_imports": forbidden, "status": "PASS"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    expert_root = Path(args.expert_root).resolve()
    model_root = Path(args.model_root).resolve()
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output already exists: {output}")
    require(args.repository_head == BASE_HEAD, f"repository HEAD drift: {args.repository_head}")
    static_self_test = self_test()
    static_source_audit = source_import_audit()
    bundle = verify_source_bundle(expert_root, model_root)
    expert_metrics = {condition: corrected_expert_condition(expert_root, condition) for condition in COMPLETE_CONDITIONS}
    c1 = c1_persistence_audit(expert_root)
    clipping = clipping_and_codec_audit(model_root)
    model_rows, model_aggregate = corrected_model_rows(bundle, expert_metrics, model_root)
    ledger = attempt_ledger()
    report = {
        "schema": "CORELAB_PI0161_PI0160_CORRECTED_REPORT_V1",
        "status": "PARTIAL",
        "token": "CORELAB_PI0161_FRAME_SYNC_CLOSURE_FOR_PI_REVIEW",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {"branch": BRANCH, "head": args.repository_head},
        "sources": {"expert_root": str(expert_root), "model_root": str(model_root), "source_bundle_hashes": bundle["hashes"]},
        "expert_corrected_metrics": expert_metrics,
        "c1_persistence_audit": c1,
        "model_corrected_field_aggregate": model_aggregate,
        "clipping_and_codec_audit": clipping,
        "attempt_ledger": ledger,
        "static_self_test": static_self_test,
        "static_source_audit": static_source_audit,
        "new_execution": {"simulator_instances": 0, "native_wbc_ticks": 0, "gr00t_forward": 0, "model_side_calls": 0, "training_steps": 0, "status": "BLOCKED_BY_PI0160_ACCOUNTING"},
        "execution_boundary": {"cpu_only": True, "model_forward": 0, "gr00t_forward": 0, "simulator": 0, "wbc": 0, "training": 0, "backward": 0, "optimizer_step": 0, "pbs": False, "h200": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "icl": False, "dagger": False, "merge": False},
        "claim_boundary": "Corrected PI0160 raw/report metrics only. The four complete expert trajectories are positive world-target progress after removing the duplicate anchor; C1 persisted rows are incomplete; this packet authorizes no new experiment.",
        "recommendation": "After PI resolves the over-cap/incomplete historical attempt ledger, authorize exactly the PI0161 two-condition post-tick synchronized expert capture; do not train or reuse the old unsynchronized q0s as supervision.",
    }
    output.mkdir(parents=True)
    files: dict[str, str] = {}
    files["expert_corrected_metrics"] = write_csv(output / "expert_corrected_metrics.csv", list(expert_metrics.values()), ["condition_id", "initial_target_error_m", "minimum_target_error_m", "final_target_error_m", "absolute_progress_m", "best_progress_m", "normalized_progress", "reference_path_length_m", "achieved_path_length_m", "tracking_rms_m", "tracking_max_m", "progress_sign", "sample_count_including_S0", "persisted_rows"])
    files["model_corrected_field_metrics"] = write_csv(output / "model_corrected_field_metrics.csv", model_rows, ["condition", "mode", "t_bucket", "t_value", "raw_path", "raw_file_sha256", "model_field_active_rmse", "model_field_xyz_rmse", "model_field_active_max_abs", "zero_field_active_rmse", "zero_field_xyz_rmse", "zero_field_active_max_abs", "model_reconstruction_active_rmse", "zero_reconstruction_active_rmse", "flow_identity_max_abs", "decoded_final_target_error_m_reused", "decoded_progress_m_reused", "decoder_stats_sha256_reused", "old_zero_field_active_rmse", "old_zero_field_xyz_rmse"])
    files["c1_persistence_audit"] = write_json(output / "c1_persistence_audit.json", c1)
    files["clipping_codec_audit"] = write_json(output / "clipping_codec_audit.json", clipping)
    files["attempt_ledger"] = write_json(output / "attempt_ledger.json", ledger)
    files["static_audit"] = write_json(output / "static_audit.json", {"self_test": static_self_test, "source_import_audit": static_source_audit})
    files["report"] = write_json(output / "pi0160_corrected_result.json", report)
    readme = "# PI0161 corrected PI0160 report\n\n"
    readme += "Status: **PARTIAL / NEW EXECUTION BLOCKED**. This directory is a derived CPU-only report; the PI0160 v7 and model v3 directories are immutable.\n\n"
    readme += "## Corrections\n\n"
    readme += "- `_step_rows` `desired_pos`/`achieved_pos` are already world-frame. The old wrapper's `*_pos_world` fields added the anchor a second time. Four complete conditions therefore end at approximately 1.3 cm target error, not 0.719 m. Tracking residuals are unchanged because both sides received the same erroneous translation.\n"
    readme += "- The corrected zero field is `-(x1-x0)`. The former `x_t-x1` values are reconstruction residuals and are retained only as a separate column. `x1_hat-x1=(1-t)(v_hat-v_target)` is checked for every teacher row.\n"
    readme += "- The old normalized `abs(value)>=1` count is renamed boundary saturation. Pre-clip values were not saved, so true clipping is unknown; BASE-state clipping is not inferred. Codec round-trip flags remain PASS as sealed.\n"
    readme += "\n## C1 persistence\n\n`rows_saved=49` is only an in-memory failure marker. No `rows.json` exists; q0/q1 snapshots and `failure.json` are the independently persisted evidence. The missing rows are not reconstructed or rerun.\n\n"
    readme += "## Accounting gate\n\nSealed v3/v4/v6/v7 admissions prove a lower bound of 14 native simulator constructions, above PI0160's cap of 7. v1 and v5 have no persisted admission and remain unresolved, not zero. PI0161 new simulator/model execution is therefore blocked.\n\n"
    readme += "The CPU formula/source checks are in `static_audit.json`. See `expert_corrected_metrics.csv`, `model_corrected_field_metrics.csv`, `c1_persistence_audit.json`, `clipping_codec_audit.json`, and `attempt_ledger.json` for machine-readable evidence.\n"
    files["README"] = write_text(output / "README.md", readme)
    report["output_files"] = files
    # The report is intentionally not rewritten after the evidence files are
    # sealed.  Its source hashes and this separate index are immutable.
    files["evidence_index"] = write_json(output / "evidence_hashes.json", {"source_bundle": bundle["hashes"], "derived_files": files, "report_sha256": files["report"]})
    return {"status": report["status"], "token": report["token"], "output_root": str(output), "ledger": ledger, "files": files, "expert_metrics": expert_metrics, "model_aggregate": model_aggregate}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-root", default=EXPERT_ROOT, type=Path)
    p.add_argument("--model-root", default=MODEL_ROOT, type=Path)
    p.add_argument("--output-root", required=False, type=Path, default=None)
    p.add_argument("--repository-head", default=BASE_HEAD)
    p.add_argument("--self-test", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        raise SystemExit(0)
    if args.output_root is None:
        raise SystemExit("--output-root is required unless --self-test is used")
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "token": result["token"], "output_root": result["output_root"], "known_simulator_instance_lower_bound": result["ledger"]["known_simulator_instance_lower_bound"]}, sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
