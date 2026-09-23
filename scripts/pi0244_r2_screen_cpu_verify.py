#!/usr/bin/env python3
"""Read-only CPU verification of the sealed PI0244 R2 screen and R0 pairing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CARD_SHA256 = "c772fc39ea37bb944b3394f7edf01b02434a7622fa3a57374699f1d271f2ff0d"
BINDING_ID = "3a3f443ffd43b967c285091d0000f525e855a24a8c6e0b0c7c4df55e89502ee3"
MANIFEST_SHA256 = "3ec77a175cdf15d690debae749665bfee6000da5cc9c1076ebf41ca19ee8ecf3"
STEPS = (0, 2500, 5000, 7500, 10000)
DT = 0.05
LANGUAGE = "move the end effector to the red target"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equivalent(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: key mismatch"]
        return [e for k in a for e in equivalent(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length mismatch"]
        return [e for i, (x, y) in enumerate(zip(a, b)) for e in equivalent(x, y, f"{path}/{i}")]
    if isinstance(a, (int, float, np.number)) and not isinstance(a, bool) and isinstance(b, (int, float, np.number)) and not isinstance(b, bool):
        return [] if np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9) else [f"{path}: numeric mismatch"]
    return [] if a == b else [f"{path}: value mismatch"]


def load_helpers(code_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    sys.path[:0] = [str(code_root / "src"), str(code_root / "scripts")]
    path = code_root / "scripts" / "pi0243_staged_learning_to_sim.py"
    spec = importlib.util.spec_from_file_location("pi0244_screen_metric_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen metric helper: {path}")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, rot6d_rows_to_matrix

    return metrics, selector, TrajectorySafetyGateV1, geodesic_angle, (matrix_to_quat, rot6d_rows_to_matrix)


def read_profile(root: Path, step: int, expected: set[str], helpers: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    metrics, selector, Gate, geodesic_angle, quat_helpers = helpers
    matrix_to_quat, rot6d_rows_to_matrix = quat_helpers
    directory = root / "screen" / f"step-{step:05d}"
    records_path = directory / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("source_episode_id")) for row in rows]
    errors: list[str] = []
    if len(rows) != 16 or len(set(ids)) != 16 or set(ids) != expected:
        errors.append(f"step {step}: record ID set/count mismatch")

    by_id: dict[str, Any] = {}
    file_names: set[str] = set()
    input_hashes: dict[str, dict[str, str]] = {}
    issue_counts: Counter[str] = Counter()
    metric_names = (
        "xyz_rmse_m", "so3_rmse_rad", "gripper_mae", "first_xyz_error_m",
        "endpoint_xyz_error_m", "max_linear_speed_mps", "max_angular_speed_radps",
    )
    for row in rows:
        source_id = str(row["source_episode_id"])
        episode = int(row["episode_index"])
        tensor = Path(row["tensor_path"]).resolve()
        expected_tensor = (directory / "cells" / f"train-{episode:06d}.npz").resolve()
        if tensor != expected_tensor or not tensor.is_file():
            errors.append(f"{source_id}: tensor path/episode binding mismatch")
            continue
        file_names.add(tensor.name)
        if sha256(tensor) != row.get("tensor_sha256"):
            errors.append(f"{source_id}: NPZ file hash mismatch")
        with np.load(tensor, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        expected_shapes = {
            "raw": (1, 40, 132), "decoded": (1, 40, 10), "expert": (40, 10),
            "state": (10,), "action_mask": (40, 132), "body_rgb": (96, 96, 3),
            "wrist_rgb": (96, 96, 3),
        }
        if set(arrays) != set(expected_shapes):
            errors.append(f"{source_id}: NPZ key set mismatch")
        for key, shape in expected_shapes.items():
            if key not in arrays or arrays[key].shape != shape:
                errors.append(f"{source_id}: {key} shape mismatch")
        for key, array in arrays.items():
            if not np.isfinite(array).all():
                errors.append(f"{source_id}: {key} contains nonfinite values")
        dtype_expect = {"raw": "float32", "decoded": "float32", "expert": "float64", "state": "float32", "action_mask": "float64", "body_rgb": "uint8", "wrist_rgb": "uint8"}
        for key, dtype in dtype_expect.items():
            if key in arrays and str(arrays[key].dtype) != dtype:
                errors.append(f"{source_id}: {key} dtype mismatch")
        if metrics.sha256_array(arrays["raw"]) != row.get("raw_sha256") or metrics.sha256_array(arrays["decoded"]) != row.get("decoded_sha256"):
            errors.append(f"{source_id}: raw/decoded tensor hash mismatch")
        if row.get("screen_step") != step or row.get("seed") != 22301 or row.get("language") != LANGUAGE or row.get("target_xyz_in_model_input") is not False:
            errors.append(f"{source_id}: frozen screen binding mismatch")

        prediction = np.asarray(arrays["decoded"][0], dtype=np.float64)
        expert = np.asarray(arrays["expert"], dtype=np.float64)
        recomputed = metrics._screen_metrics(prediction, expert, None)
        for key in metric_names:
            if key not in row or not np.isclose(recomputed[key], float(row[key]), rtol=1e-9, atol=1e-10):
                errors.append(f"{source_id}: NPZ-derived {key} differs from record")

        achieved_q = matrix_to_quat(rot6d_rows_to_matrix(np.asarray(arrays["state"][3:9], dtype=np.float64).reshape(1, 6)))[0]
        chunk = selector._build_chunk(arrays["decoded"], chunk_id=f"pi0243-A-{step}-{episode}")
        gate = Gate().evaluate(chunk, now=0.0, achieved_position=arrays["state"][:3], achieved_orientation=achieved_q).as_dict()
        recorded_gate = row.get("gate")
        decision = row.get("gate_decision")
        if decision not in {"ACCEPT", "WARN", "REJECT"}:
            errors.append(f"{source_id}: invalid SafetyGate decision enum")
        if decision != gate.get("decision") or not isinstance(recorded_gate, dict) or equivalent(gate, recorded_gate):
            errors.append(f"{source_id}: recomputed SafetyGate decision/issues/metrics mismatch")
        for issue in (recorded_gate or {}).get("issues", []):
            if issue.get("severity") not in {"REJECT", "WARNING"} or not issue.get("code"):
                errors.append(f"{source_id}: invalid SafetyGate issue enum/schema")
            issue_counts[str(issue.get("code"))] += 1

        input_hashes[source_id] = {
            key: metrics.sha256_array(arrays[key]) for key in ("expert", "action_mask", "state", "body_rgb", "wrist_rgb")
        }
        by_id[source_id] = {"row": row, "arrays": arrays, "metrics": recomputed, "gate": gate}

    actual_files = {p.name for p in (directory / "cells").glob("*.npz")}
    if actual_files != file_names:
        errors.append(f"step {step}: unexpected or missing NPZ files")
    gate_counts = {key: sum(row.get("gate_decision") == key for row in rows) for key in ("ACCEPT", "WARN", "REJECT")}
    xyz = np.asarray([by_id[source_id]["metrics"]["xyz_rmse_m"] for source_id in ids if source_id in by_id], dtype=np.float64)
    so3 = np.asarray([by_id[source_id]["metrics"]["so3_rmse_rad"] for source_id in ids if source_id in by_id], dtype=np.float64)
    vmax = np.asarray([by_id[source_id]["metrics"]["max_linear_speed_mps"] for source_id in ids if source_id in by_id], dtype=np.float64)
    wmax = np.asarray([by_id[source_id]["metrics"]["max_angular_speed_radps"] for source_id in ids if source_id in by_id], dtype=np.float64)
    profile = {
        "step": step, "count": len(rows), "records_sha256": sha256(records_path),
        "gpu_names": sorted({str(row.get("gpu", {}).get("name")) for row in rows}),
        "gate_counts": {key: value for key, value in gate_counts.items() if value},
        "issue_counts": dict(sorted(issue_counts.items())),
        "linear_speed_reject_count": sum(any(i.get("code") == "LINEAR_SPEED_REJECT" for i in by_id[s]["gate"]["issues"]) for s in by_id),
        "median_xyz_rmse_m": float(np.percentile(xyz, 50, method="linear")),
        "p95_xyz_rmse_m": float(np.percentile(xyz, 95, method="linear")),
        "median_so3_rmse_rad": float(np.percentile(so3, 50, method="linear")),
        "linear_speed_mps": {"median": float(np.median(vmax)), "p95": float(np.percentile(vmax, 95, method="linear")), "max": float(np.max(vmax))},
        "angular_speed_radps": {"median": float(np.median(wmax)), "p95": float(np.percentile(wmax, 95, method="linear")), "max": float(np.max(wmax))},
        "npz_count": len(file_names),
    }
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for key in ("count", "linear_speed_reject_count"):
        if summary.get(key) != profile[key]: errors.append(f"step {step}: summary {key} mismatch")
    if summary.get("gate_counts") != profile["gate_counts"]: errors.append(f"step {step}: summary gate-count mismatch")
    for key in ("median_xyz_rmse_m", "p95_xyz_rmse_m", "median_so3_rmse_rad"):
        if not np.isclose(float(summary.get(key, np.nan)), profile[key], rtol=1e-9, atol=1e-10): errors.append(f"step {step}: summary {key} mismatch")
    return rows, {"by_id": by_id, "input_hashes": input_hashes}, profile, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--r0-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--validator-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    r2, r0 = args.r2_root.resolve(), args.r0_root.resolve()
    helpers = load_helpers(args.code_root.resolve())
    card_path = r2 / "pi0243_run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card_hash = sha256(card_path)
    expected_list = [str(value) for value in card["fixed16_source_ids"]]
    expected = set(expected_list)
    errors: list[str] = []
    if card_hash != CARD_SHA256 or len(expected_list) != 16 or len(expected) != 16: errors.append("R2 child-card identity/source IDs invalid")
    if card.get("r2_execution_binding", {}).get("binding_id") != BINDING_ID: errors.append("R2 execution binding ID mismatch")

    training_path = r2 / "pi0244x_xyz4_training_closure.json"
    reconciliation_path = r2 / "pi0244x_xyz4_posttrain_reconciliation.json"
    manifest_path = r2 / "pi0243_training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    training_ok = (
        training.get("status") == "PASS_R2_TRAINING_CLOSED"
        and training.get("selected_checkpoint") == "NONE"
        and training.get("run_card_sha256") == card_hash
        and training.get("execution_binding_id") == BINDING_ID
        and training.get("optimizer_steps") == 10000
        and training.get("actual_sample_consumption") == 10000
        and training.get("actual_consumption_matches_all_schedule_fields") is True
        and sha256(manifest_path) == MANIFEST_SHA256
        and reconciliation.get("status") == "PASS_DERIVED_FROM_EXISTING_RUN_ARTIFACTS"
        and reconciliation.get("update_audit_file_modified") is False
        and reconciliation.get("backward_optimizer_scheduler_steps_during_recovery") == 0
    )
    if not training_ok: errors.append("R2 training/reconciliation binding is incomplete")
    if sha256(reconciliation_path) != training.get("postrun_reconciliation_sha256"):
        errors.append("R2 reconciliation hash differs from training closure")

    parent = card["r0_parent"]
    parent_paths = {
        "card": Path(parent["card_path"]),
        "schedule": Path(parent["schedule_path"]),
        "closure": Path(parent["closure_path"]),
    }
    parent_expected = {"card": parent["card_sha256"], "schedule": parent["schedule_sha256"], "closure": parent["closure_sha256"]}
    parent_checks = {key: path.is_file() and sha256(path) == parent_expected[key] for key, path in parent_paths.items()}
    if not all(parent_checks.values()): errors.append("sealed R0 parent artifact hash mismatch")

    profiles: list[dict[str, Any]] = []
    r2_data: dict[int, dict[str, Any]] = {}
    for step in STEPS:
        rows, detail, profile, profile_errors = read_profile(r2, step, expected, helpers)
        errors.extend(profile_errors)
        binding_path = r2 / "r2_screen_bindings" / f"step-{step:05d}.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding_path = r2 / "r2_screen_expected" / f"step-{step:05d}.json"
        expected_binding = json.loads(expected_binding_path.read_text(encoding="utf-8"))
        record_by = {str(row["source_episode_id"]): row for row in rows}
        tensor_map = {source: row["tensor_sha256"] for source, row in record_by.items()}
        raw_map = {source: row["raw_sha256"] for source, row in record_by.items()}
        decoded_map = {source: row["decoded_sha256"] for source, row in record_by.items()}
        if (binding.get("status") != "COMPLETE" or binding.get("run_card_sha256") != card_hash
                or binding.get("execution_binding_id") != BINDING_ID or binding.get("step") != step
                or binding.get("record_count") != 16 or binding.get("records_sha256") != profile["records_sha256"]
                or binding.get("tensor_sha256s") != tensor_map or binding.get("raw_sha256s") != raw_map
                or binding.get("decoded_sha256s") != decoded_map):
            errors.append(f"step {step}: actual profile binding mismatch")
        if (expected_binding.get("run_card_sha256") != card_hash or expected_binding.get("execution_binding_id") != BINDING_ID
                or expected_binding.get("step") != step or expected_binding.get("source_episode_ids") != expected_list
                or expected_binding.get("model_identity") != binding.get("model_identity")):
            errors.append(f"step {step}: sealed expected binding mismatch")
        if step:
            receipt_path = r2 / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_sha = sha256(receipt_path)
            receipt_tree = receipt.get("source_record", {}).get("inventory", {}).get("tree_sha256")
            closed_receipt = training.get("checkpoint_receipts", {}).get(str(step), {})
            identity = binding.get("model_identity", {})
            if (receipt_sha != closed_receipt.get("receipt_sha256") or receipt_tree != closed_receipt.get("tree_sha256")
                    or receipt_sha != identity.get("receipt_sha256") or receipt_tree != identity.get("tree_sha256")):
                errors.append(f"step {step}: durable checkpoint receipt/profile identity mismatch")
        closure = json.loads((r2 / "pi0244x_xyz4_screen_closure.json").read_text(encoding="utf-8"))
        if closure.get("profile_bindings", {}).get(str(step)) != sha256(binding_path):
            errors.append(f"step {step}: screen closure binding hash mismatch")
        profiles.append({**profile, "binding_sha256": sha256(binding_path), "expected_binding_sha256": sha256(expected_binding_path), "errors": profile_errors})
        r2_data[step] = detail

    input_keys = ("expert", "action_mask", "state", "body_rgb", "wrist_rgb")
    for step in STEPS[1:]:
        for source in expected:
            first = r2_data[0]["by_id"][source]["arrays"]
            current = r2_data[step]["by_id"][source]["arrays"]
            if any(not np.array_equal(first[key], current[key]) for key in input_keys):
                errors.append(f"source input changed across R2 profiles: step={step} source={source}")

    r0_rows, r0_detail, r0_profile, r0_errors = read_profile(r0, 10000, expected, helpers)
    errors.extend([f"R0 final: {error}" for error in r0_errors])
    r0_card = json.loads((r0 / "pi0243_run_card.json").read_text(encoding="utf-8"))
    if sha256(r0 / "pi0243_run_card.json") != parent["card_sha256"]: errors.append("R0 parent card hash mismatch")
    if r0_card.get("processor") != card.get("processor"):
        errors.append("R0/R2 processor identity differs")
    source_hashes: dict[str, Any] = {}
    for label, run_card in (("R0", r0_card), ("R2", card)):
        code = Path(run_card["paths"]["code_root"])
        source_hashes[label] = {
            rel: sha256(code / rel)
            for rel in (
                "scripts/pi0243_staged_learning_to_sim.py",
                "scripts/pi0228_train_calib_checkpoint_selection.py",
                "src/go2arx5_vla/safety/trajectory_gate.py",
                "src/go2arx5_vla/trajectory/so3.py",
            )
        }
    if source_hashes["R0"] != source_hashes["R2"]:
        errors.append("R0/R2 metric/provider/gate source hashes differ")
    if source_hashes["R2"]["scripts/pi0243_staged_learning_to_sim.py"] != sha256(args.code_root / "scripts/pi0243_staged_learning_to_sim.py"):
        errors.append("validator metric helper differs from executed R2 helper")

    paired: list[dict[str, Any]] = []
    input_mismatches: list[str] = []
    xyz_errors: dict[str, list[np.ndarray]] = {"R0": [], "R2": []}
    delta_errors: dict[str, list[np.ndarray]] = {"R0": [], "R2": []}
    group_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    gate_final: Counter[str] = Counter()
    for source in sorted(expected):
        a, b = r0_detail["by_id"][source], r2_data[10000]["by_id"][source]
        for key in input_keys:
            if not np.array_equal(a["arrays"][key], b["arrays"][key]): input_mismatches.append(f"{source}:{key}")
        am, bm = a["metrics"], b["metrics"]
        row = b["row"]
        group_rows.setdefault((str(row.get("target_family")), str(row.get("phase"))), []).append(row)
        gate_final.update(i["code"] for i in b["gate"]["issues"])
        for label, item in (("R0", a), ("R2", b)):
            err = np.asarray(item["arrays"]["decoded"][0, :, :3], dtype=np.float64) - np.asarray(item["arrays"]["expert"][:, :3], dtype=np.float64)
            xyz_errors[label].append(err)
            delta_errors[label].append(np.diff(err, axis=0))
        pred = np.asarray(b["arrays"]["decoded"][0], dtype=np.float64)
        v = np.linalg.norm(np.diff(pred[:, :3], axis=0), axis=1) / DT
        pred_r0 = np.asarray(a["arrays"]["decoded"][0], dtype=np.float64)
        v_r0 = np.linalg.norm(np.diff(pred_r0[:, :3], axis=0), axis=1) / DT
        paired.append({
            "source_episode_id": source, "family": row.get("target_family"), "phase": row.get("phase"),
            "R0_gate": a["row"]["gate_decision"], "R2_gate": row["gate_decision"],
            "R0_xyz_rmse_m": am["xyz_rmse_m"], "R2_xyz_rmse_m": bm["xyz_rmse_m"], "delta_xyz_m": bm["xyz_rmse_m"]-am["xyz_rmse_m"],
            "R0_so3_rmse_rad": am["so3_rmse_rad"], "R2_so3_rmse_rad": bm["so3_rmse_rad"], "delta_so3_rad": bm["so3_rmse_rad"]-am["so3_rmse_rad"],
            "R0_peak_linear_speed_mps": am["max_linear_speed_mps"], "R2_peak_linear_speed_mps": bm["max_linear_speed_mps"],
            "R0_peak_linear_speed_segment": int(np.argmax(v_r0)),
            "R2_peak_linear_speed_segment": int(np.argmax(v)),
            "R0_peak_angular_speed_radps": am["max_angular_speed_radps"], "R2_peak_angular_speed_radps": bm["max_angular_speed_radps"],
            "R0_input_arrays_exact_match": all(np.array_equal(a["arrays"][k], b["arrays"][k]) for k in input_keys),
        })
    if input_mismatches: errors.append("R0/R2 same-source input arrays differ: " + ",".join(input_mismatches[:8]))

    final_rows = r2_data[10000]["by_id"]
    xyz = np.asarray([item["metrics"]["xyz_rmse_m"] for item in final_rows.values()])
    so3 = np.asarray([item["metrics"]["so3_rmse_rad"] for item in final_rows.values()])
    rejects = sum(item["row"]["gate_decision"] == "REJECT" for item in final_rows.values())
    final = {"gate_rejects": rejects, "median_xyz_rmse_m": float(np.percentile(xyz, 50, method="linear")), "p95_xyz_rmse_m": float(np.percentile(xyz, 95, method="linear")), "median_so3_rmse_rad": float(np.percentile(so3, 50, method="linear"))}
    criteria = {"gate_rejects_zero": rejects == 0, "median_xyz_rmse_le_0p020": final["median_xyz_rmse_m"] <= 0.020, "p95_xyz_rmse_le_0p035": final["p95_xyz_rmse_m"] <= 0.035, "median_so3_rmse_le_0p050": final["median_so3_rmse_rad"] <= 0.050}
    screen_closure = json.loads((r2 / "pi0244x_xyz4_screen_closure.json").read_text(encoding="utf-8"))
    if screen_closure.get("screen_cells") != 80 or screen_closure.get("profile_count") != 5 or screen_closure.get("final_count") != 16: errors.append("runner closure count mismatch")
    if screen_closure.get("run_card_sha256") != card_hash or screen_closure.get("final_10000") != final or screen_closure.get("criteria") != criteria: errors.append("runner closure differs from independent recomputation")

    def decomposition(label: str) -> dict[str, Any]:
        e = np.concatenate(xyz_errors[label], axis=0)
        d = np.concatenate(delta_errors[label], axis=0)
        bias = np.mean(e, axis=0)
        return {"xyz_bias_m": bias.tolist(), "debiased_xyz_residual_rmse_m": float(np.sqrt(np.mean(np.sum((e-bias)**2, axis=1)))), "temporal_delta_error_rmse_m_per_tick": float(np.sqrt(np.mean(np.sum(d**2, axis=1)))), "temporal_velocity_error_rmse_mps": float(np.sqrt(np.mean(np.sum(d**2, axis=1)))/DT)}
    r0d, r2d = decomposition("R0"), decomposition("R2")
    report = {
        "schema": "PI0244_R2_SCREEN_CPU_VALIDATION_V1", "validator_sha256": args.validator_sha256,
        "model_queries": 0, "model_weight_loads": 0, "training_or_optimizer_steps": 0,
        "wbc_steps": 0, "sim_steps": 0, "calib_queries": 0, "dev_e_reads": 0, "final_f_reads": 0,
        "identity": {"job_id": "629335.hopper-m-02", "r2_card_sha256": card_hash, "execution_binding_id": BINDING_ID, "r0_parent_hashes": parent_checks, "R0_card_sha256": sha256(r0 / "pi0243_run_card.json"), "training_manifest_sha256": sha256(manifest_path), "training_closure_sha256": sha256(training_path), "reconciliation_sha256": sha256(reconciliation_path), "screen_closure_sha256": sha256(r2 / "pi0244x_xyz4_screen_closure.json"), "R0_R2_helper_source_hashes": source_hashes},
        "training_valid": training_ok, "profiles": profiles, "R0_final10000": r0_profile, "R2_final10000": final, "criteria": criteria,
        "R0_R2_same_source": {"paired_sources": len(paired), "input_arrays_exact_match": not input_mismatches,
            "runtime_comparison": {"processor_identity_equal": r0_card.get("processor") == card.get("processor"),
                "helper_source_hashes_equal": source_hashes["R0"] == source_hashes["R2"],
                "seed": 22301, "language": LANGUAGE, "R0_gpu_names": r0_profile["gpu_names"],
                "R2_gpu_names": sorted({name for p in profiles[-1:] for name in p["gpu_names"]}),
                "stored_sampler_noise_identity": "NOT_OBSERVED_IN_NPZ; seed and sampler implementation are bound, exact sampled noise tensors are not"},
            "decomposition_definition": "Across 16x40 decoded action points: error=prediction-expert; XYZ bias=mean error vector; debiased residual RMSE=sqrt(mean(||error-bias||^2)); temporal residual is adjacent first difference of error, also divided by dt.",
            "paired": paired, "R0_xyz_decomposition": r0d, "R2_xyz_decomposition": r2d,
            "median_delta_xyz_rmse_m": float(np.median([x["delta_xyz_m"] for x in paired])),
            "median_delta_so3_rmse_rad": float(np.median([x["delta_so3_rad"] for x in paired])),
            "xyz_worsened_windows": sum(x["delta_xyz_m"] > 0 for x in paired),
            "so3_worsened_windows": sum(x["delta_so3_rad"] > 0 for x in paired),
            "R2_final_gate_issue_codes": dict(sorted(gate_final.items())),
            "by_family_phase": {f"{family}/phase{phase}": {"n":len(items), "gate_counts":dict(Counter(x["gate_decision"] for x in items)), "median_xyz_rmse_m":float(np.median([x["xyz_rmse_m"] for x in items])), "median_so3_rmse_rad":float(np.median([x["so3_rmse_rad"] for x in items]))} for (family,phase),items in sorted(group_rows.items())}},
        "verdict": "VALID_FAIL_STOP_FOR_PI_NEW_HYPOTHESIS" if not all(criteria.values()) else "PASS_R2_XYZ4",
        "selected_checkpoint": "NONE", "errors": errors,
    }
    serialized = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        print(json.dumps({"verdict": report["verdict"], "final_10000": report["R2_final10000"], "errors": report["errors"], "output": str(args.output)}))
    else:
        print(serialized, end="")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
