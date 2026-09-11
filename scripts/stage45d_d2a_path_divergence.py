#!/usr/bin/env python3
"""Stage45D-D2A matched teacher-bridge/free-running path-divergence engine.

The static/audit modes never import Torch, GR00T, CUDA, or a model.  The
guarded execution path performs D2A admission first, then uses the accepted
D1 conditioning and direct action-head graph.  It makes two independent
action-head calls at every frozen time (including t=0), carries the
free-running state forward without reset, seals raw arrays before derived
metrics, and emits descriptive CPU summaries without a scientific threshold.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45d_d1_corelab_runtime_adapter as common  # noqa: E402
import stage45d_d1_corelab_full_runtime_adapter as d1_full  # noqa: E402
import stage45d_d2a_runtime_admission as admission  # noqa: E402


PROFILE_PATH = REPO / "manifests" / "STAGE45D_D2A_PATH_DIVERGENCE_EXECUTION_PROFILE_V1.json"
ENGINE_SCHEMA = "CORELAB_STAGE45D_D2A_PATH_DIVERGENCE_ENGINE_V1"
PROFILE_SCHEMA = "CORELAB_STAGE45D_D2A_PATH_DIVERGENCE_EXECUTION_PROFILE_V1"
DEVICE = "cuda:0"
GPU_ROLE = "HOPPER_H200_SINGLE_GPU"
MIN_VRAM_BYTES = 80 * 1024**3
RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
MASK = np.asarray(common.structured_action_mask("T16"), dtype=np.float32)
BLOCK_SLICES = {
    "geometry_xyz": (slice(0, 15), slice(0, 3)),
    "geometry_rot6d": (slice(0, 15), slice(3, 9)),
    "timing": (slice(0, 15), slice(9, 10)),
    "gripper": (slice(0, 40), slice(10, 11)),
}
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
FORBIDDEN_CALLS = {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}


class D2AEngineError(RuntimeError):
    """Raised when D2A cannot prove its execution or evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2AEngineError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = common.read_json(path)
    except Exception as exc:
        raise D2AEngineError(str(exc)) from exc
    return value


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    value = float(value)
    _require(math.isfinite(value), f"{label} is non-finite")
    return value


def _static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_imports = sorted(top_level.intersection(FORBIDDEN_IMPORTS))
    forbidden_calls = sorted(calls.intersection(FORBIDDEN_CALLS))
    markers = {
        "teacher_bridge_formula": "x_tf = ((1.0 - t) * x0 + t * x1)" in source,
        "free_running_state_update": "x_fr + admission.DT * v_fr" in source,
        "free_running_reset_forbidden": "free_running_reset" in source and "False" in source,
        "independent_t_zero_calls": "v_tf = _model_call" in source and "v_fr = _model_call" in source,
        "raw_before_derived": "_write_raw_stage" in source and "write_json_durable(derived_path" in source,
        "teacher_namespace": "NORMALIZED_LOCAL_FLOW" in source,
        "off_path_not_supervised": "not_supervised" in source,
        "paired_summary": "_paired_summary" in source,
        "late_model_loader": "_load_runtime_after_admission" in source,
    }
    return {
        "schema": "CORELAB_STAGE45D_D2A_ENGINE_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": forbidden_calls,
        "markers": markers,
        "top_level_model_imports": False,
    }


def validate_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    payload = _read_json(path)
    _require(payload.get("schema") == PROFILE_SCHEMA, "D2A profile schema drift")
    _require(payload.get("status") == "READY_FOR_EXECUTION_NOT_EXECUTED", "D2A profile status drift")
    authority = _mapping(payload.get("authority"), field="profile.authority")
    _require(authority.get("issue_11_comment") == admission.PI_GATE_COMMENT, "D2A Issue authority drift")
    _require(authority.get("pr_51_comment") == 5631608714, "D2A PR pointer drift")
    _require(authority.get("issue_50_comment") == 5631610825, "D2A Issue #50 pointer drift")
    _require(payload.get("branch") == admission.BRANCH, "D2A branch drift")
    _require(payload.get("base_execution_head") == admission.BASE_EXECUTION_HEAD, "D2A base HEAD drift")
    runtime = _mapping(payload.get("runtime_adapter"), field="profile.runtime_adapter")
    _require(runtime.get("path") == "scripts/stage45d_d2a_path_divergence.py", "D2A engine path drift")
    _require(runtime.get("sha256") == common.sha256_file(Path(__file__)), "D2A engine SHA drift")
    admission_ref = _mapping(payload.get("admission"), field="profile.admission")
    _require(admission_ref.get("path") == "scripts/stage45d_d2a_runtime_admission.py", "D2A admission path drift")
    _require(admission_ref.get("sha256") == common.sha256_file(REPO / "scripts/stage45d_d2a_runtime_admission.py"), "D2A admission SHA drift")
    execution = _mapping(payload.get("execution"), field="profile.execution")
    expected = {
        "checkpoint_order": list(admission.CHECKPOINT_ORDER),
        "window_count": admission.WINDOW_COUNT,
        "window_order": admission.WINDOW_ORDER,
        "noise_seed": admission.SEED,
        "t_values": list(admission.T_VALUES),
        "t_buckets": list(admission.T_BUCKETS),
        "dt": admission.DT,
        "path_order": list(admission.PATH_ORDER),
        "free_running_reset": False,
        "t_zero_calls_independent": True,
        "t_zero_deterministic_equality_required": True,
        "full_action_head_calls": 576,
        "full_conditioning_passes": 72,
        "full_total_model_side_calls": 648,
        "smoke_action_head_calls": 8,
        "smoke_conditioning_passes": 1,
        "smoke_total_model_side_calls": 9,
    }
    for key, value in expected.items():
        _require(execution.get(key) == value, f"D2A execution {key} drift")
    _require(payload.get("contract") == admission.CONTRACT, "D2A science contract drift")
    _require(payload.get("locked_boundaries") == admission.LOCKED_BOUNDARIES, "D2A locked boundary drift")
    guard = _static_guard()
    _require(guard["status"] == "PASS", "D2A engine static guard failed")
    return payload


def validate_static_contract() -> dict[str, Any]:
    profile = validate_profile()
    admission_static = admission.validate_static_contract()
    d1_static = d1_full.validate_static_contract()
    guard = _static_guard()
    _require(guard["status"] == "PASS", "D2A static guard failed")
    return {
        "schema": ENGINE_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "profile_sha256": common.sha256_file(PROFILE_PATH),
        "engine_sha256": common.sha256_file(Path(__file__)),
        "admission_sha256": common.sha256_file(REPO / "scripts/stage45d_d2a_runtime_admission.py"),
        "static_guard": guard,
        "d2a_admission": admission_static,
        "d1_artifact_static_contract": d1_static,
        "execution": profile["execution"],
        "return_tokens": {"smoke": admission.SMOKE_PASS_TOKEN, "full": admission.FULL_PASS_TOKEN},
    }


def _write_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_durable(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npy_durable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_raw_stage(stage_root: Path, identity: Mapping[str, Any], arrays: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = stage_root / "raw_arrays.npz"
    _write_npz_durable(raw_path, arrays)
    manifest = {
        "schema": "CORELAB_STAGE45D_D2A_RAW_STAGE_MANIFEST_V1",
        "status": "RAW_SEALED_BEFORE_DERIVED",
        "identity": dict(identity),
        "raw_archive": {"path": str(raw_path), "sha256": common.sha256_file(raw_path), "bytes": raw_path.stat().st_size},
        "arrays": {
            key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)}
            for key, value in arrays.items()
        },
        "direct_model_outputs": ["v_tf", "v_fr"],
        "algebraic_states": ["x_tf", "x_fr", "x_tf_next", "x_fr_next", "v_target"],
    }
    manifest_path = stage_root / "raw_manifest.json"
    common.write_json_durable(manifest_path, manifest)
    return {"path": str(raw_path), "sha256": common.sha256_file(raw_path), "manifest_path": str(manifest_path), "manifest_sha256": common.sha256_file(manifest_path)}


def _vector_metrics(lhs: Any, rhs: Any, active_mask: Any = MASK) -> dict[str, float]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    if left.ndim == 3:
        left = left[0]
    if right.ndim == 3:
        right = right[0]
    mask = np.asarray(active_mask, dtype=bool)
    _require(
        left.ndim == 2
        and right.ndim == 2
        and mask.ndim == 2
        and left.shape == right.shape == mask.shape,
        "D2A vector shape/mask drift",
    )
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "D2A vector contains non-finite values")
    left_flat, right_flat = left[mask], right[mask]
    difference = left_flat - right_flat
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    return {
        "l2": float(np.linalg.norm(difference)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.dot(left_flat, right_flat) / denominator) if denominator else 0.0,
    }


def _blockwise_vector_metrics(lhs: Any, rhs: Any, *, namespace: str, difference: str) -> dict[str, Any]:
    overall = _vector_metrics(lhs, rhs)
    left = np.asarray(lhs, dtype=np.float64)[0]
    right = np.asarray(rhs, dtype=np.float64)[0]
    blocks = {
        name: _vector_metrics(left[rows, cols][None, ...], right[rows, cols][None, ...], MASK[rows, cols])
        for name, (rows, cols) in BLOCK_SLICES.items()
    }
    return {"namespace": namespace, "difference": difference, "overall": overall, "blocks": blocks}


def _load_noise_bank(torch: Any) -> list[Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(admission.SEED)
    bank = [torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous() for _ in range(admission.WINDOW_COUNT)]
    _require(all(bool(torch.isfinite(item).all().item()) for item in bank), "D2A noise bank is non-finite")
    return bank


def _model_call(torch: Any, stage38: Any, model: Any, conditioning: tuple[Any, Any, Any], state_np: np.ndarray, bucket: int, counters: dict[str, Any]) -> np.ndarray:
    processed_backbone, action_input, state_features = conditioning
    device = processed_backbone.backbone_features.device
    state = torch.from_numpy(np.ascontiguousarray(state_np, dtype=np.float32)).to(device=device, dtype=next(model.parameters()).dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        velocity = stage38._offline_step(model.action_head, processed_backbone, action_input, state_features, state, int(bucket))
    counters["action_head_calls"] += 1
    counters["model_forward"] += 1
    counters["forward_latency_s"].append(time.perf_counter() - started)
    result = velocity.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    _require(result.shape == RAW_SHAPE and np.isfinite(result).all(), "D2A action-head output shape/finite drift")
    return result


def _physical_endpoint(label: str, raw: np.ndarray, snapshot: Any, snapshot_record: Mapping[str, Any], paths: Mapping[str, Path], stage45c: Any, output_dir: Path, stem: str) -> dict[str, Any]:
    physical, decoder = d1_full._decode_prediction(label, raw, snapshot, paths, common.harness, stage45c)
    physical = np.asarray(physical, dtype=np.float64)
    _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "D2A physical endpoint shape/finite drift")
    physical_path = output_dir / f"{stem}.npy"
    _write_npy_durable(physical_path, physical.astype(np.float32))
    from go2arx5_vla.evaluation.trajectory_metrics import comparison_metrics

    metrics = comparison_metrics(
        physical,
        np.asarray(snapshot.physical_action, dtype=np.float64),
        np.asarray(snapshot_record["anchor_world_m"], dtype=np.float64),
        np.asarray(snapshot.target_position_world_m, dtype=np.float64),
        prefix=f"D2A:{label}:{snapshot.example_id}:{stem}",
    )
    return {
        "array_path": str(physical_path),
        "array_sha256": common.sha256_file(physical_path),
        "array_shape": list(physical.shape),
        "decoder": decoder,
        "trajectory_metrics": metrics,
    }


def _write_stage(stage_root: Path, identity: Mapping[str, Any], arrays: Mapping[str, Any], *, label: str, snapshot: Any, snapshot_record: Mapping[str, Any], paths: Mapping[str, Path], stage45c: Any, final_stage: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_info = _write_raw_stage(stage_root, identity, arrays)
    v_tf = arrays["v_tf"]
    v_fr = arrays["v_fr"]
    v_target = arrays["v_target"]
    derived: dict[str, Any] = {
        "schema": "CORELAB_STAGE45D_D2A_DERIVED_STAGE_METRICS_V1",
        "status": "DERIVED_AFTER_RAW_SEAL",
        "raw_before_derived": True,
        "identity": dict(identity),
        "state_path_divergence": _blockwise_vector_metrics(arrays["x_fr"], arrays["x_tf"], namespace="D2A_STATE_PATH_DIVERGENCE", difference="x_FR_minus_x_TF"),
        "field_sensitivity": _blockwise_vector_metrics(v_fr, v_tf, namespace="D2A_FIELD_SENSITIVITY", difference="v_FR_minus_v_TF"),
        "teacher_bridge_local_error": _blockwise_vector_metrics(v_tf, v_target, namespace="NORMALIZED_LOCAL_FLOW", difference="v_TF_minus_v_target"),
        "off_path_diagnostic": _blockwise_vector_metrics(v_fr, v_target, namespace="DESCRIPTIVE_OFF_PATH_DIAGNOSTIC_NOT_SUPERVISED", difference="v_FR_minus_v_target_not_supervised"),
        "t_zero": {
            "checked": identity["t_bucket"] == 0,
            "independent_calls": True,
            "exact_equal": bool(np.array_equal(v_tf, v_fr)) if identity["t_bucket"] == 0 else None,
            "max_abs_difference": float(np.max(np.abs(np.asarray(v_tf, dtype=np.float64) - np.asarray(v_fr, dtype=np.float64)))),
        },
        "path_update": {
            "dt": admission.DT,
            "free_running_reset": False,
            "x_tf_next_formula": "x_TF(t)+dt*v_TF(t)",
            "x_fr_next_formula": "x_FR(t)+dt*v_FR(t)",
        },
        "raw": raw_info,
    }
    if final_stage:
        physical_dir = stage_root / "physical"
        tf_endpoint = _physical_endpoint(label, arrays["x_tf_next"], snapshot, snapshot_record, paths, stage45c, physical_dir, "teacher_bridge_implied_endpoint")
        fr_endpoint = _physical_endpoint(label, arrays["x_fr_next"], snapshot, snapshot_record, paths, stage45c, physical_dir, "free_running_endpoint")
        derived["physical"] = {
            "namespace": "DECODED_PHYSICAL_ACTION",
            "source": "D2A_final_endpoint_at_t750",
            "teacher_bridge_implied_endpoint": tf_endpoint,
            "free_running_endpoint": fr_endpoint,
        }
    derived_path = stage_root / "derived_metrics.json"
    common.write_json_durable(derived_path, derived)
    _require(derived_path.is_file(), "D2A derived seal is absent")
    row = {
        "schema": "CORELAB_STAGE45D_D2A_STAGE_ROW_V1",
        "identity": dict(identity),
        "state_path_divergence": derived["state_path_divergence"],
        "field_sensitivity": derived["field_sensitivity"],
        "teacher_bridge_local_error": derived["teacher_bridge_local_error"],
        "off_path_diagnostic": derived["off_path_diagnostic"],
        "t_zero": derived["t_zero"],
        "physical": derived.get("physical"),
        "raw_path": raw_info["path"],
        "raw_sha256": raw_info["sha256"],
        "raw_manifest_path": raw_info["manifest_path"],
        "raw_manifest_sha256": raw_info["manifest_sha256"],
        "derived_path": str(derived_path),
        "derived_sha256": common.sha256_file(derived_path),
        "finite": True,
    }
    return derived, row


def _metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for category in ("state_path_divergence", "field_sensitivity", "teacher_bridge_local_error", "off_path_diagnostic"):
        section = _mapping(row[category], field=category)
        for metric in ("l2", "rmse", "max_abs", "cosine"):
            values[f"{category}.overall.{metric}"] = _finite(section["overall"][metric], f"{category}.overall.{metric}")
        for block_name, block in section["blocks"].items():
            for metric in ("l2", "rmse", "max_abs", "cosine"):
                values[f"{category}.{block_name}.{metric}"] = _finite(block[metric], f"{category}.{block_name}.{metric}")
    return values


def _physical_metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    physical = row.get("physical")
    if isinstance(physical, Mapping):
        for path_name in ("teacher_bridge_implied_endpoint", "free_running_endpoint"):
            endpoint = physical.get(path_name)
            if not isinstance(endpoint, Mapping):
                continue
            target = _mapping(endpoint.get("trajectory_metrics", {}).get("target"), field=f"physical.{path_name}.target")
            for key in ("initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m", "endpoint_target_vector_cosine"):
                if key in target:
                    values[f"physical.{path_name}.target.{key}"] = _finite(target[key], f"physical.{path_name}.target.{key}")
    return values


def _summary(rows: Sequence[Mapping[str, Any]], *, field: str = "metric_values") -> dict[str, Any]:
    _require(rows, "D2A summary group is empty")
    names = sorted(rows[0][field])
    _require(all(set(row[field]) == set(names) for row in rows), "D2A summary metric schema drift")
    signs = Counter()
    for row in rows:
        physical = row.get("physical")
        if isinstance(physical, Mapping):
            endpoint = physical.get("free_running_endpoint")
            if isinstance(endpoint, Mapping):
                sign = endpoint.get("trajectory_metrics", {}).get("target", {}).get("progress_sign")
                if sign:
                    signs[str(sign)] += 1
    result = {
        "record_count": len(rows),
        "numeric": {name: _descriptive([row[field][name] for row in rows]) for name in names},
    }
    if signs:
        result["free_running_final_progress_sign_counts"] = dict(sorted(signs.items()))
    return result


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    vals = [_finite(value, "descriptive value") for value in values]
    ordered = sorted(vals)
    return {"count": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "std_population": statistics.pstdev(vals), "min": ordered[0], "max": ordered[-1]}


def _grouped(rows: Sequence[Mapping[str, Any]], key: str, field: str = "metric_values") -> dict[str, Any]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["identity"][key]].append(row)
    return {str(group): _summary(group, field=field) for group, group in sorted(groups.items(), key=lambda item: (str(item[0])))}


def _paired_summary(rows: Sequence[Mapping[str, Any]], left: str, right: str, *, field: str = "metric_values") -> dict[str, Any]:
    index: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = row["identity"]
        index[(identity["window_id"], str(identity["seed"]), int(identity["t_bucket"]))][identity["checkpoint"]] = row
    pairs: list[dict[str, Any]] = []
    for key, entries in sorted(index.items()):
        if left not in entries or right not in entries:
            continue
        lrow, rrow = entries[left], entries[right]
        lvalues, rvalues = lrow[field], rrow[field]
        _require(set(lvalues) == set(rvalues), "D2A paired metric schema drift")
        pairs.append({
            "identity": {"window_id": key[0], "seed": int(key[1]), "t_bucket": key[2], "target_id": lrow["identity"]["target_id"]},
            "metric_values": {name: rvalues[name] - lvalues[name] for name in sorted(lvalues)},
        })
    def grouped_pairs(key: str) -> dict[str, Any]:
        groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for pair in pairs:
            groups[pair["identity"][key]].append(pair)
        return {str(group): _summary(group) for group, group in sorted(groups.items(), key=lambda item: str(item[0]))}
    return {
        "left_checkpoint": left,
        "right_checkpoint": right,
        "delta_definition": "right_minus_left",
        "pair_key": "window_id + seed + t_bucket",
        "value_field": field,
        "paired_record_count": len(pairs),
        "overall": _summary(pairs),
        "by_target": grouped_pairs("target_id"),
        "by_t_bucket": grouped_pairs("t_bucket"),
        "by_window": grouped_pairs("window_id"),
        "by_seed": grouped_pairs("seed"),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, phase: str, counters: Mapping[str, Any]) -> dict[str, Any]:
    materialized = []
    for row in rows:
        item = dict(row)
        item["metric_values"] = _metric_values(row)
        item["physical_metric_values"] = _physical_metric_values(row)
        materialized.append(item)
    physical_final = [row for row in materialized if isinstance(row.get("physical"), Mapping)]
    return {
        "schema": "CORELAB_STAGE45D_D2A_CPU_SUMMARY_V1",
        "status": "DESCRIPTIVE_ONLY_NO_THRESHOLD",
        "phase": phase,
        "row_count": len(rows),
        "action_head_calls": counters["action_head_calls"],
        "conditioning_passes": counters["conditioning_passes"],
        "total_model_side_calls": counters["action_head_calls"] + counters["conditioning_passes"],
        "by_checkpoint": _grouped(materialized, "checkpoint"),
        "by_target": _grouped(materialized, "target_id"),
        "by_t_bucket": _grouped(materialized, "t_bucket"),
        "by_window": _grouped(materialized, "window_id"),
        "by_seed": _grouped(materialized, "seed"),
        "physical_final_readout": {
            "record_count": len(physical_final),
            "by_checkpoint": _grouped(physical_final, "checkpoint", "physical_metric_values") if physical_final else {},
            "by_target": _grouped(physical_final, "target_id", "physical_metric_values") if physical_final else {},
        },
        "paired": {
            "FULL24@1000_minus_STRUCTURED@1000": _paired_summary(materialized, "STRUCTURED@1000", "FULL24@1000"),
            "FULL24@2000_minus_FULL24@1000": _paired_summary(materialized, "FULL24@1000", "FULL24@2000"),
        } if phase == "FULL" else {},
        "paired_physical_final_readout": {
            "FULL24@1000_minus_STRUCTURED@1000": _paired_summary(
                physical_final, "STRUCTURED@1000", "FULL24@1000", field="physical_metric_values"
            ),
            "FULL24@2000_minus_FULL24@1000": _paired_summary(
                physical_final, "FULL24@1000", "FULL24@2000", field="physical_metric_values"
            ),
        } if phase == "FULL" else {},
        "scientific_interpretation": {
            "status": "PI_REVIEW_REQUIRED",
            "teacher_bridge_local_error_namespace": "NORMALIZED_LOCAL_FLOW",
            "off_path_v_minus_vtarget": "DESCRIPTIVE_OFF_PATH_DIAGNOSTIC_NOT_SUPERVISED",
            "path_divergence_namespace": "D2A_STATE_PATH_DIVERGENCE",
            "field_sensitivity_namespace": "D2A_FIELD_SENSITIVITY",
            "no_threshold_or_winner_claim": True,
        },
    }


def _failure_seal(output_root: Path | None, phase: str, reason: str, counters: Mapping[str, Any]) -> None:
    if output_root is None:
        return
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        common.write_json_durable(output_root / "FAILURE_SEAL.json", {
            "schema": "CORELAB_STAGE45D_D2A_FAILURE_SEAL_V1",
            "status": "FAIL",
            "phase": phase,
            "reason": str(reason),
            "counters": dict(counters),
            "automatic_retry": False,
            "adaptive_extension": False,
        })
    except Exception:
        pass


def execute(runtime_document: Mapping[str, Any], *, profile: Mapping[str, Any], authority_token: str, phase: str) -> dict[str, Any]:
    binding = _mapping(runtime_document.get("binding", runtime_document), field="runtime_document.binding")
    observed = _mapping(runtime_document.get("observed"), field="runtime_document.observed")
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_construction": 0,
        "conditioning_passes": 0,
        "action_head_calls": 0,
        "model_forward": 0,
        "gpu": 0,
        "pbs": 1,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_write": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "free_running_sampler_calls": 0,
        "retry_count": 0,
        "t_zero_equality_checks": 0,
        "t_zero_equality_failures": 0,
        "raw_stage_count": 0,
        "derived_stage_count": 0,
        "forward_latency_s": [],
    }
    output_root: Path | None = None
    current_phase = "PRE_MODEL_ADMISSION"
    try:
        admission.validate_future_admission(binding, observed, authority_token=authority_token, phase=phase, imported_modules=sys.modules)
        paths = d1_full._validate_paths(binding)
        output_root = paths["output_root"].resolve()
        output_root.mkdir(parents=True, exist_ok=False)
        common.write_json_durable(output_root / "admission.json", {"schema": admission.ADMISSION_RESULT_SCHEMA, "status": "PASS_PRE_MODEL", "phase": phase, "execution_head": binding["execution_checkout"]["authorized_head"]})
        common.write_json_durable(output_root / "profile.json", profile)

        # Stage45C is CPU-only identity verification here; model imports remain below.
        import stage45c_h200_eval_runtime_adapter as stage45c

        artifacts = d1_full._verify_runtime_artifacts(paths, stage45c)
        _require(not admission.model_imports_present(), "model-side import appeared during pre-model artifact verification")
        common.write_json_durable(output_root / "artifact_verification.json", artifacts)
        current_phase = "MODEL_IMPORT_AND_RUNTIME_ADMISSION"
        counters["model_import_started"] = True
        torch, stage26, stage38, stage45c, shared_runtime = d1_full._load_runtime_after_admission()
        build_runtime_processor = shared_runtime[1]
        snapshot_records = artifacts["snapshots"]
        snapshots = [stage26._load_snapshot(Path(record["path"])) for record in snapshot_records]
        runtime_processor, embodiment, modality_configs = build_runtime_processor(
            groot_source=paths["groot_source"],
            base_checkpoint=paths["base_checkpoint"],
            dataset_dir=paths["full24_dataset"],
            modality_path=paths["modality_config"],
            cosmos_snapshot=paths["cosmos_root"] / "snapshots" / admission.SOURCE_IDENTITIES["cosmos_revision"],
        )
        windows = stage45c._prepare_train24_windows(runtime_processor, embodiment, modality_configs, paths["full24_dataset"], snapshot_records)
        _require(len(windows) == admission.WINDOW_COUNT, "D2A TRAIN24 window materialization drift")
        _require([window["window_id"] for window in windows] == list(d1_full.TARGET_IDS), "D2A TRAIN24 order drift")
        labels_by_checkpoint = {
            label: d1_full._normalized_labels(snapshot_records, paths, stage26, stage38, label)
            for label in admission.CHECKPOINT_ORDER
        }
        noise_bank = _load_noise_bank(torch)
        noise_manifest = {
            "schema": "CORELAB_STAGE45D_D2A_FLOAT32_CPU_NOISE_BANK_V1",
            "seed": admission.SEED,
            "shape": list(RAW_SHAPE),
            "dtype": "float32",
            "device": "cpu",
            "window_count": admission.WINDOW_COUNT,
            "noise_sha256_by_window": [_array_sha(item.detach().cpu().numpy()) for item in noise_bank],
            "same_noise_across_checkpoints": True,
        }
        common.write_json_durable(output_root / "noise_manifest.json", noise_manifest)

        device = torch.device(DEVICE)
        _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted H200 runtime")
        _require(device.index == 0, "D2A may use only cuda:0")
        properties = torch.cuda.get_device_properties(device)
        observed_gpu = _mapping(observed.get("host_gpu_runtime"), field="observed.host_gpu_runtime")
        _require("H200" in str(properties.name).upper(), "selected GPU is not H200")
        _require(int(properties.total_memory) >= MIN_VRAM_BYTES, "selected H200 is below the 80GiB floor")
        _require(str(properties.name) == str(observed_gpu.get("gpu_name", observed_gpu.get("gpu", properties.name))), "observed GPU name differs from runtime")
        vram_binding = d1_full._validate_vram_binding(int(properties.total_memory), observed_gpu)
        counters["gpu"] = 1
        telemetry: dict[str, Any] = {
            "schema": "CORELAB_STAGE45D_D2A_H200_TELEMETRY_V1",
            "host": observed_gpu.get("host"),
            "gpu_name": str(properties.name),
            "gpu_uuid": observed_gpu.get("gpu_uuid"),
            "selected_device": DEVICE,
            "total_vram_bytes": int(properties.total_memory),
            "vram_binding": vram_binding,
            "cuda_runtime": str(torch.version.cuda),
            "model_load_latency_s_by_checkpoint": {},
            "conditioning_latency_s_by_checkpoint_window": {},
            "forward_count": 0,
        }
        checkpoint_order = ("FULL24@2000",) if phase == "SMOKE" else admission.CHECKPOINT_ORDER
        window_indices = (0,) if phase == "SMOKE" else tuple(range(admission.WINDOW_COUNT))
        raw_jsonl = output_root / "raw_probe_rows.jsonl"
        rows: list[dict[str, Any]] = []
        checkpoint_results = []
        for label in checkpoint_order:
            checkpoint_key = {"STRUCTURED@1000": "structured_1000_checkpoint", "FULL24@1000": "full24_1000_checkpoint", "FULL24@2000": "full24_2000_checkpoint"}[label]
            model_start = time.perf_counter()
            model = stage38._load_structured_model(paths[checkpoint_key], DEVICE)
            model.eval()
            counters["model_construction"] += 1
            telemetry["model_load_latency_s_by_checkpoint"][label] = time.perf_counter() - model_start
            hook_count = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_count.__setitem__("count", hook_count["count"] + 1))
            checkpoint_stage_rows = 0
            checkpoint_conditioning = 0
            try:
                for index in window_indices:
                    window = windows[index]
                    snapshot = snapshots[index]
                    snapshot_record = snapshot_records[index]
                    conditioning_start = time.perf_counter()
                    with torch.inference_mode():
                        _, action_input, processed_backbone, state_features = stage38._prepare_model_conditioning(model, window["model_inputs"])
                    conditioning = (processed_backbone, action_input, state_features)
                    counters["conditioning_passes"] += 1
                    checkpoint_conditioning += 1
                    telemetry["conditioning_latency_s_by_checkpoint_window"][f"{label}:{window['window_id']}"] = time.perf_counter() - conditioning_start
                    x0 = noise_bank[index].detach().cpu().numpy().copy()
                    x1 = np.asarray(labels_by_checkpoint[label][index], dtype=np.float32)
                    x_fr = x0.copy()
                    for stage_index, (t, bucket) in enumerate(zip(admission.T_VALUES, admission.T_BUCKETS)):
                        x_tf = ((1.0 - t) * x0 + t * x1).astype(np.float32, copy=False)
                        _, _, v_target = common.harness.construct_flow(x0, x1, t)
                        # These are intentionally two separate direct calls, including at t=0.
                        v_tf = _model_call(torch, stage38, model, conditioning, x_tf, bucket, counters)
                        v_fr = _model_call(torch, stage38, model, conditioning, x_fr, bucket, counters)
                        if bucket == 0:
                            counters["t_zero_equality_checks"] += 1
                            if not np.array_equal(v_tf, v_fr):
                                counters["t_zero_equality_failures"] += 1
                                raise D2AEngineError("D2A_DETERMINISTIC_T0_MISMATCH: independent TF/FR outputs differ")
                        x_tf_next = (x_tf + admission.DT * v_tf).astype(np.float32, copy=False)
                        x_fr_next = (x_fr + admission.DT * v_fr).astype(np.float32, copy=False)
                        identity = {
                            "checkpoint": label,
                            "checkpoint_tree_sha256": admission.CHECKPOINT_IDENTITIES[label]["tree_sha256"],
                            "stats_route": admission.STATS_ROUTES[label],
                            "stats_sha256": admission.CHECKPOINT_IDENTITIES[label]["stats_sha256"],
                            "window_id": window["window_id"],
                            "target_id": snapshot.target_id,
                            "window_index": index,
                            "noise_seed": admission.SEED,
                            "t": t,
                            "t_bucket": bucket,
                            "stage_index": stage_index,
                            "path_order": list(admission.PATH_ORDER),
                            "target_xyz_in_model_input": False,
                        }
                        stage_root = output_root / "probes" / label.replace("@", "_at_") / f"w{index:02d}" / f"seed{admission.SEED}" / f"t{bucket:03d}"
                        _derived, row = _write_stage(
                            stage_root,
                            identity,
                            {"x0": x0, "x1": x1, "x_tf": x_tf, "x_fr": x_fr, "v_target": v_target, "v_tf": v_tf, "v_fr": v_fr, "x_tf_next": x_tf_next, "x_fr_next": x_fr_next, "mask": MASK},
                            label=label,
                            snapshot=snapshot,
                            snapshot_record=snapshot_record,
                            paths=paths,
                            stage45c=stage45c,
                            final_stage=(bucket == admission.T_BUCKETS[-1]),
                        )
                        rows.append(row)
                        _write_jsonl_durable(raw_jsonl, row)
                        counters["raw_stage_count"] += 1
                        counters["derived_stage_count"] += 1
                        checkpoint_stage_rows += 1
                        x_fr = x_fr_next.copy()
            finally:
                hook.remove()
            expected_calls = len(window_indices) * len(admission.T_VALUES) * len(admission.PATH_ORDER)
            _require(hook_count["count"] == expected_calls, f"D2A action-head hook count drift for {label}: {hook_count['count']} != {expected_calls}")
            _require(checkpoint_stage_rows == len(window_indices) * len(admission.T_VALUES), f"D2A stage count drift for {label}")
            checkpoint_results.append({"checkpoint": label, "window_count": len(window_indices), "conditioning_passes": checkpoint_conditioning, "stage_count": checkpoint_stage_rows, "action_head_calls": hook_count["count"]})
            del model
            torch.cuda.empty_cache()
        expected = {
            "SMOKE": {"model_construction": 1, "conditioning_passes": 1, "action_head_calls": 8, "model_forward": 8, "raw_stage_count": 4, "t_zero_equality_checks": 1},
            "FULL": {"model_construction": 3, "conditioning_passes": 72, "action_head_calls": 576, "model_forward": 576, "raw_stage_count": 288, "t_zero_equality_checks": 72},
        }[phase]
        for key, value in expected.items():
            _require(counters[key] == value, f"D2A {phase} {key} drift: {counters[key]} != {value}")
        _require(counters["t_zero_equality_failures"] == 0, "D2A t=0 deterministic equality failure")
        telemetry["forward_count"] = counters["model_forward"]
        telemetry["forward_latency_mean_s"] = statistics.fmean(counters["forward_latency_s"]) if counters["forward_latency_s"] else 0.0
        telemetry["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        telemetry["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        telemetry["resource_margin"] = d1_full._resource_margin(torch, device, int(properties.total_memory))
        _require(telemetry["resource_margin"]["pass"], "D2A H200 resource margin below locked floor")
        cpu_summary = _aggregate_rows(rows, phase=phase, counters=counters)
        common.write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        common.write_json_durable(output_root / "telemetry.json", telemetry)
        seal = {
            "schema": "CORELAB_STAGE45D_D2A_H200_EXECUTION_SEAL_V1",
            "status": "PASS",
            "token": admission.SMOKE_PASS_TOKEN if phase == "SMOKE" else admission.FULL_PASS_TOKEN,
            "pi_gate": admission.PI_GATE_COMMENT,
            "phase": phase,
            "execution_head": binding["execution_checkout"]["authorized_head"],
            "branch": binding["execution_checkout"]["authorized_branch"],
            "profile_sha256": common.sha256_file(PROFILE_PATH),
            "binding_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "checkpoint_results": checkpoint_results,
            "counters": {key: value for key, value in counters.items() if key != "forward_latency_s"},
            "telemetry": telemetry,
            "call_budget": admission.CALL_BUDGET,
            "experiment_contract": admission.CONTRACT,
            "evidence": {
                "raw_probe_rows_jsonl": str(raw_jsonl),
                "raw_probe_rows_jsonl_sha256": common.sha256_file(raw_jsonl),
                "cpu_summary": str(output_root / "cpu_summary.json"),
                "cpu_summary_sha256": common.sha256_file(output_root / "cpu_summary.json"),
                "raw_before_derived": True,
                "stage_count": len(rows),
                "physical_final_readout_count": sum(row.get("physical") is not None for row in rows),
                "path_divergence_namespace": "D2A_STATE_PATH_DIVERGENCE",
                "field_sensitivity_namespace": "D2A_FIELD_SENSITIVITY",
                "teacher_bridge_namespace": "NORMALIZED_LOCAL_FLOW",
                "off_path_namespace": "DESCRIPTIVE_OFF_PATH_DIAGNOSTIC_NOT_SUPERVISED",
            },
            "execution_boundary": {
                "model_construction": counters["model_construction"],
                "model_forward": counters["model_forward"],
                "gpu": 1,
                "h200": 1,
                "pbs": 1,
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_write": 0,
                "public_sampler_calls": 0,
                "get_action_calls": 0,
                "free_running_sampler_calls": 0,
                "Validation-E": 0,
                "Holdout-F": 0,
                "WBC": 0,
                "simulator": 0,
                "hardware": 0,
                "D2B_solver_comparison": 0,
                "adaptive_extension": False,
                "automatic_retry": False,
            },
        }
        common.write_json_durable(output_root / ("d2a_smoke_seal.json" if phase == "SMOKE" else "d2a_full_seal.json"), seal)
        return seal
    except Exception as exc:
        _failure_seal(output_root, current_phase, f"{type(exc).__name__}: {exc}", counters)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--phase", choices=("SMOKE", "FULL"), default="FULL")
    args = parser.parse_args(argv)
    try:
        if args.plan_only or args.audit_only:
            result = validate_static_contract()
            result["mode"] = "audit-only" if args.audit_only else "plan-only"
        else:
            _require(args.runtime_binding is not None, "--execute requires --runtime-binding")
            _require(os.environ.get("STAGE45D_D2A_RUNTIME") == "authorized", "D2A H200 runtime is not authorized")
            token = os.environ.get("STAGE45D_D2A_AUTHORITY_TOKEN", "")
            result = execute(_read_json(args.runtime_binding), profile=validate_profile(args.profile), authority_token=token, phase=args.phase)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45D_D2A_ENGINE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
