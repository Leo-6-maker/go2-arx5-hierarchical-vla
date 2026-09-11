#!/usr/bin/env python3
"""Stage45D-D2B forward-Euler solver-resolution study.

The audit path is CPU-only and imports no model runtime.  The guarded
execution path performs observed-runtime admission and frozen artifact
verification before importing the model.  It then runs only direct action-head
inference on carried free-running states at E4, E8, and E16 resolution.  The
same CPU float32 initial noise is copied into every solver arm; no arm is ever
reset to a teacher bridge, and no public sampler is called.
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
import stage45d_d2b_runtime_admission as admission  # noqa: E402


PROFILE_PATH = REPO / "manifests" / "STAGE45D_D2B_SOLVER_DISCRETIZATION_EXECUTION_PROFILE_V1.json"
ENGINE_SCHEMA = "CORELAB_STAGE45D_D2B_SOLVER_DISCRETIZATION_ENGINE_V1"
PROFILE_SCHEMA = "CORELAB_STAGE45D_D2B_SOLVER_DISCRETIZATION_EXECUTION_PROFILE_V1"
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


class D2BEngineError(RuntimeError):
    """Raised when D2B cannot prove its execution or evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2BEngineError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = common.read_json(path)
    except Exception as exc:
        raise D2BEngineError(str(exc)) from exc
    return value


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    number = float(value)
    _require(math.isfinite(number), f"{label} is non-finite")
    return number


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
        "forward_euler_only": "FORWARD_EULER_ONLY" in source and "solver_comparison_only" in source,
        "solver_schedules": "SOLVER_SCHEDULES" in source and "E4" in source and "E8" in source and "E16" in source,
        "free_running_state_update": "x_fr_next = (x_fr + dt * v_fr)" in source,
        "free_running_reset_forbidden": "free_running_reset" in source and "False" in source,
        "same_initial_noise": "same_initial_noise_across_solver_arms" in source and "x0 = noise_bank" in source,
        "raw_before_derived": "_write_raw_stage" in source and "write_json_durable(derived_path" in source,
        "normalized_namespace": "D2B_NORMALIZED_FREE_RUNNING_FIELD_DIAGNOSTIC_NOT_SUPERVISED" in source,
        "physical_namespace": "DECODED_PHYSICAL_ACTION" in source,
        "paired_solver_summary": "_paired_solver_summary" in source,
        "late_model_loader": "_load_runtime_after_admission" in source,
        "no_public_sampler": "public_sampler_calls" in source and "get_action_calls" in source,
    }
    return {
        "schema": "CORELAB_STAGE45D_D2B_ENGINE_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": forbidden_calls,
        "markers": markers,
        "top_level_model_imports": False,
    }


def validate_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    payload = _read_json(path)
    _require(payload.get("schema") == PROFILE_SCHEMA, "D2B profile schema drift")
    _require(payload.get("status") == "READY_FOR_EXECUTION_NOT_EXECUTED", "D2B profile status drift")
    authority = _mapping(payload.get("authority"), field="profile.authority")
    _require(authority.get("issue_11_comment") == admission.PI_GATE_COMMENT, "D2B Issue authority drift")
    _require(authority.get("pr_52_comment") == admission.PR52_POINTER_COMMENT, "D2B PR pointer drift")
    _require(payload.get("branch") == admission.BRANCH, "D2B branch drift")
    _require(payload.get("base_execution_head") == admission.BASE_EXECUTION_HEAD, "D2B base HEAD drift")
    for key, expected_path in (
        ("runtime_adapter", REPO / "scripts" / "stage45d_d2b_solver_discretization.py"),
        ("admission", REPO / "scripts" / "stage45d_d2b_runtime_admission.py"),
        ("launcher", REPO / "scripts" / "stage45d_d2b_vanda_h200_launcher.py"),
    ):
        ref = _mapping(payload.get(key), field=f"profile.{key}")
        _require(ref.get("path") == expected_path.relative_to(REPO).as_posix(), f"D2B {key} path drift")
        _require(ref.get("sha256") == common.sha256_file(expected_path), f"D2B {key} SHA drift")
        _require(ref.get("top_level_model_imports") is False, f"D2B {key} top-level model import drift")
    execution = _mapping(payload.get("execution"), field="profile.execution")
    expected = {
        "checkpoint_order": list(admission.CHECKPOINT_ORDER),
        "window_count": admission.WINDOW_COUNT,
        "window_order": admission.WINDOW_ORDER,
        "noise_seed": admission.SEED,
        "solver_order": list(admission.SOLVER_ORDER),
        "solver_schedules": admission.SOLVER_SCHEDULES,
        "free_running_reset": False,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "full_action_head_calls": admission.FULL_ACTION_HEAD_CALLS,
        "full_conditioning_passes": admission.FULL_CONDITIONING_PASSES,
        "full_total_model_side_calls": admission.FULL_TOTAL_MODEL_SIDE_CALLS,
        "smoke_action_head_calls": admission.SMOKE_ACTION_HEAD_CALLS,
        "smoke_conditioning_passes": admission.SMOKE_CONDITIONING_PASSES,
        "smoke_total_model_side_calls": admission.SMOKE_ACTION_HEAD_CALLS + admission.SMOKE_CONDITIONING_PASSES,
        "adaptive_extension": False,
    }
    for key, value in expected.items():
        _require(execution.get(key) == value, f"D2B execution {key} drift")
    _require(payload.get("contract") == admission.CONTRACT, "D2B science contract drift")
    _require(payload.get("locked_boundaries") == admission.LOCKED_BOUNDARIES, "D2B locked boundary drift")
    guard = _static_guard()
    _require(guard["status"] == "PASS", "D2B engine static guard failed")
    return payload


def validate_static_contract() -> dict[str, Any]:
    profile = validate_profile()
    admission_static = admission.validate_static_contract()
    d1_static = d1_full.validate_static_contract()
    guard = _static_guard()
    _require(guard["status"] == "PASS", "D2B static guard failed")
    return {
        "schema": ENGINE_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "profile_sha256": common.sha256_file(PROFILE_PATH),
        "engine_sha256": common.sha256_file(Path(__file__)),
        "admission_sha256": common.sha256_file(REPO / "scripts" / "stage45d_d2b_runtime_admission.py"),
        "launcher_sha256": common.sha256_file(REPO / "scripts" / "stage45d_d2b_vanda_h200_launcher.py"),
        "static_guard": guard,
        "d2b_admission": admission_static,
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
        "schema": "CORELAB_STAGE45D_D2B_RAW_STAGE_MANIFEST_V1",
        "status": "RAW_SEALED_BEFORE_DERIVED",
        "identity": dict(identity),
        "raw_archive": {"path": str(raw_path), "sha256": common.sha256_file(raw_path), "bytes": raw_path.stat().st_size},
        "arrays": {
            key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)}
            for key, value in arrays.items()
        },
        "direct_model_outputs": ["v_fr"],
        "algebraic_states": ["x0", "x1", "x_fr", "x_fr_next", "v_target"],
        "same_initial_noise_across_solver_arms": True,
    }
    manifest_path = stage_root / "raw_manifest.json"
    common.write_json_durable(manifest_path, manifest)
    return {
        "path": str(raw_path),
        "sha256": common.sha256_file(raw_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": common.sha256_file(manifest_path),
    }


def _vector_metrics(lhs: Any, rhs: Any, active_mask: Any = MASK) -> dict[str, float]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    if left.ndim == 3:
        left = left[0]
    if right.ndim == 3:
        right = right[0]
    mask = np.asarray(active_mask, dtype=bool)
    _require(left.ndim == 2 and right.ndim == 2 and mask.ndim == 2 and left.shape == right.shape == mask.shape, "D2B vector shape/mask drift")
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "D2B vector contains non-finite values")
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
    _require(all(bool(torch.isfinite(item).all().item()) for item in bank), "D2B noise bank is non-finite")
    return bank


def _model_call(torch: Any, stage38: Any, model: Any, conditioning: tuple[Any, Any, Any], state_np: np.ndarray, timestep_bucket: int, counters: dict[str, Any]) -> np.ndarray:
    processed_backbone, action_input, state_features = conditioning
    device = processed_backbone.backbone_features.device
    state = torch.from_numpy(np.ascontiguousarray(state_np, dtype=np.float32)).to(device=device, dtype=next(model.parameters()).dtype)
    started = time.perf_counter()
    with torch.inference_mode():
        velocity = stage38._offline_step(model.action_head, processed_backbone, action_input, state_features, state, int(timestep_bucket))
    counters["action_head_calls"] += 1
    counters["model_forward"] += 1
    counters["forward_latency_s"].append(time.perf_counter() - started)
    result = velocity.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    _require(result.shape == RAW_SHAPE and np.isfinite(result).all(), "D2B action-head output shape/finite drift")
    return result


def _physical_endpoint(label: str, raw: np.ndarray, snapshot: Any, snapshot_record: Mapping[str, Any], paths: Mapping[str, Path], stage45c: Any, output_dir: Path, stem: str) -> dict[str, Any]:
    physical, decoder = d1_full._decode_prediction(label, raw, snapshot, paths, common.harness, stage45c)
    physical = np.asarray(physical, dtype=np.float64)
    _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "D2B physical endpoint shape/finite drift")
    physical_path = output_dir / f"{stem}.npy"
    _write_npy_durable(physical_path, physical.astype(np.float32))
    from go2arx5_vla.evaluation.trajectory_metrics import comparison_metrics

    metrics = comparison_metrics(
        physical,
        np.asarray(snapshot.physical_action, dtype=np.float64),
        np.asarray(snapshot_record["anchor_world_m"], dtype=np.float64),
        np.asarray(snapshot.target_position_world_m, dtype=np.float64),
        prefix=f"D2B:{label}:{snapshot.example_id}:{stem}",
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
    v_fr = arrays["v_fr"]
    v_target = arrays["v_target"]
    derived: dict[str, Any] = {
        "schema": "CORELAB_STAGE45D_D2B_DERIVED_STAGE_METRICS_V1",
        "status": "DERIVED_AFTER_RAW_SEAL",
        "raw_before_derived": True,
        "identity": dict(identity),
        "normalized_free_running_field_diagnostic": _blockwise_vector_metrics(
            v_fr, v_target,
            namespace="D2B_NORMALIZED_FREE_RUNNING_FIELD_DIAGNOSTIC_NOT_SUPERVISED",
            difference="v_FR_minus_v_target_descriptive_only",
        ),
        "normalized_state_to_endpoint": _blockwise_vector_metrics(
            arrays["x_fr_next"], arrays["x1"],
            namespace="D2B_NORMALIZED_STATE_TO_ENDPOINT",
            difference="x_FR_next_minus_x1",
        ),
        "normalized_state_step": _blockwise_vector_metrics(
            arrays["x_fr_next"], arrays["x_fr"],
            namespace="D2B_NORMALIZED_EULER_STATE_STEP",
            difference="x_FR_next_minus_x_FR",
        ),
        "path_update": {
            "solver": identity["solver"],
            "solver_family": "FORWARD_EULER",
            "dt": identity["solver_dt"],
            "free_running_reset": False,
            "formula": "x_FR_next = x_FR + dt * v_FR",
        },
        "raw": raw_info,
    }
    if final_stage:
        physical = _physical_endpoint(
            label, arrays["x_fr_next"], snapshot, snapshot_record, paths, stage45c,
            stage_root / "physical", "free_running_endpoint",
        )
        derived["physical"] = {
            "namespace": "DECODED_PHYSICAL_ACTION",
            "source": "final_free_running_state_after_last_forward_euler_update",
            "free_running_endpoint": physical,
        }
    derived_path = stage_root / "derived_metrics.json"
    common.write_json_durable(derived_path, derived)
    _require(derived_path.is_file(), "D2B derived seal is absent")
    row = {
        "schema": "CORELAB_STAGE45D_D2B_STAGE_ROW_V1",
        "identity": dict(identity),
        "normalized_free_running_field_diagnostic": derived["normalized_free_running_field_diagnostic"],
        "normalized_state_to_endpoint": derived["normalized_state_to_endpoint"],
        "normalized_state_step": derived["normalized_state_step"],
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
    for category in ("normalized_free_running_field_diagnostic", "normalized_state_to_endpoint", "normalized_state_step"):
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
    if not isinstance(physical, Mapping):
        return values
    endpoint = physical.get("free_running_endpoint")
    if not isinstance(endpoint, Mapping):
        return values

    wanted = {
        "initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m",
        "absolute_progress_m", "best_progress_m", "endpoint_target_vector_cosine",
        "position_rmse_m", "position_max_abs_m", "endpoint_error_m",
        "prediction_path_length_m", "expert_path_length_m",
        "roughness_ratio", "eef_rot6d_rmse", "eef_rot6d_max_abs",
        "gripper_rmse", "gripper_max_abs", "full_action_rmse_mixed_units",
    }
    summary_wanted = {
        "prediction_roughness_position_second_difference_m",
        "expert_roughness_position_second_difference_m",
        "roughness_position_second_difference_m",
    }

    def visit(value: Any, prefix: str) -> None:
        if not isinstance(value, Mapping):
            return
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key in wanted and isinstance(item, (int, float)) and not isinstance(item, bool):
                values[f"physical.free_running_endpoint.{key}"] = _finite(item, path)
            if key in summary_wanted and isinstance(item, Mapping):
                for summary_key in ("mean", "median", "min", "max"):
                    summary_value = item.get(summary_key)
                    if isinstance(summary_value, (int, float)) and not isinstance(summary_value, bool):
                        values[f"physical.free_running_endpoint.{key}.{summary_key}"] = _finite(summary_value, f"{path}.{summary_key}")
            if isinstance(item, Mapping):
                visit(item, path)

    visit(endpoint.get("trajectory_metrics"), "trajectory_metrics")
    return values


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    vals = [_finite(value, "descriptive value") for value in values]
    ordered = sorted(vals)
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std_population": statistics.pstdev(vals),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, field: str = "metric_values") -> dict[str, Any]:
    _require(rows, "D2B summary group is empty")
    names = sorted(rows[0][field])
    _require(all(set(row[field]) == set(names) for row in rows), "D2B summary metric schema drift")
    signs = Counter()
    for row in rows:
        physical = row.get("physical")
        if isinstance(physical, Mapping):
            endpoint = physical.get("free_running_endpoint")
            if isinstance(endpoint, Mapping):
                sign = endpoint.get("trajectory_metrics", {}).get("target", {}).get("progress_sign")
                if sign:
                    signs[str(sign)] += 1
    result = {"record_count": len(rows), "numeric": {name: _descriptive([row[field][name] for row in rows]) for name in names}}
    if signs:
        result["free_running_final_progress_sign_counts"] = dict(sorted(signs.items()))
    return result


def _grouped(rows: Sequence[Mapping[str, Any]], key: str, field: str = "metric_values") -> dict[str, Any]:
    if not rows:
        return {}
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["identity"][key]].append(row)
    return {str(group): _summary(group, field=field) for group, group in sorted(groups.items(), key=lambda item: str(item[0]))}


def _paired_solver_summary(rows: Sequence[Mapping[str, Any]], checkpoint: str, left: str, right: str, *, field: str = "metric_values") -> dict[str, Any]:
    index: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = row["identity"]
        if identity.get("checkpoint") != checkpoint:
            continue
        index[(str(identity["window_id"]), str(identity["seed"]))][str(identity["solver"])] = row
    pairs: list[dict[str, Any]] = []
    for key, entries in sorted(index.items()):
        if left not in entries or right not in entries:
            continue
        lrow, rrow = entries[left], entries[right]
        lvalues, rvalues = lrow[field], rrow[field]
        _require(set(lvalues) == set(rvalues), "D2B paired metric schema drift")
        pairs.append({
            "identity": {
                "checkpoint": checkpoint,
                "window_id": key[0],
                "seed": int(key[1]),
                "target_id": lrow["identity"]["target_id"],
            },
            "metric_values": {name: rvalues[name] - lvalues[name] for name in sorted(lvalues)},
        })

    def grouped_pairs(key: str) -> dict[str, Any]:
        groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for pair in pairs:
            groups[pair["identity"][key]].append(pair)
        return {str(group): _summary(group) for group, group in sorted(groups.items(), key=lambda item: str(item[0]))}

    result = {
        "checkpoint": checkpoint,
        "left_solver": left,
        "right_solver": right,
        "delta_definition": "right_minus_left",
        "pair_key": "checkpoint + window_id + seed",
        "value_field": field,
        "paired_record_count": len(pairs),
    }
    if pairs:
        result.update({
            "overall": _summary(pairs),
            "by_target": grouped_pairs("target_id"),
            "by_window": grouped_pairs("window_id"),
            "by_seed": grouped_pairs("seed"),
        })
    else:
        result["overall"] = {"record_count": 0, "numeric": {}}
    return result


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, phase: str, counters: Mapping[str, Any]) -> dict[str, Any]:
    materialized = []
    for row in rows:
        item = dict(row)
        item["metric_values"] = _metric_values(row)
        item["physical_metric_values"] = _physical_metric_values(row)
        materialized.append(item)
    physical_final = [row for row in materialized if isinstance(row.get("physical"), Mapping)]
    paired = {}
    paired_physical = {}
    for checkpoint in admission.CHECKPOINT_ORDER:
        for left, right in (("E4", "E8"), ("E8", "E16"), ("E4", "E16")):
            key = f"{checkpoint}:{right}_minus_{left}"
            paired[key] = _paired_solver_summary(materialized, checkpoint, left, right)
            paired_physical[key] = _paired_solver_summary(physical_final, checkpoint, left, right, field="physical_metric_values")
    return {
        "schema": "CORELAB_STAGE45D_D2B_CPU_SUMMARY_V1",
        "status": "DESCRIPTIVE_ONLY_NO_THRESHOLD",
        "phase": phase,
        "row_count": len(rows),
        "action_head_calls": counters["action_head_calls"],
        "conditioning_passes": counters["conditioning_passes"],
        "total_model_side_calls": counters["action_head_calls"] + counters["conditioning_passes"],
        "by_checkpoint": _grouped(materialized, "checkpoint"),
        "by_solver": _grouped(materialized, "solver"),
        "by_target": _grouped(materialized, "target_id"),
        "by_timestep_bucket": _grouped(materialized, "timestep_bucket"),
        "by_window": _grouped(materialized, "window_id"),
        "by_seed": _grouped(materialized, "seed"),
        "physical_final_readout": {
            "record_count": len(physical_final),
            "by_checkpoint_solver": _grouped(materialized, "checkpoint_solver", "physical_metric_values") if False else {
                f"{checkpoint}:{solver}": _summary([
                    row for row in physical_final
                    if row["identity"]["checkpoint"] == checkpoint and row["identity"]["solver"] == solver
                ], field="physical_metric_values")
                for checkpoint in admission.CHECKPOINT_ORDER
                for solver in admission.SOLVER_ORDER
                if any(row["identity"]["checkpoint"] == checkpoint and row["identity"]["solver"] == solver for row in physical_final)
            },
            "by_checkpoint": _grouped(physical_final, "checkpoint", "physical_metric_values"),
            "by_solver": _grouped(physical_final, "solver", "physical_metric_values"),
            "by_target": _grouped(physical_final, "target_id", "physical_metric_values"),
        },
        "paired_solver_normalized": paired,
        "paired_solver_physical_final_readout": paired_physical,
        "scientific_interpretation": {
            "status": "PI_REVIEW_REQUIRED",
            "normalized_namespace": "D2B_NORMALIZED_FREE_RUNNING_FIELD_DIAGNOSTIC_NOT_SUPERVISED",
            "physical_namespace": "DECODED_PHYSICAL_ACTION",
            "solver_comparison": "E4/E8/E16_FORWARD_EULER_ONLY",
            "no_threshold_or_winner_claim": True,
            "no_teacher_forced_path": True,
        },
    }


def _failure_seal(output_root: Path | None, phase: str, reason: str, counters: Mapping[str, Any]) -> None:
    if output_root is None:
        return
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        serializable = {key: value for key, value in counters.items() if key != "forward_latency_s"}
        common.write_json_durable(output_root / "FAILURE_SEAL.json", {
            "schema": "CORELAB_STAGE45D_D2B_FAILURE_SEAL_V1",
            "status": "FAIL",
            "phase": phase,
            "reason": str(reason),
            "model_execution_started": bool(counters.get("model_import_started") or counters.get("model_construction") or counters.get("model_forward")),
            "counters": serializable,
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
        "raw_stage_count": 0,
        "derived_stage_count": 0,
        "physical_final_readout_count": 0,
        "forward_latency_s": [],
    }
    output_root: Path | None = None
    current_phase = "PRE_MODEL_ADMISSION"
    try:
        admission.validate_future_admission(binding, observed, authority_token=authority_token, phase=phase, imported_modules=sys.modules)
        paths = d1_full._validate_paths(binding)
        output_root = paths["output_root"].resolve()
        output_root.mkdir(parents=True, exist_ok=False)
        common.write_json_durable(output_root / "admission.json", {
            "schema": admission.ADMISSION_RESULT_SCHEMA,
            "status": "PASS_PRE_MODEL",
            "phase": phase,
            "execution_head": binding["execution_checkout"]["authorized_head"],
        })
        common.write_json_durable(output_root / "profile.json", profile)

        # This import is intentionally after admission and CPU artifact checks.
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
        _require(len(windows) == admission.WINDOW_COUNT, "D2B TRAIN24 window materialization drift")
        _require([window["window_id"] for window in windows] == list(d1_full.TARGET_IDS), "D2B TRAIN24 order drift")
        labels_by_checkpoint = {
            label: d1_full._normalized_labels(snapshot_records, paths, stage26, stage38, label)
            for label in admission.CHECKPOINT_ORDER
        }
        noise_bank = _load_noise_bank(torch)
        noise_manifest = {
            "schema": "CORELAB_STAGE45D_D2B_FLOAT32_CPU_NOISE_BANK_V1",
            "seed": admission.SEED,
            "shape": list(RAW_SHAPE),
            "dtype": "float32",
            "device": "cpu",
            "window_count": admission.WINDOW_COUNT,
            "noise_sha256_by_window": [_array_sha(item.detach().cpu().numpy()) for item in noise_bank],
            "same_noise_across_checkpoints": True,
            "same_initial_noise_across_solver_arms": True,
        }
        common.write_json_durable(output_root / "noise_manifest.json", noise_manifest)

        device = torch.device(DEVICE)
        _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted H200 runtime")
        _require(device.index == 0, "D2B may use only cuda:0")
        properties = torch.cuda.get_device_properties(device)
        observed_gpu = _mapping(observed.get("host_gpu_runtime"), field="observed.host_gpu_runtime")
        _require("H200" in str(properties.name).upper(), "selected GPU is not H200")
        _require(int(properties.total_memory) >= MIN_VRAM_BYTES, "selected H200 is below the 80GiB floor")
        _require(str(properties.name) == str(observed_gpu.get("gpu_name", observed_gpu.get("gpu", properties.name))), "observed GPU name differs from runtime")
        vram_binding = d1_full._validate_vram_binding(int(properties.total_memory), observed_gpu)
        counters["gpu"] = 1
        telemetry: dict[str, Any] = {
            "schema": "CORELAB_STAGE45D_D2B_H200_TELEMETRY_V1",
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
            "solver_order": list(admission.SOLVER_ORDER),
        }
        checkpoint_order = ("FULL24@2000",) if phase == "SMOKE" else admission.CHECKPOINT_ORDER
        window_indices = (0,) if phase == "SMOKE" else tuple(range(admission.WINDOW_COUNT))
        raw_jsonl = output_root / "raw_probe_rows.jsonl"
        rows: list[dict[str, Any]] = []
        checkpoint_results = []
        for label in checkpoint_order:
            checkpoint_key = {
                "STRUCTURED@1000": "structured_1000_checkpoint",
                "FULL24@1000": "full24_1000_checkpoint",
                "FULL24@2000": "full24_2000_checkpoint",
            }[label]
            model_start = time.perf_counter()
            model = stage38._load_structured_model(paths[checkpoint_key], DEVICE)
            model.eval()
            counters["model_construction"] += 1
            telemetry["model_load_latency_s_by_checkpoint"][label] = time.perf_counter() - model_start
            hook_count = {arm: 0 for arm in admission.SOLVER_ORDER}
            active_solver = {"name": None}
            hook = model.action_head.model.register_forward_hook(
                lambda *_args, **_kwargs: hook_count.__setitem__(
                    active_solver["name"], hook_count[active_solver["name"]] + 1
                ) if active_solver["name"] is not None else None
            )
            checkpoint_stage_rows = 0
            checkpoint_conditioning = 0
            checkpoint_physical = 0
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
                    x0 = noise_bank[index].detach().cpu().numpy().copy().astype(np.float32, copy=False)
                    x1 = np.asarray(labels_by_checkpoint[label][index], dtype=np.float32)
                    x0_sha = _array_sha(x0)
                    for solver in admission.SOLVER_ORDER:
                        schedule = admission.SOLVER_SCHEDULES[solver]
                        _require(_array_sha(x0) == x0_sha, "D2B initial noise changed across solver arms")
                        active_solver["name"] = solver
                        x_fr = x0.copy()
                        for stage_index, (t, timestep_bucket) in enumerate(zip(schedule["t_values"], schedule["t_buckets"])):
                            dt = float(schedule["dt"])
                            _, _, v_target = common.harness.construct_flow(x0, x1, float(t))
                            v_fr = _model_call(torch, stage38, model, conditioning, x_fr, timestep_bucket, counters)
                            x_fr_next = (x_fr + dt * v_fr).astype(np.float32, copy=False)
                            _require(np.isfinite(x_fr_next).all(), "D2B free-running state is non-finite")
                            final_stage = stage_index == schedule["steps"] - 1
                            identity = {
                                "checkpoint": label,
                                "checkpoint_tree_sha256": admission.CHECKPOINT_IDENTITIES[label]["tree_sha256"],
                                "stats_route": admission.STATS_ROUTES[label],
                                "stats_sha256": admission.CHECKPOINT_IDENTITIES[label]["stats_sha256"],
                                "window_id": window["window_id"],
                                "target_id": snapshot.target_id,
                                "window_index": index,
                                "seed": admission.SEED,
                                "noise_seed": admission.SEED,
                                "initial_noise_sha256": x0_sha,
                                "solver": solver,
                                "solver_family": "FORWARD_EULER",
                                "solver_steps": schedule["steps"],
                                "solver_dt": dt,
                                "t": float(t),
                                "timestep_bucket": int(timestep_bucket),
                                "timestep_bucket_scale": admission.TIMESTEP_BUCKET_SCALE,
                                "stage_index": stage_index,
                                "final_stage": final_stage,
                                "free_running_reset": False,
                                "target_xyz_in_model_input": False,
                            }
                            stage_root = output_root / "probes" / label.replace("@", "_at_") / solver / f"w{index:02d}" / f"seed{admission.SEED}" / f"step{stage_index:02d}_t{timestep_bucket:03d}"
                            _, row = _write_stage(
                                stage_root,
                                identity,
                                {
                                    "x0": x0,
                                    "x1": x1,
                                    "x_fr": x_fr,
                                    "v_target": np.asarray(v_target, dtype=np.float32),
                                    "v_fr": v_fr,
                                    "x_fr_next": x_fr_next,
                                    "mask": MASK,
                                },
                                label=label,
                                snapshot=snapshot,
                                snapshot_record=snapshot_record,
                                paths=paths,
                                stage45c=stage45c,
                                final_stage=final_stage,
                            )
                            rows.append(row)
                            _write_jsonl_durable(raw_jsonl, row)
                            counters["raw_stage_count"] += 1
                            counters["derived_stage_count"] += 1
                            checkpoint_stage_rows += 1
                            if final_stage:
                                counters["physical_final_readout_count"] += 1
                                checkpoint_physical += 1
                            x_fr = x_fr_next.copy()
            finally:
                hook.remove()
            expected_by_solver = {solver: len(window_indices) * admission.SOLVER_STEPS[solver] for solver in admission.SOLVER_ORDER}
            for solver in admission.SOLVER_ORDER:
                _require(hook_count[solver] == expected_by_solver[solver], f"D2B action-head count drift for {label}/{solver}: {hook_count[solver]} != {expected_by_solver[solver]}")
            _require(checkpoint_stage_rows == len(window_indices) * sum(admission.SOLVER_STEPS.values()), f"D2B stage count drift for {label}")
            _require(checkpoint_physical == len(window_indices) * len(admission.SOLVER_ORDER), f"D2B physical final readout drift for {label}")
            checkpoint_results.append({
                "checkpoint": label,
                "window_count": len(window_indices),
                "conditioning_passes": checkpoint_conditioning,
                "stage_count": checkpoint_stage_rows,
                "action_head_calls_by_solver": hook_count,
                "physical_final_readouts": checkpoint_physical,
            })
            del model
            torch.cuda.empty_cache()
        expected = {
            "SMOKE": {
                "model_construction": 1, "conditioning_passes": 1,
                "action_head_calls": admission.SMOKE_ACTION_HEAD_CALLS,
                "model_forward": admission.SMOKE_ACTION_HEAD_CALLS,
                "raw_stage_count": sum(admission.SOLVER_STEPS.values()),
                "physical_final_readout_count": len(admission.SOLVER_ORDER),
            },
            "FULL": {
                "model_construction": 3, "conditioning_passes": admission.FULL_CONDITIONING_PASSES,
                "action_head_calls": admission.FULL_ACTION_HEAD_CALLS,
                "model_forward": admission.FULL_ACTION_HEAD_CALLS,
                "raw_stage_count": admission.FULL_ACTION_HEAD_CALLS,
                "physical_final_readout_count": admission.FULL_CONDITIONING_PASSES * len(admission.SOLVER_ORDER),
            },
        }[phase]
        for key, value in expected.items():
            _require(counters[key] == value, f"D2B {phase} {key} drift: {counters[key]} != {value}")
        _require(counters["public_sampler_calls"] == 0 and counters["get_action_calls"] == 0, "D2B forbidden public inference API was called")
        telemetry["forward_count"] = counters["model_forward"]
        telemetry["forward_latency_mean_s"] = statistics.fmean(counters["forward_latency_s"]) if counters["forward_latency_s"] else 0.0
        telemetry["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        telemetry["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        telemetry["resource_margin"] = d1_full._resource_margin(torch, device, int(properties.total_memory))
        _require(telemetry["resource_margin"]["pass"], "D2B H200 resource margin below locked floor")
        cpu_summary = _aggregate_rows(rows, phase=phase, counters=counters)
        common.write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        common.write_json_durable(output_root / "telemetry.json", telemetry)
        seal = {
            "schema": "CORELAB_STAGE45D_D2B_H200_EXECUTION_SEAL_V1",
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
                "physical_final_readout_count": counters["physical_final_readout_count"],
                "normalized_namespace": "D2B_NORMALIZED_FREE_RUNNING_FIELD_DIAGNOSTIC_NOT_SUPERVISED",
                "physical_namespace": "DECODED_PHYSICAL_ACTION",
                "solver_order": list(admission.SOLVER_ORDER),
                "same_initial_noise_across_solver_arms": True,
                "paired_solver_summary_in_cpu_summary": True,
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
                "D2B_solver_comparison": counters["action_head_calls"],
                "adaptive_extension": False,
                "automatic_retry": False,
            },
        }
        common.write_json_durable(output_root / ("d2b_smoke_seal.json" if phase == "SMOKE" else "d2b_full_seal.json"), seal)
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
            _require(os.environ.get("STAGE45D_D2B_RUNTIME") == "authorized", "D2B H200 runtime is not authorized")
            token = os.environ.get("STAGE45D_D2B_AUTHORITY_TOKEN", "")
            result = execute(_read_json(args.runtime_binding), profile=validate_profile(args.profile), authority_token=token, phase=args.phase)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45D_D2B_ENGINE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
