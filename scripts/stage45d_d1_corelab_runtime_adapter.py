#!/usr/bin/env python3
"""Corelab-local Stage45D-D1 bounded 4080S runtime adapter.

The committed ``--plan-only`` and ``--audit-only`` paths are CPU/static only.
They freeze one explicit preflight profile and verify the Stage45D admission
package, source map, mask, flow algebra, and device policy without importing
Torch, Transformers, GR00T, or CUDA.  The guarded runtime path performs the
future ordering exactly as follows::

    static contract -> observed binding -> validate_future_admission PASS
        -> selected artifact checks -> late model imports/construction
        -> one conditioning pass -> three direct local-field probes

No public sampler, retry, sharding, multi-GPU, training, Validation-E,
Holdout-F, WBC, simulator, or hardware path is implemented here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
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

from go2arx5_vla.structured_action import structured_action_mask  # noqa: E402
import stage45d_d1_teacher_forced_local_flow as harness  # noqa: E402
import stage45d_d1_runtime_admission as admission  # noqa: E402


PROFILE_PATH = REPO / "manifests" / "STAGE45D_D1_CORELAB_4080S_PREFLIGHT_PROFILE_V1.json"
ADAPTER_SCHEMA = "CORELAB_STAGE45D_D1_CORELAB_4080S_RUNTIME_ADAPTER_V1"
PROFILE_SCHEMA = "CORELAB_STAGE45D_D1_CORELAB_4080S_PREFLIGHT_PROFILE_V1"
RETURN_TOKEN = "STAGE45D_D1_CORELAB_RUNTIME_READY_FOR_4080S_PREFLIGHT"
SMOKE_PASS_TOKEN = "STAGE45D_D1_4080S_SMOKE_PASS_AUTO_CONTINUE_FULL"
PI_GATE_COMMENT = 5618259751
SUPPORT_HEAD = "8b5bb8c9f87ff01f79cd918d099a41173d74f7bc"
CHECKPOINT = "FULL24@2000"
CHECKPOINT_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
WINDOW_ID = "stage22_expert_red_target_train_a:expert-0000"
TARGET_ID = "red_target_train_a"
SEED = 2223
T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
T_BUCKETS = (67, 369, 783)
RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
DEVICE = "cuda:0"
GPU_ROLE = "RTX4080S_32GB_PRIMARY"
MIN_VRAM_BYTES = 34359738368
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
REQUIRED_PATHS = (
    "base_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
    "modality_config",
    "base_processor_config",
    "base_processor_statistics",
    "full24_runtime_statistics",
    "runtime_admission",
    "output_root",
)


class AdapterError(RuntimeError):
    """Raised when the Corelab preflight cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"{label} unavailable: {path}") from exc
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def regular_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"{label} unavailable: {path}") from exc
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    regular_directory(root, "tree root")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            _require(child.is_dir() and not child.is_symlink(), f"unsafe tree directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            _require(child.is_file() and not child.is_symlink(), f"unsafe tree file: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _assert_no_model_imports() -> None:
    found = admission.model_imports_present()
    _require(not found, "model-side import already present before admission: " + ",".join(found))


def static_guard() -> dict[str, Any]:
    """Audit this adapter's top-level imports and forbidden execution calls."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    forbidden_top_level = sorted(top_level.intersection(FORBIDDEN_IMPORTS))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            if name in {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}:
                forbidden_calls.append(name)
    return {
        "top_level_forbidden_imports": forbidden_top_level,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "late_import_functions": ["_load_runtime_after_admission"],
        "status": "PASS" if not forbidden_top_level and not forbidden_calls else "FAIL",
    }


def validate_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    payload = read_json(path)
    _require(payload.get("schema") == PROFILE_SCHEMA, "preflight profile schema drift")
    _require(payload.get("status") == "READY_FOR_PI_REVIEW_NOT_EXECUTED", "preflight profile is not static-only")
    _require(payload.get("return_token") == RETURN_TOKEN, "preflight return token drift")
    authority = _mapping(payload.get("authority"), field="profile.authority")
    _require(authority.get("issue_11_comment") == PI_GATE_COMMENT, "PI0109 authority drift")
    _require(authority.get("resource_policy_comment") == 5617773953, "resource policy reference drift")
    _require(authority.get("prepared_from_support_head") == SUPPORT_HEAD, "preflight support-base drift")
    adapter = _mapping(payload.get("runtime_adapter"), field="profile.runtime_adapter")
    _require(adapter.get("path") == "scripts/stage45d_d1_corelab_runtime_adapter.py", "adapter path drift")
    adapter_sha = adapter.get("sha256")
    _require(isinstance(adapter_sha, str) and adapter_sha == sha256_file(Path(__file__)), "adapter SHA drift")
    _require(adapter.get("top_level_model_imports") is False and adapter.get("imports_after_admission_only") is True, "adapter import-order contract drift")
    protocol = _mapping(payload.get("protocol"), field="profile.protocol")
    _require(protocol.get("protocol_sha256") == harness.PROTOCOL_SHA256, "profile protocol SHA drift")
    _require(protocol.get("runtime_binding_schema") == "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V2", "profile runtime binding schema drift")
    _require(protocol.get("scientific_semantics_unchanged") is True, "profile scientific semantics changed")
    checkpoint = _mapping(payload.get("checkpoint"), field="profile.checkpoint")
    for key, expected in {
        "label": CHECKPOINT,
        "tree_sha256": CHECKPOINT_TREE_SHA256,
        "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "stats_sha256": STATS_SHA256,
        "stats_payload_sha256": STATS_PAYLOAD_SHA256,
    }.items():
        _require(checkpoint.get(key) == expected, f"profile checkpoint {key} drift")
    cohort = _mapping(payload.get("cohort"), field="profile.cohort")
    for key, expected in {
        "dataset_id": "FULL24_EXISTING_TRAIN",
        "split": "TRAIN_ONLY",
        "selected_window_index": 0,
        "selected_window_id": WINDOW_ID,
        "selected_target_id": TARGET_ID,
        "window_count_in_future_preflight": 1,
        "target_xyz_in_model_input": False,
    }.items():
        _require(cohort.get(key) == expected, f"profile cohort {key} drift")
    probe = _mapping(payload.get("probe"), field="profile.probe")
    for key, expected in {
        "noise_seed": SEED,
        "t_values": list(T_VALUES),
        "t_buckets": list(T_BUCKETS),
        "noise_shape": list(RAW_SHAPE),
        "same_x0_across_t": True,
        "x_t_formula": "x_t=(1-t)x0+t*x1",
        "v_target_formula": "v_target=x1-x0",
        "implied_endpoint_formula": "x1_hat=x_t+(1-t)*v_hat",
    }.items():
        _require(probe.get(key) == expected, f"profile probe {key} drift")
    model_execution = _mapping(payload.get("model_execution"), field="profile.model_execution")
    for key, expected in {
        "conditioning_passes": 1,
        "action_head_calls": 3,
        "total_direct_model_side_calls": 4,
        "reuse_conditioning_result": True,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "model_eval": True,
        "inference_only": True,
        "training": False,
        "backward": False,
        "adaptive_extension": False,
        "automatic_retry": False,
    }.items():
        _require(model_execution.get(key) == expected, f"profile model execution {key} drift")
    mask = _mapping(payload.get("mask"), field="profile.mask")
    _require(mask.get("shape") == [40, 132] and mask.get("active_scalar_count") == 190, "profile mask drift")
    _require(mask.get("legacy_first_ten_rejected") is True, "legacy first-ten mask is not rejected")
    validate_preflight_device_request(_mapping(payload.get("device_policy"), field="profile.device_policy"))
    telemetry = _mapping(payload.get("runtime_telemetry"), field="profile.runtime_telemetry")
    _require(telemetry.get("schema") == "CORELAB_STAGE45D_D1_4080S_PREFLIGHT_TELEMETRY_V1", "telemetry schema drift")
    _require(telemetry.get("materialized_now") is False and telemetry.get("all_observed_gpu_fields_now") is None, "telemetry was materialized early")
    evidence = _mapping(payload.get("evidence"), field="profile.evidence")
    _require(evidence.get("raw_before_derived") is True and evidence.get("physical_source") == "x1_hat_only", "evidence ordering/source drift")
    _require(payload.get("locked_boundaries") == admission.LOCKED_BOUNDARIES, "profile lock map drift")
    boundary = _mapping(payload.get("execution_boundary_now"), field="profile.execution_boundary_now")
    _require(all(value == 0 for value in boundary.values()), "preflight profile has nonzero current execution counter")
    return payload


def validate_preflight_device_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future single-device policy without touching CUDA."""

    _require(request.get("selected_device") == DEVICE, "preflight must select cuda:0")
    _require(request.get("selected_gpu_role") == GPU_ROLE, "preflight must select the 32GB primary GPU")
    _require(request.get("minimum_vram_bytes") == MIN_VRAM_BYTES, "preflight VRAM floor drift")
    for key in ("allow_16gb_secondary_fallback", "allow_multi_gpu", "allow_sharding", "allow_data_parallel", "oom_retry"):
        _require(request.get(key) is False, f"preflight device policy permits forbidden mode: {key}")
    _require(request.get("device_observation_required_before_model_import") is True, "GPU observation must precede model import")
    return {
        "selected_device": DEVICE,
        "selected_gpu_role": GPU_ROLE,
        "minimum_vram_bytes": MIN_VRAM_BYTES,
        "single_device": True,
        "fallback": False,
        "multi_gpu": False,
        "sharding": False,
        "data_parallel": False,
        "oom_retry": False,
    }


def validate_runtime_paths(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate materialized future paths before any model-side import."""

    paths = _mapping(binding.get("paths"), field="binding.paths")
    _require(set(paths) >= set(REQUIRED_PATHS), "runtime binding is missing a required path")
    for key in REQUIRED_PATHS:
        value = paths.get(key)
        _require(isinstance(value, str) and value, f"runtime path is missing: {key}")
        candidate = Path(value).expanduser()
        if key == "output_root":
            _require(not candidate.exists(), "preflight output root must not already exist")
        elif key in {"base_checkpoint", "full24_2000_checkpoint", "full24_dataset", "snapshot_root", "groot_source", "cosmos_root"}:
            regular_directory(candidate, key)
        else:
            regular_file(candidate, key)
    return {"paths_bound": sorted(REQUIRED_PATHS), "output_root": str(paths["output_root"])}


def validate_selected_window(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = _mapping(binding.get("selected_window"), field="binding.selected_window")
    _require(selected.get("index") == 0, "preflight selected window index drift")
    _require(selected.get("window_id") == WINDOW_ID, "preflight selected window ID drift")
    _require(selected.get("target_id") == TARGET_ID, "preflight selected target drift")
    _require(selected.get("split") == "TRAIN_ONLY", "preflight selected window split drift")
    _require(selected.get("target_xyz_in_model_input") is False, "target XYZ leaked into preflight model input")
    _require(isinstance(selected.get("normalized_label_path"), str) and selected.get("normalized_label_path"), "normalized label path is not bound")
    _require(isinstance(selected.get("normalized_label_sha256"), str) and len(selected["normalized_label_sha256"]) == 64, "normalized label identity is not bound")
    _require(isinstance(selected.get("expert_physical_action_path"), str) and selected.get("expert_physical_action_path"), "expert physical action path is not bound")
    _require(isinstance(selected.get("expert_physical_action_sha256"), str) and len(selected["expert_physical_action_sha256"]) == 64, "expert physical action identity is not bound")
    snapshot = _mapping(selected.get("snapshot"), field="binding.selected_window.snapshot")
    for key in ("state_v1", "source_timestamps", "observation_timestamp", "anchor_world_m", "target_world_m", "source_frame", "tcp_semantic_id"):
        _require(key in snapshot, f"selected snapshot field missing: {key}")
    return selected


def _load_array(path: Path, expected_sha256: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    regular_file(path, label)
    _require(sha256_file(path) == expected_sha256, f"{label} SHA256 mismatch")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AdapterError(f"cannot load {label}") from exc
    array = np.asarray(value)
    _require(array.shape == shape and np.isfinite(array).all(), f"{label} shape/finite contract failed")
    return array.astype(np.float32, copy=False)


def normalized_probe_metrics(v_hat: Any, v_target: Any, mask: Any) -> dict[str, Any]:
    """Compute descriptive normalized local-flow metrics only."""

    prediction = np.asarray(v_hat, dtype=np.float64)
    target = np.asarray(v_target, dtype=np.float64)
    active = np.asarray(mask, dtype=np.float64)
    _require(prediction.shape == RAW_SHAPE and target.shape == RAW_SHAPE, "normalized probe shape drift")
    _require(active.shape == RAW_SHAPE[1:] and np.count_nonzero(active) == 190, "normalized probe mask drift")
    delta = (prediction[0] - target[0]) * active

    def block(rows: slice, cols: slice) -> dict[str, float]:
        lhs = prediction[0, rows, cols]
        rhs = target[0, rows, cols]
        block_mask = active[rows, cols]
        lhs = lhs[block_mask != 0]
        rhs = rhs[block_mask != 0]
        rmse = float(np.sqrt(np.mean((lhs - rhs) ** 2)))
        denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        cosine = float(np.dot(lhs, rhs) / denominator) if denominator else 0.0
        return {"rmse": rmse, "cosine": cosine}

    blocks = {
        "geometry_xyz": block(slice(0, 15), slice(0, 3)),
        "geometry_rot6d": block(slice(0, 15), slice(3, 9)),
        "timing": block(slice(0, 15), slice(9, 10)),
        "gripper": block(slice(0, 40), slice(10, 11)),
    }
    lhs_active = prediction[0][active != 0]
    rhs_active = target[0][active != 0]
    denominator = float(np.linalg.norm(lhs_active) * np.linalg.norm(rhs_active))
    weighted_cosine = float(np.dot(lhs_active, rhs_active) / denominator) if denominator else 0.0
    return {
        "namespace": "NORMALIZED_LOCAL_FLOW",
        "masked_local_flow_rmse_scalar_weighted": float(np.sqrt(np.mean(delta[active != 0] ** 2))),
        "masked_local_flow_cosine_scalar_weighted": weighted_cosine,
        "blocks": blocks,
        "block_macro_rmse": float(np.mean([item["rmse"] for item in blocks.values()])),
        "block_macro_cosine": float(np.mean([item["cosine"] for item in blocks.values()])),
    }


def build_probe_arrays(noise: Any, x1: Any, t: float, v_hat: Any) -> dict[str, np.ndarray]:
    x0, x_t, v_target = harness.construct_flow(noise, x1, t)
    predicted = np.asarray(v_hat, dtype=np.float32)
    _require(predicted.shape == RAW_SHAPE and np.isfinite(predicted).all(), "predicted local velocity shape/finite drift")
    x1_hat = harness.implied_endpoint(x_t, predicted, t)
    return {
        "x0": x0,
        "x1": np.asarray(x1, dtype=np.float32),
        "x_t": x_t,
        "v_target": v_target,
        "v_hat": predicted,
        "x1_hat": x1_hat,
        "mask": np.asarray(structured_action_mask("T16"), dtype=np.float32),
    }


def write_probe_evidence(
    output_root: Path,
    *,
    probe_id: str,
    identity: Mapping[str, Any],
    arrays: Mapping[str, Any],
    physical_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Persist all raw arrays durably before either metric namespace."""

    probe_root = Path(output_root) / "probes" / probe_id
    writer = harness.ProbeEvidenceWriter(probe_root)
    raw = writer.write_raw(identity=identity, arrays=arrays)
    normalized = normalized_probe_metrics(arrays["v_hat"], arrays["v_target"], arrays["mask"])
    derived = writer.write_derived(normalized_metrics=normalized, physical_metrics=physical_metrics)
    return {"raw_manifest": raw, "derived": derived, "probe_root": str(probe_root)}


def validate_static_contract(profile_path: Path = PROFILE_PATH) -> dict[str, Any]:
    profile = validate_profile(profile_path)
    guard = static_guard()
    _require(guard["status"] == "PASS", "Corelab adapter static guard failed")
    harness.validate_reviewed_protocol()
    harness.validate_execution_manifest()
    harness.validate_runtime_binding_template()
    harness.validate_admission_manifest()
    harness.validate_compatibility_map()
    harness.validate_mask_contract()
    harness.validate_flow_algebra()
    harness.validate_stats_routing()
    harness.validate_call_budget()
    harness.validate_evidence_lifecycle()
    _assert_no_model_imports()
    return {
        "schema": ADAPTER_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "profile_sha256": sha256_file(profile_path),
        "adapter_sha256": sha256_file(Path(__file__)),
        "static_guard": guard,
        "preflight_profile": {
            "checkpoint": CHECKPOINT,
            "window_id": WINDOW_ID,
            "noise_seed": SEED,
            "t_buckets": list(T_BUCKETS),
            "conditioning_passes": 1,
            "action_head_calls": 3,
            "total_direct_model_side_calls": 4,
        },
        "device_policy": validate_preflight_device_request(profile["device_policy"]),
        "execution_boundary": dict(profile["execution_boundary_now"]),
        "return_token": RETURN_TOKEN,
    }


def _load_runtime_after_admission() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import the accepted model-side path only after admission has passed."""

    import torch
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
    from stage38_structured_offline_evaluation import _load_structured_model, _offline_step, _prepare_model_conditioning
    from stage45c_h200_eval_runtime_adapter import _decode_v2, _prepare_train24_windows

    return torch, _build_runtime_processor, _load_structured_model, _offline_step, _prepare_model_conditioning, _decode_v2, _prepare_train24_windows


def _runtime_failure_seal(output_root: Path | None, reason: str, counters: Mapping[str, Any]) -> None:
    if output_root is None:
        return
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json_durable(output_root / "FAILURE_SEAL.json", {
            "schema": "CORELAB_STAGE45D_D1_4080S_PREFLIGHT_FAILURE_SEAL_V1",
            "status": "FAIL",
            "reason": str(reason),
            "counters": dict(counters),
            "retry": False,
            "adaptive_extension": False,
        })
    except Exception:
        pass


def execute(runtime_document: Mapping[str, Any], *, profile: Mapping[str, Any], authority_token: str) -> dict[str, Any]:
    """Guarded future execution; PI0109 does not enter this function."""

    binding = _mapping(runtime_document.get("binding", runtime_document), field="runtime_document.binding")
    observed = _mapping(runtime_document.get("observed"), field="runtime_document.observed")
    counters: dict[str, Any] = {
        "model_construction": 0,
        "conditioning_passes": 0,
        "action_head_calls": 0,
        "model_forward": 0,
        "gpu": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "retry_count": 0,
    }
    output_root: Path | None = None
    try:
        # This is intentionally the first runtime operation that can admit a
        # future model.  It checks the pre-model import guard and all exact
        # identities before any Torch/GR00T import below.
        admission_result = admission.validate_future_admission(
            binding,
            observed,
            authority_token=authority_token,
            imported_modules=sys.modules,
        )
        validate_preflight_device_request(_mapping(binding.get("device_policy"), field="binding.device_policy"))
        validate_runtime_paths(binding)
        selected = validate_selected_window(binding)
        paths = _mapping(binding.get("paths"), field="binding.paths")
        output_root = Path(str(paths["output_root"])).expanduser().resolve()
        _require(not output_root.exists(), "preflight output root was reused")
        output_root.mkdir(parents=True, exist_ok=False)
        write_json_durable(output_root / "preflight_admission.json", admission_result)
        write_json_durable(output_root / "preflight_profile.json", profile)

        normalized_x1 = _load_array(Path(selected["normalized_label_path"]), str(selected["normalized_label_sha256"]), RAW_SHAPE, "normalized expert label")
        expert_physical = _load_array(Path(selected["expert_physical_action_path"]), str(selected["expert_physical_action_sha256"]), PHYSICAL_SHAPE, "expert physical action")

        # Only now may model/runtime modules enter the process.
        torch, build_runtime_processor, load_structured_model, offline_step, prepare_conditioning, decode_v2, prepare_windows = _load_runtime_after_admission()
        _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted Corelab preflight")
        device = torch.device(DEVICE)
        _require(device.index == 0, "Corelab preflight may use only cuda:0")
        properties = torch.cuda.get_device_properties(device)
        _require(int(properties.total_memory) >= MIN_VRAM_BYTES, "selected GPU does not meet the 32GB floor")
        counters["gpu"] = 1
        torch.cuda.reset_peak_memory_stats(device)
        telemetry = {
            "schema": "CORELAB_STAGE45D_D1_4080S_PREFLIGHT_TELEMETRY_V1",
            "host": observed["host_gpu_runtime"].get("host"),
            "gpu_name": str(properties.name),
            "gpu_uuid": observed["host_gpu_runtime"].get("gpu_uuid"),
            "selected_device": DEVICE,
            "total_vram_bytes": int(properties.total_memory),
            "cuda_runtime": str(torch.version.cuda),
        }
        runtime_processor, embodiment, modality_configs = build_runtime_processor(
            groot_source=Path(paths["groot_source"]),
            base_checkpoint=Path(paths["base_checkpoint"]),
            dataset_dir=Path(paths["full24_dataset"]),
            modality_path=Path(paths["modality_config"]),
            cosmos_snapshot=Path(paths["cosmos_root"]) / "snapshots" / COSMOS_REVISION,
        )
        snapshots = [dict(selected["snapshot"], window_id=WINDOW_ID)]
        windows = prepare_windows(runtime_processor, embodiment, modality_configs, Path(paths["full24_dataset"]), snapshots)
        _require(len(windows) == 1 and windows[0]["window_id"] == WINDOW_ID, "TRAIN24 selected window materialization drift")
        model_start = time.perf_counter()
        model = load_structured_model(Path(paths["full24_2000_checkpoint"]), DEVICE)
        model.eval()
        counters["model_construction"] = 1
        telemetry["model_load_latency_s"] = time.perf_counter() - model_start
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(SEED)
        noise = torch.randn(RAW_SHAPE, generator=noise_generator, device="cpu", dtype=torch.float32).contiguous()
        noise_np = noise.numpy().copy()
        conditioning_start = time.perf_counter()
        _, action_input, processed_backbone, state_features = prepare_conditioning(model, windows[0]["model_inputs"])
        counters["conditioning_passes"] = 1
        telemetry["conditioning_latency_s"] = time.perf_counter() - conditioning_start
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: counters.__setitem__("action_head_calls", counters["action_head_calls"] + 1))
        try:
            rows = []
            for t, bucket in zip(T_VALUES, T_BUCKETS):
                _, x_t_np, v_target_np = harness.construct_flow(noise_np, normalized_x1, t)
                x_t = torch.from_numpy(x_t_np).to(device=device, dtype=next(model.parameters()).dtype)
                started = time.perf_counter()
                with torch.inference_mode():
                    v_hat = offline_step(model.action_head, processed_backbone, action_input, state_features, x_t, bucket)
                latency = time.perf_counter() - started
                v_hat_np = v_hat.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                arrays = build_probe_arrays(noise_np, normalized_x1, t, v_hat_np)
                physical, decoder_details = decode_v2(
                    arrays["x1_hat"],
                    selected["snapshot"]["state_v1"],
                    Path(paths["full24_runtime_statistics"]),
                    selected["snapshot"]["source_timestamps"],
                    selected["snapshot"]["observation_timestamp"],
                    WINDOW_ID,
                )
                physical = np.asarray(physical, dtype=np.float64)
                _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "decoded physical action drift")
                probe_id = f"seed{SEED}_t{bucket:03d}"
                evidence = write_probe_evidence(
                    output_root,
                    probe_id=probe_id,
                    identity={
                        "checkpoint": CHECKPOINT,
                        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
                        "stats_sha256": STATS_SHA256,
                        "stats_payload_sha256": STATS_PAYLOAD_SHA256,
                        "window_id": WINDOW_ID,
                        "noise_seed": SEED,
                        "t": t,
                        "t_bucket": bucket,
                    },
                    arrays=arrays,
                    physical_metrics={"namespace": "DECODED_PHYSICAL_ACTION", "decoder": decoder_details},
                )
                rows.append({"probe": probe_id, "latency_s": latency, "evidence": evidence})
            _require(counters["conditioning_passes"] == 1, "conditioning pass count drift")
            _require(counters["action_head_calls"] == 3, "action-head call count drift")
            counters["model_forward"] = 3
            if torch.cuda.is_available():
                telemetry["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                telemetry["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            total_vram = int(properties.total_memory)
            required_free = max(2 * 1024**3, int(0.05 * total_vram))
            resource_margin = {
                "total_vram_bytes": total_vram,
                "peak_reserved_bytes": peak_reserved,
                "free_after_peak_bytes": total_vram - peak_reserved,
                "required_free_bytes": required_free,
                "formula": "free_bytes >= max(2GiB, 5%*total_vram_bytes)",
                "pass": bool(total_vram - peak_reserved >= required_free),
            }
            telemetry["resource_margin"] = resource_margin
            _require(resource_margin["pass"], "CORELAB_RESOURCE_BLOCKER: VRAM margin below PI0111 floor")
            telemetry["per_probe_latency_s"] = [row["latency_s"] for row in rows]
            seal = {
                "schema": "CORELAB_STAGE45D_D1_4080S_PREFLIGHT_SEAL_V1",
                "status": "PASS",
                "token": SMOKE_PASS_TOKEN,
                "profile_sha256": sha256_file(PROFILE_PATH),
                "execution_head": binding["execution_checkout"]["authorized_head"],
                "resource_margin": resource_margin,
                "checkpoint": CHECKPOINT,
                "window_id": WINDOW_ID,
                "counters": counters,
                "telemetry": telemetry,
                "rows": rows,
                "execution_boundary": {
                    "training": False,
                    "backward": False,
                    "public_sampler_calls": 0,
                    "get_action_calls": 0,
                    "Validation-E": 0,
                    "Holdout-F": 0,
                    "WBC": 0,
                    "simulator": 0,
                    "hardware": 0,
                },
            }
            write_json_durable(output_root / "preflight_telemetry.json", telemetry)
            write_json_durable(output_root / "preflight_seal.json", seal)
            return seal
        finally:
            hook.remove()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        _runtime_failure_seal(output_root, str(exc), counters)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--runtime-binding", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan_only or args.audit_only:
            payload = validate_static_contract(args.profile)
            if args.audit_only:
                payload["mode"] = "audit-only"
            else:
                payload["mode"] = "plan-only"
        else:
            _require(args.runtime_binding is not None, "--execute requires --runtime-binding")
            profile = validate_profile(args.profile)
            _require(os.environ.get("STAGE45D_D1_CORELAB_PREFLIGHT_RUNTIME") == "authorized", "PI0109 preflight runtime is not authorized")
            authority_token = os.environ.get("STAGE45D_D1_CORELAB_PREFLIGHT_AUTHORITY_TOKEN", "")
            document = read_json(Path(args.runtime_binding))
            payload = execute(document, profile=profile, authority_token=authority_token)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45D_D1_CORELAB_RUNTIME_ADAPTER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
