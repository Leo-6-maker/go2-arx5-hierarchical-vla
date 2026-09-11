#!/usr/bin/env python3
"""Vanda H200 Stage45D-D1 full teacher-forced/local-flow runtime adapter.

The smoke adapter and this full adapter deliberately share the same frozen
runtime primitives: admission ordering, direct Stage38 action-head graph,
Stage45D flow algebra, Stage40/Stage45C decoding, and the durable raw-first
evidence writer.  ``--plan-only`` and ``--audit-only`` are CPU/static.  The
guarded ``--execute`` path is late-bound and requires a separately materialized
PI-authorized runtime binding after a successful smoke on the same HEAD.

The full execution is exactly three checkpoints x 24 TRAIN windows x three
fixed noise seeds x three fixed flow times.  It does not call a public
sampler/get_action path, train, retry, or alter precision/normalization.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45d_d1_corelab_runtime_adapter as common  # noqa: E402
import stage45d_d1_runtime_admission as admission  # noqa: E402


FULL_PROFILE_PATH = REPO / "manifests" / "STAGE45D_D1_CORELAB_FULL_EXECUTION_PROFILE_V1.json"
FULL_PROFILE_SCHEMA = "CORELAB_STAGE45D_D1_CORELAB_FULL_EXECUTION_PROFILE_V1"
FULL_ADAPTER_SCHEMA = "CORELAB_STAGE45D_D1_CORELAB_FULL_RUNTIME_ADAPTER_V1"
FULL_RETURN_TOKEN = "STAGE45D_D1_VANDA_H200_FULL_EXECUTION_COMPLETE_FOR_PI_REVIEW"
SMOKE_PASS_TOKEN = "STAGE45D_D1_H200_SMOKE_PASS_AUTO_CONTINUE_FULL"
PI_GATE_COMMENT = 5629752688
PR51_POINTER_COMMENT = 5629755293
SUPPORT_HEAD = "404d345ed81972fb84c5f25aa7e48245f4ec7c8c"
DEVICE = common.DEVICE
GPU_ROLE = common.GPU_ROLE
MIN_VRAM_BYTES = common.MIN_VRAM_BYTES
CHECKPOINT_ORDER = ("STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
WINDOW_COUNT = 24
SEEDS = (2223, 2224, 2225)
T_VALUES = common.T_VALUES
T_BUCKETS = common.T_BUCKETS
RAW_SHAPE = common.RAW_SHAPE
PHYSICAL_SHAPE = common.PHYSICAL_SHAPE
WINDOW_ORDER = "target-major then snapshot-major"
REQUIRED_PATHS = (
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
    "modality_config",
    "base_processor_config",
    "base_processor_statistics",
    "structured_1000_statistics",
    "full24_runtime_statistics",
    "golden_release_manifest",
    "groot_runtime_manifest",
    "full24_manifest",
    "checkpoint_integrity_audit",
    "runtime_admission",
    "output_root",
)
DIRECTORY_PATHS = {
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
}
TARGET_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in (
        "red_target_train_a",
        "red_target_train_b",
        "red_target_train_c",
        "red_target_train_d",
    )
    for index in range(6)
)
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
FORBIDDEN_CALLS = {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}


class FullAdapterError(RuntimeError):
    """Raised when the full D1 runtime cannot prove its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullAdapterError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = common.read_json(path)
    except Exception as exc:
        raise FullAdapterError(str(exc)) from exc
    return value


def _current_checkout() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        _require(completed.returncode == 0, f"git checkout identity command failed: {' '.join(args)}")
        return completed.stdout.strip()

    return {
        "root": str(REPO),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "clean": run("status", "--porcelain=v1") == "",
    }


def _static_guard() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    forbidden_top = sorted(top_level.intersection(FORBIDDEN_IMPORTS))
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_calls = sorted(set(calls).intersection(FORBIDDEN_CALLS))
    source = Path(__file__).read_text(encoding="utf-8")
    shared_primitives = {
        "common_adapter_imported": "stage45d_d1_corelab_runtime_adapter" in source,
        "shared_flow_builder": "common.build_probe_arrays" in source,
        "shared_evidence_writer": "common.write_probe_evidence" in source,
        "shared_normalized_metrics": "common.normalized_probe_metrics" in source,
        "shared_runtime_loader": "common._load_runtime_after_admission" in source,
        "direct_action_head_graph": "offline_step" in source,
    }
    return {
        "top_level_forbidden_imports": forbidden_top,
        "forbidden_call_hits": forbidden_calls,
        "shared_primitives": shared_primitives,
        "late_import_function": "_load_runtime_after_admission",
        "status": "PASS" if not forbidden_top and not forbidden_calls and all(shared_primitives.values()) else "FAIL",
    }


def _validate_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == FULL_PROFILE_SCHEMA, "full profile schema drift")
    _require(payload.get("status") == "READY_FOR_PI_REVIEW_NOT_EXECUTED", "full profile must remain static-only")
    _require(payload.get("return_token") == FULL_RETURN_TOKEN, "full return token drift")
    authority = _mapping(payload.get("authority"), field="full.authority")
    _require(authority.get("issue_11_comment") == PI_GATE_COMMENT, "PI0117 authority drift")
    _require(authority.get("pr_51_comment") == PR51_POINTER_COMMENT, "PR #51 pointer drift")
    _require(authority.get("prepared_from_support_head") == SUPPORT_HEAD, "full profile support HEAD drift")
    runtime = _mapping(payload.get("runtime_adapter"), field="full.runtime_adapter")
    _require(runtime.get("path") == "scripts/stage45d_d1_corelab_full_runtime_adapter.py", "full adapter path drift")
    _require(runtime.get("sha256") == common.sha256_file(Path(__file__)), "full adapter SHA drift")
    _require(runtime.get("common_adapter_path") == "scripts/stage45d_d1_corelab_runtime_adapter.py", "common adapter path drift")
    _require(runtime.get("common_adapter_sha256") == common.sha256_file(Path(common.__file__)), "common adapter SHA drift")
    protocol = _mapping(payload.get("protocol"), field="full.protocol")
    _require(protocol.get("protocol_sha256") == common.harness.PROTOCOL_SHA256, "full protocol SHA drift")
    _require(protocol.get("runtime_binding_schema") == "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V2", "full binding schema drift")
    _require(protocol.get("runtime_binding_template_path") == "manifests/STAGE45D_D1_RUNTIME_BINDING_TEMPLATE_V2.json", "full binding template path drift")
    _require(protocol.get("scientific_semantics_unchanged") is True, "full profile changes scientific semantics")
    execution = _mapping(payload.get("execution"), field="full.execution")
    expected_execution = {
        "checkpoint_order": list(CHECKPOINT_ORDER),
        "window_count": WINDOW_COUNT,
        "window_order": WINDOW_ORDER,
        "window_ids": list(TARGET_IDS),
        "noise_seeds": list(SEEDS),
        "t_values": list(T_VALUES),
        "t_buckets": list(T_BUCKETS),
        "conditioning_passes": 72,
        "action_head_calls": 648,
        "total_model_side_calls": 720,
        "one_conditioning_per_checkpoint_window": True,
        "reuse_conditioning_across_seed_t": True,
        "same_noise_bytes_across_checkpoints": True,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "free_running_sampler_calls": 0,
        "model_eval": True,
        "inference_only": True,
        "target_xyz_in_model_input": False,
        "adaptive_extension": False,
        "automatic_retry": False,
    }
    for key, expected in expected_execution.items():
        _require(execution.get(key) == expected, f"full execution {key} drift")
    mask = _mapping(payload.get("mask"), field="full.mask")
    _require(mask.get("shape") == [40, 132] and mask.get("active_scalar_count") == 190, "full mask drift")
    _require(mask.get("legacy_first_ten_rejected") is True, "legacy first-ten mask is not rejected")
    flow = _mapping(payload.get("flow"), field="full.flow")
    for key, expected in {
        "x_t": "x_t=(1-t)x0+t*x1",
        "v_target": "v_target=x1-x0",
        "x1_hat": "x1_hat=x_t+(1-t)*v_hat",
        "physical_source": "x1_hat_only",
        "noise_dtype": "float32",
        "noise_device": "CPU",
    }.items():
        _require(flow.get(key) == expected, f"full flow {key} drift")
    routes = _mapping(payload.get("checkpoint_stats_routes"), field="full.checkpoint_stats_routes")
    for label, route in {
        "STRUCTURED@1000": "HISTORICAL_STAGE40_FIXED_STATS",
        "FULL24@1000": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "FULL24@2000": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
    }.items():
        _require(_mapping(routes.get(label), field=f"route.{label}").get("route") == route, f"stats route drift: {label}")
    device = _mapping(payload.get("device_policy"), field="full.device_policy")
    common.validate_preflight_device_request(device)
    margin = _mapping(payload.get("resource_margin"), field="full.resource_margin")
    _require(margin.get("formula") == "free_bytes >= max(2GiB, 5%*total_vram_bytes)", "resource margin formula drift")
    _require(margin.get("minimum_free_bytes") == max(2 * 1024**3, int(0.05 * common.MIN_VRAM_BYTES)), "resource margin floor drift")
    repairs = _mapping(payload.get("engineering_repair_budget"), field="full.engineering_repair_budget")
    _require(repairs.get("max_repair_cycles") == 2 and repairs.get("max_smoke_attempts") == 6, "repair budget drift")
    evidence = _mapping(payload.get("evidence"), field="full.evidence")
    _require(evidence.get("raw_before_derived") is True and evidence.get("failure_seal_required") is True and evidence.get("fresh_output_root_required") is True, "full evidence lifecycle drift")
    _require(payload.get("locked_boundaries") == admission.LOCKED_BOUNDARIES, "full locked boundary drift")
    boundary = _mapping(payload.get("execution_boundary_now"), field="full.execution_boundary_now")
    _require(all(value == 0 for value in boundary.values()), "full profile current boundary is nonzero")
    return dict(payload)


def validate_profile(path: Path = FULL_PROFILE_PATH) -> dict[str, Any]:
    return _validate_profile(_read_json(path))


def _validate_binding_template_v2(path: Path | None = None) -> dict[str, Any]:
    template_path = path or (REPO / "manifests" / "STAGE45D_D1_RUNTIME_BINDING_TEMPLATE_V2.json")
    payload = _read_json(template_path)
    _require(payload.get("schema") == "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V2", "V2 binding template schema drift")
    _require(payload.get("status") == "LATE_BOUND_NOT_EXECUTED", "V2 binding template must remain late-bound")
    _require(payload.get("execution_authorized") is False and payload.get("pi_authority_token") is None, "V2 binding template is executable")
    checkout = _mapping(payload.get("execution_checkout"), field="v2_template.execution_checkout")
    _require(checkout.get("binding_mode") == "OBSERVED_RUNTIME_ONLY", "V2 binding checkout mode drift")
    _require(checkout.get("protocol_provenance_head_may_substitute") is False, "protocol provenance may substitute for execution HEAD")
    for key in ("authorized_repo_root", "authorized_branch", "authorized_head", "authorized_clean_worktree"):
        _require(checkout.get(key) is None, f"V2 template contains checkout value: {key}")
    paths = _mapping(payload.get("paths"), field="v2_template.paths")
    _require(set(paths) == set(REQUIRED_PATHS), "V2 template path key set drift")
    _require(all(value is None for value in paths.values()), "V2 template contains a node-local path")
    _require(payload.get("target_xyz_in_model_input") is False, "V2 template enables target XYZ")
    _require(payload.get("historical_comparator", {}).get("runtime_materialization_status") == "PENDING_VANDA_RECOVERY", "historical comparator materialization state drift")
    _require(payload.get("historical_comparator", {}).get("materialization_path") is None, "historical comparator scratch path was invented")
    return payload


def _validate_paths(binding: Mapping[str, Any]) -> dict[str, Any]:
    paths = _mapping(binding.get("paths"), field="full.binding.paths")
    _require(set(paths) == set(REQUIRED_PATHS), "full runtime path key set drift")
    for key in REQUIRED_PATHS:
        value = paths.get(key)
        _require(isinstance(value, str) and value, f"full runtime path missing: {key}")
        candidate = Path(value).expanduser()
        if key == "output_root":
            _require(not candidate.exists(), "full output root must be fresh")
        elif key in DIRECTORY_PATHS:
            common.regular_directory(candidate, key)
        else:
            common.regular_file(candidate, key)
        _require("validation-e" not in str(candidate).lower() and "holdout-f" not in str(candidate).lower(), f"locked data path: {key}")
    return {key: Path(str(paths[key])).expanduser() for key in REQUIRED_PATHS}


def _validate_execution_control(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    control = _mapping(binding.get("execution_control"), field="full.execution_control")
    _require(control.get("phase") == "FULL", "full execution phase drift")
    repair_cycle = control.get("repair_cycle")
    attempt = control.get("smoke_attempt")
    _require(type(repair_cycle) is int and 0 <= repair_cycle <= 2, "repair cycle exceeds PI0117 budget")
    _require(type(attempt) is int and 1 <= attempt <= 6, "smoke attempt exceeds PI0117 budget")
    _require(control.get("automatic_retry") is False, "automatic retry is forbidden")
    smoke_head = control.get("smoke_pass_head")
    _require(isinstance(smoke_head, str) and len(smoke_head) == 40, "successful smoke HEAD is not bound")
    _require(smoke_head == binding.get("execution_checkout", {}).get("authorized_head"), "full HEAD differs from smoke PASS HEAD")
    smoke_seal = control.get("smoke_seal_path")
    smoke_sha = control.get("smoke_seal_sha256")
    _require(isinstance(smoke_seal, str) and smoke_seal, "smoke seal path is not bound")
    _require(isinstance(smoke_sha, str) and len(smoke_sha) == 64, "smoke seal SHA is not bound")
    smoke_path = Path(smoke_seal).expanduser()
    common.regular_file(smoke_path, "smoke seal")
    _require(common.sha256_file(smoke_path) == smoke_sha, "smoke seal SHA mismatch")
    smoke = _read_json(smoke_path)
    _require(smoke.get("status") == "PASS" and smoke.get("token") == SMOKE_PASS_TOKEN, "smoke seal is not a PASS for auto-continuation")
    _require(smoke.get("execution_head") == smoke_head, "smoke seal HEAD mismatch")
    return control


def validate_static_contract(profile_path: Path = FULL_PROFILE_PATH) -> dict[str, Any]:
    profile = validate_profile(profile_path)
    _validate_binding_template_v2()
    guard = _static_guard()
    _require(guard["status"] == "PASS", "full adapter static guard failed")
    common_result = common.validate_static_contract()
    _require(common_result["static_guard"]["status"] == "PASS", "shared smoke runtime static guard failed")
    return {
        "schema": FULL_ADAPTER_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "profile_sha256": common.sha256_file(profile_path),
        "adapter_sha256": common.sha256_file(Path(__file__)),
        "common_adapter_sha256": common.sha256_file(Path(common.__file__)),
        "static_guard": guard,
        "common_runtime": {
            "adapter_path": "scripts/stage45d_d1_corelab_runtime_adapter.py",
            "smoke_is_exact_subset": True,
            "shared_flow": True,
            "shared_evidence_writer": True,
            "shared_decode_bridge": True,
        },
        "execution": {
            "checkpoint_order": list(CHECKPOINT_ORDER),
            "window_count": WINDOW_COUNT,
            "noise_seeds": list(SEEDS),
            "t_buckets": list(T_BUCKETS),
            "conditioning_passes": 72,
            "action_head_calls": 648,
            "total_model_side_calls": 720,
        },
        "execution_boundary": dict(profile["execution_boundary_now"]),
        "return_token": FULL_RETURN_TOKEN,
    }


def _load_runtime_after_admission() -> tuple[Any, ...]:
    """Late-import all model-side modules after admission and artifact checks."""

    import stage26_h1_c1_fixed_noise_local_flow as stage26
    import stage38_structured_offline_evaluation as stage38
    import stage45c_h200_eval_runtime_adapter as stage45c
    import torch

    shared = common._load_runtime_after_admission()
    return (torch, stage26, stage38, stage45c, shared)


def _verify_runtime_artifacts(paths: Mapping[str, Path], stage45c: Any) -> dict[str, Any]:
    """Rehash all late-bound artifacts before constructing the model."""

    result: dict[str, Any] = {}
    result["golden_release_manifest"] = stage45c._verify_release_manifest(paths["golden_release_manifest"])
    result["groot_runtime_manifest"] = stage45c._verify_groot_runtime_manifest(paths["groot_runtime_manifest"])
    result["full24_manifest"] = stage45c._verify_full24_manifest(paths["full24_manifest"])
    result["checkpoint_integrity_audit"] = stage45c._hash_identity(paths["checkpoint_integrity_audit"], stage45c.INTEGRITY_AUDIT_SHA256, "checkpoint integrity/delta audit")
    result["runtime_admission"] = stage45c._hash_identity(paths["runtime_admission"], stage45c.RUNTIME_ADMISSION_SHA256, "runtime admission")
    result["modality_config"] = stage45c._hash_identity(paths["modality_config"], stage45c.MODALITY_SHA256, "modality config")
    result["base_processor_config"] = stage45c._hash_identity(paths["base_processor_config"], stage45c.BASE_PROCESSOR_CONFIG_SHA256, "BASE processor config")
    result["base_processor_statistics"] = stage45c._hash_identity(paths["base_processor_statistics"], stage45c.BASE_PROCESSOR_STATS_SHA256, "BASE processor statistics")
    result["structured_1000_statistics"] = stage45c._verify_stats(paths["structured_1000_statistics"], stage45c.HISTORICAL_STATS_SHA256, None, "historical StructuredAction statistics")
    result["full24_runtime_statistics"] = stage45c._verify_runtime_stats(paths["full24_runtime_statistics"])
    result["dataset"] = stage45c._verify_dataset(paths["full24_dataset"])
    result["checkpoints"] = stage45c._verify_checkpoints({key: str(value) for key, value in paths.items()})
    cosmos_digest, cosmos_entries = stage45c.cosmos_tree_digest(paths["cosmos_root"])
    _require(cosmos_digest == stage45c.COSMOS_TREE_SHA256, "Cosmos cache tree identity mismatch")
    cosmos_snapshot = paths["cosmos_root"] / "snapshots" / stage45c.COSMOS_REVISION
    common.regular_directory(cosmos_snapshot, "Cosmos pinned snapshot")
    refs_main = paths["cosmos_root"] / "refs" / "main"
    common.regular_file(refs_main, "Cosmos refs/main")
    _require(refs_main.read_text(encoding="utf-8").strip() == stage45c.COSMOS_REVISION, "Cosmos revision mismatch")
    result["cosmos"] = {"tree_sha256": cosmos_digest, "entry_count": cosmos_entries, "revision": stage45c.COSMOS_REVISION}
    result["groot_source"] = stage45c._verify_groot_source_snapshot(
        paths["groot_source"], paths["groot_runtime_manifest"]
    )
    closure = stage45c._verify_train24_descriptor_closure(paths["snapshot_root"], paths["full24_dataset"])
    result["snapshot_closure"] = {key: value for key, value in closure.items() if key != "snapshots"}
    result["snapshots"] = closure["snapshots"]
    _require(len(result["snapshots"]) == WINDOW_COUNT, "TRAIN24 snapshot count drift")
    return result


def _make_noise_bank(torch: Any) -> dict[int, list[Any]]:
    bank: dict[int, list[Any]] = {}
    for seed in SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        bank[seed] = [
            torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
            for _ in range(WINDOW_COUNT)
        ]
    return bank


def _load_stats_sidecar(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    stats = payload.get("statistics")
    _require(isinstance(stats, Mapping), f"statistics sidecar missing statistics mapping: {path}")
    return payload


def _normalized_labels(snapshot_records: Sequence[Mapping[str, Any]], paths: Mapping[str, Path], stage26: Any, stage38: Any, label: str) -> list[np.ndarray]:
    stats_path = paths["structured_1000_statistics"] if label == "STRUCTURED@1000" else paths["full24_runtime_statistics"]
    sidecar = _load_stats_sidecar(stats_path)
    labels: list[np.ndarray] = []
    for record in snapshot_records:
        snapshot = stage26._load_snapshot(Path(record["path"]))
        expert_physical = np.asarray(snapshot.physical_action, dtype=np.float64)[None, ...]
        _, _, normalized = stage38._structured_label_and_normalizer(expert_physical, snapshot, sidecar)
        value = np.asarray(normalized, dtype=np.float32)[None, ...]
        _require(value.shape == RAW_SHAPE and np.isfinite(value).all(), f"normalized label drift: {record['window_id']}")
        labels.append(value)
    _require(len(labels) == WINDOW_COUNT, "normalized label count drift")
    return labels


def _decode_prediction(label: str, x1_hat: np.ndarray, snapshot: Any, paths: Mapping[str, Path], stage45d_harness: Any, stage45c: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if label == "STRUCTURED@1000":
        from go2arx5_vla.stage40_structured_decode import decode_raw_structured_action

        _, physical, details = decode_raw_structured_action(
            x1_hat,
            snapshot.state_v1.as_array(),
            stats_path=paths["structured_1000_statistics"],
            source_timestamps=snapshot.future_timestamps,
            observation_timestamp=float(snapshot.observation_timestamp),
            chunk_id=f"{label}:{snapshot.example_id}",
        )
        return np.asarray(physical, dtype=np.float64), {"stats_route": "HISTORICAL_STAGE40_FIXED_STATS", **details}
    physical, details = stage45c._decode_v2(
        x1_hat,
        snapshot.state_v1.as_array(),
        paths["full24_runtime_statistics"],
        snapshot.future_timestamps,
        float(snapshot.observation_timestamp),
        snapshot.example_id,
    )
    return np.asarray(physical, dtype=np.float64), {"stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE", **details}


def _resource_margin(torch: Any, device: Any, total_vram_bytes: int) -> dict[str, Any]:
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_after_peak = int(total_vram_bytes - peak_reserved)
    required_free = max(2 * 1024**3, int(0.05 * total_vram_bytes))
    return {
        "total_vram_bytes": int(total_vram_bytes),
        "peak_reserved_bytes": peak_reserved,
        "free_after_peak_bytes": free_after_peak,
        "required_free_bytes": required_free,
        "formula": "free_bytes >= max(2GiB, 5%*total_vram_bytes)",
        "pass": bool(free_after_peak >= required_free),
    }


def _validate_vram_binding(torch_total_vram_bytes: int, observed_gpu: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the child CUDA device to the parent observation without mixing APIs.

    ``nvidia-smi memory.total`` and ``torch.cuda``'s ``total_memory`` are
    independent driver-facing capacity reports and are not guaranteed to be
    byte-identical on Hopper.  The parent observation still must identify the
    same H200 by name and both measurements must independently clear the
    locked 80-GiB floor.  Record both values and their delta for provenance;
    never convert that measurement difference into extra usable capacity.
    """

    _require(type(torch_total_vram_bytes) is int, "Torch VRAM measurement must be a JSON integer")
    observed_total = observed_gpu.get("total_vram_bytes")
    _require(type(observed_total) is int, "observed nvidia-smi VRAM measurement is missing or non-integer")
    observed_name = str(observed_gpu.get("gpu_name", observed_gpu.get("gpu", "")))
    _require("H200" in observed_name.upper(), "observed GPU is not an H200")
    _require(torch_total_vram_bytes >= MIN_VRAM_BYTES, "selected H200 is below the 80GiB floor")
    _require(observed_total >= MIN_VRAM_BYTES, "observed H200 is below the 80GiB floor")
    return {
        "torch_total_vram_bytes": torch_total_vram_bytes,
        "nvidia_smi_total_vram_bytes": observed_total,
        "nvidia_smi_minus_torch_bytes": observed_total - torch_total_vram_bytes,
        "measurement_contract": "same observed H200 identity plus independent API capacity floors",
        "exact_cross_api_byte_equality_required": False,
    }


def _failure_seal(output_root: Path | None, reason: str, counters: Mapping[str, Any], *, token: str) -> None:
    if output_root is None:
        return
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        common.write_json_durable(output_root / "FAILURE_SEAL.json", {
            "schema": "CORELAB_STAGE45D_D1_FULL_FAILURE_SEAL_V1",
            "status": "FAIL",
            "token": token,
            "reason": str(reason),
            "counters": dict(counters),
            "retry": False,
            "adaptive_extension": False,
        })
    except Exception:
        pass


def _write_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize sealed raw rows without introducing scientific thresholds."""

    _require(rows, "cannot aggregate an empty full-D1 row set")

    def scalar_summary(values: Sequence[float]) -> dict[str, float | int]:
        finite = [float(value) for value in values]
        _require(finite and all(np.isfinite(finite)), "aggregate contains non-finite values")
        ordered = sorted(finite)
        return {
            "count": len(ordered),
            "mean": float(mean(ordered)),
            "median": float(median(ordered)),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
        }

    def group(key: str) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result: dict[str, Any] = {}
        for group_key in sorted(groups):
            metrics = [row["trajectory_metrics"]["target"] for row in groups[group_key]]
            result[group_key] = {
                "count": len(metrics),
                "initial_target_distance_m": scalar_summary([metric["initial_target_distance_m"] for metric in metrics]),
                "minimum_target_distance_m": scalar_summary([metric["minimum_target_distance_m"] for metric in metrics]),
                "final_target_distance_m": scalar_summary([metric["final_target_distance_m"] for metric in metrics]),
                "absolute_progress_m": scalar_summary([metric["absolute_progress_m"] for metric in metrics]),
                "best_progress_m": scalar_summary([metric["best_progress_m"] for metric in metrics]),
                "progress_sign_counts": {
                    sign: sum(metric.get("progress_sign") == sign for metric in metrics)
                    for sign in ("POSITIVE", "ZERO", "NEGATIVE")
                },
            }
        return result

    paired: dict[str, Any] = {}
    by_key: dict[tuple[str, str, int, int], Mapping[str, Any]] = {
        (str(row["checkpoint"]), str(row["window_id"]), int(row["seed"]), int(row["t_bucket"])): row
        for row in rows
    }
    pair_specs = (
        ("STRUCTURED@1000", "FULL24@1000", "coverage_comparison"),
        ("FULL24@1000", "FULL24@2000", "exposure_comparison"),
    )
    for left, right, name in pair_specs:
        deltas: list[float] = []
        matched = 0
        for key, left_row in by_key.items():
            if key[0] != left:
                continue
            right_row = by_key.get((right, key[1], key[2], key[3]))
            if right_row is None:
                continue
            matched += 1
            deltas.append(
                float(right_row["trajectory_metrics"]["target"]["absolute_progress_m"])
                - float(left_row["trajectory_metrics"]["target"]["absolute_progress_m"])
            )
        _require(matched == 216, f"paired row count drift for {name}: {matched}")
        paired[name] = {"matched_rows": matched, "progress_delta_right_minus_left_m": scalar_summary(deltas)}

    return {
        "schema": "CORELAB_STAGE45D_D1_FULL_CPU_SUMMARY_V1",
        "status": "DESCRIPTIVE_ONLY_NO_THRESHOLD",
        "row_count": len(rows),
        "by_checkpoint": group("checkpoint"),
        "by_target": group("target_id"),
        "by_window": group("window_id"),
        "by_seed": group("seed"),
        "by_t_bucket": group("t_bucket"),
        "paired": paired,
        "failure_taxonomy": {"finite": len(rows), "nonfinite": 0, "decode_failure": 0, "admission_failure": 0},
        "scientific_interpretation": "PI_REVIEW_REQUIRED",
    }


def execute(runtime_document: Mapping[str, Any], *, profile: Mapping[str, Any], authority_token: str) -> dict[str, Any]:
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
        "optimizer_steps": 0,
        "checkpoint_write": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "free_running_sampler_calls": 0,
        "retry_count": 0,
    }
    output_root: Path | None = None
    try:
        admission_result = admission.validate_future_admission(
            binding,
            observed,
            authority_token=authority_token,
            imported_modules=sys.modules,
        )
        common.validate_preflight_device_request(_mapping(binding.get("device_policy"), field="binding.device_policy"))
        paths = _validate_paths(binding)
        _validate_execution_control(binding)
        full_exec = _mapping(binding.get("full_execution"), field="binding.full_execution")
        _require(full_exec.get("checkpoint_order") == list(CHECKPOINT_ORDER), "full checkpoint order drift")
        _require(full_exec.get("window_count") == WINDOW_COUNT and full_exec.get("window_order") == WINDOW_ORDER, "full window contract drift")
        _require(full_exec.get("noise_seeds") == list(SEEDS) and full_exec.get("t_values") == list(T_VALUES) and full_exec.get("t_buckets") == list(T_BUCKETS), "full sampler grid drift")
        _require(full_exec.get("target_xyz_in_model_input") is False, "target XYZ inclusion was authorized")
        output_root = paths["output_root"].resolve()
        _require(not output_root.exists(), "full output root was reused")
        output_root.mkdir(parents=True, exist_ok=False)
        common.write_json_durable(output_root / "admission.json", admission_result)
        common.write_json_durable(output_root / "profile.json", profile)

        # Artifact and descriptor verification is deliberately performed with
        # the CPU-only Stage45C module before importing Torch/GR00T/model code.
        import stage45c_h200_eval_runtime_adapter as stage45c

        artifacts = _verify_runtime_artifacts(paths, stage45c)
        _require(not admission.model_imports_present(), "model-side module imported during pre-model artifact verification")

        # These imports remain after admission and all artifact checks;
        # model construction starts only below this point.
        torch, stage26, stage38, stage45c, shared_runtime = _load_runtime_after_admission()
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
        windows = stage45c._prepare_train24_windows(
            runtime_processor,
            embodiment,
            modality_configs,
            paths["full24_dataset"],
            snapshot_records,
        )
        _require(len(windows) == WINDOW_COUNT and [item["window_id"] for item in windows] == list(TARGET_IDS), "TRAIN24 window order/materialization drift")
        labels_by_checkpoint = {
            label: _normalized_labels(snapshot_records, paths, stage26, stage38, label)
            for label in CHECKPOINT_ORDER
        }
        noise_bank = _make_noise_bank(torch)
        noise_hashes = {
            str(seed): [common.sha256_bytes(item.detach().cpu().contiguous().numpy().tobytes()) for item in values]
            for seed, values in noise_bank.items()
        }
        common.write_json_durable(output_root / "noise_manifest.json", {
            "schema": "CORELAB_STAGE45D_D1_FLOAT32_CPU_NOISE_BANK_V1",
            "seeds": list(SEEDS),
            "window_ids": list(TARGET_IDS),
            "shape": list(RAW_SHAPE),
            "dtype": "float32",
            "device": "cpu",
            "same_noise_bytes_across_checkpoints": True,
            "noise_sha256_by_seed_window": noise_hashes,
        })
        device = torch.device(DEVICE)
        _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted H200 runtime")
        _require(device.index == 0, "full D1 may use only cuda:0")
        properties = torch.cuda.get_device_properties(device)
        total_vram = int(properties.total_memory)
        _require("H200" in str(properties.name).upper(), "selected GPU is not an H200")
        _require(total_vram >= MIN_VRAM_BYTES, "selected H200 is below the 80GiB floor")
        observed_gpu = _mapping(observed.get("host_gpu_runtime"), field="observed.host_gpu_runtime")
        _require(str(properties.name) == str(observed_gpu.get("gpu_name", observed_gpu.get("gpu", properties.name))), "observed GPU name differs from runtime")
        vram_binding = _validate_vram_binding(total_vram, observed_gpu)
        counters["gpu"] = 1
        torch.cuda.reset_peak_memory_stats(device)
        telemetry: dict[str, Any] = {
            "schema": "VANDA_STAGE45D_D1_H200_FULL_TELEMETRY_V1",
            "host": observed_gpu.get("host"),
            "gpu_name": str(properties.name),
            "gpu_uuid": observed_gpu.get("gpu_uuid"),
            "selected_device": DEVICE,
            "total_vram_bytes": total_vram,
            "vram_binding": vram_binding,
            "cuda_runtime": str(torch.version.cuda),
            "model_load_latency_s_by_checkpoint": {},
            "conditioning_latency_s_by_checkpoint_window": {},
            "per_probe_latency_s": [],
        }
        checkpoint_results: list[dict[str, Any]] = []
        raw_jsonl = output_root / "raw_probe_rows.jsonl"
        for label in CHECKPOINT_ORDER:
            checkpoint_path = paths[{"STRUCTURED@1000": "structured_1000_checkpoint", "FULL24@1000": "full24_1000_checkpoint", "FULL24@2000": "full24_2000_checkpoint"}[label]]
            model_start = time.perf_counter()
            model = stage38._load_structured_model(checkpoint_path, DEVICE)
            model.eval()
            counters["model_construction"] += 1
            telemetry["model_load_latency_s_by_checkpoint"][label] = time.perf_counter() - model_start
            action_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: action_counter.__setitem__("count", action_counter["count"] + 1))
            rows: list[dict[str, Any]] = []
            try:
                for index, (window, snapshot, snapshot_record) in enumerate(zip(windows, snapshots, snapshot_records)):
                    conditioning_start = time.perf_counter()
                    _, action_input, processed_backbone, state_features = stage38._prepare_model_conditioning(model, window["model_inputs"])
                    counters["conditioning_passes"] += 1
                    telemetry["conditioning_latency_s_by_checkpoint_window"][f"{label}:{window['window_id']}"] = time.perf_counter() - conditioning_start
                    for seed in SEEDS:
                        noise_cpu = noise_bank[seed][index]
                        noise_np = noise_cpu.numpy().copy()
                        x1 = labels_by_checkpoint[label][index]
                        for t, bucket in zip(T_VALUES, T_BUCKETS):
                            _, x_t_np, v_target_np = common.harness.construct_flow(noise_np, x1, t)
                            x_t = torch.from_numpy(x_t_np).to(device=device, dtype=next(model.parameters()).dtype)
                            probe_start = time.perf_counter()
                            with torch.inference_mode():
                                v_hat = stage38._offline_step(model.action_head, processed_backbone, action_input, state_features, x_t, bucket)
                            latency = time.perf_counter() - probe_start
                            telemetry["per_probe_latency_s"].append({"checkpoint": label, "window_id": window["window_id"], "seed": seed, "t_bucket": bucket, "latency_s": latency})
                            v_hat_np = v_hat.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                            arrays = common.build_probe_arrays(noise_np, x1, t, v_hat_np)
                            physical, decoder_details = _decode_prediction(label, arrays["x1_hat"], snapshot, paths, common.harness, stage45c)
                            _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "decoded physical action is nonfinite or wrong shape")
                            from go2arx5_vla.evaluation.trajectory_metrics import comparison_metrics

                            canonical_metrics = comparison_metrics(
                                physical,
                                np.asarray(snapshot.physical_action, dtype=np.float64),
                                np.asarray(snapshot_record["anchor_world_m"], dtype=np.float64),
                                np.asarray(snapshot.target_position_world_m, dtype=np.float64),
                                prefix=f"{label}:{window['window_id']}:seed{seed}:t{bucket}",
                            )
                            probe_id = f"{label.replace('@', '_at_')}/w{index:02d}/seed{seed}/t{bucket:03d}"
                            evidence = common.write_probe_evidence(
                                output_root,
                                probe_id=probe_id,
                                identity={
                                    "checkpoint": label,
                                    "checkpoint_tree_sha256": admission.CHECKPOINT_TREE_SHA256[label],
                                    "stats_sha256": admission.STATS_SHA256[label],
                                    "stats_payload_sha256": admission.STATS_PAYLOAD_SHA256[label],
                                    "window_id": window["window_id"],
                                    "noise_seed": seed,
                                    "t": t,
                                    "t_bucket": bucket,
                                    "target_xyz_in_model_input": False,
                                    "target_id": snapshot.target_id,
                                },
                                arrays=arrays,
                                physical_metrics={
                                    "namespace": "DECODED_PHYSICAL_ACTION",
                                    "source": "x1_hat_only",
                                    "decoder": decoder_details,
                                    "stage44r_trajectory_metrics": canonical_metrics,
                                },
                            )
                            row = {
                                "schema": "CORELAB_STAGE45D_D1_FULL_PROBE_ROW_V1",
                                "checkpoint": label,
                                "checkpoint_tree_sha256": admission.CHECKPOINT_TREE_SHA256[label],
                                "stats_route": admission.STATS_ROUTES[label],
                                "stats_sha256": admission.STATS_SHA256[label],
                                "stats_payload_sha256": admission.STATS_PAYLOAD_SHA256[label],
                                "window_id": window["window_id"],
                                "target_id": snapshot.target_id,
                                "window_index": index,
                                "seed": seed,
                                "t": t,
                                "t_bucket": bucket,
                                "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
                                "target_xyz_in_model_input": False,
                                "raw_before_derived": True,
                                "normalized_metrics_namespace": "NORMALIZED_LOCAL_FLOW",
                                "physical_metrics_namespace": "DECODED_PHYSICAL_ACTION",
                                "raw_evidence": evidence,
                                "trajectory_metrics": canonical_metrics,
                                "finite": True,
                            }
                            _write_jsonl_durable(raw_jsonl, row)
                            rows.append(row)
            finally:
                hook.remove()
            _require(action_counter["count"] == WINDOW_COUNT * len(SEEDS) * len(T_VALUES), f"action-head count drift for {label}")
            _require(len(rows) == WINDOW_COUNT * len(SEEDS) * len(T_VALUES), f"probe row count drift for {label}")
            counters["action_head_calls"] += action_counter["count"]
            counters["model_forward"] += action_counter["count"]
            checkpoint_results.append({"checkpoint": label, "rows": len(rows), "action_head_calls": action_counter["count"], "conditioning_passes": WINDOW_COUNT})
            del model
            torch.cuda.empty_cache()
        _require(counters["model_construction"] == len(CHECKPOINT_ORDER), "model construction count drift")
        _require(counters["conditioning_passes"] == 72, "full conditioning count drift")
        _require(counters["action_head_calls"] == 648, "full action-head count drift")
        _require(counters["model_forward"] == 648, "full model forward count drift")
        margin = _resource_margin(torch, device, total_vram)
        telemetry["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        telemetry["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        telemetry["resource_margin"] = margin
        _require(margin["pass"], "VANDA_H200_RESOURCE_BLOCKER: VRAM margin below PI0117 floor")
        all_rows = []
        with raw_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                all_rows.append(json.loads(line))
        _require(len(all_rows) == 648, "raw row count drift before CPU summary")
        summary = _aggregate_rows(all_rows)
        common.write_json_durable(output_root / "cpu_summary.json", summary)
        seal = {
            "schema": "CORELAB_STAGE45D_D1_FULL_EXECUTION_SEAL_V1",
            "status": "PASS",
            "token": FULL_RETURN_TOKEN,
            "pi_gate": PI_GATE_COMMENT,
            "execution_head": binding["execution_checkout"]["authorized_head"],
            "smoke_pass_head": binding["execution_control"]["smoke_pass_head"],
            "profile_sha256": common.sha256_file(FULL_PROFILE_PATH),
            "binding_sha256": common.sha256_bytes(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")),
            "checkpoint_order": list(CHECKPOINT_ORDER),
            "checkpoint_results": checkpoint_results,
            "counters": counters,
            "telemetry": telemetry,
            "artifacts": artifacts,
            "evidence": {
                "raw_probe_rows_jsonl_sha256": common.sha256_file(raw_jsonl),
                "cpu_summary_sha256": common.sha256_file(output_root / "cpu_summary.json"),
                "raw_before_derived": True,
                "probe_count": 648,
                "physical_source": "x1_hat_only",
                "target_xyz_in_model_input": False,
            },
            "execution_boundary": {
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_write": 0,
                "Validation-E": 0,
                "Holdout-F": 0,
                "public_sampler_calls": 0,
                "get_action_calls": 0,
                "free_running_sampler_calls": 0,
                "WBC": 0,
                "simulator": 0,
                "hardware": 0,
                "pbs": 1,
                "h200": 1,
            },
        }
        common.write_json_durable(output_root / "telemetry.json", telemetry)
        common.write_json_durable(output_root / "full_seal.json", seal)
        return seal
    except Exception as exc:
        token = "STAGE45D_D1_VANDA_H200_ENGINEERING_BLOCKED_FOR_PI_REVIEW" if "RESOURCE_BLOCKER" in str(exc) else "STAGE45D_D1_VANDA_H200_SCIENTIFIC_BLOCKER_FOR_PI_REVIEW" if any(word in str(exc).lower() for word in ("nonfinite", "shape", "stats route", "mask", "target xyz", "normalization")) else "STAGE45D_D1_VANDA_H200_ENGINEERING_BLOCKED_FOR_PI_REVIEW"
        _failure_seal(output_root, str(exc), counters, token=token)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", type=Path, default=FULL_PROFILE_PATH)
    parser.add_argument("--runtime-binding", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan_only or args.audit_only:
            payload = validate_static_contract(args.profile)
            payload["mode"] = "audit-only" if args.audit_only else "plan-only"
        else:
            _require(args.runtime_binding is not None, "--execute requires --runtime-binding")
            profile = validate_profile(args.profile)
            _require(os.environ.get("STAGE45D_D1_CORELAB_PREFLIGHT_RUNTIME") == "authorized", "PI0117 H200 full runtime is not authorized")
            authority_token = os.environ.get("STAGE45D_D1_CORELAB_PREFLIGHT_AUTHORITY_TOKEN", "")
            payload = execute(_read_json(Path(args.runtime_binding)), profile=profile, authority_token=authority_token)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45D_D1_FULL_RUNTIME_ADAPTER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
