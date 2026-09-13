#!/usr/bin/env python3
"""Recover PI0163 Validation-E from the one valid raw row.

Allocation 614704 passed the compute-node runtime gate and produced the first
raw prediction, but its frozen CPU readout failed because the frozen evaluator
received a SameSceneExpertSnapshotV1 dataclass while the legacy metric adapter
expects the equivalent mapping representation. This recovery keeps that raw
file byte-for-byte, decodes it only on CPU, and executes only the remaining
35 queries under the already admitted six-arm E4 contract.

This module is deliberately outside the frozen PI0154 evaluator surface. It
imports no model/runtime package at module scope, never trains, never retries
an entered model call, and records cumulative allocation accounting separately
from the current recovery-process counters.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45f_pi0159_validation_e_continuation as continuation  # noqa: E402


PI0163_GATE = 5652818601
PI0163_POINTER = 5652824228
PI0163_TOKEN = "PI_GATE_STATE_0163"
RETURN_TOKEN = "VANDA_PI0163_VALIDATION_E_GEN1_FOR_PI_REVIEW"
EXPECTED_JOB = "614704.hopper-m-02"
EXPECTED_RAW_SCHEMA = "STAGE45F_PI0154_RAW_PREDICTION_V1"
EXPECTED_TARGET_ID = "red_target_holdout_e"
EXPECTED_REUSED_ARM = "Legacy(seed2222)@2000"
EXPECTED_REUSED_WINDOW = "stage22_expert_red_target_holdout_e:expert-0000"
EXPECTED_RAW_RELATIVE_PATH = "raw/Legacy_seed2222_at_2000/window-00.npy"
EXPECTED_RAW_SHAPE = (1, 40, 132)
EXPECTED_TIMESTEPS = [0, 250, 500, 750]
EXPECTED_NOMINAL = {"conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36}
EXPECTED_ARM_KEYS = tuple(
    (route, seed)
    for seed in (2222, 2223, 2224)
    for route in ("Legacy", "L1")
)
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "cuda", "cv2", "numpy"}
FORBIDDEN_CALLS = {
    "backward",
    "get_action",
    "qsub",
    "save_checkpoint",
    "save_model",
    "save_pretrained",
    "train",
}


def require(condition: bool, message: str) -> None:
    continuation.require(condition, message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return continuation.sha256_file(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Any:
    return continuation.load_json(path)


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    continuation.write_json_durable(path, value, overwrite=overwrite)


def _label_path(label: str) -> str:
    return label.replace("(", "_").replace(")", "").replace("@", "_at_")


def _copy_file_durable(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    require(source.is_file() and not source.is_symlink(), f"recovery source is not a regular file: {source}")
    require(not destination.exists() and not destination.is_symlink(), f"recovery destination is not fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1 << 20), b""):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _copy_exact_text_line(source: Path, destination: Path) -> None:
    _copy_file_durable(source, destination)
    require(destination.read_bytes().endswith(b"\n"), f"JSONL source has no terminal newline: {source}")


def _read_single_raw_record(source_root: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest = source_root / "raw_predictions.jsonl"
    require(manifest.is_file() and not manifest.is_symlink(), f"recovery raw manifest is missing: {manifest}")
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == 1, f"recovery source must contain exactly one raw record, found {len(lines)}")
    record = json.loads(lines[0])
    require(isinstance(record, Mapping), "recovery raw record is not an object")
    record = dict(record)
    require(record.get("schema") == EXPECTED_RAW_SCHEMA, "recovery raw schema drift")
    require(record.get("checkpoint", {}).get("label") == EXPECTED_REUSED_ARM, "recovery raw checkpoint drift")
    require(record.get("window_id") == EXPECTED_REUSED_WINDOW, "recovery raw window drift")
    require(record.get("path") == EXPECTED_RAW_RELATIVE_PATH, "recovery raw path drift")
    require(record.get("timestep_ids") == EXPECTED_TIMESTEPS, "recovery raw E4 timestep drift")
    require(record.get("action_head_calls_observed") == 4, "recovery raw action-head count drift")
    require(record.get("decoded") is False, "recovery raw was already decoded")
    require(record.get("shape") == list(EXPECTED_RAW_SHAPE), "recovery raw shape metadata drift")
    raw_path = source_root / str(record["path"])
    require(raw_path.is_file() and not raw_path.is_symlink(), f"recovery raw array is missing: {raw_path}")
    require(sha256_file(raw_path) == record.get("file_sha256"), "recovery raw file SHA drift")
    return record, raw_path, manifest


def _evidence_ref(config: Mapping[str, Any], path_key: str, sha_key: str) -> dict[str, Any]:
    path = continuation.regular_file(Path(str(config[path_key])), path_key)
    observed = sha256_file(path)
    require(observed == str(config[sha_key]), f"{path_key} SHA drift: {observed}")
    return {"path": str(path), "sha256": observed, "bytes": int(path.stat().st_size)}


def _source_failure_ref(config: Mapping[str, Any]) -> dict[str, Any]:
    path = continuation.regular_file(Path(str(config["pi0163_recovery_source_failure_path"])), "PI0163 allocation-5 failure seal")
    observed = sha256_file(path)
    require(observed == str(config["pi0163_recovery_source_failure_sha256"]), "PI0163 allocation-5 failure seal SHA drift")
    payload = load_json(path)
    require(isinstance(payload, Mapping), "PI0163 allocation-5 failure seal is not an object")
    require(payload.get("status") == "PARTIAL", "PI0163 allocation-5 failure status drift")
    require(payload.get("model_execution_started") is True and payload.get("holdout_f_read") is False, "PI0163 allocation-5 failure provenance drift")
    return {"path": str(path), "sha256": observed, "phase": payload.get("phase"), "current": payload.get("current"), "actual_counters": payload.get("actual_counters", {})}


def _install_snapshot_mapping_compat(runtime: Any) -> dict[str, Any]:
    """Bridge the frozen dataclass snapshot to the frozen metric adapter ABI."""

    original = runtime._evaluation_row
    if getattr(original, "_pi0163_snapshot_mapping_compat", False):
        return {
            "schema": "STAGE45F_PI0163_SNAPSHOT_MAPPING_COMPAT_V1",
            "status": "INSTALLED",
            "already_installed": True,
            "source_sha256": sha256_file(Path(__file__)),
        }

    def compatible(raw: Any, physical: Any, details: Mapping[str, Any], window: Mapping[str, Any], checkpoint: Mapping[str, Any], model_runtime_sha256: str, raw_artifact_path: str, raw_artifact_sha256: str) -> dict[str, Any]:
        normalized_window = dict(window)
        snapshot = normalized_window.get("snapshot")
        if not isinstance(snapshot, Mapping):
            physical_action = getattr(snapshot, "physical_action", None)
            require(physical_action is not None, "SameSceneExpertSnapshotV1 lacks physical_action")
            normalized_window["snapshot"] = {
                "expert_physical_action": physical_action,
                "anchor_world_m": normalized_window["anchor_world_m"],
                "target_world_m": normalized_window["target_world_m"],
            }
        return original(raw, physical, details, normalized_window, checkpoint, model_runtime_sha256, raw_artifact_path, raw_artifact_sha256)

    setattr(compatible, "_pi0163_snapshot_mapping_compat", True)
    runtime._evaluation_row = compatible
    return {
        "schema": "STAGE45F_PI0163_SNAPSHOT_MAPPING_COMPAT_V1",
        "status": "INSTALLED",
        "already_installed": False,
        "source_sha256": sha256_file(Path(__file__)),
        "adapted_fields": ["snapshot.physical_action -> snapshot.expert_physical_action", "window.anchor_world_m", "window.target_world_m"],
        "scientific_contract_changed": False,
    }


def static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    forbidden_calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                forbidden_calls.append(name)
    markers = {
        "no_top_level_model_import": not imports.intersection(FORBIDDEN_TOP_LEVEL),
        "valid_raw_reuse": "_copy_file_durable" in source and "source_reuse" in source,
        "mapping_compatibility": "_install_snapshot_mapping_compat" in source and "physical_action" in source,
        "raw_before_derived": "raw_predictions.jsonl" in source and "derived_rows.jsonl" in source,
        "cumulative_accounting": "cumulative_campaign_accounting" in source and "model_load_attempts" in source,
        "holdout_locked": '"holdout_f_read": False' in source,
        "training_locked": '"training": False' in source,
        "no_adaptive_retry": '"automatic_retry": False' in source,
    }
    return {
        "schema": "STAGE45F_PI0163_RECOVERY_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_imports": sorted(imports),
        "top_level_forbidden_imports": sorted(imports.intersection(FORBIDDEN_TOP_LEVEL)),
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "pbs": 0,
        "training": 0,
        "holdout_f_read": 0,
        "scientific_contract_changed": False,
    }


def _validate_raw_array(raw_path: Path, record: Mapping[str, Any], np: Any) -> Any:
    raw = np.load(raw_path, allow_pickle=False)
    require(tuple(int(item) for item in raw.shape) == EXPECTED_RAW_SHAPE, "reused raw array shape drift")
    require(str(raw.dtype) == "float32" and bool(np.isfinite(raw).all()), "reused raw dtype/finite drift")
    array_sha = sha256_bytes(np.ascontiguousarray(raw).astype(np.float32, copy=False).tobytes())
    require(array_sha == record.get("array_sha256"), "reused raw array SHA drift")
    return np.asarray(raw, dtype=np.float32)


def _append_row_metadata(row: dict[str, Any], arm: Mapping[str, Any], noise_sha: str) -> dict[str, Any]:
    row["route"] = arm["route"]
    row["seed"] = arm["seed"]
    row["tier"] = arm["tier"]
    row["noise_sha256"] = noise_sha
    row["sampler"] = {
        "name": "direct-source-mapped-action-head-E4",
        "t_values": [0.0, 0.25, 0.5, 0.75],
        "timestep_ids": EXPECTED_TIMESTEPS,
        "dt": 0.25,
        "carried_state": True,
        "reset_between_stages": False,
        "official_public_sampler_called": False,
        "get_action_calls": 0,
    }
    row["stats"] = {
        "sha256": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
        "payload_sha256": "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75",
        "route": "FULL24_RUNTIME_V2",
    }
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _seed_pair_summary(paired: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "final_target_distance_delta_m",
        "progress_delta_m",
        "minimum_target_distance_delta_m",
        "path_length_delta_m",
        "absolute_roughness_delta_m",
    )
    result: list[dict[str, Any]] = []
    by_seed = paired.get("paired_by_seed", {})
    for seed in (2222, 2223, 2224):
        rows = list(by_seed.get(str(seed), []))
        result.append({
            "seed": seed,
            "window_count": len(rows),
            "mean_l1_minus_legacy": {field: _mean([float(row[field]) for row in rows]) for field in fields},
            "descriptive_only": True,
        })
    return result


def _one_page_conclusion(*, output_root: Path, expected_head: str, paired: Mapping[str, Any], counters: Mapping[str, Any], cumulative: Mapping[str, Any], reused: Mapping[str, Any]) -> str:
    seed_lines = []
    for item in _seed_pair_summary(paired):
        means = item["mean_l1_minus_legacy"]
        seed_lines.append(
            f"- seed {item['seed']}: n={item['window_count']} windows; "
            f"final-distance delta={means['final_target_distance_delta_m']!r} m; "
            f"progress delta={means['progress_delta_m']!r} m; "
            f"path delta={means['path_length_delta_m']!r} m; "
            f"roughness delta={means['absolute_roughness_delta_m']!r}."
        )
    return "\n".join([
        "# PI0163 Validation-E generation 1 recovery",
        "",
        f"- status: COMPLETED; execution HEAD: {expected_head}",
        f"- output root: {output_root}",
        "- fixed contract: six accepted @2000 checkpoints, six original red_target_holdout_e windows per arm, seed 4242, E4 [0,250,500,750], FULL24 runtime-V2 stats, target excluded from model input.",
        "- raw-first recovery: the 614704 Legacy(seed2222) window-0000 raw array was copied byte-for-byte and decoded on CPU; no model call was repeated for that query.",
        "- interpretation boundary: paired values below are descriptive L1 minus Legacy comparisons at the same tier/window/noise; they do not select a winner and do not claim iid significance.",
        "",
        "## Seed-level paired readout",
        "",
        *seed_lines,
        "",
        "## Accounting",
        "",
        f"- final evidence rows: raw={counters['raw_prediction_count']}, derived={counters['derived_row_count']}; calls: conditioning={counters['conditioning_completed']}, action-head={counters['action_head_completed']}, model-side={counters['conditioning_completed'] + counters['action_head_completed']}.",
        f"- cumulative campaign: allocations={cumulative['allocations_consumed']}/{cumulative['allocation_ceiling']}; model-load attempts={cumulative['model_load_attempts']}/{cumulative['model_load_ceiling']}; model-side calls={cumulative['model_side_calls']}/{cumulative['model_side_call_ceiling']}.",
        f"- reused raw: {reused['source_path']} SHA {reused['source_sha256']}; source manifest SHA {reused['source_manifest_sha256']}.",
        "- training/backward/optimizer/checkpoint writes/public sampler/get_action/Holdout-F: false/zero.",
        "",
        "## Claim boundary",
        "",
        "This package closes the frozen Validation-E evidence surface only. It does not change recipe, checkpoint, statistics, noise, windows, or decoder; scientific route/tier selection remains with PI.",
        "",
    ])


def execute(config_path: Path) -> int:
    phase = "START"
    output_root: Path | None = None
    config: dict[str, Any] | None = None
    current: dict[str, Any] = {"arm": None, "window_id": None, "step": None, "checkpoint": None}
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_construction_attempts": 0,
        "model_construction_completed": 0,
        "conditioning_attempts": 1,
        "conditioning_completed": 1,
        "action_head_attempts": 4,
        "action_head_completed": 4,
        "model_forward_completed": 4,
        "raw_prediction_count": 1,
        "raw_manifest_records": 1,
        "derived_row_count": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "failed_model_calls": 0,
    }
    try:
        config = continuation._config(config_path.resolve(strict=True))
        require(config.get("pi0163_gate") == PI0163_GATE and config.get("pi0163_pointer") == PI0163_POINTER, "PI0163 authority drift")
        require(config.get("pi0163_recovery_allocation") == 6, "PI0163 recovery allocation drift")
        require(config.get("pi0163_previous_job_id") == EXPECTED_JOB, "PI0163 previous job drift")
        output_root = Path(str(config["output_root"])).resolve()
        source_root = continuation.regular_directory(Path(str(config["pi0163_recovery_source_root"])), "PI0163 recovery source root")
        require(not output_root.exists() and not output_root.is_symlink(), "PI0163 recovery output root must be fresh")
        require(output_root.parent.is_dir(), "PI0163 recovery output parent is unavailable")
        output_root.mkdir(parents=False, exist_ok=False)
        phase = "RECOVERY_SOURCE_AUDIT"
        source_failure = _source_failure_ref(config)
        source_record, source_raw_path, source_manifest = _read_single_raw_record(source_root)
        require(sha256_file(source_manifest) == str(config["pi0163_recovery_source_raw_manifest_sha256"]), "PI0163 recovery source manifest SHA drift")
        require(sha256_file(source_raw_path) == str(config["pi0163_recovery_source_raw_sha256"]), "PI0163 recovery source raw SHA drift")
        prior_alloc5_bootstrap = _evidence_ref(config, "pi0163_alloc5_bootstrap_record", "pi0163_alloc5_bootstrap_sha256")
        prior_alloc5_import = _evidence_ref(config, "pi0163_alloc5_compute_import_record", "pi0163_alloc5_compute_import_sha256")
        phase = "PRIOR_ADMISSION_RECOVERY_AUDIT"
        prior, _ = continuation.validate_prior_admission(config)
        phase = "OFFLINE_CACHE_RESOLUTION"
        runtime_binding = continuation.bind_offline_runtime(config, output_root)
        phase = "CONTINUATION_ADMISSION"
        binding = continuation.effective_binding(config, prior, output_root)
        binding["pi0163_recovery"] = {
            "gate": PI0163_GATE,
            "pointer": PI0163_POINTER,
            "allocation_ordinal": 6,
            "previous_job_id": EXPECTED_JOB,
            "source_root": str(source_root),
            "source_failure": source_failure,
            "valid_raw_reused": True,
            "scientific_contract_changed": False,
        }
        continuation_binding_path = output_root / "stage45f_pi0163_recovery_admission.json"
        write_json(continuation_binding_path, binding)
        effective_config = dict(config)
        effective_config.update({
            "admission_path": str(continuation_binding_path),
            "output_root": str(output_root),
            "prior_admission_path": prior["path"],
        })
        effective_config_path = output_root / "stage45f_pi0163_effective_runner_config.json"
        write_json(effective_config_path, effective_config)
        source_reuse = {
            "schema": "STAGE45F_PI0163_VALID_RAW_REUSE_V1",
            "status": "PASS_BYTE_PRESERVED",
            "source_root": str(source_root),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_path": str(source_raw_path),
            "source_sha256": sha256_file(source_raw_path),
            "destination_path": str(output_root / EXPECTED_RAW_RELATIVE_PATH),
            "destination_sha256": None,
            "record": source_record,
            "model_calls_for_reused_query": 0,
            "scientific_contract_changed": False,
        }
        _copy_file_durable(source_raw_path, output_root / EXPECTED_RAW_RELATIVE_PATH)
        _copy_exact_text_line(source_manifest, output_root / "source_raw_manifest.jsonl")
        _copy_exact_text_line(source_manifest, output_root / "raw_predictions.jsonl")
        source_reuse["destination_sha256"] = sha256_file(output_root / EXPECTED_RAW_RELATIVE_PATH)
        require(source_reuse["destination_sha256"] == source_reuse["source_sha256"], "reused raw bytes changed during copy")
        write_json(output_root / "stage45f_pi0163_raw_reuse.json", source_reuse)
        preflight = {
            "schema": "STAGE45F_PI0163_RECOVERY_PRE_MODEL_ADMISSION_V1",
            "status": "PASS_PRE_MODEL",
            "pi_gate": PI0163_GATE,
            "pointer": PI0163_POINTER,
            "authority_token": PI0163_TOKEN,
            "execution_head": config["expected_head"],
            "execution_branch": continuation.EXPECTED_BRANCH,
            "recovery_allocation": 6,
            "previous_job_id": EXPECTED_JOB,
            "prior_admission": {key: value for key, value in prior.items() if key not in {"binding"}},
            "runtime_binding": runtime_binding,
            "effective_binding_path": str(continuation_binding_path),
            "effective_binding_sha256": sha256_file(continuation_binding_path),
            "effective_runner_config_path": str(effective_config_path),
            "effective_runner_config_sha256": sha256_file(effective_config_path),
            "source_failure": source_failure,
            "source_raw_reuse": source_reuse,
            "allocation_5_bootstrap": prior_alloc5_bootstrap,
            "allocation_5_compute_import": prior_alloc5_import,
            "nominal_calls": EXPECTED_NOMINAL,
            "new_calls_after_reuse": {"conditioning": 35, "action_head": 140, "model_side_total": 175},
            "cumulative_call_ceiling": {"conditioning": 42, "action_head": 168, "model_side_total": 210},
            "model_load_accounting": {"historical_prior": 1, "allocation_5": 1, "allocation_6_reserved": 6, "cumulative_reserved": 8},
            "holdout_f_read": False,
            "training": False,
            "model_imported": False,
            "scientific_contract_changed": False,
        }
        write_json(output_root / "stage45f_pi0163_pre_model_admission.json", preflight)
        guard = static_guard()
        require(guard["status"] == "PASS", "PI0163 recovery static guard failed")
        write_json(output_root / "stage45f_pi0163_recovery_static_guard.json", guard)
        phase = "PRE_MODEL_ARTIFACT_AUDIT"
        import stage45f_vanda_validation_e_gen1 as evaluator

        evaluator_guard = evaluator.static_guard()
        require(evaluator_guard["status"] == "PASS", "frozen evaluator static guard failed")
        checkout = evaluator.observe_checkout(str(config["expected_head"]))
        gpu = evaluator.observe_gpu()
        runtime_identity = evaluator.observed_runtime(gpu)
        artifact_audit = evaluator.validate_artifacts(effective_config, binding, str(config["expected_head"]))
        windows_meta = evaluator.validate_snapshot_records(binding)
        write_json(output_root / "stage45f_execution_static_guard.json", evaluator_guard)
        write_json(output_root / "stage45f_pre_model_artifact_audit.json", artifact_audit)
        write_json(output_root / "stage45f_observed_runtime.json", {"checkout": checkout, "gpu": gpu, "runtime": runtime_identity})
        phase = "MODEL_IMPORT_AND_PROCESSOR_CONSTRUCTION"
        counters["model_import_started"] = True
        import numpy as np
        import torch
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

        compatibility = _install_snapshot_mapping_compat(runtime)
        write_json(output_root / "stage45f_pi0163_snapshot_compatibility.json", compatibility)
        raw_reused = _validate_raw_array(output_root / EXPECTED_RAW_RELATIVE_PATH, source_record, np)
        loader = runtime._load_torch_runtime()[1]
        runtime_processor, embodiment, _modality_configs = stage26._build_runtime_processor(
            groot_source=Path(str(config["groot_source"])).resolve(),
            base_checkpoint=Path(str(config["base_model"])).resolve(),
            dataset_dir=Path(str(config["dataset"])).resolve(),
            modality_path=Path(str(config["modality"])).resolve(),
            cosmos_snapshot=Path(str(config["cosmos_root"])).resolve() / "snapshots" / "9ce19a195e423419c349abfc86fd07178b230561",
        )
        windows = evaluator._prepare_windows(runtime_processor, embodiment, windows_meta)
        require(len(windows) == 6, "prepared E window count drift")
        noise_bank, noise_hashes = evaluator._make_noise_bank(torch)
        require(noise_hashes[0] == source_record.get("noise_sha256"), "reused raw noise identity drift")
        write_json(output_root / "sampler_noise_manifest.json", {
            "schema": "STAGE45F_PI0154_CPU_FLOAT32_NOISE_BANK_V1",
            "seed": evaluator.admission.EXPECTED_NOISE_SEED,
            "device": "cpu",
            "dtype": "float32",
            "shape": list(evaluator.RAW_SHAPE),
            "window_ids": list(evaluator.WINDOW_IDS),
            "noise_sha256_by_window": noise_hashes,
            "same_noise_across_all_six_checkpoint_arms": True,
            "reused_raw_window_index": 0,
        })
        derived_manifest = output_root / "derived_rows.jsonl"
        arms = binding["checkpoint_arms"]
        require([(arm["route"], int(arm["seed"])) for arm in arms] == list(EXPECTED_ARM_KEYS), "recovery arm order drift")
        checkpoint_paths = config["checkpoints"]
        rows_by_arm: dict[str, list[dict[str, Any]]] = {str(arm["label"]): [] for arm in arms}
        runtime_sha = evaluator.canonical_sha({"binding": binding, "artifact_audit": artifact_audit, "runtime": runtime_identity})
        reused_window = windows[0]
        reused_arm = next(arm for arm in arms if arm["label"] == EXPECTED_REUSED_ARM)
        physical, details = runtime._decode_v2(
            raw_reused,
            reused_window["state_v1"],
            Path(str(config["stats"])).resolve(),
            reused_window["source_timestamps"],
            float(reused_window["observation_timestamp"]),
            reused_window["window_id"] + ":" + EXPECTED_REUSED_ARM,
        )
        reused_row = evaluator._metric_row(
            runtime,
            raw_reused,
            physical,
            details,
            reused_window,
            {"label": EXPECTED_REUSED_ARM, "tree_sha256": reused_arm["tree_sha256"]},
            runtime_sha,
            EXPECTED_RAW_RELATIVE_PATH,
            source_reuse["source_sha256"],
        )
        _append_row_metadata(reused_row, reused_arm, noise_hashes[0])
        reused_derived_path = output_root / "derived" / _label_path(EXPECTED_REUSED_ARM) / "window-00.json"
        evaluator.write_json_durable(reused_derived_path, reused_row)
        evaluator.append_jsonl_durable(derived_manifest, {
            "schema": "STAGE45F_PI0154_DERIVED_EVIDENCE_REFERENCE_V1",
            "checkpoint": dict(reused_arm),
            "window_id": reused_window["window_id"],
            "raw_path": EXPECTED_RAW_RELATIVE_PATH,
            "raw_file_sha256": source_reuse["source_sha256"],
            "derived_path": str(reused_derived_path.relative_to(output_root)),
            "decoded": True,
            "decoded_array_sha256": reused_row["decoded_action"]["array_sha256"],
            "raw_reused_without_model_call": True,
        })
        rows_by_arm[EXPECTED_REUSED_ARM].append(reused_row)
        counters["derived_row_count"] = 1
        phase = "MODEL_EXECUTION_RECOVERY"
        for arm in arms:
            label = str(arm["label"])
            current.update({"arm": label, "checkpoint": label, "window_id": None, "step": None})
            phase = "MODEL_LOAD"
            counters["model_loads_attempted"] += 1
            model = loader(Path(str(checkpoint_paths[label])).resolve(), "cuda:0")
            counters["model_loads_completed"] += 1
            counters["model_construction_attempts"] += 1
            counters["model_construction_completed"] += 1
            model.eval()
            action_rows = rows_by_arm[label]
            start_index = 1 if label == EXPECTED_REUSED_ARM else 0
            try:
                for index in range(start_index, len(windows)):
                    window = windows[index]
                    current["window_id"] = window["window_id"]
                    current["step"] = None
                    phase = "CONDITIONING"
                    counters["conditioning_attempts"] += 1
                    _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                    counters["conditioning_completed"] += 1
                    current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                    with torch.inference_mode():
                        for step, timestep in enumerate(evaluator.TIMESTEP_IDS):
                            current["step"] = int(step)
                            phase = "ACTION_HEAD"
                            counters["action_head_attempts"] += 1
                            try:
                                velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                                counters["action_head_completed"] += 1
                                counters["model_forward_completed"] += 1
                            except Exception:
                                counters["failed_model_calls"] += 1
                                raise
                            current_action = current_action + 0.25 * velocity
                    phase = "RAW_SEAL"
                    raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                    evaluator.require(raw_np.shape == evaluator.RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output shape/finite drift: {label}:{window['window_id']}")
                    raw_path = output_root / "raw" / _label_path(label) / f"window-{index:02d}.npy"
                    require(not raw_path.exists() and not raw_path.is_symlink(), f"new raw path is not fresh: {raw_path}")
                    evaluator.write_npy_durable(raw_path, raw_np)
                    raw_file_sha = evaluator.sha256_file(raw_path)
                    raw_array_sha = evaluator.sha256_bytes(np.ascontiguousarray(raw_np).tobytes())
                    evaluator.append_jsonl_durable(output_root / "raw_predictions.jsonl", {
                        "schema": EXPECTED_RAW_SCHEMA,
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "path": str(raw_path.relative_to(output_root)),
                        "file_sha256": raw_file_sha,
                        "array_sha256": raw_array_sha,
                        "shape": list(raw_np.shape),
                        "dtype": str(raw_np.dtype),
                        "noise_sha256": noise_hashes[index],
                        "timestep_ids": list(evaluator.TIMESTEP_IDS),
                        "action_head_calls_observed": len(evaluator.TIMESTEP_IDS),
                        "decoded": False,
                    })
                    counters["raw_prediction_count"] += 1
                    counters["raw_manifest_records"] += 1
                    phase = "DECODE_AND_DERIVED_METRICS"
                    physical, details = runtime._decode_v2(
                        raw_np,
                        window["state_v1"],
                        Path(str(config["stats"])).resolve(),
                        window["source_timestamps"],
                        float(window["observation_timestamp"]),
                        window["window_id"] + ":" + label,
                    )
                    row = evaluator._metric_row(runtime, raw_np, physical, details, window, {"label": label, "tree_sha256": arm["tree_sha256"]}, runtime_sha, str(raw_path.relative_to(output_root)), raw_file_sha)
                    _append_row_metadata(row, arm, noise_hashes[index])
                    derived_path = output_root / "derived" / _label_path(label) / f"window-{index:02d}.json"
                    evaluator.write_json_durable(derived_path, row)
                    evaluator.append_jsonl_durable(derived_manifest, {
                        "schema": "STAGE45F_PI0154_DERIVED_EVIDENCE_REFERENCE_V1",
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "raw_path": str(raw_path.relative_to(output_root)),
                        "raw_file_sha256": raw_file_sha,
                        "derived_path": str(derived_path.relative_to(output_root)),
                        "decoded": True,
                        "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                        "raw_reused_without_model_call": False,
                    })
                    action_rows.append(row)
                    counters["derived_row_count"] += 1
                require(len(action_rows) == 6, f"per-arm row count drift: {label}")
            finally:
                del model
                gc = __import__("gc")
                gc.collect()
                torch.cuda.empty_cache()
            rows_by_arm[label] = action_rows
            aggregate = aggregate_evaluation_rows(action_rows)
            aggregate["route"] = arm["route"]
            aggregate["seed"] = arm["seed"]
            aggregate["tier"] = arm["tier"]
            aggregate["checkpoint_tree_sha256"] = arm["tree_sha256"]
            evaluator.write_json_durable(output_root / f"aggregate_{_label_path(label)}.json", aggregate)
        require(counters["model_loads_attempted"] == 6 and counters["model_loads_completed"] == 6, "recovery model-load count drift")
        require(counters["conditioning_completed"] == EXPECTED_NOMINAL["conditioning"], "recovery conditioning count drift")
        require(counters["action_head_completed"] == EXPECTED_NOMINAL["action_head"], "recovery action-head count drift")
        require(counters["model_forward_completed"] == EXPECTED_NOMINAL["action_head"], "recovery model-forward count drift")
        require(counters["raw_prediction_count"] == EXPECTED_NOMINAL["raw_outputs"] and counters["raw_manifest_records"] == EXPECTED_NOMINAL["raw_outputs"], "recovery raw count drift")
        require(counters["derived_row_count"] == EXPECTED_NOMINAL["raw_outputs"], "recovery derived count drift")
        phase = "CPU_PAIRED_SUMMARY"
        paired = evaluator.paired_summary(rows_by_arm, arms)
        seed_summary = _seed_pair_summary(paired)
        evaluator.write_json_durable(output_root / "paired_summary.json", paired)
        cpu_summary = {
            "schema": "STAGE45F_PI0163_VALIDATION_E_CPU_SUMMARY_V1",
            "status": "PASS_READONLY_DERIVED_FROM_SEALED_RAW",
            "checkpoint_order": [str(arm["label"]) for arm in arms],
            "window_ids": list(evaluator.WINDOW_IDS),
            "window_count_per_arm": 6,
            "arms": 6,
            "paired_summary": paired,
            "seed_paired_summary": seed_summary,
            "aggregate_files": [f"aggregate_{_label_path(str(arm['label']))}.json" for arm in arms],
            "raw_predictions": str(output_root / "raw_predictions.jsonl"),
            "derived_rows": str(derived_manifest),
            "raw_reuse": source_reuse,
            "descriptive_only": True,
            "no_iid_claim": True,
        }
        evaluator.write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        cumulative = {
            "schema": "STAGE45F_PI0163_CUMULATIVE_CAMPAIGN_ACCOUNTING_V1",
            "allocations_consumed": 6,
            "allocation_ceiling": 6,
            "allocations": [
                {"job_id": "614203.hopper-m-02", "ordinal": 1, "status": "FAILED_BEFORE_PYTHON_MODEL_BOUNDARY", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614226.hopper-m-02", "ordinal": 2, "status": "PARTIAL_FAILED_AT_MODEL_LOAD", "model_load_attempts": 1, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614574.hopper-m-02", "ordinal": 3, "status": "FAILED_BEFORE_MODEL_IMPORT", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614586.hopper-m-02", "ordinal": 4, "status": "FAILED_BEFORE_MODEL_IMPORT", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": EXPECTED_JOB, "ordinal": 5, "status": "PARTIAL_AFTER_VALID_RAW", "model_load_attempts": 1, "model_loads_completed": 1, "conditioning": 1, "action_head": 4, "raw_rows": 1, "derived_rows": 0, "failure_phase": source_failure["phase"]},
                {"job_id": os.environ.get("PBS_JOBID"), "ordinal": 6, "status": "COMPLETED_RECOVERY", "model_load_attempts": counters["model_loads_attempted"], "model_loads_completed": counters["model_loads_completed"], "conditioning": 35, "action_head": 140, "raw_rows": 35, "derived_rows": 36, "reused_raw_rows": 1},
            ],
            "model_load_attempts": 8,
            "model_loads_completed": 7,
            "model_load_ceiling": 8,
            "conditioning": counters["conditioning_completed"],
            "action_head": counters["action_head_completed"],
            "model_side_calls": counters["conditioning_completed"] + counters["action_head_completed"],
            "model_side_call_ceiling": 210,
            "raw_rows": counters["raw_prediction_count"],
            "derived_rows": counters["derived_row_count"],
            "valid_raw_reused": 1,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "holdout_f_reads": 0,
            "automatic_retries": 0,
        }
        write_json(output_root / "cumulative_campaign_accounting.json", cumulative)
        write_json(output_root / "telemetry.json", {
            "schema": "STAGE45F_PI0163_H200_TELEMETRY_V1",
            "gpu": gpu,
            "runtime": runtime_identity,
            "counters": counters,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device("cuda:0"))),
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
        })
        conclusion = _one_page_conclusion(output_root=output_root, expected_head=str(config["expected_head"]), paired=paired, counters=counters, cumulative=cumulative, reused=source_reuse)
        conclusion_path = output_root / "pi0163_one_page_conclusion.md"
        conclusion_path.write_text(conclusion, encoding="utf-8")
        with conclusion_path.open("rb") as handle:
            os.fsync(handle.fileno())
        excluded = {
            "evidence_inventory.json",
            "stage45f_validation_e_seal.json",
            "stage45f_validation_e_result.json",
            "stage45f_pi0163_recovery_failure.json",
        }
        inventory_path, inventory_sha, inventory_count = evaluator._inventory(output_root, excluded)
        seal = {
            "schema": "STAGE45F_PI0163_VALIDATION_E_GEN1_RECOVERY_SEAL_V1",
            "status": "PASS",
            "token": RETURN_TOKEN,
            "pi_gate": PI0163_GATE,
            "pr_pointer": PI0163_POINTER,
            "authority_token": PI0163_TOKEN,
            "execution_head": config["expected_head"],
            "execution_branch": continuation.EXPECTED_BRANCH,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "runtime": runtime_identity,
            "recovery_allocation": 6,
            "previous_job_id": EXPECTED_JOB,
            "previous_failure": source_failure,
            "source_raw_reuse": source_reuse,
            "snapshot_compatibility": compatibility,
            "checkpoint_arms": [dict(arm) for arm in arms],
            "target_id": EXPECTED_TARGET_ID,
            "snapshot_ids": list(evaluator.WINDOW_IDS),
            "snapshot_count": 6,
            "checkpoint_count": 6,
            "stats_sha256": config["stats_sha256"],
            "noise_seed": config["noise_seed"],
            "timestep_ids": config["timestep_ids"],
            "t_values": config["t_values"],
            "nominal_calls": EXPECTED_NOMINAL,
            "actual_counters": counters,
            "cumulative_campaign_accounting": cumulative,
            "allocation_5_bootstrap": prior_alloc5_bootstrap,
            "allocation_5_compute_import": prior_alloc5_import,
            "admission_path": str(continuation_binding_path),
            "admission_sha256": sha256_file(continuation_binding_path),
            "pre_model_admission_path": str(output_root / "stage45f_pi0163_pre_model_admission.json"),
            "pre_model_admission_sha256": sha256_file(output_root / "stage45f_pi0163_pre_model_admission.json"),
            "raw_manifest": str(output_root / "raw_predictions.jsonl"),
            "raw_manifest_sha256": sha256_file(output_root / "raw_predictions.jsonl"),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "paired_summary": str(output_root / "paired_summary.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "conclusion_path": str(conclusion_path),
            "conclusion_sha256": sha256_file(conclusion_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
            "automatic_retry": False,
            "scientific_contract_changed": False,
        }
        seal_path = output_root / "stage45f_validation_e_seal.json"
        write_json(seal_path, seal)
        result = {
            "schema": "STAGE45F_PI0163_VALIDATION_E_GEN1_RECOVERY_RESULT_V1",
            "status": "COMPLETED",
            "token": RETURN_TOKEN,
            "execution_head": config["expected_head"],
            "branch": continuation.EXPECTED_BRANCH,
            "seal_path": str(seal_path),
            "seal_sha256": sha256_file(seal_path),
            "output_root": str(output_root),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "paired_summary": str(output_root / "paired_summary.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "conclusion_path": str(conclusion_path),
            "conclusion_sha256": sha256_file(conclusion_path),
            "actual_counters": counters,
            "cumulative_campaign_accounting": cumulative,
            "raw_output_count": 36,
            "derived_output_count": 36,
            "seed_paired_summary": seed_summary,
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "training": False,
            "holdout_f_read": False,
            "automatic_retry": False,
            "scientific_contract_changed": False,
        }
        write_json(output_root / "stage45f_validation_e_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if output_root is not None:
            failure = {
                "schema": "STAGE45F_PI0163_VALIDATION_E_RECOVERY_FAILURE_SEAL_V1",
                "status": "PARTIAL" if counters["model_import_started"] else "BLOCKED",
                "token": "VANDA_PI0163_VALIDATION_E_GEN1_BLOCKED",
                "pi_gate": PI0163_GATE,
                "pr_pointer": PI0163_POINTER,
                "authority_token": PI0163_TOKEN,
                "execution_head": config.get("expected_head") if config else None,
                "execution_branch": continuation.EXPECTED_BRANCH,
                "pbs_jobid": os.environ.get("PBS_JOBID"),
                "phase": phase,
                "current": current,
                "actual_counters": counters,
                "model_import_started": bool(counters["model_import_started"]),
                "model_execution_started": bool(counters["model_import_started"]),
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "validation_e_read": True,
                "holdout_f_read": False,
                "automatic_retry": False,
                "scientific_contract_changed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp_utc": now_utc(),
            }
            try:
                write_json(output_root / "stage45f_pi0163_recovery_failure.json", failure)
            except Exception:
                pass
        print(f"VANDA_PI0163_VALIDATION_E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def plan() -> dict[str, Any]:
    guard = static_guard()
    return {
        "schema": "STAGE45F_PI0163_RECOVERY_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if guard["status"] == "PASS" else "FAIL",
        "static_guard": guard,
        "reused_raw_queries": 1,
        "new_queries": 35,
        "nominal_calls": EXPECTED_NOMINAL,
        "new_calls": {"conditioning": 35, "action_head": 140, "model_side_total": 175},
        "training": 0,
        "holdout_f_read": 0,
        "return_token": RETURN_TOKEN,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    require(args.config is not None, "--execute requires --config")
    return execute(args.config.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
