#!/usr/bin/env python3
"""CPU-only admission contract for Stage45E-L1.

The materialized binding is generated from the actual Hopper checkout and
runtime, then checked before importing Torch/GR00T.  This module contains no
model-side import and does not inspect CUDA.  It freezes the experiment's
scientific inputs while leaving only the observed host, GPU and output root
late-bound.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


STAGE = "Stage45E-L1"
PI_GATE_COMMENT = 5634852590
PR54_POINTER_COMMENT = 5634856575
ISSUE50_POINTER_COMMENT = None
AUTHORITY_TOKEN = "PI_GATE_STATE_0123"
BRANCH = "stage45e/l1-physical-error-aware-loss"
BASE_EXECUTION_HEAD = "a843294780c1392369bfc987e468857c63fc13a5"
SMOKE_PASS_TOKEN = "STAGE45E_L1_H200_SMOKE_PASS_AUTO_CONTINUE_FULL"
FULL_PASS_TOKEN = "STAGE45E_L1_VANDA_H200_TRAIN24_COMPLETE_FOR_PI_REVIEW"
RUNTIME_BINDING_SCHEMA = "VANDA_STAGE45E_L1_H200_RUNTIME_BINDING_V1"
ADMISSION_RESULT_SCHEMA = "VANDA_STAGE45E_L1_H200_RUNTIME_ADMISSION_V1"
DEVICE = "cuda:0"
GPU_ROLE = "HOPPER_H200_SINGLE_GPU"
MIN_VRAM_BYTES = 80 * 1024**3

BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_FILE_COUNT = 85
FULL24_BYTES = 919661
FULL24_SPLIT_LOCK_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
FULL24_MANIFEST_SHA256 = "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
FULL24_STATS_SCHEMA = "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2"
FULL24_STATS_PROTOCOL = "RUNTIME_PARQUET_PROVENANCE_V2"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_ORIGIN_TREE = "43617c11aa6c4ea1be47594882d797255c1a575b"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
GROOT_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA256 = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_PROCESSOR_STATISTICS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
DECODER_SOURCE_SHA256 = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae8"
METRICS_SOURCE_SHA256 = "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4"
FFMPEG_RUNTIME_ID = "9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"
FFMPEG_RUNTIME_MANIFEST_SHA256 = "6160d579b98dc5ee4b114ada4a4e00e88ee13df6ed1fc3d308b5879d67027d2c"
VIDEO_QUALIFICATION_SHA256 = "20e197a868fb289643bd2594c655d37173e636fe39a256d09fad9f1f4b0cc03f"
RELEASE_SHA256 = "8306b76f327b260310a22a5d22551ebf299366f8fea4124bcbfbdab87e807666"
DATASET_ID = "FULL24_EXISTING_TRAIN"

CHECKPOINT_ORDER = ("FULL24@2000", "L1@1000", "L1@2000")
EVAL_WINDOW_COUNT = 24
EVAL_SEED = 4242
EVAL_T_VALUES = (0.0, 0.25, 0.50, 0.75)
EVAL_T_BUCKETS = (0, 250, 500, 750)
EVAL_DT = 0.25
EVAL_ACTION_HEAD_CALLS = len(CHECKPOINT_ORDER) * EVAL_WINDOW_COUNT * len(EVAL_T_VALUES)
EVAL_CONDITIONING_PASSES = len(CHECKPOINT_ORDER) * EVAL_WINDOW_COUNT
EVAL_TOTAL_MODEL_SIDE_CALLS = EVAL_ACTION_HEAD_CALLS + EVAL_CONDITIONING_PASSES
EVAL_SAMPLER_CONTRACT = {
    "name": "train24-e4-carried-euler-direct-action-head",
    "seed": EVAL_SEED,
    "t_values": list(EVAL_T_VALUES),
    "t_buckets": list(EVAL_T_BUCKETS),
    "dt": EVAL_DT,
    "carried_state": True,
    "reset_between_stages": False,
    "public_sampler_calls": 0,
    "get_action_calls": 0,
    "target_xyz_in_model_input": False,
}
EVAL_CALL_BUDGET = {
    "checkpoint_order": list(CHECKPOINT_ORDER),
    "windows_per_checkpoint": EVAL_WINDOW_COUNT,
    "action_head_calls": EVAL_ACTION_HEAD_CALLS,
    "conditioning_passes": EVAL_CONDITIONING_PASSES,
    "total_model_side_calls": EVAL_TOTAL_MODEL_SIDE_CALLS,
    "adaptive_extension": False,
}
TRAINING_RECIPE = {
    "base_mode": "FRESH_BASE_NO_RESUME",
    "seed": 2222,
    "max_steps": 2000,
    "save_steps": [1000, 2000],
    "global_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": "adamw_torch",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "warmup_ratio": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "fp16": False,
    "tf32": True,
    "num_gpus": 1,
    "dataloader_num_workers": 0,
    "eval_strategy": "no",
    "save_only_model": False,
    "resume_from_checkpoint": False,
    "trainable_modules": ["projector", "diffusion_model"],
    "frozen_modules": ["visual", "llm", "vlln", "vl_self_attention"],
}
L1_CONTRACT = {
    "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
    "normalization_unchanged": True,
    "stats_sha256": FULL24_STATS_SHA256,
    "mask_schema": "T16_BINARY_UNCHANGED",
    "mask_shape": [40, 132],
    "active_scalar_count": 190,
    "xyz_rows": [0, 14],
    "xyz_channels": [0, 1, 2],
    "xyz_loss_weights": {"x": 1.401885, "y": 0.472436, "z": 1.125678},
    "other_active_weight": 1.0,
    "inactive_weight": 1.0,
    "weight_matrix_sha256": "b844c1c9a0400ad25ed55d43eb9d5cfe932435bb52340539038e7f4d6ace93a9",
    "formula": "sum((prediction-target)^2 * binary_action_mask * l1_loss_weight) / binary_action_mask.sum()",
    "mask_folding": False,
    "adaptive_renormalization": False,
}
LOCKED_BOUNDARIES = {
    "FULL24_normalization_and_stats": "LOCKED_UNCHANGED",
    "T16_binary_mask_and_190_active": "LOCKED_UNCHANGED",
    "decoder": "LOCKED_UNCHANGED",
    "model_architecture": "LOCKED_UNCHANGED",
    "dataset": "FULL24_TRAIN_ONLY_LOCKED",
    "new_data": "LOCKED",
    "N1": "LOCKED",
    "N2": "LOCKED",
    "N2_plus_L1": "LOCKED",
    "D2B": "LOCKED",
    "FULL24@4000": "LOCKED",
    "Validation-E": "LOCKED",
    "Holdout-F": "LOCKED",
    "WBC": "LOCKED",
    "simulator": "LOCKED",
    "hardware": "LOCKED",
    "merge": "LOCKED",
}
SOURCE_IDENTITIES = {
    "base_tree_sha256": BASE_TREE_SHA256,
    "full24_tree_sha256": FULL24_TREE_SHA256,
    "full24_stats_sha256": FULL24_STATS_SHA256,
    "full24_stats_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
    "full24_stats_schema": FULL24_STATS_SCHEMA,
    "full24_stats_protocol": FULL24_STATS_PROTOCOL,
    "groot_commit": GROOT_COMMIT,
    "groot_origin_tree": GROOT_ORIGIN_TREE,
    "groot_payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA256,
    "groot_manifest_sha256": GROOT_MANIFEST_SHA256,
    "groot_manifest_payload_sha256": GROOT_MANIFEST_PAYLOAD_SHA256,
    "cosmos_revision": COSMOS_REVISION,
    "cosmos_tree_sha256": COSMOS_TREE_SHA256,
    "modality_sha256": MODALITY_SHA256,
    "processor_config_sha256": PROCESSOR_CONFIG_SHA256,
    "base_processor_statistics_sha256": BASE_PROCESSOR_STATISTICS_SHA256,
    "decoder_source_sha256": DECODER_SOURCE_SHA256,
    "metrics_source_sha256": METRICS_SOURCE_SHA256,
    "ffmpeg_runtime_id": FFMPEG_RUNTIME_ID,
    "ffmpeg_runtime_manifest_sha256": FFMPEG_RUNTIME_MANIFEST_SHA256,
    "video_qualification_sha256": VIDEO_QUALIFICATION_SHA256,
    "release_sha256": RELEASE_SHA256,
}
DATASET_IDENTITY = {
    "id": DATASET_ID,
    "split": "TRAIN_ONLY",
    "window_count": 24,
    "frame_count": 960,
    "regular_file_count": FULL24_FILE_COUNT,
    "total_bytes": FULL24_BYTES,
    "tree_sha256": FULL24_TREE_SHA256,
    "split_lock_sha256": FULL24_SPLIT_LOCK_SHA256,
    "manifest_sha256": FULL24_MANIFEST_SHA256,
    "window_order": "target-major then snapshot-major",
    "target_xyz_in_model_input": False,
    "validation_e_read": False,
    "holdout_f_read": False,
}
FORBIDDEN_IMPORT_PREFIXES = ("torch", "transformers", "gr00t", "cuda")


class L1AdmissionError(RuntimeError):
    """Raised when the L1 runtime cannot be admitted exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L1AdmissionError(message)


def _exact(actual: Any, expected: Any, field: str) -> None:
    _require(actual == expected, f"{field} differs from frozen PI0123 contract")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    modules = sys.modules if imported_modules is None else imported_modules
    return sorted(
        name for name in modules
        if isinstance(name, str)
        and any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    )


def validate_static_contract() -> dict[str, Any]:
    """Validate the committed L1 contract without touching runtime inputs."""

    _exact(len(CHECKPOINT_ORDER), 3, "evaluation cohort count")
    _exact(EVAL_ACTION_HEAD_CALLS, 288, "evaluation action-head budget")
    _exact(EVAL_CONDITIONING_PASSES, 72, "evaluation conditioning budget")
    _exact(EVAL_TOTAL_MODEL_SIDE_CALLS, 360, "evaluation total budget")
    _exact(L1_CONTRACT["active_scalar_count"], 190, "L1 active scalar count")
    _exact(L1_CONTRACT["stats_sha256"], FULL24_STATS_SHA256, "L1 stats identity")
    _exact(EVAL_SAMPLER_CONTRACT["t_values"], [0.0, 0.25, 0.5, 0.75], "E4 times")
    _exact(EVAL_SAMPLER_CONTRACT["t_buckets"], [0, 250, 500, 750], "E4 timestep buckets")
    _exact(EVAL_SAMPLER_CONTRACT["seed"], 4242, "evaluation seed")
    _exact(TRAINING_RECIPE["seed"], 2222, "training seed")
    _exact(TRAINING_RECIPE["max_steps"], 2000, "training max steps")
    _exact(L1_CONTRACT["mask_folding"], False, "mask folding")
    _exact(L1_CONTRACT["adaptive_renormalization"], False, "adaptive renormalization")
    return {
        "schema": "VANDA_STAGE45E_L1_STATIC_CONTRACT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "authority": {
            "issue_11_comment": PI_GATE_COMMENT,
            "pr_54_comment": PR54_POINTER_COMMENT,
        },
        "branch": BRANCH,
        "base_execution_head": BASE_EXECUTION_HEAD,
        "l1_contract": copy.deepcopy(L1_CONTRACT),
        "training_recipe": copy.deepcopy(TRAINING_RECIPE),
        "evaluation_sampler": copy.deepcopy(EVAL_SAMPLER_CONTRACT),
        "evaluation_call_budget": copy.deepcopy(EVAL_CALL_BUDGET),
        "dataset_identity": copy.deepcopy(DATASET_IDENTITY),
        "source_identities": copy.deepcopy(SOURCE_IDENTITIES),
        "locked_boundaries": copy.deepcopy(LOCKED_BOUNDARIES),
        "return_tokens": {"smoke": SMOKE_PASS_TOKEN, "full": FULL_PASS_TOKEN},
    }


def _fresh_output_root(value: Any) -> Path:
    _require(isinstance(value, str) and value.startswith("/"), "output root must be absolute")
    path = Path(value)
    _require(not path.is_symlink(), "output root may not be a symlink")
    if path.exists():
        _require(path.is_dir() and not any(path.iterdir()), "output root must be fresh and empty")
    return path.resolve()


def validate_pre_model_binding(
    binding: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    authority_token: str | None,
    phase: str,
    imported_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a materialized binding before any model-side import."""

    found = model_imports_present(imported_modules)
    _require(not found, "model-side import already present before admission: " + ",".join(found))
    _exact(binding.get("schema"), RUNTIME_BINDING_SCHEMA, "binding schema")
    _exact(binding.get("stage"), STAGE, "binding stage")
    _exact(binding.get("pi_gate"), PI_GATE_COMMENT, "PI gate comment")
    _exact(binding.get("pi_authority_token"), AUTHORITY_TOKEN, "PI authority token")
    _exact(authority_token, AUTHORITY_TOKEN, "launcher authority token")
    _exact(binding.get("phase"), phase, "admission phase")
    _exact(binding.get("execution_authorized"), True, "execution authorization")
    _exact(binding.get("base_execution_head"), BASE_EXECUTION_HEAD, "base execution HEAD")

    checkout = _mapping(binding.get("execution_checkout"), "binding.execution_checkout")
    observed_checkout = _mapping(observed.get("execution_checkout"), "observed.execution_checkout")
    _exact(checkout.get("authorized_branch"), BRANCH, "authorized branch")
    _exact(checkout.get("authorized_head"), observed_checkout.get("observed_head"), "execution HEAD")
    _exact(checkout.get("authorized_branch"), observed_checkout.get("observed_branch"), "observed branch")
    _exact(checkout.get("authorized_repo_root"), observed_checkout.get("observed_repo_root"), "repo root")
    _exact(checkout.get("authorized_clean_worktree"), True, "authorized clean worktree")
    _exact(observed_checkout.get("observed_clean_worktree"), True, "observed clean worktree")
    head = checkout.get("authorized_head")
    _require(isinstance(head, str) and len(head) == 40 and head != BASE_EXECUTION_HEAD, "execution HEAD is not a committed L1 change")

    _exact(binding.get("source_identities"), SOURCE_IDENTITIES, "source identities")
    _exact(observed.get("source_identities"), SOURCE_IDENTITIES, "observed source identities")
    _exact(binding.get("dataset_identity"), DATASET_IDENTITY, "dataset identity")
    _exact(observed.get("dataset_identity"), DATASET_IDENTITY, "observed dataset identity")
    _exact(binding.get("l1_contract"), L1_CONTRACT, "L1 contract")
    _exact(observed.get("l1_contract"), L1_CONTRACT, "observed L1 contract")
    _exact(binding.get("training_recipe"), TRAINING_RECIPE, "training recipe")
    _exact(observed.get("training_recipe"), TRAINING_RECIPE, "observed training recipe")
    _exact(binding.get("locked_boundaries"), LOCKED_BOUNDARIES, "locked boundaries")
    _exact(observed.get("locked_boundaries"), LOCKED_BOUNDARIES, "observed locked boundaries")
    if phase == "EVAL":
        _exact(binding.get("evaluation_sampler"), EVAL_SAMPLER_CONTRACT, "evaluation sampler")
        _exact(observed.get("evaluation_sampler"), EVAL_SAMPLER_CONTRACT, "observed evaluation sampler")
        _exact(binding.get("evaluation_call_budget"), EVAL_CALL_BUDGET, "evaluation call budget")
        _exact(observed.get("evaluation_call_budget"), EVAL_CALL_BUDGET, "observed evaluation call budget")

    _exact(observed.get("runtime_identity"), binding.get("runtime_identity"), "runtime identity")
    _exact(observed.get("host_gpu_runtime"), binding.get("host_gpu_runtime"), "host/GPU runtime")
    zero = _mapping(binding.get("zero_counters"), "binding.zero_counters")
    _exact(zero, {"model_construction": 0, "model_forward": 0, "backward": 0, "optimizer_steps": 0, "training_steps": 0, "gpu": 1, "pbs": 1}, "pre-model zero counters")
    _exact(observed.get("zero_counters"), zero, "observed zero counters")
    _exact(observed.get("model_imported"), False, "model imported before admission")
    _exact(observed.get("model_construction"), False, "model constructed before admission")
    _exact(observed.get("model_forward"), False, "model forwarded before admission")
    _exact(observed.get("offline_mode"), True, "offline mode")
    output = _mapping(binding.get("output"), "binding.output")
    observed_output = _mapping(observed.get("output"), "observed.output")
    _exact(observed_output.get("root"), output.get("root"), "output root")
    _exact(observed_output.get("fresh"), True, "fresh output")
    _exact(output.get("fresh_required"), True, "fresh output requirement")
    output_root = _fresh_output_root(output.get("root"))
    return {
        "schema": ADMISSION_RESULT_SCHEMA,
        "status": "PASS_PRE_MODEL",
        "phase": phase,
        "execution_branch": BRANCH,
        "execution_head": head,
        "output_root": str(output_root),
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "counters": dict(zero),
    }


def canonical_binding_sha256(binding: Mapping[str, Any]) -> str:
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ADMISSION_RESULT_SCHEMA",
    "AUTHORITY_TOKEN",
    "BASE_EXECUTION_HEAD",
    "BRANCH",
    "CHECKPOINT_ORDER",
    "DATASET_IDENTITY",
    "EVAL_CALL_BUDGET",
    "EVAL_SAMPLER_CONTRACT",
    "FULL_PASS_TOKEN",
    "L1AdmissionError",
    "L1_CONTRACT",
    "LOCKED_BOUNDARIES",
    "PI_GATE_COMMENT",
    "PR54_POINTER_COMMENT",
    "SMOKE_PASS_TOKEN",
    "SOURCE_IDENTITIES",
    "TRAINING_RECIPE",
    "canonical_binding_sha256",
    "model_imports_present",
    "validate_pre_model_binding",
    "validate_static_contract",
]
