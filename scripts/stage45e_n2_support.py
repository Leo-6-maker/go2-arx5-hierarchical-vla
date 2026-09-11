#!/usr/bin/env python3
"""CPU-only support contracts for the Stage45E N2 Vanda track.

This module is deliberately an admission/evaluation surface, not a science
implementation.  It validates a future Corelab-produced N2 statistics artifact
against the accepted FULL24 statistics, and builds fail-closed evaluation or
fallback plans.  It never imports torch/GR00T/Cosmos, constructs a model, runs a
forward/backward pass, submits PBS, or starts training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PI_GATE = "PI_GATE_STATE_0121"
PI_GATE_COMMENT = 5633708741
PR52_POINTER_COMMENT = 5633715464
ISSUE50_POINTER_COMMENT = 5633717912
BASE_HEAD = "a843294780c1392369bfc987e468857c63fc13a5"
CORELAB_BRANCH = "stage45e/n2-balanced-xyz-normalization"
VANDA_BRANCH = "stage45e/vanda-n2-eval-surface"

FULL_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
DATASET_SPLIT_LOCK_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
DESCRIPTOR_CLOSURE_SHA256 = "ee46808ad2ebf02dd488721dba88831b4543c5ae6a51d39d2ef65f80081073b5"
LEGACY_FULL24_2000_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"

ACTION_HORIZON = 40
ACTION_DIM = 132
ACTIVE_SCALARS = 190
COMMON_HALF_RANGE_M = 0.10734947131343872
XYZ_COLS = (0, 1, 2)
XYZ_AXES = ("x", "y", "z")

DEFAULT_FULL_STATS = Path(
    "/scratch/e1192548/stage45b_pi0083_runtime_parquet_v2/"
    "STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json"
)
DEFAULT_DATASET = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/lerobot_train"
)

FORBIDDEN_IMPORT_PREFIXES = ("torch", "transformers", "gr00t", "cosmos", "cuda")
FORBIDDEN_EXECUTION_FIELDS = {
    "model_execution": 0,
    "model_construction": 0,
    "model_forward": 0,
    "conditioning": 0,
    "backward": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "pbs": 0,
    "qsub": 0,
    "gpu": 0,
    "validation_e_rows_read": 0,
    "holdout_f_rows_read": 0,
    "wbc_steps": 0,
    "simulator_steps": 0,
    "hardware_steps": 0,
}

DATASET_IDENTITY = {
    "id": "FULL24_EXISTING_TRAIN",
    "split": "TRAIN_ONLY",
    "window_count": 24,
    "frame_count": 960,
    "window_order": "target-major then snapshot-major",
    "tree_sha256": DATASET_TREE_SHA256,
    "split_lock_sha256": DATASET_SPLIT_LOCK_SHA256,
    "descriptor_closure_sha256": DESCRIPTOR_CLOSURE_SHA256,
    "target_xyz_in_model_input": False,
}

LEGACY_CHECKPOINT_IDENTITY = {
    "label": "FULL24@2000",
    "tree_sha256": LEGACY_FULL24_2000_TREE_SHA256,
    "stats_sha256": FULL_STATS_SHA256,
    "stats_payload_sha256": FULL_STATS_PAYLOAD_SHA256,
    "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
}

EVALUATION_ROLES = ("FULL24@2000", "N2@1000", "N2@2000")
EVALUATION_METRIC_NAMESPACES = (
    "NORMALIZED_LOCAL_FLOW",
    "DECODED_PHYSICAL_ACTION",
    "PAIRED_DESCRIPTIVE",
)


class N2ContractError(RuntimeError):
    """Raised when a future N2 artifact or runtime binding is not admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise N2ContractError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise N2ContractError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise N2ContractError(f"non-finite JSON value: {value!r}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    modules = sys.modules if imported_modules is None else imported_modules
    return sorted(
        name
        for name in modules
        if isinstance(name, str)
        and any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    )


def _array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == shape, f"{field} shape {result.shape} != {shape}")
    _require(np.isfinite(result).all(), f"{field} contains non-finite values")
    return result


def _strict_json_numeric_array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    """Read a JSON numeric array without bool/string coercion."""

    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise N2ContractError(f"{field} is not a rectangular JSON array") from exc
    _require(raw.shape == shape, f"{field} shape {raw.shape} != {shape}")
    for item in raw.reshape(-1).tolist():
        _require(type(item) in (int, float), f"{field} contains non-JSON-number value: {item!r}")
        _require(math.isfinite(float(item)), f"{field} contains non-finite value: {item!r}")
    return np.asarray(value, dtype=np.float64)


def _exact_array(actual: Any, expected: np.ndarray, field: str) -> None:
    observed = _strict_json_numeric_array(actual, expected.shape, field)
    _require(np.array_equal(observed, expected), f"{field} is not exact-equal")


def load_accepted_full_stats(path: Path = DEFAULT_FULL_STATS) -> dict[str, Any]:
    payload = _read_json(path)
    _require(sha256_file(path) == FULL_STATS_SHA256, "accepted FULL24 stats file SHA256 mismatch")
    _require(payload.get("statistics_payload_sha256") == FULL_STATS_PAYLOAD_SHA256, "accepted FULL24 stats payload SHA mismatch")
    stats = _mapping(payload.get("statistics"), "accepted_full.statistics")
    lower = _array(stats.get("selected_lower"), (ACTION_DIM,), "accepted_full.selected_lower")
    upper = _array(stats.get("selected_upper"), (ACTION_DIM,), "accepted_full.selected_upper")
    active_mask = _array(stats.get("active_mask"), (ACTION_HORIZON, ACTION_DIM), "accepted_full.active_mask")
    _require(np.array_equal(active_mask, (active_mask > 0.0).astype(np.float64)), "accepted FULL mask is not binary")
    _require(int(np.count_nonzero(active_mask)) == ACTIVE_SCALARS, "accepted FULL mask active count drift")
    _require(stats.get("label_count") == 24, "accepted FULL label count drift")
    _require(stats.get("normalization") == "MINMAX_TO_NEG1_POS1", "accepted FULL normalization drift")
    _require(stats.get("selected_lower_bound") == "q01" and stats.get("selected_upper_bound") == "q99", "accepted FULL quantile bounds drift")
    _require(stats.get("timing_mode") == "T16", "accepted FULL timing mode drift")
    return {
        "path": str(path),
        "sha256": FULL_STATS_SHA256,
        "statistics_payload_sha256": FULL_STATS_PAYLOAD_SHA256,
        "payload": payload,
        "statistics": stats,
        "lower": lower,
        "upper": upper,
        "active_mask": active_mask,
    }


def derive_n2_bounds(full: Mapping[str, Any]) -> dict[str, Any]:
    lower = _array(full["lower"], (ACTION_DIM,), "full.lower")
    upper = _array(full["upper"], (ACTION_DIM,), "full.upper")
    active_mask = _array(full["active_mask"], (ACTION_HORIZON, ACTION_DIM), "full.active_mask")
    centers = (lower[:3] + upper[:3]) / 2.0
    half_ranges = (upper[:3] - lower[:3]) / 2.0
    common_half_range = float(np.max(half_ranges))
    _require(common_half_range == COMMON_HALF_RANGE_M, "derived FULL24 common half-range differs from frozen N2 value")
    n2_lower = lower.copy()
    n2_upper = upper.copy()
    n2_lower[:3] = centers - common_half_range
    n2_upper[:3] = centers + common_half_range
    return {
        "centers": centers,
        "half_ranges": half_ranges,
        "common_half_range": common_half_range,
        "lower": n2_lower,
        "upper": n2_upper,
        "active_mask": active_mask.copy(),
    }


def expected_n2_clipping_audit() -> dict[str, Any]:
    """Deterministic counts from the PI0120 FULL24 label population audit."""

    return {
        "label_count": 24,
        "active_scalar_count": 4560,
        "varying_active_scalar_count": 1440,
        "xyz_active_scalar_count": 1080,
        "full_clipped_count": 18,
        "full_clipped_fraction": 18 / 4560,
        "varying_clipped_fraction": 18 / 1440,
        "xyz_clipped_count": 10,
        "xyz_clipped_fraction": 10 / 1080,
        "xyz_axis_clipped_count": {"x": 8, "y": 0, "z": 2},
        "source": "PI0120 independent FULL24 label clipping audit; deterministic descriptive control",
    }


def validate_n2_statistics_artifact(
    path: Path,
    accepted_full_stats_path: Path = DEFAULT_FULL_STATS,
    *,
    expected_science_head: str | None = None,
) -> dict[str, Any]:
    """Validate a Corelab-produced N2 artifact without executing a model."""

    found = model_imports_present()
    _require(not found, "model-side import already present before N2 stats admission: " + ",".join(found))
    payload = _read_json(path)
    _require(payload.get("schema") == "CORELAB_STAGE45E_N2_BALANCED_XYZ_STATISTICS_V1", "N2 stats schema mismatch")
    _require(payload.get("status") in {"STATIC_READY_NOT_EXECUTED", "READY_FOR_TRAINING_NOT_EXECUTED"}, "N2 stats artifact is not static-only")
    _require(payload.get("pi_gate") == PI_GATE, "N2 stats PI gate mismatch")
    full = load_accepted_full_stats(accepted_full_stats_path)
    n2 = derive_n2_bounds(full)
    source = _mapping(payload.get("source_full24_statistics"), "n2.source_full24_statistics")
    _require(source.get("sha256") == FULL_STATS_SHA256, "N2 source FULL stats SHA mismatch")
    _require(source.get("statistics_payload_sha256") == FULL_STATS_PAYLOAD_SHA256, "N2 source FULL stats payload SHA mismatch")
    science_head = payload.get("science_commit")
    if expected_science_head is not None:
        _require(science_head == expected_science_head, "N2 science commit binding mismatch")
    if science_head is not None:
        _require(isinstance(science_head, str) and len(science_head) == 40 and science_head != BASE_HEAD, "N2 science commit is not a non-base full SHA")
    n2_contract = _mapping(payload.get("n2_contract"), "n2_contract")
    _require(n2_contract.get("common_half_range_m") == COMMON_HALF_RANGE_M, "N2 common half-range changed")
    _exact_array(n2_contract.get("xyz_centers_m"), n2["centers"], "N2 xyz centers")
    _exact_array(n2_contract.get("selected_lower"), n2["lower"], "N2 selected lower")
    _exact_array(n2_contract.get("selected_upper"), n2["upper"], "N2 selected upper")
    _exact_array(n2_contract.get("active_mask"), full["active_mask"], "N2 active mask")
    _require(n2_contract.get("normalization") == "MINMAX_TO_NEG1_POS1", "N2 normalization changed")
    _require(n2_contract.get("selected_lower_bound") == "q01" and n2_contract.get("selected_upper_bound") == "q99", "N2 quantile bound changed")
    _require(n2_contract.get("timing_mode") == "T16", "N2 timing mode changed")
    _require(n2_contract.get("active_scalar_count") == ACTIVE_SCALARS, "N2 active scalar count changed")
    _require(n2_contract.get("mask_binary") is True, "N2 mask is not explicitly binary")
    _require(n2_contract.get("loss_weighting") is False, "N2 artifact mixes in loss weighting")
    _require(n2_contract.get("n1_historical_bounds") is False, "N2 artifact mixes in N1 bounds")
    _require(n2_contract.get("l1_physical_loss") is False, "N2 artifact mixes in L1 loss")
    _require(n2_contract.get("non_xyz_bounds_unchanged") is True, "N2 non-XYZ bounds equality is not proven")
    _require(n2_contract.get("centers_source") == "accepted FULL24 q01/q99 midpoint per XYZ axis", "N2 center source is not frozen")
    _require(n2_contract.get("dataset_unchanged") is True, "N2 artifact does not bind unchanged dataset")
    _require(n2_contract.get("label_construction_unchanged") is True, "N2 artifact does not bind unchanged label construction")
    _require(n2_contract.get("decoder_unchanged") is True, "N2 artifact does not bind unchanged decoder")
    _require(n2_contract.get("target_xyz_in_model_input") is False, "N2 target XYZ exclusion changed")
    dataset = _mapping(payload.get("dataset_identity"), "n2.dataset_identity")
    _require(dict(dataset) == DATASET_IDENTITY, "N2 dataset identity changed")
    clipping = _mapping(payload.get("clipping_audit"), "n2.clipping_audit")
    expected_clip = expected_n2_clipping_audit()
    for key in ("label_count", "active_scalar_count", "varying_active_scalar_count", "xyz_active_scalar_count", "full_clipped_count", "xyz_clipped_count"):
        _require(clipping.get(key) == expected_clip[key], f"N2 clipping audit mismatch: {key}")
    _require(clipping.get("xyz_axis_clipped_count") == expected_clip["xyz_axis_clipped_count"], "N2 per-axis clipping audit mismatch")
    execution = _mapping(payload.get("execution_boundary"), "n2.execution_boundary")
    for key, value in FORBIDDEN_EXECUTION_FIELDS.items():
        _require(execution.get(key) == value, f"N2 execution boundary is nonzero: {key}")
    _require(execution.get("model_execution") is False, "N2 artifact claims model execution")
    _require(execution.get("training") is False, "N2 artifact claims training")
    _require(execution.get("validation_e_read") is False and execution.get("holdout_f_read") is False, "N2 artifact reads locked evaluation data")
    return {
        "schema": payload["schema"],
        "status": "PASS_STATIC_N2_STATS",
        "path": str(path),
        "sha256": sha256_file(path),
        "science_commit": science_head,
        "source_full24_statistics_sha256": FULL_STATS_SHA256,
        "common_half_range_m": COMMON_HALF_RANGE_M,
        "xyz_centers_m": n2["centers"].tolist(),
        "selected_lower": n2["lower"].tolist(),
        "selected_upper": n2["upper"].tolist(),
        "active_scalar_count": ACTIVE_SCALARS,
        "execution_boundary": dict(execution),
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def _git_succeeds(repo: Path, *args: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode == 0


def audit_corelab_science_commit(repo: Path, science_head: str) -> dict[str, Any]:
    """Prepare a conservative source audit for the first Corelab N2 commit.

    The function intentionally does not infer science correctness from a diff
    alone.  It checks lineage and forbidden-surface markers, while the N2
    artifact validator checks exact values.  A caller must supply both results
    before treating the future Corelab commit as auditable.
    """

    _require(isinstance(science_head, str) and len(science_head) == 40, "Corelab science head must be a full SHA")
    _require(science_head != BASE_HEAD, "Corelab science audit requires a non-empty commit")
    _require(_git_succeeds(repo, "cat-file", "-e", f"{science_head}^{{commit}}"), "Corelab science commit is unavailable")
    _require(_git_succeeds(repo, "merge-base", "--is-ancestor", BASE_HEAD, science_head), "Corelab commit is not based on accepted D2A HEAD")
    changed = [line for line in _git(repo, "diff", "--name-only", f"{BASE_HEAD}..{science_head}").splitlines() if line]
    _require(changed, "Corelab science commit has no changed files")
    lowered = "\n".join(changed).lower()
    _require("pr52" not in lowered and "stage45d_d2a" not in lowered, "Corelab commit touches frozen D2A history")
    diff_text = _git(repo, "diff", "--unified=0", f"{BASE_HEAD}..{science_head}")
    added_lines = "\n".join(line[1:] for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    lowered_added = added_lines.lower()
    _require("validation-e" not in lowered_added and "holdout-f" not in lowered_added, "Corelab N2 commit adds locked Validation/Holdout surface")
    _require("h200" not in lowered_added and "pbs" not in lowered_added and "qsub" not in lowered_added, "Corelab science commit mixes deployment work")
    _require("0.107349471" in added_lines or "common_half_range" in lowered_added, "Corelab diff has no visible frozen N2 gain anchor")
    return {
        "status": "PASS_STATIC_CORELAB_COMMIT_SURFACE",
        "base_head": BASE_HEAD,
        "science_head": science_head,
        "changed_files": changed,
        "forbidden_added_surface_hits": [],
        "interpretation_boundary": "source diff screening is not a substitute for exact N2 artifact/value audit",
    }


def static_evaluation_template() -> dict[str, Any]:
    return {
        "schema": "CORELAB_STAGE45E_N2_TRAIN24_EVALUATION_SURFACE_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "pi_gate": PI_GATE,
        "authority": {
            "issue_11_comment": PI_GATE_COMMENT,
            "pr_52_comment": PR52_POINTER_COMMENT,
            "issue_50_comment": ISSUE50_POINTER_COMMENT,
        },
        "science_binding": {
            "required_branch": CORELAB_BRANCH,
            "required_base_head": BASE_HEAD,
            "observed_science_head": None,
            "exact_commit_required_before_use": True,
            "audit_function": "audit_corelab_science_commit + validate_n2_statistics_artifact",
        },
        "comparison_roles": list(EVALUATION_ROLES),
        "metric_namespaces": list(EVALUATION_METRIC_NAMESPACES),
        "n2_spec": {
            "source_full24_statistics_sha256": FULL_STATS_SHA256,
            "source_full24_statistics_payload_sha256": FULL_STATS_PAYLOAD_SHA256,
            "common_half_range_m": COMMON_HALF_RANGE_M,
            "centers_source": "accepted FULL24 q01/q99 midpoint per XYZ axis",
            "non_xyz_statistics_unchanged": True,
            "binary_t16_mask_active_scalars": ACTIVE_SCALARS,
        },
        "dataset_identity": dict(DATASET_IDENTITY),
        "legacy_full24_2000": dict(LEGACY_CHECKPOINT_IDENTITY),
        "n2_roles": {
            "N2@1000": {"step": 1000, "stats_route": "STAGE45E_N2_BALANCED_XYZ_RUNTIME_V1", "checkpoint_tree_sha256": None},
            "N2@2000": {"step": 2000, "stats_route": "STAGE45E_N2_BALANCED_XYZ_RUNTIME_V1", "checkpoint_tree_sha256": None},
        },
        "required_evidence_order": [
            "observed-runtime-admission-before-model-import",
            "raw-arrays-sealed-before-derived-metrics",
            "fresh-output-root",
            "paired-descriptive-summary",
            "final-evidence-hash-inventory",
        ],
        "target_xyz_in_model_input": False,
        "locked_boundaries": {
            "training": "LOCKED_FOR_EVALUATION",
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "D2B": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
            "merge": "HOLD",
        },
        "execution_boundary": {
            "model_execution": 0,
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "pbs": 0,
            "gpu": 0,
            "validation_e_read": 0,
            "holdout_f_read": 0,
            "evaluation_authorization_required": True,
        },
        "return_token": "STAGE45E_N2_VANDA_EVAL_SURFACE_READY_FOR_PI_GATE",
    }


def static_fallback_template() -> dict[str, Any]:
    return {
        "schema": "CORELAB_STAGE45E_N2_VANDA_H200_FALLBACK_BINDING_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "pi_gate": PI_GATE,
        "science_binding": {
            "required_branch": CORELAB_BRANCH,
            "required_base_head": BASE_HEAD,
            "science_head": None,
            "n2_stats_sha256": None,
            "n2_common_half_range_m": COMMON_HALF_RANGE_M,
            "n2_source_full24_statistics_sha256": FULL_STATS_SHA256,
            "same_science_lineage_required": True,
        },
        "activation": {
            "required_corelab_token": "STAGE45E_N2_CORELAB_RESOURCE_BLOCKED_H200_FALLBACK_AUTHORIZED",
            "token_must_be_observed_from_corelab": True,
            "automatic_activation_without_token": False,
        },
        "allowed_deployment_changes": [
            "Hopper path/bootstrap",
            "offline runtime resolution",
            "H200 identity/resource admission",
            "fresh output and evidence sealing",
            "PBS wrapper only after blocker token",
        ],
        "forbidden_science_changes": [
            "N2 common half-range or centers",
            "FULL24 dataset/label construction",
            "T16 binary mask",
            "model architecture/trainable modules",
            "optimizer/scheduler/seed/steps/batch/precision",
            "loss weighting",
        ],
        "execution_boundary": dict(FORBIDDEN_EXECUTION_FIELDS),
        "return_token": "STAGE45E_N2_VANDA_H200_2000_TRAINING_COMPLETE_FOR_PI_REVIEW",
    }


def validate_h200_fallback_preflight(
    repo: Path,
    science_head: str,
    n2_stats_path: Path,
    blocker_token_path: Path,
    accepted_full_stats_path: Path = DEFAULT_FULL_STATS,
    *,
    runtime_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate conditional fallback authority without starting H200 work.

    This is deliberately a preflight-only surface.  Even with the exact
    Corelab blocker token it returns a static admission result; a future
    deployment wrapper must add a separate PI-authorized execution stage.
    """

    found = model_imports_present()
    _require(not found, "model-side import already present before H200 fallback admission: " + ",".join(found))
    _require(blocker_token_path.is_file() and not blocker_token_path.is_symlink(), "Corelab blocker token must be a regular file")
    blocker_token = blocker_token_path.read_text(encoding="utf-8").strip()
    _require(blocker_token == "STAGE45E_N2_CORELAB_RESOURCE_BLOCKED_H200_FALLBACK_AUTHORIZED", "H200 fallback token is absent or incorrect")
    commit_audit = audit_corelab_science_commit(repo.resolve(), science_head)
    stats_audit = validate_n2_statistics_artifact(
        n2_stats_path.resolve(),
        accepted_full_stats_path.resolve(),
        expected_science_head=science_head,
    )
    if runtime_profile is not None:
        _require(runtime_profile.get("device") == "cuda:0", "H200 fallback device must be single cuda:0")
        _require(runtime_profile.get("offline_mode") is True, "H200 fallback must be offline")
        _require(runtime_profile.get("single_gpu") is True, "H200 fallback must use one GPU")
    return {
        "schema": "CORELAB_STAGE45E_N2_VANDA_H200_FALLBACK_ADMISSION_V1",
        "status": "PASS_STATIC_FALLBACK_PREMODEL_NOT_EXECUTED",
        "pi_gate": PI_GATE,
        "science_head": science_head,
        "corelab_blocker_token_sha256": sha256_file(blocker_token_path),
        "corelab_commit_audit": commit_audit,
        "n2_stats_audit": stats_audit,
        "runtime_profile": dict(runtime_profile) if runtime_profile is not None else None,
        "admission_order": [
            "pre_model_import_guard",
            "corelab_resource_blocker_token",
            "exact_corelab_science_commit_lineage",
            "exact_n2_stats_artifact",
            "observed_h200_runtime_profile",
        ],
        "execution_authorized": False,
        "execution_boundary": dict(FORBIDDEN_EXECUTION_FIELDS),
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "return_token": "STAGE45E_N2_VANDA_H200_2000_TRAINING_COMPLETE_FOR_PI_REVIEW",
        "next_step": "PI-authorized deployment stage only; this preflight does not launch PBS or model work",
    }


def validate_static_templates(repo: Path) -> dict[str, Any]:
    evaluation = _read_json(repo / "manifests/STAGE45E_N2_TRAIN24_EVALUATION_SURFACE_TEMPLATE_V1.json")
    fallback = _read_json(repo / "manifests/STAGE45E_N2_VANDA_H200_FALLBACK_BINDING_TEMPLATE_V1.json")
    _require(evaluation == static_evaluation_template(), "N2 evaluation template drift")
    _require(fallback == static_fallback_template(), "N2 fallback template drift")
    return {
        "status": "PASS_STATIC_TEMPLATES",
        "evaluation_schema": evaluation["schema"],
        "fallback_schema": fallback["schema"],
        "model_execution": 0,
        "h200": 0,
        "pbs": 0,
    }


def build_static_plan() -> dict[str, Any]:
    return {
        "schema": "STAGE45E_N2_VANDA_STATIC_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "support_branch": VANDA_BRANCH,
        "base_head": BASE_HEAD,
        "corelab_science_commit": None,
        "corelab_commit_audit": "PENDING_NONEMPTY_CORELAB_N2_COMMIT",
        "evaluation_surface": static_evaluation_template(),
        "fallback_surface": static_fallback_template(),
        "execution_boundary": {
            "h200": 0,
            "pbs": 0,
            "model_execution": 0,
            "training": 0,
            "evaluation": 0,
            "validation_e_read": 0,
            "holdout_f_read": 0,
        },
        "return_token": "STAGE45E_N2_VANDA_EVAL_SURFACE_READY_FOR_PI_GATE",
    }


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTIVE_SCALARS",
    "BASE_HEAD",
    "COMMON_HALF_RANGE_M",
    "CORELAB_BRANCH",
    "DATASET_IDENTITY",
    "DEFAULT_FULL_STATS",
    "EVALUATION_ROLES",
    "FORBIDDEN_IMPORT_PREFIXES",
    "LEGACY_CHECKPOINT_IDENTITY",
    "N2ContractError",
    "PI_GATE",
    "VANDA_BRANCH",
    "audit_corelab_science_commit",
    "build_static_plan",
    "canonical_sha256",
    "derive_n2_bounds",
    "expected_n2_clipping_audit",
    "jsonable",
    "load_accepted_full_stats",
    "model_imports_present",
    "sha256_file",
    "static_evaluation_template",
    "static_fallback_template",
    "validate_h200_fallback_preflight",
    "validate_n2_statistics_artifact",
    "validate_static_templates",
    "write_json",
]
