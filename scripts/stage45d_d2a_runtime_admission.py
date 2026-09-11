#!/usr/bin/env python3
"""CPU-only admission contract for the Stage45D-D2A runtime.

The D2A experiment is intentionally a new contract layered on the accepted
D1 artifact identities.  This module contains no Torch, GR00T, CUDA, PBS, or
model import.  A materialized binding must prove the observed checkout,
runtime, artifact namespace, exact two-path call budget, and the fact that
the free-running state is carried between Euler stages.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import stage45d_d1_runtime_admission as d1


STAGE = "Stage45D-D2A"
PI_GATE_COMMENT = 5631604048
PR51_POINTER_COMMENT = 5631608714
ISSUE50_POINTER_COMMENT = 5631610825
BRANCH = "stage45d/vanda-h200-d2a-path-divergence"
BASE_EXECUTION_HEAD = "00fe100722268da3134bc7bfe2b535fa86475f7d"
AUTHORITY_TOKEN = "PI_GATE_STATE_0119"
SMOKE_PASS_TOKEN = "STAGE45D_D2A_H200_SMOKE_PASS_AUTO_CONTINUE_FULL"
FULL_PASS_TOKEN = "STAGE45D_D2A_VANDA_H200_FULL_EXECUTION_COMPLETE_FOR_PI_REVIEW"
RUNTIME_BINDING_SCHEMA = "CORELAB_STAGE45D_D2A_RUNTIME_BINDING_V1"
ADMISSION_RESULT_SCHEMA = "CORELAB_STAGE45D_D2A_RUNTIME_ADMISSION_V1"
WINDOW_ORDER = "target-major then snapshot-major"
CHECKPOINT_ORDER = tuple(d1.CHECKPOINT_ORDER)
WINDOW_COUNT = 24
SEED = 4242
T_VALUES = (0.0, 0.25, 0.50, 0.75)
T_BUCKETS = (0, 250, 500, 750)
DT = 0.25
PATH_ORDER = ("teacher_forced", "free_running")
SMOKE_WINDOW_COUNT = 1
SMOKE_ACTION_HEAD_CALLS = SMOKE_WINDOW_COUNT * len(T_VALUES) * len(PATH_ORDER)
SMOKE_CONDITIONING_PASSES = 1
FULL_ACTION_HEAD_CALLS = len(CHECKPOINT_ORDER) * WINDOW_COUNT * len(T_VALUES) * len(PATH_ORDER)
FULL_CONDITIONING_PASSES = len(CHECKPOINT_ORDER) * WINDOW_COUNT
FULL_TOTAL_MODEL_SIDE_CALLS = FULL_ACTION_HEAD_CALLS + FULL_CONDITIONING_PASSES
CALL_BUDGET = {
    "checkpoint_order": list(CHECKPOINT_ORDER),
    "windows_per_checkpoint": WINDOW_COUNT,
    "noise_seed": SEED,
    "t_values": list(T_VALUES),
    "t_buckets": list(T_BUCKETS),
    "dt": DT,
    "path_order": list(PATH_ORDER),
    "action_head_calls": FULL_ACTION_HEAD_CALLS,
    "conditioning_passes": FULL_CONDITIONING_PASSES,
    "total_model_side_calls": FULL_TOTAL_MODEL_SIDE_CALLS,
    "smoke_action_head_calls": SMOKE_ACTION_HEAD_CALLS,
    "smoke_conditioning_passes": SMOKE_CONDITIONING_PASSES,
    "smoke_total_model_side_calls": SMOKE_ACTION_HEAD_CALLS + SMOKE_CONDITIONING_PASSES,
    "adaptive_extension": False,
}
SAMPLER_CONTRACT = {
    "name": "matched-teacher-bridge-vs-free-running-four-stage-Euler",
    "noise_seed": SEED,
    "t_values": list(T_VALUES),
    "t_buckets": list(T_BUCKETS),
    "dt": DT,
    "path_order": list(PATH_ORDER),
    "free_running_reset": False,
    "t_zero_calls_independent": True,
    "t_zero_deterministic_equality_required": True,
    "public_sampler_calls": 0,
    "get_action_calls": 0,
}
SOURCE_IDENTITIES = copy.deepcopy(d1.SOURCE_IDENTITIES)
CHECKPOINT_IDENTITIES = copy.deepcopy(d1.CHECKPOINT_IDENTITIES)
STATS_ROUTES = copy.deepcopy(d1.STATS_ROUTES)
DATASET_IDENTITY = copy.deepcopy(d1.DATASET_IDENTITY)
LOCKED_BOUNDARIES = {
    **copy.deepcopy(d1.LOCKED_BOUNDARIES),
    "D2B_solver_comparison": "LOCKED",
}
RUNTIME_PATH_KEYS = (
    "base_checkpoint", "structured_1000_checkpoint", "full24_1000_checkpoint",
    "full24_2000_checkpoint", "full24_dataset", "snapshot_root", "groot_source",
    "cosmos_root", "modality_config", "base_processor_config",
    "base_processor_statistics", "structured_1000_statistics",
    "full24_runtime_statistics", "golden_release_manifest",
    "groot_runtime_manifest", "full24_manifest", "checkpoint_integrity_audit",
    "runtime_admission", "output_root",
)
DIRECTORY_PATH_KEYS = {
    "base_checkpoint", "structured_1000_checkpoint", "full24_1000_checkpoint",
    "full24_2000_checkpoint", "full24_dataset", "snapshot_root", "groot_source",
    "cosmos_root",
}
CONTRACT = {
    "stage": STAGE,
    "teacher_bridge": "x_TF(t)=(1-t)*x0+t*x1; v_TF(t)=model(x_TF(t),t)",
    "free_running": "x_FR(0)=x0; v_FR(t_i)=model(x_FR(t_i),t_i); x_FR(t_i+1)=x_FR(t_i)+dt*v_FR(t_i)",
    "free_running_reset": False,
    "state_divergence": "masked active-190 norm of x_FR(t)-x_TF(t), overall and XYZ/rot6d/timing/gripper",
    "field_sensitivity": "masked active-190 norm and cosine of v_FR(t)-v_TF(t), overall and blocks",
    "teacher_bridge_local_error": "v_TF(t)-v_target in NORMALIZED_LOCAL_FLOW namespace",
    "off_path_diagnostic": "v_FR(t)-v_target is descriptive only, not supervised error",
    "t_zero": "two independent labeled action-head calls with exact deterministic equality required",
    "raw_before_derived": True,
    "target_xyz_in_model_input": False,
    "solver_comparison": False,
}


class D2AAdmissionError(RuntimeError):
    """Raised when a materialized D2A runtime is not exactly admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2AAdmissionError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D2AAdmissionError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    """Return model-side modules already present before admission."""

    modules = sys.modules if imported_modules is None else imported_modules
    found: list[str] = []
    for name in modules:
        if not isinstance(name, str):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in d1.FORBIDDEN_IMPORT_PREFIXES):
            found.append(name)
    return sorted(found)


def validate_static_contract() -> dict[str, Any]:
    """Return the static D2A contract without resolving node-local paths."""

    _require(len(CHECKPOINT_ORDER) == 3, "D2A checkpoint cardinality drift")
    _require(WINDOW_COUNT == 24 and len(T_VALUES) == 4 and len(PATH_ORDER) == 2, "D2A grid cardinality drift")
    _require(FULL_ACTION_HEAD_CALLS == 576, "D2A action-head budget drift")
    _require(FULL_CONDITIONING_PASSES == 72 and FULL_TOTAL_MODEL_SIDE_CALLS == 648, "D2A conditioning budget drift")
    _require(SAMPLER_CONTRACT["free_running_reset"] is False, "D2A permits a free-running reset")
    _require(SAMPLER_CONTRACT["t_zero_calls_independent"] is True, "D2A t=0 calls are not independent")
    _require(CONTRACT["target_xyz_in_model_input"] is False, "D2A target XYZ exclusion drift")
    return {
        "schema": "CORELAB_STAGE45D_D2A_STATIC_CONTRACT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "stage": STAGE,
        "authority": {"issue_11_comment": PI_GATE_COMMENT, "pr_51_comment": PR51_POINTER_COMMENT, "issue_50_comment": ISSUE50_POINTER_COMMENT},
        "branch": BRANCH,
        "base_execution_head": BASE_EXECUTION_HEAD,
        "call_budget": copy.deepcopy(CALL_BUDGET),
        "sampler": copy.deepcopy(SAMPLER_CONTRACT),
        "contract": copy.deepcopy(CONTRACT),
        "locked_boundaries": copy.deepcopy(LOCKED_BOUNDARIES),
        "return_tokens": {"smoke": SMOKE_PASS_TOKEN, "full": FULL_PASS_TOKEN},
    }


def _exact(actual: Any, expected: Any, field: str) -> None:
    _require(actual == expected, f"{field} differs from the frozen D2A contract")


def _fresh_output_root(value: Any) -> Path:
    _require(isinstance(value, str) and value, "D2A output root is absent")
    path = Path(value).expanduser()
    _require(path.is_absolute() and not path.is_symlink(), "D2A output root must be an absolute non-symlink path")
    if path.exists():
        _require(path.is_dir() and not any(path.iterdir()), "D2A output root must be fresh and empty")
    return path.resolve()


def _validate_paths(binding: Mapping[str, Any]) -> dict[str, Path]:
    paths = _mapping(binding.get("paths"), field="binding.paths")
    _exact(sorted(paths), sorted(RUNTIME_PATH_KEYS), "runtime path key set")
    result: dict[str, Path] = {}
    for key in RUNTIME_PATH_KEYS:
        value = paths.get(key)
        _require(isinstance(value, str) and value.startswith("/"), f"runtime path is not absolute: {key}")
        _require("validation-e" not in value.lower() and "holdout-f" not in value.lower(), f"locked data path: {key}")
        candidate = Path(value).expanduser()
        if key == "output_root":
            _fresh_output_root(value)
        elif key in DIRECTORY_PATH_KEYS:
            _require(candidate.is_dir() and not candidate.is_symlink(), f"runtime directory unavailable: {key}")
        else:
            _require(candidate.is_file() and not candidate.is_symlink(), f"runtime file unavailable: {key}")
        result[key] = candidate
    return result


def _validate_execution_control(binding: Mapping[str, Any], observed: Mapping[str, Any], phase: str) -> None:
    control = _mapping(binding.get("execution_control"), field="binding.execution_control")
    _exact(control.get("phase"), phase, "execution phase")
    _require(type(control.get("repair_cycle")) is int and 0 <= control["repair_cycle"] <= 2, "D2A repair cycle out of range")
    _require(type(control.get("smoke_attempt")) is int and 1 <= control["smoke_attempt"] <= 6, "D2A smoke attempt out of range")
    _exact(control.get("automatic_retry"), False, "automatic retry")
    if phase == "FULL":
        smoke_path = Path(str(control.get("smoke_seal_path", ""))).expanduser()
        smoke_sha = control.get("smoke_seal_sha256")
        _require(smoke_path.is_file() and isinstance(smoke_sha, str) and len(smoke_sha) == 64, "full D2A smoke seal binding absent")
        import hashlib

        _exact(hashlib.sha256(smoke_path.read_bytes()).hexdigest(), smoke_sha, "smoke seal hash")
        smoke = _read_json(smoke_path)
        _exact(smoke.get("status"), "PASS", "smoke seal status")
        _exact(smoke.get("token"), SMOKE_PASS_TOKEN, "smoke seal token")
        _exact(smoke.get("execution_head"), binding["execution_checkout"]["authorized_head"], "smoke seal HEAD")
    _exact(observed.get("execution_control"), dict(control), "observed execution control")


def validate_future_admission(
    binding: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    authority_token: str | None,
    phase: str,
    imported_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a materialized D2A binding before any model-side import."""

    found = model_imports_present(imported_modules)
    _require(not found, "model-side import already present before D2A admission: " + ",".join(found))
    _exact(binding.get("schema"), RUNTIME_BINDING_SCHEMA, "runtime binding schema")
    _exact(binding.get("stage"), STAGE, "runtime binding stage")
    _exact(binding.get("pi_gate"), PI_GATE_COMMENT, "PI gate")
    _exact(binding.get("execution_authorized"), True, "execution authorization")
    _exact(binding.get("pi_authority_token"), AUTHORITY_TOKEN, "authority token")
    _exact(authority_token, AUTHORITY_TOKEN, "launcher authority token")
    _exact(binding.get("base_execution_head"), BASE_EXECUTION_HEAD, "D2A base execution HEAD")

    checkout = _mapping(binding.get("execution_checkout"), field="binding.execution_checkout")
    observed_checkout = _mapping(observed.get("execution_checkout"), field="observed.execution_checkout")
    _exact(checkout.get("binding_mode"), "OBSERVED_RUNTIME_ONLY", "checkout binding mode")
    for key in ("authorized_repo_root", "authorized_branch", "authorized_head", "authorized_clean_worktree"):
        _require(checkout.get(key) not in (None, ""), f"missing authorized checkout field: {key}")
    for key in ("observed_repo_root", "observed_branch", "observed_head", "observed_clean_worktree"):
        _require(observed_checkout.get(key) not in (None, ""), f"missing observed checkout field: {key}")
    _exact(checkout.get("authorized_branch"), BRANCH, "execution branch")
    _exact(observed_checkout.get("observed_repo_root"), checkout.get("authorized_repo_root"), "execution repo root")
    _exact(observed_checkout.get("observed_branch"), checkout.get("authorized_branch"), "execution branch observation")
    _exact(observed_checkout.get("observed_head"), checkout.get("authorized_head"), "execution HEAD observation")
    _exact(observed_checkout.get("observed_clean_worktree"), True, "execution worktree cleanliness")
    _exact(checkout.get("authorized_clean_worktree"), True, "authorized worktree cleanliness")
    _require(isinstance(checkout.get("authorized_head"), str) and len(checkout["authorized_head"]) == 40, "execution HEAD is not a full SHA")
    _require(checkout["authorized_head"] != BASE_EXECUTION_HEAD, "D2A execution must contain a committed D2A change")

    _exact(binding.get("source_identities"), SOURCE_IDENTITIES, "source identities")
    _exact(observed.get("source_identities"), SOURCE_IDENTITIES, "observed source identities")
    _exact(binding.get("checkpoint_identities"), CHECKPOINT_IDENTITIES, "checkpoint identities")
    _exact(observed.get("checkpoint_identities"), CHECKPOINT_IDENTITIES, "observed checkpoint identities")
    _exact(binding.get("statistics_routes"), STATS_ROUTES, "statistics routes")
    _exact(observed.get("statistics_routes"), STATS_ROUTES, "observed statistics routes")
    _exact(binding.get("dataset_identity"), DATASET_IDENTITY, "dataset identity")
    _exact(observed.get("dataset_identity"), DATASET_IDENTITY, "observed dataset identity")
    _exact(binding.get("sampler"), SAMPLER_CONTRACT, "sampler contract")
    _exact(observed.get("sampler"), SAMPLER_CONTRACT, "observed sampler contract")
    _exact(binding.get("call_budget"), CALL_BUDGET, "call budget")
    _exact(observed.get("call_budget"), CALL_BUDGET, "observed call budget")
    _exact(binding.get("contract"), CONTRACT, "D2A science contract")
    _exact(observed.get("contract"), CONTRACT, "observed D2A science contract")
    _exact(binding.get("locked_boundaries"), LOCKED_BOUNDARIES, "locked boundaries")
    _exact(observed.get("locked_boundaries"), LOCKED_BOUNDARIES, "observed locked boundaries")
    _exact(observed.get("target_xyz_in_model_input"), False, "target XYZ exclusion")
    _exact(observed.get("runtime_identity"), binding.get("runtime_identity"), "runtime identity")
    _exact(observed.get("host_gpu_runtime"), binding.get("host_gpu_runtime"), "observed H200 runtime")
    _exact(observed.get("zero_counters"), binding.get("zero_counters"), "observed zero counters")
    _exact(binding.get("zero_counters"), {"model_construction": 0, "model_forward": 0, "gpu": 0, "pbs": 1, "training_steps": 0, "action_head_calls": 0, "conditioning_passes": 0}, "pre-model zero counters")
    _validate_execution_control(binding, observed, phase)

    boundary = _mapping(binding.get("execution_boundary"), field="binding.execution_boundary")
    observed_boundary = _mapping(observed.get("execution_boundary"), field="observed.execution_boundary")
    _exact(observed_boundary, boundary, "observed execution boundary")
    for key, value in {
        "training": False, "backward": False, "optimizer_steps": 0,
        "checkpoint_write": 0, "public_sampler_calls": 0, "get_action_calls": 0,
        "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0,
        "hardware": 0, "D2B_solver_comparison": 0,
    }.items():
        _exact(boundary.get(key), value, f"locked execution boundary {key}")
    _exact(boundary.get("h200"), 1, "H200 execution boundary")
    _exact(boundary.get("pbs"), 1, "PBS execution boundary")
    _exact(boundary.get("model_construction"), "RUNTIME_ONLY", "model construction boundary")
    _exact(boundary.get("model_forward"), "RUNTIME_ONLY", "model forward boundary")
    _exact(boundary.get("gpu"), "RUNTIME_ONLY", "GPU boundary")

    output = _mapping(binding.get("output"), field="binding.output")
    observed_output = _mapping(observed.get("output"), field="observed.output")
    _exact(output.get("fresh_required"), True, "fresh output requirement")
    _exact(observed_output.get("root"), output.get("root"), "output root binding")
    _exact(observed_output.get("fresh"), True, "fresh output observation")
    output_root = _fresh_output_root(observed_output.get("root"))
    _exact(observed.get("model_imported"), False, "pre-admission model import")
    _exact(observed.get("model_construction"), False, "pre-admission model construction")
    _exact(observed.get("model_forward"), False, "pre-admission model forward")
    _exact(observed.get("offline_mode"), True, "offline mode")
    paths = _validate_paths(binding)
    _exact(str(paths["output_root"].resolve()), str(output_root), "bound output path")
    return {
        "schema": ADMISSION_RESULT_SCHEMA,
        "status": "PASS_PRE_MODEL",
        "stage": STAGE,
        "phase": phase,
        "execution_branch": checkout["authorized_branch"],
        "execution_head": checkout["authorized_head"],
        "execution_worktree_clean": True,
        "output_root": str(output_root),
        "admission_order": [
            "pre_model_import_guard", "authority_and_gate_binding", "observed_checkout_binding",
            "observed_h200_runtime_binding", "D1_artifact_identity_binding",
            "D2A_two_path_contract_binding", "call_budget_and_fresh_output_binding",
            "locked_boundary_binding",
        ],
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "counters": dict(binding["zero_counters"]),
    }


__all__ = [
    "AUTHORITY_TOKEN", "BASE_EXECUTION_HEAD", "BRANCH", "CALL_BUDGET",
    "CHECKPOINT_IDENTITIES", "CHECKPOINT_ORDER", "CONTRACT", "DATASET_IDENTITY",
    "D2AAdmissionError", "DT", "FULL_ACTION_HEAD_CALLS", "FULL_CONDITIONING_PASSES",
    "FULL_PASS_TOKEN", "FULL_TOTAL_MODEL_SIDE_CALLS", "LOCKED_BOUNDARIES", "PATH_ORDER",
    "PI_GATE_COMMENT", "RUNTIME_PATH_KEYS", "SAMPLER_CONTRACT", "SEED",
    "SMOKE_ACTION_HEAD_CALLS", "SMOKE_CONDITIONING_PASSES", "SMOKE_PASS_TOKEN",
    "SOURCE_IDENTITIES", "STATS_ROUTES", "T_BUCKETS", "T_VALUES", "WINDOW_COUNT",
    "WINDOW_ORDER", "model_imports_present", "validate_future_admission", "validate_static_contract",
]
