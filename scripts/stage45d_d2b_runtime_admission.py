#!/usr/bin/env python3
"""CPU-only admission contract for the Stage45D-D2B solver study.

This module is deliberately model-free. It binds the observed Hopper
checkout, the accepted D1 artifacts, and the exact forward-Euler resolution
grid before a child process is allowed to import Torch or construct a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import stage45d_d1_runtime_admission as d1


STAGE = "Stage45D-D2B"
PI_GATE_COMMENT = 5633959841
PR52_POINTER_COMMENT = 5633994019
BRANCH = "stage45d/vanda-h200-d2b-solver-discretization"
BASE_EXECUTION_HEAD = "a843294780c1392369bfc987e468857c63fc13a5"
AUTHORITY_TOKEN = "PI_GATE_STATE_0122"
SMOKE_PASS_TOKEN = "STAGE45D_D2B_H200_SMOKE_PASS_AUTO_CONTINUE_FULL"
FULL_PASS_TOKEN = "STAGE45D_D2B_VANDA_H200_SOLVER_STUDY_COMPLETE_FOR_PI_REVIEW"
RUNTIME_BINDING_SCHEMA = "CORELAB_STAGE45D_D2B_RUNTIME_BINDING_V1"
ADMISSION_RESULT_SCHEMA = "CORELAB_STAGE45D_D2B_RUNTIME_ADMISSION_V1"

CHECKPOINT_ORDER = tuple(d1.CHECKPOINT_ORDER)
WINDOW_ORDER = "target-major then snapshot-major"
WINDOW_COUNT = 24
SEED = 4242
SOLVER_ORDER = ("E4", "E8", "E16")
SOLVER_STEPS = {"E4": 4, "E8": 8, "E16": 16}
SOLVER_DT = {"E4": 0.25, "E8": 0.125, "E16": 0.0625}
SOLVER_T_VALUES = {
    arm: tuple(index / steps for index in range(steps))
    for arm, steps in SOLVER_STEPS.items()
}
TIMESTEP_BUCKET_SCALE = 1000
SOLVER_T_BUCKETS = {
    arm: tuple(int(value * TIMESTEP_BUCKET_SCALE) for value in values)
    for arm, values in SOLVER_T_VALUES.items()
}
SOLVER_SCHEDULES = {
    arm: {
        "solver": arm,
        "family": "FORWARD_EULER",
        "steps": SOLVER_STEPS[arm],
        "dt": SOLVER_DT[arm],
        "t_values": list(SOLVER_T_VALUES[arm]),
        "t_buckets": list(SOLVER_T_BUCKETS[arm]),
        "timestep_bucket_scale": TIMESTEP_BUCKET_SCALE,
    }
    for arm in SOLVER_ORDER
}
SMOKE_WINDOW_COUNT = 1
SMOKE_ACTION_HEAD_CALLS = SMOKE_WINDOW_COUNT * sum(SOLVER_STEPS.values())
SMOKE_CONDITIONING_PASSES = 1
FULL_ACTION_HEAD_CALLS = len(CHECKPOINT_ORDER) * WINDOW_COUNT * sum(SOLVER_STEPS.values())
FULL_CONDITIONING_PASSES = len(CHECKPOINT_ORDER) * WINDOW_COUNT
FULL_TOTAL_MODEL_SIDE_CALLS = FULL_ACTION_HEAD_CALLS + FULL_CONDITIONING_PASSES

CALL_BUDGET = {
    "checkpoint_order": list(CHECKPOINT_ORDER),
    "windows_per_checkpoint": WINDOW_COUNT,
    "window_order": WINDOW_ORDER,
    "noise_seed": SEED,
    "noise_dtype": "float32",
    "noise_device": "cpu",
    "same_initial_noise_across_solver_arms": True,
    "solver_order": list(SOLVER_ORDER),
    "solver_schedules": copy.deepcopy(SOLVER_SCHEDULES),
    "action_head_calls_per_window": {arm: SOLVER_STEPS[arm] for arm in SOLVER_ORDER},
    "action_head_calls": FULL_ACTION_HEAD_CALLS,
    "conditioning_passes": FULL_CONDITIONING_PASSES,
    "total_model_side_calls": FULL_TOTAL_MODEL_SIDE_CALLS,
    "smoke_action_head_calls": SMOKE_ACTION_HEAD_CALLS,
    "smoke_conditioning_passes": SMOKE_CONDITIONING_PASSES,
    "smoke_total_model_side_calls": SMOKE_ACTION_HEAD_CALLS + SMOKE_CONDITIONING_PASSES,
    "adaptive_extension": False,
}

SAMPLER_CONTRACT = {
    "name": "forward-euler-resolution-study",
    "solver_family": "FORWARD_EULER_ONLY",
    "solver_order": list(SOLVER_ORDER),
    "solver_schedules": copy.deepcopy(SOLVER_SCHEDULES),
    "noise_seed": SEED,
    "same_initial_noise_across_solver_arms": True,
    "free_running_reset": False,
    "public_sampler_calls": 0,
    "get_action_calls": 0,
    "adaptive_extension": False,
}

SOURCE_IDENTITIES = copy.deepcopy(d1.SOURCE_IDENTITIES)
CHECKPOINT_IDENTITIES = copy.deepcopy(d1.CHECKPOINT_IDENTITIES)
STATS_ROUTES = copy.deepcopy(d1.STATS_ROUTES)
DATASET_IDENTITY = copy.deepcopy(d1.DATASET_IDENTITY)
LOCKED_BOUNDARIES = {
    **copy.deepcopy(d1.LOCKED_BOUNDARIES),
    "D2B_solver_comparison": "AUTHORIZED_FORWARD_EULER_ONLY",
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
    "solver_comparison_only": True,
    "solver_family": "FORWARD_EULER_ONLY",
    "solver_order": list(SOLVER_ORDER),
    "solver_schedules": copy.deepcopy(SOLVER_SCHEDULES),
    "free_running": "x_FR(0)=x0; v_FR(t_i)=model(x_FR(t_i),t_i); x_FR(t_i+1)=x_FR(t_i)+dt_arm*v_FR(t_i)",
    "free_running_reset": False,
    "same_initial_noise_across_arms": True,
    "normalized_action_space_readout": "x_FR and v_FR are retained in each checkpoint's frozen normalized action space",
    "physical_readout": "final free-running state is decoded only after raw normalized arrays are sealed",
    "target_xyz_in_model_input": False,
    "stats_checkpoint_decoder_unchanged": True,
    "raw_before_derived": True,
    "public_sampler_calls": 0,
    "get_action_calls": 0,
    "no_teacher_forced_path": True,
    "no_scientific_threshold_or_winner": True,
}


class D2BAdmissionError(RuntimeError):
    """Raised when a materialized D2B runtime is not exactly admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2BAdmissionError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D2BAdmissionError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    modules = sys.modules if imported_modules is None else imported_modules
    found: list[str] = []
    for name in modules:
        if not isinstance(name, str):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in d1.FORBIDDEN_IMPORT_PREFIXES):
            found.append(name)
    return sorted(found)


def validate_static_contract() -> dict[str, Any]:
    _require(len(CHECKPOINT_ORDER) == 3, "D2B checkpoint cardinality drift")
    _require(WINDOW_COUNT == 24 and SOLVER_ORDER == ("E4", "E8", "E16"), "D2B solver grid drift")
    _require(tuple(SOLVER_STEPS.values()) == (4, 8, 16), "D2B solver step drift")
    _require(SMOKE_ACTION_HEAD_CALLS == 28, "D2B smoke action-head budget drift")
    _require(FULL_ACTION_HEAD_CALLS == 2016, "D2B action-head budget drift")
    _require(FULL_CONDITIONING_PASSES == 72 and FULL_TOTAL_MODEL_SIDE_CALLS == 2088, "D2B total budget drift")
    for arm in SOLVER_ORDER:
        schedule = SOLVER_SCHEDULES[arm]
        _require(len(schedule["t_values"]) == SOLVER_STEPS[arm], f"{arm} schedule cardinality drift")
        _require(len(schedule["t_buckets"]) == SOLVER_STEPS[arm], f"{arm} timestep bucket cardinality drift")
        _require(schedule["family"] == "FORWARD_EULER", f"{arm} is not forward Euler")
        _require(schedule["t_values"][0] == 0.0, f"{arm} does not start at t=0")
    _require(SAMPLER_CONTRACT["free_running_reset"] is False, "D2B permits a free-running reset")
    _require(SAMPLER_CONTRACT["public_sampler_calls"] == 0 and SAMPLER_CONTRACT["get_action_calls"] == 0, "D2B public API drift")
    _require(CONTRACT["target_xyz_in_model_input"] is False, "D2B target XYZ exclusion drift")
    return {
        "schema": "CORELAB_STAGE45D_D2B_STATIC_CONTRACT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "stage": STAGE,
        "authority": {"issue_11_comment": PI_GATE_COMMENT, "pr_52_comment": PR52_POINTER_COMMENT},
        "branch": BRANCH,
        "base_execution_head": BASE_EXECUTION_HEAD,
        "call_budget": copy.deepcopy(CALL_BUDGET),
        "sampler": copy.deepcopy(SAMPLER_CONTRACT),
        "contract": copy.deepcopy(CONTRACT),
        "locked_boundaries": copy.deepcopy(LOCKED_BOUNDARIES),
        "return_tokens": {"smoke": SMOKE_PASS_TOKEN, "full": FULL_PASS_TOKEN},
    }


def _exact(actual: Any, expected: Any, field: str) -> None:
    _require(actual == expected, f"{field} differs from the frozen D2B contract")


def _fresh_output_root(value: Any) -> Path:
    _require(isinstance(value, str) and value, "D2B output root is absent")
    path = Path(value).expanduser()
    _require(path.is_absolute() and not path.is_symlink(), "D2B output root must be an absolute non-symlink path")
    if path.exists():
        _require(path.is_dir() and not any(path.iterdir()), "D2B output root must be fresh and empty")
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
    _require(type(control.get("repair_cycle")) is int and 0 <= control["repair_cycle"] <= 2, "D2B repair cycle out of range")
    _require(type(control.get("smoke_attempt")) is int and 1 <= control["smoke_attempt"] <= 6, "D2B smoke attempt out of range")
    _exact(control.get("automatic_retry"), False, "automatic retry")
    if phase == "FULL":
        smoke_path = Path(str(control.get("smoke_seal_path", ""))).expanduser()
        smoke_sha = control.get("smoke_seal_sha256")
        _require(smoke_path.is_file() and isinstance(smoke_sha, str) and len(smoke_sha) == 64, "full D2B smoke seal binding absent")
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
    """Validate a materialized D2B binding before any model-side import."""

    found = model_imports_present(imported_modules)
    _require(not found, "model-side import already present before D2B admission: " + ",".join(found))
    _exact(binding.get("schema"), RUNTIME_BINDING_SCHEMA, "runtime binding schema")
    _exact(binding.get("stage"), STAGE, "runtime binding stage")
    _exact(binding.get("pi_gate"), PI_GATE_COMMENT, "PI gate")
    _exact(binding.get("execution_authorized"), True, "execution authorization")
    _exact(binding.get("pi_authority_token"), AUTHORITY_TOKEN, "authority token")
    _exact(authority_token, AUTHORITY_TOKEN, "launcher authority token")
    _exact(binding.get("base_execution_head"), BASE_EXECUTION_HEAD, "D2B base execution HEAD")

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
    _require(checkout["authorized_head"] != BASE_EXECUTION_HEAD, "D2B execution must contain a committed D2B change")

    for field, expected in (
        ("source_identities", SOURCE_IDENTITIES),
        ("checkpoint_identities", CHECKPOINT_IDENTITIES),
        ("statistics_routes", STATS_ROUTES),
        ("dataset_identity", DATASET_IDENTITY),
        ("sampler", SAMPLER_CONTRACT),
        ("call_budget", CALL_BUDGET),
        ("contract", CONTRACT),
        ("locked_boundaries", LOCKED_BOUNDARIES),
    ):
        _exact(binding.get(field), expected, f"binding {field}")
        _exact(observed.get(field), expected, f"observed {field}")
    _exact(observed.get("target_xyz_in_model_input"), False, "target XYZ exclusion")
    _exact(observed.get("runtime_identity"), binding.get("runtime_identity"), "runtime identity")
    _exact(observed.get("host_gpu_runtime"), binding.get("host_gpu_runtime"), "observed H200 runtime")
    _exact(observed.get("zero_counters"), binding.get("zero_counters"), "observed zero counters")
    _exact(binding.get("zero_counters"), {
        "model_construction": 0, "model_forward": 0, "gpu": 0, "pbs": 1,
        "training_steps": 0, "backward_steps": 0, "optimizer_steps": 0,
        "checkpoint_write": 0, "action_head_calls": 0, "conditioning_passes": 0,
    }, "pre-model zero counters")
    _validate_execution_control(binding, observed, phase)

    boundary = _mapping(binding.get("execution_boundary"), field="binding.execution_boundary")
    observed_boundary = _mapping(observed.get("execution_boundary"), field="observed.execution_boundary")
    _exact(observed_boundary, boundary, "observed execution boundary")
    for key, value in {
        "training": False, "backward": False, "optimizer_steps": 0,
        "checkpoint_write": 0, "public_sampler_calls": 0, "get_action_calls": 0,
        "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0,
        "hardware": 0,
    }.items():
        _exact(boundary.get(key), value, f"locked execution boundary {key}")
    _exact(boundary.get("D2B_solver_comparison"), "RUNTIME_ONLY", "D2B solver boundary")
    _exact(boundary.get("h200"), 1, "H200 execution boundary")
    _exact(boundary.get("pbs"), 1, "PBS execution boundary")
    for key in ("model_construction", "model_forward", "gpu"):
        _exact(boundary.get(key), "RUNTIME_ONLY", f"{key} boundary")

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
            "D2B_forward_euler_resolution_contract_binding", "call_budget_and_fresh_output_binding",
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
    "DIRECTORY_PATH_KEYS", "D2BAdmissionError", "FULL_ACTION_HEAD_CALLS",
    "FULL_CONDITIONING_PASSES", "FULL_PASS_TOKEN", "FULL_TOTAL_MODEL_SIDE_CALLS",
    "LOCKED_BOUNDARIES", "PI_GATE_COMMENT", "PR52_POINTER_COMMENT", "RUNTIME_BINDING_SCHEMA",
    "RUNTIME_PATH_KEYS", "SAMPLER_CONTRACT", "SEED", "SMOKE_ACTION_HEAD_CALLS",
    "SMOKE_CONDITIONING_PASSES", "SMOKE_PASS_TOKEN", "SOLVER_DT", "SOLVER_ORDER",
    "SOLVER_SCHEDULES", "SOLVER_STEPS", "SOLVER_T_BUCKETS", "SOLVER_T_VALUES",
    "SOURCE_IDENTITIES", "STATS_ROUTES", "TIMESTEP_BUCKET_SCALE", "WINDOW_COUNT",
    "WINDOW_ORDER", "model_imports_present", "validate_future_admission", "validate_static_contract",
]
