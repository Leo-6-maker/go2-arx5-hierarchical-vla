#!/usr/bin/env python3
"""Preparation-only admission for a future large-scale fine-tuning campaign.

The purpose is to prevent a future 'scale it up' decision from silently changing
multiple scientific variables. The input is a compact PI-prepared campaign
request containing identities/hashes of earlier audits. This tool validates the
request structure and returns a blocked-until-PI-gate plan. It never launches
training or reads model/data payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class LargeScaleAdmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LargeScaleAdmissionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{field} must be lowercase sha256")
    return value


def _positive_int_list(value: Any, field: str) -> list[int]:
    require(isinstance(value, list) and value, f"{field} must be non-empty list")
    require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value), f"{field} contains invalid value")
    require(value == sorted(set(value)), f"{field} must be strictly increasing unique integers")
    return list(value)


def _checkpoint_schedule(value: Any, tiers: Sequence[int]) -> dict[str, list[int]]:
    require(isinstance(value, Mapping), "training.checkpoint_steps_by_tier must be object")
    expected_keys = [str(tier) for tier in tiers]
    require(list(value.keys()) == expected_keys, "checkpoint_steps_by_tier key/order drift")
    result: dict[str, list[int]] = {}
    for tier in tiers:
        key = str(tier)
        steps = _positive_int_list(value[key], f"training.checkpoint_steps_by_tier[{key}]")
        require(all(step <= tier for step in steps), f"checkpoint step exceeds run tier {tier}")
        require(steps[-1] == tier, f"run tier {tier} must retain its terminal checkpoint")
        result[key] = steps
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_LARGE_SCALE_CAMPAIGN_REQUEST_V1", "schema drift")
    require(value.get("selection_source_split") == "TRAIN24", "selection source must be TRAIN24")
    require(value.get("validation_e_read") is False, "Validation-E must remain unread at campaign-prep time")
    require(value.get("holdout_f_read") is False, "Holdout-F must remain unread")
    require(value.get("recipe_frozen") is True, "recipe must be frozen before campaign preparation")
    require(value.get("pi_selected_candidate") is True, "PI candidate selection is required")

    candidate = value.get("candidate")
    require(isinstance(candidate, Mapping), "candidate block missing")
    required_candidate = (
        "candidate_label",
        "training_recipe_identity",
        "dataset_tree_sha256",
        "normalization_identity",
        "loss_identity",
        "base_model_tree_sha256",
        "sampler_identity",
        "decoder_identity",
        "metric_identity",
    )
    for field in required_candidate:
        require(isinstance(candidate.get(field), str) and candidate[field], f"candidate.{field} missing")
    _sha(candidate["dataset_tree_sha256"], "candidate.dataset_tree_sha256")
    _sha(candidate["base_model_tree_sha256"], "candidate.base_model_tree_sha256")

    training = value.get("training")
    require(isinstance(training, Mapping), "training block missing")
    seeds = _positive_int_list(training.get("predeclared_seeds"), "training.predeclared_seeds")
    step_tiers = _positive_int_list(training.get("optimizer_step_tiers"), "training.optimizer_step_tiers")
    checkpoint_steps_by_tier = _checkpoint_schedule(training.get("checkpoint_steps_by_tier"), step_tiers)
    tier_execution_mode = training.get("tier_execution_mode")
    require(tier_execution_mode in {"INDEPENDENT_FRESH_RUNS", "RESUME_CHAIN"}, "training.tier_execution_mode invalid")
    resume_continuity = training.get("resume_continuity_plan_sha256")
    if tier_execution_mode == "RESUME_CHAIN":
        _sha(resume_continuity, "training.resume_continuity_plan_sha256")
    else:
        require(resume_continuity in {None, ""}, "fresh independent tiers must not claim a resume-continuity plan")
    require(training.get("retry_selection_by_metric") is False, "metric-based retry selection must be disabled")
    require(training.get("adaptive_step_extension") is False, "adaptive step extension must be disabled before a new PI gate")
    require(isinstance(training.get("stop_extension_rule"), str) and training["stop_extension_rule"], "predeclared stop/extension rule missing")
    for field in (
        "optimizer_identity",
        "scheduler_identity",
        "scheduler_horizon_policy",
        "precision_identity",
        "effective_global_batch_identity",
        "trainable_module_identity",
    ):
        require(isinstance(training.get(field), str) and training[field], f"training.{field} missing")

    evidence = value.get("required_preflight_evidence")
    require(isinstance(evidence, Mapping), "required_preflight_evidence missing")
    required_evidence = (
        "train_decision_packet_sha256",
        "recipe_freeze_ledger_sha256",
        "scaling_admission_sha256",
        "resource_scaling_audit_sha256",
        "compute_budget_sha256",
        "exposure_instrumentation_plan_sha256",
        "dataset_support_audit_sha256",
        "storage_retention_plan_sha256",
    )
    normalized_evidence = {field: _sha(evidence.get(field), f"required_preflight_evidence.{field}") for field in required_evidence}

    resource = value.get("resource")
    require(isinstance(resource, Mapping), "resource block missing")
    require(resource.get("resource_selected") is True, "resource must be selected explicitly")
    require(isinstance(resource.get("device_role"), str) and resource["device_role"], "resource.device_role missing")
    require(isinstance(resource.get("device_count"), int) and resource["device_count"] >= 1, "resource.device_count invalid")
    require(resource.get("effective_batch_semantics_frozen") is True, "effective batch semantics must be frozen")
    require(resource.get("precision_semantics_frozen") is True, "precision semantics must be frozen")

    storage = value.get("storage")
    require(isinstance(storage, Mapping), "storage block missing")
    require(isinstance(storage.get("required_bytes_estimate"), int) and storage["required_bytes_estimate"] > 0, "storage.required_bytes_estimate invalid")
    require(isinstance(storage.get("available_bytes_verified"), int) and storage["available_bytes_verified"] > 0, "storage.available_bytes_verified invalid")
    storage_ready = storage["available_bytes_verified"] >= storage["required_bytes_estimate"]
    require(storage_ready, "verified storage is below estimated retained-artifact requirement")
    require(storage.get("retain_all_valid_runs") is True, "all scientifically valid runs must be retained")

    exposure = value.get("exposure")
    require(isinstance(exposure, Mapping), "exposure block missing")
    require(exposure.get("optimizer_step_bound_trace_planned") is True, "optimizer-step-bound exposure trace must be planned")
    require(exposure.get("epoch_claim_from_steps") is False, "steps must not be relabeled as epochs")

    return {
        "schema": "STAGE45F_LARGE_SCALE_CAMPAIGN_ADMISSION_V2",
        "status": "PASS_STRUCTURE_ONLY_PI_EXECUTION_GATE_REQUIRED",
        "decision_state": "PI_GATE_REQUIRED",
        "candidate": dict(candidate),
        "predeclared_seeds": seeds,
        "optimizer_step_tiers": step_tiers,
        "checkpoint_steps_by_tier": checkpoint_steps_by_tier,
        "tier_execution_mode": tier_execution_mode,
        "resume_continuity_plan_sha256": resume_continuity or None,
        "required_preflight_evidence": normalized_evidence,
        "resource": dict(resource),
        "storage_ready": True,
        "exact_exposure_instrumentation_required": True,
        "metric_based_retry_selection_allowed": False,
        "adaptive_extension_allowed_without_new_pi_gate": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "training_authorized": False,
        "gpu_authorized": False,
        "pbs_authorized": False,
        "claim_boundary": "This artifact only proves that a future campaign request is structurally complete. Run tiers and checkpoint schedules are separate. It does not establish scientific justification and does not authorize execution.",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(analyze(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LargeScaleAdmissionError as exc:
        print(f"STAGE45F_LARGE_SCALE_CAMPAIGN_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
