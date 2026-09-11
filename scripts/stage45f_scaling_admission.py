#!/usr/bin/env python3
"""Fail-closed CPU planner for hypothetical longer-training experiments.

The purpose is to prevent a future `max_steps` increase from being mislabeled
as a pure exposure intervention when optimizer/scheduler history, resume state,
or data exposure also changes.  This module never launches training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class ScalingAdmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalingAdmissionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{field} must be positive int")
    return value


def _finite(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} must be finite")
    return result


def _recipe(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} must be object")
    required = (
        "recipe_id",
        "dataset_tree_sha256",
        "normalization_identity",
        "loss_identity",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_steps",
        "max_steps",
        "scheduler_horizon_binding",
        "scheduler_horizon_steps",
        "seed",
        "start_mode",
    )
    for key in required:
        require(key in value, f"{name}.{key} missing")
    require(value["scheduler_horizon_binding"] in {"MAX_STEPS", "EXPLICIT_FIXED", "UNKNOWN"}, f"{name}.scheduler_horizon_binding invalid")
    require(value["start_mode"] in {"FRESH_BASE", "RESUME_CHECKPOINT"}, f"{name}.start_mode invalid")
    max_steps = _positive_int(value["max_steps"], f"{name}.max_steps")
    horizon = _positive_int(value["scheduler_horizon_steps"], f"{name}.scheduler_horizon_steps")
    warmup = int(value["warmup_steps"])
    require(warmup >= 0, f"{name}.warmup_steps invalid")
    seed = int(value["seed"])
    _finite(value["learning_rate"], f"{name}.learning_rate")
    _finite(value["weight_decay"], f"{name}.weight_decay")
    exact_sample_exposure = value.get("exact_sample_exposure")
    require(exact_sample_exposure is None or isinstance(exact_sample_exposure, Mapping), f"{name}.exact_sample_exposure invalid")
    if exact_sample_exposure is not None:
        require(isinstance(exact_sample_exposure.get("samples_seen"), int) and exact_sample_exposure["samples_seen"] >= 0, f"{name}.samples_seen invalid")
        require(isinstance(exact_sample_exposure.get("per_window_counts_available"), bool), f"{name}.per_window_counts_available invalid")
    resume = value.get("resume_state")
    if value["start_mode"] == "RESUME_CHECKPOINT":
        require(isinstance(resume, Mapping), f"{name}.resume_state required")
        for field in ("checkpoint_tree_sha256", "optimizer_state_sha256", "scheduler_state_sha256", "rng_state_sha256"):
            require(isinstance(resume.get(field), str) and resume[field], f"{name}.resume_state.{field} missing")
    return {
        "recipe_id": str(value["recipe_id"]),
        "dataset_tree_sha256": str(value["dataset_tree_sha256"]),
        "normalization_identity": str(value["normalization_identity"]),
        "loss_identity": str(value["loss_identity"]),
        "optimizer": str(value["optimizer"]),
        "learning_rate": float(value["learning_rate"]),
        "weight_decay": float(value["weight_decay"]),
        "lr_scheduler_type": str(value["lr_scheduler_type"]),
        "warmup_steps": warmup,
        "max_steps": max_steps,
        "scheduler_horizon_binding": str(value["scheduler_horizon_binding"]),
        "scheduler_horizon_steps": horizon,
        "seed": seed,
        "start_mode": str(value["start_mode"]),
        "resume_state": dict(resume) if isinstance(resume, Mapping) else None,
        "exact_sample_exposure": dict(exact_sample_exposure) if isinstance(exact_sample_exposure, Mapping) else None,
    }


def compare(baseline: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    semantic_fields = (
        "dataset_tree_sha256",
        "normalization_identity",
        "loss_identity",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_steps",
        "seed",
    )
    changed = [field for field in semantic_fields if baseline[field] != proposal[field]]
    step_change = proposal["max_steps"] != baseline["max_steps"]
    horizon_change = proposal["scheduler_horizon_steps"] != baseline["scheduler_horizon_steps"]
    horizon_binding_change = proposal["scheduler_horizon_binding"] != baseline["scheduler_horizon_binding"]
    start_mode_change = proposal["start_mode"] != baseline["start_mode"]

    confounds: list[str] = []
    if changed:
        confounds.append("recipe_semantics_changed:" + ",".join(changed))
    if horizon_change or horizon_binding_change:
        confounds.append("scheduler_horizon_or_binding_changed")
    if baseline["lr_scheduler_type"].lower().startswith("cosine") and step_change:
        if proposal["scheduler_horizon_binding"] == "MAX_STEPS":
            confounds.append("cosine_lr_history_changes_when_max_steps_changes")
        elif proposal["scheduler_horizon_binding"] == "UNKNOWN":
            confounds.append("cosine_scheduler_horizon_unknown")
    if start_mode_change:
        confounds.append("fresh_vs_resume_changed")
    if proposal["start_mode"] == "RESUME_CHECKPOINT":
        confounds.append("resume_requires_continuity_proof_not_equivalence_assumption")

    baseline_exposure = baseline.get("exact_sample_exposure")
    proposal_exposure = proposal.get("exact_sample_exposure")
    exposure_claim = "UNKNOWN"
    if baseline_exposure is not None and proposal_exposure is not None:
        if baseline_exposure.get("per_window_counts_available") and proposal_exposure.get("per_window_counts_available"):
            exposure_claim = "EXACT_ACCOUNTING_AVAILABLE"
        else:
            exposure_claim = "TOTAL_SAMPLES_ONLY_NOT_WINDOW_MATCHED"

    pure_exposure_candidate = bool(
        step_change
        and not changed
        and not horizon_change
        and not horizon_binding_change
        and not start_mode_change
        and proposal["start_mode"] == "FRESH_BASE"
        and not confounds
    )

    classification = (
        "PURE_EXPOSURE_CANDIDATE_REQUIRES_PI_AND_EXACT_LOADER_EVIDENCE"
        if pure_exposure_candidate
        else "SCALING_CONFOUNDED_OR_UNPROVEN"
    )
    return {
        "schema": "STAGE45F_SCALING_COMPARISON_V1",
        "status": "PASS_ANALYSIS_ONLY",
        "classification": classification,
        "baseline_recipe_id": baseline["recipe_id"],
        "proposal_recipe_id": proposal["recipe_id"],
        "step_change": {"baseline": baseline["max_steps"], "proposal": proposal["max_steps"]},
        "scheduler_horizon_change": {"baseline": baseline["scheduler_horizon_steps"], "proposal": proposal["scheduler_horizon_steps"]},
        "start_mode": {"baseline": baseline["start_mode"], "proposal": proposal["start_mode"]},
        "changed_semantic_fields": changed,
        "confounds": confounds,
        "exact_sample_exposure_state": exposure_claim,
        "epoch_claim_allowed": exposure_claim == "EXACT_ACCOUNTING_AVAILABLE",
        "execution_authorized": False,
        "decision_state": "PI_DECISION_REQUIRED",
        "notes": [
            "max_steps is an optimizer-step budget, not an epoch count",
            "resume from a 2000-step checkpoint is not assumed equivalent to a fresh run whose scheduler was defined for a longer horizon",
            "a cosine schedule tied to max_steps changes earlier LR values when the horizon changes",
        ],
    }


def analyze(document: Any) -> dict[str, Any]:
    require(isinstance(document, Mapping), "input must be object")
    require(document.get("schema") == "STAGE45F_HYPOTHETICAL_SCALING_PROPOSAL_V1", "input schema drift")
    require(document.get("validation_e_read") is False and document.get("holdout_f_read") is False, "locked split read indicated")
    baseline = _recipe(document.get("baseline"), "baseline")
    proposal = _recipe(document.get("proposal"), "proposal")
    return compare(baseline, proposal)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(analyze(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScalingAdmissionError as exc:
        print(f"STAGE45F_SCALING_ADMISSION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
