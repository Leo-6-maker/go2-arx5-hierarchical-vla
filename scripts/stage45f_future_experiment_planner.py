#!/usr/bin/env python3
"""CPU/static planner for possible post-Stage45E scientific branches.

This module is deliberately unable to execute models, training, simulation,
or Validation/Holdout reads. It accepts only compact PI-supplied summaries of
already-sealed TRAIN evidence and produces a scenario matrix plus explicit
confound warnings. It never authorizes an experiment or declares a winner.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests/STAGE45F_FUTURE_EXPERIMENT_PREP_V1.json"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "transformers",
    "gr00t",
    "jax",
    "tensorflow",
    "subprocess",
    "numpy",
    "isaacgym",
    "omni",
}


class PlanningError(RuntimeError):
    """Raised when a future-experiment summary violates the prep contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanningError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST)
    require(value.get("schema") == "STAGE45F_FUTURE_EXPERIMENT_PREP_V2", "manifest schema drift")
    require(value.get("status") == "PREP_ONLY_NO_EXECUTION", "manifest is not prep-only")
    require(value.get("hypothetical_only") is True, "training tiers are not hypothetical")
    return value


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    return {
        "schema": "STAGE45F_FUTURE_EXPERIMENT_STATIC_GUARD_V2",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def plan() -> dict[str, Any]:
    cfg = manifest()
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    contract_keys = (
        "training_scaling_contract",
        "exposure_instrumentation_contract",
        "dataset_expansion_contract",
        "rollout_relabel_contract",
        "replication_contract",
        "factorial_contract",
        "checkpoint_selection_contract",
        "validation_contract",
        "resource_and_storage_contract",
        "retry_recovery_contract",
        "large_scale_campaign_contract",
    )
    contracts = {}
    for key in contract_keys:
        require(isinstance(cfg.get(key), Mapping), f"manifest contract missing: {key}")
        contracts[key] = cfg[key]
    return {
        "schema": "STAGE45F_FUTURE_EXPERIMENT_PLAN_V2",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "authority": cfg["authority"],
        "scenario_classes": cfg["scenario_classes"],
        "hypothetical_training_tiers": cfg["hypothetical_training_tiers"],
        "contracts": contracts,
        "execution_supported": False,
        "decision_state": "PI_DECISION_REQUIRED",
        "static_guard": guard,
        "return_token": cfg["return_token"],
    }


def _finite_optional(value: Any, field: str) -> float | None:
    if value is None:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric or null")
    result = float(value)
    require(math.isfinite(result), f"{field} must be finite")
    return result


def validate_summary(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "summary must be an object")
    require(value.get("schema") == "STAGE45F_PI_SUPPLIED_TRAIN_EVIDENCE_SUMMARY_V1", "summary schema drift")
    require(value.get("split") == "TRAIN24", "planner accepts TRAIN24 summary only")
    require(value.get("validation_e_read") is False, "Validation-E must remain unread for this TRAIN-only summary classifier")
    require(value.get("holdout_f_read") is False, "Holdout-F must remain unread")

    selected_candidate = value.get("selected_candidate")
    require(selected_candidate is None or isinstance(selected_candidate, str), "selected_candidate must be null/string")
    recipe_frozen = value.get("recipe_frozen")
    require(isinstance(recipe_frozen, bool), "recipe_frozen must be boolean")

    cross_seed = value.get("cross_seed")
    require(isinstance(cross_seed, Mapping), "cross_seed block missing")
    valid_seed_count = cross_seed.get("valid_seed_count")
    require(isinstance(valid_seed_count, int) and valid_seed_count >= 0, "valid_seed_count invalid")
    consistent_direction = cross_seed.get("consistent_direction")
    require(consistent_direction in {True, False, None}, "consistent_direction invalid")

    exposure = value.get("exposure")
    require(isinstance(exposure, Mapping), "exposure block missing")
    trend_from_1000_to_2000 = exposure.get("trend_from_1000_to_2000")
    require(trend_from_1000_to_2000 in {"IMPROVED", "FLAT", "WORSE", "UNKNOWN"}, "exposure trend invalid")
    exact_sample_accounting = exposure.get("exact_sample_accounting_available")
    require(isinstance(exact_sample_accounting, bool), "exact_sample_accounting_available must be bool")

    data = value.get("data_support")
    require(isinstance(data, Mapping), "data_support block missing")
    target_heterogeneity = data.get("target_heterogeneity")
    require(target_heterogeneity in {"LOW", "MODERATE", "HIGH", "UNKNOWN"}, "target_heterogeneity invalid")
    support_gap = data.get("support_gap_evidence")
    require(support_gap in {True, False, None}, "support_gap_evidence invalid")
    current_stats_clipping_fraction = _finite_optional(data.get("current_stats_clipping_fraction_for_proposed_new_data"), "current_stats_clipping_fraction_for_proposed_new_data")
    if current_stats_clipping_fraction is not None:
        require(0.0 <= current_stats_clipping_fraction <= 1.0, "clipping fraction outside [0,1]")

    rollout = value.get("rollout")
    require(isinstance(rollout, Mapping), "rollout block missing")
    teacher_forced_good = rollout.get("teacher_forced_good")
    free_running_gap = rollout.get("free_running_gap")
    require(teacher_forced_good in {True, False, None}, "teacher_forced_good invalid")
    require(free_running_gap in {True, False, None}, "free_running_gap invalid")

    return {
        "schema": value["schema"],
        "split": "TRAIN24",
        "validation_e_read": False,
        "holdout_f_read": False,
        "selected_candidate": selected_candidate,
        "recipe_frozen": recipe_frozen,
        "cross_seed": {
            "valid_seed_count": valid_seed_count,
            "consistent_direction": consistent_direction,
        },
        "exposure": {
            "trend_from_1000_to_2000": trend_from_1000_to_2000,
            "exact_sample_accounting_available": exact_sample_accounting,
        },
        "data_support": {
            "target_heterogeneity": target_heterogeneity,
            "support_gap_evidence": support_gap,
            "current_stats_clipping_fraction_for_proposed_new_data": current_stats_clipping_fraction,
        },
        "rollout": {
            "teacher_forced_good": teacher_forced_good,
            "free_running_gap": free_running_gap,
        },
    }


def scenario_matrix(summary: Mapping[str, Any]) -> dict[str, Any]:
    cfg = manifest()
    rows: list[dict[str, Any]] = []

    valid_seeds = int(summary["cross_seed"]["valid_seed_count"])
    consistent = summary["cross_seed"]["consistent_direction"]
    selected = summary["selected_candidate"]
    frozen = bool(summary["recipe_frozen"])
    exposure_trend = summary["exposure"]["trend_from_1000_to_2000"]
    exact_exposure = bool(summary["exposure"]["exact_sample_accounting_available"])
    support_gap = summary["data_support"]["support_gap_evidence"]
    heterogeneity = summary["data_support"]["target_heterogeneity"]
    tf_good = summary["rollout"]["teacher_forced_good"]
    fr_gap = summary["rollout"]["free_running_gap"]
    clipping = summary["data_support"]["current_stats_clipping_fraction_for_proposed_new_data"]

    rows.append({
        "scenario": "SEED_REPLICATION",
        "relevant": selected is not None and (valid_seeds < 3 or consistent is None),
        "why": "A selected TRAIN24 candidate should not be promoted from a single training seed; preserve every valid replicate and forbid metric-based retry selection.",
        "execution_authorized": False,
    })
    rows.append({
        "scenario": "EXPOSURE_SCALING",
        "relevant": selected is not None and exposure_trend == "IMPROVED",
        "why": "Improvement from 1000 to 2000 suggests additional optimization exposure may still matter, but max_steps/scheduler changes can alter LR history.",
        "confounds": [
            "scheduler horizon / LR trajectory",
            "resume-vs-fresh optimizer and scheduler state",
            "unknown exact optimizer-step-bound per-window exposure" if not exact_exposure else "exact exposure evidence supplied",
        ],
        "hypothetical_tiers": cfg["hypothetical_training_tiers"],
        "execution_authorized": False,
    })
    rows.append({
        "scenario": "DATA_EXPANSION",
        "relevant": support_gap is True or heterogeneity == "HIGH",
        "why": "Target/window heterogeneity or explicit support gaps may justify broader training support, but data expansion is not automatically normalization- or sampling-neutral.",
        "current_stats_clipping_fraction_if_known": clipping,
        "execution_authorized": False,
    })
    rows.append({
        "scenario": "DATA_PLUS_NORMALIZATION",
        "relevant": support_gap is True and clipping is not None and clipping > 0.0,
        "why": "Any numerical change to normalization stats is data plus normalization, not data-only. The planner deliberately defines no acceptable clipping threshold.",
        "execution_authorized": False,
    })
    rows.append({
        "scenario": "ROLLOUT_RELABEL",
        "relevant": tf_good is True and fr_gap is True,
        "why": "A persistent teacher-forced/free-running gap is compatible with off-manifold rollout error. Visited states may come from the policy/WBC but labels must be independent simulator expert/oracle labels.",
        "groot_self_label_allowed": False,
        "execution_authorized": False,
    })
    rows.append({
        "scenario": "VALIDATION_RECIPE_FREEZE",
        "relevant": selected is not None and frozen and valid_seeds >= 3 and consistent is True,
        "why": "Only a preselected, frozen candidate with adequate TRAIN-only replication is structurally ready to request a separate first Validation-E gate.",
        "validation_e_authorized": False,
        "execution_authorized": False,
    })

    return {
        "schema": "STAGE45F_FUTURE_EXPERIMENT_SCENARIO_MATRIX_V2",
        "status": "PASS_DESCRIPTIVE_TRAIN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "automatic_experiment_selection": False,
        "summary": dict(summary),
        "scenarios": rows,
        "prepared_additional_scenarios": [
            scenario for scenario in cfg["scenario_classes"]
            if scenario not in {row["scenario"] for row in rows}
        ],
        "claim_boundary": {
            "no_winner_declared": True,
            "no_success_threshold_invented": True,
            "no_iid_significance_claim": True,
            "validation_e_authorized": False,
            "holdout_f_authorized": False,
            "model_execution": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--classify", action="store_true")
    p.add_argument("--input", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.input is not None, "--classify requires --input")
    summary = validate_summary(read_json(args.input))
    print(json.dumps(scenario_matrix(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanningError as exc:
        print(f"STAGE45F_FUTURE_EXPERIMENT_PLANNER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
