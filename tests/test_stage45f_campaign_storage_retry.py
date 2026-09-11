"""Synthetic regression tests for Stage45F campaign storage/retry governance."""

import pytest

from scripts import stage45f_retry_recovery_policy as retry_policy
from scripts import stage45f_storage_retention_plan as storage_plan


def _storage_request(mode: str, *, available: int = 50000) -> dict:
    return {
        "schema": "STAGE45F_STORAGE_RETENTION_REQUEST_V2",
        "validation_e_read": False,
        "holdout_f_read": False,
        "retain_all_valid_runs": True,
        "observed_full_state_checkpoint_bytes": [1000, 1200, 1100],
        "predeclared_seeds": [2222, 2223, 2224],
        "run_tiers": [
            {"max_steps": 2000, "checkpoint_steps": [1000, 2000]},
            {"max_steps": 4000, "checkpoint_steps": [1000, 2000, 4000]},
            {"max_steps": 8000, "checkpoint_steps": [1000, 2000, 4000, 8000]},
        ],
        "tier_execution_mode": mode,
        "estimated_non_checkpoint_bytes_per_run": 500,
        "available_bytes_verified": available,
        "safety_margin_fraction": 0.25,
    }


def test_independent_tiers_retain_each_run_root_and_all_intermediate_checkpoints() -> None:
    result = storage_plan.analyze(_storage_request("INDEPENDENT_FRESH_RUNS"))
    assert result["schema"] == "STAGE45F_STORAGE_RETENTION_PLAN_V2"
    assert result["observed_checkpoint_bytes"]["conservative_size_used"] == 1200
    assert result["retained_run_root_count"] == 9
    # Per seed: 2 + 3 + 4 checkpoints across three independent fresh runs.
    assert result["retained_checkpoint_count"] == 27
    assert result["retain_all_valid_runs"] is True
    assert result["automatic_deletion_allowed"] is False
    assert result["delete_worse_valid_runs_allowed"] is False
    assert result["training_authorized"] is False


def test_resume_chain_counts_union_of_checkpoint_steps_per_seed() -> None:
    result = storage_plan.analyze(_storage_request("RESUME_CHAIN"))
    assert result["retained_run_root_count"] == 3
    # Per seed union = {1000, 2000, 4000, 8000}.
    assert result["retained_checkpoint_count"] == 12
    assert result["tier_execution_mode"] == "RESUME_CHAIN"
    assert result["training_authorized"] is False


def test_storage_plan_reports_capacity_shortfall_without_deleting_evidence() -> None:
    result = storage_plan.analyze(_storage_request("INDEPENDENT_FRESH_RUNS", available=1000))
    assert result["descriptive_capacity_sufficient"] is False
    assert result["automatic_deletion_allowed"] is False
    assert result["delete_worse_valid_runs_allowed"] is False
    assert result["storage_selected"] is False


def test_storage_plan_allows_intermediate_checkpoint_below_run_tier() -> None:
    request = _storage_request("INDEPENDENT_FRESH_RUNS")
    request["run_tiers"] = [{"max_steps": 2000, "checkpoint_steps": [1000, 2000]}]
    result = storage_plan.analyze(request)
    assert result["run_tiers"] == [{"max_steps": 2000, "checkpoint_steps": [1000, 2000]}]
    assert result["retained_checkpoint_count"] == 6


def test_storage_plan_requires_terminal_checkpoint_for_each_tier() -> None:
    request = _storage_request("INDEPENDENT_FRESH_RUNS")
    request["run_tiers"] = [{"max_steps": 2000, "checkpoint_steps": [1000]}]
    with pytest.raises(storage_plan.StoragePlanError, match="terminal checkpoint"):
        storage_plan.analyze(request)


def _retry_request(failure_class: str, *, valid: bool, checkpoints: bool, science_change: bool = False) -> dict:
    return {
        "schema": "STAGE45F_RETRY_RECOVERY_REQUEST_V1",
        "validation_e_read": False,
        "holdout_f_read": False,
        "failure_class": failure_class,
        "failed_evidence_preserved": True,
        "metric_based_retry_selection": False,
        "valid_training_completed": valid,
        "checkpoint_artifacts_complete": checkpoints,
        "proposed_retry_changes_science_contract": science_change,
        "proposed_retry_changes_resource_semantics": False,
    }


def test_postprocessing_failure_after_valid_training_requires_readonly_recovery() -> None:
    result = retry_policy.analyze(
        _retry_request("POSTPROCESSING_AFTER_VALID_TRAINING", valid=True, checkpoints=True)
    )
    assert result["recommended_action"] == "READONLY_RECOVERY_AUDIT_DO_NOT_RETRAIN"
    assert result["retry_training_authorized_by_this_policy"] is False
    assert result["metric_based_retry_selection_allowed"] is False


def test_engineering_retry_cannot_hide_science_change() -> None:
    result = retry_policy.analyze(
        _retry_request("ENGINEERING_TRANSIENT", valid=False, checkpoints=False, science_change=True)
    )
    assert result["recommended_action"] == "STOP_RETRY_PROPOSAL_CHANGES_CONTRACT"
    assert result["science_contract_change_inside_engineering_retry_allowed"] is False
    assert result["retry_training_authorized_by_this_policy"] is False


def test_scientific_integrity_failure_stops_for_new_pi_gate() -> None:
    result = retry_policy.analyze(
        _retry_request("SCIENTIFIC_INTEGRITY", valid=False, checkpoints=False)
    )
    assert result["recommended_action"] == "STOP_FOR_NEW_PI_SCIENTIFIC_GATE"
    assert result["retry_training_authorized_by_this_policy"] is False


def test_postprocessing_class_requires_completed_valid_training() -> None:
    with pytest.raises(retry_policy.RetryPolicyError, match="complete valid training"):
        retry_policy.analyze(
            _retry_request("POSTPROCESSING_AFTER_VALID_TRAINING", valid=False, checkpoints=True)
        )
