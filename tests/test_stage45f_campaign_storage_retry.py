"""Synthetic regression tests for Stage45F campaign storage/retry governance."""

import pytest

from scripts import stage45f_retry_recovery_policy as retry_policy
from scripts import stage45f_storage_retention_plan as storage_plan


def test_storage_plan_retains_all_valid_runs_and_uses_conservative_checkpoint_size() -> None:
    result = storage_plan.analyze(
        {
            "schema": "STAGE45F_STORAGE_RETENTION_REQUEST_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "retain_all_valid_runs": True,
            "observed_full_state_checkpoint_bytes": [1000, 1200, 1100],
            "predeclared_seeds": [2222, 2223, 2224],
            "checkpoint_steps": [2000, 4000, 8000],
            "estimated_non_checkpoint_bytes_per_run": 500,
            "available_bytes_verified": 20000,
            "safety_margin_fraction": 0.25,
        }
    )
    assert result["observed_checkpoint_bytes"]["conservative_size_used"] == 1200
    assert result["retained_checkpoint_count"] == 9
    assert result["retain_all_valid_runs"] is True
    assert result["automatic_deletion_allowed"] is False
    assert result["delete_worse_valid_runs_allowed"] is False
    assert result["training_authorized"] is False


def test_storage_plan_reports_capacity_shortfall_without_deleting_evidence() -> None:
    result = storage_plan.analyze(
        {
            "schema": "STAGE45F_STORAGE_RETENTION_REQUEST_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "retain_all_valid_runs": True,
            "observed_full_state_checkpoint_bytes": [1000],
            "predeclared_seeds": [2222, 2223],
            "checkpoint_steps": [2000, 4000],
            "estimated_non_checkpoint_bytes_per_run": 500,
            "available_bytes_verified": 1000,
            "safety_margin_fraction": 0.0,
        }
    )
    assert result["descriptive_capacity_sufficient"] is False
    assert result["automatic_deletion_allowed"] is False
    assert result["storage_selected"] is False


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
