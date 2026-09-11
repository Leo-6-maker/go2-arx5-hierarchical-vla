"""Synthetic tests for future large-scale tier and validation-lineage semantics."""

import pytest

from scripts import stage45f_large_scale_campaign_admission as campaign


def _request(mode: str, *, validation_state: str = "NOT_READ") -> dict:
    validation = {
        "state": "NOT_READ",
        "validation_e_read": False,
        "access_generation": 0,
        "validation_access_ledger_sha256": None,
        "validation_e_evidence_sha256": None,
        "holdout_f_read": False,
    }
    if validation_state == "READ_WITH_LEDGER":
        validation = {
            "state": "READ_WITH_LEDGER",
            "validation_e_read": True,
            "access_generation": 1,
            "validation_access_ledger_sha256": "c" * 64,
            "validation_e_evidence_sha256": "d" * 64,
            "holdout_f_read": False,
        }
    value = {
        "schema": "STAGE45F_LARGE_SCALE_CAMPAIGN_REQUEST_V2",
        "selection_source_split": "TRAIN24",
        "validation": validation,
        "recipe_frozen": True,
        "pi_selected_candidate": True,
        "campaign_recipe_generation": 2,
        "candidate": {
            "candidate_label": "future-candidate",
            "training_recipe_identity": "recipe-v1",
            "dataset_tree_sha256": "1" * 64,
            "dataset_lineage_mode": "CURRENT_ACCEPTED_TRAIN",
            "normalization_identity": "stats-v1",
            "loss_identity": "loss-v1",
            "base_model_tree_sha256": "2" * 64,
            "sampler_identity": "sampler-v1",
            "decoder_identity": "decoder-v1",
            "metric_identity": "metrics-v1",
        },
        "expanded_data_preflight": None,
        "training": {
            "predeclared_seeds": [2222, 2223, 2224],
            "optimizer_step_tiers": [2000, 4000],
            "checkpoint_steps_by_tier": {
                "2000": [1000, 2000],
                "4000": [1000, 2000, 4000],
            },
            "tier_execution_mode": mode,
            "resume_continuity_plan_sha256": None,
            "retry_selection_by_metric": False,
            "adaptive_step_extension": False,
            "stop_extension_rule": "new PI gate required for any unregistered extension",
            "optimizer_identity": "adamw-v1",
            "scheduler_identity": "cosine-v1",
            "scheduler_horizon_policy": "future-pi-bound",
            "precision_identity": "bf16-v1",
            "effective_global_batch_identity": "global-batch-1",
            "trainable_module_identity": "projector+diffusion",
        },
        "required_preflight_evidence": {
            "train_decision_packet_sha256": "3" * 64,
            "checkpoint_selection_ledger_sha256": "e" * 64,
            "recipe_freeze_ledger_sha256": "4" * 64,
            "scaling_admission_sha256": "5" * 64,
            "resource_scaling_audit_sha256": "6" * 64,
            "compute_budget_sha256": "7" * 64,
            "exposure_instrumentation_plan_sha256": "8" * 64,
            "dataset_support_audit_sha256": "9" * 64,
            "storage_retention_plan_sha256": "a" * 64,
        },
        "resource": {
            "resource_selected": True,
            "device_role": "FUTURE_PI_SELECTED_GPU",
            "device_count": 1,
            "effective_batch_semantics_frozen": True,
            "precision_semantics_frozen": True,
        },
        "storage": {
            "required_bytes_estimate": 1000,
            "available_bytes_verified": 2000,
            "retain_all_valid_runs": True,
        },
        "exposure": {
            "optimizer_step_bound_trace_planned": True,
            "epoch_claim_from_steps": False,
        },
    }
    return value


def test_independent_fresh_tiers_must_not_claim_resume_plan() -> None:
    request = _request("INDEPENDENT_FRESH_RUNS")
    result = campaign.analyze(request)
    assert result["schema"] == "STAGE45F_LARGE_SCALE_CAMPAIGN_ADMISSION_V3"
    assert result["tier_execution_mode"] == "INDEPENDENT_FRESH_RUNS"
    assert result["resume_continuity_plan_sha256"] is None
    assert result["checkpoint_steps_by_tier"]["2000"] == [1000, 2000]
    assert result["validation_lineage"]["state"] == "NOT_READ"
    assert result["training_authorized"] is False


def test_resume_chain_requires_continuity_plan_hash() -> None:
    request = _request("RESUME_CHAIN")
    with pytest.raises(campaign.LargeScaleAdmissionError, match="resume_continuity_plan_sha256"):
        campaign.analyze(request)

    request["training"]["resume_continuity_plan_sha256"] = "b" * 64
    result = campaign.analyze(request)
    assert result["tier_execution_mode"] == "RESUME_CHAIN"
    assert result["resume_continuity_plan_sha256"] == "b" * 64
    assert result["training_authorized"] is False


def test_checkpoint_cannot_exceed_its_run_tier() -> None:
    request = _request("INDEPENDENT_FRESH_RUNS")
    request["training"]["checkpoint_steps_by_tier"]["2000"] = [1000, 3000]
    with pytest.raises(campaign.LargeScaleAdmissionError, match="exceeds run tier"):
        campaign.analyze(request)


def test_each_run_tier_must_retain_terminal_checkpoint() -> None:
    request = _request("INDEPENDENT_FRESH_RUNS")
    request["training"]["checkpoint_steps_by_tier"]["2000"] = [1000]
    with pytest.raises(campaign.LargeScaleAdmissionError, match="terminal checkpoint"):
        campaign.analyze(request)


def test_prior_validation_is_bound_but_does_not_validate_new_scaled_checkpoint() -> None:
    result = campaign.analyze(_request("INDEPENDENT_FRESH_RUNS", validation_state="READ_WITH_LEDGER"))
    assert result["validation_lineage"]["access_generation"] == 1
    assert result["prior_validation_supports_direction_not_new_scaled_checkpoint"] is True
    assert result["new_scaled_terminal_checkpoint_requires_new_validation_generation_before_unseen_claim"] is True
    assert result["validation_execution_authorized"] is False
    assert result["holdout_f_read"] is False


def test_validation_read_without_ledger_is_rejected() -> None:
    request = _request("INDEPENDENT_FRESH_RUNS", validation_state="READ_WITH_LEDGER")
    request["validation"]["validation_access_ledger_sha256"] = None
    with pytest.raises(campaign.LargeScaleAdmissionError, match="validation_access_ledger_sha256"):
        campaign.analyze(request)


def test_expanded_dataset_requires_full_preflight_bundle() -> None:
    request = _request("INDEPENDENT_FRESH_RUNS")
    request["candidate"]["dataset_lineage_mode"] = "EXPANDED_TRAIN"
    with pytest.raises(campaign.LargeScaleAdmissionError, match="expanded_data_preflight"):
        campaign.analyze(request)

    request["expanded_data_preflight"] = {
        "dataset_union_audit_sha256": "1" * 64,
        "exact_duplicate_audit_sha256": "2" * 64,
        "coverage_balance_audit_sha256": "3" * 64,
        "sampling_mix_audit_sha256": "4" * 64,
        "active_channel_support_audit_sha256": "5" * 64,
        "data_compute_scaling_plan_sha256": "6" * 64,
        "normalization_stats_route_decision_sha256": "7" * 64,
    }
    result = campaign.analyze(request)
    assert result["dataset_lineage_mode"] == "EXPANDED_TRAIN"
    assert result["expanded_data_preflight"]["exact_duplicate_audit_sha256"] == "2" * 64
    assert result["training_authorized"] is False
