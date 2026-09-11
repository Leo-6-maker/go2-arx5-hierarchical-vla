"""Synthetic regression tests for Stage45F scaling/instrumentation prep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from go2arx5_vla.stage45f_exposure_sidechannel import (
    ExposureSideChannel,
    ExposureSideChannelError,
    record_fetch_without_payload_mutation,
)
from scripts import stage45f_active_channel_support_audit as active_support
from scripts import stage45f_data_compute_scaling_plan as data_compute
from scripts import stage45f_large_scale_campaign_admission as large_scale
from scripts import stage45f_n2_l1_factorial_plan as factorial
from scripts import stage45f_optimizer_exposure_binding as optimizer_exposure


def test_fetch_sidechannel_returns_same_payload_and_requires_seal(tmp_path: Path) -> None:
    trace = tmp_path / "rank0.fetch.jsonl"
    seal = tmp_path / "rank0.fetch.seal.json"
    recorder = ExposureSideChannel(trace, rank=0, buffer_event_limit=2)
    payload = {"action": object(), "state": object()}
    returned = record_fetch_without_payload_mutation(
        recorder,
        payload,
        dataset_identity="FULL24",
        episode_index=0,
        source_episode_index=10,
        step_index=3,
        window_id="w0",
    )
    assert returned is payload
    assert recorder.sealed is False
    result = recorder.seal(seal)
    assert recorder.sealed is True
    assert result["status"] == "SEALED_FETCH_TRACE_ONLY"
    assert result["event_count"] == 1
    assert result["exact_fetch_count_claim_allowed"] is True
    assert result["optimizer_step_exposure_claim_allowed"] is False
    assert result["epoch_claim_allowed"] is False
    with pytest.raises(ExposureSideChannelError, match="sealed"):
        recorder.record_fetch(
            dataset_identity="FULL24",
            episode_index=0,
            source_episode_index=10,
            step_index=4,
            window_id="w1",
        )


def test_fetch_sidechannel_is_rank_bound_and_fresh(tmp_path: Path) -> None:
    trace = tmp_path / "rank1.fetch.jsonl"
    recorder = ExposureSideChannel(trace, rank=1)
    event = recorder.record_fetch(
        dataset_identity="FULL24",
        episode_index=1,
        source_episode_index=11,
        step_index=0,
        window_id="w1",
    )
    assert event["rank"] == 1
    assert event["optimizer_step"] is None
    assert event["optimizer_step_binding"] == "NOT_CAPTURED_BY_DATASET_SIDECHANNEL"
    with pytest.raises(ExposureSideChannelError, match="fresh"):
        ExposureSideChannel(trace, rank=1)


def _optimizer_event(step: int, rank: int, ids: list[str], *, world_size: int = 2) -> dict:
    return {
        "schema": optimizer_exposure.EVENT_SCHEMA,
        "optimizer_step": step,
        "rank": rank,
        "world_size": world_size,
        "gradient_accumulation_steps": 2,
        "sample_window_ids": ids,
        "microbatch_sample_counts": [1, len(ids) - 1],
        "split": "TRAIN",
        "validation_e": False,
        "holdout_f": False,
        "trainer_boundary": "AFTER_SUCCESSFUL_OPTIMIZER_STEP",
        "optimizer_step_applied": True,
    }


def test_optimizer_step_exposure_requires_all_ranks_and_counts_exactly(tmp_path: Path) -> None:
    trace = tmp_path / "optimizer.jsonl"
    rows = [
        _optimizer_event(1, 0, ["w0", "w1"]),
        _optimizer_event(1, 1, ["w2", "w3"]),
        _optimizer_event(2, 0, ["w0", "w1"]),
        _optimizer_event(2, 1, ["w2", "w3"]),
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = optimizer_exposure.analyze(
        optimizer_exposure.read_events(trace), optimizer_exposure.sha256_file(trace)
    )
    assert result["optimizer_step_count"] == 2
    assert result["world_size"] == 2
    assert result["global_sample_exposure_count"] == 8
    assert result["per_window_counts"] == {"w0": 2, "w1": 2, "w2": 2, "w3": 2}
    assert result["microbatch_sample_count_closure_verified"] is True
    assert result["exact_optimizer_step_exposure_available"] is True
    assert result["epoch_claim"] == "NOT_DERIVED_AUTOMATICALLY"
    assert result["training_authorized"] is False

    bad = tmp_path / "missing-rank.jsonl"
    bad.write_text(json.dumps(_optimizer_event(1, 0, ["w0", "w1"])) + "\n", encoding="utf-8")
    with pytest.raises(optimizer_exposure.OptimizerExposureError, match="one event per rank"):
        optimizer_exposure.analyze(
            optimizer_exposure.read_events(bad), optimizer_exposure.sha256_file(bad)
        )


def test_optimizer_step_exposure_rejects_incomplete_microbatch_serialization(tmp_path: Path) -> None:
    trace = tmp_path / "incomplete.jsonl"
    row = _optimizer_event(1, 0, ["w0", "w1"], world_size=1)
    row["microbatch_sample_counts"] = [1, 2]
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(optimizer_exposure.OptimizerExposureError, match="do not close"):
        optimizer_exposure.read_events(trace)


def test_data_cardinality_scaling_is_nominal_not_epoch_equivalence() -> None:
    result = data_compute.plan(
        {
            "schema": "STAGE45F_DATA_COMPUTE_SCALING_REQUEST_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "sampling_assumption": "UNIFORM_ONE_WINDOW_EVENT_PER_OPTIMIZER_STEP",
            "base_window_count": 24,
            "expanded_window_count": 48,
            "base_optimizer_steps": 2000,
            "proposed_fixed_optimizer_steps": 2000,
        }
    )
    assert result["fixed_step_expanded_data"]["relative_nominal_per_window_exposure_vs_base"] == pytest.approx(0.5)
    assert result["cardinality_scaled_hypothetical"]["nearest_integer_optimizer_steps"] == 4000
    assert result["epoch_claim_allowed"] is False
    assert result["exact_realized_exposure_claim_allowed"] is False
    assert result["training_authorized"] is False


def test_factorial_plan_never_auto_authorizes_n2_plus_l1() -> None:
    result = factorial.plan(
        {
            "schema": "STAGE45F_N2_L1_FACTORIAL_REQUEST_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "existing_cells": {
                "A_LEGACY": {"valid_training_seeds": [2222], "recipe_identity": "legacy"},
                "B_N2": {"valid_training_seeds": [2222, 2223, 2224], "recipe_identity": "n2"},
                "C_L1": {"valid_training_seeds": [2222], "recipe_identity": "l1"},
            },
        }
    )
    assert result["combined_training_authorized"] is False
    assert result["automatic_combination_allowed"] is False
    assert result["common_single_factor_seeds"] == [2222]
    assert result["missing_seeds_for_fully_matched_single_factor_context"] == [2223, 2224]


def test_active_channel_support_covers_all_t16_semantics_without_threshold() -> None:
    channels = {}
    for name in active_support.EXPECTED_CHANNELS:
        channels[name] = {
            "active_rows": active_support.EXPECTED_ACTIVE_ROWS[name],
            "center": 0.0,
            "half_range": 1.0,
        }
    values = {
        name: [0.0] * active_support.EXPECTED_ACTIVE_ROWS[name]
        for name in active_support.EXPECTED_CHANNELS
    }
    values["rot6d_0"][0] = 1.5
    result = active_support.analyze(
        {
            "schema": "STAGE45F_ACTIVE_CHANNEL_SUPPORT_PROPOSAL_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "channels": channels,
            "examples": [{"validation_or_holdout_origin": False, "values": values}],
        }
    )
    assert result["group_summary"]["ROT6D"]["outside_neg1_pos1_count"] == 1
    assert result["group_summary"]["XYZ"]["outside_neg1_pos1_count"] == 0
    assert result["automatic_acceptability_threshold_used"] is False
    assert result["training_authorized"] is False


def _large_scale_request() -> dict:
    return {
        "schema": "STAGE45F_LARGE_SCALE_CAMPAIGN_REQUEST_V1",
        "selection_source_split": "TRAIN24",
        "validation_e_read": False,
        "holdout_f_read": False,
        "recipe_frozen": True,
        "pi_selected_candidate": True,
        "candidate": {
            "candidate_label": "future-candidate",
            "training_recipe_identity": "recipe-v1",
            "dataset_tree_sha256": "1" * 64,
            "normalization_identity": "stats-v1",
            "loss_identity": "loss-v1",
            "base_model_tree_sha256": "2" * 64,
            "sampler_identity": "sampler-v1",
            "decoder_identity": "decoder-v1",
            "metric_identity": "metrics-v1",
        },
        "training": {
            "predeclared_seeds": [2222, 2223, 2224],
            "optimizer_step_tiers": [2000, 4000, 8000],
            "checkpoint_steps": [2000, 4000, 8000],
            "retry_selection_by_metric": False,
            "adaptive_step_extension": False,
            "stop_extension_rule": "No tier extension without a new PI gate after the predeclared tier is sealed.",
            "optimizer_identity": "adamw-v1",
            "scheduler_identity": "cosine-v1",
            "scheduler_horizon_policy": "future-pi-bound",
            "precision_identity": "bf16-v1",
            "effective_global_batch_identity": "global-batch-1",
            "trainable_module_identity": "projector+diffusion",
        },
        "required_preflight_evidence": {
            "train_decision_packet_sha256": "3" * 64,
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


def test_large_scale_campaign_is_structurally_admitted_but_never_auto_authorized() -> None:
    result = large_scale.analyze(_large_scale_request())
    assert result["status"] == "PASS_STRUCTURE_ONLY_PI_EXECUTION_GATE_REQUIRED"
    assert result["decision_state"] == "PI_GATE_REQUIRED"
    assert result["training_authorized"] is False
    assert result["gpu_authorized"] is False
    assert result["pbs_authorized"] is False
    assert result["metric_based_retry_selection_allowed"] is False
    assert result["adaptive_extension_allowed_without_new_pi_gate"] is False


def test_large_scale_campaign_rejects_insufficient_storage() -> None:
    request = _large_scale_request()
    request["storage"]["available_bytes_verified"] = 999
    with pytest.raises(large_scale.LargeScaleAdmissionError, match="storage"):
        large_scale.analyze(request)
