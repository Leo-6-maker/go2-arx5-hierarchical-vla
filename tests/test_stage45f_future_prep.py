"""Synthetic CPU tests for PI0129 Stage45F preparation-only tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45f_data_expansion_audit as data_audit
from scripts import stage45f_exposure_accounting as exposure
from scripts import stage45f_future_experiment_planner as planner
from scripts import stage45f_rollout_relabel_admission as relabel
from scripts import stage45f_scaling_admission as scaling


def _scaling_recipe(*, max_steps: int, horizon: int, binding: str = "MAX_STEPS", start_mode: str = "FRESH_BASE") -> dict:
    value = {
        "recipe_id": f"recipe-{max_steps}-{horizon}-{binding}-{start_mode}",
        "dataset_tree_sha256": "d" * 64,
        "normalization_identity": "stats-v1",
        "loss_identity": "loss-v1",
        "optimizer": "adamw_torch",
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "max_steps": max_steps,
        "scheduler_horizon_binding": binding,
        "scheduler_horizon_steps": horizon,
        "seed": 2222,
        "start_mode": start_mode,
        "exact_sample_exposure": {
            "samples_seen": max_steps,
            "per_window_counts_available": True,
        },
    }
    if start_mode == "RESUME_CHECKPOINT":
        value["resume_state"] = {
            "checkpoint_tree_sha256": "a" * 64,
            "optimizer_state_sha256": "b" * 64,
            "scheduler_state_sha256": "c" * 64,
            "rng_state_sha256": "e" * 64,
        }
    return value


def test_future_planner_is_prep_only() -> None:
    value = planner.plan()
    assert value["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert value["execution_supported"] is False
    assert value["decision_state"] == "PI_DECISION_REQUIRED"
    assert value["static_guard"]["model_forward"] == 0
    assert value["static_guard"]["gpu"] == 0
    assert value["static_guard"]["validation_e_read"] is False


def test_future_planner_surfaces_replication_exposure_and_rollout_scenarios() -> None:
    summary = planner.validate_summary(
        {
            "schema": "STAGE45F_PI_SUPPLIED_TRAIN_EVIDENCE_SUMMARY_V1",
            "split": "TRAIN24",
            "validation_e_read": False,
            "holdout_f_read": False,
            "selected_candidate": "L1",
            "recipe_frozen": True,
            "cross_seed": {"valid_seed_count": 1, "consistent_direction": None},
            "exposure": {"trend_from_1000_to_2000": "IMPROVED", "exact_sample_accounting_available": False},
            "data_support": {
                "target_heterogeneity": "HIGH",
                "support_gap_evidence": True,
                "current_stats_clipping_fraction_for_proposed_new_data": 0.1,
            },
            "rollout": {"teacher_forced_good": True, "free_running_gap": True},
        }
    )
    matrix = planner.scenario_matrix(summary)
    relevant = {item["scenario"]: item["relevant"] for item in matrix["scenarios"]}
    assert relevant["SEED_REPLICATION"] is True
    assert relevant["EXPOSURE_SCALING"] is True
    assert relevant["DATA_EXPANSION"] is True
    assert relevant["ROLLOUT_RELABEL"] is True
    assert matrix["automatic_experiment_selection"] is False


def test_cosine_max_steps_change_is_not_pure_exposure() -> None:
    baseline = scaling._recipe(_scaling_recipe(max_steps=2000, horizon=2000), "baseline")
    proposal = scaling._recipe(_scaling_recipe(max_steps=4000, horizon=4000), "proposal")
    result = scaling.compare(baseline, proposal)
    assert result["classification"] == "SCALING_CONFOUNDED_OR_UNPROVEN"
    assert "cosine_lr_history_changes_when_max_steps_changes" in result["confounds"]
    assert result["epoch_claim_allowed"] is False


def test_fixed_scheduler_horizon_beyond_budget_requires_policy() -> None:
    baseline = scaling._recipe(_scaling_recipe(max_steps=2000, horizon=2000, binding="EXPLICIT_FIXED"), "baseline")
    proposal = scaling._recipe(_scaling_recipe(max_steps=4000, horizon=2000, binding="EXPLICIT_FIXED"), "proposal")
    result = scaling.compare(baseline, proposal)
    assert "training_extends_beyond_explicit_scheduler_horizon_requires_defined_post_horizon_policy" in result["confounds"]
    assert result["classification"] == "SCALING_CONFOUNDED_OR_UNPROVEN"


def test_resume_is_never_assumed_equivalent_to_fresh_long_run() -> None:
    baseline = scaling._recipe(_scaling_recipe(max_steps=2000, horizon=2000), "baseline")
    proposal = scaling._recipe(_scaling_recipe(max_steps=4000, horizon=4000, start_mode="RESUME_CHECKPOINT"), "proposal")
    result = scaling.compare(baseline, proposal)
    assert "fresh_vs_resume_changed" in result["confounds"]
    assert "resume_requires_continuity_proof_not_equivalence_assumption" in result["confounds"]


def _data_proposal(*, changed_stats: bool) -> dict:
    current = {
        "identity": "full24-current",
        "xyz_center_m": [0.0, 0.0, 0.0],
        "xyz_half_range_m": [0.1, 0.1, 0.1],
    }
    proposed = None
    if changed_stats:
        proposed = {
            "identity": "expanded-stats",
            "xyz_center_m": [0.0, 0.0, 0.0],
            "xyz_half_range_m": [0.2, 0.2, 0.2],
        }
    return {
        "schema": "STAGE45F_DATA_EXPANSION_PROPOSAL_V1",
        "source_split": "TRAIN",
        "validation_e_read": False,
        "holdout_f_read": False,
        "contains_validation_e_ids": False,
        "contains_holdout_f_ids": False,
        "current_stats": current,
        "proposed_stats": proposed,
        "examples": [
            {
                "window_id": "new-window-0",
                "source_episode_id": "new-episode-0",
                "target_id": "train-target-new",
                "label_source": "SIMULATOR_EXPERT",
                "future_xyz_m": [[0.05, 0.0, 0.0], [0.15, 0.0, -0.2]],
            }
        ],
    }


def test_data_expansion_reports_clipping_without_threshold() -> None:
    value = data_audit.validate(_data_proposal(changed_stats=False))
    result = data_audit.audit(value)
    assert result["intervention_class"].startswith("DATA_ONLY_CANDIDATE")
    assert result["current_stats_support"]["xyz_scalar_outside_neg1_pos1_count"] == 2
    assert result["automatic_acceptability_threshold_used"] is False
    assert result["execution_authorized"] is False


def test_stats_change_is_named_data_plus_normalization() -> None:
    value = data_audit.validate(_data_proposal(changed_stats=True))
    result = data_audit.audit(value)
    assert result["intervention_class"] == "DATA_PLUS_NORMALIZATION"
    assert result["stats_change_detected"] is True


def _relabel_proposal(*, same_model: bool) -> dict:
    return {
        "schema": "STAGE45F_ROLLOUT_RELABEL_PROPOSAL_V1",
        "source_split": "TRAIN",
        "validation_e_read": False,
        "holdout_f_read": False,
        "state_generator": {
            "type": "GROOT_POLICY",
            "checkpoint_tree_sha256": "1" * 64,
            "recipe_identity": "candidate-recipe",
            "sampler_identity": "sampler-e4",
        },
        "labeler": {
            "type": "SIMULATOR_EXPERT",
            "is_same_model_as_state_generator": same_model,
            "implementation_identity": "expert-v1",
            "environment_identity": "sim-v1",
            "label_policy_identity": "oracle-v1",
        },
        "samples": [
            {
                "visited_state_id": "state-0",
                "oracle_label_id": "label-0",
                "source_episode_id": "train-episode-0",
                "label_generated_from_same_policy_output": same_model,
                "validation_or_holdout_origin": False,
                "state_artifact_sha256": "2" * 64,
                "label_artifact_sha256": "3" * 64,
            }
        ],
    }


def test_rollout_relabel_accepts_independent_oracle_provenance_only() -> None:
    value = relabel.validate(_relabel_proposal(same_model=False))
    result = relabel.audit(value)
    assert result["independent_supervision"] is True
    assert result["groot_self_labeling"] is False
    assert result["execution_authorized"] is False


def test_rollout_relabel_rejects_self_labeling() -> None:
    with pytest.raises(relabel.RolloutRelabelAdmissionError, match="self-labeling"):
        relabel.validate(_relabel_proposal(same_model=True))


def test_exact_exposure_trace_counts_windows_without_epoch_claim(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"schema": "STAGE45F_TRAIN_EXPOSURE_EVENT_V1", "optimizer_step": 1, "window_id": "w0", "split": "TRAIN", "validation_e": False, "holdout_f": False},
        {"schema": "STAGE45F_TRAIN_EXPOSURE_EVENT_V1", "optimizer_step": 2, "window_id": "w1", "split": "TRAIN", "validation_e": False, "holdout_f": False},
        {"schema": "STAGE45F_TRAIN_EXPOSURE_EVENT_V1", "optimizer_step": 3, "window_id": "w0", "split": "TRAIN", "validation_e": False, "holdout_f": False},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = exposure.analyze(exposure.read_trace(trace), exposure.sha256_file(trace))
    assert result["per_window_counts"] == {"w0": 2, "w1": 1}
    assert result["exact_per_window_exposure_available"] is True
    assert result["epoch_claim"] == "NOT_DERIVED_AUTOMATICALLY"
