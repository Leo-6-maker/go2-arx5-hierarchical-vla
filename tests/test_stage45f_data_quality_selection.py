"""Synthetic tests for Stage45F data-quality and checkpoint-selection prep."""

import pytest

from scripts import stage45f_checkpoint_selection_ledger as selection
from scripts import stage45f_coverage_balance_audit as coverage
from scripts import stage45f_dataset_duplicate_audit as duplicate


def _fingerprint_record(window_id: str, payload: str, source: str, label: str, *, added: bool) -> dict:
    value = {
        "window_id": window_id,
        "source_state_observation_sha256": source * 64,
        "desired_future_eef_label_sha256": label * 64,
        "window_payload_fingerprint_sha256": payload * 64,
    }
    if added:
        value.update({
            "validation_or_holdout_origin": False,
            "label_source": "SIMULATOR_EXPERT",
        })
    return value


def test_duplicate_audit_detects_same_payload_under_new_window_id() -> None:
    result = duplicate.analyze(
        {
            "schema": "STAGE45F_DATASET_DUPLICATE_AUDIT_REQUEST_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "base_windows": [
                _fingerprint_record("base-0", "a", "b", "c", added=False),
                _fingerprint_record("base-1", "d", "e", "f", added=False),
            ],
            "added_windows": [
                _fingerprint_record("new-id-same-payload", "a", "b", "c", added=True),
                _fingerprint_record("new-unique", "1", "2", "3", added=True),
            ],
        }
    )
    assert result["added_exact_duplicates_of_base_window_ids"] == ["new-id-same-payload"]
    assert result["exact_payload_duplicate_present"] is True
    assert result["duplicate_windows_automatically_removed"] is False
    assert result["training_authorized"] is False


def test_duplicate_audit_distinguishes_label_reuse_from_full_payload_duplicate() -> None:
    result = duplicate.analyze(
        {
            "schema": "STAGE45F_DATASET_DUPLICATE_AUDIT_REQUEST_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "base_windows": [_fingerprint_record("base", "a", "b", "c", added=False)],
            "added_windows": [_fingerprint_record("new", "d", "e", "c", added=True)],
        }
    )
    assert result["exact_payload_duplicate_present"] is False
    assert result["exact_label_reuse_present"] is True
    assert result["near_duplicate_or_perceptual_similarity_evaluated"] is False


def _coverage_record(window: str, target: str, episode: str, initial_bin: str, *, added: bool) -> dict:
    value = {
        "window_id": window,
        "target_id": target,
        "source_episode_id": episode,
        "initial_state_bin": initial_bin,
    }
    if added:
        value["validation_or_holdout_origin"] = False
    return value


def test_coverage_audit_surfaces_concentrated_new_data_without_threshold() -> None:
    base = [
        _coverage_record("b0", "t0", "e0", "i0", added=False),
        _coverage_record("b1", "t1", "e1", "i1", added=False),
    ]
    added = [
        _coverage_record(f"n{i}", "t0", f"ne{i}", "i0", added=True)
        for i in range(4)
    ]
    result = coverage.analyze(
        {
            "schema": "STAGE45F_COVERAGE_BALANCE_REQUEST_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "base_windows": base,
            "added_windows": added,
        }
    )
    assert result["added"]["target_id"]["counts"] == {"t0": 4}
    assert result["combined"]["target_id"]["counts"] == {"t0": 5, "t1": 1}
    assert result["automatic_balance_threshold_used"] is False
    assert result["automatic_resampling_authorized"] is False


def _selection_request() -> dict:
    return {
        "schema": "STAGE45F_CHECKPOINT_SELECTION_REQUEST_V1",
        "selection_split": "TRAIN24",
        "validation_e_read": False,
        "holdout_f_read": False,
        "automatic_best_checkpoint_selection": False,
        "pi_selected": True,
        "selection_generation": 1,
        "prior_selection_ledger_sha256": None,
        "candidate_checkpoints": [
            {
                "label": "candidate@2000",
                "checkpoint_tree_sha256": "1" * 64,
                "optimizer_step": 2000,
                "training_seed": 2222,
                "train24_evidence_packet_sha256": "2" * 64,
            },
            {
                "label": "candidate@4000",
                "checkpoint_tree_sha256": "3" * 64,
                "optimizer_step": 4000,
                "training_seed": 2222,
                "train24_evidence_packet_sha256": "4" * 64,
            },
        ],
        "selected_label": "candidate@2000",
        "selection_rationale": "PI-selected from pre-registered TRAIN24 evidence; no automatic threshold.",
    }


def test_checkpoint_selection_requires_explicit_pi_choice_and_preserves_candidate_set() -> None:
    result = selection.build(_selection_request())
    assert result["status"] == "FROZEN_PI_SELECTION_ON_TRAIN24"
    assert result["selected_checkpoint"]["label"] == "candidate@2000"
    assert len(result["candidate_checkpoints"]) == 2
    assert result["automatic_best_checkpoint_selection"] is False
    assert result["posthoc_unregistered_checkpoint_allowed"] is False
    assert result["validation_execution_authorized"] is False


def test_checkpoint_selection_rejects_unregistered_posthoc_choice() -> None:
    request = _selection_request()
    request["selected_label"] = "candidate@8000"
    with pytest.raises(selection.CheckpointSelectionError, match="registered candidate"):
        selection.build(request)


def test_checkpoint_selection_generation_two_requires_prior_ledger() -> None:
    request = _selection_request()
    request["selection_generation"] = 2
    with pytest.raises(selection.CheckpointSelectionError, match="prior_selection_ledger_sha256"):
        selection.build(request)
    request["prior_selection_ledger_sha256"] = "5" * 64
    result = selection.build(request)
    assert result["selection_generation"] == 2
    assert result["prior_selection_ledger_sha256"] == "5" * 64
