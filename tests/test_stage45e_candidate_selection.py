"""CPU/static tests for PI0128 neutral candidate-selection tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_candidate_selection as selection


def _normalized_fixture() -> dict:
    cfg = selection.manifest()
    windows = [f"window-{index:02d}" for index in range(cfg["window_count"])]
    candidates = {}
    for candidate_index, label in enumerate(cfg["candidate_order"]):
        rows = []
        for index, window_id in enumerate(windows):
            rows.append(
                {
                    "window_id": window_id,
                    "target_id": f"target-{index // 6}",
                    "metrics": {
                        "final_target_distance_m": 0.10 + 0.001 * candidate_index + 0.0001 * index,
                        "absolute_progress_m": 0.04 - 0.001 * candidate_index + 0.0001 * index,
                        "minimum_target_distance_m": 0.08 + 0.001 * candidate_index + 0.0001 * index,
                        "endpoint_target_vector_cosine": None if index == 0 else 0.5 + 0.001 * candidate_index,
                        "path_length_m": 1.0 + 0.01 * candidate_index + 0.001 * index,
                        "roughness_position_second_difference_m": 0.03 + 0.0001 * candidate_index,
                        "progress_sign": 1 if index % 3 else -1,
                    },
                }
            )
        candidates[label] = {
            "provenance": {
                "source_id": label,
                "sha256": f"{candidate_index + 1:064x}",
            },
            "rows": rows,
        }
    return {
        "schema": cfg["normalized_input_schema"],
        "split": "TRAIN24",
        "window_ids": windows,
        "validation_e_read": False,
        "holdout_f_read": False,
        "candidates": candidates,
    }


def test_plan_is_static_and_decision_neutral() -> None:
    plan = selection.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["decision_state"] == "PI_DECISION_REQUIRED"
    assert plan["claim_boundary"]["automatic_winner"] is False
    assert plan["optional_numeric_metrics"] == ["endpoint_target_vector_cosine"]
    assert plan["static_guard"]["model_construction"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0
    assert plan["static_guard"]["validation_e_read"] is False
    assert plan["static_guard"]["holdout_f_read"] is False


def test_validation_plan_is_fail_closed() -> None:
    plan = selection.validation_plan()
    assert plan["status"] == "LOCKED_PLAN_ONLY"
    assert plan["execute_supported"] is False
    assert plan["validation_e_read"] is False
    assert plan["holdout_f_read"] is False
    assert plan["decision_state"] == "PI_DECISION_REQUIRED"


def test_normalized_fixture_builds_descriptive_packet_only() -> None:
    normalized = selection.validate_normalized_input(_normalized_fixture())
    packet = selection.build_packet(normalized, "0" * 64)
    assert packet["status"] == "PASS_DESCRIPTIVE_TRAIN24_ONLY"
    assert packet["decision_state"] == "PI_DECISION_REQUIRED"
    assert packet["winner_declared"] is False
    assert packet["automatic_selection_performed"] is False
    assert packet["iid_significance_test_performed"] is False
    assert packet["validation_e_authorized"] is False
    assert packet["holdout_f_authorized"] is False
    assert len(packet["candidate_summaries"]) == 9
    assert len(packet["paired_comparisons"]) == 8
    assert packet["execution_counters"]["model_forward"] == 0
    assert packet["execution_counters"]["gpu"] == 0
    cosine = packet["candidate_summaries"]["LEGACY FULL24@2000"]["numeric"]["endpoint_target_vector_cosine"]
    assert cosine["count"] == 23
    assert cosine["missing_count"] == 1


def test_pair_delta_and_sign_transition_are_left_minus_right() -> None:
    cfg = selection.manifest()
    normalized = selection.validate_normalized_input(_normalized_fixture())
    packet = selection.build_packet(normalized, "1" * 64)
    name = "L1@2000_MINUS_L1@1000"
    pair = packet["paired_comparisons"][name]["summary"]
    # Fixture candidate order makes L1@2000 one index larger than L1@1000.
    assert pair["overall_numeric_delta"]["final_target_distance_m"]["mean_left_minus_right"] == pytest.approx(0.001)
    assert pair["overall_numeric_delta"]["absolute_progress_m"]["mean_left_minus_right"] == pytest.approx(-0.001)
    cosine = pair["overall_numeric_delta"]["endpoint_target_vector_cosine"]
    assert cosine["count"] == 23
    assert cosine["missing_pair_count"] == 1
    assert sum(pair["progress_sign_transition_counts_right_to_left"].values()) == cfg["window_count"]


def test_rejects_missing_candidate_window_or_boolean_sign() -> None:
    fixture = _normalized_fixture()
    fixture["candidates"].pop("L1@2000")
    with pytest.raises(selection.CandidateSelectionError, match="candidate order/set drift"):
        selection.validate_normalized_input(fixture)

    fixture = _normalized_fixture()
    fixture["candidates"]["L1@2000"]["rows"].pop()
    with pytest.raises(selection.CandidateSelectionError, match="row count drift"):
        selection.validate_normalized_input(fixture)

    fixture = _normalized_fixture()
    fixture["candidates"]["L1@2000"]["rows"][0]["metrics"]["progress_sign"] = True
    with pytest.raises(selection.CandidateSelectionError, match="must not be boolean"):
        selection.validate_normalized_input(fixture)


def test_build_from_file_writes_fresh_packet(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.json"
    output_path = tmp_path / "packet.json"
    input_path.write_text(json.dumps(_normalized_fixture(), sort_keys=True), encoding="utf-8")
    packet = selection.build_from_file(input_path, output_path)
    assert output_path.is_file()
    observed = json.loads(output_path.read_text(encoding="utf-8"))
    assert observed["schema"] == packet["schema"]
    assert observed["decision_state"] == "PI_DECISION_REQUIRED"
    with pytest.raises(selection.CandidateSelectionError, match="output must be fresh"):
        selection.build_from_file(input_path, output_path)
