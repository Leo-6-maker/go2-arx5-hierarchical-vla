from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "pi0241_fixed16_learning.py"
SPEC = importlib.util.spec_from_file_location("pi0241_fixed16_learning", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows():
    result = []
    episode = 0
    for target in "abcd":
        for phase in range(4):
            for candidate in range(4):
                result.append(
                    {
                        "split": "TRAIN",
                        "target_family": target,
                        "phase": phase,
                        "episode_index": episode,
                        "source_episode_id": f"{target}-{phase}-{candidate}",
                    }
                )
                episode += 1
    return result


def screen_rows(xyz=0.01, so3=0.02, gate="ACCEPT"):
    return [
        {
            "source_episode_id": f"row-{index}",
            "xyz_rmse_m": xyz,
            "so3_rmse_rad": so3,
            "gate_decision": gate,
        }
        for index in range(16)
    ]


def test_selection_and_schedule_are_fixed_and_balanced():
    selected = MODULE.select_fixed16(rows())
    assert [(row["target_family"], row["phase"]) for row in selected] == [
        (target, phase) for target in "abcd" for phase in range(4)
    ]
    assert selected == MODULE.select_fixed16(list(reversed(rows())))
    schedule = MODULE.build_exposure_schedule(selected)
    assert len(schedule) == 1000
    counts = {index: sum(row["fixed_index"] == index for row in schedule) for index in range(16)}
    assert set(counts.values()) == {62, 63}
    assert schedule == MODULE.build_exposure_schedule(selected)


def test_classifier_pass_and_each_scientific_failure():
    passed = MODULE.classify_final(screen_rows(), True)
    assert passed["status"] == "PASS_FIXED16_LEARNING_FEASIBILITY"
    for field, value in (("xyz_rmse_m", 0.04), ("so3_rmse_rad", 0.06), ("gate_decision", "REJECT")):
        failed = screen_rows()
        if field == "xyz_rmse_m":
            for row in failed:
                row[field] = value
        elif field == "so3_rmse_rad":
            for row in failed:
                row[field] = value
        else:
            failed[0][field] = value
        assert MODULE.classify_final(failed, True)["status"] == "SCIENCE_FAIL_FIXED16_LEARNING"


def test_classifier_rejects_missing_duplicate_nonfinite_and_bad_integrity():
    assert MODULE.classify_final(screen_rows()[:-1], True)["status"] == "INVALID_INCOMPLETE"
    duplicate = screen_rows()
    duplicate[-1]["source_episode_id"] = duplicate[0]["source_episode_id"]
    assert MODULE.classify_final(duplicate, True)["status"] == "INVALID_INCOMPLETE"
    nonfinite = screen_rows()
    nonfinite[0]["xyz_rmse_m"] = float("nan")
    assert MODULE.classify_final(nonfinite, True)["status"] == "INVALID_INCOMPLETE"
    assert MODULE.classify_final(screen_rows(), False)["status"] == "INVALID_INCOMPLETE"


def test_warn_only_can_pass_and_step1000_is_not_replaced():
    assert MODULE.classify_final(screen_rows(gate="WARN"), True)["status"] == "PASS_FIXED16_LEARNING_FEASIBILITY"
    early = MODULE.classify_final(screen_rows(), True)
    final = MODULE.classify_final(screen_rows(xyz=0.04), True)
    assert early["status"].startswith("PASS")
    assert final["status"].startswith("SCIENCE_FAIL")


def test_parameter_groups_match_authorized_trainables():
    assert MODULE._parameter_group("action_head.state_encoder.layer1.W") == "projector"
    assert MODULE._parameter_group("action_head.model.transformer_blocks.0.attn1.to_q.weight") == "diffusion"
    assert MODULE._parameter_group("action_head.vlln.weight") == "frozen_vlln"
    assert MODULE._parameter_group("backbone.model.model.visual.x") == "frozen_backbone"
