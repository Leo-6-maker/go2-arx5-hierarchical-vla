"""Tests for the read-only Stage37 structured-oracle adjudicator."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import stage37_structured_wbc_finalize as finalize  # noqa: E402


def _profile() -> dict:
    return {
        "position_m": {"mean": 0.1, "p95": 0.2, "rmse": 0.11, "max": 0.3},
        "rotation_rad": {"mean": 0.2, "p95": 0.3, "rmse": 0.21, "max": 0.4},
    }


def test_structured_hard_flags_ignore_soft_margin_diagnostic() -> None:
    row = {
        "safety_gate": {
            "hard_failure_flags": {
                "fall": False,
                "invalid_state": False,
                "joint_limit_termination": False,
                "timeout": False,
                "tracking_abort_before_reference_completion": False,
                "native_task_termination_before_reference_completion": False,
            }
        }
    }
    assert not any(finalize.structured_hard_flags(row).values())


def test_structured_hard_flags_keep_true_joint_termination() -> None:
    row = {
        "safety_gate": {
            "hard_failure_flags": {"joint_limit_termination": True}
        }
    }
    assert finalize.structured_hard_flags(row)["joint_limit_termination"] is True


def test_pair_analysis_uses_frozen_thresholds_and_same_case_ids() -> None:
    baseline_rows = []
    structured_rows = []
    for case_id in finalize.EXPECTED_CASE_IDS:
        baseline_rows.append(
            {
                "case_id": case_id,
                "row_id": "dense_%s_seed13_primary" % case_id,
                "representation": "DENSE",
                "lifecycle_qualified": True,
                "finite": True,
                "reference_complete": True,
                "profile_metrics": _profile(),
            }
        )
        structured_rows.append(
            {
                "case_id": case_id,
                "row_id": "k16_t0_%s_seed13_primary" % case_id,
                "tracking_pass": True,
                "scientific_row_finalized": True,
                "finite": True,
                "reference_complete": True,
                "base_participation": {"nontrivial_observed": True},
                "profile_metrics": _profile(),
                "safety_gate": {"hard_failure_flags": {}},
                "independent_raw_audit": {"safety_gate": {}},
            }
        )
    result = finalize.pair_analysis(
        {"rows": baseline_rows},
        {"rows": structured_rows},
    )
    assert result["status"] == "PASS"
    assert result["geometry_oracle_parity"] == "GEOMETRY_ORACLE_PARITY_QUALIFIED"
    assert result["new_failure_counts"]["true_joint_limit_terminations"] == 0
    assert len(result["pairs"]) == 32
