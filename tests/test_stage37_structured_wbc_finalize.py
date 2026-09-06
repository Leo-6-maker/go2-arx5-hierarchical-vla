"""Tests for the read-only Stage37 structured-oracle adjudicator."""

from __future__ import annotations

import hashlib
import json
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


def test_verify_seal_accepts_only_append_only_wrapper_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "stage36_job.log"
    sealed_prefix = b"campaign output before tee drain\n"
    appended_tail = b"STAGE37_CAMPAIGN_RC=139\nSTAGE37_RUN_ROOT=/scratch/run\n"
    log_path.write_bytes(sealed_prefix + appended_tail)
    seal_path = tmp_path / "stage36_artifact_seal.json"
    seal_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "path": str(log_path),
                        "relative_path": "stage36_job.log",
                        "size_bytes": len(sealed_prefix),
                        "sha256": hashlib.sha256(sealed_prefix).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    strict = finalize.verify_seal(tmp_path, seal_path)
    repaired = finalize.verify_seal(
        tmp_path, seal_path, allow_append_only_log_tail=True
    )

    assert strict["status"] == "BLOCKED_SEAL_ENTRY_MISMATCH"
    assert repaired["status"] == "PASS_APPEND_ONLY_LOG_TAIL"
    assert repaired["entries_verified"] == 1
    assert repaired["append_only_log_tails"][0]["appended_size_bytes"] == len(
        appended_tail
    )
