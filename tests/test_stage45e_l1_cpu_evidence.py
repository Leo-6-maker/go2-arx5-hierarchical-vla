"""CPU/static tests for post-evaluation seal and paired-evidence validation."""

from __future__ import annotations

from pathlib import Path

from scripts import stage45e_l1_cpu_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_evidence_contract_is_read_only_and_exact_budget() -> None:
    assert evidence.FINAL_TOKEN == "STAGE45E_L1_VANDA_H200_TRAIN24_COMPLETE_FOR_PI_REVIEW"
    assert evidence.EXPECTED_COUNTS["conditioning_passes"] == 72
    assert evidence.EXPECTED_COUNTS["action_head_calls"] == 288
    assert evidence.EXPECTED_COUNTS["model_forward"] == 288
    assert evidence.EXPECTED_COUNTS["raw_prediction_count"] == 72
    assert evidence.EXPECTED_COUNTS["training_steps"] == 0
    assert evidence.EXPECTED_COUNTS["backward_steps"] == 0
    assert evidence.EXPECTED_COUNTS["optimizer_steps"] == 0
    assert evidence.EXPECTED_COUNTS["checkpoint_writes"] == 0


def test_cpu_evidence_source_does_not_import_model_runtime() -> None:
    source = (ROOT / "scripts/stage45e_l1_cpu_evidence.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import gr00t" not in source
    assert "subprocess" not in source
    assert "PASS_READONLY_SEAL_AND_PAIRED_EVIDENCE" in source
    assert "final_seal_excluded" in source
    assert "final_result_excluded" in source
