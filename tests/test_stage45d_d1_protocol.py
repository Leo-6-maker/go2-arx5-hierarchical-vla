"""CPU-only tests for the corrected Stage45D D1 protocol contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from go2arx5_vla.structured_action import structured_action_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage45d_d1_protocol_static import (
    MANIFEST,
    ProtocolContractError,
    reject_legacy_first_ten_mask,
    validate_mask,
    validate_protocol,
)


def test_corrected_protocol_passes_cpu_contract() -> None:
    result = validate_protocol(MANIFEST)
    assert result["verdict"] == "PASS"
    assert result["mask_contract"]["shape"] == [40, 132]
    assert result["mask_contract"]["active_scalar_count"] == 190
    assert result["mask_contract"]["inactive_scalar_count"] == 5090
    assert result["legacy_first_ten_mask_negative_test"]["rejected"] is True
    assert result["h_sampler"] == "NOT_TESTED"
    assert result["h_opt"] == "SUPPORTED"
    assert result["zero_execution_boundary"] is True


def test_mask_is_exact_repository_t16_mask_and_block_partition() -> None:
    mask = structured_action_mask("T16")
    result = validate_mask(mask)
    assert mask.shape == (40, 132)
    assert np.count_nonzero(mask) == 190
    assert result["block_counts"] == {
        "geometry": 135,
        "timing": 15,
        "gripper": 40,
        "remaining": 0,
    }


def test_legacy_stage26_dense_first_ten_mask_is_rejected() -> None:
    outcome = reject_legacy_first_ten_mask()
    assert outcome["rejected"] is True
    assert outcome["legacy_active_scalar_count"] == 400

    legacy = np.zeros((40, 132), dtype=np.float64)
    legacy[:, :10] = 1.0
    with pytest.raises(ProtocolContractError, match="structured_action_mask"):
        validate_mask(legacy)


def test_corrected_manifest_has_no_legacy_topology_or_loaded_taxonomy() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    for marker in (
        "ACTIVE_DIM=10",
        "INACTIVE_DIM=122",
        "190 active / 122",
        '"persistent-good"',
        '"persistent-negative"',
        '"recovered-at-2000"',
        '"regressed-at-2000"',
        '"mixed-sign-or-zero"',
    ):
        assert marker not in text


def test_metric_spaces_and_zero_boundary_are_machine_readable() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["normalized_training_objective_space"]["name"] == "NORMALIZED_LOCAL_FLOW"
    assert metrics["physical_interpretation_space"]["name"] == "DECODED_PHYSICAL_ACTION"
    assert metrics["raw_before_derived"] is True
    assert payload["hypothesis_ranking_correction"]["ranking"][4]["status"] == "NOT_TESTED"
    assert payload["execution_boundary"]["counters"] == {
        "model_construction": 0,
        "model_forward": 0,
        "gpu": 0,
        "pbs": 0,
        "training_steps": 0,
        "validation_e_rows_read": 0,
        "holdout_f_rows_read": 0,
        "wbc_steps": 0,
    }
