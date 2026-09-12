"""CPU/filesystem tests for contract-binding recovered L1 training evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_l1_612326_contract_gate as gate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_contract() -> dict:
    return {
        "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
        "normalization_unchanged": True,
        "stats_sha256": gate.FULL24_STATS_SHA256,
        "mask_schema": "T16_BINARY_UNCHANGED",
        "mask_shape": [40, 132],
        "active_scalar_count": 190,
        "xyz_rows": [0, 14],
        "xyz_channels": [0, 1, 2],
        "xyz_loss_weights": dict(gate.XYZ_WEIGHTS),
        "other_active_weight": 1.0,
        "inactive_weight": 1.0,
        "weight_matrix_sha256": gate.L1_MATRIX_SHA256,
        "formula": gate.FORMULA,
        "mask_folding": False,
        "adaptive_renormalization": False,
    }


def _code_contract() -> dict:
    return {
        "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
        "weight_shape": [40, 132],
        "xyz_rows": [0, 14],
        "xyz_channels": [0, 1, 2],
        "xyz_loss_weights": dict(gate.XYZ_WEIGHTS),
        "all_other_active_dimensions_weight": 1.0,
        "inactive_dimensions_weight": 1.0,
        "mask_schema": "T16_BINARY_UNCHANGED",
        "active_scalar_count": 190,
        "denominator": "binary_action_mask.sum()",
        "mask_folding": False,
        "adaptive_renormalization": False,
        "mean_weight_normalization": False,
        "matrix_sha256": gate.L1_MATRIX_SHA256,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    training = tmp_path / "training"
    training.mkdir(parents=True)
    runtime_side = {
        "l1_contract": _runtime_contract(),
        "training_recipe": dict(gate.EXPECTED_TRAINING_RECIPE),
    }
    _write_json(
        training / "stage45e_l1_runtime_binding.json",
        {"binding": runtime_side, "observed": runtime_side},
    )
    _write_json(
        training / "stage45e_l1_input_audit.json",
        {
            "schema": "VANDA_STAGE45E_L1_INPUT_AUDIT_V1",
            "status": "PASS",
            "code_contract": _code_contract(),
            "training_recipe": dict(gate.EXPECTED_TRAINING_RECIPE),
            "identities": {
                "full24_stats_sha256": gate.FULL24_STATS_SHA256,
                "full24_stats_payload_sha256": gate.FULL24_STATS_PAYLOAD_SHA256,
                "normalization_unchanged": True,
            },
        },
    )
    acceptance = {
        "schema": "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1",
        "status": "PASS_READONLY_TRAINING_ARTIFACT_INTEGRITY",
        "token": gate.BASE_ACCEPT_TOKEN,
        "training_root": str(training.resolve()),
        "historical_training_head": gate.TRAINING_HEAD,
        "branch": gate.BRANCH,
        "structured_stats": {
            "statistics_payload_sha256": gate.FULL24_STATS_PAYLOAD_SHA256,
            "mask_active_scalar_count": 190,
        },
        "scientific_claim_boundary": {
            "runtime_effective_weight_dtype_captured": False,
            "runtime_effective_weight_evidence_status": "NOT_CAPTURED_IN_HISTORICAL_CANARY",
        },
        "checkpoint_1000": {"path": str(training / "checkpoint-1000"), "global_step": 1000, "tree_sha256": "1" * 64},
        "checkpoint_2000": {"path": str(training / "checkpoint-2000"), "global_step": 2000, "tree_sha256": "2" * 64},
    }
    acceptance_path = tmp_path / "base-acceptance.json"
    _write_json(acceptance_path, acceptance)
    return training, acceptance_path


def test_plan_freezes_l1_contract_without_execution() -> None:
    plan = gate.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["expected_xyz_loss_weights"] == gate.XYZ_WEIGHTS
    assert plan["expected_weight_matrix_sha256"] == gate.L1_MATRIX_SHA256
    assert plan["historical_root_mutation_allowed"] is False
    assert plan["evaluation_authorized"] is False


def test_contract_gate_binds_runtime_and_input_audit(tmp_path: Path) -> None:
    training, base_acceptance = _fixture(tmp_path)
    output = tmp_path / "contract-bound.json"
    result = gate.bind(base_acceptance, training, output)
    assert result["token"] == gate.FINAL_ACCEPT_TOKEN
    assert result["scientific_contract"]["l1_runtime_contract"]["xyz_loss_weights"] == gate.XYZ_WEIGHTS
    assert result["scientific_contract"]["l1_code_contract"]["denominator"] == "binary_action_mask.sum()"
    assert result["scientific_contract"]["training_recipe"]["max_steps"] == 2000
    assert result["evaluation_executed"] is False
    assert output.is_file()


def test_contract_gate_rejects_loss_weight_or_recipe_drift(tmp_path: Path) -> None:
    training, base_acceptance = _fixture(tmp_path)
    runtime_path = training / "stage45e_l1_runtime_binding.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["observed"]["l1_contract"]["xyz_loss_weights"]["x"] = 9.0
    _write_json(runtime_path, runtime)
    with pytest.raises(gate.L1ContractGateError, match="xyz_loss_weights differs"):
        gate.bind(base_acceptance, training, tmp_path / "bad-weight.json")

    training, base_acceptance = _fixture(tmp_path / "second")
    input_path = training / "stage45e_l1_input_audit.json"
    audit = json.loads(input_path.read_text(encoding="utf-8"))
    audit["training_recipe"]["max_steps"] = 4000
    _write_json(input_path, audit)
    with pytest.raises(gate.L1ContractGateError, match="max_steps differs"):
        gate.bind(base_acceptance, training, tmp_path / "bad-recipe.json")


def test_contract_gate_preserves_historical_dtype_gap(tmp_path: Path) -> None:
    training, base_acceptance = _fixture(tmp_path)
    result = gate.bind(base_acceptance, training, tmp_path / "out.json")
    contract = result["scientific_contract"]
    assert contract["runtime_effective_weight_dtype_captured"] is False
    assert contract["runtime_effective_weight_evidence_status"] == "NOT_CAPTURED_IN_HISTORICAL_CANARY"
