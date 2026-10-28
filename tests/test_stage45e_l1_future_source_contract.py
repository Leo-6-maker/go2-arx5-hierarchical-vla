"""Synthetic static tests for future corrected Stage45E-L1 source integration."""

from pathlib import Path

import pytest

from scripts import stage45e_l1_future_source_contract as contract


def _good_train() -> str:
    return f'''\
TRAINING_ONLY_TOKEN = "{contract.TRAINING_ONLY_TOKEN}"

def capture_effective_weight_evidence(*args):
    return {{"status": "captured"}}

def execute():
    runtime_weight_evidence = capture_effective_weight_evidence(None)
    result = {{"token": TRAINING_ONLY_TOKEN, "runtime_weight_evidence": runtime_weight_evidence}}
    path = "stage45e_l1_training_result.json"
    return path, result
'''


def _good_eval() -> str:
    return f'''\
RECOVERED_SCHEMA = "{contract.RECOVERED_SCHEMA}"
FINAL = "{contract.FINAL_TOKEN}"

def validate_recovered_input(value): return value

def paired_summary_complete(value): return value

def persist_row_bundles(value): return value

def build_immutable_inventory(value): return value

def execute(recovered_eval_input):
    value = validate_recovered_input(recovered_eval_input)
    rows = persist_row_bundles(value)
    paired = paired_summary_complete(rows)
    inventory = build_immutable_inventory(rows)
    public_sampler_calls = 0
    get_action_calls = 0
    Validation_E = "Validation-E"
    Holdout_F = "Holdout-F"
    return FINAL, paired, inventory, public_sampler_calls, get_action_calls, Validation_E, Holdout_F
'''


def _write(tmp_path: Path, train: str, evaluate: str) -> tuple[Path, Path]:
    train_path = tmp_path / "train.py"
    eval_path = tmp_path / "eval.py"
    train_path.write_text(train, encoding="utf-8")
    eval_path.write_text(evaluate, encoding="utf-8")
    return train_path, eval_path


def test_future_corrected_sources_pass_synthetic_contract(tmp_path: Path) -> None:
    train, evaluate = _write(tmp_path, _good_train(), _good_eval())
    result = contract.audit(train, evaluate)
    assert result["status"] == "PASS_STATIC_SOURCE_CONTRACT"
    assert result["training_only_token_present"] is True
    assert result["runtime_effective_weight_evidence_present"] is True
    assert result["recovered_v2_input_supported"] is True
    assert result["noncyclic_inventory_present"] is True
    assert result["execution_authorized"] is False


def test_missing_training_only_token_is_rejected(tmp_path: Path) -> None:
    train, evaluate = _write(tmp_path, _good_train().replace(contract.TRAINING_ONLY_TOKEN, "WRONG"), _good_eval())
    with pytest.raises(contract.L1FutureSourceContractError, match="training-only completion token"):
        contract.audit(train, evaluate)


def test_missing_runtime_weight_capture_is_rejected(tmp_path: Path) -> None:
    train = _good_train().replace("capture_effective_weight_evidence", "other_diagnostic")
    train_path, eval_path = _write(tmp_path, train, _good_eval())
    with pytest.raises(contract.L1FutureSourceContractError, match="runtime effective L1 weights"):
        contract.audit(train_path, eval_path)


def test_missing_recovered_eval_input_is_rejected(tmp_path: Path) -> None:
    evaluate = _good_eval().replace(contract.RECOVERED_SCHEMA, "OLD_SCHEMA")
    train_path, eval_path = _write(tmp_path, _good_train(), evaluate)
    with pytest.raises(contract.L1FutureSourceContractError, match="recovered V2"):
        contract.audit(train_path, eval_path)


def test_missing_exposure_pair_helper_is_rejected(tmp_path: Path) -> None:
    evaluate = _good_eval().replace("paired_summary_complete", "old_paired_summary")
    train_path, eval_path = _write(tmp_path, _good_train(), evaluate)
    with pytest.raises(contract.L1FutureSourceContractError, match="exposure paired comparison"):
        contract.audit(train_path, eval_path)
