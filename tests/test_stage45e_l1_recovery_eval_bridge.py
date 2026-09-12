"""CPU/filesystem tests for contract-bound recovered L1 training -> eval input bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_l1_recovery_eval_bridge as bridge


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _tree(path: Path, step: int) -> str:
    return bridge._tree_digest(bridge._tree_entries(path, f"stage45e_l1/checkpoint-{step}"))


def _scientific_contract() -> dict:
    runtime = {
        "xyz_loss_weights": dict(bridge.XYZ_WEIGHTS),
        "weight_matrix_sha256": bridge.L1_MATRIX_SHA256,
        "active_scalar_count": 190,
        "mask_folding": False,
        "adaptive_renormalization": False,
    }
    code = {
        "xyz_loss_weights": dict(bridge.XYZ_WEIGHTS),
        "matrix_sha256": bridge.L1_MATRIX_SHA256,
        "active_scalar_count": 190,
        "denominator": "binary_action_mask.sum()",
        "mask_folding": False,
        "adaptive_renormalization": False,
    }
    recipe = {
        "seed": 2222,
        "max_steps": 2000,
        "global_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "fp16": False,
        "num_gpus": 1,
    }
    return {
        "l1_runtime_contract": runtime,
        "l1_code_contract": code,
        "training_recipe": recipe,
        "full24_stats_sha256": bridge.FULL24_STATS_SHA256,
        "full24_stats_payload_sha256": bridge.FULL24_STATS_PAYLOAD_SHA256,
        "normalization_unchanged": True,
        "runtime_effective_weight_evidence_status": "NOT_CAPTURED_IN_HISTORICAL_CANARY",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    training = tmp_path / "training"
    training.mkdir(parents=True)
    checkpoints = {}
    for step in (1000, 2000):
        cp = training / f"checkpoint-{step}"
        cp.mkdir()
        _write_json(cp / "trainer_state.json", {"global_step": step})
        (cp / "optimizer.pt").write_bytes(b"optimizer" + str(step).encode())
        (cp / "scheduler.pt").write_bytes(b"scheduler" + str(step).encode())
        (cp / "rng_state.pth").write_bytes(b"rng" + str(step).encode())
        (cp / "model.safetensors").write_bytes(b"model" + str(step).encode())
        checkpoints[step] = {
            "path": str(cp.resolve()),
            "global_step": step,
            "tree_sha256": _tree(cp, step),
        }
    acceptance = {
        "schema": "STAGE45E_L1_612326_CONTRACT_BOUND_ACCEPTANCE_V1",
        "status": "PASS_READONLY_TRAINING_AND_FROZEN_CONTRACT_INTEGRITY",
        "token": bridge.ACCEPT_TOKEN,
        "training_root": str(training.resolve()),
        "historical_training_head": bridge.TRAINING_HEAD,
        "branch": bridge.BRANCH,
        "historical_root_mutated": False,
        "evaluation_executed": False,
        "scientific_contract": _scientific_contract(),
        "checkpoint_1000": checkpoints[1000],
        "checkpoint_2000": checkpoints[2000],
    }
    acceptance_path = tmp_path / "contract-bound-acceptance.json"
    _write_json(acceptance_path, acceptance)
    return training, acceptance_path


def test_plan_is_nonexecuting_and_requires_contract_bound_token() -> None:
    plan = bridge.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["accept_token"] == bridge.ACCEPT_TOKEN
    assert plan["contract_bound_required"] is True
    assert plan["evaluation_authorized"] is False
    assert plan["static_guard"]["model_forward"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0


def test_bridge_rehashes_accepted_checkpoints_and_preserves_contract(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    output = tmp_path / "eval-input.json"
    result = bridge.bridge(acceptance, training, output)
    assert result["status"] == "PASS_CONTRACT_BOUND_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION"
    assert result["schema"] == "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2"
    assert result["evaluation_authorized_by_this_bridge"] is False
    assert result["final_train24_token_emitted"] is False
    assert result["checkpoints"]["L1@1000"]["rehash_matches_acceptance"] is True
    assert result["checkpoints"]["L1@2000"]["rehash_matches_acceptance"] is True
    assert result["scientific_contract"]["l1_code_contract"]["denominator"] == "binary_action_mask.sum()"
    assert result["execution_counters"]["model_forward"] == 0
    assert output.is_file()


def test_bridge_rejects_checkpoint_mutation_after_acceptance(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    with (training / "checkpoint-2000" / "model.safetensors").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="bytes changed after recovery acceptance"):
        bridge.bridge(acceptance, training, tmp_path / "out.json")


def test_bridge_rejects_old_unbound_or_wrong_contract_acceptance(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    value = json.loads(acceptance.read_text(encoding="utf-8"))
    value["schema"] = "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1"
    value["token"] = "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTED"
    _write_json(acceptance, value)
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="contract-bound acceptance schema drift"):
        bridge.bridge(acceptance, training, tmp_path / "out-a.json")

    training, acceptance = _fixture(tmp_path / "second")
    value = json.loads(acceptance.read_text(encoding="utf-8"))
    value["scientific_contract"]["l1_code_contract"]["denominator"] = "WRONG"
    _write_json(acceptance, value)
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="denominator drift"):
        bridge.bridge(acceptance, training, tmp_path / "out-b.json")


def test_bridge_rejects_fake_historical_pass_result(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    _write_json(training / "stage45e_l1_training_result.json", {"status": "fabricated"})
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="historical PASS training result"):
        bridge.bridge(acceptance, training, tmp_path / "out.json")


def test_bridge_requires_and_preserves_reproduced_weight_attachment(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    value = json.loads(acceptance.read_text(encoding="utf-8"))
    contract = value["scientific_contract"]
    contract["runtime_effective_weight_evidence_status"] = bridge.REPRODUCED_WEIGHT_EVIDENCE_STATUS
    contract["reproduced_runtime_weight_evidence"] = {"status": bridge.REPRODUCED_WEIGHT_EVIDENCE_STATUS, "path": "/scratch/reproduced.json"}
    _write_json(acceptance, value)
    result = bridge.bridge(acceptance, training, tmp_path / "out.json")
    assert result["runtime_effective_weight_evidence_status"] == bridge.REPRODUCED_WEIGHT_EVIDENCE_STATUS
    assert result["reproduced_runtime_weight_evidence"]["status"] == bridge.REPRODUCED_WEIGHT_EVIDENCE_STATUS
