"""CPU/filesystem tests for the recovered L1 training -> eval input bridge."""

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
        "schema": "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1",
        "status": "PASS_READONLY_TRAINING_ARTIFACT_INTEGRITY",
        "token": bridge.ACCEPT_TOKEN,
        "training_root": str(training.resolve()),
        "historical_training_head": bridge.TRAINING_HEAD,
        "branch": bridge.BRANCH,
        "historical_root_mutated": False,
        "scientific_claim_boundary": {
            "train24_evaluation_executed": False,
            "runtime_effective_weight_evidence_status": "NOT_CAPTURED_IN_HISTORICAL_CANARY",
        },
        "structured_stats": {
            "statistics_payload_sha256": bridge.FULL24_STATS_PAYLOAD_SHA256,
            "mask_active_scalar_count": 190,
        },
        "checkpoint_1000": checkpoints[1000],
        "checkpoint_2000": checkpoints[2000],
    }
    acceptance_path = tmp_path / "acceptance.json"
    _write_json(acceptance_path, acceptance)
    return training, acceptance_path


def test_plan_is_nonexecuting() -> None:
    plan = bridge.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["evaluation_authorized"] is False
    assert plan["static_guard"]["model_forward"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0


def test_bridge_rehashes_accepted_checkpoints(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    output = tmp_path / "eval-input.json"
    result = bridge.bridge(acceptance, training, output)
    assert result["status"] == "PASS_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION"
    assert result["evaluation_authorized_by_this_bridge"] is False
    assert result["final_train24_token_emitted"] is False
    assert result["checkpoints"]["L1@1000"]["rehash_matches_acceptance"] is True
    assert result["checkpoints"]["L1@2000"]["rehash_matches_acceptance"] is True
    assert result["execution_counters"]["model_forward"] == 0
    assert output.is_file()


def test_bridge_rejects_checkpoint_mutation_after_acceptance(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    with (training / "checkpoint-2000" / "model.safetensors").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="bytes changed after recovery acceptance"):
        bridge.bridge(acceptance, training, tmp_path / "out.json")


def test_bridge_rejects_fake_pass_or_wrong_acceptance_token(tmp_path: Path) -> None:
    training, acceptance = _fixture(tmp_path)
    _write_json(training / "stage45e_l1_training_result.json", {"status": "fabricated"})
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="historical PASS training result"):
        bridge.bridge(acceptance, training, tmp_path / "out-a.json")

    training, acceptance = _fixture(tmp_path / "second")
    value = json.loads(acceptance.read_text(encoding="utf-8"))
    value["token"] = "WRONG"
    _write_json(acceptance, value)
    with pytest.raises(bridge.RecoveryEvalBridgeError, match="token drift"):
        bridge.bridge(acceptance, training, tmp_path / "out-b.json")
