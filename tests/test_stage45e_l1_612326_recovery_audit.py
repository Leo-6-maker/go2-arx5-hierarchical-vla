"""Synthetic CPU/filesystem tests for the PI0128-B 612326 recovery auditor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_l1_612326_recovery_audit as recovery


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _mask() -> list[list[float]]:
    mask = [[0.0 for _ in range(recovery.ACTION_DIM)] for _ in range(recovery.ACTION_HORIZON)]
    for row in range(15):
        for column in range(9):
            mask[row][column] = 1.0
        mask[row][9] = 1.0
    for row in range(recovery.ACTION_HORIZON):
        mask[row][10] = 1.0
    assert sum(item for row in mask for item in row) == recovery.ACTIVE_SCALAR_COUNT
    return mask


def _training_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / f"stage45e_l1_h200_20260911_377b5012_{recovery.EXPECTED_PBS_JOBID}"
    root.mkdir()

    _write_json(
        root / "stage45e_l1_failure.json",
        {
            "schema": "VANDA_STAGE45E_L1_FAILURE_SEAL_V1",
            "status": "FAIL",
            "phase": "POST_TRAINING_AUDIT",
            "execution_head": recovery.TRAINING_HEAD,
            "training_started": True,
            "action_head_forwards": 2000,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_steps": 0,
            "simulator_steps": 0,
            "hardware_steps": 0,
        },
    )

    events = [
        {"event": "PRE_MODEL_ADMISSION_PASS"},
        {"event": "MODEL_IMPORT_COMPLETE"},
        {"event": "ONE_BATCH_BACKWARD_CANARY_PASS"},
        {"event": "TRAINING_START", "optimizer_steps": 2000, "checkpoint_steps": [1000, 2000]},
        {"event": "TRAINING_RETURNED"},
    ]
    (root / "stage45e_l1_events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8"
    )

    _write_json(
        root / "stage45e_l1_backward_canary.json",
        {
            "schema": "VANDA_STAGE45E_L1_ONE_BATCH_BACKWARD_CANARY_V1",
            "status": "PASS",
            "loss": 1.25,
            "active_scalar_count": 190,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "parameter_delta": {"pass": True},
            "stats_unchanged": True,
            "mask_unchanged": True,
            "normalization_unchanged": True,
        },
    )

    statistics = {"active_mask": _mask(), "schema": "fixture"}
    monkeypatch.setattr(recovery, "FULL24_STATS_PAYLOAD_SHA256", recovery.canonical_sha(statistics))
    _write_json(
        root / "experiment_cfg" / "structured_action_statistics.json",
        {
            "source_dataset_untouched_by_factory": True,
            "upstream_generate_stats_called": False,
            "statistics": statistics,
            "contract": {"mask_active_scalar_count": 190},
        },
    )

    for step in (1000, 2000):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir()
        _write_json(checkpoint / "trainer_state.json", {"global_step": step})
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer" + str(step).encode())
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler" + str(step).encode())
        (checkpoint / "rng_state.pth").write_bytes(b"rng" + str(step).encode())
        (checkpoint / "model.safetensors").write_bytes(b"model" + str(step).encode())

    for name in (
        "stage45e_l1_runtime_binding.json",
        "stage45e_l1_pre_model_admission.json",
        "stage45e_l1_input_audit.json",
    ):
        _write_json(root / name, {"status": "PASS", "fixture": name})
    return root


def test_plan_is_cpu_read_only() -> None:
    plan = recovery.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["historical_root_mutation_allowed"] is False
    assert plan["evaluation_execution"] is False
    assert plan["static_guard"]["model_construction"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0


def test_valid_historical_root_produces_fresh_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _training_root(tmp_path, monkeypatch)
    output = tmp_path / "recovery-acceptance"
    result = recovery.audit(root, output, recovery.EXPECTED_PBS_JOBID)
    assert result["token"] == recovery.ACCEPT_TOKEN
    assert result["checkpoint_1000"]["global_step"] == 1000
    assert result["checkpoint_2000"]["global_step"] == 2000
    assert result["structured_stats"]["mask_active_scalar_count"] == 190
    assert result["scientific_claim_boundary"]["runtime_effective_weight_evidence_status"] == "NOT_CAPTURED_IN_HISTORICAL_CANARY"
    assert result["execution_counters_for_this_audit"]["model_forward"] == 0
    assert (output / "stage45e_l1_612326_readonly_acceptance.json").is_file()


def test_rejects_historical_pass_file_or_mutated_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _training_root(tmp_path, monkeypatch)
    _write_json(root / "stage45e_l1_training_result.json", {"status": "fabricated"})
    with pytest.raises(recovery.RecoveryAuditError, match="normal PASS training result"):
        recovery.audit(root, tmp_path / "out-a", recovery.EXPECTED_PBS_JOBID)

    root = _training_root(tmp_path / "second", monkeypatch)
    sidecar_path = root / "experiment_cfg" / "structured_action_statistics.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["statistics"]["schema"] = "mutated"
    _write_json(sidecar_path, sidecar)
    with pytest.raises(recovery.RecoveryAuditError, match="statistics payload drift"):
        recovery.audit(root, tmp_path / "out-b", recovery.EXPECTED_PBS_JOBID)


def test_rejects_incomplete_checkpoint_or_wrong_event_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _training_root(tmp_path, monkeypatch)
    (root / "checkpoint-2000" / "optimizer.pt").unlink()
    with pytest.raises(recovery.RecoveryAuditError, match="full-state closure missing"):
        recovery.audit(root, tmp_path / "out-c", recovery.EXPECTED_PBS_JOBID)

    root = _training_root(tmp_path / "third", monkeypatch)
    event_path = root / "stage45e_l1_events.jsonl"
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    rows[3], rows[4] = rows[4], rows[3]
    event_path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    with pytest.raises(recovery.RecoveryAuditError, match="event order drift"):
        recovery.audit(root, tmp_path / "out-d", recovery.EXPECTED_PBS_JOBID)
