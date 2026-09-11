"""CPU/filesystem tests for PI0128-C N2 continuation preparation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import stage45e_n2_continuation_admission as admission


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _tree(root: Path, prefix: str) -> str:
    return admission._tree_digest(admission._tree_entries(root, prefix))


def _checkpoint(root: Path, step: int) -> str:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    _write_json(path / "trainer_state.json", {"global_step": step})
    (path / "optimizer.pt").write_bytes(b"optimizer" + str(step).encode())
    (path / "scheduler.pt").write_bytes(b"scheduler" + str(step).encode())
    (path / "rng_state.pth").write_bytes(b"rng" + str(step).encode())
    (path / "model.safetensors").write_bytes(b"model" + str(step).encode())
    return _tree(path, f"stage45e/checkpoint-{step}")


def _fake_manifest(tmp_path: Path) -> tuple[dict, Path]:
    cfg = copy.deepcopy(admission.manifest())
    legacy = tmp_path / "legacy-A"
    legacy.mkdir(parents=True)
    (legacy / "model.safetensors").write_bytes(b"legacy-model")
    (legacy / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    cfg["accepted_identities"]["legacy_full24_2000_tree_sha256"] = _tree(legacy, "checkpoint-2000")
    return cfg, legacy


def _fake_n2_root(tmp_path: Path, seed: int, cfg: dict, stats_path: Path, stats_payload: dict) -> Path:
    root = tmp_path / f"n2-{seed}"
    root.mkdir(parents=True)
    cp1000 = _checkpoint(root, 1000)
    cp2000 = _checkpoint(root, 2000)
    result = {
        "status": "PASS" if seed != 2222 else "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW",
        "seed": seed,
        "starting_checkpoint": "BASE",
        "base_tree_sha256": cfg["accepted_identities"]["base_tree_sha256"],
        "full24_tree_sha256": cfg["accepted_identities"]["full24_tree_sha256"],
        "n2_stats_sha256": admission.sha256_file(stats_path),
        "n2_stats_payload_sha256": admission.canonical_sha(stats_payload),
        "training_steps": 2000,
        "checkpoint_1000": {"tree_sha256": cp1000},
        "checkpoint_2000": {"tree_sha256": cp2000},
    }
    _write_json(root / ("stage45e_n2_training_result.json" if seed == 2222 else "training_result.json"), result)
    _write_json(root / "experiment_cfg" / "structured_action_statistics.json", {"statistics": stats_payload})
    return root


def test_plan_freezes_corrected_budgets_and_no_execution() -> None:
    plan = admission.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["execution_enabled"] is False
    assert plan["continuation_order"] == ["G3_D1", "G4_D2A", "G5_D1", "G5_D2A"]
    assert plan["corrected_budgets"]["G3_D1"]["conditioning_passes"] == 96
    assert plan["corrected_budgets"]["G3_D1"]["total_model_side_calls"] == 960
    assert plan["corrected_budgets"]["G5_D1"]["conditioning_passes"] == 72
    assert plan["corrected_budgets"]["G5_D1"]["total_model_side_calls"] == 720
    assert plan["corrected_budgets"]["G4_D2A"]["total_model_side_calls"] == 864
    assert plan["corrected_budgets"]["G5_D2A"]["total_model_side_calls"] == 648
    assert plan["static_guard"]["model_forward"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0


def test_legacy_tree_identity_is_destination_name_invariant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, legacy_a = _fake_manifest(tmp_path)
    legacy_b = tmp_path / "FULL24@2000-materialized-different-name"
    legacy_b.mkdir()
    for child in legacy_a.iterdir():
        (legacy_b / child.name).write_bytes(child.read_bytes())
    monkeypatch.setattr(admission, "manifest", lambda: cfg)
    left = admission.legacy_full24_2000_tree(legacy_a)
    right = admission.legacy_full24_2000_tree(legacy_b)
    assert left["tree_sha256"] == right["tree_sha256"]
    assert left["logical_prefix"] == right["logical_prefix"] == "checkpoint-2000"


def test_n2_root_binds_stats_sidecar_and_checkpoint_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _legacy = _fake_manifest(tmp_path)
    stats_payload = {"schema": "fixture", "selected_lower": [0.0, 1.0], "selected_upper": [2.0, 3.0]}
    stats_path = tmp_path / "n2-stats.json"
    _write_json(stats_path, {"statistics_payload_sha256": admission.canonical_sha(stats_payload), "statistics": stats_payload})
    cfg["accepted_identities"]["n2_stats_sha256"] = admission.sha256_file(stats_path)
    cfg["accepted_identities"]["n2_stats_payload_sha256"] = admission.canonical_sha(stats_payload)
    monkeypatch.setattr(admission, "manifest", lambda: cfg)
    root = _fake_n2_root(tmp_path, 2223, cfg, stats_path, stats_payload)
    observed = admission.n2_root(root, 2223, stats_path)
    assert observed["seed"] == 2223
    assert observed["sidecar_statistics_binding"]["match"] is True
    assert observed["checkpoints"]["1000"]["global_step"] == 1000
    assert observed["checkpoints"]["2000"]["global_step"] == 2000


def test_n2_root_rejects_sidecar_or_result_stats_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _legacy = _fake_manifest(tmp_path)
    stats_payload = {"schema": "fixture", "selected_lower": [0.0], "selected_upper": [2.0]}
    stats_path = tmp_path / "n2-stats.json"
    _write_json(stats_path, {"statistics_payload_sha256": admission.canonical_sha(stats_payload), "statistics": stats_payload})
    cfg["accepted_identities"]["n2_stats_sha256"] = admission.sha256_file(stats_path)
    cfg["accepted_identities"]["n2_stats_payload_sha256"] = admission.canonical_sha(stats_payload)
    monkeypatch.setattr(admission, "manifest", lambda: cfg)
    root = _fake_n2_root(tmp_path, 2224, cfg, stats_path, stats_payload)

    sidecar_path = root / "experiment_cfg" / "structured_action_statistics.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["statistics"]["selected_upper"][0] += 2e-12
    _write_json(sidecar_path, sidecar)
    with pytest.raises(admission.ContinuationAdmissionError, match="sidecar statistics drift"):
        admission.n2_root(root, 2224, stats_path)

    root = _fake_n2_root(tmp_path / "other", 2224, cfg, stats_path, stats_payload)
    result_path = root / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["n2_stats_payload_sha256"] = "0" * 64
    _write_json(result_path, result)
    with pytest.raises(admission.ContinuationAdmissionError, match="payload identity drift"):
        admission.n2_root(root, 2224, stats_path)


def test_full_cpu_admission_never_enables_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, legacy = _fake_manifest(tmp_path)
    stats_payload = {"schema": "fixture", "selected_lower": [0.0], "selected_upper": [2.0]}
    stats_path = tmp_path / "n2-stats.json"
    _write_json(stats_path, {"statistics_payload_sha256": admission.canonical_sha(stats_payload), "statistics": stats_payload})
    cfg["accepted_identities"]["n2_stats_sha256"] = admission.sha256_file(stats_path)
    cfg["accepted_identities"]["n2_stats_payload_sha256"] = admission.canonical_sha(stats_payload)
    monkeypatch.setattr(admission, "manifest", lambda: cfg)
    roots = {seed: _fake_n2_root(tmp_path, seed, cfg, stats_path, stats_payload) for seed in (2222, 2223, 2224)}
    output = tmp_path / "admission.json"
    result = admission.admit(legacy, roots[2222], roots[2223], roots[2224], stats_path, output)
    assert result["status"] == "PASS_CPU_INPUT_IDENTITIES_ONLY"
    assert result["execution_enabled"] is False
    assert result["execution_counters"]["model_forward"] == 0
    assert result["execution_counters"]["gpu"] == 0
    assert result["execution_counters"]["pbs"] == 0
    assert output.is_file()
