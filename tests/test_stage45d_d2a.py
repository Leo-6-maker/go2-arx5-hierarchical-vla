"""CPU/static regression tests for the PI0119 Stage45D-D2A surface."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d2a_path_divergence as engine  # noqa: E402
import stage45d_d2a_runtime_admission as admission  # noqa: E402
import stage45d_d2a_vanda_h200_launcher as launcher  # noqa: E402


def _section() -> dict:
    metrics = {"l2": 0.0, "rmse": 0.0, "max_abs": 0.0, "cosine": 1.0}
    return {"overall": dict(metrics), "blocks": {name: dict(metrics) for name in engine.BLOCK_SLICES}}


def _row(checkpoint: str, bucket: int, *, physical: bool = False) -> dict:
    row = {
        "identity": {
            "checkpoint": checkpoint,
            "window_id": "stage22_expert_red_target_train_a:expert-0000",
            "target_id": "red_target_train_a",
            "seed": 4242,
            "t_bucket": bucket,
        },
        "state_path_divergence": _section(),
        "field_sensitivity": _section(),
        "teacher_bridge_local_error": _section(),
        "off_path_diagnostic": _section(),
        "physical": None,
    }
    if physical:
        target = {
            "initial_target_distance_m": 1.0,
            "minimum_target_distance_m": 0.8,
            "final_target_distance_m": 0.9,
            "absolute_progress_m": 0.1,
            "best_progress_m": 0.2,
            "endpoint_target_vector_cosine": 0.5,
            "progress_sign": "POSITIVE",
        }
        endpoint = {"trajectory_metrics": {"target": target}}
        row["physical"] = {
            "teacher_bridge_implied_endpoint": endpoint,
            "free_running_endpoint": endpoint,
        }
    return row


def test_d2a_static_contract_and_exact_budget() -> None:
    static = admission.validate_static_contract()
    assert static["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert admission.CALL_BUDGET["action_head_calls"] == 576
    assert admission.CALL_BUDGET["conditioning_passes"] == 72
    assert admission.CALL_BUDGET["total_model_side_calls"] == 648
    assert admission.SAMPLER_CONTRACT["t_values"] == [0.0, 0.25, 0.5, 0.75]
    assert admission.SAMPLER_CONTRACT["t_buckets"] == [0, 250, 500, 750]
    assert admission.SAMPLER_CONTRACT["noise_seed"] == 4242
    assert admission.CONTRACT["target_xyz_in_model_input"] is False
    assert admission.LOCKED_BOUNDARIES["D2B_solver_comparison"] == "LOCKED"


def test_engine_profile_and_guards_are_static_only() -> None:
    profile = engine.validate_profile()
    assert profile["status"] == "READY_FOR_EXECUTION_NOT_EXECUTED"
    assert engine._static_guard()["status"] == "PASS"
    assert engine._static_guard()["top_level_forbidden_imports"] == []
    assert engine._static_guard()["forbidden_call_hits"] == []
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    top_level = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_level.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not top_level.intersection({"torch", "transformers", "gr00t", "cuda"})


def test_launcher_is_foreign_cwd_and_empty_pythonpath_safe(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stage45d_d2a_vanda_h200_launcher.py"), "--plan-only"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["foreign_cwd_supported"] is True
    assert result["empty_pythonpath_supported"] is True
    assert result["smoke_token"] == admission.SMOKE_PASS_TOKEN
    assert result["full_token"] == admission.FULL_PASS_TOKEN


def test_d2a_path_metrics_are_blockwise_and_active_190() -> None:
    zeros = np.zeros(engine.RAW_SHAPE, dtype=np.float32)
    changed = zeros.copy()
    changed[0, 0, 0] = 2.0
    changed[0, 20, 10] = 3.0
    metrics = engine._blockwise_vector_metrics(changed, zeros, namespace="TEST", difference="changed")
    assert metrics["overall"]["rmse"] > 0.0
    assert metrics["blocks"]["geometry_xyz"]["max_abs"] == 2.0
    assert metrics["blocks"]["gripper"]["max_abs"] == 3.0
    assert int(np.sum(engine.MASK)) == 190


def test_free_running_state_is_carried_without_reset() -> None:
    source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "x_fr = x0.copy()" in source
    assert "x_fr = x_fr_next.copy()" in source
    assert "x_fr = x_tf" not in source
    assert '"free_running_reset": False' in source


def test_t0_admission_fails_closed_if_model_import_preexists() -> None:
    with pytest.raises(admission.D2AAdmissionError, match="model-side import"):
        admission.validate_future_admission(
            {},
            {},
            authority_token=None,
            phase="SMOKE",
            imported_modules={"torch": object()},
        )


def test_raw_stage_is_sealed_before_derived_and_summary_pairs_physical(tmp_path: Path) -> None:
    arrays = {key: np.zeros(engine.RAW_SHAPE, dtype=np.float32) for key in ("x0", "x1", "x_tf", "x_fr", "v_target", "v_tf", "v_fr", "x_tf_next", "x_fr_next")}
    arrays["mask"] = engine.MASK
    raw = engine._write_raw_stage(tmp_path / "probe", {"checkpoint": "FULL24@2000"}, arrays)
    assert Path(raw["path"]).is_file()
    manifest = json.loads(Path(raw["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "RAW_SEALED_BEFORE_DERIVED"
    assert not (tmp_path / "probe" / "derived_metrics.json").exists()

    rows = []
    for checkpoint in admission.CHECKPOINT_ORDER:
        rows.extend(_row(checkpoint, bucket, physical=(bucket == 750)) for bucket in admission.T_BUCKETS)
    summary = engine._aggregate_rows(
        rows,
        phase="FULL",
        counters={"action_head_calls": 24, "conditioning_passes": 3},
    )
    assert summary["paired"]["FULL24@2000_minus_FULL24@1000"]["paired_record_count"] == 4
    assert summary["paired_physical_final_readout"]["FULL24@2000_minus_FULL24@1000"]["paired_record_count"] == 1


def test_launcher_static_guard_has_no_model_imports() -> None:
    guard = launcher.static_guard()
    assert guard["status"] == "PASS"
    assert guard["top_level_forbidden_imports"] == []


def test_completed_evidence_gets_durable_hash_inventory(tmp_path: Path) -> None:
    (tmp_path / "full" / "raw").mkdir(parents=True)
    (tmp_path / "observed_runtime.json").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "full" / "raw" / "stage.json").write_text("raw\n", encoding="utf-8")

    inventory_path, inventory_sha = launcher.write_inventory(tmp_path)

    assert inventory_path == tmp_path / "inventory" / "SHA256SUMS"
    assert inventory_path.is_file()
    payload = inventory_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == inventory_sha
    lines = payload.decode("utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == [
        "full/raw/stage.json",
        "observed_runtime.json",
    ]
    assert all("inventory/SHA256SUMS" not in line for line in lines)
