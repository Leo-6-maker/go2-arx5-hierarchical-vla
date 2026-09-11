"""CPU/static regression tests for the PI0122 Stage45D-D2B surface."""

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

import stage45d_d2b_runtime_admission as admission  # noqa: E402
import stage45d_d2b_solver_discretization as engine  # noqa: E402
import stage45d_d2b_vanda_h200_launcher as launcher  # noqa: E402


PBS_SCRIPT = ROOT / "hpc" / "pbs" / "stage45d_d2b_vanda_h200.pbs"


def _section() -> dict:
    metrics = {"l2": 0.0, "rmse": 0.0, "max_abs": 0.0, "cosine": 1.0}
    return {"overall": dict(metrics), "blocks": {name: dict(metrics) for name in engine.BLOCK_SLICES}}


def _row(checkpoint: str, solver: str, *, physical: bool = False) -> dict:
    row = {
        "identity": {
            "checkpoint": checkpoint,
            "solver": solver,
            "window_id": "stage22_expert_red_target_train_a:expert-0000",
            "target_id": "red_target_train_a",
            "seed": 4242,
            "timestep_bucket": 750,
        },
        "normalized_free_running_field_diagnostic": _section(),
        "normalized_state_to_endpoint": _section(),
        "normalized_state_step": _section(),
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
            "path_length_m": 1.1,
            "roughness": 0.2,
            "progress_sign": "POSITIVE",
        }
        row["physical"] = {"free_running_endpoint": {"trajectory_metrics": {"target": target}}}
    return row


def test_d2b_static_contract_and_exact_budget() -> None:
    static = admission.validate_static_contract()
    assert static["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert admission.CALL_BUDGET["action_head_calls"] == 2016
    assert admission.CALL_BUDGET["conditioning_passes"] == 72
    assert admission.CALL_BUDGET["total_model_side_calls"] == 2088
    assert admission.CALL_BUDGET["smoke_action_head_calls"] == 28
    assert admission.CALL_BUDGET["smoke_total_model_side_calls"] == 29
    assert admission.SOLVER_SCHEDULES["E4"]["t_values"] == [0.0, 0.25, 0.5, 0.75]
    assert admission.SOLVER_SCHEDULES["E8"]["t_values"] == [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    assert admission.SOLVER_SCHEDULES["E16"]["steps"] == 16
    assert admission.SAMPLER_CONTRACT["free_running_reset"] is False
    assert admission.SAMPLER_CONTRACT["public_sampler_calls"] == 0
    assert admission.SAMPLER_CONTRACT["get_action_calls"] == 0
    assert admission.CONTRACT["target_xyz_in_model_input"] is False


def test_engine_profile_and_static_guard_are_model_free() -> None:
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
        [sys.executable, str(ROOT / "scripts" / "stage45d_d2b_vanda_h200_launcher.py"), "--plan-only"],
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
    assert result["admission_before_model_import"] is True
    assert result["smoke_token"] == admission.SMOKE_PASS_TOKEN
    assert result["full_token"] == admission.FULL_PASS_TOKEN


def test_solver_schedules_are_exact_forward_euler_and_monotonic() -> None:
    for solver in admission.SOLVER_ORDER:
        schedule = admission.SOLVER_SCHEDULES[solver]
        values = schedule["t_values"]
        assert len(values) == admission.SOLVER_STEPS[solver]
        assert values == sorted(values)
        assert all(b > a for a, b in zip(values, values[1:]))
        assert schedule["family"] == "FORWARD_EULER"
        assert schedule["dt"] == 1.0 / admission.SOLVER_STEPS[solver]
        assert schedule["t_buckets"] == [int(value * 1000) for value in values]


def test_free_running_state_is_carried_without_reset() -> None:
    source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "x_fr = x0.copy()" in source
    assert "x_fr = x_fr_next.copy()" in source
    assert "x_fr = x_tf" not in source
    assert '"free_running_reset": False' in source
    assert admission.CONTRACT["no_teacher_forced_path"] is True


def test_raw_stage_seals_before_derived_and_keeps_solver_identity(tmp_path: Path) -> None:
    arrays = {key: np.zeros(engine.RAW_SHAPE, dtype=np.float32) for key in ("x0", "x1", "x_fr", "v_target", "v_fr", "x_fr_next")}
    arrays["mask"] = engine.MASK
    raw = engine._write_raw_stage(tmp_path / "probe", {"checkpoint": "FULL24@2000", "solver": "E16"}, arrays)
    assert Path(raw["path"]).is_file()
    manifest = json.loads(Path(raw["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "RAW_SEALED_BEFORE_DERIVED"
    assert manifest["identity"]["solver"] == "E16"
    assert not (tmp_path / "probe" / "derived_metrics.json").exists()


def test_paired_solver_summary_is_by_checkpoint_and_window() -> None:
    rows = []
    for solver in admission.SOLVER_ORDER:
        row = _row("FULL24@2000", solver, physical=True)
        row["normalized_state_to_endpoint"]["overall"]["l2"] = {"E4": 3.0, "E8": 2.0, "E16": 1.0}[solver]
        row["metric_values"] = engine._metric_values(row)
        row["physical_metric_values"] = engine._physical_metric_values(row)
        rows.append(row)
    summary = engine._paired_solver_summary(rows, "FULL24@2000", "E4", "E8")
    assert summary["paired_record_count"] == 1
    key = "normalized_state_to_endpoint.overall.l2"
    assert summary["overall"]["numeric"][key]["mean"] == -1.0
    physical = engine._paired_solver_summary(rows, "FULL24@2000", "E4", "E16", field="physical_metric_values")
    assert physical["paired_record_count"] == 1


def test_admission_fails_closed_if_model_import_preexists() -> None:
    with pytest.raises(admission.D2BAdmissionError, match="model-side import"):
        admission.validate_future_admission({}, {}, authority_token=None, phase="SMOKE", imported_modules={"torch": object()})


def test_launcher_static_guard_has_no_model_imports() -> None:
    guard = launcher.static_guard()
    assert guard["status"] == "PASS"
    assert guard["top_level_forbidden_imports"] == []
    assert guard["admission_before_model_import"] is True


def test_launcher_parses_pretty_child_json_after_runtime_logs() -> None:
    payload = {"status": "PASS", "token": admission.SMOKE_PASS_TOKEN, "counters": {"action_head_calls": 28}}
    stdout = "runtime log\n[groot/hf] cache hit\n" + json.dumps(payload, indent=2) + "\n"
    assert launcher._parse_child_json(stdout) == payload


def test_pbs_has_both_runtime_aliases_and_no_retry() -> None:
    source = PBS_SCRIPT.read_text(encoding="utf-8")
    assert "/cm/shared/apps/python/3.12.3/bin/python3.12" in source
    assert "/app1/Hopper/shared/apps/python/3.12.3/bin/python3.12" in source
    assert "--untracked-files=all" in source
    assert "cd \"${TMPDIR:-/tmp}\"" in source
    assert "automatic retry" in source.lower() or "retry" in source.lower()
    assert "STAGE45D_D2B_AUTHORITY_TOKEN=PI_GATE_STATE_0122" in source
    assert "stage45d_d2b_solver_discretization_20260911_" in source


def test_completed_evidence_gets_durable_hash_inventory(tmp_path: Path) -> None:
    (tmp_path / "full" / "raw").mkdir(parents=True)
    (tmp_path / "observed_runtime.json").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "full" / "raw" / "stage.json").write_text("raw\n", encoding="utf-8")
    inventory_path, inventory_sha = launcher.write_inventory(tmp_path)
    assert inventory_path == tmp_path / "inventory" / "SHA256SUMS"
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == inventory_sha
    assert inventory_path.read_text(encoding="utf-8").splitlines() == [
        f"{hashlib.sha256((tmp_path / 'full' / 'raw' / 'stage.json').read_bytes()).hexdigest()}  full/raw/stage.json",
        f"{hashlib.sha256((tmp_path / 'observed_runtime.json').read_bytes()).hexdigest()}  observed_runtime.json",
    ]
