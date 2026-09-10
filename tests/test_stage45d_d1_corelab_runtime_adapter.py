"""CPU/static tests for the Corelab Stage45D-D1 4080S preflight adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d1_corelab_runtime_adapter as adapter  # noqa: E402
import stage45d_d1_runtime_admission as admission  # noqa: E402


def test_profile_static_contract_and_guard() -> None:
    profile = adapter.validate_profile()
    assert profile["checkpoint"]["label"] == "FULL24@2000"
    assert profile["cohort"]["selected_window_id"] == adapter.WINDOW_ID
    assert profile["model_execution"]["total_direct_model_side_calls"] == 4
    assert profile["execution_boundary_now"]["gpu"] == 0
    guard = adapter.static_guard()
    assert guard["status"] == "PASS"
    assert guard["top_level_forbidden_imports"] == []
    assert guard["forbidden_call_hits"] == []


def test_profile_device_policy_rejects_16gb_multigpu_and_retry() -> None:
    profile = adapter.validate_profile()
    request = dict(profile["device_policy"])
    request["allow_16gb_secondary_fallback"] = True
    with pytest.raises(adapter.AdapterError, match="forbidden mode"):
        adapter.validate_preflight_device_request(request)

    request = dict(profile["device_policy"])
    request["allow_multi_gpu"] = True
    with pytest.raises(adapter.AdapterError, match="forbidden mode"):
        adapter.validate_preflight_device_request(request)

    request = dict(profile["device_policy"])
    request["selected_device"] = "cuda:1"
    with pytest.raises(adapter.AdapterError, match="cuda:0"):
        adapter.validate_preflight_device_request(request)


def test_profile_window_and_stats_contract_reject_drift() -> None:
    payload = json.loads(adapter.PROFILE_PATH.read_text(encoding="utf-8"))
    payload["cohort"]["selected_window_id"] = "stage22_expert_red_target_train_b:expert-0000"
    # The profile verifier must remain anchored to the single authorized window.
    malformed = adapter.PROFILE_PATH.parent / ".stage45d_profile_test.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(adapter.AdapterError, match="selected_window_id"):
            adapter.validate_profile(malformed)
    finally:
        malformed.unlink()


def test_flow_oracle_and_normalized_namespace() -> None:
    noise = np.zeros(adapter.RAW_SHAPE, dtype=np.float32)
    x1 = np.full(adapter.RAW_SHAPE, 0.25, dtype=np.float32)
    x0, x_t, v_target = adapter.harness.construct_flow(noise, x1, adapter.T_VALUES[1])
    arrays = adapter.build_probe_arrays(noise, x1, adapter.T_VALUES[1], v_target)
    assert np.max(np.abs(arrays["x1_hat"] - x1)) <= 2e-6
    metrics = adapter.normalized_probe_metrics(arrays["v_hat"], arrays["v_target"], arrays["mask"])
    assert metrics["namespace"] == "NORMALIZED_LOCAL_FLOW"
    assert set(metrics["blocks"]) == {"geometry_xyz", "geometry_rot6d", "timing", "gripper"}


def test_raw_before_derived_probe_evidence(tmp_path: Path) -> None:
    noise = np.zeros(adapter.RAW_SHAPE, dtype=np.float32)
    x1 = np.full(adapter.RAW_SHAPE, 0.25, dtype=np.float32)
    arrays = adapter.build_probe_arrays(noise, x1, adapter.T_VALUES[0], np.zeros(adapter.RAW_SHAPE, dtype=np.float32))
    result = adapter.write_probe_evidence(
        tmp_path,
        probe_id="seed2223_t067",
        identity={
            "checkpoint": adapter.CHECKPOINT,
            "checkpoint_tree_sha256": adapter.CHECKPOINT_TREE_SHA256,
            "stats_sha256": adapter.STATS_SHA256,
            "window_id": adapter.WINDOW_ID,
            "noise_seed": adapter.SEED,
            "t": adapter.T_VALUES[0],
            "t_bucket": adapter.T_BUCKETS[0],
        },
        arrays=arrays,
        physical_metrics={"namespace": "DECODED_PHYSICAL_ACTION"},
    )
    assert result["raw_manifest"]["derived"] is False
    assert result["derived"]["raw_before_derived"] is True
    assert (tmp_path / "probes/seed2223_t067/probe_manifest.json").is_file()
    assert (tmp_path / "probes/seed2223_t067/derived_metrics.json").is_file()


def test_foreign_cwd_empty_pythonpath_plan_only(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/stage45d_d1_corelab_runtime_adapter.py"), "--plan-only"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert result["preflight_profile"]["total_direct_model_side_calls"] == 4


def test_execute_requires_future_authority_before_model_import(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage45d_d1_corelab_runtime_adapter.py"),
            "--execute",
            "--runtime-binding",
            str(tmp_path / "missing.json"),
        ],
        cwd=tmp_path,
        env={**os.environ, "STAGE45D_D1_CORELAB_PREFLIGHT_RUNTIME": "authorized"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "regular file" in completed.stderr or "required" in completed.stderr
    assert "model-side import" not in completed.stderr
