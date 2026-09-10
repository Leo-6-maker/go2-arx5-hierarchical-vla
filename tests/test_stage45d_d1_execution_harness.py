"""CPU/static regression tests for the Stage45D D1 execution harness."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d1_teacher_forced_local_flow as harness  # noqa: E402


def test_reviewed_protocol_and_execution_manifest_are_hash_bound() -> None:
    protocol = harness.validate_reviewed_protocol()
    execution = harness.validate_execution_manifest()
    assert protocol["schema"] == "CORELAB_STAGE45D_D1_TEACHER_FORCED_DIAGNOSTIC_PROTOCOL_CORRECTED_V1"
    assert execution["protocol"]["sha256"] == harness.PROTOCOL_SHA256
    assert execution["runner"]["sha256"] == harness.sha256_file(harness.REPO / execution["runner"]["path"])


def test_exact_t16_mask_and_legacy_dense_mask_rejection() -> None:
    result = harness.validate_mask_contract()
    assert result["shape"] == [40, 132]
    assert result["active_scalar_count"] == 190
    assert result["block_counts"] == {"geometry": 135, "timing": 15, "gripper": 40, "remaining": 0}
    assert result["legacy_first_ten_rejected"] is True


def test_flow_law_and_oracle_implied_endpoint_recovery() -> None:
    result = harness.validate_flow_algebra()
    assert result["oracle_v_hat_equals_v_target"] is True
    assert all(row["max_abs_oracle_endpoint_error"] <= 2e-6 for row in result["rows"])

    x0 = np.zeros(harness.RAW_SHAPE, dtype=np.float32)
    x1 = np.ones(harness.RAW_SHAPE, dtype=np.float32)
    _, x_t, v_target = harness.construct_flow(x0, x1, harness.T_VALUES[1])
    x1_hat = harness.implied_endpoint(x_t, v_target, harness.T_VALUES[1])
    assert np.allclose(x1_hat, x1, atol=2e-6, rtol=0.0)


def test_checkpoint_specific_stats_routes_and_full24_historical_rejection() -> None:
    result = harness.validate_stats_routing()
    assert result["routes"]["STRUCTURED@1000"]["route"] == "HISTORICAL_STAGE40_FIXED_STATS"
    assert result["routes"]["FULL24@1000"]["route"] == "STAGE45C_RUNTIME_V2_STATS_BRIDGE"
    assert result["routes"]["FULL24@2000"]["route"] == "STAGE45C_RUNTIME_V2_STATS_BRIDGE"
    assert result["full24_historical_route_rejected"] is True


def test_implied_endpoint_uses_v2_inverse_normalization_bridge() -> None:
    x1_hat = np.full(harness.RAW_SHAPE, 0.5, dtype=np.float32)
    lower = np.zeros((132,), dtype=np.float64)
    upper = np.ones((132,), dtype=np.float64)
    physical_input, details = harness.inverse_normalize_implied_endpoint(
        "FULL24@1000",
        x1_hat,
        lower,
        upper,
        harness.structured_action_mask("T16"),
        stats_sha256=harness.FULL24_STATS_SHA256,
        stats_payload_sha256=harness.FULL24_STATS_PAYLOAD_SHA256,
    )
    assert details["stats_route"] == "STAGE45C_RUNTIME_V2_STATS_BRIDGE"
    assert details["normalized_source"] == "x1_hat"
    assert np.allclose(physical_input[harness.structured_action_mask("T16") == 1.0], 0.75)
    assert np.all(physical_input[harness.structured_action_mask("T16") == 0.0] == 0.0)
    with pytest.raises(harness.HarnessError, match="V2 payload SHA"):
        harness.inverse_normalize_implied_endpoint(
            "FULL24@1000",
            x1_hat,
            lower,
            upper,
            harness.structured_action_mask("T16"),
            stats_sha256=harness.FULL24_STATS_SHA256,
            stats_payload_sha256="wrong",
        )


def test_target_exclusion_and_exact_call_budget() -> None:
    protocol = harness.validate_reviewed_protocol()
    assert protocol["scientific_scope"]["target_xyz_in_model_input"] is False
    assert protocol["contract"]["target_xyz_role"].startswith("evaluation/supervision metadata only")
    assert harness.validate_call_budget() == {
        "action_head_calls": 648,
        "conditioning_passes": 72,
        "total_model_side_calls": 720,
    }


def test_raw_before_derived_and_failure_seal() -> None:
    result = harness.validate_evidence_lifecycle()
    assert result == {
        "raw_before_derived": True,
        "derived_before_raw_rejected": True,
        "failure_seal_written": True,
    }


def test_foreign_cwd_and_empty_pythonpath_plan_only_bootstrap(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stage45d_d1_teacher_forced_local_flow.py"), "--plan-only"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["verdict"] == "PASS"
    assert result["call_budget"]["total_model_side_calls"] == 720


def test_plan_runner_has_no_top_level_torch_gr00t_or_cuda_import() -> None:
    tree = ast.parse((ROOT / "scripts" / "stage45d_d1_teacher_forced_local_flow.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert not imported.intersection({"torch", "gr00t", "cuda"})


def test_missing_or_malformed_runtime_identity_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(harness.RUNTIME_BINDING_PATH.read_text(encoding="utf-8"))
    payload["paths"].pop("output_root")
    malformed = tmp_path / "binding.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="path key set drift"):
        harness.validate_runtime_binding_template(malformed)
