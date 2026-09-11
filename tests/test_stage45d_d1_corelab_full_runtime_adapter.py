"""CPU/static regression tests for the common Corelab smoke/full runtime."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d1_corelab_full_runtime_adapter as full  # noqa: E402


def _row(checkpoint: str, window: str, target: str, seed: int, bucket: int) -> dict:
    metric = {
        "initial_target_distance_m": 1.0,
        "minimum_target_distance_m": 0.8,
        "final_target_distance_m": 0.9,
        "absolute_progress_m": 0.1,
        "best_progress_m": 0.2,
        "progress_sign": "POSITIVE",
    }
    return {
        "checkpoint": checkpoint,
        "window_id": window,
        "target_id": target,
        "seed": seed,
        "t_bucket": bucket,
        "trajectory_metrics": {"target": metric},
    }


def test_full_profile_and_static_contract() -> None:
    profile = full.validate_profile()
    assert profile["execution"]["total_model_side_calls"] == 720
    assert profile["resource_margin"]["minimum_free_bytes"] == 4 * 1024**3
    result = full.validate_static_contract()
    assert result["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert result["execution_boundary"]["gpu"] == 0
    assert result["static_guard"]["status"] == "PASS"


def test_full_static_guard_has_no_model_import_or_forbidden_call() -> None:
    guard = full._static_guard()
    assert guard["status"] == "PASS"
    assert guard["top_level_forbidden_imports"] == []
    assert guard["forbidden_call_hits"] == []
    assert all(guard["shared_primitives"].values())


def test_aggregate_is_exactly_paired_and_descriptive_only() -> None:
    rows = []
    for checkpoint in full.CHECKPOINT_ORDER:
        for index, window in enumerate(full.TARGET_IDS):
            target = window.split(":", 1)[0].removeprefix("stage22_expert_")
            for seed in full.SEEDS:
                for bucket in full.T_BUCKETS:
                    rows.append(_row(checkpoint, window, target, seed, bucket))
    summary = full._aggregate_rows(rows)
    assert summary["row_count"] == 648
    assert summary["paired"]["coverage_comparison"]["matched_rows"] == 216
    assert summary["paired"]["exposure_comparison"]["matched_rows"] == 216
    assert summary["status"] == "DESCRIPTIVE_ONLY_NO_THRESHOLD"


def test_full_profile_rejects_nonzero_current_boundary(tmp_path: Path) -> None:
    payload = json.loads(full.FULL_PROFILE_PATH.read_text(encoding="utf-8"))
    payload["execution_boundary_now"]["gpu"] = 1
    malformed = tmp_path / "profile.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(full.FullAdapterError, match="current boundary"):
        full.validate_profile(malformed)


def test_orchestrator_has_no_top_level_model_imports() -> None:
    tree = ast.parse((ROOT / "scripts/run_stage45d_d1_corelab.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert not imported.intersection({"torch", "transformers", "gr00t", "cuda"})
