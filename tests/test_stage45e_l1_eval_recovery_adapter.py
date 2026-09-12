"""CPU/filesystem tests for the recovered-L1 evaluation adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_l1_eval_recovery_adapter as adapter


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _recovered_input(tmp_path: Path) -> tuple[Path, Path]:
    training = tmp_path / "training"
    training.mkdir(parents=True)
    acceptance = tmp_path / "acceptance.json"
    _write_json(acceptance, {"schema": "acceptance"})
    value = {
        "schema": adapter.RECOVERED_INPUT_SCHEMA,
        "status": adapter.RECOVERED_INPUT_STATUS,
        "source_kind": adapter.RECOVERED_SOURCE_KIND,
        "acceptance_path": str(acceptance),
        "acceptance_sha256": adapter.sha256_file(acceptance),
        "training_root": str(training.resolve()),
        "historical_training_head": adapter.HISTORICAL_TRAINING_HEAD,
        "branch": adapter.BRANCH,
        "checkpoints": {
            "L1@1000": {
                "path": str((training / "checkpoint-1000").resolve()),
                "tree_sha256": "1" * 64,
                "global_step": 1000,
                "rehash_matches_acceptance": True,
            },
            "L1@2000": {
                "path": str((training / "checkpoint-2000").resolve()),
                "tree_sha256": "2" * 64,
                "global_step": 2000,
                "rehash_matches_acceptance": True,
            },
        },
        "scientific_contract": {
            "normalization_unchanged": True,
            "l1_runtime_contract": {"active_scalar_count": 190},
            "l1_code_contract": {
                "active_scalar_count": 190,
                "denominator": "binary_action_mask.sum()",
            },
            "training_recipe": {"seed": 2222, "max_steps": 2000},
        },
        "structured_stats": {
            "mask_active_scalar_count": 190,
            "normalization_unchanged": True,
        },
        "runtime_effective_weight_evidence_status": "NOT_CAPTURED_IN_HISTORICAL_CANARY",
        "evaluation_authorized_by_this_bridge": False,
        "final_train24_token_emitted": False,
    }
    path = tmp_path / "recovered-input.json"
    _write_json(path, value)
    return training, path


def test_validate_recovered_v2_input_binds_contract_and_checkpoints(tmp_path: Path) -> None:
    training, path = _recovered_input(tmp_path)
    result = adapter.validate_recovered_input(path, training)
    assert result["schema"] == "STAGE45E_L1_RECOVERY_EVAL_ADAPTER_INPUT_V2"
    assert result["checkpoints"]["L1@1000"]["tree_sha256"] == "1" * 64
    assert result["checkpoints"]["L1@2000"]["global_step"] == 2000
    assert result["scientific_contract"]["training_recipe"]["seed"] == 2222


def _rows(label: str, offset: float) -> list[dict]:
    rows = []
    targets = ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d")
    for target in targets:
        for index in range(6):
            rows.append(
                {
                    "window_id": f"stage22_expert_{target}:expert-{index:04d}",
                    "checkpoint": {"label": label},
                    "metrics": {
                        "target": {
                            "final_target_distance_m": 0.1 + offset,
                            "absolute_progress_m": 0.04 - offset,
                        }
                    },
                }
            )
    return rows


def test_complete_paired_summary_includes_exposure_pair() -> None:
    rows_by_arm = {
        "FULL24@2000": _rows("FULL24@2000", 0.0),
        "L1@1000": _rows("L1@1000", 0.01),
        "L1@2000": _rows("L1@2000", -0.01),
    }
    result = adapter.paired_summary_complete(rows_by_arm)
    assert set(result["pairs"]) == {
        "L1@1000_minus_FULL24@2000",
        "L1@2000_minus_FULL24@2000",
        "L1@2000_minus_L1@1000",
    }
    exposure = result["pairs"]["L1@2000_minus_L1@1000"]["overall"]
    assert exposure["window_count"] == 24
    assert exposure["metric_delta_left_minus_right"]["target.final_target_distance_m"]["mean"] == pytest.approx(-0.02)


def test_row_bundles_are_persisted_per_arm(tmp_path: Path) -> None:
    rows_by_arm = {
        "FULL24@2000": _rows("FULL24@2000", 0.0),
        "L1@1000": _rows("L1@1000", 0.01),
        "L1@2000": _rows("L1@2000", -0.01),
    }
    result = adapter.persist_row_bundles(tmp_path, rows_by_arm)
    assert result["FULL24@2000"]["row_count"] == 24
    assert result["L1@2000"]["row_count"] == 24
    assert (tmp_path / result["L1@2000"]["path"]).is_file()


def test_immutable_inventory_excludes_final_seal_and_result(tmp_path: Path) -> None:
    _write_json(tmp_path / "cpu_summary.json", {"ok": True})
    _write_json(tmp_path / "telemetry.json", {"ok": True})
    result = adapter.build_immutable_inventory(
        tmp_path,
        ["cpu_summary.json", "telemetry.json"],
    )
    assert result["entry_count"] == 2
    assert result["final_seal_excluded"] is True
    assert result["final_result_excluded"] is True
    with pytest.raises(adapter.L1EvalRecoveryAdapterError, match="must not be inventoried"):
        adapter.build_immutable_inventory(
            tmp_path / "second",
            ["stage45e_l1_eval_seal.json"],
        )
