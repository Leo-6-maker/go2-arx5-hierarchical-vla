from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from go2arx5_vla.metrics.h2_paired import compare_h2_profile_payloads
from scripts.stage22_audit_h2_run_integrity import audit_h2_run
from scripts.stage22_checkpoint_provenance import seal_checkpoint, verify_checkpoint


def _stats(value: float) -> dict[str, float]:
    return {"mean": value, "p95": value, "max": value}


def _profile_record(input_id: str, seed: int, value: float) -> dict:
    return {
        "input_id": input_id,
        "raw_episode_id": "episode-0",
        "target_id": "red_target_holdout_e",
        "finite": True,
        "position_error_m": _stats(value),
        "rotation_error_rad": _stats(value / 10.0),
        "endpoint_position_error_m": value * 1.1,
        "endpoint_rotation_error_rad": value / 20.0,
        "path_length_ratio": 1.0 + value,
        "mean_speed_ratio": 1.0 + value / 2.0,
        "max_speed_ratio": 1.0 + value / 3.0,
        "roughness_ratio": 1.0 + value / 4.0,
        "predicted_profile": {
            "initial_position_discontinuity_m": value / 5.0,
            "initial_rotation_discontinuity_rad": value / 6.0,
        },
        "safety_gate": {"decision": "ACCEPT"},
    }


def _overlap(value: float) -> dict:
    return {
        "raw_episode_id": "episode-0",
        "previous_observation_timestamp_s": 0.0,
        "current_observation_timestamp_s": 0.2,
        "position_error_m": _stats(value),
        "rotation_error_rad": _stats(value / 10.0),
        "translational_tangent_error_mps": _stats(value * 2.0),
        "old_prediction_at_new_observation_vs_new_first": {
            "position_error_m": value * 3.0,
            "rotation_error_rad": value / 8.0,
        },
    }


def _profile_payload(seed: int, value: float) -> dict:
    return {
        "schema": "STAGE22_ACTION_PROFILE_EVALUATION_V1",
        "split": "validation",
        "inference_seed": seed,
        "records": [_profile_record("episode-0:expert-0000", seed, value)],
        "chunk_overlap_metrics": [_overlap(value)],
    }


def test_h2_paired_comparison_aligns_snapshot_and_seed_and_emits_deltas() -> None:
    result = compare_h2_profile_payloads(
        [_profile_payload(2223, 0.10), _profile_payload(2224, 0.20)],
        [_profile_payload(2223, 0.05), _profile_payload(2224, 0.15)],
        expected_split="validation",
    )
    assert result["status"] == "PASS"
    assert result["paired_alignment"]["paired_record_count"] == 2
    assert result["paired_alignment"]["paired_overlap_count"] == 2
    assert result["inference_seeds"] == [2223, 2224]
    assert result["paired_records"][0]["delta_r1_minus_r0"]["position_mean_error_m"] < 0.0
    assert result["paired_records"][0]["delta_r1_minus_r0"]["path_length_ratio_abs_deviation"] < 0.0
    assert result["training_loss_used_for_selection"] is False
    assert result["trajectory_correction"] is False


def test_h2_paired_comparison_rejects_missing_snapshot_seed_pair() -> None:
    with pytest.raises(ValueError, match="pairing mismatch"):
        compare_h2_profile_payloads(
            [_profile_payload(2223, 0.10)],
            [_profile_payload(2224, 0.10)],
            expected_split="validation",
        )


def _run_manifest(*, recipe: str, log_path: Path) -> dict:
    return {
        "schema": "STAGE20S_GR00T_TRAINING_RUN_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "base_model_path": "/scratch/base",
        "base_model_index_sha256": "a" * 64,
        "dataset_dir": "/scratch/train",
        "groot_repo_sha": "b" * 40,
        "action_horizon": 40,
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
        "tune_projector": True,
        "tune_diffusion_model": recipe == "R1",
        "tune_llm": False,
        "tune_visual": False,
        "runtime": {"python": "3.12.3", "cuda_visible_devices": "0"},
        "cuda_telemetry": {
            "available": True,
            "device_name": "NVIDIA H200",
            "torch_version": "2.9.0+cu128",
            "cuda_version": "12.8",
            "before": {"total_memory_bytes": 143 * 1024**3},
        },
        "log_path_for_test": str(log_path),
    }


def _comparison_fixture(tmp_path: Path) -> Path:
    split_lock = {
        "dataset_role": "TRAIN",
        "selected_split": "train",
        "selected_target_ids": [
            "red_target_train_a",
            "red_target_train_b",
            "red_target_train_c",
            "red_target_train_d",
        ],
        "training_allowed": True,
        "final_test_in_training": False,
    }
    split_path = tmp_path / "stage22_split_lock.json"
    split_path.write_text(json.dumps(split_lock), encoding="utf-8")
    dataset_fingerprint = {
        "tree_sha256": "c" * 64,
        "file_count": 12,
        "total_bytes": 3456,
    }
    results = []
    for recipe in ("R0", "R1"):
        lower = recipe.lower()
        log_path = tmp_path / f"{lower}.log"
        trainable = "1,419,078,784" if recipe == "R1" else "123,456"
        log_path.write_text(f"Trainable parameters: {trainable} (x)\n", encoding="utf-8")
        run_path = tmp_path / f"{lower}_run.json"
        run = _run_manifest(recipe=recipe, log_path=log_path)
        run_path.write_text(json.dumps(run), encoding="utf-8")
        command = [
            "python",
            "stage20s_train_gr00t.py",
            "--stage10-root",
            "/scratch/runtime",
            "--base-model-path",
            "/scratch/base",
            "--dataset-dir",
            "/scratch/train",
            "--modality-config",
            "/scratch/modality.py",
            "--output-dir",
            f"/scratch/{lower}",
            "--max-steps",
            "50",
            "--save-steps",
            "25",
            "--learning-rate",
            "0.0001",
            "--seed",
            "2222",
            "--local-files-only",
        ]
        if recipe == "R1":
            command.append("--tune-diffusion-model")
        results.append(
            {
                "recipe_id": recipe,
                "label": lower,
                "status": "PASS",
                "max_steps": 50,
                "save_steps": 25,
                "command": command,
                "launcher_log": str(log_path),
                "run_manifest": str(run_path),
                "dataset_fingerprint": dataset_fingerprint,
            }
        )
    comparison = {
        "schema": "STAGE22_RECIPE_COMPARISON_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "r1_preflight_failed": False,
        "split_lock": str(split_path),
        "split_lock_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "training_target_ids": split_lock["selected_target_ids"],
        "dataset_fingerprint": dataset_fingerprint,
        "resource_fingerprint": {
            "available": True,
            "cuda_visible_devices": "0",
            "devices": [{"index": 0, "name": "NVIDIA H200", "uuid": "GPU-test", "memory_total_mib": 146432}],
        },
        "results": results,
    }
    comparison_path = tmp_path / "stage22_recipe_comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    return comparison_path


def test_h2_integrity_auditor_accepts_matched_fixture(tmp_path: Path) -> None:
    comparison_path = _comparison_fixture(tmp_path)
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    result = audit_h2_run(payload, comparison_path=comparison_path)
    assert result["status"] == "PASS", result["blocking_checks"]
    assert result["blocking_checks"] == []


def test_h2_integrity_auditor_rejects_recipe_flag_drift(tmp_path: Path) -> None:
    comparison_path = _comparison_fixture(tmp_path)
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    r1_path = Path(payload["results"][1]["run_manifest"])
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))
    r1["tune_visual"] = True
    r1_path.write_text(json.dumps(r1), encoding="utf-8")
    result = audit_h2_run(payload, comparison_path=comparison_path)
    assert result["status"] == "FAIL"
    assert "r1_tune_visual" in result["blocking_checks"]


def test_checkpoint_provenance_seal_and_verify(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{\"step\":50}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"synthetic-checkpoint")
    log_path = tmp_path / "run.log"
    log_path.write_text("Trainable parameters: 123,456 (x)\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps(_run_manifest(recipe="R0", log_path=log_path)), encoding="utf-8")
    provenance_path = tmp_path / "checkpoint.provenance.json"
    sealed = seal_checkpoint(
        checkpoint,
        run_manifest=run_manifest,
        recipe="R0",
        output=provenance_path,
    )
    assert sealed["status"] == "SEALED_FOR_HASH_VERIFICATION"
    assert sealed["selection_status"] == "CANDIDATE_NOT_SELECTED"
    assert sealed["transfer_performed"] is False
    assert verify_checkpoint(checkpoint, provenance_path)["status"] == "PASS"
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    assert verify_checkpoint(checkpoint, provenance_path)["status"] == "FAIL"
