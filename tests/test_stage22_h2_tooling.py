from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from go2arx5_vla.data.stage22_snapshot_lock import (
    EXPECTED_INFERENCE_SEEDS,
    EXPECTED_VALIDATION_SNAPSHOT_IDS,
    snapshot_ids_sha256,
)
from go2arx5_vla.metrics.h2_paired import compare_h2_profile_payloads
from scripts.stage22_audit_h2_direct_launch import (
    EXPECTED_BASE_MODEL_TREE_SHA256,
    EXPECTED_CHECKPOINT_LINEAGE_FIELDS,
    EXPECTED_COSMOS_REPO_ID,
    EXPECTED_COSMOS_REVISION,
    EXPECTED_COSMOS_TREE_SHA256,
    FROZEN_LAUNCHER_FILE_SHA256,
    EXPECTED_GROOT_REPO_SHA,
    EXPECTED_INVENTORY_COLLECTION_METHOD,
    EXPECTED_INVENTORY_MODEL_CONSTRUCTION,
    EXPECTED_MODALITY_CONFIG_SHA256,
    EXPECTED_SPLIT_LOCK_SHA256,
    EXPECTED_TRAIN_FIXTURE_TREE_SHA256,
    FROZEN_LAUNCHER_GIT_SHA,
    audit_direct_h2,
)
from scripts.stage22_audit_h2_run_integrity import audit_h2_run
from scripts.stage22_aggregate_profile_seeds import _aggregate
from scripts.stage22_checkpoint_provenance import seal_checkpoint, verify_checkpoint
from scripts.stage22_capture_gpu_evidence import validate_gpu_evidence
from scripts.stage22_probe_trainable_modules import (
    FROZEN_LAUNCHER_FILE_SHA256 as PROBE_LAUNCHER_FILE_SHA256,
)


def _stats(value: float) -> dict[str, float]:
    return {"mean": value, "p95": value, "max": value}


def _profile_record(input_id: str, value: float) -> dict:
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


def _overlap(value: float, index: int) -> dict:
    return {
        "raw_episode_id": "episode-0",
        "previous_observation_timestamp_s": 0.0,
        "current_observation_timestamp_s": 0.2 + 0.2 * index,
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
        "snapshot_manifest_sha256": "d" * 64,
        "records": [
            _profile_record(snapshot_id, value + index * 0.001)
            for index, snapshot_id in enumerate(EXPECTED_VALIDATION_SNAPSHOT_IDS)
        ],
        "chunk_overlap_metrics": [_overlap(value, index) for index in range(5)],
    }


def _snapshot_manifest() -> dict:
    snapshot_ids = list(EXPECTED_VALIDATION_SNAPSHOT_IDS)
    return {
        "schema": "STAGE22_VALIDATION_SNAPSHOT_LOCK_V1",
        "status": "LOCKED_BEFORE_H2_INFERENCE",
        "split": "validation",
        "target_ids": ["red_target_holdout_e"],
        "snapshot_ids": snapshot_ids,
        "snapshot_count": len(snapshot_ids),
        "expected_overlap_count": 5,
        "snapshot_ids_sha256": snapshot_ids_sha256(snapshot_ids),
    }


def _write_gpu_evidence(tmp_path: Path, recipe: str) -> tuple[Path, str]:
    device = {
        "index": 0,
        "name": "NVIDIA H200",
        "uuid": "GPU-test",
        "memory_total_mib": 146432,
    }
    path = tmp_path / f"{recipe.lower()}_gpu_evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema": "STAGE22_GPU_EVIDENCE_V1",
                "status": "PASS",
                "source": "nvidia-smi",
                "command": [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                "query": "index,name,uuid,memory.total",
                "hostname": "hopper-h200-test",
                "captured_utc": "2026-09-02T12:00:00+00:00",
                "cuda_visible_devices": "0",
                "devices": [device],
                "selected_device": device,
                "raw_stdout_sha256": "f" * 64,
                "contains_credentials": False,
            }
        ),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_h2_paired_comparison_aligns_snapshot_and_seed_and_emits_deltas() -> None:
    result = compare_h2_profile_payloads(
        [_profile_payload(2223, 0.10), _profile_payload(2224, 0.20), _profile_payload(2225, 0.30)],
        [_profile_payload(2223, 0.05), _profile_payload(2224, 0.15), _profile_payload(2225, 0.25)],
        expected_split="validation",
        snapshot_manifest_path="manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json",
        snapshot_manifest_sha256="d" * 64,
        snapshot_manifest=_snapshot_manifest(),
    )
    assert result["status"] == "PASS"
    assert result["paired_alignment"]["paired_record_count"] == 18
    assert result["paired_alignment"]["paired_overlap_count"] == 15
    assert result["inference_seeds"] == list(EXPECTED_INFERENCE_SEEDS)
    assert result["paired_records"][0]["delta_r1_minus_r0"]["position_mean_error_m"] < 0.0
    assert result["paired_records"][0]["delta_r1_minus_r0"]["path_length_ratio_abs_deviation"] < 0.0
    assert result["training_loss_used_for_selection"] is False
    assert result["trajectory_correction"] is False


def test_h2_paired_comparison_rejects_missing_snapshot_seed_pair() -> None:
    r1_missing = _profile_payload(2225, 0.10)
    r1_missing["records"] = r1_missing["records"][:-1]
    with pytest.raises(ValueError, match="snapshot set"):
        compare_h2_profile_payloads(
            [_profile_payload(2223, 0.10), _profile_payload(2224, 0.10), _profile_payload(2225, 0.10)],
            [_profile_payload(2223, 0.10), _profile_payload(2224, 0.10), r1_missing],
            expected_split="validation",
            snapshot_manifest_path="manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json",
            snapshot_manifest_sha256="d" * 64,
            snapshot_manifest=_snapshot_manifest(),
        )


def test_h2_paired_comparison_rejects_incomplete_seed_set() -> None:
    with pytest.raises(ValueError, match="exactly inference seeds"):
        compare_h2_profile_payloads(
            [_profile_payload(2223, 0.10), _profile_payload(2224, 0.10)],
            [_profile_payload(2223, 0.10), _profile_payload(2224, 0.10)],
            expected_split="validation",
            snapshot_manifest_path="manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json",
            snapshot_manifest_sha256="d" * 64,
            snapshot_manifest=_snapshot_manifest(),
        )


def test_profile_seed_aggregation_binds_frozen_lock_and_exact_seeds(tmp_path: Path) -> None:
    lock_path = tmp_path / "validation_snapshot_lock.json"
    lock_path.write_text(json.dumps(_snapshot_manifest()), encoding="utf-8")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    inputs = []
    for seed in EXPECTED_INFERENCE_SEEDS:
        payload = _profile_payload(seed, 0.10)
        payload["recipe_model_path"] = "/scratch/r0"
        payload["snapshot_manifest_sha256"] = lock_sha
        path = tmp_path / f"profile_{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs.append(path)
    result = _aggregate(inputs, "R0", snapshot_manifest_path=lock_path)
    assert result["status"] == "PASS"
    assert result["seeds"]
    assert result["snapshot_count_per_seed"] == 6
    assert result["aggregate_overlap_count"] == 15

    with pytest.raises(ValueError, match="exactly inference seeds"):
        _aggregate(inputs[:2], "R0", snapshot_manifest_path=lock_path)


def _run_manifest(*, recipe: str, log_path: Path) -> dict:
    return {
        "schema": "STAGE20S_GR00T_TRAINING_RUN_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "max_steps": 50,
        "base_model_path": "/scratch/base",
        "base_model_index_sha256": "a" * 64,
        "dataset_dir": "/scratch/train",
        "modality_config": "/scratch/modality.py",
        "groot_repo_sha": "b" * 40,
        "action_horizon": 40,
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
        "tune_projector": True,
        "tune_diffusion_model": recipe == "R1",
        "tune_vlln": False,
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


def test_direct_launch_audit_accepts_frozen_h2_provenance(tmp_path: Path) -> None:
    snapshot_lock_path = tmp_path / "validation_snapshot_lock.json"
    snapshot_lock = _snapshot_manifest()
    snapshot_lock_path.write_text(json.dumps(snapshot_lock), encoding="utf-8")
    lock_sha = hashlib.sha256(snapshot_lock_path.read_bytes()).hexdigest()
    runtime = {
        "python": "3.12.3",
        "torch": "2.9.0+cu128",
        "cuda": "12.8",
        "device_name": "NVIDIA H200",
        "memory_total_mib": 146432,
    }
    runs = {}
    provenance_runs = {}
    for recipe in ("R0", "R1"):
        run_path = tmp_path / f"{recipe.lower()}_direct_run.json"
        run = _run_manifest(recipe=recipe, log_path=tmp_path / f"{recipe.lower()}.log")
        run["base_model_path"] = "/base"
        run["dataset_dir"] = "/train"
        run["modality_config"] = "/modality.py"
        run["groot_repo_sha"] = EXPECTED_GROOT_REPO_SHA
        run["runtime"] = {"python": runtime["python"], "cuda_visible_devices": "0"}
        run["cuda_telemetry"]["device_name"] = runtime["device_name"]
        run["cuda_telemetry"]["torch_version"] = runtime["torch"]
        run["cuda_telemetry"]["cuda_version"] = runtime["cuda"]
        run["cuda_telemetry"]["before"]["total_memory_bytes"] = runtime["memory_total_mib"] * 1024 * 1024
        run_path.write_text(json.dumps(run), encoding="utf-8")
        runs[recipe] = run_path
        run_sha = hashlib.sha256(run_path.read_bytes()).hexdigest()
        gpu_evidence_path, gpu_evidence_sha = _write_gpu_evidence(tmp_path, recipe)

        counts = {
            "projector": 100,
            "state": 200,
            "action": 300,
            "diffusion": 1419078184,
            "visual": 0,
            "llm": 0,
            "vlln": 0,
        }
        if recipe == "R0":
            counts["diffusion"] = 0
        inventory = {
            "schema": "STAGE22_TRAINABLE_MODULE_INVENTORY_V1",
            "status": "PASS",
            "recipe": recipe,
            "source_run_manifest_sha256": run_sha,
            "collection_method": EXPECTED_INVENTORY_COLLECTION_METHOD,
            "model_construction": EXPECTED_INVENTORY_MODEL_CONSTRUCTION,
            "frozen_launcher_git_sha": FROZEN_LAUNCHER_GIT_SHA,
            "base_model_path": "/base",
            "base_model_index_sha256": "a" * 64,
            "parameter_inventory_digest_sha256": "p" * 64,
            "config_flags": {
                "tune_projector": True,
                "tune_diffusion_model": recipe == "R1",
                "tune_visual": False,
                "tune_llm": False,
                "tune_vlln": False,
            },
            "coverage": {
                "complete": True,
                "unassigned_parameter_count": 0,
                "overlapping_parameter_count": 0,
            },
            "source": {
                "probe_script": "stage22_probe_trainable_modules.py",
                "probe_schema": "STAGE22_TRAINABLE_MODULE_INVENTORY_V1",
                "launcher_file_sha256": FROZEN_LAUNCHER_FILE_SHA256,
                "no_forward": True,
                "no_backward": True,
                "no_optimizer_step": True,
                "no_inference": True,
                "no_training": True,
            },
            "modules": [
                {
                    "role": role,
                    "name": f"model.{role}",
                    "all_requires_grad": role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())),
                    "parameter_count": count,
                    "requires_grad_parameter_count": count if role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())) else 0,
                    "trainable_parameter_count": count if role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())) else 0,
                    "parameter_names_sha256": "q" * 64,
                    "requires_grad_state_sha256": "r" * 64,
                }
                for role, count in counts.items()
            ],
            "total_trainable_parameters": sum(count for role, count in counts.items() if role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set()))),
        }
        inventory_path = tmp_path / f"{recipe.lower()}_module_inventory.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        command = [
            "python",
            "/frozen/scripts/stage20s_train_gr00t.py",
            "--stage10-root", "/runtime",
            "--base-model-path", "/base",
            "--dataset-dir", "/train",
            "--modality-config", "/modality.py",
            "--output-dir", f"/output/{recipe.lower()}",
            "--max-steps", "50",
            "--save-steps", "25",
            "--learning-rate", "0.0001",
            "--seed", "2222",
            "--local-files-only",
        ]
        if recipe == "R1":
            command.append("--tune-diffusion-model")
        provenance_runs[recipe] = {
            "run_manifest": str(run_path),
            "run_manifest_sha256": run_sha,
            "command": command,
            "module_inventory": str(inventory_path),
            "module_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "gpu_uuid": "GPU-test",
            "gpu_evidence": str(gpu_evidence_path),
            "gpu_evidence_sha256": gpu_evidence_sha,
        }

    provenance = {
        "schema": "STAGE22_H2_DIRECT_EXECUTION_PROVENANCE_V1",
        "status": "H2_DIRECT_LAUNCH_COMPLETE",
        "simulation_only": True,
        "hardware_control": False,
        "frozen_launcher_git_sha": FROZEN_LAUNCHER_GIT_SHA,
        "training_seed": 2222,
        "max_steps": 50,
        "save_steps": 25,
        "learning_rate": 1e-4,
        "action_horizon": 40,
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
        "local_files_only": True,
        "base_model_tree_sha256": EXPECTED_BASE_MODEL_TREE_SHA256,
        "train_fixture_tree_sha256": EXPECTED_TRAIN_FIXTURE_TREE_SHA256,
        "split_lock_sha256": EXPECTED_SPLIT_LOCK_SHA256,
        "modality_config_sha256": EXPECTED_MODALITY_CONFIG_SHA256,
        "groot_repo_sha": EXPECTED_GROOT_REPO_SHA,
        "cosmos": {
            "repo_id": EXPECTED_COSMOS_REPO_ID,
            "resolved_snapshot_revision": EXPECTED_COSMOS_REVISION,
            "tree_sha256": EXPECTED_COSMOS_TREE_SHA256,
        },
        "h1": {"status": "PASS", "optimizer_step_completed": True},
        "inference_seeds": list(EXPECTED_INFERENCE_SEEDS),
        "validation_snapshot_manifest_sha256": lock_sha,
        "validation_snapshot_ids_sha256": snapshot_lock["snapshot_ids_sha256"],
        "validation_snapshot_ids": snapshot_lock["snapshot_ids"],
        "gpu": {"uuid": "GPU-test", "r0_uuid": "GPU-test", "r1_uuid": "GPU-test"},
        "runtime": runtime,
        "runs": provenance_runs,
        "checkpoint50_lineage_contract": {
            "required_global_step": 50,
            "required_fields": sorted(EXPECTED_CHECKPOINT_LINEAGE_FIELDS),
        },
    }
    provenance_path = tmp_path / "h2_direct_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    result = audit_direct_h2(
        r0_run_path=runs["R0"],
        r1_run_path=runs["R1"],
        execution_provenance_path=provenance_path,
        validation_snapshot_manifest_path=snapshot_lock_path,
    )
    assert result["status"] == "PASS", result["blocking_checks"]
    assert all(result["required_gate_labels"].values())

    provenance["train_fixture_tree_sha256"] = "x" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    drift_result = audit_direct_h2(
        r0_run_path=runs["R0"],
        r1_run_path=runs["R1"],
        execution_provenance_path=provenance_path,
        validation_snapshot_manifest_path=snapshot_lock_path,
    )
    assert drift_result["status"] == "FAIL"
    assert "TRAIN_FIXTURE_TREE_HASH_GATE" in drift_result["blocking_checks"]


def test_gpu_evidence_validator_rejects_tampered_device_row(tmp_path: Path) -> None:
    path, _ = _write_gpu_evidence(tmp_path, "R0")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_gpu_evidence(payload)
    payload["selected_device"]["memory_total_mib"] += 1
    with pytest.raises(ValueError, match="selected_device"):
        validate_gpu_evidence(payload)


def test_probe_and_pbs_bind_the_frozen_direct_launcher() -> None:
    launcher = Path(__file__).parents[1] / "scripts" / "stage20s_train_gr00t.py"
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == PROBE_LAUNCHER_FILE_SHA256
    pbs = (
        Path(__file__).parents[1] / "hpc" / "pbs" / "stage22_h200_r0_r1_50.pbs"
    ).read_text(encoding="utf-8")
    assert "stage20s_train_gr00t.py" in pbs
    assert "python scripts/stage22_train_recipes.py" not in pbs
    assert "--local-files-only" in pbs
    assert "--r1-preflight-steps" not in pbs


def test_direct_launch_audit_rejects_base_path_drift(tmp_path: Path) -> None:
    snapshot_lock_path = tmp_path / "validation_snapshot_lock.json"
    snapshot_lock = _snapshot_manifest()
    snapshot_lock_path.write_text(json.dumps(snapshot_lock), encoding="utf-8")
    lock_sha = hashlib.sha256(snapshot_lock_path.read_bytes()).hexdigest()
    runtime = {
        "python": "3.12.3",
        "torch": "2.9.0+cu128",
        "cuda": "12.8",
        "device_name": "NVIDIA H200",
        "memory_total_mib": 146432,
    }
    runs = {}
    provenance_runs = {}
    for recipe in ("R0", "R1"):
        run_path = tmp_path / f"{recipe.lower()}_drift_run.json"
        run = _run_manifest(recipe=recipe, log_path=tmp_path / f"{recipe.lower()}_drift.log")
        run["base_model_path"] = "/base-r0" if recipe == "R0" else "/base-r1"
        run["dataset_dir"] = "/train"
        run["modality_config"] = "/modality.py"
        run["runtime"] = {"python": runtime["python"], "cuda_visible_devices": "0"}
        run["cuda_telemetry"]["device_name"] = runtime["device_name"]
        run["cuda_telemetry"]["torch_version"] = runtime["torch"]
        run["cuda_telemetry"]["cuda_version"] = runtime["cuda"]
        run["cuda_telemetry"]["before"]["total_memory_bytes"] = runtime["memory_total_mib"] * 1024 * 1024
        run_path.write_text(json.dumps(run), encoding="utf-8")
        runs[recipe] = run_path
        run_sha = hashlib.sha256(run_path.read_bytes()).hexdigest()
        inventory = {
            "schema": "STAGE22_TRAINABLE_MODULE_INVENTORY_V1",
            "status": "PASS",
            "recipe": recipe,
            "source_run_manifest_sha256": run_sha,
            "modules": [
                {
                    "role": role,
                    "name": f"model.{role}",
                    "all_requires_grad": role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())),
                    "parameter_count": (1419078781 if role == "diffusion" and recipe == "R1" else 1) if role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())) else 0,
                    "trainable_parameter_count": (1419078781 if role == "diffusion" and recipe == "R1" else 1) if role in ({"projector", "state", "action"} | ({"diffusion"} if recipe == "R1" else set())) else 0,
                }
                for role in ("projector", "state", "action", "diffusion", "visual", "llm", "vlln")
            ],
            "total_trainable_parameters": 1419078784 if recipe == "R1" else 3,
        }
        inventory_path = tmp_path / f"{recipe.lower()}_drift_inventory.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        command = [
            "python", "/frozen/scripts/stage20s_train_gr00t.py",
            "--base-model-path", run["base_model_path"], "--dataset-dir", "/train",
            "--modality-config", "/modality.py", "--max-steps", "50", "--save-steps", "25",
            "--learning-rate", "0.0001", "--seed", "2222", "--local-files-only",
        ]
        if recipe == "R1":
            command.append("--tune-diffusion-model")
        provenance_runs[recipe] = {
            "run_manifest": str(run_path),
            "run_manifest_sha256": run_sha,
            "command": command,
            "module_inventory": str(inventory_path),
            "module_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "gpu_uuid": "GPU-test",
        }
    provenance = {
        "schema": "STAGE22_H2_DIRECT_EXECUTION_PROVENANCE_V1",
        "status": "H2_DIRECT_LAUNCH_COMPLETE",
        "simulation_only": True,
        "hardware_control": False,
        "frozen_launcher_git_sha": FROZEN_LAUNCHER_GIT_SHA,
        "training_seed": 2222, "max_steps": 50, "save_steps": 25, "learning_rate": 1e-4,
        "action_horizon": 40, "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0", "local_files_only": True,
        "base_model_tree_sha256": EXPECTED_BASE_MODEL_TREE_SHA256,
        "train_fixture_tree_sha256": "c" * 64, "split_lock_sha256": "d" * 64,
        "modality_config_sha256": "e" * 64, "groot_repo_sha": "b" * 40,
        "cosmos": {"repo_id": EXPECTED_COSMOS_REPO_ID, "resolved_snapshot_revision": EXPECTED_COSMOS_REVISION, "tree_sha256": EXPECTED_COSMOS_TREE_SHA256},
        "h1": {"status": "PASS", "optimizer_step_completed": True},
        "inference_seeds": list(EXPECTED_INFERENCE_SEEDS),
        "validation_snapshot_manifest_sha256": lock_sha,
        "validation_snapshot_ids_sha256": snapshot_lock["snapshot_ids_sha256"],
        "validation_snapshot_ids": snapshot_lock["snapshot_ids"],
        "gpu": {"uuid": "GPU-test", "r0_uuid": "GPU-test", "r1_uuid": "GPU-test"},
        "runtime": runtime,
        "runs": provenance_runs,
        "checkpoint50_lineage_contract": {"required_global_step": 50, "required_fields": sorted(EXPECTED_CHECKPOINT_LINEAGE_FIELDS)},
    }
    provenance_path = tmp_path / "h2_drift_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    result = audit_direct_h2(
        r0_run_path=runs["R0"], r1_run_path=runs["R1"],
        execution_provenance_path=provenance_path,
        validation_snapshot_manifest_path=snapshot_lock_path,
    )
    assert result["status"] == "FAIL"
    assert "same_base_model_path" in result["blocking_checks"]


def test_checkpoint_provenance_seal_and_verify(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{\"step\":50}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"synthetic-checkpoint")
    global_step_evidence = tmp_path / "checkpoint-50.global_step_evidence.json"
    global_step_evidence.write_text(
        json.dumps(
            {
                "schema": "STAGE22_CHECKPOINT_GLOBAL_STEP_EVIDENCE_V1",
                "global_step": 50,
                "source": "synthetic trainer state",
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text("Trainable parameters: 123,456 (x)\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps(_run_manifest(recipe="R0", log_path=log_path)), encoding="utf-8")
    run_manifest_sha256 = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
    lineage_manifest = tmp_path / "h2_direct_audit.json"
    lineage_manifest.write_text(
        json.dumps(
            {
                "schema": "STAGE22_H2_DIRECT_LAUNCH_AUDIT_V1",
                "status": "PASS",
                "required_gate_labels": {"DIRECT_LAUNCH_AUDIT_SUPPORTED": True},
                "lineage": {
                    "base_model_tree_sha256": "a" * 64,
                    "train_fixture_tree_sha256": "b" * 64,
                    "split_lock_sha256": "c" * 64,
                    "modality_config_sha256": "d" * 64,
                    "groot_repo_sha": "e" * 40,
                    "cosmos_snapshot_revision": "f" * 40,
                    "cosmos_tree_sha256": "g" * 64,
                    "source_run_manifest_sha256": {"R0": run_manifest_sha256, "R1": "i" * 64},
                    "checkpoint_tree_sha256": None,
                },
            }
        ),
        encoding="utf-8",
    )
    provenance_path = tmp_path / "checkpoint.provenance.json"
    sealed = seal_checkpoint(
        checkpoint,
        run_manifest=run_manifest,
        recipe="R0",
        output=provenance_path,
        global_step_evidence=global_step_evidence,
        lineage_manifest=lineage_manifest,
    )
    assert sealed["status"] == "SEALED_FOR_HASH_VERIFICATION"
    assert sealed["selection_status"] == "CANDIDATE_NOT_SELECTED"
    assert sealed["transfer_performed"] is False
    assert sealed["global_step_evidence"]["outside_raw_checkpoint_tree"] is True
    assert all("global_step_evidence" not in entry["path"] for entry in sealed["entries"])
    assert verify_checkpoint(checkpoint, provenance_path)["status"] == "PASS"
    mismatched_lineage = json.loads(lineage_manifest.read_text(encoding="utf-8"))
    mismatched_lineage["lineage"]["source_run_manifest_sha256"]["R0"] = "z" * 64
    mismatched_lineage_path = tmp_path / "h2_direct_audit_mismatch.json"
    mismatched_lineage_path.write_text(json.dumps(mismatched_lineage), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        seal_checkpoint(
            checkpoint,
            run_manifest=run_manifest,
            recipe="R0",
            output=tmp_path / "mismatch.provenance.json",
            global_step_evidence=global_step_evidence,
            lineage_manifest=mismatched_lineage_path,
        )
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    assert verify_checkpoint(checkpoint, provenance_path)["status"] == "FAIL"
