#!/usr/bin/env python3
"""CPU-only finalization and C_T admission audit for an existing Stage-A run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pi0243_checkpoint_persistence import checkpoint_record, verify_receipt, write_json_durable


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finalize_training(card: dict[str, Any], root: Path) -> dict[str, Any]:
    training = root / "training"
    schedule = read_json(root / "stage_exposure_schedule.json")["rows"]
    consumption = read_jsonl(training / "pi0241_sample_consumption.jsonl")
    contracts = read_jsonl(training / "pi0241_sample_contracts.jsonl")
    if len(schedule) != 5000 or len(consumption) != len(schedule):
        raise RuntimeError("sealed 5000-step schedule/consumption count mismatch")
    for expected, actual in zip(schedule, consumption):
        for key in ("source_episode_id", "episode_index", "fixed_index"):
            if str(expected[key]) != str(actual[key]):
                raise RuntimeError(f"sample exposure mismatch at step {expected['optimizer_step']}: {key}")
    if len(contracts) != 16 or len({row["source_episode_id"] for row in contracts}) != 16:
        raise RuntimeError("fixed16 contract count or identity mismatch")
    for row in contracts:
        if row.get("action_shape") != [1, 40, 132] or row.get("mask_shape") != [1, 40, 132]:
            raise RuntimeError(f"action/mask shape mismatch: {row['source_episode_id']}")
        if row.get("active_slots") != list(range(10)):
            raise RuntimeError(f"active slot mismatch: {row['source_episode_id']}")

    storage = card.get("storage")
    if not isinstance(storage, dict) or storage.get("checkpoint_family") != "PI0243_STAGE_A_V1":
        raise RuntimeError("Stage-A storage family mismatch")
    checkpoints: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for step in (1000, 2500, 5000):
        checkpoint = training / f"checkpoint-{step}"
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        checkpoints[str(step)] = checkpoint_record(checkpoint, expected_step=step, expected_max_steps=5000, family="PI0243_STAGE_A_V1")
        receipts[str(step)] = verify_receipt(
            checkpoint,
            receipt_path,
            expected_step=step,
            expected_max_steps=5000,
            family="PI0243_STAGE_A_V1",
        )

    audit = read_json(training / "pi0241_update_audit.json")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != 5000:
        raise RuntimeError("update audit is not complete at step 5000")
    final = audit.get("final_parameters", {})
    update_integrity = {
        "projector_updated": float(final.get("projector", {}).get("delta_l2", 0.0)) > 0.0,
        "diffusion_updated": float(final.get("diffusion", {}).get("delta_l2", 0.0)) > 0.0,
        "frozen_backbone_unchanged": float(final.get("frozen_backbone", {}).get("delta_l2", 1.0)) == 0.0,
        "frozen_vlln_unchanged": float(final.get("frozen_vlln", {}).get("delta_l2", 1.0)) == 0.0,
        "optimizer_membership_valid": not audit.get("optimizer_membership", {}).get("missing_trainable")
        and not audit.get("optimizer_membership", {}).get("illegal_frozen"),
    }
    manifest = {
        "schema": "PI0243_STAGE_TRAINING_MANIFEST_V1",
        "status": "PASS_UPDATE_INTEGRITY" if all(update_integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "stage": card["stage"]["stage"],
        "finalized_after_training": True,
        "finalization_mode": "CPU_READ_ONLY_FROM_SEALED_ARTIFACTS",
        "canonical_optimizer_steps": 5000,
        "actual_sample_consumption": len(consumption),
        "schedule_sha256": card["schedule"]["sha256"],
        "config": card["resolved_config"],
        "update_integrity": update_integrity,
        "checkpoints": checkpoints,
        "durable_receipts": receipts,
        "storage": storage,
        "sample_contract_count": len(contracts),
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json_durable(root / "pi0243_training_manifest.json", manifest)
    if not all(update_integrity.values()):
        raise RuntimeError("update integrity failed")
    return manifest


def audit_screen_contract(card: dict[str, Any], root: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    code_root = Path(card["paths"]["code_root"])
    stage10_root = Path(card["paths"]["stage10_root"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(stage10_root / "src" / "isaac-gr00t")]
    legacy = load_module(code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0243_cpu_legacy")
    legacy.bind_runtime(card)
    stage21 = load_module(code_root / "scripts" / "stage21_same_scene_groot_inference.py", "pi0243_cpu_stage21")
    processor, embodiment = stage21._build_processor(
        stage10_root,
        Path(card["paths"]["processor_base_model"]),
        Path(card["paths"]["train_dataset"]) / "meta",
        Path(card["paths"]["modality_config"]),
    )
    pi0228 = load_module(code_root / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0243_cpu_pi0228")
    stage = load_module(code_root / "scripts" / "pi0243_staged_learning_to_sim.py", "pi0243_cpu_stage")
    config, resolved = stage.resolved_config(card, root / "cpu_contract_cfg", legacy, pi0228)
    if resolved != card["resolved_config"]:
        raise RuntimeError("training config summary differs from sealed card")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    spec = config.data.datasets[0]
    dataset = ShardedSingleStepDataset(
        dataset_path=Path(card["paths"]["train_dataset"]),
        embodiment_tag=EmbodimentTag(spec.embodiment_tag),
        modality_configs=config.data.modality_configs[spec.embodiment_tag],
        shard_size=config.data.shard_size,
        episode_sampling_rate=config.data.episode_sampling_rate,
        seed=config.data.seed,
        allow_padding=config.data.allow_padding,
    )
    dataset.set_processor(processor)
    contracts = {row["source_episode_id"]: row for row in read_jsonl(root / "training" / "pi0241_sample_contracts.jsonl")}
    rows = read_json(root / "stage_cohort.json")["rows"]
    observed = []
    for row in rows:
        dp = dataset.get_datapoint(dataset.episode_loader[int(row["episode_index"])], 0)
        action = np.asarray(dp["action"])
        mask = np.asarray(dp["action_mask"])
        expected = contracts.get(row["source_episode_id"])
        if expected is None or action.shape != (1, 40, 132) or mask.shape != (1, 40, 132):
            raise RuntimeError(f"screen/training sample mismatch: {row['source_episode_id']}")
        if sha256_array(action) != expected["action_sha256"] or sha256_array(mask) != expected["mask_sha256"]:
            raise RuntimeError(f"screen/training tensor hash mismatch: {row['source_episode_id']}")
        observed.append({
            "source_episode_id": row["source_episode_id"],
            "episode_index": int(row["episode_index"]),
            "action_sha256": sha256_array(action),
            "mask_sha256": sha256_array(mask),
            "action_shape": list(action.shape),
            "mask_shape": list(mask.shape),
            "active_slots": expected["active_slots"],
            "state_shape": list(np.asarray(dp["state"]).shape),
            "image_shapes": [list(np.asarray(image).shape) for image in dp["vlm_content"]["images"]],
        })
    modality_path = Path(card["paths"]["modality_config"])
    report = {
        "schema": "PI0243_STAGE_A_SCREEN_ADMISSION_V1",
        "status": "PASS_C_T_TENSOR_PARITY",
        "screen_job": "627524.hopper-m-02",
        "screen_code_sha256": sha256_file(code_root / "scripts" / "pi0243_staged_learning_to_sim.py"),
        "stage28_helper_sha256": sha256_file(code_root / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"),
        "processor_identity": card["processor"],
        "processor_embodiment": str(embodiment),
        "modality_sha256": sha256_file(modality_path),
        "processor_config_sha256": sha256_file(Path(card["paths"]["processor_base_model"]) / "processor_config.json"),
        "stats_sha256": {
            "flat": sha256_file(Path(card["paths"]["train_dataset"]) / "meta" / "stats.json"),
            "relative": sha256_file(Path(card["paths"]["train_dataset"]) / "meta" / "relative_stats.json"),
            "model": sha256_file(Path(card["paths"]["processor_base_model"]) / "statistics.json"),
        },
        "training_config_summary_equal": True,
        "fixed16_count": len(observed),
        "tensor_hash_matches": True,
        "observed": observed,
        "target_xyz_in_model_input": False,
        "model_activity": {"weight_loads": 0, "forward": 0, "backward": 0, "optimizer": 0, "WBC": 0, "sim": 0},
        "selected_checkpoint": "NONE",
    }
    write_json_durable(root / "stage_a_screen_admission.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    args = parser.parse_args()
    card = read_json(args.run_card)
    root = Path(card["paths"]["output_root"])
    manifest = finalize_training(card, root)
    admission = audit_screen_contract(card, root)
    print(json.dumps({"manifest": manifest, "screen_admission": admission}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
