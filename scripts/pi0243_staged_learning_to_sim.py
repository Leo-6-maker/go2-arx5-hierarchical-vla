#!/usr/bin/env python3
"""PI0243 staged dense-relative learning and frozen-sim handoff.

This is intentionally a thin stage controller around the already qualified
PI0228/Stage28 trainer and PI0241 fixed-schedule dataset hook.  It adds only
the preregistered 16/64/FULL views, stage cards, and independent screens.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


PROTOCOL_COMMIT = "5d6d843235625c416d372cdbed01b185a1e6da5f"
AUDITED_PARENT = "3ea3614444ddb76ebf1abe91be31744f0a6e6ded"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
SEED = 22301
LANGUAGE = "move the end effector to the red target"
DT = 0.05
SAVE_PROBE = 250
AUTH = {
    "a": "PI0243_STAGE_A_5K_GO",
    "b": "PI0243_STAGE_B_5K_GO",
    "c": "PI0243_STAGE_C_10K_GO",
}


class PI0243Error(RuntimeError):
    pass


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PI0243Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy_module(code_root: Path) -> Any:
    return load_module(code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0243_pi0241_legacy")


def pi0228_module(code_root: Path) -> Any:
    return load_module(code_root / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0243_pi0228")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _mapped_stage_path(card: Mapping[str, Any], raw_path: str | Path, env_name: str) -> Path:
    """Map only the sealed stage root; all scientific input paths stay untouched."""

    mirror = os.environ.get(env_name)
    source_root = Path(card["paths"]["output_root"])
    path = Path(raw_path)
    if not mirror:
        return path
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return path
    return Path(mirror) / relative


def runtime_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Use compute-visible mirrors without changing the sealed scientific card."""

    runtime = json.loads(json.dumps(card))
    metadata_root = os.environ.get("PI0243_METADATA_MIRROR_ROOT")
    output_root = os.environ.get("PI0243_OUTPUT_MIRROR_ROOT")
    if not metadata_root or not output_root:
        return runtime
    for key in ("cohort", "schedule"):
        runtime[key]["path"] = str(_mapped_stage_path(card, card[key]["path"], "PI0243_METADATA_MIRROR_ROOT"))
    runtime["paths"]["output_root"] = str(output_root)
    runtime["runtime_path_map"] = {
        "sealed_stage_root": str(card["paths"]["output_root"]),
        "metadata_mirror_root": str(metadata_root),
        "output_mirror_root": str(output_root),
        "scientific_paths_unchanged": True,
    }
    return runtime


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stage_spec(stage: str) -> dict[str, Any]:
    if stage == "a":
        return {"stage": "A", "view": "16", "max_steps": 5000, "checkpoints": [1000, 2500, 5000], "screens": [0, 1000, 2500, 5000]}
    if stage == "b":
        return {"stage": "B", "view": "64", "max_steps": 5000, "checkpoints": [1000, 2500, 5000], "screens": [0, 1000, 2500, 5000]}
    if stage == "c":
        return {"stage": "C", "view": "full", "max_steps": 10000, "checkpoints": [2500, 5000, 7500, 10000], "screens": [0, 1000, 2500, 5000, 7500, 10000]}
    raise PI0243Error(f"unknown stage: {stage}")


def source_rows(paths: Mapping[str, str], view: str, legacy: Any) -> tuple[list[dict[str, Any]], str]:
    cohort_path = Path(paths["p237_root"]) / "evidence" / "pi0237_cohort.json"
    cohort = read_json(cohort_path)
    if view == "16":
        rows = legacy.select_fixed16(cohort["train"])
    elif view == "64":
        rows = sorted(
            [dict(row) for row in cohort["train"]],
            key=lambda row: (str(row["target_family"]), int(row["phase"]), str(row["source_episode_id"])),
        )
    else:
        episodes_path = Path(paths["train_dataset"]) / "meta" / "episodes.jsonl"
        rows = []
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("split", "")).lower() != "train":
                raise PI0243Error("FULL view contains a non-TRAIN episode")
            source_id = str(item["source_episode_id"])
            match = re.search(r"persistent_train_([a-d])_", source_id)
            rows.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "source_episode_id": source_id,
                    "source_rollout_id": source_id.split(":obs-", 1)[0],
                    "target_family": match.group(1) if match else "unknown",
                    "phase": None,
                    "jitter_family": "unknown",
                    "split": "TRAIN",
                }
            )
        rows.sort(key=lambda row: int(row["episode_index"]))
    if len(rows) != ({"16": 16, "64": 64, "full": 9792}[view]):
        raise PI0243Error(f"{view} view count mismatch: {len(rows)}")
    if len({str(row["source_episode_id"]) for row in rows}) != len(rows):
        raise PI0243Error(f"{view} view has duplicate source IDs")
    return rows, sha256_file(cohort_path)


def exposure_schedule(rows: list[Mapping[str, Any]], max_steps: int) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    order: list[int] = []
    while len(order) < max_steps:
        order.extend(int(value) for value in rng.permutation(len(rows)))
    schedule = []
    for step, fixed_index in enumerate(order[:max_steps], start=1):
        row = rows[fixed_index]
        schedule.append(
            {
                "optimizer_step": step,
                "fixed_index": fixed_index,
                "episode_index": int(row["episode_index"]),
                "source_episode_id": str(row["source_episode_id"]),
                "target_family": row.get("target_family"),
                "phase": row.get("phase"),
            }
        )
    if len(schedule) != max_steps:
        raise PI0243Error("schedule length mismatch")
    return schedule


def resolved_config(card: Mapping[str, Any], output_dir: Path, legacy: Any, pi0228: Any) -> tuple[Any, dict[str, Any]]:
    paths = card["paths"]
    helper = argparse.Namespace(
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["initializer"]),
        dataset_dir=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
        groot_source=Path(paths["groot_source"]),
    )
    spec = card["stage"]
    config = pi0228.build_config(
        helper,
        output_dir,
        steps=int(spec["max_steps"]),
        save_steps=SAVE_PROBE,
        save_limit=len(spec["checkpoints"]) + 2,
    )
    embodiment = config.data.datasets[0].embodiment_tag
    config.data.modality_configs = {embodiment: config.data.modality_configs[embodiment]}
    config.training.max_grad_norm = 1.0
    config.training.dataloader_num_workers = 0
    config.training.gradient_checkpointing = False
    config.training.num_gpus = 1
    config.training.eval_strategy = "no"
    config.training.logging_steps = 1
    config.data.num_shards_per_epoch = 1000
    summary = pi0228.config_summary(config)
    summary.update(
        {
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "num_gpus": 1,
            "eval_strategy": "no",
            "load_bf16": bool(config.model.load_bf16),
            "backbone_trainable_params_fp32": bool(config.model.backbone_trainable_params_fp32),
            "state_dropout_prob": float(config.model.state_dropout_prob),
            "shortest_image_edge": int(config.model.shortest_image_edge),
            "crop_fraction": float(config.model.crop_fraction),
            "modality_config_keys": [str(embodiment)],
            "checkpoint_steps": list(spec["checkpoints"]),
            "screen_steps": list(spec["screens"]),
        }
    )
    return config, summary


def make_card(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise PI0243Error(f"refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    legacy = legacy_module(args.code_root)
    pi0228 = pi0228_module(args.code_root)
    spec = stage_spec(args.stage)
    paths = {
        "code_root": str(args.code_root.resolve()),
        "groot_source": str(args.groot_source.resolve()),
        "stage10_root": str(args.stage10_root.resolve()),
        "train_dataset": str(args.train_dataset.resolve()),
        "p237_root": str(args.p237_root.resolve()),
        "modality_config": str(args.modality_config.resolve()),
        "processor_base_model": str(args.processor_base_model.resolve()),
        "initializer": str(args.initializer.resolve()),
        "cosmos_snapshot": str(args.cosmos_snapshot.resolve()),
        "cosmos_cache_root": str(args.cosmos_cache_root.resolve()),
        "output_root": str(args.output_root.resolve()),
    }
    rows, cohort_sha = source_rows(paths, spec["view"], legacy)
    schedule = exposure_schedule(rows, int(spec["max_steps"]))
    cohort_file = args.output_root / "stage_cohort.json"
    schedule_file = args.output_root / "stage_exposure_schedule.json"
    write_json(cohort_file, {"schema": "PI0243_STAGE_COHORT_V1", "view": spec["view"], "rows": rows})
    write_json(schedule_file, {"schema": "PI0243_STAGE_EXPOSURE_V1", "seed": SEED, "rows": schedule})
    fixed16_cohort = read_json(Path(paths["p237_root"]) / "evidence" / "pi0237_cohort.json")
    fixed16_rows = legacy.select_fixed16(fixed16_cohort["train"])
    card_seed = {
        "schema": "PI0243_RUN_CARD_V1",
        "status": "SEALED_READY_TO_EXECUTE",
        "protocol_commit": PROTOCOL_COMMIT,
        "audited_parent": AUDITED_PARENT,
        "stage": spec,
        "paths": paths,
        "cohort": {"path": str(cohort_file), "sha256": sha256_file(cohort_file), "count": len(rows), "source_cohort_sha256": cohort_sha},
        "schedule": {"path": str(schedule_file), "sha256": sha256_file(schedule_file), "count": len(schedule)},
        "implementation": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "legacy_path": str((args.code_root / "scripts" / "pi0241_fixed16_learning.py").resolve()),
            "legacy_sha256": sha256_file(args.code_root / "scripts" / "pi0241_fixed16_learning.py"),
            "pi0228_path": str((args.code_root / "scripts" / "pi0228_h200_large_adapt_v1.py").resolve()),
            "pi0228_sha256": sha256_file(args.code_root / "scripts" / "pi0228_h200_large_adapt_v1.py"),
        },
        "initializer": {"path": str(args.initializer.resolve()), "files": legacy._model_files(args.initializer)},
        "fixed16_source_ids": [str(row["source_episode_id"]) for row in fixed16_rows],
        "processor": {"modality_sha256": sha256_file(args.modality_config), "stats_root": str((args.train_dataset / "meta").resolve()), "identity": "C_T"},
        "target_xyz_in_model_input": False,
        "selected_checkpoint": "NONE",
        "forbidden_counts": {"calib_queries": 0, "wbc_steps": 0, "sim_steps": 0, "dev_e_reads": 0, "final_f_reads": 0, "dagger_steps": 0},
    }
    card = dict(card_seed)
    _, config_summary = resolved_config(card, args.output_root / "training", legacy, pi0228)
    card["resolved_config"] = config_summary
    card["sim_binding"] = {"status": "SEALED_FOR_POST_SELECTION", "source": "PI0223 protocol; existing native dense-relative WBC entry only", "target_xyz_in_model_input": False}
    card_path = args.output_root / "pi0243_run_card.json"
    write_json(card_path, card)
    write_json(args.output_root / "pi0243_run_card.seal.json", {"schema": "PI0243_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(card_path)})
    print(json.dumps({"status": card["status"], "run_card": str(card_path), "sha256": sha256_file(card_path)}, indent=2))


def verify_card(path: Path) -> dict[str, Any]:
    card = read_json(path)
    seal = read_json(path.with_name("pi0243_run_card.seal.json"))
    if card.get("schema") != "PI0243_RUN_CARD_V1" or card.get("status") != "SEALED_READY_TO_EXECUTE":
        raise PI0243Error("run card is not sealed")
    if seal.get("run_card_sha256") != sha256_file(path):
        raise PI0243Error("run card SHA mismatch")
    implementation = card["implementation"]
    for path_key, hash_key in (
        ("script_path", "script_sha256"),
        ("legacy_path", "legacy_sha256"),
        ("pi0228_path", "pi0228_sha256"),
    ):
        if sha256_file(Path(implementation[path_key])) != implementation[hash_key]:
            raise PI0243Error(f"implementation changed after card seal: {path_key}")
    cohort_path = _mapped_stage_path(card, card["cohort"]["path"], "PI0243_METADATA_MIRROR_ROOT")
    schedule_path = _mapped_stage_path(card, card["schedule"]["path"], "PI0243_METADATA_MIRROR_ROOT")
    if sha256_file(cohort_path) != card["cohort"]["sha256"] or sha256_file(schedule_path) != card["schedule"]["sha256"]:
        raise PI0243Error("cohort or schedule changed after card seal")
    return card


def configure_legacy(card: Mapping[str, Any], legacy: Any) -> None:
    legacy.MAX_STEPS = int(card["stage"]["max_steps"])
    legacy.CHECKPOINT_STEPS = tuple(int(value) for value in card["stage"]["checkpoints"])
    legacy.SCREEN_STEPS = tuple(int(value) for value in card["stage"]["screens"])


def train_stage(args: argparse.Namespace) -> None:
    card = runtime_card(verify_card(args.run_card))
    stage_key = str(args.stage).lower()
    if card["stage"]["stage"].lower() != stage_key or args.authorization != AUTH[stage_key]:
        raise PI0243Error("stage or authorization mismatch")
    legacy = legacy_module(Path(card["paths"]["code_root"]))
    configure_legacy(card, legacy)
    legacy.bind_runtime(card)
    output = Path(card["paths"]["output_root"]) / "training"
    if output.exists():
        raise PI0243Error(f"refusing to reuse training output: {output}")
    output.mkdir(parents=True)
    legacy.install_fixed_dataset_factory(card, output)
    legacy.install_update_audit(output)
    pi0228 = pi0228_module(Path(card["paths"]["code_root"]))
    config, summary = resolved_config(card, output, legacy, pi0228)
    if summary != card["resolved_config"]:
        raise PI0243Error("resolved config drift")
    write_json(output / "resolved_config.json", summary)
    if card.get("runtime_path_map"):
        write_json(Path(card["paths"]["output_root"]) / "pi0243_runtime_path_map.json", card["runtime_path_map"])
    started = time.time()
    pi0228.run_official(config, Path(card["paths"]["groot_source"]))
    consumption_path = output / "pi0241_sample_consumption.jsonl"
    contract_path = output / "pi0241_sample_contracts.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    schedule = read_json(Path(card["schedule"]["path"]))["rows"]
    if len(consumption) != len(schedule) or [row["source_episode_id"] for row in consumption] != [row["source_episode_id"] for row in schedule]:
        raise PI0243Error("actual sample consumption differs from sealed schedule")
    contracts = [json.loads(line) for line in contract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(contracts) != int(card["cohort"]["count"]) or any(row.get("active_slots") != list(range(10)) for row in contracts):
        raise PI0243Error("action/mask contract audit incomplete")
    checkpoints = {}
    for step in card["stage"]["checkpoints"]:
        checkpoints[str(step)] = pi0228.checkpoint_record(output / f"checkpoint-{step}", int(step), int(card["stage"]["max_steps"]))
    audit = read_json(output / "pi0241_update_audit.json")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != int(card["stage"]["max_steps"]):
        raise PI0243Error("update audit incomplete")
    final = audit.get("final_parameters", {})
    update_integrity = {
        "projector_updated": float(final.get("projector", {}).get("delta_l2", 0.0)) > 0.0,
        "diffusion_updated": float(final.get("diffusion", {}).get("delta_l2", 0.0)) > 0.0,
        "frozen_backbone_unchanged": float(final.get("frozen_backbone", {}).get("delta_l2", 1.0)) == 0.0,
        "frozen_vlln_unchanged": float(final.get("frozen_vlln", {}).get("delta_l2", 1.0)) == 0.0,
        "optimizer_membership_valid": not audit.get("optimizer_membership", {}).get("missing_trainable") and not audit.get("optimizer_membership", {}).get("illegal_frozen"),
    }
    manifest = {
        "schema": "PI0243_STAGE_TRAINING_MANIFEST_V1",
        "status": "PASS_UPDATE_INTEGRITY" if all(update_integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "stage": card["stage"]["stage"],
        "elapsed_s": time.time() - started,
        "canonical_optimizer_steps": int(card["stage"]["max_steps"]),
        "actual_sample_consumption": len(consumption),
        "schedule_sha256": card["schedule"]["sha256"],
        "config": summary,
        "update_integrity": update_integrity,
        "checkpoints": checkpoints,
        "sample_contract_count": len(contracts),
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json(Path(card["paths"]["output_root"]) / "pi0243_training_manifest.json", manifest)
    if not all(update_integrity.values()):
        raise PI0243Error("scientific optimizer/update integrity failure")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _image_hwc(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise PI0243Error(f"unexpected RGB shape: {arr.shape}")
    return np.asarray(arr, dtype=np.uint8)


def _dataset_and_examples(card: Mapping[str, Any], legacy: Any, pi0228: Any, processor: Any, embodiment: Any) -> list[dict[str, Any]]:
    import gr00t.data.embodiment_tags as embodiment_tags
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    cfg, _ = resolved_config(card, Path(card["paths"]["output_root"]) / "screen_cfg", legacy, pi0228)
    spec = cfg.data.datasets[0]
    dataset = ShardedSingleStepDataset(
        dataset_path=Path(card["paths"]["train_dataset"]),
        embodiment_tag=embodiment_tags.EmbodimentTag(spec.embodiment_tag),
        modality_configs=cfg.data.modality_configs[spec.embodiment_tag],
        shard_size=cfg.data.shard_size,
        episode_sampling_rate=cfg.data.episode_sampling_rate,
        seed=cfg.data.seed,
        allow_padding=cfg.data.allow_padding,
    )
    rows = read_json(Path(card["cohort"]["path"]))["rows"]
    examples = []
    for row in rows:
        dp = dataset.get_datapoint(dataset.episode_loader[int(row["episode_index"])], 0)
        state_full = _to_numpy(dp["state"]).reshape(-1)
        state = state_full[:10].astype(np.float32)
        images = dp["vlm_content"]["images"]
        body, wrist = _image_hwc(images[0]), _image_hwc(images[1])
        normalized = _to_numpy(dp["action"]).astype(np.float32)
        if normalized.shape != (40, 132):
            raise PI0243Error(f"expert action shape drift for {row['source_episode_id']}: {normalized.shape}")
        mask = _to_numpy(dp["action_mask"]).astype(np.float64)
        if mask.shape != (40, 132):
            raise PI0243Error(f"expert mask shape drift for {row['source_episode_id']}: {mask.shape}")
        state_arg = {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]}
        decoded_groups = processor.decode_action(normalized[None], embodiment, state_arg)
        expert = np.concatenate([np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])], axis=-1).astype(np.float64)[0]
        examples.append({"row": row, "body_rgb": body, "wrist_rgb": wrist, "state": state, "expert": expert, "mask": mask})
    return examples


def _target_map(card: Mapping[str, Any]) -> dict[str, np.ndarray]:
    authority = read_json(Path(card["paths"]["p237_root"]) / "evidence" / "pi0228_train_calib_target_authority.json")
    return {str(row["source_rollout_id"]): np.asarray(row["target_position_world_m"], dtype=np.float64) for row in authority["rollouts"]}


def _screen_metrics(model: np.ndarray, expert: np.ndarray, legacy: Any) -> dict[str, float]:
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix
    xyz_sq = np.sum((model[:, :3] - expert[:, :3]) ** 2, axis=1)
    theta = np.asarray(geodesic_angle(rot6d_rows_to_matrix(model[:, 3:9]), rot6d_rows_to_matrix(expert[:, 3:9])), dtype=np.float64)
    return {
        "xyz_rmse_m": float(np.sqrt(np.mean(xyz_sq))),
        "so3_rmse_rad": float(np.sqrt(np.mean(theta**2))),
        "gripper_mae": float(np.abs(model[:, 9] - expert[:, 9]).mean()),
        "first_xyz_error_m": float(np.sqrt(xyz_sq[0])),
        "endpoint_xyz_error_m": float(np.sqrt(xyz_sq[-1])),
        "max_linear_speed_mps": float(np.linalg.norm(np.diff(model[:, :3], axis=0), axis=1).max() / DT),
        "max_angular_speed_radps": float(np.max(geodesic_angle(rot6d_rows_to_matrix(model[:-1, 3:9]), rot6d_rows_to_matrix(model[1:, 3:9]))) / DT),
    }


def screen_stage(args: argparse.Namespace) -> None:
    card = runtime_card(verify_card(args.run_card))
    legacy = legacy_module(Path(card["paths"]["code_root"]))
    configure_legacy(card, legacy)
    legacy.bind_runtime(card)
    code_root = Path(card["paths"]["code_root"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(card["paths"]["groot_source"])]
    import torch
    import stage21_same_scene_groot_inference as stage21
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from transformers import AutoModel

    processor, embodiment = stage21._build_processor(
        Path(card["paths"]["stage10_root"]), Path(card["paths"]["processor_base_model"]), Path(card["paths"]["train_dataset"]) / "meta", Path(card["paths"]["modality_config"])
    )
    examples = _dataset_and_examples(card, legacy, pi0228_module(code_root), processor, embodiment)
    targets = _target_map(card)
    gate = TrajectorySafetyGateV1()
    root = Path(card["paths"]["output_root"]) / "screen"
    all_records: list[dict[str, Any]] = []
    for step in card["stage"]["screens"]:
        step_root = root / f"step-{int(step):05d}"
        step_root.mkdir(parents=True, exist_ok=True)
        records_path = step_root / "records.jsonl"
        existing = {json.loads(line)["source_episode_id"]: json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()} if records_path.is_file() else {}
        model_path = Path(card["paths"]["initializer"]) if int(step) == 0 else Path(card["paths"]["output_root"]) / "training" / f"checkpoint-{int(step)}"
        if not model_path.is_dir():
            raise PI0243Error(f"screen model missing: {model_path}")
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model.eval().to(device=args.device, dtype=torch.bfloat16)
        for example in examples:
            row = example["row"]
            source_id = str(row["source_episode_id"])
            if source_id in existing:
                continue
            observation = {"body_rgb": example["body_rgb"], "wrist_rgb": example["wrist_rgb"], "state": example["state"]}
            raw, decoded, latency, gpu = stage21._run_one(model, processor, embodiment, observation, language=LANGUAGE, seed=SEED, device=args.device)
            decoded10 = decoded[0].astype(np.float64)
            state_quat = matrix_to_quat(rot6d_rows_to_matrix(example["state"][3:9].reshape(1, 6)))[0]
            gate_result = gate.evaluate(selector._build_chunk(decoded, chunk_id=f"pi0243-{card['stage']['stage']}-{step}-{row['episode_index']}"), now=0.0, achieved_position=example["state"][:3], achieved_orientation=state_quat)
            metrics = _screen_metrics(decoded10, example["expert"], legacy)
            target = targets.get(str(row.get("source_rollout_id")) or source_id.split(":obs-", 1)[0])
            tensor_path = step_root / "cells" / f"train-{int(row['episode_index']):06d}.npz"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tensor_path, raw=raw, decoded=decoded, expert=example["expert"], state=example["state"], action_mask=example["mask"], body_rgb=example["body_rgb"], wrist_rgb=example["wrist_rgb"])
            record = {
                "schema": "PI0243_SCREEN_CELL_V1", "stage": card["stage"]["stage"], "screen_step": int(step),
                "source_episode_id": source_id, "episode_index": int(row["episode_index"]), "target_family": row.get("target_family"), "phase": row.get("phase"),
                **metrics, "target_min_distance_m": None if target is None else float(np.linalg.norm(decoded10[:, :3] - target[None, :], axis=1).min()),
                "gate_decision": gate_result.decision.value, "gate": gate_result.as_dict(), "raw_sha256": sha256_array(raw), "decoded_sha256": sha256_array(decoded),
                "tensor_path": str(tensor_path), "tensor_sha256": sha256_file(tensor_path), "latency_s": float(latency), "gpu": gpu, "seed": SEED, "language": LANGUAGE, "target_xyz_in_model_input": False,
            }
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
            existing[source_id] = record
        del model; gc.collect(); torch.cuda.empty_cache()
        rows = [existing[str(example["row"]["source_episode_id"])] for example in examples]
        if len(rows) != len(examples):
            raise PI0243Error(f"screen step {step} incomplete")
        summary = {
            "schema": "PI0243_SCREEN_STEP_SUMMARY_V1", "stage": card["stage"]["stage"], "step": int(step), "count": len(rows),
            "gate_counts": dict(Counter(row["gate_decision"] for row in rows)),
            "linear_speed_reject_count": sum(any(issue.get("code") == "LINEAR_SPEED_REJECT" for issue in row["gate"].get("issues", [])) for row in rows),
            "median_xyz_rmse_m": float(np.percentile([row["xyz_rmse_m"] for row in rows], 50, method="linear")),
            "p95_xyz_rmse_m": float(np.percentile([row["xyz_rmse_m"] for row in rows], 95, method="linear")),
            "median_so3_rmse_rad": float(np.percentile([row["so3_rmse_rad"] for row in rows], 50, method="linear")),
        }
        write_json(step_root / "summary.json", summary)
        all_records.extend(rows)
        if card["stage"]["stage"] == "C" and int(step) == 2500:
            s1000 = read_json(root / "step-01000" / "summary.json")
            if s1000["linear_speed_reject_count"] == len(rows) and summary["linear_speed_reject_count"] == len(rows):
                write_json(Path(card["paths"]["output_root"]) / "pi0243_early_stop.json", {"status": "SCIENCE_FAIL_FULL_TRAIN_COLLAPSE", "steps": [1000, 2500]})
                raise PI0243Error("SCIENCE_FAIL_FULL_TRAIN_COLLAPSE")
    final_step = int(card["stage"]["max_steps"])
    final_rows = [row for row in all_records if int(row["screen_step"]) == final_step]
    xyz = np.asarray([row["xyz_rmse_m"] for row in final_rows], dtype=np.float64)
    so3 = np.asarray([row["so3_rmse_rad"] for row in final_rows], dtype=np.float64)
    gate_rejects = sum(row["gate_decision"] == "REJECT" for row in final_rows)
    criteria = {
        "gate_rejects_zero": gate_rejects == 0,
        "median_xyz_rmse_le_0p020": float(np.percentile(xyz, 50, method="linear")) <= 0.020,
        "p95_xyz_rmse_le_0p035": float(np.percentile(xyz, 95, method="linear")) <= 0.035,
        "median_so3_rmse_le_0p050": float(np.percentile(so3, 50, method="linear")) <= 0.050,
    }
    if card["stage"]["stage"] == "A":
        passed = all(criteria.values())
        status = "PASS_STAGE_A" if passed else "SCIENCE_FAIL_M16_BUDGET"
    elif card["stage"]["stage"] == "B":
        # The stage card records the exact fixed16 IDs for the memory check.
        fixed16 = set(card.get("fixed16_source_ids", []))
        subset = [row for row in final_rows if row["source_episode_id"] in fixed16]
        subset_ok = len(subset) == 16 and all(row["gate_decision"] != "REJECT" for row in subset)
        subset_ok = subset_ok and float(np.percentile([row["xyz_rmse_m"] for row in subset], 50, method="linear")) <= 0.020 and float(np.percentile([row["xyz_rmse_m"] for row in subset], 95, method="linear")) <= 0.035 and float(np.percentile([row["so3_rmse_rad"] for row in subset], 50, method="linear")) <= 0.050
        criteria["old16_subset_pass"] = subset_ok
        passed = all(criteria.values())
        status = "PASS_STAGE_B" if passed else "SCIENCE_FAIL_SUPPORT_EXPANSION"
    else:
        status = "PASS_STAGE_C_SCREEN" if final_rows else "INVALID_INCOMPLETE"
    closure = {
        "schema": "PI0243_STAGE_CLOSURE_V1", "stage": card["stage"]["stage"], "status": status,
        "terminal_token": None if status.startswith("PASS_STAGE") else status + " — STOP_FOR_PI_SINGLE_RECIPE_DECISION",
        "criteria": criteria, "final_count": len(final_rows), "screen_cells": len(all_records),
        "selected_checkpoint": "NONE", "forbidden_counts": card["forbidden_counts"],
    }
    write_json(Path(card["paths"]["output_root"]) / "pi0243_stage_closure.json", closure)
    print(json.dumps(closure, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("prepare", "train", "screen"), required=True)
    p.add_argument("--stage", choices=("a", "b", "c"), required=True)
    p.add_argument("--run-card", type=Path)
    p.add_argument("--authorization")
    p.add_argument("--code-root", type=Path)
    p.add_argument("--groot-source", type=Path)
    p.add_argument("--stage10-root", type=Path)
    p.add_argument("--train-dataset", type=Path)
    p.add_argument("--p237-root", type=Path)
    p.add_argument("--modality-config", type=Path)
    p.add_argument("--processor-base-model", type=Path)
    p.add_argument("--initializer", type=Path)
    p.add_argument("--cosmos-snapshot", type=Path)
    p.add_argument("--cosmos-cache-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--device", default="cuda:0")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.phase == "prepare":
        required = (args.code_root, args.groot_source, args.stage10_root, args.train_dataset, args.p237_root, args.modality_config, args.processor_base_model, args.initializer, args.cosmos_snapshot, args.cosmos_cache_root, args.output_root)
        if any(value is None for value in required):
            raise PI0243Error("prepare requires all path arguments")
        make_card(args)
    elif args.phase == "train":
        if args.run_card is None or args.authorization is None:
            raise PI0243Error("train requires run card and authorization")
        train_stage(args)
    else:
        if args.run_card is None:
            raise PI0243Error("screen requires run card")
        screen_stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
