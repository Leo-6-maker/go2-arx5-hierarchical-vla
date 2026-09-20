#!/usr/bin/env python3
"""PI0241 fixed-16 dense-relative learning feasibility run.

This is a thin adapter around the qualified PI0228/Stage28 GR00T training
path.  It changes only the dataset view/order, the authorized 1000-step
schedule, checkpoint retention, and read-only update/screen evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROTOCOL_COMMIT = "e4b73fa9acdca05c15d117ff40c16bb582b63987"
AUDITED_PARENT = "f536aa359e265fe8e225faa29090fee5922e5baa"
EXPERIMENT = "PI0241_FIXED16_V1"
AUTHORIZATION = "PI0241_FIXED16_1000_GO"
SEED = 22301
LANGUAGE = "move the end effector to the red target"
MAX_STEPS = 1000
CHECKPOINT_STEPS = (250, 500, 1000)
SCREEN_STEPS = (0, 250, 500, 1000)
TARGETS = tuple("abcd")
PHASES = tuple(range(4))
DT = 0.05


class PI0241Error(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PI0241Error(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def select_fixed16(train_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the preregistered outcome-independent stratum/hash rule."""

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in train_rows:
        if row.get("split") != "TRAIN":
            raise PI0241Error("fixed16 source contains a non-TRAIN row")
        key = (str(row["target_family"]), int(row["phase"]))
        groups.setdefault(key, []).append(row)
    expected = {(target, phase) for target in TARGETS for phase in PHASES}
    if set(groups) != expected:
        raise PI0241Error(f"fixed16 strata mismatch: {sorted(groups)}")

    chosen = []
    for target in TARGETS:
        for phase in PHASES:
            candidates = groups[(target, phase)]
            ranked = sorted(
                candidates,
                key=lambda row: (
                    hashlib.sha256(
                        f"{EXPERIMENT}|{row['source_episode_id']}".encode("utf-8")
                    ).hexdigest(),
                    str(row["source_episode_id"]),
                ),
            )
            row = dict(ranked[0])
            row["selection_sha256"] = hashlib.sha256(
                f"{EXPERIMENT}|{row['source_episode_id']}".encode("utf-8")
            ).hexdigest()
            chosen.append(row)
    if len(chosen) != 16 or len({row["source_episode_id"] for row in chosen}) != 16:
        raise PI0241Error("fixed16 selection is not sixteen unique source episodes")
    return chosen


def build_exposure_schedule(selected: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if [(row["target_family"], int(row["phase"])) for row in selected] != [
        (target, phase) for target in TARGETS for phase in PHASES
    ]:
        raise PI0241Error("fixed16 rows are not in canonical target/phase order")
    rng = np.random.Generator(np.random.PCG64(SEED))
    order: list[int] = []
    while len(order) < MAX_STEPS:
        order.extend(int(value) for value in rng.permutation(16))
    order = order[:MAX_STEPS]
    schedule = [
        {
            "optimizer_step": step,
            "fixed_index": fixed_index,
            "source_episode_id": selected[fixed_index]["source_episode_id"],
            "episode_index": int(selected[fixed_index]["episode_index"]),
            "target_family": selected[fixed_index]["target_family"],
            "phase": int(selected[fixed_index]["phase"]),
        }
        for step, fixed_index in enumerate(order, start=1)
    ]
    counts = Counter(row["fixed_index"] for row in schedule)
    if set(counts.values()) - {62, 63} or sum(counts.values()) != MAX_STEPS:
        raise PI0241Error(f"invalid fixed exposure counts: {dict(counts)}")
    return schedule


def classify_final(rows: list[Mapping[str, Any]], integrity_pass: bool) -> dict[str, Any]:
    if not integrity_pass:
        return {"status": "INVALID_INCOMPLETE", "terminal_token": None}
    if len(rows) != 16 or len({str(row.get("source_episode_id")) for row in rows}) != 16:
        return {"status": "INVALID_INCOMPLETE", "terminal_token": None}
    xyz = np.asarray([row.get("xyz_rmse_m") for row in rows], dtype=np.float64)
    so3 = np.asarray([row.get("so3_rmse_rad") for row in rows], dtype=np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(so3).all():
        return {"status": "INVALID_INCOMPLETE", "terminal_token": None}
    gate_rejects = sum(row.get("gate_decision") == "REJECT" for row in rows)
    criteria = {
        "gate_rejects_zero": gate_rejects == 0,
        "median_xyz_rmse_le_0p020": float(np.percentile(xyz, 50, method="linear")) <= 0.020,
        "p95_xyz_rmse_le_0p035": float(np.percentile(xyz, 95, method="linear")) <= 0.035,
        "median_so3_rmse_le_0p050": float(np.percentile(so3, 50, method="linear")) <= 0.050,
    }
    passed = all(criteria.values())
    return {
        "status": "PASS_FIXED16_LEARNING_FEASIBILITY" if passed else "SCIENCE_FAIL_FIXED16_LEARNING",
        "terminal_token": (
            "PASS_FIXED16_LEARNING_FEASIBILITY — STOP_FOR_PI_SCALE_DECISION"
            if passed
            else "SCIENCE_FAIL_FIXED16_LEARNING — STOP_FOR_PI_RECIPE_DECISION"
        ),
        "criteria": criteria,
        "gate_reject_count": gate_rejects,
        "median_xyz_rmse_m": float(np.percentile(xyz, 50, method="linear")),
        "p95_xyz_rmse_m": float(np.percentile(xyz, 95, method="linear")),
        "median_so3_rmse_rad": float(np.percentile(so3, 50, method="linear")),
    }


def _model_files(model_path: Path) -> dict[str, dict[str, Any]]:
    names = ("config.json", "model.safetensors.index.json")
    paths = [model_path / name for name in names]
    paths.extend(sorted(model_path.glob("model-*.safetensors")))
    if not all(path.is_file() for path in paths[:2]) or len(paths) < 4:
        raise PI0241Error(f"initializer model is incomplete: {model_path}")
    return {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths}


def _resolved_config(args: argparse.Namespace, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    sys.path[:0] = [str(args.code_root / "scripts"), str(args.groot_source)]
    pi0228 = load_module(args.code_root / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0241_pi0228")
    helper = argparse.Namespace(
        stage10_root=args.stage10_root,
        base_model_path=args.base_model_path,
        dataset_dir=args.train_dataset,
        modality_config=args.modality_config,
        groot_source=args.groot_source,
    )
    config = pi0228.build_config(helper, output_dir, steps=MAX_STEPS, save_steps=250, save_limit=3)
    config.training.max_grad_norm = 1.0
    config.training.lr_scheduler_type = "cosine"
    config.training.dataloader_num_workers = 0
    config.training.gradient_checkpointing = False
    config.training.num_gpus = 1
    config.training.eval_strategy = "no"
    config.training.logging_steps = 1
    config.data.num_shards_per_epoch = 1000
    summary = pi0228.config_summary(config)
    summary.update(
        {
            "max_grad_norm": float(config.training.max_grad_norm),
            "gradient_checkpointing": bool(config.training.gradient_checkpointing),
            "dataloader_num_workers": int(config.training.dataloader_num_workers),
            "num_gpus": int(config.training.num_gpus),
            "eval_strategy": str(config.training.eval_strategy),
            "load_bf16": bool(config.model.load_bf16),
            "backbone_trainable_params_fp32": bool(config.model.backbone_trainable_params_fp32),
            "state_dropout_prob": float(config.model.state_dropout_prob),
            "random_rotation_angle": (
                None
                if config.model.random_rotation_angle is None
                else float(config.model.random_rotation_angle)
            ),
            "shortest_image_edge": int(config.model.shortest_image_edge),
            "crop_fraction": float(config.model.crop_fraction),
            "save_steps_effective": list(CHECKPOINT_STEPS),
            "save_step_750_suppressed_by_audit_callback": True,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
        }
    )
    expected = {
        "max_steps": 1000,
        "save_steps": 250,
        "save_total_limit": 3,
        "warmup_steps": 10,
        "warmup_ratio": 0.0,
        "scheduler": "cosine",
        "optimizer": "adamw_torch",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "save_only_model": False,
        "resume_from_checkpoint": False,
        "use_relative_action": True,
        "action_horizon": 40,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_visual": False,
        "tune_llm": False,
        "tune_vlln": False,
        "seed": SEED,
    }
    drift = {key: {"expected": value, "observed": summary.get(key)} for key, value in expected.items() if summary.get(key) != value}
    if drift:
        raise PI0241Error(f"resolved config drift: {drift}")
    return config, summary


def prepare(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=False)
    cohort_path = args.p237_root / "evidence" / "pi0237_cohort.json"
    cohort = read_json(cohort_path)
    selected = select_fixed16(cohort["train"])
    # Seal selection before opening any row-level G4 fit records.
    cohort_seal = {
        "schema": "PI0241_FIXED16_COHORT_V1",
        "experiment": EXPERIMENT,
        "selection_rule": 'SHA256(UTF8("PI0241_FIXED16_V1|" + source_episode_id))',
        "source_cohort_sha256": sha256_file(cohort_path),
        "rows": selected,
    }
    cohort_file = args.output_root / "fixed16_cohort.json"
    write_json(cohort_file, cohort_seal)

    schedule = build_exposure_schedule(selected)
    schedule_file = args.output_root / "fixed1000_schedule.json"
    write_json(schedule_file, {"schema": "PI0241_FIXED1000_SCHEDULE_V1", "rows": schedule})

    baseline_records_path = args.g4_root / "baseline1000" / "records.jsonl"
    records = [json.loads(line) for line in baseline_records_path.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["source_episode_id"]): row for row in records if row.get("split") == "TRAIN"}
    diagnostics = []
    selected_inputs = []
    for row in selected:
        source_id = str(row["source_episode_id"])
        record = by_id.get(source_id)
        if record is None or int(record["episode_index"]) != int(row["episode_index"]):
            raise PI0241Error(f"PI0240 record identity mismatch: {source_id}")
        if record["expert_gate"]["decision"] == "REJECT":
            raise PI0241Error(f"selected expert row is REJECT: {source_id}")
        tensor_path = Path(record["tensor"]["path"])
        if not tensor_path.is_file():
            raise PI0241Error(f"missing PI0240 tensor: {tensor_path}")
        with np.load(tensor_path, allow_pickle=False) as payload:
            decoded = np.asarray(payload["decoded"])[0].astype(np.float64)
            expert = np.asarray(payload["expert"]).astype(np.float64)
            mask = np.asarray(payload["action_mask"])
            if expert.shape != (40, 10) or mask.shape != (1, 40, 132):
                raise PI0241Error(f"selected tensor contract mismatch: {source_id}")
            if not np.isfinite(expert).all() or not np.isfinite(mask).all():
                raise PI0241Error(f"selected tensor nonfinite: {source_id}")
            bias = np.mean(decoded[:, :3] - expert[:, :3], axis=0)
            residual = decoded[:, :3] - expert[:, :3] - bias[None, :]
            model_speed = np.linalg.norm(np.diff(decoded[:, :3], axis=0), axis=1) / DT
            expert_speed = np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1) / DT
            diagnostics.append(
                {
                    "source_episode_id": source_id,
                    "position_bias_m": bias.tolist(),
                    "debiased_xyz_rmse_m": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                    "first_xyz_error_m": float(np.linalg.norm(decoded[0, :3] - expert[0, :3])),
                    "endpoint_xyz_error_m": float(np.linalg.norm(decoded[-1, :3] - expert[-1, :3])),
                    "model_max_linear_speed_mps": float(model_speed.max()),
                    "expert_max_linear_speed_mps": float(expert_speed.max()),
                    "model_peak_segment": int(np.argmax(model_speed)),
                    "expert_peak_segment": int(np.argmax(expert_speed)),
                }
            )
        selected_inputs.append(
            {
                "source_episode_id": source_id,
                "episode_index": int(row["episode_index"]),
                "tensor_path": str(tensor_path),
                "tensor_sha256": sha256_file(tensor_path),
                "expert_sha256": record["tensor"]["expert_sha256"],
                "mask_sha256": record["tensor"]["mask_sha256"],
                "expert_gate_decision": record["expert_gate"]["decision"],
            }
        )
    write_json(args.output_root / "step0_reused_diagnostics.json", diagnostics)

    _, config_summary = _resolved_config(args, args.output_root / "training")
    stats_files = [args.train_dataset / "meta" / name for name in ("info.json", "stats.json", "relative_stats.json")]
    for path in stats_files:
        if not path.is_file():
            raise PI0241Error(f"missing TRAIN metadata: {path}")
    card = {
        "schema": "PI0241_RUN_CARD_V1",
        "status": "SEALED_READY_TO_TRAIN",
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "audited_parent": AUDITED_PARENT,
        "host": args.host,
        "implementation": {
            "code_head": git_head(args.code_root),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "paths": {
            "code_root": str(args.code_root),
            "groot_source": str(args.groot_source),
            "stage10_root": str(args.stage10_root),
            "train_dataset": str(args.train_dataset),
            "p237_root": str(args.p237_root),
            "g4_root": str(args.g4_root),
            "base_model_path": str(args.base_model_path),
            "modality_config": str(args.modality_config),
            "cosmos_snapshot": str(args.cosmos_snapshot),
            "cosmos_cache_root": str(args.cosmos_cache_root),
            "output_root": str(args.output_root),
        },
        "cohort": {"path": str(cohort_file), "sha256": sha256_file(cohort_file), "count": 16},
        "schedule": {
            "path": str(schedule_file),
            "sha256": sha256_file(schedule_file),
            "count": 1000,
            "counts": dict(sorted(Counter(row["source_episode_id"] for row in schedule).items())),
        },
        "selected_inputs": selected_inputs,
        "initializer": {"role": "PI0226_ADAPTED_BASELINE1000", "files": _model_files(args.base_model_path)},
        "train_metadata": {path.name: sha256_file(path) for path in stats_files},
        "modality_sha256": sha256_file(args.modality_config),
        "groot_source_head": git_head(args.groot_source),
        "resolved_config": config_summary,
        "screen": {"steps": list(SCREEN_STEPS), "step0_reused": True, "unique_cells": 64},
        "target_xyz_in_model_input": False,
        "selected_checkpoint": "NONE",
        "forbidden_counts": {"calib_queries": 0, "wbc_steps": 0, "sim_steps": 0, "dev_e_reads": 0, "final_f_reads": 0},
    }
    card_path = args.output_root / "pi0241_run_card.json"
    write_json(card_path, card)
    write_json(
        args.output_root / "pi0241_run_card.seal.json",
        {"schema": "PI0241_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(card_path)},
    )
    print(json.dumps({"status": card["status"], "run_card": str(card_path), "sha256": sha256_file(card_path)}, indent=2))


def _verify_card(path: Path) -> dict[str, Any]:
    card = read_json(path)
    seal = read_json(path.with_name("pi0241_run_card.seal.json"))
    if card.get("schema") != "PI0241_RUN_CARD_V1" or card.get("status") != "SEALED_READY_TO_TRAIN":
        raise PI0241Error("run card is not sealed for training")
    if seal.get("run_card_sha256") != sha256_file(path):
        raise PI0241Error("run card SHA mismatch")
    for key in ("cohort", "schedule"):
        item = card[key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise PI0241Error(f"run card {key} changed")
    if sha256_file(Path(__file__).resolve()) != card["implementation"]["script_sha256"]:
        raise PI0241Error("PI0241 implementation changed after run-card seal")
    return card


def functional_preflight(args: argparse.Namespace) -> None:
    """Exercise the real training processor/fixed dataset/collator without a model."""

    card = _verify_card(args.run_card)
    bind_runtime(card)
    paths = card["paths"]
    output = Path(paths["output_root"]) / "functional_preflight"
    if output.exists():
        raise PI0241Error(f"refusing to reuse functional preflight output: {output}")
    cfg_dir = output / "experiment_cfg"
    cfg_dir.mkdir(parents=True)
    install_fixed_dataset_factory(card, output)
    config_args = argparse.Namespace(
        code_root=Path(paths["code_root"]),
        groot_source=Path(paths["groot_source"]),
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        train_dataset=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
    )
    config, summary = _resolved_config(config_args, output / "training")
    if summary != card["resolved_config"]:
        raise PI0241Error("functional preflight config differs from sealed config")
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
    train_dataset, eval_dataset = pipeline._create_dataset(cfg_dir)
    if eval_dataset is not None:
        raise PI0241Error("functional preflight unexpectedly built an eval dataset")
    datapoint = next(iter(train_dataset))
    batch = pipeline.processor.collator([datapoint])
    action = mask = None

    def visit(value: Any, name: str = "") -> None:
        nonlocal action, mask
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{name}.{key}" if name else str(key))
        elif hasattr(value, "shape") and tuple(int(dim) for dim in value.shape) == (1, 40, 132):
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if "mask" in name.lower():
                mask = array
            elif action is None:
                action = array

    visit(batch)
    if action is None or mask is None or not np.isfinite(action).all() or not np.isfinite(mask).all():
        raise PI0241Error("functional preflight did not produce finite [1,40,132] action/mask")
    active_slots = np.flatnonzero(np.any(mask != 0, axis=(0, 1))).tolist()
    if active_slots != list(range(10)):
        raise PI0241Error(f"functional preflight active slots drift: {active_slots}")
    consumption = [json.loads(line) for line in (output / "pi0241_sample_consumption.jsonl").read_text().splitlines()]
    planned = read_json(Path(card["schedule"]["path"]))["rows"][0]
    if len(consumption) != 1 or consumption[0]["source_episode_id"] != planned["source_episode_id"]:
        raise PI0241Error("functional preflight did not consume fixed schedule row 1")
    report = {
        "schema": "PI0241_FUNCTIONAL_PREFLIGHT_V1",
        "status": "PASS_NO_MODEL",
        "action_shape": list(action.shape),
        "mask_shape": list(mask.shape),
        "active_slots": active_slots,
        "first_source_episode_id": consumption[0]["source_episode_id"],
        "model_loads": 0,
        "model_forwards": 0,
        "backward": 0,
        "optimizer_steps": 0,
    }
    write_json(output / "functional_preflight.json", report)
    print(json.dumps(report, indent=2))


def bind_runtime(card: Mapping[str, Any]) -> dict[str, Any]:
    paths = card["paths"]
    cache_root = Path(paths["cosmos_cache_root"])
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(cache_root / "hub" if (cache_root / "hub").exists() else cache_root),
            "TRANSFORMERS_CACHE": str(cache_root / "hub" if (cache_root / "hub").exists() else cache_root),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    code_root = Path(paths["code_root"])
    stage10_root = Path(paths["stage10_root"])
    groot_source = Path(paths["groot_source"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(groot_source)]
    stage40 = load_module(code_root / "scripts" / "stage40_b0_model_control.py", "pi0241_stage40")
    cosmos = stage40._bind_local_cosmos_processor(stage10_root, Path(paths["cosmos_snapshot"]))
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    return {"offline": True, "cosmos": cosmos, "torchcodec": True}


def install_fixed_dataset_factory(card: Mapping[str, Any], training_output: Path) -> None:
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7 import setup as gr00t_setup

    selected = read_json(Path(card["cohort"]["path"]))["rows"]
    schedule = read_json(Path(card["schedule"]["path"]))["rows"]
    consumption_path = training_output / "pi0241_sample_consumption.jsonl"

    class FixedScheduleMixture(ShardedMixtureDataset):
        def __iter__(self):
            if getattr(self, "_pi0241_started", False):
                raise PI0241Error("fixed1000 schedule was requested more than once")
            self._pi0241_started = True
            dataset = self.datasets[0]
            for ordinal, planned in enumerate(schedule, start=1):
                if int(planned["optimizer_step"]) != ordinal:
                    raise PI0241Error("fixed1000 schedule ordinal drift")
                fixed = selected[int(planned["fixed_index"])]
                episode_index = int(fixed["episode_index"])
                datapoint = dataset.get_datapoint(dataset.episode_loader[episode_index], 0)
                append_jsonl(
                    consumption_path,
                    {
                        "yield_ordinal": ordinal,
                        "optimizer_step": int(planned["optimizer_step"]),
                        "fixed_index": int(planned["fixed_index"]),
                        "episode_index": episode_index,
                        "source_episode_id": fixed["source_episode_id"],
                        "observation_index": 0,
                    },
                )
                yield datapoint

        def reset_seed(self, seed: int):
            if int(seed) != SEED:
                raise PI0241Error("PI0241 does not authorize shifted-seed resume")
            self.seed = int(seed)

    class FixedDatasetFactory:
        def __init__(self, config: Any):
            self.config = config

        def build(self, processor: Any):
            specs = self.config.data.datasets
            if len(specs) != 1 or len(specs[0].dataset_paths) != 1:
                raise PI0241Error("PI0241 requires one full TRAIN source dataset")
            dataset_path = Path(specs[0].dataset_paths[0])
            if dataset_path.resolve() != Path(card["paths"]["train_dataset"]).resolve():
                raise PI0241Error("training dataset path drift")
            dataset = ShardedSingleStepDataset(
                dataset_path=dataset_path,
                embodiment_tag=EmbodimentTag(specs[0].embodiment_tag),
                modality_configs=self.config.data.modality_configs[specs[0].embodiment_tag],
                shard_size=self.config.data.shard_size,
                episode_sampling_rate=self.config.data.episode_sampling_rate,
                seed=self.config.data.seed,
                allow_padding=self.config.data.allow_padding,
            )
            mixture = FixedScheduleMixture(
                datasets=[dataset],
                weights=[1.0],
                processor=processor,
                seed=SEED,
                training=True,
                num_shards_per_epoch=1000,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )
            return mixture, None

    gr00t_setup.DatasetFactory = FixedDatasetFactory


def _parameter_group(name: str) -> str:
    if any(token in name for token in ("action_head.state_encoder", "action_head.action_encoder", "action_head.action_decoder", "action_head.position_embedding")):
        return "projector"
    if "action_head.model" in name:
        return "diffusion"
    if "action_head.vlln" in name or "action_head.vl_self_attention" in name:
        return "frozen_vlln"
    if "backbone" in name:
        return "frozen_backbone"
    return "other"


def install_update_audit(training_output: Path) -> None:
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment import experiment as experiment_module

    base_trainer = experiment_module.Gr00tTrainer
    audit_path = training_output / "pi0241_update_audit.json"
    audit_steps = set(range(1, 13)) | {32, 250, 500, 1000}

    class UpdateAudit(TrainerCallback):
        def __init__(self, model: Any):
            self.model = model
            self.named = dict(model.named_parameters())
            self.representatives: dict[str, str] = {}
            for group in ("projector", "diffusion", "frozen_backbone", "frozen_vlln"):
                for name, parameter in self.named.items():
                    if _parameter_group(name) == group and parameter.numel():
                        self.representatives[group] = name
                        break
            self.initial: dict[str, torch.Tensor] = {}
            self.report: dict[str, Any] = {
                "schema": "PI0241_UPDATE_AUDIT_V1",
                "status": "RUNNING",
                "audit_steps": sorted(audit_steps),
                "events": [],
            }

        @staticmethod
        def sample(parameter: Any):
            return parameter.detach().reshape(-1)[:4096].float().cpu().clone()

        def snapshot(self, name: str) -> dict[str, Any]:
            current = self.sample(self.named[name])
            initial = self.initial[name]
            delta = current - initial
            return {
                "name": name,
                "sample_count": int(current.numel()),
                "initial_l2": float(torch.linalg.vector_norm(initial).item()),
                "current_l2": float(torch.linalg.vector_norm(current).item()),
                "delta_l2": float(torch.linalg.vector_norm(delta).item()),
                "delta_max_abs": float(delta.abs().max().item()),
            }

        def flush(self):
            write_json(audit_path, self.report)

        def on_train_begin(self, args, state, control, **kwargs):
            optimizer = kwargs["optimizer"]
            optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            trainable = {name: parameter for name, parameter in self.named.items() if parameter.requires_grad}
            frozen = {name: parameter for name, parameter in self.named.items() if not parameter.requires_grad}
            missing = sorted(name for name, parameter in trainable.items() if id(parameter) not in optimizer_ids)
            illegal = sorted(name for name, parameter in frozen.items() if id(parameter) in optimizer_ids)
            for group, name in self.representatives.items():
                self.initial[name] = self.sample(self.named[name])
            self.report.update(
                {
                    "optimizer": type(optimizer).__name__,
                    "optimizer_membership": {
                        "trainable_parameter_tensors": len(trainable),
                        "frozen_parameter_tensors": len(frozen),
                        "missing_trainable": missing,
                        "illegal_frozen": illegal,
                        "groups": dict(Counter(_parameter_group(name) for name in trainable)),
                        "dtypes": dict(Counter(str(parameter.dtype) for parameter in trainable.values())),
                    },
                    "representatives": self.representatives,
                }
            )
            if missing or illegal or not {"projector", "diffusion"}.issubset(self.representatives):
                raise PI0241Error("optimizer/trainable membership is invalid")
            self.flush()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            step = int(state.global_step) + 1
            if step not in audit_steps:
                return
            optimizer = kwargs["optimizer"]
            sums = Counter()
            finite = True
            tensors = Counter()
            for name, parameter in self.named.items():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                finite = finite and bool(torch.isfinite(grad).all().item())
                norm = float(torch.linalg.vector_norm(grad).float().item())
                group = _parameter_group(name)
                sums[group] += norm * norm
                tensors[group] += 1
            self.report["events"].append(
                {
                    "event": "PRE_OPTIMIZER",
                    "step": step,
                    "lr": [float(group["lr"]) for group in optimizer.param_groups],
                    "gradient_finite": finite,
                    "gradient_group_l2": {key: math.sqrt(value) for key, value in sums.items()},
                    "gradient_tensor_counts": dict(tensors),
                }
            )
            if not finite:
                self.flush()
                raise PI0241Error(f"non-finite gradient at step {step}")

        def on_optimizer_step(self, args, state, control, **kwargs):
            step = int(state.global_step) + 1
            if step in audit_steps:
                self.report["events"].append(
                    {
                        "event": "POST_OPTIMIZER",
                        "step": step,
                        "parameters": {
                            group: self.snapshot(name)
                            for group, name in self.representatives.items()
                        },
                    }
                )
                self.flush()

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step == 750:
                control.should_save = False
            elif step in CHECKPOINT_STEPS:
                control.should_save = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.report["status"] = "COMPLETE"
            self.report["global_step"] = int(state.global_step)
            self.report["final_parameters"] = {
                group: self.snapshot(name) for group, name in self.representatives.items()
            }
            self.flush()

    class AuditedTrainer(base_trainer):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.add_callback(UpdateAudit(self.model))

    experiment_module.Gr00tTrainer = AuditedTrainer


def checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "model.safetensors.index.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise PI0241Error(f"checkpoint-{expected_step} missing full state: {missing}")
    state = read_json(path / "trainer_state.json")
    if int(state.get("global_step", -1)) != expected_step or int(state.get("max_steps", -1)) != MAX_STEPS:
        raise PI0241Error(f"checkpoint-{expected_step} trainer state drift")
    entries = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and not child.is_symlink():
            entries.append({"path": child.relative_to(path).as_posix(), "bytes": child.stat().st_size, "sha256": sha256_file(child)})
    return {
        "path": str(path),
        "global_step": expected_step,
        "max_steps": MAX_STEPS,
        "file_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "tree_sha256": sha256_json(entries),
        "entries": entries,
    }


def train(args: argparse.Namespace) -> None:
    if args.authorization != AUTHORIZATION:
        raise PI0241Error("training authorization token mismatch")
    card = _verify_card(args.run_card)
    bind_runtime(card)
    paths = card["paths"]
    training_output = Path(paths["output_root"]) / "training"
    if training_output.exists():
        raise PI0241Error(f"refusing to reuse training output: {training_output}")
    training_output.mkdir(parents=True)
    install_fixed_dataset_factory(card, training_output)
    install_update_audit(training_output)
    config_args = argparse.Namespace(
        code_root=Path(paths["code_root"]),
        groot_source=Path(paths["groot_source"]),
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        train_dataset=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
    )
    config, summary = _resolved_config(config_args, training_output)
    if summary != card["resolved_config"]:
        raise PI0241Error("resolved config changed after run-card seal")
    write_json(training_output / "resolved_config.json", summary)
    started = time.time()
    pi0228 = load_module(Path(paths["code_root"]) / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0241_pi0228_train")
    pi0228.run_official(config, Path(paths["groot_source"]))
    consumption = [json.loads(line) for line in (training_output / "pi0241_sample_consumption.jsonl").read_text().splitlines()]
    planned = read_json(Path(card["schedule"]["path"]))["rows"]
    if len(consumption) != 1000 or [row["source_episode_id"] for row in consumption] != [row["source_episode_id"] for row in planned]:
        raise PI0241Error("actual sample consumption differs from fixed1000 schedule")
    actual_dirs = sorted(path.name for path in training_output.glob("checkpoint-*") if path.is_dir())
    expected_dirs = [f"checkpoint-{step}" for step in CHECKPOINT_STEPS]
    if actual_dirs != expected_dirs:
        raise PI0241Error(f"checkpoint ladder drift: {actual_dirs}")
    checkpoints = {str(step): checkpoint_record(training_output / f"checkpoint-{step}", step) for step in CHECKPOINT_STEPS}
    audit = read_json(training_output / "pi0241_update_audit.json")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != 1000:
        raise PI0241Error("update audit incomplete")
    final = audit["final_parameters"]
    update_integrity = {
        "projector_updated": float(final["projector"]["delta_l2"]) > 0.0,
        "diffusion_updated": float(final["diffusion"]["delta_l2"]) > 0.0,
        "frozen_backbone_unchanged": float(final["frozen_backbone"]["delta_l2"]) == 0.0,
        "frozen_vlln_unchanged": float(final["frozen_vlln"]["delta_l2"]) == 0.0,
        "optimizer_membership_valid": not audit["optimizer_membership"]["missing_trainable"] and not audit["optimizer_membership"]["illegal_frozen"],
    }
    manifest = {
        "schema": "PI0241_TRAINING_MANIFEST_V1",
        "status": "PASS_UPDATE_INTEGRITY" if all(update_integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "elapsed_s": time.time() - started,
        "canonical_optimizer_steps": 1000,
        "actual_sample_consumption": 1000,
        "schedule_sha256": card["schedule"]["sha256"],
        "config": summary,
        "update_integrity": update_integrity,
        "checkpoints": checkpoints,
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json(Path(paths["output_root"]) / "pi0241_training_manifest.json", manifest)
    if not all(update_integrity.values()):
        raise PI0241Error("scientific optimizer/update integrity failure")
    print(json.dumps({"status": manifest["status"], "elapsed_s": manifest["elapsed_s"], "checkpoints": list(checkpoints)}, indent=2))


def _rmse_metrics(model: np.ndarray, expert: np.ndarray, modules: Mapping[str, Any]) -> dict[str, float]:
    xyz_sq = np.sum((model[:, :3] - expert[:, :3]) ** 2, axis=1)
    theta = np.asarray(
        modules["geodesic_angle"](
            modules["rot6d_rows_to_matrix"](model[:, 3:9]),
            modules["rot6d_rows_to_matrix"](expert[:, 3:9]),
        ),
        dtype=np.float64,
    )
    gripper = np.abs(model[:, 9] - expert[:, 9])
    return {
        "xyz_rmse_m": float(np.sqrt(np.mean(xyz_sq))),
        "so3_rmse_rad": float(np.sqrt(np.mean(theta**2))),
        "gripper_mae": float(gripper.mean()),
        "gripper_max": float(gripper.max()),
        "first_xyz_error_m": float(np.sqrt(xyz_sq[0])),
        "endpoint_xyz_error_m": float(np.sqrt(xyz_sq[-1])),
        "first_so3_error_rad": float(theta[0]),
        "endpoint_so3_error_rad": float(theta[-1]),
        "max_linear_speed_mps": float(np.linalg.norm(np.diff(model[:, :3], axis=0), axis=1).max() / DT),
        "max_angular_speed_radps": float(
            np.max(
                modules["geodesic_angle"](
                    modules["rot6d_rows_to_matrix"](model[:-1, 3:9]),
                    modules["rot6d_rows_to_matrix"](model[1:, 3:9]),
                )
            )
            / DT
        ),
    }


def screen_and_close(args: argparse.Namespace) -> None:
    card = _verify_card(args.run_card)
    runtime = bind_runtime(card)
    paths = card["paths"]
    code_root = Path(paths["code_root"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(paths["groot_source"])]
    import torch
    import stage20s_infer_groot as stage20
    import stage21_same_scene_groot_inference as stage21
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, rot6d_rows_to_matrix
    from transformers import AutoModel

    training_manifest = read_json(Path(paths["output_root"]) / "pi0241_training_manifest.json")
    integrity_pass = training_manifest.get("status") == "PASS_UPDATE_INTEGRITY"
    selected = read_json(Path(card["cohort"]["path"]))["rows"]
    selected_inputs = {row["source_episode_id"]: row for row in card["selected_inputs"]}
    processor, embodiment = stage21._build_processor(
        Path(paths["stage10_root"]), Path(paths["base_model_path"]), Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"])
    )
    gate = TrajectorySafetyGateV1()
    modules = {"geodesic_angle": geodesic_angle, "rot6d_rows_to_matrix": rot6d_rows_to_matrix}
    screen_root = Path(paths["output_root"]) / "screen"
    all_records = []
    for step in SCREEN_STEPS:
        step_root = screen_root / f"step-{step:04d}"
        step_root.mkdir(parents=True, exist_ok=True)
        records_path = step_root / "records.jsonl"
        existing = {row["source_episode_id"]: row for row in ([json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else [])}
        model = None
        if step:
            model_path = Path(paths["output_root"]) / "training" / f"checkpoint-{step}"
            model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            model.eval().to(device=args.device, dtype=torch.bfloat16)
        for fixed in selected:
            source_id = str(fixed["source_episode_id"])
            if source_id in existing:
                continue
            source_npz = Path(selected_inputs[source_id]["tensor_path"])
            with np.load(source_npz, allow_pickle=False) as payload:
                state = np.asarray(payload["state"])
                expert = np.asarray(payload["expert"]).astype(np.float64)
                mask = np.asarray(payload["action_mask"])
                body = np.asarray(payload["body_rgb"])
                wrist = np.asarray(payload["wrist_rgb"])
                if step == 0:
                    raw = np.asarray(payload["raw"])
                    decoded = np.asarray(payload["decoded"])
                    latency = 0.0
                    gpu = {"reused_pi0240_step0": True}
                else:
                    raw, decoded, latency, gpu = stage21._run_one(
                        model,
                        processor,
                        embodiment,
                        {"body_rgb": body, "wrist_rgb": wrist, "state": state},
                        language=LANGUAGE,
                        seed=SEED,
                        device=args.device,
                    )
            decoded10 = decoded[0].astype(np.float64)
            state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
            gate_result = gate.evaluate(
                selector._build_chunk(decoded, chunk_id=f"pi0241-step{step}-{fixed['episode_index']}"),
                now=0.0,
                achieved_position=state[:3],
                achieved_orientation=state_quat,
            )
            metrics = _rmse_metrics(decoded10, expert, modules)
            tensor_path = step_root / "cells" / f"train-{int(fixed['episode_index']):06d}.npz"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tensor_path, raw=raw, decoded=decoded, expert=expert, state=state, action_mask=mask, body_rgb=body, wrist_rgb=wrist)
            record = {
                "schema": "PI0241_SCREEN_CELL_V1",
                "screen_step": step,
                "source_episode_id": source_id,
                "episode_index": int(fixed["episode_index"]),
                "target_family": fixed["target_family"],
                "phase": int(fixed["phase"]),
                **metrics,
                "gate_decision": gate_result.decision.value,
                "gate": gate_result.as_dict(),
                "raw_sha256": sha256_array(raw),
                "decoded_sha256": sha256_array(decoded),
                "tensor_path": str(tensor_path),
                "tensor_sha256": sha256_file(tensor_path),
                "latency_s": float(latency),
                "gpu": gpu,
                "seed": SEED,
                "language": LANGUAGE,
                "target_xyz_in_model_input": False,
            }
            append_jsonl(records_path, record)
            existing[source_id] = record
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        rows = [existing[str(row["source_episode_id"])] for row in selected]
        if len(rows) != 16:
            raise PI0241Error(f"screen step {step} incomplete")
        summary = {
            "schema": "PI0241_SCREEN_STEP_SUMMARY_V1",
            "step": step,
            "count": 16,
            "gate_counts": dict(Counter(row["gate_decision"] for row in rows)),
            "median_xyz_rmse_m": float(np.percentile([row["xyz_rmse_m"] for row in rows], 50, method="linear")),
            "p95_xyz_rmse_m": float(np.percentile([row["xyz_rmse_m"] for row in rows], 95, method="linear")),
            "median_so3_rmse_rad": float(np.percentile([row["so3_rmse_rad"] for row in rows], 50, method="linear")),
            "worst_xyz": sorted(rows, key=lambda row: row["xyz_rmse_m"], reverse=True)[:3],
        }
        write_json(step_root / "summary.json", summary)
        all_records.extend(rows)
    if len(all_records) != 64 or len({(row["screen_step"], row["source_episode_id"]) for row in all_records}) != 64:
        raise PI0241Error("screen ledger is not 64 unique cells")
    final_rows = [row for row in all_records if row["screen_step"] == 1000]
    decision = classify_final(final_rows, integrity_pass)
    if decision["status"] == "INVALID_INCOMPLETE":
        raise PI0241Error("screen or update integrity is incomplete")
    closure = {
        "schema": "PI0241_CLOSURE_V1",
        **decision,
        "training_manifest_sha256": sha256_file(Path(paths["output_root"]) / "pi0241_training_manifest.json"),
        "run_card_sha256": sha256_file(args.run_card),
        "screen_cells": 64,
        "new_get_action_calls": 48,
        "reused_step0_cells": 16,
        "runtime": runtime,
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
        "screen_step_summaries": {
            str(step): read_json(screen_root / f"step-{step:04d}" / "summary.json") for step in SCREEN_STEPS
        },
    }
    write_json(Path(paths["output_root"]) / "pi0241_closure.json", closure)
    print(json.dumps(closure, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--functional-preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--screen-and-close", action="store_true")
    result.add_argument("--run-card", type=Path)
    result.add_argument("--authorization")
    result.add_argument("--host", default="unknown")
    result.add_argument("--code-root", type=Path)
    result.add_argument("--groot-source", type=Path)
    result.add_argument("--stage10-root", type=Path)
    result.add_argument("--train-dataset", type=Path)
    result.add_argument("--p237-root", type=Path)
    result.add_argument("--g4-root", type=Path)
    result.add_argument("--base-model-path", type=Path)
    result.add_argument("--modality-config", type=Path)
    result.add_argument("--cosmos-snapshot", type=Path)
    result.add_argument("--cosmos-cache-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--device", default="cuda:0")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.prepare:
        required = (args.code_root, args.groot_source, args.stage10_root, args.train_dataset, args.p237_root, args.g4_root, args.base_model_path, args.modality_config, args.cosmos_snapshot, args.cosmos_cache_root, args.output_root)
        if any(value is None for value in required):
            raise PI0241Error("prepare requires all identity/path arguments")
        prepare(args)
    elif args.functional_preflight:
        if args.run_card is None:
            raise PI0241Error("functional preflight requires --run-card")
        functional_preflight(args)
    elif args.train:
        if args.run_card is None:
            raise PI0241Error("train requires --run-card")
        train(args)
    else:
        if args.run_card is None:
            raise PI0241Error("screen-and-close requires --run-card")
        screen_and_close(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
