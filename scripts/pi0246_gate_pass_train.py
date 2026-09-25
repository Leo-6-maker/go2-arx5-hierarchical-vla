#!/usr/bin/env python3
"""PI0246 C/T/P/PT arms, reusing the sealed PI0241 fixed-dataset and update audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


STEPS = 2000
CHECKPOINTS = (1000, 2000)
SEED = 22301
EXPECTED_P1_QUERIES = 32
EXPECTED_P0_SHA256 = "6351150ac04566e8238fbfa996869d5ba039256eb5b12d1a0b9bf8a7345ba702"
EXPECTED_NOISE_SHA256 = "a98bb142616c7ab558ccd788e996ae10e7b85a2a729770bdc0f65d3da8f99df9"
CHANGEABLE_CONFIG = {"start_from_checkpoint", "max_steps", "save_steps", "save_total_limit", "learning_rate"}
ARMS = {
    "C": ("PI0245_C_NATIVE_M16_2K_V1", "L_native"),
    "T": ("PI0245_T_PHYSDELTA_M16_2K_V1", "L_native + 0.1 * L_phys_delta"),
}
P_ARM = ("PI0246_P_PHYSPOS_M16_2K_V1", "L_native + 0.1 * L_phys_pos")
PT_ARM = ("PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1", "L_native + 0.1 * L_phys_pos + 0.1 * L_phys_delta")
ARM_DEFS = {**ARMS, "P": P_ARM, "PT": PT_ARM}


class CampaignError(RuntimeError):
    pass


def needs_projector_reconciliation(arm: str, integrity: dict[str, bool]) -> bool:
    return (
        arm in {"T", "P", "PT"}
        and not integrity.get("projector_updated", False)
        and all(value for key, value in integrity.items() if key != "projector_updated")
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_execution_profile(profile: str, pbs_job_id: str | None, device_count: int, device_name: str | None) -> dict[str, Any]:
    if profile == "H200_PBS":
        if not pbs_job_id or device_count != 1 or not device_name or "H200" not in device_name:
            raise CampaignError(f"H200_PBS requires one PBS H200; job={pbs_job_id!r}, devices={device_count}, name={device_name!r}")
        launcher = "PBS"
    elif profile == "CORELAB_4080S_DIRECT":
        if pbs_job_id or device_count != 1 or device_name != "NVIDIA GeForce RTX 4080 SUPER":
            raise CampaignError(f"CORELAB_4080S_DIRECT requires one direct 4080 SUPER; job={pbs_job_id!r}, devices={device_count}, name={device_name!r}")
        launcher = "direct"
    else:
        raise CampaignError(f"unsupported execution profile: {profile}")
    return {"profile": profile, "launcher": launcher, "device_count": device_count,
            "device_name": device_name, "pbs_job_id": pbs_job_id}


def t_runtime_identity(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    root = runtime_root.resolve()
    parent = Path(card["paths"]["campaign_code_root"])
    code_root = Path(card["paths"]["code_root"])
    parent_delta = Path(card["paths"]["p1_root"]).parent / "code" / "pi0245_physical_delta.py"
    runtime_delta = root / "pi0245_physical_delta.py"
    if sha256_file(runtime_delta) != sha256_file(parent_delta):
        raise CampaignError("T runtime physical-delta helper differs from the sealed PI0245 helper")
    return {
        "schema": "PI0246_T_SAME_FORWARD_LOGGING_RUNTIME_V1",
        "run_card_sha256": sha256_file(card_path),
        "family": card["family"],
        "loss": card["loss"],
        "seed": card["seed"],
        "execution_profile": card.get("execution_profile", "H200_PBS"),
        "resolved_config_sha256": canonical_sha256(card["resolved_config"]),
        "runtime_root": str(root),
        "trainer_sha256": sha256_file(root / "pi0246_gate_pass_train.py"),
        "overlay_sha256": sha256_file(root / "pi0245_native_overlay.py"),
        "physical_delta_sha256": sha256_file(runtime_delta),
        "parent_physical_delta_sha256": sha256_file(parent_delta),
        "reconciliation_script_sha256": sha256_file(root / "pi0246_reconcile_c_update.py"),
        "wrapper_path": str(wrapper.resolve()),
        "wrapper_sha256": sha256_file(wrapper),
        "parent_campaign_root": str(parent.resolve()),
        "parent_campaign_trainer_sha256": sha256_file(parent / "pi0246_gate_pass_train.py"),
        "parent_native_overlay_sha256": sha256_file(parent / "pi0245_native_overlay.py"),
        "campaign_p1_runner_sha256": sha256_file(parent / "pi0245_p1_stock.py"),
        "training_helpers": {
            name: sha256_file(code_root / "scripts" / name)
            for name in ("pi0241_fixed16_learning.py", "pi0228_h200_large_adapt_v1.py")
        },
        "sealer_sha256": sha256_file(Path(__file__).resolve()),
    }


def seal_t_runtime(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    seal = read_json(card_path.with_name("run_card.seal.json"))
    if seal.get("sha256") != sha256_file(card_path) or card.get("arm") != "T" or card.get("loss") != ARMS["T"][1]:
        raise CampaignError("T runtime requires the unchanged sealed T card")
    if (Path(card["paths"]["output_root"]) / "training").exists():
        raise CampaignError("T training output already exists; refusing to reseal or overwrite")
    identity = t_runtime_identity(card_path, runtime_root, wrapper)
    manifest = runtime_root / "T_runtime_binding.json"
    receipt = manifest.with_name("T_runtime_binding.seal.json")
    if manifest.exists() or receipt.exists():
        raise CampaignError("T runtime binding already exists")
    write_json(manifest, identity)
    write_json(receipt, {"sha256": sha256_file(manifest)})
    return {"status": "PASS_T_RUNTIME_SEALED", "binding": identity, "binding_sha256": sha256_file(manifest)}


def verify_t_runtime(card_path: Path, runtime_root: Path, wrapper: Path, binding_path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = t_runtime_identity(card_path, runtime_root, wrapper)
    seal_path = binding_path.with_name("T_runtime_binding.seal.json")
    binding = read_json(binding_path)
    seal = read_json(seal_path)
    if sha256_file(binding_path) != expected_sha256 or seal.get("sha256") != expected_sha256 or binding != identity:
        raise CampaignError("T runtime binding, card, wrapper, or staged source SHA mismatch")
    return {"status": "PASS_T_RUNTIME_BINDING", "sha256": expected_sha256,
            "trainer_sha256": identity["trainer_sha256"], "overlay_sha256": identity["overlay_sha256"],
            "wrapper_sha256": identity["wrapper_sha256"]}


def p_runtime_identity(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    root = runtime_root.resolve()
    paths = card["paths"]
    parent_delta = Path(paths["p1_root"]).parent / "code" / "pi0245_physical_delta.py"
    runtime_delta = root / "pi0245_physical_delta.py"
    if sha256_file(runtime_delta) != sha256_file(parent_delta):
        raise CampaignError("P runtime physical decoder differs from the sealed PI0245 helper")
    return {
        "schema": "PI0246_P_PHYSPOS_SAME_FORWARD_RUNTIME_V1",
        "run_card_sha256": sha256_file(card_path),
        "family": card["family"], "loss": card["loss"], "seed": card["seed"],
        "execution_profile": card.get("execution_profile", "H200_PBS"),
        "resolved_config_sha256": canonical_sha256(card["resolved_config"]),
        "runtime_root": str(root),
        "trainer_sha256": sha256_file(root / "pi0246_gate_pass_train.py"),
        "position_overlay_sha256": sha256_file(root / "pi0246_physical_position.py"),
        "physical_decoder_sha256": sha256_file(runtime_delta),
        "parent_physical_decoder_sha256": sha256_file(parent_delta),
        "wrapper_path": str(wrapper.resolve()), "wrapper_sha256": sha256_file(wrapper),
        "p1_runner_sha256": sha256_file(Path(paths["campaign_code_root"]) / "pi0245_p1_stock.py"),
        "training_helpers": {
            name: sha256_file(Path(paths["code_root"]) / "scripts" / name)
            for name in ("pi0241_fixed16_learning.py", "pi0228_h200_large_adapt_v1.py")
        },
        "sealer_sha256": sha256_file(Path(__file__).resolve()),
    }


def seal_p_runtime(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    seal = read_json(card_path.with_name("run_card.seal.json"))
    if seal.get("sha256") != sha256_file(card_path) or card.get("arm") != "P" or card.get("loss") != P_ARM[1]:
        raise CampaignError("P runtime requires the sealed P card")
    if (Path(card["paths"]["output_root"]) / "training").exists():
        raise CampaignError("P training output already exists; refusing to reseal or overwrite")
    identity = p_runtime_identity(card_path, runtime_root, wrapper)
    manifest = runtime_root / "P_runtime_binding.json"
    receipt = manifest.with_name("P_runtime_binding.seal.json")
    if manifest.exists() or receipt.exists():
        raise CampaignError("P runtime binding already exists")
    write_json(manifest, identity)
    write_json(receipt, {"sha256": sha256_file(manifest)})
    return {"status": "PASS_P_RUNTIME_SEALED", "binding": identity, "binding_sha256": sha256_file(manifest)}


def verify_p_runtime(card_path: Path, runtime_root: Path, wrapper: Path, binding_path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = p_runtime_identity(card_path, runtime_root, wrapper)
    binding = read_json(binding_path)
    seal = read_json(binding_path.with_name("P_runtime_binding.seal.json"))
    if sha256_file(binding_path) != expected_sha256 or seal.get("sha256") != expected_sha256 or binding != identity:
        raise CampaignError("P runtime binding, card, wrapper, or staged source SHA mismatch")
    return {"status": "PASS_P_RUNTIME_BINDING", "sha256": expected_sha256,
            "trainer_sha256": identity["trainer_sha256"],
            "position_overlay_sha256": identity["position_overlay_sha256"],
            "wrapper_sha256": identity["wrapper_sha256"]}


def pt_runtime_identity(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    root = runtime_root.resolve()
    paths = card["paths"]
    parent_runtime = card["parent_loss_runtime"]
    parent_binding_path = Path(parent_runtime["binding_path"])
    parent_binding = read_json(parent_binding_path)
    parent_seal = read_json(parent_binding_path.with_name("P_runtime_binding.seal.json"))
    validate_parent_p_runtime(
        Path(parent_runtime["source_run_card_path"]), parent_binding_path,
        str(parent_runtime["binding_sha256"]),
    )
    parent_root = Path(parent_binding["runtime_root"])
    parent_delta = parent_root / "pi0245_physical_delta.py"
    parent_position = parent_root / "pi0246_physical_position.py"
    runtime_delta = root / "pi0245_physical_delta.py"
    runtime_position = root / "pi0246_physical_position.py"
    if (sha256_file(parent_binding_path) != parent_runtime["binding_sha256"]
            or parent_seal.get("sha256") != parent_runtime["binding_sha256"]
            or parent_binding.get("run_card_sha256") != card["predecessor_failure"]["P_run_card_sha256"]
            or sha256_file(parent_delta) != parent_binding.get("physical_decoder_sha256")
            or sha256_file(parent_position) != parent_binding.get("position_overlay_sha256")
            or sha256_file(runtime_delta) != sha256_file(parent_delta)
            or sha256_file(runtime_position) != sha256_file(parent_position)):
        raise CampaignError("PT helpers differ from the sealed P physical-loss/decoder implementation")
    return {
        "schema": "PI0246_PT_SINGLE_FORWARD_RUNTIME_V1",
        "run_card_sha256": sha256_file(card_path), "family": card["family"], "loss": card["loss"],
        "seed": card["seed"], "execution_profile": card.get("execution_profile", "H200_PBS"),
        "resolved_config_sha256": canonical_sha256(card["resolved_config"]),
        "runtime_root": str(root),
        "trainer_sha256": sha256_file(root / "pi0246_gate_pass_train.py"),
        "combined_overlay_sha256": sha256_file(root / "pi0246_pt_combined_overlay.py"),
        "position_helper_sha256": sha256_file(runtime_position),
        "physical_delta_sha256": sha256_file(runtime_delta),
        "parent_loss_runtime_binding_sha256": sha256_file(parent_binding_path),
        "parent_position_sha256": sha256_file(parent_position),
        "parent_physical_delta_sha256": sha256_file(parent_delta),
        "wrapper_path": str(wrapper.resolve()), "wrapper_sha256": sha256_file(wrapper),
        "source_p_card_sha256": card["predecessor_failure"]["P_run_card_sha256"],
        "p2000_verification_sha256": card["predecessor_failure"]["P2000_cpu_verification_sha256"],
        "training_helpers": {
            name: sha256_file(Path(paths["code_root"]) / "scripts" / name)
            for name in ("pi0241_fixed16_learning.py", "pi0228_h200_large_adapt_v1.py")
        },
        "sealer_sha256": sha256_file(Path(__file__).resolve()),
    }


def seal_pt_runtime(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    seal = read_json(card_path.with_name("run_card.seal.json"))
    if (seal.get("sha256") != sha256_file(card_path) or card.get("arm") != "PT"
            or card.get("loss") != PT_ARM[1]):
        raise CampaignError("PT runtime requires the sealed PT card")
    if (Path(card["paths"]["output_root"]) / "training").exists():
        raise CampaignError("PT training output already exists; refusing to reseal or overwrite")
    identity = pt_runtime_identity(card_path, runtime_root, wrapper)
    manifest = runtime_root / "PT_runtime_binding.json"
    receipt = manifest.with_name("PT_runtime_binding.seal.json")
    if manifest.exists() or receipt.exists():
        raise CampaignError("PT runtime binding already exists")
    write_json(manifest, identity)
    write_json(receipt, {"sha256": sha256_file(manifest)})
    return {"status": "PASS_PT_RUNTIME_SEALED", "binding": identity, "binding_sha256": sha256_file(manifest)}


def verify_pt_runtime(card_path: Path, runtime_root: Path, wrapper: Path, binding_path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = pt_runtime_identity(card_path, runtime_root, wrapper)
    binding = read_json(binding_path)
    seal = read_json(binding_path.with_name("PT_runtime_binding.seal.json"))
    if sha256_file(binding_path) != expected_sha256 or seal.get("sha256") != expected_sha256 or binding != identity:
        raise CampaignError("PT runtime binding, card, wrapper, or staged source SHA mismatch")
    return {"status": "PASS_PT_RUNTIME_BINDING", "sha256": expected_sha256,
            "trainer_sha256": identity["trainer_sha256"], "combined_overlay_sha256": identity["combined_overlay_sha256"],
            "wrapper_sha256": identity["wrapper_sha256"]}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_parent_p_runtime(card_path: Path, binding_path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = read_json(binding_path)
    parent_root = Path(binding["runtime_root"])
    parent_trainer = parent_root / "pi0246_gate_pass_train.py"
    if sha256_file(parent_trainer) != binding.get("trainer_sha256"):
        raise CampaignError("sealed P runtime trainer source has changed")
    parent_module = load_module(parent_trainer, "pi0246_sealed_p_runtime_trainer")
    return parent_module.verify_p_runtime(
        card_path, parent_root, Path(binding["wrapper_path"]), binding_path, expected_sha256,
    )


def verify_prefix(schedule: list[dict[str, Any]], cohort: list[dict[str, Any]]) -> None:
    """Use the original R0 prefix only if it is the exact fixed16/125-per-cell table."""
    if len(schedule) < STEPS:
        raise CampaignError(f"R0 exposure schedule has only {len(schedule)} rows")
    prefix = schedule[:STEPS]
    counts = Counter(int(row["fixed_index"]) for row in prefix)
    if set(counts) != set(range(16)) or set(counts.values()) != {125}:
        raise CampaignError(f"R0 first-2000 schedule is not 125 per fixed window: {dict(counts)}")
    for ordinal, row in enumerate(prefix, start=1):
        fixed = cohort[int(row["fixed_index"])]
        if int(row["optimizer_step"]) != ordinal or str(row["source_episode_id"]) != str(fixed["source_episode_id"]):
            raise CampaignError(f"R0 schedule/cohort identity mismatch at step {ordinal}")


def summarize_config(pi0228: Any, config: Any, embodiment: str) -> dict[str, Any]:
    result = pi0228.config_summary(config)
    result.update({
        "backbone_trainable_params_fp32": bool(config.model.backbone_trainable_params_fp32),
        "load_bf16": bool(config.model.load_bf16),
        "max_grad_norm": float(config.training.max_grad_norm),
        "dataloader_num_workers": int(config.training.dataloader_num_workers),
        "gradient_checkpointing": bool(config.training.gradient_checkpointing),
        "num_gpus": int(config.training.num_gpus),
        "eval_strategy": str(config.training.eval_strategy),
        "modality_config_keys": [str(embodiment)],
        "state_dropout_prob": float(config.model.state_dropout_prob),
        "random_rotation_angle": None if config.model.random_rotation_angle is None else float(config.model.random_rotation_angle),
        "shortest_image_edge": int(config.model.shortest_image_edge),
        "crop_fraction": float(config.model.crop_fraction),
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "save_step_set": list(CHECKPOINTS),
    })
    return result


def build_runtime_config(paths: dict[str, str], output: Path, fixed: Any) -> tuple[Any, Any, dict[str, Any]]:
    pi0228 = fixed.load_module(Path(paths["code_root"]) / "scripts/pi0228_h200_large_adapt_v1.py", "pi0246_pi0228")
    args = argparse.Namespace(
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        dataset_dir=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
        groot_source=Path(paths["groot_source"]),
    )
    config = pi0228.build_config(args, output, steps=STEPS, save_steps=1000, save_limit=3)
    config.training.max_grad_norm = 1.0
    config.training.lr_scheduler_type = "cosine"
    config.training.learning_rate = 1e-5
    config.training.weight_decay = 1e-5
    config.training.dataloader_num_workers = 0
    config.training.gradient_checkpointing = False
    config.training.num_gpus = 1
    config.training.eval_strategy = "no"
    config.training.logging_steps = 1
    config.training.seed = SEED
    config.data.seed = SEED
    config.data.num_shards_per_epoch = 1000
    embodiment = config.data.datasets[0].embodiment_tag
    config.data.modality_configs = {embodiment: config.data.modality_configs[embodiment]}
    return pi0228, config, summarize_config(pi0228, config, embodiment)


def make_config(paths: dict[str, str], output: Path) -> tuple[Any, Any, dict[str, Any]]:
    fixed = load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", "pi0246_pi0241")
    pi0228, _, summary = build_runtime_config(paths, output, fixed)
    return fixed, pi0228, summary


def functional_preflight(args: argparse.Namespace) -> None:
    card = read_json(args.run_card)
    seal = read_json(args.run_card.with_name("run_card.seal.json"))
    arm = str(card.get("arm"))
    if (seal.get("sha256") != sha256_file(args.run_card)
            or card.get("schema") != "PI0246_ARM_RUN_CARD_V1"
            or card.get("status") != "SEALED_READY_NOT_STARTED"
            or arm not in {"T", "P", "PT"}
            or card.get("family") != ARM_DEFS[arm][0]
            or card.get("loss") != ARM_DEFS[arm][1]):
        raise CampaignError("functional preflight requires a sealed PI0246 T, P, or PT card")
    prefix = f"PI0246_{arm}_RUNTIME"
    runtime_root = Path(os.environ[f"{prefix}_ROOT"])
    wrapper = Path(os.environ[f"{prefix}_WRAPPER"])
    binding_path = Path(os.environ[f"{prefix}_BINDING"])
    expected_binding = os.environ[f"{prefix}_BINDING_SHA256"]
    verifiers = {"T": verify_t_runtime, "P": verify_p_runtime, "PT": verify_pt_runtime}
    binding = verifiers[arm](args.run_card, runtime_root, wrapper, binding_path, expected_binding)
    paths = card["paths"]
    import torch
    visible = torch.cuda.device_count()
    execution = validate_execution_profile(
        str(card.get("execution_profile", "H200_PBS")), os.environ.get("PBS_JOBID"),
        visible, torch.cuda.get_device_name(0) if visible == 1 else None,
    )
    fixed = load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", "pi0246_pi0241_preflight")
    fixed.MAX_STEPS = STEPS
    fixed.CHECKPOINT_STEPS = CHECKPOINTS
    runtime = fixed.bind_runtime(card)
    output = Path(paths["output_root"]) / f"functional_preflight_{arm.lower()}_v1"
    if output.exists():
        raise CampaignError(f"refusing to reuse functional preflight output: {output}")
    output.mkdir(parents=True)
    fixed.install_fixed_dataset_factory(card, output)
    _, config, summary = build_runtime_config(paths, output / "training", fixed)
    if summary != card["resolved_config"]:
        raise CampaignError("functional preflight config differs from sealed T config")
    cfg_dir = output / "experiment_cfg"
    cfg_dir.mkdir()
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    # Pipeline construction and _create_dataset load the processor/data only; setup() would load model weights.
    pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
    train_dataset, eval_dataset = pipeline._create_dataset(cfg_dir)
    if eval_dataset is not None:
        raise CampaignError("functional preflight unexpectedly constructed an eval dataset")
    datapoint = next(iter(train_dataset))
    action, mask = fixed.find_action_mask(pipeline.processor.collator([datapoint]))
    if list(action.shape) != [1, 40, 132] or list(mask.shape) != [1, 40, 132]:
        raise CampaignError(f"functional preflight action/mask shape mismatch: {action.shape}, {mask.shape}")
    if not fixed.np.isfinite(action).all() or not fixed.np.isfinite(mask).all():
        raise CampaignError("functional preflight produced nonfinite action/mask values")
    active_slots = fixed.np.flatnonzero(fixed.np.any(mask != 0, axis=(0, 1))).tolist()
    if active_slots != list(range(10)):
        raise CampaignError(f"functional preflight active action slots drift: {active_slots}")
    consumption_path = output / "pi0241_sample_consumption.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    planned = read_json(Path(card["schedule"]["path"]))["rows"][0]
    if len(consumption) != 1 or consumption[0]["source_episode_id"] != planned["source_episode_id"]:
        raise CampaignError("functional preflight did not decode the first sealed TRAIN schedule row")
    report = {
        "schema": f"PI0246_{arm}_FUNCTIONAL_PREFLIGHT_V1", "status": "PASS_NO_MODEL",
        "run_card_sha256": sha256_file(args.run_card), "runtime_binding_sha256": binding["sha256"],
        "config_sha256": canonical_sha256(summary), "execution": execution,
        "runtime": runtime, "source_episode_id": consumption[0]["source_episode_id"],
        "action_shape": list(action.shape), "mask_shape": list(mask.shape), "active_slots": active_slots,
        "eval_dataset": None, "model_weight_loads": 0, "model_forwards": 0,
        "get_action_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
    }
    write_json(output / "functional_preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def prepare(args: argparse.Namespace) -> None:
    r0_card = read_json(args.r0_card)
    seal = read_json(args.r0_card.with_name("pi0243_run_card.seal.json"))
    if seal.get("run_card_sha256") != sha256_file(args.r0_card):
        raise CampaignError("R0 run-card seal mismatch")
    r0_paths = r0_card["paths"]
    cohort_path = args.r0_root / "stage_cohort.json"
    schedule_path = args.r0_root / "stage_exposure_schedule.json"
    if sha256_file(cohort_path) != r0_card["cohort"]["sha256"]:
        raise CampaignError("compute fixed16 cohort differs from sealed R0 tree")
    if sha256_file(schedule_path) != r0_card["schedule"]["sha256"]:
        raise CampaignError("compute exposure schedule differs from sealed R0 tree")
    cohort_source = read_json(cohort_path)["rows"]
    source_ids = [str(value) for value in r0_card["fixed16_source_ids"]]
    cohort = [row for row in cohort_source if str(row["source_episode_id"]) in set(source_ids)]
    if [str(row["source_episode_id"]) for row in cohort] != source_ids:
        raise CampaignError("compute-mirror fixed16 cohort differs from sealed R0 source order")
    schedule_all = read_json(schedule_path)["rows"]
    verify_prefix(schedule_all, cohort)
    p0 = read_json(args.p0_effective)
    if sha256_file(args.p0_effective) != EXPECTED_P0_SHA256 or p0.get("native_decode_max_abs_error", 1.0) > 1e-6 or p0.get("model_weight_loads") != 0:
        raise CampaignError("PI0245 P0 decoder binding is invalid")
    noise_manifest = args.p1_root.parent / "p1_noise_bank" / "manifest.json"
    if sha256_file(noise_manifest) != EXPECTED_NOISE_SHA256:
        raise CampaignError("sealed P1 noise bank identity mismatch")
    restore = read_json(args.r0_model_root / "pi0245_inference_restore.json")
    if restore.get("optimizer_restored") is not False:
        raise CampaignError("R0 initializer must be weights-only; optimizer restore is forbidden")
    if args.output_root.exists():
        raise CampaignError(f"refusing to reuse campaign root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    cohort_file = args.output_root / "fixed16_cohort.json"
    schedule_file = args.output_root / "fixed2000_schedule.json"
    write_json(cohort_file, {"schema": "PI0246_FIXED16_COHORT_V1", "source_sha256": sha256_file(cohort_path), "rows": cohort})
    write_json(schedule_file, {
        "schema": "PI0246_FIXED2000_R0_PREFIX_V1",
        "source_schedule_sha256": sha256_file(schedule_path),
        "source_prefix_count": STEPS,
        "rows": schedule_all[:STEPS],
    })
    paths = dict(r0_paths)
    paths.update({
        "base_model_path": str(args.r0_model_root),
        "output_root": "",
        "p0_effective": str(args.p0_effective),
        "p1_root": str(args.p1_root),
        "campaign_code_root": str(args.campaign_code_root),
    })
    arms = {}
    for key, (family, loss) in ARMS.items():
        arm_root = args.output_root / key
        arm_root.mkdir()
        arm_paths = {**paths, "output_root": str(arm_root)}
        _, _, summary = make_config(arm_paths, arm_root / "training")
        r0_config = r0_card["resolved_config"]
        mismatches = {
            name: {"r0": r0_config.get(name), "arm": summary.get(name)}
            for name in r0_config
            if name not in CHANGEABLE_CONFIG and name in summary and r0_config.get(name) != summary.get(name)
        }
        if mismatches:
            raise CampaignError(f"frozen R0 config changed: {mismatches}")
        card = {
            "schema": "PI0246_ARM_RUN_CARD_V1",
            "status": "SEALED_READY_NOT_STARTED",
            "arm": key,
            "family": family,
            "loss": loss,
            "seed": SEED,
            "parent": {"role": "R0_FINAL10000_WEIGHTS_ONLY", "model_root": str(args.r0_model_root),
                       "restore_manifest_sha256": sha256_file(args.r0_model_root / "pi0245_inference_restore.json"),
                       "revision": restore["revision"], "files": restore["files"]},
            "p0_effective_sha256": sha256_file(args.p0_effective),
            "noise_bank_manifest_sha256": sha256_file(noise_manifest),
            "r0_card_sha256": sha256_file(args.r0_card),
            "p1_closure_path": str(args.p1_root / "p1_closure.json"),
            "paths": arm_paths,
            "cohort": {"path": str(cohort_file), "sha256": sha256_file(cohort_file), "count": 16},
            "schedule": {"path": str(schedule_file), "sha256": sha256_file(schedule_file), "count": STEPS,
                         "source": "exact first 2000 rows of the sealed R0 schedule", "per_window_count": 125},
            "resolved_config": summary,
            "checkpoint_steps": list(CHECKPOINTS),
            "optimizer": {"name": "adamw_torch", "fresh": True, "learning_rate": 1e-5,
                          "weight_decay": 1e-5, "max_grad_norm": 1.0, "scheduler": "cosine",
                          "warmup_steps": 10, "max_steps": STEPS},
            "forbidden_counts": {"calib_queries": 0, "dev_e_reads": 0, "final_f_reads": 0,
                                 "wbc_steps": 0, "sim_steps": 0, "dagger_steps": 0},
            "selected_checkpoint": "NONE",
        }
        card_path = arm_root / "run_card.json"
        write_json(card_path, card)
        write_json(arm_root / "run_card.seal.json", {"sha256": sha256_file(card_path)})
        arms[key] = {"family": family, "run_card": str(card_path), "run_card_sha256": sha256_file(card_path)}
    manifest = {
        "schema": "PI0246_P2_CAMPAIGN_PREP_V1", "status": "READY_NOT_STARTED",
        "r0_card_sha256": sha256_file(args.r0_card), "p0_effective_sha256": sha256_file(args.p0_effective),
        "cohort_sha256": sha256_file(cohort_file), "schedule_sha256": sha256_file(schedule_file),
        "schedule_counts": dict(sorted(Counter(int(row["fixed_index"]) for row in schedule_all[:STEPS]).items())),
        "arms": arms, "training_started": False, "selected_checkpoint": "NONE",
    }
    write_json(args.output_root / "campaign_preparation.json", manifest)
    write_json(args.output_root / "campaign_preparation.seal.json", {"sha256": sha256_file(args.output_root / "campaign_preparation.json")})
    print(json.dumps(manifest, indent=2, sort_keys=True))


def verify_p1(path: Path, *, expected_order: list[str], expected_r0_manifest_sha256: str, expected_r0_revision: str, expected_r0_records_sha256: str) -> dict[str, Any]:
    closure = read_json(path)
    if closure.get("schema") != "PI0245_P1_CLOSURE_V1" or closure.get("status") != "PASS_COMPLETE_32_QUERY_MATRIX":
        raise CampaignError("P1 is not a complete valid 32-query closure")
    if int(closure.get("attempted_queries", -1)) != EXPECTED_P1_QUERIES or int(closure.get("completed_cells", -1)) != EXPECTED_P1_QUERIES:
        raise CampaignError("P1 query/cell count is not exactly 32")
    if len(closure.get("cells", [])) != EXPECTED_P1_QUERIES or len(set(row["cell_key"] for row in closure["cells"])) != EXPECTED_P1_QUERIES:
        raise CampaignError("P1 cell identities are incomplete or duplicated")
    if closure.get("query_order") != expected_order or [row.get("cell_key") for row in closure["cells"]] != expected_order:
        raise CampaignError("P1 cells do not match the frozen R0/R2 × bridge4 × noise order")
    if closure.get("noise_bank_manifest_sha256") != EXPECTED_NOISE_SHA256:
        raise CampaignError("P1 closure is not bound to the sealed noise bank")
    if closure.get("R0_records_sha256") != expected_r0_records_sha256:
        raise CampaignError("P1 closure source records identity mismatch")
    for key in expected_order:
        model, source, noise = key.split("__", 2)
        if model != "R0":
            continue
        source_stem = source.split(":", 1)[0]
        record_path = path.parent / "cells" / f"{model}__{source_stem}__{noise}" / "record.json"
        record = read_json(record_path)
        if record.get("manifest_sha256") != expected_r0_manifest_sha256 or record.get("revision") != expected_r0_revision:
            raise CampaignError(f"P1 R0 model file/revision identity differs from initializer: {key}")
    return closure


def prepare_p(args: argparse.Namespace) -> None:
    source_path = args.source_run_card
    source = read_json(source_path)
    seal = read_json(source_path.with_name("run_card.seal.json"))
    if (seal.get("sha256") != sha256_file(source_path)
            or source.get("arm") != "T" or source.get("family") != ARMS["T"][0]
            or source.get("loss") != ARMS["T"][1]
            or source.get("status") != "SEALED_READY_NOT_STARTED"):
        raise CampaignError("P preparation requires the exact sealed T card as its frozen config template")
    if args.output_root.exists():
        raise CampaignError(f"refusing to reuse P output root: {args.output_root}")
    paths = source["paths"]
    effective = Path(paths["p0_effective"])
    if sha256_file(effective) != EXPECTED_P0_SHA256:
        raise CampaignError("P0 effective scale-aware decoder binding changed")
    p1_script = load_module(Path(paths["campaign_code_root"]) / "pi0245_p1_stock.py", "pi0246_p1_prepare")
    expected_order = [f"{model}__{source_id}__{noise}" for model in ("R0", "R2") for source_id in p1_script.SOURCES for noise in p1_script.NOISES]
    closure = verify_p1(
        Path(source["p1_closure_path"]), expected_order=expected_order,
        expected_r0_manifest_sha256=source["parent"]["restore_manifest_sha256"],
        expected_r0_revision=source["parent"]["revision"],
        expected_r0_records_sha256=p1_script.RECORD_SHA,
    )
    restore_path = Path(paths["base_model_path"]) / "pi0245_inference_restore.json"
    restore = read_json(restore_path)
    restore_sha = sha256_file(restore_path)
    if (not Path(paths["base_model_path"]).is_dir()
            or restore_sha != source["parent"]["restore_manifest_sha256"]
            or restore.get("revision") != source["parent"]["revision"]
            or restore.get("files") != source["parent"]["files"]
            or restore.get("optimizer_restored") is not False):
        raise CampaignError("P initializer is not the sealed R0-final10000 weights-only tree")
    model_identity = {"manifest_sha256": restore_sha, "revision": restore["revision"]}
    card = json.loads(json.dumps(source))
    card.update({
        "arm": "P", "family": P_ARM[0], "loss": P_ARM[1],
        "paths": {**paths, "output_root": str(args.output_root)},
        "branch_admission": {
            "issue_comment_id": 5825569583,
            "C2000": "VALID_JOINT_GATE_FAIL",
            "T2000": "VALID_JOINT_GATE_FAIL",
            "cross_noise_diagnostic_only": True,
        },
        "loss_contract": {
            "native": "L_native", "auxiliary": "L_phys_pos", "coefficient": 0.1,
            "same_native_forward": True, "normalized_physical_reference_m": 0.05,
            "decoder": "PI0245 scale-aware XYZ inverse-normalization and fixed-reference frame",
            "mask": "all three XYZ slots valid at the point",
            "reduction": "sum of squared XYZ residuals, mean over valid points; no division by 3",
        },
        "archive": {
            "repo_id": "leoyuer/pi0246-gate-pass-campaign",
            "path_prefix": "P",
            "public": True,
            "checkpoint_steps": list(CHECKPOINTS),
        },
    })
    args.output_root.mkdir(parents=True)
    card_path = args.output_root / "run_card.json"
    write_json(card_path, card)
    card_sha = sha256_file(card_path)
    write_json(args.output_root / "run_card.seal.json", {"sha256": card_sha})
    prepared = {
        "schema": "PI0246_P_PREPARATION_V1", "status": "READY_NOT_STARTED",
        "source_T_card_sha256": sha256_file(source_path), "run_card_sha256": card_sha,
        "p1_closure_sha256": sha256_file(Path(source["p1_closure_path"])),
        "p1_cells": len(closure["cells"]), "initializer_manifest_sha256": model_identity["manifest_sha256"],
        "cohort_sha256": card["cohort"]["sha256"], "schedule_sha256": card["schedule"]["sha256"],
        "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS), "optimizer": card["optimizer"],
        "loss_contract": card["loss_contract"], "training_started": False,
        "selected_checkpoint": "NONE",
    }
    write_json(args.output_root / "preparation.json", prepared)
    write_json(args.output_root / "preparation.seal.json", {"sha256": sha256_file(args.output_root / "preparation.json")})
    print(json.dumps(prepared, indent=2, sort_keys=True))


def percentile_linear(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= quantile <= 1:
        raise CampaignError("percentile needs nonempty values and q in [0,1]")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def verify_p_final_failure(source_card_path: Path, verification_path: Path, view_path: Path) -> dict[str, Any]:
    source = read_json(source_card_path)
    seal = read_json(source_card_path.with_name("run_card.seal.json"))
    report = read_json(verification_path)
    view = read_json(view_path)
    if (seal.get("sha256") != sha256_file(source_card_path) or source.get("arm") != "P"
            or source.get("family") != P_ARM[0] or source.get("loss") != P_ARM[1]
            or report.get("schema") != "PI0246_PAIRED_PROFILE_CPU_VERIFICATION_V1"
            or report.get("status") != "PASS_READONLY_PROFILE_REPLAY"
            or report.get("profile") != "P2000" or report.get("records") != 16
            or report.get("model_weights_loaded") != 0 or report.get("model_queries") != 0
            or view.get("schema") != "PI0246_INFERENCE_VIEW_VERIFICATION_V1"
            or view.get("status") != "INFERENCE_VIEW_VERIFIED"
            or view.get("family") != P_ARM[0] or view.get("checkpoint_step") != 2000
            or view.get("optimizer_restored") is not False):
        raise CampaignError("PT admission requires a valid P2000 CPU replay and verified inference view")
    receipt_path = Path(source["paths"]["output_root"]) / "durable_receipts/checkpoint-2000.durable_receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("family") != P_ARM[0]
            or receipt.get("checkpoint_step") != 2000 or receipt.get("max_steps") != STEPS
            or view.get("full_checkpoint_receipt_sha256") != sha256_file(receipt_path)
            or view.get("source_tree_sha256") != receipt.get("source_record", {}).get("inventory", {}).get("tree_sha256")
            or view.get("revision") != receipt.get("hf_manifest_commit_id")
            or report.get("model_manifest_sha256") != view.get("restore_manifest_sha256")
            or report.get("model_revision") != view.get("revision")):
        raise CampaignError("P2000 screen, inference view, and durable full-checkpoint receipt do not join")
    comparisons = report.get("comparisons", [])
    ids = [str(row.get("source_episode_id", "")) for row in comparisons]
    if (len(comparisons) != 16 or len(set(ids)) != 16 or any(not row for row in ids)
            or any(row.get("gate") not in {"PASS", "WARN", "REJECT"} for row in comparisons)):
        raise CampaignError("P2000 screen comparisons are incomplete or malformed")
    counted = dict(Counter(str(row["gate"]) for row in comparisons))
    if report.get("gate_counts") != counted:
        raise CampaignError("P2000 gate counts differ from the individual comparisons")
    xyz = [float(row["xyz_rmse_m"]) for row in comparisons]
    so3 = [float(row["so3_rmse_rad"]) for row in comparisons]
    if not all(math.isfinite(value) for value in xyz + so3):
        raise CampaignError("P2000 CPU replay contains nonfinite metrics")
    metrics = {
        "records": 16, "gate_counts": counted,
        "median_xyz_rmse_m": statistics.median(xyz),
        "p95_xyz_rmse_m": percentile_linear(xyz, 0.95),
        "median_so3_rmse_rad": statistics.median(so3),
    }
    passed = (
        counted.get("REJECT", 0) == 0
        and metrics["median_xyz_rmse_m"] <= 0.020
        and metrics["p95_xyz_rmse_m"] <= 0.035
        and metrics["median_so3_rmse_rad"] <= 0.050
    )
    if passed:
        raise CampaignError("P2000 passes its joint gate; PT is not authorized")
    return {
        "status": "VALID_JOINT_GATE_FAIL", "metrics": metrics,
        "P_run_card_sha256": sha256_file(source_card_path),
        "P2000_cpu_verification_sha256": sha256_file(verification_path),
        "P2000_inference_view_sha256": sha256_file(view_path),
        "P2000_durable_receipt_sha256": sha256_file(receipt_path),
        "P2000_records_sha256": report["records_sha256"],
        "P2000_queries_sha256": report["queries_sha256"],
        "P2000_revision": report["model_revision"],
        "P2000_model_manifest_sha256": report["model_manifest_sha256"],
    }


def prepare_pt(args: argparse.Namespace) -> None:
    source_path = args.source_run_card
    source = read_json(source_path)
    if (source.get("arm") != "P" or source.get("family") != P_ARM[0]
            or source.get("loss") != P_ARM[1] or source.get("status") != "SEALED_READY_NOT_STARTED"):
        raise CampaignError("PT requires the original sealed P card as the fixed16/config source")
    admission = verify_p_final_failure(source_path, args.p_verification, args.p_inference_view)
    p_binding_path = args.p_runtime_binding
    p_binding = read_json(p_binding_path)
    p_binding_sha = read_json(p_binding_path.with_name("P_runtime_binding.seal.json")).get("sha256")
    validate_parent_p_runtime(source_path, p_binding_path, str(p_binding_sha))
    parent = source["parent"]
    if (parent.get("role") != "R0_FINAL10000_WEIGHTS_ONLY"
            or parent.get("model_root") != source["paths"]["base_model_path"]):
        raise CampaignError("PT must start from the sealed R0-final10000 weights-only initializer")
    paths = source["paths"]
    p1_script = load_module(Path(paths["campaign_code_root"]) / "pi0245_p1_stock.py", "pi0246_p1_pt_prepare")
    model_identity = p1_script.verify_model_files(Path(paths["base_model_path"]))
    if (model_identity["manifest_sha256"] != parent["restore_manifest_sha256"]
            or model_identity["revision"] != parent["revision"]):
        raise CampaignError("PT initializer differs from the sealed R0-final10000 source")
    if args.output_root.exists():
        raise CampaignError(f"refusing to reuse PT output root: {args.output_root}")
    pt_paths = {**paths, "output_root": str(args.output_root)}
    _, _, summary = make_config(pt_paths, args.output_root / "training")
    if summary != source["resolved_config"]:
        raise CampaignError("PT resolved training configuration differs from P/R0 frozen configuration")

    card = json.loads(json.dumps(source))
    card.update({
        "arm": "PT", "family": PT_ARM[0], "loss": PT_ARM[1], "paths": pt_paths,
        "branch_admission": {"source": "P2000", **admission},
        "predecessor_failure": admission,
        "parent_loss_runtime": {
            "binding_path": str(p_binding_path), "binding_sha256": p_binding_sha,
            "source_run_card_path": str(source_path),
            "runtime_root": p_binding["runtime_root"],
            "position_overlay_sha256": p_binding["position_overlay_sha256"],
            "physical_decoder_sha256": p_binding["physical_decoder_sha256"],
        },
        "loss_contract": {
            "native": "L_native", "auxiliary": ["L_phys_pos", "L_phys_delta"],
            "coefficients": {"L_phys_pos": 0.1, "L_phys_delta": 0.1},
            "same_native_forward": True,
            "position": source["loss_contract"],
            "delta": {
                "decoder": "PI0245 scale-aware XYZ inverse-normalization and fixed-reference frame",
                "dt_s": 0.05, "v_ref_mps": 1.0,
                "residual": "difference of decoded physical XYZ residuals at adjacent H40 indices, divided by dt and v_ref",
                "mask": "both adjacent points require all three XYZ slots valid",
                "reduction": "sum squared XYZ increment residual, mean over valid transitions; no division by 3",
            },
        },
        "archive": {**source["archive"], "path_prefix": "PT", "checkpoint_steps": list(CHECKPOINTS)},
    })
    args.output_root.mkdir(parents=True)
    evidence_dir = args.output_root / "predecessor_evidence"
    evidence_dir.mkdir()
    shutil.copyfile(args.p_verification, evidence_dir / "P2000_cpu_verification.json")
    shutil.copyfile(args.p_inference_view, evidence_dir / "P2000_inference_view_verified.json")
    shutil.copyfile(p_binding_path, evidence_dir / "P_runtime_binding.json")
    shutil.copyfile(p_binding_path.with_name("P_runtime_binding.seal.json"), evidence_dir / "P_runtime_binding.seal.json")
    card_path = args.output_root / "run_card.json"
    write_json(card_path, card)
    card_sha = sha256_file(card_path)
    write_json(args.output_root / "run_card.seal.json", {"sha256": card_sha})
    prepared = {
        "schema": "PI0246_PT_PREPARATION_V1", "status": "READY_NOT_STARTED",
        "run_card_sha256": card_sha, "source_P_card_sha256": admission["P_run_card_sha256"],
        "predecessor_failure": admission, "initializer_manifest_sha256": model_identity["manifest_sha256"],
        "cohort_sha256": card["cohort"]["sha256"], "schedule_sha256": card["schedule"]["sha256"],
        "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS), "optimizer": card["optimizer"],
        "loss": card["loss"], "loss_contract": card["loss_contract"],
        "training_started": False, "selected_checkpoint": "NONE",
    }
    write_json(args.output_root / "preparation.json", prepared)
    write_json(args.output_root / "preparation.seal.json", {"sha256": sha256_file(args.output_root / "preparation.json")})
    print(json.dumps(prepared, indent=2, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    card = read_json(args.run_card)
    if card.get("schema") != "PI0246_ARM_RUN_CARD_V1" or card.get("status") != "SEALED_READY_NOT_STARTED":
        raise CampaignError("arm card is not sealed and ready")
    seal = read_json(args.run_card.with_name("run_card.seal.json"))
    if seal.get("sha256") != sha256_file(args.run_card):
        raise CampaignError("arm card seal mismatch")
    arm = str(card["arm"])
    if arm not in ARM_DEFS or card["family"] != ARM_DEFS[arm][0] or card["loss"] != ARM_DEFS[arm][1]:
        raise CampaignError("arm/family/loss binding mismatch")
    t_binding = None
    t_runtime_root = None
    p_binding = None
    p_runtime_root = None
    pt_binding = None
    pt_runtime_root = None
    if arm == "T":
        t_runtime_root = Path(os.environ["PI0246_T_RUNTIME_ROOT"])
        t_wrapper = Path(os.environ["PI0246_T_RUNTIME_WRAPPER"])
        t_binding_path = Path(os.environ["PI0246_T_RUNTIME_BINDING"])
        t_binding = verify_t_runtime(
            args.run_card, t_runtime_root, t_wrapper, t_binding_path,
            os.environ["PI0246_T_RUNTIME_BINDING_SHA256"],
        )
    elif arm == "P":
        p_runtime_root = Path(os.environ["PI0246_P_RUNTIME_ROOT"])
        p_wrapper = Path(os.environ["PI0246_P_RUNTIME_WRAPPER"])
        p_binding = verify_p_runtime(
            args.run_card, p_runtime_root, p_wrapper,
            Path(os.environ["PI0246_P_RUNTIME_BINDING"]),
            os.environ["PI0246_P_RUNTIME_BINDING_SHA256"],
        )
    elif arm == "PT":
        pt_runtime_root = Path(os.environ["PI0246_PT_RUNTIME_ROOT"])
        pt_wrapper = Path(os.environ["PI0246_PT_RUNTIME_WRAPPER"])
        pt_binding_path = Path(os.environ["PI0246_PT_RUNTIME_BINDING"])
        pt_binding = verify_pt_runtime(
            args.run_card, pt_runtime_root, pt_wrapper, pt_binding_path,
            os.environ["PI0246_PT_RUNTIME_BINDING_SHA256"],
        )
    p1_script = load_module(Path(card["paths"]["campaign_code_root"]) / "pi0245_p1_stock.py", "pi0246_p1_identity")
    expected_order = [f"{model}__{source}__{noise}" for model in ("R0", "R2") for source in p1_script.SOURCES for noise in p1_script.NOISES]
    p1 = verify_p1(
        Path(card["p1_closure_path"]), expected_order=expected_order,
        expected_r0_manifest_sha256=card["parent"]["restore_manifest_sha256"],
        expected_r0_revision=card["parent"]["revision"],
        expected_r0_records_sha256=p1_script.RECORD_SHA,
    )
    paths = card["paths"]
    output = Path(paths["output_root"])
    training_output = output / "training"
    if training_output.exists():
        raise CampaignError(f"refusing to overwrite training attempt: {training_output}")
    import torch
    visible_devices = torch.cuda.device_count()
    execution = validate_execution_profile(
        str(card.get("execution_profile", "H200_PBS")), os.environ.get("PBS_JOBID"),
        visible_devices, torch.cuda.get_device_name(0) if visible_devices == 1 else None,
    )
    if int(card["optimizer"]["max_steps"]) != STEPS or int(card["schedule"]["count"]) != STEPS:
        raise CampaignError("fixed 2000-step contract mismatch")
    if sha256_file(Path(card["cohort"]["path"])) != card["cohort"]["sha256"] or sha256_file(Path(card["schedule"]["path"])) != card["schedule"]["sha256"]:
        raise CampaignError("cohort or exposure schedule changed after sealing")
    if sha256_file(Path(paths["p0_effective"])) != EXPECTED_P0_SHA256:
        raise CampaignError("P0 effective processor binding changed")
    model_identity = p1_script.verify_model_files(Path(paths["base_model_path"]))
    if model_identity["manifest_sha256"] != card["parent"]["restore_manifest_sha256"] or model_identity["revision"] != card["parent"]["revision"]:
        raise CampaignError("R0 final initializer changed after P1 admission")

    fixed = load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", "pi0246_pi0241_train")
    fixed.MAX_STEPS = STEPS
    fixed.CHECKPOINT_STEPS = CHECKPOINTS
    runtime = fixed.bind_runtime(card)
    runtime["execution"] = execution
    training_output.mkdir(parents=True)
    fixed.install_fixed_dataset_factory(card, training_output)
    fixed.install_update_audit(training_output, audit_steps={1, 10, 100, 1000, 2000}, expected_end_step=STEPS)
    pi0228, config, summary = build_runtime_config(paths, training_output, fixed)
    if summary != card["resolved_config"]:
        raise CampaignError("runtime config differs from sealed arm card")
    write_json(training_output / "resolved_config.json", summary)
    if arm in {"T", "P", "PT"}:
        import torch
        runtime_root = {"T": t_runtime_root, "P": p_runtime_root, "PT": pt_runtime_root}[arm]
        runtime_binding = {"T": t_binding, "P": p_binding, "PT": pt_binding}[arm]
        assert runtime_root is not None and runtime_binding is not None
        sys.path.insert(0, str(runtime_root))
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        effective = read_json(Path(paths["p0_effective"]))
        overlay_kwargs = {
            "scale": torch.tensor(effective["scale_xyz_m"], dtype=torch.float32),
            "offset": torch.tensor(effective["offset_xyz_m"], dtype=torch.float32),
            "state_min": torch.tensor(effective["state_min"], dtype=torch.float32),
            "state_max": torch.tensor(effective["state_max"], dtype=torch.float32),
        }
        if arm == "T":
            overlay_kwargs["treatment"] = True
            overlay_module = load_module(runtime_root / "pi0245_native_overlay.py", "pi0246_t_native_overlay")
            forward_log = training_output / "pi0245_native_overlay_forward.jsonl"
        elif arm == "P":
            overlay_kwargs["treatment"] = True
            overlay_module = load_module(runtime_root / "pi0246_physical_position.py", "pi0246_p_position_overlay")
            forward_log = training_output / "pi0246_physical_position_forward.jsonl"
        else:
            overlay_module = load_module(runtime_root / "pi0246_pt_combined_overlay.py", "pi0246_pt_combined_overlay")
            forward_log = training_output / "pi0246_pt_combined_forward.jsonl"
        overlay_audit = overlay_module.install(
            Gr00tN1d7ActionHead, **overlay_kwargs, audit_path=forward_log,
        )
        overlay_audit["runtime_binding_sha256"] = runtime_binding["sha256"]
    else:
        overlay_audit = None
    started = time.time()
    pi0228.run_official(config, Path(paths["groot_source"]))

    consumption_path = training_output / "pi0241_sample_consumption.jsonl"
    contracts_path = training_output / "pi0241_sample_contracts.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    contracts = [json.loads(line) for line in contracts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    planned = read_json(Path(card["schedule"]["path"]))["rows"]
    if len(consumption) != STEPS or [row["source_episode_id"] for row in consumption] != [row["source_episode_id"] for row in planned]:
        raise CampaignError("actual sample consumption differs from sealed 2000-row schedule")
    if len(contracts) != 16 or {int(row["fixed_index"]) for row in contracts} != set(range(16)):
        raise CampaignError("training action/mask contract lacks any fixed window")
    audit = read_json(training_output / "pi0241_update_audit.json")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != STEPS:
        raise CampaignError("optimizer/update audit is incomplete")
    if [int(read_json(training_output / f"checkpoint-{step}/trainer_state.json")["global_step"]) for step in CHECKPOINTS] != list(CHECKPOINTS):
        raise CampaignError("full Trainer checkpoint steps are incomplete")
    same_forward_audit = None
    if arm in {"T", "P", "PT"}:
        forward_log = training_output / (
            "pi0245_native_overlay_forward.jsonl" if arm == "T"
            else "pi0246_physical_position_forward.jsonl" if arm == "P"
            else "pi0246_pt_combined_forward.jsonl"
        )
        forward_rows = [json.loads(line) for line in forward_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(forward_rows) != STEPS or [int(row.get("forward_index", -1)) for row in forward_rows] != list(range(1, STEPS + 1)):
            raise CampaignError(f"{arm} same-forward component/tau audit is incomplete")
        aux_keys = ({"T": ("physical_delta_loss",), "P": ("physical_position_loss",),
                     "PT": ("physical_position_loss", "physical_delta_loss")})[arm]
        count_keys = ({"T": ("valid_transitions",), "P": ("valid_points",),
                       "PT": ("valid_points", "valid_transitions")})[arm]
        required_fields = {"native_loss", "total_loss", "tau", "saturated_xyz", *aux_keys, *count_keys}
        for row in forward_rows:
            if not required_fields.issubset(row):
                raise CampaignError(f"{arm} same-forward audit is missing a required component")
            numeric = [row["native_loss"], row["total_loss"], *(row[key] for key in aux_keys), *row["tau"]]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
                raise CampaignError(f"{arm} same-forward audit contains nonfinite values")
            if len(row["tau"]) != 1 or not 0.0 <= float(row["tau"][0]) <= 1.0:
                raise CampaignError(f"{arm} same-forward audit tau shape/range is invalid")
            expected_total = float(row["native_loss"]) + sum(0.1 * float(row[key]) for key in aux_keys)
            if not math.isclose(float(row["total_loss"]), expected_total, rel_tol=1e-5, abs_tol=1e-7):
                raise CampaignError(f"{arm} same-forward loss components do not sum to the prescribed objective")
            counts = [(int(row[key]), 39 if key == "valid_transitions" else 40) for key in count_keys]
            if any(not 0 < count <= maximum for count, maximum in counts) or not 0 <= int(row["saturated_xyz"]) <= 120:
                raise CampaignError(f"{arm} same-forward mask/saturation count is invalid")
        same_forward_audit = {
            "status": "COMPLETE", "rows": len(forward_rows),
            "auxiliary_loss_names": list(aux_keys), "valid_count_names": list(count_keys),
            "sha256": sha256_file(forward_log),
            "first_forward": forward_rows[0], "last_forward": forward_rows[-1],
            "native_loss_min": min(float(row["native_loss"]) for row in forward_rows),
            "native_loss_max": max(float(row["native_loss"]) for row in forward_rows),
            "auxiliary_loss_min": {key: min(float(row[key]) for row in forward_rows) for key in aux_keys},
            "auxiliary_loss_max": {key: max(float(row[key]) for row in forward_rows) for key in aux_keys},
            "valid_count_min": {key: min(int(row[key]) for row in forward_rows) for key in count_keys},
            "valid_count_max": {key: max(int(row[key]) for row in forward_rows) for key in count_keys},
            "saturated_xyz_total": sum(int(row["saturated_xyz"]) for row in forward_rows),
        }
        overlay_audit["forward_log_sha256"] = same_forward_audit["sha256"]
        overlay_audit["update_audit_sha256"] = sha256_file(training_output / "pi0241_update_audit.json")
    checkpoints = {str(step): fixed.checkpoint_record(training_output / f"checkpoint-{step}", step) for step in CHECKPOINTS}
    update = audit["final_parameters"]
    integrity = {
        "projector_updated": float(update["projector"]["delta_l2"]) > 0,
        "diffusion_updated": float(update["diffusion"]["delta_l2"]) > 0,
        "frozen_backbone_unchanged": float(update["frozen_backbone"]["delta_l2"]) == 0,
        "frozen_vlln_unchanged": float(update["frozen_vlln"]["delta_l2"]) == 0,
        "optimizer_membership_valid": not audit["optimizer_membership"]["missing_trainable"] and not audit["optimizer_membership"]["illegal_frozen"],
    }
    if arm in {"T", "P", "PT"}:
        integrity["same_forward_audit_complete"] = same_forward_audit is not None and same_forward_audit["rows"] == STEPS
    projector_reconciliation_required = needs_projector_reconciliation(arm, integrity)
    result = {
        "schema": f"PI0246_ARM_{arm}_TRAINING_MANIFEST_V1",
        "status": (
            "TRAINING_COMPLETE_PROJECTOR_RECONCILIATION_REQUIRED" if projector_reconciliation_required
            else {"P": "PASS_P_TRAINING_INTEGRITY", "PT": "PASS_PT_TRAINING_INTEGRITY"}.get(arm, "PASS_TRAINING_INTEGRITY") if all(integrity.values())
            else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT"
        ),
        "arm": arm, "family": card["family"], "run_card_sha256": sha256_file(args.run_card),
        "p1_closure_sha256": sha256_file(Path(card["p1_closure_path"])), "p1_cells": len(p1["cells"]),
        "elapsed_s": time.time() - started, "global_step": STEPS, "sample_consumption": len(consumption),
        "pbs_job_id": execution["pbs_job_id"], "execution": execution,
        "schedule_sha256": card["schedule"]["sha256"], "sample_contracts": contracts,
        "update_integrity": integrity, "update_audit_sha256": sha256_file(training_output / "pi0241_update_audit.json"),
        "projector_reconciliation_required": projector_reconciliation_required,
        "checkpoints": checkpoints, "runtime_binding": runtime,
        "overlay": overlay_audit, "selected_checkpoint": "NONE",
        "same_forward_audit": same_forward_audit,
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json(output / "training_manifest.json", result)
    if not all(integrity.values()) and not projector_reconciliation_required:
        raise CampaignError("optimizer/update integrity failure")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--prepare-p", action="store_true")
    mode.add_argument("--prepare-pt", action="store_true")
    mode.add_argument("--functional-preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--seal-t-runtime", action="store_true")
    mode.add_argument("--seal-p-runtime", action="store_true")
    mode.add_argument("--seal-pt-runtime", action="store_true")
    parser.add_argument("--r0-card", type=Path)
    parser.add_argument("--r0-root", type=Path)
    parser.add_argument("--r0-model-root", type=Path)
    parser.add_argument("--p0-effective", type=Path)
    parser.add_argument("--p1-root", type=Path)
    parser.add_argument("--campaign-code-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--source-run-card", type=Path)
    parser.add_argument("--p-verification", type=Path)
    parser.add_argument("--p-inference-view", type=Path)
    parser.add_argument("--p-runtime-binding", type=Path)
    args = parser.parse_args()
    if args.prepare:
        required = (args.r0_card, args.r0_root, args.r0_model_root, args.p0_effective, args.p1_root, args.campaign_code_root, args.output_root)
        if any(value is None for value in required):
            parser.error("--prepare requires R0/P0/P1 identities and output root")
        prepare(args)
    elif args.prepare_p:
        if args.source_run_card is None or args.output_root is None:
            parser.error("--prepare-p requires --source-run-card and --output-root")
        prepare_p(args)
    elif args.prepare_pt:
        required = (args.source_run_card, args.p_verification, args.p_inference_view,
                    args.p_runtime_binding, args.output_root)
        if any(value is None for value in required):
            parser.error("--prepare-pt requires P card, P2000 CPU verification, inference-view/runtime receipts, and output root")
        prepare_pt(args)
    elif args.functional_preflight:
        if args.run_card is None:
            parser.error("--functional-preflight requires --run-card")
        functional_preflight(args)
    elif args.train:
        if args.run_card is None:
            parser.error("--train requires --run-card")
        train(args)
    elif args.seal_t_runtime:
        required = (args.run_card, args.runtime_root, args.wrapper)
        if any(value is None for value in required):
            parser.error("--seal-t-runtime requires --run-card, --runtime-root, and --wrapper")
        print(json.dumps(seal_t_runtime(args.run_card, args.runtime_root, args.wrapper), indent=2, sort_keys=True))
    elif args.seal_pt_runtime:
        required = (args.run_card, args.runtime_root, args.wrapper)
        if any(value is None for value in required):
            parser.error("--seal-pt-runtime requires --run-card, --runtime-root, and --wrapper")
        print(json.dumps(seal_pt_runtime(args.run_card, args.runtime_root, args.wrapper), indent=2, sort_keys=True))
    else:
        required = (args.run_card, args.runtime_root, args.wrapper)
        if any(value is None for value in required):
            parser.error("--seal-p-runtime requires --run-card, --runtime-root, and --wrapper")
        print(json.dumps(seal_p_runtime(args.run_card, args.runtime_root, args.wrapper), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
