#!/usr/bin/env python3
"""PI0246 C/T arms, reusing the sealed PI0241 fixed-dataset and update audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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


class CampaignError(RuntimeError):
    pass


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


def t_runtime_identity(card_path: Path, runtime_root: Path, wrapper: Path) -> dict[str, Any]:
    card = read_json(card_path)
    root = runtime_root.resolve()
    parent = Path(card["paths"]["campaign_code_root"])
    code_root = Path(card["paths"]["code_root"])
    parent_delta = parent / "pi0245_physical_delta.py"
    runtime_delta = root / "pi0245_physical_delta.py"
    if sha256_file(runtime_delta) != sha256_file(parent_delta):
        raise CampaignError("T runtime physical-delta helper differs from the sealed PI0245 helper")
    return {
        "schema": "PI0246_T_SAME_FORWARD_LOGGING_RUNTIME_V1",
        "run_card_sha256": sha256_file(card_path),
        "family": card["family"],
        "loss": card["loss"],
        "seed": card["seed"],
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def make_config(paths: dict[str, str], output: Path) -> tuple[Any, Any, dict[str, Any]]:
    fixed = load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", "pi0246_pi0241")
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
    return fixed, pi0228, summarize_config(pi0228, config, embodiment)


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


def train(args: argparse.Namespace) -> None:
    card = read_json(args.run_card)
    if card.get("schema") != "PI0246_ARM_RUN_CARD_V1" or card.get("status") != "SEALED_READY_NOT_STARTED":
        raise CampaignError("arm card is not sealed and ready")
    seal = read_json(args.run_card.with_name("run_card.seal.json"))
    if seal.get("sha256") != sha256_file(args.run_card):
        raise CampaignError("arm card seal mismatch")
    arm = str(card["arm"])
    if arm not in ARMS or card["family"] != ARMS[arm][0] or card["loss"] != ARMS[arm][1]:
        raise CampaignError("arm/family/loss binding mismatch")
    t_binding = None
    t_runtime_root = None
    if arm == "T":
        t_runtime_root = Path(os.environ["PI0246_T_RUNTIME_ROOT"])
        t_wrapper = Path(os.environ["PI0246_T_RUNTIME_WRAPPER"])
        t_binding_path = Path(os.environ["PI0246_T_RUNTIME_BINDING"])
        t_binding = verify_t_runtime(
            args.run_card, t_runtime_root, t_wrapper, t_binding_path,
            os.environ["PI0246_T_RUNTIME_BINDING_SHA256"],
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
    if not os.environ.get("PBS_JOBID"):
        raise CampaignError("formal PI0246 training must run as a PBS job")
    import torch
    if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise CampaignError(f"PI0246 fine-tuning requires exactly one Hopper H200; visible={torch.cuda.device_count()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'}")
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
    training_output.mkdir(parents=True)
    fixed.install_fixed_dataset_factory(card, training_output)
    fixed.install_update_audit(training_output, audit_steps={1, 10, 100, 1000, 2000}, expected_end_step=STEPS)
    pi0228 = load_module(Path(paths["code_root"]) / "scripts/pi0228_h200_large_adapt_v1.py", "pi0246_pi0228_train")
    helper_args = argparse.Namespace(
        stage10_root=Path(paths["stage10_root"]), base_model_path=Path(paths["base_model_path"]),
        dataset_dir=Path(paths["train_dataset"]), modality_config=Path(paths["modality_config"]),
        groot_source=Path(paths["groot_source"]),
    )
    config = pi0228.build_config(helper_args, training_output, steps=STEPS, save_steps=1000, save_limit=3)
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
    summary = summarize_config(pi0228, config, embodiment)
    if summary != card["resolved_config"]:
        raise CampaignError("runtime config differs from sealed arm card")
    write_json(training_output / "resolved_config.json", summary)
    if arm == "T":
        import torch
        assert t_runtime_root is not None and t_binding is not None
        sys.path.insert(0, str(t_runtime_root))
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        overlay_module = load_module(t_runtime_root / "pi0245_native_overlay.py", "pi0246_t_native_overlay")
        effective = read_json(Path(paths["p0_effective"]))
        forward_log = training_output / "pi0245_native_overlay_forward.jsonl"
        overlay_audit = overlay_module.install(
            Gr00tN1d7ActionHead, scale=torch.tensor(effective["scale_xyz_m"], dtype=torch.float32),
            offset=torch.tensor(effective["offset_xyz_m"], dtype=torch.float32),
            state_min=torch.tensor(effective["state_min"], dtype=torch.float32),
            state_max=torch.tensor(effective["state_max"], dtype=torch.float32), treatment=True,
            audit_path=forward_log,
        )
        overlay_audit["runtime_binding_sha256"] = t_binding["sha256"]
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
    if arm == "T":
        forward_log = training_output / "pi0245_native_overlay_forward.jsonl"
        forward_rows = [json.loads(line) for line in forward_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(forward_rows) != STEPS or [int(row.get("forward_index", -1)) for row in forward_rows] != list(range(1, STEPS + 1)):
            raise CampaignError("T same-forward component/tau audit is incomplete")
        required_fields = {"native_loss", "physical_delta_loss", "total_loss", "tau", "valid_transitions", "saturated_xyz"}
        for row in forward_rows:
            if not required_fields.issubset(row):
                raise CampaignError("T same-forward audit is missing a required component")
            numeric = [row["native_loss"], row["physical_delta_loss"], row["total_loss"], *row["tau"]]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
                raise CampaignError("T same-forward audit contains nonfinite values")
            if len(row["tau"]) != 1 or not 0.0 <= float(row["tau"][0]) <= 1.0:
                raise CampaignError("T same-forward audit tau shape/range is invalid")
            if not math.isclose(float(row["total_loss"]), float(row["native_loss"]) + 0.1 * float(row["physical_delta_loss"]), rel_tol=1e-5, abs_tol=1e-7):
                raise CampaignError("T same-forward loss components do not sum to the prescribed objective")
            if not 0 < int(row["valid_transitions"]) <= 39 or not 0 <= int(row["saturated_xyz"]) <= 120:
                raise CampaignError("T same-forward audit transition/saturation count is invalid")
        same_forward_audit = {
            "status": "COMPLETE", "rows": len(forward_rows),
            "sha256": sha256_file(forward_log),
            "first_forward": forward_rows[0], "last_forward": forward_rows[-1],
            "native_loss_min": min(float(row["native_loss"]) for row in forward_rows),
            "native_loss_max": max(float(row["native_loss"]) for row in forward_rows),
            "physical_delta_loss_min": min(float(row["physical_delta_loss"]) for row in forward_rows),
            "physical_delta_loss_max": max(float(row["physical_delta_loss"]) for row in forward_rows),
            "valid_transitions_min": min(int(row["valid_transitions"]) for row in forward_rows),
            "valid_transitions_max": max(int(row["valid_transitions"]) for row in forward_rows),
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
    if arm == "T":
        integrity["same_forward_audit_complete"] = same_forward_audit is not None and same_forward_audit["rows"] == STEPS
    projector_reconciliation_required = (
        arm == "T" and not integrity["projector_updated"]
        and all(value for key, value in integrity.items() if key != "projector_updated")
    )
    result = {
        "schema": "PI0246_ARM_TRAINING_MANIFEST_V1",
        "status": (
            "TRAINING_COMPLETE_PROJECTOR_RECONCILIATION_REQUIRED" if projector_reconciliation_required
            else "PASS_TRAINING_INTEGRITY" if all(integrity.values())
            else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT"
        ),
        "arm": arm, "family": card["family"], "run_card_sha256": sha256_file(args.run_card),
        "p1_closure_sha256": sha256_file(Path(card["p1_closure_path"])), "p1_cells": len(p1["cells"]),
        "elapsed_s": time.time() - started, "global_step": STEPS, "sample_consumption": len(consumption),
        "pbs_job_id": os.environ["PBS_JOBID"],
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
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--seal-t-runtime", action="store_true")
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
    args = parser.parse_args()
    if args.prepare:
        required = (args.r0_card, args.r0_root, args.r0_model_root, args.p0_effective, args.p1_root, args.campaign_code_root, args.output_root)
        if any(value is None for value in required):
            parser.error("--prepare requires R0/P0/P1 identities and output root")
        prepare(args)
    elif args.train:
        if args.run_card is None:
            parser.error("--train requires --run-card")
        train(args)
    else:
        required = (args.run_card, args.runtime_root, args.wrapper)
        if any(value is None for value in required):
            parser.error("--seal-t-runtime requires --run-card, --runtime-root, and --wrapper")
        print(json.dumps(seal_t_runtime(args.run_card, args.runtime_root, args.wrapper), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
