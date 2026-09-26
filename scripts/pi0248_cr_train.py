#!/usr/bin/env python3
"""Seal and run the authorized PI0248 paired C/R continuation on Hopper."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import shutil
import sys
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


AUTH = "PI0248_M64_POSE_REPAIR_GO"
CANONICAL = 5847759659
POINTER = 5847760965
SEED = 22301
STEPS = 2000
CHECKPOINTS = (1000, 2000)
REVISION = "093c450fd49052c8fd3285bf38111a30a1f59009"
TREE_SHA256 = "1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776"

life: Any = None


def _load_life(repo_code: Path, driver_root: Path) -> Any:
    sys.path[:0] = [str(driver_root), str(repo_code / "scripts")]
    return importlib.import_module("pi0247_lu_train")


def _legacy_code_map(paths: dict[str, Any]) -> dict[str, Path]:
    repo, shared, runtime = (Path(paths[key]) for key in ("repo_code_root", "code_root", "training_runtime_root"))
    return {
        "pi0247_lu_train.py": repo / "scripts/pi0247_lu_train.py",
        "pi0247_u_rollout_overlay.py": repo / "scripts/pi0247_u_rollout_overlay.py",
        "pi0228_train_calib_checkpoint_selection.py": repo / "scripts/pi0228_train_calib_checkpoint_selection.py",
        "pi0247_m64_train_h200.pbs": repo / "hpc/pbs/pi0247_m64_train_h200.pbs",
        "pi0246_pt_combined_overlay.py": runtime / "pi0246_pt_combined_overlay.py",
        "pi0246_physical_position.py": runtime / "pi0246_physical_position.py",
        "pi0245_physical_delta.py": runtime / "pi0245_physical_delta.py",
        "pi0246_gate_pass_train.py": runtime / "pi0246_gate_pass_train.py",
        "pi0241_fixed16_learning.py": shared / "scripts/pi0241_fixed16_learning.py",
        "pi0228_h200_large_adapt_v1.py": shared / "scripts/pi0228_h200_large_adapt_v1.py",
        "stage40_b0_model_control.py": shared / "scripts/stage40_b0_model_control.py",
        "pi0243_checkpoint_persistence.py": shared / "scripts/pi0243_checkpoint_persistence.py",
        "pi0243_hf_checkpoint_sidecar.py": shared / "scripts/pi0243_hf_checkpoint_sidecar.py",
        "pi0243_hf_archive_tree.py": shared / "scripts/pi0243_hf_archive_tree.py",
        "pi0243_storage_retirement.py": shared / "scripts/pi0243_storage_retirement.py",
        "pi0243_hf_write_probe.py": shared / "scripts/pi0243_hf_write_probe.py",
        "pi0241_reconcile_update_integrity.py": shared / "scripts/pi0241_reconcile_update_integrity.py",
    }


def _schedule(cohort_path: Path, output: Path) -> dict[str, Any]:
    import numpy as np

    cohort_doc = life.read_json(cohort_path)
    rows = cohort_doc["rows"]
    if len(rows) != 64:
        raise life.LUError("PI0248 schedule requires the sealed 64-source cohort")
    ordered = sorted(enumerate(rows), key=lambda pair: (
        str(pair[1]["target_family"]), int(pair[1]["phase"]), str(pair[1]["source_episode_id"])))
    rng = np.random.Generator(np.random.PCG64(SEED))
    picks: list[int] = []
    while len(picks) < STEPS:
        picks.extend(ordered[int(index)][0] for index in rng.permutation(len(ordered)))
    schedule = []
    for step, index in enumerate(picks[:STEPS], start=1):
        row = rows[index]
        schedule.append({
            "optimizer_step": step, "fixed_index": index,
            "episode_index": int(row["episode_index"]), "phase": int(row["phase"]),
            "target_family": str(row["target_family"]),
            "source_episode_id": str(row["source_episode_id"]),
        })
    counts = Counter(row["fixed_index"] for row in schedule)
    if len(counts) != 64 or sorted(counts.values()) != [31] * 48 + [32] * 16:
        raise life.LUError("PI0248 exposure table does not give 48 windows 31 and 16 windows 32 updates")
    life.write_json(output, {
        "schema": "PI0248_CR_PAIRED_EXPOSURE_SCHEDULE_V1", "seed": SEED,
        "cohort_sha256": life.sha256_file(cohort_path), "rows": schedule,
    })
    return {"path": str(output), "sha256": life.sha256_file(output), "count": STEPS}


def _check_rotation_binding(rotation: dict[str, Any], processor_root: Path, dataset_root: Path,
                            embodiment_sha256: str) -> None:
    ct = rotation["processor_identity"]
    expected = {
        "processor_config.json": ct["processor_config_sha256"],
        "statistics.json": ct["T_card_base_statistics_sha256"],
        "embodiment_id.json": embodiment_sha256,
    }
    for name, digest in expected.items():
        source = processor_root / name
        if not source.is_file() or life.sha256_file(source) != digest:
            raise life.LUError(f"frozen C_T processor identity mismatch: {source}")
    for name, key in (("stats.json", "NEW_EMBODIMENT_stats_sha256"),
                      ("relative_stats.json", "relative_stats_sha256")):
        source = dataset_root / "meta" / name
        if not source.is_file() or life.sha256_file(source) != ct[key]:
            raise life.LUError(f"frozen C_T TRAIN statistics mismatch: {source}")


def prepare(args: argparse.Namespace) -> None:
    parent_path = args.parent_card.resolve()
    parent = life.read_json(parent_path)
    if (parent.get("schema") != "PI0247_M64_RUN_CARD_V1" or parent.get("arm") != "U"
            or parent.get("selected_checkpoint") != "NONE"):
        raise life.LUError("PI0248 must bind the sealed M64 U run card with no selected checkpoint")
    parent_seal = life.read_json(parent_path.with_name("run_card.seal.json"))
    if parent_seal.get("sha256") != life.sha256_file(parent_path):
        raise life.LUError("M64 parent run-card seal mismatch")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise life.LUError("refusing to reuse an existing PI0248 output root")
    if args.execution_root.exists() or args.execution_root.is_symlink():
        raise life.LUError("refusing to reuse an existing PI0248 execution-code root")

    m64_root = args.m64_root.resolve()
    source_root = m64_root / "recovery_attempts/633819.hopper-m-02/training"
    receipt_path = m64_root / "recovery_attempts/633819.hopper-m-02/durable_receipts/checkpoint-5000.durable_receipt.json"
    readback_path = m64_root / "recovery_attempts/633819.hopper-m-02/hf_readbacks/checkpoint-5000.json"
    receipt, readback = life.read_json(receipt_path), life.read_json(readback_path)
    if (receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("checkpoint_step") != 5000
            or receipt.get("hf_manifest_commit_id") != REVISION
            or receipt.get("source_record", {}).get("inventory", {}).get("tree_sha256") != TREE_SHA256
            or readback.get("status") != "PASS_PT_FULL_HF_READBACK"
            or readback.get("revision") != REVISION or readback.get("source_tree_sha256") != TREE_SHA256):
        raise life.LUError("PI0248 M64-final5000 durable revision/readback identity mismatch")
    entries = {row["path"]: row for row in receipt["source_record"]["inventory"]["entries"]}
    for name in ("config.json", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        source = source_root / name
        entry = entries.get(name)
        if (entry is None or not source.is_file() or source.stat().st_size != int(entry["bytes"])
                or life.sha256_file(source) != entry["sha256"]):
            raise life.LUError(f"M64-final5000 model view differs from the durable checkpoint: {name}")
    if readback.get("file_count") != 9 or readback.get("source_record_sha256") != life.canonical_sha256(receipt["source_record"]):
        raise life.LUError("M64-final5000 pinned HF readback is incomplete")

    rotation = life.read_json(args.rotation_check)
    if rotation.get("status") != "PASS_CPU_DECODER_AND_GRADIENT":
        raise life.LUError("PI0248 rotation decoder/gradient CPU check is not passing")
    output = args.output_root.resolve()
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    evidence = output / "evidence"
    evidence.mkdir()
    rotation_copy = evidence / "pi0248_rotation_cpu_check.json"
    shutil.copy2(args.rotation_check, rotation_copy)
    report_copy = evidence / "PI0248_G0_MECHANISM_REPORT.md"
    shutil.copy2(args.g0_report, report_copy)

    view = output / "initializer_M64_final5000"
    view.mkdir()
    for name in ("config.json", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (view / name).symlink_to(source_root / name)
    parent_embodiment = next(item["sha256"] for item in parent["initializer"]["model_files"]
                             if item["path"] == "embodiment_id.json")
    _check_rotation_binding(rotation, source_root / "processor", Path(parent["paths"]["train_dataset"]),
                            parent_embodiment)
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (view / name).symlink_to(source_root / "processor" / name)
    view_receipt = output / "initializer_view_receipt.json"
    life.write_json(view_receipt, {
        "status": "PASS_PI0248_M64_FINAL5000_VIEW", "revision": REVISION,
        "checkpoint_tree_sha256": TREE_SHA256, "model_files": life._model_files(view),
        "processor_identity": rotation["processor_identity"],
        "source_durable_receipt_sha256": life.sha256_file(receipt_path),
    })

    schedule_ref = _schedule(Path(parent["cohort"]["path"]), output / "paired_exposure_schedule.json")
    execution_root = args.execution_root.resolve()
    execution_root.mkdir(parents=True)
    for staged in (Path(__file__).resolve(), Path(__file__).with_name("pi0248_rotation_cpu.py").resolve(),
                   args.pbs_script.resolve()):
        shutil.copy2(staged, execution_root / staged.name)
    code_map = {name: life.sha256_file(path) for name, path in _legacy_code_map(parent["paths"]).items()}
    custom_files = {
        "pi0248_cr_train.py": execution_root / "pi0248_cr_train.py",
        "pi0248_rotation_cpu.py": execution_root / "pi0248_rotation_cpu.py",
        "pi0248_cr_train_h200.pbs": execution_root / "pi0248_cr_train_h200.pbs",
    }
    for path in custom_files.values():
        if not path.is_file():
            raise life.LUError(f"PI0248 execution source is missing: {path}")
    code_root = Path(parent["paths"]["repo_code_root"])
    base_paths = dict(parent["paths"])
    base_paths.update({
        "output_base": str(output), "base_model_path": str(view),
        "repo_code_root": str(code_root), "execution_driver_root": str(execution_root),
        "execution_driver": str(custom_files["pi0248_cr_train.py"]),
        "pbs_script": str(custom_files["pi0248_cr_train_h200.pbs"]),
        "initializer_view_receipt_path": str(view_receipt),
    })
    cohort_ref = dict(parent["cohort"])
    family_loss = {
        "C": ("PI0248_C_U_CONTINUATION_M64_2K_V1",
              "L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4)"),
        "R": ("PI0248_R_ROTATION_ENDPOINT_M64_2K_V1",
              "L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4) + 0.1*ell_rot(a_free_S4)"),
    }
    run_cards = {}
    for arm, (family, loss_name) in family_loss.items():
        arm_root = output / arm
        arm_root.mkdir()
        paths = {**base_paths, "output_root": str(arm_root)}
        archive_prefix = f"arms/PI0248/m64-pose-repair-20260927/{arm}"
        code = dict(code_map)
        for name, path in custom_files.items():
            code[f"custom:{name}"] = life.sha256_file(path)
        card = copy.deepcopy(parent)
        card.pop("qualified_recipe", None)
        card.pop("route", None)
        card.pop("nonselection_parity_budget", None)
        card.update({
            "schema": "PI0248_CR_RUN_CARD_V1", "status": "SEALED_READY_NOT_STARTED",
            "authorization": AUTH, "execution_authorization": AUTH,
            "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
            "execution_update_commit": args.execution_commit,
            "arm": arm, "family": family, "loss": loss_name, "seed": SEED,
            "selected_checkpoint": "NONE", "d1_runtime_id": "S4",
            "nonselection_parity_budget": {
                "campaign_limit": 8, "prior_rollouts": 4,
                "this_campaign_limit": 4, "planned_per_arm": 2,
                "total_if_both_arms_complete": 8,
            },
            "stage": {"name": "M64_REPAIR", "steps": STEPS,
                      "checkpoint_steps": list(CHECKPOINTS), "source_revision": REVISION},
            "support": {**parent.get("support", {}), "count": 64},
            "cohort": cohort_ref, "schedule": schedule_ref,
            "initializer": {
                "path": str(view), "family": "PI0247_M64_U_ROLLOUT_M16_2K_V1",
                "durable_receipt_path": str(receipt_path),
                "durable_receipt_sha256": life.sha256_file(receipt_path),
                "durable_revision": REVISION, "screen_revision": REVISION,
                "screen_manifest_sha256": life.sha256_file(readback_path),
                "full_checkpoint_tree_sha256": TREE_SHA256,
                "full_checkpoint_bytes": int(receipt["source_record"]["inventory"]["total_bytes"]),
                "model_files": life._model_files(view),
            },
            "loss_contract": {
                "objective": loss_name, "generation": "complete pure-noise label-free S4 rollout before physical losses",
                "horizon": 40, "physical_dt_s": 0.05, "euler_steps": 4,
                "backpropagate_through_all_euler_steps": True,
                "expert_prefix_input": False, "active_action_smoothing": False,
            },
            "loss_coefficients": {"ell_pos": 0.1, "ell_delta": 0.1, **({"ell_rot": 0.1} if arm == "R" else {})},
            "optimizer": {
                "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS),
                "learning_rate": 1e-5, "weight_decay": 1e-5,
                "betas": [0.9, 0.999], "epsilon": 1e-8, "max_grad_norm": 1.0,
                "batch_size": 1, "gradient_accumulation_steps": 1, "workers": 0,
                "scheduler": "cosine", "warmup_steps": 10,
                "optimizer_state": "fresh AdamW; no resume from M64",
            },
            "paths": paths,
            "archive": {**parent["archive"], "path_prefix": archive_prefix,
                        "checkpoint_steps": list(CHECKPOINTS), "full_trainer_checkpoints": True},
            "hf_preflight": {"status": "PENDING_PER_ARM_PUBLIC_PREFIX_PREFLIGHT"},
            "source": {
                **parent["source"], "execution_code_commit": args.execution_commit,
                "execution_code_sha256": code,
                "custom_execution_code_sha256": {name: life.sha256_file(path) for name, path in custom_files.items()},
                "initializer_view_receipt_path": str(view_receipt),
                "initializer_view_receipt_sha256": life.sha256_file(view_receipt),
                "m64_parent_run_card_path": str(parent_path),
                "m64_parent_run_card_sha256": life.sha256_file(parent_path),
                "m64_readback_path": str(readback_path),
                "m64_readback_sha256": life.sha256_file(readback_path),
                "rotation_check_path": str(rotation_copy),
                "rotation_check_sha256": life.sha256_file(rotation_copy),
                "g0_report_path": str(report_copy),
                "g0_report_sha256": life.sha256_file(report_copy),
            },
            "paired_arm_root": str(output / ("R" if arm == "C" else "C")),
        })
        card["paths"]["p_run_card_path"] = parent["paths"]["p_run_card_path"]
        card["initializer"]["model_files"] = life._model_files(view)
        life.STEPS, life.CHECKPOINTS = STEPS, CHECKPOINTS
        _validate_initializer(card)
        life._verify_runtime_bindings(paths, card["source"]["runtime_bindings"])
        _verify_code(card, code_root, Path(paths["code_root"]), Path(paths["training_runtime_root"]))
        fixed = life.load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", f"pi0248_{arm}_fixed")
        core = life.load_module(Path(paths["training_runtime_root"]) / "pi0246_gate_pass_train.py", f"pi0248_{arm}_core")
        fixed.MAX_STEPS, fixed.CHECKPOINT_STEPS = STEPS, CHECKPOINTS
        core.STEPS, core.CHECKPOINTS = STEPS, CHECKPOINTS
        _, _, resolved = core.build_runtime_config(paths, arm_root / "training", fixed)
        parent_2000_config = life.read_json(Path(parent["source"]["parent_resolved_config_path"]))
        life._check_summary(resolved, parent_2000_config, str(view))
        card["resolved_config"] = resolved
        capacity = life._validate_storage(card)
        run_card_path = arm_root / "run_card.json"
        life.write_json(run_card_path, card)
        card_sha = life.sha256_file(run_card_path)
        life.write_json(arm_root / "run_card.seal.json", {"sha256": card_sha})
        life.write_json(arm_root / "preparation.json", {
            "status": "READY_NOT_STARTED", "selected_checkpoint": "NONE",
            "run_card_sha256": card_sha, "capacity": capacity,
            "resolved_config": resolved, "schedule_sha256": schedule_ref["sha256"],
        })
        run_cards[arm] = {"path": str(run_card_path), "sha256": card_sha}
    life.write_json(output / "paired_preparation.json", {
        "status": "READY_NOT_STARTED", "execution_commit": args.execution_commit,
        "m64_revision": REVISION, "m64_tree_sha256": TREE_SHA256,
        "schedule": schedule_ref, "cards": run_cards,
        "selected_checkpoint": "NONE",
    })
    print(json.dumps(life.read_json(output / "paired_preparation.json"), indent=2, sort_keys=True))


def _validate_initializer(card: dict[str, Any], original: Any = None) -> None:
    view_path = Path(card["source"]["initializer_view_receipt_path"])
    view = life.read_json(view_path)
    if (life.sha256_file(view_path) != card["source"]["initializer_view_receipt_sha256"]
            or view.get("status") != "PASS_PI0248_M64_FINAL5000_VIEW"
            or view.get("revision") != REVISION or view.get("checkpoint_tree_sha256") != TREE_SHA256):
        raise life.LUError("PI0248 M64-final5000 initializer view receipt mismatch")
    original = original or life._validate_initializer_runtime
    schema = card["schema"]
    card["schema"] = "PI0247_M64_RUN_CARD_V1"
    try:
        original(card)
    finally:
        card["schema"] = schema


def _verify_code(card: dict[str, Any], repo_code: Path, shared_root: Path, training_root: Path,
                 original: Any = None) -> None:
    original = original or life._verify_code
    schema = card["schema"]
    card["schema"] = "PI0247_M64_RUN_CARD_V1"
    try:
        original(card, repo_code, shared_root, training_root)
    finally:
        card["schema"] = schema


def _verify_card(path: Path, arm: str) -> dict[str, Any]:
    card = life.read_json(path)
    seal = life.read_json(path.with_name("run_card.seal.json"))
    expected_family = {
        "C": "PI0248_C_U_CONTINUATION_M64_2K_V1",
        "R": "PI0248_R_ROTATION_ENDPOINT_M64_2K_V1",
    }.get(arm)
    expected_loss = {
        "C": "L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4)",
        "R": "L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4) + 0.1*ell_rot(a_free_S4)",
    }.get(arm)
    if (card.get("schema") != "PI0248_CR_RUN_CARD_V1" or card.get("status") != "SEALED_READY_NOT_STARTED"
            or card.get("authorization") != AUTH or card.get("execution_authorization") != AUTH
            or card.get("canonical_issue_comment") != CANONICAL or card.get("pointer_pr_comment") != POINTER
            or card.get("arm") != arm or arm not in ("C", "R")
            or card.get("family") != expected_family or card.get("loss") != expected_loss
            or card.get("seed") != SEED or card.get("selected_checkpoint") != "NONE"
            or card.get("stage") != {"name": "M64_REPAIR", "steps": STEPS,
                                     "checkpoint_steps": list(CHECKPOINTS), "source_revision": REVISION}
            or card.get("schedule", {}).get("count") != STEPS
            or card.get("loss_coefficients", {}).get("ell_pos") != 0.1
            or card.get("loss_coefficients", {}).get("ell_delta") != 0.1
            or (arm == "R") != (card.get("loss_coefficients", {}).get("ell_rot") == 0.1)
            or seal.get("sha256") != life.sha256_file(path)):
        raise life.LUError("PI0248 run card/seal does not match the authorized paired arm")
    source = card["source"]
    for name, expected in source["custom_execution_code_sha256"].items():
        file_path = (Path(card["paths"]["execution_driver_root"]) / name
                     if not name.endswith(".pbs") else Path(card["paths"]["pbs_script"]))
        if life.sha256_file(file_path) != expected:
            raise life.LUError(f"PI0248 execution source changed: {name}")
    if (life.sha256_file(Path(source["m64_parent_run_card_path"])) != source["m64_parent_run_card_sha256"]
            or life.sha256_file(Path(source["m64_readback_path"])) != source["m64_readback_sha256"]
            or life.sha256_file(Path(source["rotation_check_path"])) != source["rotation_check_sha256"]
            or life.sha256_file(Path(source["g0_report_path"])) != source["g0_report_sha256"]):
        raise life.LUError("PI0248 source evidence changed after card sealing")
    lineage = source.get("resume_prefix_lineage")
    if lineage:
        for path_key, hash_key in (("card_path", "card_sha256"), ("preflight_path", "preflight_sha256"),
                                   ("durable_receipt_path", "durable_receipt_sha256"),
                                   ("readback_path", "readback_sha256")):
            if life.sha256_file(Path(lineage[path_key])) != lineage[hash_key]:
                raise life.LUError(f"PI0248 resume-prefix lineage changed: {path_key}")
    life.STEPS, life.CHECKPOINTS = STEPS, CHECKPOINTS
    return card


def _science_card(card: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(card)
    result.pop("execution_update_commit", None)
    for key in ("execution_driver_root", "execution_driver", "pbs_script"):
        result["paths"].pop(key, None)
    for key in ("execution_code_commit", "execution_code_sha256", "custom_execution_code_sha256",
                "resume_prefix_lineage"):
        result["source"].pop(key, None)
    return result


def _resume_prefix_preflight(card: dict[str, Any], card_path: Path, *, create: bool) -> dict[str, Any]:
    lineage = card["source"].get("resume_prefix_lineage")
    if card["arm"] != "C" or not lineage:
        raise life.LUError("PI0248 exact C resume lacks sealed prefix lineage")
    old_card = life._verify_card(Path(lineage["card_path"]), "C")
    if _science_card(old_card) != _science_card(card):
        raise life.LUError("PI0248 exact-resume card changed the sealed C science contract")
    old_card_sha = life.sha256_file(Path(lineage["card_path"]))
    prior = life.read_json(Path(lineage["preflight_path"]))
    if (prior.get("schema") != "PI0247_LU_HF_ONLINE_PREFLIGHT_V1" or prior.get("status") != "PASS"
            or prior.get("arm") != "C" or prior.get("run_card_sha256") != old_card_sha
            or prior.get("repo_id") != card["archive"]["repo_id"]
            or prior.get("target_prefix") != card["archive"]["path_prefix"]
            or prior.get("target_prefix_empty") is not True or prior.get("repo_public") is not True
            or prior.get("identity_verified") is not True
            or prior.get("write_probe", {}).get("readback") != "PASS"):
        raise life.LUError("PI0248 prior empty-prefix preflight is not bound to its original C card")

    output = Path(card["paths"]["output_root"])
    receipt_path = Path(lineage["durable_receipt_path"])
    readback_path = Path(lineage["readback_path"])
    receipt, readback = life.read_json(receipt_path), life.read_json(readback_path)
    record = receipt.get("source_record")
    core = life.load_module(Path(card["paths"]["training_runtime_root"]) / "pi0246_gate_pass_train.py",
                            "pi0248_resume_prefix_core")
    prefix, repo_id, family = card["archive"]["path_prefix"], card["archive"]["repo_id"], card["family"]
    core.validate_pt_archive_receipt(receipt, record, repo_id=repo_id, prefix=prefix, family=family, step=1000)
    expected = {f"{prefix}/checkpoint-1000/{row['path']}": row for row in record["inventory"]["entries"]}
    observed = {str(row.get("path")): row for row in readback.get("files", [])}
    revision = receipt.get("hf_manifest_commit_id")
    if (receipt.get("status") != "DURABLE_VERIFIED"
            or readback.get("schema") != "PI0247_LU_FULL_HF_READBACK_V1"
            or readback.get("status") != "PASS_PT_FULL_HF_READBACK"
            or readback.get("pi0247_status") != "PASS_PI0247_LU_FULL_HF_READBACK"
            or readback.get("repo_id") != repo_id or readback.get("path_prefix") != prefix
            or readback.get("checkpoint_step") != 1000 or readback.get("revision") != revision
            or readback.get("source_tree_sha256") != record["inventory"]["tree_sha256"]
            or readback.get("source_record_sha256") != core.canonical_sha256(record)
            or set(observed) != set(expected)):
        raise life.LUError("PI0248 step-1000 durable receipt/readback do not close the prior C prefix")
    for path, entry in expected.items():
        if (int(observed[path].get("bytes", -1)) != int(entry["bytes"])
                or observed[path].get("sha256") != entry["sha256"]):
            raise life.LUError(f"PI0248 step-1000 readback differs at {path}")

    token_path = Path(card["archive"]["token_file"])
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise life.LUError("PI0248 resume-prefix check lacks the configured HF identity")
    offline = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.whoami()
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        siblings = {str(getattr(row, "rfilename", "")): row for row in (getattr(info, "siblings", None) or [])}
        current = {path: row for path, row in siblings.items()
                   if path == prefix or path.startswith(prefix.rstrip("/") + "/")}
    finally:
        for name, value in offline.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if getattr(info, "private", None) is not False or getattr(info, "sha", None) != revision or set(current) != set(expected):
        raise life.LUError("PI0248 current public-HF prefix is not exactly the verified step-1000 checkpoint")
    for path, entry in expected.items():
        if getattr(current[path], "size", None) != int(entry["bytes"]):
            raise life.LUError(f"PI0248 current public-HF prefix size differs at {path}")

    result = {
        "schema": "PI0248_HF_RESUME_PREFIX_PREFLIGHT_V1", "status": "PASS_PI0248_HF_RESUME_PREFIX",
        "arm": "C", "run_card_sha256": life.sha256_file(card_path), "resume_after_step": 1000,
        "repo_id": repo_id, "target_prefix": prefix, "target_prefix_empty": False,
        "prefix_was_empty_before_first_publish": True, "current_prefix_checkpoint_steps": [1000],
        "repo_public": True, "identity_verified": True, "current_repo_revision": revision,
        "source_card_sha256": old_card_sha, "source_preflight_sha256": lineage["preflight_sha256"],
        "durable_receipt_sha256": life.sha256_file(receipt_path),
        "full_hf_readback_sha256": life.sha256_file(readback_path),
        "checkpoint_tree_sha256": record["inventory"]["tree_sha256"],
        "checkpoint_file_count": len(expected), "observed_prefix_files": sorted(current),
    }
    result_path = output / f"hf_resume_preflight_{result['run_card_sha256']}.json"
    if create:
        if result_path.exists() and life.read_json(result_path) != result:
            raise life.LUError("PI0248 resume-prefix preflight receipt conflicts with existing evidence")
        if not result_path.exists():
            life.write_json(result_path, result)
    else:
        saved = life.read_json(result_path)
        if any(saved.get(key) != value for key, value in result.items()):
            raise life.LUError("PI0248 sealed resume-prefix preflight receipt changed before training")
    return result


def _patch_overlay(module: Any, card: dict[str, Any], resume_after: int) -> None:
    if card["arm"] == "C":
        if not resume_after:
            return
        output = Path(card["paths"]["output_root"])
        original_install = module.install

        def install(head_cls: type, **kwargs: Any) -> dict[str, Any]:
            audit_path = Path(kwargs["audit_path"])
            source = output / "training" / audit_path.name
            if not audit_path.exists():
                rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if [int(row["forward_index"]) for row in rows] != list(range(1, resume_after + 1)):
                    raise life.LUError("C exact resume lacks the complete S4/objective audit prefix")
                shutil.copy2(source, audit_path)
            return original_install(head_cls, **kwargs)

        module.install = install
        return
    if card["arm"] != "R":
        return
    import numpy as np
    import torch
    import pi0248_rotation_cpu as rotation

    output = Path(card["paths"]["output_root"])
    state: dict[str, Any] = {"rotation_tensor": None, "forward_index": resume_after}
    get_processor: Any = None
    check = life.read_json(Path(card["source"]["rotation_check_path"]))
    original_physical = module._physical_losses
    original_append = module._append_json
    original_install = module.install

    def physical(generated: Any, expert: Any, mask: Any, state_eef: Any,
                 scale: Any, offset: Any) -> Any:
        position, delta, details = original_physical(generated, expert, mask, state_eef, scale, offset)
        processor = get_processor() if get_processor is not None else None
        if processor is None:
            raise life.LUError("R rotation loss cannot access the frozen C_T TRAIN processor")
        params = processor.state_action_processor.norm_params[card["embodiment_tag"]]["action"]["eef_9d"]
        minimum_raw, maximum_raw = params["min"], params["max"]
        if hasattr(minimum_raw, "detach"):
            minimum_raw = minimum_raw.detach().cpu().numpy()
        if hasattr(maximum_raw, "detach"):
            maximum_raw = maximum_raw.detach().cpu().numpy()
        minimum64 = np.asarray(minimum_raw, dtype=np.float64)
        maximum64 = np.asarray(maximum_raw, dtype=np.float64)
        stats = processor
        pinned = check["pinned_statistics"]
        min_sha = rotation._hash_array(minimum64[:, 3:9])
        max_sha = rotation._hash_array(maximum64[:, 3:9])
        if (minimum64.shape != (40, 9) or maximum64.shape != (40, 9)
                or min_sha != pinned["rotation_minmax_sha256"]["minimum"]
                or max_sha != pinned["rotation_minmax_sha256"]["maximum"]
                or bool(stats.use_percentiles) != pinned["use_percentiles"]
                or bool(stats.clip_outliers) != pinned["clip_outliers"]
                or bool(stats.use_relative_action) != pinned["use_relative_action"]):
            raise life.LUError("R actual TRAIN processor differs from the frozen C_T rotation decoder")
        minimum = torch.from_numpy(minimum64[:, 3:9].astype(np.float32))
        maximum = torch.from_numpy(maximum64[:, 3:9].astype(np.float32))
        value, info = rotation.rotation_chordal_loss(
            generated, expert, mask, state_eef, minimum, maximum, theta_ref=0.05)
        state["rotation_tensor"] = value
        state["rotation_info"] = info
        state["rotation_bounds_sha256"] = {"minimum": min_sha, "maximum": max_sha}
        return position, delta, {**details, **info,
                                 "physical_rotation_loss": float(value.detach().float().item())}

    def append(path: Path, row: dict[str, Any], call_index: int) -> None:
        if "physical_rotation_loss" in row:
            if call_index != state["forward_index"] + 1:
                raise life.LUError("R objective and S4 audit update counters diverged")
            rotation_value = float(row["physical_rotation_loss"])
            row = {**row, "lambda_rotation": 0.1, "theta_ref_rad": 0.05,
                   "total_loss_with_rotation": float(row["total_loss"]) + 0.1 * rotation_value,
                   "rotation_bounds_sha256": state["rotation_bounds_sha256"]}
        original_append(path, row, call_index)

    def install(head_cls: type, **kwargs: Any) -> dict[str, Any]:
        nonlocal get_processor
        get_processor = kwargs["processor_getter"]
        if resume_after:
            audit_path = Path(kwargs["audit_path"])
            source = output / "training" / audit_path.name
            if not audit_path.exists():
                rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != resume_after:
                    raise life.LUError("R exact resume lacks the complete S4/objective audit prefix")
                shutil.copy2(source, audit_path)
        module._physical_losses = physical
        module._append_json = append
        audit = original_install(head_cls, **kwargs)
        native_forward = head_cls.forward

        def forward(self: Any, *args: Any, **kw: Any) -> Any:
            state["rotation_tensor"] = None
            result = native_forward(self, *args, **kw)
            rotation_tensor = state["rotation_tensor"]
            if rotation_tensor is None or not isinstance(result, dict) or "loss" not in result:
                raise life.LUError("R full S4 forward did not produce its differentiable rotation endpoint")
            result["loss"] = rotation.add_rotation_term(result["loss"], rotation_tensor, 0.1)
            state["forward_index"] += 1
            return result

        head_cls.forward = forward
        return audit

    module.install = install


def _install_adapters(card_path: Path, arm: str, resume_after: int) -> tuple[dict[str, Any], Any]:
    global life
    preliminary = json.loads(card_path.read_text(encoding="utf-8"))
    repo_code = Path(preliminary["paths"]["repo_code_root"])
    driver_root = Path(preliminary["paths"]["execution_driver_root"])
    life = _load_life(repo_code, driver_root)
    card = _verify_card(card_path, arm)
    life._verify_card = _verify_card
    original_validate_initializer = life._validate_initializer_runtime
    original_verify_code = life._verify_code
    life._validate_initializer_runtime = lambda card: _validate_initializer(card, original_validate_initializer)
    life._verify_code = lambda card, repo_code, shared_root, training_root: _verify_code(
        card, repo_code, shared_root, training_root, original_verify_code
    )
    original_hf_verify = life._verify_hf_preflight_receipt
    original_hf_preflight = life.hf_preflight

    def verify_hf_preflight(card: dict[str, Any], card_path: Path) -> dict[str, Any]:
        if card["arm"] == "C" and resume_after:
            return _resume_prefix_preflight(card, card_path, create=False)
        return original_hf_verify(card, card_path)

    def hf_preflight(args: argparse.Namespace) -> None:
        if args.resume_after_step != resume_after:
            raise life.LUError("PI0248 HF preflight cursor differs from the installed run mode")
        if args.arm == "C" and resume_after:
            result = _resume_prefix_preflight(life._verify_card(args.run_card, args.arm), args.run_card, create=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        receipt_path = Path(life._verify_card(args.run_card, args.arm)["paths"]["output_root"]) / "hf_preflight_receipt.json"
        if receipt_path.is_file():
            old = life.read_json(receipt_path)
            if old.get("run_card_sha256") != life.sha256_file(args.run_card):
                if (old.get("schema") != "PI0247_LU_HF_ONLINE_PREFLIGHT_V1" or old.get("status") != "PASS"
                        or old.get("arm") != args.arm or old.get("target_prefix_empty") is not True):
                    raise life.LUError("refusing to replace nonempty or unverified prior HF preflight evidence")
                backup = receipt_path.with_name(f"hf_preflight_receipt.previous_{old['run_card_sha256']}.json")
                if backup.exists():
                    if life.sha256_file(backup) != life.sha256_file(receipt_path):
                        raise life.LUError("preserved prior HF preflight receipt conflicts with the current file")
                else:
                    shutil.copy2(receipt_path, backup)
                receipt_path.unlink()
        original_hf_preflight(args)

    life._verify_hf_preflight_receipt = verify_hf_preflight
    life.hf_preflight = hf_preflight
    original_load = life.load_module

    def load_module(path: Path, name: str) -> Any:
        module = original_load(path, name)
        if path.name == "pi0247_u_rollout_overlay.py":
            _patch_overlay(module, card, resume_after)
        return module

    life.load_module = load_module
    original_chain = life._install_u_chain_check

    def install_chain(audit: dict[str, Any], output: Path, *, mode_path: Path | None = None) -> None:
        if resume_after and card["arm"] in ("C", "R"):
            mode_path = mode_path or output / "u_mode_restoration_checks.jsonl"
            source = output / "u_mode_restoration_checks.jsonl"
            if not mode_path.exists():
                rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != resume_after:
                    raise life.LUError("paired-arm exact resume lacks the full mode-restoration prefix")
                shutil.copy2(source, mode_path)
        original_chain(audit, output, mode_path=mode_path)

    life._install_u_chain_check = install_chain
    original_archive_install = life._install_hf_rolling_archive

    def install_archive(card: dict[str, Any], persistence: Any, card_path: Path, train_core: Any) -> None:
        if card["arm"] == "C" and resume_after:
            output = Path(card["paths"]["output_root"])
            checkpoint = output / "training" / f"checkpoint-{resume_after}"
            receipt = life.read_json(output / "durable_receipts" / f"checkpoint-{resume_after}.durable_receipt.json")
            record = receipt["source_record"]
            if persistence.checkpoint_record(
                    checkpoint, expected_step=resume_after, expected_max_steps=STEPS, family=card["family"]) != record:
                raise life.LUError("restored step-1000 checkpoint differs from its durable archive record")
            helper_path = Path(card["paths"]["code_root"]) / "scripts/pi0241_reconcile_update_integrity.py"
            if life.sha256_file(helper_path) != card["source"]["execution_code_sha256"]["pi0241_reconcile_update_integrity.py"]:
                raise life.LUError("step-1000 projector reconciliation source changed")
            helper = life.load_module(helper_path, "pi0248_resume_projector_reconciliation")
            report = {
                "schema": "PI0247_LU_PROJECTOR_RECONCILIATION_V1",
                "status": "PASS_FULL_PROJECTOR_TENSOR_COMPARE", "arm": "C",
                "checkpoint_step": resume_after, "run_card_sha256": life.sha256_file(card_path),
                "initializer_tree_sha256": card["initializer"]["full_checkpoint_tree_sha256"],
                "checkpoint_tree_sha256": record["inventory"]["tree_sha256"],
                "reconciliation_script_sha256": life.sha256_file(helper_path),
                "comparison": helper.compare(
                    helper.projector_tensors(Path(card["initializer"]["path"])),
                    helper.projector_tensors(checkpoint),
                ),
            }
            projector_path = output / "projector_reconciliation" / f"checkpoint-{resume_after}.json"
            if projector_path.is_file():
                old = life.read_json(projector_path)
                if old != report and old.get("run_card_sha256") != report["run_card_sha256"]:
                    backup = projector_path.with_name(
                        f"checkpoint-{resume_after}.previous_{old.get('run_card_sha256', 'unbound')}.json")
                    if backup.exists():
                        if life.sha256_file(backup) != life.sha256_file(projector_path):
                            raise life.LUError("preserved projector reconciliation conflicts with prior evidence")
                    else:
                        shutil.copy2(projector_path, backup)
                    projector_path.unlink()
                elif old != report:
                    raise life.LUError("step-1000 projector reconciliation conflicts with this exact-resume card")
            if not projector_path.exists():
                life.write_json(projector_path, report)
        original_archive_install(card, persistence, card_path, train_core)

    life._install_hf_rolling_archive = install_archive
    original_finalize = life.finalize

    def finalize(args: argparse.Namespace) -> None:
        verified = _preverify_training(card, card_path)
        _normalize_resume_overlay(verified)
        original_finalize(args)
        manifest_path = Path(card["paths"]["output_root"]) / "training_manifest.json"
        manifest = life.read_json(manifest_path)
        manifest.update({
            "schema": "PI0248_CR_TRAINING_MANIFEST_V1", "stage": "M64_REPAIR",
            "authorization": AUTH, "execution_authorization": AUTH,
            "execution_code_commit": card["source"]["execution_code_commit"],
            "arm": card["arm"], "family": card["family"], "selected_checkpoint": "NONE",
            "objective": card["loss_contract"]["objective"],
        })
        life.write_json(manifest_path, manifest)
        _write_training_integrity(card, card_path, verified)

    life.finalize = finalize
    return card, life


def _route_archive_sidecar(module: Any, card: dict[str, Any]) -> None:
    original = module.subprocess
    legacy_runner = (Path(card["paths"]["repo_code_root"]) / "scripts/pi0247_lu_train.py").resolve()
    adapter_runner = Path(__file__).resolve()

    class ArchiveSubprocess:
        def __getattr__(self, name: str) -> Any:
            return getattr(original, name)

        def run(self, command: Any, *args: Any, **kwargs: Any) -> Any:
            if (isinstance(command, (list, tuple)) and len(command) > 2
                    and Path(command[1]).resolve() == legacy_runner
                    and "--archive-sidecar" in command):
                command = [command[0], str(adapter_runner), *command[2:]]
            return original.run(command, *args, **kwargs)

    module.subprocess = ArchiveSubprocess()


def _winning_training_root(card: dict[str, Any]) -> Path:
    output = Path(card["paths"]["output_root"])
    candidates = [output / "training"]
    recovery = output / "recovery_attempts"
    if recovery.is_dir():
        candidates.extend(sorted(recovery.glob("*/training")))
    complete = []
    for root in candidates:
        audit = root / "pi0241_update_audit.json"
        if audit.is_file():
            data = life.read_json(audit)
            if data.get("status") == "COMPLETE" and int(data.get("global_step", -1)) == STEPS:
                complete.append(root)
    if not complete:
        raise life.LUError("PI0248 read-only finalizer lacks a complete 2000-update audit")
    return max(complete, key=lambda root: (root / "pi0241_update_audit.json").stat().st_mtime_ns)


def _overlay_rows(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    full_copy = root / "pi0247_U_rollout_forward.complete.jsonl"
    path = full_copy if full_copy.is_file() else root / "pi0247_U_rollout_forward.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return path, rows


def _preverify_training(card: dict[str, Any], card_path: Path) -> dict[str, Any]:
    output = Path(card["paths"]["output_root"])
    root = _winning_training_root(card)
    audit_path, rows = _overlay_rows(root)
    if len(rows) != STEPS or [int(row["forward_index"]) for row in rows] != list(range(1, STEPS + 1)):
        raise life.LUError("PI0248 full S4 objective audit is incomplete")
    for row in rows:
        if (row.get("runtime_id") != "S4" or row.get("rollout_count") != 1
                or row.get("rollout_steps") != 4 or row.get("checkpointed_euler_steps") != 4
                or row.get("backward_recomputation_head_evaluations_expected") != 4
                or row.get("expert_action_in_condition") is not False
                or row.get("active_action_modified") is not False
                or len(row.get("euler_states", [])) != 4
                or len(row.get("euler_velocities", [])) != 4
                or not row.get("native_epsilon_sha256")
                or not row.get("stock_initial_noise", {}).get("sha256")):
            raise life.LUError("PI0248 S4/full-gradient/noise-identity evidence is incomplete")
        base = (float(row["native_loss"]) + 0.1 * float(row["physical_position_loss"])
                + 0.1 * float(row["physical_delta_loss"]))
        if not math.isclose(base, float(row["total_loss"]), rel_tol=1e-5, abs_tol=1e-7):
            raise life.LUError("PI0248 original U objective terms do not sum")
        if card["arm"] == "R":
            expected = base + 0.1 * float(row["physical_rotation_loss"])
            if (row.get("lambda_rotation") != 0.1 or row.get("theta_ref_rad") != 0.05
                    or not math.isclose(expected, float(row["total_loss_with_rotation"]), rel_tol=1e-5, abs_tol=1e-7)):
                raise life.LUError("R objective does not equal the sealed C objective plus its rotation term")
    parity_path = output / "pi0247_U_sampler_parity.json"
    parity = life.read_json(parity_path)
    if (parity.get("status") != "PASS_PI0247_U_S4_PARITY" or not parity.get("gate_decision_equal")
            or not parity.get("gate_issues_equal") or parity.get("expert_action_in_condition") is not False):
        raise life.LUError("PI0248 S4/native sampler or gate parity did not pass")
    chain_path = output / "u_first_update_chain_gradient.json"
    chain = life.read_json(chain_path)
    gradients = chain.get("state_gradient_norms", {})
    if (chain.get("status") != "PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN"
            or set(gradients) != {"1", "2", "3", "4"}
            or chain.get("checkpointed_recompute_steps") != 4
            or chain.get("mode_restored_after_forward") is not True
            or chain.get("mode_restored_after_backward") is not True
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in gradients.values())):
        raise life.LUError("PI0248 first-update four-step Euler gradient proof is incomplete")
    mode_path = root.parent / "u_mode_restoration_checks.jsonl"
    mode_rows = [json.loads(line) for line in mode_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(mode_rows) != STEPS or any(int(row.get("optimizer_step", -1)) != i or row.get("restored") is not True
                                       or int(row.get("rollout_forward_head_evaluations", -1)) != 4 * i
                                       or int(row.get("backward_recompute_head_evaluations", -1)) != 4 * i
                                       for i, row in enumerate(mode_rows, 1))):
        raise life.LUError("PI0248 action-head training/eval mode was not restored after all updates")
    if card["arm"] == "C":
        for row in rows:
            if any(key in row for key in ("physical_rotation_loss", "lambda_rotation", "total_loss_with_rotation")):
                raise life.LUError("C arm unexpectedly contains the R-only rotation intervention")
    else:
        if any(not math.isfinite(float(row.get("physical_rotation_loss", math.nan))) for row in rows):
            raise life.LUError("R arm lacks finite physical rotation loss on every update")
        peer_root = Path(card["paired_arm_root"])
        peer_training = _winning_training_root({**card, "paths": {**card["paths"], "output_root": str(peer_root)}})
        _, peer_rows = _overlay_rows(peer_training)
        if len(peer_rows) != STEPS:
            raise life.LUError("R cannot close paired noise identity before C completes")
        for c_row, r_row in zip(peer_rows, rows):
            c_noise, r_noise = c_row["stock_initial_noise"], r_row["stock_initial_noise"]
            c_eps, r_eps = c_row["native_epsilon"], r_row["native_epsilon"]
            if (c_row.get("native_tau") != r_row.get("native_tau")
                    or c_row.get("native_epsilon_sha256") != r_row.get("native_epsilon_sha256")
                    or c_noise.get("sha256") != r_noise.get("sha256")
                    or c_noise.get("dtype") != r_noise.get("dtype")
                    or c_noise.get("device") != r_noise.get("device")
                    or c_noise.get("shape") != r_noise.get("shape")
                    or c_eps.get("sha256") != r_eps.get("sha256")
                    or c_eps.get("dtype") != r_eps.get("dtype")
                    or c_eps.get("device") != r_eps.get("device")
                    or c_eps.get("shape") != r_eps.get("shape")):
                raise life.LUError(f"C/R same-noise identity differs at optimizer step {r_row['forward_index']}")
    resume_after = 0
    if root != output / "training":
        resume = life.read_json(root.parent / "resume_record.json")
        resume_after = int(resume.get("resume_after_step", -1))
        if resume_after not in CHECKPOINTS[:-1]:
            raise life.LUError("PI0248 exact-resume cursor is not a sealed checkpoint prefix")
    return {"root": root, "rows": rows, "audit_path": audit_path,
            "resume_after_step": resume_after, "parity_path": parity_path, "chain_path": chain_path,
            "mode_path": mode_path, "parity": parity}


def _normalize_resume_overlay(verified: dict[str, Any]) -> None:
    resume_after = int(verified["resume_after_step"])
    if not resume_after:
        return
    root = verified["root"]
    output_path = root / "pi0247_U_rollout_forward.jsonl"
    full_copy = root / "pi0247_U_rollout_forward.complete.jsonl"
    if full_copy.is_file():
        current = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(current) == STEPS - resume_after and [int(row["forward_index"]) for row in current] == list(range(1, STEPS - resume_after + 1)):
            return
    rows = verified["rows"]
    suffix = [row for row in rows if int(row["forward_index"]) > resume_after]
    if len(suffix) != STEPS - resume_after:
        raise life.LUError("exact-resume S4 audit cannot be split at the sealed checkpoint")
    if not full_copy.exists():
        shutil.copy2(verified["audit_path"], full_copy)
    verified["audit_path"] = full_copy
    payload = "".join(json.dumps({**row, "forward_index": int(row["forward_index"]) - resume_after},
                                 sort_keys=True, separators=(",", ":")) + "\n" for row in suffix)
    temporary = output_path.with_name(output_path.name + ".finalize.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output_path)


def _write_training_integrity(card: dict[str, Any], card_path: Path, verified: dict[str, Any]) -> None:
    output = Path(card["paths"]["output_root"])
    manifest = life.read_json(output / "training_manifest.json")
    if manifest.get("status") != "PASS_TRAINING_INTEGRITY":
        raise life.LUError("PI0248 common training-integrity finalizer did not pass")
    result = {
        "schema": "PI0248_CR_TRAINING_INTEGRITY_V1", "status": "PASS_PI0248_CR_TRAINING_INTEGRITY",
        "arm": card["arm"], "family": card["family"],
        "run_card_sha256": life.sha256_file(card_path),
        "training_manifest_sha256": life.sha256_file(output / "training_manifest.json"),
        "execution_commit": card["source"]["execution_code_commit"],
        "initializer_revision": REVISION, "initializer_tree_sha256": TREE_SHA256,
        "schedule_sha256": card["schedule"]["sha256"], "updates": STEPS,
        "checkpoint_steps": list(CHECKPOINTS), "selected_checkpoint": "NONE",
        "objective_audit_path": str(verified["audit_path"]),
        "objective_audit_sha256": life.sha256_file(verified["audit_path"]),
        "objective_first": verified["rows"][0], "objective_last": verified["rows"][-1],
        "full_s4_steps_and_backward": 4 * STEPS,
        "same_forward_count": STEPS, "mode_restoration_rows": STEPS,
        "mode_audit_sha256": life.sha256_file(verified["mode_path"]),
        "parity_sha256": life.sha256_file(verified["parity_path"]),
        "gradient_chain_sha256": life.sha256_file(verified["chain_path"]),
        "paired_noise_identity": "PASS_ALL_2000_STEPS" if card["arm"] == "R" else "PENDING_R",
        "pi0247_training_manifest_status": manifest.get("status"),
        "selected_checkpoint": "NONE",
    }
    life.write_json(output / "pi0248_training_integrity.json", result)


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--hf-preflight", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--archive-sidecar", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--parent-card", type=Path)
    parser.add_argument("--m64-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--rotation-check", type=Path)
    parser.add_argument("--g0-report", type=Path)
    parser.add_argument("--pbs-script", type=Path)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--arm", choices=("C", "R"))
    parser.add_argument("--step", type=int, choices=CHECKPOINTS)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--resume-after-step", type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        if not all((args.parent_card, args.m64_root, args.output_root, args.execution_root,
                    args.execution_commit, args.rotation_check, args.g0_report, args.pbs_script)):
            parser.error("--prepare requires the parent card, M64 root, output/code roots, commit, and G0 evidence")
        parent = json.loads(args.parent_card.read_text(encoding="utf-8"))
        global life
        life = _load_life(Path(parent["paths"]["repo_code_root"]), args.execution_root)
        prepare(args)
    else:
        if args.run_card is None or args.arm is None:
            parser.error("training, finalization, and HF preflight require --run-card and --arm")
        card, module = _install_adapters(args.run_card, args.arm, args.resume_after_step)
        if args.archive_sidecar:
            if args.step is None or args.archive_root is None:
                parser.error("--archive-sidecar requires --run-card, --arm, --step, and --archive-root")
            module.archive_sidecar(args)
        elif args.hf_preflight:
            module.hf_preflight(args)
        elif args.train:
            if args.resume_after_step not in (0, 1000):
                parser.error("exact PI0248 resume cursor must be 0 or 1000")
            _route_archive_sidecar(module, card)
            module.train(args)
        else:
            module.finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
