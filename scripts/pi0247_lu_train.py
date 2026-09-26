#!/usr/bin/env python3
"""Prepare and run the authorized PI0247 L/U arms on Hopper PBS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


AUTHORIZATION = "PI0247_GENERATION_TO_SELECTED_SIM_GO"
REPO_ID = "leoyuer/pi0246-gate-pass-campaign"
PROFILES = {
    "L": ("PI0247_L_LOCAL_M16_2K_V1", "L_native + 0.1*ell_pos(a_local) + 0.1*ell_delta(a_local)"),
    "U": ("PI0247_U_ROLLOUT_M16_2K_V1", "L_native + 0.1*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4)"),
}
STEPS = 2000
CHECKPOINTS = (1000, 2000)
SEED = 22301
LOSS_COEFFICIENTS = {"ell_pos": 0.1, "ell_delta": 0.1}
R1_CANONICAL_COMMENT = 5842565868
R1_POINTER_COMMENT = 5842568324


class LUError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LUError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model_files(root: Path) -> list[dict[str, Any]]:
    names = {
        "config.json", "model.safetensors.index.json", "processor_config.json",
        "dataset_statistics.json", "statistics.json", "embodiment_id.json",
        "final_model_config.json", "final_processor_config.json",
    }
    paths = [path for path in root.iterdir() if path.is_file() and (path.name in names or path.name.endswith(".safetensors"))]
    processor = root / "processor"
    if processor.is_dir():
        paths.extend(path for path in processor.rglob("*") if path.is_file() and not path.is_symlink())
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix())
    ]


def _verify_p_initializer(bootstrap: dict[str, Any]) -> dict[str, Any]:
    paths = bootstrap["paths"]
    root = Path(paths["base_model_path"])
    receipt_path = Path(bootstrap["initializer"]["durable_receipt_path"])
    receipt = read_json(receipt_path)
    record = receipt.get("source_record") or {}
    if (receipt.get("status") != "DURABLE_VERIFIED"
            or receipt.get("family") != bootstrap["initializer"]["family"]
            or int(receipt.get("checkpoint_step", -1)) != 2000
            or int(receipt.get("max_steps", -1)) != 2000
            or receipt.get("durable_path") != f"{REPO_ID}/arms/P/checkpoint-2000"
            or receipt.get("hf_manifest_commit_id") != bootstrap["initializer"]["screen_revision"]):
        raise LUError("P-final2000 durable receipt is not the sealed PI0246 P checkpoint")
    if (record.get("inventory", {}).get("tree_sha256") != bootstrap["initializer"]["full_checkpoint_tree_sha256"]
            or int(record.get("inventory", {}).get("total_bytes", -1)) != int(bootstrap["initializer"]["full_checkpoint_bytes"])):
        raise LUError("P-final2000 full Trainer tree differs from its sealed PI0246 identity")
    archived = {row["path"]: row for row in record.get("inventory", {}).get("entries", [])}
    for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        local = root / name
        saved = archived.get(name)
        if not local.is_file() or saved is None:
            raise LUError(f"P-final2000 initializer is missing required archived model file: {name}")
        if local.stat().st_size != int(saved["bytes"]) or sha256_file(local) != saved["sha256"]:
            raise LUError(f"P-final2000 local initializer differs from its verified Trainer checkpoint: {name}")
    files = _model_files(root)
    if not any(row["path"].endswith(".safetensors") for row in files):
        raise LUError("P-final2000 local model root has no safetensors shards")
    for row in files:
        archive_path = row["path"]
        if archive_path.startswith("processor/"):
            archive_path = archive_path.removeprefix("processor/")
        saved = archived.get(archive_path)
        if saved is None or int(saved["bytes"]) != row["bytes"] or saved["sha256"] != row["sha256"]:
            raise LUError(f"P-final2000 loader file differs from its durable Trainer checkpoint: {row['path']}")
    return {
        "path": str(root),
        "durable_receipt_path": str(receipt_path),
        "durable_receipt_sha256": sha256_file(receipt_path),
        "durable_revision": receipt["hf_manifest_commit_id"],
        "screen_revision": bootstrap["initializer"]["screen_revision"],
        "screen_manifest_sha256": bootstrap["initializer"]["screen_manifest_sha256"],
        "full_checkpoint_tree_sha256": record["inventory"]["tree_sha256"],
        "full_checkpoint_bytes": int(record["inventory"]["total_bytes"]),
        "model_files": files,
    }


def _verify_runtime_bindings(paths: dict[str, Any], bindings: dict[str, Any]) -> dict[str, Any]:
    shared = Path(paths["code_root"])
    p_root = Path(paths["p_runtime_root"])
    pt_root = Path(paths["training_runtime_root"])
    documents = {}
    for name, root in (("P", p_root), ("PT", pt_root)):
        item = bindings[name]
        path = Path(item["path"])
        expected = item["sha256"]
        seal_path = path.with_name(path.stem + ".seal.json")
        if (path.parent.resolve() != root.resolve() or not path.is_file()
                or sha256_file(path) != expected
                or read_json(seal_path).get("sha256") != expected):
            raise LUError(f"sealed PI0246 {name} runtime binding or seal mismatch")
        binding = read_json(path)
        if Path(binding.get("runtime_root", "")).resolve() != root.resolve():
            raise LUError(f"sealed PI0246 {name} runtime root mismatch")
        documents[name] = binding

    p, pt = documents["P"], documents["PT"]
    if (p.get("schema") != "PI0246_P_PHYSPOS_SAME_FORWARD_RUNTIME_V1"
            or pt.get("schema") != "PI0246_PT_SINGLE_FORWARD_RUNTIME_V1"
            or pt.get("source_p_card_sha256") != p.get("run_card_sha256")
            or pt.get("resolved_config_sha256") != p.get("resolved_config_sha256")
            or pt.get("parent_position_sha256") != p.get("position_overlay_sha256")
            or pt.get("parent_physical_delta_sha256") != p.get("physical_decoder_sha256")
            or pt.get("training_helpers") != p.get("training_helpers")):
        raise LUError("sealed PI0246 PT runtime is not bound to the P source runtime")
    expected_files = {
        p_root / "pi0246_gate_pass_train.py": p["trainer_sha256"],
        p_root / "pi0246_physical_position.py": p["position_overlay_sha256"],
        p_root / "pi0245_physical_delta.py": p["physical_decoder_sha256"],
        pt_root / "pi0246_gate_pass_train.py": pt["trainer_sha256"],
        pt_root / "pi0246_pt_combined_overlay.py": pt["combined_overlay_sha256"],
        pt_root / "pi0246_physical_position.py": pt["position_helper_sha256"],
        pt_root / "pi0245_physical_delta.py": pt["physical_delta_sha256"],
        shared / "scripts/pi0228_h200_large_adapt_v1.py": pt["training_helpers"]["pi0228_h200_large_adapt_v1.py"],
        shared / "scripts/pi0241_fixed16_learning.py": pt["training_helpers"]["pi0241_fixed16_learning.py"],
        shared / "scripts/pi0243_checkpoint_persistence.py": pt["checkpoint_archive_helpers"]["pi0243_checkpoint_persistence.py"],
        shared / "scripts/pi0243_hf_archive_tree.py": pt["checkpoint_archive_helpers"]["pi0243_hf_archive_tree.py"],
        shared / "scripts/pi0243_hf_checkpoint_sidecar.py": pt["checkpoint_archive_helpers"]["pi0243_hf_checkpoint_sidecar.py"],
        shared / "scripts/pi0243_storage_retirement.py": pt["hf_readback_helper_sha256"],
    }
    if any(not path.is_file() or sha256_file(path) != expected for path, expected in expected_files.items()):
        raise LUError("PI0246 P/PT runtime files or shared archive helpers differ from their sealed bindings")
    return {
        name: {"path": str(bindings[name]["path"]), "sha256": bindings[name]["sha256"],
               "schema": documents[name].get("schema"), "run_card_sha256": documents[name].get("run_card_sha256"),
               "resolved_config_sha256": documents[name].get("resolved_config_sha256")}
        for name in ("P", "PT")
    }


def _card_with_p_runtime_paths(card: dict[str, Any]) -> dict[str, Any]:
    paths = card["paths"]
    p_card_path = Path(paths["base_model_path"]).parent / "run_card.json"
    p_binding = read_json(Path(card["source"]["runtime_bindings"]["P"]["path"]))
    expected = p_binding["run_card_sha256"]
    if (sha256_file(p_card_path) != expected
            or read_json(p_card_path.with_name("run_card.seal.json")).get("sha256") != expected):
        raise LUError("P-final initializer run card differs from its sealed runtime binding")
    p_paths = read_json(p_card_path)["paths"]
    inherited = {name: p_paths[name] for name in ("cosmos_cache_root", "cosmos_snapshot")}
    return {**card, "paths": {**inherited, **paths}}


def _hf_preflight(repo_id: str, token_path: Path, prefixes: list[str], *, check_empty: bool) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if not token_path.is_file():
        raise LUError("configured HF token file is missing")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise LUError("configured HF token file is empty")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.whoami()
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    if getattr(info, "private", None) is not False:
        raise LUError("checkpoint destination repository is not public")
    siblings = {str(getattr(row, "rfilename", "")): row for row in (getattr(info, "siblings", None) or [])}
    existing = {}
    for prefix in prefixes:
        found = [name for name in siblings if name == prefix or name.startswith(prefix.rstrip("/") + "/")]
        existing[prefix] = found
        if check_empty and found:
            raise LUError(f"refusing to overwrite existing public-HF PI0247 prefix: {prefix}")
    return {"repo_id": repo_id, "public": True, "token_identity_verified": True, "prefixes_empty": not any(existing.values()), "existing_prefix_files": existing}


def _check_summary(summary: dict[str, Any], parent: dict[str, Any], initializer_path: str) -> None:
    expected = dict(parent)
    expected["start_from_checkpoint"] = initializer_path
    if summary != expected:
        raise LUError("resolved L/U recipe differs from P except for the authorized P-final2000 initializer")
    exact = {
        "max_steps": STEPS, "save_steps": 1000, "warmup_steps": 10,
        "scheduler": "cosine", "optimizer": "adamw_torch",
        "learning_rate": 1e-5, "weight_decay": 1e-5,
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "bf16": True, "fp16": False, "resume_from_checkpoint": False,
        "max_grad_norm": 1.0, "dataloader_num_workers": 0,
        "gradient_checkpointing": False, "num_gpus": 1,
        "eval_strategy": "no", "seed": SEED,
    }
    for key, value in exact.items():
        if summary.get(key) != value:
            raise LUError(f"resolved recipe violates PI0247 fixed configuration at {key}: {summary.get(key)!r}")


def prepare(args: argparse.Namespace) -> None:
    bootstrap = read_json(args.bootstrap)
    if bootstrap.get("authorization") != AUTHORIZATION or bootstrap.get("d1_r_star") != "S4":
        raise LUError("bootstrap is not bound to the current PI0247 authorization and frozen R*=S4")
    paths = bootstrap["paths"]
    repo_code = Path(paths["repo_code_root"])
    shared_root = Path(paths["code_root"])
    training_root = Path(paths["training_runtime_root"])
    initializer = _verify_p_initializer(bootstrap)
    runtime_bindings = _verify_runtime_bindings(paths, bootstrap["runtime_bindings"])
    for field in ("cohort", "schedule"):
        item = bootstrap[field]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise LUError(f"sealed {field} file hash mismatch")
    if sha256_file(Path(paths["p0_effective"])) != bootstrap["p0_effective_sha256"]:
        raise LUError("P0 effective decoder normalization hash mismatch")
    if sha256_file(Path(paths["modality_config"])) != bootstrap["modality_config_sha256"]:
        raise LUError("modality configuration hash mismatch")
    groot_head = subprocess.run(
        ["git", "-C", paths["groot_source"], "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if groot_head != bootstrap["groot_source_commit"]:
        raise LUError("pinned GR00T source commit mismatch")

    hf = _hf_preflight(
        REPO_ID, Path(bootstrap["archive"]["token_file"]),
        [f"arms/PI0247/{arm}" for arm in PROFILES], check_empty=True,
    )
    train_core = load_module(training_root / "pi0246_gate_pass_train.py", "pi0247_lu_training_core")
    fixed = load_module(shared_root / "scripts/pi0241_fixed16_learning.py", "pi0247_lu_fixed16")
    fixed.MAX_STEPS = STEPS
    fixed.CHECKPOINT_STEPS = CHECKPOINTS
    parent_path = Path(bootstrap["parent_resolved_config_path"])
    parent = read_json(parent_path)
    if canonical_sha256(parent) != runtime_bindings["P"]["resolved_config_sha256"]:
        raise LUError("P parent resolved config differs from both sealed runtime bindings")
    cards = {}
    code_files = {
        "pi0247_lu_train.py": repo_code / "scripts/pi0247_lu_train.py",
        "pi0247_u_rollout_overlay.py": repo_code / "scripts/pi0247_u_rollout_overlay.py",
        "pi0228_train_calib_checkpoint_selection.py": repo_code / "scripts/pi0228_train_calib_checkpoint_selection.py",
        "pi0247_lu_train_h200.pbs": repo_code / "hpc/pbs/pi0247_lu_train_h200.pbs",
        "pi0246_pt_combined_overlay.py": training_root / "pi0246_pt_combined_overlay.py",
        "pi0246_physical_position.py": training_root / "pi0246_physical_position.py",
        "pi0245_physical_delta.py": training_root / "pi0245_physical_delta.py",
        "pi0246_gate_pass_train.py": training_root / "pi0246_gate_pass_train.py",
        "pi0241_fixed16_learning.py": shared_root / "scripts/pi0241_fixed16_learning.py",
        "pi0228_h200_large_adapt_v1.py": shared_root / "scripts/pi0228_h200_large_adapt_v1.py",
        "stage40_b0_model_control.py": shared_root / "scripts/stage40_b0_model_control.py",
        "pi0243_checkpoint_persistence.py": shared_root / "scripts/pi0243_checkpoint_persistence.py",
        "pi0243_hf_checkpoint_sidecar.py": shared_root / "scripts/pi0243_hf_checkpoint_sidecar.py",
        "pi0243_hf_archive_tree.py": shared_root / "scripts/pi0243_hf_archive_tree.py",
        "pi0243_storage_retirement.py": shared_root / "scripts/pi0243_storage_retirement.py",
        "pi0243_hf_write_probe.py": shared_root / "scripts/pi0243_hf_write_probe.py",
        "pi0241_reconcile_update_integrity.py": shared_root / "scripts/pi0241_reconcile_update_integrity.py",
    }
    code_hashes = {name: sha256_file(path) for name, path in code_files.items()}
    output_base = Path(paths["output_base"])
    if output_base.exists() or output_base.is_symlink():
        raise LUError(f"refusing to reuse existing PI0247 L/U output root: {output_base}")
    for arm in PROFILES:
        arm_root = output_base / arm
        if arm_root.exists() or arm_root.is_symlink():
            raise LUError(f"refusing to reuse existing {arm} output root")
    output_base.mkdir(parents=True)
    (output_base / "logs").mkdir()
    for arm, (family, loss) in PROFILES.items():
        arm_root = output_base / arm
        arm_paths = {**paths, "output_root": str(arm_root)}
        pi0228, config, summary = train_core.build_runtime_config(arm_paths, arm_root / "training", fixed)
        _check_summary(summary, parent, str(Path(paths["base_model_path"])))
        arm_root.mkdir(parents=True)
        archive = {
            "repo_id": REPO_ID,
            "path_prefix": f"arms/PI0247/{arm}",
            "public": True,
            "token_file": bootstrap["archive"]["token_file"],
            "full_trainer_checkpoints": True,
            "checkpoint_steps": list(CHECKPOINTS),
        }
        card = {
            "schema": "PI0247_LU_RUN_CARD_V1", "status": "SEALED_READY_NOT_STARTED",
            "authorization": AUTHORIZATION, "release_commit": bootstrap["release_commit"],
            "canonical_issue_comment": R1_CANONICAL_COMMENT, "pointer_pr_comment": R1_POINTER_COMMENT,
            "arm": arm, "family": family, "loss": loss, "seed": SEED,
            "embodiment_tag": str(config.data.datasets[0].embodiment_tag),
            "d1_runtime_id": "S4", "cohort": bootstrap["cohort"], "schedule": bootstrap["schedule"],
            "input_identity_audit_sha256": bootstrap["input_identity_audit_sha256"],
            "d1_report_sha256": bootstrap["d1_report_sha256"], "d2_report_sha256": bootstrap["d2_report_sha256"],
            "initializer": initializer,
            "loss_contract": bootstrap["loss_contracts"][arm],
            "loss_coefficients": LOSS_COEFFICIENTS,
            "optimizer": {
                "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS), "learning_rate": 1e-5,
                "weight_decay": 1e-5, "max_grad_norm": 1.0, "batch_size": 1,
                "gradient_accumulation_steps": 1, "workers": 0, "scheduler": "cosine",
                "warmup_steps": 10, "optimizer_state": "fresh AdamW; no resume",
            },
            "resolved_config": summary, "paths": arm_paths, "archive": archive,
            "execution_update_commit": bootstrap["execution_code_commit"],
            "resource": bootstrap["resource"], "hf_preflight": hf,
            "source": {
                "execution_code_commit": bootstrap["execution_code_commit"],
                "execution_code_sha256": code_hashes,
                "groot_source_commit": groot_head,
                "parent_resolved_config_sha256": sha256_file(Path(bootstrap["parent_resolved_config_path"])),
                "parent_resolved_config_path": bootstrap["parent_resolved_config_path"],
                "runtime_core_sha256": code_hashes["pi0246_gate_pass_train.py"],
                "runtime_bindings": runtime_bindings,
                "p0_effective_sha256": bootstrap["p0_effective_sha256"],
                "modality_config_sha256": bootstrap["modality_config_sha256"],
            },
            "physical_contract": {
                "horizon": 40, "physical_dt_s": 0.05, "decoder_unchanged": True,
                "safety_gate_unchanged": True, "active_action_smoothing": False,
                "rstar": "S4", "native_forward_per_update": 1,
                "u_full_rollout_per_update": 1 if arm == "U" else 0,
                "u_euler_steps": 4 if arm == "U" else 0,
                "u_checkpointed_euler_steps": 4 if arm == "U" else 0,
            },
            "forbidden_counts": {
                "calib_reads": 0, "dev_f_reads": 0, "final_f_reads": 0,
                "expert_prefix_inputs": 0, "rtc_conditioning": 0,
                "truncated_rollout_gradients": 0, "local_pc_model_weight_downloads": 0,
                "wbc_runs": 0, "hardware_runs": 0,
            },
        }
        card_path = arm_root / "run_card.json"
        write_json(card_path, card)
        card_sha = sha256_file(card_path)
        write_json(arm_root / "run_card.seal.json", {"sha256": card_sha})
        preparation = {
            "schema": "PI0247_LU_PREPARATION_V1", "status": "READY_NOT_STARTED",
            "arm": arm, "family": family, "run_card_sha256": card_sha,
            "execution_code_commit": bootstrap["execution_code_commit"],
            "initializer_screen_manifest_sha256": initializer["screen_manifest_sha256"],
            "initializer_full_checkpoint_tree_sha256": initializer["full_checkpoint_tree_sha256"],
            "cohort_sha256": bootstrap["cohort"]["sha256"], "schedule_sha256": bootstrap["schedule"]["sha256"],
            "resolved_config": summary, "training_started": False, "selected_checkpoint": "NONE",
        }
        write_json(arm_root / "preparation.json", preparation)
        write_json(arm_root / "preparation.seal.json", {"sha256": sha256_file(arm_root / "preparation.json")})
        cards[arm] = {"path": str(card_path), "sha256": card_sha}
    print(json.dumps({"status": "READY_NOT_STARTED", "cards": cards, "hf_preflight": hf}, indent=2, sort_keys=True))


def _verify_card(card_path: Path, arm: str) -> dict[str, Any]:
    card = read_json(card_path)
    seal_path = card_path.with_name("run_card.seal.json")
    expected_family, expected_loss = PROFILES[arm]
    if (card.get("schema") != "PI0247_LU_RUN_CARD_V1"
            or card.get("status") != "SEALED_READY_NOT_STARTED"
            or card.get("authorization") != AUTHORIZATION
            or card.get("arm") != arm or card.get("family") != expected_family or card.get("loss") != expected_loss
            or card.get("canonical_issue_comment") != R1_CANONICAL_COMMENT
            or card.get("pointer_pr_comment") != R1_POINTER_COMMENT
            or card.get("loss_coefficients") != LOSS_COEFFICIENTS
            or read_json(seal_path).get("sha256") != sha256_file(card_path)):
        raise LUError("run card or seal is not bound to the authorized PI0247 arm")
    return card


def _verify_code(card: dict[str, Any], repo_code: Path, shared_root: Path, training_root: Path) -> None:
    mapping = {
        "pi0247_lu_train.py": repo_code / "scripts/pi0247_lu_train.py",
        "pi0247_u_rollout_overlay.py": repo_code / "scripts/pi0247_u_rollout_overlay.py",
        "pi0228_train_calib_checkpoint_selection.py": repo_code / "scripts/pi0228_train_calib_checkpoint_selection.py",
        "pi0247_lu_train_h200.pbs": repo_code / "hpc/pbs/pi0247_lu_train_h200.pbs",
        "pi0246_pt_combined_overlay.py": training_root / "pi0246_pt_combined_overlay.py",
        "pi0246_physical_position.py": training_root / "pi0246_physical_position.py",
        "pi0245_physical_delta.py": training_root / "pi0245_physical_delta.py",
        "pi0246_gate_pass_train.py": training_root / "pi0246_gate_pass_train.py",
        "pi0241_fixed16_learning.py": shared_root / "scripts/pi0241_fixed16_learning.py",
        "pi0228_h200_large_adapt_v1.py": shared_root / "scripts/pi0228_h200_large_adapt_v1.py",
        "stage40_b0_model_control.py": shared_root / "scripts/stage40_b0_model_control.py",
        "pi0243_checkpoint_persistence.py": shared_root / "scripts/pi0243_checkpoint_persistence.py",
        "pi0243_hf_checkpoint_sidecar.py": shared_root / "scripts/pi0243_hf_checkpoint_sidecar.py",
        "pi0243_hf_archive_tree.py": shared_root / "scripts/pi0243_hf_archive_tree.py",
        "pi0243_storage_retirement.py": shared_root / "scripts/pi0243_storage_retirement.py",
        "pi0243_hf_write_probe.py": shared_root / "scripts/pi0243_hf_write_probe.py",
        "pi0241_reconcile_update_integrity.py": shared_root / "scripts/pi0241_reconcile_update_integrity.py",
    }
    expected = card["source"]["execution_code_sha256"]
    for name, path in mapping.items():
        if sha256_file(path) != expected[name]:
            raise LUError(f"PI0247 training source hash mismatch: {name}")


def hf_preflight(args: argparse.Namespace) -> None:
    card = _verify_card(args.run_card, args.arm)
    archive = card["archive"]
    path = Path(card["paths"]["output_root"]) / "hf_preflight_receipt.json"
    if path.exists():
        print(json.dumps(_verify_hf_preflight_receipt(card, args.run_card), indent=2, sort_keys=True))
        return
    result = _hf_preflight(archive["repo_id"], Path(archive["token_file"]), [archive["path_prefix"]], check_empty=True)
    probe_path = f"control/PI0247/preflight/{card['arm']}-{sha256_file(args.run_card)}-{time.time_ns()}.json"
    probe = subprocess.run(
        [sys.executable, str(Path(card["paths"]["code_root"]) / "scripts/pi0243_hf_write_probe.py"),
         "--token-file", archive["token_file"], "--repo-id", archive["repo_id"], "--path", probe_path],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
    )
    if probe.returncode:
        raise LUError(f"HF authenticated write/readback preflight failed (exit {probe.returncode})")
    probe_result = json.loads(probe.stdout.strip().splitlines()[-1])
    if probe_result.get("readback") != "PASS" or probe_result.get("path") != probe_path:
        raise LUError("HF write probe did not close with the sealed path")
    receipt = {
        "schema": "PI0247_LU_HF_ONLINE_PREFLIGHT_V1", "status": "PASS",
        "arm": card["arm"], "run_card_sha256": sha256_file(args.run_card),
        "repo_id": archive["repo_id"], "target_prefix": archive["path_prefix"],
        "target_prefix_empty": result["prefixes_empty"], "repo_public": result["public"],
        "identity_verified": result["token_identity_verified"], "write_probe": probe_result,
        "completed_epoch_s": time.time(),
    }
    receipt["write_probe_path"] = probe_path
    write_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _verify_hf_preflight_receipt(card: dict[str, Any], card_path: Path) -> dict[str, Any]:
    path = Path(card["paths"]["output_root"]) / "hf_preflight_receipt.json"
    receipt = read_json(path)
    if (receipt.get("schema") != "PI0247_LU_HF_ONLINE_PREFLIGHT_V1"
            or receipt.get("status") != "PASS"
            or receipt.get("arm") != card["arm"]
            or receipt.get("run_card_sha256") != sha256_file(card_path)
            or receipt.get("repo_id") != card["archive"]["repo_id"]
            or receipt.get("target_prefix") != card["archive"]["path_prefix"]
            or receipt.get("target_prefix_empty") is not True
            or receipt.get("repo_public") is not True
            or receipt.get("identity_verified") is not True
            or receipt.get("write_probe", {}).get("readback") != "PASS"
            or receipt.get("write_probe_path") != receipt.get("write_probe", {}).get("path")):
        raise LUError("content-bound online HF preflight receipt is missing or invalid")
    return receipt


def _validate_storage(card: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    output = Path(card["paths"]["output_root"])
    if (output / "training").exists() and not resume:
        raise LUError("refusing to reuse an existing fresh-training directory")
    scratch = os.statvfs(output)
    scratch_free = scratch.f_bavail * scratch.f_frsize
    reference_bytes = int(card["initializer"]["full_checkpoint_bytes"])
    required = 3 * reference_bytes + 10 * 1024**3
    if scratch_free < required:
        raise LUError(f"PERSISTENT_CHECKPOINT_STORAGE_BLOCKED: available={scratch_free} required={required}")
    return {
        "persistent_filesystem": str(output),
        "persistent_free_bytes_before_training": scratch_free,
        "required_persistent_bytes": required,
        "full_trainer_checkpoints_persisted_during_training": True,
        "tmpfs_is_not_a_checkpoint_source": True,
    }


def _validate_initializer_runtime(card: dict[str, Any]) -> None:
    root = Path(card["initializer"]["path"])
    receipt = read_json(Path(card["initializer"]["durable_receipt_path"]))
    entries = {row["path"]: row for row in receipt["source_record"]["inventory"]["entries"]}
    for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        local = root / name
        archived = entries.get(name)
        if (archived is None or not local.is_file() or local.stat().st_size != int(archived["bytes"])
                or sha256_file(local) != archived["sha256"]):
            raise LUError(f"local P-final2000 model file failed archive binding: {name}")
    for field in ("cohort", "schedule"):
        item = card[field]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise LUError(f"sealed {field} changed before training")
    paths = card["paths"]
    if sha256_file(Path(paths["p0_effective"])) != card["source"]["p0_effective_sha256"]:
        raise LUError("P0 effective physical decoder binding changed")
    if sha256_file(Path(paths["modality_config"])) != card["source"]["modality_config_sha256"]:
        raise LUError("modality config changed after sealing")
    result = subprocess.run(["git", "-C", paths["groot_source"], "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    if result.stdout.strip() != card["source"]["groot_source_commit"]:
        raise LUError("pinned GR00T runtime changed after card sealing")


def _install_persistent_archive(card: dict[str, Any], persistence: Any, card_path: Path) -> None:
    from transformers import TrainerCallback
    from gr00t.experiment import experiment as experiment_module

    runner = Path(card["paths"]["repo_code_root"]) / "scripts/pi0247_lu_train.py"

    class ArchiveOnSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            if step not in CHECKPOINTS:
                return control
            target = Path(args.output_dir) / f"checkpoint-{step}"
            if target.resolve().parent != Path(args.output_dir).resolve() or target.is_symlink():
                raise LUError("Trainer checkpoint escaped its persistent run directory")
            record = persistence.checkpoint_record(
                target, expected_step=step, expected_max_steps=STEPS, family=card["family"],
            )
            archive_root = Path(args.output_dir).parent
            receipt_path = archive_root / "persistent_receipts" / f"checkpoint-{step}.durable_receipt.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if (receipt.get("source_record") != record or receipt.get("status") != "DURABLE_VERIFIED"
                        or receipt.get("durable_path") != str(target)):
                    raise LUError(f"checkpoint-{step} persistent receipt conflicts with current bytes")
            else:
                persistence.publish_checkpoint(
                    target, durable_host=None, durable_root=archive_root,
                    expected_step=step, expected_max_steps=STEPS, family=card["family"],
                    receipt_root=archive_root / "persistent_receipts",
                    job_id=os.environ.get("PBS_JOBID", "local"),
                )
            result = subprocess.run(
                [sys.executable, str(runner), "--archive-sidecar", "--run-card", str(card_path),
                 "--arm", card["arm"], "--step", str(step), "--archive-root", str(archive_root)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
            )
            if result.returncode:
                raise LUError(
                    f"checkpoint-{step} remains persistent; isolated HF publisher failed with exit {result.returncode}"
                )
            print(f"PI0247 {card['arm']} checkpoint-{step}: persistent checkpoint and isolated HF readback verified")
            return control

    base = experiment_module.Gr00tTrainer

    class ArchiveTrainer(base):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.add_callback(ArchiveOnSave())

    experiment_module.Gr00tTrainer = ArchiveTrainer


def archive_sidecar(args: argparse.Namespace) -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    card = _verify_card(args.run_card, args.arm)
    step = int(args.step)
    if step not in CHECKPOINTS:
        raise LUError("HF publisher accepts only the sealed checkpoint steps")
    output = Path(card["paths"]["output_root"]).resolve()
    archive_root = Path(args.archive_root or output).resolve()
    if archive_root != output and (archive_root.parent != output / "recovery_attempts" or not archive_root.name):
        raise LUError("HF publisher archive root is outside this sealed run's recovery directory")
    archive = card["archive"]
    target = archive_root / "training" / f"checkpoint-{step}"
    train_core = load_module(Path(card["paths"]["training_runtime_root"]) / "pi0246_gate_pass_train.py", "pi0247_lu_archive_core")
    persistence = load_module(Path(card["paths"]["code_root"]) / "scripts/pi0243_checkpoint_persistence.py", "pi0247_lu_archive_persistence")
    storage = load_module(Path(card["paths"]["code_root"]) / "scripts/pi0243_storage_retirement.py", "pi0247_lu_archive_readback")
    record = persistence.checkpoint_record(
        target, expected_step=step, expected_max_steps=STEPS, family=card["family"],
    )
    effective_prefix = archive["path_prefix"] if archive_root == output else f"{archive['path_prefix']}/recovery/{archive_root.name}"
    receipt_path = archive_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    readback_path = archive_root / "hf_readbacks" / f"checkpoint-{step}.json"
    if not receipt_path.exists():
        command = [
            sys.executable, str(Path(card["paths"]["code_root"]) / "scripts/pi0243_hf_checkpoint_sidecar.py"),
            "--output-root", str(archive_root), "--repo-id", archive["repo_id"],
            "--token-file", archive["token_file"], "--path-prefix", effective_prefix,
            "--steps", str(step), "--family", card["family"], "--max-steps", str(STEPS),
            "--state-path", str(archive_root / "hf_archive_state.json"), "--checkpoint-timeout", "21600",
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise LUError(f"HF archive sidecar failed for checkpoint-{step} (exit {result.returncode})")
    receipt = read_json(receipt_path)
    train_core.validate_pt_archive_receipt(
        receipt, record, repo_id=archive["repo_id"], prefix=effective_prefix,
        family=card["family"], step=step,
    )
    if not readback_path.exists():
        readback = train_core.verify_pt_hf_readback(
            receipt, record, storage_helper=storage,
            token=Path(archive["token_file"]).read_text(encoding="utf-8").strip(),
            repo_id=archive["repo_id"], prefix=effective_prefix,
            family=card["family"], step=step,
        )
        readback.update({
            "schema": "PI0247_LU_FULL_HF_READBACK_V1",
            "status": "PASS_PI0247_LU_FULL_HF_READBACK", "arm": card["arm"],
            "source_record_sha256": canonical_sha256(record),
        })
        write_json(readback_path, readback)
    else:
        readback = read_json(readback_path)
        expected = {
            f"{effective_prefix}/checkpoint-{step}/{entry['path']}": (entry["bytes"], entry["sha256"])
            for entry in record["inventory"]["entries"]
        }
        observed = {row.get("path"): row for row in readback.get("files", [])}
        if (readback.get("status") != "PASS_PI0247_LU_FULL_HF_READBACK"
                or readback.get("source_tree_sha256") != record["inventory"]["tree_sha256"]
                or readback.get("revision") != receipt.get("hf_manifest_commit_id")
                or set(observed) != set(expected)
                or any(int(observed[path].get("bytes", -1)) != size or observed[path].get("sha256") != digest
                       for path, (size, digest) in expected.items())):
            raise LUError(f"checkpoint-{step} existing HF readback conflicts with its receipt")
    print(json.dumps({"status": "COMPLETE", "step": step, "tree_sha256": record["inventory"]["tree_sha256"],
                      "archive_root": str(archive_root), "prefix": effective_prefix}, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    arm = args.arm
    resume_after = int(args.resume_after_step or 0)
    if resume_after not in (0, 1000):
        raise LUError("PI0247 exact resume is bound to the sealed checkpoint-1000 cursor")
    card = _verify_card(args.run_card, arm)
    paths = card["paths"]
    repo_code = Path(paths["repo_code_root"])
    shared_root = Path(paths["code_root"])
    training_root = Path(paths["training_runtime_root"])
    _verify_code(card, repo_code, shared_root, training_root)
    _verify_runtime_bindings(paths, card["source"]["runtime_bindings"])
    _validate_initializer_runtime(card)
    hf = _verify_hf_preflight_receipt(card, args.run_card)
    output = Path(paths["output_root"])
    storage = _validate_storage(card, resume=bool(resume_after))
    persistence = load_module(shared_root / "scripts/pi0243_checkpoint_persistence.py", "pi0247_lu_persistence")
    if resume_after:
        if (output / "training_manifest.json").is_file():
            raise LUError("training manifest already exists; repair archive/finalize evidence without retraining")
        recovery_root = output / "recovery_attempts"
        prior_roots = [output / "training"] + (sorted(recovery_root.glob("*/training")) if recovery_root.is_dir() else [])
        for prior_root in prior_roots:
            complete_path = prior_root / f"checkpoint-{STEPS}"
            if complete_path.is_dir():
                try:
                    persistence.checkpoint_record(
                        complete_path, expected_step=STEPS, expected_max_steps=STEPS, family=card["family"],
                    )
                except Exception:
                    continue
                raise LUError(
                    f"valid checkpoint-{STEPS} already exists at {complete_path}; publish/finalize it without retraining"
                )
        resume_source = output / "training" / f"checkpoint-{resume_after}"
        durable_receipt_path = output / "persistent_receipts" / f"checkpoint-{resume_after}.durable_receipt.json"
        durable_receipt = persistence.verify_receipt(
            resume_source, durable_receipt_path, expected_step=resume_after,
            expected_max_steps=STEPS, family=card["family"],
        )
        resume_record = persistence.checkpoint_record(
            resume_source, expected_step=resume_after, expected_max_steps=STEPS, family=card["family"],
        )
        if (durable_receipt.get("durable_host") != "local"
                or durable_receipt.get("durable_path") != str(resume_source)
                or durable_receipt.get("source_record") != resume_record):
            raise LUError("exact resume source checkpoint is not the verified persistent Trainer state")
        job = re.sub(r"[^A-Za-z0-9_.-]", "_", os.environ.get("PBS_JOBID", f"local-{time.time_ns()}"))
        attempt_root = output / "recovery_attempts" / job
        if attempt_root.exists():
            raise LUError("refusing to reuse an existing exact-resume attempt directory")
        attempt_root.mkdir(parents=True)
        training_output = attempt_root / "training"
        training_output.mkdir()
        if arm == "U":
            for name, expected_rows in (("pi0247_U_rollout_forward.jsonl", resume_after),):
                source = output / "training" / name
                rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != expected_rows:
                    raise LUError(f"U exact resume source audit is not the full checkpoint prefix: {name}")
                shutil.copy2(source, training_output / name)
            mode_source = output / "u_mode_restoration_checks.jsonl"
            mode_rows = [line for line in mode_source.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(mode_rows) != resume_after:
                raise LUError("U exact resume lacks one mode-restoration receipt per checkpoint-prefix update")
            shutil.copy2(mode_source, attempt_root / "u_mode_restoration_checks.jsonl")
        write_json(attempt_root / "resume_record.json", {
            "schema": "PI0247_LU_EXACT_RESUME_V1", "status": "SEALED_RESUME_SOURCE",
            "arm": arm, "resume_after_step": resume_after,
            "run_card_sha256": sha256_file(args.run_card),
            "source_checkpoint_path": str(resume_source),
            "source_checkpoint_tree_sha256": resume_record["inventory"]["tree_sha256"],
            "persistent_receipt_sha256": sha256_file(durable_receipt_path),
            "schedule_sha256": card["schedule"]["sha256"],
            "remaining_rows": STEPS - resume_after,
            "optimizer_state": "restored exactly from full Trainer checkpoint",
        })
    else:
        attempt_root = output
        training_output = output / "training"
        training_output.mkdir()
        write_json(output / "pre_model_preflight.json", {
            "status": "PASS_PRE_MODEL_PREFLIGHT", "arm": arm, "family": card["family"],
            "authorization": AUTHORIZATION, "hf": hf, "storage": storage,
            "initializer_manifest_sha256": card["initializer"]["screen_manifest_sha256"],
            "cohort_sha256": card["cohort"]["sha256"], "schedule_sha256": card["schedule"]["sha256"],
            "model_weights_downloaded": False,
        })

    import torch
    if torch.cuda.device_count() != 1 or "h200" not in torch.cuda.get_device_name(0).lower():
        raise LUError("PI0247 training requires the authorized single-H200 PBS allocation")

    train_core = load_module(training_root / "pi0246_gate_pass_train.py", "pi0247_lu_train_core")
    fixed = load_module(shared_root / "scripts/pi0241_fixed16_learning.py", "pi0247_lu_fixed16_train")
    fixed.MAX_STEPS = STEPS
    fixed.CHECKPOINT_STEPS = CHECKPOINTS
    paths = card["paths"]
    groot_runtime = Path(paths["stage10_root"]) / "runtime"
    sys.path[:0] = [str(groot_runtime), str(training_root), str(shared_root / "scripts"), str(shared_root / "src"), str(paths["groot_source"])]
    fixed.bind_runtime(_card_with_p_runtime_paths(card))
    fixed.install_fixed_dataset_factory(card, training_output, resume_after_step=resume_after)
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from gr00t.model.gr00t_n1d7 import setup as gr00t_setup
    processor_ref: dict[str, Any] = {}
    factory_type = gr00t_setup.DatasetFactory
    original_factory_build = factory_type.build

    def capture_processor(factory: Any, processor: Any) -> Any:
        processor_ref["value"] = processor
        return original_factory_build(factory, processor)

    factory_type.build = capture_processor

    def get_processor() -> Any:
        return processor_ref.get("value")

    checkpoint_output = training_output
    fixed.install_update_audit(
        training_output, audit_steps={1, 10, 100, 1000, 2000},
        expected_end_step=STEPS, checkpoint_output=checkpoint_output,
    )
    _install_persistent_archive(card, persistence, args.run_card)

    effective = read_json(Path(paths["p0_effective"]))
    overlay_kwargs = {
        "scale": torch.tensor(effective["scale_xyz_m"], dtype=torch.float32),
        "offset": torch.tensor(effective["offset_xyz_m"], dtype=torch.float32),
        "state_min": torch.tensor(effective["state_min"], dtype=torch.float32),
        "state_max": torch.tensor(effective["state_max"], dtype=torch.float32),
    }
    if arm == "L":
        overlay_path = training_root / "pi0246_pt_combined_overlay.py"
        overlay = load_module(overlay_path, "pi0247_l_local_combined_overlay")
        overlay_audit_path = training_output / "pi0247_L_local_forward.jsonl"
        overlay_audit = overlay.install(Gr00tN1d7ActionHead, **overlay_kwargs, audit_path=overlay_audit_path)
    else:
        overlay_path = repo_code / "scripts/pi0247_u_rollout_overlay.py"
        overlay = load_module(overlay_path, "pi0247_u_rollout_overlay")
        overlay_audit_path = training_output / "pi0247_U_rollout_forward.jsonl"
        parity_path = output / "pi0247_U_sampler_parity.json"
        overlay_audit = overlay.install(
            Gr00tN1d7ActionHead, **overlay_kwargs, audit_path=overlay_audit_path,
            parity_path=parity_path, processor_getter=get_processor,
            embodiment_tag=card["embodiment_tag"], resume_after_step=resume_after,
            expected_r_star=card["d1_runtime_id"],
        )
        mode_path = (attempt_root / "u_mode_restoration_checks.jsonl"
                     if resume_after else output / "u_mode_restoration_checks.jsonl")
        _install_u_chain_check(overlay_audit, output, mode_path=mode_path)

    pi0228, config, summary = train_core.build_runtime_config(paths, checkpoint_output, fixed)
    if summary != card["resolved_config"]:
        raise LUError("resolved runtime config changed after card sealing")
    if resume_after:
        config.training.resume_from_checkpoint = str(output / "training" / f"checkpoint-{resume_after}")
    write_json(training_output / "resolved_config.json", summary)
    write_json(output / "execution_state.json", {
        "status": "TRAINING_STARTED", "arm": arm, "family": card["family"],
        "run_card_sha256": sha256_file(args.run_card),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "execution_code_commit": card["source"]["execution_code_commit"],
        "started_epoch_s": time.time(), "resume_after_step": resume_after,
        "training_output": str(training_output),
    })
    pi0228.run_official(config, Path(paths["groot_source"]))

    finalize(args)


def finalize(args: argparse.Namespace) -> None:
    """Close completed local evidence without loading a model or changing checkpoints."""
    arm = args.arm
    card = _verify_card(args.run_card, arm)
    paths = card["paths"]
    _verify_code(card, Path(paths["repo_code_root"]), Path(paths["code_root"]), Path(paths["training_runtime_root"]))
    _verify_runtime_bindings(paths, card["source"]["runtime_bindings"])
    _validate_initializer_runtime(card)
    output = Path(paths["output_root"])
    training_roots = [output / "training"]
    recovery_root = output / "recovery_attempts"
    if recovery_root.is_dir():
        training_roots.extend(sorted(recovery_root.glob("*/training")))
    state_path = output / "execution_state.json"
    state = read_json(state_path)
    preflight = read_json(output / "pre_model_preflight.json")
    _verify_hf_preflight_receipt(card, args.run_card)
    if (state.get("arm") != arm or state.get("run_card_sha256") != sha256_file(args.run_card)
            or preflight.get("status") != "PASS_PRE_MODEL_PREFLIGHT"
            or preflight.get("model_weights_downloaded") is not False):
        raise LUError("read-only finalize lacks the sealed pre-model admission record")

    def rows_at(root: Path, name: str) -> list[dict[str, Any]]:
        path = root / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

    complete_audits = []
    for root in training_roots:
        path = root / "pi0241_update_audit.json"
        if path.is_file():
            audit = read_json(path)
            if audit.get("status") == "COMPLETE" and int(audit.get("global_step", -1)) == STEPS:
                complete_audits.append((path, audit, root))
    if not complete_audits:
        raise LUError("read-only finalize lacks a complete 2000-step update audit")
    update_audit_path, update_audit, winning_root = max(complete_audits, key=lambda item: item[0].stat().st_mtime_ns)
    resume_after = 0
    resume_record = None
    prefix_parameters = None
    prefix_membership = None
    if winning_root != training_roots[0]:
        resume_record = read_json(winning_root.parent / "resume_record.json")
        if (resume_record.get("status") != "SEALED_RESUME_SOURCE"
                or resume_record.get("arm") != arm
                or resume_record.get("run_card_sha256") != sha256_file(args.run_card)
                or resume_record.get("schedule_sha256") != card["schedule"]["sha256"]):
            raise LUError("winning recovery attempt is not bound to this run card and fixed schedule")
        resume_after = int(resume_record.get("resume_after_step", -1))
        if resume_after != 1000:
            raise LUError("winning exact-resume attempt does not start from checkpoint-1000")
        prefix_audit = read_json(training_roots[0] / "pi0241_update_audit.json")
        prefix_event = next((row for row in prefix_audit.get("events", [])
                             if row.get("event") == "POST_OPTIMIZER" and int(row.get("step", -1)) == resume_after), None)
        if prefix_event is None:
            raise LUError("exact-resume checkpoint prefix lacks its step-1000 optimizer reconciliation")
        prefix_parameters = prefix_event.get("parameters", {})
        prefix_membership = prefix_audit.get("optimizer_membership", {})

    cohort = read_json(Path(card["cohort"]["path"]))["rows"]
    planned = read_json(Path(card["schedule"]["path"]))["rows"]
    if winning_root == training_roots[0]:
        consumption = rows_at(winning_root, "pi0241_sample_consumption.jsonl")
        contracts_all = rows_at(winning_root, "pi0241_sample_contracts.jsonl")
    else:
        consumption = [row for row in rows_at(training_roots[0], "pi0241_sample_consumption.jsonl")
                       if int(row.get("optimizer_step", -1)) <= resume_after]
        consumption.extend(row for row in rows_at(winning_root, "pi0241_sample_consumption.jsonl")
                           if int(row.get("optimizer_step", -1)) > resume_after)
        contracts_all = rows_at(training_roots[0], "pi0241_sample_contracts.jsonl")
        contracts_all.extend(rows_at(winning_root, "pi0241_sample_contracts.jsonl"))
    if len(consumption) != STEPS or len(planned) != STEPS:
        raise LUError("actual TRAIN sample consumption differs from the sealed 2000-row schedule")
    consumption.sort(key=lambda row: int(row.get("yield_ordinal", -1)))
    for ordinal, (actual, plan) in enumerate(zip(consumption, planned), start=1):
        fixed_index = int(plan["fixed_index"])
        if (fixed_index < 0 or fixed_index >= len(cohort)
                or int(actual.get("yield_ordinal", -1)) != ordinal
                or int(actual.get("optimizer_step", -1)) != ordinal
                or int(actual.get("fixed_index", -1)) != fixed_index
                or int(actual.get("episode_index", -1)) != int(cohort[fixed_index]["episode_index"])
                or actual.get("source_episode_id") != cohort[fixed_index]["source_episode_id"]
                or int(actual.get("observation_index", -1)) != 0):
            raise LUError(f"actual TRAIN sample consumption differs from the sealed schedule at step {ordinal}")
    contracts_by_index: dict[int, dict[str, Any]] = {}
    for row in contracts_all:
        index = int(row["fixed_index"])
        if index in contracts_by_index and contracts_by_index[index] != row:
            raise LUError(f"fixed16 contract changed during exact resume at fixed_index={index}")
        contracts_by_index[index] = row
    contracts = [contracts_by_index[index] for index in sorted(contracts_by_index)]
    if (len(contracts) != 16 or {int(row["fixed_index"]) for row in contracts} != set(range(16))
            or any(row["active_slots"] != list(range(10)) or row["action_shape"] != [1, 40, 132]
                   or row["mask_shape"] != [1, 40, 132]
                   or row["source_episode_id"] != cohort[int(row["fixed_index"])]["source_episode_id"]
                   for row in contracts)):
        raise LUError("actual fixed16 action/mask contracts are incomplete")

    overlay_name = "pi0247_L_local_forward.jsonl" if arm == "L" else "pi0247_U_rollout_forward.jsonl"
    overlay_rows = []
    if winning_root == training_roots[0]:
        active_overlay = [(winning_root, 0)]
    elif arm == "U":
        active_overlay = [(winning_root, 0)]
    else:
        active_overlay = [(training_roots[0], 0), (winning_root, resume_after)]
    for root, offset in active_overlay:
        local_rows = rows_at(root, overlay_name)
        for row in local_rows:
            schedule_step = int(row["forward_index"]) + offset
            if root == training_roots[0] and winning_root != training_roots[0] and schedule_step > resume_after:
                continue
            row["_pi0247_schedule_step"] = schedule_step
            overlay_rows.append(row)
    overlay_rows.sort(key=lambda row: int(row["_pi0247_schedule_step"]))
    if (len(overlay_rows) != STEPS
            or [int(row["_pi0247_schedule_step"]) for row in overlay_rows] != list(range(1, STEPS + 1))):
        raise LUError(f"{arm} same-forward objective audit is incomplete")
    required = {"native_loss", "physical_position_loss", "physical_delta_loss", "total_loss"}
    for row in overlay_rows:
        values = [float(row[key]) for key in ("native_loss", "physical_position_loss", "physical_delta_loss", "total_loss")]
        if not required.issubset(row) or not all(math.isfinite(value) for value in values):
            raise LUError(f"{arm} objective audit has missing/nonfinite fields")
        prescribed = (values[0] + float(card["loss_coefficients"]["ell_pos"]) * values[1]
                      + float(card["loss_coefficients"]["ell_delta"]) * values[2])
        if not math.isclose(values[3], prescribed, rel_tol=1e-5, abs_tol=1e-7):
            raise LUError(f"{arm} objective terms do not sum to the sealed loss")

    persistence = load_module(Path(paths["code_root"]) / "scripts/pi0243_checkpoint_persistence.py", "pi0247_lu_finalize_persistence")
    train_core = load_module(Path(paths["training_runtime_root"]) / "pi0246_gate_pass_train.py", "pi0247_lu_finalize_core")
    checkpoints = {}
    checkpoint_records = {}
    checkpoint_roots = {1000: training_roots[0], 2000: winning_root}
    for step in CHECKPOINTS:
        archive_root = checkpoint_roots[step].parent
        local_path = checkpoint_roots[step] / f"checkpoint-{step}"
        local_receipt_path = archive_root / "persistent_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt_path = archive_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        readback_path = archive_root / "hf_readbacks" / f"checkpoint-{step}.json"
        effective_prefix = (card["archive"]["path_prefix"] if archive_root == output
                            else f"{card['archive']['path_prefix']}/recovery/{archive_root.name}")
        record = persistence.checkpoint_record(
            local_path, expected_step=step, expected_max_steps=STEPS, family=card["family"],
        )
        checkpoint_records[step] = record
        persistence.verify_receipt(
            local_path, local_receipt_path, expected_step=step,
            expected_max_steps=STEPS, family=card["family"],
        )
        local_receipt = read_json(local_receipt_path)
        receipt = read_json(receipt_path)
        readback = read_json(readback_path)
        local_tree = local_receipt.get("source_record", {}).get("inventory", {})
        local_durable_tree = local_receipt.get("durable_inventory", {})
        expected_files = {
            f"{effective_prefix}/checkpoint-{step}/{entry['path']}": entry
            for entry in record["inventory"]["entries"]
        }
        observed_files = {row.get("path"): row for row in readback.get("files", [])}
        train_core.validate_pt_archive_receipt(
            receipt, record, repo_id=card["archive"]["repo_id"], prefix=effective_prefix,
            family=card["family"], step=step,
        )
        if (local_receipt.get("status") != "DURABLE_VERIFIED"
                or local_receipt.get("family") != card["family"]
                or int(local_receipt.get("checkpoint_step", -1)) != step
                or local_receipt.get("durable_host") != "local"
                or local_receipt.get("durable_path") != str(local_path)
                or local_tree.get("tree_sha256") != record["inventory"]["tree_sha256"]
                or int(local_tree.get("total_bytes", -1)) != int(record["inventory"]["total_bytes"])
                or local_durable_tree.get("tree_sha256") != record["inventory"]["tree_sha256"]
                or set(observed_files) != set(expected_files)
                or readback.get("status") != "PASS_PI0247_LU_FULL_HF_READBACK"
                or readback.get("source_tree_sha256") != record["inventory"]["tree_sha256"]
                or readback.get("revision") != receipt.get("hf_manifest_commit_id")):
            raise LUError(f"checkpoint-{step} persistence/archive/readback evidence does not close")
        for remote_path, expected_entry in expected_files.items():
            observed = observed_files[remote_path]
            if (int(observed.get("bytes", -1)) != int(expected_entry["bytes"])
                    or observed.get("sha256") != expected_entry["sha256"]):
                raise LUError(f"checkpoint-{step} pinned HF readback differs at {remote_path}")
        checkpoints[str(step)] = {
            "family": card["family"], "step": step,
            "checkpoint_path": str(local_path), "hf_prefix": effective_prefix,
            "tree_sha256": record["inventory"]["tree_sha256"],
            "total_bytes": record["inventory"]["total_bytes"],
            "file_count": record["inventory"]["file_count"],
            "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
            "durable_receipt_sha256": sha256_file(receipt_path),
            "persistent_receipt_sha256": sha256_file(local_receipt_path),
            "full_hf_readback_sha256": sha256_file(readback_path),
        }
    if resume_record is not None:
        source_receipt_path = output / "persistent_receipts" / "checkpoint-1000.durable_receipt.json"
        if (resume_record.get("source_checkpoint_path") != str(output / "training" / "checkpoint-1000")
                or resume_record.get("source_checkpoint_tree_sha256")
                   != checkpoint_records[1000]["inventory"]["tree_sha256"]
                or resume_record.get("persistent_receipt_sha256") != sha256_file(source_receipt_path)):
            raise LUError("winning recovery manifest does not restore the exact persistent checkpoint-1000")

    projector_helper = load_module(
        Path(paths["code_root"]) / "scripts/pi0241_reconcile_update_integrity.py",
        "pi0247_lu_finalize_projector",
    )
    projector_delta = projector_helper.compare(
        projector_helper.projector_tensors(Path(card["initializer"]["path"])),
        projector_helper.projector_tensors(checkpoint_roots[2000] / "checkpoint-2000"),
    )
    parameters = update_audit["final_parameters"]
    parameter_rows = [parameters] + ([prefix_parameters] if prefix_parameters is not None else [])
    memberships = [update_audit.get("optimizer_membership", {})]
    if prefix_membership is not None:
        memberships.append(prefix_membership)
    integrity = {
        "projector_updated": projector_delta["changed_tensor_count"] > 0,
        "diffusion_updated": any(float(row["diffusion"]["delta_l2"]) > 0 for row in parameter_rows),
        "frozen_backbone_unchanged": all(float(row["frozen_backbone"]["delta_l2"]) == 0 for row in parameter_rows),
        "frozen_vlln_unchanged": all(float(row["frozen_vlln"]["delta_l2"]) == 0 for row in parameter_rows),
        "optimizer_membership_valid": all(not row.get("missing_trainable") and not row.get("illegal_frozen")
                                           for row in memberships),
    }
    rollout_audit = parity = None
    if arm == "U":
        parity = read_json(output / "pi0247_U_sampler_parity.json")
        chain = read_json(output / "u_first_update_chain_gradient.json")
        mode_path = winning_root.parent / "u_mode_restoration_checks.jsonl"
        mode_rows = [json.loads(line) for line in mode_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if (parity.get("status") != "PASS_PI0247_U_S4_PARITY"
                or not parity.get("gate_decision_equal") or not parity.get("gate_issues_equal")
                or chain.get("status") != "PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN"
                or set(chain.get("state_gradient_norms", {})) != {"1", "2", "3", "4"}
                or len(mode_rows) != STEPS
                or any(int(row.get("optimizer_step", -1)) != index or row.get("restored") is not True
                       for index, row in enumerate(mode_rows, start=1))):
            raise LUError("U stock parity, first-update gradient chain, or mode-restoration evidence is incomplete")
        expected_evaluations = {
            "parity_reference": N_STEPS, "parity_candidate": N_STEPS,
            "rollout_forward": N_STEPS * STEPS, "backward_recompute": N_STEPS * STEPS,
        }
        observed_evaluations = {
            "parity_reference": int(parity.get("head_evaluations_reference", -1)),
            "parity_candidate": int(parity.get("head_evaluations_candidate", -1)),
            "rollout_forward": int(mode_rows[-1].get("rollout_forward_head_evaluations", -1)),
            "backward_recompute": int(mode_rows[-1].get("backward_recompute_head_evaluations", -1)),
        }
        if observed_evaluations != expected_evaluations:
            raise LUError(f"U full Euler forward/recompute counts do not close: {observed_evaluations}")
        rollout_audit = {
            "native_forwards": STEPS, "training_rollouts": STEPS,
            "parity_nonselection_rollouts": 2, "head_evaluations": observed_evaluations,
            "mode_restoration_forward_checks": STEPS,
            "mode_restoration_after_backward_checks": len(mode_rows),
            "first_update_euler_gradient_norms": chain["state_gradient_norms"],
        }
        integrity.update({
            "full_euler_gradient_chain_verified": True,
            "exact_s4_parity_verified": True,
            "action_head_eval_autograd_mode_restored": True,
        })

    result = {
        "schema": "PI0247_LU_TRAINING_MANIFEST_V1",
        "status": "PASS_TRAINING_INTEGRITY" if all(integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "arm": arm, "family": card["family"], "authorization": AUTHORIZATION,
        "run_card_sha256": sha256_file(args.run_card),
        "execution_code_commit": card["source"]["execution_code_commit"],
        "pbs_job_id": state.get("pbs_job_id"), "global_step": STEPS,
        "sample_consumption": len(consumption), "sample_contracts": contracts,
        "schedule_sha256": card["schedule"]["sha256"], "cohort_sha256": card["cohort"]["sha256"],
        "runtime_id": "S4", "resolved_config": card["resolved_config"],
        "update_integrity": integrity, "update_audit_sha256": sha256_file(update_audit_path),
        "full_projector_reconciliation": projector_delta,
        "objective_audit": {
            "path": [str(root / overlay_name) for root, _ in active_overlay if (root / overlay_name).is_file()],
            "sha256": [sha256_file(root / overlay_name) for root, _ in active_overlay if (root / overlay_name).is_file()],
            "rows": len(overlay_rows),
            "first": overlay_rows[0], "last": overlay_rows[-1],
            "loss_coefficients": card["loss_coefficients"],
        },
        "rollout_audit": rollout_audit, "parity": parity,
        "checkpoints": checkpoints, "selected_checkpoint": "NONE",
        "archive_repo_id": card["archive"]["repo_id"],
        "archive_prefix": card["archive"]["path_prefix"],
        "forbidden_counts": card["forbidden_counts"],
    }
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != result:
        raise LUError("read-only finalize refuses to replace a conflicting training manifest")
    if not manifest_path.exists():
        write_json(manifest_path, result)
    write_json(output / "execution_state.json", {
        "status": "TRAINING_COMPLETE", "arm": arm, "family": card["family"],
        "training_manifest_sha256": sha256_file(manifest_path),
        "completed_epoch_s": time.time(), "selected_checkpoint": "NONE",
        "pbs_job_id": state.get("pbs_job_id"), "run_card_sha256": sha256_file(args.run_card),
    })
    print(json.dumps({"status": result["status"], "arm": arm, "steps": STEPS,
                      "checkpoints": checkpoints}, indent=2, sort_keys=True))


def _install_u_chain_check(audit: dict[str, Any], output: Path, *, mode_path: Path | None = None) -> None:
    from transformers import TrainerCallback
    from gr00t.experiment import experiment as experiment_module

    base = experiment_module.Gr00tTrainer
    mode_path = mode_path or output / "u_mode_restoration_checks.jsonl"
    previous_modes = [json.loads(line) for line in mode_path.read_text(encoding="utf-8").splitlines() if line.strip()] if mode_path.exists() else []
    if any(int(row.get("optimizer_step", -1)) != index or row.get("restored") is not True
           for index, row in enumerate(previous_modes, start=1)):
        raise LUError("existing U mode-restoration evidence is not a contiguous passing prefix")
    audit["mode_restoration_after_backward_checks"] = len(previous_modes)

    class EulerChainAudit(TrainerCallback):
        def __init__(self, action_head: Any):
            self.action_head = action_head

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            expected_modes = audit.get("native_mode_snapshot")
            if expected_modes is None or [bool(child.training) for child in self.action_head.modules()] != expected_modes:
                raise LUError("U action-head modes were not restored after the Euler backward recomputation")
            optimizer_step = int(state.global_step) + 1
            with mode_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "optimizer_step": optimizer_step, "restored": True,
                    "rollout_forward_head_evaluations": audit["head_evaluations"]["rollout_forward"],
                    "backward_recompute_head_evaluations": audit["head_evaluations"]["backward_recompute"],
                }, sort_keys=True) + "\n")
                stream.flush()
                if optimizer_step % 100 == 0:
                    os.fsync(stream.fileno())
            audit["mode_restoration_after_backward_checks"] += 1
            if optimizer_step != 1:
                return control
            values = audit["first_update_euler_gradient_norms"]
            if set(values) != {"1", "2", "3", "4"} or any(not math.isfinite(float(value)) or float(value) <= 0 for value in values.values()):
                raise LUError(f"first U optimizer update lacks nonzero finite gradients through all Euler states: {values}")
            write_json(output / "u_first_update_chain_gradient.json", {
                "schema": "PI0247_U_FIRST_UPDATE_CHAIN_GRADIENT_V1",
                "status": "PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN",
                "optimizer_step": 1, "euler_steps": 4,
                "state_gradient_norms": values,
                "checkpointed_recompute_steps": 4,
                "mode_restored_after_forward": audit["mode_restoration_forward_checks"] == 1,
                "mode_restored_after_backward": True,
            })
            return control

    class ChainAuditedTrainer(base):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.add_callback(EulerChainAudit(self.model.action_head))

    experiment_module.Gr00tTrainer = ChainAuditedTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--hf-preflight", action="store_true")
    mode.add_argument("--archive-sidecar", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--arm", choices=tuple(PROFILES))
    parser.add_argument("--step", type=int, choices=CHECKPOINTS)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--resume-after-step", type=int, choices=(1000,))
    args = parser.parse_args()
    if args.prepare:
        if args.bootstrap is None:
            parser.error("--prepare requires --bootstrap")
        prepare(args)
    elif args.hf_preflight:
        if args.run_card is None or args.arm is None:
            parser.error("--hf-preflight requires --run-card and --arm")
        hf_preflight(args)
    elif args.archive_sidecar:
        if args.run_card is None or args.arm is None or args.step is None:
            parser.error("--archive-sidecar requires --run-card, --arm, and --step")
        archive_sidecar(args)
    elif args.finalize:
        if args.run_card is None or args.arm is None:
            parser.error("--finalize requires --run-card and --arm")
        finalize(args)
    else:
        if args.run_card is None or args.arm is None:
            parser.error("--train requires --run-card and --arm")
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
