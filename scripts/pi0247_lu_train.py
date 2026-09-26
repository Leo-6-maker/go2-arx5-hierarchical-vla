#!/usr/bin/env python3
"""Prepare and run the authorized PI0247 L/U arms on Hopper PBS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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


def _hf_preflight(repo_id: str, token_path: Path, prefixes: list[str], *, check_empty: bool) -> dict[str, Any]:
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
    }
    code_hashes = {name: sha256_file(path) for name, path in code_files.items()}
    output_base = Path(paths["output_base"])
    if output_base.exists() or output_base.is_symlink():
        raise LUError(f"refusing to reuse existing PI0247 L/U output root: {output_base}")
    for arm in PROFILES:
        arm_root = output_base / arm
        checkpoint_root = Path(paths["checkpoint_tmp_base"]) / arm
        if arm_root.exists() or arm_root.is_symlink() or checkpoint_root.exists() or checkpoint_root.is_symlink():
            raise LUError(f"refusing to reuse existing {arm} output or tmpfs root")
    output_base.mkdir(parents=True)
    (output_base / "logs").mkdir()
    for arm, (family, loss) in PROFILES.items():
        arm_root = output_base / arm
        checkpoint_root = Path(paths["checkpoint_tmp_base"]) / arm
        arm_paths = {**paths, "output_root": str(arm_root), "checkpoint_tmp_root": str(checkpoint_root)}
        pi0228, config, summary = train_core.build_runtime_config(arm_paths, checkpoint_root / "training", fixed)
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
            "canonical_issue_comment": 5836195300, "pointer_pr_comment": 5836198355,
            "arm": arm, "family": family, "loss": loss, "seed": SEED,
            "d1_runtime_id": "S4", "cohort": bootstrap["cohort"], "schedule": bootstrap["schedule"],
            "input_identity_audit_sha256": bootstrap["input_identity_audit_sha256"],
            "d1_report_sha256": bootstrap["d1_report_sha256"], "d2_report_sha256": bootstrap["d2_report_sha256"],
            "initializer": initializer,
            "loss_contract": bootstrap["loss_contracts"][arm],
            "optimizer": {
                "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS), "learning_rate": 1e-5,
                "weight_decay": 1e-5, "max_grad_norm": 1.0, "batch_size": 1,
                "gradient_accumulation_steps": 1, "workers": 0, "scheduler": "cosine",
                "warmup_steps": 10, "optimizer_state": "fresh AdamW; no resume",
            },
            "resolved_config": summary, "paths": arm_paths, "archive": archive,
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
            or read_json(seal_path).get("sha256") != sha256_file(card_path)):
        raise LUError("run card or seal is not bound to the authorized PI0247 arm")
    return card


def _verify_code(card: dict[str, Any], repo_code: Path, shared_root: Path, training_root: Path) -> None:
    mapping = {
        "pi0247_lu_train.py": repo_code / "scripts/pi0247_lu_train.py",
        "pi0247_u_rollout_overlay.py": repo_code / "scripts/pi0247_u_rollout_overlay.py",
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
    }
    expected = card["source"]["execution_code_sha256"]
    for name, path in mapping.items():
        if sha256_file(path) != expected[name]:
            raise LUError(f"PI0247 training source hash mismatch: {name}")


def _verify_hf_prefix(card: dict[str, Any]) -> dict[str, Any]:
    archive = card["archive"]
    result = _hf_preflight(archive["repo_id"], Path(archive["token_file"]), [archive["path_prefix"]], check_empty=True)
    if not result["public"] or not result["prefixes_empty"]:
        raise LUError("PI0247 HF archive destination did not pass the pre-model gate")
    return result


def _validate_storage(card: dict[str, Any]) -> dict[str, Any]:
    output = Path(card["paths"]["output_root"])
    temporary = Path(card["paths"]["checkpoint_tmp_root"])
    if not str(temporary.resolve()).startswith("/dev/shm/"):
        raise LUError("full checkpoints must be written to node-local tmpfs")
    if (output / "training").exists() or temporary.exists():
        raise LUError("refusing to reuse existing training or temporary checkpoint output")
    scratch = os.statvfs(output.parent)
    scratch_free = scratch.f_bavail * scratch.f_frsize
    tmpfs = os.statvfs("/dev/shm")
    tmpfs_free = tmpfs.f_bavail * tmpfs.f_frsize
    reference_bytes = int(card["initializer"]["full_checkpoint_bytes"])
    required = reference_bytes + max(reference_bytes // 5, 10 * 1024**3)
    if scratch_free < 5 * 1024**3 or tmpfs_free < required:
        raise LUError(f"insufficient pre-model storage: scratch={scratch_free} tmpfs={tmpfs_free} required_tmpfs={required}")
    temporary.mkdir(parents=True)
    return {
        "scratch_free_bytes_before_training": scratch_free,
        "tmpfs_free_bytes_before_training": tmpfs_free,
        "required_tmpfs_bytes": required,
        "checkpoint_tmp_root": str(temporary),
        "full_checkpoints_use_node_local_tmpfs": True,
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


def _install_archive(card: dict[str, Any], train_core: Any, persistence: Any, storage: Any, checkpoint_tmp_root: Path) -> None:
    from transformers import TrainerCallback
    from gr00t.experiment import experiment as experiment_module

    archive = card["archive"]
    output = Path(card["paths"]["output_root"])
    token_path = Path(archive["token_file"])
    sidecar = Path(card["paths"]["code_root"]) / "scripts/pi0243_hf_checkpoint_sidecar.py"
    state_path = output / "hf_archive_state.json"

    class ArchiveOnSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            if step not in CHECKPOINTS:
                return control
            target = Path(args.output_dir) / f"checkpoint-{step}"
            expected_target = checkpoint_tmp_root / "training" / f"checkpoint-{step}"
            if target.resolve() != expected_target.resolve() or target.is_symlink():
                raise LUError("Trainer checkpoint escaped the sealed tmpfs path")
            receipt_path = output / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
            readback_path = output / "hf_readbacks" / f"checkpoint-{step}.json"
            if receipt_path.exists() or readback_path.exists():
                raise LUError(f"checkpoint-{step} archive evidence already exists; refusing overwrite")
            record = persistence.checkpoint_record(
                target, expected_step=step, expected_max_steps=STEPS, family=card["family"],
            )
            checkpoint_link = output / "training" / f"checkpoint-{step}"
            if checkpoint_link.exists() or checkpoint_link.is_symlink():
                raise LUError(f"checkpoint-{step} scratch archive link already exists; refusing overwrite")
            checkpoint_link.symlink_to(target, target_is_directory=True)
            command = [
                sys.executable, str(sidecar), "--output-root", str(output),
                "--repo-id", archive["repo_id"], "--token-file", str(token_path),
                "--path-prefix", archive["path_prefix"], "--steps", str(step),
                "--family", card["family"], "--max-steps", str(STEPS),
                "--state-path", str(state_path), "--checkpoint-timeout", "21600",
            ]
            environment = dict(os.environ)
            environment["HF_HUB_OFFLINE"] = "0"
            environment.pop("TRANSFORMERS_OFFLINE", None)
            delay = 30
            attempt = 0
            while True:
                attempt += 1
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=environment)
                log = output / f"checkpoint-{step}.hf-archive.log"
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"attempt={attempt} exit={result.returncode}\n{result.stdout}\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if result.returncode == 0:
                    break
                lower = result.stdout.lower()
                permanent = (
                    "refusing to overwrite existing archive", "repository is not public",
                    "token file is missing", "token file is empty", "403 client error",
                    "401 client error", "gated repo", "repositorynotfounderror",
                )
                if any(message in lower for message in permanent):
                    raise LUError(f"checkpoint-{step} archive has a non-retryable identity/access failure; local copy retained")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            payload = result.stdout[result.stdout.rfind("\n{") + 1 :].strip()
            archive_result = json.loads(payload)
            if archive_result.get("status") != "COMPLETE":
                raise LUError(f"checkpoint-{step} archive sidecar did not close")
            current = persistence.checkpoint_record(
                target, expected_step=step, expected_max_steps=STEPS, family=card["family"],
            )
            if current != record:
                raise LUError(f"checkpoint-{step} changed during HF upload; local copy retained")
            receipt = read_json(receipt_path)
            train_core.validate_pt_archive_receipt(
                receipt, current, repo_id=archive["repo_id"], prefix=archive["path_prefix"],
                family=card["family"], step=step,
            )
            readback = train_core.verify_pt_hf_readback(
                receipt, current, storage_helper=storage,
                token=token_path.read_text(encoding="utf-8").strip(),
                repo_id=archive["repo_id"], prefix=archive["path_prefix"],
                family=card["family"], step=step,
            )
            readback["schema"] = "PI0247_LU_FULL_HF_READBACK_V1"
            readback["status"] = "PASS_PI0247_LU_FULL_HF_READBACK"
            readback["arm"] = card["arm"]
            readback["source_record_sha256"] = hashlib.sha256(
                json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            write_json(readback_path, readback)
            _retire_after_readback(
                target, output, current, receipt, receipt_path, readback_path,
                readback, archive["repo_id"], archive["path_prefix"], card["family"], step, card["arm"],
            )
            print(f"PI0247 {card['arm']} checkpoint-{step}: public HF full readback verified; tmpfs copy retired")
            return control

    base = experiment_module.Gr00tTrainer

    class ArchiveTrainer(base):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.add_callback(ArchiveOnSave())

    experiment_module.Gr00tTrainer = ArchiveTrainer


def _retire_after_readback(
    target: Path, output: Path, record: dict[str, Any], receipt: dict[str, Any],
    receipt_path: Path, readback_path: Path, readback: dict[str, Any],
    repo_id: str, prefix: str, family: str, step: int, arm: str,
) -> None:
    if (target.is_symlink() or target.resolve().parent != target.parent.resolve()
            or target.resolve() != (target.parent / f"checkpoint-{step}").resolve()):
        raise LUError("checkpoint retirement target escaped its sealed directory")
    checkpoint_link = output / "training" / f"checkpoint-{step}"
    if (not checkpoint_link.is_symlink() or checkpoint_link.resolve() != target.resolve()):
        raise LUError("scratch archive link does not resolve to the sealed tmpfs checkpoint")
    expected = {
        f"{prefix}/checkpoint-{step}/{entry['path']}": (int(entry["bytes"]), entry["sha256"])
        for entry in record["inventory"]["entries"]
    }
    observed = {row["path"]: row for row in readback["files"]}
    if (readback.get("status") != "PASS_PI0247_LU_FULL_HF_READBACK"
            or readback.get("revision") != receipt.get("hf_manifest_commit_id")
            or readback.get("repo_id") != repo_id or readback.get("path_prefix") != prefix
            or readback.get("family") != family or int(readback.get("checkpoint_step", -1)) != step
            or readback.get("source_tree_sha256") != record["inventory"]["tree_sha256"]
            or set(observed) != set(expected)):
        raise LUError(f"checkpoint-{step} lacks complete pinned public-HF readback")
    for path, (size, digest) in expected.items():
        if int(observed[path].get("bytes", -1)) != size or observed[path].get("sha256") != digest:
            raise LUError(f"checkpoint-{step} readback record mismatch: {path}")
    base = output / "checkpoint_retirements"
    intent = base / f"{arm}-checkpoint-{step}.predelete.json"
    completion = base / f"{arm}-checkpoint-{step}.deleted.json"
    if intent.exists() or completion.exists():
        raise LUError(f"checkpoint-{step} already has retirement evidence")
    write_json(intent, {
        "schema": "PI0247_LU_LOCAL_CHECKPOINT_RETIREMENT_V1",
        "status": "DURABLE_HF_COPY_VERIFIED_DELETE_AUTHORIZED",
        "arm": arm, "family": family, "checkpoint_step": step,
        "checkpoint_record": record,
        "checkpoint_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "durable_receipt_sha256": sha256_file(receipt_path),
        "full_hf_readback_sha256": sha256_file(readback_path),
        "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
        "reason": "verified public-HF checkpoint; retain tmpfs capacity for next sealed checkpoint",
    })
    checkpoint_link.unlink()
    shutil.rmtree(target)
    if target.exists():
        raise LUError(f"checkpoint-{step} tmpfs copy remains after retirement")
    write_json(completion, {
        "schema": "PI0247_LU_LOCAL_CHECKPOINT_RETIREMENT_COMPLETE_V1",
        "status": "LOCAL_COPY_REMOVED_AFTER_DURABLE_VERIFICATION",
        "arm": arm, "checkpoint_step": step,
        "predelete_intent_sha256": sha256_file(intent),
        "durable_receipt_sha256": sha256_file(receipt_path),
        "full_hf_readback_sha256": sha256_file(readback_path),
        "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
    })


def train(args: argparse.Namespace) -> None:
    arm = args.arm
    card = _verify_card(args.run_card, arm)
    paths = card["paths"]
    repo_code = Path(paths["repo_code_root"])
    shared_root = Path(paths["code_root"])
    training_root = Path(paths["training_runtime_root"])
    _verify_code(card, repo_code, shared_root, training_root)
    _verify_runtime_bindings(paths, card["source"]["runtime_bindings"])
    _validate_initializer_runtime(card)
    hf = _verify_hf_prefix(card)
    storage = _validate_storage(card)
    output = Path(paths["output_root"])
    training_output = output / "training"
    training_output.mkdir(parents=True)
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
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    train_core = load_module(training_root / "pi0246_gate_pass_train.py", "pi0247_lu_train_core")
    fixed = load_module(shared_root / "scripts/pi0241_fixed16_learning.py", "pi0247_lu_fixed16_train")
    fixed.MAX_STEPS = STEPS
    fixed.CHECKPOINT_STEPS = CHECKPOINTS
    persistence = load_module(shared_root / "scripts/pi0243_checkpoint_persistence.py", "pi0247_lu_persistence")
    storage_helper = load_module(shared_root / "scripts/pi0243_storage_retirement.py", "pi0247_lu_hf_readback")
    paths = card["paths"]
    groot_runtime = Path(paths["stage10_root"]) / "runtime"
    sys.path[:0] = [str(groot_runtime), str(training_root), str(shared_root / "scripts"), str(shared_root / "src"), str(paths["groot_source"])]
    fixed.bind_runtime(card)
    fixed.install_fixed_dataset_factory(card, training_output)
    checkpoint_output = Path(paths["checkpoint_tmp_root"]) / "training"
    fixed.install_update_audit(
        training_output, audit_steps={1, 10, 100, 1000, 2000},
        expected_end_step=STEPS, checkpoint_output=checkpoint_output,
    )
    _install_archive(card, train_core, persistence, storage_helper, Path(paths["checkpoint_tmp_root"]))

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
            parity_path=parity_path, expected_r_star=card["d1_runtime_id"],
        )
        _install_u_chain_check(overlay_audit, output)

    pi0228, config, summary = train_core.build_runtime_config(paths, checkpoint_output, fixed)
    if summary != card["resolved_config"]:
        raise LUError("resolved runtime config changed after card sealing")
    write_json(training_output / "resolved_config.json", summary)
    write_json(output / "execution_state.json", {
        "status": "TRAINING_STARTED", "arm": arm, "family": card["family"],
        "run_card_sha256": sha256_file(args.run_card),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "execution_code_commit": card["source"]["execution_code_commit"],
        "started_epoch_s": time.time(),
    })
    started = time.time()
    pi0228.run_official(config, Path(paths["groot_source"]))

    consumption_path = training_output / "pi0241_sample_consumption.jsonl"
    contracts_path = training_output / "pi0241_sample_contracts.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    contracts = [json.loads(line) for line in contracts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cohort = read_json(Path(card["cohort"]["path"]))["rows"]
    planned = read_json(Path(card["schedule"]["path"]))["rows"]
    if len(consumption) != STEPS or len(planned) != STEPS:
        raise LUError("actual TRAIN sample consumption differs from the sealed 2000-row schedule")
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
    if (len(contracts) != 16 or {int(row["fixed_index"]) for row in contracts} != set(range(16))
            or any(row["active_slots"] != list(range(10)) or row["action_shape"] != [1, 40, 132]
                   or row["mask_shape"] != [1, 40, 132]
                   or row["source_episode_id"] != cohort[int(row["fixed_index"])]["source_episode_id"]
                   for row in contracts)):
        raise LUError("actual fixed16 action/mask contracts are incomplete")
    update_audit_path = training_output / "pi0241_update_audit.json"
    update_audit = read_json(update_audit_path)
    if update_audit.get("status") != "COMPLETE" or int(update_audit.get("global_step", -1)) != STEPS:
        raise LUError("PI0247 update audit did not close at exactly 2000 steps")
    checkpoints = {}
    for step in CHECKPOINTS:
        receipt_path = output / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        readback_path = output / "hf_readbacks" / f"checkpoint-{step}.json"
        intent = output / "checkpoint_retirements" / f"{arm}-checkpoint-{step}.predelete.json"
        completion = output / "checkpoint_retirements" / f"{arm}-checkpoint-{step}.deleted.json"
        receipt = read_json(receipt_path)
        readback = read_json(readback_path)
        if (receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("family") != card["family"]
                or int(receipt.get("checkpoint_step", -1)) != step
                or readback.get("status") != "PASS_PI0247_LU_FULL_HF_READBACK"
                or not intent.is_file() or not completion.is_file()):
            raise LUError(f"checkpoint-{step} full public-HF archive/readback/retirement is incomplete")
        checkpoints[str(step)] = {
            "family": card["family"], "step": step,
            "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
            "total_bytes": receipt["source_record"]["inventory"]["total_bytes"],
            "file_count": receipt["source_record"]["inventory"]["file_count"],
            "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
            "durable_receipt_sha256": sha256_file(receipt_path),
            "full_hf_readback_sha256": sha256_file(readback_path),
            "local_retirement_sha256": sha256_file(completion),
        }
    overlay_rows = [json.loads(line) for line in overlay_audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(overlay_rows) != STEPS or [int(row["forward_index"]) for row in overlay_rows] != list(range(1, STEPS + 1)):
        raise LUError(f"{arm} same-forward objective audit is incomplete")
    required = {"native_loss", "physical_position_loss", "physical_delta_loss", "total_loss"}
    if any(not required.issubset(row) for row in overlay_rows):
        raise LUError(f"{arm} objective component audit fields are incomplete")
    for row in overlay_rows:
        values = [float(row[key]) for key in ("native_loss", "physical_position_loss", "physical_delta_loss", "total_loss")]
        if not all(math.isfinite(value) for value in values):
            raise LUError(f"{arm} objective audit has nonfinite loss")
        prescribed = values[0] + LAMBDA_POS * values[1] + LAMBDA_DELTA * values[2]
        if not math.isclose(values[3], prescribed, rel_tol=1e-5, abs_tol=1e-7):
            raise LUError(f"{arm} objective terms do not sum to the sealed loss")
    parameters = update_audit["final_parameters"]
    integrity = {
        "projector_updated": float(parameters["projector"]["delta_l2"]) > 0,
        "diffusion_updated": float(parameters["diffusion"]["delta_l2"]) > 0,
        "frozen_backbone_unchanged": float(parameters["frozen_backbone"]["delta_l2"]) == 0,
        "frozen_vlln_unchanged": float(parameters["frozen_vlln"]["delta_l2"]) == 0,
        "optimizer_membership_valid": not update_audit["optimizer_membership"]["missing_trainable"] and not update_audit["optimizer_membership"]["illegal_frozen"],
    }
    if arm == "U":
        parity = read_json(output / "pi0247_U_sampler_parity.json")
        chain = read_json(output / "u_first_update_chain_gradient.json")
        if parity.get("status") != "PASS_PI0247_U_S4_PARITY" or chain.get("status") != "PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN":
            raise LUError("U exact-runtime parity or first-update Euler chain-gradient gate failed")
        if overlay_audit["parity_nonselection_rollouts"] != 2:
            raise LUError("U sampler parity count differs from the sealed two-rollout pair")
        evaluations = overlay_audit["head_evaluations"]
        expected_evaluations = {
            "parity_reference": 4, "parity_candidate": 4,
            "rollout_forward": 4 * STEPS, "backward_recompute": 4 * STEPS,
        }
        if (overlay_audit["training_rollouts"] != STEPS or overlay_audit["native_forwards"] != STEPS
                or evaluations != expected_evaluations):
            raise LUError(f"U full Euler forward/recompute counts do not close: {overlay_audit}")
        if not (overlay_audit["mode_restoration_forward_checks"] == STEPS
                and overlay_audit["mode_restoration_after_backward_checks"] == STEPS):
            raise LUError("U action-head train/eval modes were not restored after all forward/backward passes")
        integrity["full_euler_gradient_chain_verified"] = True
        integrity["exact_s4_parity_verified"] = True
        integrity["action_head_eval_autograd_mode_restored"] = True
    result = {
        "schema": "PI0247_LU_TRAINING_MANIFEST_V1",
        "status": "PASS_TRAINING_INTEGRITY" if all(integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "arm": arm, "family": card["family"], "authorization": AUTHORIZATION,
        "run_card_sha256": sha256_file(args.run_card),
        "execution_code_commit": card["source"]["execution_code_commit"],
        "pbs_job_id": os.environ.get("PBS_JOBID"), "global_step": STEPS,
        "sample_consumption": len(consumption), "sample_contracts": contracts,
        "schedule_sha256": card["schedule"]["sha256"], "cohort_sha256": card["cohort"]["sha256"],
        "elapsed_s": time.time() - started, "runtime_id": "S4",
        "resolved_config": summary, "update_integrity": integrity,
        "update_audit_sha256": sha256_file(update_audit_path),
        "objective_audit": {
            "path": str(overlay_audit_path), "sha256": sha256_file(overlay_audit_path),
            "rows": len(overlay_rows), "first": overlay_rows[0], "last": overlay_rows[-1],
            "loss_coefficients": {"ell_pos": LAMBDA_POS, "ell_delta": LAMBDA_DELTA},
        },
        "rollout_audit": overlay_audit if arm == "U" else None,
        "parity": read_json(output / "pi0247_U_sampler_parity.json") if arm == "U" else None,
        "checkpoints": checkpoints, "selected_checkpoint": "NONE",
        "archive_repo_id": card["archive"]["repo_id"], "archive_prefix": card["archive"]["path_prefix"],
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json(output / "training_manifest.json", result)
    write_json(output / "execution_state.json", {
        "status": "TRAINING_COMPLETE", "arm": arm, "family": card["family"],
        "training_manifest_sha256": sha256_file(output / "training_manifest.json"),
        "completed_epoch_s": time.time(), "selected_checkpoint": "NONE",
    })
    print(json.dumps({"status": result["status"], "arm": arm, "steps": STEPS, "checkpoints": checkpoints}, indent=2, sort_keys=True))


def _install_u_chain_check(audit: dict[str, Any], output: Path) -> None:
    from transformers import TrainerCallback
    from gr00t.experiment import experiment as experiment_module

    base = experiment_module.Gr00tTrainer

    class EulerChainAudit(TrainerCallback):
        def __init__(self, action_head: Any):
            self.action_head = action_head

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            expected_modes = audit.get("native_mode_snapshot")
            if expected_modes is None or [bool(child.training) for child in self.action_head.modules()] != expected_modes:
                raise LUError("U action-head modes were not restored after the Euler backward recomputation")
            audit["mode_restoration_after_backward_checks"] += 1
            if int(state.global_step) + 1 != 1:
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
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--arm", choices=tuple(PROFILES))
    args = parser.parse_args()
    if args.prepare:
        if args.bootstrap is None:
            parser.error("--prepare requires --bootstrap")
        prepare(args)
    else:
        if args.run_card is None or args.arm is None:
            parser.error("--train requires --run-card and --arm")
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
