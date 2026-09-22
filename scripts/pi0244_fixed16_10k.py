#!/usr/bin/env python3
"""PI0244 thin preset around the sealed PI0243 trainer/screen path.

The mature implementation remains the source of truth.  This wrapper only
binds the new fixed16/10K identity, schedule, family, and terminal closure.
"""

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
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


PROTOCOL_COMMIT = "d23727c2b7764589573f6c7a7195d091980a9bbc"
AUDITED_PARENT = "69b587fe4edc3a7a0c9ff99240ba0214bc6e6e86"
R0_SOURCE_COMMIT = "5169dffaab888c9f5ec1953f541d0dc306f1a00c"
SEALED_LEGACY_SHA256 = "fdbed1db143163ad6a5c50c5c24b50aee1d66e6759ac00c6ce94b5f16aacf92d"
AUTHORIZATION = "PI0244_M16_10K_GO"
EXPERIMENT = "PI0244_M16_10K_V1"
FAMILY = EXPERIMENT
MAX_STEPS = 10000
CHECKPOINTS = [2500, 5000, 7500, 10000]
SCREENS = [0, 2500, 5000, 7500, 10000]
PREFIX_STEPS = 5000
RECOVERY_STORAGE_AUTH_COMMENT = "5782281194"


class PI0244Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base(code_root: Path) -> Any:
    path = code_root / "scripts" / "pi0243_staged_learning_to_sim.py"
    spec = importlib.util.spec_from_file_location("pi0244_pi0243_base", path)
    if spec is None or spec.loader is None:
        raise PI0244Error(f"cannot import mature stage path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    module.AUDITED_PARENT = AUDITED_PARENT
    module.AUTH = dict(module.AUTH)
    module.AUTH["a"] = AUTHORIZATION
    module.stage_spec = stage_spec
    return module


def stage_spec(stage: str) -> dict[str, Any]:
    if str(stage).lower() == "a":
        return {
            "stage": "A",
            "view": "16",
            "max_steps": MAX_STEPS,
            "checkpoints": list(CHECKPOINTS),
            "screens": list(SCREENS),
        }
    raise PI0244Error("PI0244 only permits the fixed16 10K stage")


def _config_differences(parent: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}

    def walk(left: Any, right: Any, prefix: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
            return
        if left != right:
            differences[prefix] = {"parent": left, "current": right}

    walk(parent, current, "")
    return differences


ALLOWED_CONFIG_DIFFS = {
    "max_steps",
    "save_total_limit",
    "checkpoint_steps",
    "screen_steps",
}


def validate_schedule_rows(current: list[Mapping[str, Any]], parent: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(current) != MAX_STEPS:
        raise PI0244Error(f"10K schedule length mismatch: {len(current)}")
    if current[:PREFIX_STEPS] != parent[:PREFIX_STEPS]:
        raise PI0244Error("10K schedule prefix differs from the sealed 5K schedule")
    counts = Counter(str(row["source_episode_id"]) for row in current)
    if len(counts) != 16 or set(counts.values()) != {625}:
        raise PI0244Error(f"fixed16 exposure counts are not 625 each: {dict(counts)}")
    return {"length": len(current), "prefix_steps": PREFIX_STEPS, "source_counts": dict(sorted(counts.items()))}


def _read_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PI0244Error(f"schedule rows missing: {path}")
    return rows


def _patch_card(base: Any, args: argparse.Namespace) -> Path:
    card_path = args.output_root / "pi0243_run_card.json"
    card = base.read_json(card_path)
    parent_card = base.read_json(args.parent_card)
    current_config = card["resolved_config"]
    parent_config = parent_card["resolved_config"]
    differences = _config_differences(parent_config, current_config)
    unexpected = sorted(key for key in differences if key not in ALLOWED_CONFIG_DIFFS)
    if unexpected:
        raise PI0244Error(f"unexpected resolved-config drift: {unexpected}")

    schedule_check = validate_schedule_rows(_read_rows(card["schedule"]["path"]), _read_rows(args.prefix_schedule))
    diff_path = args.output_root / "resolved_diff.json"
    diff = {
        "schema": "PI0244_RESOLVED_DIFF_V1",
        "status": "PASS",
        "protocol_commit": PROTOCOL_COMMIT,
        "experiment": EXPERIMENT,
        "parent_card": str(args.parent_card),
        "parent_card_sha256": sha256_file(args.parent_card),
        "differences": differences,
        "allowed_difference_keys": sorted(ALLOWED_CONFIG_DIFFS),
        "schedule": schedule_check,
    }
    base.write_json(diff_path, diff)

    card["experiment"] = EXPERIMENT
    card["protocol_commit"] = PROTOCOL_COMMIT
    card["audited_parent"] = AUDITED_PARENT
    card["stage_role"] = "PI0244_FIXED16_10K"
    card["resolved_diff"] = {"path": str(diff_path), "sha256": sha256_file(diff_path)}
    card["implementation"]["preset_path"] = str(Path(__file__).resolve())
    card["implementation"]["preset_sha256"] = sha256_file(Path(__file__).resolve())
    card["storage"]["checkpoint_family"] = FAMILY
    card["storage"]["capacity_binding"] = {
        "full_checkpoint_count": 4,
        "retain_all_local_hf_durable": True,
        "checkpoint_steps": list(CHECKPOINTS),
    }
    card["screen_binding"] = {
        "unique_cells": len(SCREENS) * 16,
        "screen_steps": list(SCREENS),
        "final_step": MAX_STEPS,
        "final_only_gate": True,
    }
    base.write_json(card_path, card)
    base.write_json(card_path.with_name("pi0243_run_card.seal.json"), {"schema": "PI0243_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(card_path)})
    return card_path


def prepare(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise PI0244Error(f"refusing to reuse output root: {args.output_root}")
    base = load_base(args.code_root)
    base_args = SimpleNamespace(
        stage="a",
        code_root=args.code_root,
        groot_source=args.groot_source,
        stage10_root=args.stage10_root,
        train_dataset=args.train_dataset,
        p237_root=args.p237_root,
        modality_config=args.modality_config,
        processor_base_model=args.processor_base_model,
        initializer=args.initializer,
        cosmos_snapshot=args.cosmos_snapshot,
        cosmos_cache_root=args.cosmos_cache_root,
        output_root=args.output_root,
        durable_host=None,
        durable_root=None,
        hf_repo_id=args.hf_repo_id,
        hf_token_file=args.hf_token_file,
        hf_path_prefix=args.hf_path_prefix,
    )
    base.make_card(base_args)
    card_path = _patch_card(base, args)
    print(json.dumps({"status": "SEALED_READY_TO_EXECUTE", "experiment": EXPERIMENT, "run_card": str(card_path), "run_card_sha256": sha256_file(card_path)}, indent=2))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _original_legacy_sha256(code_root: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{R0_SOURCE_COMMIT}:scripts/pi0241_fixed16_learning.py"],
        cwd=code_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _execution_binding(card_path: Path, card: Mapping[str, Any], runner_path: Path | None = None) -> dict[str, Any]:
    code_root = Path(card["paths"]["code_root"])
    legacy_path = Path(card["implementation"]["legacy_path"])
    pinned = str(card["implementation"]["legacy_sha256"])
    original = _original_legacy_sha256(code_root)
    if pinned != SEALED_LEGACY_SHA256 or original != pinned:
        raise PI0244Error("R0 sealed legacy source does not match its audited Git object")
    payload = {
        "schema": "PI0244_EXECUTION_OVERLAY_BINDING_V1",
        "run_card_sha256": sha256_file(card_path),
        "sealed_legacy_sha256": pinned,
        "audited_legacy_git_sha256": original,
        "active_legacy_sha256": sha256_file(legacy_path),
        "active_runner_sha256": sha256_file(runner_path or Path(__file__).resolve()),
        "r0_source_commit": R0_SOURCE_COMMIT,
        "protocol_commit": PROTOCOL_COMMIT,
    }
    payload["binding_id"] = _json_sha256(payload)
    return payload


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != data:
            raise PI0244Error(f"refusing to overwrite sealed evidence: {path}")


def _receipt_repo(receipt: Mapping[str, Any], card: Mapping[str, Any], step: int) -> str:
    prefix = str(card["storage"]["path_prefix"]).strip("/")
    durable_path = str(receipt.get("durable_path", ""))
    suffix = f"/{prefix}/checkpoint-{step}"
    if not durable_path.endswith(suffix):
        raise PI0244Error(f"checkpoint-{step} durable path differs from the sealed family prefix")
    repo_id = durable_path[: -len(suffix)]
    if repo_id.count("/") != 1:
        raise PI0244Error(f"checkpoint-{step} durable path has an invalid repository identity")
    return repo_id


def _storage_binding(root: Path, card: Mapping[str, Any], card_path: Path, *, create: bool = False) -> dict[str, Any]:
    path = root / "pi0244_recovery_storage_binding.json"
    seal_path = path.with_suffix(".seal.json")
    if not path.is_file():
        if not create:
            raise PI0244Error("append-only recovery storage binding is missing")
        code_root = Path(card["paths"]["code_root"])
        persistence = load_base(code_root).load_module(code_root / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
        source_receipts: dict[str, Any] = {}
        repo_ids = set()
        for step in (2500, 5000):
            checkpoint = root / "training" / f"checkpoint-{step}"
            receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
            receipt = persistence.verify_receipt(checkpoint, receipt_path, expected_step=step, expected_max_steps=MAX_STEPS, family=FAMILY)
            if receipt.get("status") != "DURABLE_VERIFIED" or not receipt.get("hf_manifest_commit_id"):
                raise PI0244Error(f"checkpoint-{step} has no complete durable HF receipt")
            repo_ids.add(_receipt_repo(receipt, card, step))
            inventory = receipt.get("source_record", {}).get("inventory", {})
            if len(receipt.get("durable_inventory", [])) != int(inventory.get("file_count", -1)):
                raise PI0244Error(f"checkpoint-{step} durable inventory is incomplete")
            source_receipts[str(step)] = {
                "receipt_sha256": sha256_file(receipt_path),
                "durable_path": receipt["durable_path"],
                "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
                "tree_sha256": inventory["tree_sha256"],
                "file_count": inventory["file_count"],
                "total_bytes": inventory["total_bytes"],
            }
        if len(repo_ids) != 1:
            raise PI0244Error("checkpoint-2500 and checkpoint-5000 receipts do not share one HF repository")
        payload = {
            "schema": "PI0244_R0_RECOVERY_STORAGE_BINDING_V1",
            "status": "SEALED_APPEND_ONLY_RECOVERY_BINDING",
            "authorization_issue_comment_id": RECOVERY_STORAGE_AUTH_COMMENT,
            "run_card_sha256": sha256_file(card_path),
            "card_repo_id": str(card["storage"]["repo_id"]),
            "effective_repo_id": next(iter(repo_ids)),
            "path_prefix": str(card["storage"]["path_prefix"]).strip("/"),
            "family": FAMILY,
            "source_receipts": source_receipts,
            "reason": "continue R0 at the already receipt-verified public archive destination without rewriting the sealed run card",
        }
        _write_once(path, payload)
        _write_once(seal_path, {"schema": "PI0244_RECOVERY_STORAGE_BINDING_SEAL_V1", "sha256": sha256_file(path)})
    binding = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("sha256") != sha256_file(path)
        or binding.get("status") != "SEALED_APPEND_ONLY_RECOVERY_BINDING"
        or binding.get("authorization_issue_comment_id") != RECOVERY_STORAGE_AUTH_COMMENT
        or binding.get("run_card_sha256") != sha256_file(card_path)
        or binding.get("card_repo_id") != card["storage"]["repo_id"]
        or binding.get("path_prefix") != str(card["storage"]["path_prefix"]).strip("/")
        or binding.get("family") != FAMILY
    ):
        raise PI0244Error("append-only recovery storage binding does not match the sealed card")
    for step in (2500, 5000):
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        pinned = binding.get("source_receipts", {}).get(str(step), {})
        if (
            pinned.get("receipt_sha256") != sha256_file(receipt_path)
            or receipt.get("durable_path") != pinned.get("durable_path")
            or _receipt_repo(receipt, card, step) != binding.get("effective_repo_id")
            or receipt.get("hf_manifest_commit_id") != pinned.get("hf_manifest_commit_id")
            or receipt.get("source_record", {}).get("inventory", {}).get("tree_sha256") != pinned.get("tree_sha256")
        ):
            raise PI0244Error(f"checkpoint-{step} differs from the sealed recovery storage binding")
    return binding


def _assert_archive_environment(plan: Mapping[str, Any]) -> None:
    storage = plan.get("storage", {})
    if os.environ.get("PI0243_HF_REPO_ID") != storage.get("repo_id") or os.environ.get("PI0243_HF_PATH_PREFIX", "").strip("/") != storage.get("path_prefix"):
        raise PI0244Error("PBS HF repo/prefix do not match the sealed recovery storage plan")


def verify_card(
    path: Path,
    *,
    bind_overlay: bool = False,
    runner_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    code_root = Path(raw["paths"]["code_root"])
    base = load_base(code_root)
    seal = json.loads(path.with_name("pi0243_run_card.seal.json").read_text(encoding="utf-8"))
    if seal.get("run_card_sha256") != sha256_file(path):
        raise PI0244Error("run-card seal hash mismatch")
    if raw.get("schema") != "PI0243_RUN_CARD_V1" or raw.get("status") != "SEALED_READY_TO_EXECUTE":
        raise PI0244Error("run card is not a sealed PI0243 card")
    implementation = raw["implementation"]
    for path_key, hash_key in (
        ("script_path", "script_sha256"),
        ("pi0228_path", "pi0228_sha256"),
        ("persistence_path", "persistence_sha256"),
    ):
        if sha256_file(Path(implementation[path_key])) != implementation[hash_key]:
            raise PI0244Error(f"pinned implementation changed: {path_key}")
    if implementation.get("legacy_sha256") != SEALED_LEGACY_SHA256:
        raise PI0244Error("unexpected sealed fixed16 helper identity")
    legacy_path = Path(implementation["legacy_path"])
    current_legacy_sha = sha256_file(legacy_path)
    if current_legacy_sha != SEALED_LEGACY_SHA256:
        binding = _execution_binding(path, raw, runner_path=runner_path)
        runtime_root = Path(base.runtime_card(raw)["paths"]["output_root"])
        binding_path = runtime_root / "execution_bindings" / f"{binding['binding_id']}.json"
        if bind_overlay:
            _write_once(binding_path, binding)
        elif not binding_path.is_file() or json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise PI0244Error("active recovery-code overlay has no matching immutable execution binding")
    for key in ("cohort", "schedule"):
        mapped = base._mapped_stage_path(raw, raw[key]["path"], "PI0243_METADATA_MIRROR_ROOT")
        if not mapped.is_file() or sha256_file(mapped) != raw[key]["sha256"]:
            raise PI0244Error(f"sealed {key} identity mismatch")
    legacy = base.legacy_module(code_root)
    if legacy._model_files(Path(raw["paths"]["initializer"])) != raw["initializer"]["files"]:
        raise PI0244Error("sealed initializer files changed")
    card = raw
    if card.get("experiment") != EXPERIMENT or card.get("stage_role") != "PI0244_FIXED16_10K":
        raise PI0244Error("PI0244 experiment identity mismatch")
    if card["stage"] != stage_spec("a"):
        raise PI0244Error("PI0244 fixed16/10K stage binding mismatch")
    if card.get("storage", {}).get("checkpoint_family") != FAMILY:
        raise PI0244Error("PI0244 checkpoint family mismatch")
    diff = card.get("resolved_diff", {})
    diff_path = Path(diff.get("path", ""))
    diff_path = base._mapped_stage_path(card, diff_path, "PI0243_METADATA_MIRROR_ROOT")
    if not diff_path.is_file() or sha256_file(diff_path) != diff.get("sha256"):
        raise PI0244Error("resolved_diff is missing or changed")
    payload = json.loads(diff_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("experiment") != EXPERIMENT:
        raise PI0244Error("resolved_diff is not PASS")
    return base, card


def _checkpoint_paths(training_root: Path) -> dict[int, Path]:
    found = {}
    for child in training_root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if child.is_dir() and match:
            found[int(match.group(1))] = child
    return found


def _fixed_schedule(base: Any, card: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = base._mapped_stage_path(card, card["schedule"]["path"], "PI0243_METADATA_MIRROR_ROOT")
    rows = _read_rows(path)
    if len(rows) != MAX_STEPS or any(int(row.get("optimizer_step", -1)) != index for index, row in enumerate(rows, 1)):
        raise PI0244Error("sealed fixed16 schedule has an invalid step cursor")
    return rows


def _seal_recovery_binding(base: Any, card: Mapping[str, Any], card_path: Path) -> dict[str, Any]:
    binding = _execution_binding(card_path, card)
    root = Path(base.runtime_card(card)["paths"]["output_root"])
    _write_once(root / "execution_bindings" / f"{binding['binding_id']}.json", binding)
    return binding


def plan_recovery(args: argparse.Namespace) -> Path:
    if (args.resume_step, args.stop_step) not in {(5000, 7500), (7500, 10000)}:
        raise PI0244Error("only the authorized R0 recovery segments 5000→7500 and 7500→10000 are allowed")
    base, card = verify_card(args.run_card, bind_overlay=True)
    binding = _seal_recovery_binding(base, card, args.run_card)
    runtime = base.runtime_card(card)
    root = Path(runtime["paths"]["output_root"])
    storage_binding = _storage_binding(root, card, args.run_card, create=True)
    training_root = root / "training"
    resume_step, stop_step = args.resume_step, args.stop_step
    checkpoints = _checkpoint_paths(training_root)
    if max(checkpoints, default=-1) != resume_step:
        raise PI0244Error(f"latest local checkpoint must be exactly {resume_step}; found {sorted(checkpoints)}")
    if stop_step in checkpoints or (training_root / f"checkpoint-{stop_step}").exists():
        raise PI0244Error(f"recovery destination checkpoint-{stop_step} already exists; preserve and quarantine before planning")
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    source_checkpoint = checkpoints[resume_step]
    source_record = legacy.checkpoint_record(source_checkpoint, resume_step)
    receipt_path = root / "durable_receipts" / f"checkpoint-{resume_step}.durable_receipt.json"
    persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    receipt = persistence.verify_receipt(
        source_checkpoint,
        receipt_path,
        expected_step=resume_step,
        expected_max_steps=MAX_STEPS,
        family=FAMILY,
    )
    rows = _fixed_schedule(base, card)
    selected_rows = rows[resume_step:stop_step]
    if len(selected_rows) != stop_step - resume_step:
        raise PI0244Error("recovery schedule suffix length mismatch")
    plan = {
        "schema": "PI0244_R0_RECOVERY_PLAN_V1",
        "status": "SEALED_READY_TO_EXECUTE",
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "segment": {"resume_step": resume_step, "stop_step": stop_step},
        "run_card_sha256": sha256_file(args.run_card),
        "execution_binding_id": binding["binding_id"],
        "r0_source_commit": R0_SOURCE_COMMIT,
        "active_runner_sha256": binding["active_runner_sha256"],
        "active_legacy_sha256": binding["active_legacy_sha256"],
        "schedule_sha256": card["schedule"]["sha256"],
        "schedule_suffix_sha256": _json_sha256(selected_rows),
        "schedule_rows": len(selected_rows),
        "resolved_config_sha256": _json_sha256(card["resolved_config"]),
        "resume_checkpoint": {
            "path": str(source_checkpoint),
            "tree_sha256": source_record["tree_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
            "durable_tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
            "hf_manifest_commit_id": receipt.get("hf_manifest_commit_id"),
        },
        "storage": {
            "repo_id": storage_binding["effective_repo_id"],
            "path_prefix": storage_binding["path_prefix"],
            "binding_sha256": sha256_file(root / "pi0244_recovery_storage_binding.json"),
            "family": FAMILY,
        },
        "allowed_config_diff": {"training.resume_from_checkpoint": {"sealed": False, "runtime": True}},
        "target_xyz_in_model_input": False,
        "selected_checkpoint": "NONE",
    }
    segment_id = f"segment-{resume_step}-to-{stop_step}-{binding['binding_id'][:12]}"
    plan_path = root / "recovery_plans" / f"{segment_id}.json"
    _write_once(plan_path, plan)
    _write_once(plan_path.with_suffix(".seal.json"), {"schema": "PI0244_RECOVERY_PLAN_SEAL_V1", "sha256": sha256_file(plan_path)})
    return plan_path


def _load_recovery_plan(
    plan_path: Path,
    run_card: Path,
    runner_path: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path]:
    base, card = verify_card(run_card, runner_path=runner_path)
    runtime = base.runtime_card(card)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seal = json.loads(plan_path.with_suffix(".seal.json").read_text(encoding="utf-8"))
    if seal.get("sha256") != sha256_file(plan_path) or plan.get("status") != "SEALED_READY_TO_EXECUTE":
        raise PI0244Error("recovery plan seal/status mismatch")
    if plan.get("run_card_sha256") != sha256_file(run_card) or plan.get("protocol_commit") != PROTOCOL_COMMIT:
        raise PI0244Error("recovery plan is bound to a different run card or protocol")
    binding = _execution_binding(run_card, card, runner_path=runner_path)
    if plan.get("execution_binding_id") != binding["binding_id"]:
        raise PI0244Error("recovery source code changed after plan seal")
    storage_binding = _storage_binding(Path(runtime["paths"]["output_root"]), card, run_card)
    if plan.get("storage") != {
        "repo_id": storage_binding["effective_repo_id"],
        "path_prefix": storage_binding["path_prefix"],
        "binding_sha256": sha256_file(Path(runtime["paths"]["output_root"]) / "pi0244_recovery_storage_binding.json"),
        "family": FAMILY,
    }:
        raise PI0244Error("recovery plan storage identity differs from its append-only binding")
    segment = plan.get("segment", {})
    resume_step, stop_step = int(segment.get("resume_step", -1)), int(segment.get("stop_step", -1))
    if (resume_step, stop_step) not in {(5000, 7500), (7500, 10000)}:
        raise PI0244Error("recovery plan contains an unauthorized segment")
    rows = _fixed_schedule(base, card)
    if plan.get("schedule_sha256") != card["schedule"]["sha256"] or plan.get("schedule_suffix_sha256") != _json_sha256(rows[resume_step:stop_step]):
        raise PI0244Error("recovery schedule suffix changed after plan seal")
    return base, card, plan, Path(runtime["paths"]["output_root"])


def validate_recovery_plan(args: argparse.Namespace) -> None:
    base, card, plan, root = _load_recovery_plan(args.plan, args.run_card)
    _assert_archive_environment(plan)
    resume_step, stop_step = (int(plan["segment"][key]) for key in ("resume_step", "stop_step"))
    training_root = root / "training"
    checkpoints = _checkpoint_paths(training_root)
    if max(checkpoints, default=-1) != resume_step or (training_root / f"checkpoint-{stop_step}").exists():
        raise PI0244Error("recovery source/destination checkpoint state differs from the sealed plan")
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    source = training_root / f"checkpoint-{resume_step}"
    record = legacy.checkpoint_record(source, resume_step)
    receipt_path = root / "durable_receipts" / f"checkpoint-{resume_step}.durable_receipt.json"
    persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    receipt = persistence.verify_receipt(source, receipt_path, expected_step=resume_step, expected_max_steps=MAX_STEPS, family=FAMILY)
    if record["tree_sha256"] != plan["resume_checkpoint"]["tree_sha256"] or sha256_file(receipt_path) != plan["resume_checkpoint"]["receipt_sha256"] or receipt.get("durable_path") != f"{plan['storage']['repo_id']}/{plan['storage']['path_prefix']}/checkpoint-{resume_step}":
        raise PI0244Error("resume checkpoint or durable receipt differs from the sealed plan")
    print(json.dumps({"status": "PASS_RECOVERY_PLAN_PREFLIGHT", "plan_sha256": sha256_file(args.plan), "resume_step": resume_step, "stop_step": stop_step, "resume_tree_sha256": record["tree_sha256"], "hf_repo_id": plan["storage"]["repo_id"], "hf_path_prefix": plan["storage"]["path_prefix"], "model_load": False, "forward": False}, sort_keys=True))


def _boundary_checkpoint_complete(output_root: Path, stop_step: int) -> bool:
    checkpoint = output_root / f"checkpoint-{stop_step}"
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        required = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
        return (
            int(state.get("global_step", -1)) == stop_step
            and int(state.get("max_steps", -1)) == MAX_STEPS
            and bool(shards)
            and all(Path(name).name == name and (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in shards | required)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _guard_final_checkpoint_export(original: Any, stop_step: int) -> Any:
    def guarded(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        output_dir = kwargs.get("output_dir", args[0] if args else None)
        internal_call = kwargs.get("_internal_call", args[1] if len(args) > 1 else False)
        if (
            output_dir is None
            and not internal_call
            and int(getattr(getattr(trainer, "state", None), "global_step", -1)) == stop_step
            and _boundary_checkpoint_complete(Path(trainer.args.output_dir), stop_step)
        ):
            print(json.dumps({"status": "PI0244_REDUNDANT_FINAL_EXPORT_SKIPPED", "checkpoint_step": stop_step, "checkpoint_root": str(Path(trainer.args.output_dir) / f"checkpoint-{stop_step}")}, sort_keys=True))
            return None
        return original(trainer, *args, **kwargs)

    return guarded


def _run_recovery_training(pi0228: Any, config: Any, groot_source: Path, stop_step: int) -> None:
    from transformers.trainer import Trainer

    original = Trainer.save_model
    Trainer.save_model = _guard_final_checkpoint_export(original, stop_step)
    try:
        pi0228.run_official(config, groot_source)
    finally:
        Trainer.save_model = original


def _close_recovery_segment(
    args: argparse.Namespace,
    base: Any,
    card: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
    job_id: str,
    attempt_root: Path,
    reconciliation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resume_step, stop_step = (int(plan["segment"][key]) for key in ("resume_step", "stop_step"))
    result_path = attempt_root / "segment_result.json"
    if result_path.exists():
        raise PI0244Error("recovery segment result already exists; refusing to overwrite first-valid evidence")
    runtime = base.runtime_card(card)
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    legacy.bind_runtime(card)
    pi0228 = base.pi0228_module(Path(card["paths"]["code_root"]))
    _, summary = base.resolved_config(runtime, attempt_root, legacy, pi0228)
    if summary != card["resolved_config"]:
        raise PI0244Error("recovery resolved config differs from sealed R0 recipe")

    rows = _fixed_schedule(base, card)[resume_step:stop_step]
    consumption_path = attempt_root / "pi0241_sample_consumption.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")
    if len(consumption) != len(rows) or any(any(actual.get(key) != planned.get(key) for key in fields) or int(actual.get("yield_ordinal", -1)) != int(planned["optimizer_step"]) for actual, planned in zip(consumption, rows)):
        raise PI0244Error("recovered sample consumption differs from the sealed schedule suffix")

    contracts_path = attempt_root / "pi0241_sample_contracts.jsonl"
    contracts = [json.loads(line) for line in contracts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(contracts) != 16 or {int(row.get("fixed_index", -1)) for row in contracts} != set(range(16)) or any(row.get("active_slots") != list(range(10)) or row.get("action_shape") != [1, 40, 132] or row.get("mask_shape") != [1, 40, 132] for row in contracts):
        raise PI0244Error("recovery action/mask contract differs from sealed fixed16 H40 dense-relative ABI")
    training_root = root / "training"
    original_contracts_path = training_root / "pi0241_sample_contracts.jsonl"
    original_contracts = [json.loads(line) for line in original_contracts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    original_by_index = {int(row["fixed_index"]): row for row in original_contracts}
    if len(original_by_index) != 16 or set(original_by_index) != set(range(16)):
        raise PI0244Error("original attempt does not contain the complete fixed16 action/mask contract")
    for row in contracts:
        original = original_by_index[int(row["fixed_index"])]
        if any(row.get(key) != original.get(key) for key in ("source_episode_id", "action_sha256", "mask_sha256", "active_slots", "action_shape", "mask_shape")):
            raise PI0244Error("resumed TRAIN action/mask differs from the original fixed-window contract")

    audit_path = attempt_root / "pi0241_update_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != stop_step or int(audit.get("expected_end_step", -1)) != stop_step:
        raise PI0244Error("recovery optimizer/update audit did not reach the planned segment boundary")
    membership = audit.get("optimizer_membership", {})
    final_parameters = audit.get("final_parameters", {})
    update_integrity = {
        "projector_updated": float(final_parameters.get("projector", {}).get("delta_l2", 0.0)) > 0.0,
        "diffusion_updated": float(final_parameters.get("diffusion", {}).get("delta_l2", 0.0)) > 0.0,
        "frozen_backbone_unchanged": float(final_parameters.get("frozen_backbone", {}).get("delta_l2", 1.0)) == 0.0,
        "frozen_vlln_unchanged": float(final_parameters.get("frozen_vlln", {}).get("delta_l2", 1.0)) == 0.0,
        "optimizer_membership_valid": not membership.get("missing_trainable") and not membership.get("illegal_frozen"),
    }
    first_gradient = next((row for row in audit.get("events", []) if row.get("event") == "PRE_OPTIMIZER" and int(row.get("step", -1)) == resume_step + 1), None)
    if first_gradient is None or not first_gradient.get("gradient_finite") or not all(update_integrity.values()):
        raise PI0244Error("recovery gradient/update or frozen-parameter integrity failed")

    target = training_root / f"checkpoint-{stop_step}"
    state = json.loads((target / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != stop_step or int(state.get("max_steps", -1)) != MAX_STEPS or not _boundary_checkpoint_complete(training_root, stop_step):
        raise PI0244Error("recovery boundary checkpoint is incomplete or has the wrong Trainer state")
    checkpoint = legacy.checkpoint_record(target, stop_step)
    result = {
        "schema": "PI0244_R0_RECOVERY_SEGMENT_V1",
        "status": "TRAINED_AWAITING_DURABLE_ARCHIVE",
        "experiment": EXPERIMENT,
        "plan_sha256": sha256_file(args.plan),
        "job_id": job_id,
        "segment": plan["segment"],
        "resume_checkpoint_tree_sha256": plan["resume_checkpoint"]["tree_sha256"],
        "resume_step": resume_step,
        "stop_step": stop_step,
        "schedule_suffix_sha256": plan["schedule_suffix_sha256"],
        "consumed_rows": len(consumption),
        "consumption_sha256": sha256_file(consumption_path),
        "contracts_count": len(contracts),
        "contracts_sha256": sha256_file(contracts_path),
        "update_audit_sha256": sha256_file(audit_path),
        "update_integrity": update_integrity,
        "first_gradient": first_gradient,
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "checkpoint_bytes": checkpoint["total_bytes"],
        "resolved_config_sha256": _json_sha256(summary),
        "resume_from_checkpoint": True,
        "selected_checkpoint": "NONE",
    }
    if reconciliation is not None:
        result["post_run_reconciliation"] = reconciliation
    _write_once(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def recover(args: argparse.Namespace) -> None:
    base, card, plan, root = _load_recovery_plan(args.plan, args.run_card)
    _assert_archive_environment(plan)
    runtime = base.runtime_card(card)
    resume_step, stop_step = (int(plan["segment"][key]) for key in ("resume_step", "stop_step"))
    training_root = root / "training"
    checkpoints = _checkpoint_paths(training_root)
    if max(checkpoints, default=-1) != resume_step or (training_root / f"checkpoint-{stop_step}").exists():
        raise PI0244Error("recovery source is no longer the latest complete checkpoint or destination is occupied")
    job_id = os.environ.get("PBS_JOBID")
    if not job_id:
        raise PI0244Error("recovery training must run under its single PBS job")
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    attempt_root = root / "recovery_attempts" / f"{resume_step}-to-{stop_step}" / safe_job_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    legacy.bind_runtime(card)
    pi0228 = base.pi0228_module(Path(card["paths"]["code_root"]))
    config, summary = base.resolved_config(runtime, attempt_root, legacy, pi0228)
    if summary != card["resolved_config"]:
        raise PI0244Error("recovery resolved config differs from sealed R0 recipe")
    config.training.output_dir = str(training_root)
    config.training.resume_from_checkpoint = True
    legacy.install_fixed_dataset_factory(runtime, attempt_root, resume_after_step=resume_step)
    audit_steps = set(range(resume_step + 1, min(resume_step + 12, stop_step) + 1)) | {min(resume_step + 32, stop_step), stop_step}
    legacy.install_update_audit(
        attempt_root,
        audit_steps=audit_steps,
        expected_end_step=stop_step,
        stop_after_step=stop_step,
        checkpoint_output=training_root,
    )
    _run_recovery_training(pi0228, config, Path(card["paths"]["groot_source"]), stop_step)
    _close_recovery_segment(args, base, card, plan, root, job_id, attempt_root)


def reconcile_recovery(args: argparse.Namespace) -> None:
    if args.job_id is None or args.runner_snapshot is None or args.failure_log is None:
        raise PI0244Error("reconcile-recovery requires exact job id, sealed runner snapshot, and failure log")
    raw_card = json.loads(args.run_card.read_text(encoding="utf-8"))
    root = Path(raw_card["paths"]["output_root"])
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    resume_step, stop_step = (int(plan["segment"][key]) for key in ("resume_step", "stop_step"))
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(args.job_id))
    attempt_root = root / "recovery_attempts" / f"{resume_step}-to-{stop_step}" / safe_job_id
    expected_snapshot = attempt_root / "source_snapshot" / "pi0244_fixed16_10k.py"
    if args.runner_snapshot.resolve() != expected_snapshot.resolve() or sha256_file(expected_snapshot) != plan.get("active_runner_sha256"):
        raise PI0244Error("reconciliation runner snapshot is not the exact runner sealed in the recovery plan")
    if args.failure_log.name != f"{args.job_id}.ER":
        raise PI0244Error("reconciliation failure log is not the exact job stderr artifact")
    failure_text = args.failure_log.read_text(encoding="utf-8", errors="replace")
    if "Disk quota exceeded" not in failure_text or "trainer.save_model()" not in failure_text:
        raise PI0244Error("failure log does not prove the known redundant final-export quota failure")
    base, card, plan, root = _load_recovery_plan(args.plan, args.run_card, runner_path=args.runner_snapshot)
    _assert_archive_environment(plan)
    _close_recovery_segment(
        args,
        base,
        card,
        plan,
        root,
        str(args.job_id),
        attempt_root,
        reconciliation={
            "mode": "POST_EXIT_BOUNDARY_CHECKPOINT_RECONCILIATION",
            "pbs_exit_status": 1,
            "failure_log_sha256": sha256_file(args.failure_log),
            "runner_snapshot_sha256": sha256_file(args.runner_snapshot),
        },
    )


def verify_recovery(args: argparse.Namespace) -> None:
    base, card, plan, root = _load_recovery_plan(args.plan, args.run_card)
    _assert_archive_environment(plan)
    resume_step, stop_step = (int(plan["segment"][key]) for key in ("resume_step", "stop_step"))
    job_id = str(args.job_id)
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)
    attempt_root = root / "recovery_attempts" / f"{resume_step}-to-{stop_step}" / safe_job_id
    result_path = attempt_root / "segment_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("job_id") != job_id or result.get("status") != "TRAINED_AWAITING_DURABLE_ARCHIVE" or result.get("plan_sha256") != sha256_file(args.plan):
        raise PI0244Error("recovery result is not bound to this plan/job")
    checkpoint_path = root / "training" / f"checkpoint-{stop_step}"
    receipt_path = root / "durable_receipts" / f"checkpoint-{stop_step}.durable_receipt.json"
    persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    receipt = persistence.verify_receipt(checkpoint_path, receipt_path, expected_step=stop_step, expected_max_steps=MAX_STEPS, family=FAMILY)
    expected_durable_path = f"{plan['storage']['repo_id']}/{plan['storage']['path_prefix']}/checkpoint-{stop_step}"
    if receipt.get("hf_manifest_commit_id") is None or receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("durable_path") != expected_durable_path:
        raise PI0244Error("recovered checkpoint has no verified immutable HF revision")
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    checkpoint = legacy.checkpoint_record(checkpoint_path, stop_step)
    if checkpoint["tree_sha256"] != result.get("checkpoint_tree_sha256"):
        raise PI0244Error("recovered checkpoint changed after training closure")
    closure = {
        "schema": "PI0244_R0_RECOVERY_ARCHIVE_CLOSURE_V1",
        "status": "PASS_RECOVERY_SEGMENT_DURABLE",
        "experiment": EXPERIMENT,
        "job_id": job_id,
        "plan_sha256": sha256_file(args.plan),
        "segment_result_sha256": sha256_file(result_path),
        "checkpoint_step": stop_step,
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "durable_receipt_sha256": sha256_file(receipt_path),
        "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
        "hf_durable_path": receipt.get("durable_path"),
        "selected_checkpoint": "NONE",
    }
    _write_once(attempt_root / "archive_closure.json", closure)
    print(json.dumps(closure, indent=2, sort_keys=True))


def retire_local_checkpoint(args: argparse.Namespace) -> None:
    base, card = verify_card(args.run_card)
    root = Path(base.runtime_card(card)["paths"]["output_root"])
    storage_binding = _storage_binding(root, card, args.run_card)
    training = root / "training"
    target = training / f"checkpoint-{args.step}"
    receipt_path = root / "durable_receipts" / f"checkpoint-{args.step}.durable_receipt.json"
    retirement_root = root / "checkpoint_retirements"
    inventory_path = retirement_root / f"checkpoint-{args.step}.predelete.json"
    completion_path = retirement_root / f"checkpoint-{args.step}.deleted.json"
    if not target.exists():
        if completion_path.is_file():
            print(completion_path.read_text(encoding="utf-8"))
            return
        raise PI0244Error(f"checkpoint-{args.step} is absent without a retirement record")
    if target.is_symlink() or target.resolve().parent != training.resolve():
        raise PI0244Error("refusing to retire a checkpoint outside the exact training directory")
    checkpoints = _checkpoint_paths(training)
    latest = max(checkpoints, default=-1)
    if args.for_screen:
        manifest_path = root / "pi0244_training_manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "PASS_R0_TRAINING_CLOSED":
            raise PI0244Error("screen-cache retirement requires a closed, fully archived R0 training manifest")
    elif int(args.step) >= latest:
        raise PI0244Error("refusing to retire the active/latest recovery checkpoint")
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    receipt = persistence.verify_receipt(target, receipt_path, expected_step=int(args.step), expected_max_steps=MAX_STEPS, family=FAMILY)
    remote_expected = f"{storage_binding['effective_repo_id']}/{storage_binding['path_prefix']}/checkpoint-{args.step}"
    if receipt.get("durable_path") != remote_expected or not receipt.get("hf_manifest_commit_id"):
        raise PI0244Error("checkpoint does not have the pinned public-HF durable receipt")
    record = legacy.checkpoint_record(target, int(args.step))
    _write_once(
        inventory_path,
        {
            "schema": "PI0244_LOCAL_CHECKPOINT_RETIREMENT_V1",
            "status": "DURABLE_REMOTE_COPY_VERIFIED_DELETE_AUTHORIZED",
            "checkpoint_step": int(args.step),
            "local_tree_sha256": record["tree_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
            "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
            "hf_durable_path": receipt["durable_path"],
            "reason": "quota-safe rolling checkpoint storage; recoverable from pinned public-HF revision",
        },
    )
    shutil.rmtree(target)
    if target.exists():
        raise PI0244Error(f"checkpoint retirement did not remove exact local target: {target}")
    _write_once(completion_path, {"schema": "PI0244_LOCAL_CHECKPOINT_RETIREMENT_COMPLETE_V1", "status": "LOCAL_COPY_REMOVED_AFTER_DURABLE_VERIFICATION", "inventory_sha256": sha256_file(inventory_path), "checkpoint_step": int(args.step)})
    print(completion_path.read_text(encoding="utf-8"))


def finalize_training(args: argparse.Namespace) -> None:
    base, card = verify_card(args.run_card)
    root = Path(base.runtime_card(card)["paths"]["output_root"])
    storage_binding = _storage_binding(root, card, args.run_card)
    rows = _fixed_schedule(base, card)
    original_path = root / "training" / "pi0241_sample_consumption.jsonl"
    original = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(original) < PREFIX_STEPS:
        raise PI0244Error("original training log does not contain the sealed first-5000 exposure prefix")
    fields = ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")

    def matches(actual: list[Mapping[str, Any]], planned: list[Mapping[str, Any]]) -> bool:
        return len(actual) == len(planned) and all(
            all(actual_row.get(key) == planned_row.get(key) for key in fields)
            and int(actual_row.get("yield_ordinal", -1)) == int(planned_row["optimizer_step"])
            for actual_row, planned_row in zip(actual, planned)
        )

    if not matches(original[:PREFIX_STEPS], rows[:PREFIX_STEPS]):
        raise PI0244Error("original first-5000 actual consumption differs from the sealed schedule")
    consumed = list(original[:PREFIX_STEPS])
    segment_bindings = {}
    for resume_step, stop_step in ((5000, 7500), (7500, 10000)):
        closures = list((root / "recovery_attempts" / f"{resume_step}-to-{stop_step}").glob("*/archive_closure.json"))
        if len(closures) != 1:
            raise PI0244Error(f"expected exactly one first-valid archived recovery for {resume_step}->{stop_step}, found {len(closures)}")
        closure_path = closures[0]
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        attempt_root = closure_path.parent
        result_path = attempt_root / "segment_result.json"
        if closure.get("status") != "PASS_RECOVERY_SEGMENT_DURABLE":
            raise PI0244Error(f"recovery segment {resume_step}->{stop_step} is not durably closed")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "TRAINED_AWAITING_DURABLE_ARCHIVE" or result.get("job_id") != closure.get("job_id"):
            raise PI0244Error("recovery segment result and archive closure are not the same attempt")
        segment_rows_path = attempt_root / "pi0241_sample_consumption.jsonl"
        segment_rows = [json.loads(line) for line in segment_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not matches(segment_rows, rows[resume_step:stop_step]):
            raise PI0244Error(f"recovered actual consumption differs from schedule rows {resume_step + 1}..{stop_step}")
        consumed.extend(segment_rows)
        segment_bindings[f"{resume_step}-{stop_step}"] = {
            "job_id": closure["job_id"],
            "plan_sha256": closure["plan_sha256"],
            "archive_closure_sha256": sha256_file(closure_path),
            "consumption_sha256": sha256_file(segment_rows_path),
            "update_audit_sha256": result["update_audit_sha256"],
            "checkpoint_tree_sha256": closure["checkpoint_tree_sha256"],
            "hf_manifest_commit_id": closure["hf_manifest_commit_id"],
        }
    if not matches(consumed, rows):
        raise PI0244Error("canonical recovered 10K exposure sequence is not exactly the sealed schedule")
    checkpoints = {}
    for step in CHECKPOINTS:
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "DURABLE_VERIFIED" or int(receipt.get("checkpoint_step", -1)) != step or int(receipt.get("max_steps", -1)) != MAX_STEPS or receipt.get("family") != FAMILY:
            raise PI0244Error(f"checkpoint-{step} durable receipt identity mismatch")
        expected_durable_path = f"{storage_binding['effective_repo_id']}/{storage_binding['path_prefix']}/checkpoint-{step}"
        if receipt.get("durable_path") != expected_durable_path or not re.fullmatch(r"[a-f0-9]{40}", str(receipt.get("hf_manifest_commit_id", ""))):
            raise PI0244Error(f"checkpoint-{step} durable receipt differs from the recovery storage binding")
        local = root / "training" / f"checkpoint-{step}"
        if local.is_dir():
            persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
            persistence.verify_receipt(local, receipt_path, expected_step=step, expected_max_steps=MAX_STEPS, family=FAMILY)
        if len(receipt.get("durable_inventory", [])) != int(receipt["source_record"]["inventory"].get("file_count", -1)):
            raise PI0244Error(f"checkpoint-{step} public-HF inventory is incomplete")
        checkpoints[str(step)] = {
            "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
            "bytes": receipt["source_record"]["inventory"]["total_bytes"],
            "file_count": receipt["source_record"]["inventory"]["file_count"],
            "receipt_sha256": sha256_file(receipt_path),
            "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
            "hf_durable_path": receipt["durable_path"],
        }
    manifest = {
        "schema": "PI0244_R0_TRAINING_CLOSURE_V1",
        "status": "PASS_R0_TRAINING_CLOSED",
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "run_card_sha256": sha256_file(args.run_card),
        "schedule_sha256": card["schedule"]["sha256"],
        "resolved_config_sha256": _json_sha256(card["resolved_config"]),
        "initial_checkpoint": str(card["paths"]["initializer"]),
        "canonical_optimizer_steps": MAX_STEPS,
        "actual_sample_consumption": len(consumed),
        "actual_consumption_reconciled_to_schedule": True,
        "original_attempt_rows_retained_but_only_steps_1_to_5000_used_from_checkpoint_5000": len(original),
        "recovery_segments": segment_bindings,
        "checkpoints": checkpoints,
        "recovery_storage_binding_sha256": sha256_file(root / "pi0244_recovery_storage_binding.json"),
        "effective_hf_repo_id": storage_binding["effective_repo_id"],
        "effective_hf_path_prefix": storage_binding["path_prefix"],
        "target_xyz_in_model_input": False,
        "selected_checkpoint": "NONE",
    }
    _write_once(root / "pi0244_training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _retire_screen_cache(root: Path, checkpoint_path: Path, step: int, binding: Mapping[str, Any]) -> None:
    expected = (root / "training" / f"checkpoint-{step}").resolve()
    if checkpoint_path.resolve() != expected or binding.get("status") != "COMPLETE" or int(binding.get("step", -1)) != step:
        raise PI0244Error("screen cache can be released only after its exact profile is complete")
    retire_local_checkpoint(SimpleNamespace(run_card=root / "pi0243_run_card.json", step=step, for_screen=True))


def _linear_percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise PI0244Error("screen metric vector is empty or non-finite")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _screen_step_records(root: Path, step: int, expected: Mapping[str, int], *, complete: bool) -> list[dict[str, Any]]:
    records_path = root / "screen" / f"step-{step:05d}" / "records.jsonl"
    if not records_path.is_file():
        if complete:
            raise PI0244Error(f"screen step {step} has no records file")
        return []
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_episode_id", ""))
        if source_id not in expected or source_id in seen or int(row.get("screen_step", -1)) != step or int(row.get("episode_index", -1)) != expected[source_id]:
            raise PI0244Error(f"screen step {step} has an unexpected or duplicate source row")
        seen.add(source_id)
        if row.get("target_xyz_in_model_input") is not False or row.get("gate_decision") not in {"ACCEPT", "WARN", "REJECT"}:
            raise PI0244Error(f"screen step {step} violates the sealed input/gate contract")
        for key in ("xyz_rmse_m", "so3_rmse_rad"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(float(row[key])):
                raise PI0244Error(f"screen step {step} has an invalid {key}")
        for key in ("raw_sha256", "decoded_sha256", "tensor_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(row.get(key, ""))):
                raise PI0244Error(f"screen step {step} lacks a valid {key}")
        cell = records_path.parent / "cells" / f"train-{expected[source_id]:06d}.npz"
        if not cell.is_file() or sha256_file(cell) != row["tensor_sha256"]:
            raise PI0244Error(f"screen step {step} tensor is missing or changed: {cell.name}")
        if Path(str(row.get("tensor_path", ""))).name != cell.name:
            raise PI0244Error(f"screen step {step} tensor path identity mismatch")
        import numpy as np

        with np.load(cell, allow_pickle=False) as arrays:
            for key, digest_key in (("raw", "raw_sha256"), ("decoded", "decoded_sha256")):
                if key not in arrays.files:
                    raise PI0244Error(f"screen tensor omits {key}: {cell.name}")
                value = np.ascontiguousarray(arrays[key])
                if hashlib.sha256(value.tobytes()).hexdigest() != row[digest_key]:
                    raise PI0244Error(f"screen tensor {key} hash differs from its row record")
    if complete and seen != set(expected):
        raise PI0244Error(f"screen step {step} coverage mismatch: {len(seen)}/{len(expected)}")
    return rows


def _screen_closeout(root: Path, expected: Mapping[str, int]) -> dict[str, Any]:
    all_rows = []
    for step in SCREENS:
        rows = _screen_step_records(root, step, expected, complete=True)
        if len(rows) != len(expected):
            raise PI0244Error(f"screen step {step} is not a one-row-per-window profile")
        all_rows.extend(rows)
    cell_ids = {(int(row["screen_step"]), str(row["source_episode_id"])) for row in all_rows}
    if len(cell_ids) != len(SCREENS) * len(expected):
        raise PI0244Error("screen does not contain 80 unique step/window cells")
    final = [row for row in all_rows if int(row["screen_step"]) == MAX_STEPS]
    if len(final) != len(expected) or len({row["source_episode_id"] for row in final}) != len(expected):
        raise PI0244Error("final-step screen coverage is not exactly 16 unique windows")
    xyz = [float(row["xyz_rmse_m"]) for row in final]
    so3 = [float(row["so3_rmse_rad"]) for row in final]
    metrics = {
        "gate_rejects": sum(row["gate_decision"] == "REJECT" for row in final),
        "median_xyz_rmse_m": _linear_percentile(xyz, 0.50),
        "p95_xyz_rmse_m": _linear_percentile(xyz, 0.95),
        "median_so3_rmse_rad": _linear_percentile(so3, 0.50),
    }
    criteria = {
        "gate_rejects_zero": metrics["gate_rejects"] == 0,
        "median_xyz_rmse_le_0p020": metrics["median_xyz_rmse_m"] <= 0.020,
        "p95_xyz_rmse_le_0p035": metrics["p95_xyz_rmse_m"] <= 0.035,
        "median_so3_rmse_le_0p050": metrics["median_so3_rmse_rad"] <= 0.050,
    }
    return {
        "schema": "PI0244_STRICT_SCREEN_CLOSEOUT_V1",
        "status": "PASS_M16_10K" if all(criteria.values()) else "SCIENCE_FAIL_M16_10K_BUDGET",
        "metrics": metrics,
        "criteria": criteria,
        "screen_cells": len(cell_ids),
        "final_count": len(final),
        "step_counts": {str(step): sum(int(row["screen_step"]) == step for row in all_rows) for step in SCREENS},
        "final_reject_source_ids": sorted(str(row["source_episode_id"]) for row in final if row["gate_decision"] == "REJECT"),
        "selected_checkpoint": "NONE",
    }


def _restore_full_checkpoint(base: Any, card: Mapping[str, Any], root: Path, step: int) -> tuple[Path, bool]:
    training_root = root / "training"
    local = training_root / f"checkpoint-{step}"
    receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    persistence = base.load_module(Path(card["paths"]["code_root"]) / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    if local.is_dir():
        persistence.verify_receipt(local, receipt_path, expected_step=step, expected_max_steps=MAX_STEPS, family=FAMILY)
        return local, False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    storage_binding = _storage_binding(root, card, root / "pi0243_run_card.json")
    storage = {**card["storage"], "repo_id": storage_binding["effective_repo_id"], "path_prefix": storage_binding["path_prefix"]}
    prefix = f"{storage['path_prefix'].rstrip('/')}/checkpoint-{step}"
    expected_path = f"{storage['repo_id']}/{prefix}"
    if (
        receipt.get("status") != "DURABLE_VERIFIED"
        or receipt.get("family") != FAMILY
        or int(receipt.get("checkpoint_step", -1)) != step
        or int(receipt.get("max_steps", -1)) != MAX_STEPS
        or receipt.get("durable_path") != expected_path
        or not re.fullmatch(r"[a-f0-9]{40}", str(receipt.get("hf_manifest_commit_id", "")))
    ):
        raise PI0244Error(f"checkpoint-{step} lacks a pinned verified public-HF receipt")
    inventory = receipt.get("source_record", {}).get("inventory", {}).get("entries", [])
    if len(inventory) != int(receipt["source_record"]["inventory"].get("file_count", -1)):
        raise PI0244Error(f"checkpoint-{step} receipt inventory is incomplete")
    if local.exists():
        raise PI0244Error(f"checkpoint-{step} restore destination exists but is not a complete directory")
    training_root.mkdir(parents=True, exist_ok=True)
    temporary = training_root / f".checkpoint-{step}.restore-{os.getpid()}"
    if temporary.exists():
        raise PI0244Error(f"restore staging path already exists: {temporary}")
    temporary.mkdir()
    token_path = Path(storage["token_file"])
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise PI0244Error("HF token file is empty")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import hf_hub_download

        for entry in inventory:
            relative = str(entry["path"])
            remote_name = f"{prefix}/{relative}"
            downloaded = Path(
                hf_hub_download(
                    repo_id=str(storage["repo_id"]),
                    filename=remote_name,
                    revision=receipt["hf_manifest_commit_id"],
                    token=token,
                    local_dir=str(temporary),
                )
            )
            candidate = temporary / remote_name
            if not candidate.is_file() and downloaded.is_file():
                candidate = downloaded
            if not candidate.is_file() or candidate.stat().st_size != int(entry["bytes"]) or sha256_file(candidate) != entry["sha256"]:
                raise PI0244Error(f"HF readback mismatch for checkpoint-{step}/{relative}")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != destination.resolve():
                if candidate.is_symlink():
                    shutil.copyfile(candidate, destination)
                else:
                    try:
                        os.replace(candidate, destination)
                    except OSError:
                        shutil.copyfile(candidate, destination)
        for child in list(temporary.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
        record = legacy.checkpoint_record(temporary, step)
        restored_entries = [{key: entry[key] for key in ("path", "bytes", "sha256")} for entry in record["entries"]]
        source_entries = [{key: entry[key] for key in ("path", "bytes", "sha256")} for entry in inventory]
        if restored_entries != source_entries:
            raise PI0244Error(f"restored checkpoint-{step} file inventory mismatch")
        os.replace(temporary, local)
        persistence.verify_receipt(local, receipt_path, expected_step=step, expected_max_steps=MAX_STEPS, family=FAMILY)
    except Exception:
        quarantine = root / "recovery_quarantine" / f"hf-restore-checkpoint-{step}-{os.getpid()}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists() and not quarantine.exists():
            os.replace(temporary, quarantine)
        raise
    finally:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return local, True


def train(args: argparse.Namespace) -> None:
    base, card = verify_card(args.run_card)
    base.train_stage(SimpleNamespace(run_card=args.run_card, stage="a", authorization=AUTHORIZATION))
    manifest_path = Path(base.runtime_card(card)["paths"]["output_root"]) / "pi0243_training_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"experiment": EXPERIMENT, "protocol_commit": PROTOCOL_COMMIT, "stage_role": "PI0244_FIXED16_10K"})
        base.write_json(manifest_path, manifest)


def _screen_step_manifest(base: Any, card: Mapping[str, Any], root: Path, step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if step == 0:
        legacy = base.legacy_module(Path(card["paths"]["code_root"]))
        model_identity = {"kind": "initializer", "files": legacy._model_files(Path(card["paths"]["initializer"]))}
    else:
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        model_identity = {
            "kind": "PI0244_M16_10K_V1",
            "step": step,
            "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
            "hf_manifest_commit_id": receipt["hf_manifest_commit_id"],
            "receipt_sha256": sha256_file(receipt_path),
        }
    records_path = root / "screen" / f"step-{step:05d}" / "records.jsonl"
    return {
        "schema": "PI0244_SCREEN_STEP_BINDING_V1",
        "status": "COMPLETE",
        "experiment": EXPERIMENT,
        "step": step,
        "model_identity": model_identity,
        "record_count": len(rows),
        "records_sha256": sha256_file(records_path),
        "raw_sha256s": {str(row["source_episode_id"]): row["raw_sha256"] for row in rows},
        "decoded_sha256s": {str(row["source_episode_id"]): row["decoded_sha256"] for row in rows},
        "selected_checkpoint": "NONE",
    }


def screen(args: argparse.Namespace) -> None:
    base, card = verify_card(args.run_card)
    root = Path(base.runtime_card(card)["paths"]["output_root"])
    training_manifest = root / "pi0244_training_manifest.json"
    if not training_manifest.is_file() or json.loads(training_manifest.read_text(encoding="utf-8")).get("status") != "PASS_R0_TRAINING_CLOSED":
        raise PI0244Error("80-cell screen is blocked until recovered 10K training and all four archives are closed")
    storage_binding = _storage_binding(root, card, args.run_card)
    if json.loads(training_manifest.read_text(encoding="utf-8")).get("recovery_storage_binding_sha256") != sha256_file(root / "pi0244_recovery_storage_binding.json"):
        raise PI0244Error("R0 training closure is not bound to the current recovery archive")
    cohort_path = base._mapped_stage_path(card, card["cohort"]["path"], "PI0243_METADATA_MIRROR_ROOT")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))["rows"]
    expected = {str(row["source_episode_id"]): int(row["episode_index"]) for row in cohort}
    if len(expected) != 16 or len(cohort) != 16:
        raise PI0244Error("sealed fixed16 cohort is not 16 unique source windows")
    code_root = Path(card["paths"]["code_root"])
    persistence = base.load_module(code_root / "scripts" / "pi0243_checkpoint_persistence.py", "pi0244_persistence")
    original_verify_receipt = persistence.verify_receipt

    def screen_verify_receipt(local_checkpoint: Path, receipt_path: Path, *, expected_step: int, expected_max_steps: int, family: str) -> dict[str, Any]:
        if expected_max_steps not in {expected_step, MAX_STEPS}:
            raise PI0244Error("screen requested an unexpected checkpoint max_steps value")
        return original_verify_receipt(local_checkpoint, receipt_path, expected_step=expected_step, expected_max_steps=MAX_STEPS, family=family)

    persistence.verify_receipt = screen_verify_receipt
    sys.modules["pi0243_checkpoint_persistence"] = persistence
    base.verify_card = lambda _path: card
    original_runtime_card = base.runtime_card
    step_bindings = root / "screen_step_bindings"
    for step in SCREENS:
        rows = _screen_step_records(root, step, expected, complete=False)
        binding_path = step_bindings / f"step-{step:05d}.json"
        if len(rows) == len(expected):
            binding = _screen_step_manifest(base, card, root, step, rows)
            if not binding_path.is_file() or json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise PI0244Error(f"complete screen step {step} has no matching pre-existing model/profile identity")
            continue
        if binding_path.exists():
            raise PI0244Error(f"screen step {step} has a completion binding but incomplete rows")
        restored = False
        checkpoint_path = None
        if step:
            checkpoint_path, restored = _restore_full_checkpoint(base, card, root, step)

        def one_step_runtime(raw: Mapping[str, Any], *, _step: int = step) -> dict[str, Any]:
            runtime = original_runtime_card(raw)
            runtime["stage"]["screens"] = [_step]
            # The mature runner's closure is per invocation; this temporary
            # view preserves the same single-step rows and original 10K receipt.
            runtime["stage"]["max_steps"] = _step
            return runtime

        base.runtime_card = one_step_runtime
        try:
            base.screen_stage(SimpleNamespace(run_card=args.run_card, stage="a", device=args.device))
        finally:
            base.runtime_card = original_runtime_card
        rows = _screen_step_records(root, step, expected, complete=True)
        binding = _screen_step_manifest(base, card, root, step, rows)
        _write_once(binding_path, binding)
        if restored and checkpoint_path is not None:
            _retire_screen_cache(root, checkpoint_path, step, binding)
    closure = _screen_closeout(root, expected)
    closure.update(
        {
            "stage_role": "PI0244_FIXED16_10K",
            "terminal_token": None if closure["status"] == "PASS_M16_10K" else "SCIENCE_FAIL_M16_10K_BUDGET — STOP_FOR_PI_OBJECTIVE_OR_OPTIMIZATION_DECISION",
            "profile_binding_sha256s": {
                str(step): sha256_file(step_bindings / f"step-{step:05d}.json") for step in SCREENS
            },
        }
    )
    _write_once(root / "pi0244_stage_closure.json", closure)
    print(json.dumps(closure, indent=2, sort_keys=True))


def self_test() -> None:
    spec = stage_spec("a")
    assert spec["view"] == "16" and spec["max_steps"] == 10000
    assert spec["checkpoints"] == CHECKPOINTS and spec["screens"] == SCREENS
    current = [{"source_episode_id": f"s{i % 16}"} for i in range(10000)]
    parent = current[:5000]
    result = validate_schedule_rows(current, parent)
    assert result["length"] == 10000 and set(result["source_counts"].values()) == {625}
    assert set(_config_differences({"max_steps": 5000, "scheduler": "cosine"}, {"max_steps": 10000, "scheduler": "cosine"})) == {"max_steps"}
    storage_card = {"storage": {"path_prefix": "pi0244-m16-10k-v1-r1"}}
    assert _receipt_repo(
        {"durable_path": "leoyuer/pi0243-stage-a-admitted-v1/pi0244-m16-10k-v1-r1/checkpoint-2500"},
        storage_card,
        2500,
    ) == "leoyuer/pi0243-stage-a-admitted-v1"
    try:
        _receipt_repo({"durable_path": "other/repo/wrong-prefix/checkpoint-2500"}, storage_card, 2500)
    except PI0244Error:
        pass
    else:
        raise AssertionError("wrong durable archive prefix was not rejected")
    prior_repo, prior_prefix = os.environ.get("PI0243_HF_REPO_ID"), os.environ.get("PI0243_HF_PATH_PREFIX")
    try:
        os.environ["PI0243_HF_REPO_ID"] = "leoyuer/pi0243-stage-a-admitted-v1"
        os.environ["PI0243_HF_PATH_PREFIX"] = "pi0244-m16-10k-v1-r1"
        _assert_archive_environment({"storage": {"repo_id": "leoyuer/pi0243-stage-a-admitted-v1", "path_prefix": "pi0244-m16-10k-v1-r1"}})
        os.environ["PI0243_HF_REPO_ID"] = "wrong/repo"
        try:
            _assert_archive_environment({"storage": {"repo_id": "leoyuer/pi0243-stage-a-admitted-v1", "path_prefix": "pi0244-m16-10k-v1-r1"}})
        except PI0244Error:
            pass
        else:
            raise AssertionError("mismatched HF environment was not rejected")
    finally:
        if prior_repo is None:
            os.environ.pop("PI0243_HF_REPO_ID", None)
        else:
            os.environ["PI0243_HF_REPO_ID"] = prior_repo
        if prior_prefix is None:
            os.environ.pop("PI0243_HF_PATH_PREFIX", None)
        else:
            os.environ["PI0243_HF_PATH_PREFIX"] = prior_prefix
    legacy_path = Path(__file__).with_name("pi0241_fixed16_learning.py")
    legacy_spec = importlib.util.spec_from_file_location("pi0244_selftest_legacy", legacy_path)
    assert legacy_spec is not None and legacy_spec.loader is not None
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    schedule = [{"optimizer_step": step} for step in range(1, 10001)]
    suffix = legacy.fixed_schedule_suffix(schedule, 5000)
    assert len(suffix) == 5000 and suffix[0]["optimizer_step"] == 5001 and suffix[-1]["optimizer_step"] == 10000
    try:
        legacy.fixed_schedule_suffix([{"optimizer_step": 2}], 0)
    except legacy.PI0241Error:
        pass
    else:
        raise AssertionError("schedule ordinal drift was not rejected")
    with tempfile.TemporaryDirectory(prefix="pi0244-screen-") as temporary:
        import numpy as np

        root = Path(temporary)
        expected = {f"source-{index}": index for index in range(16)}
        for screen_step in SCREENS:
            step_root = root / "screen" / f"step-{screen_step:05d}"
            cell_root = step_root / "cells"
            cell_root.mkdir(parents=True)
            rows = []
            for source_id, episode_index in expected.items():
                cell = cell_root / f"train-{episode_index:06d}.npz"
                raw = np.asarray([[screen_step, episode_index]], dtype=np.float32)
                decoded = np.asarray([[episode_index, screen_step]], dtype=np.float32)
                np.savez_compressed(cell, raw=raw, decoded=decoded)
                digest = sha256_file(cell)
                rows.append({
                    "screen_step": screen_step,
                    "source_episode_id": source_id,
                    "episode_index": episode_index,
                    "target_xyz_in_model_input": False,
                    "gate_decision": "WARN",
                    "xyz_rmse_m": 0.01,
                    "so3_rmse_rad": 0.02,
                    "raw_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                    "decoded_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
                    "tensor_sha256": digest,
                    "tensor_path": str(cell),
                })
            (step_root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        closeout = _screen_closeout(root, expected)
        assert closeout["status"] == "PASS_M16_10K" and closeout["screen_cells"] == 80 and closeout["final_count"] == 16
        assert closeout["criteria"] == {
            "gate_rejects_zero": True,
            "median_xyz_rmse_le_0p020": True,
            "p95_xyz_rmse_le_0p035": True,
            "median_so3_rmse_le_0p050": True,
        }
        final_records = root / "screen" / "step-10000" / "records.jsonl"
        final_rows = [json.loads(line) for line in final_records.read_text(encoding="utf-8").splitlines()]
        final_rows[0]["gate_decision"] = "REJECT"
        final_records.write_text("".join(json.dumps(row) + "\n" for row in final_rows), encoding="utf-8")
        rejected = _screen_closeout(root, expected)
        assert rejected["status"] == "SCIENCE_FAIL_M16_10K_BUDGET" and rejected["metrics"]["gate_rejects"] == 1
    with tempfile.TemporaryDirectory(prefix="pi0244-final-export-") as temporary:
        output_root = Path(temporary)
        checkpoint = output_root / "checkpoint-7500"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 7500, "max_steps": MAX_STEPS}), encoding="utf-8")
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-00001.safetensors"}}), encoding="utf-8")
        for name in ("model-00001.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"ok")

        class DummyTrainer:
            def __init__(self) -> None:
                self.args = SimpleNamespace(output_dir=str(output_root))
                self.state = SimpleNamespace(global_step=7500)
                self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

            def save_model(self, *args: Any, **kwargs: Any) -> None:
                self.calls.append((args, kwargs))

        original_save = DummyTrainer.save_model
        DummyTrainer.save_model = _guard_final_checkpoint_export(original_save, 7500)
        trainer = DummyTrainer()
        trainer.save_model()
        assert not trainer.calls
        trainer.save_model(str(checkpoint), True)
        assert len(trainer.calls) == 1
        trainer.state.global_step = 7499
        trainer.save_model()
        assert len(trainer.calls) == 2
        trainer.state.global_step = 7500
        (checkpoint / "model-00001.safetensors").unlink()
        trainer.save_model()
        assert len(trainer.calls) == 3
    print("PI0244_SELF_TEST_PASS")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "train", "plan-recovery", "validate-plan", "recover", "reconcile-recovery", "verify-recovery", "retire-checkpoint", "finalize-training", "screen", "self-test"), required=True)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--stage10-root", type=Path)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--p237-root", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--processor-base-model", type=Path)
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--cosmos-snapshot", type=Path)
    parser.add_argument("--cosmos-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parent-card", type=Path)
    parser.add_argument("--prefix-schedule", type=Path)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--hf-path-prefix")
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--runner-snapshot", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--for-screen", action="store_true")
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.phase == "self-test":
        self_test()
        return 0
    if args.phase == "prepare":
        required = (args.code_root, args.groot_source, args.stage10_root, args.train_dataset, args.p237_root, args.modality_config, args.processor_base_model, args.initializer, args.cosmos_snapshot, args.cosmos_cache_root, args.output_root, args.parent_card, args.prefix_schedule, args.hf_repo_id, args.hf_token_file, args.hf_path_prefix)
        if any(value is None for value in required):
            raise PI0244Error("prepare requires all path, parent-card, prefix-schedule, and HF arguments")
        prepare(args)
    elif args.phase == "train":
        if args.run_card is None:
            raise PI0244Error("train requires --run-card")
        train(args)
    elif args.phase == "plan-recovery":
        if args.run_card is None or args.resume_step is None or args.stop_step is None:
            raise PI0244Error("plan-recovery requires run card and authorized segment")
        print(plan_recovery(args))
    elif args.phase == "validate-plan":
        if args.run_card is None or args.plan is None:
            raise PI0244Error("validate-plan requires run card and sealed recovery plan")
        validate_recovery_plan(args)
    elif args.phase == "recover":
        if args.run_card is None or args.plan is None:
            raise PI0244Error("recover requires run card and sealed plan")
        recover(args)
    elif args.phase == "reconcile-recovery":
        if args.run_card is None or args.plan is None:
            raise PI0244Error("reconcile-recovery requires run card and sealed plan")
        reconcile_recovery(args)
    elif args.phase == "verify-recovery":
        if args.run_card is None or args.plan is None or args.job_id is None:
            raise PI0244Error("verify-recovery requires run card, plan, and exact PBS job id")
        verify_recovery(args)
    elif args.phase == "retire-checkpoint":
        if args.run_card is None or args.step is None:
            raise PI0244Error("retire-checkpoint requires run card and exact step")
        retire_local_checkpoint(args)
    elif args.phase == "finalize-training":
        if args.run_card is None:
            raise PI0244Error("finalize-training requires run card")
        finalize_training(args)
    else:
        if args.run_card is None:
            raise PI0244Error("screen requires --run-card")
        screen(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
