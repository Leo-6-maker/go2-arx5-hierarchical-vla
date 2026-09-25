#!/usr/bin/env python3
"""Close an already completed PI0246 PT run without loading or querying a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


SIDECAR_SOURCES = {
    "experiment_cfg/config.yaml": "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json": "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/conf.yaml": "experiment_cfg/conf.yaml",
    "experiment_cfg/final_model_config.json": "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json": "experiment_cfg/final_processor_config.json",
    "processor_config.json": "processor/processor_config.json",
    "wandb_config.json": "wandb_config.json",
    "embodiment_id.json": "processor/embodiment_id.json",
    "statistics.json": "processor/statistics.json",
}
REQUIRED_TRAINER_FILES = {
    "config.json", "model.safetensors.index.json", "optimizer.pt", "scheduler.pt",
    "rng_state.pth", "trainer_state.json", "training_args.bin",
}


class CloseoutError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_durable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_full_trainer_inventory(record: dict[str, Any], step: int,
                                    model_index_path: Path, tree_hash: Any) -> dict[str, Any]:
    inventory = record.get("inventory", {})
    entries = inventory.get("entries", [])
    by_path = {str(row.get("path")): row for row in entries}
    if (record.get("global_step") != step or record.get("checkpoint_step") != step
            or record.get("max_steps") != 2000 or len(by_path) != len(entries)
            or inventory.get("file_count") != len(entries)
            or inventory.get("total_bytes") != sum(int(row.get("bytes", -1)) for row in entries)):
        raise CloseoutError(f"checkpoint-{step} inventory/step accounting is inconsistent")
    if not REQUIRED_TRAINER_FILES.issubset(by_path):
        raise CloseoutError(f"checkpoint-{step} lacks required full Trainer-state files")
    if tree_hash(step, entries) != inventory.get("tree_sha256"):
        raise CloseoutError(f"checkpoint-{step} inventory tree hash is inconsistent")
    if not model_index_path.is_file() or sha256(model_index_path) != by_path["model.safetensors.index.json"].get("sha256"):
        raise CloseoutError(f"checkpoint-{step} index differs from the sealed Trainer inventory")
    weight_map = read_json(model_index_path).get("weight_map", {})
    shards = set(weight_map.values())
    if (not shards or not shards.issubset(by_path)
            or not all(name.endswith(".safetensors") for name in shards)):
        raise CloseoutError(f"checkpoint-{step} model index does not resolve to archived weight shards")
    return {"file_count": len(entries), "total_bytes": int(inventory["total_bytes"]),
            "tree_sha256": inventory["tree_sha256"], "entries": entries}


def validate_retired_payload(training_root: Path, step: int,
                             archived_paths: set[str]) -> list[dict[str, Any]]:
    target = training_root / f"checkpoint-{step}"
    if target.is_symlink():
        raise CloseoutError(f"checkpoint-{step} residual directory is a symlink")
    if not target.exists():
        return []
    if not target.is_dir():
        raise CloseoutError(f"checkpoint-{step} residual path is not a directory")
    retained = []
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            raise CloseoutError(f"checkpoint-{step} residual sidecar is a symlink: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(target).as_posix()
        if relative in archived_paths:
            raise CloseoutError(f"checkpoint-{step} Trainer payload remains locally: {relative}")
        source_relative = SIDECAR_SOURCES.get(relative)
        source = training_root / source_relative if source_relative else None
        if source is None or not source.is_file() or path.stat().st_size != source.stat().st_size or sha256(path) != sha256(source):
            raise CloseoutError(f"checkpoint-{step} has an unrecognized residual file: {relative}")
        retained.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path),
                         "canonical_source": str(source)})
    return retained


def validate_forward_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2000 or [int(row.get("forward_index", -1)) for row in rows] != list(range(1, 2001)):
        raise CloseoutError("PT same-forward audit is not exactly 2000 ordered rows")
    required = {"native_loss", "physical_position_loss", "physical_delta_loss", "total_loss",
                "tau", "valid_points", "valid_transitions", "saturated_xyz"}
    for row in rows:
        if not required.issubset(row):
            raise CloseoutError("PT same-forward row is missing required component fields")
        numeric = [row[key] for key in ("native_loss", "physical_position_loss", "physical_delta_loss", "total_loss")]
        numeric += list(row["tau"])
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise CloseoutError("PT same-forward audit contains nonfinite values")
        expected = float(row["native_loss"]) + .1 * float(row["physical_position_loss"]) + .1 * float(row["physical_delta_loss"])
        if not math.isclose(float(row["total_loss"]), expected, rel_tol=1e-5, abs_tol=1e-7):
            raise CloseoutError("PT same-forward loss components do not match the frozen objective")
        if (len(row["tau"]) != 1 or not 0 <= float(row["tau"][0]) <= 1
                or not 0 < int(row["valid_points"]) <= 40
                or not 0 < int(row["valid_transitions"]) <= 39
                or not 0 <= int(row["saturated_xyz"]) <= 120):
            raise CloseoutError("PT same-forward mask/tau/saturation contract is invalid")
    return {
        "status": "COMPLETE", "rows": len(rows), "auxiliary_loss_names": ["physical_position_loss", "physical_delta_loss"],
        "sha256": None,
        "first_forward": rows[0], "last_forward": rows[-1],
        "native_loss_min": min(float(row["native_loss"]) for row in rows),
        "native_loss_max": max(float(row["native_loss"]) for row in rows),
        "auxiliary_loss_min": {key: min(float(row[key]) for row in rows) for key in ("physical_position_loss", "physical_delta_loss")},
        "auxiliary_loss_max": {key: max(float(row[key]) for row in rows) for key in ("physical_position_loss", "physical_delta_loss")},
        "valid_count_min": {key: min(int(row[key]) for row in rows) for key in ("valid_points", "valid_transitions")},
        "valid_count_max": {key: max(int(row[key]) for row in rows) for key in ("valid_points", "valid_transitions")},
        "saturated_xyz_total": sum(int(row["saturated_xyz"]) for row in rows),
    }


def validate_checkpoint_evidence(campaign: Any, card: dict[str, Any], training_root: Path,
                                 output_root: Path, binding: dict[str, Any],
                                 tree_hash: Any, step: int) -> dict[str, Any]:
    prefix, repo, family = card["archive"]["path_prefix"], card["archive"]["repo_id"], card["family"]
    receipt_path = output_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    readback_path = output_root / "durable_receipts" / f"checkpoint-{step}.hf_readback.json"
    intent_path = output_root / "checkpoint_retirements" / f"checkpoint-{step}.predelete.json"
    completion_path = output_root / "checkpoint_retirements" / f"checkpoint-{step}.deleted.json"
    tensor_path = training_root / "projector_reconciliation" / f"checkpoint-{step}.json"
    receipt, readback = read_json(receipt_path), read_json(readback_path)
    record = receipt.get("source_record", {})
    inventory = validate_full_trainer_inventory(
        record, step, training_root / "model.safetensors.index.json", tree_hash,
    )
    campaign.validate_pt_archive_receipt(receipt, record, repo_id=repo, prefix=prefix, family=family, step=step)
    expected = {
        f"{prefix}/checkpoint-{step}/{entry['path']}": (int(entry["bytes"]), entry["sha256"])
        for entry in record["inventory"]["entries"]
    }
    observed = {str(row.get("path")): row for row in readback.get("files", [])}
    if (readback.get("status") != "PASS_PT_FULL_HF_READBACK" or readback.get("repo_id") != repo
            or readback.get("path_prefix") != prefix or readback.get("family") != family
            or readback.get("checkpoint_step") != step or readback.get("max_steps") != 2000
            or readback.get("revision") != receipt.get("hf_manifest_commit_id")
            or readback.get("source_tree_sha256") != inventory["tree_sha256"]
            or readback.get("file_count") != len(expected) or set(observed) != set(expected)):
        raise CloseoutError(f"checkpoint-{step} pinned-revision readback identity is incomplete")
    for path, (size, digest) in expected.items():
        if int(observed[path].get("bytes", -1)) != size or observed[path].get("sha256") != digest:
            raise CloseoutError(f"checkpoint-{step} pinned-revision file mismatch: {path}")
    receipt_sha, readback_sha = sha256(receipt_path), sha256(readback_path)
    intent, completion = read_json(intent_path), read_json(completion_path)
    if (intent.get("status") != "DURABLE_HF_COPY_VERIFIED_DELETE_AUTHORIZED"
            or intent.get("checkpoint_record") != record or intent.get("durable_receipt_sha256") != receipt_sha
            or intent.get("hf_manifest_commit_id") != receipt.get("hf_manifest_commit_id")
            or intent.get("full_hf_readback_path") != str(readback_path)
            or intent.get("full_hf_readback_sha256") != readback_sha
            or completion.get("status") != "LOCAL_COPY_REMOVED_AFTER_DURABLE_VERIFICATION"
            or completion.get("predelete_intent_sha256") != sha256(intent_path)
            or completion.get("durable_receipt_sha256") != receipt_sha
            or completion.get("full_hf_readback_sha256") != readback_sha
            or completion.get("hf_manifest_commit_id") != receipt.get("hf_manifest_commit_id")):
        raise CloseoutError(f"checkpoint-{step} durable-archive/retirement chain does not close")
    sidecars = validate_retired_payload(training_root, step, set(row["path"] for row in record["inventory"]["entries"]))
    tensor = read_json(tensor_path)
    comparison = tensor.get("comparison", {})
    tensors = comparison.get("tensors", [])
    reconcile_script_sha = binding.get("reconcile_script_sha256")
    if (tensor.get("status") != "PASS_PROJECTOR_TENSOR_COMPARE"
            or tensor.get("checkpoint_step") != step or tensor.get("checkpoint_tree_sha256") != inventory["tree_sha256"]
            or tensor.get("run_card_sha256") != sha256(output_root / "run_card.json")
            or tensor.get("initializer_manifest_sha256") != card["parent"]["restore_manifest_sha256"]
            or tensor.get("reconciliation_script_sha256") != reconcile_script_sha
            or not tensors or comparison.get("tensor_count") != len(tensors)
            or comparison.get("changed_tensor_count") != len(tensors)
            or any(int(item.get("changed_elements", 0)) <= 0 for item in tensors)):
        raise CloseoutError(f"checkpoint-{step} full projector comparison does not verify")
    return {
        **record, "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"],
        "durable_receipt_path": str(receipt_path), "durable_receipt_sha256": receipt_sha,
        "hf_readback_path": str(readback_path), "hf_readback_sha256": readback_sha,
        "hf_manifest_commit_id": receipt["hf_manifest_commit_id"], "durable_path": receipt["durable_path"],
        "local_retired_after_verified_archive": True, "local_trainer_payload_absent": True,
        "retained_nontrainer_sidecars": sidecars,
        "retirement_intent_sha256": sha256(intent_path), "retirement_completion_sha256": sha256(completion_path),
        "projector_reconciliation_path": str(tensor_path), "projector_reconciliation_sha256": sha256(tensor_path),
        "projector_tensor_count": comparison["tensor_count"],
        "projector_changed_tensor_count": comparison["changed_tensor_count"],
    }


def recover(args: argparse.Namespace) -> dict[str, Any]:
    card_path = args.run_card.resolve(strict=True)
    card = read_json(card_path)
    code_scripts = Path(card["paths"]["campaign_code_root"]).resolve(strict=True)
    sys.path.insert(0, str(code_scripts))
    import pi0246_gate_pass_train as campaign
    import pi0245_p1_stock as p1

    output_root = Path(card["paths"]["output_root"]).resolve(strict=True)
    training_root = Path(card["paths"]["output_root"]) / "training"
    manifest_path = output_root / "training_manifest.json"
    reconciliation_path = output_root / "training_reconciliation.json"
    seal = read_json(card_path.with_name("run_card.seal.json"))
    if (seal.get("sha256") != sha256(card_path) or card.get("arm") != "PT"
            or card.get("family") != campaign.PT_ARM[0] or card.get("loss") != campaign.PT_ARM[1]
            or card.get("status") != "SEALED_READY_NOT_STARTED"):
        raise CloseoutError("sealed PT run-card identity mismatch")
    if not training_root.is_dir() or not args.runtime_root.is_dir():
        raise CloseoutError("completed PT training or sealed runtime root is absent")
    binding_path = args.runtime_root / "PT_runtime_binding.json"
    binding_sha = args.runtime_binding_sha256
    campaign.verify_pt_runtime(card_path, args.runtime_root, args.wrapper, binding_path, binding_sha)
    binding = read_json(binding_path)
    persistence_path = campaign.checkpoint_archive_helper_root(card["paths"]) / "pi0243_checkpoint_persistence.py"
    if sha256(persistence_path) != binding["checkpoint_archive_helpers"]["pi0243_checkpoint_persistence.py"]:
        raise CloseoutError("checkpoint persistence helper differs from the sealed PT runtime")
    persistence = campaign.load_module(persistence_path, "pi0246_pt_closeout_persistence")
    tree_hash = lambda step, entries: persistence._tree_sha256(  # noqa: E731
        entries, f"pi0243/{card['family']}/checkpoint-{step}",
    )
    preflight = read_json(output_root / "functional_preflight_pt_v1" / "functional_preflight.json")
    execution = preflight.get("execution", {})
    if (preflight.get("status") != "PASS_NO_MODEL" or preflight.get("run_card_sha256") != sha256(card_path)
            or preflight.get("runtime_binding_sha256") != binding_sha
            or preflight.get("model_forwards") != 0 or preflight.get("get_action_calls") != 0
            or preflight.get("optimizer_steps") != 0 or execution.get("pbs_job_id") != args.pbs_job_id
            or execution.get("profile") != "H200_PBS" or execution.get("device_count") != 1
            or "H200" not in execution.get("device_name", "")):
        raise CloseoutError("PT no-model functional preflight does not bind this H200 job")
    if args.pbs_exit_status != 1 or args.pbs_walltime_s <= 0:
        raise CloseoutError("recovery mode requires the observed engineering-only PBS exit and walltime")
    stdout = args.pbs_stdout.read_text(encoding="utf-8", errors="replace")
    stderr = args.pbs_stderr.read_text(encoding="utf-8", errors="replace")
    expected_error = "checkpoint-1000 local full-state copy was not retired after archival"
    if ("Training completed!" not in stderr or expected_error not in stderr
            or "PI0246_PT_TRAIN_FAILURE line=81 status=1" not in stderr
            or "PT checkpoint-1000 archived and locally retired after hash/readback verification" not in stdout
            or "PT checkpoint-2000 archived and locally retired after hash/readback verification" not in stdout):
        raise CloseoutError("PBS logs do not match the bounded, post-training closeout failure")
    if sha256(Path(card["cohort"]["path"])) != card["cohort"]["sha256"] or sha256(Path(card["schedule"]["path"])) != card["schedule"]["sha256"]:
        raise CloseoutError("sealed fixed16 cohort or PT exposure schedule changed")
    if sha256(Path(card["paths"]["p0_effective"])) != campaign.EXPECTED_P0_SHA256:
        raise CloseoutError("effective physical decoder identity changed")
    model_identity = p1.verify_model_files(Path(card["paths"]["base_model_path"]))
    if (model_identity.get("manifest_sha256") != card["parent"]["restore_manifest_sha256"]
            or model_identity.get("revision") != card["parent"]["revision"]):
        raise CloseoutError("R0-final initializer identity changed")

    p1_module = campaign.load_module(code_scripts / "pi0245_p1_stock.py", "pi0246_pt_closeout_p1")
    expected_order = [f"{model}__{source}__{noise}" for model in ("R0", "R2")
                      for source in p1_module.SOURCES for noise in p1_module.NOISES]
    p1_closure = campaign.verify_p1(
        Path(card["p1_closure_path"]), expected_order=expected_order,
        expected_r0_manifest_sha256=card["parent"]["restore_manifest_sha256"],
        expected_r0_revision=card["parent"]["revision"], expected_r0_records_sha256=p1_module.RECORD_SHA,
    )
    if p1_closure.get("status") != "PASS_COMPLETE_32_QUERY_MATRIX":
        raise CloseoutError("sealed P1 predecessor is not complete")
    if read_json(training_root / "resolved_config.json") != card["resolved_config"]:
        raise CloseoutError("PT resolved training configuration differs from its sealed card")

    schedule = read_json(Path(card["schedule"]["path"]))["rows"]
    consumption_path = training_root / "pi0241_sample_consumption.jsonl"
    contracts_path = training_root / "pi0241_sample_contracts.jsonl"
    forward_path = training_root / "pi0246_pt_combined_forward.jsonl"
    consumption = [json.loads(line) for line in consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    contracts = [json.loads(line) for line in contracts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(schedule) != 2000 or len(consumption) != 2000
            or [row.get("source_episode_id") for row in consumption] != [row.get("source_episode_id") for row in schedule]
            or [int(row.get("optimizer_step", -1)) for row in consumption] != list(range(1, 2001))
            or len(contracts) != 16 or {int(row.get("fixed_index", -1)) for row in contracts} != set(range(16))):
        raise CloseoutError("actual PT sample exposure or fixed16 action/mask contracts differ from the sealed plan")
    audit_path = training_root / "pi0241_update_audit.json"
    audit = read_json(audit_path)
    membership = audit.get("optimizer_membership", {})
    params = audit.get("final_parameters", {})
    if (audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != 2000
            or membership.get("missing_trainable") or membership.get("illegal_frozen")
            or float(params.get("diffusion", {}).get("delta_l2", 0)) <= 0
            or float(params.get("frozen_backbone", {}).get("delta_l2", -1)) != 0
            or float(params.get("frozen_vlln", {}).get("delta_l2", -1)) != 0):
        raise CloseoutError("PT optimizer/update audit is incomplete or violates the frozen/trainable contract")
    forward_rows = [json.loads(line) for line in forward_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    same_forward = validate_forward_rows(forward_rows)
    same_forward["sha256"] = sha256(forward_path)

    checkpoints = {str(step): validate_checkpoint_evidence(
        campaign, card, training_root, output_root, binding, tree_hash, step
    ) for step in (1000, 2000)}
    runtime = dict(preflight.get("runtime", {}))
    runtime["execution"] = execution
    update_integrity = {
        "projector_updated": float(params.get("projector", {}).get("delta_l2", 0)) > 0,
        "diffusion_updated": float(params["diffusion"]["delta_l2"]) > 0,
        "frozen_backbone_unchanged": float(params["frozen_backbone"]["delta_l2"]) == 0,
        "frozen_vlln_unchanged": float(params["frozen_vlln"]["delta_l2"]) == 0,
        "optimizer_membership_valid": not membership.get("missing_trainable") and not membership.get("illegal_frozen"),
        "same_forward_audit_complete": same_forward["rows"] == 2000,
    }
    reconcile_required = campaign.needs_projector_reconciliation("PT", update_integrity)
    manifest = {
        "schema": "PI0246_ARM_PT_TRAINING_MANIFEST_V1",
        "status": "TRAINING_COMPLETE_PROJECTOR_RECONCILIATION_REQUIRED" if reconcile_required
                  else "PASS_PT_TRAINING_INTEGRITY" if all(update_integrity.values())
                  else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "arm": "PT", "family": card["family"], "run_card_sha256": sha256(card_path),
        "p1_closure_sha256": sha256(Path(card["p1_closure_path"])), "p1_cells": len(p1_closure["cells"]),
        "elapsed_s": args.pbs_walltime_s, "elapsed_source": "PBS resources_used.walltime",
        "global_step": 2000, "sample_consumption": len(consumption), "pbs_job_id": args.pbs_job_id,
        "pbs_exit_status": args.pbs_exit_status, "execution": execution,
        "schedule_sha256": card["schedule"]["sha256"], "sample_contracts": contracts,
        "update_integrity": update_integrity, "update_audit_sha256": sha256(audit_path),
        "projector_reconciliation_required": reconcile_required, "checkpoints": checkpoints,
        "runtime_binding": runtime, "pt_runtime_binding": binding,
        "pt_runtime_binding_sha256": binding_sha, "overlay": {
            "forward_calls": len(forward_rows), "first_forward": forward_rows[0],
            "runtime_binding_sha256": binding_sha, "forward_log_sha256": sha256(forward_path),
            "update_audit_sha256": sha256(audit_path),
        },
        "same_forward_audit": same_forward, "forbidden_counts": card["forbidden_counts"],
        "selected_checkpoint": "NONE",
        "engineering_closeout_recovery": {
            "status": "PASS_READONLY_RECOVERY_AFTER_KNOWN_FINALIZER_GUARD",
            "script_sha256": sha256(Path(__file__).resolve()),
            "original_training_pbs_job_id": args.pbs_job_id,
            "original_pbs_stdout_sha256": sha256(args.pbs_stdout),
            "original_pbs_stderr_sha256": sha256(args.pbs_stderr),
            "training_reruns": 0, "model_loads": 0, "model_forwards": 0,
            "get_action_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
        },
    }
    reconciliation = build_readonly_reconciliation(
        card, card_path, training_root, manifest, audit, checkpoints, binding,
        sha256(Path(__file__).resolve()), sha256,
    )
    if not all(reconciliation["effective_update_integrity_after_readonly_tensor_comparison"].values()):
        raise CloseoutError("read-only complete projector reconciliation did not confirm PT updates")
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha = hashlib.sha256(manifest_bytes.encode("utf-8")).hexdigest()
    reconciliation["source"]["training_manifest_sha256"] = manifest_sha
    for path, payload in ((manifest_path, manifest), (reconciliation_path, reconciliation)):
        if path.exists():
            if read_json(path) != payload:
                raise CloseoutError(f"refusing to replace a different existing closeout: {path.name}")
        else:
            write_json_durable(path, payload)
    return {"training_manifest_sha256": sha256(manifest_path),
            "training_reconciliation_sha256": sha256(reconciliation_path),
            "training_status": manifest["status"], "reconciliation_status": reconciliation["status"],
            "steps": 2000, "sample_rows": len(consumption), "checkpoints": {
                step: {"tree_sha256": checkpoints[step]["inventory"]["tree_sha256"],
                       "file_count": checkpoints[step]["file_count"],
                       "retained_nontrainer_sidecars": len(checkpoints[step]["retained_nontrainer_sidecars"])}
                for step in ("1000", "2000")},
            "selected_checkpoint": "NONE"}


def build_readonly_reconciliation(card: dict[str, Any], card_path: Path, training_root: Path,
                                  manifest: dict[str, Any], audit: dict[str, Any],
                                  checkpoints: dict[str, dict[str, Any]], binding: dict[str, Any],
                                  script_sha: str, sha_fn: Any) -> dict[str, Any]:
    comparisons = {}
    for step in ("1000", "2000"):
        record = checkpoints[step]
        tensor_path = Path(record["projector_reconciliation_path"])
        tensor = read_json(tensor_path)
        comparisons[step] = tensor["comparison"]
    original = manifest["update_integrity"]
    effective = {**original, "projector_updated": all(
        comparisons[step]["changed_tensor_count"] == comparisons[step]["tensor_count"]
        and comparisons[step]["tensor_count"] > 0 for step in ("1000", "2000"))}
    overlay_path = Path(binding["runtime_root"]) / "pi0246_pt_combined_overlay.py"
    status = "TRAINING_INTEGRITY_RECONCILED_PT_FULL_PROJECTOR_TENSORS" if all(effective.values()) else "PROJECTOR_UPDATE_NOT_CONFIRMED"
    return {
        "schema": "PI0246_PT_READONLY_UPDATE_RECONCILIATION_RECOVERY_V1", "status": status,
        "source": {
            "reconciliation_script_sha256": script_sha,
            "parameter_classifier_sha256": sha_fn(Path(card["paths"]["code_root"]) / "scripts/pi0241_fixed16_learning.py"),
            "campaign_trainer_sha256": sha_fn(Path(card["paths"]["campaign_code_root"]) / "pi0246_gate_pass_train.py"),
            "native_overlay_sha256": sha_fn(overlay_path),
            "run_card_sha256": sha_fn(card_path),
            "training_manifest_sha256": None,
            "original_training_status": manifest["status"],
            "update_audit_sha256": sha_fn(training_root / "pi0241_update_audit.json"),
            "schedule_sha256": card["schedule"]["sha256"],
            "consumed_rows": 2000, "action_mask_contracts": 16,
            "forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
        },
        "original_projector_probe": audit["final_parameters"].get("projector"),
        "original_update_integrity": original,
        "effective_update_integrity_after_readonly_tensor_comparison": effective,
        "full_projector_tensor_differences_vs_r0": comparisons,
        "interpretation": "A zero sampled projector probe was not treated as no update; the existing per-checkpoint full tensor comparisons were verified against the sealed runtime and pinned HF checkpoint trees.",
        "selected_checkpoint": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-binding-sha256", required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--pbs-job-id", required=True)
    parser.add_argument("--pbs-exit-status", type=int, required=True)
    parser.add_argument("--pbs-walltime-s", type=int, required=True)
    parser.add_argument("--pbs-stdout", type=Path, required=True)
    parser.add_argument("--pbs-stderr", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args), sort_keys=True))


if __name__ == "__main__":
    main()
