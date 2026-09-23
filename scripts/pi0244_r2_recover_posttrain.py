#!/usr/bin/env python3
"""Reconcile PI0244 R2 post-trainer evidence without loading or running a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _projector_prefix_samples(root: Path) -> dict[str, Any]:
    """Hash a small deterministic prefix from every sealed projector tensor."""
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"invalid safetensors weight map: {index_path}")
    tokens = ("action_head.state_encoder", "action_head.action_encoder", "action_head.action_decoder", "action_head.position_embedding")
    names = sorted(name for name in weight_map if any(token in name for token in tokens))
    if len(names) != 15:
        raise RuntimeError(f"expected 15 projector tensors, found {len(names)}")

    headers: dict[str, tuple[dict[str, Any], int]] = {}
    samples: dict[str, str] = {}
    shards: dict[str, int] = {}
    for name in names:
        shard = str(weight_map[name])
        shard_path = (root / shard).resolve()
        if Path(shard).name != shard or root.resolve() not in shard_path.parents:
            raise RuntimeError(f"unsafe projector shard path: {shard}")
        if shard not in headers:
            with shard_path.open("rb") as handle:
                raw_len = handle.read(8)
                if len(raw_len) != 8:
                    raise RuntimeError(f"truncated safetensors header: {shard_path}")
                header_len = struct.unpack("<Q", raw_len)[0]
                header = json.loads(handle.read(header_len))
                headers[shard] = (header, 8 + header_len)
        header, data_start = headers[shard]
        meta = header.get(name)
        if not isinstance(meta, dict) or meta.get("dtype") != "F32":
            raise RuntimeError(f"projector tensor is missing or not F32: {name}")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"invalid projector tensor metadata: {name}")
        elements = math.prod(int(dim) for dim in shape)
        start, end = (int(value) for value in offsets)
        if elements <= 0 or end - start != elements * 4:
            raise RuntimeError(f"projector tensor byte length mismatch: {name}")
        sample_elements = min(4096, elements)
        with shard_path.open("rb") as handle:
            handle.seek(data_start + start)
            payload = handle.read(sample_elements * 4)
        if len(payload) != sample_elements * 4:
            raise RuntimeError(f"truncated projector tensor prefix: {name}")
        samples[name] = hashlib.sha256(payload).hexdigest()
        shards[shard] = shard_path.stat().st_size

    canonical = "".join(name + "\0" + samples[name] + "\n" for name in sorted(samples)).encode()
    return {
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "sample_map_sha256": hashlib.sha256(canonical).hexdigest(),
        "sample_count": len(samples),
        "samples": samples,
        "shards": shards,
    }


def _changed_projector_samples(initial: dict[str, str], final: dict[str, str]) -> list[str]:
    if set(initial) != set(final):
        raise RuntimeError("projector sample key sets differ")
    return sorted(name for name in initial if initial[name] != final[name])


def validate_projector_evidence(
    evidence_path: Path,
    *,
    card: dict[str, Any],
    run_card_path: Path,
    initializer: Path,
    checkpoint: Path,
    final_receipt: dict[str, Any],
) -> dict[str, Any]:
    evidence = read_json(evidence_path)
    if evidence.get("schema") != "PI0244_R2_PROJECTOR_UPDATE_PREFIX_AUDIT_V1":
        raise RuntimeError("projector update evidence schema mismatch")
    if evidence.get("status") != "PASS_SOURCE_AND_PINNED_ARCHIVE_PREFIX_PARITY":
        raise RuntimeError("projector source/archive sample parity did not pass")
    if evidence.get("run_card_sha256") != sha256_file(run_card_path) or int(evidence.get("checkpoint_step", -1)) != int(card["stage"]["max_steps"]):
        raise RuntimeError("projector evidence is bound to another run or checkpoint")
    if evidence.get("method", {}).get("projector_tensor_count") != 15 or evidence.get("method", {}).get("sampled_prefix_count_per_checkpoint") != 15:
        raise RuntimeError("projector evidence does not cover all 15 tensors")
    if (
        evidence.get("method", {}).get("group_rule") != "PI0243_stage_a_cpu_audit._parameter_group projector token set"
        or evidence.get("method", {}).get("sample_rule") != "first min(4096, numel) raw F32 elements from each projector tensor"
        or evidence.get("method", {}).get("source_and_hf_sample_map_sha256_must_match") is not True
        or evidence.get("method", {}).get("hf_ranges_required_http_206_exact_content_range") is not True
    ):
        raise RuntimeError("projector evidence sampling or group contract differs")
    if evidence.get("method", {}).get("model_weights_constructed") is not False or any(
        int(evidence.get("method", {}).get(key, -1)) != 0
        for key in ("model_forwards", "backward_optimizer_scheduler_steps")
    ):
        raise RuntimeError("projector evidence reports unauthorized model activity")

    init_entry = evidence.get("initializer", {})
    final_entry = evidence.get("final_checkpoint", {})
    init_files = card.get("initializer", {}).get("files", {})
    final_inventory = final_receipt["source_record"]["inventory"]["entries"]
    final_files = {str(row["path"]): row for row in final_inventory}
    expected_init_index = init_files.get("model.safetensors.index.json", {}).get("sha256")
    expected_init_shard = init_files.get("model-00002-of-00002.safetensors", {}).get("sha256")
    expected_final_index = final_files.get("model.safetensors.index.json", {}).get("sha256")
    expected_final_shard = final_files.get("model-00002-of-00002.safetensors", {}).get("sha256")
    expected_repo = str(card.get("storage", {}).get("repo_id", ""))
    if (
        str(initializer) != str(card["paths"]["initializer"])
        or init_entry.get("model_index_sha256") != expected_init_index
        or init_entry.get("sealed_model_shard_sha256") != expected_init_shard
        or init_entry.get("model_shard_bytes") != init_files.get("model-00002-of-00002.safetensors", {}).get("bytes")
        or final_entry.get("model_index_sha256") != expected_final_index
        or final_entry.get("sealed_model_shard_sha256") != expected_final_shard
        or final_entry.get("model_shard_bytes") != final_files.get("model-00002-of-00002.safetensors", {}).get("bytes")
        or init_entry.get("hf_repo_id") != expected_repo
        or final_entry.get("hf_repo_id") != expected_repo
        or final_entry.get("hf_revision") != final_receipt.get("hf_manifest_commit_id")
        or final_entry.get("hf_path") != final_receipt.get("durable_path", "").removeprefix(expected_repo + "/")
    ):
        raise RuntimeError("projector evidence does not match sealed initializer/checkpoint identities")

    initial_samples = _projector_prefix_samples(initializer)
    final_samples = _projector_prefix_samples(checkpoint)
    for label, measured, recorded in (
        ("initializer", initial_samples, init_entry),
        ("final checkpoint", final_samples, final_entry),
    ):
        if (
            measured["index_sha256"] != recorded.get("model_index_sha256")
            or measured["sample_map_sha256"] != recorded.get("local_sample_map_sha256")
            or measured["sample_map_sha256"] != recorded.get("hf_sample_map_sha256")
            or measured["sample_count"] != recorded.get("sampled_prefix_count", 15)
            or measured["shards"].get("model-00002-of-00002.safetensors") != recorded.get("model_shard_bytes")
        ):
            raise RuntimeError(f"projector sample digest does not match local and pinned-HF evidence: {label}")

    changed = _changed_projector_samples(initial_samples["samples"], final_samples["samples"])
    if changed != evidence.get("result", {}).get("changed_projector_sampled_tensors") or not changed:
        raise RuntimeError("projector sample comparison does not prove a changed trainable parameter")
    return {
        "status": "PASS_PROJECTOR_PARAMETER_CHANGE_PROVEN",
        "projector_updated": True,
        "basis": "at least one exact sampled F32 prefix differs; this is sufficient for the existing any-nonzero-projector-parameter integrity predicate",
        "full_projector_l2_computed": False,
        "sampled_projector_tensor_count": len(initial_samples["samples"]),
        "changed_projector_sampled_tensors": changed,
        "initializer_sample_map_sha256": initial_samples["sample_map_sha256"],
        "final_sample_map_sha256": final_samples["sample_map_sha256"],
        "evidence_sha256": sha256_file(evidence_path),
    }


def load_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pi0244_r2_recovery_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed R2 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reconcile_audit(audit: dict[str, Any], max_steps: int, legacy_default: int) -> dict[str, Any]:
    actual = int(audit.get("global_step", -1))
    recorded_expected = int(audit.get("expected_end_step", -1))
    if actual != max_steps:
        raise RuntimeError(f"update audit ended at {actual}, expected {max_steps}")
    if audit.get("status") == "COMPLETE" and recorded_expected == max_steps:
        return {"kind": "native_complete", "expected_end_step": recorded_expected}
    if not (
        audit.get("status") == "INCOMPLETE"
        and recorded_expected == legacy_default
        and legacy_default != max_steps
    ):
        raise RuntimeError("update-audit status cannot be explained by the known stale default")
    return {
        "kind": "stale_import_time_default_reconciled",
        "original_status": audit["status"],
        "recorded_expected_end_step": recorded_expected,
        "sealed_expected_end_step": max_steps,
        "actual_global_step": actual,
        "explanation": "the mature PI0241 audit function captured its 1000-step default at import; runtime MAX_STEPS was later configured to 10000, but the default argument was not rebound",
    }


def recover(run_card_path: Path, pbs_stdout: Path, projector_evidence_path: Path) -> dict[str, Any]:
    card_hint = read_json(run_card_path)
    runner = load_runner(Path(card_hint["r2_implementation"]["runner_path"]))
    base, card = runner.verify_card(run_card_path)
    root = Path(card["paths"]["output_root"])
    training = root / "training"
    manifest_path = root / "pi0243_training_manifest.json"
    closure_path = root / "pi0244x_xyz4_training_closure.json"
    recovery_path = root / "pi0244x_xyz4_posttrain_reconciliation.json"
    if any(path.exists() for path in (manifest_path, closure_path, recovery_path)):
        raise RuntimeError("refusing to overwrite an existing post-training manifest, closure, or reconciliation")

    steps = [int(step) for step in card["stage"]["checkpoints"]]
    max_steps = int(card["stage"]["max_steps"])
    family = str(card["storage"]["checkpoint_family"])
    schedule_path = Path(base._mapped_stage_path(card, card["schedule"]["path"], "PI0243_METADATA_MIRROR_ROOT"))
    schedule = runner._source_schedule(schedule_path)
    consumption_path = training / "pi0241_sample_consumption.jsonl"
    consumption = read_jsonl(consumption_path)
    fields = ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")
    if len(consumption) != max_steps or any(
        any(actual.get(key) != expected.get(key) for key in fields)
        for actual, expected in zip(consumption, schedule)
    ):
        raise RuntimeError("consumed sample rows do not exactly match the sealed schedule")

    contracts_path = training / "pi0241_sample_contracts.jsonl"
    contracts = read_jsonl(contracts_path)
    if len(contracts) != len(card["fixed16_source_ids"]):
        raise RuntimeError("fixed16 sample-contract count mismatch")
    for row in contracts:
        if row.get("action_shape") != [1, 40, 132] or row.get("mask_shape") != [1, 40, 132] or row.get("active_slots") != list(range(10)):
            raise RuntimeError(f"sample action/mask contract mismatch: {row.get('source_episode_id')}")

    from pi0243_checkpoint_persistence import verify_receipt, write_json_durable

    checkpoints: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for step in steps:
        checkpoint = training / f"checkpoint-{step}"
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipts[str(step)] = verify_receipt(checkpoint, receipt_path, expected_step=step, expected_max_steps=max_steps, family=family)
        checkpoints[str(step)] = receipts[str(step)]["source_record"]

    audit_path = training / "pi0241_update_audit.json"
    audit = read_json(audit_path)
    legacy = base.legacy_module(Path(card["paths"]["code_root"]))
    base.configure_legacy(card, legacy)
    legacy_default = int(inspect.signature(legacy.install_update_audit).parameters["expected_end_step"].default)
    audit_reconciliation = reconcile_audit(audit, max_steps, legacy_default)
    if audit.get("optimizer_membership", {}).get("missing_trainable") or audit.get("optimizer_membership", {}).get("illegal_frozen"):
        raise RuntimeError("optimizer membership audit reports an invalid trainable/frozen set")
    if any(not event.get("gradient_finite", True) for event in audit.get("events", [])):
        raise RuntimeError("update audit contains a non-finite gradient event")
    final = audit.get("final_parameters", {})
    projector_evidence = validate_projector_evidence(
        projector_evidence_path,
        card=card,
        run_card_path=run_card_path,
        initializer=Path(card["paths"]["initializer"]),
        checkpoint=training / f"checkpoint-{max_steps}",
        final_receipt=receipts[str(max_steps)],
    )
    update_integrity = {
        "projector_updated": projector_evidence["projector_updated"],
        "diffusion_updated": float(final.get("diffusion", {}).get("delta_l2", 0.0)) > 0.0,
        "frozen_backbone_unchanged": float(final.get("frozen_backbone", {}).get("delta_l2", 1.0)) == 0.0,
        "frozen_vlln_unchanged": float(final.get("frozen_vlln", {}).get("delta_l2", 1.0)) == 0.0,
        "optimizer_membership_valid": True,
    }
    if not all(update_integrity.values()):
        raise RuntimeError(f"parameter update integrity failed: {update_integrity}")

    runtime_files = sorted(training.glob("pi0244_r2_loss_runtime_*.json"))
    first_files = sorted(training.glob("pi0244_r2_first_forward_*.json"))
    moments_files = sorted(training.glob("pi0244_r2_first_update_moments_*.json"))
    if not (len(runtime_files) == len(first_files) == len(moments_files) == 1):
        raise RuntimeError("R2 loss/forward/update evidence is not a unique training attempt")
    runtime, first, moments = (read_json(paths[0]) for paths in (runtime_files, first_files, moments_files))
    if runtime.get("status") != "TRAINER_RETURNED" or int(runtime.get("forward_calls", 0)) != max_steps:
        raise RuntimeError("native training did not return after its complete fixed update schedule")
    if first.get("status") != "RECORDED" or float(first.get("native_component_loss_delta", 1.0)) > 1e-5:
        raise RuntimeError("first native loss-hook evidence is invalid")
    rows = moments.get("trainable_moment_records", [])
    names = [row.get("name") for row in rows]
    if (
        moments.get("status") != "RECORDED"
        or int(moments.get("global_step", -1)) != 1
        or int(moments.get("trainable_parameter_count", -1)) != len(rows)
        or len(names) != len(set(names))
        or any(row.get("parameter_dtype") is None or not {"exp_avg", "exp_avg_sq"}.issubset(row.get("moments", {})) for row in rows)
    ):
        raise RuntimeError("first real update is missing complete named optimizer moments")

    config_path = training / "resolved_config.json"
    config = read_json(config_path)
    if config != card["resolved_config"]:
        raise RuntimeError("saved training configuration differs from the sealed run card")
    runtime_match = re.search(r"train_runtime['\"]?\s*[:=]\s*([0-9.]+)", pbs_stdout.read_text(encoding="utf-8", errors="replace"))
    elapsed_s = float(runtime_match.group(1)) if runtime_match else None

    recovery = {
        "schema": "PI0244_R2_XYZ4_POSTTRAIN_RECONCILIATION_V1",
        "status": "PASS_DERIVED_FROM_EXISTING_RUN_ARTIFACTS",
        "run_card_sha256": sha256_file(run_card_path),
        "execution_binding_id": card["r2_execution_binding"]["binding_id"],
        "trainer_stdout_sha256": sha256_file(pbs_stdout),
        "sample_consumption_sha256": sha256_file(consumption_path),
        "sample_contracts_sha256": sha256_file(contracts_path),
        "update_audit_sha256": sha256_file(audit_path),
        "update_audit_original_status": audit.get("status"),
        "update_audit_reconciliation": audit_reconciliation,
        "representative_projector_probe_delta_l2": float(final.get("projector", {}).get("delta_l2", 0.0)),
        "projector_update_evidence": projector_evidence,
        "update_audit_file_modified": False,
        "trainable_parameters_loaded_or_updated_during_recovery": False,
        "model_forwards_during_recovery": 0,
        "backward_optimizer_scheduler_steps_during_recovery": 0,
        "checkpoints": {str(step): checkpoints[str(step)]["inventory"]["tree_sha256"] for step in steps},
        "receipts": {str(step): sha256_file(root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json") for step in steps},
        "selected_checkpoint": "NONE",
    }
    write_json_durable(recovery_path, recovery)

    training_manifest = {
        "schema": "PI0243_STAGE_TRAINING_MANIFEST_V1",
        "status": "PASS_UPDATE_INTEGRITY",
        "stage": card["stage"]["stage"],
        "elapsed_s": elapsed_s,
        "canonical_optimizer_steps": max_steps,
        "actual_sample_consumption": len(consumption),
        "schedule_sha256": card["schedule"]["sha256"],
        "config": config,
        "update_integrity": update_integrity,
        "projector_update_evidence": projector_evidence,
        "checkpoints": checkpoints,
        "durable_receipts": receipts,
        "storage": card["storage"],
        "sample_contract_count": len(contracts),
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
        "postrun_reconciliation_sha256": sha256_file(recovery_path),
        "finalized_after_training": True,
        "finalization_mode": "READ_ONLY_RECONCILIATION_FROM_SEALED_RUN_ARTIFACTS",
    }
    write_json_durable(manifest_path, training_manifest)

    closure = {
        "schema": "PI0244_R2_XYZ4_TRAINING_CLOSURE_V1",
        "status": "PASS_R2_TRAINING_CLOSED",
        "experiment": card["experiment"],
        "protocol_commit": card["protocol_commit"],
        "run_card_sha256": sha256_file(run_card_path),
        "execution_binding_id": card["r2_execution_binding"]["binding_id"],
        "schedule_sha256": card["schedule"]["sha256"],
        "actual_sample_consumption": len(consumption),
        "actual_consumption_matches_all_schedule_fields": True,
        "optimizer_steps": int(audit["global_step"]),
        "update_integrity": update_integrity,
        "projector_update_evidence": projector_evidence,
        "r2_loss_runtime_sha256": sha256_file(runtime_files[0]),
        "first_forward_sha256": sha256_file(first_files[0]),
        "first_update_named_moments_sha256": sha256_file(moments_files[0]),
        "checkpoint_receipts": {
            str(step): {
                "receipt_sha256": sha256_file(root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"),
                "tree_sha256": receipts[str(step)]["source_record"]["inventory"]["tree_sha256"],
                "bytes": receipts[str(step)]["source_record"]["inventory"]["total_bytes"],
                "hf_path": receipts[str(step)]["durable_path"],
            }
            for step in steps
        },
        "postrun_reconciliation_sha256": sha256_file(recovery_path),
        "update_audit_original_status": audit.get("status"),
        "update_audit_reconciliation": audit_reconciliation,
        "selected_checkpoint": "NONE",
    }
    write_json_durable(closure_path, closure)
    return {"reconciliation": recovery, "manifest_sha256": sha256_file(manifest_path), "closure_sha256": sha256_file(closure_path), "status": closure["status"]}


def audit_parameters(run_card_path: Path) -> dict[str, Any]:
    card_hint = read_json(run_card_path)
    runner = load_runner(Path(card_hint["r2_implementation"]["runner_path"]))
    base, card = runner.verify_card(run_card_path)
    audit_module = base.load_module(
        Path(card["paths"]["code_root"]) / "scripts" / "pi0243_stage_a_cpu_audit.py",
        "pi0244_r2_full_parameter_audit",
    )
    output = audit_module.full_parameter_update(
        Path(card["paths"]["initializer"]),
        Path(card["paths"]["output_root"]) / "training" / f"checkpoint-{card['stage']['max_steps']}",
    )
    return {
        "schema": "PI0244_R2_CPU_FULL_PARAMETER_DELTA_AUDIT_V1",
        "status": "COMPLETE",
        "checkpoint_step": int(card["stage"]["max_steps"]),
        "initializer_role": card["initializer"]["role"],
        "run_card_sha256": sha256_file(run_card_path),
        "model_weights_constructed": False,
        "model_forwards": 0,
        "backward_optimizer_scheduler_steps": 0,
        "groups": output,
    }


def self_test() -> None:
    accepted = reconcile_audit({"status": "INCOMPLETE", "global_step": 10000, "expected_end_step": 1000}, 10000, 1000)
    assert accepted["kind"] == "stale_import_time_default_reconciled"
    for audit in (
        {"status": "INCOMPLETE", "global_step": 9999, "expected_end_step": 1000},
        {"status": "INCOMPLETE", "global_step": 10000, "expected_end_step": 10000},
    ):
        try:
            reconcile_audit(audit, 10000, 1000)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid update audit was accepted")
    assert _changed_projector_samples({"a": "1", "b": "2"}, {"a": "1", "b": "3"}) == ["b"]
    try:
        _changed_projector_samples({"a": "1"}, {"b": "1"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched projector sample keys were accepted")
    print(json.dumps({"status": "PASS_POSTTRAIN_RECONCILIATION_SELF_TEST", "model_loads": 0, "model_forwards": 0, "optimizer_steps": 0}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--parameter-audit-only", action="store_true")
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--pbs-stdout", type=Path)
    parser.add_argument("--projector-evidence", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.parameter_audit_only:
        if args.run_card is None:
            parser.error("--run-card is required for --parameter-audit-only")
        print(json.dumps(audit_parameters(args.run_card), indent=2, sort_keys=True))
        return 0
    if args.run_card is None or args.pbs_stdout is None or args.projector_evidence is None:
        parser.error("--run-card, --pbs-stdout, and --projector-evidence are required")
    result = recover(args.run_card, args.pbs_stdout, args.projector_evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
