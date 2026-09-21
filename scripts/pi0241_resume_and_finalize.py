#!/usr/bin/env python3
"""Recover PI0241's lost final checkpoint without changing the run recipe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from pi0241_fixed16_learning import (
    CHECKPOINT_STEPS,
    MAX_STEPS,
    PI0241Error,
    _resolved_config,
    _verify_card,
    bind_runtime,
    checkpoint_record,
    install_fixed_dataset_factory,
    install_update_audit,
    load_module,
    read_json,
    sha256_file,
    write_json,
)


def _archive(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                payload = read_json(path)
                if payload.get("schema") == "PI0241_CHECKPOINT_ARCHIVE_V1" and payload.get("status") == "COMPLETE":
                    return payload
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(10)
    raise PI0241Error(f"complete checkpoint archive not available: {path}")


def _combined_audit(initial: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    events = [dict(event, audit_phase="initial") for event in initial.get("events", [])]
    events.extend(dict(event, audit_phase="recovery_from_checkpoint_500") for event in recovery.get("events", []))
    post = [event for event in events if event.get("event") == "POST_OPTIMIZER"]

    def changed(group: str) -> bool:
        return any(float(event.get("parameters", {}).get(group, {}).get("delta_l2", 0.0)) > 0.0 for event in post)

    def unchanged(group: str) -> bool:
        values = [event.get("parameters", {}).get(group, {}).get("delta_l2") for event in post]
        return bool(values) and all(float(value) == 0.0 for value in values)

    initial_membership = initial.get("optimizer_membership", {})
    recovery_membership = recovery.get("optimizer_membership", {})
    valid_membership = all(
        not membership.get("missing_trainable") and not membership.get("illegal_frozen")
        for membership in (initial_membership, recovery_membership)
    )
    combined = dict(initial)
    combined.update(
        {
            "status": "COMPLETE",
            "global_step": MAX_STEPS,
            "events": events,
            "recovery": {
                "source_checkpoint": 500,
                "audit_path": recovery.get("_path"),
                "audit_sha256": recovery.get("_sha256"),
                "replayed_optimizer_steps": 500,
            },
            "final_parameters": recovery.get("final_parameters"),
        }
    )
    combined["update_integrity"] = {
        "projector_updated": changed("projector"),
        "diffusion_updated": changed("diffusion"),
        "frozen_backbone_unchanged": unchanged("frozen_backbone"),
        "frozen_vlln_unchanged": unchanged("frozen_vlln"),
        "optimizer_membership_valid": valid_membership,
    }
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--archive-timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()
    if args.authorization != "PI0241_FIXED16_1000_GO":
        raise PI0241Error("training authorization token mismatch")

    card = _verify_card(args.run_card)
    bind_runtime(card)
    paths = card["paths"]
    output_root = Path(paths["output_root"])
    training = output_root / "training"
    checkpoint500 = training / "checkpoint-500"
    checkpoint1000 = training / "checkpoint-1000"
    if not checkpoint500.is_dir():
        raise PI0241Error("complete checkpoint-500 is required for recovery")
    checkpoint_record(checkpoint500, 500)
    marker = output_root / "pi0241_recovery_manifest.json"
    if marker.exists():
        raise PI0241Error(f"refusing to reuse recovery output: {marker}")

    if checkpoint1000.exists():
        quarantine = training / f"checkpoint-1000.incomplete-{args.failed_job_id}"
        if quarantine.exists():
            raise PI0241Error(f"recovery quarantine already exists: {quarantine}")
        os.replace(checkpoint1000, quarantine)
        write_json(output_root / "pi0241_incomplete_checkpoint_quarantine.json", {
            "schema": "PI0241_INCOMPLETE_CHECKPOINT_QUARANTINE_V1",
            "failed_job_id": args.failed_job_id,
            "path": str(quarantine),
            "reason": "checkpoint-1000 save failed with Disk quota exceeded after optimizer step 1000",
        })

    recovery_root = output_root / "recovery_audit"
    recovery_root.mkdir(parents=False, exist_ok=False)
    install_fixed_dataset_factory(card, training)
    install_update_audit(recovery_root)
    config_args = argparse.Namespace(
        code_root=Path(paths["code_root"]),
        groot_source=Path(paths["groot_source"]),
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        train_dataset=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
    )
    config, summary = _resolved_config(config_args, training)
    if summary != card["resolved_config"]:
        raise PI0241Error("recovery resolved config differs from sealed config")
    config.training.resume_from_checkpoint = True
    pi0228 = load_module(Path(paths["code_root"]) / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0241_pi0228_recovery")
    pi0228.run_official(config, Path(paths["groot_source"]))

    recovery_audit_path = recovery_root / "pi0241_update_audit.json"
    recovery_audit = read_json(recovery_audit_path)
    recovery_audit["_path"] = str(recovery_audit_path)
    recovery_audit["_sha256"] = sha256_file(recovery_audit_path)
    if recovery_audit.get("status") != "COMPLETE" or int(recovery_audit.get("global_step", -1)) != MAX_STEPS:
        raise PI0241Error("recovery update audit did not complete at step 1000")
    checkpoint1000_record = checkpoint_record(checkpoint1000, MAX_STEPS)

    archive = _archive(args.archive_manifest, args.archive_timeout_seconds)
    checkpoints = {str(step): (archive.get("checkpoints") or {}).get(str(step)) for step in CHECKPOINT_STEPS}
    if any(not isinstance(record, dict) or not record.get("verified") for record in checkpoints.values()):
        raise PI0241Error("checkpoint archive records are incomplete")
    checkpoints["1000"]["local_recovery_record"] = checkpoint1000_record

    initial_audit_path = training / "pi0241_update_audit.json"
    initial_audit = read_json(initial_audit_path)
    shutil.copy2(initial_audit_path, output_root / f"pi0241_update_audit_initial_{args.failed_job_id}.json")
    combined_audit = _combined_audit(initial_audit, recovery_audit)
    write_json(initial_audit_path, combined_audit)
    integrity = combined_audit["update_integrity"]
    training_manifest = {
        "schema": "PI0241_TRAINING_MANIFEST_V1",
        "status": "PASS_UPDATE_INTEGRITY" if all(integrity.values()) else "SCIENCE_FAIL_OPTIMIZATION_OR_CONTRACT",
        "canonical_optimizer_steps": MAX_STEPS,
        "actual_sample_consumption": 1000,
        "sample_contracts": [json.loads(line) for line in (training / "pi0241_sample_contracts.jsonl").read_text().splitlines()],
        "schedule_sha256": card["schedule"]["sha256"],
        "config": summary,
        "update_integrity": integrity,
        "checkpoints": checkpoints,
        "archive_manifest": {"path": str(args.archive_manifest), "sha256": sha256_file(args.archive_manifest), "repo_id": archive.get("repo_id")},
        "engineering_recovery": {"failed_job_id": args.failed_job_id, "source_checkpoint": 500, "replayed_optimizer_steps": 500},
        "selected_checkpoint": "NONE",
        "forbidden_counts": card["forbidden_counts"],
    }
    write_json(output_root / "pi0241_training_manifest.json", training_manifest)
    write_json(marker, {
        "schema": "PI0241_RECOVERY_MANIFEST_V1",
        "status": training_manifest["status"],
        "failed_job_id": args.failed_job_id,
        "source_checkpoint": 500,
        "replayed_optimizer_steps": 500,
        "training_manifest_sha256": sha256_file(output_root / "pi0241_training_manifest.json"),
        "checkpoint1000_tree_sha256": checkpoint1000_record["tree_sha256"],
        "archive_manifest_sha256": sha256_file(args.archive_manifest),
        "selected_checkpoint": "NONE",
    })
    print(json.dumps({"status": training_manifest["status"], "checkpoint1000": checkpoint1000_record, "selected_checkpoint": "NONE"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
