#!/usr/bin/env python3
"""Exercise the PI0247 read-only finalizer against synthetic completed evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(run_card: Path, output_root: Path, runner_script: Path | None = None) -> dict[str, Any]:
    card = json.loads(run_card.read_text(encoding="utf-8"))
    real_root = Path(card["paths"]["output_root"]).resolve()
    fixture_root = output_root.resolve()
    if fixture_root == real_root or real_root in fixture_root.parents or fixture_root.exists():
        raise RuntimeError("synthetic finalizer output must be a new directory outside the real arm output")

    repo = Path(card["paths"]["repo_code_root"])
    shared = Path(card["paths"]["code_root"])
    training_runtime = Path(card["paths"]["training_runtime_root"])
    sys.path[:0] = [str(repo / "scripts"), str(shared / "scripts"), str(training_runtime)]
    runner_path = runner_script or repo / "scripts/pi0247_lu_train.py"
    spec = importlib.util.spec_from_file_location("pi0247_lu_finalizer_under_test", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed PI0247 finalizer")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    if runner_script is None:
        card["source"]["execution_code_sha256"]["pi0247_lu_train.py"] = runner.sha256_file(runner_path)
    else:
        os.environ["PI0247_FINALIZER_CODE_COMMIT"] = "SELF_CHECK"

    output = fixture_root / "results"
    card["paths"]["output_root"] = str(output)
    synthetic_card = fixture_root / "run_card.json"
    write_json(synthetic_card, card)
    card_sha = runner.sha256_file(synthetic_card)
    write_json(fixture_root / "run_card.seal.json", {"sha256": card_sha})
    output.mkdir(parents=True)
    storage = runner._validate_storage(card)
    if storage["persistent_free_bytes_before_training"] < storage["required_persistent_bytes"]:
        raise AssertionError("storage preflight accepted less than one checkpoint plus reserve")
    prefix = card["archive"]["path_prefix"]
    probe_path = f"control/PI0247/selfcheck/{card['arm']}.json"
    write_json(output / "hf_preflight_receipt.json", {
        "schema": "PI0247_LU_HF_ONLINE_PREFLIGHT_V1", "status": "PASS",
        "arm": card["arm"], "run_card_sha256": card_sha,
        "repo_id": card["archive"]["repo_id"], "target_prefix": prefix,
        "target_prefix_empty": True, "repo_public": True, "identity_verified": True,
        "write_probe_path": probe_path,
        "write_probe": {"readback": "PASS", "path": probe_path},
    })
    write_json(output / "pre_model_preflight.json", {
        "status": "PASS_PRE_MODEL_PREFLIGHT", "model_weights_downloaded": False,
    })
    write_json(output / "execution_state.json", {
        "status": "TRAINING_STARTED", "arm": card["arm"], "run_card_sha256": card_sha,
        "pbs_job_id": "synthetic-finalizer-selfcheck",
    })

    cohort = json.loads(Path(card["cohort"]["path"]).read_text(encoding="utf-8"))["rows"]
    schedule = json.loads(Path(card["schedule"]["path"]).read_text(encoding="utf-8"))["rows"]
    training = output / "training"
    training.mkdir()
    with (training / "pi0241_sample_consumption.jsonl").open("w", encoding="utf-8") as handle:
        for ordinal, plan in enumerate(schedule, start=1):
            row = cohort[int(plan["fixed_index"])]
            handle.write(json.dumps({
                "yield_ordinal": ordinal, "optimizer_step": ordinal,
                "fixed_index": int(plan["fixed_index"]), "episode_index": int(row["episode_index"]),
                "source_episode_id": row["source_episode_id"], "observation_index": 0,
            }) + "\n")
    with (training / "pi0241_sample_contracts.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(cohort):
            handle.write(json.dumps({
                "fixed_index": index, "source_episode_id": row["source_episode_id"],
                "active_slots": list(range(10)), "action_shape": [1, 40, 132],
                "mask_shape": [1, 40, 132], "action_sha256": "synthetic",
                "mask_sha256": "synthetic",
            }) + "\n")

    update_audit = {
        "status": "COMPLETE", "global_step": 2000,
        "optimizer_membership": {"missing_trainable": [], "illegal_frozen": []},
        "final_parameters": {
            "projector": {"delta_l2": 1.0}, "diffusion": {"delta_l2": 1.0},
            "frozen_backbone": {"delta_l2": 0.0}, "frozen_vlln": {"delta_l2": 0.0},
        },
    }
    write_json(training / "pi0241_update_audit.json", update_audit)
    objective_path = training / "pi0247_L_local_forward.jsonl"
    with objective_path.open("w", encoding="utf-8") as handle:
        for index in range(1, 2001):
            handle.write(json.dumps({
                "forward_index": index, "native_loss": 1.0,
                "physical_position_loss": 2.0, "physical_delta_loss": 3.0,
                "total_loss": 1.5,
            }) + "\n")

    sys.path.insert(0, str(shared / "scripts"))
    import pi0243_checkpoint_persistence as persistence
    train_core = runner.load_module(
        training_runtime / "pi0246_gate_pass_train.py", "pi0247_lu_finalizer_selfcheck_core",
    )

    checkpoint_records = {}
    for step in (1000, 2000):
        checkpoint = training / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in persistence.REQUIRED_FILES:
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "trainer_state.json":
                write_json(path, {"global_step": step, "max_steps": 2000})
            else:
                path.write_bytes(("synthetic:" + name).encode())
        record = persistence.checkpoint_record(
            checkpoint, expected_step=step, expected_max_steps=2000, family=card["family"],
        )
        checkpoint_records[step] = record
        remote_files = [{
            "path": f"{prefix}/checkpoint-{step}/{entry['path']}",
            "bytes": entry["bytes"], "sha256": entry["sha256"],
        } for entry in record["inventory"]["entries"]]
        revision = f"synthetic-selfcheck-{step}"
        write_json(output / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json", {
            "status": "DURABLE_VERIFIED", "family": card["family"],
            "checkpoint_step": step, "max_steps": 2000, "durable_host": "huggingface",
            "durable_path": f"{card['archive']['repo_id']}/{prefix}/checkpoint-{step}",
            "hf_manifest_commit_id": revision, "source_record": record,
            "durable_inventory": remote_files,
        })
        readback_path = output / "hf_readbacks" / f"checkpoint-{step}.json"
        write_json(readback_path, {
            "schema": "PI0247_LU_FULL_HF_READBACK_V1",
            "status": "PASS_PT_FULL_HF_READBACK", "pi0247_status": "PASS_PI0247_LU_FULL_HF_READBACK",
            "arm": card["arm"], "family": card["family"], "checkpoint_step": step,
            "max_steps": 2000, "repo_id": card["archive"]["repo_id"], "path_prefix": prefix,
            "source_tree_sha256": record["inventory"]["tree_sha256"],
            "source_record_sha256": runner.canonical_sha256(record),
            "revision": revision, "file_count": len(remote_files), "files": remote_files,
        })
        projector_path = output / "projector_reconciliation" / f"checkpoint-{step}.json"
        write_json(projector_path, {
            "schema": "PI0247_LU_PROJECTOR_RECONCILIATION_V1",
            "status": "PASS_FULL_PROJECTOR_TENSOR_COMPARE", "arm": card["arm"],
            "checkpoint_step": step, "run_card_sha256": card_sha,
            "initializer_tree_sha256": card["initializer"]["full_checkpoint_tree_sha256"],
            "checkpoint_tree_sha256": record["inventory"]["tree_sha256"],
            "reconciliation_script_sha256": runner.sha256_file(shared / "scripts/pi0241_reconcile_update_integrity.py"),
            "comparison": {"tensor_count": 1, "changed_tensor_count": 1},
        })
        train_core.retire_pt_checkpoint_after_verified_archive(
            output, training, step, record,
            output / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json",
            readback_path, repo_id=card["archive"]["repo_id"], prefix=prefix, family=card["family"],
        )
        write_json(training / f"checkpoint-{step}" / "experiment_cfg" / "sidecar_probe.json", {"kept": True})

    args = SimpleNamespace(run_card=synthetic_card, arm="L")
    runner.finalize(args)
    manifest_path = output / "training_manifest.json"
    manifest_sha = runner.sha256_file(manifest_path)
    original_objective = objective_path.read_bytes()
    rows = original_objective.decode("utf-8").splitlines()
    first = json.loads(rows[0])
    first["total_loss"] = 1.51
    rows[0] = json.dumps(first)
    objective_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    coefficient_mismatch_rejected = False
    try:
        runner.finalize(args)
    except runner.LUError:
        coefficient_mismatch_rejected = True
    if not coefficient_mismatch_rejected:
        raise AssertionError("finalizer accepted an objective that differs from its bound coefficients")
    objective_path.write_bytes(original_objective)
    runner.finalize(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS_TRAINING_INTEGRITY" or manifest["selected_checkpoint"] != "NONE":
        raise AssertionError("synthetic finalizer did not close the unselected training manifest")
    leftover = training / "checkpoint-1000" / checkpoint_records[1000]["inventory"]["entries"][0]["path"]
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"incomplete-local-copy")
    full_state_copy_rejected = False
    try:
        runner.finalize(args)
    except runner.LUError:
        full_state_copy_rejected = True
    finally:
        leftover.unlink(missing_ok=True)
    if not full_state_copy_rejected:
        raise AssertionError("finalizer accepted a remaining archived full-state file")
    runner.finalize(args)

    resume_root = fixture_root / "resume_test"
    resume_training = resume_root / "training"
    resume_training.mkdir(parents=True)
    resume_checkpoint = resume_training / "checkpoint-1000"
    resume_checkpoint.mkdir()
    for name in persistence.REQUIRED_FILES:
        path = resume_checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "trainer_state.json":
            write_json(path, {"global_step": 1000, "max_steps": 2000})
        else:
            path.write_bytes(("synthetic:" + name).encode())
    restore_receipt_path = resume_root / "resume_restore_receipt.json"
    restored, source_record, restore_receipt = runner._restore_checkpoint_from_hf(
        card, output, resume_training, 1000, persistence, train_core, restore_receipt_path,
    )
    if restored != resume_checkpoint or restore_receipt["restored_from_hf"] is not False:
        raise AssertionError("exact resume did not bind the local checkpoint to its pinned HF receipt")
    from transformers import Trainer

    original_rng_load = Trainer._load_rng_state
    try:
        Trainer._load_rng_state = lambda _self, _path: "rng-restored"
        runner._install_resume_checkpoint_release(
            restored, resume_root, card, persistence, source_record,
            output / "durable_receipts/checkpoint-1000.durable_receipt.json",
            output / "hf_readbacks/checkpoint-1000.json", restore_receipt_path,
        )
        rng_result = Trainer._load_rng_state(SimpleNamespace(), str(restored))
    finally:
        Trainer._load_rng_state = original_rng_load
    release = json.loads((resume_root / "resume_checkpoint_release.json").read_text(encoding="utf-8"))
    if rng_result != "rng-restored" or restored.exists() or release.get("status") != "RELEASED_AFTER_OPTIMIZER_SCHEDULER_AND_RNG_RESTORE":
        raise AssertionError("exact resume checkpoint was not released after Trainer RNG restoration")
    return {
        "schema": "PI0247_LU_FINALIZER_SELF_CHECK_V1", "status": "PASS",
        "arm": card["arm"], "synthetic_updates": 2000,
        "coefficient_mismatch_rejected": coefficient_mismatch_rejected,
        "residual_config_sidecars_accepted": True,
        "full_state_copy_rejected": full_state_copy_rejected,
        "projector_reconciliation_hf_retirement_and_exact_resume_paths_exercised": True,
        "storage_quota_probe_path": storage["persistent_capacity_probe_path"],
        "model_loads": 0, "selected_checkpoint": manifest["selected_checkpoint"],
        "manifest_sha256": manifest_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run_card, args.output_root, args.runner_script), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
