#!/usr/bin/env python3
"""Exercise the PI0247 read-only finalizer against synthetic completed evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(run_card: Path, output_root: Path) -> dict[str, Any]:
    card = json.loads(run_card.read_text(encoding="utf-8"))
    real_root = Path(card["paths"]["output_root"]).resolve()
    fixture_root = output_root.resolve()
    if fixture_root == real_root or real_root in fixture_root.parents or fixture_root.exists():
        raise RuntimeError("synthetic finalizer output must be a new directory outside the real arm output")

    repo = Path(card["paths"]["repo_code_root"])
    shared = Path(card["paths"]["code_root"])
    training_runtime = Path(card["paths"]["training_runtime_root"])
    sys.path[:0] = [str(repo / "scripts"), str(shared / "scripts"), str(training_runtime)]
    runner_path = repo / "scripts/pi0247_lu_train.py"
    spec = importlib.util.spec_from_file_location("pi0247_lu_finalizer_under_test", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed PI0247 finalizer")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    output = fixture_root / "results"
    card["paths"]["output_root"] = str(output)
    synthetic_card = fixture_root / "run_card.json"
    write_json(synthetic_card, card)
    card_sha = runner.sha256_file(synthetic_card)
    write_json(fixture_root / "run_card.seal.json", {"sha256": card_sha})
    output.mkdir(parents=True)
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
        persistence.publish_checkpoint(
            checkpoint, durable_host=None, durable_root=output,
            expected_step=step, expected_max_steps=2000, family=card["family"],
            receipt_root=output / "persistent_receipts", job_id="synthetic-finalizer-selfcheck",
        )
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
        write_json(output / "hf_readbacks" / f"checkpoint-{step}.json", {
            "status": "PASS_PI0247_LU_FULL_HF_READBACK", "arm": card["arm"],
            "source_tree_sha256": record["inventory"]["tree_sha256"],
            "revision": revision, "files": remote_files,
        })

    original_load_module = runner.load_module

    class SyntheticProjectorReconciliation:
        @staticmethod
        def projector_tensors(_root: Path) -> dict[str, int]:
            return {"synthetic": 1}

        @staticmethod
        def compare(_before: dict[str, int], _after: dict[str, int]) -> dict[str, int]:
            return {"tensor_count": 1, "changed_tensor_count": 1}

    def load_module(path: Path, name: str) -> Any:
        if name == "pi0247_lu_finalize_projector":
            return SyntheticProjectorReconciliation
        return original_load_module(path, name)

    runner.load_module = load_module
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
    return {
        "schema": "PI0247_LU_FINALIZER_SELF_CHECK_V1", "status": "PASS",
        "arm": card["arm"], "synthetic_updates": 2000,
        "coefficient_mismatch_rejected": coefficient_mismatch_rejected,
        "projector_reconciliation_path_exercised": True,
        "model_loads": 0, "selected_checkpoint": manifest["selected_checkpoint"],
        "manifest_sha256": manifest_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_card, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
