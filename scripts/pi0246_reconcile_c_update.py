#!/usr/bin/env python3
"""Read-only reconciliation of PI0246 C training's sampled projector probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

import pi0241_fixed16_learning as fixed16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index(root: Path) -> dict:
    return read(root / "model.safetensors.index.json")["weight_map"]


def tensor(root: Path, weight_map: dict, name: str) -> np.ndarray:
    with safe_open(str(root / weight_map[name]), framework="np") as f:
        return np.asarray(f.get_tensor(name))


def compare_projector(base: Path, checkpoint: Path) -> dict:
    base_map, trained_map = index(base), index(checkpoint)
    base_names = {name for name in base_map if fixed16._parameter_group(name) == "projector"}
    trained_names = {name for name in trained_map if fixed16._parameter_group(name) == "projector"}
    if not base_names or base_names != trained_names:
        raise RuntimeError(f"projector key mismatch: baseline={len(base_names)} trained={len(trained_names)}")
    rows = []
    for name in sorted(base_names):
        before, after = tensor(base, base_map, name), tensor(checkpoint, trained_map, name)
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"tensor contract changed for {name}")
        delta = after.astype(np.float64) - before.astype(np.float64)
        rows.append({
            "name": name,
            "shape": list(before.shape),
            "dtype": str(before.dtype),
            "changed_elements": int(np.count_nonzero(delta)),
            "numel": int(delta.size),
            "delta_l2": float(np.linalg.norm(delta.ravel())),
            "delta_max_abs": float(np.max(np.abs(delta))),
        })
    return {
        "tensor_count": len(rows),
        "changed_tensor_count": sum(row["changed_elements"] > 0 for row in rows),
        "changed_element_count": sum(row["changed_elements"] for row in rows),
        "max_abs_delta": max(row["delta_max_abs"] for row in rows),
        "tensors": rows,
    }


def reconcile(card_path: Path, training: Path, output: Path) -> dict:
    card = read(card_path)
    seal = read(card_path.with_name("run_card.seal.json"))
    if seal.get("sha256") != sha(card_path):
        raise RuntimeError("C sealed run-card hash mismatch")
    manifest_path = training.parent / "training_manifest.json"
    manifest = read(manifest_path)
    if manifest.get("run_card_sha256") != sha(card_path):
        raise RuntimeError("C training manifest is not bound to this run card")
    if Path(fixed16.__file__).resolve() != (Path(card["paths"]["code_root"]) / "scripts/pi0241_fixed16_learning.py").resolve():
        raise RuntimeError("parameter classifier import is not the run-card code root")
    audit_path = training / "pi0241_update_audit.json"
    audit = read(audit_path)
    schedule = read(Path(card["schedule"]["path"]))["rows"]
    consumed = [json.loads(line) for line in (training / "pi0241_sample_consumption.jsonl").read_text().splitlines() if line.strip()]
    contracts = [json.loads(line) for line in (training / "pi0241_sample_contracts.jsonl").read_text().splitlines() if line.strip()]
    if len(schedule) != 2000 or len(consumed) != 2000 or [r["source_episode_id"] for r in consumed] != [r["source_episode_id"] for r in schedule]:
        raise RuntimeError("C sample-consumption evidence is incomplete or differs from the sealed schedule")
    if len(contracts) != 16 or {int(r["fixed_index"]) for r in contracts} != set(range(16)):
        raise RuntimeError("C action/mask contract evidence is incomplete")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != 2000:
        raise RuntimeError("C update audit is not complete at step 2000")
    original_integrity = manifest.get("update_integrity", {})
    non_projector_checks = {key: value for key, value in original_integrity.items() if key != "projector_updated"}
    if not non_projector_checks or not all(non_projector_checks.values()) or original_integrity.get("projector_updated") is not False:
        raise RuntimeError(f"C failure is not isolated to the sampled projector probe: {original_integrity}")
    membership = audit["optimizer_membership"]
    if membership["missing_trainable"] or membership["illegal_frozen"]:
        raise RuntimeError("optimizer membership audit failed")
    updates = {}
    for step in (1000, 2000):
        checkpoint = training / f"checkpoint-{step}"
        state = read(checkpoint / "trainer_state.json")
        receipt = manifest["checkpoints"][str(step)]
        if int(state["global_step"]) != step or int(receipt["global_step"]) != step or int(receipt["file_count"]) != 18:
            raise RuntimeError(f"checkpoint-{step} full Trainer receipt mismatch")
        updates[str(step)] = compare_projector(Path(card["paths"]["base_model_path"]), checkpoint)
    final = audit["final_parameters"]
    status = "TRAINING_INTEGRITY_RECONCILED_PROJECTOR_SAMPLE_FALSE_NEGATIVE"
    if any(not updates[str(step)]["changed_tensor_count"] for step in (1000, 2000)):
        status = "PROJECTOR_UPDATE_NOT_CONFIRMED"
    result = {
        "schema": "PI0246_C_READONLY_UPDATE_RECONCILIATION_V1",
        "status": status,
        "source": {
            "reconciliation_script_sha256": sha(Path(__file__).resolve()),
            "parameter_classifier_sha256": sha(Path(fixed16.__file__).resolve()),
            "campaign_trainer_sha256": sha(Path(card["paths"]["campaign_code_root"]) / "pi0246_gate_pass_train.py"),
            "native_overlay_sha256": sha(Path(card["paths"]["campaign_code_root"]) / "pi0245_native_overlay.py"),
            "run_card_sha256": sha(card_path),
            "training_manifest_sha256": sha(manifest_path),
            "original_training_status": manifest.get("status"),
            "update_audit_sha256": sha(audit_path),
            "schedule_sha256": sha(Path(card["schedule"]["path"])),
            "consumed_rows": len(consumed),
            "action_mask_contracts": len(contracts),
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        },
        "original_projector_probe": final.get("projector"),
        "original_update_integrity": original_integrity,
        "original_frozen_and_optimizer_checks": {
            key: final.get(key) for key in ("diffusion", "frozen_backbone", "frozen_vlln")
        },
        "full_projector_tensor_differences_vs_r0": updates,
        "interpretation": "The training wrapper sampled only the first 4096 values of one representative projector tensor. The read-only full tensor comparison is authoritative for whether projector tensors changed; original manifests remain untouched.",
        "selected_checkpoint": "NONE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-card", type=Path, required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(reconcile(args.run_card, args.training_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
