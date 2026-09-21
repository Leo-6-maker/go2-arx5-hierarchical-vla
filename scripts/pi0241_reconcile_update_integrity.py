#!/usr/bin/env python3
"""Reconcile PI0241's sampled update audit against sealed model tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


PROJECTOR_TOKENS = (
    "action_head.state_encoder",
    "action_head.action_encoder",
    "action_head.action_decoder",
    "action_head.position_embedding",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projector_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    index = read_json(model_dir / "model.safetensors.index.json")
    result: dict[str, np.ndarray] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(model_dir / shard), framework="np") as handle:
            for name in handle.keys():
                if any(token in name for token in PROJECTOR_TOKENS):
                    result[name] = np.asarray(handle.get_tensor(name))
    return result


def compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise RuntimeError("projector tensor key set changed")
    rows = []
    for name in sorted(reference):
        delta = candidate[name].astype(np.float64) - reference[name].astype(np.float64)
        rows.append(
            {
                "name": name,
                "reference_shape": list(reference[name].shape),
                "delta_l2": float(np.linalg.norm(delta.ravel())),
                "delta_max_abs": float(np.max(np.abs(delta))),
                "changed_elements": int(np.count_nonzero(delta)),
                "element_count": int(delta.size),
            }
        )
    return {
        "tensor_count": len(rows),
        "changed_tensor_count": sum(row["delta_l2"] > 0.0 for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--checkpoint-500", type=Path, required=True)
    parser.add_argument("--checkpoint-1000", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = read_json(args.training_manifest)
    archive = read_json(args.archive_manifest)
    if archive.get("status") != "COMPLETE":
        raise RuntimeError("checkpoint archive is not complete")
    for step in ("250", "500", "1000"):
        record = (archive.get("checkpoints") or {}).get(step)
        if not isinstance(record, dict) or not record.get("verified"):
            raise RuntimeError(f"archive record is not verified: checkpoint-{step}")
    if training.get("selected_checkpoint") != "NONE":
        raise RuntimeError("selected checkpoint is not NONE")

    baseline = projector_tensors(args.initializer)
    checkpoint500 = projector_tensors(args.checkpoint_500)
    checkpoint1000 = projector_tensors(args.checkpoint_1000)
    baseline_to_final = compare(baseline, checkpoint1000)
    checkpoint500_to_final = compare(checkpoint500, checkpoint1000)
    sample_name = "action_head.state_encoder.layer1.W"
    sample_delta = checkpoint500_to_final["rows"][
        next(index for index, row in enumerate(checkpoint500_to_final["rows"]) if row["name"] == sample_name)
    ]

    if baseline_to_final["changed_tensor_count"] != baseline_to_final["tensor_count"]:
        raise RuntimeError("not every projector tensor changed from the sealed initializer")
    if checkpoint500_to_final["changed_tensor_count"] != checkpoint500_to_final["tensor_count"]:
        raise RuntimeError("not every projector tensor changed during recovery")

    payload = {
        "schema": "PI0241_UPDATE_INTEGRITY_RECONCILIATION_V1",
        "status": "PASS_UPDATE_INTEGRITY",
        "reason": "sampled first-4096 audit window missed changes outside the representative window",
        "original_training_manifest_sha256": sha256_file(args.training_manifest),
        "archive_manifest_sha256": sha256_file(args.archive_manifest),
        "initializer_model_index_sha256": sha256_file(args.initializer / "model.safetensors.index.json"),
        "checkpoint500_model_index_sha256": sha256_file(args.checkpoint_500 / "model.safetensors.index.json"),
        "checkpoint1000_model_index_sha256": sha256_file(args.checkpoint_1000 / "model.safetensors.index.json"),
        "projector_delta": {
            "initializer_to_checkpoint1000": baseline_to_final,
            "checkpoint500_to_checkpoint1000": checkpoint500_to_final,
            "sampled_audit_representative": sample_delta,
        },
        "original_update_integrity": training.get("update_integrity"),
        "selected_checkpoint": "NONE",
        "forbidden_counts": training.get("forbidden_counts"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "changed_projector_tensors": baseline_to_final["changed_tensor_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
