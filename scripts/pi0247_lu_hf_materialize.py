#!/usr/bin/env python3
"""Materialize only L inference weights from the exact public HF revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pi0247_lu_final_screen import wanted_checkpoint_file


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace conflicting materialization evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    card, manifest = read(args.run_card), read(args.training_manifest)
    card_sha, manifest_sha = sha(args.run_card), sha(args.training_manifest)
    if read(args.run_card.with_name("run_card.seal.json")).get("sha256") != card_sha:
        raise RuntimeError("L run card seal mismatch")
    step = args.step
    record = manifest.get("checkpoints", {}).get(str(step), {})
    durable, readback, state = read(args.durable_receipt), read(args.hf_readback), read(args.execution_state)
    durable_sha, readback_sha, state_sha = sha(args.durable_receipt), sha(args.hf_readback), sha(args.execution_state)
    if (
        card.get("arm") != "L" or card.get("family") != manifest.get("family")
        or card.get("archive", {}).get("public") is not True
        or manifest.get("status") != "PASS_TRAINING_INTEGRITY"
        or manifest.get("arm") != "L" or manifest.get("global_step") != 2000
        or manifest.get("selected_checkpoint") != "NONE"
        or manifest.get("run_card_sha256") != card_sha
        or manifest.get("execution_code_commit") != card.get("source", {}).get("execution_code_commit")
        or state.get("status") != "TRAINING_COMPLETE"
        or state.get("training_manifest_sha256") != manifest_sha
        or record.get("step") != step
        or record.get("family") != manifest.get("family")
        or sha(args.durable_receipt) != record.get("durable_receipt_sha256")
        or sha(args.hf_readback) != record.get("full_hf_readback_sha256")
    ):
        raise RuntimeError(f"L checkpoint-{step} lineage does not match its closed training evidence")
    inventory = durable.get("source_record", {}).get("inventory", {})
    if (
        durable.get("status") != "DURABLE_VERIFIED"
        or durable.get("family") != card.get("family")
        or durable.get("checkpoint_step") != step
        or inventory.get("tree_sha256") != record.get("tree_sha256")
        or inventory.get("file_count") != record.get("file_count")
        or inventory.get("total_bytes") != record.get("total_bytes")
        or readback.get("status") != "PASS_PT_FULL_HF_READBACK"
        or readback.get("pi0247_status") != "PASS_PI0247_LU_FULL_HF_READBACK"
        or readback.get("repo_id") != card["archive"]["repo_id"]
        or readback.get("path_prefix") != card["archive"]["path_prefix"]
        or readback.get("revision") != record.get("hf_manifest_commit_id")
        or readback.get("source_tree_sha256") != record.get("tree_sha256")
    ):
        raise RuntimeError(f"L checkpoint-{step} persistent and HF receipts disagree")

    files = [
        {key: row[key] for key in ("path", "bytes", "sha256")}
        for row in inventory.get("entries", []) if wanted_checkpoint_file(row["path"])
    ]
    if ({"config.json", "model.safetensors.index.json"} - {row["path"] for row in files}
            or sum(row["path"].endswith(".safetensors") for row in files) < 2):
        raise RuntimeError(f"L checkpoint-{step} lacks a complete inference weight set")
    prefix = f"{card['archive']['path_prefix']}/checkpoint-{step}"
    readback_files = {row["path"]: row for row in readback.get("files", [])}
    for row in files:
        remote_path = f"{prefix}/{row['path']}"
        if readback_files.get(remote_path, {}).get("sha256") != row["sha256"]:
            raise RuntimeError(f"HF readback omits or changes {remote_path}")

    snapshot_download(
        repo_id=card["archive"]["repo_id"], repo_type="model",
        revision=record["hf_manifest_commit_id"],
        allow_patterns=[f"{prefix}/{row['path']}" for row in files],
        local_dir=str(args.output_root), token=False,
    )
    checkpoint = args.output_root / prefix
    for row in files:
        path = checkpoint / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError(f"HF download differs from full readback: {path}")

    receipt = {
        "schema": "PI0247_LU_HF_INFERENCE_MATERIALIZATION_V1",
        "arm": "L", "checkpoint_step": step,
        "repo_id": card["archive"]["repo_id"], "path_prefix": card["archive"]["path_prefix"],
        "revision": record["hf_manifest_commit_id"],
        "source_checkpoint_tree_sha256": record["tree_sha256"],
        "run_card_sha256": card_sha, "training_manifest_sha256": manifest_sha,
        "durable_receipt_sha256": durable_sha, "hf_readback_sha256": readback_sha,
        "execution_state_sha256": state_sha, "files": files,
        "materialized_checkpoint_root": str(checkpoint.resolve()),
        "optimizer_downloaded": False,
    }
    write_immutable(args.receipt_output, receipt)
    return {"status": "PASS_HF_INFERENCE_MATERIALIZATION", "checkpoint_step": step,
            "revision": receipt["revision"], "files": len(files),
            "materialized_checkpoint_root": str(checkpoint.resolve()),
            "receipt": str(args.receipt_output), "receipt_sha256": sha(args.receipt_output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(1000, 2000), required=True)
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--execution-state", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path, required=True)
    parser.add_argument("--hf-readback", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    print(json.dumps(materialize(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
