#!/usr/bin/env python3
"""Restore only pinned inference files; verify every byte against sealed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted(name: str) -> bool:
    return name in {"config.json", "embodiment_id.json", "processor_config.json", "statistics.json", "model.safetensors.index.json"} or name.endswith(".safetensors") or name.startswith("experiment_cfg/")


def validate_model_index(model_root: Path, entries: list[dict]) -> None:
    index = json.loads((model_root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model index has no non-empty weight_map")
    referenced = set(weight_map.values())
    available = {entry["path"] for entry in entries if entry["path"].endswith(".safetensors")}
    if len(referenced) < 2 or referenced != available:
        raise RuntimeError("inference view does not contain exactly the indexed model shards")


def inference_identity(receipt: dict, repo: str, archive_prefix: str, step: int, model_family: str | None) -> dict:
    if model_family is None:
        return {"prefix": archive_prefix}
    if receipt.get("family") != model_family:
        raise RuntimeError("requested model family differs from durable checkpoint receipt")
    return {
        "family": model_family,
        "prefix": f"{model_family}/checkpoint-{step}",
        "hf_archive_path": f"{repo}/{archive_prefix}",
    }


def inference_view_receipt(restore: dict, full_receipt_sha256: str, source_tree_sha256: str,
                           restore_manifest_sha256: str) -> dict:
    files = restore["files"]
    return {
        "schema": "PI0246_INFERENCE_VIEW_VERIFICATION_V1",
        "status": "INFERENCE_VIEW_VERIFIED",
        "family": restore["family"],
        "checkpoint_step": restore["checkpoint_step"],
        "repo_id": restore["repo_id"],
        "hf_archive_path": restore["hf_archive_path"],
        "prefix": restore["prefix"],
        "revision": restore["revision"],
        "full_checkpoint_receipt_sha256": full_receipt_sha256,
        "source_tree_sha256": source_tree_sha256,
        "restore_manifest_sha256": restore_manifest_sha256,
        "optimizer_restored": False,
        "file_count": len(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=10000)
    parser.add_argument("--model-family")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    assert args.expected_step > 0
    assert receipt["status"] == "DURABLE_VERIFIED" and receipt["checkpoint_step"] == args.expected_step
    parts = receipt["durable_path"].split("/")
    assert len(parts) >= 4
    repo = "/".join(parts[:2])
    prefix = "/".join(parts[2:])
    revision = receipt["hf_manifest_commit_id"]
    assert len(revision) == 40
    entries = [entry for entry in receipt["source_record"]["inventory"]["entries"] if wanted(entry["path"])]
    names = {entry["path"] for entry in entries}
    assert {"config.json", "model.safetensors.index.json"} <= names
    assert sum(name.endswith(".safetensors") for name in names) >= 2
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    from huggingface_hub import hf_hub_download

    for entry in entries:
        relative = entry["path"]
        target = args.model_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(hf_hub_download(repo_id=repo, filename=f"{prefix}/{relative}", revision=revision, cache_dir=args.cache_root))
        if source.stat().st_size != entry["bytes"] or sha(source) != entry["sha256"]:
            raise RuntimeError(f"HF pinned file mismatch: {relative}")
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"conflicting prior link: {target}")
        elif target.exists():
            if target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]:
                raise RuntimeError(f"conflicting prior file: {target}")
        else:
            os.symlink(source, target)
        results.append({"path": relative, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    validate_model_index(args.model_root, entries)
    identity = inference_identity(receipt, repo, prefix, args.expected_step, args.model_family)
    record = {"schema": "PI0245_PINNED_INFERENCE_MODEL_RESTORE_V1", "checkpoint_step": args.expected_step, "receipt_sha256": sha(args.receipt), "repo_id": repo, **identity, "revision": revision, "model_root": str(args.model_root), "file_count": len(results), "files": results, "optimizer_restored": False}
    target = args.model_root / "pi0245_inference_restore.json"
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    if target.exists() and target.read_text() != payload:
        raise RuntimeError("prior restore manifest differs")
    target.write_text(payload)
    extra = {}
    if args.model_family is not None:
        view_receipt = inference_view_receipt(
            record,
            sha(args.receipt),
            receipt["source_record"]["inventory"]["tree_sha256"],
            sha(target),
        )
        view_receipt_path = args.model_root / "INFERENCE_VIEW_VERIFIED.json"
        view_payload = json.dumps(view_receipt, sort_keys=True, separators=(",", ":")) + "\n"
        if view_receipt_path.exists() and view_receipt_path.read_text() != view_payload:
            raise RuntimeError("prior inference-view verification receipt differs")
        view_receipt_path.write_text(view_payload)
        extra["view_receipt_sha256"] = sha(view_receipt_path)
    print(json.dumps({"status": "PASS_PINNED_MODEL_ONLY_RESTORE", "model_root": str(args.model_root), "manifest_sha256": sha(target), "file_count": len(results), "repo_id": repo, "revision": revision, **extra}))


if __name__ == "__main__":
    main()
