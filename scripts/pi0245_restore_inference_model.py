#!/usr/bin/env python3
"""Restore only pinned inference files; verify every byte against sealed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted(name: str) -> bool:
    return name in {"config.json", "embodiment_id.json", "processor_config.json", "statistics.json", "model.safetensors.index.json"} or name.endswith(".safetensors") or name.startswith("experiment_cfg/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=10000)
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
    assert len(entries) >= 8 and sum(entry["path"].endswith(".safetensors") for entry in entries) >= 2
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    results = []
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
    record = {"schema": "PI0245_PINNED_INFERENCE_MODEL_RESTORE_V1", "checkpoint_step": args.expected_step, "receipt_sha256": sha(args.receipt), "repo_id": repo, "prefix": prefix, "revision": revision, "model_root": str(args.model_root), "file_count": len(results), "files": results, "optimizer_restored": False}
    target = args.model_root / "pi0245_inference_restore.json"
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    if target.exists() and target.read_text() != payload:
        raise RuntimeError("prior restore manifest differs")
    target.write_text(payload)
    print(json.dumps({"status": "PASS_PINNED_MODEL_ONLY_RESTORE", "model_root": str(args.model_root), "manifest_sha256": sha(target), "file_count": len(results), "repo_id": repo, "revision": revision}))


if __name__ == "__main__":
    main()
