#!/usr/bin/env python3
"""Verify a copied PI0246 full Trainer checkpoint against its sealed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: Path, manifest_path: Path, step: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest["checkpoints"][str(step)]
    actual = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    listed = {row["path"]: row for row in receipt["entries"]}
    if len(actual) != int(receipt["file_count"]) or {p.relative_to(root).as_posix() for p in actual} != set(listed):
        raise RuntimeError("archive file inventory differs from sealed checkpoint receipt")
    tree = hashlib.sha256()
    for path in actual:
        rel = path.relative_to(root).as_posix()
        row = listed[rel]
        size = path.stat().st_size
        digest = sha(path)
        if size != int(row["bytes"]) or digest != row["sha256"]:
            raise RuntimeError(f"archive content mismatch: {rel}")
        tree.update(f"pi0241/checkpoint-{step}/{rel}\0{size}\0{digest}\n".encode())
    if tree.hexdigest() != receipt["tree_sha256"]:
        raise RuntimeError("archive tree SHA256 differs from sealed checkpoint receipt")
    return {"status": "PASS_CHECKPOINT_ARCHIVE_READBACK", "step": step,
            "file_count": len(actual), "total_bytes": sum(p.stat().st_size for p in actual),
            "tree_sha256": tree.hexdigest(),
            "training_manifest_sha256": sha(manifest_path),
            "verifier_script_sha256": sha(Path(__file__).resolve())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-manifest", type=Path, required=True)
    p.add_argument("--step", type=int, choices=(1000, 2000), required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.checkpoint, a.training_manifest, a.step), sort_keys=True))


if __name__ == "__main__":
    main()
