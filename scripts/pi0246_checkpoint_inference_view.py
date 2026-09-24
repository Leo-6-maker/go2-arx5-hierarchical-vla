#!/usr/bin/env python3
"""Build a read-only inference view over a verified full Trainer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .pi0245_p1_stock import verify_model_files
    from .pi0246_verify_checkpoint_archive import verify as verify_archive
except ImportError:
    from pi0245_p1_stock import verify_model_files
    from pi0246_verify_checkpoint_archive import verify as verify_archive


def wanted(name: str) -> bool:
    return name in {"config.json", "embodiment_id.json", "processor_config.json", "statistics.json", "model.safetensors.index.json"} or name.endswith(".safetensors") or name.startswith("experiment_cfg/")


def make_view(checkpoint: Path, manifest_path: Path, step: int, model_root: Path) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest["checkpoints"][str(step)]
    archive = verify_archive(checkpoint, manifest_path, step)
    entries = [row for row in receipt["entries"] if wanted(row["path"])]
    names = {row["path"] for row in entries}
    assert {"config.json", "model.safetensors.index.json"} <= names
    assert sum(name.endswith(".safetensors") for name in names) >= 2
    target = model_root.resolve()
    assert not target.exists() and not checkpoint.is_relative_to(target) and not target.is_relative_to(checkpoint)
    partial = target.with_name(target.name + ".partial")
    assert not partial.exists()
    partial.mkdir(parents=True)
    files = []
    for row in entries:
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        source = (checkpoint / relative).resolve(strict=True)
        assert source.is_relative_to(checkpoint)
        link = partial / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.link(source, link)
        else:
            link.symlink_to(source)
        files.append({"path": relative.as_posix(), "bytes": row["bytes"], "sha256": row["sha256"]})
    restore = {
        "schema": "PI0246_VERIFIED_FULL_TRAINER_CHECKPOINT_INFERENCE_VIEW_V1",
        "checkpoint_step": step,
        "family": manifest["family"],
        "prefix": f"{manifest['family']}/checkpoint-{step}",
        "revision": f"local-tree-sha256:{archive['tree_sha256']}",
        "source_training_manifest_sha256": archive["training_manifest_sha256"],
        "source_run_card_sha256": manifest["run_card_sha256"],
        "checkpoint_tree_sha256": archive["tree_sha256"],
        "optimizer_restored": False,
        "files": files,
    }
    restore_path = partial / "pi0245_inference_restore.json"
    payload = json.dumps(restore, sort_keys=True, separators=(",", ":")) + "\n"
    with restore_path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    verify_model_files(partial)
    os.replace(partial, target)
    result = {**archive, "model_root": str(target), "restore_manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "view_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inference_files": len(files)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(1000, 2000), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_view(args.checkpoint, args.training_manifest, args.step, args.model_root), sort_keys=True))


if __name__ == "__main__":
    main()
