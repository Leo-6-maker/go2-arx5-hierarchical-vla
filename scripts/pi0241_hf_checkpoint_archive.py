#!/usr/bin/env python3
"""Login-node rolling archive for PI0241 full Trainer checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


STEPS = (250, 500, 1000)
MAX_STEPS = 1000
SCHEMA = "PI0241_CHECKPOINT_ARCHIVE_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, step: int) -> dict:
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    state_path = root / "trainer_state.json"
    if not root.is_dir() or not all((root / name).is_file() for name in required):
        raise RuntimeError(f"checkpoint-{step} is incomplete")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != step or int(state.get("max_steps", -1)) != MAX_STEPS:
        raise RuntimeError(f"checkpoint-{step} trainer state mismatch")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in checkpoint: {path}")
        if path.is_file():
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    tree = hashlib.sha256()
    for entry in entries:
        tree.update(f"pi0241/checkpoint-{step}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return {"global_step": step, "max_steps": MAX_STEPS, "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries), "tree_sha256": tree.hexdigest(), "entries": entries}


def stable_inventory(root: Path, step: int, timeout: float, poll: float) -> dict:
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        try:
            current = inventory(root, step)
        except (OSError, ValueError, RuntimeError):
            time.sleep(poll)
            continue
        signature = (current["file_count"], current["total_bytes"], current["tree_sha256"])
        if signature == previous:
            return current
        previous = signature
        time.sleep(poll)
    raise RuntimeError(f"checkpoint-{step} did not stabilize")


def remote_inventory(api, repo_id: str, prefix: str) -> list[dict]:
    result = []
    for entry in api.list_repo_tree(repo_id=repo_id, path_in_repo=prefix, recursive=True, repo_type="model"):
        path = getattr(entry, "path", None)
        if path is None:
            continue
        lfs = getattr(entry, "lfs", None)
        if isinstance(lfs, dict):
            remote_sha = lfs.get("sha256")
        else:
            remote_sha = getattr(lfs, "sha256", None)
        result.append({"path": path, "size": getattr(entry, "size", None), "sha256": remote_sha})
    return result


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def retain_screen_model(checkpoint: Path, run_root: Path, step: int) -> Path:
    """Keep only files required by AutoModel after the full state is released."""
    destination = run_root / "screen_models" / f"checkpoint-{step}"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite screen model: {destination}")
    destination.mkdir(parents=True)
    patterns = (
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "*.safetensors.index.json",
        "model*.bin",
        "*.bin.index.json",
    )
    copied = []
    for pattern in patterns:
        for source in sorted(checkpoint.glob(pattern)):
            if source.is_file():
                target = destination / source.name
                shutil.copy2(source, target)
                copied.append(source.name)
    if not any(name.startswith("model") for name in copied) or not (destination / "config.json").is_file():
        raise RuntimeError(f"checkpoint-{step} has no complete screen model payload")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--checkpoint-timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    records = {}
    started = time.time()
    for step in STEPS:
        checkpoint = args.run_root / "training" / f"checkpoint-{step}"
        local = stable_inventory(checkpoint, step, args.checkpoint_timeout_seconds, args.poll_seconds)
        prefix = f"checkpoints/checkpoint-{step}"
        api.upload_folder(folder_path=str(checkpoint), repo_id=args.repo_id, repo_type="model", path_in_repo=prefix, commit_message=f"PI0241 full Trainer checkpoint {step}")
        remote = remote_inventory(api, args.repo_id, prefix)
        expected = {f"{prefix}/{entry['path']}": entry for entry in local["entries"]}
        observed = {entry["path"]: entry for entry in remote}
        if set(expected) != set(observed):
            raise RuntimeError(f"HF inventory mismatch at step {step}")
        for path, entry in expected.items():
            remote_entry = observed[path]
            if remote_entry.get("size") is not None and int(remote_entry["size"]) != int(entry["bytes"]):
                raise RuntimeError(f"HF size mismatch: {path}")
            remote_sha = remote_entry.get("sha256")
            if remote_sha is not None and remote_sha != entry["sha256"]:
                raise RuntimeError(f"HF LFS SHA mismatch: {path}")
        manifest_file = args.run_root / f"pi0241_checkpoint_{step}_manifest.json"
        manifest_file.write_text(json.dumps({"schema": "PI0241_CHECKPOINT_MANIFEST_V1", "step": step, **local}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        api.upload_file(path_or_fileobj=str(manifest_file), path_in_repo=f"manifests/checkpoint-{step}.json", repo_id=args.repo_id, repo_type="model", commit_message=f"PI0241 manifest {step}")
        records[str(step)] = {"verified": True, "global_step": step, "max_steps": MAX_STEPS, "file_count": local["file_count"], "total_bytes": local["total_bytes"], "tree_sha256": local["tree_sha256"], "repo_id": args.repo_id, "remote_prefix": prefix, "remote_transient_released": False}
        write_manifest(args.manifest, {"schema": SCHEMA, "status": "PARTIAL", "repo_id": args.repo_id, "started_unix": started, "checkpoints": records})
        if step < 1000:
            screen_model = retain_screen_model(checkpoint, args.run_root, step)
            shutil.rmtree(checkpoint)
            records[str(step)]["screen_model_path"] = str(screen_model)
            records[str(step)]["remote_transient_released"] = True
            write_manifest(args.manifest, {"schema": SCHEMA, "status": "PARTIAL", "repo_id": args.repo_id, "started_unix": started, "checkpoints": records})
    write_manifest(args.manifest, {"schema": SCHEMA, "status": "COMPLETE", "repo_id": args.repo_id, "started_unix": started, "finished_unix": time.time(), "checkpoints": records})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
