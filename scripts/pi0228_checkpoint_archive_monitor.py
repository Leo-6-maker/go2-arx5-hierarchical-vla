#!/usr/bin/env python3
"""Archive PI0228 full Trainer checkpoints before releasing remote transient copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


STEPS = (2500, 5000, 7500, 10000)
REQUIRED = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
SCHEMA = "PI0228_CHECKPOINT_ARCHIVE_MANIFEST_V1"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def remote_json(host: str, code: str, *arguments: str) -> dict[str, Any]:
    command = f"python3 -c {shlex.quote(code)} " + " ".join(shlex.quote(value) for value in arguments)
    result = run(["ssh", host, command])
    return json.loads(result.stdout)


REMOTE_PROBE = r'''
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
result = {"ready": False, "path": str(p)}
try:
    state_path = p / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    files = [x for x in p.rglob("*") if x.is_file()]
    symlinks = [x for x in p.rglob("*") if x.is_symlink()]
    total = sum(x.stat().st_size for x in files)
    newest = max((x.stat().st_mtime_ns for x in files), default=0)
    result.update({
        "ready": p.is_dir() and not symlinks and all((p / name).is_file() for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")),
        "global_step": int(state.get("global_step", -1)),
        "max_steps": int(state.get("max_steps", -1)),
        "file_count": len(files),
        "total_bytes": total,
        "newest_mtime_ns": newest,
        "symlink_count": len(symlinks),
    })
    result["ready"] = bool(result["ready"] and result["global_step"] == expected and result["max_steps"] == 10000)
except (OSError, ValueError, json.JSONDecodeError):
    pass
print(json.dumps(result, sort_keys=True))
'''


REMOTE_MANIFEST = r'''
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
step = int(sys.argv[2])
entries = []
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("symlink in checkpoint")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
entries.sort(key=lambda item: item["path"])
tree = hashlib.sha256()
for item in entries:
    tree.update(f"pi0228/checkpoint-{step}/{item['path']}\\0{item['bytes']}\\0{item['sha256']}\\n".encode())
print(json.dumps({"file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries), "tree_sha256": tree.hexdigest(), "entries": entries}, sort_keys=True))
'''


def local_manifest(root: Path, step: int) -> dict[str, Any]:
    entries = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink in archived checkpoint: {path}")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    entries.sort(key=lambda item: item["path"])
    tree = hashlib.sha256()
    for item in entries:
        tree.update(f"pi0228/checkpoint-{step}/{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode())
    return {"file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries), "tree_sha256": tree.hexdigest(), "entries": entries}


def copy_checkpoint(host: str, remote_root: str, step: int, stage: Path) -> Path:
    stage.mkdir(parents=True)
    command = f"tar -C {shlex.quote(remote_root + '/training')} -cf - {shlex.quote(f'checkpoint-{step}')}"
    source = subprocess.Popen(["ssh", host, command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert source.stdout is not None
    target = subprocess.Popen(["tar", "-C", str(stage), "-xf", "-"], stdin=source.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    source.stdout.close()
    _, target_err = target.communicate()
    source_err = source.stderr.read() if source.stderr is not None else b""
    source_rc = source.wait()
    if target.returncode != 0 or source_rc != 0:
        raise RuntimeError(f"checkpoint copy failed: ssh={source_rc}, tar={target.returncode}, stderr={(source_err + target_err)[:500].decode(errors='replace')}")
    copied = stage / f"checkpoint-{step}"
    if not copied.is_dir():
        raise RuntimeError(f"checkpoint copy produced no directory: {copied}")
    return copied


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def upload_manifest(host: str, local_path: Path, remote_path: str) -> None:
    temporary = remote_path + ".partial"
    run(["scp", str(local_path), f"{host}:{temporary}"])
    run(["ssh", host, f"mv -f {shlex.quote(temporary)} {shlex.quote(remote_path)}"])


def release_remote(host: str, checkpoint_path: str) -> None:
    run(["ssh", host, f"rm -rf -- {shlex.quote(checkpoint_path)}"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="nus-hopper")
    parser.add_argument("--remote-run-root", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--remote-manifest")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-wait-seconds", type=float, default=86400.0)
    parser.add_argument("--release-remote", action="store_true")
    args = parser.parse_args()
    remote_root = args.remote_run_root.rstrip("/")
    if not remote_root.startswith("/scratch/e1192548/"):
        raise SystemExit("refusing remote root outside the PI0228 scratch namespace")
    remote_manifest = args.remote_manifest or f"{remote_root}/pi0228_checkpoint_archive_manifest.json"
    if not remote_manifest.startswith(remote_root + "/") or Path(remote_manifest).name != "pi0228_checkpoint_archive_manifest.json":
        raise SystemExit("refusing remote manifest outside the PI0228 run root")
    archive_root = args.archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    if any(archive_root.iterdir()):
        raise SystemExit(f"archive root must be empty: {archive_root}")
    records: dict[str, dict[str, Any]] = {}
    manifest_path = archive_root / "pi0228_checkpoint_archive_manifest.json"
    started = time.time()
    for step in STEPS:
        previous_signature = None
        stable = False
        deadline = time.monotonic() + args.max_wait_seconds
        remote_checkpoint = f"{remote_root}/training/checkpoint-{step}"
        while time.monotonic() < deadline:
            probe = remote_json(args.remote_host, REMOTE_PROBE, remote_checkpoint, str(step))
            signature = (probe.get("file_count"), probe.get("total_bytes"), probe.get("newest_mtime_ns"))
            if probe.get("ready") and signature == previous_signature:
                stable = True
                break
            previous_signature = signature
            time.sleep(args.poll_seconds)
        if not stable:
            raise SystemExit(f"checkpoint-{step} did not reach a stable complete state")

        stage = archive_root / f".checkpoint-{step}.stage"
        partial = archive_root / f"checkpoint-{step}.partial"
        final = archive_root / f"checkpoint-{step}"
        if stage.exists() or partial.exists() or final.exists():
            raise SystemExit(f"archive destination already exists for checkpoint-{step}")
        copied = copy_checkpoint(args.remote_host, remote_root, step, stage)
        copied.rename(partial)
        stage.rmdir()
        remote_record = remote_json(args.remote_host, REMOTE_MANIFEST, remote_checkpoint, str(step))
        local_record = local_manifest(partial, step)
        if remote_record != local_record:
            raise SystemExit(f"archive verification mismatch for checkpoint-{step}")
        partial.rename(final)
        records[str(step)] = {
            "path": str(final),
            "remote_transient_path": remote_checkpoint,
            "global_step": step,
            "max_steps": 10000,
            "file_count": local_record["file_count"],
            "total_bytes": local_record["total_bytes"],
            "tree_sha256": local_record["tree_sha256"],
            "verified": True,
            "remote_transient_released": False,
        }
        payload = {"schema": SCHEMA, "status": "PARTIAL", "archive_root": str(archive_root), "started_unix": started, "checkpoints": records}
        write_manifest(manifest_path, payload)
        upload_manifest(args.remote_host, manifest_path, remote_manifest)
        if args.release_remote:
            release_remote(args.remote_host, remote_checkpoint)
            records[str(step)]["remote_transient_released"] = True
            payload["checkpoints"] = records
            write_manifest(manifest_path, payload)
            upload_manifest(args.remote_host, manifest_path, remote_manifest)

    payload = {"schema": SCHEMA, "status": "COMPLETE", "archive_root": str(archive_root), "started_unix": started, "finished_unix": time.time(), "checkpoints": records, "release_remote_copy": bool(args.release_remote)}
    write_manifest(manifest_path, payload)
    upload_manifest(args.remote_host, manifest_path, remote_manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
