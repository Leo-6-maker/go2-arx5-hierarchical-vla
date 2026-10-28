"""Small, content-bound checkpoint handoff used by PI0243 only.

This module does not load a model.  It seals a complete Trainer checkpoint,
copies it to the configured durable receiver, verifies the receiver, and
publishes a receipt only after the receiver has a complete immutable tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "model.safetensors.index.json",
)


class PersistenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(entries: list[Mapping[str, Any]], label: str) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{label}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def inventory(root: Path, *, label: str) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"checkpoint directory missing: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _require(entries, f"checkpoint has no regular files: {root}")
    return {
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
        "tree_sha256": _tree_sha256(entries, label),
    }


def checkpoint_record(
    root: Path,
    *,
    expected_step: int,
    expected_max_steps: int,
    family: str = "PI0243_STAGE_A_V1",
) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"checkpoint-{expected_step} is missing: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    _require(not missing, f"checkpoint-{expected_step} missing Trainer state: {missing}")
    state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    _require(int(state.get("global_step", -1)) == expected_step, "trainer global_step mismatch")
    _require(int(state.get("max_steps", -1)) == expected_max_steps, "trainer max_steps mismatch")
    label = f"pi0243/{family}/checkpoint-{expected_step}"
    return {
        "schema": "PI0243_DURABLE_CHECKPOINT_RECORD_V1",
        "family": family,
        "checkpoint_step": int(expected_step),
        "max_steps": int(expected_max_steps),
        "global_step": int(state["global_step"]),
        "inventory": inventory(root, label=label),
    }


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("family", "checkpoint_step", "max_steps", "global_step")
    ) and left.get("inventory", {}).get("tree_sha256") == right.get("inventory", {}).get("tree_sha256") and left.get("inventory", {}).get("total_bytes") == right.get("inventory", {}).get("total_bytes") and left.get("inventory", {}).get("file_count") == right.get("inventory", {}).get("file_count")


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _copy_tree(source: Path, destination: Path) -> None:
    _require(not destination.exists(), f"refusing to overwrite handoff directory: {destination}")
    destination.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        target = destination / relative
        if source_path.is_dir():
            target.mkdir()
        elif source_path.is_file() and not source_path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            with target.open("r+b") as handle:
                os.fsync(handle.fileno())
        else:
            raise PersistenceError(f"unsupported checkpoint entry: {source_path}")


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise PersistenceError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-4000:]}")
    return result


def _remote_inventory(host: str, root: str, *, label: str) -> dict[str, Any]:
    quoted = shlex.quote(root)
    listing = _run(["ssh", host, f"find {quoted} -type f -printf '%P\\t%s\\n' | sort"]).stdout
    sizes: dict[str, int] = {}
    for line in listing.splitlines():
        relative, size = line.split("\t", 1)
        sizes[relative] = int(size)
    digest_output = _run(["ssh", host, f"cd -- {quoted} && find . -type f -print0 | xargs -0 sha256sum"]).stdout
    hashes: dict[str, str] = {}
    for line in digest_output.splitlines():
        digest, relative = line.split(None, 1)
        hashes[relative.removeprefix("./")] = digest
    entries = [
        {"path": path, "bytes": sizes[path], "sha256": hashes[path]}
        for path in sorted(sizes)
    ]
    _require(set(sizes) == set(hashes), "durable receiver inventory/hash file sets differ")
    return {
        "root": root,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
        "tree_sha256": _tree_sha256(entries, label),
    }


def _remote_exists(host: str, path: str) -> bool:
    return _run(["ssh", host, "test", "-e", path], check=False).returncode == 0


def publish_checkpoint(
    source: Path,
    *,
    durable_host: str | None,
    durable_root: Path,
    expected_step: int,
    expected_max_steps: int,
    family: str,
    receipt_root: Path,
    job_id: str,
) -> dict[str, Any]:
    """Publish one complete checkpoint and return its durable receipt."""

    record = checkpoint_record(source, expected_step=expected_step, expected_max_steps=expected_max_steps, family=family)
    safe_job = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id or "local")
    final = durable_root / "training" / f"checkpoint-{expected_step}"
    partial = durable_root / "training" / f"checkpoint-{expected_step}.partial-{safe_job}-{os.getpid()}"

    if durable_host:
        _run(["ssh", durable_host, "mkdir", "-p", str(final.parent)])
        if _remote_exists(durable_host, str(final)):
            existing = _remote_inventory(durable_host, str(final), label=f"pi0243/{family}/checkpoint-{expected_step}")
            existing_record = {**record, "inventory": existing}
            _require(_same_record(record, existing_record), f"durable checkpoint already exists with different content: {final}")
            durable_inventory = existing
        else:
            _run(["rsync", "-a", "--partial", f"{source}/", f"{durable_host}:{partial}/"])
            durable_inventory = _remote_inventory(durable_host, str(partial), label=f"pi0243/{family}/checkpoint-{expected_step}")
            _require(_same_record(record, {**record, "inventory": durable_inventory}), "durable receiver content differs from source")
            _run(["ssh", durable_host, "mv", "--", str(partial), str(final)])
    else:
        if final.exists():
            existing = checkpoint_record(final, expected_step=expected_step, expected_max_steps=expected_max_steps, family=family)
            _require(_same_record(record, existing), f"durable checkpoint already exists with different content: {final}")
            durable_inventory = existing["inventory"]
        else:
            _copy_tree(source, partial)
            durable_inventory = checkpoint_record(partial, expected_step=expected_step, expected_max_steps=expected_max_steps, family=family)["inventory"]
            _require(_same_record(record, {**record, "inventory": durable_inventory}), "local receiver content differs from source")
            partial.replace(final)

    receipt = {
        "schema": "PI0243_DURABLE_CHECKPOINT_RECEIPT_V1",
        "status": "DURABLE_VERIFIED",
        "family": family,
        "checkpoint_step": int(expected_step),
        "max_steps": int(expected_max_steps),
        "job_id": job_id,
        "source": str(source.resolve()),
        "durable_host": durable_host or "local",
        "durable_path": str(final),
        "source_record": record,
        "durable_inventory": durable_inventory,
        "verified_at_epoch_s": time.time(),
    }
    receipt_path = receipt_root / f"checkpoint-{expected_step}.durable_receipt.json"
    write_json_durable(receipt_path, receipt)
    if durable_host:
        _run(["rsync", "-a", str(receipt_path), f"{durable_host}:{durable_root}/checkpoint-{expected_step}.durable_receipt.json"])
    return receipt


def verify_receipt(local_checkpoint: Path, receipt_path: Path, *, expected_step: int, expected_max_steps: int, family: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("status") == "DURABLE_VERIFIED", "checkpoint receipt is not ready")
    current = checkpoint_record(local_checkpoint, expected_step=expected_step, expected_max_steps=expected_max_steps, family=family)
    _require(_same_record(current, receipt["source_record"]), "local checkpoint differs from durable source receipt")
    return receipt


def _fixture() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pi0243-storage-") as temporary:
        root = Path(temporary)
        source = root / "source" / "checkpoint-1000"
        durable = root / "durable"
        source.mkdir(parents=True)
        (source / "config.json").write_text("{}\n", encoding="utf-8")
        (source / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (source / name).write_bytes(name.encode("utf-8"))
        (source / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "max_steps": 5000}) + "\n", encoding="utf-8")
        receipt = publish_checkpoint(source, durable_host=None, durable_root=durable, expected_step=1000, expected_max_steps=5000, family="FIXTURE", receipt_root=root / "receipts", job_id="fixture")
        verify_receipt(durable / "checkpoint-1000", root / "receipts" / "checkpoint-1000.durable_receipt.json", expected_step=1000, expected_max_steps=5000, family="FIXTURE")
        (durable / "checkpoint-1000" / "optimizer.pt").write_bytes(b"corrupt")
        try:
            verify_receipt(durable / "checkpoint-1000", root / "receipts" / "checkpoint-1000.durable_receipt.json", expected_step=1000, expected_max_steps=5000, family="FIXTURE")
        except PersistenceError:
            corruption_rejected = True
        else:
            corruption_rejected = False
        if not corruption_rejected:
            raise PersistenceError("fixture failed to reject corrupted durable checkpoint")
        return {"schema": "PI0243_STORAGE_FIXTURE_V1", "status": "PASS", "corruption_rejected": True, "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if not args.fixture:
        parser.error("only --fixture is available for the local no-model test")
    print(json.dumps(_fixture(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
