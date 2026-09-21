"""Node-specific, no-model PI0243 work/durable storage admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


CHECKPOINT_BYTES = 20_889_600_105
RESERVATION_BYTES = 128 * 1024**3


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-2000:]}")
    return result


def snapshot(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    stat = os.statvfs(path)
    return {
        "path": str(path.resolve()),
        "disk_usage": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "statvfs": {"block_size": stat.f_frsize, "available_bytes": stat.f_bavail * stat.f_frsize, "free_bytes": stat.f_bfree * stat.f_frsize, "free_inodes": stat.f_favail},
    }


def remote_snapshot(host: str, path: Path) -> dict[str, Any]:
    code = (
        "import json,os,shutil,sys; p=sys.argv[1]; os.makedirs(p,exist_ok=True); "
        "u=shutil.disk_usage(p); s=os.statvfs(p); "
        "print(json.dumps({'path':os.path.realpath(p),'disk_usage':{'total_bytes':u.total,'used_bytes':u.used,'free_bytes':u.free},'statvfs':{'block_size':s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize,'free_inodes':s.f_favail}}))"
    )
    return json.loads(run(["ssh", host, "python3", "-c", code, str(path)]).stdout)


def reserve_local(path: Path, size: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if not hasattr(os, "posix_fallocate"):
            raise RuntimeError("Python has no posix_fallocate")
        os.posix_fallocate(fd, 0, size)
        os.fsync(fd)
        stat = os.fstat(fd)
        allocated = int(getattr(stat, "st_blocks", 0)) * 512
        if allocated < size:
            raise RuntimeError(f"reservation is sparse or under-allocated: {allocated} < {size}")
        return {"status": "PASS", "bytes": size, "allocated_bytes": allocated, "elapsed_s": time.time() - started}
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def reserve_remote(host: str, root: Path, size: int, label: str) -> dict[str, Any]:
    path = root / f".pi0243_capacity_reservation_{label}"
    code = (
        "import os,sys,time; p=sys.argv[1]; n=int(sys.argv[2]); "
        "os.makedirs(os.path.dirname(p),exist_ok=True); t=time.time(); "
        "fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600); "
        "os.posix_fallocate(fd,0,n); os.fsync(fd); s=os.fstat(fd); "
        "a=getattr(s,'st_blocks',0)*512; print(f'{a}'); os.close(fd); os.unlink(p); "
        "raise SystemExit(0 if a>=n else 3)"
    )
    result = run(["ssh", host, "python3", "-c", code, str(path), str(size)], check=False)
    if result.returncode:
        raise RuntimeError(f"remote capacity reservation failed: {result.stderr[-2000:]}")
    allocated = int(result.stdout.strip().splitlines()[-1])
    return {"status": "PASS", "bytes": size, "allocated_bytes": allocated}


def durable_probe(host: str, root: Path, label: str) -> dict[str, Any]:
    payload = f"PI0243 durable probe {label}\n".encode("utf-8") * 1024 * 4096
    local = Path(tempfile_name := f"/tmp/pi0243_storage_probe_{label}.bin")
    local.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    remote = root / f"storage_probe_{label}.bin"
    run(["ssh", host, "mkdir", "-p", str(root)])
    run(["rsync", "-a", "--partial", str(local), f"{host}:{remote}"])
    observed = run(["ssh", host, "sha256sum", str(remote)]).stdout.split()[0]
    if observed != digest:
        raise RuntimeError(f"durable probe hash mismatch: {observed} != {digest}")
    receipt = {"schema": "PI0243_DURABLE_PROBE_V1", "status": "PASS", "remote_path": str(remote), "sha256": digest, "bytes": len(payload)}
    receipt_path = local.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    run(["rsync", "-a", str(receipt_path), f"{host}:{root}/storage_probe_{label}.json"])
    local.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--durable-host", required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    work = snapshot(args.work_root)
    durable = remote_snapshot(args.durable_host, args.durable_root)
    peak_without_headroom = 4 * CHECKPOINT_BYTES
    headroom = max(10 * 1024**3, math.ceil(0.20 * peak_without_headroom))
    result: dict[str, Any] = {
        "schema": "PI0243_STORAGE_ADMISSION_V1",
        "status": "RUNNING",
        "model_activity": {"weight_loads": 0, "forward": 0, "backward": 0, "optimizer": 0},
        "work_root": work,
        "durable_root": durable,
        "durable_host": args.durable_host,
        "quota_status": "UNKNOWN",
        "checkpoint_size_basis": {"source": "PI0241 sealed full Trainer checkpoint-1000", "bytes": CHECKPOINT_BYTES, "file_count": 18},
        "capacity_plan": {"retain_all": True, "peak_without_headroom_bytes": peak_without_headroom, "headroom_bytes": headroom, "planned_work_bound_bytes": peak_without_headroom + headroom, "planned_durable_bound_bytes": 3 * CHECKPOINT_BYTES + headroom},
    }
    try:
        result["work_reservation"] = reserve_local(args.work_root / f".pi0243_capacity_reservation_{args.label}", RESERVATION_BYTES)
        result["durable_reservation"] = reserve_remote(args.durable_host, args.durable_root, RESERVATION_BYTES, args.label)
        result["durable_probe"] = durable_probe(args.durable_host, args.durable_root, args.label)
        result["status"] = "CAPACITY_PASS_DURABLE_HANDOFF_PASS"
    except Exception as exc:
        result["status"] = "CAPACITY_NOT_PROVEN"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["completed_at_epoch_s"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "CAPACITY_PASS_DURABLE_HANDOFF_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
