"""Resolve storage roles and reserve each active world's remaining writes."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from v5_storage_tree import require


GIB = 1 << 30
HOT_LIMIT_BYTES = 150 * GIB
STORAGE_ROLES = {
    "runtime": (Path("/var/tmp/sla-corelab-2009/hot"), "/", "ext4"),
    "artifacts": (Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/native_runs"),
                  "/mnt/local/hdd1", "fuseblk"),
    "public_sources": (Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/datasets"),
                       "/mnt/local/hdd1", "fuseblk"),
    "archive": (Path("/home/liuyu-nfs/Projects/GR00T-WBC/archive"), "/home/liuyu-nfs", "nfs4"),
    "checkpoints": (Path("/home/liuyu-nfs/Projects/GR00T-WBC/checkpoints"), "/home/liuyu-nfs", "nfs4"),
}


def floor(total):
    return max(50 * GIB, 15 * total // 100) + 8 * GIB


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def mount(path):
    raw = subprocess.check_output(["findmnt", "-n", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE"], text=True)
    return raw.splitlines()[-1].split()


def plan(run, ledger):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run) is not None, "Invalid new run ID")
    require(type(ledger["staging_runtime_bytes"]) is int and ledger["staging_runtime_bytes"] >= 0
            and ledger["staging_runtime_bytes"] <= HOT_LIMIT_BYTES, "Invalid bounded hot working set")
    evidence = {}
    for role, (root, expected_target, expected_type) in STORAGE_ROLES.items():
        parent = existing_parent(root)
        require(parent.resolve(strict=True) == parent, f"Symlink in storage parent for {role}")
        target, source, flavor = mount(parent)
        require((target, flavor) == (expected_target, expected_type), f"Wrong mounted volume for {role}")
        if role in ("runtime", "artifacts"):
            require(os.access(parent, os.W_OK) or role == "runtime" and parent == Path("/var/tmp"),
                    f"Expected a writable user-space parent for {role}")
        evidence[role] = dict(root=str(root), mounted_target=target, mounted_source=source,
                              filesystem_type=flavor, device=parent.stat().st_dev,
                              stage="EXISTING" if root.exists() else "PENDING_CREATE")
    reserve = {"runtime": ledger["staging_runtime_bytes"], "artifacts": 0}
    worlds = ledger["worlds"]
    require(isinstance(worlds, list) and len({row["world"] for row in worlds}) == len(worlds) > 0,
            "Each world has one measured reservation")
    hot_root = STORAGE_ROLES["runtime"][0]
    existing_hot_bytes = 0
    if hot_root.exists():
        totals = subprocess.check_output(["du", "-sx", "-B1", "--", str(hot_root)], text=True)
        existing_hot_bytes = int(totals.split()[0])
    for role in reserve:
        for row in worlds:
            remaining = row[role + "_remaining_bytes"]
            require(type(remaining) is int and remaining >= 0, f"Invalid measured budget for {role}")
            reserve[role] += remaining
        for field in ("next_snapshot_bytes", "partial_transfers_bytes"):
            amount = ledger[role + "_" + field]
            require(type(amount) is int and amount >= 0, f"Invalid in-flight budget for {role}")
            reserve[role] += amount
        disk = shutil.disk_usage(existing_parent(STORAGE_ROLES[role][0]))
        evidence[role].update(available_bytes=disk.free, original_floor_bytes=floor(disk.total),
                              required_by_inflight_bytes=reserve[role],
                              admit=disk.free >= floor(disk.total) + reserve[role])
    require(existing_hot_bytes + reserve["runtime"] <= HOT_LIMIT_BYTES,
            "Active-world staging exceeds the hot working-set cap")
    return dict(schema="corelab-storage-layout-v1", run=run, mounts=evidence, held_roles=[
        role for role in reserve if not evidence[role]["admit"]
    ], existing_hot_bytes=existing_hot_bytes,
        roots={role: str(STORAGE_ROLES[role][0] / run) for role in ("runtime", "artifacts")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("measured_budget", type=Path)
    args = parser.parse_args()
    print(json.dumps(plan(args.run, json.loads(args.measured_budget.read_bytes())), indent=2))
