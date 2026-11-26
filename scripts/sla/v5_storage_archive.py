"""Run the approved Corelab archive batches without rewriting historical artifacts."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from v5_storage_tree import compare, file_sha, require, scan


UID = 2009
APPROVED_SHA = "ffbb6b6db2699bfcff615bb225fb909087a76d9b1f9930fb65671a3b235eac49"
ACTIVE_ROOT = Path("/var/tmp/sla-v6-core-uid2009-owned-20261118")
ARCHIVE_ROOT = Path("/home/liuyu-nfs/Projects/GR00T-WBC/archive/corelab-storage-20261127-01")
APPROVED_WAVES = {"A0_PILOT_COPY", "A1_CLOSED_HISTORY", "B1_REBIND_BEFORE_RETIREMENT",
                  "B2_STATIC_GEOMETRY_ALLOCATION_REVIEW"}


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def select(approval, name):
    require(file_sha(approval) == APPROVED_SHA, "Approval sheet changed")
    with approval.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["name"] == name and row["wave"] in APPROVED_WAVES]
    require(len(rows) == 1, "Requested source is outside the approved retirement scope")
    row, = rows
    require(Path(row["source"]) == ACTIVE_ROOT / name and
            Path(row["target"]) == ARCHIVE_ROOT / "ssd/historic" / name,
            "Approved source and destination are outside the project roots")
    return row


def check_quota(receipt, needed):
    account = read(receipt)
    require(account["uid"] == UID and account["site"] == "corelab-amd-server"
            and account["volume_source"] == "corelab-amd-server-eth:/mnt/LocalHDD1/HomeNFS/liuyu-nfs"
            and 0 <= time.time() - account["observed_unix"] <= 3600,
            "A current server-side allocation readback for this account is required")
    available = account["personal_remaining_bytes"]
    require(account["confirmed_by"] and ((type(available) is int and available >= needed) or
            (available is None and account["no_personal_quota_confirmed"] is True)),
            "Effective personal quota is insufficient or unconfirmed")
    parent = ARCHIVE_ROOT.parent.resolve(strict=True)
    observed = subprocess.check_output(
        ["findmnt", "-n", "-T", str(parent), "-o", "SOURCE,FSTYPE"], text=True).splitlines()[-1].split()
    require(observed == [account["volume_source"], "nfs4"] and shutil.disk_usage(parent).free >= needed
            and parent.stat().st_uid == UID and os.access(parent, os.W_OK),
            "Destination mount, ownership, or available space changed")
    return int(available) if available is not None else None


def check_close(source, closure, allow_static=False):
    receipt = read(closure)
    require(receipt["source"] == str(source) and receipt["closed"] is True
            and 0 <= time.time() - receipt["observed_unix"] <= 3600
            and bool(receipt["evidence"]), "Recent closure reconciliation is required")
    for item in receipt["evidence"]:
        path = source / item["member"]
        require(path.resolve(strict=True).is_relative_to(source) and file_sha(path) == item["sha256"],
                f"Closure record changed at {path}")
    lease_path = source / "control/LEASE.json"
    if lease_path.is_file():
        require(read(lease_path)["status"] == "CLOSED", "Producer lease is active")
    else:
        require(allow_static, "Missing producer lease")
    for pattern in ("cases/*/control/LEASE.json", "cases/*/native-*/control/LEASE.json",
                    "*/control/LEASE.json", "*/*/control/LEASE.json"):
        for lease in source.glob(pattern):
            require(read(lease)["status"] == "CLOSED", f"Nested lease is active at {lease}")
    proc = subprocess.run(["lsof", "-t", "+D", str(source)], capture_output=True, timeout=45, check=False)
    require(proc.returncode == 1 and not proc.stdout and not proc.stderr, "Source has open files or audit failed")
    needle = os.fsencode(source)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != UID:
                continue
            for field in ("cmdline", "environ"):
                try:
                    value = (proc / field).read_bytes()
                except PermissionError:
                    require((proc / "comm").read_text().strip() in {"sshd", "(sd-pam)"},
                            f"Cannot audit active consumer {proc.name}")
                    continue
                require(needle + b"/" not in value and needle + b"\0" not in value,
                        f"Process {proc.name} still references archive source")
        except (FileNotFoundError, ProcessLookupError):
            continue


def archive_copy(source, staging, expected):
    require(not staging.is_symlink() and staging.resolve() == staging,
            "Staging path must be a real directory")
    staging.mkdir(mode=0o700, exist_ok=True)
    subprocess.run(["rsync", "-aHAXS", "--numeric-ids", "--protect-args", "--",
                    str(source) + "/", str(staging) + "/"], check=True)
    mounted_copy = scan(staging, owner=os.getuid(), progress=True, accept_trivial_nfs_acl=True)
    compare(expected, mounted_copy)
    return mounted_copy["member_sha256"]


def confirm_retirement_subset(isolated, expected):
    remaining = scan(isolated, owner=UID, progress=True)
    original = {row["path"]: row for row in expected["members"]}
    for row in remaining["members"]:
        reference = original.get(row["path"])
        require(reference is not None, f"New member in pending retirement: {row['path']}")
        keys = set(row) | set(reference)
        require(all(row.get(key) == reference.get(key) for key in keys - {"link_member"}),
                f"Pending retirement member differs: {row['path']}")


def check_relocated_reader(path, source, destination, expected, *, rebound):
    proof = read(path)
    require(proof["schema"] == "corelab-relocated-reader-v1" and proof["source"] == str(source)
            and proof["destination"] == str(destination) and proof["member_sha256"] == expected["member_sha256"]
            and proof["original_reader_passed"] is True and proof["historical_bytes_unchanged"] is True
            and proof["production_bindings_read_back"] is True
            and 0 <= time.time() - proof["observed_unix"] <= 3600,
            "A current readback from the installed archive is required")
    if rebound:
        require(proof["versioned_rebinding_passed"] is True and bool(proof["binding_records"]),
                "Dependent V5 specifications still require readback after rebinding")
    return file_sha(path)


def run(args):
    require(os.getuid() == UID, "Use the normal Corelab project account")
    entry = select(args.approval, args.name)
    source, destination = map(Path, (entry["source"], entry["target"]))
    expected = read(args.sources)
    require(expected["root"] == str(source) and expected["regular_members"] == int(entry["regular_files"]),
            "Source manifest does not match the approved inventory")
    if args.operation == "retire":
        require(args.readback is not None, "Retirement requires a real archive reader receipt")
    if args.operation == "copy":
        check_quota(args.quota, expected["unique_file_bytes"] + args.reserved_bytes)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (ARCHIVE_ROOT / ".archive-operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allow_static = entry["wave"] == "B2_STATIC_GEOMETRY_ALLOCATION_REVIEW"
        if source.exists():
            check_close(source, args.closure, allow_static)
            compare(expected, scan(source, owner=UID, progress=True))
        if args.operation == "copy":
            require(source.exists(), "Approved source is unavailable")
            saved = destination.with_suffix(".copy.json")
            digest = expected["member_sha256"]
            if destination.exists():
                compare(expected, scan(destination, owner=UID, progress=True, accept_trivial_nfs_acl=True))
            else:
                require(not destination.is_symlink(), "Archive path is a dangling symlink")
                staging = destination.with_name("." + destination.name + ".partial." + digest[:16])
                archive_copy(source, staging, expected)
                check_close(source, args.closure, allow_static)
                staging.rename(destination)
            if saved.exists():
                require(read(saved)["member_sha256"] == digest, "Existing archive receipt disagrees")
            else:
                write(saved, dict(source=str(source), destination=str(destination), member_sha256=digest,
                                  status="VERIFIED_BYTES_READER_PENDING", observed_unix=time.time()))
        else:
            saved = read(destination.with_suffix(".copy.json"))
            require(saved["member_sha256"] == expected["member_sha256"], "Copy receipt differs")
            compare(expected, scan(destination, owner=UID, progress=True, accept_trivial_nfs_acl=True))
            readback = check_relocated_reader(args.readback, source, destination, expected,
                                               rebound=entry["wave"] == "B1_REBIND_BEFORE_RETIREMENT")
            require(shutil.rmtree.avoids_symlink_attacks, "Use a symlink-safe removal implementation")
            receipt = destination.with_suffix(".retirement.json")
            intent = receipt.with_suffix(".intent.json")
            isolated = source.with_name("." + source.name + ".retirement." + expected["member_sha256"][:16])
            binding = dict(source=str(source), isolated=str(isolated), archive=str(destination),
                           member_sha256=expected["member_sha256"])
            require(not receipt.exists() and not source.is_symlink() and not isolated.is_symlink(),
                    "Retirement already closed or a path identity changed")
            if intent.exists():
                prior = read(intent)
                require(all(prior[key] == val for key, val in binding.items()), "Retirement intent does not match")
                before = prior["free_before_bytes"]
            else:
                require(source.exists() and not isolated.exists(), "Neither original nor valid retirement intent")
                before = shutil.disk_usage(source.parent).free
                write(intent, {**binding, "reader_sha256": readback, "free_before_bytes": before})
            if source.exists():
                check_close(source, args.closure, allow_static)
                compare(expected, scan(source, owner=UID, progress=True))
                require(not isolated.exists(), "Both original and pending retirement roots exist")
                source.rename(isolated)
            if isolated.exists():
                confirm_retirement_subset(isolated, expected)
                shutil.rmtree(isolated)
            write(receipt, dict(source=str(source), archive=str(destination),
                  member_sha256=expected["member_sha256"], reader_sha256=readback,
                  free_before_bytes=before, free_after_bytes=shutil.disk_usage(source.parent).free,
                  completed=True, observed_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("copy", "retire"))
    parser.add_argument("name")
    for key in ("approval", "sources", "closure"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--quota", type=Path)
    parser.add_argument("--reserved-bytes", type=int, default=0)
    parser.add_argument("--readback", type=Path)
    args = parser.parse_args()
    require(args.reserved_bytes >= 0, "Invalid in-flight reserve")
    if args.operation == "copy":
        require(args.quota is not None, "Copy requires a current quota receipt")
    run(args)
