#!/usr/bin/env python3
"""Verify and safely stage one immutable Stage39 execution bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict


EXPECTED_IDS = ["C0_dense_expert_tiny1", "C1_structured_oracle_k16t16"] + [
    "L%02d_seed%d" % (index, seed)
    for index, seed in enumerate(range(2223, 2239), start=1)
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(
        "FAIL_CLOSED_PRE_SCIENCE\n"
        "check=bundle_contents\n"
        "expected=verified_bundle\n"
        "observed=%s" % message
    )


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    bundle = Path(args.bundle)
    staging = Path(args.staging)
    if digest(bundle) != args.expected_bundle_sha256:
        fail("bundle_sha256_mismatch")
    if staging.exists() and any(staging.iterdir()):
        fail("staging_not_empty")
    staging.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(bundle, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            name = member.name
            pure = PurePosixPath(name)
            if not name or pure.is_absolute() or ".." in pure.parts or "\\" in name:
                fail("unsafe_member:" + name)
            if name in seen:
                fail("duplicate_member:" + name)
            seen.add(name)
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                fail("unsupported_member:" + name)
        for member in members:
            target = staging / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                fail("target_exists:" + member.name)
            source = archive.extractfile(member)
            if source is None:
                fail("unreadable_member:" + member.name)
            with target.open("xb") as output:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
    manifest_path = staging / "bundle_manifest.json"
    if not manifest_path.is_file():
        fail("bundle_manifest_missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "STAGE39_EXECUTION_BUNDLE_MANIFEST_V1":
        fail("bundle_manifest_schema")
    if manifest.get("scientific_protocol_id") != "PI_GATE_STATE_0028":
        fail("bundle_manifest_protocol")
    if manifest.get("git_head") != args.expected_head or manifest.get("row_count") != 18:
        fail("bundle_manifest_identity")
    if manifest.get("row_ids") != EXPECTED_IDS:
        fail("bundle_manifest_row_order")
    records = {item["relative_path"]: item for item in manifest.get("files", [])}
    required = {
        "stage39a_result.json",
        "stage39_input_manifest.json",
        "stage39_warn_stratification.json",
        "execution_source_manifest.json",
    }
    if not required.issubset(records):
        fail("bundle_manifest_required_files")
    actual = set()
    for path in staging.rglob("*"):
        if path.is_symlink():
            fail("extracted_symlink:" + str(path))
        if path.is_file():
            actual.add(path.relative_to(staging).as_posix())
    if actual != set(records) | {"bundle_manifest.json"}:
        fail("unexpected_or_missing_extracted_files")
    for relative, record in records.items():
        path = staging / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or digest(path) != record["sha256"]
        ):
            fail("file_record:" + relative)
    admission = json.loads((staging / "stage39a_result.json").read_text())
    input_manifest = json.loads((staging / "stage39_input_manifest.json").read_text())
    if admission.get("status") != "PASS" or admission.get("pi_gate") != "PI_GATE_STATE_0028":
        fail("admission_semantics")
    if digest(staging / "stage39a_result.json") != args.expected_admission_sha256:
        fail("admission_hash")
    if digest(staging / "stage39_input_manifest.json") != args.expected_manifest_sha256:
        fail("input_manifest_hash")
    if digest(staging / "stage39_warn_stratification.json") != args.expected_warn_sha256:
        fail("warn_hash")
    rows = input_manifest.get("rows", [])
    if input_manifest.get("row_count") != 18 or [row.get("row_id") for row in rows] != EXPECTED_IDS:
        fail("input_manifest_row_order")
    for row in rows:
        for field in ("trajectory_path", "metadata_path"):
            name = "inputs/" + Path(str(row[field])).name
            if name not in records:
                fail("row_file_missing:" + name)
    verification = {
        "schema": "STAGE39_COMPUTE_NODE_BUNDLE_VERIFICATION_V1",
        "status": "PASS",
        "bundle_path": str(bundle),
        "bundle_sha256": args.expected_bundle_sha256,
        "bundle_size_bytes": bundle.stat().st_size,
        "staging_root": str(staging),
        "file_count": len(records),
        "row_count": 18,
        "row_ids": EXPECTED_IDS,
        "expected_head": args.expected_head,
        "admission_sha256": args.expected_admission_sha256,
        "input_manifest_sha256": args.expected_manifest_sha256,
        "warn_sha256": args.expected_warn_sha256,
        "hostname": os.uname().nodename,
    }
    Path(args.verification).write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print("CHECK bundle_contents PASS")
    return verification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-warn-sha256", required=True)
    parser.add_argument("--verification", required=True, type=Path)
    return 0 if verify(parser.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
