#!/usr/bin/env python3
"""Safely materialize the accepted PI0163 FFmpeg runtime capsule.

This helper is intentionally standard-library-only. It validates the capsule
before extraction, rejects absolute/path-traversal members and escaping
symlinks, preserves the accepted FFmpeg ``lib``/``compat`` payload, and then
rehashes every extracted entry. It does not import Torch, TorchCodec, GR00T,
or a model; those checks belong to the separate import-only probe and the
frozen Validation-E runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA = "STAGE45F_PI0163_FFMPEG_CAPSULE_BOOTSTRAP_V1"
SOURCE_MANIFEST = "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json"
SOURCE_CHECKSUM_FILE = "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json.sha256"
EXPECTED_RUNTIME_ID = "9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"
EXPECTED_MANIFEST_SHA256 = "6160d579b98dc5ee4b114ada4a4e00e88ee13df6ed1fc3d308b5879d67027d2c"
EXPECTED_CHECKSUM_FILE_SHA256 = "7c1c453ee1d6b91fde5836c4ee65cc031a1d0bc7bb43615c78985f03ca6f6f11"
EXPECTED_ROOTS = {"lib", "compat"}


class BootstrapError(RuntimeError):
    """Raised when an accepted capsule cannot be materialized exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular manifest required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"manifest is not an object: {path}")
    return value


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"fresh record required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def normalized_member_name(name: str) -> str:
    require(name and "\\" not in name, f"unsafe archive member name: {name!r}")
    require(not name.startswith("/"), f"absolute archive member is forbidden: {name!r}")
    trimmed = name.rstrip("/")
    require(trimmed and trimmed != ".", f"empty archive member is forbidden: {name!r}")
    normalized = posixpath.normpath(trimmed)
    require(normalized == trimmed, f"non-canonical archive member: {name!r}")
    require(normalized != ".." and not normalized.startswith("../"), f"path traversal archive member: {name!r}")
    return normalized


def expected_payload(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(manifest.get("schema") == "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1", "source manifest schema drift")
    require(manifest.get("status") == "SEALED_IMMUTABLE", "source manifest is not sealed")
    require(manifest.get("artifact_id") == EXPECTED_RUNTIME_ID, "source artifact identity drift")
    require(manifest.get("payload_tree_sha256") == EXPECTED_RUNTIME_ID, "source payload tree identity drift")
    raw_entries = manifest.get("payload_entries")
    require(isinstance(raw_entries, list) and raw_entries, "source payload entries are missing")
    entries: dict[str, dict[str, Any]] = {}
    for raw in raw_entries:
        require(isinstance(raw, Mapping), "source payload entry is not an object")
        path = normalized_member_name(str(raw.get("path")))
        require(path not in entries, f"duplicate source payload entry: {path}")
        kind = str(raw.get("kind"))
        require(kind in {"regular_file", "symlink"}, f"unsupported source payload kind: {path}: {kind}")
        require(path.startswith("lib/") or path.startswith("compat/"), f"payload outside lib/compat: {path}")
        if kind == "regular_file":
            require(isinstance(raw.get("sha256"), str) and len(str(raw["sha256"])) == 64, f"file hash missing: {path}")
            require(int(raw.get("bytes", -1)) >= 0, f"file size missing: {path}")
        else:
            target = raw.get("target")
            require(isinstance(target, str) and target, f"symlink target missing: {path}")
            require(not target.startswith("/") and "\\" not in target, f"unsafe symlink target: {path}")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
            require(resolved.startswith("lib/") or resolved.startswith("compat/"), f"symlink escapes payload root: {path}")
        entries[path] = dict(raw)
    require({path.split("/", 1)[0] for path in entries} == EXPECTED_ROOTS, "lib/compat payload roots are incomplete")
    return entries


def safe_destination(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts:
        require(part not in {"", ".", ".."}, f"unsafe destination component: {relative}")
        current = current / part
        if current.exists() or current.is_symlink():
            require(not current.is_symlink(), f"destination symlink is forbidden: {current}")
            require(current.is_dir() or current.is_file(), f"unsupported destination entry: {current}")
    return current


def archive_members(capsule: Path, payload: Mapping[str, Mapping[str, Any]]) -> list[tarfile.TarInfo]:
    allowed_files = set(payload) | {SOURCE_MANIFEST, SOURCE_CHECKSUM_FILE}
    allowed_dirs = EXPECTED_ROOTS
    members: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    with tarfile.open(capsule, mode="r:") as archive:
        for member in archive.getmembers():
            name = normalized_member_name(member.name)
            require(name not in seen, f"duplicate archive member: {name}")
            seen.add(name)
            if member.isdir():
                require(name in allowed_dirs, f"unexpected archive directory: {name}")
            elif member.isreg():
                require(name in allowed_files, f"unexpected archive file: {name}")
                if name in payload:
                    expected = payload[name]
                    require(int(member.size) == int(expected["bytes"]), f"archive size drift: {name}")
            elif member.issym():
                require(name in payload and payload[name].get("kind") == "symlink", f"unexpected archive symlink: {name}")
                require(member.linkname == payload[name].get("target"), f"archive symlink drift: {name}")
            else:
                raise BootstrapError(f"unsupported archive member type: {name}")
            members.append(member)
    require(set(payload).issubset(seen), "archive omitted an accepted payload entry")
    require(set(allowed_dirs).issubset(seen), "archive omitted lib/compat root")
    require({SOURCE_MANIFEST, SOURCE_CHECKSUM_FILE}.issubset(seen), "archive omitted source manifest")
    require(seen == set(payload) | allowed_dirs | {SOURCE_MANIFEST, SOURCE_CHECKSUM_FILE}, "archive contains undeclared entries")
    return members


def extract_members(capsule: Path, destination: Path, members: Sequence[tarfile.TarInfo]) -> None:
    require(not destination.exists() and not destination.is_symlink(), f"fresh destination required: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(capsule, mode="r:") as archive:
        for member in members:
            name = normalized_member_name(member.name)
            target = safe_destination(destination, name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=False)
                os.chmod(target, member.mode & 0o7777)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(member.linkname, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                require(source is not None, f"archive member cannot be read: {name}")
                with source, target.open("xb") as output:
                    while True:
                        block = source.read(1 << 20)
                        if not block:
                            break
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(target, member.mode & 0o7777)


def verify_extracted(destination: Path, manifest: Mapping[str, Any], payload: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = destination / SOURCE_MANIFEST
    checksum_path = destination / SOURCE_CHECKSUM_FILE
    require(sha256_file(manifest_path) == EXPECTED_MANIFEST_SHA256, "extracted source manifest SHA mismatch")
    require(sha256_file(checksum_path) == EXPECTED_CHECKSUM_FILE_SHA256, "extracted source checksum-file SHA mismatch")
    observed_manifest = load_json(manifest_path)
    require(observed_manifest == manifest, "extracted source manifest content drift")
    records: list[dict[str, Any]] = []
    destination_resolved = destination.resolve()
    for relative, expected in sorted(payload.items()):
        path = destination / relative
        kind = str(expected["kind"])
        if kind == "regular_file":
            require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"extracted file missing: {relative}")
            observed_sha = sha256_file(path)
            require(observed_sha == expected["sha256"], f"extracted file SHA mismatch: {relative}")
            require(path.stat().st_size == int(expected["bytes"]), f"extracted file size mismatch: {relative}")
            records.append({"path": relative, "kind": kind, "bytes": int(path.stat().st_size), "sha256": observed_sha})
        else:
            require(path.is_symlink(), f"extracted symlink missing: {relative}")
            target = os.readlink(path)
            require(target == expected["target"], f"extracted symlink target mismatch: {relative}")
            resolved = path.resolve(strict=True)
            require(resolved.is_file(), f"extracted symlink target is not a file: {relative}")
            require(resolved.is_relative_to(destination_resolved), f"extracted symlink escapes destination: {relative}")
            records.append({"path": relative, "kind": kind, "target": target, "resolved_path": str(resolved.relative_to(destination_resolved).as_posix())})
    return records


def materialize(*, capsule: Path, capsule_sha256: str, destination: Path, record: Path, expected_runtime_id: str, expected_manifest_sha256: str) -> dict[str, Any]:
    require(expected_runtime_id == EXPECTED_RUNTIME_ID, "requested runtime identity is not allowlisted")
    require(expected_manifest_sha256 == EXPECTED_MANIFEST_SHA256, "requested manifest identity is not allowlisted")
    capsule = capsule.resolve(strict=True)
    require(capsule.is_file() and not capsule.is_symlink(), f"regular capsule required: {capsule}")
    observed_capsule_sha256 = sha256_file(capsule)
    require(observed_capsule_sha256 == capsule_sha256, f"capsule SHA mismatch: {observed_capsule_sha256}")
    with tarfile.open(capsule, mode="r:") as archive:
        manifest_member = archive.extractfile(archive.getmember(SOURCE_MANIFEST))
        require(manifest_member is not None, "source manifest cannot be read from capsule")
        manifest = json.loads(manifest_member.read().decode("utf-8"))
    require(isinstance(manifest, Mapping), "capsule source manifest is not an object")
    payload = expected_payload(manifest)
    members = archive_members(capsule, payload)
    extract_members(capsule, destination, members)
    extracted = verify_extracted(destination, manifest, payload)
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "timestamp_utc": now_utc(),
        "source_runtime_id": EXPECTED_RUNTIME_ID,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_checksum_file_sha256": EXPECTED_CHECKSUM_FILE_SHA256,
        "capsule_path": str(capsule),
        "capsule_sha256": observed_capsule_sha256,
        "capsule_bytes": int(capsule.stat().st_size),
        "destination": str(destination.resolve()),
        "archive_member_count": len(members),
        "payload_entry_count": len(payload),
        "extracted_entries": extracted,
        "network_download": False,
        "new_ffmpeg_build": False,
    }
    write_json_durable(record, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--capsule-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expected-runtime-id", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize(
            capsule=args.capsule,
            capsule_sha256=args.capsule_sha256,
            destination=args.destination,
            record=args.record,
            expected_runtime_id=args.expected_runtime_id,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "timestamp_utc": now_utc(),
            "capsule_path": str(args.capsule),
            "destination": str(args.destination),
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            if not args.record.exists() and not args.record.is_symlink():
                write_json_durable(args.record, failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 91


if __name__ == "__main__":
    raise SystemExit(main())
