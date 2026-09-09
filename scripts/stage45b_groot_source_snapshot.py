#!/usr/bin/env python3
"""Verify and materialize the immutable Stage45B GR00T source snapshot.

The runtime payload is intentionally a content-addressed, gitless tree.  This
module is kept dependency-free so that PBS can run it before importing any
scientific Python.  It never consults a Git checkout and it never downloads
source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


SCHEMA = "STAGE45B_GROOT_SOURCE_SNAPSHOT_V2"
ORIGIN_REPOSITORY = "NVIDIA/Isaac-GR00T"
ORIGIN_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
ORIGIN_TREE = "43617c11aa6c4ea1be47594882d797255c1a575b"
ALLOWED_SOURCE_ROUTES = {"HISTORICAL_LOCAL", "OFFICIAL_EXACT_REMATERIALIZATION"}


class SnapshotError(RuntimeError):
    """Raised when a snapshot identity or extraction invariant is violated."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file(path: Path, label: str, *, immutable: bool = False) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise SnapshotError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SnapshotError(f"{label} is not a regular file: {path}")
    if immutable and mode & 0o222:
        raise SnapshotError(f"{label} is writable; immutable snapshot required: {path}")


def _safe_relative(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise SnapshotError(f"unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotError(f"unsafe {label}: {value!r}")
    return path


def _safe_symlink_target(value: str, link_path: PurePosixPath) -> None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise SnapshotError(f"unsafe symlink target for {link_path}: {value!r}")
    target = PurePosixPath(value)
    if target.is_absolute():
        raise SnapshotError(f"absolute symlink target for {link_path}: {value!r}")
    resolved = PurePosixPath(link_path.parent, target)
    if any(part in {"", ".", ".."} for part in resolved.parts):
        raise SnapshotError(f"symlink escapes source root for {link_path}: {value!r}")


def _git_mode(kind: str, value: str) -> int:
    if kind == "regular" and value in {"100644", "100755"}:
        return int(value, 8) & 0o777
    if kind == "symlink" and value == "120000":
        return stat.S_IFLNK
    if kind == "gitlink" and value == "160000":
        return stat.S_IFDIR
    raise SnapshotError(f"unsupported manifest mode/kind pair: {value}/{kind}")


def _validate_entries(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SnapshotError("snapshot manifest entries are missing")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise SnapshotError("snapshot manifest entry is not an object")
        path = str(raw.get("path", ""))
        _safe_relative(path, "manifest path")
        if path in seen:
            raise SnapshotError(f"duplicate manifest path: {path}")
        seen.add(path)
        kind = str(raw.get("kind", ""))
        mode = str(raw.get("git_mode", ""))
        if kind not in {"regular", "symlink", "gitlink"}:
            raise SnapshotError(f"unsupported manifest entry kind: {kind!r}")
        _git_mode(kind, mode)
        try:
            size = int(raw.get("size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"invalid manifest size for {path}") from exc
        if size < 0 or (kind == "gitlink" and size != 0):
            raise SnapshotError(f"invalid manifest size for {path}: {size}")
        value = raw.get("sha256_or_symlink_target")
        if kind == "regular":
            if not isinstance(value, str) or len(value) != 64:
                raise SnapshotError(f"regular entry lacks SHA256: {path}")
            int(value, 16)
        elif kind == "symlink":
            _safe_symlink_target(str(value), PurePosixPath(path))
        else:
            if not isinstance(value, str) or len(value) != 40:
                raise SnapshotError(f"gitlink lacks its exact submodule commit: {path}")
            try:
                int(value, 16)
            except ValueError as exc:
                raise SnapshotError(f"gitlink commit is not hexadecimal: {path}") from exc
        normalized.append(
            {
                "path": path,
                "git_mode": mode,
                "kind": kind,
                "size_bytes": size,
                "sha256_or_symlink_target": value,
            }
        )
    normalized.sort(key=lambda item: str(item["path"]))
    return normalized


def load_manifest(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_payload_sha256: str | None = None,
    expected_payload_tree_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _regular_file(manifest_path, "snapshot manifest", immutable=True)
    observed_file_sha = sha256_file(manifest_path)
    if expected_manifest_file_sha256 and observed_file_sha != expected_manifest_file_sha256:
        raise SnapshotError(
            f"snapshot manifest file SHA mismatch: {observed_file_sha} != {expected_manifest_file_sha256}"
        )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"could not read snapshot manifest: {manifest_path}") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise SnapshotError("snapshot manifest schema mismatch")
    if document.get("origin_repository") != ORIGIN_REPOSITORY:
        raise SnapshotError("snapshot origin repository mismatch")
    if document.get("origin_commit") != ORIGIN_COMMIT:
        raise SnapshotError("snapshot origin commit mismatch")
    if document.get("origin_tree") != ORIGIN_TREE:
        raise SnapshotError("snapshot origin Git tree mismatch")
    if document.get("source_route") not in ALLOWED_SOURCE_ROUTES:
        raise SnapshotError("snapshot source route is not PI-authorized")
    if document.get("runtime_gitlink_required") is not False:
        raise SnapshotError("snapshot does not prove GR00T gitlinks are runtime-unrequired")
    if document.get("lfs_payloads_fetched") is not False:
        raise SnapshotError("snapshot unexpectedly fetched Git-LFS payloads")
    source_prefix = str(document.get("source_prefix", ""))
    if not source_prefix.endswith("/"):
        raise SnapshotError("snapshot source_prefix must end in '/'")
    _safe_relative(source_prefix.rstrip("/"), "source_prefix")
    entries = _validate_entries(document)
    if canonical_sha256(entries) != document.get("payload_tree_sha256"):
        raise SnapshotError("snapshot payload tree digest does not match entries")
    manifest_payload = dict(document)
    manifest_payload.pop("manifest_sha256", None)
    # The sealed V2 producer excluded its explanatory definition field from
    # the canonical payload as well; preserve that exact artifact definition.
    manifest_payload.pop("manifest_sha256_definition", None)
    if canonical_sha256(manifest_payload) != document.get("manifest_sha256"):
        raise SnapshotError("snapshot manifest payload digest mismatch")
    if expected_manifest_payload_sha256 and document.get("manifest_sha256") != expected_manifest_payload_sha256:
        raise SnapshotError("snapshot manifest payload SHA does not match admission")
    if expected_payload_tree_sha256 and document.get("payload_tree_sha256") != expected_payload_tree_sha256:
        raise SnapshotError("snapshot payload tree SHA does not match admission")
    for key, expected in (
        ("tracked_entry_count", len(entries)),
        ("regular_file_count", sum(item["kind"] == "regular" for item in entries)),
        ("symlink_count", sum(item["kind"] == "symlink" for item in entries)),
        ("gitlink_count", sum(item["kind"] == "gitlink" for item in entries)),
    ):
        if int(document.get(key, -1)) != expected:
            raise SnapshotError(f"snapshot {key} mismatch")
    return document, entries


def _member_path(name: str, source_prefix: str) -> str:
    if "\x00" in name or "\\" in name:
        raise SnapshotError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotError(f"unsafe archive member name: {name!r}")
    prefix = source_prefix.rstrip("/")
    if name == prefix:
        return ""
    marker = prefix + "/"
    if not name.startswith(marker):
        raise SnapshotError(f"archive member outside source prefix: {name!r}")
    relative = name[len(marker) :]
    _safe_relative(relative, "archive relative path")
    return relative


def _archive_members(
    archive_path: Path, document: Mapping[str, Any], entries: list[dict[str, Any]]
) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo], str]:
    _regular_file(archive_path, "snapshot archive", immutable=True)
    expected_archive_sha = str(document.get("archive_sha256", ""))
    observed_archive_sha = sha256_file(archive_path)
    if observed_archive_sha != expected_archive_sha:
        raise SnapshotError(
            f"snapshot archive SHA mismatch: {observed_archive_sha} != {expected_archive_sha}"
        )
    source_prefix = str(document["source_prefix"])
    expected = {item["path"]: item for item in entries}
    try:
        archive = tarfile.open(archive_path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise SnapshotError(f"could not open snapshot archive: {archive_path}") from exc
    members: dict[str, tarfile.TarInfo] = {}
    try:
        for member in archive.getmembers():
            relative = _member_path(member.name.rstrip("/"), source_prefix)
            if member.name.rstrip("/") in {source_prefix.rstrip("/")}:
                key = ""
            else:
                key = relative
            if key in members:
                raise SnapshotError(f"duplicate archive member: {member.name}")
            if member.isdir():
                pass
            elif member.isreg() or member.issym():
                if key not in expected:
                    raise SnapshotError(f"unexpected archive payload member: {member.name}")
            else:
                raise SnapshotError(f"unsupported archive member type: {member.name}")
            members[key] = member
        expected_dirs = {""}
        for item in entries:
            path = PurePosixPath(str(item["path"]))
            expected_dirs.update(str(parent) for parent in path.parents if str(parent) != ".")
            if item["kind"] == "gitlink":
                expected_dirs.add(str(path))
        for key, member in members.items():
            if member.isdir() and key not in expected_dirs:
                raise SnapshotError(f"unexpected archive directory: {key}")
        for item in entries:
            key = str(item["path"])
            member = members.get(key)
            if member is None:
                raise SnapshotError(f"manifest entry missing from archive: {key}")
            kind = item["kind"]
            if kind == "regular" and not member.isreg():
                raise SnapshotError(f"regular manifest entry is not regular in archive: {key}")
            if kind == "symlink" and (not member.issym() or member.linkname != item["sha256_or_symlink_target"]):
                raise SnapshotError(f"symlink manifest entry mismatch in archive: {key}")
            if kind == "gitlink" and not member.isdir():
                raise SnapshotError(f"gitlink is not represented by an empty directory: {key}")
        for item in entries:
            if item["kind"] != "regular":
                continue
            member = members[str(item["path"])]
            handle = archive.extractfile(member)
            if handle is None:
                raise SnapshotError(f"archive regular member has no readable payload: {item['path']}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                size += len(block)
                digest.update(block)
            if size != item["size_bytes"] or digest.hexdigest() != item["sha256_or_symlink_target"]:
                raise SnapshotError(f"archive payload hash mismatch: {item['path']}")
    except Exception:
        archive.close()
        raise
    return archive, members, source_prefix


def _destination_path(destination: Path, relative: str) -> Path:
    path = destination / relative
    try:
        path.relative_to(destination)
    except ValueError as exc:
        raise SnapshotError(f"destination path escapes extraction root: {relative}") from exc
    return path


def _ensure_parent_directory(path: Path, destination: Path) -> None:
    parent = path.parent
    try:
        parent.relative_to(destination)
    except ValueError as exc:
        raise SnapshotError(f"extraction parent escapes root: {parent}") from exc
    if not parent.is_dir() or parent.is_symlink():
        raise SnapshotError(f"extraction parent is not a safe directory: {parent}")


def _verify_extracted_tree(
    destination: Path, document: Mapping[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    if destination.is_symlink() or not destination.is_dir():
        raise SnapshotError(f"extracted source root is not a directory: {destination}")
    expected = {str(item["path"]): item for item in entries}
    expected_dirs = {""}
    for item in entries:
        path = PurePosixPath(str(item["path"]))
        expected_dirs.update(str(parent) for parent in path.parents if str(parent) != ".")
        if item["kind"] == "gitlink":
            expected_dirs.add(str(path))
    seen_files: set[str] = set()
    seen_dirs: set[str] = set()
    for directory, dirnames, filenames in os.walk(destination, followlinks=False):
        directory_path = Path(directory)
        relative_dir = directory_path.relative_to(destination).as_posix() if directory_path != destination else ""
        seen_dirs.add(relative_dir)
        if ".git" in dirnames or ".git" in filenames:
            raise SnapshotError("git metadata is present in the runtime source tree")
        for dirname in list(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                dirnames.remove(dirname)
                rel = child.relative_to(destination).as_posix()
                if rel not in expected or expected[rel]["kind"] != "symlink":
                    raise SnapshotError(f"unexpected extracted symlink directory: {rel}")
                if os.readlink(child) != expected[rel]["sha256_or_symlink_target"]:
                    raise SnapshotError(f"extracted symlink target mismatch: {rel}")
                seen_files.add(rel)
                continue
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise SnapshotError(f"unsafe extracted directory: {child}")
            rel = child.relative_to(destination).as_posix()
            if rel not in expected_dirs:
                raise SnapshotError(f"unexpected extracted directory: {rel}")
        for filename in filenames:
            child = directory_path / filename
            rel = child.relative_to(destination).as_posix()
            if child.is_symlink():
                if rel not in expected or expected[rel]["kind"] != "symlink":
                    raise SnapshotError(f"unexpected extracted symlink: {rel}")
                if os.readlink(child) != expected[rel]["sha256_or_symlink_target"]:
                    raise SnapshotError(f"extracted symlink target mismatch: {rel}")
            elif not stat.S_ISREG(child.lstat().st_mode):
                raise SnapshotError(f"unsupported extracted file kind: {rel}")
            else:
                if rel not in expected or expected[rel]["kind"] != "regular":
                    raise SnapshotError(f"unexpected extracted regular file: {rel}")
                item = expected[rel]
                if child.stat().st_size != item["size_bytes"] or sha256_file(child) != item["sha256_or_symlink_target"]:
                    raise SnapshotError(f"extracted file hash mismatch: {rel}")
                observed_mode = stat.S_IMODE(child.lstat().st_mode)
                expected_mode = int(item["git_mode"], 8) & 0o777
                if observed_mode != expected_mode:
                    raise SnapshotError(f"extracted mode mismatch for {rel}: {oct(observed_mode)}")
            seen_files.add(rel)
    for item in entries:
        rel = str(item["path"])
        path = destination / rel
        if item["kind"] == "gitlink":
            if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
                raise SnapshotError(f"gitlink materialized unexpectedly: {rel}")
        elif rel not in seen_files:
            raise SnapshotError(f"manifest payload missing after extraction: {rel}")
    if seen_dirs != expected_dirs:
        raise SnapshotError("extracted directory set differs from manifest")
    return {
        "root": str(destination),
        "git_present": False,
        "manifest_entry_count": len(entries),
        "regular_file_count": sum(item["kind"] == "regular" for item in entries),
        "symlink_count": sum(item["kind"] == "symlink" for item in entries),
        "gitlink_count": sum(item["kind"] == "gitlink" for item in entries),
        "payload_tree_sha256": document["payload_tree_sha256"],
    }


def verify_snapshot(
    archive_path: Path,
    manifest_path: Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_payload_sha256: str | None = None,
    expected_payload_tree_sha256: str | None = None,
) -> dict[str, Any]:
    document, entries = load_manifest(
        manifest_path,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_payload_sha256=expected_manifest_payload_sha256,
        expected_payload_tree_sha256=expected_payload_tree_sha256,
    )
    if expected_archive_sha256 and document.get("archive_sha256") != expected_archive_sha256:
        raise SnapshotError("snapshot archive identity in manifest does not match admission")
    archive, _, _ = _archive_members(archive_path, document, entries)
    archive.close()
    return {
        "schema": SCHEMA,
        "origin_repository": document["origin_repository"],
        "origin_commit": document["origin_commit"],
        "origin_tree": document["origin_tree"],
        "source_route": document["source_route"],
        "archive_sha256": sha256_file(archive_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": document["manifest_sha256"],
        "payload_tree_sha256": document["payload_tree_sha256"],
        "tracked_entry_count": len(entries),
        "regular_file_count": sum(item["kind"] == "regular" for item in entries),
        "symlink_count": sum(item["kind"] == "symlink" for item in entries),
        "gitlink_count": sum(item["kind"] == "gitlink" for item in entries),
    }


def extract_snapshot(
    archive_path: Path,
    manifest_path: Path,
    destination: Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_payload_sha256: str | None = None,
    expected_payload_tree_sha256: str | None = None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"extraction destination must be fresh: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise SnapshotError(f"extraction parent is not a safe directory: {destination.parent}")
    document, entries = load_manifest(
        manifest_path,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_payload_sha256=expected_manifest_payload_sha256,
        expected_payload_tree_sha256=expected_payload_tree_sha256,
    )
    expected_archive = str(document.get("archive_sha256", ""))
    if expected_archive_sha256 and expected_archive != expected_archive_sha256:
        raise SnapshotError("snapshot archive identity in manifest does not match admission")
    archive, members, source_prefix = _archive_members(archive_path, document, entries)
    try:
        destination.mkdir(mode=0o755)
        directory_names = sorted(
            (key for key, member in members.items() if member.isdir() and key != ""),
            key=lambda value: (value.count("/"), value),
        )
        for relative in directory_names:
            path = _destination_path(destination, relative)
            if path.exists() or path.is_symlink():
                raise SnapshotError(f"duplicate extraction destination: {relative}")
            path.mkdir(mode=0o755)
        for item in entries:
            if item["kind"] != "regular":
                continue
            relative = str(item["path"])
            member = members[relative]
            path = _destination_path(destination, relative)
            _ensure_parent_directory(path, destination)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, int(item["git_mode"], 8) & 0o777)
            try:
                handle = archive.extractfile(member)
                if handle is None:
                    raise SnapshotError(f"archive payload cannot be read: {relative}")
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    shutil.copyfileobj(handle, output, length=8 * 1024 * 1024)
                os.chmod(path, int(item["git_mode"], 8) & 0o777)
            finally:
                if descriptor != -1:
                    os.close(descriptor)
        for item in entries:
            if item["kind"] != "symlink":
                continue
            relative = str(item["path"])
            path = _destination_path(destination, relative)
            _ensure_parent_directory(path, destination)
            os.symlink(str(item["sha256_or_symlink_target"]), path)
        result = _verify_extracted_tree(destination, document, entries)
    finally:
        archive.close()
    result.update(
        {
            "schema": SCHEMA,
            "origin_repository": document["origin_repository"],
            "origin_commit": document["origin_commit"],
            "origin_tree": document["origin_tree"],
            "source_route": document["source_route"],
            "archive_sha256": sha256_file(archive_path),
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": document["manifest_sha256"],
            "payload_tree_sha256": document["payload_tree_sha256"],
            "source_prefix": source_prefix,
        }
    )
    return result


def verify_extracted(
    destination: Path,
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_manifest_payload_sha256: str | None = None,
    expected_payload_tree_sha256: str | None = None,
) -> dict[str, Any]:
    document, entries = load_manifest(
        manifest_path,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_manifest_payload_sha256=expected_manifest_payload_sha256,
        expected_payload_tree_sha256=expected_payload_tree_sha256,
    )
    return _verify_extracted_tree(destination, document, entries)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "extract"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-manifest-file-sha256")
    parser.add_argument("--expected-manifest-payload-sha256")
    parser.add_argument("--expected-payload-tree-sha256")
    args = parser.parse_args()
    try:
        kwargs = {
            "expected_archive_sha256": args.expected_archive_sha256,
            "expected_manifest_file_sha256": args.expected_manifest_file_sha256,
            "expected_manifest_payload_sha256": args.expected_manifest_payload_sha256,
            "expected_payload_tree_sha256": args.expected_payload_tree_sha256,
        }
        if args.mode == "verify":
            result = verify_snapshot(args.archive.resolve(), args.manifest.resolve(), **kwargs)
        else:
            if args.destination is None:
                raise SnapshotError("--destination is required for extract mode")
            result = extract_snapshot(
                args.archive.resolve(), args.manifest.resolve(), args.destination.resolve(), **kwargs
            )
        result.update({"status": "PASS", "timestamp_utc": now_utc()})
        _write_json(args.json_output.resolve(), result)
        print("VANDA_STAGE45B_GROOT_SOURCE_SNAPSHOT_V2_VERIFIED")
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "timestamp_utc": now_utc(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(args.json_output.resolve(), failure)
        except Exception:
            pass
        print(f"STAGE45B_GROOT_SNAPSHOT_BLOCKED: {failure['error']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
