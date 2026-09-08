#!/usr/bin/env python3
"""Package a complete offline Hugging Face Cosmos cache subtree.

The package deliberately preserves the Hub cache layout, including ``blobs/``
and snapshot symlinks.  It never reads or copies authentication material.  The
manifest hashes regular files and the content addressed by symlink entries,
and defines a deterministic tree hash over the complete cache subtree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ID = "nvidia/Cosmos-Reason2-2B"
CACHE_DIR_NAME = "models--nvidia--Cosmos-Reason2-2B"
_CREDENTIAL_PATH_MARKERS = (
    ".env",
    ".netrc",
    "credential",
    "password",
    "secret",
    "token",
)


def _sha256_file(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        cache[key] = digest.hexdigest()
    return cache[key]


def _relative_if_inside(path: Path, root: Path) -> Path:
    path_resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        return path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"symlink escapes cache root: {path} -> {path_resolved}") from exc


def _walk_cache(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    digest_cache: dict[str, str] = {}

    def visit(directory: Path, relative: Path) -> None:
        children = sorted(os.scandir(directory), key=lambda item: item.name)
        for child in children:
            child_path = Path(child.path)
            child_relative = relative / child.name
            mode = os.lstat(child_path).st_mode
            relative_text = child_relative.as_posix()
            if stat.S_ISDIR(mode):
                entries.append(
                    {
                        "path": relative_text,
                        "kind": "directory",
                        "size_bytes": 0,
                        "sha256": None,
                    }
                )
                visit(child_path, child_relative)
                continue

            if stat.S_ISLNK(mode):
                target = os.readlink(child_path)
                resolved_target = child_path.resolve(strict=True)
                resolved_relative = _relative_if_inside(resolved_target, root)
                if not resolved_target.is_file():
                    raise RuntimeError(f"cache symlink does not resolve to a regular file: {child_path}")
                entries.append(
                    {
                        "path": relative_text,
                        "kind": "symlink",
                        "target": target,
                        "resolved_path": resolved_relative.as_posix(),
                        "size_bytes": resolved_target.stat().st_size,
                        "sha256": _sha256_file(resolved_target, digest_cache),
                    }
                )
                continue

            if stat.S_ISREG(mode):
                entries.append(
                    {
                        "path": relative_text,
                        "kind": "regular_file",
                        "size_bytes": child_path.stat().st_size,
                        "sha256": _sha256_file(child_path, digest_cache),
                    }
                )
                continue

            raise RuntimeError(f"unsupported filesystem entry in cache: {child_path}")

    visit(root, Path("."))
    return entries, digest_cache


def _tree_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: value["path"]):
        line = "\t".join(
            (
                str(entry["kind"]),
                str(entry["path"]),
                str(entry.get("size_bytes", 0)),
                str(entry.get("sha256") or "-"),
                str(entry.get("target") or "-"),
                str(entry.get("resolved_path") or "-"),
            )
        )
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def _credential_path_matches(entries: list[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for entry in entries:
        components = {component.lower() for component in Path(entry["path"]).parts}
        if any(marker in components for marker in _CREDENTIAL_PATH_MARKERS):
            matches.append(entry["path"])
    return matches


def _archive(cache_root: Path, output_path: Path) -> None:
    command = [
        "tar",
        "--create",
        "--file",
        str(output_path),
        "--sort=name",
        "--mtime=UTC 1970-01-01",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--format=posix",
        "-C",
        str(cache_root.parent),
        cache_root.name,
    ]
    subprocess.run(command, check=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--ref-name", default="main")
    args = parser.parse_args()

    cache_root = args.cache_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if cache_root.name != CACHE_DIR_NAME:
        raise SystemExit(f"unexpected cache root name: {cache_root.name}")
    if args.repo_id != REPO_ID:
        raise SystemExit(f"unexpected repo id: {args.repo_id}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = cache_root / "refs" / args.ref_name
    if not ref_path.is_file():
        raise SystemExit(f"missing Hub ref: {ref_path}")
    resolved_revision = ref_path.read_text(encoding="utf-8").strip()
    if not resolved_revision or any(character not in "0123456789abcdef" for character in resolved_revision.lower()):
        raise SystemExit(f"invalid resolved revision in {ref_path}")
    snapshot_root = cache_root / "snapshots" / resolved_revision
    if not snapshot_root.is_dir():
        raise SystemExit(f"resolved snapshot is missing: {snapshot_root}")

    entries, digest_cache = _walk_cache(cache_root)
    credential_matches = _credential_path_matches(entries)
    if credential_matches:
        raise SystemExit(f"credential-like paths found; refusing to package: {credential_matches}")

    regular_entries = [entry for entry in entries if entry["kind"] == "regular_file"]
    symlink_entries = [entry for entry in entries if entry["kind"] == "symlink"]
    directory_entries = [entry for entry in entries if entry["kind"] == "directory"]
    total_bytes = sum(int(entry["size_bytes"]) for entry in regular_entries)

    archive_path = output_dir / "cosmos_reason2_2b_hf_cache.tar"
    _archive(cache_root, archive_path)
    archive_sha256 = _sha256_path(archive_path)

    manifest: dict[str, Any] = {
        "schema": "STAGE22_H200_COSMOS_CACHE_SUPPLEMENT_V1",
        "repo_id": args.repo_id,
        "cache_layout": f"hub/{CACHE_DIR_NAME}",
        "ref_name": args.ref_name,
        "resolved_snapshot_revision": resolved_revision,
        "snapshot_directory": f"snapshots/{resolved_revision}",
        "file_count": len(regular_entries) + len(symlink_entries),
        "regular_file_count": len(regular_entries),
        "symlink_count": len(symlink_entries),
        "directory_count": len(directory_entries),
        "total_bytes": total_bytes,
        "tree_sha256": _tree_sha256(entries),
        "contains_credentials": False,
        "credential_path_audit": {
            "performed": True,
            "matching_paths": [],
            "token_content_read": False,
        },
        "archive": {
            "filename": archive_path.name,
            "format": "POSIX tar; symlinks preserved; no dereference",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
        },
        "entries": sorted(entries, key=lambda value: value["path"]),
        "source_integrity": {
            "ref_sha256": _sha256_file(ref_path, digest_cache),
            "snapshot_present": True,
            "all_symlinks_resolve_inside_cache": True,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "STAGE22_H200_COSMOS_CACHE_SUPPLEMENT_V1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = _sha256_path(manifest_path)
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{manifest_sha256}  {manifest_path.name}\n{archive_sha256}  {archive_path.name}\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "repo_id": args.repo_id,
        "resolved_snapshot_revision": resolved_revision,
        "file_count": manifest["file_count"],
        "total_bytes": total_bytes,
        "tree_sha256": manifest["tree_sha256"],
        "contains_credentials": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
