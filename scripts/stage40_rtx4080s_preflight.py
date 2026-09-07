#!/usr/bin/env python3
"""Stage40 RTX 4080S read-only artifact and source preflight.

This module is deliberately model-free.  It does not import Torch, GR00T,
IsaacGym, or any training/runtime package.  It binds the inputs needed by the
RTX engineering mirror before a later runner is allowed to construct a model
or simulator.  The checkpoint tree digest uses the same canonical Stage38
file-record format as the accepted Stage40 artifact identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PI_GATE = "PI_GATE_STATE_0064"
ISSUE_COMMENT = 5574239666
ACCEPTED_D1_GIT_SHA = "83779cbf03c31fe3477efc8eaba06ddeb2b46b0a"
GROOT_GIT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"

STRUCTURED_CHECKPOINT_TREE_SHA256 = (
    "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
)
STRUCTURED_CHECKPOINT_TREE_PREFIX = "stage38d/checkpoint-1000"
STRUCTURED_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
TINY1_FILE_COUNT = 13
TINY1_TOTAL_BYTES = 152958
TINY1_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
TINY1_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
TINY1_META_MODALITY_SHA256 = "1d172665da201f5b040ca6664a67bd3a9fe591237b3c8298d3a44f7e63f97c4e"
STAGE22_MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
WBC_ARCHIVE_SHA256 = "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167"
WBC_ARCHIVE_SIZE_BYTES = 963451251
WBC_ARCHIVE_MEMBER_COUNT = 887
WBC_ARCHIVE_FILE_COUNT = 742
WBC_ARCHIVE_DIRECTORY_COUNT = 145
WBC_V2_TREE_SHA256 = "9d653c39c73d5755a29d46f4dff91b69120244f660f41a977db471d58ba8b193"
HISTORICAL_WBC_TREE_SHA256 = "63311df752c9aaa5289287849635309e558f79284c91b80ddf07ec5638992567"
WBC_RUN_REL = Path(
    "runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018"
)
WBC_COMPONENTS = {
    "arm_checkpoint": (
        WBC_RUN_REL / "checkpoints_arm/ac_weights_039999.pt",
        "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    ),
    "dog_checkpoint": (
        WBC_RUN_REL / "checkpoints_dog/ac_weights_039999.pt",
        "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    ),
    "native_config": (
        WBC_RUN_REL / "scripts/config/wbc.py",
        "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
    ),
    "native_urdf": (
        Path("resources/robots/go2_x5_v3/urdf/go2_x5.urdf"),
        "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    ),
    "native_reach_table": (
        Path("resources/reach_tables/go2_x5_2d.pt"),
        "e34ca5c595b30b737209f5dcdd7f3f1e0b95cdf6232fbc0f6f50e3ee7140f5fa",
    ),
    "trajectory_module": (
        Path("modules/trajectory.py"),
        "8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a",
    ),
    "wbc_env": (
        Path("go1_gym/envs/roboduet/wbc_env.py"),
        "e39e1bb1dfe4c7a5c2b7309f4f158b6dcf1c9aac91ccbf8a60988ed8a64e992e",
    ),
    "legged_robot": (
        Path("go1_gym/envs/roboduet/legged_robot.py"),
        "4dc9107af7c1f77dc3b1b686911addf1a1d1a5179dfb194d0b1f4d0bdb8d8a60",
    ),
    "wbc_wrapper": (
        Path("go1_gym/envs/roboduet/wbc_env_wrapper.py"),
        "9fa688ce7c3f4665ec48752f0f31639f05ab3309b53d00d95e3d0afd0cb300c7",
    ),
}

REPRESENTATION = "StructuredActionV1"
GEOMETRY = "K16"
TIMING = "T16"
HORIZON = 40
UNIVERSAL_ACTION_DIM = 132
CHUNK_DURATION_S = 1.9500000000000002
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
HANDOFF_TARGET_S = 0.9750000000000001
RTX_DEVELOPMENT_SEED = 4242


class RtxPreflightError(RuntimeError):
    """Raised when a path is unsafe or cannot be bound to an identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_tree_entries(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RtxPreflightError(f"checkpoint root is not a non-symlink directory: {root}")

    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RtxPreflightError(f"checkpoint contains symlink directory entry: {child}")
            if not stat.S_ISDIR(mode):
                raise RtxPreflightError(f"checkpoint contains unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RtxPreflightError(f"checkpoint contains symlink file entry: {child}")
            if not stat.S_ISREG(mode):
                raise RtxPreflightError(f"checkpoint contains unsafe file entry: {child}")
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def canonical_tree_sha256(root: Path, prefix: str = STRUCTURED_CHECKPOINT_TREE_PREFIX) -> tuple[str, list[dict[str, Any]]]:
    """Return the accepted Stage38 canonical digest and its file records."""

    entries = _regular_tree_entries(Path(root))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest(), entries


def _path_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists() or path.is_symlink()),
        "is_symlink": bool(path.is_symlink()),
        "is_regular_file": False,
        "size_bytes": None,
        "sha256": None,
        "hash_match": False if expected_sha256 is not None else None,
    }
    if path.is_symlink():
        record["error"] = "symlink_not_allowed"
        if expected_sha256 is not None:
            record["expected_sha256"] = expected_sha256
        return record
    if not path.is_file():
        record["error"] = "missing_or_not_regular_file"
        if expected_sha256 is not None:
            record["expected_sha256"] = expected_sha256
        return record
    record["is_regular_file"] = True
    record["size_bytes"] = int(path.stat().st_size)
    record["sha256"] = sha256_file(path)
    if expected_sha256 is not None:
        record["expected_sha256"] = expected_sha256
        record["hash_match"] = record["sha256"] == expected_sha256
    return record


def audit_checkpoint_tree(path: Path) -> dict[str, Any]:
    """Hash every regular checkpoint file, rejecting all symlink entries."""

    path = Path(path)
    result: dict[str, Any] = {
        "label": "STRUCTURED@1000",
        "path": str(path),
        "expected_tree_sha256": STRUCTURED_CHECKPOINT_TREE_SHA256,
        "tree_sha256": None,
        "file_count": None,
        "total_bytes": None,
        "status": "BLOCKED",
    }
    try:
        observed, entries = canonical_tree_sha256(path)
    except (OSError, RtxPreflightError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        {
            "tree_sha256": observed,
            "file_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "status": "PASS" if observed == STRUCTURED_CHECKPOINT_TREE_SHA256 else "BLOCKED_TREE_HASH_MISMATCH",
        }
    )
    return result


def _safe_wbc_member_name(name: str, is_directory: bool) -> str:
    if not name or "\\" in name or name.startswith(("/", "\\")):
        raise RtxPreflightError(f"unsafe WBC archive member path: {name}")
    clean = name[:-1] if is_directory and name.endswith("/") else name
    if not clean or ":" in clean.split("/", 1)[0]:
        raise RtxPreflightError(f"ambiguous WBC archive member path: {name}")
    parts = clean.split("/")
    if any(part in {"", ".", ".."} for part in parts) or posixpath.normpath(clean) != clean:
        raise RtxPreflightError(f"non-canonical WBC archive member path: {name}")
    if clean != "WBCTrackingShare" and not clean.startswith("WBCTrackingShare/"):
        raise RtxPreflightError(f"WBC archive member outside authority prefix: {name}")
    return clean


def _wbc_v2_tree_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{item['path']}\t{int(item['size_bytes'])}\t{item['sha256']}\n"
        for item in sorted(records, key=lambda value: str(value["path"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_wbc_archive_members(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RtxPreflightError(f"WBC authority archive is not a regular file: {path}")
    names: set[str] = set()
    directories: set[str] = set()
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        for info in infos:
            is_directory = info.is_dir() or info.filename.endswith("/")
            clean = _safe_wbc_member_name(info.filename, is_directory)
            mode = (int(info.external_attr) >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RtxPreflightError(f"WBC archive encodes a symlink: {info.filename}")
            if clean in names:
                raise RtxPreflightError(f"WBC archive contains duplicate member: {clean}")
            names.add(clean)
            if is_directory:
                directories.add(clean)
                continue
            digest = hashlib.sha256()
            with archive.open(info, "r") as handle:
                for block in iter(lambda: handle.read(4 << 20), b""):
                    digest.update(block)
            records.append({
                "path": clean,
                "size_bytes": int(info.file_size),
                "sha256": digest.hexdigest(),
            })
    return {
        "member_count": len(infos),
        "file_count": len(records),
        "directory_count": len(directories),
        "file_records": sorted(records, key=lambda value: str(value["path"])),
        "directory_names": sorted(directories),
        "v2_tree_sha256": _wbc_v2_tree_sha256(records),
    }


def _audit_wbc_live_tree(source_root: Path) -> dict[str, Any]:
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise RtxPreflightError(f"WBC source root is not a non-symlink directory: {source_root}")
    records: list[dict[str, Any]] = []
    directories = {"WBCTrackingShare"}
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(source_root).as_posix()
        if relative_directory != ".":
            directories.add(f"WBCTrackingShare/{relative_directory}")
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not child.is_dir():
                raise RtxPreflightError(f"WBC source contains unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            if child.is_symlink() or not child.is_file():
                raise RtxPreflightError(f"WBC source contains unsafe file entry: {child}")
            records.append({
                "path": f"WBCTrackingShare/{child.relative_to(source_root).as_posix()}",
                "size_bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return {
        "file_count": len(records),
        "directory_count": len(directories),
        "file_records": sorted(records, key=lambda value: str(value["path"])),
        "directory_names": sorted(directories),
        "v2_tree_sha256": _wbc_v2_tree_sha256(records),
    }


def _compare_wbc_vectors(archive: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    archive_files = {str(item["path"]): item for item in archive["file_records"]}
    live_files = {str(item["path"]): item for item in live["file_records"]}
    missing = sorted(set(archive_files) - set(live_files))
    extra = sorted(set(live_files) - set(archive_files))
    changed = sorted(
        name for name in set(archive_files) & set(live_files)
        if int(archive_files[name]["size_bytes"]) != int(live_files[name]["size_bytes"])
        or archive_files[name]["sha256"] != live_files[name]["sha256"]
    )
    archive_dirs = set(archive["directory_names"])
    live_dirs = set(live["directory_names"])
    missing_dirs = sorted(archive_dirs - live_dirs)
    extra_dirs = sorted(live_dirs - archive_dirs)
    return {
        "missing_files": missing,
        "extra_files": extra,
        "changed_files": changed,
        "missing_directories": missing_dirs,
        "extra_directories": extra_dirs,
        "identical": not (missing or extra or changed or missing_dirs or extra_dirs),
    }


def audit_wbc_authority(archive_path: Path, source_root: Path | None) -> dict[str, Any]:
    """Verify current WBC authority without promoting the historical 633 digest."""

    archive_path = Path(archive_path)
    result: dict[str, Any] = {
        "authority_class": "STAGE39_WBC_CONTENT_AUTHORITY_V1_ATTESTED_FROM_SEALED_ARCHIVE",
        "archive": _path_record(archive_path, expected_sha256=WBC_ARCHIVE_SHA256),
        "source_root": None if source_root is None else str(source_root),
        "historical_reported_tree_sha256": HISTORICAL_WBC_TREE_SHA256,
        "historical_reported_tree_is_current_predicate": False,
        "expected_v2_tree_sha256": WBC_V2_TREE_SHA256,
        "status": "BLOCKED",
    }
    result["archive"]["expected_size_bytes"] = WBC_ARCHIVE_SIZE_BYTES
    result["archive"]["size_match"] = result["archive"]["size_bytes"] == WBC_ARCHIVE_SIZE_BYTES
    if not result["archive"]["hash_match"] or not result["archive"]["size_match"]:
        return result
    if source_root is None:
        result["error"] = "wbc_source_not_supplied"
        return result
    try:
        archive = _audit_wbc_archive_members(archive_path)
        live = _audit_wbc_live_tree(Path(source_root))
        comparison = _compare_wbc_vectors(archive, live)
        components = {
            name: _path_record(Path(source_root) / relative, expected_sha256=expected)
            for name, (relative, expected) in WBC_COMPONENTS.items()
        }
        result.update({
            "archive_members": {key: value for key, value in archive.items() if key not in {"file_records", "directory_names"}},
            "live_tree": {key: value for key, value in live.items() if key not in {"file_records", "directory_names"}},
            "archive_live_comparison": comparison,
            "components": components,
            "component_hashes_match": all(bool(value.get("hash_match")) for value in components.values()),
        })
        result["status"] = "PASS" if (
            archive["member_count"] == WBC_ARCHIVE_MEMBER_COUNT
            and archive["file_count"] == WBC_ARCHIVE_FILE_COUNT
            and archive["directory_count"] == WBC_ARCHIVE_DIRECTORY_COUNT
            and archive["v2_tree_sha256"] == WBC_V2_TREE_SHA256
            and live["file_count"] == WBC_ARCHIVE_FILE_COUNT
            and live["directory_count"] == WBC_ARCHIVE_DIRECTORY_COUNT
            and live["v2_tree_sha256"] == WBC_V2_TREE_SHA256
            and comparison["identical"]
            and result["component_hashes_match"]
        ) else "BLOCKED_IDENTITY_MISMATCH"
    except (OSError, ValueError, zipfile.BadZipFile, RtxPreflightError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _dataset_tree_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(
            f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def audit_tiny1_dataset(path: Path | None) -> dict[str, Any]:
    """Verify the exact one-episode train-only TINY1 fixture, without loaders."""

    result: dict[str, Any] = {
        "label": "TINY1_H2STATS",
        "path": None if path is None else str(path),
        "expected_tree_sha256": TINY1_TREE_SHA256,
        "tree_sha256": None,
        "expected_file_count": TINY1_FILE_COUNT,
        "expected_total_bytes": TINY1_TOTAL_BYTES,
        "status": "BLOCKED",
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if path is None:
        result["error"] = "path_not_supplied"
        return result
    root = Path(path)
    try:
        entries = _regular_tree_entries(root)
        observed = _dataset_tree_sha256(entries)
        result.update({
            "tree_sha256": observed,
            "file_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "stats_json": _path_record(root / "meta/stats.json", expected_sha256=TINY1_STATS_SHA256),
            "relative_stats_json": _path_record(
                root / "meta/relative_stats.json", expected_sha256=TINY1_RELATIVE_STATS_SHA256
            ),
            "meta_modality": _path_record(
                root / "meta/modality.json", expected_sha256=TINY1_META_MODALITY_SHA256
            ),
        })
        result["status"] = "PASS" if (
            observed == TINY1_TREE_SHA256
            and result["file_count"] == TINY1_FILE_COUNT
            and result["total_bytes"] == TINY1_TOTAL_BYTES
            and result["stats_json"]["hash_match"]
            and result["relative_stats_json"]["hash_match"]
            and result["meta_modality"]["hash_match"]
        ) else "BLOCKED_IDENTITY_MISMATCH"
    except (OSError, RtxPreflightError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _cosmos_tree_sha256(root: Path) -> tuple[str, int]:
    entries: list[dict[str, Any]] = []
    digest_cache: dict[str, str] = {}

    def file_sha(path: Path) -> str:
        key = str(path)
        if key not in digest_cache:
            digest_cache[key] = sha256_file(path)
        return digest_cache[key]

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(os.scandir(directory), key=lambda value: value.name):
            child = Path(item.path)
            rel = (relative / item.name).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(child, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(child)
                resolved = child.resolve(strict=True)
                try:
                    resolved_rel = resolved.relative_to(root.resolve(strict=True)).as_posix()
                except ValueError as exc:
                    raise RtxPreflightError(f"Cosmos symlink escapes root: {child}") from exc
                if not resolved.is_file():
                    raise RtxPreflightError(f"Cosmos symlink does not resolve to a file: {child}")
                entries.append({
                    "path": rel, "kind": "symlink", "target": target,
                    "resolved_path": resolved_rel, "size_bytes": int(resolved.stat().st_size),
                    "sha256": file_sha(resolved),
                })
            elif stat.S_ISREG(mode):
                entries.append({
                    "path": rel, "kind": "regular_file", "size_bytes": int(child.stat().st_size),
                    "sha256": file_sha(child),
                })
            else:
                raise RtxPreflightError(f"unsupported Cosmos entry: {child}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(("\t".join((
            str(entry["kind"]), str(entry["path"]), str(entry.get("size_bytes", 0)),
            str(entry.get("sha256") or "-"), str(entry.get("target") or "-"),
            str(entry.get("resolved_path") or "-"),
        )) + "\n").encode("utf-8"))
    return digest.hexdigest(), len(entries)


def audit_cosmos_cache(path: Path | None) -> dict[str, Any]:
    """Verify the materialized exact Cosmos cache; archive-only is not runnable."""

    result: dict[str, Any] = {
        "label": "COSMOS_REASON2_2B_CACHE",
        "path": None if path is None else str(path),
        "expected_revision": COSMOS_REVISION,
        "expected_tree_sha256": COSMOS_TREE_SHA256,
        "tree_sha256": None,
        "status": "BLOCKED",
        "archive_only_not_accepted": True,
    }
    if path is None:
        result["error"] = "path_not_supplied"
        return result
    root = Path(path)
    if root.is_file():
        result["error"] = "archive_only_materialization_required"
        result["kind"] = "regular_file"
        return result
    try:
        if root.is_symlink() or not root.is_dir():
            raise RtxPreflightError(f"Cosmos cache root is not a non-symlink directory: {root}")
        observed, count = _cosmos_tree_sha256(root)
        refs_main = root / "refs/main"
        snapshot = root / "snapshots" / COSMOS_REVISION
        result.update({
            "tree_sha256": observed,
            "entry_count": count,
            "refs_main": str(refs_main),
            "snapshot": str(snapshot),
            "revision": refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else None,
        })
        result["status"] = "PASS" if (
            observed == COSMOS_TREE_SHA256
            and result["revision"] == COSMOS_REVISION
            and snapshot.is_dir()
            and not snapshot.is_symlink()
        ) else "BLOCKED_IDENTITY_MISMATCH"
    except (OSError, RtxPreflightError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_modality(path: Path | None) -> dict[str, Any]:
    result = _path_record(Path(path), expected_sha256=STAGE22_MODALITY_SHA256) if path is not None else {
        "path": None, "exists": False, "is_symlink": False, "is_regular_file": False,
        "sha256": None, "expected_sha256": STAGE22_MODALITY_SHA256, "hash_match": False,
        "error": "path_not_supplied",
    }
    result["status"] = "PASS" if result.get("hash_match") else "BLOCKED"
    return result


def _git_output(source_root: Path, *args: str) -> tuple[str | None, str | None]:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(source_root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return value, None


def audit_groot_source(path: Path) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "expected_git_sha": GROOT_GIT_SHA,
        "git_sha": None,
        "worktree_status": None,
        "status": "BLOCKED",
    }
    if path.is_symlink() or not path.is_dir():
        result["error"] = "source_root_missing_or_symlink"
        return result
    sha, sha_error = _git_output(path, "rev-parse", "HEAD")
    status, status_error = _git_output(path, "status", "--porcelain=v1")
    result.update({"git_sha": sha, "worktree_status": status})
    errors = [error for error in (sha_error, status_error) if error]
    if errors:
        result["error"] = "; ".join(errors)
    result["status"] = "PASS" if sha == GROOT_GIT_SHA and status == "" and not errors else "BLOCKED"
    return result


def frozen_contract() -> dict[str, Any]:
    return {
        "representation": REPRESENTATION,
        "geometry": GEOMETRY,
        "timing": TIMING,
        "horizon": HORIZON,
        "universal_action_dim": UNIVERSAL_ACTION_DIM,
        "physical_action_dim": 10,
        "chunk_duration_s": CHUNK_DURATION_S,
        "frame": SOURCE_FRAME,
        "tcp": TCP_SEMANTIC_ID,
        "handoff_target_s": HANDOFF_TARGET_S,
        "development_seed": RTX_DEVELOPMENT_SEED,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def audit_storage_boundaries(
    *,
    materialized_root: Path | None,
    checkpoint: Path,
    structured_stats: Path,
    cosmos_cache: Path | None,
    mutable_cache_root: Path | None,
) -> dict[str, Any]:
    """Require received runnable inputs and future mutable cache to be disjoint."""

    result: dict[str, Any] = {
        "materialized_root": None if materialized_root is None else str(materialized_root),
        "mutable_cache_root": None if mutable_cache_root is None else str(mutable_cache_root),
        "received_inputs_within_materialized_root": False,
        "mutable_cache_disjoint": False,
        "status": "BLOCKED",
    }
    if materialized_root is None or mutable_cache_root is None or cosmos_cache is None:
        result["error"] = "materialized_root_mutable_cache_and_cosmos_are_required"
        return result
    materialized = Path(materialized_root)
    mutable = Path(mutable_cache_root)
    try:
        if materialized.is_symlink() or not materialized.is_dir():
            raise RtxPreflightError("materialized root must be an existing non-symlink directory")
        materialized_resolved = materialized.resolve(strict=True)
        received = [Path(checkpoint), Path(structured_stats), Path(cosmos_cache)]
        received_resolved = [path.resolve(strict=True) for path in received]
        if not all(_is_relative_to(path, materialized_resolved) for path in received_resolved):
            raise RtxPreflightError("received runnable input escapes the materialized root")
        writable_entries: list[str] = []
        for directory, dirnames, filenames in os.walk(materialized, followlinks=False):
            directory_path = Path(directory)
            for child in [directory_path, *(directory_path / name for name in dirnames + filenames)]:
                mode = child.lstat().st_mode
                if not stat.S_ISLNK(mode) and mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                    writable_entries.append(str(child))
        if writable_entries:
            raise RtxPreflightError("materialized input has writable entries")
        mutable_resolved = mutable.resolve(strict=False)
        if _is_relative_to(mutable_resolved, materialized_resolved) or _is_relative_to(
            materialized_resolved, mutable_resolved
        ):
            raise RtxPreflightError("mutable cache overlaps the immutable materialized root")
        if any(
            _is_relative_to(mutable_resolved, path) or _is_relative_to(path, mutable_resolved)
            for path in received_resolved
        ):
            raise RtxPreflightError("mutable cache overlaps an immutable runnable input")
        result.update({
            "materialized_root_realpath": str(materialized_resolved),
            "received_input_realpaths": [str(path) for path in received_resolved],
            "mutable_cache_realpath": str(mutable_resolved),
            "received_inputs_within_materialized_root": True,
            "materialized_write_bit_entries": [],
            "materialized_inputs_read_only": True,
            "mutable_cache_disjoint": True,
            "status": "PASS",
        })
    except (OSError, RtxPreflightError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_preflight(
    *,
    checkpoint: Path,
    structured_stats: Path,
    wbc_archive: Path,
    wbc_source: Path | None = None,
    groot_source: Path,
    tiny1_dataset: Path | None = None,
    modality: Path | None = None,
    cosmos_cache: Path | None = None,
    materialized_root: Path | None = None,
    mutable_cache_root: Path | None = None,
) -> dict[str, Any]:
    checkpoint_result = audit_checkpoint_tree(checkpoint)
    stats_result = _path_record(structured_stats, expected_sha256=STRUCTURED_STATS_SHA256)
    stats_result["status"] = "PASS" if stats_result["hash_match"] else "BLOCKED"
    wbc_result = audit_wbc_authority(wbc_archive, wbc_source)
    source_result = audit_groot_source(groot_source)
    tiny1_result = audit_tiny1_dataset(tiny1_dataset)
    modality_result = audit_modality(modality)
    cosmos_result = audit_cosmos_cache(cosmos_cache)
    storage_result = audit_storage_boundaries(
        materialized_root=materialized_root,
        checkpoint=checkpoint,
        structured_stats=structured_stats,
        cosmos_cache=cosmos_cache,
        mutable_cache_root=mutable_cache_root,
    )
    checks = {
        "checkpoint_tree": checkpoint_result["status"] == "PASS",
        "structured_statistics": stats_result["status"] == "PASS",
        "wbc_authority": wbc_result["status"] == "PASS",
        "groot_source": source_result["status"] == "PASS",
        "tiny1_dataset": tiny1_result["status"] == "PASS",
        "stage22_modality": modality_result["status"] == "PASS",
        "cosmos_cache": cosmos_result["status"] == "PASS",
        "storage_boundaries": storage_result["status"] == "PASS",
    }
    return {
        "schema": "STAGE40_RTX4080S_PREFLIGHT_V1",
        "classification": "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE",
        "pi_gate": PI_GATE,
        "issue_comment": ISSUE_COMMENT,
        "accepted_d1_git_sha": ACCEPTED_D1_GIT_SHA,
        "preflight_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_contract": frozen_contract(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "BLOCKED_PROVENANCE",
        "artifacts": {
            "structured_checkpoint": checkpoint_result,
            "structured_statistics": stats_result,
            "wbc_authority": wbc_result,
            "groot_source": source_result,
            "tiny1_dataset": tiny1_result,
            "stage22_modality": modality_result,
            "cosmos_cache": cosmos_result,
            "storage_boundaries": storage_result,
        },
        "execution": {
            "model_constructed": False,
            "forward": False,
            "gpu": False,
            "pbs_submitted": False,
            "training": False,
            "scientific_contract_changed": False,
            "formal_h200_science_modified": False,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structured-stats", type=Path, required=True)
    parser.add_argument("--wbc-archive", type=Path, required=True)
    parser.add_argument("--wbc-source", type=Path, required=True)
    parser.add_argument("--groot-source", type=Path, required=True)
    parser.add_argument("--tiny1-dataset", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--cosmos-cache", type=Path)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--mutable-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(
        checkpoint=args.checkpoint,
        structured_stats=args.structured_stats,
        wbc_archive=args.wbc_archive,
        wbc_source=args.wbc_source,
        groot_source=args.groot_source,
        tiny1_dataset=args.tiny1_dataset,
        modality=args.modality,
        cosmos_cache=args.cosmos_cache,
        materialized_root=args.materialized_root,
        mutable_cache_root=args.mutable_cache_root,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
