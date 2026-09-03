#!/usr/bin/env python3
"""Locate and verify an exact Stage24 O1 base checkpoint without loading a model.

The locator is deliberately index-first and bounded.  It never constructs a
model, imports torch, opens a GPU device, or follows a candidate until its
index SHA256 matches the frozen Stage22/23 identity.  A candidate is only
classified as exact after the referenced shards and the frozen model-tree
digest are explicitly verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_INDEX_SHA256 = (
    "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
)
EXPECTED_TREE_SHA256 = (
    "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
)
DEFAULT_TREE_PREFIX = "payload/GR00T-N1.7-3B"
DEFAULT_MAX_DEPTH = 8
_BLOCK_SIZE = 8 * 1024 * 1024

DEFAULT_ROOTS = (
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10",
    "/var/cache/users/liuyu-nfs/gr00t-n17",
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22",
    "/var/cache/users/liuyu-nfs/go2-arx5-stage23-o1-phase-a-20260903",
    "/home/liuyu-nfs/.cache/huggingface",
    "/var/cache/users/liuyu-nfs/.cache/huggingface",
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload",
)


class LocatorError(RuntimeError):
    """Raised when a requested verification cannot be performed safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LocatorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, LocatorError) as exc:
        raise LocatorError(f"cannot parse index {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LocatorError(f"index is not a JSON object: {path}")
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LocatorError(f"index has no non-empty weight_map: {path}")
    return value


def _safe_relative_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LocatorError(f"invalid shard filename: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise LocatorError(f"shard filename escapes model directory: {value!r}")
    return relative.as_posix()


def referenced_shards(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LocatorError("missing non-empty weight_map")
    shards = sorted({_safe_relative_filename(value) for value in weight_map.values()})
    if not shards:
        raise LocatorError("weight_map references no shards")
    return shards


def _relative_depth(path: Path, root: Path) -> int:
    return len(path.relative_to(root).parts)


def _raise_walk_error(error: OSError) -> None:
    raise LocatorError(f"cannot traverse candidate root: {error}") from error


def discover_indexes(root: Path, max_depth: int) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        return []

    matches: list[Path] = []
    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not (directory_path / name).is_symlink()
            and _relative_depth(directory_path / name, root) <= max_depth
        )
        if _relative_depth(directory_path, root) > max_depth:
            dirnames[:] = []
            continue
        for filename in sorted(filenames):
            if filename != "model.safetensors.index.json":
                continue
            path = directory_path / filename
            if not path.is_symlink():
                matches.append(path)
    return sorted(set(matches))


def _assert_regular_file(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise LocatorError(f"cannot stat {label}: {path}: {exc}") from exc
    if stat.S_IFMT(mode) != stat.S_IFREG:
        raise LocatorError(f"{label} is not a regular file: {path}")


def referenced_shard_inventory(model_dir: Path, shard_names: Iterable[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in shard_names:
        path = model_dir / name
        _assert_regular_file(path, "referenced shard")
        inventory.append({"path": name, "bytes": path.stat().st_size})
    return inventory


def _tree_entries(model_dir: Path, tree_prefix: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    model_dir = model_dir.resolve(strict=True)
    for directory, dirnames, filenames in os.walk(
        model_dir,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            path = directory_path / name
            mode = os.lstat(path).st_mode
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise LocatorError(f"tree contains symlink: {path}")
            if stat.S_IFMT(mode) != stat.S_IFDIR:
                raise LocatorError(f"tree contains unsupported directory entry: {path}")
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = directory_path / name
            mode = os.lstat(path).st_mode
            if stat.S_IFMT(mode) != stat.S_IFREG:
                raise LocatorError(f"tree contains non-regular file: {path}")
            relative = path.relative_to(model_dir).as_posix()
            entries.append(
                {
                    "path": f"{tree_prefix}/{relative}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(entries, key=lambda entry: entry["path"])


def tree_sha256(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def inspect_candidate(
    index_path: Path,
    *,
    verify_shards: bool,
    verify_tree: bool,
    tree_prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index_path": str(index_path),
        "model_dir": str(index_path.parent),
        "index_sha256": None,
        "index_sha256_expected": EXPECTED_INDEX_SHA256,
        "index_sha256_verified": False,
        "classification": "UNVERIFIED",
        "referenced_shards": [],
        "shard_hash_verification": "NOT_RUN_INDEX_MISMATCH",
        "tree_sha256_expected": EXPECTED_TREE_SHA256,
        "tree_sha256_actual": None,
        "tree_sha256_verified": False,
        "error": None,
    }
    try:
        _assert_regular_file(index_path, "index")
        result["index_sha256"] = sha256_file(index_path)
        if result["index_sha256"] != EXPECTED_INDEX_SHA256:
            result["classification"] = "INDEX_SHA_MISMATCH"
            return result
        result["index_sha256_verified"] = True
        index = load_index(index_path)
        shards = referenced_shards(index)
        result["referenced_shards"] = shards
        inventory = referenced_shard_inventory(index_path.parent, shards)
        result["referenced_shard_inventory"] = inventory
        if verify_shards:
            shard_hashes = []
            for item in inventory:
                shard_path = index_path.parent / item["path"]
                shard_hashes.append(
                    {
                        "path": item["path"],
                        "bytes": item["bytes"],
                        "sha256": sha256_file(shard_path),
                    }
                )
            result["referenced_shard_hashes"] = shard_hashes
            result["shard_hash_verification"] = "PASS"
        else:
            result["shard_hash_verification"] = "NOT_REQUESTED"
        if not verify_tree:
            result["classification"] = (
                "INDEX_MATCH_SHARDS_VERIFIED_NOT_TREE_SEALED"
                if verify_shards
                else "INDEX_MATCH_SHARDS_NOT_HASHED"
            )
            return result
        entries = _tree_entries(index_path.parent, tree_prefix)
        actual_tree = tree_sha256(entries)
        result["tree_sha256_actual"] = actual_tree
        result["tree_sha256_verified"] = actual_tree == EXPECTED_TREE_SHA256
        result["shard_hash_verification"] = "PASS_VIA_TREE_SCAN"
        result["classification"] = (
            "EXACT_O1_BASE_BYTE_EQUIVALENT_LOCAL_COPY"
            if result["tree_sha256_verified"]
            else "INDEX_MATCH_TREE_MISMATCH"
        )
    except (OSError, LocatorError, ValueError) as exc:
        result["classification"] = "INDEX_MATCH_VERIFICATION_ERROR" if result["index_sha256_verified"] else "INDEX_READ_ERROR"
        result["error"] = str(exc)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="bounded candidate root; repeat for each explicit root (defaults are known project/cache paths)",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--verify-shards", action="store_true", help="hash referenced shards after an index match")
    parser.add_argument("--verify-tree", action="store_true", help="hash the complete candidate tree and compare the frozen tree SHA")
    parser.add_argument("--tree-prefix", default=DEFAULT_TREE_PREFIX)
    parser.add_argument("--output", type=Path, help="write JSON result; refuses to overwrite an existing file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.max_depth < 0:
        print("ERROR: --max-depth must be non-negative", file=sys.stderr)
        return 2
    roots = [path.resolve() for path in (args.root or [Path(value) for value in DEFAULT_ROOTS])]
    candidates: list[Path] = []
    root_results: list[dict[str, Any]] = []
    for root in roots:
        indexes = discover_indexes(root, args.max_depth)
        root_results.append(
            {
                "root": str(root),
                "present": root.exists(),
                "indexes_found": [str(path) for path in indexes],
            }
        )
        candidates.extend(indexes)
    candidate_results = [
        inspect_candidate(
            path,
            verify_shards=args.verify_shards,
            verify_tree=args.verify_tree,
            tree_prefix=args.tree_prefix,
        )
        for path in sorted(set(candidates))
    ]
    exact = [
        result
        for result in candidate_results
        if result["classification"] == "EXACT_O1_BASE_BYTE_EQUIVALENT_LOCAL_COPY"
    ]
    payload = {
        "schema": "STAGE24_CHECKPOINT_LOCATOR_V1",
        "expected_index_sha256": EXPECTED_INDEX_SHA256,
        "expected_tree_sha256": EXPECTED_TREE_SHA256,
        "tree_prefix": args.tree_prefix,
        "max_depth": args.max_depth,
        "verification_mode": {
            "index_first": True,
            "verify_shards": bool(args.verify_shards or args.verify_tree),
            "verify_tree": bool(args.verify_tree),
        },
        "roots": root_results,
        "candidates": candidate_results,
        "exact_local_copies": exact,
        "classification": (
            "EXACT_O1_BASE_BYTE_EQUIVALENT_LOCAL_COPY" if exact else "NO_VERIFIED_LOCAL_COPY"
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
