#!/usr/bin/env python3
"""Seal or verify a future Stage22 candidate checkpoint for transfer.

This tool records provenance and hashes only.  It does not select a model,
copy a checkpoint, upload it, or place it in the repository.  The default
selection status deliberately remains ``CANDIDATE_NOT_SELECTED``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_CREDENTIAL_COMPONENTS = {".env", ".netrc", "credential", "credentials", "password", "secret", "token"}


def _sha256_file(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        cache[key] = digest.hexdigest()
    return cache[key]


def _walk(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    digest_cache: dict[str, str] = {}

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(os.scandir(directory), key=lambda entry: entry.name):
            child = Path(item.path)
            child_relative = relative / item.name
            mode = stat.S_IFMT(os.lstat(child).st_mode)
            path = child_relative.as_posix()
            if mode == stat.S_IFDIR:
                entries.append({"path": path, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(child, child_relative)
            elif mode == stat.S_IFLNK:
                target = os.readlink(child)
                resolved = child.resolve(strict=True)
                try:
                    resolved_relative = resolved.relative_to(root.resolve(strict=True))
                except ValueError as exc:
                    raise RuntimeError(f"checkpoint symlink escapes root: {child} -> {resolved}") from exc
                if not resolved.is_file():
                    raise RuntimeError(f"checkpoint symlink does not resolve to a file: {child}")
                entries.append(
                    {
                        "path": path,
                        "kind": "symlink",
                        "target": target,
                        "resolved_path": resolved_relative.as_posix(),
                        "size_bytes": resolved.stat().st_size,
                        "sha256": _sha256_file(resolved, digest_cache),
                    }
                )
            elif mode == stat.S_IFREG:
                entries.append(
                    {
                        "path": path,
                        "kind": "regular_file",
                        "size_bytes": child.stat().st_size,
                        "sha256": _sha256_file(child, digest_cache),
                    }
                )
            else:
                raise RuntimeError(f"unsupported checkpoint entry: {child}")

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


def _credential_paths(entries: list[dict[str, Any]]) -> list[str]:
    result = []
    for entry in entries:
        components = {component.lower() for component in Path(entry["path"]).parts}
        if components & _CREDENTIAL_COMPONENTS:
            result.append(entry["path"])
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def seal_checkpoint(
    checkpoint: Path,
    *,
    run_manifest: Path,
    recipe: str,
    output: Path,
    selection_status: str = "CANDIDATE_NOT_SELECTED",
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    run_manifest = run_manifest.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be a directory")
    run = _load(run_manifest)
    if run.get("schema") != "STAGE20S_GR00T_TRAINING_RUN_V1" or run.get("status") != "PASS":
        raise ValueError("checkpoint provenance requires a successful Stage20S run manifest")
    expected_diffusion = recipe == "R1"
    if recipe not in {"R0", "R1"}:
        raise ValueError("recipe must be R0 or R1")
    if run.get("tune_diffusion_model") is not expected_diffusion:
        raise ValueError("run manifest recipe does not match requested checkpoint recipe")
    if run.get("action_horizon") != 40:
        raise ValueError("checkpoint run must use action_horizon=40")
    if run.get("simulation_only") is not True or run.get("hardware_control") is not False:
        raise ValueError("checkpoint provenance only accepts simulation-only runs")

    entries, digest_cache = _walk(checkpoint)
    credential_paths = _credential_paths(entries)
    if credential_paths:
        raise ValueError(f"credential-like checkpoint paths found: {credential_paths}")
    regular = [entry for entry in entries if entry["kind"] == "regular_file"]
    symlinks = [entry for entry in entries if entry["kind"] == "symlink"]
    result: dict[str, Any] = {
        "schema": "STAGE22_CHECKPOINT_PROVENANCE_V1",
        "status": "SEALED_FOR_HASH_VERIFICATION",
        "selection_status": selection_status,
        "transfer_performed": False,
        "checkpoint_name": checkpoint.name,
        "recipe": recipe,
        "checkpoint_file_count": len(regular) + len(symlinks),
        "regular_file_count": len(regular),
        "symlink_count": len(symlinks),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in regular),
        "checkpoint_tree_sha256": _tree_sha256(entries),
        "entries": sorted(entries, key=lambda value: value["path"]),
        "contains_credentials": False,
        "credential_path_audit": {
            "performed": True,
            "matching_paths": [],
            "token_content_read": False,
        },
        "source_run_manifest": {
            "sha256": _sha256_file(run_manifest, digest_cache),
            "path_recorded": str(run_manifest),
            "groot_repo_sha": run.get("groot_repo_sha"),
            "base_model_index_sha256": run.get("base_model_index_sha256"),
            "action_horizon": run.get("action_horizon"),
            "action_sample_period_s": run.get("action_sample_period_s"),
            "action_sample_period_provenance": run.get("action_sample_period_provenance"),
            "simulation_only": run.get("simulation_only"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_checkpoint(checkpoint: Path, manifest_path: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    expected = _load(manifest_path.resolve(strict=True))
    entries, _ = _walk(checkpoint)
    actual_entries = sorted(entries, key=lambda value: value["path"])
    expected_entries = expected.get("entries")
    mismatches: list[str] = []
    if actual_entries != expected_entries:
        expected_by_path = {entry["path"]: entry for entry in expected_entries or []}
        actual_by_path = {entry["path"]: entry for entry in actual_entries}
        for path in sorted(set(expected_by_path) | set(actual_by_path)):
            if expected_by_path.get(path) != actual_by_path.get(path):
                mismatches.append(path)
    actual_tree = _tree_sha256(actual_entries)
    if actual_tree != expected.get("checkpoint_tree_sha256"):
        mismatches.append("<tree_sha256>")
    return {
        "schema": "STAGE22_CHECKPOINT_PROVENANCE_VERIFY_V1",
        "status": "PASS" if not mismatches else "FAIL",
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path.resolve()),
        "expected_tree_sha256": expected.get("checkpoint_tree_sha256"),
        "actual_tree_sha256": actual_tree,
        "mismatches": sorted(set(mismatches)),
        "transfer_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--checkpoint", type=Path, required=True)
    seal_parser.add_argument("--run-manifest", type=Path, required=True)
    seal_parser.add_argument("--recipe", choices=("R0", "R1"), required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.add_argument("--selection-status", default="CANDIDATE_NOT_SELECTED")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--checkpoint", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal_checkpoint(
            args.checkpoint,
            run_manifest=args.run_manifest,
            recipe=args.recipe,
            output=args.output,
            selection_status=args.selection_status,
        )
    else:
        result = verify_checkpoint(args.checkpoint, args.manifest)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"PASS", "SEALED_FOR_HASH_VERIFICATION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
