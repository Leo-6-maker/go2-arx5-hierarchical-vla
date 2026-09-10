"""Shared, side-effect-light helpers for reusable experiment manifests.

This module is deliberately independent of Torch, GR00T, CUDA, PBS, WBC, and
the simulator.  It defines only content identity, canonical JSON, and common
manifest-boundary checks.  Stage-specific launchers may consume these helpers
after their own runtime admission has completed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ReusableManifestError(ValueError):
    """Raised when a future-facing experiment manifest is unsafe or incomplete."""


TEMPLATE_SENTINEL = "__REQUIRED__"
LATE_BOUND_SENTINEL = "__LATE_BOUND_AT_ADMISSION__"
TRAINING_BACKEND_V1 = "groot_h200_training_backend_v1"
INFERENCE_BACKEND_V1 = "groot_h200_inference_backend_v1"
RUNTIME_LATE_BOUND_FIELDS = (
    "repository_execution_head",
    "python_runtime",
    "container_or_environment",
    "groot_source",
    "cosmos_cache_root",
    "checkpoint_paths",
    "dataset_path",
    "statistics_path",
    "output_root",
    "scheduler_project",
    "scheduler_queue",
    "scheduler_resources",
    "gpu_device",
    "host",
    "offline_mode",
    "runtime_admission_identity",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for manifest and evidence hashing."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a regular JSON object without following a symlink at the leaf."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ReusableManifestError(f"JSON artifact may not be a symlink: {candidate}")
    candidate = candidate.resolve(strict=True)
    if not candidate.is_file():
        raise ReusableManifestError(f"JSON artifact is not a regular file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReusableManifestError(f"cannot read JSON artifact: {candidate}") from exc
    if not isinstance(value, dict):
        raise ReusableManifestError(f"JSON artifact must be an object: {candidate}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write a JSON evidence file with deterministic key ordering."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without dereferencing a symlink artifact."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ReusableManifestError(f"artifact is not a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a directory as a stable list of scientific content identities.

    The digest is a content-only identity for new reusable manifests:
    relative POSIX path, byte size, and file SHA256 are encoded for every
    regular file in lexical order.  Ordinary permission bits are deployment
    metadata, not scientific content.  Symlinks and non-regular files are
    rejected so a future job cannot silently see a different tree.
    """

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ReusableManifestError(f"artifact is not a regular directory: {root}")
    entries: list[bytes] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ReusableManifestError(f"tree contains a symlink: {candidate}")
        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ReusableManifestError(
                    f"tree contains a non-regular file: {candidate}"
                )
            relative = candidate.relative_to(root).as_posix()
            stat = candidate.stat()
            entries.append(
                f"file\0{relative}\0{stat.st_size}\0{sha256_file(candidate)}\n".encode()
            )
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(entry)
    return digest.hexdigest()


def tree_deployment_metadata(path: str | Path) -> dict[str, Any]:
    """Return deployment/security metadata separately from content identity.

    The content digest intentionally ignores ordinary mode changes.  Runtime
    admission can still reject unsafe copies (symlinks, non-regular files, or
    group/world-writable entries) using this report.
    """

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ReusableManifestError(f"artifact is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    unsafe: list[str] = []
    root_mode = root.stat().st_mode & 0o7777
    if root_mode & 0o022:
        unsafe.append(".")
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                unsafe.append(relative)
                continue
            mode = candidate.stat().st_mode & 0o7777
            entries.append({"path": relative, "kind": "directory", "mode": oct(mode)})
            if mode & 0o022:
                unsafe.append(relative)
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file():
                unsafe.append(relative)
                continue
            mode = candidate.stat().st_mode & 0o7777
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": oct(mode),
                    "bytes": candidate.stat().st_size,
                }
            )
            if mode & 0o022:
                unsafe.append(relative)
    entries.sort(key=lambda item: (item["path"], item["kind"]))
    unsafe.sort()
    return {
        "schema": "GO2ARX5_REUSABLE_DEPLOYMENT_METADATA_V1",
        "root_mode": oct(root_mode),
        "entries": entries,
        "unsafe_entries": unsafe,
        "safe": not unsafe,
    }


def validate_late_bound_runtime(
    value: Any, *, field: str, allow_sentinel: bool = False
) -> dict[str, Any]:
    """Require every runtime/PBS/device field to be supplied at admission."""

    if not isinstance(value, Mapping):
        raise ReusableManifestError(f"{field} must be an object")
    missing = [name for name in RUNTIME_LATE_BOUND_FIELDS if name not in value]
    if missing:
        raise ReusableManifestError(
            f"{field} must late-bind all runtime fields; missing: {', '.join(missing)}"
        )
    result = dict(value)
    for name in RUNTIME_LATE_BOUND_FIELDS:
        item = result[name]
        if allow_sentinel and item == LATE_BOUND_SENTINEL:
            continue
        if item is None or item == "" or item == [] or item == {}:
            raise ReusableManifestError(f"{field}.{name} must be explicit")
    return result


def artifact_sha256(path: str | Path) -> str:
    """Hash either a regular file or a regular directory tree."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise ReusableManifestError(
            f"symlink artifacts are not admissible: {candidate}"
        )
    if candidate.is_file():
        return sha256_file(candidate)
    if candidate.is_dir():
        return sha256_tree(candidate)
    raise ReusableManifestError(f"artifact does not exist: {candidate}")


def require_sha256(value: Any, *, field: str) -> str:
    """Validate and normalize a SHA256 field."""

    digest = str(value or "").strip().lower()
    if len(digest) != 64:
        raise ReusableManifestError(f"{field} must be a 64-character SHA256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ReusableManifestError(f"{field} is not hexadecimal") from exc
    return digest


def verify_artifact_identity(
    artifact: Mapping[str, Any],
    *,
    base_dir: Path,
    field: str,
) -> dict[str, Any]:
    """Resolve and verify a manifest artifact's path and content identity."""

    if not isinstance(artifact, Mapping):
        raise ReusableManifestError(f"{field} must be an object")
    raw_path = str(artifact.get("path", "")).strip()
    if not raw_path:
        raise ReusableManifestError(f"{field}.path is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    expected = require_sha256(artifact.get("sha256"), field=f"{field}.sha256")
    kind = str(artifact.get("kind", "file"))
    if kind not in {"file", "tree"}:
        raise ReusableManifestError(f"{field}.kind must be file or tree")
    if kind == "file" and not path.is_file():
        raise ReusableManifestError(f"{field} declares a file but is not one: {path}")
    if kind == "tree" and not path.is_dir():
        raise ReusableManifestError(f"{field} declares a tree but is not one: {path}")
    actual = artifact_sha256(path)
    if actual != expected:
        raise ReusableManifestError(
            f"{field} SHA256 mismatch: expected {expected}, got {actual} ({path})"
        )
    result = dict(artifact)
    result["resolved_path"] = str(path.resolve())
    result["sha256"] = expected
    result["verified"] = True
    if kind == "tree":
        deployment = tree_deployment_metadata(path)
        if not deployment["safe"]:
            raise ReusableManifestError(
                f"{field} deployment/security check failed: "
                f"{deployment['unsafe_entries']}"
            )
        result["deployment_metadata"] = deployment
    return result


def reject_locked_data(value: Any, *, key_path: str = "") -> None:
    """Reject accidental reads or paths into explicitly locked datasets."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            lowered = str(key).lower().replace("_", "-")
            if lowered in {"validation-e", "holdout-f"} and not (
                child == "LOCKED" or child == "locked" or child is None
            ):
                raise ReusableManifestError(
                    f"locked boundary was opened at {child_path}"
                )
            reject_locked_data(child, key_path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_locked_data(child, key_path=f"{key_path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if "validation-e" in lowered or "holdout-f" in lowered:
            raise ReusableManifestError(f"locked dataset path embedded at {key_path}")


def require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReusableManifestError(f"{field} must be an object")
    return value


def require_nonempty_string(value: Any, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ReusableManifestError(f"{field} must be non-empty")
    return result


__all__ = [
    "ReusableManifestError",
    "INFERENCE_BACKEND_V1",
    "LATE_BOUND_SENTINEL",
    "RUNTIME_LATE_BOUND_FIELDS",
    "TEMPLATE_SENTINEL",
    "TRAINING_BACKEND_V1",
    "artifact_sha256",
    "canonical_json_bytes",
    "load_json",
    "reject_locked_data",
    "require_mapping",
    "require_nonempty_string",
    "require_sha256",
    "sha256_file",
    "sha256_tree",
    "tree_deployment_metadata",
    "validate_late_bound_runtime",
    "verify_artifact_identity",
    "write_json",
]
