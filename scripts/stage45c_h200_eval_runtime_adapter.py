#!/usr/bin/env python3
"""Stage45C H200 inference runtime adapter.

This module is the Corelab-side execution plumbing for the frozen Stage45C
protocol and the external execution admission.  It binds a
Vanda Golden Training Release V2 to the frozen Stage45C protocol and, when a
future H200 job invokes ``--execute``, performs the following single-job
pipeline::

    checkpoint -> GR00T construction -> TRAIN24 conditioning
               -> source-mapped four-step sampler -> raw universal action
               -> accepted Stage40 decode -> existing evaluator aggregate

The committed support branch contains no node-local paths and no runtime
artifact bytes.  Paths are supplied by an external, immutable binding at
execution time.  ``--plan-only`` and ``--audit-only`` are deliberately
CPU/static modes; they never import Torch/GR00T.  The execute path is
fail-closed before the first model import, binds the accepted Cosmos cache
into a fresh job-local Hugging Face namespace, and passes that namespace
through the nested GR00T loading contract.  It never trains, writes a
checkpoint, reads Validation-E/Holdout-F, runs WBC/simulation/hardware, or
changes the frozen scientific protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
# ``scripts`` is an implicit namespace package.  Keep both roots available
# for direct absolute-path launches from an unrelated CWD with no PYTHONPATH.
for _import_root in (REPO, SRC):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)


STAGE = "Stage45C"
# The checkpoint mapping's gate is immutable historical provenance.  The
# current execution authority is deliberately late-bound from the external
# runtime binding and is never represented by a stale adapter constant.
MAPPING_PROVENANCE_GATE = "PI_GATE_STATE_0091"
EXECUTION_AUTHORITY_GATE_PATTERN = re.compile(r"^PI_GATE_STATE_[0-9]{4}$")
PROTOCOL_SCHEMA = "CORELAB_STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1"
MAPPING_SCHEMA = "CORELAB_STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1"
BINDING_SCHEMA = "CORELAB_STAGE45C_H200_EVAL_RUNTIME_BINDING_V1"
ADAPTER_SCHEMA = "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_V1"
PROTOCOL_SHA256 = "0f8c564f534ce3b29a5029cc927f93a03697181b4b1df0aa9e80566af4cd020f"
PROTOCOL_SOURCE_HEAD = "54a833eaa7037cdc20397ef182b36647194040cb"
EXTERNAL_EXECUTION_HEAD = "EXTERNAL_ADMISSION_EXECUTION_HEAD"
GOLDEN_HEAD = "8197e4cd591b9c44cf9f43576d3020e9e6ff5f24"
PROMOTION_SHA256 = "a6f803e257ceeb0434027d8a35e32a7b8c66baf8226234bd6299cebf417446f9"
RELEASE_V2_SHA256 = "8306b76f327b260310a22a5d22551ebf299366f8fea4124bcbfbdab87e807666"
INTEGRITY_AUDIT_SHA256 = "a672e39b45e025f9a7681c1a1a45d807dd9be2a4641e54db60b036252bb59539"
RUNTIME_ADMISSION_SHA256 = "fc5aa17e5c41f9980b636106131233548021672db5c310bef545064775c35e57"
FULL24_MANIFEST_SHA256 = "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566"
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
STRUCTURED_1000_TREE_SHA256 = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
FULL24_1000_TREE_SHA256 = "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c"
FULL24_2000_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
FULL24_DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_SPLIT_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
BASE_PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_PROCESSOR_STATS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
HISTORICAL_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
GROOT_SOURCE_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
GROOT_RUNTIME_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
COSMOS_REPO_ID = "nvidia/Cosmos-Reason2-2B"
NESTED_COSMOS_CACHE_SCHEMA = "STAGE45C_NESTED_COSMOS_HF_CACHE_NAMESPACE_V1"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
DECODER_SOURCE_SHA256 = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"
METRICS_SOURCE_SHA256 = "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4"
CANONICAL_LABEL_SOURCE = "SEALED_FULL24_TRAIN_PARQUET"

HORIZON = 40
UNIVERSAL_DIM = 132
PHYSICAL_DIM = 10
STATE_DIM = 10
RGB_SHAPE = (96, 96, 3)
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
PHYSICAL_SHAPE = (HORIZON, PHYSICAL_DIM)
TIMESTEP_IDS = (0, 250, 500, 750)
SAMPLER_STEPS = 4
SAMPLER_DT = 0.25
DEVELOPMENT_SEED = 4242
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
TIMING_MODE = "T16"
WINDOW_ORDER = "target_major_then_snapshot_major"
CHECKPOINT_ORDER = ("BASE", "STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
TARGET_ORDER = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
TARGET_POSITIONS = {
    "red_target_train_a": (0.30, 0.10, 0.58),
    "red_target_train_b": (0.30, -0.10, 0.58),
    "red_target_train_c": (0.30, 0.05, 0.64),
    "red_target_train_d": (0.30, -0.05, 0.63),
}
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TARGET_ORDER
    for index in range(6)
)

CHECKPOINT_EXPECTATIONS = {
    "BASE": {
        "checkpoint_tree_sha256": BASE_TREE_SHA256,
        "checkpoint_tree_prefix": "payload/GR00T-N1.7-3B",
        "training_return_digest_sha256": None,
        "training_return_digest_prefix": None,
        "role": "REFERENCE",
        "decode_kind": "BASE_PROCESSOR_STATISTICS",
        "decode_stats_sha256": BASE_PROCESSOR_STATS_SHA256,
    },
    "STRUCTURED@1000": {
        "checkpoint_tree_sha256": STRUCTURED_1000_TREE_SHA256,
        "checkpoint_tree_prefix": "stage38d/checkpoint-1000",
        "training_return_digest_sha256": None,
        "training_return_digest_prefix": None,
        "role": "HISTORICAL_TINY1_COVERAGE_CONTROL",
        "decode_kind": "HISTORICAL_STRUCTURED_ACTION_STATISTICS",
        "decode_stats_sha256": HISTORICAL_STATS_SHA256,
    },
    "FULL24@1000": {
        "checkpoint_tree_sha256": FULL24_1000_TREE_SHA256,
        "checkpoint_tree_prefix": "checkpoint-1000",
        "training_return_digest_sha256": "8f0ebcf63ce61c0abb2f98b5e5f6802f9407fd4c8c132fc0dbce8f084bfce4c4",
        "training_return_digest_prefix": "stage45b/checkpoint-1000",
        "role": "PRIMARY_COVERAGE_ENDPOINT",
        "decode_kind": "FULL24_RUNTIME_V2_STATISTICS",
        "decode_stats_sha256": FULL24_STATS_SHA256,
    },
    "FULL24@2000": {
        "checkpoint_tree_sha256": FULL24_2000_TREE_SHA256,
        "checkpoint_tree_prefix": "checkpoint-2000",
        "training_return_digest_sha256": "705012d77f198514318dc1cf53cb9e31b51d8ac7a64a6135f8291546e32a3850",
        "training_return_digest_prefix": "stage45b/checkpoint-2000",
        "role": "SECONDARY_EXPOSURE_ENDPOINT",
        "decode_kind": "FULL24_RUNTIME_V2_STATISTICS",
        "decode_stats_sha256": FULL24_STATS_SHA256,
    },
}


def checkpoint_identity_namespaces() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "checkpoint_tree_sha256": values["checkpoint_tree_sha256"],
            "checkpoint_tree_prefix": values["checkpoint_tree_prefix"],
            "training_return_digest_sha256": values["training_return_digest_sha256"],
            "training_return_digest_prefix": values["training_return_digest_prefix"],
        }
        for label, values in CHECKPOINT_EXPECTATIONS.items()
    }


def checkpoint_identity_laws() -> dict[str, dict[str, str]]:
    return {
        "checkpoint_tree": {
            "field": "checkpoint_tree_sha256",
            "prefix_field": "checkpoint_tree_prefix",
            "law": "SHA256(concat(sorted((checkpoint-N + '/' + relative_path) + NUL + decimal_bytes + NUL + file_sha256 + LF)))",
            "scope": "all regular files in one materialized checkpoint directory",
        },
        "training_return": {
            "field": "training_return_digest_sha256",
            "prefix_field": "training_return_digest_prefix",
            "law": "same canonical line law with stage45b/checkpoint-N prefix",
            "scope": "same checkpoint files in PI0087 training-return namespace",
        },
    }


PATH_KEYS = (
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
    "modality_config",
    "base_processor_config",
    "base_processor_statistics",
    "structured_1000_statistics",
    "full24_runtime_statistics",
    "golden_release_manifest",
    "groot_runtime_manifest",
    "full24_manifest",
    "checkpoint_integrity_audit",
    "runtime_admission",
    "output_root",
)
DIRECTORY_PATH_KEYS = {
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
}
FILE_PATH_KEYS = set(PATH_KEYS).difference(DIRECTORY_PATH_KEYS | {"output_root"})


class AdapterError(RuntimeError):
    """Raised when the runtime adapter cannot prove an immutable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(payload)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def read_json(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"required JSON is unavailable: {path}") from exc
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot parse JSON: {path}") from exc


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"{label} is unavailable: {path}") from exc
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def regular_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"{label} is unavailable: {path}") from exc
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def _reject_runtime_paths(value: Any, *, location: str = "root") -> None:
    """Keep committed manifests pathless and reject accidental data leakage."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_runtime_paths(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_runtime_paths(child, location=f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        _require("/scratch/" not in lowered, f"node-local scratch path in {location}")
        _require("/var/cache/" not in lowered, f"node-local cache path in {location}")
        _require("/home/" not in lowered, f"node-local home path in {location}")
        _require("validation-e" not in lowered and "holdout-f" not in lowered, f"locked data path in {location}")


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    """Hash regular files with the accepted Stage22/45B tree law."""

    root = Path(root)
    regular_directory(root, "tree root")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe tree directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"unsafe tree file: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def relabel_tree_namespace(entries: Iterable[Mapping[str, Any]], old_prefix: str, new_prefix: str) -> list[dict[str, Any]]:
    """Rebind already-hashed files to a named digest namespace without rereading bytes."""

    old_prefix = old_prefix.rstrip("/")
    new_prefix = new_prefix.rstrip("/")
    result: list[dict[str, Any]] = []
    for entry in entries:
        path = str(entry["path"])
        _require(path.startswith(old_prefix + "/"), f"tree namespace prefix drift: {path}")
        result.append({
            "path": f"{new_prefix}/{path[len(old_prefix) + 1:]}",
            "bytes": int(entry["bytes"]),
            "sha256": str(entry["sha256"]),
        })
    return sorted(result, key=lambda item: str(item["path"]))


def cosmos_tree_digest(root: Path) -> tuple[str, int]:
    """Reproduce the accepted Cosmos cache digest (safe internal symlinks only)."""

    regular_directory(root, "Cosmos cache root")
    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(item)
                resolved = item.resolve(strict=True)
                _require(resolved.is_file(), f"Cosmos symlink is not a file: {item}")
                try:
                    resolved_rel = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise AdapterError(f"Cosmos symlink escapes root: {item}") from exc
                entries.append({
                    "path": rel,
                    "kind": "symlink",
                    "target": target,
                    "resolved_path": resolved_rel,
                    "size_bytes": int(resolved.stat().st_size),
                    "sha256": sha256_file(resolved),
                })
            elif stat.S_ISREG(mode):
                entries.append({
                    "path": rel,
                    "kind": "regular_file",
                    "size_bytes": int(item.stat().st_size),
                    "sha256": sha256_file(item),
                })
            else:
                raise AdapterError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        line = "\t".join((
            str(entry["kind"]),
            str(entry["path"]),
            str(entry.get("size_bytes", 0)),
            str(entry.get("sha256") or "-"),
            str(entry.get("target") or "-"),
            str(entry.get("resolved_path") or "-"),
        ))
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest(), len(entries)


def nested_cosmos_model_loading_kwargs(hub_root: Path) -> dict[str, Any]:
    """Return the exact nested Transformers kwargs used by GR00T's backbone."""

    regular_directory(Path(hub_root), "nested Cosmos Hugging Face hub")
    return {
        "trust_remote_code": True,
        "local_files_only": True,
        "cache_dir": str(Path(hub_root)),
        "revision": COSMOS_REVISION,
    }


def _resolve_nested_cosmos_config(hub_root: Path, cosmos_link: Path, cosmos_root: Path) -> dict[str, Any]:
    """Resolve a canonical Cosmos repo-id through the admitted local cache only."""

    from transformers.utils.hub import cached_file

    resolved = cached_file(
        COSMOS_REPO_ID,
        "config.json",
        cache_dir=str(hub_root),
        revision=COSMOS_REVISION,
        local_files_only=True,
    )
    _require(resolved is not None, "nested Cosmos config did not resolve from the local cache")
    logical_path = Path(resolved)
    resolved_path = logical_path.resolve(strict=True)
    expected_logical_path = cosmos_link / "snapshots" / COSMOS_REVISION / "config.json"
    expected_resolved_path = expected_logical_path.resolve(strict=True)
    _equal(resolved_path, expected_resolved_path, "nested Cosmos config resolution")
    root_resolved = Path(cosmos_root).resolve()
    _require(
        resolved_path == root_resolved or root_resolved in resolved_path.parents,
        "nested Cosmos config escaped the admitted cache root",
    )
    _require(resolved_path.is_file(), "nested Cosmos config is not a regular file")
    return {
        "repo_id": COSMOS_REPO_ID,
        "filename": "config.json",
        "revision": COSMOS_REVISION,
        "logical_path": str(logical_path),
        "resolved_path": str(resolved_path),
        "expected_logical_path": str(expected_logical_path),
        "expected_resolved_path": str(expected_resolved_path),
        "local_files_only": True,
    }


def _bind_nested_cosmos_hf_cache(
    output_root: Path,
    cosmos_root: Path,
    *,
    verified_tree_sha256: str | None = None,
    verified_entry_count: int | None = None,
) -> dict[str, Any]:
    """Bind the accepted Cosmos cache into a fresh, isolated job namespace."""

    regular_directory(output_root, "fresh Stage45C output root")
    regular_directory(cosmos_root, "accepted Cosmos cache root")
    cosmos_root = Path(cosmos_root).resolve()
    if verified_tree_sha256 is None:
        verified_tree_sha256, entry_count = cosmos_tree_digest(cosmos_root)
    else:
        entry_count = verified_entry_count
    _equal(verified_tree_sha256, COSMOS_TREE_SHA256, "accepted Cosmos cache tree")
    snapshot = cosmos_root / "snapshots" / COSMOS_REVISION
    regular_directory(snapshot, "accepted Cosmos pinned snapshot")
    refs_main = cosmos_root / "refs" / "main"
    regular_file(refs_main, "accepted Cosmos refs/main")
    _equal(refs_main.read_text(encoding="utf-8").strip(), COSMOS_REVISION, "accepted Cosmos refs/main")

    namespace_root = output_root / "stage45c_hf_runtime_namespace"
    hf_home = namespace_root / "huggingface"
    hub_root = hf_home / "hub"
    xdg_cache_home = namespace_root / "xdg_cache"
    torch_home = namespace_root / "torch"
    for path in (hub_root, xdg_cache_home, torch_home, hf_home / "datasets"):
        path.mkdir(parents=True, exist_ok=False)
    cosmos_link = hub_root / "models--nvidia--Cosmos-Reason2-2B"
    _require(not os.path.lexists(cosmos_link), "nested Cosmos cache link already exists")
    cosmos_link.symlink_to(cosmos_root, target_is_directory=True)
    _require(cosmos_link.is_symlink(), "nested Cosmos cache link was not created as a symlink")
    _equal(cosmos_link.resolve(strict=True), cosmos_root, "nested Cosmos cache link target")

    environment = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hub_root),
        "HF_HUB_CACHE": str(hub_root),
        "TRANSFORMERS_CACHE": str(hub_root),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "TORCH_HOME": str(torch_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    os.environ.update(environment)
    resolution = _resolve_nested_cosmos_config(hub_root, cosmos_link, cosmos_root)
    nested_kwargs = nested_cosmos_model_loading_kwargs(hub_root)
    return {
        "schema": NESTED_COSMOS_CACHE_SCHEMA,
        "status": "PASS",
        "namespace_root": str(namespace_root),
        "hf_home": str(hf_home),
        "hf_hub": str(hub_root),
        "cosmos_repo_id": COSMOS_REPO_ID,
        "cosmos_link": str(cosmos_link),
        "cosmos_link_target": str(cosmos_link.resolve(strict=True)),
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_tree_sha256": verified_tree_sha256,
        "cosmos_entry_count": entry_count,
        "cosmos_snapshot": str(snapshot),
        "cosmos_refs_main": str(refs_main),
        "cosmos_refs_main_value": refs_main.read_text(encoding="utf-8").strip(),
        "environment": environment,
        "nested_model_loading_kwargs": nested_kwargs,
        "repo_id_resolution": resolution,
        "namespace_isolation": {
            "ambient_repo_id_not_used": True,
            "local_files_only": True,
            "network_fetch": False,
            "fresh_namespace": True,
        },
    }


def _protocol_contract_audit(protocol: Mapping[str, Any]) -> None:
    _equal(protocol.get("schema"), PROTOCOL_SCHEMA, "frozen protocol schema")
    _equal(protocol.get("stage"), STAGE, "frozen protocol stage")
    _equal(protocol.get("source_golden_head"), GOLDEN_HEAD, "frozen Golden HEAD")
    _require(protocol.get("status") in {"BOUND_NOT_EXECUTED", "PREPARED_NOT_EXECUTED"}, "frozen protocol execution status drift")
    cohort = protocol.get("cohort")
    _require(isinstance(cohort, Mapping), "frozen TRAIN24 cohort missing")
    _equal(cohort.get("name"), "FULL24_EXISTING_TRAIN", "cohort name")
    _equal(cohort.get("split"), "TRAIN", "cohort split")
    _equal(cohort.get("expected_window_count"), 24, "cohort window count")
    _equal(cohort.get("window_order"), WINDOW_ORDER, "cohort order")
    _equal(list(cohort.get("window_ids", [])), list(WINDOW_IDS), "cohort window IDs")
    contract = protocol.get("contract")
    _require(isinstance(contract, Mapping), "frozen model contract missing")
    expected_contract = {
        "structured_action_schema": "StructuredActionV1",
        "geometry": "K16",
        "timing": TIMING_MODE,
        "horizon": "H40",
        "raw_action_shape": [1, 40, 132],
        "decoded_action_shape": [40, 10],
        "state_v1_shape": [10],
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "target_xyz_in_model_input": False,
        "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
    }
    for key, expected in expected_contract.items():
        _equal(contract.get(key), expected, f"frozen contract {key}")
    budget = protocol.get("execution_budget")
    _require(isinstance(budget, Mapping), "frozen execution budget missing")
    _equal(budget.get("forwards_per_checkpoint"), 24, "forward budget")
    _equal(budget.get("planned_forwards_total"), 96, "planned forward budget")
    _equal(budget.get("reference_updates"), 0, "reference update budget")
    _equal(budget.get("wbc_steps"), 0, "WBC budget")
    locked = protocol.get("locked_inputs")
    _require(isinstance(locked, Mapping), "frozen locked inputs missing")
    for key in ("Validation-E", "Holdout-F", "WBC", "hardware", "new_data"):
        _equal(locked.get(key), "LOCKED", f"locked input {key}")


def audit_protocol_and_mapping(protocol_path: Path, mapping_path: Path) -> dict[str, Any]:
    """Audit the frozen protocol and pathless checkpoint mapping."""

    regular_file(protocol_path, "frozen protocol")
    observed_protocol_sha = sha256_file(protocol_path)
    _equal(observed_protocol_sha, PROTOCOL_SHA256, "frozen protocol SHA256")
    protocol = read_json(protocol_path)
    _require(isinstance(protocol, Mapping), "frozen protocol must be an object")
    _protocol_contract_audit(protocol)

    regular_file(mapping_path, "checkpoint mapping")
    mapping_sha = sha256_file(mapping_path)
    mapping = read_json(mapping_path)
    _require(isinstance(mapping, Mapping), "checkpoint mapping must be an object")
    _equal(mapping.get("schema"), MAPPING_SCHEMA, "checkpoint mapping schema")
    _equal(mapping.get("stage"), STAGE, "checkpoint mapping stage")
    _equal(mapping.get("pi_gate"), MAPPING_PROVENANCE_GATE, "checkpoint mapping provenance gate")
    _equal(mapping.get("status"), "AUDITED_NOT_EXECUTED", "checkpoint mapping status")
    _equal(mapping.get("token"), "STAGE45C_CHECKPOINT_RUNTIME_MAPPING_AUDITED", "checkpoint mapping token")
    _equal(mapping.get("protocol_sha256"), PROTOCOL_SHA256, "checkpoint mapping protocol SHA")
    _equal(mapping.get("protocol_source_head"), PROTOCOL_SOURCE_HEAD, "checkpoint mapping protocol source head")
    _equal(mapping.get("execution_head"), EXTERNAL_EXECUTION_HEAD, "checkpoint mapping execution head binding")
    _equal(list(mapping.get("checkpoint_order", [])), list(CHECKPOINT_ORDER), "checkpoint order")
    _equal(
        mapping.get("checkpoint_identity_laws"),
        checkpoint_identity_laws(),
        "checkpoint identity namespace declarations",
    )
    entries = mapping.get("checkpoint_runtime_mapping")
    _require(isinstance(entries, list) and len(entries) == len(CHECKPOINT_ORDER), "four checkpoint mappings required")
    for item, label in zip(entries, CHECKPOINT_ORDER):
        _require(isinstance(item, Mapping), f"mapping entry {label} is not an object")
        expected = CHECKPOINT_EXPECTATIONS[label]
        _equal(item.get("label"), label, f"mapping label {label}")
        _equal(item.get("role"), expected["role"], f"mapping role {label}")
        _require("tree_sha256" not in item and "tree_prefix" not in item, f"ambiguous checkpoint identity fields remain for {label}")
        _equal(item.get("checkpoint_tree_sha256"), expected["checkpoint_tree_sha256"], f"mapping checkpoint tree {label}")
        _equal(item.get("checkpoint_tree_prefix"), expected["checkpoint_tree_prefix"], f"mapping checkpoint tree prefix {label}")
        _equal(item.get("training_return_digest_sha256"), expected["training_return_digest_sha256"], f"mapping training-return digest {label}")
        _equal(item.get("training_return_digest_prefix"), expected["training_return_digest_prefix"], f"mapping training-return digest prefix {label}")
        _equal(item.get("processor_config_sha256"), BASE_PROCESSOR_CONFIG_SHA256, f"processor config {label}")
        _equal(item.get("processor_statistics_sha256"), BASE_PROCESSOR_STATS_SHA256, f"processor statistics {label}")
        decode = item.get("decode_statistics")
        _require(isinstance(decode, Mapping), f"decode statistics missing for {label}")
        _equal(decode.get("kind"), expected["decode_kind"], f"decode kind {label}")
        _equal(decode.get("sha256"), expected["decode_stats_sha256"], f"decode statistics {label}")
        if label == "STRUCTURED@1000":
            _equal(decode.get("runtime_materialization_status"), "RECOVERED_CANONICAL_READ_ONLY", "historical runtime state")
            _equal(decode.get("recovery_evidence_sha256"), "c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9", "historical recovery evidence")
            _require("path" not in decode or decode.get("path") is None, "historical mapping may not carry a path")
        if label.startswith("FULL24"):
            _equal(decode.get("payload_sha256"), FULL24_STATS_PAYLOAD_SHA256, f"V2 payload {label}")

    historical = mapping.get("historical_comparator")
    _require(isinstance(historical, Mapping), "historical comparator mapping missing")
    for key, expected in {
        "label": "STRUCTURED@1000",
        "expected_checkpoint_tree_sha256": STRUCTURED_1000_TREE_SHA256,
        "expected_file_count": 19,
        "expected_total_bytes": 20889596357,
        "expected_structured_action_stats_sha256": HISTORICAL_STATS_SHA256,
        "known_package_sha256": "4e2f5ca562f108b0f0ec55cd97083d163b4e450be0e8c4ae0f1215bf98b458b6",
        "runtime_materialization_status": "RECOVERED_CANONICAL_READ_ONLY",
        "recovery_evidence_sha256": "c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9",
        "materialization_path": None,
        "substitution_allowed": False,
    }.items():
        _equal(historical.get(key), expected, f"historical comparator {key}")

    decoder = mapping.get("decoder")
    _require(isinstance(decoder, Mapping), "decoder mapping missing")
    _equal(decoder.get("name"), "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action", "decoder authority")
    _equal(decoder.get("source_sha256"), DECODER_SOURCE_SHA256, "decoder source SHA")
    _equal(decoder.get("raw_shape"), [1, 40, 132], "decoder raw shape")
    _equal(decoder.get("decoded_shape"), [40, 10], "decoder physical shape")
    _equal(decoder.get("normalization"), "checkpoint_specific_decode_statistics", "decoder normalization")
    _equal(decoder.get("source_frame"), SOURCE_FRAME, "decoder source frame")
    _equal(decoder.get("tcp_semantic_id"), TCP_SEMANTIC_ID, "decoder TCP")
    _equal(decoder.get("structured_action_schema"), "StructuredActionV1", "decoder schema")
    _equal(decoder.get("geometry"), "K16", "decoder geometry")
    _equal(decoder.get("timing"), TIMING_MODE, "decoder timing")

    sampler = mapping.get("sampler_noise_protocol")
    _require(isinstance(sampler, Mapping), "sampler protocol missing")
    for key, expected in {
        "development_seed": DEVELOPMENT_SEED,
        "noise_shape": [1, 40, 132],
        "same_noise_across_checkpoints": True,
        "sampler": "source-mapped-four-step-Euler-action-head-graph",
        "official_public_sampler_called": False,
        "steps": SAMPLER_STEPS,
        "dt": SAMPLER_DT,
        "timestep_ids": list(TIMESTEP_IDS),
        "checkpoint_order": list(CHECKPOINT_ORDER),
        "windows_per_checkpoint": 24,
        "planned_forwards_total": 96,
        "adaptive_extension": False,
    }.items():
        _equal(sampler.get(key), expected, f"sampler {key}")
    input_contract = mapping.get("model_input_contract")
    _require(isinstance(input_contract, Mapping), "model input contract missing")
    for key, expected in {
        "keys": ["body_rgb", "wrist_rgb", "state"],
        "body_rgb_shape": [96, 96, 3],
        "wrist_rgb_shape": [96, 96, 3],
        "state_v1_shape": [10],
        "language": LANGUAGE,
        "target_xyz_in_model_input": False,
    }.items():
        _equal(input_contract.get(key), expected, f"model input {key}")
    boundary = mapping.get("execution_boundary")
    _require(isinstance(boundary, Mapping), "mapping execution boundary missing")
    for key in ("optimizer_steps", "wbc_steps", "reference_updates", "pbs_submissions_by_corelab"):
        _equal(boundary.get(key), 0, f"mapping zero boundary {key}")
    for key in ("training", "backward", "resume_from_checkpoint", "checkpoint_write", "validation_e_read", "holdout_f_read", "simulator", "hardware"):
        _equal(boundary.get(key), False, f"mapping false boundary {key}")
    _reject_runtime_paths(mapping)
    return {
        "protocol_sha256": observed_protocol_sha,
        "mapping_sha256": mapping_sha,
        "mapping_provenance_gate": MAPPING_PROVENANCE_GATE,
        "protocol_source_head": PROTOCOL_SOURCE_HEAD,
        "execution_head": EXTERNAL_EXECUTION_HEAD,
        "checkpoint_order": list(CHECKPOINT_ORDER),
        "windows_per_checkpoint": 24,
        "planned_forwards_total": 96,
        "historical_materialization": "RECOVERED_CANONICAL_READ_ONLY",
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "metrics_source_sha256": METRICS_SOURCE_SHA256,
        "scientific_contract_changed": False,
        "status": "PASS",
    }


def _expected_identity_map() -> dict[str, Any]:
    return {
        "golden_release": {
            "release_id": "GOLDEN_TRAINING_RELEASE_V2",
            "promotion_manifest_sha256": PROMOTION_SHA256,
            "release_v2_sha256": RELEASE_V2_SHA256,
            "status": "ACCEPTED_READ_ONLY",
        },
        "base": {
            "tree_sha256": BASE_TREE_SHA256,
            "processor_config_sha256": BASE_PROCESSOR_CONFIG_SHA256,
            "processor_statistics_sha256": BASE_PROCESSOR_STATS_SHA256,
        },
        "full24_train24": {
            "tree_sha256": FULL24_DATASET_TREE_SHA256,
            "split_lock_sha256": FULL24_SPLIT_SHA256,
            "manifest_sha256": FULL24_MANIFEST_SHA256,
            "window_count": 24,
            "frame_count": 960,
            "window_order": WINDOW_ORDER,
            "window_ids": list(WINDOW_IDS),
            "target_order": list(TARGET_ORDER),
            "target_positions_world_m": {key: list(value) for key, value in TARGET_POSITIONS.items()},
        },
        "runtime_v2": {
            "stats_sha256": FULL24_STATS_SHA256,
            "payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "protocol_revision": "RUNTIME_PARQUET_PROVENANCE_V2",
        },
        "canonical_label_source": {
            "kind": CANONICAL_LABEL_SOURCE,
            "tree_sha256": FULL24_DATASET_TREE_SHA256,
            "parquet_files_read": 24,
            "label_records": 24,
            "stats_sha256": FULL24_STATS_SHA256,
            "surviving_historical_source_json_count": 6,
            "unavailable_historical_replica_count": 18,
            "historical_source_json_execution_authority": False,
            "missing_replicas_are_not_execution_authority": True,
            "target_xyz_in_model_input": False,
        },
        "groot_runtime": {
            "source_commit": GROOT_SOURCE_COMMIT,
            "payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA256,
            "runtime_manifest_sha256": GROOT_RUNTIME_MANIFEST_SHA256,
            "cosmos_revision": COSMOS_REVISION,
            "cosmos_tree_sha256": COSMOS_TREE_SHA256,
            "modality_sha256": MODALITY_SHA256,
            "training_config_sha256": CONFIG_SHA256,
        },
        "checkpoint_identity_namespaces": checkpoint_identity_namespaces(),
        "checkpoint_identity_laws": checkpoint_identity_laws(),
        "checkpoint_integrity_delta_audit_sha256": INTEGRITY_AUDIT_SHA256,
        "runtime_admission_sha256": RUNTIME_ADMISSION_SHA256,
        "historical_comparator": {
            "checkpoint_tree_sha256": STRUCTURED_1000_TREE_SHA256,
            "file_count": 19,
            "bytes": 20889596357,
            "stats_sha256": HISTORICAL_STATS_SHA256,
            "runtime_materialization_status": "RECOVERED_CANONICAL_READ_ONLY",
            "recovery_evidence_sha256": "c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9",
        },
    }


def validate_runtime_binding(binding: Mapping[str, Any], mapping_summary: Mapping[str, Any], *, require_paths: bool) -> dict[str, Any]:
    """Validate an external binding without resolving any committed path."""

    _equal(binding.get("schema"), BINDING_SCHEMA, "runtime binding schema")
    _equal(binding.get("stage"), STAGE, "runtime binding stage")
    execution_authority_gate = binding.get("pi_gate")
    _require(
        isinstance(execution_authority_gate, str)
        and EXECUTION_AUTHORITY_GATE_PATTERN.fullmatch(execution_authority_gate) is not None,
        "runtime binding execution authority gate is missing or malformed",
    )
    if "control_gate" in binding:
        _equal(binding.get("control_gate"), execution_authority_gate, "runtime binding control gate")
    if "execution_authority_gate" in binding:
        _equal(
            binding.get("execution_authority_gate"),
            execution_authority_gate,
            "runtime binding execution authority gate",
        )
    if require_paths:
        _require("control_gate" in binding, "bound runtime binding control gate is missing")
        _require("execution_authority_gate" in binding, "bound runtime binding execution authority gate is missing")
    _require(binding.get("status") in {"LATE_BOUND_NOT_EXECUTED", "READY_NOT_EXECUTED", "BOUND_NOT_EXECUTED", "READY_FOR_EXECUTION"}, "runtime binding status is not admissible")
    _equal(binding.get("protocol_sha256"), PROTOCOL_SHA256, "runtime binding protocol SHA")
    _equal(binding.get("mapping_sha256"), mapping_summary.get("mapping_sha256"), "runtime binding mapping SHA")
    _equal(binding.get("protocol_source_head"), PROTOCOL_SOURCE_HEAD, "runtime binding protocol source head")
    execution_head = binding.get("execution_head")
    if require_paths:
        _require(isinstance(execution_head, str) and len(execution_head) == 40 and all(char in "0123456789abcdef" for char in execution_head), "runtime binding execution head must be an exact commit SHA")
    else:
        _equal(execution_head, EXTERNAL_EXECUTION_HEAD, "runtime binding execution head placeholder")
    _equal(binding.get("source_golden_head"), GOLDEN_HEAD, "runtime binding Golden HEAD")

    identities = binding.get("identities")
    _require(isinstance(identities, Mapping), "runtime identity map missing")
    expected = _expected_identity_map()
    for section, values in expected.items():
        observed = identities.get(section)
        if isinstance(values, Mapping):
            _require(isinstance(observed, Mapping), f"runtime identity section missing: {section}")
            for key, expected_value in values.items():
                _equal(observed.get(key), expected_value, f"runtime identity {section}.{key}")
        else:
            _equal(observed, values, f"runtime identity {section}")
    _equal(binding.get("accepted_authorities", {}).get("golden_release_promotion_manifest_sha256"), PROMOTION_SHA256, "promotion authority")
    _equal(binding.get("accepted_authorities", {}).get("checkpoint_integrity_delta_audit_sha256"), INTEGRITY_AUDIT_SHA256, "integrity authority")

    paths = binding.get("paths")
    _require(isinstance(paths, Mapping), "runtime binding paths map missing")
    _equal(sorted(paths), sorted(PATH_KEYS), "runtime binding path key set")
    present: dict[str, str | None] = {}
    for key in PATH_KEYS:
        value = paths.get(key)
        if value is None:
            _require(not require_paths, f"runtime path is not late-bound: {key}")
            present[key] = None
            continue
        _require(isinstance(value, str) and value.strip(), f"runtime path must be a string or null: {key}")
        path = Path(value)
        lowered_path = value.lower()
        _require("validation-e" not in lowered_path and "holdout-f" not in lowered_path, f"locked data path at runtime: {key}")
        _require(path.is_absolute() if require_paths else True, f"runtime path must be absolute at execution: {key}")
        present[key] = str(path)
        if require_paths:
            if key == "output_root":
                _require(not path.exists(), f"output root already exists: {path}")
            elif key in DIRECTORY_PATH_KEYS:
                regular_directory(path, key)
            else:
                regular_file(path, key)

    boundary = binding.get("execution_boundary")
    _require(isinstance(boundary, Mapping), "runtime binding execution boundary missing")
    for key, expected_value in {
        "model_construction": "RUNTIME_ONLY",
        "model_forward": "RUNTIME_ONLY",
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc_steps": 0,
        "reference_updates": 0,
        "simulator": False,
        "hardware": False,
        "pbs_submissions_by_corelab": 0,
    }.items():
        _equal(boundary.get(key), expected_value, f"runtime binding boundary {key}")
    historical = binding.get("historical_comparator", {})
    _equal(historical.get("runtime_materialization_status"), "RECOVERED_CANONICAL_READ_ONLY", "binding historical status")
    _equal(historical.get("recovery_evidence_sha256"), "c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9", "binding historical recovery evidence")
    if require_paths:
        _equal(historical.get("materialization_path"), paths["structured_1000_checkpoint"], "binding historical materialization path")
    else:
        _equal(historical.get("materialization_path"), None, "binding historical path")
    _equal(historical.get("substitution_allowed"), False, "binding historical substitution")
    return {
        "paths": present,
        "status": "PASS",
        "require_paths": require_paths,
        "mapping_provenance_gate": MAPPING_PROVENANCE_GATE,
        "execution_authority_gate": execution_authority_gate,
    }


def _hash_identity(path: Path, expected: str, label: str) -> dict[str, Any]:
    regular_file(path, label)
    observed = sha256_file(path)
    _equal(observed, expected, f"{label} SHA256")
    return {"path": str(path), "sha256": observed, "bytes": int(path.stat().st_size)}


def _find_nested(value: Any, key: str) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if key in value and isinstance(value[key], Mapping):
            found.append(value[key])
        for child in value.values():
            found.extend(_find_nested(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_nested(child, key))
    return found


def _verify_full24_manifest(path: Path) -> dict[str, Any]:
    identity = _hash_identity(path, FULL24_MANIFEST_SHA256, "FULL24 manifest")
    document = read_json(path)
    candidates = _find_nested(document, "full24_train24") + _find_nested(document, "full24_identity")
    if candidates:
        candidate = next((item for item in candidates if item.get("tree_sha256") == FULL24_DATASET_TREE_SHA256), None)
        _require(candidate is not None, "FULL24 manifest lacks the accepted dataset identity")
        _equal(candidate.get("window_count"), 24, "FULL24 manifest window count")
        _equal(candidate.get("frame_count"), 960, "FULL24 manifest frame count")
        _equal(candidate.get("split_lock_sha256"), FULL24_SPLIT_SHA256, "FULL24 manifest split lock")
        _equal(candidate.get("window_order"), WINDOW_ORDER, "FULL24 manifest order")
    return identity


def _verify_release_manifest(path: Path) -> dict[str, Any]:
    identity = _hash_identity(path, PROMOTION_SHA256, "Golden Release promotion manifest")
    document = read_json(path)
    if isinstance(document, Mapping):
        if document.get("status") is not None:
            _require(
                document.get("status") in {"SEALED", "ACCEPTED_READ_ONLY", "PASS"},
                "Golden Release status is not sealed/read-only/accepted",
            )
        if document.get("release_id") is not None:
            _equal(document.get("release_id"), "GOLDEN_TRAINING_RELEASE_V2", "Golden Release ID")
    return identity


def _verify_groot_runtime_manifest(path: Path) -> dict[str, Any]:
    identity = _hash_identity(path, GROOT_RUNTIME_MANIFEST_SHA256, "GR00T runtime manifest")
    document = read_json(path)
    _require(isinstance(document, Mapping), "GR00T runtime manifest is not an object")
    source_commit = document.get("source_commit") or document.get("origin_commit") or document.get("groot_source_commit_sha")
    if source_commit is not None:
        _equal(source_commit, GROOT_SOURCE_COMMIT, "GR00T source commit")
    payload_tree = document.get("payload_tree_sha256") or document.get("payload_tree")
    if payload_tree is not None:
        _equal(payload_tree, GROOT_PAYLOAD_TREE_SHA256, "GR00T payload tree")
    return identity


def _verify_stats(path: Path, expected_sha: str, expected_payload: str | None, label: str) -> dict[str, Any]:
    identity = _hash_identity(path, expected_sha, label)
    document = read_json(path)
    _require(isinstance(document, Mapping), f"{label} is not an object")
    if expected_payload is not None:
        _equal(document.get("statistics_payload_sha256"), expected_payload, f"{label} payload SHA")
    return identity


def _verify_runtime_stats(path: Path) -> dict[str, Any]:
    identity = _verify_stats(path, FULL24_STATS_SHA256, FULL24_STATS_PAYLOAD_SHA256, "FULL24 runtime V2 statistics")
    document = read_json(path)
    statistics = document.get("statistics", document)
    _require(isinstance(statistics, Mapping), "runtime V2 statistics payload missing")
    _equal(document.get("schema"), "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2", "runtime V2 statistics artifact schema")
    _equal(statistics.get("schema"), "STAGE38_STRUCTURED_ACTION_STATISTICS_V1", "runtime V2 statistics payload schema")
    _equal(statistics.get("timing_mode"), TIMING_MODE, "runtime V2 timing")
    _equal(statistics.get("action_horizon"), HORIZON, "runtime V2 horizon")
    _equal(statistics.get("universal_action_dim"), UNIVERSAL_DIM, "runtime V2 action dimension")
    _equal(statistics.get("selected_lower_bound"), "q01", "runtime V2 lower bound")
    _equal(document.get("protocol_revision"), "RUNTIME_PARQUET_PROVENANCE_V2", "runtime V2 protocol revision")
    _equal(document.get("status"), "VANDA_STAGE45B_FULL24_RUNTIME_DATASET_QUALIFIED_V2", "runtime V2 qualification token")
    canonical = document.get("canonical_label_source")
    _require(isinstance(canonical, Mapping), "runtime V2 canonical label source is missing")
    _equal(canonical.get("kind"), CANONICAL_LABEL_SOURCE, "runtime V2 canonical label source kind")
    _equal(canonical.get("tree_sha256"), FULL24_DATASET_TREE_SHA256, "runtime V2 canonical label source tree")
    _equal(canonical.get("parquet_files_read"), 24, "runtime V2 canonical parquet count")
    bridge = document.get("algorithmic_provenance", {}).get("source_to_parquet_bridge", {})
    _equal(bridge.get("surviving_source_snapshot_count"), 6, "runtime V2 surviving source count")
    _equal(bridge.get("unavailable_historical_replica_count"), 18, "runtime V2 unavailable source count")
    _equal(statistics.get("selected_upper_bound"), "q99", "runtime V2 upper bound")
    return identity


def _verify_dataset(path: Path) -> dict[str, Any]:
    entries = tree_entries(path, "payload/lerobot_train")
    _equal(len(entries), 85, "FULL24 dataset regular file count")
    _equal(sum(int(item["bytes"]) for item in entries), 919661, "FULL24 dataset byte count")
    _equal(tree_digest(entries), FULL24_DATASET_TREE_SHA256, "FULL24 dataset tree")
    meta = path / "meta"
    split = read_json(meta / "stage22_split_lock.json")
    _equal(split.get("selected_split"), "train", "FULL24 split")
    _equal(split.get("dataset_role"), "TRAIN", "FULL24 role")
    _equal(split.get("validation_target_ids"), [], "FULL24 validation exclusion")
    _equal(split.get("final_test_target_ids"), [], "FULL24 holdout exclusion")
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _equal(len(episodes), 24, "FULL24 episode count")
    for index, episode in enumerate(episodes):
        target = TARGET_ORDER[index // 6]
        expected_id = f"stage22_expert_{target}:expert-{index % 6:04d}"
        _equal(episode.get("episode_index"), index, f"FULL24 episode index {index}")
        _equal(episode.get("length"), HORIZON, f"FULL24 episode length {index}")
        _equal(episode.get("split"), "train", f"FULL24 episode split {index}")
        _equal(episode.get("source_episode_id"), expected_id, f"FULL24 source episode {index}")
    return {
        "path": str(path),
        "tree_sha256": FULL24_DATASET_TREE_SHA256,
        "regular_file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "window_count": len(episodes),
        "frame_count": 960,
        "split_lock_sha256": sha256_file(meta / "stage22_split_lock.json"),
        "episodes_sha256": sha256_file(meta / "episodes.jsonl"),
    }


def _verify_checkpoints(paths: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, key in {
        "BASE": "base_checkpoint",
        "STRUCTURED@1000": "structured_1000_checkpoint",
        "FULL24@1000": "full24_1000_checkpoint",
        "FULL24@2000": "full24_2000_checkpoint",
    }.items():
        expected = CHECKPOINT_EXPECTATIONS[label]
        entries = tree_entries(Path(paths[key]), expected["checkpoint_tree_prefix"])
        checkpoint_tree = tree_digest(entries)
        _equal(checkpoint_tree, expected["checkpoint_tree_sha256"], f"{label} checkpoint tree")
        if label != "BASE":
            _equal(len(entries), 19, f"{label} checkpoint file count")
            _require(sum(int(item["bytes"]) for item in entries) > 0, f"{label} checkpoint is empty")
        training_return_digest = None
        if expected["training_return_digest_sha256"] is not None:
            training_entries = relabel_tree_namespace(
                entries,
                expected["checkpoint_tree_prefix"],
                expected["training_return_digest_prefix"],
            )
            training_return_digest = tree_digest(training_entries)
            _equal(
                training_return_digest,
                expected["training_return_digest_sha256"],
                f"{label} training-return digest",
            )
        result[label] = {
            "path": paths[key],
            "checkpoint_tree_sha256": checkpoint_tree,
            "checkpoint_tree_prefix": expected["checkpoint_tree_prefix"],
            "training_return_digest_sha256": training_return_digest,
            "training_return_digest_prefix": expected["training_return_digest_prefix"],
            "file_count": len(entries),
            "total_bytes": sum(int(item["bytes"]) for item in entries),
        }
    return result


def _verify_snapshots(snapshot_root: Path, expected_hashes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Read only the 24 accepted TRAIN snapshot descriptors in frozen order."""

    regular_directory(snapshot_root, "TRAIN snapshot root")
    records: list[dict[str, Any]] = []
    for index, window_id in enumerate(WINDOW_IDS):
        target = TARGET_ORDER[index // 6]
        snapshot_path = snapshot_root / f"stage22_expert_{target}" / f"expert-{index % 6:04d}.json"
        payload = read_json(snapshot_path)
        _equal(payload.get("schema"), "SameSceneExpertSnapshotV1", f"snapshot schema {window_id}")
        _equal(payload.get("example_id"), window_id, f"snapshot ID {window_id}")
        _equal(payload.get("raw_episode_id"), f"stage22_expert_{target}", f"snapshot episode {window_id}")
        _equal(payload.get("target_id"), target, f"snapshot target {window_id}")
        _equal(payload.get("target_position_world_m"), list(TARGET_POSITIONS[target]), f"snapshot target pose {window_id}")
        _equal(payload.get("source_frame"), SOURCE_FRAME, f"snapshot frame {window_id}")
        _equal(payload.get("tcp_semantic_id"), TCP_SEMANTIC_ID, f"snapshot TCP {window_id}")
        _equal(payload.get("target_excluded_from_model_input"), True, f"snapshot target exclusion {window_id}")
        observed_hash = sha256_file(snapshot_path)
        if expected_hashes is not None:
            _equal(observed_hash, expected_hashes[index], f"snapshot SHA {window_id}")
        state = payload.get("state_v1", {}).get("values")
        desired = payload.get("future_desired_eef_9d")
        gripper = payload.get("future_gripper")
        timestamps = payload.get("future_timestamps")
        _require(isinstance(state, list) and len(state) == STATE_DIM, f"snapshot state shape {window_id}")
        _require(isinstance(desired, list) and len(desired) == HORIZON and all(isinstance(row, list) and len(row) == 9 for row in desired), f"snapshot desired shape {window_id}")
        _require(isinstance(gripper, list) and len(gripper) == HORIZON and all(isinstance(row, list) and len(row) == 1 for row in gripper), f"snapshot gripper shape {window_id}")
        _require(isinstance(timestamps, list) and len(timestamps) == HORIZON, f"snapshot timestamps shape {window_id}")
        _require(all(math.isfinite(float(item)) for item in state + timestamps), f"snapshot state/timestamps nonfinite {window_id}")
        _require(all(math.isfinite(float(item)) for row in desired for item in row), f"snapshot desired nonfinite {window_id}")
        _require(all(math.isfinite(float(item)) for row in gripper for item in row), f"snapshot gripper nonfinite {window_id}")
        telemetry = payload.get("raw_telemetry", {})
        _require(isinstance(telemetry, Mapping), f"snapshot telemetry missing {window_id}")
        achieved = telemetry.get("achieved_position_world_m")
        _require(isinstance(achieved, list) and len(achieved) == 3, f"snapshot achieved world pose missing {window_id}")
        _require(all(math.isfinite(float(item)) for item in achieved), f"snapshot achieved world pose nonfinite {window_id}")
        _require(abs(float(timestamps[0]) - float(payload.get("observation_timestamp"))) <= 1e-7, f"snapshot observation timestamp drift {window_id}")
        _require(all(abs(float(timestamps[i + 1]) - float(timestamps[i]) - 0.05) <= 1e-8 for i in range(HORIZON - 1)), f"snapshot timing drift {window_id}")
        records.append({
            "window_id": window_id,
            "path": str(snapshot_path),
            "sha256": observed_hash,
            "target_id": target,
            "target_world_m": list(TARGET_POSITIONS[target]),
            "state_v1": state,
            "expert_physical_action": [list(row) + list(gripper[row_index]) for row_index, row in enumerate(desired)],
            "source_timestamps": timestamps,
            "observation_timestamp": float(payload["observation_timestamp"]),
            "anchor_world_m": [float(achieved[i]) - float(state[i]) for i in range(3)],
            "target_xyz_in_model_input": False,
            "target_excluded_from_model_input": True,
        })
    return records


def runtime_preflight(binding: Mapping[str, Any], mapping_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash every late-bound byte before importing the model runtime."""

    binding_summary = validate_runtime_binding(binding, mapping_summary, require_paths=True)
    paths = binding_summary["paths"]
    result: dict[str, Any] = {
        "binding": binding_summary,
        "repository": _verify_execution_checkout(str(binding["execution_head"])),
        "identities": {},
    }
    result["identities"]["golden_release_manifest"] = _verify_release_manifest(Path(paths["golden_release_manifest"]))
    result["identities"]["groot_runtime_manifest"] = _verify_groot_runtime_manifest(Path(paths["groot_runtime_manifest"]))
    result["identities"]["full24_manifest"] = _verify_full24_manifest(Path(paths["full24_manifest"]))
    result["identities"]["checkpoint_integrity_audit"] = _hash_identity(Path(paths["checkpoint_integrity_audit"]), INTEGRITY_AUDIT_SHA256, "checkpoint integrity/delta audit")
    result["identities"]["runtime_admission"] = _hash_identity(Path(paths["runtime_admission"]), RUNTIME_ADMISSION_SHA256, "runtime admission")
    result["identities"]["modality_config"] = _hash_identity(Path(paths["modality_config"]), MODALITY_SHA256, "modality config")
    result["identities"]["base_processor_config"] = _hash_identity(Path(paths["base_processor_config"]), BASE_PROCESSOR_CONFIG_SHA256, "BASE processor config")
    result["identities"]["base_processor_statistics"] = _hash_identity(Path(paths["base_processor_statistics"]), BASE_PROCESSOR_STATS_SHA256, "BASE processor statistics")
    result["identities"]["structured_1000_statistics"] = _verify_stats(Path(paths["structured_1000_statistics"]), HISTORICAL_STATS_SHA256, None, "historical StructuredAction statistics")
    result["identities"]["full24_runtime_statistics"] = _verify_runtime_stats(Path(paths["full24_runtime_statistics"]))
    result["dataset"] = _verify_dataset(Path(paths["full24_dataset"]))
    result["checkpoints"] = _verify_checkpoints(paths)
    cosmos_digest, cosmos_entries = cosmos_tree_digest(Path(paths["cosmos_root"]))
    _equal(cosmos_digest, COSMOS_TREE_SHA256, "Cosmos cache tree")
    cosmos_snapshot = Path(paths["cosmos_root"]) / "snapshots" / COSMOS_REVISION
    regular_directory(cosmos_snapshot, "Cosmos pinned snapshot")
    refs_main = Path(paths["cosmos_root"]) / "refs" / "main"
    regular_file(refs_main, "Cosmos refs/main")
    _equal(refs_main.read_text(encoding="utf-8").strip(), COSMOS_REVISION, "Cosmos refs/main")
    result["cosmos"] = {"tree_sha256": cosmos_digest, "entry_count": cosmos_entries, "revision": COSMOS_REVISION}

    source_root = Path(paths["groot_source"])
    regular_directory(source_root, "GR00T source snapshot")
    source_commit = None
    git_dir = source_root / ".git"
    if git_dir.exists() and git_dir.is_dir():
        import subprocess

        completed = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        _require(completed.returncode == 0, "GR00T source HEAD cannot be read")
        source_commit = completed.stdout.strip()
        _equal(source_commit, GROOT_SOURCE_COMMIT, "GR00T source HEAD")
    result["groot_source"] = {"path": str(source_root), "git_head": source_commit, "expected_commit": GROOT_SOURCE_COMMIT}
    snapshot_records, canonical_label_source = _canonical_parquet_snapshot_records(Path(paths["full24_dataset"]), Path(paths["snapshot_root"]), Path(paths["full24_runtime_statistics"]))
    result["canonical_label_source"] = canonical_label_source
    result["snapshots"] = snapshot_records
    _require(len(result["snapshots"]) == 24, "TRAIN24 snapshot count drift")
    result["status"] = "PASS"
    return result


def _canonical_parquet_snapshot_records(
    dataset_root: Path, snapshot_root: Path, stats_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize evaluator labels from the sealed FULL24 parquet source.

    Historical JSON descriptors are provenance witnesses only.  They are read
    when the six surviving train_a replicas exist, while all 24 numeric labels
    used by evaluation come directly from the already-qualified parquet tree.
    Missing historical replicas are never synthesized or used as execution
    input.
    """

    import numpy as np
    import pandas as pd

    regular_directory(dataset_root, "FULL24 canonical label dataset")
    regular_directory(snapshot_root, "historical snapshot witness root")
    stats = read_json(stats_path)
    provenance = stats.get("algorithmic_provenance", {})
    _require(isinstance(provenance, Mapping), "runtime V2 algorithmic provenance is missing")
    bridge = provenance.get("source_to_parquet_bridge", {})
    _require(isinstance(bridge, Mapping), "runtime V2 source bridge is missing")
    _equal(bridge.get("historical_source_snapshot_count"), 24, "runtime V2 historical source snapshot count")
    _equal(bridge.get("surviving_source_snapshot_count"), 6, "runtime V2 surviving source snapshot count")
    _equal(bridge.get("unavailable_historical_replica_count"), 18, "runtime V2 unavailable historical replica count")
    _equal(bridge.get("missing_replicas_are_not_execution_authority"), True, "runtime V2 missing-replica authority")
    bridge_rows = bridge.get("records")
    _require(isinstance(bridge_rows, list) and len(bridge_rows) == 6, "runtime V2 surviving source bridge records are not train_a x 6")
    bridge_by_index: dict[int, Mapping[str, Any]] = {}
    for row in bridge_rows:
        _require(isinstance(row, Mapping), "runtime V2 bridge record is not an object")
        snapshot_index = row.get("snapshot_index")
        _require(isinstance(snapshot_index, int) and 0 <= snapshot_index < 6, "runtime V2 bridge snapshot index drift")
        _require(snapshot_index not in bridge_by_index, "runtime V2 bridge snapshot index is duplicated")
        expected_sha = row.get("json_sha256")
        _require(isinstance(expected_sha, str) and len(expected_sha) == 64, "runtime V2 bridge JSON SHA is missing")
        bridge_by_index[snapshot_index] = row
    _equal(len(bridge_by_index), 6, "runtime V2 bridge record coverage")

    def rows_from_frame(frame: Any, column: str, width: int) -> np.ndarray:
        _require(column in frame.columns, f"canonical parquet column missing: {column}")
        try:
            rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[column]]
            values = np.stack(rows, axis=0)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"canonical parquet column is not numeric H40: {column}") from exc
        _require(values.shape == (HORIZON, width), f"canonical parquet shape drift for {column}: {values.shape}")
        _require(bool(np.isfinite(values).all()), f"canonical parquet column is non-finite: {column}")
        return values

    def existing_regular_file(path: Path) -> bool:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise AdapterError(f"historical snapshot locator cannot be inspected: {path}") from exc
        _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"historical snapshot witness is unsafe: {path}")
        return True

    data_root = dataset_root / "data"
    regular_directory(data_root, "FULL24 parquet data root")
    parquet_paths = sorted(data_root.glob("chunk-*/episode_*.parquet"), key=lambda candidate: candidate.as_posix())
    _equal(len(parquet_paths), 24, "canonical FULL24 parquet count")
    records: list[dict[str, Any]] = []
    surviving_json_count = 0
    unavailable_json_count = 0

    for index, parquet_path in enumerate(parquet_paths):
        regular_file(parquet_path, f"canonical parquet {index}")
        try:
            episode_index = int(parquet_path.stem.split("_")[-1])
        except ValueError as exc:
            raise AdapterError(f"canonical parquet filename is not indexed: {parquet_path}") from exc
        _equal(episode_index, index, f"canonical parquet episode index {index}")
        frame = pd.read_parquet(parquet_path, columns=["timestamp", "action", "observation.state"])
        _equal(len(frame), HORIZON, f"canonical parquet row count {index}")
        timestamps = np.asarray(frame["timestamp"].to_numpy(), dtype=np.float64)
        _require(timestamps.shape == (HORIZON,) and bool(np.isfinite(timestamps).all()), f"canonical parquet timestamps are not finite H40 at {index}")
        _require(float(np.max(np.abs(np.diff(timestamps) - 0.05))) <= 1e-9, f"canonical parquet timing drift at {index}")
        action = rows_from_frame(frame, "action", PHYSICAL_DIM)
        states = rows_from_frame(frame, "observation.state", STATE_DIM)
        _require(bool(np.array_equal(states, np.broadcast_to(states[0], states.shape))), f"canonical parquet observation state is not repeated at {index}")
        _require(float(np.linalg.norm(states[0, :3] - action[0, :3])) <= 1e-6, f"canonical parquet state/action anchor mismatch at {index}")
        _require(float(np.max(np.abs(action[:, 9] - 0.5))) <= 1e-6, f"canonical parquet action gripper drift at {index}")
        _require(float(np.max(np.abs(states[:, 9] - 0.5))) <= 1e-6, f"canonical parquet state gripper drift at {index}")

        target = TARGET_ORDER[index // 6]
        snapshot_index = index % 6
        window_id = f"stage22_expert_{target}:expert-{snapshot_index:04d}"
        target_world = np.asarray(TARGET_POSITIONS[target], dtype=np.float64)
        anchor = target_world - action[-1, :3]
        witness_path = snapshot_root / f"stage22_expert_{target}" / f"expert-{snapshot_index:04d}.json"
        witness: dict[str, Any]
        if target == TARGET_ORDER[0]:
            _require(existing_regular_file(witness_path), f"surviving train_a source snapshot is unavailable: {witness_path}")
            payload = read_json(witness_path)
            _equal(payload.get("schema"), "SameSceneExpertSnapshotV1", f"snapshot schema {window_id}")
            _equal(payload.get("example_id"), window_id, f"snapshot ID {window_id}")
            _equal(payload.get("raw_episode_id"), f"stage22_expert_{target}", f"snapshot episode {window_id}")
            _equal(payload.get("target_id"), target, f"snapshot target {window_id}")
            _equal(payload.get("target_position_world_m"), list(TARGET_POSITIONS[target]), f"snapshot target pose {window_id}")
            _equal(payload.get("source_frame"), SOURCE_FRAME, f"snapshot frame {window_id}")
            _equal(payload.get("tcp_semantic_id"), TCP_SEMANTIC_ID, f"snapshot TCP {window_id}")
            _equal(payload.get("target_excluded_from_model_input"), True, f"snapshot target exclusion {window_id}")
            witness_state = np.asarray(payload.get("state_v1", {}).get("values"), dtype=np.float32).astype(np.float64)
            witness_action = np.asarray(payload.get("future_desired_eef_9d"), dtype=np.float32).astype(np.float64)
            witness_gripper = np.asarray(payload.get("future_gripper"), dtype=np.float32).reshape(-1, 1).astype(np.float64)
            witness_timestamps = np.asarray(payload.get("future_timestamps"), dtype=np.float64)
            _require(witness_state.shape == (STATE_DIM,) and witness_action.shape == (HORIZON, 9) and witness_gripper.shape == (HORIZON, 1) and witness_timestamps.shape == (HORIZON,), f"historical witness shape drift {window_id}")
            bridge_row = bridge_by_index[snapshot_index]
            witness_sha = sha256_file(witness_path)
            _equal(witness_sha, bridge_row["json_sha256"], f"historical witness SHA {window_id}")
            bridge_errors = {
                "state_v1": float(np.max(np.abs(witness_state - states[0]))),
                "desired_eef_9d": float(np.max(np.abs(witness_action - action[:, :9]))),
                "gripper": float(np.max(np.abs(witness_gripper - action[:, 9:10]))),
                "timestamps": float(np.max(np.abs(witness_timestamps - timestamps))),
            }
            _require(all(error == 0.0 for error in bridge_errors.values()), f"historical witness float32 bridge drift {window_id}: {bridge_errors}")
            telemetry = payload.get("raw_telemetry", {})
            _require(isinstance(telemetry, Mapping), f"historical witness telemetry missing {window_id}")
            achieved = np.asarray(telemetry.get("achieved_position_world_m"), dtype=np.float64)
            _require(achieved.shape == (3,) and bool(np.isfinite(achieved).all()), f"historical witness achieved pose missing {window_id}")
            witness_anchor = achieved - witness_state[:3]
            anchor_error = float(np.max(np.abs(witness_anchor - anchor)))
            _require(anchor_error <= 1e-6, f"historical witness anchor mismatch {window_id}: {anchor_error}")
            _require(abs(float(witness_timestamps[0]) - float(payload.get("observation_timestamp"))) <= 1e-7, f"historical witness observation timestamp drift {window_id}")
            witness = {
                "status": "PASS",
                "execution_authority": False,
                "path": str(witness_path),
                "sha256": witness_sha,
                "float32_bridge_max_abs_error": bridge_errors,
                "anchor_world_max_abs_error_m": anchor_error,
            }
            surviving_json_count += 1
        else:
            if existing_regular_file(witness_path):
                witness = {
                    "status": "PRESENT_NOT_USED",
                    "execution_authority": False,
                    "path": str(witness_path),
                    "sha256": sha256_file(witness_path),
                }
            else:
                witness = {
                    "status": "UNAVAILABLE",
                    "execution_authority": False,
                    "path": str(witness_path),
                }
                unavailable_json_count += 1

        parquet_sha = sha256_file(parquet_path)
        records.append(
            {
                "window_id": window_id,
                "path": str(parquet_path),
                "sha256": parquet_sha,
                "source_artifact_kind": CANONICAL_LABEL_SOURCE,
                "canonical_parquet_path": str(parquet_path),
                "canonical_parquet_sha256": parquet_sha,
                "historical_source_json": witness,
                "target_id": target,
                "target_world_m": list(TARGET_POSITIONS[target]),
                "state_v1": states[0].tolist(),
                "expert_physical_action": action.tolist(),
                "source_timestamps": timestamps.tolist(),
                "observation_timestamp": float(timestamps[0]),
                "anchor_world_m": anchor.tolist(),
                "target_xyz_in_model_input": False,
                "target_excluded_from_model_input": True,
            }
        )

    _equal(surviving_json_count, 6, "surviving historical source JSON count")
    _equal(unavailable_json_count, 18, "unavailable historical source JSON count")
    summary = {
        "kind": CANONICAL_LABEL_SOURCE,
        "root": str(dataset_root),
        "tree_sha256": FULL24_DATASET_TREE_SHA256,
        "parquet_files_read": len(records),
        "label_records": len(records),
        "label_materialization": "canonical_parquet_action_and_observation_state",
        "anchor_derivation": "target_world_m_minus_canonical_parquet_action_pose39",
        "historical_source_json_root": str(snapshot_root),
        "surviving_historical_source_json_count": surviving_json_count,
        "unavailable_historical_replica_count": unavailable_json_count,
        "historical_source_json_execution_authority": False,
        "missing_replicas_are_not_execution_authority": True,
        "target_xyz_in_model_input": False,
    }
    return records, summary

def _snapshot_expectations_from_stats(path: Path) -> list[str] | None:
    document = read_json(path)
    candidates = _find_nested(document, "source_snapshots")
    for candidate in candidates:
        rows = candidate.get("ordered_records")
        if isinstance(rows, list) and len(rows) == 24:
            hashes = [row.get("json_sha256") for row in rows if isinstance(row, Mapping)]
            if len(hashes) == 24 and all(isinstance(item, str) and len(item) == 64 for item in hashes):
                return [str(item) for item in hashes]
    return None


def _verify_execution_checkout(expected_head: str) -> dict[str, Any]:
    """Bind the runtime process to the final external execution-head seal."""

    import subprocess

    _require(isinstance(expected_head, str) and len(expected_head) == 40 and all(char in "0123456789abcdef" for char in expected_head), "execution head is not an exact commit SHA")
    observed_head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    _require(observed_head.returncode == 0, "cannot read Stage45C execution HEAD")
    head = observed_head.stdout.strip()
    _equal(head, expected_head, "Stage45C execution HEAD")
    observed_branch = subprocess.run(
        ["git", "-C", str(REPO), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    _equal(observed_branch.returncode, 0, "Stage45C branch query")
    _equal(observed_branch.stdout.strip(), "stage45c/b2-checkpoint-train24-evaluation", "Stage45C execution branch")
    observed_status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain=v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    _equal(observed_status.returncode, 0, "Stage45C worktree query")
    _equal(observed_status.stdout, "", "Stage45C worktree cleanliness")
    return {
        "protocol_source_head": PROTOCOL_SOURCE_HEAD,
        "execution_head": head,
        "branch": observed_branch.stdout.strip(),
        "worktree_clean": True,
    }


def _plan(mapping_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": ADAPTER_SCHEMA,
        "stage": STAGE,
        "mapping_provenance_gate": MAPPING_PROVENANCE_GATE,
        "execution_authority_gate": None,
        "execution_authority_gate_source": "EXACT_EXTERNAL_RUNTIME_BINDING",
        "status": "READY_NOT_EXECUTED",
        "protocol_sha256": PROTOCOL_SHA256,
        "mapping_sha256": mapping_summary["mapping_sha256"],
        "protocol_source_head": PROTOCOL_SOURCE_HEAD,
        "execution_head": EXTERNAL_EXECUTION_HEAD,
        "checkpoint_identity_laws": checkpoint_identity_laws(),
        "checkpoint_identity_namespaces": checkpoint_identity_namespaces(),
        "checkpoint_order": list(CHECKPOINT_ORDER),
        "windows_per_checkpoint": 24,
        "planned_checkpoint_window_forwards": 96,
        "planned_action_head_dit_forwards": 384,
        "sampler": {
            "development_seed": DEVELOPMENT_SEED,
            "noise_shape": list(RAW_SHAPE),
            "same_noise_across_checkpoints": True,
            "steps": SAMPLER_STEPS,
            "dt": SAMPLER_DT,
            "timestep_ids": list(TIMESTEP_IDS),
            "official_public_sampler_called": False,
        },
        "model_input_contract": {
            "keys": ["body_rgb", "wrist_rgb", "state"],
            "body_rgb_shape": list(RGB_SHAPE),
            "wrist_rgb_shape": list(RGB_SHAPE),
            "state_v1_shape": [STATE_DIM],
            "language": LANGUAGE,
            "target_xyz_in_model_input": False,
        },
        "execution_boundary": {
            "model_construction": False,
            "model_forward": False,
            "gpu": False,
            "h200": False,
            "training": False,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "checkpoint_write": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_steps": 0,
            "reference_updates": 0,
            "simulator": False,
            "hardware": False,
            "pbs_submissions_by_corelab": 0,
        },
        "historical_comparator": {
            "checkpoint_tree_sha256": STRUCTURED_1000_TREE_SHA256,
            "stats_sha256": HISTORICAL_STATS_SHA256,
            "runtime_materialization_status": "RECOVERED_CANONICAL_READ_ONLY",
            "recovery_evidence_sha256": "c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9",
            "materialization_path": None,
            "substitution_allowed": False,
        },
        "return_token": "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_READY",
    }


def _load_stage45c_structured_model(checkpoint: Path, device: str, nested_cache: Mapping[str, Any]) -> Any:
    """Construct one evaluator model with the explicit nested Cosmos contract."""

    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401 - AutoModel registration side effect
    from transformers import AutoModel

    hub_root = Path(str(nested_cache["hf_hub"]))
    nested_kwargs = nested_cosmos_model_loading_kwargs(hub_root)
    _equal(nested_kwargs, nested_cache["nested_model_loading_kwargs"], "nested Cosmos loading kwargs")
    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        transformers_loading_kwargs=nested_kwargs,
    )
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model


def _load_torch_runtime() -> tuple[Any, Any]:
    """Import Torch only after all identity gates and cache bindings have passed."""

    import torch

    return torch, _load_stage45c_structured_model


def _make_noise_bank(torch: Any) -> list[Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DEVELOPMENT_SEED)
    bank: list[Any] = []
    for _ in WINDOW_IDS:
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32)
        _require(tuple(int(value) for value in noise.shape) == RAW_SHAPE, "sampler noise shape drift")
        _require(bool(torch.isfinite(noise).all().item()), "sampler noise is nonfinite")
        bank.append(noise.contiguous())
    return bank


def _prepare_train24_windows(processor: Any, embodiment: Any, modality_configs: Mapping[str, Any], dataset_dir: Path, snapshot_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use the official loader while constructing inference-only inputs."""

    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType, VLAStepData

    loader = LeRobotEpisodeLoader(dataset_dir, modality_configs)
    _equal(len(loader), 24, "TRAIN24 loader length")
    _equal(list(loader.episode_lengths), [HORIZON] * 24, "TRAIN24 episode lengths")
    windows: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshot_records):
        episode = loader[index]
        step_data = extract_step_data(episode, 0, modality_configs, embodiment, allow_padding=False)
        inference_step = VLAStepData(
            images=step_data.images,
            states=step_data.states,
            actions={},
            text=LANGUAGE,
            embodiment=embodiment,
        )
        processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
        batch = processor.collator([processed])
        model_inputs = batch.get("inputs", {})
        _require(isinstance(model_inputs, Mapping), f"model inputs are not a mapping at {index}")
        _require("action" not in model_inputs and "action_mask" not in model_inputs, f"expert action leaked into inference input at {index}")
        for forbidden in ("target_position_world_m", "target_xyz", "target_world_m"):
            _require(forbidden not in model_inputs, f"numeric target leaked into model input at {index}")
        # The collator owns tensor layout; this check only records the logical
        # observation ABI and avoids making a second numeric target channel.
        state = model_inputs.get("state")
        if state is not None:
            _require(bool(torch.isfinite(state).all().item()), f"processed state is nonfinite at {index}")
        windows.append({
            "window_id": snapshot["window_id"],
            "model_inputs": model_inputs,
            "snapshot": dict(snapshot),
            "logical_model_input_keys": ["body_rgb", "wrist_rgb", "state"],
            "target_xyz_in_model_input": False,
            "loader_index": index,
        })
    return windows


def _offline_step(action_head: Any, processed_backbone: Any, action_input: Any, state_features: Any, actions: Any, timestep: int) -> Any:
    """The accepted Stage38 source-mapped action-head graph, copied verbatim."""

    import torch

    vl_embeds = processed_backbone.backbone_features
    device = vl_embeds.device
    batch_size = int(actions.shape[0])
    timesteps = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
    action_features = action_head.action_encoder(actions, timesteps, action_input.embodiment_id)
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    if action_head.config.use_alternate_vl_dit:
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps,
            image_mask=processed_backbone.image_mask,
            backbone_attention_mask=processed_backbone.backbone_attention_mask,
        )
    else:
        model_output = action_head.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds, timestep=timesteps)
    pred = action_head.action_decoder(model_output, action_input.embodiment_id)
    velocity = pred[:, -actions.shape[1] :]
    _require(tuple(int(value) for value in velocity.shape) == RAW_SHAPE, "action-head velocity shape drift")
    _require(bool(torch.isfinite(velocity).all().item()), "action-head velocity is nonfinite")
    return velocity


def _prepare_model_conditioning(model: Any, model_inputs: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    import torch

    backbone_inputs, action_inputs = model.prepare_input(dict(model_inputs))
    with torch.inference_mode():
        backbone_output = model.backbone(backbone_inputs)
        processed = model.action_head.process_backbone_output(backbone_output)
        _equal(int(action_inputs.state.shape[1]), int(model.action_head.config.state_history_length), "state history length")
        state = action_inputs.state.view(action_inputs.state.shape[0], 1, -1)
        state_features = model.action_head.state_encoder(state, action_inputs.embodiment_id)
    return backbone_inputs, action_inputs, processed, state_features


def _sample_one(model: Any, model_inputs: Mapping[str, Any], noise: Any, torch: Any) -> tuple[Any, int]:
    _, action_input, processed, state_features = _prepare_model_conditioning(model, model_inputs)
    current = noise.to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
    action_head_calls = 0
    for timestep in TIMESTEP_IDS:
        velocity = _offline_step(model.action_head, processed, action_input, state_features, current, timestep)
        current = current + SAMPLER_DT * velocity
        action_head_calls += 1
    _require(action_head_calls == SAMPLER_STEPS, "sampler step count drift")
    raw = current.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _require(tuple(int(value) for value in raw.shape) == RAW_SHAPE, "raw universal action shape drift")
    _require(bool(torch.isfinite(raw).all().item()), "raw universal action is nonfinite")
    return raw, action_head_calls


def _decode_v2(raw: Any, state: Any, stats_path: Path, timestamps: Any, observation_timestamp: float, chunk_id: str) -> tuple[Any, dict[str, Any]]:
    """Apply the unchanged Stage40 inverse-normalization/materialization law to V2 stats."""

    import numpy as np
    from go2arx5_vla.structured_action import decode_structured_action, materialize_structured_action, structured_action_contract_summary, structured_action_mask
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows

    document = read_json(stats_path)
    statistics = document.get("statistics", document)
    lower = np.asarray(document.get("selected_lower", statistics.get("selected_lower")), dtype=np.float64)
    upper = np.asarray(document.get("selected_upper", statistics.get("selected_upper")), dtype=np.float64)
    mask = np.asarray(document.get("active_mask", statistics.get("active_mask")), dtype=np.float64)
    _require(lower.shape == (UNIVERSAL_DIM,) and upper.shape == (UNIVERSAL_DIM,), "V2 selected bounds must be [132]")
    expected_mask = structured_action_mask(TIMING_MODE)
    _require(mask.shape == expected_mask.shape and np.array_equal(mask, expected_mask), "V2 active mask drift")
    raw_np = np.asarray(raw, dtype=np.float64)
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    _require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), "V2 raw shape/finite contract")
    _require(state_np.shape == (STATE_DIM,) and np.isfinite(state_np).all(), "V2 state shape/finite contract")
    normalized = np.clip(raw_np[0], -1.0, 1.0)
    values = (normalized + 1.0) * 0.5 * (upper[None, :] - lower[None, :]) + lower[None, :]
    values[mask == 0.0] = 0.0
    _require(np.isfinite(values).all(), "V2 inverse-normalized values are nonfinite")
    intent = decode_structured_action(
        values,
        anchor_state_v1=state_np.astype(np.float64),
        source_timestamps=np.asarray(timestamps, dtype=np.float64),
        source_duration_s=1.9500000000000002,
        timing_mode=TIMING_MODE,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        observation_timestamp=float(observation_timestamp),
        intent_id=chunk_id,
        provenance={
            "stage45c_decoder_bridge": "Stage40_inverse_normalization_and_materialization",
            "structured_stats_sha256": FULL24_STATS_SHA256,
            "statistics_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "no_dense_decode_fallback": True,
        },
    )
    chunk = materialize_structured_action(values, intent, source_timestamps=np.asarray(timestamps, dtype=np.float64))
    rows = rotation_matrix_to_rot6d_rows(quat_to_matrix(np.asarray(chunk.eef_orientation, dtype=np.float64)))
    physical = np.concatenate([np.asarray(chunk.eef_position, dtype=np.float64), rows, np.asarray(chunk.gripper, dtype=np.float64).reshape(HORIZON, 1)], axis=1)
    _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "V2 decoded physical shape/finite contract")
    return physical, {
        "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
        "decoder_bridge": "V2_stats_with_Stage40_inverse_normalization_and_materialization",
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "contract": structured_action_contract_summary(TIMING_MODE),
        "stats_sha256": FULL24_STATS_SHA256,
        "statistics_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
        "raw_shape": list(RAW_SHAPE),
        "physical_shape": list(PHYSICAL_SHAPE),
        "raw_array_sha256": sha256_bytes(raw_np.astype(np.float32, copy=False).tobytes()),
    }


def _decode_base(raw: Any, state: Any, processor: Any, embodiment: Any) -> tuple[Any, dict[str, Any]]:
    """Decode BASE with the exact Golden processor statistics contract."""

    import numpy as np

    raw_np = np.asarray(raw, dtype=np.float32)
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    _require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), "BASE raw shape/finite contract")
    _require(state_np.shape == (STATE_DIM,) and np.isfinite(state_np).all(), "BASE state shape/finite contract")
    decoded_groups = processor.decode_action(
        raw_np,
        embodiment,
        {
            "eef_9d": state_np[None, None, :9],
            "gripper_position": state_np[None, None, 9:],
        },
    )
    physical = np.concatenate(
        [np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])],
        axis=-1,
    ).astype(np.float64)
    _require(physical.shape == (1, HORIZON, PHYSICAL_DIM) and np.isfinite(physical).all(), "BASE decoded physical shape/finite contract")
    return physical[0], {
        "decoder": "Gr00tN1d7Processor.decode_action",
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "decoder_bridge": "Golden_Release_BASE_processor_statistics",
        "stats_sha256": BASE_PROCESSOR_STATS_SHA256,
        "raw_shape": list(RAW_SHAPE),
        "physical_shape": list(PHYSICAL_SHAPE),
    }


def _evaluation_row(raw: Any, physical: Any, details: Mapping[str, Any], window: Mapping[str, Any], checkpoint: Mapping[str, Any], model_runtime_sha256: str, raw_artifact_path: str, raw_artifact_sha256: str) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import array_sha256, comparison_metrics

    snapshot = window["snapshot"]
    raw_np = np.asarray(raw, dtype=np.float32)
    physical_np = np.asarray(physical, dtype=np.float64)
    expert = np.asarray(snapshot["expert_physical_action"], dtype=np.float64)
    metrics = comparison_metrics(physical_np, expert, snapshot["anchor_world_m"], snapshot["target_world_m"], prefix=f"{window['window_id']}.prediction")
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "window_id": window["window_id"],
        "raw_prediction_artifact": {
            "path": raw_artifact_path,
            "sha256": raw_artifact_sha256,
            "array_sha256": array_sha256(raw_np),
            "shape": list(raw_np.shape),
        },
        "checkpoint": {
            "label": checkpoint["label"],
            "sha256": checkpoint["checkpoint_tree_sha256"],
            "checkpoint_tree_sha256": checkpoint["checkpoint_tree_sha256"],
            "training_return_digest_sha256": checkpoint.get("training_return_digest_sha256"),
        },
        "model": {
            "label": checkpoint["label"],
            "sha256": checkpoint["checkpoint_tree_sha256"],
            "checkpoint_tree_sha256": checkpoint["checkpoint_tree_sha256"],
            "training_return_digest_sha256": checkpoint.get("training_return_digest_sha256"),
        },
        "model_runtime_identity": {"sha256": model_runtime_sha256, "label": "Golden_Release_V2_GR00T_runtime"},
        "decoded_action": {
            "array_sha256": array_sha256(physical_np),
            "shape": list(physical_np.shape),
            "decoder_details": dict(details),
        },
        "metrics": metrics,
        "safety_gate_decision": None,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "target_xyz_in_model_input": False,
        "finite": True,
    }


def _run_execution(binding: Mapping[str, Any], mapping_summary: Mapping[str, Any], protocol_path: Path, mapping_path: Path, device: str) -> dict[str, Any]:
    """Run one admitted H200 process; never called by Corelab during preparation."""

    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required for H200 execution")
    _require(str(device).startswith("cuda"), "H200 execution requires a CUDA device")
    preflight = runtime_preflight(binding, mapping_summary)
    paths = preflight["binding"]["paths"]
    output_root = Path(paths["output_root"])
    _require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    nested_cosmos_cache = _bind_nested_cosmos_hf_cache(
        output_root,
        Path(paths["cosmos_root"]),
        verified_tree_sha256=preflight["cosmos"]["tree_sha256"],
        verified_entry_count=preflight["cosmos"]["entry_count"],
    )
    write_json(output_root / "stage45c_runtime_preflight.json", preflight)
    write_json(output_root / "stage45c_nested_cosmos_runtime_binding.json", nested_cosmos_cache)
    protocol_sha = sha256_file(protocol_path)
    mapping_sha = sha256_file(mapping_path)
    torch, load_structured_model = _load_torch_runtime()
    _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted H200 job")
    gpu_name = torch.cuda.get_device_name(torch.device(device))
    _require("H200" in str(gpu_name), f"the admitted GPU is not H200: {gpu_name}")

    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    snapshot_records = preflight["snapshots"]
    # The preflight JSON contains enough immutable metadata for the model
    # stage; re-read dataclasses only after all hashes and exclusion checks.
    runtime_processor, embodiment, modality_configs = _build_runtime_processor(
        groot_source=Path(paths["groot_source"]),
        base_checkpoint=Path(paths["base_checkpoint"]),
        dataset_dir=Path(paths["full24_dataset"]),
        modality_path=Path(paths["modality_config"]),
        cosmos_snapshot=Path(paths["cosmos_root"]) / "snapshots" / COSMOS_REVISION,
    )
    windows = _prepare_train24_windows(runtime_processor, embodiment, modality_configs, Path(paths["full24_dataset"]), snapshot_records)
    noise_bank = _make_noise_bank(torch)
    noise_hashes = [sha256_bytes(noise.detach().cpu().contiguous().numpy().tobytes()) for noise in noise_bank]
    write_json(output_root / "sampler_noise_manifest.json", {
        "schema": "CORELAB_STAGE45C_SAMPLER_NOISE_BANK_V1",
        "development_seed": DEVELOPMENT_SEED,
        "noise_shape": list(RAW_SHAPE),
        "window_ids": list(WINDOW_IDS),
        "noise_sha256_by_window": noise_hashes,
        "same_noise_across_checkpoints": True,
    })
    raw_manifest = output_root / "raw_predictions.jsonl"
    checkpoint_rows: list[dict[str, Any]] = []
    all_action_head_calls = 0
    all_window_forwards = 0
    runtime_model_sha256 = RELEASE_V2_SHA256
    checkpoint_paths = {
        "BASE": paths["base_checkpoint"],
        "STRUCTURED@1000": paths["structured_1000_checkpoint"],
        "FULL24@1000": paths["full24_1000_checkpoint"],
        "FULL24@2000": paths["full24_2000_checkpoint"],
    }
    stats_paths = {
        "BASE": paths["base_processor_statistics"],
        "STRUCTURED@1000": paths["structured_1000_statistics"],
        "FULL24@1000": paths["full24_runtime_statistics"],
        "FULL24@2000": paths["full24_runtime_statistics"],
    }
    stats_kinds = {label: CHECKPOINT_EXPECTATIONS[label]["decode_kind"] for label in CHECKPOINT_ORDER}
    for label in CHECKPOINT_ORDER:
        checkpoint_identity = {
            "label": label,
            "checkpoint_tree_sha256": CHECKPOINT_EXPECTATIONS[label]["checkpoint_tree_sha256"],
            "checkpoint_tree_prefix": CHECKPOINT_EXPECTATIONS[label]["checkpoint_tree_prefix"],
            "training_return_digest_sha256": CHECKPOINT_EXPECTATIONS[label]["training_return_digest_sha256"],
            "training_return_digest_prefix": CHECKPOINT_EXPECTATIONS[label]["training_return_digest_prefix"],
        }
        model = load_structured_model(Path(checkpoint_paths[label]), device, nested_cosmos_cache)
        action_head_call_counter = {"count": 0}

        def count_action_head(*_args: Any, **_kwargs: Any) -> None:
            action_head_call_counter["count"] += 1

        hook_handle = model.action_head.model.register_forward_hook(count_action_head)
        rows: list[dict[str, Any]] = []
        try:
            for index, window in enumerate(windows):
                raw, sampler_calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
                raw_path = output_root / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                import numpy as np

                np.save(raw_path, raw.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
                raw_file_sha = sha256_file(raw_path)
                raw_array_sha = sha256_bytes(raw.detach().cpu().contiguous().numpy().tobytes())
                append_jsonl(raw_manifest, {
                    "checkpoint": checkpoint_identity,
                    "window_id": window["window_id"],
                    "path": str(raw_path.relative_to(output_root)),
                    "file_sha256": raw_file_sha,
                    "array_sha256": raw_array_sha,
                    "shape": list(RAW_SHAPE),
                    "noise_sha256": noise_hashes[index],
                    "sampler_calls_observed": sampler_calls,
                    "decoded": False,
                })
                if stats_kinds[label] == "HISTORICAL_STRUCTURED_ACTION_STATISTICS":
                    from go2arx5_vla.evaluation.checkpoint_eval import evaluate_raw_prediction

                    snapshot = window["snapshot"]
                    evaluator_record = {
                        "window_id": window["window_id"],
                        "checkpoint_label": label,
                        "checkpoint_sha256": checkpoint_identity["checkpoint_tree_sha256"],
                        "model_label": label,
                        "model_sha256": checkpoint_identity["checkpoint_tree_sha256"],
                        "raw_prediction_artifact_path": str(raw_path.relative_to(output_root)),
                        "raw_prediction_artifact_sha256": raw_file_sha,
                        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
                        "target_xyz_in_model_input": False,
                        "raw_universal_action": raw.detach().cpu().numpy(),
                        "state_v1": snapshot["state_v1"],
                        "expert_physical_action": snapshot["expert_physical_action"],
                        "anchor_world_m": snapshot["anchor_world_m"],
                        "target_world_m": snapshot["target_world_m"],
                        "source_timestamps": snapshot["source_timestamps"],
                        "observation_timestamp": snapshot["observation_timestamp"],
                    }
                    row = evaluate_raw_prediction(evaluator_record, stats_path=Path(stats_paths[label]))
                    row["model_runtime_identity"] = {"sha256": runtime_model_sha256, "label": "Golden_Release_V2_GR00T_runtime"}
                elif stats_kinds[label] == "BASE_PROCESSOR_STATISTICS":
                    physical, decoder_details = _decode_base(
                        raw.detach().cpu().numpy(),
                        window["snapshot"]["state_v1"],
                        runtime_processor,
                        embodiment,
                    )
                    row = _evaluation_row(raw.detach().cpu().numpy(), physical, decoder_details, window, checkpoint_identity, runtime_model_sha256, str(raw_path.relative_to(output_root)), raw_file_sha)
                else:
                    physical, decoder_details = _decode_v2(raw.detach().cpu().numpy(), window["snapshot"]["state_v1"], Path(stats_paths[label]), window["snapshot"]["source_timestamps"], window["snapshot"]["observation_timestamp"], window["window_id"])
                    row = _evaluation_row(raw.detach().cpu().numpy(), physical, decoder_details, window, checkpoint_identity, runtime_model_sha256, str(raw_path.relative_to(output_root)), raw_file_sha)
                row["noise_sha256"] = noise_hashes[index]
                row["sampler"] = {
                    "name": "source-mapped-four-step-Euler-action-head-graph",
                    "timestep_ids": list(TIMESTEP_IDS),
                    "steps_observed": sampler_calls,
                    "official_public_sampler_called": False,
                }
                rows.append(row)
                append_jsonl(raw_manifest, {
                    "checkpoint": checkpoint_identity,
                    "window_id": window["window_id"],
                    "decoded": True,
                    "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                    "metric_semantics_source": "ACCEPTED_STAGE44R",
                })
        finally:
            hook_handle.remove()
        expected_action_head_calls = 24 * SAMPLER_STEPS
        _equal(action_head_call_counter["count"], expected_action_head_calls, f"observed action-head forward count {label}")
        _equal(len(rows), 24, f"evaluation row count {label}")
        aggregate = aggregate_evaluation_rows(rows)
        aggregate["runtime_observed"] = {
            "checkpoint_window_forwards": len(rows),
            "action_head_dit_forward_calls": action_head_call_counter["count"],
            "raw_prediction_count": len(rows),
            "reference_updates": 0,
            "wbc_steps": 0,
        }
        write_json(output_root / f"aggregate_{label.replace('@', '_at_')}.json", aggregate)
        checkpoint_rows.append({
            "checkpoint": label,
            "checkpoint_tree_sha256": checkpoint_identity["checkpoint_tree_sha256"],
            "checkpoint_tree_prefix": checkpoint_identity["checkpoint_tree_prefix"],
            "training_return_digest_sha256": checkpoint_identity["training_return_digest_sha256"],
            "training_return_digest_prefix": checkpoint_identity["training_return_digest_prefix"],
            "aggregate": aggregate,
            "action_head_dit_forward_calls": action_head_call_counter["count"],
            "checkpoint_window_forwards": len(rows),
        })
        all_action_head_calls += action_head_call_counter["count"]
        all_window_forwards += len(rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seal = {
        "schema": "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_SEAL_V1",
        "status": "PASS",
        "token": "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_READY",
        "mapping_provenance_gate": MAPPING_PROVENANCE_GATE,
        "execution_authority_gate": binding.get("pi_gate"),
        "protocol_sha256": protocol_sha,
        "mapping_sha256": mapping_sha,
        "protocol_source_head": PROTOCOL_SOURCE_HEAD,
        "execution_head": preflight["repository"]["execution_head"],
        "binding_sha256": canonical_sha256(binding),
        "nested_cosmos_cache": nested_cosmos_cache,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "checkpoint_identity_laws": checkpoint_identity_laws(),
        "device": str(device),
        "checkpoint_identity_namespaces": checkpoint_identity_namespaces(),
        "gpu_name": str(gpu_name),
        "checkpoint_order": list(CHECKPOINT_ORDER),
        "checkpoint_results": checkpoint_rows,
        "runtime_counters": {
            "checkpoint_window_forwards": all_window_forwards,
            "action_head_dit_forward_calls": all_action_head_calls,
            "raw_prediction_count": all_window_forwards,
            "reference_updates": 0,
            "wbc_steps": 0,
        },
        "evidence_artifacts": {
            "raw_predictions_jsonl_sha256": sha256_file(raw_manifest),
            "aggregate_sha256_by_checkpoint": {
                item["checkpoint"]: sha256_file(
                    output_root / f"aggregate_{item['checkpoint'].replace('@', '_at_')}.json"
                )
                for item in checkpoint_rows
            },
        },
        "sampler_noise": {
            "development_seed": DEVELOPMENT_SEED,
            "noise_shape": list(RAW_SHAPE),
            "same_noise_across_checkpoints": True,
            "noise_manifest_sha256": sha256_file(output_root / "sampler_noise_manifest.json"),
        },
        "scientific_semantics": {
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
            "metric_source_sha256": METRICS_SOURCE_SHA256,
            "target_xyz_in_model_input": False,
            "target_threshold": None,
            "safety_gate_executed": False,
        },
        "execution_boundary": {
            "model_construction": True,
            "model_forward": True,
            "gpu": True,
            "h200": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_write": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_steps": 0,
            "reference_updates": 0,
            "simulator": False,
            "hardware": False,
            "pbs_submissions_by_corelab": 0,
        },
    }
    write_json(output_root / "stage45c_runtime_adapter_seal.json", seal)
    return seal


def _static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    forbidden_imports = sorted(top_level_imports.intersection({"torch", "transformers", "gr00t", "isaacgym", "omni"}))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}:
                forbidden_calls.append(name)
    return {
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "status": "PASS" if not forbidden_imports and not forbidden_calls else "BLOCKED",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO / "manifests/STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1.json")
    parser.add_argument("--mapping", type=Path, default=REPO / "manifests/STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1.json")
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--device", default="cuda:0")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        mapping_summary = audit_protocol_and_mapping(args.protocol, args.mapping)
        if args.plan_only:
            payload = _plan(mapping_summary)
            payload["static_guard"] = _static_guard()
        elif args.audit_only:
            payload = {"schema": ADAPTER_SCHEMA, "audit": mapping_summary, "static_guard": _static_guard(), "return_token": "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_READY"}
            if args.runtime_binding is not None:
                regular_file(args.runtime_binding, "runtime binding")
                binding = read_json(args.runtime_binding)
                _require(isinstance(binding, Mapping), "runtime binding must be an object")
                paths = binding.get("paths")
                bound_paths = isinstance(paths, Mapping) and all(paths.get(key) is not None for key in PATH_KEYS)
                if bound_paths:
                    payload["production_binding_audit"] = runtime_preflight(binding, mapping_summary)
                else:
                    payload["binding_audit"] = validate_runtime_binding(binding, mapping_summary, require_paths=False)
        else:
            _require(args.runtime_binding is not None, "--execute requires --runtime-binding")
            regular_file(args.runtime_binding, "runtime binding")
            binding = read_json(args.runtime_binding)
            _require(isinstance(binding, Mapping), "runtime binding must be an object")
            payload = _run_execution(binding, mapping_summary, args.protocol, args.mapping, args.device)
        print(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
