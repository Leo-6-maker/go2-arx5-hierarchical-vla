#!/usr/bin/env python3
"""Audit the exact N1.7 NEW_EMBODIMENT checkpoint rows without model construction.

This is a CPU/static audit.  It verifies the frozen checkpoint identity and
reads only category-specific safetensor rows through ``safe_open``'s CPU
PyTorch backend (needed for bfloat16 decoding).  It never constructs a GR00T
model, opens a CUDA device, or runs model inference/training.

The report deliberately distinguishes observations from the conservative
``ID10`` heuristic. A checkpoint row's RMS alone is not sufficient evidence
that a slot was trained; the classification is promoted only when a stable,
cross-block separation from both named and unmapped comparison cohorts is
observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


EXPECTED_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
EXPECTED_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_TREE_PREFIX = "payload/GR00T-N1.7-3B"
INITIALIZER_RMS = 0.02
REFERENCE_IDS = (10, 24, 25, 26, 27, 11)
KNOWN_PRETRAINED_REFERENCE_IDS = (24, 25, 26, 27)
WEIGHT_KEYS = (
    "action_head.action_encoder.W1.W",
    "action_head.action_encoder.W2.W",
    "action_head.action_encoder.W3.W",
    "action_head.action_decoder.layer1.W",
    "action_head.action_decoder.layer2.W",
)
_BLOCK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_regular(path: Path) -> None:
    mode = os.lstat(path).st_mode
    if stat.S_IFMT(mode) != stat.S_IFREG:
        raise ValueError(f"not a regular file: {path}")


def _tree_entries(root: Path, tree_prefix: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root = root.resolve(strict=True)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                raise ValueError(f"checkpoint tree contains symlink: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            _assert_regular(child)
            relative = child.relative_to(root).as_posix()
            entries.append(
                {
                    "path": f"{tree_prefix}/{relative}",
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def tree_sha256(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"invalid checkpoint path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"checkpoint path escapes root: {value!r}")
    return path.as_posix()


def _safe_slice(handle: Any, key: str, row: int) -> np.ndarray:
    # The Stage10 safetensors are bfloat16.  safetensors 0.8's NumPy backend
    # cannot decode that dtype, so use its CPU PyTorch backend and immediately
    # convert only the requested category row to float64 for statistics.
    try:
        value = handle.get_slice(key)[row]
    except AttributeError:
        value = handle.get_tensor(key)[row]
    if not torch.is_tensor(value):
        return np.asarray(value)
    return value.detach().to(dtype=torch.float64, device="cpu").numpy()


def _stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    flat = values.reshape(-1)
    squared = np.square(flat)
    abs_values = np.abs(flat)
    finite = bool(np.isfinite(flat).all())
    return {
        "shape": list(values.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(math.sqrt(float(np.mean(squared)))),
        "l2": float(math.sqrt(float(np.sum(squared)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean_abs": float(np.mean(abs_values)),
        "max_abs": float(np.max(abs_values)),
        "zero_fraction": float(np.mean(flat == 0)),
        "initializer_rms_ratio": float(math.sqrt(float(np.mean(squared))) / INITIALIZER_RMS),
    }


def _block_stats(array: np.ndarray, axis: int, split: int) -> dict[str, Any]:
    values = np.asarray(array)
    if values.shape[axis] <= split:
        raise ValueError(f"cannot split shape {values.shape} at {split} on axis {axis}")
    active = np.take(values, np.arange(split), axis=axis)
    inactive = np.take(values, np.arange(split, values.shape[axis]), axis=axis)
    active_summary = _stats(active)
    inactive_summary = _stats(inactive)
    active_energy = float(np.sum(np.square(np.asarray(active, dtype=np.float64))))
    inactive_energy = float(np.sum(np.square(np.asarray(inactive, dtype=np.float64))))
    total_energy = active_energy + inactive_energy
    return {
        "split_axis": axis,
        "active_range": [0, split],
        "inactive_range": [split, int(values.shape[axis])],
        "active": active_summary,
        "inactive": inactive_summary,
        "inactive_to_active_rms": float(inactive_summary["rms"] / active_summary["rms"])
        if active_summary["rms"] > 0
        else None,
        "inactive_energy_fraction": float(inactive_energy / total_energy)
        if total_energy > 0
        else None,
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _cohort_names(embodiment_id_path: Path, num_categories: int) -> dict[int, list[str]]:
    if not embodiment_id_path.exists():
        return {index: [] for index in range(num_categories)}
    raw = _load_json(embodiment_id_path)
    names: dict[int, list[str]] = {index: [] for index in range(num_categories)}
    for name, value in raw.items():
        if isinstance(value, int) and 0 <= value < num_categories:
            names[value].append(str(name))
    return names


def _classify_id10(
    profiles: dict[str, dict[int, dict[str, Any]]],
    named_ids: set[int],
    unmapped_ids: set[int],
) -> tuple[str, dict[str, Any]]:
    """Make a deliberately conservative cross-block classification.

    The classification is only promoted out of ``INDETERMINATE`` when every
    audited weight block agrees on the direction of the row-10 separation.
    It is evidence about weight shape, not a claim about training history.
    """

    id10 = 10
    per_key: dict[str, Any] = {}
    unused_votes = 0
    trained_votes = 0
    usable = 0
    for key, rows in profiles.items():
        if id10 not in rows:
            continue
        row10 = rows[id10]["flat"]
        named = [rows[index]["flat"] for index in named_ids if index in rows and index != id10]
        unmapped = [rows[index]["flat"] for index in unmapped_ids if index in rows and index != id10]
        if not named or not unmapped:
            continue
        named_similarity = float(np.mean([_cosine(row10, value) for value in named]))
        unmapped_similarity = float(np.mean([_cosine(row10, value) for value in unmapped]))
        named_distance = float(np.mean([np.linalg.norm(row10 - value) for value in named]))
        unmapped_distance = float(np.mean([np.linalg.norm(row10 - value) for value in unmapped]))
        named_rms = [profiles[key][index]["rms"] for index in named_ids]
        unmapped_rms = [profiles[key][index]["rms"] for index in unmapped_ids]
        id10_rms = profiles[key][id10]["rms"]
        named_rms_median = float(np.median(named_rms))
        unmapped_rms_median = float(np.median(unmapped_rms))
        named_log_rms_distance = abs(math.log(id10_rms / named_rms_median))
        unmapped_log_rms_distance = abs(math.log(id10_rms / unmapped_rms_median))
        initializer_like = profiles[key][id10]["initializer_rms_ratio"] <= 1.10
        # RMS cohort separation is the useful signal here: high-dimensional
        # Require the same direction in every usable block and keep a
        # conservative fallback when the cohorts are not separated.
        looks_unused = (
            initializer_like
            and unmapped_log_rms_distance < named_log_rms_distance * 0.50
        )
        looks_trained = (
            not initializer_like
            and named_log_rms_distance < unmapped_log_rms_distance * 0.50
        )
        unused_votes += int(looks_unused)
        trained_votes += int(looks_trained)
        usable += 1
        per_key[key] = {
            "named_mean_cosine": named_similarity,
            "unmapped_mean_cosine": unmapped_similarity,
            "named_mean_l2": named_distance,
            "unmapped_mean_l2": unmapped_distance,
            "id10_rms": id10_rms,
            "named_rms_median": named_rms_median,
            "unmapped_rms_median": unmapped_rms_median,
            "named_log_rms_distance": named_log_rms_distance,
            "unmapped_log_rms_distance": unmapped_log_rms_distance,
            "initializer_like": initializer_like,
            "unused_like_vote": looks_unused,
            "pretrained_or_posttrained_like_vote": looks_trained,
        }
    if usable >= 3 and unused_votes == usable:
        classification = "ID10_UNUSED_LIKE"
    elif usable >= 3 and trained_votes == usable:
        classification = "ID10_PRETRAINED_OR_POSTTRAINED_LIKE"
    else:
        classification = "ID10_INDETERMINATE_FROM_WEIGHTS"
    return classification, {
        "usable_blocks": usable,
        "unused_like_votes": unused_votes,
        "pretrained_or_posttrained_like_votes": trained_votes,
        "per_block": per_key,
        "rule": "all usable blocks must agree on initializer-like scale and RMS-cohort proximity; otherwise indeterminate",
    }


def audit_checkpoint(checkpoint: Path, *, tree_prefix: str) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    index_path = checkpoint / "model.safetensors.index.json"
    _assert_regular(index_path)
    index_sha = sha256_file(index_path)
    if index_sha != EXPECTED_INDEX_SHA256:
        raise ValueError(f"index SHA mismatch: {index_sha} != {EXPECTED_INDEX_SHA256}")
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model index has no weight_map")
    entries = _tree_entries(checkpoint, tree_prefix)
    actual_tree = tree_sha256(entries)
    if actual_tree != EXPECTED_TREE_SHA256:
        raise ValueError(f"tree SHA mismatch: {actual_tree} != {EXPECTED_TREE_SHA256}")

    referenced_shard_names = sorted({_safe_relative_path(value) for value in weight_map.values()})
    for shard_name in referenced_shard_names:
        _assert_regular(checkpoint / shard_name)
    key_shards = {key: checkpoint / _safe_relative_path(weight_map[key]) for key in WEIGHT_KEYS if key in weight_map}
    missing_keys = sorted(set(WEIGHT_KEYS) - set(key_shards))
    if missing_keys:
        raise ValueError(f"required weight keys missing from index: {missing_keys}")
    shard_hashes = {
        shard_name: sha256_file(checkpoint / shard_name)
        for shard_name in referenced_shard_names
    }
    all_rows: dict[str, dict[int, dict[str, Any]]] = {}
    block_reports: dict[str, dict[str, Any]] = {}
    num_categories: int | None = None
    for key in WEIGHT_KEYS:
        path = key_shards[key]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(key).get_shape())
            if not shape or shape[0] <= max(REFERENCE_IDS):
                raise ValueError(f"unexpected category dimension for {key}: {shape}")
            num_categories = int(shape[0])
            rows: dict[int, dict[str, Any]] = {}
            for category_id in range(num_categories):
                row = _safe_slice(handle, key, category_id)
                summary = _stats(row)
                summary["flat"] = np.asarray(row, dtype=np.float64).reshape(-1)
                rows[category_id] = summary
            all_rows[key] = rows
        if key.endswith("action_encoder.W1.W"):
            block_reports[key] = {
                str(category_id): _block_stats(
                    np.asarray(rows[category_id]["flat"]).reshape(shape[1:]), axis=0, split=10
                )
                for category_id in REFERENCE_IDS
            }
        elif key.endswith("action_decoder.layer2.W"):
            block_reports[key] = {
                str(category_id): _block_stats(
                    np.asarray(rows[category_id]["flat"]).reshape(shape[1:]), axis=1, split=10
                )
                for category_id in REFERENCE_IDS
            }

    assert num_categories is not None
    names = _cohort_names(checkpoint / "embodiment_id.json", num_categories)
    named_ids = set(KNOWN_PRETRAINED_REFERENCE_IDS).intersection(range(num_categories))
    unmapped_ids = {
        index for index, values in names.items() if not values and index != 10
    }
    classification, classification_evidence = _classify_id10(all_rows, named_ids, unmapped_ids)

    serialized_profiles: dict[str, Any] = {}
    for key, rows in all_rows.items():
        serialized_profiles[key] = {}
        for category_id in REFERENCE_IDS:
            summary = dict(rows[category_id])
            summary.pop("flat", None)
            summary["category_id"] = category_id
            summary["category_names"] = names.get(category_id, [])
            serialized_profiles[key][str(category_id)] = summary

    return {
        "schema": "STAGE24_STATIC_CHECKPOINT_AUDIT_V1",
        "checkpoint": str(checkpoint),
        "identity": {
            "index_sha256": index_sha,
            "index_sha256_expected": EXPECTED_INDEX_SHA256,
            "tree_sha256": actual_tree,
            "tree_sha256_expected": EXPECTED_TREE_SHA256,
            "tree_prefix": tree_prefix,
            "tree_file_count": len(entries),
            "tree_total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "referenced_shard_sha256": shard_hashes,
        },
        "checkpoint_shape": {
            "num_categories": num_categories,
            "reference_ids": list(REFERENCE_IDS),
            "category_names": {str(index): values for index, values in names.items()},
            "unmapped_comparison_ids": sorted(unmapped_ids),
            "weight_keys": WEIGHT_KEYS,
        },
        "id10": {
            "classification": classification,
            "comparison": classification_evidence,
            "claim_boundary": "static weight-shape evidence only; no training-history claim",
        },
        "category_weight_profiles": serialized_profiles,
        "active_inactive_block_audit": block_reports,
        "execution_boundary": {
            "torch_imported": True,
            "torch_device": "cpu",
            "cuda_used": False,
            "model_constructed": False,
            "forward": False,
            "inference": False,
            "training": False,
            "gpu_used": False,
            "checkpoint_modified": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tree-prefix", default=EXPECTED_TREE_PREFIX)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = audit_checkpoint(args.checkpoint, tree_prefix=args.tree_prefix)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
