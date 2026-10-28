#!/usr/bin/env python3
"""Preparation-only helpers for the future recovered-L1 TRAIN24 evaluator.

The active evaluator currently expects a historical PASS training-result file,
omits the L1@2000-vs-L1@1000 paired comparison, does not persist per-window row
bundles, and builds an inventory before mutating the final seal. This module
prepares small, standard-library helpers that a future separately gated patch
can import to fix those issues without changing model/sampler/decoder semantics.

This module itself never loads a model, Torch, a checkpoint payload, or TRAIN24.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


RECOVERED_INPUT_SCHEMA = "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2"
RECOVERED_INPUT_STATUS = "PASS_CONTRACT_BOUND_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION"
RECOVERED_SOURCE_KIND = "CONTRACT_BOUND_READONLY_RECOVERY_NOT_HISTORICAL_PASS_RESULT"
HISTORICAL_TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
BRANCH = "stage45e/l1-physical-error-aware-loss"
ARM_ORDER = ("FULL24@2000", "L1@1000", "L1@2000")
WINDOW_COUNT = 24


class L1EvalRecoveryAdapterError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise L1EvalRecoveryAdapterError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_fresh(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_recovered_input(path: Path, training_root: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    value = read_json(path)
    require(value.get("schema") == RECOVERED_INPUT_SCHEMA, "recovered eval-input schema drift")
    require(value.get("status") == RECOVERED_INPUT_STATUS, "recovered eval-input status drift")
    require(value.get("source_kind") == RECOVERED_SOURCE_KIND, "recovered source-kind drift")
    require(Path(str(value.get("training_root"))).resolve() == training_root, "recovered training-root identity drift")
    require(value.get("historical_training_head") == HISTORICAL_TRAINING_HEAD, "recovered historical training HEAD drift")
    require(value.get("branch") == BRANCH, "recovered training branch drift")
    require(value.get("evaluation_authorized_by_this_bridge") is False, "bridge must not claim execution authority")
    require(value.get("final_train24_token_emitted") is False, "bridge already claims final TRAIN24 completion")
    scientific_contract = value.get("scientific_contract")
    require(isinstance(scientific_contract, Mapping), "contract-bound L1 scientific contract is missing")
    runtime_contract = scientific_contract.get("l1_runtime_contract")
    code_contract = scientific_contract.get("l1_code_contract")
    recipe = scientific_contract.get("training_recipe")
    require(isinstance(runtime_contract, Mapping), "runtime L1 contract missing")
    require(isinstance(code_contract, Mapping), "code L1 contract missing")
    require(isinstance(recipe, Mapping), "L1 training recipe binding missing")
    require(runtime_contract.get("active_scalar_count") == 190, "runtime active mask drift")
    require(code_contract.get("active_scalar_count") == 190, "code active mask drift")
    require(code_contract.get("denominator") == "binary_action_mask.sum()", "L1 denominator drift")
    require(scientific_contract.get("normalization_unchanged") is True, "recovered normalization drift")
    require(recipe.get("seed") == 2222 and recipe.get("max_steps") == 2000, "recovered training seed/steps drift")

    checkpoints = value.get("checkpoints")
    require(isinstance(checkpoints, Mapping), "recovered checkpoint block missing")
    normalized = {}
    for label, step in (("L1@1000", 1000), ("L1@2000", 2000)):
        record = checkpoints.get(label)
        require(isinstance(record, Mapping), f"{label} recovered checkpoint missing")
        tree = record.get("tree_sha256")
        require(isinstance(tree, str) and len(tree) == 64, f"{label} recovered tree missing")
        require(record.get("global_step") == step, f"{label} recovered global step drift")
        require(record.get("rehash_matches_acceptance") is True, f"{label} was not rehashed against acceptance")
        require(
            Path(str(record.get("path"))).resolve() == (training_root / f"checkpoint-{step}").resolve(),
            f"{label} recovered checkpoint path drift",
        )
        normalized[label] = dict(record)
    require(
        value.get("runtime_effective_weight_evidence_status")
        in {"CAPTURED", "NOT_CAPTURED_IN_HISTORICAL_CANARY", "REPRODUCED_RUNTIME_COEFFICIENT_EVIDENCE"},
        "recovered runtime-weight evidence status drift",
    )
    return {
        "schema": "STAGE45E_L1_RECOVERY_EVAL_ADAPTER_INPUT_V2",
        "recovered_eval_input_path": str(path),
        "recovered_eval_input_sha256": sha256_file(path),
        "acceptance_path": value.get("acceptance_path"),
        "acceptance_sha256": value.get("acceptance_sha256"),
        "training_root": str(training_root),
        "historical_training_head": value.get("historical_training_head"),
        "branch": value.get("branch"),
        "structured_stats": value.get("structured_stats"),
        "scientific_contract": dict(scientific_contract),
        "checkpoints": normalized,
        "runtime_effective_weight_evidence_status": value.get("runtime_effective_weight_evidence_status"),
    }


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def _describe_pair(
    row_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    left: str,
    right: str,
    window_ids: Sequence[str],
) -> dict[str, Any]:
    flattened = {
        label: {wid: _flatten_numeric(row_maps[label][wid].get("metrics", {})) for wid in window_ids}
        for label in (left, right)
    }
    keys = sorted(set().union(*(set(flattened[label][wid]) for label in (left, right) for wid in window_ids)))
    metric_delta: dict[str, Any] = {}
    for key in keys:
        values = [
            flattened[left][wid][key] - flattened[right][wid][key]
            for wid in window_ids
            if key in flattened[left][wid] and key in flattened[right][wid]
        ]
        if values:
            metric_delta[key] = {
                "count": len(values),
                "mean": float(statistics.fmean(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
    return {
        "left": left,
        "right": right,
        "window_count": len(window_ids),
        "metric_delta_left_minus_right": metric_delta,
    }


def paired_summary_complete(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    require(tuple(rows_by_arm.keys()) == ARM_ORDER, "L1 evaluator arm order/set drift")
    row_maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    reference_windows: list[str] | None = None
    for label in ARM_ORDER:
        rows = rows_by_arm[label]
        require(len(rows) == WINDOW_COUNT, f"{label} row count drift")
        mapping = {str(row.get("window_id")): row for row in rows}
        require(len(mapping) == WINDOW_COUNT and all(mapping), f"{label} duplicate/missing window ids")
        window_ids = list(mapping)
        if reference_windows is None:
            reference_windows = window_ids
        else:
            require(window_ids == reference_windows, f"{label} window order mismatch")
        row_maps[label] = mapping
    assert reference_windows is not None

    def target_of(window_id: str) -> str:
        return window_id.split(":", 1)[0].removeprefix("stage22_expert_")

    pairs = {
        "L1@1000_minus_FULL24@2000": ("L1@1000", "FULL24@2000"),
        "L1@2000_minus_FULL24@2000": ("L1@2000", "FULL24@2000"),
        "L1@2000_minus_L1@1000": ("L1@2000", "L1@1000"),
    }
    output: dict[str, Any] = {"baseline": "FULL24@2000", "pairs": {}}
    targets = sorted({target_of(wid) for wid in reference_windows})
    for pair_name, (left, right) in pairs.items():
        output["pairs"][pair_name] = {
            "overall": _describe_pair(row_maps, left, right, reference_windows),
            "by_target": {
                target: _describe_pair(
                    row_maps,
                    left,
                    right,
                    [wid for wid in reference_windows if target_of(wid) == target],
                )
                for target in targets
            },
            "by_window": {
                wid: _describe_pair(row_maps, left, right, [wid]) for wid in reference_windows
            },
        }
    return output


def persist_row_bundles(output_root: Path, rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    output_root = Path(output_root)
    require(tuple(rows_by_arm.keys()) == ARM_ORDER, "row-bundle arm order/set drift")
    result: dict[str, Any] = {}
    for label in ARM_ORDER:
        rows = list(rows_by_arm[label])
        require(len(rows) == WINDOW_COUNT, f"{label} row-bundle cardinality drift")
        path = output_root / "rows" / f"{label.replace('@', '_at_')}_rows.json"
        write_json_fresh(path, rows)
        result[label] = {
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
            "row_count": len(rows),
        }
    return result


def build_immutable_inventory(
    output_root: Path,
    relative_paths: Sequence[str],
    *,
    inventory_relative_path: str = "inventory/IMMUTABLE_SHA256SUMS",
) -> dict[str, Any]:
    """Inventory only immutable evidence files, never the final seal/result.

    The caller explicitly supplies the immutable evidence set after all model
    and CPU-derived evidence has been written. The final evaluation seal/result
    are then written *after* this inventory and may reference its hash without a
    self-referential/cyclic inventory.
    """

    output_root = Path(output_root).resolve()
    forbidden = {
        "stage45e_l1_eval_seal.json",
        "stage45e_l1_eval_result.json",
        inventory_relative_path,
    }
    unique = list(dict.fromkeys(relative_paths))
    require(unique, "immutable inventory input set is empty")
    require(not any(path in forbidden for path in unique), "final seal/result/inventory must not be inventoried")
    lines: list[str] = []
    for relative in sorted(unique):
        path = output_root / relative
        require(path.is_file() and not path.is_symlink(), f"immutable evidence missing/unsafe: {relative}")
        lines.append(f"{sha256_file(path)}  {relative}")
    inventory = output_root / inventory_relative_path
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    require(not inventory.exists() and not inventory.is_symlink(), "immutable inventory path must be fresh")
    inventory.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory.with_name(inventory.name + ".tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, inventory)
    return {
        "path": str(inventory.relative_to(output_root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "entry_count": len(lines),
        "final_seal_excluded": True,
        "final_result_excluded": True,
        "inventory_is_immutable_after_creation": True,
    }


__all__ = [
    "ARM_ORDER",
    "L1EvalRecoveryAdapterError",
    "RECOVERED_INPUT_SCHEMA",
    "build_immutable_inventory",
    "paired_summary_complete",
    "persist_row_bundles",
    "validate_recovered_input",
]
