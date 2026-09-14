#!/usr/bin/env python3
"""Recover PI0163 Validation-E from allocation-8's six valid raw rows.

Allocation 615034 passed the compute-node runtime gate and produced the full
Legacy(seed2222) six-window raw/derived set, but its frozen CPU readout failed
because the legacy aggregate default expected the unrelated 24-window FULL24
TRAIN cohort. This recovery keeps all six allocation-8 raw files byte-for-byte,
recomputes their derived rows on CPU, skips that completed arm entirely, and
executes only the remaining 30 queries under the already admitted six-arm E4
contract.

This module is deliberately outside the frozen PI0154 evaluator surface. It
imports no model/runtime package at module scope, never trains, never retries
an entered model call, and records cumulative allocation accounting separately
from the current recovery-process counters.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45f_pi0159_validation_e_continuation as continuation  # noqa: E402


PI0167_GATE = 5657908043
PI0167_POINTER = 5657909976
PI0167_TOKEN = "PI_GATE_STATE_0167"
PI0165_SCIENCE_GATE = 5654073910
PI0165A_TAKEOVER_GATE = 5654552673
PI0165A_ERRATUM = 5654559228
RETURN_TOKEN = "VANDA_PI0167_VALIDATION_E_GEN1_FOR_PI_REVIEW — COMPLETED"
EXPECTED_JOB = "615034.hopper-m-02"
EXPECTED_ALLOCATION_7_FAILURE_JOB = "615030.hopper-m-02"
EXPECTED_ALLOCATION_9_FAILURE_JOB = "615124.hopper-m-02"
EXPECTED_SOURCE_ALLOCATION = 8
RECOVERY_ALLOCATION = 10
MAX_TOTAL_ALLOCATIONS = 11
MAX_TOTAL_MODEL_LOAD_ATTEMPTS = 8
HISTORICAL_MODEL_LOAD_ATTEMPTS = 3
EXPECTED_RAW_SCHEMA = "STAGE45F_PI0154_RAW_PREDICTION_V1"
EXPECTED_TARGET_ID = "red_target_holdout_e"
EXPECTED_REUSED_ARM = "Legacy(seed2222)@2000"
EXPECTED_REUSED_WINDOW = "stage22_expert_red_target_holdout_e:expert-0000"
EXPECTED_RAW_RELATIVE_PATH = "raw/Legacy_seed2222_at_2000/window-00.npy"
EXPECTED_RAW_SHAPE = (1, 40, 132)
EXPECTED_TIMESTEPS = [0, 250, 500, 750]
EXPECTED_WINDOW_IDS = tuple(
    f"stage22_expert_red_target_holdout_e:expert-{index:04d}"
    for index in range(6)
)
EXPECTED_NOMINAL = {"conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36}
EXPECTED_RECOVERY = {"reused_raw_queries": 6, "new_queries": 30, "conditioning": 30, "action_head": 120, "model_side_total": 150, "model_loads": 5}
MAX_NEW_MODEL_LOADS = EXPECTED_RECOVERY["model_loads"]
MAX_NEW_MODEL_SIDE_CALLS = EXPECTED_RECOVERY["model_side_total"]
EXPECTED_ARM_KEYS = tuple(
    (route, seed)
    for seed in (2222, 2223, 2224)
    for route in ("Legacy", "L1")
)
EXPECTED_ARM_LABELS = tuple(f"{route}(seed{seed})@2000" for route, seed in EXPECTED_ARM_KEYS)
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "cuda", "cv2", "numpy"}
FORBIDDEN_CALLS = {
    "backward",
    "get_action",
    "qsub",
    "save_checkpoint",
    "save_model",
    "save_pretrained",
    "train",
}
RAW_INVENTORY_VERIFIER = SCRIPTS / "stage45f_pi0167_raw_inventory_verifier.py"


def require(condition: bool, message: str) -> None:
    continuation.require(condition, message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return continuation.sha256_file(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Any:
    return continuation.load_json(path)


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    continuation.write_json_durable(path, value, overwrite=overwrite)


def _label_path(label: str) -> str:
    return label.replace("(", "_").replace(")", "").replace("@", "_at_")


def _copy_file_durable(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    require(source.is_file() and not source.is_symlink(), f"recovery source is not a regular file: {source}")
    require(not destination.exists() and not destination.is_symlink(), f"recovery destination is not fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1 << 20), b""):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _copy_exact_text_line(source: Path, destination: Path) -> None:
    _copy_file_durable(source, destination)
    require(destination.read_bytes().endswith(b"\n"), f"JSONL source has no terminal newline: {source}")


def _read_raw_manifest(source_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest = source_root / "raw_predictions.jsonl"
    require(manifest.is_file() and not manifest.is_symlink(), f"recovery raw manifest is missing: {manifest}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        require(line.strip(), f"recovery raw manifest contains a blank line {line_number}")
        value = json.loads(line)
        require(isinstance(value, Mapping), f"recovery raw record {line_number} is not an object")
        records.append(dict(value))
    return manifest, records


def _expected_raw_path(label: str, window_index: int) -> str:
    return f"raw/{_label_path(label)}/window-{window_index:02d}.npy"


def _validate_raw_record_identity(record: Mapping[str, Any], source_root: Path, np_module: Any) -> dict[str, Any]:
    require(record.get("schema") == EXPECTED_RAW_SCHEMA, "recovery raw schema drift")
    checkpoint = record.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "recovery raw checkpoint is not an object")
    label = checkpoint.get("label")
    require(label in EXPECTED_ARM_LABELS, f"recovery raw checkpoint is outside the frozen matrix: {label!r}")
    expected_route, expected_seed = next((route, seed) for route, seed in EXPECTED_ARM_KEYS if f"{route}(seed{seed})@2000" == label)
    require(checkpoint.get("route") == expected_route, "recovery raw route identity drift")
    require(checkpoint.get("seed") == expected_seed, "recovery raw seed identity drift")
    require(checkpoint.get("tier") == 2000, "recovery raw tier identity drift")
    window_id = record.get("window_id")
    require(window_id in EXPECTED_WINDOW_IDS, f"recovery raw window is outside Validation-E: {window_id!r}")
    window_index = EXPECTED_WINDOW_IDS.index(str(window_id))
    path_value = record.get("path")
    require(isinstance(path_value, str), "recovery raw path is not text")
    relative_path = Path(path_value)
    require(
        not relative_path.is_absolute()
        and relative_path.as_posix() == path_value
        and all(part not in {"", ".", ".."} for part in relative_path.parts),
        "recovery raw path is not a normalized relative path",
    )
    expected_path = _expected_raw_path(str(label), window_index)
    require(path_value == expected_path, f"recovery raw path drift: {path_value} != {expected_path}")
    require(record.get("timestep_ids") == EXPECTED_TIMESTEPS, "recovery raw E4 timestep drift")
    require(record.get("action_head_calls_observed") == 4, "recovery raw action-head count drift")
    require(record.get("decoded") is False, "recovery raw was already decoded")
    require(record.get("shape") == list(EXPECTED_RAW_SHAPE), "recovery raw shape metadata drift")
    require(record.get("dtype") == "float32", "recovery raw dtype metadata drift")
    raw_path = source_root / relative_path
    require(raw_path.is_file() and not raw_path.is_symlink(), f"recovery raw array is missing: {raw_path}")
    observed_sha = sha256_file(raw_path)
    require(observed_sha == record.get("file_sha256"), "recovery raw file SHA drift")
    try:
        raw = np_module.load(raw_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"recovery raw array cannot be loaded without pickle: {raw_path}") from exc
    require(tuple(int(item) for item in raw.shape) == EXPECTED_RAW_SHAPE, "recovery raw array shape drift")
    require(str(raw.dtype) == "float32", "recovery raw array dtype drift")
    require(bool(np_module.isfinite(raw).all()), "recovery raw array contains non-finite values")
    observed_array_sha = sha256_bytes(np_module.ascontiguousarray(raw).astype(np_module.float32, copy=False).tobytes())
    require(observed_array_sha == record.get("array_sha256"), "recovery raw array SHA drift")
    return {
        "arm": str(label),
        "window_id": str(window_id),
        "path": path_value,
        "file_sha256": observed_sha,
        "array_sha256": observed_array_sha,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "timestep_ids": list(record["timestep_ids"]),
        "bytes": int(raw_path.stat().st_size),
        "record": dict(record),
    }


def raw_inventory(source_root: Path) -> dict[str, Any]:
    """Enumerate actual raw cells with the NumPy-backed deep verifier.

    This is the implementation entrypoint for the isolated verifier child.
    The execution parent must call :func:`raw_inventory_isolated` instead so
    NumPy never enters the parent ``sys.modules`` before frozen admission.
    """

    source_root = continuation.regular_directory(Path(source_root), "PI0167 raw inventory source root")
    import numpy as np

    manifest, records = _read_raw_manifest(source_root)
    entries = [_validate_raw_record_identity(record, source_root, np) for record in records]
    by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (entry["arm"], entry["window_id"])
        require(key not in by_cell, f"duplicate raw inventory cell: {key}")
        by_cell[key] = entry

    raw_directory = source_root / "raw"
    require(raw_directory.is_dir() and not raw_directory.is_symlink(), f"recovery raw directory is missing: {raw_directory}")
    discovered_paths: list[str] = []
    discovered_directories: list[str] = []
    for path in sorted(raw_directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"recovery raw inventory contains a symlink: {path}")
        if path.is_dir():
            discovered_directories.append(path.relative_to(source_root).as_posix())
            continue
        require(path.is_file(), f"recovery raw directory contains a non-regular entry: {path}")
        relative = path.relative_to(source_root).as_posix()
        require(relative.endswith(".npy"), f"recovery raw directory contains a non-array file: {relative}")
        discovered_paths.append(relative)
    manifest_paths = sorted(entry["path"] for entry in entries)
    require(discovered_paths == manifest_paths, "raw inventory files and manifest records disagree")
    manifest_directories = sorted({Path(entry["path"]).parent.as_posix() for entry in entries})
    require(discovered_directories == manifest_directories, "raw inventory directories and manifest records disagree")

    expected_cells = [(label, window_id) for label in EXPECTED_ARM_LABELS for window_id in EXPECTED_WINDOW_IDS]
    preserved_cells = [
        {
            "arm": label,
            "window_id": window_id,
            "path": by_cell[(label, window_id)]["path"],
            "file_sha256": by_cell[(label, window_id)]["file_sha256"],
            "array_sha256": by_cell[(label, window_id)]["array_sha256"],
            "shape": by_cell[(label, window_id)]["shape"],
            "dtype": by_cell[(label, window_id)]["dtype"],
            "timestep_ids": by_cell[(label, window_id)]["timestep_ids"],
            "bytes": by_cell[(label, window_id)]["bytes"],
            "record": by_cell[(label, window_id)]["record"],
        }
        for label, window_id in expected_cells
        if (label, window_id) in by_cell
    ]
    missing_cells = [
        {"arm": label, "window_id": window_id}
        for label, window_id in expected_cells
        if (label, window_id) not in by_cell
    ]
    return {
        "schema": "STAGE45F_PI0167_RAW_CELL_INVENTORY_V1",
        "status": "PASS_ACTUAL_RAW_INVENTORY",
        "source_root": str(source_root),
        "raw_manifest_path": str(manifest),
        "raw_manifest_sha256": sha256_file(manifest),
        "expected_arm_count": len(EXPECTED_ARM_LABELS),
        "expected_window_count_per_arm": len(EXPECTED_WINDOW_IDS),
        "expected_cell_count": len(expected_cells),
        "raw_manifest_record_count": len(entries),
        "raw_file_count": len(discovered_paths),
        "preserved_raw_count": len(preserved_cells),
        "missing_raw_count": len(missing_cells),
        "preserved_cells": preserved_cells,
        "missing_cells": missing_cells,
        "manifest_paths": manifest_paths,
        "actual_raw_paths": sorted(discovered_paths),
        "actual_raw_directories": discovered_directories,
        "extras": sorted(set(discovered_paths) - set(manifest_paths)),
        "missing_paths": sorted(set(manifest_paths) - set(discovered_paths)),
        "extra_directories": sorted(set(discovered_directories) - set(manifest_directories)),
        "missing_directories": sorted(set(manifest_directories) - set(discovered_directories)),
        "symlinks": [],
        "non_npy_entries": [],
        "scientific_contract_changed": False,
    }


def raw_inventory_isolated(source_root: Path) -> dict[str, Any]:
    """Run :func:`raw_inventory` in a child and return its sealed JSON result.

    Only standard-library code runs in this parent path.  The child imports
    NumPy to validate every reused array, then exits; no child module state can
    be present in the execution parent's ``sys.modules``.
    """

    source_root = continuation.regular_directory(Path(source_root), "isolated raw inventory source root")
    require(RAW_INVENTORY_VERIFIER.is_file() and not RAW_INVENTORY_VERIFIER.is_symlink(), f"raw inventory verifier is unavailable: {RAW_INVENTORY_VERIFIER}")
    completed = subprocess.run(
        [sys.executable, str(RAW_INVENTORY_VERIFIER), "--source-root", str(source_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"isolated raw inventory verifier failed ({completed.returncode}): {detail[-1200:]}")
    output = completed.stdout.strip()
    require(output, "isolated raw inventory verifier returned no JSON")
    try:
        inventory = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"isolated raw inventory verifier returned invalid JSON: {output[-1200:]}") from exc
    require(isinstance(inventory, Mapping), "isolated raw inventory verifier result is not an object")
    require(inventory.get("status") == "PASS_ACTUAL_RAW_INVENTORY", "isolated raw inventory verifier status drift")
    return dict(inventory)


def validate_allocation8_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Require allocation 8's exact six-cell preserved prefix and 30-cell remainder."""

    expected_preserved = [(EXPECTED_REUSED_ARM, window_id) for window_id in EXPECTED_WINDOW_IDS]
    observed_preserved = [(str(item["arm"]), str(item["window_id"])) for item in inventory.get("preserved_cells", [])]
    expected_all = [(label, window_id) for label in EXPECTED_ARM_LABELS for window_id in EXPECTED_WINDOW_IDS]
    observed_missing = [(str(item["arm"]), str(item["window_id"])) for item in inventory.get("missing_cells", [])]
    expected_missing = [cell for cell in expected_all if cell not in expected_preserved]
    require(int(inventory.get("preserved_raw_count", -1)) == 6, "allocation-8 preserved raw count drift")
    require(int(inventory.get("missing_raw_count", -1)) == 30, "allocation-8 missing raw count drift")
    require(observed_preserved == expected_preserved, "allocation-8 preserved cells are not the complete first arm")
    require(observed_missing == expected_missing, "allocation-8 missing-cell order or membership drift")
    require(int(inventory.get("raw_manifest_record_count", -1)) == 6, "allocation-8 manifest record count drift")
    require(int(inventory.get("raw_file_count", -1)) == 6, "allocation-8 raw file count drift")
    require(
        not inventory.get("extras")
        and not inventory.get("missing_paths")
        and not inventory.get("extra_directories")
        and not inventory.get("missing_directories")
        and not inventory.get("symlinks"),
        "allocation-8 raw file-set closure drift",
    )
    return dict(inventory)


def recovery_execution_plan(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Build the fail-closed 6-reused/30-new allocation-10 execution plan."""

    validated = validate_allocation8_inventory(inventory)
    preserved = {(str(item["arm"]), str(item["window_id"])) for item in validated["preserved_cells"]}
    expected_cells = [(label, window_id) for label in EXPECTED_ARM_LABELS for window_id in EXPECTED_WINDOW_IDS]
    new_cells = [cell for cell in expected_cells if cell not in preserved]
    model_load_arms = [label for label in EXPECTED_ARM_LABELS if label != EXPECTED_REUSED_ARM]
    model_loads_by_arm = {label: int(label in model_load_arms) for label in EXPECTED_ARM_LABELS}
    forward_cells_by_arm = {
        label: sum((label, window_id) in new_cells for window_id in EXPECTED_WINDOW_IDS)
        for label in EXPECTED_ARM_LABELS
    }
    plan = {
        "schema": "STAGE45F_PI0167_VALIDATION_E_RECOVERY_EXECUTION_PLAN_V1",
        "status": "PASS_6_REUSED_30_NEW",
        "preserved_cells": sorted(preserved),
        "new_cells": new_cells,
        "completed_arm": EXPECTED_REUSED_ARM,
        "model_load_arms": model_load_arms,
        "model_loads_by_arm": model_loads_by_arm,
        "forward_cells_by_arm": forward_cells_by_arm,
        "reused_raw_queries": len(preserved),
        "new_queries": len(new_cells),
        "new_calls": {"conditioning": len(new_cells), "action_head": len(new_cells) * 4, "model_side_total": len(new_cells) * 5},
        "model_load_attempts": len(model_load_arms),
        "first_arm_model_loads": model_loads_by_arm[EXPECTED_REUSED_ARM],
        "first_arm_forward_cells": forward_cells_by_arm[EXPECTED_REUSED_ARM],
        "scientific_contract_changed": False,
    }
    require(plan["reused_raw_queries"] == EXPECTED_RECOVERY["reused_raw_queries"], "recovery preserved query count drift")
    require(plan["new_queries"] == EXPECTED_RECOVERY["new_queries"], "recovery new query count drift")
    require(plan["new_calls"]["conditioning"] == EXPECTED_RECOVERY["conditioning"], "recovery conditioning plan drift")
    require(plan["new_calls"]["action_head"] == EXPECTED_RECOVERY["action_head"], "recovery action-head plan drift")
    require(plan["new_calls"]["model_side_total"] == EXPECTED_RECOVERY["model_side_total"], "recovery model-side plan drift")
    require(plan["model_load_attempts"] == EXPECTED_RECOVERY["model_loads"], "recovery model-load plan drift")
    require(plan["first_arm_model_loads"] == 0 and plan["first_arm_forward_cells"] == 0, "completed first arm is not model-skipped")
    return plan


def _validate_frozen_artifacts_from_clean_parent(
    evaluator: Any,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    expected_head: str,
) -> tuple[dict[str, Any], list[str]]:
    """Cross the frozen artifact-admission call only from a clean parent."""

    parent_imports = evaluator.admission.model_imports_present()
    require(parent_imports == [], f"forbidden parent imports immediately before frozen artifact admission: {parent_imports}")
    artifact_audit = evaluator.validate_artifacts(config, binding, expected_head)
    return artifact_audit, list(parent_imports)


def _git_text(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, f"git command failed: {' '.join(arguments)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def pre_submit(config_path: Path, *, repo: Path = REPO) -> dict[str, Any]:
    """Run the CPU-only gate that must pass before qsub is called."""

    config = continuation._config(config_path.resolve(strict=True))
    expected_head = str(config["expected_head"])
    observed_head = _git_text(repo, "rev-parse", "HEAD")
    observed_branch = _git_text(repo, "branch", "--show-current")
    observed_status = _git_text(repo, "status", "--porcelain=v1", "--untracked-files=all")
    require(observed_branch == continuation.EXPECTED_BRANCH, f"pre-submit branch drift: {observed_branch}")
    require(observed_head == expected_head, f"pre-submit expected_head mismatch: {expected_head} != {observed_head}")
    require(not observed_status, "pre-submit checkout is not clean, including untracked files")
    require(config.get("pi0167_support_head") == expected_head, "pre-submit support HEAD drift")
    require(config.get("pi0167_recovery_allocation") == RECOVERY_ALLOCATION, "pre-submit recovery allocation drift")
    require(config.get("max_total_allocations") == MAX_TOTAL_ALLOCATIONS, "pre-submit allocation ceiling drift")
    require(config.get("pi0167_max_new_model_loads") == MAX_NEW_MODEL_LOADS, "pre-submit PI0167 model-load ceiling drift")
    require(config.get("pi0167_max_new_model_side_calls") == MAX_NEW_MODEL_SIDE_CALLS, "pre-submit PI0167 model-side ceiling drift")
    require(HISTORICAL_MODEL_LOAD_ATTEMPTS + MAX_NEW_MODEL_LOADS <= MAX_TOTAL_MODEL_LOAD_ATTEMPTS, "pre-submit cumulative model-load ceiling drift")
    source_root = continuation.regular_directory(Path(str(config["pi0163_recovery_source_root"])), "PI0163 recovery source root")
    inventory = validate_allocation8_inventory(raw_inventory_isolated(source_root))
    require(inventory["raw_manifest_sha256"] == str(config["pi0163_recovery_source_raw_manifest_sha256"]), "pre-submit raw manifest SHA drift")
    source_inventory_path = continuation.regular_file(Path(str(config["pi0167_recovery_source_inventory_path"])), "PI0167 allocation-8 raw inventory")
    require(sha256_file(source_inventory_path) == str(config["pi0167_recovery_source_inventory_sha256"]), "allocation-8 inventory artifact SHA drift")
    source_inventory = load_json(source_inventory_path)
    require(isinstance(source_inventory, Mapping), "allocation-8 inventory artifact is not an object")
    require(source_inventory.get("status") == "PASS_ACTUAL_ALLOCATION8_RAW_INVENTORY", "allocation-8 inventory artifact status drift")
    require(source_inventory.get("raw_manifest_sha256") == inventory["raw_manifest_sha256"], "allocation-8 inventory artifact manifest SHA drift")
    require(source_inventory.get("preserved_raw_count") == inventory["preserved_raw_count"], "allocation-8 inventory artifact preserved count drift")
    require(source_inventory.get("missing_raw_count") == inventory["missing_raw_count"], "allocation-8 inventory artifact missing count drift")
    require(_inventory_cell_identity(source_inventory) == _inventory_cell_identity(inventory), "allocation-8 inventory artifact cell/hash drift")
    require(source_inventory.get("actual_raw_paths") == inventory["actual_raw_paths"], "allocation-8 inventory artifact path drift")
    allocation8_failure = _allocation_failure_ref(
        config,
        path_key="pi0167_alloc8_failure_path",
        sha_key="pi0167_alloc8_failure_sha256",
        allocation_label="allocation-8",
        expected_status="PARTIAL",
        require_model_execution=True,
    )
    failure_counters = allocation8_failure["actual_counters"]
    require(failure_counters.get("model_loads_attempted") == 1, "allocation-8 model-load attempt drift")
    require(failure_counters.get("model_loads_completed") == 1, "allocation-8 model-load completion drift")
    require(failure_counters.get("raw_prediction_count") == 6, "allocation-8 raw count drift")
    require(failure_counters.get("derived_row_count") == 6, "allocation-8 derived count drift")
    allocation9_failure = _allocation_failure_ref(
        config,
        path_key="pi0167_alloc9_failure_path",
        sha_key="pi0167_alloc9_failure_sha256",
        allocation_label="allocation-9",
        expected_status="BLOCKED",
        require_model_execution=False,
    )
    _validate_allocation9_zero_boundary(allocation9_failure)
    output_root = Path(str(config["output_root"])).resolve()
    require(not output_root.exists() and not output_root.is_symlink(), "pre-submit output root is not fresh")
    guard = static_guard()
    require(guard["status"] == "PASS", "pre-submit recovery static guard failed")
    return {
        "schema": "STAGE45F_PI0167_RECOVERY_PRE_SUBMIT_GATE_V1",
        "status": "PASS_PRE_SUBMIT_CPU_ONLY",
        "repo": str(Path(repo).resolve()),
        "branch": observed_branch,
        "expected_head": expected_head,
        "observed_head": observed_head,
        "clean_worktree": True,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "raw_inventory": inventory,
        "recovery_execution_plan": recovery_execution_plan(inventory),
        "allocation_8_failure": allocation8_failure,
        "allocation_9_failure": allocation9_failure,
        "static_guard": guard,
        "output_root": str(output_root),
        "model_imported": False,
        "gpu_requested": False,
        "qsub_called": False,
        "scientific_contract_changed": False,
    }


def _preserved_raw_records(inventory: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for item in inventory.get("preserved_cells", []):
        require(isinstance(item, Mapping) and isinstance(item.get("record"), Mapping), "preserved raw record provenance is missing")
        key = (str(item["arm"]), str(item["window_id"]))
        require(key not in records, f"duplicate preserved raw record: {key}")
        records[key] = dict(item)
    require(len(records) == int(inventory["preserved_raw_count"]), "preserved raw record count drift")
    return records


def _evidence_ref(config: Mapping[str, Any], path_key: str, sha_key: str) -> dict[str, Any]:
    path = continuation.regular_file(Path(str(config[path_key])), path_key)
    observed = sha256_file(path)
    require(observed == str(config[sha_key]), f"{path_key} SHA drift: {observed}")
    return {"path": str(path), "sha256": observed, "bytes": int(path.stat().st_size)}


def _inventory_cell_identity(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize inventory cells for comparison with the independently sealed artifact."""

    result = []
    for item in inventory.get("preserved_cells", []):
        require(isinstance(item, Mapping), "raw inventory preserved cell is not an object")
        result.append({
            "arm": str(item["arm"]),
            "window_id": str(item["window_id"]),
            "path": str(item["path"]),
            "file_sha256": str(item["file_sha256"]),
            "array_sha256": str(item["array_sha256"]),
            "shape": list(item["shape"]),
            "dtype": str(item["dtype"]),
            "timestep_ids": list(item["timestep_ids"]),
        })
    return result


def validate_complete_raw_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Require the post-recovery raw surface to be exactly the frozen 36-cell matrix."""

    expected_cells = [(label, window_id) for label in EXPECTED_ARM_LABELS for window_id in EXPECTED_WINDOW_IDS]
    observed_cells = [(str(item["arm"]), str(item["window_id"])) for item in inventory.get("preserved_cells", [])]
    require(int(inventory.get("preserved_raw_count", -1)) == len(expected_cells), "final raw inventory preserved count drift")
    require(int(inventory.get("missing_raw_count", -1)) == 0, "final raw inventory still has missing cells")
    require(int(inventory.get("raw_manifest_record_count", -1)) == len(expected_cells), "final raw manifest record count drift")
    require(int(inventory.get("raw_file_count", -1)) == len(expected_cells), "final raw file count drift")
    require(observed_cells == expected_cells, "final raw inventory cell order or membership drift")
    require(
        not inventory.get("extras")
        and not inventory.get("missing_paths")
        and not inventory.get("extra_directories")
        and not inventory.get("missing_directories")
        and not inventory.get("symlinks"),
        "final raw inventory file-set closure drift",
    )
    return dict(inventory)


def validate_complete_derived_manifest(path: Path, raw_inventory_value: Mapping[str, Any]) -> int:
    """Check all derived references against the exact 36-cell raw inventory."""

    path = continuation.regular_file(path, "complete derived evidence manifest")
    raw_by_cell = {
        (str(item["arm"]), str(item["window_id"])): item
        for item in raw_inventory_value.get("preserved_cells", [])
    }
    expected_cells = [(label, window_id) for label in EXPECTED_ARM_LABELS for window_id in EXPECTED_WINDOW_IDS]
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        require(line.strip(), f"derived evidence manifest contains a blank line {line_number}")
        record = json.loads(line)
        require(isinstance(record, Mapping), f"derived evidence record {line_number} is not an object")
        require(record.get("schema") == "STAGE45F_PI0167_DERIVED_EVIDENCE_REFERENCE_V1", "derived evidence schema drift")
        checkpoint = record.get("checkpoint")
        require(isinstance(checkpoint, Mapping), "derived evidence checkpoint is not an object")
        cell = (str(checkpoint.get("label")), str(record.get("window_id")))
        require(cell in raw_by_cell, f"derived evidence cell is outside the raw matrix: {cell}")
        require(cell not in seen, f"duplicate derived evidence cell: {cell}")
        seen.add(cell)
        raw_entry = raw_by_cell[cell]
        require(record.get("raw_path") == raw_entry["path"], f"derived raw path drift: {cell}")
        require(record.get("raw_file_sha256") == raw_entry["file_sha256"], f"derived raw SHA drift: {cell}")
        derived_path = record.get("derived_path")
        derived_relative = Path(derived_path) if isinstance(derived_path, str) else Path(".")
        require(
            isinstance(derived_path, str)
            and not derived_relative.is_absolute()
            and derived_relative.as_posix() == derived_path
            and all(part not in {"", ".", ".."} for part in derived_relative.parts),
            "derived evidence path is not normalized",
        )
        derived_file = path.parent / derived_path
        require(derived_file.is_file() and not derived_file.is_symlink(), f"derived evidence file is missing: {derived_file}")
        require(record.get("decoded") is True, f"derived evidence decode flag drift: {cell}")
        expect_reused = cell[0] == EXPECTED_REUSED_ARM
        require(record.get("raw_reused_without_model_call") is expect_reused, f"derived evidence model-call provenance drift: {cell}")
        require(record.get("source_allocation") == (EXPECTED_SOURCE_ALLOCATION if expect_reused else RECOVERY_ALLOCATION), f"derived evidence allocation provenance drift: {cell}")
    require(seen == set(expected_cells), "derived evidence does not close the 36-cell matrix")
    return len(seen)


def _earliest_e_access_ref(prior: Mapping[str, Any]) -> dict[str, Any]:
    path = continuation.regular_file(Path(str(prior["access_ledger"])), "earliest Validation-E access ledger")
    observed = sha256_file(path)
    require(observed == str(prior["access_ledger_sha256"]), "earliest Validation-E access ledger SHA drift")
    ledger = load_json(path)
    require(isinstance(ledger, Mapping), "earliest Validation-E access ledger is not an object")
    require(ledger.get("target_id") == EXPECTED_TARGET_ID, "earliest Validation-E target binding drift")
    require(ledger.get("holdout_f_read") is False, "earliest Validation-E ledger opened Holdout-F")
    return {
        "path": str(path),
        "sha256": observed,
        "status": ledger.get("status"),
        "first_payload_read_at_utc": ledger.get("first_payload_read_at_utc"),
        "finished_at_utc": ledger.get("finished_at_utc"),
        "descriptor_reads": ledger.get("descriptor_reads"),
        "frame_reads": ledger.get("frame_reads"),
        "holdout_f_read": ledger.get("holdout_f_read"),
    }


def _source_failure_ref(config: Mapping[str, Any]) -> dict[str, Any]:
    path = continuation.regular_file(Path(str(config["pi0163_recovery_source_failure_path"])), "PI0163 allocation-5 failure seal")
    observed = sha256_file(path)
    require(observed == str(config["pi0163_recovery_source_failure_sha256"]), "PI0163 allocation-5 failure seal SHA drift")
    payload = load_json(path)
    require(isinstance(payload, Mapping), "PI0163 allocation-5 failure seal is not an object")
    require(payload.get("status") == "PARTIAL", "PI0163 allocation-5 failure status drift")
    require(payload.get("model_execution_started") is True and payload.get("holdout_f_read") is False, "PI0163 allocation-5 failure provenance drift")
    return {"path": str(path), "sha256": observed, "phase": payload.get("phase"), "current": payload.get("current"), "actual_counters": payload.get("actual_counters", {})}


def _allocation_failure_ref(
    config: Mapping[str, Any],
    *,
    path_key: str = "pi0163_alloc6_failure_path",
    sha_key: str = "pi0163_alloc6_failure_sha256",
    allocation_label: str = "allocation-6",
    expected_status: str = "BLOCKED",
    require_model_execution: bool = False,
) -> dict[str, Any]:
    ref = _evidence_ref(config, path_key, sha_key)
    payload = load_json(Path(ref["path"]))
    require(isinstance(payload, Mapping), f"PI0163 {allocation_label} failure seal is not an object")
    require(payload.get("status") == expected_status, f"PI0163 {allocation_label} failure status drift")
    if require_model_execution:
        require(payload.get("model_import_started") is True and payload.get("model_execution_started") is True, f"PI0163 {allocation_label} did not cross the model boundary")
    else:
        require(
            payload.get("model_import_started") is False and payload.get("model_execution_started") is False,
            f"PI0163 {allocation_label} crossed the model boundary",
        )
    require(payload.get("training") is False and payload.get("holdout_f_read") is False, f"PI0163 {allocation_label} boundary drift")
    return {
        **ref,
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "error": payload.get("error"),
        "actual_counters": payload.get("actual_counters", {}),
    }


def _validate_allocation9_zero_boundary(failure: Mapping[str, Any]) -> None:
    """Bind allocation 9 as a pre-model, zero-new-call engineering failure."""

    counters = failure.get("actual_counters")
    require(isinstance(counters, Mapping), "allocation-9 failure counters are missing")
    for key in (
        "allocation9_model_loads_attempted",
        "allocation9_model_loads_completed",
        "allocation9_conditioning_attempts",
        "allocation9_conditioning_completed",
        "allocation9_action_head_attempts",
        "allocation9_action_head_completed",
        "allocation9_model_forward_completed",
        "allocation9_raw_prediction_count",
        "allocation9_derived_row_count",
    ):
        require(int(counters.get(key, -1)) == 0, f"allocation-9 new counter drift: {key}")
    require(int(counters.get("raw_prediction_count", -1)) == 6, "allocation-9 preserved raw count drift")
    require(int(counters.get("derived_row_count", -1)) == 0, "allocation-9 derived count drift")


def _install_snapshot_mapping_compat(runtime: Any) -> dict[str, Any]:
    """Bridge the frozen dataclass snapshot to the frozen metric adapter ABI."""

    original = runtime._evaluation_row
    if getattr(original, "_pi0163_snapshot_mapping_compat", False):
        return {
            "schema": "STAGE45F_PI0167_SNAPSHOT_MAPPING_COMPAT_V1",
            "status": "INSTALLED",
            "already_installed": True,
            "source_sha256": sha256_file(Path(__file__)),
        }

    def compatible(raw: Any, physical: Any, details: Mapping[str, Any], window: Mapping[str, Any], checkpoint: Mapping[str, Any], model_runtime_sha256: str, raw_artifact_path: str, raw_artifact_sha256: str) -> dict[str, Any]:
        normalized_window = dict(window)
        snapshot = normalized_window.get("snapshot")
        if not isinstance(snapshot, Mapping):
            physical_action = getattr(snapshot, "physical_action", None)
            require(physical_action is not None, "SameSceneExpertSnapshotV1 lacks physical_action")
            normalized_window["snapshot"] = {
                "expert_physical_action": physical_action,
                "anchor_world_m": normalized_window["anchor_world_m"],
                "target_world_m": normalized_window["target_world_m"],
            }
        return original(raw, physical, details, normalized_window, checkpoint, model_runtime_sha256, raw_artifact_path, raw_artifact_sha256)

    setattr(compatible, "_pi0163_snapshot_mapping_compat", True)
    runtime._evaluation_row = compatible
    return {
        "schema": "STAGE45F_PI0167_SNAPSHOT_MAPPING_COMPAT_V1",
        "status": "INSTALLED",
        "already_installed": False,
        "source_sha256": sha256_file(Path(__file__)),
        "adapted_fields": ["snapshot.physical_action -> snapshot.expert_physical_action", "window.anchor_world_m", "window.target_world_m"],
        "scientific_contract_changed": False,
    }


def static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    forbidden_calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                forbidden_calls.append(name)
    markers = {
        "no_top_level_model_import": not imports.intersection(FORBIDDEN_TOP_LEVEL),
        "valid_raw_reuse": "_copy_file_durable" in source and "source_reuse" in source,
        "mapping_compatibility": "_install_snapshot_mapping_compat" in source and "physical_action" in source,
        "raw_before_derived": "raw_predictions.jsonl" in source and "derived_rows.jsonl" in source,
        "cumulative_accounting": "cumulative_campaign_accounting" in source and "model_load_attempts" in source,
        "actual_raw_inventory": "def raw_inventory" in source and "preserved_raw_count" in source and "missing_raw_count" in source,
        "allocation8_inventory_gate": "validate_allocation8_inventory" in source and "PASS_6_REUSED_30_NEW" in source,
        "completed_arm_model_skip": "first_arm_model_loads" in source and "first_arm_forward_cells" in source and "if label == EXPECTED_REUSED_ARM" in source,
        "six_window_aggregate": "expected_window_ids=evaluator.WINDOW_IDS" in source,
        "pre_submit_expected_head": "def pre_submit" in source and "pre-submit expected_head mismatch" in source,
        "holdout_locked": '"holdout_f_read": False' in source,
        "training_locked": '"training": False' in source,
        "no_adaptive_retry": '"automatic_retry": False' in source,
    }
    return {
        "schema": "STAGE45F_PI0167_RECOVERY_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_imports": sorted(imports),
        "top_level_forbidden_imports": sorted(imports.intersection(FORBIDDEN_TOP_LEVEL)),
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "pbs": 0,
        "training": 0,
        "holdout_f_read": 0,
        "actual_raw_inventory": 6,
        "pre_submit_expected_head": 1,
        "scientific_contract_changed": False,
    }


def _validate_raw_array(raw_path: Path, record: Mapping[str, Any], np: Any) -> Any:
    raw = np.load(raw_path, allow_pickle=False)
    require(tuple(int(item) for item in raw.shape) == EXPECTED_RAW_SHAPE, "reused raw array shape drift")
    require(str(raw.dtype) == "float32" and bool(np.isfinite(raw).all()), "reused raw dtype/finite drift")
    array_sha = sha256_bytes(np.ascontiguousarray(raw).astype(np.float32, copy=False).tobytes())
    require(array_sha == record.get("array_sha256"), "reused raw array SHA drift")
    return np.asarray(raw, dtype=np.float32)


def _append_row_metadata(row: dict[str, Any], arm: Mapping[str, Any], noise_sha: str) -> dict[str, Any]:
    row["route"] = arm["route"]
    row["seed"] = arm["seed"]
    row["tier"] = arm["tier"]
    row["noise_sha256"] = noise_sha
    row["sampler"] = {
        "name": "direct-source-mapped-action-head-E4",
        "t_values": [0.0, 0.25, 0.5, 0.75],
        "timestep_ids": EXPECTED_TIMESTEPS,
        "dt": 0.25,
        "carried_state": True,
        "reset_between_stages": False,
        "official_public_sampler_called": False,
        "get_action_calls": 0,
    }
    row["stats"] = {
        "sha256": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
        "payload_sha256": "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75",
        "route": "FULL24_RUNTIME_V2",
    }
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _seed_pair_summary(paired: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "final_target_distance_delta_m",
        "progress_delta_m",
        "minimum_target_distance_delta_m",
        "path_length_delta_m",
        "absolute_roughness_delta_m",
    )
    result: list[dict[str, Any]] = []
    by_seed = paired.get("paired_by_seed", {})
    for seed in (2222, 2223, 2224):
        rows = list(by_seed.get(str(seed), []))
        result.append({
            "seed": seed,
            "window_count": len(rows),
            "mean_l1_minus_legacy": {field: _mean([float(row[field]) for row in rows]) for field in fields},
            "descriptive_only": True,
        })
    return result


def _one_page_conclusion(*, output_root: Path, expected_head: str, paired: Mapping[str, Any], counters: Mapping[str, Any], cumulative: Mapping[str, Any], reused: Mapping[str, Any]) -> str:
    seed_lines = []
    for item in _seed_pair_summary(paired):
        means = item["mean_l1_minus_legacy"]
        seed_lines.append(
            f"- seed {item['seed']}: n={item['window_count']} windows; "
            f"final-distance delta={means['final_target_distance_delta_m']!r} m; "
            f"progress delta={means['progress_delta_m']!r} m; "
            f"path delta={means['path_length_delta_m']!r} m; "
            f"roughness delta={means['absolute_roughness_delta_m']!r}."
        )
    return "\n".join([
        "# PI0167 Validation-E generation 1 recovery",
        "",
        f"- status: COMPLETED; execution HEAD: {expected_head}",
        f"- output root: {output_root}",
        "- fixed contract: six accepted @2000 checkpoints, six original red_target_holdout_e windows per arm, seed 4242, E4 [0,250,500,750], FULL24 runtime-V2 stats, target excluded from model input.",
        "- raw-first recovery: all six allocation-8 Legacy(seed2222) raw arrays were copied byte-for-byte and decoded on CPU; the completed first arm was not loaded or forwarded by allocation 10.",
        "- interpretation boundary: paired values below are descriptive L1 minus Legacy comparisons at the same tier/window/noise; they do not select a winner and do not claim iid significance.",
        "",
        "## Seed-level paired readout",
        "",
        *seed_lines,
        "",
        "## Accounting",
        "",
        f"- final evidence rows: raw={counters['raw_prediction_count']}, derived={counters['derived_row_count']}; calls: conditioning={counters['conditioning_completed']}, action-head={counters['action_head_completed']}, model-side={counters['conditioning_completed'] + counters['action_head_completed']}.",
        f"- cumulative campaign: allocations={cumulative['allocations_consumed']}/{cumulative['allocation_ceiling']}; model-load attempts={cumulative['model_load_attempts']}/{cumulative['model_load_ceiling']}; model-side calls={cumulative['model_side_calls']}/{cumulative['model_side_call_ceiling']}.",
        f"- reused raw cells: {len(reused['cells'])}; source manifest SHA {reused['source_manifest_sha256']}; allocation-8 failure seal SHA {reused['allocation_8_failure_sha256']}.",
        "- training/backward/optimizer/checkpoint writes/public sampler/get_action/Holdout-F: false/zero.",
        "",
        "## Claim boundary",
        "",
        "This package closes the frozen Validation-E evidence surface only. It does not change recipe, checkpoint, statistics, noise, windows, or decoder; scientific route/tier selection remains with PI.",
        "",
    ])


def execute(config_path: Path) -> int:
    phase = "START"
    output_root: Path | None = None
    config: dict[str, Any] | None = None
    current: dict[str, Any] = {"arm": None, "window_id": None, "step": None, "checkpoint": None}
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_construction_attempts": 0,
        "model_construction_completed": 0,
        # These six/24 counts are the accepted allocation-8 first-arm calls;
        # they are cumulative evidence counts, not calls repeated by allocation 10.
        "conditioning_attempts": 6,
        "conditioning_completed": 6,
        "action_head_attempts": 24,
        "action_head_completed": 24,
        "model_forward_completed": 24,
        "raw_prediction_count": 6,
        "raw_manifest_records": 6,
        "derived_row_count": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "failed_model_calls": 0,
        "allocation10_model_loads_attempted": 0,
        "allocation10_model_loads_completed": 0,
        "allocation10_conditioning_attempts": 0,
        "allocation10_conditioning_completed": 0,
        "allocation10_action_head_attempts": 0,
        "allocation10_action_head_completed": 0,
        "allocation10_model_forward_completed": 0,
        "allocation10_raw_prediction_count": 0,
        "allocation10_derived_row_count": 0,
    }
    try:
        config = continuation._config(config_path.resolve(strict=True))
        require(config.get("pi0167_gate") == PI0167_GATE and config.get("pi0167_pointer") == PI0167_POINTER, "PI0167 authority drift")
        require(
            config.get("latest_controlling_gate") == {
                "pi0165_science": PI0165_SCIENCE_GATE,
                "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                "pi0165a_erratum": PI0165A_ERRATUM,
            },
            "latest controlling gate drift",
        )
        require(config.get("pi0167_recovery_allocation") == RECOVERY_ALLOCATION, "PI0167 recovery allocation drift")
        require(config.get("max_total_allocations") == MAX_TOTAL_ALLOCATIONS, "PI0167 allocation ceiling drift")
        require(config.get("pi0167_support_head") == config.get("expected_head"), "PI0167 support HEAD drift")
        require(config.get("pi0167_max_new_model_loads") == MAX_NEW_MODEL_LOADS, "PI0167 model-load ceiling drift")
        require(config.get("pi0167_max_new_model_side_calls") == MAX_NEW_MODEL_SIDE_CALLS, "PI0167 model-side ceiling drift")
        require(HISTORICAL_MODEL_LOAD_ATTEMPTS + MAX_NEW_MODEL_LOADS <= MAX_TOTAL_MODEL_LOAD_ATTEMPTS, "PI0167 cumulative model-load ceiling drift")
        require(config.get("pi0163_previous_job_id") == EXPECTED_JOB, "PI0167 previous job drift")
        require(config.get("pi0163_alloc7_failure_job_id") == EXPECTED_ALLOCATION_7_FAILURE_JOB, "PI0163 allocation-7 failure job drift")
        output_root = Path(str(config["output_root"])).resolve()
        source_root = continuation.regular_directory(Path(str(config["pi0163_recovery_source_root"])), "PI0167 recovery source root")
        require(not output_root.exists() and not output_root.is_symlink(), "PI0167 recovery output root must be fresh")
        require(output_root.parent.is_dir(), "PI0163 recovery output parent is unavailable")
        output_root.mkdir(parents=False, exist_ok=False)
        phase = "RECOVERY_SOURCE_AUDIT"
        source_failure = _source_failure_ref(config)
        allocation_6_failure = _allocation_failure_ref(config)
        allocation_7_failure = _allocation_failure_ref(
            config,
            path_key="pi0163_alloc7_failure_path",
            sha_key="pi0163_alloc7_failure_sha256",
            allocation_label="allocation-7",
        )
        allocation8_failure = _allocation_failure_ref(
            config,
            path_key="pi0167_alloc8_failure_path",
            sha_key="pi0167_alloc8_failure_sha256",
            allocation_label="allocation-8",
            expected_status="PARTIAL",
            require_model_execution=True,
        )
        failure_counters = allocation8_failure["actual_counters"]
        require(failure_counters.get("model_loads_attempted") == 1, "allocation-8 model-load attempt drift")
        require(failure_counters.get("model_loads_completed") == 1, "allocation-8 model-load completion drift")
        require(failure_counters.get("raw_prediction_count") == 6, "allocation-8 raw count drift")
        require(failure_counters.get("derived_row_count") == 6, "allocation-8 derived count drift")
        allocation9_failure = _allocation_failure_ref(
            config,
            path_key="pi0167_alloc9_failure_path",
            sha_key="pi0167_alloc9_failure_sha256",
            allocation_label="allocation-9",
            expected_status="BLOCKED",
            require_model_execution=False,
        )
        _validate_allocation9_zero_boundary(allocation9_failure)
        actual_raw_inventory = validate_allocation8_inventory(raw_inventory_isolated(source_root))
        execution_plan = recovery_execution_plan(actual_raw_inventory)
        write_json(output_root / "stage45f_pi0167_allocation8_raw_inventory.json", actual_raw_inventory)
        source_inventory_path = continuation.regular_file(Path(str(config["pi0167_recovery_source_inventory_path"])), "PI0167 allocation-8 raw inventory")
        require(sha256_file(source_inventory_path) == str(config["pi0167_recovery_source_inventory_sha256"]), "PI0167 allocation-8 inventory artifact SHA drift")
        source_inventory = load_json(source_inventory_path)
        require(isinstance(source_inventory, Mapping), "PI0167 allocation-8 inventory artifact is not an object")
        require(source_inventory.get("status") == "PASS_ACTUAL_ALLOCATION8_RAW_INVENTORY", "PI0167 allocation-8 inventory artifact status drift")
        require(source_inventory.get("raw_manifest_sha256") == actual_raw_inventory["raw_manifest_sha256"], "PI0167 allocation-8 inventory artifact manifest SHA drift")
        require(source_inventory.get("preserved_raw_count") == actual_raw_inventory["preserved_raw_count"], "PI0167 allocation-8 inventory artifact preserved count drift")
        require(source_inventory.get("missing_raw_count") == actual_raw_inventory["missing_raw_count"], "PI0167 allocation-8 inventory artifact missing count drift")
        require(_inventory_cell_identity(source_inventory) == _inventory_cell_identity(actual_raw_inventory), "PI0167 allocation-8 inventory artifact cell/hash drift")
        require(source_inventory.get("actual_raw_paths") == actual_raw_inventory["actual_raw_paths"], "PI0167 allocation-8 inventory artifact path drift")
        source_manifest = continuation.regular_file(Path(str(config["pi0163_recovery_source_raw_manifest_path"])), "PI0167 allocation-8 raw manifest")
        require(sha256_file(source_manifest) == str(config["pi0163_recovery_source_raw_manifest_sha256"]), "PI0167 recovery source manifest SHA drift")
        preserved_records = _preserved_raw_records(actual_raw_inventory)
        prior_alloc5_bootstrap = _evidence_ref(config, "pi0163_alloc5_bootstrap_record", "pi0163_alloc5_bootstrap_sha256")
        prior_alloc5_import = _evidence_ref(config, "pi0163_alloc5_compute_import_record", "pi0163_alloc5_compute_import_sha256")
        phase = "PRIOR_ADMISSION_RECOVERY_AUDIT"
        prior, _ = continuation.validate_prior_admission(config)
        earliest_e_access = _earliest_e_access_ref(prior)
        phase = "OFFLINE_CACHE_RESOLUTION"
        runtime_binding = continuation.bind_offline_runtime(config, output_root)
        phase = "CONTINUATION_ADMISSION"
        binding = continuation.effective_binding(config, prior, output_root)
        binding["pi0167_recovery"] = {
            "gate": PI0167_GATE,
            "pointer": PI0167_POINTER,
            "latest_controlling_gate": {
                "pi0165_science": PI0165_SCIENCE_GATE,
                "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                "pi0165a_erratum": PI0165A_ERRATUM,
            },
            "allocation_ordinal": RECOVERY_ALLOCATION,
            "previous_job_id": EXPECTED_JOB,
            "source_root": str(source_root),
            "source_failure": source_failure,
            "allocation_6_failure": allocation_6_failure,
            "allocation_7_failure": allocation_7_failure,
            "allocation_8_failure": allocation8_failure,
            "allocation_9_failure": allocation9_failure,
            "raw_inventory": actual_raw_inventory,
            "source_inventory_artifact": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
            },
            "earliest_e_access": earliest_e_access,
            "execution_plan": execution_plan,
            "valid_raw_reused": 6,
            "scientific_contract_changed": False,
        }
        continuation_binding_path = output_root / "stage45f_pi0167_recovery_admission.json"
        write_json(continuation_binding_path, binding)
        effective_config = dict(config)
        effective_config.update({
            "admission_path": str(continuation_binding_path),
            "output_root": str(output_root),
            "prior_admission_path": prior["path"],
        })
        effective_config_path = output_root / "stage45f_pi0167_effective_runner_config.json"
        write_json(effective_config_path, effective_config)
        source_reuse = {
            "schema": "STAGE45F_PI0167_VALID_RAW_REUSE_V1",
            "status": "PASS_BYTE_PRESERVED",
            "source_root": str(source_root),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_inventory_artifact_path": str(source_inventory_path),
            "source_inventory_artifact_sha256": sha256_file(source_inventory_path),
            "allocation_8_failure_path": allocation8_failure["path"],
            "allocation_8_failure_sha256": allocation8_failure["sha256"],
            "cells": [],
            "reused_raw_queries": 6,
            "model_loads_for_reused_arm": 0,
            "model_forwards_for_reused_arm": 0,
            "scientific_contract_changed": False,
        }
        _copy_exact_text_line(source_manifest, output_root / "source_raw_manifest.jsonl")
        _copy_exact_text_line(source_manifest, output_root / "raw_predictions.jsonl")
        for key in [(EXPECTED_REUSED_ARM, window_id) for window_id in EXPECTED_WINDOW_IDS]:
            preserved = preserved_records[key]
            source_path = source_root / str(preserved["path"])
            destination_path = output_root / str(preserved["path"])
            _copy_file_durable(source_path, destination_path)
            require(sha256_file(destination_path) == str(preserved["file_sha256"]), f"preserved raw bytes changed during copy: {key}")
            provenance = {
                "source_allocation": EXPECTED_SOURCE_ALLOCATION,
                "source_job_id": EXPECTED_JOB,
                "source_path": str(source_path),
                "source_file_sha256": str(preserved["file_sha256"]),
                "source_array_sha256": str(preserved["array_sha256"]),
                "path": str(preserved["path"]),
                "arm": key[0],
                "window_id": key[1],
                "shape": list(preserved["shape"]),
                "dtype": str(preserved["dtype"]),
                "timestep_ids": list(preserved["timestep_ids"]),
                "source_record": dict(preserved["record"]),
            }
            source_reuse["cells"].append(provenance)
        write_json(output_root / "stage45f_pi0167_raw_reuse.json", source_reuse)
        preflight = {
            "schema": "STAGE45F_PI0167_RECOVERY_PRE_MODEL_ADMISSION_V1",
            "status": "PASS_PRE_MODEL",
            "pi_gate": PI0167_GATE,
            "pointer": PI0167_POINTER,
            "authority_token": PI0167_TOKEN,
            "latest_controlling_gate": {
                "pi0165_science": PI0165_SCIENCE_GATE,
                "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                "pi0165a_erratum": PI0165A_ERRATUM,
            },
            "execution_head": config["expected_head"],
            "execution_branch": continuation.EXPECTED_BRANCH,
            "recovery_allocation": RECOVERY_ALLOCATION,
            "previous_job_id": EXPECTED_JOB,
            "prior_admission": {key: value for key, value in prior.items() if key not in {"binding"}},
            "runtime_binding": runtime_binding,
            "effective_binding_path": str(continuation_binding_path),
            "effective_binding_sha256": sha256_file(continuation_binding_path),
            "effective_runner_config_path": str(effective_config_path),
            "effective_runner_config_sha256": sha256_file(effective_config_path),
            "source_failure": source_failure,
            "allocation_6_failure": allocation_6_failure,
            "allocation_7_failure": allocation_7_failure,
            "allocation_8_failure": allocation8_failure,
            "allocation_9_failure": allocation9_failure,
            "raw_inventory": actual_raw_inventory,
            "raw_inventory_path": str(output_root / "stage45f_pi0167_allocation8_raw_inventory.json"),
            "raw_inventory_sha256": sha256_file(output_root / "stage45f_pi0167_allocation8_raw_inventory.json"),
            "source_inventory_artifact": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
            },
            "earliest_e_access": earliest_e_access,
            "recovery_execution_plan": execution_plan,
            "source_raw_reuse": source_reuse,
            "allocation_5_bootstrap": prior_alloc5_bootstrap,
            "allocation_5_compute_import": prior_alloc5_import,
            "nominal_calls": EXPECTED_NOMINAL,
            "new_calls_after_reuse": {"conditioning": 30, "action_head": 120, "model_side_total": 150},
            "cumulative_call_ceiling": {"conditioning": 36, "action_head": 144, "model_side_total": 180},
            "model_load_accounting": {"historical_attempts_through_allocation_8": HISTORICAL_MODEL_LOAD_ATTEMPTS, "allocation_10_reserved": MAX_NEW_MODEL_LOADS, "cumulative_reserved": HISTORICAL_MODEL_LOAD_ATTEMPTS + MAX_NEW_MODEL_LOADS, "cumulative_ceiling": MAX_TOTAL_MODEL_LOAD_ATTEMPTS},
            "holdout_f_read": False,
            "training": False,
            "model_imported": False,
            "scientific_contract_changed": False,
        }
        write_json(output_root / "stage45f_pi0167_pre_model_admission.json", preflight)
        guard = static_guard()
        require(guard["status"] == "PASS", "PI0163 recovery static guard failed")
        write_json(output_root / "stage45f_pi0167_recovery_static_guard.json", guard)
        phase = "PRE_MODEL_ARTIFACT_AUDIT"
        import stage45f_vanda_validation_e_gen1 as evaluator

        evaluator_guard = evaluator.static_guard()
        require(evaluator_guard["status"] == "PASS", "frozen evaluator static guard failed")
        checkout = evaluator.observe_checkout(str(config["expected_head"]))
        gpu = evaluator.observe_gpu()
        runtime_identity = evaluator.observed_runtime(gpu)
        artifact_audit, parent_imports_before_validate_artifacts = _validate_frozen_artifacts_from_clean_parent(
            evaluator,
            effective_config,
            binding,
            str(config["expected_head"]),
        )
        windows_meta = evaluator.validate_snapshot_records(binding)
        write_json(output_root / "stage45f_execution_static_guard.json", evaluator_guard)
        write_json(output_root / "stage45f_pre_model_artifact_audit.json", artifact_audit)
        write_json(output_root / "stage45f_pi0167_import_purity.json", {
            "schema": "STAGE45F_PI0167_IMPORT_PURITY_EVIDENCE_V1",
            "status": "PASS_CLEAN_PARENT_BEFORE_FROZEN_ARTIFACT_ADMISSION",
            "raw_inventory_verifier": str(RAW_INVENTORY_VERIFIER),
            "raw_inventory_verifier_isolated_process": True,
            "parent_forbidden_imports_immediately_before_validate_artifacts": parent_imports_before_validate_artifacts,
            "numpy_in_parent_after_isolated_verification": "numpy" in sys.modules,
            "frozen_validate_artifacts_completed": True,
            "scientific_contract_changed": False,
        })
        write_json(output_root / "stage45f_observed_runtime.json", {"checkout": checkout, "gpu": gpu, "runtime": runtime_identity})
        phase = "MODEL_IMPORT_AND_PROCESSOR_CONSTRUCTION"
        counters["model_import_started"] = True
        import numpy as np
        import torch
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

        compatibility = _install_snapshot_mapping_compat(runtime)
        write_json(output_root / "stage45f_pi0167_snapshot_compatibility.json", compatibility)
        loader = runtime._load_torch_runtime()[1]
        runtime_processor, embodiment, _modality_configs = stage26._build_runtime_processor(
            groot_source=Path(str(config["groot_source"])).resolve(),
            base_checkpoint=Path(str(config["base_model"])).resolve(),
            dataset_dir=Path(str(config["dataset"])).resolve(),
            modality_path=Path(str(config["modality"])).resolve(),
            cosmos_snapshot=Path(str(config["cosmos_root"])).resolve() / "snapshots" / "9ce19a195e423419c349abfc86fd07178b230561",
        )
        windows = evaluator._prepare_windows(runtime_processor, embodiment, windows_meta)
        require(len(windows) == 6, "prepared E window count drift")
        noise_bank, noise_hashes = evaluator._make_noise_bank(torch)
        for index, window_id in enumerate(EXPECTED_WINDOW_IDS):
            preserved = preserved_records[(EXPECTED_REUSED_ARM, window_id)]
            require(preserved["record"].get("noise_sha256") == noise_hashes[index], f"reused raw noise identity drift: {window_id}")
        write_json(output_root / "sampler_noise_manifest.json", {
            "schema": "STAGE45F_PI0154_CPU_FLOAT32_NOISE_BANK_V1",
            "seed": evaluator.admission.EXPECTED_NOISE_SEED,
            "device": "cpu",
            "dtype": "float32",
            "shape": list(evaluator.RAW_SHAPE),
            "window_ids": list(evaluator.WINDOW_IDS),
            "noise_sha256_by_window": noise_hashes,
            "same_noise_across_all_six_checkpoint_arms": True,
            "reused_raw_window_indices": list(range(6)),
        })
        derived_manifest = output_root / "derived_rows.jsonl"
        arms = binding["checkpoint_arms"]
        require([(arm["route"], int(arm["seed"])) for arm in arms] == list(EXPECTED_ARM_KEYS), "recovery arm order drift")
        checkpoint_paths = config["checkpoints"]
        rows_by_arm: dict[str, list[dict[str, Any]]] = {str(arm["label"]): [] for arm in arms}
        runtime_sha = evaluator.canonical_sha({"binding": binding, "artifact_audit": artifact_audit, "runtime": runtime_identity})
        reused_arm = next(arm for arm in arms if arm["label"] == EXPECTED_REUSED_ARM)
        for index, reused_window in enumerate(windows):
            window_id = str(reused_window["window_id"])
            preserved = preserved_records[(EXPECTED_REUSED_ARM, window_id)]
            raw_path = output_root / str(preserved["path"])
            raw_reused = _validate_raw_array(raw_path, preserved["record"], np)
            physical, details = runtime._decode_v2(
                raw_reused,
                reused_window["state_v1"],
                Path(str(config["stats"])).resolve(),
                reused_window["source_timestamps"],
                float(reused_window["observation_timestamp"]),
                window_id + ":" + EXPECTED_REUSED_ARM,
            )
            reused_row = evaluator._metric_row(
                runtime,
                raw_reused,
                physical,
                details,
                reused_window,
                {"label": EXPECTED_REUSED_ARM, "tree_sha256": reused_arm["tree_sha256"]},
                runtime_sha,
                str(preserved["path"]),
                str(preserved["file_sha256"]),
            )
            _append_row_metadata(reused_row, reused_arm, noise_hashes[index])
            reused_derived_path = output_root / "derived" / _label_path(EXPECTED_REUSED_ARM) / f"window-{index:02d}.json"
            evaluator.write_json_durable(reused_derived_path, reused_row)
            evaluator.append_jsonl_durable(derived_manifest, {
                "schema": "STAGE45F_PI0167_DERIVED_EVIDENCE_REFERENCE_V1",
                "checkpoint": dict(reused_arm),
                "window_id": window_id,
                "raw_path": str(preserved["path"]),
                "raw_file_sha256": str(preserved["file_sha256"]),
                "derived_path": str(reused_derived_path.relative_to(output_root)),
                "decoded": True,
                "decoded_array_sha256": reused_row["decoded_action"]["array_sha256"],
                "raw_reused_without_model_call": True,
                "source_allocation": EXPECTED_SOURCE_ALLOCATION,
            })
            rows_by_arm[EXPECTED_REUSED_ARM].append(reused_row)
        counters["raw_prediction_count"] = 6
        counters["raw_manifest_records"] = 6
        counters["derived_row_count"] = 6
        phase = "MODEL_EXECUTION_RECOVERY"
        for arm in arms:
            label = str(arm["label"])
            current.update({"arm": label, "checkpoint": label, "window_id": None, "step": None})
            action_rows = rows_by_arm[label]
            if label == EXPECTED_REUSED_ARM:
                require(len(action_rows) == 6, "completed first arm raw/derived row count drift")
                aggregate = aggregate_evaluation_rows(action_rows, expected_window_ids=evaluator.WINDOW_IDS)
                aggregate["route"] = arm["route"]
                aggregate["seed"] = arm["seed"]
                aggregate["tier"] = arm["tier"]
                aggregate["checkpoint_tree_sha256"] = arm["tree_sha256"]
                aggregate["model_loads"] = 0
                aggregate["model_forwards"] = 0
                aggregate["raw_reused_without_model_call"] = True
                evaluator.write_json_durable(output_root / f"aggregate_{_label_path(label)}.json", aggregate)
                continue
            phase = "MODEL_LOAD"
            require(counters["model_loads_attempted"] < MAX_NEW_MODEL_LOADS, "allocation-10 model-load plan exceeded")
            counters["model_loads_attempted"] += 1
            counters["allocation10_model_loads_attempted"] += 1
            model = loader(Path(str(checkpoint_paths[label])).resolve(), "cuda:0")
            counters["model_loads_completed"] += 1
            counters["allocation10_model_loads_completed"] += 1
            counters["model_construction_attempts"] += 1
            counters["model_construction_completed"] += 1
            model.eval()
            try:
                for index in range(len(windows)):
                    window = windows[index]
                    current["window_id"] = window["window_id"]
                    current["step"] = None
                    phase = "CONDITIONING"
                    counters["conditioning_attempts"] += 1
                    counters["allocation10_conditioning_attempts"] += 1
                    _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                    counters["conditioning_completed"] += 1
                    counters["allocation10_conditioning_completed"] += 1
                    current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                    with torch.inference_mode():
                        for step, timestep in enumerate(evaluator.TIMESTEP_IDS):
                            current["step"] = int(step)
                            phase = "ACTION_HEAD"
                            counters["action_head_attempts"] += 1
                            counters["allocation10_action_head_attempts"] += 1
                            try:
                                velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                                counters["action_head_completed"] += 1
                                counters["model_forward_completed"] += 1
                                counters["allocation10_action_head_completed"] += 1
                                counters["allocation10_model_forward_completed"] += 1
                            except Exception:
                                counters["failed_model_calls"] += 1
                                raise
                            current_action = current_action + 0.25 * velocity
                    phase = "RAW_SEAL"
                    raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                    evaluator.require(raw_np.shape == evaluator.RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output shape/finite drift: {label}:{window['window_id']}")
                    raw_path = output_root / "raw" / _label_path(label) / f"window-{index:02d}.npy"
                    require(not raw_path.exists() and not raw_path.is_symlink(), f"new raw path is not fresh: {raw_path}")
                    evaluator.write_npy_durable(raw_path, raw_np)
                    raw_file_sha = evaluator.sha256_file(raw_path)
                    raw_array_sha = evaluator.sha256_bytes(np.ascontiguousarray(raw_np).tobytes())
                    evaluator.append_jsonl_durable(output_root / "raw_predictions.jsonl", {
                        "schema": EXPECTED_RAW_SCHEMA,
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "path": str(raw_path.relative_to(output_root)),
                        "file_sha256": raw_file_sha,
                        "array_sha256": raw_array_sha,
                        "shape": list(raw_np.shape),
                        "dtype": str(raw_np.dtype),
                        "noise_sha256": noise_hashes[index],
                        "timestep_ids": list(evaluator.TIMESTEP_IDS),
                        "action_head_calls_observed": len(evaluator.TIMESTEP_IDS),
                        "decoded": False,
                        "source_allocation": RECOVERY_ALLOCATION,
                        "unique_query": True,
                    })
                    counters["raw_prediction_count"] += 1
                    counters["raw_manifest_records"] += 1
                    counters["allocation10_raw_prediction_count"] += 1
                    phase = "DECODE_AND_DERIVED_METRICS"
                    physical, details = runtime._decode_v2(
                        raw_np,
                        window["state_v1"],
                        Path(str(config["stats"])).resolve(),
                        window["source_timestamps"],
                        float(window["observation_timestamp"]),
                        window["window_id"] + ":" + label,
                    )
                    row = evaluator._metric_row(runtime, raw_np, physical, details, window, {"label": label, "tree_sha256": arm["tree_sha256"]}, runtime_sha, str(raw_path.relative_to(output_root)), raw_file_sha)
                    _append_row_metadata(row, arm, noise_hashes[index])
                    derived_path = output_root / "derived" / _label_path(label) / f"window-{index:02d}.json"
                    evaluator.write_json_durable(derived_path, row)
                    evaluator.append_jsonl_durable(derived_manifest, {
                        "schema": "STAGE45F_PI0167_DERIVED_EVIDENCE_REFERENCE_V1",
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "raw_path": str(raw_path.relative_to(output_root)),
                        "raw_file_sha256": raw_file_sha,
                        "derived_path": str(derived_path.relative_to(output_root)),
                        "decoded": True,
                        "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                        "raw_reused_without_model_call": False,
                        "source_allocation": RECOVERY_ALLOCATION,
                    })
                    action_rows.append(row)
                    counters["derived_row_count"] += 1
                    counters["allocation10_derived_row_count"] += 1
                require(len(action_rows) == 6, f"per-arm row count drift: {label}")
            finally:
                del model
                gc = __import__("gc")
                gc.collect()
                torch.cuda.empty_cache()
            rows_by_arm[label] = action_rows
            # The frozen Validation-E recovery cohort has six red_target_holdout_e
            # windows per arm.  checkpoint_eval's default is the unrelated
            # 24-window FULL24 train cohort, so bind the already validated
            # six-window list explicitly at this mechanical boundary.
            aggregate = aggregate_evaluation_rows(action_rows, expected_window_ids=evaluator.WINDOW_IDS)
            aggregate["route"] = arm["route"]
            aggregate["seed"] = arm["seed"]
            aggregate["tier"] = arm["tier"]
            aggregate["checkpoint_tree_sha256"] = arm["tree_sha256"]
            aggregate["model_loads"] = 1
            aggregate["model_forwards"] = 24
            aggregate["raw_reused_without_model_call"] = False
            evaluator.write_json_durable(output_root / f"aggregate_{_label_path(label)}.json", aggregate)
        final_raw_inventory = validate_complete_raw_inventory(raw_inventory_isolated(output_root))
        final_raw_inventory_path = output_root / "stage45f_pi0167_final_raw_inventory.json"
        write_json(final_raw_inventory_path, final_raw_inventory)
        final_derived_count = validate_complete_derived_manifest(derived_manifest, final_raw_inventory)
        require(final_derived_count == EXPECTED_NOMINAL["raw_outputs"], "final derived evidence count drift")
        require(counters["model_loads_attempted"] == EXPECTED_RECOVERY["model_loads"] and counters["model_loads_completed"] == EXPECTED_RECOVERY["model_loads"], "recovery model-load count drift")
        require(counters["allocation10_model_loads_attempted"] == MAX_NEW_MODEL_LOADS and counters["allocation10_model_loads_completed"] == MAX_NEW_MODEL_LOADS, "allocation-10 model-load count drift")
        require(counters["allocation10_conditioning_attempts"] == EXPECTED_RECOVERY["conditioning"], "allocation-10 conditioning attempt count drift")
        require(counters["allocation10_conditioning_completed"] == EXPECTED_RECOVERY["conditioning"], "allocation-10 conditioning count drift")
        require(counters["allocation10_action_head_attempts"] == EXPECTED_RECOVERY["action_head"], "allocation-10 action-head attempt count drift")
        require(counters["allocation10_action_head_completed"] == EXPECTED_RECOVERY["action_head"], "allocation-10 action-head count drift")
        require(counters["allocation10_model_forward_completed"] == EXPECTED_RECOVERY["action_head"], "allocation-10 model-forward count drift")
        require(counters["allocation10_raw_prediction_count"] == EXPECTED_RECOVERY["new_queries"], "allocation-10 raw count drift")
        require(counters["allocation10_derived_row_count"] == EXPECTED_RECOVERY["new_queries"], "allocation-10 derived count drift")
        require(counters["conditioning_completed"] == EXPECTED_NOMINAL["conditioning"], "cumulative conditioning count drift")
        require(counters["action_head_completed"] == EXPECTED_NOMINAL["action_head"], "cumulative action-head count drift")
        require(counters["model_forward_completed"] == EXPECTED_NOMINAL["action_head"], "cumulative model-forward count drift")
        require(counters["raw_prediction_count"] == EXPECTED_NOMINAL["raw_outputs"] and counters["raw_manifest_records"] == EXPECTED_NOMINAL["raw_outputs"], "recovery raw count drift")
        require(counters["derived_row_count"] == EXPECTED_NOMINAL["raw_outputs"], "recovery derived count drift")
        phase = "CPU_PAIRED_SUMMARY"
        paired = evaluator.paired_summary(rows_by_arm, arms)
        seed_summary = _seed_pair_summary(paired)
        evaluator.write_json_durable(output_root / "paired_summary.json", paired)
        cpu_summary = {
            "schema": "STAGE45F_PI0167_VALIDATION_E_CPU_SUMMARY_V1",
            "status": "PASS_READONLY_DERIVED_FROM_SEALED_RAW",
            "checkpoint_order": [str(arm["label"]) for arm in arms],
            "window_ids": list(evaluator.WINDOW_IDS),
            "window_count_per_arm": 6,
            "arms": 6,
            "paired_summary": paired,
            "seed_paired_summary": seed_summary,
            "aggregate_files": [f"aggregate_{_label_path(str(arm['label']))}.json" for arm in arms],
            "raw_predictions": str(output_root / "raw_predictions.jsonl"),
            "derived_rows": str(derived_manifest),
            "raw_reuse": source_reuse,
            "final_raw_inventory_path": str(final_raw_inventory_path),
            "final_raw_inventory_sha256": sha256_file(final_raw_inventory_path),
            "final_raw_inventory": final_raw_inventory,
            "final_derived_count": final_derived_count,
            "descriptive_only": True,
            "no_iid_claim": True,
        }
        evaluator.write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        cumulative = {
            "schema": "STAGE45F_PI0167_CUMULATIVE_CAMPAIGN_ACCOUNTING_V1",
            "allocations_consumed": RECOVERY_ALLOCATION,
            "allocation_ceiling": MAX_TOTAL_ALLOCATIONS,
            "allocations": [
                {"job_id": "614203.hopper-m-02", "ordinal": 1, "status": "FAILED_BEFORE_PYTHON_MODEL_BOUNDARY", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614226.hopper-m-02", "ordinal": 2, "status": "PARTIAL_FAILED_AT_MODEL_LOAD", "model_load_attempts": 1, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614574.hopper-m-02", "ordinal": 3, "status": "FAILED_BEFORE_MODEL_IMPORT", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614586.hopper-m-02", "ordinal": 4, "status": "FAILED_BEFORE_MODEL_IMPORT", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0},
                {"job_id": "614704.hopper-m-02", "ordinal": 5, "status": "PARTIAL_AFTER_VALID_RAW", "model_load_attempts": 1, "model_loads_completed": 1, "conditioning": 1, "action_head": 4, "raw_rows": 1, "derived_rows": 0, "failure_phase": source_failure["phase"]},
                {"job_id": "614858.hopper-m-02", "ordinal": 6, "status": "BLOCKED_STALE_EXPECTED_HEAD", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0, "derived_rows": 0, "failure_phase": allocation_6_failure["phase"]},
                {"job_id": EXPECTED_ALLOCATION_7_FAILURE_JOB, "ordinal": 7, "status": "BLOCKED_PBS_PRE_SUBMIT_CHECKOUT_DIRTY", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 0, "derived_rows": 0, "failure_phase": allocation_7_failure["phase"]},
                {"job_id": EXPECTED_JOB, "ordinal": 8, "status": "PARTIAL_EVALUATION_CONTRACT", "model_load_attempts": 1, "model_loads_completed": 1, "conditioning": 6, "action_head": 24, "raw_rows": 6, "derived_rows": 6, "failure_phase": allocation8_failure["phase"]},
                {"job_id": EXPECTED_ALLOCATION_9_FAILURE_JOB, "ordinal": 9, "status": "BLOCKED_PRE_MODEL_IMPORT_PURITY", "model_load_attempts": 0, "model_loads_completed": 0, "conditioning": 0, "action_head": 0, "raw_rows": 6, "derived_rows": 0, "failure_phase": allocation9_failure["phase"]},
                {"job_id": os.environ.get("PBS_JOBID"), "ordinal": RECOVERY_ALLOCATION, "status": "COMPLETED_RECOVERY", "model_load_attempts": counters["allocation10_model_loads_attempted"], "model_loads_completed": counters["allocation10_model_loads_completed"], "conditioning_attempts": counters["allocation10_conditioning_attempts"], "conditioning": counters["allocation10_conditioning_completed"], "action_head_attempts": counters["allocation10_action_head_attempts"], "action_head": counters["allocation10_action_head_completed"], "model_forward": counters["allocation10_model_forward_completed"], "raw_rows": counters["allocation10_raw_prediction_count"], "derived_rows": counters["allocation10_derived_row_count"], "reused_raw_rows": 6},
            ],
            "model_load_attempts": HISTORICAL_MODEL_LOAD_ATTEMPTS + MAX_NEW_MODEL_LOADS,
            "model_loads_completed": 7,
            "model_load_ceiling": MAX_TOTAL_MODEL_LOAD_ATTEMPTS,
            "conditioning": counters["conditioning_completed"],
            "action_head": counters["action_head_completed"],
            "model_side_calls": counters["conditioning_completed"] + counters["action_head_completed"],
            "model_side_call_ceiling": 180,
            "raw_rows": counters["raw_prediction_count"],
            "derived_rows": counters["derived_row_count"],
            "valid_raw_reused": 6,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "holdout_f_reads": 0,
            "automatic_retries": 0,
        }
        write_json(output_root / "cumulative_campaign_accounting.json", cumulative)
        write_json(output_root / "telemetry.json", {
            "schema": "STAGE45F_PI0167_H200_TELEMETRY_V1",
            "gpu": gpu,
            "runtime": runtime_identity,
            "counters": counters,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device("cuda:0"))),
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
        })
        conclusion = _one_page_conclusion(output_root=output_root, expected_head=str(config["expected_head"]), paired=paired, counters=counters, cumulative=cumulative, reused=source_reuse)
        conclusion_path = output_root / "pi0167_one_page_conclusion.md"
        conclusion_path.write_text(conclusion, encoding="utf-8")
        with conclusion_path.open("rb") as handle:
            os.fsync(handle.fileno())
        excluded = {
            "evidence_inventory.json",
            "stage45f_validation_e_seal.json",
            "stage45f_validation_e_result.json",
            "stage45f_pi0167_recovery_failure.json",
        }
        inventory_path, inventory_sha, inventory_count = evaluator._inventory(output_root, excluded)
        seal = {
            "schema": "STAGE45F_PI0167_VALIDATION_E_GEN1_RECOVERY_SEAL_V1",
            "status": "PASS",
            "token": RETURN_TOKEN,
            "pi_gate": PI0167_GATE,
            "pr_pointer": PI0167_POINTER,
            "authority_token": PI0167_TOKEN,
            "execution_head": config["expected_head"],
            "execution_branch": continuation.EXPECTED_BRANCH,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "runtime": runtime_identity,
            "recovery_allocation": RECOVERY_ALLOCATION,
            "previous_job_id": EXPECTED_JOB,
            "previous_failure": source_failure,
            "allocation_8_failure": allocation8_failure,
            "allocation_9_failure": allocation9_failure,
            "source_raw_reuse": source_reuse,
            "source_inventory_artifact": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
            },
            "earliest_e_access": earliest_e_access,
            "snapshot_compatibility": compatibility,
            "checkpoint_arms": [dict(arm) for arm in arms],
            "target_id": EXPECTED_TARGET_ID,
            "snapshot_ids": list(evaluator.WINDOW_IDS),
            "snapshot_count": 6,
            "checkpoint_count": 6,
            "stats_sha256": config["stats_sha256"],
            "noise_seed": config["noise_seed"],
            "timestep_ids": config["timestep_ids"],
            "t_values": config["t_values"],
            "nominal_calls": EXPECTED_NOMINAL,
            "actual_counters": counters,
            "cumulative_campaign_accounting": cumulative,
            "latest_controlling_gate": {
                "pi0165_science": PI0165_SCIENCE_GATE,
                "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                "pi0165a_erratum": PI0165A_ERRATUM,
            },
            "allocation_5_bootstrap": prior_alloc5_bootstrap,
            "allocation_5_compute_import": prior_alloc5_import,
            "allocation_6_failure": allocation_6_failure,
            "allocation_7_failure": allocation_7_failure,
            "allocation_9_failure": allocation9_failure,
            "raw_inventory_path": str(output_root / "stage45f_pi0167_allocation8_raw_inventory.json"),
            "raw_inventory_sha256": sha256_file(output_root / "stage45f_pi0167_allocation8_raw_inventory.json"),
            "raw_inventory": actual_raw_inventory,
            "final_raw_inventory_path": str(final_raw_inventory_path),
            "final_raw_inventory_sha256": sha256_file(final_raw_inventory_path),
            "final_raw_inventory": final_raw_inventory,
            "final_derived_count": final_derived_count,
            "admission_path": str(continuation_binding_path),
            "admission_sha256": sha256_file(continuation_binding_path),
            "pre_model_admission_path": str(output_root / "stage45f_pi0167_pre_model_admission.json"),
            "pre_model_admission_sha256": sha256_file(output_root / "stage45f_pi0167_pre_model_admission.json"),
            "import_purity_path": str(output_root / "stage45f_pi0167_import_purity.json"),
            "import_purity_sha256": sha256_file(output_root / "stage45f_pi0167_import_purity.json"),
            "raw_manifest": str(output_root / "raw_predictions.jsonl"),
            "raw_manifest_sha256": sha256_file(output_root / "raw_predictions.jsonl"),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "paired_summary": str(output_root / "paired_summary.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "conclusion_path": str(conclusion_path),
            "conclusion_sha256": sha256_file(conclusion_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "raw_before_derived": True,
            "allocation10_new_counters": {key: value for key, value in counters.items() if key.startswith("allocation10_")},
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
            "automatic_retry": False,
            "scientific_contract_changed": False,
        }
        seal_path = output_root / "stage45f_validation_e_seal.json"
        write_json(seal_path, seal)
        result = {
            "schema": "STAGE45F_PI0167_VALIDATION_E_GEN1_RECOVERY_RESULT_V1",
            "status": "COMPLETED",
            "token": RETURN_TOKEN,
            "execution_head": config["expected_head"],
            "branch": continuation.EXPECTED_BRANCH,
            "seal_path": str(seal_path),
            "seal_sha256": sha256_file(seal_path),
            "output_root": str(output_root),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "paired_summary": str(output_root / "paired_summary.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "conclusion_path": str(conclusion_path),
            "conclusion_sha256": sha256_file(conclusion_path),
            "actual_counters": counters,
            "allocation10_new_counters": {key: value for key, value in counters.items() if key.startswith("allocation10_")},
            "cumulative_campaign_accounting": cumulative,
            "latest_controlling_gate": {
                "pi0165_science": PI0165_SCIENCE_GATE,
                "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                "pi0165a_erratum": PI0165A_ERRATUM,
            },
            "allocation_6_failure": allocation_6_failure,
            "allocation_7_failure": allocation_7_failure,
            "allocation_8_failure": allocation8_failure,
            "allocation_9_failure": allocation9_failure,
            "source_inventory_artifact": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
            },
            "earliest_e_access": earliest_e_access,
            "raw_inventory": actual_raw_inventory,
            "final_raw_inventory_path": str(final_raw_inventory_path),
            "final_raw_inventory_sha256": sha256_file(final_raw_inventory_path),
            "final_raw_inventory": final_raw_inventory,
            "final_derived_count": final_derived_count,
            "raw_output_count": 36,
            "derived_output_count": 36,
            "seed_paired_summary": seed_summary,
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "training": False,
            "holdout_f_read": False,
            "automatic_retry": False,
            "scientific_contract_changed": False,
        }
        write_json(output_root / "stage45f_validation_e_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if output_root is not None:
            failure = {
                "schema": "STAGE45F_PI0167_VALIDATION_E_RECOVERY_FAILURE_SEAL_V1",
                "status": "PARTIAL" if counters["model_import_started"] else "BLOCKED",
                "token": "VANDA_PI0167_VALIDATION_E_GEN1_FOR_PI_REVIEW — BLOCKED",
                "pi_gate": PI0167_GATE,
                "pr_pointer": PI0167_POINTER,
                "authority_token": PI0167_TOKEN,
                "latest_controlling_gate": {
                    "pi0165_science": PI0165_SCIENCE_GATE,
                    "pi0165a_takeover": PI0165A_TAKEOVER_GATE,
                    "pi0165a_erratum": PI0165A_ERRATUM,
                },
                "execution_head": config.get("expected_head") if config else None,
                "execution_branch": continuation.EXPECTED_BRANCH,
                "pbs_jobid": os.environ.get("PBS_JOBID"),
                "phase": phase,
                "current": current,
                "actual_counters": counters,
                "allocation10_new_counters": {key: value for key, value in counters.items() if key.startswith("allocation10_")},
                "output_root": str(output_root),
                "raw_manifest": str(output_root / "raw_predictions.jsonl") if (output_root / "raw_predictions.jsonl").is_file() else None,
                "raw_manifest_sha256": sha256_file(output_root / "raw_predictions.jsonl") if (output_root / "raw_predictions.jsonl").is_file() else None,
                "model_import_started": bool(counters["model_import_started"]),
                "model_execution_started": bool(counters["model_import_started"]),
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "validation_e_read": True,
                "holdout_f_read": False,
                "automatic_retry": False,
                "scientific_contract_changed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp_utc": now_utc(),
            }
            try:
                write_json(output_root / "stage45f_pi0167_recovery_failure.json", failure)
            except Exception:
                pass
        print(f"VANDA_PI0167_VALIDATION_E_GEN1_FOR_PI_REVIEW — BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def plan(raw_inventory_root: Path | None = None) -> dict[str, Any]:
    guard = static_guard()
    result: dict[str, Any] = {
        "schema": "STAGE45F_PI0167_RECOVERY_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if guard["status"] == "PASS" else "FAIL",
        "static_guard": guard,
        "reused_raw_queries": 6,
        "new_queries": 30,
        "nominal_calls": EXPECTED_NOMINAL,
        "new_calls": {"conditioning": 30, "action_head": 120, "model_side_total": 150},
        "new_model_loads": 5,
        "training": 0,
        "holdout_f_read": 0,
        "raw_inventory_verified": False,
        "raw_inventory_required_before_qsub": True,
        "return_token": RETURN_TOKEN,
    }
    if raw_inventory_root is not None:
        inventory = validate_allocation8_inventory(raw_inventory_isolated(raw_inventory_root))
        result.update({
            "status": "PASS_STATIC_AND_RAW_INVENTORY" if guard["status"] == "PASS" else "FAIL",
            "raw_inventory_verified": True,
            "raw_inventory": inventory,
            "reused_raw_queries": inventory["preserved_raw_count"],
            "new_queries": inventory["missing_raw_count"],
            "new_calls": {
                "conditioning": inventory["missing_raw_count"],
                "action_head": inventory["missing_raw_count"] * 4,
                "model_side_total": inventory["missing_raw_count"] * 5,
            },
            "recovery_execution_plan": recovery_execution_plan(inventory),
            "new_model_loads": recovery_execution_plan(inventory)["model_load_attempts"],
        })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--pre-submit", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--raw-inventory-root", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    if args.plan_only:
        inventory_root = args.raw_inventory_root.resolve(strict=True) if args.raw_inventory_root is not None else None
        print(json.dumps(plan(inventory_root), indent=2, sort_keys=True))
        return 0
    if args.pre_submit:
        require(args.config is not None, "--pre-submit requires --config")
        print(json.dumps(pre_submit(args.config.resolve(strict=True), repo=args.repo.resolve(strict=True)), indent=2, sort_keys=True))
        return 0
    require(args.config is not None, "--execute requires --config")
    return execute(args.config.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
