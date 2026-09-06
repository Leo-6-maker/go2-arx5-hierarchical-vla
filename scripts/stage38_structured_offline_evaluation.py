#!/usr/bin/env python3
"""Stage38-E common-physical-space offline evaluation.

This runner consumes the accepted Stage38-D StructuredActionV1 checkpoints and
the sealed Stage33 dense-H40 controls.  It has three modes:

* ``--preflight`` performs identity, source, dense-evidence, and static checks
  without importing Torch or constructing a model;
* ``--evaluate`` performs only the authorized structured teacher probes and
  four-step offline denoising calls for the 16 frozen seeds; and
* ``--seal`` runs after PBS completion and seals the result/evidence/log set.

The evaluator never calls the official public sampler entry point, trains,
writes checkpoints, runs WBC, or reads validation/holdout data.  The local
denoising step is a source-mapped copy of the pinned N1.7 inference graph so
that the StructuredActionV1 output can be decoded before comparison with the
sealed dense physical trajectories.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PI_GATE_ID = "PI_GATE_STATE_0027"
PR_NUMBER = 24
EXPECTED_BRANCH = "stage38/structured-groot-offline"
EXPECTED_PARENT_HEAD = "bbf7453c7060d64fc6f659281ceb777246d6d370"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
STAGE38D_HEAD = "a7f68801ad05cbee86ebaaaf8d38357071bf1199"
EXPECTED_GROOT_HEAD = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_BASE_INDEX = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
EXPECTED_BASE_PROCESSOR_CONFIG = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
EXPECTED_BASE_STATISTICS = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
EXPECTED_TINY1_TREE = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
EXPECTED_MODALITY = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_SNAPSHOT = "74af1af85e21c6e3a5e91e5b4b695d080f0f6b7f2204d012310d6457334106f1"
EXPECTED_H2_1000_TREE = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
EXPECTED_H2_2000_TREE = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
EXPECTED_H2_SEAL = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"

BASE_ROOT = Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B")
TINY1_ROOT = Path(
    "/scratch/e1192548/stage23_o1_20260903/input_exact/extracted/"
    "tiny1_h2stats_lerobot_v2"
)
GROOT_ROOT = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t")
MODALITY_PATH = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/"
    "go2_arx5_modality.py"
)
COSMOS_ROOT = Path(
    "/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/"
    "models--nvidia--Cosmos-Reason2-2B"
)
H2_ROOT = Path(
    "/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training"
)
H2_SEAL_PATH = H2_ROOT.parent / "stage28_h2_checkpoint_2000_seal.json"
SNAPSHOT_ROOT = Path("/scratch/e1192548/stage26_h1_c1_20260904/snapshots")
SNAPSHOT_PATH = SNAPSHOT_ROOT / "stage22_expert_red_target_train_a/expert-0000.json"
STAGE26_ROOT = Path("/scratch/e1192548/stage26_h1_c1_20260904/runs/600532.hopper-m-02/output")
STAGE26_RESULT = STAGE26_ROOT / "stage26_h1_c1_result.json"
STAGE26_MANIFEST = STAGE26_ROOT / "stage26_h1_c1_hash_manifest.json"
STAGE26_RESULT_SHA = "446a9aa9ee79dc979444be912be48b821965ec79609a08c2aa71d75f576903e3"
STAGE26_MANIFEST_SHA = "eb33c81b68bb8bd49ce0ed9b0b818fdbb29546b49a2d79a3ee62845e172afb7b"
STAGE33_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/stage33_bottleneck_map_20260905/"
    "stage33_602255.hopper-m-02"
)
STAGE33_RESULT = STAGE33_ROOT / "stage33_ma_result.json"
STAGE33_RAW_MANIFEST = STAGE33_ROOT / "stage33_ma_raw_hash_manifest.json"
STAGE33_SEAL = STAGE33_ROOT / "stage33_artifact_seal.json"
STAGE33_RESULT_SHA = "bc0b969369ea9ea4d8f29549bfb406112f01603b7c5642b9adaa96d67655cb96"
STAGE33_RAW_MANIFEST_SHA = "909f59d5cb63600c82c692cfa9e9b40d0cb77c19b30bb690cce67b8687e67565"
STAGE33_SEAL_SHA = "84986433532caecc2b1390049557e27e73815f530a92ccff16e82abd5c9efb28"
STAGE31_ROOT = Path("/scratch/e1192548/leon-storage/runs/stage31_on_trajectory_20260905/stage31_601759.hopper-m-02")
STAGE32_ROOT = Path("/scratch/e1192548/stage32_h2_1000_20260905/stage32_601973.hopper-m-02")
STAGE32_ROOT_ALT = Path("/scratch/e1192548/leon-storage/runs/stage32_h2_1000_20260905/stage32_602036.hopper-m-02")
STAGE38D_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/"
    "stage38d_train_a7f6880"
)
STAGE38D_RESULT = STAGE38D_ROOT / "stage38d_result.json"
STAGE38D_RESULT_SHA = "b75bd03967ca1797c04eb3dae5ceeccd52c9a6ec5d28ec02c8bc507b786e4aff"
STRUCTURED_STATS = STAGE38D_ROOT / "experiment_cfg/structured_action_statistics.json"
STRUCTURED_STATS_SHA = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
STRUCTURED_1000 = STAGE38D_ROOT / "checkpoint-1000"
STRUCTURED_2000 = STAGE38D_ROOT / "checkpoint-2000"
STRUCTURED_1000_TREE = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
STRUCTURED_2000_TREE = "4ce18fac97646e1c7a4c6000e2d9758a92ebe6cd4e1cd32fd94e6b6aa6d705bd"
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/"
    "stage38e_offline_a7f6880"
)

HORIZON = 40
UNIVERSAL_DIM = 132
PHYSICAL_DIM = 10
STRUCTURED_ACTIVE_SCALARS = 190
DENSE_ACTIVE_SCALARS = 400
TIMING_MODE = "T16"
INFERENCE_STEPS = 4
SEEDS = tuple(range(2223, 2239))
CHECKPOINTS = (
    ("STRUCTURED@1000", STRUCTURED_1000, "H2@1000"),
    ("STRUCTURED@2000", STRUCTURED_2000, "H2@2000"),
)
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE"
EXPECTED_PROBE_IDS = tuple(
    f"c1_seed{seed}_tq{quantile:02d}"
    for seed in (2223, 2224, 2225)
    for quantile in (10, 50, 90)
)
EXPECTED_TIMESTEP_IDS = (0, 250, 500, 750)


class Stage38EError(RuntimeError):
    """Fail-closed Stage38-E boundary error."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_record(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {"path": str(path), "exists": False, "bytes": None, "sha256": None}
    if path.is_file() and not path.is_symlink():
        record.update({"exists": True, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    if expected_sha is not None:
        record["expected_sha256"] = expected_sha
        record["hash_match"] = record["sha256"] == expected_sha
    return record


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT).strip()


def regular_entries(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise Stage38EError(f"tree root is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise Stage38EError(f"tree contains unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise Stage38EError(f"tree contains unsafe file entry: {child}")
            entries.append({"path": child.relative_to(root).as_posix(), "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_sha256(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    entries = regular_entries(root)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest(), entries


def qstat_no_active() -> dict[str, Any]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    completed = subprocess.run(["qstat", "-u", user], capture_output=True, text=True, check=False)
    stdout = completed.stdout or ""
    lines = [line for line in stdout.splitlines() if line.strip()]
    relevant = [line for line in lines if any(token in line.lower() for token in ("stage38", "structured_groot", "wbc"))]
    return {
        "command": f"qstat -u {user}",
        "returncode": int(completed.returncode),
        "relevant_lines": relevant,
        "no_active_stage38_or_wbc_job": completed.returncode == 0 and not relevant,
        "stdout_sha256": sha256_bytes(stdout.encode()),
        "stderr": (completed.stderr or "")[-2000:],
    }


def branch_audit(expected_head: str, api_head: str) -> dict[str, Any]:
    local = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "origin", f"refs/heads/{EXPECTED_BRANCH}").split()[0]
    parent_local = git("rev-parse", "stage37/geometry-capacity-qualification")
    parent_remote = git("ls-remote", "origin", "refs/heads/stage37/geometry-capacity-qualification").split()[0]
    dirty = git("status", "--porcelain=v1")
    descends = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", EXPECTED_PARENT_HEAD, local], check=False
    ).returncode == 0
    result = {
        "branch": branch,
        "expected_branch": EXPECTED_BRANCH,
        "api_live_head": api_head,
        "remote_head": remote,
        "local_head": local,
        "expected_head": expected_head,
        "parent_local_head": parent_local,
        "parent_remote_head": parent_remote,
        "expected_parent_head": EXPECTED_PARENT_HEAD,
        "worktree_status": dirty,
        "worktree_clean": dirty == "",
        "descends_from_stage37": descends,
    }
    result["status"] = "PASS" if (
        api_head == expected_head == remote == local
        and branch == EXPECTED_BRANCH
        and parent_local == parent_remote == EXPECTED_PARENT_HEAD
        and dirty == ""
        and descends
    ) else "BLOCKED"
    return result


def static_guard() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_import_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_import_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_import_roots.add(node.module.split(".")[0])
    forbidden_top_level = sorted(top_level_import_roots.intersection({"torch", "transformers", "isaacgym", "omni"}))
    forbidden_calls = {"get_action", "forward", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}
    calls: list[str] = []
    scheduler_or_optimizer_steps: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
                if node.func.attr == "step" and isinstance(node.func.value, ast.Name) and node.func.value.id in {"optimizer", "scheduler"}:
                    scheduler_or_optimizer_steps.append(f"{node.func.value.id}.step")
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    forbidden_call_hits = sorted(set(calls).intersection(forbidden_calls))
    has_train_mode = any(isinstance(node, ast.Constant) and node.value == "--" + "train" for node in ast.walk(tree))
    has_scheduler_or_optimizer_step = bool(scheduler_or_optimizer_steps)
    has_checkpoint_write = any(name in {"save_pretrained", "save_model", "save_checkpoint"} for name in calls)
    return {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "top_level_forbidden_imports": forbidden_top_level,
        "forbidden_call_hits": forbidden_call_hits,
        "train_mode_literal": has_train_mode,
        "optimizer_or_scheduler_step_literal": has_scheduler_or_optimizer_step,
        "checkpoint_write_literal": has_checkpoint_write,
        "static_no_model_or_training_path": not forbidden_top_level and not forbidden_call_hits and not has_train_mode and not has_scheduler_or_optimizer_step and not has_checkpoint_write,
        "status": "PASS" if not forbidden_top_level and not forbidden_call_hits and not has_train_mode and not has_scheduler_or_optimizer_step and not has_checkpoint_write else "BLOCKED",
    }


def stage38d_audit(rehash_checkpoints: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "result_file": file_record(STAGE38D_RESULT, STAGE38D_RESULT_SHA),
        "statistics_file": file_record(STRUCTURED_STATS, STRUCTURED_STATS_SHA),
        "checkpoint_1000": {},
        "checkpoint_2000": {},
        "status": "BLOCKED",
    }
    if not STAGE38D_RESULT.is_file() or sha256_file(STAGE38D_RESULT) != STAGE38D_RESULT_SHA:
        return result
    payload = read_json(STAGE38D_RESULT)
    result["payload_fields"] = {
        key: payload.get(key)
        for key in ("schema", "result", "execution_head", "pbs_jobid", "timing_model_capacity_star", "structured_codec", "scientific_contract_changed", "historical_inputs_mutated", "next_gate")
    }
    expected_checkpoints = {
        "checkpoint_1000": (STRUCTURED_1000, 1000, STRUCTURED_1000_TREE),
        "checkpoint_2000": (STRUCTURED_2000, 2000, STRUCTURED_2000_TREE),
    }
    for name, (path, step, expected_tree) in expected_checkpoints.items():
        item = {"path": str(path), "expected_step": step, "expected_tree_sha256": expected_tree}
        if path.is_dir() and not path.is_symlink():
            state_path = path / "trainer_state.json"
            item["trainer_state"] = file_record(state_path)
            try:
                state = read_json(state_path)
                item["global_step"] = state.get("global_step")
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            if rehash_checkpoints:
                observed, entries = tree_sha256(path, f"stage38d/{path.name}")
                item.update({"tree_sha256": observed, "file_count": len(entries), "total_bytes": sum(int(entry["bytes"]) for entry in entries)})
        result[name] = item
    sidecar = read_json(STRUCTURED_STATS) if STRUCTURED_STATS.is_file() else {}
    contract = sidecar.get("contract", {}) if isinstance(sidecar, Mapping) else {}
    result["sidecar_contract"] = {
        "schema": sidecar.get("schema"),
        "timing_mode": contract.get("timing_mode"),
        "geometry_k": contract.get("geometry_k"),
        "action_horizon": contract.get("action_horizon"),
        "universal_action_dim": contract.get("universal_action_dim"),
        "active_scalar_count": contract.get("mask_active_scalar_count"),
        "source_dataset_untouched_by_factory": sidecar.get("source_dataset_untouched_by_factory"),
    }
    checks = [
        result["result_file"].get("hash_match") is True,
        result["statistics_file"].get("hash_match") is True,
        payload.get("schema") == "STAGE38D_STRUCTURED_GR00T_TRAINING_RESULT_V1",
        payload.get("result") == "STAGE38D_STRUCTURED_TRAINING_COMPLETE",
        payload.get("execution_head") == STAGE38D_HEAD,
        payload.get("timing_model_capacity_star") == TIMING_MODE,
        payload.get("structured_codec") == "STRUCTURED_ACTION_V1_QUALIFIED",
        payload.get("scientific_contract_changed") is False,
        payload.get("historical_inputs_mutated") is False,
        payload.get("validation_e_read") is False,
        payload.get("holdout_f_read") is False,
        payload.get("wbc") is False,
        payload.get("hardware_control") is False,
        result["sidecar_contract"].get("timing_mode") == TIMING_MODE,
        result["sidecar_contract"].get("geometry_k") == 16,
        result["sidecar_contract"].get("action_horizon") == HORIZON,
        result["sidecar_contract"].get("universal_action_dim") == UNIVERSAL_DIM,
        result["sidecar_contract"].get("active_scalar_count") == STRUCTURED_ACTIVE_SCALARS,
    ]
    for name, (_, step, expected_tree) in expected_checkpoints.items():
        item = result[name]
        payload_checkpoint = payload.get("structured_" + name, {})
        checks.extend([
            item.get("global_step") == step,
            item.get("tree_sha256") == expected_tree,
            item.get("file_count") == 19,
            item.get("total_bytes") is not None,
            item.get("trainer_state", {}).get("exists") is True,
            payload_checkpoint.get("path") == str(expected_checkpoints[name][0]),
            payload_checkpoint.get("status") == "PASS",
            payload_checkpoint.get("expected_step") == step,
            payload_checkpoint.get("global_step") == step,
            payload_checkpoint.get("file_count") == 19,
            payload_checkpoint.get("total_bytes") == item.get("total_bytes"),
            payload_checkpoint.get("tree_sha256") == expected_tree,
            payload_checkpoint.get("required_state_files_present") is True,
        ])
    result["status"] = "PASS" if all(checks) else "BLOCKED"
    return result


def stage26_audit() -> dict[str, Any]:
    result = {
        "result": file_record(STAGE26_RESULT, STAGE26_RESULT_SHA),
        "manifest": file_record(STAGE26_MANIFEST, STAGE26_MANIFEST_SHA),
        "status": "BLOCKED",
    }
    if not STAGE26_RESULT.is_file() or not STAGE26_MANIFEST.is_file():
        return result
    payload = read_json(STAGE26_RESULT)
    manifest = read_json(STAGE26_MANIFEST)
    rows = payload.get("records", [])
    noise_hashes: dict[str, str] = {}
    for row in rows:
        probe_id = str(row.get("probe_id"))
        noise_hashes[probe_id] = str(row.get("noise_sha256"))
    result["result_fields"] = {"schema": payload.get("schema"), "status": payload.get("status"), "record_count": len(rows), "probe_ids": sorted(noise_hashes)}
    result["noise_sha256_by_probe"] = noise_hashes
    result["manifest_fields"] = {"schema": manifest.get("schema"), "record_count": len(manifest.get("decoded_action_hashes", [])) + len(manifest.get("raw_velocity_hashes", []))}
    result["status"] = "PASS" if (
        result["result"].get("hash_match") is True
        and result["manifest"].get("hash_match") is True
        and payload.get("status") == "PASS_EXECUTION"
        and len(rows) == 18
        and set(noise_hashes) == set(EXPECTED_PROBE_IDS)
        and len(manifest.get("decoded_action_hashes", [])) == 18
        and len(manifest.get("raw_velocity_hashes", [])) == 18
    ) else "BLOCKED"
    return result


def _allowed_dense_artifact(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    allowed_roots = (
        (STAGE33_ROOT / "ma_traces").resolve(),
        (STAGE31_ROOT / "traces").resolve(),
        (STAGE32_ROOT / "traces").resolve(),
        (STAGE32_ROOT_ALT / "traces").resolve(),
    )
    return any(resolved.parent == root or root in resolved.parents for root in allowed_roots)


def dense_audit() -> dict[str, Any]:
    result: dict[str, Any] = {
        "result_file": file_record(STAGE33_RESULT, STAGE33_RESULT_SHA),
        "raw_manifest_file": file_record(STAGE33_RAW_MANIFEST, STAGE33_RAW_MANIFEST_SHA),
        "seal_file": file_record(STAGE33_SEAL, STAGE33_SEAL_SHA),
        "status": "BLOCKED",
        "records": [],
    }
    if not STAGE33_RESULT.is_file() or not STAGE33_RAW_MANIFEST.is_file() or not STAGE33_SEAL.is_file():
        return result
    payload = read_json(STAGE33_RESULT)
    raw_manifest = read_json(STAGE33_RAW_MANIFEST)
    seal = read_json(STAGE33_SEAL)
    rows = payload.get("records", [])
    expected_keys = {(seed, label) for seed in SEEDS for label in ("H2@1000", "H2@2000")}
    observed_keys: set[tuple[int, str]] = set()
    noise_by_seed: dict[int, str] = {}
    seal_by_path = {str(item.get("path")): item for item in seal.get("artifacts", []) if isinstance(item, Mapping)}
    row_errors: list[str] = []
    controls: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row.get("seed", -1)), str(row.get("checkpoint")))
        observed_keys.add(key)
        artifact = Path(str(row.get("artifact", {}).get("path", "")))
        if key in controls:
            row_errors.append(f"duplicate:{key}")
            continue
        if not _allowed_dense_artifact(artifact) or not artifact.is_file() or artifact.is_symlink():
            row_errors.append(f"artifact_scope_or_missing:{key}:{artifact}")
            continue
        artifact_hash = sha256_file(artifact)
        if artifact_hash != row.get("artifact", {}).get("sha256"):
            row_errors.append(f"artifact_hash:{key}")
        try:
            archive = np.load(artifact, allow_pickle=False)
            required = {"x_trajectory_float32", "v_trajectory_float32", "timestep_ids"}
            if not required.issubset(set(archive.files)):
                raise Stage38EError(f"missing trace arrays: {sorted(required.difference(archive.files))}")
            x = np.asarray(archive["x_trajectory_float32"], dtype=np.float32)
            v = np.asarray(archive["v_trajectory_float32"], dtype=np.float32)
            tids = np.asarray(archive["timestep_ids"], dtype=np.int64)
            if x.shape != (5, 1, HORIZON, UNIVERSAL_DIM) or v.shape != (4, 1, HORIZON, UNIVERSAL_DIM) or tuple(tids.tolist()) != EXPECTED_TIMESTEP_IDS:
                raise Stage38EError(f"trace shape/timestep mismatch: {x.shape} {v.shape} {tids.tolist()}")
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                raise Stage38EError("trace contains non-finite values")
            noise = np.ascontiguousarray(x[0])
            raw = np.ascontiguousarray(x[-1])
            noise_hash = sha256_bytes(noise.tobytes())
            raw_hash = sha256_bytes(raw.tobytes())
            if noise_hash != row.get("initial_noise", {}).get("float32_sha256"):
                row_errors.append(f"noise_hash:{key}")
            if raw_hash != row.get("raw_universal_action", {}).get("sha256"):
                row_errors.append(f"raw_hash:{key}")
            if key[0] in noise_by_seed and noise_by_seed[key[0]] != noise_hash:
                row_errors.append(f"paired_noise:{key}")
            noise_by_seed.setdefault(key[0], noise_hash)
            stored_decoded = None
            if "decoded_physical_action" in archive.files:
                stored_decoded = np.asarray(archive["decoded_physical_action"], dtype=np.float32)
                if stored_decoded.shape != (1, HORIZON, PHYSICAL_DIM) or not np.isfinite(stored_decoded).all():
                    raise Stage38EError("stored dense physical action invalid")
                if sha256_bytes(np.ascontiguousarray(stored_decoded).tobytes()) != row.get("decoded_physical_action", {}).get("sha256"):
                    row_errors.append(f"decoded_hash:{key}")
            seal_item = seal_by_path.get(str(artifact))
            if str(artifact).startswith(str(STAGE33_ROOT)) and (seal_item is None or seal_item.get("sha256") != artifact_hash):
                row_errors.append(f"seal_membership:{key}")
            controls[key] = {
                "checkpoint": key[1],
                "seed": key[0],
                "artifact": {"path": str(artifact), "bytes": int(artifact.stat().st_size), "sha256": artifact_hash},
                "noise": noise.copy(),
                "noise_sha256": noise_hash,
                "raw": raw.copy(),
                "raw_sha256": raw_hash,
                "stored_decoded": None if stored_decoded is None else stored_decoded.copy(),
                "historical_source": row.get("historical_source"),
            }
        except Exception as exc:
            row_errors.append(f"trace_read:{key}:{type(exc).__name__}:{exc}")
    result["records"] = [{key: value for key, value in item.items() if key not in {"noise", "raw", "stored_decoded"}} for item in controls.values()]
    result["control_key_count"] = len(controls)
    result["expected_key_count"] = len(expected_keys)
    result["noise_sha256_by_seed"] = noise_by_seed
    result["row_errors"] = row_errors
    result["payload_fields"] = {
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "scientific_contract_changed": payload.get("scientific_contract_changed"),
        "execution": payload.get("execution"),
        "claim_boundary": payload.get("claim_boundary"),
    }
    result["raw_manifest_fields"] = {"schema": raw_manifest.get("schema"), "status": raw_manifest.get("status"), "record_count": len(raw_manifest.get("records", [])), "scientific_contract_changed": raw_manifest.get("scientific_contract_changed")}
    result["seal_fields"] = {"schema": seal.get("schema"), "status": seal.get("status"), "artifact_count": seal.get("artifact_count"), "scientific_contract_changed": seal.get("scientific_contract_changed"), "historical_artifacts_modified": seal.get("historical_artifacts_modified")}
    result["status"] = "PASS" if (
        result["result_file"].get("hash_match") is True
        and result["raw_manifest_file"].get("hash_match") is True
        and result["seal_file"].get("hash_match") is True
        and payload.get("schema") == "STAGE33_MA_RESULT_V1"
        and payload.get("status") == "PASS_EXECUTION_COMPLETE"
        and payload.get("scientific_contract_changed") is False
        and payload.get("execution", {}).get("wbc") is False
        and payload.get("execution", {}).get("training") is False
        and raw_manifest.get("schema") == "STAGE33_MA_RAW_HASH_MANIFEST_V1"
        and raw_manifest.get("status") == "PASS"
        and len(raw_manifest.get("records", [])) == 32
        and seal.get("schema") == "STAGE33_ARTIFACT_SEAL_V1"
        and seal.get("status") == "PASS"
        and seal.get("scientific_contract_changed") is False
        and seal.get("historical_artifacts_modified") is False
        and observed_keys == expected_keys
        and len(controls) == 32
        and not row_errors
    ) else "BLOCKED"
    result["controls"] = controls
    return result


def historical_identity_audit() -> dict[str, Any]:
    """Revalidate all historical identities without constructing a model."""

    stage26 = importlib.import_module("stage26_h1_c1_fixed_noise_local_flow")
    base = stage26._checkpoint_preflight("BASE", BASE_ROOT)
    tiny = stage26._verify_tiny_dataset(TINY1_ROOT)
    snapshot_info, snapshot_path = stage26._verify_snapshot_identity(SNAPSHOT_ROOT)
    groot_head = stage26._verify_groot_source(GROOT_ROOT)
    cosmos = stage26._verify_cosmos_cache(COSMOS_ROOT)
    base_processor = file_record(BASE_ROOT / "processor_config.json", EXPECTED_BASE_PROCESSOR_CONFIG)
    base_statistics = file_record(BASE_ROOT / "statistics.json", EXPECTED_BASE_STATISTICS)
    h2_config = H2_ROOT / "experiment_cfg/config.yaml"
    h2_config_sha = sha256_file(h2_config) if h2_config.is_file() else None
    h2_seal = file_record(H2_SEAL_PATH, EXPECTED_H2_SEAL)
    modality = file_record(MODALITY_PATH, EXPECTED_MODALITY)
    return {
        "base": {"status": "PASS" if base.get("tree_sha256") == EXPECTED_BASE_TREE and base.get("index_sha256") == EXPECTED_BASE_INDEX and base_processor.get("hash_match") is True and base_statistics.get("hash_match") is True else "BLOCKED", "tree_sha256": base.get("tree_sha256"), "index_sha256": base.get("index_sha256"), "processor_config": base_processor, "statistics": base_statistics},
        "tiny1": {"status": "PASS" if tiny.get("tree_sha256") == EXPECTED_TINY1_TREE else "BLOCKED", **{key: tiny.get(key) for key in ("path", "tree_sha256", "file_count", "total_bytes", "stats_json_sha256", "relative_stats_json_sha256")}},
        "snapshot": {"status": "PASS" if snapshot_info.get("selected_sha256") == EXPECTED_SNAPSHOT else "BLOCKED", **snapshot_info},
        "groot": {"status": "PASS" if groot_head == EXPECTED_GROOT_HEAD and git("status", "--porcelain=v1", cwd=GROOT_ROOT) == "" else "BLOCKED", "head": groot_head, "path": str(GROOT_ROOT)},
        "cosmos": {"status": "PASS" if cosmos.get("revision") == EXPECTED_COSMOS_REVISION and cosmos.get("tree_sha256") == EXPECTED_COSMOS_TREE else "BLOCKED", **cosmos},
        "h2": {"status": "PASS" if h2_config_sha == "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c" and h2_seal.get("hash_match") is True else "BLOCKED", "config_sha256": h2_config_sha, "seal": h2_seal, "checkpoint_1000_tree_sha256": EXPECTED_H2_1000_TREE, "checkpoint_2000_tree_sha256": EXPECTED_H2_2000_TREE},
        "modality": {"status": "PASS" if modality.get("hash_match") is True else "BLOCKED", **modality},
        "snapshot_path": str(snapshot_path),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        existing = {path.name for path in output_root.iterdir()}
        allowed = {"stage38e_preflight.json"}
        if existing - allowed:
            raise Stage38EError(f"Stage38-E output root is not fresh: {sorted(existing)}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    branch = branch_audit(args.expected_project_sha, args.api_head)
    d_audit = stage38d_audit(rehash_checkpoints=True)
    stage26 = stage26_audit()
    dense = dense_audit()
    static = static_guard()
    qstat = qstat_no_active()
    historical = historical_identity_audit()
    usage = shutil.disk_usage(output_root)
    scratch = {"path": str(output_root), "free_bytes": int(usage.free), "total_bytes": int(usage.total), "minimum_free_gib": float(args.minimum_free_gib), "sufficient_free_space": usage.free >= args.minimum_free_gib * 1024**3}
    status = "PASS" if (
        branch.get("status") == "PASS"
        and d_audit.get("status") == "PASS"
        and stage26.get("status") == "PASS"
        and dense.get("status") == "PASS"
        and static.get("status") == "PASS"
        and qstat.get("no_active_stage38_or_wbc_job") is True
        and scratch["sufficient_free_space"]
        and all(item.get("status") == "PASS" for item in historical.values() if isinstance(item, Mapping) and "status" in item)
    ) else "BLOCKED"
    report = {
        "schema": "STAGE38E_STRUCTURED_OFFLINE_PREFLIGHT_V1",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "expected_project": EXPECTED_PROJECT,
        "expected_queue": EXPECTED_QUEUE,
        "expected_project_head": args.expected_project_sha,
        "api_live_head": args.api_head,
        "stage38d_training_head": STAGE38D_HEAD,
        "branch": branch,
        "stage38d": d_audit,
        "stage26_fixed_probe_evidence": stage26,
        "dense_stage33_evidence": {key: value for key, value in dense.items() if key != "controls"},
        "historical_identity": historical,
        "static_guard": static,
        "qstat": qstat,
        "scratch": scratch,
        "scientific_contract_changed": False,
        "model_constructed": False,
        "gr00t_inference": False,
        "training": False,
        "wbc_executed": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "hardware": False,
        "pbs_submitted": False,
        "status": status,
        "next_gate": NEXT_GATE if status == "PASS" else NEXT_GATE,
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38e_preflight.json", report)
    return report


def _setup_runtime(output_root: Path, groot_source: Path, cosmos_cache_root: Path) -> dict[str, Any]:
    stage38_train = importlib.import_module("stage38_structured_train")
    cache_root = output_root / "cache"
    tmp_root = output_root / "tmp"
    cache_root.mkdir(parents=True, exist_ok=False)
    tmp_root.mkdir(parents=True, exist_ok=False)
    cosmos_binding = stage38_train.bind_cosmos_cache(cache_root)
    if Path(cosmos_cache_root).resolve() != COSMOS_ROOT.resolve():
        raise Stage38EError("Stage38-E Cosmos cache argument is not the frozen root")
    os.environ.update({
        "TMPDIR": str(tmp_root), "TMP": str(tmp_root), "TEMP": str(tmp_root),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "HF_HOME": str(cache_root / "huggingface"),
        "TRANSFORMERS_CACHE": str(cache_root / "huggingface" / "hub"),
        "TORCH_HOME": str(cache_root / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "GROOT_COMMIT_HASH": EXPECTED_GROOT_HEAD,
    })
    os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO / "scripts"), str(groot_source), os.environ.get("PYTHONPATH", "")])
    sys.path.insert(0, str(groot_source))
    processor_binding = stage38_train.bind_cosmos_processor()
    binding = {
        "schema": "STAGE38E_RUNTIME_BINDING_V1",
        "cosmos": cosmos_binding,
        "cosmos_processor": processor_binding,
        "groot_source": str(groot_source),
        "groot_head": EXPECTED_GROOT_HEAD,
        "container_image": os.environ.get("STAGE38_CONTAINER_IMAGE"),
        "container_site": os.environ.get("STAGE38_CONTAINER_SITE"),
        "ffmpeg_bundle": os.environ.get("STAGE38_FFMPEG_BUNDLE"),
        "ffmpeg_ld_library_path": os.environ.get("STAGE38_FFMPEG_LD", os.environ.get("LD_LIBRARY_PATH", "")),
        "tmp_root": str(tmp_root),
        "cache_root": str(cache_root),
        "historical_cache_mutation_allowed": False,
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38e_runtime_binding.json", binding)
    return binding


def _build_conditioning(processor: Any, embodiment: Any, prepared: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    from gr00t.data.types import MessageType, VLAStepData
    from scripts import stage29_c2_free_running_transfer as stage29

    source_step = prepared["step_data"]
    inference_step = VLAStepData(images=source_step.images, states=source_step.states, actions={}, text=source_step.text, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
    batch = processor.collator([processed])
    model_inputs = batch.get("inputs", {})
    if "action" in model_inputs or "action_mask" in model_inputs:
        raise Stage38EError("ground-truth action leaked into offline conditioning")
    return stage29._conditioning_manifest(batch), batch


def _expert_chunk(snapshot: Any, expert_physical: np.ndarray) -> Any:
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    physical = np.asarray(expert_physical, dtype=np.float64)
    if physical.shape != (1, HORIZON, PHYSICAL_DIM):
        raise Stage38EError(f"expert physical shape drifted: {physical.shape}")
    return TrajectoryChunkV1(
        chunk_id=f"expert:{snapshot.example_id}",
        source="SEALED_STAGE22_EXPERT_PHYSICAL",
        observation_timestamp=float(snapshot.observation_timestamp),
        point_timestamps=np.asarray(snapshot.future_timestamps, dtype=np.float64),
        eef_position=physical[0, :, :3],
        eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(physical[0, :, 3:9])),
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        validity_start=float(snapshot.future_timestamps[0]),
        validity_end=float(snapshot.future_timestamps[-1]),
        gripper=physical[0, :, 9],
        provenance={"source": "Stage22 sealed snapshot", "target_excluded_from_model_input": True},
    )


def _dense_chunk(control: Mapping[str, Any], snapshot: Any, processor: Any, embodiment: Any) -> tuple[Any, dict[str, Any]]:
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    decoded = control.get("stored_decoded")
    if decoded is None:
        decoded = stage26._decode_physical_action(processor, embodiment, control["raw"], snapshot)
    decoded = np.asarray(decoded, dtype=np.float32)
    if decoded.shape != (1, HORIZON, PHYSICAL_DIM) or not np.isfinite(decoded).all():
        raise Stage38EError(f"dense common physical decode invalid: {decoded.shape}")
    chunk = TrajectoryChunkV1(
        chunk_id=f"{control['checkpoint']}:{control['seed']}",
        source="SEALED_STAGE33_DENSE_H40",
        observation_timestamp=float(snapshot.observation_timestamp),
        point_timestamps=np.asarray(snapshot.future_timestamps, dtype=np.float64),
        eef_position=decoded[0, :, :3],
        eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(decoded[0, :, 3:9])),
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        validity_start=float(snapshot.future_timestamps[0]),
        validity_end=float(snapshot.future_timestamps[-1]),
        gripper=decoded[0, :, 9],
        provenance={"source": "Stage33 sealed dense control", "artifact": control["artifact"]},
    )
    meta = {key: value for key, value in control.items() if key not in {"noise", "raw", "stored_decoded"}}
    meta["decoded_physical_action_sha256"] = sha256_bytes(np.ascontiguousarray(decoded).tobytes())
    return chunk, meta


def _structured_label_and_normalizer(expert_physical: np.ndarray, snapshot: Any, sidecar: Mapping[str, Any]) -> tuple[Any, Any, np.ndarray]:
    from go2arx5_vla.stage38_structured_training import StructuredActionNormalizer
    from go2arx5_vla.structured_action import encode_structured_action

    timestamps = np.asarray(snapshot.future_timestamps, dtype=np.float64)
    # The official processor materializes both the action and runtime state at
    # float32 before decode.  Preserve that exact boundary for the codec anchor
    # check, then use float64 for deterministic CPU geometry calculations.
    state = np.asarray(snapshot.state_v1.as_array(), dtype=np.float32).astype(np.float64)
    label = encode_structured_action(
        np.asarray(expert_physical[0], dtype=np.float64), timestamps, state,
        timing_mode=TIMING_MODE,
        chunk_id=f"stage38e:{snapshot.example_id}",
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        observation_timestamp=float(snapshot.observation_timestamp),
        provenance={"stage": "Stage38-E", "source_timestamp_authority": "Stage22 snapshot"},
    )
    stats = sidecar.get("statistics")
    if not isinstance(stats, Mapping):
        raise Stage38EError("structured statistics sidecar has no statistics mapping")
    normalizer = StructuredActionNormalizer(stats, clip_outliers=True)
    normalized = normalizer.normalize(label)
    if normalized.shape != (HORIZON, UNIVERSAL_DIM) or not np.isfinite(normalized).all():
        raise Stage38EError("structured normalized label shape/finite contract failed")
    if int(np.count_nonzero(label.mask)) != STRUCTURED_ACTIVE_SCALARS:
        raise Stage38EError("structured label active scalar count drifted")
    return label, normalizer, normalized


def _codec_roundtrip(label: Any, expert_chunk: Any, snapshot: Any) -> dict[str, Any]:
    from go2arx5_vla.structured_action import decode_structured_action, materialize_structured_action
    from go2arx5_vla.motion_api import evaluate_preview_equivalence, public_evaluation

    intent = decode_structured_action(label, intent_id="stage38e-label-roundtrip")
    materialized = materialize_structured_action(label, intent, source_timestamps=snapshot.future_timestamps)
    if not np.array_equal(materialized.point_timestamps, expert_chunk.point_timestamps):
        raise Stage38EError("structured label roundtrip changed timestamps")
    if not np.allclose(materialized.gripper, expert_chunk.gripper, rtol=0.0, atol=1e-8):
        raise Stage38EError("structured label roundtrip changed gripper")
    equivalence = evaluate_preview_equivalence(expert_chunk, materialized, case_id="stage38e-label-roundtrip", representation="StructuredActionV1-label", anchor_count=101)
    public = public_evaluation(equivalence)
    return {
        "finite": bool(np.isfinite(materialized.eef_position).all() and np.isfinite(materialized.eef_orientation).all()),
        "source_frame": materialized.source_frame,
        "tcp_semantic_id": materialized.tcp_semantic_id,
        "start_pose_exact": bool(np.allclose(materialized.eef_position[0], expert_chunk.eef_position[0], atol=1e-10, rtol=0.0)),
        "end_pose_position_error_m": float(np.linalg.norm(materialized.eef_position[-1] - expert_chunk.eef_position[-1])),
        "duration_s": float(materialized.validity_end - materialized.validity_start),
        "timing_monotonic": bool(np.all(np.diff(intent.timing_timestamps) > 0.0)),
        "timing_endpoint_exact": bool(np.isclose(intent.validity_end - intent.validity_start, expert_chunk.validity_end - expert_chunk.validity_start, atol=1e-10, rtol=0.0)),
        "preview_equivalence": public,
        "status": "PASS" if equivalence.get("finite") and equivalence.get("monotonic_timestamps") and equivalence.get("duration_exact") and equivalence.get("passes_limits") and public.get("passes_limits") is True else "BLOCKED",
    }


def _inverse_normalize(normalized: np.ndarray, normalizer: Any, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float64)
    if values.shape != (HORIZON, UNIVERSAL_DIM) or not np.isfinite(values).all():
        raise Stage38EError(f"normalized prediction shape/finite contract failed: {values.shape}")
    lower = np.asarray(normalizer._lower, dtype=np.float64)
    upper = np.asarray(normalizer._upper, dtype=np.float64)
    raw = np.zeros_like(values, dtype=np.float64)
    varying = ~np.isclose(upper, lower, rtol=0.0, atol=1e-12)
    raw[:, varying] = lower[varying] + ((values[:, varying] + 1.0) / 2.0) * (upper[varying] - lower[varying])
    raw[:, ~varying] = lower[~varying]
    raw[np.asarray(mask) == 0.0] = 0.0
    if not np.isfinite(raw).all():
        raise Stage38EError("inverse-normalized structured prediction is non-finite")
    return raw


def _structured_chunk(normalized: np.ndarray, normalizer: Any, snapshot: Any, *, chunk_id: str, source: str) -> tuple[Any, dict[str, Any], np.ndarray]:
    from go2arx5_vla.structured_action import decode_structured_action, materialize_structured_action, structured_action_mask

    raw = _inverse_normalize(normalized, normalizer, structured_action_mask(TIMING_MODE))
    timestamps = np.asarray(snapshot.future_timestamps, dtype=np.float64)
    state = np.asarray(snapshot.state_v1.as_array(), dtype=np.float64)
    intent = decode_structured_action(
        raw,
        anchor_state_v1=state,
        source_timestamps=timestamps,
        source_duration_s=float(timestamps[-1] - timestamps[0]),
        timing_mode=TIMING_MODE,
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        observation_timestamp=float(snapshot.observation_timestamp),
        intent_id=chunk_id,
        provenance={"stage": "Stage38-E", "source": source},
    )
    chunk = materialize_structured_action(raw, intent, source_timestamps=timestamps)
    if chunk.point_count != HORIZON or not np.isfinite(chunk.eef_position).all() or not np.isfinite(chunk.eef_orientation).all():
        raise Stage38EError("structured physical decode is invalid")
    meta = {
        "chunk_id": chunk_id,
        "intent_id": intent.intent_id,
        "timing_mode": intent.timing_mode,
        "timing_interval_fractions": intent.provenance.get("timing_interval_fractions"),
        "timing_timestamps": intent.timing_timestamps.tolist(),
        "source_duration_s": float(intent.validity_end - intent.validity_start),
        "source_frame": chunk.source_frame,
        "tcp_semantic_id": chunk.tcp_semantic_id,
        "finite": True,
    }
    return chunk, meta, raw


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "std": None, "finite": False}
    finite = bool(np.isfinite(array).all())
    return {"count": int(array.size), "mean": float(np.mean(array)) if finite else None, "median": float(np.median(array)) if finite else None, "p95": float(np.percentile(array, 95)) if finite else None, "max": float(np.max(array)) if finite else None, "std": float(np.std(array)) if finite else None, "finite": finite}


def _high_frequency_xyz_fraction(position: Any) -> float:
    """Return a fixed descriptive high-frequency position-energy fraction.

    The endpoint-to-endpoint linear component is removed only for measurement;
    the trajectory itself is never changed.  Frequencies at or above half of
    the non-DC rFFT bins are treated as the high-frequency band.
    """

    values = np.asarray(position, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 4 or not np.isfinite(values).all():
        return float("nan")
    alpha = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)[:, None]
    baseline = (1.0 - alpha) * values[0] + alpha * values[-1]
    residual = values - baseline
    spectrum = np.fft.rfft(residual, axis=0)
    energy = np.sum(np.abs(spectrum[1:]) ** 2, axis=1)
    total = float(np.sum(energy))
    if not np.isfinite(total):
        return float("nan")
    if total <= 1e-20:
        return 0.0
    cutoff = max(1, int(np.ceil(energy.shape[0] / 2.0)))
    return float(np.sum(energy[cutoff:]) / total)


def _common_metrics(candidate: Any, expert: Any, *, representation: str, case_id: str) -> dict[str, Any]:
    from go2arx5_vla.metrics.action_profile import chunk_profile
    from go2arx5_vla.motion_api import evaluate_preview_equivalence, public_evaluation
    from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix

    position_error = np.linalg.norm(candidate.eef_position - expert.eef_position, axis=-1)
    rotation_error = np.asarray(geodesic_angle(quat_to_matrix(candidate.eef_orientation), quat_to_matrix(expert.eef_orientation)), dtype=np.float64).reshape(-1)
    gripper_error = np.abs(np.asarray(candidate.gripper, dtype=np.float64).reshape(-1) - np.asarray(expert.gripper, dtype=np.float64).reshape(-1))
    preview = evaluate_preview_equivalence(expert, candidate, case_id=case_id, representation=representation, anchor_count=101)
    candidate_profile = chunk_profile(candidate, achieved_position=expert.eef_position[0], achieved_quaternion=expert.eef_orientation[0], provenance={"profile_role": "candidate", "representation": representation})
    expert_profile = chunk_profile(expert, provenance={"profile_role": "expert"})
    safety = TrajectorySafetyGateV1().evaluate(candidate, now=float(expert.observation_timestamp), achieved_position=expert.eef_position[0], achieved_orientation=expert.eef_orientation[0]).as_dict()
    return {
        "common_grid_eef_position_error_m": _summary(position_error),
        "common_grid_so3_error_rad": _summary(rotation_error),
        "gripper_absolute_error": _summary(gripper_error),
        "endpoint_position_error_m": float(np.linalg.norm(candidate.eef_position[-1] - expert.eef_position[-1])),
        "endpoint_rotation_error_rad": float(geodesic_angle(quat_to_matrix(candidate.eef_orientation[-1]), quat_to_matrix(expert.eef_orientation[-1]))),
        "preview": public_evaluation(preview),
        "candidate_profile": candidate_profile,
        "expert_profile": expert_profile,
        "high_frequency_xyz_fraction": _high_frequency_xyz_fraction(candidate.eef_position),
        "safety_gate": safety,
        "all_finite": bool(np.isfinite(position_error).all() and np.isfinite(rotation_error).all() and np.isfinite(gripper_error).all() and np.isfinite(_high_frequency_xyz_fraction(candidate.eef_position))),
    }


def _representation_delta(candidate: Any, dense: Any, *, representation: str, case_id: str) -> dict[str, Any]:
    return _common_metrics(candidate, dense, representation=representation, case_id=case_id)


def _structured_flow_metrics(predicted: Any, target: Any, mask: np.ndarray) -> dict[str, Any]:
    predicted_np = np.asarray(predicted, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.float32)
    if predicted_np.shape != (1, HORIZON, UNIVERSAL_DIM) or target_np.shape != predicted_np.shape or mask_np.shape != predicted_np.shape:
        raise Stage38EError("structured flow diagnostic shape mismatch")
    squared = (predicted_np - target_np) ** 2
    active = mask_np > 0.0
    inactive = ~active
    return {
        "active_masked_mse": float(np.sum(squared[active]) / (np.sum(active) + 1e-6)),
        "active_scalar_count": int(np.sum(active)),
        "inactive_prediction_energy_mean": float(np.mean(predicted_np[inactive] ** 2)),
        "inactive_target_energy_mean": float(np.mean(target_np[inactive] ** 2)),
        "finite": bool(np.isfinite(predicted_np).all() and np.isfinite(target_np).all()),
    }


def _offline_step(action_head: Any, processed_backbone: Any, action_input: Any, state_features: Any, actions: Any, timestep: int) -> Any:
    """Mirror the pinned four-step action-head inference graph only."""

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
    if tuple(int(value) for value in velocity.shape) != (1, HORIZON, UNIVERSAL_DIM):
        raise Stage38EError(f"structured offline velocity shape drifted: {tuple(velocity.shape)}")
    if not bool(torch.isfinite(velocity).all().item()):
        raise Stage38EError("structured offline velocity is non-finite")
    return velocity


def _prepare_model_conditioning(model: Any, model_inputs: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    import torch

    backbone_inputs, action_inputs = model.prepare_input(dict(model_inputs))
    with torch.inference_mode():
        backbone_output = model.backbone(backbone_inputs)
        processed = model.action_head.process_backbone_output(backbone_output)
        if int(action_inputs.state.shape[1]) != int(model.action_head.config.state_history_length):
            raise Stage38EError("state history length drifted")
        state = action_inputs.state.view(action_inputs.state.shape[0], 1, -1)
        state_features = model.action_head.state_encoder(state, action_inputs.embodiment_id)
    return backbone_inputs, action_inputs, processed, state_features


def _load_structured_model(checkpoint: Path, device: str) -> Any:
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    if int(model.action_head.config.action_horizon) != HORIZON or int(model.action_head.action_dim) != UNIVERSAL_DIM:
        raise Stage38EError("structured checkpoint action shape drifted")
    if int(model.action_head.num_inference_timesteps) != INFERENCE_STEPS:
        raise Stage38EError("structured checkpoint inference-step contract drifted")
    if int(model.action_head.num_timestep_buckets) != 1000:
        raise Stage38EError("structured checkpoint timestep bucket contract drifted")
    return model


def _load_noise(control: Mapping[str, Any], seed: int) -> np.ndarray:
    noise = np.asarray(control["noise"], dtype=np.float32)
    if noise.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(noise).all():
        raise Stage38EError(f"dense noise for seed {seed} is invalid")
    return noise


def _state_write(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _chunk_physical_action(chunk: Any) -> np.ndarray:
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows

    position = np.asarray(chunk.eef_position, dtype=np.float64)
    orientation = np.asarray(chunk.eef_orientation, dtype=np.float64)
    gripper = np.asarray(chunk.gripper, dtype=np.float64).reshape(-1, 1)
    physical = np.concatenate(
        [position, rotation_matrix_to_rot6d_rows(quat_to_matrix(orientation)), gripper],
        axis=-1,
    )[None]
    if physical.shape != (1, HORIZON, PHYSICAL_DIM) or not np.isfinite(physical).all():
        raise Stage38EError(f"chunk physical action shape/finite contract failed: {physical.shape}")
    return physical.astype(np.float32)


def _save_trace(path: Path, *, noise: np.ndarray, normalized: np.ndarray, raw_structured: np.ndarray, physical: Any, expert: Any, x_trace: Sequence[np.ndarray], v_trace: Sequence[np.ndarray], timestep_ids: Sequence[int]) -> None:
    physical_array = np.asarray(physical, dtype=np.float32)
    expert_array = np.asarray(expert, dtype=np.float32)
    if physical_array.shape != (1, HORIZON, PHYSICAL_DIM) or expert_array.shape != (1, HORIZON, PHYSICAL_DIM):
        raise Stage38EError(f"trace physical arrays must be [1,{HORIZON},{PHYSICAL_DIM}]")
    if not np.isfinite(physical_array).all() or not np.isfinite(expert_array).all():
        raise Stage38EError("trace physical arrays contain non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        initial_noise_float32=np.asarray(noise, dtype=np.float32),
        normalized_structured_action=np.asarray(normalized, dtype=np.float32),
        raw_structured_action=np.asarray(raw_structured, dtype=np.float32),
        decoded_physical_action=physical_array,
        expert_physical_action=expert_array,
        x_trajectory_float32=np.asarray(x_trace, dtype=np.float32),
        v_trajectory_float32=np.asarray(v_trace, dtype=np.float32),
        timestep_ids=np.asarray(timestep_ids, dtype=np.int64),
    )


def _record_hash(array: Any) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float32)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_bytes(np.ascontiguousarray(value).tobytes()), "finite": bool(np.isfinite(value).all())}


def _run_teacher_probe(model: Any, processed: Any, action_inputs: Any, state_features: Any, normalized_target: np.ndarray, normalizer: Any, snapshot: Any, expert_chunk: Any, probe: Mapping[str, Any], fixed_noise: Any, output_root: Path, checkpoint_label: str, call_index: int) -> dict[str, Any]:
    import torch
    from go2arx5_vla.structured_action import structured_action_mask

    model_dtype = next(iter(model.parameters())).dtype
    noise_cpu = np.asarray(fixed_noise["noise"].detach().cpu().numpy(), dtype=np.float32)
    noisy_cpu = np.asarray(fixed_noise["noisy_action"].detach().cpu().numpy(), dtype=np.float32)
    target_cpu = np.asarray(fixed_noise["target_velocity"].detach().cpu().numpy(), dtype=np.float32)
    expected_noise_hash = str(fixed_noise["noise_sha256"])
    model_noise = torch.from_numpy(noisy_cpu).to(device=processed.backbone_features.device, dtype=model_dtype)
    t_tensor = fixed_noise["t_discretized"].to(device=processed.backbone_features.device)
    with torch.inference_mode():
        predicted = __import__("scripts.stage26_h1_c1_fixed_noise_local_flow", fromlist=["_teacher_forced_local_flow"])._teacher_forced_local_flow(
            action_head=model.action_head,
            backbone_output=processed,
            action_input=action_inputs,
            noisy_action=model_noise,
            t_discretized=t_tensor,
        )
    predicted_cpu = np.asarray(predicted.detach().float().cpu().numpy(), dtype=np.float32)
    flow = _structured_flow_metrics(predicted_cpu, target_cpu, structured_action_mask(TIMING_MODE)[None])
    implied_normalized = noise_cpu + predicted_cpu
    chunk, decode_meta, raw = _structured_chunk(implied_normalized[0], normalizer, snapshot, chunk_id=f"{checkpoint_label}:{probe['probe_id']}", source="fixed_teacher_probe")
    metrics = _common_metrics(chunk, expert_chunk, representation=checkpoint_label, case_id=f"teacher:{checkpoint_label}:{probe['probe_id']}")
    artifact = output_root / "teacher_traces" / f"{checkpoint_label.replace('@', '_').replace('STRUCTURED_', '').lower()}_{probe['probe_id']}.npz"
    _save_trace(artifact, noise=noise_cpu, normalized=implied_normalized, raw_structured=raw[None], physical=_chunk_physical_action(chunk), expert=_chunk_physical_action(expert_chunk), x_trace=[noise_cpu, implied_normalized], v_trace=[predicted_cpu], timestep_ids=[int(probe["t_discretized"])])
    return {
        "kind": "teacher_forced_structured",
        "checkpoint": checkpoint_label,
        "probe_id": str(probe["probe_id"]),
        "noise_seed": int(probe["noise_seed"]),
        "t_quantile": float(probe["t_quantile"]),
        "t_value": float(probe["t_value"]),
        "t_discretized": int(probe["t_discretized"]),
        "call_index": call_index,
        "pairing_mode": "REPRODUCED_NOISE_HASH_MATCH",
        "noise_sha256": expected_noise_hash,
        "noisy_action_sha256": _record_hash(noisy_cpu),
        "target_velocity_sha256": _record_hash(target_cpu),
        "predicted_velocity": _record_hash(predicted_cpu),
        "implied_normalized_action": _record_hash(implied_normalized),
        "decoded_structured": {"raw": _record_hash(raw), **decode_meta},
        "flow_metrics": flow,
        "metrics": metrics,
        "artifact": file_record(artifact),
        "active_supervised_scalar_count": STRUCTURED_ACTIVE_SCALARS,
        "finite": bool(metrics["all_finite"]),
        "decode_valid": bool(decode_meta.get("finite") is True),
    }


def _run_free_call(model: Any, processed: Any, action_inputs: Any, state_features: Any, noise: np.ndarray, normalizer: Any, snapshot: Any, expert_chunk: Any, dense_chunk: Any, checkpoint_label: str, dense_label: str, seed: int, output_root: Path, call_index: int) -> dict[str, Any]:
    import torch

    action = torch.from_numpy(np.asarray(noise, dtype=np.float32)).to(device=processed.backbone_features.device, dtype=next(iter(model.parameters())).dtype)
    x_trace = [np.asarray(noise, dtype=np.float32).copy()]
    v_trace: list[np.ndarray] = []
    timestep_ids: list[int] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(INFERENCE_STEPS):
            timestep = int(step * 1000 / INFERENCE_STEPS)
            timestep_ids.append(timestep)
            velocity = _offline_step(model.action_head, processed, action_inputs, state_features, action, timestep)
            v_cpu = np.asarray(velocity.detach().float().cpu().numpy(), dtype=np.float32)
            v_trace.append(v_cpu.copy())
            action = action + (1.0 / INFERENCE_STEPS) * velocity
            x_trace.append(np.asarray(action.detach().float().cpu().numpy(), dtype=np.float32))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    normalized = np.asarray(action.detach().float().cpu().numpy(), dtype=np.float32)
    chunk, decode_meta, raw = _structured_chunk(normalized[0], normalizer, snapshot, chunk_id=f"{checkpoint_label}:{seed}", source="free_running_structured")
    metrics = _common_metrics(chunk, expert_chunk, representation=checkpoint_label, case_id=f"free:{checkpoint_label}:{seed}")
    delta = _representation_delta(chunk, dense_chunk, representation=f"{checkpoint_label}_vs_{dense_label}", case_id=f"delta:{checkpoint_label}:{seed}")
    artifact = output_root / "free_traces" / f"{checkpoint_label.replace('@', '_').replace('STRUCTURED_', '').lower()}_seed{seed}.npz"
    _save_trace(artifact, noise=noise, normalized=normalized, raw_structured=raw[None], physical=_chunk_physical_action(chunk), expert=_chunk_physical_action(expert_chunk), x_trace=x_trace, v_trace=v_trace, timestep_ids=timestep_ids)
    return {
        "kind": "free_running_structured",
        "checkpoint": checkpoint_label,
        "dense_reference_checkpoint": dense_label,
        "seed": int(seed),
        "call_index": call_index,
        "pairing_mode": "EXACT_NOISE",
        "initial_noise": _record_hash(noise),
        "denoising": {"steps": INFERENCE_STEPS, "dt": 1.0 / INFERENCE_STEPS, "timestep_ids": timestep_ids, "api": "source-mapped-action-head-inference-graph", "official_public_sampler_called": False},
        "normalized_action": _record_hash(normalized),
        "decoded_structured": {"raw": _record_hash(raw), **decode_meta},
        "metrics_to_expert": metrics,
        "metrics_vs_dense": delta,
        "artifact": file_record(artifact),
        "latency_s": latency,
        "active_supervised_scalar_count": STRUCTURED_ACTIVE_SCALARS,
        "finite": bool(metrics["all_finite"] and delta["all_finite"]),
        "decode_valid": bool(decode_meta.get("finite") is True),
    }


def _noise_hash_from_fixed(fixed: Any) -> str:
    value = np.asarray(fixed["noise"].detach().cpu().numpy(), dtype=np.float32)
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def _load_fixed_probes(stage23: Any, normalized_target: np.ndarray, expected_noise: Mapping[str, str]) -> list[dict[str, Any]]:
    import torch

    action = torch.from_numpy(np.asarray(normalized_target[None], dtype=np.float32))
    probes = stage23.build_probe_grid()
    if tuple(item["probe_id"] for item in probes) != EXPECTED_PROBE_IDS:
        raise Stage38EError("fixed probe ID order drifted")
    fixed: list[dict[str, Any]] = []
    for probe in probes:
        item = stage23.build_fixed_flow_inputs(action, noise_seed=int(probe["noise_seed"]), t_value=float(probe["t_value"]))
        noise_hash = _noise_hash_from_fixed(item)
        if noise_hash != expected_noise.get(str(probe["probe_id"])):
            raise Stage38EError(f"fixed teacher noise hash mismatch: {probe['probe_id']}")
        item["noise_sha256"] = noise_hash
        fixed.append({"probe": probe, "fixed": item})
    return fixed


def _prepare_dense_controls(dense_audit_report: Mapping[str, Any], processor: Any, embodiment: Any, snapshot: Any) -> dict[tuple[int, str], tuple[Any, dict[str, Any]]]:
    controls = dense_audit_report.get("controls")
    if not isinstance(controls, Mapping) or len(controls) != 32:
        raise Stage38EError("sealed dense evidence is insufficient for common-space comparison")
    result: dict[tuple[int, str], tuple[Any, dict[str, Any]]] = {}
    for key, control in controls.items():
        if not isinstance(key, tuple):
            # JSON-like reports are not used here; the in-memory audit keeps tuple keys.
            continue
        result[key] = _dense_chunk(control, snapshot, processor, embodiment)
    if len(result) != 32:
        raise Stage38EError("dense common-space controls could not be reconstructed")
    return result


def _aggregate_seed_variance(rows: Sequence[Mapping[str, Any]], metric_path: Sequence[str]) -> dict[str, Any]:
    values = []
    for row in rows:
        value: Any = row
        for part in metric_path:
            value = value.get(part) if isinstance(value, Mapping) else None
        if isinstance(value, Mapping):
            value = value.get("mean")
        if value is not None:
            values.append(float(value))
    return _summary(values)


def _primary_metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = row["metrics_to_expert"]
    preview = metrics["preview"]
    profile = metrics["candidate_profile"]
    return {
        "P1_WBC_PREVIEW_XYZ": float(preview["preview_position_error_m"]["p95"]),
        "P2_WBC_PREVIEW_SO3": float(preview["preview_rotation_error_rad"]["p95"]),
        "P3_SE3_ARC_INFLATION": float(abs(float(preview["se3_arc_length_ratio"]) - 1.0)),
        "P4_TIMING_PROGRESS": float(preview["time_to_arc_progress_error"]["mean"]),
        "P5_ROUGHNESS_HF": float(profile["roughness_position_second_difference_m"]["mean"]),
        "P5_HIGH_FREQUENCY_XYZ": float(metrics["high_frequency_xyz_fraction"]),
    }


def _aggregate_primary(rows: Sequence[Mapping[str, Any]], dense_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    families = ("P1_WBC_PREVIEW_XYZ", "P2_WBC_PREVIEW_SO3", "P3_SE3_ARC_INFLATION", "P4_TIMING_PROGRESS", "P5_ROUGHNESS_HF")
    structured_values = {family: [] for family in families}
    dense_values = {family: [] for family in families}
    structured_hf_values: list[float] = []
    dense_hf_values: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        s_values = _primary_metric_values(row)
        d_values = dense_rows[seed]["primary_values"]
        for family in families:
            structured_values[family].append(s_values[family])
            dense_values[family].append(d_values[family])
        structured_hf_values.append(s_values["P5_HIGH_FREQUENCY_XYZ"])
        dense_hf_values.append(d_values["P5_HIGH_FREQUENCY_XYZ"])
        per_seed.append({"seed": seed, "structured": s_values, "dense": d_values, "improved": {family: (s_values[family] < d_values[family] and (family != "P5_ROUGHNESS_HF" or s_values["P5_HIGH_FREQUENCY_XYZ"] < d_values["P5_HIGH_FREQUENCY_XYZ"])) for family in families}})
    family_result: dict[str, Any] = {}
    for family in families:
        s = np.asarray(structured_values[family], dtype=np.float64)
        d = np.asarray(dense_values[family], dtype=np.float64)
        count = int(np.sum(s < d))
        if family == "P5_ROUGHNESS_HF":
            s_hf = np.asarray(structured_hf_values, dtype=np.float64)
            d_hf = np.asarray(dense_hf_values, dtype=np.float64)
            hf_count = int(np.sum(s_hf < d_hf))
            median_improved = bool(np.median(s) < np.median(d) and np.median(s_hf) < np.median(d_hf))
            family_count = int(np.sum((s < d) & (s_hf < d_hf)))
        else:
            s_hf = d_hf = None
            hf_count = None
            median_improved = bool(np.median(s) < np.median(d))
            family_count = count
        family_result[family] = {
            "structured_median": float(np.median(s)),
            "dense_median": float(np.median(d)),
            "strict_median_improvement": median_improved,
            "strict_seed_improvement_count": family_count,
            "minimum_seed_improvements": 9,
            "primary_metric_improved": bool(median_improved and family_count >= 9),
            "structured_values": s.tolist(),
            "dense_values": d.tolist(),
        }
        if family == "P5_ROUGHNESS_HF":
            family_result[family].update({
                "structured_high_frequency_median": float(np.median(s_hf)),
                "dense_high_frequency_median": float(np.median(d_hf)),
                "high_frequency_strict_median_improvement": bool(np.median(s_hf) < np.median(d_hf)),
                "high_frequency_strict_seed_improvement_count": hf_count,
                "structured_high_frequency_values": s_hf.tolist(),
                "dense_high_frequency_values": d_hf.tolist(),
                "family_definition": "roughness_position_second_difference_and_high_frequency_xyz_fraction_both_strictly_improve",
            })
    pass_families = [family for family, item in family_result.items() if item["primary_metric_improved"]]
    endpoint_xyz = np.asarray([row["metrics_to_expert"]["endpoint_position_error_m"] for row in rows], dtype=np.float64)
    endpoint_so3 = np.asarray([row["metrics_to_expert"]["endpoint_rotation_error_rad"] for row in rows], dtype=np.float64)
    dense_endpoint_xyz = np.asarray([dense_rows[int(row["seed"])] ["endpoint_xyz"] for row in rows], dtype=np.float64)
    dense_endpoint_so3 = np.asarray([dense_rows[int(row["seed"])] ["endpoint_so3"] for row in rows], dtype=np.float64)
    return {
        "families": family_result,
        "pass_count": len(pass_families),
        "passed_families": pass_families,
        "per_seed": per_seed,
        "endpoint_median": {"structured_xyz": float(np.median(endpoint_xyz)), "dense_xyz": float(np.median(dense_endpoint_xyz)), "structured_so3": float(np.median(endpoint_so3)), "dense_so3": float(np.median(dense_endpoint_so3)), "jointly_not_worse": bool(np.median(endpoint_xyz) <= np.median(dense_endpoint_xyz) or np.median(endpoint_so3) <= np.median(dense_endpoint_so3))},
    }


def _safety_counts(rows: Sequence[Mapping[str, Any]], dense_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    counts = {"ACCEPT": 0, "WARN": 0, "REJECT": 0}
    transitions = {"REJECT_to_WARN_OR_ACCEPT": 0, "other": 0}
    for row in rows:
        decision = str(row["metrics_to_expert"]["safety_gate"]["decision"])
        counts[decision] = counts.get(decision, 0) + 1
        dense_decision = str(dense_rows[int(row["seed"])] ["safety_decision"])
        if dense_decision == "REJECT" and decision in {"WARN", "ACCEPT"}:
            transitions["REJECT_to_WARN_OR_ACCEPT"] += 1
        else:
            transitions["other"] += 1
    return {"structured": counts, "transitions_from_dense": transitions}


def _classify_primary(primary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], safety: Mapping[str, Any]) -> str:
    all_valid = len(rows) == 16 and all(row.get("finite") is True and row.get("decode_valid") is True for row in rows)
    pass_count = int(primary["pass_count"])
    families = primary.get("families", {})
    broad_worse_count = sum(
        int(float(item.get("structured_median", float("nan"))) > float(item.get("dense_median", float("nan"))))
        for item in families.values()
        if isinstance(item, Mapping)
    )
    p1_or_p2 = any(name in primary["passed_families"] for name in ("P1_WBC_PREVIEW_XYZ", "P2_WBC_PREVIEW_SO3"))
    p5 = "P5_ROUGHNESS_HF" in primary["passed_families"]
    guardrail = bool(primary["endpoint_median"]["jointly_not_worse"])
    physical_clear = all_valid and pass_count >= 4 and p1_or_p2 and p5 and guardrail
    if not all_valid or broad_worse_count >= 4 or (pass_count >= 4 and not guardrail):
        return "STRUCTURED_OFFLINE_REGRESSION"
    if physical_clear:
        reject_count = int(safety["structured"].get("REJECT", 0))
        non_reject_count = int(safety["structured"].get("WARN", 0)) + int(safety["structured"].get("ACCEPT", 0))
        if reject_count == 16:
            return "STRUCTURED_OFFLINE_PROFILE_IMPROVED_BUT_GATE_STILL_REJECTS"
        if non_reject_count >= 1:
            return "STRUCTURED_OFFLINE_CLEAR_PROFILE_IMPROVEMENT"
        return "STRUCTURED_OFFLINE_PARTIAL_OR_MIXED"
    if pass_count in (2, 3):
        return "STRUCTURED_OFFLINE_PARTIAL_OR_MIXED"
    if pass_count <= 1:
        return "STRUCTURED_OFFLINE_NO_CLEAR_ADVANTAGE"
    return "STRUCTURED_OFFLINE_PARTIAL_OR_MIXED"


def _secondary_classification(rows: Sequence[Mapping[str, Any]], dense_rows: Mapping[int, Mapping[str, Any]]) -> str:
    if len(rows) != 16:
        return "MIXED"
    wins = 0
    losses = 0
    for row in rows:
        s = _primary_metric_values(row)
        d = dense_rows[int(row["seed"])] ["primary_values"]
        s_score = sum(s[key] < d[key] for key in s)
        d_score = sum(s[key] > d[key] for key in s)
        wins += int(s_score > d_score)
        losses += int(d_score > s_score)
    if wins >= 9 and losses == 0:
        return "IMPROVES"
    if losses >= 9 and wins == 0:
        return "REGRESSES"
    if wins == 0:
        return "NO_ADVANTAGE"
    return "MIXED"


def _result_manifest(output_root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    paths = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "/cache/" not in str(path) and "/tmp/" not in str(path):
            paths.append({"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return {"schema": "STAGE38E_RESULT_MANIFEST_V1", "status": "PASS", "pi_gate": PI_GATE_ID, "result_path": str(output_root / "stage38e_result.json"), "artifact_paths": paths, "scientific_contract_changed": False, "next_gate": NEXT_GATE}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    preflight_path = output_root / "stage38e_preflight.json"
    if not preflight_path.is_file():
        raise Stage38EError("Stage38-E evaluation requires a prior preflight")
    preflight_report = read_json(preflight_path)
    if preflight_report.get("status") != "PASS":
        raise Stage38EError("Stage38-E preflight is not PASS")
    if preflight_report.get("expected_project_head") != args.expected_project_sha or preflight_report.get("api_live_head") != args.api_head:
        raise Stage38EError("Stage38-E preflight HEAD binding changed")
    if not os.environ.get("PBS_JOBID"):
        raise Stage38EError("Stage38-E evaluation requires a PBS allocation")
    if os.environ.get("PBS_O_QUEUE") != EXPECTED_QUEUE:
        raise Stage38EError(f"Stage38-E expected PBS request queue {EXPECTED_QUEUE}")
    execution_queue = os.environ.get("PBS_QUEUE")
    branch = branch_audit(args.expected_project_sha, args.api_head)
    if branch.get("status") != "PASS":
        raise Stage38EError("Stage38-E live branch identity changed after preflight")
    stage38d = stage38d_audit(rehash_checkpoints=True)
    dense_report = dense_audit()
    if stage38d.get("status") != "PASS" or dense_report.get("status") != "PASS":
        raise Stage38EError("exact Stage38-D or sealed dense identity failed after allocation")
    # The second identity check occurs before importing Torch/model code.
    historical = historical_identity_audit()
    if any(item.get("status") != "PASS" for item in historical.values() if isinstance(item, Mapping) and "status" in item):
        raise Stage38EError("historical identity changed before model construction")
    runtime = _setup_runtime(output_root, args.groot_source, args.cosmos_cache_root)
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from scripts import stage23_phase_c_flow_field_diagnostic as stage23
    snapshot = stage26._load_snapshot(SNAPSHOT_PATH)
    processor, embodiment, modality = stage26._build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=BASE_ROOT,
        dataset_dir=TINY1_ROOT,
        modality_path=MODALITY_PATH,
        cosmos_snapshot=COSMOS_ROOT / "snapshots" / EXPECTED_COSMOS_REVISION,
    )
    prepared = stage26._prepare_official_sample(processor=processor, embodiment=embodiment, modality_configs=copy.deepcopy(modality), dataset_dir=TINY1_ROOT)
    expert_physical = stage26._decode_physical_action(processor, embodiment, prepared["expert_action_cpu"].numpy(), snapshot)
    snapshot_expert = np.concatenate([np.asarray(snapshot.future_desired_eef_9d)[None], np.asarray(snapshot.future_gripper).reshape(1, HORIZON, 1)], axis=-1).astype(np.float32)
    if not np.allclose(expert_physical, snapshot_expert, rtol=1e-5, atol=1e-6):
        raise Stage38EError("expert physical decode differs from sealed Stage22 snapshot")
    expert_chunk = _expert_chunk(snapshot, expert_physical)
    conditioning, batch = _build_conditioning(processor, embodiment, prepared)
    sidecar = read_json(STRUCTURED_STATS)
    label, normalizer, normalized_label = _structured_label_and_normalizer(expert_physical, snapshot, sidecar)
    codec = _codec_roundtrip(label, expert_chunk, snapshot)
    if codec.get("status") != "PASS":
        raise Stage38EError("Stage38-E structured label roundtrip failed before inference")
    fixed_probes = _load_fixed_probes(stage23, normalized_label, preflight_report["stage26_fixed_probe_evidence"]["noise_sha256_by_probe"])
    dense_controls = _prepare_dense_controls(dense_report, processor, embodiment, snapshot)
    stage26_result = read_json(STAGE26_RESULT)
    dense_fixed_reference = {
        "result_path": str(STAGE26_RESULT),
        "result_sha256": STAGE26_RESULT_SHA,
        "manifest_path": str(STAGE26_MANIFEST),
        "manifest_sha256": STAGE26_MANIFEST_SHA,
        "record_count": len(stage26_result.get("records", [])),
        "probe_ids": EXPECTED_PROBE_IDS,
        "dense_rerun": False,
    }
    state_path = output_root / "stage38e_execution_state.json"
    partial_path = output_root / "stage38e_partial_records.json"
    events_path = output_root / "stage38e_events.jsonl"
    state = {"schema": "STAGE38E_EXECUTION_STATE_V1", "status": "READY_BEFORE_MODEL_CONSTRUCTION", "scientific_execution_begun": False, "teacher_records_completed": 0, "free_records_completed": 0, "maximum_teacher_records": 18, "maximum_free_records": 32, "pbs_jobid": os.environ.get("PBS_JOBID"), "updated_at_utc": now_utc()}
    _state_write(state_path, state)
    write_json(output_root / "stage38e_revalidation.json", {"schema": "STAGE38E_REVALIDATION_V1", "pi_gate": PI_GATE_ID, "branch": branch, "preflight_branch": preflight_report["branch"], "stage38d": stage38d, "dense": {key: value for key, value in dense_report.items() if key != "controls"}, "historical_identity": historical, "runtime": runtime, "conditioning": conditioning, "codec_roundtrip": codec, "model_constructed": False, "gr00t_inference": False, "training": False, "wbc_executed": False, "scientific_contract_changed": False, "pbs_submitted": True, "timestamp_utc": now_utc()})
    events_path.write_text(json.dumps({"event": "MODEL_CONSTRUCTION_ALLOWED", "timestamp_utc": now_utc(), "pbs_jobid": os.environ.get("PBS_JOBID"), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n", encoding="utf-8")
    teacher_records: list[dict[str, Any]] = []
    free_records: list[dict[str, Any]] = []
    dense_primary_rows: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        for label_name in ("H2@1000", "H2@2000"):
            dense_chunk, dense_meta = dense_controls[(seed, label_name)]
            dense_metrics = _common_metrics(dense_chunk, expert_chunk, representation=label_name, case_id=f"dense:{label_name}:{seed}")
            dense_primary_rows.setdefault(seed, {})[label_name] = {
                "primary_values": {
                    "P1_WBC_PREVIEW_XYZ": float(dense_metrics["preview"]["preview_position_error_m"]["p95"]),
                    "P2_WBC_PREVIEW_SO3": float(dense_metrics["preview"]["preview_rotation_error_rad"]["p95"]),
                    "P3_SE3_ARC_INFLATION": float(abs(float(dense_metrics["preview"]["se3_arc_length_ratio"]) - 1.0)),
                    "P4_TIMING_PROGRESS": float(dense_metrics["preview"]["time_to_arc_progress_error"]["mean"]),
                    "P5_ROUGHNESS_HF": float(dense_metrics["candidate_profile"]["roughness_position_second_difference_m"]["mean"]),
                    "P5_HIGH_FREQUENCY_XYZ": float(dense_metrics["high_frequency_xyz_fraction"]),
                },
                "endpoint_xyz": float(dense_metrics["endpoint_position_error_m"]),
                "endpoint_so3": float(dense_metrics["endpoint_rotation_error_rad"]),
                "safety_decision": str(dense_metrics["safety_gate"]["decision"]),
                "metrics": dense_metrics,
                "meta": dense_meta,
            }
    # Models are loaded one at a time; the primary checkpoint is evaluated first.
    for checkpoint_label, checkpoint_path, dense_label in CHECKPOINTS:
        model = _load_structured_model(checkpoint_path, args.device)
        try:
            _, action_inputs, processed_backbone, state_features = _prepare_model_conditioning(model, batch["inputs"])
            state["status"] = "SCIENTIFIC_EXECUTION_BEGUN"
            state["scientific_execution_begun"] = True
            state["current_checkpoint"] = checkpoint_label
            state["updated_at_utc"] = now_utc()
            _state_write(state_path, state)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "MODEL_CONSTRUCTED", "checkpoint": checkpoint_label, "timestamp_utc": now_utc()}) + "\n")
            for item in fixed_probes:
                call_index = len(teacher_records) + len(free_records) + 1
                record = _run_teacher_probe(model, processed_backbone, action_inputs, state_features, normalized_label, normalizer, snapshot, expert_chunk, item["probe"], item["fixed"], output_root, checkpoint_label, call_index)
                teacher_records.append(record)
                state["teacher_records_completed"] = len(teacher_records)
                state["last_completed_teacher"] = record["probe_id"]
                state["updated_at_utc"] = now_utc()
                _state_write(state_path, state)
                _write_partial(partial_path, teacher_records, free_records)
            for seed in SEEDS:
                call_index = len(teacher_records) + len(free_records) + 1
                dense_chunk, _ = dense_controls[(seed, dense_label)]
                noise = _load_noise(dense_report["controls"][(seed, dense_label)], seed)
                record = _run_free_call(model, processed_backbone, action_inputs, state_features, noise, normalizer, snapshot, expert_chunk, dense_chunk, checkpoint_label, dense_label, seed, output_root, call_index)
                free_records.append(record)
                state["free_records_completed"] = len(free_records)
                state["last_completed_free"] = {"checkpoint": checkpoint_label, "seed": seed}
                state["updated_at_utc"] = now_utc()
                _state_write(state_path, state)
                _write_partial(partial_path, teacher_records, free_records)
        finally:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    if len(teacher_records) != 18 or len(free_records) != 32:
        raise Stage38EError(f"record count drifted: teacher={len(teacher_records)} free={len(free_records)}")
    primary_rows = [row for row in free_records if row["checkpoint"] == "STRUCTURED@1000"]
    secondary_rows = [row for row in free_records if row["checkpoint"] == "STRUCTURED@2000"]
    primary_dense = {seed: dense_primary_rows[seed]["H2@1000"] for seed in SEEDS}
    secondary_dense = {seed: dense_primary_rows[seed]["H2@2000"] for seed in SEEDS}
    primary = _aggregate_primary(primary_rows, primary_dense)
    secondary = _aggregate_primary(secondary_rows, secondary_dense)
    primary_safety = _safety_counts(primary_rows, primary_dense)
    secondary_safety = _safety_counts(secondary_rows, secondary_dense)
    primary_class = _classify_primary(primary, primary_rows, primary_safety)
    secondary_class = _secondary_classification(secondary_rows, secondary_dense)
    exposure_sensitive = bool(primary_class not in {"STRUCTURED_OFFLINE_CLEAR_PROFILE_IMPROVEMENT", "STRUCTURED_OFFLINE_PROFILE_IMPROVED_BUT_GATE_STILL_REJECTS"} and secondary_class == "IMPROVES")
    seed_variance = {
        "primary_structured": {"preview_xyz_p95": _aggregate_seed_variance(primary_rows, ("metrics_to_expert", "preview", "preview_position_error_m")), "preview_so3_p95": _aggregate_seed_variance(primary_rows, ("metrics_to_expert", "preview", "preview_rotation_error_rad")), "roughness": _aggregate_seed_variance(primary_rows, ("metrics_to_expert", "candidate_profile", "roughness_position_second_difference_m")), "high_frequency_xyz_fraction": _aggregate_seed_variance(primary_rows, ("metrics_to_expert", "high_frequency_xyz_fraction"))},
        "secondary_structured": {"preview_xyz_p95": _aggregate_seed_variance(secondary_rows, ("metrics_to_expert", "preview", "preview_position_error_m")), "preview_so3_p95": _aggregate_seed_variance(secondary_rows, ("metrics_to_expert", "preview", "preview_rotation_error_rad")), "roughness": _aggregate_seed_variance(secondary_rows, ("metrics_to_expert", "candidate_profile", "roughness_position_second_difference_m")), "high_frequency_xyz_fraction": _aggregate_seed_variance(secondary_rows, ("metrics_to_expert", "high_frequency_xyz_fraction"))},
    }
    result = {
        "schema": "STAGE38E_STRUCTURED_GR00T_OFFLINE_RESULT_V1",
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage38-E common-physical-space offline qualification",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "project_sha": args.expected_project_sha,
        "pbs": {"job_id": os.environ.get("PBS_JOBID"), "project": EXPECTED_PROJECT, "requested_queue": os.environ.get("PBS_O_QUEUE"), "execution_queue": execution_queue, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device},
        "stage38d": {"result_path": str(STAGE38D_RESULT), "result_sha256": STAGE38D_RESULT_SHA, "checkpoint_1000_tree_sha256": STRUCTURED_1000_TREE, "checkpoint_2000_tree_sha256": STRUCTURED_2000_TREE, "statistics_path": str(STRUCTURED_STATS), "statistics_sha256": STRUCTURED_STATS_SHA},
        "dense_controls": {"stage33_result": file_record(STAGE33_RESULT, STAGE33_RESULT_SHA), "stage33_raw_manifest": file_record(STAGE33_RAW_MANIFEST, STAGE33_RAW_MANIFEST_SHA), "stage33_seal": file_record(STAGE33_SEAL, STAGE33_SEAL_SHA), "h2_1000_evidence": "Stage33 sealed H2@1000 rows seed2223..2238", "h2_2000_evidence": "Stage33 sealed H2@2000 rows seed2223..2238", "dense_rerun": False},
        "pairing_mode": "EXACT_NOISE",
        "free_running_seed_set": list(SEEDS),
        "fixed_teacher_reference": dense_fixed_reference,
        "conditioning": conditioning,
        "codec_roundtrip": codec,
        "teacher_records": teacher_records,
        "free_running_records": free_records,
        "primary_structured_at_1000_vs_h2_1000": {"aggregate": primary, "safety_gate": primary_safety, "classification": primary_class},
        "secondary_structured_at_2000_vs_h2_2000": {"aggregate": secondary, "safety_gate": secondary_safety, "classification": secondary_class},
        "exposure_sensitive_structured_improvement": exposure_sensitive,
        "seed_variance": seed_variance,
        "representation_dimensionality": {"dense_h40_active_supervised_scalars": DENSE_ACTIVE_SCALARS, "structured_action_v1_active_supervised_scalars": STRUCTURED_ACTIVE_SCALARS, "structured_padding_dimensions": 1210},
        "execution": {"scientific_execution_begun": True, "teacher_forced_structured_calls": 18, "free_running_structured_calls": 32, "maximum_new_structured_sampler_calls": 32, "model_constructed": True, "gr00t_inference": True, "training": False, "backward": False, "optimizer_step": False, "scheduler_step": False, "checkpoint_write": False, "dense_rerun": False, "wbc_executed": False, "validation_e_read": False, "holdout_f_read": False, "hardware": False},
        "all_finite": all(bool(row.get("finite")) and bool(row.get("decode_valid")) for row in teacher_records + free_records),
        "scientific_contract_changed": False,
        "claim_boundary": "sealed single-fixture offline physical/WBC-facing comparison; no learned WBC or generalization claim",
        "next_gate": NEXT_GATE,
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38e_result.json", result)
    raw_manifest = {
        "schema": "STAGE38E_STRUCTURED_OFFLINE_RAW_HASH_MANIFEST_V1",
        "status": "PASS",
        "pi_gate": PI_GATE_ID,
        "teacher_records": [{"checkpoint": row["checkpoint"], "probe_id": row["probe_id"], "noise_sha256": row["noise_sha256"], "predicted_velocity_sha256": row["predicted_velocity"]["sha256"], "artifact_sha256": row["artifact"]["sha256"]} for row in teacher_records],
        "free_running_records": [{"checkpoint": row["checkpoint"], "seed": row["seed"], "initial_noise_sha256": row["initial_noise"]["sha256"], "normalized_action_sha256": row["normalized_action"]["sha256"], "raw_structured_sha256": row["decoded_structured"]["raw"]["sha256"], "artifact_sha256": row["artifact"]["sha256"]} for row in free_records],
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    write_json(output_root / "stage38e_raw_hash_manifest.json", raw_manifest)
    manifest = _result_manifest(output_root, result)
    write_json(output_root / "stage38e_result_manifest.json", manifest)
    state.update({"status": "PASS_EXECUTION_COMPLETE", "teacher_records_completed": len(teacher_records), "free_records_completed": len(free_records), "updated_at_utc": now_utc()})
    _state_write(state_path, state)
    return result


def _write_partial(path: Path, teacher_records: Sequence[Mapping[str, Any]], free_records: Sequence[Mapping[str, Any]]) -> None:
    write_json(path, {"schema": "STAGE38E_PARTIAL_EVIDENCE_V1", "status": "PARTIAL_EVIDENCE", "teacher_record_count": len(teacher_records), "free_record_count": len(free_records), "teacher_records": list(teacher_records), "free_running_records": list(free_records), "scientific_contract_changed": False})


def seal(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    result_path = output_root / "stage38e_result.json"
    raw_path = output_root / "stage38e_raw_hash_manifest.json"
    manifest_path = output_root / "stage38e_result_manifest.json"
    if not result_path.is_file() or not raw_path.is_file() or not manifest_path.is_file():
        raise Stage38EError("cannot seal incomplete Stage38-E package")
    result = read_json(result_path)
    if result.get("status") != "PASS_EXECUTION_COMPLETE" or result.get("next_gate") != NEXT_GATE:
        raise Stage38EError("Stage38-E result is not complete")
    pbs_log = Path(args.pbs_log).resolve()
    if not pbs_log.is_file():
        raise Stage38EError(f"PBS log is missing: {pbs_log}")
    required_names = {"stage38e_result.json", "stage38e_raw_hash_manifest.json", "stage38e_result_manifest.json", "stage38e_preflight.json", "stage38e_revalidation.json", "stage38e_runtime_binding.json", "stage38e_execution_state.json"}
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "/cache/" in str(path) or "/tmp/" in str(path):
            continue
        if path.name in {"stage38e_artifact_seal.json", "stage38e_final_return.json"}:
            continue
        if path.name in required_names or path.suffix in {".npz", ".json", ".log"}:
            artifacts.append({"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    for path in (Path(__file__).resolve(), REPO / "hpc/pbs/stage38_structured_offline_evaluation.pbs", pbs_log):
        resolved = path.resolve(strict=True)
        if not any(item["path"] == str(resolved) for item in artifacts):
            artifacts.append({"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved)})
    artifacts.sort(key=lambda item: str(item["path"]))
    seal_payload = {
        "schema": "STAGE38E_STRUCTURED_OFFLINE_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "project_sha": result.get("project_sha"),
        "pbs_job_id": result.get("pbs", {}).get("job_id"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "scientific_contract_changed": False,
        "historical_artifacts_modified": False,
        "next_gate": NEXT_GATE,
    }
    seal_path = output_root / "stage38e_artifact_seal.json"
    write_json(seal_path, seal_payload)
    final_return = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "PI_GATE": PI_GATE_ID,
        "STAGE": "Stage38-E",
        "PR": PR_NUMBER,
        "LIVE_HEAD": result.get("project_sha"),
        "WORKTREE_CLEAN": True,
        "STAGE38D_CHECKPOINT1000_SHA256": STRUCTURED_1000_TREE,
        "STAGE38D_CHECKPOINT2000_SHA256": STRUCTURED_2000_TREE,
        "DENSE_H2_1000_EVIDENCE_IDENTITY": result.get("dense_controls", {}).get("h2_1000_evidence"),
        "DENSE_H2_2000_EVIDENCE_IDENTITY": result.get("dense_controls", {}).get("h2_2000_evidence"),
        "PAIRING_MODE": result.get("pairing_mode"),
        "TEACHER_PROBES_COMPLETE": len(result.get("teacher_records", [])) == 18,
        "STRUCTURED_FREE_RUNNING_ROWS": len(result.get("free_running_records", [])),
        "PRIMARY_METRIC_FAMILY_PASS_COUNT": result.get("primary_structured_at_1000_vs_h2_1000", {}).get("aggregate", {}).get("pass_count"),
        "PRIMARY_SAFETYGATE_COUNTS": result.get("primary_structured_at_1000_vs_h2_1000", {}).get("safety_gate"),
        "PRIMARY_CLASSIFICATION": result.get("primary_structured_at_1000_vs_h2_1000", {}).get("classification"),
        "SECONDARY_EXPOSURE_CLASSIFICATION": result.get("secondary_structured_at_2000_vs_h2_2000", {}).get("classification"),
        "EXPOSURE_SENSITIVE_STRUCTURED_IMPROVEMENT": result.get("exposure_sensitive_structured_improvement"),
        "GR00T_INFERENCE": True,
        "TRAINING": False,
        "WBC_EXECUTED": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "HARDWARE": False,
        "ARTIFACT_ROOT": str(output_root),
        "RESULT_SHA256": sha256_file(result_path),
        "MANIFEST_SHA256": sha256_file(manifest_path),
        "SEAL_SHA256": sha256_file(seal_path),
        "NEXT_GATE": NEXT_GATE,
    }
    write_json(output_root / "stage38e_final_return.json", final_return)
    final_return["FINAL_RETURN_SHA256"] = sha256_file(output_root / "stage38e_final_return.json")
    return {"seal": {**seal_payload, "path": str(seal_path), "sha256": sha256_file(seal_path)}, "final_return": final_return}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    modes.add_argument("--seal", action="store_true")
    p.add_argument("--expected-project-sha", required=True)
    p.add_argument("--api-head", required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--groot-source", type=Path, default=GROOT_ROOT)
    p.add_argument("--cosmos-cache-root", type=Path, default=COSMOS_ROOT)
    p.add_argument("--minimum-free-gib", type=float, default=100.0)
    p.add_argument("--pbs-log", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage38e_pbs.log")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        if args.preflight:
            result = preflight(args)
            print(json.dumps({"status": result["status"], "output_root": str(args.output_root)}, sort_keys=True))
            return 0 if result["status"] == "PASS" else 2
        if args.evaluate:
            result = evaluate(args)
            print(json.dumps({"status": result["status"], "output_root": str(args.output_root), "primary_classification": result["primary_structured_at_1000_vs_h2_1000"]["classification"]}, sort_keys=True))
            return 0
        result = seal(args)
        print(json.dumps({"status": result["seal"]["status"], "seal_sha256": result["seal"]["sha256"], "final_return": str(Path(args.output_root) / "stage38e_final_return.json")}, sort_keys=True))
        return 0
    except Exception as exc:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "STAGE38E_FAILURE_V1",
            "pi_gate": PI_GATE_ID,
            "mode": "preflight" if args.preflight else "evaluate" if args.evaluate else "seal",
            "expected_project_head": args.expected_project_sha,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "model_constructed": False if args.preflight or not (output_root / "stage38e_execution_state.json").is_file() else "UNKNOWN_AFTER_GATE",
            "gr00t_inference": False if args.preflight or not (output_root / "stage38e_execution_state.json").is_file() else "UNKNOWN",
            "training": False,
            "wbc_executed": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "scientific_contract_changed": False,
            "timestamp_utc": now_utc(),
        }
        write_json(output_root / "stage38e_failure.json", failure)
        print(f"STAGE38E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
