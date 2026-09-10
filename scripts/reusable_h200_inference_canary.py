#!/usr/bin/env python3
"""Reusable one-window H200 inference qualification canary.

This is the committed qualification entrypoint.  It binds the materialized
Hopper state immediately before the first GR00T/Transformers import, then
hands one sealed raw prediction to the generic CPU evaluator.  It never
trains, writes checkpoints, reads locked evaluation data, retries, or submits
PBS jobs.  Runtime paths and the expected checkout identity are supplied by
an external, late-bound JSON configuration; no node-local runtime path is
embedded in this source file.
"""

from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Mapping

# This bootstrap is intentionally before every repository ``scripts`` or
# ``go2arx5_vla`` import.  ``scripts`` is a namespace package (no
# ``__init__.py``), so both the repository parent and ``src`` are required
# when this file is invoked by absolute path from a foreign CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _import_root in (REPO_ROOT, SRC_ROOT):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

import numpy as np


# These values are the frozen PI0102 one-window scientific contract.  Runtime
# identity, device, and artifact paths are deliberately not defaults.
EXPECTED_HEAD = ""
EXPECTED_BRANCH = ""
EXPECTED_CANARY_TOKEN = "REUSABLE_H200_V1_INFERENCE_CANARY_PASS_FOR_PI_REVIEW"
EXECUTION_GATE = ""
DEVICE = ""

CHECKPOINT_LABEL = "FULL24@2000"
WINDOW_INDEX = 0
WINDOW_ID = "stage22_expert_red_target_train_a:expert-0000"
TARGET_ID = "red_target_train_a"
SAMPLER_PROFILE = "GROOT_N1D7_SOURCE_MAPPED_EULER4_V1"
TIMESTEPS = [0, 250, 500, 750]
SAMPLER_DT = 0.25
SAMPLER_CALLS = 4
NOISE_SEED = 4242

DATASET: Path
SNAPSHOT_ROOT: Path
CHECKPOINT: Path
BASE_CHECKPOINT: Path
STATS: Path
GROOT_SOURCE: Path
COSMOS_ROOT: Path
MODALITY: Path
GROOT_RUNTIME_MANIFEST: Path
STAGE26_FLOW: Path
RUNTIME_CONFIG_PATH: Path | None = None
RUNTIME_CONFIG_SHA256 = ""
QUALIFICATION_ENTRYPOINT = Path(__file__).resolve()


class CanaryFailure(RuntimeError):
    """Raised for a fail-closed canary condition."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryFailure(message)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        jsonable(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    jsonable(row),
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def event(evidence_root: Path, phase: str, **details: Any) -> None:
    path = evidence_root / "execution_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"time_utc": now_utc(), "phase": phase, **jsonable(details)},
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CanaryFailure(f"required file is unavailable: {path}") from exc
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_string(value: Any, *, field: str) -> str:
    require(type(value) is str and bool(value.strip()), f"{field} must be a non-empty string")
    return value.strip()


def _config_path(paths: Mapping[str, Any], *, field: str) -> Path:
    raw = _config_string(paths.get(field), field=f"paths.{field}")
    candidate = Path(raw).expanduser()
    require(candidate.is_absolute(), f"paths.{field} must be absolute")
    return candidate


def _validate_repo_root(repo: Path) -> Path:
    """Require the explicit runtime root to be this committed checkout."""

    candidate = Path(repo)
    require(not candidate.is_symlink(), f"repo root may not be a symlink: {candidate}")
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure(f"repo root cannot be resolved: {candidate}") from exc
    require(candidate == REPO_ROOT, f"repo root does not match launcher checkout: {candidate}")
    require((candidate / "scripts").is_dir(), f"repo scripts directory is missing: {candidate}")
    require((candidate / "src").is_dir(), f"repo src directory is missing: {candidate}")
    return candidate


def _activate_runtime_config(config_path: Path, repo: Path) -> dict[str, Any]:
    """Load late-bound runtime identity without importing model-side code."""

    global BASE_CHECKPOINT
    global CHECKPOINT
    global COSMOS_ROOT
    global DATASET
    global DEVICE
    global EXECUTION_GATE
    global EXPECTED_BRANCH
    global EXPECTED_HEAD
    global GROOT_RUNTIME_MANIFEST
    global GROOT_SOURCE
    global MODALITY
    global RUNTIME_CONFIG_PATH
    global RUNTIME_CONFIG_SHA256
    global SNAPSHOT_ROOT
    global STAGE26_FLOW
    global STATS

    _validate_repo_root(repo)
    candidate = Path(config_path)
    require(not candidate.is_symlink(), f"runtime config may not be a symlink: {candidate}")
    require(candidate.is_file(), f"runtime config is unavailable: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryFailure(f"runtime config is not valid JSON: {candidate}") from exc
    require(isinstance(payload, dict), "runtime config must be a JSON object")
    require(
        payload.get("schema") == "REUSABLE_H200_V1_CANARY_RUNTIME_CONFIG_V1",
        "runtime config schema mismatch",
    )

    expected_head = _config_string(payload.get("expected_head"), field="expected_head")
    require(
        re.fullmatch(r"[0-9a-fA-F]{40}", expected_head) is not None,
        "expected_head must be a 40-character hexadecimal Git SHA",
    )
    expected_branch = _config_string(payload.get("expected_branch"), field="expected_branch")
    execution_gate = _config_string(payload.get("execution_gate"), field="execution_gate")
    require(
        re.fullmatch(r"PI_GATE_STATE_[0-9]{4}", execution_gate) is not None,
        "execution_gate must be a PI_GATE_STATE_NNNN token",
    )
    require(
        payload.get("canary_token") == EXPECTED_CANARY_TOKEN,
        "runtime config canary token is not the frozen token",
    )

    device = _config_string(payload.get("device"), field="device")
    require(device.startswith("cuda:"), "device must be an explicit CUDA device")

    contract = payload.get("scientific_contract")
    require(isinstance(contract, dict), "scientific_contract is required")
    require(
        contract.get("checkpoint_label") == CHECKPOINT_LABEL,
        "checkpoint label differs from the frozen one-window contract",
    )
    require(type(contract.get("window_index")) is int, "scientific_contract.window_index must be JSON integer")
    require(
        contract.get("window_index") == WINDOW_INDEX,
        "window index differs from the frozen one-window contract",
    )
    require(contract.get("window_id") == WINDOW_ID, "window ID differs from the frozen canary contract")
    require(contract.get("target_id") == TARGET_ID, "target ID differs from the frozen canary contract")
    require(contract.get("sampler_profile") == SAMPLER_PROFILE, "sampler profile differs from the frozen contract")
    contract_timesteps = contract.get("timesteps")
    require(isinstance(contract_timesteps, list), "scientific_contract.timesteps must be a list")
    require(
        all(type(value) is int for value in contract_timesteps),
        "scientific_contract.timesteps must contain JSON integers",
    )
    require(contract_timesteps == TIMESTEPS, "sampler timesteps differ from the frozen canary contract")
    require(type(contract.get("dt")) in (int, float), "scientific_contract.dt must be a JSON number")
    require(float(contract["dt"]) == SAMPLER_DT, "sampler dt differs from the frozen canary contract")
    require(type(contract.get("expected_calls")) is int, "scientific_contract.expected_calls must be JSON integer")
    require(
        contract.get("expected_calls") == SAMPLER_CALLS,
        "sampler call count differs from the frozen canary contract",
    )
    require(type(contract.get("noise_seed")) is int, "scientific_contract.noise_seed must be JSON integer")
    require(contract.get("noise_seed") == NOISE_SEED, "noise seed differs from the frozen canary contract")

    paths = payload.get("paths")
    require(isinstance(paths, dict), "runtime config paths mapping is required")
    required_paths = (
        "dataset",
        "snapshot_root",
        "checkpoint",
        "base_checkpoint",
        "stats",
        "groot_source",
        "cosmos_root",
        "modality",
        "groot_runtime_manifest",
        "stage26_flow",
    )
    resolved_paths = {key: _config_path(paths, field=key) for key in required_paths}

    EXPECTED_HEAD = expected_head
    EXPECTED_BRANCH = expected_branch
    EXECUTION_GATE = execution_gate
    DEVICE = device
    DATASET = resolved_paths["dataset"]
    SNAPSHOT_ROOT = resolved_paths["snapshot_root"]
    CHECKPOINT = resolved_paths["checkpoint"]
    BASE_CHECKPOINT = resolved_paths["base_checkpoint"]
    STATS = resolved_paths["stats"]
    GROOT_SOURCE = resolved_paths["groot_source"]
    COSMOS_ROOT = resolved_paths["cosmos_root"]
    MODALITY = resolved_paths["modality"]
    GROOT_RUNTIME_MANIFEST = resolved_paths["groot_runtime_manifest"]
    STAGE26_FLOW = resolved_paths["stage26_flow"]
    RUNTIME_CONFIG_PATH = candidate.resolve()
    RUNTIME_CONFIG_SHA256 = sha256_file(RUNTIME_CONFIG_PATH)
    return payload


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_artifact(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"file artifact unavailable: {path}")
    return {"kind": "file", "path": str(path), "sha256": sha256_file(path)}


def tree_artifact(path: Path) -> dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), f"tree artifact unavailable: {path}")
    from go2arx5_vla.experiments.reusable import sha256_tree

    return {"kind": "tree", "path": str(path), "sha256": sha256_tree(path)}


def run_text(argv: list[str], *, required: bool = True) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if required:
        require(
            completed.returncode == 0,
            f"command failed ({completed.returncode}): {' '.join(argv)}: "
            f"{completed.stderr.strip()}",
        )
    return completed.stdout.strip()


def observe_repository(repo: Path) -> dict[str, Any]:
    head = run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
    branch = run_text(["git", "-C", str(repo), "branch", "--show-current"])
    status = run_text(["git", "-C", str(repo), "status", "--porcelain=v1"])
    require(head.lower() == EXPECTED_HEAD, f"execution HEAD drift: {head}")
    require(branch == EXPECTED_BRANCH, f"execution branch drift: {branch}")
    require(status == "", f"execution worktree is dirty: {status}")
    return {
        "path": str(repo),
        "branch": branch,
        "head": head,
        "worktree_clean": True,
    }


def observe_gpu() -> dict[str, Any]:
    query = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line.strip() for line in query.splitlines() if line.strip()]
    require(len(rows) >= 1, "nvidia-smi returned no GPU")
    fields = [part.strip() for part in rows[0].split(",")]
    require(len(fields) >= 5, f"nvidia-smi identity row is malformed: {rows[0]}")
    name = fields[1]
    require("H200" in name, f"admitted GPU is not H200: {name}")
    return {
        "query": query,
        "index": fields[0],
        "name": name,
        "uuid": fields[2],
        "memory_total_mib": fields[3],
        "driver_version": fields[4],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": DEVICE,
    }


def model_modules_present() -> list[str]:
    prefixes = ("torch", "transformers", "gr00t", "safetensors")
    return sorted(
        name
        for name in sys.modules
        if name == "torch"
        or name == "transformers"
        or name == "gr00t"
        or name.startswith(prefixes[0] + ".")
        or name.startswith(prefixes[1] + ".")
        or name.startswith(prefixes[2] + ".")
        or name.startswith(prefixes[3] + ".")
    )


def _bootstrap_self_check(repo: Path) -> dict[str, Any]:
    """Exercise the committed import/admission boundary without model code."""

    import importlib
    import importlib.util

    resolved_repo = _validate_repo_root(repo)
    expected_repo = str(resolved_repo)
    expected_src = str(resolved_repo / "src")
    require(expected_repo in sys.path, "repository root was not added to sys.path")
    require(expected_src in sys.path, "repository src was not added to sys.path")

    dependency_name = "scripts.stage45c_h200_eval_runtime_adapter"
    dependency_spec = importlib.util.find_spec(dependency_name)
    require(dependency_spec is not None, f"canary dependency is not importable: {dependency_name}")
    dependency = importlib.import_module(dependency_name)
    require(not model_modules_present(), "model package imported during scripts bootstrap")
    static_guard = dependency._static_guard()
    require(static_guard.get("status") == "PASS", "canary dependency static guard failed")
    require(
        static_guard.get("top_level_forbidden_imports") == [],
        "canary dependency has a top-level model import",
    )

    # Touch the same admission validator used by the real canary with an
    # intentionally incomplete document.  The expected error proves that the
    # zero-model self-check reaches the admission boundary and fails closed;
    # it does not create any runtime artifact or enter model code.
    from go2arx5_vla.evaluation.h200_admission import (
        H200AdmissionError,
        validate_h200_inference_admission,
    )

    try:
        validate_h200_inference_admission(
            {},
            {},
            manifest_path=resolved_repo / "does-not-exist-for-bootstrap-self-check.json",
        )
    except H200AdmissionError:
        admission_boundary = "REACHED_FAIL_CLOSED"
    else:
        raise CanaryFailure("incomplete self-check admission unexpectedly passed")

    modules = model_modules_present()
    require(not modules, f"model modules entered during zero-model self-check: {modules}")
    return {
        "schema": "REUSABLE_H200_V1_CANARY_BOOTSTRAP_SELF_CHECK_V1",
        "status": "PASS",
        "repository_root": expected_repo,
        "src_root": expected_src,
        "import_roots": {
            "repository": expected_repo in sys.path,
            "src": expected_src in sys.path,
        },
        "scripts_dependency": dependency_name,
        "scripts_dependency_origin": str(dependency_spec.origin),
        "scripts_dependency_imported": True,
        "admission_boundary": admission_boundary,
        "model_modules_present": modules,
        "torch_imported": False,
        "gr00t_imported": False,
        "transformers_imported": False,
        "cuda_entered": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "cwd": os.getcwd(),
        "python": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def write_npy_durable(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def first_image(value: Any, *, name: str) -> tuple[np.ndarray, str]:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    source = np.asarray(value)
    if source.dtype == object:
        try:
            source = np.stack([np.asarray(item) for item in value], axis=0)
        except (TypeError, ValueError) as exc:
            raise CanaryFailure(f"{name} image object array is not stackable") from exc
    if source.ndim == 4 and source.shape[0] == 1:
        source = source[0]
    if source.ndim == 4 and source.shape[0] == 40:
        source = source[0]
    if source.ndim == 3 and source.shape == (3, 96, 96):
        source = np.moveaxis(source, 0, -1)
    require(tuple(source.shape) == (96, 96, 3), f"{name} source image shape drift: {source.shape}")
    require(np.issubdtype(source.dtype, np.integer), f"{name} source image is not integer")
    require(bool(np.isfinite(source).all()), f"{name} source image is not finite")
    require(bool(np.all((source >= 0) & (source <= 255))), f"{name} source image is out of uint8 range")
    return source.astype(np.uint8, copy=False), str(source.dtype)


def build_manifest_and_admission(
    *,
    repo: Path,
    evidence_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, dict[str, Any]]:
    """Recompute manifest and observed admission from this Hopper process."""

    require(not output_root.exists(), f"fresh output root already exists: {output_root}")
    require(not output_root.is_symlink(), f"output root is a symlink: {output_root}")
    event(evidence_root, "observed_state_start", output_root=str(output_root))

    repository = observe_repository(repo)
    gpu = observe_gpu()
    job_id = os.environ.get("PBS_JOBID", "").strip()
    require(job_id, "PBS_JOBID is missing")
    queue = os.environ.get("PBS_QUEUE", "").strip()
    require(queue, "PBS_QUEUE is missing from the actual PBS runtime")
    project = os.environ.get("PBS_ACCOUNT", "").strip() or os.environ.get(
        "PBS_PROJECT", ""
    ).strip()
    require(project, "PBS_ACCOUNT/PBS_PROJECT is missing from the actual PBS runtime")
    python_real = str(Path(sys.executable).resolve())
    environment_identity = (
        f"host={socket.gethostname()};kernel={platform.release()};"
        f"python={platform.python_version()};executable={python_real};"
        f"job={job_id}"
    )
    runtime_identity = f"{EXECUTION_GATE}:{EXPECTED_HEAD}:{job_id}:{socket.gethostname()}"

    require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE is not enabled")
    require(
        os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "TRANSFORMERS_OFFLINE is not enabled",
    )
    require(
        os.environ.get("HF_DATASETS_OFFLINE") == "1",
        "HF_DATASETS_OFFLINE is not enabled",
    )

    # These are runtime-owned checks, before any model package import.  The
    # old adapter supplies the already accepted canonical TRAIN24 materializer
    # and V2 runtime-statistics checks; importing that module is model-free.
    import importlib

    legacy = importlib.import_module("scripts.stage45c_h200_eval_runtime_adapter")
    require(not model_modules_present(), "model package was imported before admission")
    legacy._verify_runtime_stats(STATS)
    legacy._verify_dataset(DATASET)
    legacy._hash_identity(
        BASE_CHECKPOINT / "processor_config.json",
        legacy.BASE_PROCESSOR_CONFIG_SHA256,
        "BASE processor_config.json",
    )
    legacy._hash_identity(
        BASE_CHECKPOINT / "statistics.json",
        legacy.BASE_PROCESSOR_STATS_SHA256,
        "BASE processor statistics",
    )
    require(sha256_file(MODALITY) == legacy.MODALITY_SHA256, "modality identity drift")
    legacy._verify_groot_runtime_manifest(GROOT_RUNTIME_MANIFEST)
    cosmos_tree_sha, cosmos_entry_count = legacy.cosmos_tree_digest(COSMOS_ROOT)
    require(cosmos_tree_sha == legacy.COSMOS_TREE_SHA256, "Cosmos tree identity drift")
    cosmos_snapshot = COSMOS_ROOT / "snapshots" / legacy.COSMOS_REVISION
    require(cosmos_snapshot.is_dir() and not cosmos_snapshot.is_symlink(), "Cosmos snapshot unavailable")
    refs_main = COSMOS_ROOT / "refs" / "main"
    require(refs_main.is_file() and refs_main.read_text(encoding="utf-8").strip() == legacy.COSMOS_REVISION, "Cosmos refs/main drift")
    for relative, anchors in {
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
            "process_backbone_output",
            "self.state_encoder",
            "self.action_encoder",
            "return_all_hidden_states=True",
            "self.action_decoder",
        ),
        "gr00t/model/gr00t_n1d7/setup.py": (
            "AutoModel.from_pretrained",
            "load_bf16",
        ),
    }.items():
        source_file = GROOT_SOURCE / relative
        require(source_file.is_file() and not source_file.is_symlink(), f"GR00T source file unavailable: {source_file}")
        source_text = source_file.read_text(encoding="utf-8")
        for anchor in anchors:
            require(anchor in source_text, f"GR00T source anchor missing: {relative}: {anchor}")

    event(
        evidence_root,
        "materialized_inputs_observed",
        repository=repository,
        gpu=gpu,
        job_id=job_id,
        queue=queue,
        project=project,
        checkpoint_path=str(CHECKPOINT),
        statistics_path=str(STATS),
        dataset_path=str(DATASET),
        groot_source_path=str(GROOT_SOURCE),
        cosmos_root=str(COSMOS_ROOT),
        model_modules_present=model_modules_present(),
    )

    snapshots, canonical_label_source = legacy._canonical_parquet_snapshot_records(
        DATASET, SNAPSHOT_ROOT, STATS
    )
    require(len(snapshots) == 24, f"TRAIN24 materialization count drift: {len(snapshots)}")
    window = dict(snapshots[WINDOW_INDEX])
    require(window["window_id"] == WINDOW_ID, "selected window ID drift")
    require(window["target_id"] == TARGET_ID, "selected target ID drift")
    require(
        [item["window_id"] for item in snapshots[:1]] == [WINDOW_ID],
        "selected window order is not explicit target-major order",
    )

    protocol_path = repo / "manifests/STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1.json"
    mapping_path = repo / "manifests/STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1.json"
    protocol_sha = sha256_file(protocol_path)
    mapping_artifact = file_artifact(mapping_path)
    checkpoint_artifact = tree_artifact(CHECKPOINT)
    statistics_artifact = file_artifact(STATS)
    dataset_artifact = tree_artifact(DATASET)

    noise_path = evidence_root / "noise_bank.npy"
    noise = np.random.default_rng(NOISE_SEED).standard_normal(
        (1, 40, 132), dtype=np.float32
    )
    write_npy_durable(noise_path, noise)
    noise_artifact = file_artifact(noise_path)
    launcher_artifact = file_artifact(QUALIFICATION_ENTRYPOINT)
    require(RUNTIME_CONFIG_PATH is not None, "runtime config was not activated")
    runtime_config_artifact = file_artifact(RUNTIME_CONFIG_PATH)

    manifest_runtime_late_bound = {
        "repository_execution_head": repository["head"],
        "python_runtime": f"{python_real};version={platform.python_version()}",
        "container_or_environment": environment_identity,
        "groot_source": str(GROOT_SOURCE),
        "cosmos_cache_root": str(COSMOS_ROOT),
        "checkpoint_paths": {CHECKPOINT_LABEL: dict(checkpoint_artifact)},
        "dataset_path": dict(dataset_artifact),
        "statistics_path": {CHECKPOINT_LABEL: dict(statistics_artifact)},
        "output_root": str(output_root),
        "scheduler_project": project,
        "scheduler_queue": queue,
        "scheduler_resources": "select=1:ngpus=1,walltime=08:00:00",
        "gpu_device": f"{DEVICE};gpu={gpu['name']};uuid={gpu['uuid']}",
        "host": socket.gethostname(),
        "offline_mode": True,
        "runtime_admission_identity": runtime_identity,
    }
    manifest = {
        "schema": "GO2ARX5_REUSABLE_CHECKPOINT_EVALUATION_MANIFEST_V1",
        "status": "READY_NOT_EXECUTED",
        "experiment": {
            "id": "reusable-h200-v1-full24-at-2000-one-window",
            "stage": f"Stage45C_{EXECUTION_GATE}",
            "execution_head": EXPECTED_HEAD,
            "protocol_sha256": protocol_sha,
        },
        "timestamp_profile": "STAGE40_T16_20HZ_V1",
        "cohort": {
            "name": "FULL24_EXISTING_TRAIN",
            "split": "TRAIN",
            "identity": dict(dataset_artifact),
            "window_order": "explicit_manifest_order",
            "windows": [
                {
                    "id": WINDOW_ID,
                    "target_id": TARGET_ID,
                    "anchor_world_m": list(window["anchor_world_m"]),
                    "target_world_m": list(window["target_world_m"]),
                    "source_timestamps": list(window["source_timestamps"]),
                    "observation_timestamp": float(window["observation_timestamp"]),
                }
            ],
        },
        "checkpoints": [
            {
                "label": CHECKPOINT_LABEL,
                "identity": dict(checkpoint_artifact),
                "runtime_mapping": dict(mapping_artifact),
                "statistics": dict(statistics_artifact),
            }
        ],
        "contract": {
            "raw_action_shape": [1, 40, 132],
            "decoded_action_shape": [40, 10],
            "state_v1_shape": [10],
            "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
            "target_xyz_in_model_input": False,
            "raw_before_decode": True,
            "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
            "inverse_normalization": "manifest_bound_stats",
            "geometry": "K16",
            "timing": "T16",
        },
        "sampler": {
            "profile": SAMPLER_PROFILE,
            "timesteps": list(TIMESTEPS),
            "dt": SAMPLER_DT,
            "expected_calls_per_window": SAMPLER_CALLS,
            "same_noise_across_checkpoints": True,
            "noise_bank": str(noise_path),
            "noise_bank_sha256": noise_artifact["sha256"],
        },
        "runtime": {
            "adapter": "reusable_h200_inference_canary_launcher_v1",
            "offline": True,
            "no_model_construction": True,
            "no_training": True,
            "identity": runtime_identity,
            "backend": "groot_h200_inference_backend_v1",
            "qualification_entrypoint": dict(launcher_artifact),
            "runtime_config": dict(runtime_config_artifact),
            "late_bound": manifest_runtime_late_bound,
        },
        "output": {
            "root": str(output_root),
            "raw_predictions_manifest": "raw_predictions.jsonl",
            "fresh": True,
        },
        "locked_boundaries": {
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
            "new_data": "LOCKED",
        },
        "analysis": {"paired_comparisons": []},
    }

    # Construct the admission from observations, independently of the
    # manifest object.  Artifact values are re-read/recomputed above and are
    # inserted into a separate mapping; equality is then enforced by the
    # canonical cross-binding validator.
    observed_checkpoint = {
        "kind": "tree",
        "path": str(CHECKPOINT),
        "sha256": checkpoint_artifact["sha256"],
    }
    observed_statistics = {
        "kind": "file",
        "path": str(STATS),
        "sha256": statistics_artifact["sha256"],
    }
    observed_dataset = {
        "kind": "tree",
        "path": str(DATASET),
        "sha256": dataset_artifact["sha256"],
    }
    observed_noise = {
        "kind": "file",
        "path": str(noise_path),
        "sha256": noise_artifact["sha256"],
    }
    observed_runtime_late_bound = {
        "repository_execution_head": repository["head"],
        "python_runtime": f"{python_real};version={platform.python_version()}",
        "container_or_environment": environment_identity,
        "groot_source": str(GROOT_SOURCE),
        "cosmos_cache_root": str(COSMOS_ROOT),
        "checkpoint_paths": {CHECKPOINT_LABEL: observed_checkpoint},
        "dataset_path": observed_dataset,
        "statistics_path": {CHECKPOINT_LABEL: observed_statistics},
        "output_root": str(output_root),
        "scheduler_project": project,
        "scheduler_queue": queue,
        "scheduler_resources": "select=1:ngpus=1,walltime=08:00:00",
        "gpu_device": f"{DEVICE};gpu={gpu['name']};uuid={gpu['uuid']}",
        "host": socket.gethostname(),
        "offline_mode": True,
        "runtime_admission_identity": runtime_identity,
    }
    admission = {
        "schema": "GO2ARX5_REUSABLE_H200_INFERENCE_ADMISSION_PREFLIGHT_V1",
        "backend": "groot_h200_inference_backend_v1",
        "runtime_identity": runtime_identity,
        "qualification_entrypoint": dict(launcher_artifact),
        "runtime_config": dict(runtime_config_artifact),
        "timestamp_profile": "STAGE40_T16_20HZ_V1",
        "runtime_late_bound": observed_runtime_late_bound,
        "cohort": {
            "name": "FULL24_EXISTING_TRAIN",
            "split": "TRAIN",
            "window_ids": [WINDOW_ID],
            "target_ids": [TARGET_ID],
        },
        "sampler": {
            "profile": SAMPLER_PROFILE,
            "timesteps": list(TIMESTEPS),
            "dt": SAMPLER_DT,
            "expected_calls_per_window": SAMPLER_CALLS,
            "same_noise_across_checkpoints": True,
            "noise_bank": observed_noise,
        },
        "output_fresh": True,
        "locked_boundaries": {
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
            "new_data": "LOCKED",
        },
        "observed_runtime": {
            "source": "actual_hopper_runtime_state",
            "observed_at_utc": now_utc(),
            "repository": repository,
            "gpu": gpu,
            "host": socket.gethostname(),
            "python": {
                "executable": python_real,
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "environment_identity": environment_identity,
            "pbs": {
                "job_id": job_id,
                "queue": queue,
                "project": project,
                "requested_resources": "select=1:ngpus=1,walltime=08:00:00",
            },
            "offline": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
                "network_fetch": False,
            },
            "materialized": {
                "checkpoint": observed_checkpoint,
                "statistics": observed_statistics,
                "dataset": observed_dataset,
                "noise_bank": observed_noise,
                "groot_source": {
                    "path": str(GROOT_SOURCE),
                    "runtime_manifest": str(GROOT_RUNTIME_MANIFEST),
                    "runtime_manifest_sha256": sha256_file(GROOT_RUNTIME_MANIFEST),
                    "expected_source_commit": legacy.GROOT_SOURCE_COMMIT,
                    "payload_tree_sha256": legacy.GROOT_PAYLOAD_TREE_SHA256,
                },
                "cosmos": {
                    "path": str(COSMOS_ROOT),
                    "revision": legacy.COSMOS_REVISION,
                    "tree_sha256": cosmos_tree_sha,
                    "entry_count": cosmos_entry_count,
                },
            },
            "selected_window": {
                "index": WINDOW_INDEX,
                "window_id": WINDOW_ID,
                "target_id": TARGET_ID,
                "selection_rule": "first explicit target-major TRAIN24 window",
                "cohort_window_count": len(snapshots),
                "canonical_label_source": canonical_label_source,
            },
            "model_imported_before_validation": False,
            "model_construction_before_validation": False,
            "model_forward_before_validation": False,
        },
    }
    require(not model_modules_present(), "model package imported before admission document")
    manifest_path = evidence_root / "evaluation_manifest.json"
    admission_path = evidence_root / "observed_h200_admission.json"
    durable_json(manifest_path, manifest)
    durable_json(admission_path, admission)
    event(
        evidence_root,
        "admission_documents_written",
        manifest_path=str(manifest_path),
        admission_path=str(admission_path),
        model_modules_present=model_modules_present(),
    )
    return manifest, admission, manifest_path, admission_path, {
        "repository": repository,
        "gpu": gpu,
        "job_id": job_id,
        "queue": queue,
        "project": project,
        "runtime_identity": runtime_identity,
        "cosmos_tree_sha256": cosmos_tree_sha,
        "cosmos_entry_count": cosmos_entry_count,
        "snapshots": snapshots,
        "window": window,
        "noise_path": noise_path,
        "checkpoint_artifact": checkpoint_artifact,
        "statistics_artifact": statistics_artifact,
        "dataset_artifact": dataset_artifact,
        "mapping_artifact": mapping_artifact,
        "canonical_label_source": canonical_label_source,
        "legacy": legacy,
    }


def extract_observations(
    *,
    dataset: Path,
    modality_configs: Mapping[str, Any],
    embodiment: Any,
    evidence_root: Path,
) -> dict[str, dict[str, Any]]:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    loader = LeRobotEpisodeLoader(dataset, modality_configs)
    require(len(loader) == 24, f"TRAIN24 loader length changed: {len(loader)}")
    episode = loader[WINDOW_INDEX]
    step_data = extract_step_data(
        episode,
        0,
        modality_configs,
        embodiment,
        allow_padding=False,
    )
    images = step_data.images
    require("external_rgb" in images and "wrist_rgb" in images, "required RGB modalities are missing")
    body, body_source_dtype = first_image(images["external_rgb"], name="external_rgb")
    wrist, wrist_source_dtype = first_image(images["wrist_rgb"], name="wrist_rgb")
    body_path = evidence_root / "observations" / "body_rgb.npy"
    wrist_path = evidence_root / "observations" / "wrist_rgb.npy"
    write_npy_durable(body_path, body)
    write_npy_durable(wrist_path, wrist)
    return {
        "body_rgb": {
            **file_artifact(body_path),
            "shape": [96, 96, 3],
            "dtype": "uint8",
            "source_modality": "external_rgb",
            "source_dtype": body_source_dtype,
        },
        "wrist_rgb": {
            **file_artifact(wrist_path),
            "shape": [96, 96, 3],
            "dtype": "uint8",
            "source_modality": "wrist_rgb",
            "source_dtype": wrist_source_dtype,
        },
    }


def evidence_inventory(root: Path, *, exclude: set[Path]) -> tuple[Path, str]:
    inventory = root / "evidence_inventory_sha256.txt"
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path in exclude:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    with inventory.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    return inventory, sha256_file(inventory)


def run_canary(
    repo: Path,
    evidence_root: Path,
    output_root: Path,
    runtime_config: Path,
) -> dict[str, Any]:
    repo = _validate_repo_root(repo)
    _activate_runtime_config(runtime_config, repo)
    require(str(repo) in sys.path, "repository root is absent from sys.path")
    require(str(repo / "src") in sys.path, "repository src is absent from sys.path")
    require(evidence_root.is_dir() and not evidence_root.is_symlink(), "evidence root is unavailable")
    event(evidence_root, "canary_start", expected_head=EXPECTED_HEAD, output_root=str(output_root))

    manifest, admission, manifest_path, admission_path, observed = build_manifest_and_admission(
        repo=repo,
        evidence_root=evidence_root,
        output_root=output_root,
    )

    from go2arx5_vla.evaluation.h200_admission import validate_h200_inference_admission
    from go2arx5_vla.evaluation.reusable import (
        GENERIC_PREDICTION_RECORD_SCHEMA,
        load_prediction_records,
        run_evaluation,
    )
    from go2arx5_vla.evaluation.structured_action_eval import DecodedStructuredPrediction

    event(
        evidence_root,
        "admission_validation_start",
        model_modules_present=model_modules_present(),
    )
    require(not model_modules_present(), "model package imported before cross-binding")
    admission_result = validate_h200_inference_admission(
        manifest,
        admission,
        manifest_path=manifest_path,
    )
    require(admission_result["status"] == "PASS_STATIC_PREFLIGHT", "H200 admission did not pass")
    require(admission_result["model_imported"] is False, "admission reports a model import")
    require(admission_result["model_construction"] is False, "admission reports model construction")
    require(admission_result["model_forward"] is False, "admission reports model forward")
    require(admission_result["pbs_submission"] is False, "admission reports a nested PBS submission")
    require(not model_modules_present(), "model package imported during cross-binding")
    durable_json(evidence_root / "admission_validation_pass.json", admission_result)
    event(
        evidence_root,
        "admission_validation_passed_before_model_import",
        model_modules_present=model_modules_present(),
        cross_binding_status=admission_result["status"],
    )

    legacy = observed["legacy"]
    runtime_namespace_root = evidence_root / "runtime_namespace_root"
    runtime_namespace_root.mkdir()
    nested_cache = legacy._bind_nested_cosmos_hf_cache(
        runtime_namespace_root,
        COSMOS_ROOT,
        verified_tree_sha256=observed["cosmos_tree_sha256"],
        verified_entry_count=observed["cosmos_entry_count"],
    )
    durable_json(evidence_root / "nested_cosmos_runtime_binding.json", nested_cache)
    event(evidence_root, "runtime_namespace_bound", model_modules_present=model_modules_present())

    stage26_root = STAGE26_FLOW.parent
    if str(stage26_root) not in sys.path:
        sys.path.insert(0, str(stage26_root))
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor

    processor, embodiment, modality_configs = _build_runtime_processor(
        groot_source=GROOT_SOURCE,
        base_checkpoint=BASE_CHECKPOINT,
        dataset_dir=DATASET,
        modality_path=MODALITY,
        cosmos_snapshot=COSMOS_ROOT / "snapshots" / legacy.COSMOS_REVISION,
    )
    windows = legacy._prepare_train24_windows(
        processor,
        embodiment,
        modality_configs,
        DATASET,
        [observed["window"]],
    )
    require(len(windows) == 1, "canary prepared more than one window")
    prepared_window = windows[0]
    require(prepared_window["window_id"] == WINDOW_ID, "prepared window ID drift")
    require(prepared_window["logical_model_input_keys"] == ["body_rgb", "wrist_rgb", "state"], "model input key drift")
    require(prepared_window["target_xyz_in_model_input"] is False, "target XYZ entered model input")
    observations = extract_observations(
        dataset=DATASET,
        modality_configs=modality_configs,
        embodiment=embodiment,
        evidence_root=evidence_root,
    )
    event(evidence_root, "one_train24_window_conditioned", window_id=WINDOW_ID)

    import torch

    require(torch.cuda.is_available(), "CUDA became unavailable after runtime import")
    gpu_name_after_import = str(torch.cuda.get_device_name(torch.device(DEVICE)))
    require("H200" in gpu_name_after_import, f"Torch sees a non-H200 device: {gpu_name_after_import}")
    model = legacy._load_stage45c_structured_model(CHECKPOINT, DEVICE, nested_cache)
    event(evidence_root, "model_constructed", model_modules_present=model_modules_present(), checkpoint=CHECKPOINT_LABEL)
    action_head_calls = {"count": 0}

    def count_action_head(*_args: Any, **_kwargs: Any) -> None:
        action_head_calls["count"] += 1

    hook = model.action_head.model.register_forward_hook(count_action_head)
    try:
        noise = torch.from_numpy(np.load(observed["noise_path"], allow_pickle=False)).contiguous()
        require(tuple(noise.shape) == (1, 40, 132), "sealed noise shape drift")
        require(noise.dtype == torch.float32, "sealed noise dtype drift")
        with torch.inference_mode():
            raw, sampler_calls = legacy._sample_one(
                model,
                prepared_window["model_inputs"],
                noise,
                torch,
            )
    finally:
        hook.remove()
    raw_np = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    require(tuple(raw_np.shape) == (1, 40, 132), f"raw shape drift: {raw_np.shape}")
    require(bool(np.isfinite(raw_np).all()), "raw prediction is non-finite")
    require(sampler_calls == SAMPLER_CALLS, f"sampler calls drift: {sampler_calls}")
    require(action_head_calls["count"] == SAMPLER_CALLS, f"observed action-head calls drift: {action_head_calls['count']}")
    event(
        evidence_root,
        "one_window_forward_complete",
        checkpoint=CHECKPOINT_LABEL,
        window_id=WINDOW_ID,
        sampler_calls=sampler_calls,
        action_head_calls=action_head_calls["count"],
        inference_mode=True,
    )

    raw_path = evidence_root / "raw_handoff" / "FULL24_at_2000__stage22_expert_red_target_train_a_expert-0000.npy"
    write_npy_durable(raw_path, raw_np)
    raw_file_sha = sha256_file(raw_path)
    raw_array_sha = array_sha256(raw_np)
    noise_sha = observed["noise_path"] and sha256_file(observed["noise_path"])
    record = {
        "schema": GENERIC_PREDICTION_RECORD_SCHEMA,
        "checkpoint_label": CHECKPOINT_LABEL,
        "checkpoint_sha256": observed["checkpoint_artifact"]["sha256"],
        "window_id": WINDOW_ID,
        "raw_path": str(raw_path),
        "raw_sha256": raw_file_sha,
        "raw_array_sha256": raw_array_sha,
        "noise_sha256": noise_sha,
        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "target_xyz_in_model_input": False,
        "observations": observations,
        "state_v1": observed["window"]["state_v1"],
        "expert_physical_action": observed["window"]["expert_physical_action"],
    }
    raw_manifest = evidence_root / "raw_handoff" / "raw_predictions.jsonl"
    durable_jsonl(raw_manifest, [record])
    raw_seal = {
        "schema": "PI_GATE_STATE_0102_RAW_HANDOFF_SEAL_V1",
        "status": "PASS_RAW_SEALED_BEFORE_DECODE",
        "raw_persisted_before_decode": True,
        "decoder_started": False,
        "checkpoint_label": CHECKPOINT_LABEL,
        "checkpoint_sha256": observed["checkpoint_artifact"]["sha256"],
        "window_id": WINDOW_ID,
        "target_id": TARGET_ID,
        "raw_path": str(raw_path),
        "raw_sha256": raw_file_sha,
        "raw_array_sha256": raw_array_sha,
        "raw_shape": [1, 40, 132],
        "noise_path": str(observed["noise_path"]),
        "noise_sha256": noise_sha,
        "sampler_profile": SAMPLER_PROFILE,
        "timesteps": list(TIMESTEPS),
        "dt": SAMPLER_DT,
        "sampler_calls_observed": sampler_calls,
        "action_head_calls_observed": action_head_calls["count"],
        "model_imported": True,
        "model_construction": True,
        "model_forward": True,
        "training": False,
        "backward": False,
        "optimizer": False,
        "checkpoint_write": False,
    }
    durable_json(evidence_root / "raw_handoff_seal.json", raw_seal)
    event(
        evidence_root,
        "raw_sealed_before_decode",
        raw_path=str(raw_path),
        raw_sha256=raw_file_sha,
        raw_array_sha256=raw_array_sha,
        raw_shape=[1, 40, 132],
    )

    # Drop the GPU model before the generic CPU-only handoff.  The decoder
    # below is pure NumPy/Stage40 logic over the sealed runtime-V2 stats; the
    # generic evaluator still owns all manifest/artifact/raw/metric output.
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    event(evidence_root, "model_released_before_generic_cpu_decode")

    from go2arx5_vla.evaluation.structured_action_eval import DecodedStructuredPrediction

    def runtime_v2_cpu_decoder(
        raw_value: Any,
        state_value: Any,
        *,
        stats_path: Path,
        source_timestamps: Any,
        observation_timestamp: float,
        chunk_id: str,
    ) -> DecodedStructuredPrediction:
        require(not torch.is_grad_enabled(), "CPU decoder entered with gradients enabled")
        raw_value_np = np.asarray(raw_value)
        state_value_np = np.asarray(state_value)
        physical, details = legacy._decode_v2(
            raw_value_np,
            state_value_np,
            Path(stats_path),
            source_timestamps,
            observation_timestamp,
            chunk_id,
        )
        physical_np = np.asarray(physical)
        require(tuple(physical_np.shape) == (40, 10), "decoded action shape drift")
        require(bool(np.isfinite(physical_np).all()), "decoded action is non-finite")
        details = dict(details)
        details["cpu_only_runtime_v2_decoder"] = True
        details["stats_path"] = str(stats_path)
        return DecodedStructuredPrediction(
            physical_action=physical_np,
            decoder_details=details,
            trajectory_chunk=None,
        )

    records = load_prediction_records(raw_manifest)
    require(len(records) == 1, "raw handoff contains more than one record")
    event(evidence_root, "generic_cpu_decode_start", records=len(records), decoder="Stage40_runtime_V2_CPU_bridge")
    with torch.inference_mode():
        generic_summary = run_evaluation(
            manifest,
            records,
            output_dir=output_root,
            manifest_path=manifest_path,
            records_base_dir=evidence_root,
            decoder=runtime_v2_cpu_decoder,
        )
    require(generic_summary["raw_prediction_count"] == 1, "generic evaluator raw count drift")
    require(generic_summary["sampler_calls"] == 4, "generic evaluator sampler call count drift")
    require(generic_summary["raw_prediction_shape"] == [1, 40, 132], "generic raw ABI drift")
    require(generic_summary["decoded_action_shape"] == [40, 10], "generic decoded ABI drift")
    event(evidence_root, "generic_cpu_decode_complete", output_root=str(output_root))

    output_manifest = output_root / "run_manifest.json"
    output_inventory = output_root / "artifact_inventory_sha256.txt"
    require(output_manifest.is_file(), "generic output run manifest is missing")
    require(output_inventory.is_file(), "generic output inventory is missing")
    evidence_files_exclude = {
        evidence_root / "evidence_inventory_sha256.txt",
        evidence_root / "evidence_bundle.json",
    }
    event(evidence_root, "evidence_bundle_prepared", output_root=str(output_root))
    inventory_path, inventory_sha = evidence_inventory(evidence_root, exclude=evidence_files_exclude)
    bundle = {
        "schema": "REUSABLE_H200_V1_H200_INFERENCE_CANARY_EVIDENCE_BUNDLE_V1",
        "status": "PASS",
        "token": EXPECTED_CANARY_TOKEN,
        "gate": EXECUTION_GATE,
        "execution_head": EXPECTED_HEAD,
        "branch": EXPECTED_BRANCH,
        "qualification_entrypoint": dict(launcher_artifact),
        "runtime_config": dict(runtime_config_artifact),
        "pbs": {
            "submissions": 1,
            "job_id": observed["job_id"],
            "project": observed["project"],
            "queue": observed["queue"],
            "host": socket.gethostname(),
            "device": DEVICE,
            "gpu_name_from_nvidia_smi": observed["gpu"]["name"],
            "gpu_name_from_torch": gpu_name_after_import,
            "terminal_status": "PASS",
        },
        "runtime_admission": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "admission_path": str(admission_path),
            "admission_sha256": sha256_file(admission_path),
            "validation_path": str(evidence_root / "admission_validation_pass.json"),
            "validation_sha256": sha256_file(evidence_root / "admission_validation_pass.json"),
            "status": admission_result["status"],
            "before_model_import": True,
            "observed_runtime_source": "actual_hopper_runtime_state",
            "model_imported_before_validation": False,
            "model_construction_before_validation": False,
            "model_forward_before_validation": False,
        },
        "materialized_inputs": {
            "checkpoint": {
                "label": CHECKPOINT_LABEL,
                "path": str(CHECKPOINT),
                "tree_sha256": observed["checkpoint_artifact"]["sha256"],
            },
            "statistics": dict(observed["statistics_artifact"]),
            "cohort": dict(observed["dataset_artifact"]),
            "window_count_admitted": 1,
            "selected_window_index": WINDOW_INDEX,
            "selected_window_id": WINDOW_ID,
            "selected_target_id": TARGET_ID,
            "selection_rule": "first explicit target-major TRAIN24 window",
            "noise_bank": {
                "path": str(observed["noise_path"]),
                "sha256": noise_sha,
            },
        },
        "sampler": {
            "profile": SAMPLER_PROFILE,
            "timesteps": list(TIMESTEPS),
            "dt": SAMPLER_DT,
            "expected_calls": SAMPLER_CALLS,
            "observed_calls": sampler_calls,
            "observed_action_head_calls": action_head_calls["count"],
            "same_noise": True,
        },
        "raw_seal": {
            "path": str(raw_path),
            "sha256": raw_file_sha,
            "array_sha256": raw_array_sha,
            "shape": [1, 40, 132],
            "persisted_before_decode": True,
            "raw_handoff_manifest": str(raw_manifest),
            "raw_handoff_manifest_sha256": sha256_file(raw_manifest),
        },
        "generic_cpu_evaluation": {
            "output_root": str(output_root),
            "run_manifest": str(output_manifest),
            "run_manifest_sha256": sha256_file(output_manifest),
            "output_inventory": str(output_inventory),
            "output_inventory_sha256": sha256_file(output_inventory),
            "raw_prediction_count": generic_summary["raw_prediction_count"],
            "decoded_action_shape": generic_summary["decoded_action_shape"],
            "decoder_boundary": "Stage40_runtime_V2_CPU_bridge",
            "model_construction": False,
            "model_forward": "external_canary_completed_before_handoff",
            "training": False,
        },
        "execution_counters": {
            "checkpoint_window_forwards": 1,
            "action_head_dit_forward_calls": action_head_calls["count"],
            "raw_prediction_count": 1,
            "decoded_prediction_count": 1,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "resume": False,
            "checkpoint_write": False,
            "Validation-E_read": False,
            "Holdout-F_read": False,
            "WBC_steps": 0,
            "simulator": False,
            "hardware": False,
            "retry": False,
            "nested_pbs_submission": False,
        },
        "locked_boundaries": {
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
            "new_data": "LOCKED",
        },
        "evidence_inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha,
        },
        "no_retry_or_merge": True,
        "scientific_interpretation": "DEFERRED_TO_PI_REVIEW",
    }
    durable_json(evidence_root / "evidence_bundle.json", bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    import argparse

    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(raw_args)
    try:
        repo = _validate_repo_root(args.repo_root)
        require(
            args.self_check or args.execute,
            "select exactly one mode: --self-check or --execute",
        )
        require(
            not (args.self_check and args.execute),
            "--self-check and --execute are mutually exclusive",
        )
        if args.self_check:
            result = _bootstrap_self_check(repo)
        else:
            require(args.runtime_config is not None, "--runtime-config is required for --execute")
            require(args.evidence_root is not None, "--evidence-root is required for --execute")
            require(args.output_root is not None, "--output-root is required for --execute")
            result = run_canary(
                repo,
                args.evidence_root.resolve(),
                args.output_root.resolve(),
                args.runtime_config,
            )
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as exc:
        failure = {
            "schema": "REUSABLE_H200_V1_CANARY_FAILURE_V1",
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "launcher_path": str(QUALIFICATION_ENTRYPOINT),
            "repository_root": str(REPO_ROOT),
            "import_roots": {
                "repository": str(REPO_ROOT) in sys.path,
                "src": str(SRC_ROOT) in sys.path,
            },
            "execution_head_expected": EXPECTED_HEAD or None,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "model_modules_present": model_modules_present(),
            "output_root": str(args.output_root) if args.output_root is not None else None,
            "time_utc": now_utc(),
        }
        evidence_arg = next(
            (
                raw_args[index + 1]
                for index, item in enumerate(raw_args)
                if item == "--evidence-root" and index + 1 < len(raw_args)
            ),
            None,
        )
        if evidence_arg:
            try:
                durable_json(Path(evidence_arg).resolve() / "canary_failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
