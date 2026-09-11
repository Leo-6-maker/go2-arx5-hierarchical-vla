#!/usr/bin/env python3
"""Evaluate the sealed L1 checkpoints on the exact TRAIN24 E4 contract.

The evaluator is deliberately a separate, inference-only process.  It binds
the observed Hopper checkout, the finished L1 training output, the historical
FULL24@2000 comparator, and the accepted FULL24 runtime artifacts before the
first Torch/GR00T import.  It then executes exactly three checkpoint arms x 24
TRAIN windows x four carried Euler stages.  Raw universal actions are durably
sealed before the unchanged Stage40/V2 physical decoder or any metric code is
called.

No loss patch is installed during evaluation: L1 is represented by the
checkpoint, while normalization, decoder, sampler, and TRAIN24 inputs are
identical across all three arms.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45e_l1_runtime_admission as admission  # noqa: E402


EVAL_SCHEMA = "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_V1"
EVAL_SEAL_SCHEMA = "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_SEAL_V1"
PI_GATE = admission.PI_GATE_COMMENT
DEVICE = admission.DEVICE
SNAPSHOT_ROOT = Path("/scratch/e1192548/stage45d_d1_train24_descriptor_closure_20260911_v1")
LEGACY_FULL24_2000 = Path("/scratch/e1192548/stage45b_pi0087_b2_h200_20260909T052311Z/checkpoint-2000")
DEFAULTS = {
    "base_model": "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B",
    "dataset": "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/lerobot_train",
    "groot_source": "/scratch/e1192548/stage45c_pi0094_native_runtime_20260909/groot-source-v4",
    "groot_manifest": "/scratch/e1192548/stage45b_pi0086_groot_snapshot__zhu4zz2/STAGE45B_GROOT_SOURCE_SNAPSHOT_V2.manifest.json",
    "cosmos_root": "/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B",
    "modality": "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py",
    "config": "/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training/experiment_cfg/config.yaml",
    "stats": "/scratch/e1192548/stage45b_pi0083_runtime_parquet_v2/STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json",
    "release": "/scratch/e1192548/stage45b_pi0087_admission_IWmsW5/STAGE45B_B2_TRAINING_RELEASE_V2.json",
    "video_qualification": "/scratch/e1192548/stage45b_pi0087_final_video_qual_DXDgKu/VANDA_STAGE45B_PI0087_FINAL_ADMISSION_QUALIFIED.json",
    "ffmpeg_root": "/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf",
}
LEGACY_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
TRAIN_RESULT_NAME = "stage45e_l1_training_result.json"
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in (
        "red_target_train_a",
        "red_target_train_b",
        "red_target_train_c",
        "red_target_train_d",
    )
    for index in range(6)
)
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
FORBIDDEN_CALLS = {"backward", "get_action", "save_pretrained", "save_model", "save_checkpoint", "train"}


class L1EvaluationError(RuntimeError):
    """Raised when the evaluation cannot prove its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L1EvaluationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
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


def write_json_durable(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_npy_durable(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    _require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def checkout_observation(expected_head: str) -> dict[str, Any]:
    root = _run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = _run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = _run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = _run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    _require(root == str(REPO), f"repository root drift: {root}")
    _require(branch == admission.BRANCH, f"branch drift: {branch}")
    _require(head == expected_head, f"HEAD drift: {head} != {expected_head}")
    _require(not status, "evaluation checkout is dirty, including untracked files")
    ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", admission.BASE_EXECUTION_HEAD, head], check=False)
    _require(ancestor.returncode == 0 and head != admission.BASE_EXECUTION_HEAD, "evaluation HEAD ancestry drift")
    return {
        "observed_repo_root": root,
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
    }


def gpu_observation() -> dict[str, Any]:
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required for H200 evaluation")
    query = "index,name,uuid,memory.total,memory.used"
    raw = _run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        _require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
        rows.append({"index": int(fields[0]), "gpu_name": fields[1], "gpu_uuid": fields[2], "total_vram_mib": int(fields[3]), "used_vram_mib": int(fields[4])})
    _require(len(rows) == 1 and rows[0]["index"] == 0, f"evaluation requires one GPU at index 0: {rows!r}")
    _require("H200" in rows[0]["gpu_name"].upper(), f"evaluation GPU is not H200: {rows[0]['gpu_name']}")
    _require(rows[0]["total_vram_mib"] * 1024**2 >= admission.MIN_VRAM_BYTES, "evaluation H200 is below 80 GiB")
    return {
        "schema": "VANDA_STAGE45E_L1_OBSERVED_H200_EVAL_GPU_V1",
        "host": socket.gethostname(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "selected_device": DEVICE,
        "gpu_role": admission.GPU_ROLE,
        "gpu_name": rows[0]["gpu_name"],
        "gpu_uuid": rows[0]["gpu_uuid"],
        "total_vram_bytes": rows[0]["total_vram_mib"] * 1024**2,
        "memory_used_mib_at_admission": rows[0]["used_vram_mib"],
        "visible_gpu_count": 1,
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_query": query,
        "nvidia_smi_stdout_sha256": sha256_bytes((raw + "\n").encode("utf-8")),
    }


def runtime_identity(gpu: Mapping[str, Any]) -> dict[str, Any]:
    offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    _require(all(value == "1" for value in offline.values()), f"offline mode is incomplete: {offline}")
    return {
        "schema": "VANDA_STAGE45E_L1_OBSERVED_H200_EVAL_RUNTIME_V1",
        "host": gpu["host"],
        "pbs_jobid": gpu["pbs_jobid"],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "selected_device": DEVICE,
        "offline_environment": offline,
    }


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.add(node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "")
    forbidden_imports = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    forbidden_calls = sorted(calls.intersection(FORBIDDEN_CALLS))
    markers = {
        "three_checkpoint_arms": "FULL24@2000" in source and "L1@1000" in source and "L1@2000" in source,
        "e4_times": "EVAL_T_VALUES" in source or "EVAL_T_BUCKETS" in source,
        "raw_before_derived": False,
        "durable_raw_writer": "write_npy_durable" in source,
        "accepted_v2_decoder": "_decode_v2" in source,
        "paired_summary": "paired_summary" in source,
        "no_training": "training" in source,
    }
    # The raw marker is checked by structure rather than by executing the
    # model: the false decoded record is appended before the decode call.
    markers["raw_before_derived"] = '"decoded": False' in source and "_decode_v2" in source and source.index('"decoded": False') < source.index("_decode_v2(")
    status = "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL"
    return {
        "schema": "VANDA_STAGE45E_L1_TRAIN24_E4_EVAL_STATIC_GUARD_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": forbidden_calls,
        "markers": markers,
        "foreign_cwd_supported": True,
        "empty_pythonpath_supported": True,
        "model_imported": False,
        "pbs": 0,
        "h200": 0,
        "training": 0,
    }


def _namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        base_model=Path(args.base_model),
        dataset=Path(args.dataset),
        groot_source=Path(args.groot_source),
        groot_manifest=Path(args.groot_manifest),
        cosmos_root=Path(args.cosmos_root),
        modality=Path(args.modality),
        config=Path(args.config),
        stats=Path(args.stats),
        release=Path(args.release),
        video_qualification=Path(args.video_qualification),
        ffmpeg_root=Path(args.ffmpeg_root),
    )


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    import stat

    root = Path(root).resolve()
    mode = root.lstat().st_mode
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"checkpoint root is unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(dirnames):
            child = directory_path / name
            child_mode = child.lstat().st_mode
            _require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"checkpoint directory is unsafe: {child}")
        for name in sorted(filenames):
            child = directory_path / name
            child_mode = child.lstat().st_mode
            _require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"checkpoint file is unsafe: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _verify_checkpoint(path: Path, prefix: str, expected_tree: str | None = None) -> dict[str, Any]:
    entries = _tree_entries(path, prefix)
    digest = tree_digest(entries)
    if expected_tree is not None:
        _require(digest == expected_tree, f"checkpoint tree drift: {path}: {digest} != {expected_tree}")
    state_path = path / "trainer_state.json"
    _require(state_path.is_file() and not state_path.is_symlink(), f"trainer state missing: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    _require(all((path / name).is_file() and not (path / name).is_symlink() for name in required), f"checkpoint state closure missing: {path}")
    return {
        "path": str(path.resolve()),
        "tree_sha256": digest,
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "global_step": state.get("global_step"),
        "required_state_files": list(required),
    }


def _build_binding(checkout: Mapping[str, Any], gpu: Mapping[str, Any], runtime: Mapping[str, Any], paths: Mapping[str, str], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    zero = {"model_construction": 0, "model_forward": 0, "backward": 0, "optimizer_steps": 0, "training_steps": 0, "gpu": 1, "pbs": 1}
    boundary = {
        "model_construction": "RUNTIME_AFTER_ADMISSION",
        "model_forward": "RUNTIME_AFTER_ADMISSION",
        "gpu": 1,
        "h200": 1,
        "pbs": 1,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "N1": 0,
        "N2": 0,
        "N2_plus_L1": 0,
        "D2B": 0,
        "FULL24@4000": 0,
        "automatic_retry": False,
    }
    common = {
        "schema": admission.RUNTIME_BINDING_SCHEMA,
        "stage": admission.STAGE,
        "pi_gate": PI_GATE,
        "pi_authority_token": admission.AUTHORITY_TOKEN,
        "execution_authorized": True,
        "phase": "EVAL",
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
        "execution_checkout": {
            "binding_mode": "OBSERVED_RUNTIME_ONLY",
            "authorized_repo_root": checkout["observed_repo_root"],
            "authorized_branch": checkout["observed_branch"],
            "authorized_head": checkout["observed_head"],
            "authorized_clean_worktree": True,
        },
        "runtime_identity": dict(runtime),
        "host_gpu_runtime": dict(gpu),
        "source_identities": admission.SOURCE_IDENTITIES,
        "dataset_identity": admission.DATASET_IDENTITY,
        "l1_contract": admission.L1_CONTRACT,
        "training_recipe": admission.TRAINING_RECIPE,
        "evaluation_sampler": admission.EVAL_SAMPLER_CONTRACT,
        "evaluation_call_budget": admission.EVAL_CALL_BUDGET,
        "locked_boundaries": admission.LOCKED_BOUNDARIES,
        "zero_counters": zero,
        "execution_boundary": boundary,
        "output": {"root": str(output_root), "fresh_required": True},
        "paths": dict(paths),
        "device_policy": {
            "selected_device": DEVICE,
            "selected_gpu_role": admission.GPU_ROLE,
            "minimum_vram_bytes": admission.MIN_VRAM_BYTES,
            "allow_multi_gpu": False,
            "allow_sharding": False,
            "allow_offload": False,
            "allow_precision_change": False,
            "oom_retry": False,
        },
    }
    observed = {
        "execution_checkout": dict(checkout),
        "runtime_identity": dict(runtime),
        "host_gpu_runtime": dict(gpu),
        "source_identities": admission.SOURCE_IDENTITIES,
        "dataset_identity": admission.DATASET_IDENTITY,
        "l1_contract": admission.L1_CONTRACT,
        "training_recipe": admission.TRAINING_RECIPE,
        "evaluation_sampler": admission.EVAL_SAMPLER_CONTRACT,
        "evaluation_call_budget": admission.EVAL_CALL_BUDGET,
        "locked_boundaries": admission.LOCKED_BOUNDARIES,
        "zero_counters": zero,
        "execution_boundary": boundary,
        "output": {"root": str(output_root), "fresh": True},
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "offline_mode": True,
    }
    return common, observed


def _audit_artifacts(args: argparse.Namespace, training_root: Path) -> dict[str, Any]:
    import stage45e_l1_h200_train as train
    import stage45c_h200_eval_runtime_adapter as stage45c

    base_args = _namespace(args)
    paths = train._paths(base_args)
    # The training-side audit is reused only for accepted source artifacts;
    # it does not import Torch/GR00T and does not inspect a model output.
    input_audit = train._validate_inputs(paths)
    closure = stage45c._verify_train24_descriptor_closure(Path(args.snapshot_root), paths["full24_dataset"])
    _require(len(closure["snapshots"]) == 24, "TRAIN24 descriptor closure count drift")

    training_result_path = training_root / TRAIN_RESULT_NAME
    _require(training_result_path.is_file() and not training_result_path.is_symlink(), "L1 training result is missing")
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    _require(training_result.get("status") == admission.FULL_PASS_TOKEN, "L1 training result is not PASS")
    _require(training_result.get("token") == admission.FULL_PASS_TOKEN, "L1 training token drift")
    _require(training_result.get("execution_head") == args.expected_head, "training result HEAD drift")
    _require(training_result.get("branch") == admission.BRANCH, "training result branch drift")
    _require(training_result.get("normalization_unchanged") is True, "training changed normalization")
    _require(training_result.get("mask_unchanged") is True, "training changed mask")
    _require(training_result.get("decoder_unchanged") is True, "training changed decoder")
    _require(training_result.get("training_steps") == 2000 and training_result.get("optimizer_steps") == 2000, "training step closure drift")
    _require(training_result.get("validation_e_read") is False and training_result.get("holdout_f_read") is False, "locked data was read")
    intervention = training_result.get("intervention")
    _require(intervention == admission.L1_CONTRACT, "training L1 intervention identity drift")
    _require(training_result.get("structured_stats", {}).get("statistics_payload_sha256") == admission.FULL24_STATS_PAYLOAD_SHA256, "training statistics changed")

    legacy = _verify_checkpoint(Path(args.legacy_checkpoint), "checkpoint-2000", LEGACY_TREE_SHA256)
    l1_1000 = _verify_checkpoint(training_root / "checkpoint-1000", "stage45e_l1/checkpoint-1000", training_result["checkpoint_1000"]["tree_sha256"])
    l1_2000 = _verify_checkpoint(training_root / "checkpoint-2000", "stage45e_l1/checkpoint-2000", training_result["checkpoint_2000"]["tree_sha256"])
    _require(l1_1000["global_step"] == 1000 and l1_2000["global_step"] == 2000, "L1 checkpoint global-step drift")

    resolved_paths = {key: str(value.resolve()) for key, value in paths.items()}
    resolved_paths.update({
        "snapshot_root": str(Path(args.snapshot_root).resolve()),
        "legacy_full24_2000_checkpoint": str(Path(args.legacy_checkpoint).resolve()),
        "l1_training_root": str(training_root.resolve()),
        "l1_checkpoint_1000": str((training_root / "checkpoint-1000").resolve()),
        "l1_checkpoint_2000": str((training_root / "checkpoint-2000").resolve()),
    })
    return {
        "input_audit": input_audit,
        "descriptor_closure": closure,
        "training_result": training_result,
        "checkpoints": {"FULL24@2000": legacy, "L1@1000": l1_1000, "L1@2000": l1_2000},
        "paths": resolved_paths,
        "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE_FOR_ALL_THREE_ARMS",
        "normalization_unchanged": True,
        "target_xyz_in_model_input": False,
    }


def _runtime_environment(output_root: Path, cosmos_root: Path) -> dict[str, str]:
    cache = output_root / "runtime_cache"
    tmp = output_root / "runtime_tmp"
    hub = cache / "huggingface" / "hub"
    for path in (cache, tmp, hub):
        path.mkdir(parents=True, exist_ok=False)
    (hub / "models--nvidia--Cosmos-Reason2-2B").symlink_to(cosmos_root, target_is_directory=True)
    values = {
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(hub),
        "HUGGINGFACE_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub),
        "TORCH_HOME": str(cache / "torch"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GROOT_COMMIT_HASH": admission.GROOT_COMMIT,
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
    }
    os.environ.update(values)
    return values


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def _mean(values: Sequence[float]) -> float:
    _require(values, "cannot summarize empty paired values")
    return float(statistics.fmean(values))


def _paired_summary(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    order = list(admission.CHECKPOINT_ORDER)
    row_maps = {label: {str(row["window_id"]): row for row in rows_by_arm[label]} for label in order}
    _require(all(len(row_maps[label]) == admission.EVAL_WINDOW_COUNT for label in order), "paired row cardinality drift")
    base = order[0]
    pairs = {
        "L1@1000_minus_FULL24@2000": ("L1@1000", base),
        "L1@2000_minus_FULL24@2000": ("L1@2000", base),
    }

    def target_of(window_id: str) -> str:
        return window_id.split(":", 1)[0].removeprefix("stage22_expert_")

    def describe(left: str, right: str, window_ids: Sequence[str]) -> dict[str, Any]:
        flattened = {label: {wid: _flatten_numeric(row_maps[label][wid].get("metrics", {})) for wid in window_ids} for label in (left, right)}
        keys = sorted(set().union(*(set(flattened[label][wid]) for label in (left, right) for wid in window_ids)))
        result: dict[str, Any] = {"left": left, "right": right, "window_count": len(window_ids), "metric_delta_left_minus_right": {}}
        for key in keys:
            values = [flattened[left][wid][key] - flattened[right][wid][key] for wid in window_ids if key in flattened[left][wid] and key in flattened[right][wid]]
            if values:
                result["metric_delta_left_minus_right"][key] = {
                    "count": len(values),
                    "mean": _mean(values),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
        return result

    output: dict[str, Any] = {"baseline": base, "pairs": {}}
    all_windows = list(WINDOW_IDS)
    for pair_name, (left, right) in pairs.items():
        by_target = {
            target: describe(left, right, [wid for wid in all_windows if target_of(wid) == target])
            for target in sorted({target_of(wid) for wid in all_windows})
        }
        output["pairs"][pair_name] = {
            "overall": describe(left, right, all_windows),
            "by_target": by_target,
            "by_window": {wid: describe(left, right, [wid]) for wid in all_windows},
        }
    return output


def _inventory(root: Path) -> tuple[Path, str]:
    inventory = root / "inventory" / "SHA256SUMS"
    inventory.parent.mkdir(parents=True, exist_ok=False)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != inventory)
    payload = ("\n".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files) + "\n").encode("utf-8")
    temporary = inventory.with_name(f".{inventory.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, inventory)
    return inventory, sha256_bytes(payload)


def execute(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_construction": 0,
        "conditioning_passes": 0,
        "action_head_calls": 0,
        "model_forward": 0,
        "raw_prediction_count": 0,
        "derived_row_count": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "retry_count": 0,
    }
    phase = "START"
    try:
        _require(not output_root.exists() and not output_root.is_symlink(), f"evaluation output root is not fresh: {output_root}")
        checkout = checkout_observation(args.expected_head)
        gpu = gpu_observation()
        runtime = runtime_identity(gpu)
        guard = static_guard()
        _require(guard["status"] == "PASS", "evaluation static guard failed")
        phase = "PRE_MODEL_ARTIFACT_AUDIT"
        artifact_audit = _audit_artifacts(args, training_root)
        paths = artifact_audit["paths"]
        binding, observed = _build_binding(checkout, gpu, runtime, paths, output_root)
        admission_result = admission.validate_pre_model_binding(
            binding, observed, authority_token=admission.AUTHORITY_TOKEN, phase="EVAL"
        )
        output_root.mkdir(parents=False, exist_ok=False)
        write_json_durable(output_root / "stage45e_l1_eval_static_guard.json", guard)
        write_json_durable(output_root / "stage45e_l1_eval_runtime_binding.json", {"binding": binding, "observed": observed})
        write_json_durable(output_root / "stage45e_l1_eval_pre_model_admission.json", admission_result)
        write_json_durable(output_root / "stage45e_l1_eval_artifact_audit.json", artifact_audit)
        env_values = _runtime_environment(output_root, Path(args.cosmos_root).resolve())
        if str(Path(args.groot_source).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(args.groot_source).resolve()))
        _require(not admission.model_imports_present(), "model module was imported before evaluation admission")

        phase = "MODEL_IMPORT_AND_EVALUATION"
        counters["model_import_started"] = True
        # These imports intentionally occur only after the binding, source
        # hashes, descriptor closure, checkpoint trees, and output freshness
        # have all passed.
        import numpy as np
        import stage45c_h200_eval_runtime_adapter as stage45c
        from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows
        from stage38_structured_offline_evaluation import _load_structured_model

        runtime_processor, embodiment, modality_configs = _build_runtime_processor(
            groot_source=Path(args.groot_source).resolve(),
            base_checkpoint=Path(args.base_model).resolve(),
            dataset_dir=Path(args.dataset).resolve(),
            modality_path=Path(args.modality).resolve(),
            cosmos_snapshot=Path(args.cosmos_root).resolve() / "snapshots" / admission.COSMOS_REVISION,
        )
        snapshot_records = artifact_audit["descriptor_closure"]["snapshots"]
        windows = stage45c._prepare_train24_windows(
            runtime_processor, embodiment, modality_configs, Path(args.dataset).resolve(), snapshot_records
        )
        _require(len(windows) == admission.EVAL_WINDOW_COUNT, "TRAIN24 window materialization drift")
        _require([window["window_id"] for window in windows] == list(WINDOW_IDS), "TRAIN24 window order drift")
        torch, _ = stage45c._load_torch_runtime()
        _require(torch.cuda.is_available(), "CUDA is unavailable on the admitted H200")
        device = torch.device(DEVICE)
        _require(device.index == 0 and "H200" in torch.cuda.get_device_name(device).upper(), "selected CUDA device is not the admitted H200")
        noise_bank = stage45c._make_noise_bank(torch)
        noise_hashes = [sha256_bytes(noise.detach().cpu().contiguous().numpy().tobytes()) for noise in noise_bank]
        write_json_durable(output_root / "sampler_noise_manifest.json", {
            "schema": "VANDA_STAGE45E_L1_FLOAT32_CPU_NOISE_BANK_V1",
            "seed": admission.EVAL_SEED,
            "dtype": "float32",
            "device": "cpu",
            "shape": [1, 40, 132],
            "window_ids": list(WINDOW_IDS),
            "noise_sha256_by_window": noise_hashes,
            "same_noise_across_checkpoint_arms": True,
        })

        checkpoint_paths = {
            "FULL24@2000": Path(args.legacy_checkpoint).resolve(),
            "L1@1000": training_root / "checkpoint-1000",
            "L1@2000": training_root / "checkpoint-2000",
        }
        checkpoint_identities = {
            label: {"label": label, "tree_sha256": artifact_audit["checkpoints"][label]["tree_sha256"]}
            for label in admission.CHECKPOINT_ORDER
        }
        raw_manifest = output_root / "raw_predictions.jsonl"
        rows_by_arm: dict[str, list[dict[str, Any]]] = {label: [] for label in admission.CHECKPOINT_ORDER}
        checkpoint_summaries: list[dict[str, Any]] = []
        runtime_binding_sha = canonical_sha({"binding": binding, "artifact_audit": {"checkpoints": artifact_audit["checkpoints"], "descriptor_manifest": artifact_audit["descriptor_closure"].get("manifest_sha256")}})
        for label in admission.CHECKPOINT_ORDER:
            model_start = time.perf_counter()
            model = _load_structured_model(checkpoint_paths[label], DEVICE)
            model.eval()
            counters["model_construction"] += 1
            hook_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_counter.__setitem__("count", hook_counter["count"] + 1))
            arm_rows: list[dict[str, Any]] = []
            try:
                for index, window in enumerate(windows):
                    raw, sampler_calls = stage45c._sample_one(model, window["model_inputs"], noise_bank[index], torch)
                    counters["conditioning_passes"] += 1
                    counters["raw_prediction_count"] += 1
                    counters["action_head_calls"] += sampler_calls
                    counters["model_forward"] += sampler_calls
                    raw_np = raw.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
                    _require(tuple(raw_np.shape) == (1, 40, 132) and bool(np.isfinite(raw_np).all()), f"raw prediction shape/finite drift: {label}:{index}")
                    raw_path = output_root / "raw" / label.replace("@", "_at_") / f"window-{index:02d}.npy"
                    write_npy_durable(raw_path, raw_np)
                    raw_file_sha = sha256_file(raw_path)
                    raw_array_sha = sha256_bytes(raw_np.tobytes())
                    append_jsonl_durable(raw_manifest, {
                        "schema": "VANDA_STAGE45E_L1_RAW_PREDICTION_ARTIFACT_V1",
                        "checkpoint": checkpoint_identities[label],
                        "window_id": window["window_id"],
                        "path": str(raw_path.relative_to(output_root)),
                        "file_sha256": raw_file_sha,
                        "array_sha256": raw_array_sha,
                        "shape": list(raw_np.shape),
                        "dtype": str(raw_np.dtype),
                        "noise_sha256": noise_hashes[index],
                        "sampler_calls_observed": sampler_calls,
                        "decoded": False,
                    })
                    snapshot = window["snapshot"]
                    physical, decoder_details = stage45c._decode_v2(
                        raw_np,
                        snapshot["state_v1"],
                        Path(args.stats).resolve(),
                        snapshot["source_timestamps"],
                        float(snapshot["observation_timestamp"]),
                        window["window_id"],
                    )
                    row = stage45c._evaluation_row(
                        raw_np,
                        physical,
                        decoder_details,
                        window,
                        checkpoint_identities[label],
                        runtime_binding_sha,
                        str(raw_path.relative_to(output_root)),
                        raw_file_sha,
                    )
                    row["model_runtime_identity"] = {"sha256": runtime_binding_sha, "label": "VANDA_STAGE45E_L1_H200_RUNTIME"}
                    row["noise_sha256"] = noise_hashes[index]
                    row["sampler"] = {
                        "name": admission.EVAL_SAMPLER_CONTRACT["name"],
                        "seed": admission.EVAL_SEED,
                        "t_values": list(admission.EVAL_T_VALUES),
                        "t_buckets": list(admission.EVAL_T_BUCKETS),
                        "dt": admission.EVAL_DT,
                        "steps_observed": sampler_calls,
                        "carried_state": True,
                        "reset_between_stages": False,
                        "official_public_sampler_called": False,
                    }
                    row["stats_route"] = "STAGE45C_RUNTIME_V2_STATS_BRIDGE"
                    row["target_xyz_in_model_input"] = False
                    arm_rows.append(row)
                    counters["derived_row_count"] += 1
                    append_jsonl_durable(raw_manifest, {
                        "schema": "VANDA_STAGE45E_L1_DERIVED_EVIDENCE_REFERENCE_V1",
                        "checkpoint": checkpoint_identities[label],
                        "window_id": window["window_id"],
                        "raw_path": str(raw_path.relative_to(output_root)),
                        "decoded": True,
                        "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                        "metric_semantics_source": "ACCEPTED_STAGE44R",
                    })
            finally:
                hook.remove()
            _require(hook_counter["count"] == admission.EVAL_WINDOW_COUNT * len(admission.EVAL_T_VALUES), f"action-head count drift: {label}: {hook_counter['count']}")
            _require(len(arm_rows) == admission.EVAL_WINDOW_COUNT, f"evaluation row count drift: {label}")
            rows_by_arm[label] = arm_rows
            aggregate = aggregate_evaluation_rows(arm_rows)
            aggregate["runtime_observed"] = {
                "checkpoint": label,
                "checkpoint_tree_sha256": checkpoint_identities[label]["tree_sha256"],
                "window_count": len(arm_rows),
                "conditioning_passes": len(arm_rows),
                "action_head_calls": hook_counter["count"],
                "model_side_calls": len(arm_rows) + hook_counter["count"],
                "raw_prediction_count": len(arm_rows),
                "derived_row_count": len(arm_rows),
                "model_load_latency_s": time.perf_counter() - model_start,
                "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            }
            write_json_durable(output_root / f"aggregate_{label.replace('@', '_at_')}.json", aggregate)
            checkpoint_summaries.append({"checkpoint": label, "aggregate": aggregate, "rows": len(arm_rows), "action_head_calls": hook_counter["count"]})
            del model
            torch.cuda.empty_cache()

        _require(counters["model_construction"] == 3, "model construction count drift")
        _require(counters["conditioning_passes"] == admission.EVAL_CONDITIONING_PASSES, "conditioning count drift")
        _require(counters["action_head_calls"] == admission.EVAL_ACTION_HEAD_CALLS, "action-head call budget drift")
        _require(counters["model_forward"] == admission.EVAL_ACTION_HEAD_CALLS, "model-forward call budget drift")
        _require(counters["raw_prediction_count"] == admission.EVAL_ACTION_HEAD_CALLS // len(admission.EVAL_T_VALUES), "raw artifact count drift")
        _require(counters["derived_row_count"] == counters["raw_prediction_count"], "raw/derived cardinality drift")
        paired = _paired_summary(rows_by_arm)
        cpu_summary = {
            "schema": "VANDA_STAGE45E_L1_TRAIN24_E4_CPU_SUMMARY_V1",
            "status": "PASS_READONLY_DERIVED_FROM_SEALED_RAW",
            "checkpoint_order": list(admission.CHECKPOINT_ORDER),
            "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            "normalization_unchanged": True,
            "window_count_per_checkpoint": admission.EVAL_WINDOW_COUNT,
            "seed": admission.EVAL_SEED,
            "t_values": list(admission.EVAL_T_VALUES),
            "dt": admission.EVAL_DT,
            "aggregate_by_checkpoint": {item["checkpoint"]: item["aggregate"] for item in checkpoint_summaries},
            "paired_summary": paired,
            "independence_boundary": "descriptive paired comparison; seed/t probes share each TRAIN window",
        }
        write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        telemetry = {
            "schema": "VANDA_STAGE45E_L1_H200_EVAL_TELEMETRY_V1",
            "host_gpu_runtime": gpu,
            "runtime_identity": runtime,
            "selected_device": DEVICE,
            "counters": counters,
            "cuda_runtime": str(torch.version.cuda),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "raw_before_derived": True,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
        }
        write_json_durable(output_root / "telemetry.json", telemetry)
        seal = {
            "schema": EVAL_SEAL_SCHEMA,
            "status": "PASS",
            "token": "STAGE45E_L1_TRAIN24_E4_EVALUATION_COMPLETE",
            "stage": admission.STAGE,
            "pi_gate": PI_GATE,
            "execution_head": checkout["observed_head"],
            "branch": checkout["observed_branch"],
            "pbs_jobid": os.environ["PBS_JOBID"],
            "gpu": gpu,
            "runtime_binding_sha256": canonical_sha(binding),
            "runtime_identity_sha256": runtime_binding_sha,
            "training_root": str(training_root),
            "output_root": str(output_root),
            "artifact_identities": artifact_audit["checkpoints"],
            "call_budget": admission.EVAL_CALL_BUDGET,
            "observed_counters": counters,
            "raw_manifest": str(raw_manifest),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "normalization_unchanged": True,
            "mask_unchanged": True,
            "decoder_unchanged": True,
            "validation_e_read": False,
            "holdout_f_read": False,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "automatic_retry": False,
        }
        write_json_durable(output_root / "stage45e_l1_eval_seal.json", seal)
        inventory_path, inventory_sha = _inventory(output_root)
        seal["inventory_path"] = str(inventory_path)
        seal["inventory_sha256"] = inventory_sha
        write_json_durable(output_root / "stage45e_l1_eval_seal.json", seal)
        completion = {
            "schema": EVAL_SCHEMA,
            "status": admission.FULL_PASS_TOKEN,
            "token": admission.FULL_PASS_TOKEN,
            "execution_head": checkout["observed_head"],
            "training_result": str(training_root / TRAIN_RESULT_NAME),
            "evaluation_seal": str(output_root / "stage45e_l1_eval_seal.json"),
            "evaluation_seal_sha256": sha256_file(output_root / "stage45e_l1_eval_seal.json"),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "call_budget": admission.EVAL_CALL_BUDGET,
            "observed_counters": counters,
            "normalization_unchanged": True,
            "validation_e_read": False,
            "holdout_f_read": False,
            "training": False,
            "automatic_retry": False,
        }
        write_json_durable(output_root / "stage45e_l1_eval_result.json", completion)
        print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            write_json_durable(output_root / "stage45e_l1_eval_failure.json", {
                "schema": "VANDA_STAGE45E_L1_EVAL_FAILURE_SEAL_V1",
                "status": "FAIL",
                "pi_gate": PI_GATE,
                "phase": phase,
                "execution_head": args.expected_head,
                "branch": admission.BRANCH,
                "pbs_jobid": os.environ.get("PBS_JOBID"),
                "model_import_started": counters["model_import_started"],
                "model_construction": counters["model_construction"],
                "conditioning_passes": counters["conditioning_passes"],
                "action_head_calls": counters["action_head_calls"],
                "raw_prediction_count": counters["raw_prediction_count"],
                "derived_row_count": counters["derived_row_count"],
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "automatic_retry": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        print(f"STAGE45E_L1_EVAL_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def plan() -> dict[str, Any]:
    guard = static_guard()
    _require(guard["status"] == "PASS", "evaluation static guard failed")
    return {
        "schema": "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "stage": admission.STAGE,
        "pi_gate": PI_GATE,
        "branch": admission.BRANCH,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
        "checkpoint_order": list(admission.CHECKPOINT_ORDER),
        "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "evaluation_sampler": admission.EVAL_SAMPLER_CONTRACT,
        "call_budget": admission.EVAL_CALL_BUDGET,
        "raw_before_derived": True,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "pbs": 0,
        "h200": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "return_token": admission.FULL_PASS_TOKEN,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path, default=LEGACY_FULL24_2000)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--expected-head", default=os.environ.get("STAGE45E_EXPECTED_HEAD"))
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, default=Path(default))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.training_root is None or args.output_root is None or not args.expected_head:
        parser.error("--execute requires --training-root, --output-root, and --expected-head")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
