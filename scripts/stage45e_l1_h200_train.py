#!/usr/bin/env python3
"""Execute the single authorized Stage45E-L1 H200 training run.

The runner has a CPU/static ``--plan-only`` mode and a guarded ``--execute``
mode for one PBS allocation.  All node-local identity checks and the
pre-model admission happen before importing Torch/GR00T.  The only
model-side intervention is the project-owned L1 action-head loss patch; the
FULL24 statistics, StructuredAction mask, decoder, source dataset, and
official training graph remain unchanged.

This program never resumes, retries, changes precision/batch/model modules,
opens Validation-E/Holdout-F, or launches WBC/simulator/hardware.  A failure
is sealed in the attempt's scratch output root with the phase and observed
counters that were actually reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45e_l1_runtime_admission as admission  # noqa: E402
from go2arx5_vla import stage45e_l1_loss as l1  # noqa: E402


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
    "python": "/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python",
}


class L1TrainingError(RuntimeError):
    """Raised when the L1 training attempt cannot continue fail-closed."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L1TrainingError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    _require(path.is_file() and not path.is_symlink(), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise L1TrainingError(f"invalid JSON: {path}") from exc


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    _require(completed.returncode == 0, f"git command failed: {' '.join(args)}")
    return completed.stdout.strip()


def checkout_observation(expected_head: str | None = None) -> dict[str, Any]:
    root = git("rev-parse", "--show-toplevel")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    _require(root == str(REPO), f"repository root drift: {root} != {REPO}")
    _require(branch == admission.BRANCH, f"branch drift: {branch}")
    if expected_head is not None:
        _require(head == expected_head, f"HEAD drift: {head} != {expected_head}")
    _require(not status, "execution checkout is dirty, including untracked files")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", admission.BASE_EXECUTION_HEAD, head],
        check=False,
    )
    _require(ancestor.returncode == 0, "L1 HEAD is not based on accepted D2A HEAD")
    _require(head != admission.BASE_EXECUTION_HEAD, "L1 execution branch has no committed change")
    return {
        "observed_repo_root": root,
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
    }


def _regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise L1TrainingError(f"tree root unavailable: {root}") from exc
    _require(not stat.S_ISLNK(mode) and stat.S_ISDIR(mode), f"tree root is unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            _require(not stat.S_ISLNK(child_mode) and stat.S_ISDIR(child_mode), f"unsafe tree directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            _require(not stat.S_ISLNK(child_mode) and stat.S_ISREG(child_mode), f"unsafe tree file: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _cosmos_tree_digest(root: Path) -> tuple[str, int]:
    root = Path(root).resolve()
    _require(root.is_dir() and not root.is_symlink(), f"Cosmos root is unsafe: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                resolved = item.resolve(strict=True)
                _require(resolved.is_file() and resolved.is_relative_to(root), f"Cosmos symlink escapes root: {item}")
                entries.append({
                    "path": rel, "kind": "symlink", "target": os.readlink(item),
                    "resolved_path": resolved.relative_to(root).as_posix(),
                    "size_bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved),
                })
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "kind": "regular_file", "size_bytes": int(item.stat().st_size), "sha256": sha256_file(item)})
            else:
                raise L1TrainingError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        digest.update("\t".join((
            str(entry["kind"]), str(entry["path"]), str(entry.get("size_bytes", 0)),
            str(entry.get("sha256") or "-"), str(entry.get("target") or "-"),
            str(entry.get("resolved_path") or "-"),
        )).encode("utf-8") + b"\n")
    return digest.hexdigest(), len(entries)


def _regular_file(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")


def _regular_dir(path: Path, label: str) -> None:
    _require(path.is_dir() and not path.is_symlink(), f"{label} is not a regular directory: {path}")


def gpu_observation() -> dict[str, Any]:
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required for L1 execution")
    query = "index,name,uuid,memory.total,memory.used"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    _require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr[-500:]}")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        _require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
        rows.append({
            "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
        })
    _require(len(rows) == 1 and rows[0]["index"] == 0, f"L1 requires one GPU at index 0: {rows!r}")
    _require("H200" in rows[0]["name"].upper(), f"L1 requires H200: {rows[0]['name']}")
    _require(rows[0]["memory_total_mib"] * 1024**2 >= admission.MIN_VRAM_BYTES, "H200 is below 80 GiB floor")
    return {
        "schema": "VANDA_STAGE45E_L1_OBSERVED_H200_V1",
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "pbs_jobname": os.environ.get("PBS_JOBNAME"),
        "selected_device": admission.DEVICE,
        "gpu_role": admission.GPU_ROLE,
        "gpu_name": rows[0]["name"],
        "gpu_uuid": rows[0]["uuid"],
        "total_vram_bytes": rows[0]["memory_total_mib"] * 1024**2,
        "memory_used_mib_at_admission": rows[0]["memory_used_mib"],
        "visible_gpu_count": 1,
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_query": query,
        "nvidia_smi_stdout_sha256": hashlib.sha256((completed.stdout + "\n").encode("utf-8")).hexdigest(),
    }


def runtime_identity(gpu: Mapping[str, Any]) -> dict[str, Any]:
    offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    _require(all(value == "1" for value in offline.values()), f"offline environment is incomplete: {offline}")
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is absent")
    return {
        "schema": "VANDA_STAGE45E_L1_OBSERVED_RUNTIME_V1",
        "host": gpu["host"],
        "pbs_jobid": os.environ["PBS_JOBID"],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "selected_device": admission.DEVICE,
        "offline_environment": offline,
    }


def _bind_cosmos_processor(snapshot: Path) -> dict[str, Any]:
    _regular_dir(snapshot, "Cosmos snapshot")
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    marker = "_stage45e_l1_original_build_processor"
    original = getattr(processing, marker, None)
    if original is None:
        original = processing.build_processor
        setattr(processing, marker, original)

        def build_local_processor(model_name: str, loading_kwargs: dict[str, Any]) -> Any:
            if model_name == "nvidia/Cosmos-Reason2-2B":
                model_name = str(snapshot)
            return original(model_name, loading_kwargs)

        processing.build_processor = build_local_processor
    return {
        "repo_id": "nvidia/Cosmos-Reason2-2B",
        "snapshot": str(snapshot),
        "revision": admission.COSMOS_REVISION,
    }


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    result = {
        "base_checkpoint": Path(args.base_model),
        "full24_dataset": Path(args.dataset),
        "groot_source": Path(args.groot_source),
        "groot_manifest": Path(args.groot_manifest),
        "cosmos_root": Path(args.cosmos_root),
        "modality_config": Path(args.modality),
        "training_config": Path(args.config),
        "full24_runtime_statistics": Path(args.stats),
        "golden_release_manifest": Path(args.release),
        "video_qualification": Path(args.video_qualification),
        "ffmpeg_runtime_root": Path(args.ffmpeg_root),
        "base_processor_config": Path(args.base_model) / "processor_config.json",
        "base_processor_statistics": Path(args.base_model) / "statistics.json",
        "decoder_source": REPO / "src/go2arx5_vla/stage40_structured_decode.py",
        "metrics_source": REPO / "src/go2arx5_vla/evaluation/trajectory_metrics.py",
        "l1_loss_source": REPO / "src/go2arx5_vla/stage45e_l1_loss.py",
        "l1_manifest": REPO / "manifests/STAGE45E_L1_PHYSICAL_ERROR_AWARE_LOSS_V1.json",
    }
    for key, path in result.items():
        path = path.expanduser().resolve()
        if key in {"base_checkpoint", "full24_dataset", "groot_source", "cosmos_root", "ffmpeg_runtime_root"}:
            _regular_dir(path, key)
        else:
            _regular_file(path, key)
        result[key] = path
    return result


def _validate_l1_code() -> dict[str, Any]:
    contract = l1.loss_weight_contract(l1.canonical_action_mask())
    _require(contract["schema"] == admission.L1_CONTRACT["schema"], "L1 schema drift")
    _require(contract["active_scalar_count"] == admission.L1_CONTRACT["active_scalar_count"], "L1 mask count drift")
    _require(contract["xyz_loss_weights"] == admission.L1_CONTRACT["xyz_loss_weights"], "L1 XYZ weights drift")
    _require(contract["matrix_sha256"] == admission.L1_CONTRACT["weight_matrix_sha256"], "L1 matrix hash drift")
    _require(contract["mask_folding"] is False and contract["adaptive_renormalization"] is False, "L1 loss contract is relaxed")
    return contract


def _validate_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    base_entries = _regular_tree_entries(paths["base_checkpoint"], "payload/GR00T-N1.7-3B")
    base_tree = _tree_digest(base_entries)
    _require(base_tree == admission.BASE_TREE_SHA256, f"BASE tree identity mismatch: {base_tree}")
    dataset_entries = _regular_tree_entries(paths["full24_dataset"], "payload/lerobot_train")
    dataset_tree = _tree_digest(dataset_entries)
    _require(dataset_tree == admission.FULL24_TREE_SHA256, f"FULL24 tree identity mismatch: {dataset_tree}")
    _require(len(dataset_entries) == admission.FULL24_FILE_COUNT, "FULL24 file count drift")
    _require(sum(int(item["bytes"]) for item in dataset_entries) == admission.FULL24_BYTES, "FULL24 byte count drift")
    cosmos_tree, cosmos_count = _cosmos_tree_digest(paths["cosmos_root"])
    _require(cosmos_tree == admission.COSMOS_TREE_SHA256, f"Cosmos tree identity mismatch: {cosmos_tree}")
    revision_path = paths["cosmos_root"] / "refs/main"
    _regular_file(revision_path, "Cosmos refs/main")
    revision = revision_path.read_text(encoding="utf-8").strip()
    _require(revision == admission.COSMOS_REVISION, f"Cosmos revision mismatch: {revision}")

    stats = read_json(paths["full24_runtime_statistics"])
    stats_sha = sha256_file(paths["full24_runtime_statistics"])
    _require(stats_sha == admission.FULL24_STATS_SHA256, "FULL24 runtime stats SHA mismatch")
    _require(stats.get("schema") == admission.FULL24_STATS_SCHEMA, "FULL24 stats schema drift")
    _require(stats.get("pi_gate") == "PI_GATE_STATE_0083", "FULL24 stats authority drift")
    _require(stats.get("protocol_revision") == admission.FULL24_STATS_PROTOCOL, "FULL24 stats protocol drift")
    _require(stats.get("statistics_payload_sha256") == admission.FULL24_STATS_PAYLOAD_SHA256, "FULL24 stats payload field drift")
    _require(canonical_sha(stats.get("statistics", {})) == admission.FULL24_STATS_PAYLOAD_SHA256, "FULL24 stats payload content drift")
    _require(stats.get("canonical_label_source", {}).get("tree_sha256") == admission.FULL24_TREE_SHA256, "stats label source drift")
    _require(sha256_file(paths["base_processor_config"]) == admission.PROCESSOR_CONFIG_SHA256, "BASE processor config drift")
    _require(sha256_file(paths["base_processor_statistics"]) == admission.BASE_PROCESSOR_STATISTICS_SHA256, "BASE processor statistics drift")
    _require(sha256_file(paths["modality_config"]) == admission.MODALITY_SHA256, "modality config drift")
    _require(sha256_file(paths["training_config"]) == admission.CONFIG_SHA256, "training config drift")
    _require(sha256_file(paths["decoder_source"]) == admission.DECODER_SOURCE_SHA256, "decoder source changed")
    _require(sha256_file(paths["metrics_source"]) == admission.METRICS_SOURCE_SHA256, "metrics source changed")
    _require(sha256_file(paths["l1_manifest"]) is not None, "L1 manifest unreadable")

    release = read_json(paths["golden_release_manifest"])
    _require(sha256_file(paths["golden_release_manifest"]) == admission.RELEASE_SHA256, "release artifact SHA mismatch")
    _require(release.get("schema") == "STAGE45B_B2_TRAINING_RELEASE_V2", "release schema drift")
    _require(release.get("status") == "SEALED", "release is not sealed")
    _require(release.get("token") == "VANDA_STAGE45B_B2_PRODUCTION_TRAINING_RELEASE_V2_QUALIFIED", "release token drift")
    release_runtime = release.get("runtime", {})
    _require(release_runtime.get("ffmpeg_runtime_artifact_id") == admission.FFMPEG_RUNTIME_ID, "release FFmpeg identity drift")
    _require(release_runtime.get("ffmpeg_runtime_manifest_sha256") == admission.FFMPEG_RUNTIME_MANIFEST_SHA256, "release FFmpeg manifest drift")

    video = read_json(paths["video_qualification"])
    _require(sha256_file(paths["video_qualification"]) == admission.VIDEO_QUALIFICATION_SHA256, "video qualification SHA mismatch")
    _require(video.get("status") == "PASS", "video qualification status drift")
    _require(video.get("token") == "VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED", "video qualification token drift")
    ffmpeg_manifest = paths["ffmpeg_runtime_root"] / "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json"
    _regular_file(ffmpeg_manifest, "FFmpeg manifest")
    _require(sha256_file(ffmpeg_manifest) == admission.FFMPEG_RUNTIME_MANIFEST_SHA256, "FFmpeg manifest SHA mismatch")
    _regular_dir(paths["ffmpeg_runtime_root"] / "lib", "FFmpeg lib")
    _regular_dir(paths["ffmpeg_runtime_root"] / "compat", "FFmpeg compat")
    _require(paths["ffmpeg_runtime_root"].name == f"STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__{admission.FFMPEG_RUNTIME_ID}", "FFmpeg root identity drift")

    from stage45b_groot_source_snapshot import verify_extracted

    snapshot = verify_extracted(
        paths["groot_source"], paths["groot_manifest"],
        expected_manifest_file_sha256=admission.GROOT_MANIFEST_SHA256,
        expected_manifest_payload_sha256=admission.GROOT_MANIFEST_PAYLOAD_SHA256,
        expected_payload_tree_sha256=admission.GROOT_PAYLOAD_TREE_SHA256,
    )
    _require(snapshot.get("payload_tree_sha256") == admission.GROOT_PAYLOAD_TREE_SHA256, "GR00T snapshot tree drift")
    return {
        "base_tree_sha256": base_tree,
        "full24_tree_sha256": dataset_tree,
        "full24_file_count": len(dataset_entries),
        "full24_bytes": sum(int(item["bytes"]) for item in dataset_entries),
        "cosmos_tree_sha256": cosmos_tree,
        "cosmos_entry_count": cosmos_count,
        "cosmos_revision": revision,
        "full24_stats_sha256": stats_sha,
        "full24_stats_payload_sha256": stats["statistics_payload_sha256"],
        "base_processor_config_sha256": sha256_file(paths["base_processor_config"]),
        "base_processor_statistics_sha256": sha256_file(paths["base_processor_statistics"]),
        "modality_sha256": sha256_file(paths["modality_config"]),
        "training_config_sha256": sha256_file(paths["training_config"]),
        "decoder_source_sha256": sha256_file(paths["decoder_source"]),
        "metrics_source_sha256": sha256_file(paths["metrics_source"]),
        "groot_snapshot": snapshot,
        "release_sha256": sha256_file(paths["golden_release_manifest"]),
        "video_qualification_sha256": sha256_file(paths["video_qualification"]),
        "ffmpeg_runtime_manifest_sha256": sha256_file(ffmpeg_manifest),
        "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "normalization_unchanged": True,
    }


def _build_binding(
    checkout: Mapping[str, Any], gpu: Mapping[str, Any], runtime: Mapping[str, Any],
    paths: Mapping[str, Path], output_root: Path, phase: str = "TRAIN",
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_values = {key: str(value) for key, value in paths.items()}
    path_values["output_root"] = str(output_root)
    zero = {
        "model_construction": 0, "model_forward": 0, "backward": 0,
        "optimizer_steps": 0, "training_steps": 0, "gpu": 1, "pbs": 1,
    }
    boundary = {
        "model_construction": "RUNTIME_AFTER_ADMISSION",
        "model_forward": "RUNTIME_AFTER_ADMISSION",
        "gpu": 1, "h200": 1, "pbs": 1,
        "training": True, "backward": True, "optimizer_steps": 2000,
        "checkpoint_write": 2, "public_sampler_calls": 0, "get_action_calls": 0,
        "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0,
        "N1": 0, "N2": 0, "N2_plus_L1": 0, "D2B": 0, "FULL24@4000": 0,
        "automatic_retry": False,
    }
    execution_checkout = {
        "binding_mode": "OBSERVED_RUNTIME_ONLY",
        "authorized_repo_root": checkout["observed_repo_root"],
        "authorized_branch": checkout["observed_branch"],
        "authorized_head": checkout["observed_head"],
        "authorized_clean_worktree": True,
    }
    binding: dict[str, Any] = {
        "schema": admission.RUNTIME_BINDING_SCHEMA,
        "stage": admission.STAGE,
        "pi_gate": admission.PI_GATE_COMMENT,
        "pi_authority_token": admission.AUTHORITY_TOKEN,
        "execution_authorized": True,
        "phase": phase,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
        "execution_checkout": execution_checkout,
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
        "paths": path_values,
        "device_policy": {
            "selected_device": admission.DEVICE,
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
    return binding, observed


def _set_runtime_environment(output_root: Path, cosmos_root: Path) -> dict[str, str]:
    cache = output_root / "runtime_cache"
    tmp = output_root / "runtime_tmp"
    hub = cache / "huggingface" / "hub"
    for path in (cache, tmp, hub):
        path.mkdir(parents=True, exist_ok=False)
    cosmos_link = hub / "models--nvidia--Cosmos-Reason2-2B"
    cosmos_link.symlink_to(cosmos_root, target_is_directory=True)
    values = {
        "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(cache / "xdg"), "HF_HOME": str(cache / "huggingface"),
        # The admitted Cosmos snapshot is a Hugging Face *hub cache entry*
        # under ``HF_HOME/hub``.  Bind every cache variable to that same hub
        # root so nested Qwen3 ``from_pretrained(repo_id)`` calls cannot look
        # in a second, empty ``HF_HOME`` namespace while offline.
        "HF_HUB_CACHE": str(hub), "HUGGINGFACE_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub), "TORCH_HOME": str(cache / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "GROOT_COMMIT_HASH": admission.GROOT_COMMIT,
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "GROOT_HF_LOCAL_FIRST": "1", "GROOT_PATCH_MISTRAL": "1",
    }
    os.environ.update(values)
    return values


def _load_config(config_path: Path, output_root: Path, base_model: Path, dataset: Path) -> Any:
    from gr00t.configs.base_config import get_default_config
    from stage38_structured_train import normalize_loaded_modality_configs

    config = get_default_config().load(config_path)
    config.load_config_path = None
    config.training.output_dir = str(output_root)
    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(base_model)
    config.training.resume_from_checkpoint = False
    config.training.save_only_model = False
    config.training.max_steps = admission.TRAINING_RECIPE["max_steps"]
    config.training.save_steps = 1000
    config.training.optim = admission.TRAINING_RECIPE["optimizer"]
    config.training.learning_rate = admission.TRAINING_RECIPE["learning_rate"]
    config.training.weight_decay = admission.TRAINING_RECIPE["weight_decay"]
    config.training.lr_scheduler_type = admission.TRAINING_RECIPE["lr_scheduler_type"]
    config.training.warmup_steps = admission.TRAINING_RECIPE["warmup_steps"]
    config.training.warmup_ratio = admission.TRAINING_RECIPE["warmup_ratio"]
    config.training.max_grad_norm = admission.TRAINING_RECIPE["max_grad_norm"]
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.training.global_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.dataloader_num_workers = 0
    config.training.eval_strategy = "no"
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    # GR00T forwards this value to the nested Qwen3 backbone.  It must be the
    # hub cache root, not HF_HOME itself, because the observed Cosmos binding
    # lives at ``HF_HUB_CACHE/models--nvidia--Cosmos-Reason2-2B``.
    config.training.transformers_cache_dir = str(Path(os.environ["HF_HUB_CACHE"]))
    config.data.seed = admission.TRAINING_RECIPE["seed"]
    _require(len(config.data.datasets) == 1, "L1 requires exactly one dataset specification")
    config.data.datasets[0].dataset_paths = [str(dataset)]
    config.data.datasets[0].mix_ratio = 1.0
    config.model.action_horizon = 40
    config.model.max_action_dim = 132
    config.model.max_state_dim = 132
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_visual = False
    config.model.tune_llm = False
    config.model.tune_vlln = False
    config.model.use_relative_action = True
    config.model.use_percentiles = True
    normalize_loaded_modality_configs(config)
    _assert_config(config, base_model, dataset)
    return config


def _assert_config(config: Any, base_model: Path, dataset: Path) -> None:
    checks = {
        "base": config.training.start_from_checkpoint == str(base_model),
        "dataset": len(config.data.datasets) == 1 and config.data.datasets[0].dataset_paths == [str(dataset)],
        "seed": config.data.seed == 2222,
        "steps": config.training.max_steps == 2000 and config.training.save_steps == 1000,
        "batch": config.training.global_batch_size == 1 and config.training.gradient_accumulation_steps == 1,
        "optimizer": config.training.optim == "adamw_torch",
        "lr": float(config.training.learning_rate) == 1e-4 and float(config.training.weight_decay) == 1e-5,
        "schedule": config.training.lr_scheduler_type == "cosine" and config.training.warmup_steps == 10 and float(config.training.warmup_ratio) == 0.0,
        "clip": float(config.training.max_grad_norm) == 1.0,
        "precision": config.training.bf16 is True and config.training.fp16 is False and config.training.tf32 is True,
        "fresh": config.training.save_only_model is False and config.training.resume_from_checkpoint is False,
        "model_shape": config.model.action_horizon == 40 and config.model.max_action_dim == 132 and config.model.max_state_dim == 132,
        "trainable": config.model.tune_projector is True and config.model.tune_diffusion_model is True,
        "frozen": config.model.tune_visual is False and config.model.tune_llm is False and config.model.tune_vlln is False,
        "action": config.model.use_relative_action is True and config.model.use_percentiles is True,
        "no_eval": config.training.eval_strategy == "no",
        "workers": config.training.num_gpus == 1 and config.training.dataloader_num_workers == 0,
    }
    _require(all(checks.values()), "L1 loaded training config drift: " + ",".join(key for key, ok in checks.items() if not ok))


def _parameter_hashes(model: Any, torch: Any) -> dict[str, str]:
    import numpy as np

    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        cpu = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        payload = np.asarray(cpu.numpy()).tobytes()
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(payload)
        result[name] = digest.hexdigest()
    _require(result, "no trainable parameters found for L1 canary")
    return result


def _parameter_delta(before: Mapping[str, str], model: Any, torch: Any) -> dict[str, Any]:
    after = _parameter_hashes(model, torch)
    names = sorted(set(before) | set(after))
    changed = [name for name in names if before.get(name) != after.get(name)]
    return {"parameter_count": len(names), "changed_parameter_count": len(changed), "changed_parameters": changed[:20], "pass": not changed}


def _validate_sidecar(output_root: Path, stats_payload_sha: str) -> dict[str, Any]:
    sidecar_path = output_root / "experiment_cfg" / "structured_action_statistics.json"
    sidecar = read_json(sidecar_path)
    _require(sidecar.get("source_dataset_untouched_by_factory") is True, "structured dataset factory mutation flag drift")
    _require(sidecar.get("upstream_generate_stats_called") is False, "upstream stats generation was called")
    statistics = sidecar.get("statistics", {})
    _require(canonical_sha(statistics) == stats_payload_sha, "L1 changed FULL24 statistics")
    contract = sidecar.get("contract", {})
    # The accepted StructuredAction sidecar names this frozen invariant
    # ``mask_active_scalar_count`` (the dense 40x132 contract is a separate
    # field).  Validate the actual serialized binary mask as well, so a stale
    # count cannot certify a changed mask.
    _require(contract.get("mask_active_scalar_count") == 190, "L1 changed active mask count")
    active_mask = statistics.get("active_mask")
    _require(isinstance(active_mask, list) and len(active_mask) == 40, "L1 active mask horizon drift")
    _require(all(isinstance(row, list) and len(row) == 132 for row in active_mask), "L1 active mask dimension drift")
    flat_mask = [value for row in active_mask for value in row]
    _require(all(type(value) in (int, float) and value in (0, 1) for value in flat_mask), "L1 active mask is not binary")
    _require(sum(flat_mask) == 190, "L1 serialized active mask count drift")
    return {
        "path": str(sidecar_path),
        "sha256": sha256_file(sidecar_path),
        "statistics_payload_sha256": canonical_sha(statistics),
        "mask_active_scalar_count": 190,
        "mask_shape": [40, 132],
        "mask_binary": True,
    }


def _checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    _regular_dir(path, f"checkpoint-{expected_step}")
    state = read_json(path / "trainer_state.json")
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    missing = [name for name in required if not (path / name).is_file()]
    entries = _regular_tree_entries(path, f"stage45e_l1/{path.name}")
    return {
        "path": str(path), "global_step": state.get("global_step"), "expected_step": expected_step,
        "required_state_files_present": not missing, "missing_state_files": missing,
        "tree_sha256": _tree_digest(entries), "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "status": "PASS" if state.get("global_step") == expected_step and not missing else "BLOCKED",
    }


def _append_event(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "timestamp_utc": now_utc(), **jsonable(payload)}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _one_batch_backward_canary(
    config: Any, canary_root: Path, base_model: Path, dataset: Path, *, state: dict[str, Any]
) -> dict[str, Any]:
    import gc
    import torch

    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    state["phase"] = "CANARY_MODEL_CONSTRUCTION"
    state["model_construction_started"] = True
    # GR00T writes its resolved model config during ``setup`` rather than
    # creating the supplied save directory itself.  Keep the canary's config
    # namespace separate from the official training root and materialize it
    # before model construction begins.
    canary_config_dir = canary_root / "experiment_cfg"
    canary_config_dir.mkdir(parents=True, exist_ok=False)
    pipeline = Gr00tN1d7Pipeline(config, canary_config_dir)
    pipeline.setup()
    state["model_construction_completed"] = True
    model = pipeline.return_model()
    model.to(device=admission.DEVICE)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(trainable, "L1 canary found no trainable parameters")
    frozen_bad = [name for name, parameter in model.named_parameters() if any(token in name.lower() for token in ("backbone", "vlln", "vl_self_attention")) and parameter.requires_grad]
    _require(not frozen_bad, "L1 canary observed frozen-module parameters as trainable: " + ",".join(frozen_bad[:10]))
    sample = next(iter(pipeline.return_dataset()[0]))
    batch = pipeline.return_collator()([sample])
    inputs = batch["inputs"]
    mask = inputs["action_mask"].detach().cpu().numpy()[0]
    _validate_l1_code()
    l1.loss_weight_contract(mask)

    model.train()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=admission.TRAINING_RECIPE["learning_rate"],
        weight_decay=admission.TRAINING_RECIPE["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    before = _parameter_hashes(model, torch)
    state["phase"] = "CANARY_BACKWARD"
    state["backward_started"] = True
    # The official GR00T run creates ``TrainingArguments(bf16=True)`` and
    # Accelerate wraps ``model.forward`` in CUDA BF16 autocast.  This canary
    # calls the model directly, so reproduce that exact execution context;
    # casting the model or changing the frozen precision contract would be a
    # different experiment.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
    loss = outputs["loss"]
    _require(bool(torch.isfinite(loss).item()), "L1 canary loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    _require(gradients, "L1 canary produced no gradients")
    _require(all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients), "L1 canary gradient is non-finite")
    delta = _parameter_delta(before, model, torch)
    _require(delta["pass"], "L1 canary parameter mutation occurred before optimizer.step")
    checkpoint_dirs = list(canary_root.glob("checkpoint-*"))
    _require(not checkpoint_dirs, "L1 canary wrote a checkpoint")
    counters = l1.runtime_counters()
    result = {
        "schema": "VANDA_STAGE45E_L1_ONE_BATCH_BACKWARD_CANARY_V1",
        "status": "PASS",
        "device": admission.DEVICE,
        "loss": float(loss.detach().float().cpu().item()),
        "action_mask_shape": list(mask.shape),
        "active_scalar_count": int(mask.sum()),
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in trainable),
        "trainable_parameter_tensor_count": len(trainable),
        "gradient_tensor_count": len(gradients),
        "parameter_delta": delta,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "action_head_forwards": counters["action_head_forwards"],
        "precision": {"bf16": True, "fp16": False},
        "stats_unchanged": True,
        "mask_unchanged": True,
        "normalization_unchanged": True,
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer, outputs, batch, inputs, model, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def execute(args: argparse.Namespace) -> int:
    state: dict[str, Any] = {
        "phase": "START",
        "model_imported": False,
        "model_construction_started": False,
        "model_construction_completed": False,
        "training_started": False,
        "backward_started": False,
        "optimizer_steps_observed": None,
        "checkpoint_writes_observed": None,
        "action_head_forwards": 0,
        "retry_permitted": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc_steps": 0,
        "simulator_steps": 0,
        "hardware_steps": 0,
    }
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        _require(os.environ.get("PBS_JOBID"), "--execute requires a PBS allocation")
        _require(Path(sys.executable).resolve() == Path(args.python).expanduser().resolve(), "exact runtime Python mismatch")
        checkout = checkout_observation(args.expected_head)
        gpu = gpu_observation()
        runtime = runtime_identity(gpu)
        paths = _paths(args)
        _require(str(paths["base_checkpoint"]) == DEFAULTS["base_model"], "BASE path is not the sealed path")
        _require(str(paths["full24_dataset"]) == DEFAULTS["dataset"], "FULL24 path is not the sealed path")
        _require(str(paths["groot_source"]) == DEFAULTS["groot_source"], "GR00T source path is not the sealed path")
        _require(str(paths["cosmos_root"]) == DEFAULTS["cosmos_root"], "Cosmos path is not the sealed path")
        _require(str(paths["full24_runtime_statistics"]) == DEFAULTS["stats"], "stats path is not the sealed path")
        code_contract = _validate_l1_code()
        input_audit = _validate_inputs(paths)
        _require(not output_root.exists() and not output_root.is_symlink(), f"L1 output root must be fresh: {output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
        usage = shutil.disk_usage(output_root)
        _require(usage.free >= 64 * 1024**3, f"insufficient free scratch for full-state checkpoints: {usage.free}")
        binding, observed = _build_binding(checkout, gpu, runtime, paths, output_root, "TRAIN")
        admission_result = admission.validate_pre_model_binding(
            binding, observed, authority_token=admission.AUTHORITY_TOKEN, phase="TRAIN"
        )
        write_json(output_root / "stage45e_l1_runtime_binding.json", {"binding": binding, "observed": observed})
        write_json(output_root / "stage45e_l1_pre_model_admission.json", admission_result)
        write_json(output_root / "stage45e_l1_input_audit.json", {"schema": "VANDA_STAGE45E_L1_INPUT_AUDIT_V1", "status": "PASS", "code_contract": code_contract, "identities": input_audit, "paths": {key: str(value) for key, value in paths.items()}, "training_recipe": admission.TRAINING_RECIPE})
        events_path = output_root / "stage45e_l1_events.jsonl"
        events_path.touch()
        _append_event(events_path, "PRE_MODEL_ADMISSION_PASS", execution_head=checkout["observed_head"], gpu=gpu)
        env_values = _set_runtime_environment(output_root, paths["cosmos_root"])
        sys.path.insert(0, str(paths["groot_source"]))
        _bind_cosmos_processor(paths["cosmos_root"] / "snapshots" / admission.COSMOS_REVISION)
        from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
        install_structured_dataset_factory()
        from gr00t.experiment.experiment import run as gr00t_run
        l1_install = l1.install_l1_loss_patch()
        state["model_imported"] = True
        state["phase"] = "MODEL_IMPORT_COMPLETE"
        _append_event(events_path, "MODEL_IMPORT_COMPLETE", l1_install=l1_install)

        canary_root = output_root / "canary"
        canary_root.mkdir(parents=False, exist_ok=False)
        canary_config = _load_config(paths["training_config"], canary_root, paths["base_checkpoint"], paths["full24_dataset"])
        l1.reset_runtime_counters()
        canary = _one_batch_backward_canary(canary_config, canary_root, paths["base_checkpoint"], paths["full24_dataset"], state=state)
        write_json(output_root / "stage45e_l1_backward_canary.json", canary)
        _append_event(events_path, "ONE_BATCH_BACKWARD_CANARY_PASS", canary=canary)

        l1.reset_runtime_counters()
        state["phase"] = "FULL_TRAINING"
        state["training_started"] = True
        _append_event(events_path, "TRAINING_START", optimizer_steps=2000, checkpoint_steps=[1000, 2000])
        config = _load_config(paths["training_config"], output_root, paths["base_checkpoint"], paths["full24_dataset"])
        gr00t_run(config)
        _append_event(events_path, "TRAINING_RETURNED")
        state["phase"] = "POST_TRAINING_AUDIT"
        checkpoint_1000 = _checkpoint_record(output_root / "checkpoint-1000", 1000)
        checkpoint_2000 = _checkpoint_record(output_root / "checkpoint-2000", 2000)
        _require(checkpoint_1000["status"] == "PASS" and checkpoint_2000["status"] == "PASS", "L1 checkpoint closure failed")
        sidecar = _validate_sidecar(output_root, admission.FULL24_STATS_PAYLOAD_SHA256)
        counters = l1.runtime_counters()
        _require(counters["action_head_forwards"] == 2000, f"L1 training action-head count drift: {counters}")
        state["optimizer_steps_observed"] = 2000
        state["checkpoint_writes_observed"] = 2
        result = {
            "schema": "VANDA_STAGE45E_L1_H200_TRAINING_RESULT_V1",
            "status": admission.FULL_PASS_TOKEN,
            "token": admission.FULL_PASS_TOKEN,
            "stage": admission.STAGE,
            "pi_gate": admission.PI_GATE_COMMENT,
            "branch": checkout["observed_branch"],
            "execution_head": checkout["observed_head"],
            "pbs_jobid": os.environ["PBS_JOBID"],
            "gpu": gpu,
            "runtime_identity": runtime,
            "intervention": admission.L1_CONTRACT,
            "input_audit": input_audit,
            "l1_patch": l1_install,
            "one_batch_backward_canary": canary,
            "checkpoint_1000": checkpoint_1000,
            "checkpoint_2000": checkpoint_2000,
            "structured_stats": sidecar,
            "runtime_counters": counters,
            "training_steps": 2000,
            "backward_steps": 2000,
            "optimizer_steps": 2000,
            "checkpoint_writes": 2,
            "model_construction": 2,
            "model_forward": 2000,
            "normalization_unchanged": True,
            "mask_unchanged": True,
            "decoder_unchanged": True,
            "evaluation": {"authorized": True, "executed": False, "next_entrypoint": "stage45e_l1_train24_eval.py"},
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_steps": 0,
            "simulator_steps": 0,
            "hardware_steps": 0,
            "automatic_retry": False,
            "timestamp_utc": now_utc(),
        }
        write_json(output_root / "stage45e_l1_training_result.json", result)
        return 0
    except Exception as exc:
        state["action_head_forwards"] = l1.runtime_counters().get("action_head_forwards", 0)
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
        state["timestamp_utc"] = now_utc()
        state["status"] = "STAGE45E_L1_TRAINING_BLOCKED"
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "stage45e_l1_failure.json", {
                "schema": "VANDA_STAGE45E_L1_FAILURE_SEAL_V1",
                "status": "FAIL",
                "pi_gate": admission.PI_GATE_COMMENT,
                "execution_head": git("rev-parse", "HEAD") if (REPO / ".git").exists() else None,
                "phase": state.get("phase"),
                "model_imported": state.get("model_imported"),
                "model_construction_started": state.get("model_construction_started"),
                "model_construction_completed": state.get("model_construction_completed"),
                "training_started": state.get("training_started"),
                "backward_started": state.get("backward_started"),
                "optimizer_steps_observed": state.get("optimizer_steps_observed"),
                "checkpoint_writes_observed": state.get("checkpoint_writes_observed"),
                "action_head_forwards": state.get("action_head_forwards"),
                "retry_permitted": False,
                "validation_e_read": False,
                "holdout_f_read": False,
                "wbc_steps": 0,
                "simulator_steps": 0,
                "hardware_steps": 0,
                "error": state.get("error"),
                "traceback": state.get("traceback"),
                "timestamp_utc": state.get("timestamp_utc"),
            })
        except Exception:
            pass
        print(f"STAGE45E_L1_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def plan() -> dict[str, Any]:
    static = admission.validate_static_contract()
    code = _validate_l1_code()
    return {
        "schema": "VANDA_STAGE45E_L1_H200_TRAINING_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "stage": admission.STAGE,
        "authority": static["authority"],
        "branch": admission.BRANCH,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
        "l1_contract": code,
        "training_recipe": admission.TRAINING_RECIPE,
        "evaluation_call_budget": admission.EVAL_CALL_BUDGET,
        "evaluation_sampler": admission.EVAL_SAMPLER_CONTRACT,
        "h200": 0,
        "pbs": 0,
        "model_construction": 0,
        "forward": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "automatic_retry": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, required=False)
    parser.add_argument("--expected-head", default=os.environ.get("STAGE45E_EXPECTED_HEAD"))
    for name, default in DEFAULTS.items():
        option = "--" + name.replace("_", "-")
        dest = name
        parser.add_argument(option, dest=dest, type=Path, default=Path(default))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required with --execute")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
