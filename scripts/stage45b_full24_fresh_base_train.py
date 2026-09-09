#!/usr/bin/env python3
"""Run the single authorized Stage45B FULL24 fresh-base H200 exposure.

The script is intentionally a handoff runner rather than a scheduler.  The
Hopper-side PBS template supplies the exact paths and allocation.  All
identity, contract, and counter checks happen before the first model import;
once the optimizer has stepped, a failure is recorded and is never retried by
this program.
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
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from stage45b_external_admission import (  # noqa: E402
    BASE_TREE_SHA256,
    B0_PAYLOAD_SHA256,
    B0_STATS_SHA256,
    COSMOS_REVISION,
    COSMOS_TREE_SHA256,
    FULL24_TREE_SHA256,
    GROOT_COMMIT,
    GROOT_TREE,
    MODALITY_SHA256,
    PI_GATE,
    RUNTIME_PROTOCOL_REVISION,
    RUNTIME_STATS_SCHEMA,
    RUNTIME_STATS_SHA256,
    RUNTIME_PAYLOAD_SHA256,
    assert_bundle,
)
from stage45b_groot_source_snapshot import verify_extracted  # noqa: E402
from stage45b_production_release import assert_release  # noqa: E402
from stage45b_video_decoder_runtime import verify_environment, verify_runtime_artifact  # noqa: E402


STAGE = "Stage45B"
BRANCH = "stage45b/full24-fresh-base-execution"
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
EXPECTED_TRAINING_TOKEN = "HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_COMPLETE"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            if child.is_symlink() or not stat.S_ISREG(mode):
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _cosmos_tree_digest(root: Path) -> tuple[str, int]:
    root = Path(root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Cosmos cache root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(item)
                resolved = item.resolve(strict=True)
                try:
                    resolved_rel = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise RuntimeError(f"Cosmos symlink escapes root: {item}") from exc
                if not resolved.is_file():
                    raise RuntimeError(f"Cosmos symlink does not resolve to a file: {item}")
                entries.append({
                    "path": rel, "kind": "symlink", "target": target,
                    "resolved_path": resolved_rel,
                    "size_bytes": int(resolved.stat().st_size),
                    "sha256": sha256_file(resolved),
                })
            elif stat.S_ISREG(mode):
                entries.append({
                    "path": rel, "kind": "regular_file",
                    "size_bytes": int(item.stat().st_size),
                    "sha256": sha256_file(item),
                })
            else:
                raise RuntimeError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(
            "\t".join((
                str(entry["kind"]), str(entry["path"]),
                str(entry.get("size_bytes", 0)),
                str(entry.get("sha256") or "-"),
                str(entry.get("target") or "-"),
                str(entry.get("resolved_path") or "-"),
            )).encode("utf-8") + b"\n"
        )
    return digest.hexdigest(), len(entries)


def _gpu_identity() -> dict[str, Any]:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for Stage45B training")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {query.stderr[-1000:]}")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Stage45B requires exactly one visible H200, observed {rows!r}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 5 or "H200" not in fields[1]:
        raise RuntimeError(f"Stage45B requires H200, observed {rows[0]!r}")
    return {
        "pbs_jobid": os.environ["PBS_JOBID"], "host": subprocess.check_output(["hostname"], text=True).strip(),
        "index": fields[0], "name": fields[1], "uuid": fields[2],
        "memory_total_mib": int(fields[3]), "memory_used_mib_at_start": int(fields[4]),
    }


def _assert_bundle(bundle: Mapping[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    return assert_bundle(
        bundle,
        repo=REPO,
        expected_head=git("rev-parse", "HEAD"),
        bundle_path=bundle_path,
    )


def _load_config(config_path: Path, *, output_root: Path, base_model: Path, dataset: Path) -> Any:
    # Imports below are deliberately after all identity/GPU checks.
    from gr00t.configs.base_config import get_default_config
    from stage38_structured_train import normalize_loaded_modality_configs

    config = get_default_config().load(config_path)
    config.load_config_path = None
    config.training.output_dir = str(output_root)
    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(base_model)
    config.training.resume_from_checkpoint = False
    config.training.save_only_model = False
    config.training.max_steps = 2000
    config.training.save_steps = 1000
    config.training.optim = "adamw_torch"
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-5
    config.training.lr_scheduler_type = "cosine"
    config.training.warmup_steps = 10
    config.training.warmup_ratio = 0.0
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.training.global_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.dataloader_num_workers = 0
    config.training.eval_strategy = "no"
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.data.seed = 2222
    if len(config.data.datasets) != 1:
        raise RuntimeError("Stage45B requires one dataset specification")
    config.data.datasets[0].dataset_paths = [str(dataset)]
    config.data.datasets[0].mix_ratio = 1.0
    config.model.action_horizon = HORIZON
    config.model.max_action_dim = ACTION_DIM
    config.model.max_state_dim = ACTION_DIM
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_visual = False
    config.model.tune_llm = False
    config.model.tune_vlln = False
    config.model.use_relative_action = True
    config.model.use_percentiles = True
    normalize_loaded_modality_configs(config)
    return config


def _bind_cosmos_processor(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise RuntimeError(f"Cosmos snapshot is unavailable: {snapshot}")
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
    original = getattr(processing, "_stage45b_original_build_processor", None)
    if original is None:
        original = processing.build_processor
        processing._stage45b_original_build_processor = original

        def build_local_processor(model_name: str, loading_kwargs: dict) -> Any:
            if model_name == "nvidia/Cosmos-Reason2-2B":
                model_name = str(snapshot)
            return original(model_name, loading_kwargs)

        processing.build_processor = build_local_processor
    return {"repo_id": "nvidia/Cosmos-Reason2-2B", "snapshot": str(snapshot), "revision": COSMOS_REVISION}


def _checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"missing checkpoint directory: {path}")
    state_path = path / "trainer_state.json"
    state = read_json(state_path)
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    missing = [name for name in required if not (path / name).is_file()]
    entries = _regular_tree_entries(path, f"stage45b/{path.name}")
    return {
        "path": str(path), "global_step": state.get("global_step"), "expected_step": expected_step,
        "required_state_files_present": not missing, "missing_state_files": missing,
        "tree_sha256": _tree_digest(entries), "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "status": "PASS" if state.get("global_step") == expected_step and not missing else "BLOCKED",
    }


def run(args: argparse.Namespace) -> int:
    bundle_path_arg = Path(args.bundle)
    bundle_path = bundle_path_arg.resolve()
    bundle = read_json(bundle_path)
    _assert_bundle(bundle, bundle_path=bundle_path_arg)
    release = None
    runtime_root = None
    video_runtime = None
    if args.release is not None:
        if not args.expected_release_sha256:
            raise RuntimeError("production release SHA is required when a release is supplied")
        release_path_arg = Path(args.release)
        release = read_json(release_path_arg)
        assert_release(
            release,
            repo=REPO,
            expected_head=git("rev-parse", "HEAD"),
            release_path=release_path_arg,
            expected_release_sha256=args.expected_release_sha256,
            bundle_path=bundle_path_arg,
        )
        video_runtime = release.get("video_decoder_runtime")
        if not isinstance(video_runtime, Mapping):
            raise RuntimeError("Stage45B Release V2 has no native video runtime binding")
        runtime_root = Path(str(video_runtime.get("ffmpeg_runtime_root", ""))).resolve()
        verify_runtime_artifact(
            runtime_root,
            expected_id=str(video_runtime.get("ffmpeg_runtime_artifact_id", "")),
        )
        verify_environment(runtime_root)
        if video_runtime.get("ffmpeg_runtime_artifact_id") != release["runtime"].get("ffmpeg_runtime_artifact_id"):
            raise RuntimeError("Release V2 FFmpeg identity is internally inconsistent")
    else:
        raise RuntimeError("Stage45B B2 requires the immutable Release V2 admission")
    gpu = _gpu_identity()

    paths = bundle["runtime_paths"]
    base_model = Path(args.base_model_path or paths["base_model"]).resolve()
    dataset = Path(args.dataset_dir or paths["dataset"]).resolve()
    groot_value = args.groot_source or paths.get("groot_source")
    if not groot_value:
        raise RuntimeError("sealed gitless GR00T source root must be supplied by PBS")
    groot = Path(groot_value).resolve()
    cosmos = Path(args.cosmos_root or paths["cosmos_cache"]).resolve()
    modality = Path(args.modality_config or paths["modality_config"]).resolve()
    config_path = Path(args.config_path or paths["training_config"]).resolve()
    stats_artifact = Path(args.stats_artifact or paths["structured_stats_artifact"]).resolve()
    for path, label in ((base_model, "BASE"), (dataset, "FULL24"), (groot, "GR00T"), (cosmos, "Cosmos"),
                        (modality, "modality"), (config_path, "training config"),
                        (stats_artifact, "runtime-parquet V2 statistics")):
        if not path.exists():
            raise RuntimeError(f"{label} path is unavailable: {path}")

    base_entries = _regular_tree_entries(base_model, "payload/GR00T-N1.7-3B")
    if _tree_digest(base_entries) != BASE_TREE_SHA256:
        raise RuntimeError("BASE tree identity mismatch")
    data_entries = _regular_tree_entries(dataset, "payload/lerobot_train")
    if _tree_digest(data_entries) != FULL24_TREE_SHA256:
        raise RuntimeError("FULL24 tree identity mismatch")
    cosmos_tree, cosmos_count = _cosmos_tree_digest(cosmos)
    if cosmos_tree != COSMOS_TREE_SHA256:
        raise RuntimeError("Cosmos cache tree identity mismatch")
    revision = (cosmos / "refs/main").read_text(encoding="utf-8").strip()
    if revision != COSMOS_REVISION:
        raise RuntimeError(f"Cosmos revision mismatch: {revision}")
    snapshot = bundle["source_snapshot"]
    snapshot_check = verify_extracted(
        groot,
        Path(str(snapshot["manifest_path"])),
        expected_manifest_file_sha256=str(snapshot["manifest_file_sha256"]),
        expected_manifest_payload_sha256=str(snapshot["manifest_payload_sha256"]),
        expected_payload_tree_sha256=str(snapshot["payload_tree_sha256"]),
    )
    if snapshot_check["payload_tree_sha256"] != snapshot["payload_tree_sha256"]:
        raise RuntimeError("GR00T sealed source payload identity mismatch")
    groot_head = GROOT_COMMIT
    if sha256_file(modality) != MODALITY_SHA256:
        raise RuntimeError("modality configuration hash mismatch")
    if sha256_file(config_path) != bundle["training_config"]["sha256"]:
        raise RuntimeError("training config hash mismatch")
    stats_document = read_json(stats_artifact)
    if sha256_file(stats_artifact) != bundle["structured_stats"]["sha256"]:
        raise RuntimeError("runtime-parquet V2 statistics artifact hash mismatch")
    if (
        stats_document.get("schema") != RUNTIME_STATS_SCHEMA
        or stats_document.get("pi_gate") != "PI_GATE_STATE_0083"
        or stats_document.get("protocol_revision") != RUNTIME_PROTOCOL_REVISION
        or stats_document.get("statistics_payload_sha256")
        != bundle["structured_stats"]["payload_sha256"]
        or stats_document.get("canonical_label_source", {}).get("tree_sha256")
        != FULL24_TREE_SHA256
    ):
        raise RuntimeError("runtime-parquet V2 statistics contract mismatch")

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"training output root must be new: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    usage = shutil.disk_usage(output_root)
    if usage.free < 64 * 1024**3:
        raise RuntimeError(f"insufficient free space for full-state checkpoints: {usage.free}")
    cache_root = output_root / "cache"
    tmp_root = output_root / "tmp"
    cache_root.mkdir(); tmp_root.mkdir()
    hub_root = cache_root / "huggingface" / "hub"
    hub_root.mkdir(parents=True)
    cosmos_link = hub_root / "models--nvidia--Cosmos-Reason2-2B"
    cosmos_link.symlink_to(cosmos, target_is_directory=True)

    write_json(output_root / "stage45b_runtime_preflight.json", {
        "schema": "CORELAB_STAGE45B_RUNTIME_PREFLIGHT_V1", "pi_gate": PI_GATE,
        "repository_head": git("rev-parse", "HEAD"), "gpu": gpu,
        "paths": {"base_model": str(base_model), "dataset": str(dataset), "groot": str(groot),
                  "cosmos": str(cosmos), "modality": str(modality), "stats_artifact": str(stats_artifact),
                  "ffmpeg_runtime": str(runtime_root)},
        "base_tree_sha256": BASE_TREE_SHA256, "full24_tree_sha256": FULL24_TREE_SHA256,
        "cosmos_tree_sha256": cosmos_tree, "cosmos_entry_count": cosmos_count,
        "groot_sha": groot_head,
        "groot_tree": GROOT_TREE,
        "groot_snapshot": snapshot_check,
        "runtime_stats_sha256": sha256_file(stats_artifact),
        "statistics_payload_sha256": stats_document["statistics_payload_sha256"],
        "historical_b0_stats_sha256": B0_STATS_SHA256,
        "protocol_revision": RUNTIME_PROTOCOL_REVISION,
        "ffmpeg_runtime_artifact_id": video_runtime["ffmpeg_runtime_artifact_id"],
        "ffmpeg_runtime_manifest_sha256": video_runtime["ffmpeg_runtime_manifest_sha256"],
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "model_constructed": False, "gr00t_forwards": 0, "wbc_steps": 0, "reference_updates": 0,
        "validation_e_read": False, "holdout_f_read": False, "training_steps": 0, "optimizer_steps": 0,
        "status": "PASS",
    })
    os.environ.update({"TMPDIR": str(tmp_root), "TMP": str(tmp_root), "TEMP": str(tmp_root),
                       "XDG_CACHE_HOME": str(cache_root / "xdg"), "HF_HOME": str(cache_root / "huggingface"),
                       "TRANSFORMERS_CACHE": str(hub_root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "GROOT_COMMIT_HASH": GROOT_COMMIT, "PYTHONNOUSERSITE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1",
                       "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1"})
    sys.path.insert(0, str(groot))
    _bind_cosmos_processor(cosmos / "snapshots" / COSMOS_REVISION)
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
    install_structured_dataset_factory()
    from gr00t.experiment.experiment import run as gr00t_run
    config = _load_config(config_path, output_root=output_root, base_model=base_model, dataset=dataset)
    with (output_root / "stage45b_events.jsonl").open("w", encoding="utf-8") as events:
        events.write(json.dumps({
            "event": "MODEL_CONSTRUCTION_ALLOWED",
            "timestamp_utc": now_utc(),
            "label_anchor_reconciliation": "STAGE45B_LABEL_ANCHOR_RECONCILIATION",
        }) + "\n")
        events.flush()
        events.write(json.dumps({"event": "TRAIN_CALL_ENTER", "timestamp_utc": now_utc()}) + "\n")
        events.flush()
    gr00t_run(config)
    with (output_root / "stage45b_events.jsonl").open("a", encoding="utf-8") as events:
        events.write(json.dumps({"event": "TRAIN_CALL_RETURN", "timestamp_utc": now_utc()}) + "\n")

    checkpoint_1000 = _checkpoint_record(output_root / "checkpoint-1000", 1000)
    checkpoint_2000 = _checkpoint_record(output_root / "checkpoint-2000", 2000)
    sidecar = read_json(output_root / "experiment_cfg" / "structured_action_statistics.json")
    if canonical_sha(sidecar.get("statistics", {})) != stats_document["statistics_payload_sha256"]:
        raise RuntimeError("runtime StructuredAction statistics differ from runtime-parquet V2")
    result = {
        "schema": "CORELAB_STAGE45B_FULL24_FRESH_BASE_TRAINING_RESULT_V2",
        "status": EXPECTED_TRAINING_TOKEN if checkpoint_1000["status"] == "PASS" and checkpoint_2000["status"] == "PASS" else "HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_INCOMPLETE",
        "pi_gate": PI_GATE, "stage": STAGE, "intervention": "B_FRESH_BASE_FULL24",
        "repository_head": git("rev-parse", "HEAD"), "pbs_jobid": gpu["pbs_jobid"], "gpu": gpu,
        "checkpoint_1000": checkpoint_1000, "checkpoint_2000": checkpoint_2000,
        "structured_stats_sha256": sha256_file(stats_artifact),
        "structured_stats_payload_sha256": stats_document["statistics_payload_sha256"],
        "historical_b0_stats_sha256": B0_STATS_SHA256,
        "groot_source_commit": groot_head,
        "groot_source_tree": GROOT_TREE,
        "groot_snapshot": snapshot_check,
        "protocol_revision": RUNTIME_PROTOCOL_REVISION,
        "ffmpeg_runtime_artifact_id": video_runtime["ffmpeg_runtime_artifact_id"],
        "ffmpeg_runtime_manifest_sha256": video_runtime["ffmpeg_runtime_manifest_sha256"],
        "label_anchor_reconciliation": True,
        "training_steps": 2000,
        "optimizer_steps": 2000, "backward_steps": 2000, "gr00t_forwards": 0, "wbc_steps": 0,
        "reference_updates": 0, "validation_e_read": False, "holdout_f_read": False,
        "model_constructed": True, "training": True,
        "next_gate": "GPT_PI_REVIEW_AFTER_STAGE45B_B2_PI0087_RETURN", "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage45b_training_result.json", result)
    return 0 if result["status"] == EXPECTED_TRAINING_TOKEN else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--cosmos-root", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--stats-artifact", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--expected-release-sha256")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "stage45b_failure.json", {
            "schema": "CORELAB_STAGE45B_FAILURE_V1", "pi_gate": PI_GATE, "stage": STAGE,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "optimizer_step_observed": None, "retry_permitted_by_runner": False,
            "model_construction_status": "UNKNOWN", "gr00t_forwards": "UNKNOWN", "wbc_steps": 0,
            "validation_e_read": False, "holdout_f_read": False, "timestamp_utc": now_utc(),
        })
        print(f"STAGE45B_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
