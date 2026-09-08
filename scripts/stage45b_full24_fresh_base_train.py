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

PI_GATE = "PI_GATE_STATE_0079"
STAGE = "Stage45B"
BRANCH = "stage45b/full24-fresh-base-execution"
SOURCE_HEAD = "SOURCE_HEAD_FILLED_BY_HANDOFF_MANIFEST"
B0_STATS_SHA256 = "49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
STATS_PAYLOAD_SHA256 = "c31c818121ac01a402fc70e577db225ffb6f5d110649aef9dada94338c8a5032"
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


def _assert_bundle(bundle: Mapping[str, Any], *, bundle_path: Path) -> None:
    if bundle.get("schema") != "CORELAB_STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1":
        raise RuntimeError("unexpected Stage45B bundle schema")
    if bundle.get("status") != "CORELAB_STAGE45B_H200_ADMISSION_READY":
        raise RuntimeError("B1 bundle is not admission-ready")
    if bundle.get("pi_gate") != PI_GATE or bundle.get("stage") != STAGE:
        raise RuntimeError("PI gate/stage binding mismatch")
    repository = bundle.get("repository")
    if not isinstance(repository, Mapping) or repository.get("branch") != BRANCH:
        raise RuntimeError("Stage45B branch binding mismatch")
    source_head = str(repository.get("source_head", ""))
    if len(source_head) != 40 or source_head == SOURCE_HEAD:
        raise RuntimeError("bundle source_head is not filled with a commit SHA")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("current checkout is not the Stage45B branch")
    current_head = git("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", source_head, current_head],
        check=False,
    ).returncode != 0:
        raise RuntimeError(f"checkout HEAD {current_head} does not descend from bundle source {source_head}")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("Stage45B checkout must be clean")
    files = bundle.get("execution_files")
    if not isinstance(files, Mapping):
        raise RuntimeError("execution_files binding is missing")
    for key in ("runner", "pbs"):
        relative = Path(str(files[key]["path"]))
        observed = sha256_file(REPO / relative)
        if observed != files[key]["sha256"]:
            raise RuntimeError(f"{key} hash mismatch: {observed}")
    stats = bundle.get("structured_stats")
    if not isinstance(stats, Mapping) or stats.get("sha256") != B0_STATS_SHA256 or stats.get("payload_sha256") != STATS_PAYLOAD_SHA256:
        raise RuntimeError("B0 statistics identity mismatch")
    transport = stats.get("embedded_transport")
    if not isinstance(transport, Mapping):
        raise RuntimeError("embedded B0 transport binding is missing")
    transport_path = REPO / str(transport.get("path", ""))
    if not transport_path.is_file() or transport_path.is_symlink() or sha256_file(transport_path) != transport.get("sha256"):
        raise RuntimeError("embedded B0 transport hash mismatch")
    contract = bundle.get("contract")
    expected_contract = {
        "language": "move the end effector to the red target", "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "target_xyz_in_model_input": False, "state_v1_shape": [STATE_DIM], "body_rgb_shape": [96, 96, 3],
        "wrist_rgb_shape": [96, 96, 3], "structured_action_schema": "StructuredActionV1", "geometry": "K16",
        "timing": "T16", "horizon": HORIZON, "universal_action_shape": [1, HORIZON, ACTION_DIM],
        "decoded_physical_action_shape": [HORIZON, STATE_DIM], "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
        "tcp_semantic_id": "x5_link6_gripper_center",
    }
    if not isinstance(contract, Mapping):
        raise RuntimeError("contract binding is missing")
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise RuntimeError(f"contract mismatch for {key}: {contract.get(key)!r}")
    recipe = bundle.get("recipe")
    expected_recipe = {
        "seed": 2222, "max_steps": 2000, "save_steps": [1000, 2000], "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "optimizer": "adamw_torch", "learning_rate": 1e-4, "weight_decay": 1e-5,
        "lr_scheduler_type": "cosine", "warmup_steps": 10, "warmup_ratio": 0.0, "bf16": True, "fp16": False,
        "tf32": True, "resume_from_checkpoint": False, "save_only_model": False, "tune_projector": True,
        "tune_diffusion_model": True, "tune_visual": False, "tune_llm": False, "tune_vlln": False,
    }
    if not isinstance(recipe, Mapping):
        raise RuntimeError("recipe binding is missing")
    for key, value in expected_recipe.items():
        if recipe.get(key) != value:
            raise RuntimeError(f"recipe mismatch for {key}: {recipe.get(key)!r}")
    if bundle.get("intervention") != "B_FRESH_BASE_FULL24":
        raise RuntimeError("non-primary intervention is forbidden")
    forbidden = bundle.get("forbidden", {})
    if forbidden.get("option_a_warm_start") is not True or forbidden.get("validation_e_read") is not True or forbidden.get("holdout_f_read") is not True:
        raise RuntimeError("bundle does not lock forbidden branches")
    if bundle_path.is_symlink():
        raise RuntimeError("bundle path must not be a symlink")


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
    bundle_path = Path(args.bundle).resolve()
    bundle = read_json(bundle_path)
    _assert_bundle(bundle, bundle_path=bundle_path)
    gpu = _gpu_identity()

    paths = bundle["runtime_paths"]
    base_model = Path(args.base_model_path or paths["base_model"]).resolve()
    dataset = Path(args.dataset_dir or paths["dataset"]).resolve()
    groot = Path(args.groot_source or paths["groot_source"]).resolve()
    cosmos = Path(args.cosmos_root or paths["cosmos_cache"]).resolve()
    modality = Path(args.modality_config or paths["modality_config"]).resolve()
    config_path = Path(args.config_path or paths["training_config"]).resolve()
    stats_artifact = Path(args.stats_artifact or paths["structured_stats_artifact"]).resolve()
    for path, label in ((base_model, "BASE"), (dataset, "FULL24"), (groot, "GR00T"), (cosmos, "Cosmos"),
                        (modality, "modality"), (config_path, "training config"), (stats_artifact, "B0 statistics")):
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
    groot_head = subprocess.check_output(["git", "-C", str(groot), "rev-parse", "HEAD"], text=True).strip()
    if groot_head != GROOT_SHA:
        raise RuntimeError("GR00T source revision mismatch")
    if subprocess.check_output(["git", "-C", str(groot), "status", "--porcelain=v1"], text=True):
        raise RuntimeError("GR00T source checkout is dirty")
    if sha256_file(modality) != MODALITY_SHA256:
        raise RuntimeError("modality configuration hash mismatch")
    if sha256_file(config_path) != bundle["training_config"]["sha256"]:
        raise RuntimeError("training config hash mismatch")
    stats_document = read_json(stats_artifact)
    if sha256_file(stats_artifact) != bundle["structured_stats"]["sha256"]:
        raise RuntimeError("B0 statistics artifact hash mismatch")
    if stats_document.get("statistics_payload_sha256") != STATS_PAYLOAD_SHA256:
        raise RuntimeError("B0 statistics payload mismatch")
    if stats_document.get("dataset", {}).get("tree_sha256") != FULL24_TREE_SHA256:
        raise RuntimeError("B0 statistics dataset binding mismatch")

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
                  "cosmos": str(cosmos), "modality": str(modality), "stats_artifact": str(stats_artifact)},
        "base_tree_sha256": BASE_TREE_SHA256, "full24_tree_sha256": FULL24_TREE_SHA256,
        "cosmos_tree_sha256": cosmos_tree, "cosmos_entry_count": cosmos_count,
        "groot_sha": groot_head, "statistics_payload_sha256": STATS_PAYLOAD_SHA256,
        "model_constructed": False, "gr00t_forwards": 0, "wbc_steps": 0, "reference_updates": 0,
        "validation_e_read": False, "holdout_f_read": False, "training_steps": 0, "optimizer_steps": 0,
        "status": "PASS",
    })
    os.environ.update({"TMPDIR": str(tmp_root), "TMP": str(tmp_root), "TEMP": str(tmp_root),
                       "XDG_CACHE_HOME": str(cache_root / "xdg"), "HF_HOME": str(cache_root / "huggingface"),
                       "TRANSFORMERS_CACHE": str(hub_root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "GROOT_COMMIT_HASH": GROOT_SHA, "PYTHONNOUSERSITE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1"})
    sys.path.insert(0, str(groot))
    _bind_cosmos_processor(cosmos / "snapshots" / COSMOS_REVISION)
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
    install_structured_dataset_factory()
    from gr00t.experiment.experiment import run as gr00t_run
    config = _load_config(config_path, output_root=output_root, base_model=base_model, dataset=dataset)
    with (output_root / "stage45b_events.jsonl").open("w", encoding="utf-8") as events:
        events.write(json.dumps({"event": "MODEL_CONSTRUCTION_ALLOWED", "timestamp_utc": now_utc()}) + "\n")
        events.write(json.dumps({"event": "TRAINING_START", "timestamp_utc": now_utc()}) + "\n")
    gr00t_run(config)
    with (output_root / "stage45b_events.jsonl").open("a", encoding="utf-8") as events:
        events.write(json.dumps({"event": "TRAINING_RETURNED", "timestamp_utc": now_utc()}) + "\n")

    checkpoint_1000 = _checkpoint_record(output_root / "checkpoint-1000", 1000)
    checkpoint_2000 = _checkpoint_record(output_root / "checkpoint-2000", 2000)
    sidecar = read_json(output_root / "experiment_cfg" / "structured_action_statistics.json")
    if canonical_sha(sidecar.get("statistics", {})) != STATS_PAYLOAD_SHA256:
        raise RuntimeError("runtime StructuredAction statistics differ from B0")
    result = {
        "schema": "CORELAB_STAGE45B_FULL24_FRESH_BASE_TRAINING_RESULT_V1",
        "status": EXPECTED_TRAINING_TOKEN if checkpoint_1000["status"] == "PASS" and checkpoint_2000["status"] == "PASS" else "HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_INCOMPLETE",
        "pi_gate": PI_GATE, "stage": STAGE, "intervention": "B_FRESH_BASE_FULL24",
        "repository_head": git("rev-parse", "HEAD"), "pbs_jobid": gpu["pbs_jobid"], "gpu": gpu,
        "checkpoint_1000": checkpoint_1000, "checkpoint_2000": checkpoint_2000,
        "structured_stats_payload_sha256": STATS_PAYLOAD_SHA256, "training_steps": 2000,
        "optimizer_steps": 2000, "backward_steps": 2000, "gr00t_forwards": 0, "wbc_steps": 0,
        "reference_updates": 0, "validation_e_read": False, "holdout_f_read": False,
        "model_constructed": True, "training": True,
        "next_gate": "CORELAB_STAGE45B_FULL24_TRAIN_ONLY_EVALUATION_COMPLETE", "timestamp_utc": now_utc(),
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
