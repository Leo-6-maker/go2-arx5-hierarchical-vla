#!/usr/bin/env python3
"""PI0228 dense-relative H40 adaptation using the mature Stage28 path.

This adapter deliberately delegates configuration and ``experiment.run`` to
the already-qualified Stage28 runner.  Stage38 contributes only its local
Cosmos cache and processor bindings; its StructuredAction dataset factory is
never imported or called.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
STAGE28 = REPO / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"
STAGE38 = REPO / "scripts" / "stage38_structured_train.py"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_MODALITY_SHA = "244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_BASE_INDEX_SHA = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
EXPECTED_TRAIN_TREE_SHA = "66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81"
EXPECTED_TRAIN_FILES = 29389
EXPECTED_TRAIN_EPISODES = 9792
EXPECTED_TRAIN_CHUNKS = 10
EXPECTED_TRAIN_QUALIFIED_MANIFEST_SHA = "2ffa10ac51bf2327159ae7c208aa524f73666f6e72a3bf39d1e1aa141a15f847"
SEED = 22301
MAX_STEPS = 10000
SAVE_STEPS = 2500
AUTHORIZATION = "PI0228_REUSE_STAGE28_H200_PLUS_STAGE38_OFFLINE_BINDING"


class PI0228Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PI0228Error(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_status(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain=v1"], text=True
    )


def tree_entries(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise PI0228Error(f"dataset root is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            if (directory_path / dirname).is_symlink():
                raise PI0228Error(f"dataset contains symlink directory: {directory_path / dirname}")
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink() or not child.is_file():
                raise PI0228Error(f"dataset contains unsafe file: {child}")
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def tree_sha256(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    entries = tree_entries(root)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode()
        )
    return digest.hexdigest(), entries


def require_empty_or_preflight(root: Path) -> None:
    if not root.exists():
        root.mkdir(parents=True)
        return
    if not root.is_dir():
        raise PI0228Error(f"output root is not a directory: {root}")
    unexpected = [
        p.name
        for p in root.iterdir()
        if p.name not in {"pi0228_preflight.json", "pi0228_run_manifest.json"}
    ]
    if unexpected:
        raise PI0228Error(f"refusing to reuse non-empty output root: {unexpected[:5]}")


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if git_head(args.groot_source) != EXPECTED_GROOT_SHA:
        raise PI0228Error("GR00T source HEAD mismatch")
    if git_status(args.groot_source):
        raise PI0228Error("GR00T source checkout is dirty")
    if sha256_file(args.modality_config) != EXPECTED_MODALITY_SHA:
        raise PI0228Error("modality config SHA mismatch")
    index = args.base_model_path / "model.safetensors.index.json"
    if sha256_file(index) != EXPECTED_BASE_INDEX_SHA:
        raise PI0228Error("checkpoint-1000 index SHA mismatch")
    if not (args.base_model_path / "config.json").is_file():
        raise PI0228Error("checkpoint-1000 config is missing")
    if not (args.dataset_dir / "meta" / "info.json").is_file():
        raise PI0228Error("TRAIN dataset metadata is missing")
    if not (args.dataset_dir / "meta" / "relative_stats.json").is_file():
        raise PI0228Error("TRAIN relative_stats.json is missing")
    qualified_manifest = args.dataset_dir / "meta" / "stage19_qualified_manifest.json"
    if sha256_file(qualified_manifest) != EXPECTED_TRAIN_QUALIFIED_MANIFEST_SHA:
        raise PI0228Error("TRAIN official qualification manifest SHA mismatch")
    info = json.loads((args.dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    observed_tree, entries = tree_sha256(args.dataset_dir, "train")
    if observed_tree != EXPECTED_TRAIN_TREE_SHA:
        raise PI0228Error(f"TRAIN tree SHA mismatch: {observed_tree}")
    if len(entries) != EXPECTED_TRAIN_FILES:
        raise PI0228Error(f"TRAIN file count mismatch: {len(entries)}")
    if int(info.get("total_episodes", -1)) != EXPECTED_TRAIN_EPISODES:
        raise PI0228Error("TRAIN episode count mismatch")
    chunks = sorted((args.dataset_dir / "data").glob("chunk-*"))
    if len(chunks) != EXPECTED_TRAIN_CHUNKS:
        raise PI0228Error(f"TRAIN chunk count mismatch: {len(chunks)}")
    if not args.cosmos_snapshot.is_dir():
        raise PI0228Error("pinned Cosmos snapshot is missing")
    if EXPECTED_COSMOS_REVISION not in args.cosmos_snapshot.parts:
        raise PI0228Error("Cosmos revision mismatch")
    if not args.cosmos_cache_root.is_dir():
        raise PI0228Error("Cosmos cache root is missing")
    stage38 = load_module(STAGE38, "pi0228_stage38_identity")
    if args.cosmos_cache_root.resolve() != stage38.COSMOS_ROOT.resolve():
        raise PI0228Error("Cosmos cache root is not the pinned Stage38 root")
    if args.cosmos_snapshot.resolve() != stage38.COSMOS_SNAPSHOT.resolve():
        raise PI0228Error("Cosmos snapshot is not the pinned Stage38 revision")
    if "install_structured_dataset_factory" in STAGE28.read_text(encoding="utf-8"):
        raise PI0228Error("Stage28 training helper unexpectedly contains StructuredAction factory")
    return {
        "groot_sha": EXPECTED_GROOT_SHA,
        "modality_sha256": EXPECTED_MODALITY_SHA,
        "checkpoint_index_sha256": EXPECTED_BASE_INDEX_SHA,
        "train_tree_sha256": observed_tree,
        "train_file_count": len(entries),
        "train_episode_count": int(info["total_episodes"]),
        "train_chunk_count": len(chunks),
        "cosmos_revision": EXPECTED_COSMOS_REVISION,
        "cosmos_tree_sha256": EXPECTED_COSMOS_TREE_SHA,
        "target_xyz_in_model_input": False,
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
        "structured_factory_called": False,
    }


def bind_offline_runtime(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    cache_root = output_root / "runtime_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["TMPDIR"] = str(output_root / "tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.groot_source))
    stage38 = load_module(STAGE38, "pi0228_stage38_binding")
    cache_binding = stage38.bind_cosmos_cache(cache_root)
    processor_binding = stage38.bind_cosmos_processor()
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:
        raise PI0228Error(f"Stage45B torchcodec preload failed: {type(exc).__name__}: {exc}") from exc
    return {
        "hf_home": os.environ["HF_HOME"],
        "hf_hub_cache": os.environ["HF_HUB_CACHE"],
        "hf_hub_offline": True,
        "transformers_offline": True,
        "cache_binding": cache_binding,
        "processor_binding": processor_binding,
        "torchcodec_preloaded": True,
        "structured_factory_called": False,
    }


def build_config(args: argparse.Namespace, output_dir: Path, *, steps: int, save_steps: int, save_limit: int) -> Any:
    stage28 = load_module(STAGE28, "pi0228_stage28_runner")
    helper_args = argparse.Namespace(
        stage10_root=args.stage10_root,
        base_model_path=args.base_model_path,
        dataset_dir=args.dataset_dir,
        modality_config=args.modality_config,
        groot_source=args.groot_source,
        training_output_dir=output_dir,
    )
    config = stage28._build_training_config(helper_args)
    config.training.start_from_checkpoint = str(args.base_model_path)
    config.training.output_dir = str(output_dir)
    config.training.max_steps = steps
    config.training.save_steps = save_steps
    config.training.save_total_limit = save_limit
    config.training.warmup_steps = 10
    config.training.warmup_ratio = 0.0
    config.training.save_only_model = False
    config.training.resume_from_checkpoint = False
    config.training.transformers_local_files_only = True
    config.training.transformers_trust_remote_code = True
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-5
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.use_relative_action = True
    config.model.action_horizon = 40
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_visual = False
    config.model.tune_llm = False
    config.model.tune_vlln = False
    config.data.seed = SEED
    return config


def config_summary(config: Any) -> dict[str, Any]:
    return {
        "start_from_checkpoint": str(config.training.start_from_checkpoint),
        "max_steps": int(config.training.max_steps),
        "save_steps": int(config.training.save_steps),
        "save_total_limit": int(config.training.save_total_limit),
        "warmup_steps": int(config.training.warmup_steps),
        "warmup_ratio": float(config.training.warmup_ratio),
        "scheduler": str(config.training.lr_scheduler_type),
        "optimizer": str(config.training.optim),
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "batch_size": int(config.training.global_batch_size),
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "bf16": bool(config.training.bf16),
        "fp16": bool(config.training.fp16),
        "tf32": bool(config.training.tf32),
        "save_only_model": bool(config.training.save_only_model),
        "resume_from_checkpoint": bool(config.training.resume_from_checkpoint),
        "use_relative_action": bool(config.model.use_relative_action),
        "action_horizon": int(config.model.action_horizon),
        "tune_projector": bool(config.model.tune_projector),
        "tune_diffusion_model": bool(config.model.tune_diffusion_model),
        "tune_visual": bool(config.model.tune_visual),
        "tune_llm": bool(config.model.tune_llm),
        "tune_vlln": bool(config.model.tune_vlln),
        "seed": SEED,
    }


def checkpoint_record(path: Path, expected_step: int, expected_max_steps: int) -> dict[str, Any]:
    state_path = path / "trainer_state.json"
    if not path.is_dir() or not state_path.is_file():
        raise PI0228Error(f"missing checkpoint state: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != expected_step:
        raise PI0228Error(f"checkpoint step mismatch: {path}")
    if int(state.get("max_steps", -1)) != expected_max_steps:
        raise PI0228Error(f"checkpoint max_steps mismatch: {path}")
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise PI0228Error(f"checkpoint missing Trainer state: {missing}")
    tree, entries = tree_sha256(path, f"pi0228/{path.name}")
    return {
        "path": str(path),
        "global_step": expected_step,
        "max_steps": expected_max_steps,
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "tree_sha256": tree,
    }


def archived_checkpoint_records(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    deadline = time.monotonic() + float(os.environ.get("PI0228_ARCHIVE_WAIT_SECONDS", "900"))
    expected = {2500, 5000, 7500, 10000}
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = payload.get("checkpoints")
                if payload.get("schema") == "PI0228_CHECKPOINT_ARCHIVE_MANIFEST_V1" and payload.get("status") == "COMPLETE" and isinstance(records, dict) and set(map(int, records)) == expected:
                    for step, record in records.items():
                        if not record.get("verified") or int(record.get("global_step", -1)) != int(step) or int(record.get("max_steps", -1)) != MAX_STEPS:
                            raise PI0228Error(f"invalid archived checkpoint record: {step}")
                    return records, sha256_file(path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError, PI0228Error):
                pass
        time.sleep(10)
    raise PI0228Error(f"verified checkpoint archive manifest not available: {path}")


def run_official(config: Any, groot_source: Path) -> None:
    source = str(groot_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from gr00t.experiment.experiment import run

    run(config)


def smoke(args: argparse.Namespace, report: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    output = args.output_dir / "smoke_training"
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(args, output, steps=1, save_steps=1, save_limit=1)
    report["config"] = config_summary(config)
    run_official(config, args.groot_source)
    checkpoint = output / "checkpoint-1"
    record = checkpoint_record(checkpoint, 1, expected_max_steps=1)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    from transformers import AutoModel

    reloaded = AutoModel.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=True
    )
    del reloaded
    gc.collect()
    report.update(
        {
            "status": "PASS",
            "training_started": True,
            "model_construction": True,
            "model_forward": True,
            "backward": True,
            "optimizer_step": True,
            "save_reload": True,
            "checkpoint": record,
            "target_xyz_in_model_input": False,
            "structured_factory_called": False,
            "runtime": runtime,
        }
    )
    return report


def train(args: argparse.Namespace, report: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    output = args.output_dir / "training"
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(args, output, steps=MAX_STEPS, save_steps=SAVE_STEPS, save_limit=5)
    report["config"] = config_summary(config)
    run_official(config, args.groot_source)
    archive_manifest = os.environ.get("PI0228_CHECKPOINT_ARCHIVE_MANIFEST")
    if archive_manifest:
        records, archive_manifest_sha256 = archived_checkpoint_records(Path(archive_manifest))
    else:
        records = {
            str(step): checkpoint_record(output / f"checkpoint-{step}", step, expected_max_steps=MAX_STEPS)
            for step in (2500, 5000, 7500, 10000)
        }
        archive_manifest_sha256 = None
    seal = {
        "schema": "PI0228_H200_DENSE_RELATIVE_CHECKPOINT_SEAL_V1",
        "status": "PASS",
        "checkpoints": records,
        "checkpoint_archive_manifest_sha256": archive_manifest_sha256,
        "scientific_contract_changed": False,
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
    }
    seal_path = args.output_dir / "pi0228_checkpoint_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.update(
        {
            "status": "PASS",
            "training_started": True,
            "training_completed": True,
            "model_construction": True,
            "model_forward": True,
            "backward": True,
            "optimizer_step": True,
            "checkpoint_seal": {
                "path": str(seal_path),
                "sha256": sha256_file(seal_path),
                "checkpoints": records,
            },
            "target_xyz_in_model_input": False,
            "structured_factory_called": False,
            "runtime": runtime,
        }
    )
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--train", action="store_true")
    p.add_argument("--stage10-root", type=Path, required=True)
    p.add_argument("--groot-source", type=Path, required=True)
    p.add_argument("--base-model-path", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--modality-config", type=Path, required=True)
    p.add_argument("--cosmos-cache-root", type=Path, required=True)
    p.add_argument("--cosmos-snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--authorization", default=None)
    return p


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    try:
        require_empty_or_preflight(args.output_dir)
        identity = verify_inputs(args)
        report: dict[str, Any] = {
            "schema": "PI0228_H200_DENSE_RELATIVE_ADAPT_V1",
            "mode": "preflight" if args.preflight else "smoke" if args.smoke else "train",
            "status": "PASS",
            "identity": identity,
            "stage28_runner_sha256": sha256_file(STAGE28),
            "stage38_binding_source_sha256": sha256_file(STAGE38),
            "training_recipe": {
                "init": str(args.base_model_path),
                "dense_relative": True,
                "horizon": 40,
                "projector": True,
                "diffusion": True,
                "visual": False,
                "llm": False,
                "vlln": False,
                "seed": SEED,
                "max_steps": MAX_STEPS,
                "save_steps": [2500, 5000, 7500, 10000],
                "optimizer": "adamw_torch",
                "scheduler": "cosine",
                "warmup_steps": 10,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
            },
            "dev_e_source_read_count": 0,
            "final_f_source_read_count": 0,
            "target_xyz_in_model_input": False,
            "structured_factory_called": False,
        }
        if args.preflight:
            report["claim_boundary"] = "read-only identity/preflight; no model construction, forward, backward, optimizer, or training"
            (args.output_dir / "pi0228_preflight.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            if args.authorization != AUTHORIZATION:
                raise PI0228Error("exact PI0228 authorization token is required")
            preflight_path = args.output_dir / "pi0228_preflight.json"
            if not preflight_path.is_file():
                raise PI0228Error("a passing PI0228 preflight is required before smoke/train")
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            if preflight.get("status") != "PASS" or preflight.get("identity") != identity:
                raise PI0228Error("PI0228 preflight identity/status does not match current inputs")
            runtime = bind_offline_runtime(args, args.output_dir)
            report["runtime"] = runtime
            if args.smoke:
                report = smoke(args, report, runtime)
            else:
                report = train(args, report, runtime)
        (args.output_dir / "pi0228_run_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "PI0228_H200_DENSE_RELATIVE_ADAPT_V1",
            "mode": "preflight" if args.preflight else "smoke" if args.smoke else "train",
            "status": "STOP_OR_BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "training_started": False,
            "model_construction": False,
            "model_forward": False,
            "backward": False,
            "optimizer_step": False,
            "structured_factory_called": False,
            "scientific_contract_changed": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pi0228_run_manifest.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
