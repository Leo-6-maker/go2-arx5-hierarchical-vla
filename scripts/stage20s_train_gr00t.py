#!/usr/bin/env python3
"""Run a resource-bounded, simulation-only official N1.7 adaptation.

The launcher uses the official Isaac-GR00T ``experiment.run`` path but keeps
the large backbone and diffusion transformer frozen by default.  Only the
action-head projector/encoders are adapted, which is the largest adaptation
that can be tested on the currently available 16 GB node without pretending
that it is the full-memory official recipe.  No hardware, transport, WBC, or
real data path is imported here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_modality(path: Path, stage10_root: Path) -> None:
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(path.parent))
    importlib.import_module(path.stem)


def _build_config(args: argparse.Namespace):
    stage10_root = args.stage10_root
    _load_modality(args.modality_config, stage10_root)
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_dir)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Explicit Stage20S simulation adaptation policy.  The official launcher
    # exposes the first four flags but not tune_vlln/load_bf16, so this small
    # project-owned wrapper records them rather than editing shared GR00T.
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = bool(args.tune_diffusion_model)
    config.model.tune_vlln = False
    config.model.load_bf16 = True
    config.model.backbone_trainable_params_fp32 = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.use_relative_action = True
    config.model.action_horizon = 40
    config.model.state_dropout_prob = 0.0
    config.model.color_jitter_params = None
    config.model.random_rotation_angle = None
    config.model.image_crop_size = None
    config.model.image_target_size = None
    config.model.shortest_image_edge = 96
    config.model.crop_fraction = 1.0
    config.model.use_albumentations_transforms = True

    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(args.base_model_path)
    config.training.transformers_local_files_only = bool(args.local_files_only)
    config.training.transformers_trust_remote_code = True
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = 1
    config.training.per_gpu_batch_size = None
    config.training.gradient_accumulation_steps = 1
    config.training.dataloader_num_workers = 0
    config.training.learning_rate = float(args.learning_rate)
    config.training.output_dir = str(args.output_dir)
    config.training.save_steps = int(args.save_steps)
    config.training.save_total_limit = 2
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.max_steps = int(args.max_steps)
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.training.gradient_checkpointing = False
    config.training.logging_steps = 1
    config.training.shard_size = 256
    config.training.episode_sampling_rate = 1.0
    config.training.num_shards_per_epoch = 1
    config.training.save_only_model = True
    config.training.resume_from_checkpoint = False
    config.training.skip_weight_loading = False
    config.data.seed = int(args.seed)
    return config


def _cuda_telemetry(*, before: dict | None, run_elapsed_s: float | None) -> dict:
    """Capture resource telemetry without changing the training recipe."""

    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_import_failed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda_unavailable"}
    device = torch.cuda.current_device()
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        pass
    allocated_before_cleanup = int(torch.cuda.memory_allocated(device))
    reserved_before_cleanup = int(torch.cuda.memory_reserved(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        pass
    return {
        "available": True,
        "device_logical_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "before": before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocated_before_cleanup_bytes": allocated_before_cleanup,
        "reserved_before_cleanup_bytes": reserved_before_cleanup,
        "allocated_after_cleanup_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_after_cleanup_bytes": int(torch.cuda.memory_reserved(device)),
        "run_elapsed_s": run_elapsed_s,
    }


def _write_manifest(
    args: argparse.Namespace,
    *,
    status: str,
    error: str | None = None,
    cuda_telemetry: dict | None = None,
) -> None:
    output = args.output_dir
    files = {}
    if output.exists():
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"run_manifest.json"}:
                files[str(path.relative_to(output))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path) if path.stat().st_size < 32 * 1024 * 1024 else None,
                }
    payload = {
        "schema": "STAGE20S_GR00T_TRAINING_RUN_V1",
        "status": status,
        "simulation_only": True,
        "hardware_control": False,
        "dataset_dir": str(args.dataset_dir),
        "base_model_path": str(args.base_model_path),
        "base_model_index_sha256": _sha256(args.base_model_path / "model.safetensors.index.json")
        if (args.base_model_path / "model.safetensors.index.json").is_file()
        else None,
        "output_dir": str(output),
        "max_steps": int(args.max_steps),
        "save_steps": int(args.save_steps),
        "tune_projector": True,
        "tune_diffusion_model": bool(args.tune_diffusion_model),
        "tune_vlln": False,
        "tune_llm": False,
        "tune_visual": False,
        "load_bf16": True,
        "memory_policy": "RESOURCE_BOUNDED_PROJECTOR_ADAPTATION_NOT_FULL_ACTION_HEAD_RECIPE",
        "action_horizon": 40,
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
        "modality_config": str(args.modality_config),
        "stage10_root": str(args.stage10_root),
        "groot_repo_sha": subprocess.check_output(
            ["git", "-C", str(args.stage10_root / "src" / "isaac-gr00t"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cuda_telemetry": cuda_telemetry,
        "artifacts": files,
        "error": error,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--tune-diffusion-model", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.save_steps < 1:
        parser.error("--max-steps and --save-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is non-empty; choose a new node-local path: {args.output_dir}")
    if not (args.dataset_dir / "meta" / "info.json").is_file():
        parser.error(f"materialized dataset is missing: {args.dataset_dir}")
    if not (args.base_model_path / "config.json").is_file():
        parser.error(f"base model is missing: {args.base_model_path}")

    cuda_before = None
    run_start = time.perf_counter()
    try:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                cuda_before = {
                    "device_logical_index": int(device),
                    "device_name": torch.cuda.get_device_name(device),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                }
                torch.cuda.reset_peak_memory_stats(device)
        except ImportError:
            cuda_before = {"available": False, "reason": "torch_import_failed"}
        config = _build_config(args)
        from gr00t.experiment.experiment import run

        run(config)
    except Exception as exc:
        _write_manifest(
            args,
            status="FAILED",
            error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            cuda_telemetry=_cuda_telemetry(
                before=cuda_before, run_elapsed_s=time.perf_counter() - run_start
            ),
        )
        raise
    _write_manifest(
        args,
        status="PASS",
        cuda_telemetry=_cuda_telemetry(
            before=cuda_before, run_elapsed_s=time.perf_counter() - run_start
        ),
    )
    print(json.dumps(json.loads((args.output_dir / "run_manifest.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
