#!/usr/bin/env python3
"""Run the pre-declared Stage22 R0/R1 adaptation comparison.

This wrapper delegates model construction and optimization to the existing
official Stage20S launcher.  R0 freezes the diffusion transformer; R1 enables
the officially exposed ``tune_diffusion_model`` knob while keeping the
language and visual backbones frozen.  Outputs are intentionally written to a
node-local directory and the wrapper records the command, resource gate, and
run manifests without copying model weights into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _recipe_config(name: str) -> dict[str, Any]:
    if name == "r0":
        return {
            "recipe_id": "R0",
            "tune_projector": True,
            "tune_diffusion_model": False,
            "tune_llm": False,
            "tune_visual": False,
            "interpretation": "Stage20S_control_projector_state_action_encoder_only",
        }
    if name == "r1":
        return {
            "recipe_id": "R1",
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_llm": False,
            "tune_visual": False,
            "interpretation": "action_generation_diffusion_module_tuning_hypothesis",
        }
    raise ValueError(f"unknown recipe: {name}")


def _run_recipe(args: argparse.Namespace, name: str, *, max_steps: int, label: str) -> dict[str, Any]:
    config = _recipe_config(name)
    output_dir = args.output_root / label
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "launcher.log"
    command = [
        sys.executable,
        str(args.launcher),
        "--stage10-root",
        str(args.stage10_root),
        "--base-model-path",
        str(args.base_model_path),
        "--dataset-dir",
        str(args.dataset_dir),
        "--modality-config",
        str(args.modality_config),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(int(max_steps)),
        "--save-steps",
        str(int(args.save_steps)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--seed",
        str(int(args.seed)),
        "--local-files-only",
    ]
    if config["tune_diffusion_model"]:
        command.append("--tune-diffusion-model")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    run_manifest = output_dir / "run_manifest.json"
    payload: dict[str, Any] = {
        **config,
        "label": label,
        "status": "PASS" if completed.returncode == 0 and run_manifest.is_file() else "FAILED",
        "returncode": int(completed.returncode),
        "max_steps": int(max_steps),
        "save_steps": int(args.save_steps),
        "command": command,
        "launcher_log": str(log_path),
        "run_manifest": str(run_manifest) if run_manifest.is_file() else None,
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if run_manifest.is_file():
        try:
            payload["launcher_manifest"] = json.loads(run_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload["manifest_parse_error"] = str(exc)
    (output_dir / "stage22_recipe_result.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, default=Path(__file__).with_name("stage20s_train_gr00t.py"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recipe", choices=("r0", "r1", "both"), default="both")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--r1-preflight-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2222)
    args = parser.parse_args()
    if args.max_steps < 1 or args.r1_preflight_steps < 1 or args.save_steps < 1:
        parser.error("step counts must be positive")
    if not (args.dataset_dir / "meta" / "info.json").is_file():
        parser.error(f"materialized dataset is missing: {args.dataset_dir}")
    split_lock_path = args.dataset_dir / "meta" / "stage22_split_lock.json"
    if not split_lock_path.is_file():
        parser.error(
            "training requires a Stage22 train-only dataset with meta/stage22_split_lock.json; "
            "the full fixture is not an allowed training input"
        )
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    expected_train = [
        "red_target_train_a",
        "red_target_train_b",
        "red_target_train_c",
        "red_target_train_d",
    ]
    if (
        split_lock.get("dataset_role") != "TRAIN"
        or split_lock.get("selected_split") != "train"
        or split_lock.get("selected_target_ids") != expected_train
        or split_lock.get("training_allowed") is not True
        or split_lock.get("final_test_in_training") is not False
    ):
        parser.error("dataset split lock is not the frozen train_a-d-only Stage22 split")
    if not (args.base_model_path / "config.json").is_file():
        parser.error(f"base model is missing: {args.base_model_path}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(f"output root is non-empty; choose a new node-local path: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    recipes = ("r0", "r1") if args.recipe == "both" else (args.recipe,)
    results: list[dict[str, Any]] = []
    r1_preflight_failed = False
    if "r1" in recipes:
        results.append(_run_recipe(args, "r1", max_steps=args.r1_preflight_steps, label="r1_preflight_1step"))
        if results[-1]["status"] != "PASS":
            r1_preflight_failed = True
    runnable_recipes = tuple(name for name in recipes if name != "r1" or not r1_preflight_failed)
    for name in runnable_recipes:
        results.append(_run_recipe(args, name, max_steps=args.max_steps, label=name))
    manifest = {
        "schema": "STAGE22_RECIPE_COMPARISON_V1",
        "status": "PASS" if not r1_preflight_failed and all(item["status"] == "PASS" for item in results) else "PARTIAL_OR_FAILED",
        "simulation_only": True,
        "hardware_control": False,
        "training_claim": "resource_and_profile_adaptation_experiment; not_task_success",
        "base_model_path": str(args.base_model_path),
        "dataset_dir": str(args.dataset_dir),
        "split_lock": str(split_lock_path),
        "split_lock_sha256": hashlib.sha256(split_lock_path.read_bytes()).hexdigest(),
        "training_target_ids": expected_train,
        "r1_preflight_required": True,
        "r1_preflight_steps": int(args.r1_preflight_steps),
        "r1_preflight_failed": r1_preflight_failed,
        "r1_full_run_skipped_after_preflight_failure": r1_preflight_failed,
        "results": results,
    }
    (args.output_root / "stage22_recipe_comparison.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0 if manifest["status"] == "PASS" else (2 if r1_preflight_failed else 1)


if __name__ == "__main__":
    raise SystemExit(main())
