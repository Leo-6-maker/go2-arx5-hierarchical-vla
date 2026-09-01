#!/usr/bin/env python3
"""Run the gated one-step Stage22 R1 GPU qualification.

This is a resource gate, not a training run.  It invokes the existing
official Stage20S launcher for exactly one optimizer step with the declared
R1 recipe, while recording device occupancy, conservative device-level peak
usage, PyTorch allocated/reserved telemetry, trainable parameters, elapsed
time, cleanup, and headroom.  It refuses to start when a compute workload is
already present on the selected GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parse_gpu_snapshot(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(float(fields[3])),
                    "memory_used_mib": int(float(fields[4])),
                    "utilization_gpu_percent": int(float(fields[5])),
                }
            )
        except ValueError:
            continue
    return rows


def _gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {completed.stderr.strip()}")
    rows = _parse_gpu_snapshot(completed.stdout)
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return rows


def _compute_apps(gpu_index: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "process_name": fields[1],
                    "used_memory_mib": int(float(fields[2])),
                }
            )
        except ValueError:
            continue
    return rows


def _select_gpu(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    matches = [row for row in rows if row["index"] == index]
    if len(matches) != 1:
        raise RuntimeError(f"requested physical GPU index {index} was not found")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_trainable_parameters(log_text: str) -> int | None:
    matches = re.findall(r"Trainable parameters:\s*([0-9][0-9,]*)", log_text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def _parse_train_runtime(log_text: str) -> float | None:
    matches = re.findall(r"train_runtime['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", log_text)
    if not matches:
        return None
    return float(matches[-1])


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is non-empty: {args.output_dir}")
    if not (args.dataset_dir / "meta" / "info.json").is_file():
        raise RuntimeError(f"materialized dataset is missing: {args.dataset_dir}")
    split_lock = args.dataset_dir / "meta" / "stage22_split_lock.json"
    if not split_lock.is_file():
        raise RuntimeError("R1 preflight requires the locked train-only dataset")
    split_payload = _read_json(split_lock)
    if not split_payload or split_payload.get("dataset_role") != "TRAIN":
        raise RuntimeError("R1 preflight requires dataset_role=TRAIN")
    before_rows = _gpu_snapshot()
    before = _select_gpu(before_rows, args.gpu_index)
    existing_apps = _compute_apps(args.gpu_index)
    if existing_apps:
        payload = {
            "schema": "STAGE22_R1_GPU_PREFLIGHT_V1",
            "status": "BLOCKED_OCCUPIED_GPU",
            "reason": "compute workload was present before launch",
            "gpu": before,
            "existing_compute_apps": existing_apps,
            "hardware_control": False,
            "training_steps_requested": 1,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "stage22_r1_preflight.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher_output = args.output_dir / "r1_one_step"
    log_path = args.output_dir / "launcher.log"
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
        str(launcher_output),
        "--max-steps",
        "1",
        "--save-steps",
        "1",
        "--learning-rate",
        str(float(args.learning_rate)),
        "--seed",
        str(int(args.seed)),
        "--tune-diffusion-model",
        "--local-files-only",
    ]
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    started = time.perf_counter()
    samples: list[dict[str, Any]] = [before]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=run_env,
        )
        while process.poll() is None:
            time.sleep(0.2)
            try:
                samples.append(_select_gpu(_gpu_snapshot(), args.gpu_index))
            except RuntimeError:
                pass
        returncode = int(process.wait())
    elapsed_s = time.perf_counter() - started
    try:
        samples.append(_select_gpu(_gpu_snapshot(), args.gpu_index))
    except RuntimeError:
        pass
    after_rows = _gpu_snapshot()
    after = _select_gpu(after_rows, args.gpu_index)
    launcher_manifest = _read_json(launcher_output / "run_manifest.json")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    telemetry = (launcher_manifest or {}).get("cuda_telemetry")
    total_mib = int(before["memory_total_mib"])
    peak_used_mib = max(int(row["memory_used_mib"]) for row in samples)
    required_headroom_mib = max(0.15 * total_mib, 2048.0)
    headroom_mib = float(total_mib - peak_used_mib)
    payload: dict[str, Any] = {
        "schema": "STAGE22_R1_GPU_PREFLIGHT_V1",
        "status": "PASS"
        if returncode == 0 and launcher_manifest and launcher_manifest.get("status") == "PASS"
        else "FAILED",
        "simulation_only": True,
        "hardware_control": False,
        "training_steps_requested": 1,
        "optimizer_step_required": True,
        "cleanup_required": True,
        "gpu": {
            "physical_index": int(args.gpu_index),
            "uuid": before["uuid"],
            "name": before["name"],
            "total_memory_mib": total_mib,
            "baseline_used_mib": int(before["memory_used_mib"]),
            "peak_device_used_mib": peak_used_mib,
            "post_run_used_mib": int(after["memory_used_mib"]),
            "residual_project_allocation_mib": max(
                0, int(after["memory_used_mib"]) - int(before["memory_used_mib"])
            ),
            "utilization_samples": [int(row["utilization_gpu_percent"]) for row in samples],
        },
        "headroom": {
            "mib": headroom_mib,
            "required_mib": required_headroom_mib,
            "criterion": "at_least_max_15_percent_total_or_2GiB",
            "qualified": headroom_mib >= required_headroom_mib,
        },
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "allocated_after_cleanup_mib": None,
        "reserved_after_cleanup_mib": None,
        "trainable_parameter_count": _parse_trainable_parameters(log_text),
        "batch_size": 1,
        "step_time_s": _parse_train_runtime(log_text),
        "one_step_process_elapsed_s": elapsed_s,
        "returncode": returncode,
        "launcher_command": command,
        "launcher_log": str(log_path),
        "launcher_manifest": str(launcher_output / "run_manifest.json")
        if launcher_manifest
        else None,
        "cuda_telemetry": telemetry,
        "preflight_existing_compute_apps": existing_apps,
        "post_run_compute_apps": _compute_apps(args.gpu_index),
        "runtime": {
            "python": platform.python_version(),
            "cuda_visible_devices": run_env.get("CUDA_VISIBLE_DEVICES"),
            "stage10_root": str(args.stage10_root),
        },
    }
    if isinstance(telemetry, dict):
        for source_key, output_key in (
            ("peak_allocated_bytes", "peak_allocated_mib"),
            ("peak_reserved_bytes", "peak_reserved_mib"),
            ("allocated_after_cleanup_bytes", "allocated_after_cleanup_mib"),
            ("reserved_after_cleanup_bytes", "reserved_after_cleanup_mib"),
        ):
            if telemetry.get(source_key) is not None:
                payload[output_key] = float(telemetry[source_key]) / (1024.0 * 1024.0)
    payload["status"] = (
        "PASS"
        if payload["status"] == "PASS" and payload["headroom"]["qualified"]
        else ("INSUFFICIENT_HEADROOM" if payload["status"] == "PASS" else payload["status"])
    )
    (args.output_dir / "stage22_r1_preflight.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, default=Path(__file__).with_name("stage20s_train_gr00t.py"))
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2222)
    args = parser.parse_args()
    try:
        payload = _run(args)
    except Exception as exc:
        payload = {
            "schema": "STAGE22_R1_GPU_PREFLIGHT_V1",
            "status": "FAILED_PREFLIGHT_SETUP",
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            "hardware_control": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "stage22_r1_preflight.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
