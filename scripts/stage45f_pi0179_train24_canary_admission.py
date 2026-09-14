#!/usr/bin/env python3
"""PI0179 shared-RTX admission wrapper for the one TRAIN24 canary.

The wrapper is deliberately independent of Torch and the model runner.  It
records two read-only occupancy snapshots, accepts foreign processes without
interference, and launches the already-reviewed PI0177 canary exactly once.
No process is killed, signalled, reprioritized, or otherwise modified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PI_GATE = 5663251430
EXPECTED_GPU_NAME = "RTX 4080"
MIN_TOTAL_MIB = 30_000
MIN_FREE_MIB = 12_288
REQUIRED_SNAPSHOTS = 2
EXPECTED_COUNTS = {"model_loads": 1, "conditioning": 1, "action_head_calls": 4, "model_side_total": 5}


class AdmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AdmissionError(f"durable admission record already exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _gpu_snapshot() -> dict[str, Any]:
    query = "index,name,uuid,memory.total,memory.used,memory.free"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, f"nvidia-smi GPU query failed: {completed.stderr[-500:]}")
    gpus = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 6:
            continue
        gpus.append({
            "index": int(values[0]),
            "name": values[1],
            "uuid": values[2],
            "memory_total_mib": int(values[3]),
            "memory_used_mib": int(values[4]),
            "memory_free_mib": int(values[5]),
        })
    require(gpus, "nvidia-smi returned no GPUs")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    processes: list[dict[str, Any]] = []
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) == 4:
                processes.append({"gpu_uuid": values[0], "pid": int(values[1]), "process_name": values[2], "used_memory_mib": int(values[3])})
    return {
        "query": query,
        "gpus": gpus,
        "foreign_processes": processes,
        "foreign_processes_allowed": True,
        "foreign_process_interference": "NONE",
        "captured_by": "stage45f_pi0179_train24_canary_admission.py",
    }


def _selected(snapshot: Mapping[str, Any], gpu_index: int) -> Mapping[str, Any]:
    rows = [row for row in snapshot["gpus"] if int(row["index"]) == int(gpu_index)]
    require(len(rows) == 1, f"configured GPU index unavailable or ambiguous: {gpu_index}")
    return rows[0]


def validate_admission(snapshots: Sequence[Mapping[str, Any]], gpu_index: int) -> Mapping[str, Any]:
    require(len(snapshots) == REQUIRED_SNAPSHOTS, "exactly two occupancy snapshots are required")
    selected = [_selected(snapshot, gpu_index) for snapshot in snapshots]
    uuid = str(selected[0]["uuid"])
    require(all(str(row["uuid"]) == uuid for row in selected), "GPU UUID changed between admission snapshots")
    require(all(EXPECTED_GPU_NAME in str(row["name"]) for row in selected), "selected GPU is not an RTX 4080")
    require(all(int(row["memory_total_mib"]) >= MIN_TOTAL_MIB for row in selected), "RTX total VRAM is below 30 GiB admission floor")
    require(all(int(row["memory_free_mib"]) >= MIN_FREE_MIB for row in selected), "RTX free VRAM is below 12 GiB admission floor")
    return {
        "status": "PASS",
        "gpu_index": int(gpu_index),
        "gpu_uuid": uuid,
        "gpu_name": str(selected[0]["name"]),
        "minimum_total_mib": MIN_TOTAL_MIB,
        "minimum_free_mib": MIN_FREE_MIB,
        "observed_minimum_total_mib": min(int(row["memory_total_mib"]) for row in selected),
        "observed_minimum_free_mib": min(int(row["memory_free_mib"]) for row in selected),
        "foreign_process_policy": "allowed_read_only_no_kill_signal_reprioritize",
        "snapshot_count": len(snapshots),
    }


def plan() -> dict[str, Any]:
    return {
        "schema": "STAGE45F_PI0179_TRAIN24_CANARY_ADMISSION_PLAN_V1",
        "pi_gate": PI_GATE,
        "gpu_policy": {
            "same_uuid": True,
            "minimum_total_mib": MIN_TOTAL_MIB,
            "minimum_free_mib": MIN_FREE_MIB,
            "occupancy_snapshots": REQUIRED_SNAPSHOTS,
            "foreign_processes": "allowed",
            "process_interference": False,
        },
        "canary_ceiling": {**EXPECTED_COUNTS, "raw_outputs": 1, "validation_e_model_calls": 0},
        "execution": "one_runner_invocation_after_admission",
        "training": False,
        "wbc": False,
        "simulator": False,
        "hardware": False,
    }


def run_canary(config_path: Path, runner_script: Path, repo: Path, python_bin: str, interval_s: float) -> dict[str, Any]:
    config = _json(config_path)
    output_root = Path(str(config["output_root"])).resolve()
    require(output_root.is_dir(), f"CPU/static output root is unavailable: {output_root}")
    admission_path = output_root / "resource_admission.json"
    require(not admission_path.exists(), "resource admission already recorded; no repeat")
    gpu_index = int(config.get("gpu_index", 0))
    snapshots = [_gpu_snapshot()]
    if interval_s > 0:
        time.sleep(interval_s)
    snapshots.append(_gpu_snapshot())
    admission = validate_admission(snapshots, gpu_index)
    admission_record = {
        "schema": "STAGE45F_PI0179_TRAIN24_CANARY_RESOURCE_ADMISSION_V1",
        "status": admission["status"],
        "pi_gate": PI_GATE,
        "captured_before_model_load": True,
        "admission": dict(admission),
        "snapshots": snapshots,
        "model_loads_before_record": 0,
        "conditioning_before_record": 0,
        "action_head_before_record": 0,
        "validation_e_model_calls": 0,
        "training": False,
        "wbc": False,
        "simulator": False,
        "hardware": False,
    }
    _write_json(admission_path, admission_record)
    command = [
        python_bin,
        str(runner_script.resolve()),
        "--canary",
        "--repo",
        str(repo.resolve()),
        "--config",
        str(config_path.resolve()),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (output_root / "canary_wrapper.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_root / "canary_wrapper.stderr.log").write_text(completed.stderr, encoding="utf-8")
    result_path = output_root / "canary_result.json"
    require(completed.returncode == 0 and result_path.is_file(), f"PI0177 canary failed: {completed.stderr[-1200:]}")
    result = _json(result_path)
    require(result.get("status") == "PASS", "PI0177 canary did not report PASS")
    for key, value in EXPECTED_COUNTS.items():
        require(int(result.get(key, -1)) == value, f"canary count drift for {key}: {result.get(key)}")
    require(int(result.get("model_calls", -1)) == 5, "canary model call count drift")
    require(int(result.get("validation_e_model_calls", -1)) == 0, "Validation-E model calls were not zero")
    require(result.get("training") is False and int(result.get("optimizer_steps", -1)) == 0, "canary entered training")
    require(result.get("holdout_f_read") is False, "canary read Holdout-F")
    selected_after = next(row for row in result["gpu_after"]["gpus"] if int(row["index"]) == gpu_index)
    require(str(selected_after["uuid"]) == str(admission["gpu_uuid"]), "GPU UUID changed during canary")
    wrapper_result = {
        "schema": "STAGE45F_PI0179_TRAIN24_CANARY_WRAPPER_RESULT_V1",
        "status": "PASS",
        "pi_gate": PI_GATE,
        "resource_admission": admission,
        "runner_result": "canary_result.json",
        "runner_command": command,
        "runner_status": result,
        "foreign_process_interference": False,
        "validation_e_model_calls": 0,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "wbc": False,
        "simulator": False,
        "hardware": False,
    }
    _write_json(output_root / "stage45f_pi0179_train24_canary_result.json", wrapper_result)
    return wrapper_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runner-script", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--snapshot-interval-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    require(args.config and args.runner_script and args.repo, "--run requires --config, --runner-script, and --repo")
    require(math.isfinite(args.snapshot_interval_s) and args.snapshot_interval_s >= 0.0, "snapshot interval must be finite and non-negative")
    print(json.dumps(run_canary(args.config.resolve(strict=True), args.runner_script.resolve(strict=True), args.repo.resolve(strict=True), args.python_bin, args.snapshot_interval_s), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
