#!/usr/bin/env python3
"""A40 phase telemetry and fail-closed Stage40 result sealing.

The collector is safe to call from a PBS job at each runtime boundary.  It
records observed values and never infers capacity from a missing GPU query.
The seal path validates the full result schema before writing any seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from vanda_stage40_admission import (
        validate_stage40_receding_horizon_result_schema,
        validate_stage40_result_schema,
    )
except ModuleNotFoundError:  # package import from the repository root
    from scripts.vanda_stage40_admission import (
        validate_stage40_receding_horizon_result_schema,
        validate_stage40_result_schema,
    )


GpuRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _host_memory_mib() -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "host_mem_total_mib": None,
        "host_mem_available_mib": None,
        "host_mem_used_mib": None,
    }
    try:
        fields: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition(":")
            if separator and raw.strip().endswith(" kB"):
                fields[key] = float(raw.strip()[:-3].strip()) / 1024.0
        values["host_mem_total_mib"] = fields.get("MemTotal")
        values["host_mem_available_mib"] = fields.get("MemAvailable")
        if values["host_mem_total_mib"] is not None and values["host_mem_available_mib"] is not None:
            values["host_mem_used_mib"] = (
                values["host_mem_total_mib"] - values["host_mem_available_mib"]
            )
    except (OSError, ValueError):
        pass
    return values


def _gpu_snapshot(
    *,
    command_runner: GpuRunner | None = None,
    gpu_index: str | None = None,
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    runner = subprocess.run if command_runner is None else command_runner
    try:
        completed = runner(command, check=False, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        return {
            "status": "NOT_AVAILABLE",
            "error": f"{type(exc).__name__}: {exc}",
            "query": command,
            "rows": [],
        }
    if completed.returncode != 0:
        return {
            "status": "QUERY_FAILED",
            "returncode": int(completed.returncode),
            "stderr": completed.stderr.strip(),
            "query": command,
            "rows": [],
        }
    rows: list[dict[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        fields = [item.strip() for item in raw_line.split(",")]
        if len(fields) != 7:
            continue
        row = {
            "index": fields[0],
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": fields[3],
            "memory_used_mib": fields[4],
            "driver_version": fields[5],
            "compute_capability": fields[6],
        }
        if gpu_index is None or row["index"] == str(gpu_index):
            rows.append(row)
    return {"status": "PASS", "query": command, "rows": rows}


def collect_phase(
    phase: str,
    output_dir: Path,
    *,
    job_id: str | None = None,
    gpu_index: str | None = None,
    runtime_root: str | None = None,
    cache_root: str | None = None,
    command_runner: GpuRunner | None = None,
) -> dict[str, Any]:
    """Append one observed phase row to the stage40_telemetry.jsonl file."""

    phase_name = str(phase).strip()
    if not phase_name:
        raise ValueError("telemetry phase must be non-empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "schema": "VANDA_STAGE40_A40_TELEMETRY_V1",
        "timestamp_utc": _utc_now(),
        "phase": phase_name,
        "job_id": job_id or os.environ.get("PBS_JOBID"),
        "host": platform.node(),
        "pid": os.getpid(),
        "gpu_index_requested": gpu_index,
        "runtime_root": runtime_root,
        "cache_root": cache_root,
        "gpu": _gpu_snapshot(command_runner=command_runner, gpu_index=gpu_index),
        "host_memory": _host_memory_mib(),
    }
    path = output / "stage40_telemetry.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return row


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def seal_result_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate and atomically-ish write a canonical Stage40 result seal."""

    source = Path(input_path)
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("schema") == "VANDA_STAGE40_V6_RESULT_V1":
        validate_stage40_receding_horizon_result_schema(result)
    else:
        validate_stage40_result_schema(result)
    sealed = {
        "schema": "VANDA_STAGE40_RESULT_SEAL_V1",
        "result": result,
    }
    payload = canonical_json_bytes(sealed)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    digest_path = Path(str(destination) + ".sha256")
    digest_path.write_text(digest + "  " + destination.name + "\n", encoding="utf-8")
    return {
        "schema": "VANDA_STAGE40_RESULT_SEAL_V1",
        "seal_path": str(destination),
        "sha256_path": str(digest_path),
        "sha256": digest,
        "result_schema_valid": True,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu-index")
    parser.add_argument("--runtime-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--seal-result", type=Path)
    parser.add_argument("--seal-output", type=Path)
    args = parser.parse_args(argv)
    if args.seal_result is not None:
        if args.seal_output is None:
            parser.error("--seal-output is required with --seal-result")
        print(json.dumps(seal_result_file(args.seal_result, args.seal_output), sort_keys=True))
        return 0
    if args.phase is None or args.output_dir is None:
        parser.error("--phase and --output-dir are required for telemetry collection")
    print(json.dumps(collect_phase(
        args.phase,
        args.output_dir,
        job_id=args.job_id,
        gpu_index=args.gpu_index,
        runtime_root=args.runtime_root,
        cache_root=args.cache_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
