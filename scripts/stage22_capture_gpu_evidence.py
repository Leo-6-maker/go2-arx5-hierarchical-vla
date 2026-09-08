#!/usr/bin/env python3
"""Capture an artifact-backed, read-only GPU identity observation.

This helper only queries ``nvidia-smi``.  It does not start training, attach to
another process, change scheduling, or inspect credentials.  The resulting
JSON is intended to be hashed and referenced by the Stage22 H2 execution
provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GPU_EVIDENCE_SCHEMA = "STAGE22_GPU_EVIDENCE_V1"
GPU_QUERY = "index,name,uuid,memory.total"
GPU_COMMAND = [
    "nvidia-smi",
    f"--query-gpu={GPU_QUERY}",
    "--format=csv,noheader,nounits",
]
_DEVICE_FIELDS = ("index", "name", "uuid", "memory_total_mib")
_CAPTURE_KINDS = {"UNSCOPED_READ_ONLY", "PBS_PROSPECTIVE_H2"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GPU evidence {field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"GPU evidence {field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"GPU evidence {field} must include a timezone")
    return parsed


def _parse_devices(stdout: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            devices.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(float(fields[3])),
                }
            )
        except (TypeError, ValueError):
            continue
    return devices


def validate_gpu_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the non-secret fields of one captured GPU evidence artifact."""

    if payload.get("schema") != GPU_EVIDENCE_SCHEMA:
        raise ValueError("unexpected GPU evidence schema")
    if payload.get("status") != "PASS":
        raise ValueError("GPU evidence status is not PASS")
    if payload.get("source") != "nvidia-smi":
        raise ValueError("GPU evidence source is not nvidia-smi")
    if payload.get("command") != GPU_COMMAND:
        raise ValueError("GPU evidence command is not the frozen nvidia-smi identity query")
    if not isinstance(payload.get("hostname"), str) or not payload["hostname"].strip():
        raise ValueError("GPU evidence hostname is missing")
    if not isinstance(payload.get("captured_utc"), str) or not payload["captured_utc"].strip():
        raise ValueError("GPU evidence timestamp is missing")
    captured = _parse_utc(payload["captured_utc"], field="timestamp")
    if payload.get("query") != GPU_QUERY:
        raise ValueError("GPU evidence query is not the frozen nvidia-smi identity query")
    if payload.get("contains_credentials") is not False:
        raise ValueError("GPU evidence credential flag is not false")
    raw_hash = payload.get("raw_stdout_sha256")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise ValueError("GPU evidence raw stdout hash is missing")
    devices = payload.get("devices")
    selected = payload.get("selected_device")
    if not isinstance(devices, list) or not devices or not isinstance(selected, Mapping):
        raise ValueError("GPU evidence devices or selected_device is missing")
    normalized_devices = []
    for device in devices:
        if not isinstance(device, Mapping) or any(key not in device for key in _DEVICE_FIELDS):
            raise ValueError("GPU evidence contains an incomplete device row")
        if not isinstance(device.get("index"), int) or isinstance(device.get("index"), bool):
            raise ValueError("GPU evidence device index is invalid")
        if not isinstance(device.get("name"), str) or not device["name"].strip():
            raise ValueError("GPU evidence device name is missing")
        if not isinstance(device.get("uuid"), str) or not device["uuid"].strip():
            raise ValueError("GPU evidence device UUID is missing")
        if not isinstance(device.get("memory_total_mib"), int) or device["memory_total_mib"] <= 0:
            raise ValueError("GPU evidence device VRAM is invalid")
        normalized_devices.append(dict(device))
    if len({device["index"] for device in normalized_devices}) != len(normalized_devices):
        raise ValueError("GPU evidence contains duplicate device indices")
    if len({device["uuid"] for device in normalized_devices}) != len(normalized_devices):
        raise ValueError("GPU evidence contains duplicate device UUIDs")
    if any(key not in selected for key in _DEVICE_FIELDS):
        raise ValueError("GPU evidence selected_device is incomplete")
    if dict(selected) not in normalized_devices:
        raise ValueError("GPU evidence selected_device is not a complete device row")
    matching = [device for device in devices if isinstance(device, Mapping) and device.get("uuid") == selected.get("uuid")]
    if len(matching) != 1 or dict(matching[0]) != dict(selected):
        raise ValueError("GPU evidence selected_device is not uniquely present in devices")
    context = payload.get("capture_context")
    if not isinstance(context, Mapping):
        raise ValueError("GPU evidence capture_context is missing")
    kind = context.get("kind")
    if kind not in _CAPTURE_KINDS:
        raise ValueError("GPU evidence capture_context kind is invalid")
    if kind == "PBS_PROSPECTIVE_H2":
        if not isinstance(context.get("pbs_job_id"), str) or not context["pbs_job_id"].strip():
            raise ValueError("PBS prospective GPU evidence job ID is missing")
        window_start = _parse_utc(context.get("window_start_utc"), field="window_start_utc")
        window_end = _parse_utc(context.get("window_end_utc"), field="window_end_utc")
        if window_end <= window_start:
            raise ValueError("PBS prospective GPU evidence window is invalid")
        if not window_start <= captured <= window_end:
            raise ValueError("GPU evidence timestamp is outside the PBS job window")
    return dict(payload)


def capture_gpu_evidence(
    output: Path,
    *,
    device_index: int | None = None,
    pbs_job_id: str | None = None,
    pbs_window_start_utc: str | None = None,
    pbs_window_end_utc: str | None = None,
) -> dict[str, Any]:
    """Run the frozen identity query and write one new evidence artifact."""

    output = output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite GPU evidence artifact: {output}")
    window_values = (pbs_job_id, pbs_window_start_utc, pbs_window_end_utc)
    if any(value is not None for value in window_values) and not all(
        value is not None for value in window_values
    ):
        raise ValueError("PBS prospective GPU evidence requires job ID and both window timestamps")
    completed = subprocess.run(GPU_COMMAND, capture_output=True, check=False)
    stdout = completed.stdout
    devices = _parse_devices(stdout.decode("utf-8", errors="replace"))
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi identity query failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    if device_index is None:
        if len(devices) != 1:
            raise ValueError("one visible GPU is required unless --device-index is supplied")
        selected = devices[0]
    else:
        selected_matches = [device for device in devices if device["index"] == int(device_index)]
        if len(selected_matches) != 1:
            raise ValueError(f"GPU index {device_index} is not uniquely visible")
        selected = selected_matches[0]
    captured_utc = datetime.now(timezone.utc).isoformat()
    if all(value is not None for value in window_values):
        capture_context = {
            "kind": "PBS_PROSPECTIVE_H2",
            "pbs_job_id": str(pbs_job_id),
            "window_start_utc": str(pbs_window_start_utc),
            "window_end_utc": str(pbs_window_end_utc),
        }
    else:
        capture_context = {"kind": "UNSCOPED_READ_ONLY"}
    payload = {
        "schema": GPU_EVIDENCE_SCHEMA,
        "status": "PASS",
        "source": "nvidia-smi",
        "command": list(GPU_COMMAND),
        "query": GPU_QUERY,
        "hostname": socket.gethostname(),
        "captured_utc": captured_utc,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
        "selected_device": selected,
        "capture_context": capture_context,
        "raw_stdout_sha256": _sha256_bytes(stdout),
        "contains_credentials": False,
    }
    validate_gpu_evidence(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--pbs-window-start-utc")
    parser.add_argument("--pbs-window-end-utc")
    args = parser.parse_args()
    payload = capture_gpu_evidence(
        args.output,
        device_index=args.device_index,
        pbs_job_id=args.pbs_job_id,
        pbs_window_start_utc=args.pbs_window_start_utc,
        pbs_window_end_utc=args.pbs_window_end_utc,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
