#!/usr/bin/env python3
"""Recover H2 GPU identity from immutable training-time artifacts.

This adapter never runs ``nvidia-smi``.  It accepts a read-only PBS log plus a
structured PBS accounting sidecar, extracts only an explicitly logged
``nvidia-smi --query-gpu=index,name,uuid,memory.total`` result, and binds the
identity to the source hashes, job window, recipe, and run manifest.  Ambiguous
or incomplete historical evidence is rejected rather than replaced by a
present-time observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


HISTORICAL_GPU_EVIDENCE_SCHEMA = "STAGE22_GPU_EVIDENCE_FROM_IMMUTABLE_H2_LOG_V1"
GPU_QUERY = "index,name,uuid,memory.total"
GPU_QUERY_MARKER = f"--query-gpu={GPU_QUERY}"
_DEVICE_FIELDS = ("index", "name", "uuid", "memory_total_mib")
_CSV_DEVICE = re.compile(
    r"^\s*(\d+)\s*,\s*([^,\r\n]+?)\s*,\s*(GPU-[^,\s]+)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*(?:MiB)?\s*$"
)
_ISO_TIMESTAMP = re.compile(
    r"20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
_CVD = re.compile(r"(?:CUDA_VISIBLE_DEVICES|CVD)\s*[=:]\s*([^\s,;]+)")
_PBS_JOB = re.compile(r"PBS_JOBID\s*[=:]\s*([^\s,;]+)")
_HOSTNAME_ASSIGNMENT = re.compile(r"(?:HOSTNAME|host)\s*[=:]\s*([^\s,;]+)")
_HOSTNAME_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"historical GPU evidence {field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"historical GPU evidence {field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"historical GPU evidence {field} must include a timezone")
    return parsed


def _validate_device(device: Mapping[str, Any]) -> dict[str, Any]:
    if any(key not in device for key in _DEVICE_FIELDS):
        raise ValueError("historical GPU evidence device row is incomplete")
    if not isinstance(device.get("index"), int) or isinstance(device.get("index"), bool):
        raise ValueError("historical GPU evidence device index is invalid")
    if not isinstance(device.get("name"), str) or not device["name"].strip():
        raise ValueError("historical GPU evidence device name is missing")
    if not isinstance(device.get("uuid"), str) or not device["uuid"].strip():
        raise ValueError("historical GPU evidence device UUID is missing")
    if not isinstance(device.get("memory_total_mib"), int) or device["memory_total_mib"] <= 0:
        raise ValueError("historical GPU evidence device VRAM is invalid")
    return dict(device)


def _parse_devices(text: str) -> list[dict[str, Any]]:
    if GPU_QUERY_MARKER not in text:
        raise ValueError("historical log does not contain the frozen nvidia-smi query marker")
    marker_line = next(
        index for index, line in enumerate(text.splitlines()) if GPU_QUERY_MARKER in line
    )
    devices: list[dict[str, Any]] = []
    for line in text.splitlines()[marker_line + 1 :]:
        match = _CSV_DEVICE.match(line)
        if match is None:
            continue
        device = {
            "index": int(match.group(1)),
            "name": match.group(2).strip(),
            "uuid": match.group(3).strip(),
            "memory_total_mib": int(float(match.group(4))),
        }
        if device not in devices:
            devices.append(device)
    if not devices:
        raise ValueError("historical log has no parseable output for the frozen nvidia-smi query")
    if len({device["index"] for device in devices}) != len(devices):
        raise ValueError("historical log contains conflicting GPU index rows")
    if len({device["uuid"] for device in devices}) != len(devices):
        raise ValueError("historical log contains conflicting GPU UUID rows")
    return devices


def _parse_hostname(text: str) -> str:
    """Recover one hostname from explicit log output, never from the host now."""

    candidates = set(_HOSTNAME_ASSIGNMENT.findall(text))
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() not in {"hostname", "/bin/hostname"}:
            continue
        for following in lines[index + 1 : index + 4]:
            candidate = following.strip()
            if _HOSTNAME_VALUE.fullmatch(candidate):
                candidates.add(candidate)
                break
    candidates = {candidate for candidate in candidates if _HOSTNAME_VALUE.fullmatch(candidate)}
    if len(candidates) != 1:
        raise ValueError("historical log does not provide one unambiguous hostname")
    return next(iter(candidates))


def _validate_accounting(path: Path) -> dict[str, Any]:
    payload = _load_json(path.resolve(strict=True))
    if payload.get("schema") != "STAGE22_PBS_ACCOUNTING_V1":
        raise ValueError("PBS accounting sidecar has an unexpected schema")
    if payload.get("status") != "COMPLETED":
        raise ValueError("PBS accounting sidecar is not completed")
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("PBS accounting job ID is missing")
    start = _parse_utc(payload.get("start_utc"), field="PBS start_utc")
    end = _parse_utc(payload.get("end_utc"), field="PBS end_utc")
    if end <= start:
        raise ValueError("PBS accounting time window is invalid")
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "job_id": job_id,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
    }


def _validate_run_manifest(path: Path, *, recipe: str) -> dict[str, Any]:
    payload = _load_json(path.resolve(strict=True))
    if payload.get("schema") != "STAGE20S_GR00T_TRAINING_RUN_V1":
        raise ValueError("historical GPU evidence run manifest schema is unexpected")
    if (
        payload.get("status") != "PASS"
        or payload.get("simulation_only") is not True
        or payload.get("hardware_control") is not False
        or payload.get("max_steps") != 50
        or payload.get("action_horizon") != 40
        or payload.get("tune_projector") is not True
        or payload.get("tune_diffusion_model") is not (recipe == "R1")
        or payload.get("tune_visual") is not False
        or payload.get("tune_llm") is not False
        or payload.get("tune_vlln") is not False
    ):
        raise ValueError("historical GPU evidence run manifest does not match the frozen run")
    return payload


def validate_historical_gpu_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a historical evidence payload without reading its source log."""

    if payload.get("schema") != HISTORICAL_GPU_EVIDENCE_SCHEMA:
        raise ValueError("unexpected historical GPU evidence schema")
    if payload.get("status") != "PASS":
        raise ValueError("historical GPU evidence status is not PASS")
    if payload.get("evidence_kind") != "IMMUTABLE_H2_LOG":
        raise ValueError("historical GPU evidence kind is invalid")
    if payload.get("contains_credentials") is not False:
        raise ValueError("historical GPU evidence credential flag is not false")
    if payload.get("source_query") != GPU_QUERY:
        raise ValueError("historical GPU evidence query is not frozen")
    if not isinstance(payload.get("hostname"), str) or not _HOSTNAME_VALUE.fullmatch(payload["hostname"]):
        raise ValueError("historical GPU evidence hostname is missing")
    for key in ("source_artifact_sha256", "accounting_sha256", "run_manifest_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"historical GPU evidence {key} is missing")
    observed = _parse_utc(payload.get("observed_utc"), field="observed_utc")
    accounting = payload.get("pbs_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("historical GPU evidence PBS accounting is missing")
    start = _parse_utc(accounting.get("start_utc"), field="PBS start_utc")
    end = _parse_utc(accounting.get("end_utc"), field="PBS end_utc")
    if accounting.get("schema") != "STAGE22_PBS_ACCOUNTING_V1" or accounting.get("status") != "COMPLETED":
        raise ValueError("historical GPU evidence accounting identity is invalid")
    if not start <= observed <= end:
        raise ValueError("historical GPU evidence is outside the PBS window")
    devices = payload.get("devices")
    selected = payload.get("selected_device")
    if not isinstance(devices, list) or not devices or not isinstance(selected, Mapping):
        raise ValueError("historical GPU evidence devices are missing")
    normalized = [_validate_device(device) for device in devices if isinstance(device, Mapping)]
    if (
        len(normalized) != len(devices)
        or len({device["index"] for device in normalized}) != len(normalized)
        or len({device["uuid"] for device in normalized}) != len(normalized)
        or dict(selected) not in normalized
    ):
        raise ValueError("historical GPU evidence selected device is invalid")
    if payload.get("pbs_job_id") != accounting.get("job_id"):
        raise ValueError("historical GPU evidence PBS job identity is invalid")
    return dict(payload)


def recover_historical_gpu_evidence(
    *,
    source_artifact: Path,
    accounting_json: Path,
    run_manifest: Path,
    recipe: str,
    observed_utc: str,
    output: Path,
    device_index: int | None = None,
) -> dict[str, Any]:
    if recipe not in {"R0", "R1"}:
        raise ValueError("recipe must be R0 or R1")
    if output.exists():
        raise ValueError(f"refusing to overwrite historical GPU evidence: {output}")
    source_artifact = source_artifact.resolve(strict=True)
    accounting_json = accounting_json.resolve(strict=True)
    run_manifest = run_manifest.resolve(strict=True)
    accounting = _validate_accounting(accounting_json)
    run = _validate_run_manifest(run_manifest, recipe=recipe)
    observed = _parse_utc(observed_utc, field="observed_utc")
    start = _parse_utc(accounting["start_utc"], field="PBS start_utc")
    end = _parse_utc(accounting["end_utc"], field="PBS end_utc")
    if not start <= observed <= end:
        raise ValueError("observed GPU timestamp is outside the PBS accounting window")

    raw = source_artifact.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    timestamp_matches = [_parse_utc(value, field="log timestamp") for value in _ISO_TIMESTAMP.findall(text)]
    if not any(value == observed for value in timestamp_matches):
        raise ValueError("observed GPU timestamp is not present in the immutable source log")
    devices = _parse_devices(text)
    hostname = _parse_hostname(text)
    if device_index is None:
        if len(devices) != 1:
            raise ValueError("device index is required when historical log contains multiple GPUs")
        selected = devices[0]
    else:
        matches = [device for device in devices if device["index"] == int(device_index)]
        if len(matches) != 1:
            raise ValueError("requested historical GPU index is not unique")
        selected = matches[0]
    cvd_matches = sorted(set(_CVD.findall(text)))
    job_matches = sorted(set(_PBS_JOB.findall(text)))
    if job_matches and job_matches != [accounting["job_id"]]:
        raise ValueError("historical log PBS_JOBID disagrees with accounting")
    payload = {
        "schema": HISTORICAL_GPU_EVIDENCE_SCHEMA,
        "status": "PASS",
        "evidence_kind": "IMMUTABLE_H2_LOG",
        "recipe": recipe,
        "source_artifact_path": str(source_artifact),
        "source_artifact_sha256": _sha256_file(source_artifact),
        "source_artifact_read_only": True,
        "source_query": GPU_QUERY,
        "hostname": hostname,
        "observed_utc": observed.isoformat(),
        "pbs_accounting_path": str(accounting_json),
        "pbs_accounting_sha256": _sha256_file(accounting_json),
        "accounting_sha256": _sha256_file(accounting_json),
        "pbs_accounting": accounting,
        "pbs_job_id": accounting["job_id"],
        "cuda_visible_devices": cvd_matches[0] if len(cvd_matches) == 1 else None,
        "cuda_visible_devices_observations": cvd_matches,
        "pbs_job_id_observations": job_matches,
        "devices": devices,
        "selected_device": selected,
        "run_manifest_path": str(run_manifest),
        "run_manifest_sha256": _sha256_file(run_manifest),
        "run_manifest_identity": {
            "schema": run.get("schema"),
            "status": run.get("status"),
            "max_steps": run.get("max_steps"),
            "action_horizon": run.get("action_horizon"),
            "tune_diffusion_model": run.get("tune_diffusion_model"),
        },
        "contains_credentials": False,
    }
    validate_historical_gpu_evidence(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--accounting-json", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--recipe", choices=("R0", "R1"), required=True)
    parser.add_argument("--observed-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int)
    args = parser.parse_args()
    payload = recover_historical_gpu_evidence(
        source_artifact=args.source_artifact,
        accounting_json=args.accounting_json,
        run_manifest=args.run_manifest,
        recipe=args.recipe,
        observed_utc=args.observed_utc,
        output=args.output,
        device_index=args.device_index,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
