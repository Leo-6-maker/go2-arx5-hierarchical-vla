#!/usr/bin/env python3
"""Construct the frozen Stage40 GR00T runtime and hold at a CT3 barrier.

This process is deliberately construction-only.  It binds the local Cosmos
snapshot, builds the exact processor and STRUCTURED@1000 model, moves the
model to the selected RTX device, records construction telemetry, and waits
for the native process to finish its same-lane qualification.  No observation
is submitted and no model forward is possible through this entry point.
"""

from __future__ import annotations

import typing as _stdlib_typing  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CUTOVER_TOKENS = (
    "VANDA_STAGE40_A40_QUEUE_RELEASED_CORELAB_PRIMARY",
    "VANDA_STAGE40_A40_EXECUTION_ENDED_CORELAB_PRIMARY",
)
RELEASE_TOKEN = "CORELAB_STAGE40_CT3_RELEASE"
DEV_SEED = 4242


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _verify_cutover(token_file: Path, gpu_uuid: str) -> str:
    _require(token_file.is_file() and not token_file.is_symlink(), "cutover token file is missing or symlinked")
    observed = token_file.read_text(encoding="utf-8")
    lines = observed.splitlines()
    _require(len(lines) == 1 and lines[0] in CUTOVER_TOKENS, "explicit Vanda release token is required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid, "CUDA_VISIBLE_DEVICES must be the selected GPU UUID")
    return lines[0]


def _wait_for_release(release_marker: Path, abort_marker: Path, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if abort_marker.exists():
            raise RuntimeError("CT3_ABORT_MARKER_PRESENT")
        if release_marker.is_file():
            payload = json.loads(release_marker.read_text(encoding="utf-8"))
            _require(payload.get("release_token") == RELEASE_TOKEN, "invalid CT3 release marker")
            return payload
        time.sleep(0.25)
    raise TimeoutError("timed out waiting for CT3 release marker")


def _path_info(path: Path, *, directory: bool = False) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if directory:
        _require(resolved.is_dir(), f"required directory is missing: {resolved}")
    else:
        _require(resolved.is_file(), f"required file is missing: {resolved}")
    return {"path": str(resolved), "is_symlink": Path(path).is_symlink()}


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_marker = Path(args.ready_marker).resolve()
    release_marker = Path(args.release_marker).resolve()
    abort_marker = Path(args.abort_marker).resolve()
    gpu_uuid = str(args.gpu_uuid)
    _require(int(args.seed) == DEV_SEED, "CT3 development seed must be 4242")
    cutover_token = _verify_cutover(Path(args.cutover_token_file).resolve(), gpu_uuid)
    path_info = {
        "stage10_root": _path_info(Path(args.stage10_root), directory=True),
        "base_model_path": _path_info(Path(args.base_model_path), directory=True),
        "model_path": _path_info(Path(args.model_path), directory=True),
        "dataset_dir": _path_info(Path(args.dataset_dir), directory=True),
        "modality_config": _path_info(Path(args.modality_config)),
        "cosmos_snapshot": _path_info(Path(args.cosmos_snapshot), directory=True),
    }

    ready_payload: dict[str, Any] = {
        "schema": "CORELAB_STAGE40_CT3_MODEL_READY_V1",
        "status": "STARTING",
        "stage": "RTX-runtime",
        "execution_class": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
        "dev_seed": DEV_SEED,
        "gpu_uuid": gpu_uuid,
        "cutover_token": cutover_token,
        "gr00t_forward_count": 0,
        "wbc_step_count": 0,
        "reference_update_count": 0,
    }
    model = None
    try:
        # Only the exact checked-in helpers are imported after authorization.
        # They build the same processor path used by the accepted Stage21
        # worker, with its Cosmos dependency rebound to the sealed snapshot.
        repo_root = Path(args.repo_root).resolve()
        sys.path.insert(0, str(repo_root / "scripts"))
        sys.path.insert(0, str(repo_root / "src"))
        sys.path.insert(0, str(Path(args.stage10_root).resolve() / "src" / "isaac-gr00t"))
        from stage40_b0_model_control import _bind_local_cosmos_processor
        import stage21_same_scene_groot_inference as inference
        import torch

        _bind_local_cosmos_processor(Path(args.stage10_root), Path(args.cosmos_snapshot))
        processor, embodiment = inference._build_processor(
            Path(args.stage10_root),
            Path(args.base_model_path),
            Path(args.dataset_dir) / "meta",
            Path(args.modality_config),
        )
        # Importing the registration class is required before AutoModel, but
        # no method on the model is called that could produce an action.
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
        from transformers import AutoModel

        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = AutoModel.from_pretrained(
            Path(args.model_path), local_files_only=True, trust_remote_code=True
        )
        model.eval()
        model.to(device=str(args.device), dtype=torch.bfloat16)
        parameter_count = int(sum(int(parameter.numel()) for parameter in model.parameters()))
        gpu = {
            "available": bool(torch.cuda.is_available()),
            "device": str(args.device),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 3)
            if torch.cuda.is_available()
            else 0.0,
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20, 3)
            if torch.cuda.is_available()
            else 0.0,
        }
        runtime = {
            "schema": "CORELAB_STAGE40_CT3_MODEL_RUNTIME_V1",
            "status": "READY",
            "stage": "RTX-runtime",
            "execution_class": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
            "gpu_uuid": gpu_uuid,
            "cutover_token": cutover_token,
            "device": str(args.device),
            "model_constructor": "transformers.AutoModel.from_pretrained",
            "model_path": str(Path(args.model_path).resolve()),
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "modality_config": str(Path(args.modality_config).resolve()),
            "cosmos_snapshot": str(Path(args.cosmos_snapshot).resolve()),
            "processor_embodiment": str(embodiment),
            "parameter_count": parameter_count,
            "parameter_dtype": "torch.bfloat16",
            "gpu": gpu,
            "path_info": path_info,
            "counters": {
                "gr00t_forward_count": {"before": 0, "after": 0, "delta": 0},
                "wbc_step_count": {"before": 0, "after": 0, "delta": 0},
                "reference_update_count": {"before": 0, "after": 0, "delta": 0},
            },
            "forbidden_operations": {
                "gr00t_forward": False,
                "wbc_step": False,
                "reference_update": False,
                "training": False,
                "checkpoint_write": False,
            },
            "observation_submitted": False,
        }
        _write_json(output_dir / "model_runtime.json", runtime)
        ready_payload.update({
            "status": "READY",
            "model_runtime": str(output_dir / "model_runtime.json"),
            "counters": runtime["counters"],
        })
        _write_json(ready_marker, ready_payload)
        release = _wait_for_release(release_marker, abort_marker, float(args.wait_timeout_s))
        runtime.update({"status": "PASS", "released": True, "release": release})
        _write_json(output_dir / "model_runtime.json", runtime)
        ready_payload.update({"status": "PASS", "released": True})
        _write_json(ready_marker, ready_payload)
        return 0
    except Exception as exc:
        error = {
            "schema": "CORELAB_STAGE40_CT3_MODEL_RUNTIME_ERROR_V1",
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_uuid": gpu_uuid,
            "cutover_token": cutover_token,
            "gr00t_forward_count": 0,
            "wbc_step_count": 0,
            "reference_update_count": 0,
        }
        _write_json(output_dir / "model_runtime_error.json", error)
        ready_payload.update(error)
        try:
            _write_json(ready_marker, ready_payload)
        except Exception:
            pass
        return 2
    finally:
        # Deleting the reference after the release keeps teardown out of the
        # coexistence measurement and avoids an implicit second construction.
        if model is not None:
            del model


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--cosmos-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--release-marker", type=Path, required=True)
    parser.add_argument("--abort-marker", type=Path, required=True)
    parser.add_argument("--cutover-token-file", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--wait-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.wait_timeout_s <= 0.0:
        parser.error("--wait-timeout-s must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
