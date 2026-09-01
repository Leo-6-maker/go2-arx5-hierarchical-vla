#!/usr/bin/env python3
"""Offline shared-filesystem worker for Stage21 synchronous paused simulation.

The native IsaacGym process remains the owner of the simulator.  This worker
only consumes immutable same-scene RGB/StateV1 requests and writes actual
GR00T outputs.  The mailbox is an experiment harness, not a production
network transport; deployment transport remains intentionally unspecified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--mailbox", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2121)
    parser.add_argument("--poll-period-s", type=float, default=0.10)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.poll_period_s <= 0.0:
        parser.error("--poll-period-s must be positive")

    # Import the Stage21 inference helpers only after the CLI is validated.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    import stage21_same_scene_groot_inference as inference
    import torch

    requests = args.mailbox / "requests"
    responses = args.mailbox / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    processor, embodiment = inference._build_processor(
        args.stage10_root,
        args.base_model_path,
        args.dataset_dir / "meta",
        args.modality_config,
    )
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401; register model
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)

    print(json.dumps({
        "schema": "STAGE21_GR00T_WORKER_READY_V1",
        "status": "READY",
        "mailbox": str(args.mailbox),
        "device": str(args.device),
    }), flush=True)
    while True:
        stop_marker = args.mailbox / "STOP"
        if stop_marker.exists():
            return 0
        request_files = sorted(requests.glob("*.json"))
        progressed = False
        for request_path in request_files:
            request_id = request_path.stem
            response_path = responses / f"{request_id}.json"
            if response_path.exists():
                continue
            request = json.loads(request_path.read_text(encoding="utf-8"))
            progressed = True
            try:
                observation = inference._load_observation(Path(request["observation_npz"]))
                raw, decoded, latency, gpu = inference._run_one(
                    model,
                    processor,
                    embodiment,
                    observation,
                    language=str(request["language"]),
                    seed=int(request.get("seed", args.seed)),
                    device=args.device,
                )
                artifact = responses / f"{request_id}.npz"
                temporary_artifact = responses / f"{request_id}.tmp.npz"
                np.savez_compressed(
                    temporary_artifact,
                    raw_universal_action=raw,
                    decoded_physical_action=decoded,
                    input_state=observation["state"],
                )
                temporary_artifact.replace(artifact)
                _write_json_atomic(
                    response_path,
                    {
                        "schema": "STAGE21_GR00T_WORKER_RESPONSE_V1",
                        "status": "PASS",
                        "request_id": request_id,
                        "observation_npz": str(request["observation_npz"]),
                        "artifact": str(artifact),
                        "raw_shape": list(raw.shape),
                        "decoded_shape": list(decoded.shape),
                        "raw_sha256": inference._sha256_array(raw),
                        "decoded_sha256": inference._sha256_array(decoded),
                        "latency_s": latency,
                        "gpu": gpu,
                        "seed": int(request.get("seed", args.seed)),
                        "simulation_only": True,
                        "hardware_control": False,
                    },
                )
                print(json.dumps({"request_id": request_id, "status": "PASS", "latency_s": latency}), flush=True)
            except Exception as exc:
                _write_json_atomic(
                    response_path,
                    {
                        "schema": "STAGE21_GR00T_WORKER_RESPONSE_V1",
                        "status": "ERROR",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(json.dumps({"request_id": request_id, "status": "ERROR", "error": str(exc)}), flush=True)
        if not progressed:
            time.sleep(float(args.poll_period_s))


if __name__ == "__main__":
    raise SystemExit(main())
