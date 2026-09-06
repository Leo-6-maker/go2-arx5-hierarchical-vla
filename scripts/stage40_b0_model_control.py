#!/usr/bin/env python3
"""Stage40 B0: read-only GR00T control on the sealed TINY1 observation.

The RGB frames come from the frozen TINY1 video fixture and the StateV1 JSON
comes from the sealed Stage22 train-a snapshot.  No simulator, WBC, backward
pass, optimizer, or checkpoint write is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _first_video_frame(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read first frame from {path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise RuntimeError(f"invalid RGB frame shape {value.shape}")
    return value


def _load_state(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = np.asarray(payload["state_v1"]["values"], dtype=np.float32).reshape(-1)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise RuntimeError("sealed StateV1 is not finite [10]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--body-video", type=Path, required=True)
    parser.add_argument("--wrist-video", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="move the end effector to the red target")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, action="append", default=[2223, 2228, 2233, 2238])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    body = _first_video_frame(args.body_video)
    wrist = _first_video_frame(args.wrist_video)
    state = _load_state(args.state_json)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    import stage21_same_scene_groot_inference as inference
    import torch

    processor, embodiment = inference._build_processor(
        args.stage10_root, args.base_model_path, args.dataset_dir / "meta", args.modality_config
    )
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path, local_files_only=bool(args.local_files_only), trust_remote_code=True)
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    observation = {"body_rgb": body, "wrist_rgb": wrist, "state": state}
    rows = []
    for seed in tuple(dict.fromkeys(int(v) for v in args.seed)):
        raw, decoded, latency, gpu = inference._run_one(
            model, processor, embodiment, observation, language=args.language, seed=seed, device=args.device
        )
        rows.append({
            "seed": seed,
            "raw_shape": list(raw.shape),
            "decoded_shape": list(decoded.shape),
            "raw_sha256": inference._sha256_array(raw),
            "decoded_sha256": inference._sha256_array(decoded),
            "finite": bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
            "latency_s": float(latency),
            "gpu": gpu,
        })
    result = {
        "schema": "STAGE40_B0_TINY1_MODEL_CONTROL_V1",
        "stage": "Stage40",
        "pi_gate": "PI_GATE_STATE_0041",
        "status": "PASS" if all(row["finite"] for row in rows) else "FAIL",
        "seeds": [row["seed"] for row in rows],
        "input": {
            "body_video": str(args.body_video),
            "body_video_sha256": hashlib.file_digest(args.body_video.open("rb"), "sha256").hexdigest(),
            "body_frame_shape": list(body.shape),
            "body_frame_sha256": _sha256_array(body),
            "wrist_video": str(args.wrist_video),
            "wrist_video_sha256": hashlib.file_digest(args.wrist_video.open("rb"), "sha256").hexdigest(),
            "wrist_frame_shape": list(wrist.shape),
            "wrist_frame_sha256": _sha256_array(wrist),
            "state_json": str(args.state_json),
            "state_v1": state.tolist(),
        },
        "rows": rows,
        "forbidden_operations": {"wbc": False, "backward": False, "optimizer": False, "checkpoint_write": False},
    }
    (args.output_dir / "stage40_b0_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
