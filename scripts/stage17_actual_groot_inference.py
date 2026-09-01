#!/usr/bin/env python3
"""Run exactly one offline Stage10 custom GR00T inference.

This launcher is intentionally node-local in its defaults and writes the
output artifact outside Git.  It does not start a simulator, transport,
camera service, ROS node, or WBC process.  The model output is only decoded
to the physical 40x10 representation; a later Stage17 seal validates it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    return value


def build_custom_processor(stage10_root: Path, model_dir: Path):
    import sys

    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(stage10_root / "artifacts"))
    from go2_arx5_modality import GO2_ARX5_MODALITY_CONFIG
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_config = json.loads((model_dir / "processor_config.json").read_text())
    model_statistics = json.loads((model_dir / "statistics.json").read_text())
    synthetic_meta = stage10_root / "datasets" / "go2_arx5_synthetic" / "meta"
    flat_stats = json.loads((synthetic_meta / "stats.json").read_text())
    relative_stats = json.loads((synthetic_meta / "relative_stats.json").read_text())

    def split_stats(values: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
        keys = ("min", "max", "mean", "std", "q01", "q99")
        return {
            "eef_9d": {key: values[key][:9] for key in keys},
            "gripper_position": {key: values[key][9:10] for key in keys},
        }

    custom_statistics = {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "state": split_stats(flat_stats["observation.state"]),
            "action": split_stats(flat_stats["action"]),
            "relative_action": {"eef_9d": relative_stats["eef_9d"]},
        }
    }
    kwargs = copy.deepcopy(processor_config["processor_kwargs"])
    kwargs["modality_configs"][EmbodimentTag.NEW_EMBODIMENT.value] = GO2_ARX5_MODALITY_CONFIG
    kwargs["statistics"] = {**model_statistics, **custom_statistics}
    # The released processor metadata serializes these as null, while the
    # official processor's constructor defaults are the gated Cosmos model.
    kwargs["model_name"] = "nvidia/Cosmos-Reason2-2B"
    kwargs["model_type"] = "qwen"
    processor = Gr00tN1d7Processor(**kwargs)
    processor.eval()
    return processor, EmbodimentTag.NEW_EMBODIMENT


def read_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read first frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[None, ...]


def load_sample(stage10_root: Path, episode: int) -> tuple[dict[str, Any], np.ndarray, str, dict[str, str]]:
    dataset = stage10_root / "datasets" / "go2_arx5_synthetic"
    table_path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pd.read_parquet(table_path)
    state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32)
    if state.shape != (10,):
        raise ValueError(f"expected 10D state, got {state.shape}")
    tasks = [json.loads(line) for line in (dataset / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    language = next(row["task"] for row in tasks if int(row["task_index"]) == episode)
    dataset_videos = dataset / "videos" / "chunk-000"
    paths = {
        "external_rgb": str(dataset_videos / "observation.images.external_rgb" / f"episode_{episode:06d}.mp4"),
        "wrist_rgb": str(dataset_videos / "observation.images.wrist_rgb" / f"episode_{episode:06d}.mp4"),
    }
    video = {key: read_first_frame(Path(path)) for key, path in paths.items()}
    observation = {
        "video": video,
        "state": {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
        "language": {"annotation.language.task_description": [[language]]},
    }
    return observation, state, language, {key: sha256_file(Path(path)) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, default=Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage17/actual_groot"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu is the safe default while another user's GPU job is active")
    args = parser.parse_args()

    stage10_root = args.stage10_root
    model_dir = args.model_dir or stage10_root / "assets" / "GR00T-N1.7-3B"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing the local GR00T source is required for model registration; the
    # source itself is never copied into this repository or output directory.
    import sys

    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: F401

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    processor, embodiment = build_custom_processor(stage10_root, model_dir)
    observation, state, language, image_hashes = load_sample(stage10_root, args.episode)
    content = VLAStepData(
        images={key: value for key, value in observation["video"].items()},
        states={key: value[0] for key, value in observation["state"].items()},
        actions={},
        text=language,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    inputs = processor.collator([processed])
    inputs = _rec_to_dtype(inputs, dtype=torch.bfloat16)

    model_load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    model_load_s = time.perf_counter() - model_load_start

    inference_start = time.perf_counter()
    with torch.inference_mode():
        prediction = model.get_action(**inputs)
    inference_s = time.perf_counter() - inference_start
    raw = prediction["action_pred"].float().detach().cpu().numpy().astype(np.float32)
    if raw.shape != (1, 40, 132):
        raise RuntimeError(f"unexpected universal output shape: {raw.shape}")
    batched_state = {
        "eef_9d": state[None, None, :9],
        "gripper_position": state[None, None, 9:],
    }
    decoded_groups = processor.decode_action(raw, embodiment, batched_state)
    decoded = np.concatenate(
        [decoded_groups["eef_9d"], decoded_groups["gripper_position"]], axis=-1
    ).astype(np.float32)
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        raise RuntimeError(f"unexpected decoded output: shape={decoded.shape}, finite={np.isfinite(decoded).all()}")

    artifact_path = output_dir / "actual_groot_output.npz"
    np.savez_compressed(
        artifact_path,
        raw_universal_action=raw,
        decoded_physical_action=decoded,
        input_state=state,
    )
    source_repo = stage10_root / "src" / "isaac-gr00t"
    repo_sha = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    processor_config_path = model_dir / "processor_config.json"
    checkpoint_index = model_dir / "model.safetensors.index.json"
    gpu = {}
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        device_index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(device_index),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device_index) / 2**20, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device_index) / 2**20, 3),
        }
    else:
        gpu = {"name": "not used", "peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    manifest = {
        "schema": "STAGE17_ACTUAL_GR00T_OUTPUT_V1",
        "status": "PASS",
        "actual_model_used": True,
        "stage10_root": str(stage10_root),
        "model_dir": str(model_dir),
        "checkpoint_identity": {
            "model_name": "nvidia/GR00T-N1.7-3B",
            "safetensors_index_sha256": sha256_file(checkpoint_index),
            "checkpoint_sha256_manifest": str(stage10_root / "logs" / "checkpoint_sha256.txt"),
        },
        "groot_repo_sha": repo_sha,
        "processor_config_sha256": sha256_file(processor_config_path),
        "sample": {
            "dataset": str(stage10_root / "datasets" / "go2_arx5_synthetic"),
            "episode": int(args.episode),
            "state": state.tolist(),
            "language": language,
            "image_sha256": image_hashes,
            "image_frame_index": 0,
            "image_note": "Stage10 synthetic MP4 first frame; no live camera",
        },
        "input_shapes": {"external_rgb": list(observation["video"]["external_rgb"].shape), "wrist_rgb": list(observation["video"]["wrist_rgb"].shape), "state": [1, 1, 10]},
        "output_shapes": {"raw_universal": list(raw.shape), "decoded_physical": list(decoded.shape)},
        "raw_universal_sha256": sha256_array(raw),
        "decoded_physical_sha256": sha256_array(decoded),
        "output_artifact": str(artifact_path),
        "device_requested": str(args.device),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(args.device),
            "gpu": gpu,
            "model_load_s": model_load_s,
            "inference_latency_s": inference_s,
        },
        "physical_output_order": ["eef_9d[9]", "gripper_position[1]"],
        "eef_rot6d_layout": "GR00T_FIRST_TWO_ROWS",
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_STAGE10_SYNTHETIC_DATASET",
        "task_performance_evaluated": False,
    }
    (output_dir / "actual_groot_output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
