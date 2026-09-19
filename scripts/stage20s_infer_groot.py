#!/usr/bin/env python3
"""Run one actual Stage20S GR00T inference on a materialized sim fixture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    return value


def _load_modality(path: Path, stage10_root: Path):
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    return module.SIM_VISUAL_REACH_MODALITY_CONFIG


def _split_stats(values: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
    keys = ("min", "max", "mean", "std", "q01", "q99")
    return {
        "eef_9d": {key: values[key][:9] for key in keys},
        "gripper_position": {key: values[key][9:10] for key in keys},
    }


def _build_processor(stage10_root: Path, base_model: Path, dataset_meta: Path, modality_path: Path):
    modality = _load_modality(modality_path, stage10_root)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_config = json.loads((base_model / "processor_config.json").read_text())
    model_statistics = json.loads((base_model / "statistics.json").read_text())
    flat_stats = json.loads((dataset_meta / "stats.json").read_text())
    relative_stats = json.loads((dataset_meta / "relative_stats.json").read_text())
    custom_statistics = {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "state": _split_stats(flat_stats["observation.state"]),
            "action": _split_stats(flat_stats["action"]),
            "relative_action": {"eef_9d": relative_stats["eef_9d"]},
        }
    }
    kwargs = copy.deepcopy(processor_config["processor_kwargs"])
    kwargs["modality_configs"][EmbodimentTag.NEW_EMBODIMENT.value] = modality
    kwargs["statistics"] = {**model_statistics, **custom_statistics}
    kwargs["model_name"] = "nvidia/Cosmos-Reason2-2B"
    kwargs["model_type"] = "qwen"
    processor = Gr00tN1d7Processor(**kwargs)
    processor.eval()
    return processor, EmbodimentTag.NEW_EMBODIMENT


def _read_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode video frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[None, ...]


from stage21_same_scene_groot_inference import _run_one  # compatibility export; canonical dense-relative path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--action-period-provenance", default="TEST_ONLY_SIM_VISUAL_REACH_V0")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401; registers model type
    from transformers import AutoModel
    import pandas as pd

    parquet = args.dataset_dir / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    table = pd.read_parquet(parquet)
    state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32)
    desired = np.stack([np.asarray(value, dtype=np.float32) for value in table.iloc[:40]["action"]], axis=0)
    tasks = [json.loads(line) for line in (args.dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    task_index = int(table.iloc[0]["task_index"])
    language = next(row["task"] for row in tasks if int(row["task_index"]) == task_index)
    body_video = args.dataset_dir / "videos" / "chunk-000" / "observation.images.external_rgb" / f"episode_{args.episode:06d}.mp4"
    wrist_video = args.dataset_dir / "videos" / "chunk-000" / "observation.images.wrist_rgb" / f"episode_{args.episode:06d}.mp4"
    video = {
        "external_rgb": _read_first_frame(body_video),
        "wrist_rgb": _read_first_frame(wrist_video),
    }

    processor, embodiment = _build_processor(
        args.stage10_root,
        args.base_model_path,
        args.dataset_dir / "meta",
        args.modality_config,
    )
    content = VLAStepData(
        images=video,
        states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]},
        actions={},
        text=language,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    inputs = _rec_to_dtype(processor.collator([processed]), torch.bfloat16)

    model_kwargs = {
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": True,
    }
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model.get_action(**inputs)
    latency = time.perf_counter() - start
    raw = prediction["action_pred"].float().detach().cpu().numpy().astype(np.float32)
    if raw.shape != (1, 40, 132) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid universal action output: shape={raw.shape}")
    batched_state = {
        "eef_9d": state[None, None, :9],
        "gripper_position": state[None, None, 9:],
    }
    decoded_groups = processor.decode_action(raw, embodiment, batched_state)
    decoded = np.concatenate(
        [np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])],
        axis=-1,
    ).astype(np.float32)
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        raise RuntimeError(f"invalid physical action output: shape={decoded.shape}")

    artifact = args.output_dir / "groot_visual_reach_output.npz"
    np.savez_compressed(artifact, raw_universal_action=raw, decoded_physical_action=decoded, input_state=state, desired_action=desired)
    source_repo = args.stage10_root / "src" / "isaac-gr00t"
    index_files = sorted(args.model_path.glob("*.safetensors.index.json"))
    index_sha = _sha256_file(index_files[0]) if index_files else None
    gpu = {"name": "not used", "peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(index),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / 2**20, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / 2**20, 3),
        }
    manifest = {
        "schema": "STAGE20S_GR00T_INFERENCE_V1",
        "status": "PASS",
        "simulation_only": True,
        "hardware_control": False,
        "model_path": str(args.model_path),
        "base_model_path": str(args.base_model_path),
        "checkpoint_index_sha256": index_sha,
        "groot_repo_sha": subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip(),
        "dataset_dir": str(args.dataset_dir),
        "episode": int(args.episode),
        "seed": int(args.seed),
        "input": {
            "image_provenance": {"body_video": str(body_video), "wrist_video": str(wrist_video), "frame_index": 0},
            "state": state.tolist(),
            "language": language,
            "state_shape": [1, 1, 10],
        },
        "output": {
            "raw_universal_shape": list(raw.shape),
            "decoded_physical_shape": list(decoded.shape),
            "finite": bool(np.isfinite(decoded).all()),
            "raw_universal_sha256": _sha256_array(raw),
            "decoded_physical_sha256": _sha256_array(decoded),
            "desired_action_sha256": _sha256_array(desired),
            "artifact": str(artifact),
            "split": {"eef": [0, 9], "gripper": [9, 10]},
        },
        "timing": {
            "action_sample_period_s": float(args.action_period_s),
            "action_sample_period_provenance": args.action_period_provenance,
            "inference_latency_s": latency,
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(args.device), "gpu": gpu},
        "task_performance_evaluated": False,
    }
    path = args.output_dir / "groot_visual_reach_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
