#!/usr/bin/env python3
"""Run actual robot-specific GR00T inference on native same-scene observations.

The observations are rendered by the Stage21 native IsaacGym WBC process and
replayed here through the already-qualified Stage20S custom embodiment path.
This is an offline cross-process handoff: it does not claim real-time camera
transport.  The script records the full universal output and the decoded
physical output, but it does not execute the trajectory through WBC.
"""

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


def _sha256_image(value: np.ndarray) -> str:
    return _sha256_array(np.asarray(value, dtype=np.uint8))


def _load_modality(path: Path, stage10_root: Path):
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    # Stage22's frozen modality module uses the authoritative Go2+ARX5 name;
    # retain the older Stage20 name for already-sealed visual fixtures.
    if hasattr(module, "GO2_ARX5_MODALITY_CONFIG"):
        return module.GO2_ARX5_MODALITY_CONFIG
    if hasattr(module, "SIM_VISUAL_REACH_MODALITY_CONFIG"):
        return module.SIM_VISUAL_REACH_MODALITY_CONFIG
    raise AttributeError(f"{path} has no supported modality configuration symbol")


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


def _load_observation(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = ("body_rgb", "wrist_rgb", "state")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        target = None if "target_position_world_m" not in data else np.asarray(
            data["target_position_world_m"], dtype=np.float64
        ).reshape(-1)
    for name, image in (("body_rgb", body), ("wrist_rgb", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError(f"{path} {name} must have shape [H,W,3]")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(f"{path} {name} is not finite numeric RGB data")
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError(f"{path} state must be finite [10]")
    if target is not None and (target.shape != (3,) or not np.isfinite(target).all()):
        raise ValueError(f"{path} target_position_world_m must be finite [3]")
    return {
        "path": str(path),
        "body_rgb": body.astype(np.uint8, copy=False),
        "wrist_rgb": wrist.astype(np.uint8, copy=False),
        "state": state,
        "target_position_world_m": target,
    }


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _run_one_raw(model: Any, processor: Any, embodiment: Any, observation: dict[str, Any], *, language: str, seed: int, device: str):
    _set_seed(seed)
    from gr00t.data.types import MessageType, VLAStepData

    state = observation["state"]
    content = VLAStepData(
        images={
            "external_rgb": observation["body_rgb"][None, ...],
            "wrist_rgb": observation["wrist_rgb"][None, ...],
        },
        states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]},
        actions={},
        text=language,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    inputs = _rec_to_dtype(processor.collator([processed]), torch.bfloat16)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model.get_action(**inputs)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    raw = prediction["action_pred"].float().detach().cpu().numpy().astype(np.float32)
    if raw.shape != (1, 40, 132) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid universal output {raw.shape}")
    gpu = {"name": "not used", "peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        index = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(index),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / 2**20, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / 2**20, 3),
        }
    return raw, latency, gpu


def _run_one(model: Any, processor: Any, embodiment: Any, observation: dict[str, Any], *, language: str, seed: int, device: str):
    raw, latency, gpu = _run_one_raw(
        model, processor, embodiment, observation, language=language, seed=seed, device=device
    )
    state = observation["state"]
    batched_state = {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]}
    decoded_groups = processor.decode_action(raw, embodiment, batched_state)
    decoded = np.concatenate(
        [np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])], axis=-1
    ).astype(np.float32)
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        raise RuntimeError(f"invalid decoded output {decoded.shape}")
    return raw, decoded, latency, gpu


def _pairwise_diagnostic(results: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix

    diagnostics = []
    for left_index in range(len(results)):
        for right_index in range(left_index + 1, len(results)):
            left = results[left_index]
            right = results[right_index]
            left_state = np.asarray(left["input_state"], dtype=np.float64)
            right_state = np.asarray(right["input_state"], dtype=np.float64)
            left_decoded = left["decoded_physical_action"][0]
            right_decoded = right["decoded_physical_action"][0]
            rotation_error = geodesic_angle(
                rot6d_rows_to_matrix(left_decoded[:, 3:9]),
                rot6d_rows_to_matrix(right_decoded[:, 3:9]),
            )
            decoded_delta = np.abs(left_decoded - right_decoded)
            diagnostics.append(
                {
                    "left": left["input_id"],
                    "right": right["input_id"],
                    "same_state": bool(np.max(np.abs(left_state - right_state)) <= 1e-7),
                    "same_language": language == left["language"] == right["language"],
                    "body_image_changed": left["body_sha256"] != right["body_sha256"],
                    "wrist_image_changed": left["wrist_sha256"] != right["wrist_sha256"],
                    "target_left_world_m": left.get("target_position_world_m"),
                    "target_right_world_m": right.get("target_position_world_m"),
                    "raw_max_abs_delta": float(np.max(np.abs(left["raw_universal_action"] - right["raw_universal_action"]))),
                    "decoded_max_abs_delta": float(np.max(decoded_delta)),
                    "decoded_eef_position_mean_abs_delta_m": float(np.mean(decoded_delta[:, :3])),
                    "decoded_eef_position_endpoint_delta_m": float(np.linalg.norm(left_decoded[-1, :3] - right_decoded[-1, :3])),
                    "decoded_rotation_mean_delta_rad": float(np.mean(rotation_error)),
                    "decoded_rotation_max_delta_rad": float(np.max(rotation_error)),
                    "gripper_mean_abs_delta": float(np.mean(decoded_delta[:, 9])),
                    "visual_output_difference_at_1e-5": bool(np.max(decoded_delta) > 1e-5),
                    "interpretation": "counterfactual_output_sensitivity_only; not proof of visual_grounding",
                }
            )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="move the end effector to the red target")
    parser.add_argument("--action-period-s", type=float, default=0.05)
    parser.add_argument("--action-period-provenance", default="TEST_ONLY_STAGE21_SIMULATION")
    parser.add_argument("--seed", type=int, default=2121)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.action_period_s) or args.action_period_s <= 0.0:
        parser.error("--action-period-s must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = [_load_observation(path) for path in args.input]
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401; registers model
    from transformers import AutoModel

    processor, embodiment = _build_processor(
        args.stage10_root, args.base_model_path, args.dataset_dir / "meta", args.modality_config
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
    )
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    source_repo = args.stage10_root / "src" / "isaac-gr00t"
    index_files = sorted(args.model_path.glob("*.safetensors.index.json"))
    inference_results: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        raw, decoded, latency, gpu = _run_one(
            model,
            processor,
            embodiment,
            observation,
            language=args.language,
            # Keep the RNG seed identical across counterfactual inputs so a
            # decoded difference is attributable to the supplied observation
            # as far as the model runtime permits.
            seed=int(args.seed),
            device=args.device,
        )
        input_id = Path(observation["path"]).stem
        artifact = args.output_dir / f"{input_id}_groot_output.npz"
        np.savez_compressed(
            artifact,
            raw_universal_action=raw,
            decoded_physical_action=decoded,
            input_state=observation["state"],
            body_rgb=observation["body_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
        inference_results.append(
            {
                "input_id": input_id,
                "input_path": observation["path"],
                "language": args.language,
                "input_state": observation["state"].tolist(),
                "target_position_world_m": None
                if observation["target_position_world_m"] is None
                else observation["target_position_world_m"].tolist(),
                "body_shape": list(observation["body_rgb"].shape),
                "wrist_shape": list(observation["wrist_rgb"].shape),
                "body_sha256": _sha256_image(observation["body_rgb"]),
                "wrist_sha256": _sha256_image(observation["wrist_rgb"]),
                "raw_universal_action": raw,
                "decoded_physical_action": decoded,
                "raw_shape": list(raw.shape),
                "decoded_shape": list(decoded.shape),
                "finite": bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
                "raw_sha256": _sha256_array(raw),
                "decoded_sha256": _sha256_array(decoded),
                "latency_s": latency,
                "gpu": gpu,
                "artifact": str(artifact),
                "seed": int(args.seed),
            }
        )

    public_results = []
    for item in inference_results:
        public_results.append({key: value for key, value in item.items() if key not in {"raw_universal_action", "decoded_physical_action"}})
    manifest = {
        "schema": "STAGE21_SAME_SCENE_GR00T_INFERENCE_V1",
        "status": "PASS",
        "stage": "Stage21",
        "simulation_only": True,
        "hardware_control": False,
        "actual_wbc_execution": False,
        "same_scene_observation_replayed": True,
        "offline_cross_process_handoff": True,
        "model_path": str(args.model_path),
        "base_model_path": str(args.base_model_path),
        "checkpoint_index_sha256": None if not index_files else _sha256_file(index_files[0]),
        "groot_repo_sha": subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip(),
        "inputs": public_results,
        "counterfactual_diagnostics": _pairwise_diagnostic(inference_results, args.language),
        "output_contract": {
            "raw_universal": [1, 40, 132],
            "decoded_physical": [1, 40, 10],
            "eef": [40, 9],
            "gripper": [40, 1],
            "split": {"eef": [0, 9], "gripper": [9, 10]},
            "finite": all(bool(np.isfinite(item["raw_universal_action"]).all() and np.isfinite(item["decoded_physical_action"]).all()) for item in inference_results),
        },
        "timing": {
            "action_sample_period_s": float(args.action_period_s),
            "action_sample_period_provenance": args.action_period_provenance,
            "production_action_period": "TBD_BY_ADAPTATION_DATASET",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(args.device),
        },
        "claim_boundary": "same_scene_rgb_and_state_replay; actual_model_output_audit_only; no_task_success_claim",
    }
    (args.output_dir / "same_scene_groot_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
