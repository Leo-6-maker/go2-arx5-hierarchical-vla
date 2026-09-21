#!/usr/bin/env python3
"""PI0242 read-only bridge attribution.

P0 is deliberately model-free: it replays sealed screen artifacts and builds
the actual training dataset/processor boundary without constructing a model.
P1 is added only after the P0 admission object is sealed.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SEED = 22301
DT = 0.05
HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
SCREEN_STEPS = (0, 250, 500, 1000)
FORBIDDEN_INPUT_TOKENS = ("target", "source_episode", "source_id", "phase", "expert", "future")
LANGUAGE = "move the end effector to the red target"
BRIDGE4_PREFIX = "PI0242_BRIDGE4_V1|"
P1_QUANTILES = (0.0, 0.1, 0.5, 0.9)


class PI0242Blocker(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0242Blocker(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Enum):
        return {"__enum__": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name, "value": _jsonable(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode())


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    x = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = x[..., :3], x[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(n1, 1e-12)
    a2o = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(a2o, axis=-1, keepdims=True)
    b2 = a2o / np.maximum(n2, 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", a, b)
    cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _screen_cell(record: Mapping[str, Any], tensor_path: Path) -> dict[str, Any]:
    with np.load(tensor_path, allow_pickle=False) as payload:
        raw_sha = sha256_array(np.asarray(payload["raw"], dtype=np.float32)) if "raw" in payload else None
        decoded = np.asarray(payload["decoded"], dtype=np.float64)[0]
        expert = np.asarray(payload["expert"], dtype=np.float64)
        state = np.asarray(payload["state"], dtype=np.float64).reshape(-1)
        mask = np.asarray(payload["action_mask"])
    require(decoded.shape == (HORIZON, ACTIVE_DIM), f"decoded shape drift: {tensor_path}")
    require(expert.shape == (HORIZON, ACTIVE_DIM), f"expert shape drift: {tensor_path}")
    require(mask.shape == (1, HORIZON, UNIVERSAL_DIM), f"mask shape drift: {tensor_path}")
    position_error = decoded[:, :3] - expert[:, :3]
    xyz_sq = np.sum(position_error**2, axis=1)
    theta = _geodesic(_rot6d_rows_to_matrix(decoded[:, 3:9]), _rot6d_rows_to_matrix(expert[:, 3:9]))
    model_linear = np.linalg.norm(np.diff(decoded[:, :3], axis=0), axis=1) / DT
    expert_linear = np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1) / DT
    model_angular = _geodesic(
        _rot6d_rows_to_matrix(decoded[:-1, 3:9]), _rot6d_rows_to_matrix(decoded[1:, 3:9])
    ) / DT
    expert_angular = _geodesic(
        _rot6d_rows_to_matrix(expert[:-1, 3:9]), _rot6d_rows_to_matrix(expert[1:, 3:9])
    ) / DT
    bias = np.mean(position_error, axis=0)
    residual = position_error - bias[None, :]
    observed = {
        "xyz_rmse_m": float(np.sqrt(np.mean(xyz_sq))),
        "so3_rmse_rad": float(np.sqrt(np.mean(theta**2))),
        "gripper_mae": float(np.mean(np.abs(decoded[:, 9] - expert[:, 9]))),
        "first_xyz_error_m": float(np.sqrt(xyz_sq[0])),
        "endpoint_xyz_error_m": float(np.sqrt(xyz_sq[-1])),
        "first_so3_error_rad": float(theta[0]),
        "endpoint_so3_error_rad": float(theta[-1]),
        "position_bias_m": bias.tolist(),
        "debiased_xyz_rmse_m": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "model_max_linear_speed_mps": float(np.max(model_linear)),
        "expert_max_linear_speed_mps": float(np.max(expert_linear)),
        "model_peak_linear_segment": int(np.argmax(model_linear)),
        "expert_peak_linear_segment": int(np.argmax(expert_linear)),
        "model_max_angular_speed_radps": float(np.max(model_angular)),
        "expert_max_angular_speed_radps": float(np.max(expert_angular)),
        "model_peak_angular_segment": int(np.argmax(model_angular)),
        "expert_peak_angular_segment": int(np.argmax(expert_angular)),
    }
    record_metrics = {
        key: record.get(key)
        for key in (
            "xyz_rmse_m", "so3_rmse_rad", "gripper_mae", "first_xyz_error_m",
            "endpoint_xyz_error_m", "first_so3_error_rad", "endpoint_so3_error_rad",
            "max_linear_speed_mps", "max_angular_speed_radps", "gate_decision",
        )
    }
    recomputed = {
        "xyz_rmse_m": observed["xyz_rmse_m"],
        "so3_rmse_rad": observed["so3_rmse_rad"],
        "gripper_mae": observed["gripper_mae"],
        "first_xyz_error_m": observed["first_xyz_error_m"],
        "endpoint_xyz_error_m": observed["endpoint_xyz_error_m"],
        "first_so3_error_rad": observed["first_so3_error_rad"],
        "endpoint_so3_error_rad": observed["endpoint_so3_error_rad"],
        "max_linear_speed_mps": observed["model_max_linear_speed_mps"],
        "max_angular_speed_radps": observed["model_max_angular_speed_radps"],
    }
    deltas = {
        key: None if record_metrics[key] is None else float(recomputed[key]) - float(record_metrics[key])
        for key in recomputed
    }
    return {
        "source_episode_id": record.get("source_episode_id"),
        "episode_index": int(record["episode_index"]),
        "target_family": record.get("target_family"),
        "phase": int(record.get("phase", -1)),
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256_file(tensor_path),
        "record_gate_decision": record.get("gate_decision"),
        "record_gate_issue_codes": [item.get("code") for item in record.get("gate", {}).get("issues", [])],
        "record_metrics": record_metrics,
        "recomputed_metrics": recomputed,
        "record_minus_recomputed": deltas,
        "recomputed": observed,
        "raw_sha256": raw_sha,
    }


def recompute_screen(run_root: Path) -> dict[str, Any]:
    by_step: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for step in SCREEN_STEPS:
        root = run_root / "screen" / f"step-{step:04d}"
        records = _load_records(root / "records.jsonl")
        cells = []
        for record in records:
            tensor_path = Path(str(record["tensor_path"]))
            if not tensor_path.is_file():
                tensor_path = root / "cells" / f"train-{int(record['episode_index']):06d}.npz"
            require(tensor_path.is_file(), f"missing sealed screen tensor: {tensor_path}")
            cells.append(_screen_cell(record, tensor_path))
        require(len(cells) == 16, f"screen step {step} has {len(cells)} cells")
        xyz = np.asarray([row["recomputed_metrics"]["xyz_rmse_m"] for row in cells], dtype=np.float64)
        so3 = np.asarray([row["recomputed_metrics"]["so3_rmse_rad"] for row in cells], dtype=np.float64)
        gate = Counter(row["record_gate_decision"] for row in cells)
        max_abs_delta = max(
            abs(float(value))
            for row in cells
            for value in row["record_minus_recomputed"].values()
            if value is not None
        )
        by_step[str(step)] = cells
        summaries[str(step)] = {
            "count": len(cells),
            "gate_counts": dict(sorted(gate.items())),
            "median_xyz_rmse_m": float(np.percentile(xyz, 50, method="linear")),
            "p95_xyz_rmse_m": float(np.percentile(xyz, 95, method="linear")),
            "median_so3_rmse_rad": float(np.percentile(so3, 50, method="linear")),
            "max_record_vs_recomputed_abs_delta": max_abs_delta,
            "record_recompute_match": max_abs_delta <= 1e-9,
        }
    return {"steps": list(SCREEN_STEPS), "cells": by_step, "summaries": summaries}


def _leaf_inventory(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            prefix: {
                "shape": list(array.shape), "dtype": str(value.dtype),
                "sha256": sha256_array(np.ascontiguousarray(array)),
                "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
            }
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            prefix: {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": sha256_array(array),
                "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
            }
        }
    if hasattr(value, "data") and not isinstance(value, (str, bytes, dict, list, tuple)):
        return _leaf_inventory(value.data, prefix)
    if isinstance(value, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key, child in value.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            result.update(_leaf_inventory(child, path))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_leaf_inventory(child, f"{prefix}/{index}"))
        return result
    return {}


def _find_leaf(value: Any, needle: str, *, exclude: tuple[str, ...] = ()) -> np.ndarray | None:
    for path, item in _leaf_inventory(value).items():
        lower = path.lower()
        if needle in lower and not any(token in lower for token in exclude):
            # The caller needs the bytes, not only the inventory entry.
            pass
    try:
        import torch
    except ImportError:
        torch = None

    def visit(child: Any, path: str = "") -> np.ndarray | None:
        if torch is not None and isinstance(child, torch.Tensor):
            if needle in path.lower() and not any(token in path.lower() for token in exclude):
                return child.detach().cpu().numpy()
            return None
        if isinstance(child, np.ndarray):
            return child if needle in path.lower() and not any(token in path.lower() for token in exclude) else None
        if hasattr(child, "data") and not isinstance(child, (str, bytes, dict, list, tuple)):
            return visit(child.data, path)
        if isinstance(child, Mapping):
            for key, nested in child.items():
                found = visit(nested, f"{path}/{key}")
                if found is not None:
                    return found
        return None

    return visit(value)


def _observation_message(observation: Mapping[str, Any], embodiment: Any, language: str) -> Any:
    from gr00t.data.types import VLAStepData

    state = np.asarray(observation["state"], dtype=np.float32)
    return VLAStepData(
        images={
            "external_rgb": np.asarray(observation["body_rgb"], dtype=np.uint8)[None, ...],
            "wrist_rgb": np.asarray(observation["wrist_rgb"], dtype=np.uint8)[None, ...],
        },
        states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]},
        actions={},
        text=language,
        embodiment=embodiment,
    )


def _screen_observation(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "body_rgb": np.asarray(payload["body_rgb"], dtype=np.uint8),
            "wrist_rgb": np.asarray(payload["wrist_rgb"], dtype=np.uint8),
            "state": np.asarray(payload["state"], dtype=np.float32).reshape(-1),
        }


def _processor_summary(processor: Any, embodiment: Any) -> dict[str, Any]:
    state_action = getattr(processor, "state_action_processor", None)
    statistics = getattr(state_action, "statistics", None)
    return {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "training_mode": bool(getattr(processor, "training", False)),
        "use_relative_action": bool(getattr(processor, "use_relative_action", False)),
        "use_albumentations": bool(getattr(processor, "use_albumentations", False)),
        "random_rotation_angle": getattr(processor, "random_rotation_angle", None),
        "color_jitter_params": getattr(processor, "color_jitter_params", None),
        "shortest_image_edge": getattr(processor, "shortest_image_edge", None),
        "crop_fraction": getattr(processor, "crop_fraction", None),
        "embodiment": str(getattr(embodiment, "value", embodiment)),
        "embodiment_id_mapping": _jsonable(getattr(processor, "embodiment_id_mapping", None)),
        "modality_configs_sha256": _canonical_hash(getattr(processor, "modality_configs", None)),
        "statistics_sha256": _canonical_hash(statistics),
        "state_action_processor_class": None if state_action is None else f"{type(state_action).__module__}.{type(state_action).__name__}",
    }


def _condition_diff(train_inventory: Mapping[str, Any], infer_inventory: Mapping[str, Any]) -> dict[str, Any]:
    shared = sorted(set(train_inventory) & set(infer_inventory))
    differences = []
    for key in shared:
        left, right = train_inventory[key], infer_inventory[key]
        same = left == right
        differences.append({"path": key, "exact_equal": same, "train": left, "infer": right})
    return {
        "train_only": sorted(set(train_inventory) - set(infer_inventory)),
        "infer_only": sorted(set(infer_inventory) - set(train_inventory)),
        "shared_count": len(shared),
        "exact_equal_count": sum(item["exact_equal"] for item in differences),
        "differences": [item for item in differences if not item["exact_equal"]],
        "forbidden_input_paths": sorted(
            key for key in set(train_inventory) | set(infer_inventory)
            if any(token in key.lower() for token in FORBIDDEN_INPUT_TOKENS)
        ),
    }


def _actual_train_conditions(args: argparse.Namespace, card: Mapping[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the official Pipeline dataset/processor only; never create model."""
    pi0241 = _load_script(args.code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0241_for_pi0242")
    paths = card["paths"]
    pi0241.bind_runtime(card)
    cfg_dir = args.output_root / "ct_pipeline_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    training_output = args.output_root / "ct_training_boundary"
    training_output.mkdir(parents=True, exist_ok=True)
    pi0241.install_fixed_dataset_factory(card, training_output)
    config_args = argparse.Namespace(
        code_root=args.code_root,
        groot_source=Path(paths["groot_source"]),
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        train_dataset=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
    )
    config, config_summary = pi0241._resolved_config(config_args, training_output)
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
    train_dataset, eval_dataset = pipeline._create_dataset(cfg_dir)
    require(eval_dataset is None, "official training pipeline unexpectedly created eval dataset")
    dataset = train_dataset.datasets[0]
    processor = pipeline.processor
    config_embodiment = config.data.datasets[0].embodiment_tag
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment = config_embodiment if hasattr(config_embodiment, "value") else EmbodimentTag(config_embodiment)
    records = []
    for row in selected:
        episode_index = int(row["episode_index"])
        datapoint = dataset.get_datapoint(dataset.episode_loader[episode_index], 0)
        batch = processor.collator([datapoint])
        leaves = _leaf_inventory(batch)
        action = _find_leaf(batch, "action", exclude=("mask",))
        mask = _find_leaf(batch, "action_mask")
        require(action is not None and mask is not None, f"C_T missing action/mask for {row['source_episode_id']}")
        records.append({
            "source_episode_id": row["source_episode_id"],
            "episode_index": episode_index,
            "batch_inventory": leaves,
            "action": np.asarray(action),
            "mask": np.asarray(mask),
        })
    return {
        "config_summary": config_summary,
        "processor": processor,
        "embodiment": embodiment,
        "modality": config.data.modality_configs[config_embodiment],
        "records": records,
        "processor_summary": _processor_summary(processor, embodiment),
        "dataset_statistics_sha256": _canonical_hash(getattr(train_dataset, "global_stats", None)),
    }


def _inference_conditions(args: argparse.Namespace, card: Mapping[str, Any], selected: list[dict[str, Any]], modality: Any) -> dict[str, Any]:
    paths = card["paths"]
    pi0241 = _load_script(args.code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0241_for_pi0242_infer")
    pi0241.bind_runtime(card)
    sys.path[:0] = [str(args.code_root / "scripts"), str(args.code_root / "src"), str(paths["groot_source"])]
    stage21 = _load_script(args.code_root / "scripts" / "stage21_same_scene_groot_inference.py", "stage21_for_pi0242_p1_exec")

    original_loader = stage21._load_modality
    stage21._load_modality = lambda _path, _stage10_root: modality
    try:
        processor, embodiment = stage21._build_processor(
            Path(paths["stage10_root"]), Path(paths["base_model_path"]), Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"])
        )
    finally:
        stage21._load_modality = original_loader
    records = []
    for row in selected:
        npz = Path(card["selected_inputs"][[x["source_episode_id"] for x in card["selected_inputs"]].index(row["source_episode_id"])] ["tensor_path"])
        observation = _screen_observation(npz)
        from gr00t.data.types import MessageType

        processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": _observation_message(observation, embodiment, "move the end effector to the red target")}])
        batch = processor.collator([processed])
        records.append({
            "source_episode_id": row["source_episode_id"],
            "episode_index": int(row["episode_index"]),
            "batch_inventory": _leaf_inventory(batch),
            "observation_npz": str(npz),
        })
    return {"processor": processor, "embodiment": embodiment, "records": records, "processor_summary": _processor_summary(processor, embodiment)}


def _load_script(path: Path, name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sealed_stage45b(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the sealed Stage45B loader order; no fallback to ambient FFmpeg."""
    ffmpeg_root = Path(args.ffmpeg_root or os.environ.get("PI0242_FFMPEG_ROOT", ""))
    torch_site = Path(args.torch_site or os.environ.get("PI0242_TORCH_SITE", ""))
    require(ffmpeg_root.is_dir(), f"sealed Stage45B FFmpeg root missing: {ffmpeg_root}")
    require(torch_site.is_dir(), f"runtime torch site missing: {torch_site}")
    ordered = [
        ffmpeg_root / "lib",
        ffmpeg_root / "compat",
        torch_site / "torchcodec",
        torch_site / "torch" / "lib",
    ]
    require(all(path.is_dir() for path in ordered), f"incomplete Stage45B loader paths: {ordered}")
    ambient = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*(str(path) for path in ordered), ambient]).rstrip(":")
    return {
        "root": str(ffmpeg_root),
        "torch_site": str(torch_site),
        "ordered_paths": [str(path) for path in ordered],
        "ambient_appended": bool(ambient),
        "ld_library_path_sha256": sha256_bytes(os.environ["LD_LIBRARY_PATH"].encode()),
    }


def _bridge4_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in ("a", "b", "c", "d"):
        candidates = [dict(row) for row in rows if str(row["target_family"]) == family]
        require(candidates, f"PI0242 bridge4 has no target family {family}")
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    hashlib.sha256((BRIDGE4_PREFIX + str(row["source_episode_id"])).encode("utf-8")).hexdigest(),
                    str(row["source_episode_id"]),
                ),
            )
        )
    return selected


def _policy_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    inputs = batch.get("inputs", batch)
    require(isinstance(inputs, Mapping), "processor collator inputs are not a mapping")
    forbidden = {"action", "action_mask"}
    return {key: value for key, value in inputs.items() if key not in forbidden}


def _tensor_cpu_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu", dtype=getattr(__import__("torch"), "float32"))
        return value.contiguous().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _snapshot_tree(value: Any, *, to_cpu: bool = False) -> Any:
    """Make an independent tensor/container snapshot without changing dtype."""

    if hasattr(value, "detach") and hasattr(value, "clone"):
        snapshot = value.detach().clone()
        return snapshot.cpu() if to_cpu else snapshot
    if isinstance(value, Mapping):
        snapshot = copy.deepcopy(value)
        for key, item in value.items():
            snapshot[key] = _snapshot_tree(item, to_cpu=to_cpu)
        return snapshot
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item, to_cpu=to_cpu) for item in value)
    if isinstance(value, list):
        return [_snapshot_tree(item, to_cpu=to_cpu) for item in value]
    return copy.deepcopy(value)


def _move_tree(value: Any, *, device: Any) -> Any:
    """Move an independent snapshot to a model device while preserving dtype."""

    if hasattr(value, "detach") and hasattr(value, "to"):
        return value.detach().clone().to(device=device)
    if isinstance(value, Mapping):
        moved = copy.deepcopy(value)
        for key, item in value.items():
            moved[key] = _move_tree(item, device=device)
        return moved
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device=device) for item in value]
    return copy.deepcopy(value)


def _tensor_provenance(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach()
    cpu = tensor.contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sha256": sha256_bytes(raw),
    }


def _backbone_provenance(value: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in ("backbone_features", "backbone_attention_mask", "image_mask"):
            if key in value and hasattr(value[key], "detach"):
                fields[key] = _tensor_provenance(value[key])
    return fields


def _feature_delta(left: Any, right: Any) -> dict[str, Any]:
    left_np = _tensor_cpu_float32(left)
    right_np = _tensor_cpu_float32(right)
    require(left_np.shape == right_np.shape, f"feature shape mismatch: {left_np.shape} != {right_np.shape}")
    delta = right_np.astype(np.float64) - left_np.astype(np.float64)
    return {
        "shape": list(left_np.shape),
        "exact": bool(np.array_equal(left_np, right_np)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    }


def _feature_transform_provenance(raw: Any, once: Any, twice: Any) -> dict[str, Any]:
    require(isinstance(raw, Mapping) and isinstance(once, Mapping) and isinstance(twice, Mapping), "backbone snapshots must be mappings")
    require("backbone_features" in raw and "backbone_features" in once and "backbone_features" in twice, "backbone feature field missing")
    return {
        "raw": _backbone_provenance(raw),
        "processed_once": _backbone_provenance(once),
        "processed_twice": _backbone_provenance(twice),
        "feature_transform_deltas": {
            "raw_to_processed_once": _feature_delta(raw["backbone_features"], once["backbone_features"]),
            "processed_once_to_processed_twice": _feature_delta(once["backbone_features"], twice["backbone_features"]),
            "raw_to_processed_twice": _feature_delta(raw["backbone_features"], twice["backbone_features"]),
        },
    }


def _action_head_config_summary(action_head: Any) -> dict[str, Any]:
    config = getattr(action_head, "config", None)
    return {
        "use_vlln": bool(getattr(config, "use_vlln", False)),
        "vl_self_attention_type": type(action_head.vl_self_attention).__name__,
        "vlln_type": type(action_head.vlln).__name__,
        "vl_self_attention_cfg": _jsonable(getattr(config, "vl_self_attention_cfg", None)),
        "use_alternate_vl_dit": bool(getattr(config, "use_alternate_vl_dit", False)),
        "add_pos_embed": bool(getattr(config, "add_pos_embed", False)),
    }


def _build_p1_conditions(
    args: argparse.Namespace,
    card: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one immutable C_T/C_I batch per bridge window; no model is made here."""
    pi0241 = _load_script(args.code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0241_for_pi0242_p1")
    paths = card["paths"]
    pi0241.bind_runtime(card)
    cfg_dir = args.output_root / "ct_pipeline_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    training_output = args.output_root / "ct_training_boundary"
    training_output.mkdir(parents=True, exist_ok=True)
    pi0241.install_fixed_dataset_factory(card, training_output)
    config_args = argparse.Namespace(
        code_root=args.code_root,
        groot_source=Path(paths["groot_source"]),
        stage10_root=Path(paths["stage10_root"]),
        base_model_path=Path(paths["base_model_path"]),
        train_dataset=Path(paths["train_dataset"]),
        modality_config=Path(paths["modality_config"]),
    )
    config, config_summary = pi0241._resolved_config(config_args, training_output)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
    train_dataset, eval_dataset = pipeline._create_dataset(cfg_dir)
    require(eval_dataset is None, "P1 official training pipeline unexpectedly created eval dataset")
    dataset = train_dataset.datasets[0]
    config_embodiment = config.data.datasets[0].embodiment_tag
    embodiment = config_embodiment if hasattr(config_embodiment, "value") else EmbodimentTag(config_embodiment)
    modality = config.data.modality_configs[config_embodiment]
    ct_records: dict[str, dict[str, Any]] = {}
    for row in selected:
        source_id = str(row["source_episode_id"])
        tensor_path = Path(next(item["tensor_path"] for item in card["selected_inputs"] if item["source_episode_id"] == source_id))
        datapoint = dataset.get_datapoint(dataset.episode_loader[int(row["episode_index"])], 0)
        batch = pipeline.processor.collator([datapoint])
        inputs = batch["inputs"]
        action = _find_leaf(batch, "action", exclude=("mask",))
        mask = _find_leaf(batch, "action_mask")
        require(action is not None and mask is not None, f"P1 C_T action/mask missing: {source_id}")
        with np.load(tensor_path, allow_pickle=False) as payload:
            state = np.asarray(payload["state"], dtype=np.float32)
        policy = _policy_inputs(batch)
        inventory = _leaf_inventory(policy)
        forbidden = sorted(key for key in inventory if any(token in key.lower() for token in FORBIDDEN_INPUT_TOKENS))
        require(not forbidden, f"P1 C_T forbidden model input path: {source_id}: {forbidden}")
        ct_records[source_id] = {
            "source_episode_id": source_id,
            "processor": pipeline.processor,
            "embodiment": embodiment,
            "policy_inputs": policy,
            "policy_inventory": inventory,
            "condition_hash": _canonical_hash(inventory),
            "expert_normalized": _tensor_cpu_float32(action),
            "action_mask": _tensor_cpu_float32(mask),
            "state": state,
            "tensor_path": str(tensor_path),
        }

    stage21 = _load_script(args.code_root / "scripts" / "stage21_same_scene_groot_inference.py", "stage21_for_pi0242_p1")
    original_loader = stage21._load_modality
    stage21._load_modality = lambda _path, _stage10_root: modality
    try:
        infer_processor, infer_embodiment = stage21._build_processor(
            Path(paths["stage10_root"]),
            Path(paths["base_model_path"]),
            Path(paths["train_dataset"]) / "meta",
            Path(paths["modality_config"]),
        )
    finally:
        stage21._load_modality = original_loader
    ci_records: dict[str, dict[str, Any]] = {}
    for row in selected:
        source_id = str(row["source_episode_id"])
        tensor_path = Path(next(item["tensor_path"] for item in card["selected_inputs"] if item["source_episode_id"] == source_id))
        observation = _screen_observation(tensor_path)
        from gr00t.data.types import MessageType

        content = _observation_message(observation, infer_embodiment, LANGUAGE)
        processed = infer_processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
        batch = infer_processor.collator([processed])
        policy = _policy_inputs(batch)
        inventory = _leaf_inventory(policy)
        forbidden = sorted(key for key in inventory if any(token in key.lower() for token in FORBIDDEN_INPUT_TOKENS))
        require(not forbidden, f"P1 C_I forbidden model input path: {source_id}: {forbidden}")
        with np.load(tensor_path, allow_pickle=False) as payload:
            expert_normalized = np.asarray(payload["expert_normalized_132"], dtype=np.float32)
            action_mask = np.asarray(payload["action_mask"], dtype=np.float32)
        ci_records[source_id] = {
            "source_episode_id": source_id,
            "processor": infer_processor,
            "embodiment": infer_embodiment,
            "policy_inputs": policy,
            "policy_inventory": inventory,
            "condition_hash": _canonical_hash(inventory),
            "expert_normalized": expert_normalized,
            "action_mask": action_mask,
            "state": observation["state"],
            "tensor_path": str(tensor_path),
        }
    return {
        "config_summary": config_summary,
        "C_T": ct_records,
        "C_I": ci_records,
        "training_processor_summary": _processor_summary(pipeline.processor, embodiment),
        "inference_processor_summary": _processor_summary(infer_processor, infer_embodiment),
    }


def _load_p1_model(checkpoint: Path, groot_source: Path, device: str) -> Any:
    import torch

    require(torch.cuda.is_available() and str(device).startswith("cuda"), "PI0242 P1 requires an available CUDA device")
    sys.path.insert(0, str(groot_source))
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    torch.manual_seed(SEED)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    return model


def _p1_prepare_inputs(model: Any, condition: Mapping[str, Any], stage21: Any) -> tuple[Any, Any]:
    import torch

    policy = copy.deepcopy(dict(condition["policy_inputs"]))
    policy = stage21._rec_to_dtype(policy, torch.bfloat16)
    backbone_inputs, action_inputs = model.prepare_input(policy)
    return policy, action_inputs


def _fixed_from_noise(action: np.ndarray, noise: np.ndarray, t_value: float) -> dict[str, Any]:
    import torch

    actions = torch.from_numpy(np.asarray(action, dtype=np.float32).copy())
    noise_tensor = torch.from_numpy(np.asarray(noise, dtype=np.float32).copy())
    t = float(t_value)
    require(0.0 <= t <= 0.999, f"P1 probe t out of frozen range: {t}")
    t_tensor = torch.full((actions.shape[0], 1, 1), t, dtype=torch.float32)
    noisy = (1.0 - t_tensor) * noise_tensor + t_tensor * actions
    target = actions - noise_tensor
    mask = torch.zeros_like(actions)
    mask[..., :ACTIVE_DIM] = 1.0
    return {
        "noise": noise_tensor,
        "noisy_action": noisy,
        "target_velocity": target,
        "action_mask": mask,
        "t": t_tensor,
        "t_discretized": torch.full((actions.shape[0],), int(t * 1000.0), dtype=torch.long),
    }


def _decode_physical(processor: Any, embodiment: Any, raw_or_universal: np.ndarray, state: np.ndarray) -> np.ndarray:
    decoded = processor.decode_action(
        np.asarray(raw_or_universal, dtype=np.float32),
        embodiment,
        {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
    )
    physical = np.concatenate([np.asarray(decoded["eef_9d"]), np.asarray(decoded["gripper_position"])], axis=-1).astype(np.float32)
    require(physical.shape == (1, HORIZON, ACTIVE_DIM) and np.isfinite(physical).all(), "P1 physical decode contract failed")
    return physical


def _physical_profile(decoded: np.ndarray, expert: np.ndarray, state: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(decoded, dtype=np.float64)[0]
    label = np.asarray(expert, dtype=np.float64)
    xyz_error = predicted[:, :3] - label[:, :3]
    predicted_rotations = _rot6d_rows_to_matrix(predicted[:, 3:9])
    label_rotations = _rot6d_rows_to_matrix(label[:, 3:9])
    rotation_error = _geodesic(predicted_rotations, label_rotations)
    linear_speed = np.linalg.norm(np.diff(predicted[:, :3], axis=0), axis=1) / DT
    angular_speed = _geodesic(predicted_rotations[:-1], predicted_rotations[1:]) / DT
    expert_angular_speed = _geodesic(label_rotations[:-1], label_rotations[1:]) / DT
    return {
        "xyz_rmse_m": float(np.sqrt(np.mean(np.sum(xyz_error**2, axis=1)))),
        "xyz_max_m": float(np.max(np.linalg.norm(xyz_error, axis=1))),
        "so3_rmse_rad": float(np.sqrt(np.mean(rotation_error**2))),
        "gripper_rmse": float(np.sqrt(np.mean((predicted[:, 9] - label[:, 9]) ** 2))),
        "first_point_jump_m": float(np.linalg.norm(predicted[0, :3] - np.asarray(state[:3], dtype=np.float64))),
        "endpoint_xyz_error_m": float(np.linalg.norm(predicted[-1, :3] - label[-1, :3])),
        "max_linear_speed_mps": float(np.max(linear_speed)) if linear_speed.size else 0.0,
        "max_angular_speed_radps": float(np.max(angular_speed)) if angular_speed.size else 0.0,
        "expert_max_angular_speed_radps": float(np.max(expert_angular_speed)) if expert_angular_speed.size else 0.0,
        "angular_speed_definition": "geodesic(predicted[t], predicted[t+1])/dt",
        "target_min_distance_m": float(np.min(np.linalg.norm(predicted[:, :3] - target[None, :], axis=1))),
    }


def _flow_group_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    active = np.asarray(mask, dtype=np.float64)
    groups = {"active10": slice(0, 10), "xyz3": slice(0, 3), "rot6d6": slice(3, 9), "gripper1": slice(9, 10)}
    result: dict[str, Any] = {
        "active_mask_sum": float(np.sum(active[..., :ACTIVE_DIM])),
        "inactive_prediction_energy_mean": float(np.mean(np.asarray(predicted)[..., ACTIVE_DIM:] ** 2)),
        "inactive_target_energy_mean": float(np.mean(np.asarray(target)[..., ACTIVE_DIM:] ** 2)),
        "inactive_mask_sum": float(np.sum(active[..., ACTIVE_DIM:])),
    }
    for name, group in groups.items():
        result[f"{name}_velocity_mse"] = float(np.mean((error[..., group] ** 2) * active[..., group]))
    result["active_masked_velocity_mse"] = float(np.sum(error[..., :ACTIVE_DIM] ** 2 * active[..., :ACTIVE_DIM]) / max(np.sum(active[..., :ACTIVE_DIM]), 1.0))
    return result


def _target_map(card: Mapping[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    authority = read_json(Path(card["paths"]["p237_root"]) / "evidence" / "pi0228_train_calib_target_authority.json")
    return {
        (str(row["anchor_id"]).removeprefix("train_"), str(row["jitter_id"])): np.asarray(row["target_position_world_m"], dtype=np.float64)
        for row in authority["rollouts"]
    }


def _stock_query(
    model: Any,
    policy: dict[str, Any],
    stage21: Any,
    fixed_noise: np.ndarray | None,
) -> dict[str, Any]:
    """Call the official outer get_action once and capture its native sampler trace."""
    import torch

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    prepared_policy, action_inputs = _p1_prepare_inputs(model, {"policy_inputs": policy}, stage21)
    backbone_outputs: list[Any] = []
    trace: list[dict[str, Any]] = []
    def capture_backbone(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        # The official action head mutates this container during native get_action.
        # Keep a detached CPU copy before that mutation; direct probes never use
        # the live object returned by the stock call.
        backbone_outputs.append(_snapshot_tree(output, to_cpu=True))

    backbone_handle = model.backbone.register_forward_hook(capture_backbone)
    encoder = model.action_head.action_encoder
    decoder = model.action_head.action_decoder

    def capture_encoder(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) >= 2:
            trace.append({"x": _tensor_cpu_float32(inputs[0]), "t": _tensor_cpu_float32(inputs[1])})

    def capture_decoder(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, (tuple, list)) else output
        require(trace, "P1 sampler decoder ran before action encoder")
        trace[-1]["v"] = _tensor_cpu_float32(value)[..., -HORIZON:, :]

    encoder_handle = encoder.register_forward_pre_hook(capture_encoder)
    decoder_handle = decoder.register_forward_hook(capture_decoder)
    original_randn = torch.randn
    consumed = False
    captured: list[np.ndarray] = []

    def hooked_randn(*args: Any, **kwargs: Any) -> Any:
        nonlocal consumed
        size = kwargs.get("size", args[0] if args else None)
        shape = tuple(int(value) for value in size) if isinstance(size, (tuple, list, torch.Size)) else None
        if shape == (1, HORIZON, UNIVERSAL_DIM) and not consumed:
            consumed = True
            dtype = kwargs.get("dtype", torch.float32)
            device = kwargs.get("device", None)
            if fixed_noise is None:
                value = original_randn(*args, **kwargs)
                captured.append(_tensor_cpu_float32(value))
                return value
            value = torch.from_numpy(np.asarray(fixed_noise, dtype=np.float32)).to(device=device, dtype=dtype)
            captured.append(np.asarray(fixed_noise, dtype=np.float32).copy())
            return value
        return original_randn(*args, **kwargs)

    torch.randn = hooked_randn
    try:
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model.get_action(prepared_policy)
        elapsed = time.perf_counter() - start
    finally:
        torch.randn = original_randn
        backbone_handle.remove()
        encoder_handle.remove()
        decoder_handle.remove()
    require(consumed and len(captured) == 1, "P1 stock sampler did not expose exactly one initial noise tensor")
    require(len(backbone_outputs) == 1, f"P1 expected one backbone forward, observed {len(backbone_outputs)}")
    require(trace and all("v" in row for row in trace), "P1 stock sampler trace is incomplete")
    raw = _tensor_cpu_float32(prediction["action_pred"])
    raw_backbone = backbone_outputs[0]
    model_device = next(iter(model.parameters())).device
    with torch.inference_mode():
        once_device = _move_tree(raw_backbone, device=model_device)
        processed_once = model.action_head.process_backbone_output(once_device)
        twice_device = _snapshot_tree(processed_once)
        processed_twice = model.action_head.process_backbone_output(twice_device)
    once_backbone = _snapshot_tree(processed_once, to_cpu=True)
    twice_backbone = _snapshot_tree(processed_twice, to_cpu=True)
    return {
        "raw": raw,
        "noise": captured[0],
        "trace": trace,
        "backbone_raw": raw_backbone,
        "backbone_once": once_backbone,
        "backbone_twice": twice_backbone,
        "backbone_output": once_backbone,
        "feature_provenance": _feature_transform_provenance(raw_backbone, once_backbone, twice_backbone),
        "action_head_config": _action_head_config_summary(model.action_head),
        "action_inputs": action_inputs,
        "elapsed_s": float(elapsed),
        "sampler_steps": len(trace),
    }


def _stock_gate(decoded: np.ndarray, condition: Mapping[str, Any], row: Mapping[str, Any], target: np.ndarray, selector: Any, gate: Any) -> dict[str, Any]:
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    state = np.asarray(condition["state"], dtype=np.float64)
    state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    result = gate.evaluate(
        selector._build_chunk(decoded, chunk_id=f"pi0242-{row['source_episode_id']}"),
        now=0.0,
        achieved_position=state[:3],
        achieved_orientation=state_quat,
    )
    return result.as_dict()


def _direct_query(
    *,
    model: Any,
    stage26: Any,
    stage23: Any,
    processed_backbone: Any,
    action_inputs: Any,
    condition: Mapping[str, Any],
    row: Mapping[str, Any],
    model_label: str,
    condition_label: str,
    noise: np.ndarray,
    t_quantile: float,
    t_value: float,
    cell_key: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    action = np.asarray(condition["expert_normalized"], dtype=np.float32)
    mask = np.asarray(condition["action_mask"], dtype=np.float32)
    fixed = _fixed_from_noise(action, noise, t_value)
    device = next(iter(model.parameters())).device
    dtype = next(iter(model.parameters())).dtype
    predicted = stage26._teacher_forced_local_flow(
        action_head=model.action_head,
        backbone_output=processed_backbone,
        action_input=action_inputs,
        noisy_action=fixed["noisy_action"].to(device=device, dtype=dtype),
        t_discretized=fixed["t_discretized"].to(device=device),
    )
    predicted_cpu = predicted.detach().to(device="cpu", dtype=torch.float32).contiguous()
    predicted_np = predicted_cpu.numpy().astype(np.float32, copy=False)
    metrics = {
        **stage26._flow_metrics(stage23, predicted_cpu, fixed),
        **_flow_group_metrics(predicted_np, fixed["target_velocity"].numpy(), mask),
    }
    implied = fixed["noise"] + predicted_cpu
    implied_np = implied.numpy().astype(np.float32, copy=False)
    decoded = _decode_physical(condition["processor"], condition["embodiment"], implied_np, condition["state"])
    physical = _physical_profile(decoded, np.asarray(row["expert"], dtype=np.float32), condition["state"], np.asarray(row["target"], dtype=np.float64))
    record = {
        "schema": "PI0242_BRIDGE_DIRECT_FLOW_V1",
        "kind": "direct_flow",
        "model": model_label,
        "condition": condition_label,
        "source_episode_id": str(row["source_episode_id"]),
        "target_family": str(row["target_family"]),
        "phase": int(row["phase"]),
        "cell_key": cell_key,
        "t_quantile": float(t_quantile),
        "t_value": float(t_value),
        "t_discretized": int(fixed["t_discretized"].item()),
        "noise_shape": list(noise.shape),
        "noise_sha256": sha256_array(noise),
        "conditioning_hash": str(condition["condition_hash"]),
        "expert_normalized_sha256": sha256_array(action),
        "action_mask_sha256": sha256_array(mask),
        "raw_predicted_velocity_sha256": sha256_array(predicted_np),
        "implied_universal_action_sha256": sha256_array(implied_np),
        "decoded_physical_action_sha256": sha256_array(decoded),
        "flow_metrics": metrics,
        "physical_metrics": physical,
        "teacher_forced_oracle": True,
        "off_trajectory_oracle_velocity": False,
        "gate_scope": "NOT_APPLIED_TO_TEACHER_FORCED_DIAGNOSTIC",
        "model_forward_scope": "direct_action_head_after_one_shared_backbone",
        "training": False,
        "backward": False,
        "optimizer_step": False,
        "wbc_executed": False,
    }
    arrays = {
        f"{cell_key}__noise": noise.astype(np.float32, copy=False),
        f"{cell_key}__noisy_action": fixed["noisy_action"].numpy().astype(np.float32, copy=False),
        f"{cell_key}__target_velocity": fixed["target_velocity"].numpy().astype(np.float32, copy=False),
        f"{cell_key}__action_mask": mask.astype(np.float32, copy=False),
        f"{cell_key}__predicted_velocity": predicted_np,
        f"{cell_key}__implied_action": implied_np,
        f"{cell_key}__decoded_physical": decoded.astype(np.float32, copy=False),
    }
    return record, arrays


def _stock_record(
    *,
    stock: Mapping[str, Any],
    condition: Mapping[str, Any],
    row: Mapping[str, Any],
    model_label: str,
    condition_label: str,
    target: np.ndarray,
    selector: Any,
    gate: Any,
    cell_key: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = np.asarray(stock["raw"], dtype=np.float32)
    decoded = _decode_physical(condition["processor"], condition["embodiment"], raw, condition["state"])
    expert = np.asarray(row["expert"], dtype=np.float32)
    gate_data = _stock_gate(decoded, condition, row, target, selector, gate)
    trace_x = np.stack([item["x"] for item in stock["trace"]], axis=0).astype(np.float32)
    trace_v = np.stack([item["v"] for item in stock["trace"]], axis=0).astype(np.float32)
    trace_t = np.concatenate([np.asarray(item["t"]).reshape(-1) for item in stock["trace"]]).astype(np.int64)
    record = {
        "schema": "PI0242_BRIDGE_STOCK_V1",
        "kind": "stock_sampler",
        "model": model_label,
        "condition": condition_label,
        "source_episode_id": str(row["source_episode_id"]),
        "target_family": str(row["target_family"]),
        "phase": int(row["phase"]),
        "cell_key": cell_key,
        "condition_hash": str(condition["condition_hash"]),
        "noise_shape": list(stock["noise"].shape),
        "noise_sha256": sha256_array(stock["noise"]),
        "raw_universal_action_sha256": sha256_array(raw),
        "decoded_physical_action_sha256": sha256_array(decoded),
        "physical_metrics": _physical_profile(decoded, expert, condition["state"], target),
        "sampler_steps": int(stock["sampler_steps"]),
        "sampler_trace_x_sha256": sha256_array(trace_x),
        "sampler_trace_v_sha256": sha256_array(trace_v),
        "sampler_trace_t": trace_t.tolist(),
        "sampler_trace_active_energy_mean": float(np.mean(trace_x[..., :ACTIVE_DIM] ** 2)),
        "sampler_trace_inactive_energy_mean": float(np.mean(trace_x[..., ACTIVE_DIM:] ** 2)),
        "gate_decision": str(gate_data["decision"]),
        "gate": gate_data,
        "latency_s": float(stock["elapsed_s"]),
        "stock_sampler_query": True,
        "model_get_action_call": True,
        "training": False,
        "backward": False,
        "optimizer_step": False,
        "wbc_executed": False,
    }
    arrays = {
        f"{cell_key}__noise": np.asarray(stock["noise"], dtype=np.float32),
        f"{cell_key}__trace_x": trace_x,
        f"{cell_key}__trace_v": trace_v,
        f"{cell_key}__trace_t": trace_t,
        f"{cell_key}__raw": raw,
        f"{cell_key}__decoded": decoded.astype(np.float32, copy=False),
    }
    return record, arrays


def _register_pre_hook_with_kwargs(module: Any, callback: Callable[..., None]) -> Any:
    try:
        return module.register_forward_pre_hook(callback, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(lambda hooked_module, inputs: callback(hooked_module, inputs, {}))


def _numeric_diff(left: Any, right: Any) -> dict[str, Any]:
    left_np = _tensor_cpu_float32(left) if hasattr(left, "detach") else np.asarray(left, dtype=np.float32)
    right_np = _tensor_cpu_float32(right) if hasattr(right, "detach") else np.asarray(right, dtype=np.float32)
    if left_np.shape != right_np.shape:
        return {"shape_left": list(left_np.shape), "shape_right": list(right_np.shape), "shape_equal": False}
    delta = right_np.astype(np.float64) - left_np.astype(np.float64)
    return {
        "shape": list(left_np.shape),
        "shape_equal": True,
        "exact": bool(np.array_equal(left_np, right_np)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    }


def _call_field(call: Mapping[str, Any], name: str) -> Any | None:
    kwargs = call.get("kwargs", {})
    if isinstance(kwargs, Mapping) and name in kwargs:
        return kwargs[name]
    return None


def _native_helper_pair(
    *,
    model: Any,
    stage26: Any,
    stock: Mapping[str, Any],
    condition: Mapping[str, Any],
    t_value: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare native action-head.forward with Stage26 on captured native inputs."""

    import torch

    device = next(iter(model.parameters())).device
    dtype = next(iter(model.parameters())).dtype
    action = np.asarray(condition["expert_normalized"], dtype=np.float32)
    mask = np.asarray(condition["action_mask"], dtype=np.float32)
    noise = np.asarray(stock["noise"], dtype=np.float32)
    fixed = _fixed_from_noise(action, noise, t_value)
    base_action_input = _snapshot_tree(stock["action_inputs"])
    native_action_input = _snapshot_tree(base_action_input)
    native_action_input["action"] = torch.from_numpy(action).to(device=device, dtype=dtype)
    native_action_input["action_mask"] = torch.from_numpy(mask).to(device=device, dtype=dtype)
    raw_backbone = _move_tree(stock["backbone_raw"], device=device)

    native_encoder: dict[str, Any] = {}
    native_decoder: dict[str, Any] = {}
    native_dit: dict[str, Any] = {}

    def capture_encoder(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) >= 2:
            native_encoder["x"] = _snapshot_tree(inputs[0])
            native_encoder["t"] = _snapshot_tree(inputs[1])
            if len(inputs) >= 3:
                native_encoder["embodiment_id"] = _snapshot_tree(inputs[2])

    def capture_decoder(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, (tuple, list)) else output
        native_decoder["full"] = _snapshot_tree(value)
        native_decoder["predicted_velocity"] = _snapshot_tree(value[:, -HORIZON:])

    def capture_dit(_module: Any, inputs: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        native_dit["args"] = _snapshot_tree(inputs)
        native_dit["kwargs"] = _snapshot_tree(kwargs)

    encoder_handle = model.action_head.action_encoder.register_forward_pre_hook(capture_encoder)
    decoder_handle = model.action_head.action_decoder.register_forward_hook(capture_decoder)
    dit_handle = _register_pre_hook_with_kwargs(model.action_head.model, capture_dit)
    original_randn = torch.randn
    original_sample_time = model.action_head.sample_time
    consumed = False

    def hooked_randn(*args: Any, **kwargs: Any) -> Any:
        nonlocal consumed
        size = kwargs.get("size", args[0] if args else None)
        shape = tuple(int(value) for value in size) if isinstance(size, (tuple, list, torch.Size)) else None
        if shape == (1, HORIZON, UNIVERSAL_DIM) and not consumed:
            consumed = True
            return torch.from_numpy(noise).to(device=kwargs.get("device", device), dtype=kwargs.get("dtype", dtype))
        return original_randn(*args, **kwargs)

    def fixed_sample_time(batch_size: int, sample_device: Any, sample_dtype: Any) -> Any:
        return torch.full((batch_size,), float(t_value), device=sample_device, dtype=sample_dtype)

    try:
        torch.randn = hooked_randn
        model.action_head.sample_time = fixed_sample_time
        with torch.inference_mode():
            model.action_head.forward(raw_backbone, native_action_input)
    finally:
        torch.randn = original_randn
        model.action_head.sample_time = original_sample_time
        encoder_handle.remove()
        decoder_handle.remove()
        dit_handle.remove()

    require(consumed, "R1 native control did not consume its fixed noise")
    require("x" in native_encoder and "t" in native_encoder, "R1 native action-encoder capture missing")
    require("predicted_velocity" in native_decoder, "R1 native action-decoder capture missing")
    require("kwargs" in native_dit, "R1 native diffusion call capture missing")
    native_kwargs = native_dit["kwargs"]
    required = ("hidden_states", "encoder_hidden_states", "timestep", "image_mask", "backbone_attention_mask")
    require(all(name in native_kwargs for name in required), f"R1 native diffusion arguments missing: {required}")

    helper_backbone = _move_tree(stock["backbone_once"], device=device)
    helper_backbone["backbone_features"] = _snapshot_tree(native_kwargs["encoder_hidden_states"])
    helper_backbone["backbone_attention_mask"] = _snapshot_tree(native_kwargs["backbone_attention_mask"])
    helper_backbone["image_mask"] = _snapshot_tree(native_kwargs["image_mask"])
    helper_action_input = _snapshot_tree(base_action_input)
    helper_encoder: dict[str, Any] = {}
    helper_dit: dict[str, Any] = {}

    def capture_helper_encoder(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) >= 2:
            helper_encoder["x"] = _snapshot_tree(inputs[0])
            helper_encoder["t"] = _snapshot_tree(inputs[1])

    def capture_helper_dit(_module: Any, inputs: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        helper_dit["args"] = _snapshot_tree(inputs)
        helper_dit["kwargs"] = _snapshot_tree(kwargs)

    helper_encoder_handle = model.action_head.action_encoder.register_forward_pre_hook(capture_helper_encoder)
    helper_dit_handle = _register_pre_hook_with_kwargs(model.action_head.model, capture_helper_dit)
    try:
        with torch.inference_mode():
            helper_predicted = stage26._teacher_forced_local_flow(
                action_head=model.action_head,
                backbone_output=helper_backbone,
                action_input=helper_action_input,
                noisy_action=native_encoder["x"],
                t_discretized=native_encoder["t"],
            )
    finally:
        helper_encoder_handle.remove()
        helper_dit_handle.remove()

    require("x" in helper_encoder and "t" in helper_encoder, "R1 helper action-encoder capture missing")
    require("kwargs" in helper_dit, "R1 helper diffusion call capture missing")
    native_pred = native_decoder["predicted_velocity"]
    helper_pred = helper_predicted
    fields = ("hidden_states", "encoder_hidden_states", "timestep", "image_mask", "backbone_attention_mask")
    input_diffs = {
        "action_encoder_x": _numeric_diff(native_encoder["x"], helper_encoder["x"]),
        "action_encoder_t": _numeric_diff(native_encoder["t"], helper_encoder["t"]),
        "diffusion_inputs": {
            name: _numeric_diff(_call_field(native_dit, name), _call_field(helper_dit, name))
            for name in fields
        },
    }
    output_diff = _numeric_diff(native_pred, helper_pred)
    exact_inputs = all(
        bool(item.get("exact", False))
        for item in (
            input_diffs["action_encoder_x"],
            input_diffs["action_encoder_t"],
            *input_diffs["diffusion_inputs"].values(),
        )
    )
    parity_status = "EXACT" if exact_inputs and bool(output_diff.get("exact", False)) else "FLOAT_DIFFERENCE"
    record = {
        "schema": "PI0242_R1_NATIVE_HELPER_PARITY_V1",
        "kind": "native_helper_control",
        "t_value": float(t_value),
        "t_discretized": int(native_encoder["t"].detach().cpu().reshape(-1)[0].item()),
        "noise_sha256": sha256_array(noise),
        "condition_hash": str(condition["condition_hash"]),
        "native_input_provenance": {
            "action_encoder_x": _tensor_provenance(native_encoder["x"]),
            "action_encoder_t": _tensor_provenance(native_encoder["t"]),
            "diffusion_kwargs": {name: _tensor_provenance(_call_field(native_dit, name)) for name in fields},
        },
        "input_diffs": input_diffs,
        "output_diff": output_diff,
        "parity_status": parity_status,
        "native_forward": True,
        "helper_forward": True,
        "training": False,
        "backward": False,
        "optimizer_step": False,
        "wbc_executed": False,
    }
    arrays = {
        "native_predicted_velocity": _tensor_cpu_float32(native_pred),
        "helper_predicted_velocity": _tensor_cpu_float32(helper_pred),
        "predicted_velocity_delta": _tensor_cpu_float32(helper_pred - native_pred),
    }
    return record, arrays


def _r1_metric_correction(
    *,
    args: argparse.Namespace,
    card: Mapping[str, Any],
    fixed_rows: Sequence[Mapping[str, Any]],
    target_map: Mapping[tuple[str, str], np.ndarray],
) -> dict[str, Any]:
    """Recompute only the angular-speed sidecar from the sealed old NPZ."""

    old_root = args.prior_p1_root
    records_path = old_root / "pi0242_bridge_records.jsonl"
    arrays_path = old_root / "pi0242_p1_stock_traces.npz"
    require(records_path.is_file() and arrays_path.is_file(), "R1 prior P1 stock artifacts are incomplete")
    old_records = [row for row in _load_records(records_path) if row.get("kind") == "stock_sampler"]
    rows_by_source = {str(row["source_episode_id"]): row for row in fixed_rows}
    tensor_by_source = {
        str(row["source_episode_id"]): Path(str(row["tensor_path"]))
        for row in card["selected_inputs"]
    }
    corrected: list[dict[str, Any]] = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for old in old_records:
            source_id = str(old["source_episode_id"])
            row = rows_by_source[source_id]
            target_key = (str(row["target_family"]), str(row["jitter_family"]).split("_", 1)[0])
            with np.load(tensor_by_source[source_id], allow_pickle=False) as payload:
                decoded = np.asarray(arrays[f"{old['cell_key']}__decoded"], dtype=np.float32)
                expert = np.asarray(payload["expert"], dtype=np.float32)
                state = np.asarray(payload["state"], dtype=np.float32)
            new_metrics = _physical_profile(decoded, expert, state, target_map[target_key])
            old_metrics = dict(old["physical_metrics"])
            corrected.append({
                "schema": "PI0242_R1_METRIC_CORRECTION_V1",
                "kind": "stock_metric_correction",
                "cell_key": str(old["cell_key"]),
                "source_episode_id": source_id,
                "model": str(old["model"]),
                "condition": str(old["condition"]),
                "gate_decision_preserved": str(old["gate_decision"]),
                "old_max_angular_speed_radps": old_metrics.get("max_angular_speed_radps"),
                "corrected_max_angular_speed_radps": new_metrics["max_angular_speed_radps"],
                "delta_max_angular_speed_radps": None
                if old_metrics.get("max_angular_speed_radps") is None
                else new_metrics["max_angular_speed_radps"] - float(old_metrics["max_angular_speed_radps"]),
                "corrected_expert_max_angular_speed_radps": new_metrics["expert_max_angular_speed_radps"],
                "definition": new_metrics["angular_speed_definition"],
                "decoded_sha256": sha256_array(decoded),
                "original_record_path": str(records_path),
                "original_arrays_path": str(arrays_path),
                "safety_gate_recomputed": False,
            })
    path = args.output_root / "pi0242_r1_metric_correction.json"
    payload = {
        "schema": "PI0242_R1_METRIC_CORRECTION_V1",
        "status": "PASS_READ_ONLY",
        "source_records_sha256": sha256_file(records_path),
        "source_arrays_sha256": sha256_file(arrays_path),
        "count": len(corrected),
        "rows": corrected,
        "gate_decisions_untouched": True,
    }
    write_json(path, payload)
    return {"path": str(path), "sha256": sha256_file(path), "count": len(corrected), "rows": corrected}


def _r1_metric_delta(corrected: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(corrected[key]) - float(original[key])
        for key in ("active_masked_velocity_mse", "xyz_rmse_m", "so3_rmse_rad")
        if key in corrected and key in original
    }


def r1(args: argparse.Namespace) -> dict[str, Any]:
    import gc
    import torch

    require(args.prior_p1_root is not None, "R1 requires --prior-p1-root")
    require(not args.output_root.exists(), f"R1 output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    p0_admission = read_json(args.p0_root / "pi0242_p0_admission.json")
    require(p0_admission.get("status") == "PASS", "R1 requires sealed P0 PASS")
    prior_closure = read_json(args.prior_p1_root / "pi0242_p1_closure.json")
    require(prior_closure.get("status") == "PASS_READ_ONLY", "R1 prior P1 closure is not valid")
    card = read_json(args.run_root / "pi0241_run_card.json")
    fixed_rows = read_json(args.run_root / "fixed16_cohort.json")["rows"]
    bridge_rows = _bridge4_rows(fixed_rows)
    paths = card["paths"]
    conditions = _build_p1_conditions(args, card, bridge_rows)
    target_map = _target_map(card)
    metric_correction = _r1_metric_correction(args=args, card=card, fixed_rows=bridge_rows, target_map=target_map)

    import stage21_same_scene_groot_inference as stage21

    stage26 = _load_script(args.code_root / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py", "stage26_for_pi0242_r1")
    stage23 = _load_script(args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py", "stage23_for_pi0242_r1")
    stage23.assert_protocol_static()
    selector = _load_script(args.code_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py", "selector_for_pi0242_r1")
    from go2arx5_vla.safety import TrajectorySafetyGateV1

    gate = TrajectorySafetyGateV1()
    checkpoint_paths = {
        "initializer": Path(paths["base_model_path"]),
        "final1000": args.run_root / "training" / "checkpoint-1000",
    }
    for label, path in checkpoint_paths.items():
        require((path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file(), f"R1 {label} checkpoint identity incomplete: {path}")

    prior_records = _load_records(args.prior_p1_root / "pi0242_bridge_records.jsonl")
    prior_by_cell = {str(row["cell_key"]): row for row in prior_records}
    output_stock = args.output_root / "pi0242_r1_stock_records.jsonl"
    output_direct = args.output_root / "pi0242_r1_direct_records.jsonl"
    output_parity = args.output_root / "pi0242_r1_native_parity.jsonl"
    stock_arrays: dict[str, np.ndarray] = {}
    direct_arrays: dict[str, np.ndarray] = {}
    parity_arrays: dict[str, np.ndarray] = {}
    noises: dict[str, np.ndarray] = {}
    counters = Counter()
    parity_records: list[dict[str, Any]] = []
    direct_records: list[dict[str, Any]] = []
    stock_records: list[dict[str, Any]] = []

    with output_stock.open("w", encoding="utf-8") as stock_handle, output_direct.open("w", encoding="utf-8") as direct_handle, output_parity.open("w", encoding="utf-8") as parity_handle:
        for model_label, checkpoint in checkpoint_paths.items():
            model = _load_p1_model(checkpoint, Path(paths["groot_source"]), args.device)
            try:
                for row in bridge_rows:
                    source_id = str(row["source_episode_id"])
                    target_key = (str(row["target_family"]), str(row["jitter_family"]).split("_", 1)[0])
                    require(target_key in target_map, f"R1 target authority missing: {target_key}")
                    row_for_metrics = dict(row)
                    tensor_path = next(item["tensor_path"] for item in card["selected_inputs"] if item["source_episode_id"] == source_id)
                    with np.load(tensor_path, allow_pickle=False) as payload:
                        row_for_metrics["expert"] = np.asarray(payload["expert"], dtype=np.float32)
                    row_for_metrics["target"] = target_map[target_key]
                    for condition_label in ("C_I", "C_T"):
                        condition = conditions[condition_label][source_id]
                        cell_key = f"{model_label}__{condition_label}__{source_id.split(':', 1)[0]}"
                        fixed_noise = None if model_label == "initializer" and condition_label == "C_I" else noises.get(source_id)
                        require(fixed_noise is not None or (model_label == "initializer" and condition_label == "C_I"), f"R1 missing paired noise: {source_id}")
                        stock = _stock_query(model, condition["policy_inputs"], stage21, fixed_noise)
                        if fixed_noise is None:
                            noises[source_id] = np.asarray(stock["noise"], dtype=np.float32)
                        require(np.array_equal(np.asarray(stock["noise"], dtype=np.float32), noises[source_id]), f"R1 paired noise mismatch: {source_id}")
                        stock_record, stock_payload = _stock_record(
                            stock=stock,
                            condition=condition,
                            row=row_for_metrics,
                            model_label=model_label,
                            condition_label=condition_label,
                            target=target_map[target_key],
                            selector=selector,
                            gate=gate,
                            cell_key=cell_key,
                        )
                        old_stock = prior_by_cell.get(cell_key, {})
                        stock_record["r1_feature_provenance"] = stock["feature_provenance"]
                        stock_record["r1_action_head_config"] = stock["action_head_config"]
                        stock_record["original_stock_reference"] = {
                            "record_path": str(args.prior_p1_root / "pi0242_bridge_records.jsonl"),
                            "gate_decision": old_stock.get("gate_decision"),
                            "raw_universal_action_sha256": old_stock.get("raw_universal_action_sha256"),
                            "gate_decision_unchanged": old_stock.get("gate_decision") == stock_record["gate_decision"],
                        }
                        stock_handle.write(json.dumps(_jsonable(stock_record), sort_keys=True) + "\n")
                        stock_handle.flush()
                        stock_records.append(stock_record)
                        stock_arrays.update(stock_payload)
                        counters["stock_sampler_queries"] += 1
                        counters["model_get_action_calls"] += 1
                        counters["backbone_forwards"] += 1
                        counters["stock_action_head_sampler_calls"] += 1
                        counters["stock_action_head_denoising_forwards"] += int(stock["sampler_steps"])

                        for t_index, quantile in enumerate(P1_QUANTILES):
                            t_value = float(stage23.official_t_quantile(quantile))
                            direct_key = f"{cell_key}__t{t_index}"
                            direct_record, direct_payload = _direct_query(
                                model=model,
                                stage26=stage26,
                                stage23=stage23,
                                processed_backbone=_move_tree(stock["backbone_once"], device=next(iter(model.parameters())).device),
                                action_inputs=stock["action_inputs"],
                                condition=condition,
                                row=row_for_metrics,
                                model_label=model_label,
                                condition_label=condition_label,
                                noise=noises[source_id],
                                t_quantile=quantile,
                                t_value=t_value,
                                cell_key=direct_key,
                            )
                            original = prior_by_cell.get(direct_key, {})
                            direct_record["r1_feature_provenance"] = stock["feature_provenance"]
                            direct_record["original_reference"] = {
                                "record_path": str(args.prior_p1_root / "pi0242_bridge_records.jsonl"),
                                "cell_key": direct_key,
                                "raw_predicted_velocity_sha256": original.get("raw_predicted_velocity_sha256"),
                                "implied_universal_action_sha256": original.get("implied_universal_action_sha256"),
                                "flow_metrics": original.get("flow_metrics"),
                                "physical_metrics": original.get("physical_metrics"),
                            }
                            direct_record["original_to_corrected_flow_delta"] = _r1_metric_delta(direct_record["flow_metrics"], original.get("flow_metrics", {}))
                            direct_record["original_to_corrected_physical_delta"] = {
                                key: float(direct_record["physical_metrics"][key]) - float(original["physical_metrics"][key])
                                for key in ("xyz_rmse_m", "so3_rmse_rad")
                                if key in original.get("physical_metrics", {})
                            }
                            direct_handle.write(json.dumps(_jsonable(direct_record), sort_keys=True) + "\n")
                            direct_handle.flush()
                            direct_records.append(direct_record)
                            direct_arrays.update(direct_payload)
                            counters["direct_flow_cells"] += 1
                            counters["direct_action_head_forwards"] += 1

                        for parity_quantile in (0.0, 0.5):
                            parity_value = float(stage23.official_t_quantile(parity_quantile))
                            parity_record, parity_payload = _native_helper_pair(
                                model=model,
                                stage26=stage26,
                                stock=stock,
                                condition=condition,
                                t_value=parity_value,
                            )
                            parity_record.update({
                                "model": model_label,
                                "condition": condition_label,
                                "source_episode_id": source_id,
                                "target_family": str(row["target_family"]),
                                "phase": int(row["phase"]),
                                "cell_key": f"{cell_key}__parity_q{int(parity_quantile * 100):02d}",
                                "t_quantile": parity_quantile,
                                "feature_provenance": stock["feature_provenance"],
                            })
                            parity_handle.write(json.dumps(_jsonable(parity_record), sort_keys=True) + "\n")
                            parity_handle.flush()
                            parity_records.append(parity_record)
                            parity_arrays.update({f"{parity_record['cell_key']}__{key}": value for key, value in parity_payload.items()})
                            counters["native_helper_pairs"] += 1
                            counters["native_action_head_forwards"] += 1
                            counters["helper_action_head_forwards"] += 1
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

    require(counters["stock_sampler_queries"] == 16, f"R1 stock capture budget drift: {counters['stock_sampler_queries']}")
    require(counters["direct_flow_cells"] == 64, f"R1 direct budget drift: {counters['direct_flow_cells']}")
    require(counters["native_helper_pairs"] == 32, f"R1 parity budget drift: {counters['native_helper_pairs']}")
    stock_path = args.output_root / "pi0242_r1_stock_traces.npz"
    direct_path = args.output_root / "pi0242_r1_direct_arrays.npz"
    parity_path = args.output_root / "pi0242_r1_native_parity_arrays.npz"
    np.savez_compressed(stock_path, **stock_arrays)
    np.savez_compressed(direct_path, **direct_arrays)
    np.savez_compressed(parity_path, **parity_arrays)

    parity_counts = Counter(str(row["parity_status"]) for row in parity_records)
    direct_by_group: dict[str, dict[str, float]] = {}
    for model in checkpoint_paths:
        for condition_label in ("C_I", "C_T"):
            for quantile in P1_QUANTILES:
                rows = [row for row in direct_records if row["model"] == model and row["condition"] == condition_label and float(row["t_quantile"]) == quantile]
                direct_by_group[f"{model}:{condition_label}:q{int(quantile * 100):02d}"] = {
                    metric: float(np.median([row["flow_metrics"][metric] for row in rows]))
                    for metric in ("active_masked_velocity_mse", "xyz_rmse_m", "so3_rmse_rad")
                }
    run_card = {
        "schema": "PI0242_R1_RUN_CARD_V1",
        "protocol_commit": "030bfa291c0f12dd83141974db846a103f04f2b2",
        "audited_parent": "b3a6c92012ef6949bb08dc4a4ed41dd9760a907e",
        "original_pi0242_closure": "5755139669",
        "p0_evidence": {"path": str(args.p0_root / "pi0242_evidence_index.json"), "sha256": sha256_file(args.p0_root / "pi0242_evidence_index.json")},
        "prior_p1_root": str(args.prior_p1_root),
        "prior_p1_closure_sha256": sha256_file(args.prior_p1_root / "pi0242_p1_closure.json"),
        "bridge4_rows": [{key: value for key, value in row.items() if key in {"source_episode_id", "target_family", "phase", "episode_index", "jitter_family"}} for row in bridge_rows],
        "seed": SEED,
        "time_probes": [{"quantile": q, "t": float(stage23.official_t_quantile(q)), "t_discretized": int(stage23.official_t_quantile(q) * 1000.0)} for q in P1_QUANTILES],
        "noise_sha256_by_source_episode": {key: sha256_array(value) for key, value in sorted(noises.items())},
        "budgets": {"stock_captures": 16, "corrected_direct_cells": 64, "native_helper_control_pairs": 32, "observed": dict(counters)},
        "checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "groot_head": _git_head(Path(paths["groot_source"])), "code_head": _git_head(args.code_root)},
        "action_head_config": stock_records[0]["r1_action_head_config"],
        "selected_checkpoint": "NONE",
        "forbidden_counts": {"training": 0, "backward": 0, "optimizer_steps": 0, "wbc_steps": 0, "sim_steps": 0, "calib_queries": 0, "dev_e_reads": 0, "final_f_reads": 0, "dagger": 0},
    }
    write_json(args.output_root / "pi0242_r1_run_card.json", run_card)
    summary = {
        "schema": "PI0242_R1_SUMMARY_V1",
        "status": "PASS_READ_ONLY",
        "stock_count": len(stock_records),
        "direct_count": len(direct_records),
        "native_helper_count": len(parity_records),
        "stock_gate_counts": dict(Counter(str(row["gate_decision"]) for row in stock_records)),
        "stock_gate_unchanged_count": sum(1 for row in stock_records if row["original_stock_reference"]["gate_decision_unchanged"]),
        "parity_status_counts": dict(parity_counts),
        "corrected_direct_metrics_by_model_condition_t": direct_by_group,
        "metric_correction": metric_correction,
        "feature_transform": {
            "raw_to_once_max_abs_median": float(np.median([row["r1_feature_provenance"]["feature_transform_deltas"]["raw_to_processed_once"]["max_abs"] for row in stock_records])),
            "once_to_twice_max_abs_median": float(np.median([row["r1_feature_provenance"]["feature_transform_deltas"]["processed_once_to_processed_twice"]["max_abs"] for row in stock_records])),
            "once_to_twice_exact_count": sum(1 for row in stock_records if row["r1_feature_provenance"]["feature_transform_deltas"]["processed_once_to_processed_twice"]["exact"]),
        },
        "stock_records_sha256": sha256_file(output_stock),
        "direct_records_sha256": sha256_file(output_direct),
        "parity_records_sha256": sha256_file(output_parity),
        "stock_arrays_sha256": sha256_file(stock_path),
        "direct_arrays_sha256": sha256_file(direct_path),
        "parity_arrays_sha256": sha256_file(parity_path),
        "selected_checkpoint": "NONE",
        "claim_boundary": "corrected read-only parity and affected-matrix attribution only; no promotion, WBC, or task-success claim",
    }
    write_json(args.output_root / "pi0242_r1_summary.json", summary)
    closure = {
        "schema": "PI0242_R1_CLOSURE_V1",
        "status": "PASS_READ_ONLY",
        "run_card_sha256": sha256_file(args.output_root / "pi0242_r1_run_card.json"),
        "summary_sha256": sha256_file(args.output_root / "pi0242_r1_summary.json"),
        "metric_correction_sha256": metric_correction["sha256"],
        "selected_checkpoint": "NONE",
        "counts": dict(counters),
        "parity_status_counts": dict(parity_counts),
        "forbidden_counts": run_card["forbidden_counts"],
    }
    write_json(args.output_root / "pi0242_r1_closure.json", closure)
    return {"status": "PASS_READ_ONLY", "output_root": str(args.output_root), "counts": dict(counters), "summary": summary}


def p1(args: argparse.Namespace) -> dict[str, Any]:
    import gc
    import torch

    p0_evidence_path = args.p0_root / "pi0242_evidence_index.json"
    p0_admission = read_json(args.p0_root / "pi0242_p0_admission.json")
    require(p0_admission.get("status") == "PASS", "P1 admission requires sealed P0 PASS")
    p0_evidence = read_json(p0_evidence_path)
    require(p0_evidence.get("status") == "PASS_READ_ONLY", "P1 P0 evidence status drift")
    card = read_json(args.run_root / "pi0241_run_card.json")
    fixed_rows = read_json(args.run_root / "fixed16_cohort.json")["rows"]
    bridge_rows = _bridge4_rows(fixed_rows)
    paths = card["paths"]
    conditions = _build_p1_conditions(args, card, bridge_rows)
    target_map = _target_map(card)
    import stage21_same_scene_groot_inference as stage21

    stage26 = _load_script(args.code_root / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py", "stage26_for_pi0242_p1")
    stage23 = _load_script(args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py", "stage23_for_pi0242_p1")
    stage23.assert_protocol_static()
    selector = _load_script(args.code_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py", "selector_for_pi0242_p1")
    from go2arx5_vla.safety import TrajectorySafetyGateV1

    gate = TrajectorySafetyGateV1()
    checkpoint_paths = {
        "initializer": Path(paths["base_model_path"]),
        "final1000": args.run_root / "training" / "checkpoint-1000",
    }
    for label, path in checkpoint_paths.items():
        require((path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file(), f"P1 {label} checkpoint identity incomplete: {path}")
    records: list[dict[str, Any]] = []
    stock_arrays: dict[str, np.ndarray] = {}
    direct_arrays: dict[str, np.ndarray] = {}
    noises: dict[str, np.ndarray] = {}
    counters = Counter()
    for model_label, checkpoint in checkpoint_paths.items():
        model = _load_p1_model(checkpoint, Path(paths["groot_source"]), args.device)
        try:
            for row in bridge_rows:
                source_id = str(row["source_episode_id"])
                target_key = (str(row["target_family"]), str(row["jitter_family"]).split("_", 1)[0])
                require(target_key in target_map, f"P1 target authority missing: {target_key}")
                row_for_metrics = dict(row)
                tensor_path = next(item["tensor_path"] for item in card["selected_inputs"] if item["source_episode_id"] == source_id)
                with np.load(tensor_path, allow_pickle=False) as payload:
                    row_for_metrics["expert"] = np.asarray(payload["expert"], dtype=np.float32)
                row_for_metrics["target"] = target_map[target_key]
                for condition_label in ("C_I", "C_T"):
                    condition = conditions[condition_label][source_id]
                    cell_key = f"{model_label}__{condition_label}__{source_id.split(':', 1)[0]}"
                    fixed_noise = None if model_label == "initializer" and condition_label == "C_I" else noises.get(source_id)
                    require(fixed_noise is not None or (model_label == "initializer" and condition_label == "C_I"), f"P1 missing paired noise for {source_id}")
                    stock = _stock_query(model, condition["policy_inputs"], stage21, fixed_noise)
                    if fixed_noise is None:
                        noises[source_id] = np.asarray(stock["noise"], dtype=np.float32)
                    require(np.array_equal(np.asarray(stock["noise"], dtype=np.float32), noises[source_id]), f"P1 paired noise mismatch: {source_id}")
                    stock_record, stock_payload = _stock_record(
                        stock=stock,
                        condition=condition,
                        row=row_for_metrics,
                        model_label=model_label,
                        condition_label=condition_label,
                        target=target_map[target_key],
                        selector=selector,
                        gate=gate,
                        cell_key=cell_key,
                    )
                    records.append(stock_record)
                    stock_arrays.update(stock_payload)
                    counters["stock_sampler_queries"] += 1
                    counters["model_get_action_calls"] += 1
                    counters["backbone_forwards"] += 1
                    counters["stock_action_head_sampler_calls"] += 1
                    counters["stock_action_head_denoising_forwards"] += int(stock["sampler_steps"])
                    model_device = next(iter(model.parameters())).device
                    processed_backbone = _move_tree(stock["backbone_once"], device=model_device)
                    for t_index, quantile in enumerate(P1_QUANTILES):
                        t_value = float(stage23.official_t_quantile(quantile))
                        direct_key = f"{cell_key}__t{t_index}"
                        direct_record, direct_payload = _direct_query(
                            model=model,
                            stage26=stage26,
                            stage23=stage23,
                            processed_backbone=processed_backbone,
                            action_inputs=stock["action_inputs"],
                            condition=condition,
                            row=row_for_metrics,
                            model_label=model_label,
                            condition_label=condition_label,
                            noise=noises[source_id],
                            t_quantile=quantile,
                            t_value=t_value,
                            cell_key=direct_key,
                        )
                        records.append(direct_record)
                        direct_arrays.update(direct_payload)
                        counters["direct_flow_cells"] += 1
                        counters["direct_action_head_forwards"] += 1
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    require(counters["stock_sampler_queries"] == 16, f"P1 stock query budget drift: {counters['stock_sampler_queries']}")
    require(counters["direct_flow_cells"] == 64, f"P1 direct-flow budget drift: {counters['direct_flow_cells']}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stock_trace_path = args.output_root / "pi0242_p1_stock_traces.npz"
    direct_trace_path = args.output_root / "pi0242_p1_direct_arrays.npz"
    np.savez_compressed(stock_trace_path, **stock_arrays)
    np.savez_compressed(direct_trace_path, **direct_arrays)
    records_path = args.output_root / "pi0242_bridge_records.jsonl"
    records_path.write_text("".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in records), encoding="utf-8")
    run_card = {
        "schema": "PI0242_RUN_CARD_V1",
        "protocol_commit": "07d96cf750b7374ab43fd53a044d48e8ab7fab53",
        "audited_parent": "c1b2bd4634cc8cd5b756c9e3d9423d70be88b397",
        "p0_evidence": {"path": str(p0_evidence_path), "sha256": sha256_file(p0_evidence_path)},
        "bridge4_selection_rule": "SHA256(UTF8(PI0242_BRIDGE4_V1|source_episode_id)) ascending per target family a/b/c/d; source ID tie-break",
        "bridge4_rows": [{key: value for key, value in row.items() if key in {"source_episode_id", "target_family", "phase", "episode_index", "jitter_family"}} for row in bridge_rows],
        "seed": SEED,
        "language": LANGUAGE,
        "matrix": "4 windows x initializer/final1000 x C_T/C_I",
        "time_probes": [{"quantile": q, "t": float(stage23.official_t_quantile(q)), "t_discretized": int(stage23.official_t_quantile(q) * 1000.0)} for q in P1_QUANTILES],
        "budgets": {"stock_sampler_queries": 16, "direct_flow_cells": 64, "observed": dict(counters)},
        "noise_sha256_by_source_episode": {key: sha256_array(value) for key, value in sorted(noises.items())},
        "checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "groot_head": _git_head(Path(paths["groot_source"])), "code_head": _git_head(args.code_root)},
        "forbidden_counts": {"training": 0, "backward": 0, "optimizer_steps": 0, "wbc_steps": 0, "sim_steps": 0, "calib_queries": 0, "dev_e_reads": 0, "final_f_reads": 0, "dagger": 0},
        "selected_checkpoint": "NONE",
    }
    write_json(args.output_root / "pi0242_run_card.json", run_card)
    direct = [item for item in records if item["kind"] == "direct_flow"]
    stock = [item for item in records if item["kind"] == "stock_sampler"]
    summary = {
        "schema": "PI0242_P1_SUMMARY_V1",
        "stock_count": len(stock),
        "direct_count": len(direct),
        "stock_gate_counts": dict(Counter(item["gate_decision"] for item in stock)),
        "stock_metrics_by_model_condition": {
            f"{model}:{condition}": {
                "xyz_rmse_m": float(np.median([item["physical_metrics"]["xyz_rmse_m"] for item in stock if item["model"] == model and item["condition"] == condition])),
                "so3_rmse_rad": float(np.median([item["physical_metrics"]["so3_rmse_rad"] for item in stock if item["model"] == model and item["condition"] == condition])),
                "target_min_distance_m": float(np.median([item["physical_metrics"]["target_min_distance_m"] for item in stock if item["model"] == model and item["condition"] == condition])),
            }
            for model in checkpoint_paths for condition in ("C_T", "C_I")
        },
        "direct_metrics_by_model_condition_t": {
            f"{model}:{condition}:q{int(q * 100):02d}": {
                "active_masked_velocity_mse": float(np.median([item["flow_metrics"]["active_masked_velocity_mse"] for item in direct if item["model"] == model and item["condition"] == condition and item["t_quantile"] == q])),
                "xyz_rmse_m": float(np.median([item["physical_metrics"]["xyz_rmse_m"] for item in direct if item["model"] == model and item["condition"] == condition and item["t_quantile"] == q])),
                "so3_rmse_rad": float(np.median([item["physical_metrics"]["so3_rmse_rad"] for item in direct if item["model"] == model and item["condition"] == condition and item["t_quantile"] == q])),
            }
            for model in checkpoint_paths for condition in ("C_T", "C_I") for q in P1_QUANTILES
        },
        "records_sha256": sha256_file(records_path),
        "stock_trace_sha256": sha256_file(stock_trace_path),
        "direct_arrays_sha256": sha256_file(direct_trace_path),
        "selected_checkpoint": "NONE",
        "claim_boundary": "paired local-flow and stock-sampler attribution only; no promotion, WBC, or task-success claim",
    }
    write_json(args.output_root / "pi0242_p1_summary.json", summary)
    write_json(args.output_root / "pi0242_p1_closure.json", {"schema": "PI0242_P1_CLOSURE_V1", "status": "PASS_READ_ONLY", "run_card_sha256": sha256_file(args.output_root / "pi0242_run_card.json"), "summary_sha256": sha256_file(args.output_root / "pi0242_p1_summary.json"), "selected_checkpoint": "NONE", "counts": dict(counters)})
    return {"status": "PASS_READ_ONLY", "output_root": str(args.output_root), "counts": dict(counters), "summary": summary}


def p0(args: argparse.Namespace) -> dict[str, Any]:
    runtime_binding = _bind_sealed_stage45b(args)
    card = read_json(args.run_root / "pi0241_run_card.json")
    closure = read_json(args.run_root / "pi0241_closure.json")
    training_manifest = read_json(args.run_root / "pi0241_training_manifest.json")
    recovery_manifest = read_json(args.run_root / "pi0241_recovery_manifest.json")
    reconciliation = read_json(args.run_root / "pi0241_update_integrity_reconciliation.json")
    archive = read_json(args.run_root / "pi0241_checkpoint_archive.json")
    require(card["selected_checkpoint"] == "NONE", "PI0241 selected checkpoint is not NONE")
    require(closure["selected_checkpoint"] == "NONE", "PI0241 closure selected checkpoint drift")
    require(closure["forbidden_counts"]["calib_queries"] == 0, "PI0241 CALIB boundary drift")

    screen = recompute_screen(args.run_root)
    selected = read_json(args.run_root / "fixed16_cohort.json")["rows"]
    require(len(selected) == 16, "PI0241 cohort is not fixed16")
    ct = _actual_train_conditions(args, card, selected)
    ci = _inference_conditions(args, card, selected, ct["modality"])
    by_ct = {row["source_episode_id"]: row for row in ct["records"]}
    by_ci = {row["source_episode_id"]: row for row in ci["records"]}
    parity_rows = []
    input_forbidden = []
    for row in selected:
        source_id = row["source_episode_id"]
        ct_row, ci_row = by_ct[source_id], by_ci[source_id]
        diff = _condition_diff(ct_row["batch_inventory"], ci_row["batch_inventory"])
        input_forbidden.extend(diff["forbidden_input_paths"])
        tensor_path = Path(next(item["tensor_path"] for item in card["selected_inputs"] if item["source_episode_id"] == source_id))
        with np.load(tensor_path, allow_pickle=False) as payload:
            expert = np.asarray(payload["expert"], dtype=np.float32)
            expected_normalized = np.asarray(payload["expert_normalized_132"], dtype=np.float32)
            expected_mask = np.asarray(payload["action_mask"], dtype=np.float32)
            state = np.asarray(payload["state"], dtype=np.float32)
        ct_action = np.asarray(ct_row["action"], dtype=np.float32)
        ct_mask = np.asarray(ct_row["mask"], dtype=np.float32)
        require(ct_action.shape == (1, HORIZON, UNIVERSAL_DIM), f"C_T action shape drift: {source_id}")
        require(ct_mask.shape == (1, HORIZON, UNIVERSAL_DIM), f"C_T mask shape drift: {source_id}")
        active_slots = np.flatnonzero(np.any(ct_mask != 0, axis=(0, 1))).tolist()
        require(active_slots == list(range(ACTIVE_DIM)), f"C_T active slots drift: {source_id}: {active_slots}")
        ct_decoded = ct["processor"].decode_action(
            ct_action,
            ct["embodiment"],
            {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
        )
        ct_physical = np.concatenate(
            [np.asarray(ct_decoded["eef_9d"]), np.asarray(ct_decoded["gripper_position"])], axis=-1
        ).astype(np.float32)
        ci_decoded = ci["processor"].decode_action(
            expected_normalized,
            ci["embodiment"],
            {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
        )
        ci_physical = np.concatenate(
            [np.asarray(ci_decoded["eef_9d"]), np.asarray(ci_decoded["gripper_position"])], axis=-1
        ).astype(np.float32)
        parity_rows.append({
            "source_episode_id": source_id,
            "episode_index": int(row["episode_index"]),
            "condition": diff,
            "ct_action_sha256": sha256_array(ct_action),
            "sealed_expert_normalized_sha256": sha256_array(expected_normalized),
            "ct_action_max_abs_vs_sealed_normalized": float(np.max(np.abs(ct_action - expected_normalized))),
            "ct_action_rmse_vs_sealed_normalized": float(np.sqrt(np.mean((ct_action - expected_normalized) ** 2))),
            "ct_mask_exact_vs_sealed": bool(np.array_equal(ct_mask, expected_mask)),
            "ct_active_slots": active_slots,
            "ct_physical_max_abs_vs_expert": float(np.max(np.abs(ct_physical[0] - expert))),
            "ct_physical_xyz_rmse_m": float(np.sqrt(np.mean(np.sum((ct_physical[0, :, :3] - expert[:, :3]) ** 2, axis=1)))),
            "ci_decode_max_abs_vs_expert": float(np.max(np.abs(ci_physical[0] - expert))),
            "ci_decode_xyz_rmse_m": float(np.sqrt(np.mean(np.sum((ci_physical[0, :, :3] - expert[:, :3]) ** 2, axis=1)))),
        })

    trainer_state = read_json(args.run_root / "training" / "checkpoint-1000" / "trainer_state.json")
    log_history = trainer_state.get("log_history", [])
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    lrs = [float(item["learning_rate"]) for item in log_history if "learning_rate" in item]
    grads = [float(item["grad_norm"]) for item in log_history if "grad_norm" in item]
    update_audit = read_json(args.run_root / "training" / "pi0241_update_audit.json")
    recovery_audit = read_json(args.run_root / "recovery_audit" / "pi0241_update_audit.json")
    lineage = {
        "observed": {
            "valid_recovery_job": "625395.hopper-m-02",
            "valid_recovery_exit_status": 0,
            "valid_recovery_exec_host": "hopper-14/5*12",
            "valid_screen_job": "625411.hopper-m-02",
            "valid_screen_exit_status": 0,
            "valid_screen_exec_host": "hopper-14/5*12",
            "checkpoint500_tree_sha256": archive["checkpoints"]["500"]["tree_sha256"],
            "checkpoint1000_tree_sha256": archive["checkpoints"]["1000"]["tree_sha256"],
            "checkpoint500_release_after_optimizer_load": read_json(args.run_root / "pi0241_checkpoint500_release_after_optimizer_load.json"),
            "trainer_state_global_step": int(trainer_state["global_step"]),
            "trainer_state_max_steps": int(trainer_state["max_steps"]),
            "recovery_manifest_source_checkpoint": recovery_manifest["source_checkpoint"],
            "recovery_manifest_replayed_optimizer_steps": recovery_manifest["replayed_optimizer_steps"],
            "recovery_audit_global_step": recovery_audit["global_step"],
            "training_consumption_rows_total": len(_load_records(args.run_root / "training" / "pi0241_sample_consumption.jsonl")),
        },
        "deterministically_reconstructed": {
            "resume_code_fix_commit": "518f2b13f959d8bb016437d81bef132aecbd37bd",
            "official_trainer_skip_is_used": True,
            "fixed_schedule_not_manually_skipped_in_resume_wrapper": True,
            "step500_to_1000_optimizer_step_count_from_recovery_manifest": 500,
        },
        "not_available": [
            "independent raw recovery sample-consumption rows for 501-1000",
            "independent recovery sample-contract rows",
            "a log that binds each recovery optimizer step to its consumed source_episode_id",
        ],
        "constant_fields_not_used_as_independent_proof": ["actual_sample_consumption=1000"],
        "old_failed_attempts_not_used_as_valid_recovery_evidence": ["625381", "625384"],
    }
    source_files = {}
    for rel in (
        "scripts/pi0241_fixed16_learning.py", "scripts/pi0241_resume_and_finalize.py",
        "scripts/pi0228_h200_large_adapt_v1.py", "scripts/stage21_same_scene_groot_inference.py",
    ):
        path = args.code_root / rel
        source_files[rel] = {"path": str(path), "sha256": sha256_file(path)}
    model_identity = {}
    for label, path in (
        ("initializer", Path(card["paths"]["base_model_path"])),
        ("final1000", args.run_root / "training" / "checkpoint-1000"),
        ("screen250", args.run_root / "screen_models" / "checkpoint-250"),
        ("screen500", args.run_root / "screen_models" / "checkpoint-500"),
    ):
        model_identity[label] = {
            "path": str(path),
            "config_sha256": sha256_file(path / "config.json"),
            "index_sha256": sha256_file(path / "model.safetensors.index.json"),
        }
    evidence = {
        "schema": "PI0242_P0_BRIDGE_EVIDENCE_V1",
        "status": "PASS_READ_ONLY",
        "protocol_commit": "07d96cf750b7374ab43fd53a044d48e8ab7fab53",
        "audited_parent": "c1b2bd4634cc8cd5b756c9e3d9423d70be88b397",
        "created_by": "scripts/pi0242_train_infer_bridge.py",
        "runtime": {"python": platform.python_version(), "code_head": _git_head(args.code_root), "groot_head": _git_head(Path(card["paths"]["groot_source"])), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "stage45b_binding": runtime_binding, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0, "backward": 0, "optimizer_steps": 0},
        "sealed_inputs": {
            "run_card": {"path": str(args.run_root / "pi0241_run_card.json"), "sha256": sha256_file(args.run_root / "pi0241_run_card.json")},
            "closure": {"path": str(args.run_root / "pi0241_closure.json"), "sha256": sha256_file(args.run_root / "pi0241_closure.json")},
            "training_manifest": {"path": str(args.run_root / "pi0241_training_manifest.json"), "sha256": sha256_file(args.run_root / "pi0241_training_manifest.json")},
            "recovery_manifest": {"path": str(args.run_root / "pi0241_recovery_manifest.json"), "sha256": sha256_file(args.run_root / "pi0241_recovery_manifest.json")},
            "reconciliation": {"path": str(args.run_root / "pi0241_update_integrity_reconciliation.json"), "sha256": sha256_file(args.run_root / "pi0241_update_integrity_reconciliation.json")},
            "archive": {"path": str(args.run_root / "pi0241_checkpoint_archive.json"), "sha256": sha256_file(args.run_root / "pi0241_checkpoint_archive.json")},
        },
        "screen_recomputed": screen,
        "training_curve": {
            "trainer_state_path": str(args.run_root / "training" / "checkpoint-1000" / "trainer_state.json"),
            "log_count": len(log_history),
            "loss_count": len(losses),
            "loss_first12": losses[:12],
            "loss_last12": losses[-12:],
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "lr_first12": lrs[:12],
            "lr_last12": lrs[-12:],
            "grad_first12": grads[:12],
            "grad_last12": grads[-12:],
            "update_audit_sha256": sha256_file(args.run_root / "training" / "pi0241_update_audit.json"),
            "recovery_audit_sha256": sha256_file(args.run_root / "recovery_audit" / "pi0241_update_audit.json"),
            "update_audit_observed": update_audit,
            "recovery_audit_observed": recovery_audit,
        },
        "recovery_lineage": lineage,
        "model_identity": model_identity,
        "source_identity": source_files,
        "condition_paths": {
            "C_T": {
                "definition": "official Gr00tN1d7Pipeline._create_dataset + sealed PI0241 fixed dataset factory; no model construction",
                "processor": ct["processor_summary"],
                "dataset_statistics_sha256": ct["dataset_statistics_sha256"],
                "config_summary": ct["config_summary"],
            },
            "C_I": {
                "definition": "Stage21 _build_processor + original screen VLAStepData path; no model construction",
                "processor": ci["processor_summary"],
            },
            "parity_rows": parity_rows,
            "forbidden_condition_paths": sorted(set(input_forbidden)),
            "training_label_comparison": "C_T collated normalized132/action_mask was compared to sealed expert_normalized_132/action_mask; C_I uses the sealed normalized label only for decoder round-trip, never as model input.",
        },
        "selected_checkpoint": "NONE",
        "forbidden_counts": {"model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0, "backward": 0, "optimizer_steps": 0, "training_steps": 0, "wbc_steps": 0, "sim_steps": 0, "calib_queries": 0, "dev_e_reads": 0, "final_f_reads": 0},
        "admission": {"P0": "PASS_READ_ONLY", "P1": "AUTHORIZED_ONLY_AFTER_P0", "next": "P1 fixed bridge4 paired diagnostics"},
    }
    write_json(args.output_root / "pi0242_evidence_index.json", evidence)
    write_json(args.output_root / "pi0242_p0_admission.json", {"schema": "PI0242_P0_ADMISSION_V1", "status": "PASS", "evidence_index_sha256": sha256_file(args.output_root / "pi0242_evidence_index.json"), "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0})
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("p0", "p1", "r1"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg-root", type=Path)
    parser.add_argument("--torch-site", type=Path)
    parser.add_argument("--prior-p1-root", type=Path)
    args = parser.parse_args()
    if args.phase == "p0":
        result = p0(args)
    elif args.phase == "p1":
        require(args.p0_root is not None, "P1 requires --p0-root")
        _bind_sealed_stage45b(args)
        result = p1(args)
    elif args.phase == "r1":
        require(args.p0_root is not None, "R1 requires --p0-root")
        require(args.prior_p1_root is not None, "R1 requires --prior-p1-root")
        _bind_sealed_stage45b(args)
        result = r1(args)
    else:
        raise AssertionError(args.phase)
    print(json.dumps({"phase": args.phase, "status": result["status"], "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
