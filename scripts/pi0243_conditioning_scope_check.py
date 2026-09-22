#!/usr/bin/env python3
"""Zero-model C_T versus current Stage-A screen conditioning check."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if hasattr(value, "data") and not isinstance(value, (str, bytes, dict, list, tuple)):
        return _leaves(value.data, prefix)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            result.update(_leaves(child, path))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_leaves(child, f"{prefix}/{index}"))
        return result
    return {}


def _meta(value: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        raw = np.ascontiguousarray(array).tobytes()
        return {"shape": list(array.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _numeric_array(value: Any) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if not (torch.is_floating_point(value) or torch.is_complex(value)):
            return None
        return value.detach().cpu().float().numpy()
    array = np.asarray(value)
    return array.astype(np.float64, copy=False) if np.issubdtype(array.dtype, np.number) else None


def _processor_summary(processor: Any, embodiment: Any, bridge: Any) -> dict[str, Any]:
    return bridge._processor_summary(processor, embodiment)


def _compare(ct: Mapping[str, Any], ci: Mapping[str, Any], bridge: Any) -> dict[str, Any]:
    ct_leaves = _leaves(ct)
    ci_leaves = _leaves(ci)
    paths = sorted(set(ct_leaves) | set(ci_leaves))
    rows = []
    for path in paths:
        left = ct_leaves.get(path)
        right = ci_leaves.get(path)
        row: dict[str, Any] = {"path": path, "ct_present": left is not None, "ci_present": right is not None}
        if left is None or right is None:
            rows.append(row)
            continue
        row["ct"] = _meta(left)
        row["ci"] = _meta(right)
        row["same_bytes"] = row["ct"]["dtype"] == row["ci"]["dtype"] and row["ct"]["shape"] == row["ci"]["shape"] and row["ct"]["sha256"] == row["ci"]["sha256"]
        left_num = _numeric_array(left)
        right_num = _numeric_array(right)
        if left_num is not None and right_num is not None and left_num.shape == right_num.shape:
            delta = right_num.astype(np.float64) - left_num.astype(np.float64)
            row["max_abs"] = float(np.max(np.abs(delta))) if delta.size else 0.0
            row["rmse"] = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
        rows.append(row)
    shared = [row for row in rows if row["ct_present"] and row["ci_present"]]
    return {
        "ct_leaf_count": len(ct_leaves),
        "ci_leaf_count": len(ci_leaves),
        "paths": rows,
        "missing_in_ci": [row["path"] for row in rows if row["ct_present"] and not row["ci_present"]],
        "missing_in_ct": [row["path"] for row in rows if row["ci_present"] and not row["ct_present"]],
        "same_bytes_count": sum(bool(row.get("same_bytes")) for row in shared),
        "shared_count": len(shared),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--output-subdir", default="conditioning_scope")
    args = parser.parse_args()
    card = read_json(args.run_card)
    root = Path(card["paths"]["output_root"])
    out = root / args.output_subdir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""})
    code_root = Path(card["paths"]["code_root"])
    stage10_root = Path(card["paths"]["stage10_root"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(stage10_root / "src" / "isaac-gr00t")]

    legacy = load_module(code_root / "scripts" / "pi0241_fixed16_learning.py", "pi0243_scope_legacy")
    legacy.bind_runtime(card)
    stage = load_module(code_root / "scripts" / "pi0243_staged_learning_to_sim.py", "pi0243_scope_stage")
    pi0228 = load_module(code_root / "scripts" / "pi0228_h200_large_adapt_v1.py", "pi0243_scope_pi0228")
    bridge = load_module(code_root / "scripts" / "pi0242_train_infer_bridge.py", "pi0243_scope_bridge")
    stage21 = load_module(code_root / "scripts" / "stage21_same_scene_groot_inference.py", "pi0243_scope_stage21")

    cfg_dir = out / "current_a_config"
    training_boundary = out / "current_a_training_boundary"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    legacy.install_fixed_dataset_factory(card, training_boundary)
    config, resolved = stage.resolved_config(card, cfg_dir, legacy, pi0228)
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline(config, cfg_dir)
    train_dataset, eval_dataset = pipeline._create_dataset(cfg_dir)
    if eval_dataset is not None:
        raise RuntimeError("current A pipeline unexpectedly created eval dataset")
    dataset = train_dataset.datasets[0]
    ct_processor = pipeline.processor
    if dataset.processor is not ct_processor:
        raise RuntimeError("official training pipeline did not bind its processor to the source dataset")

    config_embodiment = config.data.datasets[0].embodiment_tag
    modality = config.data.modality_configs[config_embodiment]
    original_loader = stage21._load_modality
    stage21._load_modality = lambda _path, _stage10_root: modality
    try:
        ci_processor, ci_embodiment = stage21._build_processor(
            stage10_root,
            Path(card["paths"]["processor_base_model"]),
            Path(card["paths"]["train_dataset"]) / "meta",
            Path(card["paths"]["modality_config"]),
        )
    finally:
        stage21._load_modality = original_loader

    rows = read_json(Path(card["cohort"]["path"]))["rows"]
    records = []
    for row in rows:
        source_id = str(row["source_episode_id"])
        episode = dataset.episode_loader[int(row["episode_index"])]
        datapoint = dataset.get_datapoint(episode, 0)
        ct_batch = ct_processor.collator([datapoint])
        ct_policy = bridge._policy_inputs(ct_batch)
        state_full = np.asarray(datapoint["state"]).reshape(-1)
        state = state_full[:10].astype(np.float32)
        images = datapoint["vlm_content"]["images"]
        body = stage._image_hwc(images[0])
        wrist = stage._image_hwc(images[1])
        message = VLAStepData(
            images={"external_rgb": body[None, ...], "wrist_rgb": wrist[None, ...]},
            states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]},
            actions={}, text=stage.LANGUAGE, embodiment=ci_embodiment,
        )
        ci_processed = ci_processor([{"type": MessageType.EPISODE_STEP.value, "content": message}])
        ci_batch = ci_processor.collator([ci_processed])
        ci_policy = bridge._policy_inputs(ci_batch)
        comparison = _compare(ct_policy, ci_policy, bridge)
        forbidden = sorted(path for path in set(_leaves(ct_policy)) | set(_leaves(ci_policy)) if any(token in path.lower() for token in bridge.FORBIDDEN_INPUT_TOKENS))
        records.append({
            "source_episode_id": source_id,
            "episode_index": int(row["episode_index"]),
            "target_family": row.get("target_family"),
            "ct_processor": _processor_summary(ct_processor, config_embodiment, bridge),
            "ci_processor": _processor_summary(ci_processor, ci_embodiment, bridge),
            "ct_inventory": bridge._leaf_inventory(ct_policy),
            "ci_inventory": bridge._leaf_inventory(ci_policy),
            "comparison": comparison,
            "forbidden_model_input_paths": forbidden,
            "target_xyz_in_model_input": bool(forbidden),
            "model_activity": {"weight_loads": 0, "forward": 0, "get_action": 0, "backward": 0, "optimizer": 0},
        })

    all_paths = [row for record in records for row in record["comparison"]["paths"]]
    full_match = all(
        not record["comparison"]["missing_in_ci"]
        and not record["comparison"]["missing_in_ct"]
        and all(bool(item.get("same_bytes")) for item in record["comparison"]["paths"])
        and not record["forbidden_model_input_paths"]
        for record in records
    )
    report = {
        "schema": "PI0243_STAGE_A_CONDITIONING_SCOPE_V1",
        "status": "FULL_CONDITIONING_MATCH" if full_match else "CONDITIONING_DIFF_OBSERVED",
        "model_activity": {"weight_loads": 0, "forward": 0, "get_action": 0, "backward": 0, "optimizer": 0, "WBC": 0, "sim": 0},
        "selected_checkpoint": "NONE",
        "run_card_sha256": sha256_file(args.run_card),
        "code_sha256": {"stage": sha256_file(code_root / "scripts" / "pi0243_staged_learning_to_sim.py"), "stage21": sha256_file(code_root / "scripts" / "stage21_same_scene_groot_inference.py"), "bridge_helpers": sha256_file(code_root / "scripts" / "pi0242_train_infer_bridge.py")},
        "resolved_config": resolved,
        "cohort_sha256": card["cohort"]["sha256"],
        "count": len(records),
        "records": records,
        "aggregate": {"path_count": len(all_paths), "all_forbidden_paths": sorted({path for record in records for path in record["forbidden_model_input_paths"]})},
    }
    report_path = out / "stage_a_conditioning_scope_check.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "path": str(report_path), "sha256": sha256_file(report_path), "count": len(records), "model_activity": report["model_activity"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
