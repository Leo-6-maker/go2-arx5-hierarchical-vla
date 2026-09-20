#!/usr/bin/env python3
"""PI0240 fixed-cohort G4 diagnostic; no training or WBC execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SEED = 22301
LANGUAGE = "move the end effector to the red target"
DT = 0.05


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": None if not array.size else float(array.mean()),
        "rmse": None if not array.size else float(np.sqrt(np.mean(array ** 2))),
        "p95": None if not array.size else float(np.percentile(array, 95)),
        "max": None if not array.size else float(array.max()),
    }


def target_map(authority: dict[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for row in authority["rollouts"]:
        key = (str(row["anchor_id"]).removeprefix("train_"), str(row["jitter_id"]))
        value = np.asarray(row["target_position_world_m"], dtype=np.float64)
        if key in result and not np.array_equal(result[key], value):
            raise RuntimeError(f"target authority conflict: {key}")
        result[key] = value
    return result


def find_action_and_mask(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    action = mask = None

    def visit(value: Any, name: str = "") -> None:
        nonlocal action, mask
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{name}.{key}" if name else str(key))
        elif hasattr(value, "shape") and tuple(int(x) for x in value.shape) == (1, 40, 132):
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if "mask" in name.lower():
                mask = array
            elif action is None:
                action = array

    visit(batch)
    if action is None or mask is None:
        raise RuntimeError("collator did not expose [1,40,132] action and mask")
    return action, mask


def metric_bundle(model: np.ndarray, expert: np.ndarray, modules: dict[str, Any]) -> dict[str, Any]:
    xyz = np.linalg.norm(model[:, :3] - expert[:, :3], axis=1)
    so3 = modules["geodesic_angle"](
        modules["rot6d_rows_to_matrix"](model[:, 3:9]),
        modules["rot6d_rows_to_matrix"](expert[:, 3:9]),
    )
    gripper = np.abs(model[:, 9] - expert[:, 9])
    model_linear = np.linalg.norm(np.diff(model[:, :3], axis=0), axis=1) / DT
    expert_linear = np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1) / DT
    model_angular = modules["geodesic_angle"](
        modules["rot6d_rows_to_matrix"](model[:-1, 3:9]),
        modules["rot6d_rows_to_matrix"](model[1:, 3:9]),
    ) / DT
    expert_angular = modules["geodesic_angle"](
        modules["rot6d_rows_to_matrix"](expert[:-1, 3:9]),
        modules["rot6d_rows_to_matrix"](expert[1:, 3:9]),
    ) / DT
    return {
        "xyz_m": {**stats(xyz.tolist()), "first": float(xyz[0]), "endpoint": float(xyz[-1])},
        "so3_rad": {**stats(np.asarray(so3).tolist()), "first": float(so3[0]), "endpoint": float(so3[-1])},
        "gripper_abs": {**stats(gripper.tolist()), "first": float(gripper[0]), "endpoint": float(gripper[-1])},
        "velocity": {
            "model_linear_mps": stats(model_linear.tolist()), "expert_linear_mps": stats(expert_linear.tolist()),
            "model_angular_radps": stats(np.asarray(model_angular).tolist()), "expert_angular_radps": stats(np.asarray(expert_angular).tolist()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-role", choices=("baseline1000", "fresh10000"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--calib-dataset", type=Path)
    parser.add_argument("--p237-root", type=Path)
    parser.add_argument("--stage10-root", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--cosmos-snapshot", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.self_test:
        assert stats([3.0, 4.0])["max"] == 4.0
        assert target_map({"rollouts": [{"anchor_id": "train_a", "jitter_id": "j00", "target_position_world_m": [1, 2, 3]}]})[("a", "j00")].tolist() == [1, 2, 3]
        print("PASS")
        return 0
    required = (args.model_role, args.model_path, args.baseline, args.train_dataset, args.calib_dataset, args.p237_root, args.stage10_root, args.code_root, args.modality_config, args.cosmos_snapshot, args.output_root)
    if any(value is None for value in required):
        parser.error("G4 execution requires all path and model arguments")
    for path in (args.model_path, args.baseline, args.train_dataset, args.calib_dataset, args.p237_root):
        if any(token in str(path).lower() for token in ("dev-e", "dev_e", "final-f", "final_f", "holdout-f", "holdout_f")):
            raise RuntimeError(f"locked dataset/model path: {path}")

    sys.path[:0] = [str(args.code_root / "scripts"), str(args.code_root / "src"), str(args.stage10_root / "src" / "isaac-gr00t")]
    import torch
    import pandas as pd
    import pi0238_audit_to_sim as mature
    import pi0228_train_calib_checkpoint_selection as selector
    import stage21_same_scene_groot_inference as stage21
    import stage40_b0_model_control as stage40
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    if SEED != 22301 or LANGUAGE != selector.LANGUAGE:
        raise RuntimeError("frozen G4 contract drift")
    args.output_root.mkdir(parents=True, exist_ok=True)
    role_root = args.output_root / args.model_role
    role_root.mkdir(parents=True, exist_ok=True)
    if args.model_role == "fresh10000":
        baseline_summary = args.output_root / "baseline1000" / "summary.json"
        if not baseline_summary.is_file() or json.loads(baseline_summary.read_text()).get("completed_queries") != 128:
            raise RuntimeError("fresh10000 requires the frozen complete baseline1000 result")
    cohort = json.loads((args.p237_root / "evidence" / "pi0237_cohort.json").read_text(encoding="utf-8"))
    ordered = [("CALIB", args.calib_dataset, row) for row in sorted(cohort["calib"], key=lambda x: int(x["episode_index"]))]
    ordered += [("TRAIN", args.train_dataset, row) for row in sorted(cohort["train"], key=lambda x: int(x["episode_index"]))]
    if len(ordered) != 128:
        raise RuntimeError(f"fixed cohort drift: {len(ordered)}")
    authority = json.loads((args.p237_root / "evidence" / "pi0228_train_calib_target_authority.json").read_text(encoding="utf-8"))
    targets = target_map(authority)
    modality, modality_identity = mature.bind_modality_source(args)
    modality_config = getattr(modality, "GO2_ARX5_MODALITY_CONFIG", getattr(modality, "SIM_VISUAL_REACH_MODALITY_CONFIG", None))
    if modality_config is None:
        raise RuntimeError("modality config missing")
    stage40._bind_local_cosmos_processor(args.stage10_root, args.cosmos_snapshot)
    processor, embodiment = stage21._build_processor(args.stage10_root, args.baseline, args.train_dataset / "meta", args.modality_config)
    datasets = {
        str(root): ShardedSingleStepDataset(root, EmbodimentTag.NEW_EMBODIMENT, modality_config, shard_size=128, episode_sampling_rate=1.0, seed=SEED, allow_padding=False)
        for root in (args.train_dataset, args.calib_dataset)
    }
    for dataset in datasets.values():
        dataset.processor = processor
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    gate = __import__("go2arx5_vla.safety", fromlist=["TrajectorySafetyGateV1"]).TrajectorySafetyGateV1()
    modules = {
        "geodesic_angle": __import__("go2arx5_vla.trajectory.so3", fromlist=["geodesic_angle"]).geodesic_angle,
        "rot6d_rows_to_matrix": __import__("go2arx5_vla.trajectory.so3", fromlist=["rot6d_rows_to_matrix"]).rot6d_rows_to_matrix,
        "matrix_to_quat": __import__("go2arx5_vla.trajectory.so3", fromlist=["matrix_to_quat"]).matrix_to_quat,
    }
    ledger = args.output_root / "attempts.jsonl"
    records_path = role_root / "records.jsonl"
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); completed[(row["split"], int(row["episode_index"]))] = row
    started = time.time()
    for split, root, item in ordered:
        index = int(item["episode_index"]); key = (split, index)
        if key in completed:
            continue
        attempt = {"event": "ATTEMPTED", "model_role": args.model_role, "split": split, "episode_index": index, "source_episode_id": item["source_episode_id"], "seed": SEED, "time": time.time()}
        append_jsonl(ledger, attempt)
        dataset = datasets[str(root)]
        datapoint = dataset.get_datapoint(dataset.episode_loader[index], 0)
        expert132, mask = find_action_and_mask(processor.collator([datapoint]))
        state, expert, timestamps = mature.load_window(root, item)
        chunk = index // 1000; stem = f"episode_{index:06d}.mp4"
        body_path = root / "videos" / f"chunk-{chunk:03d}" / "observation.images.external_rgb" / stem
        wrist_path = root / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist_rgb" / stem
        body = stage21._read_first_frame(body_path)[0]; wrist = stage21._read_first_frame(wrist_path)[0]
        observation = {"body_rgb": body, "wrist_rgb": wrist, "state": state}
        raw, decoded, latency, gpu = stage21._run_one(model, processor, embodiment, observation, language=LANGUAGE, seed=SEED, device=args.device)
        if raw.shape != (1, 40, 132) or decoded.shape != (1, 40, 10) or not np.isfinite(raw).all() or not np.isfinite(decoded).all():
            raise RuntimeError(f"illegal model output at {args.model_role}/{split}/{index}")
        state_quat = modules["matrix_to_quat"](modules["rot6d_rows_to_matrix"](state[3:9].reshape(1, 6)))[0]
        model_gate = gate.evaluate(selector._build_chunk(decoded, chunk_id=f"pi0240-g4-{args.model_role}-{split.lower()}-{index:05d}"), now=0.0, achieved_position=state[:3], achieved_orientation=state_quat)
        expert_gate = gate.evaluate(selector._build_chunk(expert[None, ...], chunk_id=f"pi0240-g4-expert-{split.lower()}-{index:05d}"), now=0.0, achieved_position=state[:3], achieved_orientation=state_quat)
        jitter = str(item["jitter_family"]).split("_", 1)[0]
        target = targets[(str(item["target_family"]), jitter)]
        tensor_path = role_root / "cells" / f"{split.lower()}-{index:06d}.npz"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tensor_path, raw=raw, decoded=decoded, state=state, expert=expert, timestamps=timestamps, expert_normalized_132=expert132, action_mask=mask, body_rgb=body, wrist_rgb=wrist)
        model_issues = [issue.code for issue in model_gate.issues]; expert_issues = [issue.code for issue in expert_gate.issues]
        record = {
            "schema": "PI0240_G4_CELL_V1", "model_role": args.model_role, "split": split, "episode_index": index,
            "source_episode_id": item["source_episode_id"], "source_rollout_id": item["source_rollout_id"], "target_family": item["target_family"], "jitter_family": item["jitter_family"], "phase": item["phase"],
            "input": {"keys": ["body_rgb", "wrist_rgb", "state"], "body_sha256": sha256_array(body), "wrist_sha256": sha256_array(wrist), "state_sha256": sha256_array(state), "target_xyz_in_model_input": False},
            "tensor": {"path": str(tensor_path), "raw_sha256": sha256_array(raw), "decoded_sha256": sha256_array(decoded), "expert_sha256": sha256_array(expert), "mask_sha256": sha256_array(mask), "active_slots": mature.active_mask_slots(mask)[0]},
            "fit": metric_bundle(decoded[0].astype(np.float64), expert.astype(np.float64), modules),
            "target_min_distance_m": float(np.min(np.linalg.norm(decoded[0, :, :3].astype(np.float64) - target[None, :], axis=1))),
            "model_gate": model_gate.as_dict(), "expert_gate": expert_gate.as_dict(), "model_extra_issues": sorted(set(model_issues) - set(expert_issues)),
            "latency_s": float(latency), "gpu": gpu, "seed": SEED, "language": LANGUAGE, "action_contract": "official_dense_relative_H40", "action_dt_s": DT,
        }
        append_jsonl(records_path, record); completed[key] = record
        append_jsonl(ledger, {"event": "COMPLETE", "model_role": args.model_role, "split": split, "episode_index": index, "raw_sha256": record["tensor"]["raw_sha256"], "decoded_sha256": record["tensor"]["decoded_sha256"], "time": time.time()})
    records = [completed[(split, int(item["episode_index"]))] for split, _, item in ordered]
    summary = {
        "schema": "PI0240_G4_MODEL_SUMMARY_V1", "status": "COMPLETE", "model_role": args.model_role, "checkpoint": str(args.model_path),
        "checkpoint_files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(args.model_path.glob("model*.safetensors*"))},
        "cohort_count": len(records), "attempted_queries": sum(1 for line in ledger.read_text().splitlines() if json.loads(line).get("event") == "ATTEMPTED" and json.loads(line).get("model_role") == args.model_role),
        "completed_queries": len(records), "order": "CALIB_ASC_THEN_TRAIN_ASC", "modality": modality_identity,
        "gate_counts": dict(Counter(row["model_gate"]["decision"] for row in records)),
        "expert_gate_counts": dict(Counter(row["expert_gate"]["decision"] for row in records)),
        "xyz_mean_m": {split: stats([row["fit"]["xyz_m"]["mean"] for row in records if row["split"] == split]) for split in ("TRAIN", "CALIB")},
        "so3_mean_rad": {split: stats([row["fit"]["so3_rad"]["mean"] for row in records if row["split"] == split]) for split in ("TRAIN", "CALIB")},
        "gripper_mean": {split: stats([row["fit"]["gripper_abs"]["mean"] for row in records if row["split"] == split]) for split in ("TRAIN", "CALIB")},
        "model_extra_issue_counts": dict(Counter(issue for row in records for issue in row["model_extra_issues"])),
        "elapsed_s": time.time() - started, "selected_checkpoint": "NONE", "wbc_steps": 0, "training_backward_calls": 0, "optimizer_steps": 0, "dev_e_read_count": 0, "final_f_read_count": 0,
    }
    write_json(role_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
