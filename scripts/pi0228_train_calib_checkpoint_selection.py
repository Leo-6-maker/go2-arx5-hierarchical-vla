#!/usr/bin/env python3
"""PI0228 TRAIN-CALIB model-reference evaluation and frozen selection.

This runner only evaluates decoded dense-relative references on TRAIN-CALIB.
It never imports WBC, starts a simulator, reads DEV-E/F, or sends a reference
to execution.  The checkpoint choice is frozen before any physical evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_STEPS = (2500, 5000, 7500, 10000)
SEED = 22301
ACTION_DT_S = 0.05
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
LANGUAGE = "move the end effector to the red target"
MODEL_INPUT_KEYS = ["body_rgb", "wrist_rgb", "state"]
TARGET_AUTHORITY_SCHEMA = "PI0228_TRAIN_CALIB_TARGET_AUTHORITY_V1"
PI0227_GRID_MANIFEST_SHA256 = "d6e6e51ebf7b9e9a43cb691e8896fa2ddfe10de43f936aede4fa2898ac3f2bc4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        return {"count": int(array.size), "median": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _reject_locked_path(path: Path) -> None:
    lowered = str(path).lower()
    forbidden = ("dev-e", "dev_e", "holdout-f", "holdout_f", "final-f", "final_f")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"locked DEV/F path is not admissible: {path}")


def _load_calib_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    _reject_locked_path(dataset_dir)
    lock_path = dataset_dir / "meta" / "pi0227_train_only_split_lock.json"
    rows_path = dataset_dir / "meta" / "episodes.jsonl"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "PI0227_TRAIN_ONLY_SOURCE_ROLLOUT_SPLIT_V1":
        raise ValueError("TRAIN-CALIB split-lock schema drift")
    if lock.get("role") != "TRAIN_CALIB" or lock.get("training_allowed") is not False:
        raise ValueError("selection requires the non-training TRAIN-CALIB split")
    if lock.get("e_source_read_count") != 0 or lock.get("f_source_read_count") != 0:
        raise ValueError("TRAIN-CALIB E/F read count is nonzero")
    if lock.get("target_xyz_in_model_input") is not False:
        raise ValueError("TRAIN-CALIB target-input boundary drift")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("TRAIN-CALIB episode indices are not contiguous")
    if len({row["source_episode_id"] for row in rows}) != len(rows):
        raise ValueError("TRAIN-CALIB source episode IDs are duplicated")
    return rows


def _source_rollout_id(source_episode_id: str) -> str:
    rollout_id, separator, _ = source_episode_id.partition(":obs-")
    if not separator or not rollout_id:
        raise ValueError(f"invalid source episode ID: {source_episode_id}")
    return rollout_id


def _load_target_authority(path: Path) -> tuple[dict[str, Any], dict[str, tuple[str, np.ndarray]]]:
    _reject_locked_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != TARGET_AUTHORITY_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("TRAIN-CALIB target authority is not PASS")
    if payload.get("role") != "TRAIN_CALIB":
        raise ValueError("target authority is not bound to TRAIN-CALIB")
    if payload.get("source_grid_manifest_sha256") != PI0227_GRID_MANIFEST_SHA256:
        raise ValueError("target authority is not bound to the frozen PI0227 grid manifest")
    if payload.get("target_xyz_in_model_input") is not False:
        raise ValueError("target authority boundary drift")
    entries = payload.get("rollouts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("target authority has no rollout entries")
    authority: dict[str, tuple[str, np.ndarray]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid target authority rollout entry")
        rollout_id = str(entry.get("source_rollout_id", ""))
        target_id = str(entry.get("target_id", ""))
        target = np.asarray(entry.get("target_position_world_m"), dtype=np.float64)
        if not rollout_id or not target_id or rollout_id in authority:
            raise ValueError(f"invalid or duplicate target authority rollout: {rollout_id}")
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError(f"invalid target_position_world_m for {rollout_id}")
        authority[rollout_id] = (target_id, target)
    expected_count = payload.get("source_rollout_count")
    if expected_count is not None and int(expected_count) != len(authority):
        raise ValueError("target authority rollout count mismatch")
    return payload, authority


def _target_for(
    source_episode_id: str, authority: dict[str, tuple[str, np.ndarray]]
) -> tuple[str, np.ndarray]:
    rollout_id = _source_rollout_id(source_episode_id)
    try:
        return authority[rollout_id]
    except KeyError as exc:
        raise ValueError(f"missing authoritative target for {rollout_id}") from exc


def _build_chunk(decoded: np.ndarray, *, chunk_id: str):
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider

    provider = GR00TTrajectoryProvider(
        decoded,
        config=GR00TProviderConfig(
            source_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_SEMANTIC_ID,
            action_sample_period_s=ACTION_DT_S,
            timestamp_provenance="PI0228_TRAIN_CALIB_SELECTION",
        ),
        observation_timestamp=0.0,
        chunk_id=chunk_id,
        provenance={"target_xyz_in_model_input": False, "wbc_execution": False},
    )
    chunk = provider.next_chunk(0.0)
    if chunk is None:
        raise RuntimeError("dense decoder emitted no H40 chunk")
    return chunk


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _reject_locked_path(args.train_dataset)
    _reject_locked_path(args.calib_dataset)
    rows = _load_calib_rows(args.calib_dataset)
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    if seal.get("schema") != "PI0228_TRAIN_CALIB_SEAL_V1" or seal.get("status") != "PASS":
        raise ValueError("TRAIN-CALIB seal is not PASS")
    sealed_root = Path(seal["train_calib"]["root"]).resolve()
    if sealed_root != args.calib_dataset.resolve():
        raise ValueError("evaluation dataset does not match sealed TRAIN-CALIB root")
    if args.seed != SEED:
        raise ValueError(f"PI0228 selection seed is frozen at {SEED}")
    authority_payload, target_authority = _load_target_authority(args.target_authority)
    if authority_payload.get("calib_seal_sha256") != sha256_file(args.seal):
        raise ValueError("target authority is bound to a different TRAIN-CALIB seal")
    if authority_payload.get("train_calib_tree_sha256") != seal["train_calib"].get("tree_sha256"):
        raise ValueError("target authority is bound to a different TRAIN-CALIB tree")
    source_rollouts = {_source_rollout_id(str(row["source_episode_id"])) for row in rows}
    authority_rollouts = set(target_authority)
    if source_rollouts != authority_rollouts:
        missing = sorted(source_rollouts - authority_rollouts)
        extra = sorted(authority_rollouts - source_rollouts)
        raise ValueError(f"target authority rollout mismatch: missing={missing[:3]} extra={extra[:3]}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    from stage20s_infer_groot import _build_processor, _read_first_frame, _run_one
    from stage40_b0_model_control import _bind_local_cosmos_processor
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel
    import pandas as pd

    _bind_local_cosmos_processor(args.stage10_root, args.cosmos_snapshot)
    processor, embodiment = _build_processor(
        args.stage10_root, args.base_model, args.train_dataset / "meta", args.modality_config
    )
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model.eval().to(device=args.device, dtype=__import__("torch").bfloat16)
    gate = TrajectorySafetyGateV1()
    records: list[dict[str, Any]] = []
    forward_count = 0
    for row in rows:
        index = int(row["episode_index"])
        chunk = index // 1000
        stem = f"episode_{index:06d}.mp4"
        body_path = args.calib_dataset / "videos" / f"chunk-{chunk:03d}" / "observation.images.external_rgb" / stem
        wrist_path = args.calib_dataset / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist_rgb" / stem
        parquet_path = args.calib_dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
        table = pd.read_parquet(parquet_path)
        state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError(f"invalid StateV1 at episode {index}")
        body = _read_first_frame(body_path)[0]
        wrist = _read_first_frame(wrist_path)[0]
        observation = {"body_rgb": body, "wrist_rgb": wrist, "state": state}
        raw, decoded, latency, gpu = _run_one(
            model, processor, embodiment, observation,
            language=LANGUAGE, seed=args.seed, device=args.device,
        )
        forward_count += 1
        source_rollout_id = _source_rollout_id(str(row["source_episode_id"]))
        target_id, target = _target_for(str(row["source_episode_id"]), target_authority)
        chunk_value = _build_chunk(decoded, chunk_id=f"pi0228-calib-{index:04d}")
        state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
        gate_result = gate.evaluate(
            chunk_value, now=0.0, achieved_position=state[:3], achieved_orientation=state_quat
        )
        distances = np.linalg.norm(decoded[0, :, :3].astype(np.float64) - target[None, :], axis=1)
        records.append({
            "episode_index": index,
            "source_episode_id": str(row["source_episode_id"]),
            "source_rollout_id": source_rollout_id,
            "target_id": target_id,
            "target_world_m": target.tolist(),
            "safety_gate": gate_result.as_dict(),
            "reference_min_target_distance_m": float(np.min(distances)),
            "reference_min_target_tick": int(np.argmin(distances)),
            "raw_sha256": sha256_array(raw),
            "decoded_sha256": sha256_array(decoded),
            "latency_s": float(latency),
            "gpu": gpu,
            "model_input_keys": MODEL_INPUT_KEYS,
            "target_xyz_in_model_input": False,
            "wbc_execution": False,
        })
    rejects = [item for item in records if item["safety_gate"]["decision"] == "REJECT"]
    accepted_distances = [item["reference_min_target_distance_m"] for item in records if item not in rejects]
    by_cell: dict[str, list[float]] = defaultdict(list)
    for item in records:
        if item not in rejects:
            by_cell[item["source_rollout_id"]].append(item["reference_min_target_distance_m"])
    cell_medians = {key: float(np.median(value)) for key, value in by_cell.items()}
    summary = {
        "schema": "PI0228_TRAIN_CALIB_PROFILE_V1",
        "status": "PASS" if not rejects else "SAFETY_GATE_REJECT",
        "checkpoint_step": int(args.checkpoint_step),
        "checkpoint_path": str(args.model_path),
        "checkpoint_index_sha256": sha256_file(next(args.model_path.glob("*.safetensors.index.json"))),
        "seed": SEED,
        "action_contract": "official_dense_relative_H40",
        "action_dt_s": ACTION_DT_S,
        "calib_seal_sha256": sha256_file(args.seal),
        "target_authority_sha256": sha256_file(args.target_authority),
        "target_authority_source_grid_manifest_sha256": authority_payload["source_grid_manifest_sha256"],
        "model_input_keys": MODEL_INPUT_KEYS,
        "target_xyz_in_model_input": False,
        "wbc_execution": False,
        "forward_count": forward_count,
        "expected_forward_count": len(rows),
        "safety_gate_counts": {
            decision: sum(item["safety_gate"]["decision"] == decision for item in records)
            for decision in ("ACCEPT", "WARN", "REJECT")
        },
        "reference_min_target_distance_m": _stats(accepted_distances),
        "worst_cell_median_reference_min_target_distance_m": None if not cell_medians else max(cell_medians.values()),
        "cell_medians_reference_min_target_distance_m": cell_medians,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def select(args: argparse.Namespace) -> dict[str, Any]:
    if list(args.profile_steps) != list(EXPECTED_STEPS):
        raise ValueError(f"selection requires exactly checkpoint steps {EXPECTED_STEPS}")
    profiles = []
    for path, expected_step in zip(args.profiles, args.profile_steps):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "PI0228_TRAIN_CALIB_PROFILE_V1":
            raise ValueError(f"profile schema drift: {path}")
        if int(payload.get("checkpoint_step", -1)) != expected_step:
            raise ValueError(f"profile step mismatch: {path}")
        profiles.append((payload, path))
    admissible = [
        (payload, path) for payload, path in profiles
        if payload.get("status") == "PASS"
        and payload.get("safety_gate_counts", {}).get("REJECT", 1) == 0
        and payload.get("forward_count") == payload.get("expected_forward_count")
    ]
    if not admissible:
        raise RuntimeError("no TRAIN-CALIB checkpoint is admissible")
    ranked = sorted(
        admissible,
        key=lambda pair: (
            float(pair[0]["reference_min_target_distance_m"]["median"]),
            float(pair[0]["worst_cell_median_reference_min_target_distance_m"]),
            int(pair[0]["checkpoint_step"]),
        ),
    )
    selected, selected_path = ranked[0]
    result = {
        "schema": "PI0228_TRAIN_CALIB_SELECTION_V1",
        "status": "PASS",
        "selection_frozen_before_wbc": True,
        "dev_e_read_count": 0,
        "final_f_read_count": 0,
        "wbc_execution": False,
        "rule": {
            "exclude_any_safety_gate_reject": True,
            "primary": "lowest median reference min target distance over admissible TRAIN-CALIB windows",
            "tie_break": "lowest worst source-rollout-cell median reference min target distance",
            "final_tie_break": "lowest checkpoint step",
        },
        "candidate_profiles": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_step": int(payload["checkpoint_step"]),
                "status": payload.get("status"),
                "safety_gate_counts": payload.get("safety_gate_counts"),
                "median_reference_min_target_distance_m": payload.get("reference_min_target_distance_m", {}).get("median"),
                "worst_cell_median_reference_min_target_distance_m": payload.get("worst_cell_median_reference_min_target_distance_m"),
            }
            for payload, path in profiles
        ],
        "selected_checkpoint_step": int(selected["checkpoint_step"]),
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_profile_sha256": sha256_file(selected_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _self_test() -> dict[str, Any]:
    import tempfile

    nominal = "pi0227_persistent_train_a_j00_nominal_s00_c0_r00_r00_227000"
    jittered = "pi0227_persistent_train_b_j01_xp015_s00_c0_r00_r00_227036"
    with tempfile.TemporaryDirectory(prefix="pi0228-target-authority-") as directory:
        path = Path(directory) / "authority.json"
        path.write_text(
            json.dumps(
                {
                    "schema": TARGET_AUTHORITY_SCHEMA,
                    "status": "PASS",
                    "role": "TRAIN_CALIB",
                    "source_grid_manifest_sha256": PI0227_GRID_MANIFEST_SHA256,
                    "target_xyz_in_model_input": False,
                    "source_rollout_count": 2,
                    "rollouts": [
                        {
                            "source_rollout_id": nominal,
                            "target_id": "red_target_train_a_j00_nominal",
                            "target_position_world_m": [0.30, 0.10, 0.58],
                        },
                        {
                            "source_rollout_id": jittered,
                            "target_id": "red_target_train_b_j01_xp015",
                            "target_position_world_m": [0.315, -0.10, 0.58],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        _, authority = _load_target_authority(path)
        nominal_id, nominal_target = _target_for(nominal + ":obs-0000", authority)
        jitter_id, jitter_target = _target_for(jittered + ":obs-0000", authority)
        assert nominal_id.endswith("j00_nominal")
        assert jitter_id.endswith("j01_xp015")
        assert np.allclose(nominal_target, [0.30, 0.10, 0.58])
        assert np.allclose(jitter_target, [0.315, -0.10, 0.58])
        assert not np.allclose(jitter_target, nominal_target)
        assert MODEL_INPUT_KEYS == ["body_rgb", "wrist_rgb", "state"]
    return {
        "status": "PASS",
        "nominal_and_non_nominal_rollouts": 2,
        "target_xyz_in_model_input": False,
        "model_input_keys": MODEL_INPUT_KEYS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate", "select", "self-test"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage10-root", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--calib-dataset", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--cosmos-snapshot", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--target-authority", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--profiles", type=Path, nargs="*")
    parser.add_argument("--profile-steps", type=int, nargs="*")
    args = parser.parse_args()
    if args.mode == "self-test":
        result = _self_test()
    elif args.mode == "evaluate":
        required = ("stage10_root", "base_model", "model_path", "train_dataset", "calib_dataset", "modality_config", "cosmos_snapshot", "seal", "target_authority", "checkpoint_step", "output")
        if any(getattr(args, name) is None for name in required):
            parser.error("evaluate requires stage10/model/dataset/modality/Cosmos/seal/target-authority/checkpoint/output inputs")
        result = evaluate(args)
    else:
        if len(args.profiles or []) != 4 or len(args.profile_steps or []) != 4 or args.output is None:
            parser.error("select requires four --profiles and four --profile-steps values")
        result = select(args)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
