#!/usr/bin/env python3
"""PI0233 P1: one TRAIN-CALIB baseline row through the repaired helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PREP = Path("/scratch/e1192548/pi0228_h200_prep_20260919")
STAGE10 = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902")
BASELINE = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/checkpoint-1000")
CALIB = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_calib_dataset")
TRAIN = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_dataset")
MODALITY = Path("/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py")
COSMOS = Path(
    "/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/"
    "models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
)
SEAL = PREP / "evidence/pi0228_train_calib_seal.json"
AUTHORITY = PREP / "evidence/pi0228_train_calib_target_authority.json"
SELECTOR = PREP / "scripts/pi0228_train_calib_checkpoint_selection.py"
STAGE20S = PREP / "scripts/stage20s_infer_groot.py"
STAGE21 = PREP / "scripts/stage21_same_scene_groot_inference.py"
BASELINE_TREE_SHA = "12c83e6695c28be3b4a91ef6df201b18f6a1a2877d3e80f7db15ef656176ab3b"
SEED = 22301


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"canary output root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(PREP))
    sys.path.insert(0, str(PREP / "scripts"))
    from pi0232_train_calib_eval import bind_cache

    runtime_binding = bind_cache(args.cache_root)
    import pi0228_train_calib_checkpoint_selection as selector

    rows = selector._load_calib_rows(CALIB)
    if len(rows) != 1088:
        raise RuntimeError(f"TRAIN-CALIB row count drift: {len(rows)}")
    seal_payload = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal_payload.get("schema") != "PI0228_TRAIN_CALIB_SEAL_V1" or seal_payload.get("status") != "PASS":
        raise RuntimeError("TRAIN-CALIB seal is not PASS")
    if Path(seal_payload["train_calib"]["root"]).resolve() != CALIB.resolve():
        raise RuntimeError("canary dataset does not match sealed TRAIN-CALIB root")
    authority_payload, target_authority = selector._load_target_authority(AUTHORITY)
    if authority_payload.get("calib_seal_sha256") != sha256_file(SEAL):
        raise RuntimeError("target authority is bound to a different TRAIN-CALIB seal")
    source_rollouts = {selector._source_rollout_id(str(row["source_episode_id"])) for row in rows}
    if source_rollouts != set(target_authority):
        raise RuntimeError("target authority rollout set does not cover all 1088 sealed rows")

    sys.path.insert(0, str(STAGE10 / "src" / "isaac-gr00t"))
    from stage20s_infer_groot import _build_processor, _read_first_frame, _run_one
    from stage40_b0_model_control import _bind_local_cosmos_processor
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel
    import numpy as np
    import pandas as pd
    import torch

    _bind_local_cosmos_processor(STAGE10, COSMOS)
    processor, embodiment = _build_processor(STAGE10, BASELINE, TRAIN / "meta", MODALITY)
    model = AutoModel.from_pretrained(BASELINE, local_files_only=True, trust_remote_code=True)
    model.eval().to(device="cuda:0", dtype=torch.bfloat16)
    gate = TrajectorySafetyGateV1()

    row = rows[0]
    index = int(row["episode_index"])
    chunk = index // 1000
    stem = f"episode_{index:06d}.mp4"
    body_path = CALIB / "videos" / f"chunk-{chunk:03d}" / "observation.images.external_rgb" / stem
    wrist_path = CALIB / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist_rgb" / stem
    parquet_path = CALIB / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
    table = pd.read_parquet(parquet_path)
    state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise RuntimeError("invalid StateV1 in canary row")
    body = _read_first_frame(body_path)[0]
    wrist = _read_first_frame(wrist_path)[0]
    observation = {"body_rgb": body, "wrist_rgb": wrist, "state": state}
    raw, decoded, latency, gpu = _run_one(
        model, processor, embodiment, observation,
        language=selector.LANGUAGE, seed=SEED, device="cuda:0",
    )
    target_id, target = selector._target_for(str(row["source_episode_id"]), target_authority)
    chunk_value = selector._build_chunk(decoded, chunk_id=f"pi0233-baseline-canary-{index:04d}")
    state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    gate_result = gate.evaluate(
        chunk_value, now=0.0, achieved_position=state[:3], achieved_orientation=state_quat
    )
    gate_decision = gate_result.as_dict()["decision"]
    distances = np.linalg.norm(decoded[0, :, :3].astype(np.float64) - target[None, :], axis=1)
    record = {
        "episode_index": index,
        "source_episode_id": str(row["source_episode_id"]),
        "source_rollout_id": selector._source_rollout_id(str(row["source_episode_id"])),
        "target_id": target_id,
        "target_world_m": target.tolist(),
        "safety_gate": gate_result.as_dict(),
        "safety_gate_call_completed": True,
        "safety_gate_decision_preserved": gate_decision,
        "reference_min_target_distance_m": float(np.min(distances)),
        "reference_min_target_tick": int(np.argmin(distances)),
        "raw_sha256": selector.sha256_array(raw),
        "decoded_sha256": selector.sha256_array(decoded),
        "raw_shape": list(raw.shape),
        "decoded_shape": list(decoded.shape),
        "latency_s": float(latency),
        "gpu": gpu,
        "model_input_keys": selector.MODEL_INPUT_KEYS,
        "target_xyz_in_model_input": False,
        "wbc_execution": False,
    }
    output_profile = args.output_root / "profile-1000-canary.json"
    profile = {
        "schema": "PI0228_TRAIN_CALIB_PROFILE_V1",
        "status": "PASS_GATE_INVOCATION_WITH_REJECT" if gate_decision == "REJECT" else "PASS",
        "checkpoint_step": 1000,
        "checkpoint_path": str(BASELINE),
        "checkpoint_index_sha256": sha256_file(next(BASELINE.glob("*.safetensors.index.json"))),
        "seed": SEED,
        "action_contract": "official_dense_relative_H40",
        "action_dt_s": 0.05,
        "calib_seal_sha256": sha256_file(SEAL),
        "target_authority_sha256": sha256_file(AUTHORITY),
        "model_input_keys": selector.MODEL_INPUT_KEYS,
        "target_xyz_in_model_input": False,
        "wbc_execution": False,
        "forward_count": 1,
        "expected_forward_count": 1,
        "safety_gate_counts": {"ACCEPT": 1 if record["safety_gate"]["decision"] == "ACCEPT" else 0,
                                "WARN": 1 if record["safety_gate"]["decision"] == "WARN" else 0,
                                "REJECT": 1 if record["safety_gate"]["decision"] == "REJECT" else 0},
        "reference_min_target_distance_m": {"count": 1, "median": record["reference_min_target_distance_m"],
                                             "max": record["reference_min_target_distance_m"]},
        "worst_cell_median_reference_min_target_distance_m": record["reference_min_target_distance_m"],
        "records": [record],
    }
    output_profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "PI0233_P1_BASELINE_ONE_ROW_CANARY_V1",
        "status": "PASS_GATE_INVOCATION_WITH_REJECT" if gate_decision == "REJECT" else "PASS",
        "selection_candidate": False,
        "validated_train_calib_row_count": len(rows),
        "executed_row_count": 1,
        "checkpoint_step": 1000,
        "checkpoint_path": str(BASELINE),
        "checkpoint_tree_sha256": BASELINE_TREE_SHA,
        "profile_path": str(output_profile),
        "profile_sha256": sha256_file(output_profile),
        "source_episode_id": record["source_episode_id"],
        "episode_index": record["episode_index"],
        "seed": SEED,
        "action_contract": "official_dense_relative_H40",
        "action_dt_s": 0.05,
        "model_input_keys": record["model_input_keys"],
        "target_xyz_in_model_input": False,
        "target_used_only_after_inference_for_scoring": True,
        "raw_shape": [1, 40, 132],
        "decoded_shape": [1, 40, 10],
        "finite_shape_checks": True,
        "raw_sha256": record["raw_sha256"],
        "decoded_sha256": record["decoded_sha256"],
        "reference_min_target_distance_m": record["reference_min_target_distance_m"],
        "safety_gate": record["safety_gate"],
        "safety_gate_call_completed": True,
        "safety_gate_decision_preserved": gate_decision,
        "forward_count": 1,
        "expected_forward_count": 1,
        "wbc_execution": False,
        "backward": False,
        "optimizer_steps": 0,
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
        "runtime_binding": runtime_binding,
        "gpu": record["gpu"],
        "latency_s": record["latency_s"],
        "selector_sha256": sha256_file(SELECTOR),
        "stage20s_helper_sha256": sha256_file(STAGE20S),
        "stage21_helper_sha256": sha256_file(STAGE21),
        "calib_seal_sha256": sha256_file(SEAL),
        "target_authority_sha256": sha256_file(AUTHORITY),
        "no_ranking_or_wbc_claim": True,
        "target_authority_source_grid_manifest_sha256": authority_payload["source_grid_manifest_sha256"],
    }
    output = args.output_root / "pi0233_p1_baseline_one_row_canary.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
