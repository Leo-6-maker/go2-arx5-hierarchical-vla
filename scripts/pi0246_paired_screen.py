#!/usr/bin/env python3
"""PI0246 fixed16 paired stock screen; reuse sealed PI0245 sampling/gate helpers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import pi0245_p1_stock as p1


COHORT_SHA = "3d679d9cf5814d6aca63ca52e5953e1a50fb6097e3d8d67a7b875ab1178e36cd"
RECORD_SHA = p1.RECORD_SHA
P1_CLOSURE_SHA = "77712a2598ae7f8d8a076139c4160ec480c4d9ff624b2e45be9a0a51a55cc70b"
C_CARD_SHA = "f3a9fdffffce91d97891dc0b919c5b64cb12a8819292eb2ad78e34c7c3188601"
T_CARD_SHA = "469c0a29571e05c30d50cf63ab0a13d8e3495a91469044bcf14feb840600e938"
P_CARD_SHA = "7aa496963ee99ed241bad1506779a63e1f0383617a008f4ac09cfefde04041ce"
PROFILES = {"S0": (None, 0), "C1000": ("C", 1000), "C2000": ("C", 2000),
            "T1000": ("T", 1000), "T2000": ("T", 2000),
            "P1000": ("P", 1000), "P2000": ("P", 2000)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bind_offline_runtime(paths: dict[str, str]) -> None:
    sys.path[:0] = [str(Path(paths["code_root"]) / "scripts"), str(Path(paths["code_root"]) / "src"), paths["groot_source"]]
    cache = Path(paths["stage10_root"]) / "cache/huggingface"
    assert (cache / "hub").is_dir(), f"sealed Hugging Face cache absent: {cache}"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def inputs(root: Path) -> tuple[list[dict], dict[str, str], dict]:
    cohort = root / "screen_inputs/fixed16_cohort.json"
    records = root / "r0/screen/step-10000/records.jsonl"
    c_card = root / "screen_inputs/C_run_card.json"
    t_card = root / "screen_inputs/T_run_card.json"
    p_card = root / "screen_inputs/P_run_card.json"
    assert p1.sha(cohort) == COHORT_SHA and p1.sha(records) == RECORD_SHA
    assert p1.sha(c_card) == C_CARD_SHA and p1.sha(t_card) == T_CARD_SHA
    assert p1.sha(p_card) == P_CARD_SHA
    assert p1.sha(root / "p1_corelab/p1_closure.json") == P1_CLOSURE_SHA
    cards = {"C": read_json(c_card), "T": read_json(t_card), "P": read_json(p_card)}
    for arm, card in cards.items():
        assert card["arm"] == arm and card["cohort"]["sha256"] == COHORT_SHA
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["source_episode_id"] for row in read_json(cohort)["rows"]]
    assert len(rows) == len(ids) == len(set(ids)) == 16
    assert [row["source_episode_id"] for row in rows] == ids
    for row in rows:
        path = p1.resolve_tensor_path(root / "r0", row["tensor_path"])
        assert p1.sha(path) == row["tensor_sha256"]
        with np.load(path, allow_pickle=False) as payload:
            assert payload["raw"].shape == (1, 40, 132)
            assert payload["expert"].shape == (40, 10)
            assert payload["state"].shape == (10,)
            assert payload["action_mask"].shape == (40, 132)
            assert payload["body_rgb"].shape == payload["wrist_rgb"].shape == (96, 96, 3)
            assert all(np.isfinite(payload[key]).all() for key in payload.files)
        row["resolved_tensor_path"] = str(path)
    old_card = read_json(root / "r0/pi0243_run_card.json")
    old_card["_sha256"] = p1.sha(root / "r0/pi0243_run_card.json")
    paths = p1.resolve_path_overlay(old_card, root / "pi0246_p1_path_overlay_corelab_20260924.json")
    return rows, paths, cards


def policy_for(row: dict, processor, embodiment, bridge, stage43):
    from gr00t.data.types import MessageType, VLAStepData

    with np.load(row["resolved_tensor_path"], allow_pickle=False) as payload:
        body, wrist = payload["body_rgb"].copy(), payload["wrist_rgb"].copy()
        state, expert, mask = payload["state"].copy(), payload["expert"].copy(), payload["action_mask"].copy()
    _, error = p1.encode_expert(processor, embodiment, expert, state)
    assert error <= 1e-5
    content = VLAStepData(
        images={"external_rgb": body[None], "wrist_rgb": wrist[None]},
        states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:10]},
        actions={}, text=stage43.LANGUAGE, embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    policy = bridge._policy_inputs(processor.collator([processed]))
    inventory = bridge._leaf_inventory(policy)
    assert not [key for key in inventory if any(token in key.lower() for token in bridge.FORBIDDEN_INPUT_TOKENS)]
    return policy, bridge._canonical_hash(inventory), state, expert, mask, body, wrist


def validate_existing(cell: Path, source: str, profile: str) -> dict:
    record = read_json(cell / "record.json")
    assert record["schema"] == "PI0246_PAIRED_SCREEN_CELL_V1"
    assert record["source_episode_id"] == source and record["profile"] == profile
    assert p1.sha(cell / "tensors.npz") == record["tensor_sha256"]
    assert p1.sha(cell / "attempt.json") == record["attempt_sha256"]
    if profile == "S0":
        assert p1.sha(cell / "condition.pt") == record["condition_sha256"]
    return record


def run(args: argparse.Namespace) -> None:
    rows, paths, cards = inputs(args.stage_root)
    arm, step = PROFILES[args.profile]
    restore = read_json(args.model_root / "pi0245_inference_restore.json")
    model_files = p1.verify_model_files(args.model_root)
    if arm is None:
        assert model_files["manifest_sha256"] == cards["C"]["parent"]["restore_manifest_sha256"]
    else:
        assert restore["checkpoint_step"] == step and restore["optimizer_restored"] is False
        assert cards[arm]["family"] in restore["prefix"]
    bind_offline_runtime(paths)
    import torch
    from transformers import AutoModel
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import stage21_same_scene_groot_inference as stage21
    import stage40_b0_model_control as stage40
    stage40._bind_local_cosmos_processor(Path(paths["stage10_root"]), Path(paths["cosmos_snapshot"]))
    processor, embodiment = stage21._build_processor(
        Path(paths["stage10_root"]), Path(paths["processor_base_model"]),
        Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"]),
    )
    gate = TrajectorySafetyGateV1()
    profile_root = args.output_root / args.profile
    profile_root.mkdir(parents=True, exist_ok=True)
    ledger = profile_root / "queries.jsonl"
    records = profile_root / "records.jsonl"
    attempted = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    completed = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    ids = [row["source_episode_id"] for row in rows]
    assert [row["source_episode_id"] for row in attempted] == ids[:len(attempted)]
    assert [row["source_episode_id"] for row in completed] == ids[:len(completed)]
    assert len(attempted) <= 16 and len(completed) <= len(attempted)
    assert all(a["source_episode_id"] == c["source_episode_id"] and a["model_manifest_sha256"] == c["model_manifest_sha256"]
               for a, c in zip(attempted, completed))
    model = AutoModel.from_pretrained(args.model_root, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    for row in rows:
        source = row["source_episode_id"]
        cell = profile_root / "cells" / source.split(":", 1)[0]
        if (cell / "record.json").exists():
            record = validate_existing(cell, source, args.profile)
            assert source in {value["source_episode_id"] for value in attempted}
            assert record["model_manifest_sha256"] == model_files["manifest_sha256"]
            if source not in {value["source_episode_id"] for value in completed}:
                with records.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
                completed.append(record)
            continue
        assert not cell.exists() and source not in {value["source_episode_id"] for value in attempted}
        cell.mkdir(parents=True)
        if arm is None:
            policy, identity, state, expert, mask, body, wrist = policy_for(row, processor, embodiment, bridge, stage43)
            p1.durable_torch(cell / "condition.pt", policy)
            fixed_noise = None
            condition_sha = p1.sha(cell / "condition.pt")
        else:
            parent = args.output_root / "S0/cells" / source.split(":", 1)[0]
            s0 = validate_existing(parent, source, "S0")
            condition_sha = p1.sha(parent / "condition.pt")
            assert condition_sha == s0["condition_sha256"]
            policy = torch.load(parent / "condition.pt", map_location="cpu", weights_only=False)
            with np.load(parent / "noise_packet.npz", allow_pickle=False) as packet:
                fixed_noise = packet["source_float32"].copy()
                initial_noise = packet["initial_float32"].copy()
            assert p1.sha(parent / "noise_packet.npz") == s0["noise_packet_sha256"]
            identity = s0["condition_identity"]
            with np.load(row["resolved_tensor_path"], allow_pickle=False) as payload:
                state, expert, mask = payload["state"].copy(), payload["expert"].copy(), payload["action_mask"].copy()
                body, wrist = payload["body_rgb"].copy(), payload["wrist_rgb"].copy()
        attempt = {"schema": "PI0246_PAIRED_SCREEN_ATTEMPT_V1", "profile": args.profile,
                   "source_episode_id": source, "source_tensor_sha256": row["tensor_sha256"],
                   "condition_sha256": condition_sha, "condition_identity": identity,
                   "model_manifest_sha256": model_files["manifest_sha256"], "model_revision": model_files["revision"],
                   "runner_sha256": p1.sha(Path(__file__).resolve()), "target_xyz_in_model_input": False}
        p1.durable_json(cell / "attempt.json", attempt)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
        attempted.append(attempt)
        stock = bridge._stock_query(model, policy, stage21, fixed_noise)
        raw = np.asarray(stock["raw"], dtype=np.float32)
        noise = np.asarray(stock["noise"], dtype=np.float32)
        trace_x = np.stack([item["x"] for item in stock["trace"]])
        trace_v = np.stack([item["v"] for item in stock["trace"]])
        trace_t = np.stack([item["t"] for item in stock["trace"]])
        if arm is None:
            p1.durable_npz(cell / "noise_packet.npz", source_float32=noise, initial_float32=trace_x[0])
            packet_sha = p1.sha(cell / "noise_packet.npz")
        else:
            assert np.array_equal(noise, fixed_noise) and np.array_equal(trace_x[0], initial_noise)
            packet_sha = s0["noise_packet_sha256"]
        if raw.shape != (1, 40, 132) or not np.isfinite(raw).all():
            p1.durable_npz(cell / "invalid_output.npz", raw=raw, expert=expert, state=state,
                           action_mask=mask, noise_source_float32=noise, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
            p1.durable_json(cell / "record.json", {**attempt, "schema": "PI0246_INVALID_STOCK_CELL_V1",
                                                "status": "INVALID_RAW_SHAPE_OR_NONFINITE",
                                                "tensor_sha256": p1.sha(cell / "invalid_output.npz")})
            raise RuntimeError("invalid native stock output; preserved first attempt")
        try:
            decoded = bridge._decode_physical(processor, embodiment, raw, state)
        except Exception as exc:
            p1.durable_npz(cell / "invalid_output.npz", raw=raw, expert=expert, state=state,
                           action_mask=mask, noise_source_float32=noise, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
            p1.durable_json(cell / "record.json", {**attempt, "schema": "PI0246_INVALID_DECODE_CELL_V1",
                                                "status": "DECODE_FAILED", "error": repr(exc),
                                                "tensor_sha256": p1.sha(cell / "invalid_output.npz")})
            raise
        if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
            p1.durable_npz(cell / "invalid_output.npz", raw=raw, decoded=decoded, expert=expert, state=state,
                           action_mask=mask, noise_source_float32=noise, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
            p1.durable_json(cell / "record.json", {**attempt, "schema": "PI0246_INVALID_STOCK_CELL_V1",
                                                "status": "INVALID_DECODED_SHAPE_OR_NONFINITE",
                                                "tensor_sha256": p1.sha(cell / "invalid_output.npz")})
            raise RuntimeError("invalid native decode; preserved first attempt")
        quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
        gate_result = gate.evaluate(selector._build_chunk(decoded, chunk_id=f"pi0246-{args.profile}-{source}"),
                                    now=0.0, achieved_position=state[:3], achieved_orientation=quat)
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
        residual = decoded[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
        squared = np.sum(residual * residual, axis=1)
        metrics["physical_residual_delta_max_mps"] = float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05)
        metrics["tail35_39_pooled_sse_fraction"] = float(squared[35:].sum() / squared.sum()) if squared.sum() else 0.0
        p1.durable_npz(cell / "tensors.npz", raw=raw, decoded=decoded, expert=expert, state=state,
                       action_mask=mask, body_rgb=body, wrist_rgb=wrist, noise_source_float32=noise,
                       noise_initial=trace_x[0], trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
        record = {**attempt, "schema": "PI0246_PAIRED_SCREEN_CELL_V1", "attempt_sha256": p1.sha(cell / "attempt.json"),
                  "tensor_sha256": p1.sha(cell / "tensors.npz"), "condition_sha256": condition_sha,
                  "noise_packet_sha256": packet_sha, "metrics": metrics, "gate_decision": gate_result.decision.value,
                  "gate": gate_result.as_dict(), "get_action_calls": 1, "backbone_forwards": 1,
                  "action_head_denoising_forwards": len(stock["trace"]), "target_xyz_in_model_input": False}
        p1.durable_json(cell / "record.json", record)
        with records.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
        completed.append(record)
        print(json.dumps({"profile": args.profile, "completed": len(completed), "gate": record["gate_decision"],
                          "xyz_rmse_m": metrics["xyz_rmse_m"]}), flush=True)
    assert len(completed) == 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    rows, paths, cards = inputs(args.stage_root)
    if args.preflight:
        bind_offline_runtime(paths)
        import pi0242_train_infer_bridge as bridge
        import pi0243_staged_learning_to_sim as stage43
        import stage21_same_scene_groot_inference as stage21
        import stage40_b0_model_control as stage40
        stage40._bind_local_cosmos_processor(Path(paths["stage10_root"]), Path(paths["cosmos_snapshot"]))
        processor, embodiment = stage21._build_processor(
            Path(paths["stage10_root"]), Path(paths["processor_base_model"]),
            Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"]),
        )
        _, identity, *_ = policy_for(rows[0], processor, embodiment, bridge, stage43)
        print(json.dumps({"status": "PASS_FIXED16_SCREEN_INPUTS", "rows": len(rows),
                          "cohort_sha256": COHORT_SHA, "conditioning_identity_row0": identity,
                          "cards": {arm: p1.sha(args.stage_root / f"screen_inputs/{arm}_run_card.json") for arm in cards}}))
        return
    assert args.model_root is not None
    run(args)


if __name__ == "__main__":
    main()
