#!/usr/bin/env python3
"""PI0246 fixed16 paired stock screen; reuse sealed PI0245 sampling/gate helpers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pi0245_p1_stock as p1
except ModuleNotFoundError:
    from scripts import pi0245_p1_stock as p1


COHORT_SHA = "3d679d9cf5814d6aca63ca52e5953e1a50fb6097e3d8d67a7b875ab1178e36cd"
RECORD_SHA = p1.RECORD_SHA
P1_CLOSURE_SHA = "77712a2598ae7f8d8a076139c4160ec480c4d9ff624b2e45be9a0a51a55cc70b"
C_CARD_SHA = "f3a9fdffffce91d97891dc0b919c5b64cb12a8819292eb2ad78e34c7c3188601"
T_CARD_SHA = "469c0a29571e05c30d50cf63ab0a13d8e3495a91469044bcf14feb840600e938"
PT_FAMILY = "PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1"
PT_REPO = "leoyuer/pi0246-gate-pass-campaign"
PT_CARD_SHA = "9d8d0c34a98b077be22f696b4e45420c1719d8c733796ec9e7e2ea1473e5ff00"
PT_SEAL_SHA = "78b6ea51ca47d29606168aabaa3e2274c4985f437b9e1742f40ac430fb9ed5b3"
PT_TRAINING_MANIFEST_SHA = "e43d2506737fd93211ca75eaca00f74cee9cee104b7593a9f1d10139b2ccb090"
PT_RECONCILIATION_SHA = "c7bacdfea43589051bd275b917792906375ee5aeb797b4ac98d1d011ec9bfc7d"
PT_UPDATE_AUDIT_SHA = "bc8dfdcc2ccd06bfb30b7b86eea6369b9378a32fc9bdf4320f2d259257ee8961"
PT_RECEIPT_SHA = {
    1000: "fee95d5bf753304fbc5988739547f286072c67cec1dda698836ff569b1369957",
    2000: "75d6605987f417bb40704d0346aa5c4082358cc5d08755a20513d4da8ce2e95a",
}
PT_READBACK_SHA = {
    1000: "f3f1bc8056639aa101ea6d9adb07f016b6b7a4e26728c752156ef547a4e139fe",
    2000: "17efbdff3a98c2016f3240cb978e3e82c411b0aec1e5ff8802e98a8363cf27c2",
}
PT_REVISION = {1000: "a1d699834abf184502756e74fff21c7eee6e15c6",
               2000: "e4da899edb981cf2a0e9e8269249da4c08f45891"}
PT_TREE = {1000: "821c5aa6117231d3edb9db6ffb8d6d6c4e90471979575d22f8d29dd12ef7560a",
           2000: "b3b87b8d01609e18483ff84bb0e4edebf16e9b2b4a1f2b7b708abbc9adb0ef51"}
PROFILES = {"PT1000": ("PT", 1000), "PT2000": ("PT", 2000)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wanted(name: str) -> bool:
    return name in {"config.json", "embodiment_id.json", "processor_config.json", "statistics.json",
                    "model.safetensors.index.json"} or name.endswith(".safetensors") or name.startswith("experiment_cfg/")


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
    pt_card = root / "screen_inputs/PT_run_card.json"
    assert p1.sha(cohort) == COHORT_SHA and p1.sha(records) == RECORD_SHA
    assert p1.sha(c_card) == C_CARD_SHA and p1.sha(t_card) == T_CARD_SHA
    assert p1.sha(pt_card) == PT_CARD_SHA
    assert p1.sha(root / "screen_inputs/PT_run_card.seal.json") == PT_SEAL_SHA
    assert read_json(root / "screen_inputs/PT_run_card.seal.json") == {"sha256": PT_CARD_SHA}
    assert p1.sha(root / "p1_corelab/p1_closure.json") == P1_CLOSURE_SHA
    cards = {"C": read_json(c_card), "T": read_json(t_card), "PT": read_json(pt_card)}
    for arm, card in cards.items():
        assert card["arm"] == arm and card["cohort"]["sha256"] == COHORT_SHA
    assert cards["PT"]["family"] == PT_FAMILY and cards["PT"]["checkpoint_steps"] == [1000, 2000]
    manifest_path = root / "screen_inputs/PT_training_manifest.json"
    reconciliation_path = root / "screen_inputs/PT_training_reconciliation.json"
    audit_path = root / "screen_inputs/PT_update_audit.json"
    assert p1.sha(manifest_path) == PT_TRAINING_MANIFEST_SHA
    assert p1.sha(reconciliation_path) == PT_RECONCILIATION_SHA
    assert p1.sha(audit_path) == PT_UPDATE_AUDIT_SHA
    manifest = read_json(manifest_path)
    reconciliation = read_json(reconciliation_path)
    audit = read_json(audit_path)
    assert manifest["family"] == PT_FAMILY and manifest["run_card_sha256"] == PT_CARD_SHA
    assert manifest["global_step"] == manifest["sample_consumption"] == 2000
    assert audit["global_step"] == 2000 and audit["expected_end_step"] == 2000
    assert reconciliation["status"] == "TRAINING_INTEGRITY_RECONCILED_PT_FULL_PROJECTOR_TENSORS"
    integrity = reconciliation["effective_update_integrity_after_readonly_tensor_comparison"]
    assert all(integrity[key] is True for key in (
        "diffusion_updated", "frozen_backbone_unchanged", "frozen_vlln_unchanged",
        "optimizer_membership_valid", "projector_updated", "same_forward_audit_complete"))
    for step in (1000, 2000):
        validate_training_archive(root, step, manifest)
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


def validate_training_archive(root: Path, step: int, manifest: dict | None = None) -> tuple[dict, dict]:
    receipt_path = root / f"screen_inputs/PT_checkpoint-{step}.durable_receipt.json"
    readback_path = root / f"screen_inputs/PT_checkpoint-{step}.hf_readback.json"
    assert p1.sha(receipt_path) == PT_RECEIPT_SHA[step]
    assert p1.sha(readback_path) == PT_READBACK_SHA[step]
    receipt, readback = read_json(receipt_path), read_json(readback_path)
    tree = PT_TREE[step]
    revision = PT_REVISION[step]
    assert receipt["status"] == "DURABLE_VERIFIED"
    assert receipt["family"] == PT_FAMILY and receipt["checkpoint_step"] == step
    assert receipt["durable_path"] == f"{PT_REPO}/arms/PT/checkpoint-{step}"
    assert receipt["hf_manifest_commit_id"] == revision
    assert receipt["source_record"]["inventory"]["tree_sha256"] == tree
    assert receipt["source_record"]["inventory"]["file_count"] == 9
    assert receipt["source_record"]["global_step"] == step
    assert readback["status"] == "PASS_PT_FULL_HF_READBACK"
    assert readback["family"] == PT_FAMILY and readback["checkpoint_step"] == step
    assert readback["repo_id"] == PT_REPO and readback["path_prefix"] == "arms/PT"
    assert readback["revision"] == revision and readback["source_tree_sha256"] == tree
    assert readback["file_count"] == 9
    if manifest is not None:
        row = manifest["checkpoints"][str(step)]
        assert row["durable_receipt_sha256"] == PT_RECEIPT_SHA[step]
        assert row["hf_readback_sha256"] == PT_READBACK_SHA[step]
        assert row["hf_manifest_commit_id"] == revision and row["inventory"]["tree_sha256"] == tree
    return receipt, readback


def validate_inference_view(root: Path, model_root: Path, step: int) -> dict:
    receipt, _ = validate_training_archive(root, step)
    restore = read_json(model_root / "pi0245_inference_restore.json")
    view = read_json(model_root / "INFERENCE_VIEW_VERIFIED.json")
    model_files = p1.verify_model_files(model_root)
    from pi0245_restore_inference_model import validate_model_index

    validate_model_index(model_root, restore["files"])
    expected_path = f"{PT_REPO}/arms/PT/checkpoint-{step}"
    inventory = receipt["source_record"]["inventory"]["entries"]
    expected_files = sorted(
        ({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in inventory if wanted(row["path"])),
        key=lambda row: row["path"],
    )
    actual_files = sorted(restore["files"], key=lambda row: row["path"])
    assert actual_files == expected_files
    assert restore["family"] == PT_FAMILY and restore["checkpoint_step"] == step
    assert restore["prefix"] == f"{PT_FAMILY}/checkpoint-{step}"
    assert restore["hf_archive_path"] == expected_path
    assert restore["revision"] == PT_REVISION[step] and model_files["revision"] == PT_REVISION[step]
    assert restore["receipt_sha256"] == PT_RECEIPT_SHA[step]
    assert view["status"] == "INFERENCE_VIEW_VERIFIED"
    assert view["family"] == PT_FAMILY and view["checkpoint_step"] == step
    assert view["repo_id"] == PT_REPO and view["hf_archive_path"] == expected_path
    assert view["prefix"] == restore["prefix"] and view["revision"] == PT_REVISION[step]
    assert view["full_checkpoint_receipt_sha256"] == PT_RECEIPT_SHA[step]
    assert view["source_tree_sha256"] == PT_TREE[step]
    assert view["restore_manifest_sha256"] == p1.sha(model_root / "pi0245_inference_restore.json")
    assert view["optimizer_restored"] is False and view["file_count"] == len(expected_files)
    assert sorted(view["files"], key=lambda row: row["path"]) == expected_files
    return {"restore": restore, "view": view, "model_files": model_files,
            "receipt_sha256": PT_RECEIPT_SHA[step], "source_tree_sha256": PT_TREE[step]}


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


def validate_s0_packets(output_root: Path, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        source = row["source_episode_id"]
        cell = output_root / "S0/cells" / source.split(":", 1)[0]
        record = validate_existing(cell, source, "S0")
        assert p1.sha(cell / "condition.pt") == record["condition_sha256"]
        packet_path = cell / "noise_packet.npz"
        assert p1.sha(packet_path) == record["noise_packet_sha256"]
        with np.load(packet_path, allow_pickle=False) as packet:
            assert packet["source_float32"].shape == packet["initial_float32"].shape == (1, 40, 132)
            assert np.isfinite(packet["source_float32"]).all() and np.isfinite(packet["initial_float32"]).all()
        result.append({"source_episode_id": source, "condition_sha256": record["condition_sha256"],
                       "noise_packet_sha256": record["noise_packet_sha256"]})
    return result


def run(args: argparse.Namespace) -> None:
    rows, paths, cards = inputs(args.stage_root)
    arm, step = PROFILES[args.profile]
    identity = validate_inference_view(args.stage_root, args.model_root, step)
    restore, model_files = identity["restore"], identity["model_files"]
    s0_packets = {row["source_episode_id"]: row for row in validate_s0_packets(args.output_root, rows)}
    inference_view_sha = p1.sha(args.model_root / "INFERENCE_VIEW_VERIFIED.json")
    runner_sha = p1.sha(Path(__file__).resolve())
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
            attempt_record = read_json(cell / "attempt.json")
            ledger_record = next(value for value in attempted if value["source_episode_id"] == source)
            s0 = s0_packets[source]
            assert all(record.get(key) == value for key, value in attempt_record.items() if key != "schema")
            assert attempt_record == ledger_record
            assert record["model_manifest_sha256"] == model_files["manifest_sha256"]
            assert record["model_revision"] == model_files["revision"]
            assert record["family"] == PT_FAMILY and record["checkpoint_step"] == step
            assert record["run_card_sha256"] == PT_CARD_SHA
            assert record["full_checkpoint_receipt_sha256"] == PT_RECEIPT_SHA[step]
            assert record["inference_view_sha256"] == inference_view_sha
            assert record["condition_sha256"] == s0["condition_sha256"]
            assert record["noise_packet_sha256"] == s0["noise_packet_sha256"]
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
                   "runner_sha256": runner_sha, "target_xyz_in_model_input": False,
                   "family": PT_FAMILY, "checkpoint_step": step,
                   "run_card_sha256": PT_CARD_SHA, "full_checkpoint_receipt_sha256": PT_RECEIPT_SHA[step],
                   "inference_view_sha256": inference_view_sha}
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
        arm, step = PROFILES[args.profile]
        view_identity = validate_inference_view(args.stage_root, args.model_root, step)
        s0_packets = validate_s0_packets(args.output_root, rows)
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
        print(json.dumps({"status": "PASS_PT_SCREEN_ADMISSION_NO_MODEL_QUERY", "rows": len(rows),
                          "profile": args.profile, "step": step, "cohort_sha256": COHORT_SHA,
                          "conditioning_identity_row0": identity, "cards": {key: p1.sha(args.stage_root / f"screen_inputs/{key}_run_card.json") for key in ("C", "T")},
                          "pt_card_sha256": PT_CARD_SHA,
                          "runner_sha256": p1.sha(Path(__file__).resolve()),
                          "s0_packets": s0_packets,
                          "receipt_sha256": view_identity["receipt_sha256"],
                          "inference_view_sha256": p1.sha(args.model_root / "INFERENCE_VIEW_VERIFIED.json"),
                          "model_loads": 0, "queries": 0, "optimizer_steps": 0}))
        return
    assert args.model_root is not None
    run(args)


if __name__ == "__main__":
    main()
