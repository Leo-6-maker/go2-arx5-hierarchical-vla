#!/usr/bin/env python3
"""CPU-only integrity, gate, and metric verification for a sealed PI0246 profile."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

import pi0246_paired_screen as screen
import pi0245_p1_stock as p1

PROFILE_CHOICES = tuple(dict.fromkeys((*screen.PROFILES, "P1000", "P2000")))


def screen_module_name(profile: str) -> str:
    return "pi0246_paired_screen_P_extension" if profile in {"P1000", "P2000"} else "pi0246_paired_screen"


def verify(stage_root: Path, output_root: Path, model_root: Path, profile: str, report_path: Path) -> dict:
    paired_screen = importlib.import_module(screen_module_name(profile))
    rows, paths, cards = paired_screen.inputs(stage_root)
    ids = [row["source_episode_id"] for row in rows]
    profile_root = output_root / profile
    ledger = [json.loads(line) for line in (profile_root / "queries.jsonl").read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in (profile_root / "records.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("source_episode_id") for row in ledger] != ids or [row.get("source_episode_id") for row in records] != ids:
        raise RuntimeError("profile does not contain the exact sealed 16-source order")
    arm, step = paired_screen.PROFILES[profile]
    restore = paired_screen.read_json(model_root / "pi0245_inference_restore.json")
    model = p1.verify_model_files(model_root)
    if arm is None:
        if model["manifest_sha256"] != cards["C"]["parent"]["restore_manifest_sha256"]:
            raise RuntimeError("S0 model identity differs from the sealed R0 initializer")
    elif int(restore.get("checkpoint_step", -1)) != step or restore.get("optimizer_restored") is not False:
        raise RuntimeError("profile inference view is not the requested weights-only checkpoint")
    if arm is not None:
        family = cards[arm]["family"]
        if (restore.get("family") != family
                or restore.get("prefix") != f"{family}/checkpoint-{step}"
                or restore.get("hf_archive_path") != f"{restore.get('repo_id')}/arms/{arm}/checkpoint-{step}"
                or restore.get("revision") != model["revision"]):
            raise RuntimeError("profile inference view does not bind the card family, step, HF path, and pinned model revision")
    for query, record in zip(ledger, records):
        if query.get("source_episode_id") != record.get("source_episode_id") or query.get("model_manifest_sha256") != model["manifest_sha256"]:
            raise RuntimeError("query ledger and profile record provenance differ")
        if record.get("schema") != "PI0246_PAIRED_SCREEN_CELL_V1" or record.get("profile") != profile:
            raise RuntimeError("profile contains a non-valid or wrong-profile cell")
        if record.get("model_manifest_sha256") != model["manifest_sha256"] or record.get("model_revision") != model["revision"]:
            raise RuntimeError("cell model identity differs from the verified inference view")
        if record.get("source_tensor_sha256") != rows[ids.index(record["source_episode_id"])]["tensor_sha256"]:
            raise RuntimeError("cell source tensor is not the fixed TRAIN source row")
        if record.get("target_xyz_in_model_input") is not False:
            raise RuntimeError("target XYZ was present in the model input")
        if (int(record.get("get_action_calls", 0)), int(record.get("backbone_forwards", 0))) != (1, 1):
            raise RuntimeError("valid cell does not have exactly one policy/backbone query")
        cell = profile_root / "cells" / record["source_episode_id"].split(":", 1)[0]
        paired_screen.validate_existing(cell, record["source_episode_id"], profile)
        payload = np.load(cell / "tensors.npz", allow_pickle=False)
        with payload:
            required = {"raw", "decoded", "expert", "action_mask", "state", "body_rgb", "wrist_rgb",
                        "noise_source_float32", "noise_initial", "trace_x", "trace_v", "trace_t"}
            if set(payload.files) != required:
                raise RuntimeError(f"tensor fields differ from the sealed profile contract: {sorted(payload.files)}")
            if any(not np.isfinite(payload[name]).all() for name in payload.files):
                raise RuntimeError("profile tensor archive contains nonfinite values")
            raw, decoded = payload["raw"], payload["decoded"]
            expert, mask, state = payload["expert"], payload["action_mask"], payload["state"]
            if raw.shape != (1, 40, 132) or decoded.shape != (1, 40, 10) or expert.shape != (40, 10):
                raise RuntimeError("profile raw/decoded/expert tensor shape mismatch")
            if mask.shape != (40, 132) or state.shape != (10,) or payload["body_rgb"].shape != (96, 96, 3) or payload["wrist_rgb"].shape != (96, 96, 3):
                raise RuntimeError("profile input tensor shape mismatch")
            source_path = Path(rows[ids.index(record["source_episode_id"])]["resolved_tensor_path"])
            with np.load(source_path, allow_pickle=False) as source:
                for key in ("expert", "action_mask", "state", "body_rgb", "wrist_rgb"):
                    if not np.array_equal(payload[key], source[key]):
                        raise RuntimeError(f"profile {key} differs from its sealed source tensor")
            if len(payload["trace_t"]) != int(record["action_head_denoising_forwards"]) or len(payload["trace_x"]) != len(payload["trace_v"]) or len(payload["trace_x"]) != len(payload["trace_t"]):
                raise RuntimeError("stock sampler trace cardinality mismatch")
            if arm is None:
                packet_path = cell / "noise_packet.npz"
                if p1.sha(packet_path) != record["noise_packet_sha256"]:
                    raise RuntimeError("S0 condition noise packet hash mismatch")
                with np.load(packet_path, allow_pickle=False) as packet:
                    source_noise, initial_noise = packet["source_float32"], packet["initial_float32"]
            else:
                s0_cell = output_root / "S0/cells" / record["source_episode_id"].split(":", 1)[0]
                s0 = screen.validate_existing(s0_cell, record["source_episode_id"], "S0")
                if record["condition_sha256"] != s0["condition_sha256"] or record["noise_packet_sha256"] != s0["noise_packet_sha256"]:
                    raise RuntimeError("paired profile condition/noise identity differs from S0")
                with np.load(s0_cell / "noise_packet.npz", allow_pickle=False) as packet:
                    source_noise, initial_noise = packet["source_float32"], packet["initial_float32"]
            if not np.array_equal(payload["noise_source_float32"], source_noise) or not np.array_equal(payload["noise_initial"], initial_noise):
                raise RuntimeError("stored query noise differs from the fixed source packet")
    paired_screen.bind_offline_runtime(paths)
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
    comparisons = []
    for row, record in zip(rows, records):
        cell = profile_root / "cells" / record["source_episode_id"].split(":", 1)[0]
        with np.load(cell / "tensors.npz", allow_pickle=False) as payload:
            raw, saved = payload["raw"], payload["decoded"]
            state, expert = payload["state"], payload["expert"]
            recomputed = bridge._decode_physical(processor, embodiment, raw, state)
            if recomputed.shape != saved.shape or not np.allclose(recomputed, saved, rtol=0, atol=1e-6):
                raise RuntimeError("saved physical decode differs from the frozen dense-relative decoder")
            metrics = stage43._screen_metrics(saved[0].astype(np.float64), expert.astype(np.float64), None)
            residual = saved[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
            squared = np.sum(residual * residual, axis=1)
            metrics["physical_residual_delta_max_mps"] = float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05)
            metrics["tail35_39_pooled_sse_fraction"] = float(squared[35:].sum() / squared.sum()) if squared.sum() else 0.0
            for key, value in metrics.items():
                if not math.isclose(float(record["metrics"][key]), float(value), rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError(f"metric replay mismatch for {record['source_episode_id']}:{key}")
            quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
            expected_gate = gate.evaluate(
                selector._build_chunk(saved, chunk_id=f"pi0246-{profile}-{record['source_episode_id']}"),
                now=0.0, achieved_position=state[:3], achieved_orientation=quat,
            ).as_dict()
            if record["gate_decision"] not in {"ACCEPT", "WARN", "REJECT"} or record["gate"] != expected_gate or record["gate_decision"] != expected_gate["decision"]:
                raise RuntimeError("SafetyGate enum/issues/metrics replay mismatch")
            comparisons.append({"source_episode_id": record["source_episode_id"], "gate": record["gate_decision"],
                                "xyz_rmse_m": metrics["xyz_rmse_m"], "so3_rmse_rad": metrics["so3_rmse_rad"],
                                "max_linear_speed_mps": metrics["max_linear_speed_mps"],
                                "max_angular_speed_radps": metrics["max_angular_speed_radps"]})
    report = {
        "schema": "PI0246_PAIRED_PROFILE_CPU_VERIFICATION_V1", "status": "PASS_READONLY_PROFILE_REPLAY",
        "profile": profile, "records": len(records), "model_queries": 0, "model_weights_loaded": 0,
        "verifier_sha256": p1.sha(Path(__file__).resolve()),
        "runner_sha256": p1.sha(Path(paired_screen.__file__).resolve()),
        "records_sha256": p1.sha(profile_root / "records.jsonl"),
        "queries_sha256": p1.sha(profile_root / "queries.jsonl"),
        "model_manifest_sha256": model["manifest_sha256"], "model_revision": model["revision"],
        "gate_counts": dict(Counter(row["gate"] for row in comparisons)),
        "comparisons": comparisons,
    }
    if report_path.exists():
        if screen.read_json(report_path) != report:
            raise RuntimeError("first-valid CPU verification report already exists with different contents")
    else:
        p1.durable_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(verify(args.stage_root, args.output_root, args.model_root, args.profile, args.report),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
