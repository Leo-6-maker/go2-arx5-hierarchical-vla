#!/usr/bin/env python3
"""Independent CPU replay of saved PI0247 L/U cells; never loads a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pi0247_d1_screen as d1
import pi0247_d2_full16_screen as d2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-card", type=Path, required=True)
    parser.add_argument("--d2-card-sha", type=Path, required=True)
    parser.add_argument("--u1000-report", type=Path, required=True)
    parser.add_argument("--u2000-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("L", "U"), default="U")
    args = parser.parse_args()

    card, card_sha = d2.read_card(args.d2_card, args.d2_card_sha)
    facts, context = d1.prepare(Path(card["paths"]["stage_root"]), Path(card["paths"]["code_root"]))
    context["facts"] = facts
    if d1.canonical_sha(facts) != card["static_preflight_canonical_sha256"]:
        raise RuntimeError("D2 static inputs differ from the sealed preflight")
    d2.prepare_inputs(card, context)

    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    source_ids = [row["source_episode_id"] for row in context["rows"]]
    expected_cells = {(source, noise) for source in source_ids for noise in ("B0", "B1")}
    reports = {}
    replay_rows = []
    final_rows = {"B0": [], "B1": []}
    gate = TrajectorySafetyGateV1()

    for step, report_path in ((1000, args.u1000_report), (2000, args.u2000_report)):
        report = read(report_path)
        if (report.get("schema") != "PI0247_LU_FINAL_SCREEN_REPORT_V1"
                or report.get("arm") != args.arm or report.get("checkpoint_step") != step
                or report.get("runtime_id") != "S4" or report.get("selected_checkpoint") != "NONE"
                or len(report.get("cells", [])) != 32):
            raise RuntimeError(f"U checkpoint-{step} report identity/count mismatch")
        score_path = report_path.parent / "score_card.json"
        score = read(score_path)
        evidence = report_path.parent.parent / f"{args.arm}_evidence"
        training_path = evidence / "training_manifest.json"
        training = read(training_path)
        card_path = evidence / ("L_run_card.json" if args.arm == "L" else "run_card.json")
        binding = training["checkpoints"][str(step)]
        if (report["checkpoint_binding"] != binding
                or report["training_manifest_sha256"] != sha(training_path)
                or report["run_card_sha256"] != sha(card_path)
                or report["hf_revision"] != binding["hf_manifest_commit_id"]
                or report["checkpoint_tree_sha256"] != binding["tree_sha256"]
                or report["family"] != training["family"]
                or report["score_card_sha256"] != sha(score_path)
                or score["d2_run_card_sha256"] != card_sha
                or set(report["source_episode_ids"]) != set(source_ids)):
            raise RuntimeError("screen model/source/card binding mismatch")
        reports[str(step)] = {"sha256": sha(report_path), "status": report["status"]}
        seen = set()

        for row in report["cells"]:
            source, noise = row["source_episode_id"], row["noise_name"]
            key = (source, noise)
            if key not in expected_cells or key in seen:
                raise RuntimeError(f"unexpected or duplicate U checkpoint-{step} cell: {key}")
            seen.add(key)
            cell = Path(row["source_cell"])
            record = d1.collect_record(cell / "record.json", context["p1"])
            if (record.get("status") != "VALID" or record.get("cell_key") != row.get("cell_key")
                    or record.get("tensor_sha256") != row.get("tensor_sha256")
                    or record.get("run_card_sha256") != sha(score_path)
                    or record.get("model_manifest_sha256") != score["inference_restore_sha256"]):
                raise RuntimeError(f"U checkpoint-{step} cell record/hash/status mismatch: {cell}")

            with np.load(cell / "tensors.npz", allow_pickle=False) as data:
                raw_scored = data["raw_scored"].copy()
                stored_decoded = data["decoded"].copy()
                expert = data["expert"].copy()
                state = data["state"].copy()
                noise_initial = data["noise_initial"].copy()
                trace_x = data["trace_x"].copy()
            arrays = (raw_scored, stored_decoded, expert, state, noise_initial, trace_x)
            if (raw_scored.shape != (1, 40, 132) or stored_decoded.shape != (1, 40, 10)
                    or expert.shape != (40, 10) or not all(np.isfinite(x).all() for x in arrays)):
                raise RuntimeError("saved full-H40 array shape/finite mismatch")

            fixed_noise = context["d2_noise"][source][noise]
            import torch
            expected_noise = torch.from_numpy(np.asarray(fixed_noise, dtype=np.float32)).to(
                dtype=torch.bfloat16
            ).to(dtype=torch.float32).numpy()
            if not np.array_equal(noise_initial, expected_noise) or trace_x.shape[0] != 4:
                raise RuntimeError(f"U checkpoint-{step} noise packet/S4 trace mismatch: {key}")

            decoded = bridge._decode_physical(
                context["processor"], context["embodiment"], raw_scored, state
            )
            max_abs = float(np.max(np.abs(decoded - stored_decoded)))
            if decoded.shape != (1, 40, 10) or max_abs > 1e-7:
                raise RuntimeError(f"U checkpoint-{step} CPU decode mismatch: {key}")

            metrics = stage43._screen_metrics(
                decoded[0].astype(np.float64), expert.astype(np.float64), None
            )
            residual = decoded[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
            squared = np.sum(residual * residual, axis=1)
            metrics["physical_residual_delta_max_mps"] = float(
                np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / 0.05
            )
            metrics["tail35_39_pooled_sse_fraction"] = float(
                squared[35:].sum() / squared.sum()
            ) if squared.sum() else 0.0
            saved_metrics = row.get("metrics") or {}
            if set(metrics) != set(saved_metrics) or any(
                not same_number(metrics[name], saved_metrics[name]) for name in metrics
            ):
                raise RuntimeError(f"U checkpoint-{step} CPU metric mismatch: {key}")

            state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
            gate_result = gate.evaluate(
                selector._build_chunk(
                    decoded,
                    chunk_id=f"pi0247-S4-{args.arm}{step}-{source}-{'B0' if noise == 'B0' else 'B1'}",
                ),
                now=0.0,
                achieved_position=state[:3],
                achieved_orientation=state_quat,
            )
            if gate_result.decision.value != row.get("gate_decision"):
                raise RuntimeError(f"U checkpoint-{step} CPU gate mismatch: {key}")
            saved_issues = record["gate"]["issues"]
            replayed_issues = gate_result.as_dict()["issues"]
            if len(saved_issues) != len(replayed_issues):
                raise RuntimeError("independent gate issue count mismatch")
            for saved, replayed in zip(saved_issues, replayed_issues):
                if set(saved) != set(replayed) or any(
                    not same_number(saved[k], replayed[k]) if isinstance(saved[k], (float, int))
                    else saved[k] != replayed[k] for k in saved
                ):
                    raise RuntimeError("independent gate issue content mismatch")

            if step == 2000:
                final_rows[noise].append({
                    "status": "VALID",
                    "metrics": metrics,
                    "gate_decision": gate_result.decision.value,
                })
            replay_rows.append({
                "checkpoint_step": step,
                "source_episode_id": source,
                "noise_name": noise,
                "tensor_sha256": record["tensor_sha256"],
                "cpu_decode_max_abs": max_abs,
                "noise_initial_matches_fixed_packet_bf16": True,
                "sampler_steps": int(trace_x.shape[0]),
                "cpu_metrics": metrics,
                "screen_gate_decision": row["gate_decision"],
                "cpu_gate_decision": gate_result.decision.value,
                "cpu_gate_issues": replayed_issues,
                "model_and_source_binding_verified": True,
            })

        if seen != expected_cells:
            raise RuntimeError(f"U checkpoint-{step} does not close 16 sources × B0/B1")

    recomputed_groups = {name: d2.evaluate_group(rows) for name, rows in final_rows.items()}
    report2000 = read(args.u2000_report)
    for name, values in recomputed_groups.items():
        saved = report2000["group_results"][name]
        if (values["quality_pass"] != saved["quality_pass"]
                or values["valid_count"] != saved["valid_count"]
                or values["invalid_count"] != saved["invalid_count"]
                or values["reject_count"] != saved["reject_count"]
                or not same_number(values["median_xyz_rmse_m"], saved["median_xyz_rmse_m"])
                or not same_number(values["p95_xyz_rmse_m"], saved["p95_xyz_rmse_m"])
                or not same_number(values["median_so3_rmse_rad"], saved["median_so3_rmse_rad"])):
            raise RuntimeError(f"U final2000 CPU group verdict mismatch: {name}")

    payload = {
        "schema": "PI0247_LU_SCREEN_INDEPENDENT_CPU_REPLAY_V2",
        "arm": args.arm,
        "status": "PASS_INDEPENDENT_CPU_REPLAY",
        "d2_run_card_sha256": card_sha,
        "reports": reports,
        "counts": {"cells_replayed": len(replay_rows), "checkpoint1000": 32, "final2000": 32,
                   "model_loads": 0, "model_forwards": 0},
        "final2000_recomputed_group_results": recomputed_groups,
        "cells": replay_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to replace conflicting CPU replay evidence: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cells": len(replay_rows),
                      "output": str(args.output), "sha256": sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
