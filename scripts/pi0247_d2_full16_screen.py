#!/usr/bin/env python3
"""PI0247 D2: original 16 windows, per-source S0 B0 and shared P1 N1 B1."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import pi0247_d1_screen as d1


INVALID = {"INVALID_RAW_NONFINITE", "INVALID_DECODE"}
S4 = {"id": "S4", "steps": 4, "inactive": "stock"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_group(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("metrics") and row.get("status") not in INVALID]
    values = [row["metrics"] for row in valid]
    xyz = np.asarray([x["xyz_rmse_m"] for x in values], dtype=np.float64)
    so3 = np.asarray([x["so3_rmse_rad"] for x in values], dtype=np.float64)
    rejects = sum(row.get("gate_decision") == "REJECT" for row in rows)
    summary = {
        "cell_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "reject_count": rejects,
        "gate_counts": {
            decision: sum(row.get("gate_decision") == decision for row in rows)
            for decision in ("PASS", "WARN", "REJECT", "UNSCORABLE")
        },
        "median_xyz_rmse_m": float(np.median(xyz)) if xyz.size else None,
        "p95_xyz_rmse_m": float(np.quantile(xyz, 0.95, method="linear")) if xyz.size else None,
        "median_so3_rmse_rad": float(np.median(so3)) if so3.size else None,
    }
    summary["quality_pass"] = bool(
        len(rows) == 16 and len(valid) == 16 and rejects == 0
        and summary["median_xyz_rmse_m"] is not None and summary["median_xyz_rmse_m"] <= 0.020
        and summary["p95_xyz_rmse_m"] is not None and summary["p95_xyz_rmse_m"] <= 0.035
        and summary["median_so3_rmse_rad"] is not None and summary["median_so3_rmse_rad"] <= 0.050
    )
    return summary


def self_check() -> None:
    good = [
        {"status": "VALID", "gate_decision": "WARN",
         "metrics": {"xyz_rmse_m": 0.010, "so3_rmse_rad": 0.030}}
        for _ in range(16)
    ]
    b0 = evaluate_group(good)
    bad_group = [dict(row) for row in good]
    bad_group[0] = {**bad_group[0], "gate_decision": "REJECT"}
    b1 = evaluate_group(bad_group)
    assert b0["quality_pass"] and not b1["quality_pass"]


def read_card(path: Path, sidecar: Path) -> tuple[dict, str]:
    card = d1.read_json(path)
    digest = sha(path)
    assert sidecar.read_text(encoding="ascii").strip() == digest
    assert card["schema"] == "PI0247_D2_RUN_CARD_V1"
    assert sha(Path(__file__).resolve()) == card["runner_sha256"]
    assert sha(Path(d1.__file__).resolve()) == card["query_helper_sha256"]
    return card, digest


def prepare_inputs(card: dict, context: dict) -> dict:
    import torch

    p1 = context["p1"]
    facts = context["facts"]
    audit_path = Path(card["paths"]["input_identity_audit"])
    assert sha(audit_path) == card["input_identity_audit_sha256"]
    audit = d1.read_json(audit_path)
    assert audit["d1_r_star"] == "S4" and audit["d2_logical_cells"] == 32
    assert audit["exact_d1_b1_cells_reused"] == 4 and audit["new_d2_queries"] == 28

    source_rows = context["rows"]
    sources = [row["source_episode_id"] for row in source_rows]
    audit_rows = {row["source_episode_id"]: row for row in audit["source_rows"]}
    assert len(sources) == 16 and set(sources) == set(audit_rows)
    s0_root = Path(card["paths"]["s0_packets_root"])
    p1_cells = context["source_cells"]
    d2_cells, d2_noise = {}, {}
    noise_b1 = context["noise"]["N1"]
    noise_b1_sha = hashlib.sha256(noise_b1.tobytes()).hexdigest()
    assert noise_b1_sha == audit["authorized_b1_source_tensor_sha256"]

    import pi0242_train_infer_bridge as query_bridge
    for row in source_rows:
        source = row["source_episode_id"]
        base = source.split(":", 1)[0]
        packet_dir = s0_root / base
        source_audit = audit_rows[source]
        record = d1.read_json(packet_dir / "record.json")
        attempt = d1.read_json(packet_dir / "attempt.json")
        assert record["source_episode_id"] == source and attempt["source_episode_id"] == source
        assert record["source_tensor_sha256"] == row["tensor_sha256"] == source_audit["source_tensor_sha256"]
        assert p1.sha(packet_dir / "condition.pt") == record["condition_sha256"]
        assert p1.sha(packet_dir / "noise_packet.npz") == record["noise_packet_sha256"]
        policy = torch.load(packet_dir / "condition.pt", map_location="cpu", weights_only=False)
        condition_identity = query_bridge._canonical_hash(query_bridge._leaf_inventory(policy))
        assert condition_identity == record["condition_identity"] == source_audit["condition_identity"]
        with np.load(packet_dir / "noise_packet.npz", allow_pickle=False) as data:
            b0 = data["source_float32"].copy()
            initial = data["initial_float32"].copy()
        b0_sha = hashlib.sha256(b0.tobytes()).hexdigest()
        assert b0.shape == (1, 40, 132) and np.isfinite(b0).all() and np.array_equal(b0, initial)
        assert b0_sha == source_audit["b0_source_tensor_sha256"]
        assert not np.array_equal(b0, noise_b1)
        d2_noise[source] = {"B0": b0, "B1": noise_b1}
        s0_info = {"cell_path": str(packet_dir), "condition_identity": condition_identity}
        d2_cells[source] = {"B0": s0_info, "B1": s0_info}
        if source in p1_cells:
            assert p1_cells[source]["N1"]["condition_identity"] == condition_identity
            d2_cells[source]["B1"] = p1_cells[source]["N1"]

    context["d2_noise"] = d2_noise
    context["source_cells"] = d2_cells
    context["d2_audit"] = audit
    context["d2_audit_rows"] = audit_rows
    context["d2_p1_source_cells"] = p1_cells
    context["facts"]["noise_files_sha256"]["B1"] = facts["noise_files_sha256"]["N1"]
    return audit


def reuse_d1_b1(
    source: str, card: dict, context: dict, d1_card: dict, d1_card_sha: str,
    expected_manifest_sha: str, expected_noise_sha: str,
) -> dict:
    p1 = context["p1"]
    base = source.split(":", 1)[0]
    cell = Path(card["paths"]["d1_cells_root"]) / f"P2000__S4__{base}__N1"
    rec_path, att_path, tensor_path = cell / "record.json", cell / "attempt.json", cell / "tensors.npz"
    rec, att = d1.read_json(rec_path), d1.read_json(att_path)
    p1_source = context["d2_p1_source_cells"][source]["N1"]
    assert rec["status"] == "VALID" and rec["model_name"] == "P2000" and rec["runtime"] == "S4"
    assert rec["sampler_steps"] == 4 and rec["source_episode_id"] == source and rec["noise_name"] == "N1"
    assert rec["model_manifest_sha256"] == att["model_manifest_sha256"] == expected_manifest_sha
    assert rec["condition_identity"] == p1_source["condition_identity"]
    assert rec["condition_sha256"] == p1_source["condition_sha256"]
    assert rec["source_tensor_sha256"] == att["source_tensor_sha256"]
    assert rec["noise_source_sha256"] == expected_noise_sha
    assert rec["target_xyz_in_model_input"] is False
    assert p1.sha(tensor_path) == rec["tensor_sha256"]
    assert p1.sha(att_path) == rec["attempt_sha256"]
    assert rec["run_card_sha256"] == d1_card_sha
    assert d1_card["resource"]["device"] == card["resource"]["device"]
    assert d1_card["resource"]["gpu_name"] == card["resource"]["gpu_name"]
    return {
        "cell_key": f"P2000__S4__{base}__B1", "status": "REUSED_D1_VALID",
        "model_name": "P2000", "runtime": "S4", "source_episode_id": source,
        "noise_name": "B1", "condition_identity": rec["condition_identity"],
        "condition_sha256": rec["condition_sha256"], "metrics": rec["metrics"],
        "gate_decision": rec["gate_decision"], "gate": rec["gate"],
        "sampler_steps": 4, "record_sha256": p1.sha(rec_path),
        "attempt_sha256": rec["attempt_sha256"], "tensor_sha256": rec["tensor_sha256"],
        "source_cell": str(cell), "source_run_card_sha256": rec["run_card_sha256"],
        "execution_disposition": "REUSED_D1_VALID",
    }


def summarize(
    card: dict, card_sha: str, context: dict, rows: list[dict],
    noise_identities: dict,
) -> dict:
    for row in rows:
        tensor_name = "invalid_output.npz" if row.get("status") in INVALID else "tensors.npz"
        with np.load(Path(row["source_cell"]) / tensor_name, allow_pickle=False) as data:
            row["noise_initial_sha256"] = hashlib.sha256(data["noise_initial"].tobytes()).hexdigest()
    by_group = {group: [row for row in rows if row["noise_name"] == group] for group in ("B0", "B1")}
    groups = {group: evaluate_group(group_rows) for group, group_rows in by_group.items()}
    noise_identities["B0"] = {
        source: {**identity, "initial_bfloat16_sha256": next(
            row["noise_initial_sha256"] for row in by_group["B0"] if row["source_episode_id"] == source
        )}
        for source, identity in noise_identities["B0"].items()
    }
    b1_initial = {row["noise_initial_sha256"] for row in by_group["B1"]}
    assert len(b1_initial) == 1
    noise_identities["B1"]["initial_bfloat16_sha256"] = next(iter(b1_initial))
    typed_failure = any(row.get("status") in INVALID for row in rows)
    both_pass = all(result["quality_pass"] for result in groups.values())
    if typed_failure:
        status, route = "D2_TYPED_FAILURE_STOP", "STOP_BEFORE_TRAINING"
    elif both_pass:
        status, route = "D2_PASS_BOTH_NOISE_GROUPS", "DIRECT_M64"
    else:
        status, route = "D2_VALID_FAIL_LU_REQUIRED", "RUN_BOTH_LU_2000"
    return {
        "schema": "PI0247_D2_FULL16_REPORT_V1", "run_card_sha256": card_sha,
        "release_commit": card["release_commit"], "execution_parent": card["execution_parent"],
        "d1_run_card_sha256": card["d1_run_card_sha256"],
        "d1_report_sha256": card["d1_report_sha256"],
        "d2_input_identity_audit_sha256": card["input_identity_audit_sha256"],
        "r_star_frozen": card["d1_r_star"], "runtime_ids_deduplicated": ["S4"],
        "weight": card["weights"][0], "source_episode_ids": [x["source_episode_id"] for x in context["rows"]],
        "noise_identity": noise_identities,
        "thresholds_per_group": {"invalid_count": 0, "reject_count": 0,
            "median_xyz_rmse_m_max": 0.020, "p95_xyz_rmse_m_max": 0.035,
            "median_so3_rmse_rad_max": 0.050, "p95_method": "linear", "valid_points": 40},
        "counts": {"logical_cells": 32,
            "new_queries_this_run": sum(x.get("execution_disposition") == "NEW_QUERY_THIS_RUN" for x in rows),
            "reused_d1_valid_cells": sum(x["status"] == "REUSED_D1_VALID" for x in rows),
            "reused_completed_d2_cells": sum(x.get("execution_disposition") == "REUSED_D2_COMPLETED" for x in rows),
            "head_denoising_forwards": sum(int(x.get("head_denoising_forwards", x.get("sampler_steps", 0))) for x in rows)},
        "group_results": groups, "route": route, "status": status, "cells": rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run(args: argparse.Namespace) -> None:
    card, card_sha = read_card(args.card, args.card_sha)
    code_root = Path(card["paths"]["code_root"])
    facts, context = d1.prepare(Path(card["paths"]["stage_root"]), code_root)
    context["facts"] = facts
    assert d1.canonical_sha(facts) == card["static_preflight_canonical_sha256"]
    assert card["release_commit"] == "50e66efe6ccfbe814d04caf12499493821be7957"
    assert card["d1_r_star"] == "S4" and card["runtimes"] == [S4]
    assert [weight["id"] for weight in card["weights"]] == ["P2000"]
    assert card["padding_proof"] == facts["padding_proof"] and facts["m_applicable"]
    assert str(Path(card["paths"]["output_root"])) == str(args.output_root)
    assert sha(Path(card["paths"]["d1_report"])) == card["d1_report_sha256"]
    audit = prepare_inputs(card, context)
    assert len(context["rows"]) == 16 and len(audit["source_rows"]) == 16

    p1 = context["p1"]
    weight = card["weights"][0]
    model_root = Path(weight["model_root"])
    model_files = p1.verify_model_files(model_root)
    assert model_files["manifest_sha256"] == weight["manifest_sha256"]
    assert model_files["revision"] == weight["revision"]
    restore = d1.read_json(model_root / "pi0245_inference_restore.json")
    assert restore["checkpoint_step"] == 2000 and weight["family"] in restore["prefix"]

    d1_card_path = Path(card["paths"]["d1_run_card"])
    d1_card = d1.read_json(d1_card_path)
    d1_card_sha = sha(d1_card_path)
    assert d1_card_sha == card["d1_run_card_sha256"]
    report_d1 = d1.read_json(Path(card["paths"]["d1_report"]))
    assert report_d1["run_card_sha256"] == d1_card_sha and report_d1["p_weight_routing"]["r_star"] == "S4"
    n1_sha = hashlib.sha256(context["noise"]["N1"].tobytes()).hexdigest()
    reused_d1 = {
        source: reuse_d1_b1(source, card, context, d1_card, d1_card_sha,
            model_files["manifest_sha256"], n1_sha)
        for source in context["d2_p1_source_cells"]
    }
    assert len(reused_d1) == 4 and card["expected_counts"]["new_queries"] == 28
    if args.preflight_only:
        print(json.dumps({"status": "D2_PREFLIGHT_PASS", "logical_cells": 32,
            "new_queries": 28, "reused_d1_b1": len(reused_d1),
            "model_manifest_sha256": model_files["manifest_sha256"]}, sort_keys=True), flush=True)
        return

    from transformers import AutoModel
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=card["resource"]["device"], dtype=__import__("torch").bfloat16)
    assert int(model.action_head.num_inference_timesteps) == 4

    rstar = S4
    gate = TrajectorySafetyGateV1()
    all_rows, new_records = [], []
    noise_identities = {"B0": {}, "B1": {"shared_across_sources": True,
        "source_tensor_sha256": n1_sha,
        "file_sha256": facts["noise_files_sha256"]["N1"]}}
    for source_row in context["rows"]:
        source = source_row["source_episode_id"]
        base = source.split(":", 1)[0]
        source_audit = context["d2_audit_rows"][source]
        noise_identities["B0"][source] = {
            "source_tensor_sha256": source_audit["b0_source_tensor_sha256"],
            "noise_packet_sha256": source_audit["s0_noise_packet_sha256"],
            "per_source_packet": True,
        }
        for group in ("B0", "B1"):
            if group == "B1" and source in context["d2_p1_source_cells"]:
                record = reused_d1[source]
            else:
                source_dir = Path(context["source_cells"][source][group]["cell_path"])
                if group == "B0":
                    noise_path = source_dir / "noise_packet.npz"
                    noise_sha = p1.sha(noise_path)
                    context["noise"][group] = context["d2_noise"][source][group]
                else:
                    noise_path = Path(context["noise_root"]) / "N1.npz"
                    noise_sha = p1.sha(noise_path)
                    context["noise"][group] = context["d2_noise"][source][group]
                context["facts"]["noise_files_sha256"][group] = noise_sha
                cell_key = f"P2000__S4__{base}__{group}"
                cell_path = args.output_root / "cells" / cell_key
                existed = cell_path.exists()
                record = d1.run_one(
                    args=args, card=card, card_sha=card_sha, context=context,
                    model_name="P2000", runtime=rstar, source=source, noise_name=group,
                    model=model, model_files=model_files, bridge=bridge, selector=selector,
                    stage43=stage43, gate=gate,
                )
                record = {**record, "source_cell": str(cell_path),
                    "execution_disposition": "REUSED_D2_COMPLETED" if existed else "NEW_QUERY_THIS_RUN"}
                new_records.append(record)
            all_rows.append(record)

    with (args.output_root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in new_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = summarize(card, card_sha, context, all_rows, noise_identities)
    p1.durable_json(args.output_root / "pi0247_d2_full16_report.json", report)
    del model
    __import__("torch").cuda.empty_cache()
    print(json.dumps({"status": report["status"], "route": report["route"],
        "groups": report["group_results"], "report": str(args.output_root / "pi0247_d2_full16_report.json")}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--card", type=Path)
    parser.add_argument("--card-sha", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("PI0247 D2 per-noise gate self-check: PASS")
        return
    assert args.card and args.card_sha and args.output_root
    run(args)


if __name__ == "__main__":
    main()
