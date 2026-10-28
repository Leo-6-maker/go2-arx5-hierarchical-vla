#!/usr/bin/env python3
"""Read-only PI0248 breakdown of the sealed PI0247 M64 screens."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p95_linear": float(np.quantile(a, .95, method="linear")),
            "max": float(a.max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exposure-audit", type=Path, required=True)
    p.add_argument("--rotation-check", type=Path, required=True)
    args = p.parse_args()

    import pi0247_m64_screen as m64
    import pi0247_d1_screen as d1
    card, fit, ctx = m64.inputs()
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix
    old16 = set(card["support"]["old16_source_ids"])
    row_by_source = {r["source_episode_id"]: r for r in ctx["rows"]}
    assert len(row_by_source) == 64 and len(old16) == 16
    slices = {"old16": old16, "new48": set(row_by_source) - old16}
    assert len(slices["new48"]) == 48
    reports = {}
    breakdown = {}
    per_source = {}
    for step in (0, 1000, 2500, 5000):
        path = m64.M64 / f"screen_step{step}" / "screen_report.json"
        report = _read(path)
        assert report["checkpoint_step"] == step and report["selected_checkpoint"] == "NONE"
        reports[str(step)] = {"path": str(path), "sha256": m64.sha(path),
                              "status": report["status"], "counts": report["counts"]}
        rows = report["cells"]
        expected_noises = {"B0", "B1"} if step == 5000 else {"B0"}
        assert len(rows) == 64 * len(expected_noises)
        details = []
        for row in rows:
            source, noise = row["source_episode_id"], row["noise_name"]
            assert source in row_by_source and noise in expected_noises
            rec_path = Path(row["source_cell"]) / "record.json"
            rec = d1.collect_record(rec_path, ctx["p1"])
            assert rec["source_episode_id"] == source and rec["status"] in ("VALID",)
            cell = rec_path.parent
            with np.load(cell / "tensors.npz", allow_pickle=False) as z:
                decoded, expert = z["decoded"][0].copy(), z["expert"].copy()
                state, raw = z["state"].copy(), z["raw_scored"][0].copy()
            assert decoded.shape == expert.shape == (40, 10) and state.shape == (10,)
            angles = geodesic_angle(rot6d_rows_to_matrix(decoded[:, 3:9]),
                                    rot6d_rows_to_matrix(expert[:, 3:9]))
            residual = decoded[:, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
            xyz_rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            so3_rmse = float(np.sqrt(np.mean(angles ** 2)))
            metrics = row["metrics"]
            assert math.isclose(xyz_rmse, metrics["xyz_rmse_m"], rel_tol=1e-9, abs_tol=1e-11)
            assert math.isclose(so3_rmse, metrics["so3_rmse_rad"], rel_tol=1e-9, abs_tol=1e-11)
            meta = row_by_source[source]
            base = source.split(":obs-", 1)[0]
            details.append({"source_episode_id": source, "rollout_base": base,
                "support": "old16" if source in old16 else "new48",
                "target_family": meta.get("target_family"), "phase": meta.get("phase"),
                "noise": noise, "status": row["status"], "gate": row["gate_decision"],
                "xyz_rmse_m": xyz_rmse, "so3_rmse_rad": so3_rmse,
                "so3_p95_rad": float(np.quantile(angles, .95, method="linear")),
                "position_bias_xyz_m": residual.mean(axis=0).tolist(),
                "position_tail35_39_sse_fraction": float(np.sum(residual[35:] ** 2) /
                    np.sum(residual ** 2)) if np.sum(residual ** 2) else 0.0,
                "position_residual_delta_max_mps": float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05),
                "position_clipped_values": int(np.count_nonzero(np.abs(raw[:, :3]) > 1.0)),
                "rotation_clipped_values": int(np.count_nonzero(np.abs(raw[:, 3:9]) > 1.0)),
                "rotation_error_by_horizon_rad": angles.tolist(),
                "position_residual_by_horizon_xyz_m": residual.tolist()})
            per_source.setdefault(source, {"target_family": meta.get("target_family"),
                "phase": meta.get("phase"), "rollout_base": base,
                "target_xyz_mean_m": expert[:, :3].mean(axis=0).tolist(),
                "target_xyz_std_m": expert[:, :3].std(axis=0).tolist(),
                "target_rot6d_mean": expert[:, 3:9].mean(axis=0).tolist(),
                "target_rot6d_std": expert[:, 3:9].std(axis=0).tolist()})
        by = {}
        for support, ids in slices.items():
            support_rows = [x for x in details if x["source_episode_id"] in ids]
            by[support] = {}
            for noise in sorted(expected_noises):
                for family in sorted({str(x["target_family"]) for x in support_rows}):
                    for phase in sorted({x["phase"] for x in support_rows if str(x["target_family"]) == family}, key=str):
                        group = [x for x in support_rows if x["noise"] == noise
                                 and str(x["target_family"]) == family and x["phase"] == phase]
                        if not group:
                            continue
                        key = f"{noise}/{family}/phase-{phase}"
                        by[support][key] = {
                            "windows": len(group), "rollout_bases": len({x["rollout_base"] for x in group}),
                            "gate_counts": dict(Counter(x["gate"] for x in group)),
                            "xyz_rmse_m": _stats([x["xyz_rmse_m"] for x in group]),
                            "so3_rmse_rad": _stats([x["so3_rmse_rad"] for x in group]),
                            "so3_p95_per_window_rad": _stats([x["so3_p95_rad"] for x in group]),
                            "worst_xyz": max(group, key=lambda x: x["xyz_rmse_m"])["source_episode_id"],
                            "worst_so3": max(group, key=lambda x: x["so3_rmse_rad"])["source_episode_id"],
                            "tail35_39_sse_fraction_median": float(np.median([x["position_tail35_39_sse_fraction"] for x in group])),
                            "position_bias_xyz_m_mean": np.mean([x["position_bias_xyz_m"] for x in group], axis=0).tolist(),
                            "position_clipped_values": sum(x["position_clipped_values"] for x in group),
                            "rotation_clipped_values": sum(x["rotation_clipped_values"] for x in group),
                        }
        breakdown[str(step)] = {"available_noise_groups": sorted(expected_noises),
                                "support_phase_groups": by, "cells": details}

    u_report_path = m64.ROOT / "lu/U_screen_step2000/final_screen_report.json"
    u_report = _read(u_report_path)
    u_by_key = {(x["source_episode_id"], x["noise_name"]): x for x in u_report["cells"]}
    m64_zero = {(x["source_episode_id"], x["noise"]): x
                for x in breakdown["0"]["cells"]}
    matches = []
    final_by_key = {(x["source_episode_id"], x["noise"]): x
                    for x in breakdown["5000"]["cells"]}
    start_report_rows = {(x["source_episode_id"], x["noise_name"]): x
                         for x in _read(m64.M64 / "screen_step0/screen_report.json")["cells"]}
    for source in sorted(old16):
        for noise in ("B0", "B1"):
            u = u_by_key.get((source, noise))
            m = m64_zero.get((source, noise))
            u_record = (d1.collect_record(Path(u["source_cell"]) / "record.json", ctx["p1"])
                        if u else None)
            m_row = start_report_rows.get((source, noise))
            final = final_by_key.get((source, noise))
            final_row = next((r for r in _read(m64.M64 / "screen_step5000/screen_report.json")["cells"]
                              if r["source_episode_id"] == source and r["noise_name"] == noise), None)
            m_record = (d1.collect_record(Path(final_row["source_cell"]) / "record.json", ctx["p1"])
                        if final_row else None)
            matches.append({"source_episode_id": source, "noise": noise,
                "u_status": u.get("status") if u else None,
                "m64_start_status": m.get("status") if m else None,
                "same_scoring_input_and_noise": bool(u_record and m_record
                    and u_record.get("condition_identity") == m_record.get("condition_identity")
                    and u_record.get("source_tensor_sha256") == m_record.get("source_tensor_sha256")
                    and u_record.get("noise_source_sha256") == m_record.get("noise_source_sha256")),
                "u_condition_identity": u_record.get("condition_identity") if u_record else None,
                "m64_start_condition_identity": m_record.get("condition_identity") if m_record else None,
                "u_noise_source_sha256": u_record.get("noise_source_sha256") if u_record else None,
                "m64_start_noise_source_sha256": m_record.get("noise_source_sha256") if m_record else None,
                "u_xyz_rmse_m": (u.get("metrics") or {}).get("xyz_rmse_m") if u else None,
                "m64_final_xyz_rmse_m": final.get("xyz_rmse_m") if final else None,
                "u_so3_rmse_rad": (u.get("metrics") or {}).get("so3_rmse_rad") if u else None,
                "m64_final_so3_rmse_rad": final.get("so3_rmse_rad") if final else None,
                "delta_xyz_rmse_m_m64_minus_u": (final.get("xyz_rmse_m") - (u.get("metrics") or {}).get("xyz_rmse_m"))
                    if final and u and (u.get("metrics") or {}).get("xyz_rmse_m") is not None else None,
                "delta_so3_rmse_rad_m64_minus_u": (final.get("so3_rmse_rad") - (u.get("metrics") or {}).get("so3_rmse_rad"))
                    if final and u and (u.get("metrics") or {}).get("so3_rmse_rad") is not None else None,
                "final_gate": final.get("gate") if final else None})
    training = _read(m64.M64 / "training_manifest.json")
    exposure = _read(args.exposure_audit)
    assert exposure["status"] == "PASS_EXACT_PLANNED_ACTUAL_5000"
    assert exposure["training_manifest_schedule_sha256"] == training["schedule_sha256"]
    rotation_check = _read(args.rotation_check)
    assert rotation_check["status"] == "PASS_CPU_DECODER_AND_GRADIENT"
    source_ids = sorted(row_by_source)
    bases = {source: source.split(":obs-", 1)[0] for source in source_ids}
    result = {
        "schema": "PI0248_M64_G0_BREAKDOWN_V3", "status": "PASS_READ_ONLY_BREAKDOWN",
        "run_card_sha256": m64.sha(m64.START / "run_card.json"),
        "fit_recipe_sha256": m64.sha(m64.ROOT / "lu/fit_recipe_manifest.json"),
        "source64_manifest_sha256": m64.sha(m64.SOURCES / "manifest.json"),
        "source_support": {"count": 64, "old16": sorted(old16), "new48": sorted(slices["new48"]),
            "unique_rollout_bases": len(set(bases.values())),
            "rollout_bases_by_support": {k: len({bases[s] for s in v}) for k, v in slices.items()},
            "old16_arrays_and_condition_match_parent": True,
            "processor_identity": fit["processor_identity"],
            "training_manifest_sha256": m64.sha(m64.M64 / "training_manifest.json"),
            "training_manifest_status": training.get("status"),
            "actual_global_step": training.get("global_step"),
            "actual_sample_consumption": training.get("sample_consumption"),
            "actual_schedule_sha256": training.get("schedule_sha256"),
            "sample_contract_count": len(training.get("sample_contracts", [])),
            "forbidden_counts": training.get("forbidden_counts")},
        "training_exposure": exposure,
        "screen_reports": reports, "screens_by_support_phase": breakdown,
        "rotation_decoder_cpu_check": {"status": rotation_check["status"],
            "sha256": m64.sha(args.rotation_check), "path": str(args.rotation_check),
            "matrix_max_abs": rotation_check["cpu_decoder_parity"]["matrix_max_abs"],
            "gradient_finite_difference": rotation_check["self_check"]["finite_difference_centered"],
            "rotation_bounds_sha256": rotation_check["pinned_statistics"]["rotation_minmax_sha256"]},
        "old16_u_vs_m64_comparison": {"u_report_sha256": m64.sha(u_report_path),
            "matched_cells": len(matches), "same_scoring_input_and_noise_matches": sum(x["same_scoring_input_and_noise"] for x in matches),
            "identity_note": "only exact source/noise tensor identity counts; absent cells or hash mismatches remain unmatched",
            "cells": matches},
        "per_source_expert_target_summary": per_source,
        "unobserved": ["historical per-dimension native rotation loss", "historical per-dimension gradient conflict"],
        "model_queries_added": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to replace conflicting G0 report: {args.output}")
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
        "sha256": m64.sha(args.output), "source_count": 64, "old16": 16, "new48": 48,
        "matched_old16_u_cells": result["old16_u_vs_m64_comparison"]["same_scoring_input_and_noise_matches"],
        "reports": list(reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
