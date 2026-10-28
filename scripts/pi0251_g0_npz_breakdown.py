#!/usr/bin/env python3
"""Read-only PI0251 G0 breakdown of the completed PI0250 K/X screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "median": float(np.median(array)),
            "p95_linear": float(np.quantile(array, .95, method="linear")),
            "max": float(np.max(array))}


def summarize_group(rows: list[dict], report_group: dict, horizon_rows: list[np.ndarray]) -> dict:
    residual = np.stack(horizon_rows).astype(np.float64)
    per_step = np.linalg.norm(residual, axis=-1)
    mean_bias_by_step = residual.mean(axis=0)
    rmse_by_step = np.sqrt(np.mean(per_step ** 2, axis=0))
    result = {
        "windows": len(rows),
        "gate_counts": dict(Counter(row["gate"] for row in rows)),
        "quality_gate_from_original_report": {
            key: report_group.get(key) for key in (
                "quality_pass", "reject_count", "median_xyz_rmse_m",
                "p95_xyz_rmse_m", "median_so3_rmse_rad")
        },
        "npz_recomputed_xyz_rmse_m": quantiles([row["xyz_rmse_m"] for row in rows]),
        "npz_recomputed_so3_rmse_rad_from_original_report": quantiles(
            [row["so3_rmse_rad"] for row in rows]),
        "position_bias": {
            "mean_window_bias_xyz_m": np.mean(
                [row["position_bias_xyz_m"] for row in rows], axis=0).tolist(),
            "median_window_bias_norm_m": float(np.median(
                [row["position_bias_norm_m"] for row in rows])),
            "median_bias_energy_fraction": float(np.median(
                [row["bias_energy_fraction"] for row in rows])),
        },
        "temporal_error": {
            "mean_bias_xyz_by_horizon_m": mean_bias_by_step.tolist(),
            "pooled_rmse_m_by_horizon": rmse_by_step.tolist(),
            "median_rmse_m_by_band": {
                key: float(np.median([row["band_rmse_m"][key] for row in rows]))
                for key in ("h00_09", "h10_19", "h20_29", "h30_39")
            },
            "median_tail35_39_sse_fraction": float(np.median(
                [row["tail35_39_sse_fraction"] for row in rows])),
            "median_last10_to_first10_rmse_ratio": float(np.median(
                [row["last10_to_first10_rmse_ratio"] for row in rows])),
            "median_residual_delta_max_mps": float(np.median(
                [row["residual_delta_max_mps"] for row in rows])),
        },
        "raw_xyz_boundary_exposure": {
            "scalars_abs_gt_1_total": sum(row["raw_xyz_abs_gt_1"] for row in rows),
            "cells_with_any_abs_gt_1": sum(row["raw_xyz_abs_gt_1"] > 0 for row in rows),
            "scalars_abs_eq_1_total": sum(row["raw_xyz_abs_eq_1"] for row in rows),
            "scalars_abs_ge_0_99_total": sum(row["raw_xyz_abs_ge_0_99"] for row in rows),
            "scalars_scanned": len(rows) * 40 * 3,
            "max_abs_raw_xyz": max(row["raw_xyz_max_abs"] for row in rows),
        },
        "target_motion": {
            "median_expert_path_length_m": float(np.median(
                [row["expert_path_length_m"] for row in rows])),
            "median_expert_max_speed_mps": float(np.median(
                [row["expert_max_speed_mps"] for row in rows])),
        },
        "hardest_windows_by_xyz_rmse": [
            {key: row[key] for key in (
                "source_episode_id", "support", "gate", "xyz_rmse_m",
                "position_bias_xyz_m", "bias_energy_fraction",
                "band_rmse_m", "tail35_39_sse_fraction",
                "residual_delta_max_mps", "raw_xyz_abs_gt_1",
                "expert_path_length_m", "expert_max_speed_mps", "tensor_sha256")}
            for row in sorted(rows, key=lambda row: row["xyz_rmse_m"], reverse=True)[:8]
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi0250-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.pi0250_root.resolve()
    arms: dict[str, dict] = {}
    shared_support: set[str] | None = None
    shared_old16: set[str] | None = None
    validated_cells = 0
    checked_metrics = 0
    all_details: list[dict] = []
    residuals_by_key: dict[tuple[str, str, str], list[np.ndarray]] = {}

    for arm in ("K2000", "X2000"):
        screen = root / "screen" / arm
        report_path = screen / "screen_report.json"
        card_path = screen / "score_card.json"
        cpu_path = screen / "cpu_replay.json"
        report, card, cpu = read(report_path), read(card_path), read(cpu_path)
        support = set(card["support"]["source_episode_ids"])
        old16 = set(card["support"]["old16_source_ids"])
        if len(support) != 64 or len(old16) != 16 or not old16 <= support:
            raise ValueError(f"{arm}: frozen cohort identity is not 64/16")
        if shared_support is None:
            shared_support, shared_old16 = support, old16
        elif support != shared_support or old16 != shared_old16:
            raise ValueError(f"{arm}: source cohort differs from paired arm")
        if report.get("status") != "SCREEN_COMPLETE" or report.get("checkpoint_step") != 2000:
            raise ValueError(f"{arm}: screen report is not the completed final-2000 screen")
        if card.get("selected_checkpoint") != "NONE" or report.get("selected_checkpoint") != "NONE":
            raise ValueError(f"{arm}: selected checkpoint field must remain NONE")
        if report.get("score_card_sha256") != sha(card_path):
            raise ValueError(f"{arm}: score-card hash does not match screen report")
        if report.get("source_episode_ids") and set(report["source_episode_ids"]) != support:
            raise ValueError(f"{arm}: report source set differs from frozen cohort")
        cells = report["cells"]
        expected = {(source, noise) for source in support for noise in ("B0", "B1")}
        seen: set[tuple[str, str]] = set()
        detail: list[dict] = []
        residual_lookup: dict[tuple[str, str], np.ndarray] = {}
        for item in cells:
            source, noise = item["source_episode_id"], item["noise_name"]
            key = (source, noise)
            if key in seen or key not in expected:
                raise ValueError(f"{arm}: unexpected or duplicate cell {key}")
            seen.add(key)
            if item.get("status") != "VALID":
                raise ValueError(f"{arm}: non-valid screen cell {key}")
            cell = Path(item["source_cell"])
            record_path = cell / "record.json"
            tensor_path = cell / "tensors.npz"
            if sha(record_path) != item["record_sha256"]:
                raise ValueError(f"{arm}: record hash mismatch for {key}")
            record = read(record_path)
            tensor_hash = sha(tensor_path)
            if tensor_hash != item["tensor_sha256"] or tensor_hash != record["tensor_sha256"]:
                raise ValueError(f"{arm}: NPZ hash mismatch for {key}")
            with np.load(tensor_path, allow_pickle=False) as data:
                raw_native = data["raw_native"].copy()
                raw_scored = data["raw_scored"].copy()
                decoded = data["decoded"][0].astype(np.float64)
                expert = data["expert"].astype(np.float64)
            if raw_native.shape != (1, 40, 132) or decoded.shape != expert.shape or decoded.shape != (40, 10):
                raise ValueError(f"{arm}: unexpected tensor shape for {key}")
            if not np.isfinite(decoded).all() or not np.isfinite(expert).all():
                raise ValueError(f"{arm}: non-finite saved physical action for {key}")
            if not np.array_equal(raw_native, raw_scored):
                raise ValueError(f"{arm}: raw scored/native differ for {key}")
            residual = decoded[:, :3] - expert[:, :3]
            point_error = np.linalg.norm(residual, axis=1)
            energy = float(np.mean(point_error ** 2))
            xyz_rmse = math.sqrt(energy)
            bias = residual.mean(axis=0)
            centered_energy = float(np.mean(np.sum((residual - bias) ** 2, axis=1)))
            if not math.isclose(energy, float(np.dot(bias, bias)) + centered_energy,
                                rel_tol=1e-10, abs_tol=1e-12):
                raise ArithmeticError(f"{arm}: bias/variance decomposition failed for {key}")
            bands = {f"h{start:02d}_{start + 9:02d}": float(np.sqrt(np.mean(
                np.sum(residual[start:start + 10] ** 2, axis=1))))
                for start in (0, 10, 20, 30)}
            tail_energy = float(np.sum(residual[35:40] ** 2))
            total_energy = float(np.sum(residual ** 2))
            tail_fraction = tail_energy / total_energy if total_energy else 0.0
            delta_max = float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05)
            actual_metrics = item["metrics"]
            for metric, actual in (("xyz_rmse_m", xyz_rmse),
                                   ("tail35_39_pooled_sse_fraction", tail_fraction),
                                   ("physical_residual_delta_max_mps", delta_max)):
                if not math.isclose(float(actual_metrics[metric]), actual,
                                    rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"{arm}: {metric} does not replay from NPZ for {key}")
                checked_metrics += 1
            raw_xyz = raw_native[0, :, :3]
            expert_steps = np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1) / .05
            source_row = {
                "source_episode_id": source,
                "support": "old16" if source in old16 else "new48",
                "noise": noise,
                "gate": item["gate_decision"],
                "xyz_rmse_m": xyz_rmse,
                "so3_rmse_rad": float(actual_metrics["so3_rmse_rad"]),
                "position_bias_xyz_m": bias.tolist(),
                "position_bias_norm_m": float(np.linalg.norm(bias)),
                "bias_energy_fraction": float(np.dot(bias, bias) / energy) if energy else 0.0,
                "band_rmse_m": bands,
                "tail35_39_sse_fraction": tail_fraction,
                "last10_to_first10_rmse_ratio": bands["h30_39"] / bands["h00_09"] if bands["h00_09"] else 0.0,
                "residual_delta_max_mps": delta_max,
                "raw_xyz_abs_gt_1": int(np.count_nonzero(np.abs(raw_xyz) > 1.0)),
                "raw_xyz_abs_eq_1": int(np.count_nonzero(np.abs(raw_xyz) == 1.0)),
                "raw_xyz_abs_ge_0_99": int(np.count_nonzero(np.abs(raw_xyz) >= .99)),
                "raw_xyz_max_abs": float(np.max(np.abs(raw_xyz))),
                "expert_path_length_m": float(np.sum(np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1))),
                "expert_max_speed_mps": float(expert_steps.max()),
                "tensor_sha256": tensor_hash,
                "record_sha256": sha(record_path),
            }
            detail.append(source_row)
            all_details.append({"arm": arm, **source_row})
            residual_lookup[key] = residual
            validated_cells += 1
        if seen != expected or len(cells) != 128:
            raise ValueError(f"{arm}: final screen does not close the 64x2 matrix")

        groups = {}
        for noise in ("B0", "B1"):
            report_groups = (
                ("all64", support, report["group_results"][noise]),
                ("old16", old16, report["old16_group_results"][noise]),
                ("new48", support - old16, None),
            )
            for cohort, ids, report_group in report_groups:
                members = [row for row in detail if row["noise"] == noise and row["source_episode_id"] in ids]
                if report_group is None:
                    report_group = {
                        "quality_pass": None, "reject_count": sum(row["gate"] == "REJECT" for row in members),
                        "median_xyz_rmse_m": None, "p95_xyz_rmse_m": None, "median_so3_rmse_rad": None,
                    }
                rows_by_cell = [residual_lookup[(row["source_episode_id"], noise)] for row in members]
                groups[f"{cohort}_{noise}"] = summarize_group(members, report_group, rows_by_cell)
                if cohort != "new48":
                    computed = groups[f"{cohort}_{noise}"]["npz_recomputed_xyz_rmse_m"]
                    if not math.isclose(computed["median"], report_group["median_xyz_rmse_m"], rel_tol=1e-8, abs_tol=1e-10):
                        raise ValueError(f"{arm}/{cohort}/{noise}: median does not match screen report")
                    if not math.isclose(computed["p95_linear"], report_group["p95_xyz_rmse_m"], rel_tol=1e-8, abs_tol=1e-10):
                        raise ValueError(f"{arm}/{cohort}/{noise}: p95 does not match screen report")
        arms[arm] = {
            "screen_report": {"path": str(report_path), "sha256": sha(report_path),
                              "schema": report.get("schema"), "status": report["status"],
                              "counts": report["counts"]},
            "score_card": {"path": str(card_path), "sha256": sha(card_path),
                           "checkpoint_revision": card["hf_revision"],
                           "checkpoint_tree_sha256": card["checkpoint_tree_sha256"],
                           "source64_manifest_sha256": card["source64_manifest_sha256"],
                           "frozen_screen_inputs_sha256": card["frozen_screen_inputs_sha256"],
                           "noise_manifest_sha256": card["noise_manifest_sha256"],
                           "durable_receipt_sha256": card["durable_receipt_sha256"],
                           "hf_readback_sha256": card["hf_readback_sha256"],
                           "sampler_steps": card["sampler_steps"],
                           "selected_checkpoint": card["selected_checkpoint"]},
            "cpu_replay": {"path": str(cpu_path), "sha256": sha(cpu_path),
                           "schema": cpu.get("schema"), "status": cpu.get("status"),
                           "cell_count": len(cpu.get("cells", cpu.get("replay", [])))},
            "groups": groups,
        }

    result = {
        "schema": "PI0251_G0_ACTUAL_NPZ_BREAKDOWN_V1",
        "status": "PASS_READ_ONLY_NPZ_BREAKDOWN",
        "scope": "Existing PI0250 K2000/X2000 final-2000 screen only; no model queries, inference, retraining, threshold, or sample changes.",
        "cohort": {"source_count": 64, "old16_count": 16, "new48_count": 48,
                   "source64_manifest_sha256": arms["K2000"]["score_card"]["source64_manifest_sha256"],
                   "old16_source_ids": sorted(shared_old16 or [])},
        "pairwise_screen_inputs_match": all(
            arms["K2000"]["score_card"][key] == arms["X2000"]["score_card"][key]
            for key in ("source64_manifest_sha256", "frozen_screen_inputs_sha256", "noise_manifest_sha256")),
        "npz_integrity": {"valid_cells": validated_cells, "expected_cells": 256,
                          "record_and_tensor_hashes_verified": True,
                          "recomputed_metrics_compared_per_cell": checked_metrics,
                          "raw_native_equals_raw_scored": True,
                          "physical_metric": "sqrt(mean_t(sum_xyz(residual[t]^2))); no division by xyz dimension"},
        "arms": arms,
        "model_queries_added": 0,
    }
    if validated_cells != 256 or not result["pairwise_screen_inputs_match"]:
        raise ValueError("G0 evidence closure failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to replace conflicting G0 output: {args.output}")
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": sha(args.output), "cells": validated_cells,
                      "metric_checks": checked_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
