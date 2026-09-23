#!/usr/bin/env python3
"""Read sealed R0/R2 final TRAIN NPZs; no model, sampler, or data mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


REPORT_SHA = "d73ed22d90567ab5c4aaef0052fede9b5e1f9ac1a045b9976f237b4d348a8580"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(root: Path, expected_ids: set[str], expected_sha: str, lo: np.ndarray, hi: np.ndarray) -> dict:
    records = root / "screen" / "step-10000" / "records.jsonl"
    assert file_sha(records) == expected_sha
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
    assert len(rows) == len(expected_ids) == 16
    assert {str(row["source_episode_id"]) for row in rows} == expected_ids

    error_sq, prediction_jump, expert_jump, error_jump, raw_jump, clipped_jump = [], [], [], [], [], []
    source_rows, inputs = [], {}
    for row in rows:
        source = str(row["source_episode_id"])
        path = Path(row["tensor_path"]).resolve()
        assert path.parent == (root / "screen" / "step-10000" / "cells").resolve()
        assert file_sha(path) == row["tensor_sha256"]
        with np.load(path, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        assert arrays["raw"].shape == (1, 40, 132)
        assert arrays["decoded"].shape == (1, 40, 10)
        assert arrays["expert"].shape == (40, 10)
        assert arrays["action_mask"].shape == (40, 132)
        assert all(np.isfinite(value).all() for value in arrays.values())

        raw = arrays["raw"][0, :, :3].astype(np.float64)
        clipped = np.clip(raw, -1.0, 1.0)
        relative = (clipped + 1.0) * (hi - lo) / 2.0 + lo
        pred = arrays["decoded"][0, :, :3].astype(np.float64)
        expert = arrays["expert"][:, :3].astype(np.float64)
        err = pred - expert
        e2 = np.sum(err * err, axis=1)
        pj = np.linalg.norm(np.diff(pred, axis=0), axis=1)
        ej = np.linalg.norm(np.diff(expert, axis=0), axis=1)
        dj = np.linalg.norm(np.diff(err, axis=0), axis=1)
        rj = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        cj = np.linalg.norm(np.diff(clipped, axis=0), axis=1)
        relative_jump = np.linalg.norm(np.diff(relative, axis=0), axis=1)
        geometry_max_error = float(np.max(np.abs(relative_jump - pj)))
        assert geometry_max_error < 1e-5, (source, geometry_max_error)

        # Exact decomposition of the decoded relative XYZ increment at segment 37.
        raw_part = (clipped[38] - clipped[37]) * (hi[37] - lo[37]) / 2.0
        stats_part = (lo[38] - lo[37]) + (clipped[38] + 1.0) * ((hi[38] - lo[38]) - (hi[37] - lo[37])) / 2.0
        assert np.allclose(raw_part + stats_part, relative[38] - relative[37], atol=1e-12)
        mask = arrays["action_mask"]
        source_rows.append({
            "source_episode_id": source,
            "family": row["target_family"], "phase": row["phase"],
            "predicted_peak_segment": int(np.argmax(pj)),
            "raw_xyz_peak_segment": int(np.argmax(rj)),
            "clipped_raw_xyz_peak_segment": int(np.argmax(cj)),
            "predicted_jump_37_m": float(pj[37]),
            "expert_jump_37_m": float(ej[37]),
            "error_jump_37_m": float(dj[37]),
            "raw_xyz_jump_37_norm": float(rj[37]),
            "clipped_raw_xyz_jump_37_norm": float(cj[37]),
            "raw_effect_37_m": float(np.linalg.norm(raw_part)),
            "stats_effect_37_m": float(np.linalg.norm(stats_part)),
            "tail35_39_sq_error_fraction": float(e2[35:].sum() / e2.sum()),
            "whole_xyz_rmse_m": float(np.sqrt(np.mean(e2))),
            "diagnostic_xyz_rmse_t0_34_m": float(np.sqrt(np.mean(e2[:35]))),
            "tail_xyz_mask_fraction_active": float(np.mean(mask[35:, :3] > 0)),
            "tail_active10_mask_fraction_active": float(np.mean(mask[35:, :10] > 0)),
            "raw_xyz_clipped_scalars_total": int(np.sum(np.abs(raw) > 1.0)),
            "raw_xyz_clipped_scalars_t37_38": int(np.sum(np.abs(raw[37:39]) > 1.0)),
            "decoded_relative_jump_max_abs_mismatch": geometry_max_error,
        })
        inputs[source] = {key: arrays[key] for key in ("expert", "action_mask", "state", "body_rgb", "wrist_rgb")}
        error_sq.append(e2)
        prediction_jump.append(pj)
        expert_jump.append(ej)
        error_jump.append(dj)
        raw_jump.append(rj)
        clipped_jump.append(cj)

    e2 = np.stack(error_sq)
    pj, ej, dj, rj, cj = (np.stack(items) for items in (prediction_jump, expert_jump, error_jump, raw_jump, clipped_jump))
    return {
        "records_sha256": expected_sha,
        "source_count": len(source_rows),
        "peak_segment_counts": dict(sorted(Counter(row["predicted_peak_segment"] for row in source_rows).items())),
        "raw_peak_segment_counts": dict(sorted(Counter(row["raw_xyz_peak_segment"] for row in source_rows).items())),
        "clipped_raw_peak_segment_counts": dict(sorted(Counter(row["clipped_raw_xyz_peak_segment"] for row in source_rows).items())),
        "tail35_39_pooled_sq_error_fraction": float(e2[:, 35:].sum() / e2.sum()),
        "tail35_39_median_per_window_sq_error_fraction": float(np.median([row["tail35_39_sq_error_fraction"] for row in source_rows])),
        "median_whole_xyz_rmse_m": float(np.median([row["whole_xyz_rmse_m"] for row in source_rows])),
        "median_diagnostic_xyz_rmse_t0_34_m": float(np.median([row["diagnostic_xyz_rmse_t0_34_m"] for row in source_rows])),
        "mean_xyz_error_norm_by_t_m": np.mean(np.sqrt(e2), axis=0).tolist(),
        "sq_error_fraction_by_t": (e2.sum(axis=0) / e2.sum()).tolist(),
        "median_predicted_jump_by_segment_m": np.median(pj, axis=0).tolist(),
        "median_expert_jump_by_segment_m": np.median(ej, axis=0).tolist(),
        "median_error_jump_by_segment_m": np.median(dj, axis=0).tolist(),
        "segment37": {
            "median_predicted_jump_m": float(np.median(pj[:, 37])),
            "p95_predicted_jump_m": float(np.percentile(pj[:, 37], 95)),
            "median_expert_jump_m": float(np.median(ej[:, 37])),
            "max_expert_jump_m": float(np.max(ej[:, 37])),
            "median_error_jump_m": float(np.median(dj[:, 37])),
            "median_raw_xyz_jump_norm": float(np.median(rj[:, 37])),
            "median_clipped_raw_xyz_jump_norm": float(np.median(cj[:, 37])),
            "median_raw_effect_m": float(np.median([row["raw_effect_37_m"] for row in source_rows])),
            "median_stats_effect_m": float(np.median([row["stats_effect_37_m"] for row in source_rows])),
            "sq_error_increment_fraction": float(np.sum(dj[:, 37] ** 2) / np.sum(dj ** 2)),
        },
        "tail_xyz_mask_all_active": all(row["tail_xyz_mask_fraction_active"] == 1.0 for row in source_rows),
        "tail_active10_mask_all_active": all(row["tail_active10_mask_fraction_active"] == 1.0 for row in source_rows),
        "raw_xyz_clipped_scalars_total": sum(row["raw_xyz_clipped_scalars_total"] for row in source_rows),
        "source_rows": source_rows,
        "_inputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--r0-root", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--relative-stats", type=Path, required=True)
    args = parser.parse_args()
    assert file_sha(args.report) == REPORT_SHA
    sealed = json.loads(args.report.read_text())
    assert sealed["errors"] == [] and sealed["selected_checkpoint"] == "NONE"
    ids = {row["source_episode_id"] for row in sealed["R0_R2_same_source"]["paired"]}
    assert len(ids) == 16
    stats = json.loads(args.relative_stats.read_text())["eef_9d"]
    lo, hi = np.asarray(stats["min"], dtype=np.float64)[:, :3], np.asarray(stats["max"], dtype=np.float64)[:, :3]
    assert lo.shape == hi.shape == (40, 3)
    r0 = profile(args.r0_root, ids, sealed["R0_final10000"]["records_sha256"], lo, hi)
    r2 = profile(args.r2_root, ids, sealed["profiles"][-1]["records_sha256"], lo, hi)
    for source in ids:
        assert all(np.array_equal(r0["_inputs"][source][key], r2["_inputs"][source][key]) for key in r0["_inputs"][source])
    del r0["_inputs"], r2["_inputs"]
    print(json.dumps({
        "schema": "PI0244_R0_R2_TEMPORAL_NPZ_READONLY_V1",
        "source_report_sha256": REPORT_SHA,
        "relative_stats_sha256": file_sha(args.relative_stats),
        "read_scope": "sealed R0/R2 final10000 TRAIN only, original fixed16",
        "new_model_queries": 0, "weight_loads": 0, "training_steps": 0, "wbc_steps": 0,
        "R0": r0, "R2": r2,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
