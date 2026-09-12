#!/usr/bin/env python3
"""CPU-only closure for the PI0131 matched Legacy/N2 cohort.

This script consumes only sealed raw arrays, checkpoint/evaluation sidecars,
and accepted Stage40/Stage45C metric functions.  It never imports Torch or
GR00T, constructs a model, or reruns inference.  The N2 G2 per-window rows
are reconstructed from their already-sealed ``.npy`` files and must reproduce
the original G2 aggregates before any paired summary is emitted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
WINDOW_IDS = tuple(f"stage22_expert_{target}:expert-{index:04d}" for target in ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d") for index in range(6))
LEGACY_BASELINE_LABELS = ("FULL24@1000", "FULL24@2000")
N2_LABELS = ("R1@1000", "R1@2000", "R2@1000", "R2@2000")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    mode = Path(path).lstat().st_mode
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(), f"JSON is not a regular file: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite CPU evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np
    arr = np.asarray(list(values), dtype=np.float64)
    require(arr.size > 0 and np.isfinite(arr).all(), "numeric summary received empty/nonfinite values")
    return {"count": int(arr.size), "min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95.0))}


def static_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    forbidden = sorted(set(imports) & {"torch", "gr00t", "transformers", "isaacgym", "omni"})
    return {"schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_CPU_STATIC_AUDIT_V1", "status": "PASS" if not forbidden else "BLOCKED", "top_level_forbidden_imports": forbidden, "model_construction": False, "model_forward": False, "training": False, "gpu": False, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}


def load_context(args: argparse.Namespace) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any], Any, Any]:
    import stage45e_n2_train24_eval as ev
    sealed_root = Path(args.sealed_baseline).resolve()
    snapshots_path = Path(args.snapshot_preflight).resolve()
    sealed = ev._verify_sealed_baseline(sealed_root, snapshots_path)
    records = ev._snapshot_lookup(read_json(snapshots_path)["snapshots"])
    require([item["window_id"] for item in records] == list(WINDOW_IDS), "snapshot order drift")
    runtime = read_json(Path(args.runtime_stats).resolve())
    n2_doc = read_json(Path(args.n2_stats).resolve())
    require(sha256_file(Path(args.runtime_stats)) == RUNTIME_STATS_SHA, "runtime stats SHA drift")
    require(sha256_file(Path(args.n2_stats)) == N2_STATS_SHA and n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats identity drift")
    return ev, sealed, records, runtime, n2_doc, sealed["noise_manifest"]


def reconstruct_legacy_baseline(ev: Any, sealed: Mapping[str, Any], runtime: Mapping[str, Any], output: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows, reproduction = ev._read_legacy_rows(sealed, runtime, output)
    require(set(rows) == set(LEGACY_BASELINE_LABELS), "legacy baseline row labels drift")
    return rows, reproduction


def reconstruct_n2_g2(ev: Any, sealed: Mapping[str, Any], records: Sequence[Mapping[str, Any]], n2_doc: Mapping[str, Any], args: argparse.Namespace, output: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows
    g2_root = Path(args.n2_g2_root).resolve()
    seal = read_json(g2_root / "g2_seal.json")
    require(seal.get("status") == "PASS" and seal.get("counters", {}).get("conditioning_passes") == 96 and seal.get("counters", {}).get("action_head_calls") == 384, "sealed N2 G2 budget/status drift")
    lines = [json.loads(line) for line in (g2_root / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == 96, f"sealed N2 G2 raw row count drift: {len(lines)}")
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in N2_LABELS}
    checkpoint_meta: dict[str, Mapping[str, Any]] = {}
    for line in lines:
        checkpoint = line.get("checkpoint")
        require(isinstance(checkpoint, Mapping), "G2 checkpoint metadata missing")
        label = str(checkpoint.get("label"))
        require(label in by_label, f"unexpected G2 label: {label}")
        checkpoint_meta.setdefault(label, checkpoint)
        index = len(by_label[label])
        require(str(line.get("window_id")) == str(records[index]["window_id"]), f"G2 window order drift {label}/{index}")
        source = g2_root / str(line["path"])
        require(source.is_file() and not source.is_symlink(), f"G2 raw array unavailable: {source}")
        require(sha256_file(source) == line.get("file_sha256"), f"G2 raw SHA drift: {source}")
        raw = np.load(source, allow_pickle=False)
        require(raw.shape == (1, 40, 132) and np.isfinite(raw).all(), f"G2 raw shape/finite drift: {source}")
        record = records[index]
        physical, details = ev._decode_with_stats(raw, record["state_v1"], n2_doc, stats_sha256=N2_STATS_SHA, payload_sha256=N2_STATS_PAYLOAD_SHA, chunk_id=str(record["window_id"]), timestamps=record["source_timestamps"], observation_timestamp=float(record["observation_timestamp"]))
        normalized = ev._normalized_metrics(raw, record["expert_physical_action"], record["state_v1"], record["source_timestamps"], n2_doc)
        row = ev._evaluation_row(raw, physical, details, normalized, record, {"label": label, "tree_sha256": checkpoint["tree_sha256"]}, Path(line["path"]), line["file_sha256"], GROOT_COMMIT)
        row.update({"schema": "CORELAB_STAGE45E_N2_G2_RECONSTRUCTED_ROW_V1", "stats_route": "N2_BALANCED_XYZ", "seed": int(checkpoint["seed"]), "sampler": {"steps_observed": int(line["sampler_calls_observed"]), "timestep_ids": [0, 250, 500, 750], "dt": 0.25, "official_public_sampler_called": False}, "finite": True})
        by_label[label].append(row)
    for label, rows in by_label.items():
        require(len(rows) == 24 and [row["window_id"] for row in rows] == list(WINDOW_IDS), f"G2 reconstructed row order/count drift {label}")
    reproduction: dict[str, Any] = {}
    for label, rows in by_label.items():
        aggregate = aggregate_evaluation_rows(rows)
        expected = read_json(g2_root / f"aggregate_{label.replace('@', '_at_')}.json")
        mismatches = ev._compare_json(aggregate, expected, tolerance=1.0e-12)
        require(not mismatches, f"sealed G2 aggregate reproduction failed for {label}: {mismatches[:8]}")
        write_json(output / f"reconstructed_{label.replace('@', '_at_')}_rows.json", rows)
        write_json(output / f"reconstructed_{label.replace('@', '_at_')}_aggregate.json", aggregate)
        reproduction[label] = {"status": "PASS", "rows": 24, "computed_aggregate_sha256": sha256_file(output / f"reconstructed_{label.replace('@', '_at_')}_aggregate.json"), "sealed_aggregate_sha256": sha256_file(g2_root / f"aggregate_{label.replace('@', '_at_')}.json"), "mismatch_count": 0, "checkpoint_tree_sha256": checkpoint_meta[label]["tree_sha256"]}
    return by_label, reproduction


def load_new_legacy_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("rows_*.json")):
        name = path.name.removeprefix("rows_").removesuffix(".json").replace("_at_", "@")
        value = read_json(path)
        require(name.startswith("LEGACY(seed") and len(value) == 24, f"new legacy E4 row file invalid: {path}")
        require([item["window_id"] for item in value] == list(WINDOW_IDS), f"new legacy E4 order drift: {name}")
        rows[name] = value
    require(set(rows) == {"LEGACY(seed2223)@1000", "LEGACY(seed2223)@2000", "LEGACY(seed2224)@1000", "LEGACY(seed2224)@2000"}, f"new legacy E4 arms missing: {sorted(rows)}")
    return rows


def load_existing_n2_seed2222(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for step in (1000, 2000):
        value = read_json(root / f"N2_at_{step}_rows.json")
        require(len(value) == 24 and [item["window_id"] for item in value] == list(WINDOW_IDS), f"existing N2 seed2222 row drift @ {step}")
        rows[f"N2(seed2222)@{step}"] = value
    return rows


def rename_baseline(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {f"LEGACY(seed2222)@{step}": [dict(row) for row in rows[f"FULL24@{step}"]] for step in (1000, 2000)}


def rename_n2(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    mapping = {"R1@1000": "N2(seed2223)@1000", "R1@2000": "N2(seed2223)@2000", "R2@1000": "N2(seed2224)@1000", "R2@2000": "N2(seed2224)@2000"}
    return {mapping[label]: [dict(row) for row in values] for label, values in rows.items()}


def pair(rows: Mapping[str, Sequence[Mapping[str, Any]]], left_label: str, right_label: str, name: str) -> dict[str, Any]:
    left, right = rows[left_label], rows[right_label]
    require(len(left) == len(right) == 24, f"pair count drift {name}")
    require([item["window_id"] for item in left] == [item["window_id"] for item in right], f"pair order drift {name}")
    physical_fields = ("position_rmse_m", "position_max_abs_m", "endpoint_error_m", "prediction_path_length_m", "eef_rot6d_rmse", "gripper_rmse", "full_action_rmse_mixed_units")
    target_fields = ("initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m", "endpoint_target_vector_cosine")
    detail: list[dict[str, Any]] = []
    for lrow, rrow in zip(left, right):
        lm, rm = lrow["metrics"], rrow["metrics"]
        lt, rt = lm["target"], rm["target"]
        detail.append({"window_id": lrow["window_id"], "target_id": lrow["target_id"], "physical_delta_right_minus_left": {field: float(rm[field] - lm[field]) for field in physical_fields}, "target_delta_right_minus_left": {field: None if rt.get(field) is None or lt.get(field) is None else float(rt[field] - lt[field]) for field in target_fields}, "progress_sign_left": lt.get("progress_sign"), "progress_sign_right": rt.get("progress_sign"), "normalized_active_rmse_left": float(lrow["normalized_metrics"]["active_rmse"]), "normalized_active_rmse_right": float(rrow["normalized_metrics"]["active_rmse"]), "normalized_active_rmse_delta_right_minus_left": float(rrow["normalized_metrics"]["active_rmse"] - lrow["normalized_metrics"]["active_rmse"])})
    summary = {"name": name, "left": left_label, "right": right_label, "window_count": 24, "physical_metric_delta": {field: numeric_summary([row["physical_delta_right_minus_left"][field] for row in detail]) for field in physical_fields}, "target_metric_delta": {field: numeric_summary([row["target_delta_right_minus_left"][field] for row in detail if row["target_delta_right_minus_left"][field] is not None]) for field in target_fields}, "normalized_active_rmse_delta": numeric_summary([row["normalized_active_rmse_delta_right_minus_left"] for row in detail]), "progress_sign_transition_counts": {f"{a}->{b}": sum(row["progress_sign_left"] == a and row["progress_sign_right"] == b for row in detail) for a in ("POSITIVE", "ZERO", "NEGATIVE") for b in ("POSITIVE", "ZERO", "NEGATIVE")}}
    return {"summary": summary, "rows": detail}


def per_target(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_PER_TARGET_V1", "targets": {}}
    for target in ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d"):
        result["targets"][target] = {}
        for label, values in rows.items():
            selected = [row for row in values if row["target_id"] == target]
            require(len(selected) == 6, f"per-target count drift {label}/{target}")
            result["targets"][target][label] = {"window_ids": [row["window_id"] for row in selected], "physical": {field: numeric_summary([float(row["metrics"][field]) for row in selected]) for field in ("position_rmse_m", "endpoint_error_m", "full_action_rmse_mixed_units")}, "target": {field: numeric_summary([float(row["metrics"]["target"][field]) for row in selected]) for field in ("initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m")}, "normalized": {field: numeric_summary([float(row["normalized_metrics"][field]) for row in selected]) for field in ("active_rmse", "active_mae", "active_max_abs", "raw_active_outside_neg1_pos1_fraction")}, "progress_sign_counts": {sign: sum(row["metrics"]["target"]["progress_sign"] == sign for row in selected) for sign in ("POSITIVE", "ZERO", "NEGATIVE")}}
    return result


def failure_taxonomy(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_FAILURE_TAXONOMY_V1", "checkpoints": {}}
    for label, values in rows.items():
        result["checkpoints"][label] = {"row_count": len(values), "nonfinite": sum(not row.get("finite", False) for row in values), "sampler_step_mismatch": sum(row.get("sampler", {}).get("steps_observed") != 4 for row in values), "target_progress_positive": sum(row["metrics"]["target"]["progress_sign"] == "POSITIVE" for row in values), "target_progress_zero": sum(row["metrics"]["target"]["progress_sign"] == "ZERO" for row in values), "target_progress_negative": sum(row["metrics"]["target"]["progress_sign"] == "NEGATIVE" for row in values), "normalized_clipping_rows": sum(row["normalized_metrics"]["raw_active_outside_neg1_pos1_count"] > 0 for row in values)}
    result["policy"] = "Physical decoded metrics are primary across normalization routes; normalized metrics are route-specific descriptive only; no iid significance or automatic winner."
    return result


def cross_seed_summary(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Descriptive across-seed summaries; no inferential test or winner."""
    result: dict[str, Any] = {"schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_CROSS_SEED_V1", "groups": {}}
    for route in ("LEGACY", "N2"):
        for step in (1000, 2000):
            labels = [f"{route}(seed{seed})@{step}" for seed in (2222, 2223, 2224)]
            values = [row for label in labels for row in rows[label]]
            result["groups"][f"{route}@{step}"] = {
                "seed_count": 3,
                "window_count": len(values),
                "physical": {field: numeric_summary([float(row["metrics"][field]) for row in values]) for field in ("position_rmse_m", "endpoint_error_m", "full_action_rmse_mixed_units")},
                "target": {field: numeric_summary([float(row["metrics"]["target"][field]) for row in values]) for field in ("initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m")},
                "progress_sign_counts": {sign: sum(row["metrics"]["target"]["progress_sign"] == sign for row in values) for sign in ("POSITIVE", "ZERO", "NEGATIVE")},
            }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.mode == "cpu", "CPU summary has no runtime mode")
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"CPU output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    ev, sealed, records, runtime, n2_doc, _noise = load_context(args)
    legacy_baseline_rows, baseline_reproduction = reconstruct_legacy_baseline(ev, sealed, runtime, output)
    n2_rows, n2_reproduction = reconstruct_n2_g2(ev, sealed, records, n2_doc, args, output)
    new_legacy = load_new_legacy_rows(Path(args.new_legacy_e4_root).resolve())
    existing_n2 = load_existing_n2_seed2222(Path(args.existing_n2_eval_root).resolve())
    rows: dict[str, list[dict[str, Any]]] = {}
    rows.update(rename_baseline(legacy_baseline_rows))
    rows.update(new_legacy)
    rows.update(load_existing_n2_seed2222(Path(args.existing_n2_eval_root).resolve()))
    rows.update(rename_n2(n2_rows))
    require(set(rows) == {f"LEGACY(seed{seed})@{step}" for seed in (2222, 2223, 2224) for step in (1000, 2000)} | {f"N2(seed{seed})@{step}" for seed in (2222, 2223, 2224) for step in (1000, 2000)}, "matched cohort arm set drift")
    pairs: dict[str, Any] = {}
    for seed in (2222, 2223, 2224):
        for step in (1000, 2000):
            pairs[f"legacy_to_n2_seed{seed}_at_{step}"] = pair(rows, f"LEGACY(seed{seed})@{step}", f"N2(seed{seed})@{step}", f"legacy_to_n2_seed{seed}_at_{step}")
        pairs[f"legacy_exposure_seed{seed}"] = pair(rows, f"LEGACY(seed{seed})@1000", f"LEGACY(seed{seed})@2000", f"legacy_exposure_seed{seed}")
        pairs[f"n2_exposure_seed{seed}"] = pair(rows, f"N2(seed{seed})@1000", f"N2(seed{seed})@2000", f"n2_exposure_seed{seed}")
    write_json(output / "matched_paired_comparisons.json", pairs)
    target = per_target(rows); write_json(output / "matched_per_target_summary.json", target)
    taxonomy = failure_taxonomy(rows); write_json(output / "matched_failure_taxonomy.json", taxonomy)
    cross_seed = cross_seed_summary(rows); write_json(output / "matched_cross_seed_summary.json", cross_seed)
    aggregate_hashes = {label: sha256_file(output / f"reconstructed_{label.replace('@', '_at_')}_aggregate.json") for label in n2_rows}
    result = {"schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_CPU_CLOSURE_V1", "status": "PASS", "pi_gate": "PI_GATE_STATE_0131", "baseline_reproduction": baseline_reproduction, "n2_g2_reproduction": n2_reproduction, "matched_arms": sorted(rows), "paired_comparisons_sha256": sha256_file(output / "matched_paired_comparisons.json"), "per_target_sha256": sha256_file(output / "matched_per_target_summary.json"), "cross_seed_sha256": sha256_file(output / "matched_cross_seed_summary.json"), "failure_taxonomy_sha256": sha256_file(output / "matched_failure_taxonomy.json"), "n2_reconstructed_aggregate_sha256": aggregate_hashes, "route_policy": {"cross_route_primary": "decoded physical metrics", "normalized": "route-specific descriptive only", "automatic_winner": False, "iid_significance": False}, "execution_boundary": {"gpu": False, "model_construction": 0, "model_forward": 0, "training": 0, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}, "timestamp_utc": now_utc(), "return_token": "STAGE45E_LEGACY_MATCHED_SEED_REPLICATION_COMPLETE_FOR_PI_REVIEW"}
    write_json(output / "cpu_closure_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "cpu"), default="static")
    parser.add_argument("--runtime-stats", type=Path)
    parser.add_argument("--n2-stats", type=Path)
    parser.add_argument("--sealed-baseline", type=Path)
    parser.add_argument("--snapshot-preflight", type=Path)
    parser.add_argument("--n2-g2-root", type=Path)
    parser.add_argument("--new-legacy-e4-root", type=Path)
    parser.add_argument("--existing-n2-eval-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.mode == "static":
        print(json.dumps(static_audit(), indent=2, sort_keys=True)); return 0
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True)); return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
