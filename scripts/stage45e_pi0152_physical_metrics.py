#!/usr/bin/env python3
"""Build the PI0152 physical E4 readout from sealed rows, CPU-only.

This postprocessor never imports a model and never rewrites a producer row.
It joins the current E4 row bundles with the explicitly bound historical and
recovery sources so the 18-arm/432-window physical view is independently
inspectable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import statistics
from pathlib import Path
import re
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import stage45e_h200_unified_replication as campaign


LABEL_RE = re.compile(r"(LEGACY|L1|N2)\(seed(\d+)\)@(\d+)")
SCALAR_METRICS = (
    "endpoint_error_m",
    "position_rmse_m",
    "position_max_abs_m",
    "prediction_path_length_m",
    "roughness_ratio",
    "eef_rot6d_rmse",
    "full_action_rmse_mixed_units",
    "gripper_rmse",
    "gripper_max_abs",
)
TARGET_METRICS = (
    "initial_target_distance_m",
    "final_target_distance_m",
    "minimum_target_distance_m",
    "absolute_progress_m",
    "best_progress_m",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [campaign.json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = campaign.read_json(path)
    return value if isinstance(value, list) else [value]


def _sha(path: Path) -> str:
    return campaign.sha256_file(path)


def _metric_namespace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metric = row.get("metrics")
    if isinstance(metric, Mapping):
        return metric
    physical = row.get("physical")
    if isinstance(physical, Mapping):
        endpoint = physical.get("free_running_endpoint")
        if isinstance(endpoint, Mapping) and isinstance(endpoint.get("trajectory_metrics"), Mapping):
            return endpoint["trajectory_metrics"]
    raise RuntimeError("sealed row has no physical metric namespace")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"non-finite physical value {label}: {value!r}")
    return float(value)


def _physical_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metric = _metric_namespace(row)
    target = metric.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeError("sealed row has no target physical metrics")
    result: dict[str, Any] = {key: _finite_number(metric.get(key), key) for key in SCALAR_METRICS}
    result["target"] = {key: _finite_number(target.get(key), key) for key in TARGET_METRICS}
    result["progress_sign"] = target.get("progress_sign")
    if not isinstance(result["progress_sign"], str):
        raise RuntimeError("sealed row has no progress sign")
    displacement = target.get("target_displacement_m")
    if not isinstance(displacement, list) or len(displacement) != 3:
        raise RuntimeError("sealed row target displacement is not a 3-vector")
    result["target_displacement_m"] = [_finite_number(value, "target_displacement_m") for value in displacement]
    return result


def _canonical_row_sha(row: Mapping[str, Any]) -> str:
    return campaign.sha256_bytes(campaign._canonical_json_bytes(row))


def _summary(rows: list[Mapping[str, Any]], getter) -> dict[str, Any]:
    values = [float(getter(row)) for row in rows]
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def build_report(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root).resolve()
    name = campaign_root.name
    diagnostics = campaign_root.parent.parent / "diagnostics"
    windows = list(campaign.HISTORICAL_DIAGNOSTIC_WINDOWS)
    arms: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, Any]] = {}

    def add_source(path: Path, role: str) -> None:
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"physical report source unavailable: {path}")
        key = str(path)
        source_files.setdefault(key, {"path": key, "bytes": path.stat().st_size, "sha256": _sha(path), "roles": []})
        if role not in source_files[key]["roles"]:
            source_files[key]["roles"].append(role)

    def add_arm(label: str, route: str, seed: int, step: int, source_kind: str, rows_path: Path, raw_path: Path, source_phase: str, declared: Mapping[str, Any] | None = None) -> None:
        rows = _read_rows(rows_path)
        if source_kind == "HISTORICAL_LEGACY_D2B_E4":
            rows = [
                row for row in rows
                if row.get("identity", {}).get("solver") == "E4"
                and row.get("identity", {}).get("checkpoint") == f"FULL24@{step}"
                and row.get("identity", {}).get("final_stage") is True
            ]
        if len(rows) != 24 or label in arms:
            raise RuntimeError(f"physical arm cardinality drift: {label} ({len(rows)})")
        per_window: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            identity = row.get("identity", {})
            window_id = row.get("window_id") or identity.get("window_id")
            if window_id != windows[index]:
                raise RuntimeError(f"window order drift for {label} at {index}: {window_id}")
            if row.get("target_xyz_in_model_input", identity.get("target_xyz_in_model_input")) is not False:
                raise RuntimeError(f"target exclusion drift for {label}")
            per_window.append({
                "arm": label,
                "route": route,
                "seed": seed,
                "checkpoint_step": step,
                "window_index": index,
                "window_id": window_id,
                "target_id": row.get("target_id") or identity.get("target_id") or window_id.split(":", 1)[0],
                "source_kind": source_kind,
                "source_row_sha256": _canonical_row_sha(row),
                "stats_route": row.get("stats_route") or identity.get("stats_route"),
                "checkpoint_tree_sha256": (row.get("checkpoint") or {}).get("sha256") or identity.get("checkpoint_tree_sha256"),
                "physical_metrics_namespace": "DECODED_PHYSICAL_ACTION",
                "physical_metrics": _physical_metrics(row),
            })
        add_source(rows_path, "physical_row_source")
        add_source(raw_path, "raw_evidence")
        arms[label] = {
            "label": label,
            "route": route,
            "seed": seed,
            "checkpoint_step": step,
            "source_kind": source_kind,
            "row_count": 24,
            "rows_path": str(rows_path.resolve()),
            "rows_sha256": _sha(rows_path),
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": _sha(raw_path),
            "stats_route": per_window[0]["stats_route"],
            "checkpoint_tree_sha256": per_window[0]["checkpoint_tree_sha256"],
            "per_window": per_window,
            "declared_source": dict(declared or {}),
            "source_phase": source_phase,
        }
        for key in SCALAR_METRICS:
            arms[label].setdefault("metrics", {})[key] = _summary(per_window, lambda row, key=key: row["physical_metrics"][key])
        arms[label]["progress"] = {}
        for key in TARGET_METRICS:
            arms[label]["progress"][key] = _summary(per_window, lambda row, key=key: row["physical_metrics"]["target"][key])
        arms[label]["progress"]["sign_counts"] = dict(sorted(Counter(row["physical_metrics"]["progress_sign"] for row in per_window).items()))

    for diagnostic_root in sorted(diagnostics.glob(name + "_E4_*")):
        for rows_path in sorted(diagnostic_root.glob("rows_*.json")):
            rows = _read_rows(rows_path)
            label = str(rows[0]["checkpoint"]["label"])
            match = LABEL_RE.fullmatch(label)
            if not match:
                raise RuntimeError(f"unexpected current E4 label: {label}")
            route, seed, step = match.group(1).lower(), int(match.group(2)), int(match.group(3))
            add_arm(label, route, seed, step, "CURRENT_H200_SEALED_E4", rows_path, diagnostic_root / "raw_predictions.jsonl", "E4_CURRENT", {"unified_seal": str(diagnostic_root / "unified_phase_seal.json"), "unified_seal_sha256": _sha(diagnostic_root / "unified_phase_seal.json")})

    legacy_root = diagnostics / (name + "_E4_legacy")
    legacy_seal = campaign.read_json(legacy_root / "unified_phase_seal.json")
    for recovery_root in legacy_seal.get("recovery_source_roots", []):
        recovery = Path(recovery_root)
        for rows_path in sorted(recovery.glob("rows_*.json")):
            rows = _read_rows(rows_path)
            label = str(rows[0]["checkpoint"]["label"])
            if label in arms:
                continue
            match = LABEL_RE.fullmatch(label)
            if not match:
                raise RuntimeError(f"unexpected recovery E4 label: {label}")
            route, seed, step = match.group(1).lower(), int(match.group(2)), int(match.group(3))
            add_arm(label, route, seed, step, "CURRENT_H200_RECOVERY_SOURCE", rows_path, recovery / "raw_predictions.jsonl", "E4_RECOVERY", {"recovery_source_root": str(recovery), "recovery_source_seal_sha256": _sha(recovery / "FAILURE_SEAL.json")})

    catalog = campaign.read_json(campaign_root / "historical_diagnostic_evidence_audit.json")
    for step, reference in catalog["references"]["l1"]["E4"].items():
        step = int(step)
        add_arm(f"L1(seed2222)@{step}", "l1", 2222, step, "HISTORICAL_L1_E4_READONLY", Path(reference["rows_path"]), Path(reference["raw_prediction_path"]), "HISTORICAL_E4", reference)
    for step, reference in catalog["references"]["legacy"]["E4"].items():
        step = int(step)
        rows_path = Path(reference["rows_path"])
        add_arm(f"LEGACY(seed2222)@{step}", "legacy", 2222, step, "HISTORICAL_LEGACY_D2B_E4_READONLY", rows_path, rows_path, "HISTORICAL_E4", reference)

    expected = {f"{route}(seed{seed})@{step}" for route in ("LEGACY", "L1", "N2") for seed in (2222, 2223, 2224) for step in (1000, 2000)}
    if set(arms) != expected:
        raise RuntimeError(f"cohort drift: missing={sorted(expected - set(arms))}, extra={sorted(set(arms) - expected)}")
    per_window = [row for label in sorted(arms) for row in arms[label]["per_window"]]
    if len(per_window) != 432:
        raise RuntimeError(f"physical row count drift: {len(per_window)}")
    overall = {key: _summary(per_window, lambda row, key=key: row["physical_metrics"][key]) for key in SCALAR_METRICS}
    for key in ("absolute_progress_m", "final_target_distance_m"):
        overall[key] = _summary(per_window, lambda row, key=key: row["physical_metrics"]["target"][key])
    overall["sign_counts"] = dict(sorted(Counter(row["physical_metrics"]["progress_sign"] for row in per_window).items()))
    by_route = {}
    for route in ("legacy", "l1", "n2"):
        route_rows = [row for row in per_window if row["route"] == route]
        by_route[route] = {key: _summary(route_rows, lambda row, key=key: row["physical_metrics"][key]) for key in ("endpoint_error_m", "position_rmse_m", "prediction_path_length_m", "roughness_ratio")}
        by_route[route]["absolute_progress_m"] = _summary(route_rows, lambda row: row["physical_metrics"]["target"]["absolute_progress_m"])
    return {
        "schema": "VANDA_STAGE45E_PI0152_PHYSICAL_E4_METRICS_REPORT_V1",
        "status": "PASS_READONLY_DERIVED_FROM_SEALED_E4_ROWS",
        "pi_gate": campaign.PI0152_GATE,
        "issue_comment": campaign.PI0152_ISSUE_COMMENT,
        "pr_comment": campaign.PI0152_PR_COMMENT,
        "campaign_root": str(campaign_root),
        "execution_head": "eab15ec185203d69c03ab82b7843bcaea9200281",
        "post_execution_provenance_code_head": campaign.git(REPO, "rev-parse", "HEAD"),
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "raw_before_derived": True,
        "source_rows_untouched": True,
        "cohort": {
            "arm_count": 18,
            "physical_row_count": 432,
            "current_phase_rows": 336,
            "current_recovery_source_rows": 24,
            "historical_readonly_rows": 96,
            "missing": [],
            "labels": sorted(arms),
        },
        "actual_source_files": sorted(source_files.values(), key=lambda item: item["path"]),
        "arm_summaries": [arms[label] for label in sorted(arms)],
        "overall_summary": overall,
        "by_route_summary": by_route,
        "per_window_rows": per_window,
        "notes": [
            "CPU-only descriptive read of sealed E4 rows; no model or training execution.",
            "The canonical phase-reference index remains separate and is not treated as physical metrics.",
            "Historical Legacy E4 physical rows are final-stage E4 rows selected from the preserved D2B source; the full source binding remains in the historical catalog.",
            "No winner selection, Validation-E/Holdout-F, WBC, simulator, hardware, or new execution occurred during report generation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.campaign_root)
    path = Path(args.campaign_root).resolve() / "physical_metrics_report.json"
    campaign.write_json(path, report)
    print(json.dumps({"path": str(path), "sha256": campaign.sha256_file(path), "arm_count": report["cohort"]["arm_count"], "physical_row_count": report["cohort"]["physical_row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
