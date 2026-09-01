#!/usr/bin/env python3
"""Analyze node-local Stage16 native rollout artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from go2arx5_vla.benchmark.progressive_relocation import (
    _reach_ratio,
    load_npz,
    load_reach_table,
)
from go2arx5_vla.metrics.internal_plan import analyze_internal_plan
from go2arx5_vla.metrics.whole_body import (
    classify_progressive_behavior,
    compute_whole_body_metrics,
)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _boundary_times(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    result = {}
    for row in payload.get("selected", []):
        fixed = row.get("fixed_base", {})
        value = fixed.get("first_fixed_base_unreachable_time_s")
        if value is None:
            value = row.get("challenge_start_time_s")
        if value is not None:
            result[str(row["trajectory_id"])] = float(value)
    return result


def _reachability_burden(
    desired_pos: np.ndarray,
    *,
    trace: Mapping[str, Any],
    reach_table: np.ndarray,
) -> np.ndarray:
    return _reach_ratio(
        np.asarray(desired_pos, dtype=np.float64),
        np.asarray(trace["shoulder_position"], dtype=np.float64),
        np.asarray(trace["shoulder_quaternion_xyzw"], dtype=np.float64),
        reach_table,
    )


def _analyze_case(
    case: Mapping[str, Any],
    *,
    rollout_root: Path,
    trace: Mapping[str, Any],
    reach_table: np.ndarray,
    boundary_times: Mapping[str, float],
) -> Dict[str, Any]:
    trajectory_id = str(case["trajectory_id"])
    case_dir = rollout_root / trajectory_id
    summary_path = case_dir / "native_external_summary.json"
    npz_path = case_dir / "native_external_rollout.npz"
    if not summary_path.exists() or not npz_path.exists():
        return {
            "trajectory_id": trajectory_id,
            "status": "MISSING_ROLLOUT_ARTIFACT",
            "summary_path": str(summary_path),
            "npz_path": str(npz_path),
        }
    summary = json.loads(summary_path.read_text())
    data = load_npz(npz_path)
    boundary_time = boundary_times.get(trajectory_id)
    metrics = compute_whole_body_metrics(data, boundary_time_s=boundary_time)
    metrics["termination_code"] = summary.get("termination_code", summary.get("termination", "UNKNOWN"))
    metrics["status"] = summary.get("status")
    metrics["requested_steps"] = summary.get("requested_steps")
    metrics["summary_steps"] = summary.get("steps")
    burden = _reachability_burden(
        np.asarray(data["desired_pos"], dtype=np.float64),
        trace=trace,
        reach_table=reach_table,
    )
    metrics["reachability_burden_rho"] = {
        "mean": float(np.mean(burden)),
        "p95": float(np.percentile(burden, 95)),
        "max": float(np.max(burden)),
        "above_one_fraction": float(np.mean(burden > 1.0)),
    }
    metrics["internal_plan"] = analyze_internal_plan(
        data,
        burden=burden,
        boundary_time_s=boundary_time,
    )
    if boundary_time is not None:
        metrics["behavior_classification"] = classify_progressive_behavior(
            metrics,
            boundary_time_s=boundary_time,
        )
    metrics["case_metadata"] = summary.get("case_metadata", case)
    metrics["summary_path"] = str(summary_path)
    metrics["npz_path"] = str(npz_path)
    return {"trajectory_id": trajectory_id, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--reset-trace", type=Path, required=True)
    parser.add_argument("--reach-table", type=Path, required=True)
    parser.add_argument("--boundary-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    trace = load_npz(args.reset_trace)
    reach_table = load_reach_table(args.reach_table)
    boundaries = _boundary_times(args.boundary_json)
    analyzed = [
        _analyze_case(
            case,
            rollout_root=args.rollout_root,
            trace=trace,
            reach_table=reach_table,
            boundary_times=boundaries,
        )
        for case in cases
    ]
    statuses = Counter(
        item.get("metrics", {}).get("status", item.get("status", "UNKNOWN"))
        for item in analyzed
    )
    terminations = Counter(
        item.get("metrics", {}).get("termination_code", "MISSING")
        for item in analyzed
    )
    payload = {
        "schema": "STAGE16_ROLLOUT_METRICS_V1",
        "manifest": str(args.manifest),
        "rollout_root": str(args.rollout_root),
        "case_count": len(cases),
        "analyzed_count": sum("metrics" in item for item in analyzed),
        "status_counts": dict(sorted(statuses.items())),
        "termination_counts": dict(sorted(terminations.items())),
        "cases": analyzed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "case_count": len(cases),
        "analyzed_count": payload["analyzed_count"],
        "status_counts": payload["status_counts"],
        "termination_counts": payload["termination_counts"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
