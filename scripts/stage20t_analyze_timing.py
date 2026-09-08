#!/usr/bin/env python3
"""Analyze Stage20T native timing traces without changing rollout artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix


TIMING_THRESHOLD_FRACTION = 0.70


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _derived_row(row: dict[str, Any]) -> dict[str, Any]:
    length = float(row.get("traj_L", np.nan))
    error = float(row.get("traj_timing_err", np.nan))
    fraction = abs(error) / length if np.isfinite(length) and length > 0.0 else float("nan")
    return {
        "step": int(row.get("step", -1)),
        "sim_time": float(row.get("sim_time", np.nan)),
        "traj_s": float(row.get("traj_s", np.nan)),
        "traj_s_ref": float(row.get("traj_s_ref", np.nan)),
        "traj_s_ref_minus_traj_s": float(row.get("traj_s_ref", np.nan) - row.get("traj_s", np.nan)),
        "traj_timing_err": error,
        "traj_timing_error_fraction": fraction,
        "timing_margin_m": float(TIMING_THRESHOLD_FRACTION * length - abs(error)),
        "timing_margin_fraction": float(TIMING_THRESHOLD_FRACTION - fraction),
        "traj_d_lat": float(row.get("traj_d_lat", np.nan)),
        "desired_pos": row.get("desired_pos"),
        "desired_quat": row.get("desired_quat"),
        "achieved_pos": row.get("achieved_pos"),
        "achieved_quat": row.get("achieved_quat"),
        "eef_linear_velocity": row.get("eef_linear_velocity"),
        "eef_angular_velocity": row.get("eef_angular_velocity"),
        "base_pos": row.get("base_pos"),
        "base_lin_vel": row.get("base_lin_vel"),
        "base_feedforward": row.get("base_feedforward"),
        "delta_velocity": row.get("delta_velocity"),
        "internal_body_plan_9d": row.get("internal_body_plan_9d"),
        "termination_code": row.get("termination_code", "NONE"),
        "episode_done": bool(row.get("episode_done", False)),
        "reference_complete": bool(row.get("reference_complete", False)),
    }


def _initial_runtime_continuity(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure the first post-physics reference/achieved EEF discontinuity.

    The candidate audit compares the first reference point with the source
    observation state.  A seeded native reset can differ from that source
    state, so this second measurement is taken from the first authoritative
    rollout row and is intentionally reported separately.
    """

    if not trace:
        return {
            "position_discontinuity_m": None,
            "rotation_discontinuity_rad": None,
            "definition": "first post-physics desired-vs-achieved EEF sample",
        }
    first = trace[0]
    desired_pos = np.asarray(first.get("desired_pos"), dtype=np.float64)
    achieved_pos = np.asarray(first.get("achieved_pos"), dtype=np.float64)
    desired_quat = np.asarray(first.get("desired_quat"), dtype=np.float64)
    achieved_quat = np.asarray(first.get("achieved_quat"), dtype=np.float64)
    valid = (
        desired_pos.shape == (3,)
        and achieved_pos.shape == (3,)
        and desired_quat.shape == (4,)
        and achieved_quat.shape == (4,)
        and np.isfinite(desired_pos).all()
        and np.isfinite(achieved_pos).all()
        and np.isfinite(desired_quat).all()
        and np.isfinite(achieved_quat).all()
    )
    if not valid:
        return {
            "position_discontinuity_m": None,
            "rotation_discontinuity_rad": None,
            "definition": "first post-physics desired-vs-achieved EEF sample",
        }
    return {
        "position_discontinuity_m": float(np.linalg.norm(desired_pos - achieved_pos)),
        "rotation_discontinuity_rad": float(
            geodesic_angle(
                quat_to_matrix(desired_quat[None, :]),
                quat_to_matrix(achieved_quat[None, :]),
            )[0]
        ),
        "definition": "first post-physics desired-vs-achieved EEF sample",
        "step": int(first["step"]),
        "sim_time": float(first["sim_time"]),
    }


def analyze_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace = [_derived_row(row) for row in rows]
    if not trace:
        return {
            "row_count": 0,
            "termination_code": "NO_TRACE",
            "reference_completed": False,
            "trace": [],
        }
    terminal_index = next(
        (index for index, row in enumerate(trace) if row["episode_done"] or row["termination_code"] != "NONE"),
        len(trace) - 1,
    )
    crossing = next(
        (row for row in trace if np.isfinite(row["traj_timing_error_fraction"]) and row["traj_timing_error_fraction"] > TIMING_THRESHOLD_FRACTION),
        None,
    )
    terminal = trace[terminal_index]
    native_codes = Counter(str(row["termination_code"]) for row in trace)
    initial_runtime = _initial_runtime_continuity(trace)
    return {
        "row_count": len(trace),
        "terminal_index": terminal_index,
        "terminal_step": terminal["step"],
        "termination_code": terminal["termination_code"],
        "reference_completed": bool(any(row["reference_complete"] for row in trace)),
        "native_termination_codes_by_row": dict(sorted(native_codes.items())),
        "first_timing_threshold_crossing": crossing,
        "max_abs_timing_error_fraction": float(max(row["traj_timing_error_fraction"] for row in trace)),
        "min_timing_margin_m": float(min(row["timing_margin_m"] for row in trace)),
        "min_timing_margin_fraction": float(min(row["timing_margin_fraction"] for row in trace)),
        "max_lateral_distance_m": float(max(row["traj_d_lat"] for row in trace)),
        "initial_runtime_continuity": initial_runtime,
        "terminal_row": terminal,
        "terminal_trace_last_10": trace[max(0, terminal_index - 9) : terminal_index + 1],
        "trace_artifact_is_authoritative": True,
    }


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record["native"].get("execution_status", "UNKNOWN")) for record in records)
    termination_counts = Counter(str(record["native"].get("termination_code", "UNKNOWN")) for record in records)
    traces = [record["trace_analysis"] for record in records if record.get("trace_analysis", {}).get("row_count", 0)]
    native_metrics = [record["native"].get("summary", {}).get("metrics", {}) for record in records]
    position_means = [metric.get("position_m", {}).get("mean", np.nan) for metric in native_metrics]
    rotation_means = [metric.get("rotation_rad", {}).get("mean", np.nan) for metric in native_metrics]
    path_lengths = [record["trajectory_audit"].get("translation_path_length_m", np.nan) for record in records]
    durations = [record["trajectory_audit"].get("duration_s", np.nan) for record in records]
    max_speeds = [record["trajectory_audit"].get("linear_speed_mps", {}).get("max", np.nan) for record in records]
    initial_position_jumps = [
        record.get("trace_analysis", {})
        .get("initial_runtime_continuity", {})
        .get("position_discontinuity_m", np.nan)
        for record in records
    ]
    initial_rotation_jumps = [
        record.get("trace_analysis", {})
        .get("initial_runtime_continuity", {})
        .get("rotation_discontinuity_rad", np.nan)
        for record in records
    ]
    return {
        "run_count": len(records),
        "completion_count": int(sum(bool(record["native"].get("reference_completed")) for record in records)),
        "completion_fraction": float(np.mean([bool(record["native"].get("reference_completed")) for record in records])) if records else None,
        "execution_status_counts": dict(sorted(status_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "trajectory_descriptors": {
            "translation_path_length_m": _stats(path_lengths),
            "duration_s": _stats(durations),
            "max_linear_speed_mps": _stats(max_speeds),
        },
        "initial_runtime_continuity": {
            "position_discontinuity_m": _stats(initial_position_jumps),
            "rotation_discontinuity_rad": _stats(initial_rotation_jumps),
            "definition": "first post-physics desired-vs-achieved EEF sample",
        },
        "native_tracking_metrics": {
            "position_mean_m": _stats(position_means),
            "rotation_mean_rad": _stats(rotation_means),
        },
        "timing_trace_metrics": {
            "per_run_max_abs_timing_error_fraction": _stats(
                trace["max_abs_timing_error_fraction"] for trace in traces
            ),
            "per_run_min_timing_margin_m": _stats(trace["min_timing_margin_m"] for trace in traces),
            "per_run_min_timing_margin_fraction": _stats(
                trace["min_timing_margin_fraction"] for trace in traces
            ),
            "terminal_timing_margin_m": _stats(
                trace["terminal_row"]["timing_margin_m"] for trace in traces
            ),
            "terminal_lateral_distance_m": _stats(
                trace["terminal_row"]["traj_d_lat"] for trace in traces
            ),
        },
    }


def analyze_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in manifest.get("runs", []):
        rows_path = record.get("native", {}).get("rows_path")
        rows = _load_rows(Path(rows_path)) if rows_path else []
        enriched = dict(record)
        enriched["trace_analysis"] = analyze_trace(rows)
        records.append(enriched)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    condition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{record['condition']}:scale={float(record['scale']):.2f}"
        grouped[key].append(record)
        condition_groups[str(record["condition"])].append(record)

    stage16_records = [
        record
        for record in records
        if record["condition"] == "C_STAGE16_CONTROL" and record["native"].get("reference_completed")
    ]
    stage16_trace = [record["trace_analysis"] for record in stage16_records if record["trace_analysis"].get("row_count", 0)]
    envelope = {
        "qualified_control_run_count": len(stage16_records),
        "qualified_control_completion_fraction": float(np.mean([True] * len(stage16_records))) if stage16_records else None,
        "observed_max_abs_timing_fraction": _stats(
            trace["max_abs_timing_error_fraction"] for trace in stage16_trace
        ),
        "observed_min_timing_margin_m": _stats(trace["min_timing_margin_m"] for trace in stage16_trace),
        "observed_min_timing_margin_fraction": _stats(
            trace["min_timing_margin_fraction"] for trace in stage16_trace
        ),
        "scope": "three Stage16 control seeds in Stage20T, not a replacement for the full Stage16 28-trajectory distribution",
    }
    return {
        "schema": "STAGE20T_TIMING_ANALYSIS_V1",
        "timing_threshold_fraction": TIMING_THRESHOLD_FRACTION,
        "manifest_schema": manifest.get("schema"),
        "run_count": len(records),
        "group_summaries": {key: _group_summary(value) for key, value in sorted(grouped.items())},
        "condition_summaries": {key: _group_summary(value) for key, value in sorted(condition_groups.items())},
        "stage16_qualified_timing_envelope": envelope,
        "runs": records,
        "analysis_notes": {
            "timing_error_definition": "signed traj_timing_err = traj_s - s_ref; termination uses absolute value",
            "timing_margin_definition": "0.70 * traj_L - abs(traj_timing_err)",
            "rows_are_pre_reset_terminal_aware": True,
            "retimed_success_is_not_unchanged_model_qualification": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = analyze_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "run_count": analysis["run_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
