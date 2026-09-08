#!/usr/bin/env python3
"""Apply the predeclared Stage22 simulation profile selection/qualification gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "median_path_length_ratio_min": 0.75,
    "median_path_length_ratio_max": 1.25,
    "median_mean_speed_ratio_min": 0.75,
    "median_mean_speed_ratio_max": 1.25,
    "median_max_speed_ratio_min": 0.75,
    "median_max_speed_ratio_max": 1.25,
    "median_position_mean_error_m_max": 0.02,
    "median_endpoint_position_error_m_max": 0.03,
    "common_future_overlap_p95_m_max": 0.025,
    "common_future_overlap_max_m_max": 0.05,
}


def _median(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    return None if median is None else float(median)


def _max(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return None
    maximum = value.get("max")
    return None if maximum is None else float(maximum)


def qualification_checks(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    decisions = summary.get("safety_gate_decisions", {})
    values = {
        "median_path_length_ratio": _median(summary, "path_length_ratio"),
        "median_mean_speed_ratio": _median(summary, "mean_speed_ratio"),
        "median_max_speed_ratio": _median(summary, "max_speed_ratio"),
        "median_position_mean_error_m": _median(summary, "position_mean_error_m"),
        "median_endpoint_position_error_m": _median(summary, "endpoint_position_error_m"),
        "common_future_overlap_p95_m": _max(summary, "overlap_position_error_p95_m"),
        "common_future_overlap_max_m": _max(summary, "overlap_position_error_max_m"),
    }
    checks = {
        "all_outputs_finite": all(
            bool(record.get("finite", True)) for record in payload.get("records", [])
        ),
        "safety_gate_reject_count_zero": int(decisions.get("REJECT", 0)) == 0,
        "median_path_length_ratio_in_range": values["median_path_length_ratio"] is not None
        and THRESHOLDS["median_path_length_ratio_min"]
        <= values["median_path_length_ratio"]
        <= THRESHOLDS["median_path_length_ratio_max"],
        "median_mean_speed_ratio_in_range": values["median_mean_speed_ratio"] is not None
        and THRESHOLDS["median_mean_speed_ratio_min"]
        <= values["median_mean_speed_ratio"]
        <= THRESHOLDS["median_mean_speed_ratio_max"],
        "median_max_speed_ratio_in_range": values["median_max_speed_ratio"] is not None
        and THRESHOLDS["median_max_speed_ratio_min"]
        <= values["median_max_speed_ratio"]
        <= THRESHOLDS["median_max_speed_ratio_max"],
        "median_position_mean_error_within_limit": values["median_position_mean_error_m"] is not None
        and values["median_position_mean_error_m"] <= THRESHOLDS["median_position_mean_error_m_max"],
        "median_endpoint_error_within_limit": values["median_endpoint_position_error_m"] is not None
        and values["median_endpoint_position_error_m"] <= THRESHOLDS["median_endpoint_position_error_m_max"],
        "common_future_overlap_p95_within_limit": values["common_future_overlap_p95_m"] is not None
        and values["common_future_overlap_p95_m"] <= THRESHOLDS["common_future_overlap_p95_m_max"],
        "common_future_overlap_max_within_limit": values["common_future_overlap_max_m"] is not None
        and values["common_future_overlap_max_m"] <= THRESHOLDS["common_future_overlap_max_m_max"],
    }
    return {
        "passed": all(checks.values()),
        "thresholds": THRESHOLDS,
        "values": values,
        "checks": checks,
        "interpretation": "simulation qualification threshold only; not a real-robot safety limit",
    }


def _metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    summary = payload.get("summary", {})
    path = _median(summary, "path_length_ratio")
    mean_speed = _median(summary, "mean_speed_ratio")
    max_speed = _median(summary, "max_speed_ratio")
    return {
        "position_mean_error_m": _median(summary, "position_mean_error_m"),
        "endpoint_position_error_m": _median(summary, "endpoint_position_error_m"),
        "path_length_ratio_abs_deviation": None if path is None else abs(path - 1.0),
        "mean_speed_ratio_abs_deviation": None if mean_speed is None else abs(mean_speed - 1.0),
        "max_speed_ratio_abs_deviation": None if max_speed is None else abs(max_speed - 1.0),
    }


def select_recipe(r0: dict[str, Any], r1: dict[str, Any]) -> dict[str, Any]:
    r0_values = _metric_values(r0)
    r1_values = _metric_values(r1)
    wins: dict[str, str] = {}
    for key in r0_values:
        left, right = r0_values[key], r1_values[key]
        if left is None or right is None:
            wins[key] = "UNAVAILABLE"
        elif right < left:
            wins[key] = "R1"
        elif left < right:
            wins[key] = "R0"
        else:
            wins[key] = "TIE"
    r1_wins = sum(value == "R1" for value in wins.values())
    r0_score = sum(value for value in r0_values.values() if value is not None)
    r1_score = sum(value for value in r1_values.values() if value is not None)
    r1_checks = qualification_checks(r1)
    r0_checks = qualification_checks(r0)
    if r1_wins >= 4 and r1_checks["passed"]:
        selected = "R1"
        reason = "R1 wins at least 4 of 5 primary profile quantities and passes absolute validation gate"
    elif r0_checks["passed"] and r0_score <= r1_score:
        selected = "R0"
        reason = "R0 passes absolute validation gate and is no worse by the frozen profile score"
    else:
        selected = "R1" if r1_score < r0_score else "R0"
        reason = "neither recipe satisfies the complete gate; selected lower frozen profile score for reporting only"
    return {
        "selected_recipe": selected,
        "selection_reason": reason,
        "primary_profile_winner_by_quantity": wins,
        "r1_primary_profile_wins": r1_wins,
        "r0_profile_score": r0_score,
        "r1_profile_score": r1_score,
        "r0_qualification": r0_checks,
        "r1_qualification": r1_checks,
        "selection_policy": "validation_only; final_test_f_is_not_read_during_selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "final"), required=True)
    parser.add_argument("--r0", type=Path)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--selected-recipe", choices=("R0", "R1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "select":
        if args.r0 is None or args.r1 is None:
            parser.error("select mode requires --r0 and --r1")
        r0 = json.loads(args.r0.read_text(encoding="utf-8"))
        r1 = json.loads(args.r1.read_text(encoding="utf-8"))
        if r0.get("split") != "validation" or r1.get("split") != "validation":
            parser.error("selection may only read the locked validation split")
        payload = {
            "schema": "STAGE22_PROFILE_SELECTION_GATE_V1",
            "status": "PASS",
            "mode": "select",
            "r0_input": str(args.r0),
            "r1_input": str(args.r1),
            **select_recipe(r0, r1),
        }
    else:
        if args.aggregate is None or args.selected_recipe is None:
            parser.error("final mode requires --aggregate and --selected-recipe")
        aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
        if aggregate.get("split") != "final_test":
            parser.error("final mode requires the locked final_test split")
        checks = qualification_checks(aggregate)
        payload = {
            "schema": "STAGE22_PROFILE_SELECTION_GATE_V1",
            "status": "PASS" if checks["passed"] else "FAIL",
            "mode": "final",
            "aggregate_input": str(args.aggregate),
            "selected_recipe": args.selected_recipe,
            "qualification": checks,
            "final_test_policy": "single_locked_read_after_model_selection",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
