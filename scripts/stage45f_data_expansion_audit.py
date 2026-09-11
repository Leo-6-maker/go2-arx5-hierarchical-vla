#!/usr/bin/env python3
"""CPU-only audit for hypothetical Stage45F training-data expansion.

The input is a compact JSON proposal containing only proposed TRAIN examples,
XYZ physical labels, current normalization center/half-range, and provenance.
No repository dataset is opened by this tool.  It reports support/clipping
under the frozen current stats and classifies a numerical stats change as
DATA_PLUS_NORMALIZATION rather than DATA_ONLY. A provenance-only stats identity
change is reported separately and is not treated as a scientific normalization
change by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class DataExpansionAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataExpansionAuditError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    out = float(value)
    require(math.isfinite(out), f"{field} must be finite")
    return out


def _stats(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} must be object")
    center = value.get("xyz_center_m")
    half = value.get("xyz_half_range_m")
    require(isinstance(center, list) and len(center) == 3, f"{name}.xyz_center_m must have 3 values")
    require(isinstance(half, list) and len(half) == 3, f"{name}.xyz_half_range_m must have 3 values")
    c = [finite(item, f"{name}.center[{i}]") for i, item in enumerate(center)]
    h = [finite(item, f"{name}.half[{i}]") for i, item in enumerate(half)]
    require(all(item > 0.0 for item in h), f"{name} half-ranges must be positive")
    identity = value.get("identity")
    require(isinstance(identity, str) and identity, f"{name}.identity missing")
    return {"identity": identity, "xyz_center_m": c, "xyz_half_range_m": h}


def validate(document: Any) -> dict[str, Any]:
    require(isinstance(document, Mapping), "input must be object")
    require(document.get("schema") == "STAGE45F_DATA_EXPANSION_PROPOSAL_V1", "input schema drift")
    require(document.get("source_split") == "TRAIN", "proposed examples must be TRAIN-only")
    require(document.get("validation_e_read") is False, "Validation-E read indicated")
    require(document.get("holdout_f_read") is False, "Holdout-F read indicated")
    require(document.get("contains_validation_e_ids") is False, "Validation-E identities included")
    require(document.get("contains_holdout_f_ids") is False, "Holdout-F identities included")

    current = _stats(document.get("current_stats"), "current_stats")
    proposed_stats_raw = document.get("proposed_stats")
    proposed_stats = None if proposed_stats_raw is None else _stats(proposed_stats_raw, "proposed_stats")

    examples = document.get("examples")
    require(isinstance(examples, list) and examples, "examples must be non-empty list")
    window_ids: set[str] = set()
    source_ids: set[str] = set()
    normalized_examples: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        require(isinstance(example, Mapping), f"example[{index}] must be object")
        window_id = example.get("window_id")
        source_episode_id = example.get("source_episode_id")
        target_id = example.get("target_id")
        label_source = example.get("label_source")
        require(isinstance(window_id, str) and window_id, f"example[{index}].window_id missing")
        require(window_id not in window_ids, f"duplicate window_id: {window_id}")
        window_ids.add(window_id)
        require(isinstance(source_episode_id, str) and source_episode_id, f"example[{index}].source_episode_id missing")
        source_ids.add(source_episode_id)
        require(isinstance(target_id, str) and target_id, f"example[{index}].target_id missing")
        require(label_source in {"INDEPENDENT_EXPERT", "SIMULATOR_EXPERT", "ORACLE", "HUMAN_EXPERT"}, f"example[{index}] label source is not independent expert/oracle")
        xyz = example.get("future_xyz_m")
        require(isinstance(xyz, list) and xyz, f"example[{index}].future_xyz_m missing")
        clean_xyz: list[list[float]] = []
        for row_index, row in enumerate(xyz):
            require(isinstance(row, list) and len(row) == 3, f"example[{index}].future_xyz_m[{row_index}] shape")
            clean_xyz.append([finite(item, f"example[{index}].xyz[{row_index}][{axis}]") for axis, item in enumerate(row)])
        normalized_examples.append({
            "window_id": window_id,
            "source_episode_id": source_episode_id,
            "target_id": target_id,
            "label_source": label_source,
            "future_xyz_m": clean_xyz,
        })

    return {
        "schema": document["schema"],
        "current_stats": current,
        "proposed_stats": proposed_stats,
        "examples": normalized_examples,
        "window_count": len(normalized_examples),
        "source_episode_count": len(source_ids),
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def audit(value: Mapping[str, Any]) -> dict[str, Any]:
    current = value["current_stats"]
    centers = current["xyz_center_m"]
    halves = current["xyz_half_range_m"]
    axis_names = ("x", "y", "z")
    per_axis = {
        axis: {
            "sample_count": 0,
            "outside_neg1_pos1_count": 0,
            "physical_min_m": None,
            "physical_max_m": None,
            "normalized_min": None,
            "normalized_max": None,
            "max_abs_normalized": 0.0,
        }
        for axis in axis_names
    }
    target_counts: dict[str, int] = {}
    label_source_counts: dict[str, int] = {}
    window_any_clip = 0
    scalar_total = 0
    scalar_clipped = 0

    for example in value["examples"]:
        target_counts[example["target_id"]] = target_counts.get(example["target_id"], 0) + 1
        label_source_counts[example["label_source"]] = label_source_counts.get(example["label_source"], 0) + 1
        clipped_this_window = False
        for row in example["future_xyz_m"]:
            for axis_index, axis in enumerate(axis_names):
                physical = row[axis_index]
                normalized = (physical - centers[axis_index]) / halves[axis_index]
                record = per_axis[axis]
                record["sample_count"] += 1
                scalar_total += 1
                if abs(normalized) > 1.0:
                    record["outside_neg1_pos1_count"] += 1
                    scalar_clipped += 1
                    clipped_this_window = True
                record["physical_min_m"] = physical if record["physical_min_m"] is None else min(record["physical_min_m"], physical)
                record["physical_max_m"] = physical if record["physical_max_m"] is None else max(record["physical_max_m"], physical)
                record["normalized_min"] = normalized if record["normalized_min"] is None else min(record["normalized_min"], normalized)
                record["normalized_max"] = normalized if record["normalized_max"] is None else max(record["normalized_max"], normalized)
                record["max_abs_normalized"] = max(record["max_abs_normalized"], abs(normalized))
        if clipped_this_window:
            window_any_clip += 1

    for axis in axis_names:
        record = per_axis[axis]
        record["outside_neg1_pos1_fraction"] = (
            record["outside_neg1_pos1_count"] / record["sample_count"] if record["sample_count"] else None
        )

    proposed_stats = value["proposed_stats"]
    stats_identity_changed = proposed_stats is not None and proposed_stats["identity"] != current["identity"]
    stats_values_changed = bool(
        proposed_stats is not None
        and (
            proposed_stats["xyz_center_m"] != current["xyz_center_m"]
            or proposed_stats["xyz_half_range_m"] != current["xyz_half_range_m"]
        )
    )
    intervention_class = "DATA_PLUS_NORMALIZATION" if stats_values_changed else "DATA_ONLY_CANDIDATE_REQUIRES_PI_SUPPORT_JUDGMENT"
    return {
        "schema": "STAGE45F_DATA_EXPANSION_SUPPORT_AUDIT_V1",
        "status": "PASS_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "intervention_class": intervention_class,
        "window_count": value["window_count"],
        "source_episode_count": value["source_episode_count"],
        "target_window_counts": dict(sorted(target_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "current_stats": current,
        "proposed_stats": proposed_stats,
        "current_stats_support": {
            "per_axis": per_axis,
            "xyz_scalar_total": scalar_total,
            "xyz_scalar_outside_neg1_pos1_count": scalar_clipped,
            "xyz_scalar_outside_neg1_pos1_fraction": scalar_clipped / scalar_total if scalar_total else None,
            "windows_with_any_xyz_outside_neg1_pos1": window_any_clip,
            "window_fraction_with_any_xyz_outside_neg1_pos1": window_any_clip / value["window_count"],
        },
        "stats_identity_changed": stats_identity_changed,
        "stats_numeric_normalization_changed": stats_values_changed,
        "stats_change_detected": stats_values_changed,
        "normalization_decision_required": True,
        "automatic_acceptability_threshold_used": False,
        "groot_self_labels_present": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_authorized": False,
        "input_contract_sha256": canonical_sha(value),
        "claim_boundary": "Clipping/support is descriptive. A provenance-only stats identity change is reported separately from a numerical center/half-range change. The compatibility field stats_change_detected means numerical normalization change only. The tool defines no acceptable clipping threshold and authorizes no data collection or training.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    value = validate(read_json(args.input))
    print(json.dumps(audit(value), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataExpansionAuditError as exc:
        print(f"STAGE45F_DATA_EXPANSION_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
