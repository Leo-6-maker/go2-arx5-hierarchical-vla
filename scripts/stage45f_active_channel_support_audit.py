#!/usr/bin/env python3
"""CPU-only support audit across all active StructuredAction semantic channels.

Unlike the XYZ-focused expansion audit, this tool can inspect proposed values
for EEF XYZ, rot6d, timing, and gripper under caller-supplied frozen centers and
half-ranges. It uses an explicit semantic-channel contract and never opens the
training dataset or statistics files itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


EXPECTED_CHANNELS = (
    "eef_x", "eef_y", "eef_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "timing", "gripper",
)
EXPECTED_ACTIVE_ROWS = {
    "eef_x": 15, "eef_y": 15, "eef_z": 15,
    "rot6d_0": 15, "rot6d_1": 15, "rot6d_2": 15,
    "rot6d_3": 15, "rot6d_4": 15, "rot6d_5": 15,
    "timing": 15,
    "gripper": 40,
}


class ActiveSupportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveSupportError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    out = float(value)
    require(math.isfinite(out), f"{field} must be finite")
    return out


def analyze(document: Any) -> dict[str, Any]:
    require(isinstance(document, Mapping), "input must be object")
    require(document.get("schema") == "STAGE45F_ACTIVE_CHANNEL_SUPPORT_PROPOSAL_V1", "schema drift")
    require(document.get("source_split") == "TRAIN", "proposal must be TRAIN-only")
    require(document.get("validation_e_read") is False and document.get("holdout_f_read") is False, "locked split read indicated")
    channels = document.get("channels")
    require(isinstance(channels, Mapping), "channels block missing")
    require(tuple(channels.keys()) == EXPECTED_CHANNELS, "active semantic channel order/set drift")

    frozen: dict[str, dict[str, float | int]] = {}
    for name in EXPECTED_CHANNELS:
        spec = channels[name]
        require(isinstance(spec, Mapping), f"channel {name} spec missing")
        rows = spec.get("active_rows")
        require(rows == EXPECTED_ACTIVE_ROWS[name], f"channel {name} active_rows drift")
        center = finite(spec.get("center"), f"{name}.center")
        half = finite(spec.get("half_range"), f"{name}.half_range")
        require(half > 0.0, f"{name}.half_range must be positive")
        frozen[name] = {"active_rows": rows, "center": center, "half_range": half}

    examples = document.get("examples")
    require(isinstance(examples, list) and examples, "examples missing")
    per_channel: dict[str, dict[str, Any]] = {
        name: {
            "sample_count": 0,
            "outside_neg1_pos1_count": 0,
            "normalized_min": None,
            "normalized_max": None,
            "max_abs_normalized": 0.0,
        }
        for name in EXPECTED_CHANNELS
    }
    window_any = 0
    for index, example in enumerate(examples):
        require(isinstance(example, Mapping), f"example[{index}] must be object")
        require(example.get("validation_or_holdout_origin") is False, f"example[{index}] comes from locked split")
        values = example.get("values")
        require(isinstance(values, Mapping), f"example[{index}].values missing")
        require(tuple(values.keys()) == EXPECTED_CHANNELS, f"example[{index}] channel order/set drift")
        clipped_window = False
        for name in EXPECTED_CHANNELS:
            sequence = values[name]
            require(isinstance(sequence, list) and len(sequence) == EXPECTED_ACTIVE_ROWS[name], f"example[{index}].{name} length drift")
            spec = frozen[name]
            record = per_channel[name]
            for row_index, raw in enumerate(sequence):
                physical = finite(raw, f"example[{index}].{name}[{row_index}]")
                normalized = (physical - float(spec["center"])) / float(spec["half_range"])
                record["sample_count"] += 1
                if abs(normalized) > 1.0:
                    record["outside_neg1_pos1_count"] += 1
                    clipped_window = True
                record["normalized_min"] = normalized if record["normalized_min"] is None else min(record["normalized_min"], normalized)
                record["normalized_max"] = normalized if record["normalized_max"] is None else max(record["normalized_max"], normalized)
                record["max_abs_normalized"] = max(record["max_abs_normalized"], abs(normalized))
        if clipped_window:
            window_any += 1

    groups = {
        "XYZ": ("eef_x", "eef_y", "eef_z"),
        "ROT6D": tuple(name for name in EXPECTED_CHANNELS if name.startswith("rot6d_")),
        "TIMING": ("timing",),
        "GRIPPER": ("gripper",),
    }
    for record in per_channel.values():
        record["outside_neg1_pos1_fraction"] = record["outside_neg1_pos1_count"] / record["sample_count"]
    group_summary: dict[str, Any] = {}
    for group_name, names in groups.items():
        count = sum(per_channel[name]["sample_count"] for name in names)
        outside = sum(per_channel[name]["outside_neg1_pos1_count"] for name in names)
        group_summary[group_name] = {
            "sample_count": count,
            "outside_neg1_pos1_count": outside,
            "outside_neg1_pos1_fraction": outside / count if count else None,
        }

    return {
        "schema": "STAGE45F_ACTIVE_CHANNEL_SUPPORT_AUDIT_V1",
        "status": "PASS_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "window_count": len(examples),
        "frozen_channel_contract": frozen,
        "per_channel": per_channel,
        "group_summary": group_summary,
        "windows_with_any_active_channel_outside_neg1_pos1": window_any,
        "window_fraction_with_any_active_channel_outside_neg1_pos1": window_any / len(examples),
        "automatic_acceptability_threshold_used": False,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Support/clipping is descriptive under caller-bound frozen normalization. This tool does not choose whether current stats are scientifically acceptable for expanded data.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(analyze(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ActiveSupportError as exc:
        print(f"STAGE45F_ACTIVE_CHANNEL_SUPPORT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
