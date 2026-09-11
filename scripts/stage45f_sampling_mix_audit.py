#!/usr/bin/env python3
"""CPU-only sampler-mix audit for hypothetical expanded TRAIN datasets.

Adding windows changes training composition even when model/loss/stats stay
fixed.  This tool forces the future sampling policy to be explicit and reports
nominal group/window exposure probabilities.  It never loads a dataset or
launches training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class SamplingMixError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SamplingMixError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_nonnegative(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    result = float(value)
    require(math.isfinite(result) and result >= 0.0, f"{field} must be finite nonnegative")
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "input must be object")
    require(value.get("schema") == "STAGE45F_SAMPLING_MIX_PROPOSAL_V1", "schema drift")
    require(value.get("source_split") == "TRAIN", "sampler mix must be TRAIN-only")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    policy = value.get("policy")
    require(policy in {"UNIFORM_WINDOW", "EXPLICIT_GROUP_WEIGHT"}, "unsupported sampling policy")
    groups = value.get("groups")
    require(isinstance(groups, list) and groups, "groups missing")

    clean: list[dict[str, Any]] = []
    names: set[str] = set()
    total_windows = 0
    explicit_weight_sum = 0.0
    for index, group in enumerate(groups):
        require(isinstance(group, Mapping), f"group[{index}] must be object")
        name = group.get("name")
        require(isinstance(name, str) and name, f"group[{index}].name missing")
        require(name not in names, f"duplicate group name: {name}")
        names.add(name)
        count = group.get("window_count")
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0, f"group[{index}].window_count invalid")
        role = group.get("role")
        require(role in {"BASE_FULL24", "NEW_STATIC_EXPERT", "ROLLOUT_ORACLE_RELABEL"}, f"group[{index}].role invalid")
        weight = group.get("group_weight")
        if policy == "EXPLICIT_GROUP_WEIGHT":
            weight = finite_nonnegative(weight, f"group[{index}].group_weight")
            require(weight > 0.0, f"group[{index}].group_weight must be positive")
            explicit_weight_sum += weight
        else:
            require(weight is None, "UNIFORM_WINDOW must not carry group_weight")
        clean.append({"name": name, "window_count": count, "role": role, "group_weight": weight})
        total_windows += count

    output_groups: list[dict[str, Any]] = []
    for group in clean:
        if policy == "UNIFORM_WINDOW":
            group_probability = group["window_count"] / total_windows
            per_window_probability = 1.0 / total_windows
        else:
            group_probability = group["group_weight"] / explicit_weight_sum
            per_window_probability = group_probability / group["window_count"]
        output_groups.append({
            **group,
            "nominal_group_probability": group_probability,
            "nominal_per_window_probability": per_window_probability,
        })

    base_groups = [item for item in output_groups if item["role"] == "BASE_FULL24"]
    require(len(base_groups) == 1, "exactly one BASE_FULL24 group is required")
    base = base_groups[0]
    return {
        "schema": "STAGE45F_SAMPLING_MIX_AUDIT_V1",
        "status": "PASS_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "policy": policy,
        "total_window_count": total_windows,
        "groups": output_groups,
        "base_full24_nominal_group_probability": base["nominal_group_probability"],
        "base_full24_nominal_per_window_probability": base["nominal_per_window_probability"],
        "composition_changes_when_new_windows_added": True,
        "exact_realized_exposure_requires_runtime_trace": True,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Nominal probabilities are sampler-contract calculations only. Realized per-window exposure must be checked from the exact exposure trace.",
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
    except SamplingMixError as exc:
        print(f"STAGE45F_SAMPLING_MIX_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
