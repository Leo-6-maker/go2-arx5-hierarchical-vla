#!/usr/bin/env python3
"""CPU-only compute/storage estimator for hypothetical future scaling.

This tool consumes observed runtime telemetry supplied by PI/agents and reports
wall-time/storage estimates for hypothetical step/seed tiers.  It never selects
a GPU, submits PBS, or authorizes training.  Estimates are descriptive and
carry their source telemetry identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class ComputeBudgetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComputeBudgetError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def finite_positive(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0.0, f"{field} must be finite positive")
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "input must be object")
    require(value.get("schema") == "STAGE45F_COMPUTE_BUDGET_INPUT_V1", "input schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    telemetry = value.get("telemetry")
    require(isinstance(telemetry, list) and telemetry, "telemetry must be non-empty")
    tiers = value.get("hypothetical_tiers")
    require(isinstance(tiers, list) and tiers, "hypothetical_tiers missing")
    seed_counts = value.get("hypothetical_seed_counts")
    require(isinstance(seed_counts, list) and seed_counts, "hypothetical_seed_counts missing")
    checkpoint_steps = value.get("checkpoint_steps")
    require(isinstance(checkpoint_steps, list) and checkpoint_steps, "checkpoint_steps missing")

    clean_telemetry: list[dict[str, Any]] = []
    for index, row in enumerate(telemetry):
        require(isinstance(row, Mapping), f"telemetry[{index}] must be object")
        device_role = row.get("device_role")
        require(device_role in {"CORELAB_4080S", "VANDA_H200"}, f"telemetry[{index}].device_role invalid")
        seconds_per_step = finite_positive(row.get("seconds_per_step"), f"telemetry[{index}].seconds_per_step")
        checkpoint_bytes = int(row.get("checkpoint_bytes"))
        require(checkpoint_bytes > 0, f"telemetry[{index}].checkpoint_bytes invalid")
        source_sha = row.get("source_telemetry_sha256")
        require(isinstance(source_sha, str) and len(source_sha) == 64, f"telemetry[{index}].source_telemetry_sha256 invalid")
        clean_telemetry.append({
            "device_role": device_role,
            "seconds_per_step": seconds_per_step,
            "checkpoint_bytes": checkpoint_bytes,
            "source_telemetry_sha256": source_sha,
        })

    clean_tiers = []
    for step in tiers:
        require(isinstance(step, int) and not isinstance(step, bool) and step > 0, "tier step invalid")
        clean_tiers.append(step)
    clean_seeds = []
    for count in seed_counts:
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0, "seed count invalid")
        clean_seeds.append(count)
    clean_checkpoints = []
    for step in checkpoint_steps:
        require(isinstance(step, int) and not isinstance(step, bool) and step > 0, "checkpoint step invalid")
        clean_checkpoints.append(step)

    estimates: list[dict[str, Any]] = []
    for row in clean_telemetry:
        for step_budget in clean_tiers:
            checkpoints_within = [step for step in clean_checkpoints if step <= step_budget]
            for seed_count in clean_seeds:
                seconds = row["seconds_per_step"] * step_budget * seed_count
                storage = row["checkpoint_bytes"] * len(checkpoints_within) * seed_count
                estimates.append({
                    "device_role": row["device_role"],
                    "step_budget_per_seed": step_budget,
                    "seed_count": seed_count,
                    "checkpoint_steps_within_budget": checkpoints_within,
                    "estimated_serial_walltime_seconds": seconds,
                    "estimated_serial_walltime_hours": seconds / 3600.0,
                    "estimated_checkpoint_storage_bytes": storage,
                    "source_telemetry_sha256": row["source_telemetry_sha256"],
                })

    normalized_input = {
        "telemetry": clean_telemetry,
        "hypothetical_tiers": clean_tiers,
        "hypothetical_seed_counts": clean_seeds,
        "checkpoint_steps": clean_checkpoints,
    }
    return {
        "schema": "STAGE45F_COMPUTE_BUDGET_ESTIMATE_V1",
        "status": "PASS_ESTIMATE_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "input_sha256": canonical_sha(normalized_input),
        "estimates": estimates,
        "resource_selected": False,
        "training_authorized": False,
        "pbs_submissions": 0,
        "gpu_execution": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Linear wall-time/storage projections from observed telemetry. No queueing, caching, checkpoint-I/O, or scaling-efficiency guarantee is implied.",
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
    except ComputeBudgetError as exc:
        print(f"STAGE45F_COMPUTE_BUDGET_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
