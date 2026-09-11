#!/usr/bin/env python3
"""CPU-only audit for hypothetical resource changes during future scaling.

Moving a frozen recipe between GPUs is engineering-only only if precision,
per-device batch, gradient accumulation, effective global batch, optimizer
semantics and distributed strategy remain scientifically equivalent.  This
tool reports differences and never selects or launches hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class ResourceScalingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceScalingError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _config(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{name} missing")
    required = (
        "device_role",
        "device_count",
        "per_device_batch",
        "gradient_accumulation_steps",
        "precision",
        "distributed_strategy",
        "optimizer_identity",
        "scheduler_identity",
    )
    for field in required:
        require(field in value, f"{name}.{field} missing")
    device_count = value["device_count"]
    per_device_batch = value["per_device_batch"]
    grad_accum = value["gradient_accumulation_steps"]
    for field, item in (("device_count", device_count), ("per_device_batch", per_device_batch), ("gradient_accumulation_steps", grad_accum)):
        require(isinstance(item, int) and not isinstance(item, bool) and item > 0, f"{name}.{field} invalid")
    precision = value["precision"]
    require(precision in {"BF16", "FP32", "FP16"}, f"{name}.precision invalid")
    strategy = value["distributed_strategy"]
    require(strategy in {"SINGLE_GPU", "DDP", "FSDP", "OTHER"}, f"{name}.distributed_strategy invalid")
    return {
        "device_role": str(value["device_role"]),
        "device_count": device_count,
        "per_device_batch": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_global_batch": device_count * per_device_batch * grad_accum,
        "precision": precision,
        "distributed_strategy": strategy,
        "optimizer_identity": str(value["optimizer_identity"]),
        "scheduler_identity": str(value["scheduler_identity"]),
    }


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "input must be object")
    require(value.get("schema") == "STAGE45F_RESOURCE_SCALING_PROPOSAL_V1", "schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    baseline = _config(value.get("baseline"), "baseline")
    proposal = _config(value.get("proposal"), "proposal")
    fields = (
        "device_count",
        "per_device_batch",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "precision",
        "distributed_strategy",
        "optimizer_identity",
        "scheduler_identity",
    )
    differences = {
        field: {"baseline": baseline[field], "proposal": proposal[field]}
        for field in fields
        if baseline[field] != proposal[field]
    }
    science_affecting_fields = {
        "per_device_batch",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "precision",
        "optimizer_identity",
        "scheduler_identity",
    }
    science_differences = sorted(set(differences).intersection(science_affecting_fields))
    distributed_change = baseline["distributed_strategy"] != proposal["distributed_strategy"] or baseline["device_count"] != proposal["device_count"]
    classification = (
        "RESOURCE_PORT_ONLY_CANDIDATE_REQUIRES_NUMERICAL_EQUIVALENCE_CHECK"
        if not science_differences and not distributed_change
        else "RESOURCE_CHANGE_CONFOUNDS_TRAINING_SEMANTICS_OR_DISTRIBUTION"
    )
    if distributed_change and not science_differences:
        classification = "DISTRIBUTED_EXECUTION_CHANGE_REQUIRES_SEPARATE_NUMERICAL_EQUIVALENCE_GATE"
    return {
        "schema": "STAGE45F_RESOURCE_SCALING_AUDIT_V1",
        "status": "PASS_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "classification": classification,
        "baseline": baseline,
        "proposal": proposal,
        "differences": differences,
        "science_affecting_differences": science_differences,
        "distributed_execution_changed": distributed_change,
        "resource_selected": False,
        "training_authorized": False,
        "gpu_execution": 0,
        "pbs_submissions": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Changing hardware is not assumed science-neutral when batch, precision, optimizer/scheduler, device count or distributed execution changes.",
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
    except ResourceScalingError as exc:
        print(f"STAGE45F_RESOURCE_SCALING_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
