#!/usr/bin/env python3
"""Preparation-only retry/recovery governance for future experiments.

This tool encodes lessons from prior engineering failures: a completed valid
training run should be recovered/audited rather than retrained because a later
post-processing step failed; resource or scientific changes must never be hidden
inside an engineering retry; and valid retries cannot be selected by metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


FAILURE_CLASSES = {
    "ENGINEERING_TRANSIENT",
    "ENGINEERING_DETERMINISTIC",
    "POSTPROCESSING_AFTER_VALID_TRAINING",
    "RESOURCE_LIMIT",
    "SCIENTIFIC_INTEGRITY",
    "UNKNOWN",
}


class RetryPolicyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RetryPolicyError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_RETRY_RECOVERY_REQUEST_V1", "schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    failure_class = value.get("failure_class")
    require(failure_class in FAILURE_CLASSES, "failure_class invalid")
    require(value.get("failed_evidence_preserved") is True, "failed attempt evidence must be preserved")
    require(value.get("metric_based_retry_selection") is False, "metric-based retry selection is forbidden")
    valid_training_completed = value.get("valid_training_completed")
    require(isinstance(valid_training_completed, bool), "valid_training_completed must be bool")
    checkpoint_complete = value.get("checkpoint_artifacts_complete")
    require(isinstance(checkpoint_complete, bool), "checkpoint_artifacts_complete must be bool")
    proposed_science_change = value.get("proposed_retry_changes_science_contract")
    require(isinstance(proposed_science_change, bool), "proposed_retry_changes_science_contract must be bool")
    proposed_resource_semantics_change = value.get("proposed_retry_changes_resource_semantics")
    require(isinstance(proposed_resource_semantics_change, bool), "proposed_retry_changes_resource_semantics must be bool")

    if failure_class == "POSTPROCESSING_AFTER_VALID_TRAINING":
        require(valid_training_completed and checkpoint_complete, "postprocessing recovery requires complete valid training/checkpoints")
        recommended = "READONLY_RECOVERY_AUDIT_DO_NOT_RETRAIN"
        retry_training_allowed_by_this_policy = False
    elif failure_class == "SCIENTIFIC_INTEGRITY":
        recommended = "STOP_FOR_NEW_PI_SCIENTIFIC_GATE"
        retry_training_allowed_by_this_policy = False
    elif failure_class == "RESOURCE_LIMIT":
        recommended = "RESOURCE_PORT_OR_SAME_GRAPH_RETRY_REQUIRES_SEPARATE_RESOURCE_AUDIT"
        retry_training_allowed_by_this_policy = False
    elif failure_class in {"ENGINEERING_TRANSIENT", "ENGINEERING_DETERMINISTIC"}:
        if proposed_science_change or proposed_resource_semantics_change:
            recommended = "STOP_RETRY_PROPOSAL_CHANGES_CONTRACT"
            retry_training_allowed_by_this_policy = False
        elif valid_training_completed:
            recommended = "PREFER_READONLY_RECOVERY_OR_DOWNSTREAM_RESTART_NOT_RETRAIN"
            retry_training_allowed_by_this_policy = False
        else:
            recommended = "EXACT_SAME_CONTRACT_RETRY_CANDIDATE_REQUIRES_EXISTING_OR_NEW_PI_AUTHORITY"
            retry_training_allowed_by_this_policy = False
    else:
        recommended = "STOP_AND_CLASSIFY_FAILURE_BEFORE_RETRY"
        retry_training_allowed_by_this_policy = False

    return {
        "schema": "STAGE45F_RETRY_RECOVERY_POLICY_V1",
        "status": "PASS_POLICY_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "failure_class": failure_class,
        "recommended_action": recommended,
        "valid_training_completed": valid_training_completed,
        "checkpoint_artifacts_complete": checkpoint_complete,
        "failed_evidence_preserved": True,
        "metric_based_retry_selection_allowed": False,
        "science_contract_change_inside_engineering_retry_allowed": False,
        "resource_semantics_change_inside_engineering_retry_allowed": False,
        "retry_training_authorized_by_this_policy": retry_training_allowed_by_this_policy,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "This policy classifies recovery/retry handling only. It never creates execution authority; valid completed runs must not be discarded in favor of a better-performing retry.",
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
    except RetryPolicyError as exc:
        print(f"STAGE45F_RETRY_POLICY_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
