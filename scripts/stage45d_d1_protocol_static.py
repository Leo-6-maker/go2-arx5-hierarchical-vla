#!/usr/bin/env python3
"""CPU-only contract verifier for the corrected Stage45D D1 protocol.

This verifier imports only the repository's CPU StructuredActionV1 codec.  It
does not import GR00T, construct a model, read a checkpoint, generate an
inference tensor, submit PBS, or access any simulator/hardware.  Its purpose
is to make the D1 protocol correction fail closed before a later PI-authorized
execution pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.structured_action import (  # noqa: E402
    structured_action_mask,
    structured_active_scalar_count,
)


MANIFEST = ROOT / "manifests" / "STAGE45D_D1_PROTOCOL_CORRECTED_V1.json"
EXPECTED_SHAPE = (40, 132)
EXPECTED_ACTIVE_SCALARS = 190
EXPECTED_CHECKPOINTS = ("STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
LEGACY_MARKERS = (
    "ACTIVE_DIM=10",
    "INACTIVE_DIM=122",
    "190 active / 122",
    '"persistent-good"',
    '"persistent-negative"',
    '"recovered-at-2000"',
    '"regressed-at-2000"',
    '"mixed-sign-or-zero"',
)


class ProtocolContractError(ValueError):
    """Raised when a corrected D1 contract is not fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolContractError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolContractError(f"cannot read JSON manifest: {path}") from exc
    _require(isinstance(payload, dict), "manifest root must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mask(candidate: Any) -> dict[str, Any]:
    """Validate a candidate against the canonical Stage38/45 T16 mask."""

    observed = np.asarray(candidate, dtype=np.float64)
    _require(observed.shape == EXPECTED_SHAPE, f"mask shape drifted: {observed.shape}")
    canonical = structured_action_mask("T16")
    declarative = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    declarative[:15, :9] = 1.0
    declarative[:15, 9] = 1.0
    declarative[:, 10] = 1.0
    _require(np.array_equal(canonical, declarative), "repository T16 mask does not match the D1 block contract")
    _require(np.array_equal(observed, canonical), "mask is not structured_action_mask(\"T16\")")
    _require(
        int(np.count_nonzero(observed)) == EXPECTED_ACTIVE_SCALARS,
        "mask active scalar count is not exactly 190",
    )
    _require(
        structured_active_scalar_count("T16") == EXPECTED_ACTIVE_SCALARS,
        "repository StructuredActionV1 active count changed",
    )

    block_counts = {
        "geometry": int(np.count_nonzero(observed[:15, :9])),
        "timing": int(np.count_nonzero(observed[:15, 9])),
        "gripper": int(np.count_nonzero(observed[:, 10])),
        "remaining": int(np.count_nonzero(observed[:, 11:])),
    }
    _require(block_counts == {"geometry": 135, "timing": 15, "gripper": 40, "remaining": 0},
             f"mask block topology drifted: {block_counts}")
    return {
        "shape": list(observed.shape),
        "active_scalar_count": int(np.count_nonzero(observed)),
        "inactive_scalar_count": int(observed.size - np.count_nonzero(observed)),
        "block_counts": block_counts,
        "source": "go2arx5_vla.structured_action.structured_action_mask(\"T16\")",
    }


def reject_legacy_first_ten_mask() -> dict[str, Any]:
    """Negative contract test for the historical dense first-ten mask."""

    legacy = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    legacy[:, :10] = 1.0
    try:
        validate_mask(legacy)
    except ProtocolContractError as exc:
        return {
            "rejected": True,
            "legacy_active_scalar_count": int(np.count_nonzero(legacy)),
            "reason": str(exc),
        }
    raise ProtocolContractError("legacy dense first-ten mask was accepted")


def validate_protocol(path: Path = MANIFEST) -> dict[str, Any]:
    payload = _load(path)
    serialized = path.read_text(encoding="utf-8")
    for marker in LEGACY_MARKERS:
        _require(marker not in serialized, f"legacy D1 marker leaked into corrected manifest: {marker}")

    _require(
        payload.get("schema") == "CORELAB_STAGE45D_D1_TEACHER_FORCED_DIAGNOSTIC_PROTOCOL_CORRECTED_V1",
        "wrong corrected D1 schema",
    )
    _require(payload.get("status") == "CORRECTED_FOR_PI_REVIEW_NOT_EXECUTED", "D1 is not review-only")
    authority = payload["authority"]
    _require(authority["issue_11_comment"] == 5617025466, "PI0104 Issue #11 authority drift")
    _require(authority["issue_50_comment"] == 5617028048, "PI0104 Issue #50 authority drift")
    _require(authority["historical_evidence_preserved"] is True, "historical evidence preservation missing")

    cohort = payload["scientific_scope"]
    _require(cohort["split"] == "TRAIN_ONLY", "D1 cohort is not TRAIN_ONLY")
    _require(cohort["window_count"] == 24, "D1 window count drift")
    _require(cohort["target_xyz_in_model_input"] is False, "numeric target leaked into model inputs")

    contract = payload["contract"]
    _require(contract["raw_action_shape"] == [1, 40, 132], "raw action ABI drift")
    _require(contract["model_input_keys"] == ["body_rgb", "wrist_rgb", "state", "language"], "model input ABI drift")

    bindings = payload["checkpoint_bindings"]
    _require(tuple(item["label"] for item in bindings) == EXPECTED_CHECKPOINTS, "checkpoint order drift")
    _require(all(item["statistics_sha256"] for item in bindings), "checkpoint statistics identity missing")

    mask = validate_mask(structured_action_mask("T16"))
    declared_mask = payload["supervision_mask"]
    _require(declared_mask["shape"] == list(EXPECTED_SHAPE), "declared mask shape drift")
    _require(declared_mask["active_scalar_count"] == EXPECTED_ACTIVE_SCALARS, "declared mask count drift")
    _require(declared_mask["inactive_scalar_count"] == 5090, "declared inactive scalar count drift")
    _require(declared_mask["active_blocks"]["geometry"]["scalar_count"] == 135, "geometry block drift")
    _require(declared_mask["active_blocks"]["timing"]["scalar_count"] == 15, "timing block drift")
    _require(declared_mask["active_blocks"]["gripper"]["scalar_count"] == 40, "gripper block drift")

    metrics = payload["metrics"]
    normalized = metrics["normalized_training_objective_space"]
    physical = metrics["physical_interpretation_space"]
    _require(normalized["name"] == "NORMALIZED_LOCAL_FLOW", "normalized metric namespace drift")
    _require(physical["name"] == "DECODED_PHYSICAL_ACTION", "physical metric namespace drift")
    _require(len(normalized["macro_summary"]) == 3, "normalized macro-summary contract incomplete")
    _require("eef_position_rmse_m" in physical["per_probe_or_aggregate"], "physical EEF metric missing")
    _require(metrics["raw_before_derived"] is True, "raw-before-derived ordering missing")

    hypotheses = {item["hypothesis"]: item for item in payload["hypothesis_ranking_correction"]["ranking"]}
    _require(hypotheses["H_OPT"]["status"] == "SUPPORTED", "H_OPT narrow status drift")
    _require("continued optimization along the existing 1000→2000 schedule improves sealed TRAIN24 behavior" in hypotheses["H_OPT"]["claim"], "H_OPT claim is not narrow")
    _require(hypotheses["H_SAMPLER"]["status"] == "NOT_TESTED", "H_SAMPLER must be NOT_TESTED")
    _require(hypotheses["H_POLICY"]["status"] == "NOT_TESTED", "H_POLICY must be NOT_TESTED")
    _require(hypotheses["H_DATA_DISTRIBUTION"]["status"] == "SUPPORTED", "H_DATA distribution correction missing")
    _require(hypotheses["H_NORMALIZATION_EFFECT"]["status"] == "PLAUSIBLE", "normalization causal claim too strong")

    taxonomy = payload["taxonomy"]
    _require(all("-" in label for label in taxonomy["progress_sign_labels"]), "taxonomy labels are not descriptive")
    _require("positive-progress-delta-at-coverage1000" in taxonomy["coverage_delta_labels"], "coverage taxonomy missing")
    _require("positive-progress-delta-at-exposure2000" in taxonomy["exposure_delta_labels"], "exposure taxonomy missing")

    boundary = payload["execution_boundary"]
    for key in (
        "model_construction",
        "model_forward",
        "gpu_h200",
        "pbs_qsub",
        "training",
        "validation_e_read",
        "holdout_f_read",
        "wbc_executed",
        "simulator_executed",
        "hardware_executed",
    ):
        _require(boundary[key] is False, f"forbidden operation not zero: {key}")
    _require(all(value == 0 for value in boundary["counters"].values()), "nonzero execution counter")
    _require(payload["execution_plan"]["status"] == "D1_EXECUTION_HOLD_PENDING_PI_AUTHORIZATION", "D1 execution is not held")

    legacy_test = reject_legacy_first_ten_mask()
    return {
        "verdict": "PASS",
        "schema": payload["schema"],
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "mask_contract": mask,
        "legacy_first_ten_mask_negative_test": legacy_test,
        "checkpoint_labels": list(EXPECTED_CHECKPOINTS),
        "normalized_metric_namespace": normalized["name"],
        "physical_metric_namespace": physical["name"],
        "h_sampler": hypotheses["H_SAMPLER"]["status"],
        "h_opt": hypotheses["H_OPT"]["status"],
        "zero_execution_boundary": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        result = validate_protocol(args.manifest)
    except (OSError, ProtocolContractError, KeyError, TypeError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
