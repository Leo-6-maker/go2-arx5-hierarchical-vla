"""Manifest-driven Stage45C evaluation orchestration.

The module is intentionally a *runtime adapter boundary*, not a GR00T
launcher.  Vanda's Golden Release/P0-P1 manifests are late-bound here; until
they are present the manifest remains ``PREPARED_NOT_EXECUTED`` and no model
construction is possible.  Given an externally supplied raw prediction, the
CPU pipeline is:

``raw prediction -> accepted Stage40 decode -> per-window metrics -> aggregate``

Raw arrays are never folded into aggregate statistics.  Their artifact path
and array hashes remain explicit in each row so a later H200 runner can keep
the raw evidence alongside the metric report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .structured_action_eval import (
    EXPECTED_MODEL_INPUT_KEYS,
    PHYSICAL_ACTION_SHAPE,
    RAW_ACTION_SHAPE,
    STATE_V1_SHAPE,
    decode_raw_prediction,
    validate_model_input_contract,
)
from .trajectory_metrics import (
    ACTION_HORIZON,
    ACTION_WIDTH,
    EvaluationContractError,
    aggregate_trajectory_metrics,
    array_sha256,
    comparison_metrics,
    numeric_summary,
)


EVALUATION_MANIFEST_SCHEMA = "CORELAB_STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1"
EVALUATION_ROW_SCHEMA = "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1"
EXPECTED_WINDOW_COUNT = 24
EXPECTED_TRAIN_TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
EXPECTED_WINDOWS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in EXPECTED_TRAIN_TARGETS
    for index in range(6)
)
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"


class EvaluationManifestError(ValueError):
    """Raised when a Stage45C manifest is incomplete or semantically unsafe."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Load a regular JSON object and validate its Stage45C schema."""

    manifest_path = Path(path).resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvaluationManifestError(f"manifest is not a regular file: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationManifestError(f"cannot read manifest: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise EvaluationManifestError("Stage45C manifest must be a JSON object")
    validate_evaluation_manifest(value)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationManifestError(message)


def _scan_for_locked_data_paths(value: Any, *, key_path: str = "") -> None:
    """Reject accidental Validation-E/Holdout-F materialization in manifests."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            lowered = str(key).lower()
            if lowered in {"validation_e", "holdout_f", "validation-e", "holdout-f"}:
                _require(child in ("LOCKED", "locked", None), f"locked data field is not locked: {child_path}")
            _scan_for_locked_data_paths(child, key_path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_for_locked_data_paths(child, key_path=f"{key_path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        _require("validation-e" not in lowered and "holdout-f" not in lowered, f"locked data path embedded at {key_path}")


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the design and future bound forms of the Stage45C manifest."""

    _require(manifest.get("schema") == EVALUATION_MANIFEST_SCHEMA, "Stage45C manifest schema mismatch")
    _require(manifest.get("stage") == "Stage45C", "Stage45C manifest stage mismatch")
    _require(str(manifest.get("source_golden_head", "")) == "8197e4cd591b9c44cf9f43576d3020e9e6ff5f24", "Golden HEAD mismatch")
    _require(manifest.get("status") in {"PREPARED_NOT_EXECUTED", "BOUND_NOT_EXECUTED", "READY_FOR_EXECUTION"}, "invalid Stage45C manifest status")

    cohort = manifest.get("cohort")
    _require(isinstance(cohort, Mapping), "cohort section is required")
    _require(cohort.get("name") == "FULL24_EXISTING_TRAIN", "Stage45C cohort must be FULL24_EXISTING_TRAIN")
    _require(cohort.get("split") == "TRAIN", "Stage45C cohort split must be TRAIN")
    _require(int(cohort.get("expected_window_count", -1)) == EXPECTED_WINDOW_COUNT, "Stage45C cohort count must be 24")
    _require(cohort.get("window_order") == "target_major_then_snapshot_major", "Stage45C window order drift")
    _require(list(cohort.get("window_ids", [])) == list(EXPECTED_WINDOWS), "Stage45C window IDs/order drift")

    contract = manifest.get("contract")
    _require(isinstance(contract, Mapping), "contract section is required")
    _require(contract.get("structured_action_schema") == "StructuredActionV1", "StructuredAction schema drift")
    _require(list(contract.get("raw_action_shape", [])) == list(RAW_ACTION_SHAPE), "raw action shape drift")
    _require(list(contract.get("decoded_action_shape", [])) == list(PHYSICAL_ACTION_SHAPE), "decoded action shape drift")
    _require(list(contract.get("state_v1_shape", [])) == list(STATE_V1_SHAPE), "StateV1 shape drift")
    _require(contract.get("source_frame") == SOURCE_FRAME, "source frame drift")
    _require(contract.get("tcp_semantic_id") == TCP_SEMANTIC_ID, "TCP semantic drift")
    _require(tuple(contract.get("model_input_keys", [])) == EXPECTED_MODEL_INPUT_KEYS, "model input key drift")
    _require(contract.get("target_xyz_in_model_input") is False, "numeric target input boundary drift")
    _require(contract.get("decoder") == "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action", "decoder authority drift")

    locked = manifest.get("locked_inputs")
    _require(isinstance(locked, Mapping), "locked_inputs section is required")
    _require(locked.get("Validation-E") == "LOCKED" and locked.get("Holdout-F") == "LOCKED", "Validation-E/Holdout-F must remain locked")
    _require(locked.get("WBC") == "LOCKED" and locked.get("new_data") == "LOCKED", "WBC/new-data boundary drift")

    late_bound = manifest.get("late_bound")
    _require(isinstance(late_bound, Sequence) and not isinstance(late_bound, (str, bytes)), "late_bound list is required")
    required_late_bound = {
        "golden_release_manifest",
        "groot_runtime_manifest",
        "full24_manifest",
        "checkpoint_manifest",
        "runtime_admission",
        "structured_statistics",
        "sampler_noise_seed",
        "output_root",
    }
    _require(required_late_bound.issubset(set(late_bound)), "required Golden Release identities are not late-bound")

    budget = manifest.get("execution_budget")
    _require(isinstance(budget, Mapping), "execution_budget section is required")
    _require(int(budget.get("forwards_per_checkpoint", -1)) == EXPECTED_WINDOW_COUNT, "forward budget drift")
    _require(int(budget.get("reference_updates", -1)) == 0 and int(budget.get("wbc_steps", -1)) == 0, "WBC budget must remain zero")

    _scan_for_locked_data_paths(manifest)


def runtime_is_bound(manifest: Mapping[str, Any]) -> bool:
    """Return whether Vanda's immutable runtime/checkpoint identities are present."""

    binding = manifest.get("runtime_binding")
    if not isinstance(binding, Mapping):
        return False
    required_identity_maps = (
        "golden_release_manifest",
        "groot_runtime_manifest",
        "full24_manifest",
        "checkpoint_manifest",
        "runtime_admission",
        "structured_statistics",
    )
    if not all(
        isinstance(binding.get(key), Mapping) and bool(binding[key].get("sha256"))
        for key in required_identity_maps
    ):
        return False
    return binding.get("sampler_noise_seed") is not None and bool(binding.get("output_root"))


def require_runtime_bound(manifest: Mapping[str, Any]) -> None:
    validate_evaluation_manifest(manifest)
    if not runtime_is_bound(manifest):
        raise EvaluationManifestError(
            "Stage45C runtime identities are late-bound; wait for Vanda P0/P1 Golden Release manifests"
        )


def validate_safety_gate_decision(decision: str) -> None:
    if str(decision) not in {"ACCEPT", "WARN"}:
        raise EvaluationContractError(f"SafetyGate decision is not admissible: {decision}")


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if tuple(array.shape) != shape or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise EvaluationContractError(f"{name} must be finite numeric {shape}")
    return array


def _required_identity(record: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    value = str(record.get(f"{label}_label", "")).strip()
    digest = str(record.get(f"{label}_sha256", "")).strip().lower()
    if not value:
        raise EvaluationContractError(f"{label}_label is required")
    if len(digest) != 64:
        raise EvaluationContractError(f"{label}_sha256 must be a 64-character digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise EvaluationContractError(f"{label}_sha256 is not hexadecimal") from exc
    return value, digest


def evaluate_raw_prediction(
    record: Mapping[str, Any],
    *,
    stats_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one externally supplied raw prediction without model access."""

    window_id = str(record.get("window_id", ""))
    if window_id not in EXPECTED_WINDOWS:
        raise EvaluationContractError(f"unknown FULL24 window_id: {window_id}")
    checkpoint_label, checkpoint_sha256 = _required_identity(record, label="checkpoint")
    model_label, model_sha256 = _required_identity(record, label="model")
    raw_artifact_path = str(record.get("raw_prediction_artifact_path", "")).strip()
    raw_artifact_sha256 = str(record.get("raw_prediction_artifact_sha256", "")).strip().lower()
    if not raw_artifact_path:
        raise EvaluationContractError("raw_prediction_artifact_path is required")
    if len(raw_artifact_sha256) != 64:
        raise EvaluationContractError("raw_prediction_artifact_sha256 must be a 64-character digest")
    try:
        int(raw_artifact_sha256, 16)
    except ValueError as exc:
        raise EvaluationContractError("raw_prediction_artifact_sha256 is not hexadecimal") from exc
    validate_model_input_contract(
        record.get("model_input_keys", ()),
        target_xyz_in_model_input=bool(record.get("target_xyz_in_model_input", True)),
    )
    if "safety_gate_decision" in record:
        validate_safety_gate_decision(str(record["safety_gate_decision"]))
    raw = _array(record.get("raw_universal_action"), name="raw_universal_action", shape=RAW_ACTION_SHAPE)
    state = _array(record.get("state_v1"), name="state_v1", shape=STATE_V1_SHAPE)
    expert = _array(record.get("expert_physical_action"), name="expert_physical_action", shape=PHYSICAL_ACTION_SHAPE)
    anchor = _array(record.get("anchor_world_m"), name="anchor_world_m", shape=(3,))
    target = _array(record.get("target_world_m"), name="target_world_m", shape=(3,))
    source_timestamps = _array(record.get("source_timestamps"), name="source_timestamps", shape=(ACTION_HORIZON,))
    decoded = decode_raw_prediction(
        raw,
        state,
        stats_path=Path(stats_path),
        source_timestamps=source_timestamps,
        observation_timestamp=float(record.get("observation_timestamp", source_timestamps[0])),
        chunk_id=window_id,
    )
    metrics = comparison_metrics(
        decoded.physical_action,
        expert,
        anchor,
        target,
        prefix=f"{window_id}.prediction",
    )
    return {
        "schema": EVALUATION_ROW_SCHEMA,
        "window_id": window_id,
        "raw_prediction_artifact": {
            "path": raw_artifact_path,
            "sha256": raw_artifact_sha256,
            "array_sha256": array_sha256(raw),
            "shape": list(raw.shape),
        },
        "checkpoint": {"label": checkpoint_label, "sha256": checkpoint_sha256},
        "model": {"label": model_label, "sha256": model_sha256},
        "decoded_action": {
            "array_sha256": array_sha256(decoded.physical_action),
            "shape": list(decoded.physical_action.shape),
            "decoder_details": dict(decoded.decoder_details),
        },
        "metrics": metrics,
        "safety_gate_decision": record.get("safety_gate_decision"),
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "target_xyz_in_model_input": False,
        "finite": True,
    }


def aggregate_evaluation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_window_ids: Sequence[str] = EXPECTED_WINDOWS,
) -> dict[str, Any]:
    """Aggregate per-window rows while retaining raw/decoded hash identity."""

    normalized = [dict(row) for row in rows]
    expected = list(expected_window_ids)
    if len(normalized) != len(expected):
        raise EvaluationContractError(f"expected {len(expected)} evaluation rows, got {len(normalized)}")
    observed_ids = [str(row.get("window_id", "")) for row in normalized]
    if observed_ids != expected or len(set(observed_ids)) != len(observed_ids):
        raise EvaluationContractError("evaluation rows do not preserve the frozen cohort order")
    checkpoint_identities = [row.get("checkpoint") for row in normalized]
    model_identities = [row.get("model") for row in normalized]
    if any(not isinstance(item, Mapping) for item in checkpoint_identities + model_identities):
        raise EvaluationContractError("each row must carry checkpoint and model identities")
    if len({(item.get("label"), item.get("sha256")) for item in checkpoint_identities}) != 1:
        raise EvaluationContractError("checkpoint identity drift within one aggregate")
    if len({(item.get("label"), item.get("sha256")) for item in model_identities}) != 1:
        raise EvaluationContractError("model identity drift within one aggregate")
    target_metrics = [row.get("metrics", {}).get("target") for row in normalized]
    if any(not isinstance(item, Mapping) for item in target_metrics):
        raise EvaluationContractError("each row must contain target metrics")
    scalar_fields = (
        "position_rmse_m",
        "position_max_abs_m",
        "endpoint_error_m",
        "prediction_path_length_m",
        "expert_path_length_m",
        "eef_rot6d_rmse",
        "eef_rot6d_max_abs",
        "gripper_rmse",
        "gripper_max_abs",
        "full_action_rmse_mixed_units",
    )
    comparison_summary = {
        field: numeric_summary(float(row["metrics"][field]) for row in normalized)
        for field in scalar_fields
    }
    roughness_ratios = [
        row["metrics"].get("roughness_ratio")
        for row in normalized
        if row["metrics"].get("roughness_ratio") is not None
    ]
    comparison_summary["roughness_ratio"] = (
        numeric_summary(float(value) for value in roughness_ratios)
        if roughness_ratios
        else None
    )
    # JSON object keys are strings; keep the missing-decision bucket stable
    # instead of letting sort_keys compare None with the named decisions.
    safety_counts = {
        "null" if decision is None else decision: int(sum(row.get("safety_gate_decision") == decision for row in normalized))
        for decision in ("ACCEPT", "WARN", None)
    }
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_AGGREGATE_V1",
        "window_count": len(normalized),
        "window_ids": observed_ids,
        "checkpoint": dict(checkpoint_identities[0]),
        "model": dict(model_identities[0]),
        "raw_prediction_array_sha256": [row["raw_prediction_artifact"]["array_sha256"] for row in normalized],
        "decoded_action_array_sha256": [row["decoded_action"]["array_sha256"] for row in normalized],
        "target_metrics": aggregate_trajectory_metrics(target_metrics),
        "comparison_metrics": comparison_summary,
        "safety_gate_decision_counts": safety_counts,
        "raw_predictions_preserved_separately": True,
        "threshold_used": None,
        "finite": True,
    }


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "EVALUATION_ROW_SCHEMA",
    "EXPECTED_WINDOWS",
    "EvaluationManifestError",
    "aggregate_evaluation_rows",
    "canonical_json_bytes",
    "evaluate_raw_prediction",
    "load_evaluation_manifest",
    "require_runtime_bound",
    "runtime_is_bound",
    "validate_evaluation_manifest",
    "validate_safety_gate_decision",
]
