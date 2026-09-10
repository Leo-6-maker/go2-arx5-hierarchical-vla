"""Static PI0102 one-window qualification fixture.

The full Stage45C compatibility proof and the PI0102 qualification fixture
are deliberately separate namespaces.  The former describes the frozen
TRAIN24 scientific protocol; this fixture describes only the one existing
``FULL24@2000`` / window-zero canary contract.  It does not read artifacts,
resolve an adapter, import a model framework, or execute a forward.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .stage45c_compatibility import (
    CHECKPOINT_LABEL,
    DECODED_SHAPE,
    FULL24_2000_TREE_SHA256,
    FULL24_DATASET_TREE_SHA256,
    FULL24_SPLIT_SHA256,
    FULL24_STATS_PAYLOAD_SHA256,
    FULL24_STATS_SHA256,
    LANGUAGE,
    MODEL_INPUT_KEYS,
    RAW_SHAPE,
    SAMPLER_CALLS,
    SAMPLER_DT,
    SAMPLER_LAW,
    SAMPLER_PROFILE,
    SAMPLER_PROTOCOL_REVISION,
    SOURCE_FRAME,
    STATE_V1_SHAPE,
    STAGE45C_COMPATIBILITY_SCOPE,
    TCP_SEMANTIC_ID,
    TIMESTEPS,
    WINDOW_IDS,
    WINDOW_ORDER,
)


PI0102_ONE_WINDOW_FIXTURE_SCHEMA = (
    "GO2ARX5_REUSABLE_PI0102_ONE_WINDOW_QUALIFICATION_FIXTURE_V1"
)
PI0102_ONE_WINDOW_SCOPE = "PI0102_ONE_WINDOW_H200_QUALIFICATION"
PI0102_FIXTURE_STATUS = "STATIC_COMPATIBLE_NOT_EXECUTED"
SELECTED_WINDOW_INDEX = 0
SELECTED_WINDOW_ID = WINDOW_IDS[SELECTED_WINDOW_INDEX]
SELECTED_TARGET_ID = "red_target_train_a"
CHECKPOINT_COUNT = 1
WINDOW_COUNT = 1
EXPECTED_MODEL_FORWARDS = 1
EXPECTED_SAMPLER_CALLS = SAMPLER_CALLS


class PI0102QualificationFixtureError(ValueError):
    """Raised when the one-window fixture loses a frozen identity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0102QualificationFixtureError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _exact(actual: Any, expected: Any, *, field: str) -> None:
    _require(actual == expected, f"{field} differs from the frozen PI0102 fixture")


def build_pi0102_one_window_fixture(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Build the static one-window projection from the frozen protocol."""

    from .stage45c_compatibility import build_stage45c_compatibility_proof

    full_proof = build_stage45c_compatibility_proof(protocol)
    _exact(
        full_proof.get("scope"),
        STAGE45C_COMPATIBILITY_SCOPE,
        field="full_protocol.scope",
    )
    full_projection = _mapping(
        full_proof.get("generic_manifest_projection"),
        field="full_protocol.generic_manifest_projection",
    )
    full_checkpoints = full_projection.get("checkpoints")
    _require(
        isinstance(full_checkpoints, list) and full_checkpoints,
        "full.checkpoints must be a non-empty list",
    )
    full_checkpoint = _mapping(full_checkpoints[0], field="full.checkpoints[0]")
    _exact(full_checkpoint.get("label"), CHECKPOINT_LABEL, field="checkpoint.label")
    _exact(
        full_checkpoint.get("identity", {}).get("sha256"),
        FULL24_2000_TREE_SHA256,
        field="checkpoint.identity.sha256",
    )
    full_contract = dict(_mapping(full_projection.get("contract"), field="full.contract"))
    full_sampler = dict(_mapping(full_projection.get("sampler"), field="full.sampler"))
    full_runtime = dict(
        _mapping(full_projection.get("runtime_binding"), field="full.runtime_binding")
    )
    fixture = {
        "schema": PI0102_ONE_WINDOW_FIXTURE_SCHEMA,
        "scope": PI0102_ONE_WINDOW_SCOPE,
        "status": PI0102_FIXTURE_STATUS,
        "executable": False,
        "source_protocol": {
            "scope": STAGE45C_COMPATIBILITY_SCOPE,
            "schema": full_proof["historical_protocol"]["schema"],
            "protocol_sha256": full_proof["historical_protocol"]["protocol_sha256"],
            "source_head": full_proof["historical_protocol"]["protocol_source_head"],
        },
        "selection": {
            "checkpoint_count": CHECKPOINT_COUNT,
            "checkpoint_label": CHECKPOINT_LABEL,
            "window_count": WINDOW_COUNT,
            "selected_window_index": SELECTED_WINDOW_INDEX,
            "selected_window_id": SELECTED_WINDOW_ID,
            "selected_target_id": SELECTED_TARGET_ID,
            "full_cohort_window_count": 24,
            "full_cohort_window_order": WINDOW_ORDER,
            "expected_model_forwards": EXPECTED_MODEL_FORWARDS,
            "expected_sampler_calls": EXPECTED_SAMPLER_CALLS,
        },
        "generic_manifest_projection": {
            "checkpoints": [
                {
                    "label": CHECKPOINT_LABEL,
                    "count": CHECKPOINT_COUNT,
                    "identity": {
                        "kind": "tree",
                        "sha256": FULL24_2000_TREE_SHA256,
                        "path": "late_bound_from_observed_runtime.checkpoint_paths[FULL24@2000]",
                    },
                    "statistics": {
                        "kind": "file",
                        "sha256": FULL24_STATS_SHA256,
                        "payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
                        "path": "late_bound_from_observed_runtime.statistics_path[FULL24@2000]",
                    },
                }
            ],
            "cohort": {
                "name": "FULL24_EXISTING_TRAIN",
                "split": "TRAIN",
                "identity_tree_sha256": FULL24_DATASET_TREE_SHA256,
                "split_lock_sha256": FULL24_SPLIT_SHA256,
                "window_count": WINDOW_COUNT,
                "window_order": "explicit_manifest_order",
                "selected_window_index": SELECTED_WINDOW_INDEX,
                "windows": [
                    {
                        "id": SELECTED_WINDOW_ID,
                        "target_id": SELECTED_TARGET_ID,
                        "source_timestamps": "late_bound_from_sealed_FULL24_window[0].source_timestamps",
                        "observation_timestamp": "late_bound_from_sealed_FULL24_window[0].observation_timestamp",
                    }
                ],
                "timestamp_binding": {
                    "source": "sealed FULL24 TRAIN window index 0",
                    "horizon": "H40",
                    "frequency_hz": 20,
                    "required_source_timestamps": True,
                },
            },
            "contract": full_contract,
            "sampler": {
                **full_sampler,
                "scope": "one_selected_window",
                "windows_per_checkpoint": WINDOW_COUNT,
                "planned_forwards_total": EXPECTED_MODEL_FORWARDS,
                "expected_total_sampler_calls": EXPECTED_SAMPLER_CALLS,
            },
            "runtime_binding": full_runtime,
            "output": {
                "raw_before_derived": True,
                "raw_shape": list(RAW_SHAPE),
                "evidence_bundle": "generic engine raw seal plus CPU-derived evaluator inventory",
            },
        },
        "identity_anchors": {
            "checkpoint_tree_sha256": FULL24_2000_TREE_SHA256,
            "dataset_tree_sha256": FULL24_DATASET_TREE_SHA256,
            "split_lock_sha256": FULL24_SPLIT_SHA256,
            "stats_sha256": FULL24_STATS_SHA256,
            "stats_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "sampler_profile": SAMPLER_PROFILE,
            "sampler_protocol_revision": SAMPLER_PROTOCOL_REVISION,
            "sampler_law": SAMPLER_LAW,
            "timesteps": list(TIMESTEPS),
            "dt": SAMPLER_DT,
            "noise_seed": 4242,
        },
        "execution_boundary": {
            "model_import": False,
            "model_construction": False,
            "model_forward": False,
            "training": False,
            "expected_model_forwards": EXPECTED_MODEL_FORWARDS,
            "expected_sampler_calls": EXPECTED_SAMPLER_CALLS,
            "pbs_submissions": 0,
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
        },
        "evidence_boundary": "static PI0102 one-window compatibility fixture only; no H200 or model execution",
    }
    validate_pi0102_one_window_fixture(fixture)
    return fixture


def validate_pi0102_one_window_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate a detached PI0102 fixture without model imports."""

    _exact(fixture.get("schema"), PI0102_ONE_WINDOW_FIXTURE_SCHEMA, field="fixture.schema")
    _exact(fixture.get("scope"), PI0102_ONE_WINDOW_SCOPE, field="fixture.scope")
    _exact(fixture.get("status"), PI0102_FIXTURE_STATUS, field="fixture.status")
    _exact(fixture.get("executable"), False, field="fixture.executable")
    source = _mapping(fixture.get("source_protocol"), field="fixture.source_protocol")
    _exact(source.get("scope"), STAGE45C_COMPATIBILITY_SCOPE, field="source_protocol.scope")
    selection = _mapping(fixture.get("selection"), field="fixture.selection")
    for field, expected in (
        ("checkpoint_count", CHECKPOINT_COUNT),
        ("window_count", WINDOW_COUNT),
        ("selected_window_index", SELECTED_WINDOW_INDEX),
        ("expected_model_forwards", EXPECTED_MODEL_FORWARDS),
        ("expected_sampler_calls", EXPECTED_SAMPLER_CALLS),
    ):
        _exact(selection.get(field), expected, field=f"selection.{field}")
    _exact(selection.get("checkpoint_label"), CHECKPOINT_LABEL, field="selection.checkpoint_label")
    _exact(selection.get("selected_window_id"), SELECTED_WINDOW_ID, field="selection.selected_window_id")
    _exact(selection.get("selected_target_id"), SELECTED_TARGET_ID, field="selection.selected_target_id")
    _exact(selection.get("full_cohort_window_count"), 24, field="selection.full_cohort_window_count")
    _exact(selection.get("full_cohort_window_order"), WINDOW_ORDER, field="selection.full_cohort_window_order")

    projection = _mapping(
        fixture.get("generic_manifest_projection"),
        field="fixture.generic_manifest_projection",
    )
    checkpoints = projection.get("checkpoints")
    _require(isinstance(checkpoints, list) and len(checkpoints) == CHECKPOINT_COUNT, "fixture must contain one checkpoint")
    _exact(checkpoints[0].get("label"), CHECKPOINT_LABEL, field="projection.checkpoints[0].label")
    _exact(checkpoints[0].get("count"), CHECKPOINT_COUNT, field="projection.checkpoints[0].count")
    cohort = _mapping(projection.get("cohort"), field="projection.cohort")
    _exact(cohort.get("window_count"), WINDOW_COUNT, field="projection.cohort.window_count")
    _exact(cohort.get("selected_window_index"), SELECTED_WINDOW_INDEX, field="projection.cohort.selected_window_index")
    windows = cohort.get("windows")
    _require(isinstance(windows, list) and len(windows) == WINDOW_COUNT, "fixture must contain one explicit window")
    _exact(windows[0].get("id"), SELECTED_WINDOW_ID, field="projection.cohort.windows[0].id")
    _exact(windows[0].get("target_id"), SELECTED_TARGET_ID, field="projection.cohort.windows[0].target_id")
    _exact(
        windows[0].get("source_timestamps"),
        "late_bound_from_sealed_FULL24_window[0].source_timestamps",
        field="projection.cohort.windows[0].source_timestamps",
    )
    sampler = _mapping(projection.get("sampler"), field="projection.sampler")
    for field, expected in (
        ("profile", SAMPLER_PROFILE),
        ("protocol_revision", SAMPLER_PROTOCOL_REVISION),
        ("sampler", SAMPLER_LAW),
        ("timesteps", list(TIMESTEPS)),
        ("dt", SAMPLER_DT),
        ("expected_calls_per_window", SAMPLER_CALLS),
        ("noise_seed", 4242),
        ("windows_per_checkpoint", WINDOW_COUNT),
        ("planned_forwards_total", EXPECTED_MODEL_FORWARDS),
        ("expected_total_sampler_calls", EXPECTED_SAMPLER_CALLS),
    ):
        _exact(sampler.get(field), expected, field=f"projection.sampler.{field}")
    contract = _mapping(projection.get("contract"), field="projection.contract")
    _exact(contract.get("structured_action_schema"), "StructuredActionV1", field="projection.contract.schema")
    _exact(contract.get("raw_action_shape"), list(RAW_SHAPE), field="projection.contract.raw_shape")
    _exact(contract.get("decoded_action_shape"), list(DECODED_SHAPE), field="projection.contract.decoded_shape")
    _exact(contract.get("state_v1_shape"), list(STATE_V1_SHAPE), field="projection.contract.state_v1_shape")
    _exact(contract.get("geometry"), "K16", field="projection.contract.geometry")
    _exact(contract.get("timing"), "T16", field="projection.contract.timing")
    _exact(contract.get("horizon"), "H40", field="projection.contract.horizon")
    _exact(contract.get("source_frame"), SOURCE_FRAME, field="projection.contract.source_frame")
    _exact(contract.get("tcp_semantic_id"), TCP_SEMANTIC_ID, field="projection.contract.tcp_semantic_id")
    _exact(contract.get("model_input_keys"), list(MODEL_INPUT_KEYS), field="projection.contract.model_input_keys")
    _exact(contract.get("language"), LANGUAGE, field="projection.contract.language")
    execution = _mapping(fixture.get("execution_boundary"), field="fixture.execution_boundary")
    for field in ("model_import", "model_construction", "model_forward", "training"):
        _exact(execution.get(field), False, field=f"execution_boundary.{field}")
    _exact(execution.get("pbs_submissions"), 0, field="execution_boundary.pbs_submissions")


__all__ = [
    "CHECKPOINT_COUNT",
    "EXPECTED_MODEL_FORWARDS",
    "EXPECTED_SAMPLER_CALLS",
    "PI0102_FIXTURE_STATUS",
    "PI0102_ONE_WINDOW_FIXTURE_SCHEMA",
    "PI0102_ONE_WINDOW_SCOPE",
    "PI0102QualificationFixtureError",
    "SELECTED_TARGET_ID",
    "SELECTED_WINDOW_ID",
    "SELECTED_WINDOW_INDEX",
    "WINDOW_COUNT",
    "build_pi0102_one_window_fixture",
    "validate_pi0102_one_window_fixture",
]
