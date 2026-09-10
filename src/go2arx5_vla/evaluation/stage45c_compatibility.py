"""Static compatibility proof for the frozen Stage45C evaluation contract.

This module is intentionally separate from the generic engine.  It translates
the historical, already-frozen protocol into the generic manifest fields and
checks that no scientific field is lost.  It never resolves an adapter,
imports a model framework, reads a runtime artifact, or executes a forward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


STAGE45C_COMPATIBILITY_PROOF_SCHEMA = (
    "GO2ARX5_REUSABLE_STAGE45C_COMPATIBILITY_PROOF_V1"
)
STAGE45C_COMPATIBILITY_SCOPE = "STAGE45C_FULL_TRAIN24_PROTOCOL"
FROZEN_PROTOCOL_SCHEMA = "CORELAB_STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1"
FROZEN_PROTOCOL_STATUS = "BOUND_NOT_EXECUTED"

PROTOCOL_SHA256 = "0f8c564f534ce3b29a5029cc927f93a03697181b4b1df0aa9e80566af4cd020f"
PROTOCOL_SOURCE_HEAD = "54a833eaa7037cdc20397ef182b36647194040cb"
FULL24_2000_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
FULL24_DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_SPLIT_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_SOURCE_COMMIT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_SOURCE_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
DECODER_SOURCE_SHA256 = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"

CHECKPOINT_LABEL = "FULL24@2000"
COHORT_NAME = "FULL24_EXISTING_TRAIN"
COHORT_SPLIT = "TRAIN"
WINDOW_ORDER = "target_major_then_snapshot_major"
TARGET_ORDER = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TARGET_ORDER
    for index in range(6)
)
SAMPLER_PROFILE = "GROOT_N1D7_SOURCE_MAPPED_EULER4_V1"
SAMPLER_PROTOCOL_REVISION = "RUNTIME_PARQUET_PROVENANCE_V2"
SAMPLER_LAW = "source-mapped-four-step-Euler-action-head-graph"
TIMESTEPS = (0, 250, 500, 750)
SAMPLER_DT = 0.25
SAMPLER_CALLS = 4
SAMPLER_CHECKPOINT_ORDER = ("BASE", "STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
SAMPLER_WINDOWS_PER_CHECKPOINT = 24
SAMPLER_PLANNED_FORWARDS = 96
NOISE_SEED = 4242
RAW_SHAPE = (1, 40, 132)
DECODED_SHAPE = (40, 10)
STATE_V1_SHAPE = (10,)
MODEL_INPUT_KEYS = ("body_rgb", "wrist_rgb", "state")
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
FULL24_FRAME_COUNT = 960
LANGUAGE = "move the end effector to the red target"


class Stage45CCompatibilityError(ValueError):
    """Raised when the historical protocol cannot be expressed losslessly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage45CCompatibilityError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _strict_integer(value: Any, *, field: str) -> int:
    _require(type(value) is int, f"{field} must be a JSON integer")
    return value


def _strict_number(value: Any, *, field: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{field} must be an explicit finite JSON number",
    )
    return float(value)


def _exact(actual: Any, expected: Any, *, field: str) -> None:
    _require(actual == expected, f"{field} differs from the frozen Stage45C contract")


def _window_ids(value: Any, *, field: str) -> list[str]:
    _require(isinstance(value, list), f"{field} must be an explicit list")
    _require(
        all(type(item) is str and bool(item.strip()) for item in value),
        f"{field} must contain only non-empty JSON strings",
    )
    result = list(value)
    _exact(result, list(WINDOW_IDS), field=field)
    return result


def _check_protocol(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    _exact(protocol.get("schema"), FROZEN_PROTOCOL_SCHEMA, field="protocol.schema")
    _exact(protocol.get("status"), FROZEN_PROTOCOL_STATUS, field="protocol.status")
    _exact(protocol.get("scientific_contract_changed"), False, field="protocol.scientific_contract_changed")
    cohort = _mapping(protocol.get("cohort"), field="protocol.cohort")
    _exact(cohort.get("name"), COHORT_NAME, field="protocol.cohort.name")
    _exact(cohort.get("split"), COHORT_SPLIT, field="protocol.cohort.split")
    _exact(cohort.get("expected_window_count"), len(WINDOW_IDS), field="protocol.cohort.expected_window_count")
    _exact(cohort.get("window_order"), WINDOW_ORDER, field="protocol.cohort.window_order")
    _window_ids(cohort.get("window_ids"), field="protocol.cohort.window_ids")

    contract = _mapping(protocol.get("structured_action_contract"), field="protocol.structured_action_contract")
    _exact(contract.get("schema"), "StructuredActionV1", field="contract.schema")
    _exact(contract.get("raw_shape"), list(RAW_SHAPE), field="contract.raw_shape")
    _exact(contract.get("decoded_shape"), list(DECODED_SHAPE), field="contract.decoded_shape")
    _exact(contract.get("state_v1_shape"), list(STATE_V1_SHAPE), field="contract.state_v1_shape")
    _exact(contract.get("geometry"), "K16", field="contract.geometry")
    _exact(contract.get("timing"), "T16", field="contract.timing")
    _exact(contract.get("horizon"), "H40", field="contract.horizon")
    _exact(contract.get("model_input_keys"), list(MODEL_INPUT_KEYS), field="contract.model_input_keys")
    _exact(contract.get("target_xyz_in_model_input"), False, field="contract.target_xyz_in_model_input")
    _exact(contract.get("normalization"), "runtime_v2_stats_only", field="contract.normalization")
    _exact(contract.get("source_frame"), SOURCE_FRAME, field="contract.source_frame")
    _exact(contract.get("tcp_semantic_id"), TCP_SEMANTIC_ID, field="contract.tcp_semantic_id")
    _exact(contract.get("decoder_source_sha256"), DECODER_SOURCE_SHA256, field="contract.decoder_source_sha256")

    sampler = _mapping(protocol.get("sampler_noise_protocol"), field="protocol.sampler_noise_protocol")
    _exact(sampler.get("adaptive_extension"), False, field="sampler.adaptive_extension")
    _exact(sampler.get("checkpoint_order"), list(SAMPLER_CHECKPOINT_ORDER), field="sampler.checkpoint_order")
    _exact(sampler.get("development_seed"), NOISE_SEED, field="sampler.development_seed")
    _exact(sampler.get("noise_shape"), list(RAW_SHAPE), field="sampler.noise_shape")
    _exact(sampler.get("same_noise_across_checkpoints"), True, field="sampler.same_noise_across_checkpoints")
    _exact(sampler.get("sampler"), SAMPLER_LAW, field="sampler.sampler")
    _exact(sampler.get("steps"), SAMPLER_CALLS, field="sampler.steps")
    _exact(sampler.get("dt"), SAMPLER_DT, field="sampler.dt")
    _exact(sampler.get("timestep_ids"), list(TIMESTEPS), field="sampler.timestep_ids")
    _exact(sampler.get("windows_per_checkpoint"), SAMPLER_WINDOWS_PER_CHECKPOINT, field="sampler.windows_per_checkpoint")
    _exact(sampler.get("planned_forwards_total"), SAMPLER_PLANNED_FORWARDS, field="sampler.planned_forwards_total")
    return cohort, contract, sampler


def project_stage45c_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Project one concrete historical window without losing timestamps."""

    item = _mapping(window, field="window")
    _exact(item.get("id"), WINDOW_IDS[0], field="window.id")
    _exact(item.get("target_id"), TARGET_ORDER[0], field="window.target_id")
    timestamps_value = item.get("source_timestamps")
    _require(isinstance(timestamps_value, list), "window.source_timestamps must be an explicit list")
    _require(len(timestamps_value) == RAW_SHAPE[1], "window.source_timestamps must contain H40 values")
    timestamps = [
        _strict_number(value, field=f"window.source_timestamps[{index}]")
        for index, value in enumerate(timestamps_value)
    ]
    observation = _strict_number(
        item.get("observation_timestamp"), field="window.observation_timestamp"
    )
    for index, value in enumerate(timestamps):
        _require(
            abs(value - (timestamps[0] + index * 0.05)) <= 1e-9,
            "window.source_timestamps are not the frozen 20Hz H40 sequence",
        )
    _require(
        abs(timestamps[0] - observation) <= 1e-7,
        "window observation timestamp differs from its source timestamp anchor",
    )
    return {
        "id": item["id"],
        "target_id": item["target_id"],
        "source_timestamps": timestamps,
        "observation_timestamp": observation,
        "timestamp_binding": {
            "source": "explicit historical TRAIN24 window fields",
            "horizon": "H40",
            "frequency_hz": 20,
            "delta_seconds": 0.05,
        },
    }


def build_stage45c_compatibility_proof(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Build the static exact-compatibility projection from frozen protocol JSON."""

    protocol = _mapping(protocol, field="protocol")
    cohort, contract, sampler = _check_protocol(protocol)
    checkpoints = protocol.get("checkpoints")
    _require(isinstance(checkpoints, list), "protocol.checkpoints must be an explicit list")
    checkpoint_by_label = {
        str(item.get("label")): _mapping(item, field="protocol.checkpoints[]")
        for item in checkpoints
    }
    checkpoint = _mapping(
        checkpoint_by_label.get(CHECKPOINT_LABEL),
        field="protocol.checkpoints[FULL24@2000]",
    )
    _exact(
        checkpoint.get("tree_sha256"),
        FULL24_2000_TREE_SHA256,
        field="protocol.checkpoints[FULL24@2000].tree_sha256",
    )
    full24 = _mapping(protocol.get("full24_identity"), field="protocol.full24_identity")
    _exact(full24.get("tree_sha256"), FULL24_DATASET_TREE_SHA256, field="full24_identity.tree_sha256")
    _exact(full24.get("split_lock_sha256"), FULL24_SPLIT_SHA256, field="full24_identity.split_lock_sha256")
    _exact(full24.get("frame_count"), FULL24_FRAME_COUNT, field="full24_identity.frame_count")
    _exact(full24.get("window_count"), len(WINDOW_IDS), field="full24_identity.window_count")
    _exact(full24.get("window_ids"), list(WINDOW_IDS), field="full24_identity.window_ids")
    _exact(full24.get("window_order"), WINDOW_ORDER, field="full24_identity.window_order")
    _exact(full24.get("target_order"), list(TARGET_ORDER), field="full24_identity.target_order")
    runtime_v2 = _mapping(protocol.get("runtime_binding"), field="protocol.runtime_binding")
    runtime_v2_stats = _mapping(runtime_v2.get("structured_statistics"), field="runtime_binding.structured_statistics")
    _exact(runtime_v2_stats.get("protocol_revision"), SAMPLER_PROTOCOL_REVISION, field="runtime_v2.protocol_revision")
    _exact(runtime_v2_stats.get("sha256"), FULL24_STATS_SHA256, field="runtime_v2.stats_sha256")
    _exact(runtime_v2_stats.get("payload_sha256"), FULL24_STATS_PAYLOAD_SHA256, field="runtime_v2.payload_sha256")
    groot = _mapping(
        runtime_v2.get("groot_runtime_manifest"),
        field="runtime_binding.groot_runtime_manifest",
    )
    runtime_admission = _mapping(
        runtime_v2.get("runtime_admission"),
        field="runtime_binding.runtime_admission",
    )
    _exact(groot.get("source_commit"), GROOT_SOURCE_COMMIT_SHA, field="groot_runtime_manifest.source_commit")
    _exact(groot.get("payload_tree_sha256"), GROOT_SOURCE_TREE_SHA256, field="groot_runtime_manifest.payload_tree_sha256")
    _exact(runtime_admission.get("cosmos_revision"), COSMOS_REVISION, field="runtime_admission.cosmos_revision")
    _exact(runtime_admission.get("cosmos_tree_sha256"), COSMOS_TREE_SHA256, field="runtime_admission.cosmos_tree_sha256")
    _exact(runtime_admission.get("modality_sha256"), MODALITY_SHA256, field="runtime_admission.modality_sha256")

    locked = _mapping(protocol.get("locked_inputs"), field="protocol.locked_inputs")
    for field in ("Validation-E", "Holdout-F", "WBC", "hardware"):
        _exact(locked.get(field), "LOCKED", field=f"locked_inputs.{field}")
    for field in ("simulator", "new_data"):
        if field in locked:
            _exact(locked.get(field), "LOCKED", field=f"locked_inputs.{field}")

    projection = {
        "schema": STAGE45C_COMPATIBILITY_PROOF_SCHEMA,
        "scope": STAGE45C_COMPATIBILITY_SCOPE,
        "status": "STATIC_COMPATIBLE_NOT_EXECUTED",
        "executable": False,
        "historical_protocol": {
            "schema": protocol["schema"],
            "status": protocol["status"],
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_source_head": PROTOCOL_SOURCE_HEAD,
        },
        "generic_manifest_projection": {
            "checkpoints": [
                {
                    "label": CHECKPOINT_LABEL,
                    "count": 1,
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
                "name": COHORT_NAME,
                "split": COHORT_SPLIT,
                "frame_count": FULL24_FRAME_COUNT,
                "window_count": len(WINDOW_IDS),
                "window_order": WINDOW_ORDER,
                "window_ids": list(WINDOW_IDS),
                "target_order": list(TARGET_ORDER),
                "window_records": "explicit manifest cohort.windows entries; no inferred count/order",
                "timestamp_fields": {
                    "source_timestamps": "cohort.windows[*].source_timestamps",
                    "observation_timestamp": "cohort.windows[*].observation_timestamp",
                    "required_shape": [RAW_SHAPE[1]],
                    "frequency_hz": 20,
                },
            },
            "contract": {
                "structured_action_schema": "StructuredActionV1",
                "raw_action_shape": list(RAW_SHAPE),
                "decoded_action_shape": list(DECODED_SHAPE),
                "state_v1_shape": list(STATE_V1_SHAPE),
                "geometry": "K16",
                "timing": "T16",
                "horizon": "H40",
                "model_input_keys": list(MODEL_INPUT_KEYS),
                "target_xyz_in_model_input": False,
                "language": LANGUAGE,
                "normalization": "runtime_v2_stats_only",
                "source_frame": SOURCE_FRAME,
                "tcp_semantic_id": TCP_SEMANTIC_ID,
                "decoder_source_sha256": DECODER_SOURCE_SHA256,
            },
            "sampler": {
                "profile": SAMPLER_PROFILE,
                "protocol_revision": SAMPLER_PROTOCOL_REVISION,
                "sampler": SAMPLER_LAW,
                "timesteps": list(TIMESTEPS),
                "dt": SAMPLER_DT,
                "expected_calls_per_window": SAMPLER_CALLS,
                "noise_seed": NOISE_SEED,
                "noise_shape": list(RAW_SHAPE),
                "same_noise_across_checkpoints": True,
                "adaptive_extension": False,
                "windows_per_checkpoint": SAMPLER_WINDOWS_PER_CHECKPOINT,
                "planned_forwards_total": SAMPLER_PLANNED_FORWARDS,
            },
            "runtime_binding": {
                "dataset_tree_sha256": FULL24_DATASET_TREE_SHA256,
                "split_lock_sha256": FULL24_SPLIT_SHA256,
                "groot_source_commit_sha": GROOT_SOURCE_COMMIT_SHA,
                "groot_source_tree_sha256": GROOT_SOURCE_TREE_SHA256,
                "cosmos_revision": COSMOS_REVISION,
                "cosmos_tree_sha256": COSMOS_TREE_SHA256,
                "modality_sha256": MODALITY_SHA256,
                "runtime_v2_protocol_revision": SAMPLER_PROTOCOL_REVISION,
                "language": LANGUAGE,
                "paths": "all materialized paths are late-bound through the admission-owned runtime binding",
            },
            "output": {
                "raw_before_derived": True,
                "evidence_bundle": "generic engine raw seal plus CPU-derived evaluator inventory",
            },
        },
        "identity_anchors": {
            "full24_2000_tree_sha256": FULL24_2000_TREE_SHA256,
            "full24_dataset_tree_sha256": FULL24_DATASET_TREE_SHA256,
            "full24_split_sha256": FULL24_SPLIT_SHA256,
            "runtime_v2_stats_sha256": FULL24_STATS_SHA256,
            "runtime_v2_stats_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "groot_source_commit_sha": GROOT_SOURCE_COMMIT_SHA,
            "groot_source_tree_sha256": GROOT_SOURCE_TREE_SHA256,
            "cosmos_revision": COSMOS_REVISION,
            "cosmos_tree_sha256": COSMOS_TREE_SHA256,
            "modality_sha256": MODALITY_SHA256,
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
        },
        "execution_boundary": {
            "model_import": False,
            "model_construction": False,
            "model_forward": False,
            "training": False,
            "pbs_submissions": 0,
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
        },
        "evidence_boundary": "static exact compatibility projection only; no historical scientific result is rerun",
    }
    _validate_projection(projection)
    return projection


def _validate_projection(proof: Mapping[str, Any]) -> None:
    _exact(proof.get("schema"), STAGE45C_COMPATIBILITY_PROOF_SCHEMA, field="proof.schema")
    _exact(proof.get("scope"), STAGE45C_COMPATIBILITY_SCOPE, field="proof.scope")
    _exact(proof.get("status"), "STATIC_COMPATIBLE_NOT_EXECUTED", field="proof.status")
    _exact(proof.get("executable"), False, field="proof.executable")
    projection_value = proof.get("generic_manifest_projection")
    projection_field = "proof.generic_manifest_projection"
    if projection_value is None:
        projection_value = proof.get("generic_manifest_expression")
        projection_field = "proof.generic_manifest_expression"
    projection = _mapping(projection_value, field=projection_field)
    checkpoints = projection.get("checkpoints")
    if checkpoints is None and projection.get("checkpoint") is not None:
        checkpoints = [projection.get("checkpoint")]
    _require(isinstance(checkpoints, list) and len(checkpoints) == 1, "proof must express exactly one checkpoint")
    checkpoint = _mapping(checkpoints[0], field="proof.checkpoints[0]")
    _exact(checkpoint.get("label"), CHECKPOINT_LABEL, field="proof.checkpoints[0].label")
    _exact(projection["cohort"].get("window_count"), 24, field="proof.cohort.window_count")
    _exact(projection["cohort"].get("frame_count"), FULL24_FRAME_COUNT, field="proof.cohort.frame_count")
    _exact(projection["cohort"].get("window_order"), WINDOW_ORDER, field="proof.cohort.window_order")
    _exact(projection["sampler"].get("timesteps"), list(TIMESTEPS), field="proof.sampler.timesteps")
    _exact(projection["sampler"].get("protocol_revision"), SAMPLER_PROTOCOL_REVISION, field="proof.sampler.protocol_revision")
    _exact(projection["sampler"].get("sampler"), SAMPLER_LAW, field="proof.sampler.sampler")
    _exact(projection["sampler"].get("expected_calls_per_window"), SAMPLER_CALLS, field="proof.sampler.expected_calls_per_window")
    _exact(projection["sampler"].get("windows_per_checkpoint"), SAMPLER_WINDOWS_PER_CHECKPOINT, field="proof.sampler.windows_per_checkpoint")
    _exact(projection["sampler"].get("planned_forwards_total"), SAMPLER_PLANNED_FORWARDS, field="proof.sampler.planned_forwards_total")
    _exact(projection["contract"].get("raw_action_shape"), list(RAW_SHAPE), field="proof.contract.raw_action_shape")
    _exact(projection["contract"].get("decoded_action_shape"), list(DECODED_SHAPE), field="proof.contract.decoded_action_shape")
    _exact(projection["contract"].get("geometry"), "K16", field="proof.contract.geometry")
    _exact(projection["contract"].get("timing"), "T16", field="proof.contract.timing")
    _exact(projection["contract"].get("source_frame"), SOURCE_FRAME, field="proof.contract.source_frame")
    _exact(projection["contract"].get("tcp_semantic_id"), TCP_SEMANTIC_ID, field="proof.contract.tcp_semantic_id")
    _exact(proof["execution_boundary"].get("model_import"), False, field="proof.execution_boundary.model_import")


def validate_stage45c_compatibility_proof(proof: Mapping[str, Any]) -> None:
    """Validate a detached proof document without importing model code."""

    _validate_projection(_mapping(proof, field="proof"))


__all__ = [
    "CHECKPOINT_LABEL",
    "DECODED_SHAPE",
    "FULL24_2000_TREE_SHA256",
    "FULL24_DATASET_TREE_SHA256",
    "FULL24_SPLIT_SHA256",
    "FULL24_STATS_PAYLOAD_SHA256",
    "FULL24_STATS_SHA256",
    "MODEL_INPUT_KEYS",
    "RAW_SHAPE",
    "SAMPLER_CALLS",
    "SAMPLER_DT",
    "SAMPLER_LAW",
    "SAMPLER_PROTOCOL_REVISION",
    "SAMPLER_PROFILE",
    "STAGE45C_COMPATIBILITY_SCOPE",
    "Stage45CCompatibilityError",
    "TIMESTEPS",
    "WINDOW_IDS",
    "build_stage45c_compatibility_proof",
    "project_stage45c_window",
    "validate_stage45c_compatibility_proof",
]
