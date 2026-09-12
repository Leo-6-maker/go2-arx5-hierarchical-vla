#!/usr/bin/env python3
"""CPU-only ContextV1 builder and leakage/causality verifier for PI0145.

The module deliberately stops at a pathless, immutable context artifact.  It
does not import a model framework, construct a model, read a checkpoint, or
perform inference.  Context records are validated before they can be sorted
into an index; unknown fields and missing-vs-null distinctions fail closed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import stage45f_corelab_evidence_convergence as convergence


REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "manifests/ICL_CONTEXTV1_CONTRACT.json"
CAPABILITY_PATH = REPO / "manifests/STAGE45F_GR00T_CONTEXT_CAPABILITY_AUDIT_V1.json"
OPTIONS_PATH = REPO / "manifests/STAGE45F_ICL_INTEGRATION_OPTIONS_V1.json"
LADDER_PATH = REPO / "manifests/STAGE45F_ICL_EXPERIMENT_LADDER_V1.json"
RISK_PATH = REPO / "manifests/STAGE45F_ICL_COMPUTE_RISK_REGISTER_V1.json"
TENSOR_PLAN_PATH = REPO / "manifests/ICL_CONTEXTTENSORV1_PLAN.json"
PARITY_PATH = REPO / "manifests/STAGE45F_ICL_C1_PARITY_COUNTEREXAMPLE_V1.json"

CONTEXT_SCHEMA = "ICL_CONTEXTV1_RECORD_V1"
INDEX_SCHEMA = "ICL_CONTEXTV1_INDEX_V1"
CONTEXT_VERSION = "1.1"
BUILDER_IDENTITY = "STAGE45F_ICL_CONTEXTV1_CPU_BUILDER_V1"
UPSTREAM_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
ZERO_SHA256 = "0" * 64
STRUCTURED_SHAPE = [40, 132]
ACTIVE_SCALARS = 190
TIMING_FIELDS = ("dt",)
OUTCOME_LABELS = (
    "no_progress_observed",
    "progress_observed",
    "recovered",
    "unrecovered",
    "unknown",
)
CAUSAL_ORDERING_LAW = "event_time ascending, then available_time ascending, then model-visible payload identity"
LOCAL_DYNAMICS_CLOSURE_LAW = "event_time + dt <= available_time <= query_time; state_after/delta are closed by available_time"

SECTION_FIELDS = {
    "task_demo_context": (
        "demo_id", "demo_source_sha256", "language", "observation_digest",
        "commanded_action_digest", "demo_time_range",
    ),
    "interaction_history": (
        "event_time", "available_time", "observation_state_digest", "commanded_action_digest",
        "executed_or_observed_response_digest", "delta_state_digest", "timing",
    ),
    "failure_outcome_context": (
        "attempt_id", "failure_observation_digest", "outcome_label",
        "observed_recovery_digest", "event_time", "available_time",
    ),
    "local_dynamics_context": (
        "event_time", "available_time", "state_before_digest", "commanded_action_digest",
        "executed_or_observed_response_digest", "state_after_digest",
        "delta_state_digest", "dt",
    ),
}
SECTIONS = tuple(SECTION_FIELDS)
RECORD_REQUIRED_FIELDS = {
    "schema", "version", "context_id", "query_id", "query_time",
    "builder_identity", "upstream", "source_identity", *SECTIONS,
}
RECORD_OPTIONAL_FIELDS = {"window_id", "target_id", "notes"}
SOURCE_FIELDS = {"artifact_id", "sha256"}
UPSTREAM_FIELDS = {"project", "model", "commit"}
METADATA_ONLY_PATHS = {
    "context_id", "query_id", "window_id", "target_id", "builder_identity",
    "upstream", "source_identity", "task_demo_context.demo_id",
    "task_demo_context.demo_source_sha256", "task_demo_context.demo_time_range",
    "interaction_history.event_time", "interaction_history.available_time",
    "failure_outcome_context.attempt_id", "failure_outcome_context.event_time",
    "failure_outcome_context.available_time", "local_dynamics_context.event_time",
    "local_dynamics_context.available_time",
}
MODEL_VISIBLE_PATHS = {
    "task_demo_context.language",
    "task_demo_context.observation_payload_resolved",
    "task_demo_context.commanded_action_payload_resolved",
    "interaction_history.observation_state_payload_resolved",
    "interaction_history.commanded_action_payload_resolved",
    "interaction_history.executed_or_observed_response_payload_resolved",
    "interaction_history.delta_state_payload_resolved",
    "interaction_history.timing_payload",
    "failure_outcome_context.failure_observation_payload_resolved",
    "failure_outcome_context.outcome_label",
    "failure_outcome_context.observed_recovery_payload_resolved",
    "local_dynamics_context.state_before_payload_resolved",
    "local_dynamics_context.commanded_action_payload_resolved",
    "local_dynamics_context.executed_or_observed_response_payload_resolved",
    "local_dynamics_context.state_after_payload_resolved",
    "local_dynamics_context.delta_state_payload_resolved",
    "local_dynamics_context.dt_payload",
}

# These are semantic traps, rather than a general-purpose natural-language
# filter.  They are checked on keys and scalar values; an ordinary language
# field remains usable unless it explicitly carries one of the forbidden
# channels.
FORBIDDEN_KEY_FRAGMENTS = (
    "target_xyz", "validation_e", "holdout_f", "future_observation",
    "future_action", "future_expert", "expert_action", "expert_label",
    "absolute_path", "filesystem_path", "source_path", "checkpoint_path",
    "artifact_path", "cwd", "worktree", "scratch_path",
)
FORBIDDEN_VALUE_FRAGMENTS = (
    "validation_e", "holdout_f", "target_xyz",
    "future_expert", "future_action", "future_observation", "expert_action",
    "expert_label",
)
MODEL_VISIBLE_METADATA_FRAGMENTS = (
    "target_id", "target_xyz", "query_time", "source_id", "source_digest",
    "source_sha", "source_path", "demo_id", "attempt_id", "provenance",
    "checkpoint", "notes", "validation_e", "holdout_f", "locked_split",
)
PATHLIKE_KEY_FRAGMENTS = ("path", "file", "directory", "dirname")
REASON_CODES = {
    "SCHEMA_DRIFT", "UNKNOWN_FIELD", "MISSING_REQUIRED_FIELD",
    "INVALID_SHA256", "INVALID_GIT_SHA", "INVALID_IDENTIFIER",
    "CAUSAL_TIMESTAMP_VIOLATION", "INVALID_TIMESTAMP", "NONFINITE_VALUE",
    "MISSING_EXPLICIT_NULL_RESPONSE", "FABRICATED_RESPONSE",
    "FORBIDDEN_TARGET_XYZ", "LOCKED_SPLIT_CONTAMINATION",
    "FUTURE_EXPERT_INFORMATION", "PATH_LEAKAGE", "SOURCE_IDENTITY_MISSING",
    "DYNAMICS_TRACE_MISSING", "BUILDER_IDENTITY_DRIFT", "ABI_DRIFT",
    "DUPLICATE_QUERY_ID", "CONTEXT_ORDER_DRIFT", "UPSTREAM_IDENTITY_DRIFT",
    "EXECUTION_BOUNDARY_DRIFT", "STATIC_IMPORT_VIOLATION",
    "AVAILABILITY_TIME_MISSING", "MODEL_VISIBILITY_DRIFT", "TENSOR_PLAN_DRIFT",
}
FORBIDDEN_IMPORTS = {
    "torch", "numpy", "transformers", "gr00t", "subprocess", "jax",
    "tensorflow", "pandas", "isaacgym", "omni",
}


class ContextV1Error(convergence.ConvergenceError):
    """Fail-closed context error with a stable machine-readable code."""

    def __init__(self, reason_code: str, detail: str):
        if reason_code not in REASON_CODES:
            raise ValueError(f"unregistered ContextV1 reason code: {reason_code}")
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def reject(reason_code: str, detail: str) -> None:
    raise ContextV1Error(reason_code, detail)


def demand(condition: bool, reason_code: str, detail: str) -> None:
    if not condition:
        reject(reason_code, detail)


def _read_json(path: Path) -> Any:
    try:
        return convergence.read_json(Path(path).resolve())
    except convergence.ConvergenceError as exc:
        reject("SCHEMA_DRIFT", str(exc))


def _write_json(path: Path, value: Any) -> None:
    convergence.write_json(Path(path).resolve(), value)


def _sha256(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    demand(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value),
        "INVALID_SHA256",
        f"{field} must be a lowercase SHA256",
    )
    return value


def _git_sha(value: Any, field: str) -> str:
    demand(
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value),
        "INVALID_GIT_SHA",
        f"{field} must be a lowercase Git SHA",
    )
    return value


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    demand(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        "NONFINITE_VALUE",
        f"{field} must be finite numeric",
    )
    return float(value)


def _identifier(value: Any, field: str) -> str:
    demand(isinstance(value, str) and bool(value), "INVALID_IDENTIFIER", f"{field} missing")
    # IDs are intentionally pathless.  This also rejects URI and traversal
    # forms without making assumptions about the ID namespace.
    demand("/" not in value and "\\" not in value and ".." not in value, "PATH_LEAKAGE", f"{field} contains a path-like separator")
    demand(not value.startswith("."), "PATH_LEAKAGE", f"{field} looks like a relative path")
    return value


def _key_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _is_model_visible_nested(location: str) -> bool:
    """Identify nested payloads where metadata keys must be rejected.

    Top-level ``target_id``/``query_time``/source identity remain explicitly
    metadata-only.  Once a caller supplies a model-visible payload (including
    the nested timing mapping), those same names are forbidden recursively.
    """

    token = _key_token(location)
    return ".timing" in token or "payload_resolved" in token


def _scan_forbidden(value: Any, location: str = "context") -> None:
    """Reject forbidden channels before interpreting any producer fields."""
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            demand(isinstance(raw_key, str), "UNKNOWN_FIELD", f"non-string key at {location}")
            token = _key_token(raw_key)
            if any(fragment in token for fragment in FORBIDDEN_KEY_FRAGMENTS):
                code = "FORBIDDEN_TARGET_XYZ" if "target_xyz" in token else (
                    "LOCKED_SPLIT_CONTAMINATION" if token in {"validation_e", "holdout_f"} or "validation_e" in token or "holdout_f" in token else "FUTURE_EXPERT_INFORMATION"
                )
                reject(code, f"forbidden key at {location}.{raw_key}")
            if _is_model_visible_nested(location) and any(fragment in token for fragment in MODEL_VISIBLE_METADATA_FRAGMENTS):
                reject("MODEL_VISIBILITY_DRIFT", f"metadata key in model-visible payload at {location}.{raw_key}")
            if any(fragment in token for fragment in PATHLIKE_KEY_FRAGMENTS):
                # Source SHA fields are explicit immutable references, not
                # filesystem paths; all other *path/*file fields are banned.
                demand(token.endswith("_sha256") or token in {"artifact_id"}, "PATH_LEAKAGE", f"path-like key at {location}.{raw_key}")
            _scan_forbidden(item, f"{location}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_forbidden(item, f"{location}[{index}]")
        return
    if isinstance(value, str):
        normalized = _key_token(value)
        if any(fragment in normalized for fragment in FORBIDDEN_VALUE_FRAGMENTS):
            code = "FORBIDDEN_TARGET_XYZ" if "target_xyz" in normalized else (
                "LOCKED_SPLIT_CONTAMINATION" if "validation_e" in normalized or "holdout_f" in normalized else "FUTURE_EXPERT_INFORMATION"
            )
            reject(code, f"forbidden value at {location}")
        if _is_model_visible_nested(location) and any(fragment in normalized for fragment in MODEL_VISIBLE_METADATA_FRAGMENTS):
            reject("MODEL_VISIBILITY_DRIFT", f"metadata value in model-visible payload at {location}")
        demand(not value.startswith("/") and not value.startswith("file:"), "PATH_LEAKAGE", f"absolute/URI path at {location}")
        demand("../" not in value and "..\\" not in value and "\\" not in value, "PATH_LEAKAGE", f"traversal path at {location}")


def _check_exact_keys(value: Mapping[str, Any], allowed: set[str], required: set[str], location: str) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = required - keys
    demand(not unknown, "UNKNOWN_FIELD", f"unknown fields at {location}: {sorted(unknown)}")
    demand(not missing, "MISSING_REQUIRED_FIELD", f"missing fields at {location}: {sorted(missing)}")


def _check_digest(value: Any, field: str, *, nullable: bool = False) -> None:
    _sha256(value, field, nullable=nullable)


def validate_tensorization_plan(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the plan-only ContextTensorV1 visibility and tensor ABI."""
    plan = _read_json(TENSOR_PLAN_PATH) if value is None else value
    demand(isinstance(plan, Mapping), "TENSOR_PLAN_DRIFT", "tensorization plan must be an object")
    demand(plan.get("schema") == "ICL_CONTEXTTENSORV1_PLAN" and plan.get("version") == "1.0", "TENSOR_PLAN_DRIFT", "ContextTensorV1 plan schema/version drift")
    demand(plan.get("status") == "PLAN_ONLY_NO_TENSOR_MATERIALIZATION", "TENSOR_PLAN_DRIFT", "tensorization plan is executable")
    upstream = plan.get("upstream")
    demand(isinstance(upstream, Mapping) and upstream.get("commit") == UPSTREAM_COMMIT and upstream.get("project") == "isaac-gr00t" and upstream.get("model") == "GR00T N1.7", "UPSTREAM_IDENTITY_DRIFT", "tensor plan upstream drift")
    _git_sha(upstream["commit"], "tensor_plan.upstream.commit")
    resolver = plan.get("resolver")
    demand(isinstance(resolver, Mapping) and resolver.get("identity") == "STAGE45F_ICL_CONTEXTV1_IMMUTABLE_DIGEST_RESOLVER_V1", "TENSOR_PLAN_DRIFT", "resolver identity drift")
    demand(resolver.get("filesystem_paths") is False and resolver.get("checkpoint_or_locked_split_reads") is False, "PATH_LEAKAGE", "tensor resolver may read paths/locked splits")
    membership = plan.get("source_membership")
    demand(isinstance(membership, Mapping), "TENSOR_PLAN_DRIFT", "immutable TRAIN source membership contract missing")
    demand(membership.get("split") == "TRAIN24" and membership.get("source_kind") == "SEALED_TRAIN_DESCRIPTOR" and membership.get("descriptor_identity_required") is True and membership.get("immutable_membership_sha256_required") is True and membership.get("window_order") == "target-major_then_snapshot-major", "TENSOR_PLAN_DRIFT", "tensor resolver source membership drift")
    demand(plan.get("availability_source") == "source-derived; available_time is carried from immutable source and cannot be caller-overridden", "TENSOR_PLAN_DRIFT", "availability source law drift")
    metadata = plan.get("metadata_only_fields")
    visible = plan.get("model_visible_fields")
    demand(isinstance(metadata, list) and isinstance(visible, list) and len(metadata) == len(set(metadata)) and len(visible) == len(set(visible)), "MODEL_VISIBILITY_DRIFT", "visibility lists are malformed or duplicated")
    demand(set(metadata) == METADATA_ONLY_PATHS, "MODEL_VISIBILITY_DRIFT", "metadata-only whitelist drift")
    demand(set(visible) == MODEL_VISIBLE_PATHS, "MODEL_VISIBILITY_DRIFT", "model-visible whitelist drift")
    demand(not set(metadata).intersection(visible), "MODEL_VISIBILITY_DRIFT", "metadata field entered model-visible set")
    demand(not any("target_id" in str(path) or "source_identity" in str(path) or "demo_id" in str(path) or "attempt_id" in str(path) for path in visible), "MODEL_VISIBILITY_DRIFT", "metadata shortcut appears in model-visible set")
    digest_resolution = plan.get("digest_resolution")
    demand(isinstance(digest_resolution, Mapping) and all(digest_resolution.get(key) is True for key in ("record_digest_fields_are_references_not_model_inputs", "resolved_payload_must_match_referenced_sha256", "model_visible_paths_are_exact_allowlist", "target_id_and_all_provenance_are_excluded")), "MODEL_VISIBILITY_DRIFT", "digest resolution policy drift")
    layout = plan.get("tensor_layout")
    demand(isinstance(layout, Mapping) and layout.get("schema") == "ContextTensorV1" and layout.get("status") == "PLAN_ONLY", "TENSOR_PLAN_DRIFT", "ContextTensorV1 layout status drift")
    demand(layout.get("event_slots", {}).get("shape") == ["B", "N", "F"] and layout.get("event_slots", {}).get("dtype") == "float32", "TENSOR_PLAN_DRIFT", "event tensor shape/dtype drift")
    demand(layout.get("language_tokens", {}).get("shape") == ["B", "N_text"] and layout.get("language_tokens", {}).get("dtype") == "int32", "TENSOR_PLAN_DRIFT", "language tensor shape/dtype drift")
    for name, shape, dtype in (("event_mask", ["B", "N"], "bool"), ("section_mask", ["B", "N", 4], "bool"), ("missingness_mask", ["B", "N", "F"], "bool")):
        item = layout.get(name, {})
        demand(item.get("shape") == shape and item.get("dtype") == dtype, "TENSOR_PLAN_DRIFT", f"{name} shape/dtype drift")
    demand(layout.get("section_mask", {}).get("section_order") == list(SECTIONS), "TENSOR_PLAN_DRIFT", "section order drift")
    timing_schema = plan.get("nested_schemas", {}).get("interaction_history.timing")
    demand(isinstance(timing_schema, Mapping) and timing_schema.get("fields") == list(TIMING_FIELDS) and timing_schema.get("units") == "seconds" and timing_schema.get("dtype") == "float32" and timing_schema.get("missingness") == "not_allowed", "MODEL_VISIBILITY_DRIFT", "nested interaction timing schema drift")
    outcome_schema = plan.get("nested_schemas", {}).get("failure_outcome_context.outcome_label")
    demand(isinstance(outcome_schema, Mapping) and outcome_schema.get("allowed_values") == list(OUTCOME_LABELS) and outcome_schema.get("dtype") == "string", "MODEL_VISIBILITY_DRIFT", "outcome label schema drift")
    abi = plan.get("structured_action_abi")
    demand(isinstance(abi, Mapping) and abi.get("schema") == "StructuredActionV1" and abi.get("shape") == STRUCTURED_SHAPE and abi.get("active_scalars") == ACTIVE_SCALARS and abi.get("decoder_semantics") == "unchanged", "ABI_DRIFT", "tensor plan StructuredAction ABI drift")
    boundary = plan.get("execution_boundary", {})
    demand(isinstance(boundary, Mapping) and all(value == 0 for value in boundary.values()), "EXECUTION_BOUNDARY_DRIFT", "tensor plan execution boundary is nonzero")
    return {
        "schema": "ICL_CONTEXTTENSORV1_PLAN_VALIDATION_RESULT_V1",
        "status": "PASS_CPU_ONLY_PLAN",
        "resolver_identity": resolver["identity"],
        "model_visible_count": len(visible),
        "metadata_only_count": len(metadata),
        "target_id_model_visible": False,
        "pathless": True,
        "tensor_materialization": 0,
        "model_forward": 0,
    }


def build_tensorization_plan(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plan-only view; no record metadata or payload is tensorized."""
    validate_context(context)
    validation = validate_tensorization_plan()
    plan = _read_json(TENSOR_PLAN_PATH)
    return {
        "schema": "ICL_CONTEXTTENSORV1_PLAN_ONLY_RESULT_V1",
        "status": "PLAN_ONLY_NO_TENSOR_MATERIALIZATION",
        "resolver_identity": plan["resolver"]["identity"],
        "model_visible_fields": list(plan["model_visible_fields"]),
        "metadata_only_fields_excluded": list(plan["metadata_only_fields"]),
        "context_payload_materialized": False,
        "context_metadata_forwarded": False,
        "target_id_forwarded": False,
        "validation": validation,
        "execution_boundary": {"tensor_materialization": 0, "model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "simulator": 0, "WBC": 0, "hardware": 0},
    }


def c1_zero_gate_attention_counterexample() -> dict[str, Any]:
    """Return the scalar proof that zero-valued context is not parity.

    With one original key (score=0, value=1), softmax attention returns 1.
    Appending a zero-valued context key with the same score changes the
    denominator and returns 1/2.  This is a design-time arithmetic check; it
    does not construct a model or materialize any tensor.
    """

    old_denominator = math.exp(0.0)
    old_output = old_denominator * 1.0 / old_denominator
    appended_denominator = math.exp(0.0) + math.exp(0.0)
    appended_output = (math.exp(0.0) * 1.0 + math.exp(0.0) * 0.0) / appended_denominator
    demand(old_output == 1.0 and appended_output == 0.5, "TENSOR_PLAN_DRIFT", "scalar attention parity counterexample changed")
    return {
        "schema": "STAGE45F_ICL_C1_ZERO_GATE_PARITY_COUNTEREXAMPLE_V1",
        "status": "PASS_CPU_SCALAR_ONLY",
        "old_keys": 1,
        "context_keys": 1,
        "old_output": old_output,
        "zero_context_output": appended_output,
        "exact_parity_from_zero_values": False,
        "implication": "zero gate must isolate context residual or preserve original attention mask/positions; concatenation alone is insufficient",
        "model_forward": 0,
        "tensor_materialization": 0,
    }


def _check_section_entry(section: str, entry: Mapping[str, Any], query_time: float, index: int) -> None:
    location = f"{section}[{index}]"
    allowed = set(SECTION_FIELDS[section])
    if section in {"interaction_history", "local_dynamics_context"}:
        demand("executed_or_observed_response_digest" in entry, "MISSING_EXPLICIT_NULL_RESPONSE", f"{location} response key absent; use explicit null")
    if section in {"interaction_history", "local_dynamics_context", "failure_outcome_context"}:
        demand("available_time" in entry, "AVAILABILITY_TIME_MISSING", f"{location}.available_time is required")
    _check_exact_keys(entry, allowed, allowed, location)
    if section == "task_demo_context":
        _identifier(entry["demo_id"], f"{location}.demo_id")
        _check_digest(entry["demo_source_sha256"], f"{location}.demo_source_sha256")
        demand(isinstance(entry["language"], str) and bool(entry["language"]), "MISSING_REQUIRED_FIELD", f"{location}.language missing")
        _check_digest(entry["observation_digest"], f"{location}.observation_digest")
        _check_digest(entry["commanded_action_digest"], f"{location}.commanded_action_digest")
        time_range = entry["demo_time_range"]
        demand(isinstance(time_range, list) and len(time_range) == 2, "INVALID_TIMESTAMP", f"{location}.demo_time_range must have two values")
        start = _finite(time_range[0], f"{location}.demo_time_range[0]")
        end = _finite(time_range[1], f"{location}.demo_time_range[1]")
        demand(start <= end <= query_time, "CAUSAL_TIMESTAMP_VIOLATION", f"{location}.demo_time_range is after query")
        return
    event_time = _finite(entry["event_time"], f"{location}.event_time")
    if section in {"interaction_history", "local_dynamics_context", "failure_outcome_context"}:
        available_time = _finite(entry["available_time"], f"{location}.available_time")
        demand(event_time <= available_time <= query_time, "CAUSAL_TIMESTAMP_VIOLATION", f"{location} violates event_time <= available_time <= query_time")
    else:
        available_time = event_time
        demand(event_time <= query_time, "CAUSAL_TIMESTAMP_VIOLATION", f"{location}.event_time is after query")
    if section == "interaction_history":
        _check_digest(entry["observation_state_digest"], f"{location}.observation_state_digest")
        _check_digest(entry["commanded_action_digest"], f"{location}.commanded_action_digest")
        _check_digest(entry["executed_or_observed_response_digest"], f"{location}.executed_or_observed_response_digest", nullable=True)
        _check_digest(entry["delta_state_digest"], f"{location}.delta_state_digest", nullable=True)
        demand(isinstance(entry["timing"], Mapping), "MISSING_REQUIRED_FIELD", f"{location}.timing missing")
        demand(set(entry["timing"]) == set(TIMING_FIELDS), "MODEL_VISIBILITY_DRIFT", f"{location}.timing must contain exactly {list(TIMING_FIELDS)}")
        timing_dt = _finite(entry["timing"]["dt"], f"{location}.timing.dt")
        demand(timing_dt > 0, "DYNAMICS_TRACE_MISSING", f"{location}.timing.dt must be positive")
        return
    if section == "failure_outcome_context":
        _identifier(entry["attempt_id"], f"{location}.attempt_id")
        _check_digest(entry["failure_observation_digest"], f"{location}.failure_observation_digest")
        demand(isinstance(entry["outcome_label"], str) and bool(entry["outcome_label"]), "MISSING_REQUIRED_FIELD", f"{location}.outcome_label missing")
        demand(entry["outcome_label"] in OUTCOME_LABELS, "MODEL_VISIBILITY_DRIFT", f"{location}.outcome_label is outside the declared outcome vocabulary")
        _check_digest(entry["observed_recovery_digest"], f"{location}.observed_recovery_digest", nullable=True)
        return
    # local_dynamics_context: this section is the only one allowed to claim a
    # physical transition, so state-before/after and delta must be explicit.
    _check_digest(entry["state_before_digest"], f"{location}.state_before_digest")
    _check_digest(entry["commanded_action_digest"], f"{location}.commanded_action_digest")
    _check_digest(entry["executed_or_observed_response_digest"], f"{location}.executed_or_observed_response_digest", nullable=True)
    _check_digest(entry["state_after_digest"], f"{location}.state_after_digest")
    _check_digest(entry["delta_state_digest"], f"{location}.delta_state_digest")
    dt = _finite(entry["dt"], f"{location}.dt")
    demand(dt > 0, "DYNAMICS_TRACE_MISSING", f"{location}.dt must be positive")
    demand(event_time + dt <= available_time, "CAUSAL_TIMESTAMP_VIOLATION", f"{location} transition closes after available_time; require event_time + dt <= available_time")


def _canonical_section(section: str, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort by causal time and canonical JSON to make input order irrelevant."""
    metadata_keys = {
        "task_demo_context": {"demo_id", "demo_source_sha256", "demo_time_range"},
        "interaction_history": {"event_time", "available_time"},
        "failure_outcome_context": {"attempt_id", "event_time", "available_time"},
        "local_dynamics_context": {"event_time", "available_time"},
    }[section]

    def key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
        if section == "task_demo_context":
            primary: tuple[Any, ...] = (float(entry["demo_time_range"][0]), float(entry["demo_time_range"][1]))
        else:
            primary = (float(entry["event_time"]), float(entry["available_time"]))
        # IDs and timestamps are metadata-only and must not decide the order
        # of otherwise distinguishable visible payloads.  The source digest
        # references remain in the payload identity because they select the
        # immutable bytes that a future resolver would expose.
        visible = {key: entry[key] for key in sorted(entry) if key not in metadata_keys}
        visible_json = json.dumps(visible, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # If model-visible payloads are identical, metadata is only a final
        # deterministic tie-breaker.  It cannot reorder distinguishable
        # visible payloads because ``visible_json`` precedes this field.
        full_json = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return (*primary, visible_json, full_json)
    return [copy.deepcopy(dict(item)) for item in sorted(entries, key=key)]


def validate_context(context: Mapping[str, Any], *, query_time: float | None = None) -> dict[str, Any]:
    """Validate one ContextV1 record and return a compact audit result."""
    demand(isinstance(context, Mapping), "SCHEMA_DRIFT", "context must be an object")
    _scan_forbidden(context)
    _check_exact_keys(context, RECORD_REQUIRED_FIELDS | RECORD_OPTIONAL_FIELDS, RECORD_REQUIRED_FIELDS, "context")
    demand(context.get("schema") == CONTEXT_SCHEMA and context.get("version") == CONTEXT_VERSION, "SCHEMA_DRIFT", "ContextV1 schema/version drift")
    _identifier(context["context_id"], "context_id")
    _identifier(context["query_id"], "query_id")
    observed_query_time = _finite(context["query_time"], "query_time")
    if query_time is not None:
        requested = _finite(query_time, "query_time argument")
        demand(abs(observed_query_time - requested) <= 0.0, "CAUSAL_TIMESTAMP_VIOLATION", "query_time argument differs from record")
    demand(context["builder_identity"] == BUILDER_IDENTITY, "BUILDER_IDENTITY_DRIFT", "unknown ContextV1 builder")
    upstream = context["upstream"]
    demand(isinstance(upstream, Mapping), "UPSTREAM_IDENTITY_DRIFT", "upstream identity missing")
    _check_exact_keys(upstream, UPSTREAM_FIELDS, UPSTREAM_FIELDS, "upstream")
    demand(upstream.get("project") == "isaac-gr00t" and upstream.get("model") == "GR00T N1.7", "UPSTREAM_IDENTITY_DRIFT", "upstream project/model drift")
    demand(upstream.get("commit") == UPSTREAM_COMMIT, "UPSTREAM_IDENTITY_DRIFT", "GR00T source commit drift")
    _git_sha(upstream["commit"], "upstream.commit")
    source = context["source_identity"]
    demand(isinstance(source, Mapping), "SOURCE_IDENTITY_MISSING", "source identity missing")
    _check_exact_keys(source, SOURCE_FIELDS, SOURCE_FIELDS, "source_identity")
    _identifier(source["artifact_id"], "source_identity.artifact_id")
    _check_digest(source["sha256"], "source_identity.sha256")
    for section in SECTIONS:
        entries = context[section]
        demand(isinstance(entries, list), "SCHEMA_DRIFT", f"{section} must be a list")
        for index, entry in enumerate(entries):
            demand(isinstance(entry, Mapping), "SCHEMA_DRIFT", f"{section}[{index}] must be an object")
            _check_section_entry(section, entry, observed_query_time, index)
    if "window_id" in context:
        _identifier(context["window_id"], "window_id")
    if "target_id" in context:
        _identifier(context["target_id"], "target_id")
    if "notes" in context:
        demand(isinstance(context["notes"], str), "SCHEMA_DRIFT", "notes must be a string")
    return {
        "schema": "ICL_CONTEXTV1_VALIDATION_RESULT_V1",
        "status": "PASS_CPU_ONLY",
        "context_id": context["context_id"],
        "query_id": context["query_id"],
        "section_counts": {section: len(context[section]) for section in SECTIONS},
        "pathless": True,
        "causal_cutoff": observed_query_time,
        "availability_time_law": "event_time <= available_time <= query_time",
        "availability_checked_sections": ["interaction_history", "failure_outcome_context", "local_dynamics_context"],
        "future_information_rejected": True,
        "target_xyz_injected": False,
        "model_visible_fields_resolved_only": True,
        "metadata_only_fields_excluded": True,
        "structured_action_abi": {"schema": "StructuredActionV1", "shape": STRUCTURED_SHAPE, "active_scalars": ACTIVE_SCALARS},
    }


def build_context_record(
    *,
    context_id: str,
    query_id: str,
    query_time: float,
    source_artifact_id: str,
    source_sha256: str,
    task_demo_context: Sequence[Mapping[str, Any]] = (),
    interaction_history: Sequence[Mapping[str, Any]] = (),
    failure_outcome_context: Sequence[Mapping[str, Any]] = (),
    local_dynamics_context: Sequence[Mapping[str, Any]] = (),
    window_id: str | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, pathless ContextV1 record from explicit data."""
    result: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "version": CONTEXT_VERSION,
        "context_id": context_id,
        "query_id": query_id,
        "query_time": query_time,
        "builder_identity": BUILDER_IDENTITY,
        "upstream": {"project": "isaac-gr00t", "model": "GR00T N1.7", "commit": UPSTREAM_COMMIT},
        "source_identity": {"artifact_id": source_artifact_id, "sha256": source_sha256},
        "task_demo_context": _canonical_section("task_demo_context", task_demo_context),
        "interaction_history": _canonical_section("interaction_history", interaction_history),
        "failure_outcome_context": _canonical_section("failure_outcome_context", failure_outcome_context),
        "local_dynamics_context": _canonical_section("local_dynamics_context", local_dynamics_context),
    }
    if window_id is not None:
        result["window_id"] = window_id
    if target_id is not None:
        result["target_id"] = target_id
    validate_context(result)
    return result


def build_context_index(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and canonically order records; no producer fields are guessed."""
    demand(isinstance(records, Sequence) and not isinstance(records, (str, bytes)), "SCHEMA_DRIFT", "records must be a sequence")
    canonical: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for index, record in enumerate(records):
        demand(isinstance(record, Mapping), "SCHEMA_DRIFT", f"record[{index}] is not an object")
        validate_context(record)
        query_id = str(record["query_id"])
        demand(query_id not in query_ids, "DUPLICATE_QUERY_ID", f"duplicate query_id: {query_id}")
        query_ids.add(query_id)
        canonical.append({
            **copy.deepcopy(dict(record)),
            "task_demo_context": _canonical_section("task_demo_context", record["task_demo_context"]),
            "interaction_history": _canonical_section("interaction_history", record["interaction_history"]),
            "failure_outcome_context": _canonical_section("failure_outcome_context", record["failure_outcome_context"]),
            "local_dynamics_context": _canonical_section("local_dynamics_context", record["local_dynamics_context"]),
        })
    canonical.sort(key=lambda item: (float(item["query_time"]), item["query_id"], item["context_id"]))
    return {
        "schema": INDEX_SCHEMA,
        "version": "1.0",
        "status": "CPU_PREPARATION_ONLY",
        "builder_identity": BUILDER_IDENTITY,
        "upstream": {"project": "isaac-gr00t", "model": "GR00T N1.7", "commit": UPSTREAM_COMMIT},
        "pathless": True,
        "target_xyz_in_model_input": False,
        "records": canonical,
        "record_count": len(canonical),
        "repeated_window_cluster_preserved": True,
        "execution_boundary": {"model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "simulator": 0, "WBC": 0, "hardware": 0, "winner_selection": 0, "merge": 0},
    }


def _fixture_digest(number: int) -> str:
    return hashlib.sha256(f"stage45f-icl-fixture-{number}".encode("utf-8")).hexdigest()


def synthetic_context_fixture() -> dict[str, Any]:
    """Small all-sections fixture used by deterministic CPU regression."""
    digest = _fixture_digest(1)
    demo = {
        "demo_id": "demo-a",
        "demo_source_sha256": digest,
        "language": "move the end effector to the red target",
        "observation_digest": _fixture_digest(2),
        "commanded_action_digest": _fixture_digest(3),
        "demo_time_range": [0.0, 0.5],
    }
    interaction = {
        "event_time": 0.6,
        "available_time": 0.6,
        "observation_state_digest": _fixture_digest(4),
        "commanded_action_digest": _fixture_digest(5),
        "executed_or_observed_response_digest": None,
        "delta_state_digest": None,
        "timing": {"dt": 0.1},
    }
    failure = {
        "attempt_id": "attempt-a",
        "failure_observation_digest": _fixture_digest(6),
        "outcome_label": "no_progress_observed",
        "observed_recovery_digest": _fixture_digest(7),
        "event_time": 0.7,
        "available_time": 0.9,
    }
    dynamics = {
        "event_time": 0.8,
        "available_time": 0.95,
        "state_before_digest": _fixture_digest(8),
        "commanded_action_digest": _fixture_digest(9),
        "executed_or_observed_response_digest": _fixture_digest(10),
        "state_after_digest": _fixture_digest(11),
        "delta_state_digest": _fixture_digest(12),
        "dt": 0.1,
    }
    return build_context_record(
        context_id="ctx-a", query_id="query-a", query_time=1.0,
        source_artifact_id="sealed-train24-window-a", source_sha256=digest,
        task_demo_context=[demo], interaction_history=[interaction],
        failure_outcome_context=[failure], local_dynamics_context=[dynamics],
        window_id="window-000", target_id="target-a",
    )


def _expect_rejection(context: Mapping[str, Any], reason_code: str) -> dict[str, str]:
    try:
        validate_context(context)
    except ContextV1Error as exc:
        demand(exc.reason_code == reason_code, "CONTEXT_ORDER_DRIFT", f"expected {reason_code}, got {exc.reason_code}")
        return {"reason_code": exc.reason_code, "detail": exc.detail}
    reject("CONTEXT_ORDER_DRIFT", f"mutation unexpectedly accepted; expected {reason_code}")


def cpu_regression() -> dict[str, Any]:
    """Run deterministic builder, order-invariance, and leakage mutations."""
    baseline = synthetic_context_fixture()
    first = validate_context(baseline)
    tensor_result = build_tensorization_plan(baseline)
    demand(tensor_result["target_id_forwarded"] is False and tensor_result["context_metadata_forwarded"] is False, "MODEL_VISIBILITY_DRIFT", "metadata entered tensorization result")
    reversed_record = copy.deepcopy(baseline)
    for section in SECTIONS:
        reversed_record[section] = list(reversed(reversed_record[section]))
    index_a = build_context_index([baseline])
    index_b = build_context_index([reversed_record])
    demand(index_a == index_b, "CONTEXT_ORDER_DRIFT", "section ordering changes canonical index")
    mutations: list[tuple[str, str]] = []
    future = copy.deepcopy(baseline)
    future["interaction_history"][0]["event_time"] = 2.0
    mutations.append(("future_timestamp", _expect_rejection(future, "CAUSAL_TIMESTAMP_VIOLATION")["reason_code"]))
    interaction_availability = copy.deepcopy(baseline)
    interaction_availability["interaction_history"][0]["event_time"] = 0.8
    interaction_availability["interaction_history"][0]["available_time"] = 1.1
    mutations.append(("interaction_future_availability", _expect_rejection(interaction_availability, "CAUSAL_TIMESTAMP_VIOLATION")["reason_code"]))
    dynamics_availability = copy.deepcopy(baseline)
    dynamics_availability["local_dynamics_context"][0]["event_time"] = 0.85
    dynamics_availability["local_dynamics_context"][0]["available_time"] = 1.1
    mutations.append(("dynamics_future_availability", _expect_rejection(dynamics_availability, "CAUSAL_TIMESTAMP_VIOLATION")["reason_code"]))
    failure_availability = copy.deepcopy(baseline)
    failure_availability["failure_outcome_context"][0]["event_time"] = 0.8
    failure_availability["failure_outcome_context"][0]["available_time"] = 1.1
    mutations.append(("failure_future_availability", _expect_rejection(failure_availability, "CAUSAL_TIMESTAMP_VIOLATION")["reason_code"]))
    missing_availability = copy.deepcopy(baseline)
    del missing_availability["local_dynamics_context"][0]["available_time"]
    mutations.append(("missing_availability_time", _expect_rejection(missing_availability, "AVAILABILITY_TIME_MISSING")["reason_code"]))
    xyz = copy.deepcopy(baseline)
    xyz["target_xyz_world"] = [0.3, 0.1, 0.58]
    mutations.append(("target_xyz_injection", _expect_rejection(xyz, "FORBIDDEN_TARGET_XYZ")["reason_code"]))
    locked = copy.deepcopy(baseline)
    locked["notes"] = "Validation-E must remain unseen"
    mutations.append(("locked_split", _expect_rejection(locked, "LOCKED_SPLIT_CONTAMINATION")["reason_code"]))
    expert = copy.deepcopy(baseline)
    expert["failure_outcome_context"][0]["future_expert_action"] = _fixture_digest(99)
    mutations.append(("future_expert_action", _expect_rejection(expert, "FUTURE_EXPERT_INFORMATION")["reason_code"]))
    timing_metadata = copy.deepcopy(baseline)
    timing_metadata["interaction_history"][0]["timing"]["target_id"] = "target-a"
    mutations.append(("nested_timing_metadata", _expect_rejection(timing_metadata, "MODEL_VISIBILITY_DRIFT")["reason_code"]))
    timing_unknown = copy.deepcopy(baseline)
    timing_unknown["interaction_history"][0]["timing"]["latency_s"] = 0.1
    mutations.append(("nested_timing_unknown_field", _expect_rejection(timing_unknown, "MODEL_VISIBILITY_DRIFT")["reason_code"]))
    transition_closure = copy.deepcopy(baseline)
    transition_closure["local_dynamics_context"][0]["event_time"] = 0.9
    transition_closure["local_dynamics_context"][0]["available_time"] = 0.95
    mutations.append(("transition_closes_after_availability", _expect_rejection(transition_closure, "CAUSAL_TIMESTAMP_VIOLATION")["reason_code"]))
    path = copy.deepcopy(baseline)
    path["source_identity"]["artifact_id"] = "../private-record"
    mutations.append(("path_leakage", _expect_rejection(path, "PATH_LEAKAGE")["reason_code"]))
    missing_response = copy.deepcopy(baseline)
    del missing_response["interaction_history"][0]["executed_or_observed_response_digest"]
    mutations.append(("missing_null_response", _expect_rejection(missing_response, "MISSING_EXPLICIT_NULL_RESPONSE")["reason_code"]))
    fabricated = copy.deepcopy(baseline)
    fabricated["interaction_history"][0]["executed_or_observed_response_digest"] = "not-a-sha"
    mutations.append(("fabricated_response", _expect_rejection(fabricated, "INVALID_SHA256")["reason_code"]))
    visibility_mutation = _read_json(TENSOR_PLAN_PATH)
    visibility_mutation["model_visible_fields"].append("target_id")
    try:
        validate_tensorization_plan(visibility_mutation)
    except ContextV1Error as exc:
        demand(exc.reason_code == "MODEL_VISIBILITY_DRIFT", "MODEL_VISIBILITY_DRIFT", f"metadata visibility mutation yielded {exc.reason_code}")
        visibility_result = {"reason_code": exc.reason_code, "rejected": True}
    else:
        reject("MODEL_VISIBILITY_DRIFT", "metadata visibility mutation unexpectedly accepted")
    parity = c1_zero_gate_attention_counterexample()
    multi = copy.deepcopy(baseline)
    second = copy.deepcopy(multi["interaction_history"][0])
    second["available_time"] = 0.8
    second["observation_state_digest"] = _fixture_digest(40)
    multi["interaction_history"].append(second)
    canonical_multi = _canonical_section("interaction_history", multi["interaction_history"])
    demand(canonical_multi[0]["available_time"] == 0.6 and canonical_multi[1]["available_time"] == 0.8, "CONTEXT_ORDER_DRIFT", "equal-event entries are not ordered by availability")
    reversed_multi = _canonical_section("interaction_history", list(reversed(multi["interaction_history"])))
    demand(canonical_multi == reversed_multi, "CONTEXT_ORDER_DRIFT", "reversed equal-event input changes canonical ordering")
    return {
        "schema": "STAGE45F_ICL_CONTEXTV1_CPU_REGRESSION_RESULT_V1",
        "status": "PASS",
        "token": "STAGE45F_ICL_CONTEXTV1_CPU_REGRESSION_PASS",
        "baseline": first,
        "tensor_plan": tensor_result,
        "order_invariant": True,
        "mutation_results": [{"name": name, "reason_code": code, "rejected": True} for name, code in mutations],
        "mutation_count": len(mutations),
        "metadata_visibility_mutation": visibility_result,
        "equal_event_availability_order": "PASS",
        "transition_closure_law": LOCAL_DYNAMICS_CLOSURE_LAW,
        "c1_zero_gate_parity_counterexample": parity,
        "execution_boundary": {"model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }


def static_audit() -> dict[str, Any]:
    """Audit this module and all ICL manifests without importing model code."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    demand(not forbidden, "STATIC_IMPORT_VIOLATION", f"forbidden imports: {forbidden}")
    contract = _read_json(CONTRACT_PATH)
    capability = _read_json(CAPABILITY_PATH)
    options = _read_json(OPTIONS_PATH)
    ladder = _read_json(LADDER_PATH)
    risk = _read_json(RISK_PATH)
    tensor_plan = _read_json(TENSOR_PLAN_PATH)
    parity_plan = _read_json(PARITY_PATH)
    demand(contract.get("schema") == "ICL_CONTEXTV1_CONTRACT" and contract.get("version") == "1.1" and contract.get("status") == "PREPARATION_ONLY", "SCHEMA_DRIFT", "ContextV1 contract drift")
    demand(contract.get("upstream", {}).get("commit") == UPSTREAM_COMMIT, "UPSTREAM_IDENTITY_DRIFT", "contract upstream drift")
    demand(contract.get("path_policy", {}).get("pathless") is True and contract.get("path_policy", {}).get("filesystem_paths_allowed") is False, "PATH_LEAKAGE", "path policy drift")
    causal = contract.get("causal_query", {})
    demand(causal.get("relative_time_rule") == "event_time <= available_time <= query_time" and causal.get("availability_field") == "available_time", "EXECUTION_BOUNDARY_DRIFT", "availability-time law drift")
    demand(causal.get("ordering_law") == CAUSAL_ORDERING_LAW, "CONTEXT_ORDER_DRIFT", "causal ordering law drift")
    demand(causal.get("local_dynamics_closure_law") == LOCAL_DYNAMICS_CLOSURE_LAW, "CAUSAL_TIMESTAMP_VIOLATION", "local transition closure law drift")
    demand(contract.get("output_abi", {}).get("shape") == STRUCTURED_SHAPE and contract.get("output_abi", {}).get("active_scalars") == ACTIVE_SCALARS, "ABI_DRIFT", "StructuredAction ABI drift")
    visibility = contract.get("model_visibility", {})
    demand(visibility.get("target_id_model_visible") is False and visibility.get("model_visible_fields_are_resolved_payload_only") is True and visibility.get("tensor_plan_schema") == "ICL_CONTEXTTENSORV1_PLAN", "MODEL_VISIBILITY_DRIFT", "ContextV1 model visibility contract drift")
    demand(set(visibility.get("metadata_only_fields", [])) == METADATA_ONLY_PATHS, "MODEL_VISIBILITY_DRIFT", "ContextV1 metadata-only fields drift")
    demand(set(visibility.get("model_visible_fields", [])) == MODEL_VISIBLE_PATHS, "MODEL_VISIBILITY_DRIFT", "ContextV1 model-visible fields drift")
    for section in ("interaction_history", "failure_outcome_context", "local_dynamics_context"):
        demand("available_time" in contract.get("sections", {}).get(section, {}).get("fields", []), "AVAILABILITY_TIME_MISSING", f"contract missing {section}.available_time")
    timing_contract = contract.get("sections", {}).get("interaction_history", {}).get("timing_schema", {})
    demand(timing_contract.get("fields") == list(TIMING_FIELDS) and timing_contract.get("units") == "seconds" and timing_contract.get("dtype") == "float32" and timing_contract.get("missingness") == "not_allowed", "MODEL_VISIBILITY_DRIFT", "interaction timing schema drift")
    outcome_contract = contract.get("sections", {}).get("failure_outcome_context", {}).get("outcome_label_schema", {})
    demand(outcome_contract.get("allowed_values") == list(OUTCOME_LABELS) and outcome_contract.get("dtype") == "string", "MODEL_VISIBILITY_DRIFT", "outcome label vocabulary drift")
    contract_boundary = contract.get("execution_boundary", {})
    demand(all(value == 0 for value in contract_boundary.values()), "EXECUTION_BOUNDARY_DRIFT", "contract execution boundary is nonzero")
    demand(capability.get("source", {}).get("commit") == UPSTREAM_COMMIT and capability.get("status") == "PASS_READONLY_SOURCE_AUDIT", "UPSTREAM_IDENTITY_DRIFT", "capability audit drift")
    demand(set(options.get("options", {})) == {"A_MULTIFRAME_MULTIMODAL_PROMPT", "B_EXTENDED_STATE_HISTORY", "C1_ACTION_SIDE_CONTEXT_TOKENS", "C2_VL_MEMORY_CONTEXT_TOKENS", "D_EXPLICIT_MEMORY_CROSS_ATTENTION"}, "SCHEMA_DRIFT", "integration options are not the five neutral families")
    demand(options.get("selection_policy", {}).get("automatic_selection") is False and options.get("selection_policy", {}).get("PI_DECISION_REQUIRED") is True, "EXECUTION_BOUNDARY_DRIFT", "option selection policy drift")
    architecture = options.get("architecture_contract", {})
    demand(architecture.get("c1_action_side_dim") == 1536 and architecture.get("c2_vl_memory_dim") == 2048 and architecture.get("c1_zero_gate_alpha0") == 0 and architecture.get("c1_final_action_tokens") == 40 and architecture.get("c2_attention_mask") is True and architecture.get("c2_image_mask") is False, "TENSOR_PLAN_DRIFT", "C1/C2 architecture contract drift")
    c1_gate = options.get("options", {}).get("C1_ACTION_SIDE_CONTEXT_TOKENS", {}).get("gate", {})
    demand(c1_gate.get("base_policy_parity") == "DESIGN_OBLIGATION_NOT_PROVEN", "TENSOR_PLAN_DRIFT", "C1 parity claim must remain unproven until a base-path-preserving design is selected")
    demand(c1_gate.get("zero_gate_contract") == "isolated_context_residual_or_mask_preserving_bypass", "TENSOR_PLAN_DRIFT", "C1 zero-gate path contract drift")
    demand(c1_gate.get("gradient_contract") == "gate_and_residual_gradients_are_distinct", "TENSOR_PLAN_DRIFT", "C1 zero-gate gradient contract drift")
    demand(c1_gate.get("gate_gradient_at_alpha0") == "may_be_nonzero when R_phi is nonzero: dv/dalpha = R_phi", "TENSOR_PLAN_DRIFT", "C1 gate-gradient semantics drift")
    demand(c1_gate.get("context_encoder_gradient_at_alpha0") == "initially_zero_from_alpha_scaling: dv/dphi = alpha * dR_phi/dphi", "TENSOR_PLAN_DRIFT", "C1 encoder-gradient semantics drift")
    demand("not_claim" in c1_gate.get("gradient_claim_boundary", ""), "TENSOR_PLAN_DRIFT", "C1 gradient claim boundary drift")
    demand("stall" in c1_gate.get("zero_gate_learning_risk", ""), "TENSOR_PLAN_DRIFT", "C1 gate stall-risk boundary drift")
    gradient_equations = architecture.get("c1_gradient_equations", {})
    demand(gradient_equations.get("gate") == "dv/dalpha at alpha=0 = R_phi (may be nonzero)" and gradient_equations.get("context_parameters") == "dv/dphi at alpha=0 = alpha * dR_phi/dphi = 0", "TENSOR_PLAN_DRIFT", "C1 gradient equations drift")
    variants = ladder.get("context_variants")
    demand(isinstance(variants, list) and len(variants) == 9 and len(set(variants)) == 9, "SCHEMA_DRIFT", "context variant ladder drift")
    experiments = ladder.get("experiments", {})
    demand(set(experiments) == {"I0_EXISTING_INTERFACE_CONTEXT_PROBE", "I0_ADAPTER_ZERO_GATE_PARITY", "I1_CONTEXT_ADAPTER_ONLY", "I2_CONTEXT_AWARE_JOINT_POST_TRAINING", "I3_HISTORY_CONDITIONED_DAGGER"}, "SCHEMA_DRIFT", "I0-I3 ladder drift")
    demand(all(item.get("execution_enabled") is False and item.get("model_forward") == 0 and item.get("training") == 0 for item in experiments.values()), "EXECUTION_BOUNDARY_DRIFT", "I0-I3 execution boundary drift")
    demand(experiments["I0_EXISTING_INTERFACE_CONTEXT_PROBE"].get("new_modules") is False and experiments["I0_EXISTING_INTERFACE_CONTEXT_PROBE"].get("new_parameter_shape") is False, "EXECUTION_BOUNDARY_DRIFT", "I0 existing-interface scope drift")
    demand(experiments["I0_ADAPTER_ZERO_GATE_PARITY"].get("actual_forward_locked_until_future_gate") is True and experiments["I1_CONTEXT_ADAPTER_ONLY"].get("first_learned_context_stage") is True, "EXECUTION_BOUNDARY_DRIFT", "I0/I1 adapter boundary drift")
    demand(risk.get("selection_policy", {}).get("automatic_winner") is False and risk.get("selection_policy", {}).get("PI_DECISION_REQUIRED") is True, "EXECUTION_BOUNDARY_DRIFT", "risk selection policy drift")
    validate_tensorization_plan(tensor_plan)
    demand(parity_plan.get("schema") == "STAGE45F_ICL_C1_PARITY_COUNTEREXAMPLE_V1" and parity_plan.get("status") == "CPU_SCALAR_PLAN_ONLY", "TENSOR_PLAN_DRIFT", "C1 parity counterexample manifest drift")
    expected = parity_plan.get("expected", {})
    demand(expected.get("original_softmax_output") == 1.0 and expected.get("zero_context_appended_output") == 0.5 and expected.get("exact_parity_from_zero_values") is False, "TENSOR_PLAN_DRIFT", "C1 scalar parity counterexample changed")
    obligation = parity_plan.get("c1_obligation", {})
    demand(obligation.get("context_dim") == 1536 and obligation.get("preserve_original_positions_and_attention_denominator") is True and obligation.get("isolated_context_residual_receives_gradient") is True and obligation.get("claim") == "DESIGN_OBLIGATION_NOT_PROVEN_BY_ZERO_VALUES_ALONE", "TENSOR_PLAN_DRIFT", "C1 base-path preservation obligation drift")
    return {
        "schema": "STAGE45F_ICL_CONTEXTV1_STATIC_AUDIT_RESULT_V1",
        "status": "PASS",
        "forbidden_top_level_imports": forbidden,
        "upstream_commit": UPSTREAM_COMMIT,
        "manifest_sha256": {
            "contract": convergence.sha256_file(CONTRACT_PATH),
            "capability": convergence.sha256_file(CAPABILITY_PATH),
            "options": convergence.sha256_file(OPTIONS_PATH),
            "ladder": convergence.sha256_file(LADDER_PATH),
            "risk": convergence.sha256_file(RISK_PATH),
            "tensor_plan": convergence.sha256_file(TENSOR_PLAN_PATH),
            "c1_parity_counterexample": convergence.sha256_file(PARITY_PATH),
        },
        "context_variant_count": len(variants),
        "integration_option_count": 5,
        "experiment_count": 5,
        "i0_existing_interface_variant_count": len(experiments["I0_EXISTING_INTERFACE_CONTEXT_PROBE"]["variants"]),
        "execution_boundary": {"model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "winner_selection": 0, "merge": 0},
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("static", "regression", "validate", "index"), default="static")
    value.add_argument("--input", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "static":
        result = static_audit()
    elif args.mode == "regression":
        result = cpu_regression()
    else:
        demand(args.input is not None, "SCHEMA_DRIFT", "--input required")
        value = _read_json(args.input)
        if args.mode == "validate":
            result = validate_context(value)
        else:
            demand(isinstance(value, list), "SCHEMA_DRIFT", "index input must be a JSON list")
            result = build_context_index(value)
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContextV1Error as exc:
        print(f"STAGE45F_ICL_CONTEXTV1_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
