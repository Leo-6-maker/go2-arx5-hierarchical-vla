#!/usr/bin/env python3
"""CPU-only ContextV1 builder and leakage/causality verifier for PI0143.

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

CONTEXT_SCHEMA = "ICL_CONTEXTV1_RECORD_V1"
INDEX_SCHEMA = "ICL_CONTEXTV1_INDEX_V1"
BUILDER_IDENTITY = "STAGE45F_ICL_CONTEXTV1_CPU_BUILDER_V1"
UPSTREAM_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
ZERO_SHA256 = "0" * 64
STRUCTURED_SHAPE = [40, 132]
ACTIVE_SCALARS = 190

SECTION_FIELDS = {
    "task_demo_context": (
        "demo_id", "demo_source_sha256", "language", "observation_digest",
        "commanded_action_digest", "demo_time_range",
    ),
    "interaction_history": (
        "event_time", "observation_state_digest", "commanded_action_digest",
        "executed_or_observed_response_digest", "delta_state_digest", "timing",
    ),
    "failure_outcome_context": (
        "attempt_id", "failure_observation_digest", "outcome_label",
        "observed_recovery_digest", "event_time",
    ),
    "local_dynamics_context": (
        "event_time", "state_before_digest", "commanded_action_digest",
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
    "validation-e", "holdout-f", "target_xyz", "target xyz",
    "future_expert", "future_action", "future observation", "expert_action",
    "expert action", "expert_label", "expert label",
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
        lowered = value.lower().replace("_", "-")
        if any(fragment in lowered for fragment in FORBIDDEN_VALUE_FRAGMENTS):
            code = "FORBIDDEN_TARGET_XYZ" if "target-xyz" in lowered or "target xyz" in lowered else (
                "LOCKED_SPLIT_CONTAMINATION" if "validation-e" in lowered or "holdout-f" in lowered else "FUTURE_EXPERT_INFORMATION"
            )
            reject(code, f"forbidden value at {location}")
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


def _check_section_entry(section: str, entry: Mapping[str, Any], query_time: float, index: int) -> None:
    location = f"{section}[{index}]"
    allowed = set(SECTION_FIELDS[section])
    if section in {"interaction_history", "local_dynamics_context"}:
        demand("executed_or_observed_response_digest" in entry, "MISSING_EXPLICIT_NULL_RESPONSE", f"{location} response key absent; use explicit null")
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
    demand(event_time <= query_time, "CAUSAL_TIMESTAMP_VIOLATION", f"{location}.event_time is after query")
    if section == "interaction_history":
        _check_digest(entry["observation_state_digest"], f"{location}.observation_state_digest")
        _check_digest(entry["commanded_action_digest"], f"{location}.commanded_action_digest")
        _check_digest(entry["executed_or_observed_response_digest"], f"{location}.executed_or_observed_response_digest", nullable=True)
        _check_digest(entry["delta_state_digest"], f"{location}.delta_state_digest", nullable=True)
        demand(isinstance(entry["timing"], Mapping), "MISSING_REQUIRED_FIELD", f"{location}.timing missing")
        for key, value in entry["timing"].items():
            _identifier(str(key), f"{location}.timing key")
            _finite(value, f"{location}.timing.{key}", nullable=True)
        return
    if section == "failure_outcome_context":
        _identifier(entry["attempt_id"], f"{location}.attempt_id")
        _check_digest(entry["failure_observation_digest"], f"{location}.failure_observation_digest")
        demand(isinstance(entry["outcome_label"], str) and bool(entry["outcome_label"]), "MISSING_REQUIRED_FIELD", f"{location}.outcome_label missing")
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


def _canonical_section(section: str, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort by causal time and canonical JSON to make input order irrelevant."""
    def key(entry: Mapping[str, Any]) -> tuple[Any, str]:
        if section == "task_demo_context":
            primary: Any = entry["demo_id"]
        elif section == "failure_outcome_context":
            primary = (float(entry["event_time"]), entry["attempt_id"])
        else:
            primary = (float(entry["event_time"]),)
        return primary, json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return [copy.deepcopy(dict(item)) for item in sorted(entries, key=key)]


def validate_context(context: Mapping[str, Any], *, query_time: float | None = None) -> dict[str, Any]:
    """Validate one ContextV1 record and return a compact audit result."""
    demand(isinstance(context, Mapping), "SCHEMA_DRIFT", "context must be an object")
    _scan_forbidden(context)
    _check_exact_keys(context, RECORD_REQUIRED_FIELDS | RECORD_OPTIONAL_FIELDS, RECORD_REQUIRED_FIELDS, "context")
    demand(context.get("schema") == CONTEXT_SCHEMA and context.get("version") == "1.0", "SCHEMA_DRIFT", "ContextV1 schema/version drift")
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
        "future_information_rejected": True,
        "target_xyz_injected": False,
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
        "version": "1.0",
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
        "execution_boundary": {"model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "simulator": 0, "WBC": 0, "hardware": 0},
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
    }
    dynamics = {
        "event_time": 0.8,
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
    xyz = copy.deepcopy(baseline)
    xyz["target_xyz_world"] = [0.3, 0.1, 0.58]
    mutations.append(("target_xyz_injection", _expect_rejection(xyz, "FORBIDDEN_TARGET_XYZ")["reason_code"]))
    locked = copy.deepcopy(baseline)
    locked["notes"] = "Validation-E must remain unseen"
    mutations.append(("locked_split", _expect_rejection(locked, "LOCKED_SPLIT_CONTAMINATION")["reason_code"]))
    expert = copy.deepcopy(baseline)
    expert["failure_outcome_context"][0]["future_expert_action"] = _fixture_digest(99)
    mutations.append(("future_expert_action", _expect_rejection(expert, "FUTURE_EXPERT_INFORMATION")["reason_code"]))
    path = copy.deepcopy(baseline)
    path["source_identity"]["artifact_id"] = "../private-record"
    mutations.append(("path_leakage", _expect_rejection(path, "PATH_LEAKAGE")["reason_code"]))
    missing_response = copy.deepcopy(baseline)
    del missing_response["interaction_history"][0]["executed_or_observed_response_digest"]
    mutations.append(("missing_null_response", _expect_rejection(missing_response, "MISSING_EXPLICIT_NULL_RESPONSE")["reason_code"]))
    fabricated = copy.deepcopy(baseline)
    fabricated["interaction_history"][0]["executed_or_observed_response_digest"] = "not-a-sha"
    mutations.append(("fabricated_response", _expect_rejection(fabricated, "INVALID_SHA256")["reason_code"]))
    return {
        "schema": "STAGE45F_ICL_CONTEXTV1_CPU_REGRESSION_RESULT_V1",
        "status": "PASS",
        "token": "STAGE45F_ICL_CONTEXTV1_CPU_REGRESSION_PASS",
        "baseline": first,
        "order_invariant": True,
        "mutation_results": [{"name": name, "reason_code": code, "rejected": True} for name, code in mutations],
        "mutation_count": len(mutations),
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
    demand(contract.get("schema") == "ICL_CONTEXTV1_CONTRACT" and contract.get("status") == "PREPARATION_ONLY", "SCHEMA_DRIFT", "ContextV1 contract drift")
    demand(contract.get("upstream", {}).get("commit") == UPSTREAM_COMMIT, "UPSTREAM_IDENTITY_DRIFT", "contract upstream drift")
    demand(contract.get("path_policy", {}).get("pathless") is True and contract.get("path_policy", {}).get("filesystem_paths_allowed") is False, "PATH_LEAKAGE", "path policy drift")
    demand(contract.get("output_abi", {}).get("shape") == STRUCTURED_SHAPE and contract.get("output_abi", {}).get("active_scalars") == ACTIVE_SCALARS, "ABI_DRIFT", "StructuredAction ABI drift")
    contract_boundary = contract.get("execution_boundary", {})
    demand(all(value == 0 for value in contract_boundary.values()), "EXECUTION_BOUNDARY_DRIFT", "contract execution boundary is nonzero")
    demand(capability.get("source", {}).get("commit") == UPSTREAM_COMMIT and capability.get("status") == "PASS_READONLY_SOURCE_AUDIT", "UPSTREAM_IDENTITY_DRIFT", "capability audit drift")
    demand(set(options.get("options", {})) == {"A_MULTIFRAME_MULTIMODAL_PROMPT", "B_EXTENDED_STATE_HISTORY", "C_CONTEXT_ENCODER_K_TOKENS", "D_EXPLICIT_MEMORY_CROSS_ATTENTION"}, "SCHEMA_DRIFT", "integration options are not the four neutral families")
    demand(options.get("selection_policy", {}).get("automatic_selection") is False and options.get("selection_policy", {}).get("PI_DECISION_REQUIRED") is True, "EXECUTION_BOUNDARY_DRIFT", "option selection policy drift")
    variants = ladder.get("context_variants")
    demand(isinstance(variants, list) and len(variants) == 9 and len(set(variants)) == 9, "SCHEMA_DRIFT", "context variant ladder drift")
    experiments = ladder.get("experiments", {})
    demand(set(experiments) == {"I0_FROZEN_POLICY_CONTEXT_SENSITIVITY", "I1_CONTEXT_ADAPTER_ONLY", "I2_CONTEXT_AWARE_JOINT_POST_TRAINING", "I3_HISTORY_CONDITIONED_DAGGER"}, "SCHEMA_DRIFT", "I0-I3 ladder drift")
    demand(all(item.get("execution_enabled") is False and item.get("model_forward") == 0 and item.get("training") == 0 for item in experiments.values()), "EXECUTION_BOUNDARY_DRIFT", "I0-I3 execution boundary drift")
    demand(risk.get("selection_policy", {}).get("automatic_winner") is False and risk.get("selection_policy", {}).get("PI_DECISION_REQUIRED") is True, "EXECUTION_BOUNDARY_DRIFT", "risk selection policy drift")
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
        },
        "context_variant_count": len(variants),
        "integration_option_count": 4,
        "experiment_count": 4,
        "execution_boundary": {"model_forward": 0, "training": 0, "gpu": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "winner_selection": 0},
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
