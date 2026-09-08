#!/usr/bin/env python3
"""Fail-closed bundles and seals for the Stage41 LH10 engineering run.

This module is CPU-only.  It validates the bounded 10-handoff contract,
hashes every required evidence role, and binds each run to the accepted
Stage40 M2 seal without importing a model, simulator, or CUDA runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import stage40_corelab_evidence as stage40


PI_GATE = "PI_GATE_STATE_0066"
ISSUE_COMMENT = 5579564143
DEV_SEED = 4242
CLASSIFICATION = "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE"
STAGE_ID = "LH10"
EXECUTION_CLASS = "CORELAB_STAGE41_10H_LONG_HORIZON_V1"
HANDOFF_S = 0.975
BUNDLE_SCHEMA = "CORELAB_STAGE41_EXECUTION_BUNDLE_V1"
RESULT_SCHEMA = "CORELAB_STAGE41_LH10_RESULT_V1"
SEAL_SCHEMA = "CORELAB_STAGE41_STAGE_SEAL_V1"
HANDOFF_COUNT = 10
GR00T_FORWARD_COUNT = HANDOFF_COUNT + 1
REFERENCE_UPDATE_COUNT = HANDOFF_COUNT
SAFETY_DECISION_COUNT = GR00T_FORWARD_COUNT
COUNTER_NAMES = ("gr00t_forward_count", "wbc_step_count", "reference_update_count")
FORBIDDEN_FALSE_FIELDS = (
    "training", "formal_seed_used", "validation_e_read", "holdout_f_read",
    "hardware_executed", "h200_used", "checkpoint_substitution",
    "model_substitution", "processor_substitution", "precision_change",
    "quantization", "cpu_offload", "clipping", "smoothing", "blending",
    "adaptive_retiming", "physics_reset", "physics_teleport",
    "policy_history_clear",
)
REQUIRED_EVIDENCE_ROLES = {
    "observation_body", "observation_wrist", "state_v1", "worker_request",
    "worker_response", "raw_action_d132", "decoded_action_d10_h40",
    "safety_gate", "initial_bind", "root_state", "dof_state",
    "executor_bookkeeping", "rows_before_handoff", "rows_after_handoff",
    "handoff", "counter_observations", "mailbox", "logs", "telemetry",
    "run_metadata",
}


class Stage41EvidenceError(ValueError):
    """Raised when a Stage41 bundle, run, or seal fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage41EvidenceError(message)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required: %s" % path)
    return value


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return sha256_bytes(payload)


def _resolved_child(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path))
    _require(not relative.is_absolute(), "evidence path must be relative")
    _require(relative.parts and all(part not in {"", ".", ".."} for part in relative.parts), "non-canonical evidence path")
    root_real = Path(root).resolve(strict=True)
    cursor = root_real
    for part in relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "symlink evidence is forbidden")
    candidate = (root_real / relative).resolve(strict=True)
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise Stage41EvidenceError("evidence path escapes evidence root") from exc
    _require(candidate.is_file(), "evidence is not a regular file")
    return candidate


def expected_forbidden_operations() -> Dict[str, bool]:
    return {name: False for name in FORBIDDEN_FALSE_FIELDS}


def build_evidence_manifest(evidence_root: Path, role_paths: Mapping[str, str]) -> Dict[str, Any]:
    roles = {str(key): str(value) for key, value in role_paths.items()}
    missing = sorted(REQUIRED_EVIDENCE_ROLES - set(roles))
    _require(not missing, "missing required evidence roles: " + ", ".join(missing))
    _require(len(set(roles.values())) == len(roles), "one evidence file cannot satisfy multiple roles")
    records = []
    for role, relative in sorted(roles.items()):
        path = _resolved_child(Path(evidence_root), relative)
        records.append({
            "role": role,
            "relative_path": Path(relative).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return {
        "schema": "CORELAB_STAGE41_EVIDENCE_MANIFEST_V1",
        "stage_id": STAGE_ID,
        "records": records,
        "record_count": len(records),
    }


def verify_evidence_manifest(manifest: Mapping[str, Any], evidence_root: Path) -> Dict[str, Any]:
    _require(manifest.get("schema") == "CORELAB_STAGE41_EVIDENCE_MANIFEST_V1", "wrong Stage41 evidence manifest schema")
    _require(manifest.get("stage_id") == STAGE_ID, "wrong Stage41 evidence manifest stage")
    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == int(manifest.get("record_count", -1)), "evidence record count mismatch")
    roles = []
    paths = []
    for record in records:
        _require(isinstance(record, dict), "evidence record is not an object")
        role = str(record.get("role", ""))
        relative = str(record.get("relative_path", ""))
        path = _resolved_child(Path(evidence_root), relative)
        _require(int(record.get("size_bytes", -1)) == path.stat().st_size, "evidence size mismatch: " + role)
        _require(str(record.get("sha256")) == sha256_file(path), "evidence hash mismatch: " + role)
        roles.append(role)
        paths.append(relative)
    _require(len(roles) == len(set(roles)), "duplicate evidence role")
    _require(len(paths) == len(set(paths)), "duplicate evidence path")
    _require(not (REQUIRED_EVIDENCE_ROLES - set(roles)), "evidence manifest omits required roles")
    return {"stage_id": STAGE_ID, "record_count": len(records), "status": "PASS"}


def build_counter_observation(
    before: Mapping[str, int],
    after: Mapping[str, int],
    gr00t_requests: Sequence[int],
    wbc_segments: Sequence[int],
    reference_updates: Sequence[int],
) -> Dict[str, Any]:
    counters = {}
    for name in COUNTER_NAMES:
        left = before.get(name)
        right = after.get(name)
        _require(isinstance(left, int) and isinstance(right, int), "counter values must be observed integers")
        _require(left >= 0 and right >= left, "counters must be non-negative and monotonic")
        counters[name] = {"before": left, "after": right, "delta": right - left}
    return {
        "schema": "CORELAB_STAGE41_COUNTER_OBSERVATION_V1",
        "counters": counters,
        "gr00t_request_deltas": list(gr00t_requests),
        "wbc_segment_deltas": list(wbc_segments),
        "reference_update_deltas": list(reference_updates),
    }


def verify_counter_observations(counters: Mapping[str, Any], manifest: Mapping[str, Any], evidence_root: Path) -> Dict[str, int]:
    records = [item for item in manifest.get("records", []) if item.get("role") == "counter_observations"]
    _require(len(records) == 1, "exactly one counter observation record is required")
    observation = read_json(_resolved_child(Path(evidence_root), str(records[0].get("relative_path"))))
    _require(observation.get("schema") == "CORELAB_STAGE41_COUNTER_OBSERVATION_V1", "wrong counter evidence schema")
    observed = observation.get("counters")
    _require(isinstance(observed, dict), "counter observations missing")
    deltas = {}
    for name in COUNTER_NAMES:
        row = observed.get(name)
        _require(isinstance(row, dict), "missing observed counter: " + name)
        before, after, delta = row.get("before"), row.get("after"), row.get("delta")
        _require(isinstance(before, int) and isinstance(after, int) and isinstance(delta, int), "counter observation must use integers")
        _require(before >= 0 and after >= before and delta == after - before, "forged/non-monotonic counter evidence: " + name)
        _require(counters.get(name) == delta, "result counter differs from observed delta: " + name)
        deltas[name] = delta
    requests = observation.get("gr00t_request_deltas")
    updates = observation.get("reference_update_deltas")
    segments = observation.get("wbc_segment_deltas")
    _require(requests == [1] * GR00T_FORWARD_COUNT and sum(requests) == deltas["gr00t_forward_count"], "per-request GR00T counter evidence mismatch")
    _require(updates == [1] * REFERENCE_UPDATE_COUNT and sum(updates) == deltas["reference_update_count"], "per-update reference counter evidence mismatch")
    _require(isinstance(segments, list) and len(segments) == GR00T_FORWARD_COUNT and all(isinstance(item, int) and item > 0 for item in segments) and sum(segments) == deltas["wbc_step_count"], "per-segment WBC counter evidence mismatch")
    return deltas


def validate_long_horizon_contract(
    cases: Sequence[Mapping[str, Any]],
    handoffs: Sequence[Mapping[str, Any]],
    wbc_segments: Sequence[int],
    safety_decisions: Sequence[str],
    completion: Mapping[str, Any],
    *,
    expected_native_steps: Optional[Sequence[int]] = None,
    native_dt_s: Optional[float] = None,
    expected_executor_prefix: Optional[int] = None,
    expected_action_dt_s: Optional[float] = None,
    observation_timestamps: Optional[Sequence[float]] = None,
    continuity_events: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate the bounded causal schedule using only serialized observations."""

    _require(len(cases) == GR00T_FORWARD_COUNT, "LH10 requires exactly 11 GR00T cases")
    expected_ids = ["stage41-q%d" % index for index in range(GR00T_FORWARD_COUNT)]
    actual_ids = [str(case.get("request_id")) for case in cases]
    _require(actual_ids == expected_ids, "fresh observation/request order mismatch")
    _require(len(handoffs) == HANDOFF_COUNT, "LH10 requires exactly 10 reference handoffs")
    _require(len(wbc_segments) == GR00T_FORWARD_COUNT and all(isinstance(item, int) and item > 0 for item in wbc_segments), "LH10 WBC segment vector mismatch")
    _require(len(safety_decisions) == SAFETY_DECISION_COUNT and all(item in {"ACCEPT", "WARN"} for item in safety_decisions), "LH10 SafetyGate decision vector is inadmissible")
    timestamps = [] if observation_timestamps is None else [float(item) for item in observation_timestamps]
    if timestamps:
        _require(len(timestamps) == GR00T_FORWARD_COUNT and all(math.isfinite(item) for item in timestamps), "invalid observation timestamps")
        _require(timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps), "observation timestamps are stale or reordered")
    for index, case in enumerate(cases):
        response = case.get("response", {})
        _require(int(response.get("worker_forward_count_delta", -1)) == 1, "worker forward delta is not exactly one at q%d" % index)
        _require(response.get("action_contract") == "STRUCTURED_ACTION_V1", "action contract mismatch at q%d" % index)
        _require(bool(case.get("finite", True)), "non-finite model evidence at q%d" % index)
        gate = case.get("safety_gate", {})
        _require(str(gate.get("decision")) in {"ACCEPT", "WARN"}, "SafetyGate rejected q%d" % index)
    native_steps = [int(event.get("native_handoff_steps", event.get("actual_native_control_steps", 0))) for event in handoffs]
    executor_prefix = [int(event.get("executor_prefix_steps", 0)) for event in handoffs]
    action_dt = [float(event.get("action_sample_dt_s", float("nan"))) for event in handoffs]
    _require(all(item > 0 for item in native_steps), "native handoff schedule is invalid")
    if native_dt_s is not None:
        native_dt = float(native_dt_s)
        _require(math.isfinite(native_dt) and native_dt > 0.0, "observed native dt is invalid")
        derived_native_steps = int(round(HANDOFF_S / native_dt))
        _require(derived_native_steps > 0 and native_steps == [derived_native_steps] * HANDOFF_COUNT, "native handoff schedule does not match observed native dt")
    if expected_native_steps is not None:
        _require(native_steps == [int(item) for item in expected_native_steps], "native handoff schedule drift")
    _require(all(math.isfinite(item) and item > 0.0 for item in action_dt), "action sample period is invalid")
    reference_action_dt = action_dt[0]
    _require(all(math.isclose(item, reference_action_dt, rel_tol=0.0, abs_tol=1e-12) for item in action_dt), "action sample period changed across handoffs")
    derived_executor_prefix = [int(round(HANDOFF_S / item)) for item in action_dt]
    _require(all(item > 0 for item in derived_executor_prefix) and executor_prefix == derived_executor_prefix, "executor action prefix does not match observed action period")
    if expected_executor_prefix is not None:
        _require(executor_prefix == [int(expected_executor_prefix)] * HANDOFF_COUNT, "executor action prefix schedule drift")
    if expected_action_dt_s is not None:
        _require(math.isclose(reference_action_dt, float(expected_action_dt_s), rel_tol=0.0, abs_tol=1e-12), "action sample period drift")
    if continuity_events is not None:
        _require(len(continuity_events) == HANDOFF_COUNT + 1, "continuity event count mismatch")
        true_fields = ("physics_root_state_unchanged", "physics_dof_state_unchanged", "episode_length_unchanged", "arm_policy_history_unchanged", "dog_policy_history_unchanged", "reset_buf_unchanged")
        false_fields = ("physics_reset", "physics_teleport", "policy_history_reset")
        for event in continuity_events:
            _require(all(event.get(field) is True for field in true_fields), "lifecycle continuity failed")
            _require(all(event.get(field) is False for field in false_fields), "forbidden lifecycle effect observed")
    _require(bool(completion.get("endpoint_reached")), "LH10 completion endpoint was not reached")
    return {
        "status": "PASS",
        "handoff_count": HANDOFF_COUNT,
        "gr00t_forward_count": GR00T_FORWARD_COUNT,
        "reference_update_count": REFERENCE_UPDATE_COUNT,
        "native_handoff_steps": native_steps,
        "executor_prefix_steps": executor_prefix,
        "action_sample_dt_s": action_dt,
    }


def _validate_bundle(bundle: Mapping[str, Any]) -> None:
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "wrong Stage41 execution bundle schema")
    _require(bundle.get("stage_id") == STAGE_ID, "wrong Stage41 stage id")
    _require(bundle.get("pi_gate") == PI_GATE, "wrong Stage41 PI gate")
    _require(bundle.get("classification") == CLASSIFICATION, "wrong execution classification")
    _require(bundle.get("execution_class") == EXECUTION_CLASS, "wrong Stage41 execution class")
    _require(bundle.get("dev_seed") == DEV_SEED, "wrong Stage41 development seed")
    _require(isinstance(bundle.get("repository_head"), str) and len(bundle["repository_head"]) == 40, "invalid repository HEAD")
    lineage = bundle.get("executable_lineage")
    _require(isinstance(lineage, dict) and bool(lineage.get("runner_sha256")) and bool(lineage.get("source_sha256")), "missing executable lineage")
    _require(isinstance(bundle.get("artifact_identities"), dict) and bool(bundle["artifact_identities"]), "missing artifact identities")
    protocol = bundle.get("protocol")
    _require(isinstance(protocol, dict), "missing Stage41 protocol")
    _require(protocol.get("handoff_count") == HANDOFF_COUNT, "wrong Stage41 handoff count")
    _require(protocol.get("gr00t_forward_count") == GR00T_FORWARD_COUNT, "wrong Stage41 forward count")
    _require(protocol.get("reference_update_count") == REFERENCE_UPDATE_COUNT, "wrong Stage41 update count")
    _require(protocol.get("native_handoff_schedule") == "ROUND_HANDOFF_S_OVER_OBSERVED_NATIVE_DT", "native schedule is not dynamic")
    _require(protocol.get("executor_prefix_schedule") == "ROUND_HANDOFF_S_OVER_VERIFIED_ACTION_DT", "action prefix schedule is not dynamic")
    budgets = bundle.get("budgets")
    _require(isinstance(budgets, dict), "missing Stage41 budgets")
    _require(budgets.get("gr00t_forward_count") == GR00T_FORWARD_COUNT, "wrong GR00T budget")
    _require(budgets.get("reference_update_count") == REFERENCE_UPDATE_COUNT, "wrong reference-update budget")
    _require(isinstance(budgets.get("wbc_step_count"), int) and budgets["wbc_step_count"] > 0, "invalid WBC budget")
    forbidden = bundle.get("forbidden_operations")
    _require(isinstance(forbidden, dict), "missing forbidden-operation map")
    for name in FORBIDDEN_FALSE_FIELDS:
        _require(forbidden.get(name) is False, "forbidden operation is not false: " + name)
    predecessor = bundle.get("predecessor")
    _require(isinstance(predecessor, dict) and predecessor.get("stage_id") == "M2" and isinstance(predecessor.get("seal_sha256"), str) and len(predecessor["seal_sha256"]) == 64, "Stage41 must bind the accepted M2 seal")


def validate_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_bundle(bundle)
    return {"stage_id": STAGE_ID, "status": "PASS"}


def _verify_m2_predecessor(predecessor_seal: Path, predecessor_evidence_root: Path) -> Dict[str, Any]:
    return stage40.verify_stage_seal(
        Path(predecessor_seal), Path(predecessor_evidence_root), expected_stage="M2", _admit_without_predecessor=True
    )


def _role_path(manifest: Mapping[str, Any], evidence_root: Path, role: str) -> Path:
    matches = [item for item in manifest.get("records", []) if item.get("role") == role]
    _require(len(matches) == 1, "evidence role is not uniquely materialized: " + role)
    return _resolved_child(Path(evidence_root), str(matches[0].get("relative_path")))


def _validate_serialized_run(manifest: Mapping[str, Any], evidence_root: Path, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-check the serialized run, not only its manifest hashes.

    The runner validates live objects before writing evidence.  This second
    pass makes the seal fail closed if a malformed PASS-shaped evidence set is
    assembled later or if a row/lifecycle field is edited before manifesting.
    It intentionally stays standard-library-only so it can run on a login
    node without importing CUDA, NumPy, or the model stack.
    """

    metadata = read_json(_role_path(manifest, evidence_root, "run_metadata"))
    _require(metadata.get("schema") == "CORELAB_STAGE41_LH10_RUN_V1", "wrong Stage41 run metadata schema")
    _require(metadata.get("stage") == STAGE_ID and metadata.get("status") == "PASS", "serialized Stage41 run is not PASS")
    _require(metadata.get("seed") == DEV_SEED and metadata.get("dev_seed") == DEV_SEED, "serialized Stage41 seed mismatch")
    _require(metadata.get("handoff_count") == HANDOFF_COUNT, "serialized Stage41 handoff count mismatch")
    _require(metadata.get("bundle_sha256") == result.get("execution_bundle_sha256"), "serialized run/bundle hash mismatch")
    _require(metadata.get("forbidden_operations") == expected_forbidden_operations(), "serialized run contains forbidden operation")
    _require(metadata.get("error") is None, "serialized PASS run contains an error")
    _require(metadata.get("worker_return_code") == 0, "GR00T worker did not exit cleanly")
    native_dt = metadata.get("native_control_dt_s")
    _require(isinstance(native_dt, (int, float)) and math.isfinite(float(native_dt)) and float(native_dt) > 0.0, "serialized native dt is invalid")

    state = read_json(_role_path(manifest, evidence_root, "state_v1"))
    observations = state.get("observations")
    _require(isinstance(observations, list), "StateV1 observations are missing")
    safety = read_json(_role_path(manifest, evidence_root, "safety_gate"))
    decisions = safety.get("decisions")
    _require(isinstance(decisions, list), "SafetyGate decisions are missing")
    _require(decisions == [str(item.get("safety_gate", {}).get("decision")) for item in observations], "SafetyGate/state observation mismatch")

    handoff_doc = read_json(_role_path(manifest, evidence_root, "handoff"))
    handoffs = handoff_doc.get("events")
    _require(isinstance(handoffs, list), "handoff events are missing")
    completion = handoff_doc.get("completion")
    _require(isinstance(completion, dict), "completion evidence is missing")
    initial_bind = read_json(_role_path(manifest, evidence_root, "initial_bind"))
    initial_events = initial_bind.get("events")
    _require(isinstance(initial_events, list), "initial bind events are missing")
    rows_before_doc = read_json(_role_path(manifest, evidence_root, "rows_before_handoff"))
    rows_after_doc = read_json(_role_path(manifest, evidence_root, "rows_after_handoff"))
    rows_before = rows_before_doc.get("segments")
    rows_after = rows_after_doc.get("segments")
    _require(isinstance(rows_before, list) and isinstance(rows_after, list), "WBC row segments are missing")
    _require(len(rows_before) == HANDOFF_COUNT and len(rows_after) == 1, "serialized LH10 segment topology mismatch")
    all_rows = list(rows_before) + list(rows_after)
    for segment in all_rows:
        _require(isinstance(segment, list) and segment, "serialized WBC segment is empty")
        for row in segment:
            _require(isinstance(row, dict), "serialized WBC row is not an object")
            _require(row.get("done") is False and row.get("reset_buf") is False and row.get("invalid_state") is False, "serialized WBC lifecycle violation")
            _require(str(row.get("termination_code", "NONE")) == "NONE", "serialized WBC termination is not NONE")

    _require(len(observations) == GR00T_FORWARD_COUNT, "serialized LH10 observation count mismatch")
    expected_ids = ["stage41-q%d" % index for index in range(GR00T_FORWARD_COUNT)]
    _require([str(item.get("request_id")) for item in observations] == expected_ids, "serialized observation/request order mismatch")
    timestamps = [float(item.get("observation_timestamp", float("nan"))) for item in observations]
    _require(all(math.isfinite(item) for item in timestamps) and timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps), "serialized observations are stale or reordered")
    state_hashes = [str(item.get("state_v1_sha256", "")) for item in observations]
    _require(all(len(item) == 64 for item in state_hashes) and len(set(state_hashes)) == len(state_hashes), "StateV1 observations are not fresh/distinct")
    for index, observation in enumerate(observations):
        _require(observation.get("body_shape") == [96, 96, 3] and observation.get("body_dtype") == "uint8", "body observation ABI mismatch")
        _require(observation.get("wrist_shape") == [96, 96, 3] and observation.get("wrist_dtype") == "uint8", "wrist observation ABI mismatch")
        _require(observation.get("state_v1_shape") == [10], "StateV1 ABI mismatch")
        _require(observation.get("raw_shape") == [1, 40, 132] and observation.get("decoded_shape") == [40, 10], "action tensor ABI mismatch")
        _require(observation.get("finite") is True, "serialized action/state evidence is non-finite")
        request = observation.get("request")
        response = observation.get("response")
        _require(isinstance(request, dict) and request.get("request_id") == expected_ids[index] and request.get("seed") == DEV_SEED, "worker request identity mismatch")
        _require(isinstance(response, dict) and response.get("status") == "PASS" and response.get("action_contract") == "STRUCTURED_ACTION_V1", "worker response contract mismatch")
        _require(response.get("worker_forward_count_before") == index and response.get("worker_forward_count_after") == index + 1 and response.get("worker_forward_count_delta") == 1, "worker forward counter is not causal at q%d" % index)
        _require(str(observation.get("safety_gate", {}).get("decision")) in {"ACCEPT", "WARN"}, "serialized SafetyGate decision is inadmissible")

    native_steps = metadata.get("native_handoff_steps")
    executor_prefix = metadata.get("executor_prefix_steps")
    action_dt = metadata.get("action_sample_dt_s")
    _require(isinstance(native_steps, list) and isinstance(executor_prefix, list) and isinstance(action_dt, list), "dynamic schedule vectors are missing")
    contract = validate_long_horizon_contract(
        observations,
        handoffs,
        [len(segment) for segment in all_rows],
        [str(item) for item in decisions],
        completion,
        native_dt_s=float(native_dt),
        expected_native_steps=native_steps,
        observation_timestamps=timestamps,
        continuity_events=list(initial_events) + list(handoffs),
    )
    _require(executor_prefix == contract["executor_prefix_steps"], "serialized executor prefix vector mismatch")
    _require(action_dt == contract["action_sample_dt_s"], "serialized action period vector mismatch")
    _require(metadata.get("gr00t_forward_count") == GR00T_FORWARD_COUNT, "serialized GR00T count mismatch")
    _require(metadata.get("reference_update_count") == REFERENCE_UPDATE_COUNT, "serialized reference update count mismatch")
    _require(metadata.get("wbc_step_count") == sum(len(segment) for segment in all_rows), "serialized WBC count mismatch")
    return {"status": "PASS", "native_control_dt_s": float(native_dt), **contract}


def _validate_result_core(result: Mapping[str, Any], bundle: Mapping[str, Any], evidence_root: Path) -> None:
    _validate_bundle(bundle)
    _require(result.get("schema") == RESULT_SCHEMA, "wrong Stage41 result schema")
    _require(result.get("stage_id") == STAGE_ID and result.get("pi_gate") == PI_GATE, "wrong Stage41 result identity")
    _require(result.get("classification") == CLASSIFICATION and result.get("execution_class") == EXECUTION_CLASS, "wrong Stage41 result class")
    _require(result.get("status") == "PASS" and result.get("dev_seed") == DEV_SEED, "only PASS seed-4242 results may be sealed")
    _require(result.get("repository_head") == bundle.get("repository_head"), "result/bundle HEAD mismatch")
    _require(result.get("execution_bundle_sha256") == sha256_bytes(canonical_json_bytes(bundle)), "result/bundle hash mismatch")
    _require(result.get("executable_lineage") == bundle.get("executable_lineage"), "result/bundle lineage mismatch")
    _require(result.get("forbidden_operations") == expected_forbidden_operations(), "result forbidden-operation map mismatch")
    manifest = result.get("evidence_manifest")
    _require(isinstance(manifest, dict), "result evidence manifest missing")
    _require(result.get("evidence_manifest_sha256") == sha256_bytes(canonical_json_bytes(manifest)), "result evidence manifest hash mismatch")
    verify_evidence_manifest(manifest, evidence_root)
    counters = result.get("counters")
    _require(isinstance(counters, dict), "result counters missing")
    verify_counter_observations(counters, manifest, evidence_root)
    decisions = result.get("safety_gate_decisions")
    _require(isinstance(decisions, list) and len(decisions) == SAFETY_DECISION_COUNT and all(item in {"ACCEPT", "WARN"} for item in decisions), "inadmissible SafetyGate vector")
    _require(decisions == read_json(_role_path(manifest, evidence_root, "safety_gate")).get("decisions"), "result/SafetyGate evidence mismatch")
    _validate_serialized_run(manifest, evidence_root, result)
    predecessor = result.get("predecessor")
    _require(isinstance(predecessor, dict) and predecessor.get("stage_id") == "M2", "Stage41 predecessor result binding missing")
    _require(predecessor.get("seal_sha256") == bundle["predecessor"]["seal_sha256"], "Stage41 predecessor hash mismatch")


def build_result_document(
    bundle_path: Path,
    manifest_path: Path,
    evidence_root: Path,
    output_path: Path,
    safety_gate_decisions: Sequence[str],
    predecessor_seal: Path,
    predecessor_evidence_root: Path,
) -> Dict[str, Any]:
    bundle = read_json(bundle_path)
    validate_bundle(bundle)
    manifest = read_json(manifest_path)
    _require(Path(manifest_path).read_bytes() == canonical_json_bytes(manifest), "evidence manifest is not canonical JSON")
    verify_evidence_manifest(manifest, evidence_root)
    predecessor_info = _verify_m2_predecessor(predecessor_seal, predecessor_evidence_root)
    _require(bundle["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "bundle predecessor seal mismatch")
    result = {
        "schema": RESULT_SCHEMA,
        "stage_id": STAGE_ID,
        "pi_gate": PI_GATE,
        "classification": CLASSIFICATION,
        "execution_class": EXECUTION_CLASS,
        "status": "PASS",
        "dev_seed": DEV_SEED,
        "repository_branch": bundle.get("repository_branch"),
        "repository_head": bundle["repository_head"],
        "executable_lineage": bundle["executable_lineage"],
        "execution_bundle_sha256": sha256_bytes(canonical_json_bytes(bundle)),
        "predecessor": {
            "stage_id": "M2",
            "seal_sha256": predecessor_info["seal_sha256"],
            "repository_head": predecessor_info["repository_head"],
            "execution_class": predecessor_info["execution_class"],
            "dev_seed": predecessor_info["dev_seed"],
            "status": predecessor_info["status"],
        },
        "evidence_manifest": manifest,
        "evidence_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "counters": read_json(_resolved_child(evidence_root, next(item["relative_path"] for item in manifest["records"] if item["role"] == "counter_observations"))).get("counters", {}),
        "safety_gate_decisions": [str(item) for item in safety_gate_decisions],
        "forbidden_operations": expected_forbidden_operations(),
    }
    # Store deltas in the compact result, not the before/after evidence rows.
    result["counters"] = {name: result["counters"].get(name, {}).get("delta") for name in COUNTER_NAMES}
    _validate_result_core(result, bundle, evidence_root)
    digest = write_canonical_json(output_path, result)
    return {"stage_id": STAGE_ID, "result_sha256": digest, "path": str(output_path), "status": "PASS"}


def _read_sidecar(seal_path: Path) -> str:
    sidecar = Path(str(seal_path) + ".sha256")
    _require(sidecar.is_file(), "seal SHA sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    _require(len(fields) == 2 and fields[1] == seal_path.name and len(fields[0]) == 64, "invalid seal SHA sidecar")
    return fields[0]


def seal_result(result_path: Path, bundle_path: Path, evidence_root: Path, output_path: Path, predecessor_seal: Path) -> Dict[str, Any]:
    result = read_json(result_path)
    bundle = read_json(bundle_path)
    _require(Path(result_path).read_bytes() == canonical_json_bytes(result), "result is not canonical JSON")
    _require(Path(bundle_path).read_bytes() == canonical_json_bytes(bundle), "bundle is not canonical JSON")
    _validate_result_core(result, bundle, evidence_root)
    predecessor_root = Path(evidence_root).parent / "M2"
    predecessor_info = _verify_m2_predecessor(predecessor_seal, predecessor_root)
    _require(result["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "result predecessor seal mismatch")
    seal = {
        "schema": SEAL_SCHEMA,
        "stage_id": STAGE_ID,
        "pi_gate": PI_GATE,
        "classification": CLASSIFICATION,
        "execution_bundle_sha256": sha256_bytes(canonical_json_bytes(bundle)),
        "result_sha256": sha256_bytes(canonical_json_bytes(result)),
        "execution_bundle": bundle,
        "result": result,
    }
    digest = write_canonical_json(output_path, seal)
    Path(str(output_path) + ".sha256").write_text("{}  {}\n".format(digest, Path(output_path).name), encoding="utf-8")
    verify_stage_seal(output_path, evidence_root, predecessor_seal=predecessor_seal)
    return {"stage_id": STAGE_ID, "seal_sha256": digest, "path": str(output_path), "status": "PASS"}


def verify_stage_seal(seal_path: Path, evidence_root: Path, expected_head: Optional[str] = None, predecessor_seal: Optional[Path] = None) -> Dict[str, Any]:
    seal_path = Path(seal_path)
    expected_digest = _read_sidecar(seal_path)
    actual_digest = sha256_file(seal_path)
    _require(actual_digest == expected_digest, "seal SHA mismatch")
    seal = read_json(seal_path)
    _require(seal_path.read_bytes() == canonical_json_bytes(seal), "seal is not canonical JSON")
    _require(seal.get("schema") == SEAL_SCHEMA and seal.get("stage_id") == STAGE_ID, "wrong Stage41 seal identity")
    bundle, result = seal.get("execution_bundle"), seal.get("result")
    _require(isinstance(bundle, dict) and isinstance(result, dict), "seal lacks bundle/result")
    _require(seal.get("execution_bundle_sha256") == sha256_bytes(canonical_json_bytes(bundle)), "embedded bundle SHA mismatch")
    _require(seal.get("result_sha256") == sha256_bytes(canonical_json_bytes(result)), "embedded result SHA mismatch")
    if expected_head is not None:
        _require(result.get("repository_head") == expected_head, "unexpected Stage41 repository HEAD")
    _validate_result_core(result, bundle, Path(evidence_root))
    predecessor_root = Path(evidence_root).parent / "M2"
    if predecessor_seal is not None:
        predecessor_info = _verify_m2_predecessor(predecessor_seal, predecessor_root)
        _require(result["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "predecessor seal binding mismatch")
    return {
        "stage_id": STAGE_ID,
        "seal_sha256": actual_digest,
        "repository_head": result["repository_head"],
        "execution_class": result["execution_class"],
        "dev_seed": result["dev_seed"],
        "status": "PASS",
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--evidence-root", type=Path, required=True)
    manifest.add_argument("--roles-json", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    result = sub.add_parser("build-result")
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--safety-decisions", required=True)
    result.add_argument("--predecessor-seal", type=Path, required=True)
    result.add_argument("--predecessor-evidence-root", type=Path, required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--result", type=Path, required=True)
    seal.add_argument("--bundle", type=Path, required=True)
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--predecessor-seal", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--seal", type=Path, required=True)
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--expected-head")
    verify.add_argument("--predecessor-seal", type=Path)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        value = build_evidence_manifest(args.evidence_root, read_json(args.roles_json))
        digest = write_canonical_json(args.output, value)
        value = {"stage_id": STAGE_ID, "manifest_sha256": digest, "path": str(args.output)}
    elif args.command == "build-result":
        value = build_result_document(args.bundle, args.manifest, args.evidence_root, args.output, args.safety_decisions.split(","), args.predecessor_seal, args.predecessor_evidence_root)
    elif args.command == "seal":
        value = seal_result(args.result, args.bundle, args.evidence_root, args.output, args.predecessor_seal)
    else:
        value = verify_stage_seal(args.seal, args.evidence_root, args.expected_head, args.predecessor_seal)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
