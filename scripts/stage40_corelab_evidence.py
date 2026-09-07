#!/usr/bin/env python3
"""Fail-closed Stage40 Corelab execution bundles and canonical result seals.

This module is deliberately CPU-only and standard-library-only.  It binds raw
evidence, observed monotonic counters, execution lineage, and the immediately
preceding stage seal before a result can be sealed.  It never constructs a
model, imports a simulator, submits a scheduler job, or changes the scientific
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PI_GATE = "PI_GATE_STATE_0064"
ISSUE_COMMENT = 5574239666
DEV_SEED = 4242
CLASSIFICATION = "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE"

STAGES = ("RTX-runtime", "RTX-Q0", "RTX-bind", "M1", "M2")
PREDECESSOR = {
    "RTX-runtime": None,
    "RTX-Q0": "RTX-runtime",
    "RTX-bind": "RTX-Q0",
    "M1": "RTX-bind",
    "M2": "M1",
}
EXECUTION_CLASS = {
    "RTX-runtime": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
    "RTX-Q0": "CORELAB_STAGE40_EXACT_Q0_V1",
    "RTX-bind": "CORELAB_STAGE40_INITIAL_BIND_V1",
    "M1": "CORELAB_STAGE40_ONE_REPLAN_CLOSED_LOOP_V1",
    "M2": "CORELAB_STAGE40_THREE_HANDOFF_RECEDING_HORIZON_V1",
}
RESULT_SCHEMA = {
    stage: "CORELAB_STAGE40_{}_RESULT_V1".format(stage.upper().replace("-", "_"))
    for stage in STAGES
}

COUNTER_NAMES = ("gr00t_forward_count", "wbc_step_count", "reference_update_count")
COUNTER_EXPECTATIONS = {
    "RTX-runtime": {"gr00t_forward_count": (0, 0), "wbc_step_count": (0, 0), "reference_update_count": (0, 0)},
    "RTX-Q0": {"gr00t_forward_count": (1, 1), "wbc_step_count": (0, 0), "reference_update_count": (0, 0)},
    "RTX-bind": {"gr00t_forward_count": (1, 1), "wbc_step_count": (0, 0), "reference_update_count": (0, 0)},
    "M1": {"gr00t_forward_count": (2, 2), "wbc_step_count": (1, None), "reference_update_count": (1, 1)},
    "M2": {"gr00t_forward_count": (4, 4), "wbc_step_count": (1, None), "reference_update_count": (3, 3)},
}
GR00T_BUDGET = {"RTX-runtime": 0, "RTX-Q0": 1, "RTX-bind": 1, "M1": 2, "M2": 4}
REFERENCE_BUDGET = {"RTX-runtime": 0, "RTX-Q0": 0, "RTX-bind": 0, "M1": 1, "M2": 3}
SAFETY_DECISION_COUNT = {"RTX-runtime": 0, "RTX-Q0": 1, "RTX-bind": 1, "M1": 2, "M2": 4}

FORBIDDEN_FALSE_FIELDS = (
    "training",
    "formal_seed_used",
    "validation_e_read",
    "holdout_f_read",
    "hardware_executed",
    "h200_used",
    "checkpoint_substitution",
    "model_substitution",
    "processor_substitution",
    "precision_change",
    "quantization",
    "cpu_offload",
    "clipping",
    "smoothing",
    "blending",
    "adaptive_retiming",
    "physics_reset",
    "physics_teleport",
    "policy_history_clear",
)

REQUIRED_EVIDENCE_ROLES = {
    "RTX-runtime": {
        "observation_body", "observation_wrist", "state_v1", "native_runtime",
        "model_runtime", "counter_observations", "logs", "telemetry",
    },
    "RTX-Q0": {
        "observation_body", "observation_wrist", "state_v1", "worker_request",
        "worker_response", "raw_action_d132", "decoded_action_d10_h40",
        "safety_gate", "counter_observations", "mailbox", "logs", "telemetry",
    },
    "RTX-bind": {
        "initial_bind", "root_state", "dof_state", "executor_bookkeeping",
        "safety_gate", "counter_observations", "logs", "telemetry",
    },
    "M1": {
        "observation_body", "observation_wrist", "state_v1", "worker_request",
        "worker_response", "raw_action_d132", "decoded_action_d10_h40",
        "safety_gate", "initial_bind", "root_state", "dof_state",
        "executor_bookkeeping", "rows_before_handoff", "rows_after_handoff",
        "handoff", "counter_observations", "mailbox", "logs", "telemetry",
    },
    "M2": {
        "observation_body", "observation_wrist", "state_v1", "worker_request",
        "worker_response", "raw_action_d132", "decoded_action_d10_h40",
        "safety_gate", "initial_bind", "root_state", "dof_state",
        "executor_bookkeeping", "rows_before_handoff", "rows_after_handoff",
        "handoff", "counter_observations", "mailbox", "logs", "telemetry",
    },
}


class Stage40EvidenceError(ValueError):
    """Raised when a Stage40 execution/evidence boundary does not verify."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage40EvidenceError(message)


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
    _require(isinstance(value, dict), "JSON object required: {}".format(path))
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
        raise Stage40EvidenceError("evidence path escapes evidence root") from exc
    _require(candidate.is_file(), "evidence is not a regular file")
    return candidate


def build_evidence_manifest(stage_id: str, evidence_root: Path, role_paths: Mapping[str, str]) -> Dict[str, Any]:
    _require(stage_id in STAGES, "unknown Stage40 stage")
    roles = {str(key): str(value) for key, value in role_paths.items()}
    missing = sorted(REQUIRED_EVIDENCE_ROLES[stage_id] - set(roles))
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
        "schema": "CORELAB_STAGE40_EVIDENCE_MANIFEST_V1",
        "stage_id": stage_id,
        "records": records,
        "record_count": len(records),
    }


def verify_evidence_manifest(manifest: Mapping[str, Any], evidence_root: Path) -> Dict[str, Any]:
    _require(manifest.get("schema") == "CORELAB_STAGE40_EVIDENCE_MANIFEST_V1", "wrong evidence manifest schema")
    stage_id = str(manifest.get("stage_id"))
    _require(stage_id in STAGES, "wrong evidence manifest stage")
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
    missing = sorted(REQUIRED_EVIDENCE_ROLES[stage_id] - set(roles))
    _require(not missing, "evidence manifest omits required roles: " + ", ".join(missing))
    return {"stage_id": stage_id, "record_count": len(records), "status": "PASS"}


def expected_forbidden_operations() -> Dict[str, bool]:
    return {name: False for name in FORBIDDEN_FALSE_FIELDS}


def build_counter_observation(
    before: Mapping[str, int],
    after: Mapping[str, int],
    gr00t_requests: Optional[Sequence[int]] = None,
    wbc_segments: Optional[Sequence[int]] = None,
    reference_updates: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    counters = {}
    for name in COUNTER_NAMES:
        left = before.get(name)
        right = after.get(name)
        _require(isinstance(left, int) and isinstance(right, int), "counter values must be observed integers")
        _require(left >= 0 and right >= left, "counters must be non-negative and monotonic")
        counters[name] = {"before": left, "after": right, "delta": right - left}
    return {
        "schema": "CORELAB_STAGE40_COUNTER_OBSERVATION_V1",
        "counters": counters,
        "gr00t_request_deltas": list(gr00t_requests or []),
        "wbc_segment_deltas": list(wbc_segments or []),
        "reference_update_deltas": list(reference_updates or []),
    }


def _counter_record_path(manifest: Mapping[str, Any], evidence_root: Path) -> Path:
    records = [item for item in manifest.get("records", []) if item.get("role") == "counter_observations"]
    _require(len(records) == 1, "exactly one counter observation record is required")
    return _resolved_child(Path(evidence_root), str(records[0].get("relative_path")))


def verify_counter_observations(stage_id: str, counters: Mapping[str, Any], manifest: Mapping[str, Any], evidence_root: Path) -> Dict[str, int]:
    observation = read_json(_counter_record_path(manifest, evidence_root))
    _require(observation.get("schema") == "CORELAB_STAGE40_COUNTER_OBSERVATION_V1", "wrong counter evidence schema")
    observed = observation.get("counters")
    _require(isinstance(observed, dict), "counter observations missing")
    deltas = {}
    for name in COUNTER_NAMES:
        row = observed.get(name)
        _require(isinstance(row, dict), "missing observed counter: " + name)
        before = row.get("before")
        after = row.get("after")
        delta = row.get("delta")
        _require(isinstance(before, int) and isinstance(after, int) and isinstance(delta, int), "counter observation must use integers")
        _require(before >= 0 and after >= before and delta == after - before, "forged/non-monotonic counter evidence: " + name)
        _require(counters.get(name) == delta, "result counter differs from observed delta: " + name)
        lower, upper = COUNTER_EXPECTATIONS[stage_id][name]
        _require(delta >= lower and (upper is None or delta <= upper), "counter budget mismatch: " + name)
        deltas[name] = delta
    expected_forwards = COUNTER_EXPECTATIONS[stage_id]["gr00t_forward_count"][0]
    expected_updates = COUNTER_EXPECTATIONS[stage_id]["reference_update_count"][0]
    request_deltas = observation.get("gr00t_request_deltas")
    update_deltas = observation.get("reference_update_deltas")
    wbc_deltas = observation.get("wbc_segment_deltas")
    _require(
        isinstance(request_deltas, list)
        and request_deltas == [1] * expected_forwards
        and sum(request_deltas) == deltas["gr00t_forward_count"],
        "per-request GR00T counter evidence mismatch",
    )
    _require(
        isinstance(update_deltas, list)
        and update_deltas == [1] * expected_updates
        and sum(update_deltas) == deltas["reference_update_count"],
        "per-update reference counter evidence mismatch",
    )
    expected_wbc_segments = 2 if stage_id == "M1" else (4 if stage_id == "M2" else 0)
    _require(
        isinstance(wbc_deltas, list)
        and len(wbc_deltas) == expected_wbc_segments
        and all(isinstance(item, int) and item > 0 for item in wbc_deltas)
        and sum(wbc_deltas) == deltas["wbc_step_count"],
        "per-segment WBC counter evidence mismatch",
    )
    return deltas


def _validate_stage_id(stage_id: str) -> None:
    _require(stage_id in STAGES, "unknown Stage40 stage")


def validate_bundle(bundle: Mapping[str, Any], expected_stage: Optional[str] = None) -> Dict[str, Any]:
    _require(bundle.get("schema") == "CORELAB_STAGE40_EXECUTION_BUNDLE_V1", "wrong execution bundle schema")
    stage_id = str(bundle.get("stage_id"))
    _validate_stage_id(stage_id)
    if expected_stage is not None:
        _require(stage_id == expected_stage, "wrong execution bundle stage")
    _require(bundle.get("pi_gate") == PI_GATE, "wrong PI gate")
    _require(bundle.get("classification") == CLASSIFICATION, "wrong execution classification")
    _require(bundle.get("execution_class") == EXECUTION_CLASS[stage_id], "wrong execution class")
    _require(bundle.get("dev_seed") == DEV_SEED, "wrong development seed")
    _require(isinstance(bundle.get("repository_head"), str) and len(bundle["repository_head"]) == 40, "invalid repository HEAD")
    lineage = bundle.get("executable_lineage")
    _require(isinstance(lineage, dict) and bool(lineage.get("runner_sha256")) and bool(lineage.get("source_sha256")), "missing executable lineage")
    _require(isinstance(bundle.get("artifact_identities"), dict) and bool(bundle["artifact_identities"]), "missing artifact identities")
    budgets = bundle.get("budgets")
    _require(isinstance(budgets, dict), "missing execution budgets")
    for name in COUNTER_NAMES:
        _require(name in budgets and isinstance(budgets[name], int) and budgets[name] >= 0, "invalid counter budget: " + name)
    _require(budgets["gr00t_forward_count"] == GR00T_BUDGET[stage_id], "wrong GR00T forward budget")
    _require(budgets["reference_update_count"] == REFERENCE_BUDGET[stage_id], "wrong reference update budget")
    if stage_id in {"RTX-runtime", "RTX-Q0", "RTX-bind"}:
        _require(budgets["wbc_step_count"] == 0, "WBC budget must be zero before M1")
    else:
        _require(budgets["wbc_step_count"] > 0, "M1/M2 require a positive bounded WBC budget")
    forbidden = bundle.get("forbidden_operations")
    _require(isinstance(forbidden, dict), "missing forbidden operation map")
    for name in FORBIDDEN_FALSE_FIELDS:
        _require(forbidden.get(name) is False, "forbidden operation is not false: " + name)
    predecessor = bundle.get("predecessor")
    expected_predecessor = PREDECESSOR[stage_id]
    if expected_predecessor is None:
        _require(predecessor is None, "root stage must not bind a predecessor")
    else:
        _require(isinstance(predecessor, dict), "missing predecessor bundle binding")
        _require(predecessor.get("stage_id") == expected_predecessor, "wrong predecessor stage in bundle")
        _require(isinstance(predecessor.get("seal_sha256"), str) and len(predecessor["seal_sha256"]) == 64, "missing predecessor seal hash")
    return {"stage_id": stage_id, "status": "PASS"}


def freeze_bundle(spec_path: Path, output_path: Path) -> Dict[str, Any]:
    spec = read_json(Path(spec_path))
    validate_bundle(spec)
    digest = write_canonical_json(Path(output_path), spec)
    return {"stage_id": spec["stage_id"], "bundle_sha256": digest, "path": str(output_path)}


def build_result_document(
    bundle_path: Path,
    manifest_path: Path,
    evidence_root: Path,
    output_path: Path,
    safety_gate_decisions: Sequence[str],
    predecessor_seal: Optional[Path] = None,
    predecessor_evidence_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create the canonical result from already-written, observed evidence."""

    bundle = read_json(Path(bundle_path))
    validate_bundle(bundle)
    stage_id = bundle["stage_id"]
    manifest = read_json(Path(manifest_path))
    _require(Path(manifest_path).read_bytes() == canonical_json_bytes(manifest), "evidence manifest is not canonical JSON")
    verify_evidence_manifest(manifest, Path(evidence_root))
    _require(manifest.get("stage_id") == stage_id, "bundle/evidence stage mismatch")
    counter_observation = read_json(_counter_record_path(manifest, Path(evidence_root)))
    observed = counter_observation.get("counters")
    _require(isinstance(observed, dict), "counter observations missing")
    counters = {name: observed.get(name, {}).get("delta") for name in COUNTER_NAMES}
    predecessor = None
    expected_predecessor = PREDECESSOR[stage_id]
    if expected_predecessor is None:
        _require(predecessor_seal is None, "root stage does not accept a predecessor")
    else:
        _require(predecessor_seal is not None and predecessor_evidence_root is not None, "predecessor seal and evidence root are required")
        info = verify_stage_seal(
            Path(predecessor_seal),
            Path(predecessor_evidence_root),
            expected_stage=expected_predecessor,
            _admit_without_predecessor=True,
        )
        _require(bundle["predecessor"]["seal_sha256"] == info["seal_sha256"], "bundle predecessor seal mismatch")
        predecessor = {
            "stage_id": info["stage_id"],
            "seal_sha256": info["seal_sha256"],
            "repository_head": info["repository_head"],
            "execution_class": info["execution_class"],
            "dev_seed": info["dev_seed"],
            "status": info["status"],
        }
    result = {
        "schema": RESULT_SCHEMA[stage_id],
        "stage_id": stage_id,
        "pi_gate": PI_GATE,
        "classification": CLASSIFICATION,
        "status": "PASS",
        "dev_seed": DEV_SEED,
        "repository_head": bundle["repository_head"],
        "execution_class": bundle["execution_class"],
        "executable_lineage": bundle["executable_lineage"],
        "execution_bundle_sha256": sha256_bytes(canonical_json_bytes(bundle)),
        "predecessor": predecessor,
        "evidence_manifest": manifest,
        "evidence_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "counters": counters,
        "safety_gate_decisions": [str(item) for item in safety_gate_decisions],
        "forbidden_operations": expected_forbidden_operations(),
    }
    _validate_result_core(result, bundle, Path(evidence_root))
    digest = write_canonical_json(Path(output_path), result)
    return {"stage_id": stage_id, "result_sha256": digest, "path": str(output_path), "status": "PASS"}


def _read_sidecar(seal_path: Path) -> str:
    sidecar = Path(str(seal_path) + ".sha256")
    _require(sidecar.is_file(), "predecessor seal SHA sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    _require(len(fields) == 2 and fields[1] == seal_path.name, "invalid seal SHA sidecar")
    _require(len(fields[0]) == 64, "invalid seal SHA value")
    return fields[0]


def _validate_result_core(result: Mapping[str, Any], bundle: Mapping[str, Any], evidence_root: Path) -> Dict[str, Any]:
    stage_id = str(bundle["stage_id"])
    _require(result.get("schema") == RESULT_SCHEMA[stage_id], "wrong inner result schema")
    _require(result.get("stage_id") == stage_id, "wrong result stage")
    _require(result.get("pi_gate") == PI_GATE, "wrong result PI gate")
    _require(result.get("classification") == CLASSIFICATION, "wrong result classification")
    _require(result.get("status") == "PASS", "only PASS results may be sealed")
    _require(result.get("dev_seed") == DEV_SEED, "wrong result seed")
    _require(result.get("repository_head") == bundle["repository_head"], "result/bundle HEAD mismatch")
    _require(result.get("execution_class") == EXECUTION_CLASS[stage_id], "wrong result execution class")
    _require(result.get("executable_lineage") == bundle["executable_lineage"], "result/bundle executable lineage mismatch")
    _require(result.get("execution_bundle_sha256") == sha256_bytes(canonical_json_bytes(bundle)), "execution bundle hash mismatch")
    forbidden = result.get("forbidden_operations")
    _require(isinstance(forbidden, dict), "result forbidden operation map missing")
    for name in FORBIDDEN_FALSE_FIELDS:
        _require(forbidden.get(name) is False, "result records forbidden operation: " + name)
    manifest = result.get("evidence_manifest")
    _require(isinstance(manifest, dict), "result evidence manifest missing")
    _require(manifest.get("stage_id") == stage_id, "result evidence stage mismatch")
    _require(result.get("evidence_manifest_sha256") == sha256_bytes(canonical_json_bytes(manifest)), "evidence manifest hash mismatch")
    verify_evidence_manifest(manifest, evidence_root)
    counters = result.get("counters")
    _require(isinstance(counters, dict), "result counters missing")
    verified_counters = verify_counter_observations(stage_id, counters, manifest, evidence_root)
    decisions = result.get("safety_gate_decisions")
    _require(isinstance(decisions, list), "SafetyGate decisions must be a list")
    _require(len(decisions) == SAFETY_DECISION_COUNT[stage_id], "wrong SafetyGate decision count")
    _require(all(item in {"ACCEPT", "WARN"} for item in decisions), "SafetyGate REJECT cannot be sealed")
    return {"stage_id": stage_id, "counters": verified_counters, "status": "PASS"}


def verify_stage_seal(
    seal_path: Path,
    evidence_root: Path,
    expected_stage: Optional[str] = None,
    expected_head: Optional[str] = None,
    predecessor_seal: Optional[Path] = None,
    _admit_without_predecessor: bool = False,
) -> Dict[str, Any]:
    seal_path = Path(seal_path)
    expected_digest = _read_sidecar(seal_path)
    actual_digest = sha256_file(seal_path)
    _require(actual_digest == expected_digest, "seal SHA mismatch")
    seal = read_json(seal_path)
    _require(seal_path.read_bytes() == canonical_json_bytes(seal), "seal is not canonical JSON")
    _require(seal.get("schema") == "CORELAB_STAGE40_STAGE_SEAL_V1", "wrong seal schema")
    result = seal.get("result")
    bundle = seal.get("execution_bundle")
    _require(isinstance(result, dict) and isinstance(bundle, dict), "seal lacks embedded result/bundle")
    info = validate_bundle(bundle, expected_stage=expected_stage)
    stage_id = info["stage_id"]
    _require(seal.get("stage_id") == stage_id, "seal stage identity mismatch")
    _require(seal.get("result_sha256") == sha256_bytes(canonical_json_bytes(result)), "inner result SHA mismatch")
    _require(seal.get("execution_bundle_sha256") == sha256_bytes(canonical_json_bytes(bundle)), "embedded bundle SHA mismatch")
    if expected_head is not None:
        _require(result.get("repository_head") == expected_head, "unexpected repository HEAD")
    expected_predecessor = PREDECESSOR[stage_id]
    if expected_predecessor is None:
        _require(predecessor_seal is None, "root stage does not accept a predecessor")
        _require(result.get("predecessor") is None, "root result must not bind a predecessor")
    else:
        binding = result.get("predecessor")
        _require(isinstance(binding, dict), "result predecessor binding missing")
        _require(binding.get("stage_id") == expected_predecessor, "wrong result predecessor stage")
        _require(binding.get("seal_sha256") == bundle["predecessor"]["seal_sha256"], "result/bundle predecessor seal mismatch")
        _require(binding.get("execution_class") == EXECUTION_CLASS[expected_predecessor], "predecessor execution class mismatch")
        _require(isinstance(binding.get("repository_head"), str) and len(binding["repository_head"]) == 40, "predecessor HEAD is invalid")
        _require(binding.get("dev_seed") == DEV_SEED, "predecessor seed mismatch")
        _require(binding.get("status") == "PASS", "predecessor status mismatch")
        if predecessor_seal is None:
            _require(_admit_without_predecessor, "predecessor seal is required")
        else:
            predecessor_info = verify_stage_seal(
                Path(predecessor_seal),
                Path(evidence_root).parent / expected_predecessor,
                expected_stage=expected_predecessor,
                _admit_without_predecessor=True,
            )
            _require(binding.get("seal_sha256") == predecessor_info["seal_sha256"], "predecessor seal binding mismatch")
            _require(bundle["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "bundle predecessor seal binding mismatch")
            _require(binding.get("repository_head") == predecessor_info["repository_head"], "predecessor HEAD binding mismatch")
            _require(binding.get("execution_class") == predecessor_info["execution_class"], "predecessor execution class mismatch")
    _validate_result_core(result, bundle, Path(evidence_root))
    return {
        "stage_id": stage_id,
        "seal_sha256": actual_digest,
        "repository_head": result["repository_head"],
        "execution_class": result["execution_class"],
        "dev_seed": result["dev_seed"],
        "status": "PASS",
    }


def seal_result(
    result_path: Path,
    bundle_path: Path,
    evidence_root: Path,
    output_path: Path,
    predecessor_seal: Optional[Path] = None,
) -> Dict[str, Any]:
    result = read_json(Path(result_path))
    bundle = read_json(Path(bundle_path))
    _require(Path(result_path).read_bytes() == canonical_json_bytes(result), "result is not canonical JSON")
    _require(Path(bundle_path).read_bytes() == canonical_json_bytes(bundle), "execution bundle is not canonical JSON")
    validate_bundle(bundle)
    _validate_result_core(result, bundle, Path(evidence_root))
    expected_predecessor = PREDECESSOR[bundle["stage_id"]]
    if expected_predecessor is None:
        _require(predecessor_seal is None, "root stage does not accept a predecessor")
        _require(result.get("predecessor") is None, "root result must not bind a predecessor")
    else:
        _require(predecessor_seal is not None, "predecessor seal is required")
        predecessor_info = verify_stage_seal(
            Path(predecessor_seal),
            Path(evidence_root).parent / expected_predecessor,
            expected_stage=expected_predecessor,
            _admit_without_predecessor=True,
        )
        _require(predecessor_info["stage_id"] == expected_predecessor, "wrong predecessor stage")
        binding = result.get("predecessor")
        _require(isinstance(binding, dict), "result predecessor binding missing")
        _require(binding.get("stage_id") == expected_predecessor, "wrong result predecessor stage")
        _require(binding.get("seal_sha256") == predecessor_info["seal_sha256"], "predecessor seal binding mismatch")
        _require(bundle["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "bundle predecessor seal mismatch")
        _require(binding.get("repository_head") == predecessor_info["repository_head"], "predecessor HEAD mismatch")
        _require(binding.get("execution_class") == predecessor_info["execution_class"], "predecessor execution class mismatch")
        _require(binding.get("dev_seed") == DEV_SEED and binding.get("status") == "PASS", "predecessor result is inadmissible")
    seal = {
        "schema": "CORELAB_STAGE40_STAGE_SEAL_V1",
        "stage_id": bundle["stage_id"],
        "pi_gate": PI_GATE,
        "classification": CLASSIFICATION,
        "execution_bundle_sha256": sha256_bytes(canonical_json_bytes(bundle)),
        "result_sha256": sha256_bytes(canonical_json_bytes(result)),
        "execution_bundle": bundle,
        "result": result,
    }
    digest = write_canonical_json(Path(output_path), seal)
    sidecar = Path(str(output_path) + ".sha256")
    sidecar.write_text("{}  {}\n".format(digest, Path(output_path).name), encoding="utf-8")
    verify_stage_seal(Path(output_path), Path(evidence_root), predecessor_seal=predecessor_seal)
    return {"stage_id": bundle["stage_id"], "seal_sha256": digest, "path": str(output_path), "status": "PASS"}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-bundle")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--stage", choices=STAGES, required=True)
    manifest.add_argument("--evidence-root", type=Path, required=True)
    manifest.add_argument("--roles-json", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    result = subparsers.add_parser("build-result")
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--safety-decisions", default="")
    result.add_argument("--predecessor-seal", type=Path)
    result.add_argument("--predecessor-evidence-root", type=Path)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--result", type=Path, required=True)
    seal.add_argument("--bundle", type=Path, required=True)
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--predecessor-seal", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--seal", type=Path, required=True)
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--expected-stage", choices=STAGES)
    verify.add_argument("--expected-head")
    verify.add_argument("--predecessor-seal", type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze-bundle":
        value = freeze_bundle(args.spec, args.output)
    elif args.command == "manifest":
        role_paths = read_json(args.roles_json)
        value = build_evidence_manifest(args.stage, args.evidence_root, role_paths)
        digest = write_canonical_json(args.output, value)
        value = {"stage_id": args.stage, "manifest_sha256": digest, "path": str(args.output)}
    elif args.command == "build-result":
        decisions = [] if not args.safety_decisions else args.safety_decisions.split(",")
        value = build_result_document(
            args.bundle,
            args.manifest,
            args.evidence_root,
            args.output,
            decisions,
            args.predecessor_seal,
            args.predecessor_evidence_root,
        )
    elif args.command == "seal":
        value = seal_result(args.result, args.bundle, args.evidence_root, args.output, args.predecessor_seal)
    else:
        value = verify_stage_seal(args.seal, args.evidence_root, args.expected_stage, args.expected_head, args.predecessor_seal)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
