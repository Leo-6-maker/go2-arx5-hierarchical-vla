#!/usr/bin/env python3
"""CPU-only verifier and characterizer for the Stage43 cohort.

The five GPU runs are immutable inputs.  This script never imports IsaacGym,
CUDA, GR00T, or a checkpoint; it verifies every per-condition manifest/seal,
reconstructs the continuous target metrics, checks hard URDF limits separately
from the soft diagnostic flag, and emits the PI stop-gate result.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import stage43_corelab_evidence as evidence


STAGE = evidence.STAGE_ID
GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
CONDITION_RESULT_SCHEMA = "CORELAB_STAGE43_CONDITION_RESULT_V1"
COHORT_MANIFEST_SCHEMA = "CORELAB_STAGE43_COHORT_EVIDENCE_MANIFEST_V1"
RESULT_SEAL_SCHEMA = "CORELAB_STAGE43_RESULT_SEAL_V1"


class FinalizationError(ValueError):
    """Raised when a Stage43 condition or cohort is not evidence-complete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def _read_canonical_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "JSON evidence is not a regular file: %s" % path)
    payload = evidence.read_json(path)
    _require(path.read_bytes() == evidence.canonical_json_bytes(payload), "JSON evidence is not canonical: %s" % path)
    return payload


def _finite(value: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(tuple(array.shape) == shape and bool(np.isfinite(array).all()), "%s must be finite %s" % (label, shape))
    return array


def _close(actual: Any, expected: Any, label: str, atol: float = 1e-9) -> None:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    _require(a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol)), "%s mismatch: %s != %s" % (label, a.tolist(), b.tolist()))


def _load_npz(path: Path, expected_ids: Sequence[str], shape: Tuple[int, ...], label: str) -> Dict[str, np.ndarray]:
    path = Path(path).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "%s is missing" % label)
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(expected_ids), "%s query set mismatch" % label)
        values = {key: np.asarray(archive[key]) for key in expected_ids}
    for key, value in values.items():
        _require(tuple(value.shape) == shape and bool(np.isfinite(value).all()), "%s %s has invalid shape/finiteness" % (label, key))
    return values


def _hard_limits(source_root: Path) -> Tuple[Path, np.ndarray, List[str]]:
    path = Path(source_root) / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"
    path = path.resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "Stage43 URDF is missing")
    root = ET.parse(str(path)).getroot()
    limits: List[Tuple[float, float]] = []
    names: List[str] = []
    for element in root.findall(".//joint"):
        if element.get("type") == "fixed":
            continue
        limit = element.find("limit")
        _require(limit is not None and limit.get("lower") is not None and limit.get("upper") is not None, "URDF limit is incomplete: %s" % element.get("name"))
        limits.append((float(limit.get("lower")), float(limit.get("upper"))))
        names.append(str(element.get("name")))
    _require(len(limits) == 20, "expected 20 non-fixed URDF joints, got %d" % len(limits))
    return path, np.asarray(limits, dtype=np.float64), names


def _check_forbidden(payload: Mapping[str, Any], label: str) -> None:
    _require(payload.get("forbidden_operations") == evidence.expected_forbidden_operations(), "%s forbidden-operation map drift" % label)


def _check_rows(root: Path, expected_ids: Sequence[str], target: np.ndarray, hard: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Any]]:
    before = _read_canonical_json(root / "rows_before_handoff.json")
    after = _read_canonical_json(root / "rows_after_handoff.json")
    before_segments = before.get("segments")
    after_segments = after.get("segments")
    _require(isinstance(before_segments, list) and isinstance(after_segments, list), "row segment containers are missing")
    _require([len(segment) for segment in before_segments] == list(evidence.COHORT_SEGMENT_SCHEDULE[:3]), "pre-completion segment schedule mismatch")
    _require([len(segment) for segment in after_segments] == [evidence.COHORT_SEGMENT_SCHEDULE[3]], "completion segment schedule mismatch")
    rows = [row for segment in list(before_segments) + list(after_segments) for row in segment]
    _require(len(rows) == sum(evidence.COHORT_SEGMENT_SCHEDULE), "row count mismatch")
    achieved: List[np.ndarray] = []
    desired: List[np.ndarray] = []
    hard_margin_cells = 0
    soft_flag_rows = 0
    termination_rows = 0
    reset_rows = 0
    invalid_rows = 0
    for row in rows:
        _require(isinstance(row, dict), "row is not an object")
        achieved.append(_finite(row.get("achieved_pos"), (3,), "achieved_pos"))
        desired.append(_finite(row.get("desired_pos"), (3,), "desired_pos"))
        q = _finite(row.get("dof_pos"), (20,), "dof_pos")
        _finite(row.get("joint_limit_margin"), (20,), "joint_limit_margin")
        hard_margin = np.minimum(q - hard[:, 0], hard[:, 1] - q)
        hard_margin_cells += int(np.count_nonzero(hard_margin < 0.0))
        soft_flag_rows += int(bool(row.get("joint_limit_violation")))
        termination_rows += int(str(row.get("termination_code", "NONE")) != "NONE")
        reset_rows += int(bool(row.get("reset_buf")) or bool(row.get("done")))
        invalid_rows += int(bool(row.get("invalid_state")))
        if "joint_limit_termination" in row:
            _require(not bool(row.get("joint_limit_termination")), "hard joint-limit termination was reported")
    _require(hard_margin_cells == 0, "hard URDF joint-limit violation observed (%d cells)" % hard_margin_cells)
    _require(termination_rows == 0 and reset_rows == 0 and invalid_rows == 0, "row termination/reset/nonfinite state observed")
    positions = np.asarray(achieved, dtype=np.float64)
    desired_array = np.asarray(desired, dtype=np.float64)
    return positions, {"hard_limit_violation_cell_count": hard_margin_cells, "soft_diagnostic_flag_row_count": soft_flag_rows, "termination_row_count": termination_rows, "reset_or_done_row_count": reset_rows, "invalid_state_row_count": invalid_rows}, {"desired": desired_array, "rows": rows}


def _metric_reconstruction(setup: Mapping[str, Any], positions: np.ndarray, desired: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    initial = _finite(setup["s0"]["eef_world_m"], (3,), "S0 EEF")
    all_positions = np.vstack((initial[None, :], positions))
    distances = np.linalg.norm(all_positions - target[None, :], axis=1)
    tracking = np.linalg.norm(positions - desired, axis=1)
    _require(bool(np.isfinite(distances).all() and np.isfinite(tracking).all()), "reconstructed target metric is non-finite")
    index = int(np.argmin(distances))
    d0 = float(distances[0]); dmin = float(distances[index]); dfinal = float(distances[-1])
    return {
        "sample_count_including_S0": int(len(distances)),
        "initial_target_error_m": d0,
        "minimum_target_error_m": dmin,
        "minimum_error_sample_index": index,
        "final_target_error_m": dfinal,
        "absolute_progress_m": float(d0 - dfinal),
        "best_progress_m": float(d0 - dmin),
        "normalized_progress": float((d0 - dfinal) / d0) if d0 > 0.0 else None,
        "initial_eef_world_m": initial.tolist(),
        "final_eef_world_m": all_positions[-1].tolist(),
        "tracking_rms_m": float(np.sqrt(np.mean(tracking * tracking))),
        "tracking_max_m": float(np.max(tracking)),
        "progress_sign": "POSITIVE" if d0 > dfinal else ("ZERO" if np.isclose(d0, dfinal, atol=1e-12) else "NEGATIVE"),
        "threshold_used": None,
        "target_world_m": target.tolist(),
        "frame": "world",
        "eef_point": evidence.TCP_SEMANTIC_ID,
    }


def _query_deltas(values: Mapping[str, Dict[str, np.ndarray]], baseline: str, metric: str, conditions: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"baseline_condition": baseline, "query_ids": ["q%d" % index for index in range(evidence.COHORT_FORWARD_COUNT)], "metrics": {}}
    for condition in conditions:
        entries: Dict[str, Any] = {}
        for index in range(evidence.COHORT_FORWARD_COUNT):
            query = "%s-q%d" % (condition, index)
            base_query = "%s-q%d" % (baseline, index)
            delta = np.asarray(values[condition][query], dtype=np.float64) - np.asarray(values[baseline][base_query], dtype=np.float64)
            entries["q%d" % index] = {"max_abs": float(np.max(np.abs(delta))), "l2": float(np.linalg.norm(delta.reshape(-1))), "mean_abs": float(np.mean(np.abs(delta))), "finite": bool(np.isfinite(delta).all())}
        result["metrics"][condition] = entries
    return result


def _verify_condition(bundle: Mapping[str, Any], bundle_sha: str, cohort_root: Path, condition_id: str, hard: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    root = (cohort_root / "conditions" / condition_id).resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "condition directory is invalid: %s" % condition_id)
    manifest = _read_canonical_json(root / "condition_manifest.json")
    evidence.verify_condition_manifest(manifest, root)
    manifest_sha = evidence.sha256_file(root / "condition_manifest.json")
    result = _read_canonical_json(root / "condition_result.json")
    _require(result.get("schema") == CONDITION_RESULT_SCHEMA and result.get("stage") == STAGE, "wrong condition result schema: %s" % condition_id)
    _require(result.get("status") == "PASS" and result.get("condition_id") == condition_id, "condition is not PASS: %s" % condition_id)
    _require(result.get("bundle_sha256") == bundle_sha and result.get("condition_manifest_sha256") == manifest_sha, "condition provenance hash mismatch: %s" % condition_id)
    _require(result.get("manifest") == manifest, "condition result does not embed its canonical manifest: %s" % condition_id)
    _close(result.get("condition_delta_base_xy_m"), evidence.COHORT_DELTAS_M[condition_id], "condition delta %s" % condition_id, atol=0.0)
    _require(result.get("schedule_native_ticks") == list(evidence.COHORT_SEGMENT_SCHEDULE), "condition schedule drift: %s" % condition_id)
    _require(result.get("cases") == evidence.COHORT_FORWARD_COUNT and result.get("gr00t_forward_count") == evidence.COHORT_FORWARD_COUNT, "GR00T count mismatch: %s" % condition_id)
    _require(result.get("reference_update_count") == evidence.COHORT_REFERENCE_UPDATE_COUNT and result.get("wbc_step_count") == sum(evidence.COHORT_SEGMENT_SCHEDULE), "execution count mismatch: %s" % condition_id)
    _require(result.get("wbc_segment_lengths") == list(evidence.COHORT_SEGMENT_SCHEDULE), "serialized WBC segment mismatch: %s" % condition_id)
    _require(result.get("error") is None and all(item in {"ACCEPT", "WARN"} for item in result.get("safety_gate_decisions", [])), "SafetyGate result is inadmissible: %s" % condition_id)
    _check_forbidden(result, "condition result %s" % condition_id)

    setup_payload = _read_canonical_json(root / "condition_setup.json")
    setup = setup_payload.get("event")
    _require(isinstance(setup, dict) and setup == result.get("setup"), "condition setup mismatch: %s" % condition_id)
    _close(setup.get("delta_base_xy_m"), evidence.COHORT_DELTAS_M[condition_id], "setup delta %s" % condition_id, atol=0.0)
    _require(setup.get("physics_steps_before_S0") == 0 and setup.get("settling_steps") == 0 and not setup.get("physics_reset") and not setup.get("physics_teleport") and not setup.get("policy_history_reset"), "initial-condition timing/reset contract drift: %s" % condition_id)
    for flag in ("root_xy_delta_exact", "root_non_xy_unchanged", "rigid_body_position_delta_exact", "rigid_body_non_position_unchanged", "other_actor_roots_unchanged", "target_root_unchanged", "dof_state_unchanged", "eef_translation_matches_delta", "eef_orientation_unchanged", "lifecycle_unchanged", "policy_histories_unchanged"):
        _require(setup.get(flag) is True, "initial-condition scope flag failed (%s): %s" % (flag, condition_id))
    _close(np.asarray(setup["base_position_after_m"]) - np.asarray(setup["base_position_before_m"]), evidence.COHORT_DELTAS_M[condition_id], "base position delta %s" % condition_id, atol=1e-6)
    _close(np.asarray(setup["eef_world_after_m"]) - np.asarray(setup["eef_world_before_m"]), evidence.COHORT_DELTAS_M[condition_id], "EEF position delta %s" % condition_id, atol=1e-5)
    _close(setup["s0"]["target_world_m"], evidence.TARGET_POSITION_WORLD_M, "S0 target %s" % condition_id, atol=0.0)

    metadata = _read_canonical_json(root / "run_metadata.json")
    _require(metadata.get("status") == "PASS" and metadata.get("condition_id") == condition_id and metadata.get("bundle_sha256") == bundle_sha, "run metadata mismatch: %s" % condition_id)
    _require(metadata.get("gpu_uuid") == GPU_UUID and metadata.get("cuda_visible_devices") == GPU_UUID, "GPU binding drift: %s" % condition_id)
    _require(metadata.get("gr00t_forward_count") == 4 and metadata.get("reference_update_count") == 3 and metadata.get("wbc_step_count") == 245 and metadata.get("wbc_segment_lengths") == list(evidence.COHORT_SEGMENT_SCHEDULE), "run metadata counters mismatch: %s" % condition_id)
    _check_forbidden(metadata, "run metadata %s" % condition_id)

    state_payload = _read_canonical_json(root / "state_v1.json")
    observations = state_payload.get("observations")
    expected_ids = ["%s-q%d" % (condition_id, index) for index in range(evidence.COHORT_FORWARD_COUNT)]
    _require(isinstance(observations, list) and [item.get("request_id") for item in observations] == expected_ids, "StateV1 query order mismatch: %s" % condition_id)
    state_values = np.asarray([_finite(item.get("state_v1"), (10,), "StateV1").tolist() for item in observations], dtype=np.float64)
    for item in observations:
        request = item.get("request", {})
        _require(request.get("language") == evidence.LANGUAGE and request.get("target_position_world_m") == evidence.TARGET_POSITION_WORLD_M.tolist() and request.get("source_frame") == evidence.SOURCE_FRAME and request.get("tcp_semantic_id") == evidence.TCP_SEMANTIC_ID, "observation contract drift: %s" % condition_id)
        _require(item.get("finite") is True and item.get("raw_shape") == [1, 40, 132] and item.get("decoded_shape") == [40, 10], "observation shape/finiteness drift: %s" % condition_id)
    raw = _load_npz(root / "raw_action_d132.npz", expected_ids, (1, 40, 132), "raw action %s" % condition_id)
    decoded = _load_npz(root / "decoded_action_d10_h40.npz", expected_ids, (40, 10), "decoded action %s" % condition_id)

    responses = _read_canonical_json(root / "worker_response.json").get("responses")
    _require(isinstance(responses, list) and [item.get("request_id") for item in responses] == expected_ids, "worker response order mismatch: %s" % condition_id)
    for item in responses:
        _require(item.get("status") == "PASS" and item.get("worker_forward_count_delta") == 1 and item.get("raw_shape") == [1, 40, 132], "worker response counter mismatch: %s" % condition_id)
    safety = _read_canonical_json(root / "safety_gate.json")
    _require(safety.get("decisions") == result.get("safety_gate_decisions") and len(safety.get("cases", [])) == 4 and all(item.get("decision") in {"ACCEPT", "WARN"} for item in safety.get("cases", [])), "SafetyGate evidence mismatch: %s" % condition_id)

    counters = _read_canonical_json(root / "counter_observations.json")
    _require(counters.get("final") == {"gr00t_forward_count": 4, "reference_update_count": 3, "wbc_step_count": 245}, "counter final mismatch: %s" % condition_id)
    counter_events = counters.get("events", [])
    forward_events = [item for item in counter_events if item.get("event") == "GR00T_FORWARD"]
    _require(len(forward_events) == 4 and [item.get("request_id") for item in forward_events] == expected_ids and all(item.get("gr00t_forward_delta_observed") == 1 and item.get("worker_forward_count_delta") == 1 for item in forward_events), "GR00T counter event mismatch: %s" % condition_id)
    segments = [item for item in counter_events if item.get("event") in {"WBC_SEGMENT", "WBC_COMPLETION_SEGMENT"}]
    _require([item.get("steps_observed") for item in segments] == list(evidence.COHORT_SEGMENT_SCHEDULE) and all(item.get("termination") is None for item in segments), "WBC counter event mismatch: %s" % condition_id)

    bind = _read_canonical_json(root / "initial_bind.json")
    _require(len(bind.get("events", [])) == 1, "initial bind event count mismatch: %s" % condition_id)
    bind_event = bind["events"][0]
    _require(bind_event.get("native_handoff_steps") == 49 and bind_event.get("handoff_steps") == 49 and bind_event.get("executor_prefix_steps") == 20 and bind_event.get("point_count") == 40 and bind_event.get("physics_root_state_unchanged") is True and bind_event.get("physics_dof_state_unchanged") is True and bind_event.get("reset_buf_unchanged") is True and not bind_event.get("physics_reset") and not bind_event.get("physics_teleport") and not bind_event.get("policy_history_reset"), "initial bind contract mismatch: %s" % condition_id)
    handoff = _read_canonical_json(root / "handoff.json")
    _require(len(handoff.get("events", [])) == 3, "handoff event count mismatch: %s" % condition_id)
    for index, item in enumerate(handoff["events"], start=1):
        _require(item.get("condition_id") == condition_id and item.get("chunk_id") == "%s-q%d" % (condition_id, index) and item.get("reference_only") is True and item.get("reference_update_delta_observed") == 1 and not item.get("physics_reset") and not item.get("physics_teleport") and not item.get("policy_history_reset"), "reference-only handoff mismatch: %s" % condition_id)
    completion = result.get("completion")
    _require(completion.get("endpoint_reached") is True and completion.get("termination") is None and completion.get("native_handoff_steps") == 49 and completion.get("schedule_native_ticks") == list(evidence.COHORT_SEGMENT_SCHEDULE), "completion contract mismatch: %s" % condition_id)
    executor = _read_canonical_json(root / "executor_bookkeeping.json")
    _require(len(executor.get("events", [])) == 4, "executor event count mismatch: %s" % condition_id)
    for index, item in enumerate(executor["events"]):
        _require(item.get("chunk_id") == expected_ids[index] and item.get("lifecycle_status") == "FRESH" and item.get("prefix_steps") == 20 and not item.get("physics_reset") and not item.get("physics_teleport") and not item.get("policy_history_reset"), "executor lifecycle mismatch: %s" % condition_id)
    for filename in ("root_state.json", "dof_state.json"):
        state_events = _read_canonical_json(root / filename).get("events")
        _require(isinstance(state_events, list) and len(state_events) == 4 and all(item.get("unchanged") is True for item in state_events), "%s continuity mismatch: %s" % (filename, condition_id))

    positions, row_counts, row_data = _check_rows(root, expected_ids, evidence.TARGET_POSITION_WORLD_M, hard)
    reconstructed = _metric_reconstruction(setup, positions, row_data["desired"], evidence.TARGET_POSITION_WORLD_M)
    metric_payload = _read_canonical_json(root / "condition_metrics.json")
    metrics = metric_payload.get("metrics")
    _require(metric_payload.get("schema") == "CORELAB_STAGE43_TARGET_METRICS_V1" and metrics == result.get("metrics"), "condition metric payload mismatch: %s" % condition_id)
    for key, expected in reconstructed.items():
        actual = metrics.get(key)
        if isinstance(expected, list):
            _close(actual, expected, "%s %s" % (key, condition_id), atol=1e-8)
        elif isinstance(expected, float):
            _require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-10), "%s metric mismatch: %s" % (key, condition_id))
        else:
            _require(actual == expected, "%s metric mismatch: %s" % (key, condition_id))
    state_by_query = {expected_ids[index]: state_values[index] for index in range(len(expected_ids))}
    return {"condition_id": condition_id, "delta_base_xy_m": list(evidence.COHORT_DELTAS_M[condition_id]), "initial_target_error_m": metrics["initial_target_error_m"], "minimum_target_error_m": metrics["minimum_target_error_m"], "final_target_error_m": metrics["final_target_error_m"], "absolute_progress_m": metrics["absolute_progress_m"], "best_progress_m": metrics["best_progress_m"], "normalized_progress": metrics["normalized_progress"], "progress_sign": metrics["progress_sign"], "tracking_rms_m": metrics["tracking_rms_m"], "tracking_max_m": metrics["tracking_max_m"], "initial_eef_world_m": metrics["initial_eef_world_m"], "final_eef_world_m": metrics["final_eef_world_m"], "target_world_m": metrics["target_world_m"], "safety_gate": list(result["safety_gate_decisions"]), "execution_integrity": "PASS", "trajectory_completion": True, "gr00t_forward_count": 4, "reference_update_count": 3, "wbc_step_count": 245, "hard_joint_limit_violation_cell_count": row_counts["hard_limit_violation_cell_count"], "soft_diagnostic_flag_row_count": row_counts["soft_diagnostic_flag_row_count"]}, {"raw": raw, "decoded": decoded, "state": state_by_query}, {"metrics": metrics, "row_counts": row_counts}


def finalize(bundle_path: Path, cohort_root: Path, output_path: Path, repo_root: Path) -> Dict[str, Any]:
    bundle_path = Path(bundle_path).resolve(strict=True)
    bundle = _read_canonical_json(bundle_path)
    evidence.verify_stage43_bundle_files(bundle, repo_root=Path(repo_root).resolve(strict=True))
    bundle_sha = evidence.sha256_file(bundle_path)
    _require(bundle_sha == evidence.sha256_file(bundle_path), "bundle hash read failure")
    _require(bundle.get("protocol", {}).get("execution_order") == list(evidence.COHORT_ORDER) and bundle.get("protocol", {}).get("stop_on_first_failure") is True, "cohort protocol drift")
    _require(bundle.get("budgets") == {"episodes": 5, "gr00t_forward_count": 20, "reference_update_count": 15, "wbc_step_count": 1225}, "cohort budget drift")
    _require(bundle.get("target", {}).get("numeric_xyz_in_model_input") is False, "numeric target model-input contract drift")
    g0 = _read_canonical_json(Path(bundle["g0"]["path"]))
    g1 = _read_canonical_json(Path(bundle["g1"]["path"]))
    _require(g0.get("token") == "CORELAB_STAGE43_TARGET_AND_TASK_METRICS_QUALIFIED" and g1.get("token") == "CORELAB_STAGE43_COHORT_HARNESS_READY", "G0/G1 token drift")
    _require(g0.get("status") == "PASS" and g1.get("status") == "PASS", "G0/G1 are not PASS")
    hard_path, hard_limits, joint_names = _hard_limits(Path(bundle["artifacts"]["wbc_source"]["path"]))
    condition_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, Dict[str, np.ndarray]] = {"raw": {}, "decoded": {}, "state": {}}
    diagnostics: Dict[str, Any] = {}
    for condition_id in evidence.COHORT_ORDER:
        row, values, detail = _verify_condition(bundle, bundle_sha, Path(cohort_root).resolve(strict=True), condition_id, hard_limits)
        condition_rows.append(row)
        arrays["raw"][condition_id] = values["raw"]
        arrays["decoded"][condition_id] = values["decoded"]
        arrays["state"][condition_id] = values["state"]
        diagnostics[condition_id] = detail
    raw_deltas = _query_deltas(arrays["raw"], evidence.COHORT_ORDER[0], "raw_action_d132", evidence.COHORT_ORDER)
    decoded_deltas = _query_deltas(arrays["decoded"], evidence.COHORT_ORDER[0], "decoded_action_d10_h40", evidence.COHORT_ORDER)
    state_deltas = _query_deltas(arrays["state"], evidence.COHORT_ORDER[0], "state_v1", evidence.COHORT_ORDER)
    result: Dict[str, Any] = {
        "schema": evidence.RESULT_SCHEMA,
        "stage": STAGE,
        "status": "PASS",
        "token": "CORELAB_STAGE43_CONTROLLED_INITIAL_CONDITION_COHORT_CHARACTERIZED",
        "return_token": evidence.RETURN_TOKEN,
        "pi_gate": evidence.PI_GATE,
        "issue_comment": evidence.ISSUE_COMMENT,
        "bundle_sha256": bundle_sha,
        "g0": {"sha256": bundle["g0"]["sha256"], "token": g0["token"], "status": g0["status"]},
        "g1": {"sha256": bundle["g1"]["sha256"], "token": g1["token"], "status": g1["status"]},
        "repository": bundle["repository"],
        "gpu": {"uuid": GPU_UUID, "sharing_allowed": True, "host": "corelab-4080s", "one_stage40_workload": True},
        "target_semantics": {"g0_token": g0["token"], "target_checks": g0.get("target_checks"), "evaluation_world_m": evidence.TARGET_POSITION_WORLD_M.tolist(), "language": evidence.LANGUAGE, "numeric_xyz_in_model_input": False, "hard_urdf_source": {"path": str(hard_path), "sha256": evidence.sha256_file(hard_path), "joint_count": len(joint_names), "joint_names": joint_names}},
        "condition_order": list(evidence.COHORT_ORDER),
        "conditions": condition_rows,
        "cross_condition_deltas": {"baseline_condition": evidence.COHORT_ORDER[0], "raw_action_d132": raw_deltas["metrics"], "decoded_action_d10_h40": decoded_deltas["metrics"], "state_v1": state_deltas["metrics"], "raw_action_d132_summary": raw_deltas, "decoded_action_d10_h40_summary": decoded_deltas, "state_v1_summary": state_deltas},
        "observed_counters": {"episodes": len(condition_rows), "gr00t_forward_count": int(sum(item["gr00t_forward_count"] for item in condition_rows)), "reference_update_count": int(sum(item["reference_update_count"] for item in condition_rows)), "wbc_step_count": int(sum(item["wbc_step_count"] for item in condition_rows))},
        "planned_budgets": bundle["budgets"],
        "evaluation_policy": {"continuous_metrics_only": True, "binary_success_threshold_m": None, "target_progress_is_descriptive": True, "threshold_used": None},
        "claim_boundary": "This is a fixed five-condition controlled initial-condition characterization, not a general robustness or task-success claim; endpoint_reached is reference trajectory completion and target progress remains continuous/descriptive.",
        "forbidden_operations": evidence.expected_forbidden_operations(),
        "characterizer": {"relative_path": "scripts/stage43_corelab_finalize.py", "sha256": evidence.sha256_file(Path(__file__).resolve())},
        "stop_gate": {"status": "STOP_PI", "next_action": "GPT_PI_REVIEW_AFTER_STAGE43_CONTROLLED_COHORT", "do_not_start": ["Stage44", "target relocation", "training", "fine-tuning", "Vanda A40", "Hopper H200", "hardware deployment"]},
    }
    output_path = Path(output_path).resolve()
    _require(not output_path.exists(), "final result output already exists: %s" % output_path)
    result_sha = evidence.write_canonical_json(output_path, result)
    cohort_manifest = {
        "schema": COHORT_MANIFEST_SCHEMA,
        "stage": STAGE,
        "bundle_sha256": bundle_sha,
        "result_sha256": result_sha,
        "g0_sha256": bundle["g0"]["sha256"],
        "g1_sha256": bundle["g1"]["sha256"],
        "condition_order": list(evidence.COHORT_ORDER),
        "conditions": [],
        "characterizer_sha256": evidence.sha256_file(Path(__file__).resolve()),
    }
    for condition_id in evidence.COHORT_ORDER:
        root = Path(cohort_root).resolve() / "conditions" / condition_id
        manifest_path = root / "condition_manifest.json"
        result_path = root / "condition_result.json"
        seal_path = root / "condition_seal.json"
        seal_sidecar = seal_path.with_name("condition_seal.json.sha256")
        _require(seal_path.is_file() and seal_sidecar.is_file(), "condition seal is missing: %s" % condition_id)
        seal = _read_canonical_json(seal_path)
        _require(seal.get("schema") == evidence.CONDITION_SEAL_SCHEMA and seal.get("status") == "PASS" and seal.get("bundle_sha256") == bundle_sha and seal.get("condition_manifest_sha256") == evidence.sha256_file(manifest_path) and seal.get("result_sha256") == evidence.sha256_file(result_path), "condition seal mismatch: %s" % condition_id)
        sidecar_fields = seal_sidecar.read_text(encoding="utf-8").strip().split()
        _require(len(sidecar_fields) >= 2 and sidecar_fields[0] == evidence.sha256_file(seal_path) and sidecar_fields[-1] == "condition_seal.json", "condition seal sidecar mismatch: %s" % condition_id)
        cohort_manifest["conditions"].append({"condition_id": condition_id, "condition_manifest_sha256": evidence.sha256_file(manifest_path), "condition_result_sha256": evidence.sha256_file(result_path), "condition_seal_sha256": evidence.sha256_file(seal_path)})
    cohort_manifest_path = output_path.with_name("stage43_cohort_manifest.json")
    _require(not cohort_manifest_path.exists(), "cohort manifest output already exists")
    cohort_manifest_sha = evidence.write_canonical_json(cohort_manifest_path, cohort_manifest)
    seal = {"schema": RESULT_SEAL_SCHEMA, "stage": STAGE, "status": "PASS", "bundle_sha256": bundle_sha, "result_sha256": result_sha, "cohort_manifest_sha256": cohort_manifest_sha, "condition_seal_sha256": [item["condition_seal_sha256"] for item in cohort_manifest["conditions"]]}
    seal_path = output_path.with_name("stage43_result.seal.json")
    _require(not seal_path.exists(), "result seal output already exists")
    seal_sha = evidence.write_canonical_json(seal_path, seal)
    sidecar_path = seal_path.with_name("stage43_result.seal.json.sha256")
    _require(not sidecar_path.exists(), "result seal sidecar output already exists")
    sidecar_path.write_text("{}  {}\n".format(seal_sha, seal_path.name), encoding="utf-8")
    return {"status": "PASS", "token": result["token"], "return_token": result["return_token"], "result_path": str(output_path), "result_sha256": result_sha, "cohort_manifest_path": str(cohort_manifest_path), "cohort_manifest_sha256": cohort_manifest_sha, "seal_path": str(seal_path), "seal_sha256": seal_sha, "bundle_sha256": bundle_sha, "observed_counters": result["observed_counters"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.bundle, args.cohort_root, args.output, args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
