#!/usr/bin/env python3
"""Build a CPU-only characterization profile from the sealed Stage40 M2 run.

The profile is deliberately descriptive.  It re-verifies the immutable
Stage40 predecessor chain and hashes the existing evidence, then extracts
tracking, lifecycle, SafetyGate, latency, memory-snapshot, and reference
progress metrics without constructing a model or simulator and without
adding any pass threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage40_corelab_evidence as stage40  # noqa: E402


PROFILE_SCHEMA = "CORELAB_STAGE41_M2_BASELINE_PROFILE_V1"
PROFILE_TOKEN = "CORELAB_STAGE41_M2_BASELINE_PROFILE_PASS"
M2_STAGE = "M2"
M2_REQUEST_IDS = ["stage40-q0", "stage40-q1", "stage40-q2", "stage40-q3"]
EXPECTED_NATIVE_SEGMENTS = [49, 49, 49, 98]
EXPECTED_EXECUTOR_PREFIX = 20
EXPECTED_ACTION_DT_S = 0.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Dict[str, Any]:
    return stage40.read_json(Path(path))


def _finite(value: Any, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), label + " must be finite")
    return number


def _finite_vector(value: Any, label: str, length: Optional[int] = None) -> List[float]:
    vector = [_finite(item, label) for item in list(value)]
    if length is not None:
        _require(len(vector) == length, "%s must have length %d" % (label, length))
    return vector


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [_finite(item, "metric") for item in values]
    _require(bool(numbers), "metric sequence is empty")
    ordered = sorted(numbers)
    index = int(round(0.95 * (len(ordered) - 1)))
    return {
        "count": len(numbers),
        "mean": float(statistics.fmean(numbers)),
        "rms": float(math.sqrt(statistics.fmean([item * item for item in numbers]))),
        "p95": float(ordered[index]),
        "max": float(max(numbers)),
        "min": float(min(numbers)),
    }


def _position_error(row: Mapping[str, Any]) -> float:
    achieved = _finite_vector(row.get("achieved_pos"), "achieved_pos", 3)
    desired = _finite_vector(row.get("desired_pos"), "desired_pos", 3)
    return float(math.sqrt(sum((left - right) ** 2 for left, right in zip(achieved, desired))))


def _orientation_error(row: Mapping[str, Any]) -> float:
    achieved = _finite_vector(row.get("achieved_quat"), "achieved_quat", 4)
    desired = _finite_vector(row.get("desired_quat"), "desired_quat", 4)
    achieved_norm = math.sqrt(sum(item * item for item in achieved))
    desired_norm = math.sqrt(sum(item * item for item in desired))
    _require(achieved_norm > 1e-12 and desired_norm > 1e-12, "quaternion norm is zero")
    dot = abs(sum(left * right for left, right in zip(achieved, desired)) / (achieved_norm * desired_norm))
    return float(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _row_metrics(rows: Sequence[Mapping[str, Any]], native_dt_s: float, label: str) -> Dict[str, Any]:
    _require(bool(rows), label + " is empty")
    position_errors = [_position_error(row) for row in rows]
    orientation_errors = [_orientation_error(row) for row in rows]
    termination_codes = {}
    for row in rows:
        code = str(row.get("termination_code", "NONE"))
        termination_codes[code] = termination_codes.get(code, 0) + 1
    last = rows[-1]
    first = rows[0]
    timing_values = [abs(_finite(row.get("traj_timing_err"), "traj_timing_err")) for row in rows]
    lateral_values = [abs(_finite(row.get("traj_d_lat"), "traj_d_lat")) for row in rows]
    joint_margins = []
    for row in rows:
        joint_margins.extend(_finite_vector(row.get("joint_limit_margin", []), "joint_limit_margin"))
    return {
        "label": label,
        "native_tick_count": len(rows),
        "simulated_duration_s": float(len(rows) * native_dt_s),
        "sim_time_start_s": _finite(first.get("sim_time"), "sim_time"),
        "sim_time_end_s": _finite(last.get("sim_time"), "sim_time"),
        "eef_position_tracking_m": _stats(position_errors),
        "eef_orientation_tracking_rad": _stats(orientation_errors),
        "reference_progress": {
            "traj_s_first": _finite(first.get("traj_s"), "traj_s"),
            "traj_s_last": _finite(last.get("traj_s"), "traj_s"),
            "traj_s_ref_first": _finite(first.get("traj_s_ref"), "traj_s_ref"),
            "traj_s_ref_last": _finite(last.get("traj_s_ref"), "traj_s_ref"),
            "traj_L_last": _finite(last.get("traj_L"), "traj_L"),
            "traj_d_lat_abs_max_m": float(max(lateral_values)),
            "traj_timing_error_abs_max": float(max(timing_values)),
        },
        "lifecycle_observations": {
            "done_count": sum(bool(row.get("done")) for row in rows),
            "reset_buf_count": sum(bool(row.get("reset_buf")) for row in rows),
            "invalid_state_count": sum(bool(row.get("invalid_state")) for row in rows),
            "joint_limit_violation_count": sum(bool(row.get("joint_limit_violation")) for row in rows),
            "termination_codes": termination_codes,
            "joint_limit_margin_min": float(min(joint_margins)) if joint_margins else None,
        },
    }


def _parse_gpu_query(query: str) -> List[Dict[str, Any]]:
    samples = []
    for line in str(query).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        try:
            samples.append({
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            })
        except (TypeError, ValueError):
            continue
    return samples


def _verify_stage40_chain(m2_root: Path) -> Dict[str, Dict[str, Any]]:
    chain_root = Path(m2_root).resolve().parent
    infos: Dict[str, Dict[str, Any]] = {}
    for stage_id in stage40.STAGES:
        stage_root = chain_root / stage_id
        seal = stage_root / (stage_id + "_result.seal.json")
        predecessor = stage40.PREDECESSOR[stage_id]
        predecessor_seal = None if predecessor is None else chain_root / predecessor / (predecessor + "_result.seal.json")
        infos[stage_id] = stage40.verify_stage_seal(
            seal,
            stage_root,
            expected_stage=stage_id,
            predecessor_seal=predecessor_seal,
        )
    _require(infos[M2_STAGE]["status"] == "PASS", "M2 predecessor chain is not PASS")
    return infos


def _state_profile(state: Mapping[str, Any]) -> Dict[str, Any]:
    observations = state.get("observations")
    _require(isinstance(observations, list) and len(observations) == len(M2_REQUEST_IDS), "M2 observation count mismatch")
    fingerprints = []
    state_deltas = []
    previous = None
    timestamps = []
    for index, observation in enumerate(observations):
        _require(observation.get("request_id") == M2_REQUEST_IDS[index], "observation order mismatch")
        vector = _finite_vector(observation.get("state_v1"), "StateV1", 10)
        timestamps.append(_finite(observation.get("observation_timestamp"), "observation_timestamp"))
        fingerprints.append({
            "request_id": observation.get("request_id"),
            "observation_timestamp": timestamps[-1],
            "body_shape": observation.get("body_shape"),
            "body_dtype": observation.get("body_dtype"),
            "body_sha256": observation.get("body_sha256"),
            "wrist_shape": observation.get("wrist_shape"),
            "wrist_dtype": observation.get("wrist_dtype"),
            "wrist_sha256": observation.get("wrist_sha256"),
            "state_v1_sha256": observation.get("state_v1_sha256"),
            "raw_action_sha256": observation.get("raw_sha256"),
            "decoded_action_sha256": observation.get("decoded_sha256"),
        })
        if previous is not None:
            delta = [left - right for left, right in zip(vector, previous)]
            state_deltas.append({
                "from_request_id": M2_REQUEST_IDS[index - 1],
                "to_request_id": M2_REQUEST_IDS[index],
                "l2": float(math.sqrt(sum(item * item for item in delta))),
                "max_abs": float(max(abs(item) for item in delta)),
            })
        previous = vector
    _require(timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps), "observations are not strictly ordered")
    _require(all(fingerprint["body_shape"] == [96, 96, 3] for fingerprint in fingerprints), "body RGB shape drift")
    _require(all(fingerprint["wrist_shape"] == [96, 96, 3] for fingerprint in fingerprints), "wrist RGB shape drift")
    return {
        "request_order": M2_REQUEST_IDS,
        "timestamps_s": timestamps,
        "fingerprints": fingerprints,
        "state_v1_deltas": state_deltas,
        "all_state_hashes_distinct": len({item["state_v1_sha256"] for item in fingerprints}) == len(fingerprints),
        "all_body_hashes_distinct": len({item["body_sha256"] for item in fingerprints}) == len(fingerprints),
        "all_wrist_hashes_distinct": len({item["wrist_sha256"] for item in fingerprints}) == len(fingerprints),
    }


def _latency_profile(state: Mapping[str, Any]) -> Dict[str, Any]:
    values = []
    per_request = []
    for observation in state["observations"]:
        latency = _finite(observation.get("response", {}).get("latency_s"), "latency_s")
        values.append(latency)
        per_request.append({"request_id": observation.get("request_id"), "latency_s": latency})
    return {"per_request": per_request, "summary": _stats(values)}


def _safety_profile(safety: Mapping[str, Any]) -> Dict[str, Any]:
    decisions = [str(item) for item in safety.get("decisions", [])]
    _require(len(decisions) == 4 and all(item in {"ACCEPT", "WARN"} for item in decisions), "invalid M2 SafetyGate decisions")
    cases = safety.get("cases")
    _require(isinstance(cases, list) and len(cases) == 4, "M2 SafetyGate case count mismatch")
    known_maxima: Dict[str, float] = {}
    per_case = []
    for case in cases:
        metrics = case.get("metrics", {})
        compact = {"chunk_id": case.get("chunk_id"), "decision": case.get("decision")}
        compact_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict) and "max" in value:
                number = _finite(value["max"], "SafetyGate " + str(key) + ".max")
                compact_metrics[str(key)] = {"max": number, "mean": value.get("mean"), "median": value.get("median"), "p95": value.get("p95")}
                known_maxima[str(key)] = max(known_maxima.get(str(key), number), number)
            elif isinstance(value, (int, float)):
                number = _finite(value, "SafetyGate " + str(key))
                compact_metrics[str(key)] = number
                known_maxima[str(key)] = max(known_maxima.get(str(key), number), number)
        per_case.append({**compact, "metrics": compact_metrics, "issues": list(case.get("issues", []))})
    return {"decisions": decisions, "per_case": per_case, "detailed_maxima": known_maxima}


def _handoff_profile(initial_bind: Mapping[str, Any], handoff: Mapping[str, Any]) -> Dict[str, Any]:
    initial_events = initial_bind.get("events", [])
    handoff_events = handoff.get("events", [])
    _require(len(initial_events) == 1 and len(handoff_events) == 3, "M2 bind/handoff event count mismatch")
    initial = initial_events[0]
    native_steps = [int(event.get("native_handoff_steps")) for event in handoff_events]
    executor_prefix = [int(event.get("executor_prefix_steps")) for event in handoff_events]
    action_dt = [float(event.get("action_sample_dt_s")) for event in handoff_events]
    _require(native_steps == [49, 49, 49], "M2 native handoff schedule drift")
    _require(executor_prefix == [EXPECTED_EXECUTOR_PREFIX] * 3, "M2 executor prefix schedule drift")
    _require(all(math.isclose(value, EXPECTED_ACTION_DT_S, rel_tol=0.0, abs_tol=1e-12) for value in action_dt), "M2 action sample period drift")
    continuity_fields = ("physics_root_state_unchanged", "physics_dof_state_unchanged", "episode_length_unchanged", "arm_policy_history_unchanged", "dog_policy_history_unchanged", "reset_buf_unchanged", "physics_reset", "physics_teleport", "policy_history_reset")
    continuity = []
    for event in [initial] + list(handoff_events):
        continuity.append({field: bool(event.get(field)) for field in continuity_fields})
    return {
        "initial_bind": {
            "native_handoff_steps": int(initial.get("native_handoff_steps")),
            "executor_prefix_steps": int(initial.get("executor_prefix_steps")),
            "action_sample_dt_s": float(initial.get("action_sample_dt_s")),
        },
        "native_handoff_steps": native_steps,
        "executor_prefix_steps": executor_prefix,
        "action_sample_dt_s": action_dt,
        "continuity_events": continuity,
        "all_continuity_predicates_true": all(
            item[field] is True
            for item in continuity
            for field in ("physics_root_state_unchanged", "physics_dof_state_unchanged", "episode_length_unchanged", "arm_policy_history_unchanged", "dog_policy_history_unchanged", "reset_buf_unchanged")
        ),
        "all_forbidden_lifecycle_effects_false": all(
            item[field] is False
            for item in continuity
            for field in ("physics_reset", "physics_teleport", "policy_history_reset")
        ),
    }


def _memory_profile(telemetry: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    target_uuid = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
    snapshots = []
    for sample in telemetry.get("samples", []):
        parsed = _parse_gpu_query(sample.get("query", ""))
        target = [item for item in parsed if item.get("uuid") == target_uuid]
        if target:
            snapshots.append({"label": sample.get("label"), **target[0]})
    used = [int(item["memory_used_mib"]) for item in snapshots]
    worker_peaks = []
    for observation in state.get("observations", []):
        gpu = observation.get("response", {}).get("gpu", {})
        if gpu.get("peak_allocated_mib") is not None:
            worker_peaks.append(_finite(gpu.get("peak_allocated_mib"), "peak_allocated_mib"))
    reserved = []
    for observation in state.get("observations", []):
        gpu = observation.get("response", {}).get("gpu", {})
        if gpu.get("peak_reserved_mib") is not None:
            reserved.append(_finite(gpu.get("peak_reserved_mib"), "peak_reserved_mib"))
    _require(bool(snapshots), "M2 GPU telemetry has no target UUID sample")
    return {
        "target_gpu_uuid": target_uuid,
        "nvidia_smi_snapshots": snapshots,
        "memory_used_mib_baseline": used[0],
        "memory_used_mib_peak_snapshot": max(used),
        "memory_used_mib_after_phase": used[-1],
        "worker_peak_allocated_mib": max(worker_peaks) if worker_peaks else None,
        "worker_peak_reserved_mib": max(reserved) if reserved else None,
        "per_replan_gpu_snapshots_available": False,
        "host_ram": {"available": False, "reason": "Stage40 M2 did not record host RAM telemetry"},
    }


def build_profile(m2_seal: Path, m2_evidence_root: Path) -> Dict[str, Any]:
    m2_seal = Path(m2_seal).resolve(strict=True)
    m2_root = Path(m2_evidence_root).resolve(strict=True)
    _require(m2_root.name == M2_STAGE, "M2 evidence root must be the canonical M2 chain directory")
    chain = _verify_stage40_chain(m2_root)
    result = _read(m2_root / "M2_result.json")
    manifest = _read(m2_root / "evidence_manifest.json")
    _require(stage40.sha256_file(m2_seal) == chain[M2_STAGE]["seal_sha256"], "M2 seal path does not match verified chain")
    stage40.verify_evidence_manifest(manifest, m2_root)
    counters = stage40.verify_counter_observations(M2_STAGE, result.get("counters", {}), manifest, m2_root)
    metadata = _read(m2_root / "run_metadata.json")
    phase = _read(m2_root / "phase_result.json")
    state = _read(m2_root / "state_v1.json")
    safety = _read(m2_root / "safety_gate.json")
    before = _read(m2_root / "rows_before_handoff.json").get("segments", [])
    after = _read(m2_root / "rows_after_handoff.json").get("segments", [])
    _require([len(segment) for segment in before + after] == EXPECTED_NATIVE_SEGMENTS, "M2 WBC segment vector drift")
    native_dt_s = _finite(metadata.get("native_control_dt_s"), "native_control_dt_s")
    segment_profiles = [_row_metrics(segment, native_dt_s, "handoff_%d_prefix" % (index + 1)) for index, segment in enumerate(before)]
    segment_profiles.append(_row_metrics(after[0], native_dt_s, "completion"))
    all_rows = [row for segment in before + after for row in segment]
    result_completion = phase.get("completion")
    _require(isinstance(result_completion, dict) and bool(result_completion.get("endpoint_reached")), "M2 endpoint completion evidence missing")
    return {
        "schema": PROFILE_SCHEMA,
        "status": "PASS",
        "token": PROFILE_TOKEN,
        "classification": "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE",
        "thresholds_added": False,
        "source": {
            "m2_seal_sha256": chain[M2_STAGE]["seal_sha256"],
            "m2_result_sha256": stage40.sha256_file(m2_root / "M2_result.json"),
            "m2_evidence_manifest_sha256": stage40.sha256_file(m2_root / "evidence_manifest.json"),
            "m2_bundle_sha256": result.get("execution_bundle_sha256"),
            "repository_head": result.get("repository_head"),
            "evidence_root": str(m2_root),
            "chain": [{"stage_id": stage_id, **chain[stage_id]} for stage_id in stage40.STAGES],
        },
        "contract": {
            "dev_seed": int(metadata.get("dev_seed")),
            "native_control_dt_s": native_dt_s,
            "handoff_target_s": 0.9750000000000001,
            "chunk_duration_s": 1.9500000000000002,
            "action_sample_dt_s": EXPECTED_ACTION_DT_S,
            "executor_prefix_samples": EXPECTED_EXECUTOR_PREFIX,
            "frame": "NATIVE_WBC_RELATIVE_REFERENCE",
            "tcp_semantic_id": "x5_link6_gripper_center",
            "action_contract": "StructuredActionV1_K16_T16_H40_D132_to_D10",
        },
        "counters": counters,
        "tracking": {
            "overall": _row_metrics(all_rows, native_dt_s, "all_M2_rows"),
            "segments": segment_profiles,
        },
        "safety_gate": _safety_profile(safety),
        "observations": _state_profile(state),
        "latency": _latency_profile(state),
        "native_wbc": {
            "step_count": int(metadata.get("wbc_step_count")),
            "native_ticks_by_segment": EXPECTED_NATIVE_SEGMENTS,
            "simulated_duration_s": float(sum(EXPECTED_NATIVE_SEGMENTS) * native_dt_s),
            "wall_duration_available": False,
        },
        "memory": _memory_profile(_read(m2_root / "telemetry.json"), state),
        "reference_progress": {
            "completion": result_completion,
            "handoff": _handoff_profile(_read(m2_root / "initial_bind.json"), _read(m2_root / "handoff.json")),
        },
        "lifecycle": {
            "phase_status": phase.get("status"),
            "worker_return_code": metadata.get("worker_return_code"),
            "error": metadata.get("error"),
            "forbidden_operations": metadata.get("forbidden_operations"),
        },
    }


def write_report(profile: Mapping[str, Any], output: Path, profile_sha256: str) -> None:
    tracking = profile["tracking"]["overall"]
    safety = profile["safety_gate"]
    observations = profile["observations"]
    latency = profile["latency"]["summary"]
    memory = profile["memory"]
    completion = profile["reference_progress"]["completion"]
    lines = [
        "# Stage41 L0 — sealed Stage40 M2 baseline profile",
        "",
        "Status: `CORELAB_STAGE41_M2_BASELINE_PROFILE_PASS`",
        "",
        "This is a CPU/read-only characterization of the accepted M2 evidence. It adds no pass threshold and does not rerun the model, simulator, or WBC.",
        "",
        "## Identity",
        "",
        "- M2 seal: `%s`" % profile["source"]["m2_seal_sha256"],
        "- M2 result: `%s`" % profile["source"]["m2_result_sha256"],
        "- evidence manifest: `%s`" % profile["source"]["m2_evidence_manifest_sha256"],
        "- bundle: `%s`" % profile["source"]["m2_bundle_sha256"],
        "- repository HEAD: `%s`" % profile["source"]["repository_head"],
        "- profile SHA-256: `%s`" % profile_sha256,
        "",
        "## Closed-loop baseline",
        "",
        "- GR00T forwards: `4`; reference updates: `3`; WBC segments: `[49,49,49,98]`; total ticks: `245`.",
        "- SafetyGate: `%s`." % ", ".join(safety["decisions"]),
        "- endpoint reached: `%s`; native reference time: `%s s`." % (completion["endpoint_reached"], completion["native_time_s"]),
        "- native handoff schedule: `49` ticks; executor action prefix: `20` samples at `0.05 s`.",
        "",
        "## Tracking characterization",
        "",
        "- EEF position error RMS/max: `%.9f m` / `%.9f m`." % (tracking["eef_position_tracking_m"]["rms"], tracking["eef_position_tracking_m"]["max"]),
        "- EEF orientation error RMS/max: `%.9f rad` / `%.9f rad`." % (tracking["eef_orientation_tracking_rad"]["rms"], tracking["eef_orientation_tracking_rad"]["max"]),
        "- SafetyGate detailed maxima: `%s`." % json.dumps(safety["detailed_maxima"], sort_keys=True),
        "- StateV1 hashes distinct across q0–q3: `%s`; adjacent state deltas: `%s`." % (observations["all_state_hashes_distinct"], json.dumps(observations["state_v1_deltas"], sort_keys=True)),
        "",
        "## Runtime characterization",
        "",
        "- GR00T latency mean/p95/max: `%.6f / %.6f / %.6f s`." % (latency["mean"], latency["p95"], latency["max"]),
        "- target GPU nvidia-smi used-memory baseline/peak/after: `%s / %s / %s MiB`." % (memory["memory_used_mib_baseline"], memory["memory_used_mib_peak_snapshot"], memory["memory_used_mib_after_phase"]),
        "- worker peak allocated/reserved: `%s / %s MiB`; host RAM: unavailable in M2 evidence." % (memory["worker_peak_allocated_mib"], memory["worker_peak_reserved_mib"]),
        "",
        "## Boundary",
        "",
        "This profile characterizes one sealed TINY1 development fixture at seed 4242. It is not a robustness, generalization, real-time, or hardware claim.",
        "",
    ]
    Path(output).write_text("\n".join(lines), encoding="utf-8")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-seal", type=Path, required=True)
    parser.add_argument("--m2-evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    profile = build_profile(args.m2_seal, args.m2_evidence_root)
    digest = stage40.write_canonical_json(args.output, profile)
    if args.report is not None:
        write_report(profile, args.report, digest)
    print(json.dumps({"status": profile["status"], "token": profile["token"], "profile_sha256": digest, "path": str(args.output), "report": None if args.report is None else str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
