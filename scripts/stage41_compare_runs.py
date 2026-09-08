#!/usr/bin/env python3
"""Compare the sealed M2 baseline with two sealed Stage41 LH10 runs.

The comparison is descriptive: it reports drift, timing, safety severity,
latency, GPU/RAM telemetry, and hash repeatability without inventing a new
scientific threshold.  It is CPU-only and validates both Stage41 seals before
reading their serialized evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage40_corelab_evidence as stage40  # noqa: E402
import stage41_corelab_evidence as evidence  # noqa: E402
import stage41_m2_baseline_profile as baseline_profile  # noqa: E402


REPORT_SCHEMA = "CORELAB_STAGE41_LONG_HORIZON_ENGINEERING_REPORT_V1"
REPORT_TOKEN = "CORELAB_STAGE41_LONG_HORIZON_ENGINEERING_CLOSED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Dict[str, Any]:
    return stage40.read_json(Path(path))


def _finite(value: Any, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), label + " must be finite")
    return number


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [_finite(item, "metric") for item in values]
    _require(bool(numbers), "metric sequence is empty")
    ordered = sorted(numbers)
    return {
        "count": len(numbers),
        "mean": float(statistics.fmean(numbers)),
        "rms": float(math.sqrt(statistics.fmean([item * item for item in numbers]))),
        "p95": float(ordered[int(round(0.95 * (len(ordered) - 1)))]),
        "max": float(max(numbers)),
        "min": float(min(numbers)),
    }


def _parse_gpu_query(query: str) -> List[Dict[str, Any]]:
    parsed = []
    for line in str(query).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        try:
            parsed.append({
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            })
        except (TypeError, ValueError):
            continue
    return parsed


def _telemetry_profile(telemetry: Mapping[str, Any], target_uuid: str) -> Dict[str, Any]:
    gpu_samples = []
    host_samples = []
    for sample in telemetry.get("samples", []):
        label = sample.get("label")
        gpu = sample.get("gpu", {})
        for row in _parse_gpu_query(gpu.get("query", "")):
            if row.get("uuid") == target_uuid:
                gpu_samples.append({"label": label, **row})
        host = sample.get("host")
        if isinstance(host, dict):
            process = host.get("process", {}).get("status_kib", {})
            system = host.get("system_kib", {})
            host_samples.append({
                "label": label,
                "process_vm_rss_kib": process.get("VmRSS"),
                "process_vm_hwm_kib": process.get("VmHWM"),
                "process_vm_size_kib": process.get("VmSize"),
                "system_mem_available_kib": system.get("MemAvailable"),
            })
    used = [int(item["memory_used_mib"]) for item in gpu_samples]
    rss = [int(item["process_vm_rss_kib"]) for item in host_samples if item.get("process_vm_rss_kib") is not None]
    hwm = [int(item["process_vm_hwm_kib"]) for item in host_samples if item.get("process_vm_hwm_kib") is not None]
    return {
        "target_gpu_uuid": target_uuid,
        "gpu_samples": gpu_samples,
        "gpu_memory_used_mib": {"baseline": used[0], "peak": max(used), "after": used[-1], "delta_after_minus_baseline": used[-1] - used[0]} if used else None,
        "host_samples": host_samples,
        "host_process_vm_rss_kib": {"baseline": rss[0], "peak": max(rss), "after": rss[-1], "delta_after_minus_baseline": rss[-1] - rss[0]} if rss else None,
        "host_process_vm_hwm_kib": max(hwm) if hwm else None,
    }


def _safety_profile(state: Mapping[str, Any]) -> Dict[str, Any]:
    decisions = []
    per_handoff = []
    for index, observation in enumerate(state.get("observations", [])):
        gate = observation.get("safety_gate", {})
        decision = str(gate.get("decision"))
        decisions.append(decision)
        metrics = gate.get("metrics", {})
        compact: Dict[str, Any] = {"request_id": observation.get("request_id"), "decision": decision}
        for key in ("linear_speed_mps", "angular_speed_radps"):
            value = metrics.get(key, {})
            if isinstance(value, dict) and "max" in value:
                compact[key + "_max"] = _finite(value["max"], key + ".max")
        for key in ("initial_position_discontinuity_m", "initial_rotation_discontinuity_rad", "se3_arc_length_m", "translation_path_length_m", "chunk_age_s"):
            if key in metrics:
                compact[key] = _finite(metrics[key], key)
        compact["issues"] = list(gate.get("issues", []))
        per_handoff.append(compact)
    return {
        "decisions": decisions,
        "admissible": len(decisions) == evidence.SAFETY_DECISION_COUNT and all(item in {"ACCEPT", "WARN"} for item in decisions),
        "per_forward": per_handoff,
        "warn_count": sum(item == "WARN" for item in decisions),
        "reject_count": sum(item == "REJECT" for item in decisions),
        "linear_speed_max_mps": max((item.get("linear_speed_mps_max", 0.0) for item in per_handoff), default=0.0),
        "angular_speed_max_radps": max((item.get("angular_speed_radps_max", 0.0) for item in per_handoff), default=0.0),
    }


def _observation_profile(state: Mapping[str, Any]) -> Dict[str, Any]:
    observations = list(state.get("observations", []))
    return {
        "request_ids": [str(item.get("request_id")) for item in observations],
        "timestamps_s": [_finite(item.get("observation_timestamp"), "observation_timestamp") for item in observations],
        "state_v1_sha256": [str(item.get("state_v1_sha256")) for item in observations],
        "raw_action_sha256": [str(item.get("raw_sha256")) for item in observations],
        "decoded_action_sha256": [str(item.get("decoded_sha256")) for item in observations],
        "body_sha256": [str(item.get("body_sha256")) for item in observations],
        "wrist_sha256": [str(item.get("wrist_sha256")) for item in observations],
        "all_state_hashes_distinct": len({item.get("state_v1_sha256") for item in observations}) == len(observations),
    }


def _run_summary(root: Path, target_uuid: str) -> Dict[str, Any]:
    root = Path(root).resolve(strict=True)
    metadata = _read(root / "run_metadata.json")
    phase = _read(root / "phase_result.json")
    state = _read(root / "state_v1.json")
    safety = _safety_profile(state)
    before = _read(root / "rows_before_handoff.json").get("segments", [])
    after = _read(root / "rows_after_handoff.json").get("segments", [])
    all_segments = list(before) + list(after)
    native_dt = _finite(metadata.get("native_control_dt_s"), "native_control_dt_s")
    segment_profiles = [baseline_profile._row_metrics(segment, native_dt, "segment_%02d" % index) for index, segment in enumerate(all_segments)]
    all_rows = [row for segment in all_segments for row in segment]
    handoff = _read(root / "handoff.json")
    handoffs = list(handoff.get("events", []))
    initial = _read(root / "initial_bind.json").get("events", [])
    observations = _observation_profile(state)
    latency_values = [_finite(item.get("response", {}).get("latency_s"), "latency_s") for item in state.get("observations", [])]
    timing_values = [abs(_finite(row.get("traj_timing_err"), "traj_timing_err")) for row in all_rows]
    phase_completion = phase.get("completion")
    contract = {
        "phase_status_pass": phase.get("status") == "PASS",
        "gr00t_forward_count_exact": metadata.get("gr00t_forward_count") == evidence.GR00T_FORWARD_COUNT,
        "reference_update_count_exact": metadata.get("reference_update_count") == evidence.REFERENCE_UPDATE_COUNT,
        "safety_admissible": safety["admissible"],
        "native_handoff_steps": list(metadata.get("native_handoff_steps", [])),
        "executor_prefix_steps": list(metadata.get("executor_prefix_steps", [])),
        "action_sample_dt_s": list(metadata.get("action_sample_dt_s", [])),
        "endpoint_reached": bool(isinstance(phase_completion, dict) and phase_completion.get("endpoint_reached")),
        "worker_return_code_zero": metadata.get("worker_return_code") == 0,
        "forbidden_operations_clear": metadata.get("forbidden_operations") == evidence.expected_forbidden_operations(),
        "continuity_events": list(initial) + handoffs,
    }
    contract["continuity_pass"] = all(
        event.get(field) is True
        for event in contract["continuity_events"]
        for field in ("physics_root_state_unchanged", "physics_dof_state_unchanged", "episode_length_unchanged", "arm_policy_history_unchanged", "dog_policy_history_unchanged", "reset_buf_unchanged")
    ) and all(
        event.get(field) is False
        for event in contract["continuity_events"]
        for field in ("physics_reset", "physics_teleport", "policy_history_reset")
    )
    contract["all_required_predicates"] = all(value for key, value in contract.items() if key.endswith("_exact") or key in {"phase_status_pass", "safety_admissible", "endpoint_reached", "worker_return_code_zero", "forbidden_operations_clear", "continuity_pass"})
    return {
        "root": str(root),
        "metadata": {
            "schema": metadata.get("schema"),
            "status": metadata.get("status"),
            "seed": metadata.get("seed"),
            "gpu_uuid": metadata.get("gpu_uuid"),
            "native_control_dt_s": native_dt,
            "gr00t_forward_count": metadata.get("gr00t_forward_count"),
            "wbc_step_count": metadata.get("wbc_step_count"),
            "reference_update_count": metadata.get("reference_update_count"),
            "bundle_sha256": metadata.get("bundle_sha256"),
        },
        "contract": contract,
        "completion": phase_completion,
        "handoff_events": handoffs,
        "observations": observations,
        "safety_gate": safety,
        "latency": {"per_forward_s": latency_values, "summary": _stats(latency_values)},
        "tracking": {"overall": baseline_profile._row_metrics(all_rows, native_dt, "all_LH10_rows"), "segments": segment_profiles},
        "timing": {"absolute_traj_timing_error": _stats(timing_values)},
        "telemetry": _telemetry_profile(_read(root / "telemetry.json"), target_uuid),
    }


def _hash_comparison(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    fields = ("state_v1_sha256", "raw_action_sha256", "decoded_action_sha256", "body_sha256", "wrist_sha256")
    result = {}
    for field in fields:
        left, right = a["observations"].get(field, []), b["observations"].get(field, [])
        result[field + "_identical"] = left == right
        result[field + "_a"] = left
        result[field + "_b"] = right
    return result


def _profile_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    def delta(path: Sequence[str]) -> Optional[float]:
        left: Any = a
        right: Any = b
        for key in path:
            left, right = left[key], right[key]
        return float(right) - float(left)

    return {
        "tracking_position_rms_m": delta(("tracking", "overall", "eef_position_tracking_m", "rms")),
        "tracking_position_max_m": delta(("tracking", "overall", "eef_position_tracking_m", "max")),
        "tracking_orientation_rms_rad": delta(("tracking", "overall", "eef_orientation_tracking_rad", "rms")),
        "latency_mean_s": delta(("latency", "summary", "mean")),
        "latency_p95_s": delta(("latency", "summary", "p95")),
        "gpu_memory_after_minus_baseline_mib": (b.get("telemetry", {}).get("gpu_memory_used_mib") or {}).get("delta_after_minus_baseline"),
        "host_rss_after_minus_baseline_kib": (b.get("telemetry", {}).get("host_process_vm_rss_kib") or {}).get("delta_after_minus_baseline"),
    }


def build_report(baseline_path: Path, run_a: Path, run_b: Path, seal_a: Path, seal_b: Path, predecessor_seal: Path, output: Path, report: Path) -> Dict[str, Any]:
    baseline = _read(baseline_path)
    _require(baseline.get("status") == "PASS" and baseline.get("token") == "CORELAB_STAGE41_M2_BASELINE_PROFILE_PASS", "L0 baseline profile is not PASS")
    info_a = evidence.verify_stage_seal(seal_a, run_a, predecessor_seal=predecessor_seal)
    info_b = evidence.verify_stage_seal(seal_b, run_b, predecessor_seal=predecessor_seal)
    target_uuid = str(_read(run_a / "run_metadata.json").get("gpu_uuid"))
    summary_a = _run_summary(run_a, target_uuid)
    summary_b = _run_summary(run_b, target_uuid)
    hashes = _hash_comparison(summary_a, summary_b)
    baseline_tracking = baseline.get("tracking", {}).get("overall", {})
    baseline_latency = baseline.get("latency", {}).get("summary", {})
    report_value: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if summary_a["contract"]["all_required_predicates"] and summary_b["contract"]["all_required_predicates"] else "BLOCKED",
        "token": REPORT_TOKEN,
        "thresholds_added": False,
        "baseline": {
            "profile_sha256": stage40.sha256_file(baseline_path),
            "m2_seal_sha256": baseline.get("source", {}).get("m2_seal_sha256"),
            "tracking": baseline_tracking,
            "latency": baseline_latency,
            "memory": baseline.get("memory"),
        },
        "runs": {"LH10-A": summary_a, "LH10-B": summary_b},
        "repeatability": {
            **hashes,
            "contract_identical": summary_a["contract"]["all_required_predicates"] and summary_b["contract"]["all_required_predicates"],
            "tracking_delta_B_minus_A": _profile_delta(summary_a, summary_b),
        },
        "seal_verification": {"LH10-A": info_a, "LH10-B": info_b},
        "engineering_readout": {
            "long_horizon_contract_pass": summary_a["contract"]["all_required_predicates"] and summary_b["contract"]["all_required_predicates"],
            "state_hashes_byte_identical": hashes["state_v1_sha256_identical"],
            "raw_action_hashes_byte_identical": hashes["raw_action_sha256_identical"],
            "decoded_action_hashes_byte_identical": hashes["decoded_action_sha256_identical"],
            "note": "Hash differences are reported, not treated as contract failure; no new tracking, safety, latency, or memory threshold is introduced.",
        },
    }
    stage40.write_canonical_json(output, report_value)
    write_human_report(report_value, report)
    return {"status": report_value["status"], "token": REPORT_TOKEN, "report_sha256": stage40.sha256_file(output), "path": str(output), "human_report": str(report)}


def write_human_report(value: Mapping[str, Any], path: Path) -> None:
    runs = value["runs"]
    lines = [
        "# Stage41 long-horizon engineering closure",
        "",
        "Status: `%s`" % value["token"],
        "",
        "This report compares the sealed Stage40 M2 characterization with two fresh Stage41 LH10 runs. It adds no scientific threshold.",
        "",
        "## Contract",
        "",
        "- LH10-A contract predicates: `%s`; LH10-B: `%s`." % (runs["LH10-A"]["contract"]["all_required_predicates"], runs["LH10-B"]["contract"]["all_required_predicates"]),
        "- Forward/update counts: A `%s/%s`; B `%s/%s`." % (runs["LH10-A"]["metadata"]["gr00t_forward_count"], runs["LH10-A"]["metadata"]["reference_update_count"], runs["LH10-B"]["metadata"]["gr00t_forward_count"], runs["LH10-B"]["metadata"]["reference_update_count"]),
        "- Native handoff vectors: A `%s`; B `%s`." % (runs["LH10-A"]["contract"]["native_handoff_steps"], runs["LH10-B"]["contract"]["native_handoff_steps"]),
        "- Executor prefix vectors: A `%s`; B `%s`." % (runs["LH10-A"]["contract"]["executor_prefix_steps"], runs["LH10-B"]["contract"]["executor_prefix_steps"]),
        "- SafetyGate WARN/REJECT counts: A `%s/%s`; B `%s/%s`." % (runs["LH10-A"]["safety_gate"]["warn_count"], runs["LH10-A"]["safety_gate"]["reject_count"], runs["LH10-B"]["safety_gate"]["warn_count"], runs["LH10-B"]["safety_gate"]["reject_count"]),
        "- Endpoint: A `%s`; B `%s`." % (runs["LH10-A"]["contract"]["endpoint_reached"], runs["LH10-B"]["contract"]["endpoint_reached"]),
        "",
        "## Long-horizon trends",
        "",
        "- LH10-A tracking position RMS/max: `%.9f / %.9f m`; LH10-B: `%.9f / %.9f m`." % (runs["LH10-A"]["tracking"]["overall"]["eef_position_tracking_m"]["rms"], runs["LH10-A"]["tracking"]["overall"]["eef_position_tracking_m"]["max"], runs["LH10-B"]["tracking"]["overall"]["eef_position_tracking_m"]["rms"], runs["LH10-B"]["tracking"]["overall"]["eef_position_tracking_m"]["max"]),
        "- GR00T latency mean/p95/max: A `%.6f / %.6f / %.6f s`; B `%.6f / %.6f / %.6f s`." % (runs["LH10-A"]["latency"]["summary"]["mean"], runs["LH10-A"]["latency"]["summary"]["p95"], runs["LH10-A"]["latency"]["summary"]["max"], runs["LH10-B"]["latency"]["summary"]["mean"], runs["LH10-B"]["latency"]["summary"]["p95"], runs["LH10-B"]["latency"]["summary"]["max"]),
        "- Target GPU used-memory baseline/peak/after MiB: A `%s/%s/%s`; B `%s/%s/%s`." % (runs["LH10-A"]["telemetry"]["gpu_memory_used_mib"]["baseline"], runs["LH10-A"]["telemetry"]["gpu_memory_used_mib"]["peak"], runs["LH10-A"]["telemetry"]["gpu_memory_used_mib"]["after"], runs["LH10-B"]["telemetry"]["gpu_memory_used_mib"]["baseline"], runs["LH10-B"]["telemetry"]["gpu_memory_used_mib"]["peak"], runs["LH10-B"]["telemetry"]["gpu_memory_used_mib"]["after"]),
        "- Host RSS baseline/peak/after KiB: A `%s/%s/%s`; B `%s/%s/%s`." % (runs["LH10-A"]["telemetry"]["host_process_vm_rss_kib"]["baseline"], runs["LH10-A"]["telemetry"]["host_process_vm_rss_kib"]["peak"], runs["LH10-A"]["telemetry"]["host_process_vm_rss_kib"]["after"], runs["LH10-B"]["telemetry"]["host_process_vm_rss_kib"]["baseline"], runs["LH10-B"]["telemetry"]["host_process_vm_rss_kib"]["peak"], runs["LH10-B"]["telemetry"]["host_process_vm_rss_kib"]["after"]),
        "",
        "## Repeatability",
        "",
        "- StateV1 hashes identical: `%s`; raw action hashes identical: `%s`; decoded action hashes identical: `%s`." % (value["repeatability"]["state_v1_sha256_identical"], value["repeatability"]["raw_action_sha256_identical"], value["repeatability"]["decoded_action_sha256_identical"]),
        "- Hash differences are recorded for diagnosis and do not fail the contract when lifecycle, timing, counts, SafetyGate admissibility, and completion pass.",
        "",
        "## Boundary",
        "",
        "This remains one TINY1 development fixture at seed 4242. It is not a robustness, generalization, real-time, or hardware-readiness claim.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--seal-a", type=Path, required=True)
    parser.add_argument("--seal-b", type=Path, required=True)
    parser.add_argument("--predecessor-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_report(args.baseline_profile, args.run_a, args.run_b, args.seal_a, args.seal_b, args.predecessor_seal, args.output, args.report)
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
