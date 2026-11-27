"""Measure read-side policy association on closed, full N1 episodes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from v64_arm_request_shadow import ArmReceiptMatcher, REQUEST_STAMP_SLACK_TICKS
from v64_observation_readback import grant_timing, receiver, rows, tick
from v64_wbc_entry_readback import JOINTS, sha, stamp


def analyze(case, *, window=None):
    terminal = case / "RESULT.private.json"
    spec = case / "EXECUTION_SPEC.json"
    lease = json.loads((case / "control/LEASE.json").read_bytes())
    result = json.loads(terminal.read_bytes())
    if (lease["status"] != "CLOSED" or lease["spec_sha256"] != sha(spec)
            or not result["shutdown_clean"] or result["technical_failure"]):
        raise ValueError("Expected a closed, accounted, originally executed N1 case")

    raw = case / "native-01"
    packet_path = raw / "MOBILE_SAMPLES.private.jsonl"
    packets = [row for _, row in rows(packet_path) if row["segment"] == "MOBILE_TASK"]
    if not packets:
        raise ValueError("No completed task intervals")
    start = packets[0]["native_start_step"]
    end = packets[-1]["native_end_step"]
    intervals = {p["native_start_step"]: p["installed_reference_sha256"] for p in packets}
    if (list(intervals) != list(range(start, end, 25)) or len(intervals) != len(packets)
            or any(p["native_end_step"] != p["native_start_step"] + 25 or not p["safety_valid"] for p in packets)
            or end > result["actual_native_integrations"]):
        raise ValueError("Expected an uninterrupted, safe, actually completed task prefix")

    window_steps = set()
    if window is not None:
        window_start, window_end = window
        if not start <= window_start < window_end <= end or any((step - start) % 25 for step in window):
            raise ValueError("Require a diagnostic window on the completed 50 ms grid")
        window_steps = set(range(window_start, window_end, 25))

    handoff_path = raw / "RECOVERY-HANDOFF-INTENT.private.json"
    handoff = None
    handoff_intervals = {}
    if handoff_path.exists():
        intent = json.loads(handoff_path.read_bytes())
        handoff = intent["native_step"]
        query_dir = raw / "g7-queries" / f"{intent['actual_student_queries'] - 1:04d}"
        last_query = json.loads((query_dir / "QUERY_ACTUAL.private.json").read_bytes())
        prior_wire = last_query["confirmed_intervals"][-1]["wire_sha256"]
        if (not intent["clock_owner_unchanged"] or intent["next_role"] != "TEACHER"
                or not last_query["full_P10"] or last_query["terminal_native_step"] != handoff
                or intervals.get(handoff - 25) != prior_wire
                or intent.get("original_wire_sha256", prior_wire) != prior_wire):
            raise ValueError("The recorded Recovery handoff must name the actual preceding installed wire")
        handoff_intervals = {step: Counter() for step in range(handoff - 50, handoff + 50, 25) if step in intervals}

    installed, _, expired, _ = receiver(raw)
    if expired:
        raise ValueError("Original receiver recorded an expired acceptance")
    ros = raw / "ROS_CONSUMPTION.private.jsonl"
    matcher = ArmReceiptMatcher()
    request_counts, binding_counts, offsets = Counter(), Counter(), Counter()
    per_atom = {step: Counter() for step in intervals}
    first_current = {}
    for _, row in rows(ros):
        topic = row["topic_key"]
        msg = row["message"]
        if topic == "policy":
            matcher.policy(msg)
        if topic != "arm_request":
            continue
        stamped = tick(stamp(msg))
        if not start + REQUEST_STAMP_SLACK_TICKS <= stamped < end:
            continue
        (point,) = msg["points"]
        if (msg["joint_names"] != JOINTS or len(point["positions"]) != len(JOINTS)
                or not np.isfinite(point["positions"]).all()):
            raise ValueError("Unexpected arm controller request")
        candidates = matcher.request(msg)
        solution_ids = {key for _, key in candidates}
        control_ticks = {step for step, _ in candidates}
        if len(candidates) != 1:
            request_counts["no_candidate" if not candidates else "multiple_candidates"] += 1
            if candidates and len(solution_ids) == 1:
                request_counts["multiple_ticks_same_solution"] += 1
            if candidates and len(control_ticks) == 1:
                request_counts["one_tick_multiple_solutions"] += 1
            continue
        request_counts["unique_tick_and_solution"] += 1
        control_tick, (_, target, _) = next(iter(candidates))
        offsets[stamped - control_tick] += 1
        boundary = start + ((control_tick - start) // 25) * 25
        eligible = {event["wire"] for event in installed.get(target, ())
                    if event["installed_monotonic_s"] <= row["received_monotonic_s"]}
        current = intervals[boundary]
        previous = intervals.get(boundary - 25)
        if eligible == {current}:
            status = "current"
        elif previous is not None and eligible == {previous}:
            status = "prior"
        elif not eligible:
            status = "no_captured_install"
        elif len(eligible) > 1:
            status = "multiple_installs"
        else:
            status = "other_install"
        binding_counts[status] += 1
        per_atom[boundary][status] += 1
        if boundary in handoff_intervals:
            handoff_intervals[boundary][status] += 1
        if status == "current" and (boundary in handoff_intervals or boundary in window_steps):
            first_current.setdefault(boundary, (control_tick, row["received_monotonic_s"]))

    summary = dict(case=case.name, terminal_sha256=sha(terminal), ros_sha256=sha(ros),
                receiver_sha256=sha(raw / "planner.jsonl"), intervals_sha256=sha(packet_path),
                native_range=[start, end], requests=dict(request_counts), inferred_install_binding=dict(binding_counts),
                receipt_minus_inferred_ticks=dict(offsets), full_task_intervals=len(intervals),
                intervals_with_current_inference=sum(bool(counter["current"]) for counter in per_atom.values()),
                mixed_current_prior_intervals=sum(bool(counter["current"] and counter["prior"]) for counter in per_atom.values()),
                no_unique_request_intervals=sum(not counter for counter in per_atom.values()),
                native_stop=result["physics_stops"], independent_controller_witness=False)
    if handoff is not None:
        summary["recovery_handoff"] = dict(native_step=handoff, intent_sha256=sha(handoff_path),
            first_current_offsets_ticks={step: first_current[step][0] - step
                                         for step in handoff_intervals if step in first_current},
            inferred_bindings_by_interval=handoff_intervals,
            ros_receipt_clock_timings=grant_timing(raw, {step: first_current[step]
                for step in handoff_intervals if step in first_current}, result["actual_native_integrations"]))
    if window_steps:
        summary["diagnostic_window"] = [
            dict(start_tick=step, end_tick=packet["native_end_step"], inferred_install_binding=per_atom[step],
                 first_current_offset_ticks=first_current[step][0] - step if step in first_current else None,
                 accepted_endpoint_z_m=packet["target_endpoint_EEF10"][2],
                 actual_tcp_z_m=packet["actual_TCP"][2][3], actual_mount_z_m=packet["actual_mount"][2][3],
                 actual_base_z_m=packet["qpos"][2])
            for packet in packets if (step := packet["native_start_step"]) in window_steps
        ]
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    cli.add_argument("--window", type=int, nargs=2, metavar=("START_TICK", "END_TICK"))
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [analyze(case, window=args.window) for case in args.cases]
    with args.output.open("x") as stream:
        json.dump(dict(records=records, physics_ticks=0, model_forwards=0), stream, indent=2, allow_nan=False)
        stream.write("\n")
