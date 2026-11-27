"""Measure read-side policy association on closed, full N1 episodes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from v64_arm_request_shadow import ArmReceiptMatcher, REQUEST_STAMP_SLACK_TICKS
from v64_observation_readback import receiver, rows, tick
from v64_wbc_entry_readback import JOINTS, sha, stamp


def analyze(case):
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

    installed, _, expired, _ = receiver(raw)
    if expired:
        raise ValueError("Original receiver recorded an expired acceptance")
    ros = raw / "ROS_CONSUMPTION.private.jsonl"
    matcher = ArmReceiptMatcher()
    request_counts, binding_counts, offsets = Counter(), Counter(), Counter()
    per_atom = {step: Counter() for step in intervals}
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

    return dict(case=case.name, terminal_sha256=sha(terminal), ros_sha256=sha(ros),
                receiver_sha256=sha(raw / "planner.jsonl"), intervals_sha256=sha(packet_path),
                native_range=[start, end], requests=dict(request_counts), inferred_install_binding=dict(binding_counts),
                receipt_minus_inferred_ticks=dict(offsets), full_task_intervals=len(intervals),
                intervals_with_current_inference=sum(bool(counter["current"]) for counter in per_atom.values()),
                mixed_current_prior_intervals=sum(bool(counter["current"] and counter["prior"]) for counter in per_atom.values()),
                no_unique_request_intervals=sum(not counter for counter in per_atom.values()),
                native_stop=result["physics_stops"], independent_controller_witness=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [analyze(case) for case in args.cases]
    with args.output.open("x") as stream:
        json.dump(dict(records=records, physics_ticks=0, model_forwards=0), stream, indent=2, allow_nan=False)
        stream.write("\n")
