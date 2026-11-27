"""Join the selected-policy witness to the original clock and receiver evidence."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from v64_selected_policy_match import digest, match


def rows(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            yield number, json.loads(line)


def tick(time_s):
    step = round(time_s / .002)
    if abs(time_s - step * .002) > 1e-8:
        raise ValueError(f"Expected the original 2 ms native clock, got {time_s}")
    return step


def receiver(raw):
    proposed = {row["wire_sha256"]: row for _, row in rows(raw/"COMMANDS.private.jsonl")
                if row["phase"] == "proposed"}
    accepted = {}
    installed = defaultdict(list)
    expired, automatic = 0, 0
    for line, receipt in rows(raw/"planner.jsonl"):
        kind = receipt["kind"]
        if kind not in ("accept", "installed"):
            continue
        wire = receipt["ref_id"]
        if kind == "accept":
            p = proposed[wire]
            if receipt["monotonic_s"] < p["wall_monotonic_s"]:
                raise ValueError("Receiver accepted a wire before its proposal")
            accepted[wire] = receipt
            expired += int(receipt["receiver_mpc_s"] > receipt["deadline"])
        if kind == "installed":
            if wire not in accepted:
                automatic += 1
                continue
            p, ack = proposed[wire], accepted[wire]
            ts = receipt["monotonic_s"]
            if not p["wall_monotonic_s"] <= ack["monotonic_s"] <= ts:
                raise ValueError("Proposal, acceptance and verified installation are out of order")
            target = digest({k: receipt[k] for k in ("times", "poses", "inputs")})
            installed[target].append(dict(line=line, wire=wire, sequence=p["sequence"],
                installed_monotonic_s=ts, observation_s=receipt["observation_time_s"],
                proposal_native_s=p["sim_time_s"], proposal_deadline_s=p["deadline_s"]))
    return installed, proposed, expired, automatic


def grant_timing(raw, first_active, end):
    grants = defaultdict(list)
    for _, receipt in rows(raw/"CLOCK_RESUME_REQUESTS.jsonl"):
        grants[receipt["requested_step"]].append(receipt["monotonic_s"])
    elapsed, allowed_after = [], []
    for start, (_, entered) in first_active.items():
        (grant,) = grants[start+25]
        elapsed.append((entered-grant)*1000)
        for next_grant in grants[start+50]:
            allowed_after.append((next_grant-entered)*1000)

    def describe(values):
        if not values:
            return None
        ordered = sorted(values)
        return dict(count=len(ordered), minimum=ordered[0], p50=median(ordered),
                    p95=ordered[(95*len(ordered)+99)//100-1], maximum=ordered[-1])

    return dict(grants_after_terminal_step=sum(len(values) for step, values in grants.items() if step > end),
                active_before_current_clock_grant=sum(x < 0 for x in elapsed),
                grant_to_first_active_ms=describe(elapsed),
                first_active_to_following_grant_ms=describe(allowed_after),
                basis="ORIGINAL_GRANTED_CLOCK_TRACE_NO_CHANGE_TO_FIXED_WAIT")


def readback(case):
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    raw = case/"native-01"
    final = json.loads((case/"RESULT.private.json").read_text())
    detail = final["detail"]
    all_steps = set()
    for row in packets(raw/"state.bin.journal"):
        if tick(row["time_s"]) != row["step"]:
            raise ValueError("The original native clock differs from the saved step")
        all_steps.add(row["step"])
    begin = detail["native_step_start"]
    end = detail["terminal_native_step"]
    all_atoms = [row for _, row in rows(raw/"MOBILE_SAMPLES.private.jsonl")]
    atoms = [row for row in all_atoms if begin <= row["native_start_step"] < end]
    starts = [row["native_start_step"] for row in atoms]
    if (starts != list(range(begin, end, 25)) or
            any(row["native_end_step"] != row["native_start_step"]+25 or row["ticks"] != 25 for row in atoms)):
        raise ValueError("The original contiguous 50 ms receipt grid is absent")
    interval_targets = {row["native_start_step"]: set(row["solver_exact_installed_targets"]) for row in all_atoms}
    arm_stamps, chronological_arms = defaultdict(list), []
    for line, row in rows(raw/"ROS_CONSUMPTION.private.jsonl"):
        if row["topic_key"] != "arm_request":
            continue
        stamp = row["message"]["header"]["stamp"]
        stamp_ns = stamp["sec"] * 10**9 + stamp["nanosec"]
        step = tick(stamp_ns/10**9)
        arm = dict(line=line, monotonic_s=row["received_monotonic_s"], stamp_native_step=step)
        arm_stamps[step].append(arm)
        chronological_arms.append(arm)
    installed, proposals, expired, automatic = receiver(raw)
    live = raw/"SELECTED_CONTROLLER.private.jsonl"
    joins = match(live, raw/"ROS_CONSUMPTION.private.jsonl", fixture=False)
    calls = [row for index, row in rows(live) if index > 1]
    marks, details = Counter(), []
    coverage = Counter()
    first_active, returned_to_older = {}, set()
    observed_order = []
    for row, wire_match in zip(calls, joins["matches"], strict=True):
        source = tick(row["native_observation_time_s"])
        entered = row["monotonic_ns"]/10**9
        observed_order.append((source, row["monotonic_ns"]))
        if not begin <= source < end:
            marks["outside_task_range"] += 1
            continue
        atom = atoms[bisect_right(starts, source)-1]
        start = atom["native_start_step"]
        target = row["trajectory_sha256"]
        coverage[start] += 1
        candidates = [r for r in installed[target] if r["installed_monotonic_s"] <= entered]
        unique_installs = {r["wire"] for r in candidates}
        marks["task_selected"] += 1
        marks["no_native_packet"] += int(source not in all_steps)
        marks["no_same_tick_arm_request"] += int(not arm_stamps[source])
        marks["no_matched_solver_payload"] += int(wire_match["receipt_status"] == "MISSING")
        marks["ambiguous_solver_payload"] += int(wire_match["receipt_status"] == "AMBIGUOUS_WIRE")
        marks["no_causal_install"] += int(not candidates)
        marks["ambiguous_install"] += int(len(unique_installs) > 1)
        current = interval_targets[start]
        marks["two_current_candidates"] += int(len(current) > 1)
        marks["bound_to_consumed_interval"] += int(target in current)
        marks["only_previous_interval_in_adjacent_pair"] += int(
            target not in current and target in interval_targets.get(start-25, ()))
        active = atom["installed_reference_sha256"]
        if unique_installs == {active}:
            first_active.setdefault(start, (source-start, entered))
        elif unique_installs:
            marks["selected_other_installed_wire"] += 1
            if start in first_active:
                returned_to_older.add(start)
        marks["arm_stamp_repeated"] += int(len(arm_stamps[source]) > 1)
        details.append(dict(sequence=row["sequence"], native_step=source, entered_ns=row["monotonic_ns"],
            selected_target=row["trajectory_sha256"], selected_solution=row["active_solution_sha256"],
            selected_solver_receipts=wire_match["matched_messages"], same_tick_arm_requests=arm_stamps[source],
            confirmed_atom=[atom["native_start_step"],atom["native_end_step"]],
            emitted_plan_wire=atom["installed_reference_sha256"], install_candidates=candidates))
    marks["clock_or_host_order_regressions"] = sum(b[0] < a[0] or b[1] <= a[1]
        for a, b in zip(observed_order, observed_order[1:]))
    ordinal = dict(selected_calls=len(calls), arm_request_receipts=len(chronological_arms),
                   identity_established=False)
    if len(calls) == len(chronological_arms):
        ordinal["stamp_minus_observation_ticks"] = dict(Counter(
            arm["stamp_native_step"] - step for (step, _), arm in zip(observed_order, chronological_arms, strict=True)))
        ordinal["early_receipts"] = sum(arm["monotonic_s"] < ns/1e9
            for (_, ns), arm in zip(observed_order, chronological_arms, strict=True))
    stop = detail.get("expected_project_STOP")
    timing = grant_timing(raw, first_active, end)
    proposal_past_deadline = sum(p["sim_time_s"] > p["deadline_s"] for p in proposals.values())
    duration = sorted(row["observer_read_ns"]/10**6 for row in calls)
    return dict(status="OBSERVED_DIAGNOSTIC_EVIDENCE", scope="SINGLE_REGISTERED_WORLD", native_range=[begin, end],
                controller_calls=len(calls), coverage=dict(marks), all_native_packets=len(all_steps),
                arm_request_order_diagnostic=ordinal,
                first_active_wire_ticks=dict(Counter(ticks for ticks, _ in first_active.values())),
                intervals_reverting_after_active_wire=len(returned_to_older),
                fully_clocked_intervals=len(coverage), native_intervals=len(atoms),
                observer_read_ms=dict(p50=median(duration),p95=duration[(95*len(duration)+99)//100-1],maximum=max(duration)),
                project_stop=stop, clean_shutdown=final["shutdown_clean"], grant_timing=timing,
                grants_after_terminal_step=timing["grants_after_terminal_step"],
                late_proposals=proposal_past_deadline, late_receiver_accepts=expired, automatic_start_installs=automatic,
                terminal_integrations=final["actual_native_integrations"], records=details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    result = readback(options.case)
    with options.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
