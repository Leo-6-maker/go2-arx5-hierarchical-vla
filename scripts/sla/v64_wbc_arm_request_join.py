"""Test exact, causal selected-grid to arm-request association on closed cards."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from v64_observation_readback import rows, tick
from v64_selected_policy_match import digest, identity
from v64_wbc_entry_readback import ARM_LOOKAHEAD, JOINTS, sha, stamp


def arm_trajectory(policy):
    times = np.asarray(policy["time_trajectory"])
    poses = np.asarray([row["value"][-6:] for row in policy["state_trajectory"]])
    if (len(times) == 0 or poses.shape != (len(times), 6) or not np.isfinite(poses).all()
            or not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
        return None
    return times, poses.T


def interpolate_signature(trajectory, observation):
    times, positions = trajectory
    request_time = min(observation + ARM_LOOKAHEAD, times[-1])
    if request_time < times[0]:
        return None
    value = [np.interp(request_time, times, col) for col in positions]
    return np.asarray(np.clip(value, -3.2, 3.7), dtype="<f4").tobytes()


def position_signature(policy, observation):
    trajectory = arm_trajectory(policy)
    return interpolate_signature(trajectory, observation) if trajectory else None


def analyze(case):
    definition = case / "EXECUTION_SPEC.json"
    lease = json.loads((case / "control/LEASE.json").read_text())
    if lease["status"] != "CLOSED" or lease["spec_sha256"] != sha(definition):
        raise ValueError("The source case must be closed and its original spec intact")
    terminal = json.loads((case / "RESULT.private.json").read_text())
    if not terminal["shutdown_clean"] or terminal["unsafe"] or terminal["technical_failure"]:
        raise ValueError("Expected a clean finished diagnostic case")
    start, end = (terminal["detail"][key] for key in ("native_step_start", "terminal_native_step"))
    raw = case / "native-01"
    trace = raw / "SELECTED_CONTROLLER.private.jsonl"
    archive = raw / "ROS_CONSUMPTION.private.jsonl"
    records = (record for _, record in rows(trace))
    header = next(records)
    entries = list(records)
    if (header["schema"] != "OCS2_SELECTED_REFERENCE_BREAKPOINT_02" or header["fixture"]
            or [r["sequence"] for r in entries] != list(range(1, len(entries)+1))):
        raise ValueError("The raw selected-control sequence is incomplete")
    wanted = [(item["mpc_init_time_s"], item["trajectory_sha256"], item["active_solution_sha256"]) for item in entries]
    needed = set(wanted)
    clocks = {record[0] for record in needed}
    policies, requests = defaultdict(dict), defaultdict(list)
    stamp_population = Counter()
    for line, item in rows(archive):
        name, message = item["topic_key"], item["message"]
        if name == "policy" and message["init_observation"]["time"] in clocks:
            key = identity(message)
            if key in needed:
                policies[key][digest(message)] = message
        elif name == "arm_request":
            (point,) = message["points"]
            values = np.asarray(point["positions"], dtype="<f4")
            if values.shape != (6,) or message["joint_names"] != JOINTS or not np.isfinite(values).all():
                raise ValueError("Unexpected frozen arm request structure")
            measured = tick(stamp(message))
            requests[values.tobytes()].append((line, measured, item["received_monotonic_s"]))
            stamp_population[measured] += 1
    candidate_matches, inverse = {}, defaultdict(set)
    unresolved_policy = set()
    for item, key in zip(entries, wanted, strict=True):
        seq = item["sequence"]
        if len(policies[key]) != 1:
            unresolved_policy.add(seq)
            continue
        value = position_signature(next(iter(policies[key].values())), item["native_observation_time_s"])
        if value is None:
            unresolved_policy.add(seq)
            continue
        observed = tick(item["native_observation_time_s"])
        entered = item["monotonic_ns"] / 1e9
        options = [(line, clock) for line, clock, received in requests[value] if clock >= observed and received >= entered]
        candidate_matches[seq] = options
        for line, _ in options:
            inverse[line].add(seq)
    counts, offsets, bindings = Counter(), Counter(), []
    for item in entries:
        observed = tick(item["native_observation_time_s"])
        if not start <= observed < end:
            continue
        seq = item["sequence"]
        counts["task_selected_calls"] += 1
        counts["no_original_same_tick_receipt"] += int(stamp_population[observed] == 0)
        options = candidate_matches.get(seq, [])
        if len(options) == 1 and inverse[options[0][0]] == {seq}:
            line, clock = options[0]
            counts["unique_exact_positions_and_causality"] += 1
            counts["recovered_without_same_tick_receipt"] += int(stamp_population[observed] == 0)
            offsets[clock-observed] += 1
            bindings.append(dict(controller_sequence=seq, observed_tick=observed, request_tick=clock, ros_line=line))
        elif seq in unresolved_policy:
            counts["no_unique_usable_policy"] += 1
        elif options:
            counts["ambiguous_exact_content"] += 1
        else:
            counts["no_causal_exact_content"] += 1
    return dict(case=case.name, native_range=[start, end], counts=dict(counts), request_minus_observation_ticks=dict(offsets),
                trace_sha256=sha(trace), receipt_sha256=sha(archive), exact_matches=bindings)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    opts = cli.parse_args()
    records = [analyze(case) for case in opts.cases]
    with opts.output.open("x") as stream:
        json.dump(dict(records=records, native_integrations=0, model_forwards=0), stream, indent=2, allow_nan=False)
        stream.write("\n")
