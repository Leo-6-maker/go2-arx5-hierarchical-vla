"""Replay arm-request identity from ROS receipts, scoring against the debugger trace."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from v64_observation_readback import rows, tick
from v64_selected_policy_match import identity
from v64_wbc_arm_request_join import arm_trajectory, interpolate_signature
from v64_wbc_entry_readback import TICK, sha, stamp


DELAY_COMPENSATION = .010
REQUEST_STAMP_SLACK_TICKS = 3


def predict(recent, stamp_tick, signature):
    found = set()
    for observed in range(stamp_tick-REQUEST_STAMP_SLACK_TICKS, stamp_tick+1):
        time = observed * TICK
        evaluation = time + DELAY_COMPENSATION
        for trajectory, reference in recent:
            grid = trajectory[0]
            if (reference[0] <= time + 1e-8 and grid[0] <= evaluation <= grid[-1]
                    and signature == interpolate_signature(trajectory, time)):
                found.add((observed, reference))
    return found


def analyze(case, proof):
    native = case / "native-01"
    observation = native / "SELECTED_CONTROLLER.private.jsonl"
    archive = native / "ROS_CONSUMPTION.private.jsonl"
    if proof["case"] != case.name or (sha(observation), sha(archive)) != (proof["trace_sha256"], proof["receipt_sha256"]):
        raise ValueError("The fixed observer-scored case and ROS receipts differ")
    observed = {r["sequence"]: r for i, r in rows(observation) if i > 1}
    truth = {}
    for binding in proof["exact_matches"]:
        call = observed[binding["controller_sequence"]]
        identity_key = tuple(call[k] for k in ("mpc_init_time_s", "trajectory_sha256", "active_solution_sha256"))
        truth[binding["ros_line"]] = (binding["observed_tick"], identity_key)
    if len(truth) != proof["counts"]["task_selected_calls"]:
        raise ValueError("Require one independently established truth for every evaluated call")

    recent, counts, interpretations = [], Counter(), []
    for line, row in rows(archive):
        topic, message = row["topic_key"], row["message"]
        if topic == "policy" and message["time_trajectory"]:
            trajectory = arm_trajectory(message)
            if trajectory is not None:
                recent.append((trajectory, identity(message)))
            continue
        if topic != "arm_request":
            continue
        request_tick = tick(stamp(message))
        earliest = (request_tick - REQUEST_STAMP_SLACK_TICKS) * TICK + DELAY_COMPENSATION
        recent = [(trajectory, ref) for trajectory, ref in recent if trajectory[0][-1] >= earliest]
        if line not in truth:
            continue
        signature = np.asarray(message["points"][0]["positions"], dtype="<f4").tobytes()
        candidates = predict(recent, request_tick, signature)
        exact = truth[line] in candidates
        if len(candidates) == 1:
            verdict = "unique_correct" if exact else "unique_wrong"
        elif candidates:
            verdict = "ambiguous_with_actual" if exact else "ambiguous_without_actual"
        else:
            verdict = "missing"
        counts[verdict] += 1
        interpretations.append(dict(ros_line=line, candidate_count=len(candidates), verdict=verdict))
    if sum(counts.values()) != len(truth):
        raise ValueError("Did not evaluate the full archive of known arm requests")
    return dict(case=case.name, observer_scored_requests=len(truth), counts=dict(counts), interpretations=interpretations)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("truth", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    args = cli.parse_args()
    archive = {r["case"]: r for r in json.loads(args.truth.read_text())["records"]}
    result = [analyze(case, archive[case.name]) for case in args.cases]
    with args.output.open("x") as stream:
        json.dump(dict(records=result, truth_receipt_sha256=sha(args.truth), native_integrations=0,
                       model_forwards=0, diagnostic_only=True), stream, allow_nan=False, indent=2)
        stream.write("\n")
