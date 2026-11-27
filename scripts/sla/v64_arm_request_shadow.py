"""Replay arm-request identity from ROS receipts, scoring against the debugger trace."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
from struct import Struct
from time import perf_counter_ns

import numpy as np

from v64_observation_readback import rows, tick
from v64_selected_policy_match import identity
from v64_wbc_arm_request_join import arm_trajectory
from v64_wbc_entry_readback import ARM_LOOKAHEAD, TICK, sha, stamp


DELAY_COMPENSATION = .010
REQUEST_STAMP_SLACK_TICKS = 3
PROJECTION_CACHE_SIZE = 2 * (REQUEST_STAMP_SLACK_TICKS + 1)
PACK_POSITIONS = Struct("<6f").pack


class ArmProjection:
    """Reuse one observed policy's six-channel interpolation and recent results."""

    def __init__(self, trajectory):
        times, positions = trajectory
        self.times = tuple(times.tolist())
        self.positions = positions.T.tolist()
        self.slopes = (np.diff(positions, axis=1) / np.diff(times)).T.tolist()
        self.recent = {}

    def request(self, observed, time):
        if observed in self.recent:
            return self.recent[observed]
        request_time = min(time + ARM_LOOKAHEAD, self.times[-1])
        index = bisect_right(self.times, request_time) - 1
        if index < 0:
            return None
        positions = self.positions[index]
        if request_time != self.times[index]:
            offset = request_time - self.times[index]
            positions = [start + offset*slope for start, slope in zip(positions, self.slopes[index], strict=True)]
        signature = PACK_POSITIONS(*(min(3.7, max(-3.2, position)) for position in positions))
        if len(self.recent) >= PROJECTION_CACHE_SIZE:
            del self.recent[next(iter(self.recent))]
        self.recent[observed] = signature
        return signature


def predict(recent, stamp_tick, signature):
    found = set()
    for observed in range(stamp_tick-REQUEST_STAMP_SLACK_TICKS, stamp_tick+1):
        time = observed * TICK
        evaluation = time + DELAY_COMPENSATION
        for projection, reference in recent:
            grid = projection.times
            if (reference[0] <= time + 1e-8 and grid[0] <= evaluation <= grid[-1]
                    and signature == projection.request(observed, time)):
                found.add((observed, reference))
    return found


class ArmReceiptMatcher:
    """Consume one independent ROS receipt stream in arrival order."""

    def __init__(self):
        self.recent = []

    def policy(self, message):
        if message["time_trajectory"]:
            trajectory = arm_trajectory(message)
            if trajectory is not None:
                self.recent.append((ArmProjection(trajectory), identity(message)))

    def request(self, message, *, match=True):
        stamp_tick = tick(stamp(message))
        earliest = (stamp_tick - REQUEST_STAMP_SLACK_TICKS) * TICK + DELAY_COMPENSATION
        self.recent = [(projection, reference) for projection, reference in self.recent if projection.times[-1] >= earliest]
        if not match:
            return set()
        signature = np.asarray(message["points"][0]["positions"], dtype="<f4").tobytes()
        return predict(self.recent, stamp_tick, signature)


def analyze(case, proof, *, profile=False):
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

    matcher, counts, interpretations = ArmReceiptMatcher(), Counter(), []
    compute_ns, message_ns, queue_sizes = [], [], []
    for line, row in rows(archive):
        topic, message = row["topic_key"], row["message"]
        if topic == "policy":
            started = perf_counter_ns() if profile else 0
            matcher.policy(message)
            if profile:
                message_ns.append(perf_counter_ns()-started)
            continue
        if topic != "arm_request":
            continue
        verified = line in truth
        started = perf_counter_ns() if profile and verified else 0
        candidates = matcher.request(message, match=verified)
        if not verified:
            continue
        if profile:
            compute_ns.append(perf_counter_ns()-started)
            queue_sizes.append(len(matcher.recent))
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
    result = dict(case=case.name, observer_scored_requests=len(truth), counts=dict(counts), interpretations=interpretations)
    if profile:
        def measure(samples):
            return dict(samples=len(samples), wall_p50_ms=float(np.median(samples))/1e6,
                        wall_p95_ms=float(np.percentile(samples,95))/1e6, maximum_ms=max(samples)/1e6,
                        total_s=sum(samples)/1e9)

        result["passive_match_cost"] = dict(request=measure(compute_ns), policy_receipt=measure(message_ns),
                                            maximum_candidate_policies=max(queue_sizes))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("truth", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    cli.add_argument("--profile", action="store_true")
    args = cli.parse_args()
    archive = {r["case"]: r for r in json.loads(args.truth.read_text())["records"]}
    result = [analyze(case, archive[case.name], profile=args.profile) for case in args.cases]
    with args.output.open("x") as stream:
        json.dump(dict(records=result, truth_receipt_sha256=sha(args.truth), native_integrations=0,
                       model_forwards=0, diagnostic_only=True), stream, allow_nan=False, indent=2)
        stream.write("\n")
