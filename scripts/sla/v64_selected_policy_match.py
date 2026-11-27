"""Match an active OCS2 command and solution to exact emitted solver contents."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(message):
    target = message["plan_target_trajectories"]
    command = dict(times=target["time_trajectory"], poses=[item["value"] for item in target["state_trajectory"]],
                   inputs=[item["value"] for item in target["input_trajectory"]])
    mode = message["mode_schedule"]
    solution = dict(times=message["time_trajectory"], states=[item["value"] for item in message["state_trajectory"]],
                    inputs=[item["value"] for item in message["input_trajectory"]], post_events=message["post_event_indices"],
                    event_times=mode["event_times"], modes=mode["mode_sequence"])
    return message["init_observation"]["time"], digest(command), digest(solution)


def match(trace, archive, *, fixture):
    with trace.open() as stream:
        header = json.loads(next(stream))
        observed = [json.loads(line) for line in stream]
    if (header["schema"] != "OCS2_SELECTED_REFERENCE_BREAKPOINT_02" or header["fixture"] != fixture
            or not observed or len(observed) > header["max_samples"]
            or observed[-1].get("diagnostic_capture_limit_reached")
            or [row["sequence"] for row in observed] != list(range(1, len(observed) + 1))):
        raise ValueError("Unexpected observer configuration or a missing sequence")
    wanted = {(row["mpc_init_time_s"], row["trajectory_sha256"], row["active_solution_sha256"]) for row in observed}
    times = {key[0] for key in wanted}
    hits = defaultdict(dict)
    with archive.open() as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["topic_key"] != "policy":
                continue
            message = row["message"]
            if message["init_observation"]["time"] not in times:
                continue
            key = identity(message)
            if key in wanted:
                wire = digest(message)
                hits[key].setdefault(wire, []).append(dict(line=index, monotonic_s=row["received_monotonic_s"]))
    counts, matches = defaultdict(int), []
    for row in observed:
        key = (row["mpc_init_time_s"], row["trajectory_sha256"], row["active_solution_sha256"])
        witnesses = hits[key]
        status = {0: "MISSING", 1: "UNIQUE_CONTENT"}.get(len(witnesses), "AMBIGUOUS_WIRE")
        counts[status] += 1
        matches.append(dict(sequence=row["sequence"], eval_time_s=row["eval_time_s"], receipt_status=status,
                            matched_messages=witnesses))
    return dict(fixture=fixture, observer_events=len(observed), unique_selected_states=len(wanted),
                counts=dict(counts), exact_content_only=True, matches=matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "archive", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--fixture", action="store_true")
    options = vars(parser.parse_args())
    output = options.pop("output")
    with output.open("x") as stream:
        json.dump(match(**options), stream, indent=2, allow_nan=False)
        stream.write("\n")
