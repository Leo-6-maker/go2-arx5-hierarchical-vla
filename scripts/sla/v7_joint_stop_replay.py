"""Identify the registered joint-limit trigger from the exact closed Treatment HOLD archive."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write


RESULT = "19d3e367ea56a2cb15ae4e05c3f2303d95ee37fcf3ba1187a6ba9c307a22f25f"
SCORE = "40e94422f07b363c2e7a2c4904c1cccba6fedbd26c1b22fd7abc7f1ced3fc7ff"
OBSERVER = "7ddd5bcd44622cba59341029815ae24493be4e89d8f087f2e65f4b061389bb55"
LIBRARY = "43de4483b6be58d9e983fb0029a19b7932a87a9cf174e577065527609431ca83"
PERIOD_S = .020
SLIDE, HINGE = 2, 3
EPSILON = {SLIDE: 1e-6, HINGE: 1e-4}
IMMEDIATE = {SLIDE: 1e-4, HINGE: 1e-3}


def latest_samples(path, stop, steps):
    digest = hashlib.sha256()
    samples = deque(maxlen=steps + 1)
    with path.open("rb") as stream:
        raw = next(stream)
        digest.update(raw)
        header = json.loads(raw)
        terminal = None
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row["kind"] == "sample":
                samples.append(row)
            elif row["kind"] == "stop":
                require(terminal is None, "More than one original STOP")
                terminal = row
    require(terminal == {"kind": "stop", "reason": "REGISTERED_JOINT_SAFETY", "steps": stop}
            and list(s["step"] for s in samples) == list(range(stop - steps, stop + 1))
            and not any(r["joint_stop"] for r in list(samples)[:-1]) and samples[-1]["joint_stop"],
            "Missing exact consecutive samples at the original joint STOP")
    return header, list(samples), digest.hexdigest()


def registered_crossing(header, states, ticks):
    violations = []
    for row in header["joints"]:
        joint = row["type"]
        if not row["limited"] or joint not in EPSILON:
            continue
        lower, upper = row["range"]
        q = [sample["qpos"][row["qposadr"]] for sample in states]
        excess = [max(0., lower - value, value - upper) for value in q]
        persistent = all(value > EPSILON[joint] for value in excess[-ticks:])
        immediate = excess[-1] > IMMEDIATE[joint]
        if persistent or immediate:
            require(immediate or excess[0] <= EPSILON[joint], "Cannot locate the start of this persistent violation")
            violations.append(dict(joint=row["name"], range=row["range"], terminal_qpos=q[-1],
                                   terminal_excess=excess[-1], immediate_threshold=IMMEDIATE[joint],
                                   persistent_threshold=EPSILON[joint], persistence_required_s=PERIOD_S,
                                   terminal_trigger="IMMEDIATE" if immediate else "PERSISTENT",
                                   samples=[dict(tick=s["step"], qpos=position, excess=amount)
                                            for s, position, amount in zip(states, q, excess, strict=True)]))
    require(len(violations) == 1, "The registered STOP does not have a unique reproduced joint trigger")
    return violations[0]


def audit(case, score_file, build):
    pin(build / "serialized_observer.cpp", OBSERVER)
    library = pin(build / "terminal_step.so", LIBRARY)
    result = read(pin(case / "RESULT.private.json", RESULT))
    score = read(pin(score_file, SCORE))
    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED" and score["result_sha256"] == RESULT and score["after_native_stop"] == 0,
            "Only the original closed scored sample may be used")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    pin(spec["native_binary"], spec["frozen_files"][spec["native_binary"]])
    native = case / "native-01"
    process, = [row for row in read(native / "PROCESS_COMMANDS.private.json") if row["name"] == "sim"]
    require(any(str(library) in field.removeprefix("LD_PRELOAD=").split(":")
                for field in process["argv"] if field.startswith("LD_PRELOAD=")), "Wrong original simulator guard library")
    stop = result["actual_native_integrations"]
    require(stop == score["actual_stop"]["native_tick"] and result["trace_audit"]["integrations_after_first_STOP"] == 0,
            "Simulation STOP ledger disagrees")
    steps = 10
    header, samples, physics = latest_samples(native / "physics.jsonl", stop, steps)
    require(header["timestep"] * steps == PERIOD_S and not samples[-1]["illegal_contact"], "Wrong original clock or STOP kind")
    trigger = registered_crossing(header, samples, steps)
    request = read(native / "g7-queries" / "0084/QUERY_ACTUAL.private.json")
    last = request["confirmed_intervals"][-1]
    require(request["full_H40_compiled"] and request["terminal_native_step"] == stop
            and last["native_end_step"] + request["partial_native_ticks"] == stop
            and last["accepted_boundary"]["exact_receiver_knot"], "Missing the last confirmed command before STOP")
    return dict(case=case.name, source_result_sha256=RESULT, independent_score_sha256=SCORE,
                observer_source_sha256=OBSERVER, observed_guard_library_sha256=LIBRARY,
                physics_sha256=physics, native_tick=stop, trigger=trigger,
                last_confirmed_jaw_command_m=last["emitted_gripper_width_m"],
                last_confirmed_command_end_tick=last["native_end_step"], partial_ticks_before_STOP=request["partial_native_ticks"],
                integrations_after_STOP=0, new_native_integrations=0, new_model_forwards=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("case", "score_file", "build", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    write(args.output, audit(args.case, args.score_file, args.build))
