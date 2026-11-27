"""Read the accepted reference and measured TCP preceding a recorded contact STOP."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector_mm(a, b):
    return [(x - y) * 1000 for x, y in zip(a, b, strict=True)]


def review(case):
    native = case / "native-01"
    paths = dict(result=case / "RESULT.private.json", lease=case / "control/LEASE.json",
                 physics=native / "physics.jsonl", commands=native / "COMMANDS.private.jsonl",
                 samples=native / "MOBILE_SAMPLES.private.jsonl",
                 queries=native / "LIVE_DATA/QUERIES.private.json")
    result = read(paths["result"])
    stop, = result["physics_stops"]
    tick = stop["steps"]
    if (stop["reason"] != "ILLEGAL_CONTACT" or tick != result["actual_native_integrations"]
            or result["trace_audit"]["integrations_after_first_STOP"] != 0
            or read(paths["lease"])["status"] != "CLOSED" or not result["shutdown_clean"]
            or result["technical_failure"]):
        raise ValueError("Expected a closed illegal-contact terminal")
    recent = [r for r in lines(paths["samples"]) if tick - 175 <= r["native_end_step"] < tick]
    if not recent:
        raise ValueError("No confirmed boundary around the contact")
    commands = {r["native_step"]: r for r in lines(paths["commands"])
                if r["phase"] == "mobile-accepted-boundary-050" and tick - 175 <= r["native_step"] < tick}
    query = read(paths["queries"])[-1]
    last, = (r for r in lines(paths["physics"]) if r["kind"] == "sample" and r["step"] == tick)
    if not last["contact_stop"] or not last["illegal_contact"] or query["phase"] != "DESCEND":
        raise ValueError("Archived trigger and final teacher phase disagree")
    joined = []
    for row in recent:
        boundary = row["native_end_step"]
        if boundary not in commands:
            raise ValueError(f"Accepted 50 ms pose missing at tick {boundary}")
        command = commands[boundary]
        actual = [row["actual_TCP"][i][3] for i in range(3)]
        expected = row["target_endpoint_EEF10"][:3]
        accepted = command["accepted_pose7"][:3]
        if (command["accepted_wire_sha256"] != row["installed_reference_sha256"]
                or not command["exact_receiver_knot"] or math.dist(accepted, expected) > 1e-6):
            raise ValueError(f"Accepted receiver knot disagrees with archived sample at tick {boundary}")
        joined.append(dict(tick=boundary, accepted_xyz_m=accepted, actual_xyz_m=actual,
                           actual_minus_accepted_mm=vector_mm(actual, accepted),
                           receiver_wire_sha256=command["accepted_wire_sha256"],
                           selected_policy_exported=False))
    before = joined[-1]
    unconfirmed = query["intervals"][-1]
    if unconfirmed["start_native_step"] != before["tick"] or unconfirmed["confirmed_ticks"] != 0:
        raise ValueError("The partial interval is not the recorded post-boundary STOP")
    if unconfirmed["installed_reference_sha256"] != before["receiver_wire_sha256"]:
        raise ValueError("Unexpected reference identity for the partial interval")
    return dict(condition=case.name, terminal_tick=tick, post_STOP_ticks=0, final_teacher_phase=query["phase"],
                final_query_tick=query["native_step"], last_boundaries=joined,
                last_confirmed_tick=before["tick"], partial_ticks_in_physics=tick - before["tick"],
                partial_ticks_confirmed=0, actual_TCP_at_STOP_m=last["tcp_xyz"],
                actual_TCP_motion_after_confirmed_boundary_mm=vector_mm(last["tcp_xyz"], before["actual_xyz_m"]),
                previous_gripper_width_m=query["previous_confirmed_command"]["width_m"],
                native_integrations_performed_by_audit=0, inputs={name: sha(path) for name, path in paths.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", type=Path, nargs="+")
    args = parser.parse_args()
    result = [review(case) for case in args.cases]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
