"""Read the two closed native cards without starting a ROS node or integrating physics."""

import argparse
import fcntl
import math
from collections import Counter
from pathlib import Path

from identity import read, record, rows, sha


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def audit(folder):
    spec = read(folder / "EXECUTION_SPEC.json")
    result = read(folder / "RESULT.private.json")
    lease = read(folder / "control/LEASE.json")
    native = folder / result["episode"]
    ready = read(native / "GUARD_READY.json")
    commands = read(native / "PROCESS_COMMANDS.private.json")
    evidence = read(native / "LOADED_EXECUTOR_00.private.json")
    require(
        ready
        == {
            "episode": native.name,
            "pid": next(c["pid"] for c in commands if c["name"] == "planner"),
        },
        "receiver session",
    )
    require(
        lease["status"] == "CLOSED"
        and lease["spec_sha256"] == sha(folder / "EXECUTION_SPEC.json"),
        "lease",
    )
    require(
        result["completed"] and result["sealed"],
        "card did not pass; retain failure and inspect raw result",
    )
    require(all(proc["exit_code"] == 0 for proc in result["owned_exits"]), "owned exit")
    expected = {}
    for role, key in (
        ("sim", "native_binary"),
        ("solver", "solver_binary"),
        ("controller", "controller_binary"),
    ):
        expected[role] = spec["frozen_files"][spec[key]]
        require(evidence[role]["executable_sha256"] == expected[role], role)
    history = list(rows(native / "MOBILE_SAMPLES.private.jsonl"))
    after = [r for r in history if r["native_start_step"] >= 5703]
    first = next(r for r in history if r["native_end_step"] == 5703)
    target = after[0]["target_endpoint_EEF10"]
    require(
        all(r["target_endpoint_EEF10"] == target for r in after), "changing endpoint"
    )
    require(
        target[:3] == [first["actual_TCP"][i][3] for i in range(3)],
        "measured reference origin",
    )
    require(
        target[3:9] == [first["actual_TCP"][i][j] for i in range(2) for j in range(3)],
        "measured reference attitude",
    )
    actual = [[r["actual_TCP"][i][3] for i in range(3)] for r in after]
    pos = [1000 * math.dist(target[:3], p) for p in actual]
    rms = math.sqrt(sum(x * x for x in pos) / len(pos))
    require(abs(rms - result["measured"]["position_rms_mm"]) < 1e-8, "position RMS")
    refs = {r["installed_reference_sha256"] for r in after}
    installed = [
        r
        for r in rows(native / "planner.jsonl")
        if r["kind"] == "installed" and r["ref_id"] in refs
    ]
    require(
        {r["ref_id"] for r in installed} == refs, "missing measured-phase installation"
    )
    require(
        all(
            len(r["poses"]) == len(r["times"]) == len(r["inputs"])
            and all(len(p) == 14 for p in r["poses"])
            and all(b > a for a, b in zip(r["times"], r["times"][1:], strict=False))
            and abs(r["times"][-1] - r["times"][0] - 2.0) < 1e-6
            for r in installed
        ),
        "unexpected solver target dimensions",
    )
    wire_error = max(
        1000 * math.dist(p[:3], target[:3]) for r in installed for p in r["poses"]
    )
    reported = {}
    modes = []
    for r in rows(native / "ROS_CONSUMPTION.private.jsonl"):
        if r["topic_key"] == "ee":
            reported[round(r["message"]["time"], 6)] = r["message"]["state"]["value"][
                :3
            ]
        elif r["topic_key"] == "mode":
            modes.append(r["message"]["data"])
    same_time = [
        (idx, reported[key])
        for idx, row in enumerate(after)
        if (key := round(row["native_end_step"] * 0.002, 6)) in reported
    ]
    sensor_deltas = [1000 * math.dist(actual[i], p) for i, p in same_time]
    require(modes[-2:] == ["WBC", "HOLD"], "independent mode state")
    step = result["journal_integrations"]
    ack = read(native / "pause_ack.json")
    grants = list(rows(native / "CLOCK_RESUME_REQUESTS.jsonl"))
    require(
        ack["step"]
        == step
        == max(g["requested_step"] for g in grants)
        == result["native_trace"]["native_step_records"]
        <= spec["max_ticks"]
        and ack["clock_paused"]
        and not ack["running"],
        "final native count",
    )
    evidence_files = [
        "MOBILE_SAMPLES.private.jsonl",
        "CLOCK_RESUME_REQUESTS.jsonl",
        "ROS_CONSUMPTION.private.jsonl",
        "LOADED_EXECUTOR_00.private.json",
        "planner.jsonl",
        "GUARD_READY.json",
        "pause_ack.json",
        "init.trace",
        "init.trace.summary",
        "state.bin.journal",
        "physics.jsonl",
    ]
    clock = None
    if (native / "GUARD_TERMINAL.json").is_file():
        evidence_files += ["GUARD_TERMINAL.json", "B2_REWIND_REQUEST.json"]
        fault = read(native / "GUARD_TERMINAL.json")
        request = read(native / "B2_REWIND_REQUEST.json")
        require(
            request["episode"] == fault["episode"] == native.name
            and request["step"] == result["hold_ack_tick"] == step,
            "fault session",
        )
        clock = {
            "callback_fault": fault["fault"],
            "fault_tick": request["step"],
            "HOLD_state_tick": result["hold_ack_tick"],
            "post_fault_native_steps": step - request["step"],
            "references_invalidated": fault["reference_invalidated"],
            "target_publications_after_fault": sum(
                r["kind"] == "installed" and r["monotonic_s"] >= fault["monotonic_s"]
                for r in rows(native / "planner.jsonl")
            ),
            "quota_resume_requests_after_fault": sum(
                g["monotonic_s"] >= fault["monotonic_s"] for g in grants
            ),
        }
    output = {
        "card": result["card"],
        "native_requests": result["native_requests"],
        "completed": result["completed"],
        "cap": spec["max_ticks"],
        "actual_ticks": step,
        "initialization_ticks": 5703,
        "test_seconds": (step - 5703) * 0.002,
        "metrics": result["measured"],
        "first_100ms_endpoint_errors_mm": pos[:2],
        "actual_minus_target_final_xyz_mm": [
            1000 * (a - b) for a, b in zip(actual[-1], target[:3], strict=True)
        ],
        "same_time_EE_state_vs_reconstructed_TCP": {
            "exact_matches": len(sensor_deltas),
            "total_endpoints": len(after),
            "max_position_difference_mm": max(sensor_deltas, default=None),
        },
        "native_windows_checked": len(installed),
        "native_window_point_counts": dict(Counter(len(r["times"]) for r in installed)),
        "max_native_EEF_target_position_difference_mm": wire_error,
        "last_mode_transition": modes[-2:],
        "native_safety_stops": result["native_safety_stops"],
        "native_trace": result["native_trace"],
        "mobile_guard_stop": result["mobile_guard_stop"],
        "closed_lease": True,
        "clean_owned_process_exits": len(result["owned_exits"]),
        "clock_test": clock,
        "result_sha256": sha(folder / "RESULT.private.json"),
        "spec_sha256": lease["spec_sha256"],
        "evidence_sha256": {f: sha(native / f) for f in evidence_files},
    }
    return output, expected, spec


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("group", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    first, pins, s0 = audit(a.group / "T0_STATIC_HOLD")
    second, pins1, s1 = audit(a.group / "T1_CLOCK_FAULT_STOP")
    require(
        pins == pins1
        and s0["b2_code"] == s1["b2_code"]
        and s0["mobile_world_lock"] == s1["mobile_world_lock"],
        "shared frozen identity",
    )
    require(
        read(a.group / "T0_STATIC_HOLD/RESULT.private.json")["episode"]
        != read(a.group / "T1_CLOCK_FAULT_STOP/RESULT.private.json")["episode"],
        "session reuse",
    )
    with Path(s0["mobile_world_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    record(
        a.output,
        {
            "scope": "MuJoCo B2 standing contact-free technical cards",
            "authorized_requests": 2,
            "actual_requests": 2,
            "completed_cards": 2,
            "authorized_ticks": 12906,
            "actual_ticks": first["actual_ticks"] + second["actual_ticks"],
            "fresh_owner_lock_available_at_audit": True,
            "frozen_sha256": pins,
            "experiment_code_sha256": s0["b2_code"],
            "training_updates": 0,
            "model_forwards": 0,
            "manipulation_episodes": 0,
            "cards": [first, second],
        },
    )


if __name__ == "__main__":
    main()
