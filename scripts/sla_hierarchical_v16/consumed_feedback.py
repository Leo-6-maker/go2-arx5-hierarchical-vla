"""Join a completed native interval to the actual student gripper receiver and measured joints."""

import argparse
from collections import defaultdict
from bisect import bisect_left
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from demonstrated_grip import pin, read, require, rows
from sealed_panel import configure_site


def completed_interval(
    episode,
    epoch,
    atom,
    accepted,
    installed,
    observed_echo_times,
    issuer,
    ack,
    steps,
    measured,
):
    """Return the last consumed width for G7. Solver echoes do not identify the selected WBC policy."""
    wire = atom["installed_reference_sha256"]
    raw = accepted["raw"]
    meta = json.loads(raw)["_pi0294"]
    start, end = atom["native_start_step"], atom["native_end_step"]
    require(
        meta["episode"] == episode
        and meta["operation"] == "install"
        and accepted["kind"] == "accept"
        and installed
        and all(
            r["kind"] == "installed" and r["ref_id"] == wire for r in installed.values()
        )
        and accepted["ref_id"] == wire
        and hashlib.sha256(raw.encode()).hexdigest() == wire,
        "mismatched EEF receiver",
    )
    require(
        end == start + 25 and atom["ticks"] == 25 and len(steps) == 25,
        "incomplete native interval",
    )
    seq = issuer["sequence"]
    require(
        ack[0] == "RECEIVER_ACK"
        and ack[1] == epoch
        and int(ack[2]) == issuer["tick"] == start
        and int(ack[4]) == seq,
        "wrong gripper ACK",
    )
    source, width, expiry = map(float, ack[5:8])
    payload = struct.pack("<8sQddd", b"GRTGRIP1", seq, source, width, expiry)
    require(
        hashlib.sha256(payload).hexdigest() == issuer["packet_sha256"]
        and abs(source - start * 0.002) < 1e-8
        and abs(source - issuer["source_s"]) < 1e-8
        and abs(0.044 * width - issuer["width_m"]) < 1e-12
        and 0 <= width <= 1
        and expiry >= source,
        "gripper packet differs",
    )
    for k, r in enumerate(steps, 1):
        require(
            r[0] == "NATIVE_STEP"
            and r[1] == epoch
            and int(r[2]) == start + k
            and int(r[3]) == 1
            and int(r[4]) == seq
            and tuple(map(float, r[5:8])) == (source, width, expiry),
            "no continuous student consumption",
        )
    encoder = tuple(map(float, steps[-1][10:12]))
    require(
        np.asarray(measured).shape == (2,)
        and np.allclose(measured, encoder, rtol=0, atol=1e-12),
        "independent finger state differs",
    )
    echoes = atom["solver_exact_installed_targets"]
    require(
        len(observed_echo_times) == len(echoes)
        and all(
            h in installed and installed[h]["monotonic_s"] <= wall
            for h, wall in zip(echoes, observed_echo_times, strict=True)
        ),
        "solver echo precedes receiver publication",
    )
    return dict(
        episode=episode,
        epoch=epoch,
        start_tick=start,
        end_tick=end,
        sequence=seq,
        previous_student_command_m=0.044 * width,
        previous_gripper_wire_sha256=issuer["packet_sha256"],
        last_applied_targets_m=list(map(float, steps[-1][8:10])),
        measured_slides_m=list(encoder),
        legacy_eef_width_m=0.044 * atom["installed_EEF10"][9],
        installed_reference_sha256=wire,
        matching_solver_echoes=len(echoes),
        distinct_echo_contents=len(set(echoes)),
        safety_valid=atom["safety_valid"],
        controller_reference_evidence=atom["controller_selected_reference"],
    )


def feedback_since(epoch, query_tick, current_tick, intervals):
    """Input is the verified interval ending at query, then every completed interval through handoff."""
    require(
        intervals
        and intervals[0]["end_tick"] == query_tick
        and intervals[-1]["end_tick"] == current_tick,
        "feedback boundary mismatch",
    )
    first = intervals[0]
    for i, row in enumerate(intervals):
        require(
            row["epoch"] == epoch
            and row["episode"] == first["episode"]
            and row["sequence"] == first["sequence"] + i
            and row["end_tick"] == query_tick + 25 * i
            and row["start_tick"] == row["end_tick"] - 25
            and (not i or not intervals[i - 1]["terminal_at_end"]),
            "feedback gap, reset or STOP",
        )
    final = intervals[-1]
    return dict(
        query_tick=query_tick,
        current_tick=current_tick,
        consumed_after_query=len(intervals) - 1,
        previous_sequence=final["sequence"],
        previous_student_command_m=final["previous_student_command_m"],
        previous_gripper_wire_sha256=final["previous_gripper_wire_sha256"],
        previous_eef_wire_sha256=final["installed_reference_sha256"],
        terminal_at_end=final["terminal_at_end"],
        safety_valid=all(r["safety_valid"] for r in intervals),
        controller_reference_evidence=final["controller_reference_evidence"],
    )


def gate(previous, next_width):
    speed = abs(next_width - previous) / 0.05
    return dict(rate_mm_s=1000 * speed, passes=bool(speed <= 0.04 + 1e-10))


def audit(root, repo):
    import mujoco
    from go2arx5_vla.world_model.target_reference import (
        require_live_window,
        trajectory_identity,
    )
    from verify_grip_card import score, sha

    case = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B"
    verified = score(root, case)
    require(
        verified
        == read(
            repo / "reports/SLA_HIERARCHICAL_V16/V2_STUDENT_GRIPPER_OWNER_ACTUAL.json"
        ),
        "V2 independent audit differs",
    )
    spec = read(case / "EXECUTION_SPEC.json")
    channel = case / "student"
    native = case / read(case / "RESULT.private.json")["episode"]
    auth = read(channel / "authority.json")
    require(
        auth["episode"] == native.name and auth["owner"] == spec["owner"],
        "wrong native epoch",
    )
    source = rows(channel / "issuer.jsonl")
    with (channel / "receiver.csv").open() as f:
        csvrows = list(csv.reader(f))
    acks = [r for r in csvrows if r[0] == "RECEIVER_ACK"]
    ticks = {int(r[2]): r for r in csvrows if r[0] == "NATIVE_STEP"}
    stop = [r for r in csvrows if "STOP" in r[0]]
    require(
        len(stop) == 1
        and stop[0][1] == auth["epoch"]
        and int(stop[0][2]) == 6503
        and max(ticks) == 6503,
        "receiver stop",
    )
    af = native / "MOBILE_SAMPLES.private.jsonl"
    pf = native / "planner.jsonl"
    events = rows(pf)
    accepts = {
        r["ref_id"]: r for r in events if r["kind"] == "accept" and r.get("ref_id")
    }
    # A single accepted wire can publish several rolling target contents before the next install.
    installs = defaultdict(dict)
    for e in events:
        if e["kind"] == "installed" and e.get("ref_id"):
            h = trajectory_identity(e["times"], e["poses"], e["inputs"])
            installs[e["ref_id"]].setdefault(h, e)
    ros = native / "ROS_CONSUMPTION.private.jsonl"
    echo_times = defaultdict(list)
    with ros.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["topic_key"] == "policy":
                t = r["message"]["plan_target_trajectories"]
                h = trajectory_identity(
                    t["time_trajectory"],
                    [x["value"] for x in t["state_trajectory"]],
                    [x["value"] for x in t["input_trajectory"]],
                )
                echo_times[h].append(r["received_monotonic_s"])
    resumes = native / "CLOCK_RESUME_REQUESTS.jsonl"
    grants = {r["requested_step"]: r["monotonic_s"] for r in rows(resumes)}
    last_echo = float("-inf")
    model = mujoco.MjModel.from_binary_path(spec["model_path"])
    fingers = [
        int(model.jnt_qposadr[model.joint(name).id])
        for name in ("task_left_slide", "task_right_slide")
    ]
    records = []
    seams = []
    for atom, issued, ack, previously_verified in zip(
        rows(af), source, acks, verified["verified_command_intervals"], strict=True
    ):
        key = atom["installed_reference_sha256"]
        t = atom["native_start_step"]
        meta = json.loads(accepts[key]["raw"])["_pi0294"]
        require_live_window(
            meta["source_mpc_s"],
            atom["gripper_command"]["source_native_s"],
            atom["ticks"],
        )
        wall = []
        for h in atom["solver_exact_installed_targets"]:
            require(
                h in installs[key], "solver content was not published under this wire"
            )
            start_wall = max(
                installs[key][h]["monotonic_s"],
                grants[t + 25],
                np.nextafter(last_echo, np.inf),
            )
            times = echo_times[h]
            k = bisect_left(times, start_wall)
            require(
                k < len(times) and times[k] < grants.get(t + 50, float("inf")),
                "missing clock-matched solver receipt",
            )
            wall.append(times[k])
            last_echo = times[k]
        r = completed_interval(
            native.name,
            auth["epoch"],
            atom,
            accepts[key],
            installs[key],
            wall,
            issued,
            ack,
            [ticks[i] for i in range(t + 1, t + 26)],
            np.asarray(atom["qpos"])[fingers],
        )
        require(
            r["end_tick"] == previously_verified["end_tick"]
            and r["sequence"] == previously_verified["sequence"]
            and r["previous_gripper_wire_sha256"] == previously_verified["wire_sha256"]
            and abs(r["previous_student_command_m"] - previously_verified["width_m"])
            < 1e-12,
            "V2 checked command differs",
        )
        r["terminal_at_end"] = r["end_tick"] == int(stop[0][2])
        if records:
            prev = records[-1]
            require(
                prev["end_tick"] == r["start_tick"]
                and prev["sequence"] + 1 == r["sequence"],
                "unjoined previous native interval",
            )
            seams.append(
                dict(
                    at_tick=t,
                    next_sequence=r["sequence"],
                    next_student_width_mm=r["previous_student_command_m"] * 1000,
                    actual_previous_mm=prev["previous_student_command_m"] * 1000,
                    legacy_previous_mm=prev["legacy_eef_width_m"] * 1000,
                    actual=gate(
                        prev["previous_student_command_m"],
                        r["previous_student_command_m"],
                    ),
                    legacy=gate(
                        prev["legacy_eef_width_m"], r["previous_student_command_m"]
                    ),
                )
            )
        records.append(r)
    require(len(records) == 160 and len(seams) == 159, "incomplete native denominator")

    def summary(group):
        return dict(
            n=len(group),
            first_sequence=group[0]["sequence"],
            last_sequence=group[-1]["sequence"],
            any_current_echo=sum(bool(r["matching_solver_echoes"]) for r in group),
            multiple_rolling_targets=sum(
                r["distinct_echo_contents"] > 1 for r in group
            ),
            no_echo=sum(not r["matching_solver_echoes"] for r in group),
        )

    observation_file = repo / "reports/SLA_HIERARCHICAL_V16/OWNED_FEEDBACK_ACTUAL.json"
    captures = read(
        pin(
            observation_file,
            "2fa3977bc94c5b2a8139e4f5c1d95d7c57a9deeb95eafbaf5535b4aed49e4c93",
        )
    )
    require(
        captures["epoch"] == auth["epoch"]
        and captures["terminal_tick"] == 6503
        and captures["terminal_query_refused"],
        "different captured feedback session",
    )
    windows = {w["tick"]: w for w in captures["windows"]}
    intervals = []
    for t in [5703, 5903, 6103, 6303]:
        group = [r for r in records if t <= r["start_tick"] < t + 200]
        require(
            len(group) == 8 and group[-1]["end_tick"] == t + 200,
            "incomplete feedback window",
        )
        at_query = records[group[0]["sequence"] - 1]
        record = windows[t]
        require(
            record["previous_sequence"] == at_query["sequence"]
            and record["previous_wire_sha256"]
            == at_query["previous_gripper_wire_sha256"]
            and abs(
                record["previous_width_mm"]
                - at_query["previous_student_command_m"] * 1000
            )
            < 1e-9,
            "wrong command at observation",
        )
        f = root / "v16/feedback-observation02" / record["file"]
        pin(f, record["input_sha256"])
        joined = feedback_since(auth["epoch"], t, t + 200, [at_query, *group])
        seam = None
        if not joined["terminal_at_end"]:
            following = records[group[-1]["sequence"] + 1]
            require(following["start_tick"] == t + 200, "wrong next actual packet")
            nxt = following["previous_student_command_m"]
            seam = dict(
                next_known_sequence=following["sequence"],
                next_known_width_mm=nxt * 1000,
                using_at_query=gate(at_query["previous_student_command_m"], nxt),
                using_at_handoff=gate(joined["previous_student_command_m"], nxt),
            )
        intervals.append(
            dict(
                start_tick=t,
                end_tick=t + 200,
                **summary(group),
                observation_sha256=record["input_sha256"],
                previous_width_at_query_mm=record["previous_width_mm"],
                handoff=joined,
                next_known_seam=seam,
            )
        )
    return dict(
        native_case=case.name,
        receiver_sha256=auth["student_binary"],
        native_sources_sha256={
            p.name: sha(p)
            for p in (
                af,
                pf,
                ros,
                resumes,
                channel / "authority.json",
                channel / "issuer.jsonl",
                channel / "receiver.csv",
                native / "state.bin.journal",
            )
        },
        summary=summary(records),
        controller_reference_evidence=sorted(
            set(r["controller_reference_evidence"] for r in records)
        ),
        query_boundaries=[
            {k: v for k, v in r.items() if k != "epoch"}
            for r in records
            if r["end_tick"] in (5703, 5903, 6103, 6303, 6503)
        ],
        seams=seams,
        feedback_windows=intervals,
        threshold_mm_s=40,
        true_rejections=sum(not x["actual"]["passes"] for x in seams),
        legacy_rejections=sum(not x["legacy"]["passes"] for x in seams),
        legacy_only_rejections=sum(
            x["actual"]["passes"] and not x["legacy"]["passes"] for x in seams
        ),
        legacy_only_accepts=sum(
            not x["actual"]["passes"] and x["legacy"]["passes"] for x in seams
        ),
        new_model_forwards=0,
        new_physics_ticks=0,
        student_policy_grasps=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    result = audit(root, repo)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
