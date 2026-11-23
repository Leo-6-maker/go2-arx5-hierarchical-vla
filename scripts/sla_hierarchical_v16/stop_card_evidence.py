"""Score a completed V3 fault from the receiver, independent native trace, plant and owner."""

import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from stop_card_readiness import displacement


def require(value, reason):
    if not value:
        raise ValueError(reason)


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def terminal_evidence(trace, audit, last_grant, installs, exits, absent):
    stops = [r for r in trace if r["kind"] == 13]
    require(len(stops) == 1, "expected one whole-system trace STOP")
    stop = stops[0]
    require(
        stop["token"] == 6439
        and abs(stop["sim_time_s"] - 12.878) < 1e-8
        and stop["values"] == [1.0, 157.0],
        "wrong STOP identity",
    )
    steps = [row["token"] for row in trace if row["kind"] == 12]
    require(
        steps == list(range(1, 6440))
        and not any(
            row["kind"] in (11, 12) and row["ticket"] > stop["ticket"] for row in trace
        ),
        "trace step sequence differs from the closed state journal",
    )
    writer = audit.get("writer") or {}
    require(
        audit["status"] == "RECORDED_TRACE_CHECKS_PASS"
        and audit["native_step_records"] == 6439
        and audit["integrations_after_first_STOP"] == 0
        and writer.get("stop_latched") is True
        and writer.get("dropped") == 0
        and writer.get("written") == len(trace)
        and writer.get("events") == len(trace),
        "incomplete native trace or post-STOP integration",
    )
    boundary = stop["monotonic_ns"] * 1e-9
    require(
        last_grant["requested_step"] == 6453 and last_grant["monotonic_s"] < boundary,
        "no outstanding moving grant",
    )
    require(
        installs and all(r["monotonic_s"] < boundary for r in installs),
        "installation at or after STOP",
    )
    roles = {"planner", "controller", "solver", "sim"}
    require(
        len(exits) == len(roles)
        and {r["name"] for r in exits} == roles
        and all(type(r["exit_code"]) is int for r in exits)
        and absent,
        "process terminal evidence incomplete",
    )
    return dict(
        stop_tick=stop["token"],
        stop_monotonic_ns=stop["monotonic_ns"],
        stop_simulation_s=stop["sim_time_s"],
        installs=len(installs),
        last_grant_end=last_grant["requested_step"],
        after_stop_ticks=audit["integrations_after_first_STOP"],
        owned_exits=exits,
    )


def score(root, sp):
    import mujoco
    from go2arx5_vla.world_model.init_trace_reader import audit_trace, read_trace
    from native_known import tcp, target
    from fault_reference import floor_feet, motion_from_install
    from receiver_fault_replay import sha, read
    from verify_grip_card import states, support

    spec = read(sp)
    case = sp.parent
    lease = read(case / "control/LEASE.json")
    require(
        lease["status"] == "CLOSED"
        and lease["spec_sha256"] == sha(sp)
        and lease["owner"] == spec["owner"]
        and case.name == "V3-STOP-01",
        "closed matching lease required",
    )
    run = read(case / "RESULT.private.json")
    require(
        run.get("eligible")
        and run["last_requested_tick"] == 6453
        and not run.get("close_error"),
        "card did not reach the registered fault grant",
    )
    out = case / run["episode"]
    channel = case / "student"
    grant = read(channel / "authority.json")
    receiver = Path(spec["outer_library"])
    require(
        grant["owner"] == lease["owner"]
        and grant["episode"] == out.name
        and grant["receiver_sha256"] == sha(receiver),
        "receiver ownership mismatch",
    )
    reset = read(out / "BEFORE_FIRST_INTEGRATION.json")
    require(
        reset["sim_time_s"] == 0
        and not reset["running"]
        and reset["arm_mode"] == "HOLD",
        "unverified cold reset",
    )
    require(
        not any(
            (out / n).exists()
            for n in ("MOBILE_SAFETY_STOP.private.json", "GUARD_TERMINAL.json")
        ),
        "competing safety stop",
    )
    loaded = read(out / "LOADED_EXECUTOR_00.private.json")
    for role, k in (
        ("sim", "native_binary"),
        ("solver", "solver_binary"),
        ("controller", "controller_binary"),
    ):
        require(
            loaded[role]["executable_sha256"] == spec["frozen_files"][spec[k]],
            "unexpected loaded executable",
        )
    for k in (
        "outer_library",
        "trace_library",
        "observer_library",
        "mobile_safety_library",
    ):
        require(
            run["loaded_at_command_reset"][k] == spec["frozen_files"][spec[k]],
            "unverified library",
        )
    trace = read_trace(out / "init.trace")
    audit = audit_trace(out / "init.trace", max_native_ticks=6503)
    sent = list(lines(out / "CLOCK_RESUME_REQUESTS.jsonl"))
    require(
        sent and all(r["requested_step"] < 6453 for r in sent[:-1]),
        "later or duplicate grant",
    )
    installs = [r for r in lines(out / "planner.jsonl") if r["kind"] == "installed"]
    terminal = terminal_evidence(
        trace,
        audit,
        sent[-1],
        installs,
        run["owned_exits"],
        run["owned_processes_absent"],
    )
    ack = []
    observed = []
    stop = []
    with (channel / "receiver.csv").open() as stream:
        for row in csv.reader(stream):
            require(len(row) == 12 and row[1] == grant["epoch"], "receiver identity")
            if row[0] == "RECEIVER_ACK":
                ack.append(row)
            elif row[0] == "NATIVE_STEP":
                observed.append(row)
            elif "STOP" in row[0]:
                stop.append(row)
    require(
        len(stop) == 1
        and stop[0][0] == "RECEIVER_REJECT_STOP"
        and int(stop[0][2]) == 6439
        and int(stop[0][4]) == 157,
        "receiver STOP differs",
    )
    commands = list(lines(channel / "issuer.jsonl"))
    require(len(ack) == len(commands) == 158, "command/ACK count")
    for n, (a, c) in enumerate(zip(ack, commands, strict=True)):
        payload = struct.pack("<8sQddd", b"GRTGRIP1", int(a[4]), *map(float, a[5:8]))
        tick = 2503 + 25 * n
        expires = 6438.5 * 0.002 if n == 157 else tick * 0.002 + 0.399
        require(
            int(a[4]) == c["sequence"] == n
            and int(a[2]) == c["tick"] == tick
            and abs(float(a[5]) - tick * 0.002) < 1e-8
            and abs(float(a[7]) - expires) < 1e-8
            and abs(0.044 * float(a[6]) - 0.042) < 1e-12
            and hashlib.sha256(payload).hexdigest() == c["packet_sha256"],
            "unexpected actual open command",
        )
    model = mujoco.MjModel.from_binary_path(spec["model_path"])
    require(
        sha(Path(spec["model_path"])) == spec["frozen_files"][spec["model_path"]],
        "scene identity",
    )
    data = mujoco.MjData(model)
    site = model.site("x5_ee").id
    address = [
        int(model.jnt_qposadr[model.joint(k).id])
        for k in ("task_left_slide", "task_right_slide")
    ]
    names = [model.geom(i).name for i in range(model.ngeom)]
    t = []
    xyz = []
    feet_at_fault = []
    require(len(observed) == 6439, "incomplete or excess receiver steps")
    for s, row in zip(states(out), observed, strict=True):
        tick = s["step"]
        require(
            int(row[2]) == tick
            and abs(s["time_s"] - tick * 0.002) < 1e-8
            and all(
                support(names[c["geom0"]], names[c["geom1"]]) for c in s["contacts"]
            ),
            "state clock or contact",
        )
        require(
            np.allclose(
                np.asarray(s["qpos"])[address],
                np.asarray(row[10:12], float),
                rtol=0,
                atol=1e-12,
            ),
            "bilateral physical evidence",
        )
        if tick > 2503:
            require(
                int(row[4]) == (tick - 2504) // 25
                and np.allclose(np.asarray(row[8:10], float), 0.042, atol=1e-9, rtol=0),
                "student-applied open command differs",
            )
        if tick >= 6414:
            feet_at_fault.append(floor_feet(names, s))
            t.append(tick)
            xyz.append(tcp(model, data, site, s)[:3, 3])
    ticks = np.array(t)
    achieved = displacement(np.array(xyz), ticks, 6439, 25)
    initial = read(case / "INITIAL.private.json")["features"]
    anchor = np.array(initial["old_ee"])
    intended = target(anchor, (ticks - 5703) * 0.002)
    programmed = displacement(intended[:, :3, 3], ticks, 6439, 25)
    bound = motion_from_install(
        out,
        6403,
        terminal["stop_monotonic_ns"] * 1e-9,
        anchor,
        np.asarray(initial["old_base"]),
    )
    goal = bound["installed_motion"]
    require(
        achieved["distance_mm"] > 0.10 and goal["distance_mm"] > 0,
        "stationary or unconfirmed fault",
    )
    return dict(
        card=case.name,
        attempts=1,
        qualified=True,
        terminal=terminal,
        achieved=achieved,
        desired=goal,
        programmed=programmed,
        installed_reference_evidence=bound,
        distinct_floor_feet_at_motion_samples=feet_at_fault,
        acknowledged_commands=len(ack),
        actual_integrations=len(observed),
        native_sla_attempts=0,
        raw_state_sha256=sha(out / "state.bin.journal"),
        native_trace_sha256=sha(out / "init.trace"),
    )
