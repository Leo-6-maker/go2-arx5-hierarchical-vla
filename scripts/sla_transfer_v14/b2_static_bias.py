"""B2-D0: reconstruct closed simulator records using forward kinematics only."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from b2_owner import configure
from identity import read, record, rows, sha

START = 5703
DT = 0.002
JOINTS = [f"x5_joint{i}" for i in range(1, 7)]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest):
    check(sha(path) == digest, f"mismatched evidence {path.name}")


def summary(values):
    import numpy as np

    if not len(values):
        return None
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def vec(a):
    return [round(float(x), 7) for x in a]


def timings(times, duration):
    import numpy as np

    delta = np.diff(times)
    return {
        "events": len(times),
        "events_per_sim_s": len(times) / duration,
        "adjacent_source_interval_ms": summary(delta * 1000),
        "duplicate_stamps": int(np.sum(abs(delta) < 1e-9)),
        "rewinds": int(np.sum(delta < -1e-9)),
    }


def stamp(message):
    value = message["header"]["stamp"]
    return value["sec"] + 1e-9 * value["nanosec"]


def pose(model, data, q, bid, site):
    import mujoco

    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return (
        data.xpos[bid].copy(),
        data.xmat[bid].reshape(3, 3).copy(),
        data.site_xpos[site].copy(),
        data.site_xmat[site].reshape(3, 3).copy(),
    )


def frames(model, packets, atom, native_trace):
    import mujoco
    import numpy as np

    from go2arx5_vla.trajectory.so3 import geodesic_angle

    def roll_pitch_yaw(r):
        return np.degrees(
            [
                np.arctan2(r[2, 1], r[2, 2]),
                np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])),
                np.arctan2(r[1, 0], r[0, 0]),
            ]
        )

    data = mujoco.MjData(model)
    base = model.body("base_link").id
    tool = model.site("x5_ee").id
    joint = [model.joint(name).id for name in JOINTS]
    aj = np.array(model.jnt_qposadr[joint])
    av = np.array(model.jnt_dofadr[joint])
    aa = [model.actuator(name).id for name in JOINTS]
    check(
        model.jnt_bodyid[0] == base
        and np.array_equal(model.actuator_trnid[aa, 0], joint),
        "base and joint-to-actuator mapping",
    )
    target = np.array(atom[0]["target_endpoint_EEF10"][:3])
    start = packets[START - 1]
    p0, r0, x0, s0 = pose(model, data, start["qpos"], base, tool)
    check(np.allclose(x0, target, atol=1e-12, rtol=0), "measured target anchor")
    rel0 = r0.T @ (x0 - p0)
    tick_wall = {r["token"]: r["monotonic_ns"] for r in native_trace if r["kind"] == 12}
    pd = sorted(
        (r for r in native_trace if r["kind"] == 9), key=lambda r: r["monotonic_ns"]
    )
    pdns = [r["monotonic_ns"] for r in pd]
    step_before = {
        r["token"]: r["monotonic_ns"] for r in native_trace if r["kind"] == 11
    }
    endpoints = {r["native_end_step"]: r for r in atom}
    out = []
    fkerr = []
    step_error = []
    for state in packets[START - 1 :]:
        tick = state["step"]
        q = state["qpos"]
        p, r, x, s = pose(model, data, q, base, tool)
        bx = p + r @ rel0
        qbase = q.copy()
        qbase[aj] = start["qpos"][aj]
        *_, cx, _ = pose(model, data, qbase, base, tool)
        check(np.max(abs(cx - bx)) < 1e-10, "base-only FK closure")
        index = bisect_right(pdns, step_before[tick]) - 1
        check(
            index >= 0 and index == bisect_right(pdns, tick_wall[tick]) - 1,
            "PD changed during integration",
        )
        pdrow = pd[index]
        check(
            pdrow["n"] == 90 and pdrow["monotonic_ns"] <= tick_wall[tick],
            "prior PD target",
        )
        motor = np.array(pdrow["values"]).reshape(5, 18)
        qcmd = motor[0, 12:18]
        qc = q.copy()
        qc[aj] = qcmd
        *_, xd, _ = pose(model, data, qc, base, tool)
        relcmd = r.T @ (xd - p)
        if tick == START:
            relcmd0 = relcmd.copy()
        arm = 1000 * (x - bx)
        cmd = 1000 * r @ (relcmd - relcmd0)
        actual = 1000 * (x - target)
        body = 1000 * (bx - target)
        relangles = roll_pitch_yaw(r) - roll_pitch_yaw(r0)
        out.append(
            {
                "t": tick,
                "ms": 2 * (tick - START),
                "abs": actual,
                "base": body,
                "arm": arm,
                "origin": 1000 * (p - p0),
                "roll_pitch_yaw_deg": relangles,
                "cmd": cmd,
                "tracking_change": arm - cmd,
                "pd_fk_minus_target": 1000 * (xd - target),
                "actual_minus_pd_fk": 1000 * (x - xd),
                "delta": 1000 * (x - x0),
                "attitude_deg": np.degrees(geodesic_angle(s, s0)),
                "q": q[aj],
                "dq": state["qvel"][av],
                "qcmd": qcmd,
                "ctrl": state["ctrl"][aa],
                "ff": motor[4, 12:],
                "ctrl_fraction": abs(state["ctrl"][aa])
                / np.where(
                    state["ctrl"][aa] >= 0,
                    model.actuator_ctrlrange[aa, 1],
                    -model.actuator_ctrlrange[aa, 0],
                ),
            }
        )
        step_error.append(np.max(abs(actual - body - arm)))
        if tick in endpoints:
            site_pose = np.array(endpoints[tick]["actual_TCP"])
            fkerr.append(np.max(abs(x - site_pose[:3, 3])))
            check(
                np.max(abs(s - site_pose[:3, :3])) < 1e-10, "archived TCP orientation"
            )
    check(
        len(fkerr) == len(endpoints) and max(fkerr) < 1e-10 and max(step_error) < 1e-7,
        "all endpoint checks",
    )
    return out, pd, tick_wall, max(fkerr) * 1000


def snapshot(r, first):
    from numpy.linalg import norm

    result = {
        "t_ms": r["ms"],
        **{
            key: vec(r[key])
            for key in (
                "abs",
                "base",
                "arm",
                "origin",
                "roll_pitch_yaw_deg",
                "cmd",
                "tracking_change",
                "pd_fk_minus_target",
                "actual_minus_pd_fk",
            )
        },
        "total_mm": float(norm(r["abs"])),
        "absolute_minus_incremental_mm": float(norm(r["abs"] - r["delta"])),
        "orientation_error_deg": float(r["attitude_deg"]),
        "actual_joint_delta_deg": vec((r["q"] - first["q"]) * 180 / 3.141592653589793),
        "pd_joint_target_delta_deg": vec(
            (r["qcmd"] - first["qcmd"]) * 180 / 3.141592653589793
        ),
        "pd_minus_actual_deg": vec((r["qcmd"] - r["q"]) * 180 / 3.141592653589793),
    }
    return result


def counter(window):
    import numpy as np

    return {
        "max_abs_joint_velocity_rad_s": vec(
            np.max([abs(r["dq"]) for r in window], axis=0)
        ),
        "max_abs_motor_input": vec(np.max([abs(r["ctrl"]) for r in window], axis=0)),
        "max_fraction_of_mjb_input_limit": vec(
            np.max([r["ctrl_fraction"] for r in window], axis=0)
        ),
        "at_or_beyond_input_limit_ticks": int(
            sum(np.any(r["ctrl_fraction"] >= 1.0 - 1e-8) for r in window)
        ),
        "max_abs_pd_feedforward": vec(np.max([abs(r["ff"]) for r in window], axis=0)),
    }


def ros_analysis(native, atom, pd, tick_wall, trace):
    import numpy as np
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    first_ref = atom[0]["installed_reference_sha256"]
    plan = list(rows(native / "planner.jsonl"))
    windows = {
        trajectory_identity(r["times"], r["poses"], r["inputs"]): r
        for r in plan
        if r["kind"] == "installed" and r["ref_id"] == first_ref
    }
    begin = min(r["monotonic_s"] for r in windows.values())
    end = atom[-1]["native_end_step"]
    t0 = START * DT
    t1 = end * DT
    stamps = defaultdict(list)
    req = []
    forwarded = []
    modes = []
    echo = []
    policy_fields = set()
    for event in rows(native / "ROS_CONSUMPTION.private.jsonl"):
        k = event["topic_key"]
        m = event["message"]
        wall = event["received_monotonic_s"]
        if k == "mode":
            modes.append((wall, m["data"]))
        if k in ("arm_request", "arm_target"):
            st = stamp(m)
            check(m["joint_names"] == JOINTS, "arm order")
            if t0 - 0.2 <= st <= t1 + 1e-8:
                point = m["points"][0]
                item = (st, wall, tuple(point["positions"]), tuple(point["velocities"]))
                (req if k == "arm_request" else forwarded).append(item)
        elif k == "policy":
            policy_fields.update(m)
            st = m["init_observation"]["time"]
            if begin <= wall and st <= t0 + 0.11 and st >= t0 - 0.10:
                tr = m["plan_target_trajectories"]
                key = trajectory_identity(
                    tr["time_trajectory"],
                    [s["value"] for s in tr["state_trajectory"]],
                    [s["value"] for s in tr["input_trajectory"]],
                )
                if key in windows and windows[key]["monotonic_s"] <= wall:
                    echo.append((wall, st))
        elif k in ("observation", "ee"):
            st = m["time"]
        else:
            continue
        if t0 + 1e-8 < st <= t1 + 1e-8:
            stamps[k].append(st)
    check(echo, "first installed reference never echoed")
    physics_keys = sorted((ns * 1e-9, i) for i, ns in tick_wall.items())
    walltimes = [s for s, _ in physics_keys]

    def sim_offset(wall):
        return 2 * (physics_keys[bisect_right(walltimes, wall) - 1][1] - START)

    modes_at_start = [x for wall, x in modes if wall <= begin]
    running_modes = [
        r["values"][3:7]
        for r in trace
        if r["kind"] == 16 and t0 + 1e-8 < r["sim_time_s"] <= t1 + 1e-8
    ]
    check(
        running_modes and all(x == [0, 1, 0, 0] for x in running_modes),
        "mode changed during test",
    )
    stamps["PD_callback"] = [
        r["sim_time_s"] for r in pd if t0 + 1e-8 < r["sim_time_s"] <= t1 + 1e-8
    ]
    stamps["physics"] = [i * DT for i in range(START + 1, end + 1)]
    # Match the six positions by exact content. Observer delivery order is not an actuator ACK.
    fwd_by_q = defaultdict(list)
    pd_by_q = defaultdict(list)
    for st, wall, q, _ in forwarded:
        fwd_by_q[q].append((st, wall))
    for r in pd:
        if t0 - 0.2 <= r["sim_time_s"] <= t1 + 1e-8:
            pd_by_q[tuple(r["values"][12:18])].append(
                (r["sim_time_s"], r["monotonic_ns"] * 1e-9)
            )
    joint_req = [r for r in req if t0 + 1e-8 < r[0] <= t1 + 1e-8]

    def match(events, requests):
        lag = []
        ambiguous = 0
        multiplicity = Counter(q for _, _, q, _ in req)
        for st, _, q, _ in requests:
            if multiplicity[q] != 1:
                ambiguous += 1
                continue
            candidates = [t for t, _ in events[q] if t >= st - 1e-8]
            if candidates:
                lag.append(1000 * (min(candidates) - st))
        return {
            "matched_unique_requests": len(lag),
            "ambiguous_requests": ambiguous,
            "requests": len(requests),
            "first_matching_source_stamp_minus_request_ms": summary(np.round(lag, 6)),
        }

    pre = max((r for r in req if r[0] <= t0 + 1e-8), key=lambda r: r[0])
    early = [
        {
            "source_ms": round((r[0] - t0) * 1000, 3),
            "max_abs_delta_from_anchor_deg": float(
                np.degrees(np.max(abs(np.array(r[2]) - pre[2])))
            ),
        }
        for r in joint_req
        if r[0] <= t0 + 0.025 + 1e-8
    ]
    fe = min(echo)
    return {
        "by_source_stamp": {k: timings(v, t1 - t0) for k, v in stamps.items()},
        "published_arm_requests_consume_windows": sum(r["arm_requests"] for r in atom),
        "at_first_install_mode": modes_at_start[-1],
        "mode_samples_WBC": len(running_modes),
        "mode_transitions": [
            {"mode": k, "at_completed_tick_relative_ms": sim_offset(wall)}
            for wall, k in modes
            if wall >= begin
        ],
        "initial_install_completed_tick_relative_ms": sim_offset(begin),
        "first_exact_policy_echo": {
            "observer_wall_ms_since_install": (fe[0] - begin) * 1000,
            "latest_completed_tick_relative_ms": sim_offset(fe[0]),
            "policy_initial_observation_relative_ms": round(1000 * (fe[1] - t0), 4),
        },
        "arm_request_changes_first_25ms": early,
        "exact_request_to_forwarded_q": match(fwd_by_q, joint_req),
        "exact_request_to_sim_PD_q": match(pd_by_q, joint_req),
        "first_100ms_request_to_PD_q": match(
            pd_by_q, [r for r in joint_req if r[0] <= t0 + 0.1 + 1e-8]
        ),
        "solver_policy_message_fields": sorted(policy_fields),
    }


def reference_handoff(native, history):
    import numpy as np

    old = next(r for r in history if r["native_end_step"] == START)
    new = next(r for r in history if r["native_start_step"] == START)
    events = [r for r in rows(native / "planner.jsonl") if r["kind"] == "installed"]

    def published(row):
        return [r for r in events if r["ref_id"] == row["installed_reference_sha256"]]

    before = published(old)[-1]
    after = published(new)[0]
    check(before["monotonic_s"] < after["monotonic_s"], "handoff order")
    previous = np.array(before["poses"][-1])
    following = np.array(after["poses"][0])
    check(
        np.allclose(before["poses"], previous, rtol=0, atol=1e-6)
        and np.allclose(after["poses"], following, rtol=0, atol=1e-6),
        "static windows across anchor",
    )
    result = {}
    for name, offset in (("ee", 0), ("base_reference", 7)):
        a = previous[offset : offset + 7]
        b = following[offset : offset + 7]
        delta = 1000 * (b[:3] - a[:3])
        cos = abs(
            np.dot(a[3:], b[3:]) / (np.linalg.norm(a[3:]) * np.linalg.norm(b[3:]))
        )
        result[name] = {
            "new_minus_old_world_mm": vec(delta),
            "displacement_mm": float(np.linalg.norm(delta)),
            "orientation_step_deg": float(
                np.degrees(2 * np.arccos(np.clip(cos, 0, 1)))
            ),
        }
    return result


def effort_evidence(model, trace, tick_wall, end):
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    qadr = [int(model.joint(name).qposadr[0]) for name in JOINTS]
    vadr = [int(model.joint(name).dofadr[0]) for name in JOINTS]
    act = [model.actuator(name).id for name in JOINTS]
    check(np.all(model.actuator_gear[act, 0] == 1), "arm gear scaling")
    latest = None
    measurements = []
    residual = []
    for r in trace:
        if r["kind"] == 9:
            latest = r
        if r["kind"] != 10 or not (
            START * DT - 0.02 <= r["sim_time_s"] <= end * DT + 1e-8
        ):
            continue
        check(
            latest
            and latest["sim_time_s"] == r["sim_time_s"]
            and latest["ticket"] < r["ticket"],
            "paired PD callbacks",
        )
        goal, dgoal, kp, kd, ff = np.array(latest["values"]).reshape(5, 18)[:, 12:]
        state = np.array(r["values"])
        q = state[:34]
        dq = state[34:66]
        ctrl = state[66:][act]
        proportional = kp * (goal - q[qadr])
        derivative = kd * (dgoal - dq[vadr])
        residual.append(abs(ctrl - proportional - derivative - ff))
        measurements.append(
            (r["monotonic_ns"], q, ctrl, proportional, derivative, ff, dgoal, dq[vadr])
        )
    ns = [m[0] for m in measurements]
    sample = []
    for step in (START, START + 50, end):
        _, q, ctrl, p, d, ff, dgoal, dq = measurements[
            bisect_right(ns, tick_wall[step]) - 1
        ]
        # Unconstrained rigid-body gravity term at the archived posture, zero velocity.
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        sample.append(
            {
                "t_ms": 2 * (step - START),
                "motor_input_Nm": vec(ctrl),
                "P_Nm": vec(p),
                "D_Nm": vec(d),
                "feedforward_Nm": vec(ff),
                "rigid_body_gravity_bias_Nm": vec(data.qfrc_bias[vadr]),
                "desired_velocity_rad_s": vec(dgoal),
                "actual_velocity_rad_s": vec(dq),
            }
        )
    return {
        "sampled": sample,
        "max_PD_equation_residual_Nm": vec(np.max(residual, axis=0)),
    }


def audit(card, receipt):
    import mujoco
    import numpy as np
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from go2arx5_vla.world_model.native_replay import forbid_integration, packets

    spec_path = card / "EXECUTION_SPEC.json"
    pinned(spec_path, receipt["spec_sha256"])
    pinned(card / "RESULT.private.json", receipt["result_sha256"])
    spec = read(spec_path)
    result = read(card / "RESULT.private.json")
    native = card / result["episode"]
    check(
        read(card / "control/LEASE.json")["status"] == "CLOSED",
        "closed records required",
    )
    for name in (
        "init.trace",
        "state.bin.journal",
        "planner.jsonl",
        "ROS_CONSUMPTION.private.jsonl",
        "MOBILE_SAMPLES.private.jsonl",
        "LOADED_EXECUTOR_00.private.json",
    ):
        pinned(native / name, receipt["evidence_sha256"][name])
    scene = Path(spec["model_path"])
    pinned(scene, spec["frozen_files"][str(scene)])
    states = packets(native / "state.bin.journal")
    check(
        [r["step"] for r in states]
        == list(range(1, result["journal_integrations"] + 1)),
        "missing tick",
    )
    history = list(rows(native / "MOBILE_SAMPLES.private.jsonl"))
    atom = [r for r in history if r["native_start_step"] >= START]
    trace = read_trace(native / "init.trace")
    with forbid_integration(mujoco) as calls:
        model = mujoco.MjModel.from_binary_path(str(scene))
        trajectory, pd, tick_wall, max_fk = frames(model, states, atom, trace)
        effort = effort_evidence(model, trace, tick_wall, states[-1]["step"])
    check(
        not calls,
        "integration attempted",
    )
    rms = float(
        np.sqrt(np.mean([np.dot(r["abs"], r["abs"]) for r in trajectory[25::25]]))
    )
    check(
        abs(rms - receipt["metrics"]["position_rms_mm"]) < 1e-8,
        "prior 50ms endpoint RMS",
    )
    safety = [
        r
        for r in rows(native / "MOBILE_SAFETY_TRACE.private.jsonl")
        if r["kind"] == "mobile_actual" and START <= r["tick"] <= states[-1]["step"]
    ]
    ctl = ros_analysis(native, atom, pd, tick_wall, trace)
    solver_log = (native / "solver.log").read_text(errors="replace")
    for key, expected in (("mpcDesiredFrequency", 100), ("mrtDesiredFrequency", 400)):
        check(
            any(
                key in line and line.rstrip().endswith(str(expected))
                for line in solver_log.splitlines()
            ),
            "solver frequency",
        )
    cmd = next(
        c["argv"]
        for c in read(native / "PROCESS_COMMANDS.private.json")
        if c["name"] == "controller"
    )
    check(
        not any("loopFrequency" in arg or "--params-file" in arg for arg in cmd),
        "controller frequency override",
    )
    small = [
        snapshot(trajectory[i], trajectory[0])
        for i in sorted({0, 1, 2, 5, 10, 25, 50, 125, 250, len(trajectory) - 1})
    ]
    curve = [
        {
            "ms": r["ms"],
            "world_mm": vec(r["abs"]),
            "base_only_mm": vec(r["base"]),
            "arm_relative_mm": vec(r["arm"]),
            "pd_command_motion_mm": vec(r["cmd"]),
        }
        for r in trajectory
        if r["ms"] <= 100 or r["ms"] % 50 == 0
    ]
    return {
        "card": card.name,
        "source_ticks": len(states),
        "test_ticks": len(trajectory) - 1,
        "endpoint_rms_mm": rms,
        "first_100ms_max_base_only_displacement_mm": max(
            float(np.linalg.norm(r["base"])) for r in trajectory[:51]
        ),
        "first_down_10mm_sim_ms": {
            key: next((r["ms"] for r in trajectory if r[key][2] <= -10), None)
            for key in ("cmd", "abs")
        },
        "fk_endpoint_max_difference_mm": max_fk,
        "milestones": small,
        "curve": curve,
        "actuator_first100": counter(trajectory[1:51]),
        "actuator_full": counter(trajectory[1:]),
        "timing_and_identity": ctl,
        "handoff_at_start": reference_handoff(native, history),
        "joint_effort": effort,
        "foot_contacts_registered_samples": dict(
            Counter(r["foot_contacts"] for r in safety)
        ),
        "max_unsupported_s": max(r["unsupported_s"] for r in safety),
        "additional_inputs_sha256": {
            f: sha(native / f)
            for f in (
                "MOBILE_SAFETY_TRACE.private.jsonl",
                "solver.log",
                "PROCESS_COMMANDS.private.json",
            )
        },
        "scene_sha256": sha(scene),
        "max_abs_minus_incremental_mm": max(
            float(np.linalg.norm(r["abs"] - r["delta"])) for r in trajectory
        ),
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("group", type=Path)
    cli.add_argument("receipt", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    pinned(
        args.receipt, "ed2d54421c6d2d2005f6c1b2f542fcd2bb7970f6a240f4dd196db6558e034ea9"
    )
    prior = read(args.receipt)
    spec_path = args.group / prior["cards"][0]["card"] / "EXECUTION_SPEC.json"
    pinned(spec_path, prior["cards"][0]["spec_sha256"])
    spec = read(spec_path)
    configure(spec)
    custody = read(Path(spec["controller_binary"]).parent / "MANIFEST.private.json")
    check(
        custody["binary"]["sha256"] == prior["frozen_sha256"]["controller"],
        "frozen controller provenance",
    )
    rec = Path(custody["recovery_spec"]["path"])
    pinned(rec, custody["recovery_spec"]["sha256"])
    recipe = read(rec)
    cpp = Path(recipe["controller_source"])
    pinned(cpp, recipe["input_pins"]["controller_source"])
    source = cpp.read_text()
    check(
        all(
            s in source
            for s in (
                '"loopFrequency", 200.0',
                "create_wall_timer",
                "if (source_tick == pi0294_last_tick_) return;",
                "mrt_->updatePolicy();",
                "core_->Evaluate(observation, command)",
            )
        ),
        "unexpected recovered controller loop",
    )
    reports = [audit(args.group / c["card"], c) for c in prior["cards"]]
    record(
        args.output,
        {
            "inputs_b2_public_sha256": sha(args.receipt),
            "controller_cpp_sha256": sha(cpp),
            "script_sha256": sha(Path(__file__)),
            "recovery_spec_sha256": sha(rec),
            "new_native_requests": 0,
            "new_integrations": 0,
            "model_forwards": 0,
            "cards": reports,
        },
    )


if __name__ == "__main__":
    main()
