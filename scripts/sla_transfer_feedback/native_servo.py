"""Independently qualified no-contact A1 controller paired to a scored A0 trajectory."""

import json
from pathlib import Path

import numpy as np
from d1_readiness import pinned, read, sha
from feedback import OPEN, START, ReferenceManager, StopMonitor, require, rotvec
from native_known import atom_receipt, install, packet, read_line, tcp


def measure(out, tick, sample):
    require(abs(sample["time_s"] - tick * 0.002) < 1e-8, "MEASUREMENT_TIME")
    return {
        "episode": out.name,
        "tick": tick,
        "TCP": sample["tcp"],
        "safe": sample["safety_valid"] and sample["contact_valid"],
        "contact": sample["robot_object_contact"],
    }


def preflight(root, cfg):
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate

    from scripts.pi0291_arx5_data.native_dataset import pose10

    c = cfg["controller"]
    g = cfg["geometry"]
    outcomes = {}
    site = (
        Path(__file__).resolve().parents[2] / "reports/SLA_FROZEN_FEEDBACK/F0_SITE.json"
    )
    ev = read(site)
    require(
        len(ev["records"]) == 12
        and all(len(r["checks"]["negative"]) >= 12 for r in ev["records"]),
        "offline fault tests",
    )
    for name, expected in (
        ("static_bias", "NO_PROGRESS"),
        ("oscillating", "OSCILLATION"),
        ("hold", "HOLD_COMPLETE"),
        ("lost", "HOLD_LOST"),
    ):
        monitor = StopMonitor()
        for t in range(0, 4002, 2):
            p = (
                30
                if name == "static_bias"
                else (
                    10 + 3 * np.sin(t * np.pi / 100)
                    if name == "oscillating"
                    else (12 if name == "lost" and t >= 230 else 1)
                )
            )
            status = monitor.observe(t, np.array([p, 0.0, 0.0]), np.zeros(3), 4, 2)
            if status not in ("HOLD", "CONTINUE"):
                break
        require(status == expected, "negative_" + name)
        outcomes[name] = [status, t]
    origin = np.eye(4)
    origin[:3, 3] = [0.26, 0, 0.72]
    actual = origin.copy()
    actual[0, 3] -= 0.03
    manager = ReferenceManager("cpu", origin, c)
    anchor = origin
    previous = origin
    wire = "initial"

    def rec(t, source):
        return {
            "episode": "cpu",
            "start": t - 25,
            "end": t,
            "source": source,
            "accepted": wire,
            "installed": wire,
            "consumed": wire,
            "solver_echo": True,
            "end_pose": anchor,
            "start_pose": previous,
            "finger_fraction": OPEN,
            "finger_owner": "cpu",
        }

    def state(t):
        return {
            "episode": "cpu",
            "tick": t,
            "TCP": actual,
            "safe": True,
            "contact": False,
        }

    receipt = rec(START, START - 50)
    for i in range(240):
        t = START + 50 * i
        try:
            win = manager.propose("A1", t, wire, receipt, state(t), origin)
        except ValueError as exc:
            require(
                str(exc) == "WORKSPACE_CAP" and i == 150, "cumulative correction limit"
            )
            outcomes["saturation"] = [str(exc), i]
            break
        full_eef_gate(pose10(win[:40], np.full(40, OPEN)), actual)
        previous = win[1]
        anchor = win[2]
        wire = str(i)
        receipt = rec(t + 50, t)
        manager.consumed(t + 50, wire, win, receipt, state(t + 50))
    require(
        "saturation" in outcomes and g["abs_position_bound_mm"] == 4,
        "unbounded reference",
    )
    return outcomes


def drive(adapter, folder, parts, sample, seq, receipt, old, config):
    import mujoco
    from b2_native import paused
    from d1_geometry import sweep
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    pair = read(folder / "INITIAL.private.json")["pair"]
    root = folder.parent.parent
    f1 = root / "sla-feedback-f1-a0-20261123-01" / pair
    result = read(f1 / "RESULT.private.json")
    curve = f1 / "TRACK.private.npz"
    pinned(curve, result["curve_sha256"])
    with np.load(curve, allow_pickle=False) as saved:
        desired = saved["goal"]
    require(
        desired.shape == (12001, 4, 4)
        and np.allclose(old, desired[0], rtol=0, atol=1e-9),
        "paired desired path",
    )
    out = adapter.output
    model = adapter.actual_state.model
    data = mujoco.MjData(model)
    tool = model.site("x5_ee").id
    guard = binding(adapter.spec["native_root"])
    manager = ReferenceManager(out.name, parts["old_ee"], config["controller"])
    monitor = StopMonitor()
    eps = config["geometry"]["abs_position_bound_mm"]
    angle = config["geometry"]["rotation_bound_deg"]

    def observe(ix, T):
        goal = desired[ix]
        status = monitor.observe(
            ix * 2,
            (goal[:3, 3] - T[:3, 3]) * 1000,
            np.degrees(rotvec(goal[:3, :3] @ T[:3, :3].T)),
            eps,
            angle,
        )
        if status == "HOLD" and ix < 1450:
            monitor.hold = None
            return "CONTINUE"
        return status

    tick = START
    atoms = 0
    installs = 0
    wire = receipt["consumed"]
    status = observe(0, sample["tcp"])
    reason = None
    detection = None
    ledger = folder / "SERVO.private.jsonl"
    with (
        (out / "state.bin.journal").open("rb") as stream,
        ledger.open("x", buffering=1) as log,
    ):
        for step in range(1, START + 1):
            row = packet(stream)
            require(row["step"] == step, "initial journal clock")
        require(np.array_equal(row["qpos"], sample["qpos"]), "initial sample mismatch")
        while tick < START + 12000:
            paused(adapter, tick)
            require(
                not any(
                    (p / "STOP.json").exists()
                    for p in (folder.parent, Path(adapter.spec["campaign_root"]))
                ),
                "operator STOP",
            )
            mode = "A0" if status == "HOLD" else "A1"
            measured = measure(out, tick, sample)
            goal = desired[tick - START]
            try:
                knots = manager.propose(mode, tick, wire, receipt, measured, goal)
            except ValueError as exc:
                if str(exc) != "WORKSPACE_CAP":
                    raise
                reason = str(exc)
                detection = tick
                break
            try:
                with forbid_integration(mujoco) as calls:
                    geo = sweep(
                        model,
                        sample,
                        dict(parts, old_ee=manager.anchor),
                        knots[2],
                        parts["old_base"],
                        guard,
                    )
                require(not calls, "unexpected kinematic integration")
            except ValueError as exc:
                reason = "REFERENCE_GEOMETRY_REJECT"
                detection = tick
                log.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "rejected": reason,
                            "detail": str(exc),
                            "candidate": knots[2].tolist(),
                        }
                    )
                    + "\n"
                )
                break
            prior = wire
            action, wire = install(
                adapter, knots, parts["old_base"], seq + 1 + installs, tick
            )
            installs += 1
            source = tick
            log.write(
                json.dumps(
                    {
                        "tick": tick,
                        "desired": goal.tolist(),
                        "actual": sample["tcp"].tolist(),
                        "previous_wire": prior,
                        "command_anchor": manager.anchor.tolist(),
                        "command_endpoint": knots[2].tolist(),
                        "wire": wire,
                        "mode": mode,
                        "source_tick": tick,
                        "deadline_tick": tick + 200,
                        "checked_poses": geo["poses_checked"],
                        "min_joint_margin_rad": geo["minimum_joint_margin_rad"],
                        "policy_id": None,
                        "policy_age": None,
                    }
                )
                + "\n"
            )
            for _ in range(2):
                sample, _ = mobile.consume(adapter, action, sequence=128 + atoms)
                atoms += 1
                last = read_line(out / "MOBILE_SAMPLES.private.jsonl")
                receipt = atom_receipt(out, last, sample, source, wire)
                require(
                    last["gripper_command"]["sequence"] == 127 + atoms
                    and receipt["start"] == tick
                    and receipt["end"] == tick + 25,
                    "consumed clock",
                )
                for step in range(tick + 1, tick + 26):
                    p = packet(stream)
                    require(
                        p["step"] == step and abs(p["time_s"] - 0.002 * step) < 1e-8,
                        "2ms feedback grid",
                    )
                    T = tcp(model, data, tool, p)
                    if reason is None:
                        status = observe(step - START, T)
                        if status not in ("HOLD", "CONTINUE"):
                            reason = status
                            detection = step
                tick += 25
                require(
                    np.allclose(T, sample["tcp"], rtol=0, atol=1e-10),
                    "feedback TCP differs from journal",
                )
                log.write(
                    json.dumps(
                        {
                            "end": tick,
                            "wire": wire,
                            "delivered_index": atoms,
                            "solver_echoes": len(
                                last["solver_exact_installed_targets"]
                            ),
                            "arm_requests": last["arm_requests"],
                            "actual_endpoint": sample["tcp"].tolist(),
                            "first_terminal_tick": detection,
                        }
                    )
                    + "\n"
                )
                if reason:
                    break
            if reason:
                break
            manager.consumed(tick, wire, knots, receipt, measure(out, tick, sample))
    if reason is None:
        reason = "TIMEOUT"
        detection = tick
    return {
        "outcome": reason,
        "end": tick,
        "atoms": atoms,
        "installs": installs,
        "first_scientific_decision_tick": detection,
        "ticks_between_decision_and_paused_boundary": tick - detection,
        "first_qualified_hold_tick": START + monitor.hold // 2
        if monitor.hold is not None
        else None,
        "servo_ledger_sha256": sha(ledger),
        "paired_F1_curve_sha256": sha(curve),
    }
