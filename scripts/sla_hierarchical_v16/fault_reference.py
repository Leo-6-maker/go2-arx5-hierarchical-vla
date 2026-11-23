"""Identify the EEF target actually installed and echoed before a moving STOP."""

import hashlib
import json

import numpy as np
from demonstrated_grip import require
from stop_card_readiness import displacement


def records(path):
    with path.open("rb") as f:
        for line in f:
            yield json.loads(line)


def poses_match(values, expected, *, position_atol, rotation_atol):
    from go2arx5_vla.trajectory.so3 import quat_to_matrix

    return (
        values.shape == (41, 7)
        and np.isfinite(values).all()
        and np.allclose(values[:, :3], expected[:, :3, 3], atol=position_atol, rtol=0)
        and np.all(np.linalg.norm(values[:, 3:], axis=1) > 0)
        and np.allclose(
            quat_to_matrix(values[:, 3:]),
            expected[:, :3, :3],
            atol=rotation_atol,
            rtol=0,
        )
    )


def bound_path(planner, atom, episode, tick, grants, cutoff, goal, base):
    """Match raw installation, receiver output and last completed native atom before the partial grant."""
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    require(
        len(grants) == 2
        and [r["requested_step"] for r in grants] == [tick + 25, tick + 50],
        "missing or duplicate adjacent grants",
    )
    first, last = [r["monotonic_s"] for r in grants]
    require(first < last < cutoff, "fault grant clock")
    require(
        atom["native_start_step"] == tick
        and atom["native_end_step"] == tick + 25
        and atom["ticks"] == 25
        and atom["safety_valid"]
        and atom["solver_exact_installed_targets"],
        "missing pre-fault completed atom",
    )
    wire = atom["installed_reference_sha256"]
    timeline = []
    for row in planner:
        if row["kind"] in ("accept", "installed", "lifecycle"):
            require(np.isfinite(row["monotonic_s"]), "missing planner clock")
            if row["monotonic_s"] <= cutoff:
                timeline.append(row)
    accepts = [r for r in timeline if r["kind"] == "accept" and r.get("ref_id") == wire]
    require(len(accepts) == 1, "missing or repeated original wire identity")
    receipt = accepts[0]
    accepted = receipt["monotonic_s"]
    require(
        accepted <= first
        and hashlib.sha256(receipt["raw"].encode()).hexdigest() == wire,
        "invalid pre-grant wire identity",
    )
    doc = json.loads(receipt["raw"])
    meta = doc["_pi0294"]
    require(
        meta["operation"] == "install"
        and meta["episode"] == episode
        and abs(meta["source_mpc_s"] - 0.002 * tick) < 1e-8
        and meta["ttl_s"] == 0.4,
        "wrong registered reference epoch or clock",
    )
    for field, expected in (("ee", goal), ("base", base)):
        pts = doc[field]["waypoints"]
        require(
            len(pts) == 41
            and np.allclose(
                [p["time"] for p in pts],
                0.05 * np.arange(41),
                atol=1e-8,
                rtol=0,
            )
            and poses_match(
                np.array([p["position"] + p["quaternion_xyzw"] for p in pts]),
                expected,
                position_atol=2e-7,
                rotation_atol=7e-7,
            ),
            "wire differs from the known path",
        )
    installed = target_id = None
    for row in sorted(timeline, key=lambda r: r["monotonic_s"]):
        wall = row["monotonic_s"]
        if row is receipt or wall < accepted:
            continue
        if row["kind"] != "installed" or row.get("ref_id") != wire:
            raise ValueError("reference revoked or superseded before boundary")
        require(
            installed is not None or wall <= first,
            "installation after first grant",
        )
        times = np.asarray(row["times"], float)
        poses = np.asarray(row["poses"], float)
        require(
            times.shape == (41,)
            and np.allclose(
                times, tick * 0.002 + 0.05 * np.arange(41), atol=1e-8, rtol=0
            )
            and poses.shape == (41, 14)
            and poses_match(poses[:, :7], goal, position_atol=1e-7, rotation_atol=2e-7)
            and poses_match(poses[:, 7:], base, position_atol=1e-7, rotation_atol=2e-7),
            "installed EEF/BASE or receiver clock differs",
        )
        inputs = np.asarray(row["inputs"])
        require(
            inputs.shape == (41, 12) and np.isfinite(inputs).all(),
            "invalid installed source inputs",
        )
        target = trajectory_identity(row["times"], row["poses"], row["inputs"])
        require(
            target_id is None or target_id == target,
            "target changed during fault interval",
        )
        if installed is None:
            installed = row
            target_id = target
    require(
        installed is not None
        and set(atom["solver_exact_installed_targets"]) == {target_id},
        "missing matching installed target",
    )
    return installed, target_id, last


def solver_echo(ros, installed, target_id, before_grant):
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    matches = []
    for r in ros:
        wall = r["received_monotonic_s"]
        if (
            not installed["monotonic_s"] <= wall < before_grant
            or r["topic_key"] != "policy"
        ):
            continue
        m = r["message"]
        t = m["plan_target_trajectories"]
        identity = trajectory_identity(
            t["time_trajectory"],
            [x["value"] for x in t["state_trajectory"]],
            [x["value"] for x in t["input_trajectory"]],
        )
        if identity == target_id:
            obs_s = m["init_observation"]["time"]
            require(
                installed["times"][0] - 1e-8
                <= obs_s
                <= installed["times"][0] + 0.05 + 1e-8,
                "solver observation clock outside completed atom",
            )
            matches.append(
                {
                    "received_monotonic_s": wall,
                    "solver_observation_s": m["init_observation"]["time"],
                }
            )
    require(matches, "no independent solver echo before final grant")
    return matches


def motion_from_install(out, start, cutoff, anchor, mount):
    from native_known import window

    goal = window(anchor, (start - 5703) * 0.002)
    base = np.broadcast_to(mount, (41, 4, 4))
    selected = [
        r
        for r in records(out / "MOBILE_SAMPLES.private.jsonl")
        if r["native_start_step"] == start
    ]
    require(len(selected) == 1, "missing or duplicate native atom")
    grants = [
        r
        for r in records(out / "CLOCK_RESUME_REQUESTS.jsonl")
        if r["requested_step"] in (start + 25, start + 50)
    ]
    ins, identity, before = bound_path(
        records(out / "planner.jsonl"),
        selected[0],
        out.name,
        start,
        grants,
        cutoff,
        goal,
        base,
    )
    # Qualify a target only on a real solver message received before the outstanding grant.
    echoes = solver_echo(
        records(out / "ROS_CONSUMPTION.private.jsonl"), ins, identity, before
    )
    ticks = np.arange(start + 11, start + 37)
    positions = np.asarray(ins["poses"])[:, :3]
    xyz = np.column_stack(
        [np.interp(ticks * 0.002, ins["times"], axis) for axis in positions.T]
    )
    return {
        "source_tick": start,
        "first_measured_tick": int(ticks[0]),
        "last_measured_tick": int(ticks[-1]),
        "wire_sha256": selected[0]["installed_reference_sha256"],
        "installed_target_sha256": identity,
        "solver_echoes": len(echoes),
        "last_echo": max(echoes, key=lambda x: x["received_monotonic_s"]),
        "final_grant_monotonic_s": before,
        "installed_motion": displacement(xyz, ticks, int(ticks[-1]), 25),
        "controller_selection": "unobserved",
    }


def floor_feet(names, packet):
    feet = set()
    require(
        abs(packet["contact_time_s"] - (packet["step"] - 1) * 0.002) < 1e-8,
        "foot contact clock",
    )
    for c in packet["contacts"]:
        ids = c["geom0"], c["geom1"]
        require(
            all(type(i) is int and 0 <= i < len(names) for i in ids),
            "invalid floor contact geometry",
        )
        pair = {names[i] for i in ids}
        if "floor" in pair:
            feet.update(pair & {"FL", "FR", "RL", "RR"})
    return tuple(sorted(feet))
