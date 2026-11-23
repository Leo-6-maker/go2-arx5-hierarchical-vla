"""Pure reference selection and historical boundary verification for the D1 experiment."""

import hashlib
import json

import numpy as np
from go2arx5_vla.world_model.fusion_trajectory_adapter import (
    MECHANICAL_TRAVEL_M,
    WORKING_OPEN_M,
)
from go2arx5_vla.world_model.target_reference import trajectory_identity

from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix
from scripts.pi0291_arx5_data.native_dataset import pose10

START = 5703
CELLS = ("OO", "MO", "OM", "MM")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def mpose(p, q):
    mat = np.eye(4)
    mat[:3, :3] = quat_to_matrix(np.asarray(q, float))
    mat[:3, 3] = p
    return mat


def pose7(mat):
    return np.r_[mat[:3, 3], matrix_to_quat(mat[:3, :3])]


def held_part(doc, key):
    part = doc[key]
    pts = part["waypoints"]
    check(
        len(pts) == 41 and part["frame"] == "world", "world 41-knot reference required"
    )
    check(
        all(
            abs(pt["time"] - i * 0.05) < 1e-8
            and pt["position"] == pts[0]["position"]
            and pt["quaternion_xyzw"] == pts[0]["quaternion_xyzw"]
            for i, pt in enumerate(pts)
        ),
        "previous reference is not constant",
    )
    return mpose(pts[0]["position"], pts[0]["quaternion_xyzw"])


def boundary(output, last_sample):
    from identity import rows

    key = last_sample["installed_reference_sha256"]
    events = list(rows(output / "planner.jsonl"))
    accepts = [e for e in events if e["kind"] == "accept" and e["ref_id"] == key]
    installed = [e for e in events if e["kind"] == "installed" and e["ref_id"] == key]
    check(
        len(accepts) == 1 and installed, "missing accepted or installed old reference"
    )
    raw = accepts[0]["raw"]
    doc = json.loads(raw)
    meta = doc["_pi0294"]
    check(
        hashlib.sha256(raw.encode()).hexdigest() == key
        and meta["episode"] == output.name
        and meta["source_mpc_s"]
        <= START * 0.002
        < meta["source_mpc_s"] + meta["ttl_s"],
        "old packet identity or TTL",
    )
    ee, base = held_part(doc, "ee"), held_part(doc, "base")
    expected = np.r_[pose7(ee), pose7(base)]
    for e in installed:
        check(
            np.allclose(e["poses"], expected, rtol=0, atol=2e-7),
            "wrong installed spatial fields",
        )
    keys = {trajectory_identity(e["times"], e["poses"], e["inputs"]) for e in installed}
    check(
        bool(keys.intersection(last_sample["solver_exact_installed_targets"])),
        "missing solver echo before boundary",
    )
    check(
        last_sample["native_end_step"] == START
        and last_sample["safety_valid"]
        and not last_sample["native_grasp"],
        "invalid anchor",
    )
    return (
        {
            "old_ee": ee,
            "old_base": base,
            "measured_ee": np.array(last_sample["actual_TCP"]),
            "measured_base": np.array(last_sample["actual_mount"]),
        },
        key,
        meta["sequence"],
    )


def fixed(factor, parts):
    check(factor in CELLS, "unknown factorial cell")
    ee = parts[("old_" if factor[0] == "O" else "measured_") + "ee"]
    base = parts[("old_" if factor[1] == "O" else "measured_") + "base"]
    row = pose10(ee, WORKING_OPEN_M / MECHANICAL_TRAVEL_M)
    return np.repeat(row[None], 40, axis=0), np.repeat(base[None], 41, axis=0)


def initial_features(packet, parts, pd, gripper):
    q = packet["qpos"].copy()
    for ix in (3, 30):
        if q[ix] < 0:
            q[ix : ix + 4] *= -1
    dq = packet["qvel"]
    return {
        "qpos_linear_m": q[[0, 1, 2, 25, 26, 27, 28, 29]].tolist(),
        "qpos_hinge_rad": q[7:25].tolist(),
        "free_quaternions": q[[3, 4, 5, 6, 30, 31, 32, 33]].tolist(),
        "qvel_linear_m_s": dq[[0, 1, 2, 24, 25, 26, 27, 28]].tolist(),
        "qvel_angular_rad_s": dq[[*range(3, 24), 29, 30, 31]].tolist(),
        "arm_dq_rad_s": dq[18:24].tolist(),
        "arm_pd_gap_rad": (np.asarray(pd) - q[19:25]).tolist(),
        "gripper_command": [float(gripper)],
        **{k: v.tolist() for k, v in parts.items()},
    }


def prior_pd(trace):
    from go2arx5_vla.world_model.init_trace_reader import read_trace

    records = read_trace(trace)
    before = next(
        r["monotonic_ns"] for r in records if r["kind"] == 11 and r["token"] == START
    )
    latest = max(
        (r for r in records if r["kind"] == 9 and r["monotonic_ns"] <= before),
        key=lambda r: r["monotonic_ns"],
    )
    check(latest["n"] == 90, "incomplete PD target")
    return latest["values"][12:18]


def distances(a, b):
    out = {}
    for key in a:
        x, y = np.array(a[key]), np.array(b[key])
        if x.shape == (4, 4):
            out[key + "_m"] = float(np.linalg.norm(x[:3, 3] - y[:3, 3]))
            out[key + "_rad"] = float(geodesic_angle(x[:3, :3], y[:3, :3]))
        else:
            out[key] = float(np.max(np.abs(x - y)))
    return out


def matching_rule(anchors):
    diff = distances(*anchors)
    # Per-feature maximum difference against EACH sealed anchor and earlier D1 cases.
    # Small predefined engineering equivalence bands prevent identical archived
    # readings from imposing bitwise equality on asynchronous future resets.
    floors = {
        "qpos_linear_m": 0.001,
        "qpos_hinge_rad": 0.005,
        "free_quaternions": 0.002,
        "qvel_linear_m_s": 0.01,
        "qvel_angular_rad_s": 0.05,
        "arm_dq_rad_s": 0.03,
        "arm_pd_gap_rad": 0.005,
        "gripper_command": 1e-10,
    }
    return {
        k: max(3 * v, floors.get(k, 0.004 if k.endswith("_rad") else 0.001))
        for k, v in diff.items()
    }


def verify_initial(features, anchors, limits):
    delta = {k: max(distances(features, h)[k] for h in anchors) for k in limits}
    failures = {
        k: {"observed": v, "limit": limits[k]}
        for k, v in delta.items()
        if v > limits[k]
    }
    return {"observed_max": delta, "violations": failures, "matched": not failures}
