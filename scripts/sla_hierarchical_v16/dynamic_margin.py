"""Replay observed TCP channels and bound pad clearance for the registered V3 paths."""

import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sealed_panel import configure_site
from known_dynamic_curve import trajectory


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sat_lower_bound(centers, axes, half, obstacle, obstacle_axes, obstacle_half):
    """Largest separating projection of two boxes in metres, clipped at zero."""
    b = np.broadcast_to(obstacle_axes, axes.shape)
    # columns of each matrix are box axes; the candidate directions are rows below.
    u, v = axes.swapaxes(1, 2), b.swapaxes(1, 2)
    cross = np.cross(u[:, :, None, :], v[:, None, :, :]).reshape(-1, 9, 3)
    directions = np.concatenate((u, v, cross), axis=1)
    length = np.linalg.norm(directions, axis=2, keepdims=True)
    normals = np.divide(
        directions, length, out=np.zeros_like(directions), where=length > 1e-10
    )
    separation = abs(np.einsum("nki,ni->nk", normals, centers - obstacle))
    ra = abs(normals @ axes) @ half
    rb = abs(normals @ b) @ obstacle_half
    return np.maximum(0, (separation - ra - rb).max(axis=1))


def quat_error_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if abs(np.linalg.norm(a) - 1) > 1e-5 or abs(np.linalg.norm(b) - 1) > 1e-5:
        raise ValueError("invalid channel quaternion")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(
        np.rad2deg(
            4
            * np.arctan2(
                min(np.linalg.norm(a - b), np.linalg.norm(a + b)),
                max(np.linalg.norm(a - b), np.linalg.norm(a + b)),
            )
        )
    )


def metrics(observed):
    if not observed:
        return None
    a = np.asarray(observed)
    return dict(
        matches=len(a),
        unique_ticks=len(np.unique(a[:, 0])),
        position_max_mm=float(a[:, 1].max()),
        position_p95_mm=float(np.percentile(a[:, 1], 95)),
        angle_max_deg=float(a[:, 2].max()),
        angle_p95_deg=float(np.percentile(a[:, 2], 95)),
        tick_at_max_position=int(a[a[:, 1].argmax(), 0]),
        tick_at_max_angle=int(a[a[:, 2].argmax(), 0]),
    )


def channel(case, record, model):
    import mujoco
    from scipy.spatial.transform import Rotation
    from scripts.pi0294.task_state import parse_state
    from audit_sealed import read, pin

    result = case / "RESULT.private.json"
    pin(result, record["result_sha256"])
    folder = case / read(result)["episode"]
    state = folder / "state.bin.journal"
    ros = folder / "ROS_CONSUMPTION.private.jsonl"
    pin(state, record["journal_sha256"])
    by_tick = defaultdict(list)
    off_grid = 0
    outside = 0
    with ros.open("rb") as f:
        for line in f:
            # The native archive is JSONL; ignore other event payloads without deserializing the large MPC policies.
            if b'"ee"' not in line:
                continue
            row = json.loads(line)
            if row["topic_key"] != "ee":
                continue
            msg = row["message"]
            t = float(msg["time"])
            tick = round(t / 0.002)
            if t < 5203 * 0.002 - 1e-7 or t > record["ticks"] * 0.002 + 1e-7:
                outside += 1
                continue
            if abs(t - 0.002 * tick) > 1e-7:
                off_grid += 1
                continue
            pose = np.asarray(msg["state"]["value"])
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("invalid ee message")
            by_tick[tick].append(pose)
    data = mujoco.MjData(model)
    tool = model.site("x5_ee").id
    phases = defaultdict(list)
    snap = None
    fingers = []
    with state.open("rb") as f:
        for i in range(1, record["ticks"] + 1):
            (n,) = struct.unpack("<I", f.read(4))
            if not 40 <= n <= 2**20:
                raise ValueError("invalid journal length")
            if i < 5203:
                f.seek(n, 1)
                continue
            pkt = parse_state(f.read(n))
            if pkt["step"] != i or abs(pkt["time_s"] - 0.002 * i) > 1e-7:
                raise ValueError("wrong journal tick")
            fingers.append(pkt["qpos"][25:27])
            if i == 5703:
                snap = pkt["qpos"].copy()
            if i not in by_tick:
                continue
            data.qpos[:] = pkt["qpos"]
            mujoco.mj_kinematics(model, data)
            q = Rotation.from_matrix(data.site_xmat[tool].reshape(3, 3)).as_quat()
            p = data.site_xpos[tool]
            phase = (
                "startup"
                if i <= 5703
                else "moving"
                if 6203 < i <= 7153
                else "parked"
                if i > 7153
                else "initial_hold"
            )
            for x in by_tick[i]:
                phases[phase].append(
                    [
                        i,
                        float(1000 * np.linalg.norm(p - x[:3])),
                        quat_error_deg(q, x[3:]),
                    ]
                )
        if f.read(1):
            raise ValueError("unaccounted native suffix")
    if snap is None or not phases["startup"]:
        raise ValueError("incomplete startup evidence")
    expected = {
        "startup": 501,
        "initial_hold": min(500, max(0, record["ticks"] - 5703)),
        "moving": min(950, max(0, record["ticks"] - 6203)),
        "parked": max(0, record["ticks"] - 7153),
    }
    reduced = {}
    for k, n in expected.items():
        m = metrics(phases[k])
        if n and not m:
            raise ValueError("no actual channel observations in " + k)
        if m:
            m.update(native_ticks=n, unobserved_ticks=n - m["unique_ticks"])
        reduced[k] = m
    return dict(
        card=case.name,
        phases=reduced,
        ros_sha256=sha(ros),
        state_sha256=record["journal_sha256"],
        off_grid=off_grid,
        outside_interval=outside,
        slide_min_mm=(1000 * np.min(fingers, axis=0)).tolist(),
        slide_max_mm=(1000 * np.max(fingers, axis=0)).tolist(),
    ), snap


def clearances(model, qpos, origin, rate, slide_bound):
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    site = model.site("x5_ee").id
    tcp = np.eye(4)
    tcp[:3, :3] = data.site_xmat[site].reshape(3, 3)
    tcp[:3, 3] = data.site_xpos[site]
    length = np.linalg.norm(tcp[:3, 3] - origin[:3, 3])
    angle = Rotation.from_matrix(tcp[:3, :3].T @ origin[:3, :3]).magnitude()
    n = max(1, math.ceil(max(length / 0.0005, angle / np.deg2rad(0.1))))
    u = np.linspace(0, 1, n + 1)
    join = np.repeat(tcp[None], n + 1, axis=0)
    join[:, :3, 3] += u[:, None] * (origin[:3, 3] - tcp[:3, 3])
    join[:, :3, :3] = Slerp(
        [0, 1], Rotation.from_matrix([tcp[:3, :3], origin[:3, :3]])
    )(u).as_matrix()
    curve = trajectory(origin, np.arange(3501) * 0.002, *rate)
    result = []
    for j, pad in enumerate(("task_left_pad", "task_right_pad")):
        g = model.geom(pad).id
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(pad)
        local_p = tcp[:3, :3].T @ (data.geom_xpos[g] - tcp[:3, 3])
        local_r = tcp[:3, :3].T @ data.geom_xmat[g].reshape(3, 3)
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * model.geom_size[g]
        )
        lever = float(np.linalg.norm(local_p + corners @ local_r.T, axis=1).max())
        for name in ("task_table", "task_cube_geom"):
            o = model.geom(name).id
            if model.geom_type[o] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError(name)
            for segment, poses, allowance in (
                ("join", join, (length + lever * angle) / (2 * n)),
                ("curve", curve, 0.001 * (rate[0] + lever * rate[1])),
            ):
                pos = poses[:, :3, :3] @ local_p + poses[:, :3, 3]
                axes = poses[:, :3, :3] @ local_r
                bound = sat_lower_bound(
                    pos,
                    axes,
                    model.geom_size[g],
                    data.geom_xpos[o],
                    data.geom_xmat[o].reshape(3, 3),
                    model.geom_size[o],
                )
                idx = int(bound.argmin())
                minimum = float(1000 * bound[idx])
                result.append(
                    dict(
                        pad=pad,
                        obstacle=name,
                        segment=segment,
                        samples=len(poses),
                        minimum_index=idx,
                        sample_lower_bound_mm=minimum,
                        between_sample_reserve_mm=1000 * allowance,
                        observed_slide_excursion_reserve_mm=float(slide_bound[j]),
                        lever_mm=lever * 1000,
                        net_lower_bound_mm=minimum - 1000 * allowance - slide_bound[j],
                    )
                )
    return result


def run(root, repo):
    import mujoco
    from audit_sealed import read, pin

    base = repo / "reports/SLA_HIERARCHICAL_V16"
    v0 = read(base / "V0_EVIDENCE_MATRIX_ACTUAL.json")
    strata = read(base / "V3_STARTUP_PREFLIGHT_ACTUAL.json")["components"]
    row_by_card = {}
    snapshots = {}
    records = []
    model = None
    scene_hash = None
    for stage in ("F1", "F2"):
        f = root / f"feedback/{stage}_VERIFIED.json"
        pin(f, v0["source_hashes"][stage])
        folder = next((root / "feedback").glob("sla-feedback-" + stage.lower() + "-*"))
        for r in read(f)["records"]:
            case = folder / r["card"]
            specfile = case / "EXECUTION_SPEC.json"
            pin(specfile, read(case / "control/LEASE.json")["spec_sha256"])
            s = read(specfile)
            scene = Path(s["model_path"])
            h = s["frozen_files"][str(scene)]
            if model is None:
                pin(scene, h)
                model = mujoco.MjModel.from_binary_path(str(scene))
                scene_hash = h
            if h != scene_hash:
                raise ValueError("scene changed")
            v, snap = channel(case, r, model)
            records.append(v)
            row_by_card[v["card"]] = v
            if r["card"] in ("F1-01", "F1-02"):
                p = case / "INITIAL.private.json"
                pin(p, r["initial_sha256"])
                snapshots[r["card"]] = (snap, np.asarray(read(p)["features"]["old_ee"]))
    if len(records) != 18 or any(r["off_grid"] for r in records):
        raise ValueError("population or off-grid channel")
    bounds = []
    channels = []
    for group in strata:
        keys = group["cards"]
        witness = min(keys)
        subset = [row_by_card[k] for k in keys]
        measured = [m for row in subset for m in row["phases"].values() if m]
        channels.append(
            dict(
                witness=witness,
                cards=keys,
                position_mm=max(v["position_max_mm"] for v in measured),
                angle_deg=max(v["angle_max_deg"] for v in measured),
                matched_messages=sum(v["matches"] for v in measured),
                covered_ticks=sum(v["unique_ticks"] for v in measured),
            )
        )
        q, origin = snapshots[witness]
        lo = np.min([v["slide_min_mm"] for v in subset], axis=0)
        hi = np.max([v["slide_max_mm"] for v in subset], axis=0)
        excursion = np.maximum(abs(lo - 1000 * q[25:27]), abs(hi - 1000 * q[25:27]))
        for mode, rate in (
            ("SLOW", (0.004, np.deg2rad(0.6))),
            ("SOURCE_SPEED", (0.01125, 0.045)),
        ):
            bounds.append(
                dict(
                    witness=witness,
                    mode=mode,
                    checks=clearances(model, q, origin, rate, excursion),
                )
            )
    clearance = min(v["net_lower_bound_mm"] for b in bounds for v in b["checks"])
    lever = max(57, max(v["lever_mm"] for b in bounds for v in b["checks"]))
    xyz = max(1.8, max(r["position_mm"] for r in channels))
    angle = max(r["angle_deg"] for r in channels)
    rot = 2 * lever * np.sin(np.deg2rad(2 + angle) / 2)
    eps = math.floor(10 * min(4, min(8, clearance) - xyz - rot)) / 10
    return dict(
        records=records,
        strata=channels,
        geometry=bounds,
        frozen_scene_sha256=scene_hash,
        clearance_lower_bound_mm=clearance,
        translation_channel_reserve_mm=xyz,
        observed_angular_disagreement_deg=angle,
        lever_mm=lever,
        rotation_sweep_reserve_mm=rot,
        candidate_position_tolerance_mm=eps,
        conditional_geometric_margin_pass=eps > 0,
        sensor_noise_variance=None,
        new_native_requests=0,
        new_physics_ticks=0,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    print(json.dumps(run(root, repo), indent=2, allow_nan=False))
