"""Standing no-contact A0 goals and a journal readout on the native clock."""

import hashlib
import struct

import numpy as np
from d3_known_path import poses_at
from feedback import OPEN, START, require

from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0294.task_state import parse_state


def target(old, t):
    return poses_at(old, "translation", "slow", np.maximum(0, np.asarray(t) - 1.0))


def window(old, elapsed):
    if elapsed < 1:
        return np.repeat(old[None], 41, axis=0)
    return target(old, elapsed + np.arange(41) * 0.05)


def packet(stream):
    (n,) = struct.unpack("<I", stream.read(4))
    require(40 <= n <= 2 * 1024**2, "JOURNAL_LENGTH")
    return parse_state(stream.read(n))


def reset_hash(out):
    with (out / "state.bin.journal").open("rb") as f:
        p = packet(f)
    require(p["step"] == 1, "FIRST_STEP")
    return hashlib.sha256(
        np.r_[p["qpos"], p["qvel"]].astype("<f8").tobytes()
    ).hexdigest()


def read_line(path):
    import json

    with path.open("rb") as f:
        end = f.seek(0, 2)
        pos = end
        while pos:
            pos = max(0, pos - 4096)
            f.seek(pos)
            buf = f.read(end - pos).rstrip(b"\n")
            if b"\n" in buf or not pos:
                return json.loads(buf.split(b"\n")[-1])
    raise ValueError("EMPTY_LOG")


def install(adapter, poses, base, seq, tick):
    import json

    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    action = pose10(poses[:40], np.full(40, OPEN))
    mobile.install(adapter, action, np.repeat(base[None], 41, axis=0))
    ref = adapter.active_reference
    wire = ref["wire_sha256"]
    events = ref["receiver_events"]
    accepted = next(r for r in events if r["kind"] == "accept" and r["ref_id"] == wire)
    raw = accepted["raw"]
    meta = json.loads(raw)["_pi0294"]
    require(
        hashlib.sha256(raw.encode()).hexdigest() == wire
        and meta["episode"] == adapter.output.name
        and meta["sequence"] == seq
        and abs(meta["source_mpc_s"] - tick * 0.002) < 1e-8
        and meta["ttl_s"] == 0.4,
        "WIRE_OR_CLOCK",
    )
    installed = [e for e in events if e["kind"] == "installed" and e["ref_id"] == wire]
    require(bool(installed), "MISSING_INSTALL_ACK")
    for ev in installed:
        v = np.asarray(ev["poses"])
        require(
            v.shape == (41, 14)
            and np.allclose(
                ev["times"], tick * 0.002 + np.arange(41) * 0.05, rtol=0, atol=1e-8
            )
            and np.allclose(v[:, :3], poses[:, :3, 3], rtol=0, atol=1e-7)
            and np.allclose(
                quat_to_matrix(v[:, 3:7]), poses[:, :3, :3], rtol=0, atol=2e-7
            )
            and np.allclose(v[:, 7:10], base[:3, 3], rtol=0, atol=1e-7)
            and np.allclose(quat_to_matrix(v[:, 10:]), base[:3, :3], rtol=0, atol=2e-7),
            "INSTALLED_FIELDS",
        )
    return action, wire


def atom_receipt(out, last, sample, source, wire):
    require(
        last["installed_reference_sha256"] == wire
        and last["solver_exact_installed_targets"]
        and last["safety_valid"]
        and sample["contact_valid"]
        and sample["safety_valid"]
        and not (sample["robot_object_contact"] or sample["native_grasp"])
        and last["installed_EEF10"][9] == OPEN
        and last["target_endpoint_EEF10"][9] == OPEN
        and abs(
            last["gripper_command"]["source_native_s"]
            - last["native_start_step"] * 0.002
        )
        < 1e-8,
        "CONTACT_OR_UNACKNOWLEDGED_COMMAND",
    )
    return {
        "episode": out.name,
        "source": source,
        "start": last["native_start_step"],
        "end": last["native_end_step"],
        "accepted": wire,
        "installed": wire,
        "consumed": last["installed_reference_sha256"],
        "solver_echo": True,
        "finger_owner": out.name,
        "finger_fraction": last["installed_EEF10"][9],
        "start_pose": matrix10(np.asarray(last["installed_EEF10"])),
        "end_pose": matrix10(np.asarray(last["target_endpoint_EEF10"])),
    }


def tcp(model, data, tool, p):
    import mujoco

    data.qpos[:] = p["qpos"]
    mujoco.mj_kinematics(model, data)
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[tool]
    T[:3, :3] = data.site_xmat[tool].reshape(3, 3)
    return T


def readout(out, old, model, end, save):
    import mujoco

    tool = model.site("x5_ee").id
    data = mujoco.MjData(model)
    actual = []
    with (out / "state.bin.journal").open("rb") as f:
        for tick in range(1, end + 1):
            p = packet(f)
            require(
                p["step"] == tick and abs(p["time_s"] - 0.002 * tick) < 1e-8,
                "JOURNAL_CLOCK",
            )
            if tick >= START:
                actual.append(tcp(model, data, tool, p))
        require(not f.read(1), "POST_ENDPOINT_PHYSICS")
    actual = np.stack(actual)
    t = np.arange(len(actual)) * 0.002
    goal = target(old, t)
    # The actual receiver interpolates the 50 ms position knots linearly.
    lattice = target(old, np.arange(0, 24.051, 0.05))
    for j in range(3):
        goal[:, j, 3] = np.interp(t, np.arange(len(lattice)) * 0.05, lattice[:, j, 3])
    x = actual[:, :3, 3] * 1000
    g = goal[:, :3, 3] * 1000
    R = actual[:, :3, :3]
    G = goal[:, :3, :3]
    delta = (x - x[0]) - (g - g[0])
    error = x - g
    abs_angle = np.degrees(geodesic_angle(R, G))
    d_angle = np.degrees(geodesic_angle(R @ R[0].T, G @ G[0].T))

    def norm_rms(v):
        return float(np.sqrt(np.mean(np.sum(v * v, axis=1))))

    np.savez_compressed(
        save,
        elapsed_ms=t * 1000,
        actual=actual,
        goal=goal,
        abs_mm=np.linalg.norm(error, axis=1),
        abs_deg=abs_angle,
        delta_mm=np.linalg.norm(delta, axis=1),
        delta_deg=d_angle,
    )
    result = {
        "samples": len(t) - 1,
        "abs_rms_mm": norm_rms(error[1:]),
        "delta_rms_mm": norm_rms(delta[1:]),
        "abs_rms_deg": float(np.sqrt(np.mean(abs_angle[1:] ** 2))),
        "delta_rms_deg": float(np.sqrt(np.mean(d_angle[1:] ** 2))),
        "end_error_mm_deg": [float(np.linalg.norm(error[-1])), float(abs_angle[-1])],
    }
    if len(t) >= 1501:
        result.update(
            z_100ms_mm=float(x[50, 2] - x[0, 2]),
            hold_rms_mm=norm_rms(error[1:501]),
            motion_rms_mm=norm_rms(error[501:1501]),
            motion_xyz_mm=(x[1500] - x[500]).tolist(),
        )
    return result
