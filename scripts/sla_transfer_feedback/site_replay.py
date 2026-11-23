"""CPU-only archive/receiver and task-geometry checks; no ROS graph or native owner."""

import argparse
import itertools
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_transfer_v15"))
from d1_readiness import baseline, configure, pinned, read, record, rows, sha


def alignment(model, geom_cfg):
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    for name in ("task_left_slide", "task_right_slide"):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.044 - 0.002
    mujoco.mj_forward(model, data)
    tcp = data.site("x5_ee")
    centers = []
    lever = 0
    for n in ("task_left_pad", "task_right_pad"):
        geom = data.geom(n)
        center = (tcp.xmat.reshape(3, 3).T @ (geom.xpos - tcp.xpos)) * 1000
        axes = tcp.xmat.reshape(3, 3).T @ geom.xmat.reshape(3, 3)
        half = 1000 * model.geom(n).size
        assert np.allclose(axes, np.eye(3)) and np.allclose(
            half, geom_cfg["pad_half_size_mm"]
        )
        centers.append(center)
        for signs in itertools.product([-1, 1], repeat=3):
            lever = max(lever, float(np.linalg.norm(center + axes @ (half * signs))))
    cube = 1000 * model.geom("task_cube_geom").size
    assert np.allclose(2 * cube, geom_cfg["cube_side_mm"])
    overlap = geom_cfg["min_linear_overlap_fraction"]
    available = np.array(
        [
            min(
                cube[i] + half[i] - overlap * min(2 * cube[i], 2 * half[i]) - abs(c[i])
                for c in centers
            )
            for i in (0, 2)
        ]
    )
    gap = [centers[0][1] - half[1], -centers[1][1] - half[1]]
    margin = min(*available, *np.subtract(gap, cube[1]))
    usable = (
        margin
        - geom_cfg["observability_reserve_mm"]
        - 2
        * geom_cfg["max_corner_lever_mm"]
        * np.sin(np.deg2rad(geom_cfg["rotation_bound_deg"] / 2))
    )
    assert (
        lever <= geom_cfg["max_corner_lever_mm"]
        and np.floor(usable) == geom_cfg["abs_position_bound_mm"]
    )
    return {
        "pad_centers_in_tcp_mm": [c.tolist() for c in centers],
        "open_inner_gap_mm": float(sum(gap)),
        "centered_cube_lateral_margins_mm": list(np.subtract(gap, cube[1])),
        "min_xz_overlap_margins_mm": available.tolist(),
        "maximum_pad_corner_lever_mm": lever,
        "net_tracking_allowance_mm": float(usable),
    }


def negative(manager, tick, wire, receipt, state, goal, knots):
    import numpy as np
    from feedback import OPEN
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate

    from scripts.pi0291_arx5_data.native_dataset import pose10

    bad = knots.copy()
    bad[1, 0, 3] += 0.028
    wrong_anchor = goal.copy()
    wrong_anchor[0, 3] += 0.028
    cases = [
        (
            "previous_atom_jump",
            "CONSUMED_SEAM",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "start_pose": wrong_anchor}, state, goal
            ),
        ),
        (
            "future_feedback",
            "WRONG_CLOCK",
            lambda: manager.propose(
                "A1", tick, wire, receipt, {**state, "tick": tick + 1}, goal
            ),
        ),
        (
            "stale_feedback",
            "WRONG_CLOCK",
            lambda: manager.propose(
                "A1", tick, wire, receipt, {**state, "tick": tick - 1}, goal
            ),
        ),
        (
            "fake_install",
            "UNVERIFIED_REFERENCE",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "installed": "foreign"}, state, goal
            ),
        ),
        (
            "missing_echo",
            "UNVERIFIED_REFERENCE",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "solver_echo": False}, state, goal
            ),
        ),
        (
            "wrong_anchor",
            "ACK_ANCHOR_MISMATCH",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "end_pose": wrong_anchor}, state, goal
            ),
        ),
        (
            "expired_receipt",
            "WRONG_CLOCK",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "source": tick - 200}, state, goal
            ),
        ),
        (
            "foreign_finger_owner",
            "GRIPPER_OWNERSHIP",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "finger_owner": "student"}, state, goal
            ),
        ),
        (
            "seam_width",
            "GRIPPER_OWNERSHIP",
            lambda: manager.propose(
                "A1", tick, wire, {**receipt, "finger_fraction": 0.5}, state, goal
            ),
        ),
        (
            "contact",
            "CONTACT_OR_STOP",
            lambda: manager.propose(
                "A1", tick, wire, receipt, {**state, "contact": True}, goal
            ),
        ),
        (
            "large_knot_jump",
            "MOBILE_ORIGINAL_FULL40_EEF_GATE",
            lambda: full_eef_gate(pose10(bad[:40], np.full(40, OPEN)), state["TCP"]),
        ),
    ]
    result = {}
    for label, reason, fn in cases:
        try:
            fn()
        except ValueError as exc:
            assert reason == str(exc)
            result[label] = reason
        else:
            raise AssertionError(label)
    assert np.array_equal(
        knots, manager.propose("A1", tick, wire, receipt, state, goal)
    )
    new = dict(
        receipt,
        start=tick + 25,
        end=tick + 50,
        source=tick,
        start_pose=knots[1],
        end_pose=knots[2],
        accepted="fixture",
        installed="fixture",
        consumed="fixture",
    )
    future = dict(state, tick=tick + 50)
    manager.consumed(tick + 50, "fixture", knots, new, future)
    try:
        manager.consumed(tick + 50, "fixture", knots, new, future)
    except ValueError as exc:
        assert str(exc) == "DUPLICATE_OR_SKIPPED_CONSUMPTION"
        result["duplicate_consumption"] = str(exc)
    else:
        raise AssertionError("duplicate consumption")
    return result


def case(root, r, folder, cfg, model, guard):
    import numpy as np
    from clock_replay import GuardedManager, episode
    from d1_factorial_reference_matrix import boundary
    from d1_geometry import sweep
    from feedback import OPEN, START, ReferenceManager
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import document, full_eef_gate
    from go2arx5_vla.world_model.native_replay import packets
    from shadow import check, message, observe

    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0294.task_state import ActualState

    card = root / "v15/sla-d3-known-paths-20261123-01" / r["id"]
    pinned(card / "RESULT.private.json", r["result_sha256"])
    out = card / read(card / "RESULT.private.json")["episode"]
    journal = out / "state.bin.journal"
    pinned(journal, r["journal_sha256"])
    native = packets(journal)[START - 1]
    last = next(
        x
        for x in rows(out / "MOBILE_SAMPLES.private.jsonl")
        if x["native_end_step"] == START
    )
    parts, wire, seq = boundary(out, last)
    initial = card / "INITIAL.private.json"
    pinned(initial, r["initial_sha256"])
    assert wire == read(initial)["old_wire"] and seq == read(initial)["old_sequence"]
    ack = next(
        x
        for x in rows(out / "planner.jsonl")
        if x["kind"] == "accept" and x["ref_id"] == wire
    )
    source_tick = round(ack["source_mpc_s"] / 0.002)
    base = parts["old_base"]
    old = parts["old_ee"]
    actual = parts["measured_ee"]
    measured = ActualState(model).update(native)
    assert (
        measured["contact_valid"]
        and measured["safety_valid"]
        and not measured["robot_object_contact"]
    )
    assert np.array_equal(native["qpos"], last["qpos"]) and np.allclose(
        measured["tcp"], actual, rtol=0, atol=1e-10
    )
    assert native["step"] == START and last["gripper_command"]["sequence"] == 127
    receipt = {
        "episode": out.name,
        "start": last["native_start_step"],
        "end": START,
        "source": source_tick,
        "accepted": wire,
        "installed": wire,
        "consumed": last["installed_reference_sha256"],
        "solver_echo": True,
        "start_pose": matrix10(np.asarray(last["installed_EEF10"])),
        "end_pose": matrix10(np.asarray(last["target_endpoint_EEF10"])),
        "finger_owner": out.name,
        "finger_fraction": last["installed_EEF10"][9],
    }
    state = {
        "episode": out.name,
        "tick": START,
        "TCP": actual,
        "safe": last["safety_valid"],
        "contact": measured["robot_object_contact"],
    }
    metrics = {}
    for mode in ("A0", "A1"):
        manager = ReferenceManager(out.name, old, cfg["controller"])
        knots = manager.propose(mode, START, wire, receipt, state, old)
        action = pose10(knots[:40], np.full(40, OPEN))
        full = full_eef_gate(action, actual)
        assert np.all(action[:, 9] == last["installed_EEF10"][9])
        report = sweep(model, native, parts, knots[2], base, guard)
        d = folder / (r["id"] + "_" + mode)
        d.mkdir()
        with episode(d.name, d):
            receiver = GuardedManager()
            now = START * 0.002
            observe(receiver, now)
            doc = document(
                action, mode="M_ASSIST", base_poses=np.repeat(base[None], 41, axis=0)
            )
            doc["_pi0294"] = {
                "episode": d.name,
                "sequence": 0,
                "source_mpc_s": now,
                "ttl_s": 0.4,
                "operation": "install",
            }
            receiver.on_trajectory_json(message(doc))
            assert len(receiver.target_pub) == 1
            msg = receiver.target_pub[0]
            v = check(msg, 14)
            assert v.shape == (41, 14) and np.allclose(
                msg.time_trajectory, now + np.arange(41) * 0.05, rtol=0, atol=1e-8
            )
            assert np.allclose(v[:, :3], knots[:, :3, 3], rtol=0, atol=1e-7)
            assert np.allclose(
                quat_to_matrix(v[:, 3:7]), knots[:, :3, :3], rtol=0, atol=2e-7
            )
            assert np.allclose(v[:, 7:10], base[:3, 3], rtol=0, atol=1e-7)
            assert np.allclose(
                quat_to_matrix(v[:, 10:]), base[:3, :3], rtol=0, atol=2e-7
            )
        metrics[mode] = {
            "original_gate": full,
            "geometry": report,
            "offline_receiver_knots": len(v),
        }
        if mode == "A1":
            metrics["negative"] = negative(
                manager, START, wire, receipt, state, old, knots
            )
    return {
        "id": r["id"],
        "old_wire_sha256": wire,
        "old_sequence": seq,
        "checks": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    spec = baseline(root)
    configure(spec)
    import mujoco
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.native_replay import forbid_integration

    cfg = read(BASE / "reports/SLA_FROZEN_FEEDBACK/F0_REGISTRATION.json")
    public = root / "v15/d3-verified01/D3_VERIFIED.json"
    pinned(public, cfg["source_D3_sha256"])
    scene = Path(spec["model_path"])
    pinned(scene, cfg["geometry"]["scene_sha256"])
    model = mujoco.MjModel.from_binary_path(str(scene))
    out = a.output.resolve()
    out.mkdir()
    with forbid_integration(mujoco) as calls:
        geometry = alignment(model, cfg["geometry"])
        cases = [
            case(root, r, out, cfg, model, binding(spec["native_root"]))
            for r in read(public)["records"]
        ]
    assert not calls
    record(
        out / "F0_SITE.json",
        {
            "scope": "ARCHIVED_BOUNDARY_CPU_ONLY",
            "scene_sha256": sha(scene),
            "site_audit_sha256": sha(Path(__file__)),
            "control_sha256": sha(Path(__file__).with_name("feedback.py")),
            "geometry": geometry,
            "records": cases,
            "physics_ticks": 0,
            "native_requests": 0,
            "receiver_source_sha256": sha(
                BASE / "scripts/sla_transfer_v14/clock_guard.py"
            ),
        },
    )


if __name__ == "__main__":
    main()
