"""Reconcile the accepted 41-point references with native 2 ms body motion."""

import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np
from d1_readiness import baseline, configure, pinned, read, record, rows, sha


def lag_gain(command, achieved):
    """Fit one scalar gain with a shared causal lag; both vectors are relative to the actual start."""
    if np.max(np.linalg.norm(command, axis=1)) < 1e-5:
        return None
    these = []
    for lag in range(151):
        shifted = command[np.maximum(np.arange(len(command)) - lag, 0)]
        power = float(np.sum(shifted**2))
        if power == 0:
            continue
        gain = float(np.sum(shifted * achieved) / power)
        rms = float(np.sqrt(np.mean(np.sum((achieved - gain * shifted) ** 2, axis=1))))
        these.append((rms, lag, gain))
    rms, lag, gain = min(these)
    return {
        "lag_ms": 2 * lag,
        "gain": gain,
        "residual_rms": rms,
        "at_search_boundary": lag in (0, 150),
    }


def rms(vector):
    return float(np.sqrt(np.mean(np.sum(vector**2, axis=-1))))


def rotation_vector_deg(r):
    from go2arx5_vla.trajectory.so3 import matrix_to_quat

    q = matrix_to_quat(r)
    q = np.where(q[:, 3:] < 0, -q, q)
    n = np.linalg.norm(q[:, :3], axis=1, keepdims=True)
    return q[:, :3] * np.degrees(2 * np.arctan2(n, q[:, 3:])) / np.maximum(n, 1e-14)


def commands(folder, atoms, parts, condition):
    import json

    from d1_factorial_reference_matrix import START, check
    from d3_known_path import poses_at
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    from go2arx5_vla.trajectory.so3 import quat_slerp, quat_to_matrix

    events = list(rows(folder / "planner.jsonl"))
    sent = [
        r
        for r in rows(folder / "COMMANDS.private.jsonl")
        if r["phase"] == "proposed" and r["native_step"] >= START
    ]
    check(
        len(sent) == 20 and len(atoms) == 40,
        "all native installs and applied intervals",
    )
    xyz = []
    rot = []
    overlap = []
    previous = None
    for i, send in enumerate(sent):
        key = send["wire_sha256"]
        accepts = [
            e for e in events if e["kind"] == "accept" and e.get("ref_id") == key
        ]
        installed = [
            e for e in events if e["kind"] == "installed" and e.get("ref_id") == key
        ]
        check(len(accepts) == 1 and installed, "reference ACK")
        raw = accepts[0]["raw"]
        import hashlib

        check(hashlib.sha256(raw.encode()).hexdigest() == key, "wire digest")
        meta = json.loads(raw)["_pi0294"]
        t = (START + 50 * i) * 0.002
        check(
            abs(meta["source_mpc_s"] - t) < 1e-8
            and meta["episode"] == folder.name
            and meta["ttl_s"] == 0.4
            and meta["sequence"] == parts["old_sequence"] + 1 + i,
            "wire epoch clock",
        )
        inst = installed[0]
        knots = np.asarray(inst["poses"])
        check(
            knots.shape == (41, 14)
            and np.allclose(inst["times"], t + np.arange(41) * 0.05, rtol=0, atol=1e-8),
            "installed future grid",
        )
        expected = poses_at(
            np.asarray(parts["features"]["old_ee"]),
            condition["family"],
            condition["rate"],
            (2 * i + np.arange(41)) * 0.05,
        )
        expected_base = np.asarray(parts["features"]["old_base"])
        check(
            np.array_equal(knots[:, :3], expected[:, :3, 3].astype(np.float32))
            and np.allclose(
                quat_to_matrix(knots[:, 3:7]), expected[:, :3, :3], rtol=0, atol=2e-7
            )
            and np.allclose(knots[:, 7:10], expected_base[:3, 3], rtol=0, atol=2e-7)
            and np.allclose(
                quat_to_matrix(knots[:, 10:]), expected_base[:3, :3], rtol=0, atol=2e-7
            ),
            "decoded intended path",
        )
        ids = {
            trajectory_identity(e["times"], e["poses"], e["inputs"]) for e in installed
        }
        for j, atom in enumerate(atoms[2 * i : 2 * i + 2]):
            check(
                atom["installed_reference_sha256"] == key
                and ids.intersection(atom["solver_exact_installed_targets"])
                and atom["native_start_step"] == START + 50 * i + 25 * j
                and atom["ticks"] == 25,
                "native command join",
            )
        if previous is not None:
            overlap.append(float(np.max(abs(knots[:39] - previous[2:]))))
        previous = knots
        # Last install owns the 2 s endpoint. Earlier intervals are half open.
        u = np.arange(51 if i == 19 else 50) / 25
        idx = np.floor(u).astype(int)
        alpha = (u - idx)[:, None]
        xyz.extend((1 - alpha) * knots[idx, :3] + alpha * knots[idx + 1, :3])
        rot.extend(
            quat_to_matrix(
                quat_slerp(knots[idx, 3:7], knots[idx + 1, 3:7], alpha[:, 0])
            )
        )
    return np.asarray(xyz), np.asarray(rot), max(overlap)


def audit(folder, card, curve_out):
    import mujoco
    from b2_static_bias import ros_analysis
    from d1_factorial_reference_matrix import START, check
    from d2_policy_age_audit import JOINTS, kinematics
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from go2arx5_vla.world_model.native_replay import forbid_integration, packets

    from go2arx5_vla.trajectory.so3 import geodesic_angle

    result = read(folder / "RESULT.private.json")
    spec = read(folder / "EXECUTION_SPEC.json")
    lease = read(folder / "control/LEASE.json")
    check(
        lease["status"] == "CLOSED"
        and lease["spec_sha256"] == sha(folder / "EXECUTION_SPEC.json"),
        "lease binding",
    )
    out = folder / result["episode"]
    mpath = Path(spec["model_path"])
    pinned(mpath, spec["frozen_files"][str(mpath)])
    state = packets(out / "state.bin.journal")
    pa = read(out / "pause_ack.json")
    check(
        len(state) == result["ticks"] == pa["step"]
        and pa["clock_paused"]
        and not pa["running"],
        "native counters",
    )
    status = dict(
        **card,
        status="COMPLETE" if result["completed"] else "TECHNICAL_STOP",
        native_ticks=len(state),
        result_sha256=sha(folder / "RESULT.private.json"),
        journal_sha256=sha(out / "state.bin.journal"),
        post_init_ticks=max(0, len(state) - START),
        native_STOP=result.get("trace", {}).get("STOP_observed"),
        after_STOP_ticks=result.get("trace", {}).get("integrations_after_first_STOP"),
    )
    if not result["completed"]:
        return status
    parts = read(folder / "INITIAL.private.json")
    atoms = [
        a
        for a in rows(out / "MOBILE_SAMPLES.private.jsonl")
        if a["native_start_step"] >= START
    ]
    cmd, rcmd, seam = commands(out, atoms, parts, card)
    tr = read_trace(out / "init.trace")
    pd = sorted((r for r in tr if r["kind"] == 9), key=lambda r: r["monotonic_ns"])
    pdt = [r["monotonic_ns"] for r in pd]
    before = {r["token"]: r["monotonic_ns"] for r in tr if r["kind"] == 11}
    after = {r["token"]: r["monotonic_ns"] for r in tr if r["kind"] == 12}
    timing = ros_analysis(out, atoms, pd, after, tr)
    actual = []
    ra = []
    mount = []
    mount_r = []
    pd_tcp = []
    actual_q = []
    target_q = []
    loads = []
    contact_counts = []
    with forbid_integration(mujoco) as calls:
        model = mujoco.MjModel.from_binary_path(str(mpath))
        data = mujoco.MjData(model)
        tool = model.site("x5_ee").id
        body = model.body("x5_base_link").id
        joints = [model.joint(x).id for x in JOINTS]
        qa = model.jnt_qposadr[joints]
        motors = [model.actuator(x).id for x in JOINTS]
        for i, s in enumerate(state[START - 1 :]):
            t = START + i
            check(s["step"] == t, "journal step")
            ix = bisect_right(pdt, before[t]) - 1
            check(
                ix >= 0 and ix == bisect_right(pdt, after[t]) - 1,
                "unmatched native PD callback",
            )
            q = s["qpos"]
            qc = q.copy()
            qc[qa] = pd[ix]["values"][12:18]
            x, R, b, r = kinematics(model, data, q, tool, body)
            xcmd = kinematics(model, data, qc, tool, body)[0]
            actual.append(x)
            ra.append(R)
            mount.append(b)
            mount_r.append(r)
            pd_tcp.append(xcmd)
            actual_q.append(q[qa])
            target_q.append(qc[qa])
            loads.append(s["ctrl"][motors])
            contact_counts.append(len(s["contacts"]))
            if i and i % 25 == 0:
                a = atoms[i // 25 - 1]
                check(
                    a["safety_valid"]
                    and not a["native_grasp"]
                    and np.allclose(
                        x, np.array(a["actual_TCP"])[:3, 3], rtol=0, atol=1e-10
                    ),
                    "actual FK/mobile atom",
                )
    check(not calls and len(actual) == len(cmd) == 1001, "known path length")
    actual, ra, mount, mount_r, pd_tcp, actual_q, target_q = map(
        np.asarray, (actual, ra, mount, mount_r, pd_tcp, actual_q, target_q)
    )
    ideal_delta = 1000 * (cmd - cmd[0])
    delta = 1000 * (actual - actual[0])
    ideal_rot = rotation_vector_deg(rcmd @ rcmd[0].T)
    ach_rot = rotation_vector_deg(ra @ ra[0].T)
    error = 1000 * (actual - cmd)
    rel = delta - ideal_delta
    e_angle = np.degrees(geodesic_angle(ra, rcmd))
    inc_angle = np.degrees(geodesic_angle(ra @ ra[0].T, rcmd @ rcmd[0].T))

    def axis_gains(ideal, achieved):
        return [
            float(y / x) if abs(x) > 1e-5 else None
            for x, y in zip(ideal, achieved, strict=True)
        ]

    seams = []
    for ms in (500, 1000, 1500):
        k = ms // 2
        seams.append(
            {
                "ms": ms,
                "interval_ms": 4,
                "reference_delta_mm": (1000 * (cmd[k + 1] - cmd[k - 1])).tolist(),
                "PD_FK_delta_mm": (1000 * (pd_tcp[k + 1] - pd_tcp[k - 1])).tolist(),
                "actual_delta_mm": (1000 * (actual[k + 1] - actual[k - 1])).tolist(),
                "max_joint_target_difference_deg": float(
                    np.degrees(np.max(abs(target_q[k + 1] - target_q[k - 1])))
                ),
                "max_actual_joint_difference_deg": float(
                    np.degrees(np.max(abs(actual_q[k + 1] - actual_q[k - 1])))
                ),
            }
        )
    np.savez_compressed(
        curve_out,
        elapsed_ms=np.arange(1001) * 2,
        reference_xyz_m=cmd,
        actual_xyz_m=actual,
        pd_fk_xyz_m=pd_tcp,
        reference_rotation=rcmd,
        actual_rotation=ra,
        base_xyz_m=mount,
        base_rotation=mount_r,
        actual_joint_rad=actual_q,
        target_joint_rad=target_q,
    )
    status.update(
        B2_anchor_matched=parts["original_B2_match"]["matched"],
        initial_sha256=sha(folder / "INITIAL.private.json"),
        abs_position_RMS_mm=rms(error[1:]),
        abs_position_p95_mm=float(np.percentile(np.linalg.norm(error[1:], axis=1), 95)),
        abs_orientation_RMS_deg=float(np.sqrt(np.mean(e_angle[1:] ** 2))),
        relative_position_RMS_mm=rms(rel[1:]),
        relative_orientation_RMS_deg=float(np.sqrt(np.mean(inc_angle[1:] ** 2))),
        end_absolute_error_mm=error[-1].tolist(),
        command_motion_mm=ideal_delta[-1].tolist(),
        achieved_motion_mm=delta[-1].tolist(),
        endpoint_axis_gains=axis_gains(ideal_delta[-1], delta[-1]),
        command_rotation_vector_deg=ideal_rot[-1].tolist(),
        achieved_rotation_vector_deg=ach_rot[-1].tolist(),
        endpoint_rotation_axis_gains=axis_gains(ideal_rot[-1], ach_rot[-1]),
        position_lag_fit=lag_gain(ideal_delta, delta),
        rotation_lag_fit=lag_gain(ideal_rot, ach_rot),
        base_max_displacement_mm=float(
            1000 * np.max(np.linalg.norm(mount - mount[0], axis=1))
        ),
        base_max_rotation_deg=float(
            np.degrees(np.max(geodesic_angle(mount_r, mount_r[0])))
        ),
        pd_fk_minus_actual_z_mm_start_end=(
            1000 * (pd_tcp - actual)[[0, -1], 2]
        ).tolist(),
        maximum_absolute_arm_ctrl_Nm=np.max(abs(np.asarray(loads)), axis=0).tolist(),
        contact_pairs_per_tick_min_max=[min(contact_counts), max(contact_counts)],
        robot_object_contact_intervals=0,
        exact_solver_echo_atoms=40,
        wire_installs=20,
        high_level_revisions=result["revisions"],
        p10_boundaries=seams,
        maximum_shared_wire_field_change=seam,
        solver_first_echo=timing["first_exact_policy_echo"],
        selected_MRT_policy_ID=None,
        selected_policy_age_ms=None,
        completion_fraction=1,
        private_curve_sha256=sha(curve_out),
    )
    return status


def main():
    cli = argparse.ArgumentParser()
    for a in ("root", "campaign", "output"):
        cli.add_argument(a, type=Path)
    arg = cli.parse_args()
    root = arg.root.resolve()
    configure(baseline(root))
    from d1_factorial_reference_matrix import verify_initial
    from d3_native_owner import PLAN_PIN

    folder = arg.campaign.resolve()
    out = arg.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    p = folder / "REGISTRATION.json"
    pinned(p, PLAN_PIN)
    plan = read(p)
    closed = read(folder / "PANEL_ACTUAL.json")
    specs = read(root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json")
    records = []
    for row in plan["roster"]:
        case = folder / row["id"]
        if not (case / "NATIVE_REQUEST.json").exists():
            records.append(
                dict(**row, status="NOT_RUN", native_ticks=0, post_init_ticks=0)
            )
            continue
        records.append(audit(case, row, out / (row["id"] + "_curve.private.npz")))
    pairs = []
    for a, b in zip(records[::2], records[1::2], strict=True):
        pair = {
            "cards": [a["id"], b["id"]],
            "family": a["family"],
            "rate": a["rate"],
            "qualified": False,
        }
        if a["status"] == b["status"] == "COMPLETE":
            left, right = [
                read(folder / x["id"] / "INITIAL.private.json") for x in (a, b)
            ]
            m = verify_initial(
                left["features"], [right["features"]], specs["pairwise_initial_limits"]
            )
            pair.update(
                within_pair=m["matched"],
                both_B2=a["B2_anchor_matched"] and b["B2_anchor_matched"],
                violations=m["violations"],
            )
            pair["qualified"] = pair["within_pair"] and pair["both_B2"]
            if pair["qualified"]:
                one, p10 = sorted((a, b), key=lambda x: x["mode"])
                pair["P10_minus_ONE"] = {
                    k: p10[k] - one[k]
                    for k in (
                        "abs_position_RMS_mm",
                        "relative_position_RMS_mm",
                        "abs_orientation_RMS_deg",
                        "relative_orientation_RMS_deg",
                    )
                }
        pairs.append(pair)
    ticks = sum(r["native_ticks"] for r in records)
    assert ticks == closed["native_ticks"] and len(closed["actual"]) == sum(
        r["status"] != "NOT_RUN" for r in records
    )
    record(
        out / "D3_VERIFIED.json",
        {
            "planned": 12,
            "attempted": len(closed["actual"]),
            "complete": closed["completed"],
            "native_ticks": ticks,
            "test_ticks": sum(r["post_init_ticks"] for r in records),
            "records": records,
            "pairs": pairs,
            "new_model_forwards": 0,
            "training_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
