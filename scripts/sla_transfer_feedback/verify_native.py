"""Reconcile closed native cards against wire, journal, PD phase and paired goal."""

import argparse
from bisect import bisect_right
from pathlib import Path

from native_owner import baseline, configure, pinned, read, record, rows, sha


def verify_card(case, row, phase, model, control):
    import mujoco
    import numpy as np
    from d2_policy_age_audit import JOINTS
    from feedback import OPEN, START, require
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from native_known import packet, tcp

    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0294.task_contacts import allowed

    source = case / "RESULT.private.json"
    pinned(source, row["result_sha256"])
    res = read(source)
    lease = read(case / "control/LEASE.json")
    pinned(case / "EXECUTION_SPEC.json", lease["spec_sha256"])
    require(lease["status"] == "CLOSED", "unclosed owner")
    out = case / res["episode"]
    j = out / "state.bin.journal"
    if row["outcome"] == "TECHNICAL_STOP":
        return {
            "card": case.name,
            "outcome": row["outcome"],
            "eligible": row.get("eligible"),
            "ticks": row.get("ticks", 0),
            "result_sha256": sha(source),
            "journal_sha256": sha(j) if j.exists() else None,
            "lease_closed": True,
            "physics_stop": row.get("trace", {}).get("STOP_observed"),
            "after_physical_stop_ticks": row.get("trace", {}).get(
                "integrations_after_first_STOP"
            ),
            "errors": [
                key
                for key in ("failure", "audit_error", "seal_error", "close_error")
                if key in res
            ],
        }
    pinned(j, res["journal_sha256"])
    initial = read(case / "INITIAL.private.json")
    status = {
        "card": case.name,
        "outcome": row["outcome"],
        "eligible": row["eligible"],
        "ticks": row["ticks"],
        "reset_state_sha256": initial["reset_state_sha256"],
        "initial_sha256": sha(case / "INITIAL.private.json"),
        "result_sha256": sha(source),
        "journal_sha256": sha(j),
        "loaded_sha256": res["loaded"],
        "violations": row.get("violations", {}),
        "lease_closed": True,
        "physics_stop": row["trace"]["STOP_observed"],
        "after_physical_stop_ticks": row["trace"]["integrations_after_first_STOP"],
    }
    if row["outcome"] == "SCREENED_OUT":
        require(
            row["ticks"] == START and not initial["match"]["matched"],
            "screen accounting",
        )
        return status
    if not res["atoms"]:
        return dict(
            status,
            post_init_ticks=0,
            first_scientific_decision_tick=res.get("first_scientific_decision_tick"),
            issued_atoms=0,
            paired_F1=res.get("pair"),
        )
    curve = case / "TRACK.private.npz"
    pinned(curve, res["curve_sha256"])
    with np.load(curve, allow_pickle=False) as arr:
        actual = arr["actual"]
        goal = arr["goal"]
        ae = arr["abs_mm"]
        ang = arr["abs_deg"]
    own = initial["features"]
    prior = np.asarray(own["old_ee"])
    atom = [
        a
        for a in rows(out / "MOBILE_SAMPLES.private.jsonl")
        if a["native_start_step"] >= START
    ]
    require(
        len(atom) * 25 + 1 == len(ae) == len(actual) and len(atom) == res["atoms"],
        "atom count",
    )
    tr = read_trace(out / "init.trace")
    pd = sorted((e for e in tr if e["kind"] == 9), key=lambda e: e["monotonic_ns"])
    times = [e["monotonic_ns"] for e in pd]
    before = {e["token"]: e["monotonic_ns"] for e in tr if e["kind"] == 11}
    after = {e["token"]: e["monotonic_ns"] for e in tr if e["kind"] == 12}
    qa = model.jnt_qposadr[[model.joint(name).id for name in JOINTS]]
    tool = model.site("x5_ee").id
    data = mujoco.MjData(model)
    geoms = [model.geom(i).name for i in range(model.ngeom)]
    fk = []
    cmd_gap = []
    cmd_excursion = []
    cmd_steps = []
    safety = 0
    contact = 0
    ambiguous = 0
    pause = read(out / "pause_ack.json")
    require(
        pause["clock_paused"]
        and not pause["running"]
        and pause["step"] == res["ticks"],
        "terminal pause",
    )
    for i, a in enumerate(atom):
        require(
            a["native_start_step"] == START + 25 * i
            and a["native_end_step"] == START + 25 * (i + 1)
            and a["gripper_command"]["sequence"] == 128 + i
            and a["installed_EEF10"][9] == OPEN
            and a["solver_exact_installed_targets"],
            "consumed atom provenance",
        )
        p = matrix10(np.array(a["installed_EEF10"]))
        q = matrix10(np.array(a["target_endpoint_EEF10"]))
        require(np.allclose(p, prior, rtol=0, atol=2e-7), "cross-atom command seam")
        prior = q
        idx = (i + 1) * 25
        actual_end = np.array(a["actual_TCP"])
        require(
            np.allclose(actual[idx], actual_end, atol=1e-9, rtol=0), "native TCP phase"
        )
        cmd_gap.append(1000 * float(np.linalg.norm(q[:3, 3] - actual_end[:3, 3])))
        cmd_excursion.append(
            1000 * float(np.linalg.norm(q[:3, 3] - np.asarray(own["old_ee"])[:3, 3]))
        )
        cmd_steps.append(1000 * float(np.linalg.norm(q[:3, 3] - p[:3, 3])))
    with j.open("rb") as f:
        for tick in range(1, row["ticks"] + 1):
            p = packet(f)
            require(p["step"] == tick, "journal step")
            if tick < START:
                continue
            if tick > START:
                for c in p["contacts"]:
                    a, b = geoms[c["geom0"]], geoms[c["geom1"]]
                    if not allowed(a, b):
                        safety += 1
                    if "task_cube_geom" in (a, b) and "task_table" not in (a, b):
                        contact += 1
            idx = bisect_right(times, before[tick]) - 1
            if idx < 0 or idx != bisect_right(times, after[tick]) - 1:
                ambiguous += 1
                continue
            qc = p["qpos"].copy()
            qc[qa] = pd[idx]["values"][12:18]
            ef = tcp(model, data, tool, {"qpos": qc})
            fk.append((1000 * (ef[:3, 3] - actual[tick - START, :3, 3])).tolist())
        require(not f.read(1), "extra journal state")

    def rms(v):
        return float(np.sqrt(np.mean(np.asarray(v) ** 2)))

    status.update(
        curve_sha256=sha(curve),
        atoms=len(atom),
        installs=res["installs"],
        post_init_ticks=row["ticks"] - START,
        task_tracking=res["tracking"],
        max_last_second_mm_deg=[float(max(ae[-501:])), float(max(ang[-501:]))],
        ticks_with_ambiguous_PD_phase=ambiguous,
        fk_minus_actual_end_mm=fk[-1] if fk else None,
        fk_minus_actual_rms_mm=rms(np.linalg.norm(fk, axis=1)) if fk else None,
        issued_minus_achieved_at_atom_end_rms_mm=rms(cmd_gap),
        max_cmd_excursion_mm=max(cmd_excursion),
        max_50ms_step_mm=max(cmd_steps),
        prohibited_contact_records=safety,
        robot_cube_contact_records=contact,
        selected_MRT_policy_id=None,
        selected_policy_age=None,
        actual_solver_echo_atoms=len(atom),
    )
    if phase == "F1":
        status["baseline_reproduced"] = row["baseline_reproduced"]
    else:
        other = control / res["pair"]
        other_curve = other / "TRACK.private.npz"
        pinned(other_curve, res["paired_F1_curve_sha256"])
        require(
            initial["reset_state_sha256"]
            == read(other / "INITIAL.private.json")["reset_state_sha256"],
            "pair reset",
        )
        with np.load(other_curve, allow_pickle=False) as paired:
            n = len(ae)
            require(np.array_equal(goal, paired["goal"][:n]), "identical paired goals")
            status["paired_control"] = {
                "card": res["pair"],
                "shared_ms": 2 * (n - 1),
                "abs_rms_mm_deg": [rms(paired[k][1:n]) for k in ["abs_mm", "abs_deg"]],
                "delta_rms_mm_deg": [
                    rms(paired[k][1:n]) for k in ["delta_mm", "delta_deg"]
                ],
                "end_mm_deg": [float(paired[k][n - 1]) for k in ["abs_mm", "abs_deg"]],
                "max_last_second_mm_deg": [
                    float(max(paired[k][max(0, n - 501) : n]))
                    for k in ["abs_mm", "abs_deg"]
                ],
            }
        for k in [
            "first_scientific_decision_tick",
            "ticks_between_decision_and_paused_boundary",
            "first_qualified_hold_tick",
            "servo_ledger_sha256",
        ]:
            status[k] = res[k]
        pinned(case / "SERVO.private.jsonl", res["servo_ledger_sha256"])
        if res["outcome"] == "HOLD_COMPLETE":
            start = res["first_qualified_hold_tick"] - START
            end = res["first_scientific_decision_tick"] - START
            require(
                end - start == 500
                and start >= 1450
                and np.max(ae[start : end + 1]) <= 4 + 1e-9
                and np.max(ang[start : end + 1]) <= 2 + 1e-9,
                "scored stable dwell",
            )
    return status


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("phase", choices=["F1", "F2"])
    cli.add_argument("dest", type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    spec = baseline(root)
    configure(spec)
    import mujoco
    from feedback import require
    from go2arx5_vla.world_model.native_replay import forbid_integration

    registered = (
        Path(__file__).resolve().parents[2]
        / "reports/SLA_FROZEN_FEEDBACK/F1_F2_REGISTRATION.json"
    )
    campaign = root / "feedback" / read(registered)["campaigns"][a.phase]
    summary = read(campaign / "ACTUAL.json")
    pinned(campaign / "REGISTRATION.json", sha(registered))
    require(summary["registration_sha256"] == sha(registered), "registration")
    scene = Path(spec["model_path"])
    pinned(scene, read(registered)["scene_sha256"])
    model = mujoco.MjModel.from_binary_path(str(scene))
    controls = root / "feedback" / read(registered)["campaigns"]["F1"]
    with forbid_integration(mujoco) as calls:
        records = [
            verify_card(campaign / r["card"], r, a.phase, model, controls)
            for r in summary["records"]
        ]
    require(
        not calls
        and len(list(campaign.glob(a.phase + "-*/NATIVE_REQUEST.json")))
        == len(records),
        "native request census",
    )
    clean = all(
        r.get("prohibited_contact_records", 0) == 0
        and r.get("robot_cube_contact_records", 0) == 0
        and not r["physics_stop"]
        and r["lease_closed"]
        for r in records
    )
    ready = (
        summary["promote"] and clean
        if a.phase == "F1"
        else (
            clean
            and all(r["outcome"] in ("SCREENED_OUT", "HOLD_COMPLETE") for r in records)
            and summary["controls"] == summary["paired"] > 0
        )
    )
    record(
        a.dest,
        {
            "phase": a.phase,
            "stage_qualified": ready,
            "registered": 12,
            "started": len(records),
            "unstarted": 12 - len(records),
            "screened": sum(r["outcome"] == "SCREENED_OUT" for r in records),
            "physics_ticks": summary["physics_ticks"],
            "test_ticks": sum(max(0, r["ticks"] - 5703) for r in records),
            "records": records,
            "planned_controls": summary["controls"],
            "scientific_comparisons": sum("paired_control" in r for r in records),
            "SLA_forwards": 0,
            "training_updates": 0,
            "registration_sha256": sha(registered),
            "cohort_sha256": sha(campaign / "ACTUAL.json"),
            "audit_sha256": sha(Path(__file__)),
            "owner_code_sha256": read(campaign / "OWNER.json")["code"],
        },
    )


if __name__ == "__main__":
    main()
