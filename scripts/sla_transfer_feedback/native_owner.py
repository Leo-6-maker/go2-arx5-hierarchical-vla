"""A new registered F1/F2 population on the unchanged no-contact site owner."""

import argparse
import os
import sys
import time
import traceback
from contextlib import closing
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_transfer_v15"))
from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, rows, sha


def eligibility(adapter, folder, anchors, limits, pair):
    from b2_native import fixed, paused
    from d1_factorial_reference_matrix import boundary, initial_features, verify_initial
    from d1_native_owner import native_prefix_pd
    from feedback import START, require
    from native_known import atom_receipt, read_line, reset_hash
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    adapter.prepare_dynamic()
    out = adapter.output
    require(
        read(out / "GUARD_READY.json")
        == {
            "episode": out.name,
            "pid": next(p.pid for n, p in adapter.processes if n == "planner"),
        },
        "guard owner",
    )
    mount = mobile.arm(adapter)
    sample = adapter.measured_task_state()
    action, base = fixed(sample, mount)
    for i in range(128):
        if i % 7 == 0:
            mobile.install(adapter, action, base)
        sample, _ = mobile.consume(adapter, action, sequence=i)
    paused(adapter, START)
    last = read_line(out / "MOBILE_SAMPLES.private.jsonl")
    parts, wire, seq = boundary(out, last)
    f = initial_features(
        sample, parts, native_prefix_pd(out / "init.trace"), last["installed_EEF10"][9]
    )
    match = verify_initial(f, anchors, limits)
    seed = reset_hash(out)
    if pair is not None:
        other = read(pair / "INITIAL.private.json")
        assert other["match"]["matched"]
        if seed != other["reset_state_sha256"]:
            match["matched"] = False
            match["violations"]["first_qpos_qvel_sha256"] = {
                "observed": seed,
                "required": other["reset_state_sha256"],
            }
    record(
        folder / "INITIAL.private.json",
        {
            "features": f,
            "match": match,
            "reset_state_sha256": seed,
            "old_wire": wire,
            "pair": pair.name if pair else None,
        },
    )
    source = round(adapter.active_reference["source_mpc_s"] / 0.002)
    rec = atom_receipt(out, last, sample, source, wire)
    return parts, sample, seq, rec, match


def known_task(adapter, folder, parts, sample, seq):
    from d3_known_path_preview import sweep_path
    from feedback import START, require
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from native_known import atom_receipt, install, read_line, window
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    sweep = sweep_path(
        adapter.actual_state.model,
        sample,
        parts,
        "translation",
        "slow",
        binding(adapter.spec["native_root"]),
    )
    record(folder / "SWEEP.private.json", sweep)
    for atom in range(480):
        tick = START + 25 * atom
        if atom % 2 == 0:
            poses = window(parts["old_ee"], atom * 0.05)
            action, wire = install(
                adapter, poses, parts["old_base"], seq + 1 + atom // 2, tick
            )
            source = tick
        sample, _ = mobile.consume(adapter, action, sequence=128 + atom)
        last = read_line(adapter.output / "MOBILE_SAMPLES.private.jsonl")
        receipt = atom_receipt(adapter.output, last, sample, source, wire)
        require(
            receipt["end"] == tick + 25
            and last["gripper_command"]["sequence"] == 128 + atom,
            "atom clock",
        )
        if (folder.parent / "STOP.json").exists():
            raise RuntimeError("operator STOP")
    return {
        "outcome": "A0_COMPLETE",
        "end": START + 12000,
        "atoms": 480,
        "installs": 240,
    }


def run_case(spec, sp, phase, anchors, limits, pair, frozen, gate):
    import numpy as np
    from b2_native import TechnicalAdapter, paused, samples
    from clock_guard import new_receiver_output
    from feedback import START, require
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.native_replay import packets
    from native_known import readout
    from scripts.wm02.mobile_boundary_runner_v1 import registered

    folder = sp.parent
    registered(spec, sp)
    out = new_receiver_output(folder)
    adapter = TechnicalAdapter(spec, out)
    result = {
        "card": folder.name,
        "episode": out.name,
        "outcome": "TECHNICAL_STOP",
        "pair": pair.name if pair else None,
    }
    record(
        folder / "NATIVE_REQUEST.json",
        {"epoch": time.time(), "episode": out.name, "pid": os.getpid()},
    )
    try:
        adapter.start()
        parts, sample, seq, receipt, match = eligibility(
            adapter, folder, anchors, limits, pair
        )
        result["eligible"] = match["matched"]
        result["violations"] = match["violations"]
        if not match["matched"]:
            result.update(outcome="SCREENED_OUT", end=START, atoms=0, installs=0)
        else:
            old = (
                parts["old_ee"]
                if pair is None
                else np.asarray(
                    read(pair / "INITIAL.private.json")["features"]["old_ee"]
                )
            )
            if phase == "F1":
                details = known_task(adapter, folder, parts, sample, seq)
            else:
                from native_servo import drive

                details = drive(adapter, folder, parts, sample, seq, receipt, old, gate)
            result.update(details)
        end = result["end"]
        paused(adapter, end)
        cancel = adapter.cancel_reference()
        require(
            cancel["mode"] == "HOLD" and cancel["native_step"] == end, "terminal HOLD"
        )
        adapter.seal_trace(end)
        if match["matched"] and end > START:
            curve = folder / "TRACK.private.npz"
            result["tracking"] = readout(
                out, old, adapter.actual_state.model, end, curve
            )
            result["curve_sha256"] = sha(curve)
    except BaseException:  # noqa: BLE001 - record interruptions and close the native owner.
        result["failure"] = traceback.format_exc()
    finally:
        if adapter.node is not None:
            try:
                if adapter.processes and all(
                    p.poll() is None for _, p in adapter.processes
                ):
                    adapter.capture_loaded_identity()
                if not adapter.trace_sealed:
                    adapter.seal_trace(adapter.observe()["step"])
            except BaseException:  # noqa: BLE001 - record interruptions and close the native owner.
                result["seal_error"] = traceback.format_exc()
        try:
            adapter.close()
        except BaseException:  # noqa: BLE001 - record interruptions and close the native owner.
            result["close_error"] = traceback.format_exc()
    try:
        dur = len(packets(out / "state.bin.journal"))
        result["ticks"] = dur
        trace = audit_trace(out / "init.trace", max_native_ticks=spec["max_ticks"])
        result["trace"] = {
            k: trace[k]
            for k in (
                "status",
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
            )
        }
        result["physics_stops"] = sum(
            r["kind"] == "stop" for r in rows(out / "physics.jsonl")
        )
        result["guard_stop"] = (out / "MOBILE_SAFETY_STOP.private.json").exists() or (
            out / "GUARD_TERMINAL.json"
        ).exists()
        result["loaded"] = {
            k: v["executable_sha256"]
            for k, v in read(out / "LOADED_EXECUTOR_00.private.json").items()
            if k in frozen
        }
        result["closure"] = {
            "sealed": adapter.trace_sealed,
            "exit_codes": [r["exit_code"] for r in adapter.exit_records],
        }
        m = samples(out / "MOBILE_SAMPLES.private.jsonl")
        result["atom_evidence"] = m
        require(
            trace["status"] == "RECORDED_TRACE_CHECKS_PASS"
            and dur == trace["native_step_records"] == result["end"]
            and result["physics_stops"] == 0
            and not (trace["STOP_observed"] or result["guard_stop"])
            and result["loaded"] == frozen
            and result["closure"] == {"sealed": True, "exit_codes": [0] * 4}
            and (
                m.get("atoms", 0)
                == m.get("atoms_with_exact_solver_echo", 0)
                == m.get("valid_no_grasp_atoms", 0)
                == result["atoms"]
            )
            and m.get("distinct_installs", 0) == result["installs"],
            "closure or command evidence",
        )
        if result["atoms"]:
            require(m["arm_requests"] > 0 and m["base_commands"] > 0, "actual requests")
        result["journal_sha256"] = sha(out / "state.bin.journal")
    except BaseException:  # noqa: BLE001 - record interruptions and close the native owner.
        result["audit_error"] = traceback.format_exc()
    if any(
        k in result for k in ("failure", "seal_error", "close_error", "audit_error")
    ):
        result["outcome"] = "TECHNICAL_STOP"
    record(folder / "RESULT.private.json", result)
    return result


def qualified(row, rule):
    t = row["tracking"]

    def within(key, value):
        return rule[key][0] <= value <= rule[key][1]

    return (
        abs(t["z_100ms_mm"]) <= rule["max_abs_z_first_100ms_mm"]
        and within("hold_abs_rms_mm", t["hold_rms_mm"])
        and within("motion_abs_rms_mm", t["motion_rms_mm"])
        and within("achieved_x_during_motion_mm", t["motion_xyz_mm"][0])
        and within("achieved_z_during_motion_mm", t["motion_xyz_mm"][2])
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("phase", choices=["F1", "F2"])
    p.add_argument("plan_sha")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    base = baseline(root)
    configure(base)
    from b2_owner import check_occupancy, dry_receiver, verify
    from d1_matched_owner import with_case_lease
    from feedback import require

    from scripts.pi0292.durable import OwnerLock

    rdir = BASE / "reports/SLA_FROZEN_FEEDBACK"
    rp = rdir / "F1_F2_REGISTRATION.json"
    pinned(rp, a.plan_sha)
    plan = read(rp)
    for name, h in plan["f0_pins"].items():
        pinned(rdir / name, h)
    cfg = read(rdir / "F0_REGISTRATION.json")
    pinned(
        rdir / "F0_REGISTRATION.json",
        read(rdir / "F0_MODEL.json")["registration_sha256"],
    )
    pinned(
        Path(__file__).with_name("feedback.py"),
        read(rdir / "F0_SITE.json")["control_sha256"],
    )
    scene = Path(base["model_path"])
    pinned(scene, plan["scene_sha256"])
    static = scene.parent / "STATIC_RESULT.private.json"
    pinned(static, plan["scene_receipt_sha256"])
    require(
        read(static)["saved_startup_input_sha256"] == plan["fixed_reset_input_sha256"],
        "task seed",
    )
    pub = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    pinned(pub, B2_PIN)
    frozen = read(pub)["frozen_sha256"]
    regpath = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pinned(regpath, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    reg = read(regpath)
    limits = reg["pairwise_initial_limits"]
    anchors = []
    for name, h in read(pub)["experiment_code_sha256"].items():
        pinned(BASE / "scripts/sla_transfer_v14" / name, h)
    for f in sorted(regpath.parent.glob("*_anchor.private.json")):
        require(sha(f) in reg["prior_anchor_sha256"], "B2 anchor")
        anchors.append(read(f)["initial_features"])
    require(len(anchors) == 2, "two B2 anchors")
    hist = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    pins = verify(base, hist)
    dry_receiver(base)
    dest = root / "feedback" / plan["campaigns"][a.phase]
    controls = []
    cursor = 0
    preflight_result = None
    if a.phase == "F2":
        from native_servo import preflight

        preflight_result = preflight(root, cfg)
        f1 = dest.parent / plan["campaigns"]["F1"]
        prev = read(f1 / "ACTUAL.json")
        require(
            prev["promote"] and prev["registration_sha256"] == sha(rp),
            "F1 control gate",
        )
        validated = read(root / "feedback" / "F1_VERIFIED.json")
        require(
            validated["stage_qualified"]
            and validated["cohort_sha256"] == sha(f1 / "ACTUAL.json"),
            "independent F1 physical readout",
        )
        for card in validated["records"]:
            if card["outcome"] == "A0_COMPLETE":
                cp = f1 / card["card"]
                pinned(cp / "RESULT.private.json", card["result_sha256"])
                pinned(cp / "INITIAL.private.json", card["initial_sha256"])
                pinned(cp / "TRACK.private.npz", card["curve_sha256"])
        offline = read(rdir / "F2_PREFLIGHT.json")
        for source in ("native_servo.py", "feedback.py", "verify_native.py"):
            pinned(Path(__file__).with_name(source), offline["source_sha256"][source])
        require(
            offline["fault_receipts"] == preflight_result,
            "F2 negative controls drifted",
        )
        controls = [
            f1 / r["card"] for r in prev["records"] if r["outcome"] == "A0_COMPLETE"
        ]
    os.umask(0o077)
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(not dest.exists(), "registered campaign already present")
        if a.preflight:
            print(a.phase, "preflight passed", flush=True)
            return
        dest.mkdir()
        record(dest / "REGISTRATION.json", plan)
        if preflight_result is not None:
            record(dest / "SERVO_PREFLIGHT.json", preflight_result)
        record(
            dest / "OWNER.json",
            {
                "epoch": time.time(),
                "pid": os.getpid(),
                "plan_sha256": sha(rp),
                "code": {f.name: sha(f) for f in Path(__file__).parent.glob("*.py")},
            },
        )
        outcomes = []
        for i in range(12):
            if a.phase == "F2" and cursor == len(controls):
                break
            require(
                not any(
                    (x / "STOP.json").exists()
                    for x in [dest, Path(base["campaign_root"])]
                ),
                "operator STOP",
            )
            check_occupancy(base)
            pair = controls[cursor] if controls else None
            prior = (
                [*anchors, read(pair / "INITIAL.private.json")["features"]]
                if pair
                else anchors
            )
            name = a.phase + "-" + str(i + 1).zfill(2)
            folder = dest / name
            fields = {
                "scope": a.phase + "_STANDING_KNOWN_GOAL_SIM_ONLY",
                "mobile_condition": {"phase": a.phase, "card": name},
                "max_ticks": plan["caps"]["max_ticks_per_enrolled_start"],
            }

            def execute(spec, sp, prior=prior, pair=pair):
                return run_case(spec, sp, a.phase, prior, limits, pair, frozen, cfg)

            result = with_case_lease(folder, base, pins, sha(rp), fields, execute)
            row = {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "episode",
                    "failure",
                    "seal_error",
                    "audit_error",
                    "close_error",
                )
            }
            row["result_sha256"] = sha(folder / "RESULT.private.json")
            row["lease_closed"] = (
                read(folder / "control/LEASE.json")["status"] == "CLOSED"
            )
            if row["outcome"] == "A0_COMPLETE":
                row["baseline_reproduced"] = qualified(row, plan["F1_gate"])
            outcomes.append(row)
            print(name, row["outcome"], row.get("ticks"), flush=True)
            if not row["lease_closed"] or row["outcome"] not in (
                "A0_COMPLETE",
                "SCREENED_OUT",
                "HOLD_COMPLETE",
            ):
                break
            if row["outcome"] != "SCREENED_OUT":
                anchors.append(read(folder / "INITIAL.private.json")["features"])
                cursor += 1
        f1_ok = [r for r in outcomes if r["outcome"] == "A0_COMPLETE"]
        promote = bool(
            a.phase == "F1"
            and len(outcomes) == 12
            and f1_ok
            and all(r.get("baseline_reproduced", False) for r in f1_ok)
            and all(
                r["outcome"] in ("A0_COMPLETE", "SCREENED_OUT") and r["lease_closed"]
                for r in outcomes
            )
        )
        record(
            dest / "ACTUAL.json",
            {
                "phase": a.phase,
                "registered": 12,
                "started": len(outcomes),
                "unstarted": 12 - len(outcomes),
                "records": outcomes,
                "registration_sha256": sha(rp),
                "promote": promote,
                "controls": len(controls),
                "paired": cursor if controls else 0,
                "physics_ticks": sum(r.get("ticks", 0) for r in outcomes),
                "new_model_forwards": 0,
                "training_updates": 0,
            },
        )


if __name__ == "__main__":
    main()
