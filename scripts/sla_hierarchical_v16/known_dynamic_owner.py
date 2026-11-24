"""Fresh per-card owner and CPU entry rehearsal for the fixed #472 population."""

import argparse
import json
import os
import secrets
import sys
import traceback
from contextlib import closing
from pathlib import Path

from demonstrated_grip import pin, read, require
from known_dynamic_panel import CARDS, KnownCurve
from moving_stop_card import (
    BASE,
    REPORTS,
    baseline,
    check_occupancy,
    configure,
    groups_alive,
    inputs,
    receiver_adapter,
    record,
    sha,
)

PLAN = REPORTS / "KNOWN_DYNAMIC_NATIVE_REQUEST.json"
CAMPAIGN = "known-dynamic-472"
CAP = 21203


def controls(root):
    import numpy as np
    from fault_reference import records
    from feedback import OPEN, ReferenceManager
    from known_curve_scoring import metrics
    from known_dynamic_curve import trajectory

    from scripts.pi0292.eval_contract import matrix10

    cfgpath = BASE / "reports/SLA_FROZEN_FEEDBACK/F0_REGISTRATION.json"
    cfg = read(
        pin(cfgpath, read(cfgpath.with_name("F0_MODEL.json"))["registration_sha256"])
    )
    pin(
        Path(sys.modules["feedback"].__file__),
        read(cfgpath.with_name("F0_SITE.json"))["control_sha256"],
    )
    f = root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01"
    verified = read(
        pin(
            root / "feedback/F2_VERIFIED.json",
            read(REPORTS / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["F2"],
        )
    )
    witness = next(x for x in verified["records"] if x["card"] == "F2-01")
    original = read(pin(f / "RESULT.private.json", witness["result_sha256"]))
    with np.load(pin(f / "TRACK.private.npz", original["curve_sha256"])) as z:
        actual, goal = z["actual"][-1], z["goal"][-1]
    p = f / original["episode"] / "MOBILE_SAMPLES.private.jsonl"
    from native_known import read_line

    atom = read_line(p)
    logged = list(
        records(pin(f / "SERVO.private.jsonl", original["servo_ledger_sha256"]))
    )
    final = logged[-1]
    wire = atom["installed_reference_sha256"]
    entry = original["end"]
    anchor = matrix10(np.asarray(atom["target_endpoint_EEF10"]))
    start = matrix10(np.asarray(atom["installed_EEF10"]))
    require(
        final["wire"] == wire
        and final["end"] == entry
        and atom["safety_valid"]
        and np.allclose(final["actual_endpoint"], actual, rtol=0, atol=1e-9)
        and atom["solver_exact_installed_targets"],
        "independent archive entry witness",
    )
    import mujoco
    from verify_grip_card import states

    model = mujoco.MjModel.from_binary_path(
        str(Path(read(f / "EXECUTION_SPEC.json")["model_path"]))
    )
    names = [model.geom(i).name for i in range(model.ngeom)]
    count = 0
    for packet in states(f / original["episode"]):
        if 5703 <= packet["step"] <= entry:
            KnownCurve.support(names, packet)
            count += 1
    require(count == entry - 5703 + 1, "incomplete common-entry physical support")
    summaries = {}
    for card in CARDS:
        curve = KnownCurve(card)
        manager = ReferenceManager(
            "REPLAY",
            read(f / "INITIAL.private.json")["features"]["old_ee"],
            cfg["controller"],
        )
        manager.next_tick = entry
        manager.anchor = anchor.copy()
        manager.last_wire = wire
        receipt = {
            "episode": "REPLAY",
            "start": entry - 25,
            "end": entry,
            "source": entry - 50,
            "finger_fraction": OPEN,
            "finger_owner": "REPLAY",
            "solver_echo": True,
            "accepted": wire,
            "installed": wire,
            "consumed": wire,
            "end_pose": anchor.copy(),
            "start_pose": start.copy(),
        }
        for i in range(70):
            tick = entry + i * 50
            val = actual if i == 0 else trajectory(goal, [i * 0.1], *curve.rate)[0]
            measured = {
                "episode": "REPLAY",
                "tick": tick,
                "TCP": val,
                "safe": True,
                "contact": False,
            }
            knots, _ = curve.preview(
                manager, receipt, measured, goal, anchor, entry, tick
            )
            curve.admit(knots, receipt, {"tcp": val})
            h = "offline-" + str(i)
            receipt.update(
                start=tick + 25,
                end=tick + 50,
                source=tick,
                accepted=h,
                installed=h,
                consumed=h,
                start_pose=knots[1],
                end_pose=knots[2],
            )
            manager.consumed(
                tick + 50, h, knots, receipt, dict(measured, tick=tick + 50)
            )
        nominal = trajectory(goal, np.arange(500, 2501) * 0.002, *curve.rate)
        good = metrics(nominal, nominal)
        park = metrics(np.repeat(nominal[:1], 2001, axis=0), nominal)
        summaries[card] = {
            "rehearsed_updates": 70,
            "actual_archive_entry_tick": entry,
            "four_foot_archive_samples": count,
            "constant": park,
            "positive": good,
            "entry_offset_mm": float(
                1000 * np.linalg.norm(anchor[:3, 3] - goal[:3, 3])
            ),
        }
    return cfg, summaries


def preflight(root):
    prepared = inputs(root)
    cfg, panel = controls(root)
    r = read(root / "v17/cap-replay02.json")
    old, new = r["6503"], r["21203"]
    require(
        sha(prepared[2]) == old["receiver_sha256"]
        and new["frozen_trace_sha256"] == old["frozen_trace_sha256"]
        and old["treatments"] == new["treatments"],
        "cap-only receiver replay mismatch",
    )
    path = root / "v17/cap-21203-replay02/receiver/student_grip.so"
    pin(path, new["receiver_sha256"])
    fp = read(BASE / "reports/SLA_FROZEN_FEEDBACK/F1_F2_REGISTRATION.json")
    scene = pin(Path(prepared[0]["model_path"]), fp["scene_sha256"])
    static = read(
        pin(scene.parent / "STATIC_RESULT.private.json", fp["scene_receipt_sha256"])
    )
    require(
        static["saved_startup_input_sha256"] == fp["fixed_reset_input_sha256"],
        "registered scene/reset",
    )
    return (prepared[0], prepared[1], path, *prepared[3:]), cfg, panel


def execute(root, sp, spec, prepared, cfg, *, coverage=None):
    from b2_native import paused
    from clock_guard import new_receiver_output
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from native_owner import eligibility
    from native_servo import drive
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile
    from verify_grip_card import states

    case = sp.parent
    base, pins, receiver, anchors, limits = prepared
    lease = mobile.registered(spec, sp)
    require(
        spec["max_ticks"] == CAP
        and spec["b2_code"] == pins
        and Path(spec["outer_library"]) == receiver
        and spec["frozen_files"]
        == {**base["frozen_files"], str(receiver): sha(receiver)}
        and case.name in (CARDS if coverage is None else coverage.slots)
        and Path(spec["registered_root"]) == case,
        "world binding",
    )
    out = new_receiver_output(case)
    channel = case / "student"
    channel.mkdir(mode=0o700)
    epoch = secrets.token_hex(32)
    record(
        channel / "authority.json",
        {
            "owner": lease["owner"],
            "epoch": epoch,
            "episode": out.name,
            "receiver_sha256": sha(receiver),
            "lease_at_start_sha256": sha(case / "control/LEASE.json"),
        },
    )
    result = {
        "card": case.name,
        "episode": out.name,
        "outcome": "TECHNICAL_STOP",
        "last_requested_tick": 0,
    }
    a = receiver_adapter(base, spec, case, out, channel, epoch, result, grant_cap=CAP)
    record(
        case / "NATIVE_REQUEST.json",
        {"card": case.name, "max_ticks": CAP, "episode": out.name},
    )
    try:
        a.start()
        parts, sample, seq, receipt, match = eligibility(a, case, anchors, limits, None)
        result.update(eligible=match["matched"], violations=match["violations"])
        condition = case.name if coverage is None else coverage.assign(case, match)
        if match["matched"]:
            result["condition"] = condition
            result.update(
                drive(
                    a,
                    case,
                    parts,
                    sample,
                    seq,
                    receipt,
                    parts["old_ee"],
                    cfg,
                    dynamic=KnownCurve(condition),
                )
            )
        else:
            result.update(outcome="SCREENED_OUT", end=5703, atoms=0, installs=0)
        paused(a, result["end"])
        cancel = a.cancel_reference()
        require(
            cancel["mode"] == "HOLD" and cancel["native_step"] == result["end"],
            "terminal HOLD",
        )
        a.seal_trace(result["end"])
    except Exception:  # noqa: BLE001 - persist errors and always release the native owner.
        result["exception"] = traceback.format_exc()
        result["outcome"] = "TECHNICAL_STOP"
    finally:
        try:
            a.close()
        except Exception:  # noqa: BLE001 - persist errors and always release the native owner.
            result["close_error"] = traceback.format_exc()
        result["owned_exits"] = a.exit_records
        result["survivors"] = groups_alive({p.pid for _, p in a.processes})
        result["owned_child_absent"] = (
            bool(result.get("sim_child_pid"))
            and not (Path("/proc") / str(result["sim_child_pid"])).exists()
        )
        record(case / "RESULT.private.json", result)
    last = 0
    for state in states(out):
        require(state["step"] == last + 1, "native state gap")
        last += 1
    audit = audit_trace(out / "init.trace", max_native_ticks=CAP)
    snap = {
        "outcome": result["outcome"],
        "ticks": last,
        "trace": {
            k: audit[k]
            for k in (
                "status",
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
            )
        },
        "receiver_stop_observed": any(
            "STOP" in line.split(",")[0] for line in (channel / "receiver.csv").open()
        ),
        "owned_exits": [r["exit_code"] for r in a.exit_records],
        "survivors": result["survivors"],
        "child_absent": result["owned_child_absent"],
        "scene_guard_stop": any(
            (out / name).exists()
            for name in ("GUARD_TERMINAL.json", "MOBILE_SAFETY_STOP.private.json")
        ),
        "physics_stops": sum(
            json.loads(s)["kind"] == "stop" for s in (out / "physics.jsonl").open()
        ),
    }
    snap["clean"] = bool(
        audit["status"] == "RECORDED_TRACE_CHECKS_PASS"
        and audit["native_step_records"] == last == result.get("end")
        and last <= CAP
        and not audit["STOP_observed"]
        and not snap["receiver_stop_observed"]
        and not snap["scene_guard_stop"]
        and not snap["physics_stops"]
        and not result["survivors"]
        and snap["child_absent"]
        and snap["owned_exits"] == [0] * 4
        and result["outcome"] not in ("TECHNICAL_STOP", "CONTACT")
    )
    record(case / "AUDIT.json", snap)
    return snap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--execute", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    os.umask(0o077)
    from d1_matched_owner import with_case_lease

    from scripts.pi0292.durable import OwnerLock

    prepared, cfg, panel = preflight(root)
    if args.preflight:
        record(
            args.preflight,
            {
                "controls": panel,
                "receiver_sha256": sha(prepared[2]),
                "physics_ticks": 0,
            },
        )
        return
    plan = read(args.execute)
    reg = read(PLAN)
    pin(root / "v17/cap-replay02.json", plan["cpu_stop_comparison_sha256"])
    receipt = read(
        pin(root / "v17/panel-final-preflight.json", plan["cpu_dynamic_preview_sha256"])
    )
    require(receipt["controls"] == panel, "prelaunch controls changed")
    require(
        plan["candidate_sha256"] == sha(PLAN)
        and plan["order"] == reg["scientific_slots"] == list(CARDS)
        and plan["per_card_cap"] == reg["max_ticks_per_start"] == CAP
        and plan["total_ticks"] == reg["max_total_ticks"] == 4 * CAP
        and plan["receiver_sha256"] == sha(prepared[2])
        and plan["issued"],
        "unallocated card/binding",
    )
    base, pins, receiver, _, _ = prepared
    dest = root / "v17" / CAMPAIGN
    dest.mkdir(mode=0o700, exist_ok=False)
    record(dest / "REGISTRATION.json", plan)
    os.sched_setaffinity(0, base["cpus"])
    outcome = []
    try:
        for card in CARDS:
            case = dest / card
            with closing(OwnerLock(Path(base["mobile_world_lock"]))):
                check_occupancy(base)
                require(
                    not any(
                        (p / "STOP.json").exists()
                        for p in (dest, Path(base["campaign_root"]))
                    ),
                    "operator STOP",
                )
                fields = {
                    "outer_library": str(receiver),
                    "max_ticks": CAP,
                    "scope": "#472_KNOWN_DYNAMIC_STANDING_SIM_ONLY",
                    "frozen_files": {
                        **base["frozen_files"],
                        str(receiver): sha(receiver),
                    },
                }
                with_case_lease(
                    case,
                    base,
                    pins,
                    sha(dest / "REGISTRATION.json"),
                    fields,
                    lambda spec, sp: execute(root, sp, spec, prepared, cfg),
                )
            row = read(case / "AUDIT.json")
            require(
                read(case / "control/LEASE.json")["status"] == "CLOSED",
                "lease remains active",
            )
            outcome.append(row)
            print(card, row["outcome"], row["ticks"], row["clean"], flush=True)
            if not row["clean"]:
                break
    finally:
        record(
            dest / "PANEL_CLOSE.json",
            {"completed_case_audits": outcome, "scientific_roster": list(CARDS)},
        )


if __name__ == "__main__":
    main()
