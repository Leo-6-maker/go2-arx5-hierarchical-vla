"""Fixed-start matched-B2 cohort. Reuse the sealed D1 native episode and original gates."""

import argparse
import hashlib
import os
import time
from contextlib import closing
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, rows, sha


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def run_leased(
    case_dir, conditional_factor, base, runtime_pins, cohort_hash, reg, prior
):
    from d1_native_owner import run_case

    fields = {
        "scope": "D1_MATCHED_B2_STANDING_SIM_ONLY",
        "d1_case": case_dir.name,
        "d1_factor": conditional_factor,
        "max_ticks": 6203,
        "mobile_condition": {"conditional_on_matching": conditional_factor},
    }
    return with_case_lease(
        case_dir,
        base,
        runtime_pins,
        cohort_hash,
        fields,
        lambda spec, sp: run_case(spec, sp, reg, prior),
    )


def with_case_lease(case_dir, base, runtime_pins, cohort_hash, fields, execute):
    from b2_owner import check_occupancy

    from scripts.pi0292.durable import OwnerLock, write

    for name in ("control", "tmp", "logs", "cache", "data"):
        (case_dir / name).mkdir(parents=True, mode=0o700)
    uid = case_dir.parent.name + "-" + case_dir.name
    spec = dict(base, **fields)
    spec.update(
        role=uid,
        owner=uid,
        registered_root=str(case_dir),
        mobile_group=str(case_dir.parent),
        b2_card=case_dir.name,
        b2_code=runtime_pins,
        deadline_epoch=time.time() + base["wall_limit_s"],
        stage_identity_sha256=hashlib.sha256((cohort_hash + uid).encode()).hexdigest(),
        **{k: str(case_dir / k) for k in ("logs", "tmp", "cache", "data")},
    )
    sp = case_dir / "EXECUTION_SPEC.json"
    lp = case_dir / "control/LEASE.json"
    record(sp, spec)
    lease = {
        "status": "ACTIVE",
        "spec_sha256": sha(sp),
        "owner": uid,
        "run_uuid": uid,
        "deadline_epoch": spec["deadline_epoch"],
    }
    lock = Path(base["mobile_world_lock"])
    registry = Path(base["registry"])
    with closing(OwnerLock(lock.parent / "registration.lock")):
        check_occupancy(base)
        db = read(registry)
        require(
            not any(
                r.get("path") == str(case_dir) or r.get("run_uuid") == uid
                for r in db["entries"]
            ),
            "duplicate start",
        )
        db["entries"].append(
            {
                "path": str(case_dir),
                "run_uuid": uid,
                "role": uid,
                "source_sha": spec["source_sha"],
                "owner": uid,
                "scope": spec["scope"],
                "executor_node": spec["host"],
                "local_lease": str(lp),
                "ROS_DOMAIN_ID": spec["ROS_DOMAIN_ID"],
                "corelab_world_owner": str(lock),
            }
        )
        write(registry, db)
        write(lp, lease)
    try:
        return execute(spec, sp)
    finally:
        write(lp, dict(lease, status="CLOSED", closed_epoch=time.time()))


def classify(folder, val, reg):
    from d2_policy_age_audit import read_case

    out = folder / val["episode"]
    match_path = folder / "INITIAL_MATCH.private.json"
    match = read(match_path)["evaluation"] if match_path.exists() else {}
    resp = {
        "slot": folder.name,
        "eligible": match.get("matched"),
        "factor": None,
        "ticks": val.get("ticks"),
        "status": "TECHNICAL_STOP",
        "violations": match.get("violations", {}),
        "result_sha256": sha(folder / "RESULT.private.json"),
    }
    close = read(folder / "CLOSE.private.json")
    trace = val.get("trace", {})
    if (
        "audit_error" in val
        or "seal_error" in val
        or "close_error" in val
        or val.get("loaded") != reg["frozen_executables"]
        or val.get("physics_stops") != 0
        or val.get("guard_stop") is not False
        or not close["trace_sealed"]
        or len(close["owned_exits"]) != 4
        or any(r["exit_code"] != 0 for r in close["owned_exits"])
        or trace.get("status") != "RECORDED_TRACE_CHECKS_PASS"
        or trace.get("STOP_observed") is not False
        or trace.get("native_step_records") != val.get("ticks")
        or (out / "GUARD_TERMINAL.json").exists()
    ):
        return resp
    pa = read(out / "pause_ack.json")
    if pa["step"] != val["ticks"] or not pa["clock_paused"] or pa["running"]:
        return resp
    commands = [
        r
        for r in rows(out / "COMMANDS.private.jsonl")
        if r["phase"] == "proposed" and r["native_step"] >= 5703
    ]
    if (
        match.get("matched") is False
        and match["violations"]
        and val.get("ticks") == 5703
        and not commands
        and not (folder / "D1_REF.private.json").exists()
        and val.get("failure", "")
        .strip()
        .rpartition("\n")[2]
        .startswith("ValueError: UNKNOWN_INITIAL_MATCH:")
    ):
        resp["status"] = "SCREENED_OUT"
        return resp
    if match.get("matched") is True:
        resp["factor"] = "D1-" + val["factor"]
    if not val["completed"] or len(commands) != 10 or match.get("matched") is not True:
        return resp
    wire_ids = {r["wire_sha256"] for r in commands}
    events = list(rows(out / "planner.jsonl"))
    if len(wire_ids) != 10 or any(
        {r.get("ref_id") for r in events if r["kind"] == kind} & wire_ids != wire_ids
        for kind in ("accept", "installed")
    ):
        return resp
    curve, ev = read_case(folder)
    record(folder / "CURVE.private.json", curve)
    record(folder / "RESPONSE.private.json", ev)
    resp.update(
        response={
            k: ev[k]
            for k in (
                "Y_actual_z_100ms_mm",
                "Y_PD_z_100ms_mm",
                "absolute_position_rms_mm",
                "incremental_position_rms_mm",
                "applied_PD_q_without_matching_known_request",
            )
        },
        native_installs=len(wire_ids),
        response_sha256=sha(folder / "RESPONSE.private.json"),
    )
    if ev["applied_PD_q_without_matching_known_request"] == 0:
        resp["status"] = "COMPLETE"
    return resp


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("plan", type=Path)
    cli.add_argument("plan_sha256")
    cli.add_argument("--preflight", action="store_true")
    a = cli.parse_args()
    root = a.root.resolve()
    base = baseline(root)
    configure(base)
    from b2_owner import check_occupancy, dry_receiver, verify
    from d1_native_readout import contrasts

    from scripts.pi0292.durable import OwnerLock

    pinned(a.plan, a.plan_sha256)
    plan = read(a.plan)
    schedule = [
        f for block in plan["treatment_sequence_by_eligible_block"] for f in block
    ]
    require(
        all(
            sorted(b) == ["MM", "MO", "OM", "OO"]
            for b in plan["treatment_sequence_by_eligible_block"]
        )
        and len(schedule) == plan["fixed_start_slots"] == 40,
        "allocation blocks",
    )
    ready = root / "v15/readiness01"
    rp = ready / "D1_FACTORIAL_4X2_REGISTER.json"
    pinned(rp, plan["old_register_sha256"])
    reg = read(rp)
    rr = read(ready / "D1_READINESS_ACTUAL.json")
    pub = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    pinned(pub, B2_PIN)
    reg["frozen_executables"] = read(pub)["frozen_sha256"]
    require(
        rr["cpu_ready"]
        and rr["frozen_sha256"] == reg["frozen_executables"]
        and rr["registration_sha256"] == sha(rp),
        "readiness",
    )
    for name, pin in reg["source_sha256"].items():
        pinned(Path(__file__).with_name(name), pin)
    pinned(Path(__file__).with_name("d1_native_owner.py"), plan["old_owner_sha256"])
    for name, pin in read(pub)["experiment_code_sha256"].items():
        pinned(Path(__file__).resolve().parent.parent / "sla_transfer_v14" / name, pin)
    anchors = []
    for h in rr["anchors"]:
        path = ready / (h["source"] + "_anchor.private.json")
        pinned(path, h["private_anchor_sha256"])
        require(
            h["private_anchor_sha256"] in reg["prior_anchor_sha256"],
            "unregistered anchor",
        )
        anchors.append(read(path)["initial_features"])
    require(len(anchors) == 2, "two archived anchors required")
    historic = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    runtime_pins = verify(base, historic)
    dry_receiver(base)
    out = root / "v15" / plan["campaign"]
    os.umask(0o077)
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(not out.exists(), "campaign already exists")
        if a.preflight:
            print("CPU, identity, receiver, occupancy and anchors PASS", flush=True)
            return
        out.mkdir(mode=0o700)
        record(out / "REGISTRATION.json", plan)
        record(
            out / "OWNER.json",
            {
                "pid": os.getpid(),
                "epoch": time.time(),
                "plan_sha256": sha(a.plan),
                "code": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            },
        )
        outcomes = []
        enrolled = []
        for slot in range(1, 1 + len(schedule)):
            require(
                not any(
                    (p / "STOP.json").exists()
                    for p in (out, Path(base["campaign_root"]))
                ),
                "operator STOP",
            )
            check_occupancy(base)
            folder = out / f"S{slot:02}"
            val = run_leased(
                folder,
                schedule[len(enrolled)],
                base,
                runtime_pins,
                sha(a.plan),
                reg,
                anchors,
            )
            row = classify(folder, val, reg)
            record(folder / "ENROLLMENT.json", row)
            outcomes.append(row)
            print(slot, row["status"], row["factor"], row["ticks"], flush=True)
            if row["status"] == "TECHNICAL_STOP":
                break
            if row["status"] == "COMPLETE":
                anchors.append(read(folder / "INITIAL_MATCH.private.json")["features"])
                enrolled.append(row)
        record(
            out / "COHORT_ACTUAL.json",
            {
                "registered_starts": len(schedule),
                "attempted": len(outcomes),
                "outcomes": outcomes,
                "complete_blocks": [
                    contrasts(enrolled[i : i + 4])
                    for i in range(0, len(enrolled) - 3, 4)
                ],
                "native_ticks": sum(
                    r["ticks"] for r in outcomes if r["ticks"] is not None
                ),
                "terminal": outcomes[-1]["status"]
                if outcomes[-1]["status"] == "TECHNICAL_STOP"
                else "FIXED_ROSTER_EXHAUSTED",
            },
        )


if __name__ == "__main__":
    main()
