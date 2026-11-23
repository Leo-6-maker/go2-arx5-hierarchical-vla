"""Acquire one registered V3 simulation lease or audit a closed case without a launch."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path

from moving_stop_card import (
    CARD,
    REPORTS,
    baseline,
    check_occupancy,
    configure,
    inputs,
    read,
    record,
    require,
    run_case,
    sha,
)

CAMPAIGN = "sla-v16-moving-stop-native-01"
CANDIDATE = REPORTS / "V3_NATIVE_STOP_CARD_CANDIDATE.json"


def validate_allocation(plan):
    require(
        (
            plan["card"],
            plan["campaign"],
            plan["max_native_starts"],
            plan["maximum_native_ticks"],
            plan["candidate_sha256"],
        )
        == (CARD, CAMPAIGN, 1, 6503, sha(CANDIDATE)),
        "allocation differs from the fixed candidate",
    )


def native_counts(case):
    """Keep the request, observed launch, complete state prefix and audited tick count distinct."""
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from verify_grip_card import states

    sp = case / "EXECUTION_SPEC.json"
    spec = read(sp)
    lease = read(case / "control/LEASE.json")
    require(
        lease["status"] == "CLOSED"
        and lease["spec_sha256"] == sha(sp)
        and lease["owner"] == spec["owner"]
        and Path(spec["registered_root"]) == case,
        "closed matching lease required",
    )
    request = case / "NATIVE_REQUEST.json"
    result_path = case / "RESULT.private.json"
    result = read(result_path) if result_path.is_file() else {}
    row = dict(
        card=case.name,
        registered_requests=1,
        issued_requests=int(request.exists()),
        simulator_starts=None,
        state_prefix_ticks=0,
        actual_ticks=None,
        errors=[],
        evidence_sha256={
            p.name: sha(p)
            for p in (sp, case / "control/LEASE.json", request, result_path)
            if p.is_file()
        },
    )
    if not request.is_file():
        row["errors"].append("no native request receipt")
        return row
    ep = read(request)["episode"]
    require(
        ep.startswith("native-")
        and Path(ep).name == ep
        and result.get("episode", ep) == ep,
        "ambiguous episode identity",
    )
    out = case / ep
    native = out / "state.bin.journal"
    trace = out / "init.trace"
    commands = out / "PROCESS_COMMANDS.private.json"
    sim_log = out / "sim.log"
    loaded = out / "LOADED_EXECUTOR_00.private.json"
    sim_wrappers = [
        r for r in (read(commands) if commands.is_file() else []) if r["name"] == "sim"
    ]
    row["sim_wrapper_receipts"] = len(sim_wrappers)
    require(len(sim_wrappers) <= 1, "multiple simulator launch receipts for one card")
    if loaded.is_file():
        require(
            read(loaded)["sim"]["executable_sha256"]
            == spec["frozen_files"][spec["native_binary"]],
            "observed simulator differs",
        )
        row["simulator_starts"] = 1
    if native.is_file():
        try:
            for s in states(out):
                row["state_prefix_ticks"] = s["step"]
        except (ValueError, OSError) as exc:
            row["errors"].append(str(exc))
        if row["state_prefix_ticks"]:
            row["simulator_starts"] = 1
    if trace.is_file():
        try:
            audit = audit_trace(trace, max_native_ticks=spec["max_ticks"])
            row["trace_step_records"] = audit["native_step_records"]
            if (
                audit["status"] == "RECORDED_TRACE_CHECKS_PASS"
                and native.is_file()
                and row["state_prefix_ticks"] == row["trace_step_records"]
                and not row["errors"]
            ):
                row["actual_ticks"] = row["state_prefix_ticks"]
            else:
                row["errors"].append("trace and state coverage not confirmed")
        except Exception as exc:
            row["errors"].append(str(exc))
    if (
        result
        and out.is_dir()
        and not any(
            p.exists()
            for p in (
                native,
                trace,
                sim_log,
                loaded,
                out / "BEFORE_FIRST_INTEGRATION.json",
                case / "student/receiver.csv",
            )
        )
        and not sim_wrappers
        and not any(r["name"] == "sim" for r in result.get("owned_exits", []))
    ):
        # The frozen launcher opens sim.log before Popen. A failed launch after that point stays unknown.
        row.update(simulator_starts=0, actual_ticks=0)
    if row["actual_ticks"] is None and not row["errors"]:
        row["errors"].append("independent tick count unavailable")
    for p in (native, trace, commands, loaded):
        if p.is_file():
            row["evidence_sha256"][p.name] = sha(p)
    return row


def closed_report(root, case):
    row = native_counts(case)
    row.update(outcome="TECHNICAL_STOP", qualified=False)
    try:
        return classify(root, case, row)
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        return row


def classify(root, case, row):
    from stop_card_evidence import score

    result_path = case / "RESULT.private.json"
    if not result_path.exists() or row["errors"]:
        return row
    result = read(result_path)
    if result.get("outcome") == "SCREENED_OUT":
        match = read(case / "INITIAL.private.json")["match"]
        paused = read(case / result["episode"] / "pause_ack.json")
        if (
            match["matched"] is False
            and match["violations"] == result.get("violations")
            and match["violations"]
            and row["actual_ticks"] == paused["step"] == 5703
            and paused["clock_paused"]
            and not paused["running"]
            and result["cancel"]["mode"] == "HOLD"
            and result["cancel"]["native_step"] == 5703
            and result["owned_processes_absent"]
            and not result.get("close_error")
            and len(result["owned_exits"]) == 4
            and {r["name"] for r in result["owned_exits"]}
            == {"sim", "solver", "controller", "planner"}
            and all(r["exit_code"] == 0 for r in result["owned_exits"])
        ):
            row.update(outcome="SCREENED_OUT", violations=match["violations"])
        else:
            row["errors"].append("unverified screening or cleanup")
        return row
    if result.get("outcome") == "FAULT_GRANT_AWAITING_INDEPENDENT_SCORE":
        scored = score(root, case / "EXECUTION_SPEC.json")
        require(
            row["actual_ticks"] == scored["actual_integrations"],
            "independent counts differ",
        )
        row.update(outcome="QUALIFIED", qualified=True, score=scored)
    return row


def run(root, allocation):
    from d1_matched_owner import with_case_lease
    from scripts.pi0292.durable import OwnerLock

    base, pins, receiver, _, _ = inputs(root)
    plan = read(allocation)
    validate_allocation(plan)
    dest = root / "v16" / CAMPAIGN
    os.umask(0o077)
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(
            not (Path(base["campaign_root"]) / "STOP.json").exists(), "operator STOP"
        )
        dest.mkdir(mode=0o700, exist_ok=False)
        record(dest / "REGISTRATION.json", plan)
        case = dest / CARD
        fields = dict(
            outer_library=str(receiver),
            max_ticks=6503,
            frozen_files={**base["frozen_files"], str(receiver): sha(receiver)},
        )
        try:
            with_case_lease(
                case,
                base,
                pins,
                sha(dest / "REGISTRATION.json"),
                fields,
                lambda spec, sp: run_case(root, spec, sp),
            )
        except Exception as exc:
            record(
                dest / "OWNER_INTERRUPTION.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            lp = case / "control/LEASE.json"
            if not lp.is_file() or read(lp)["status"] != "CLOSED":
                raise
        row = closed_report(root, case)
        record(dest / "CLOSED_READOUT.json", row)
    return row


def archive_control(root):
    f1 = root / "feedback/F1_VERIFIED.json"
    require(
        sha(f1)
        == read(REPORTS / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["F1"],
        "F1 verified record differs",
    )
    f1_report = read(f1)
    cohort = root / "feedback/sla-feedback-f1-a0-20261123-01"
    archived = []
    for old in f1_report["records"]:
        case = cohort / old["card"]
        require(
            sha(case / "RESULT.private.json") == old["result_sha256"],
            "F1 outcome differs",
        )
        row = native_counts(case)
        require(
            not row["errors"]
            and row["actual_ticks"] == old["ticks"]
            and row["simulator_starts"] == row["issued_requests"] == 1
            and row["evidence_sha256"]["state.bin.journal"] == old["journal_sha256"],
            "F1 physical count differs",
        )
        archived.append(dict(row, archived_outcome=old["outcome"]))
    require(
        len(archived) == f1_report["registered"] == 12
        and sum(r["actual_ticks"] for r in archived) == f1_report["physics_ticks"],
        "F1 population differs",
    )
    prior = read(REPORTS / "V2_STUDENT_GRIPPER_OWNER_ACTUAL.json")
    for tail in ("", "b"):
        case = (
            root
            / ("v16/sla-v16-student-gripper-native-01" + tail)
            / ("V2-GRIP-01" + tail.upper())
        )
        row = native_counts(case)
        if tail:
            require(
                row["actual_ticks"] == prior["native_ticks"]
                and row["simulator_starts"] == 1
                and row["evidence_sha256"]["state.bin.journal"]
                == prior["source_sha256"]["state.bin.journal"],
                "V2 native positive control differs",
            )
        else:
            require(
                row["actual_ticks"] == row["simulator_starts"] == 0
                and prior["full_denominator"]["technical_stops_before_process_start"]
                == 1,
                "V2 zero-start control differs",
            )
        require(not row["errors"] and row["issued_requests"] == 1, "V2 request differs")
        archived.append(
            dict(row, archived_outcome=read(case / "RESULT.private.json")["outcome"])
        )
    return dict(
        records=archived,
        historical_requests=len(archived),
        historical_starts=sum(r["simulator_starts"] for r in archived),
        historical_ticks=sum(r["actual_ticks"] for r in archived),
        new_native_requests=0,
        new_physics_ticks=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", type=Path, metavar="FINAL_SCIENTIFIC_ALLOCATION")
    group.add_argument("--read-closed", action="store_true")
    group.add_argument("--audit-archives", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    if args.execute:
        result = run(root, args.execute.resolve())
    elif args.audit_archives:
        result = archive_control(root)
    elif args.read_closed:
        result = closed_report(root, root / "v16" / CAMPAIGN / CARD)
    else:
        from scripts.pi0292.durable import OwnerLock

        base, _, receiver, anchors, _ = inputs(root)
        with closing(OwnerLock(Path(base["mobile_world_lock"]))):
            check_occupancy(base)
            require(
                not (Path(base["campaign_root"]) / "STOP.json").exists(),
                "operator STOP",
            )
            require(not (root / "v16" / CAMPAIGN).exists(), "campaign already exists")
        result = dict(
            candidate_sha256=sha(CANDIDATE),
            receiver_sha256=sha(receiver),
            anchors=len(anchors),
            availability_check_released=True,
            new_case_leases=0,
            new_native_starts=0,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
