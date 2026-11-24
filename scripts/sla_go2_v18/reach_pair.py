"""Run the four preallocated #483 Reach slots on the qualified frozen two-cube sites."""

import argparse
import os
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
from reach_native import (
    BASE,
    check_occupancy,
    configure,
    execute,
    pin,
    read,
    record,
    require,
    sha,
    sites,
)
from reach_policy_input import sensor_input

ORDER = ("G00-H0", "G00-A", "G01-H0", "G01-A")
REG = BASE / "reports/SLA_GO2_V18/Z1_NATIVE/REGISTRATION.json"


def paired_audit(case, r, spec):
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from reach_native import audit

    if case.name.endswith("-H0") and r["outcome"] == "H0_PASS":
        val = audit(case, r, spec)
        return {**val, "completed": True, "outcome": r["outcome"]}
    n = case / r["episode"]
    trace = audit_trace(
        n / "init.trace", max_native_ticks=20000, state_abi=(41, 38, 20)
    )
    ticks = sum(1 for _ in packets(n / "state.bin.journal"))
    known = r["outcome"] in (
        "A_PASS",
        "A_REFERENCE_REFUSED",
        "A_REACH_FAILED",
        "H0_REACH_FAILED",
    )
    require(
        known
        and r.get("end") == ticks == trace["native_step_records"]
        and trace["status"] == "RECORDED_TRACE_CHECKS_PASS"
        and not trace["STOP_observed"]
        and r["absent"]
        and len(r["exits"]) == 4
        and all(p["exit_code"] == 0 for p in r["exits"])
        and not r.get("close_error")
        and r["cancel"]["mode"] == "HOLD",
        "incomplete settled paired outcome",
    )
    identities = read(n / "LOADED_EXECUTOR_00.private.json")
    for role, key in (
        ("sim", "native_binary"),
        ("solver", "solver_binary"),
        ("controller", "controller_binary"),
    ):
        require(
            identities[role]["executable_sha256"] == spec["frozen_files"][spec[key]],
            "frozen binary mismatch",
        )
    return {
        "qualified": r["outcome"] == "A_PASS",
        "completed": True,
        "ticks": ticks,
        "outcome": r["outcome"],
        "student": r.get("student"),
        "task": r.get("score"),
        "refusal": r.get("refusal"),
    }


def setup(root, rows):
    base = rows[0][1]
    scratch = root / "v18/pair-scratch"
    for d in ("tmp", "cache", "logs"):
        (scratch / d).mkdir(parents=True, exist_ok=True, mode=0o700)
    configure(dict(base, native_environment=environment(base, scratch)))
    sys.path[:0] = [str(root / "v12"), str(root / "v12/CPU_LIB")]


def environment(base, d):
    return {
        **base["native_environment"],
        "TMPDIR": str(d / "tmp"),
        "XDG_CACHE_HOME": str(d / "cache"),
        "ROS_LOG_DIR": str(d / "logs"),
    }


def preflight(root, rows, reg):
    import mujoco
    from compiler import SourceCurve
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from reach_forward import site
    from standing_reach import StateMonitor

    qual = read(pin(root / "v18/e1-final.json", reg["qualification_sha256"]))
    require(
        qual["qualified_for_Z1"]
        and qual["native_requests"] == 3
        and all(x["status"] == "PASS" for x in qual["cases"]),
        "E1 incomplete",
    )
    old = read(BASE / "reports/SLA_HIERARCHICAL_V16/V4_FEEDBACK_NATIVE_INFERENCE.json")[
        "calls"
    ]["PARKED"]
    with np.load(
        pin(root / "v16/V4-FEEDBACK-01/PARKED.npz", old["file_sha256"]),
        allow_pickle=False,
    ) as saved:
        raw = saved["absolute"].copy()
    samples = []
    for i, slot in enumerate(("G00-H0", "G01-H0")):
        case = root / "v18/Q_E1_482_H0" / slot
        spec = read(case / "EXECUTION_SPEC.json")
        require(
            read(case / "control/LEASE.json")["status"] == "CLOSED"
            and read(case / "AUDIT.json")["qualified"]
            and spec["source_sha"] == rows[i][1]["source_sha"],
            "wrong source episode",
        )
        native = case / read(case / "NATIVE_REQUEST.json")["episode"]
        tick = read(case / "REACH_ENTRY.json")["tick"]
        obs, s = sensor_input(root, case, native, tick)
        require(
            obs["images"]["cam_high"].shape == (10, 256, 256, 3)
            and s["head_ticks"][-1] == tick
            and len(set(s["head_ticks"])) == 10,
            "invalid history",
        )
        with forbid_integration(mujoco) as calls:
            curve = SourceCurve(
                raw, np.array(s["world_origin"]), np.array(s["link6_tcp"])
            )
            require(
                curve.knots.shape == (50, 4, 4) and not calls, "archived decoder probe"
            )
            model = mujoco.MjModel.from_binary_path(spec["model_path"])
            StateMonitor(model)
        samples.append(
            {
                "slot": slot,
                "entry": tick,
                "sensor_sha256": s["input_sha256"],
                "scene": s["scene_sha256"],
                "old_return_only": old["file_sha256"],
            }
        )
    _, info = site(root)
    return {
        "slots": list(ORDER),
        "qualification_sha256": reg["qualification_sha256"],
        "CPU_sensor_and_decode": samples,
        "owner_code": {
            k: sha(Path(__file__).with_name(k))
            for k in (
                "reach_pair.py",
                "reach_native.py",
                "standing_reach.py",
                "reach_forward.py",
                "reach_student.py",
                "reach_policy_input.py",
            )
        },
        "gpu_uuid": info["gpu"],
        "frozen_forwards": 0,
        "native_ticks": 0,
    }


def main(args):
    root = args.root.resolve()
    reg = read(REG)
    require(
        reg["issue"] == 483
        and tuple(reg["slots"]) == ORDER
        and reg["ticks"] == 80000
        and reg["total_forwards"] == 24,
        "registration",
    )
    rows = sites(root)
    os.umask(0o077)
    setup(root, rows)
    from reach_student import Student

    proof = preflight(root, rows, reg)
    if args.preflight:
        record(args.preflight, proof)
        return
    require(read(args.execute) == proof, "student preflight drift")
    from d1_matched_owner import with_case_lease

    from scripts.pi0292.durable import OwnerLock

    dest = root / "v18/Z1_REACH_483"
    dest.mkdir(mode=0o700, exist_ok=False)
    record(dest / "REGISTRATION.json", reg)
    record(dest / "QUALIFICATION.json", read(root / "v18/e1-final.json"))
    record(dest / "PREFLIGHT.json", proof)
    pinpath = root / "v18/Q_E1_482_H0/G00-H0/EXECUTION_SPEC.json"
    pins = read(pinpath)["b2_code"]
    receiver = pin(
        root / "v18/receiver-preview02/student_grip.so", reg["receiver_sha256"]
    )
    results = []
    try:
        for slot in ORDER:
            if slot.endswith("-A") and results[-1]["outcome"] != "H0_PASS":
                results.append(
                    {"slot": slot, "outcome": "UNSTARTED_MATCHED_H0_FAILED", "ticks": 0}
                )
                continue
            profile, base = rows[int(slot.startswith("G01"))]
            os.sched_setaffinity(0, base["cpus"])
            with closing(OwnerLock(Path(base["mobile_world_lock"]))):
                check_occupancy(base)
                require(
                    not any(
                        (p / "STOP.json").exists()
                        for p in (dest, Path(base["campaign_root"]))
                    ),
                    "site stop",
                )
                case = dest / slot
                student = Student(root) if slot.endswith("-A") else None
                fields = {
                    "scope": "#483_PAIRED_STANDING_REACH",
                    "max_ticks": 20000,
                    "campaign_root": str(dest),
                    "outer_library": str(receiver),
                    "frozen_files": {
                        **base["frozen_files"],
                        str(receiver): sha(receiver),
                    },
                    "native_environment": environment(base, case),
                }
                value = with_case_lease(
                    case,
                    base,
                    pins,
                    sha(dest / "REGISTRATION.json"),
                    fields,
                    lambda spec, sp, base=base, profile=profile, student=student: (
                        execute(
                            base,
                            sp,
                            spec,
                            profile,
                            student=student,
                            scorer=paired_audit,
                        )
                    ),
                )
            require(
                read(case / "control/LEASE.json")["status"] == "CLOSED", "open lease"
            )
            outcome = read(case / "RESULT.private.json")["outcome"]
            results.append({"slot": slot, **value, "outcome": outcome})
            print(slot, outcome, value["ticks"], flush=True)
            if not value.get("completed"):
                break
    finally:
        record(
            dest / "PANEL_CLOSE.json",
            {"results": results, "unstarted": list(ORDER[len(results) :])},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", type=Path)
    g.add_argument("--execute", type=Path)
    main(p.parse_args())
