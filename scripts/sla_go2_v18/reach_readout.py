"""Summarize closed #482 evidence, using measured native TCP and actual solver receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SLOTS = ("G00-STOP", "G00-H0", "G01-H0")


def read(path):
    return json.loads(path.read_text())


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def summarize(root, output):
    from scipy.spatial.transform import Rotation

    v = root / "v18"
    first, h0 = v / "Q_E1_482_d193", v / "Q_E1_482_H0"
    stop = read(v / "stop-reaudit.json")
    panel = read(h0 / "PANEL_CLOSE.json")
    check(panel["attempts"][0] == stop, "different STOP receipt")
    cases = []
    for name in SLOTS:
        c = (first if name == SLOTS[0] else h0) / name
        if not (c / "NATIVE_REQUEST.json").exists():
            check(name in panel["unstarted"], "unacknowledged unstarted slot")
            cases.append({"slot": name, "status": "UNSTARTED", "ticks": 0})
            continue
        req, r, lease = [
            read(c / p)
            for p in (
                "NATIVE_REQUEST.json",
                "RESULT.private.json",
                "control/LEASE.json",
            )
        ]
        native = c / r["episode"]
        check(
            lease["status"] == "CLOSED"
            and lease["spec_sha256"] == digest(c / "EXECUTION_SPEC.json")
            and r["absent"]
            and r["episode"] == req["episode"],
            "unclosed or unmatched owner",
        )
        audit = stop if name == SLOTS[0] else read(c / "AUDIT.json")
        entry = (
            read(c / "REACH_ENTRY.json") if (c / "REACH_ENTRY.json").exists() else None
        )
        row = {
            "slot": name,
            "ticks": audit["ticks"],
            "entry_tick": entry and entry["tick"],
            "status": (
                "PASS" if audit["qualified"] else r.get("outcome", "TECHNICAL_STOP")
            ),
            "lease_closed": True,
            "remaining_processes": len(r["survivors"]),
            "process_exits": {p["name"]: p["exit_code"] for p in r["exits"]},
            "receiver_sha256": read(c / "EXECUTION_SPEC.json")["frozen_files"][
                read(c / "EXECUTION_SPEC.json")["outer_library"]
            ],
        }
        files = [
            c / "NATIVE_REQUEST.json",
            c / "AUDIT.json",
            c / "RESULT.private.json",
            c / "EXECUTION_SPEC.json",
            c / "control/LEASE.json",
            c / "REACH_ENTRY.json",
            c / "REACH_WIRES.jsonl",
            c / "student/receiver.csv",
            *[
                native / f
                for f in (
                    "state.bin.journal",
                    "init.trace",
                    "init.trace.summary",
                    "init.trace.snapshots",
                    "planner.jsonl",
                    "CLOCK_RESUME_REQUESTS.jsonl",
                    "LOADED_EXECUTOR_00.private.json",
                    "MOBILE_SAMPLES.private.jsonl",
                    "physics.jsonl",
                    "COMMANDS.private.jsonl",
                    "ROS_CONSUMPTION.private.jsonl",
                    "initialization.jsonl",
                )
            ],
        ]
        row["evidence_sha256"] = {
            str(p.relative_to(c)).replace(r["episode"], "native"): digest(p)
            for p in files
            if p.is_file()
        }
        if entry and (native / "MOBILE_SAMPLES.private.jsonl").exists():
            anchor, direction = (
                np.asarray(entry["actual_xyz"]),
                np.asarray(entry["direction"]),
            )
            ref0 = np.asarray(entry["consumed"])[:3, 3]
            first_rot = np.asarray(entry["consumed"])[:3, :3]
            steps = []
            with (native / "MOBILE_SAMPLES.private.jsonl").open() as f:
                for x in map(json.loads, f):
                    step = x["native_end_step"]
                    if step <= entry["tick"]:
                        continue
                    actual = np.asarray(x["actual_TCP"])
                    installed = np.asarray(x["target_endpoint_EEF10"][:3])
                    steps.append(
                        (
                            step,
                            float(1000 * ((installed - ref0) @ direction)),
                            float(1000 * ((actual[:3, 3] - anchor) @ direction)),
                            float(1000 * np.linalg.norm(actual[:3, 3] - installed)),
                            bool(
                                x["solver_exact_installed_targets"]
                                and x["safety_valid"]
                            ),
                            float(
                                np.degrees(
                                    Rotation.from_matrix(
                                        actual[:3, :3] @ first_rot.T
                                    ).magnitude()
                                )
                            ),
                        )
                    )
            if steps:
                row["received_motion"] = {
                    "receipts": len(steps),
                    "all_solver_matches": all(z[4] for z in steps),
                    "last_tick": steps[-1][0],
                    "last_installed_projection_mm": steps[-1][1],
                    "last_achieved_projection_mm": steps[-1][2],
                    "last_absolute_tracking_error_mm": steps[-1][3],
                    "last_absolute_tracking_rotation_deg": steps[-1][5],
                }
        if name == SLOTS[0]:
            row.update(
                fault_tick=stop["source_tick"],
                stop_tick=stop["terminal"]["stop_tick"],
                actual_before_fault_50ms_mm=stop["actual_50ms_mm"],
                pending_reference_100ms_mm=stop["next_100ms_reference_mm"],
                outstanding_grant_end=stop["terminal"]["last_grant_end"],
                integrations_after_stop=stop["after_stop_integrations"],
                installs_after_stop=stop["after_stop_installs"],
                writer=stop["trace_writer"],
                initial_readout_error=read(c / "AUDIT.json")[
                    "readout_error"
                ].splitlines()[-1],
            )
        else:
            if r.get("score"):
                row["task"] = r["score"]
            if r.get("interrupt"):
                row["interruption"] = r["interrupt"].splitlines()[-1]
            if audit.get("readout_error"):
                row["audit_error"] = audit["readout_error"].splitlines()[-1]
        cases.append(row)
    q = all(x["status"] == "PASS" for x in cases)
    result = {
        "issue": 482,
        "cases": cases,
        "qualified_for_Z1": q,
        "native_requests": sum(r["status"] != "UNSTARTED" for r in cases),
        "ticks": sum(r["ticks"] for r in cases),
        "forwards": 0,
        "updates": 0,
        "previous_busy_probe_starts": 0,
    }
    with output.open("x") as dst:
        json.dump(result, dst, indent=2, allow_nan=False)
        dst.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)
