"""Count native slots, completed network calls, installed actions and causal source opportunities."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(path.read_text())


def lines(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(x) for x in f]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def demand(ok, text):
    if not ok:
        raise ValueError(text)


def geometry(raw, origin, tool, direction):
    arm = raw[:, 8:]
    T = np.tile(np.eye(4), (len(raw), 1, 1))
    T[:, :3, 3] = arm[:, :3]
    T[:, :3, :3] = Rotation.from_quat(arm[:, [4, 5, 6, 3]]).as_matrix()
    xyz = (origin @ T @ tool)[:, :3, 3]
    delta = xyz[-1] - xyz[0]
    return {
        "first_to_last_mm": float(1000 * np.linalg.norm(delta)),
        "first_to_last_task_projection_mm": float(1000 * (delta @ direction)),
        "source_working_opening_mm": [float(v) for v in 2 + 40 * arm[[0, -1], 7]],
    }


def run(root):
    group = root / "v18/Z1_REACH_483"
    panel = read(group / "PANEL_CLOSE.json")
    reg = read(group / "REGISTRATION.json")
    evidence = []
    for slot in reg["slots"]:
        c = group / slot
        if not (c / "NATIVE_REQUEST.json").exists():
            evidence.append(
                {
                    "slot": slot,
                    "outcome": "UNSTARTED",
                    "ticks": 0,
                    "forwards": 0,
                    "returns": 0,
                }
            )
            continue
        r = read(c / "RESULT.private.json")
        n = c / r["episode"]
        audit = read(c / "AUDIT.json")
        ticks = audit["ticks"]
        lease = read(c / "control/LEASE.json")
        demand(
            lease["status"] == "CLOSED"
            and lease["spec_sha256"] == sha(c / "EXECUTION_SPEC.json"),
            "lease",
        )
        entry = (
            read(c / "REACH_ENTRY.json") if (c / "REACH_ENTRY.json").exists() else {}
        )
        owner = lines(c / "POLICY_OWNER.jsonl")
        forwards = returns = 0
        queries = []
        for q in sorted(c.glob("query-*")):
            req = read(q / "REQUEST.json")
            events = lines(q / "events.jsonl")
            enter = sum(x["event"] == "FORWARD_ENTERED" for x in events)
            ret = sum(x["event"] == "FORWARD_RETURNED" for x in events)
            demand(
                0 <= ret <= enter <= 1 and req["episode"] == n.name,
                "query ownership/count",
            )
            info = {
                "number": req["number"],
                "tick": req["tick"],
                "seed": req["seed"],
                "input_sha256": req["input_sha256"],
                "model_entered": enter,
                "returned": ret,
            }
            forwards += enter
            returns += ret
            if ret:
                reply = read(q / "RETURN.json")
                demand(
                    reply["input_sha256"] == req["input_sha256"]
                    and reply["output_sha256"] == sha(q / "OUTPUT.npz"),
                    "unmatched return",
                )
                info["output_sha256"] = reply["output_sha256"]
                info["forward_call_seconds"] = next(
                    x["time_s"] for x in events if x["event"] == "FORWARD_RETURNED"
                ) - next(x["time_s"] for x in events if x["event"] == "FORWARD_ENTERED")
                record = next(
                    (
                        x
                        for x in owner
                        if x["kind"] == "RETURN" and x["tick"] == req["tick"]
                    ),
                    None,
                )
                with np.load(q / "OUTPUT.npz", allow_pickle=False) as z:
                    info.update(
                        raw_shape=list(z["actions"].shape),
                        absolute_shape=list(z["absolute"].shape),
                    )
                    if record:
                        meta = record["sensor"]
                        info.update(
                            head_ticks=meta["head_ticks"],
                            post_motion=record["post_motion"],
                            same_state_old_output_mean_difference_mm=record[
                                "same_state_old_output_mean_difference_mm"
                            ],
                            raw_direction=geometry(
                                z["absolute"],
                                np.asarray(meta["world_origin"]),
                                np.asarray(meta["link6_tcp"]),
                                np.asarray(entry["direction"]),
                            ),
                        )
            queries.append(info)
        wires = [
            w for w in lines(c / "REACH_WIRES.jsonl") if w.get("entry") is not None
        ]
        consumed = [
            x
            for x in lines(n / "MOBILE_SAMPLES.private.jsonl")
            if entry and x["native_start_step"] >= entry["tick"]
        ]
        recv = list(csv.reader((c / "student/receiver.csv").open()))
        after = sum(int(row[2]) > ticks for row in recv if row[0] == "NATIVE_STEP")
        demand(
            after == 0
            and r["absent"]
            and len(r["exits"]) == 4
            and all(isinstance(x["exit_code"], int) for x in r["exits"]),
            "unclosed simulator",
        )
        row = {
            "slot": slot,
            "outcome": r.get("outcome"),
            "completed": bool(audit.get("completed")),
            "ticks": ticks,
            "entry_tick": entry.get("tick"),
            "lease_closed": True,
            "no_survivors": r["absent"],
            "exit_codes": {x["name"]: x["exit_code"] for x in r["exits"]},
            "forwards": forwards,
            "returns": returns,
            "queries": queries,
            "task_reference_installs": len(wires),
            "post_entry_intervals": len(consumed),
            "solver_echoed_all": all(
                x["solver_exact_installed_targets"] for x in consumed
            )
            if consumed
            else None,
            "refusal": r.get("refusal"),
            "student": r.get("student"),
            "task": r.get("score"),
            "compiled": [
                {k: v for k, v in x.items() if k != "kind"}
                for x in owner
                if x["kind"] == "COMPILED"
            ],
        }
        if entry and consumed:
            start = np.asarray(entry["actual_xyz"])
            direction = np.asarray(entry["direction"])
            acts = np.asarray([x["actual_TCP"] for x in consumed])[:, :3, 3]
            row["actual_end_projection_mm"] = float(
                1000 * ((acts[-1] - start) @ direction)
            )
            row["max_actual_excursion_mm"] = float(
                1000 * max(np.linalg.norm(acts - start, axis=1))
            )
        if r.get("interrupt"):
            row["interruption"] = r["interrupt"].splitlines()[-1]
        if audit.get("readout_error"):
            row["audit_error"] = audit["readout_error"].splitlines()[-1]
        evidence.append(row)

    def samplecount(field):
        return sum(r.get(field, 0) for r in evidence)

    total = samplecount("ticks")
    calls = samplecount("forwards")
    demand(
        total <= reg["ticks"]
        and calls <= reg["total_forwards"]
        and all(r["forwards"] <= 12 for r in evidence),
        "panel bounds",
    )
    return {
        "issue": 483,
        "native_requests": sum(x["outcome"] != "UNSTARTED" for x in evidence),
        "predecessor_qualification": reg["qualification_sha256"],
        "native_ticks": total,
        "frozen_forwards": calls,
        "returned": samplecount("returns"),
        "training_updates": 0,
        "cases": evidence,
        "unstarted_slots": panel["unstarted"],
    }


if __name__ == "__main__":
    result = run(Path(sys.argv[1]).resolve())
    with Path(sys.argv[2]).open("x") as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write("\n")
