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


def geometry(raw, origin, tool, direction, red, actual):
    arm = raw[:, 8:]
    T = np.tile(np.eye(4), (len(raw), 1, 1))
    T[:, :3, 3] = arm[:, :3]
    T[:, :3, :3] = Rotation.from_quat(arm[:, [4, 5, 6, 3]]).as_matrix()
    xyz = (origin @ T @ tool)[:, :3, 3]
    delta = xyz[-1] - xyz[0]
    return {
        "measured_start_distance_to_red_mm": float(1000 * np.linalg.norm(actual - red)),
        "first_distance_to_red_mm": float(1000 * np.linalg.norm(xyz[0] - red)),
        "last_distance_to_red_mm": float(1000 * np.linalg.norm(xyz[-1] - red)),
        "first_to_last_red_gain_mm": float(
            1000 * (np.linalg.norm(xyz[0] - red) - np.linalg.norm(xyz[-1] - red))
        ),
        "first_to_last_mm": float(1000 * np.linalg.norm(delta)),
        "first_to_last_task_projection_mm": float(1000 * (delta @ direction)),
        "source_working_opening_mm": [float(v) for v in 2 + 40 * arm[[0, -1], 7]],
    }


def source_only_native(case, native, entry, phases, pregrasp):
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    spec = read(case / "EXECUTION_SPEC.json")
    model = mujoco.MjModel.from_binary_path(spec["model_path"])
    data = mujoco.MjData(model)
    start = np.asarray(entry["actual_xyz"])
    direction = np.asarray(entry["direction"])
    cumulative = np.zeros(3)
    tail = []
    prev = None
    count = 0
    for packet in packets(native / "state.bin.journal"):
        tick = packet["step"]
        if tick < entry["tick"]:
            continue
        data.qpos[:] = packet["qpos"]
        mujoco.mj_kinematics(model, data)
        tcp = data.site("x5_ee").xpos.copy()
        if prev is None:
            demand(
                np.allclose(tcp, start, rtol=0, atol=2e-8),
                "wrong source accounting start",
            )
            red = data.geom("task_cube_geom").xpos.copy()
        else:
            begin = entry["tick"] + 25 * ((tick - entry["tick"] - 1) // 25)
            demand(begin in phases, "missing phase at measured tick")
            if phases[begin] == "source":
                cumulative += tcp - prev
                count += 1
        prev = tcp
        virtual = start + cumulative
        tail.append(
            [
                float(1000 * cumulative @ direction),
                float(
                    1000 * (np.linalg.norm(start - red) - np.linalg.norm(virtual - red))
                ),
                float(
                    1000
                    * (
                        np.linalg.norm(start - pregrasp)
                        - np.linalg.norm(virtual - pregrasp)
                    )
                ),
            ]
        )
        if len(tail) > 251:
            tail.pop(0)
    vals = np.asarray(tail)
    return {
        "source_native_intervals": count,
        "final_projection_mm": float(vals[-1, 0]),
        "final_red_gain_mm": float(vals[-1, 1]),
        "minimum_last_half_second_projection_mm": float(vals[:, 0].min()),
        "minimum_last_half_second_red_gain_mm": float(vals[:, 1].min()),
        "minimum_last_half_second_pregrasp_gain_mm": float(vals[:, 2].min()),
        "thresholds_met": bool(
            len(tail) == 251 and np.all(vals.min(axis=0) >= [12, 10, 10])
        ),
    }


def run(root):
    group = root / "v18/Z1_REACH_483"
    continuation = root / "v18/Z1_REACH_483_G01"
    panel = read(group / "PANEL_CLOSE.json")
    continuation_close = read(continuation / "PANEL_CLOSE.json")
    demand(panel["unstarted"] == ["G01-H0", "G01-A"], "different continuation")
    reg = read(group / "REGISTRATION.json")
    evidence = []
    for slot in reg["slots"]:
        if slot.startswith("G01"):
            demand(not (group / slot).exists(), "duplicate native registration")
            c = continuation / slot
        else:
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
        samples = lines(n / "MOBILE_SAMPLES.private.jsonl")
        by_tick = {x["native_end_step"]: x for x in samples}
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
                                np.asarray(by_tick[req["tick"]]["object_pose"])[:3, 3],
                                np.asarray(by_tick[req["tick"]]["actual_TCP"])[:3, 3],
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
        if slot.endswith("-A") and entry:
            phases = {}
            for event in owner:
                if event["kind"] == "PREVIEW":
                    for j in range(2):
                        phase = event["source_s"][j : j + 2]
                        phases[event["tick"] + 25 * j] = (
                            "source" if phase[1] > phase[0] else "alignment_or_hold"
                        )
            stats = {
                kind: {
                    "consumed_intervals": 0,
                    "actual_gain_to_red_mm": 0.0,
                    "actual_TCP_path_mm": 0.0,
                }
                for kind in ("source", "alignment_or_hold")
            }
            for interval in consumed:
                begin = interval["native_start_step"]
                require_phase = phases.get(begin)
                demand(
                    require_phase is not None and begin in by_tick,
                    "missing consumed policy phase",
                )
                prev = by_tick[begin]
                ac0 = np.asarray(prev["actual_TCP"])[:3, 3]
                ac1 = np.asarray(interval["actual_TCP"])[:3, 3]
                red = np.asarray(interval["object_pose"])[:3, 3]
                z = stats[require_phase]
                z["consumed_intervals"] += 1
                z["actual_gain_to_red_mm"] += float(
                    1000 * (np.linalg.norm(ac0 - red) - np.linalg.norm(ac1 - red))
                )
                z["actual_TCP_path_mm"] += float(1000 * np.linalg.norm(ac0 - ac1))
            row["phase_accounting"] = stats
            profiles = read(
                Path(__file__).resolve().parents[2]
                / "reports/SLA_GO2_V18/REACH/PROFILE_ACTUAL.json"
            )["results"]
            target = np.asarray(
                profiles[int(slot.startswith("G01"))]["original_teacher_pregrasp"]
            )
            row["source_only"] = source_only_native(c, n, entry, phases, target)
            row["registered_success"] = (
                r["outcome"] == "A_PASS" and row["source_only"]["thresholds_met"]
            )

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
        "unstarted_slots": continuation_close["unstarted"],
    }


if __name__ == "__main__":
    result = run(Path(sys.argv[1]).resolve())
    with Path(sys.argv[2]).open("x") as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write("\n")
