"""Verify the #488 denominator, owner closeout and task scores from real native records."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from reach_adapter_panel import ORDER, verify_registration
from reach_native import read, require, sha


def jsonl(path):
    if path.is_file():
        with path.open() as stream:
            yield from map(json.loads, stream)


def feedback_readout(case, episode, ledger):
    events = [r for r in ledger if r["kind"] in ("COMPILED", "VISUAL_REACH")]
    if len(events) != 2:
        return None
    start = events[1]["tick"]
    entry = read(case / "REACH_ENTRY.json")
    axis = np.asarray(entry["direction"], float)
    atoms = {
        r["native_end_step"]: r
        for r in jsonl(case / episode / "MOBILE_SAMPLES.private.jsonl")
    }
    if start not in atoms:
        raise ValueError("replan missing native boundary")
    data = dict(
        intervals=0,
        command_path_mm=0.0,
        actual_path_mm=0.0,
        actual_task_advance_mm=0.0,
        actual_red_progress_mm=0.0,
    )
    for r in ledger:
        if r["kind"] not in ("PREVIEW", "ADAPTED_PREVIEW") or r["tick"] < start:
            continue
        phases = r.get("source_s", r.get("source_phase_s"))
        if r["kind"] == "ADAPTED_PREVIEW":
            # The compiled 50-point array holds the last selected pose after source 0.50 s.
            phases = np.minimum(phases, 0.50)
        for k in (0, 1):
            t = r["tick"] + 25 * k
            if phases[k + 1] <= phases[k] or t + 25 not in atoms:
                continue
            a, b = atoms[t], atoms[t + 25]
            require(
                b["solver_exact_installed_targets"],
                "post-feedback source not solver acknowledged",
            )
            x, y = np.asarray([a["actual_TCP"], b["actual_TCP"]])[:, :3, 3]
            dx = np.asarray(b["target_endpoint_EEF10"][:3]) - np.asarray(
                a["target_endpoint_EEF10"][:3]
            )
            goal_a, goal_b = np.asarray([a["object_pose"], b["object_pose"]])[:, :3, 3]
            data["intervals"] += 1
            data["command_path_mm"] += 1000 * np.linalg.norm(dx)
            data["actual_path_mm"] += 1000 * np.linalg.norm(y - x)
            data["actual_task_advance_mm"] += 1000 * (y - x) @ axis
            data["actual_red_progress_mm"] += 1000 * (
                np.linalg.norm(x - goal_a) - np.linalg.norm(y - goal_b)
            )
    return dict(query_tick=start, solver_acknowledged_selected_phase=data)


def endpoint_readout(case, episode):
    entry_path = case / "REACH_ENTRY.json"
    if not entry_path.exists():
        return None
    entry = read(entry_path)
    atoms = {
        r["native_end_step"]: r
        for r in jsonl(case / episode / "MOBILE_SAMPLES.private.jsonl")
    }
    if not atoms or max(atoms) <= entry["tick"]:
        return None
    a, b = atoms[entry["tick"]], atoms[max(atoms)]
    installed = np.asarray(b["target_endpoint_EEF10"][:3]) - np.asarray(
        a["target_endpoint_EEF10"][:3]
    )
    actual = np.asarray(b["actual_TCP"])[:3, 3] - np.asarray(a["actual_TCP"])[:3, 3]
    axis = np.asarray(entry["direction"])
    return dict(
        last_atom_tick=max(atoms),
        installed_axis_mm=float(1000 * installed @ axis),
        installed_endpoint_error_mm=float(
            1000 * np.linalg.norm(installed - 0.02 * axis)
        ),
        actual_axis_mm=float(1000 * actual @ axis),
        actual_endpoint_error_mm=float(1000 * np.linalg.norm(actual - 0.02 * axis)),
        incremental_tracking_error_mm=float(1000 * np.linalg.norm(actual - installed)),
    )


def readout(panel):
    reg = read(panel / "REGISTRATION.json")
    verify_registration(reg)
    qual = read(panel / "QUALIFICATION.json")
    require(
        qual["ready_for_488"]
        and qual["registration_sha256"] == sha(panel / "REGISTRATION.json"),
        "panel unqualified",
    )
    closed = read(panel / "CLOSE.json")
    result = []
    for slot in ORDER:
        case = panel / slot
        if not (case / "NATIVE_REQUEST.json").is_file():
            result.append(
                dict(slot=slot, status="UNSTARTED", ticks=0, forwards=0, returned=0)
            )
            continue
        owner = read(case / "RESULT.private.json")
        audit = read(case / "AUDIT.json")
        lease = read(case / "control/LEASE.json")
        require(
            owner["card"] == slot and lease["status"] == "CLOSED" and owner["absent"],
            "unclosed case",
        )
        require(len(owner["exits"]) == 4, "missing process closure evidence")
        episode = owner["episode"]
        request = read(case / "NATIVE_REQUEST.json")
        require(request["episode"] == episode, "original episode mismatch")
        queries = []
        for q in sorted(case.glob("query-*")):
            events = [event["event"] for event in jsonl(q / "events.jsonl")]
            req = read(q / "REQUEST.json")
            ret = read(q / "RETURN.json") if (q / "RETURN.json").exists() else None
            require(req["episode"] == episode, "cross-episode feedback")
            require(
                not ret
                or (
                    ret["episode"] == episode
                    and ret["output_sha256"] == sha(q / "OUTPUT.npz")
                ),
                "unbound model result",
            )
            queries.append(
                dict(
                    tick=req["tick"],
                    seed=req["seed"],
                    input_sha256=req["input_sha256"],
                    entered=events.count("FORWARD_ENTERED"),
                    returned=bool(ret),
                    output_sha256=None if ret is None else ret["output_sha256"],
                )
            )
        require(
            len(queries) <= (0 if slot.endswith("-GEOMETRY") else 2),
            "per-card forward cap",
        )
        ticks = audit.get("ticks")
        require(
            isinstance(ticks, int) and 0 <= ticks <= request["max_ticks"] == 20000,
            "native accounting",
        )
        task = owner.get("score")
        ledger = list(jsonl(case / "POLICY_OWNER.jsonl"))
        ref = owner.get("refusal")
        result.append(
            dict(
                slot=slot,
                status=owner["outcome"],
                ticks=ticks,
                forwards=sum(q["entered"] for q in queries),
                returned=sum(q["returned"] for q in queries),
                closed=True,
                no_survivors=True,
                exit_codes=[v["exit_code"] for v in owner["exits"]],
                task=task,
                end_tick=owner.get("end"),
                entry=read(case / "REACH_ENTRY.json")["tick"]
                if (case / "REACH_ENTRY.json").is_file()
                else None,
                student=owner.get("student"),
                query_receipts=queries,
                refusal=ref,
                visual_updates=[
                    {
                        k: v[k]
                        for k in (
                            "tick",
                            "phase",
                            "profile",
                            "input",
                            "post_own_motion",
                        )
                    }
                    for v in ledger
                    if v["kind"] == "VISUAL_REACH"
                ],
                feedback_source=feedback_readout(case, episode, ledger),
                last_endpoint=endpoint_readout(case, episode),
                result_sha256=sha(case / "RESULT.private.json"),
                audit_sha256=sha(case / "AUDIT.json"),
                technical_summary=(
                    owner.get("interrupt") or audit.get("readout_error") or ""
                )
                .strip()
                .split("\n")[-1],
            )
        )
    started = [r["slot"] for r in result if r["status"] != "UNSTARTED"]
    entry_residuals = {}
    for layout in ("G00", "G01"):
        sources = [
            read(panel / r["slot"] / "REACH_ENTRY.json")
            for r in result
            if r["slot"].startswith(layout) and r.get("entry") is not None
        ]
        if len(sources) >= 2:
            entry_residuals[layout] = dict(
                count=len(sources),
                maximum_pairwise_actual_TCP_mm=float(
                    1000
                    * max(
                        np.linalg.norm(np.asarray(a["actual_xyz"]) - b["actual_xyz"])
                        for a, b in combinations(sources, 2)
                    )
                ),
            )
    ticks = sum(r["ticks"] for r in result)
    entered = sum(r["forwards"] for r in result)
    require(
        started == closed["started"]
        and ticks <= 120000
        and entered <= 8
        and closed["updates"] == 0,
        "cohort accounting",
    )
    return dict(
        issue=488,
        registration_sha256=sha(panel / "REGISTRATION.json"),
        preflight=qual["proof"],
        starts=len(started),
        entry_residuals=entry_residuals,
        total_ticks=ticks,
        forwards_entered=entered,
        forwards_returned=sum(r["returned"] for r in result),
        updates=0,
        rows=result,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", type=Path)
    print(json.dumps(readout(p.parse_args().panel), indent=2, allow_nan=False))
