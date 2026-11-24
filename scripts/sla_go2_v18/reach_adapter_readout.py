"""Verify the #488 denominator, owner closeout and task scores from real native records."""

import argparse
import json
from pathlib import Path

from reach_adapter_panel import ORDER, verify_registration
from reach_native import read, require, sha


def jsonl(path):
    if path.is_file():
        with path.open() as stream:
            yield from map(json.loads, stream)


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
