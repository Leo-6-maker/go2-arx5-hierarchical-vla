"""Use only the two unstarted #483 G01 slots after G00's closed resource interruption."""

import argparse
import fcntl
import os
from pathlib import Path

from reach_native import pin, read, record, require, sha, sites
from reach_pair import REG, preflight, run_slot, setup


def main(args):
    root = args.root.resolve()
    os.umask(0o077)
    reg = read(REG)
    rows = sites(root)
    setup(root, rows)
    from reach_forward import site

    initial = root / "v18/Z1_REACH_483"
    first = read(initial / "PANEL_CLOSE.json")
    require(
        reg == read(initial / "REGISTRATION.json")
        and reg["issue"] == 483
        and first["unstarted"] == ["G01-H0", "G01-A"]
        and [r["slot"] for r in first["results"]] == ["G00-H0", "G00-A"]
        and first["results"][0]["outcome"] == "H0_PASS"
        and first["results"][1]["outcome"] == "TECHNICAL_STOP"
        and not any((initial / s).exists() for s in first["unstarted"]),
        "unused allocation",
    )
    require(os.environ.get("REACH_SHARED_GPU") == "1", "user sharing decision required")
    c = initial / "G00-A"
    prev = read(c / "RESULT.private.json")
    require(
        prev["absent"]
        and prev["last_requested_tick"] == 8203
        and prev["interrupt"].splitlines()[-1]
        == "ValueError: GPU0 has additional users"
        and read(c / "control/LEASE.json")["status"] == "CLOSED"
        and not (c / "query-01/events.jsonl").exists(),
        "previous interrupted query already entered",
    )
    proof = preflight(root, rows, reg)
    proof["continuation_code"] = sha(Path(__file__))
    proof["closed_panel"] = sha(initial / "PANEL_CLOSE.json")
    if args.preflight:
        record(args.preflight, proof)
        return
    require(proof == read(args.execute), "new owner preflight mismatch")
    dest = root / "v18/Z1_REACH_483_G01"
    dest.mkdir(mode=0o700)
    record(dest / "REGISTRATION.json", reg)
    record(dest / "QUALIFICATION.json", read(initial / "QUALIFICATION.json"))
    record(dest / "PREFLIGHT.json", proof)
    record(
        dest / "RESOURCE_DECISION.json",
        {
            "shared_GPU0_authorized": True,
            "fresh_exclusive_world_lease": True,
            "unchanged_remaining_slots": first["unstarted"],
            "previous_receipt": proof["closed_panel"],
        },
    )
    pins = read(initial / "G00-H0/EXECUTION_SPEC.json")["b2_code"]
    receiver = pin(
        root / "v18/receiver-preview02/student_grip.so", reg["receiver_sha256"]
    )
    results = []
    try:
        for slot in first["unstarted"]:
            if slot.endswith("H0"):
                v = run_slot(root, dest, slot, reg, rows, pins, receiver)
            else:
                if (
                    not results[0].get("completed")
                    or results[0]["outcome"] != "H0_PASS"
                ):
                    break
                lock, _ = site(root)
                with lock.open("rb") as gpu:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _, info = site(root)
                    record(dest / "CARD_GPU_START.json", info)
                    os.environ["REACH_CARD_GPU_FD"] = str(gpu.fileno())
                    try:
                        v = run_slot(root, dest, slot, reg, rows, pins, receiver)
                    finally:
                        del os.environ["REACH_CARD_GPU_FD"]
                        record(
                            dest / "CARD_GPU_END.json",
                            {"worker_returned_or_reaped": True},
                        )
            require(
                read(dest / slot / "control/LEASE.json")["status"] == "CLOSED",
                "world lease still open",
            )
            v = {
                **v,
                "slot": slot,
                "outcome": read(dest / slot / "RESULT.private.json")["outcome"],
            }
            results.append(v)
            print(slot, v["outcome"], v["ticks"], flush=True)
            if not v.get("completed"):
                break
    finally:
        record(
            dest / "PANEL_CLOSE.json",
            {"results": results, "unstarted": first["unstarted"][len(results) :]},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", type=Path)
    group.add_argument("--execute", type=Path)
    main(parser.parse_args())
