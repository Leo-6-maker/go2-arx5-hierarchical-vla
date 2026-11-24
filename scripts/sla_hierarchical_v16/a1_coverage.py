"""Run or audit the separately allocated #477 eligible-rank cohort."""

import argparse
import itertools
import os
import sys
from contextlib import closing
from pathlib import Path

from demonstrated_grip import pin, read, require
from known_dynamic_owner import CAP, execute, preflight
from moving_stop_card import BASE, baseline, check_occupancy, configure, record, sha

REPORT = BASE / "reports/SLA_GO2_V17"
CANDIDATE = REPORT / "DYNAMIC_472/A1_COVERAGE_REQUEST.json"
PARENT = REPORT / "DYNAMIC_472/ALLOCATION.json"
SLOTS = tuple(f"A1C-{i:02}" for i in range(1, 9))
CONDITIONS = ("SLOW_A1", "SOURCE_SPEED_A1")
CAMPAIGN = "known-dynamic-a1-477"


def decision(matched, previous):
    return CONDITIONS[previous % len(CONDITIONS)] if matched else None


class Coverage:
    slots = SLOTS

    def assign(self, case, match):
        initial = case / "INITIAL.private.json"
        require(read(initial)["match"] == match, "initial gate must be sealed first")
        rank = 0
        for name in self.slots[: self.slots.index(case.name)]:
            previous = case.parent / name
            old = read(previous / "ASSIGNMENT.json")
            require(
                read(previous / "control/LEASE.json")["status"] == "CLOSED",
                "prior lease open",
            )
            require(
                old["condition"] == decision(old["matched"], rank),
                "prior assignment mismatch",
            )
            rank += int(old["matched"])
        matched = match["matched"]
        condition = decision(matched, rank)
        record(
            case / "ASSIGNMENT.json",
            {
                "slot": case.name,
                "initial_sha256": sha(initial),
                "matched": matched,
                "previous_eligible": rank,
                "eligible_rank": rank + 1 if matched else None,
                "condition": condition,
            },
        )
        return condition


def rehearsal(root):
    candidate = read(CANDIDATE)
    original = read(PARENT)
    require(
        candidate["reset_slots"] == list(SLOTS)
        and candidate["conditions"] == list(CONDITIONS)
        and candidate["maximum_total_ticks"] == len(SLOTS) * CAP
        and candidate["maximum_ticks_per_start"] == CAP,
        "registered roster",
    )
    prepared, cfg, controls = preflight(root)
    require(sha(prepared[2]) == original["receiver_sha256"], "changed receiver")
    require(
        read(
            pin(
                root / "v17/panel-final-preflight.json",
                original["cpu_dynamic_preview_sha256"],
            )
        )["controls"]
        == controls,
        "parent CPU preview changed",
    )
    for bits in itertools.product((False, True), repeat=len(SLOTS)):
        ranks = [decision(b, sum(bits[:i])) for i, b in enumerate(bits)]
        expected = iter(itertools.cycle(CONDITIONS))
        require(
            all(v == (next(expected) if b else None) for v, b in zip(ranks, bits)),
            "assignment order",
        )
    return (
        prepared,
        cfg,
        {
            "issue": 477,
            "original_assignment_patterns_checked": 2 ** len(SLOTS),
            "roster": list(SLOTS),
            "candidate_sha256": sha(CANDIDATE),
            "receiver_sha256": sha(prepared[2]),
            "original_cpu_preview_sha256": original["cpu_dynamic_preview_sha256"],
            "parent_close_sha256": sha(root / "v17/known-dynamic-472/PANEL_CLOSE.json"),
            "physics_ticks": 0,
        },
    )


def launch(root, allocation, prepared, cfg):
    from d1_matched_owner import with_case_lease

    from scripts.pi0292.durable import OwnerLock

    base, pins, receiver, _, _ = prepared
    dest = root / "v17" / CAMPAIGN
    dest.mkdir(mode=0o700, exist_ok=False)
    record(dest / "REGISTRATION.json", allocation)
    os.sched_setaffinity(0, base["cpus"])
    rows = []
    coverage = Coverage()
    try:
        for slot in SLOTS:
            case = dest / slot
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
                    "scope": "#477_A1_STANDING_SIM_ONLY",
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
                    lambda spec, sp: execute(
                        root, sp, spec, prepared, cfg, coverage=coverage
                    ),
                )
            row = read(case / "AUDIT.json")
            require(
                read(case / "control/LEASE.json")["status"] == "CLOSED",
                "lease remains active",
            )
            rows.append(row)
            print(slot, row["outcome"], row["ticks"], row["clean"], flush=True)
            if not row["clean"]:
                break
    finally:
        record(
            dest / "PANEL_CLOSE.json",
            {"completed_case_audits": rows, "scientific_roster": list(SLOTS)},
        )


def audit(root):
    from known_dynamic_readout import score

    dest = root / "v17" / CAMPAIGN
    closed = read(dest / "PANEL_CLOSE.json")
    require(closed["scientific_roster"] == list(SLOTS), "unregistered slots")
    cases, eligible = [], 0
    for slot in SLOTS:
        case = dest / slot
        if not (case / "NATIVE_REQUEST.json").is_file():
            require(not (case / "RESULT.private.json").exists(), "orphan result")
            continue
        assign = (
            read(case / "ASSIGNMENT.json")
            if (case / "ASSIGNMENT.json").is_file()
            else None
        )
        if assign is not None:
            initial = read(pin(case / "INITIAL.private.json", assign["initial_sha256"]))
            matched = initial["match"]["matched"]
            require(
                assign["slot"] == slot
                and assign["matched"] == matched
                and assign["previous_eligible"] == eligible
                and assign["eligible_rank"] == (eligible + 1 if matched else None)
                and assign["condition"] == decision(matched, eligible),
                "independent rank audit",
            )
            eligible += int(matched)
        outcome = read(case / "RESULT.private.json")["outcome"]
        require(
            outcome != "DYNAMIC_COMPLETE" or assign and assign["matched"],
            "orphan dynamic result",
        )
        row = score(case, condition=assign["condition"] if assign else None)
        row["assignment"] = assign
        row["loaded_executables"] = {
            k: v["executable_sha256"]
            for k, v in read(
                next(case.glob("native-*/LOADED_EXECUTOR_00.private.json"))
            ).items()
            if k in ("sim", "solver", "controller")
        }
        baseline_exec = read(REPORT / "DYNAMIC_472/FROZEN_EXECUTORS.json")[
            "sha256_by_card"
        ]["SLOW_A0"]
        require(row["loaded_executables"] == baseline_exec, "frozen executor mismatch")
        cases.append(row)
    require(
        [r["audit"] for r in cases] == closed["completed_case_audits"],
        "panel ledger mismatch",
    )
    return {
        "registered": len(SLOTS),
        "started": len(cases),
        "eligible": eligible,
        "native_ticks": sum(r["audit"]["ticks"] for r in cases),
        "case_audits": cases,
        "unstarted": list(SLOTS[len(cases) :]),
        "model_forwards": 0,
        "training_updates": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "execute", "audit"):
        group.add_argument("--" + mode, type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    os.umask(0o077)
    if args.audit:
        record(args.audit, audit(root))
        return
    prepared, cfg, receipt = rehearsal(root)
    if args.preflight:
        record(args.preflight, receipt)
        return
    allocation = read(args.execute)
    require(
        allocation["issue"] == 477
        and allocation["issued"] is True
        and allocation["slots"] == list(SLOTS)
        and allocation["ticks"] == len(SLOTS) * CAP
        and allocation["per_card_cap"] == CAP
        and allocation["forwards"] == allocation["updates"] == 0
        and allocation["candidate_sha256"] == sha(CANDIDATE)
        and allocation["receiver_sha256"] == sha(prepared[2])
        and read(
            pin(root / "v17/a1-477-preflight.json", allocation["preflight_sha256"])
        )
        == receipt,
        "scientific allocation or preflight mismatch",
    )
    launch(root, allocation, prepared, cfg)


if __name__ == "__main__":
    main()
