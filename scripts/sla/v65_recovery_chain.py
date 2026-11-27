"""Reconcile a finite 3K continuation boundary against the immutable update bank."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v65_recovery_course import BATCH, PARTS, PLAN, SEED
from v65_recovery_endpoint import VISITS
from v65_recovery_extension import BASE_SEAL, BASE_TERMINAL, COURSE, END, START


def ledger(run, bank, begin, end, *, settle):
    """Verify actual unique realized visit rows, without relying on the claimed counter."""
    require(START <= begin < end <= END, "Invalid V6.5 continuation range")
    with np.load(pin(bank.with_name("BANK.npz"), read(bank)["bank_sha256"]), allow_pickle=False) as data:
        batches = {role: data[role] for role in BATCH}
        slots, rates = data["auxiliary_slots"], data["rates"]
    recent, overhead = [], []
    with (run / "UPDATES.jsonl").open() as f:
        for step in range(begin+1, end+1):
            row = json.loads(f.readline())
            idx = step-1
            terms = row["components"]
            require(row["step"] == step and row["lr"] == rates[idx] and row["go2"] == PARTS
                    and row["robodojo_old"] == row["robodojo_expanded"] == 8
                    and len(row["recovery_ids"]) == len(set(row["recovery_ids"])) == 2
                    and sum(row["go2_phase"].values()) == 96
                    and row["global_gradient_norm"] > 0 and np.isfinite(row["global_gradient_norm"])
                    and row["seconds"] > 0 and np.isfinite(row["seconds"])
                    and {k:v["visits"] for k,v in terms.items()} == dict(BATCH, auxiliary=4)
                    and all(np.isfinite(v["loss"]) and np.isfinite(v["gradient_l2"]) and v["gradient_l2"] > 0
                            and np.isfinite(v["seconds"]) and v["seconds"] >= 0 for v in terms.values())
                    and terms["auxiliary"]["slots"] == slots[idx].tolist()
                    and all(terms[k]["visit_sha256"] == hashlib.sha256(v[idx].tobytes()).hexdigest()
                            for k,v in batches.items()), f"Step {step} does not match the sealed training recipe")
            recent.append(row["seconds"])
            overhead.append(row["seconds"] - sum(term["seconds"] for term in terms.values()))
        if settle:
            require(not f.readline(), "Terminal omits recorded optimizer updates")
    warm = recent[min(5, len(recent)-1):]
    residual = overhead[min(5, len(overhead)-1):]
    return dict(start=begin, end=end, updates=end-begin, actual_visits={k:n*(end-begin) for k,n in VISITS.items()},
                warm_p50_s=float(np.median(warm)), warm_p95_s=float(np.percentile(warm,95)),
                time_outside_domain_components_p50_s=float(np.median(residual)),
                warm_samples=len(warm))


def boundary(base, previous, bank):
    run = previous / "run"
    terminal = read(run / "TERMINAL.json")
    contract = read(pin(run / "RUN_CONTRACT.json", terminal["contract_sha256"]))
    binding = read(pin(previous / "BINDING.json", contract["binding_sha256"]))
    resume = read(run / "RESUME.json")
    allocation = read(previous / "START.json")
    start, end = contract["start"], terminal["formal_updates"]
    require(START <= start < end < END and contract["end"] == END
            and terminal["status"] == "SAVED_AT_FINITE_BOUNDARY" and terminal["segment_updates"] == end-start
            and not (previous / "STOP.json").exists()
            and (run / "TERMINAL.json").stat().st_mtime >= allocation["deadline"]-930
            and terminal["start"] == start and terminal["end"] == END and terminal["resumed_from_formal_1k"]
            and not terminal["engineering_updates"] and not contract["new_adam_from_r4"]
            and contract["plan"] == binding["plan"] == PLAN and contract["seed"] == SEED
            and contract["course"] == binding["course"] == COURSE
            and binding["base_capsule"] == str(base) and binding["bank"]["sha256"] == sha(bank)
            and binding["base_seal"] == BASE_SEAL and binding["base_terminal"] == BASE_TERMINAL
            and resume["step"] == start and resume["rng_restored"] and resume["bank_sha256"] == sha(bank)
            and resume["adam_counters"] == dict.fromkeys(("shared", *BATCH), start),
            "Only the same course's naturally sealed, exact finite boundary is eligible")
    previous_parent = binding.get("previous_boundary")
    if start == START:
        require(previous_parent is None and resume["parent_seal"] == contract["previous_seal"] == BASE_SEAL
                and contract["previous_terminal"] == BASE_TERMINAL,
                "First continuation predecessor changed")
    else:
        require(previous_parent["seal"] == resume["parent_seal"] == contract["previous_seal"]
                and previous_parent["terminal"] == contract["previous_terminal"] and previous_parent["step"] == start,
                "Predecessor of the finite segment changed")
    summary = ledger(run, bank, start, end, settle=True)
    seal = run / f"STEP{end:04d}_SEALED.json"
    checkpoint = read(seal)
    require(terminal["optimizer_visits"] == summary["actual_visits"] and terminal["checkpoint"] == checkpoint
            and checkpoint["bank_cursor"] == checkpoint["step"] == end
            and checkpoint["adam_counts"] == dict.fromkeys(("shared", *BATCH), end),
            "The immutable continuation did not seal all recorded state")
    return dict(path=str(previous), step=end, seal=sha(seal), terminal=sha(run / "TERMINAL.json")), checkpoint


class StepTiming:
    """Collect disjoint runtime spans already traced by the unchanged learner."""

    def __init__(self):
        self.seconds = defaultdict(float)

    @contextmanager
    def __call__(self, name):
        start = monotonic()
        try:
            yield
        finally:
            self.seconds[name] += monotonic() - start

    def take(self):
        result = dict(self.seconds)
        self.seconds.clear()
        return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "bank", "output"):
        cli.add_argument(name, type=Path)
    for name in ("begin", "end"):
        cli.add_argument(name, type=int)
    args = cli.parse_args()
    require(not args.output.exists(), "Preserve the first training ledger receipt")
    result = ledger(args.run, args.bank, args.begin, args.end, settle=False)
    write(args.output, result)
    print(json.dumps(result), flush=True)
