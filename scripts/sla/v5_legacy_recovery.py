"""Recover closed, uncounted legacy cases without changing a stale campaign ledger."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_consumed_data import PHASES, consume


def recover(campaign, admitted, out):
    from scripts.pi0292.durable import write

    campaign_lease = read(campaign / "control/LEASE.json")
    worker = read(campaign / "SUPERVISOR.private.json")
    current = read(campaign / "CURRENT_CASE.private.json")
    spec = read(pin(campaign / "EXECUTION_SPEC.json", campaign_lease["spec_sha256"]))
    ledger = campaign / "RECORDS.private.json"
    attempts = read(ledger)
    require(campaign_lease["deadline_epoch"] < time.time()
            and len(attempts) == spec["new_physical_requests_budget"]
            and all(not (Path("/proc") / str(entry["pid"])).exists() for entry in (worker, current)),
            "Predecessor is live or lacks a complete paid roster")
    origin = read(admitted / "MANIFEST.json")
    ids = {r["episode"] for r in origin["episodes"]}
    seen = set()
    out.mkdir(mode=0o700)
    raw = []
    passed = []
    for row in attempts:
        identity = row["condition"]
        require(identity not in seen and identity not in ids and row["requests"] == 1, "An old attempt would be duplicated")
        seen.add(identity)
        terminal = pin(row["result"]["path"], row["result"]["sha256"])
        case = terminal.parent
        result = read(terminal)
        require(case == campaign / "cases" / identity
                and read(case / "control/LEASE.json")["status"] == "CLOSED"
                and result["shutdown_clean"] and result["actual_native_integrations"] == row["ticks"],
                "Missing original terminal owner or integration denominator")
        raw.append(dict(id=identity, native_ticks=row["ticks"], result_sha256=row["result"]["sha256"],
                        original_admitted=row["admitted"], outcome=row["first_failure"]))
        if not row["admitted"]:
            continue
        package = row["episode"]
        file = pin(package["path"], package["sha256"])
        entry = read(file)
        require(entry["id"] == identity and entry["scene_sha256"] not in origin["excluded_dev_scenes"]
                and entry["split"] == package["split"] == "TRAIN" and entry["individual_pass"],
                "A recovered native success conflicts with the held-out population")
        archive = file.parent.parent
        for descriptor in (entry["labels"], *entry["frame_shards"]):
            pin(archive / descriptor["file"], descriptor["sha256"])
        arrays, proof = consume(case, entry, dict(native_result_sha256=row["result"]["sha256"],
                                  confirmed_P10_atoms=package["confirmed_P10_atoms"]), archive)
        require(set(arrays["phase"].tolist()) <= PHASES.keys(), "Unsupported task phase")
        target = out / (identity + ".npz")
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        passed.append(dict(proof, split="TRAIN", scene_group=entry["scene_sha256"], label=target.name,
                           label_sha256=sha(target), complete_H40=int(arrays["valid"].all(axis=1).sum()),
                           readmission_origin=package["sha256"], episode=entry))
    summary = dict(campaign_spec_sha256=campaign_lease["spec_sha256"], original_ledger_sha256=sha(ledger),
                   predecessor_lease_word=campaign_lease["status"], expired_and_processes_absent=True,
                   attempts=len(raw), actual_previous_ticks=sum(r["native_ticks"] for r in raw), rows=raw,
                   recovered=len(passed), recovered_windows=sum(r["windows"] for r in passed),
                   complete_H40=sum(r["complete_H40"] for r in passed),
                   recovered_tasks=dict(Counter(r["task"].split("-")[0] for r in passed)), admissions=passed,
                   new_native_requests=0, new_optimizer_updates=0, new_model_forwards=0)
    write(out / "ACTUAL.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "admitted", "out"):
        parser.add_argument(name, type=Path)
    result = recover(**vars(parser.parse_args()))
    print({k: v for k, v in result.items() if k not in ("rows", "admissions")})
