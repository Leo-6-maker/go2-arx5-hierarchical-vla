"""Recover exact Teacher-phase exposure from the completed R4 training receipts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


LEDGER = "5eaa17e34056a73747a72b2fc19f54ffb5e2e8fa45683d33b8433171bdf06282"


def audit(archive, legacy, bank, ledger, output):
    from scripts.pi0292.durable import write

    controls = read(legacy)
    allowed = {r["episode"] for r in controls["episodes"] if r["origin"] == "LEGACY_V4"}
    original = read(pin(archive / "MANIFEST.private.json", controls["signed_original_manifest_sha256"]))
    episodes = [r for r in original["episodes"] if r["id"] in allowed]
    phases = [f'{e["task"].split("-")[0]}:{phase}' for e in episodes for phase in e["expert_stage_audit_only"]]
    require(len(episodes) == 52 and len(phases) == 4450, "Original learner index order changed")
    bank_manifest = read(bank / "ACTUAL.json")
    with np.load(pin(bank / "BANK.npz", bank_manifest["archive_sha256"]), allow_pickle=False) as data:
        indices, slots = data["target_indices"], data["auxiliary_slots"]
    actual = list(map(json.loads, pin(ledger, LEDGER).read_text().splitlines()))
    require(len(actual) == 200, "The archived optimizer interval has changed")
    target, auxiliary = Counter(), Counter()
    for i, row in enumerate(actual):
        visit = row["components"]["target"]
        aux = row["components"]["auxiliary"]
        samples = indices[i]
        require(row["step"] == i + 1 and visit["visits"] == len(samples) == 64
                and hashlib.sha256(samples.tobytes()).hexdigest() == visit["visit_sha256"]
                and aux["visits"] == len(aux["slots"]) == 4 and slots[i].tolist() == aux["slots"],
                "Actual optimizer access and frozen sample bank do not match")
        target.update(phases[k] for k in samples)
        auxiliary.update(phases[k] for k in samples[slots[i]])
    write(output, dict(actual_go2_FM_visits=dict(target), actual_go2_4_slot_aux_visits=dict(auxiliary),
          optimizer_updates_used=len(actual), verified_target_hashes_and_aux_slots=len(actual),
          source_pbs_job="735699.hopper-m-02", update_ledger_sha256=LEDGER, new_optimizer_updates=0))
    print(json.dumps(read(output)))
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "legacy", "bank", "ledger", "output"):
        parser.add_argument(name, type=Path)
    audit(**vars(parser.parse_args()))
