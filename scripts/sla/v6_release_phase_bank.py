"""Prepare a matched Teacher release-phase contrast without running the optimizer."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v6_reexpress_legacy_delta import ORIGINAL


STEPS = 200
SEED = 38002006
CHANGES = ("OPEN", "OPEN", "WITHDRAW", "WITHDRAW")
DONORS = {"LOWER", "CARRY"}


def build(archive, corrected, bank, output):
    from scripts.pi0292.durable import write

    canonical = read(pin(corrected / "MANIFEST.json", ORIGINAL))
    picked = {r["episode"]: r for r in canonical["episodes"]}
    seal_path = archive / "SEALED.private.json"
    seal = read(pin(seal_path, "5b97bd29ca3a4b32c4e16a64da3d7c1c3c6988f617fb3e6192305aed6428821f"))
    originals = read(pin(archive / "MANIFEST.private.json", seal["manifest_sha256"]))
    names, phases, supports, eligible = [], [], [], defaultdict(list)
    for episode in originals["episodes"]:
        identity = episode["id"]
        if identity not in picked:
            continue
        row = picked[identity]
        with np.load(pin(corrected / row["labels"], row["sha256"]), allow_pickle=False) as payload:
            phase, mask = payload["phase"], payload["valid"]
        require(episode["split"] == "TRAIN" and phase.tolist() == episode["expert_stage_audit_only"]
                and mask.shape == (len(phase), 40) and mask[:, :10].all(), "Original TRAIN phase or support differs")
        for state, valid in zip(phase, mask.all(axis=1), strict=True):
            index = len(names)
            names.append(identity)
            phases.append(state)
            supports.append(bool(valid))
            if valid and episode["task"].startswith("PLACE-") and state in CHANGES:
                eligible[identity, state].append(index)
    require(len(names) == 4450 and len(set(names)) == 52 and supports.count(True) == 4294,
            "Expected the R4 target row index and actual full H40 support")
    baseline = read(bank / "ACTUAL.json")
    with np.load(pin(bank / "BANK.npz", baseline["archive_sha256"]), allow_pickle=False) as source:
        original, auxiliary = source["target_indices"], source["auxiliary_slots"]
    require(original.shape == (2000, 64) and auxiliary.shape == (2000, 4), "Unexpected matched batch shape")
    visits = original.copy()
    rng = np.random.default_rng(SEED)
    replacements = []
    for update in range(STEPS):
        row = visits[update]
        reserved = set(auxiliary[update].tolist())
        for phase in CHANGES:
            for i, index in enumerate(row):
                if i in reserved or not supports[index] or phases[index] not in DONORS:
                    continue
                alternatives = [value for value in eligible.get((names[index], phase), []) if value not in row]
                if not alternatives:
                    continue
                chosen = int(rng.choice(alternatives))
                replacements.append(dict(step=update + 1, slot=i, episode=names[index], before=int(index),
                                         after=chosen, from_phase=phases[index], to_phase=phase))
                reserved.add(i)
                row[i] = chosen
                break
    require(np.array_equal(np.take_along_axis(original, auxiliary, axis=1), np.take_along_axis(visits, auxiliary, axis=1))
            and all(names[a] == names[b] and supports[a] == supports[b]
                    for a, b in zip(original.flat, visits.flat, strict=True)),
            "Contrast changed an episode, full-mask status or auxiliary input")
    output.mkdir(mode=0o700)
    path = output / "BANK.npz"
    np.savez(path, target_indices=visits, auxiliary_slots=auxiliary)
    write(output / "REPLACEMENTS.json", replacements)
    write(output / "ACTUAL.json", dict(optimizer_updates=0, proposed_updates=STEPS, target_per_step=64,
         aux_per_step=4, unchanged_auxiliary=True, seed=SEED, matched_episodes_per_slot=True,
         matched_full_mask_per_slot=True, shifted_visits=len(replacements),
         changed_updates=len({r["step"] for r in replacements}),
         shifted_from=dict(Counter(r["from_phase"] for r in replacements)),
         shifted_to=dict(Counter(r["to_phase"] for r in replacements)),
         original_bank_sha256=baseline["archive_sha256"],
         bank_sha256=sha(path), target_visit_bank_sha256=hashlib.sha256(visits.tobytes()).hexdigest(),
         replacement_sha256=sha(output / "REPLACEMENTS.json"), label_manifest_sha256=ORIGINAL))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "corrected", "bank", "output"):
        parser.add_argument(arg, type=Path)
    build(**vars(parser.parse_args()))
