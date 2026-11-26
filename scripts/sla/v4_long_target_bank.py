"""Prepare the exact old Go2 command visits for the V4 long-training fallback."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_consumed_data import PHASES
from v3_main_place_learn import PLACEMENT_COUNTS, data_scope


UPDATES, TARGET_BATCH, AUX_BATCH = 2000, 64, 4
SEED = 38002000


def old_target_bank(reader):
    pools = defaultdict(lambda: defaultdict(list))
    labels = []
    for index, (episode, offset) in enumerate(reader.rows):
        phase = ("hold" if episode["task"].startswith("HOLD-")
                 else PHASES[episode["expert_stage_audit_only"][offset]])
        pools[phase][episode["id"]].append(index)
        labels.append(phase)
    require(set(pools) == set(PLACEMENT_COUNTS) | {"hold"}, "Incomplete old target curriculum")

    rng = np.random.default_rng(SEED)
    hold_total = UPDATES * TARGET_BATCH * 2 // 5
    place_total = UPDATES * TARGET_BATCH - hold_total
    fractions = np.asarray(list(PLACEMENT_COUNTS.values()), dtype=float)
    exact = place_total * fractions / fractions.sum()
    counts = np.floor(exact).astype(int)
    remainder = place_total - counts.sum()
    counts[np.argsort(-(exact - counts), kind="stable")[:remainder]] += 1
    target = {**dict(zip(PLACEMENT_COUNTS, counts.tolist(), strict=True)), "hold": hold_total}

    def visit(group, count):
        episodes = list(group.values())
        return [int(rng.choice(episodes[j])) for j in rng.integers(len(episodes), size=count)]

    by_group = {phase: visit(episodes, target[phase]) for phase, episodes in pools.items()}
    hold = np.asarray(by_group.pop("hold"), dtype="<i8")
    place = np.asarray([i for group in by_group.values() for i in group], dtype="<i8")
    rng.shuffle(place)
    rng.shuffle(hold)
    support = np.asarray([reader.support(i).all() for i in range(len(reader.rows))])
    visits = np.empty((UPDATES, TARGET_BATCH), dtype="<i8")
    auxiliary = np.empty((UPDATES, AUX_BATCH), dtype="<i8")
    for k in range(UPDATES):
        h0 = k * hold_total // UPDATES
        h1 = (k + 1) * hold_total // UPDATES
        p0, p1 = k * TARGET_BATCH - h0, (k + 1) * TARGET_BATCH - h1
        row = visits[k]
        row[:] = np.concatenate((hold[h0:h1], place[p0:p1]))
        rng.shuffle(row)
        legal = np.flatnonzero(support[row])
        require(len(legal) >= AUX_BATCH, "No four fully consumed auxiliary labels in a logical batch")
        auxiliary[k] = np.sort(rng.choice(legal, size=AUX_BATCH, replace=False))

    require(Counter(labels[i] for i in visits.flat) == target
            and support[np.take_along_axis(visits, auxiliary, axis=1)].all(),
            "Realized visit or auxiliary support changed")
    return visits, auxiliary, target


def prepare(reference, originals, commands, out):
    baseline = read(reference / "CHECKPOINT.json")
    stats = dict(state=read(pin(reference / "STATE.json", baseline["state_sha256"]))["state"],
                 actions=read(pin(reference / "ACTION.json", baseline["action_sha256"]))["actions"])
    reader = data_scope(originals, commands, stats)
    visits, auxiliary, phases = old_target_bank(reader)
    out.mkdir(mode=0o700)
    np.savez(out / "BANK.npz", target_indices=visits, auxiliary_slots=auxiliary)
    proof = dict(candidate="OLD_DATA_ONLY_PENDING_FIRST_LONG_RUN_BINDING", seed=SEED, updates=UPDATES,
                 target_per_update=TARGET_BATCH, auxiliary_per_update=AUX_BATCH,
                 episode_count=len({episode["id"] for episode, _ in reader.rows}),
                 query_count=len(reader.rows), new_skill_visits=0, visits_by_phase=phases,
                 matchable_target_visits_per_endpoint=1000*TARGET_BATCH,
                 matchable_auxiliary_visits_per_endpoint=1000*AUX_BATCH,
                 target_visit_bank_sha256=hashlib.sha256(visits.tobytes()).hexdigest(),
                 aux_slot_bank_sha256=hashlib.sha256(auxiliary.tobytes()).hexdigest(),
                 archive_sha256=sha(out / "BANK.npz"), consumed_command_manifest_sha256=sha(commands / "MANIFEST.json"),
                 actual_optimizer_updates=0)
    (out / "ACTUAL.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "originals", "commands", "out"):
        parser.add_argument(name, type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
