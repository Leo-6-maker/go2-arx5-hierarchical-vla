"""Seal the V5 legacy readmission in four-episode increments and verify on each site."""

import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import data_scope


def verify(archive, actions, state, snapshot):
    from scripts.pi0292.durable import write

    manifest = read(snapshot / "MANIFEST.json")
    stats = {"actions": read(actions)["actions"], "state": read(state)["state"]}
    dataset = data_scope(archive, snapshot, stats)
    bounds = {}
    complete = 0
    phases = Counter()
    for index, (ep, offset) in enumerate(dataset.rows):
        first, _ = bounds.setdefault(ep["id"], (index, index))
        bounds[ep["id"]] = first, index
        mask = dataset.support(index)
        require(mask[:10].all() and not np.any(np.diff(mask.astype(np.int8)) > 0),
                "A native command mask lost its contiguous consumed prefix")
        complete += int(mask.all())
        phases[ep["expert_stage_audit_only"][offset]] += 1
    for first, last in bounds.values():
        for index in (first, last):
            item = dataset[index]
            valid = dataset.support(index)
            require(item["state"].shape == (32,) and item["actions"].shape == (40, 32)
                    and np.isfinite(item["actions"]).all() and not item["actions"][~valid].any(),
                    "Real training reader returned an invalid or unmasked target")
    require(len(bounds) == len(manifest["episodes"]), "The training loader did not read every signed member")
    proof = dict(episodes=len(bounds), windows=len(dataset), complete_H40=complete,
                 read_images_at_fixed_episode_boundaries=2 * len(bounds), phase_windows=dict(phases),
                 manifest_sha256=sha(snapshot / "MANIFEST.json"), action_statistics_sha256=sha(actions),
                 state_statistics_sha256=sha(state), actual_optimizer_visits=0,
                 native_integrations=0, model_forwards=0, receiver_ready=True)
    write(snapshot / "READY.json", proof)
    return proof


def seal(archive, actions, state, old, candidate, out):
    from scripts.pi0292.durable import OwnerLock, write
    from contextlib import closing

    manifest = read(candidate / "CANDIDATE_MANIFEST.json")
    actual = read(candidate / "ACTUAL.json")
    require(actual["admitted"] == len(manifest["episodes"]) and actual["original_admitted"] == 52
            and actual["actual_training_visits"] == actual["new_physics_ticks"] == 0, "Unbound re-admission")
    out.mkdir(mode=0o700)
    atoms = [r for r in manifest["episodes"] if r["origin"] == "LEGACY_V4"]
    extras = [r for r in manifest["episodes"] if r["origin"] == "NEW_LEGACY_READMISSION"]
    require(len(atoms) == 52 and len(extras) == actual["recovered"], "Incorrect legacy additions")
    with closing(OwnerLock(out / "OWNER.lock")):
        milestones = list(range(4, len(extras), 4)) + [len(extras)]
        for step in milestones:
            rows = atoms + extras[:step]
            label = "TRAIN-LEGACY-%03d" % len(rows)
            pending = out / ("." + label + str(os.getpid()))
            pending.mkdir(mode=0o700)
            for row in rows:
                folder = old if row["origin"] == "LEGACY_V4" else candidate
                name = Path(row["labels"])
                require(name.name == str(name), "Labels must be locally referenced")
                os.link(pin(folder / name, row["sha256"]), pending / name)
            write(pending / "MANIFEST.json", dict(manifest, episodes=rows,
                  core_readmission_sha256=sha(candidate / "ACTUAL.json"), sealed_epoch=time.time()))
            proof = verify(archive, actions, state, pending)
            pending.rename(out / label)
            print(label, proof["windows"], proof["complete_H40"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", action="store_true")
    cli.add_argument("archive", type=Path)
    cli.add_argument("actions", type=Path)
    cli.add_argument("state", type=Path)
    cli.add_argument("paths", type=Path, nargs="+")
    a = cli.parse_args()
    if a.verify:
        snapshot, = a.paths
        print(verify(a.archive, a.actions, a.state, snapshot))
    else:
        old, candidate, output = a.paths
        seal(a.archive, a.actions, a.state, old, candidate, output)
