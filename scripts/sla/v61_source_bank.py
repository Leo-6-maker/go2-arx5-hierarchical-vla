"""Seal the V6.1 EARLY visit order on Corelab from the verified TRAIN populations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v6_dual_course import balanced_source, target_replay


PLAN = "221cc714f7648bb6f596745f9af1aaf1cae530d1"
SEED = 523_202611
UPDATES = 5000
BATCH = dict(target=16, single=16, robodojo=80, libero=16)
FM_WEIGHTS = {role: width / sum(BATCH.values()) for role, width in BATCH.items()}
AUX_COEFFICIENT = 1 / 16
PEAK_LR = 1.6962141658312244e-6
ENDPOINTS = (250, 500, 1000, 5000)
PINS = {
    "target_archive": "80876f04260377da07c20b547ac6c923751f5a84dbba8db6dbdb361ded164639",
    "target_commands": "ea8b99355b0a8fc2367d17b5e2f18e8a4d583c73ddb39b3848b1add25e143f96",
    "target_corrected": "1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21",
    "target_aux": "584fe14655538cbe22509cd27f5fc432026808d5c577c84b6ab9ed81d2e7838b",
    "dual_old": "5a558e07034f2f683a26705585716760930e73acca1c396898ef6a01c3b25e71",
    "dual_new": "fb33c0da28af0359b3b157d9288cd2b6fb3ac6c59d12d20f2e859a33f5a01317",
    "single": "6eb41d184fdd7296d2f117d7bc782e6549a409d1c87d64d1e732224cb1948609",
    "libero": "ba86ee2f88803fff79bda950ee54add91dcefaa105575ac22dade5d650b3fa8a",
}


class EpisodeIndex:
    def __init__(self, rows, identify):
        self.episodes = {str(row[identify]): row for row in rows}
        require(len(self.episodes) == len(rows), "Duplicate TRAIN episode in the reader order")
        self.size = sum(row["windows"] for row in rows)

    def __len__(self):
        return self.size


class TargetIndex:
    def __init__(self, archive, commands_path, corrected, aux_path):
        grouped = {r["episode"]: r for r in read(commands_path)["episodes"]}
        fixed = {r["episode"]: r for r in read(corrected)["episodes"]}
        widths = {r["episode"]: r for r in read(aux_path)["episodes"]}
        self.rows, self.masks = [], []
        for e in read(archive)["episodes"]:
            eid = e["id"]
            if eid not in grouped:
                continue
            source = grouped[eid]
            row = fixed[eid]
            require(e["split"] == "TRAIN" and row["source_sha256"] == source["sha256"]
                    and e["scene_sha256"] == source["scene"], "Target population changed")
            with np.load(pin(corrected.parent / row["labels"], row["sha256"]), allow_pickle=False) as data:
                mask = data["valid"]
                require(mask.shape == (e["windows"], 40) and mask[:, :10].all()
                        and list(data["phase"]) == e["expert_stage_audit_only"], "Target support or phase changed")
            aux = widths[eid]
            with np.load(pin(aux_path.parent / aux["file"], aux["sha256"]), allow_pickle=False) as data:
                require(np.array_equal(mask, data["valid"]) and np.isfinite(data["actual_width_m"][mask]).all()
                        and np.isfinite(data["prior_actual_width_m"]).all(), "True H40 auxiliaries changed")
            self.rows.extend((e, i) for i in range(len(mask)))
            self.masks.extend(mask)
        require(len(self.rows) == 4450 and len(grouped) == len(fixed) == len(widths) == 52,
                "Expected the initial 52 matched R4 Teacher episodes")

    def __len__(self):
        return len(self.rows)

    def support(self, index):
        return self.masks[index]


def index_identity(items):
    encoded = json.dumps(list(items), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_identity(reader):
    return index_identity((str(key), row["windows"] if "windows" in row else row["window_count"])
                          for key, row in reader.episodes.items())


def target_identity(reader):
    return index_identity((episode["id"], int(anchor)) for episode, anchor in reader.rows)


def build(manifests, output):
    from scripts.pi0292.durable import write

    paths = {key: pin(manifests[key], expected) for key, expected in PINS.items()}
    target = TargetIndex(*(paths[name] for name in ("target_archive", "target_commands", "target_corrected", "target_aux")))
    old_data, new_data, single_data = (read(paths[k]) for k in ("dual_old", "dual_new", "single"))
    old, new = (EpisodeIndex([r for r in source["episodes"] if r["split"] == "TRAIN" and r["valid"]], "id")
                for source in (old_data, new_data))
    single = EpisodeIndex([dict(r, task=single_data["task"]) for r in single_data["episodes"]
                           if r["split"] == "TRAIN"], "episode")
    libero = EpisodeIndex([dict(r, windows=r["window_count"]) for r in read(paths["libero"])["rows"]
                           if r["split"] == "TRAIN" and r["admitted"]], "episode")
    sizes = (len(old), len(new), len(single), len(libero))
    require(sizes == (210398, 115223, 3691, 209146)
            and tuple(len(i.episodes) for i in (old, new, single)) == (200, 107, 30),
            "A registered source episode or window disappeared")
    ids = [{r["raw_sha256"] for r in x.episodes.values()} for x in (old, new)]
    require(not ids[0] & ids[1] and len({r["task"] for r in new.episodes.values()}) == 6,
            "New RoboDojo source coverage was not preserved")
    rng = np.random.default_rng(SEED)
    target_bank, aux, phases = target_replay(target, UPDATES, rng, distinct_aux=True)
    single_bank = balanced_source(single, UPDATES, BATCH["single"], rng)
    old_bank = balanced_source(old, UPDATES, 40, rng)
    new_bank = balanced_source(new, UPDATES, 40, rng) + len(old)
    dual = np.concatenate((old_bank, new_bank), axis=1)
    libero_bank = balanced_source(libero, UPDATES, BATCH["libero"], rng)
    rates = np.arange(1, UPDATES + 1)
    rates = PEAK_LR * np.where(rates <= 100, rates / 100,
                             .1 + .45 * (1 + np.cos(np.pi * (rates - 100) / (UPDATES - 100))))
    require(np.isclose(rates[-1], .1 * PEAK_LR) and np.isclose(rates[99], PEAK_LR), "Unbound SOURCE schedule")
    banks = dict(target=target_bank, single=single_bank, robodojo=dual, libero=libero_bank)
    output.mkdir(mode=0o700)
    bank = output / "BANK.npz"
    np.savez(bank, **banks, auxiliary_slots=aux, rates=rates)
    identities = dict(target=target_identity(target), single=source_identity(single), dual_old=source_identity(old),
                      dual_new=source_identity(new), libero=source_identity(libero))
    prefix = {str(end): {k: int(v[:end].size) for k, v in banks.items()} | {"auxiliary": int(aux[:end].size)}
              for end in ENDPOINTS}
    write(output / "ACTUAL.json", dict(plan=PLAN, parents_ready=False, optimizer_visits=0,
          manifest_sha256=PINS, bank_file_sha256=sha(bank), reader_order_sha256=identities,
          domain_batch=BATCH, fm_weights=FM_WEIGHTS, auxiliary_coefficient=AUX_COEFFICIENT, peak_lr=PEAK_LR,
          workers=4, prefetch_batches=2, auxiliary_actual_distinct=4, numpy_rng_seed=SEED,
          endpoints=list(ENDPOINTS), declared_visits=prefix, dual_old_and_new_each_step=[40, 40],
          target_phase_counts=phases, dual_train_tasks=dict(Counter(row["task"] for row in new.episodes.values()))))
    print(sha(output / "ACTUAL.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PINS:
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    parser.add_argument("output", type=Path)
    args = vars(parser.parse_args())
    build({k: args[k] for k in PINS}, args["output"])
