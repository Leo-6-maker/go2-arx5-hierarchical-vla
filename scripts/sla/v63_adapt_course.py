"""Deterministic A74 / ADAPT-A337 training input and event-aware bank."""

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import PINS, index_identity, source_identity, target_identity
from v61_source_course import SINGLE_CURSOR, read_bank, source_readers
from v63_adapt_data import PLAN, read_added
from v6_dual_course import DualSourcePool, four_domain_adamw, initialize_single, reduced_fm


END = 3000
SEED = 526_202611
PEAK = 8.481070829156122e-7
BATCH = dict(target=64, single=16, robodojo=32, libero=16)
MICRO = dict(target=32, single=16, robodojo=32, libero=16)
AUX = .25
EVENTS = frozenset(("CARRY", "LOWER", "OPEN", "WITHDRAW"))
RELEASE = frozenset(("OPEN", "WITHDRAW"))
# (FM width, explicitly scheduled release, explicitly scheduled carry/lower)
TARGET = {"base": (32, 4, 4), "legacy": (24, 2, 2), "newest": (8, 2, 2)}


def additions(root, expected):
    manifest = read(pin(root / "MANIFEST.json", expected))
    require(manifest["plan"] == PLAN and manifest["corpus"] == "A74"
            and manifest["inherited"] == 52 and manifest["added"] == len(manifest["episodes"]) == 22
            and Counter(row["population"] for row in manifest["episodes"]) == {"legacy": 17, "newest": 5},
            "The fixed A74 population changed")
    return manifest


def seal(archive, corrected, extra, extra_sha, old_bank, out):
    fixed = read(pin(corrected / "MANIFEST.json", PINS["target_corrected"]))
    chosen = {row["episode"]: row for row in fixed["episodes"]}
    inherited = read(pin(archive, PINS["target_archive"]))
    episodes, masks, population = [], [], {}
    for e in inherited["episodes"]:
        if e["id"] in chosen:
            row = chosen[e["id"]]
            with np.load(pin(corrected / row["labels"], row["sha256"]), allow_pickle=False) as z:
                masks.extend(z["valid"])
            episodes.append(e)
            population[e["id"]] = "base"
    require(len(episodes) == 52 and len(masks) == 4450, "Original base index changed")
    for row in additions(extra, extra_sha)["episodes"]:
        e, correction = row["episode"], row["corrected"]
        with np.load(pin(extra / correction["file"], correction["sha256"]), allow_pickle=False) as z:
            masks.extend(z["valid"])
        episodes.append(e)
        population[e["id"]] = row["population"]
    rows = [(e, i) for e in episodes for i in range(e["windows"])]
    require(len(episodes) == len({r["id"] for r in episodes}) == 74 and len(rows) == len(masks),
            "A74 index includes duplicate or missing rows")
    full = np.asarray(masks, dtype=bool).all(axis=1)
    rng = np.random.default_rng(SEED)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    phases, ids, groups = [], [], []
    for i, (e, anchor) in enumerate(rows):
        phase = e["expert_stage_audit_only"][anchor]
        is_place = e["task"].startswith("PLACE-")
        group = population[e["id"]]
        phases.append(phase)
        ids.append(e["id"])
        groups.append(group)
        grouped[group, "any"][e["task"]][e["id"]].append(i)
        if is_place and phase in EVENTS:
            kind = "release" if phase in RELEASE else "transport"
            grouped[group, kind][e["task"]][e["id"]].append(i)

    def draw(group, kind, selected, n):
        tasks = grouped[group, kind]
        require(tasks, f"Registered {group}/{kind} event bucket has no real data")
        rotation = rng.permutation(sorted(tasks)).tolist()
        for k in range(n):
            task = rotation[k % len(rotation)]
            candidates = [(eid, [i for i in indices if i not in selected])
                          for eid, indices in tasks[task].items()]
            candidates = [(eid, candidates) for eid, candidates in candidates if candidates]
            if not candidates:
                # Event buckets are filled before the generic slots.
                raise ValueError(f"Cannot fill {group}/{kind} without repeats")
            _, choices = candidates[int(rng.integers(len(candidates)))]
            selected.add(int(rng.choice(choices)))

    target = np.empty((END, BATCH["target"]), dtype=np.int64)
    aux = np.empty((END,4), dtype=np.int64)
    for step in range(END):
        slots = []
        for group, (width, release, carry) in TARGET.items():
            selected = set()
            draw(group, "release", selected, release)
            draw(group, "transport", selected, carry)
            draw(group, "any", selected, width-len(selected))
            slots.extend(sorted(selected))
        target[step] = rng.permutation(slots)
        pool = {group: np.asarray([i for i, idx in enumerate(target[step])
                    if groups[idx] == group and full[idx]], dtype=np.int64) for group in TARGET}
        require(len(pool["base"]) >= 2 and len(pool["legacy"]) and len(pool["newest"]),
                "Expected two old and two new real complete H40 labels")
        aux[step] = [*rng.choice(pool["base"], size=2, replace=False),
                     rng.choice(pool["legacy"]), rng.choice(pool["newest"])]
    original, source, _, _ = read_bank(old_bank)
    dual_old = np.stack([rng.choice(row[:40],size=16,replace=False) for row in source["robodojo"][:END]])
    dual_new = np.stack([rng.choice(row[40:],size=16,replace=False) for row in source["robodojo"][:END]])
    banks = dict(target=target, single=source["single"][:END], robodojo=np.concatenate((dual_old,dual_new),axis=1),
                 libero=source["libero"][:END])
    step = np.arange(1, END+1)
    rates = PEAK * np.where(step <= 100, step / 100, .1+.45*(1+np.cos(np.pi*(step-100)/(END-100))))
    out.mkdir(mode=0o700)
    bank = out / "BANK.npz"
    np.savez(bank, **banks, auxiliary_slots=aux, rates=rates)
    identities = dict(original["reader_order_sha256"])
    identities["target"] = index_identity((ep["id"], anchor) for ep, anchor in rows)
    record = dict(plan=PLAN, population="A74", added_manifest_sha256=extra_sha,
                  bank_file_sha256=sha(bank), reader_order_sha256=identities, domain_batch=BATCH, micro=MICRO,
                  aux_coefficient=AUX, peak_lr=PEAK, final_lr=float(rates[-1]), seed=SEED, updates=END,
                  target_group_slots={key: value[0] for key,value in TARGET.items()}, release_slots_min=8,
                  transport_slots_min=8, optimizer_visits=0, source_337_ancestor=sha(old_bank / "ACTUAL.json"),
                  first250_target_episodes=dict(Counter(ids[i] for i in target[:250].ravel())),
                  first250_target_phase=dict(Counter(phases[i] for i in target[:250].ravel())),
                  first250_target_unique_windows=len(set(target[:250].ravel())))
    write(out / "ACTUAL.json", record)
    print(sha(out / "ACTUAL.json"), flush=True)


class Target:
    def __init__(self, original, extension, original_aux):
        self.original, self.extension, self.original_aux = original, extension, original_aux
        self.rows = original.rows + extension.rows
        self.statistics = original.statistics

    def __len__(self):
        return len(self.rows)

    def member(self, index):
        if index < len(self.original):
            return self.original, index
        return self.extension, index-len(self.original)

    def __getitem__(self, index):
        reader, i = self.member(index)
        return reader[i]

    def support(self, index):
        reader, i = self.member(index)
        return reader.support(i)

    def aux(self, indices):
        result = []
        for idx in indices:
            i = int(idx)
            if i < len(self.original):
                prior, action = self.original_aux([i])
                result.append((prior[0], action[0]))
            else:
                result.append(self.extension.jaw(i-len(self.original)))
        return tuple(np.asarray(part, dtype=np.float32) for part in zip(*result, strict=True))


def bind(paths, extra, bank, bank_sha):
    from openpi_client.image_tools import resize_with_pad

    record = read(pin(bank / "ACTUAL.json", bank_sha))
    manifest = additions(extra, record["added_manifest_sha256"])
    require(record["plan"] == PLAN and record["domain_batch"] == BATCH and record["aux_coefficient"] == AUX,
            "Unexpected A74 bank or reduction")
    with np.load(pin(bank / "BANK.npz", record["bank_file_sha256"]), allow_pickle=False) as z:
        banks = {key: z[key] for key in BATCH}
        slots, rates = z["auxiliary_slots"], z["rates"]
    require(all(v.shape == (END, BATCH[k]) and v.dtype == np.int64 for k,v in banks.items())
            and slots.shape == (END,4) and rates.shape == (END,) and np.isfinite(rates).all()
            and rates[99] == PEAK and rates[-1] == .1*PEAK, "A74 bank lost its shape or schedule")
    keys = ("custody", "short", "prepared", "corrected", "extension", "single")
    original, single, old, new, libero, tokenizer, original_aux = source_readers(*(Path(paths[k]) for k in keys))
    ref_root = Path(paths["short"]) / "incoming/reference"
    statistics = {}
    for key, name, receipt_key in (("state", "STATE.json", "state_statistics"),
                                   ("actions", "ACTION.json", "action_statistics")):
        statistics[key] = read(pin(ref_root / name, manifest["statistics_sha256"][receipt_key]))[key]
    added = read_added(extra, manifest, statistics, resize_with_pad)
    target = Target(original, added, original_aux)
    identities = dict(target=target_identity(target), single=source_identity(single), dual_old=source_identity(old),
                      dual_new=source_identity(new), libero=source_identity(libero))
    pool = DualSourcePool(old,new)
    datasets = dict(target=target, single=single, robodojo=pool, libero=libero)
    require(identities == record["reader_order_sha256"]
            and all(0 <= v.min() < v.max() < len(datasets[k]) for k,v in banks.items())
            and np.all(banks["robodojo"][:,:16] < pool.base_length)
            and np.all(banks["robodojo"][:,16:] >= pool.base_length), "Live reader differs from sealed 337 source schedule")
    full = np.asarray([target.support(i).all() for i in range(len(target))])
    selected = np.take_along_axis(banks["target"], slots, axis=1)
    group_map = {row["episode"]["id"]:row["population"] for row in manifest["episodes"]}
    group = np.asarray([("base" if ep["id"] not in group_map else group_map[ep["id"]]) for ep,_ in target.rows])
    require(full[selected].all() and np.all(group[selected] == ["base","base","legacy","newest"]) and
            np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
            and all(np.all((group[banks["target"]]==k).sum(axis=1)==v[0]) for k,v in TARGET.items()),
            "Actual physical auxiliary or per-update Go2 balance changed")

    def factory(graphs, parameters, frozen, rng):
        from v4_long_engine import LongLearner

        graphs = dict(graphs, single=graphs["target"])
        parameters = dict(parameters, single=initialize_single(parameters["target"]))
        return LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=datasets,
                           tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=target.aux,
                           cursors=dict(target=0, single=SINGLE_CURSOR, robodojo=0, libero=0), rates=rates,
                           rng=rng, batch_by_domain=BATCH, gradient_join=partial(reduced_fm,batch=BATCH,aux_weight=AUX),
                           domain_roles=dict(target="target",single="target",robodojo="robodojo",libero="libero"),
                           support_domains=("target","single"),optimizer_factory=four_domain_adamw)

    return factory, datasets, tokenizer, banks, slots, record, group


def host_readback(paths, extra, bank, bank_sha, output):
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch

    _, datasets, tokenizer, banks, slots, record, group = bind(read(paths),extra,bank,bank_sha)
    shapes = {}
    targets = datasets["target"]
    for role, dataset in datasets.items():
        key = "target" if role == "single" else role
        value = prepare_host_batch(dataset, tokenizer, key, range(BATCH[role]), banks[role][0])
        require(np.isfinite(value["actions"]).all(), "One actual training sample is nonfinite")
        shapes[role] = list(value["actions"].shape)
    p, w = targets.aux(banks["target"][0,slots[0]])
    require(p.shape == (4,) and w.shape == (4,40) and np.isfinite(w).all(), "Four actual auxiliary inputs required")
    write(output, dict(plan=PLAN, bank_sha256=bank_sha, added_manifest_sha256=record["added_manifest_sha256"],
          reader_population=dict(Counter(group.tolist())), target_windows=len(targets), actual_batch_shapes=shapes,
          auxiliary_shape=list(w.shape), optimizer_visits=0, model_forwards=0, native_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal","readback"))
    parser.add_argument("paths", nargs="+", type=Path)
    cli = parser.parse_args()
    args = cli.paths
    if cli.command == "seal" and len(args) == 6:
        seal(args[0],args[1],args[2],str(args[3]),args[4],args[5])
    elif cli.command == "readback" and len(args) == 5:
        paths, extra, bank, digest, output = args
        host_readback(paths,extra,bank,str(digest),output)
    else:
        parser.error("seal needs 6 arguments and readback needs 5")
