"""Bind the explicit V6 dual-only fallback to existing Go2, RoboDojo and LIBERO readers."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SOURCE_BATCH = {"target": 16, "robodojo": 96, "libero": 16}
SOURCE_MICRO = {"target": 16, "robodojo": 32, "libero": 16}
SEED = 522_202611
TOTAL = 5000
PARENT = "502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361"


def reduced_fm(fm, auxiliary, *, tree_map):
    weights = {domain: size / 128 for domain, size in SOURCE_BATCH.items()}
    # V4 used an unnormalized FM sum of four. SOURCE retains one quarter
    # of its normalized auxiliary weight relative to the smaller Go2 share.
    aux_weight = 1 / 16
    shared = tree_map(lambda t, r, l, a: weights["target"] * t + weights["robodojo"] * r
        + weights["libero"] * l + aux_weight * a,
        fm["target"]["shared"], fm["robodojo"]["shared"], fm["libero"]["shared"], auxiliary["shared"])
    result = dict(shared=shared)
    for domain in SOURCE_BATCH:
        result[domain] = tree_map(lambda leaf: weights[domain] * leaf, fm[domain]["private"])
    result["target"] = tree_map(lambda t, a: t + aux_weight * a, result["target"], auxiliary["private"])
    return result


class DualSourcePool:
    """Keep immutable visits and explicit old/new provenance for a compatible codec."""

    def __init__(self, original, extension):
        self.members = (original, extension)
        self.base_length = len(original)
        if not self.base_length or not len(extension):
            raise ValueError("Both underlying source members must contain real data")
        old = {row["raw_sha256"] for row in original.episodes.values()}
        new = {row["raw_sha256"] for row in extension.episodes.values()}
        if old.intersection(new):
            raise ValueError("Dual-role source members overlap")
        for field in ("state", "actions"):
            for statistic in ("q01", "q99"):
                a, b = (getattr(reader.statistics[field], statistic) for reader in self.members)
                if not np.array_equal(a, b):
                    raise ValueError("Separate normalization requires a separate source codec")

    def __len__(self):
        return sum(len(x) for x in self.members)

    def __getitem__(self, index):
        if index < self.base_length:
            return self.members[0][index]
        return self.members[1][index - self.base_length]


def balanced_source(reader, count, width, rng):
    groups = defaultdict(list)
    first = 0
    for episode in reader.episodes.values():
        size = episode["windows"]
        groups[episode["task"]].append((first, size))
        first += size
    if first != len(reader) or not groups:
        raise ValueError("Reader index disagrees with source episode manifest")
    tasks = sorted(groups)
    assignment = np.resize(rng.permutation(len(tasks)), count * width)
    output = np.empty(count * width, dtype="<i8")
    for index, task in enumerate(tasks):
        locations = np.flatnonzero(assignment == index)
        episodes = np.asarray(groups[task], dtype=np.int64)
        picks = episodes[rng.integers(len(episodes), size=len(locations))]
        output[locations] = picks[:, 0] + rng.integers(picks[:, 1])
    return output.reshape(count, width)


def target_replay(reader, count, rng):
    from v3_main_consumed_data import PHASES
    from v3_main_place_learn import PLACEMENT_COUNTS

    groups = defaultdict(lambda: defaultdict(list))
    for index, (episode, offset) in enumerate(reader.rows):
        phase = "hold" if episode["task"].startswith("HOLD-") else PHASES[episode["expert_stage_audit_only"][offset]]
        groups[phase][episode["id"]].append(index)
    if set(groups) != set(PLACEMENT_COUNTS) | {"hold"}:
        raise ValueError("Frozen Go2 replay skills or phases are incomplete")
    frames = {k: list(v.values()) for k, v in groups.items()}
    place = list(PLACEMENT_COUNTS)
    weights = np.asarray(list(PLACEMENT_COUNTS.values()), dtype=float)
    weights /= weights.sum()
    support = np.asarray([reader.support(i).all() for i in range(len(reader))])
    width = SOURCE_BATCH["target"]
    rows = np.empty((count, width), dtype="<i8")
    aux = np.empty((count, 4), dtype="<i8")
    counts = Counter()
    for step, row in enumerate(rows):
        phases = ["hold"] * (6 + (step % 5 in (1, 3)))
        phases.extend(rng.choice(place, size=width - len(phases), p=weights))
        counts.update(phases)
        for j, phase in enumerate(phases):
            episodes = frames[phase]
            row[j] = rng.choice(episodes[rng.integers(len(episodes))])
        rng.shuffle(row)
        full = np.flatnonzero(support[row])
        if len(full) < 4:
            raise ValueError("A Go2 batch cannot supply four actual H40 auxiliary samples")
        aux[step] = np.sort(rng.choice(full, size=4, replace=False))
    return rows, aux, dict(counts)


def bind(custody, short_run, old_bank, extension_path, extension_manifest, *, count=TOTAL):
    from m38_gpu_preflight_v1 import pin, read, require, sha
    from v4_long_inputs import inputs
    from v4_long_engine import LongLearner
    from go2arx5_vla.sla_adapter.scale20k_data import RobodojoDataset
    from openpi_client.image_tools import resize_with_pad

    class ExpansionDataset(RobodojoDataset):
        RECIPE = "SLA_V6_EXISTING_ROBODOJO_CODEC_EXTENSION"

    if not 1 <= count <= TOTAL:
        raise ValueError("Expected a bounded new course")
    parent = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    parent_receipt = read(pin(parent / "COMPLETE.json", PARENT))
    datasets, tokenizer, _, _, auxiliary, _, old_rates = inputs(custody, short_run, old_bank, parent)
    release = read(extension_path / "READY.json")
    require(release["hopper_readback"] is True and release["threshold_for_v6_source_met"] is True
        and release["newly_admitted_train"] >= 100 and release["manifest_sha256"] == extension_manifest,
        "The source extension requires an independently read Hopper READY version")
    manifest = extension_path / "MANIFEST.private.json"
    require(sha(manifest) == extension_manifest, "The site source manifest changed")
    expanded = ExpansionDataset(extension_path, dict(path=str(manifest), sha256=extension_manifest),
        raw_binding=dict(path=str(extension_path / "RAW_BINDING.private.json"), sha256=release["raw_binding_sha256"]),
        split="TRAIN", resize_with_pad=resize_with_pad)
    require(len(expanded.episodes) == release["newly_admitted_train"] and len(expanded) == release["windows"],
        "The released source count differs from actual native source windows")

    rng = np.random.default_rng(SEED)
    old, new = (balanced_source(r, (count + 1) // 2, SOURCE_BATCH["robodojo"], rng)
               for r in (datasets["robodojo"], expanded))
    pool = DualSourcePool(datasets["robodojo"], expanded)
    source = np.empty((count, SOURCE_BATCH["robodojo"]), dtype="<i8")
    source[::2] = old
    source[1::2] = new[:count // 2] + pool.base_length
    datasets["robodojo"] = pool
    target, slots, phases = target_replay(datasets["target"], count, rng)
    lib_base = np.load(custody / "banks.verified/libero.npy", mmap_mode="r", allow_pickle=False)
    lib_offset = parent_receipt["progress"]["role_visits"]["libero"]
    lib = lib_base[lib_offset:lib_offset + count * SOURCE_BATCH["libero"]].reshape(count, SOURCE_BATCH["libero"])
    need_lib = int(lib.min()) >= 0 and int(lib.max()) < len(datasets["libero"])
    require(need_lib, "LIBERO replay bank outside TRAIN")
    banks = dict(target=target, robodojo=source, libero=lib)
    peak = float(old_rates.max()) * .5
    scale = np.arange(1, TOTAL + 1)
    rates = peak * np.where(scale <= 100, scale / 100,
        .1 + .45 * (1 + np.cos(np.pi * (scale - 100) / (TOTAL - 100))))
    settings = dict(recipe="V6_SOURCE_DUAL_ONLY_FALLBACK", phase_steps=count,
                    visits=dict(SOURCE_BATCH), fm_micro=dict(SOURCE_MICRO), aux_micro=4,
                    source_old_new_alternation="OLD_ON_STEP_ZERO", target_phases=phases,
                    source_extension_manifest=extension_manifest, rng_seed=SEED, peak_lr=peak,
                    fixed_cut_at=TOTAL, goal2k_v4_resumed=False)

    def make(graphs, parameters, frozen, jax_rng):
        return LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=datasets,
            tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary,
            cursors=dict.fromkeys(SOURCE_BATCH, 0), rates=rates[:count], rng=jax_rng,
            batch_by_domain=SOURCE_BATCH, gradient_join=reduced_fm)

    return make, datasets, banks, slots, auxiliary, settings
