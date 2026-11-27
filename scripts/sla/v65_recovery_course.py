"""Bind V6.5's fixed 1,000-step A93/R3 and source curriculum."""

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import source_identity, target_identity
from v61_source_course import read_bank, source_readers
from v63_adapt_course import Target, additions
from v63_adapt_data import read_added
from v64_a78_delta import A74, statistics
from v64_a87_increment import A83
from v64_go2_delta4 import A78
from v64_a93_increment import A87
from v65_recovery_data import A93, SCHEMA, dataset
from v6_dual_course import DualSourcePool, balanced_source, reduced_fm


PLAN = "d7f4c1357fd9a48a6f3612c52c63172195a1da60"
SEED, END = 655202611, 1000
PEAK, AUX = 8.481070829156122e-7, .375
BATCH = dict(target=96, robodojo=16, libero=16)
MICRO = dict(target=32, robodojo=16, libero=16)
PARTS = {"old": 48, "new": 46, "recovery": 2}
RELEASE = frozenset(("OPEN", "WITHDRAW"))
TRANSPORT = frozenset(("CARRY", "LOWER"))
VERSIONS = ((74, A74), (78, A78), (83, A83), (87, A87), (93, A93))


def readers(paths, roots, recovery, recovery_sha):
    from openpi_client.image_tools import resize_with_pad

    keys = ("custody", "short", "prepared", "corrected", "extension", "single")
    original, _, robo_old, robo_new, libero, tokenizer, original_aux = source_readers(*(Path(paths[k]) for k in keys))
    require(len(original) == 4450 and len({ep["id"] for ep, _ in original.rows}) == 52, "Expected original 52 Go2 episodes")
    root_stats = Path(paths["short"]) / "incoming/reference"
    target, earlier = original, original_aux
    first_stats = None
    for version, expected in VERSIONS:
        root = roots[version]
        data = additions(root, expected) if version == 74 else read(pin(root / "MANIFEST.json", expected))
        first_stats = first_stats or data["statistics_sha256"]
        require(first_stats == data["statistics_sha256"], "Cold Go2 increment changed R4 statistics")
        stats = statistics(data, root_stats / "STATE.json", root_stats / "ACTION.json")
        added = read_added(root, data, stats, resize_with_pad)
        target = Target(target, added, earlier)
        earlier = target.aux
    require(len(target) == 7916 and len({ep["id"] for ep, _ in target.rows}) == 93, "Incomplete A93 curriculum")
    manifest = read(pin(recovery / "MANIFEST.json", recovery_sha))
    require(manifest["schema"] == SCHEMA and manifest["a93_manifest_sha256"] == A93
            and manifest["statistics_sha256"] == first_stats, "Incompatible actual Recovery curriculum")
    suffixes = dataset(recovery, manifest, stats, resize_with_pad)
    require(len(suffixes) == 115, "Incomplete Recovery suffixes")
    complete = Target(target, suffixes, target.aux)
    observed = dict(target=target_identity(complete), robo_old=source_identity(robo_old),
                    robo_new=source_identity(robo_new), libero=source_identity(libero))
    return dict(target=complete, robodojo=DualSourcePool(robo_old, robo_new), libero=libero), tokenizer, observed


def demographics(target):
    episodes, phases, groups = [], [], []
    for index, (ep, anchor) in enumerate(target.rows):
        episodes.append(ep["id"])
        phases.append(ep["expert_stage_audit_only"][anchor])
        groups.append("old" if index < 4450 else "new" if index < 7916 else "recovery")
    return np.asarray(episodes), np.asarray(phases), np.asarray(groups)


def target_schedule(target, rng, *, start=0, end=END):
    episodes, phases, groups = demographics(target)
    masks = np.asarray([target.support(i).all() for i in range(len(target))])
    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    recovery = defaultdict(lambda: defaultdict(list))
    for i, (ep, _) in enumerate(target.rows):
        phase, group, name, task = phases[i], groups[i], episodes[i], ep["task"]
        if group == "recovery":
            recovery[name][phase].append(i)
            continue
        kinds = ["any"]
        if masks[i]:
            kinds.append("full")
        if task.startswith("HOLD-"):
            kinds.append("hold")
        if phase in RELEASE and task.startswith("PLACE-"):
            kinds.append("release")
        if phase in TRANSPORT and task.startswith("PLACE-"):
            kinds.append("transport")
        for kind in kinds:
            pools[group, kind][task][name].append(i)

    def sample(group, kind, count, selected):
        by_task = pools[group, kind]
        require(by_task, f"No actual {group}/{kind} window pool")
        order = rng.permutation(sorted(by_task))
        result = []
        for k in range(count):
            names = by_task[order[k % len(order)]]
            choices = [(name, [i for i in names[name] if i not in selected]) for name in rng.permutation(sorted(names))]
            _, candidates = next((p for p in choices if p[1]), (None, []))
            require(candidates, f"Cannot assign a distinct {group}/{kind} window")
            index = int(rng.choice(candidates))
            selected.add(index)
            result.append(index)
        return result

    rows, slots = [], []
    roster = sorted(recovery)
    require(len(roster) == 3 and len(target) == 8031, "Recovery strata changed")
    for step in range(start, end):
        row, selected = [], set()
        for group in ("old", "new"):
            before = len(row)
            for kind, count in (("full", 2), ("hold", 24 if group == "old" else 0), ("release", 4), ("transport", 4)):
                if count:
                    row.extend(sample(group, kind, count, selected))
            row.extend(sample(group, "any", PARTS[group] - (len(row) - before), selected))
        for shift in (0, 1):
            member = recovery[roster[(step+shift) % 3]]
            phase = sorted(member)[(step // 3 + shift) % len(member)]
            row.append(int(rng.choice(member[phase])))
        rows.append(row)
        slots.append([0, 1, 48, 49])
    return np.asarray(rows, dtype=np.int64), np.asarray(slots, dtype=np.int64)


def schedule(step):
    return PEAK * (step/50 if step <= 50 else .1 + .45*(1+np.cos(np.pi*(step-50)/950)))


def validate(datasets, banks, slots, rates, *, end=END, rate_schedule=schedule):
    for role, value in banks.items():
        require(value.shape == (end, BATCH[role]) and value.dtype == np.int64
                and 0 <= value.min() <= value.max() < len(datasets[role]), "Invalid actual reader index or batch width")
    target = datasets["target"]
    indices, robo = banks["target"], banks["robodojo"]
    episodes, phases, groups = demographics(target)
    selected = np.take_along_axis(indices, slots, axis=1)
    require(slots.shape == (end, 4) and slots.dtype == np.int64
            and (groups[selected] == ["old", "old", "new", "new"]).all()
            and all(target.support(int(i)).all() for i in np.unique(selected))
            and (np.diff(np.sort(indices, axis=1), axis=1) > 0).all()
            and all(((groups[indices] == role).sum(axis=1) == n).all() for role, n in PARTS.items())
            and (groups[indices[:, 94:]] == "recovery").all()
            and (episodes[indices[:, 94]] != episodes[indices[:, 95]]).all()
            and (np.isin(phases[indices[:, :94]], tuple(RELEASE)).sum(axis=1) >= 8).all()
            and (np.isin(phases[indices[:, :94]], tuple(TRANSPORT)).sum(axis=1) >= 8).all()
            and (robo[:, :8] < datasets["robodojo"].base_length).all()
            and (robo[:, 8:] >= datasets["robodojo"].base_length).all()
            and np.array_equal(rates, [rate_schedule(i) for i in range(1, end+1)]), "V6.5 all-row curriculum disagrees")
    return episodes, phases, groups


def courses(config):
    return readers(config["parent_paths"], {int(k): Path(v) for k, v in config["a93_roots"].items()},
                   Path(config["recovery"]), config["recovery_manifest_sha256"])


def seal(config, output):
    datasets, _, digests = courses(config)
    rng = np.random.default_rng(SEED)
    target, slots = target_schedule(datasets["target"], rng)
    dual = datasets["robodojo"]
    old, new = (balanced_source(reader, END, 8, rng) for reader in dual.members)
    previous, bank, _, _ = read_bank(Path(config["parent_paths"]["bank"]))
    require(previous["reader_order_sha256"]["libero"] == digests["libero"], "The original LIBERO order changed")
    banks = dict(target=target, robodojo=np.column_stack((old, new + dual.base_length)), libero=bank["libero"][:END])
    rates = np.asarray([schedule(i) for i in range(1, END+1)])
    episodes, phases, groups = validate(datasets, banks, slots, rates)
    output.mkdir(mode=0o700)
    bank_file = output / "BANK.npz"
    with bank_file.open("xb") as stream:
        np.savez_compressed(stream, **banks, auxiliary_slots=slots, rates=rates)
    write(output / "ACTUAL.json", dict(plan=PLAN, seed=SEED, reader_order=digests, batches=BATCH,
          aux_coefficient=AUX, bank_sha256=sha(bank_file), recovery_manifest_sha256=config["recovery_manifest_sha256"],
          go2_recovery_episode=dict(Counter(episodes[target[:, 94:]].ravel())),
          go2_recovery_phase=dict(Counter(phases[target[:, 94:]].ravel())),
          go2_populations=dict(Counter(groups.ravel())), actual_optimizer_visits=0))


def bind(config, root, expected):
    record = read(pin(root / "ACTUAL.json", expected))
    datasets, tokenizer, identity = courses(config)
    require(record["plan"] == PLAN and record["reader_order"] == identity
            and record["recovery_manifest_sha256"] == config["recovery_manifest_sha256"], "Sealed V6.5 reader identity changed")
    with np.load(pin(root / "BANK.npz", record["bank_sha256"]), allow_pickle=False) as file:
        banks = {role: file[role] for role in BATCH}
        slots, rates = file["auxiliary_slots"], file["rates"]
    demographics_ = validate(datasets, banks, slots, rates)
    return datasets, tokenizer, banks, slots, rates, demographics_


def host_readback(data, output, *, step=0, bank_sha256):
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch

    datasets, tokenizer, banks, slots, _, demographics_ = data
    shapes = {}
    for role, data in datasets.items():
        cursor = step * BATCH[role]
        batch = prepare_host_batch(data, tokenizer, role, range(cursor, cursor+BATCH[role]), banks[role][step])
        shapes[role] = list(batch["actions"].shape)
    targets = banks["target"][step]
    mask = np.asarray([datasets["target"].support(int(i)) for i in targets])
    prior, widths = datasets["target"].aux(targets[slots[step]])
    require(shapes == {"target": [96, 40, 32], "robodojo": [16, 50, 32], "libero": [16, 10, 32]}
            and widths.shape == (4, 40) and prior.shape == (4,) and np.isfinite(widths).all()
            and mask[-2:, :10].all(), "Real training input shapes or mask differ")
    write(output, dict(plan=PLAN, bank=bank_sha256, cursor=step, actual_shapes=shapes, auxiliary_shape=list(widths.shape),
                       recovery_episodes=demographics_[0][targets[-2:]].tolist(), optimizer_visits=0))


def learner(data, graphs, parameters, frozen, rng, *, trace=None):
    from v4_long_engine import LongLearner

    datasets, tokenizer, banks, slots, rates, _ = data
    return LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=datasets,
                       tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=datasets["target"].aux,
                       cursors=dict.fromkeys(BATCH, 0), rates=rates, rng=rng, batch_by_domain=BATCH,
                       gradient_join=partial(reduced_fm, batch=BATCH, aux_weight=AUX), domain_roles={k:k for k in BATCH}, trace=trace)


def host(config, root, expected, output):
    host_readback(bind(config, root, expected), output, bank_sha256=expected)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("config", type=Path)
    cli.add_argument("root", type=Path)
    cli.add_argument("--readback", type=Path)
    args = cli.parse_args()
    if args.readback:
        host(read(args.config), args.root, sha(args.root / "ACTUAL.json"), args.readback)
    else:
        seal(read(args.config), args.root)
