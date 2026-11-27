"""Freeze the public-ARX5 part of the SOURCE608 schedule from the real TRAIN manifest."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from v6_dual_course import balanced_source


CATALOGUE = "165f388dabc4f6d5ecd147f709c1749ee726daa820a828dc560ef985f936877f"
PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
EPISODES = {"robodojo": 499, "ville": 58, "sc_vla": 51}
NATIVE = {"robodojo": (30, 50), "ville": (20, 40), "sc_vla": (15, 30)}
WIDTH, STEPS, SEED = 32, 5000, 527_202612
PUBLIC_RECEIPT = "fb09287c6f20579bea1972ca1a2376a91860a8408bf48a79d490d7abaf52a3fc"
V61_REPLAY_RECEIPT = "1ba0092a0c982c51e785849b86332bfc9c3db636bad6a82cab97548bed027891"
JOINT = {"target": 16, "ville": 32, "sc_vla": 32, "robodojo": 32, "libero": 16}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_pinned(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Declared dataset manifest changed: {path.name}")
    return json.loads(raw)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def split_index(document):
    records = document["episodes"]
    if (document["train_episodes"] != len(records) or document["episodes_by_domain"] != EPISODES
            or not sum(r["windows"] for r in records) == document["train_windows"] == 544_694
            or not document["independent_source_dev_preserved"]):
        raise ValueError("The fixed 608 population or independent holdout changed")
    versions = {}
    for row in document["versions"]:
        manifest = row["manifest"]
        sha = manifest["sha256"]
        if sha in versions:
            raise ValueError("The catalogue repeats one admitted version")
        body = read_pinned(Path(manifest["path"]), sha)
        role = row["domain"]
        rate = body["nominal_frequency_Hz"] if role == "robodojo" else body["native_hz"]
        if (rate, body["horizon"]) != NATIVE[role]:
            raise ValueError("A source release differs from the actual native clock contract")
        versions[sha] = (row, body)
    pools = {domain: [] for domain in EPISODES}
    for entry in records:
        row, manifest = versions[entry["version"]]
        role = entry["domain"]
        if row["domain"] != role:
            raise ValueError("A train episode references a different source codec")
        if role == "robodojo":
            originals = [x for x in manifest["episodes"] if x["id"] == entry["episode"]
                         and x["raw_sha256"] == entry["raw_sha256"] and x["valid"]]
        else:
            originals = [x for x in manifest["episodes"] if x["raw_numeric_sha256"] == entry["numeric_sha256"]]
        if len(originals) != 1 or originals[0]["split"] != "TRAIN" or originals[0]["windows"] != entry["windows"]:
            raise ValueError("A scheduled episode is not the exact admitted TRAIN member")
        pools[role].append(dict(entry, reader_episode=originals[0].get("id", originals[0].get("episode")),
                               normalization_sha256=row["normalized_by"]))
    for role, rows in pools.items():
        identities = {(r["version"], r["reader_episode"]) for r in rows}
        if not len(identities) == len(rows) == EPISODES[role]:
            raise ValueError("Duplicate or missing exact TRAIN episodes")
    return pools


def build(catalogue, output):
    pools = split_index(read_pinned(catalogue, CATALOGUE))
    rng = np.random.default_rng(SEED)
    output.mkdir(mode=0o700)
    manifest = output / "INDEX.json"
    write(manifest, pools)
    arrays, summary = {}, {}
    for role, rows in pools.items():
        counts = np.asarray([r["windows"] for r in rows], dtype=np.int64)
        cumulative = np.cumsum(counts)

        class Pool:
            episodes = {i: row for i, row in enumerate(rows)}

            def __len__(self):
                return int(cumulative[-1])

        chosen = balanced_source(Pool(), STEPS, WIDTH, rng)
        episode = np.searchsorted(cumulative, chosen, side="right")
        anchors = chosen - np.r_[0, cumulative[:-1]][episode]
        if (anchors < 0).any() or (anchors >= counts[episode]).any():
            raise ValueError("Scheduled SOURCE608 window escaped its own episode")
        arrays[f"{role}_episode"] = episode.astype("<i4")
        arrays[f"{role}_anchor"] = anchors.astype("<i4")
        summary[role] = dict(episodes=len(rows), visits=chosen.size,
                             first_250_tasks=dict(Counter(rows[i]["task"] for i in episode[:250].ravel())),
                             nominal_hz=NATIVE[role][0], horizon=NATIVE[role][1])
    bank = output / "BANK.npz"
    with bank.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write(output / "ACTUAL.json", dict(plan=PLAN, catalogue_sha256=CATALOGUE, seed=SEED,
          steps=STEPS, samples_per_source_per_step=WIDTH, source_only=True, provenance=summary,
          episode_index_sha256=digest(manifest), bank_sha256=digest(bank), formal_optimizer_visits=0,
          candidate_requires_full_go2_libero_binding=True))


def bind_replay(public, replay, output):
    source = read_pinned(public / "ACTUAL.json", PUBLIC_RECEIPT)
    prior = read_pinned(replay / "ACTUAL.json", V61_REPLAY_RECEIPT)
    if (source["steps"] != STEPS or source["samples_per_source_per_step"] != WIDTH
            or sum(JOINT.values()) != 128 or prior["optimizer_visits"] != 0):
        raise ValueError("Use the exact five-domain candidate and sealed replay")
    public_index = public / "INDEX.json"
    read_pinned(public_index, source["episode_index_sha256"])
    prepared = (public / "BANK.npz", replay / "BANK.npz")
    expected = (source["bank_sha256"], prior["bank_file_sha256"])
    if tuple(digest(path) for path in prepared) != expected:
        raise ValueError("An immutable visit bank has changed")
    with np.load(prepared[0], allow_pickle=False) as data:
        indices = {f"{role}_{field}": data[f"{role}_{field}"] for role in EPISODES
                   for field in ("episode", "anchor")}
    with np.load(prepared[1], allow_pickle=False) as data:
        indices.update(target=data["target"], libero=data["libero"], auxiliary_slots=data["auxiliary_slots"])
    shapes = {f"{role}_{field}": (STEPS, WIDTH) for role in EPISODES for field in ("episode", "anchor")}
    shapes.update(target=(STEPS, 16), libero=(STEPS, 16), auxiliary_slots=(STEPS, 4))
    if any(arr.shape != shapes[name] or arr.dtype.kind not in "iu" or arr.min() < 0
           for name, arr in indices.items()):
        raise ValueError("SOURCE608 candidate index shapes or visit range differ")
    slots = indices["auxiliary_slots"]
    if slots.max() >= JOINT["target"]:
        raise ValueError("An H40 auxiliary slot escaped its replay row")
    aux = np.take_along_axis(indices["target"], slots, axis=1)
    if np.any(np.diff(np.sort(aux, axis=1), axis=1) <= 0):
        raise ValueError("Each row must retain four distinct actual H40 visits")
    output.mkdir(mode=0o700)
    bank = output / "BANK.npz"
    with bank.open("xb") as stream:
        np.savez_compressed(stream, **indices)
    write(output / "ACTUAL.json", dict(plan=PLAN, scope="SOURCE608_JOINT_INDEX_CANDIDATE", steps=STEPS,
          effective_batch=JOINT, actual_h40_auxiliary_slots=4, replay_scope="SEALED_V61_GO2_LIBERO",
          public_receipt_sha256=PUBLIC_RECEIPT, replay_receipt_sha256=V61_REPLAY_RECEIPT,
          replay_reader_order_sha256={role: prior["reader_order_sha256"][role] for role in ("target", "libero")},
          underlying_bank_sha256=list(expected), public_episode_index_sha256=digest(public_index),
          joint_bank_sha256=digest(bank), scheduled_target_unique_first250=int(np.unique(indices["target"][:250]).size),
          optimizer_visits=0, launch_admitted=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bind-replay", type=Path)
    args = parser.parse_args()
    if args.bind_replay:
        bind_replay(args.source, args.bind_replay, args.output)
    else:
        build(args.source, args.output)
