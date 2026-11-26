"""Register disjoint public RoboDojo additions and verify raw windows in waves."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from v6_robodojo_pilot import CAMERAS, REPO, REV
from v6_ville_pilot import need, write


TASKS = (
    "cover_blocks", "insert_tubes", "make_bread",
    "make_food", "pack_and_pour_fruit", "store_in_safe",
)
SOURCE_PREFIX = "data/RoboDojo_real/"
R200_CORE_SHA = "5a558e07034f2f683a26705585716760930e73acca1c396898ef6a01c3b25e71"
GIB = 1024**3
MAX_MEMBER = 320 * 1024**2


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def population(task):
    prefix = f"{SOURCE_PREFIX}{task}/arx_x5/data/"
    endpoint = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{prefix.rstrip('/')}?limit=1000"
    with urlopen(endpoint, timeout=40) as response:
        need(not response.headers.get("Link"), "Incomplete API population")
        rows = json.load(response)
    need(len(rows) == 100, "Exact task directory size changed")
    result = []
    for row in rows:
        member = row["path"]
        index = int(Path(member).stem.removeprefix("episode_"))
        need(member == f"{prefix}episode_{index:07d}.hdf5"
             and row["type"] == "file" and row["size"] == row["lfs"]["size"]
             and len(bytes.fromhex(row["lfs"]["oid"])) == 32, "Expected pinned LFS member")
        result.append(dict(task=task, index=index, member=member, bytes=row["size"], sha256=row["lfs"]["oid"]))
    need(len({x["sha256"] for x in result}) == 100
         and {x["index"] for x in result} == set(range(100)), "Ambiguous task members")
    return result


def rank(rows, seed):
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}/{row['member']}".encode()).hexdigest())


def roster(baseline_path, output):
    need(digest(baseline_path) == R200_CORE_SHA, "Existing Core R200 manifest changed")
    baseline = read(baseline_path)
    need(baseline["HF_revision"] == REV and baseline["TEST_labels_read"] is False
         and baseline["target_commands"] is False, "Existing source lineage changed")
    old = baseline["episodes"]
    need(Counter(x["split"] for x in old) == {"TRAIN": 200, "SOURCE_DEV": 12}
         and len({x["raw_sha256"] for x in old}) == 212, "Expected exact existing train/dev")
    protected = set(x["raw_sha256"] for x in old)
    existing_paths = set(x["public_archive"] for x in old if x.get("public_archive"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = list(pool.map(population, TASKS))
    listed = {row["sha256"] for rows in tasks for row in rows}
    need(len(listed) == 600 and protected <= listed, "Old raw identities absent from the pinned source")
    # Recover the exact original 60 candidates, including its unopened TEST
    # metadata, using the released selection seed and path hash ordering.
    historical = [row for rows in tasks for row in rank(rows, "2026111801")[:10]]
    need(len({x["sha256"] for x in historical}) == 60, "Ambiguous original 60")
    historical_test = [row for rows in tasks for row in rank(rows, "2026111801")[8:10]]
    historical_non_test = [row for rows in tasks for row in rank(rows, "2026111801")[:8]]
    need({x["sha256"] for x in historical_non_test} <= protected
         and not protected.intersection(x["sha256"] for x in historical_test),
         "Original TEST split reconstruction disagrees with the legacy manifest")
    protected.update(row["sha256"] for row in historical)
    protected_paths = existing_paths | {row["member"] for row in historical}
    per_task = []
    for rows in tasks:
        eligible = [x for x in rows if x["sha256"] not in protected and x["member"] not in protected_paths
                    and 0 < x["bytes"] <= MAX_MEMBER]
        need(len(eligible) >= 18, "Not enough disjoint candidates")
        per_task.append(rank(eligible, "SLA-V6-RD-20261126-01")[:18])
    candidates = [row for group in zip(*per_task, strict=True) for row in group]
    need(len({x["sha256"] for x in candidates}) == 108, "Ambiguous V6 candidates")
    amount = sum(x["bytes"] for x in candidates)
    need(amount < 25 * GIB, "Raw stage reservation exceeds 25 GiB")
    write(output, dict(repo=REPO, revision=REV, license="Apache-2.0",
                       baseline_sha256=R200_CORE_SHA, old_train=200, old_dev=12,
                       protected_test_members=historical_test, protected_unique_members=len(protected),
                       raw_bytes=amount, candidates=candidates, admitted=0))


def actual_episode(row, raw_dir):
    from go2arx5_vla.sla_adapter.source_native_v6 import load_numeric, native_rgb
    import h5py
    import numpy as np

    endpoint = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row['member']}"
    path = raw_dir / f"{row['task']}-{row['index']:07d}.hdf5"
    partial = path.with_suffix(".unverified.hdf5")
    need(not path.exists() and not partial.exists(), "Requires independent reconciliation of existing bytes")
    actual = 0
    hasher = hashlib.sha256()
    with urlopen(endpoint, timeout=90) as response, partial.open("xb") as sink:
        declared = response.headers.get("Content-Length")
        need(declared is None or int(declared) == row["bytes"], "Pinned raw member length changed")
        for block in iter(lambda: response.read(1024**2), b""):
            actual += len(block)
            need(actual <= row["bytes"], "Unexpected excess member bytes")
            hasher.update(block)
            sink.write(block)
    need(actual == row["bytes"] and hasher.hexdigest() == row["sha256"], "Raw byte seal failed")
    partial.rename(path)
    states, actions, instruction, audit = load_numeric(path)
    need(actions.shape == (len(states), 50, 14)
         and audit["labels"] == "FUTURE_LOGGED_STATE_REFERENCE"
         and all(x["SO3_max_rad"] <= 1e-6 for x in audit["roundtrip"]),
         "Canonical source window changed")
    feature = hashlib.sha256()
    anchor_summary = []
    with h5py.File(path) as source:
        for anchor in (0, len(states) // 2, len(states) - 1):
            observed = {}
            for camera in CAMERAS:
                image = native_rgb(source[f"vision/{camera}/colors"][anchor])
                need(image.shape == (480, 640, 3) and image.dtype == np.uint8,
                     "Existing reader RGB ABI changed")
                observed[camera] = hashlib.sha256(image.tobytes()).hexdigest()
            anchor_summary.append(dict(index=anchor, rgb=observed))
    feature.update(states.tobytes())
    feature.update(actions.tobytes())
    return dict(**row, raw_file=str(path), windows=len(states), instruction=instruction,
                canonical=audit, rgb=anchor_summary, numeric_sha256=feature.hexdigest(),
                split="CANDIDATE", ready_for_training=False)


def wave(roster_path, expected_sha, start, count, output, *, actual_capacity=False):
    need(digest(roster_path) == expected_sha, "Roster bytes changed")
    roster_data = read(roster_path)
    need(roster_data["repo"] == REPO and roster_data["revision"] == REV
         and roster_data["baseline_sha256"] == R200_CORE_SHA
         and roster_data["license"] == "Apache-2.0", "Roster identity changed")
    candidates = roster_data["candidates"]
    need(0 <= start < len(candidates) and 1 <= count <= 24 and start + count <= len(candidates),
         "Expected one bounded wave")
    batch = candidates[start:start + count]
    distinct = {x["sha256"] for x in batch}
    need(len(distinct) == count
         and not distinct.intersection(x["sha256"] for x in roster_data["protected_test_members"]),
         "Protected or duplicate episode in the wave")
    reserved = sum(x["bytes"] for x in batch)
    overhead = 3 * MAX_MEMBER if actual_capacity else 60 * GIB
    need(shutil.disk_usage(output.parent).free > reserved + overhead,
         "Storage headroom unavailable for this wave")
    output.mkdir(mode=0o700)
    raw_dir = output / "raw"
    raw_dir.mkdir(mode=0o700)
    write(output / "START.json", dict(roster_sha256=expected_sha, start=start, count=count,
                                      reserved_raw_bytes=reserved))

    def candidate(row):
        try:
            proof = actual_episode(row, raw_dir)
        except Exception as err:
            proof = dict(**row, error=f"{type(err).__name__}: {err}", ready_for_training=False)
        write(output / f"{row['task']}-{row['index']:07d}.json", proof)
        return proof

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(candidate, batch))
    passed = [x for x in results if "canonical" in x]
    write(output / "ACTUAL.json", dict(registered=count, codec_pass=len(passed),
        windows=sum(x["windows"] for x in passed), physical_requests=0,
        optimizer_updates=0, newly_admitted_train=0, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    register = command.add_parser("roster")
    register.add_argument("baseline", type=Path)
    register.add_argument("output", type=Path)
    run = command.add_parser("wave")
    run.add_argument("roster", type=Path)
    run.add_argument("sha")
    run.add_argument("start", type=int)
    run.add_argument("count", type=int)
    run.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "roster":
        roster(args.baseline, args.output)
    else:
        wave(args.roster, args.sha, args.start, args.count, args.output)
