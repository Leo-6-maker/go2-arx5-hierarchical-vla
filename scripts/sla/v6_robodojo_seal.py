"""Seal read-back RoboDojo additions with the exact existing source codec and stats."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil

from v6_robodojo_intake import GIB, R200_CORE_SHA, digest, read
from v6_robodojo_pilot import REV
from v6_ville_pilot import need, write


R48_STATS = "c8afc74108b66b497122502e3b9b1d27cfc9565855fec88f4e01c3ebcb34f269"


def numeric_hash(states, actions):
    value = hashlib.sha256()
    value.update(states.tobytes())
    value.update(actions.tobytes())
    return value.hexdigest()


def seal(roster_path, roster_sha, baseline_path, stats_path, output, wave_pins):
    import numpy as np
    from go2arx5_vla.sla_adapter.source_native_v6 import (
        COORDINATES, HORIZON, SCHEMA, SourceDataset, anchor_strata, load_numeric,
    )
    from openpi_client.image_tools import resize_with_pad

    need(digest(roster_path) == roster_sha and digest(baseline_path) == R200_CORE_SHA
         and digest(stats_path) == R48_STATS, "An original or registered source input changed")
    roster = read(roster_path)
    old = read(baseline_path)
    candidate_map = {row["sha256"]: row for row in roster["candidates"]}
    protected = {row["raw_sha256"] for row in old["episodes"]}
    protected.update(row["sha256"] for row in roster["protected_test_members"])
    known_numeric = {row["numeric_fingerprint"] for row in old["episodes"] if "numeric_fingerprint" in row}
    need(len(candidate_map) == len(roster["candidates"]) and not protected.intersection(candidate_map)
         and roster["revision"] == old["HF_revision"] == REV,
         "Additions overlap old train/dev/test or revision")
    records = []
    seen = set()
    rejects = []
    for path, sha in wave_pins:
        need(digest(path) == sha, "Wave receipt changed")
        proof = read(path)
        for row in proof["results"]:
            raw_sha = row["sha256"]
            need(raw_sha in candidate_map and raw_sha not in seen
                 and all(row[key] == value for key, value in candidate_map[raw_sha].items()),
                 "Unregistered or duplicated scientific sample")
            seen.add(raw_sha)
            if "canonical" not in row:
                rejects.append(dict(sha256=raw_sha, reason=row["error"]))
                continue
            raw = Path(row["raw_file"])
            need(raw.resolve(strict=True).is_relative_to(path.parent.resolve(strict=True))
                 and digest(raw) == raw_sha and raw.stat().st_size == row["bytes"],
                 "Actual raw archive changed")
            states, actions, instruction, audit = load_numeric(raw)
            fingerprint = numeric_hash(states, actions)
            need(fingerprint == row["numeric_sha256"] and len(actions) == row["windows"]
                 and instruction == row["instruction"] and audit == row["canonical"],
                 "Existing codec no longer reproduces a real window")
            if fingerprint in known_numeric:
                rejects.append(dict(sha256=raw_sha, reason="EXACT_NUMERIC_DUPLICATE"))
                continue
            known_numeric.add(fingerprint)
            records.append((row, states, actions, fingerprint))

    need(records and not output.exists(), "Expected an unused, nonempty source seal")
    need(shutil.disk_usage(output.parent).free > 60 * GIB, "Source seal storage reserve unavailable")
    stage = output.with_name(output.name + ".staging")
    stage.mkdir(mode=0o700)
    (stage / "numeric").mkdir()
    statistics = read(stats_path)
    iqr = np.asarray(statistics["state"]["IQR"])
    expected_bytes = sum(states.nbytes + actions.nbytes for _, states, actions, _ in records)
    need(expected_bytes < 2 * GIB, "Dense archive exceeds registered compact storage plan")
    shutil.copyfile(stats_path, stage / "STATISTICS.json")
    need(digest(stage / "STATISTICS.json") == R48_STATS, "Inherited statistics copy corrupted")
    episodes = []
    for index, (row, states, actions, fingerprint) in enumerate(records):
        filename = f"numeric/{index:03d}.npz"
        arrays = stage / filename
        dynamic = anchor_strata(states, actions, iqr)
        delta = actions[:, -1].copy()
        delta[:, [6, 13]] -= states[:, [6, 13]]
        future = np.any(np.abs(delta) > .01 * iqr, axis=-1)
        with arrays.open("xb") as handle:
            np.savez(handle, state=states, actions=actions, dynamic=dynamic, future_span_dynamic=future)
        episodes.append(dict(id="V6-RD-" + row["sha256"], split="TRAIN", valid=True,
            task=row["task"], windows=row["windows"], instruction=row["instruction"],
            raw_path=row["raw_file"], public_archive=row["member"], raw_sha256=row["sha256"],
            numeric_fingerprint=fingerprint, arrays=dict(file=filename, sha256=digest(arrays))))
    manifest = stage / "MANIFEST.private.json"
    write(manifest, dict(schema=SCHEMA, recipe="SLA_V6_EXISTING_ROBODOJO_CODEC_EXTENSION",
           coordinates=COORDINATES, HF_revision=REV, horizon=HORIZON, nominal_frequency_Hz=30,
           statistics=dict(file="STATISTICS.json", sha256=R48_STATS), original_manifest_sha256=R200_CORE_SHA,
           test_labels_read=False, target_commands=False, raw_roster_sha256=roster_sha, episodes=episodes))

    manifest_sha = digest(manifest)
    reader = SourceDataset(stage, manifest_sha, split="TRAIN", resize_with_pad=resize_with_pad)
    n = 0
    for ep in episodes:
        for local in (0, ep["windows"] - 1):
            item = reader[n + local]
            need(item["actions"].shape == (50, 32) and item["state"].shape == (32,)
                 and np.isfinite(item["actions"]).all() and np.isfinite(item["state"]).all()
                 and not np.any(item["actions"][:, 14:]) and not np.any(item["state"][14:])
                 and all(item["image_mask"].values())
                 and all(x.shape == (224, 224) and x.any() and not x.all()
                         for x in item["image_padding_mask"].values()),
                 "Real source learner readback failed")
        n += ep["windows"]
    by_task = Counter(ep["task"] for ep in episodes)
    write(stage / "READY.json", dict(core_readback=True, hopper_readback=False, manifest_sha256=manifest_sha,
        inherited_statistics_sha256=R48_STATS, candidates_closed=len(seen), readback_samples=2 * len(episodes),
        newly_admitted_train=len(episodes), windows=n, by_task=dict(by_task), rejected=rejects,
        threshold_for_v6_source_met=len(episodes) >= 100 and len(by_task) >= 3,
        full_H50_support=True, training_visits=0, native_requests=0))
    stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster_path", type=Path)
    parser.add_argument("roster_sha")
    for name in ("baseline_path", "stats_path", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--wave", nargs=2, metavar=("ACTUAL", "SHA256"), action="append", required=True)
    args = parser.parse_args()
    seal(args.roster_path, args.roster_sha, args.baseline_path, args.stats_path, args.output,
         [(Path(path), sha) for path, sha in args.wave])
