"""Seal the Corelab ARX5 training inventory from independent, read-back releases."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from v61_source_bank import PINS
from v62_source_admission import V62
from v62_robodojo_first_wave import FIRST_RELEASE, FOURTH_RELEASE, SECOND_RELEASE, THIRD_RELEASE


VILLE_READER = "b643899475ff509419fbc730c87e4922cb003203ac0aaa8cbe3a463cca7bad6b"
VILLE_FILTERS = "fddab694e9248fe37530b963ec60d001e004412ac44561fe96b1dec145f135d8"
VILLE_FILTERS_READER = "acf428c8f0156b16b255aab67c3c30922aa564a45605a677d4382a28c7900a79"
R48_STATS = "c8afc74108b66b497122502e3b9b1d27cfc9565855fec88f4e01c3ebcb34f269"
BRIDGE = "69385b5cd515d7d63a69a1dd6602dbb2a45ab2b03d4046c3e3832315d7c5e9cf"
POST500 = "efdac32969f2fdb508920e4f191eb00d97d76c3c6eb8c5e55c41648e48297e6d"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def build(legacy, dual_extension, single, expansions, out, post500=None, ville_filters=None):
    from scripts.pi0292.durable import write

    require(len(expansions) == 5, "Use the four closed waves and the fixed Mix500 wave")
    legacy_blob = read(pin(legacy, PINS["dual_old"]))
    raw = set()
    canonical = set()
    episodes = []
    versions = []

    def dual_rows(blob, source, expected, ready=None):
        require(blob["HF_revision"] == legacy_blob["HF_revision"] and blob["target_commands"] is False
                and blob["nominal_frequency_Hz"] == 30 and blob["horizon"] == 50,
                "RoboDojo normalization or recorded action ABI changed")
        # Reserve every historical identity before adding another release.
        rows = blob["episodes"]
        accepted = [r for r in rows if r["split"] == "TRAIN" and r["valid"]]
        require(len(accepted) == expected and len({r["raw_sha256"] for r in rows}) == len(rows),
                "RoboDojo release population changed")
        new_raw = {r["raw_sha256"] for r in rows}
        fingerprints = {r["numeric_fingerprint"] for r in rows if "numeric_fingerprint" in r}
        require(not (raw & new_raw or canonical & fingerprints), "RoboDojo data overlaps a protected split or release")
        raw.update(new_raw)
        canonical.update(fingerprints)
        item = dict(manifest=ref(source), domain="robodojo", episodes=expected,
                    windows=sum(r["windows"] for r in accepted), normalized_by=R48_STATS)
        if ready:
            item["core_ready"] = ref(ready)
        versions.append(item)
        episodes.extend(dict(domain="robodojo", episode=r["id"], task=r["task"], windows=r["windows"],
                             raw_sha256=r["raw_sha256"], version=item["manifest"]["sha256"]) for r in accepted)

    dual_rows(legacy_blob, legacy, 200)
    pins = (PINS["dual_new"], FIRST_RELEASE, SECOND_RELEASE, THIRD_RELEASE, FOURTH_RELEASE, BRIDGE)
    counts = (107, 12, 18, 24, 30, 84)
    releases = (dual_extension, *expansions)
    if post500 is not None:
        releases += (post500,)
        pins += (POST500,)
        counts += (24,)
    for path, expected_sha, expected_count in zip(releases, pins, counts, strict=True):
        proof_file = path / "READY.json"
        proof = read(proof_file)
        digest = proof["manifest_sha256"]
        count = proof["newly_admitted_train"]
        require(digest == expected_sha and count == expected_count and proof["core_readback"]
                and proof["full_H50_support"] and proof["readback_samples"] == count * 2
                and proof["inherited_statistics_sha256"] == R48_STATS, "An extension has no matching Core READY readback")
        manifest = pin(path / "MANIFEST.private.json", digest)
        dual_rows(read(manifest), manifest, count, proof_file)
        require(versions[-1]["windows"] == proof["windows"], "RoboDojo window count differs")

    ville_numeric = set()

    def ville_rows(path, expected_manifest, expected_readback, expected_count, task_key=None):
        manifest_file = pin(path / "MANIFEST.json", expected_manifest)
        manifest = read(manifest_file)
        readback_file = pin(path / "CORE_READER.json", expected_readback)
        held = read(readback_file)
        rows = [r for r in manifest["episodes"] if r["split"] == "TRAIN"]
        numeric = {r["raw_numeric_sha256"] for r in manifest["episodes"]}
        training = held["by_split"]["TRAIN"]
        require(held["manifest_sha256"] == expected_manifest and training["episodes"] == len(rows) == expected_count
                and training["windows"] == sum(row["windows"] for row in rows)
                and manifest["native_hz"] == 20 and manifest["horizon"] == 40 and manifest["target_command"] is False
                and len(numeric) == len(manifest["episodes"]) and not (numeric & ville_numeric),
                "Ville split or source contract changed")
        ville_numeric.update(numeric)
        versions.append(dict(manifest=ref(manifest_file), core_ready=ref(readback_file), domain="ville",
                             normalized_by=manifest["statistics"]["sha256"], episodes=expected_count,
                             windows=training["windows"]))
        for row in rows:
            episode = row["episode"] if task_key is None else f"{task_key}/{row['episode']}"
            episodes.append(dict(domain="ville", episode=episode, task=manifest["task"], windows=row["windows"],
                                 numeric_sha256=row["raw_numeric_sha256"], version=expected_manifest))

    ville_rows(single, PINS["single"], VILLE_READER, 30)
    if ville_filters is not None:
        ville_rows(ville_filters, VILLE_FILTERS, VILLE_FILTERS_READER, 8, "coffee_filters")
    require(len({(r["domain"], r["episode"]) for r in episodes}) == len(episodes), "Duplicate dataset episode identity")
    registry = dict(plan=V62, kind="ARX5_MIX500_CORE", mix500_achieved=len(episodes) >= 500,
                    train_episodes=len(episodes), train_windows=sum(r["windows"] for r in episodes),
                    episodes_by_domain=dict(Counter(r["domain"] for r in episodes)),
                    episodes_by_task=dict(Counter(r["task"] for r in episodes)), versions=versions, episodes=episodes,
                    hopper_readback=False, optimizer_visits=0, independent_source_dev_preserved=True)
    if post500 is not None or ville_filters is not None:
        registry.update(kind="ARX5_MIX2K_PROGRESS_CORE", mix2k_achieved=len(episodes) >= 2000)
    write(out, registry)
    print(sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "dual_extension", "single", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("expansions", type=Path, nargs=5)
    parser.add_argument("--post500", type=Path)
    parser.add_argument("--ville-filters", type=Path)
    build(**vars(parser.parse_args()))
