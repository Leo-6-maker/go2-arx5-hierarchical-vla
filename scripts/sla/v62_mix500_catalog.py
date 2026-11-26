"""Seal the Corelab ARX5 training inventory from independent, read-back releases."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from v61_source_bank import PINS
from v62_source_admission import V62
from v62_robodojo_first_wave import FIRST_RELEASE, FOURTH_RELEASE, SECOND_RELEASE, THIRD_RELEASE


VILLE_READER = "b643899475ff509419fbc730c87e4922cb003203ac0aaa8cbe3a463cca7bad6b"
R48_STATS = "c8afc74108b66b497122502e3b9b1d27cfc9565855fec88f4e01c3ebcb34f269"
BRIDGE = "69385b5cd515d7d63a69a1dd6602dbb2a45ab2b03d4046c3e3832315d7c5e9cf"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def build(legacy, dual_extension, single, expansions, out):
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
    for path, expected_sha, expected_count in zip((dual_extension, *expansions), pins, counts, strict=True):
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

    single_file = pin(single / "MANIFEST.json", PINS["single"])
    single_manifest = read(single_file)
    single_readback = pin(single / "CORE_READER.json", VILLE_READER)
    held = read(single_readback)
    rows = [r for r in single_manifest["episodes"] if r["split"] == "TRAIN"]
    require(held["manifest_sha256"] == PINS["single"] and held["by_split"]["TRAIN"]["episodes"] == len(rows) == 30
            and held["by_split"]["TRAIN"]["windows"] == sum(row["windows"] for row in rows)
            and single_manifest["native_hz"] == 20 and single_manifest["horizon"] == 40
            and single_manifest["target_command"] is False
            and len({r["raw_numeric_sha256"] for r in single_manifest["episodes"]}) == len(single_manifest["episodes"]),
            "Ville split or source contract changed")
    versions.append(dict(manifest=ref(single_file), core_ready=ref(single_readback), domain="ville",
                         normalized_by=single_manifest["statistics"]["sha256"], episodes=30,
                         windows=held["by_split"]["TRAIN"]["windows"]))
    episodes.extend(dict(domain="ville", episode=r["episode"], task=single_manifest["task"], windows=r["windows"],
                         numeric_sha256=r["raw_numeric_sha256"], version=PINS["single"]) for r in rows)
    require(len({(r["domain"], r["episode"]) for r in episodes}) == len(episodes), "Duplicate dataset episode identity")
    write(out, dict(plan=V62, kind="ARX5_MIX500_CORE", mix500_achieved=len(episodes) >= 500,
                    train_episodes=len(episodes), train_windows=sum(r["windows"] for r in episodes),
                    episodes_by_domain=dict(Counter(r["domain"] for r in episodes)),
                    episodes_by_task=dict(Counter(r["task"] for r in episodes)), versions=versions, episodes=episodes,
                    hopper_readback=False, optimizer_visits=0, independent_source_dev_preserved=True))
    print(sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "dual_extension", "single", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("expansions", type=Path, nargs=5)
    build(**vars(parser.parse_args()))
