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
VILLE_CAPSULES = "99b66bf9186318c2d825ac927e5725fe64782a865650634a7cab51fcbbd3b763"
VILLE_CAPSULES_READER = "91cb0766c8a71d379be08481d94ea527a0a9a692013700db66b71da685d839b6"
SC_BALL = "f7f799b785904c133b97f87f68f1766db039fdc247a2697037b694537e9cffb3"
SC_BALL_READER = "bcffa52031f536aac2896523b1ffe8c05906e93c69449158035031613d6c733e"
ADDITIONAL_SINGLE = {
    "ff5f9a358ec050860a342d0a3c8a2fe3ef9c5766df6ee378b852f21249bb8a74":
        ("e76a8c2bb9d0b0f93abec52559578116f1f52774b379c7e08302d05ffaa8f7f4", 7, "coffee_filters", "ville"),
    "710d3a5554b4e038118a50f516820cf2708a9ae88d06df8ecd5b84c1ed6be2ca":
        ("0693e6945c40b2e56f977101d91bd13ce644f7f15b343ca925f1d81ee0ddfe64", 6, "coffee_capsules", "ville"),
    "4b4860b9d1bac120cf8d4fff9e9cc2f625bcc18c4b2cc99756b10de86b968418":
        ("b70eab3f428f1528fe380983cc54b7fd2bad52a677b5e353ffa5bb3c489d30f9", 10, None, "sc_vla"),
    "f88612ffdb43cb85c8ebd3cc98db08f28119722f8d4b5488016f32a7a8e68b91":
        ("44a96a3fe774cf8edf9555b96dfa1d3272f9b46c8c92f86f0b904ec31e01a704", 10, None, "sc_vla"),
    "1ab80d48a6d64a091dc6cdb36e8ba622a75af837a6a3249b36f3aa87818483e3":
        ("d194cbf0789f3477ab02ec19ad1d84ecdf352133d51ce1f256c8677e4bc9f438", 9, None, "sc_vla"),
    "8307e8d7a1f1cca2996f0f3a416894a66e632b4973dfc27810e201ddb7786923":
        ("232ddc5b6815df58c5c41e92af574de8e26f69bf650e87faca7b14e3bfac0420", 4, None, "sc_vla"),
    "50075f8c28df38e686bc6fdc14d269893147e2e37b9463809b4a13fce96113e0":
        ("f4babb8dbac9607573eea73e488b4e1d0cd1d5dc223758812bc2941bd80a2500", 4, None, "sc_vla"),
    "48591ef65c646cc11ccd1491d1728e0784556431dbe9c2b508ac57f12101b787":
        ("36d1d37247da0410951f8884deae1359e26ed53135988077bfa43e0c206b04be", 4, None, "sc_vla"),
}
R48_STATS = "c8afc74108b66b497122502e3b9b1d27cfc9565855fec88f4e01c3ebcb34f269"
BRIDGE = "69385b5cd515d7d63a69a1dd6602dbb2a45ab2b03d4046c3e3832315d7c5e9cf"
POST500 = "efdac32969f2fdb508920e4f191eb00d97d76c3c6eb8c5e55c41648e48297e6d"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def build(legacy, dual_extension, single, expansions, out, post500=None, ville_filters=None, ville_capsules=None,
          sc_ball=None, additional_single=()):
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

    single_numeric = set()
    single_ids = set()

    def single_rows(path, expected_manifest, expected_readback, expected_count, task_key=None,
                    *, domain="ville", native_hz=20, horizon=40):
        manifest_file = pin(path / "MANIFEST.json", expected_manifest)
        manifest = read(manifest_file)
        readback_file = pin(path / "CORE_READER.json", expected_readback)
        held = read(readback_file)
        rows = [r for r in manifest["episodes"] if r["split"] == "TRAIN"]
        dev = [r for r in manifest["episodes"] if r["split"] == "SOURCE_DEV"]
        numeric = {r["raw_numeric_sha256"] for r in manifest["episodes"]}
        identities = {(domain, task_key, r["episode"]) for r in manifest["episodes"]}
        training = held["by_split"]["TRAIN"]
        validation = held["by_split"]["SOURCE_DEV"]
        require(held["manifest_sha256"] == expected_manifest and training["episodes"] == len(rows) == expected_count
                and training["windows"] == sum(row["windows"] for row in rows)
                and validation["episodes"] == len(dev) and validation["windows"] == sum(row["windows"] for row in dev)
                and training["read_image_pairs"] == 2 * len(rows) and validation["read_image_pairs"] == 2 * len(dev)
                and manifest["native_hz"] == held["actual_native_hz"] == native_hz
                and manifest["horizon"] == horizon and held["learner_shape"] == [horizon, 32]
                and manifest["target_command"] is False
                and len(numeric) == len(manifest["episodes"]) and not (numeric & single_numeric),
                "Single-arm split or source contract changed")
        require(len(identities) == len(manifest["episodes"]) and not (identities & single_ids),
                "An admitted single-arm episode crosses release or split boundaries")
        single_ids.update(identities)
        single_numeric.update(numeric)
        versions.append(dict(manifest=ref(manifest_file), core_ready=ref(readback_file), domain=domain,
                             normalized_by=manifest["statistics"]["sha256"], episodes=expected_count,
                             windows=training["windows"]))
        for row in rows:
            episode = row["episode"] if task_key is None else f"{task_key}/{row['episode']}"
            episodes.append(dict(domain=domain, episode=episode, task=manifest["task"], windows=row["windows"],
                                 numeric_sha256=row["raw_numeric_sha256"], version=expected_manifest))

    single_rows(single, PINS["single"], VILLE_READER, 30)
    if ville_filters is not None:
        single_rows(ville_filters, VILLE_FILTERS, VILLE_FILTERS_READER, 8, "coffee_filters")
    if ville_capsules is not None:
        single_rows(ville_capsules, VILLE_CAPSULES, VILLE_CAPSULES_READER, 7, "coffee_capsules")
    if sc_ball is not None:
        single_rows(sc_ball, SC_BALL, SC_BALL_READER, 10, domain="sc_vla", native_hz=15, horizon=30)
    for path in additional_single:
        manifest_pin = sha(path / "MANIFEST.json")
        require(manifest_pin in ADDITIONAL_SINGLE, "Single-arm release lacks a registered reader receipt")
        reader_pin, episodes_count, task_key, domain = ADDITIONAL_SINGLE[manifest_pin]
        hz = 15 if domain == "sc_vla" else 20
        single_rows(path, manifest_pin, reader_pin, episodes_count, task_key, domain=domain, native_hz=hz, horizon=2 * hz)
    require(len({(r["domain"], r["episode"]) for r in episodes}) == len(episodes), "Duplicate dataset episode identity")
    registry = dict(plan=V62, kind="ARX5_MIX500_CORE", mix500_achieved=len(episodes) >= 500,
                    train_episodes=len(episodes), train_windows=sum(r["windows"] for r in episodes),
                    episodes_by_domain=dict(Counter(r["domain"] for r in episodes)),
                    episodes_by_task=dict(Counter(r["task"] for r in episodes)), versions=versions, episodes=episodes,
                    hopper_readback=False, optimizer_visits=0, independent_source_dev_preserved=True)
    if additional_single or any(path is not None for path in (post500, ville_filters, ville_capsules, sc_ball)):
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
    parser.add_argument("--ville-capsules", type=Path)
    parser.add_argument("--sc-ball", type=Path)
    parser.add_argument("--additional-single", type=Path, action="append", default=[])
    build(**vars(parser.parse_args()))
