"""Seal and transfer the next Reach episodes downstream of A87."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v63_adapt_data import materialize
from v64_a78_delta import A74
from v64_a87_increment import A83, readback
from v64_go2_delta4 import A78


A87 = "12af4e2db606688074e056ef0ebca5e89f3b13d0649cf1c472bcafbce627a737"
WAVE2 = "376506d99de68ad342bd5ba5ebd8f40fe31251d5b98696a9fb526b094422fff2"
WAVE2_CASES = ("EF5_T_REACH_03", "EF5_T_REACH_04")
WAVE3_CASES = ("EF5_T_REACH_05", "EF5_T_REACH_06", "EF5_T_REACH_07", "EF5_T_REACH_08")


def closed_entries(root, expected, roster, statistics):
    batch = read(pin(root / "MANIFEST.json", expected))
    require(batch["schema"] == "V64_REACH_ORIENT_R4_ACTUAL_CONSUMED"
            and batch["statistics_sha256"] == statistics and batch["train"] == len(batch["episodes"]),
            "Require the original portable Reach label contract and statistics")
    wave = batch["wave"]
    path = Path(wave["path"])
    group = read(pin(path, wave["sha256"]))
    records = read(pin(path.parent / "RECORDS.private.json", batch["records"]["sha256"]))
    require(read(path.parent / "control/LEASE.json")["status"] == "CLOSED"
            and group["unreconciled"] is None and group["paid_requests"] == batch["paid"] == len(records)
            and records == group["records"] and tuple(x["condition"] for x in records) == roster[:len(records)],
            "Only the fixed closed native roster may supply training data")
    actual = {row["condition"]: row["result"]["sha256"] for row in records}
    skipped = {row["case"] for row in batch["rejected"]}
    require(len(actual) == len(batch["episodes"]) + len(skipped), "Retain all paid rejects in the denominator")
    for stored in batch["episodes"]:
        episode = stored["episode"]
        require(episode["split"] == "TRAIN" and episode["individual_pass"] and episode["id"] in actual
                and episode["id"] not in skipped and episode["task"].startswith("REACH-ORIENT-")
                and stored["native_result_sha256"] == actual[episode["id"]], "Require actual independent task admission")
    return batch["episodes"]


def build(work, prior, protected, original, wave3_sha, state, actions, out):
    prior_versions = [read(pin(prior / f"A{n}.json", digest))
                      for n, digest in ((74,A74), (78,A78), (83,A83), (87,A87))]
    base = prior_versions[-1]
    require(base["cumulative_target_episodes"] == 87 and base["cumulative_target_windows"] == 7715,
            "This increment follows the independently read A87")
    old = [row["episode"] for version in prior_versions for row in version["episodes"]]
    old.extend(read(original)["episodes"])
    scenes = {row["scene_sha256"] for row in old}
    ids = {row["id"] for row in old}
    excluded = read(protected)["clusters"]
    scenes.update(row["scene"]["sha256"] for row in excluded if row.get("split") in ("QUAL", "DEV", "VAL", "TEST"))
    refs = base["statistics_sha256"]
    members = []
    for wave, digest, roster in ((2,WAVE2,WAVE2_CASES), (3,wave3_sha,WAVE3_CASES)):
        root = work / f"REACH-TRAIN-{wave:02d}-LABELS-R4"
        for row in closed_entries(root, digest, roster, refs):
            episode = row["episode"]
            identity, geometry = episode["id"], episode["scene_sha256"]
            require(identity not in ids and geometry not in scenes, "Prior or protected membership overlap")
            ids.add(identity)
            scenes.add(geometry)
            members.append((root, row))
    require(len(members) >= 4, "Fewer than four newly admitted episodes. Keep the existing A87 reader version.")
    out.mkdir(mode=0o700)
    episodes = []
    for root, row in members:
        ep = row["episode"]
        for item in (ep["labels"], *ep["frame_shards"], row["corrected"], row["aux"]):
            materialize(pin(root / item["file"], item["sha256"]), out / item["file"])
        episodes.append({key: row[key] for key in ("episode", "corrected", "aux", "population",
                          "native_result_sha256", "original_execution_spec_sha256")})
    windows = sum(row["episode"]["windows"] for row in episodes)
    manifest = dict(schema="V64_REACH_COLD_INCREMENT_FROM_A87", base_a87_sha256=A87,
                    cumulative_target_episodes=87+len(episodes), cumulative_target_windows=7715+windows,
                    added=len(episodes), episodes=episodes, windows=windows,
                    full_H40=sum(row["aux"]["complete_H40"] for row in episodes),
                    statistics_sha256=refs, original_label_batches=[WAVE2, wave3_sha],
                    official_training_admission=False, optimizer_visits=0)
    receipt = readback(out, manifest, state, actions)
    write(out / "MANIFEST.json", manifest)
    write(out / "CORE_READBACK.json", dict(manifest_sha256=sha(out / "MANIFEST.json"), **receipt))


def receive(root, parent, expected, state, actions, output):
    base = read(pin(parent, A87))
    manifest = read(pin(root / "MANIFEST.json", expected))
    core = read(root / "CORE_READBACK.json")
    admitted = tuple(row["episode"]["id"] for row in manifest["episodes"])
    roster = (*WAVE2_CASES, *WAVE3_CASES)
    require(manifest["base_a87_sha256"] == A87 and core["manifest_sha256"] == expected
            and manifest["added"] == len(admitted) >= 4 and admitted == tuple(r for r in roster if r in admitted)
            and manifest["statistics_sha256"] == base["statistics_sha256"]
            and not manifest["official_training_admission"], "Invalid portable cold version")
    actual = readback(root, manifest, state, actions)
    require(all(core[key] == value for key, value in actual.items()), "Cross-site training readback mismatch")
    write(output, dict(manifest_sha256=expected, verified_episodes=manifest["added"],
                       verified_windows=manifest["windows"], **actual,
                       official_training_admission=False, actual_optimizer_visits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, fields in (("build", ("work", "prior", "protected", "original", "wave3_sha", "state", "actions", "out")),
                         ("receive", ("root", "parent", "expected", "state", "actions", "output"))):
        command = modes.add_parser(mode)
        for name in fields:
            command.add_argument(name, type=str if name in ("expected", "wave3_sha") else Path)
    args = vars(parser.parse_args())
    {"build": build, "receive": receive}[args.pop("mode")](**args)
