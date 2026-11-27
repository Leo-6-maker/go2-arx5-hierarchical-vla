"""Transport the next cold R4 increment after A78 and verify the actual reader."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import PINS
from v63_adapt_data import physical_aux, signed_case, zip_episode
from v64_a78_delta import A74, PLAN, PROTECTED, boundary_reads, statistics


A78 = "b16f0993e88a4fa7058ed54fdac163021b45d95e46777166b88d2a245058cc1b"
BATCHES = (
    ("MOBILE-F1-RED-TRAIN-02", "06e7d1e4fc5c6ccc67f279e594b83be6312ee1550b8f09eb45b9b30f8ceddc4f"),
    ("DIVERSE-PLACE-TRAIN-06", "1a7ff398ece8bf3ee79f3271b60937aca05361a907032dc4fc16c81614ddd0d9"),
)
FIXED = ("EF5_T_MOBILE_09", "EF5_T_MOBILE_10", "EF5_DIVERSE_PLACE_012")
BLUE = ("EF5_T_MOBILE_11", "EF5_T_MOBILE_12")


def expected(ids):
    return ids[:3] == FIXED and ids[3:] in ((BLUE[0],), (BLUE[1],), BLUE)


def known_members(base, prior, archive, corrected):
    a74, a78 = read(pin(base, A74)), read(pin(prior, A78))
    original = read(pin(archive, PINS["target_archive"]))
    labels = read(pin(corrected, PINS["target_corrected"]))
    fixed = {row["episode"] for row in labels["episodes"]}
    members = [row for row in original["episodes"] if row["id"] in fixed]
    members.extend(row["episode"] for source in (a74, a78) for row in source["episodes"])
    require(len(fixed) == 52 and len({r["id"] for r in members}) == len(members) == 78
            and a78["cumulative_target_windows"] == 7046, "The exact A78 parent must remain available")
    return a78, members


def build(work, base, prior, archive, corrected, protected, out):
    a78, preceding = known_members(base, prior, archive, corrected)
    ids, scenes = {r["id"] for r in preceding}, {r["scene_sha256"] for r in preceding}
    excluded = {r["scene"]["sha256"] for r in read(pin(protected, PROTECTED))["clusters"]
                if r.get("split") != "TRAIN"}
    require(not scenes & excluded, "The parent overlaps protected evaluation")
    blue = "MOBILE-F1-BLUE-TRAIN-02"
    terminal = read(work / blue / "RESULT.private.json")
    ready = work / (blue + "-LABELS-R4") / "MANIFEST.json"
    require(read(work / blue / "control/LEASE.json")["status"] == "CLOSED" and terminal["exit_code"] == 0
            and terminal["paid_requests"] == 2 and terminal["unreconciled"] is None
            and tuple(r["condition"] for r in terminal["records"]) == BLUE
            and 1 <= terminal["admitted_train"] <= 2, "Require both real closed blue attempts and a qualified increment")
    stages, phases = [], Counter()
    batches = (*BATCHES, (blue, sha(ready)))
    for name, digest in batches:
        source = read(pin(work / (name + "-LABELS-R4") / "MANIFEST.json", digest))
        bindings = {key: source[key] for key in ("state_statistics", "action_statistics")}
        require(source["original_transport"] and source["mask_preserved"] and source["gripper_preserved"]
                and source["actual_boundary_reads"] == 2 * len(source["episodes"])
                and {key: value["sha256"] for key, value in bindings.items()} == a78["statistics_sha256"],
                "Preserve the original R4 statistics, masks, and action contract")
        if name == blue:
            require(tuple(row["episode"] for row in source["episodes"]) ==
                    tuple(row["condition"] for row in terminal["records"] if row["admitted"]),
                    "Blue admissions differ from the closed teacher wave")
        for row in source["episodes"]:
            ref = row["original_episode"]
            episode = read(pin(ref["path"], ref["sha256"]))
            require(episode["split"] == "TRAIN" and episode["individual_pass"] and episode["id"] not in ids
                    and episode["scene_sha256"] not in scenes | excluded, "Duplicate or unqualified target episode")
            ids.add(episode["id"])
            scenes.add(episode["scene_sha256"])
            phases.update(row["phase_windows"])
            stages.append((Path(row["archive"]), episode, row, row["corrected"], signed_case(work, row)))
    require(expected(tuple(row[1]["id"] for row in stages)), "The four or five fixed target additions changed")
    out.mkdir(mode=0o700)
    members = []
    for root, episode, row, labels, case in stages:
        stored = zip_episode(root, episode, row, labels, out)
        stored["aux"] = physical_aux(case, out / stored["corrected"]["file"], out / episode["id"] / "R4_AUX.npz")
        members.append(stored)
    windows = sum(row["episode"]["windows"] for row in members)
    manifest = dict(plan=PLAN, corpus=f"A{len(ids)}_COLD_INCREMENT", base_a74_sha256=A74, base_a78_sha256=A78,
                    cumulative_target_episodes=len(ids), cumulative_target_windows=7046 + windows,
                    already_attested=78, episodes=members, added=len(members), windows=windows,
                    full_H40=sum(row["aux"]["complete_H40"] for row in members),
                    actual_support=sum(row["aux"]["actual_support"] for row in members),
                    original_label_batches=[digest for _, digest in batches],
                    phase_windows=dict(phases), statistics_sha256=a78["statistics_sha256"],
                    training_admission=False, optimizer_visits=0)
    state = Path(bindings["state_statistics"]["path"])
    actions = Path(bindings["action_statistics"]["path"])
    actual = boundary_reads(out, manifest, statistics(manifest, state, actions))
    write(out / "MANIFEST.json", manifest)
    write(out / "CORE_READBACK.json", dict(manifest_sha256=sha(out / "MANIFEST.json"),
          actual_RGB_boundaries=actual, verified_episodes=len(members), actual_optimizer_visits=0))


def receive(root, expected_sha, state, actions, output):
    manifest = read(pin(root / "MANIFEST.json", expected_sha))
    core = read(root / "CORE_READBACK.json")
    require(manifest["plan"] == PLAN and manifest["base_a78_sha256"] == A78
            and manifest["base_a74_sha256"] == A74 and core["manifest_sha256"] == expected_sha
            and expected(tuple(row["episode"]["id"] for row in manifest["episodes"]))
            and manifest["training_admission"] is False, "The cold increment differs from Corelab")
    actual = boundary_reads(root, manifest, statistics(manifest, state, actions))
    require(actual == core["actual_RGB_boundaries"] and len(actual) == 2 * manifest["added"],
            "The readers disagree on original RGB or corrected action boundaries")
    write(output, dict(manifest_sha256=expected_sha, actual_RGB_boundaries=actual,
          verified_episodes=manifest["added"], verified_windows=manifest["windows"],
          base_a78_sha256=A78, actual_optimizer_visits=0, official_training_admission=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name, arguments in (("build", ("work", "base", "prior", "archive", "corrected", "protected", "out")),
                            ("receive", ("root", "expected_sha", "state", "actions", "output"))):
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=str if argument == "expected_sha" else Path)
    options = vars(parser.parse_args())
    {"build": build, "receive": receive}[options.pop("operation")](**options)
