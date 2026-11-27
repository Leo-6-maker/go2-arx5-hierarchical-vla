"""Seal the four new PLACE episodes downstream of the frozen A74 corpus."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_bank import PINS
from v63_adapt_data import physical_aux, read_added, signed_case, zip_episode


PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
A74 = "e3487b9322a6ce7e795a69a596c762a775e7475af3752d6d0142ae5f37d3f0fd"
PROTECTED = "a1dfa32f329cb91d474e20488cbeeafa3dc41858f32dacd6cd593dbed61bbb26"
BATCHES = (
    ("03", "955ad71fe4ddeb8e47fc184b2a3c3b8fc49678a8b4d546dfd73f64102d52af69"),
    ("04", "7903047946c60cb538b54e195fb65b1b88a7d79024aef4a766c06613a81dc8b5"),
    ("05", "d9402d9586069944cff0977206863ecf2e9dea1a60e361b7dcc5a290d38b85dd"),
)
EXPECTED = ("EF5_DIVERSE_PLACE_005", "EF5_DIVERSE_PLACE_007", "EF5_DIVERSE_PLACE_008", "EF5_DIVERSE_PLACE_010")


def boundary_reads(root, manifest, stats):
    from openpi_client.image_tools import resize_with_pad

    dataset = read_added(root, manifest, stats, resize_with_pad)
    offset, proofs = 0, []
    for record in manifest["episodes"]:
        ep = record["episode"]
        for index in (offset, offset + ep["windows"] - 1):
            sample = dataset[index]
            mask = dataset.support(index)
            expected_views = dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False)
            require(sample["actions"].shape == (40,32) and sample["state"].shape == (32,)
                    and np.isfinite(sample["actions"]).all() and not sample["actions"][~mask].any()
                    and sample["image_mask"] == expected_views, "Real R4 training reader contract changed")
            proofs.append(dict(episode=ep["id"], index=index, support=int(mask.sum()),
                               action_sha256=hashlib.sha256(sample["actions"].tobytes()).hexdigest(),
                               rgb={key: hashlib.sha256(sample["image"][key].tobytes()).hexdigest()
                                    for key, valid in expected_views.items() if valid}))
        offset += ep["windows"]
    return proofs


def statistics(manifest, state, actions):
    references = {"state": (state, "state_statistics"), "actions": (actions, "action_statistics")}
    return {role: read(pin(path, manifest["statistics_sha256"][key]))[role]
            for role, (path, key) in references.items()}


def build(work, base, archive, base_corrected, protected, out):
    old = read(pin(base, A74))
    original = read(pin(archive, PINS["target_archive"]))
    corrected = read(pin(base_corrected, PINS["target_corrected"]))
    old52 = {row["episode"] for row in corrected["episodes"]}
    base_members = [row for row in original["episodes"] if row["id"] in old52]
    base_members.extend(row["episode"] for row in old["episodes"])
    ids = {row["id"] for row in base_members}
    scenes = {row["scene_sha256"] for row in base_members}
    excluded = {row["scene"]["sha256"] for row in read(pin(protected, PROTECTED))["clusters"]
                if row.get("split") != "TRAIN"}
    require(len(old52) == 52 and len(ids) == len(base_members) == 74 and old["windows"] == 2094
            and not scenes.intersection(excluded), "The unchanged A74 membership must be independently bound")
    staged, phases, bindings = [], Counter(), None
    for wave, digest in BATCHES:
        data = read(pin(work / f"DIVERSE-PLACE-TRAIN-{wave}-LABELS-R4/MANIFEST.json", digest))
        refs = {key: data[key] for key in ("state_statistics", "action_statistics")}
        require(data["original_transport"] and data["mask_preserved"] and data["gripper_preserved"]
                and data["actual_boundary_reads"] == 2 * len(data["episodes"])
                and {key: value["sha256"] for key, value in refs.items()} == old["statistics_sha256"],
                "New data must use the original R4 labels, split and statistics")
        bindings = refs
        for row in data["episodes"]:
            source = row["original_episode"]
            episode = read(pin(source["path"], source["sha256"]))
            require(episode["id"] not in ids and episode["scene_sha256"] not in scenes | excluded
                    and episode["split"] == "TRAIN" and episode["individual_pass"], "Overlapping or unsigned new data")
            ids.add(episode["id"])
            scenes.add(episode["scene_sha256"])
            phases.update(row["phase_windows"])
            staged.append((Path(row["archive"]), episode, row, row["corrected"], signed_case(work, row)))
    require(tuple(args[1]["id"] for args in staged) == EXPECTED and len(ids) == 78,
            "The fixed four-episode A78 increment changed")
    out.mkdir(mode=0o700)
    members = []
    for root, ep, row, labels, case in staged:
        stored = zip_episode(root, ep, row, labels, out)
        stored["aux"] = physical_aux(case, out / stored["corrected"]["file"], out / ep["id"] / "R4_AUX.npz")
        members.append(stored)
    windows = sum(row["episode"]["windows"] for row in members)
    manifest = dict(plan=PLAN, corpus="A78_COLD_INCREMENT", base_a74_sha256=A74, already_attested=74,
                    cumulative_target_episodes=78, episodes=members, added=len(members), windows=windows,
                    cumulative_target_windows=4450 + old["windows"] + windows,
                    full_H40=sum(row["aux"]["complete_H40"] for row in members),
                    actual_support=sum(row["aux"]["actual_support"] for row in members),
                    phase_windows=dict(phases), statistics_sha256=old["statistics_sha256"],
                    original_label_batches=[digest for _, digest in BATCHES], training_admission=False,
                    native_integrations=0, model_forwards=0, optimizer_visits=0)
    require((windows, manifest["full_H40"], manifest["cumulative_target_windows"]) == (502, 490, 7046),
            "A78 source and supported-window reconciliation failed")
    stats = statistics(manifest, Path(bindings["state_statistics"]["path"]), Path(bindings["action_statistics"]["path"]))
    actual = boundary_reads(out, manifest, stats)
    dataset_path = out / "MANIFEST.json"
    write(dataset_path, manifest)
    write(out / "CORE_READBACK.json", dict(manifest_sha256=sha(dataset_path), actual_RGB_boundaries=actual,
          verified_episodes=len(members), actual_optimizer_visits=0))


def receive(root, expected, state, actions, output):
    manifest = read(pin(root / "MANIFEST.json", expected))
    core = read(root / "CORE_READBACK.json")
    require(manifest["plan"] == PLAN and manifest["base_a74_sha256"] == A74
            and core["manifest_sha256"] == expected
            and tuple(row["episode"]["id"] for row in manifest["episodes"]) == EXPECTED
            and manifest["training_admission"] is False, "The cold increment differs from Corelab")
    actual = boundary_reads(root, manifest, statistics(manifest, state, actions))
    require(actual == core["actual_RGB_boundaries"] and len(actual) == 8,
            "Hopper and Corelab reads differ on the same original RGB and R4 action rows")
    write(output, dict(manifest_sha256=expected, actual_RGB_boundaries=actual,
          verified_episodes=manifest["added"], verified_windows=manifest["windows"],
          base_a74_sha256=A74, actual_optimizer_visits=0, official_training_admission=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name, entries in (
        ("build", ("work", "base", "archive", "base_corrected", "protected", "out")),
        ("receive", ("root", "expected", "state", "actions", "output")),
    ):
        command = modes.add_parser(name)
        for key in entries:
            command.add_argument(key, type=str if key == "expected" else Path)
    args = vars(parser.parse_args())
    {"build": build, "receive": receive}[args.pop("mode")](**args)
