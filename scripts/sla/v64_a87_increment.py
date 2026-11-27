"""Transfer the next four closed R4 episodes and read them on both sites."""

import argparse
import hashlib
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v63_adapt_data import materialize, physical_aux, read_added, signed_case, zip_episode
from v64_a78_delta import A74, boundary_reads, statistics
from v64_go2_delta4 import A78


A83 = "91b2c0c2c4b4f8df887474bb74e1deabcc0559414b8220e2025220e240fff91f"
PAST = (
    ("MOBILE-F1-BLUE-TRAIN-03", "3f3416aed488a0918b2683272a3af8a887cd773561e23d9fa06a78f81ca1bd1f"),
    ("DIVERSE-PLACE-TRAIN-07", "d76c2d6465701b0697a9c223c1cc6ca855c8d68bb4c3e73bbc9ec790847aaaee"),
)
REACH = "e5c542bc56ccdd5d0bb466110f8b80ea7eeb875a9aa6a8a3eeade895cfb815a8"
EXPECTED = ("EF5_T_MOBILE_13", "EF5_DIVERSE_PLACE_013", "EF5_T_REACH_01", "EF5_T_REACH_02")


def readback(root, manifest, state, actions):
    from openpi_client.image_tools import resize_with_pad

    stats = statistics(manifest, state, actions)
    boundaries = boundary_reads(root, manifest, stats)
    dataset = read_added(root, manifest, stats, resize_with_pad)
    prompts = []
    first = 0
    for row in manifest["episodes"]:
        episode = row["episode"]
        prompt = dataset[first]["prompt"]
        require(prompt == episode["language"], "The training reader must expose the original task language")
        prompts.append(dict(episode=episode["id"], task=episode["task"],
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        first += episode["windows"]
    return dict(actual_RGB_action_boundaries=boundaries, training_prompt_readbacks=prompts)


def build(work, prior, protected, original, state, actions, out):
    a74, a78, a83 = (read(pin(prior / f"A{n}.json", digest))
                     for n, digest in ((74, A74), (78, A78), (83, A83)))
    require(a83["cumulative_target_episodes"] == 83 and a83["cumulative_target_windows"] == 7455,
            "The previous cold version must have the original membership")
    archive = read(original)["episodes"]
    old = [row["episode"] for version in (a74,a78,a83) for row in version["episodes"]]
    scenes = {row["scene_sha256"] for row in [*archive, *old]}
    ids = {row["id"] for row in [*archive, *old]}
    exclusion = read(protected)["clusters"]
    scenes.update(row["scene"]["sha256"] for row in exclusion if row.get("split") in ("QUAL", "DEV", "VAL", "TEST"))
    refs = a83["statistics_sha256"]
    out.mkdir(mode=0o700)
    entries = []
    for wave, digest in PAST:
        batch = read(pin(work / (wave + "-LABELS-R4") / "MANIFEST.json", digest))
        require({k: batch[k]["sha256"] for k in refs} == refs and batch["train"] == 1 and batch["mask_preserved"],
                "Existing collection must preserve its signed R4 contract")
        added, = batch["episodes"]
        original_episode = added["original_episode"]
        episode = read(pin(original_episode["path"], original_episode["sha256"]))
        case = signed_case(work, added)
        stored = zip_episode(Path(added["archive"]), episode, added, added["corrected"], out)
        stored["aux"] = physical_aux(case, out / stored["corrected"]["file"], out / episode["id"] / "R4_AUX.npz")
        entries.append(stored)
    reach_root = work / "REACH-TRAIN-01-LABELS-R4"
    reach = read(pin(reach_root / "MANIFEST.json", REACH))
    require(reach["train"] == reach["paid"] == 2 and reach["statistics_sha256"] == refs and not reach["rejected"],
            "Require both independently sealed Reach examples")
    for row in reach["episodes"]:
        episode = row["episode"]
        require(episode["task"].startswith("REACH-ORIENT-") and episode["language"].startswith("Reach above"),
                "Reach must carry its own task prompt")
        for item in (episode["labels"], *episode["frame_shards"], row["corrected"], row["aux"]):
            materialize(pin(reach_root / item["file"], item["sha256"]), out / item["file"])
        entries.append(row)
    for row in entries:
        episode = row["episode"]
        require(episode["id"] not in ids and episode["scene_sha256"] not in scenes
                and episode["split"] == "TRAIN" and episode["individual_pass"],
                "New evidence overlaps previously used TRAIN or protected scenes")
        ids.add(episode["id"])
        scenes.add(episode["scene_sha256"])
    require(tuple(row["episode"]["id"] for row in entries) == EXPECTED, "Fixed A87 increment changed")
    windows = sum(row["episode"]["windows"] for row in entries)
    manifest = dict(schema="V64_A87_COLD_INCREMENT", base_a83_sha256=A83, cumulative_target_episodes=87,
                    cumulative_target_windows=7455+windows, added=4, episodes=entries, windows=windows,
                    full_H40=sum(row["aux"]["complete_H40"] for row in entries), statistics_sha256=refs,
                    original_label_batches=[*[digest for _, digest in PAST], REACH],
                    official_training_admission=False, optimizer_visits=0)
    receipt = readback(out, manifest, state, actions)
    write(out / "MANIFEST.json", manifest)
    write(out / "CORE_READBACK.json", dict(manifest_sha256=sha(out / "MANIFEST.json"), **receipt))


def receive(root, expected, state, actions, output):
    manifest = read(pin(root / "MANIFEST.json", expected))
    core = read(root / "CORE_READBACK.json")
    require(manifest["base_a83_sha256"] == A83 and core["manifest_sha256"] == expected
            and tuple(row["episode"]["id"] for row in manifest["episodes"]) == EXPECTED
            and not manifest["official_training_admission"], "Unexpected scientific input")
    actual = readback(root, manifest, state, actions)
    require(all(core[key] == value for key, value in actual.items()), "The two sites disagree on actual inputs")
    write(output, dict(manifest_sha256=expected, verified_episodes=manifest["added"], verified_windows=manifest["windows"],
                      **actual, official_training_admission=False, actual_optimizer_visits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, parameters in (("build", ("work", "prior", "protected", "original", "state", "actions", "out")),
                             ("receive", ("root", "expected", "state", "actions", "output"))):
        command = modes.add_parser(mode)
        for name in parameters:
            command.add_argument(name, type=str if name == "expected" else Path)
    args = vars(parser.parse_args())
    {"build": build, "receive": receive}[args.pop("mode")](**args)
