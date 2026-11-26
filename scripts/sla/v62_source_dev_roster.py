"""Read and seal the independent new-source development windows on Corelab."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v61_source_bank import PINS
from v62_source_admission import V62


R48 = "57e9836d62bf0775bca7c7b7396ab4251d299a6d23116de66771af0fd3960991"
SEED = 524250


def read_samples(reader, domain, per_episode, horizon, dimensions):
    rows = []
    offset = 0
    for episode, entry in reader.episodes.items():
        windows = entry["windows"]
        anchors = [windows * (i + 1) // (per_episode + 1) for i in range(per_episode)]
        require(len(set(anchors)) == per_episode and 0 < anchors[0] <= anchors[-1] < windows,
                "The fixed anchors do not fit an independent source episode")
        for anchor in anchors:
            index = offset + anchor
            item = reader[index]
            mask = item.get("support", np.ones(horizon, dtype=np.bool_))
            require(reader.rows[index] == (episode, anchor) and item["actions"].shape == (horizon, 32)
                    and mask.shape == (horizon,) and mask.dtype == np.bool_ and mask.all()
                    and np.isfinite(item["actions"]).all() and not np.any(item["actions"][:, dimensions:])
                    and np.isfinite(item["state"]).all(), "An independent source window lost its native support or codec")
            hashes = {field: hashlib.sha256(item[field].tobytes()).hexdigest() for field in ("state", "actions")}
            hashes["support"] = hashlib.sha256(mask.tobytes()).hexdigest()
            hashes["views"] = {key: hashlib.sha256(value.tobytes()).hexdigest()
                               for key, value in item["image"].items() if item["image_mask"][key]}
            require(len(hashes["views"]) == (2 if domain == "single" else 3), "A real source view is missing")
            rows.append(dict(episode=episode, anchor=anchor, index=index, task=entry["task"], hashes=hashes,
                             horizon=horizon, dimensions=dimensions, evaluation_noise_seed=SEED + len(rows)))
        offset += windows
    require(offset == len(reader) and len(rows) >= 16, "Insufficient independent source development coverage")
    return dict(rows=rows, episodes=len(reader.episodes), task_windows=dict(Counter(r["task"] for r in rows)))


def build(dual, dual_train, single, single_video, out):
    from go2arx5_vla.sla_adapter.scale20k_data import RobodojoDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.pi0292.durable import write
    from v6_ville_reader import SingleArmReader

    dual_ref = dict(path=str(pin(dual / "MANIFEST.private.json", R48)), sha256=R48)
    old = read(pin(dual_train, PINS["dual_old"]))
    used = {row["raw_sha256"] for row in old["episodes"] if row["split"] == "TRAIN"}
    bimanual = RobodojoDataset(dual, dual_ref, split="SOURCE_DEV", resize_with_pad=resize_with_pad)
    require(len(bimanual.episodes) == 12 and not used.intersection(row["raw_sha256"] for row in bimanual.episodes.values()),
            "The Bimanual DEV population overlaps historical training")
    hand = SingleArmReader(single, PINS["single"], single_video, split="SOURCE_DEV", resize=resize_with_pad)
    used_single = {row["episode"] for row in hand.manifest["episodes"] if row["split"] == "TRAIN"}
    require(len(hand.episodes) == 6 and not used_single.intersection(hand.episodes), "Ville DEV overlaps TRAIN")
    data = dict(robodojo=read_samples(bimanual, "robodojo", 2, 50, 14),
                single=read_samples(hand, "single", 3, 40, 7))
    write(out, dict(plan=V62, split="SOURCE_DEV", evaluation_only=True, actual_readback=data,
                    source_manifest_sha256=dict(robodojo=R48, single=PINS["single"]),
                    model_forwards=0, optimizer_updates=0, physics_ticks=0))
    print(sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dual", "dual_train", "single", "single_video", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
