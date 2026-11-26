"""Read the isolated Ville single-arm source ABI from verified numeric and RGB bytes."""

import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np

from v6_robodojo_intake import digest, read
from v6_ville_pilot import need, write
from v6_ville_source import HORIZON, SCHEMA


CAMERAS = {"observation.images.front": "base_0_rgb", "observation.images.wrist": "right_wrist_0_rgb"}


class SingleArmReader:
    def __init__(self, root, manifest_sha, video_root, *, split, resize, cache_episodes=12,
                 schema=SCHEMA, native_hz=20, horizon=HORIZON, cameras=CAMERAS):
        receipt = root / "MANIFEST.json"
        need(digest(receipt) == manifest_sha and split in ("TRAIN", "SOURCE_DEV"), "Unverified single-arm manifest")
        self.manifest = read(receipt)
        need(self.manifest["schema"] == schema and self.manifest["native_hz"] == native_hz
             and self.manifest["horizon"] == horizon
             and self.manifest["target_command"] is False, "Wrong single-arm source ABI")
        statfile = root / self.manifest["statistics"]["file"]
        need(digest(statfile) == self.manifest["statistics"]["sha256"], "Single-arm statistics changed")
        stats = read(statfile)
        need(stats["split"] == "TRAIN", "Single-arm statistics require TRAIN only")
        self.numerical = {}
        self.stats = {field: tuple(np.asarray(stats[field][key]) for key in ("q01", "q99"))
                      for field in ("state", "actions")}
        self.resize, self.video_root = resize, video_root
        self.cameras = cameras
        self.episodes = {row["episode"]: {**row, "task": self.manifest["task"]}
                         for row in self.manifest["episodes"] if row["split"] == split}
        self.rows = [(episode, i) for episode, row in self.episodes.items() for i in range(row["windows"])]
        self.cache = OrderedDict()
        self.cache_episodes = cache_episodes
        for episode, row in self.episodes.items():
            meta = row["arrays"]
            path = root / meta["file"]
            need(digest(path) == meta["sha256"], "Single-arm numerical bytes changed")
            with np.load(path, allow_pickle=False) as data:
                samples = row["windows"]
                need(data["actions"].shape == (samples, horizon, 7) and data["state"].shape == (samples, 7)
                     and data["mask"].shape == (samples, horizon) and data["mask"].all()
                     and np.isfinite(data["actions"]).all() and np.isfinite(data["state"]).all()
                     and np.allclose(data["future_timestamp"] - data["observation_timestamp"][:, None],
                                     np.arange(1, horizon + 1) / native_hz, atol=1e-5, rtol=0),
                     "Unexpected physical source frames or support")
                self.numerical[episode] = (data["state"], data["actions"], data["mask"], data["observation_timestamp"])

    def __len__(self):
        return len(self.rows)

    def normalize(self, value, field):
        lo, hi = self.stats[field]
        normalized = ((value - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)
        result = np.zeros((*value.shape[:-1], 32), dtype=np.float32)
        result[..., :7] = normalized
        return result

    def support(self, index):
        episode, anchor = self.rows[index]
        return self.numerical[episode][2][anchor]

    def video(self, episode):
        import av

        if episode not in self.cache:
            captures = {}
            ts = self.numerical[episode][3]
            for key, name in self.cameras.items():
                info = self.episodes[episode]["views"][key]
                path = self.video_root / info["member"]
                need(path.resolve(strict=True).is_relative_to(self.video_root.resolve(strict=True))
                     and digest(path) == info["sha256"], "Single-arm camera file changed")
                images, padding = [], None
                with av.open(path) as clip:
                    for index, frame in enumerate(clip.decode(video=0)):
                        if index >= len(ts):
                            break
                        rgb = frame.to_ndarray(format="rgb24")
                        anchors = info["anchors"]
                        if str(index) in anchors:
                            need(hashlib.sha256(rgb.tobytes()).hexdigest() == anchors[str(index)],
                                 "Single-arm actual RGB anchor bytes changed")
                        need(abs(float(frame.pts * frame.time_base) - ts[index]) < 1e-5,
                             "Single-arm clock and camera frame mismatch")
                        small, this_mask = self.resize(rgb, 224, 224, return_mask=True)
                        if padding is None:
                            padding = this_mask
                        images.append(small)
                need(len(images) == len(ts), "An actual source camera lost frames")
                captures[name] = (np.stack(images), padding)
            self.cache[episode] = captures
            while len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode]

    def __getitem__(self, index):
        episode, anchor = self.rows[index]
        images = self.video(episode)
        state, actions, support, _ = self.numerical[episode]
        return dict(image={**{key: v[0][anchor] for key, v in images.items()},
                           "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
                    image_padding_mask={**{key: v[1] for key, v in images.items()},
                                        "left_wrist_0_rgb": np.zeros((224, 224), dtype=bool)},
                    image_mask=self.manifest["camera_mask"].copy(),
                    state=self.normalize(state[anchor], "state"), actions=self.normalize(actions[anchor], "actions"),
                    support=support[anchor], prompt=self.manifest["task"])


def preflight(root, manifest_sha, video_root, output, *, schema=SCHEMA, native_hz=20, horizon=HORIZON, cameras=CAMERAS):
    from openpi_client.image_tools import resize_with_pad

    by_split = {}
    for split in ("TRAIN", "SOURCE_DEV"):
        reader = SingleArmReader(root, manifest_sha, video_root, split=split, resize=resize_with_pad,
                                 schema=schema, native_hz=native_hz, horizon=horizon, cameras=cameras)
        cursor = 0
        for row in reader.episodes.values():
            for index in (cursor, cursor + row["windows"] - 1):
                item = reader[index]
                need(item["actions"].shape == (horizon, 32) and item["support"].all()
                     and all(np.isfinite(item[k]).all() for k in ("state", "actions"))
                     and not np.any(item["actions"][:, 7:]) and not item["image_mask"]["left_wrist_0_rgb"],
                     "Single-arm model input ABI failed")
            cursor += row["windows"]
        by_split[split] = dict(episodes=len(reader.episodes), windows=len(reader),
                             read_image_pairs=2 * len(reader.episodes))
    write(output, dict(by_split=by_split, manifest_sha256=manifest_sha, actual_native_hz=native_hz,
                       learner_shape=[horizon, 32], optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest_sha")
    parser.add_argument("video_root", type=Path)
    parser.add_argument("output", type=Path)
    preflight(**vars(parser.parse_args()))
