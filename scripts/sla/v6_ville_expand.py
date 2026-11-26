"""Check fixed extra single-arm episodes after the initial RGB and FK pilots."""

import argparse
import hashlib
import json
from pathlib import Path

from v6_ville_pilot import CAMERAS, SOURCES, fetch, inspect, need, read, request, write


def expand(pilot, out):
    out.mkdir(mode=0o700)
    budget = 1024**3
    records = []
    for subset, revision in SOURCES[:2]:
        source = pilot / subset
        repo = f"villekuosmanen/{subset}"
        with request(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}") as response:
            card = json.load(response)
        need(card["sha"] == revision and not card["gated"]
             and card["cardData"]["license"] == "apache-2.0", "Unverified source card")
        info = read(source / "meta/info.json")
        task, = map(json.loads, (source / "meta/tasks.jsonl").read_text().splitlines())
        episodes = {row["episode_index"]: row for row in map(json.loads,
                    (source / "meta/episodes.jsonl").read_text().splitlines())}
        need(info["fps"] == 20 and info["robot_type"] == "arx5" and len(episodes) == info["total_episodes"],
             "Expected the original single-arm source metadata")
        selected = sorted((index for index in episodes if index not in (0, 1)),
                          key=lambda i: hashlib.sha256(f"SLA-V6-VILLE-20261126/{subset}/{i}".encode()).hexdigest())[:12]
        split = {index: "DEV_CANDIDATE" if order >= 10 else "TRAIN_CANDIDATE"
                 for order, index in enumerate(selected)}
        root = out / subset
        root.mkdir()
        write(root / "ROSTER.json", dict(repo=repo, revision=revision,
                                       samples=[dict(index=i, split=split[i]) for i in selected]))
        names = []
        for index in selected:
            layout = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
            names.append(info["data_path"].format(**layout))
            names.extend(info["video_path"].format(video_key=cam, **layout) for cam in CAMERAS)
        files = fetch(root, repo, revision, names, budget)
        budget -= sum(row["bytes"] for row in files)
        scored = []
        for index in selected:
            row = inspect(root, index, info, task["task_index"])
            need(row["rows"] == episodes[index]["length"], "Declared single-arm episode length changed")
            scored.append(dict(**row, proposed_split=split[index]))
        actual = dict(repo=repo, revision=revision, scanned=scored,
                      downloaded_bytes=sum(row["bytes"] for row in files), files=files,
                      source_train_added=0, model_updates=0)
        write(root / "ACTUAL.json", actual)
        records.append(dict(repo=repo, actual_sha256=hashlib.sha256((root / "ACTUAL.json").read_bytes()).hexdigest(),
                            decoded_episodes=len(scored), source_train_added=0))
    write(out / "ACTUAL.json", dict(sources=records, downloaded_bytes=1024**3 - budget,
                                    new_train_episodes=0, optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("out", type=Path)
    expand(**vars(parser.parse_args()))
