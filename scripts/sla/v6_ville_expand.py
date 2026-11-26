"""Check fixed extra single-arm episodes after the initial RGB and FK pilots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from v6_ville_pilot import CAMERAS, SOURCES, fetch, inspect, need, read, request, write


def expand(pilot, out, previous=None, recover_from=None, *, earlier=(), only=None):
    out.mkdir(mode=0o700)
    budget = 1024**3
    records = []
    for subset, revision in SOURCES[:2]:
        if only is not None and subset != only:
            continue
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
        excluded = {0, 1}
        for folder in (*earlier, previous):
            if folder is None:
                continue
            prior = read(folder / subset / "ROSTER.json")
            need(prior["repo"] == repo and prior["revision"] == revision,
                 "The previous sampled cohort has different provenance")
            excluded.update(r["index"] for r in prior["samples"])
        selected = sorted((index for index in episodes if index not in excluded),
                          key=lambda i: hashlib.sha256(f"SLA-V6-VILLE-20261126/{subset}/{i}".encode()).hexdigest())[:12]
        need(len(selected) == 12, "Insufficient independent source episodes for the fixed next cohort")
        split = {index: "DEV_CANDIDATE" if order >= 10 else "TRAIN_CANDIDATE"
                 for order, index in enumerate(selected)}
        root = out / subset
        root.mkdir()
        write(root / "ROSTER.json", dict(repo=repo, revision=revision, excluded=sorted(excluded),
                                       samples=[dict(index=i, split=split[i]) for i in selected]))
        names = []
        for index in selected:
            layout = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
            names.append(info["data_path"].format(**layout))
            names.extend(info["video_path"].format(video_key=cam, **layout) for cam in CAMERAS)
        files = []
        if recover_from is not None and (recover_from / subset / "ROSTER.json").is_file():
            old = recover_from / subset
            need(read(old / "ROSTER.json") == read(root / "ROSTER.json"),
                 "A recovery must use the identical immutable source cohort")
            for name in names:
                if not (old / name).is_file():
                    continue
                raw = old / name
                count = raw.stat().st_size
                need(0 < count <= 120*1024**2, "Previously downloaded file is not bounded")
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(raw, target)
                files.append(dict(path=name, bytes=count, sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                                  reused_sealed_download=True))
        missing = sorted(set(names) - {row["path"] for row in files})
        files.extend(fetch(root, repo, revision, missing, budget - sum(row["bytes"] for row in files)))
        budget -= sum(row["bytes"] for row in files)
        write(root / "FILE_SET.json", dict(files=files))
        scored, rejected = [], []
        for index in selected:
            try:
                row = inspect(root, index, info, task["task_index"])
                need(row["rows"] == episodes[index]["length"], "Declared single-arm episode length changed")
                scored.append(dict(**row, proposed_split=split[index]))
            except ValueError as exc:
                rejected.append(dict(episode=index, proposed_split=split[index], reason=str(exc)))
        actual = dict(repo=repo, revision=revision, scanned=scored,
                      rejected=rejected, files=files, bound_bytes=sum(row["bytes"] for row in files),
                      downloaded_bytes=sum(row["bytes"] for row in files if not row.get("reused_sealed_download")),
                      source_train_added=0, model_updates=0)
        write(root / "ACTUAL.json", actual)
        records.append(dict(repo=repo, actual_sha256=hashlib.sha256((root / "ACTUAL.json").read_bytes()).hexdigest(),
                            decoded_episodes=len(scored), rejected_episodes=len(rejected),
                            downloaded_bytes=actual["downloaded_bytes"], source_train_added=0))
    write(out / "ACTUAL.json", dict(sources=records, bound_bytes=1024**3 - budget,
                                    downloaded_bytes=sum(row["downloaded_bytes"] for row in records),
                                    new_train_episodes=0, optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--earlier", action="append", type=Path, default=[])
    parser.add_argument("--only", choices=[subset for subset, _ in SOURCES[:2]])
    parser.add_argument("--recover-from", type=Path)
    expand(**vars(parser.parse_args()))
