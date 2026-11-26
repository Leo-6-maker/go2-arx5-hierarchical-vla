"""Fetch and inspect a bounded, pinned single-arm V6 sample without granting training admission."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request


SOURCES = (
    ("pick_coffee_prop_20Hz", "b1088b5595c92183450f306f9a1199d0771bf7dd"),
    ("drop_footbag_into_dice_tower", "cb6f2fb33737afba63852784d71db35c17076adb"),
    ("rice-cooker_step_2", "83c9d785c551716f17f94d142db2dddb9fc93cb6"),
)
CAMERAS = ("observation.images.front", "observation.images.wrist")
EPISODES = (0, 1)
BYTE_LIMIT = 512 * 1024**2


def need(condition, failure):
    if not condition:
        raise ValueError(failure)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def request(url, *, method="GET"):
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers={
        "User-Agent": "ARX5-Source-Expansion-pinned-pilot", "Accept-Encoding": "identity"}), timeout=45)


def fetch(root, repo, revision, names, budget, *, member_limit=120 * 1024**2):
    base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
    plans = []
    for name in names:
        url = base + urllib.parse.quote(name, safe="/")
        with request(url, method="HEAD") as response:
            size = int(response.headers.get("Content-Length", "0"))
        need(0 < size <= member_limit, "Remote member is unbounded or exceeds the pilot cap")
        plans.append((name, url, size))
    total = sum(size for _, _, size in plans)
    need(total <= budget and shutil.disk_usage(root).free >= total + 20 * 1024**3, "Pilot capacity unavailable")
    manifests = []
    for name, url, size in plans:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        used = 0
        pending = file.with_suffix(file.suffix + ".pending")
        with request(url) as response, pending.open("xb") as sink:
            while block := response.read(1024**2):
                used += len(block)
                need(used <= size, "Received more bytes than the bound file")
                digest.update(block)
                sink.write(block)
        need(used == size, "Incomplete source member")
        pending.rename(file)
        manifests.append(dict(path=name, bytes=size, sha256=digest.hexdigest()))
    return manifests


def inspect(root, ep, info, task):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    layout = dict(episode_chunk=ep//info["chunks_size"], episode_index=ep)
    table = pq.read_table(root / info["data_path"].format(**layout))
    def values(name):
        return np.asarray(table[name].to_pylist())
    state, action = values("observation.state"), values("action")
    clocks, frames = values("timestamp").reshape(-1), values("frame_index").reshape(-1)
    count, fps = len(clocks), info["fps"]
    need(count >= 4 and state.shape == action.shape == (count, 7)
         and np.isfinite(state).all() and np.isfinite(action).all()
         and np.array_equal(frames, np.arange(count))
         and np.all(values("episode_index") == ep) and np.all(values("task_index") == task)
         and np.max(np.abs(clocks - np.arange(count)/fps)) < 1e-4,
         "Unexpected native episode identity, axis, or numeric clock")
    selected = [0, count//3, 2*count//3, count-1]
    cameras = {}
    for name in CAMERAS:
        video = root / info["video_path"].format(video_key=name, **layout)
        frames_seen, images = 0, {}
        max_error = 0.
        with av.open(str(video)) as clip:
            for index, frame in enumerate(clip.decode(video=0)):
                need(index < count and frame.pts is not None, "Decoded stream exceeds trajectory or has no PTS")
                max_error = max(max_error, abs(float(frame.pts*frame.time_base)-clocks[index]))
                frames_seen += 1
                if index in selected:
                    rgb = frame.to_ndarray(format="rgb24")
                    need(rgb.shape == tuple(info["features"][name]["shape"]), "Camera image geometry differs")
                    images[str(index)] = hashlib.sha256(rgb.tobytes()).hexdigest()
        need(frames_seen == count and len(images) == len(selected)
             and max_error <= .5/fps, "Camera stream cannot be aligned to native frames")
        cameras[name] = dict(decoded_frames=frames_seen, selected_rgb_sha256=images, max_pts_error_s=max_error)
    horizon = 2 * fps
    rows = [dict(anchor=i, supported_time=min(horizon, count-i),
                 unsupported_tail=horizon-min(horizon, count-i)) for i in selected]
    eef = None
    if "observation.eef_6d_pose" in table.column_names:
        eef = values("observation.eef_6d_pose")
        need(eef.shape == (count, 6) and np.isfinite(eef).all(), "Invalid recorded EEF")
    return dict(episode=ep, rows=count, fps=fps, frames=rows, cameras=cameras,
                state_min=state.min(axis=0).tolist(), state_max=state.max(axis=0).tolist(),
                action_min=action.min(axis=0).tolist(), action_max=action.max(axis=0).tolist(),
                action_observation_abs_max=np.abs(action-state).max(axis=0).tolist(),
                eef_observation_span=None if eef is None else np.ptp(eef, axis=0).tolist(),
                candidate_horizon=horizon, action_units_and_joint_order_verified=False,
                tcp_and_gripper_semantics_verified=False, source_outcome_verified=False,
                training_eligible=False)


def run(out):
    out.mkdir(mode=0o700)
    results = []
    budget = BYTE_LIMIT
    for name, revision in SOURCES:
        repo = "villekuosmanen/" + name
        root = out / name
        root.mkdir(mode=0o700)
        with request(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}") as response:
            card = json.load(response)
        need(card["sha"] == revision and not card["gated"]
             and card["cardData"]["license"] == "apache-2.0", "Unverified public sample license or revision")
        files = fetch(root, repo, revision, ["README.md", "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"], budget)
        budget -= sum(row["bytes"] for row in files)
        info = read(root / "meta/info.json")
        need(info["robot_type"] == "arx5" and info["codebase_version"] == "v2.1"
             and info["fps"] in (20,50) and set(info["splits"]) == {"train"},
             "Unexpected sample dataset or robot")
        tasks = [json.loads(row) for row in (root / "meta/tasks.jsonl").read_text().splitlines()]
        records = {row["episode_index"]: row for row in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())}
        task, = tasks
        names = []
        for index in EPISODES:
            layout = dict(episode_index=index, episode_chunk=index//info["chunks_size"])
            names.append(info["data_path"].format(**layout))
            names.extend(info["video_path"].format(video_key=key, **layout) for key in CAMERAS)
        picked = fetch(root, repo, revision, names, budget)
        files.extend(picked)
        budget -= sum(row["bytes"] for row in picked)
        scored = [inspect(root, ep, info, task["task_index"]) for ep in EPISODES]
        need(all(records[row["episode"]]["length"] == row["rows"] for row in scored), "Public episode counts changed")
        result = dict(repo=repo, revision=revision, license="Apache-2.0", task=task["task"], member_hashes=files,
                      declared_episodes=info["total_episodes"], episodes=scored, source_train_added=0,
                      status="ACTUAL_STATE_ACTION_AND_RGB_READ_WAITING_FK_AND_PROVENANCE")
        write(root / "ACTUAL.json", result)
        results.append(result)
    write(out / "ACTUAL.json", dict(licensed_source_pilot_count=len(results), downloaded_bytes=BYTE_LIMIT-budget,
          physical_requests=0, optimizer_updates=0, new_train_episodes=0, sources=results))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
