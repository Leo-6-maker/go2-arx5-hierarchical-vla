"""Read small real samples from two disjoint public ARX5 candidate domains on Corelab."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from v6_robodojo_pilot import CAMERAS as DOJO_CAMERAS, REPO, REV, inspect as inspect_dojo
from v6_ville_pilot import fetch, need, read, request, write


SC_REPO = "Kisaragi0/arx5_real_world_datasets"
SC_REV = "147e96997bc63d8781e5837b7ada9067e3989249"
SC_CAMERAS = ("observation.images.image", "observation.images.wrist_image")
SC_PILOT_PIN = "bbbbd7fd314de19775ffefb6aa35ef53be85698adc40040967edfae31d5b29b8"
SC_MODALITY_PIN = "4f5d4d0e7395f24e73631b4e42dedbcffee3206319e19cf36374e4c8e2e61ef4"
SIM_TASKS = ("push_T", "pack_objects_into_box")
DOJO_CONVERTED = "data/RoboDojo_ee_lerobot_v30_video"
DOJO_VIDEO = dict(zip(DOJO_CAMERAS, ("observation.images.cam_high", "observation.images.cam_left_wrist",
                                    "observation.images.cam_right_wrist"), strict=True))
DOJO_RAW_PIN = "26297c5efbe71ec7f887dcceacd97af299d7e0519dc79bc4ccca22115854ae49"
DOJO_META_PIN = "0f801f3645f838f485cb51078ea39cade3b8a4cdbe28d4c9a22c899352ea953e"
BLOCK = 256 * 1024
RANGE_CAP = 48 * 1024**2


def metadata(url):
    with request(url) as response:
        need(not response.headers.get("Link"), "Incomplete pinned file metadata")
        return json.load(response)


def license_is_apache(repo, revision):
    page = metadata(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?expand=sha&expand=gated&expand=cardData")
    need(page["sha"] == revision and not page["gated"] and page["cardData"]["license"] == "apache-2.0",
         "The public pinned source does not establish an Apache-2.0 dataset license")


def verify_receipt(path, expected):
    with path.open("rb") as f:
        need(hashlib.file_digest(f, "sha256").hexdigest() == expected, "Sealed source member changed")
    return path


class RangedMember(io.RawIOBase):
    """Seek inside an HDF5 file without staging gigabyte-scale camera archives."""

    def __init__(self, url, size):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.used, self.receipts = 0, []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        basis = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = basis[whence] + offset
        need(0 <= self.position <= self.size, "HDF5 read attempted to leave the pinned member")
        return self.position

    @lru_cache(maxsize=32)
    def block(self, start):
        end = min(start + BLOCK, self.size)
        size = end - start
        need(self.used + size <= RANGE_CAP, "Reached bounded HDF5 range budget")
        span = f"{start}-{end-1}"
        source = urllib.request.Request(self.url, headers={"Range": f"bytes={span}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(source, timeout=25) as response:
            need(response.status == 206 and response.headers.get("Content-Range") == f"bytes {span}/{self.size}",
                 "The endpoint did not return the exact bounded byte range")
            value = response.read(size + 1)
        need(len(value) == size, "Incomplete bounded HDF5 byte range")
        self.used += size
        self.receipts.append(dict(start=start, bytes=size, sha256=hashlib.sha256(value).hexdigest()))
        return value

    def readinto(self, target):
        end = min(self.position + len(target), self.size)
        need(end - self.position <= RANGE_CAP, "Single HDF5 read exceeds the registered bound")
        offset = 0
        while self.position < end:
            start = self.position // BLOCK * BLOCK
            chunk = self.block(start)
            available = min(len(chunk) - (self.position - start), end - self.position)
            target[offset:offset+available] = chunk[self.position-start:self.position-start+available]
            self.position += available
            offset += available
        return offset


def dojo(out):
    import h5py
    import numpy as np

    license_is_apache(REPO, REV)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}"
    candidates = []
    for task in SIM_TASKS:
        parent = f"data/RoboDojo/{task}/arx_x5/data"
        files = {Path(entry["path"]).name: entry for entry in metadata(f"{api}/{parent}?limit=1000")}
        for index in (0, 1):
            item = files[f"episode_{index:07d}.hdf5"]
            need(item["type"] == "file" and item["size"] == item["lfs"]["size"], "Unbound simulator LFS identity")
            candidates.append(dict(task=task, index=index, member=item["path"], bytes=item["size"],
                                   repository_lfs_sha256=item["lfs"]["oid"]))
    write(out / "ROSTER.json", dict(repo=REPO, revision=REV, license="Apache-2.0", official_sim_training=candidates))

    def sample(row):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row['member']}"
        with RangedMember(url, row["bytes"]) as stream:
            try:
                result = inspect_dojo(stream, expected_fps=25)
                with h5py.File(stream, "r") as source:
                    comparisons = {}
                    for arm in ("left", "right"):
                        values = source[f"state/{arm}_ee_poses"][:]
                        future = source[f"action/{arm}_ee_poses"][:]
                        comparisons[arm] = float(np.max(np.abs(values[1:] - future[:-1])))
                    result["next_frame_eef_max_abs"] = comparisons
            except (KeyError, OSError, ValueError) as error:
                result = dict(readability_failure=f"{type(error).__name__}: {error}")
            record = dict(row, actual_range_bytes=stream.used, ranges=stream.receipts, **result)
        write(out / f"{row['task']}-{row['index']}.json", record)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(sample, candidates))
    write(out / "ACTUAL.json", dict(repo=REPO, revision=REV, registered=len(candidates), results=results,
                                    actual_range_bytes=sum(row["actual_range_bytes"] for row in results),
                                    training_admitted=0))


def sc_vla(out, pilot=None):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    license_is_apache(SC_REPO, SC_REV)
    budget = 192 * 1024**2
    files = fetch(out, SC_REPO, SC_REV, ["README.md", "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"],
                  budget, actual_capacity=True)
    info = read(out / "meta/info.json")
    tasks = {row["task_index"]: row["task"] for row in map(json.loads, (out / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = list(map(json.loads, (out / "meta/episodes.jsonl").read_text().splitlines()))
    need(info["robot_type"] == "ARX5_follower" and info["fps"] == 15 and len(episodes) == info["total_episodes"],
         "SC-VLA follower metadata changed")
    already_read = set()
    if pilot is not None:
        initial = read(verify_receipt(pilot / "ACTUAL.json", SC_PILOT_PIN))
        verify_receipt(pilot / "MODALITY-READBACK.json", SC_MODALITY_PIN)
        already_read = {r["episode"] for r in initial["results"]}
    selected = []
    for task_id in (1, 3):
        available = [ep for ep in episodes if ep["tasks"] == [tasks[task_id]] and ep["episode_index"] not in already_read]
        count = 2 if pilot is None else 12
        if pilot is not None:
            available.sort(key=lambda ep: hashlib.sha256(f"SLA-V62-SC-WAVE1/{task_id}/{ep['episode_index']}".encode()).digest())
        need(len(available) >= count, "Each fixed skill needs the complete disjoint source cohort")
        selected.extend((row, task_id, "PILOT" if pilot is None else "TRAIN_CANDIDATE" if i < 10 else "DEV_CANDIDATE")
                        for i, row in enumerate(available[:count]))
    write(out / "ROSTER.json", dict(repo=SC_REPO, revision=SC_REV, excluded=sorted(already_read),
         registered=[dict(episode=r["episode_index"], task=task, split=split) for r, task, split in selected]))
    results = []
    for row, task, split in selected:
        ep, count = row["episode_index"], row["length"]
        layout = dict(episode_index=ep, episode_chunk=ep // info["chunks_size"])
        numeric = info["data_path"].format(**layout)
        camera = {name: info["video_path"].format(video_key=name, **layout) for name in SC_CAMERAS}
        files += fetch(out, SC_REPO, SC_REV, [numeric, *camera.values()], budget - sum(f["bytes"] for f in files),
                       actual_capacity=True)
        data = pq.read_table(out / numeric)
        value = {name: np.asarray(data[name].to_pylist()) for name in ("action", "observation.state", "timestamp", "frame_index", "task_index")}
        state, action, ts = value["observation.state"], value["action"], value["timestamp"].reshape(-1)
        need(state.shape == action.shape == (count, 13) and count >= 4 and np.isfinite(state).all()
             and np.isfinite(action).all() and np.array_equal(value["frame_index"].reshape(-1), np.arange(count))
             and np.all(value["task_index"] == task) and np.max(np.abs(ts - np.arange(count)/15)) < 1e-4,
             "SC-VLA time, episode, task or actual feature width differs")
        views = {}
        for name, path in camera.items():
            anchors, error, n = {}, 0., 0
            with av.open(out / path) as clip:
                for i, frame in enumerate(clip.decode(video=0)):
                    need(i < count and frame.pts is not None, "RGB stream exceeds sample or lacks PTS")
                    error = max(error, abs(float(frame.pts * frame.time_base) - ts[i]))
                    if i in (0, count // 2, count-1):
                        image = frame.to_ndarray(format="rgb24")
                        need(image.shape == (480, 640, 3), "SC-VLA RGB geometry differs")
                        anchors[i] = hashlib.sha256(image.tobytes()).hexdigest()
                    n += 1
            need(n == count and len(anchors) == 3 and error <= .5/15, "SC-VLA image timing mismatch")
            views[name] = dict(rgb_sha256=anchors, max_pts_error_s=error)
        results.append(dict(episode=ep, task=tasks[task], split=split, rows=count, views=views,
                            state_range=[state.min(0).tolist(), state.max(0).tolist()],
                            action_range=[action.min(0).tolist(), action.max(0).tolist()],
                            unnamed_nonzero=int(np.count_nonzero(state[:, 7:]) + np.count_nonzero(action[:, 7:])),
                            max_next_state_error=np.abs(action[:-1] - state[1:]).max(0).tolist()))
    write(out / "ACTUAL.json", dict(repo=SC_REPO, revision=SC_REV, native_fps=15, license="Apache-2.0",
         results=results, member_hashes=files, downloaded_bytes=sum(row["bytes"] for row in files),
         numeric_width=13, named_axes=info["features"]["action"]["names"], training_admitted=0))


def pair_dojo(out, pilot):
    import av
    import h5py
    import numpy as np
    from PIL import Image
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    raw_result = read(verify_receipt(pilot / "ACTUAL.json", DOJO_RAW_PIN))
    inventory = read(verify_receipt(pilot / "OFFICIAL-LE_ROBOT-INVENTORY.json", DOJO_META_PIN))
    inputs = {r["path"]: verify_receipt(pilot / r["path"], r["sha256"]) for r in inventory["file_set"]}
    table = pq.read_table(inputs[DOJO_CONVERTED + "/meta/episodes/chunk-000/file-000.parquet"])
    rows = {r["episode_index"]: r for r in table.select([k for k in table.column_names if not k.startswith("stats/")]).to_pylist()}
    info = read(inputs[DOJO_CONVERTED + "/meta/info.json"])
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}"
    resolve = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}"

    def member(relative):
        name = DOJO_CONVERTED + "/" + relative
        files = metadata(f"{api}/{Path(name).parent}?limit=1000")
        entry, = (row for row in files if row["path"] == name)
        need(entry["type"] == "file" and entry["size"] == entry["lfs"]["size"], "Unbound converted file")
        return RangedMember(f"{resolve}/{name}", entry["size"])

    def addressed(template, row, scope, **kw):
        return template.format(chunk_index=row[scope + "/chunk_index"], file_index=row[scope + "/file_index"], **kw)

    results = []
    for task, index in zip(SIM_TASKS, (2600, 1800), strict=True):
        raw, = (r for r in raw_result["results"] if r["task"] == task and r["index"] == 0)
        row = rows[index]
        n = row["length"]
        need(n == raw["rows"] and row["tasks"] == [raw["instruction"]] and info["fps"] == 25,
             "The pinned raw and official converted episode addresses differ")
        raw_bytes = RangedMember(f"{resolve}/{raw['member']}", raw["bytes"])
        data = member(addressed(info["data_path"], row, "data"))
        anchors = (0, n // 3, 2*n // 3, n-1)
        with raw_bytes, data, h5py.File(raw_bytes, "r") as original:
            raw_state, raw_action = tuple(np.column_stack([original[f"{group}/{arm}{field}"][:]
                                       for arm in ("left", "right") for field in ("_ee_poses", "_ee_joint_states")])
                                       for group in ("state", "action"))
            names = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            part = pq.read_table(data, filters=[("episode_index", "=", index)], columns=names, use_threads=False)
            a = {key: np.asarray(part[key].to_pylist()) for key in names}
            need(part.num_rows == n and np.all(a["episode_index"] == index)
                 and np.array_equal(a["frame_index"], np.arange(n)), "Converted numeric frames differ")
            clock = float(np.max(np.abs(a["timestamp"].reshape(-1) - np.arange(n)/25)))
            state_err = float(np.max(np.abs(a["observation.state"] - raw_state)))
            action_err = float(np.max(np.abs(a["action"] - raw_action)))
            views = {}
            for camera, feature in DOJO_VIDEO.items():
                expected = {}
                for i in anchors:
                    with Image.open(io.BytesIO(np.asarray(original[f"vision/{camera}/colors"][i]).tobytes())) as im:
                        expected[i] = np.asarray(im.convert("RGB"))
                field = "videos/" + feature
                source = member(addressed(info["video_path"], row, field, video_key=feature))
                start = row[field + "/from_timestamp"]
                pts, pixels = {}, {}
                with source, av.open(source, format="mp4") as container:
                    stream = container.streams.video[0]
                    container.seek(int(start/stream.time_base), stream=stream, backward=True)
                    for frame in container.decode(stream):
                        time = float(frame.pts * frame.time_base)
                        i = round((time - start)*25)
                        if i >= n:
                            break
                        if i in expected:
                            im = frame.to_ndarray(format="rgb24").astype(np.float32)
                            pixels[i] = float(np.abs(expected[i] - im).mean())
                            pts[i] = abs(time - start - i/25)
                        if len(pixels) == len(set(anchors)):
                            break
                need(pixels.keys() == expected.keys(), "Missing converted RGB anchors")
                views[feature] = dict(time_error_s=pts, mean_rgb_abs_error_8bit=pixels,
                                      read_bytes=source.used, ranges=source.receipts)
        results.append(dict(raw_task=task, raw_member=raw["member"], converted_episode=index, rows=n,
                            state_max_abs=state_err, action_max_abs=action_err, timestamp_error_s=clock,
                            native_source_bytes=raw_bytes.used, converted_numeric_bytes=data.used, views=views))
    write(out / "ACTUAL.json", dict(source_revision=REV, actual_pairs=results, new_train_episodes=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("dojo", "sc_vla", "dojo_pair", "sc_wave"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--raw-pilot", type=Path)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700)
    if args.domain in ("dojo_pair", "sc_wave"):
        need(args.raw_pilot is not None, "Bind the observed raw and converted input receipts")
        {"dojo_pair": pair_dojo, "sc_wave": sc_vla}[args.domain](args.out, args.raw_pilot)
    else:
        {"dojo": dojo, "sc_vla": sc_vla}[args.domain](args.out)
