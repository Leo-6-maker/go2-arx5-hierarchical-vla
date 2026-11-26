"""Verify real dual-arm arrays and video clocks for four fixed public RoboTwin samples."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

from v6_ville_pilot import fetch, need, read, write


REPO = "Traly/Robotwin2.0-lerobot"
REV = "7cd65fc4a41a6d04695abf5c8e31ff02285e1ac6"
ORIGINAL_REPO = "TianxingChen/RoboTwin2.0"
ORIGINAL_REV = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
CAMERAS = tuple("observation.images." + name for name in ("head", "left_wrist", "right_wrist"))
TASKS = ("adjust_bottle", "blocks_ranking_rgb")


def file_for(template, row, prefix, **extra):
    return template.format(chunk_index=row[prefix+"/chunk_index"], file_index=row[prefix+"/file_index"], **extra)


def video_four(path, start, count, anchors):
    import av
    import numpy as np

    hashes = {}
    largest = 0.
    with av.open(str(path)) as handle:
        stream = handle.streams.video[0]
        stream.thread_count = 1
        handle.seek(int(start / stream.time_base), stream=stream, any_frame=False, backward=True)
        for frame in handle.decode(stream):
            stamp = float(frame.pts * frame.time_base)
            index = int(round((stamp-start)*30))
            if index >= count:
                break
            if index in anchors:
                need(index not in hashes, "Two RGB frames match one episode anchor")
                image = frame.to_ndarray(format="rgb24")
                need(image.shape == (240,320,3), "Unverified camera geometry")
                hashes[index] = hashlib.sha256(image.tobytes()).hexdigest()
                largest = max(largest, abs(stamp - (start + index/30)))
    need(set(hashes) == set(anchors) and largest < .5/30, "Shared video offsets did not locate all four frames")
    return dict(rgb_sha256=hashes, max_pts_error_s=largest)


def inspect(root, selected, data, info):
    import numpy as np
    from scipy.spatial.transform import Rotation

    episode, count = selected["episode_index"], selected["length"]
    def array(name):
        return np.asarray(data[name].to_pylist())
    state, action = array("observation.state_ee"), array("action_ee")
    clock = array("timestamp").reshape(-1)
    need(data.num_rows == count and state.shape == action.shape == (count,16) and count >= 4
         and np.isfinite(state).all() and np.isfinite(action).all()
         and np.all(array("observation.embodiment_id") == [0,1,0,0,0])
         and np.array_equal(array("frame_index").reshape(-1), np.arange(count))
         and np.max(abs(clock - np.arange(count)/30)) < .0001,
         "The sample did not resolve to a native 30 Hz ARX-X5 episode")
    err = float(np.max(np.abs(action[:-1]-state[1:])))
    need(err < 1e-6, "Stored action differs from the documented next-frame EEF")
    anchors = [0, count//3, 2*count//3, count-1]
    motions = []
    for offset in (0,8):
        q = state[:,offset+3:offset+7]
        normalization_error = float(np.max(abs(np.linalg.norm(q,axis=1)-1)))
        need(normalization_error < 1e-4 and np.min(state[:,offset+7]) >= -1e-6
             and np.max(state[:,offset+7]) <= 1+1e-6, "Invalid source pose or unit gripper")
        rot = Rotation.from_quat(q[:,[1,2,3,0]])
        motion = (rot[1:] * rot[:-1].inv()).as_rotvec()
        need(np.isfinite(motion).all(), "Invalid source-frame rotation conversion")
        motions.append(dict(quat_norm_error=normalization_error, max_step_rotation_rad=float(np.linalg.norm(motion,axis=1).max()),
                            grip_min=float(np.min(state[:,offset+7])), grip_max=float(np.max(state[:,offset+7]))))
    frames = {}
    for cam in CAMERAS:
        key = "videos/" + cam
        start, end = (selected[key+"/"+k] for k in ("from_timestamp", "to_timestamp"))
        need(start is not None and abs(end-start-count/30) < .0001, "The declared shared-video slice changed")
        path = root / file_for(info["video_path"], selected, key, video_key=cam)
        frames[cam] = video_four(path,start,count,anchors)
    return dict(episode=episode, task=selected["tasks"][0], native_rows=count, real_embodiment="arx-x5",
                source_next_frame_max_error=err, raw_quat_order="wxyz", grip_semantics="unit fraction 1=open",
                arm_audits=motions, declared_camera_focus=frames,
                fixed_H50_temporal_support={anchor:min(50,count-anchor-1) for anchor in anchors},
                candidate_action_dims=14, absent_base_masked=True, terminal_duplicated_row_usable=False,
                status="ACTUAL_NEXT_FRAME_AND_RGB_CHECKED_PROVENANCE_PENDING")


def run(meta, out):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    out.mkdir(mode=0o700)
    info = read(meta/"meta/info.json")
    declared = info["robotwin_meta"]
    need(info["fps"] == 30 and declared["embodiment_id_order"][1] == "arx-x5"
         and declared["variant"] == "randomized_500" and "wxyz" in declared["state_encoding"]["observation.state_ee"]["space"],
         "The actual pinned converter first needs a new contract")
    groups = defaultdict(list)
    metadata_report = read(meta/"FIRST_META.json")
    for entry in metadata_report["meta"]:
        source = meta / entry["path"]
        for row in pq.read_table(source).to_pylist():
            task = row["tasks"]
            if len(task) != 1 or not task[0].startswith("[arx-x5] "):
                continue
            name = task[0].split(":")[0].split(" ", 1)[1]
            if name in TASKS:
                groups[name].append(row)
    chosen = []
    for task in TASKS:
        grouped = defaultdict(list)
        for row in groups[task]:
            key = (file_for(info["data_path"], row, "data"), *(file_for(info["video_path"], row, "videos/"+cam, video_key=cam)
                   for cam in CAMERAS))
            grouped[key].append(row)
        match = next((option for option in grouped.values() if len(option) >= 2), None)
        need(match is not None, "No predeclared two-row group shares bounded assets")
        chosen.extend(match[:2])
    names = sorted({name for row in chosen for name in (
        file_for(info["data_path"], row, "data"), *(file_for(info["video_path"], row, "videos/"+cam, video_key=cam)
                  for cam in CAMERAS))})
    write(out/"ROSTER.json",dict(repo=REPO,revision=REV,episodes=[r["episode_index"] for r in chosen],files=names))
    files = fetch(out,REPO,REV,names,2*1024**3,member_limit=250*1024**2)
    results = []
    for row in chosen:
        data = pq.read_table(out / file_for(info["data_path"],row,"data"))
        episode = row["episode_index"]
        subset = data.filter(pc.equal(data["episode_index"],episode))
        results.append(inspect(out,row,subset,info))
    write(out/"ACTUAL.json",dict(repo=REPO,revision=REV,license="Apache-2.0",original_repo=ORIGINAL_REPO,
                               original_revision=ORIGINAL_REV,original_repo_license="MIT",variant="randomized_500",
                               decoded_episodes=results,member_hashes=files,downloaded_bytes=sum(r["bytes"] for r in files),
                               new_train_episodes=0,new_native_requests=0,new_optimizer_updates=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("meta",type=Path)
    p.add_argument("out",type=Path)
    run(**vars(p.parse_args()))
