"""Seal reviewed SC-VLA ARX5 episodes as a separate 15 Hz source domain."""

import argparse
import hashlib
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from v62_arx5_new_source_pilot import SC_MODALITY_PIN, SC_REPO, SC_REV
from v6_robodojo_intake import digest, read
from v6_ville_pilot import need, write
from v6_ville_reader import preflight


SCHEMA = "SLA_V62_SCVLA_ARX5_FUTURE_LOGGED_STATE_15HZ"
NATIVE_HZ = 15
HORIZON = 30
CAMERAS = {"observation.images.image": "base_0_rgb", "observation.images.wrist_image": "right_wrist_0_rgb"}
WAVE = "932dac5f5f5494e132ffa9c91549419c4be97fc28c6ca2e62b889fab8c6f9a51"
TASK = "Pick up the ball and place into the box"


def checked(path, sha):
    need(digest(path) == sha, f"Recorded source bytes changed: {path}")
    return path


def arrays_at_native_clock(observed, ts):
    ee, grip = observed[:, :6], observed[:, 6:7]
    need(np.all((-0.005 <= grip) & (grip <= 0.095)), "Logged source gripper is outside the audited scale")
    rot = Rotation.from_euler("xyz", ee[:, 3:]).as_matrix()
    count = len(ts) - HORIZON
    future = np.arange(count)[:, None] + np.arange(1, HORIZON + 1)
    relative = rot[future] @ rot[:count, None].swapaxes(-1, -2)
    rvec = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(count, HORIZON, 3)
    state = np.c_[ee[:count, :3], Rotation.from_matrix(rot[:count]).as_rotvec(), grip[:count]]
    actions = np.concatenate((ee[future, :3] - ee[:count, None, :3], rvec, grip[future]), axis=-1)
    return dict(state=state.astype(np.float32), actions=actions.astype(np.float32),
                mask=np.ones((count, HORIZON), dtype=bool), observation_timestamp=ts[:count], future_timestamp=ts[future])


def video_proof(video, ts, last_anchor, original):
    wanted = {0, last_anchor, len(ts) // 2, len(ts) - 1}
    hashes = {}
    with av.open(video) as clip:
        for index, frame in enumerate(clip.decode(video=0)):
            need(index < len(ts) and abs(float(frame.pts * frame.time_base) - ts[index]) < 1e-5,
                 "Camera and numerical clocks diverged")
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                need(rgb.shape == (480, 640, 3), "Native RGB dimensions changed")
                hashes[str(index)] = hashlib.sha256(rgb.tobytes()).hexdigest()
    need(len(hashes) == len(wanted) and all(hashes[k] == sha for k, sha in original.items()),
         "Native camera failed original sampled RGB readback")
    return {str(i): hashes[str(i)] for i in (0, last_anchor)}


def seal(root, modality, review, review_sha, output):
    scan = read(checked(root / "ACTUAL.json", WAVE))
    audit = read(checked(review, review_sha))
    axes = read(checked(modality, SC_MODALITY_PIN))["modalities"]["state"]
    need(axes == dict(position=dict(start=0, end=3), rotation=dict(start=3, end=6, rotation_type="euler_angles_rpy"),
                      gripper=dict(start=6, end=7)), "The pinned logged-state axes changed")
    need((scan["repo"], scan["revision"], scan["license"], scan["native_fps"], scan["training_admitted"])
         == (SC_REPO, SC_REV, "Apache-2.0", NATIVE_HZ, 0) and audit["scan_sha256"] == WAVE, "Wrong fixed cohort")
    roster = read(checked(root / "ROSTER.json", audit["roster_sha256"]))
    outcomes = {r["episode"]: r for r in audit["outcomes"]}
    planned = {r["episode"]: r["split"] for r in roster["registered"]}
    need(len(outcomes) == len(planned) == len(audit["outcomes"]) == len(scan["results"]) == 24
         and set(outcomes) == set(planned) == {r["episode"] for r in scan["results"]}
         and not set(planned).intersection(roster["excluded"]), "Outcome review omits registered or includes pilot episodes")
    members = {x["path"]: x["sha256"] for x in scan["member_hashes"]}
    info = read(checked(root / "meta/info.json", members["meta/info.json"]))
    output.mkdir(mode=0o700)
    episodes, train = [], []
    for row in scan["results"]:
        index = row["episode"]
        outcome = outcomes[index]
        need(outcome["task"] == row["task"] and planned[index] == row["split"] == outcome["split"],
             "Task or independent split differs from the registered population")
        if outcome["status"] != "GRASP_TRANSPORT_RELEASE_CONFIRMED":
            continue
        need(row["task"] == TASK and row["unnamed_nonzero"] == 0, "Incorrect task or unverified raw dimensions")
        layout = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
        numeric = info["data_path"].format(**layout)
        table = pq.read_table(checked(root / numeric, members[numeric]),
                              columns=["timestamp", "frame_index", "observation.state", "action"])
        ts = np.asarray(table["timestamp"], dtype=np.float64)
        observed = np.asarray(table["observation.state"].to_pylist())
        raw_action = np.asarray(table["action"].to_pylist())
        need(len(ts) == row["rows"] > HORIZON and np.array_equal(np.asarray(table["frame_index"]), np.arange(len(ts)))
             and np.allclose(ts, np.arange(len(ts)) / NATIVE_HZ, atol=1e-5, rtol=0)
             and observed.shape == raw_action.shape == (len(ts), 13)
             and all(np.isfinite(a).all() and not np.any(a[:, 7:]) for a in (observed, raw_action)),
             "Actual logged clock, support or raw axes changed")
        packed = arrays_at_native_clock(observed, ts)
        windows = len(packed["state"])
        split = {"TRAIN_CANDIDATE": "TRAIN", "DEV_CANDIDATE": "SOURCE_DEV"}[row["split"]]
        views = {}
        for camera in CAMERAS:
            member = info["video_path"].format(video_key=camera, **layout)
            anchors = video_proof(checked(root / member, members[member]), ts, windows - 1,
                                  row["views"][camera]["rgb_sha256"])
            views[camera] = dict(member=member, sha256=members[member], anchors=anchors, enabled=True)
        path = output / f"episode_{index:06d}.npz"
        with path.open("xb") as handle:
            np.savez(handle, **packed, raw_action=raw_action, raw_observed_state=observed)
        episodes.append(dict(episode=index, split=split, windows=windows, raw_numeric_sha256=members[numeric],
                             views=views, arrays=dict(file=path.name, sha256=digest(path))))
        if split == "TRAIN":
            train.append(packed)
    need(len(episodes) == 12 and len(train) == 10, "Reviewed Ball cohort count changed")
    need(len({row["raw_numeric_sha256"] for row in episodes}) == len(episodes), "Duplicate numeric source bytes")
    stats = {}
    for name in ("state", "actions"):
        values = np.concatenate([a[name].reshape(-1, 7) for a in train])
        lo, q25, q75, hi = np.quantile(values, [.01, .25, .75, .99], axis=0)
        stats[name] = dict(q01=lo.tolist(), q99=hi.tolist(), IQR=(q75-q25).tolist(), samples=len(values))
    statfile = output / "STATISTICS.json"
    write(statfile, dict(**stats, split="TRAIN", schema=SCHEMA))
    manifest = output / "MANIFEST.json"
    write(manifest, dict(schema=SCHEMA, repo=SC_REPO, revision=SC_REV, license="Apache-2.0", task=TASK,
                         scan_sha256=WAVE, modality_sha256=SC_MODALITY_PIN, review_sha256=review_sha,
                         native_hz=NATIVE_HZ, horizon=HORIZON, label_kind="FUTURE_LOGGED_STATE_REFERENCE",
                         target_command=False, video_root=str(root), episodes=episodes,
                         camera_mask=dict(base_0_rgb=True, left_wrist_0_rgb=False, right_wrist_0_rgb=True),
                         statistics=dict(file=statfile.name, sha256=digest(statfile))))
    preflight(output, digest(manifest), root, output / "CORE_READER.json", schema=SCHEMA,
              native_hz=NATIVE_HZ, horizon=HORIZON, cameras=CAMERAS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "modality", "review"):
        parser.add_argument(name, type=Path)
    parser.add_argument("review_sha")
    parser.add_argument("output", type=Path)
    seal(**vars(parser.parse_args()))
