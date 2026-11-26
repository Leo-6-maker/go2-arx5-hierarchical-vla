"""Seal outcome-reviewed single-arm future references at their native 20 Hz."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from v6_robodojo_intake import digest, read
from v6_ville_pilot import CAMERAS, SOURCES, need, write


SCHEMA = "SLA_V6_VILLE_ARX5_FUTURE_LOGGED_STATE_20HZ"
POSITIVE = frozenset({88, 50, 2, 23, 84, 36, 87, 16, 83, 28, 14, 15})
DEV = frozenset({14, 15})
HORIZON = 40


def seal(root, scan_sha, fk_receipt, fk_sha, output):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    actual = root / "ACTUAL.json"
    need(digest(actual) == scan_sha and digest(fk_receipt) == fk_sha, "Fixed outcome cohort changed")
    scan = read(actual)
    fk = read(fk_receipt)
    _, revision = SOURCES[1]
    need(scan["repo"] == "villekuosmanen/drop_footbag_into_dice_tower"
         and scan["revision"] == revision and scan["source_train_added"] == 0, "Source revision changed")
    proof = {x["episode"]: x for x in fk["episodes"] if x["fit_role"] != "FIT"}
    rows = {x["episode"]: x for x in scan["scanned"]}
    role = {i: "SOURCE_DEV" if i in DEV else "TRAIN" for i in POSITIVE}
    need(set(rows) == POSITIVE and {i for i in rows if rows[i]["proposed_split"] == "DEV_CANDIDATE"} == DEV
         and all(i in proof and proof[i]["fit_role"] in ("TRAIN_CANDIDATE", "DEV_CANDIDATE") for i in rows),
         "Original fixed episode-level outcome population changed")
    member = {x["path"]: x["sha256"] for x in scan["files"]}
    output.mkdir(mode=0o700)
    episodes = []
    fits = []
    for index, metadata in rows.items():
        key = f"episode_{index:06d}"
        parquet = f"data/chunk-000/{key}.parquet"
        need(digest(root / parquet) == member[parquet], "Actual numeric file changed")
        table = pq.read_table(root / parquet,
            columns=["observation.eef_6d_pose", "observation.state", "timestamp", "action"])
        ee = np.asarray(table["observation.eef_6d_pose"].to_pylist())
        q = np.asarray(table["observation.state"].to_pylist())
        raw_action = np.asarray(table["action"].to_pylist())
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        count = len(ts)
        need(ee.shape == (count, 6) and q.shape == raw_action.shape == (count, 7)
             and all(np.isfinite(x).all() for x in (ee, q, raw_action)) and count == metadata["rows"] > HORIZON
             and np.max(np.abs(ts - np.arange(count) / 20)) < 1e-5, "Unexpected measured source frame")
        rot = Rotation.from_euler("xyz", ee[:, 3:]).as_matrix()
        anchors = count - HORIZON
        future = np.arange(anchors)[:, None] + 1 + np.arange(HORIZON)
        relative = rot[future] @ rot[:anchors, None].swapaxes(-1, -2)
        rvec = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(anchors, HORIZON, 3)
        states = np.c_[ee[:anchors, :3], Rotation.from_matrix(rot[:anchors]).as_rotvec(), q[:anchors, 6]].astype(np.float32)
        action = np.concatenate((ee[future, :3] - ee[:anchors, None, :3],
                                 rvec, q[future, 6:7]), axis=-1).astype(np.float32)
        need((q[:, 6] >= -0.005).all() and (q[:, 6] <= .095).all(), "Unexpected observed gripper scale")
        if role[index] == "TRAIN":
            fits.append((states, action))
        checks = {}
        for camera in CAMERAS:
            video = f"videos/chunk-000/{camera}/{key}.mp4"
            need(digest(root / video) == member[video], "Actual video bytes changed")
            wanted = {0, anchors - 1}
            observed = {}
            with av.open(root / video) as clip:
                for k, frame in enumerate(clip.decode(video=0)):
                    if k in wanted:
                        rgb = frame.to_ndarray(format="rgb24")
                        need(rgb.shape == (480, 640, 3) and abs(float(frame.pts * frame.time_base) - ts[k]) < 1e-5,
                             "Observed RGB does not match native frame clock")
                        observed[str(k)] = hashlib.sha256(rgb.tobytes()).hexdigest()
            need(len(observed) == 2, "Both actual RGB anchors are required")
            need(observed["0"] == metadata["cameras"][camera]["selected_rgb_sha256"]["0"], "Initial camera hash changed")
            checks[camera] = dict(member=video, sha256=member[video], anchors=observed, enabled=True)
        packed = output / f"{key}.npz"
        with packed.open("xb") as handle:
            np.savez(handle, state=states, actions=action, mask=np.ones((anchors, HORIZON), dtype=bool),
                     observation_timestamp=ts[:anchors], future_timestamp=ts[future],
                     raw_action=raw_action, observed_pose=ee)
        episodes.append(dict(episode=index, split=role[index], windows=anchors, previewed_positive_rgb=True,
             raw_numeric_sha256=member[parquet], source_codec=SCHEMA, native_hz=20, views=checks,
             arrays=dict(file=packed.name, sha256=digest(packed))))
    stats = {}
    for name, values in (
        ("state", np.concatenate([s for s, _ in fits])),
        ("actions", np.concatenate([a.reshape(-1, 7) for _, a in fits])),
    ):
        q = np.quantile(values, [.01, .25, .75, .99], axis=0)
        stats[name] = dict(q01=q[0].tolist(), q99=q[3].tolist(), IQR=(q[2]-q[1]).tolist(), samples=len(values))
    statfile = output / "STATISTICS.json"
    write(statfile, dict(**stats, split="TRAIN", schema=SCHEMA))
    manifest = output / "MANIFEST.json"
    write(manifest, dict(schema=SCHEMA, license="Apache-2.0", revision=revision, scan_sha256=scan_sha,
        fk_check_sha256=fk_sha, native_hz=20, horizon=HORIZON, task="drop the footbag into the cardboard dice tower",
        label_kind="FUTURE_LOGGED_STATE_REFERENCE", target_command=False, video_root=str(root),
        camera_mask=dict(base_0_rgb=True, left_wrist_0_rgb=False, right_wrist_0_rgb=True),
        statistics=dict(file=statfile.name, sha256=digest(statfile)), episodes=episodes))
    write(output / "ACTUAL.json", dict(manifest_sha256=digest(manifest), samples_by_split=dict(Counter(role.values())),
         windows_by_split={name: sum(row["windows"] for row in episodes if row["split"] == name)
                           for name in ("TRAIN", "SOURCE_DEV")},
         release_scope="CORE_STANDALONE_SOURCE", hopper_readback=False, calibrated_robot_commands=False,
         demo_outcomes="HUMAN_REVIEW_STYLE_FIXED_RGB_PHASES", optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("scan_sha")
    parser.add_argument("fk_receipt", type=Path)
    parser.add_argument("fk_sha")
    parser.add_argument("output", type=Path)
    seal(**vars(parser.parse_args()))
