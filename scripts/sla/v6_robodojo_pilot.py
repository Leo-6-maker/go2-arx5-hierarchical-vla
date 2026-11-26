"""Check four fixed out-of-R200 real ARX-X5 episodes using the public HDF5 schema."""

import argparse
import hashlib
import io
from pathlib import Path

from v6_ville_pilot import fetch, need, write


REPO = "RoboDojo-Benchmark/RoboDojo"
REV = "1127bda94f172d757386945af3eb3a6754d1a4e6"
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SAFE_ROSTER = "476dbe0e03ed309170ba76a333f19ff1b1fc18a7e2ac3bafff4bbff1696ec496"


def inspect(path, *, expected_fps=30):
    import h5py
    import numpy as np
    from PIL import Image

    with h5py.File(path) as h:
        fps = int(h["additional_info/frequency"][()])
        arms = ("left","right")
        packed = {}
        for section in ("state","action"):
            for arm in arms:
                key = section + "/" + arm
                packed[key] = tuple(np.asarray(h[key+name]) for name in
                                    ("_arm_joint_states","_ee_joint_states","_ee_poses"))
        count = len(packed["state/left"][0])
        need(fps == expected_fps and count >= 8 and all(
            tuple(a.shape for a in arrays) == ((count,6),(count,1),(count,7))
            and all(np.isfinite(x).all() for x in arrays) for arrays in packed.values()), "Unexpected real ARX5 ABI")
        indexes = (0,count//3,2*count//3,count-1)
        images = {}
        for cam in CAMERAS:
            stream = h["vision/"+cam+"/colors"]
            shape = tuple(h["vision/"+cam+"/shape"][:])
            need(len(stream) == count and shape[-1] == 3, "A camera stream lost a row")
            for anchor in indexes:
                encoded = np.asarray(stream[anchor])
                with Image.open(io.BytesIO(encoded.tobytes())) as photo:
                    rgb = np.asarray(photo.convert("RGB"))
                need(rgb.shape == shape, "RGB decode differs from the public shape")
                images.setdefault(cam,{})[str(anchor)] = hashlib.sha256(rgb.tobytes()).hexdigest()
        pose = {}
        for arm in arms:
            s = packed["state/"+arm]
            a = packed["action/"+arm]
            pose[arm] = dict(state_unit_quat_max_error=float(np.max(abs(np.linalg.norm(s[2][:,3:],axis=1)-1))),
                action_unit_quat_max_error=float(np.max(abs(np.linalg.norm(a[2][:,3:],axis=1)-1))),
                state_gripper_range=[float(s[1].min()),float(s[1].max())],
                action_gripper_range=[float(a[1].min()),float(a[1].max())],
                mean_position_state_action_difference_m=float(np.linalg.norm(s[2][:,:3]-a[2][:,:3],axis=1).mean()))
        return dict(rows=count,native_fps=fps,decoder_camera_order=CAMERAS,
                    h5_format=h["data_format_version"][()].decode(),instruction=h["instruction"][()].decode(),
                    source_pose_audit=pose,genuine_rgb_samples=images,
                    candidate_H50_available_rows={k:min(count-k,50) for k in indexes},
                    exact_timestamps_exported="timestamp" in h,role="PUBLIC_SOURCE_SAMPLE")


def run(out, approved_roster):
    from v6_ville_pilot import read

    need(hashlib.sha256(approved_roster.read_bytes()).hexdigest() == SAFE_ROSTER,
         "An independently checked train/dev/test exclusion roster is required before reading samples")
    allowed = read(approved_roster)
    wanted = []
    for task in ("cover_blocks", "insert_tubes"):
        wanted.extend([x for x in allowed["candidates"] if x["task"] == task][:2])
    protected = {x["sha256"] for x in allowed["protected_test_members"]}
    need(len(wanted) == 4 and not protected.intersection(x["sha256"] for x in wanted),
         "Pilot contains a protected member")
    out.mkdir(mode=0o700)
    names = [row["member"] for row in wanted]
    write(out/"ROSTER.json",dict(repo=REPO,revision=REV,paths=names,
                                exclusion_roster_sha256=SAFE_ROSTER))
    files = fetch(out,REPO,REV,names,1024**3,member_limit=300*1024**2)
    records = []
    for member, row in zip(wanted,files,strict=True):
        need(row["sha256"] == member["sha256"], "Raw pilot member hash changed")
        task, index = member["task"], member["index"]
        record = dict(task=task,index=index,file=row,**inspect(out/row["path"]))
        write(out/f"{task}-{index:07d}.json",record)
        records.append(record)
    write(out/"ACTUAL.json",dict(repo=REPO,revision=REV,license="Apache-2.0",records=records,
                                downloaded_bytes=sum(x["bytes"] for x in files),new_train_episodes=0,
                                requires_canonical_codec_and_split=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("approved_roster", type=Path)
    args = parser.parse_args()
    run(args.out, args.approved_roster)
