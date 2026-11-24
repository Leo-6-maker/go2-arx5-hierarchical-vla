"""Test whether wrist landmarks can recover the recorded head camera across episodes."""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from source_custody import SOURCE_H5, array_hash, pin, require, write_new

SIDES = ("left", "right")
SIZE = (640, 480)
FRAME_STRIDE = 9
CALIBRATION_MEDIAN_PX = 2.0
CALIBRATION_P95_PX = 5.0
URDF_SHA = "13b4a47a689afb03afd3a5f08577f0afd79559bbf4193b25072b053d0c20737b"
POSE_CLOCK_OFFSETS = (-4, -2, -1, 0, 1, 2, 4)
# Initial orientation of an optical camera viewing forward from above the bench.
OPTICAL0 = np.array([[0.0, -.8, .6], [-1.0, 0.0, 0.0], [0.0, -.6, -.8]])
INITIAL = np.array([
    0., 0., 0., -.36, 0., .76, 440., 320., 240., 0., .6, 0.,
    .045, -.035, 0., .045, .035, 0., 0., 0.,
])
LOWER = np.array([
    -.9, -.9, -.9, -1.2, -.7, .25, 220., 260., 185., -.20, .4, -.15,
    -.08, -.13, -.09, -.08, -.13, -.09, -.5, -.3,
])
UPPER = np.array([
    .9, .9, .9, .40, .7, 1.7, 850., 380., 295., .20, .86, .15,
    .14, .13, .10, .14, .13, .10, .5, .3,
])


def tape_center(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    color = cv2.inRange(hsv, (120, 171, 65), (127, 255, 255)).astype(bool)
    mask = (color & (rgb[..., 0] < 73) & (rgb[..., 1] < 60)).astype(np.uint8)
    count, _, boxes, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = boxes[i]
        if (x >= 8 and x + width <= SIZE[0] - 8 and 45 <= y <= 385
                and 85 <= area <= 1800 and min(width, height) >= 7
                and area >= .42 * width * height):
            candidates.append(centers[i])
    return candidates


def extract(path):
    from XPolicyLab.utils.process_data import decode_image_bit

    rows, coverage = [], {}
    with h5py.File(path) as h:
        require(int(h["additional_info/frequency"][()]) == 30, "unexpected camera clock")
        poses = np.stack([h[f"state/{side}_ee_poses"][:] for side in SIDES], axis=1)
        rgb = h["vision/cam_head/colors"]
        require(len(poses) == len(rgb), "head/state length mismatch")
        for frame in range(0, len(rgb), FRAME_STRIDE):
            # The idle gripper is near the camera boundary. Single-arm
            # excitation provides an unambiguous side without reading the task.
            active = np.flatnonzero(poses[frame, :, 0] > .10)
            if len(active) != 1:
                reason = "NO_UNIQUE_ARM_EXCITATION"
            else:
                im = decode_image_bit(rgb[frame])
                require(im.dtype == np.uint8 and im.shape == (*SIZE[::-1], 3), "head RGB shape")
                candidates = tape_center(im)
                if len(candidates) == 1:
                    side = int(active[0])
                    rows.append(dict(frame=frame, side=SIDES[side], xy=candidates[0].tolist(),
                                     pose=poses[frame, side].tolist(), rgb_sha256=array_hash(im)))
                    reason = "OBSERVED"
                else:
                    reason = "MISSING_OR_AMBIGUOUS_TAPE"
            coverage[reason] = coverage.get(reason, 0) + 1
    return rows, coverage


def arrays(rows):
    poses = np.array([r["pose"] for r in rows])
    arms = np.array([SIDES.index(r["side"]) for r in rows])
    uv = np.array([r["xy"] for r in rows])
    return poses[:, :3], Rotation.from_quat(poses[:, [4, 5, 6, 3]]), arms, uv


def project(x, data, radial):
    pos, rot, arms, _ = data
    camera = Rotation.from_rotvec(x[:3]).as_matrix() @ OPTICAL0
    center, focal, principal = x[3:6], x[6], x[7:9]
    # Recorded poses share axes and have arm-local home translations. Fit
    # one relative arm-base translation and independent mounting landmarks.
    bases = np.stack((x[9:12] / 2, -x[9:12] / 2))
    landmark = pos + bases[arms] + rot.apply(x[12:18].reshape(2, 3)[arms])
    vec = (landmark - center) @ camera
    sensor = vec[:, :2] / np.maximum(.02, vec[:, 2, None])
    if radial:
        r2 = np.sum(sensor * sensor, axis=1)
        sensor *= (1. + x[18]*r2 + x[19]*r2**2)[:, None]
    return focal * sensor + principal, vec[:, 2]


def score(x, rows, radial, *, per_frame=True):
    data = arrays(rows)
    projection, depth = project(x, data, radial)
    delta = projection - data[-1]
    errors = np.linalg.norm(delta, axis=1)
    values = dict(
        n=len(rows), median_px=float(np.median(errors)),
        p95_px=float(np.percentile(errors, 95)), max_px=float(max(errors)),
        positive_depth=bool(np.min(depth) > 0),
    )
    if per_frame:
        values["rows"] = [
            dict(frame=row["frame"], side=row["side"], px=float(e), observed=row["xy"],
                 projected=uv.tolist(), rgb_sha256=row["rgb_sha256"])
            for row, e, uv in zip(rows, errors, projection)
        ]
    values["meets_pixel_gate"] = bool(
        values["median_px"] <= CALIBRATION_MEDIAN_PX
        and values["p95_px"] <= CALIBRATION_P95_PX and values["positive_depth"]
    )
    return values


def fit(rows, radial):
    data = arrays(rows)
    n = len(INITIAL) if radial else len(INITIAL)-2
    differences = lambda x: (project(x, data, radial)[0] - data[-1]).ravel()
    return least_squares(
        differences, INITIAL[:n], bounds=(LOWER[:n], UPPER[:n]), loss="soft_l1",
        f_scale=3, x_scale="jac", max_nfev=400,
    )


def read_joint6(urdf):
    joint = ET.parse(pin(urdf, URDF_SHA)).find("./joint[@name='joint6']")
    require(
        joint.find("parent").get("link") == "link5"
        and joint.find("child").get("link") == "link6",
        "unexpected wrist mounting links",
    )
    origin = joint.find("origin")
    return (
        np.fromstring(origin.get("xyz"), sep=" "),
        Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")),
        np.fromstring(joint.find("axis").get("xyz"), sep=" "),
    )


def relative_mount_controls(paths, episodes, urdf):
    # Preserve every pixel association. Pose lag and the adjacent mounting link
    # are varied without revising segmentation rules or replacing any frame.
    origin, parent_rotation, joint_axis = read_joint6(urdf)
    pose_series, joint_series = [], []
    for path in paths:
        with h5py.File(path) as h:
            pose_series.append({side: h[f"state/{side}_ee_poses"][:] for side in SIDES})
            joint_series.append({side: h[f"state/{side}_arm_joint_states"][:, 5] for side in SIDES})

    cases = [(6, lag) for lag in POSE_CLOCK_OFFSETS] + [(5, 0)]
    controls = []
    for link, lag in cases:
        pair = []
        for ep, poses, joints in zip(episodes, pose_series, joint_series):
            shifted = []
            for r in ep["rows"]:
                s, t = r["side"], r["frame"] + lag
                require(0 <= t < len(poses[s]), "pose lag leaves the original trace")
                pose = poses[s][t]
                if link == 5:
                    r6 = Rotation.from_quat(pose[[4, 5, 6, 3]])
                    r5 = r6 * (parent_rotation * Rotation.from_rotvec(joint_axis*joints[s][t])).inv()
                    xyz = pose[:3] - r5.apply(origin)
                    pose = np.r_[xyz, r5.as_quat()[[3, 0, 1, 2]]]
                shifted.append(dict(r, pose=pose.tolist()))
            pair.append(shifted)

        for i in (0, 1):
            result = fit(pair[i], False)
            controls.append(dict(
                mount_link=link, pose_frame_minus_rgb_frame=lag, fit_episode_index=i,
                convergence=bool(result.success), fit=score(result.x, pair[i], False, per_frame=False),
                validation=score(result.x, pair[1-i], False, per_frame=False),
            ))
    return controls


def run(root, out, urdf):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only recorded calibration")
    episodes, paths = [], []
    for name, sha in SOURCE_H5.items():
        paths.append(pin(root / "data" / name, sha))
        rows, coverage = extract(paths[-1])
        counts = {side: sum(r["side"] == side for r in rows) for side in SIDES}
        require(min(counts.values()) >= 12, "insufficient identifiable poses for two-arm fit")
        episodes.append(dict(episode=name, sha256=sha, coverage=coverage, per_arm=counts, rows=rows))
    trials = []
    for radial in (False, True):
        for train_index in (0, 1):
            tr, val = episodes[train_index], episodes[1-train_index]
            result = fit(tr["rows"], radial)
            trials.append(dict(
                model="PINHOLE_K1_K2" if radial else "PINHOLE",
                converged=bool(result.success), train_episode=tr["episode"], validation_episode=val["episode"],
                fit=score(result.x, tr["rows"], radial), validation=score(result.x, val["rows"], radial),
                bound_indices=np.flatnonzero(result.active_mask).tolist(),
                parameters=result.x.tolist(),
            ))
    controls = relative_mount_controls(paths, episodes, urdf)
    out.mkdir(mode=0o700)
    write_new(out / "SOURCE_HEAD_CALIBRATION_ACTUAL.json", dict(
        roster="TWO_SEA0_REAL_EPISODES", task_outcomes_used=False,
        frame_stride=FRAME_STRIDE, image_size=SIZE,
        heldout_median_threshold_px=CALIBRATION_MEDIAN_PX, heldout_p95_threshold_px=CALIBRATION_P95_PX,
        episodes=[{k: v for k, v in row.items() if k != "rows"} for row in episodes],
        models=trials, mount_and_clock_controls=controls, source_urdf_sha256=URDF_SHA,
        status="NATURAL_LANDMARK_REPROJECTION_DIAGNOSTIC",
        qualified_camera_packet=False, wrist_camera_calibrated=False,
        tool_gap_calibrated=False, sampler_forwards=0, physics_ticks=0, updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("urdf", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.out.resolve(), args.urdf.resolve())
