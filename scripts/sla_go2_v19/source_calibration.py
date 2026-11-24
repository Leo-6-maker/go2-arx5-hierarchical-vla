"""Join camera and tool calibration to the exact SEA source recording frames."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pinhole_camera import PinholeCamera
from source_custody import (
    BASE, O0, SOURCE_H5, VIEWS, array_hash, digest, pin, read, real_source_observation, require, source_history, write_new,
)
from observation_stream import rigid
from policy_query import input_identity

ROSTER = BASE / "reports/SLA_GO2_V19/SEA0/SEA0_PAIRED_ROSTER.json"
SIDES = ("left", "right")
OPTICAL_FROM_RENDERER = np.diag([1., -1., -1., 1.])
SCHEMA = "SEA_SOURCE_RECORDING_CALIBRATION_V1"


class GripperCalibration:
    def __init__(self, measured):
        values = np.asarray(measured, dtype=float)
        require(values.ndim == 2 and values.shape[1:] == (2,) and len(values) >= 2
                and np.isfinite(values).all(), "expected measured gripper code/gap pairs")
        self.code, self.gap_m = values.T
        delta = np.diff(self.gap_m)
        require(np.all((0 <= self.code) & (self.code <= 1)) and np.all(np.diff(self.code) > 0)
                and np.all(self.gap_m >= 0) and (np.all(delta >= 0) or np.all(delta <= 0)),
                "gripper calibration must be monotonic over its measured range")

    def gap(self, code):
        # Preserve the fraction when outside the measurements, without extrapolating a jaw gap.
        return float(np.interp(code, self.code, self.gap_m)) if self.code[0] <= code <= self.code[-1] else None


def pose(row):
    q = np.asarray(row)
    require(q.shape == (7,) and np.isfinite(q).all() and abs(np.linalg.norm(q[3:])-1) <= 1e-6,
            "invalid recorded source end-effector pose")
    out = np.eye(4)
    out[:3, 3] = q[:3]
    out[:3, :3] = Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
    return out


def optics(c, recorded_shape):
    raw = PinholeCamera(**c["captured_intrinsics"])
    ideal = PinholeCamera(**c["rectified_intrinsics"])
    require(type(raw.width) is int and type(raw.height) is int
            and type(ideal.width) is int and type(ideal.height) is int,
            "camera rasters require integer dimensions")
    require((raw.height, raw.width, 3) == tuple(recorded_shape), "calibration and archived raster differ")
    coeff = np.asarray(c["brown_coefficients"], dtype=float)
    require(coeff.shape in ((4,), (5,), (8,)) and np.isfinite(coeff).all(), "unsupported lens coefficients")
    return dict(captured_intrinsics=vars(raw), rectified_intrinsics=vars(ideal), brown_coefficients=coeff.tolist())


def optical_pose(view, spec, frame, link6):
    kind = spec["kind"]
    if kind == "fixed_world":
        require(view == "cam_head", "a moving wrist requires frame or hand-eye calibration")
        return rigid(spec["world_from_optical"])
    if kind == "link6_mount":
        side = spec["side"]
        require(side in SIDES and view == f"cam_{side}_wrist", "camera hand-eye belongs to another arm")
        return link6[side] @ rigid(spec["link6_from_optical"])
    require(kind == "per_recorded_frame", "unsupported optical pose convention")
    require(str(frame) in spec["world_from_optical_by_frame"], "missing calibrated camera at requested frame")
    return rigid(spec["world_from_optical_by_frame"][str(frame)])


def calibrated_anchor(h, anchor, package, observation):
    require(package["archive_sha256"] == anchor["source_h5_sha256"], "calibration is for a different recording")
    require(set(package["arms"]) == set(SIDES) and set(package["cameras"]) == set(VIEWS),
            "expected both source arms and all three cameras")
    require(package["fps"] == h["additional_info/frequency"][()] == 30, "calibration recording clock changed")
    require(input_identity(observation)[0] == anchor["input_sha256"], "calibration requested for another observation")
    arms = package["arms"]
    transforms = {s: {k: rigid(arms[s][k]) for k in (
        "world_from_recorded_reference", "recorded_ee_from_link6", "world_from_robot_base",
    )} for s in SIDES}
    tcp = {s: None if arms[s].get("link6_from_tcp") is None else rigid(arms[s]["link6_from_tcp"]) for s in SIDES}
    grippers = {s: GripperCalibration(arms[s]["measured_gripper_code_gap_m"]) if
                arms[s].get("measured_gripper_code_gap_m") is not None else None for s in SIDES}
    camera_specs = package["cameras"]
    images = observation["images"]
    history = source_history(anchor["frame"])
    require(anchor["history_frames"] == history, "source calibration history indices differ from O0 cadence")
    wanted = [("cam_head", i, im) for i, im in zip(history, images["cam_high"], strict=True)]
    wanted.extend((name, anchor["frame"], images[name]) for name in VIEWS[1:])
    require(len(wanted) == 12 and all(0 <= i <= anchor["frame"] for _, i, _ in wanted),
            "missing or future source observation frame")
    lenses = {name: optics(camera_specs[name], h[f"vision/{name}/shape"][:]) for name in VIEWS}
    arm_frames, result = {}, []
    for view, frame, rgb in wanted:
        if frame not in arm_frames:
            row = {}
            for side, tr in transforms.items():
                local = pose(h[f"state/{side}_ee_poses"][frame])
                link6 = tr["world_from_recorded_reference"] @ local @ tr["recorded_ee_from_link6"]
                code = float(h[f"state/{side}_ee_joint_states"][frame, 0])
                joints = h[f"state/{side}_arm_joint_states"][frame]
                require(joints.shape == (6,) and np.isfinite(joints).all() and 0 <= code <= 1,
                        "missing measured joint or gripper state")
                row[side] = dict(world_from_link6=link6.tolist(),
                                 world_from_tcp=None if tcp[side] is None else (link6 @ tcp[side]).tolist(),
                                 joint_q=joints.tolist(), gripper_code=code,
                                 pad_gap_m=None if grippers[side] is None else grippers[side].gap(code))
            arm_frames[frame] = row
        link6 = {s: np.asarray(arm_frames[frame][s]["world_from_link6"]) for s in SIDES}
        t = optical_pose(view, camera_specs[view]["extrinsics"], frame, link6)
        result.append(dict(view=view, frame=frame, t_s=frame/30, rgb_sha256=array_hash(rgb),
                           world_from_optical=t.tolist(), world_from_renderer=(t @ OPTICAL_FROM_RENDERER).tolist()))
    return dict(id=anchor["id"], episode=anchor["episode"], input_sha256=anchor["input_sha256"],
                cameras=lenses, views=result, arm_frames=arm_frames,
                world_from_robot_base={s: transforms[s]["world_from_robot_base"].tolist() for s in SIDES})


def load_anchors(root, bundle, check=None):
    import h5py

    require(bundle["schema"] == SCHEMA and bundle["world_units"] == "m"
            and bundle["camera_axes"] == "OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD", "unsupported source calibration convention")
    require(set(bundle["episodes"]) == set(SOURCE_H5), "expected the two sealed source recordings")
    roster = read(ROSTER)["source"]
    require(len(roster) == 4, "sealed source population changed")
    receipts = {(z["episode"], z["call"]): z for z in read(O0)["source"]}
    resolved = []
    for episode, sha in SOURCE_H5.items():
        with h5py.File(pin(root / "data" / episode, sha)) as h:
            for a in (row for row in roster if row["episode"] == episode):
                key = (a["episode"], a["o0_call"])
                f = root / f"evidence/O0_PANEL_01/call_{a['o0_call']:02d}_input.json"
                observation = real_source_observation(h, receipts[key], read(pin(f, a["o0_input_receipt_sha256"])))
                cal = bundle["episodes"][episode]
                out = calibrated_anchor(h, a, cal, observation)
                if check is not None:
                    check(h, a, cal, observation, out)
                resolved.append(out)
    require(len(resolved) == len(roster), "source calibration anchor count changed")
    return resolved


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "bundle", "output"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU source calibration only")
    records = load_anchors(args.root, read(args.bundle))
    write_new(args.output, dict(scope="SOURCE_FRAME_CALIBRATION_JOIN", calibration_bundle_sha256=digest(args.bundle),
                                records=records, qualified_scene_pairs=0, native_ticks=0, sampler_forwards=0,
                                training_updates=0))
