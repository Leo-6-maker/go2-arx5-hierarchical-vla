"""Exercise calibration intake on the four frozen real captures with known fixture transforms."""

import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lens_remap import LensRemap
from pinhole_camera import PinholeCamera
from source_calibration import ROSTER, SCHEMA, calibrated_anchor, load_anchors
from source_custody import SOURCE_H5, VIEWS, array_hash, digest, read, require, write_new


def transform(xyz, degrees):
    value = np.eye(4)
    value[:3, 3] = xyz
    value[:3, :3] = Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()
    return value.tolist()


def frame_camera(frame):
    return transform([.35, -.21, .81 + .0005 * frame], [26, -27, 17])


def fixture_bundle():
    result = {}
    anchors = read(ROSTER)["source"]
    for i, (episode, sha) in enumerate(SOURCE_H5.items()):
        frames = sorted({f for a in anchors if a["episode"] == episode for f in a["history_frames"]})
        sensor = dict(width=640, height=480, fx=530., fy=520., cx=309.3, cy=223.9)
        idealk = dict(sensor, fx=435., fy=425.)
        cams = {v: dict(captured_intrinsics=sensor, rectified_intrinsics=idealk,
                        brown_coefficients=[-.10, .02, .001, -.002, .004]) for v in VIEWS}
        cams["cam_head"]["extrinsics"] = (
            dict(kind="fixed_world", world_from_optical=frame_camera(0)) if i == 0 else
            dict(kind="per_recorded_frame", world_from_optical_by_frame={str(f): frame_camera(f) for f in frames})
        )
        arms = {}
        for side, y in (("left", .3), ("right", -.3)):
            cams[f"cam_{side}_wrist"]["extrinsics"] = dict(
                kind="link6_mount", side=side, link6_from_optical=transform([.03, .04, .05], [24, 21, -35]))
            arms[side] = dict(
                world_from_recorded_reference=transform([.1, y, .42], [3, 7, 19]),
                recorded_ee_from_link6=transform([.015, .001, .004], [-2, -9, 31]),
                world_from_robot_base=transform([-.12, y+.04, .50], [0, 0, 19]),
                link6_from_tcp=transform([.16, .007, -.005], [0, 1, -2]),
                measured_gripper_code_gap_m=[[.10, .005], [.64, .043], [.9, .081]],
            )
        result[episode] = dict(archive_sha256=sha, fps=30, arms=arms, cameras=cams)
    return dict(schema=SCHEMA, world_units="m", camera_axes="OPENCV_X_RIGHT_Y_DOWN_Z_FORWARD", episodes=result)


def independent_pose(data):
    xyz, q = np.asarray(data[:3]), np.asarray(data[3:])
    return xyz, Rotation.from_quat(np.roll(q, -1)).as_matrix()


def verify_join(h, anchor, package, observation, out):
    keys = [("cam_head", j) for j in anchor["history_frames"]] + [(v, anchor["frame"]) for v in VIEWS[1:]]
    require([(v["view"], v["frame"]) for v in out["views"]] == keys, "unexpected source view order")
    images = observation["images"]
    ordered = [*images["cam_high"], images["cam_left_wrist"], images["cam_right_wrist"]]
    require(all(row["rgb_sha256"] == array_hash(img) and row["t_s"] == tick/30
                for row, img, (_, tick) in zip(out["views"], ordered, keys, strict=True)), "captured frame bytes shifted")
    maximum = 0.
    for row in out["views"]:
        t, v = row["frame"], row["view"]
        x = np.asarray(row["world_from_optical"])
        if v == "cam_head":
            fixed = package["cameras"][v]["extrinsics"]["kind"] == "fixed_world"
            reference = np.asarray(frame_camera(0 if fixed else t))
        else:
            side = v.split("_")[1]
            arm = package["arms"][side]
            rig = np.asarray(arm["world_from_recorded_reference"])
            flange = np.asarray(arm["recorded_ee_from_link6"])
            cam = np.asarray(package["cameras"][v]["extrinsics"]["link6_from_optical"])
            xyz, r = independent_pose(h[f"state/{side}_ee_poses"][t])
            # Compose the translation and axes independently from the matrix-chain loader.
            reference = np.eye(4)
            end_rotation = rig[:3, :3] @ r
            moving = end_rotation @ flange[:3, :3]
            location = rig[:3, 3] + rig[:3, :3] @ xyz + end_rotation @ flange[:3, 3]
            reference[:3, 3] = location + moving @ cam[:3, 3]
            reference[:3, :3] = moving @ cam[:3, :3]
            tcp = np.asarray(out["arm_frames"][t][side]["world_from_tcp"])
            offset = np.asarray(arm["link6_from_tcp"])
            require(np.allclose(tcp[:3, 3], location + moving @ offset[:3, 3], rtol=0, atol=1e-12)
                    and np.allclose(tcp[:3, :3], moving @ offset[:3, :3], rtol=0, atol=1e-12),
                    "tool pose used a different recorded arm frame")
        rendered = np.asarray(row["world_from_renderer"])
        require(np.array_equal(rendered[:3, 3], x[:3, 3]) and np.array_equal(rendered[:3, 2], -x[:3, 2])
                and np.array_equal(rendered[:3, 1], -x[:3, 1]), "renderer and optical axes differ")
        maximum = max(maximum, float(np.max(abs(x-reference))))
    require(maximum < 1e-12, "same-frame camera geometry altered")
    return maximum


def refusals(h, anchor, cal, obs):
    expected = {
        "wrong_archive": "calibration is for a different recording",
        "wrong_history": "source calibration history indices differ from O0 cadence",
        "wrong_raster": "calibration and archived raster differ",
        "wrong_arm": "camera hand-eye belongs to another arm",
        "missing_frame": "missing calibrated camera at requested frame",
        "reflected_world": "invalid measured or calibrated rigid transform",
    }
    found = {}
    for name, message in expected.items():
        a, p = deepcopy(anchor), deepcopy(cal)
        head = p["cameras"]["cam_head"]
        if name == "wrong_archive":
            p["archive_sha256"] = "0" * 64
        elif name == "wrong_history":
            a["history_frames"][1] = a["frame"] + 1
        elif name == "wrong_raster":
            head["captured_intrinsics"]["width"] = 639
        elif name == "wrong_arm":
            p["cameras"]["cam_left_wrist"]["extrinsics"]["side"] = "right"
        elif name == "missing_frame":
            head["extrinsics"] = dict(kind="per_recorded_frame", world_from_optical_by_frame={})
        else:
            p["arms"]["right"]["world_from_robot_base"] = np.diag([-1, 1, 1, 1]).tolist()
        try:
            calibrated_anchor(h, a, p, obs)
        except ValueError as exc:
            require(str(exc) == message, "calibration was refused at a different boundary")
            found[name] = message
        else:
            raise ValueError("invalid source calibration accepted")
    return found


def run(root, out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU source calibration intake only")
    out.mkdir(mode=0o700)
    bundle_path = out / "FIXTURE_CALIBRATION.private.json"
    write_new(bundle_path, fixture_bundle())
    errors, rejected = [], {}

    def check(h, a, p, obs, record):
        errors.append(verify_join(h, a, p, obs, record))
        if not rejected:
            rejected.update(refusals(h, a, p, obs))
            missing = deepcopy(p)
            missing["arms"]["right"]["link6_from_tcp"] = None
            missing["arms"]["right"]["measured_gripper_code_gap_m"] = None
            resolved = calibrated_anchor(h, a, missing, obs)
            require(resolved["views"] == record["views"] and all(
                row["right"]["world_from_tcp"] is None and row["right"]["pad_gap_m"] is None
                for row in resolved["arm_frames"].values()), "unknown tool blocked or fabricated camera data")

    records = load_anchors(root, read(bundle_path), check)
    cams = records[0]["cameras"]
    remap_counts = {}
    for name, c in cams.items():
        lens = LensRemap(PinholeCamera(**c["rectified_intrinsics"]), PinholeCamera(**c["captured_intrinsics"]),
                         c["brown_coefficients"])
        remap_counts[name] = int(lens.valid.sum())
    rows = [dict(id=r["id"], input_sha256=r["input_sha256"], frames=[v["frame"] for v in r["views"]],
                 unique_pose_frames=len(r["arm_frames"]), uncalibrated_fixture_gap_samples=sum(
                     v["pad_gap_m"] is None for state in r["arm_frames"].values() for v in state.values()),
                 independent_transform_max_error=error) for r, error in zip(records, errors, strict=True)]
    write_new(out / "SOURCE_CALIBRATION_INTAKE_ACTUAL.json", dict(
        scope="KNOWN_FIXTURE_GEOMETRY_ON_SEALED_RECORDING_BYTES", fixture_sha256=digest(bundle_path),
        source_index_clock_hz=30, rows=rows, supported_lens_pixels=remap_counts, refused=rejected,
        absent_tool_kept_unknown=True, qualified_source_cameras=0, qualified_scene_pairs=0,
        sampler_forwards=0, native_ticks=0, updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "output"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
