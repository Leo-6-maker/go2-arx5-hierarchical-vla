"""Verify the recorded EEF basis directly from synchronized ARX joint states."""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from source_custody import BASE, SOURCE_H5, digest, pin, read, require, write_new
from sea1_archive_readout import S0_PIN

REFERENCE = "13b4a47a689afb03afd3a5f08577f0afd79559bbf4193b25072b053d0c20737b"
REFERENCE_TOL = 1e-8
SIDES = ("left", "right")


class FlangeKinematics:
    def __init__(self, path):
        root = ET.parse(path)
        self.joints = []
        parent = "base_link"
        for i in range(1, 7):
            joint = root.find(f"./joint[@name='joint{i}']")
            require(joint.get("type") == "revolute" and joint.find("parent").get("link") == parent,
                    "six-axis ARX chain changed")
            parent = joint.find("child").get("link")
            origin = joint.find("origin")
            position = np.fromstring(origin.get("xyz"), sep=" ")
            rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            require(abs(np.linalg.norm(axis) - 1) < 1e-10, "expected a unit revolute axis")
            self.joints.append((position, rotation, axis))
        require(parent == "link6", "unexpected terminal EEF link")

    def forward(self, q):
        require(q.ndim == 2 and q.shape[1] == len(self.joints) and np.isfinite(q).all(), "invalid joint record")
        p = np.zeros((len(q), 3))
        r = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
        for i, (position, rotation, axis) in enumerate(self.joints):
            p += r @ position
            r = r @ rotation @ Rotation.from_rotvec(q[:, i, None] * axis).as_matrix()
        return p, r


def compare(p, r, observed):
    require(observed.shape == (len(p), 7) and np.isfinite(observed).all()
            and np.allclose(np.linalg.norm(observed[:, 3:], axis=1), 1., rtol=0, atol=1e-8),
            "unexpected recorded EEF representation")
    translation = np.linalg.norm(p - observed[:, :3], axis=1)
    relative = Rotation.from_matrix(r).inv() * Rotation.from_quat(observed[:, [4, 5, 6, 3]])
    angle = relative.magnitude()
    return dict(xyz_max_m=float(translation.max()), xyz_p95_m=float(np.percentile(translation, 95)),
                angle_max_rad=float(angle.max()), angle_p95_rad=float(np.percentile(angle, 95)),
                passes=bool(translation.max() <= REFERENCE_TOL and angle.max() <= REFERENCE_TOL))


def run(root, output):
    exact = root / "v19/sea0-optical-calibration-01/x5.urdf"
    alternative = root / "v19/sim-x5-assets/X5A.urdf"
    asset = read(BASE / "reports/SLA_GO2_V19/SEA0/SEA0_ROBOT_GEOMETRY.json")
    alt_sha = asset["source_assets"]["files"]["X5A.urdf"]["sha256"]
    flanges = {"recorded_fk_candidate": FlangeKinematics(pin(exact, REFERENCE)),
               "public_simulation_control": FlangeKinematics(pin(alternative, alt_sha))}
    zeros = {key: fk.forward(np.zeros((1, 6)))[0][0] for key, fk in flanges.items()}
    expected_zero = zeros["recorded_fk_candidate"]
    old = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_PIN))
    reg = root / "v12/TARGET_REGISTRATION.json"
    require(np.allclose(expected_zero, read(pin(reg, old["inputs"][str(reg)]))["source_origin"],
                        rtol=0, atol=1e-10), "historical Go2 origin differs from ARX zero-FK translation")
    results = []
    counts = 0
    for name, sha in SOURCE_H5.items():
        with h5py.File(pin(root / "data" / name, sha)) as h:
            require(int(h["additional_info/frequency"][()]) == 30, "unexpected source frame clock")
            side_scores = {}
            for side in SIDES:
                q = h[f"state/{side}_arm_joint_states"][:]
                observed = h[f"state/{side}_ee_poses"][:]
                score = {}
                for kind, kinematics in flanges.items():
                    p, r = kinematics.forward(q)
                    score[kind] = compare(p - zeros[kind], r, observed)
                p, r = flanges["recorded_fk_candidate"].forward(q)
                score["incorrect_no_zero_subtraction"] = compare(p, r, observed)
                score["incorrect_one_future_joint_frame"] = compare(p[1:] - expected_zero, r[1:], observed[:-1])
                require(score["recorded_fk_candidate"]["passes"]
                        and not any(row["passes"] for label, row in score.items()
                                    if label != "recorded_fk_candidate"), "direct FK or a negative control disagrees")
                side_scores[side] = dict(frames=len(q), joint_excursion_rad=np.ptp(q, axis=0).tolist(),
                                         scores=score)
                counts += len(q)
            results.append(dict(episode=name, sha256=sha, sides=side_scores))
    write_new(output, dict(
        status="RECORDED_EEF_DIRECT_LINK6_FK_VERIFIED", paired_arm_frames=counts,
        equation="p_recorded = p_base_link6(q) - p_base_link6(0); R_recorded = R_base_link6(q)",
        urdf_sha256=REFERENCE, simulation_control_sha256=alt_sha,
        zero_translation_m=expected_zero.tolist(), historical_target_registration_sha256=digest(reg),
        zero_translation_matches_existing_go2_observation_bridge=True, no_fitted_parameters=True,
        absolute_tolerance_m_rad=REFERENCE_TOL, episodes=results,
        physical_source_pad_gap=None, physical_source_flange_to_common_tool=None,
        source_world_mount_and_camera_extrinsics=None, model_forwards=0, physics_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU state comparison only")
    run(args.root.resolve(), args.output)
