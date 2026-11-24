"""Compare the pinned source X5 URDF, target MJCF and measured target states without fitting."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reach_intent_replay import BASE, read, require, pin, standing
from world_intent_clock import h40_fraction_to_width, sensed_width_to_state


def parse_v(node, name, default=None):
    return np.fromstring(node.get(name, default), sep=" ")


def chains(urdf, mjcf):
    raw = ET.parse(urdf)
    # MuJoCo accepts embedded "--" in comments, outside the generic XML grammar.
    model = ET.fromstring(re.sub(rb"<!--.*?-->", b"", mjcf.read_bytes(), flags=re.S))
    src, target = [], []
    limb = model.find(".//body[@name='x5_base_link']")
    for n in range(1, 7):
        joint = raw.find(f".//joint[@name='joint{n}']")
        org = joint.find("origin")
        src.append(
            (
                parse_v(org, "xyz"),
                Rotation.from_euler("xyz", parse_v(org, "rpy")),
                parse_v(joint.find("axis"), "xyz"),
            )
        )
        limb = limb.find(f"./body[@name='x5_link{n}']")
        require(
            not set(limb.attrib) & {"quat", "axisangle", "euler", "xyaxes", "zaxis"},
            "unhandled target limb rotation",
        )
        j = limb.find(f"./joint[@name='x5_joint{n}']")
        require(
            np.all(parse_v(j, "pos", "0 0 0") == 0) and float(j.get("ref", "0")) == 0,
            "unhandled target joint origin",
        )
        target.append(
            (parse_v(limb, "pos", "0 0 0"), Rotation.identity(), parse_v(j, "axis"))
        )
    names = [
        e.get("name")
        for e in model.find("worldbody").iter()
        if e.tag in ("joint", "freejoint")
    ]
    require(
        names[:1] == [None] and names[13:19] == [f"x5_joint{k}" for k in range(1, 7)],
        "unexpected native joint ABI",
    )
    site = limb.find("./site[@name='x5_ee']")
    require(
        not set(site.attrib) & {"quat", "axisangle", "euler", "xyaxes", "zaxis"},
        "unhandled target TCP rotation",
    )
    return src, target, parse_v(site, "pos")


def forward(chain, values):
    q = np.asarray(values)
    p = np.zeros((len(q), 3))
    r = Rotation.identity(len(q))
    axes = []
    for j, (shift, fixed, local_axis) in enumerate(chain):
        p += r.apply(shift)
        r *= fixed
        axes.append(r.apply(local_axis))
        r *= Rotation.from_rotvec(q[:, j, None] * local_axis)
    return p, r, np.stack(axes, axis=1)


def run(root, mjcf, source):
    model_sha = "b306e99adfb4f5e8af73649df2818ac14472395dadd1ac0b1db1c8473bbe8501"
    s, t, tool = chains(
        pin(source, "13b4a47a689afb03afd3a5f08577f0afd79559bbf4193b25072b053d0c20737b"),
        pin(mjcf, model_sha),
    )
    q0 = np.zeros((1, 6))
    zerop_s, zero_s, axes_s = forward(s, q0)
    zerop_t, zero_t, axes_t = forward(t, q0)
    compare = []
    for row in standing(root):
        spec = row["spec"]
        require(
            any(
                p.endswith("/go2_x5.xml") and sha == model_sha
                for p, sha in spec["frozen_files"].items()
            ),
            "target model registration",
        )
        at = row["at"]
        q = np.asarray(at["qpos"])
        marker = 19
        p, R, _ = forward(t, q[None, marker : marker + 6])
        src, S, _ = forward(s, q[None, marker : marker + 6])
        mount = np.asarray(at["actual_mount"])
        tcp = np.asarray(at["actual_TCP"])
        alleged = mount[:3, :3] @ (p[0] + R.apply(tool)[0]) + mount[:3, 3]
        fk_mm = float(1000 * np.linalg.norm(alleged - tcp[:3, 3]))
        fk_deg = float(
            np.degrees(
                (
                    Rotation.from_matrix(mount[:3, :3] @ R.as_matrix()[0]).inv()
                    * Rotation.from_matrix(tcp[:3, :3])
                ).magnitude()
            )
        )
        require(fk_mm < 0.005 and fk_deg < 0.005, "target FK versus measurement")
        compare.append(
            dict(
                id=row["id"],
                target_tcp_mm=fk_mm,
                target_tcp_deg=fk_deg,
                same_joint_source_vs_target_link6_mm=float(
                    1000 * np.linalg.norm(src - p)
                ),
                same_joint_source_vs_target_link6_deg=float(
                    np.degrees((S.inv() * R).magnitude()[0])
                ),
                consumed_H40_slide_command_mm=float(
                    1000 * h40_fraction_to_width(at["target_endpoint_EEF10"][9])
                ),
                actual_mean_slide_mm=float(1000 * q[25:27].mean()),
                actual_slide_state=float(sensed_width_to_state(q[25:27].mean())),
            )
        )
    axisdiff = np.degrees(
        np.arccos(np.clip(np.sum(axes_s[0] * axes_t[0], axis=1), -1, 1))
    )
    return dict(
        source_zero_link6_mm=(1000 * zerop_s[0]).tolist(),
        target_zero_link6_mm=(1000 * zerop_t[0]).tolist(),
        zero_link6_rotation_difference_deg=float(
            np.degrees((zero_s.inv() * zero_t).magnitude()[0])
        ),
        corresponding_positive_axis_angles_deg=axisdiff.tolist(),
        target_link6_tool_mm=(1000 * tool).tolist(),
        fitted_parameters=0,
        rows=compare,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run(*(Path(s).resolve() for s in sys.argv[1:])), indent=2, allow_nan=False
        )
    )
