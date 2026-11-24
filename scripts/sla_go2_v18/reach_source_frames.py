"""Independently check official real-arm joint/EEF semantics and state normalization on CPU."""

import json
import sys
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from reach_crossed_seeds import case_inputs
from demonstrated_grip import pin, read, require


def run(root, urdf):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms
    from openpi.policies.robodojo_policy import (
        local_arm_to_target_pose,
        target_to_local_arm_pose,
    )

    _, data, stats = configured_transforms(root)
    rules = data.data_transforms.inputs
    dist = next(t.arm_base_distance for t in rules if hasattr(t, "arm_base_distance"))
    model = ET.parse(
        pin(urdf, "13b4a47a689afb03afd3a5f08577f0afd79559bbf4193b25072b053d0c20737b")
    )

    def fk(q):
        xyz = np.zeros((len(q), 3))
        rot = R.identity(len(q))
        for j in range(6):
            joint = model.find(f".//joint[@name='joint{j + 1}']")
            org = joint.find("origin")
            xyz += rot.apply(np.fromstring(org.get("xyz"), sep=" "))
            rot = (
                rot
                * R.from_euler("xyz", np.fromstring(org.get("rpy"), sep=" "))
                * R.from_rotvec(
                    q[:, j, None]
                    * np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                )
            )
        return xyz, rot

    home = fk(np.zeros((1, 6)))[0][0]
    rows = []
    for item in read(root / "evidence/P0_SOURCE_FILES.actual.json")["files"]:
        p = pin(root / "data" / Path(item["path"]).name, item["published_lfs_sha256"])
        with h5py.File(p) as f:
            for kind in ("state", "action"):
                for side in ("left", "right"):
                    q = f[f"{kind}/{side}_arm_joint_states"][:]
                    ee = f[f"{kind}/{side}_ee_poses"][:]
                    xyz, rot = fk(q)
                    dr = (
                        (rot.inv() * R.from_quat(ee[:, [4, 5, 6, 3]])).magnitude().max()
                    )
                    dp = np.linalg.norm(xyz - home - ee[:, :3], axis=1).max()
                    # Verify the official shared-frame translation and its inverse on the whole record.
                    local = np.column_stack([ee, np.ones(len(q))])
                    mid = local_arm_to_target_pose(
                        local, is_left=side == "left", arm_base_distance=dist
                    )
                    want = local[:, :3] + np.array(
                        [0, (1 if side == "left" else -1) * dist / 2, 0]
                    )
                    back = target_to_local_arm_pose(
                        mid, is_left=side == "left", arm_base_distance=dist
                    )
                    dm = np.max(np.abs(mid[:, :3] - want))
                    require(
                        dp < 1e-8
                        and dr < 1e-8
                        and dm < 1e-8
                        and np.allclose(back, local, atol=1e-9, rtol=0),
                        "frame semantics",
                    )
                    rows.append(
                        dict(
                            episode=p.name,
                            sha256=item["published_lfs_sha256"],
                            side=side,
                            kind=kind,
                            count=len(q),
                            translation_max_m=dp,
                            angle_max_rad=dr,
                            shared_frame_translation_max_m=dm,
                        )
                    )
    physical = transforms.compose(rules)
    lo, hi = (np.asarray(getattr(stats["state"], s)) for s in ("q01", "q99"))
    target = []
    # Include G00 even though its second recorded input never entered inference.
    for name, panel in (("G00-A", "Z1_REACH_483"), ("G01-A", "Z1_REACH_483_G01")):
        for q in sorted((root / "v18" / panel / name).glob("query-*")):
            with np.load(q / "INPUT.npz", allow_pickle=False) as z:
                obs = dict(
                    state=z["state"],
                    prompt=str(z["prompt"]),
                    images=dict(
                        cam_high=z["head"],
                        cam_right_wrist=z["wrist"],
                        cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                    ),
                )
            from policy_query import input_identity

            request = read(q / "REQUEST.json")
            require(input_identity(obs)[0] == request["input_sha256"], "input")
            st = physical(obs)["state"]
            require(
                len(st) == len(lo) == len(hi) == 14, "official physical state channels"
            )
            target.append(
                dict(
                    layout=name,
                    tick=request["tick"],
                    state=st.tolist(),
                    outside_q01_q99=np.flatnonzero((st < lo) | (st > hi)).tolist(),
                )
            )
    case_inputs(root)
    return dict(
        source_rows=rows,
        source_home_m=home.tolist(),
        actual_base_separation_m=dist,
        state_q01=lo.tolist(),
        state_q99=hi.tolist(),
        target=target,
        new_forwards=0,
        updates=0,
        integrations=0,
    )


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
    print(json.dumps(result, indent=2, allow_nan=False))
