"""Contact-free kinematic witnesses on independent MjData; never advance dynamics."""

import mujoco
import numpy as np
from d1_factorial_reference_matrix import check, mpose

from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
)
from scripts.pi0294.mp_client import contact_permitted


def transform(x, r):
    v = np.eye(4)
    v[:3, 3] = x
    v[:3, :3] = r.reshape(3, 3)
    return v


def interpolate(a, b, t):
    return mpose(
        a[:3, 3] * (1 - t) + b[:3, 3] * t,
        quat_slerp(matrix_to_quat(a[:3, :3]), matrix_to_quat(b[:3, :3]), t),
    )


def sweep(model, sample, parts, ee, base, guard):
    return sweep_profile(model, sample, parts, ee, base, guard, (34, 32, 22), _single_object_approach)


def _single_object_approach(names, distance):
    return contact_permitted(names, distance, "approach") and not (
        "task_cube_geom" in names and distance < -0.004
    )


def sweep_profile(model, sample, parts, ee, base, guard, scene_dimensions, permitted, witness=None):
    data = mujoco.MjData(model)
    data.qpos[:] = sample["qpos"]
    mujoco.mj_forward(model, data)
    root = model.body("base_link").id
    mount = model.body("x5_base_link").id
    tool = model.site("x5_ee").id
    feet = [model.geom(n).id for n in ("FL", "FR", "RL", "RR")]
    floor = model.geom("floor").id
    root0 = transform(data.xpos[root], data.xmat[root])
    mount0 = transform(data.xpos[mount], data.xmat[mount])
    mount_to_root = np.linalg.solve(mount0, root0)
    foot0 = data.geom_xpos[feet].copy()
    joints = np.arange(1, 19)
    check(
        np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE)
        and np.all(model.jnt_limited[joints]),
        "native 18-hinge mapping",
    )
    qa, vd = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    check((model.nq, model.nv, model.njnt) == scene_dimensions, "native scene ABI")
    J = np.zeros((18, model.nv))
    jr = np.zeros((3, model.nv))
    steps = 0
    margin = float("inf")
    maxerr = 0.0

    def solve(tcp, desired_mount):
        nonlocal margin, maxerr, steps
        root_pose = desired_mount @ mount_to_root
        data.qpos[:3] = root_pose[:3, 3]
        data.qpos[3:7] = np.roll(matrix_to_quat(root_pose[:3, :3]), 1)
        for _ in range(12):
            mujoco.mj_fwdPosition(model, data)
            diff = np.zeros(18)
            diff[:3] = tcp[:3, 3] - data.site_xpos[tool]
            mujoco.mj_jacSite(model, data, J[:3], J[3:6], tool)
            targetq = np.roll(matrix_to_quat(tcp[:3, :3]), 1)
            currentq = np.empty(4)
            mujoco.mju_mat2Quat(currentq, data.site_xmat[tool])
            inv = np.empty(4)
            dq = np.empty(4)
            mujoco.mju_negQuat(inv, currentq)
            mujoco.mju_mulQuat(dq, targetq, inv)
            mujoco.mju_quat2Vel(diff[3:6], dq, 1.0)
            for i, g in enumerate(feet):
                off = 6 + 3 * i
                diff[off : off + 3] = foot0[i] - data.geom_xpos[g]
                mujoco.mj_jacGeom(model, data, J[off : off + 3], jr, g)
            err = float(np.max(abs(diff)))
            if err < 2e-6:
                break
            jac = J[:, vd]
            data.qpos[qa] += np.linalg.solve(
                jac.T @ jac + 1e-8 * np.eye(18), jac.T @ diff
            )
        else:
            raise ValueError("whole-body swept IK did not converge")
        maxerr = max(maxerr, err)
        bound = np.minimum(
            data.qpos[qa] - model.jnt_range[joints, 0],
            model.jnt_range[joints, 1] - data.qpos[qa],
        )
        margin = min(margin, float(bound.min()))
        check(margin > 0, "swept native joint limit")
        contacts = set()
        for c in data.contact:
            names = [model.geom(int(g)).name for g in c.geom]
            check(
                permitted(names, float(c.dist)),
                "forbidden swept contact " + str(names),
            )
            if floor in c.geom:
                contacts.update(int(g) for g in c.geom if g != floor)
        if witness is not None:
            witness(data)
        check(
            set(feet) <= contacts, "missing standing foot support in kinematic witness"
        )
        z = data.xpos[mount, 2]
        R = data.xmat[mount].reshape(3, 3)
        for value, key in [
            (z, "mount_position_z"),
            (np.arcsin(-R[2, 0]), "pitch"),
            (np.arctan2(R[2, 1], R[2, 2]), "roll"),
        ]:
            low, high = guard[key]
            check(low <= value <= high, "mobile operating bound " + key)
        check(
            np.linalg.norm(data.xpos[mount, :2] - mount0[:2, 3])
            < guard["workspace_radius_m"],
            "mobile workspace",
        )
        steps += 1

    start_tcp = transform(data.site_xpos[tool], data.site_xmat[tool])
    # Cover both the displacement from actual state and the old-to-new reference seam.
    pairs = [
        (start_tcp, mount0, parts["old_ee"], parts["old_base"]),
        (parts["old_ee"], parts["old_base"], ee, base),
        (ee, base, start_tcp, mount0),
        (start_tcp, mount0, ee, base),
    ]
    for a, b, x, y in pairs:
        lengths = [
            np.linalg.norm(a[:3, 3] - x[:3, 3]),
            np.linalg.norm(b[:3, 3] - y[:3, 3]),
        ]
        angles = [
            float(geodesic_angle(a[:3, :3], x[:3, :3])),
            float(geodesic_angle(b[:3, :3], y[:3, :3])),
        ]
        count = max(
            1, int(np.ceil(max(max(lengths) / 0.0005, max(angles) / np.deg2rad(0.1))))
        )
        for i in range(count + 1):
            solve(interpolate(a, x, i / count), interpolate(b, y, i / count))
    return {
        "poses_checked": steps,
        "minimum_joint_margin_rad": margin,
        "maximum_IK_component_residual": maxerr,
        "all_four_feet": True,
        "forbidden_contacts": 0,
        "scope": "discrete kinematic witness, 0.5 mm / 0.1 degree subdivisions; native dynamic guard remains necessary",
    }
