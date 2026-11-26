"""Screen carried-object corridors using terminal poses from two successful mobile HOLDs."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, require, sha


COLD = "3d9ad98d7b92e36b4ca10eaf7232370af9f9ff2d2908b169ac6b1b465aa87cd3"
SCORES = ("03f81c894a2a54595dbebf3ebe605328f55a536620a9df4fd5a5521e227dbae4",
          "79491ddc5937da265852c3807b242f3b4850742e4874328e10ae1c7630db4f4b")
OBJECTS = {"red": "task_cube", "blue": "task_blue_cube"}
GRASP_CLEARANCE = .003
SPACING = .002


def terminal_pose(folder, expected):
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    score = read(pin(folder / "RESULT.private.json", expected))
    require(read(folder / "control/LEASE.json")["status"] == "CLOSED"
            and not score["unsafe"] and not score["technical_failure"] and score["shutdown_clean"],
            "Use a closed successful original M_FREE run")
    actual = score["detail"]["score"]
    require(actual["task_success"] and actual["target_grasp"] and actual["maximum_lift_m"] >= .02,
            "Expected a real supported grasp")
    state = folder / "native-01/state.bin.journal"
    last = None
    for last in packets(state):
        pass
    require(last is not None and last["step"] == actual["native_step"], "Exact final physical snapshot required")
    return last["qpos"], dict(tick=actual["native_step"], state_sha256=sha(state), result_sha256=expected)


def wall_top(mj, model, data, geom):
    vertical = data.geom_xmat[geom].reshape(3, 3)[2]
    if model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
        asset = model.geom_dataid[geom]
        start = model.mesh_vertadr[asset]
        points = model.mesh_vert[start:start + model.mesh_vertnum[asset]]
        return float(data.geom_xpos[geom][2] + (points @ vertical).max())
    require(model.geom_type[geom] == mj.mjtGeom.mjGEOM_CYLINDER, "Unexpected analytic bowl part")
    radius, half_height = model.geom_size[geom][:2]
    return float(data.geom_xpos[geom][2] + radius*np.linalg.norm(vertical[:2]) + half_height*abs(vertical[2]))


def solve(mj, model, data, position, orientation, site, arm):
    q, v = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(6):
        mj.mj_fwdPosition(model, data)
        translation = position - data.site_xpos[site]
        angles = Rotation.from_matrix(orientation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(translation) <= 1e-5 and np.linalg.norm(angles) <= 1e-4:
            return True
        mj.mj_jacSite(model, data, jp, jr, site)
        step = np.linalg.lstsq(np.vstack((jp[:, v], jr[:, v])), np.r_[translation, angles], rcond=1e-10)[0]
        if not np.isfinite(step).all():
            return False
        data.qpos[q] += step
    return False


def clearance(mj, model, data, moving, fixture):
    return min((mj.mj_geomDistance(model, data, g, wall, .3, None), g, wall)
               for g in moving for wall in fixture)


def above_rim_release(mj, model, qpos, scene, joint, geom, site, arm, links, bowl, insert_z):
    data = mj.MjData(model)
    data.qpos[:] = qpos
    finger_joint = [model.joint(name).id for name in ("task_left_slide", "task_right_slide")]
    fingers = model.jnt_qposadr[finger_joint]
    opened = model.jnt_range[finger_joint, 1]
    arm_q, limits = model.jnt_qposadr[arm], model.jnt_range[arm]
    minima = {}
    for phase in ("OPEN", "CENTRED_FALL_HYPOTHESIS", "WITHDRAW"):
        stage_gaps = []
        if phase == "OPEN":
            start = data.qpos[fingers].copy()
            path = np.linspace(start, opened, 24)
        elif phase == "CENTRED_FALL_HYPOTHESIS":
            midpoint = data.qpos[joint:joint+3].copy()
            path = np.linspace(midpoint[2], insert_z, 28)
        else:
            position = data.site_xpos[site].copy()
            orientation = data.site_xmat[site].reshape(3, 3).copy()
            path = np.linspace(0, .02, 11)
        for node in path:
            if phase == "OPEN":
                data.qpos[fingers] = node
            elif phase == "CENTRED_FALL_HYPOTHESIS":
                data.qpos[joint+2] = node
            elif not solve(mj, model, data, position + [0, 0, node], orientation, site, arm):
                return dict(passed=False, phase=phase, kind="LOCAL_IK", minimum_clearance_mm=minima)
            if np.any(data.qpos[arm_q] < limits[:, 0]) or np.any(data.qpos[arm_q] > limits[:, 1]):
                return dict(passed=False, phase=phase, kind="ARM_JOINT_LIMIT", minimum_clearance_mm=minima)
            mj.mj_fwdPosition(model, data)
            gap, link, part = clearance(mj, model, data, links, bowl)
            if phase == "CENTRED_FALL_HYPOTHESIS":
                walls = [g for g in bowl if model.geom(g).name != "ef5_static_bowl_base"]
                object_gap = clearance(mj, model, data, [geom], (*walls, *links))
                gap, link, part = min((gap, link, part), object_gap)
            stage_gaps.append(gap)
            if gap <= 0:
                return dict(passed=False, phase=phase, kind="BOWL_CLEARANCE", observed_mm=gap*1000,
                            geom=[model.geom(g).name for g in (link, part)], minimum_clearance_mm=minima)
        minima[phase] = min(stage_gaps)*1000
    return dict(passed=True, minimum_clearance_mm=minima, stock_open_joint_mm=(opened*1000).tolist(),
                assumed_object_drop_mm=(qpos[joint+2]-insert_z)*1000, source_scene=scene)


def corridor(mj, model, actual, row, carry_clearance):
    from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted

    data = mj.MjData(model)
    site = model.site("x5_ee").id
    obj = OBJECTS[row["target"]]
    object_joint = model.joint(obj + "_free").id
    object_geom = model.geom(obj + "_geom").id
    obj_q = model.jnt_qposadr[object_joint]
    root, = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
             and model.jnt_bodyid[j] == model.body("base_link").id]
    root_q = model.jnt_qposadr[root]
    shift = data.qpos[obj_q:obj_q+2] - actual[obj_q:obj_q+2]
    # Use a real held body posture, placed relative to the candidate pickup
    # object. The held object pose relative to the wrist stays exactly fixed.
    robot_end = min(model.jnt_qposadr[model.joint(name + "_free").id] for name in OBJECTS.values())
    require(len(actual) == model.nq and root_q + 7 <= robot_end, "Original mobile state ABI changed")
    data.qpos[:robot_end] = actual[:robot_end]
    data.qpos[root_q:root_q+2] += shift
    data.qpos[obj_q:obj_q+7] = actual[obj_q:obj_q+7]
    data.qpos[obj_q:obj_q+2] += shift
    arm = np.asarray([model.joint("x5_joint" + str(i)).id for i in range(1, 7)])
    stock = model.jnt_range[arm]
    bowl = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("ef5_static_bowl").id
            and model.geom_contype[g]]
    links = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith("x5_")
             and model.geom_contype[g]]
    require(len(bowl) == 25 and links and model.opt.timestep == .002, "Changed compiled bowl or robot")
    mj.mj_fwdPosition(model, data)
    tcp = data.site_xpos[site].copy()
    orientation = data.site_xmat[site].reshape(3, 3).copy()
    initial = data.qpos[obj_q:obj_q+3].copy()
    require(model.geom_type[object_geom] == mj.mjtGeom.mjGEOM_BOX, "Expected the original task object box")
    half_height = abs(data.geom_xmat[object_geom].reshape(3, 3)[2]) @ model.geom_size[object_geom][:3]
    rim = max(wall_top(mj, model, data, g) for g in bowl)
    high = max(initial[2], rim + half_height + carry_clearance)
    overhead = np.r_[row["fixture_world_xyz"][:2], high]
    insert = overhead.copy()
    insert[2] = row["fixture_floor_m"] + half_height + GRASP_CLEARANCE
    points = (("RAISE", np.r_[initial[:2], high]), ("CARRY", overhead), ("LOWER_INTO_BOWL", insert))
    receipts, higher_release = [], None
    for phase, goal in points:
        start = data.qpos[obj_q:obj_q+3].copy()
        n = max(1, int(np.ceil(np.linalg.norm(goal - start) / SPACING)))
        gap = np.inf
        failed = None
        for node in np.linspace(start, goal, n+1):
            if not solve(mj, model, data, tcp + node - initial, orientation, site, arm):
                failed = dict(kind="LOCAL_IK", object_xyz_m=node.tolist())
                break
            if np.any(data.qpos[model.jnt_qposadr[arm]] < stock[:, 0]) or np.any(data.qpos[model.jnt_qposadr[arm]] > stock[:, 1]):
                failed = dict(kind="ARM_JOINT_LIMIT", object_xyz_m=node.tolist())
                break
            data.qpos[obj_q:obj_q+3] = initial + data.site_xpos[site] - tcp
            mj.mj_fwdPosition(model, data)
            pair_gap, prox, wall = clearance(mj, model, data, (object_geom, *links), bowl)
            gap = min(gap, pair_gap)
            if pair_gap <= 0:
                failed = dict(kind="BOWL_CLEARANCE", geometry=[model.geom(g).name for g in (prox, wall)],
                              bodies=[model.body(model.geom_bodyid[g]).name for g in (prox, wall)],
                              observed_mm=pair_gap * 1000, object_xyz_m=node.tolist(),
                              geometric_object_bottom_m=float(data.geom_xpos[object_geom][2] - half_height),
                              intersected_piece_top_m=wall_top(mj, model, data, wall))
                break
            for contact in data.contact:
                names = [model.geom(g).name for g in (contact.geom1, contact.geom2)]
                if (any(g in links or g == object_geom for g in (contact.geom1, contact.geom2))
                        and not observer_permitted(*names, contact.dist, row["target"])):
                    failed = dict(kind="ORIGINAL_CONTACT_PREDICATE", geometry=names, observed_mm=contact.dist * 1000,
                                  object_xyz_m=node.tolist())
                    break
            if failed is not None:
                break
        receipts.append(dict(phase=phase, end_object_xyz_m=goal.tolist(), planned_nodes=n+1, failure=failed,
                             minimum_bowl_clearance_mm=None if np.isinf(gap) else gap * 1000))
        if failed is not None:
            break
        if phase == "CARRY":
            higher_release = above_rim_release(mj, model, data.qpos, row["id"], obj_q, object_geom,
                                               site, arm, links, bowl, insert[2])
    return dict(stage=receipts, nominal_held_height_m=initial[2], base_translation_xy_m=shift.tolist(),
                physical_bowl_collision_top_m=rim, carried_object_half_height_m=half_height,
                above_rim_release=higher_release,
                full_kinematic_carry_and_insert=len(receipts)==3 and receipts[-1]["failure"] is None)


def run(source, first, second, output, carry_clearance_mm):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    rows = read(pin(source / "ACTUAL.json", COLD))["rows"]
    require(len(rows) == 2 and all(row["status"]=="COLD_PASS" for row in rows), "Expected both project bowl scenes")
    results = []
    with forbid_integration(mj) as effects:
        for row, folder, signature in zip(rows, (first, second), SCORES, strict=True):
            scene = pin(source / (row["id"] + ".mjb"), row["compiled_sha256"])
            model = mj.MjModel.from_binary_path(str(scene))
            pose, evidence = terminal_pose(folder, signature)
            results.append(dict(id=row["id"], scene_sha256=row["compiled_sha256"], source_evidence=evidence,
                                **corridor(mj, model, pose, row, .001 * carry_clearance_mm)))
    require(not effects, "Cold geometry must never integrate or change native control")
    write(output, dict(rows=results, carry_clearance_mm=carry_clearance_mm, native_integrations=0,
                       model_forwards=0, native_dispatch_qualified=False))
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "first", "second", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--carry-clearance-mm", type=float, choices=(3., 10.), default=3.)
    run(**vars(parser.parse_args()))
