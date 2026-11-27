"""Compile project plate-push qualification geometry from the existing Go2 tabletop."""

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v5_scene_factory import QUAL


LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_TASK_SHA256 = "26f22f8ad39a4b2b344b049115c45661a877fa0b4821fb78246725e7e87d8c0b"
LIBERO_TASK = "libero/libero/bddl_files/libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
N1_SPEC_SHA256 = "50f16beead8eb039a57f64558eb2fd08d8f9a438eeea70f7efa6105fea40298d"
RADIUS, HALF_HEIGHT, MASS = .046, .009, .035
SUPPORT_OVERLAP_M = .0003
ROSTER = (
    ("EF5_Q_PUSH_01", "EF5_Q_MOBILE_01", 702801, (.062, 0.)),
    ("EF5_Q_PUSH_02", "EF5_Q_MOBILE_02", 702802, (0., .053)),
)


def plate_xml(xml, object_names, z, goal, table_z):
    root = ElementTree.fromstring(xml)
    body = root.find(f".//body[@name='{object_names['body']}']")
    plate = body.find(f"./geom[@name='{object_names['geom']}']")
    position = body.attrib["pos"].split()
    position[2] = str(z)
    body.set("pos", " ".join(position))
    plate.set("type", "cylinder")
    plate.set("size", f"{RADIUS} {HALF_HEIGHT}")
    plate.set("mass", str(MASS))
    for key in root.findall("./keyframe/key"):
        state = key.attrib["qpos"].split()
        require(len(state) == 41, "Expected the frozen robot plus two passive objects")
        state[29] = str(z)
        key.set("qpos", " ".join(state))
    for number in range(4):
        marker = root.find(f"./worldbody/geom[@name='task_goal_{number}']")
        require(marker.get("contype") == marker.get("conaffinity") == "0", "Visual target cannot make contact")
        if number in (0, 2):
            marker.set("rgba", "0 0 0 0")
        else:
            marker.set("pos", f"{goal[0]} {goal[1]} {table_z + (number*.00015)}")
            if number == 1:
                marker.set("size", f"{RADIUS + .010} {RADIUS + .010} .0001")
    return ElementTree.tostring(root, encoding="unicode")


def screen(mj, model, geometry, target_xy):
    plate, distractor = (model.geom(geometry["objects"][color]["geom"]).id for color in ("red", "blue"))
    table = model.geom("task_table").id
    target_body = model.body(geometry["objects"]["red"]["body"]).id
    free = model.joint(geometry["objects"]["red"]["joint"])
    require(free.type[0] == mj.mjtJoint.mjJNT_FREE and model.geom_type[plate] == mj.mjtGeom.mjGEOM_CYLINDER
            and np.isclose(model.body_mass[target_body], MASS), "Compiled plate dynamics differ")
    address = int(free.qposadr[0])
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    start = data.geom_xpos[plate].copy()
    support = []
    for index, contact in enumerate(data.contact):
        if {int(contact.geom1), int(contact.geom2)} == {plate, table}:
            wrench = np.zeros(6)
            mj.mj_contactForce(model, data, index, wrench)
            support.append(float(wrench[0]))
    require(sum(support) > 0 and np.isclose(start[2] - HALF_HEIGHT,
            geometry["table_top_world_z"] - SUPPORT_OVERLAP_M), "The reset plate must have solved tabletop support")
    initial_robot_clearance = min(mj.mj_geomDistance(model, data, plate, i, .5, None)
        for i in range(model.ngeom) if model.geom_contype[i] and model.geom_bodyid[i] not in (
            0, target_body, model.geom_bodyid[distractor]))
    bounds = model.geom_size[table][:2]
    center = data.geom_xpos[table, :2].copy()
    clearances, margins = [], []
    for xy in np.linspace(start[:2], target_xy, 27):
        data.qpos[address:address+2] = xy
        mj.mj_fwdPosition(model, data)
        clearances.append(1000 * mj.mj_geomDistance(model, data, plate, distractor, .5, None))
        margins.append(1000 * np.min(bounds - abs(xy-center) - RADIUS))
    require(min(clearances) > 0 and min(margins) > 0 and initial_robot_clearance > 0,
            "Plate slide corridor intersects the fixed distractor, table edge, or reset robot")
    require(np.all(abs(target_xy-center) + RADIUS + .010 < bounds), "Full target zone must fit on the tabletop")
    return dict(reset_center_m=start.tolist(), goal_center_XY_m=target_xy.tolist(), goal_half_size_XY_m=[RADIUS + .010]*2,
                support_force_n=sum(support), robot_reset_clearance_mm=initial_robot_clearance * 1000,
                distractor_slide_clearance_min_mm=min(clearances), tabletop_slide_margin_min_mm=min(margins),
                translation_samples=len(clearances))


def build(source, qualification, task, output):
    import mujoco as mj
    from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml, robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write
    from scripts.wm02.multitask_scene_audit_v6 import asset_binding

    spec = read(pin(source, N1_SPEC_SHA256))
    s1 = read(pin(qualification, QUAL))
    language = pin(task, LIBERO_TASK_SHA256).read_bytes()
    require(b"(:language Push the plate to the front of the stove)" in language
            and b"(On plate_1 main_table_stove_front_region)" in language, "Source task changed")
    lookup = {row["id"]: row for row in s1["clusters"]}
    spec["asset_root"], _ = asset_binding(spec)
    prior = mj.MjModel.from_binary_path(str(pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])))
    records = []
    output.mkdir(mode=0o700)
    with forbid_integration(mj) as integrations:
        for identity, original_id, reset, displacement in ROSTER:
            cluster = lookup[original_id]
            family = cluster["family"]
            xml, geometry = overlay_xml(spec, prior, cluster["card"])
            table_z = geometry["table_top_world_z"]
            search_z = table_z + HALF_HEIGHT - SUPPORT_OVERLAP_M
            goal = np.asarray(cluster["card"]["object_positions_world"]["red"][:2]) + displacement
            old = mj.MjModel.from_xml_string(xml)
            witness = robot_contract(prior, old)
            require(witness["fingerprint"] == cluster["robot_fingerprint"], "Unchanged S1 robot controls required")
            model = mj.MjModel.from_xml_string(plate_xml(xml, geometry["objects"]["red"], search_z, goal, table_z))
            for field in ("actuator_gear", "actuator_gainprm", "actuator_biasprm", "jnt_type", "jnt_range",
                          "sensor_adr", "sensor_dim", "cam_pos", "cam_quat"):
                np.testing.assert_array_equal(getattr(old, field), getattr(model, field))
            np.testing.assert_array_equal(old.qpos0[:27], model.qpos0[:27])
            np.testing.assert_array_equal(old.key_qpos[:, :27], model.key_qpos[:, :27])
            require((model.nq, model.nv, model.nu, model.opt.timestep) == (41, 38, 20, .002), "Native ABI changed")
            result = screen(mj, model, geometry, goal)
            path = output / (identity + ".mjb")
            mj.mj_saveModel(model, str(path), None)
            records.append(dict(result, id=identity, family=family, split="QUAL", reset=reset,
                                source_layout=original_id, scene_sha256=sha(path), robot_fingerprint=witness["fingerprint"]))
    require(not integrations and len({row["scene_sha256"] for row in records}) == len(ROSTER),
            "Qualification must end with separate compiled scenes and zero integration")
    write(output / "ACTUAL.json", dict(rows=records, source_revision=LIBERO_REVISION, source_task=LIBERO_TASK,
         source_task_sha256=LIBERO_TASK_SHA256, project_primitive=dict(radius_m=RADIUS, half_height_m=HALF_HEIGHT, mass_kg=MASS),
         source_mesh_imported=False, new_integrations=0, native_dispatch_qualified=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "qualification", "task", "output"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
