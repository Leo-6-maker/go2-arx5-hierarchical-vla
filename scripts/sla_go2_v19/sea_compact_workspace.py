"""Cold, compact fruit workspaces around the measured standing TCP."""

import json
import os
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_array_equal

from sea_fruit_h0_cold import (
    BASE, BASELINE, BASELINE_SHA, DEV_LAYOUTS, HALF_SIZE_M, LAYOUT_ARCHIVE,
    LAYOUT_SHA, ROBOT, fruit_xml, load_ref, pin, read, require, sha, verify_geometry,
)

TABLE_TOP_M = .652
FRUIT_XY = {"G00": (.282, -.024), "G01": (.292, .001)}
DISTRACTOR_Y_OFFSET_M = .090
HOVER_ABOVE_GRASP_M = .022
PROFILE = "FRUIT_COMPACT_STANCE_01"
HOLD_SECONDS = 1.
LIFT_TRAVEL_M = .026


def positions(role):
    x, y = FRUIT_XY[role]
    center = TABLE_TOP_M + HALF_SIZE_M[2]
    return dict(red=[x, y, center], blue=[x, y+DISTRACTOR_Y_OFFSET_M, center])


def compact_xml(old_xml, old_top, xyz, *, top=TABLE_TOP_M):
    root = ET.fromstring(old_xml)
    edits = []
    shift = top - old_top

    def change(elem, key, value):
        require(elem is not None and key in elem.attrib, "frozen XML scene landmark missing")
        edits.append((elem, key, elem.attrib[key]))
        elem.set(key, " ".join(format(float(n), ".17g") for n in value))

    for name in ("task_table", *(f"task_goal_{i}" for i in range(4))):
        elem = root.find(f"./worldbody/geom[@name='{name}']")
        xyz0 = np.fromstring(elem.get("pos"), sep=" ")
        change(elem, "pos", xyz0+np.array([0., 0., shift]))
    for role, name in (("red", "task_cube"), ("blue", "task_blue_cube")):
        change(root.find(f"./worldbody/body[@name='{name}']"), "pos", xyz[role])
    for elem in root.findall("./keyframe/key"):
        q = np.fromstring(elem.get("qpos"), sep=" ")
        require(q.shape == (41,), "frozen robot+two-object keyframe ABI")
        q[27:30], q[34:37] = xyz["red"], xyz["blue"]
        change(elem, "qpos", q)
    xml = ET.tostring(root, encoding="unicode")
    for elem, key, value in edits:
        elem.set(key, value)
    require(ET.tostring(root, encoding="unicode") == old_xml, "non-scene XML changed")
    return xml


def verify_frozen(old, new):
    """Only table/marker XYZ and passive-object reset positions may differ."""
    for k in ("nq", "nv", "nu", "nbody", "ngeom", "njnt", "neq", "nmocap", "nsensor"):
        require(getattr(old, k) == getattr(new, k), f"frozen model field {k}")
    changed_geoms = [old.geom(name).id for name in ("task_table", *(f"task_goal_{i}" for i in range(4)))]
    rest = np.ones(old.ngeom, dtype=bool)
    rest[changed_geoms] = False
    assert_array_equal(old.geom_pos[rest], new.geom_pos[rest])
    assert_array_equal(old.qpos0[:27], new.qpos0[:27])
    assert_array_equal(old.key_qpos[:, :27], new.key_qpos[:, :27])
    for k in ("key_ctrl", "key_qvel", "geom_size", "geom_type", "geom_friction",
              "geom_contype", "geom_conaffinity", "body_mass", "actuator_gear", "actuator_trnid",
              "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "sensor_objid",
              "jnt_qposadr", "dof_damping", "dof_frictionloss", "cam_pos", "cam_quat"):
        assert_array_equal(getattr(old, k), getattr(new, k), err_msg=k)
    require(old.opt.timestep == new.opt.timestep == .002, "original native tick")


def source(scene_archive, fruit_folder):
    import mujoco
    from go2arx5_vla.world_model import multitask_scene_v6 as scene
    from scripts.wm02 import multitask_scene_audit_v6 as assets

    baseline = read(pin(scene_archive / BASELINE / "COLD_INPUT.private.json", BASELINE_SHA))
    previous = read(pin(scene_archive / LAYOUT_ARCHIVE / "population/COMPILED_POPULATION.private.json",
                        LAYOUT_SHA))
    template = load_ref(baseline["templates"]["F1"])
    template["asset_root"], _ = assets.asset_binding(template)
    path = template["model_path"]
    original = mujoco.MjModel.from_binary_path(str(pin(Path(path), template["frozen_files"][path])))
    report = BASE / "reports/SLA_GO2_V193/GRIP_H0/FRUIT_SCENES_CPU.json"
    layouts = {d["layout"]: d for d in read(pin(fruit_folder / "RESULT.json", sha(report)))["layouts"]}
    rows = {c["id"]: c["card"] for c in previous["clusters"] if c["id"] in DEV_LAYOUTS}
    return original, template, rows, layouts


def build(scene_archive, fruit_folder, out):
    import mujoco
    from go2arx5_vla.world_model import multitask_scene_v6 as scene
    from go2arx5_vla.world_model.native_replay import forbid_integration

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and mujoco.__version__ == "3.2.7",
            "original CPU MuJoCo version required")
    original, template, rows, layouts = source(scene_archive, fruit_folder)
    out.mkdir(mode=0o700)
    records = {}
    for role, name in zip(FRUIT_XY, DEV_LAYOUTS, strict=True):
        old_scene = pin(fruit_folder / name / "SCENE.private.mjb", layouts[name]["scene_sha256"])
        xyz = positions(role)
        box_xml, geometry = scene.overlay_xml(template, original, rows[name])
        with forbid_integration(mujoco) as calls:
            box = mujoco.MjModel.from_xml_string(box_xml)
            require(scene.robot_contract(original, box)["fingerprint"] == ROBOT, "original robot binding")
            xml = compact_xml(fruit_xml(box_xml), geometry["table_top_world_z"], xyz)
            m = mujoco.MjModel.from_xml_string(xml)
            old = mujoco.MjModel.from_binary_path(str(old_scene))
            verify_frozen(old, m)
            static = verify_geometry(mujoco, m, {"positions_world_m": xyz})
            file = out / f"{role}.mjb"
            mujoco.mj_saveModel(m, str(file), None)
        require(not calls and sha(file) != sha(old_scene), "distinct geometry without integration")
        new_geometry = deepcopy(layouts[name]["scoring_geometry"])
        new_geometry["table_top_world_z"] = TABLE_TOP_M
        records[role] = dict(scene_sha256=sha(file), parent_scene_sha256=sha(old_scene),
                             positions_world_m=xyz, scoring_geometry=new_geometry, static=static)
    result = dict(profile=PROFILE, table_top_m=TABLE_TOP_M, geometry=records, integrations=0,
                  native_dispatch_qualified=False)
    (out / "COMPILED.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
