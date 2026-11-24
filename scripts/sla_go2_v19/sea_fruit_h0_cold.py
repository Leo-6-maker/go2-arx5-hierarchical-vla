"""Build two development fruit scenes and replay the frozen contact roles on CPU."""

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_array_equal

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_go2_v18"))
from compile_teacher_holdouts import (
    BASELINE, BASELINE_SHA, ROBOT, load_ref, pin, read,
    registered_historical_scenes, require, sha, verify_geometry,
)
from holdout_effort_preflight import check_contact_roles

DEV_LAYOUTS = ("CHV1-TRAIN-F1-G00", "CHV1-TRAIN-F1-G01")
LAYOUT_ARCHIVE = "sla-core-hopper-v1-geometries-20261121-01"
LAYOUT_SHA = "d92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914"
FRUIT = "task_cube_geom"  # Existing red-role contact ABI, with a green visual.
HALF_SIZE_M = (.026, .021, .020)
COLOR = ".32 .62 .13 1"


def fruit_xml(xml):
    root = ET.fromstring(xml)
    fruit = root.find(f"./worldbody/body[@name='task_cube']/geom[@name='{FRUIT}']")
    require(fruit is not None and fruit.get("type") == "box", "original free box required")
    require(fruit.get("mass") is not None, "fruit mass must be explicit in frozen template")
    old = dict(fruit.attrib)
    fruit.set("type", "ellipsoid")
    fruit.set("size", " ".join(map(str, HALF_SIZE_M)))
    fruit.set("rgba", COLOR)
    changed = ET.tostring(root, encoding="unicode")
    fruit.attrib = old
    require(ET.tostring(root, encoding="unicode") == xml, "scene XML changed outside fruit")
    return changed


def protect_model(box, fruit):
    g = box.geom(FRUIT).id
    b = box.body("task_cube").id
    all_geoms = np.arange(box.ngeom) != g
    all_bodies = np.arange(box.nbody) != b
    # XML comparison covers names and settings. Verify compiled fields that could
    # change after MuJoCo recomputes the ellipsoid's inertia and collision bounds.
    for name in (
        "qpos0", "qpos_spring", "key_qpos", "key_qvel", "key_ctrl",
        "dof_damping", "dof_armature", "dof_frictionloss", "jnt_qposadr",
        "actuator_trnid", "actuator_gear", "actuator_ctrlrange", "actuator_gainprm",
        "actuator_biasprm", "sensor_adr", "sensor_objid", "cam_pos", "cam_quat",
        "cam_fovy", "geom_friction", "geom_contype", "geom_conaffinity",
        "geom_solref", "geom_solimp", "body_mass",
    ):
        assert_array_equal(getattr(box, name), getattr(fruit, name), err_msg=name)
    for name in ("geom_type", "geom_size", "geom_rgba", "geom_rbound", "geom_aabb"):
        assert_array_equal(getattr(box, name)[all_geoms], getattr(fruit, name)[all_geoms],
                           err_msg=name)
    for name in ("body_inertia", "body_ipos", "body_iquat"):
        assert_array_equal(getattr(box, name)[all_bodies], getattr(fruit, name)[all_bodies],
                           err_msg=name)
    require(fruit.opt.timestep == .002 and fruit.nmocap == box.nmocap
            and fruit.nu == box.nu and fruit.neq == box.neq, "control/clock ABI changed")
    return float(fruit.body_mass[b])


def pin_module(spec, module, relative):
    path = Path(module.__file__).resolve()
    require(path == (Path(spec["source_root"]) / relative).resolve(), "different module origin")
    pin(path, spec["source_files"][relative])


def compile_two(site, archive, out):
    import mujoco
    from go2arx5_vla.world_model import multitask_scene_v6 as scene
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.wm02 import multitask_scene_audit_v6 as assets

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and mujoco.__version__ == "3.2.7"
            and mujoco.get_mjcb_control() is None, "CPU-only MuJoCo 3.2.7 without control callback")
    sealed = read(pin(site / "v18/sealed-Z1/members.json",
                      read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")[
                          "member_manifest_sha256"]))
    key = "Z1_REACH_483/G00-A/EXECUTION_SPEC.json"
    spec = read(pin(site / "v18" / key, sealed[key]))
    pin_module(spec, scene, "src/go2arx5_vla/world_model/multitask_scene_v6.py")
    pin_module(spec, assets, "scripts/wm02/multitask_scene_audit_v6.py")
    baseline = read(pin(archive / BASELINE / "COLD_INPUT.private.json", BASELINE_SHA))
    previous = read(pin(archive / LAYOUT_ARCHIVE / "population/COMPILED_POPULATION.private.json",
                        LAYOUT_SHA))
    protected = registered_historical_scenes(archive, baseline)
    slots = {r["id"]: r for r in previous["clusters"] if r["id"] in DEV_LAYOUTS}
    require(set(slots) == set(DEV_LAYOUTS)
            and {r["split"] for r in slots.values()} == {"TRAIN"}
            and {r["family"] for r in slots.values()} == {"F1"}, "fixed TRAIN layout carriers")
    template = load_ref(baseline["templates"]["F1"])
    template["asset_root"], _ = assets.asset_binding(template)
    model_path = template["model_path"]
    base = mujoco.MjModel.from_binary_path(str(pin(Path(model_path), template["frozen_files"][model_path])))
    roles = check_contact_roles(spec, "HOLD-red")

    rows = []
    out.mkdir(mode=0o700)
    for layout in DEV_LAYOUTS:
        card = slots[layout]["card"]
        xml, geometry = scene.overlay_xml(template, base, card)
        with forbid_integration(mujoco) as calls:
            box = mujoco.MjModel.from_xml_string(xml)
            require(scene.robot_contract(base, box)["fingerprint"] == ROBOT,
                    "frozen robot described by the archived compiler")
            model = mujoco.MjModel.from_xml_string(fruit_xml(xml))
            require(model.geom_type[model.geom(FRUIT).id] == mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    "explicit ellipsoid required")
            mass = protect_model(box, model)
            geometry["object_half_sizes"]["red"] = list(HALF_SIZE_M)
            static = verify_geometry(mujoco, model, {"positions_world_m": card["object_positions_world"]})
            path = out / layout / "SCENE.private.mjb"
            path.parent.mkdir(mode=0o700)
            mujoco.mj_saveModel(model, str(path), None)
        require(not calls, "cold scene compiler attempted integration")
        model_sha = sha(path)
        require(model_sha not in protected, "new scene repeats sealed historical scene bytes")
        protected.add(model_sha)
        rows.append(dict(layout=layout, development_only=True, task="HOLD-red",
                         positions_world_m=card["object_positions_world"], scoring_geometry=geometry,
                         scene_sha256=model_sha, scene_bytes=path.stat().st_size,
                         target_mass_kg=mass, static=static))
    return dict(status="TWO_FRUIT_SCENES_COMPILED_COLD_ONLY", layouts=rows,
                fruit_half_size_m=HALF_SIZE_M, fruit_shape="ellipsoid",
                fruit_geometry_is_nominal_proxy=True, robot_fingerprint=ROBOT,
                frozen_contact_roles=roles, source_layout_manifest_sha256=LAYOUT_SHA,
                protected_historical_scenes=len(protected) - len(rows),
                native_dispatch_qualified=False, new_native_requests=0, new_integrations=0,
                new_rendered_frames=0, new_model_generations=0, new_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compile_two(args.site, args.archive, args.out)
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
