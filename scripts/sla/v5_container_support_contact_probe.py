"""Check whether the existing M_FREE contact rule recognizes intended bowl support."""

import argparse
from itertools import product
from pathlib import Path

import mujoco as mj
import numpy as np

from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted
from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v5_mobile_container_corridor import COLD, OBJECTS, SCORES, terminal_pose, wall_top


OFFSETS_MM = (-5, 0, 5)
CONTACT_DEPTH_M = .0005


def run(factory, hold_red, hold_blue, output):
    layout_root = factory / "SCENE_FACTORY_CONTAINER-PROJECT-01"
    layouts = read(pin(layout_root / "ACTUAL.json", COLD))["rows"]
    samples = []
    with forbid_integration(mj) as effects:
        for layout, history, score in zip(layouts, (hold_red, hold_blue), SCORES, strict=True):
            scene = pin(layout_root / (layout["id"] + ".mjb"), layout["compiled_sha256"])
            model = mj.MjModel.from_binary_path(str(scene))
            data = mj.MjData(model)
            qpos, evidence = terminal_pose(history, score)
            color = layout["target"]
            object_name = OBJECTS[color]
            target = model.geom(object_name + "_geom").id
            floor = model.geom("ef5_static_bowl_base").id
            q = model.jnt_qposadr[model.joint(object_name + "_free").id]
            data.qpos[q+3:q+7] = qpos[q+3:q+7]
            mj.mj_fwdPosition(model, data)
            half_height = abs(data.geom_xmat[target].reshape(3, 3)[2]) @ model.geom_size[target][:3]
            floor_z = wall_top(mj, model, data, floor)
            for dx, dy in product(OFFSETS_MM, repeat=2):
                center = np.array(layout["fixture_world_xyz"][:2]) + .001*np.array([dx, dy])
                data.qpos[q:q+3] = (*center, floor_z + half_height - CONTACT_DEPTH_M)
                mj.mj_fwdPosition(model, data)
                contacts = []
                for item in data.contact:
                    if target not in (item.geom1, item.geom2) or item.dist >= 0:
                        continue
                    a, b = (model.geom(index).name for index in (item.geom1, item.geom2))
                    contacts.append(dict(geoms=[a,b], penetration_mm=-float(item.dist)*1e3,
                                         floor=floor in (item.geom1, item.geom2),
                                         existing_observer_allows=observer_permitted(a,b,item.dist,color)))
                samples.append(dict(id=layout["id"], offset_xy_mm=[dx,dy], source_evidence=evidence,
                                    scene_sha256=layout["compiled_sha256"], contacts=contacts))
    require(not effects and len(samples) == 18, "Incomplete or integrated contact test")
    floors = [row for row in samples if any(item["floor"] for item in row["contacts"])]
    allowed = [row for row in floors if all(item["existing_observer_allows"] for item in row["contacts"])]
    write(output, dict(samples=samples, probe_penetration_mm=CONTACT_DEPTH_M*1e3,
                       floor_contact_samples=len(floors), admitted_by_existing_observer=len(allowed),
                       native_integrations=0, model_forwards=0, training_updates=0))
    print(sha(output), len(floors), len(allowed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("factory", "hold_red", "hold_blue", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
