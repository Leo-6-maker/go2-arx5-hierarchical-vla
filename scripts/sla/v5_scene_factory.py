"""Compile the fixed first V5 reach and mobile-grasp qualification scenes on CPU."""

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


BASE = "4185a00c6ea6f868394ed8e550c9a720b2a0f3490c0765b8014df3c14c3d5ef9"
PLAN = "2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05"
ROSTER = (
    ("EF5_Q_REACH_01", "F1", "REACH_ORIENT", "red", (-5, 7), (7, -5), 702151),
    ("EF5_Q_REACH_02", "F2", "REACH_ORIENT", "blue", (5, -7), (-7, 5), 702152),
    ("EF5_Q_MOBILE_01", "F1", "MOBILE_GRASP", "blue", (3, -9), (-8, 6), 702153),
    ("EF5_Q_MOBILE_02", "F2", "MOBILE_GRASP", "red", (-3, 9), (8, -6), 702154),
)


def build(source, protected, out):
    import mujoco as mj
    from scripts.pi0292.durable import write
    from scripts.wm02.multitask_population_v6 import compile_population
    from go2arx5_vla.world_model.native_replay import forbid_integration

    spec = read(pin(source, BASE))
    catalog = read(protected)
    existing = {row["scene_sha256"] for row in catalog["episodes"]}
    existing_reset = {row["reset"] for row in catalog["episodes"]}
    clusters = []
    conditions = []
    for identity, family, skill, target, red, blue, reset in ROSTER:
        card = deepcopy(spec["prototype_cards"][family])
        for color, (dx, dy) in zip(("red", "blue"), (red, blue), strict=True):
            xyz = card["object_positions_world"][color]
            xyz[:2] = [xyz[0] + dx * 1e-3, xyz[1] + dy * 1e-3]
        require(reset not in existing_reset, "The proposed reset is already recorded in TRAIN")
        clusters.append(dict(id=identity, family=family, split="QUAL", stage="QUAL", reset=reset, card=card))
        conditions.append(dict(id=identity, cluster=identity, split="QUAL", family=family, reset=reset,
                               skill=skill, target=target, original_2ms=True, physics_eligible=False))
    out.mkdir(mode=0o700)
    prepared = dict(schema="SLA_CORE_EMBODIMENT_FACTORY_V5_S1", plan=PLAN, clusters=clusters,
                    conditions={"QUAL": conditions}, independent_target_and_distractor_positions=True)
    report = compile_population(dict(spec, registered_root=str(out), deadline_epoch=time.time()+1800,
                physics=False, new_physical_requests_budget=0, new_model_generations_budget=0), proposed=prepared)
    sizes = []
    with forbid_integration(mj) as calls:
        for cluster in clusters:
            scene = cluster["scene"]
            require(scene["sha256"] not in existing, "The compiled scene is in the signed TRAIN archive")
            model = mj.MjModel.from_binary_path(str(pin(scene["path"], scene["sha256"])))
            data = mj.MjData(model)
            mj.mj_fwdPosition(model, data)
            geoms = [model.geom(name) for name in ("task_cube_geom", "task_blue_cube_geom")]
            positions = [data.geom(geom.id).xpos.copy() for geom in geoms]
            table = model.geom("task_table")
            masses = [float(model.body_mass[geom.bodyid[0]]) for geom in geoms]
            for geom, position in zip(geoms, positions, strict=True):
                require(abs(position[2]-geom.size[2]-cluster["scoring_geometry"]["table_top_world_z"]) < 1e-9
                        and np.all(np.abs(position[:2]-table.pos[:2])+geom.size[:2] < table.size[:2]),
                        "Compiled object lost the expected tabletop support")
            require(min(masses) > 0 and np.any(np.abs(positions[0][:2]-positions[1][:2]) >= .045),
                    "The compiled scene has invalid object mass or overlap")
            sizes.append(dict(id=cluster["id"], body_world_xyz=[p.tolist() for p in positions], masses_kg=masses,
                              geom_friction=[g.friction.tolist() for g in geoms], scene_sha256=scene["sha256"]))
    require(not calls, "Scene Factory screening attempted to integrate")
    write(out / "ACTUAL.json", dict(report, plan=PLAN, fixed_roster=conditions, geometry=sizes,
          inherited_template_sha256=BASE, existing_train_manifest_sha256=sha(protected), native_integrations=0,
          native_task_complete=0, renderer_qualified=False, external_assets=[]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "protected", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
