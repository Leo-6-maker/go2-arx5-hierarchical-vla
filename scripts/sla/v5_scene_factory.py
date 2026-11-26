"""Compile fixed V5 qualification or mobile-grasp production scenes on CPU."""

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


BASE = "4185a00c6ea6f868394ed8e550c9a720b2a0f3490c0765b8014df3c14c3d5ef9"
PLAN = "2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05"
QUAL = "f10b07348ad2746876a3e3ba1399a4b3b950ae5875c965f7c1d909d07b50df39"
F2_REPLAY = "9d0d116a768cfb00f24b42e57e4cb0df1d7ef1e9cf139479ed3b1b6668c697a1"
F2_REPAIR_ID = "EF5_Q_MOBILE_F2_RED_R2"
ROSTER = (
    ("EF5_Q_REACH_01", "F1", "REACH_ORIENT", "red", (-5, 7), (7, -5), 702151),
    ("EF5_Q_REACH_02", "F2", "REACH_ORIENT", "blue", (5, -7), (-7, 5), 702152),
    ("EF5_Q_MOBILE_01", "F1", "MOBILE_GRASP", "blue", (3, -9), (-8, 6), 702153),
    ("EF5_Q_MOBILE_02", "F2", "MOBILE_GRASP", "red", (-3, 9), (8, -6), 702154),
)
MOBILE_TRAIN = (
    ("EF5_T_MOBILE_01", "F1", "MOBILE_GRASP", "blue", (-7, 5), (5, -11), 702301),
    ("EF5_T_MOBILE_02", "F2", "MOBILE_GRASP", "red", (-5, 11), (7, -5), 702302),
    ("EF5_T_MOBILE_03", "F1", "MOBILE_GRASP", "red", (4, 8), (-10, 3), 702303),
    ("EF5_T_MOBILE_04", "F2", "MOBILE_GRASP", "blue", (10, -3), (-4, -8), 702304),
    ("EF5_T_MOBILE_05", "F1", "MOBILE_GRASP", "blue", (-6, -5), (1, -12), 702305),
    ("EF5_T_MOBILE_06", "F2", "MOBILE_GRASP", "red", (-1, 12), (6, 5), 702306),
    ("EF5_T_MOBILE_07", "F1", "MOBILE_GRASP", "red", (8, 4), (-9, 7), 702307),
    ("EF5_T_MOBILE_08", "F2", "MOBILE_GRASP", "blue", (9, -7), (-8, -4), 702308),
)
DIVERSE_PLACE_TRAIN = (
    ("EF5_DIVERSE_PLACE_001", "F1", "DIVERSE_PLACE", "red", (-1, 3), (-5, 0), 702501),
    ("EF5_DIVERSE_PLACE_002", "F2", "DIVERSE_PLACE", "blue", (4, 4), (2, -3), 702502),
)
PLACE_ZONE_OFFSETS_MM = {"EF5_DIVERSE_PLACE_001": (0, -4), "EF5_DIVERSE_PLACE_002": (5, 4)}


def build(source, protected, out, *, qualification=None, protected_population=None, repair=None, diverse_place=False):
    import mujoco as mj
    from scripts.pi0292.durable import write
    from scripts.wm02.multitask_population_v6 import compile_population
    from go2arx5_vla.world_model.native_replay import forbid_integration

    spec = read(pin(source, BASE))
    catalog = read(protected)
    existing = {row["scene_sha256"] for row in catalog["episodes"]}
    existing_reset = {row["reset"] for row in catalog["episodes"]}
    split, roster = "QUAL", ROSTER
    require(repair is None or qualification is not None, "Protect the original QUAL population before a repair")
    require(not diverse_place or (qualification is not None and repair is None),
            "Diverse PLACE requires the existing population exclusions")
    if qualification:
        prior = read(pin(qualification, QUAL))
        require(prior["plan"] == PLAN and
                [row["id"] for row in prior["clusters"]] == [row[0] for row in ROSTER],
                "The immutable QUAL population is required for a disjoint TRAIN wave")
        require(protected_population is not None, "Bind the existing TRAIN and evaluation scene population")
        history = read(protected_population)
        for population in (prior, history):
            existing.update(row["scene"]["sha256"] for row in population["clusters"])
            existing_reset.update(row["reset"] for row in population["clusters"])
        split, roster = "TRAIN", DIVERSE_PLACE_TRAIN if diverse_place else MOBILE_TRAIN
    if repair:
        evidence = read(pin(repair, F2_REPLAY))
        row, = (r for r in evidence["offsets"] if r["blue_offset_mm"] == [-12, -12])
        original, = (r for r in prior["clusters"] if r["id"] == "EF5_Q_MOBILE_02")
        require(evidence["scene_sha256"] == original["scene"]["sha256"] and evidence["native_integrations"] == 0
                and row["feasible_table_and_red"] and row["minimum_arm_clearance_mm"] > 0,
                "Unexpected prior F2 collision or cold geometry")
        # Keep the F2 red target at its exact QUAL position. Only the distractor moves.
        _, family, skill, target, red, blue, _ = ROSTER[-1]
        adjusted = tuple(value + offset for value, offset in zip(blue, row["blue_offset_mm"], strict=True))
        split, roster = "QUAL", ((F2_REPAIR_ID, family, skill, target, red, adjusted, 702401),)
    clusters = []
    conditions = []
    for identity, family, skill, target, red, blue, reset in roster:
        card = deepcopy(spec["prototype_cards"][family])
        for color, (dx, dy) in zip(("red", "blue"), (red, blue), strict=True):
            xyz = card["object_positions_world"][color]
            xyz[:2] = [xyz[0] + dx * 1e-3, xyz[1] + dy * 1e-3]
        if diverse_place:
            right = card["zones"]["right"]["center_world_XY"]
            right[:] = [position + delta * 1e-3 for position, delta in
                        zip(right, PLACE_ZONE_OFFSETS_MM[identity], strict=True)]
        require(reset not in existing_reset, "The proposed reset overlaps a protected experiment")
        existing_reset.add(reset)
        clusters.append(dict(id=identity, family=family, split=split, stage=split, reset=reset, card=card))
        condition = dict(id=identity, cluster=identity, split=split, family=family, reset=reset,
                         skill=skill, target=target, original_2ms=True, physics_eligible=False)
        if diverse_place:
            condition["instruction"] = f"PLACE-{target}-right"
        conditions.append(condition)
    out.mkdir(mode=0o700)
    prepared = dict(schema="SLA_CORE_EMBODIMENT_FACTORY_V5_S1", plan=PLAN, clusters=clusters,
                    conditions={split: conditions}, independent_target_and_distractor_positions=True)
    report = compile_population(dict(spec, registered_root=str(out), deadline_epoch=time.time()+1800,
                physics=False, new_physical_requests_budget=0, new_model_generations_budget=0), proposed=prepared)
    sizes = []
    with forbid_integration(mj) as calls:
        for cluster in clusters:
            scene = cluster["scene"]
            require(scene["sha256"] not in existing, "The compiled scene overlaps a protected experiment")
            existing.add(scene["sha256"])
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
    actual = dict(report, plan=PLAN, fixed_roster=conditions, geometry=sizes,
          inherited_template_sha256=BASE, existing_train_manifest_sha256=sha(protected), native_integrations=0,
          native_task_complete=0, renderer_qualified=False, external_assets=[])
    if qualification:
        actual.update(protected_qualification_sha256=QUAL, protected_population_sha256=sha(protected_population))
    if repair:
        actual["repair_trace_sha256"] = F2_REPLAY
    write(out / "ACTUAL.json", actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "protected", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--protected-population", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--diverse-place", action="store_true")
    build(**vars(parser.parse_args()))
