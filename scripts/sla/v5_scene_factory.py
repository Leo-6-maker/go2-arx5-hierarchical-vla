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
REACH_TRAIN = (
    ("EF5_T_REACH_01", "F1", "REACH_ORIENT", "red", (-3, 9), (9, -8), 702601),
    ("EF5_T_REACH_02", "F2", "REACH_ORIENT", "blue", (7, -5), (-9, 7), 702602),
)
DIVERSE_PLACE_TRAIN = (
    ("EF5_DIVERSE_PLACE_001", "F1", "DIVERSE_PLACE", "red", (-1, 3), (-5, 0), 702501),
    ("EF5_DIVERSE_PLACE_002", "F2", "DIVERSE_PLACE", "blue", (4, 4), (2, -3), 702502),
)
PLACE_ZONE_OFFSETS_MM = {"EF5_DIVERSE_PLACE_001": (0, -4), "EF5_DIVERSE_PLACE_002": (5, 4)}
DIVERSE_PLACE_TRAIN_V2 = (
    ("EF5_DIVERSE_PLACE_003", "F1", "DIVERSE_PLACE", "red", (5, 2), (-9, 3), 702503),
    ("EF5_DIVERSE_PLACE_004", "F2", "DIVERSE_PLACE", "red", (3, -6), (-5, 9), 702504),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_003=(3, -6), EF5_DIVERSE_PLACE_004=(-3, 4))
DIVERSE_PLACE_TRAIN_V3 = (
    ("EF5_DIVERSE_PLACE_005", "F1", "DIVERSE_PLACE", "blue", (-6, 2), (2, 3), 702505),
    ("EF5_DIVERSE_PLACE_006", "F2", "DIVERSE_PLACE", "red", (3, -6), (-17, 21), 702506),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_005=(3, -2), EF5_DIVERSE_PLACE_006=(-3, 4))
DIVERSE_PLACE_TRAIN_V4 = (
    ("EF5_DIVERSE_PLACE_007", "F1", "DIVERSE_PLACE", "blue", (-2, 7), (4, 1), 702507),
    ("EF5_DIVERSE_PLACE_008", "F1", "DIVERSE_PLACE", "red", (3, 4), (-6, 3), 702508),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_007=(2, 1), EF5_DIVERSE_PLACE_008=(1, -5))
DIVERSE_PLACE_TRAIN_V5 = (
    ("EF5_DIVERSE_PLACE_009", "F1", "DIVERSE_PLACE", "blue", (6, -4), (-5, 5), 702509),
    ("EF5_DIVERSE_PLACE_010", "F1", "DIVERSE_PLACE", "red", (-5, -2), (-1, 6), 702510),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_009=(-2, 3), EF5_DIVERSE_PLACE_010=(-4, -3))
DIVERSE_PLACE_TRAIN_V6 = (
    ("EF5_DIVERSE_PLACE_011", "F1", "DIVERSE_PLACE", "blue", (-8, 4), (3, 3), 702511),
    ("EF5_DIVERSE_PLACE_012", "F1", "DIVERSE_PLACE", "red", (2, 0), (-9, -2), 702512),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_011=(4, -4), EF5_DIVERSE_PLACE_012=(-1, -6))
DIVERSE_PLACE_TRAIN_V7 = (
    ("EF5_DIVERSE_PLACE_013", "F1", "DIVERSE_PLACE", "red", (4, -2), (-12, 4), 702513),
)
PLACE_ZONE_OFFSETS_MM.update(EF5_DIVERSE_PLACE_013=(1, -4))
PLACE_WAVE5_HISTORY = (
    "838cd05519316ee5fc16d97a051d9a707e9203fdf4d452ba842f9885c8da723b",
    "25c15ab27d2fa528dca803432f7de2f4d0c3a423f13255c427da7099ad9bd6fe",
    "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e",
    "7a9e350319006ce628a718f533de2dd520222dc8c3042a7b738697957762418e",
)
PLACE_WAVE6_HISTORY = (*PLACE_WAVE5_HISTORY, "664aee826dcb87082ec6b26ef17d700dd07775afff60ce4a090cb12a6eca4d22")
F2_PLACE_DESCEND = "5578d875d85f925d501a835f79692ba023ee50b4f84924a8d8a21457686c20c1"
PLACE_WAVE7_HISTORY = (*PLACE_WAVE6_HISTORY, "7a5f0a2466cc139c589112b186bdfcd7f76c2fd6b610fc84b43359ea1a15b001")
RECENT_PLACE_ROSTERS = {5: (DIVERSE_PLACE_TRAIN_V5, PLACE_WAVE5_HISTORY),
                        6: (DIVERSE_PLACE_TRAIN_V6, PLACE_WAVE6_HISTORY),
                        7: (DIVERSE_PLACE_TRAIN_V7, PLACE_WAVE7_HISTORY)}
F2_PLACE_WAVE2 = "25c15ab27d2fa528dca803432f7de2f4d0c3a423f13255c427da7099ad9bd6fe"
F2_PLACE_WAVE3 = "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e"


def build(source, protected, out, *, qualification=None, protected_population=None, repair=None, diverse_place=False,
          place_wave=1, place_f2_proof=None, previous_place=None, place_history=(), reach_train=False):
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
    require(place_wave == 1 or diverse_place, "Unexpected diverse PLACE population")
    require(not reach_train or (qualification is not None and not diverse_place and repair is None),
            "Reach TRAIN uses the protected qualification population")
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
        if reach_train:
            roster = REACH_TRAIN
        if place_wave == 2:
            roster = DIVERSE_PLACE_TRAIN_V2
        if place_wave == 3:
            replay = read(pin(place_f2_proof, F2_PLACE_DESCEND))
            prior_place = read(pin(previous_place, F2_PLACE_WAVE2))
            original, = (c for c in prior_place["clusters"] if c["id"] == "EF5_DIVERSE_PLACE_004")
            offset, = (r for r in replay["offsets"] if r["blue_offset_mm"] == [-12, 12])
            require(replay["scene_sha256"] == original["scene"]["sha256"] and replay["native_integrations"] == 0
                    and offset["feasible_table_and_red"] and offset["minimum_arm_clearance_mm"] > 0,
                    "The F2 successor must match the archived contact and blue-only clearance result")
            roster = DIVERSE_PLACE_TRAIN_V3
        if place_wave == 4:
            previous = read(pin(previous_place, F2_PLACE_WAVE3))
            existing.update(row["scene"]["sha256"] for row in previous["clusters"])
            existing_reset.update(row["reset"] for row in previous["clusters"])
            roster = DIVERSE_PLACE_TRAIN_V4
        if place_wave in RECENT_PLACE_ROSTERS:
            roster, history_digests = RECENT_PLACE_ROSTERS[place_wave]
            for path, digest in zip(place_history, history_digests, strict=True):
                previous = read(pin(path, digest))
                existing.update(row["scene"]["sha256"] for row in previous["clusters"])
                existing_reset.update(row["reset"] for row in previous["clusters"])
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
            left_placements = {"EF5_DIVERSE_PLACE_004", "EF5_DIVERSE_PLACE_005",
                               "EF5_DIVERSE_PLACE_006", "EF5_DIVERSE_PLACE_007", "EF5_DIVERSE_PLACE_009",
                               "EF5_DIVERSE_PLACE_011"}
            side = "left" if identity in left_placements else "right"
            center = card["zones"][side]["center_world_XY"]
            center[:] = [position + delta * 1e-3 for position, delta in
                         zip(center, PLACE_ZONE_OFFSETS_MM[identity], strict=True)]
        require(reset not in existing_reset, "The proposed reset overlaps a protected experiment")
        existing_reset.add(reset)
        clusters.append(dict(id=identity, family=family, split=split, stage=split, reset=reset, card=card))
        condition = dict(id=identity, cluster=identity, split=split, family=family, reset=reset,
                         skill=skill, target=target, original_2ms=True, physics_eligible=False)
        if diverse_place:
            condition["instruction"] = f"PLACE-{target}-{side}"
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
    if place_wave == 3:
        actual["f2_failed_place_replay_sha256"] = F2_PLACE_DESCEND
    if place_wave == 4:
        actual["protected_previous_place_population_sha256"] = F2_PLACE_WAVE3
    if place_wave in RECENT_PLACE_ROSTERS:
        actual["protected_place_populations_sha256"] = list(history_digests)
    write(out / "ACTUAL.json", actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "protected", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--protected-population", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--diverse-place", action="store_true")
    parser.add_argument("--reach-train", action="store_true")
    parser.add_argument("--place-wave", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--place-f2-proof", type=Path)
    parser.add_argument("--previous-place", type=Path)
    parser.add_argument("--place-history", type=Path, nargs="+", default=())
    build(**vars(parser.parse_args()))
