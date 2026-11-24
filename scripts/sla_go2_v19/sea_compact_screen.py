"""Bounded, deterministic static screen for two compact standing fruit layouts."""

import argparse
from itertools import product
import json
import os
from pathlib import Path

import numpy as np

from sea_compact_workspace import DEV_LAYOUTS, HALF_SIZE_M, ROBOT, compact_xml, fruit_xml, source, verify_frozen
from sea_compact_preflight import startup, test_geometry
from sea_fruit_h0_cold import pin, require, sha, verify_geometry
from sea_stance_preflight import reached_case
from go2arx5_vla.world_model import multitask_scene_v6 as scene
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration

# Physically supported positions only. Rank static passes by maximum required
# excursion from the archived standing TCP, using a deterministic tie-break.
TOP = (.635, .644, .652)
X = (.260, .275, .290, .305)
Y = (-.053, -.035, -.017, .001, .019)
DISTRACTOR = (.440, .135)
MIN_PAIR_SEPARATION_M = .015


def position(top, x, y):
    return dict(red=[x, y, top+HALF_SIZE_M[2]],
                blue=[*DISTRACTOR, top+HALF_SIZE_M[2]])


def seated_packets(site, role):
    _, entry, files, is_fruit = reached_case(site, role)
    journal = files["state.bin.journal"]
    prefix = []
    for pkt in packets(journal):
        prefix.append(pkt)
        if pkt["step"] == entry:
            break
    require(len(prefix) == entry, "exact full archived startup required")
    return prefix, entry, sha(journal), is_fruit


def candidate_xyz(scene_model, xyz):
    passive = scene_model.qpos0[27:].copy()
    passive[:3], passive[7:10] = xyz["red"], xyz["blue"]
    return passive


def screen_role(models, depth, entry, record):
    passed, rows = [], []
    for top, x, y in product(TOP, X, Y):
        m = models[top]
        xyz = position(top, x, y)
        row = dict(table_top_m=top, positions_world_m=xyz)
        phase = "startup"
        try:
            initial, warmup = startup(m, depth, entry, object_reset=candidate_xyz(m, xyz))
            phase = "task_geometry"
            data = test_geometry(m, initial, dict(scoring_geometry=record))
        except ValueError as error:
            row.update(status="REJECTED", phase=phase, reason=str(error))
        else:
            row.update(status="STATIC_PASS", launch_state_fixture_replayed=warmup, **data)
            passed.append(row)
        rows.append(row)
    passed.sort(key=lambda p: (p["maximum_TCP_excursion_mm"], p["table_top_m"], *p["positions_world_m"]["red"][:2]))
    return rows, passed


def selected_model(old, xml, original_top, geom, row, out, role):
    import mujoco

    m = mujoco.MjModel.from_xml_string(compact_xml(xml, original_top, row["positions_world_m"],
                                                   top=row["table_top_m"]))
    verify_frozen(old, m)
    binding = verify_geometry(mujoco, m, {"positions_world_m": row["positions_world_m"]})
    file = out / f"{role}.mjb"
    mujoco.mj_saveModel(m, str(file), None)
    geometry = dict(geom, table_top_world_z=row["table_top_m"])
    return dict(row, scene_sha256=sha(file), static=binding, scoring_geometry=geometry)


def run(site, archive, prior, out):
    import mujoco

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and mujoco.__version__ == "3.2.7",
            "exact CPU MuJoCo runtime required")
    out.mkdir(mode=0o700)
    original, template, old_rows, layouts = source(archive, prior)
    fixtures = {role: seated_packets(site, role) for role in ("G00", "G01")}
    # Same robot, scene type, duration and policy settings at every cell.
    statuses, options, roots, parents = {}, {}, {}, {}
    with forbid_integration(mujoco) as calls:
        for role, name in zip(fixtures, DEV_LAYOUTS, strict=True):
            box_xml, geometry = scene.overlay_xml(template, original, old_rows[name])
            box = mujoco.MjModel.from_xml_string(box_xml)
            require(scene.robot_contract(original, box)["fingerprint"] == ROBOT, "original robot binding")
            xml = fruit_xml(box_xml)
            roots[role] = xml, geometry["table_top_world_z"]
            parents[role] = mujoco.MjModel.from_binary_path(
                str(pin(prior / name / "SCENE.private.mjb", layouts[name]["scene_sha256"])))
            models = {top: mujoco.MjModel.from_xml_string(
                compact_xml(xml, geometry["table_top_world_z"], position(top, X[0], Y[0]), top=top))
                for top in TOP}
            for m in models.values():
                verify_frozen(parents[role], m)
            records, passes = screen_role(models, *fixtures[role][:2], layouts[name]["scoring_geometry"])
            statuses[role], options[role] = records, passes
        pairs = [
            (a, b) for a, b in product(options["G00"], options["G01"])
            if np.linalg.norm(np.asarray(a["positions_world_m"]["red"]) -
                              np.asarray(b["positions_world_m"]["red"])) >= MIN_PAIR_SEPARATION_M-1e-12
        ]
        selected = {}
        if pairs:
            first = min(pairs, key=lambda ab: (
                max(r["maximum_TCP_excursion_mm"] for r in ab),
                sum(r["maximum_TCP_excursion_mm"] for r in ab)))
            for role, name, chosen in zip(fixtures, DEV_LAYOUTS, first, strict=True):
                depth, entry, _, _ = fixtures[role]
                xml, old_top = roots[role]
                pick = selected_model(parents[role], xml, old_top, layouts[name]["scoring_geometry"],
                                      chosen, out, role)
                # Recheck the final downloadable MJB with no private object-pose overwrite.
                model = mujoco.MjModel.from_binary_path(str(pin(out / f"{role}.mjb", pick["scene_sha256"])))
                init, warmup = startup(model, depth, entry)
                check = test_geometry(model, init, dict(scoring_geometry=pick["scoring_geometry"]))
                require(warmup == pick["launch_state_fixture_replayed"]
                        and abs(check["maximum_TCP_excursion_mm"]-pick["maximum_TCP_excursion_mm"]) < 1e-8
                        and check["close_geometry"] == pick["close_geometry"],
                        "downloadable MJB geometry differs from screen")
                selected[role] = pick
    require(not calls and all(len(v) == len(TOP)*len(X)*len(Y) for v in statuses.values()),
            "screen incomplete or attempted integration")
    return dict(
        status="TWO_STATIC_LAYOUTS_SELECTED" if selected else "NO_STATIC_PAIR",
        grid=dict(table_top_m=TOP, x_m=X, y_m=Y, distractor_xy_m=DISTRACTOR),
        matched_pair_min_spacing_mm=MIN_PAIR_SEPARATION_M*1000,
        cases=statuses, selected=selected,
        carriers={r: dict(journal_sha256=x[2], entry_tick=x[1], actual_fruit_carrier=x[3])
                  for r, x in fixtures.items()},
        new_physics_ticks=0, frozen_model_forwards=0, training_updates=0,
        still_requires_native_qualification=True,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("site", "archive", "parent", "out"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    a = cli.parse_args()
    result = run(a.site.resolve(), a.archive.resolve(), a.parent.resolve(), a.out.resolve())
    (a.out / "SCREEN.json").write_text(json.dumps(result, indent=2) + "\n")
