"""Compile the preregistered independent VAL/TEST scenes without starting physics."""

import argparse
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from reserve_teacher_holdouts import BASE as BASELINE_SHA
from reserve_teacher_holdouts import generate, load_ref

ROSTER = "fa87f8250f4e7fa9711d841637be5090dfc59fbd7232387b4c955405c4672011"
ROBOT = "67d4b804c438beabec9790ecf40b60584d2b45ec0c5f1799f377bf784f316bc8"
BASELINE = "sla-goal-v2-train-hold4-cold-20261120-01"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def registered_historical_scenes(archive, baseline):
    inventory = load_ref(baseline["split_inventory"])
    refs = [
        *inventory["original_populations"],
        inventory["new_evaluation_population"],
        *[r for k, r in baseline.items() if k.endswith("_population")],
        {
            "path": str(
                archive / BASELINE / "population/COMPILED_POPULATION.private.json"
            ),
            "sha256": "297b21cc67baa4b2cbdd78af68d244b550337b4cb9213fd27b26818e3958bec0",
        },
        {
            "path": str(
                archive
                / "sla-core-hopper-v1-geometries-20261121-01/population/COMPILED_POPULATION.private.json"
            ),
            "sha256": "d92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914",
        },
    ]
    protected = set(inventory["protected_scene_hashes"])
    for ref in {r["sha256"]: r for r in refs}.values():
        protected.update(c["scene"]["sha256"] for c in load_ref(ref)["clusters"])
    protected.update(c["sha256"] for c in load_ref(inventory["C20_scene_catalog"]))
    protected.update(
        c["scene"]["sha256"] for c in load_ref(inventory["V52_split"])["layouts"]
    )
    return protected


def verify_geometry(mujoco, model, registered):
    from go2arx5_vla.world_model.native_replay import forbid_integration

    require(
        (model.nq, model.nv, model.nu, model.njnt) == (41, 38, 20, 23)
        and model.opt.timestep == 0.002,
        "frozen scene ABI or clock changed",
    )
    d = mujoco.MjData(model)
    with forbid_integration(mujoco) as calls:
        mujoco.mj_kinematics(model, d)
        support = model.geom("task_table").id
        top = d.geom_xpos[support, 2] + model.geom_size[support, 2]
        edges = {}
        for color, name in (("red", "task_cube_geom"), ("blue", "task_blue_cube_geom")):
            idx = model.geom(name).id
            p = d.geom_xpos[idx]
            require(
                np.allclose(
                    p, registered["positions_world_m"][color], rtol=0, atol=1e-10
                ),
                f"compiled {color} position differs from reserved roster",
            )
            require(
                np.isclose(p[2] - model.geom_size[idx, 2], top, rtol=0, atol=1e-10),
                "cube unsupported in default geometry",
            )
            gap = (
                model.geom_size[support, :2]
                - np.abs(p[:2] - d.geom_xpos[support, :2])
                - model.geom_size[idx, :2]
            )
            require(np.all(gap >= 0.005 - 1e-10), "cube footprint leaves table inset")
            edges[color] = float(1000 * np.min(gap))
        ids = [
            model.geom(name).id for name in ("task_cube_geom", "task_blue_cube_geom")
        ]
        gap = np.abs(np.diff(d.geom_xpos[ids, :2], axis=0)[0]) - model.geom_size[
            ids, :2
        ].sum(axis=0)
        require(
            np.max(gap) >= 0.005 - 1e-10 and not calls,
            "cube separation or no-integration invariant",
        )
    return dict(
        default_cube_edge_clearance_mm=edges,
        default_cube_separation_mm=float(1000 * np.linalg.norm(np.maximum(gap, 0))),
    )


def compile_holdouts(site, archive, out):
    from scripts.wm02 import multitask_population_v6 as compiler
    from go2arx5_vla.world_model import multitask_scene_v6 as scene
    import mujoco

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "cold-only runtime")
    roster = read(pin(BASE / "reports/SLA_GO2_V17/HELD_OUT_ROSTER.json", ROSTER))
    require(
        generate(archive) == roster, "historical split separation or roster changed"
    )
    baseline = read(pin(archive / BASELINE / "COLD_INPUT.private.json", BASELINE_SHA))
    sealed = read(
        pin(
            site / "v18/sealed-Z1/members.json",
            read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")[
                "member_manifest_sha256"
            ],
        )
    )
    key = "Z1_REACH_483/G00-A/EXECUTION_SPEC.json"
    receipt = read(pin(site / "v18" / key, sealed[key]))
    for module, rel in (
        (compiler, "scripts/wm02/multitask_population_v6.py"),
        (scene, "src/go2arx5_vla/world_model/multitask_scene_v6.py"),
    ):
        require(
            Path(module.__file__).resolve()
            == (Path(receipt["source_root"]) / rel).resolve(),
            "wrong installed compiler",
        )
        pin(Path(module.__file__), receipt["source_files"][rel])
    protected = registered_historical_scenes(archive, baseline)
    clusters = []
    conditions = {"VAL": [], "TEST": []}
    for r in roster["roster"]:
        card = deepcopy(baseline["prototype_cards"][r["family"]])
        card["object_positions_world"] = deepcopy(r["positions_world_m"])
        clusters.append(
            dict(
                id=r["layout"],
                stage=r["split"],
                split=r["split"],
                family=r["family"],
                reset=r["reset"],
                scene_seed=r["seed"],
                card=card,
            )
        )
        conditions[r["split"]].append(
            dict(
                id=r["episode"],
                cluster=r["layout"],
                family=r["family"],
                instruction=r["task"],
                reset=r["reset"],
                seed=r["seed"],
            )
        )
    require(
        {k: len(v) for k, v in conditions.items()} == {"VAL": 8, "TEST": 12},
        "wrong split membership",
    )
    proposed = dict(
        schema="SLA_GO2_INDEPENDENT_HELDOUT_COLD_SCENES",
        clusters=clusters,
        conditions=conditions,
        roster_sha256=ROSTER,
        native_dispatch_qualified=False,
    )
    out.mkdir(mode=0o700)
    report = compiler.compile_population(
        dict(
            baseline,
            registered_root=str(out),
            deadline_epoch=time.time() + 600,
            physics=False,
            new_physical_requests_budget=0,
            new_model_generations_budget=0,
        ),
        proposed=proposed,
    )
    manifest_path = out / "COMPILED_POPULATION.private.json"
    manifest = read(pin(manifest_path, report["population_sha256"]))
    require(
        report["protection_fingerprints"] == [ROBOT]
        and manifest["conditions"] == conditions
        and not report["native_dispatch_qualified"],
        "model protection, membership or execution boundary changed",
    )
    slots = {}
    for slot, compiled in zip(roster["roster"], manifest["clusters"], strict=True):
        require(slot["layout"] == compiled["id"], "compiled roster mismatch")
        ref = compiled["scene"]
        p = pin(Path(ref["path"]), ref["sha256"])
        require(
            ref["sha256"] not in protected, "compiled scene reuses historical model"
        )
        binding = verify_geometry(mujoco, mujoco.MjModel.from_binary_path(str(p)), slot)
        slots[slot["episode"]] = dict(
            scene_sha256=ref["sha256"],
            scene_bytes=p.stat().st_size,
            split=slot["split"],
            family=slot["family"],
            task=slot["task"],
            seed=slot["seed"],
            reset=slot["reset"],
            **binding,
        )
    require(
        len({r["scene_sha256"] for r in slots.values()}) == 20,
        "compiled geometry duplicated",
    )
    return dict(
        status="COMPILED_STATIC_ONLY",
        roster_sha256=ROSTER,
        compiler_source_sha256=sha(Path(compiler.__file__)),
        scene_source_sha256=sha(Path(scene.__file__)),
        model_protection_fingerprint=ROBOT,
        protected_historical_scenes=len(protected),
        registered_condition_counts=report["registered_condition_counts"],
        compiled_manifest_sha256=sha(manifest_path),
        slots=slots,
        training_updates=0,
        new_integrations=0,
        native_requests=0,
        model_forwards=0,
        collected_VAL=0,
        collected_TEST=0,
        native_dispatch_qualified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "archive", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    actual = compile_holdouts(
        args.site.resolve(), args.archive.resolve(), args.out.resolve()
    )
    print(json.dumps(actual, indent=2, allow_nan=False))
