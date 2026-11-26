"""Localize the first F2 mobile contact and screen distractor offsets offline."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


RESULT = "987c5499236556823e227c6a3b06f1a25d16a431fdfc2cb5ad71d49386f868a2"
OFFSETS_MM = (-12, -8, -4, 0, 4, 8, 12)


def screen(case, out):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    spec = read(case / "EXECUTION_SPEC.json")
    terminal = read(pin(case / "RESULT.private.json", RESULT))
    stop, = terminal["physics_stops"]
    require(read(case / "control/LEASE.json")["status"] == "CLOSED" and stop["reason"] == "ILLEGAL_CONTACT"
            and terminal["trace_audit"]["integrations_after_first_STOP"] == 0 and mj.__version__ == "3.2.7",
            "Only the closed first F2 failure is eligible for this replay")
    step = stop["steps"]
    trace = case / "native-01/physics.jsonl"
    rows = []
    with trace.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["kind"] == "sample" and step-125 <= item["step"] <= step:
                rows.append(item)
    require(rows[-1]["step"] == step and rows[-1]["contact_stop"]
            and all(not row["illegal_contact"] for row in rows[:-1]), "No first forbidden contact in the native trace")

    path = pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])
    model = mj.MjModel.from_binary_path(str(path))
    blue, red, table = (model.geom(name) for name in ("task_blue_cube_geom", "task_cube_geom", "task_table"))
    robot = [i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name.startswith("x5_")
             and (model.geom_contype[i] or model.geom_conaffinity[i])]
    blue_body = model.body(blue.bodyid[0])
    blue_joint = model.jnt(blue_body.jntadr[0])
    require(blue_body.jntnum[0] == 1 and blue_joint.type[0] == mj.mjtJoint.mjJNT_FREE, "The distractor must have its own free joint")
    xy = int(blue_joint.qposadr[0])
    hits = [c for c in rows[-1]["contacts"] if blue.id in (c["geom1"], c["geom2"])
            and any(i in robot for i in (c["geom1"], c["geom2"]))]
    require(len(hits) == 1, "Expected the archived blue distractor and arm contact")

    data = mj.MjData(model)
    sweeps = []
    with forbid_integration(mj) as calls:
        for dx, dy in product(OFFSETS_MM, repeat=2):
            min_gap = .10
            fits = True
            for row in rows:
                data.qpos[:] = row["qpos"]
                data.qpos[xy:xy+2] += np.array((dx, dy)) * .001
                mj.mj_fwdPosition(model, data)
                position = data.geom(blue.id).xpos[:2]
                within_table = np.all(np.abs(position-table.pos[:2])+blue.size[:2] < table.size[:2])
                other_gap = mj.mj_geomDistance(model, data, blue.id, red.id, .10, None)
                fits = bool(fits and within_table and other_gap > 0)
                min_gap = min(min_gap, *(mj.mj_geomDistance(model, data, blue.id, i, .10, None) for i in robot))
            sweeps.append(dict(blue_offset_mm=[dx,dy], feasible_table_and_red=fits, minimum_arm_clearance_mm=1000*min_gap))
    require(not calls, "The counterfactual uses frozen achieved states with no integration")
    nominal, = (row for row in sweeps if row["blue_offset_mm"] == [0,0])
    require(nominal["minimum_arm_clearance_mm"] < 0, "The unmodified stop must reproduce the original overlap")
    contact = hits[0]
    write(out, dict(result_sha256=RESULT, native_trace_sha256=sha(trace), scene_sha256=sha(path), stop=stop,
                    native_contact=contact, contact_bodies=[model.body(model.geom_bodyid[contact[key]]).name
                    for key in ("geom1", "geom2")], fixed_achieved_samples=len(rows), offsets=sweeps,
                    native_integrations=0, model_forwards=0))
    print(sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path)
    screen(**vars(parser.parse_args()))
