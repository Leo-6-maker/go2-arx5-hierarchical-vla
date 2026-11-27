"""Check the fixed red-right successor against actual achieved PLACE states."""

import argparse
import json
from pathlib import Path

import numpy as np

from v62_mobile_qualification import DIVERSE_WAVE6_TERMINAL, digest, plain


CANDIDATE = "EF5_DIVERSE_PLACE_013"
WITNESS = "EF5_DIVERSE_PLACE_012"
OBJECTS = ("task_cube_geom", "task_blue_cube_geom")
TARGET_PADS = ("task_left_pad", "task_right_pad")


def sweep(wave, compiled, output):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    summary = wave / "RESULT.private.json"
    if digest(summary) != DIVERSE_WAVE6_TERMINAL or plain(wave / "control/LEASE.json")["status"] != "CLOSED":
        raise ValueError("The archived witness wave must be closed")
    admitted, = (row for row in plain(summary)["records"] if row["condition"] == WITNESS)
    case = wave / "cases" / WITNESS
    spec = plain(case / "EXECUTION_SPEC.json")
    terminal = case / "RESULT.private.json"
    result = plain(terminal)
    if (digest(terminal) != admitted["result"]["sha256"] or not admitted["admitted"]
            or not result["metrics"]["task_and_required_motion_success"] or result["unsafe"]
            or result["technical_failure"] or not result["shutdown_clean"]):
        raise ValueError("Use a full, safe, completely admitted PLACE witness")
    cluster, = plain(compiled)["clusters"]
    old_path = Path(spec["model_path"])
    new_path = Path(cluster["scene"]["path"])
    if (cluster["id"] != CANDIDATE or cluster["family"] != "F1"
            or digest(old_path) != spec["frozen_files"][str(old_path)]
            or digest(new_path) != cluster["scene"]["sha256"]):
        raise ValueError("Scene identity mismatch")
    old = mj.MjModel.from_binary_path(str(old_path))
    model = mj.MjModel.from_binary_path(str(new_path))
    if (old.nq, old.nv, old.ngeom) != (model.nq, model.nv, model.ngeom):
        raise ValueError("The successor must preserve the frozen robot and joint structure")
    robot = [i for i in range(model.ngeom) if
             (model.body(model.geom_bodyid[i]).name.startswith("x5_") or model.geom(i).name in TARGET_PADS)
             and (model.geom_contype[i] or model.geom_conaffinity[i])]
    pads = {model.geom(name).id for name in TARGET_PADS}
    data = mj.MjData(model)
    objects = [model.geom(name) for name in OBJECTS]
    qpos = {}
    for g in objects:
        body = model.body(g.bodyid[0])
        joint = model.jnt(body.jntadr[0])
        if body.jntnum[0] != 1 or joint.type[0] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("Each object must have a separate free joint")
        i = int(joint.qposadr[0])
        qpos[g.name] = (i, model.qpos0[i:i+2] - old.qpos0[i:i+2])
    distances = {g.name: .10 for g in objects}
    closest = {}
    trace = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    samples = 0
    with trace.open() as stream, forbid_integration(mj) as calls:
        for row in map(json.loads, stream):
            if row["native_start_step"] < 5703:
                continue
            if not row["safety_valid"] or row["ticks"] != 25:
                raise ValueError("The witness contains an unconfirmed or unsafe atom")
            data.qpos[:] = row["qpos"]
            for index, shift in qpos.values():
                data.qpos[index:index+2] += shift
            mj.mj_fwdPosition(model, data)
            for g in objects:
                for arm in robot:
                    if g.name == OBJECTS[0] and arm in pads:
                        continue
                    distance = mj.mj_geomDistance(model, data, g.id, arm, .10, None)
                    if distance < distances[g.name]:
                        distances[g.name] = distance
                        closest[g.name] = dict(tick=row["native_end_step"], geom=model.geom(arm).name)
            samples += 1
    if calls or samples != result["metrics"]["task_atoms"]:
        raise ValueError("Expected all historical actual 50 ms atoms with zero integrations")
    write(output, dict(candidate=CANDIDATE, compiled_population_sha256=digest(compiled),
          original_result_sha256=digest(terminal), task_trace_sha256=digest(trace), samples=samples,
          minimum_clearance_mm={k: 1000*v for k, v in distances.items()}, closest=closest,
          object_xy_shifts_mm={k: (1000*v).tolist() for k, (_, v) in qpos.items()},
          target_pad_pairs_excluded=True, new_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wave", "compiled", "output"):
        parser.add_argument(name, type=Path)
    sweep(**vars(parser.parse_args()))
