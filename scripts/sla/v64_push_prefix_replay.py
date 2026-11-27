"""Measure contact clearance on archived robot approaches retargeted to the cold plate."""

import argparse
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v5_mobile_container_corridor import clearance


COLD_SHA = "4b061ce151ab55b86bad397c5aea82a2ed75164cc746be3799e9ad962540e75e"
ORIGINS = ("03f81c894a2a54595dbebf3ebe605328f55a536620a9df4fd5a5521e227dbae4",
           "c93c6e5c9a2330ebcb4136fc449ff84bbd6f046e4a9c8b97001c150f6a453428")


def replay(mj, model, case, card, expected):
    terminal = read(pin(case / "RESULT.private.json", expected))
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED" and terminal["shutdown_clean"] and not terminal["technical_failure"]
            and spec["mobile_condition"]["family"] == card["family"] and spec["task_instruction_id"] == "HOLD-red",
            "Require the matched closed red-object source approach")
    path = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    with path.open() as stream:
        candidates = []
        for line in stream:
            atom = json.loads(line)
            if atom["segment"] != "MOBILE_TASK":
                continue
            if atom["native_grasp"]:
                break
            require(atom["safety_valid"] and atom["native_end_step"] < terminal["actual_native_integrations"]
                    and atom["ticks"] == 25 and len(atom["qpos"]) == model.nq, "Invalid archived pre-contact snapshot")
            candidates.append(atom)
    require(len(candidates) > 100, "Expected a measured free approach prefix")
    target = np.asarray(card["reset_center_m"])
    old_center = np.asarray(candidates[0]["object_pose"])[:3, 3]
    shift = target[:2] - old_center[:2]
    plate, table, other = (model.geom(name).id for name in ("task_cube_geom", "task_table", "task_blue_cube_geom"))
    pads = (model.geom("task_left_pad").id, model.geom("task_right_pad").id)
    links = {i for i in range(model.ngeom) if model.geom_contype[i] and
             model.body(int(model.geom_bodyid[i])).name.startswith("x5_")}
    links.update(pads)
    chassis = model.joint(int(model.body("base_link").jntadr[0]))
    root = int(chassis.qposadr[0])
    robot_end = int(model.joint("task_cube_free").qposadr[0])
    data = mj.MjData(model)
    stats, safe = [], []
    for atom in candidates:
        data.qpos[:robot_end] = atom["qpos"][:robot_end]
        data.qpos[root:root+2] += shift
        mj.mj_fwdPosition(model, data)
        by_pad, by_link, by_table, by_other = (
            clearance(mj, model, data, group, (obstacle,))[0] * 1000
            for group, obstacle in ((pads, plate), (links-set(pads), plate), (links, table), (links, other))
        )
        record = dict(tick=atom["native_end_step"], pad_to_plate_mm=by_pad, nonpad_to_plate_mm=by_link,
                      arm_to_table_mm=by_table, arm_to_distractor_mm=by_other)
        stats.append(record)
        if min(by_link, by_table, by_other) >= 3:
            safe.append(record)
    near = min(safe, key=lambda row: row["pad_to_plate_mm"]) if safe else None
    overlaps = {key:sum(row[key] < 0 for row in stats) for key in (
        "pad_to_plate_mm", "nonpad_to_plate_mm", "arm_to_table_mm", "arm_to_distractor_mm")}
    return dict(id=card["id"], scene_sha256=card["scene_sha256"], source_terminal_sha256=expected,
                source_native_samples_sha256=sha(path), source_was_safety_stopped=terminal["unsafe"],
                snapshots=len(stats), fixed_base_translation_xy_m=shift.tolist(), static_overlap_counts=overlaps,
                diagnostic_clearance_mm=3, minimum_eligible_pad_gap=near,
                eligible_snapshots_within_1_mm=sum(-1 <= row["pad_to_plate_mm"] <= 1 for row in safe))


def examine(cold, first, second, output):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    scenes = read(pin(cold / "ACTUAL.json", COLD_SHA))["rows"]
    with forbid_integration(mj) as effects:
        rows = [
            replay(mj, mj.MjModel.from_binary_path(str(pin(cold / (card["id"]+".mjb"), card["scene_sha256"]))),
                   folder, card, expected)
            for card, folder, expected in zip(scenes, (first, second), ORIGINS, strict=True)
        ]
    require(not effects, "The held-out pose review must not integrate")
    write(output, dict(rows=rows, native_integrations=0, model_forwards=0, native_dispatch_qualified=False))
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "first", "second", "output"):
        parser.add_argument(name, type=Path)
    examine(**vars(parser.parse_args()))
