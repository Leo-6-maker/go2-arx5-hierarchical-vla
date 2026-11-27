"""Measure contact clearance on archived robot approaches retargeted to the cold plate."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, require, sha
from v5_mobile_container_corridor import SPACING, clearance, solve


COLD_SHA = "4b061ce151ab55b86bad397c5aea82a2ed75164cc746be3799e9ad962540e75e"
ORIGINS = ("03f81c894a2a54595dbebf3ebe605328f55a536620a9df4fd5a5521e227dbae4",
           "c93c6e5c9a2330ebcb4136fc449ff84bbd6f046e4a9c8b97001c150f6a453428")
FREE_CLEARANCE = .003
PAD_TOLERANCE = .001
SEED_COUNT = 7


def replay(mj, model, case, card, expected, side_model=None):
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
    stats, safe, pad_witnesses, seeds = [], [], [], []
    if side_model is not None:
        side_data = mj.MjData(side_model)
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
            if side_model is not None:
                side_data.qpos[:] = data.qpos
                mj.mj_fwdPosition(side_model, side_data)
                witness, _ = pad_witness(mj, side_model, side_data, card, record["tick"], plate, pads)
                pad_witnesses.append(witness)
                seeds.append((witness, side_data.qpos.copy()))
    near = min(safe, key=lambda row: row["pad_to_plate_mm"]) if safe else None
    overlaps = {key:sum(row[key] < 0 for row in stats) for key in (
        "pad_to_plate_mm", "nonpad_to_plate_mm", "arm_to_table_mm", "arm_to_distractor_mm")}
    result = dict(id=card["id"], scene_sha256=card["scene_sha256"], source_terminal_sha256=expected,
                source_native_samples_sha256=sha(path), source_was_safety_stopped=terminal["unsafe"],
                snapshots=len(stats), fixed_base_translation_xy_m=shift.tolist(), static_overlap_counts=overlaps,
                diagnostic_clearance_mm=3, minimum_eligible_pad_gap=near,
                eligible_snapshots_within_1_mm=sum(-1 <= row["pad_to_plate_mm"] <= 1 for row in safe))
    if side_model is not None:
        result["nativeccd_pad_witnesses"] = pad_witnesses
        nearest = sorted(seeds, key=lambda item: abs(item[0]["initial_pad_gap_mm"]))[:SEED_COUNT]
        result["side_approach_grid"] = approach_grid(mj, side_model, card, nearest, root, plate, pads,
                                                     links, table, other)
    return result


def pad_witness(mj, model, data, card, tick, plate, pads):
    center = data.geom_xpos[plate].copy()
    direction = np.asarray(card["goal_center_XY_m"]) - center[:2]
    direction /= np.linalg.norm(direction)
    pad_gap, _, pad = clearance(mj, model, data, (plate,), pads)
    witnesses = np.zeros(6)
    mj.mj_geomDistance(model, data, plate, pad, .3, witnesses)
    radial = witnesses[:2] - center[:2]
    radius = np.linalg.norm(radial)
    height = witnesses[2] - center[2]
    alignment = -float(radial @ direction) / radius if radius else 0.
    side = (radius >= model.geom_size[plate, 0] - PAD_TOLERANCE
            and abs(height) <= model.geom_size[plate, 1] - PAD_TOLERANCE and alignment >= .7)
    state = dict(tick=tick, pad=model.geom(pad).name, initial_pad_gap_mm=1000*pad_gap,
                 plate_point_m=witnesses[:3].tolist(), point_radius_mm=1000*radius,
                 point_height_mm=1000*height, inward_direction_cosine=alignment, side_facing=bool(side),
                 side_contact=bool(side and abs(pad_gap) <= PAD_TOLERANCE))
    return state, pad


def approach_grid(mj, model, card, seeds, root, plate, pads, links, table, other):
    data = mj.MjData(model)
    site = model.site("x5_ee").id
    arm = np.asarray([model.joint(f"x5_joint{i}").id for i in range(1, 7)])
    joints = model.jnt_qposadr[arm]
    limits = model.jnt_range[arm]
    center = np.asarray(card["reset_center_m"])
    direction = np.r_[np.asarray(card["goal_center_XY_m"]) - center[:2], 0.]
    direction /= np.linalg.norm(direction)
    inward = np.arctan2(-direction[1], -direction[0])
    aligned, eligible = [], []
    attempts, ik_rejections, limit_rejections = 0, 0, 0
    approach_values = SPACING * np.arange(-7, 8)
    lower_values = SPACING * np.arange(18)
    for seed, stock in seeds:
        data.qpos[:] = stock
        mj.mj_fwdPosition(model, data)
        radial = data.geom_xpos[model.geom(seed["pad"]).id, :2] - center[:2]
        for yaw in (0., inward - np.arctan2(radial[1], radial[0])):
            turn = Rotation.from_rotvec([0., 0., yaw])
            shifted = stock.copy()
            shifted[root:root+3] = center + turn.apply(stock[root:root+3] - center)
            orientation = turn * Rotation.from_quat(stock[root+3:root+7][[1, 2, 3, 0]])
            shifted[root+3:root+7] = orientation.as_quat()[[3, 0, 1, 2]]
            data.qpos[:] = shifted
            mj.mj_fwdPosition(model, data)
            tcp = data.site_xpos[site].copy()
            wrist = data.site_xmat[site].reshape(3, 3).copy()
            for advance in approach_values:
                for lower in lower_values:
                    attempts += 1
                    data.qpos[:] = shifted
                    target = tcp + advance*direction - [0., 0., lower]
                    if not solve(mj, model, data, target, wrist, site, arm):
                        ik_rejections += 1
                        continue
                    if np.any(data.qpos[joints] < limits[:, 0]) or np.any(data.qpos[joints] > limits[:, 1]):
                        limit_rejections += 1
                        continue
                    witness, pad = pad_witness(mj, model, data, card, seed["tick"], plate, pads)
                    if not witness["side_facing"]:
                        continue
                    gap, link, obstacle = min(clearance(mj, model, data, links - {pad}, (plate,)),
                                              clearance(mj, model, data, links, (table, other)))
                    row = dict(witness, advance_mm=1000*advance, lowered_mm=1000*lower, base_yaw_rad=yaw,
                               other_gap_mm=1000*gap, other_geometries=[model.geom(i).name for i in (link, obstacle)],
                               other_bodies=[model.body(int(model.geom_bodyid[i])).name for i in (link, obstacle)])
                    aligned.append(row)
                    if gap >= FREE_CLEARANCE:
                        eligible.append(row)
    return dict(seed_ticks=[seed["tick"] for seed, _ in seeds], samples=attempts, ik_rejections=ik_rejections,
                limit_rejections=limit_rejections, side_facing_samples=len(aligned), cleared_samples=len(eligible),
                side_contact_samples=sum(row["side_contact"] for row in eligible),
                closest_side_facing=min(aligned, key=lambda r: abs(r["initial_pad_gap_mm"]), default=None),
                closest_cleared=min(eligible, key=lambda r: abs(r["initial_pad_gap_mm"]), default=None))


def examine(cold, first, second, output, side_contact=False):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    scenes = read(pin(cold / "ACTUAL.json", COLD_SHA))["rows"]
    rows = []
    with forbid_integration(mj) as effects:
        for card, folder, expected in zip(scenes, (first, second), ORIGINS, strict=True):
            path = str(pin(cold / (card["id"]+".mjb"), card["scene_sha256"]))
            model = mj.MjModel.from_binary_path(path)
            side_model = None
            if side_contact:
                side_model = mj.MjModel.from_binary_path(path)
                side_model.opt.enableflags |= int(mj.mjtEnableBit.mjENBL_NATIVECCD)
            rows.append(replay(mj, model, folder, card, expected, side_model))
    require(not effects, "The held-out pose review must not integrate")
    write(output, dict(rows=rows, native_integrations=0, model_forwards=0, native_dispatch_qualified=False))
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "first", "second", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--side-contact", action="store_true", help="screen lateral contact poses via static IK")
    examine(**vars(parser.parse_args()))
