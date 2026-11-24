"""Restore the two TRAIN geometries and exercise the existing noncontact entry checks on the original states."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from known_dynamic_panel import KnownCurve
from moving_stop_card import baseline, configure, record, sha

SLOTS = ("CHV1-TRAIN-F1-HOLD-red-00", "CHV1-TRAIN-F1-HOLD-red-01")


def inspect(root, name, source, registered, frozen):
    import mujoco
    from d1_geometry import sweep, sweep_profile
    from feedback import OPEN
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding

    from scripts.pi0292.eval_contract import matrix10

    plans = read(
        pin(
            source / "REGISTERED_PANEL.private.json",
            registered["registered_panel_sha256"],
        )
    )
    plan = next(x for x in plans if x["id"] == name)
    case = source / "cases" / name
    out = case / "native-01"
    spec = read(case / "EXECUTION_SPEC.json")
    old = registered["episodes"][name]
    require(
        name.startswith("CHV1-TRAIN-F1-HOLD-red-")
        and plan["condition"]["mode"] == "M_FREE",
        "wrong historical profile",
    )
    pin(case / "RESULT.private.json", old["result_sha256"])
    scene = pin(Path(spec["model_path"]), old["scene"])
    require(
        str(scene) == plan["scene"]["path"] and plan["scene"]["sha256"] == old["scene"],
        "scene roster",
    )
    loaded = read(out / "LOADED_EXECUTOR_00.private.json")
    actual = {k: loaded[k]["executable_sha256"] for k in frozen}
    require(actual == frozen, "different sim/solver/controller")
    for k, expected in actual.items():
        pin(Path(loaded[k]["executable"]), expected)
    model = mujoco.MjModel.from_binary_path(str(scene))
    data = mujoco.MjData(model)
    names = [model.geom(i).name for i in range(model.ngeom)]
    # Historical two-object packets use their own sealed ABI. Keep the current
    # receiver's one-object parser in a separate process and preserve its verdict.
    overlay = Path(spec["source_root"])
    module = "src/go2arx5_vla/world_model/multitask_state_v6.py"
    pin(overlay / module, spec["source_files"][module])
    pin(
        overlay / "scripts/pi0294/task_state.py",
        spec["source_files"]["scripts/pi0294/task_state.py"],
    )
    reader = """
import json,sys
from go2arx5_vla.world_model.multitask_state_v6 import packets
row=next(x for x in packets(sys.argv[1]) if x['step']==5703)
print(json.dumps({k:row[k] for k in ('step','qpos','contacts','time_s','contact_time_s')},default=lambda x:x.tolist()))
"""
    state = json.loads(
        subprocess.check_output(
            [spec["python"], "-B", "-c", reader, str(out / "state.bin.journal")],
            cwd=overlay,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(overlay),
                        str(overlay / "src"),
                        spec["native_environment"].get("PYTHONPATH", ""),
                    ]
                ),
            },
        )
    )
    atom = next(
        x
        for x in map(json.loads, (out / "MOBILE_SAMPLES.private.jsonl").open())
        if x["native_end_step"] == 5703
    )
    data.qpos[:] = state["qpos"]
    mujoco.mj_forward(model, data)
    sample = {"qpos": data.qpos.copy(), "tcp": np.array(atom["actual_TCP"])}
    require(
        np.allclose(atom["qpos"], data.qpos, atol=1e-9, rtol=0)
        and atom["safety_valid"]
        and atom["solver_exact_installed_targets"],
        "real prior boundary",
    )
    require(
        np.allclose(data.site("x5_ee").xpos, sample["tcp"][:3, 3], rtol=0, atol=1e-8),
        "FK",
    )
    previous, anchor = (
        matrix10(np.array(atom[k]))
        for k in ("installed_EEF10", "target_endpoint_EEF10")
    )
    original = next(
        map(
            json.loads,
            (root / "v17/train-attempt01" / name / "native_audit.jsonl").open(),
        )
    )
    goal = np.array(original["raw_task_intent"]["goal"])
    direction = goal - sample["tcp"][:3, 3]
    direction /= np.linalg.norm(direction)
    offset = direction * 0.020
    base = np.array(atom["actual_mount"])
    geom_contacts = sorted(
        {
            tuple(sorted((names[j["geom0"]], names[j["geom1"]])))
            for j in state["contacts"]
        }
    )
    support_error = None
    try:
        KnownCurve.support(names, state)
    except ValueError as e:
        support_error = str(e)
    geom_error = None
    try:
        sweep(
            model,
            sample,
            {"old_ee": anchor, "old_base": base},
            anchor,
            base,
            binding(spec["native_root"]),
        )
    except ValueError as e:
        geom_error = str(e)
    # Twenty millimetres along the archived teacher's approach at fixed base. The
    # commanded seam begins at the real previous command. Future states are unmeasured.
    ts = np.arange(121) * 0.05
    u = np.clip(ts / 4, 0, 1)
    progress = u**3 * (10 - 15 * u + 6 * u * u)
    path = np.repeat(anchor[None], len(ts), axis=0)
    path[:, :3, 3] += progress[:, None] * offset
    gripper = float(atom["target_endpoint_EEF10"][9])
    issues = []
    for i in range(41):
        j = 2 * i
        knots = path[j : j + 41].copy()
        knots[-1] = knots[-2]
        receipt = {
            "start_pose": previous if j == 0 else path[j - 1],
            "end_pose": path[j],
            "finger_fraction": gripper,
        }
        try:
            KnownCurve.admit(knots, receipt, sample)
        except ValueError as e:
            issues.append({"update": i, "reason": str(e)})
    module = "src/go2arx5_vla/world_model/multitask_contact_v6.py"
    verified = pin(overlay / module, spec["source_files"][module])
    contact_spec = importlib.util.spec_from_file_location(
        "archived_reach_contact", verified
    )
    predicates = importlib.util.module_from_spec(contact_spec)
    contact_spec.loader.exec_module(predicates)

    def permitted(pair, distance):
        # H0 Reach additionally prohibits *all* robot-object contact.
        return predicates.observer_permitted(*pair, distance, "red") and not (
            set(pair) & set(predicates.OBJECT_GEOMS.values())
            and "task_table" not in pair
        )

    require(
        not any(
            permitted((pad, obj), 0.0)
            for pad in predicates.PADS
            for obj in predicates.OBJECT_GEOMS.values()
        ),
        "unexpected pad/object permission",
    )
    require(
        not permitted(("task_left_pad", "task_table"), 0.0)
        and not permitted(("task_blue_cube_geom", "task_table"), -0.005),
        "clearance negative controls",
    )
    require(
        all(
            permitted((names[p["geom0"]], names[p["geom1"]]), p["distance_m"])
            for p in state["contacts"]
        ),
        "original scene contact rejects entry",
    )
    geometry = {
        "requested_unique_goals": 81,
        "completed_unique_goals": 0,
        "swept_poses": 0,
        "first_failure": None,
        "original_two_object_contact_sha256": sha(verified),
    }
    root_body = model.body("base_link").id
    robot = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        while body and body != root_body:
            body = int(model.body_parentid[body])
        if body == root_body and (model.geom_contype[g] or model.geom_conaffinity[g]):
            robot.append(g)
    require(len(robot) >= 18, "robot collision geometry coverage")
    obj = [model.geom(k).id for k in predicates.OBJECT_GEOMS.values()]
    geometry["robot_collision_geoms"] = len(robot)
    geometry["minimum_robot_to_either_cube_mm"] = 1000.0

    def witness(data):
        clearance = min(
            float(mujoco.mj_geomDistance(model, data, g, o, 0.25, None))
            for g in robot
            for o in obj
        )
        geometry["minimum_robot_to_either_cube_mm"] = min(
            geometry["minimum_robot_to_either_cube_mm"], clearance * 1000
        )
        require(clearance >= 0.015, "15 mm Reach clearance")

    for j in range(81):
        try:
            g = sweep_profile(
                model,
                sample,
                {"old_ee": anchor, "old_base": base},
                path[j],
                base,
                binding(spec["native_root"]),
                (41, 38, 23),
                permitted,
                witness,
            )
        except ValueError as ex:
            geometry["first_failure"] = {"knot": j, "reason": str(ex)}
            break
        geometry["completed_unique_goals"] += 1
        geometry["swept_poses"] += g["poses_checked"]
        geometry["minimum_joint_margin_rad"] = min(
            geometry.get("minimum_joint_margin_rad", 10.0),
            g["minimum_joint_margin_rad"],
        )
    centers = {
        color: data.geom(plan["geometry"]["objects"][color]["geom"]).xpos.copy()
        for color in ("red", "blue")
    }
    distances = {
        color: float(np.linalg.norm(sample["tcp"][:3, 3] - v) * 1000)
        for color, v in centers.items()
    }
    held = read(root / "v17/known-dynamic-472/SLOW_A0/EXECUTION_SPEC.json")
    return {
        "episode": name,
        "development_geometry": plan["condition"],
        "source_mode": plan["condition"]["mode"],
        "compiled_scene": old["scene"],
        "initial_state_sha256": hashlib.sha256(
            np.asarray(state["qpos"], dtype="<f8").tobytes()
        ).hexdigest(),
        "historical_base80mm_tick": old["native_event_ticks"]["base80mm"],
        "actual_model_shape": [model.nq, model.nv, model.njnt],
        "recorded_source_commit": spec["source_sha"],
        "source_spec_sha256": sha(case / "EXECUTION_SPEC.json"),
        "loaded": actual,
        "entry_tick": 5703,
        "first_consumed_wire": atom["installed_reference_sha256"],
        "actual_start_world": sample["tcp"].tolist(),
        "base_pose_at_entry": base.tolist(),
        "last_consumed_world": anchor.tolist(),
        "original_teacher_pregrasp": goal.tolist(),
        "instruction_for_student": "Move the gripper toward the red cube and stop before touching it.",
        "candidate_standing_translation_world_m": offset.tolist(),
        "entry_tcp_to_cube_centers_mm": distances,
        "gripper_fraction": gripper,
        "original_open": OPEN,
        "prior_contact_pairs": geom_contacts,
        "frozen_receiver_scene_sha256": held["frozen_files"][held["model_path"]],
        "archived_two_object_kinematic_preview": geometry,
        "original_support_check_error": support_error,
        "original_kinematic_sweep_error": geom_error,
        "full40_plus_consumed_seam_windows": 41,
        "full40_rejects": issues,
        "committed_teacher_evidence": old["prepared"]["artifacts"],
        "current_standing_H0_qualification": False,
    }


def main(args):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from train_visual_progress import P1

    reg = read(BASE / "reports/SLA_GO2_V17/TRAIN_QUALIFICATION.json")
    source = Path(
        read(pin(args.root / "evidence/P1_REGISTRATION.json", P1))["target_windows"][0][
            "episode_pin"
        ]["path"]
    ).parents[3]
    frozen = read(BASE / "reports/SLA_GO2_V17/DYNAMIC_472/FROZEN_EXECUTORS.json")[
        "sha256_by_card"
    ]["SLOW_A0"]
    eligible = [
        v for v in reg["episodes"].values() if v["status"].startswith("verified_")
    ]
    with forbid_integration(mujoco) as calls:
        results = [inspect(args.root, n, source, reg, frozen) for n in SLOTS]
    require(not calls and len(eligible) == 57, "cohort or integration changed")
    record(
        args.out,
        {
            "eligible_teachers": len(eligible),
            "base80mm_events": sum(
                "base80mm" in v["native_event_ticks"] for v in eligible
            ),
            "selection": "First two distinct TRAIN F1 HOLD-red qualified geometry identifiers in lexical order, subject to both views localizing red at the first query.",
            "results": results,
            "native_starts": 0,
            "physics_ticks": 0,
            "model_forwards": 0,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    configure(baseline(args.root))
    sys.path.insert(0, str(args.root / "v12/CPU_LIB"))
    main(args)
