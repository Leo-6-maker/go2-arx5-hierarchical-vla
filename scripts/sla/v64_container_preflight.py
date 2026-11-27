"""Replay consumption-gated release and the compiled container Teacher path on CPU."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_container_release_contract import PLACE_RESULT, ReleaseTracker, publish_release, release_readback
from v64_container_teacher import COMMAND_CLEARANCE_M, ContainerTeacher, RIM_CLEARANCE_M


def verify_release(case, handoff_path=None):
    import mujoco as mj

    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from v64_recovery_seal import actual_queries, accepted_rows

    expected, _ = release_readback(case)
    result = read(pin(case / "RESULT.private.json", PLACE_RESULT))
    spec = read(case / "EXECUTION_SPEC.json")
    _, queries = actual_queries(case, result["detail"], spec)
    accepted_rows(case, queries, result["actual_native_integrations"])
    raw = case / "native-01"
    start = queries[0]["native_step"]
    intervals = {
        atom["start_native_step"]: (atom["boundary"], query["phase"] == "OPEN")
        for query in queries for atom in query["intervals"]
    }
    with (raw / "MOBILE_SAMPLES.private.jsonl").open() as stream:
        atoms = {row["native_start_step"]: row for row in map(json.loads, stream)
                 if row["native_start_step"] in intervals}
    require(atoms.keys() == intervals.keys(), "Complete actual consume log required")
    path = pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])
    reader = TwoObjectState(mj.MjModel.from_binary_path(str(path)), "red")
    tracked = {}
    bad_mark = {}
    count = 0
    first_publish = None
    with forbid_integration(mj) as effects:
        for packet in packets(raw / "state.bin.journal"):
            step = packet["step"]
            if step < start:
                continue
            state = reader.update(packet)
            if not tracked:
                z = state["object_pose"][2, 3]
                tracked = {name: ReleaseTracker(start, z, "red")
                           for name in ("exact", "no_OPEN", "gap", "recontact", "reversed_jaw")}
            count += 1
            for name, tracker in tracked.items():
                if name == "gap" and step == expected.separation_tick - 2:
                    continue
                if name == "recontact" and step == result["actual_native_integrations"]:
                    wrong = deepcopy(state)
                    wrong["objects"]["red"]["hand_contact"] = True
                    tracker.observe(wrong)
                else:
                    tracker.observe(state)
                if (handoff_path is not None and name == "exact" and step in intervals
                        and first_publish is None
                        and publish_release(handoff_path, "EF5_Q_CONTAINER_01", step, tracker)):
                    first_publish = step
                if step in intervals:
                    accepted, opened = intervals[step]
                    atom = atoms[step]
                    if name == "reversed_jaw" and step == expected.open_ack_tick:
                        atom = deepcopy(atom)
                        atom["installed_EEF10"][9] = 0.
                    tracker.consumed(atom, accepted, opened and name != "no_OPEN")
                    if name == "exact" and step == expected.open_issue_tick:
                        require(tracker.result() is None, "An OPEN command and ACK cannot prove release")
                        for key, change in (("wrong_wire", {"accepted_wire_sha256": "0" * 64}),
                                            ("wrong_knot", {"receiver_knot_time_s": .002 * (step + 26)})):
                            altered = deepcopy(tracker)
                            try:
                                altered.consumed(atoms[step], {**accepted, **change}, opened)
                            except ValueError:
                                bad_mark[key] = True
                            else:
                                raise ValueError(f"Release evidence accepted: {key}")
    status = {name: dict(fault=tracker.fault, release=tracker.result())
              for name, tracker in tracked.items()}
    exact = tracked["exact"]
    active = exact.result()
    require(not effects and active is not None and exact.separation_ticks[0] == expected.separation_tick
            and (active.grasp_tick, active.lift_tick, active.open_issue_tick, active.open_ack_tick)
            == (expected.grasp_tick, expected.lift_tick, expected.open_issue_tick, expected.open_ack_tick)
            and len(bad_mark) == 2
            and all(tracker.result() is None for name, tracker in tracked.items() if name != "exact"),
            f"Release latch replay mismatch against {expected}: {status}")
    return dict(tabletop_native_result=PLACE_RESULT, state_packets=count, exact_accepted_atoms=len(atoms),
                first_separation=asdict(expected), active_release=asdict(active), open_wire=exact.open_wire,
                separation_ticks=exact.separation_ticks, recontact_ticks=exact.recontact_ticks,
                first_published_boundary=first_publish,
                rejected=sorted([*bad_mark, "no_OPEN", "gap", "recontact", "reversed_jaw"]))


def teacher_path(factory, hold_red, hold_blue):
    import mujoco as mj

    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from v5_mobile_container_corridor import COLD, SCORES, terminal_pose, wall_top

    compiled = factory / "SCENE_FACTORY_CONTAINER-PROJECT-01"
    layouts = read(pin(compiled / "ACTUAL.json", COLD))["rows"]
    cases = []
    with forbid_integration(mj) as effects:
        for layout, hold, signature in zip(layouts, (hold_red, hold_blue), SCORES, strict=True):
            identity = layout["id"]
            model = mj.MjModel.from_binary_path(str(pin(compiled / f"{identity}.mjb", layout["compiled_sha256"])))
            color = layout["target"]
            reader = TwoObjectState(model, color)
            original, proof = terminal_pose(hold, signature)
            spec = read(hold / "EXECUTION_SPEC.json")
            geometry = spec["multitask_geometry"]
            raw = hold / "native-01"
            data = reader.data
            fixture = model.body("ef5_static_bowl").id
            data.qpos[:] = original
            mj.mj_fwdPosition(model, data)
            rim_z = max(wall_top(mj, model, data, g) for g in range(model.ngeom)
                        if model.geom_contype[g] and model.geom_bodyid[g] == fixture)
            center = data.xpos[fixture, :2].copy()
            with (raw / "planner.jsonl").open() as stream:
                base = next(np.asarray(poses[0][7:]) for row in map(json.loads, stream)
                            if row["kind"] == "installed" and (poses := row["poses"]) and len(poses[0]) == 14)
            states = packets(raw / "state.bin.journal")
            initial = reader.update(next(p for p in states if p["step"] == 5703))
            held = reader.update(next(p for p in states if p["step"] == proof["tick"]))
            require(held["objects"][color]["native_grasp"], "Use the actual stable HOLD terminal")

            def candidate():
                return ContainerTeacher(initial, spec["mobile_condition"], geometry, identity, center, rim_z)

            def check(sample, teacher, stage):
                program, _ = teacher.decide(sample, sample["tcp"], base)
                require(program.decoded_H40.shape == (40, 10) and teacher.stage == stage,
                        "The unchanged H40 entry must accept the declared stage")
                return program.decoded_H40

            check(initial, candidate(), "PREGRASP")
            transfer = candidate()
            transfer.stage = "HOLD"
            transfer.lift_origin = held["tcp"][:3, 3].copy() - [0., 0., .026]
            transfer.close_start = held["native_step"] - 1000
            raised = check(held, transfer, "CARRY")

            # These candidate poses exercise the original H40 entry and stage
            # predicates. The held state itself comes from the native HOLD.
            above = deepcopy(held)
            target = above["objects"][color]
            cube = target["pose"][:3, 3]
            half_z = abs(target["pose"][2, :3]) @ transfer.half_size
            shift = np.r_[center - cube[:2], rim_z + half_z + COMMAND_CLEARANCE_M - cube[2]]
            cube += shift
            above["object_pose"] = target["pose"]
            above["tcp"][:3, 3] += shift
            target["linear_velocity"][:] = 0.
            above["base_pose"][:2, 3] = transfer.base_origin + .100 * transfer.direction
            below = deepcopy(above)
            below["objects"][color]["pose"][2, 3] -= .005
            check(below, deepcopy(transfer), "CARRY")
            offset = deepcopy(above)
            offset["objects"][color]["pose"][0, 3] += .005
            check(offset, deepcopy(transfer), "CARRY")
            opened = check(above, transfer, "OPEN")
            require(opened[-1, 9] > opened[0, 9] and not np.allclose(raised[:, :3], raised[0, :3]),
                    "Expected actual opening preview and a nonzero raise preview")
            later = deepcopy(above)
            later["native_step"] += 1250
            later["gripper_open_fraction"] = 1.
            later["objects"][color].update(native_grasp=False, hand_contact=True)
            check(later, transfer, "OPEN")
            later["objects"][color]["hand_contact"] = False
            withdrew = check(later, transfer, "WITHDRAW")
            require(withdrew[-1, 2] > withdrew[0, 2], "Withdrawal must move upward under the original H40")
            cases.append(dict(id=identity, source_hold=proof, compiled_scene=layout["compiled_sha256"],
                              rim_world_z_m=rim_z, minimum_declared_clearance_m=RIM_CLEARANCE_M,
                              command_clearance_m=COMMAND_CLEARANCE_M,
                              original_H40_roles=["PREGRASP", "CARRY", "OPEN", "WITHDRAW"],
                              prevented_opening=["below_rim_clearance", "off_center", "contact_after_open"]))
    require(not effects and len(cases) == 2, "Both qualification layouts must be checked with no integration")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "factory", "hold_red", "hold_blue", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = dict(release=verify_release(args.case), teacher=teacher_path(args.factory, args.hold_red, args.hold_blue),
                  native_integrations=0, native_requests=0, container_admissions=0)
    write(args.output, result)
    print(sha(args.output))
