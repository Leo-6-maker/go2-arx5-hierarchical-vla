"""Compile one early Teacher preview at each sealed fifth-cohort G7 rejection."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_CLOSED_M, WORKING_OPEN_M
from go2arx5_vla.world_model.multitask_contact_v6 import OBJECT_GEOMS, PADS, allowed_pair, observer_permitted
from go2arx5_vla.world_model.multitask_score_v6 import Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import packets, score_through
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read, write
from scripts.sla.v64_recovery_early_teacher import prepare_early, rejected_boundary
from v5_mobile_container_corridor import solve
from v64_recovery_carry import ATOM_TICKS, PREFIX_TICKS, checked_plan
from v64_recovery_release_prefix import BoundaryReader


FIFTH_CLOSE = "06cbfdc794d9fd05a6336c8e15a4f6017b946e13aaac067625003bad93d7771e"
CASES = ("V64_RECOVERY_TRAIN_09", "V64_RECOVERY_TRAIN_10")
STUDENT_START = 5703


def ref(path):
    return dict(sha256=digest(path))


def static_prefix(model, sample, program, color, tcp_id, holding):
    """Sample the intended TCP at the original P10 grid with the measured base fixed."""
    data = mujoco.MjData(model)
    data.qpos[:] = sample["qpos"]
    path = program.decoded_H40[:11]
    rotations = rot6d_rows_to_matrix(path[:, 3:9])
    arm = [model.joint(f"x5_joint{i}").id for i in range(1, 7)]
    jaws = [model.joint(name).id for name in ("task_left_slide", "task_right_slide")]
    moving = arm + jaws
    joints = model.jnt_qposadr[moving]
    limits = model.jnt_range[moving]
    fingers = model.jnt_qposadr[jaws]
    object_address = int(model.joint(OBJECT_GEOMS[color].removesuffix("_geom") + "_free").qposadr[0])
    initial_object = data.qpos[object_address:object_address + 3].copy()
    arm_geoms = [i for i in range(model.ngeom) if model.geom(i).name in PADS or (
        model.geom_contype[i] and model.body(int(model.geom_bodyid[i])).name.startswith("x5_"))]
    environment = [model.geom(name).id for name in ("task_table", *OBJECT_GEOMS.values())]
    pairs = [(a, b) for a in arm_geoms for b in environment
             if not allowed_pair(model.geom(a).name, model.geom(b).name, color)]
    checked, reject = [], None
    for knot, (target, rotation) in enumerate(zip(path, rotations, strict=True)):
        data.qpos[fingers] = sample["finger_q"] + (target[9] - path[0, 9]) * (WORKING_OPEN_M - WORKING_CLOSED_M)
        if holding:
            data.qpos[object_address:object_address + 3] = initial_object + target[:3] - path[0, :3]
        if not solve(mujoco, model, data, target[:3], rotation, tcp_id, arm):
            reject = dict(knot=knot, reason="STATIC_IK")
            break
        if np.any(data.qpos[joints] < limits[:, 0]) or np.any(data.qpos[joints] > limits[:, 1]):
            reject = dict(knot=knot, reason="STATIC_JOINT_LIMIT")
            break
        denied = [dict(geometries=(model.geom(c.geom1).name, model.geom(c.geom2).name), distance_m=float(c.dist))
                  for c in data.contact if not observer_permitted(model.geom(c.geom1).name, model.geom(c.geom2).name,
                                                                 c.dist, color)]
        if denied:
            reject = dict(knot=knot, reason="STATIC_CONTACT", contacts=denied)
            break
        gap, a, b = min((mujoco.mj_geomDistance(model, data, a, b, .3, None), a, b) for a, b in pairs)
        nearest = dict(knot=knot, forbidden_geometry_gap_m=float(gap), geometry_names=(model.geom(a).name, model.geom(b).name),
                       robot_body=model.body(int(model.geom_bodyid[a])).name)
        if gap <= 0:
            reject = dict(nearest, reason="STATIC_CLEARANCE")
            break
        checked.append(nearest)
    return dict(knot_count=len(path), passed_knots=len(checked), first_failure=reject,
                closest_pair=min(checked, key=lambda row: row["forbidden_geometry_gap_m"], default=None),
                movable_arm=True, held_target_assumed_rigid_translation=holding,
                original_contact_predicate=True, native_motion_qualified=False)


def clock_fixture(teacher, program, margin, sample, base):
    """Exercise commit and replan clocks from fixed sensor values without physics or labels."""
    span = WORKING_OPEN_M - WORKING_CLOSED_M
    last = PREFIX_TICKS // ATOM_TICKS - 1
    rows = []
    for index in range(5):
        start = sample["native_step"]
        raw = program.decoded_H40
        final_width = float(WORKING_CLOSED_M + span * raw[last, 9])
        rows.append(dict(step=start, stage=teacher.stage, confirmed_as_fixture_only=True,
                         extreme_speed_m_s=margin["maximum_speed_m_s"],
                         first_width_m=float(WORKING_CLOSED_M + span * raw[0, 9]), assumed_last_width_m=final_width))
        end = start + PREFIX_TICKS
        reference = teacher.commit(program, end)
        if index == 0:
            try:
                teacher.decide(sample, reference, base)
            except ValueError as error:
                if str(error) != "CONTINUOUS_TEACHER_REQUIRES_EXACT_CONSUMED_P10_CLOCK":
                    raise
            else:
                raise ValueError("Early Teacher accepted a repeated decision clock")
        if index < 4:
            sample = dict(sample, native_step=end, time_s=end * .002)
            previous = dict(width_m=final_width, source_native_s=(end - ATOM_TICKS) * .002, query_native_s=end * .002)
            teacher, program, _, margin = checked_plan(teacher, sample, reference, previous, base)
    return dict(fixed_measured_sensor_values=True, command_history_simulated=True,
                rejected_repeated_clock=True, planned_prefixes=rows)


def rejected_boundary_fixture(rejected, consumed, step):
    decision = rejected["rejection"]
    invalid = (
        ("native_stop", dict(rejected, rejection=dict(decision, reason="NATIVE_SAFETY_STOP")), consumed, step),
        ("already_installed", dict(rejected, rejection=dict(decision, installed=True)), consumed, step),
        ("install_ack", dict(rejected, install_ACKs=[{"fixture_only": True}]), consumed, step),
        ("partial_student_prefix", rejected, dict(consumed, full_P10=False), step),
        ("advanced_clock", rejected, consumed, step + 1),
    )
    refused = []
    for name, proposal, prefix, tick in invalid:
        try:
            rejected_boundary(proposal, prefix, tick)
        except ValueError:
            refused.append(name)
        else:
            raise ValueError(f"Accepted an invalid early handoff boundary: {name}")
    return refused


def preview(initial, sample, result, spec, rejected, consumed, previous, base,
            static_geometry=None, check_planner_clock=False):
    teacher, candidate, evidence, margin = prepare_early(initial, sample, result, spec, rejected, consumed, previous, base)
    raw = candidate.decoded_H40
    span = WORKING_OPEN_M - WORKING_CLOSED_M
    width = previous["width_m"]
    np.testing.assert_allclose(WORKING_CLOSED_M + span * raw[0, 9], width, rtol=0, atol=1e-12)
    tcp = np.asarray(sample["tcp"])[:3, 3]
    report = dict(stage=evidence["stage"], original_teacher_motion_path=True,
                maximum_lift_m=result["maximum_lift_m"], native_target_grasp=result["target_grasp"],
                previous_width_m=width, candidate_end_width_m=float(WORKING_CLOSED_M + span * raw[-1, 9]),
                desired_P10_delta_m=(raw[10, :3] - raw[0, :3]).tolist(),
                teacher_goal_delta_m=(np.asarray(evidence["privileged_goal"]) - tcp).tolist(),
                original_H40_shadow_margin=margin, preview=candidate.evidence())
    if static_geometry is not None:
        model, tcp_id = static_geometry
        color = Task.parse(spec["task_instruction_id"]).color
        report["static_prefix"] = static_prefix(model, sample, candidate, color, tcp_id, result["target_grasp"])
    if check_planner_clock:
        report["counterfactual_clock_fixture"] = clock_fixture(teacher, candidate, margin, sample, base)
        report["invalid_boundary_fixtures_refused"] = rejected_boundary_fixture(rejected, consumed, sample["native_step"])
    return report


def scan(case, record, check_static, check_planner_clock):
    path = case / "RESULT.private.json"
    terminal = read(path)
    spec_path = case / "EXECUTION_SPEC.json"
    spec = read(spec_path)
    lease = read(case / "control/LEASE.json")
    scene = Path(spec["model_path"])
    rejection = terminal["detail"]["valid_policy_rejection"]
    if (path != Path(record["result"]["path"]) or digest(path) != record["result"]["sha256"]
            or lease["status"] != "CLOSED" or lease["spec_sha256"] != digest(spec_path)
            or digest(scene) != spec["frozen_files"][str(scene)] or terminal["unsafe"] or terminal["technical_failure"]
            or not terminal["shutdown_clean"] or spec["collection_split"] != "TRAIN"
            or spec["sim_only_role"] != "STUDENT" or spec["WBC_changes"] or spec["safety_changes"]
            or spec["hardware_connections_allowed"] or "ORIGINAL_G7_RATE" not in rejection["reason"]):
        raise ValueError("Use the exact closed fifth-cohort TRAIN student G7 rejections")

    native = case / "native-01"
    journal = native / "state.bin.journal"
    step, query = terminal["actual_native_integrations"], rejection["query"]
    rejected_path = native / "g7-queries" / f"{query:04d}/QUERY_ACTUAL.private.json"
    consumed_path = native / "g7-queries" / f"{query - 1:04d}/QUERY_ACTUAL.private.json"
    rejected, consumed = read(rejected_path), read(consumed_path)
    if rejected["rejection"] != rejection:
        raise ValueError("The archived policy rejection differs from the closed result")
    meta = read(rejected_path.with_name("INPUT_META.private.json"))
    previous = meta["previous_reference"]
    if meta["previous_reference_basis"] != "CONFIRMED_COMPLETE_NATIVE_ATOM_COMMAND_NOT_MEASURED_APERTURE":
        raise ValueError("Require the previous command from the actual consumer")

    task = spec["task_instruction_id"]
    model = mujoco.MjModel.from_binary_path(str(scene))
    reader = BoundaryReader(model, Task.parse(task).color)
    with forbid_integration(mujoco) as calls:
        initial = next(reader.update(row) for row in packets(journal) if row["step"] == STUDENT_START)
    if calls:
        raise ValueError("Original state extraction attempted integration")
    score = Score(task, initial, spec["multitask_geometry"])
    with journal.open("rb") as stream:
        measured = score_through(stream, reader, score, step)
    recorded = terminal["detail"]["score"]
    if (measured["events"] != recorded["events"] or measured["target_grasp"] != recorded["target_grasp"]
            or not np.isclose(measured["maximum_lift_m"], recorded["maximum_lift_m"], rtol=0, atol=1e-10)):
        raise ValueError("The full original trajectory did not reproduce its scores")

    with (native / "COMMANDS.private.jsonl").open() as commands:
        base = next(row["measured_BASE_pose_hold"] for row in map(json.loads, commands)
                    if row.get("measured_BASE_pose_hold") is not None)
    with forbid_integration(mujoco) as calls:
        result = preview(initial, reader.latest, measured, spec, rejected, consumed, previous, base,
                         (model, reader.tcp) if check_static else None, check_planner_clock)
    if calls:
        raise ValueError("The Teacher preview may not integrate")
    return dict(case=case.name, task=task, native_step=step, actual_rejected_query=query,
                archived_terminal=ref(path), physical_state=ref(journal), rejected_receipt=ref(rejected_path),
                last_actual_prefix=ref(consumed_path), issued_wire=consumed["confirmed_intervals"][-1]["wire_sha256"],
                original_score_events=measured["events"], candidate=result)


def run(closure, cases, output, check_static=False, check_planner_clock=False):
    report = read(closure)
    if (digest(closure) != FIFTH_CLOSE or not report["complete_reconciliation"]
            or report["terminal_cases"] != 2 or tuple(case.name for case in cases) != CASES):
        raise ValueError("Require the entire fixed fifth cohort")
    results = [scan(case, row, check_static, check_planner_clock) for case, row in zip(cases, report["cases"], strict=True)]
    write(output, dict(fifth_cohort=ref(closure), cases=results, native_integrations=0,
                      model_forwards=0, optimizer_updates=0, new_training_labels=0))
    print(json.dumps(dict(receipt=ref(output), cases=[dict(case=row["case"], **row["candidate"]) for row in results])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closure", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs=2, type=Path)
    parser.add_argument("--check-static", action="store_true")
    parser.add_argument("--check-planner-clock", action="store_true")
    run(**vars(parser.parse_args()))
