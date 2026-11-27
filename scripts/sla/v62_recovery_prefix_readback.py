"""Read archived student consumption boundaries for a release Recovery candidate."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.trajectory.so3 import quat_to_matrix
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from go2arx5_vla.world_model.multitask_score_v6 import Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, read_journal_packet
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.pi0292.durable import write


P02 = "6c4d553d8cf6fc3989fedbbd98c08b5d7dca626922a4aa2ddcd5323797317496"


def load(path, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Archived evidence changed for {path.name}")
    return json.loads(data), digest


def preview(root, spec, task, initial, actual, record, lift_origin):
    index = record["query"]
    command, _ = load(root / "g7-queries" / f"{index:04d}/QUERY_ACTUAL.private.json",
                      record["command_receipt_sha256"])
    next_query, _ = load(root / "g7-queries" / f"{index + 1:04d}/INPUT_META.private.json")
    atom = command["confirmed_intervals"][-1]
    boundary = atom["accepted_boundary"]
    prior = next_query["previous_reference"]
    if (not boundary["exact_receiver_knot"] or boundary["native_step"] != next_query["native_step"]
            or prior["width_m"] != atom["emitted_gripper_width_m"]
            or prior["source_native_s"] != atom["emitted_gripper_source_native_s"]):
        raise ValueError("Received reference and consumed gripper must match at the paused boundary")
    pose = np.asarray(boundary["accepted_pose7"])
    reference = np.eye(4)
    reference[:3, 3], reference[:3, :3] = pose[:3], quat_to_matrix(pose[3:])
    with (root / "COMMANDS.private.jsonl").open() as commands:
        base = next(entry["measured_BASE_pose_hold"] for entry in map(json.loads, commands)
                    if entry.get("measured_BASE_pose_hold") is not None)
    teacher = MultitaskTeacher(initial, spec["mobile_condition"], spec["multitask_geometry"], task.instruction)
    teacher.stage = "CARRY"
    teacher.rotation = actual["tcp"][:3, :3].copy()
    teacher.lift_origin = lift_origin.copy()
    teacher.close_start = actual["native_step"] - 1000
    program, _ = teacher.decide(actual, reference, base)
    speed = label_margin_report(program.decoded_H40, prior)
    return dict(native_step=record["native_step"], teacher_phase=teacher.stage, full_H40_compiled=True,
                original_gripper_gate=speed, receiver_selected_reference=boundary["controller_selected_reference"],
                installed=False, collision_check_performed=False, physics_steps=0)


def readback(case, output):
    result, result_sha = load(case / "RESULT.private.json", P02)
    spec, spec_sha = load(case / "EXECUTION_SPEC.json")
    lease, lease_sha = load(case / "control/LEASE.json")
    if (lease["status"] != "CLOSED" or not result["shutdown_clean"] or result["unsafe"]
            or result["technical_failure"] or result["detail"]["teacher_takeover"]):
        raise ValueError("Only the clean closed student trajectory is eligible for this readback")
    task = Task.parse(result["detail"]["task_instruction"])
    root = case / "native-01"
    boundaries = {}
    for path in sorted((root / "g7-queries").glob("*/QUERY_ACTUAL.private.json")):
        row, digest = load(path)
        if row["actual_model_generations"] != 1 or row.get("full_P10") is not True:
            raise ValueError("Expected each P02 query to consume ten real intervals")
        intervals = row["confirmed_intervals"]
        end, start = row["terminal_native_step"], row["native_start_step"]
        if (len(intervals) != 10 or end != start + 250 or end in boundaries or any(
                item["native_end_step"] != start + (i + 1) * 25
                for i, item in enumerate(intervals))):
            raise ValueError("The archived carry candidate must be a confirmed native P10 boundary")
        boundaries[end] = dict(query=row["query"], command_receipt_sha256=digest)
    if len(boundaries) != result["detail"]["actual_SLA_queries"]:
        raise ValueError("The actual query denominator changed")
    first, last = min(boundaries) - 250, max(boundaries)
    geometry = spec["multitask_geometry"]
    scene = Path(spec["model_path"])
    with scene.open("rb") as f:
        scene_sha = hashlib.file_digest(f, "sha256").hexdigest()
    if scene_sha != spec["frozen_files"][str(scene)]:
        raise ValueError("Scene binary differs from the native evaluator binding")
    model = mujoco.MjModel.from_binary_path(str(scene))
    reader = TwoObjectState(model, task.color)
    zone = geometry["zones"][task.destination]
    corners = np.asarray(list(product((-1., 1.), repeat=3))) * geometry["object_half_sizes"][task.color]
    journal = root / "state.bin.journal"
    score = None
    initial = lift_origin = None
    candidates = []
    with journal.open("rb") as stream, forbid_integration(mujoco) as calls:
        while (packet := read_journal_packet(stream)) is not None:
            step = packet["step"]
            if step < first:
                continue
            if step > last:
                break
            actual = reader.update(packet)
            if score is None:
                if step != first:
                    raise ValueError("The original first paused query boundary is missing")
                initial = actual
                score = Score(task.instruction, actual, geometry)
            row = score.update(actual)
            if row["target_grasp"] and lift_origin is None:
                lift_origin = actual["tcp"][:3, 3].copy()
            if (step not in boundaries or score.gaps or score.unsafe_states or score.wrong_contact
                    or score.hold_since is None or actual["time_s"] - score.hold_since < .3
                    or row["inside_selected_region"] or row["base_directional_motion_m"] < .080):
                continue
            pose = actual["object_pose"]
            footprint = corners @ pose[:3, :3].T + pose[:3, 3]
            gap = np.max(np.abs(footprint[:, :2] - zone["center_world_XY"]) - zone["half_size_XY"])
            candidate = dict(boundaries[step], native_step=step, continuous_grasp_lift_s=
                actual["time_s"] - score.hold_since, lift_mm=row["lift_m"] * 1e3,
                original_footprint_gap_mm=float(gap * 1e3))
            candidates.append((candidate, actual))
    if calls or score is None or score.gaps or score.unsafe_states or score.last != last:
        raise ValueError("Incomplete or unsafe native journal, or unexpected simulator integration")
    archived = result["detail"]["score"]
    if score.events != archived["events"] or not np.isclose(score.maximum_lift, archived["maximum_lift_m"]):
        raise ValueError("Native scoring replay differs from the archived result")
    closest, closest_state = min(candidates, key=lambda r: r[0]["original_footprint_gap_mm"])
    first_candidate, first_state = candidates[0]
    previews = [preview(root, spec, task, initial, state, candidate, lift_origin)
                for candidate, state in ((first_candidate, first_state), (closest, closest_state))]
    write(output, dict(parser="ORIGINAL_TWO_OBJECT_ABI_AND_SCORER", scope="ARCHIVED_EVALUATION_ONLY",
          result_sha256=result_sha, spec_sha256=spec_sha, lease_sha256=lease_sha, scene_sha256=scene_sha,
          evaluated_student_prefixes=1, actual_P10_boundaries=len(boundaries),
          original_scored_states=score.states, safe_held_outside_region_boundaries=len(candidates),
          closest_safe_boundary=closest, first_safe_boundary=first_candidate, teacher_previews=previews,
          actual_teacher_takeovers=0,
          recovery_TRAIN_admissions=0, native_integrations=0, model_forwards=0, training_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    readback(**vars(parser.parse_args()))
