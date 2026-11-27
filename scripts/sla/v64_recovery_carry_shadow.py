"""Replay closed PLACE diagnostics with the V6.4 carry handoff."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M
from go2arx5_vla.world_model.multitask_score_v6 import Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, read_journal_packet
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from v64_recovery_carry import CarryStall, prepare


CLOSED_R5 = "6c4d553d8cf6fc3989fedbbd98c08b5d7dca626922a4aa2ddcd5323797317496"
CLOSED_A250 = "34bbcc873cc07b64268326cc6b61c813188e704f83ddcadb93dad2262c704022"
CLOSED_A250_RED = "b62d0b4a4bc2bb14a8f22153106fa6dba2b2cb0cb080e9728b6cb6b61da9628b"
ARCHIVES = {CLOSED_R5: "R5", CLOSED_A250: "ADAPT0250", CLOSED_A250_RED: "ADAPT0250"}


def read(path):
    return json.loads(path.read_bytes())


def run(case, output):
    result_path = case / "RESULT.private.json"
    original = digest(result_path)
    if original not in ARCHIVES:
        raise ValueError("Read only a closed, recorded PLACE diagnostic")
    terminal = read(result_path)
    spec = read(case / "EXECUTION_SPEC.json")
    if (read(case / "control/LEASE.json")["status"] != "CLOSED" or not terminal["shutdown_clean"]
            or terminal["unsafe"] or terminal["technical_failure"] or terminal["detail"]["teacher_takeover"]):
        raise ValueError("Require the closed student-only episode")
    geometry = spec["multitask_geometry"]
    task = Task.parse(terminal["detail"]["task_instruction"])
    native = case / "native-01"
    queries = native / "g7-queries"
    consumption = {}
    for path in sorted(queries.glob("*/QUERY_ACTUAL.private.json")):
        command = read(path)
        tick = command["terminal_native_step"]
        if tick in consumption:
            raise ValueError("More than one query ended at the same native tick")
        consumption[tick] = (command, digest(path))
    if len(consumption) != terminal["detail"]["actual_SLA_queries"]:
        raise ValueError("The actual student query denominator changed")
    first, last = min(consumption)-250, max(consumption)
    scene = Path(spec["model_path"])
    if digest(scene) != spec["frozen_files"][str(scene)]:
        raise ValueError("The bound MuJoCo scene changed")
    model = mujoco.MjModel.from_binary_path(str(scene))
    reader = TwoObjectState(model, task.color)
    stall = CarryStall(task, geometry)
    score = initial = pilot = None
    possible = []
    with (native/"COMMANDS.private.jsonl").open() as stream:
        base = next(row["measured_BASE_pose_hold"] for row in map(json.loads, stream)
                    if row.get("measured_BASE_pose_hold") is not None)
    with (native/"state.bin.journal").open("rb") as stream, forbid_integration(mujoco) as calls:
        while (packet := read_journal_packet(stream)) is not None:
            tick = packet["step"]
            if tick < first:
                continue
            if tick > last:
                break
            sample = reader.update(packet)
            if score is None:
                initial = sample
                score = Score(task.instruction, initial, geometry)
            score.update(sample)
            if tick not in consumption:
                continue
            command, sha256 = consumption[tick]
            event = stall.observe(score, sample, command)
            if event is None:
                continue
            event.update(query=command["query"], recorded_student_consumption_sha256=sha256)
            possible.append(event)
            if pilot is None and tick != last:
                previous = read(queries / f"{command['query']+1:04d}/INPUT_META.private.json")["previous_reference"]
                try:
                    teacher, program, _, margin = prepare(initial, sample, spec["mobile_condition"], geometry,
                        task.instruction, command, previous, base, model)
                except ValueError as error:
                    pilot = dict(event, full_H40_compiled=False, reason=str(error))
                    continue
                actual_first_width = float(program.reference_path[0, 9])*MECHANICAL_TRAVEL_M
                if abs(actual_first_width-previous["width_m"]) > 1e-12:
                    raise ValueError("The first candidate Teacher command moved the holding jaw")
                pilot = dict(event, full_H40_compiled=True, teacher_phase=teacher.stage,
                             measured_open_fraction=sample["gripper_open_fraction"],
                             last_student_issued_width_m=previous["width_m"], first_teacher_width_m=actual_first_width,
                             original_full_H40_margin=margin)
    recorded = terminal["detail"]["score"]
    if (calls or score is None or score.last != last or score.states != last-first+1
            or score.events != recorded["events"] or score.unsafe_states or score.gaps
            or not np.isclose(score.maximum_lift, recorded["maximum_lift_m"])):
        raise ValueError("Validate the original scored trajectory")
    write(output, dict(scope="ARCHIVED_EVALUATION_ONLY", model=ARCHIVES[original], closed_result_sha256=original,
          individual_verified_P10=len(consumption), scored_native_states=score.states,
          eligible_stalls=len(possible), first_handoff_candidate=pilot, candidates=possible,
          actual_teacher_takeovers=0, recovery_train_admissions=0, new_native_integrations=0,
          original_reference_installed_by_replay=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
