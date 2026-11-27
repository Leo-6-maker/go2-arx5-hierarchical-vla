"""Locate supported OPEN handoffs on the original four Recovery student prefixes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco

from go2arx5_vla.world_model.multitask_score_v6 import Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets, score_through
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read, write
from v64_recovery_carry import exact_boundary, prepare_supported_release


class BoundaryReader(TwoObjectState):
    def update(self, packet):
        self.latest = super().update(packet)
        return self.latest


def scan(case):
    terminal_path = case / "RESULT.private.json"
    spec_path = case / "EXECUTION_SPEC.json"
    result = read(terminal_path)
    spec = read(spec_path)
    lease = read(case / "control/LEASE.json")
    scene = Path(spec["model_path"])
    task = spec["task_instruction_id"]
    if (lease["status"] != "CLOSED" or lease["spec_sha256"] != digest(spec_path)
            or result["technical_failure"] or spec["collection_split"] != "TRAIN"
            or spec["sim_only_role"] != "STUDENT" or spec["policy_queries"] != 180
            or digest(scene) != spec["frozen_files"][str(scene)]):
        raise ValueError("Use the actual sealed Recovery TRAIN student prefixes")
    native = case / "native-01"
    model = mujoco.MjModel.from_binary_path(str(scene))
    reader = BoundaryReader(model, Task.parse(task).color)
    journal = native / "state.bin.journal"
    with forbid_integration(mujoco) as calls:
        initial = next(reader.update(packet) for packet in packets(journal) if packet["step"] == 5703)
    if calls:
        raise ValueError("Initial-state readback cannot integrate")
    score = Score(task, initial, spec["multitask_geometry"])
    with (native / "COMMANDS.private.jsonl").open() as commands:
        base = next(row["measured_BASE_pose_hold"] for row in map(json.loads, commands)
                    if row.get("measured_BASE_pose_hold") is not None)
    admitted, rejections, steps = [], Counter(), []
    with journal.open("rb") as stream:
        for path in sorted((native / "g7-queries").glob("*/QUERY_ACTUAL.private.json")):
            row = read(path)
            if not row.get("full_P10") or not row.get("confirmed_intervals"):
                continue
            step = row["terminal_native_step"]
            if steps and step != steps[-1] + 250:
                raise ValueError("The confirmed student P10s have a clock gap")
            steps.append(step)
            now = score_through(stream, reader, score, step)
            sample = reader.latest
            if sample["native_step"] != step or score.last != step:
                raise ValueError("The exact actual boundary could not be rescored")
            if not now["ever_target_lift20mm"] or not now["inside_selected_region"]:
                rejections["outside_region_or_no_20mm_lift"] += 1
                continue
            with forbid_integration(mujoco) as calls:
                last = exact_boundary(row, step)
                previous = dict(width_m=last["emitted_gripper_width_m"],
                                source_native_s=last["emitted_gripper_source_native_s"], query_native_s=sample["time_s"])
                try:
                    teacher, _, _, margin = prepare_supported_release(
                        initial, sample, now, spec["mobile_condition"], spec["multitask_geometry"], task, row,
                        previous, base, model)
                except ValueError as error:
                    rejections[str(error)] += 1
                    continue
                admitted.append(dict(query=row["query"], native_step=step, stage=teacher.stage,
                                     previous_width_m=previous["width_m"],
                                     actual_in_region=now["inside_selected_region"],
                                     original_H40_speed=margin, remaining_ticks=spec["max_ticks"]-step))
            if calls:
                raise ValueError("OPEN feasibility may not integrate")
    if not steps or len(steps) != sum(rejections.values()) + len(admitted):
        raise ValueError("Every original fully consumed P10 must be accounted for")
    return dict(case=case.name, task=task, closed_result_sha256=digest(terminal_path),
                physical_state_sha256=digest(journal), full_student_P10_boundaries=len(steps),
                original_terminal_native_step=result["actual_native_integrations"], reached=admitted, rejected=rejections)


def replay(cases, output):
    evidence = [scan(path) for path in cases]
    write(output, dict(cases=evidence, new_model_forwards=0, new_native_integrations=0,
                      new_recovery_admissions=0, new_optimizer_updates=0))
    print(json.dumps(dict(sha256=digest(output), cases=[dict(case=row["case"],
          total=row["full_student_P10_boundaries"], eligible=len(row["reached"]),
          first_tick=row["reached"][0]["native_step"] if row["reached"] else None,
          rejected=row["rejected"]) for row in evidence])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", type=Path, nargs=4)
    options = parser.parse_args()
    replay(options.cases, options.output)
