"""Test an OPEN-phase Teacher handoff on the sealed blue TRAIN placement."""

import argparse
import json
from pathlib import Path

import mujoco

from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from v64_recovery_carry import prepare_supported_release


BLUE_TRAIN = "aeda26a012f10572a1dc78194678a82e9e56c4cf0655e35a7e8af065b21380ef"


def read(path):
    return json.loads(path.read_bytes())


def replay(case, output):
    result_path = case / "RESULT.private.json"
    if digest(result_path) != BLUE_TRAIN:
        raise ValueError("Require the original closed TRAIN blue-left outcome")
    result = read(result_path)
    detail = result["detail"]
    if (read(case / "control/LEASE.json")["status"] != "CLOSED" or not result["shutdown_clean"]
            or result["unsafe"] or result["technical_failure"] or detail["teacher_takeover"]):
        raise ValueError("Require a clean student-only outcome and intact lease")
    raw = case / "native-01"
    spec = read(case / "EXECUTION_SPEC.json")
    scene = Path(spec["model_path"])
    if spec["collection_split"] != "TRAIN" or digest(scene) != spec["frozen_files"][str(scene)]:
        raise ValueError("Unexpected TRAIN scene configuration")
    rejected = detail["valid_policy_rejection"]["query"]
    query = raw / f"g7-queries/{rejected - 1:04d}/QUERY_ACTUAL.private.json"
    consumed = read(query)
    rejected_input = raw / f"g7-queries/{rejected:04d}/INPUT_META.private.json"
    meta = read(rejected_input)
    terminal = detail["score"]["native_step"]
    if terminal != meta["native_step"] or detail["valid_policy_rejection"]["installed"]:
        raise ValueError("Require the original paused boundary before the rejected command")
    start = 5703
    stream = raw / "state.bin.journal"
    model = mujoco.MjModel.from_binary_path(str(scene))
    reader = TwoObjectState(model, "blue")
    with forbid_integration(mujoco) as calls:
        states = {p["step"]: reader.update(p) for p in packets(stream) if p["step"] in (start, terminal)}
        with (raw / "COMMANDS.private.jsonl").open() as commands:
            base = next(r["measured_BASE_pose_hold"] for r in map(json.loads, commands)
                        if r.get("measured_BASE_pose_hold") is not None)
        last = consumed["confirmed_intervals"][-1]
        previous = dict(width_m=last["emitted_gripper_width_m"],
                        source_native_s=last["emitted_gripper_source_native_s"], query_native_s=meta["native_time_s"])
        teacher, program, _, margin = prepare_supported_release(states[start], states[terminal], detail["score"],
            spec["mobile_condition"], spec["multitask_geometry"], detail["task_instruction"], consumed, previous,
            base, model)
    first_width = float(program.reference_path[0, 9]) * MECHANICAL_TRAVEL_M
    if calls or abs(first_width - previous["width_m"]) > 1e-12:
        raise ValueError("The shadow plan must preserve the actual previous command")
    write(output, dict(closed_result_sha256=BLUE_TRAIN, state_journal_sha256=digest(stream),
          last_consumption_sha256=digest(query), next_input_sha256=digest(rejected_input), native_step=terminal,
          teacher_phase=teacher.stage, previous_actual_width_m=previous["width_m"], first_teacher_width_m=first_width,
          original_full_H40_margin=margin, actual_teacher_consumptions=0,
          recovery_train_admissions=0, new_native_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    replay(**vars(parser.parse_args()))
