"""Read a closed PLACE case across model intent, consumed jaw and 2 ms task truth."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco

from go2arx5_vla.world_model.mobile_wbc_reference_v1 import START
from go2arx5_vla.world_model.multitask_score_v6 import LIFT, OPEN_FRACTION, Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_place_first_divergence import queries


# EEF10 jaw fraction and native two-finger state both use the installed .044 m range.
OPEN_RANGE_M = .044


def observe(case, spec, archived, terminal_tick, entry):
    scene = spec["model_path"]
    model = mujoco.MjModel.from_binary_path(str(pin(scene, spec["frozen_files"][scene])))
    task = Task.parse(spec["task_instruction_id"])
    reader = TwoObjectState(model, task.color)
    state_path = case / "native-01/state.bin.journal"
    counts, first = Counter(), {}
    peak_open_in_zone, peak_retreat_in_zone = 0., 0.
    scorer = None
    with forbid_integration(mujoco):
        for packet in packets(state_path):
            tick = packet["step"]
            if tick < START:
                continue
            require(tick <= terminal_tick, "Native data continued past the closed terminal")
            observation = reader.update(packet)
            if scorer is None:
                require(tick == START, "Initial task state was not retained")
                scorer = Score(task.instruction, observation, spec["multitask_geometry"])
            score = scorer.update(observation)
            if entry is None or tick < entry:
                continue
            target = observation["objects"][task.color]
            inside = score["inside_selected_region"]
            support = target["table_contact"]
            opening = score["gripper_open_fraction"]
            motion = score["TCP_retreat_m"]
            counts["states_after_first_region_entry"] += 1
            states = dict(inside=inside, outside=not inside, table_contact=support,
                          in_zone_table_contact=inside and support, grasp=score["target_grasp"],
                          released=score["released"], in_zone_release=inside and score["released"],
                          in_zone_open=inside and opening >= OPEN_FRACTION, stable=score["stable_release_condition"])
            for key, state in states.items():
                if state:
                    counts[key] += 1
                    first.setdefault(key, dict(tick=tick, lift_mm=1000 * score["lift_m"],
                                               opening_fraction=opening, retreat_mm=1000 * motion))
            if inside:
                peak_open_in_zone = max(peak_open_in_zone, opening)
                peak_retreat_in_zone = max(peak_retreat_in_zone, 1000 * motion)
    replay = scorer.result()
    require(replay == archived and replay["native_step"] == terminal_tick, "Current native replay changed the original score")
    return dict(counts=dict(counts), first=first, peak_gripper_open_fraction_in_zone=peak_open_in_zone,
                peak_TCP_retreat_mm_in_zone=peak_retreat_in_zone, journal_sha256=sha(state_path))


def consumed_jaw(case, terminal, entry):
    count, confirmed = 0, []
    decoded = []
    for query in queries(case):
        count += 1
        folder = case / "native-01/g7-queries" / f"{query['index']:04d}"
        report = read(folder / "QUERY_ACTUAL.private.json")
        intervals = report["confirmed_intervals"]
        require(len(intervals) == report["confirmed_atoms"], "Inconsistent original consumption receipt")
        for item in intervals:
            boundary = item["accepted_boundary"]
            require(boundary["exact_receiver_knot"] and boundary["native_step"] == item["native_end_step"]
                    and boundary["accepted_wire_sha256"] == item["wire_sha256"], "Unbound actually emitted jaw command")
            if entry is not None and item["native_end_step"] >= entry:
                confirmed.append(item["emitted_gripper_width_m"])
        if entry is not None and query["tick"] >= entry:
            jaw = query["decoded"][:, 9]
            decoded.append((float(jaw[:10].max()), float(jaw.max())))
    require(count == terminal["model_generations"], "Independent raw action count differs from the closed episode")
    limit = OPEN_FRACTION * OPEN_RANGE_M
    return dict(archived_model_generations=count, queries_started_after_entry=len(decoded),
                p10_open_proposals=sum(a >= OPEN_FRACTION for a, _ in decoded),
                h40_open_proposals=sum(b >= OPEN_FRACTION for _, b in decoded),
                peak_raw_p10_jaw_mm=max((1000 * OPEN_RANGE_M * a for a, _ in decoded), default=None),
                peak_raw_h40_jaw_mm=max((1000 * OPEN_RANGE_M * b for _, b in decoded), default=None),
                confirmed_intervals_from_entry=len(confirmed),
                confirmed_open_intervals=sum(w >= limit for w in confirmed),
                peak_confirmed_jaw_command_mm=max((1000 * w for w in confirmed), default=None))


def audit(case, expected_result):
    result_file = pin(case / "RESULT.private.json", expected_result)
    terminal = read(result_file)
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED" and terminal["shutdown_clean"] and not terminal["technical_failure"],
            "This readout requires a clean original terminal")
    archived = terminal["detail"]["score"]
    require(Task.parse(archived["instruction"]).operation == "PLACE", "PLACE case required")
    tick = terminal["actual_native_integrations"]
    entry = archived["events"].get("correct_region", {}).get("native_step")
    physical = observe(case, spec, archived, tick, entry)
    jaw = consumed_jaw(case, terminal, entry)
    return dict(case=case.name, result_sha256=expected_result, spec_sha256=lease["spec_sha256"],
                first_region_entry_tick=entry, native_ticks=tick, original_score=archived, jaw=jaw, physical=physical,
                open_fraction=OPEN_FRACTION, full_open_command_mm=1000 * OPEN_FRACTION * OPEN_RANGE_M,
                lift_threshold_mm=1000 * LIFT, uses_eval_truth_as_model_input=False,
                new_integrations=0, new_model_forwards=0, new_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("result_sha256")
    parser.add_argument("output", type=Path)
    settings = parser.parse_args()
    write(settings.output, audit(settings.case, settings.result_sha256))
