"""Compare measured and accepted height after the first native PLACE grasp."""

import argparse
import json
from pathlib import Path

import numpy as np

from v64_adapt_pregrasp_attribution import digest, journal, read, trajectory


ATOM_TICKS = 25
FOLLOW_TICKS = 5000


def analyze(case, comparison):
    spec_path = case / "EXECUTION_SPEC.json"
    terminal = case / "RESULT.private.json"
    lease = read(case / "control/LEASE.json")
    spec = read(spec_path)
    result = read(terminal)
    if (lease["status"] != "CLOSED" or lease["spec_sha256"] != digest(spec_path)
            or case.name != comparison["id"] or digest(terminal) != comparison["result_sha256"]
            or not result["shutdown_clean"] or result["technical_failure"]
            or result["actual_native_integrations"] != comparison["native_ticks"]
            or spec["interface_comparison_link"] != "L1" or not spec["task_instruction_id"].startswith("PLACE-")):
        raise ValueError("Use the original, paired, closed L1 PLACE archive and independent score")

    common = dict(case=case.name, seed=spec["comparison_policy_seed"], task=spec["task_instruction_id"],
                  scene_sha256=spec["frozen_files"][spec["model_path"]], terminal_sha256=digest(terminal))
    grasp = comparison["task_events"].get("target_grasp")
    if grasp is None:
        return dict(common, first_grasp_tick=None)
    grasp_tick = grasp["native_step"]
    samples, archive = trajectory(case)
    selected = {step: row for step, row in samples.items() if grasp_tick <= step < grasp_tick + FOLLOW_TICKS}
    if not selected:
        raise ValueError("No confirmed post-grasp interval was retained")
    expected_start = min(selected)
    height_fields = ("eef_command", "actual_tcp", "arm_mount", "object_center")
    height = []
    for atom in journal(case / "native-01/MOBILE_SAMPLES.private.jsonl"):
        step = atom["native_end_step"]
        if step not in selected:
            continue
        sample = selected[step]
        heights = [sample["reference"][2], sample["tcp"][2], sample["mount"][2], atom["object_pose"][2][3]]
        if not np.isfinite(heights).all() or not atom["safety_valid"] or atom["segment"] != "MOBILE_TASK":
            raise ValueError("Only confirmed and safe physical task samples are eligible")
        height.append((step, sample["grasp"], heights))
    if ([row[0] for row in height] != list(selected) or expected_start - grasp_tick >= ATOM_TICKS):
        raise ValueError("Actual grasp and first measured task boundary must be within 50 ms")

    origin = np.asarray(height[0][2])
    relative_mm = 1000 * (np.asarray([row[2] for row in height]) - origin)
    residual_mm = 1000 * (origin[1] - origin[0]) + relative_mm[:, 1] - relative_mm[:, 0]
    at_object_peak = int(np.argmax(relative_mm[:, 3]))
    peak = relative_mm.argmax(axis=0)
    return dict(
        common, archive=archive, first_grasp_tick=grasp_tick, first_boundary=expected_start,
        confirmed_intervals=len(height), horizon_ticks=FOLLOW_TICKS, actual_grasp_intervals=sum(row[1] for row in height),
        initial_tcp_minus_command_mm=float(residual_mm[0]),
        maximum_rise_mm=dict(zip(height_fields, relative_mm.max(axis=0).tolist(), strict=True)),
        maximum_rise_tick={field: height[index][0] for field, index in zip(height_fields, peak, strict=True)},
        at_object_peak_mm=dict(zip(height_fields, relative_mm[at_object_peak].tolist(), strict=True)),
        tcp_minus_command_at_object_peak_mm=float(residual_mm[at_object_peak]),
        maximum_native_lift_m=comparison["maximum_lift_m"],
    )


def run(output, comparison_path, cell, cases):
    comparison = read(comparison_path)
    (pair,) = (pair for pair in comparison["pairs"] if pair["candidate_1k"]["id"].endswith("_" + cell))
    originals = {"V6.5 1K": pair["candidate_1k"], **pair["baselines"]}
    if len(cases) != len(originals) or {case.name for case in cases} != {row["id"] for row in originals.values()}:
        raise ValueError("Retain the entire group of registered same-condition models")
    by_name = {case.name: case for case in cases}
    reports = {key: analyze(by_name[value["id"]], value) for key, value in originals.items()}
    if len({(r["seed"], r["task"], r["scene_sha256"]) for r in reports.values()}) != 1:
        raise ValueError("The matched model conditions differ")
    with output.open("x") as stream:
        json.dump(dict(cell=cell, comparison_sha256=digest(comparison_path), reports=reports,
                       analysis_window_s=FOLLOW_TICKS * .002, new_physics_ticks=0, new_model_forwards=0),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("comparison_path", type=Path)
    parser.add_argument("cell")
    parser.add_argument("cases", type=Path, nargs="+")
    run(**vars(parser.parse_args()))
