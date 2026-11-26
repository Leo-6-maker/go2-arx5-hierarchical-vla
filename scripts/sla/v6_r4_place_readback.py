"""Compare requested, consumed, and measured jaw openings after a matched PLACE approach."""

import argparse
import json
from pathlib import Path

import numpy as np

from m38_a_native_v1 import bind_native
from m38_gpu_preflight_v1 import pin, read, require, sha


def audit(case):
    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "Read only a closed native card")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    bind_native(spec)
    from go2arx5_vla.world_model.fusion_trajectory_adapter import (
        MECHANICAL_TRAVEL_M, WORKING_CLOSED_M, WORKING_OPEN_M,
    )
    from go2arx5_vla.world_model.multitask_score_v6 import OPEN_FRACTION

    path = case / "RESULT.private.json"
    terminal = read(path)
    observed = (terminal.get("detail") or {}).get("score") or {}
    start = observed.get("events", {}).get("correct_region", {}).get("native_step")
    row = dict(id=case.name, valid=not terminal["technical_failure"], unsafe=terminal["unsafe"],
               result_sha256=sha(path), task_success=observed.get("task_success", False),
               events=observed.get("events", {}), terminal_target_grasp=observed.get("target_grasp"),
               terminal_released=observed.get("released"), terminal_open_fraction=observed.get("gripper_open_fraction"),
               required_measured_open_mm=1000 * OPEN_FRACTION * MECHANICAL_TRAVEL_M,
               first_region_tick=start, native_integrations=0, model_forwards=0)
    if start is None:
        return row

    raw_h40, raw_p10, emitted, measured = [], [], [], []
    for path in sorted((case / "native-01/g7-queries").glob("*/QUERY_ACTUAL.private.json")):
        query = read(path)
        if query["native_start_step"] >= start:
            action = query["action"]
            with np.load(pin(action["path"], action["sha256"]), allow_pickle=False) as data:
                width = 1000 * (WORKING_CLOSED_M + data["physical_H40"][:, 9] * (WORKING_OPEN_M - WORKING_CLOSED_M))
            raw_h40.append(float(width.max()))
            raw_p10.append(float(width[:10].max()))
        for interval in query.get("confirmed_intervals", []):
            if interval["native_start_step"] >= start:
                emitted.append(1000 * interval["emitted_gripper_width_m"])
                measured.append(1000 * interval["measured_gripper_width_m"])
    return dict(row, returned_queries_after_first_region=len(raw_h40), complete_50ms_intervals_after_first_region=len(emitted),
                max_raw_h40_width_mm=max(raw_h40, default=None), max_raw_p10_width_mm=max(raw_p10, default=None),
                max_emitted_width_mm=max(emitted, default=None), max_measured_width_mm=max(measured, default=None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.case)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
