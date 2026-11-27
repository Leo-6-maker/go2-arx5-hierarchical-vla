"""Read archived jaw intent, emitted commands and measured width after PLACE entry."""

import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v62_source_native import reference


OPEN_DIAGNOSTIC_MM = 39.6


def after_entry(case):
    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "The native case must have a closed owner")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    terminal = case / "RESULT.private.json"
    result = read(terminal)
    require(spec["interface_comparison_link"] == "L1" and spec["task_instruction_id"].startswith("PLACE-")
            and result["shutdown_clean"] and not result["technical_failure"], "Expected an accounted native L1 PLACE")
    bound = dict(case=case.name, geometry=spec["frozen_files"][spec["model_path"]], seed=spec["comparison_policy_seed"],
                 task=spec["task_instruction_id"], terminal=reference(terminal), safety_stopped=result["unsafe"])
    score = (result["detail"] or {}).get("score")
    if score is None:
        progress = case / "native-01/MULTITASK_SCORE_PROGRESS.private.json"
        score = read(progress)
        require(result["unsafe"] and 0 < score["native_step"] <= result["actual_native_integrations"],
                "Partial score requires an accounted safety terminal")
        bound["score_progress"] = reference(progress)
        bound["scored_through_tick"] = score["native_step"]
    events = score.get("events", {})
    if "correct_region" not in events:
        return dict(bound, entered=False, events=events)
    entered = events["correct_region"]["native_step"]
    queries = []
    for returned in sorted((case / "native-01/g7-queries").glob("*/MODEL_RETURN.private.json")):
        query = returned.parent
        request = read(query / "MODEL_REQUEST.private.json")
        if request["native_step"] < entered:
            continue
        response = read(returned)["response"]
        record = query / "QUERY_ACTUAL.private.json"
        actual = read(record) if record.exists() else None
        action = response["action"]
        if actual is not None:
            require(actual["native_start_step"] == request["native_step"] and actual["action"] == action
                    and actual["actual_model_generations"] == 1 and not actual["mock_consumption"]
                    and len(actual["confirmed_intervals"]) == actual["confirmed_atoms"],
                    "Archived raw action and actual consumed intervals differ")
        with np.load(pin(action["path"], action["sha256"]), allow_pickle=False) as data:
            raw = data["physical_H40"]
        require(raw.shape == (40, 10) and np.isfinite(raw).all(), "Malformed archived H40")
        widths = 2. + 40. * raw[:, 9]
        atoms = []
        for interval in actual["confirmed_intervals"] if actual else ():
            start, end = (interval[k] for k in ("native_start_step", "native_end_step"))
            knot = interval["accepted_boundary"]
            require(end - start == 25 and start >= request["native_step"] and knot["native_step"] == end
                    and knot["exact_receiver_knot"] and knot["accepted_wire_sha256"] == interval["wire_sha256"],
                    "The physical width must be joined to a confirmed 50 ms atom")
            requested, measured = (1000 * interval[k] for k in ("emitted_gripper_width_m", "measured_gripper_width_m"))
            require(np.isfinite([requested, measured]).all(), "Physical jaw width is missing")
            atoms.append(dict(start_tick=start, emitted_mm=requested, measured_mm=measured))
        queries.append(dict(query=int(query.name), start_tick=request["native_step"],
                            raw_p10_max_mm=float(widths[:10].max()), raw_h40_max_mm=float(widths.max()),
                            actual_recorded=actual is not None, atoms=atoms))
    atoms = [row for q in queries for row in q["atoms"]]
    require(all(b["start_tick"] > a["start_tick"] for a, b in zip(atoms, atoms[1:])),
            "The same jaw consumption tick appears more than once")
    largest = {
        "raw_p10_mm": max((q["raw_p10_max_mm"] for q in queries), default=None),
        "raw_h40_mm": max((q["raw_h40_max_mm"] for q in queries), default=None),
        "emitted_mm": max((a["emitted_mm"] for a in atoms), default=None),
        "measured_mm": max((a["measured_mm"] for a in atoms), default=None),
    }
    return dict(bound, entered=True, first_region_tick=entered, events=events, post_entry_raw_queries=len(queries),
                queries_without_consumption_record=sum(not q["actual_recorded"] for q in queries),
                confirmed_atoms=len(atoms), maximum_widths=largest,
                raw_p10_open_candidates=sum(q["raw_p10_max_mm"] >= OPEN_DIAGNOSTIC_MM for q in queries),
                emitted_open_atoms=sum(a["emitted_mm"] >= OPEN_DIAGNOSTIC_MM for a in atoms),
                last_score={k: score[k] for k in ("task_success", "target_grasp", "released", "maximum_lift_m",
                    "longest_stable_released_retreat_s")}, queries=queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", type=Path, nargs="+")
    args = parser.parse_args()
    require(not args.output.exists(), "Keep the first archived jaw readout")
    records = [after_entry(case) for case in args.cases]
    require(len({(r["geometry"], r["seed"], r["task"]) for r in records}) == 1,
            "Only the same PLACE cell belongs in one paired report")
    write(args.output, dict(open_diagnostic_mm=OPEN_DIAGNOSTIC_MM, records=records, native_ticks=0, model_forwards=0))
