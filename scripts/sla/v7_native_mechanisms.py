"""Read stage-specific pair outcomes and episode-level mechanisms from closed V7-A scores."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v64_adapt_native_gate import place_stages
from v7_native_binding import ARMS, PHASES, PLAN


PLACE_STAGES = ("grasp", "lift", "region", "released", "stable_release", "withdrawn", "complete")


def rejection_reason(value):
    if isinstance(value, dict):
        return value.get("reason") or "UNCLASSIFIED_REJECTION"
    return value


def episode(row, phase):
    valid = row.get("valid_science", True)
    stop = row.get("actual_stop")
    rejection = rejection_reason(row.get("named_rejection"))
    if not valid:
        mechanism = "TECHNICAL"
    elif stop or row.get("unsafe"):
        reason = (stop or {}).get("reason") or "UNCLASSIFIED_SAFETY_STOP"
        period = "AFTER_FIRST_GRASP" if row.get("grasp") else "BEFORE_FIRST_GRASP"
        mechanism = f"{period}:{reason}"
    elif rejection:
        mechanism = rejection
    elif row["task_success"]:
        mechanism = "SUCCESS"
    else:
        mechanism = row.get("terminal_reason") or "END_WITHOUT_TASK_SUCCESS"
    events = row.get("task_events", {})
    stages = None
    if valid:
        stages = place_stages(row) if phase == "PLACE" else {"task_success": row["task_success"]}
    retimed_g7 = []
    for repair in row.get("retiming_consumption", ()):
        reason = rejection_reason(repair.get("original_timing_rejection"))
        if repair["confirmed_atoms"] > 0 and reason and "G7" in reason.upper():
            retimed_g7.append(repair["query"])
    return dict(case=row["prior_cell"], episode=row["id"], valid_science=valid, mechanism=mechanism,
                stages=stages, first_release_observed="released" in events,
                terminal_g7=bool(rejection and "G7" in rejection.upper()),
                confirmed_g7_retiming_queries=retimed_g7, actual_stop=stop,
                model_forwards=row.get("forwards"), native_ticks=row.get("native_ticks"),
                result_sha256=row.get("result_sha256"))


def stage_pairs(prior, candidate, phase):
    require(len(prior) == len(candidate) == len({row["case"] for row in candidate})
            and {row["case"] for row in prior} == {row["case"] for row in candidate}, "Unpaired native condition")
    earlier = {row["case"]: row for row in prior}
    categories = {(True, True): "kept", (False, True): "gained", (True, False): "lost", (False, False): "both_failed"}
    stages = PLACE_STAGES if phase == "PLACE" else ("task_success",)
    by_stage = {key: {category: [] for category in (*categories.values(), "technical")} for key in stages}
    for row in candidate:
        case, previous = row["case"], earlier[row["case"]]
        for key, groups in by_stage.items():
            if row["stages"] is None or previous["stages"] is None:
                category = "technical"
            else:
                category = categories[bool(previous["stages"][key]), bool(row["stages"][key])]
            groups[category].append(case)
    return by_stage


def readout(settlement, expected):
    settled = read(pin(settlement, expected))
    require(settled["plan"] == PLAN and settled["registered_per_arm"] == 10
            and set(settled["original_native_rows"]) == set(ARMS), "Requires the original closed V7-A pair")
    rows, parent = settled["original_native_rows"], settled["historical_r4_by_case"]
    require(all(len(rows[arm]) == len(PHASES) for arm in ARMS), "Changed original block roster")
    counts, pairs = {}, {}
    for idx, block in enumerate(PHASES):
        original = rows[ARMS[0]][idx]
        require(len(original) == len(rows[ARMS[1]][idx]) == (2, 4, 4)[idx], "Changed original block denominator")
        baseline = [episode(dict(parent[row["prior_cell"]], prior_cell=row["prior_cell"]), block) for row in original]
        panels = {arm: [episode(row, block) for row in rows[arm][idx]] for arm in ARMS}
        pairs[block] = {"treatment_vs_control": stage_pairs(panels[ARMS[0]], panels[ARMS[1]], block),
                        "each_vs_R4": {arm: stage_pairs(baseline, panel, block) for arm, panel in panels.items()}}
        counts[block] = {arm: dict(registered=len(panel), valid=sum(r["valid_science"] for r in panel),
                                  terminal_mechanism=dict(Counter(r["mechanism"] for r in panel)),
                                  terminal_g7_science_episodes=sum(r["valid_science"] and r["terminal_g7"] for r in panel),
                                  g7_retiming_science_episodes=sum(r["valid_science"] and bool(r["confirmed_g7_retiming_queries"]) for r in panel),
                                  episodes=panel) for arm, panel in panels.items()}
    return dict(plan=PLAN, settlement_sha256=expected, episodes=counts, stage_pairs=pairs,
                original_release={arm: settled["arms"][arm]["passes_original_release"] for arm in ARMS},
                new_native_integrations=0, new_model_forwards=0, new_optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settlement", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, readout(args.settlement, args.sha256))
