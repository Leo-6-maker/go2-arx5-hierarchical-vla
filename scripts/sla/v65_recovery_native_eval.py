"""Read out the ten original 1K native cells and apply the registered release criteria."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v61_native_gate import stop_mechanism
from v62_source_native import FEEDBACK_ROSTER, PARENT_RELEASE, R4_PANELS, read_phase, reference
from v64_adapt_native_gate import place_stages
from v65_recovery_course import END, PLAN


PHASES = ("HOLD", "PLACE", "INDEPENDENT_HOLD")


def native_phase(root, phase, reservation, output):
    bound = read(reservation)
    expected = bound["groups"][phase]
    declaration = root / "COMPARISON.json"
    consent = read(declaration)
    require(bound["plan"] == consent["v62_plan"] == PLAN and consent["source_step"] == END
            and consent["source_audit"] == expected["source_audit"], "Native evaluation revision changed")
    result = read_phase(root, "HOLD" if phase == "INDEPENDENT_HOLD" else phase, END)
    require(result["checkpoint"] == bound["checkpoint"] and len(result["arms"]) == len(expected["cases"]),
            "Invalid checkpoint or native denominator")
    for row, cell in zip(result["arms"], expected["cases"], strict=True):
        case = root / row["id"]
        spec = read(case / "EXECUTION_SPEC.json")
        require(row["id"] == cell["id"] and spec["frozen_files"][spec["model_path"]] == cell["geometry"]
                and row["native_ticks"] <= cell["max_ticks"] and row["forwards"] <= cell["max_queries"],
                "Native geometry or persistent limits differ from admitted scope")
        row["prior_cell"] = cell["prior_cell"]
        row["control_witness"] = "SELECTED_TRACE_PRESENT_UNAUDITED" if (
            case / "native-01/SELECTED_CONTROLLER.private.jsonl").is_file() else "NATIVE_SELECTED_NOT_OBSERVED"
    write(output, dict(result, phase=phase, reservation=reference(reservation), comparison=reference(declaration)))


def paired_metrics(records, precedents):
    n02, place, independent = records
    hold_success = sum(row["valid_science"] and row["task_success"] for row in n02)
    independent_success = sum(row["valid_science"] and row["task_success"] for row in independent)
    baseline, current, pregrasp, hold_novel = Counter(), Counter(), Counter(), Counter()
    standard_stops = []
    for row in (*n02, *place):
        old = precedents[row["prior_cell"]]
        if row["unsafe"] or row.get("actual_stop"):
            standard_stops.append(row["id"])
        rejection = row.get("named_rejection")
        if rejection and not row["grasp"] and rejection != old.get("named_rejection"):
            pregrasp[rejection if isinstance(rejection, str) else rejection.get("reason", "UNCLASSIFIED")] += 1
    for row in place:
        baseline.update({k: int(v) for k,v in place_stages(precedents[row["prior_cell"]]).items()})
        if row["valid_science"]:
            current.update({k: int(v) for k,v in place_stages(row).items()})
    for row in independent:
        old = precedents[row["prior_cell"]]
        mechanism = stop_mechanism(row)
        if mechanism is not None and mechanism != stop_mechanism(old):
            hold_novel[repr(mechanism)] += 1
    improved = [k for k,n in baseline.items() if current[k] > n]
    retained = all(current[k] >= n for k,n in baseline.items())
    after_stop = sum(row.get("after_native_stop") or 0 for block in records for row in block)
    unverified_stop = [row["id"] for block in records for row in block
                       if (row["unsafe"] or row.get("actual_stop")) and row.get("after_native_stop") is None]
    valid = [sum(row["valid_science"] for row in block) for block in records]
    return dict(n02_success=hold_success, independent_hold_success=independent_success, valid=valid, stop_after_ticks=after_stop,
                unverified_stop_counts=unverified_stop,
                native_pass=valid == [2,4,4] and hold_success == 2 and independent_success >= 2
                and current["grasp"] == 4 and current["lift"] >= 3 and current["region"] >= 2 and retained and bool(improved)
                and not (after_stop or unverified_stop or standard_stops
                         or any(n >= 2 for n in (*pregrasp.values(), *hold_novel.values()))),
                place_stages={key:current[key] for key in baseline}, original_place_stages=dict(baseline),
                improved_stages=improved, standard_safety_stops=standard_stops, new_pregrasp_rejections=dict(pregrasp),
                new_independent_hold_failure_mechanisms=dict(hold_novel))


def settle(site, parent_release, reservation, reports, dev, feedback, output):
    parent = read(pin(parent_release, PARENT_RELEASE))
    require(parent["released"] and parent["independent_HOLD"]["r4_full_success"] == 2,
            "The original R4 retention floor changed")
    precedents = {}
    for phase in ("HOLD", "PLACE"):
        old = read(pin(site / f"R4-DEV-0200-{phase}-NATIVE-01/INDEPENDENT_PANEL_READBACK.json", R4_PANELS[phase]))
        precedents.update({r["id"].replace("R4_DEV_0200", "ADAPT0250"): r for r in old["arms"]})
    precedents.update({r["r4"]["id"]: r["r4"] for r in parent["independent_HOLD"]["pairs"]})
    bound = read(reservation)
    panels = [read(path) for path in reports]
    require(len(panels) == len(PHASES) and [p["phase"] for p in panels] == list(PHASES)
            and all(p["reservation"] == reference(reservation) and p["checkpoint"] == bound["checkpoint"] for p in panels)
            and {row["prior_cell"] for pnl in panels for row in pnl["arms"]} == set(precedents),
            "Independent scores are incomplete or not paired to the exact old population")
    offline, replay = read(dev), read(feedback)
    require(offline["plan"] == PLAN and offline["source_step"] == replay["step"] == END
            and offline["checkpoint"] == replay["checkpoint"] == bound["checkpoint"]
            and offline["audit"] == panels[0]["audit"] == panels[1]["audit"] == panels[2]["audit"]
            and replay["roster"]["sha256"] == FEEDBACK_ROSTER
            and offline["model_forwards"] == 42 and replay["model_forwards"] == len(replay["rows"]) == 24
            and {k:len(v["rows"]) for k,v in offline["domains"].items()} == {"robodojo":24, "single":18}
            and all(r["actual_sampler_sha256"] == r["noise_sha256"] for r in replay["rows"])
            and offline["physics_ticks"] == replay["physics_ticks"] == offline["optimizer_updates"] == replay["training_updates"] == 0,
            "The registered zero-integration offline cohort is incomplete")
    metrics = paired_metrics([p["arms"] for p in panels], precedents)
    alerts = offline["retention_alert_domains"]
    write(output, dict(plan=PLAN, step=END, checkpoint=bound["checkpoint"], source_drift_alert_domains=alerts,
                       scientific_release=metrics["native_pass"] and not alerts, automatic_training_submission=False,
                       measurements=metrics, native_groups=[reference(p) for p in reports],
                       feedback=reference(feedback), source_dev=reference(dev)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="mode", required=True)
    group = mode.add_parser("native-phase")
    group.add_argument("phase", choices=PHASES)
    settled = mode.add_parser("settle")
    for parser, names in ((group, ("root", "reservation", "output")),
                          (settled, ("site", "parent_release", "reservation", "dev", "feedback", "output"))):
        for name in names:
            parser.add_argument(name, type=Path)
    settled.add_argument("reports", type=Path, nargs=3)
    args = vars(cli.parse_args())
    action = args.pop("mode")
    require(not args["output"].exists(), "Preserve existing evidence")
    {"native-phase": native_phase, "settle": settle}[action](**args)
