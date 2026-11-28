"""Apply the original native scoring to V7-A and retain all paired outcomes."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_native_binding import ARMS, BOUND, PHASES, PLAN, ROLES, STEPS, check_preflight


def phase(site, panel, native, output, phase_name, arm):
    from v62_source_native import read_phase

    check_preflight(site, panel, native, phase_name, arm)
    result = read_phase(native, "HOLD" if phase_name == "INDEPENDENT_HOLD" else phase_name, STEPS)
    binding = read(panel)
    require(result["checkpoint"] == binding["checkpoint_sha256"] and result["audit"] == binding["source_audit"],
            "Scored native output used a different endpoint")
    bound = read(pin(site / "v65-3k-native-code-02/BOUND-NATIVE-ENTRY.json", BOUND))["groups"][phase_name]
    for row, case in zip(result["arms"], bound["cases"], strict=True):
        require(row["id"] == case["id"].replace("RECOVERY3000", ROLES[arm]), "Scored case differs from fixed roster")
        row["prior_cell"] = case["prior_cell"]
    write(output, dict(result, plan=PLAN, arm=arm, phase=phase_name, comparison_sha256=sha(native / "COMPARISON.json")))


def paired_outcomes(groups, old):
    result = {}
    for name, rows in zip(PHASES, groups, strict=True):
        field = "grasp" if name == "PLACE" else "task_success"
        cohorts = {"kept": [], "gained": [], "lost": [], "both_failed": [], "technical": []}
        for row in rows:
            case = row["prior_cell"]
            if not row["valid_science"] or not old[case].get("valid_science", True):
                cohorts["technical"].append(case)
                continue
            before, after = bool(old[case][field]), bool(row[field])
            cohort = {(True, True): "kept", (False, True): "gained", (True, False): "lost",
                      (False, False): "both_failed"}[before, after]
            cohorts[cohort].append(case)
        result[name] = cohorts
    return result


def settle(site, evaluation, output):
    from v62_source_native import PARENT_RELEASE, R4_PANELS
    from v65_recovery_native_eval import paired_metrics

    parent = read(pin(site / "PARENT_RELEASE.json", PARENT_RELEASE))
    require(parent["released"] and parent["independent_HOLD"]["r4_full_success"] == 2, "Original R4 floor differs")
    r4 = {}
    for name in PHASES[:2]:
        file = site.parent / f"R4-DEV-0200-{name}-NATIVE-01/INDEPENDENT_PANEL_READBACK.json"
        rows = read(pin(file, R4_PANELS[name]))["arms"]
        r4.update({r["id"].replace("R4_DEV_0200", "ADAPT0250"): r for r in rows})
    r4.update({row["r4"]["id"]: row["r4"] for row in parent["independent_HOLD"]["pairs"]})
    measurements, states = {}, {}
    for i, arm in enumerate(ARMS):
        groups = [read(evaluation / f"SCORE-{i}-{name}.json") for name in PHASES]
        require(all(group["plan"] == PLAN and group["phase"] == name and group["arm"] == arm
                    for name, group in zip(PHASES, groups, strict=True)), "Unexpected phase report")
        require([len(group["arms"]) for group in groups] == [2, 4, 4], "Incomplete native denominator")
        rows = [group["arms"] for group in groups]
        require({row["prior_cell"] for group in rows for row in group} == set(r4), "Changed cohort")
        source = read(evaluation / f"{arm}-SOURCE/RESULT.json")
        require(source["arm"] == arm and source["plan"] == PLAN and source["model_forwards"] == 42
                and all(group["checkpoint"] == source["checkpoint"] for group in groups), "Changed source and native endpoint")
        native = paired_metrics(rows, r4)
        stops = [row["id"] for group in rows for row in group if row.get("actual_stop") or row["unsafe"]]
        g7 = [row["id"] for group in rows for row in group if "G7" in str(row.get("named_rejection")).upper()]
        measurements[arm] = dict(native=native, by_case_vs_r4=paired_outcomes(rows, r4), actual_stop_episodes=stops,
                                 g7_terminal_episodes=g7, query_returns=sum(row["forwards"] for group in rows for row in group),
                                 source_drift_alert_domains=source["retention_alert_domains"],
                                 passes_original_release=bool(native["native_pass"] and not source["retention_alert_domains"]))
        states[arm] = rows
    first, second = (states[arm] for arm in ARMS)
    control = {row["prior_cell"]: row for group in first for row in group}
    write(output, dict(plan=PLAN, arms=measurements, treatment_vs_control=paired_outcomes(second, control),
                       original_native_rows=states, historical_r4_by_case=r4, registered_per_arm=10))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    score = commands.add_parser("phase")
    for name in ("site", "panel", "native", "output"):
        score.add_argument(name, type=Path)
    score.add_argument("phase_name", choices=PHASES)
    score.add_argument("arm", choices=ARMS)
    collect = commands.add_parser("settle")
    for name in ("site", "evaluation", "output"):
        collect.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    {"phase": phase, "settle": settle}[args.pop("command")](**args)
