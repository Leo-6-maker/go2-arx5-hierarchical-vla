"""Settle the fixed A250 panel against its R4 parent and offline receipts."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v62_source_native import (
    FEEDBACK_ROSTER, PARENT_RELEASE, R4_PANELS, read_phase, reference, source_role,
)
from v64_adapt_endpoint import BANK, END, PLAN, R4


def place_stages(row):
    events, obj = row["task_events"], row["terminal_object"]
    return dict(grasp=row["grasp"], lift=row["lift20"], region="correct_region" in events,
                released=obj["released"], stable_release=obj["stable_release_condition"],
                withdrawn=obj["longest_stable_released_retreat_s"] > 0, complete=row["task_success"])


def settle(base, parent_release, hold, place, dev, feedback, output):
    parent = read(pin(parent_release, PARENT_RELEASE))
    require(parent["released"] and parent["r4_receipt_sha256"] == R4, "R4 parent has no released evaluation")
    role = source_role(END, PLAN)
    panels = {"HOLD": read_phase(hold, "HOLD", END), "PLACE": read_phase(place, "PLACE", END)}
    h, p = panels.values()
    audit = read(pin(h["audit"]["path"], h["audit"]["sha256"]))
    require(h["audit"] == p["audit"] and h["checkpoint"] == p["checkpoint"]
            and audit["science_plan"] == PLAN and audit["bank_receipt_sha256"] == BANK
            and audit["complete_training_endpoint"] and audit["rng_restored"],
            "A250 panels used different weights or an unverified training endpoint")
    candidate_counts, parent_counts, stops, rejections = Counter(), Counter(), [], Counter()
    for phase, panel in panels.items():
        previous = base / f"R4-DEV-0200-{phase}-NATIVE-01/INDEPENDENT_PANEL_READBACK.json"
        paired = {row["id"].removeprefix("R4_DEV_0200_"): row for row in read(pin(previous, R4_PANELS[phase]))["arms"]}
        require(len(paired) == (2 if phase == "HOLD" else 4)
                and set(paired) == {row["id"].removeprefix(role + "_") for row in panel["arms"]}
                and all(row["id"].startswith(role + "_") for row in panel["arms"]),
                "A250 does not cover the exact six independent paired conditions")
        for row in panel["arms"]:
            old = paired[row["id"].removeprefix(role + "_")]
            require(row["policy_seed"] == old["policy_seed"], "The matched policy seed changed")
            if row["unsafe"] or row.get("actual_stop") or row.get("after_native_stop", 0):
                stops.append(row["id"])
            rejection = row.get("named_rejection")
            if rejection and not row["grasp"] and rejection != old.get("named_rejection"):
                key = rejection if isinstance(rejection, str) else rejection.get("reason", "UNCLASSIFIED")
                rejections[key] += 1
            if phase == "PLACE":
                parent_counts.update({k: int(v) for k, v in place_stages(old).items()})
                if row["valid_science"]:
                    candidate_counts.update({k: int(v) for k, v in place_stages(row).items()})
    closed_hold = sum(row["valid_science"] and row["task_success"] for row in h["arms"])
    stages_retained = all(candidate_counts[k] >= n for k, n in parent_counts.items())
    stages_improved = [k for k, n in candidate_counts.items() if n > parent_counts[k]]
    repeated = any(n >= 2 for n in rejections.values())
    native_pass = (h["valid"] == closed_hold == 2 and p["valid"] == candidate_counts["grasp"] == 4
                   and candidate_counts["lift"] >= 3 and candidate_counts["region"] >= 2
                   and stages_retained and bool(stages_improved) and not stops and not repeated)
    offline, replay = read(dev), read(feedback)
    require(offline["plan"] == PLAN and offline["source_step"] == replay["step"] == END
            and offline["checkpoint"] == replay["checkpoint"] == h["checkpoint"]
            and offline["audit"] == h["audit"] and offline["model_forwards"] == 42
            and sorted(d["summary"]["windows"] for d in offline["domains"].values()) == [18, 24]
            and replay["roster"]["sha256"] == FEEDBACK_ROSTER
            and replay["model_forwards"] == len(replay["rows"]) == 24
            and all(row["noise_sha256"] == row["actual_sampler_sha256"] for row in replay["rows"])
            and offline["physics_ticks"] == offline["optimizer_updates"] == replay["physics_ticks"]
            == replay["training_updates"] == 0, "Incomplete matched offline evidence")
    alerts = offline["retention_alert_domains"]
    write(output, dict(plan=PLAN, completed_step=END, bank_sha256=BANK, checkpoint=h["checkpoint"],
                       native_pass=native_pass, hold_valid=h["valid"], hold_success=closed_hold,
                       place_valid=p["valid"], original_place_stages=dict(parent_counts),
                       actual_place_stages=dict(candidate_counts), improved_stages=stages_improved,
                       safety_stop_cases=stops, pregrasp_rejections=dict(rejections),
                       source_drift_alert_domains=alerts, released=bool(native_pass and not alerts),
                       panels=panels, dev=reference(dev), feedback=reference(feedback),
                       training=audit["training_terminal_sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "parent_release", "hold", "place", "dev", "feedback", "output"):
        parser.add_argument(name, type=Path)
    settle(**vars(parser.parse_args()))
