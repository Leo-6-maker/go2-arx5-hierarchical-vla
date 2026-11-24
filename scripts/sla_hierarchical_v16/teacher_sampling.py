"""Build a TRAIN-only sampler and independently re-read chronological command and task events."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from demonstrated_grip import events, pin, read, require, rows
from visual_lift_panel import write_new


def build(root, qualification):
    ep = read(qualification)["episodes"]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    receipts = {}
    for name, v in ep.items():
        if v["status"] == "excluded_original_failure":
            continue
        require(
            v["status"] in ("verified_seed", "verified_expansion")
            and "-TRAIN-" in name,
            "unverified group",
        )
        folder = (
            root
            / (
                "v16/target-labels04"
                if v["status"] == "verified_seed"
                else "v17/train-attempt01"
            )
            / name
        )
        report = v["prepared"]
        evidence = rows(
            pin(
                folder / "native_audit.jsonl", report["artifacts"]["native_audit.jsonl"]
            )
        )
        with np.load(
            pin(folder / "task_audit.npz", report["artifacts"]["task_audit.npz"])
        ) as f:
            nominal, installed, mask = (
                f[k]
                for k in (
                    "nominal_world_H40",
                    "installed_reference_H40",
                    "confirmed_native_interval_mask",
                )
            )
        n = len(evidence)
        require(n == report["official_rows"] and mask.shape == (n, 40), "example count")
        consumed_ticks, commanded, previous = [], [], None
        for i, row in enumerate(evidence):
            groups[v["task"]][row["phase"]][name].append(i)
            before = row["previous_command"]["width_m"] / 0.044
            if previous is not None:
                require(abs(before - previous) < 1e-8, "discontinuous consumed command")
            else:
                consumed_ticks.append(row["tick"] - 25)
                commanded.append(before)
            atoms = row["confirmed_native"]
            indices = [a["index"] for a in atoms]
            require(
                indices == list(range(10))
                and indices == np.flatnonzero(mask[i]).tolist(),
                "incomplete consumed P10",
            )
            for a in atoms:
                j = a["index"]
                width = a["installed_EEF10"][-1]
                require(
                    a["start"] == consumed_ticks[-1] + 25
                    and a["end"] == a["start"] + 25
                    and width == installed[i, j, -1],
                    "wire or clock disagrees",
                )
                consumed_ticks.append(a["start"])
                commanded.append(width)
            previous = commanded[-1]
        ev = events(consumed_ticks, commanded, step=True)
        ticks = v["native_event_ticks"]
        require(
            ticks["target_grasp"] <= ticks["target_lift20mm"]
            and (
                not v["task"].startswith("PLACE")
                or ticks["target_lift20mm"] < ticks["released"]
            ),
            "native task event order",
        )
        receipts[name] = {
            "consumed_command_threshold_crossings": [
                [int(e["s"]), e["kind"]] for e in ev
            ],
            "target_grasp_tick": ticks["target_grasp"],
            "lift20mm_tick": ticks["target_lift20mm"],
            "released_tick": ticks.get("released"),
            "task": v["task"],
            "layout": v["layout"],
            "nominal_full_open_points": int(np.sum(nominal[:, :, 9] == 1)),
        }
    examples = []
    phase_counts = Counter()
    for task, phases in sorted(groups.items()):
        for phase, episodes in sorted(phases.items()):
            for episode, indices in sorted(episodes.items()):
                weight = 1 / (len(groups) * len(phases) * len(episodes) * len(indices))
                examples.extend(
                    {"episode": episode, "query": i, "probability": weight}
                    for i in indices
                )
                phase_counts[phase] += len(indices)
    keys = {(r["episode"], r["query"]) for r in examples}
    require(
        len(keys)
        == len(examples)
        == sum(v["prepared"]["official_rows"] for v in ep.values() if "prepared" in v)
        and abs(sum(r["probability"] for r in examples) - 1) < 1e-10,
        "sampler denominator",
    )
    return {
        "design": "equal task, then equal recorded teacher phase within task, then episode within phase, then query; offline sample weights only",
        "episodes": receipts,
        "phase_query_counts": dict(phase_counts),
        "samples": examples,
        "training_draws": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("qualification", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    write_new(args.out, build(args.root, args.qualification))
