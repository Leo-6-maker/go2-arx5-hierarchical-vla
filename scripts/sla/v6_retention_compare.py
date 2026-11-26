"""Compare fixed-input endpoint predictions on Corelab without native execution."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


STEPS = (0, 100, 250, 500, 1000)


def read(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def key(row):
    return row["id"], row["query"]


def feedback(row):
    return (row["phase"], row["tick"], row["input"]["sha256"],
            row["request"]["sha256"], row["noise_sha256"])


def measured(row):
    metrics = row["metrics"]
    return dict(p10_z_mm=metrics["p10_displacement_world_mm"][2],
                h40_z_mm=metrics["h40_displacement_world_mm"][2],
                p10_min_width_mm=metrics["p10_tightest_width_mm"],
                h40_min_width_mm=metrics["h40_tightest_width_mm"])


def aggregate(rows, parent):
    output = {}
    for name in measured(rows[0]):
        values = [measured(row)[name] for row in rows]
        paired = [value - measured(parent[key(row)])[name]
                  for row, value in zip(rows, values, strict=True)]
        output[name] = dict(median=statistics.median(values), delta_median=statistics.median(paired),
                            delta_min=min(paired), delta_max=max(paired))
    channels = [row["metrics"] for row in rows]
    return dict(windows=len(rows), metrics=output,
                original_h40_pass=sum(m["original_envelope"]["original_full_H40_pass"] for m in channels),
                original_jaw_speed_reject=sum(m["jaw"]["original_execution_speed_exceeded"] for m in channels))


def compare(root, steps):
    roster, roster_sha = read(root / "ROSTER24.json")
    expected = {key(row): row for row in roster["rows"]}
    if len(expected) != 24 or len(roster["rows"]) != 24 or 0 not in steps:
        raise ValueError("Expected 24 distinct registered windows and their common parent")
    receipts, results, parent = {}, {}, {}
    for step in steps:
        path = root / f"STEP{step:04d}_RETENTION" / "RESULT.json"
        result, digest = read(path)
        rows = result["rows"]
        if (result["step"] != step or result["roster"]["sha256"] != roster_sha
                or result["physics_ticks"] or result["training_updates"] or result["closed_loop"]
                or len(rows) != 24 or {key(row) for row in rows} != set(expected)
                or result["model_forwards"] + result["reused_checked_forwards"] != len(rows)):
            raise ValueError(f"Unmatched or incomplete checkpoint result at step {step}")
        for row in rows:
            if (feedback(row) != feedback(expected[key(row)])
                    or row["actual_sampler_sha256"] != row["noise_sha256"]):
                raise ValueError(f"Input identity changed at step {step}, window {key(row)}")
        if step == 0:
            parent = {key(row): row for row in rows}
        phases = defaultdict(list)
        for row in rows:
            phases[row["phase"]].append(row)
        results[step] = dict(all=aggregate(rows, parent),
                             phases={phase: aggregate(group, parent) for phase, group in sorted(phases.items())})
        receipts[step] = dict(sha256=digest, checkpoint=result["checkpoint"],
                              new_forwards=result["model_forwards"], reused=result["reused_checked_forwards"])
    return dict(roster_sha256=roster_sha, receipts=receipts, results=results,
                model_forwards=sum(r["new_forwards"] for r in receipts.values()),
                reused_forwards=sum(r["reused"] for r in receipts.values()),
                physics_ticks=0, training_updates=0, closed_loop=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, choices=STEPS, default=STEPS)
    args = parser.parse_args()
    result = compare(args.root, sorted(set(args.steps)))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
