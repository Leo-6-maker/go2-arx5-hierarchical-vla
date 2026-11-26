"""Check a coordinate-only repair on existing model outputs and real paused feedback."""

import argparse
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from v4_long_watch import pinned, score
from v6_reference_anchors import metric, transport
from v6_retention_compare import aggregate, key


def run(root, output):
    baseline_file = root / "STEP0000_RETENTION/RESULT.json"
    original_file = root / "STEP1000_RETENTION/RESULT.json"
    baseline, original = read(baseline_file), read(original_file)
    require(len(baseline["rows"]) == len(original["rows"]) == 24
            and baseline["roster"]["sha256"] == original["roster"]["sha256"],
            "Match all sealed feedback windows and their original model outputs")
    fixed, roundtrip = [], []
    for row in original["rows"]:
        request = read(pinned(row["request"]))
        with np.load(pinned(row["input"]), allow_pickle=False) as packet, np.load(
                pinned(row["action"]), allow_pickle=False) as contains:
            physical = contains["physical_H40"]
            policy = packet["policy_from_world"]
            actual = np.linalg.inv(policy) @ matrix_from10(packet["state_eef10"][None])[0]
            accepted = packet["accepted_reference_TCP"]
            emitted = np.linalg.inv(policy) @ matrix_from10(physical)
            candidate = pose10(policy @ transport(emitted, accepted, actual), physical[:, 9])
            transported = transport(np.linalg.inv(policy) @ matrix_from10(candidate), actual, accepted)
            roundtrip.append(metric(transported, emitted).max(axis=0))
            corrected = score(candidate, packet, request["previous_reference"])
            require(corrected["jaw"] == row["metrics"]["jaw"], "This pose transform must preserve the real jaw gate")
        fixed.append(dict(row, metrics=corrected))
    anchor = {key(row): row for row in baseline["rows"]}
    phases = sorted({row["phase"] for row in fixed})
    with output.open("x") as stream:
        json.dump(dict(roster_sha256=original["roster"]["sha256"], original_result_sha256=sha(original_file),
                       parent_result_sha256=sha(baseline_file), rows=len(fixed),
                       frozen_checkpoint=original["checkpoint"],
                       original=aggregate(original["rows"], anchor), corrected=aggregate(fixed, anchor),
                       corrected_by_phase={p: aggregate([r for r in fixed if r["phase"] == p], anchor) for p in phases},
                       max_roundtrip_mm_deg=np.asarray(roundtrip).max(axis=0).tolist(),
                       model_forwards=0, native_integrations=0, optimizer_updates=0),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
