"""Verify previous consumed commands at all V7 Recovery3 training queries."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v65_recovery_data import ROSTER, checked


RECOVERY_MANIFEST = "362c8756c7b7ea62a539a441dcfa5a6bdf175f4eb92f79088ed7a7ec7a884991"


def audit(site, portable):
    manifest = read(pin(portable / "MANIFEST.json", RECOVERY_MANIFEST))
    catalog = {episode["id"]: episode for episode in manifest["episodes"]}
    result = []
    for wave, (suffix, windows, identity) in enumerate(ROSTER, 1):
        name = f"V64_RECOVERY_TRAIN_{suffix}"
        episode = catalog[name]
        raw_path = site / f"RECOVERY-TRAIN-SEALED-{wave:02d}" / name / "ACTUAL.json"
        raw = read(pin(raw_path, identity))
        terminal = checked(raw["native_result"])
        root = terminal.parent
        lease = read(root / "control/LEASE.json")
        spec = read(pin(root / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        require(lease["status"] == "CLOSED" and spec["collection_split"] == "TRAIN"
                and episode["split"] == spec["evaluation_split"] == "TRAIN_RECOVERY",
                "Only independently sealed TRAIN recovery is eligible")
        capture = read(checked(raw["capture"]))
        queries = read(checked(capture["labels"]))
        with checked(raw["actual_native_samples"]).open() as stream:
            past = {row["native_end_step"]: row for row in map(json.loads, stream)}
        field = episode["gripper"]
        with np.load(pin(portable / field["file"], field["sha256"]), allow_pickle=False) as vals:
            expected = vals["prior"]
        native = root / "native-01"
        handoff_path = native / "RECOVERY-HANDOFF-INTENT.private.json"
        handoff = read(handoff_path)
        step = queries[0]["native_step"]
        last_query = handoff["actual_student_queries"] - 1
        last_path = native / "g7-queries" / f"{last_query:04d}" / "QUERY_ACTUAL.private.json"
        student = read(last_path)
        last = student["confirmed_intervals"][-1]
        require(handoff["native_step"] == step and handoff["prior_role"] == "STUDENT" and handoff["next_role"] == "TEACHER"
                and handoff["clock_owner_unchanged"] and student["query"] == last_query and student["full_P10"]
                and student["confirmed_atoms"] == len(student["confirmed_intervals"]) == 10
                and last["native_end_step"] == step and last["accepted_boundary"]["exact_receiver_knot"]
                and last["wire_sha256"] == last["accepted_boundary"]["accepted_wire_sha256"]
                == past[step]["installed_reference_sha256"]
                and abs(last["emitted_gripper_width_m"] - queries[0]["previous_confirmed_command"]["width_m"]) < 1e-12,
                "The first teacher context does not match the last student command actually consumed")
        rows = []
        require(len(queries) == windows == len(expected), "Fixed Recovery window population changed")
        for query, value in zip(queries, expected, strict=True):
            tick = query["native_step"]
            source_tick = tick - 25
            row = past[tick]
            command = query["previous_confirmed_command"]
            width = command["width_m"]
            source = command["source_native_s"]
            now = command["query_native_s"]
            require(query["supervision_role"] == "RECOVERY_TEACHER"
                    and row["native_start_step"] == source_tick and row["ticks"] == 25 and row["safety_valid"]
                    and np.isfinite(width) and .002 <= width <= .042 and math.isclose(width, value, abs_tol=1e-12, rel_tol=0)
                    and abs(width - row["installed_EEF10"][9] * .044) < 1e-12
                    and abs(now - tick * .002) < 1e-8 and abs(source - source_tick * .002) < 1e-8
                    and source == row["gripper_command"]["source_native_s"],
                    "Actual previous command, phase or 50 ms clock differs")
            rows.append(dict(query=query["query"], query_tick=tick, source_tick=source_tick,
                             previous_consumed_width_m=width, valid=True))
        result.append(dict(id=name, source_receipt_sha256=identity, student_handoff_sha256=sha(handoff_path),
                           preceding_student_query_sha256=sha(last_path), rows=rows))
    count = sum(len(episode["rows"]) for episode in result)
    require(count == 115, "The complete fixed Recovery3 population must reconcile")
    return dict(recovery_manifest_sha256=RECOVERY_MANIFEST, episodes=result, valid_windows=count,
                verified_student_teacher_boundaries=len(result), model_forwards=0, optimizer_updates=0,
                physics_ticks=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "portable", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    require(not args.output.exists(), "Preserve the original V7 context audit")
    write(args.output, audit(args.site, args.portable))
