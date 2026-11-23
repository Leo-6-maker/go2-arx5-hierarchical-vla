"""Inspect both reset and WBC clocks in all sealed original F1/F2 starts."""

import hashlib
import json
import math
import sys
from pathlib import Path


def read(f):
    return json.loads(f.read_bytes())


def digest(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def first(f):
    with f.open() as h:
        return json.loads(next(h))


def main(root, repo):
    v0 = read(repo / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")
    records = []
    for phase in ("F1", "F2"):
        verified = root / f"feedback/{phase}_VERIFIED.json"
        if digest(verified) != v0["source_hashes"][phase]:
            raise ValueError("changed verified roster")
        stage = read(verified)
        campaign = next(
            (root / "feedback").glob("sla-feedback-" + phase.lower() + "-*")
        )
        for row in stage["records"]:
            case = campaign / row["card"]
            result = case / "RESULT.private.json"
            if digest(result) != row["result_sha256"]:
                raise ValueError("result seal")
            folder = case / read(result)["episode"]
            before, init, ctrl, samples = [
                folder / name
                for name in (
                    "BEFORE_FIRST_INTEGRATION.json",
                    "initialization.jsonl",
                    "control.jsonl",
                    "MOBILE_SAMPLES.private.jsonl",
                )
            ]
            reset = read(before)
            act = first(init)
            cmd = first(samples)
            with ctrl.open() as h:
                wbc = next(
                    x
                    for line in h
                    if (x := json.loads(line))["arm_mode"] == "WBC" and x["running"]
                )
            if (
                act["time"] != 0
                or reset["sim_time_s"] != 0
                or len(act["qpos"]) != 34
                or not all(math.isfinite(q) for q in act["qpos"])
            ):
                raise ValueError("malformed physical reset")
            command = cmd["gripper_command"]
            records.append(
                dict(
                    card=row["card"],
                    historical_outcome=row["outcome"],
                    reset_arm_mode=reset["arm_mode"],
                    reset_running=reset["running"],
                    first_wbc_s=wbc["sim_time_s"],
                    reset_slide_positions_m=act["qpos"][25:27],
                    first_recorded_interval_start=cmd["native_start_step"],
                    first_recorded_gripper_seq=command["sequence"],
                    first_recorded_gripper_source_s=command["source_native_s"],
                    source_sha256={
                        p.name: digest(p) for p in (before, init, ctrl, samples)
                    },
                )
            )
    if len(records) != 18:
        raise ValueError("roster incomplete")
    return dict(
        records=records,
        actual_native_starts_in_this_audit=0,
        pure_legacy_tick_zero_wbc=sum(
            r["reset_arm_mode"] == "WBC" and r["reset_running"] for r in records
        ),
    )


if __name__ == "__main__":
    print(
        json.dumps(
            main(Path(sys.argv[1]), Path(__file__).resolve().parents[2]),
            indent=2,
            allow_nan=False,
        )
    )
