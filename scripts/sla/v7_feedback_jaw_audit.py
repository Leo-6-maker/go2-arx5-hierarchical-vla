"""Separate seam and within-chunk G7 effects on the fixed V7 feedback pair."""

import argparse
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.dense_g7_v6 import DT, GRIP_LIMIT, label_margin_report, previous_command_context
from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M, WORKING_CLOSED_M, WORKING_OPEN_M
from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v7_feedback_eval import FEEDBACK_ROSTER


ROLES = ("CONTROL_NEUTRAL", "TREATMENT_PREVIOUS_EXECUTED_GRIPPER_COMMAND")
RECEIPTS = ("33b302f2cce088ba60f58e924013426aaf7e14abb73d362b4b8253f057c1a9f9",
            "8cf4162b6cbac27a495b364120e141df0bed8155b59994bc964bc3222a38fb13")


def member(record):
    return pin(record["path"], record["sha256"])


def jaw_motion(row, reference):
    with np.load(member(row["action"]), allow_pickle=False) as action:
        raw = action["physical_H40"]
    original = label_margin_report(raw, reference)
    require(original == row["metrics"]["jaw"], "Original G7 replay differs")
    previous = previous_command_context(reference)
    width = WORKING_CLOSED_M + (WORKING_OPEN_M - WORKING_CLOSED_M) * raw[:, 9]
    speed = np.abs(np.diff(np.r_[previous, width])) / DT
    require(abs(speed[0] - original["first_seam_speed_m_s"]) < 1e-12
            and abs(speed.max() - original["maximum_speed_m_s"]) < 1e-12, "Working-range codec drift")
    rejected = speed > GRIP_LIMIT + 1e-10
    return dict(raw_first_width_mm=1000 * float(width[0]), first_seam_mm_s=1000 * float(speed[0]),
                peak_inside_mm_s=1000 * float(speed[1:].max()), first_seam_rejected=bool(rejected[0]),
                interior_rejected=bool(rejected[1:].any()), interior_rejected_atoms=(np.flatnonzero(rejected[1:]) + 2).tolist(),
                original_H40_pass=row["metrics"]["original_envelope"]["original_full_H40_pass"])


def audit(root):
    first, second = [read(pin(root / f"{role}-FEEDBACK/RESULT.json", digest)) for role, digest in zip(ROLES, RECEIPTS, strict=True)]
    require(all(run["model_forwards"] == len(run["rows"]) == 24 and run["step"] == 500
                and run["physics_ticks"] == run["training_updates"] == 0 and run["roster"]["sha256"] == FEEDBACK_ROSTER
                for run in (first, second)), "Expected the original 24 paired feedback windows")
    rows = []
    for left, right in zip(first["rows"], second["rows"], strict=True):
        require(all(left[k] == right[k] for k in ("id", "phase", "tick", "input", "request", "noise_sha256", "actual_sampler_sha256"))
                and left["noise_sha256"] == left["actual_sampler_sha256"], "Changed fixed state or CUDA noise")
        reference = read(member(left["request"]))["previous_reference"]
        previous = previous_command_context(reference)
        with np.load(member(left["input"]), allow_pickle=False) as observation:
            require(float(observation["previous_width_m"]) == previous, "Wrong original actual previous command")
            measured = MECHANICAL_TRAVEL_M * float(observation["state_eef10"][9])
        rows.append(dict(id=left["id"], phase=left["phase"], previous_mm=1000 * previous, measured_mm=1000 * measured,
                         measured_minus_previous_mm=1000 * (measured - previous),
                         input_sha256=left["input"]["sha256"], original_noise_sha256=left["noise_sha256"],
                         control=jaw_motion(left, reference), treatment=jaw_motion(right, reference)))
    gap = 1000 * GRIP_LIMIT * DT
    summary = {}
    for key, subset in (("all", rows), ("gap_over_seam_range", [r for r in rows if abs(r["measured_minus_previous_mm"]) > gap])):
        summary[key] = dict(windows=len(subset), arms={name: dict(first_seam_rejections=sum(r[name]["first_seam_rejected"] for r in subset),
                                        interior_rejections=sum(r[name]["interior_rejected"] for r in subset))
                                      for name in ("control", "treatment")},
                            improved_seams=sum(r["treatment"]["first_seam_mm_s"] < r["control"]["first_seam_mm_s"] for r in subset),
                            median_treatment_minus_control_mm_s=float(np.median([r["treatment"]["first_seam_mm_s"] - r["control"]["first_seam_mm_s"]
                                                                                 for r in subset])) if subset else None)
    return dict(feedback_input_sha256=list(RECEIPTS), original_roster_sha256=FEEDBACK_ROSTER, first_seam_range_mm=gap,
                fixed_observation_diagnostic_only=True, summary=summary, windows=rows, new_model_forwards=0,
                new_native_integrations=0, new_optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, audit(args.root))
