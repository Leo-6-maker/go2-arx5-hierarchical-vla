"""Compare nominal 100 ms increments to the unchanged total-step limits of A1."""

import argparse
from pathlib import Path

import numpy as np
from demonstrated_grip import read
from known_dynamic_curve import trajectory
from known_dynamic_panel import RATES
from scipy.spatial.transform import Rotation
from visual_lift_panel import write_new


def demand(config):
    cfg = config["controller"]
    rows = {}
    for name, rate in RATES.items():
        poses = trajectory(np.eye(4), 1 + np.arange(41) * 0.1, *rate)
        mm = 1000 * np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        angles = np.degrees(
            Rotation.from_matrix(
                poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)
            ).magnitude()
        )
        rows[name] = {
            "movement_intervals": 40,
            "max_100ms_translation_mm": float(mm.max()),
            "max_100ms_rotation_deg": float(angles.max()),
            "count_exceeding_A1_translation_increment": int(
                np.sum(mm > cfg["max_step_mm"] + 1e-10)
            ),
            "count_exceeding_A1_rotation_increment": int(
                np.sum(angles > cfg["max_step_deg"] + 1e-10)
            ),
        }
    return {
        "A1_total_increment_limit_mm": cfg["max_step_mm"],
        "A1_total_increment_limit_deg": cfg["max_step_deg"],
        "clock_ms": cfg["cadence_ms"],
        "conditions": rows,
        "interpretation": "Exact consecutive nominal endpoints under the existing feedback-only total-increment cap. This is not a lower bound on tube error or an A1 measurement.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    write_new(a.out, demand(read(a.config)))
