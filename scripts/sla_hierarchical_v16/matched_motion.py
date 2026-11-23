"""Read each registered A0/A1 pair at identical task times from the frozen native archive."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def load_checked(f, h):
    if sha(f) != h:
        raise ValueError("archive digest mismatch: " + f.name)
    return json.loads(f.read_bytes())


def score(track):
    t = track["elapsed_ms"]
    grid = np.arange(len(t)) * 2
    if not np.allclose(t, grid, atol=1e-7, rtol=0):
        raise ValueError("2 ms native time grid")
    t = grid
    pos = track["actual"][:, :3, 3]
    goal = track["goal"][:, :3, 3]
    ang = track["abs_deg"]
    absolute = 1000 * np.linalg.norm(pos - goal, axis=1)
    relative = 1000 * np.linalg.norm((pos - pos[0]) - (goal - goal[0]), axis=1)
    if not (
        np.allclose(absolute, track["abs_mm"], atol=1e-6)
        and np.allclose(relative, track["delta_mm"], atol=1e-6)
    ):
        raise ValueError("measured Cartesian errors disagree with frozen evaluator")
    errors = [absolute, ang, relative, track["delta_deg"]]
    intervals = {
        "initial": (0, 1000),
        "moving": (1000, 2900),
        "first_parked_second": (2900, 3900),
        "full": (0, t[-1]),
    }
    summary = {}
    for name, (a, b) in intervals.items():
        mask = (t > a) & (t <= b)
        if not mask.any():
            raise ValueError("missing scoring interval")
        summary[name] = dict(
            samples=int(mask.sum()),
            rms=[float(np.sqrt(np.mean(x[mask] ** 2))) for x in errors],
            end=[float(x[mask][-1]) for x in errors],
            peak_actual_and_goal_speed_mm_s=[
                float(
                    np.linalg.norm(np.diff(x, axis=0)[mask[1:]], axis=1).max() * 500000
                )
                for x in (pos, goal)
            ],
        )
    streak = 0
    first = None
    for i, ok in enumerate((absolute <= 4) & (ang <= 2) & (t >= 2900)):
        streak = streak + 1 if ok else 0
        if streak >= 501:
            first = [float(t[i - 500] / 1000), float(t[i] / 1000)]
            break
    return dict(intervals=summary, first_full_dwell_start_and_end_s=first)


def run(root, repo):
    v0 = json.loads(
        (
            repo / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json"
        ).read_bytes()
    )
    stages = {
        s: load_checked(root / f"feedback/{s}_VERIFIED.json", v0["source_hashes"][s])
        for s in ("F1", "F2")
    }
    archive = root / "feedback"

    def trace(stage, row):
        campaign = next(archive.glob("sla-feedback-" + stage.lower() + "-*"))
        folder = campaign / row["card"]
        result = load_checked(folder / "RESULT.private.json", row["result_sha256"])
        if (
            sha(folder / "TRACK.private.npz") != row["curve_sha256"]
            or result["curve_sha256"] != row["curve_sha256"]
        ):
            raise ValueError("native curve identity")
        with np.load(folder / "TRACK.private.npz", allow_pickle=False) as z:
            tr = {k: z[k] for k in z.files}
        ret = score(tr)
        ret.update(
            card=row["card"],
            curve_sha256=row["curve_sha256"],
            violations=row["violations"],
            physics_stop=row["physics_stop"],
            max_reference_excursion_mm=row["max_cmd_excursion_mm"],
        )
        return tr, ret

    controls = {r["card"]: r for r in stages["F1"]["records"] if r["eligible"]}
    paired = []
    for r in stages["F2"]["records"]:
        if not r["eligible"]:
            continue
        c = controls.pop(r["paired_control"]["card"])
        ta, a = trace("F1", c)
        tb, b = trace("F2", r)
        length = min(len(ta["goal"]), len(tb["goal"]))
        if not np.array_equal(ta["goal"][:length], tb["goal"][:length]):
            raise ValueError("matched goals differ")
        paired.append(dict(A0=a, A1=b, common_goal_samples=length))
    if controls or len(paired) != 4:
        raise ValueError("incomplete paired population")
    return dict(
        populations={
            s: {
                k: stages[s][k]
                for k in (
                    "registered",
                    "started",
                    "unstarted",
                    "screened",
                    "physics_ticks",
                )
            }
            for s in stages
        },
        pairs=paired,
        metric_order=["abs_mm", "abs_deg", "delta_mm", "delta_deg"],
        new_physics_ticks=0,
        moving_stop_observations=0,
        orientation_excitation_deg=0,
        decision="V3_DYNAMIC_TRACKING_BLOCKED",
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run(Path(sys.argv[1]), Path(__file__).resolve().parents[2]),
            indent=2,
            allow_nan=False,
        )
    )
