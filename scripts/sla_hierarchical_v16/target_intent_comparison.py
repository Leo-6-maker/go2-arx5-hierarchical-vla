"""Compare all existing raw Go2 returns with the nominal plan from the query that generated them."""

import argparse
from pathlib import Path

import numpy as np
from compiler import SourceCurve
from demonstrated_grip import pin, read, require, rows
from scipy.spatial.transform import Rotation
from visual_lift_panel import S0, write_new


def errors(path, width, goal, target_width):
    delta = np.linalg.norm(path[:, :3, 3] - goal[:, :3, 3], axis=1) * 1000
    angle = (
        Rotation.from_matrix(
            path[:, :3, :3].transpose(0, 2, 1) @ goal[:, :3, :3]
        ).magnitude()
        * 180
        / np.pi
    )

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    return {
        "position_mm": rms(delta),
        "orientation_deg": rms(angle),
        "gripper_width_mm": rms(1000 * (width - target_width)),
        "first_15_position_mm": rms(delta[:15]),
    }


def score(root, prepared):
    repo = Path(__file__).resolve().parents[2]
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    paths = {value: Path(path) for path, value in pins.items() if path.endswith(".npy")}
    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    official = repo / "reports/SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json"
    old = read(
        pin(
            official,
            read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["O1"],
        )
    )["windows"]
    manifest = read(prepared / "result.json")["episodes"]
    results = []
    for episode, entry in manifest.items():
        folder = prepared / episode
        for name, sha in entry["artifacts"].items():
            pin(folder / name, sha)
        audit = rows(folder / "native_audit.jsonl")
        lookup = {r["tick"]: i for i, r in enumerate(audit)}
        with (
            np.load(folder / "task_audit.npz") as z,
            np.load(folder / "sensor_action.npz") as sensor,
        ):
            goal = z["nominal_world_at_source_times"]
            widths = 0.002 + 0.040 * sensor["nominal_action"][:, :, 15].astype(float)
            require(list(lookup) == list(sensor["query_tick"]), "query identity")
            for w in old:
                if w["episode"] != episode:
                    continue
                i = lookup[w["tick"]]
                q = audit[i]
                start = q["previous_command"]["width_m"]
                require(
                    abs(w["previous_consumed_grip_m"] - start) < 1e-10
                    and w["phase"] == q["phase"]
                    and z["confirmed_native_interval_mask"][i].tolist()
                    == [True] * 10 + [False] * 30,
                    "gripper start, phase or grant",
                )
                raw = np.load(
                    pin(paths[w["raw_sha256"]], w["raw_sha256"]), allow_pickle=False
                )
                curve = SourceCurve(raw, q["origin"], z["link6_tcp"])
                frozen = errors(
                    curve.knots,
                    0.002 + 0.040 * raw[:, 15].astype(float),
                    goal[i],
                    widths[i],
                )
                hold = errors(
                    np.broadcast_to(q["measured_tcp"], (50, 4, 4)),
                    np.full(50, start),
                    goal[i],
                    widths[i],
                )
                results.append(
                    {
                        "id": w["id"],
                        "episode": episode,
                        "phase": q["phase"],
                        "raw_sha256": w["raw_sha256"],
                        "tick": w["tick"],
                        "frozen": frozen,
                        "hold": hold,
                    }
                )
    require(
        sorted(r["id"] for r in results) == [f"T{i:02d}" for i in range(1, 17)],
        "missing window",
    )
    summary = {}
    for key in results[0]["frozen"]:
        f, h = [np.array([r[arm][key] for r in results]) for arm in ("frozen", "hold")]
        summary[key] = {
            "n": len(h),
            "frozen_lower": int(sum(f < h)),
            "tie": int(sum(f == h)),
            "median_frozen": float(np.median(f)),
            "median_hold": float(np.median(h)),
            "median_paired_frozen_minus_hold": float(np.median(f - h)),
        }
    return {
        "id": "V4-NOMINAL-01",
        "summary": summary,
        "windows": sorted(results, key=lambda r: r["id"]),
        "native_starts": 0,
        "new_forwards": 0,
        "training_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    write_new(a.out, score(a.root.resolve(), a.prepared))
