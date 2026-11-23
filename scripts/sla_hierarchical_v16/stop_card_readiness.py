"""Evaluate a single registered moving STOP interval on the complete historical population."""

import json
import sys
from pathlib import Path

import numpy as np

from receiver_fault_replay import read, sha
from sealed_panel import configure_site


def displacement(positions, ticks, stop_tick, baseline_ticks):
    indices = np.flatnonzero(
        (ticks >= stop_tick - baseline_ticks) & (ticks <= stop_tick)
    )
    if not np.array_equal(
        ticks[indices], np.arange(stop_tick - baseline_ticks, stop_tick + 1)
    ):
        raise ValueError("missing or duplicate motion samples")
    p = positions[indices]
    if p.shape != (baseline_ticks + 1, 3) or not np.isfinite(p).all():
        raise ValueError("invalid measured positions")
    distance = 1000 * float(np.linalg.norm(p[-1] - p[0]))
    return {
        "distance_mm": distance,
        "mean_displacement_speed_mm_s": distance / (0.002 * baseline_ticks),
        "last_step_speed_mm_s": float(np.linalg.norm(p[-1] - p[-2]) * 500000),
    }


def run(root):
    from b2_owner import baseline, check_occupancy
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock

    repo = Path(__file__).resolve().parents[2]
    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    config = read(reports / "V3_NATIVE_STOP_CARD_CANDIDATE.json")
    hashes = read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]
    stop_tick = config["planned_stop_tick"]
    out = []
    for group in ("F1", "F2"):
        roster = root / "feedback" / (group + "_VERIFIED.json")
        if sha(roster) != hashes[group]:
            raise ValueError("changed population")
        cases = read(roster)["records"]
        for row in cases:
            if not row["eligible"]:
                out.append(dict(card=row["card"], eligible=False))
                continue
            (folder,) = (root / "feedback").glob(
                "sla-feedback-" + group.lower() + "-*/" + row["card"]
            )
            track = folder / "TRACK.private.npz"
            r = folder / "RESULT.private.json"
            if sha(track) != row["curve_sha256"] or sha(r) != row["result_sha256"]:
                raise ValueError("changed history " + row["card"])
            with np.load(track, allow_pickle=False) as z:
                elapsed = z["elapsed_ms"]
                ticks = (elapsed / 2).round().astype(int) + 5703
                if not np.allclose(elapsed, (ticks - 5703) * 2, rtol=0, atol=1e-8):
                    raise ValueError("off-grid clock")
                motion = displacement(z["actual"][:, :3, 3], ticks, stop_tick, 25)
                goal = displacement(z["goal"][:, :3, 3], ticks, stop_tick, 25)
            out.append(
                dict(
                    card=row["card"],
                    eligible=True,
                    curve_sha256=row["curve_sha256"],
                    achieved=motion,
                    desired=goal,
                    moving=motion["distance_mm"] > 0.10 and goal["distance_mm"] > 0,
                )
            )
    if len(out) != 18 or sum(row["eligible"] for row in out) != 8:
        raise ValueError("changed start denominator")
    # This short preflight releases the mutex on return. It never creates a lease.
    spec = baseline(root)
    with closing(OwnerLock(Path(spec["mobile_world_lock"]))):
        check_occupancy(spec)
    return dict(
        candidate=config["id"],
        stop_tick=stop_tick,
        histories=out,
        original_starts=len(out),
        admitted_curves=8,
        exclusive_mutex_checked_and_released=True,
        live_lease_issued=False,
        new_native_starts=0,
        new_physics_ticks=0,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    configure_site(root, Path(__file__).resolve().parents[2])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_transfer_v14"))
    result = run(root)
    with Path(sys.argv[2]).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
