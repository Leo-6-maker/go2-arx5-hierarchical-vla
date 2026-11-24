"""Score frozen and adapted SLA intent on the same Go2 command clock and teacher receipt."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from compiler import SourceCurve
from demonstrated_grip import SOURCE_T, events, pin, read, require, rows
from target_intent_comparison import errors
from target_labels import task_labels
from visual_lift_panel import S0, write_new

GRID = np.arange(10) / 20
GO2_RANGE_M = (0.0, 0.044)
TRANSFER_RANGE_M = (0.002, 0.042)
MIN_DIRECTION_M = 0.001
CUSTODY = BASE / "reports/SLA_GO2_V17/ARCHIVE_READBACK.json"


def physical_width(fraction, limits):
    lo, hi = limits
    return lo + (hi - lo) * np.asarray(fraction, dtype=np.float64)


def event_metrics(width, target, previous):
    # Use the last actually consumed command at the preceding 50 ms knot.
    # The first predicted command can cross the threshold at t=0.
    clock = np.r_[-0.05, GRID]
    predicted = events(clock, np.r_[previous, width] / GO2_RANGE_M[1], step=True)
    expert = events(clock, np.r_[previous, target] / GO2_RANGE_M[1], step=True)
    seq = [e["kind"] for e in predicted]
    expected = [e["kind"] for e in expert]
    p, t = Counter(seq), Counter(expected)
    return dict(
        proposed=predicted, teacher=expert, order_equal=seq == expected,
        extra=dict(p - t), missed=dict(t - p),
        paired_error_ms=[1000 * (a["s"] - b["s"]) for a, b in zip(predicted, expert)]
        if seq == expected else None,
    )


def direction_metrics(pose, goal, current):
    intended = goal[-1, :3, 3] - current
    proposed = pose[-1, :3, 3] - current
    target_norm = np.linalg.norm(intended)
    if target_norm < MIN_DIRECTION_M:
        return dict(reference_mm=float(1000 * target_norm), signed_mm=None,
                    lateral_mm=None, cosine=None)
    unit = intended / target_norm
    signed = float(proposed @ unit)
    norm = float(np.linalg.norm(proposed))
    return dict(
        reference_mm=float(1000 * target_norm),
        signed_mm=1000 * signed,
        lateral_mm=float(1000 * np.linalg.norm(proposed - signed * unit)),
        cosine=signed / norm if norm >= MIN_DIRECTION_M else None,
    )


def score_window(raw, nominal, mask, audit, tool, *, limits, teacher_on_source):
    """Evaluate intent at ten P10 command instants and separately across source support."""
    previous = float(audit["previous_command"]["width_m"])
    measured = np.asarray(audit["measured_tcp"], dtype=np.float64)
    origin = np.asarray(audit["origin"], dtype=np.float64)
    tick = audit["tick"]
    intervals = audit["confirmed_native"]
    require(
        np.array_equal(mask, [True] * 10 + [False] * 30)
        and len(intervals) == 10
        and all(
            r["index"] == j and r["start"] == tick + 25*j and r["end"] == tick + 25*(j+1)
            for j, r in enumerate(intervals)
        )
        and abs(audit["previous_command"]["source_native_s"] - (tick - 25)*.002) < 1e-8
        and 0 <= previous <= GO2_RANGE_M[1] + 1e-10,
        "finite P10 receipt and real previous command required",
    )

    def score(grid_pose, grid_width, source_pose, source_width):
        near = errors(grid_pose, grid_width, target_pose, target_width)
        del near["first_15_position_mm"]  # The near-term grid has ten points.
        return dict(
            P10_command_instants=near,
            source_horizon_diagnostic=errors(source_pose, source_width, teacher_pose, teacher_width),
            P10_gripper_events=event_metrics(grid_width, target_width, previous),
            P10_direction=direction_metrics(grid_pose, target_pose, measured[:3, 3]),
            P10_nominal_width_max_slew_mm_s=float(
                np.max(abs(np.diff(np.r_[previous, grid_width]))) * 1000 / .05
            ),
        )

    # Sample the teacher's original 20 Hz geometry at the exact coincident 30 Hz
    # knots and at the intervening ones, using its original SE(3) / width codec.
    encoded, teacher_pose = task_labels(nominal, origin, tool, np.zeros(8))
    require(np.allclose(teacher_pose, teacher_on_source, rtol=0, atol=1e-10),
            "archived nominal world trajectory changed")
    # Use the exact ten original 20 Hz teacher knots. Interpolate only the
    # model side for this grid, keeping both arms in the same world frame.
    x, y = nominal[:10, 3:6], nominal[:10, 6:9]
    target_pose = np.tile(np.eye(4), (10, 1, 1))
    target_pose[:, :3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    target_pose[:, :3, 3] = nominal[:10, :3]
    target_width = physical_width(nominal[:10, 9], GO2_RANGE_M)
    teacher_width = physical_width(encoded[:, 15], GO2_RANGE_M)
    curve = SourceCurve(raw, origin, tool)
    return dict(
        candidate=score(
            curve.at(GRID * 30), physical_width(curve.grip(GRID * 30), limits),
            curve.knots, physical_width(curve.grip(SOURCE_T * 30), limits),
        ),
        hold=score(
            np.broadcast_to(measured, (10, 4, 4)), np.full(10, previous),
            np.broadcast_to(measured, (50, 4, 4)), np.full(50, previous),
        ),
    )


def summarize(group):
    by_episode = defaultdict(list)
    for r in group:
        by_episode[r["episode"]].append(r)
    metrics = {}
    for key in ("position_mm", "orientation_deg", "gripper_width_mm"):
        result = {}
        for arm in ("A_original_codec", "A_common_codec", "hold"):
            key_path = "P10_command_instants"
            equally_weighted_episodes = [
                np.mean([r[arm][key_path][key] for r in windows])
                for windows in by_episode.values()
            ]
            result[arm] = float(np.mean(equally_weighted_episodes))
        metrics[key] = result
    return dict(
        episodes=len(by_episode), windows=len(group), P10_episode_macro_mean=metrics,
        P10_nontrivial_direction_windows=sum(
            r["A_original_codec"]["P10_direction"]["signed_mm"] is not None for r in group
        ),
        P10_negative_direction_windows=sum(
            r["A_original_codec"]["P10_direction"]["signed_mm"] is not None
            and r["A_original_codec"]["P10_direction"]["signed_mm"] < 0 for r in group
        ),
        P10_teacher_event_windows=sum(bool(r["hold"]["P10_gripper_events"]["teacher"]) for r in group),
        P10_event_order_matches={
            arm: sum(r[arm]["P10_gripper_events"]["order_equal"] for r in group)
            for arm in ("A_original_codec", "A_common_codec", "hold")
        },
    )


def replay(root):
    v17 = read(CUSTODY)["files"]
    train = BASE / "reports/SLA_GO2_V17/TRAIN_QUALIFICATION.json"
    prepared = read(pin(train, v17[train.name]))["episodes"]
    reports = BASE / "reports/SLA_HIERARCHICAL_V16"
    old = BASE / "reports/SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json"
    requested = read(pin(old, read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["O1"]))
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    raw_paths = {sha: Path(path) for path, sha in pins.items() if path.endswith(".npy")}
    scored, releases = [], []
    for name in sorted({r["episode"] for r in requested["windows"]}):
        entry = prepared[name]
        require(entry["status"] == "verified_seed", "fixed six TRAIN controls")
        folder = root / "v16/target-labels04" / name
        for path, sha in entry["prepared"]["artifacts"].items():
            pin(folder / path, sha)
        audit = rows(folder / "native_audit.jsonl")
        lookup = {r["tick"]: i for i, r in enumerate(audit)}
        with np.load(folder / "task_audit.npz", allow_pickle=False) as z:
            nominal = z["nominal_world_H40"]
            teacher = z["nominal_world_at_source_times"]
            mask, tool = z["confirmed_native_interval_mask"], z["link6_tcp"]
            require(np.array_equal(z["nominal_clock_s"], np.arange(40)*.05)
                    and np.array_equal(z["source_clock_s"], SOURCE_T), "archive clocks changed")
        for w in requested["windows"]:
            if w["episode"] != name:
                continue
            i = lookup[w["tick"]]
            query = audit[i]
            require(query["phase"] == w["phase"] and
                    abs(query["previous_command"]["width_m"] - w["previous_consumed_grip_m"]) < 1e-10,
                    "previous consumed command or phase changed")
            raw = np.load(pin(raw_paths[w["raw_sha256"]], w["raw_sha256"]), allow_pickle=False)
            def run(limits):
                return score_window(raw, nominal[i], mask[i], query, tool,
                                    limits=limits, teacher_on_source=teacher[i])
            original = run(TRANSFER_RANGE_M)
            common = run(GO2_RANGE_M)
            scored.append(dict(id=w["id"], episode=name, phase=query["phase"], tick=w["tick"],
                               raw_sha256=w["raw_sha256"], A_original_codec=original["candidate"],
                               A_common_codec=common["candidate"], hold=original["hold"]))
        # A held-out opening prediction does not exist in this archive. Check
        # that the metric can detect omitted release using a measured PLACE receipt.
        if entry["task"].startswith("PLACE"):
            for i, q in enumerate(audit):
                if q["phase"] != "OPEN":
                    continue
                prior = q["previous_command"]["width_m"]
                width = physical_width(nominal[i, :10, 9], GO2_RANGE_M)
                oracle = event_metrics(width, width, prior)
                if any(event["kind"] == "open" for event in oracle["teacher"]):
                    follower = event_metrics(np.full(10, prior), width, prior)
                    require(oracle["order_equal"] and follower["missed"]["open"] >= 1,
                            "release-control regression")
                    releases.append(dict(episode=name, tick=q["tick"], missed_by_hold=follower["missed"]))
                    break
    require(sorted(r["id"] for r in scored) == [f"T{i:02d}" for i in range(1, 17)] and releases,
            "archive population or opening positive is missing")
    phase = {p: summarize([r for r in scored if r["phase"] == p]) for p in sorted({r["phase"] for r in scored})}
    return dict(
        scope="ARCHIVED_TRAIN_REPLAY", evaluation_minutes_simulated=0,
        P10_command_times_s=GRID.tolist(), source_support_s=[float(SOURCE_T[0]), float(SOURCE_T[-1])],
        teacher_and_common_width_m=list(GO2_RANGE_M), original_A_width_m=list(TRANSFER_RANGE_M),
        release_metric_original_teacher_controls=releases, total=summarize(scored), phases=phase,
        windows=sorted(scored, key=lambda r: r["id"]),
        independent_VAL_windows=0, TEST_windows=0, C_forwards=0, new_model_forwards=0,
        optimizer_updates=0, native_starts=0, integration_ticks=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    write_new(args.out, replay(args.root.resolve()))
