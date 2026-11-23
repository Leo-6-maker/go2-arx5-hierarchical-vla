"""Read-only controls for the registered moving-stop scoring window in original native A0 traces."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require
from fault_reference import (
    bound_path,
    floor_feet,
    motion_from_install,
    records,
    solver_echo,
)
from sealed_panel import configure_site
from stop_card_readiness import displacement


def negative_controls(native, cutoff, goal, base, reference):
    start = reference["source_tick"]
    wire = reference["wire_sha256"]
    (atom,) = [
        a
        for a in records(native / "MOBILE_SAMPLES.private.jsonl")
        if a["native_start_step"] == start
    ]
    grants = [
        g
        for g in records(native / "CLOCK_RESUME_REQUESTS.jsonl")
        if g["requested_step"] in (start + 25, start + 50)
    ]
    planner = list(records(native / "planner.jsonl"))
    accepted = next(
        r for r in planner if r.get("ref_id") == wire and r["kind"] == "accept"
    )
    installed = next(
        r for r in planner if r.get("ref_id") == wire and r["kind"] == "installed"
    )
    previous = next(
        r
        for r in reversed(planner)
        if r["kind"] == "installed" and r["monotonic_s"] < accepted["monotonic_s"]
    )
    require(previous["ref_id"] != wire, "missing independent prior wire")
    amended = sorted(
        [
            *planner,
            {**previous, "monotonic_s": (grants[-1]["monotonic_s"] + cutoff) / 2},
        ],
        key=lambda r: r["monotonic_s"],
    )
    try:
        bound_path(amended, atom, native.name, start, grants, cutoff, goal, base)
    except ValueError as exc:
        require(
            str(exc) == "reference revoked or superseded before boundary",
            "wrong stale-install rejection",
        )
    else:
        raise ValueError("conflicting actual install admitted")

    def without_target():
        for row in records(native / "ROS_CONSUMPTION.private.jsonl"):
            if row["topic_key"] == "policy":
                target = row["message"]["plan_target_trajectories"]["time_trajectory"]
                if target and abs(target[0] - start * 0.002) < 1e-8:
                    continue
            yield row

    try:
        solver_echo(
            without_target(),
            installed,
            reference["installed_target_sha256"],
            grants[-1]["monotonic_s"],
        )
    except ValueError as exc:
        require(
            str(exc) == "no independent solver echo before final grant",
            "wrong missing-echo rejection",
        )
    else:
        raise ValueError("missing actual solver echo admitted")
    return {
        "old_wire_reinstalled": "rejected",
        "solver_echo_removed": "rejected",
        "donor_wire_sha256": previous["ref_id"],
    }


def controls(root, repo):
    configure_site(root, repo)
    sys.path.insert(0, str(repo / "scripts/sla_transfer_feedback"))
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from native_known import target, window
    from verify_grip_card import states

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    matrix = read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")
    roster = read(
        pin(
            repo / "reports/SLA_FROZEN_FEEDBACK/F1_ACTUAL.json",
            matrix["source_hashes"]["F1"],
        )
    )["records"]
    source = {
        r["card"]: r for r in read(reports / "NATIVE_REVISION_ACTUAL.json")["records"]
    }
    previous_counts = {
        r["card"]: r
        for r in read(reports / "V3_CLOSED_COUNT_ARCHIVE_ACTUAL.json")["records"]
    }
    out = []
    for r in roster:
        result = {
            "card": r["card"],
            "eligible": r["eligible"],
            "original_outcome": r["outcome"],
        }
        out.append(result)
        if not r["eligible"]:
            continue
        case = root / "feedback/sla-feedback-f1-a0-20261123-01" / r["card"]
        evidence = read(pin(case / "RESULT.private.json", r["result_sha256"]))
        initial = read(pin(case / "INITIAL.private.json", r["initial_sha256"]))[
            "features"
        ]
        native = case / evidence["episode"]
        anchor = np.asarray(initial["old_ee"])
        base = np.asarray(initial["old_base"])
        result["source_sha256"] = {
            "ROS_CONSUMPTION.private.jsonl": source[r["card"]]["ros_sha256"],
            "state.bin.journal": r["journal_sha256"],
            "planner.jsonl": source[r["card"]]["planner_sha256"],
            "MOBILE_SAMPLES.private.jsonl": source[r["card"]]["atomic_record_sha256"],
            "init.trace": previous_counts[r["card"]]["evidence_sha256"]["init.trace"],
        }
        for name, digest in result["source_sha256"].items():
            pin(native / name, digest)
        limits = {
            l["token"]: l["monotonic_ns"] * 1e-9
            for l in read_trace(native / "init.trace")
            if l["kind"] == 12 and l["token"] in (5939, 6439)
        }
        with (native / "physics.jsonl").open() as f:
            header = json.loads(f.readline())
        geos = header["geoms"]
        require(all(g["id"] == i for i, g in enumerate(geos)), "geometry indices")
        names = [g["name"] for g in geos]
        supports = {}
        for packet in states(native):
            t = packet["step"]
            if 5914 <= t <= 5939 or 6414 <= t <= 6439:
                supports[t] = floor_feet(names, packet)
            if t == 6439:
                break
        track = pin(case / "TRACK.private.npz", r["curve_sha256"])
        with np.load(track) as z:
            body = z["actual"][:, :3, 3]
            native_ticks = np.rint(z["elapsed_ms"] / 2).astype(int) + 5703
        result["windows"] = {}
        for kind, origin in (("HOLD", 5903), ("MOVING", 6403)):
            end = origin + 36
            ticks = np.arange(end - 25, end + 1)
            info = motion_from_install(native, origin, limits[end], anchor, base)
            actual = displacement(body, native_ticks, end, 25)
            plan = displacement(
                target(anchor, (ticks - 5703) * 0.002)[:, :3, 3], ticks, end, 25
            )
            feet = [supports[i] for i in ticks]
            result["windows"][kind] = dict(
                **info,
                achieved=actual,
                analytic=plan,
                commanded_motion_positive=info["installed_motion"]["distance_mm"] > 0,
                meets_motion_pair=info["installed_motion"]["distance_mm"] > 0
                and actual["distance_mm"] > 0.10,
                distinct_foot_sets=sorted(set(feet)),
                four_feet_samples=sum(len(f) == 4 for f in feet),
                samples=len(feet),
            )
        result["negative_controls"] = negative_controls(
            native,
            limits[6439],
            window(anchor, 1.4),
            np.broadcast_to(base, (41, 4, 4)),
            result["windows"]["MOVING"],
        )
        print(r["card"], flush=True)
    require(len(out) == 12 and sum(r["eligible"] for r in out) == 4, "F1 population")
    return {
        "card": "V3-STOP-01",
        "controls": out,
        "archived_starts": len(out),
        "new_native_starts": 0,
        "new_gpu_forwards": 0,
        "training_updates": 0,
        "whole_system_moving_STOP_qualified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = controls(args.root.resolve(), Path(__file__).resolve().parents[2])
    with args.out.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
