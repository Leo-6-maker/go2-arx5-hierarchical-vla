"""Compare pre-source clocks on the sealed model outputs and original teacher contact stream."""

import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sealed_panel import configure_site


def entry_bounds(compiled, limits):
    """Independent per-axis necessary times, allowing instantaneous acceleration switches."""
    from compiler import DT, relative_angle

    displacement = {
        "linear": np.linalg.norm(compiled.first[:3, 3] - compiled.stopped[:3, 3]),
        "angular": relative_angle(compiled.first[:3, :3], compiled.stopped[:3, :3]),
    }
    r = compiled.cruise_clock.ramp_fraction
    peak_rate = 1 / (compiled.align * (1 - r))
    peak_accel = 1.5 * peak_rate / (r * compiled.align)
    axes = {}
    for key, d in displacement.items():
        a = 0.9 * limits[key + "_acceleration"]
        v = min(0.9 * limits[key + "_velocity"], a * DT)
        lower = 2 * math.sqrt(d / a) if d < v * v / a else d / v + v / a
        axes[key] = {
            "distance": float(d),
            "velocity_cap": v,
            "acceleration_cap": a,
            "peak_velocity": d * peak_rate,
            "peak_acceleration": d * peak_accel,
            "minimum_s": lower,
        }
    return axes


def first_plan(root, case, scene, old, baseline):
    from compiler import DT, Compiler
    from demonstrated_grip import require
    from gripper_alignment import alignment_widths
    from handoff_factors import contact, gates
    from prefix_replay import WireProgram
    from reachability import check_wire
    from retime import crossing

    before = case["context"][-1]["width_m"]
    require(abs(case["grip"]["previous_command_m"] - before) < 1e-10, "consumed width")
    model = case["curve"]
    c = Compiler(
        model,
        case["context"][3],
        case["incoming"],
        case["limits"],
        cruise_from_width_m=before,
    )
    t = DT * np.arange(round(c.end / DT) + 1)
    poses, width, phase = c.at(t)
    # Verify the whole source suffix and the exact incoming brake against the archived wire.
    new_start = round(c.lead / DT)
    old_start = round(baseline["lead_s"] / DT)
    brake = round(c.brake / DT) + 1
    with np.load(
        root / "v16/v1-run01" / case["key"] / "C2_EXACT.private.npz", allow_pickle=False
    ) as z:
        for name, array in (("poses", poses), ("grip", width), ("source_s", phase)):
            ref = z[name]
            require(
                np.allclose(array[:brake], ref[:brake], atol=1e-13, rtol=0),
                "altered incoming brake",
            )
            tail = ref[old_start : old_start + len(t) - new_start]
            require(
                np.allclose(array[new_start:], tail, atol=1e-11, rtol=0),
                "changed source path, events or timing",
            )
    aligned, entry = alignment_widths(t, phase, width, before, c.lead, contact(case))
    row = {
        "id": case["key"],
        "old_lead_s": baseline["lead_s"],
        "lead_s": c.lead,
        "brake_s": c.brake,
        "entry_bounds": entry_bounds(c, case["limits"]),
        "alignment": entry,
        "old_qualified": old["aligned_qualified"],
        "phase_proxy": old["phase_proxy"],
        "aligned_qualified": False,
    }
    if aligned is None:
        return row, None
    a, b = list(crossing(aligned)), list(crossing(width))
    require(
        a == b and [k for k, _ in a] == [k for k, _ in model.events],
        "source threshold changed",
    )
    plan = WireProgram(case["episode"], poses, aligned, phase)
    gate = gates(plan, before, math.ceil(c.end / 0.1 - 1e-10) + 1)
    geom = check_wire(scene, case, t, poses, aligned)
    row.update(
        gates=gate,
        geometry=geom,
        events_preserved=baseline["events_preserved"],
        boundary_pass=baseline["boundary_pass"],
        aligned_qualified=bool(
            baseline["boundary_pass"]
            and baseline["events_preserved"]
            and not any(gate["envelopes"].values())
            and gate["g7"]
            and geom["pass"]
        ),
    )
    return row, plan


def run(root, repo, out):
    from ablation import output
    from compiler import DT
    from continuous_revisions import episode_replay, qualify, timeline
    from demonstrated_grip import pin, read, require, rows
    from frozen_geometry import Scene
    from noncontact_replay import main as contact_replay
    from revision_replay import primary_calls
    from revision_wire import first_publication
    from sealed_panel import cases

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    h = read(
        pin(
            reports / "HANDOFF_FACTORS_ACTUAL.json",
            "c0b6a25270e793078482ba3416a1c795d2d88277d1cc38a61f447f9fc2ac610b",
        )
    )
    prior = read(
        pin(
            reports / "H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )
    noncontact = read(
        pin(
            reports / "NONCONTACT_FEEDBACK_ACTUAL.json",
            "8aa2793bfe4668b0bb23ccdabb5404f79bddf055e62455eeecf9ca4ebda9d22a",
        )
    )
    old_windows = {x["id"]: x for x in h["initial"]}
    ablation = read(
        pin(
            reports / "V1_COMPILER_ABLATION_ACTUAL.json",
            "766965b3528f27b56feabc15ec70b6c346d101196832ea736c9fa359b6d01419",
        )
    )
    baselines = {x["id"]: x["modes"]["C2_EXACT"] for x in ablation["windows"]}
    pop = {c["key"]: c for c in cases(root, repo)}
    calls = primary_calls(
        root, pop, read(reports / "V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json")
    )
    require(
        prior["calls"] == calls and len(pop) == len(old_windows) == 16,
        "model population",
    )
    out.mkdir()
    scenes, first, programs = {}, {}, {}
    for key, case in pop.items():
        path = Path(case["spec"]["model_path"])
        if path not in scenes:
            scenes[path] = Scene(
                path, root / "v16/scene_layout", case["spec"]["frozen_files"][str(path)]
            )
        first[key], programs[key] = first_plan(
            root, case, scenes[path], old_windows[key], baselines[key]
        )
    for key, r in first.items():
        lower = max(axis["minimum_s"] for axis in r["entry_bounds"].values())
        # First positive source knot is strictly past completion of grid-aligned entry.
        grid_intervals = math.ceil(lower / DT - 1e-12)
        r["earliest_current_age_if_initial_s"] = (
            first_publication(calls[key]["seconds"]) / 10
            + r["brake_s"]
            + (grid_intervals + 1) * DT
        )
        require(
            r["lead_s"] + 1e-10 >= r["brake_s"] + grid_intervals * DT,
            "entry faster than physical bound",
        )
    pairs = {}
    for old in h["pairs"]:
        if old["protection"] != "FULL" or not old["alignment"]:
            continue
        a, b = old["old"], old["new"]
        group = timeline([calls[a], calls[b]], False)
        cut_at = group[1]["publish"] - group[0]["publish"]
        row = {
            "old": a,
            "new": b,
            "old_pass": old["geometric_and_numeric_pass"],
            "geometric_and_numeric_pass": False,
            "old_motion_age_s": old.get("new_source_motion_age_s"),
        }
        pairs[a, b] = row
        if not first[a]["aligned_qualified"]:
            row["reason"] = "INITIAL_REJECTED"
            continue
        scene = scenes[Path(pop[a]["spec"]["model_path"])]
        plan, evidence = qualify(
            programs[a], pop[a], pop[b], cut_at, pop[a], scene, cruise_alignment=True
        )
        row.update(evidence=evidence, reason=evidence["reason"])
        if plan is not None:
            t = DT * np.arange(round(plan.execution_times[-1] / DT) + 1)
            pose, _, phase = plan.at(t)
            moved = (
                (np.arange(len(t)) > plan.cut)
                & (phase > 0)
                & (
                    np.linalg.norm(
                        pose[:, :3, 3] - pop[b]["curve"].knots[0, :3, 3], axis=1
                    )
                    > 1e-9
                )
            )
            row.update(
                geometric_and_numeric_pass=True,
                new_motion_age_s=float(
                    t[np.flatnonzero(moved)[0]]
                    + first_publication(calls[a]["seconds"]) / 10
                    - group[1]["capture_s"]
                ),
            )
    groups = defaultdict(list)
    for call in calls.values():
        groups[call["episode"]].append(call)
    require(len(groups) == 6 and len(pairs) == 10, "episode or pair population")
    archive = out / "continuous"
    archive.mkdir()
    episodes = {}
    for ep, group in groups.items():
        group.sort(key=lambda x: x["tick"])
        key = group[0]["id"]
        scene = scenes[Path(pop[key]["spec"]["model_path"])]
        episodes[ep] = {}
        for mode, fifo in (("fifo", True), ("overlap", False)):
            episodes[ep][mode] = episode_replay(
                archive / (ep + "-" + mode),
                pop,
                timeline(group, fifo),
                programs[key],
                first[key],
                scene,
                pairs,
                None,
                cruise_alignment=True,
            )
    output(archive / "ACTUAL.json", {"episodes": episodes})
    digest = hashlib.sha256((archive / "ACTUAL.json").read_bytes()).hexdigest()
    contact_dir = out / "contact"
    contact_replay(root, repo, contact_dir, archive=archive, record_sha256=digest)
    guarded = read(contact_dir / "ACTUAL.json")["episodes"]
    compact = {}
    for ep, conditions in episodes.items():
        compact[ep] = {}
        for mode, e in conditions.items():
            actual = guarded[ep][mode]
            prior_contact = noncontact["episodes"][ep][mode]
            cutoff = min(
                actual["stop"]["pause_tick"], prior_contact["stop"]["observed_tick"]
            )
            if actual["stop"]["reason"] == "UNEXPECTED_CONTACT":
                require(
                    actual["stop"] == prior_contact["stop"],
                    "contact identity or clock changed",
                )
            require(
                actual["receiver_hold"] == ["HOLD"]
                and not actual["installed_receipts_after_prefix"],
                "unguarded callback after stop",
            )
            received = {
                r["ref_id"]
                for r in rows(
                    contact_dir / (ep + "-" + mode) / "CALLBACKS.private.jsonl"
                )
                if r["kind"] == "installed"
            }
            details = {}
            for k, r in e["rows"].items():
                due = []
                for m in r.get("current_motion", []):
                    tick = calls[k]["tick"] + round(m["age_s"] / 0.002)
                    if tick < cutoff:
                        require(
                            m["callback_sha256"] in received,
                            "no actual contact-gated install",
                        )
                        due.append(dict(**m, tick=tick))
                details[k] = {
                    "status": r["status"],
                    "first_current_age_s": r["first_nonzero_source_motion_s"],
                    "due_before_contact": len(due),
                    "first": due[0] if due else None,
                    "last": due[-1] if due else None,
                    "max_displacement_mm": max(
                        (d["displacement_mm"] for d in due), default=0.0
                    ),
                }
            compact[ep][mode] = {
                "first_teacher_contact_tick": prior_contact["stop"]["observed_tick"],
                "stop": actual["stop"],
                "guarded_installs": actual["actual_prefix_installs"],
                "guarded_prefix_sha256": actual["prefix_sha256"],
                "terminal_hold": actual["receiver_hold"],
                "rows": details,
            }
    output(
        out / "ACTUAL.json",
        {
            "windows": list(first.values()),
            "pairs": list(pairs.values()),
            "episodes": compact,
            "continuous_sha256": digest,
            "contact_replay_sha256": hashlib.sha256(
                (contact_dir / "ACTUAL.json").read_bytes()
            ).hexdigest(),
            "native_starts": 0,
            "new_forwards": 0,
            "training_updates": 0,
        },
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path.insert(0, str(root / "v16/cpu-fcl"))
    run(root, repo, Path(sys.argv[2]).resolve())
