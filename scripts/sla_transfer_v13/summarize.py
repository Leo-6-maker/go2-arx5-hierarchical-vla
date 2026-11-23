"""Publish only scalar results, counts, hashes and identifiers; keep sensor/command arrays private."""

from pathlib import Path
import sys
import numpy as np
from audit import read, record, sha
from acceptance import handover, nontrivial, physical_grip_kinematics
from retime import NAMES


def quant(vals):
    return None if not len(vals) else [float(x) for x in np.quantile(vals, [0, 0.5, 0.95, 1.0])]


def main(out):
    a = [read(out / f"S{i}_ACTUAL.json") for i in range(4)]
    s0, s1, s2, s3 = a
    can = read(out / "CANONICAL_ACTUAL.json")["rows"]
    cg = read(out / "CANONICAL_GATE.json")["rows"]
    canonical = {(r["id"], r["arm"]): r for r in cg}
    groups = {}
    full_grip = {}
    table = []
    for pop, f in [
        ("TARGET", lambda r: r["population"] == "TARGET"),
        ("SOURCE_EXPERT", lambda r: "expert_absolute" in r["id"]),
        ("SOURCE_O0", lambda r: "native_absolute" in r["id"]),
    ]:
        rows = [r for r in s3["windows"] if f(r)]
        gp = {}
        for name in ["C0", "UNIFORM_DIAGNOSTIC", "C1", "C2", "C3"]:
            arms = [(r, r["arms"][name]) for r in rows]
            windows = [(r, m) for r, m in arms if "array_file" in m]
            ms = [v for _, v in windows]
            target = pop == "TARGET"
            canonical_ok = [canonical[(r["id"], name)] for r, _ in windows]
            gp[name] = dict(
                registered=len(arms),
                compiled=len(ms),
                blocked=len(arms) - len(ms),
                EEF_dynamic_pass=sum(m["nominal_pose_channels_pass"] for m in ms),
                complete_H40_pass=sum(m["H40"] for m in ms),
                G7_pass=sum(
                    not m["native_gate"]["G7"]["original_execution_speed_exceeded"] for m in ms
                )
                if target
                else None,
                H40_and_G7_pass=sum(m["native_gate"]["H40_G7_pass"] for m in ms)
                if target
                else None,
                canonical_H40_pass=sum(bool(r.get("H40")) for r in canonical_ok),
                canonical_both_pass=sum(bool(r.get("legal")) for r in canonical_ok)
                if target
                else None,
                motion_pass=sum(bool(m["progress"]["motion_pass"]) for m in ms) if target else None,
                events_pass=sum(m["progress"]["event_pass"] for m in ms),
                nontrivial_admissible=sum(nontrivial(m, c) for m, c in zip(ms, canonical_ok))
                if target
                else None,
                duration_s_min_med_p95_max=quant(
                    [m["timing"]["duration_s"] for m in ms if m.get("timing")]
                ),
                P10_travel_mm_min_med_p95_max=quant(
                    [1000 * m["progress"]["P10_travel_m"] for m in ms]
                ),
                preview_fraction_min_med_p95_max=quant(
                    [m["progress"]["preview_fraction"] for m in ms]
                ),
            )
            for r, m in windows:
                with np.load(out / "arrays" / m["array_file"]) as z:
                    g = z["H40"][:, 9]
                last = (
                    next(x["previous_command_m"] for x in s2["windows"] if x["id"] == r["id"])
                    if target
                    else None
                )
                full_grip[r["id"] + "_" + name] = physical_grip_kinematics(g, last)
        groups[pop] = gp
    for d, t, g, w in zip(
        s0["windows"], s1["windows"], s2["windows"], s3["windows"][:16], strict=True
    ):
        c = w["arms"]["C1"]
        u = w["arms"]["UNIFORM_DIAGNOSTIC"]
        decision = handover(g)
        assert (
            decision.startswith("HANDOVER_BLOCKED")
            and w["arms"]["C3"]["status"] == "HANDOVER_BLOCKED"
        )
        act = c["progress"]
        canrow = next(r for r in can if r["id"] == w["id"] and r["arm"] == "C1")
        table.append(
            dict(
                id=w["id"],
                seed=w["seed"],
                phase=w["phase"],
                H40_G7=c["native_gate"]["H40_G7_pass"],
                H40=c["H40"],
                G7=not c["native_gate"]["G7"]["original_execution_speed_exceeded"],
                EEF=c["nominal_pose_channels_pass"],
                motion=act["motion_pass"],
                events=act["event_pass"],
                source_event_count=len(act["events_source"]),
                compiled_event_count=len(act["events_compiled"]),
                P10_mm=1000 * act["P10_travel_m"],
                min_P10_mm=1000 * act["P10_min_m"],
                P10_z_mm=1000 * act["P10_net_m"][2],
                projected_mm=1000 * act["projected_m"],
                coverage=act["preview_fraction"],
                duration_s=c["timing"]["duration_s"],
                uniform_s=u["timing"]["duration_s"],
                needed_replans=c["timing"]["planning_updates"],
                canonical_s_ref_end_m=canrow["s_ref"][-1],
                canonical_delta_mm=1000 * canrow["pos_deviation_max_m"],
                canonical_angle_rad=canrow["rotation_deviation_max_rad"],
                consumed_boundary_linear_accel=t["first_accel_from_consumed"][0],
                consumed_boundary_angular_accel=t["first_accel_from_consumed"][1],
                original_seam_mm_s=1000 * g["seam_speed_m_s"],
                measured_mm=1000 * g["measured_m"],
                previous_mm=1000 * g["previous_command_m"],
                first_mm=1000 * g["new_first_m"],
                handover=decision,
                contact=g["contact"],
                grasp=g["grasp"],
                source30_internal_peak_ratio={
                    k: t["raw30"][k]["regions"]["interior"]["ratios_med_p95_max"][-1] for k in NAMES
                },
                H40_internal_peak_ratio={
                    k: t["resampled_H40"][k]["regions"]["interior"]["ratios_med_p95_max"][-1]
                    for k in NAMES
                },
                suffix_peak_ratio={
                    k: t["resampled_H40"][k]["regions"]["suffix"]["ratios_med_p95_max"][-1]
                    for k in NAMES
                },
                teacher_peak_ratio={
                    k: t["teacher"][k]["regions"]["all"]["ratios_med_p95_max"][-1] for k in NAMES
                },
            )
        )
    record(out / "S3_GRIP_KINEMATICS.json", full_grip)
    s4 = dict(
        status="SEALED_STOP_NO_QUALIFYING_C3",
        independent_episodes_tested=0,
        development_episodes=len(set(w["episode"] for w in s0["windows"])),
        development_windows=16,
        reason="No documented transfer of ownership, and none of the C1 candidates satisfy threshold, nontrivial progress and event gates together. Fixed recipe retained for diagnosis.",
        source_cases=12,
        optimizer_updates=0,
        gpu_forwards=0,
        new_native_integrations=0,
        candidate_config_sha256=s0["registration"],
    )
    record(out / "S4_ACTUAL.json", s4)
    record(
        out / "S5_ACTUAL.json",
        dict(
            status="NOT_RUN",
            reason="S4 did not advance and no independent physical GO",
            requests=0,
            ticks=0,
            task_success_rate=None,
        ),
    )
    files = list(out.glob("*ACTUAL.json")) + [
        out / "CANONICAL_GATE.json",
        out / "S3_GRIP_KINEMATICS.json",
    ]
    durations = [r["duration_s"] for r in table]
    counts = dict(
        contact=sum(w["contact"] for w in table),
        grasp=sum(w["grasp"] for w in table),
        owner_grants=0,
        measured_previous_exceedance=sum(
            r["consumed_boundary_linear_accel"] > 0.25 or r["consumed_boundary_angular_accel"] > 1
            for r in table
        ),
        P10_motion=sum(w["motion"] for w in table),
        event_windows=sum(w["source_event_count"] > 0 for w in table),
        eventful_preserved=sum(w["source_event_count"] > 0 and w["events"] for w in table),
        duration_s=quant(durations),
        P10_travel_mm=quant([w["P10_mm"] for w in table]),
        coverage=quant([w["coverage"] for w in table]),
        canonical_peak_mm=max(w["canonical_delta_mm"] for w in table),
        teacher_original_pass=16,
    )
    strat = {
        name: {
            str(value): {
                "n": len(ss),
                "C1_h40_g7": sum(r["H40_G7"] for r in ss),
                "C1_motion": sum(r["motion"] for r in ss),
            }
            for value in sorted({r[name] for r in table})
            if (ss := [r for r in table if r[name] == value])
        }
        for name in ("seed", "phase")
    }
    result = dict(
        status=s4["status"],
        roster="16 development windows from 6 TRAIN episodes, plus 6 source expert and 6 source model controls",
        stages=groups,
        counts=counts,
        stratified=strat,
        windows=table,
        private_receipts_sha256={p.name: sha(p) for p in files},
        protocol=s0["protocol"],
        weights=s0["weights"],
        xpolicy=s0["xpolicy"],
        resource=dict(gpu_forwards=0, optimizer_updates=0, native_requests=0, new_integrations=0),
        canonical_attribution_exceptions=[
            {k: r[k] for k in ["id", "arm", "H40", "attribution_unavailable"]}
            for r in cg
            if "attribution_unavailable" in r
        ],
    )
    record(out / "V13_READOUT.json", result)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
