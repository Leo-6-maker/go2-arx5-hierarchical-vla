"""Count every registered cell and audit closed native outcomes and initialization divergence."""

import argparse
from pathlib import Path

from d1_readiness import (
    B2_PIN,
    D0_PIN,
    baseline,
    configure,
    pinned,
    read,
    record,
    rows,
    sha,
)


def inspect_prefix(paths):
    import numpy as np
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from go2arx5_vla.world_model.native_replay import packets

    sets = []
    facts = []
    for folder in paths:
        result = read(folder / "RESULT.private.json")
        out = folder / result["episode"]
        p = packets(out / "state.bin.journal")
        z = next(rows(out / "initialization.jsonl"))
        sets.append((z, p))
        trace = read_trace(out / "init.trace")
        tw = {r["token"]: r["monotonic_ns"] for r in trace if r["kind"] == 12}

        def first_request(k, out=out, tw=tw):
            for r in rows(out / "ROS_CONSUMPTION.private.jsonl"):
                if r["topic_key"] == k:
                    m = r["message"]
                    s = (
                        m["init_observation"]["time"]
                        if k == "policy"
                        else m["header"]["stamp"]["sec"]
                        + m["header"]["stamp"]["nanosec"] * 1e-9
                    )
                    return {
                        "source_s": s,
                        "last_completed_tick_at_observer": max(
                            (
                                t
                                for t, ns in tw.items()
                                if ns <= r["received_monotonic_s"] * 1e9
                            ),
                            default=0,
                        ),
                    }

        facts.append(
            {
                "case": folder.name,
                "startup_sha256": sha(out / "initialization.jsonl"),
                "native_state_sha256": sha(out / "state.bin.journal"),
                "clock_grants_sha256": sha(out / "CLOCK_RESUME_REQUESTS.jsonl"),
                "init_trace_sha256": sha(out / "init.trace"),
                "first_arm_request": first_request("arm_request"),
                "first_solver_return": first_request("policy"),
            }
        )
    comparisons = []
    for (z, p), f in zip(sets[:-1], facts[:-1], strict=True):
        a, b = sets[-1]

        def distance(i, p=p, b=b):
            q = p[i - 1]["qpos"]
            other = b[i - 1]["qpos"]
            return {
                "tick": i,
                "base_translation_mm": float(np.linalg.norm(q[:3] - other[:3]) * 1000),
                "arm_max_joint_difference_deg": float(
                    np.degrees(max(abs(q[19:25] - other[19:25])))
                ),
                "leg_max_joint_difference_deg": float(
                    np.degrees(max(abs(q[7:19] - other[7:19])))
                ),
            }

        comparisons.append(
            {
                "versus": f["case"],
                "same_initial_qpos": z["qpos"] == a["qpos"],
                "same_initial_qvel": z.get("qvel") == a.get("qvel")
                if "qvel" in a and "qvel" in z
                else None,
                "tick1_qpos_equal": np.array_equal(p[0]["qpos"], b[0]["qpos"]),
                "tick1_qvel_equal": np.array_equal(p[0]["qvel"], b[0]["qvel"]),
                "milestones": [
                    distance(i)
                    for i in (1, 2500, 2503, 2528, 2703, 3203, 4103, 5203, 5703)
                ],
            }
        )
    return {"cases": facts, "comparisons": comparisons}


def contrasts(records):
    if len(records) != 4 or any(r.get("status") != "COMPLETE" for r in records):
        return None
    result = {}
    for key in ("Y_actual_z_100ms_mm", "Y_PD_z_100ms_mm"):
        v = {r["factor"]: r["response"][key] for r in records}
        oo, mo, om, mm = [v[k] for k in ("D1-OO", "D1-MO", "D1-OM", "D1-MM")]
        result[key] = {
            "EEF_main": 0.5 * ((mo - oo) + (mm - om)),
            "base_main": 0.5 * ((om - oo) + (mm - mo)),
            "interaction": mm - mo - om + oo,
        }
    return result


def main():
    cli = argparse.ArgumentParser()
    for k in ("root", "ready", "campaign", "out"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    campaign = args.campaign.resolve()
    ready = args.ready.resolve()
    out = args.out.resolve()
    configure(baseline(root))
    from d1_factorial_reference_matrix import check
    from d2_policy_age_audit import read_case
    from go2arx5_vla.world_model.native_replay import packets

    out.mkdir(exist_ok=False, parents=True)
    b2root = root / "v14/b2-technical-20261123-01"
    pinned(b2root / "B2_PUBLIC_READOUT.json", B2_PIN)
    pinned(root / "v14/B2_STATIC_BIAS_ATTRIBUTION.json", D0_PIN)
    public = read(b2root / "B2_PUBLIC_READOUT.json")
    reg = read(ready / "D1_FACTORIAL_4X2_REGISTER.json")
    b2rows = []
    results = []
    d1complete = []
    source = []
    for c in public["cards"]:
        folder = b2root / c["card"]
        source.append(folder)
        pinned(folder / "RESULT.private.json", c["result_sha256"])
        native = folder / read(folder / "RESULT.private.json")["episode"]
        for name, pin in c["evidence_sha256"].items():
            pinned(native / name, pin)
        curve, ev = read_case(folder)
        record(out / (folder.name + "_curve.private.json"), curve)
        ev["private_curve_sha256"] = sha(out / (folder.name + "_curve.private.json"))
        b2rows.append(dict(case=folder.name, **ev))
    for wave in ("wave_A", "wave_B"):
        records = []
        for cell in (c for c in reg["roster"] if c["wave"] == wave):
            folder = campaign / wave / cell["id"]
            r = {
                "id": cell["id"],
                "factor": cell["factor"],
                "status": "NOT_RUN",
                "requests": 0,
                "ticks": 0,
                "test_ticks": 0,
                "installs": 0,
            }
            if (folder / "NATIVE_REQUEST.json").exists():
                val = read(folder / "RESULT.private.json")
                spec = read(folder / "EXECUTION_SPEC.json")
                lease = read(folder / "control/LEASE.json")
                native = folder / val["episode"]
                check(
                    lease["status"] == "CLOSED"
                    and lease["spec_sha256"] == sha(folder / "EXECUTION_SPEC.json"),
                    "closed owner/spec mismatch",
                )
                ps = packets(native / "state.bin.journal")
                tick = len(ps)
                pa = read(native / "pause_ack.json")
                check(
                    tick
                    == val["ticks"]
                    == val["trace"]["native_step_records"]
                    == pa["step"]
                    and pa["clock_paused"]
                    and not pa["running"],
                    "independent native counters",
                )
                afterrows = [
                    row
                    for row in rows(native / "MOBILE_SAMPLES.private.jsonl")
                    if row["native_start_step"] >= 5703
                ]
                issued = [
                    row
                    for row in rows(native / "COMMANDS.private.jsonl")
                    if row["phase"] == "proposed" and row["native_step"] >= 5703
                ]
                ids = {rr["wire_sha256"] for rr in issued}
                accepted = [
                    rr
                    for rr in rows(native / "planner.jsonl")
                    if rr["kind"] == "accept" and rr.get("ref_id") in ids
                ]
                installed_wire_ids = {
                    event["ref_id"]
                    for event in rows(native / "planner.jsonl")
                    if event["kind"] == "installed" and event.get("ref_id") in ids
                }
                loaded = read(native / "LOADED_EXECUTOR_00.private.json")
                check(
                    all(
                        loaded[k]["executable_sha256"] == v
                        for k, v in public["frozen_sha256"].items()
                    ),
                    "backend differs",
                )
                match_path = folder / "INITIAL_MATCH.private.json"
                match = read(match_path)["evaluation"] if match_path.exists() else None
                r.update(
                    status="COMPLETE"
                    if val["completed"]
                    else "INITIAL_STATE_MISMATCH"
                    if match and not match["matched"]
                    else "TECHNICAL_OR_SAFETY_STOP",
                    requests=1,
                    ticks=tick,
                    test_ticks=max(0, tick - 5703),
                    issued=len(issued),
                    accepted=len(accepted),
                    installs=len(installed_wire_ids),
                    consumed_intervals=len(afterrows),
                    native_STOP=val["trace"]["STOP_observed"],
                    post_native_STOP_ticks=val["trace"][
                        "integrations_after_first_STOP"
                    ],
                    native_lease_closed=True,
                    owned_exit_codes={
                        x["name"]: x["exit_code"]
                        for x in read(folder / "CLOSE.private.json")["owned_exits"]
                    },
                    guard_stop=val["guard_stop"],
                    matching=match,
                    stage_sha256=spec["stage_identity_sha256"],
                    case_spec_sha256=sha(folder / "EXECUTION_SPEC.json"),
                    result_sha256=sha(folder / "RESULT.private.json"),
                    input_sha256={
                        n: sha(native / n)
                        for n in (
                            "MOBILE_SAMPLES.private.jsonl",
                            "CLOCK_RESUME_REQUESTS.jsonl",
                            "planner.jsonl",
                            "state.bin.journal",
                            "init.trace",
                            "pause_ack.json",
                            "physics.jsonl",
                        )
                    },
                )
                if r["status"] == "INITIAL_STATE_MISMATCH":
                    r["initialization_diagnostics"] = inspect_prefix([*source, folder])
                if val["completed"]:
                    curve, ev = read_case(folder)
                    record(out / (cell["id"] + "_curve.private.json"), curve)
                    d1complete.append(dict(case=cell["id"], **ev))
                    r["response"] = {
                        k: ev[k]
                        for k in (
                            "Y_actual_z_100ms_mm",
                            "Y_PD_z_100ms_mm",
                            "absolute_position_rms_mm",
                            "incremental_position_rms_mm",
                            "applied_PD_q_without_matching_known_request",
                        )
                    }
            records.append(r)
        effects = contrasts(records)
        result = {
            "wave": wave,
            "registered": len(records),
            "attempted": sum(r["requests"] for r in records),
            "completed": sum(r["status"] == "COMPLETE" for r in records),
            "integrations": sum(r["ticks"] for r in records),
            "test_ticks": sum(r["test_ticks"] for r in records),
            "records": records,
            "descriptive_contrasts_mm": effects,
            "promotion_ready": bool(
                effects is not None
                and all(
                    r["response"]["applied_PD_q_without_matching_known_request"] == 0
                    for r in records
                )
            ),
            "new_gpu_forwards": 0,
            "new_optimizer_updates": 0,
        }
        results.append(result)
        record(out / ("D1_" + wave.upper() + "_ACTUAL.json"), result)
        if (campaign / wave).exists() and not (
            campaign / wave / "WAVE_READOUT.private.json"
        ).exists():
            record(campaign / wave / "WAVE_READOUT.private.json", result)
    possible = len([r for r in results if r["descriptive_contrasts_mm"] is not None])
    record(
        out / "D1_MAIN_EFFECTS_AND_INTERACTIONS.json",
        {
            "status": "ESTIMABLE" if possible else "NOT_ESTIMABLE",
            "registration_sha256": sha(ready / "D1_FACTORIAL_4X2_REGISTER.json"),
            "planned_cells": 8,
            "registered_waves": 2,
            "eligible_complete_waves": possible,
            "waves": [r["descriptive_contrasts_mm"] for r in results],
            "historical_MM_imputation": False,
            "new_requests": sum(r["attempted"] for r in results),
            "new_ticks": sum(r["integrations"] for r in results),
            "full_two_wave_effects": None
            if possible < 2
            else "see paired descriptive wave values",
        },
    )
    record(
        out / "D2_LOW_LEVEL_TRACKING_AND_POLICY_AGE.json",
        {
            "source": "closed native records only",
            "historical_B2_cases": b2rows,
            "D1_completed_cases": d1complete,
            "D1_four_cell_low_level_comparison": "UNAVAILABLE"
            if not possible
            else "see closed cases",
            "selected_policy": "POLICY_ID_UNOBSERVED",
            "new_native_requests": 0,
            "new_ticks": 0,
            "new_gpu_forwards": 0,
            "new_training_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
