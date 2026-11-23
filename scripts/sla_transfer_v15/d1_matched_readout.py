"""Independently reconcile all fixed start slots, including pre-treatment exclusions."""

import argparse
from collections import Counter
from pathlib import Path

from d1_readiness import baseline, configure, pinned, read, record, rows, sha


def audit_slot(folder, reg, anchors, next_factor, executables):
    import hashlib
    import json

    import numpy as np
    from d1_factorial_reference_matrix import (
        boundary,
        check,
        fixed,
        held_part,
        initial_features,
        pose7,
        prior_pd,
        verify_initial,
    )
    from d2_policy_age_audit import read_case
    from go2arx5_vla.world_model.native_replay import packets
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    from scripts.pi0292.eval_contract import matrix10

    result_path = folder / "RESULT.private.json"
    result = read(result_path)
    epoch = folder / result["episode"]
    report = read(folder / "ENROLLMENT.json")
    pinned(result_path, report["result_sha256"])
    lease = read(folder / "control/LEASE.json")
    pinned(folder / "EXECUTION_SPEC.json", lease["spec_sha256"])
    check(
        lease["status"] == "CLOSED"
        and read(folder / "NATIVE_REQUEST.json")["epoch_id"] == epoch.name,
        "lease or request",
    )
    check(
        read(folder / "EXECUTION_SPEC.json")["d1_factor"] == next_factor,
        "prospective schedule",
    )
    out = {**report, "spec_sha256": lease["spec_sha256"], "attempted": 1}
    if not (epoch / "state.bin.journal").exists():
        check(
            report["status"] == "TECHNICAL_STOP" and not result.get("ticks"),
            "missing native journal",
        )
        return out
    states = packets(epoch / "state.bin.journal")
    check(
        len(states) == report["ticks"]
        and all(p["step"] == i for i, p in enumerate(states, 1)),
        "state counter",
    )
    out["test_ticks"] = max(0, len(states) - 5703)
    out["journal_sha256"] = sha(epoch / "state.bin.journal")
    if len(states) < 5703 or not (folder / "INITIAL_MATCH.private.json").exists():
        check(report["status"] == "TECHNICAL_STOP", "unclassified early stop")
        return out
    source = list(rows(epoch / "MOBILE_SAMPLES.private.jsonl"))
    sample = next(r for r in source if r["native_end_step"] == 5703)
    parts, old, seq = boundary(epoch, sample)
    features = initial_features(
        states[5702],
        parts,
        prior_pd(epoch / "init.trace"),
        sample["installed_EEF10"][9],
    )
    match = read(folder / "INITIAL_MATCH.private.json")
    check(
        features == match["features"] and old == match["prior_wire"], "boundary replay"
    )
    evaluation = verify_initial(features, anchors, reg["pairwise_initial_limits"])
    check(
        evaluation == match["evaluation"]
        and evaluation["matched"] == report["eligible"],
        "all pairwise bounds",
    )
    out["B2_anchor_violations"] = verify_initial(
        features, anchors[:2], reg["pairwise_initial_limits"]
    )["violations"]
    trace = result["trace"]
    out["native_STOP"] = trace["STOP_observed"]
    out["integrations_after_first_STOP"] = trace["integrations_after_first_STOP"]
    out["owner_exit_codes"] = [
        r["exit_code"] for r in read(folder / "CLOSE.private.json")["owned_exits"]
    ]
    loaded = read(epoch / "LOADED_EXECUTOR_00.private.json")
    check(
        all(loaded[k]["executable_sha256"] == v for k, v in executables.items()),
        "loaded backend mismatch",
    )
    proposals = [
        r
        for r in rows(epoch / "COMMANDS.private.jsonl")
        if r["phase"] == "proposed" and r["native_step"] >= 5703
    ]
    if not evaluation["matched"]:
        check(
            not proposals and not out["test_ticks"] and report["factor"] is None,
            "treatment before enrollment",
        )
        return out
    if report["status"] != "COMPLETE":
        return out
    check(
        report["factor"] == "D1-" + next_factor
        and len(proposals) == 10
        and out["test_ticks"] == 500,
        "allocation or exposure",
    )
    events = list(rows(epoch / "planner.jsonl"))
    action, base = fixed(next_factor, parts)
    ee = matrix10(action[0])
    expected = np.r_[pose7(ee), pose7(base[0])]
    atoms = [r for r in source if r["native_start_step"] >= 5703]
    check(len(atoms) == 20, "missing consumed interval")
    prev_clock = None
    for i, r in enumerate(proposals):
        wire = r["wire_sha256"]
        ack = [a for a in events if a["kind"] == "accept" and a.get("ref_id") == wire]
        publications = [
            a for a in events if a["kind"] == "installed" and a.get("ref_id") == wire
        ]
        check(
            len(ack) == 1
            and publications
            and hashlib.sha256(ack[0]["raw"].encode()).hexdigest() == wire,
            "wire hash/receipt",
        )
        doc = json.loads(ack[0]["raw"])
        meta = doc["_pi0294"]
        check(
            meta["episode"] == epoch.name
            and meta["sequence"] == seq + 1 + i
            and meta["ttl_s"] == 0.4
            and meta["operation"] == "install"
            and (
                prev_clock is None
                or abs(meta["source_mpc_s"] - prev_clock - 0.1) < 1e-8
            ),
            "clock or sequence",
        )
        prev_clock = meta["source_mpc_s"]
        check(
            np.allclose(held_part(doc, "ee"), ee, atol=1e-9, rtol=0)
            and np.allclose(held_part(doc, "base"), base[0], atol=1e-9, rtol=0),
            "wire geometry",
        )

        keys = {
            trajectory_identity(a["times"], a["poses"], a["inputs"])
            for a in publications
            if np.allclose(a["poses"], expected, atol=2e-7, rtol=0)
        }
        for j in range(2):
            atom = atoms[2 * i + j]
            check(
                atom["installed_reference_sha256"] == wire
                and keys.intersection(atom["solver_exact_installed_targets"])
                and atom["ticks"] == 25
                and atom["native_start_step"] == 5703 + 25 * (2 * i + j)
                and atom["safety_valid"]
                and not atom["native_grasp"],
                "native/solver echo join",
            )
    curve, ev = read_case(folder)
    check(
        curve == read(folder / "CURVE.private.json")
        and ev == read(folder / "RESPONSE.private.json"),
        "post-treatment replay differs",
    )
    for key, value in report["response"].items():
        check(value == ev[key], "reported response")
    out["milestones"] = ev["milestones"]
    out["orientation_rms_deg"] = ev["orientation_rms_deg"]
    out["mrt_selected_policy_id"] = ev["selected_MRT_policy_ID"]
    out["per_tick_curve_sha256"] = sha(folder / "CURVE.private.json")
    out["native_trace_sha256"] = sha(epoch / "init.trace")
    anchors.append(features)
    return out


def main():
    cli = argparse.ArgumentParser()
    for arg in ("root", "campaign", "output"):
        cli.add_argument(arg, type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    campaign = a.campaign.resolve()
    configure(baseline(root))
    from d1_factorial_reference_matrix import check
    from d1_native_readout import contrasts

    ready = root / "v15/readiness01"
    manifest = campaign / "REGISTRATION.json"
    pinned(manifest, "6e97c79b59581b0cbd2ca6673387f4af9417c927b14c70e3c85cd4185b9f2a9c")
    plan = read(manifest)
    original = ready / "D1_FACTORIAL_4X2_REGISTER.json"
    pinned(original, plan["old_register_sha256"])
    reg = read(original)
    terminal = read(campaign / "COHORT_ACTUAL.json")
    finished = terminal["attempted"]
    check(finished == len(list(campaign.glob("S*/NATIVE_REQUEST.json"))), "attempts")
    rr = read(ready / "D1_READINESS_ACTUAL.json")
    anchors = []
    for anchor in rr["anchors"]:
        p = ready / (anchor["source"] + "_anchor.private.json")
        pinned(p, anchor["private_anchor_sha256"])
        anchors.append(read(p)["initial_features"])
    schedule = [f for b in plan["treatment_sequence_by_eligible_block"] for f in b]
    records = []
    complete = []
    for i in range(plan["fixed_start_slots"]):
        slot = f"S{i + 1:02}"
        if i >= finished:
            records.append(
                {
                    "slot": slot,
                    "status": "NOT_RUN",
                    "factor": None,
                    "attempted": 0,
                    "ticks": 0,
                    "test_ticks": 0,
                }
            )
            continue
        row = audit_slot(
            campaign / slot, reg, anchors, schedule[len(complete)], rr["frozen_sha256"]
        )
        records.append(row)
        check(
            all(row[k] == v for k, v in terminal["outcomes"][i].items()),
            "owner readout mismatch",
        )
        if row["status"] == "COMPLETE":
            complete.append(row)
    effects = [contrasts(complete[i : i + 4]) for i in range(0, len(complete) - 3, 4)]
    check(
        terminal["complete_blocks"] == effects
        and terminal["native_ticks"] == sum(r["ticks"] or 0 for r in records),
        "totals",
    )
    record(
        a.output,
        {
            "terminal": terminal["terminal"],
            "registration_sha256": sha(manifest),
            "starts": finished,
            "registered_starts": len(records),
            "status_counts": dict(Counter(r["status"] for r in records)),
            "native_ticks": terminal["native_ticks"],
            "test_ticks": sum(r.get("test_ticks", 0) for r in records),
            "blocks": effects,
            "records": records,
            "new_SLA_forwards": 0,
            "training_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
