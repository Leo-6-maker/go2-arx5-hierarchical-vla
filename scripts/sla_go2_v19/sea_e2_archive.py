"""Compare observation opportunities using closed #494 consumption, with no fresh installs."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sea2_native_profile_preflight import CARDS
from sea2_native_score import ATOM, STEP, endpoints, publication_poses, read_capture
from source_custody import BASE, digest, pin, read, require, write_new
from demonstrated_grip import rows
from policy_query import tensor_sha

SEA2 = BASE / "reports/SLA_GO2_V191/SEA2/ACTUAL.json"
SEA1A = BASE / "reports/SLA_GO2_V191/SEA1A/ACTUAL.json"
ARCHIVE = BASE / "reports/SLA_GO2_V19/SEA2/V16_CONTINUOUS.json"
CAMPAIGN = "SEA2-NATIVE-KNOWN-PHYSICAL-01"
PUB = 2 * ATOM
TTL_TICKS = round(.4 / STEP)


def publish_after(tick, origin):
    return origin + ((tick-origin + PUB-1)//PUB) * PUB


def owner_inputs(panel, signed):
    require(signed["lease_closed"] and signed["native_processes_absent"], "case not closed")
    case = panel / signed["slot"]
    seals = signed["seals"]
    report = read(pin(case / "AUDIT.json", seals["scientific_audit"]))
    result = read(pin(case / "RESULT.private.json", seals["private_receiver_evidence"]))
    spec = read(pin(case / "EXECUTION_SPEC.json", seals["execution"]))
    lease = read(pin(case / "control/LEASE.json", seals["lease"]))
    require(report["status"] == "PROFILE_PASS" and lease["status"] == "CLOSED"
            and result["absent"] and result["cancel"]["mode"] == "HOLD",
            "unexpected result or live historical owner")
    return case, report, result, spec


def rolling_observation(start, end, required, motion, duration, ledger, installed, atoms):
    if not required or motion is None:
        return dict(status="NO_ELIGIBLE_SOURCE_MOTION")
    require(start < motion["tick"] < end and (motion["translation"] or motion["rotation"]),
            "source progress must be real and inside the native interval")
    # At a publication boundary take a new observation, but let the next 100 ms
    # of the already installed plan continue. No hypothetical return is installed.
    earliest = publish_after(motion["tick"], start)
    legacy = publish_after(max(motion["tick"], start+int(round(duration/STEP))), start)
    protected_end = earliest+PUB
    require(earliest in ledger and legacy in ledger and protected_end <= end, "no recorded protected prefix")
    prev = ledger[earliest-PUB]
    consumed = atoms[earliest]
    require(consumed["installed_reference_sha256"] == prev["wire"]
            and consumed["solver_exact_installed_targets"] and consumed["native_end_step"] == earliest,
            "latest acknowledged publication has no proven consumption")
    require(0 <= earliest-prev["tick"] < TTL_TICKS, "active publication expired before observation")
    frozen = installed[prev["wire"]][2:5]
    witnessed = np.stack([endpoints(atoms[t])[1] for t in (earliest, earliest+ATOM, protected_end)])
    require(protected_end - prev["tick"] < TTL_TICKS, "protected prefix reaches the original expiry")
    require(np.allclose(frozen, witnessed, rtol=0, atol=2e-7),
            "lookahead from active installation does not match the protective prefix")
    v = (witnessed[1,:3,3]-witnessed[0,:3,3])/(ATOM*STEP)
    d_r = witnessed[1,:3,:3] @ witnessed[0,:3,:3].T
    w = Rotation.from_matrix(d_r).as_rotvec()/(ATOM*STEP)
    return dict(
        status="ARCHIVAL_EARLY_OBSERVATION_WITH_VERIFIED_PREFIX",
        first_signed_source_tick=motion["tick"], observation_tick=earliest,
        old_query_eligible_tick_unlimited=legacy, protected_prefix_end_tick=protected_end,
        protected_prefix_sha256=tensor_sha(frozen), preceding_consumed_wire=prev["wire"],
        age_of_active_wire_at_prefix_end_s=(protected_end-prev["tick"])*STEP,
        exact_protected_prefix=True, actual_new_queries=0, actual_new_installs=0,
        proposed_observation_delay_s=(earliest-start)*STEP, legacy_observation_delay_s=(legacy-start)*STEP,
        estimated_observation_delay_reduction=(legacy-earliest)/(legacy-start),
        protected_next_linear_speed_mm_s=float(1000*np.linalg.norm(v)),
        protected_next_angular_speed_rad_s=float(np.linalg.norm(w)),
        actual_contact_or_model_feedback_qualified=False,
    )


def refusal_checks(start, end, motion, duration, ledger, installed, atoms, estimated):
    if estimated["status"] == "NO_ELIGIBLE_SOURCE_MOTION":
        return {"stationary_or_missing_motion_refuses": True}
    at = estimated["observation_tick"]
    prev = at-PUB
    wire = ledger[prev]["wire"]
    faulty_atoms = {**atoms, at: {**atoms[at], "solver_exact_installed_targets": False}}
    faulty_install = {**installed, wire: installed[wire].copy()}
    faulty_install[wire][3, 0, 3] += .0001
    old_ledger = {**ledger, prev: {**ledger[prev], "tick": at-TTL_TICKS}}
    scenarios = {
        "no_measured_progress": (end, {**motion, "translation": False, "rotation": False}, ledger, installed, atoms),
        "ack_only": (end, motion, ledger, installed, faulty_atoms),
        "changed_protected_prefix": (end, motion, ledger, faulty_install, atoms),
        "expired_reference": (end, motion, old_ledger, installed, atoms),
        "protected_prefix_would_cross_terminal_tick": (at+ATOM, motion, ledger, installed, atoms),
    }
    refused = {}
    for name, (terminal, evidence, wire_rows, wire_poses, consumed_rows) in scenarios.items():
        try:
            rolling_observation(start, terminal, True, evidence, duration, wire_rows, wire_poses, consumed_rows)
        except ValueError as exc:
            refused[name] = str(exc)
        else:
            raise AssertionError(f"{name} incorrectly admitted")
    return refused


def card(root, signed, dimensions):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration

    case, report, result, spec = owner_inputs(root, signed)
    native = case / result["episode"]
    with forbid_integration(mujoco) as invocations:
        start, actual, consumed, _, _, clearance = read_capture(case, result, spec)
    require(not invocations and len(actual) == 4501 and len(consumed) == 181
            and abs(clearance-report["actual_clearance_mm"]) < 1e-7,
            "closed 2 ms capture did not reproduce")
    end = result["end"]
    ledger = {w["tick"]: w for w in rows(case / "REACH_WIRES.jsonl") if start <= w["tick"] < end}
    installed = publication_poses(rows(native / "planner.jsonl"), list(ledger.values()), result["episode"])
    atoms = {a["native_end_step"]: a for a in rows(native / "MOBILE_SAMPLES.private.jsonl")
             if start < a["native_end_step"] <= end}
    logs = rows(case / "SEA_REFERENCE.jsonl")
    returned = [q for q in logs if q["kind"] == "RETURN_ADMITTED"]
    source = report["source_motion"]
    require(len(returned) == 1 and result["student"]["accepted_returns"] == 1
            and result["student"]["source_motion"] == source, "unexpected additional historical query")
    duration = returned[0]["plan_end"]
    motion = rolling_observation(start, end, dimensions, source, duration, ledger, installed, atoms)
    controls = refusal_checks(start, end, source, duration, ledger, installed, atoms, motion)
    return dict(
        slot=signed["slot"], verified_2ms_samples=len(actual), exact_consumed_50ms_atoms=len(atoms),
        installed_receipts=len(installed), clock_start=start, selected_MRT_ref=None,
        actual_terminal_tick=end, actual_after_terminal_extra_integrations=result["end"]-signed["native_ticks"],
        original_actual_queries=len(returned), original_query_cap=1,
        original_plan_lead_s=returned[0]["plan_lead"], original_plan_end_s=returned[0]["plan_end"],
        estimates=motion, verified_refusals=controls,
    )


def run(root, output):
    panel = root / "v19" / CAMPAIGN
    official = read(SEA2)
    close = read(pin(panel / "PANEL_CLOSE.json", official["evidence"]["panel_close_sha256"]))
    require(close["observed_integrations"] == 42812 and len(close["results"]) == 4
            and not close["unstarted"] and not close["unresolved_requested"], "wrong closed native cohort")
    known = [card(panel, entry, bool(any(xyz) or any(angles)))
             for entry, (name, _, xyz, angles) in zip(official["cards"], CARDS, strict=True) if entry["slot"] == name]
    require(len(known) == 4, "native card order changed")
    archive = read(ARCHIVE)
    p = read(SEA1A)
    require(len(archive["windows"]) == 19 and len(p["full_pose_results"]) == 16, "fixed offline evidence changed")
    estimates = [r["estimates"]["estimated_observation_delay_reduction"] for r in known
                 if "estimated_observation_delay_reduction" in r["estimates"]]
    write_new(output, dict(
        status="OBSERVATION_TIMING_PROOF_ONLY_NO_REPLACEMENT_QUALIFICATION",
        baseline_cards=known, historical_ticks=42812, median_estimated_observation_reduction=float(np.median(estimates)),
        archived_19_windows_fitted=[dict(id=w["id"], source_s=w["prefix_source_s"], legacy_full_s=w["previous_clock"]["absolute_full_s"],
                                        compiled_s=w["lead_s"]+w["duration_s"], runtime_consumption_verified=False)
                                    for w in archive["windows"]],
        sea1a_pair_count=p["scored_pairs"], evidence_sha256=dict(native=digest(SEA2), archived=digest(ARCHIVE), sea1a=digest(SEA1A)),
        model_forwards=0, native_starts=0, physics_ticks=0, reference_installs=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU archive audit only")
    run(args.root.resolve(), args.output.resolve())
