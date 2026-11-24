"""Cold acceptance for the SEA2 site owner. Every dynamic trace is archival or synthetic."""

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sea2_native_profile_preflight import CARDS, archive_sample, calibration_student, known_intent
import sea2_native_score as scoring
from sea2_native_score import (
    DT, STEP, consume_poses, fixed_clock_score, interpolate, participation,
    publication_poses, read_capture,
)
from sea2_preflight import sealed_archive
from sea_reference import ReferenceSegment
from demonstrated_grip import read, require, rows
from go2arx5_vla.world_model.native_replay import forbid_integration
from reach_pair import paired_audit
import sea2_native_owner as owner


def rejects(operation, text):
    try:
        operation()
    except ValueError as exc:
        require(str(exc) == text, f"unexpected refusal: {exc}")
    else:
        raise AssertionError("expected fail-closed refusal")
    return True


def archive_readback(root, sealed, layout):
    prefix = "Z1_REACH_483" if layout == "G00" else "Z1_REACH_483_G01"
    case = root / "v18" / prefix / (layout + "-H0")
    result = read(sealed(case / "RESULT.private.json"))
    spec = read(sealed(case / "EXECUTION_SPEC.json"))
    native = case / result["episode"]
    for path in (
        native / "state.bin.journal", native / "MOBILE_SAMPLES.private.jsonl",
        native / "planner.jsonl", native / "init.trace", native / "LOADED_EXECUTOR_00.private.json",
        case / "REACH_ENTRY.json", case / "REACH_WIRES.jsonl",
    ):
        sealed(path)
    old = paired_audit(case, result, spec)
    start, actual, knots, _, _, clearance = read_capture(case, result, spec)
    issued = interpolate(knots)
    require(old["qualified"] and len(issued) == len(actual) == 2251,
            "previous H0 no longer verifies")

    ledger = [r for r in rows(case / "REACH_WIRES.jsonl") if start <= r["tick"] < result["end"]]
    events = list(rows(native / "planner.jsonl"))
    installed = publication_poses(events, ledger, result["episode"])
    atoms = [a for a in rows(native / "MOBILE_SAMPLES.private.jsonl")
             if start <= a["native_start_step"] < result["end"]]
    tampered = [dict(atoms[0], solver_exact_installed_targets=[]), *atoms[1:]]
    refused_atom = rejects(lambda: consume_poses(tampered, ledger, installed, start, result["end"]),
                           "unverified consumed atom")
    bad = [dict(row, raw=row["raw"] + " ") if row.get("kind") == "accept"
           and row.get("ref_id") == ledger[0]["wire"] else row for row in events]
    refused_wire = rejects(lambda: publication_poses(bad, ledger, result["episode"]),
                           "accepted wire hash changed")
    return dict(
        slot=layout+"-H0", entry_tick=start, end_tick=result["end"],
        measured_2ms_pose_count=len(actual), consumed_50ms_atoms=len(knots)-1,
        minimum_clearance_mm=clearance, archived_h0_qualification=old["qualified"],
        bad_wire_refused=refused_wire, unverified_atom_refused=refused_atom,
        new_sea_profile_observed=False,
    )


def score_callback(card, limits, intent, actual, knots, receipt, credited):
    first = min((x["tick"] for x in credited.values() if x is not None), default=None)
    reject_expected = "physical intent did not survive exact byte readback"
    with tempfile.TemporaryDirectory() as folder:
        case = Path(folder)
        evidence = case / "SEA_REFERENCE.jsonl"
        source = dict(
            tick=intent.tick, input=intent.observation_sha256,
            actual_tcp=intent.actual_tcp.tolist(),
        )
        result = dict(
            episode=intent.episode, outcome="A_REACH_FAILED",
            student=dict(accepted_returns=1, source_motion={"tick": first} if first is not None else None),
        )
        measured = (intent.tick, actual, knots, receipt["start_pose"], receipt["end_pose"], 99.)
        with patch.object(scoring, "paired_audit", return_value={"ticks": intent.tick+4500}), \
                patch.object(scoring, "read_capture", return_value=measured):
            def callback():
                return scoring.score_case(case, result, None, card, limits)

            def write_log(raw):
                evidence.write_text(
                    json.dumps(dict(source, kind="KNOWN_PHYSICAL_INPUT")) + "\n"
                    + json.dumps(dict(source, kind="RETURN_ADMITTED", input=raw)) + "\n"
                )

            write_log(intent.observation_sha256)
            score = callback()
            write_log("0"*64)
            refused = rejects(callback, reject_expected)
    return dict(status=score["status"], wrong_intent_hash_refused=refused)


def score_controls(root, sealed, limits):
    controls = []
    for card in CARDS:
        name, layout, position, euler = card
        _, native, tick, _, receipt, sample = archive_sample(root, sealed, layout)
        intent = known_intent(native.name, tick, sample["tcp"], position, euler)
        segment = ReferenceSegment(intent.curve, sample["tcp"], receipt["end_pose"], receipt["start_pose"], limits)
        knots = segment.at(np.arange(181)*DT)[0]
        issued = interpolate(knots)
        clocks = np.arange(len(issued))*STEP
        phase = segment.at(clocks)[2]
        ideal = intent.curve.at(30*phase)
        anchor = ideal[0]
        parked = np.repeat(anchor[None], len(ideal), axis=0)
        reverse = ideal.copy()
        reverse[:, :3, 3] = 2*anchor[:3, 3] - ideal[:, :3, 3]
        r0 = anchor[:3, :3]
        reverse[:, :3, :3] = r0 @ ideal[:, :3, :3].transpose(0,2,1) @ r0
        scores = {
            label: fixed_clock_score(actual, issued, segment, hold=name=="SEA2-HOLD")
            for label, actual in (("same_direction", ideal), ("stationary", parked), ("opposed", reverse))
        }
        credited = {
            label: participation(actual, knots, segment, tick)
            for label, actual in (("same_direction", ideal), ("stationary", parked), ("opposed", reverse))
        }
        ideal_channels = [c for c, v in credited["same_direction"].items() if v is not None]
        expect_channels = {
            "SEA2-HOLD": [], "SEA2-XYZ": ["translation"],
            "SEA2-SO3": ["rotation"], "SEA2-MIXED": ["translation", "rotation"],
        }[name]
        require(ideal_channels == expect_channels
                and scores["same_direction"]["achieved_to_intent"]["in_tube"] == 1
                and all(v is None for variant in ("stationary", "opposed") for v in credited[variant].values()),
                "dynamic scorer accepted a false motion control")
        shapes = scores["same_direction"]["shapes_to_intent"]
        for axes in shapes.values():
            for value in axes.values():
                if value is not None:
                    require(abs(value["signed_gain"]-1) < 1e-10
                            and abs(value["relative_shape_score"]-1) < 1e-10, "shape scoring drift")
        positive_callback = score_callback(card, limits, intent, ideal, knots, receipt,
                                           credited["same_direction"])
        parked_callback = score_callback(card, limits, intent, parked, knots, receipt,
                                         credited["stationary"])
        require(positive_callback["status"] == "PROFILE_PASS"
                and parked_callback["status"] == ("PROFILE_PASS" if name == "SEA2-HOLD" else "PROFILE_FAIL"),
                "scoring callback used legacy task outcome or credited a stationary dynamic case")
        controls.append(dict(slot=name, credited_channels=ideal_channels,
                             scores=scores, signed_motion=credited,
                             callback_ideal=positive_callback, callback_stationary=parked_callback))
    return controls


def allocation_guard(root, shared):
    # Deliberately malformed and expired local fixtures; never enter the live callback.
    proof = dict(shared, cold_proof_sha256="cpu-fixture")
    fields = dict(
        id=owner.CAMPAIGN, candidate_sha256=proof["candidate_sha256"],
        cold_proof_sha256=proof["cold_proof_sha256"], authorization_id="CPU_FIXTURE",
        physics_limits=proof["physics_limits"], slots=owner.ORDER,
    )
    reason = "independent matching SEA2 allocation is missing or expired"
    denied = rejects(lambda: owner.verify_allocation(
        dict(fields, issued=False, valid_until_epoch=time.time()+60), proof), reason)
    expired = rejects(lambda: owner.verify_allocation(
        dict(fields, issued=True, valid_until_epoch=time.time()-1), proof), reason)
    wrong = rejects(lambda: owner.verify_allocation(
        dict(fields, issued=True, valid_until_epoch=time.time()+60, slots=owner.ORDER[:-1]), proof), reason)
    with tempfile.TemporaryDirectory() as folder, patch.object(owner, "execute_panel") as execute:
        base = Path(folder)
        auth = base / "allocation.json"
        cold = base / "cold.json"
        owner.record(auth, dict(fields, issued=False, valid_until_epoch=time.time()+60))
        owner.record(cold, {"status": "CPU_OWNER_BINDING_VERIFIED"})
        args = SimpleNamespace(root=root, cold=None, execute=auth, allocation_sha256=owner.sha(auth),
                               preflight=cold)
        denied_entry = rejects(lambda: owner.main(args), reason)
        require(not execute.called, "unallocated CLI reached the live callback")
    return dict(unissued_refused=denied, expired_refused=expired, wrong_roster_refused=wrong,
                main_cli_refused_before_live_callback=denied_entry)


def refusal_routing(limits):
    result = {"outcome": "A_REFERENCE_REFUSED"}
    readings = {}
    for reason in ("UNVERIFIED_REFERENCE", "WRONG_CLOCK", "CONTACT_OR_STOP",
                   "COMPILED_PREFIX_EXCEEDS_REGISTERED_WINDOW"):
        physical = {"ticks": 6203, "refusal": {"reason": reason}}
        with patch.object(scoring, "paired_audit", return_value=physical):
            scored = scoring.score_case(None, result, None, CARDS[0], limits)
        terminal = reason != "COMPILED_PREFIX_EXCEEDS_REGISTERED_WINDOW"
        require(scored["completed"] == (not terminal), "panel refusal did not respect terminal scope")
        readings[reason] = scored["status"]
    return readings


def late_window(root, sealed, limits):
    _, native, _, _, receipt, sample = archive_sample(root, sealed, "G00")
    student = calibration_student(CARDS[0], limits)
    tick = 15503
    expected = "REGISTERED_WINDOW_EXCEEDS_HARD_TICK_CAP"
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / native.name
        adapter = SimpleNamespace(output=output)
        require(rejects(
            lambda: student.preview(adapter, tick, receipt, sample, None, None, None), expected
        ) and student.calls == 0 and student.path == output.parent / "SEA_REFERENCE.jsonl",
                "late start was not a safely reportable refusal")
    return expected


def cold_check(root, prepared):
    _, _, _, _, _, limits = prepared
    sealed = sealed_archive(root)
    with forbid_integration(mujoco) as steps:
        shared = owner.binding(root, prepared)
        archive = [archive_readback(root, sealed, layout) for layout in ("G00", "G01")]
        controls = score_controls(root, sealed, limits)
        guard = allocation_guard(root, shared)
        late = late_window(root, sealed, limits)
        refusal = refusal_routing(limits)
    require(not steps, "cold owner acceptance entered dynamics")
    return dict(
        status="CPU_OWNER_BINDING_VERIFIED", binding=shared, archival_readbacks=archive,
        synthetic_controls=controls, allocation_guard=guard, late_window_refusal=late,
        refusal_routing=refusal,
        new_native_starts=0, physics_ticks=0, gpu_forwards=0, updates=0,
    )
