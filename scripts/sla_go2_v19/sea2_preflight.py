"""Replay SEA references through the same no-contact admission used by the standing receiver."""

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sea_reference import (
    BASE, DT, OPEN, PhysicalIntent, ReferenceSegment, SEAStandingStudent, admit_intent,
    receiver_clock_limits,
)
from compiler import SourceCurve
from d1_geometry import sweep_profile
from known_dynamic_panel import KnownCurve
from demonstrated_grip import pin, read, require, rows
from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
from go2arx5_vla.world_model.multitask_state_v6 import read_journal_packet
from go2arx5_vla.world_model.native_replay import forbid_integration
from reach_interface_audit import replay as previous_clock
from reach_intent_replay import history, standing
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.eval_contract import matrix10
from standing_reach import StateMonitor
from visual_lift_panel import write_new
from world_intent_clock import world_offset


def step_ids(at):
    previous = matrix10(np.asarray(at["installed_EEF10"]))
    consumed = matrix10(np.asarray(at["target_endpoint_EEF10"]))
    return previous, consumed, np.asarray(at["actual_TCP"])


def frozen_model(spec):
    scene = Path(spec["model_path"])
    return mujoco.MjModel.from_binary_path(str(pin(scene, spec["frozen_files"][str(scene)])))


def archive_boundary(tick, at, native, spec, sealed):
    previous, consumed, actual = step_ids(at)
    wire = at["installed_reference_sha256"]
    trigger = at["gripper_command"]["EEF_plan_source_s"]
    require(
        at["native_end_step"] == tick and at["native_start_step"] == tick-25
        and at["ticks"] == 25 and at["solver_exact_installed_targets"] and at["safety_valid"]
        and not at["native_grasp"]
        and at["installed_EEF10"][9] == at["target_endpoint_EEF10"][9] == OPEN
        and abs(at["gripper_command"]["source_native_s"]-(tick-25)*.002) < 1e-8,
        "NO_VERIFIED_OPEN_ARCHIVE_BOUNDARY",
    )
    commands = {
        r["phase"]: r for r in rows(sealed(native / "COMMANDS.private.jsonl"))
        if r.get("wire_sha256") == wire
    }
    accepted, installed = commands["accepted"]["evidence"], commands["installed"]["evidence"]
    require(
        any(e["kind"] == "accept" and e["ref_id"] == wire
            and abs(e["source_mpc_s"]-trigger) < 1e-8 for e in accepted)
        and any(e["kind"] == "installed" and e["ref_id"] == wire
                and len(e["times"]) == 41 and abs(e["bound_start"]-trigger) < 1e-8
                for e in installed),
        "ARCHIVE_WIRE_DID_NOT_ACCEPT_AND_INSTALL",
    )
    monitor = StateMonitor(frozen_model(spec))
    with sealed(native / "state.bin.journal").open("rb") as fp, forbid_integration(mujoco) as calls:
        packet = read_journal_packet(fp)
        while packet is not None and packet["step"] <= tick:
            if packet["step"] >= tick-25:
                last = monitor.observe(packet)
            if packet["step"] == tick:
                break
            packet = read_journal_packet(fp)
        require(
            packet is not None and packet["step"] == tick and len(monitor.stable) == 26
            and np.allclose(packet["qpos"], at["qpos"], atol=2e-7, rtol=0)
            and np.allclose(last[0], actual[:3, 3], atol=2e-7, rtol=0)
            and np.allclose(last[1], actual[:3, :3], atol=2e-7, rtol=0)
            and not calls,
            "ARCHIVE_CONTACT_AND_CLOCK_READBACK_MISMATCH",
        )
    return dict(
        episode=native.name, start=tick-25, end=tick, source=round(trigger/.002),
        accepted=wire, installed=wire, consumed=wire, solver_echo=True,
        finger_fraction=OPEN, finger_owner=native.name, start_pose=previous, end_pose=consumed,
    ), dict(tcp=actual, qpos=np.asarray(at["qpos"]), safety_valid=True, contact_valid=True,
            native_grasp=False, robot_object_contact=False)

def intent_for(row):
    return PhysicalIntent(
        row["episode"], row["tick"], row["input_sha256"],
        "RAW_SENSOR", np.asarray(row["at"]["actual_TCP"]), row["curve"],
    )


def sweep(row, preview):
    spec = row["spec"]
    at = row["at"]
    model = frozen_model(spec)
    monitor = StateMonitor(model)
    base = np.asarray(at["actual_mount"])
    guard = binding(spec["native_root"])
    previous, consumed, actual = step_ids(at)
    sample = dict(tcp=actual, qpos=np.asarray(at["qpos"]))
    checked, error = 0, None

    def witness(data):
        clearance = monitor.clearance(data)
        require(clearance >= .015, f"ORIGINAL_CLEARANCE: {clearance:.8f} m")

    with forbid_integration(mujoco) as calls:
        try:
            for p in np.unique(preview.reshape(-1, 16), axis=0).reshape(-1, 4, 4):
                sweep_profile(
                    model, sample, dict(old_ee=consumed, old_base=base),
                    p, base, guard, (41, 38, 23), monitor.permitted, witness,
                )
                checked += 1
        except (ValueError, RuntimeError, AssertionError) as exc:
            error = str(exc)
        require(not calls, "integrations in a CPU geometry replay")
    return dict(distinct_completed_sweeps=checked, first_refusal=error)


def frozen_gate_readback(preview, receipt, sample):
    from go2arx5_vla.world_model import fusion_trajectory_adapter as e

    thresholds = {
        "maximum_translation_speed_m_s": e.MAX_TRANSLATION_SPEED_M_S,
        "maximum_angular_speed_rad_s": e.MAX_ANGULAR_SPEED_RAD_S,
        "maximum_finger_speed_m_s": e.MAX_FINGER_SPEED_M_S,
        "maximum_discrete_translation_acceleration_m_s2": e.MAX_TRANSLATION_ACCELERATION_M_S2,
        "maximum_discrete_angular_acceleration_rad_s2": e.MAX_ANGULAR_ACCELERATION_RAD_S2,
    }
    action = pose10(preview[:40], np.full(40, OPEN))
    before = pose10(np.asarray([receipt["start_pose"], receipt["end_pose"]]), np.full(2, OPEN))
    readback = {}
    for name, wire in (("H40", action), ("WITH_PRIOR_ATOM", np.vstack((before, action[1:39])))):
        geometry = e._geometry(wire, sample["tcp"])
        readback[name] = {
            k: dict(observed=float(geometry[k]), threshold=float(v),
                    exceeds=bool(geometry[k]>v*(1+1e-10))) for k,v in thresholds.items()
        }
    return readback


def rolling_h40(segment, receipt, sample, horizon_s):
    """Check later publications with a nominal, explicitly unexecuted receipt."""
    total = 1 + round(np.floor(min(horizon_s, segment.end + .1) / .1 + 1e-9))
    checked = 0
    prior = receipt.copy()
    for k in range(total):
        preview = segment.preview(k * .1)[0]
        try:
            KnownCurve.admit(preview, prior, sample)
        except ValueError as exc:
            return dict(checked=checked, scheduled=total, first_refusal_s=k*.1,
                        reason=str(exc), frozen_gate_readback=frozen_gate_readback(preview, prior, sample))
        prior.update(start_pose=preview[1], end_pose=preview[2])
        checked += 1
    return dict(checked=checked, scheduled=total, first_refusal_s=None, reason=None)


def standing_admission(row, limits, scratch, native, sealed, available, segment):
    case = scratch / row["id"]
    case.mkdir()
    directory = case / row["episode"]
    directory.mkdir()
    adapter = SimpleNamespace(output=directory)
    tick = row["tick"]
    receipt, sample = archive_boundary(tick, row["at"], native, row["spec"], sealed)
    nominal_roll = rolling_h40(segment, receipt, sample, available)
    student = SEAStandingStudent(None, lambda *args: intent_for(row), limits, 1)
    try:
        preview = student.preview(adapter, tick, receipt, sample,
                                  SimpleNamespace(last=tick), None, None)
    except ValueError as exc:
        result = dict(first_refusal=str(exc), original_receiver_installs=0, nominal_rolling_H40=nominal_roll)
        if str(exc) == "MOBILE_ORIGINAL_FULL40_EEF_GATE":
            result["frozen_gate_readback"] = frozen_gate_readback(segment.preview(0.)[0], receipt, sample)
        return result
    require(student.calls == 1 and np.array_equal(preview[40], preview[39])
            and np.allclose(preview[0], receipt["end_pose"], atol=2e-7, rtol=0),
            "production preview seam, padding or denominator changed")
    return dict(first_refusal=None, original_H40_gate="PASS", original_receiver_installs=0,
                frozen_gate_readback=frozen_gate_readback(preview, receipt, sample),
                nominal_rolling_H40=nominal_roll,
                first_preview=sweep(row, preview),
                selected_source_from_archived_pose=sweep(row, student.compiled.selected_source))


def compile_rows(root, limits, old_limits, out, sealed):
    geo = read(BASE / "reports/SLA_GO2_V18/Z1_ATTRIBUTION/IMAGE_GEOMETRY.json")
    records = {(r["slot"], r["tick"]): r["input_sha256"] for r in geo["samples"]}
    reports = []
    for row in [*history(root), *standing(root)]:
        previous, consumed, actual = step_ids(row["at"])
        segment = ReferenceSegment(row["curve"], actual, consumed, previous, limits)
        check = segment.check()
        old = previous_clock(row, old_limits)["clock"]
        delta = segment.selected_source[-1, :3, 3] - segment.selected_source[0, :3, 3]
        rotation = segment.selected_source[-1, :3, :3] @ segment.selected_source[0, :3, :3].T
        report = dict(
            selected_source_net_translation_mm=float(1000*np.linalg.norm(delta)),
            selected_source_net_rotation_deg=float(np.degrees(Rotation.from_matrix(rotation).magnitude())),
            previous_clock=dict(
                absolute_full_lead_s=old["absolute_full_lead_s"], absolute_full_s=old["absolute_full_s"],
                same_prefix_stop_go_s=old["matched_stop_go_motion_s"],
                prior_joint_50ms_pass=old["joint_grid_limits_with_seam"]["pass"],
            ),
            id=row["id"], standing=row["standing"], archived_tick=row["tick"],
            prefix_source_s=segment.clock.end_index/30,
            fit_mm=segment.clock.shape_error_mm, fit_deg=segment.clock.shape_error_deg,
            lead_s=segment.lead, duration_s=segment.clock.duration,
            predicted_events=segment.events, simulated_install=False, **check,
        )
        if row["standing"]:
            tick = row["tick"]
            slot = row["id"].rsplit("-", 1)[0]
            row["input_sha256"] = records[slot, tick]
            panel = "Z1_REACH_483_G01" if slot == "G01-A" else "Z1_REACH_483"
            native = root / "v18" / panel / slot / row["episode"]
            available = 9.0-(tick-row["entry"]["tick"])*.002
            report.update(time_remaining_s=available, full_prefix_within_original_clock=segment.end <= available,
                          standing_receiver_preview=standing_admission(row, limits, out, native, sealed, available, segment))
        else:
            report["standing_receiver_preview"] = dict(
                original_receiver_installs=0, scope="HISTORICAL_TASK_WINDOW_GEOMETRY_ONLY",
                prior_gripper_m=.044*row["at"]["target_endpoint_EEF10"][9],
            )
        reports.append(report)
    require(len(reports) == 19 and sum(r["standing"] for r in reports) == 3,
            "fixed 19-row population changed")
    return reports


def sealed_archive(root):
    seals = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    archive = read(pin(root / "v18/sealed-Z1/members.json", seals["member_manifest_sha256"]))

    def sealed(p):
        return pin(p, archive[str(p.relative_to(root / "v18"))])

    return sealed


def fixture_checks(root, limits, sealed, out):
    case = root / "v18/Z1_REACH_483/G00-H0"
    native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    spec = read(sealed(case / "EXECUTION_SPEC.json"))
    entry = read(sealed(case / "REACH_ENTRY.json"))["tick"]
    source_tick = entry+1000
    stream = sealed(native / "MOBILE_SAMPLES.private.jsonl")
    samples = {z["native_end_step"]: z for z in rows(stream)}
    first = samples[source_tick]
    receipt, sample = archive_boundary(source_tick, first, native, spec, sealed)
    q = np.stack([matrix10(np.asarray(samples[t]["target_endpoint_EEF10"]))
                  for t in range(source_tick, source_tick+826, 25)])
    knots = np.arange(34)*DT
    clock = np.arange(50)/30
    p = np.stack([np.interp(clock, knots, axis) for axis in q[:, :3, 3].T], axis=1)
    r = Slerp(knots, Rotation.from_matrix(q[:, :3, :3]))(clock).as_matrix()
    world = np.repeat(np.eye(4)[None], 50, axis=0)
    world[:, :3, :3], world[:, :3, 3] = r, p
    # Remove the original measured/consumed bias from the supplied physical
    # path, then let the common compiler restore the frozen receiver seam.
    physical = world_offset(world, receipt["end_pose"], sample["tcp"])
    raw = np.zeros((50, 16))
    raw[:, 8:11] = physical[:, :3, 3]
    raw[:, 11:15] = Rotation.from_matrix(physical[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    raw[:, 15] = 1
    candidate = PhysicalIntent(native.name, source_tick, hashlib.sha256(stream.read_bytes()).hexdigest(),
                               "KNOWN_PHYSICAL", sample["tcp"], SourceCurve(raw, np.eye(4), np.eye(4)))
    folder = out / "G00-H0" / native.name
    folder.mkdir(parents=True)
    student = SEAStandingStudent(None, lambda *args: candidate, limits, 1)
    preview = student.preview(SimpleNamespace(output=folder), source_tick, receipt, sample,
                              SimpleNamespace(last=source_tick), None, None)
    good = student.compiled
    control = dict(at=first, spec=spec)
    first_sweep = sweep(control, preview)
    prefix_sweep = sweep(control, good.selected_source)
    require(first_sweep["first_refusal"] is None and prefix_sweep["first_refusal"] is None,
            "ARCHIVED_POSITIVE_SWEEP_REFUSED")

    motion = 1000*np.linalg.norm(good.selected_source[-1, :3, 3]-good.selected_source[0, :3, 3])
    require(motion > 1, "recorded positive source path has no motion")

    raw[:] = raw[0]
    static = replace(candidate, curve=SourceCurve(raw, np.eye(4), np.eye(4)))
    quiet = admit_intent(static, receipt, sample, source_tick, limits)
    extent = float(np.max(np.linalg.norm(
        quiet.selected_source[:, :3, 3]-quiet.selected_source[0, :3, 3], axis=1)))
    require(extent < 1e-10, "static control received source-motion credit")
    negatives = {}
    for name, model, prior, observed in (
        ("oracle", replace(candidate, provenance="ORACLE"), receipt, sample),
        ("stale", replace(candidate, tick=source_tick-25), receipt, sample),
        ("wrong_wire", candidate, dict(receipt, installed="wrong"), sample),
        ("expired_receiver", candidate, dict(receipt, source=source_tick-200), sample),
        ("contact", candidate, receipt, dict(sample, robot_object_contact=True)),
        ("wrong_gripper_seam", candidate, dict(receipt, finger_fraction=.3), sample),
    ):
        try:
            admit_intent(model, prior, observed, source_tick, limits)
        except ValueError as exc:
            negatives[name] = str(exc)
        else:
            raise AssertionError("negative receipt reached the H40 emitter: "+name)
    def unexpected_provider(adapter, tick):
        raise AssertionError("invalid receipt reached the observation provider")

    before_query = SEAStandingStudent(None, unexpected_provider, limits, 1)
    try:
        before_query.preview(
            SimpleNamespace(output=folder), source_tick, receipt,
            dict(sample, robot_object_contact=True), SimpleNamespace(last=source_tick), None, None,
        )
    except ValueError as exc:
        require(str(exc) == "CONTACT_OR_STOP", "contact rejection moved after provider")
    else:
        raise AssertionError("contact receipt produced a preview")
    return dict(
        provider_skipped_for_contact=True,
        archived_H0_slot=case.name, archived_tick=source_tick,
        archive_atom_file_sha256=candidate.observation_sha256,
        positive_H40_receipt_pass=True, positive_prefix_motion_mm=float(motion),
        positive_first_H40_sweep=first_sweep, positive_selected_source_sweep=prefix_sweep,
        static_prefix_motion_mm=1000*extent, negative_refusals=negatives,
        actual_solver_selected_MRT_reference=None,
    )


def run(root, out, receiver_endpoints):
    old_limits = read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")["limits"]
    limits = receiver_clock_limits(old_limits) if receiver_endpoints else old_limits
    out.mkdir(mode=0o700)
    sealed = sealed_archive(root)
    per_window = compile_rows(root, limits, old_limits, out, sealed)
    controls = fixture_checks(root, limits, sealed, out)
    write_new(out/"SEA2_CPU_ACTUAL.json", dict(
        scope="SEA2_SAME_NATIVE_REFERENCE_ADMISSION_CPU", windows=per_window, controls=controls,
        compiler_profile="FROZEN_RECEIVER_PUBLICATION_ENDPOINTS" if receiver_endpoints else "V16_CONTINUOUS_GEOMETRY",
        previous_compiler_envelope=old_limits, compiler_envelope=limits,
        production_reference_producer="SEAStandingStudent for standing_reach.drive",
        original_install_api="native_known.install", source_gripper_execution="FIXED_OPEN",
        source_physical_pad_gap_known=False, native_motion_profile_qualified=False,
        new_native_starts=0, new_sampler_forwards=0, integration_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receiver-endpoints", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.receiver_endpoints)
