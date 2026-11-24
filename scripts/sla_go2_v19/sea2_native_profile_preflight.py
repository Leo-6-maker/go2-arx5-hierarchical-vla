"""Cold-replay a fixed no-contact SEA calibration panel without opening an owner."""

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from sea_reference import (
    BASE, DT, SourceCurve, PhysicalIntent, SEAStandingStudent, receiver_clock_limits,
)
from sea2_preflight import archive_boundary, rolling_h40, sealed_archive, sweep
from demonstrated_grip import pin, read, require, rows
from go2arx5_vla.world_model.native_replay import forbid_integration
from reach_native import audit as audit_receiver
from visual_lift_panel import write_new
from world_intent_clock import world_offset

# Fixed small-amplitude calibration proposals, in metres and degrees. The
# components of the known pose curves are world axes, independent of the scene.
CARDS = (
    ("SEA2-HOLD", "G00", (0., 0., 0.), (0., 0., 0.)),
    ("SEA2-XYZ", "G00", (.003, .0023, .0018), (0., 0., 0.)),
    ("SEA2-SO3", "G01", (0., 0., 0.), (.85, .55, .65)),
    ("SEA2-MIXED", "G01", (.0021, .0016, .0013), (.60, .40, .50)),
)
NATIVE_WINDOW_S = 9.0
IDENTITIES = ("outer_library", "native_binary", "solver_binary", "controller_binary")


def known_intent(episode, tick, tcp, translation, degrees):
    phase = np.minimum(np.arange(50), 15) / 15
    shape = np.sin(2*np.pi*phase)
    double = np.sin(4*np.pi*phase)
    xyz = np.column_stack((shape, shape, double)) * translation
    angles = np.column_stack((shape, double, shape)) * degrees
    rotations = Rotation.from_euler("xyz", angles, degrees=True).as_matrix() @ tcp[:3, :3]
    action = np.zeros((50, 16))
    action[:, 8:11] = tcp[:3, 3] + xyz
    action[:, 11:15] = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    action[:, 15] = 1.
    binding = episode.encode() + tick.to_bytes(8, "little") + np.ascontiguousarray(action).tobytes()
    return PhysicalIntent(
        episode, tick, hashlib.sha256(binding).hexdigest(),
        "KNOWN_PHYSICAL", tcp.copy(), SourceCurve(action, np.eye(4), np.eye(4)),
    )


def calibration_student(card, limits, query_cap=1):
    """Build the known-physical provider consumed by the standing receiver."""
    _, _, translation, rotation = card

    def provider(adapter, tick):
        actual = adapter.measured_task_state()["tcp"]
        return known_intent(adapter.output.name, tick, actual, translation, rotation)

    return SEAStandingStudent(None, provider, limits, query_cap)


def archive_sample(root, sealed, layout):
    parent = "Z1_REACH_483" if layout == "G00" else "Z1_REACH_483_G01"
    case = root / "v18" / parent / (layout + "-H0")
    spec = read(sealed(case / "EXECUTION_SPEC.json"))
    entry = read(sealed(case / "REACH_ENTRY.json"))["tick"]
    native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    atom = next(a for a in rows(sealed(native / "MOBILE_SAMPLES.private.jsonl"))
                if a["native_end_step"] == entry)
    receipt, sample = archive_boundary(entry, atom, native, spec, sealed)
    return spec, native, entry, atom, receipt, sample


def stop_predecessor(root, sealed):
    import mujoco

    actual = read(BASE / "reports/SLA_GO2_V18/E1_NATIVE/ACTUAL.json")
    public = next(q for q in actual["cases"] if q["slot"] == "G00-STOP")
    case = root / "v18/Q_E1_482_d193/G00-STOP"
    result = read(pin(case / "RESULT.private.json", public["evidence_sha256"]["RESULT.private.json"]))
    spec = read(pin(case / "EXECUTION_SPEC.json", public["evidence_sha256"]["EXECUTION_SPEC.json"]))
    out = case / result["episode"]
    for name, fingerprint in public["evidence_sha256"].items():
        f = out / name.removeprefix("native/") if name.startswith("native/") else case / name
        pin(f, fingerprint)
    with forbid_integration(mujoco) as steps:
        check = audit_receiver(case, result, spec)
    require(
        not steps and public["lease_closed"] and public["remaining_processes"] == 0
        and public["integrations_after_stop"] == public["installs_after_stop"] == 0
        and check["qualified"] and check["ticks"] == public["stop_tick"]
        and check["after_stop_integrations"] == check["after_stop_installs"] == 0,
        "archived STOP did not requalify",
    )
    versions = {k: spec["frozen_files"][spec[k]] for k in IDENTITIES}
    paired = []
    for layout in ("G00", "G01"):
        evaluation = archive_sample(root, sealed, layout)[0]
        require(all(evaluation["frozen_files"][evaluation[k]] == v for k,v in versions.items())
                and evaluation["max_ticks"] == spec["max_ticks"] == 20000,
                "candidate runtime differs from the STOP identity")
        if layout == "G00":
            require(evaluation["frozen_files"][evaluation["model_path"]]
                    == spec["frozen_files"][spec["model_path"]], "G00 STOP scene mismatch")
        paired.append(layout)
    return dict(
        issue=482, slot=public["slot"], stop_tick=check["ticks"],
        after_stop_integrations=0, after_stop_installs=0, active_motion_mm=check["actual_50ms_mm"],
        identical_receiver_and_frozen_executables_for=paired, direct_scene_match=["G00"],
        other_scene="G01 uses the same binary and 41/38/20 state ABI",
        terminal_lease_closed=public["lease_closed"], runtime_identities=versions,
        sea_profile_executed_in_predecessor=False,
    )


def synthetic_motion(student, receipt, sample, *, response, fault=False):
    segment = student.compiled
    base = student.reference_tick
    anchor = receipt["end_pose"]
    began_source = False
    for elapsed in range(25, int(np.ceil((segment.end+.1) / DT))*25+1, 25):
        tick = base + elapsed
        times = elapsed*.002 + np.array([-DT, 0.])
        trace, _, phase = segment.at(times)
        ref = trace[1].copy()
        source = base+50*((elapsed-1)//50)
        fixture_wire = hashlib.sha256(f"CPU-only:{receipt['episode']}:{source}".encode()).hexdigest()
        status = dict(receipt, start=tick-25, end=tick, source=source,
                      accepted=fixture_wire, installed=fixture_wire, consumed=fixture_wire,
                      start_pose=trace[0], end_pose=ref)
        observed = sample["tcp"].copy()
        if response != "HOLD":
            observed = world_offset(ref[None], anchor, observed)[0]
        if response == "OPPOSED":
            observed[:3, 3] = 2*sample["tcp"][:3, 3] - observed[:3, 3]
            rot0 = sample["tcp"][:3, :3]
            observed[:3, :3] = rot0 @ observed[:3, :3].T @ rot0
        if fault and times[0] >= segment.lead-1e-9 and phase[1] > phase[0]:
            ref[0, 3] += .0005
            try:
                student.consumed(tick, status, dict(sample, tcp=observed))
            except ValueError as exc:
                require(str(exc) == "SOURCE_PHASE_WITHOUT_EXACT_CONSUMPTION", "wrong fault branch")
                return str(exc)
            raise AssertionError("modified endpoint received source credit")
        student.consumed(tick, status, dict(sample, tcp=observed))
        if times[0] >= segment.lead-1e-9 and phase[1] > phase[0]:
            began_source = True
        if not began_source:
            require(student.motion is None, "lead-in or hold received source-motion credit")
    return student.motion


def rotation_refresh(adapter, card, limits, receipt, sample):
    tick = receipt["end"]
    student = calibration_student(card, limits, query_cap=2)
    student.preview(adapter, tick, receipt, sample, SimpleNamespace(last=tick), None, None)
    positive = synthetic_motion(student, receipt, sample, response="TRACKED")
    require(positive is not None and positive["rotation"] and not positive["translation"],
            "rotation refresh fixture has no eligible motion")
    segment = student.compiled
    offset = 50 * int(np.ceil((segment.end + .1) / .1))
    fresh = tick + offset
    old, new = segment.at(offset*.002 + np.array([-DT, 0.]))[0]
    fixture_wire = hashlib.sha256(f"CPU-only:rotation-refresh:{fresh}".encode()).hexdigest()
    continuation = dict(receipt, start=fresh-25, end=fresh, source=fresh-50,
                        start_pose=old, end_pose=new, accepted=fixture_wire,
                        installed=fixture_wire, consumed=fixture_wire)
    student.preview(adapter, fresh, continuation, sample, SimpleNamespace(last=fresh), None, None)
    summary = student.summary()
    require(summary["accepted_returns"] == 2 and summary["post_motion_forward"]
            and summary["source_motion"] == positive,
            "verified rotation did not permit a fresh reference query")
    return True


def axis_extent(segment):
    source = segment.selected_source
    p = 1000*(source[:, :3, 3] - source[0, :3, 3])
    r = np.rad2deg(Rotation.from_matrix(source[:, :3, :3] @ source[0, :3, :3].T).as_rotvec())
    return dict(xyz_min_mm=p.min(0).tolist(), xyz_max_mm=p.max(0).tolist(),
                rotvec_world_min_deg=r.min(0).tolist(), rotvec_world_max_deg=r.max(0).tolist())


def read_legacy(root):
    output = []
    for issue, artifact in (
        (472, "known-dynamic-actual01.json"), (477, "a1-477-verification01.json"),
    ):
        path = root / "v17" / artifact
        report = read(path)
        raw = report["case_audits"]
        require(all(x["lease"] == "CLOSED" for x in raw)
                and len(raw) == report["registered"] and report["unstarted"] == [],
                "legacy card accounting changed")
        records = []
        for row in raw:
            record = dict(card=row["card"], outcome=row["outcome"], ticks=row["end"])
            if "achieved_to_nominal" in row:
                result = row["achieved_to_nominal"]
                nominal, issued = result["motion"], row["achieved_to_issued"]["motion"]
                record.update(
                    nominal_full_tube=nominal["in_tube"] == 1,
                    nominal_rms_peak_mm=nominal["mm_rms_max"],
                    issued_rms_peak_mm=issued["mm_rms_max"],
                    translation_shape=result["translation"], rotation_shape=result["rotation"],
                )
            records.append(record)
        output.append(dict(issue=issue, reviewed_records=records, legacy_ticks=report["native_ticks"],
                           actual_readout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           sea_compiler_used=False))
    return output


def execute_fixture(root, out, limits, sealed, card):
    name, layout, translation, rotation = card
    spec, native, tick, atom, receipt, sample = archive_sample(root, sealed, layout)
    folder = out / name / native.name
    folder.mkdir(parents=True)
    adapter = SimpleNamespace(output=folder, measured_task_state=lambda: sample)
    monitor = SimpleNamespace(last=tick)

    def student():
        result = calibration_student(card, limits)
        result.preview(adapter, tick, receipt, sample, monitor, None, None)
        return result

    candidate = student()
    segment = candidate.compiled
    rolling = rolling_h40(segment, receipt, sample, NATIVE_WINDOW_S)
    # Sweep all nominal publication endpoints on this archived starting state.
    plan = segment.at(np.arange(round(segment.end / DT)+1)*DT)[0]
    geometry = sweep(dict(at=atom, spec=spec), plan)
    require(segment.end <= NATIVE_WINDOW_S and rolling["reason"] is None and geometry["first_refusal"] is None,
            "candidate CPU preflight refused")
    perfect = synthetic_motion(candidate, receipt, sample, response="TRACKED")
    stationary = synthetic_motion(student(), receipt, sample, response="HOLD")
    opposed = synthetic_motion(student(), receipt, sample, response="OPPOSED")
    require(stationary is None and opposed is None, "stationary or opposed TCP received motion credit")
    if name == "SEA2-HOLD":
        require(perfect is None, "zero-motion reference qualified")
    elif name == "SEA2-XYZ":
        require(perfect is not None and perfect["translation"] and not perfect["rotation"],
                "translation-only fixture")
    elif name == "SEA2-SO3":
        require(perfect is not None and perfect["rotation"] and not perfect["translation"],
                "rotation-only fixture cannot open the observation boundary")
    else:
        require(perfect is not None, "mixed fixture lacks any motion proof")
    fault = None if perfect is None else synthetic_motion(
        student(), receipt, sample, response="TRACKED", fault=True
    )
    return dict(
        id=name, layout=layout, archive_case=layout+"-H0", archive_tick=tick,
        frozen_scene_sha256=spec["frozen_files"][spec["model_path"]],
        receiver_sha256=spec["frozen_files"][spec["outer_library"]],
        translation_parameters_m=translation, xyz_euler_parameters_deg=rotation,
        compiled_source_phase_s=segment.clock.end_index/30, compiled_total_s=segment.end,
        short_path=axis_extent(segment), nominal_rolling_h40=rolling,
        initial_state_geometry=geometry, simulated_motion_branch=perfect,
        simulated_hold_received_motion_credit=False, simulated_opposed_received_motion_credit=False,
        synthetic_modified_endpoint_refusal=fault,
        synthetic_fresh_query_after_rotation=(
            rotation_refresh(adapter, card, limits, receipt, sample) if name == "SEA2-SO3" else None
        ),
    )


def run(root, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only")
    limits = receiver_clock_limits(read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")["limits"])
    sealed = sealed_archive(root)
    output.mkdir(mode=0o700)
    stop = stop_predecessor(root, sealed)
    result = [execute_fixture(root, output, limits, sealed, card) for card in CARDS]
    write_new(output / "SEA2_NATIVE_PROFILE_CPU.json", dict(
        status="CANDIDATE_CPU_ONLY",
        legacy=read_legacy(root), predecessor_stop=stop, candidate=result,
        physics_ticks=0, frozen_forwards=0, training_updates=0, actual_native_starts=0,
        real_dynamic_profile_qualified=False, independent_science_allocation=False,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
