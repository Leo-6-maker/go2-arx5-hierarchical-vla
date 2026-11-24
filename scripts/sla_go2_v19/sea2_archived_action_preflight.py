"""Replay seven sealed official returns through the frozen standing reference admission."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from sea_action import bind_decoded_return
from sea_reference import PhysicalIntent, ReferenceSegment, SEAStandingStudent, receiver_clock_limits
from sea1_archive_readout import S0_PIN, mask_and_output, pose_rows
from sea2_preflight import archive_boundary, frozen_gate_readback, rolling_h40, sealed_archive, sweep
from source_custody import BASE, digest, pin, read, require, target_case, target_obs, write_new
from demonstrated_grip import events, rows
from policy_query import tensor_sha

SEA = BASE / "reports/SLA_GO2_V19"


def captured_frame(root, item, sealed, calibration):
    case = target_case(root, item["layout"])
    spec = read(sealed(case / "EXECUTION_SPEC.json"))
    native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    tick = item["tick"]
    records = {p["native_end_step"]: p for p in rows(sealed(native / "MOBILE_SAMPLES.private.jsonl"))}
    request_path = case / f"query-{int(item['id'][-2:]):02d}"
    req = read(sealed(request_path / "REQUEST.json"))
    require(req["tick"] == tick and req["episode"] == native.name and req["input_sha256"] == item["input_sha256"],
            "archive capture and query differ")
    observed = target_obs(pin(sealed(request_path / "INPUT.npz"), item["input_file_sha256"]))
    at = records[tick]
    receipt, sample = archive_boundary(tick, at, native, spec, sealed)
    relative = np.eye(4)
    relative[:3, 3] = calibration["source_origin"]
    origin = np.asarray(at["actual_mount"]) @ relative
    entry = read(sealed(case / "REACH_ENTRY.json"))["tick"]
    return dict(case=case, native=native, spec=spec, at=at, receipt=receipt, sample=sample,
                elapsed_s=(tick - entry) * .002, origin=origin,
                link6_to_tcp=np.asarray(calibration["T_link6_TCP"]), observation=observed)


def retained_payload(root, sealed, cell, anchor, verification):
    issue, masks, payload = mask_and_output(root, sealed, anchor, cell)
    require(verification["recorded_seed_and_inverse_verified"] and issue == verification["issue"]
            and masks == verification["image_masks"] and cell["output_sha256"] == verification["output_sha256"]
            and cell["input_sha256"] == verification["input_sha256"], "prior official inverse receipt differs")
    with np.load(payload, allow_pickle=False) as z:
        raw = z["actions"].astype(str(z["actions_dtype"]))
        state = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
        absolute = z["absolute"].copy()
    require(tensor_sha(raw) == verification["raw_actions_sha256"]
            and tensor_sha(state) == verification["sampler_state_sha256"]
            and tensor_sha(absolute) == verification["decoded_absolute_sha256"], "sampler boundary bytes differ")
    return absolute


def projected_channel(arm, current, times):
    require(np.max(abs(np.linalg.norm(arm[:, 3:7], axis=1) - 1)) <= 1e-4,
            "archived arm rotation changed")
    path = pose_rows(arm)
    current_pose = pose_rows(current[None])[0]
    error_mm = np.linalg.norm(path[:, :3, 3] - current_pose[:3, 3], axis=1) * 1000
    angle_deg = np.rad2deg(Rotation.from_matrix(path[:, :3, :3] @ current_pose[:3, :3].T).magnitude())
    return dict(
        first_from_query_link6_mm=float(error_mm[0]), max_from_query_link6_mm=float(error_mm.max()),
        max_from_query_rotation_deg=float(angle_deg.max()),
        min_gripper_fraction=float(arm[:, -1].min()), max_gripper_fraction=float(arm[:, -1].max()),
        gripper_events=events(times, arm[:, -1], step=False),
    )


def admit_preview(frame, intent: PhysicalIntent, segment, limits, folder):
    receipt, sample = frame["receipt"], frame["sample"]
    tick = intent.tick
    student = SEAStandingStudent(None, lambda *_: intent, limits, 1)
    directory = folder / frame["native"].name
    directory.mkdir(parents=True)
    try:
        preview = student.preview(SimpleNamespace(output=directory), tick, receipt, sample,
                                  SimpleNamespace(last=tick), None, None)
    except ValueError as exc:
        return dict(refusal=str(exc), receiver_installs=0, frozen_gate=frozen_gate_readback(
            segment.preview(0.)[0], receipt, sample))
    require(student.calls == 1 and np.array_equal(preview[40], preview[39])
            and np.allclose(preview[0], receipt["end_pose"], rtol=0, atol=2e-7),
            "archived action preview did not preserve the standing receiver seam")
    return dict(refusal=None, receiver_installs=0, frozen_gate=frozen_gate_readback(preview, receipt, sample),
                preview_hash=tensor_sha(preview), first_preview=sweep(frame, preview),
                selected_prefix_sweep=sweep(frame, segment.selected_source),
                nominal_rolling_H40=rolling_h40(segment, receipt, sample, 9.-frame["elapsed_s"]))


def run_cell(root, sealed, item, cell, saved, frame, limits, out):
    absolute = retained_payload(root, sealed, cell, item, saved)
    args = dict(episode=frame["native"].name, tick=item["tick"], observation=frame["observation"],
                expected_observation_sha256=item["input_sha256"], return_observation_sha256=cell["input_sha256"],
                absolute=absolute, decoded_sha256=saved["decoded_absolute_sha256"], policy_origin=frame["origin"],
                link6_to_tcp=frame["link6_to_tcp"], actual_tcp=frame["sample"]["tcp"], provenance="RAW_SENSOR")
    bound = bind_decoded_return(**args)
    intent = bound.intent
    require(np.allclose(intent.curve.knots, frame["origin"] @ pose_rows(absolute[:, 8:]) @ frame["link6_to_tcp"],
                        rtol=0, atol=1e-12), "right arm world conversion differs from verified inverse replay")
    channels = {label: projected_channel(bound.decoded[:, start:start+8], bound.query_state[start:start+8], bound.times_s)
                for start, label in ((0, "left"), (8, "right"))}
    common = dict(id=item["id"], tick=intent.tick, condition=cell["arm"], seed=cell["seed"], issue=saved["issue"],
                  input_sha256=intent.observation_sha256, decoded_absolute_sha256=bound.decoded_sha256,
                  output_sha256=cell["output_sha256"], input_image_masks=saved["image_masks"],
                  source_rows=len(bound.times_s), last_source_grid_s=float(bound.times_s[-1]),
                  left_proposal_sha256=tensor_sha(bound.decoded[:, :8]), original_channels=channels,
                  source_physical_gap_known=False, available_s=9.-frame["elapsed_s"])
    try:
        r, sample = frame["receipt"], frame["sample"]
        segment = ReferenceSegment(intent.curve, sample["tcp"], r["end_pose"], r["start_pose"], limits)
    except ValueError as exc:
        return dict(**common, compile_refusal=str(exc), native_calls=0), args
    folder = out / f"{item['id']}-{cell['arm']}-seed{cell['seed'][1]}"
    selected = segment.selected_source
    net = selected[-1, :3, 3] - selected[0, :3, 3]
    report = dict(
        **common, compile_refusal=None, source_selected_s=segment.clock.end_index / 30,
        selected_net_mm=float(1000 * np.linalg.norm(net)), world_offset_lead_s=segment.lead,
        complete_reference_s=segment.end, within_original_remaining_time=segment.end <= common["available_s"],
        predicted_events=segment.events, grids=segment.check(),
        standing_preview=admit_preview(frame, intent, segment, limits, folder), native_calls=0,
    )
    return report, args


def refusal_controls(args):
    changed = args["absolute"].copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    moved_origin = args["policy_origin"].copy()
    moved_origin[0, 3] += .001
    controls = (
        ("stale_return", {"return_observation_sha256": "0" * 64}),
        ("changed_left_return", {"absolute": changed}),
        ("misbound_tool_frame", {"policy_origin": moved_origin}),
        ("oracle_provenance", {"provenance": "ORACLE"}),
    )
    refused = {}
    for label, override in controls:
        try:
            bind_decoded_return(**(args | override))
        except ValueError as exc:
            refused[label] = str(exc)
        else:
            raise ValueError(f"{label} reached reference compilation")
    return refused


def run(root, out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU archive replay only")
    roster_path = SEA / "SEA0/SEA0_PAIRED_ROSTER.json"
    verification_path = SEA / "SEA1/ARCHIVE_READBACK.json"
    roster, verification = read(roster_path), read(verification_path)
    require(verification["roster_sha256"] == digest(roster_path), "frozen archive roster changed")
    anchors = {r["id"]: r for r in roster["go2"]}
    checked = {(r["id"], r["condition"], tuple(r["seed"])): r for r in verification["archived_target_cells"]}
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_PIN))["inputs"]
    f = root / "v12/TARGET_REGISTRATION.json"
    calibration = read(pin(f, pins[str(f)]))
    sealed = sealed_archive(root)
    limits = receiver_clock_limits(read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")["limits"])
    out.mkdir(mode=0o700)
    frames, results, negative = {}, [], None
    for cell in roster["exact_archive_candidates"]:
        item = anchors[cell["id"]]
        if item["id"] not in frames:
            frames[item["id"]] = captured_frame(root, item, sealed, calibration)
        key = (cell["id"], cell["arm"], tuple(cell["seed"]))
        report, args = run_cell(root, sealed, item, cell, checked[key], frames[item["id"]], limits, out)
        results.append(report)
        if negative is None:
            negative = refusal_controls(args)
    absent = sorted(set(anchors) - set(frames))
    require(len(results) == len(checked) == 7 and absent == ["G00-01"], "frozen standing return population changed")
    write_new(out / "ARCHIVED_POLICY_ACTION_ACTUAL.json", dict(
        scope="FROZEN_ARCHIVAL_MODEL_RETURNS_CPU_H40_ONLY", exact_readback_sha256=digest(verification_path),
        geometry_scope="GO2_REGISTERED_POLICY_ORIGIN_AND_TARGET_TCP", left_command_executed=False,
        right_gripper_reference="FIXED_OPEN_PER_SLIDE_42MM", missing_return_observation_ids=absent,
        replayed=results, precompile_refusal_controls=negative,
        new_sampler_forwards=0, new_installs=0, native_starts=0, integration_ticks=0, training_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
