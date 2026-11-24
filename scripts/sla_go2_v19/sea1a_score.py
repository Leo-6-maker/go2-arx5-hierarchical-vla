"""Evaluate SEA1A's sealed returns in the measured Go2 TCP frame."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sea1a_panel import BASE, BRIDGE, PAINT, RAW, REG_SHA256, SLOTS, registration, run_path
from sea1_archive_readout import S0_PIN, alignment, pose_rows
from source_custody import digest, pin, read, require, target_case, target_obs, write_new
from demonstrated_grip import events, rows
from policy_query import input_identity, tensor_sha

TIMES = np.arange(50) / 30


def horizon_metrics(poses, actual, centers, axis):
    xyz = poses[:, :3, 3]
    displacement = xyz - xyz[0]
    forward = displacement @ axis
    transverse = np.linalg.norm(displacement - forward[:, None] * axis, axis=1)
    angle = Rotation.from_matrix(poses[:, :3, :3] @ poses[0, :3, :3].T).magnitude()
    point = actual[:3, 3]
    offset = Rotation.from_matrix(poses[0, :3, :3] @ actual[:3, :3].T).magnitude()
    return dict(
        row_end_s=(len(xyz)-1)/30, successor_state_end_s=len(xyz)/30,
        measured_start_offset_mm=1000*float(np.linalg.norm(xyz[0]-point)),
        measured_start_offset_deg=float(np.rad2deg(offset)),
        model_net_mm=1000*float(np.linalg.norm(displacement[-1])),
        axis_net_mm=1000*float(forward[-1]), axis_direction_cos=alignment(displacement[-1], axis),
        transverse_net_mm=1000*float(transverse[-1]), transverse_peak_mm=1000*float(transverse.max()),
        rotation_net_deg=float(np.rad2deg(angle[-1])), rotation_peak_deg=float(np.rad2deg(angle.max())),
        object_forecast={k: None if center is None else dict(
            measured_to_last_gain_mm=1000*float(np.linalg.norm(point-center)-np.linalg.norm(xyz[-1]-center)),
            first_to_last_gain_mm=1000*float(np.linalg.norm(xyz[0]-center)-np.linalg.norm(xyz[-1]-center)),
            last_distance_mm=1000*float(np.linalg.norm(xyz[-1]-center)),
            direction_cos=alignment(displacement[-1], center-xyz[0]),
        ) for k, center in centers.items()},
    )


def grip(fraction):
    return dict(first=float(fraction[0]), last=float(fraction[-1]), minimum=float(fraction.min()),
                maximum=float(fraction.max()), events=events(TIMES, fraction, step=False))


def event_counts(row):
    return Counter((side, e["kind"]) for side, prop in row["gripper"].items() for e in prop["events"])


def compare(reports):
    contrasts = []
    slots = {(r["state_id"], r["seed"][1], r["arm"]): r for r in reports}
    for i in range(2):
        for s in SLOTS:
            a, b = slots[s, i, RAW], slots[s, i, PAINT]
            pair = dict(state=s, seed=[0, i], statuses=[a["status"], b["status"]])
            if a["status"] == b["status"] == "RETURNED":
                require(a["noise_sha256"] == b["noise_sha256"], "paired noise mismatch")
                if all(r["P10"]["object_forecast"]["red"] is not None for r in (a, b)):
                    diff = {
                        t: b[t]["object_forecast"]["red"]["measured_to_last_gain_mm"]
                        - a[t]["object_forecast"]["red"]["measured_to_last_gain_mm"]
                        for t in ("P10", "FULL50")
                    }
                    new_events = event_counts(b) - event_counts(a)
                    pair.update(
                        P10_improvement_mm=diff["P10"], FULL50_improvement_mm=diff["FULL50"],
                        noise_sha256=a["noise_sha256"],
                        added_gripper_crossings=[dict(side=k[0], kind=k[1], count=n)
                                                 for k, n in sorted(new_events.items())],
                    )
            contrasts.append(pair)
    usable = [p for p in contrasts if "P10_improvement_mm" in p]
    values = {s: [p["P10_improvement_mm"] for p in usable if p["state"] == s] for s in SLOTS}
    gains = [s for s, v in values.items() if len(v) == 2 and min(v) >= 2.]
    conflicting = [s for s, v in values.items() if len(v) == 2 and min(v) < 0 < max(v)]
    added = sum(e["count"] for p in usable for e in p["added_gripper_crossings"])
    median = float(np.median([p["FULL50_improvement_mm"] for p in usable])) if usable else None
    if len(usable) != 8:
        decision = "INCONCLUSIVE_MISSING_PAIR"
    elif len(gains) >= 3 and median >= 0 and not added:
        decision = "POSITIVE_DEVELOPMENT_SCREEN"
    elif conflicting:
        decision = "INCONCLUSIVE_CONTRADICTORY_SEED_EFFECT"
    else:
        decision = "NEGATIVE_DEVELOPMENT_SCREEN"
    return dict(
        pairs=contrasts, scored_pairs=len(usable), paired_state_groups=sum(len(v) == 2 for v in values.values()),
        states_both_seeds_improve_at_least_2mm=gains, contradictory_state_ids=conflicting,
        FULL50_paired_median_mm=median, added_gripper_crossing_count=added, next_decision=decision,
    )


def context(root, item, sealed, calibration, geometry):
    case = target_case(root, item["layout"])
    native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    query = case / f"query-{item['id'][-2:]}"
    raw = target_obs(pin(sealed(query / "INPUT.npz"), item["input_file_sha256"]))
    record = next(x for x in rows(sealed(native / "MOBILE_SAMPLES.private.jsonl"))
                  if x["native_end_step"] == item["tick"])
    relative = np.eye(4)
    relative[:3, 3] = calibration["source_origin"]
    origin = np.asarray(record["actual_mount"]) @ relative
    actual = np.asarray(record["actual_TCP"])
    tool = np.asarray(calibration["T_link6_TCP"])
    image = geometry[case.name, item["tick"]]
    require(image["current_rgb_exact"] and image["input_sha256"] == input_identity(raw)[0] == item["input_sha256"]
            and np.allclose(origin @ pose_rows(raw["state"][None, 8:])[0] @ tool, actual, atol=1e-7, rtol=0),
            "measured tool frame and archived visual score identity")
    axis = np.asarray(read(sealed(case / "REACH_ENTRY.json"))["direction"])
    require(abs(np.linalg.norm(axis)-1) < 1e-7, "registered historical axis is not unit length")
    centers = {k: None if v["center_world_m"] is None else np.asarray(v["center_world_m"])
               for k, v in image["image_objects"].items()}
    return origin, tool, actual, centers, axis


def static_replay(frame, item, cell, absolute, poses, limits, folder):
    from sea2_archived_action_preflight import admit_preview
    from sea_reference import PhysicalIntent, ReferenceSegment, SourceCurve

    if isinstance(frame, str):
        return dict(archive_boundary_refusal=frame)
    available = 9. - frame["elapsed_s"]
    r, sample = frame["receipt"], frame["sample"]
    curve = SourceCurve(absolute, frame["origin"], frame["link6_to_tcp"])
    require(np.allclose(curve.knots, poses, rtol=0, atol=1e-12), "SEA-E action frame changed")
    provenance = "RAW_SENSOR" if cell["arm"] == RAW else "DEPLOYABLE_SEA"
    intent = PhysicalIntent(frame["native"].name, item["tick"], cell["input_sha256"], provenance, sample["tcp"], curve)
    try:
        segment = ReferenceSegment(curve, sample["tcp"], r["end_pose"], r["start_pose"], limits)
    except ValueError as exc:
        return dict(compile_refusal=str(exc), available_s=available)
    return dict(compile_refusal=None, available_s=available, selected_end_row=segment.clock.end_index,
                selected_source_s=segment.clock.end_index / 30,
                compiled_s=segment.end, fits_time=segment.end <= available,
                standing_preview=admit_preview(frame, intent, segment, limits, folder))


def score(root, output):
    from sea2_archived_action_preflight import captured_frame
    from sea2_preflight import sealed_archive
    from sea_reference import receiver_clock_limits

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only evaluator")
    reg = registration()
    out = run_path(root)
    start, end = read(out / "START.json"), read(out / "OWNER_EXIT.json")
    require(start["registration_sha256"] == REG_SHA256 and end["project_lock_released"]
            and end["native_starts"] == end["physics_ticks"] == end["training_updates"] == end["render_jobs"] == 0,
            "GPU worker or exclusive lease not closed")
    sealed = sealed_archive(root)
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_PIN))["inputs"]
    path = root / "v12/TARGET_REGISTRATION.json"
    cal = read(pin(path, pins[str(path)]))
    archive = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    geom = read(BASE / "reports/SLA_GO2_V18/Z1_ATTRIBUTION/IMAGE_GEOMETRY.json")
    require(geom["source_archive_sha256"] == archive["archive_sha256"], "visual centers belong to another archive")
    geometry = {(g["slot"], g["tick"]): g for g in geom["samples"]}
    limits = receiver_clock_limits(read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")["limits"])
    bridge_file = "reports/SLA_GO2_V19/SEA1/TARGET_BRIDGE_ACTUAL.json"
    bridge = read(pin(root / BRIDGE / "TARGET_BRIDGE_ACTUAL.json", digest(BASE / bridge_file)))
    edit_counts = {
        r["id"]: [{k: x[k] for k in ("tick", "actual_edits", "unresolved_pixels")}
                  for x in (*r["head"], r["wrist"])] for r in bridge["slots"]
    }
    states = {s["id"]: s for s in reg["states"]}
    contexts, frames, reports = {}, {}, []
    output.mkdir(mode=0o700)
    for cell in reg["cells"]:
        slot, state_id = cell["slot"], cell["state_id"]
        item = states[state_id]
        row = {k: cell[k] for k in ("slot", "state_id", "seed", "arm", "input_sha256")}
        entered, returned = out / (slot + ".entered.json"), out / (slot + ".returned.json")
        if not returned.exists():
            status = "TECHNICAL_ENTERED" if entered.exists() else "UNSTARTED"
            reports.append(dict(row, status=status))
            continue

        receipt, entry = read(returned), read(entered)
        require(all(entry[k] == cell[k] for k in ("slot", "seed", "input_sha256")), "return has wrong entry identity")
        with np.load(out / f"noise-{cell['seed'][1]}.npz", allow_pickle=False) as n:
            require(n["sample_key"].tolist() == entry["sample_key"]
                    and tensor_sha(n["noise"]) == entry["noise_sha256"], "noise storage mismatch")
        with np.load(pin(out / (slot + ".npz"), receipt["output_sha256"]), allow_pickle=False) as z:
            absolute = z["absolute"].copy()
            require(tensor_sha(absolute) == receipt["decoded_sha256"], "inverse bytes changed")
            raw_sha = tensor_sha(z["actions"])
            raw_dtype = str(z["actions_dtype"])
            state_sha = tensor_sha(z["sampler_state"])
            state_dtype = str(z["sampler_state_dtype"])

        if state_id not in contexts:
            contexts[state_id] = context(root, item, sealed, cal, geometry)
            try:
                frames[state_id] = captured_frame(root, item, sealed, cal)
            except ValueError as exc:
                frames[state_id] = str(exc)
        origin, tool, actual, centers, axis = contexts[state_id]
        poses = origin @ pose_rows(absolute[:, 8:]) @ tool
        with (output / (slot + ".world.private.npz")).open("xb") as f:
            np.savez(f, world_tcp=poses)
        reports.append(dict(
            row, status="RETURNED", **receipt, noise_sha256=entry["noise_sha256"],
            raw_storage_sha256=raw_sha, raw_original_dtype=raw_dtype, sampler_state_storage_sha256=state_sha,
            sampler_state_original_dtype=state_dtype, sampler_fields=entry["sampler_fields"], world_tcp_sha256=tensor_sha(poses),
            P10=horizon_metrics(poses[:10], actual, centers, axis),
            FULL50=horizon_metrics(poses, actual, centers, axis),
            gripper=dict(left=grip(absolute[:, 7]), right=grip(absolute[:, 15])),
            source_physical_gripper_width_calibrated=False,
            left_proposal_sha256=tensor_sha(absolute[:, :8]),
            static=static_replay(frames[state_id], item, cell, absolute, poses, limits, output / slot),
        ))
    usage = dict(Counter(r["status"] for r in reports))
    journal = list(rows(out / "CALLS.jsonl")) if (out / "CALLS.jsonl").exists() else []
    worker_end = next((r for r in journal if r["event"] == "PANEL_RETURNED"), {})
    write_new(output / "ACTUAL.json", dict(
        registration_sha256=REG_SHA256, issue=496, roster_slots=[r["slot"] for r in reports],
        reused_verified=0, entered=usage.get("RETURNED", 0)+usage.get("TECHNICAL_ENTERED", 0),
        returned=usage.get("RETURNED", 0), technical=usage.get("TECHNICAL_ENTERED", 0),
        unstarted=usage.get("UNSTARTED", 0), screened=0,
        processor_identity=reg["processor_identity"], worker_exit=end, peak_rss_kib=worker_end.get("resident_peak_kib"),
        edit_count_and_unresolved=edit_counts, full_pose_results=reports, **compare(reports),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    score(args.root.resolve(), args.output.resolve())
