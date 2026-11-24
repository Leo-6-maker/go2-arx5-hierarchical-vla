"""Verify the consumed EEF and physical gripper seams of the closed compact H0 pair."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation

from source_custody import BASE, digest, pin, read, require, write_new
from demonstrated_grip import rows

STEP, ATOM = .002, 25
TRAVEL_M, LIMIT_M_S = .044, .040
H0_SHA = "d2d7b94ea4ac4d33fdfa15082d12edd0c87276710ffc380c3e2ecd83a81fe443"


def receiver_evidence(native, sha):
    from go2arx5_vla.world_model.target_reference import trajectory_identity

    accepted, installs, latest_echo = {}, defaultdict(dict), {}
    for row in rows(pin(native / "planner.jsonl", sha)):
        ref = row.get("ref_id")
        if row["kind"] == "accept":
            require(hashlib.sha256(row["raw"].encode()).hexdigest() == ref, "invalid accepted EEF digest")
            meta = json.loads(row["raw"])["_pi0294"]
            require(meta["episode"] == native.name, "foreign reference in the closed native archive")
            accepted[ref] = meta
        elif row["kind"] == "installed" and ref in accepted:
            tag = trajectory_identity(row["times"], row["poses"], row["inputs"])
            installs[ref].setdefault(tag, row)
    for event in rows(native / "ROS_CONSUMPTION.private.jsonl"):
        if event["topic_key"] != "policy":
            continue
        target = event["message"]["plan_target_trajectories"]
        identity = trajectory_identity(target["time_trajectory"], [v["value"] for v in target["state_trajectory"]],
                                       [v["value"] for v in target["input_trajectory"]])
        latest_echo[identity] = max(event["received_monotonic_s"], latest_echo.get(identity, 0.))
    return accepted, installs, latest_echo


def verify_receiver(witness, sample, accepted, installs, echoes):
    from scripts.pi0292.eval_contract import matrix10

    wire = witness["accepted_wire_sha256"]
    target = witness["solver_target_identity"]
    meta = accepted[wire]
    source, ttl = meta["source_mpc_s"], meta["ttl_s"]
    tick = sample["native_end_step"]
    at = tick * STEP
    require(wire == sample["installed_reference_sha256"] and ttl == .4
            and meta["operation"] == "install"
            and abs(source - sample["gripper_command"]["EEF_plan_source_s"]) <= 1e-8
            and sample["safety_valid"] and witness["exact_receiver_knot"]
            and abs(witness["native_time_s"] - at) <= 1e-8
            and source <= at < source + ttl, "old P10 EEF receipt is not causal and live")
    proven = sample["solver_exact_installed_targets"]
    require(proven and all(h in installs[wire] and installs[wire][h]["monotonic_s"] <= echoes.get(h, -1.)
                           for h in {*proven, target}), "missing actual solver/receiver targets")
    row = installs[wire][target]
    i = witness["receiver_knot_index"]
    p = np.asarray(row["poses"][i])
    require(abs(row["times"][i] - at) <= 1e-8 and np.array_equal(p[:7], witness["accepted_pose7"]),
            "P10 accepted knot does not match the sealed installation")
    endpoint = matrix10(np.asarray(sample["target_endpoint_EEF10"]))
    require(np.allclose(p[:3], endpoint[:3, 3], rtol=0, atol=2e-7)
            and np.allclose(Rotation.from_quat(p[3:7]).as_matrix(), endpoint[:3, :3], rtol=0, atol=2e-7),
            "authorized boundary pose differs from the consumed program endpoint")
    return wire, float((at-source)*1000)


def gripper_evidence(case):
    acks, steps = {}, {}
    channel = case / "student"
    with (channel / "receiver.csv").open() as f:
        for row in csv.reader(f):
            if row[0] == "RECEIVER_ACK":
                require(int(row[2]) not in acks, "duplicate gripper receipt")
                acks[int(row[2])] = row
            elif row[0] == "NATIVE_STEP":
                require(int(row[2]) not in steps, "duplicate native gripper state")
                steps[int(row[2])] = row
    issuer = {r["sequence"]: r for r in rows(channel / "issuer.jsonl")}
    return read(channel / "authority.json")["epoch"], acks, steps, issuer


def verify_gripper(grip, sample, epoch, acks, steps, issuer):
    start, end = (sample[k] for k in ("native_start_step", "native_end_step"))
    ack = acks[start]
    source, fraction, expire = map(float, ack[5:8])
    sequence = grip["sequence"]
    packet = struct.pack("<8sQddd", b"GRTGRIP1", sequence, source, fraction, expire)
    command = issuer[sequence]
    width = TRAVEL_M * fraction
    require(command["packet_sha256"] == hashlib.sha256(packet).hexdigest()
            and command["tick"] == start and sample["gripper_command"]["sequence"] == sequence
            and int(ack[4]) == sequence and ack[1] == epoch and ack[3] == "1"
            and abs(source - start*STEP) < 1e-8 and abs(grip["width_m"] - width) < 1e-12
            and end*STEP < expire and end == start+ATOM, "gripper grant or receipt mismatch")
    for tick in range(start+1, end+1):
        row = steps[tick]
        require(row[1] == epoch and row[3:8] == ack[3:8], "unconsumed or changed 2 ms jaw command")
    require(np.array_equal(np.asarray(steps[end][10:12], float), grip["two_measured_finger_slide_m"]),
            "actual double slide receipt differs")
    return width


def audit(case, proof):
    from scripts.pi0292.eval_contract import matrix10

    pins = proof["evidence_sha256"]
    result = read(pin(case / "RESULT.private.json", pins["RESULT.private.json"]))
    lease = read(pin(case / "control/LEASE.json", pins["LEASE.json"]))
    native = case / result["episode"]
    start, end = result["entry_tick"], proof["physical_ticks"]
    terminal = read(pin(native / "pause_ack.json", pins["pause_ack.json"]))
    require(lease["status"] == "CLOSED" and proof["qualified"] and result["absent"]
            and terminal["step"] == end == result["cancel"]["native_step"]
            and terminal["clock_paused"] and not terminal["running"], "unclosed H0 replay")
    accepted, installs, echoes = receiver_evidence(native, pins["planner.jsonl"])
    epoch, acks, steps, issuer = gripper_evidence(case)
    samples = {r["native_end_step"]: r for r in rows(native / "MOBILE_SAMPLES.private.jsonl")}
    prior = samples[start]
    previous_pose = matrix10(np.asarray(prior["target_endpoint_EEF10"]))
    previous_width = TRAVEL_M * float(steps[start][6])
    motion_count, query_seam_count, install_seam_count = 0, 0, 0
    rates, translations, angles, ages = [], [], [], []
    last_wire = prior["installed_reference_sha256"]
    hold_path = case / "HOLD_CONSUMPTION.private.jsonl"
    atoms = list(rows(hold_path))
    for i, atom in enumerate(atoms):
        tick = start + (i+1)*ATOM
        sample = samples[tick]
        query, offset = divmod(i, 10)
        require((atom["query"], atom["atom"]) == (query, offset)
                and atom["gripper"]["native_step"] == tick and sample["native_start_step"] == tick-ATOM,
                "P10 consumption lost its 50 ms grid")
        wire, age = verify_receiver(atom["witness"], sample, accepted, installs, echoes)
        width = verify_gripper(atom["gripper"], sample, epoch, acks, steps, issuer)
        begin = matrix10(np.asarray(sample["installed_EEF10"]))
        finish = matrix10(np.asarray(sample["target_endpoint_EEF10"]))
        if wire != last_wire:
            install_seam_count += 1
            require(offset in (0, 7), "unexpected P10 replacement position")
        if offset == 0:
            query_seam_count += 1
        rates.append(abs(width-previous_width)/(ATOM*STEP))
        translations.append(np.linalg.norm(begin[:3, 3]-previous_pose[:3, 3]))
        angles.append(Rotation.from_matrix(begin[:3, :3] @ previous_pose[:3, :3].T).magnitude())
        ages.append(age)
        motion_count += not np.allclose(begin, finish, rtol=0, atol=2e-7)
        previous_pose, previous_width, last_wire = finish, width, wire
    require(len(atoms) == proof["world"]["consumed_50ms_atoms"] and start + len(atoms)*ATOM == end,
            "incomplete P10 denominator")
    return dict(start_tick=start, end_tick=end, verified_consumed_atoms=len(atoms),
                query_seams=query_seam_count, wire_changes=install_seam_count, nonstationary_reference_atoms=motion_count,
                max_consumed_eef_age_ms=max(ages), max_gripper_command_rate_mm_s=1000*max(rates),
                old_gripper_rate_limit_mm_s=1000*LIMIT_M_S, old_gripper_rate_limit_passed=bool(max(rates) <= LIMIT_M_S+1e-10),
                max_recorded_pose_seam_mm=1000*max(translations),
                max_recorded_rotation_seam_deg=float(np.degrees(max(angles))),
                original_archive_sha256={p.name: digest(p) for p in (
                    hold_path, native / "ROS_CONSUMPTION.private.jsonl", native / "MOBILE_SAMPLES.private.jsonl",
                    case / "student/receiver.csv", case / "student/issuer.jsonl",
                )},
                no_actual_controller_selected_policy_identity=True)


def run(site, output):
    cohort = read(pin(BASE / "reports/SLA_GO2_V193/GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    panel = site / "v19/V193-COMPACT-FRUIT-H0-01"
    pin(panel / "PANEL_CLOSE.json", cohort["campaign_sha256"]["PANEL_CLOSE.json"])
    require(cohort["qualified"] == cohort["started"] == 2, "actual compact teacher controls required")
    report = {key: audit(panel / key, proof) for key, proof in cohort["rows"].items()}
    write_new(output, dict(status="SEA_E2_BASELINE_H0_P10_CONSUMPTION_VERIFIED",
                           compact_report_sha256=H0_SHA, rows=report, student_policy_queries=0,
                           asynchronous_candidate_replacements=0, native_starts=0, physics_ticks=0, optimizer_updates=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("site", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU archive analysis only")
    run(args.site.resolve(), args.output)
