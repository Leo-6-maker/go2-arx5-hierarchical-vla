"""Exercise the proposed issuer, world reference and scorer against sealed site payloads."""

import csv
import hashlib
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from receiver_fault_replay import read, sha
from stop_card_evidence import lines, require, terminal_evidence


def run(root, repo):
    from feedback import OPEN
    from grip_card import emit
    from native_known import window
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate
    from go2arx5_vla.world_model.init_trace_reader import read_trace, audit_trace
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from go2arx5_vla.trajectory.so3 import quat_to_matrix

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    control = root / "v16/v3-stop-trace02/BRIDGE/EXPIRY/receiver.csv"
    evidence = read(reports / "V3_STOP_TRACE_ACTUAL.json")["bridge"]["treatments"]
    require(
        sha(control)
        == next(r["receiver_log_sha256"] for r in evidence if r["method"] == "EXPIRY"),
        "unverified receiver archive",
    )
    with control.open() as stream:
        ack = [r for r in csv.reader(stream) if r[0] == "RECEIVER_ACK"]
    require(len(ack) == 158, "original timing count")
    with TemporaryDirectory() as temp:
        path = Path(temp)
        for i, r in enumerate(ack):
            t = float(r[5])
            tick = int(r[2])
            e = 6438.5 * 0.002 if tick == 6428 else t + 0.399
            emit(path, r[1], i, t, opening_m=0.042, expiry_s=e)
            received = struct.pack(
                "<8sQddd", b"GRTGRIP1", int(r[4]), *map(float, r[5:8])
            )
            require(
                (path / "command.bin").read_bytes() == r[1].encode() + received,
                "differing issuer packet",
            )
            require(
                (path / "command.bin").stat().st_mode & 0o077 == 0, "packet permissions"
            )
        require(len(list(lines(path / "issuer.jsonl"))) == 158, "issuer rows")
    # The unchanged V2 caller supplies only its original four arguments.
    v2 = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B/student"
    with (v2 / "receiver.csv").open() as stream:
        prior_ack = [r for r in csv.reader(stream) if r[0] == "RECEIVER_ACK"]
    prior_cmd = list(lines(v2 / "issuer.jsonl"))
    require(
        len(prior_cmd) == len(prior_ack) == 160, "missing qualified V2 command history"
    )
    with TemporaryDirectory() as temp:
        path = Path(temp)
        for i, (row, command) in enumerate(zip(prior_ack, prior_cmd, strict=True)):
            emit(path, row[1], i, command["source_s"])
            content = (path / "command.bin").read_bytes()
            require(
                content[:64] == row[1].encode()
                and hashlib.sha256(content[64:]).hexdigest() == command["packet_sha256"]
                and content[64:]
                == struct.pack(
                    "<8sQddd", b"GRTGRIP1", int(row[4]), *map(float, row[5:8])
                ),
                "existing V2 caller changed bytes",
            )
    f = root / "feedback/F1_VERIFIED.json"
    require(
        sha(f)
        == read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["F1"],
        "population changed",
    )
    cases = []
    for rr in read(f)["records"]:
        if not rr["eligible"]:
            continue
        case = root / "feedback/sla-feedback-f1-a0-20261123-01" / rr["card"]
        rp = case / "RESULT.private.json"
        ip = case / "INITIAL.private.json"
        track = case / "TRACK.private.npz"
        require(
            sha(rp) == rr["result_sha256"]
            and sha(ip) == rr["initial_sha256"]
            and sha(track) == rr["curve_sha256"],
            "case identity",
        )
        out = case / read(rp)["episode"]
        starts = {5703 + i * 50 for i in range(15)}
        atoms = {
            r["native_start_step"]: r
            for r in lines(out / "MOBILE_SAMPLES.private.jsonl")
            if r["native_start_step"] in starts
        }
        needed = {r["installed_reference_sha256"] for r in atoms.values()}
        accepted = {}
        installs = set()
        for event in lines(out / "planner.jsonl"):
            key = event.get("ref_id")
            if key in needed:
                if event["kind"] == "accept":
                    accepted[key] = event
                elif event["kind"] == "installed":
                    installs.add(key)
            if set(accepted) == installs == needed:
                break
        initial = read(ip)["features"]
        anchor = np.asarray(initial["old_ee"])
        with np.load(track, allow_pickle=False) as z:
            actual = z["actual"]
        for i in range(15):
            tick = 5703 + 50 * i
            pos = window(anchor, i * 0.1)
            full_eef_gate(pose10(pos[:40], np.full(40, OPEN)), actual[tick - 5703])
            prev = atoms[tick]
            wire = prev["installed_reference_sha256"]
            raw = accepted[wire]["raw"]
            require(
                wire in installs
                and hashlib.sha256(raw.encode()).hexdigest() == wire
                and prev["solver_exact_installed_targets"],
                "missing genuine archived consumption",
            )
            doc = json.loads(raw)
            require(
                abs(doc["_pi0294"]["source_mpc_s"] - tick * 0.002) < 1e-8,
                "H0 install time",
            )
            pts = doc["ee"]["waypoints"]
            require(
                len(pts) == 41
                and np.allclose(
                    [p["position"] for p in pts], pos[:, :3, 3], rtol=0, atol=2e-7
                )
                and np.allclose(
                    quat_to_matrix(np.asarray([p["quaternion_xyzw"] for p in pts])),
                    pos[:, :3, :3],
                    rtol=0,
                    atol=7e-7,
                ),
                "published reference differs from registered H0 path",
            )
        # Genuine plant motion and clean task completion alone must not qualify a missing moving STOP.
        audit = audit_trace(out / "init.trace", max_native_ticks=17703)
        rows = read_trace(out / "init.trace")
        require(
            audit["status"] == "RECORDED_TRACE_CHECKS_PASS",
            "historical trace unreadable",
        )
        try:
            terminal_evidence(rows, audit, {}, [], [], False)
        except ValueError as exc:
            require(
                str(exc) == "expected one whole-system trace STOP",
                "unexpected negative control",
            )
        else:
            raise ValueError("historical motion incorrectly qualified")
        del rows
        cases.append(
            dict(
                card=rr["card"], matched_old_wires=15, negative_native_stop_control=True
            )
        )
    require(len(cases) == 4, "missing F1 matched cases")
    return dict(
        issuer_packets_equal_to_actual_receiver_ack=len(ack),
        v2_original_caller_packets_equal_to_native_ack=len(prior_ack),
        old_native_cases=cases,
        new_original_gate_previews=15 * len(cases),
        native_starts=0,
        physics_ticks=0,
    )
