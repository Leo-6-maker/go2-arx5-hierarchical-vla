"""Run the exact V2 receiver binary on sealed moving states with a verified non-integrating next call."""

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def run(root, out):
    code = Path(__file__).resolve().parent
    reports = code.parents[1] / "reports/SLA_HIERARCHICAL_V16"
    frozen = read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]
    roster = root / "feedback/F2_VERIFIED.json"
    if sha(roster) != frozen["F2"]:
        raise ValueError("F2 population changed")
    rr = next(a for a in read(roster)["records"] if a["card"] == "F2-01")
    f = root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01"
    result = f / "RESULT.private.json"
    sp = f / "EXECUTION_SPEC.json"
    if (
        sha(result) != rr["result_sha256"]
        or sha(sp) != read(f / "control/LEASE.json")["spec_sha256"]
    ):
        raise ValueError("invalid native case")
    episode = f / read(result)["episode"]
    spec = read(sp)
    base = Path(spec["outer_library"])
    args = next(
        v["argv"]
        for v in read(base.parent / "COMMANDS.private.json")
        if v["label"] == "task_actuation_r1_compile"
    )
    engine = Path(next(x for x in args if x.endswith("/libmujoco.so")))
    inc = next(x for x in args if x.startswith("-I") and "/library/mujoco/include" in x)
    build = read(reports / "V2_RECEIVER_BUILD.json")
    receiver = root / "v16/v2-build03/student_grip.so"
    scene = Path(spec["model_path"])
    journal = episode / "state.bin.journal"
    track = f / "TRACK.private.npz"
    for p, want in (
        (receiver, build["student_binary"]),
        (scene, spec["frozen_files"][str(scene)]),
        (journal, rr["journal_sha256"]),
        (track, rr["curve_sha256"]),
    ):
        if sha(p) != want:
            raise ValueError("changed input: " + p.name)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    library = out / "librecorded_step.so"
    exe = out / "fault_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            inc,
            str(code / "recorded_step.cc"),
            str(engine),
            "-Wl,-rpath," + str(engine.parent),
            "-o",
            str(library),
        ],
        check=True,
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-rdynamic",
            inc,
            str(code / "receiver_fault_probe.cc"),
            "-Wl,--no-as-needed",
            str(receiver),
            str(library),
            str(engine),
            "-ldl",
            "-Wl,-rpath," + str(engine.parent),
            "-o",
            str(exe),
        ],
        check=True,
    )
    with (episode / "initialization.jsonl").open() as stream:
        reset = json.loads(next(stream))
    if reset["time"] != 0:
        raise ValueError("archive begins after reset")
    reset_file = out / "reset.bin"
    np.r_[reset["qpos"], reset["qvel"]].astype("<f8").tofile(reset_file)
    with np.load(track, allow_pickle=False) as z:
        i = 6439 - 5703
        if abs(z["elapsed_ms"][i] - i * 2) > 1e-7:
            raise ValueError("archive fault clock mismatch")
        speed_mm_s = float(
            np.linalg.norm(z["actual"][i, :3, 3] - z["actual"][i - 1, :3, 3]) * 500000
        )
        demand_mm_s = float(
            np.linalg.norm(z["goal"][i, :3, 3] - z["goal"][i - 1, :3, 3]) * 500000
        )
    code_hash = sha(code / "receiver_fault_probe.cc")
    rows = []
    for method in ("VALID", "EXPIRY", "OVER_RATE"):
        d = out / method
        d.mkdir(mode=0o700)
        stop = method != "VALID"
        epoch = hashlib.sha256((method + code_hash).encode()).hexdigest()
        (d / "binding").write_text(
            "GRTQUOTA2\n" + method + "\n" + epoch + "\n" + method + "\n6503\n5\n128\n"
        )
        env = dict(
            os.environ,
            GR00T_TASK_GRIPPER_COMMAND=str(d / "unused_legacy.bin"),
            GR00T_TASK_STATE=str(d / "record.bin"),
            GR00T_TASK_STEP_QUOTA=str(d / "quota"),
            GR00T_TASK_QUOTA_BINDING=str(d / "binding"),
            GR00T_TASK_RUN_UUID=method,
            GR00T_TASK_CARD_SHA256=epoch,
            GR00T_TASK_RESERVATION_ID=method,
            GR00T_TASK_ERROR=str(d / "reject.txt"),
            V16_GRIP_DIR=str(d),
            V16_GRIP_EPOCH=epoch,
            V16_GRIP_STOP_TICK="6503",
            V3_REPLAY_JOURNAL=str(journal),
        )
        proc = subprocess.run(
            [
                str(exe),
                str(receiver),
                str(library),
                str(scene),
                str(reset_file),
                method,
            ],
            cwd=d,
            env=env,
            capture_output=True,
            timeout=120,
        )
        routes = (d / "route.txt").read_text().splitlines()
        if [Path(v).resolve() for v in routes] != [
            receiver.resolve(),
            library.resolve(),
        ]:
            raise ValueError("unverified call routing")
        calls = np.fromfile(d / "delivered_ticks.bin", dtype="<u8")
        log = list(csv.reader((d / "receiver.csv").open()))
        events = [v for v in log if "STOP" in v[0]]
        limit = 6439 if stop else 6453
        if (
            proc.returncode != (-signal.SIGINT if stop else 0)
            or not np.array_equal(calls, np.arange(1, limit + 1))
            or len(events) != int(stop)
        ):
            raise ValueError("unexpected receiver outcome " + method)
        pauses = [int(v) for v in (d / "pause_calls.txt").read_text().splitlines()]
        if int((d / "quota").read_text()) != 6453 or pauses[-1] != (
            6428 if stop else 6453
        ):
            raise ValueError("incorrect in-flight grant")
        if stop and (
            events[0][0] != "RECEIVER_REJECT_STOP"
            or int(events[0][2]) != limit
            or (d / "reject.txt").read_text().strip() != "V16_GRIPPER_TERMINAL"
        ):
            raise ValueError("missing genuine receiver rejection")
        rows.append(
            dict(
                method=method,
                process_returncode=proc.returncode,
                last_replayed_tick=int(calls[-1]),
                last_paused_tick=pauses[-1],
                outstanding_grant_tick=6453,
                receiver_stop_tick=limit if stop else None,
                delivered_after_receiver_stop=int(sum(calls > limit)) if stop else None,
                receiver_log_sha256=sha(d / "receiver.csv"),
                recorded_step_counter_sha256=sha(d / "delivered_ticks.bin"),
            )
        )
    return dict(
        treatments=rows,
        receiver_sha256=sha(receiver),
        test_step_sha256=sha(library),
        archived_motion_sha256=sha(track),
        archived_state_sha256=sha(journal),
        achieved_archive_speed_mm_s=speed_mm_s,
        desired_archive_speed_mm_s=demand_mm_s,
        new_native_simulator_starts=0,
        new_physics_ticks=0,
        whole_system_moving_STOP_qualified=False,
    )


if __name__ == "__main__":
    os.umask(0o077)
    print(
        json.dumps(
            run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()),
            indent=2,
            allow_nan=False,
        )
    )
