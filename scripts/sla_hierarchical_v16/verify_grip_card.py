"""Independently score gripper receiver bytes against the frozen native state journal."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(BASE / "scripts/sla_transfer_v14"),
    str(BASE / "scripts/sla_transfer_v15"),
]
from b2_owner import baseline, configure
from identity import read, sha


def require(cond, message):
    if not cond:
        raise ValueError(message)


def states(folder):
    from scripts.pi0294.task_state import parse_state

    with (folder / "state.bin.journal").open("rb") as f:
        for i in itertools.count(1):
            size = f.read(4)
            if not size:
                return
            require(len(size) == 4, "incomplete state frame length")
            (n,) = struct.unpack("<I", size)
            require(40 <= n <= 2**20, "invalid state frame")
            raw = f.read(n)
            require(len(raw) == n, "incomplete state frame")
            s = parse_state(raw)
            require(i == s["step"], "unobserved native step")
            require(abs(s["time_s"] - i * 0.002) <= 1e-8, "native state clock")
            yield s


def inspect_historical(root):
    import mujoco

    spec = baseline(root)
    m = mujoco.MjModel.from_binary_path(spec["model_path"])
    geom = [m.geom(i).name for i in range(m.ngeom)]
    f = root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01"
    r = read(f / "RESULT.private.json")
    folder = f / r["episode"]
    require(
        sha(folder / "state.bin.journal") == r["journal_sha256"],
        "archived native state",
    )
    for row in states(folder):
        require(
            all(support(geom[c["geom0"]], geom[c["geom1"]]) for c in row["contacts"]),
            "non-support contact in archived startup",
        )
        if row["step"] == 6503:
            return
    raise ValueError("archived replay too short")


def support(a, b):
    pair = {a, b}
    return pair == {"task_table", "task_cube_geom"} or (
        "floor" in pair and bool(pair & {"FL", "FR", "RL", "RR"})
    )


def score(root, case):
    import mujoco

    r = read(case / "RESULT.private.json")
    spec = read(case / "EXECUTION_SPEC.json")
    lease = read(case / "control/LEASE.json")
    out = case / r["episode"]
    ch = case / "student"
    require(
        lease["status"] == "CLOSED"
        and lease["spec_sha256"] == sha(case / "EXECUTION_SPEC.json"),
        "closed lease identity",
    )
    grant = read(ch / "authority.json")
    epoch = grant["epoch"]
    require(
        grant["episode"] == out.name
        and grant["owner"] == lease["owner"]
        and grant["student_binary"] == sha(Path(spec["outer_library"])),
        "gripper authority",
    )
    initial = read(out / "BEFORE_FIRST_INTEGRATION.json")
    require(
        initial["sim_time_s"] == 0
        and initial["arm_mode"] == "HOLD"
        and not initial["running"],
        "physical startup clock",
    )
    commands = [json.loads(l) for l in (ch / "issuer.jsonl").read_text().splitlines()]
    w = read(ch / "command_epoch.json")
    require(
        w["epoch"] == epoch
        and w["absolute_reset_tick"] == 2503
        and w["physical_reset_tick"] == 0
        and w["arm_mode"] == "WBC"
        and w["no_previous_student_packet"],
        "command reset",
    )
    with (out / "control.jsonl").open() as file:
        control = [json.loads(l) for l in file]
    require(
        any(
            c["arm_mode"] == "WBC"
            and c["running"]
            and abs(c["sim_time_s"] - 5.006) <= 0.025
            for c in control
        ),
        "independent WBC state",
    )
    rows = list(csv.reader((ch / "receiver.csv").open()))
    for row in rows:
        require(len(row) == 12 and row[1] == epoch, "wrong receiver epoch")
    require(
        rows[0][0] == "PHYSICAL_RESET"
        and int(rows[0][2]) == 0
        and int(rows[0][3]) == 0,
        "no physical owner reset",
    )
    accepted = [row for row in rows if row[0] == "RECEIVER_ACK"]
    resets = [row for row in rows if row[0] == "COMMAND_EPOCH_RESET"]
    require(
        len(resets) == 1 and int(resets[0][2]) == 2503 and int(resets[0][3]) == 0,
        "no receiver command reset",
    )
    require(len(accepted) == len(commands) == 160, "incomplete command population")
    for i, (a, c) in enumerate(zip(accepted, commands)):
        wire = struct.pack("<8sQddd", b"GRTGRIP1", int(a[4]), *map(float, a[5:8]))
        require(
            c["sequence"] == i == int(a[4])
            and c["tick"] == int(a[2]) == 2503 + 25 * i
            and c["packet_sha256"] == hashlib.sha256(wire).hexdigest(),
            "ACK differs from issued packet",
        )
        require(
            math.isclose(c["source_s"], int(a[2]) * 0.002, abs_tol=1e-8),
            "absolute packet clock",
        )
    terminal = [row for row in rows if "STOP" in row[0]]
    require(
        len(terminal) == 1
        and terminal[0][0] == "RECEIVER_STOP_AT_CAP"
        and rows[-1] == terminal[0]
        and int(terminal[0][2]) == 6503,
        "terminal stop",
    )
    observed = [row for row in rows if row[0] == "NATIVE_STEP"]
    require(len(observed) == 6503, "native applied-step coverage")
    m = mujoco.MjModel.from_binary_path(spec["model_path"])
    geoms = [m.geom(i).name for i in range(m.ngeom)]
    adr = [
        int(m.jnt_qposadr[m.joint(k).id])
        for k in ("task_left_slide", "task_right_slide")
    ]
    orig = np.asarray(list(map(float, rows[0][8:10])))
    before = orig.copy()
    measured = []
    max_seam = 0.0
    support_count = 0
    for s, rec in zip(states(out), observed, strict=True):
        i = s["step"]
        q = s["qpos"][adr]
        require(
            i == int(rec[2])
            and np.allclose(q, list(map(float, rec[10:12])), atol=1e-12, rtol=0),
            "independent slide disagreement",
        )
        all_support = all(
            support(geoms[c["geom0"]], geoms[c["geom1"]]) for c in s["contacts"]
        )
        require(all_support, "prohibited contact")
        support_count += len(s["contacts"])
        active = i > 2503
        require(bool(int(rec[3])) == active, "unexpected ownership activation")
        expect = (
            orig if not active else np.repeat(commands[(i - 2504) // 25]["width_m"], 2)
        )
        if active:
            require(int(rec[4]) == (i - 2504) // 25, "wrong native sequence")
        tgt = np.asarray(list(map(float, rec[8:10])))
        require(
            np.allclose(
                tgt,
                before + np.clip(expect - before, -0.00008, 0.00008),
                rtol=0,
                atol=1e-10,
            ),
            "bilateral target clamp failed",
        )
        max_seam = max(max_seam, float(abs(tgt - before).max() / 0.002))
        before = tgt
        measured.append(q)

    def plateau(last_atom):
        end = 2503 + (last_atom + 1) * 25
        return np.median(np.asarray(measured)[end - 75 : end], axis=0)

    initial_width, closed, opened = [plateau(i) for i in (127, 143, 159)]
    close_mm = 1000 * (initial_width - closed)
    reopen_mm = 1000 * (opened - closed)
    final_mm = 1000 * abs(opened - initial_width)
    moved = bool(np.all(close_mm >= 2) & np.all(reopen_mm >= 2) & np.all(final_mm <= 1))
    require(
        r["outcome"] == "CARD_EXECUTION_COMPLETE"
        and "close_error" not in r
        and r["cancel"]["mode"] == "HOLD"
        and r["cancel"]["native_step"] == 6503,
        "clean termination",
    )
    from d1_readiness import B2_PIN

    old = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    require(sha(old) == B2_PIN, "B2 identity")
    frozen = read(old)["frozen_sha256"]
    loaded = read(out / "LOADED_EXECUTOR_00.private.json")
    require(
        all(loaded[k]["executable_sha256"] == h for k, h in frozen.items()),
        "loaded native binary mismatch",
    )
    prior = root / "v16/sla-v16-student-gripper-native-01/V2-GRIP-01"
    prev = read(prior / "RESULT.private.json")
    prev_out = prior / prev["episode"]
    require(
        prev["outcome"] == "TECHNICAL_STOP"
        and prev["issued"] == 0
        and read(prior / "control/LEASE.json")["status"] == "CLOSED"
        and not (prev_out / "PROCESS_COMMANDS.private.json").exists()
        and not (prev_out / "state.bin.journal").exists()
        and not (prior / "student/receiver.csv").exists(),
        "zero-process interruption evidence",
    )
    timeline = []
    for i, c in enumerate(commands):
        t = c["tick"]
        timeline.append(
            dict(
                sequence=i,
                source_tick=t,
                receiver_ack_tick=int(accepted[i][2]),
                end_tick=t + 25,
                width_m=c["width_m"],
                wire_sha256=c["packet_sha256"],
                applied_native_steps=25,
                measured_slide_start_mm=(1000 * measured[t - 1]).tolist(),
                measured_slide_end_mm=(1000 * measured[t + 24]).tolist(),
            )
        )
    return dict(
        full_denominator=dict(
            registered_requests=2,
            technical_stops_before_process_start=1,
            actual_simulator_starts=1,
            completed_noncontact_cards=1,
            actual_student_SLA_tasks=0,
        ),
        stage_gate="noncontact nominal receiver verified; physical adversarial rejection and moving full-system STOP unmeasured",
        original_legacy_tick_zero_WBC_gate=False,
        frozen_native_sha256=frozen,
        verified_command_intervals=timeline,
        outcome="V2_GRIPPER_QUALIFIED_ONE_CARD"
        if moved
        else "V2_BILATERAL_MOTION_BLOCKED",
        started=1,
        technical_failures=0,
        qualified_cards=int(moved),
        native_ticks=6503,
        physical_reset_arm_mode="HOLD",
        wbc_command_epoch_tick=2503,
        gripper_receiver_stop_tick=6503,
        after_receiver_stop_ticks=0,
        ack_count=len(accepted),
        applied_native_ticks=len(observed) - 2503,
        parked_reset_ticks=2503,
        peak_applied_rate_mm_s=max_seam * 1000,
        non_support_contacts=0,
        support_contacts=support_count,
        per_slide_close_mm=close_mm.tolist(),
        per_slide_reopen_mm=reopen_mm.tolist(),
        final_vs_parked_mm=final_mm.tolist(),
        native_trace=r["trace"],
        lease_closed=True,
        student_sla_attempts=0,
        source_sha256={
            p.name: sha(p)
            for p in (
                ch / "receiver.csv",
                ch / "issuer.jsonl",
                out / "state.bin.journal",
                case / "RESULT.private.json",
                case / "EXECUTION_SPEC.json",
                case / "control/LEASE.json",
            )
        },
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    configure(baseline(root))
    if sys.argv[2] == "--historical":
        inspect_historical(root)
        print("historical noncontact curve checked through 6503 ticks")
    else:
        print(json.dumps(score(root, Path(sys.argv[2])), indent=2, allow_nan=False))
