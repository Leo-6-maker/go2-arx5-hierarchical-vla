"""One preregistered noncontact MuJoCo gripper card; gates and once-only lease precede launch."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import sys
import traceback

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(BASE / "scripts/sla_transfer_v14"),
    str(BASE / "scripts/sla_transfer_v15"),
]
from b2_owner import baseline, check_occupancy, configure, dry_receiver, verify
from identity import sha
from identity import read, record

END = 6503
START = 2503
PLAN = "V2_NATIVE_REPAIR_REGISTRATION.json"


def width(atom):
    k = atom - 128
    if k < 0:
        return 0.042
    if k < 8:
        return 0.042 - (k + 1) * 0.0005
    if k < 16:
        return 0.038
    if k < 24:
        return 0.038 + (k - 15) * 0.0005
    return 0.042


def emit(folder, epoch, atom, t):
    payload = struct.pack(
        "<8sQddd", b"GRTGRIP1", atom, t, width(atom) / 0.044, t + 0.399
    )
    raw = epoch.encode() + payload
    pending = folder / "command.pending"
    with pending.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, folder / "command.bin")
    with (folder / "issuer.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    sequence=atom,
                    tick=START + 25 * atom,
                    source_s=t,
                    width_m=width(atom),
                    packet_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def execute(spec, sp, root, build):
    from b2_native import TechnicalAdapter, fixed, paused
    from clock_guard import new_receiver_output
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile
    from go2arx5_vla.world_model.target_sequential import owned_simulator_pid
    from go2arx5_vla.world_model.init_trace_reader import audit_trace

    lease = mobile.registered(spec, sp)
    case = sp.parent
    out = new_receiver_output(case)
    channel = case / "student"
    channel.mkdir(mode=0o700)
    epoch = secrets.token_hex(32)
    record(
        channel / "authority.json",
        dict(
            event="exclusive_grant",
            episode=out.name,
            owner=lease["owner"],
            lease_sha256=sha(case / "control/LEASE.json"),
            epoch=epoch,
            granted_before_launch=True,
            student_binary=build["student_binary"],
        ),
    )

    class StudentAdapter(TechnicalAdapter):
        def launch(self, argv, name, env):
            if name == "sim":
                env = dict(
                    env,
                    V16_GRIP_DIR=str(channel),
                    V16_GRIP_EPOCH=epoch,
                    V16_GRIP_STOP_TICK=str(END),
                )
            super().launch(argv, name, env)

    a = StudentAdapter(spec, out)
    result = dict(
        card=case.name,
        episode=out.name,
        outcome="TECHNICAL_STOP",
        native_requests=1,
        issued=0,
    )
    record(
        case / "NATIVE_REQUEST.json",
        dict(card=case.name, episode=out.name, authorized_tick_cap=END),
    )
    try:
        a.start()
        a.prepare_dynamic()
        paused(a, START)
        if a.latest["mode"].data != "WBC":
            raise ValueError("first command requires measured WBC")
        pid = owned_simulator_pid(
            next(p.pid for n, p in a.processes if n == "sim"), spec["native_binary"]
        )
        maps = (Path("/proc") / str(pid) / "maps").read_text()
        for k in (
            "outer_library",
            "mobile_safety_library",
            "observer_library",
            "trace_library",
        ):
            if spec[k] not in maps:
                raise ValueError("required library not loaded: " + k)
        if str(Path(baseline(root)["outer_library"])) in maps:
            raise ValueError("two gripper actuator owners loaded")
        mount = mobile.arm(a)
        sample = a.measured_task_state()
        if not sample["safety_valid"] or sample["native_grasp"]:
            raise ValueError("initial contact/safety")
        action, base = fixed(sample, mount)
        record(
            channel / "command_epoch.json",
            dict(
                epoch=epoch,
                absolute_reset_tick=START,
                physical_reset_tick=0,
                arm_mode=a.latest["mode"].data,
                receiver_sha256=build["student_binary"],
                no_previous_student_packet=not (channel / "command.bin").exists(),
            ),
        )
        for atom in range(160):
            tick = START + 25 * atom
            if (case.parent / "STOP.json").exists():
                raise RuntimeError("operator stop")
            if atom % 7 == 0:
                mobile.install(a, action, base)
            before = paused(a, tick)
            emit(channel, epoch, atom, before["time_s"])
            result["issued"] += 1
            sample, _ = mobile.consume(a, action, sequence=atom)
            if not sample["safety_valid"] or sample["native_grasp"]:
                raise ValueError("no-contact admission lost")
            # Last flushed receiver evidence must agree before another command can be issued.
            with (channel / "receiver.csv").open("rb") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - 1024))
                tail = f.read().decode().splitlines()[-1].split(",")
            if (
                tail[0] != ("RECEIVER_STOP_AT_CAP" if atom == 159 else "NATIVE_STEP")
                or tail[1] != epoch
                or int(tail[2]) != tick + 25
                or int(tail[4]) != atom
            ):
                raise ValueError("missing native consumption or STOP")
        paused(a, END)
        result["cancel"] = a.cancel_reference()
        if result["cancel"]["mode"] != "HOLD" or result["cancel"]["native_step"] != END:
            raise ValueError("no terminal HOLD")
        a.seal_trace(END)
        result["outcome"] = "CARD_EXECUTION_COMPLETE"
    except BaseException:
        result["failure"] = traceback.format_exc()
        try:
            step = a.observe()["step"]
            paused(a, step)
            a.seal_trace(step)
        except Exception:
            result["seal_error"] = traceback.format_exc()
    finally:
        try:
            a.close()
        except BaseException:
            result["close_error"] = traceback.format_exc()
    try:
        report = audit_trace(out / "init.trace", max_native_ticks=END)
        result["trace"] = {
            k: report[k]
            for k in (
                "status",
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
            )
        }
    except Exception:
        result["trace_error"] = traceback.format_exc()
    record(case / "RESULT.private.json", result)
    return result


def main(root, mode):
    from d1_matched_owner import with_case_lease
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0294 import task_state
    from b2_native import TechnicalAdapter

    del task_state, TechnicalAdapter
    base = baseline(root)
    rdir = BASE / "reports/SLA_HIERARCHICAL_V16"
    reg = read(rdir / PLAN)
    hist = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    pins = verify(base, hist)
    dry_receiver(base)
    build = read(rdir / "V2_RECEIVER_BUILD.json")
    so = root / "v16/v2-build03/student_grip.so"
    if (
        sha(so) != build["student_binary"]
        or sha(Path(base["outer_library"])) != build["rebuilt_control"]
        or sha(Path(__file__).with_name("student_receiver.hpp"))
        != build["student_header"]
    ):
        raise ValueError("unverified receiver")
    if reg["maximum_native_ticks"] != END or reg["card"] != "V2-GRIP-01B":
        raise ValueError("registration mismatch")
    dest = root / "v16" / reg["campaign"]
    os.umask(0o077)
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        if dest.exists():
            raise ValueError("registered sample already allocated")
        if mode == "--preflight":
            print(
                "verified frozen site, independent gripper build, world lock, original guards and receiver preflight"
            )
            return
        if mode != "--execute":
            raise ValueError("explicit execution required")
        dest.mkdir(mode=0o700)
        record(dest / "REGISTRATION.json", reg)
        fields = dict(
            outer_library=str(so),
            max_ticks=END,
            frozen_files={**base["frozen_files"], str(so): build["student_binary"]},
        )
        result = with_case_lease(
            dest / reg["card"],
            base,
            pins,
            sha(rdir / PLAN),
            fields,
            lambda spec, sp: execute(spec, sp, root, build),
        )
        print(
            result["outcome"],
            result["issued"],
            result.get("failure", "")[-620:],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    main(root, "--execute" if args.execute else "--preflight")
