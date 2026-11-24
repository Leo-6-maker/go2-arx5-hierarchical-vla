"""Site owner for the three allocated #482 MuJoCo qualification slots."""

import argparse
import csv
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from contextlib import closing
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from moving_stop_card import (
    check_occupancy,
    configure,
    groups_alive,
    receiver_adapter,
    record,
    sha,
)
from reach_profile import BASE, SLOTS

REPORT = BASE / "reports/SLA_GO2_V18/REACH"
SLOTS_E1 = ("G00-STOP", "G00-H0", "G01-H0")


def sites(root):
    profile = read(REPORT / "PROFILE_ACTUAL.json")
    home = Path(
        read(root / "evidence/P1_REGISTRATION.json")["target_windows"][0][
            "episode_pin"
        ]["path"]
    ).parents[3]
    site = read(root / "v17/known-dynamic-472/SLOW_A0/EXECUTION_SPEC.json")
    domain = site["ROS_DOMAIN_ID"]
    lock = Path(site["mobile_world_lock"])
    require(
        domain == 193 and lock.name == "ros-193.lock", "transport reservation binding"
    )

    def bind(spec):
        require(
            site["registry"] == spec["registry"]
            and site["host"] == spec["host"]
            and lock.parent == Path(spec["mobile_world_lock"]).parent,
            "different simulator resource registry",
        )
        return dict(
            spec,
            ROS_DOMAIN_ID=domain,
            mobile_world_lock=str(lock),
            native_environment={
                **spec["native_environment"],
                "ROS_DOMAIN_ID": str(domain),
            },
        )

    return [
        (
            r,
            bind(
                read(
                    pin(
                        home / "cases" / name / "EXECUTION_SPEC.json",
                        r["source_spec_sha256"],
                    )
                )
            ),
        )
        for name, r in zip(SLOTS, profile["results"])
    ]


def adapter_class():
    from b2_native import TechnicalAdapter
    from go2arx5_vla.world_model.multitask_adapter_v6 import MultitaskAdapter

    class StandingAdapter(TechnicalAdapter, MultitaskAdapter):
        pass

    return StandingAdapter


def preflight(root):
    import mujoco
    from b2_owner import dry_receiver
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from known_dynamic_panel import KnownCurve
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile
    from standing_reach import StateMonitor, knots

    from scripts.pi0292.eval_contract import matrix10

    klass = adapter_class()
    receiver = pin(
        root / "v18/receiver-preview02/student_grip.so",
        read(REPORT / "RECEIVER_CPU.json")["receiver_sha256"],
    )
    pins = read(root / "v17/known-dynamic-472/SLOW_A0/EXECUTION_SPEC.json")["b2_code"]
    for n, h in pins.items():
        pin(BASE / "scripts/sla_transfer_v14" / n, h)
    samples = []
    checked_planners = set()
    for r, spec in sites(root):
        for module, relative in (
            (mobile, "scripts/wm02/mobile_boundary_runner_v1.py"),
            (
                sys.modules[klass.__mro__[2].__module__],
                "src/go2arx5_vla/world_model/multitask_adapter_v6.py",
            ),
        ):
            pin(Path(module.__file__), spec["source_files"][relative])
        require(
            spec["physics_dt_s"] == 0.002 and spec["task_instruction_id"] == "HOLD-red",
            "task profile",
        )
        if spec["source_sha"] not in checked_planners:
            dry_receiver(spec)
            checked_planners.add(spec["source_sha"])
        old = Path(spec["registered_root"]) / "native-01"
        model = mujoco.MjModel.from_binary_path(
            str(pin(Path(spec["model_path"]), r["compiled_scene"]))
        )
        mon = StateMonitor(model)
        guard = read(REPORT / "RECEIVER_CPU.json")
        require(
            all(
                t["delivered_after_receiver_stop"] == 0
                for c in guard["results"]
                for t in c["treatments"]
                if t["method"] != "VALID"
            ),
            "missing STOP rehearsal",
        )
        with forbid_integration(mujoco) as calls:
            pkt = next(
                x for x in packets(old / "state.bin.journal") if x["step"] == 5703
            )
            mon.observe(pkt)
            a = next(
                x
                for x in map(json.loads, (old / "MOBILE_SAMPLES.private.jsonl").open())
                if x["native_end_step"] == 5703
            )
            anchor = matrix10(np.asarray(a["target_endpoint_EEF10"]))
            previous = matrix10(np.asarray(a["installed_EEF10"]))
            direction = np.asarray(r["candidate_standing_translation_world_m"]) / 0.020
            for i in range(41):
                v = knots(anchor, direction, i / 10)
                prior = (
                    previous if i == 0 else knots(anchor, direction, (i - 1) / 10)[1]
                )
                KnownCurve.admit(
                    v,
                    {
                        "start_pose": prior,
                        "end_pose": v[0],
                        "finger_fraction": a["target_endpoint_EEF10"][9],
                    },
                    {"tcp": np.array(a["actual_TCP"])},
                )
            require(not calls, "unexpected integration")
        samples.append(
            {
                "layout": r["episode"],
                "tick": pkt["step"],
                "source": spec["source_sha"],
                "windows": 41,
                "ROS_DOMAIN_ID": spec["ROS_DOMAIN_ID"],
            }
        )
    return (
        receiver,
        pins,
        {
            "receiver_sha256": sha(receiver),
            "samples": samples,
            "task_spec_sha256": sha(REPORT / "SCIENCE_REQUEST.json"),
            "owner_code_sha256": sha(Path(__file__)),
            "motion_code_sha256": sha(Path(__file__).with_name("standing_reach.py")),
            "native_integrations": 0,
        },
    )


def audit(case, r, spec):
    import mujoco
    from go2arx5_vla.world_model.init_trace_reader import audit_trace, read_trace
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from stop_card_evidence import terminal_evidence

    out = case / r["episode"]
    model = mujoco.MjModel.from_binary_path(spec["model_path"])
    abi = (model.nq, model.nv, model.nu)
    require(abi == (41, 38, 20), "wrong scene state ABI")
    trace_a = audit_trace(out / "init.trace", max_native_ticks=20000, state_abi=abi)
    ticks = sum(1 for _ in packets(out / "state.bin.journal"))
    require(
        ticks == trace_a["native_step_records"] <= 20000 and r["absent"],
        "journal or live owner",
    )
    for key, role in (
        ("native_binary", "sim"),
        ("solver_binary", "solver"),
        ("controller_binary", "controller"),
    ):
        require(
            read(out / "LOADED_EXECUTOR_00.private.json")[role]["executable_sha256"]
            == spec["frozen_files"][spec[key]],
            "executor drift",
        )
    summary = {
        "ticks": ticks,
        "objects": 2,
        "after_stop_integrations": trace_a["integrations_after_first_STOP"],
        "exits": r["exits"],
        "no_survivors": r["absent"],
    }
    if case.name != "G00-STOP":
        require(
            not trace_a["STOP_observed"]
            and trace_a["status"] == "RECORDED_TRACE_CHECKS_PASS"
            and r["outcome"] == "H0_PASS"
            and r["score"]["passed"]
            and ticks == r["end"]
            and all(x["exit_code"] == 0 for x in r["exits"]),
            "H0 task failed or incomplete",
        )
        return {**summary, "qualified": True, "task": r["score"]}
    entry = read(case / "REACH_ENTRY.json")["tick"]
    source = entry + 1000
    stop = source + 11
    grants = list(map(json.loads, (out / "CLOCK_RESUME_REQUESTS.jsonl").open()))
    planner = list(map(json.loads, (out / "planner.jsonl").open()))
    installed = [x for x in planner if x["kind"] == "installed"]
    term = terminal_evidence(
        read_trace(out / "init.trace"),
        trace_a,
        grants[-1],
        installed,
        r["exits"],
        r["absent"],
        stop_tick=stop,
        grant_end=source + 25,
        sequence=(source - 2503) // 25,
    )
    require(
        r["last_requested_tick"] == source + 25 and ticks == stop,
        "unexpected terminal grant",
    )
    recv = list(csv.reader((case / "student/receiver.csv").open()))
    stops = [x for x in recv if "STOP" in x[0]]
    require(
        len(stops) == 1
        and stops[0][0] == "RECEIVER_REJECT_STOP"
        and int(stops[0][2]) == stop,
        "receiver fault identity",
    )
    steps = [int(x[2]) for x in recv if x[0] == "NATIVE_STEP"]
    require(steps == list(range(1, stop + 1)), "native receiver tick gap")
    last = json.loads((case / "REACH_WIRES.jsonl").read_text().splitlines()[-1])
    require(
        last["tick"] == source and any(x["ref_id"] == last["wire"] for x in installed),
        "no in-flight reference",
    )
    data = mujoco.MjData(model)
    poses = {}
    for p in packets(out / "state.bin.journal"):
        if p["step"] >= source - 25:
            data.qpos[:] = p["qpos"]
            mujoco.mj_kinematics(model, data)
            poses[p["step"]] = data.site("x5_ee").xpos.copy()
    actual = float(1000 * np.linalg.norm(poses[source] - poses[source - 25]))
    terminal_motion = float(1000 * np.linalg.norm(poses[stop] - poses[stop - 25]))
    ref = float(1000 * np.linalg.norm(np.asarray(last["next"]) - last["start"]))
    require(
        len(poses) == stop - source + 26 and actual > 0.005 and ref > 0.01,
        "no measurable TCP or pending nominal motion",
    )
    return {
        **summary,
        "qualified": True,
        "terminal": term,
        "source_tick": source,
        "actual_50ms_mm": actual,
        "terminal_50ms_mm": terminal_motion,
        "state_abi": list(abi),
        "trace_writer": trace_a["writer"],
        "next_100ms_reference_mm": ref,
        "after_stop_installs": 0,
    }


def execute(base, sp, spec, profile):
    from clock_guard import new_receiver_output
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile
    from standing_reach import drive

    case = sp.parent
    mobile.registered(spec, sp)
    out = new_receiver_output(case)
    channel = case / "student"
    channel.mkdir(mode=0o700)
    epoch = secrets.token_hex(32)
    record(
        channel / "authority.json",
        {
            "epoch": epoch,
            "owner": spec["owner"],
            "episode": out.name,
            "receiver_sha256": sha(Path(spec["outer_library"])),
        },
    )
    r = {
        "card": case.name,
        "episode": out.name,
        "last_requested_tick": 0,
        "outcome": "TECHNICAL_STOP",
    }
    fault = {}
    a = receiver_adapter(
        base,
        spec,
        case,
        out,
        channel,
        epoch,
        r,
        grant_cap=20000,
        adapter_type=adapter_class(),
        fault_schedule=fault,
    )
    watch = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,signal,sys,time; time.sleep(float(sys.argv[2])); os.kill(int(sys.argv[1]),signal.SIGINT)",
            str(os.getpid()),
            str(min(900, spec["deadline_epoch"] - time.time())),
        ]
    )
    try:
        record(
            case / "NATIVE_REQUEST.json",
            {"episode": out.name, "max_ticks": 20000, "card": case.name},
        )
        a.start()
        drive(a, case, case.name, profile, fault, r)
    except BaseException:  # noqa: BLE001 - retain interruption and always release this owner.
        r["interrupt"] = traceback.format_exc()
    finally:
        try:
            a.close()
        except BaseException:  # noqa: BLE001 - retain interruption and always release this owner.
            r["close_error"] = traceback.format_exc()
        watch.terminate()
        watch.wait(timeout=5)
        r["exits"] = a.exit_records
        r["survivors"] = groups_alive({p.pid for _, p in a.processes})
        r["absent"] = (
            not r["survivors"]
            and bool(r.get("sim_child_pid"))
            and not Path("/proc", str(r.get("sim_child_pid"))).exists()
        )
        record(case / "RESULT.private.json", r)
    try:
        verdict = audit(case, r, spec)
    except Exception:  # noqa: BLE001 - seal incomplete science and stop later slots.
        from go2arx5_vla.world_model.multitask_state_v6 import packets

        j = out / "state.bin.journal"
        verdict = {
            "qualified": False,
            "ticks": sum(1 for _ in packets(j)) if j.exists() else 0,
            "readout_error": traceback.format_exc(),
            "interruption": r.get("interrupt"),
            "no_survivors": r["absent"],
        }
    record(case / "AUDIT.json", verdict)
    return verdict


def main(args):
    root = args.root.resolve()
    rows = sites(root)
    scratch = root / "v18/owner-scratch"
    scratch.mkdir(exist_ok=True, mode=0o700)

    def runtime_env(base, dest):
        return {
            k: (
                str(dest / "tmp")
                if k == "TMPDIR"
                else str(dest / "cache")
                if k == "XDG_CACHE_HOME"
                else str(dest / "logs")
                if k == "ROS_LOG_DIR"
                else v
            )
            for k, v in base["native_environment"].items()
        }

    for leaf in ("tmp", "cache", "logs"):
        (scratch / leaf).mkdir(exist_ok=True, mode=0o700)
    configure(dict(rows[0][1], native_environment=runtime_env(rows[0][1], scratch)))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    os.umask(0o077)
    receiver, pins, proof = preflight(root)
    if args.preflight:
        record(args.preflight, proof)
        return
    prior = root / "v18/Q_E1_482_d193"
    prior_case = prior / SLOTS_E1[0]
    close = read(prior / "PANEL_CLOSE.json")
    require(
        close["unstarted"] == list(SLOTS_E1[1:])
        and len(close["attempts"]) == 1
        and read(prior_case / "control/LEASE.json")["status"] == "CLOSED"
        and not any((prior / slot).exists() for slot in SLOTS_E1[1:]),
        "initial panel not closed with two unused H0 slots",
    )
    prior_verdict = audit(
        prior_case,
        read(prior_case / "RESULT.private.json"),
        read(prior_case / "EXECUTION_SPEC.json"),
    )
    if args.reaudit_stop:
        record(args.reaudit_stop, prior_verdict)
        return
    from d1_matched_owner import with_case_lease

    from scripts.pi0292.durable import OwnerLock

    allocation = read(args.execute)
    require(
        allocation["issue"] == 482
        and allocation["issued"]
        and allocation["slots"] == list(SLOTS_E1)
        and allocation["max_ticks"] == 60000
        and read(
            pin(root / "v18/owner-preflight-h0.json", allocation["preflight_sha256"])
        )
        == proof,
        "allocation or preflight mismatch",
    )
    first = root / "v18/Q_E1_482"
    closure = read(first / "PANEL_CLOSE.json")
    require(
        not closure["attempts"]
        and closure["unstarted"] == list(SLOTS_E1)
        and not any(first.glob("*/NATIVE_REQUEST.json"))
        and not any(first.glob("*/control/LEASE.json")),
        "existing scientific request or owner",
    )
    require(
        read(
            pin(root / "v18/stop-reaudit.json", allocation["closed_stop_audit_sha256"])
        )
        == prior_verdict,
        "STOP regrade differs",
    )
    dest = root / "v18/Q_E1_482_H0"
    dest.mkdir(mode=0o700, exist_ok=False)
    record(dest / "REGISTRATION.json", allocation)
    results = [prior_verdict]
    try:
        for slot in SLOTS_E1[1:]:
            profile, base = rows[int(slot.startswith("G01"))]
            os.sched_setaffinity(0, base["cpus"])
            with closing(OwnerLock(Path(base["mobile_world_lock"]))):
                check_occupancy(base)
                require(
                    not any(
                        (p / "STOP.json").exists()
                        for p in (dest, Path(base["campaign_root"]))
                    ),
                    "operator stop",
                )
                fields = {
                    "scope": "#482_TWO_OBJECT_STANDING_REACH",
                    "max_ticks": 20000,
                    "campaign_root": str(dest),
                    "outer_library": str(receiver),
                    "frozen_files": {
                        **base["frozen_files"],
                        str(receiver): sha(receiver),
                    },
                }
                case = dest / slot
                fields["native_environment"] = runtime_env(base, case)
                v = with_case_lease(
                    case,
                    base,
                    pins,
                    sha(dest / "REGISTRATION.json"),
                    fields,
                    lambda spec, sp, base=base, profile=profile: execute(
                        base, sp, spec, profile
                    ),
                )
            require(
                read(case / "control/LEASE.json")["status"] == "CLOSED", "open lease"
            )
            results.append(v)
            print(slot, v["qualified"], v["ticks"], flush=True)
            if not v["qualified"]:
                break
    finally:
        record(
            dest / "PANEL_CLOSE.json",
            {
                "slots": list(SLOTS_E1),
                "attempts": results,
                "unstarted": list(SLOTS_E1[len(results) :]),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", type=Path)
    group.add_argument("--preflight", type=Path)
    group.add_argument("--reaudit-stop", type=Path)
    main(p.parse_args())
