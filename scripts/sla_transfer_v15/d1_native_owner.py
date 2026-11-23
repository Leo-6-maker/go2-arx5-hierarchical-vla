"""One-use, registered native D1 waves. All motion stays inside the commissioned simulator."""

import argparse
import hashlib
import json
import os
import time
import traceback
from contextlib import closing
from pathlib import Path

from d1_readiness import (
    B2_PIN,
    PLAN,
    PLAN_PIN,
    SRC,
    baseline,
    configure,
    pinned,
    read,
    rows,
    sha,
)


def native_prefix_pd(path):
    import struct

    from go2arx5_vla.world_model.init_trace_reader import RECORD

    deadline = time.monotonic() + 5
    with path.open("rb") as f:
        head = f.read(16)
        if head[:8] != b"WM02IT01" or struct.unpack("<I", head[8:12])[0] != RECORD.size:
            raise ValueError("native trace ABI")
        previous = None
        ticket = 0
        while time.monotonic() < deadline:
            pos = f.tell()
            raw = f.read(RECORD.size)
            if len(raw) < RECORD.size:
                f.seek(pos)
                time.sleep(0.015)
                continue
            r = RECORD.unpack(raw)
            if r[0] != ticket:
                raise ValueError("missing native trace ticket")
            ticket += 1
            if r[3] == 9 and r[4] == 90:
                previous = list(r[22:28])
            if r[3] == 11 and r[9] == 5703 and previous is not None:
                return previous
    raise ValueError("missing physical PD target at measured boundary")


def episode(adapter, registration, prior, folder):
    import mujoco
    import numpy as np
    from b2_native import fixed as initialization_hold
    from b2_native import paused
    from d1_factorial_reference_matrix import (
        boundary,
        check,
        fixed,
        held_part,
        initial_features,
        verify_initial,
    )
    from d1_geometry import sweep
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import matrix10

    adapter.prepare_dynamic()
    out = adapter.output
    check(
        read(out / "GUARD_READY.json")
        == {
            "episode": out.name,
            "pid": next(p.pid for n, p in adapter.processes if n == "planner"),
        },
        "new guarded receiver identity",
    )
    mount = mobile.arm(adapter)
    sample = adapter.measured_task_state()
    action, base = initialization_hold(sample, mount)
    for i in range(128):
        if i % 7 == 0:
            mobile.install(adapter, action, base)
        sample, mount = mobile.consume(adapter, action, sequence=i)
    paused(adapter, 5703)
    last = list(rows(out / "MOBILE_SAMPLES.private.jsonl"))[-1]
    parts, old, seq = boundary(out, last)
    feat = initial_features(
        sample, parts, native_prefix_pd(out / "init.trace"), last["installed_EEF10"][9]
    )
    match = verify_initial(feat, prior, registration["pairwise_initial_limits"])
    write(
        folder / "INITIAL_MATCH.private.json",
        {"features": feat, "evaluation": match, "prior_wire": old},
    )
    check(match["matched"], "UNKNOWN_INITIAL_MATCH:" + json.dumps(match["violations"]))
    action, base = fixed(adapter.spec["d1_factor"], parts)
    ee = matrix10(action[0])
    with forbid_integration(mujoco) as calls:
        geo = sweep(
            adapter.actual_state.model,
            sample,
            parts,
            ee,
            base[0],
            binding(adapter.spec["native_root"]),
        )
    check(not calls, "sweep attempted integration")
    write(
        folder / "D1_REF.private.json",
        {"parts": {k: v.tolist() for k, v in parts.items()}, "geometry": geo},
    )
    for i in range(20):
        if i % 2 == 0:
            mobile.install(adapter, action, base)
            ref = adapter.active_reference
            doc = json.loads(
                next(
                    e["raw"]
                    for e in adapter.reference_events()
                    if e["kind"] == "accept" and e["ref_id"] == ref["wire_sha256"]
                )
            )
            m = doc["_pi0294"]
            check(
                m["sequence"] == seq + 1 + i // 2
                and m["episode"] == out.name
                and m["ttl_s"] == 0.4
                and np.allclose(held_part(doc, "ee"), ee, rtol=0, atol=1e-9)
                and np.allclose(held_part(doc, "base"), base[0], rtol=0, atol=1e-9),
                "reference content/ownership",
            )
        sample, _ = mobile.consume(adapter, action, sequence=128 + i)
        check(
            sample["safety_valid"]
            and not sample["native_grasp"]
            and not (out / "GUARD_TERMINAL.json").exists(),
            "native safety or receiver fault",
        )
    paused(adapter, 6203)
    cancel = adapter.cancel_reference()
    adapter.seal_trace(6203)
    check(
        cancel["mode"] == "HOLD" and cancel["native_step"] == 6203,
        "HOLD acknowledgement",
    )
    return cancel


def run_case(spec, sp, registration, prior):
    from b2_native import TechnicalAdapter, samples
    from clock_guard import new_receiver_output
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.native_replay import packets
    from scripts.wm02.mobile_boundary_runner_v1 import registered

    from scripts.pi0292.durable import write

    folder = sp.parent
    registered(spec, sp)
    out = new_receiver_output(folder)
    adapter = TechnicalAdapter(spec, out)
    value = {
        "case": spec["d1_case"],
        "factor": spec["d1_factor"],
        "episode": out.name,
        "new_native_requests": 1,
        "cap": spec["max_ticks"],
        "completed": False,
        "training_updates": 0,
        "gpu_forwards": 0,
    }
    write(
        folder / "NATIVE_REQUEST.json",
        {"epoch": time.time(), "pid": os.getpid(), "epoch_id": out.name},
    )
    try:
        adapter.start()
        value["cancel"] = episode(adapter, registration, prior, folder)
    except BaseException:
        value["failure"] = traceback.format_exc()
    finally:
        if adapter.node is not None:
            try:
                if adapter.processes and all(
                    p.poll() is None for _, p in adapter.processes
                ):
                    adapter.capture_loaded_identity()
                if not adapter.trace_sealed:
                    adapter.seal_trace(adapter.observe()["step"])
            except BaseException:
                value["seal_error"] = traceback.format_exc()
        try:
            adapter.close()
        except BaseException:
            value["close_error"] = traceback.format_exc()
    write(
        folder / "CLOSE.private.json",
        {"owned_exits": adapter.exit_records, "trace_sealed": adapter.trace_sealed},
    )
    try:
        journal = out / "state.bin.journal"
        value["ticks"] = (
            len(packets(journal)) if journal.is_file() and journal.stat().st_size else 0
        )
        trace = audit_trace(out / "init.trace", max_native_ticks=6203)
        value["trace"] = {
            k: trace[k]
            for k in (
                "status",
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
                "errors",
            )
        }
        value["physics_stops"] = sum(
            r["kind"] == "stop" for r in rows(out / "physics.jsonl")
        )
        value["guard_stop"] = (out / "MOBILE_SAFETY_STOP.private.json").exists()
        m = samples(out / "MOBILE_SAMPLES.private.jsonl")
        value["measured"] = m
        identity = read(out / "LOADED_EXECUTOR_00.private.json")
        frozen = {
            role: identity[role]["executable_sha256"]
            for role in ("sim", "controller", "solver")
        }
        value["loaded"] = frozen
        value["completed"] = bool(
            "failure" not in value
            and "close_error" not in value
            and "seal_error" not in value
            and frozen == registration["frozen_executables"]
            and not value["guard_stop"]
            and value["physics_stops"] == 0
            and trace["status"] == "RECORDED_TRACE_CHECKS_PASS"
            and value["ticks"] == trace["native_step_records"] == 6203
            and adapter.trace_sealed
            and len(adapter.exit_records) == 4
            and all(r["exit_code"] == 0 for r in adapter.exit_records)
            and m.get("atoms")
            == m.get("valid_no_grasp_atoms")
            == m.get("atoms_with_exact_solver_echo")
            == 20
            and m.get("distinct_installs") == 10
            and m.get("arm_requests", 0) > 0
            and m.get("base_commands", 0) > 0
        )
    except BaseException:
        value["audit_error"] = traceback.format_exc()
    write(folder / "RESULT.private.json", value)
    return value


def run_wave(root, stage, wave, readiness, auth):
    from b2_owner import check_occupancy, dry_receiver, verify
    from d1_factorial_reference_matrix import check

    from scripts.pi0292.durable import OwnerLock, write

    base = baseline(root)
    public_path = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    pinned(public_path, B2_PIN)
    public = read(public_path)
    pinned(SRC / PLAN, PLAN_PIN)
    rp = readiness / "D1_FACTORIAL_4X2_REGISTER.json"
    rr = read(readiness / "D1_READINESS_ACTUAL.json")
    pinned(rp, auth["registration_sha256"])
    pinned(rp, rr["registration_sha256"])
    reg = read(rp)
    check(
        rr["cpu_ready"]
        and reg["plan_sha256"] == PLAN_PIN
        and rr["frozen_sha256"] == public["frozen_sha256"],
        "CPU registration",
    )
    for name, h in reg["source_sha256"].items():
        pinned(Path(__file__).with_name(name), h)
    for name, h in public["experiment_code_sha256"].items():
        pinned(SRC / "scripts/sla_transfer_v14" / name, h)
    reg["frozen_executables"] = rr["frozen_sha256"]
    historical = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    runtime_pins = verify(base, historical)
    dry_receiver(base)
    selected = [k for k in reg["roster"] if k["wave"] == wave]
    check(
        auth["wave"] == wave
        and auth["requests"] == len(selected) == 4
        and auth["tick_cap"] == sum(k["cap"] for k in selected) == 24812,
        "unauthorized population",
    )
    prior = []
    for card in rr["anchors"]:
        path = readiness / (card["source"] + "_anchor.private.json")
        pinned(path, card["private_anchor_sha256"])
        prior.append(read(path)["initial_features"])
    if wave == "wave_B":
        wa = stage / "wave_A"
        proof = wa / "WAVE_READOUT.private.json"
        pinned(proof, auth["wave_A_readout_sha256"])
        check(read(proof)["promotion_ready"], "wave A failed the promotion gate")
        prior.extend(
            read(wa / k["id"] / "INITIAL_MATCH.private.json")["features"]
            for k in reg["roster"]
            if k["wave"] == "wave_A"
        )
    group = stage / wave
    lock = Path(base["mobile_world_lock"])
    registry = Path(base["registry"])
    os.sched_setaffinity(0, base["cpus"])
    os.umask(0o077)
    with closing(OwnerLock(lock)):
        check_occupancy(base)
        group.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(
            group / "OWNER.json",
            {
                "host": base["host"],
                "domain": base["ROS_DOMAIN_ID"],
                "pid": os.getpid(),
                "epoch": time.time(),
                "registration_sha256": sha(rp),
                "authorization": auth,
                "source_sha256": {
                    p.name: sha(p) for p in Path(__file__).parent.glob("*.py")
                },
            },
        )
        for k in selected:
            folder = group / k["id"]
            check_occupancy(base)
            check(
                not any(
                    (p / "STOP.json").exists()
                    for p in (group, stage, Path(base["campaign_root"]))
                ),
                "operator STOP",
            )
            for sub in ("control", "tmp", "logs", "cache", "data"):
                (folder / sub).mkdir(parents=True, mode=0o700)
            case = k["id"]
            uid = stage.name + "-" + case
            factor = k["factor"].removeprefix("D1-")
            spec = dict(
                base,
                role=uid,
                owner=uid,
                registered_root=str(folder),
                mobile_group=str(group),
                scope="D1_STANDING_FACTORIAL_SIM_ONLY",
                d1_case=case,
                d1_factor=factor,
                b2_card=case,
                b2_code=runtime_pins,
                max_ticks=6203,
                deadline_epoch=time.time() + base["wall_limit_s"],
                stage_identity_sha256=hashlib.sha256(
                    (sha(rp) + uid).encode()
                ).hexdigest(),
                mobile_condition={"factor": factor, "wave": wave},
                **{n: str(folder / n) for n in ("logs", "tmp", "cache", "data")},
            )
            sp = folder / "EXECUTION_SPEC.json"
            write(sp, spec)
            lp = folder / "control/LEASE.json"
            lease = {
                "status": "ACTIVE",
                "spec_sha256": sha(sp),
                "owner": uid,
                "run_uuid": uid,
                "deadline_epoch": spec["deadline_epoch"],
            }
            with closing(OwnerLock(lock.parent / "registration.lock")):
                check_occupancy(base)
                regdb = read(registry)
                check(
                    not any(
                        r.get("path") == str(folder) or r.get("run_uuid") == uid
                        for r in regdb["entries"]
                    ),
                    "prior registered request",
                )
                regdb["entries"].append(
                    {
                        "path": str(folder),
                        "run_uuid": uid,
                        "role": uid,
                        "source_sha": spec["source_sha"],
                        "owner": uid,
                        "scope": spec["scope"],
                        "executor_node": spec["host"],
                        "local_lease": str(lp),
                        "ROS_DOMAIN_ID": spec["ROS_DOMAIN_ID"],
                        "corelab_world_owner": str(lock),
                    }
                )
                write(registry, regdb)
                write(lp, lease)
            try:
                result = run_case(spec, sp, reg, prior)
            finally:
                write(lp, dict(lease, status="CLOSED", closed_epoch=time.time()))
            print(
                json.dumps(
                    {
                        name: result.get(name)
                        for name in (
                            "case",
                            "completed",
                            "ticks",
                            "failure",
                            "audit_error",
                        )
                    }
                ),
                flush=True,
            )
            if not result["completed"]:
                return 1
            prior.append(read(folder / "INITIAL_MATCH.private.json")["features"])
    return 0


def main():
    cli = argparse.ArgumentParser()
    for k in ("root", "readiness", "campaign", "authorization"):
        cli.add_argument(k, type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    auth = read(a.authorization)
    return run_wave(
        root, a.campaign.resolve(), auth["wave"], a.readiness.resolve(), auth
    )


if __name__ == "__main__":
    raise SystemExit(main())
