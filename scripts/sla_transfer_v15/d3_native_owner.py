"""Run the registered 12 standing D3 paths with the existing native mobile adapter."""

import argparse
import hashlib
import json
import os
import time
import traceback
from contextlib import closing
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, rows, sha

PLAN_PIN = "7a63dd5cecb75620fa3ba28c9a02d1b207b8cbc1de922bd493349bbf3cd62819"


def episode(adapter, card, folder, anchors, limits):
    import numpy as np
    from b2_native import fixed, paused
    from d1_factorial_reference_matrix import (
        boundary,
        check,
        initial_features,
        verify_initial,
    )
    from d1_native_owner import native_prefix_pd
    from d3_known_path import KnownPath
    from d3_known_path_preview import sweep_path
    from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_OPEN_FRACTION
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    from scripts.pi0291_arx5_data.native_dataset import pose10

    adapter.prepare_dynamic()
    out = adapter.output
    check(
        read(out / "GUARD_READY.json")
        == {
            "episode": out.name,
            "pid": next(p.pid for name, p in adapter.processes if name == "planner"),
        },
        "guard owner",
    )
    mount = mobile.arm(adapter)
    sample = adapter.measured_task_state()
    action, base = fixed(sample, mount)
    for i in range(128):
        if i % 7 == 0:
            mobile.install(adapter, action, base)
        sample, _ = mobile.consume(adapter, action, sequence=i)
    paused(adapter, 5703)
    last = list(rows(out / "MOBILE_SAMPLES.private.jsonl"))[-1]
    parts, old, previous_sequence = boundary(out, last)
    feat = initial_features(
        sample, parts, native_prefix_pd(out / "init.trace"), last["installed_EEF10"][9]
    )
    check(
        sample["contact_valid"]
        and sample["safety_valid"]
        and not sample["robot_object_contact"],
        "initial contact",
    )
    record(
        folder / "INITIAL.private.json",
        {
            "features": feat,
            "old_wire": old,
            "original_B2_match": verify_initial(feat, anchors, limits),
            "old_sequence": previous_sequence,
        },
    )
    family, rate, mode = (card[k] for k in ("family", "rate", "mode"))
    sweep = sweep_path(
        adapter.actual_state.model,
        sample,
        parts,
        family,
        rate,
        binding(adapter.spec["native_root"]),
    )
    record(folder / "PATH_SWEEP.private.json", sweep)
    path = KnownPath(parts["old_ee"], family, rate, mode)
    base = np.repeat(parts["old_base"][None], 41, axis=0)
    for atom in range(40):
        if atom % 2 == 0:
            poses = path.window(atom)
            action = pose10(poses[:40], np.full(40, WORKING_OPEN_FRACTION))
            mobile.install(adapter, action, base)
            wire = adapter.active_reference["wire_sha256"]
            events = adapter.reference_events()
            raw = next(
                r["raw"]
                for r in events
                if r["kind"] == "accept" and r["ref_id"] == wire
            )
            doc = json.loads(raw)
            meta = doc["_pi0294"]
            check(
                hashlib.sha256(raw.encode()).hexdigest() == wire
                and meta["episode"] == out.name
                and meta["sequence"] == previous_sequence + 1 + atom // 2
                and meta["ttl_s"] == 0.4
                and abs(meta["source_mpc_s"] - (5703 + 25 * atom) * 0.002) < 1e-8,
                "D3 ownership/clock",
            )
            for key, expected in (("ee", poses), ("base", base)):
                pts = doc[key]["waypoints"]
                check(
                    doc[key]["frame"] == "world"
                    and len(pts) == 41
                    and np.allclose(
                        [p["time"] for p in pts],
                        np.arange(41) * 0.05,
                        rtol=0,
                        atol=1e-10,
                    )
                    and np.allclose(
                        [p["position"] for p in pts],
                        expected[:, :3, 3],
                        rtol=0,
                        atol=1e-10,
                    )
                    and np.allclose(
                        quat_to_matrix(np.array([p["quaternion_xyzw"] for p in pts])),
                        expected[:, :3, :3],
                        rtol=0,
                        atol=1e-8,
                    ),
                    "D3 installed geometry",
                )
        sample, _ = mobile.consume(adapter, action, sequence=128 + atom)
        check(
            sample["safety_valid"]
            and sample["contact_valid"]
            and not sample["robot_object_contact"]
            and not sample["native_grasp"]
            and not (out / "GUARD_TERMINAL.json").exists(),
            "no-contact control",
        )
    paused(adapter, 6703)
    cancel = adapter.cancel_reference()
    check(cancel["mode"] == "HOLD" and cancel["native_step"] == 6703, "terminal HOLD")
    adapter.seal_trace(6703)
    return {
        "cancel": cancel,
        "revisions": path.revisions,
        "installs": path.next_install,
    }


def run_case(spec, sp, card, anchors, limits, frozen):
    from b2_native import TechnicalAdapter, samples
    from clock_guard import new_receiver_output
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.native_replay import packets
    from scripts.wm02.mobile_boundary_runner_v1 import registered

    folder = sp.parent
    registered(spec, sp)
    out = new_receiver_output(folder)
    adapter = TechnicalAdapter(spec, out)
    result = {"card": card["id"], "episode": out.name, "completed": False}
    record(
        folder / "NATIVE_REQUEST.json",
        {"epoch": time.time(), "pid": os.getpid(), "epoch_id": out.name},
    )
    try:
        adapter.start()
        result.update(episode(adapter, card, folder, anchors, limits))
    except BaseException:  # noqa: BLE001 - retain interruptions and always close the native owner.
        result["failure"] = traceback.format_exc()
    finally:
        if adapter.node is not None:
            try:
                if adapter.processes and all(
                    p.poll() is None for _, p in adapter.processes
                ):
                    adapter.capture_loaded_identity()
                if not adapter.trace_sealed:
                    adapter.seal_trace(adapter.observe()["step"])
            except BaseException:  # noqa: BLE001 - retain interruptions and always close the native owner.
                result["seal_error"] = traceback.format_exc()
        try:
            adapter.close()
        except BaseException:  # noqa: BLE001 - retain interruptions and always close the native owner.
            result["close_error"] = traceback.format_exc()
    record(
        folder / "CLOSE.private.json",
        {"sealed": adapter.trace_sealed, "owned_exits": adapter.exit_records},
    )
    try:
        j = out / "state.bin.journal"
        result["ticks"] = len(packets(j)) if j.is_file() and j.stat().st_size else 0
        trace = audit_trace(out / "init.trace", max_native_ticks=spec["max_ticks"])
        result["trace"] = {
            k: trace[k]
            for k in (
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
                "status",
                "errors",
            )
        }
        result["physics_stops"] = sum(
            r["kind"] == "stop" for r in rows(out / "physics.jsonl")
        )
        result["guard_stop"] = (out / "MOBILE_SAFETY_STOP.private.json").exists()
        loaded = read(out / "LOADED_EXECUTOR_00.private.json")
        result["loaded"] = {k: loaded[k]["executable_sha256"] for k in frozen}
        m = samples(out / "MOBILE_SAMPLES.private.jsonl")
        result["atoms"] = m
        result["completed"] = bool(
            not any(k in result for k in ("failure", "seal_error", "close_error"))
            and adapter.trace_sealed
            and len(adapter.exit_records) == 4
            and all(r["exit_code"] == 0 for r in adapter.exit_records)
            and result["ticks"] == trace["native_step_records"] == 6703
            and trace["status"] == "RECORDED_TRACE_CHECKS_PASS"
            and not trace["STOP_observed"]
            and result["physics_stops"] == 0
            and not result["guard_stop"]
            and result["loaded"] == frozen
            and m.get("atoms")
            == m.get("valid_no_grasp_atoms")
            == m.get("atoms_with_exact_solver_echo")
            == 40
            and m.get("distinct_installs") == result["installs"] == 20
            and m["arm_requests"] > 0
            and m["base_commands"] > 0
            and result["revisions"] == (3 if card["mode"] == "P10_REVISION" else 0)
        )
    except BaseException:  # noqa: BLE001 - retain interruptions and always close the native owner.
        result["audit_error"] = traceback.format_exc()
    record(folder / "RESULT.private.json", result)
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("plan", type=Path)
    cli.add_argument("--d1-readout", type=Path)
    cli.add_argument("--d1-sha256")
    cli.add_argument("--cpu-only", action="store_true")
    a = cli.parse_args()
    root = a.root.resolve()
    base = baseline(root)
    configure(base)
    from b2_owner import check_occupancy, dry_receiver, verify
    from d1_factorial_reference_matrix import check
    from d1_matched_owner import with_case_lease

    from scripts.pi0292.durable import OwnerLock

    pinned(a.plan, PLAN_PIN)
    plan = read(a.plan)
    preview = root / "v15/d3-path-preview03/D3_KNOWN_PATH_PREVIEW.json"
    pinned(preview, plan["preview_sha256"])
    reg_path = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pinned(reg_path, plan["d1_original_register_sha256"])
    reg = read(reg_path)
    pub = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    pinned(pub, B2_PIN)
    public = read(pub)
    for name, h in reg["source_sha256"].items():
        pinned(Path(__file__).with_name(name), h)
    for name, h in public["experiment_code_sha256"].items():
        pinned(Path(__file__).resolve().parent.parent / "sla_transfer_v14" / name, h)
    historic = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    pins = verify(base, historic)
    dry_receiver(base)
    anchors = []
    rr = read(root / "v15/readiness01/D1_READINESS_ACTUAL.json")
    for r in rr["anchors"]:
        f = reg_path.parent / (r["source"] + "_anchor.private.json")
        pinned(f, r["private_anchor_sha256"])
        check(
            r["private_anchor_sha256"] in reg["prior_anchor_sha256"],
            "anchor registration",
        )
        anchors.append(read(f)["initial_features"])
    if a.cpu_only:
        print("CPU readiness only. No new native requests.", flush=True)
        return
    pinned(a.d1_readout, a.d1_sha256)
    d1 = read(a.d1_readout)
    check(
        d1["registration_sha256"] == plan["d1_registration_sha256"]
        and d1["terminal"] == "FIXED_ROSTER_EXHAUSTED"
        and d1["starts"] == d1["registered_starts"] == 40
        and len(d1["blocks"]) >= 2,
        "D1 science gate",
    )
    d2 = root / "v15/pd-phase03/D2_PD_SENSOR_PHASE_ACTUAL.json"
    pinned(d2, plan["d2_pd_sha256"])
    stage = root / "v15" / plan["campaign"]
    lock = Path(base["mobile_world_lock"])
    os.umask(0o077)
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(lock)):
        check_occupancy(base)
        stage.mkdir(mode=0o700)
        record(stage / "REGISTRATION.json", plan)
        record(
            stage / "OWNER.json",
            {
                "epoch": time.time(),
                "pid": os.getpid(),
                "registration_sha256": PLAN_PIN,
                "d1_readout_sha256": a.d1_sha256,
                "code": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            },
        )
        actual = []
        for card in plan["roster"]:
            check(
                not any(
                    (p / "STOP.json").exists()
                    for p in (stage, Path(base["campaign_root"]))
                ),
                "operator STOP",
            )
            check_occupancy(base)
            fields = {
                "scope": "D3_STANDING_KNOWN_PATH_SIM_ONLY",
                "max_ticks": 6703,
                "mobile_condition": card,
            }

            def execute(spec, sp, card=card):
                return run_case(
                    spec,
                    sp,
                    card,
                    anchors,
                    reg["pairwise_initial_limits"],
                    public["frozen_sha256"],
                )

            val = with_case_lease(
                stage / card["id"], base, pins, PLAN_PIN, fields, execute
            )
            actual.append(
                {
                    k: val.get(k)
                    for k in (
                        "card",
                        "completed",
                        "ticks",
                        "physics_stops",
                        "guard_stop",
                    )
                }
            )
            print(actual[-1], flush=True)
            if not val["completed"]:
                break
        record(
            stage / "PANEL_ACTUAL.json",
            {
                "actual": actual,
                "planned": 12,
                "completed": sum(x["completed"] for x in actual),
                "native_ticks": sum(
                    x["ticks"] for x in actual if x["ticks"] is not None
                ),
            },
        )


if __name__ == "__main__":
    main()
