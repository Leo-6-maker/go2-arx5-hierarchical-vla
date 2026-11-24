"""Bind the two standing H0 cards to the existing site owner and finite lease."""

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from sea_fruit_h0_cold import BASE, DEV_LAYOUTS, pin, read, require, sha

for part in ("sla_transfer_v14", "sla_transfer_v15", "sla_transfer_feedback"):
    sys.path.insert(0, str(BASE / "scripts" / part))

from reach_native import check_occupancy, configure, execute, record, sites
from reach_pair import environment
from d1_factorial_reference_matrix import held_part
from sea2_native_profile_preflight import stop_predecessor
from sea2_preflight import sealed_archive
from sea_fruit_h0_native import drive, grip_at_grant, verify_fixed_base
from sea_fruit_receiver import CAP

REPORT = BASE / "reports/SLA_GO2_V193/GRIP_H0"
CAMPAIGN = "V193-STANDING-FRUIT-H0-01"
SLOTS = ("G00-H0", "G01-H0")
LIMITS = dict(native_starts=2, ticks_per_start=CAP, total_ticks=2*CAP,
              mujoco_only=True, contact=True, frozen_model_forwards=0, training_updates=0)
RUNTIME = ("native_binary", "solver_binary", "controller_binary", "observer_library",
           "mobile_safety_library", "trace_library")


def site_binding(root, fruit, receiver_root):
    scene_receipt = read(REPORT / "FRUIT_SCENES_CPU.json")
    require(read(pin(fruit / "RESULT.json", sha(REPORT / "FRUIT_SCENES_CPU.json"))) == scene_receipt,
            "scene receipt changed")
    receiver_proof = REPORT / "FRUIT_RECEIVER_CPU.json"
    proof = read(pin(receiver_root / "RESULT.json", sha(receiver_proof)))
    driver = read(REPORT / "FRUIT_DRIVER_CPU.json")
    require(proof["status"] == "CPU_RECEIVER_REPLAY_VERIFIED" and proof["card_tick_cap"] == CAP
            and driver["status"] == "CPU_H0_DRIVER_SEAMS_VERIFIED"
            and driver["receiver_proof_sha256"] == sha(receiver_proof),
            "CPU evidence does not identify this finite receiver")
    receiver = pin(receiver_root / "fruit_grip.so", proof["receiver_sha256"])
    sealed = sealed_archive(root)
    stop = stop_predecessor(root, sealed)
    receipt = read(sealed(root / "v18/Z1_REACH_483/G00-A/EXECUTION_SPEC.json"))
    runtime_pins = receipt["b2_code"]
    for name, digest in runtime_pins.items():
        pin(BASE / "scripts/sla_transfer_v14" / name, digest)
    rows = []
    for role, layout, (_, base), record_ in zip(
            SLOTS, DEV_LAYOUTS, sites(root), scene_receipt["layouts"], strict=True):
        require(record_["layout"] == layout and record_["development_only"]
                and base["task_instruction_id"] == "HOLD-red" and base["physics_dt_s"] == .002
                and base["mobile_stage"] == "L1" and base["mobile_wbc_boundary_v1"],
                "different stationary contact site or geometry")
        for field in RUNTIME:
            path = base[field]
            pin(Path(path), base["frozen_files"][path])
            require(base["frozen_files"][path] == receipt["frozen_files"][receipt[field]],
                    "runtime identity differs between the STOP and H0 profiles")
        path = pin(fruit / layout / "SCENE.private.mjb", record_["scene_sha256"])
        rows.append((role, base, path, record_))
    check_occupancy(rows[0][1])
    binding = dict(
        campaign=CAMPAIGN, slots=list(SLOTS), limits=LIMITS, receiver_sha256=sha(receiver),
        compiled_scene_sha256=[r[3]["scene_sha256"] for r in rows],
        source_sha=[r[1]["source_sha"] for r in rows], runtime={k: sha(Path(receipt[k])) for k in RUNTIME},
        stop_predecessor=stop,
        code={p.name: sha(p) for p in (Path(__file__),
              Path(__file__).with_name("sea_fruit_h0_native.py"),
              Path(__file__).with_name("sea_fruit_h0_capture.py"),
              Path(__file__).with_name("sea_fruit_camera.py"),
              Path(__file__).with_name("sea_fruit_h0_score.py"),
              BASE / "scripts/sla_go2_v18/reach_native.py",
              BASE / "scripts/sla_hierarchical_v16/moving_stop_card.py")},
        receiver_proof_sha256=sha(receiver_proof), driver_proof_sha256=sha(REPORT / "FRUIT_DRIVER_CPU.json"),
    )
    return binding, rows, receiver, runtime_pins


def audit(case, result, spec):
    from go2arx5_vla.world_model import init_trace_reader as trace
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    out = case / result["episode"]
    report = trace.audit_trace(out / "init.trace", max_native_ticks=CAP, state_abi=(41, 38, 20))
    ticks = sum(1 for _ in packets(out / "state.bin.journal"))
    require(ticks == report["native_step_records"] <= result["last_requested_tick"] <= CAP
            and result["absent"] and not result.get("close_error")
            and report["integrations_after_first_STOP"] == (0 if report["STOP_observed"] else None),
            "native HOLD terminal accounting mismatch")
    identities = read(out / "LOADED_EXECUTOR_00.private.json")
    for role, name in (("sim", "native_binary"), ("solver", "solver_binary"),
                       ("controller", "controller_binary")):
        require(identities[role]["executable_sha256"] == spec["frozen_files"][spec[name]],
                "loaded native executable changed")
    task = result.get("score")
    complete = bool(
        result["outcome"] in ("H0_HOLD_PASS", "H0_BUDGET_EXHAUSTED", "H0_STAGE_BUDGET_EXHAUSTED",
                              "H0_REFERENCE_REFUSED")
        and result.get("cancel") is not None and result["cancel"]["mode"] == "HOLD"
        and result["cancel"]["native_step"] == result["end"] == ticks
        and all(p["exit_code"] == 0 for p in result["exits"])
        and not report["STOP_observed"] and report["status"] == "RECORDED_TRACE_CHECKS_PASS"
    )
    qualified = bool(complete and result["outcome"] == "H0_HOLD_PASS"
                     and task and task["task_success"])
    return dict(qualified=qualified, completed=complete, ticks=ticks, outcome=result["outcome"],
                task=task, observer_STOP=report["STOP_observed"],
                new_integrations_after_STOP=report["integrations_after_first_STOP"],
                no_survivors=True, exits=result["exits"], observation_events=result.get("observation_events"))


def run_slot(dest, row, receiver, runtime_pins, valid_until, cohort_hash):
    from d1_matched_owner import with_case_lease
    from scripts.pi0292.durable import OwnerLock

    slot, base, scene, compiled = row
    case = dest / slot
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(time.time() + base["wall_limit_s"] <= valid_until
                and not any((p / "STOP.json").exists() for p in (dest, Path(base["campaign_root"]))),
                "expired allocation or operator STOP")
        fields = dict(
            scope=CAMPAIGN, max_ticks=CAP, campaign_root=str(dest), outer_library=str(receiver),
            model_path=str(scene), multitask_geometry=compiled["scoring_geometry"],
            native_environment=environment(base, case),
            frozen_files={**base["frozen_files"], str(receiver): sha(receiver),
                          str(scene): compiled["scene_sha256"]},
        )
        return with_case_lease(case, base, runtime_pins, cohort_hash, fields,
            lambda spec, sp: execute(base, sp, spec, None, scorer=audit, task_driver=drive,
                                     grant_cap=CAP, opening_command=grip_at_grant))


def closed_prefix(root):
    """Verify the startup interruption and the one unspent slot of this allocation."""
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    panel = root / "v19" / CAMPAIGN
    p = read(panel / "PANEL_CLOSE.json")
    require(p["started"] == ["G00-H0"] and p["unstarted"] == ["G01-H0"]
            and p["observed_integrations"] == 2503 and not (panel / "STOP.json").exists(),
            "expected only the first registered slot to have started")
    case = panel / "G00-H0"
    l = read(case / "control/LEASE.json")
    r = read(case / "RESULT.private.json")
    require(l["status"] == "CLOSED" and r["outcome"] == "TECHNICAL_STOP" and not r["survivors"]
            and "fixed base 50 ms reference not installed" in r["interrupt"]
            and len(r["exits"]) == 4 and all(e["exit_code"] == 0 for e in r["exits"]),
            "first slot did not have the verified startup interruption")
    n = case / r["episode"]
    owners = read(n / "LOADED_EXECUTOR_00.private.json")
    pids = {v["pid"] for v in owners.values()} | {v["pid"] for v in r["exits"]}
    require(all(not Path("/proc", str(pid)).exists() for pid in pids), "old owner still alive")
    data = audit_trace(n / "init.trace", max_native_ticks=CAP, state_abi=(41, 38, 20))
    ticks = [row["step"] for row in packets(n / "state.bin.journal")]
    require(ticks == list(range(1, 2504)) and data["native_step_records"] == 2503
            and data["integrations_after_first_STOP"] is None and not data["STOP_observed"]
            and data["status"] == "RECORDED_TRACE_CHECKS_PASS",
            "startup state and terminal trace disagree")
    with (n / "planner.jsonl").open() as stream:
        planner = list(map(json.loads, stream))
    accepted, = [v for v in planner if v["kind"] == "accept"]
    installed = [v for v in planner if v.get("ref_id") == accepted["ref_id"] and v["kind"] == "installed"]
    require([len(v["poses"]) for v in installed] == [81], "different startup receipt")
    hold = held_part(json.loads(accepted["raw"]), "base")
    verify_fixed_base(installed, hold)
    return dict(prior_panel_sha256=sha(panel / "PANEL_CLOSE.json"),
                prior_result_sha256=sha(case / "RESULT.private.json"),
                prior_lease_sha256=sha(case / "control/LEASE.json"),
                prior_trace_sha256=sha(n / "init.trace"), prior_planner_sha256=sha(n / "planner.jsonl"),
                prior_ticks=2503, startup_installed_points=81, prior_processes_absent=True,
                prior_slot="G00-H0", remaining_slot="G01-H0")


def launch(root, binding, rows, receiver, runtime_pins, allocation):
    valid_until = allocation["valid_until_epoch"]
    require(allocation.get("issued") is True and allocation.get("campaign") == CAMPAIGN
            and isinstance(allocation.get("authorization_id"), str) and allocation["authorization_id"].strip()
            and allocation.get("binding") == binding and type(valid_until) in (int, float)
            and math.isfinite(valid_until) and valid_until > time.time(), "unissued or expired science allocation")
    base = rows[0][1]
    dest = root / "v19" / CAMPAIGN
    prefix = None
    if allocation.get("continue_after") is not None:
        prefix = closed_prefix(root)
        require(allocation["continue_after"] == prefix, "continuation does not match closed first card")
        prior = read(dest / "ALLOCATION.json")
        require(allocation["authorization_id"] == prior["authorization_id"]
                and valid_until == prior["valid_until_epoch"] and binding["limits"] == prior["binding"]["limits"]
                and binding["compiled_scene_sha256"] == prior["binding"]["compiled_scene_sha256"]
                and binding["receiver_sha256"] == prior["binding"]["receiver_sha256"],
                "continuation changed scientific allocation")
        rows = rows[1:]
        dest = dest.with_name(CAMPAIGN + "-REMAINING")
    selected = [row[0] for row in rows]
    scratch = root / "v19/fruit-h0-owner-environment"
    for name in ("tmp", "cache", "logs"):
        (scratch / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    owner_environment = environment(base, scratch)
    owner_environment["PYTHONPATH"] = os.pathsep.join((
        str(BASE), str(root / "v12"), str(root / "v12/CPU_LIB"),
        owner_environment["PYTHONPATH"], "/usr/lib/python3/dist-packages",
    ))
    configure(dict(base, native_environment=owner_environment))
    dest.mkdir(mode=0o700)
    record(dest / "BINDING.json", binding)
    record(dest / "ALLOCATION.json", allocation)
    if prefix is not None:
        record(dest / "CLOSED_PREFIX.json", prefix)
    done = {}
    try:
        for row in rows:
            done[row[0]] = run_slot(dest, row, receiver, runtime_pins, valid_until, sha(dest / "BINDING.json"))
            require(read(dest / row[0] / "control/LEASE.json")["status"] == "CLOSED", "lease still active")
            if not done[row[0]].get("completed"):
                break
            if (dest / row[0] / "H0_OBSERVATION_EVENTS.private.json").exists():
                from sea_fruit_h0_capture import capture

                try:
                    observations = capture(dest / row[0])
                    done[row[0]]["sensor_events"] = {name: event["status"]
                                                    for name, event in observations.items()}
                except Exception:
                    # A sensor artifact failure has a separate denominator after physics closes.
                    record(dest / row[0] / "CAPTURE_FAILURE.private.json", dict(error=traceback.format_exc()))
                    done[row[0]]["sensor_events"] = "TECHNICAL_FAILURE"
    finally:
        started = [s for s in selected if (dest / s / "NATIVE_REQUEST.json").exists()]
        record(dest / "PANEL_CLOSE.json", dict(
            results=done, started=started, unstarted=[s for s in selected if s not in started],
            missing_verdict=[s for s in started if s not in done],
            observed_integrations=sum(r["ticks"] for r in done.values()),
            frozen_model_forwards=0, training_updates=0,
        ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "fruit", "receiver"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--allocation", type=Path)
    cli.add_argument("--allocation-sha256")
    cli.add_argument("--closed-prefix", action="store_true")
    args = cli.parse_args()
    os.umask(0o077)
    site = args.site.resolve()
    prepared = site_binding(site, args.fruit.resolve(), args.receiver.resolve())
    if args.closed_prefix:
        require(args.allocation is None, "read-only continuation audit")
        print(json.dumps(dict(status="CPU_OWNER_BOUND", binding=prepared[0],
                              continue_after=closed_prefix(site)), indent=2))
    elif args.allocation:
        require(args.allocation_sha256, "exact allocation SHA-256 required")
        allocation = read(pin(args.allocation, args.allocation_sha256))
        launch(site, *prepared, allocation)
    else:
        print(json.dumps(dict(status="CPU_OWNER_BOUND", binding=prepared[0], new_native_starts=0,
                              new_integrations=0, model_forwards=0), indent=2))
