"""Independently audit the two terminal native H0 slots and preserve their denominator."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sea_fruit_h0_cold import pin, read, require, sha
from sea_fruit_h0_owner import CAP, CAMPAIGN, closed_prefix
from sea_fruit_h0_native import verify_fixed_base
from sea_fruit_h0_score import FEET, StationaryHoldScore
from d1_factorial_reference_matrix import held_part
from go2arx5_vla.world_model.init_trace_reader import audit_trace
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration


def stationary_case(root):
    import mujoco

    case = root / "v19" / (CAMPAIGN + "-REMAINING") / "G01-H0"
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    result = read(case / "RESULT.private.json")
    audit = read(case / "AUDIT.json")
    native = case / result["episode"]
    identity = read(native / "LOADED_EXECUTOR_00.private.json")
    all_pids = {p["pid"] for p in identity.values()} | {p["pid"] for p in result["exits"]}
    require(lease["status"] == "CLOSED" and result["absent"]
            and all(not Path("/proc", str(pid)).exists() for pid in all_pids)
            and len(result["exits"]) == 4 and all(r["exit_code"] == 0 for r in result["exits"]),
            "site closure and independent process check")
    starts, end = result["entry_tick"], result["end"]
    pause = read(native / "pause_ack.json")
    trace = audit_trace(native / "init.trace", max_native_ticks=CAP, state_abi=(41, 38, 20))
    require(trace["status"] == "RECORDED_TRACE_CHECKS_PASS" and trace["native_step_records"] == end
            and trace["writer"]["dropped"] == 0 and trace["writer"]["sealed_at_explicit_pause"]
            and not trace["STOP_observed"] and pause["step"] == end and pause["running"] is False
            and pause["clock_paused"] is True and result["cancel"]["native_step"] == end
            and result["cancel"]["mode"] == "HOLD", "terminal pause and frozen trace disagree")
    with (native / "planner.jsonl").open() as f:
        planner = list(map(json.loads, f))
    commands = {e["ref_id"]: json.loads(e["raw"]) for e in planner if e["kind"] == "accept"}
    installs = [e for e in planner if e["kind"] == "installed" and e.get("ref_id") in commands]
    anchor = held_part(next(iter(commands.values())), "base")
    verify_fixed_base(installs, anchor)
    require(all(d["_pi0294"]["source_mpc_s"] < end*.002 for d in commands.values())
            and all(e["bound_start"] < end*.002 for e in installs),
            "unexpected installation at or after terminal pause")
    with (case / "student/receiver.csv").open() as f:
        fingers = list(csv.reader(f))
    with (case / "HOLD_CONSUMPTION.private.jsonl").open() as f:
        gripper = list(map(json.loads, f))
    steps = [int(r[2]) for r in fingers if r[0] == "NATIVE_STEP"]
    require(steps == list(range(1, end+1))
            and len(gripper)*25 == end - starts
            and all(abs(g["gripper"]["width_m"] - .042) <= 1e-12 for g in gripper)
            and {g["stage"] for g in gripper} == {"PREGRASP"},
            "terminal consumption or gripper action changed")
    scene = Path(spec["model_path"])
    m = mujoco.MjModel.from_binary_path(str(pin(scene, spec["frozen_files"][str(scene)])))
    s = StationaryHoldScore(m, spec["multitask_geometry"])
    task_samples = {}
    with forbid_integration(mujoco) as calls:
        for i, packet in enumerate(packets(native / "state.bin.journal"), 1):
            require(packet["step"] == i, "missing real native tick")
            if i < starts:
                continue
            s.update_packet(packet)
            if i in (starts, end-1, end):
                data = s.state.data
                task_samples[i] = dict(
                    feet=data.geom_xpos[s.feet].copy(),
                    tcp=data.site("x5_ee").xpos.copy(),
                    fruit=data.geom("task_cube_geom").xpos.copy(),
                )
    require(not calls and i == end and s.result() == result["score"] == audit["task"]
            and result["observation_events"] == {"before_close": None, "first_query_after_grasp": None},
            "replay and live stop diverged")
    initial = task_samples[starts]
    before, terminal = task_samples[end - 1], task_samples[end]
    deltas = (terminal["feet"] - initial["feet"])*1000
    d = np.linalg.norm(deltas, axis=1)
    previous = np.linalg.norm((before["feet"] - initial["feet"])*1000, axis=1)
    violation = int(d.argmax())
    target = initial["fruit"] + np.array([0., -.0017, .080])
    return dict(
        outcome=result["outcome"], qualified=False, entry_tick=starts, terminal_tick=end,
        consumed_50ms_atoms=len(gripper), fixed_base_install_publications=len(installs),
        consumed_queries_before_stop=1+gripper[-1]["query"], last_stage=gripper[-1]["stage"],
        first_violation_tick=s.first_violation["tick"], foot=FEET[violation],
        foot_displacement_mm=d.tolist(), foot_displacement_xyz_mm=deltas.tolist(),
        previous_violating_foot_displacement_mm=float(previous[violation]),
        actual_TCP_displacement_mm=float(np.linalg.norm(terminal["tcp"] - initial["tcp"])*1000),
        TCP_to_teacher_pregrasp_initial_mm=float(np.linalg.norm(initial["tcp"]-target)*1000),
        TCP_to_teacher_pregrasp_terminal_mm=float(np.linalg.norm(terminal["tcp"]-target)*1000),
        terminal_pause_confirmed=True, new_installs_at_or_after_terminal_pause=0,
        processes_absent=True, lease_closed=True, score=audit["task"],
        sensor_events=result["observation_events"],
        evidence_sha256={p.name: sha(p) for p in (
            case / "AUDIT.json", case / "RESULT.private.json", native / "init.trace",
            native / "planner.jsonl", native / "state.bin.journal",
            case / "HOLD_CONSUMPTION.private.jsonl", case / "control/LEASE.json")},
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    site = cli.parse_args().site.resolve()
    startup = closed_prefix(site)
    standing = stationary_case(site)
    print(json.dumps(dict(
        campaign=CAMPAIGN, authorized=2, started=2, qualified=0, teacher_grasps=0,
        G00=dict(outcome="TECHNICAL_BEFORE_TEACHER", **startup),
        G01=standing, total_physics_ticks=startup["prior_ticks"]+standing["terminal_tick"],
        actual_rendered_frames=0, frozen_model_forwards=0, optimizer_updates=0,
    ), indent=2))
