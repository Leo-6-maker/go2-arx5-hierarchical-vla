"""Verify the terminal measured-BASE pilot against the original G01 stop."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sea_fruit_h0_cold import pin, read, require, sha
from sea_fruit_h0_owner import ANCHOR_CAMPAIGN, CAP
from sea_fruit_h0_native import verify_fixed_base
from sea_fruit_h0_score import StationaryHoldScore
from sea_stance_attribution import DT, foot_window, gait_join, trace_window
from d1_factorial_reference_matrix import held_part
from go2arx5_vla.world_model.init_trace_reader import audit_trace
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration


def target_bind(native, chosen, end):
    new_base = np.asarray(chosen["measured_MPC_mount_world"])
    with (native / "planner.jsonl").open() as stream:
        lines = [json.loads(r) for r in stream]
    accepted = {r["ref_id"]: json.loads(r["raw"]) for r in lines if r["kind"] == "accept"}
    cutover_s = chosen["cutover_tick"]*DT
    after = {k: r for k, r in accepted.items() if r["_pi0294"]["source_mpc_s"] >= cutover_s-1e-8}
    if not after:
        return None
    first = min(after.values(), key=lambda r: r["_pi0294"]["source_mpc_s"])
    np.testing.assert_allclose(held_part(first, "ee"), chosen["old_EEF_accepted_world"], atol=2e-7, rtol=0)
    for wire in after.values():
        np.testing.assert_allclose(held_part(wire, "base"), new_base, atol=2e-7, rtol=0)
    installed = [r for r in lines if r["kind"] == "installed" and r["ref_id"] in after]
    verify_fixed_base(installed, new_base)
    require(all(r["_pi0294"]["source_mpc_s"] < end*DT for r in after.values())
            and all(r["bound_start"] < end*DT for r in installed), "task reference after terminal pause")
    return dict(reference_accepts=len(after), verified_fixed_base_publications=len(installed),
                new_task_installs_at_or_after_terminal_tick=0)


def scoped_measurements(spec, result, native, end):
    import mujoco
    import yaml

    selected = result.get("stance_anchor")
    if selected is None:
        return None
    cutover = selected["cutover_tick"]
    cfg_path, = [Path(k) for k in spec["frozen_files"] if Path(k).name == "config.yaml"]
    cfg = next(iter(yaml.safe_load(pin(cfg_path, spec["frozen_files"][str(cfg_path)]).read_text()).values()))
    scale = np.asarray(cfg["dog_commands_scale"][:3])
    trace = trace_window(native / "init.trace", cutover, end)
    leg = [r for r in trace if r[3] == 3]
    phase = dict(
        effective_gait_cycles=sum(r[10+45] > 0 for r in leg),
        leg_policy_cycles=len(leg),
        maximum_compiled_command_norm=max((float(np.linalg.norm(np.asarray(r[49:52])/scale))
                                           for r in leg), default=None),
    )
    score = result.get("score")
    data = {}
    if score is not None:
        path = Path(spec["model_path"])
        model = mujoco.MjModel.from_binary_path(str(pin(path, spec["frozen_files"][str(path)])))
        stand = mujoco.MjData(model)
        replay = None
        first_tcp = None
        with forbid_integration(mujoco) as calls:
            for p in packets(native / "state.bin.journal"):
                tick = p["step"]
                if tick == cutover:
                    stand.qpos[:] = p["qpos"]
                    mujoco.mj_kinematics(model, stand)
                    origin = (stand.body("base_link").xpos.copy(),
                              stand.body("base_link").xmat.reshape(3, 3).copy(),
                              np.stack([stand.geom(s).xpos for s in ("FL", "FR", "RL", "RR")]))
                    replay = StationaryHoldScore(model, spec["multitask_geometry"], stance_origin=origin)
                if tick >= result["entry_tick"]:
                    replay.update_packet(p)
                    if first_tcp is None:
                        first_tcp = replay.state.data.site("x5_ee").xpos.copy()
        require(not calls and replay.last == end and replay.result() == score, "independent stance scoring differs")
        data["task_tcp_displacement_mm"] = float(
            1000*np.linalg.norm(replay.state.data.site("x5_ee").xpos-first_tcp))
        failed = score["first_stance_or_contact_violation"]
        if failed is not None and failed["foot_drift_m"] > .005 and phase["effective_gait_cycles"] == 1:
            ros = native / "ROS_CONSUMPTION.private.jsonl"
            joined = gait_join(trace, cfg, end, ros)
            data.update(gait=joined, left_front=foot_window(path, native / "state.bin.journal",
                        cutover, end, joined["consumed_ticks"][0], violation_tick=failed["tick"]),
                        ros_evidence_sha256=sha(ros))
    return dict(anchor_cutover_tick=cutover, settled_tick=selected.get("settled_tick"),
                shift_mm=float(np.linalg.norm(selected["base_reference_shift_mm"])),
                rotation_deg=selected["base_reference_rotation_deg"], phase=phase,
                target=target_bind(native, selected, end), **data)


def row(case):
    result = read(case / "RESULT.private.json")
    audit = read(case / "AUDIT.json")
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    native = case / result["episode"]
    require(lease["status"] == "CLOSED", "run owner still active")
    loaded = native / "LOADED_EXECUTOR_00.private.json"
    pids = {r["pid"] for r in result["exits"]} | {r["pid"] for r in read(loaded).values()}
    require(not result["survivors"] and all(not Path("/proc", str(pid)).exists() for pid in pids),
            "native child still exists")
    trace_file = native / "init.trace"
    trace = audit_trace(trace_file, max_native_ticks=CAP, state_abi=(41, 38, 20))
    journal = native / "state.bin.journal"
    end = trace["native_step_records"]
    require([p["step"] for p in packets(journal)] == list(range(1, end+1))
            and audit["ticks"] == end and trace["writer"]["dropped"] == 0
            and trace["integrations_after_first_STOP"] in (0, None), "native tick accounting")
    if result.get("cancel") is not None:
        pause = read(native / "pause_ack.json")
        require(pause["step"] == result["cancel"]["native_step"] == end
                and not pause["running"] and pause["clock_paused"]
                and result["cancel"]["mode"] == "HOLD" and trace["writer"]["sealed_at_explicit_pause"],
                "native terminal pause not confirmed")
    interruption = result.get("interrupt", "").splitlines()
    return dict(
        outcome=result["outcome"], qualified=audit["qualified"], complete=audit.get("completed", False),
        physical_ticks=end, stop_observed=trace["STOP_observed"], trace_signed_at_pause=trace["writer"]["sealed_at_explicit_pause"],
        stage_transitions=result.get("stage_transitions"), anchor=scoped_measurements(spec, result, native, end),
        result_score=result.get("score"), observation_events=result.get("observation_events"),
        first_error=interruption[-1] if interruption else None,
        processes_absent=True, lease_closed=True,
        evidence_sha256={p.name: sha(p) for p in (trace_file, journal, case / "RESULT.private.json",
                         case / "AUDIT.json", case / "control/LEASE.json", native / "planner.jsonl")},
    )


def run(site):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU readout")
    folder = site / "v19" / ANCHOR_CAMPAIGN
    panel = read(folder / "PANEL_CLOSE.json")
    readout = {slot: row(folder / slot) for slot in panel["started"]}
    ticks = sum(r["physical_ticks"] for r in readout.values())
    require(ticks == panel["observed_integrations"] and len(readout) <= 2
            and set(panel["unstarted"]) ^ set(readout) == {"G01-H0", "G00-H0"}
            and all(not (folder / s / "NATIVE_REQUEST.json").exists() for s in panel["unstarted"])
            and panel["frozen_model_forwards"] == panel["training_updates"] == 0,
            "closed panel denominator differs")
    return dict(
        status="CLOSED_NATIVE_H0_MEASURED_BASE_PANEL", authorized=2, started=len(readout),
        requested_order=["G01-H0", "G00-H0"], cases=readout, unstarted=panel["unstarted"],
        physical_ticks=ticks, frozen_model_forwards=0, optimizer_updates=0,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    print(json.dumps(run(cli.parse_args().site.resolve()), indent=2))
