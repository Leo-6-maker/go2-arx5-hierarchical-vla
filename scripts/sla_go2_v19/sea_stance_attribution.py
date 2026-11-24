"""Join the closed G01 H0 gait input, native PD targets and physical foot trace."""

import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import sys

import numpy as np

from sea_fruit_h0_cold import BASE, pin, read, require, sha

sys.path.insert(0, str(BASE / "scripts/sla_transfer_v15"))
from d1_gait_replay import TRACE

REPORT = BASE / "reports/SLA_GO2_V193/GRIP_H0/FRUIT_H0_ACTUAL.json"
G01 = "v19/V193-STANDING-FRUIT-H0-01-REMAINING/G01-H0"
GAIT_THRESHOLD = .1  # Deployed SDK's mixed-unit norm over planar velocity and yaw.
DT = .002


def trace_window(path, start, end):
    with path.open("rb") as f:
        h = f.read(16)
        require(h[:8] == b"WM02IT01" and int.from_bytes(h[8:12], "little") == TRACE.size,
                "different trace ABI")
        return [r for r in TRACE.iter_unpack(f.read())
                if start*DT-1e-8 <= r[2] <= end*DT+1e-8 and r[3] in (3, 5, 9, 11, 12)]


def gait_join(events, cfg, end, ros):
    frames = {r[9]: r for r in events if r[3] == 3}
    activated = [v[9] for v in frames.values() if v[10+45] > 0]
    token, = activated
    old, gait = frames[token-1], frames[token]
    scale = np.asarray(cfg["dog_commands_scale"][:3], float)
    old_cmd, command = [np.array(r[10+39:10+42])/scale for r in (old, gait)]
    before, after = (float(np.linalg.norm(cmd)) for cmd in (old_cmd, command))
    require(old[10+45] == 0 and np.isclose(gait[10+45], cfg["gait_frequency"])
            and before < GAIT_THRESHOLD <= after, "frozen gait transition mismatch")
    out = next(r for r in events if r[3] == 5 and r[9] == token)
    native_pd = sorted((r for r in events if r[3] == 9), key=lambda r: r[1])
    walls = [r[1] for r in native_pd]
    expected = np.asarray(cfg["default_dof_pos"][:12]) + np.asarray(cfg["action_scale"][:12])*np.asarray(out[10:22])
    stop_by_tick = {r[9]: r for r in events if r[3] == 12}
    consumed = []
    peak = 0.
    for step in (r for r in events if r[3] == 11 and r[7] == token):
        index = bisect_right(walls, step[1])-1
        output = native_pd[index]
        terminal = stop_by_tick[step[9]]
        require(index >= 0 and index == bisect_right(walls, terminal[1])-1
                and output[7] == terminal[7] == token, "mixed PD consumption")
        peak = max(peak, float(np.max(abs(expected - np.asarray(output[10:22])))))
        consumed.append(step[9])
    require(consumed == list(range(consumed[0], end+1)) and peak < 2e-6,
            "gait output was not joined to all stop-window integrations")

    # Receive timestamps are observational. The frozen trace's applied token
    # supplies the actual native consumption identity.
    prior = stop_by_tick[consumed[0]-25][1]*1e-9 if consumed[0]-25 in stop_by_tick else gait[1]*1e-9-.7
    last = stop_by_tick[end][1]*1e-9
    matches = []
    with ros.open("rb") as f:
        for raw in f:
            row = json.loads(raw)
            wall = row["received_monotonic_s"]
            if prior <= wall <= last and row["topic_key"] == "base_command":
                if np.array_equal(row["message"]["data"][:3], command):
                    matches.append(wall)
    require(matches, "no real WBC base command matched the applied input")
    previous_target = next(r[10:22] for r in reversed(native_pd)
                           if r[7] == token-1 and r[1] <= gait[1])
    return dict(
        gait_tokens_after_task_entry=activated, transition_s=gait[2], norm_before=before,
        norm_after=after, threshold=GAIT_THRESHOLD, frequency_before_Hz=old[10+45],
        frequency_after_Hz=gait[10+45], contributing_base_command=command.tolist(),
        exact_base_message_matches=len(matches), consumed_ticks=consumed, max_target_join_error_rad=peak,
        FL_PD_target_before_rad=list(previous_target[:3]), FL_PD_target_after_rad=expected[:3].tolist(),
    )


def foot_window(scene, journal, start, end, first_gait_tick):
    import mujoco as mj
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    model = mj.MjModel.from_binary_path(str(scene))
    data = mj.MjData(model)
    foot, floor = model.geom("FL").id, model.geom("floor").id
    base, tip = model.body("base_link").id, model.site("x5_ee").id
    rows = []
    with forbid_integration(mj) as calls:
        for p in packets(journal):
            tick = p["step"]
            if tick < start:
                continue
            data.qpos[:] = p["qpos"]
            mj.mj_kinematics(model, data)
            contacts = [c for c in p["contacts"] if {c["geom0"], c["geom1"]} == {foot, floor}]
            rows.append(dict(tick=tick, fl=data.geom_xpos[foot].copy(), base=data.xpos[base].copy(),
                rotation=data.xmat[base].reshape(3, 3).copy(), tcp=data.site_xpos[tip].copy(),
                fl_normal_force_N=float(sum(c["wrench"][0] for c in contacts)),
                fl_tangent_force_N=float(sum(np.linalg.norm(c["wrench"][1:3]) for c in contacts)),
                FL_effort_Nm=p["ctrl"][:3].tolist()))
    require(not calls and [r["tick"] for r in rows] == list(range(start, end+1)), "native task clock")
    initial, terminal = rows[0], rows[-1]
    local_foot = initial["rotation"].T @ (initial["fl"] - initial["base"])
    base_only = terminal["base"] + terminal["rotation"] @ local_foot - initial["fl"]
    shift = np.stack([r["fl"] - initial["fl"] for r in rows])*1000
    speeds = np.linalg.norm(np.diff(shift, axis=0), axis=1)/DT
    before = first_gait_tick-start-1
    require(int(np.flatnonzero(np.linalg.norm(shift, axis=1) > 5.)[0])+start == end,
            "replayed foot stop mismatch")
    tail = [dict(tick=r["tick"], foot_shift_mm=float(np.linalg.norm(shift[r["tick"]-start])),
                 fl_normal_force_N=r["fl_normal_force_N"], fl_tangent_force_N=r["fl_tangent_force_N"],
                 FL_effort_Nm=r["FL_effort_Nm"]) for r in rows[before:]]
    return dict(max_before_gait_foot_shift_mm=float(np.linalg.norm(shift[:before+1],axis=1).max()),
                peak_pre_gait_FL_speed_mm_s=float(max(speeds[:before])),
                peak_during_gait_FL_speed_mm_s=float(max(speeds[before:])),
                terminal_base_component_xyz_mm=(base_only*1000).tolist(),
                terminal_leg_articulation_component_xyz_mm=(shift[-1]-base_only*1000).tolist(),
                tcp_move_during_applied_gait_mm=float(np.linalg.norm(terminal["tcp"]-rows[before]["tcp"])*1000),
                local_window=tail)


def run(site):
    import yaml

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU replay only")
    published = read(REPORT)["G01"]
    pins = published["evidence_sha256"]
    case = site / G01
    lease = read(pin(case / "control/LEASE.json", pins["LEASE.json"]))
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    result = read(pin(case / "RESULT.private.json", pins["RESULT.private.json"]))
    require(lease["status"] == "CLOSED" and result["absent"], "closed H0 only")
    native = case / result["episode"]
    trace = pin(native / "init.trace", pins["init.trace"])
    journal = pin(native / "state.bin.journal", pins["state.bin.journal"])
    scene = Path(spec["model_path"])
    pin(scene, spec["frozen_files"][str(scene)])
    config, = [Path(k) for k in spec["frozen_files"] if Path(k).name == "config.yaml"]
    cfg = next(iter(yaml.safe_load(pin(config, spec["frozen_files"][str(config)]).read_text()).values()))
    start, end = published["entry_tick"], published["terminal_tick"]
    ros = native / "ROS_CONSUMPTION.private.jsonl"
    ros_sha = sha(ros)
    gait = gait_join(trace_window(trace, start, end), cfg, end, ros)
    pin(ros, ros_sha)
    foot = foot_window(scene, journal, start, end, gait["consumed_ticks"][0])
    planner = pin(native / "planner.jsonl", pins["planner.jsonl"])
    with planner.open() as f:
        publications = [json.loads(r) for r in f]
    last = [e for e in publications if e["kind"] == "accept"][-1]
    clock = json.loads(last["raw"])["_pi0294"]
    return dict(
        status="CLOSED_G01_LEG_GAIT_TRANSITION_EXPLAINED", gait=gait, left_front=foot,
        last_reference_source_s=clock["source_mpc_s"], last_reference_ttl_s=clock["ttl_s"],
        gait_transition_reference_age_s=gait["transition_s"]-clock["source_mpc_s"],
        final_physics_tick=end, evidence_sha256={"trace": sha(trace), "journal": sha(journal),
                                              "ros": ros_sha, "config": sha(config), "planner": sha(planner)},
        new_integrations=0, new_frozen_policy_forwards=0, training_updates=0,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    print(json.dumps(run(cli.parse_args().site.resolve()), indent=2))
