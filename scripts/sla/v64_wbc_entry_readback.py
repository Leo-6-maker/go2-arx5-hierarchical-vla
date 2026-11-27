"""Read the first WBC request back to its measured HOLD target and selected grid."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v64_selected_policy_match import digest, identity


TICK = .002
ARM_LOOKAHEAD = .150
JOINTS = [f"x5_joint{i}" for i in range(1, 7)]


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def stamp(message):
    value = message["header"]["stamp"]
    return value["sec"] + value["nanosec"] / 1e9


def readback(case):
    data = case / "native-01"
    trace, topic = (data / name for name in ("SELECTED_CONTROLLER.private.jsonl", "ROS_CONSUMPTION.private.jsonl"))
    spec_file = case / "EXECUTION_SPEC.json"
    lease = read(case / "control/LEASE.json")
    result = read(case / "RESULT.private.json")
    need(lease["status"] == "CLOSED" and lease["spec_sha256"] == sha(spec_file)
         and result["shutdown_clean"] and not result["unsafe"] and not result["technical_failure"],
         "Require an intact closed, noncontact short card")
    with trace.open() as stream:
        header, first = (json.loads(next(stream)) for _ in range(2))
    need(header["schema"] == "OCS2_SELECTED_REFERENCE_BREAKPOINT_02" and not header["fixture"]
         and first["sequence"] == 1, "Require the first real controller call")
    key = tuple(first[name] for name in ("mpc_init_time_s", "trajectory_sha256", "active_solution_sha256"))
    anchor = first["first_target_time"]
    observation = first["native_observation_time_s"]
    needs = {"ee", "mode", "policy", "observation", "arm_request"}
    rows = {name: [] for name in needs}
    with topic.open() as stream:
        for i, line in enumerate(stream, 1):
            row = json.loads(line)
            name = row["topic_key"]
            if name in needs:
                rows[name].append((i, row))
            if name == "observation" and row["message"]["time"] >= observation + .05:
                break
    switch_line, switch = next((i, row) for i, row in rows["mode"] if row["message"]["data"] == "WBC")
    first_ready_line, ready = next((i, row) for i, row in rows["ee"] if row["message"]["mode"] == 1)
    ready_state = ready["message"]
    arm_line, arm = rows["arm_request"][0]
    point, = arm["message"]["points"]
    suitable = [(i, row) for i, row in rows["policy"] if identity(row["message"]) == key]
    need(len({digest(row["message"]) for _, row in suitable}) == 1, "Need unique exact wire at the initial selected grid")
    policy_line, selected = suitable[0]
    policy = selected["message"]
    target = policy["plan_target_trajectories"]
    need(ready_state["time"] == anchor and target["time_trajectory"] == [anchor]
         and target["state_trajectory"] == [ready_state["state"]]
         and target["input_trajectory"] == [ready_state["input"]]
         and not any(ready_state["input"]["value"]) and switch_line < first_ready_line <= policy_line < arm_line
         and arm["message"]["joint_names"] == JOINTS and stamp(arm["message"]) == observation,
         "The selected target must match the first ready measured EE state and the actual stamped arm request")
    times = np.asarray(policy["time_trajectory"])
    states = np.asarray([r["value"] for r in policy["state_trajectory"]])
    preview = observation + ARM_LOOKAHEAD
    need(first["solution_knots"] == len(times) and states.shape == (len(times), 12)
         and (np.diff(times) > 0).all() and times[0] <= preview <= times[-1], "Unexpected original grid at arm preview")
    q = np.asarray([np.interp(preview, times, values) for values in states[:, -6:].T], dtype=np.float32)
    need(np.array_equal(q, point["positions"]), "The selected grid must exactly reproduce all six requested positions")
    installed = round(observation / TICK)
    prior = [(i, row) for i, row in rows["ee"] if row["message"]["mode"] == 0 and i < switch_line]
    first_policy = rows["policy"][0][1]["message"]
    return dict(condition=case.name, result_sha256=sha(case / "RESULT.private.json"), trace_sha256=sha(trace),
                records_sha256=sha(topic), wbc_mode_receipt_line=switch_line, first_ready_ee_line=first_ready_line,
                prior_hold_observation_tick=round(prior[-1][1]["message"]["time"] / TICK),
                target_anchor_tick=round(anchor / TICK), arm_request_observation_tick=installed,
                first_emitted_solver_tick=round(first_policy["init_observation"]["time"] / TICK),
                actual_selected_solver_init_tick=round(first["mpc_init_time_s"] / TICK), actual_eval_s=first["eval_time_s"],
                arm_preview_s=preview, selected_knots=len(times), published_policy_line=policy_line, arm_request_line=arm_line,
                first_arm_q_rad=q.tolist(), exact_six_joint_reconstruction=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs=3)
    args = cli.parse_args()
    entries = [readback(path) for path in args.cases]
    with args.output.open("x") as stream:
        json.dump(dict(entries=entries, selected_at_request_reconstructed=18, native_integrations=0),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
