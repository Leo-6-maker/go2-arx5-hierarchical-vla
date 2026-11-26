"""Join the archived arm PD targets to the actual joint states on exact MuJoCo time."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


ARM = [f"x5_joint{i}" for i in range(1, 7)]
INDICES = slice(12, 18)


def read_clock(path, dt):
    clock, duplicated = {}, set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["running"]:
                continue
            t = row["sim_time_s"]
            tick = round(t / dt)
            if abs(t - tick * dt) > 1e-8:
                raise ValueError("Controller's actual time lies off the native 2 ms grid")
            if tick in clock:
                duplicated.add(tick)
            clock[tick] = row
    return {t: row for t, row in clock.items() if t not in duplicated}


def paired_errors(case, prior):
    import mujoco as mj

    spec = json.loads((case / "EXECUTION_SPEC.json").read_bytes())
    dt = spec["physics_dt_s"]
    model = mj.MjModel.from_binary_path(spec["model_path"])
    if dt != .002 or model.opt.timestep != dt:
        raise ValueError("Unexpected frozen integration clock")
    actuators = [model.actuator(i).name for i in range(12, 18)]
    actuated_joints = [model.joint(int(i)).name for i in model.actuator_trnid[INDICES, 0]]
    if actuators != ARM or actuated_joints != ARM:
        raise ValueError("The six published PD target indexes do not match the arm ABI")
    path = case / "native-01/control.jsonl"
    clock = read_clock(path, dt)
    rows = {}
    for t, state in prior.items():
        tick = int(t)
        if tick not in clock:
            continue
        c = clock[tick]
        if any(len(c[key]) != 18 for key in ("q_target", "kp", "kd", "tau")):
            raise ValueError("Incomplete raw control vector")
        ref = np.asarray(c["q_target"][INDICES])
        q = np.asarray(state["arm_q_rad"])
        rows[tick] = dict(target=ref.tolist(), q=q.tolist(), error=(ref-q).tolist(),
                          kp=c["kp"][INDICES], kd=c["kd"][INDICES], arm_mode=c["arm_mode"],
                          clearance_mm=state["clearance_mm"]["x5_link2"])
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    error = np.asarray([row["error"] for row in rows])
    return dict(samples=len(rows),
                median_error_norm_rad=float(np.median(np.linalg.norm(error, axis=1))) if rows else None,
                median_absolute_error_rad=np.median(np.abs(error), axis=0).tolist() if rows else None)


def run(root, prior, output):
    index = json.loads((prior / "ACTUAL.json").read_bytes())
    output.mkdir(mode=0o700)
    cases, twins = [], {}
    for item in index["rows"]:
        name, role, task = item["id"], item["role"], item["task"]
        group = root / (f"V4-LONG_{task}-{role}-NATIVE-01" + ("-RUN" if role == "M1000" else ""))
        states = json.loads((prior / f"{name}.json").read_bytes())
        rows, digest = paired_errors(group / name, states)
        twins.setdefault((task, name.split("_", 2)[-1]), {})[role] = rows
        task_rows = [value for t, value in rows.items() if t >= 5703]
        cases.append(dict(id=name, task=task, role=role, qpos_samples=len(states), exact_clock_matches=len(rows),
                          task_phase=summarize(task_rows), control_trace_sha256=digest))
        (output / f"{name}.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
    pairs = []
    for (task, ident), group in sorted(twins.items()):
        p, m = (group[role] for role in ("PARENT", "M1000"))
        common = sorted(t for t in p.keys() & m.keys() if t >= 5703)
        differences = [float(np.linalg.norm(np.asarray(m[t]["target"]) - p[t]["target"])) for t in common]
        pairs.append(dict(task=task, identity=ident, common_task_ticks=len(common),
                          q_target_difference_median_rad=statistics.median(differences) if common else None,
                          latest_common_tick=common[-1] if common else None))
    (output / "ACTUAL.json").write_text(json.dumps(dict(cases=cases, pairs=pairs, clock="EXACT_2_MS",
          policy_reference_identity_from_PD=False, model_forwards=0, native_integrations=0, optimizer_updates=0),
          indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prior", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
