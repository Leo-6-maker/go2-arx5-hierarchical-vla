"""Reconstruct the archived frozen MuJoCo scenes without integration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def run(root, out):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration

    out.mkdir(mode=0o700)
    proofs, paired = [], {}
    with forbid_integration(mj) as calls:
        for task, expected in (("PLACE", 4), ("HOLD", 2)):
            for role in ("PARENT", "M1000"):
                group = root / (f"V4-LONG_{task}-{role}-NATIVE-01" + ("-RUN" if role == "M1000" else ""))
                cases = [p for p in sorted(group.glob(f"LONG_{role}_*")) if (p / "RESULT.private.json").is_file()]
                if len(cases) != expected:
                    raise ValueError("Archived native panel denominator changed")
                for case in cases:
                    scene, terminal, states = measure(case, mj)
                    name = case.name.split("_", 2)[-1]
                    paired.setdefault((task, name), {})[role] = (scene, states)
                    last = max(states)
                    proofs.append(dict(id=case.name, task=task, role=role, samples=len(states), latest_tick=last,
                        last_link2_mm=states[last]["clearance_mm"]["x5_link2"],
                        min_link2_mm=min(row["clearance_mm"]["x5_link2"] for row in states.values()),
                        actual_selected_reference_ids=sum(r["selected_reference"] !=
                            "DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED" for r in states.values()),
                        terminal_sha256=terminal))
                    write(out / f"{case.name}.json", states)
        if calls:
            raise RuntimeError("Archived geometry check performed a native step")
    comparisons = []
    for (task, identity), records in sorted(paired.items()):
        p_scene, parent = records["PARENT"]
        m_scene, model = records["M1000"]
        if p_scene != m_scene:
            raise ValueError("Matched pair changed the original scene")
        ticks = sorted(set(parent) & set(model))
        gap = {i: model[i]["clearance_mm"]["x5_link2"] - parent[i]["clearance_mm"]["x5_link2"] for i in ticks}
        last = ticks[-1] if ticks else None
        delta = {key: float(np.linalg.norm(np.array(model[last][key]) - np.array(parent[last][key])))
                 for key in ("tcp_xyz", "mount_xyz", "arm_q_rad")} if last is not None else {}
        comparisons.append(dict(task=task, identity=identity, same_tick_pairs=len(ticks), latest_common_tick=last,
            link2_delta_at_last_common_mm=gap.get(last), latest_common_difference=delta,
            first_link2_drop_over_2mm_tick=next((i for i in ticks if gap[i] < -2.), None)))
    write(out / "ACTUAL.json", dict(rows=proofs, paired=comparisons, sampled_states=sum(x["samples"] for x in proofs),
          model_forwards=0, native_integrations=0, optimizer_updates=0))


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def measure(case, mj):
    spec = json.loads((case / "EXECUTION_SPEC.json").read_bytes())
    result = (case / "RESULT.private.json").read_bytes()
    if json.loads(result)["technical_failure"]:
        raise ValueError("Technical case cannot represent the native pair")
    scene = spec["model_path"]
    model = mj.MjModel.from_binary_path(scene)
    data = mj.MjData(model)
    table = model.geom("task_table").id
    arm = {}
    for i in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[i])).name
        if name.startswith("x5_link") and (model.geom_contype[i] or model.geom_conaffinity[i]):
            arm.setdefault(name, []).append(i)
    joints = [model.joint(f"x5_joint{i}").id for i in range(1, 7)]
    if "x5_link2" not in arm or len(joints) != 6:
        raise ValueError("Frozen arm geometry changed")
    points = np.empty(6)
    states = {}
    with (case / "native-01/MOBILE_SAMPLES.private.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            data.qpos[:] = row["qpos"]
            data.qvel[:] = row["qvel"]
            mj.mj_fwdPosition(model, data)
            clearance = {name: 1000 * min(mj.mj_geomDistance(model, data, i, table, 2., points)
                         for i in geoms) for name, geoms in arm.items()}
            states[row["native_end_step"]] = dict(clearance_mm=clearance,
                tcp_xyz=np.asarray(row["actual_TCP"])[:3, 3].tolist(),
                mount_xyz=np.asarray(row["actual_mount"])[:3, 3].tolist(), base_dof_velocity=data.qvel[:6].tolist(),
                arm_q_rad=data.qpos[model.jnt_qposadr[joints]].tolist(),
                arm_dq_rad_s=data.qvel[model.jnt_dofadr[joints]].tolist(),
                installed_reference_sha256=row["installed_reference_sha256"],
                solver_exact_installed_targets=row["solver_exact_installed_targets"],
                selected_reference=row["controller_selected_reference"])
    return scene, hashlib.sha256(result).hexdigest(), states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
