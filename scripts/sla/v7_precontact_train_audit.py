"""Audit a pre-contact Teacher warning on closed TRAIN success and failure states."""
import argparse
from pathlib import Path
from tempfile import mkdtemp

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_adapt_pregrasp_attribution import journal

THRESHOLDS_MM = (5., 10., 15., 20.)


def inspect(root, role):
    lease = read(root / "control/LEASE.json")
    spec = read(pin(root / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    result = read(root / "RESULT.private.json")
    require(lease["status"] == "CLOSED" and spec.get("collection_split") == "TRAIN"
            and result["shutdown_clean"], "Only closed independent TRAIN states are permitted")
    path = root / "native-01/physics.jsonl"
    metadata = {}
    queries = []
    if "TEACHER" in role:
        require(not result["technical_failure"] and not result["unsafe"], "Teacher control must be clean")
        capture = result["detail"]["capture"]
        manifest = read(pin(capture["path"], capture["sha256"]))
        metadata = {row["native_step"]: row for row in read(pin(manifest["labels"]["path"], manifest["labels"]["sha256"]))}
        queries = [(tick, row["phase"]) for tick, row in metadata.items()]
    else:
        for p in sorted((root / "native-01/g7-queries").glob("*/QUERY_ACTUAL.private.json")):
            r = read(p)
            if not r["mock_consumption"]:
                queries.append((r["native_start_step"], "STUDENT_BOUNDARY"))
    require(queries, "No archived query boundary")
    requested = {tick for tick, _ in queries}
    physics = {}
    stop = None
    for row in journal(path):
        if "qpos" not in row:
            continue
        tick = row["step"]
        if row["contact_stop"] and stop is None:
            stop = tick
        if any(tick <= boundary <= tick+2 for boundary in requested):
            physics[tick] = row
    model_path = pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    target_body = "task_blue_cube" if "blue" in spec["task_instruction_id"] else "task_cube"
    wrist = [i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name == "x5_link6"]
    target = [i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name == target_body]
    wrong = [i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name in
             ({"task_cube", "task_blue_cube"} - {target_body})]
    require(wrist and target, "Original wrist or target geometry missing")
    points, missing = [], []
    with forbid_integration(mujoco) as integrated:
        for tick, phase in queries:
            if stop is not None and tick >= stop:
                continue
            candidates = [t for t in physics if tick-2 <= t <= tick]
            if not candidates:
                missing.append(tick)
                continue
            state_tick = max(candidates)
            state = physics[state_tick]
            require(not state["contact_stop"] and not state["illegal_contact"], "Do not revive a stopped world")
            data.qpos[:] = state["qpos"]
            mujoco.mj_fwdPosition(model, data)
            distance = 1000*min(mujoco.mj_geomDistance(model, data, a, b, .25, None) for a in wrist for b in target)
            wrong_distance = None if not wrong else 1000*min(
                mujoco.mj_geomDistance(model, data, a, b, .25, None) for a in wrist for b in wrong)
            points.append(dict(tick=tick, state_tick=state_tick, phase=phase, wrist_target_mm=distance,
                               wrist_other_cube_mm=wrong_distance,
                               seconds_before_first_stop=None if stop is None else .002*(stop-tick)))
    require(not integrated, "The retrospective warning must not integrate physics")
    warnings = {}
    for threshold in THRESHOLDS_MM:
        selected = [p for p in points if p["wrist_target_mm"] < threshold]
        warnings[str(threshold)] = dict(count=len(selected), first=None if not selected else selected[0],
            by_phase={phase:sum(p["phase"] == phase for p in selected) for phase in sorted({p["phase"] for p in points})})
    return dict(case=root.name, role=role, scene_sha256=sha(model_path), result_sha256=sha(root/"RESULT.private.json"),
        physics_sha256=sha(path), task=spec["task_instruction_id"], original_queries=len(queries),
        measured_queries=len(points), missing_boundaries=missing, first_contact_stop=stop, thresholds_mm=warnings,
        boundary_measurements=points, false_alarm_interpretation="WARNING_ON_CLEAN_CONTROL_IS_AN_INTERVENTION_COST_NOT_A_SAFETY_FAILURE")


def run(site):
    previous = read(pin(site/"RECOVERY-TRAIN-ROSTER-05.json",
                        "a140b5a5577807582d2009ce24999821ac9397acf189bc6180a3d81f495834cf"))
    controls = []
    for card in previous["cases"]:
        source = previous["teacher_sources"][card["id"]]
        root = Path(source["teacher_wave"])/"cases"/card["original_teacher"]
        controls.append((root, "COMPLETE_TEACHER_CONTROL"))
    controls += [(site/"V7-D0-EARLY-NATIVE-01/V7_D0_EARLY_TRAIN_11", "R4_LOCAL_SKILL_CONTROL")]
    controls += [(site/"V7-D0-EARLY-NATIVE-02"/name, "R4_CONTACT_FAILURE")
                 for name in ("V7_D0_EARLY_TRAIN_13", "V7_D0_EARLY_TRAIN_14")]
    require(len(controls) == 5 and len({str(p.resolve()) for p, _ in controls}) == 5, "Fixed control cohort changed")
    rows = [inspect(root, role) for root, role in controls]
    output = Path(mkdtemp(prefix="v7-d1-warning-train-audit.", dir=site))/"ACTUAL.json"
    write(output, dict(scope="D1_CLOSED_TRAIN_WARNING_TRADEOFF", cases=rows, thresholds_mm=THRESHOLDS_MM,
          new_native_integrations=0, new_model_forwards=0, new_optimizer_updates=0,
          student_input_additions=0, prospective_trigger_qualified=False))
    print(output, sha(output), flush=True)
    for row in rows:
        print(row["case"],row["role"],row["measured_queries"],row["thresholds_mm"],flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site",type=Path)
    run(**vars(cli.parse_args()))
