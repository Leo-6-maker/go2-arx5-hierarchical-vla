"""Read collision lead time at completed TRAIN query boundaries without integration."""
from pathlib import Path
from tempfile import mkdtemp
import json
import sys

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, sha
from v64_adapt_pregrasp_attribution import journal

def run(site):
    seal = site / "V7-D0-EARLY-SEALED-02/RESULT.json"
    closed = read(pin(seal,"47455a46b49c09124c98cba7a9d5b910221ff19b0362d0ceac1e40b7beca7068"))
    rows = []
    for case in closed["cases"]:
        root = site/"V7-D0-EARLY-NATIVE-02"/case["id"]
        result = read(pin(root/"RESULT.private.json",case["result"]["sha256"]))
        lease = read(root/"control/LEASE.json")
        spec = read(pin(root/"EXECUTION_SPEC.json",lease["spec_sha256"]))
        assert lease["status"] == "CLOSED" and spec["collection_split"] == "TRAIN"
        assert case["after_first_stop_ticks"] == 0 and result["shutdown_clean"]
        model = mujoco.MjModel.from_binary_path(str(pin(spec["model_path"],spec["frozen_files"][spec["model_path"]])))
        target = "task_blue_cube" if "blue" in spec["task_instruction_id"] else "task_cube"
        geometries = {"wrist":[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name=="x5_link6"],
                      "target":[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name==target]}
        assert all(geometries.values())
        queries = [read(p) for p in sorted((root/"native-01/g7-queries").glob("*/QUERY_ACTUAL.private.json"))]
        assert len(queries)==case["forwards"]
        physics = {p["step"]:p for p in journal(root/"native-01/physics.jsonl") if "qpos" in p}
        points = []
        data = mujoco.MjData(model)
        with forbid_integration(mujoco) as integrated:
            for query in queries:
                tick = query["native_start_step"]
                if query["mock_consumption"] or tick >= case["ticks"]:
                    continue
                # These are completed historical boundaries preceding the next request.
                p = max((p for t,p in physics.items() if tick-2<=t<=tick),key=lambda p:p["step"])
                assert not p["contact_stop"] and not p["illegal_contact"]
                data.qpos[:] = p["qpos"]
                mujoco.mj_fwdPosition(model,data)
                distance = min(mujoco.mj_geomDistance(model,data,a,b,.25,None)
                               for a in geometries["wrist"] for b in geometries["target"])
                points.append(dict(query=query["query"],tick=tick,observed_state_tick=p["step"],
                                   wrist_target_mm=1000*distance,seconds_before_stop=.002*(case["ticks"]-tick)))
        assert not integrated
        warnings = {}
        for threshold in (5.,10.,15.,20.):
            matching = [p for p in points if p["wrist_target_mm"] < threshold]
            warnings[str(threshold)] = dict(first=None if not matching else matching[0],count=len(matching))
        rows.append(dict(id=case["id"],task=spec["task_instruction_id"],terminal_tick=case["ticks"],
                         original_queries=len(queries),exact_or_preceding_state_tolerance_ticks=2,thresholds_mm=warnings,
                         boundary_measurements=points,result_sha256=case["result"]["sha256"]))
    output = Path(mkdtemp(prefix="v7-early-contact-leadtime.",dir=site))/"ACTUAL.json"
    output.write_text(json.dumps(dict(scope="CLOSED_TRAIN_RETROSPECTIVE_CONTACT_LEAD_TIME",
         cases=rows,teacher_privileged_geometry=True,student_input_additions=0,
         prospective_trigger_qualified=False,new_native_integrations=0,new_model_forwards=0,new_optimizer_updates=0),indent=2)+"\n")
    print(output,sha(output))
    print(json.dumps([{k:v for k,v in row.items() if k!="boundary_measurements"} for row in rows]))

if __name__=="__main__":
    run(Path(sys.argv[1]))
