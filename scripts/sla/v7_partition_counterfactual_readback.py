"""Independently read all frozen swap outputs and recompute physical raw G7 speeds."""

import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha


REGISTRY = "7d0aa1e231b0f3d251eeae7bd388a09b1731852650f008692d4690328312c9b5"
FORWARD = "c7510f6b9f82ad269d71611905a42d8eb2d5bfe2145fd2553b9256403f3a8537"


def run(root):
    registry = read(pin(root / "REGISTRY.json", REGISTRY))
    actual = read(pin(root / "OUTPUT/ACTUAL.json", FORWARD))
    assert actual["complete"] and actual["actual_new_frozen_forwards"] == 32
    assert actual["registry_sha256"] == REGISTRY
    pid = int((root / "measure.pid").read_text())
    assert not (Path("/proc") / str(pid)).exists()
    frames = {f["id"]: f for f in registry["frames"]}
    records = {}
    for row in registry["original_outputs"]["rows"] + actual["rows"]:
        key = row["model"], row["frame"], row["repeat"]
        assert key not in records
        directory = root / "OUTPUT" if row in actual["rows"] else Path(registry["original_outputs"]["root"])
        digest = row["artifact_sha256"] if directory == root / "OUTPUT" else row["action_sha256"]
        with np.load(pin(directory / row["artifact"], digest), allow_pickle=False) as payload:
            action = payload["physical_H40"].copy()
            normalized = payload["normalized"].copy()
        assert action.shape == (40,10) and normalized.shape in ((40,32),(1,40,32))
        assert np.isfinite(action).all() and np.isfinite(normalized).all()
        width = .002 + .040*action[:,9]
        prior = frames[row["frame"]]["previous_command"]["width_m"]
        rate = np.abs(np.diff(np.r_[prior, width]))/.05
        assert abs(rate[0]-row["g7"]["first_seam_speed_m_s"]) < 1e-12
        assert abs(rate.max()-row["g7"]["maximum_speed_m_s"]) < 1e-12
        assert bool(rate.max() > .040) == row["g7"]["original_execution_speed_exceeded"]
        records[key] = (row, action, width)
    names = ("R4", "LAMBDA_ZERO", "SHARED_L2SP", *(m["id"] for m in registry["models"]))
    assert set(records) == {(name,f,rep) for name in names for f in frames for rep in range(2)}
    table = []
    for f in frames:
        for name in names:
            row, action, width = records[name,f,0]
            assert np.array_equal(action, records[name,f,1][1])
            table.append(dict(frame=f, model=name, first_seam_mm_s=1000*row["g7"]["first_seam_speed_m_s"],
                full_H40_max_mm_s=1000*row["g7"]["maximum_speed_m_s"],
                raw_original_limit_exceeded=row["g7"]["original_execution_speed_exceeded"],
                first_mechanical_width_mm=1000*width[0], p10_up_request_mm=row["p10_up_request_mm"]))
    interactions = []
    for arm, shared_only, private_only in (("LAMBDA_ZERO","ZERO_SHARED_R4_IO","R4_SHARED_ZERO_IO"),
                                            ("SHARED_L2SP","L2SP_SHARED_R4_IO","R4_SHARED_L2SP_IO")):
        for f in frames:
            base, full, expert, private = (records[name,f,0][2] for name in ("R4",arm,shared_only,private_only))
            joint = 1000*(full-expert-private+base)
            interactions.append(dict(arm=arm, frame=f, first_width_interaction_mm=float(joint[0]),
                p10_interaction_absmean_mm=float(np.abs(joint[:10]).mean()),
                definition="full - shared-only - private-only + R4, signed mechanical width before absolute speed"))
    output = Path(mkdtemp(prefix="v7-partition-counterfactual-readback.",dir=root.parent)) / "ACTUAL.json"
    result = dict(status="ALL_56_FROZEN_OUTPUTS_AND_RAW_G7_INDEPENDENTLY_VERIFIED", registry_sha256=REGISTRY,
        forward_sha256=FORWARD, recomputed_outputs=len(records), new_frozen_forwards=32, exact_reused_outputs=24,
        all_repeats_exact=True, source_process_gone=True, table=table, width_interactions=interactions,
        raw_illegal_selected_frames={name:sum(t["raw_original_limit_exceeded"] for t in table if t["model"]==name) for name in names},
        scope="FOUR_FIXED_TRAIN_INPUTS_ONE_ORIGINAL_NOISE_PER_INPUT_TWO_IDENTICAL_REPEATS",
        no_dynamic_L1_or_WBC_execution=True, new_model_forwards=0, native_integrations=0, optimizer_updates=0)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(output,sha(output),flush=True)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
