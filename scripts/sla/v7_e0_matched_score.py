"""Compare full H40 and P10 predictions on the sealed V7 TRAIN roster."""

import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_e0_matched_forward import MANIFEST


def change(a,b):
    delta = a-b
    xyz = np.linalg.norm(delta[:, :3],axis=-1)*1000
    return dict(p10_xyz_max_mm=float(xyz[:10].max()),h40_xyz_max_mm=float(xyz.max()),
                jaw_first_delta_mm=float(delta[0,9]*44),candidate_p10_endpoint_delta_mm=(1000*delta[9,:3]).tolist())


def main(roster, results, output):
    expected = read(pin(roster,MANIFEST))
    outcome = read(results/"ACTUAL.json")
    require(outcome["input_sha256"] == MANIFEST and outcome["native_integrations"] == outcome["optimizer_updates"] == 0
            and outcome["actual_model_forwards"] == len(outcome["rows"]) == len(expected["conditions"]) == 15,
            "Require all fixed frozen output cells")
    outputs = {}
    for row in outcome["rows"]:
        key = (row["frame"],row["model"],row["task"])
        require(key not in outputs and row["noise_sha256"] == expected["frames"][row["frame"]]["observed_noise_sha256"],
                "A model used different observation noise or a duplicated role")
        with np.load(pin(results/row["arrays_file"],row["arrays_sha256"]),allow_pickle=False) as data:
            action = data["physical_H40"]
            require(action.shape == (40,10) and np.isfinite(action).all(), "Need finite full H40")
            outputs[key] = (action,row["g7"])
    main_rows = []
    for frame in expected["frames"]:
        tasks = list(dict.fromkeys(r["task"] for r in expected["conditions"] if r["frame"] == frame))
        for task in tasks:
            base,_ = outputs[frame,"R4",task]
            for model in expected["models"]:
                action,speed = outputs[frame,model,task]
                main_rows.append(dict(frame=frame,task=task,model=model,
                                      compared_to_r4=change(action,base),g7=speed))
    language = []
    across = (("PLACE-blue-right","PLACE-red-right"),("HOLD-blue","HOLD-red"),
              ("HOLD-red","PLACE-red-right"),("HOLD-blue","PLACE-blue-right"))
    for model in expected["models"]:
        for a,b in across:
            left,_ = outputs["RED_PREGRASP",model,a]
            right,_ = outputs["RED_PREGRASP",model,b]
            language.append(dict(model=model,prompt_pair=[a,b],response=change(left,right)))
    write(output,dict(input_sha256=MANIFEST,raw_results_sha256=sha(results/"ACTUAL.json"),
          model_delta=main_rows,language_delta=language,native_integrations=0,optimizer_updates=0))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("roster","results","output"):
        cli.add_argument(key,type=Path)
    main(**vars(cli.parse_args()))
