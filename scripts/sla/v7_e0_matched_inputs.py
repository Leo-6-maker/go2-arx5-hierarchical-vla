"""Freeze independent TRAIN inputs for R4/1K/3K action and instruction comparisons."""

import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PLAN = "67d15fb8fad688a2bbd82e225111bb49ec34559b"
REPLAY = "e524b5d6d8e239e019dfe575ea848e8ceb88fbc56abce481be7bd4d7dbda81de"
TRAIN = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
MODELS = {
    "R4": "e825a48ed540b5d4d6c3623da24ac56d5aa3b51f2bf3aa488b313f21e044da5e",
    "V65_1K": "b6bad8e39830d9631485f2f8bc871295863522018f3c235a4db4c2a23a924422",
    "V65_3K": "07ac95ce51cbfdce08bce0f80c0af01edd4a9147c7bc29cff449ef4ddb5a345e",
}
FRAMES = (
    ("BLUE_LIFT", "V64_RECOVERY_TRAIN_09", 61, 20953, "LIFT", ("PLACE-blue-left",)),
    ("RED_PREGRASP", "V64_RECOVERY_TRAIN_10", 46, 17203, "PREGRASP",
     ("PLACE-red-right","PLACE-blue-right","HOLD-red","HOLD-blue")),
)


def ref(path):
    return dict(path=str(path),sha256=sha(path))


def freeze(site, train, r4, one, three, output):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context

    models = {}
    for model, root in zip(MODELS,(r4,one,three),strict=True):
        checkpoint = root/"CHECKPOINT.json"
        identity = read(pin(checkpoint,MODELS[model]))
        models[model] = dict(root=str(root),manifest=ref(checkpoint),
                            checkpoint_sha256=identity["checkpoint_sha256"])
    require(len({m["checkpoint_sha256"] for m in models.values()}) == len(MODELS),"Duplicate comparison checkpoints")
    train = read(pin(train/"index.json",TRAIN))
    prompts = {}
    for task in {task for *_,tasks in FRAMES for task in tasks}:
        source = {ep["language"] for ep in train["episodes"] if ep["split"] == "TRAIN" and ep["task"] == task}
        require(len(source)==1,"The instruction must be an original unambiguous TRAIN prompt")
        prompts[task] = source.pop()
    archive = read(pin(site/"v64-recovery-early-teacher-code-01/ACTUAL-EARLY-TEACHER-GUARDED-CLOCK-01.json",REPLAY))
    frames, conditions = {}, []
    for name, case, query, tick, stage, tasks in FRAMES:
        replay, = (row for row in archive["cases"] if row["case"] == case)
        folder = site/"RECOVERY-TRAIN-LIVE-05"/case
        spec = read(folder/"EXECUTION_SPEC.json")
        require(read(folder/"control/LEASE.json")["status"] == "CLOSED" and spec["collection_split"] == "TRAIN"
                and spec["evaluation_split"] == "TRAIN_RECOVERY" and replay["actual_rejected_query"] == query
                and replay["candidate"]["stage"] == stage and replay["native_step"] == tick,
                "Use the independently isolated original TRAIN state")
        path = folder/"native-01/g7-queries"/f"{query:04d}"
        meta, request, result = (read(path/n) for n in ("INPUT_META.private.json","MODEL_REQUEST.private.json",
                                                       "MODEL_RETURN.private.json"))
        source = meta["input"]
        actual = pin(path/"INPUT.npz",source["sha256"])
        require(Path(source["path"]) == actual and meta["native_step"] == request["native_step"] == tick
                and meta["query_index"] == query and request["language"] == prompts[tasks[0]]
                and meta["previous_reference"] == request["previous_reference"]
                and result["response"]["comparison_checkpoint_sha256"] == models["R4"]["checkpoint_sha256"]
                and result["response"]["new_generations_this_request"] == 1, "The source frame and original request differ")
        with np.load(actual,allow_pickle=False) as x:
            require(x["body_rgb"].shape == x["wrist_rgb"].shape == (256,256,3)
                    and x["body_rgb"].dtype == x["wrist_rgb"].dtype == np.uint8
                    and x["state_eef10"].shape == (10,) and x["T_policy_from_mount"].shape == (4,4)
                    and x["previous_width_m"] == previous_command_context(meta["previous_reference"]),
                    "The actual original observation is incomplete")
        frames[name] = dict(case=case,phase=stage,query=query,tick=tick,input=source,
                     original_language=request["language"],observed_noise_id=request["noise_id"],
                     observed_noise_sha256=result["response"]["comparison_noise_sha256"],
                     previous_command=meta["previous_reference"],go2_encoder="ORIGINAL_RGB_STATE_EEF10")
        for task in tasks:
            for model in MODELS:
                conditions.append(dict(frame=name,model=model,task=task,language=prompts[task],noise_from_frame=True,
                                       context_from_frame=True,original_prompt=task==tasks[0]))
    require(len(conditions)==15,"The fixed TRAIN roster has fifteen action-only conditions")
    write(output,dict(plan=PLAN,split="TRAIN_DIAGNOSTIC",frames=frames,models=models,conditions=conditions,
          exact_noise_reproduction_required=True,after_policy_actual_native_integrations=0,
          actual_model_forwards=0,actual_optimizer_updates=0))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("site","train","r4","one","three","output"):
        parser.add_argument(name,type=Path)
    freeze(**vars(parser.parse_args()))
