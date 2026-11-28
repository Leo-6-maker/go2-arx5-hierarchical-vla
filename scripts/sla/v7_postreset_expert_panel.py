"""Extend one unchanged Expert-reset hybrid to the fixed 37-state TRAIN panel."""

import argparse
from contextlib import closing
import gc
import hashlib
import importlib.util
from pathlib import Path
import shutil

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write


WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"
PANEL = "a52c72a1d0c14ac4823f084ab51f5dd0a67c5ad80fa0433285e39a3a3afb0afd"
BASELINE = "08fcd9cac0437b78dfaed9dc3762c244c665607a56866886406fde324aff7b97"
SWAP_REGISTRY = "7d0aa1e231b0f3d251eeae7bd388a09b1731852650f008692d4690328312c9b5"
SWAP_OUTPUT = "c7510f6b9f82ad269d71611905a42d8eb2d5bfe2145fd2553b9256403f3a8537"
MODEL = "R4_SHARED_ZERO_IO"
SCOPE = "POSTRESET_R4_EXPERT_ZERO250_IO_FIXED37_TRAIN_DIAGNOSTIC"


def prepare(work, output):
    require(not output.exists(), "Keep the existing immutable diagnostic")
    base = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02")
    original = base/"v7-b6-p250.dWnpqo"
    swap = base/"v7-b6-partition-counterfactual-20261128-01"
    roster_path = base/"v7-b6-p100.NlJUH4/TRAIN-PANEL/PANEL.json"
    panel = read(pin(roster_path,PANEL))
    registry = read(pin(swap/"REGISTRY.json",SWAP_REGISTRY))
    old = read(pin(swap/"OUTPUT/ACTUAL.json",SWAP_OUTPUT))
    baseline = read(pin(original/"TRAIN-PANEL-OUTPUT/ACTUAL.json",BASELINE))
    entry, = [r for r in registry["models"] if r["id"] == MODEL]
    manifest = read(pin(Path(entry["root"])/"CHECKPOINT.json",entry["manifest_sha256"]))
    require(manifest["diagnostic_sources"] == dict(shared="R4",target="LAMBDA_ZERO")
            and not manifest["native_admission"] and not manifest["formal_inference_eligible"],
            "Use only the original unqualified parameter intervention")
    require(len(panel["frames"]) == len({r["id"] for r in panel["frames"]}) == 37
            and panel["split"] == "TRAIN_DIAGNOSTIC" and baseline["complete"] and old["complete"]
            and baseline["panel_sha256"] == PANEL,"Require the complete unchanged TRAIN panel")
    roster = {r["id"]:r for r in panel["frames"]}
    require(all(roster[f["id"]] == f for f in registry["frames"]),"An earlier input, prompt or prior command changed")
    reused = [r for r in old["rows"] if r["model"] == MODEL]
    require(len(reused) == 8 and len({(r["frame"],r["repeat"]) for r in reused}) == 8,
            "The old four-condition hybrid output is incomplete")
    for r in reused:
        pin(swap/"OUTPUT"/r["artifact"],r["artifact_sha256"])
    for row in manifest["arrays"]:
        pin(Path(entry["root"])/row["file"],row["sha256"])
    require(len(baseline["rows"]) == 222,"The three original models must retain both repeats on all 37 states")
    for row in baseline["rows"]:
        pin(original/"TRAIN-PANEL-OUTPUT"/row["artifact"],row["action_sha256"])
    for frame in panel["frames"]:
        pin(frame["input"]["path"],frame["input"]["sha256"])
    output.mkdir(mode=0o700)
    write(output/"REGISTRY.json",dict(scope=SCOPE,model=entry,frames=panel["frames"],panel_sha256=PANEL,
        worker=dict(path=str(work/"ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json"),sha256=WORKER),
        original_provider=registry["original_provider"],
        original_outputs=dict(root=str(original/"TRAIN-PANEL-OUTPUT"),sha256=BASELINE,rows=baseline["rows"]),
        reused_hybrid=dict(root=str(swap/"OUTPUT"),receipt_sha256=SWAP_OUTPUT,rows=reused),
        maximum_new_frozen_forwards=66,reused_hybrid_outputs=8,total_hybrid_outputs=74,
        original_baseline_outputs=222,total_comparison_outputs=296,repetitions=2,
        selection="Unregularized private IO plus original shared Expert, expanding the same intervention from four states to every original state",
        primary_question="Do the apparent grasp/open recoveries survive the whole fixed TRAIN panel without new raw-illegal frames?",
        decision="Record all new illegal frames against R4, retained original failures and stage-label error; no native admission from this diagnostic",
        teacher_labels="Original recorded 30 TRAIN Teacher states only; no fabricated labels for student states",
        native_integrations=0,optimizer_updates=0,protected_conditions=0,native_admission=False,
        interpretation="Post-training parameter reset, not a newly trained frozen-Expert endpoint or a native task result"))
    print(output/"REGISTRY.json",sha(output/"REGISTRY.json"),flush=True)


def measure(registry_path, expected):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    registry = read(pin(registry_path,expected))
    require(registry["scope"] == SCOPE and len(registry["frames"]) == 37,"Use the fixed extension")
    output = registry_path.parent/"OUTPUT"
    require(not output.exists(),"Do not repeat completed or unknown forwards")
    spec = read(pin(registry["worker"]["path"],WORKER))
    bind_imports(spec)
    original = registry["original_provider"]
    path = pin(original["provider_source"]["path"],original["provider_source"]["sha256"])
    module_spec = importlib.util.spec_from_file_location("postreset_original_provider",path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    previous = registry["reused_hybrid"]
    reused = {(r["frame"],r["repeat"]):r for r in previous["rows"]}
    baseline = {(r["model"],r["frame"],r["repeat"]):r for r in registry["original_outputs"]["rows"]}
    rows,new_count = [],0
    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        gpu = check_gpu(spec["GPU_uuid"],spec["allocator_limit_bytes"],spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu","Use the original actual CUDA path")
        _,_,receiver = accepted(spec)
        source = pinned(receiver["input"])["target_D1"]
        target = AdaptTargetDataset(source["legacy"],source["captures"],resize_with_pad=resize_with_pad)
        target.bind_state(source["statistics"])
        output.mkdir(mode=0o700)
        write(output/"START.json",dict(registry_sha256=expected,gpu=gpu,maximum_new_frozen_forwards=66))
        entry = registry["model"]
        root = Path(entry["root"])
        model = module.provider_from_manifest(root,read(pin(root/"CHECKPOINT.json",entry["manifest_sha256"])),
            receiver["model"],target,receiver["tokenizer"]["path"])
        for frame in registry["frames"]:
            noise = np.asarray(jax.random.normal(jax.random.PRNGKey(frame["noise_id"]),(1,40,32)))
            noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
            require(all(baseline[name,frame["id"],rep]["noise_sha256"] == noise_sha
                        for name in ("R4","LAMBDA_ZERO","SHARED_L2SP") for rep in (0,1)),"Original actual noise changed")
            pin(frame["input"]["path"],frame["input"]["sha256"])
            prior = None
            with np.load(frame["input"]["path"],allow_pickle=False) as packet:
                for repeat in (0,1):
                    if (frame["id"],repeat) in reused:
                        row = dict(reused[frame["id"],repeat])
                        require(row["noise_sha256"] == noise_sha,"Reused intervention noise changed")
                        src = pin(Path(previous["root"])/row["artifact"],row["artifact_sha256"])
                        shutil.copyfile(src,output/row["artifact"])
                        pin(output/row["artifact"],row["artifact_sha256"])
                        with np.load(src,allow_pickle=False) as saved:
                            action = saved["physical_H40"].copy()
                        row["exactly_reused_from_receipt_sha256"] = previous["receipt_sha256"]
                    else:
                        result = model.predict(packet["state_eef10"],[packet["body_rgb"],packet["wrist_rgb"]],
                            frame["language"],packet["T_policy_from_mount"],noise_id=frame["noise_id"],previous_reference=frame["previous_command"])
                        action = result["physical_H40"]
                        require(action.shape == (40,10) and np.isfinite(action).all(),"Invalid actual complete H40")
                        artifact = output/f"{MODEL}-{frame['id']}-R{repeat}.npz"
                        with artifact.open("xb") as stream:
                            np.savez(stream,physical_H40=action,normalized=result["normalized"])
                        row = dict(model=MODEL,frame=frame["id"],repeat=repeat,artifact=artifact.name,
                            artifact_sha256=sha(artifact),noise_sha256=noise_sha,g7=result["G7_gripper_speed"],
                            p10_up_request_mm=float(1000*(action[:10,2].max()-packet["state_eef10"][2])))
                        new_count += 1
                    row["repeat_max_abs"] = 0. if prior is None else float(np.max(np.abs(action-prior)))
                    require(row["repeat_max_abs"] == 0.,"The original identical repeat is not deterministic")
                    write(output/f"{MODEL}-{frame['id']}-R{repeat}.json",row)
                    rows.append(row)
                    prior = action
                    print(f"POSTRESET_OUTPUT {len(rows)}/74 NEW {new_count}/66 {frame['id']} R{repeat}",flush=True)
        del model
        jax.clear_caches()
        gc.collect()
        require(new_count == 66 and len(rows) == 74,"The expanded model panel is incomplete")
        write(output/"ACTUAL.json",dict(scope=SCOPE,registry_sha256=expected,panel_sha256=PANEL,rows=rows,
            actual_new_frozen_forwards=new_count,reused_hybrid_outputs=8,reused_baseline_outputs=222,
            total_comparison_outputs=296,complete=True,optimizer_updates=0,native_integrations=0,native_admission=False))
        print(output/"ACTUAL.json",sha(output/"ACTUAL.json"),flush=True)


def summarize(registry_path, actual_path):
    from scipy.spatial.transform import Rotation
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10

    registry,actual = read(registry_path),read(actual_path)
    require(actual["complete"] and actual["registry_sha256"] == sha(registry_path),"Incomplete or altered panel")
    original = registry["original_outputs"]
    baseline = {(r["model"],r["frame"],r["repeat"]):r for r in original["rows"]}
    hybrid = {(r["frame"],r["repeat"]):r for r in actual["rows"]}
    result = []
    for frame in registry["frames"]:
        paths = {}
        for model in ("R4","LAMBDA_ZERO","SHARED_L2SP",MODEL):
            r = hybrid[frame["id"],0] if model == MODEL else baseline[model,frame["id"],0]
            root = actual_path.parent if model == MODEL else Path(original["root"])
            digest = r["artifact_sha256"] if model == MODEL else r["action_sha256"]
            with np.load(pin(root/r["artifact"],digest),allow_pickle=False) as f:
                paths[model] = dict(row=r,action=f["physical_H40"].copy(),teacher=f["teacher_H40"].copy() if "teacher_H40" in f else None,
                                    valid=f["valid"].copy() if "valid" in f else None)
        base = paths["R4"]["action"]
        labels,support = paths["R4"]["teacher"],paths["R4"]["valid"]
        for model,data in paths.items():
            r,action = data["row"],data["action"]
            widths = .002+.040*action[:,9]
            speed = np.abs(np.diff(np.r_[frame["previous_command"]["width_m"],widths]))/.05
            require(abs(speed.max()-r["g7"]["maximum_speed_m_s"]) < 1e-12
                    and abs(speed[0]-r["g7"]["first_seam_speed_m_s"]) < 1e-12,
                    "Independent mechanical-width G7 readback differs")
            angle = np.degrees(Rotation.from_matrix(matrix_from10(action)[:,:3,:3]
                               @ np.swapaxes(matrix_from10(base)[:,:3,:3],-1,-2)).magnitude())
            value = dict(frame=frame["id"],phase=frame["phase"],episode=frame["episode"],model=model,
                first_seam_mm_s=float(1000*speed[0]),full_H40_max_mm_s=float(1000*speed.max()),raw_illegal=bool(speed.max()>.040),
                p10_up_request_mm=r["p10_up_request_mm"],p10_xyz_rms_from_R4_mm=float(1000*np.sqrt(np.mean((action[:10,:3]-base[:10,:3])**2))),
                p10_jaw_mae_from_R4_mm=float(40*np.abs(action[:10,9]-base[:10,9]).mean()),
                p10_rotation_mean_from_R4_deg=float(angle[:10].mean()))
            if labels is not None:
                valid = support[:10]
                value.update(p10_teacher_xyz_rms_mm=float(1000*np.sqrt(np.mean((action[:10,:3]-labels[:10,:3])[valid]**2))),
                    p10_teacher_jaw_mae_mm=float(40*np.abs(action[:10,9]-labels[:10,9])[valid].mean()))
            result.append(value)
    index = {(r["model"],r["frame"]):r for r in result}
    sets = {model:{r["frame"] for r in result if r["model"]==model and r["raw_illegal"]}
            for model in ("R4","LAMBDA_ZERO","SHARED_L2SP",MODEL)}
    summary = dict(scope=SCOPE,actual_sha256=sha(actual_path),registry_sha256=sha(registry_path),rows=result,
        raw_illegal_frames={k:sorted(v) for k,v in sets.items()},
        new_illegal_vs_R4=sorted(sets[MODEL]-sets["R4"]),recovered_illegal_vs_dense250=sorted(sets["LAMBDA_ZERO"]-sets[MODEL]),
        new_illegal_vs_dense250=sorted(sets[MODEL]-sets["LAMBDA_ZERO"]),
        all_37_compared=True,all_physical_speeds_independently_recomputed=True,
        actual_new_frozen_forwards=66,reused_hybrid_outputs=8,reused_baseline_outputs=222,
        native_integrations=0,optimizer_updates=0,native_admission=False)
    path = actual_path.parent/"SUMMARY.json"
    write(path,summary)
    print(path,sha(path),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","measure","summarize"))
    parser.add_argument("first",type=Path)
    parser.add_argument("second")
    args=parser.parse_args()
    if args.mode == "prepare":prepare(args.first,Path(args.second))
    elif args.mode == "measure":measure(args.first,args.second)
    else:summarize(args.first,Path(args.second))
