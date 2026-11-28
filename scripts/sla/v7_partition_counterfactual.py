"""Frozen TRAIN-only parameter swaps; no optimizer, native admission or physics."""

import argparse
from contextlib import closing
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write


WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"
PANEL = "a52c72a1d0c14ac4823f084ab51f5dd0a67c5ad80fa0433285e39a3a3afb0afd"
PROOF = "c22d2bf4dd6c05411cea0d3b0874254c366bc2b0d4a1aec6e44074151ada7771"
BASELINE = "08fcd9cac0437b78dfaed9dc3762c244c665607a56866886406fde324aff7b97"
FRAMES = ("R4_SAFE_GRASP", "R4_SAFE_LIFT", "TEACHER-2-LIFT", "TEACHER-4-OPEN")
SWAPS = (("R4_SHARED_ZERO_IO", "R4", "LAMBDA_ZERO"),
         ("ZERO_SHARED_R4_IO", "LAMBDA_ZERO", "R4"),
         ("R4_SHARED_L2SP_IO", "R4", "SHARED_L2SP"),
         ("L2SP_SHARED_R4_IO", "SHARED_L2SP", "R4"))


def prepare(work, output):
    require(not output.exists(), "Use a fresh immutable diagnostic identity")
    root = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/v7-b6-p250.dWnpqo")
    panel = root.parent / "v7-b6-p100.NlJUH4/TRAIN-PANEL/PANEL.json"
    roster = read(pin(panel, PANEL))
    proof = read(pin(root / "CORE-PROVIDERS/ACTUAL.json", PROOF))
    require(proof["step"] == 250 and proof["complete_training_endpoint"], "Only full independently restored endpoints")
    original = read(root / "R4_REFERENCE.json")
    roots = {"R4": Path(original["root"]), **{arm: root / "CORE-PROVIDERS" / arm for arm in ("LAMBDA_ZERO", "SHARED_L2SP")}}
    manifests = {name: read(pin(path / "CHECKPOINT.json", original["manifest_sha256"] if name == "R4"
                              else proof["arms"][name]["provider_sha256"])) for name, path in roots.items()}
    baseline = read(pin(root / "TRAIN-PANEL-OUTPUT/ACTUAL.json", BASELINE))
    require(baseline["complete"] and baseline["panel_sha256"] == PANEL, "Exact old outputs required")
    frames = [next(f for f in roster["frames"] if f["id"] == name) for name in FRAMES]
    require(roster["split"] == "TRAIN_DIAGNOSTIC" and len(frames) == 4, "Use the fixed TRAIN mechanism states")
    for f in frames:
        pin(f["input"]["path"], f["input"]["sha256"])
    output.mkdir(mode=0o700)
    models = []
    for name, shared, target in SWAPS:
        directory = output / name
        directory.mkdir(mode=0o700)
        current = dict(manifests["R4"])
        choices, counts = [], {"shared_trainable": 0, "target": 0, "frozen_lora": 0}
        source_maps = {n: {r["key"]: r for r in m["arrays"]} for n, m in manifests.items()}
        for original_row in current["arrays"]:
            key = original_row["key"]
            if "lora" in key.lower():
                source = "R4"
                counts["frozen_lora"] += 1
                require(all(source_maps[n][key]["sha256"] == original_row["sha256"] for n in source_maps), "Frozen LoRA changed")
            elif key.startswith("['parameters']['shared']"):
                source = shared
                counts["shared_trainable"] += 1
            else:
                require(key.startswith("['parameters']['target']"), "Unknown parameter partition")
                source = target
                counts["target"] += 1
            row = source_maps[source][key]
            require(row["shape"] == original_row["shape"] and row["dtype"] == original_row["dtype"], "Array ABI changed")
            dest = directory / row["file"]
            dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(pin(roots[source] / row["file"], row["sha256"]), dest)
            pin(dest, row["sha256"])
            choices.append(dict(row, diagnostic_source=source))
        require(counts == {"shared_trainable": 11, "target": 8, "frozen_lora": 10}, "Unexpected swap partition")
        current.update(arrays=choices, checkpoint_sha256=hashlib.sha256(json.dumps([r["sha256"] for r in choices],separators=(",", ":")).encode()).hexdigest(),
            role="V7_FROZEN_PARTITION_COUNTERFACTUAL_ONLY", formal_inference_eligible=False, native_admission=False,
            hardware_allowed=False, physics=False, diagnostic_sources=dict(shared=shared, target=target),
            phase=None, training_event_sha256=None, training_source="NO_NEW_TRAINING_COUNTERFACTUAL",
            training_receipt_sha256=None, training_contract_sha256=None)
        write(directory / "CHECKPOINT.json", current)
        models.append(dict(id=name, shared_source=shared, go2_io_source=target, root=str(directory),
                           manifest_sha256=sha(directory / "CHECKPOINT.json"), array_counts=counts))
    selected = [r for r in baseline["rows"] if r["frame"] in FRAMES]
    require(len(selected) == 24, "The original three models/two repeats must be complete")
    for row in selected:
        pin(root / "TRAIN-PANEL-OUTPUT" / row["artifact"], row["action_sha256"])
    registry = dict(scope="B6_PARAMETER_PARTITION_COUNTERFACTUAL_TRAIN_ONLY", frames=frames, models=models,
        worker=dict(path=str(work / "ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json"), sha256=WORKER),
        original_provider=original, panel_sha256=PANEL, complete_endpoint_proof_sha256=PROOF,
        original_outputs=dict(root=str(root / "TRAIN-PANEL-OUTPUT"), sha256=BASELINE, rows=selected),
        repetitions=2, maximum_new_frozen_forwards=32, exact_reused_outputs=24,
        primary_question="Does replacing only shared Expert or only Go2 IO restore the original safe raw grasp/open function?",
        selection="Fixed four TRAIN states: two retention failures, safe lift, and the opposite-direction teacher lift change",
        interpretation="Only local parameter-swap effects; hybrid composition is not a trained or deployment-qualified endpoint",
        evaluation_split="TRAIN_DIAGNOSTIC", protected_conditions=0, native_integrations=0, optimizer_updates=0,
        frozen_WBC=True, original_L1=True, original_G7_mm_s=40., no_dynamic_L1_or_WBC_execution=True)
    write(output / "REGISTRY.json", registry)
    print(output / "REGISTRY.json", sha(output / "REGISTRY.json"), flush=True)


def measure(registry, expected):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    r = read(pin(registry, expected))
    spec = read(pin(r["worker"]["path"], r["worker"]["sha256"]))
    bind_imports(spec)
    output = registry.parent / "OUTPUT"
    require(not output.exists(), "Never repeat an existing diagnostic")
    original = r["original_provider"]
    path = pin(original["provider_source"]["path"], original["provider_source"]["sha256"])
    module_spec = importlib.util.spec_from_file_location("partition_original_provider", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu", "Use actual CUDA inference")
        _, _, receiver = accepted(spec)
        source = pinned(receiver["input"])["target_D1"]
        target = AdaptTargetDataset(source["legacy"], source["captures"], resize_with_pad=resize_with_pad)
        target.bind_state(source["statistics"])
        output.mkdir(mode=0o700)
        write(output / "START.json", dict(registry_sha256=expected, gpu=gpu, maximum_new_frozen_forwards=32))
        previous = r["original_outputs"]
        originals = {(row["model"], row["frame"], row["repeat"]): row for row in previous["rows"]}
        rows = []
        for entry in r["models"]:
            root = Path(entry["root"])
            model = module.provider_from_manifest(root, read(pin(root / "CHECKPOINT.json", entry["manifest_sha256"])),
                receiver["model"], target, receiver["tokenizer"]["path"])
            for frame in r["frames"]:
                noise = np.asarray(jax.random.normal(jax.random.PRNGKey(frame["noise_id"]), (1,40,32)))
                noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
                require(all(originals[name,frame["id"],0]["noise_sha256"] == noise_sha
                            for name in ("R4","LAMBDA_ZERO","SHARED_L2SP")), "Actual original CUDA noise changed")
                with np.load(pin(frame["input"]["path"], frame["input"]["sha256"]),allow_pickle=False) as packet:
                    prior = None
                    for repeat in range(2):
                        prediction = model.predict(packet["state_eef10"], [packet["body_rgb"],packet["wrist_rgb"]],
                            frame["language"], packet["T_policy_from_mount"], noise_id=frame["noise_id"], previous_reference=frame["previous_command"])
                        action = prediction["physical_H40"]
                        require(action.shape == (40,10) and np.isfinite(action).all(), "Invalid full raw H40")
                        name = f"{entry['id']}-{frame['id']}-R{repeat}"
                        artifact = output / (name+".npz")
                        with artifact.open("xb") as stream:
                            np.savez(stream, physical_H40=action, normalized=prediction["normalized"])
                        baselines = {}
                        for original_name in ("R4","LAMBDA_ZERO","SHARED_L2SP"):
                            record = originals[original_name,frame["id"],repeat]
                            with np.load(pin(Path(previous["root"])/record["artifact"],record["action_sha256"]),allow_pickle=False) as saved:
                                delta = action-saved["physical_H40"]
                            baselines[original_name] = dict(p10_xyz_rms_mm=float(1000*np.sqrt(np.mean(delta[:10,:3]**2))),
                                p10_jaw_mae_mm=float(40*np.abs(delta[:10,9]).mean()))
                        row = dict(model=entry["id"], frame=frame["id"], repeat=repeat, artifact=artifact.name,
                            artifact_sha256=sha(artifact), noise_sha256=noise_sha, g7=prediction["G7_gripper_speed"],
                            p10_up_request_mm=float(1000*(action[:10,2].max()-packet["state_eef10"][2])),
                            from_original=baselines, repeat_max_abs=0. if prior is None else float(np.max(np.abs(action-prior))))
                        write(output/(name+".json"),row)
                        rows.append(row)
                        prior = action
                        print(json.dumps(dict(done=len(rows), model=entry["id"], frame=frame["id"], repeat=repeat)),flush=True)
            del model
            jax.clear_caches()
            gc.collect()
        require(len(rows) == 32 and max(row["repeat_max_abs"] for row in rows) == 0., "Incomplete or nonrepeatable diagnostic")
        write(output / "ACTUAL.json", dict(registry_sha256=expected, rows=rows, actual_new_frozen_forwards=32,
            exact_reused_outputs=24, complete=True, optimizer_updates=0, native_integrations=0,
            scope=r["scope"], trained_endpoint=False, formal_or_native_admission=False))
        print(output / "ACTUAL.json",sha(output/"ACTUAL.json"),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "measure"))
    parser.add_argument("first", type=Path)
    parser.add_argument("second")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.first, Path(args.second))
    else:
        measure(args.first, args.second)
