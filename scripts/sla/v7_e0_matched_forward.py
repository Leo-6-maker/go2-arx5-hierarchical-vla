"""Replay the fixed V7 TRAIN frames using three actual frozen Go2 providers."""

import argparse
from contextlib import closing
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write


MANIFEST = "a6246288dd47f003309d058b8288aeea4454c3f63b14ad7ba13d54b1a2504bdb"
WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"
PROVIDER_SOURCE = "7b7787af54868ac7498833e35ae2e66b09a6eb367a03f860f4feb7a5a9e0af46"


def run(manifest, worker_path, provider_source, output):
    import gc
    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    source = read(pin(manifest, MANIFEST))
    worker = read(pin(worker_path, WORKER))
    module = importlib.util.spec_from_file_location("v7_original_native_provider",pin(provider_source,PROVIDER_SOURCE))
    original = importlib.util.module_from_spec(module)
    module.loader.exec_module(original)
    output.mkdir(mode=0o700)
    with closing(OwnerLock(Path(worker["neural_owner_lock"]))):
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1,
                "Expected one original CUDA provider device")
        _, _, receiver = accepted(worker)
        acquisition = pinned(receiver["input"])["target_D1"]
        target = AdaptTargetDataset(acquisition["legacy"],acquisition["captures"],resize_with_pad=resize_with_pad)
        target.bind_state(acquisition["statistics"])
        for row in source["frames"].values():
            noise = jax.random.normal(jax.random.PRNGKey(row["observed_noise_id"]), (1,40,32))
            require(hashlib.sha256(np.asarray(noise).tobytes()).hexdigest() == row["observed_noise_sha256"],
                    "Current CUDA backend cannot reproduce the original noise")
        outputs = []
        for model, reference in source["models"].items():
            root = Path(reference["root"])
            old = reference["manifest"]
            weights = read(pin(root/"CHECKPOINT.json",old["sha256"]))
            require(weights["checkpoint_sha256"] == reference["checkpoint_sha256"],"Provider identity changed")
            provider = original.provider_from_manifest(root,weights,receiver["model"],target,receiver["tokenizer"]["path"])
            planned = [row for row in source["conditions"] if row["model"] == model]
            for row in planned:
                frame = source["frames"][row["frame"]]
                path = frame["input"]
                with np.load(pin(path["path"],path["sha256"]),allow_pickle=False) as data:
                    state = data["state_eef10"]
                    images = [data["body_rgb"],data["wrist_rgb"]]
                    transform = data["T_policy_from_mount"]
                actual = provider.predict(state,images,row["language"],transform,
                             noise_id=frame["observed_noise_id"],previous_reference=frame["previous_command"])
                dest = output/f"{model}-{row['frame']}-{row['task']}.npz"
                with dest.open("xb") as stream:
                    np.savez(stream,normalized=actual["normalized"],physical_H40=actual["physical_H40"])
                outputs.append(dict(row,arrays_sha256=sha(dest),arrays_file=dest.name,
                                     noise_sha256=frame["observed_noise_sha256"],g7=actual["G7_gripper_speed"]))
            del provider
            jax.clear_caches()
            gc.collect()
        require(len(outputs) == len(source["conditions"]) == 15,"Incomplete fixed matching cells")
        write(output/"ACTUAL.json",dict(scope="TRAIN_ONLY_FROZEN_ACTION_DIVERGENCE",input_sha256=MANIFEST,
              rows=outputs,actual_model_forwards=len(outputs),optimizer_updates=0,native_integrations=0))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","worker_path","provider_source","output"):
        p.add_argument(key,type=Path)
    run(**vars(p.parse_args()))
