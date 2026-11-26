"""Read a fixed SC-VLA source-DEV cohort through the actual H30 FM network on Corelab."""

import argparse
from contextlib import closing
import hashlib
import math
import os
from pathlib import Path
import sys
import time

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from v62_sc_source import CAMERAS, HORIZON, NATIVE_HZ, SCHEMA, SOURCE_TASKS
from v62_sc_training_inputs import ROLE, abstract_model, flow_loss, frozen_draws, prepare
from v62_source_dev_eval import restore_groups
from v62_source_native import checked_manifest, reference, source_role
from v62_source_admission import V62
from v6_ville_reader import SingleArmReader


CATALOGUE = "165f388dabc4f6d5ecd147f709c1749ee726daa820a828dc560ef985f936877f"
ENDPOINT_AUDIT = "1d4b4b671b4a99a8bdd489e181f3e2da2167cb951eb7b1c46f4df49cd649c4fe"
STEP = 250
SEED = 2026112701


def readers(path):
    from openpi_client.image_tools import resize_with_pad

    chosen = {}
    for version in read(pin(path, CATALOGUE))["versions"]:
        if version["domain"] != ROLE:
            continue
        ref = version["manifest"]
        file = pin(ref["path"], ref["sha256"])
        manifest = read(file)
        task = manifest["task"]
        if task in chosen:
            continue
        dev = read(pin(version["core_ready"]["path"], version["core_ready"]["sha256"]))
        require(dev["manifest_sha256"] == ref["sha256"] and dev["by_split"]["SOURCE_DEV"]["episodes"] >= 2,
                "The fixed source validation episodes must be readable")
        chosen[task] = SingleArmReader(Path(file).parent, ref["sha256"], Path(manifest["video_root"]),
            split="SOURCE_DEV", resize=resize_with_pad, schema=SCHEMA, native_hz=NATIVE_HZ, horizon=HORIZON, cameras=CAMERAS)
    require(set(chosen) == set(SOURCE_TASKS), "A core H30 task or its validation split is absent")
    return chosen


def run(worker, catalogue, endpoint, out):
    spec = read(worker)
    bind_imports(spec)

    import jax
    import numpy as np
    from flax import nnx
    from go2arx5_vla.sla_adapter import batch_v2, scale20k_model
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.sla.core_hopper_native_scope import accepted

    _, gate, receiver = accepted(spec)
    parent = read(pin(Path(gate["package"]["path"]) / "CHECKPOINT.json", gate["package"]["sha256"]))
    audit = reference(pin(endpoint / "CPU-ENDPOINT-READBACK.json", ENDPOINT_AUDIT))
    folder = endpoint / "PROVIDER"
    model = checked_manifest(dict(path=str(folder), sha256=sha(folder / "CHECKPOINT.json")), parent,
                             dict(v4_model_role=source_role(STEP), source_step=STEP, v62_plan=V62, source_audit=audit))
    checkpoint = endpoint / "run" / f"STEP{STEP:04d}"
    from v3_main_place_learn import endpoint_files

    terminal = read(pin(endpoint / "run/TERMINAL.json", model["training_receipt_sha256"]))
    require(terminal["checkpoint"]["files"] == endpoint_files(checkpoint), "The exact frozen SOURCE state is required")
    data = readers(catalogue)
    require(os.uname().nodename == spec["host"] and os.getuid() == spec["uid"] and sys.executable == spec["python"]
            and all(os.environ.get(k) == spec["gpu_environment"][k] for k in
                    ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION")),
            "This engineering diagnostic uses the qualified Corelab CUDA environment")
    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "Corelab GPU not routed to the model")
        out.mkdir(mode=0o700)
        jax.config.update("jax_compilation_cache_dir", str(out / "jit-cache"))
        write(out / "START.json", dict(plan=V62, step=STEP, audit=audit, catalogue_sha256=CATALOGUE,
            gpu=gpu, maximum_forward_batches=3, maximum_windows=6, seed=SEED, zero_updates=True))
        asset = receiver["model"]
        require(asset["parent_manifest"]["sha256"] == model["foundation_sha256"], "Frozen model asset changed")
        _, template, frozen, identity = scale20k_model.load(asset["parent_manifest"], asset["parent_params"])
        require(identity["frozen_fingerprint"] == model["frozen_model"]
                and canonical_hash(identity["inventory"]) == model["partition_sha256"], "Frozen model partition changed")
        tokenizer = Tokenizer(str(pin(receiver["tokenizer"]["path"], model["tokenizer_sha256"])))
        restored = restore_groups(checkpoint, ("shared", "single"), STEP)
        active = {name: jax.tree.map(lambda value: value, template[role])
                  for name, role in (("shared", "shared"), ("private", "target"))}
        active["shared"].replace_by_pure_dict(restored["shared"])
        active["private"].replace_by_pure_dict(restored["single"])
        graph, _ = nnx.split(abstract_model())
        evaluate = jax.jit(flow_loss(graph))
        draw = frozen_draws(2)
        returned = []
        for ordinal, task in enumerate(SOURCE_TASKS):
            reader = data[task]
            indices = [0, len(reader) - 1]
            rows = [reader.rows[i] for i in indices]
            require(rows[0][0] != rows[1][0], "Validation windows must start in different episodes")
            cursor = 27_000_000 + 2 * ordinal
            batch = prepare(reader, tokenizer, [cursor, cursor + 1], indices, training=False)
            obs, actions = batch_v2.device_batch(batch)
            support = jax.device_put(batch["action_support"])
            noise, tau = draw(jax.random.PRNGKey(SEED), cursor)
            record = dict(task=task, rows=rows, cursor=cursor, shape=list(actions.shape),
                          supported_values=int(support.sum()), noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest())
            write(out / f"{ordinal}.ENTERED.json", record)
            began = time.monotonic()
            value = float(evaluate(active, frozen, obs, actions, noise, tau, support))
            require(math.isfinite(value), "The actual H30 FM forward returned nonfinite loss")
            result = dict(record, fm_loss=value, elapsed_s=time.monotonic() - began)
            write(out / f"{ordinal}.RETURNED.json", result)
            returned.append(result)
        write(out / "ACTUAL.json", dict(plan=V62, role=ROLE, model=model["checkpoint_sha256"],
            source_dev_only=True, actual_forward_batches=len(returned), actual_examples=2*len(returned),
            native_integrations=0, optimizer_updates=0, rows=returned))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("worker", "catalogue", "endpoint", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
