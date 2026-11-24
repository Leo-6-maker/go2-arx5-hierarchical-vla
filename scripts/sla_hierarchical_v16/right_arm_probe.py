"""CPU warm-load and derivative checks on exactly two sealed TRAIN teacher queries."""

import argparse
import dataclasses
import hashlib
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from checkpoint_inputs import configured_transforms
from demonstrated_grip import pin, read, require
from flax import nnx
from flax.traverse_util import flatten_dict
from openpi.models.model import Observation
from openpi.training.data_loader import transform_dataset
from right_arm_recipe import active_velocity_loss, candidate, flow_inputs
from target_labels import TargetExamples
from train_archive_qualification import SEED_REPORT
from visual_lift_panel import write_new


def run(root):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and jax.default_backend() == "cpu",
        "CPU only",
    )
    cfg, data, stats = configured_transforms(root)
    vendor = root / "vendor/XPolicyLab/policy/SLA_Memory"
    manifest = read(root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json")
    require(
        manifest["model_revision"] == "7f6880569c3be777291e646fad1b4951245e8553"
        and manifest["source_commit"].startswith("c1b205ec"),
        "parent identity",
    )
    for item in manifest["verified_checkpoint_files"]:
        pin(vendor / "checkpoints" / item["path"], item["local_content_sha256"])
    ckpt = vendor / "checkpoints/SLA_Memory_robodojo_real_ckpt/params"
    recipe = candidate(cfg, ckpt)
    state = jax.eval_shape(
        lambda k: nnx.state(recipe.model.create(k)), jax.random.key(1729)
    )
    flat = flatten_dict(state.to_pure_dict())
    trainable = flatten_dict(state.filter(recipe.trainable_filter).to_pure_dict())
    with ocp.PyTreeCheckpointer() as cp:
        meta = {
            k[:-1]: v
            for k, v in flatten_dict(cp.metadata(ckpt)["params"]).items()
            if k[-1] == "value"
        }
    require(
        set(flat) - set(meta) == set(trainable)
        and set(meta) <= set(flat)
        and trainable
        and all("lora" in k[-1] for k in trainable)
        and all(tuple(flat[k].shape) == tuple(meta[k].shape) for k in meta),
        "incompatible parent tree",
    )
    # Orbax restores NumPy arrays; allow enough available RAM for the checkpoint and dtype conversion.
    with open("/proc/meminfo") as f:
        available = (
            int(next(line.split()[1] for line in f if line.startswith("MemAvailable:")))
            * 1024
        )
    require(
        available
        > 2 * sum(v.dtype.itemsize * math.prod(v.shape) for v in flat.values())
        + (2 << 30),
        "insufficient free host RAM for warm-load",
    )
    print(
        "CPU restore", len(meta), "base leaves", len(trainable), "adapters", flush=True
    )
    weights = flatten_dict(recipe.weight_loader.load(state.to_pure_dict()))
    require(weights.keys() == flat.keys(), "partial warm start")
    digests = {}
    for k, v in weights.items():
        require(
            v.shape == flat[k].shape and v.dtype == flat[k].dtype,
            "warm start shape/dtype",
        )
        digests["/".join(k)] = hashlib.sha256(
            memoryview(np.ascontiguousarray(v).view(np.uint8))
        ).hexdigest()
    for k in trainable:
        if k[-1].endswith("lora_b"):
            require(not np.any(weights[k]), "parent output perturbed at initialization")
        else:
            require(
                k[-1].endswith("lora_a") and np.any(weights[k]), "degenerate adapter"
            )
    del weights
    idx = read(pin(root / "v16/target-labels04/result.json", SEED_REPORT))["episodes"]
    picks = [
        ("CHV1-TRAIN-F1-HOLD-red-02", "CLOSE"),
        ("CHV1-TRAIN-F1-PLACE-red-right-01", "OPEN"),
    ]
    points = []
    samples = []
    for name, phase in picks:
        folder = root / "v16/target-labels04" / name
        for file, sha in idx[name]["artifacts"].items():
            pin(folder / file, sha)
        from demonstrated_grip import rows

        audit = rows(folder / "native_audit.jsonl")
        i = next(i for i, x in enumerate(audit) if x["phase"] == phase)
        ds = transform_dataset(
            TargetExamples(folder), dataclasses.replace(data, norm_stats=stats)
        )
        samples.append(ds[i])
        points.append({"episode": name, "phase": phase, "tick": audit[i]["tick"]})
    batch = jax.tree.map(lambda *leaves: jnp.asarray(np.stack(leaves)), *samples)
    actions = batch["actions"]
    observation = Observation.from_dict(batch)
    loss_shape = jax.eval_shape(
        lambda k, obs, targets: recipe.model.create(k).compute_loss(
            k, obs, targets, train=True
        ),
        jax.random.key(12),
        observation,
        actions,
    )
    require(loss_shape.shape == (2, 50), "end-to-end masked training loss shape")
    noise = jax.random.normal(jax.random.key(75), actions.shape)
    time = jnp.array([0.25, 0.75])[:, None, None]
    flags = (jnp.arange(32) >= 7) & (jnp.arange(32) < 14)

    def probe(a):
        x, target = flow_inputs(a, noise, time)
        # Dense diagnostic coupling makes any leak through the noisy input detectable.
        predicted = jnp.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
        return active_velocity_loss(predicted, target).mean()

    baseline = float(probe(actions))
    changed = float(probe(jnp.where(flags, actions, actions + 123)))
    grad = np.asarray(jax.grad(probe)(actions))
    axis_norms = np.linalg.norm(grad.reshape(-1, 32), axis=0)
    require(
        changed == baseline
        and np.all(axis_norms[~np.asarray(flags)] == 0)
        and np.all(axis_norms[np.asarray(flags)] > 0),
        "target-mask derivative leak",
    )
    return {
        "parent_revision": manifest["model_revision"],
        "source": manifest["source_commit"],
        "parent_file_count": len(manifest["verified_checkpoint_files"]),
        "base_leaves": len(meta),
        "trainable_parameters": sum(math.prod(v.shape) for v in trainable.values()),
        "all_parameters": sum(math.prod(v.shape) for v in flat.values()),
        "abstract_loss_shape": loss_shape.shape,
        "trainable_shapes": {"/".join(k): v.shape for k, v in trainable.items()},
        "warm_loaded_sha256": digests,
        "inactive_label_test": {
            "official_samples": points,
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "active_values_per_slot": 100,
            "diagnostic_loss_before_after": [baseline, changed],
            "label_gradient_norm_by_slot": axis_norms.tolist(),
        },
        "network_sample_forwards": 0,
        "optimizer_updates": 0,
        "physics_ticks": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    write_new(a.out, run(a.root.resolve()))
