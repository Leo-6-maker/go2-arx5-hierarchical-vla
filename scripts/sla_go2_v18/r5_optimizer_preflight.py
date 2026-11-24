"""Trace the right-arm-only optimizer on sealed RGB, without executing or loading weights."""

import argparse
import dataclasses
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require, rows


def update_right_arm(
    model, optimizer, optimizer_state, trainable, rng, observation, actions
):
    """One candidate update, using the official optimizer recipe and only LoRA leaves."""
    import jax.numpy as jnp
    import optax
    from flax import nnx

    def loss(net):
        return jnp.mean(net.compute_loss(rng, observation, actions, train=True))

    right = nnx.state(model, trainable)
    value, grad = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, trainable))(model)
    update, next_state = optimizer.update(grad, optimizer_state, right)
    next_params = optax.apply_updates(right, update)
    nnx.update(model, next_params)
    return next_state, value, optax.global_norm(grad)


def pack_examples(root, data, stats):
    import jax
    from target_labels import TargetExamples
    from train_archive_qualification import SEED_REPORT
    from openpi.training.data_loader import transform_dataset

    bank = root / "v16/target-labels04"
    index = read(pin(bank / "result.json", SEED_REPORT))["episodes"]
    picks = (
        ("CHV1-TRAIN-F1-HOLD-red-02", "CLOSE"),
        ("CHV1-TRAIN-F1-PLACE-red-right-01", "OPEN"),
    )
    examples, identity = [], []
    for episode, phase in picks:
        folder = bank / episode
        for file, digest in index[episode]["artifacts"].items():
            pin(folder / file, digest)
        i, record = next(
            (i, a)
            for i, a in enumerate(rows(folder / "native_audit.jsonl"))
            if a["phase"] == phase
        )
        ds = transform_dataset(
            TargetExamples(folder), dataclasses.replace(data, norm_stats=stats)
        )
        examples.append(ds[i])
        identity.append(
            {"episode": episode, "tick": record["tick"], "phase_for_audit": phase}
        )
    return jax.tree.map(lambda *x: np.stack(x), *examples), identity


def shape_check(root):
    import jax
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from openpi.models.model import Observation
    from openpi.training.optimizer import create_optimizer
    from checkpoint_inputs import configured_transforms
    from right_arm_recipe import candidate

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and jax.default_backend() == "cpu",
        "abstract CPU check only",
    )
    cfg, data, stats = configured_transforms(root)
    ckpt = (
        root
        / "vendor/XPolicyLab/policy/SLA_Memory/checkpoints/SLA_Memory_robodojo_real_ckpt/params"
    )
    recipe = candidate(cfg, ckpt)
    batch, picks = pack_examples(root, data, stats)
    obs = Observation.from_dict(batch)
    optimizer = create_optimizer(cfg.optimizer, cfg.lr_schedule)

    def abstract_transaction(key, observation, actions):
        init_key, step_key = jax.random.split(key)
        model = recipe.model.create(init_key)
        adapters = nnx.state(model, recipe.trainable_filter)
        optim = optimizer.init(adapters)
        next_optim, loss, norm = update_right_arm(
            model,
            optimizer,
            optim,
            recipe.trainable_filter,
            step_key,
            observation,
            actions,
        )
        return (
            adapters,
            nnx.state(model, recipe.trainable_filter),
            optim,
            next_optim,
            loss,
            norm,
        )

    before, after, opt_init, opt_after, loss, norm = jax.eval_shape(
        abstract_transaction, jax.random.key(1729), obs, batch["actions"]
    )

    def layout(tree):
        return {
            "/".join(k): {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in flatten_dict(tree.to_pure_dict()).items()
        }

    keys = layout(before)
    prior = read(BASE / "reports/SLA_GO2_V17/RIGHT_ARM_RECIPE_CPU.json")
    require(
        keys == layout(after)
        and {k: v["shape"] for k, v in keys.items()} == prior["trainable_shapes"],
        "there must be ten and only ten registered LoRA leaves",
    )
    require(
        jax.tree.structure(opt_init) == jax.tree.structure(opt_after)
        and loss.shape == norm.shape == (),
        "optimizer transaction ABI changed",
    )
    for x, y in zip(jax.tree.leaves(opt_init), jax.tree.leaves(opt_after), strict=True):
        require(
            x.shape == y.shape and x.dtype == y.dtype,
            "optimizer slot changed after a step",
        )
    return dict(
        batch_count=len(picks),
        examples=picks,
        abstract_loss=list(loss.shape),
        abstract_grad_norm=list(norm.shape),
        optimizer_config=dataclasses.asdict(cfg.optimizer),
        schedule_config=dataclasses.asdict(cfg.lr_schedule),
        trainable=keys,
        frozen_params_updated=0,
        model_forward_executions=0,
        optimizer_updates_executed=0,
        optimizer_state_leaf_count=len(jax.tree.leaves(opt_after)),
        policy_checkpoint_exported=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = shape_check(args.root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
