"""Trace the live R5 optimizer entry and test its admission without running training."""

import argparse
from contextlib import redirect_stdout
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from r5_training_owner import BASE, checksum, require_inherited_gpu_owner
from r5_training_kernel import exact_frontier, make_kernel, prepare
from r5_optimizer_preflight import pack_examples
from demonstrated_grip import require
from visual_lift_panel import write_new

import jax
import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict


def rejected(operation, error, message):
    try:
        operation()
    except error as exc:
        require(str(exc) == message, f"wrong rejection: {exc}")
    else:
        raise AssertionError("unexpected admission: " + message)


def guard_replay(root, legacy):
    trial = "r5-smoke-cold-guard-check"
    output = root / "v18" / trial
    require(not output.exists(), "isolated cold name is already used")
    owner = BASE / "scripts/sla_go2_v18/r5_training_owner.py"
    with tempfile.TemporaryDirectory(dir=root / "v18", prefix="r5-guard-") as work:
        folder = Path(work)
        spec = dict(trial=trial, mode="START", start_step=0, stop_step=1,
                    training_step_cap=100, deadline_epoch=time.time() + 10,
                    scientific_allocation=None)
        for permit, expected in (
            (None, "R5_SCIENTIFIC_ALLOCATION_REQUIRED"),
            (dict(path=str(legacy), sha256=checksum(legacy)), "R5_EXACT_FINITE_GPU_SCIENCE_SCOPE_REQUIRED"),
        ):
            item = dict(spec, scientific_allocation=permit)
            receipt = folder / ("old.spec.json" if permit else "empty.spec.json")
            write_new(receipt, item)
            cmd = [sys.executable, "-B", str(owner), str(root), str(receipt)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            require(proc.returncode != 0 and f"PermissionError: {expected}" in proc.stderr
                    and not output.exists(), "real entry crossed a cold allocation gate")

        lock = folder / "isolated.lock"
        lock.touch()
        lease = dict(gpu_lock=str(lock))
        with lock.open("rb") as fd, lock.open("rb") as other:
            rejected(lambda: require_inherited_gpu_owner(lease, fd.fileno()), PermissionError,
                     "R5_GPU_LOCK_NOT_HELD")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_inherited_gpu_owner(lease, fd.fileno())
            rejected(lambda: require_inherited_gpu_owner(lease, other.fileno()), PermissionError,
                     "R5_GPU_LOCK_NOT_INHERITED")

        archive = folder / "frontier-fixture"
        cp, updates = archive / "checkpoints", archive / "updates"
        cp.mkdir(parents=True)
        updates.mkdir()
        (cp / "step-00000").mkdir()
        (cp / "step-00001").mkdir()
        intent = updates / "step-00001.json"
        write_new(intent, dict(status="INTENT", completed_steps=1))
        rejected(lambda: exact_frontier(archive, 0), ValueError,
                 "checkpoint frontier differs from allocation; reconcile any unmatched intent")
        rejected(lambda: exact_frontier(archive, 1), ValueError,
                 "previous update lacks a committed receipt")

    return dict(missing_science_rejected=True, old_science_rejected=True,
                isolated_inherited_flock_pass=True, uncommitted_step_never_replayed=True)


def kernel_trace(root):
    cfg, data, stats, recipe, layout, release = prepare(root)
    packed, selected = pack_examples(root, data, stats)
    tx, step = make_kernel(cfg, recipe)

    def transaction(key, batch):
        model = recipe.model.create(key)
        model.train()
        frozen = nnx.state(model, nnx.Not(recipe.trainable_filter))
        nnx.update(model, jax.tree.map(lambda p: p.astype(jnp.bfloat16), frozen))
        optimizer = tx.init(nnx.state(model, recipe.trainable_filter))
        updated, stats = step(model, optimizer, batch, jnp.int32(0))
        return nnx.state(model, recipe.trainable_filter), updated, stats

    right, moments, metrics = jax.eval_shape(transaction, jax.random.key(1729), packed)
    got = {"/".join(k): {"shape": list(x.shape), "dtype": str(x.dtype)}
           for k, x in flatten_dict(right.to_pure_dict()).items()}
    require(got == layout and len(jax.tree.leaves(moments)) == 22
            and all(leaf.shape == () for leaf in metrics.values()),
            "production transaction ABI differs")
    return dict(
        actual_rgb_receipts=selected, batch_shape=list(packed["actions"].shape),
        model_trainable_leaves=len(got), adamw_saved_leaves=len(jax.tree.leaves(moments)),
        scalar_metrics={k: str(v.dtype) for k, v in metrics.items()},
        frozen_dtype="bfloat16",
        official_manifest_sha256=checksum(release),
        implementation_sha256={
            key: checksum(BASE / "scripts/sla_go2_v18" / key)
            for key in (
                "r5_training_owner.py", "r5_training_kernel.py", "r5_checkpoint.py",
                "audit_r5_training.py", "r5_optimizer_preflight.py",
            )
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("legacy_allocation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and jax.default_backend() == "cpu",
            "cold CPU path only")
    with redirect_stdout(sys.stderr):
        result = dict(
            scope="COLD_TRAINING_OWNER_AND_ABSTRACT_LIVE_KERNEL", status="AWAITING_SCIENTIFIC_ALLOCATION",
            maximum_science_candidate_updates=100, actual_optimizer_updates=0,
            model_forward_executions=0, native_starts=0, integration_ticks=0,
            resource_leases_acquired=0, **guard_replay(args.root, args.legacy_allocation),
            **kernel_trace(args.root),
        )
    write_new(args.output, result)
