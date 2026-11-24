"""Finite right-arm optimizer wiring. The owner gates the only runnable entry."""

import dataclasses
import math
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from audit_r5_training import (
    BATCH_SIZE, CUSTODY, SAMPLER_SEED, addressable_schedule, digest,
    official_batches, phase_tick_metrics, source_index,
)
from checkpoint_inputs import configured_transforms
from demonstrated_grip import pin, read, require
from openpi.models.model import Observation
from openpi.training.optimizer import create_optimizer
from openpi.training.utils import train_rng
from r5_checkpoint import binding_digest, payload_digest, restore, save
from r5_optimizer_preflight import update_right_arm
from right_arm_recipe import candidate
from target_labels import checksum

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from scripts.pi0292.durable import write

REPORTS = BASE / "reports/SLA_GO2_V18/P1_P2"
PARENT_REPORT = BASE / "reports/SLA_GO2_V17/RIGHT_ARM_RECIPE_CPU.json"


def prepare(root):
    cfg, data, stats = configured_transforms(root)
    release = root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json"
    official = read(release)
    info = read(REPORTS / "OPTIMIZER_ABI.json")
    require(
        official["model_revision"] == "7f6880569c3be777291e646fad1b4951245e8553"
        and official["source_commit"].startswith("c1b205ec")
        and dataclasses.asdict(cfg.optimizer) == info["optimizer_config"]
        and dataclasses.asdict(cfg.lr_schedule) == info["schedule_config"],
        "registered parent or optimizer profile changed",
    )
    policy = root / "vendor/XPolicyLab/policy/SLA_Memory"
    checkpoint = policy / "checkpoints/SLA_Memory_robodojo_real_ckpt/params"
    return cfg, data, stats, candidate(cfg, checkpoint), info["trainable"], release


def make_kernel(cfg, recipe):
    from optax import global_norm

    tx = create_optimizer(cfg.optimizer, cfg.lr_schedule)
    schedule = cfg.lr_schedule.create()
    sampler_base, _ = jax.random.split(jax.random.key(cfg.seed))

    def attempt(model, state, packed, step):
        obs = Observation.from_dict(packed)
        key = train_rng(sampler_base, step, 0, 1)
        next_state, loss, norm = update_right_arm(
            model, tx, state, recipe.trainable_filter, key, obs, packed["actions"]
        )
        return next_state, {
            "loss": loss, "grad_norm": norm, "lr": schedule(step),
            "adapter_norm": global_norm(nnx.state(model, recipe.trainable_filter)),
        }

    return tx, nnx.jit(attempt)


def initialize_frozen(root, recipe, layout, release):
    policy = root / "vendor/XPolicyLab/policy/SLA_Memory"
    for item in read(release)["verified_checkpoint_files"]:
        pin(policy / "checkpoints" / item["path"], item["local_content_sha256"])
    model_shape = jax.eval_shape(lambda k: nnx.state(recipe.model.create(k)), jax.random.key(1729))
    shape_leaves = jax.tree.leaves(model_shape)
    minimum_ram = 2 * sum(math.prod(x.shape) * x.dtype.itemsize for x in shape_leaves) + (2 << 30)
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    free_ram = int(next(line.split()[1] for line in meminfo if line.startswith("MemAvailable:"))) * 1024
    require(free_ram > minimum_ram, "insufficient host RAM for frozen checkpoint warm load")
    raw = recipe.weight_loader.load(model_shape.to_pure_dict())
    warm = read(PARENT_REPORT)["warm_loaded_sha256"]
    flat = flatten_dict(raw)
    require(set(warm) == {"/".join(k) for k in flat}, "frozen parent parameter set changed")
    for key, value in flat.items():
        path = "/".join(key)
        require(
            digest(value) == warm[path],
            "frozen warm load or initialized LoRA changed:" + path,
        )
    del flat
    # Follow the official training entry's frozen BF16 conversion. Keep the
    # audited right-arm LoRA FP32 and do not export a duplicate parent.
    placed = {}
    for key, value in flatten_dict(raw).items():
        dtype = np.float32 if "/".join(key) in layout else jnp.bfloat16
        placed[key] = jax.device_put(np.asarray(value, dtype=dtype))
    del raw
    model = recipe.model.load(unflatten_dict(placed), remove_extra_params=False)
    model.train()
    return model


def frozen_fingerprints(model, recipe):
    frozen = nnx.state(model, nnx.Not(recipe.trainable_filter)).to_pure_dict()
    return {"/".join(k): digest(v) for k, v in flatten_dict(frozen).items()}


def exact_frontier(output, start):
    checkpoints = output / "checkpoints"
    updates = output / "updates"
    require(checkpoints.is_dir() and updates.is_dir(), "existing trial missing its durable roots")
    steps = sorted(int(p.name[5:]) for p in checkpoints.glob("step-[0-9][0-9][0-9][0-9][0-9]"))
    records = sorted(int(p.stem[5:]) for p in updates.glob("step-[0-9][0-9][0-9][0-9][0-9].json"))
    require(steps == list(range(start + 1)) and records == list(range(1, start + 1)),
            "checkpoint frontier differs from allocation; reconcile any unmatched intent")
    for step in records:
        record = read(updates / f"step-{step:05d}.json")
        require(record["status"] == "COMMITTED" and record["completed_steps"] == step,
                "previous update lacks a committed receipt")
    require(not (output / "STOP").exists(), "trial is explicitly stopped")


def train(root, output, spec, available_mib):
    require(jax.default_backend() == "gpu" and len(jax.devices()) == 1, "actual single allocated GPU required")
    start, stop = spec["start_step"], spec["stop_step"]
    fresh = spec["mode"] == "START"
    if fresh:
        require(start == 0 and not output.exists(), "an initial segment needs a new trial")
    else:
        require(spec["mode"] == "RESUME", "invalid start mode")
        exact_frontier(output, start)

    cfg, data, stats, recipe, layout, release = prepare(root)
    binding = dict(
        scope=spec["trial"], run_purpose="GO2_POST_TRAINING_REAL_GPU_SMOKE",
        max_steps=spec["training_step_cap"], batch_size=BATCH_SIZE, sampler="post_grasp_reserved",
        official_train_seed=cfg.seed, sampler_seed=SAMPLER_SEED,
        source_files=spec["source_files"], parent_manifest_sha256=checksum(release),
        train_custody_sha256=checksum(CUSTODY),
        train_preview_sha256=checksum(REPORTS / "TRAIN_INPUT_ACTUAL.json"),
        validation_sha256=spec["independent_val"]["sha256"],
    )
    identity = binding_digest(binding)
    if not fresh:
        require(read(output / "BINDING.json") == binding, "continuation identity changed")
    verified, keys, probability = source_index(root)
    points = {name: phase_tick_metrics(folder, info) for name, (folder, info) in verified.items()}
    draw = addressable_schedule(verified, keys, probability, points)
    batch = official_batches(verified, data, stats)
    registered = read(REPORTS / "TRAIN_INPUT_ACTUAL.json")["official_preview_hashes"]["post_grasp_reserved"]
    require(len(registered) == 100, "missing reviewed 100-step itinerary")
    tx, apply_update = make_kernel(cfg, recipe)
    model = initialize_frozen(root, recipe, layout, release)
    base = frozen_fingerprints(model, recipe)
    require(len(base) == 51, "frozen parent leaf count changed")
    adapters = nnx.state(model, recipe.trainable_filter)
    adamw = tx.init(adapters)
    if fresh:
        output.mkdir(mode=0o700)
        (output / "checkpoints").mkdir()
        (output / "updates").mkdir()
        write(output / "BINDING.json", binding)
        write(output / "FROZEN_SHA256.json", base)
        save(output / "checkpoints", completed=0, adapters=adapters.to_pure_dict(),
             optimizer=adamw, binding=binding, adapter_layout=layout)
    else:
        require(base == read(output / "FROZEN_SHA256.json"), "restored parent drift")
        template = dict(completed_steps=np.int32(0), adapters=adapters.to_pure_dict(), adamw=adamw)
        recovered = restore(output / "checkpoints", completed=start, template=template,
                            binding=binding, adapter_layout=layout)
        adapters.replace_by_pure_dict(jax.device_put(recovered["adapters"]))
        nnx.update(model, adapters)
        adamw = jax.device_put(recovered["adamw"])

    allocation_sha = spec["scientific_allocation"]["sha256"]
    segment = output / f"segment-{start:05d}-{stop:05d}-{allocation_sha[:16]}.json"
    require(not segment.exists(), "segment already recorded")
    write(segment, dict(status="ACTIVE", binding_sha256=identity, allocation_sha256=allocation_sha,
                        start_step=start, stop_step=stop,
                        free_vram_mib_before=available_mib, actual_updates=0, unknown_step=None))
    committed, unknown = start, None
    failure = None
    try:
        for step in range(start, stop):
            if (output / "STOP").exists() or time.time() >= spec["deadline_epoch"]:
                break
            indices = draw(step, reserve_post_grasp=True)
            packed = batch(indices)
            truth = registered[step]
            require(
                truth["step"] == step and truth["sample_ids"] == [list(i) for i in indices]
                and truth["actions_sha256"] == digest(packed["actions"])
                and truth["history_sha256"] == digest(packed["extra_images"])
                and truth["tokens_sha256"] == digest(packed["tokenized_prompt"]),
                "production batch differs from the reviewed itinerary",
            )
            if (output / "STOP").exists() or time.time() >= spec["deadline_epoch"]:
                break

            slot = output / "updates" / f"step-{step + 1:05d}.json"
            require(not slot.exists(), "update was already attempted")
            details = dict(completed_steps=step + 1, batch_sha256=payload_digest(packed),
                           samples=[list(i) for i in indices], binding_sha256=identity)
            write(slot, dict(status="INTENT", **details))
            unknown = step + 1
            adamw, metadata = apply_update(model, adamw, jax.device_put(packed), jnp.int32(step))
            metrics = {key: float(value) for key, value in jax.device_get(metadata).items()}
            require(all(np.isfinite(v) for v in metrics.values()) and metrics["grad_norm"] > 0,
                    "nonfinite update or no adapter gradient")
            save(output / "checkpoints", completed=step + 1,
                 adapters=nnx.state(model, recipe.trainable_filter).to_pure_dict(),
                 optimizer=adamw, binding=binding, adapter_layout=layout)
            write(slot, dict(status="COMMITTED", metrics=metrics, **details))
            committed, unknown = step + 1, None
        require(frozen_fingerprints(model, recipe) == base, "frozen weights changed during training")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status = "FAILED" if failure else ("COMPLETED" if committed == stop else "SAFE_STOP")
        write(segment, dict(status=status, binding_sha256=identity, allocation_sha256=allocation_sha,
                            start_step=start, stop_step=stop,
                            actual_updates=committed - start, last_committed_step=committed,
                            unknown_step=unknown, failure=failure,
                            free_vram_mib_before=available_mib, native_starts=0, sampling_forwards=0))
    return 0
