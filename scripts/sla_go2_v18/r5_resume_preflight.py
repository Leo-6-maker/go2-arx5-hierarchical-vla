"""Exercise the production-format R5 checkpoint and real RGB batch restore on CPU."""

import argparse
from contextlib import redirect_stdout
import dataclasses
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import numpy as np

from audit_r5_training import (
    BATCH_SIZE, CUSTODY, SAMPLER_SEED, addressable_schedule, digest, official_batches,
    phase_tick_metrics, source_index,
)
from checkpoint_inputs import configured_transforms
from demonstrated_grip import read, require
from openpi.training import optimizer as optimizer_api
from openpi.training.utils import train_rng
from r5_checkpoint import COUNTERS, binding_digest, payload_digest, restore, save
from right_arm_recipe import candidate, initialize_lora
from target_labels import checksum
from visual_lift_panel import write_new
from flax import nnx
from flax.traverse_util import flatten_dict

BASE = Path(__file__).resolve().parents[2]
REPORTS = BASE / "reports/SLA_GO2_V18/P1_P2"
INITIAL = BASE / "reports/SLA_GO2_V17/RIGHT_ARM_RECIPE_CPU.json"
FIXTURE_STEP = 37


def layout_and_binding(root, cfg):
    abi_file = REPORTS / "OPTIMIZER_ABI.json"
    preview = REPORTS / "TRAIN_INPUT_ACTUAL.json"
    release = root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json"
    official = read(release)
    abi = read(abi_file)
    require(
        official["model_revision"] == "7f6880569c3be777291e646fad1b4951245e8553"
        and official["source_commit"].startswith("c1b205ec")
        and dataclasses.asdict(cfg.optimizer) == abi["optimizer_config"]
        and dataclasses.asdict(cfg.lr_schedule) == abi["schedule_config"],
        "parent or optimizer profile drift",
    )
    bindings = dict(
        # Tests have real tensor shapes and RGB. Their synthetic marker cannot
        # be restored by a scientific run with a different binding.
        run_purpose="CPU_SERIALIZATION_FIXTURE",
        parent_revision=official["model_revision"],
        parent_manifest_sha256=checksum(release),
        official_train_seed=cfg.seed,
        sampler_seed=SAMPLER_SEED,
        sampler="post_grasp_reserved",
        batch_size=BATCH_SIZE,
        train_custody_sha256=checksum(CUSTODY),
        preview_sha256=checksum(preview),
        optimizer_abi_sha256=checksum(abi_file),
        implementation_sha256={
            name: checksum(BASE / name)
            for name in (
                "scripts/sla_go2_v18/audit_r5_training.py",
                "scripts/sla_go2_v18/r5_checkpoint.py",
                "scripts/sla_hierarchical_v16/right_arm_recipe.py",
            )
        },
        max_steps=100,
    )
    return abi["trainable"], bindings


def initialized(root, cfg):
    checkpoint = (
        root / "vendor/XPolicyLab/policy/SLA_Memory/checkpoints/SLA_Memory_robodojo_real_ckpt/params"
    )
    recipe = candidate(cfg, checkpoint)
    state = jax.eval_shape(
        lambda key: nnx.state(recipe.model.create(key), recipe.trainable_filter),
        jax.random.key(1729),
    )
    initial = initialize_lora(state.to_pure_dict(), seed=recipe.weight_loader.seed)
    state.replace_by_pure_dict(initial)
    # Compare each leaf against the earlier actual official checkpoint warm load.
    archive = read(INITIAL)["warm_loaded_sha256"]
    weights = {
        "/".join(key): digest(value) for key, value in flatten_dict(initial).items()
    }
    require(all(archive[k] == v for k, v in weights.items()), "LoRA initialization drift")
    optimizer = optimizer_api.create_optimizer(cfg.optimizer, cfg.lr_schedule).init(state)
    return dict(completed_steps=np.int32(0), adapters=initial, adamw=optimizer)


def batch_readback(root, data, stats, cfg, step):
    verified, keys, probability = source_index(root)
    points = {
        name: phase_tick_metrics(folder, record)
        for name, (folder, record) in verified.items()
    }
    draw = addressable_schedule(verified, keys, probability, points)
    preview = read(REPORTS / "TRAIN_INPUT_ACTUAL.json")["official_preview_hashes"]
    # Verify both candidate and original itinerary across all 100 committed preview steps.
    for label, reserve in (("post_grasp_reserved", True), ("original", False)):
        require(
            all(
                [list(key) for key in draw(i, reserve_post_grasp=reserve)] == rec["sample_ids"]
                for i, rec in enumerate(preview[label])
            ),
            "sampler itinerary changed",
        )
    indices = draw(step, reserve_post_grasp=True)
    batch = official_batches(verified, data, stats)(indices)
    record = dict(
        step=step,
        sample_ids=[list(key) for key in indices],
        actions_sha256=digest(batch["actions"]),
        history_sha256=digest(batch["extra_images"]),
        tokens_sha256=digest(batch["tokenized_prompt"]),
    )
    require(record == preview["post_grasp_reserved"][step], "resumed official processor drift")
    # Include current views, masks, state and task tokens in the cross-process check.
    record["whole_batch_sha256"] = payload_digest(batch)
    rng, _ = jax.random.split(jax.random.key(cfg.seed))
    record["flow_rng_sha256"] = digest(jax.random.key_data(train_rng(rng, step, 0, 1)))
    return record


def mock_optimizer_marker(tree):
    # Materialize nonzero moments and independent optimizer counters without
    # submitting an optimizer update. This is an I/O fixture only.
    def replace(path, value):
        key = jax.tree_util.keystr(path)
        return np.full(value.shape, FIXTURE_STEP if key in COUNTERS else 0.125, value.dtype)
    return jax.tree_util.tree_map_with_path(replace, tree)


def read_in_fresh_process(root, checkpoints, output):
    cfg, data, stats = configured_transforms(root)
    layout, binding = layout_and_binding(root, cfg)
    template = initialized(root, cfg)
    restored = restore(checkpoints, completed=FIXTURE_STEP, template=template,
                       binding=binding, adapter_layout=layout)
    observed = int(restored["completed_steps"])
    write_new(output, dict(payload_sha256=payload_digest(restored),
                           next_batch=batch_readback(root, data, stats, cfg, observed)))


def run(root):
    cfg, data, stats = configured_transforms(root)
    layout, binding = layout_and_binding(root, cfg)
    template = initialized(root, cfg)
    with tempfile.TemporaryDirectory(dir=root / "v18", prefix="r5-resume-cpu-") as isolated:
        directory = Path(isolated)
        save(directory, completed=0, adapters=template["adapters"], optimizer=template["adamw"],
             binding=binding, adapter_layout=layout)
        initial = restore(directory, completed=0, template=template, binding=binding, adapter_layout=layout)
        require(payload_digest(initial) == payload_digest(template), "initial round trip changed")

        fixture = mock_optimizer_marker(template["adamw"])
        save(directory, completed=FIXTURE_STEP, adapters=template["adapters"], optimizer=fixture,
             binding=binding, adapter_layout=layout)
        expected = payload_digest(dict(completed_steps=np.int32(FIXTURE_STEP),
                                       adapters=template["adapters"], adamw=fixture))
        try:
            restore(directory, completed=FIXTURE_STEP, template=template,
                    binding=dict(binding, sampler="original"), adapter_layout=layout)
        except ValueError as exc:
            require(str(exc) == "checkpoint run identity changed", "wrong binding was not pin checked")
        else:
            raise AssertionError("wrong sampler restored")
        try:
            save(directory, completed=FIXTURE_STEP + 1, adapters=template["adapters"], optimizer=fixture,
                 binding=binding, adapter_layout=layout)
        except ValueError as exc:
            require("counters" in str(exc), "wrong counter failure")
        else:
            raise AssertionError("inconsistent counts were saved")
        (directory / f"step-{FIXTURE_STEP + 1:05d}.orbax-checkpoint-tmp").mkdir()
        try:
            restore(directory, completed=FIXTURE_STEP + 1, template=template,
                    binding=binding, adapter_layout=layout)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("unfinished checkpoint restored")

        uninterrupted = batch_readback(root, data, stats, cfg, FIXTURE_STEP)
        output = directory / "RESTORED.json"
        subprocess.run(
            [sys.executable, "-B", __file__, str(root), str(output),
             "--restore-fixture", str(directory)], check=True, timeout=300,
        )
        restored = read(output)
        require(restored["payload_sha256"] == expected and restored["next_batch"] == uninterrupted,
                "checkpoint replay differs from the uninterrupted path")

    return dict(
        scope="CPU_SERIALIZATION_AND_RGB_RESUME_FIXTURE",
        adapter_layout=layout, adapter_parameter_count=sum(np.prod(v["shape"]).item() for v in layout.values()),
        adamw_initial_round_trip=True, optimizer_moments_are_flax_nnx_state=True,
        synthetic_step_marker=FIXTURE_STEP,
        synthetic_state_sha256=expected, binding_sha256=binding_digest(binding),
        binding=binding,
        wrong_sampler_rejected=True, wrong_counter_rejected=True, partial_restore_rejected=True,
        restart_in_fresh_process=True, both_sampler_itineraries_verified=200,
        next_batch=uninterrupted, new_gpu_forwards=0, model_forward_executions=0,
        optimizer_updates_executed=0, native_starts=0, integration_ticks=0,
    )


if __name__ == "__main__":
    import os

    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("root", type=Path)
    parse.add_argument("output", type=Path)
    parse.add_argument("--restore-fixture", type=Path)
    args = parse.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and jax.default_backend() == "cpu",
            "CPU fixture only")
    with redirect_stdout(sys.stderr):
        if args.restore_fixture:
            read_in_fresh_process(args.root.resolve(), args.restore_fixture, args.output)
        else:
            write_new(args.output, run(args.root.resolve()))
