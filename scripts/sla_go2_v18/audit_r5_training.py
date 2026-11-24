"""CPU approval-independent audit of the frozen R5 training input, labels and balanced sampler."""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require, rows
from target_labels import TargetExamples

CUSTODY = BASE / "reports/SLA_GO2_V17/ARCHIVE_READBACK.json"
THRESHOLDS_M = (0.002, 0.042)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_index(root):
    files = read(CUSTODY)["files"]
    desc = pin(
        root / "v17/train-attempt01/result.json", files["TRAIN_QUALIFICATION.json"]
    )
    require(
        desc.read_bytes() == (CUSTODY.parent / "TRAIN_QUALIFICATION.json").read_bytes(),
        "published TRAIN receipt differs",
    )
    ep = read(desc)["episodes"]
    sampler = read(pin(root / "v17/sampling01.json", files["sampling01.json"]))
    require(sampler["training_draws"] == 0, "source must be the frozen offline sampler")
    expected = {}
    for episode, record in ep.items():
        if record["status"] == "excluded_original_failure":
            continue
        require(
            "-TRAIN-" in episode
            and record["status"] in ("verified_seed", "verified_expansion"),
            "non-TRAIN example",
        )
        prepared = record["prepared"]
        folder = (
            root
            / (
                "v16/target-labels04"
                if record["status"] == "verified_seed"
                else "v17/train-attempt01"
            )
            / episode
        )
        for file, sha in prepared["artifacts"].items():
            pin(folder / file, sha)
        expected[episode] = (folder, record)
    keys = [(r["episode"], r["query"]) for r in sampler["samples"]]
    all_keys = {
        (n, i)
        for n, (_, r) in expected.items()
        for i in range(r["prepared"]["official_rows"])
    }
    probability = np.array([r["probability"] for r in sampler["samples"]])
    require(
        len(expected) == 57
        and len(keys) == len(all_keys) == len(set(keys)) == 4898
        and set(keys) == all_keys
        and np.isfinite(probability).all()
        and np.all(probability > 0)
        and abs(probability.sum() - 1) < 1e-10,
        "sampler TRAIN denominator mismatch",
    )
    return expected, keys, probability


def phase_tick_metrics(folder, record):
    lo, hi = THRESHOLDS_M
    with np.load(folder / "sensor_action.npz", allow_pickle=False) as z:
        # The old target row stores the H40 driver's fraction. Keep the saved archive intact.
        grip_target = 0.044 * z["nominal_action"][:, :, 15].astype(np.float64)
        grip_state = 0.044 * z["state"][:, 15].astype(np.float64)
        ticks = z["query_tick"]
    n = len(ticks)
    audit = rows(folder / "native_audit.jsonl")
    require(
        n == len(audit)
        and np.all(np.diff(ticks) > 0)
        and np.array_equal(ticks, [r["tick"] for r in audit])
        and np.isfinite(grip_target).all()
        and np.isfinite(grip_state).all(),
        "sensor, label and clock binding",
    )
    events = record["native_event_ticks"]
    first = int(np.searchsorted(ticks, events["target_grasp"]))
    phase = [r["phase"] for r in audit]
    diag = dict(
        first_post_grasp_query=first if first < n else None,
        has_post_grasp_before_lift=bool(
            first < n and ticks[first] < events["target_lift20mm"]
        ),
        grip_command_points=int(grip_target.size),
        below_2mm=int(np.count_nonzero(grip_target < lo - 2e-8)),
        above_42mm=int(np.count_nonzero(grip_target > hi + 2e-8)),
        equal_44mm=int(
            np.count_nonzero(np.isclose(grip_target, 0.044, rtol=0, atol=2e-8))
        ),
        command_range_mm=[
            float(1000 * grip_target.min()),
            float(1000 * grip_target.max()),
        ],
        state_range_mm=[float(1000 * grip_state.min()), float(1000 * grip_state.max())],
        worst_command_error_mm_if_clipped_to_adapter=float(
            1000 * np.max(np.abs(grip_target - np.clip(grip_target, lo, hi)))
        ),
    )
    return ticks, phase, diag


def run(root):
    import jax
    import jax.numpy as jnp
    from checkpoint_inputs import configured_transforms
    from openpi.models.model import Observation
    from openpi.training.data_loader import transform_dataset
    from right_arm_recipe import candidate

    require(
        jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "CPU only",
    )
    cfg, data, stats = configured_transforms(root)
    verified, keys, probability = source_index(root)
    points = {}
    aggregate = Counter()
    count_by_family = Counter()
    mass = defaultdict(float)
    firsts = set()
    for name, (folder, desc) in verified.items():
        ticks, phase, measured = phase_tick_metrics(folder, desc)
        first = measured["first_post_grasp_query"]
        if first is not None:
            firsts.add((name, first))
        count_by_family[f"{desc['family']}/{desc['task']}"] += 1
        for k in ("grip_command_points", "below_2mm", "above_42mm", "equal_44mm"):
            aggregate[k] += measured[k]
        points[name] = (ticks, phase, measured)
    for (name, i), p in zip(keys, probability, strict=True):
        mass[verified[name][1]["task"] + "/" + points[name][1][i]] += float(p)

    # Stable draw by absolute batch index. Restarting at step 37 reproduces the
    # same choices without retaining a random-generator cursor in the checkpoint.
    seed = 1729
    batch_size = 2
    shadow_batches = 100
    sampler_cdf = np.cumsum(probability)
    sampler_cdf[-1] = 1.0

    def draw(step):
        r = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step])))
        return [
            keys[i]
            for i in np.searchsorted(sampler_cdf, r.random(batch_size), side="right")
        ]

    itinerary = [draw(i) for i in range(shadow_batches)]
    event_banks = defaultdict(list)
    for key in sorted(firsts):
        name, _ = key
        if points[name][2]["has_post_grasp_before_lift"]:
            event_banks[verified[name][1]["task"]].append(key)
    tasks = sorted(event_banks)
    require(
        len(tasks) == 3 and sum(map(len, event_banks.values())) == 57,
        "uncovered post-grasp branch",
    )

    def event_draw(step):
        batch = draw(step)
        if step % 5 == 0:
            task = tasks[(step // 5) % len(tasks)]
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, 1]))
            bank = event_banks[task]
            batch[0] = bank[int(rng.integers(len(bank)))]
        return batch

    event_itinerary = [event_draw(i) for i in range(shadow_batches)]
    require(
        draw(37) == itinerary[37] and event_draw(35) == event_itinerary[35],
        "nonreproducible addressable sampler",
    )
    require(
        all(
            b[0] in firsts and b[1] == a[1] if i % 5 == 0 else a == b
            for i, (a, b) in enumerate(zip(itinerary, event_itinerary, strict=True))
        ),
        "events changed an unreserved slot",
    )
    prepared = dataclasses.replace(data, norm_stats=stats)

    @lru_cache(maxsize=4)
    def dataset(episode):
        return transform_dataset(TargetExamples(verified[episode][0]), prepared)

    def batch(indices):
        examples = [dataset(n)[i] for n, i in indices]
        return jax.tree.map(lambda *xs: np.stack(xs), *examples)

    files = (
        root
        / "vendor/XPolicyLab/policy/SLA_Memory/checkpoints/SLA_Memory_robodojo_real_ckpt/params"
    )
    recipe = candidate(cfg, files)
    network_shape = None
    dataset_digests = defaultdict(list)
    visits = defaultdict(Counter)
    for name_of_sampler, schedule in (
        ("original", itinerary),
        ("post_grasp_reserved", event_itinerary),
    ):
        for step, indices in enumerate(schedule):
            obs = batch(indices)
            y = obs["actions"]
            require(
                y.shape == (batch_size, 50, 32)
                and np.isfinite(y).all()
                and obs["extra_images"].shape == (batch_size, 9, 224, 224, 3)
                and len(obs["image"]) == 3
                and np.all(y[:, :, 14:] == 0),
                "official processor sample contract",
            )
            for name, i in indices:
                visits[name_of_sampler][
                    verified[name][1]["task"] + "/" + points[name][1][i]
                ] += 1
            dataset_digests[name_of_sampler].append(
                dict(
                    step=step,
                    sample_ids=indices,
                    actions_sha256=digest(y),
                    history_sha256=digest(obs["extra_images"]),
                    tokens_sha256=digest(obs["tokenized_prompt"]),
                )
            )
            if network_shape is None:
                copied = jax.tree.map(jnp.asarray, obs)
                network_shape = jax.eval_shape(
                    lambda key, x, target: recipe.model.create(key).compute_loss(
                        key, x, target, train=True
                    ),
                    jax.random.key(seed),
                    Observation.from_dict(copied),
                    copied["actions"],
                ).shape
                require(network_shape == (batch_size, 50), "model and data ABI drift")
    return dict(
        archive_episodes=len(verified),
        queries=len(keys),
        episode_count_by_family_task=dict(count_by_family),
        grip_domain_points=dict(aggregate),
        grip_ranges_mm={n: p[2] for n, p in points.items()},
        post_grasp_before_lift_episodes=sum(
            p[2]["has_post_grasp_before_lift"] for p in points.values()
        ),
        first_post_grasp_probability=float(
            sum(p for k, p in zip(keys, probability, strict=True) if k in firsts)
        ),
        first_post_grasp_preview_count=sum(i in firsts for b in itinerary for i in b),
        preview_seed=seed,
        preview_batch_size=batch_size,
        preview_batches=shadow_batches,
        sampler_mass=dict(mass),
        event_reservation={
            "fixed_batches": 20,
            "candidate_examples": len(firsts),
            "first_post_grasp_draws": sum(
                i in firsts for b in event_itinerary for i in b
            ),
            "distinct_first_post_grasp_draws": len(
                {i for b in event_itinerary for i in b if i in firsts}
            ),
            "rule": "Replace only batch position 0 on absolute steps divisible by 5; round-robin task.",
        },
        preview_phase_visits={k: dict(v) for k, v in visits.items()},
        official_preview_hashes=dict(dataset_digests),
        abstract_loss_shape=list(network_shape),
        real_training_steps=0,
        trainable_parameters_updated=0,
        physics_ticks=0,
        frozen_forwards=0,
        data_scope="TRAIN only; nominal H40 command mapped to mm for diagnosis, no changes to actions or normalization",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    with redirect_stdout(sys.stderr):
        result = run(a.root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
