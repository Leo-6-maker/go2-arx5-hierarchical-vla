"""Replay the recorded V7-A Control step 251 without committing an update."""

import argparse
from contextlib import closing
from functools import partial
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v7_formal_worker import exact_noise_witness
from v7_full_model_qualification import MODEL_NOISE_SEED, PLAN, STEPS, load_inputs


BINDING = "5624b9e1226b10389f62908d1ef7959812183895452c89ac1a95fa7952360e09"
TERMINAL = "8995ff3926e25bde34632d75859e4d5c5578e986ff27a2a9b587ed065be54da8"
START = 250
SEAL = "522f9a0bb26e9cde8eb913b2f85aeb9353f743a098040750929b00c0ff3773f5"
ARM = "CONTROL_NEUTRAL"


def region_losses(learner, graph, original, delta, labels, micro, stop, dot):
    import jax
    import jax.numpy as jnp
    import optax
    from go2arx5_vla.sla_adapter.batch_v2 import device_batch, error32
    from go2arx5_vla.sla_adapter.scale20k_training import compile_fm_accumulation
    from v7_full_model_paths import conditioned_policy

    batch = learner.prepare(START, "target")
    observation, actions = device_batch(batch.host)
    noise, tau = learner.draws["target"](learner.rng, batch.cursor)
    support = jax.device_put(batch.support)
    condition = jax.device_put(batch.condition)
    active = {"shared": learner.parameters["shared"], "private": learner.parameters["target"]}
    update = {"shared": delta["shared"], "private": delta["target"]}

    def picked_loss(active, frozen, obs, actions, noise, tau, support, condition, selector):
        masked = jnp.where(support[..., None], actions, 0.)
        with jax.default_matmul_precision("highest"):
            policy = conditioned_policy(graph, active["shared"], active["private"], frozen, condition)
            error = error32(policy, obs, masked, noise, tau)
        terms = jnp.where(support[..., None] & selector, error, 0.)
        return jnp.mean(jnp.sum(terms, axis=(-2, -1)) / (32 * jnp.sum(support, axis=-1)))

    accumulate = compile_fm_accumulation(jax.jit(jax.value_and_grad(picked_loss)))
    populations = labels[batch.indices]
    channels, steps = np.arange(32)[None, None, :], np.arange(40)[None, :, None]
    groups = {
        "population": {name: (populations == name)[:, None, None] for name in ("old", "new", "recovery")},
        "physical_channels": {"xyz": channels < 3, "rotation": (channels >= 3) & (channels < 6),
                              "gripper": channels == 6, "padding": channels >= 7},
        "horizon": {"p10": steps < 10, "suffix": steps >= 10},
    }
    report = {}
    for group, selectors in groups.items():
        combined = jax.tree.map(jnp.zeros_like, active)
        results = {}
        for label, selection in selectors.items():
            chosen = jax.device_put(np.broadcast_to(selection, actions.shape))
            gradient = jax.tree.map(jnp.zeros_like, active)
            values = []
            for start in range(0, len(batch.indices), micro):
                stop()
                part = slice(start, start + micro)
                obs = jax.tree.map(lambda x: x[part], observation)
                loss, gradient = accumulate(gradient, micro / len(batch.indices), active, learner.frozen,
                              obs, actions[part], noise[part], tau[part], support[part], condition[part], chosen[part])
                values.append(loss)
            results[label] = dict(loss=float(jnp.mean(jnp.stack(values))), gradient_l2=float(optax.global_norm(gradient)),
                                  proposed_parameter_delta_dot_gradient=float(dot(gradient, update)))
            combined = jax.tree.map(jnp.add, combined, gradient)
        error = jax.tree.map(jnp.subtract, combined, original)
        report[group] = dict(terms=results, gradient_reconstruction_l2=float(optax.global_norm(error)))
        del combined, gradient, error
    return report


def run(capsule, output, deadline):
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v3_main_place_learn import endpoint_files
    from v4_long_engine import LongLearner
    from v4_long_probe import snapshot
    from v6_dual_course import reduced_fm
    from v61_source_course import corrected_parent
    from v65_recovery_course import AUX, BATCH, MICRO, PARTS, demographics
    from v65_recovery_worker import partitions
    from v7_full_model_paths import extend_target_io

    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "Expect one qualified H200")
    binding = read(pin(capsule / "BINDING.json", BINDING))
    settled = read(pin(capsule / "TERMINAL.json", TERMINAL))
    require(settled["plan"] == PLAN and settled["formal_updates"] == 1000, "Original pair must be terminal")
    arm = capsule / ARM
    trial = settled["arms"][ARM]
    end = read(pin(arm / "TERMINAL.json", trial["terminal_sha256"]))
    ledger = pin(arm / "UPDATES.jsonl", end["optimizer_ledger_sha256"])
    expected = json.loads(ledger.read_bytes().splitlines()[START])
    require(expected["go2"] == PARTS and expected["step"] == START + 1, "Wrong archived diagnostic step")
    checkpoint = arm / "STEP0250"
    seal = read(pin(arm / "STEP0250_SEALED.json", SEAL))
    require(seal["step"] == seal["bank_cursor"] == START and seal["files"] == endpoint_files(checkpoint),
            "Exact intermediate state changed")
    config, datasets, tokenizer, banks, slots, rates, context = load_inputs(binding)
    p = config["parent_paths"]
    graphs, parent, backbone, _ = corrected_parent(Path(p["custody"]), Path(p["short"]), Path(p["r4"]))
    active, frozen, _, inventory = partitions(parent, backbone)
    require(inventory["total_frozen_sha256"] == seal["frozen_weight_sha256"]
            and inventory["parent_lora_sha256"] == seal["lora_sha256"], "Original frozen backbone and LoRA required")
    _, _, labels = demographics(datasets["target"])
    def stop():
        if (output / "STOP.json").exists() or time.time() >= deadline - 90:
            raise TimeoutError("Bounded V7 loss replay stopped")

    @jax.jit
    def dot(left, right):
        return sum(jnp.vdot(a, b).real for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))

    with closing(LongLearner(graphs=graphs, parameters=extend_target_io(active), frozen=frozen, datasets=datasets,
            tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=datasets["target"].aux,
            cursors=dict.fromkeys(BATCH, 0), rates=rates, rng=jax.random.PRNGKey(MODEL_NOISE_SEED),
            batch_by_domain=BATCH, gradient_join=partial(reduced_fm, batch=BATCH, aux_weight=AUX),
            target_context=context, context_treatment=False)) as learner:
        with ocp.PyTreeCheckpointer() as reader:
            saved = reader.restore(checkpoint.resolve(), item=snapshot(learner))
        require(int(saved["step"]) == START and np.array_equal(saved["rng"], learner.rng)
                and {k: int(v[0].count) for k, v in saved["optimizer"].items()} == seal["adam_counts"],
                "Wrong training counters or RNG")
        for key, values in saved["parameters"].items():
            learner.parameters[key].replace_by_pure_dict(values)
        learner.optimizer, learner.rng, learner.step = saved["optimizer"], saved["rng"], START
        observation = exact_noise_witness(learner)
        joint, parts, metrics = learner.components(START, learner.prepared(START, START + 1), MICRO, 4, stop)
        require({k: list(v) for k, v in observation.items()} == expected["noise"], "Actual replay CUDA noise/tau differs")
        require(all(metrics[k]["visit_sha256"] == expected["components"][k]["visit_sha256"] for k in BATCH)
                and list(metrics["auxiliary"]["slots"]) == expected["components"]["auxiliary"]["slots"],
                "Replay did not visit the recorded TRAIN examples")
        rate = float(rates[START])
        require(rate == expected["lr"], "Scheduled LR differs")
        norms = {name: float(optax.global_norm(part)) for name, part in parts.items()}
        write(output / "INPUT-REPLAY.json", dict(step=START + 1, noise=expected["noise"],
            sample_visits=expected["go2"], rate=rate, component_norms=norms, joint_norm=float(optax.global_norm(joint)),
            archived_joint_norm=expected["global_gradient_norm"], committed_updates=0))
        advanced, shadow, norm = learner.apply(learner.parameters, learner.optimizer, joint, rate)
        delta = jax.tree.map(jnp.subtract, advanced, learner.parameters)
        require(all(int(state[0].count) == START + 1 for state in shadow.values()), "Wrong shadow Adam time")
        del advanced, shadow
        weights = {**{name: size / 128 for name, size in BATCH.items()}, "auxiliary": AUX}
        couples = {}
        for left, right in itertools.combinations(parts, 2):
            value = float(dot(parts[left]["shared"], parts[right]["shared"]))
            if {left, right} == {"target", "auxiliary"}:
                value += float(dot(parts[left]["private"], parts[right]["private"]))
            scale = norms[left] * norms[right]
            couples[f"{left}+{right}"] = dict(dot=value, cosine=value / scale if scale else None,
                                               weighted_cross_term=2 * value * weights[left] * weights[right])
        components = {}
        for name, part in parts.items():
            role = "target" if name == "auxiliary" else name
            components[name] = dict(loss=metrics[name]["loss"], gradient_l2=norms[name],
                expected_gradient_l2=expected["components"][name]["gradient_l2"],
                expected_loss=expected["components"][name]["loss"],
                proposed_parameter_delta_dot_gradient=float(dot(part["shared"], delta["shared"]) +
                                                            dot(part["private"], delta[role])))
        combined_square = sum((norms[k] * weights[k])**2 for k in parts) + sum(r["weighted_cross_term"] for r in couples.values())
        write(output / "BASE-REPLAY.json", dict(step=START + 1, components=components, pairwise=couples,
            joint_norm=norm, logged_joint_norm=expected["global_gradient_norm"],
            gram_reconstructed_norm=math.sqrt(max(0., combined_square)), global_clip_factor=min(1., 1. / norm),
            actual_noise_and_visits_match=True, shadow_optimizer_evaluations=1, committed_updates=0))
        go2 = parts["target"]
        del parts, joint, saved
        terms = region_losses(learner, graphs["target"], go2, delta, labels, MICRO["target"], stop, dot)
        require(learner.step == START and all(int(v[0].count) == START for v in learner.optimizer.values())
                and frozen_fingerprint(learner.frozen) == seal["frozen_weight_sha256"],
                "Diagnostic changed the authoritative training state")
        write(output / "TERMINAL.json", dict(plan=PLAN, step=START + 1, initial_seal=SEAL,
            formal_terminal=TERMINAL, gradient_replay_sha256=sha(output / "BASE-REPLAY.json"),
            go2_fm_breakdowns=terms, shadow_optimizer_evaluations=1, committed_updates=0, native_integrations=0))


def main(capsule, output):
    import jax
    from v4_main250_hopper import allocation
    from v62_source_admission import award

    binding = read(output / "BINDING.json")
    require(binding["plan"] == PLAN and binding["formal_terminal"] == TERMINAL and binding["training_updates"] == 0,
            "The allocation must bind the fixed, uncommitted diagnostic")
    with closing(OwnerLock(output / "OWNER.lock")):
        site = allocation()
        site["project_award"] = award(binding, output, site)
        write(output / "START.json", site)
        jax.config.update("jax_compilation_cache_dir", str(output / "jit-cache"))
        run(capsule, output, site["deadline"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capsule", type=Path)
    p.add_argument("output", type=Path)
    main(**vars(p.parse_args()))
