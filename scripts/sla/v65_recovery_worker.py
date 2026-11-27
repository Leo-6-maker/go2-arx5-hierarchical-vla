"""Qualify the V6.5 tree and run the uninterrupted R4-to-1K curriculum."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v3_main_place_learn import endpoint_files
from v4_long_probe import fixed_prediction, snapshot
from v4_main250_hopper import allocation
from v61_source_course import R4, corrected_parent
from v61_source_worker import parent_gate
from v62_source_admission import award, write_capacity
from v63_adapt_worker import PARENT_RELEASE, SOURCE_RELEASE
from v65_recovery_course import BATCH, END, MICRO, PARTS, PLAN, SEED, bind, learner


PROBE = 2
RESERVE_SECONDS = 900


def partitions(parameters, frozen):
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from go2arx5_vla.sla_adapter.scale20k_contract import parameter_partition

    all_target = nnx.State.merge(parameters["shared"], parameters["target"], frozen)
    complete = flatten_dict(all_target.to_pure_dict())
    scope = {k: set(v) for k, v in parameter_partition(complete).items()}
    expert, lora = parameters["shared"].split(lambda path, _: tuple(path) in scope["action_expert"], ...)
    frozen_after = nnx.State.merge(frozen, lora)
    active = dict(parameters, shared=expert)
    available = {name: flatten_dict(tree.to_pure_dict()) for name, tree in active.items()}
    require(set(available) == {"shared", *BATCH} and set(available["shared"]) == scope["action_expert"]
            and set(flatten_dict(lora.to_pure_dict())) == scope["vlm_lora"]
            and set(flatten_dict(frozen.to_pure_dict())) == scope["frozen"]
            and all(set(available[role]) == scope["domain_template"] for role in BATCH)
            and all(str(x.dtype) == "float32" for group in available.values() for x in group.values()),
            "Actual parent freeze/train allowlist does not match V6.5")
    for role in BATCH:
        before = nnx.State.merge(parameters["shared"], parameters[role], frozen)
        after = nnx.State.merge(active["shared"], active[role], frozen_after)
        b, a = (flatten_dict(tree.to_pure_dict()) for tree in (before, after))
        require(a.keys() == b.keys() and all(a[k] is b[k] for k in b),
                "Step 0 moved, changed dtype, or replaced an actual parent value")
    inventory = {}
    for name, leaves in {**available, "parent_lora": flatten_dict(lora.to_pure_dict()),
                         "frozen_base": flatten_dict(frozen.to_pure_dict())}.items():
        inventory[name] = [dict(path="/".join(map(str, key)), shape=list(x.shape), dtype=str(x.dtype), size=x.size)
                           for key, x in sorted(leaves.items())]
    return active, frozen_after, lora, dict(inventory=inventory,
                   counts={key: sum(x["size"] for x in rows) for key,rows in inventory.items()},
                   parent_lora_sha256=frozen_fingerprint(lora), total_frozen_sha256=frozen_fingerprint(frozen_after))


def isolated_recovery(learner_, graph):
    import jax
    import optax
    from go2arx5_vla.sla_adapter.batch_v2 import device_batch
    from v4_long_engine import masked_target_loss

    batch = learner_.prepare(0, "target")
    obs, action = device_batch(jax.tree.map(lambda a: a[94:], batch.host))
    noise, tau = learner_.draws["target"](learner_.rng, batch.cursor)
    active = {"shared": learner_.parameters["shared"], "private": learner_.parameters["target"]}
    loss, gradient = masked_target_loss(graph)(active, learner_.frozen, obs, action, noise[94:], tau[94:], batch.support[94:])
    norms = {key: float(optax.global_norm(tree)) for key,tree in gradient.items()}
    require(np.isfinite(float(loss)) and all(np.isfinite(v) and v > 0 for v in norms.values()),
            "The two real Recovery windows contributed no usable gradient")
    return dict(indices=batch.indices[94:].tolist(), actual_mask=batch.support[94:].sum(axis=1).tolist(),
                loss=float(loss), gradient_norm=norms, optimizer_visits=0)


def recover(learner_, path):
    import jax
    import orbax.checkpoint as ocp

    template = snapshot(learner_)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(path.resolve(), item=template)
    require(jax.tree.structure(template) == jax.tree.structure(restored)
            and all(np.array_equal(a, b) for a,b in zip(jax.tree.leaves(template), jax.tree.leaves(restored), strict=True)),
            "Complete parameters, Adam, RNG or cursor did not cold-restore")
    for domain, arrays in restored["parameters"].items():
        fresh = jax.tree.map(lambda x: x, learner_.parameters[domain])
        fresh.replace_by_pure_dict(arrays)
        learner_.parameters[domain] = fresh
    learner_.optimizer, learner_.rng, learner_.step = restored["optimizer"], restored["rng"], int(restored["step"])


def run(capsule):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint

    binding_file = capsule / "BINDING.json"
    binding = read(binding_file)
    require(binding["plan"] == PLAN and binding["r4_receipt"] == R4 and binding["formal_end"] == END
            and binding["engineering_updates"] == PROBE and PROBE <= binding["engineering_remaining"]
            and binding["formal_prior_updates"] + END <= 8000
            and binding["parent"]["sha256"] == PARENT_RELEASE, "Unregistered V6.5 execution")
    parent_gate(binding["parent"])
    hold = binding["source_hold"]
    require(hold["sha256"] == SOURCE_RELEASE and not read(pin(hold["path"], hold["sha256"]))["released"],
            "Retain the actual old SOURCE500 hold")
    config = read(pin(binding["data"]["path"], binding["data"]["sha256"]))
    paths = config["parent_paths"]
    r4 = Path(paths["r4"])
    with closing(OwnerLock(capsule / "OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "One allocated H200 is required")
        site["project_award"] = award(binding, capsule, site)
        site["write_capacity"] = write_capacity(capsule, r4, "1000", 0, END, binding["budget_seconds"])
        write(capsule / "START.json", site)
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        data = bind(config, Path(binding["bank"]["path"]).parent, binding["bank"]["sha256"])
        graphs, inherited, base, ancestry = corrected_parent(Path(paths["custody"]), Path(paths["short"]), r4)
        active, frozen, lora, audit = partitions(inherited, base)
        write(capsule / "PARAMETERS.json", audit)
        clock = jax.random.PRNGKey(SEED)
        _, _, banks, slots, _, (episodes, phases, groups) = data
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        def guard():
            if stopped or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - RESERVE_SECONDS:
                raise TimeoutError("Original stop or checkpoint reserve reached")

        def make():
            return learner(data, graphs, jax.tree.map(lambda x: x, active), frozen, clock)

        @jax.jit
        def finite(parameters):
            return jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(parameters)]).all()

        def update(current, end):
            guard()
            row = current.update(end, MICRO, 4, guard)
            index = current.step - 1
            target = banks["target"][index]
            parts = dict(Counter(groups[target].tolist()))
            require(parts == PARTS and bool(finite(current.parameters))
                    and all(row["components"][role]["visits"] == BATCH[role]
                            and row["components"][role]["visit_sha256"] == hashlib.sha256(bank[index].tobytes()).hexdigest()
                            for role, bank in banks.items())
                    and row["components"]["auxiliary"]["slots"] == slots[index].tolist()
                    and all(np.isfinite(c["gradient_l2"]) and c["gradient_l2"] > 0 for c in row["components"].values()),
                    "Actual optimizer update diverged from the sealed recipe")
            n_old = int(np.count_nonzero(banks["robodojo"][index] < data[0]["robodojo"].base_length))
            row.update(go2=parts, robodojo_old=n_old, robodojo_expanded=16-n_old,
                       go2_phase=dict(Counter(phases[target].tolist())), recovery_ids=episodes[target[-2:]].tolist())
            return row

        def save(current, directory):
            require(shutil.disk_usage(capsule).free > 2*site["write_capacity"]["checkpoint_bytes"],
                    "Complete write/overlap reserve fell below the inherited bound")
            path = directory / f"STEP{current.step:04d}"
            with ocp.PyTreeCheckpointer() as writer:
                writer.save(path.resolve(), snapshot(current))
            require(frozen_fingerprint(lora) == audit["parent_lora_sha256"], "Frozen parent LoRA changed")
            counts = {key: int(value[0].count) for key, value in current.optimizer.items()}
            require(set(counts) == {"shared", *BATCH} and set(counts.values()) == {current.step},
                    "An exact Adam cursor was not saved")
            proof = dict(step=current.step, files=endpoint_files(path), adam_counts=counts, bank_cursor=current.step,
                         frozen_weight_sha256=audit["total_frozen_sha256"], lora_sha256=audit["parent_lora_sha256"])
            write(directory / f"{path.name}_SEALED.json", proof)
            return path, proof

        qual = capsule / "qualification"
        qual.mkdir(mode=0o700)
        with closing(make()) as current:
            old = SimpleNamespace(parameters=inherited, frozen=base, rng=clock, prepare=current.prepare)
            original = fixed_prediction(old, graphs["target"])
            partitioned = fixed_prediction(current, graphs["target"])
            require(np.array_equal(original, partitioned), "Step 0 fixed prediction changed after freezing parent LoRA")
            write(qual / "STEP0.json", dict(fixed_H40_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                  exact_equal=True, recovery_gradient=isolated_recovery(current, graphs["target"])))
            with (qual / "UPDATES.jsonl").open("x", buffering=1) as history:
                while current.step < PROBE:
                    history.write(json.dumps(update(current, PROBE), allow_nan=False)+"\n")
            guard()
            checkpoint, proof = save(current, qual)
            expected = fixed_prediction(current, graphs["target"])
            with closing(make()) as cold:
                # Use the completed qualification tree only as the restore schema, preserving fresh runtime ownership.
                cold.optimizer, cold.rng, cold.step = current.optimizer, current.rng, current.step
                cold.parameters = jax.tree.map(lambda x: x, current.parameters)
                recover(cold, checkpoint)
                restored = fixed_prediction(cold, graphs["target"])
            memory = jax.local_devices()[0].memory_stats()
            require(np.array_equal(expected, restored) and frozen_fingerprint(frozen) == audit["total_frozen_sha256"]
                    and 0 < memory["peak_bytes_in_use"] < .9 * memory["bytes_limit"],
                    "Actual qualification did not prove cold restore, freeze and GPU headroom")
            write(qual / "TERMINAL.json", dict(status="QUALIFICATION_PASS", updates=PROBE, checkpoint=proof,
                  recovered_prediction_sha256=hashlib.sha256(restored.tobytes()).hexdigest(), memory=memory))

        formal = capsule / "run"
        formal.mkdir(mode=0o700)
        write(formal / "RUN_CONTRACT.json", dict(plan=PLAN, seed=SEED, start=0, end=END,
              ancestry=ancestry, qualification_sha256=sha(qual / "TERMINAL.json"), binding_sha256=sha(binding_file),
              actual_training_scope_sha256=sha(capsule / "PARAMETERS.json"), new_adam_from_r4=True))
        with closing(make()) as current:
            require(current.step == 0 and frozen_fingerprint(lora) == audit["parent_lora_sha256"], "Start from R4 step 0")
            status, previous, latest = "TECHNICAL_INTERRUPTION", time.monotonic(), None
            totals = Counter()
            try:
                with (formal / "UPDATES.jsonl").open("x", buffering=1) as history:
                    while current.step < END:
                        row = update(current, END)
                        history.write(json.dumps(row, allow_nan=False)+"\n")
                        totals.update(row["go2"])
                        totals.update(robodojo_old=row["robodojo_old"], robodojo_expanded=row["robodojo_expanded"],
                                      libero=row["components"]["libero"]["visits"], aux=row["components"]["auxiliary"]["visits"])
                        if current.step % 100 == 0 or current.step == 250 or time.monotonic()-previous >= 1800:
                            _, latest = save(current, formal)
                            previous = time.monotonic()
                status = "FORMAL_V65_1K_COMPLETE"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                if latest is None or latest["step"] != current.step:
                    _, latest = save(current, formal)
                write(formal / "TERMINAL.json", dict(status=status, formal_updates=current.step, engineering_updates=PROBE,
                      checkpoint=latest, optimizer_visits=dict(totals), contract_sha256=sha(formal / "RUN_CONTRACT.json")))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule", type=Path)
    run(cli.parse_args().capsule)
