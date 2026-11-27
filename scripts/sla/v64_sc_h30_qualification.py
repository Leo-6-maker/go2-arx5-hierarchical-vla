"""Exercise the real SC-VLA H30 backward path and a process-separated restore."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v3_main_place_learn import endpoint_files
from v61_source_course import R4, corrected_parent
from v62_source_admission import award, write_capacity
from v63_adapt_course import PEAK
from v6_dual_course import four_domain_adamw, initialize_single


PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
HOST_READER = "214326b03cd6206f8ffb205ad982240d790f3581af65f070f02b03c1321d1d4d"
DATA_MEMBERS = "f7be73c4b77809a9bf0be7305c580268cc28441cf466bec28fd4afb4871c3d98"
A250_TERMINAL = "95e8cbe39d279136e86b40f1eb646fbc429dabc4d2e35833926d146a3edf2dc6"
BATCH, MICRO, UPDATES = 32, 16, 3
TASKS = ("peg", "ball", "stack")
SEED, CURSOR = 527_202611, 29_000_000


def training_indices(reader):
    """Choose eight fixed anchors from each of four different TRAIN episodes."""
    sample, offset = [], 0
    for episode, row in list(reader.episodes.items())[:4]:
        size = row["windows"]
        require(row["split"] == "TRAIN" and size >= 8, "The fixed episode cannot supply eight TRAIN anchors")
        selected = (offset + np.linspace(0, size - 1, 8, dtype=np.int64)).tolist()
        require(all(reader.rows[i][0] == episode for i in selected), "Source index ordering changed")
        sample.extend(selected)
        offset += size
    require(len(sample) == len(set(sample)) == BATCH, "Incomplete independent H30 TRAIN batch")
    return sample


def host_batches(data, reference):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from v62_sc_source import CAMERAS, HORIZON, NATIVE_HZ, SCHEMA, SOURCE_TASKS
    from v62_sc_training_inputs import prepare
    from v6_ville_reader import SingleArmReader

    receipt = read(pin(data / "HOPPER-CPU-TRAIN-READBACK.json", HOST_READER))
    require(receipt["received_source_train_episodes"] == 51 and receipt["actual_verified_batches"] == 3
            and receipt["verified_file_manifest_sha256"] == DATA_MEMBERS
            and sha(data / "DATA-SHA256SUMS") == DATA_MEMBERS, "The portable SC-VLA receipt changed")
    checkpoint = read(reference / "CHECKPOINT.json")
    tokenizer = Tokenizer(str(pin(reference / "TOKENIZER.model", checkpoint["tokenizer_sha256"])))
    for step, (key, task, expected) in enumerate(zip(TASKS, SOURCE_TASKS, receipt["rows"], strict=True)):
        pool = data / "data" / key
        reader = SingleArmReader(pool, expected["manifest_sha256"], pool / "raw", split="TRAIN",
                                 resize=resize_with_pad, schema=SCHEMA, native_hz=NATIVE_HZ,
                                 horizon=HORIZON, cameras=CAMERAS, cache_episodes=2)
        require(reader.manifest["task"] == expected["task"] == task and len(reader) == expected["windows_in_population"],
                "The fixed H30 training task changed")
        indices = training_indices(reader)
        cursor = CURSOR + step * BATCH
        batch = prepare(reader, tokenizer, range(cursor, cursor+BATCH), indices)
        support = batch["action_support"]
        require(batch["actions"].shape == support.shape == (BATCH, HORIZON, 32)
                and support[..., :7].all() and not support[..., 7:].any(), "The H30 support mask changed")
        yield batch, dict(task=key, cursor=cursor, rows=[list(reader.rows[i]) for i in indices],
                          input_sha256=hashlib.sha256(batch["actions"].tobytes()+support.tobytes()).hexdigest())


def training_state(parameters, optimizer, rng, step):
    return dict(parameters={name: state.to_pure_dict() for name, state in parameters.items()},
                optimizer=optimizer, rng=rng, step=step)


def state_hashes(state):
    import jax

    hashes = {}
    for name, tree in state.items():
        digest = hashlib.sha256()
        leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
        for path, leaf in leaves:
            parts = [str(getattr(key, "key", getattr(key, "name", getattr(key, "idx", key)))) for key in path]
            values = np.asarray(leaf)
            digest.update(f"{'/'.join(parts)}:{values.shape},{values.dtype}\n".encode())
            digest.update(values.tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def update_path(binding, capsule, run, site):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter import batch_v2, scale20k_training
    from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v62_sc_training_inputs import ROLE, abstract_model, flow_loss, frozen_draws

    paths = read(pin(binding["paths"]["path"], binding["paths"]["sha256"]))
    custody, short, r4 = (Path(paths[k]) for k in ("custody", "short", "r4"))
    data = Path(binding["data"])
    reference = short / "incoming/reference"
    graph, _ = nnx.split(abstract_model())
    _, base, frozen, ancestry = corrected_parent(custody, short, r4)
    frozen_before = frozen_fingerprint(frozen)
    parameters = {"shared": base["shared"], ROLE: initialize_single(base["target"])}
    del base
    init, apply = four_domain_adamw(**OPTIMIZER, domains=(ROLE,))
    optimizer = init(parameters)
    rng = jax.random.PRNGKey(SEED)
    draw = frozen_draws(BATCH)
    loss = flow_loss(graph)
    calculate = scale20k_training.compile_fm_accumulation(jax.jit(jax.value_and_grad(loss)))
    evaluation = jax.jit(loss)

    def guard():
        require(time.time() < site["deadline"] - 900 and not (capsule / "STOP.json").exists(),
                "Finite H30 boundary or operator STOP")
        require(shutil.disk_usage(capsule).free > 2 * site["write_capacity"]["checkpoint_bytes"],
                "The sealed checkpoint write reserve is unavailable")

    def device_inputs(batch, visit):
        obs, actions = batch_v2.device_batch(batch)
        noise, tau = draw(rng, visit["cursor"])
        return obs, actions, noise, tau, jax.device_put(batch["action_support"])

    def fixed_loss(active, inputs):
        result = float(evaluation(active, frozen, *jax.tree.map(lambda x: x[:MICRO], inputs)))
        require(np.isfinite(result), "The fixed H30 forward was nonfinite")
        return result

    guard()
    visited = []
    evaluation_inputs = None
    for step, (batch, visit) in enumerate(host_batches(data, reference), 1):
        require(step <= UPDATES, "The qualification has exactly three authorized updates")
        guard()
        began = time.monotonic()
        inputs = device_inputs(batch, visit)
        if evaluation_inputs is None:
            evaluation_inputs = inputs
        active = {"shared": parameters["shared"], "private": parameters[ROLE]}
        total = jax.tree.map(jnp.zeros_like, active)
        losses = []
        for start in range(0, BATCH, MICRO):
            guard()
            part = jax.tree.map(lambda x: x[start:start+MICRO], inputs)
            value, total = calculate(total, MICRO / BATCH, active, frozen, *part)
            losses.append(float(value))
        # Apply the SC-VLA share of the registered 128-FM objective.
        gradients = {"shared": total["shared"], ROLE: total["private"]}
        gradients = jax.tree.map(lambda x: x * (BATCH/128), gradients)
        norms = {name: float(optax.global_norm(group)) for name, group in gradients.items()}
        require(all(np.isfinite(x) for x in losses) and all(np.isfinite(x) and x > 0 for x in norms.values()),
                "Nonfinite or zero native H30 gradients")
        guard()
        rate = PEAK * step / 100
        pending = dict(visit, step=step, rate=rate, losses=losses, gradient_l2=norms, visits=BATCH, micro=MICRO)
        write(run / f"{step:02d}.ENTERED.json", pending)
        parameters, optimizer, norm = apply(parameters, optimizer, gradients, rate)
        jax.block_until_ready((parameters, optimizer))
        returned = dict(pending, global_gradient_norm=norm, seconds=time.monotonic()-began)
        write(run / f"{step:02d}.RETURNED.json", returned)
        visited.append(returned)

    require(len(visited) == UPDATES and ancestry["initial_r4_receipt_sha256"] == R4, "Wrong H30 step or R4 origin")
    guard()
    state = training_state(parameters, optimizer, rng, UPDATES)
    endpoint = run / "STEP0003"
    began = time.monotonic()
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(endpoint.resolve(), state)
    save_s = time.monotonic()-began
    memory = jax.local_devices()[0].memory_stats()
    require(memory["peak_bytes_in_use"] < .9*memory["bytes_limit"], "H30 requires measured GPU memory headroom")
    require(frozen_fingerprint(frozen) == frozen_before, "The frozen foundation changed")
    evaluation_loss = fixed_loss({"shared": parameters["shared"], "private": parameters[ROLE]}, evaluation_inputs)
    write(run / "SAVED.json", dict(plan=PLAN, updates=len(visited), engineering_only=True,
          formal_updates=0, physics_ticks=0, training_visits=BATCH*len(visited), frozen_sha256=frozen_before,
          checkpoint_sha256=state_hashes(state), files=endpoint_files(endpoint), save_s=save_s,
          fixed_h30_loss=evaluation_loss, vram_peak_bytes=memory["peak_bytes_in_use"],
          vram_limit_bytes=memory["bytes_limit"], final_cursor=CURSOR + len(visited)*BATCH))


def cold_readback(binding, run):
    import jax
    import orbax.checkpoint as ocp

    begin = time.monotonic()
    proof = read(run / "SAVED.json")
    endpoint = run / "STEP0003"
    require(proof["files"] == endpoint_files(endpoint) and proof["updates"] == UPDATES, "H30 archive changed")
    with ocp.PyTreeCheckpointer() as saver:
        meta = saver.metadata(endpoint).tree
        args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)
        state = saver.restore(endpoint, args=ocp.args.PyTreeRestore(item=meta, restore_args=args, transforms={}))
    require(jax.default_backend() == "cpu" and set(state["parameters"]) == {"shared", "sc_vla"}
            and set(state["optimizer"]) == set(state["parameters"]) and int(state["step"]) == UPDATES
            and np.array_equal(state["rng"], np.asarray(jax.random.PRNGKey(SEED))), "Wrong H30 recovery state")
    counters = {name: [int(x) for x in jax.tree.leaves(optim)
        if np.asarray(x).ndim == 0 and np.issubdtype(np.asarray(x).dtype, np.integer)]
        for name, optim in state["optimizer"].items()}
    require(all(c == [UPDATES] for c in counters.values()) and state_hashes(state) == proof["checkpoint_sha256"],
            "Recovered optimizer, parameters or RNG do not match the complete H30 archive")
    write(run / "TERMINAL.json", dict(proof, status="H30_ISOLATED_QUALIFICATION_PASS", cold_restore=True,
          adam_counters=counters, restore_s=time.monotonic()-begin, final_cursor=CURSOR+UPDATES*BATCH,
          binding_sha256=binding, mixed_source_qualified=False))


def main(mode, capsule):
    if mode == "readback":
        cold_readback(sha(capsule / "BINDING.json"), capsule / "run")
        return

    import jax
    from v4_main250_hopper import allocation

    binding = read(capsule / "BINDING.json")
    terminal = read(pin(binding["A250"]["path"], A250_TERMINAL))
    require(binding["plan"] == PLAN and terminal["updates"] == 250
            and binding["engineering_used"] + UPDATES <= 24 and binding["engineering_limit"] == UPDATES,
            "The real first A boundary and available engineering allocation are required")
    with closing(OwnerLock(capsule / "OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "One PBS H200 is required")
        paths = read(pin(binding["paths"]["path"], binding["paths"]["sha256"]))
        site["award"] = award(binding, capsule, site)
        site["write_capacity"] = write_capacity(capsule, Path(paths["r4"]), "probe", 0, UPDATES, binding["budget_seconds"])
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        run = capsule / "run"
        run.mkdir(mode=0o700)
        write(run / "START.json", site)
        update_path(binding, capsule, run, site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "readback"))
    parser.add_argument("capsule", type=Path)
    main(**vars(parser.parse_args()))
