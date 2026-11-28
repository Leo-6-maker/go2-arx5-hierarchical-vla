"""Train the two registered V7-A arms from independent copies of R4-200."""

import argparse
from collections import Counter
from contextlib import closing
from functools import partial
from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v61_source_course import R4, corrected_parent
from v61_source_worker import parent_gate
from v62_source_admission import award
from v65_recovery_course import AUX, BATCH, MICRO, PARTS
from v65_recovery_worker import partitions
from v7_full_model_paths import extend_target_io
from v7_full_model_qualification import (
    ARMS, PLAN, STEPS, MODEL_NOISE_SEED, bytes_hash, file_ref, input_evidence, load_inputs, projection,
)


SAVE_STEPS = (250, STEPS)
SAVE_RESERVE_S = 900
EXPECTED_VISITS = dict(old=24000, new=23000, recovery=1000, robodojo=8000, libero=8000, auxiliary=2000)


def allocated_bytes(path):
    return int(subprocess.check_output(["du", "-sB1", "--", str(path)], text=True).split()[0])


def capacity(capsule, root, bound):
    root = root.resolve(strict=True)
    require(capsule.resolve().is_relative_to(root) and root.stat().st_uid == os.getuid()
            and root.stat().st_dev == capsule.stat().st_dev and bound > 0, "Wrong personal quota root")
    # Two checkpoints per arm plus one reserve to close at a finite boundary.
    planned = (len(ARMS) * len(SAVE_STEPS) + 1) * bound
    free = min(shutil.disk_usage(root).free, shutil.disk_usage(capsule).free)
    require(free > planned, "Available personal quota is below the V7 pair checkpoint reserve")
    with tempfile.TemporaryFile(dir=capsule) as trial:
        os.posix_fallocate(trial.fileno(), 0, 64 * 1024**2)
        os.fsync(trial.fileno())
    return dict(free_bytes=free, planned_bytes=planned, checkpoint_bytes=bound, save_steps=SAVE_STEPS)


def exact_noise_witness(learner):
    """Hash the arrays handed to the running FM and auxiliary gradient calls."""
    observed = {}

    def record_draws(name, draw):
        def batch(rng, cursor):
            noise, tau = draw(rng, cursor)
            observed[name] = (bytes_hash(noise), bytes_hash(tau))
            return noise, tau
        return batch

    learner.draws = {role: record_draws(role, draw) for role, draw in learner.draws.items()}
    auxiliary = learner.aux_add

    def aux_draw(accumulator, factor, active, frozen, observation, noise, *inputs):
        observed["auxiliary"] = (bytes_hash(noise),)
        return auxiliary(accumulator, factor, active, frozen, observation, noise, *inputs)

    learner.aux_add = aux_draw
    return observed


def save(learner, run, state, space):
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v3_main_place_learn import endpoint_files
    from v4_long_probe import snapshot

    require(shutil.disk_usage(run).free > space, "Formal checkpoint write reserve exhausted")
    require(frozen_fingerprint(learner.frozen) == state["total_frozen_sha256"], "Frozen R4 state changed")
    endpoint = run / f"STEP{learner.step:04d}"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(endpoint.resolve(), snapshot(learner))
    adam = {name: int(value[0].count) for name, value in learner.optimizer.items()}
    require(adam == dict.fromkeys(("shared", *BATCH), learner.step), "Wrong Adam or sampler cursor")
    size = allocated_bytes(endpoint)
    receipt = dict(step=learner.step, files=endpoint_files(endpoint), allocated_bytes=size, adam_counts=adam,
                   frozen_weight_sha256=state["total_frozen_sha256"], lora_sha256=state["parent_lora_sha256"],
                   projection_sha256=bytes_hash(projection(learner.parameters)), bank_cursor=learner.step)
    write(endpoint.with_name(endpoint.name + "_SEALED.json"), receipt)
    require(0 < size <= space, "Formal state exceeded the measured checkpoint bound")
    return receipt


def run_arm(learner, run, arm, groups, previous, qualification_input, state, space, guard, binding_sha):
    import jax

    run.mkdir(mode=0o700)
    require(learner.step == 0 and all(int(v[0].count) == 0 for v in learner.optimizer.values())
            and not np.asarray(projection(learner.parameters)).any(), "Start with fresh parent and optimizer")
    initial = input_evidence(learner, 0)
    require(initial == qualification_input, "Formal sample, backend noise, or CUDA input differs from qualification")
    contract = run / "RUN_CONTRACT.json"
    write(contract, dict(plan=PLAN, arm=arm, steps=STEPS, new_adam_from_r4=True, seed=MODEL_NOISE_SEED,
                         parent=R4, binding_sha256=binding_sha, initial=initial,
                         effective_batch=BATCH, micro=MICRO, aux_coefficient=AUX))
    noise = exact_noise_witness(learner)
    totals, current, latest = Counter(), 0, None
    path = run / "UPDATES.jsonl"
    status = "FORMAL_INTERRUPTED"

    @jax.jit
    def finite(parameters):
        import jax.numpy as jnp
        return jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(parameters)]).all()

    try:
        with path.open("x", buffering=1) as journal:
            while current < STEPS:
                noise.clear()
                row = learner.update(STEPS, MICRO, 4, guard)
                roles = row["components"]
                part = dict(Counter(groups[learner.banks["target"][current]]))
                expected = dict(BATCH, auxiliary=4)
                require(row["step"] == current + 1 and row["lr"] == learner.rates[current]
                        and part == PARTS and {k: v["visits"] for k, v in roles.items()} == expected
                        and set(noise) == set(expected) and bool(finite(learner.parameters))
                        and all(roles[name]["visit_sha256"] == bytes_hash(bank[current])
                                for name, bank in learner.banks.items())
                        and roles["auxiliary"]["slots"] == learner.auxiliary_slots[current].tolist(),
                        "Formal effect differs from the original fixed batch or optimizer contract")
                # The two arms use identical recorded FM draws, time and aux arrays.
                if previous is not None:
                    control = json.loads(next(previous))
                    require(control["step"] == row["step"] and control["lr"] == row["lr"]
                            and control["noise"] == {k: list(v) for k,v in noise.items()},
                            "The formal CUDA draws differ from the earlier matched arm")
                row.update(go2=part, noise=dict(noise))
                journal.write(json.dumps(row, allow_nan=False)+"\n")
                current = learner.step
                totals.update(part)
                totals.update({role: expected[role] for role in ("robodojo", "libero", "auxiliary")})
                if current in SAVE_STEPS:
                    latest = save(learner, run, state, space)
        require(dict(totals) == EXPECTED_VISITS
                and np.isfinite(np.asarray(projection(learner.parameters))).all()
                and bool(np.asarray(projection(learner.parameters)).any()) == (arm == ARMS[1]),
                "Final fixed exposure or condition parameter state changed")
        status = "FORMAL_V7_ARM_500_COMPLETE"
    except TimeoutError:
        status = "SAVED_AT_FINITE_BOUNDARY"
    finally:
        if current == learner.step and current > 0 and (latest is None or latest["step"] != current):
            latest = save(learner, run, state, space)
        write(run/"TERMINAL.json", dict(status=status, arm=arm, step=current,
              updates_not_in_ledger=learner.step-current, formal_updates=current, engineering_updates=0,
              visits=dict(totals), checkpoint=latest, binding_sha256=binding_sha,
              optimizer_ledger_sha256=sha(path) if path.exists() else None, contract_sha256=sha(contract)))


def train(capsule, binding, site):
    import jax
    from v4_long_engine import LongLearner
    from v65_recovery_course import demographics
    from v6_dual_course import reduced_fm

    config,datasets,tokenizer,banks,slots,schedule,context = load_inputs(binding)
    p = config["parent_paths"]
    graphs, parent, base, ancestry = corrected_parent(Path(p["custody"]), Path(p["short"]), Path(p["r4"]))
    active, frozen, _, audit = partitions(parent,base)
    qualification_input = read(file_ref(binding["qualification_input"]))
    expected = read(file_ref(binding["qualification_parameters"]))
    for key in ("parent_lora_sha256", "total_frozen_sha256", "counts"):
        require(audit[key] == expected[key], "Formal R4 does not match the qualified parent")
    write(capsule/"PARAMETERS.json",dict(audit,ancestry=ancestry,added_projection=[1,1024]))
    _, _, groups = demographics(datasets["target"])
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def guard():
        if stopping or (capsule/"STOP.json").exists() or time.time() >= site["deadline"]-SAVE_RESERVE_S:
            raise TimeoutError("Original STOP or finite checkpoint deadline")

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    done = {}
    binding_sha = sha(capsule/"BINDING.json")
    for arm in ARMS:
        guard()
        learner = LongLearner(graphs=graphs, parameters=extend_target_io(active), frozen=frozen, datasets=datasets,
                   tokenizer=tokenizer,banks=banks,auxiliary_slots=slots,auxiliary=datasets["target"].aux,
                   cursors=dict.fromkeys(BATCH,0),rates=schedule,rng=jax.random.PRNGKey(MODEL_NOISE_SEED),
                   batch_by_domain=BATCH,gradient_join=partial(reduced_fm,batch=BATCH,aux_weight=AUX),
                   target_context=context,context_treatment=arm == ARMS[1])
        with closing(learner):
            path = capsule/arm
            args = (qualification_input,audit,site["write_capacity"]["checkpoint_bytes"],guard,binding_sha)
            if arm == ARMS[1]:
                with (capsule/ARMS[0]/"UPDATES.jsonl").open() as previous:
                    run_arm(learner,path,arm,groups,previous,*args)
            else:
                run_arm(learner,path,arm,groups,None,*args)
        terminal = path/"TERMINAL.json"
        result = read(terminal)
        done[arm] = dict(formal_updates=result["formal_updates"], terminal_sha256=sha(terminal))
        if result["status"] != "FORMAL_V7_ARM_500_COMPLETE":
            break
    total = sum(row["formal_updates"] for row in done.values())
    write(capsule/"TERMINAL.json",dict(status="FORMAL_V7_PAIR_COMPLETE" if total == STEPS*2 else "PARTIAL_V7_PAIR",
          plan=PLAN,formal_updates=total,engineering_updates=0,arms=done,source500_released=False))


def main(capsule):
    import jax
    from v4_main250_hopper import allocation

    binding = read(capsule/"BINDING.json")
    terminal = read(file_ref(binding["qualification_terminal"]))
    require(binding["plan"] == PLAN and binding["r4_receipt"] == R4
            and binding["formal_steps_per_arm"] == STEPS and binding["arms"] == list(ARMS)
            and terminal["status"] == "GPU_QUALIFIED" and terminal["r4"] == R4 and terminal["formal_updates"] == 0
            and terminal["actual_engineering_updates"] == 4, "GPU qualification is required")
    parent_gate(binding["parent"])
    with closing(OwnerLock(capsule/"OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind,"Formal V7-A requires one H200")
        site["project_award"] = award(binding,capsule,site)
        site["write_capacity"] = capacity(capsule,Path(binding["quota_root"]),binding["checkpoint_bytes"])
        write(capsule/"START.json",site)
        jax.config.update("jax_compilation_cache_dir",str(capsule/"jit-cache"))
        train(capsule,binding,site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule",type=Path)
    main(parser.parse_args().capsule)
