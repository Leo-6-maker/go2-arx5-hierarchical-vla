"""Run one bound 200-update MAIN3.8K retention contrast on Hopper."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
from pathlib import Path
import json
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import MAIN, endpoint_files
from v4_long_checkpoint import load_parent
from v4_long_engine import LongLearner
from v4_long_inputs import inputs
from v4_long_probe import SEED, snapshot
from v4_main250_hopper import allocation


FACTORS = {"R1_LOW_LR": (.25, False), "R2_PHASE_MATCHED_REPLAY": (1., False),
           "R3_TARGET_GRADIENT_ONLY": (1., True)}
ENDPOINTS = (100, 200)
MICRO = dict(target=64, robodojo=32, libero=32)


def activate_replay(replay, datasets, banks, slots, original_archive):
    from v6_retention_replay import ReplayDataset, bind, read_replay

    evidence = read(pin(replay["candidates"]["path"], replay["candidates"]["sha256"]))
    fixed = read(pin(replay["roster"]["path"], replay["roster"]["sha256"]))
    host = read(pin(replay["host"]["path"], replay["host"]["sha256"]))
    require(fixed["replay_candidate_sha256"] == replay["candidates"]["sha256"]
            and host["fixed_roster_sha256"] == replay["roster"]["sha256"]
            and host["new_real_replay_windows_read"] == 16 and host["optimizer_updates"] == 0,
            "The local matched replay must be the one actually read on Corelab")
    target = datasets["target"]
    additional = read_replay(Path(replay["legacy_site"]), original_archive, Path(replay["closed_raw"]), target,
                             evidence, Path(replay["closed_labels"]))
    datasets["target"] = ReplayDataset(target, additional)
    banks["target"] = bind(banks["target"], slots, datasets["target"], fixed)
    digest = hashlib.sha256(banks["target"].tobytes()).hexdigest()
    require(digest == host["visits_bank_sha256"], "Replay visits differ from the Corelab consumer")
    return digest


def run(capsule, custody, short_run, bank_root):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule / "OWNER.lock")):
        binding = read(capsule / "BINDING.json")
        role = binding["role"]
        lr_scale, target_only = FACTORS[role]
        with_replay = role == "R2_PHASE_MATCHED_REPLAY"
        core = read(pin(binding["core_release"]["path"], binding["core_release"]["sha256"]))
        require(core["role"] == role and core["parent"] == MAIN and core["corelab_diagnosis_complete"]
                and binding["updates"] == ENDPOINTS[-1] and core["active_fixed_h40_aux"] == 4,
                "Expected the actual Corelab-authorized fixed retention contrast")
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "Expected one authorized H200")
        result = capsule / "run"
        result.mkdir(mode=0o700)
        write(result / "START.json", site)
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        parent = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
        graphs, base, frozen, model = load_parent(custody, parent, MAIN, short_run / "incoming/reference")
        datasets, tokenizer, banks, slots, auxiliary, cursors, rates = inputs(custody, short_run, bank_root, parent)
        rates = lr_scale * rates
        original_banks = {key: hashlib.sha256(arr.tobytes()).hexdigest() for key, arr in banks.items()}
        require(original_banks == core["original_visit_bank_sha256"] and
                hashlib.sha256(slots.tobytes()).hexdigest() == core["original_aux_ordinal_sha256"],
                "The matched exposure or actual auxiliary sample bank changed")
        config = dict(role=role, parent=model, original_visit_banks=original_banks, micro=MICRO,
                    exact_lr_factor=lr_scale, peak_lr=float(rates.max()), endpoints=ENDPOINTS,
                    four_cpu_workers=True, prefetch_batches=2, target_matches_original=True,
                    active_auxiliary=4, source_updates_enabled=not target_only)
        if with_replay:
            config["target_matches_original"] = False
            config["target_replay_bank_sha256"] = activate_replay(core["replay"], datasets, banks, slots,
                custody / "foundation-layout-v2/train-ready/TRAIN-069")
        write(result / "RUN_CONTRACT.json", config)
        stop = False

        def stop_signal(*_):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, stop_signal)
        signal.signal(signal.SIGTERM, stop_signal)

        def guard():
            if stop or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 900:
                raise TimeoutError("The safe checkpoint write interval was reached")

        with closing(LongLearner(graphs=graphs, parameters=base, frozen=frozen, datasets=datasets,
                tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary, cursors=cursors,
                rates=rates, rng=jax.random.PRNGKey(SEED), target_control=target_only)) as learner:
            summaries = []

            def seal():
                path = result / f"STEP{learner.step:04d}"
                if not path.exists():
                    with ocp.PyTreeCheckpointer() as writer:
                        writer.save(path.resolve(), snapshot(learner))
                return dict(step=learner.step, checkpoint_path=str(path), files=endpoint_files(path))

            status = "TECHNICAL_INTERRUPTION"
            try:
                with (result / "UPDATES.jsonl").open("x", buffering=1) as ledger:
                    while learner.step < ENDPOINTS[-1]:
                        guard()
                        row = learner.update(ENDPOINTS[-1], MICRO, 4, guard)
                        observed = {d: value["visits"] for d, value in row["components"].items()}
                        expected = {"target": 64, "auxiliary": 4} if target_only else {**MICRO, "auxiliary": 4}
                        require(observed == expected, "Real updates lost fixed target/control exposure")
                        if with_replay:
                            target = datasets["target"]
                            identities = Counter(target.rows[i][0]["id"] for i in banks["target"][learner.step - 1]
                                                 if i >= len(target.original))
                            require(sum(identities.values()) == 8, "Actual replay visits differ from the bound batch")
                            row["replay_visits"] = dict(identities)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        if learner.step in ENDPOINTS:
                            sealed = seal()
                            summaries.append(sealed)
                            write(result / f"STEP{learner.step:04d}_SEALED.json", sealed)
                if target_only:
                    for domain in ("robodojo", "libero"):
                        require(all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(base[domain]),
                                     jax.tree.leaves(learner.parameters[domain]), strict=True)),
                                "Inactive private source weights changed")
                status = "FORMAL_RETENTION_200"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                write(result / "TERMINAL.json", dict(status=status, checkpoints=summaries, current=seal(),
                      updates=learner.step, role=role, contract_sha256=sha(result / "RUN_CONTRACT.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capsule", "custody", "short_run", "bank_root"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
