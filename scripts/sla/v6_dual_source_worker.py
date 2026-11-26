"""Read, qualify or run the first bounded V6 dual-source capsule."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from v3_main_place_learn import endpoint_files
from v4_long_checkpoint import load_parent
from v4_long_probe import recover, snapshot
from v4_main250_hopper import allocation
from v6_dual_course import PARENT, SEED, SOURCE_BATCH, SOURCE_MICRO, bind


STEPS = {"probe": 6, "formal": 1000}
GIB = 1024**3


def inputs_for(custody, short_run, old_bank, source, manifest):
    make, readers, banks, slots, auxiliary, settings = bind(custody, short_run, old_bank, source, manifest)
    settings.update(bank_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in banks.items()},
                    aux_slots_sha256=hashlib.sha256(slots.tobytes()).hexdigest(),
                    aux_per_update=4, original_parent=PARENT)
    return make, readers, banks, slots, auxiliary, settings


def preflight(custody, short_run, old_bank, source, manifest):
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
    from v4_long_inputs import inputs

    _, datasets, banks, slots, auxiliary, settings = inputs_for(custody, short_run, old_bank, source, manifest)
    parent = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    _, tokenizer, *_ = inputs(custody, short_run, old_bank, parent)
    shapes = []
    for step in (0, 1):
        shapes.append({d: list(prepare_host_batch(datasets[d], tokenizer, d,
                        range(step * size, (step + 1) * size), banks[d][step])["actions"].shape)
                       for d, size in SOURCE_BATCH.items()})
        preceding, executed = auxiliary(banks["target"][step, slots[step]])
        require(preceding.shape == (4,) and executed.shape == (4, 40) and np.isfinite(executed).all(),
                "Native target auxiliary sample or preceding width is missing")
    sources = datasets["robodojo"]
    require(np.all(banks["robodojo"][0] < sources.base_length)
            and np.all(banks["robodojo"][1] >= sources.base_length), "New source has no real scheduled visit")
    return dict(settings=settings, actual_host_batch_shapes=shapes,
                read_new_source_windows=int(len(banks["robodojo"][1])), optimizer_updates=0)


def run(mode, capsule, custody, short_run, old_bank, source, manifest):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule / "OWNER.lock")):
        declared = read(capsule / "BINDING.json")
        require(declared["mode"] == mode and declared["manifest"] == manifest
                and declared["parent"] == PARENT and declared["steps"] == STEPS[mode],
                "The original mode/manifest/parent binding is required")
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "Wrong actual JAX GPU")
        reserve = (70 if mode == "formal" else 15) * GIB
        require(shutil.disk_usage(capsule).free >= reserve, "Checkpoint write and recovery reserve unavailable")
        result = capsule / "run"
        result.mkdir(mode=0o700)
        write(result / "START.json", site)
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        make, datasets, banks, _, _, settings = inputs_for(custody, short_run, old_bank, source, manifest)
        if mode == "formal":
            proof = Path(declared["probe_terminal"]["path"])
            require(sha(proof) == declared["probe_terminal"]["sha256"], "Qualification receipt changed")
            good = read(proof)
            require(good["status"] == "PROBE_PASS" and good["settings"] == settings
                    and good["updates"] == STEPS["probe"] and good["restore"]["prediction_equal"] is True,
                    "The exact complete source qualification is required")
        selected = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
        graphs, parameters, frozen, model = load_parent(custody, selected, PARENT, short_run / "incoming/reference")
        write(result / "RUN_CONTRACT.json", dict(settings=settings, parent=model, mode=mode))
        rng = jax.random.PRNGKey(SEED)
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def guard():
            if stopped or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 900:
                raise TimeoutError("Reserved checkpoint time or STOP reached")

        def factory():
            return make(graphs, dict(parameters), frozen, rng)

        with closing(factory()) as learner:
            updates = STEPS[mode]
            status = "INTERRUPTED"
            report = {}
            last_save = time.monotonic()
            first_new = None
            new_visits = 0

            def seal():
                point = result / f"STEP{learner.step:04d}"
                if not point.exists():
                    with ocp.PyTreeCheckpointer() as writer:
                        writer.save(point.resolve(), snapshot(learner))
                return dict(step=learner.step, files=endpoint_files(point), path=str(point))

            try:
                with (result / "UPDATES.jsonl").open("x", buffering=1) as ledger:
                    while learner.step < updates:
                        guard()
                        row = learner.update(updates, SOURCE_MICRO, 4, guard)
                        visits = {d: c["visits"] for d, c in row["components"].items()}
                        require(visits == {**SOURCE_BATCH, "auxiliary": 4}, "Effective visit count changed")
                        slots = banks["robodojo"][learner.step - 1]
                        count = int(np.count_nonzero(slots >= datasets["robodojo"].base_length))
                        require(count == (96 if learner.step % 2 == 0 else 0), "Source round-robin was not consumed")
                        row["v6_new_source_visits"] = count
                        if count and first_new is None:
                            first_new = row["step"]
                        new_visits += count
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        if mode == "formal" and (learner.step % 100 == 0 or learner.step == 250
                                                or time.monotonic() - last_save >= 1800):
                            write(result / "RECOVERABLE.json", seal())
                            last_save = time.monotonic()
                if mode == "probe":
                    report["restore"] = recover(learner, factory, result / "STEP0006",
                                                graphs["target"], (SOURCE_MICRO, 4), guard)
                    require(report["restore"]["prediction_equal"] is True, "Exact restart prediction changed")
                    stats = jax.local_devices()[0].memory_stats()
                    require(stats["peak_bytes_in_use"] < .9 * stats["bytes_limit"], "Insufficient device margin")
                    report["peak_allocated_bytes"] = stats["peak_bytes_in_use"]
                status = "PROBE_PASS" if mode == "probe" else "FORMAL_1000"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                write(result / "TERMINAL.json", dict(**report, status=status, settings=settings,
                    updates=learner.step, first_new_source_optimizer_step=first_new, new_source_visits=new_visits,
                    checkpoint=seal(), contract_sha256=sha(result / "RUN_CONTRACT.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "probe", "formal"))
    for name in ("capsule", "custody", "short_run", "old_bank", "source"):
        parser.add_argument(name, type=Path)
    parser.add_argument("manifest")
    a = parser.parse_args()
    if a.mode == "preflight":
        print(json.dumps(preflight(a.custody, a.short_run, a.old_bank, a.source, a.manifest)))
    else:
        run(**vars(a))
