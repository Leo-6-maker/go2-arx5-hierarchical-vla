"""Run the fixed 1K mixed-source arm with a measured, recoverable batch profile."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import endpoint_files
from v4_long_checkpoint import load_parent
from v4_long_engine import LongLearner
from v4_long_inputs import inputs
from v4_long_probe import LADDER, MEMORY_RESERVE_FRACTION, SEED, UPDATES as PROBE_UPDATES, snapshot
from v4_main250_hopper import allocation


UPDATES = 1000


def select(probe, receipt_pin):
    receipt = read(pin(probe / "run/ACCEPTED.json", receipt_pin))
    rows = read(pin(probe / "run/PROFILES.json", receipt["profile_sha256"]))
    eligible = [r for r in rows if r.get("restore") and not r.get("failure")
                and r["engineering_updates"] == PROBE_UPDATES
                and r["headroom_fraction"] >= MEMORY_RESERVE_FRACTION]
    require(eligible, "No recoverable measured profile")
    best = min(eligible, key=lambda r: (r["p50_s"], r["p95_s"]))
    widths = {**best["fm_micro"], "auxiliary": best["aux_micro"]}
    require(receipt["thresholds"] == "DIAGNOSTIC_ONLY" and receipt["workers"] == 4
            and receipt["prefetched_batches"] == 2
            and receipt["selected"] == {"fm": best["fm_micro"], "auxiliary": best["aux_micro"]}
            and set(widths) == set(LADDER) and all(widths[d] in LADDER[d] for d in widths)
            and all(np.isfinite(best[k]) and best[k] > 0 for k in ("p50_s", "p95_s")),
            "Expected the fastest stable measured parallel profile")
    return best


def train(capsule, custody, short_run, bank_root, probe):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule / "OWNER.lock")):
        charter = read(capsule / "incoming/FORMAL_BINDING.json")
        require(charter["updates"] == UPDATES and charter["target_mix"] == "OLD_DATA_FALLBACK"
                and charter["formal_new_skill_teacher_ready"] is False,
                "This capsule has only the fixed 1K old-target fallback plan")
        for key, path in (("target_bank_sha256", bank_root / "prepared/ACTUAL.json"),
                          ("auxiliary_sha256", bank_root / "native_aux-01/MANIFEST.json"),
                          ("short_terminal_sha256", short_run / "run/TRAIN_TERMINAL.json")):
            require(sha(path) == charter[key], "Formal input changed: " + key)
        profile = select(probe, charter["performance_sha256"])
        result = capsule / "run"
        result.mkdir(mode=0o700)
        site = allocation()
        write(result / "START.json", site)
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "H200 runtime identity changed")
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        parent = Path(charter["parent_path"])
        require(parent.resolve(strict=True).is_relative_to(custody.resolve(strict=True)),
                "Parent must come from the qualified archive")
        v3 = short_run / "run" if charter.get("parent_v3_overlay") else None
        graphs, parameters, frozen, model = load_parent(
            custody, parent, charter["parent_sha256"], short_run / "incoming/reference", v3_endpoint=v3)
        datasets, tokenizer, banks, slots, auxiliary, cursors, rates = inputs(
            custody, short_run, bank_root, parent)
        rng = jax.random.PRNGKey(SEED)
        keys = dict(batch_by_domain={"target":64,"robodojo":32,"libero":32}, aux_per_update=4,
                    workers=4, prefetch_batches=2, fm_micro=profile["fm_micro"], aux_micro=profile["aux_micro"],
                    phase_cursors=cursors, random_seed=SEED, exact_bank_hashes={
                        k:hashlib.sha256(values.tobytes()).hexdigest() for k,values in banks.items()},
                    aux_bank_sha256=hashlib.sha256(slots.tobytes()).hexdigest(),
                    lr_bank_sha256=hashlib.sha256(rates.tobytes()).hexdigest())
        write(result / "RUN_CONTRACT.json", dict(charter=charter, model=model, settings=keys))
        stop = False

        def signal_stop(*_):
            nonlocal stop
            stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal_stop)

        def guard():
            if stop or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 900:
                raise TimeoutError("Stop requested or final checkpoint time reserve reached")

        with closing(LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=datasets,
                                tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary,
                                cursors=cursors, rates=rates, rng=rng)) as learner:
            last_save = time.monotonic()

            def seal():
                path = result / f"STEP{learner.step:04d}"
                started = time.monotonic()
                if not path.exists():
                    with ocp.PyTreeCheckpointer() as writer:
                        writer.save(path.resolve(), snapshot(learner))
                return dict(step=learner.step, endpoint=str(path), files=endpoint_files(path),
                            seal_seconds=time.monotonic()-started)

            status = "INTERRUPTED"
            try:
                with (result / "UPDATES.jsonl").open("x", buffering=1) as ledger:
                    while learner.step < UPDATES:
                        guard()
                        row = learner.update(UPDATES, keys["fm_micro"], keys["aux_micro"], guard)
                        require({k:v["visits"] for k,v in row["components"].items()}
                                == {"target":64,"robodojo":32,"libero":32,"auxiliary":4},
                                "A formal update lost its original effective count")
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        if (learner.step % 100 == 0 or learner.step == 250
                                or time.monotonic() - last_save >= 1800):
                            write(result / "RECOVERABLE.json", seal())
                            last_save = time.monotonic()
                status = "FORMAL_1000"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                write(result / "TRAIN_TERMINAL.json", dict(seal(), status=status,
                      formal_endpoint=learner.step == UPDATES and status == "FORMAL_1000",
                      completed_updates=learner.step,
                      contract_sha256=sha(result / "RUN_CONTRACT.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capsule", "custody", "short_run", "bank_root", "probe"):
        parser.add_argument(name, type=Path)
    train(**vars(parser.parse_args()))
