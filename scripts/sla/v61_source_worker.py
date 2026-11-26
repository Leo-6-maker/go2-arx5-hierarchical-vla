"""Run one admitted V6.1 SOURCE qualification or formal interval on Hopper."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import endpoint_files
from v4_long_probe import UPDATES as PROBE_STEPS, recover, snapshot
from v4_main250_hopper import allocation
from v61_source_bank import BATCH, ENDPOINTS, PLAN, SEED
from v61_source_course import BANK_RECEIPT, MICRO, R4, bind, corrected_parent


GIB = 1024**3
MODES = ("probe", *(str(end) for end in ENDPOINTS))


def parent_gate(reference):
    proof = read(pin(reference["path"], reference["sha256"]))
    require(proof["plan"] == PLAN and proof["r4_receipt_sha256"] == R4
            and proof["previous_N02_success"] == 2 and proof["place"]["valid"] == 4
            and proof["place"]["correct_grasp"] >= 3 and proof["independent_HOLD"]["paired_valid"] == 4
            and 1 <= proof["independent_HOLD"]["r4_full_success"] <= 4
            and proof["independent_HOLD"]["r4_full_success"] >= proof["independent_HOLD"]["parent_full_success"]
            and not proof["new_repeated_pregrasp_stop"] and proof["released"],
            "V6.1 requires the actual closed native parent decision")


def validated_boundary(reference, ledger_end):
    proof = read(pin(reference["path"], reference["sha256"]))
    require(proof["plan"] == PLAN and proof["source_step"] == ledger_end
            and proof["bank_receipt_sha256"] == BANK_RECEIPT
            and proof["source_learned_or_underconverged"]
            and proof["n02_valid"] == proof["n02_success"] == 2
            and proof["place_valid"] == proof["place_grasp"] == 4
            and not proof["new_repeated_pregrasp_stop"], "The prior COURSE segment lacks Corelab release")


def resume(learner, previous, proof, contract, start):
    import orbax.checkpoint as ocp

    terminal = read(pin(previous / "TERMINAL.json", proof["training_terminal_sha256"]))
    point = previous / f"STEP{start:04d}"
    require(terminal["status"] in ("FORMAL_SOURCE_BOUNDARY", "SAVED_AT_FINITE_BOUNDARY")
            and terminal["updates"] == start
            and terminal["contract_sha256"] == sha(previous / "RUN_CONTRACT.json")
            and terminal["checkpoint"]["files"] == endpoint_files(point)
            and terminal["course"] == contract, "Continuation requires the exact admitted complete SOURCE endpoint")
    with ocp.PyTreeCheckpointer() as checkpointer:
        state = checkpointer.restore(point, item=snapshot(learner))
    require(int(state["step"]) == start and np.array_equal(state["rng"], learner.rng)
            and set(state["parameters"]) == set(learner.parameters), "The resumed course identity changed")
    for name, value in state["parameters"].items():
        learner.parameters[name].replace_by_pure_dict(value)
    learner.optimizer, learner.rng, learner.step = state["optimizer"], state["rng"], int(state["step"])


def run(mode, capsule, custody, short, prepared, corrected, extension, single, bank, r4):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import OwnerLock, write

    declared_file = capsule / "BINDING.json"
    declared = read(declared_file)
    goal = PROBE_STEPS if mode == "probe" else int(mode)
    base = 0 if mode in ("probe", "250") else ENDPOINTS[ENDPOINTS.index(goal) - 1]
    start = declared["start"]
    require(declared["plan"] == PLAN and declared["stage"] == mode
            and declared["end"] == goal and declared["bank_receipt_sha256"] == BANK_RECEIPT
            and declared["r4_receipt_sha256"] == R4 and type(start) is int and base <= start < goal
            and (mode != "probe" or start == 0), "The bound SOURCE interval changed")
    parent_gate(declared["core_parent_release"])
    course = dict(plan=PLAN, bank_receipt_sha256=BANK_RECEIPT, initial_r4_receipt_sha256=R4,
                  effective_batch=BATCH, auxiliary=4, micro=MICRO, source_rng_seed=SEED)
    if mode != "probe":
        ref = declared["probe_terminal"]
        probed = read(pin(ref["path"], ref["sha256"]))
        require(probed["status"] == "QUALIFICATION_PASS" and probed["updates"] == PROBE_STEPS
                and probed["course"] == course and probed["restore"]["prediction_equal"],
                "Run actual source qualification before the first formal update")
    if base:
        validated_boundary(declared["core_segment_release"], base)
    with closing(OwnerLock(capsule / "OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "Expected one admitted H200")
        require(shutil.disk_usage(capsule).free >= (15 if mode == "probe" else 70) * GIB,
                "Original checkpoint save reserve is unavailable")
        result = capsule / "run"
        result.mkdir(mode=0o700)
        write(result / "START.json", site)
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        create, datasets, banks, _, _, admitted = bind(custody, short, prepared, corrected, extension, single, bank)
        graphs, parameters, frozen, ancestry = corrected_parent(custody, short, r4)
        clock = jax.random.PRNGKey(SEED)
        contract = dict(course=course, start=start, end=goal, mode=mode, ancestry=ancestry,
                        input_sha256=admitted["manifest_sha256"], binding_sha256=sha(declared_file))
        write(result / "RUN_CONTRACT.json", contract)

        def factory():
            copied = jax.tree.map(lambda x: x, parameters)
            return create(graphs, copied, frozen, clock)

        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        def guard():
            if stopped or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 900:
                raise TimeoutError("Original checkpoint save interval or registered STOP")

        def seal(learner):
            path = result / f"STEP{learner.step:04d}"
            if not path.exists():
                with ocp.PyTreeCheckpointer() as saver:
                    saver.save(path.resolve(), snapshot(learner))
            return dict(step=learner.step, path=str(path), files=endpoint_files(path))

        with closing(factory()) as learner:
            status = "TECHNICAL_INTERRUPTION"
            last_save = time.monotonic()
            checkpoint = None
            metrics = {}
            consumed = dict(dual_old=0, dual_new=0, single=0)
            if start:
                ref = declared["core_segment_release"] if start == base else declared["partial_segment_readback"]
                resume(learner, Path(declared["previous"]), read(pin(ref["path"], ref["sha256"])), course, start)
            try:
                with (result / "UPDATES.jsonl").open("x", buffering=1) as ledger:
                    while learner.step < goal:
                        guard()
                        row = learner.update(goal, MICRO, 4, guard)
                        idx = learner.step - 1
                        visits = {name: data["visits"] for name, data in row["components"].items()}
                        require(visits == {**BATCH, "auxiliary": 4}
                                and all(hashlib.sha256(bank_rows[idx].tobytes()).hexdigest()
                                        == row["components"][name]["visit_sha256"] for name, bank_rows in banks.items()),
                                "Actual learner visits differ from the sealed bank")
                        boundary = datasets["robodojo"].base_length
                        new = int(np.count_nonzero(banks["robodojo"][idx] >= boundary))
                        actual = dict(dual_old=visits["robodojo"] - new, dual_new=new, single=visits["single"])
                        require(actual == dict(dual_old=40, dual_new=40, single=16), "Actual source population changed")
                        for key, value in actual.items():
                            consumed[key] += value
                        row["source_visits_from_actual_indices"] = actual
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        if mode != "probe" and (learner.step % 100 == 0 or learner.step == goal
                                                or time.monotonic() - last_save >= 1800):
                            checkpoint = seal(learner)
                            write(result / f"STEP{learner.step:04d}_SEALED.json", checkpoint)
                            last_save = time.monotonic()
                if mode == "probe":
                    metrics["restore"] = recover(learner, factory, result / "STEP0006", graphs["target"],
                                                 (MICRO, 4), guard)
                    memory = jax.local_devices()[0].memory_stats()
                    require(metrics["restore"]["prediction_equal"]
                            and memory["peak_bytes_in_use"] < .9 * memory["bytes_limit"],
                            "Qualification requires exact restart and actual H200 reserve")
                    metrics["peak_allocated_bytes"] = memory["peak_bytes_in_use"]
                status = "QUALIFICATION_PASS" if mode == "probe" else "FORMAL_SOURCE_BOUNDARY"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                if checkpoint is None or checkpoint["step"] != learner.step:
                    checkpoint = seal(learner)
                write(result / "TERMINAL.json", dict(status=status, course=course, updates=learner.step,
                      interval_updates=learner.step - start, optimized_source_visits=consumed,
                      checkpoint=checkpoint, contract_sha256=sha(result / "RUN_CONTRACT.json"), **metrics))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=MODES)
    for field in ("capsule", "custody", "short", "prepared", "corrected", "extension", "single", "bank", "r4"):
        cli.add_argument(field, type=Path)
    run(**vars(cli.parse_args()))
