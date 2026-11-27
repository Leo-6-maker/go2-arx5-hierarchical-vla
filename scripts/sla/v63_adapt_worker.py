"""Run one admitted A74 shape qualification followed by a fresh formal A250."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v3_main_place_learn import endpoint_files
from v4_long_probe import UPDATES as PROBE_STEPS, recover, snapshot
from v4_main250_hopper import allocation
from v61_source_course import R4, corrected_parent
from v61_source_worker import parent_gate
from v62_source_admission import award, write_capacity
from v63_adapt_course import BATCH, MICRO, SEED, TARGET, bind
from v63_adapt_data import PLAN


PARENT_RELEASE = "0896270957c478197e586a71f2d2d3a55464a5bb7436cf3d390fa14804926433"
SOURCE_RELEASE = "5f8000336c3e6871b9947fe2aae6ca72f46bf8edea4781ca87ad58b9300ecc6a"
FIRST = 250


def validate_boundary(binding):
    require(binding["plan"] == PLAN and binding["r4_receipt"] == R4 and binding["end"] == FIRST
            and binding["parent"]["sha256"] == PARENT_RELEASE
            and binding["source_hold"]["sha256"] == SOURCE_RELEASE
            and binding["engineering_updates"] == PROBE_STEPS
            and binding["engineering_upper_left"] >= PROBE_STEPS and binding["formal_prior_updates"] >= 250
            and binding["formal_prior_updates"] + FIRST <= 8000,
            "Use the exact available A250 / engineering scope")
    parent_gate(binding["parent"])
    hold = binding["source_hold"]
    source = read(pin(hold["path"],hold["sha256"]))
    require(not source["released"] and source["source_step"] == 250, "Original SOURCE500 remains held")


def assess(learner, banks, group, datasets, step, guard, phases):
    row = learner.update(step, MICRO, 4, guard)
    index = learner.step - 1
    visits = {k: v["visits"] for k,v in row["components"].items()}
    require(visits == {**BATCH, "auxiliary":4}
            and all(hashlib.sha256(value[index].tobytes()).hexdigest() == row["components"][role]["visit_sha256"]
                    for role,value in banks.items())
            and np.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0.
            and all(np.isfinite(v["gradient_l2"]) and v["gradient_l2"] > 0. for v in row["components"].values()),
            "Actual gradients or visits do not match the registered A250 input")
    tgt = banks["target"][index]
    n_old = int(np.count_nonzero(banks["robodojo"][index] < datasets["robodojo"].base_length))
    population = dict(Counter(group[tgt].tolist()))
    require(n_old == 16 and population == {k: v[0] for k,v in TARGET.items()},
            "Exact old and new optimization exposure changed")
    aux_slots = row["components"]["auxiliary"]["slots"]
    row["actual_go2_population"] = population
    row["actual_go2_phase"] = dict(Counter(phases[i] for i in tgt))
    row["actual_go2_aux_population"] = dict(Counter(group[tgt[aux_slots]].tolist()))
    row["actual_robodojo_old"] = n_old
    row["actual_robodojo_expanded"] = 32-n_old
    return row


def run(capsule):
    import jax
    import orbax.checkpoint as ocp

    seal = capsule / "BINDING.json"
    binding = read(seal)
    validate_boundary(binding)
    paths = read(pin(binding["inputs"]["path"],binding["inputs"]["sha256"]))
    extra, bank, r4 = (Path(paths[k]) for k in ("extra","adapt_bank","r4"))
    with closing(OwnerLock(capsule / "OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "One admitted H200 is required")
        site["project_award"] = award(binding,capsule,site)
        site["write_capacity"] = write_capacity(capsule,r4,"250",0,FIRST,binding["budget_seconds"])
        write(capsule / "START.json",site)
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        create, datasets, _, banks, slots, record, group = bind(paths,extra,bank,binding["bank_sha256"])
        graphs, parameters, frozen, ancestry = corrected_parent(Path(paths["custody"]),Path(paths["short"]),r4)
        ancestry.pop("source_rng_seed",None)
        clock = jax.random.PRNGKey(SEED)
        phases = [ep["expert_stage_audit_only"][index] for ep,index in datasets["target"].rows]

        def factory():
            return create(graphs,jax.tree.map(lambda x:x,parameters),frozen,clock)

        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)

        def guard():
            free = shutil.disk_usage(capsule).free
            if stopped or (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 900:
                raise TimeoutError("Original checkpoint reserve or registered STOP")
            if free < 2 * site["write_capacity"]["checkpoint_bytes"]:
                raise TimeoutError("Actual terminal write reserve is unavailable")

        def checkpoint(learner, path):
            with ocp.PyTreeCheckpointer() as saver:
                saver.save(path.resolve(),snapshot(learner))
            proof = dict(step=learner.step,path=str(path),files=endpoint_files(path))
            write(path.with_name(path.name+"_SEALED.json"),proof)
            return proof

        course = dict(plan=PLAN, r4_parent=R4, course_seed=SEED, bank_sha256=binding["bank_sha256"],
                      data_sha256=record["added_manifest_sha256"], batches=BATCH,micro=MICRO,auxiliary=4)
        probe = capsule / "qualification"
        probe.mkdir(mode=0o700)
        with closing(factory()) as learner:
            status = "QUALIFICATION_INTERRUPTED"
            restore, peak = None, None
            try:
                with (probe / "UPDATES.jsonl").open("x",buffering=1) as history:
                    while learner.step < PROBE_STEPS:
                        row = assess(learner,banks,group,datasets,PROBE_STEPS,guard,phases)
                        history.write(json.dumps(row,allow_nan=False)+"\n")
                guard()
                restore = recover(learner,factory,probe / f"STEP{PROBE_STEPS:04d}",graphs["target"],(MICRO,4),guard)
                memory = jax.local_devices()[0].memory_stats()
                peak = memory["peak_bytes_in_use"]
                require(restore["prediction_equal"] and peak < .9*memory["bytes_limit"],
                        "Actual cold restore and safe accelerator capacity required")
                recent = [json.loads(line)["seconds"] for line in (probe / "UPDATES.jsonl").read_text().splitlines()][-3:]
                require(time.time()+900+1.15*FIRST*np.median(recent) < site["deadline"],
                        "Measured cold qualification leaves insufficient time for A250")
                status = "QUALIFICATION_PASS"
            finally:
                path = probe / f"STEP{learner.step:04d}"
                if not path.exists():
                    checkpoint(learner,path)
                write(probe / "TERMINAL.json",dict(status=status,updates=learner.step,optimizer="NEW_ADAM_FOR_QUAL_ONLY",
                      restore=restore,peak_allocated_bytes=peak,course=course,files=endpoint_files(path)))
        formal = capsule / "run"
        formal.mkdir(mode=0o700)
        proof = probe / "TERMINAL.json"
        write(formal / "RUN_CONTRACT.json",dict(course=course,start=0,end=FIRST,ancestry=ancestry,
              probe_receipt=sha(proof),binding_sha256=sha(seal), optimizer="NEW_ADAM_FROM_R4",
              rng="INDEPENDENT_ADAPT_SEED", lr_schedule="WARM100_COSINE3000"))
        with closing(factory()) as learner:
            require(learner.step == 0, "A250 must start from the untouched R4 parent")
            status = "TECHNICAL_INTERRUPTION"
            visits, go2_groups, go2_phases = Counter(), Counter(), Counter()
            point, last = None,time.monotonic()
            try:
                with (formal / "UPDATES.jsonl").open("x",buffering=1) as ledger:
                    while learner.step < FIRST:
                        row = assess(learner,banks,group,datasets,FIRST,guard,phases)
                        visits.update({k:v["visits"] for k,v in row["components"].items()})
                        visits.update(robodojo_old=row["actual_robodojo_old"],
                                      robodojo_expanded=row["actual_robodojo_expanded"])
                        go2_groups.update(row["actual_go2_population"])
                        go2_phases.update(row["actual_go2_phase"])
                        ledger.write(json.dumps(row,allow_nan=False)+"\n")
                        if learner.step % 100 == 0 or learner.step == FIRST or time.monotonic()-last >= 1800:
                            point = checkpoint(learner,formal / f"STEP{learner.step:04d}")
                            last = time.monotonic()
                status = "FORMAL_ADAPT_A250_BOUNDARY"
            except TimeoutError:
                status = "SAVED_AT_FINITE_BOUNDARY"
            finally:
                if point is None or point["step"] != learner.step:
                    point = checkpoint(learner,formal / f"STEP{learner.step:04d}")
                write(formal / "TERMINAL.json",dict(status=status,updates=learner.step,course=course,
                      engineering_updates=PROBE_STEPS,optimizer_visits=dict(visits),go2_population=dict(go2_groups),
                      go2_phase=dict(go2_phases),checkpoint=point,bank_cursor=learner.step,
                      contract_sha256=sha(formal / "RUN_CONTRACT.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule",type=Path)
    run(parser.parse_args().capsule)
