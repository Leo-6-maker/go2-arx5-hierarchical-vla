"""Finite IO-only candidate, gated by real full-model qualification and independent restore."""

import argparse
from collections import Counter
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v7_frozen_expert_optimizer import SCOPE, make_frozen_expert_adamw
from v7_frozen_expert_checkpoint import save, validate_optimizer_clock, validate_parameters


REVIEW = "8e6ffa192d3e6c0adce7e2f89ca97e8ea20f461a"
QUALIFICATION = "V7_FROZEN_SHARED_EXPERT_FULL_MODEL_QUALIFICATION_V1"
CONTROL = "V7_B6_SHARED_L2SP_MATCHED_V1"
END = 250
SAVE_STEPS = (100,250)


def checked(reference):
    return pin(reference["path"],reference["sha256"])


def admission(binding):
    require(binding["scope"] == SCOPE and binding["reviewed_plan"] == REVIEW
            and binding["end_step"] == binding["formal_updates"] == END
            and binding["engineering_updates"] == 0, "Require the single finite registered candidate")
    qualification = read(checked(binding["gpu_qualification"]))
    restored = read(checked(binding["independent_core_restore"]))
    require(qualification.get("scope") == restored.get("scope") == QUALIFICATION
            and qualification["status"] == "FULL_MODEL_GPU_QUALIFIED"
            and qualification["actual_engineering_model_updates"] >= 2
            and qualification["finite_real_component_gradients"]
            and qualification["initial_three_paths_unchanged"]
            and qualification["shared_parameters_and_adam_exactly_frozen"]
            and qualification["original_private_clip_and_adam_oracle_equal"]
            and restored["gpu_qualification_sha256"] == binding["gpu_qualification"]["sha256"]
            and restored["independent_complete_restore_equal"]
            and restored["next_input_and_noise_equal"],
            "CPU mechanics or shape traces are not full-model update/restore qualification")
    source = Path(binding["source_pair"])
    original = read(pin(source/"BINDING.json",binding["source_pair_binding_sha256"]))
    require(qualification["data_bank_sha256"] == original["bank"]["sha256"]
            and binding["unchanged_model_paths"] and binding["reuse_control_250"],
            "The matched bank or original model paths changed")
    control_root = Path(binding["control_250"]["path"])
    control = read(pin(control_root/"TERMINAL.json",binding["control_250"]["terminal_sha256"]))
    control_binding = read(control_root/"BINDING.json")
    require(control["scope"] == CONTROL and control["status"] == "PAIR_COMPLETE" and control["end_step"] == END,
            "Reference control requires the original complete matched 250 endpoint")
    require(control_binding["source_pair_binding_sha256"] == binding["source_pair_binding_sha256"],
            "The reused dense control belongs to a different course")
    zero = read(pin(control_root/"LAMBDA_ZERO/TERMINAL.json",control["arms"]["LAMBDA_ZERO"]["terminal_sha256"]))
    require(zero["coefficient"] == 0. and zero["updates_not_in_ledger"] == 0,
            "The reused control must be the unregularized complete arm")
    return original,qualification


def train(capsule,binding,allocation):
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v4_long_engine import LongLearner
    from v6_dual_course import reduced_fm
    from v61_source_course import corrected_parent
    from v65_recovery_course import AUX,BATCH,MICRO,PARTS,demographics
    from v65_recovery_worker import partitions
    from v7_full_model_qualification import load_inputs,MODEL_NOISE_SEED
    from v7_full_model_paths import extend_target_io
    from v7_formal_worker import input_evidence,exact_noise_witness

    original,qualification = admission(binding)
    config,datasets,tokenizer,banks,slots,rates,context = load_inputs(original)
    p = config["parent_paths"]
    graphs,parent,backbone,_ = corrected_parent(Path(p["custody"]),Path(p["short"]),Path(p["r4"]))
    active,frozen,_,inventory = partitions(parent,backbone)
    shared_hash = frozen_fingerprint(active["shared"])
    require(shared_hash == qualification["initial_shared_expert_sha256"]
            and inventory["total_frozen_sha256"] == qualification["original_frozen_values_sha256"],
            "The full R4 parent or inherited frozen values changed")
    _,_,groups = demographics(datasets["target"])
    source = Path(binding["source_pair"])
    control = read(pin(source/"CONTROL_NEUTRAL/TERMINAL.json",binding["source_control_500_terminal_sha256"]))
    old_ledger = [json.loads(line) for line in pin(source/"CONTROL_NEUTRAL/UPDATES.jsonl",control["optimizer_ledger_sha256"]).read_text().splitlines()]
    require(len(old_ledger) == 500 and [row["step"] for row in old_ledger] == list(range(1,501)),
            "The fixed original control schedule/noise ledger is incomplete")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def guard():
        if stopping or (capsule/"STOP.json").exists() or time.time() >= allocation["deadline"]-900:
            raise TimeoutError("Frozen-Expert candidate reached the original finite save boundary")

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    learner = LongLearner(graphs=graphs,parameters=extend_target_io(active),frozen=frozen,datasets=datasets,
        tokenizer=tokenizer,banks=banks,auxiliary_slots=slots,auxiliary=datasets["target"].aux,
        cursors=dict.fromkeys(BATCH,0),rates=rates,rng=jax.random.PRNGKey(MODEL_NOISE_SEED),batch_by_domain=BATCH,
        gradient_join=partial(reduced_fm,batch=BATCH,aux_weight=AUX),target_context=context,context_treatment=False,
        optimizer_factory=make_frozen_expert_adamw)
    run = capsule/"FROZEN_SHARED_EXPERT"
    run.mkdir(mode=0o700)
    with closing(learner):
        require(input_evidence(learner,0) == read(checked(original["qualification_input"])),
                "Frozen-Expert candidate changed initial samples, actual noise or support")
        validate_optimizer_clock(learner.optimizer,0)
        validate_parameters(learner,inventory,shared_hash)
        witness = exact_noise_witness(learner)
        write(run/"RUN_CONTRACT.json",dict(scope=SCOPE,binding_sha256=sha(capsule/"BINDING.json"),end_step=END,
            learning_rate_schedule="ORIGINAL_500_PREFIX_NO_REWARMUP",original_effective_batch=BATCH,micro=MICRO,
            original_aux_coefficient=AUX,frozen_shared_expert_sha256=shared_hash,shared_adam_count=0,
            data_bank_sha256=original["bank"]["sha256"],neutral_previous_command=True,reuse_dense_control=True))
        recorded = 0
        saved = None
        status = "INTERRUPTED"
        try:
            with (run/"UPDATES.jsonl").open("x",buffering=1) as stream:
                for i in range(END):
                    guard()
                    witness.clear()
                    row = learner.update(END,MICRO,4,guard)
                    expected = old_ledger[i]
                    require(row["step"] == i+1 and row["lr"] == expected["lr"] == rates[i]
                            and {key:list(value) for key,value in witness.items()} == expected["noise"]
                            and dict(Counter(groups[banks["target"][i]])) == PARTS
                            and all(row["components"][key]["visit_sha256"] == expected["components"][key]["visit_sha256"] for key in BATCH)
                            and row["components"]["auxiliary"]["slots"] == expected["components"]["auxiliary"]["slots"]
                            and np.isfinite(row["global_gradient_norm"]),
                            "Actual samples/noise/loss support changed from the paired control")
                    counts = validate_optimizer_clock(learner.optimizer,learner.step)
                    require(all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(learner.parameters)),
                            "Nonfinite actual candidate parameters")
                    row.update(noise=dict(witness),go2=PARTS,adam_counts=counts,
                        clip_scope="PRIVATE_IO_ONLY_ORIGINAL_CLIP_NORM",shared_optimizer_invoked=False)
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
                    recorded = learner.step
                    if learner.step in SAVE_STEPS:
                        saved = save(learner,run,inventory,binding["checkpoint_bytes"],shared_hash)
            status = "FORMAL_FROZEN_EXPERT_250_COMPLETE"
        except TimeoutError:
            status = "SAVED_AT_FINITE_BOUNDARY"
        finally:
            if learner.step and (saved is None or saved["step"] != learner.step):
                saved = save(learner,run,inventory,binding["checkpoint_bytes"],shared_hash)
            write(run/"TERMINAL.json",dict(scope=SCOPE,status=status,formal_updates=learner.step,
                updates_not_in_ledger=learner.step-recorded,engineering_updates=0,checkpoint=saved,
                ledger_sha256=sha(run/"UPDATES.jsonl"),frozen_shared_expert_sha256=shared_hash))


def main(capsule):
    import jax
    import shutil
    from v4_main250_hopper import allocation
    from v62_source_admission import award

    binding = read(capsule/"BINDING.json")
    admission(binding)
    require(shutil.disk_usage(capsule).free > 3*binding["checkpoint_bytes"],"Full endpoint reserve unavailable")
    with closing(OwnerLock(capsule/"OWNER.lock")):
        current = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind,"Use one allocated H200")
        current["project_award"] = award(binding,capsule,current)
        write(capsule/"START.json",current)
        jax.config.update("jax_compilation_cache_dir",str(capsule/"jit-cache"))
        train(capsule,binding,current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule",type=Path)
    main(**vars(parser.parse_args()))
