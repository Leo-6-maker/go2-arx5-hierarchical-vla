"""Train a matched neutral-input pair with one shared R4 parameter anchor."""
import argparse
from collections import Counter
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v7_formal_worker import exact_noise_witness, input_evidence, save
from v7_full_model_qualification import load_inputs, MODEL_NOISE_SEED
from v7_full_model_paths import extend_target_io
from v7_parent_anchor import SharedParentAnchor
from v7_parent_anchor_qualification import SCOPE as QUALIFICATION_SCOPE

SCOPE = "V7_B6_SHARED_L2SP_MATCHED_V1"
ARMS = ("LAMBDA_ZERO", "SHARED_L2SP")
STEPS = 500
SAVE_STEPS = (100,250,500)


def train(capsule, binding, allocation):
    import jax
    import jax.numpy as jnp
    from v4_long_engine import LongLearner
    from v6_dual_course import reduced_fm
    from v61_source_course import corrected_parent
    from v65_recovery_course import AUX, BATCH, MICRO, PARTS, demographics
    from v65_recovery_worker import partitions

    source = Path(binding["source_pair"])
    end_step = binding["end_step"]
    require(end_step in (100,250,500), "Use a preregistered finite endpoint")
    original = read(pin(source/"BINDING.json",binding["source_pair_binding_sha256"]))
    calibration = read(pin(binding["calibration"]["path"],binding["calibration"]["sha256"]))
    recovered = read(pin(binding["core_restore"]["path"],binding["core_restore"]["sha256"]))
    require(recovered["scope"]==QUALIFICATION_SCOPE and recovered["complete_training_endpoint"]
            and calibration["zero_coefficient_original_update_equal"]
            and calibration["gradient_fraction"]==.1 and calibration["train_step"]==251,
            "Use the completed independent qualification and fixed TRAIN calibration")
    coefficient = calibration["coefficient"]
    require(np.isfinite(coefficient) and coefficient>0, "Fixed positive coefficient required")
    config,datasets,tokenizer,banks,slots,rates,context = load_inputs(original)
    p = config["parent_paths"]
    graphs,parent,backbone,_ = corrected_parent(Path(p["custody"]),Path(p["short"]),Path(p["r4"]))
    active,frozen,_,inventory = partitions(parent,backbone)
    require(inventory["total_frozen_sha256"]==recovered["frozen_weight_sha256"],"Qualified parent changed")
    _,_,groups=demographics(datasets["target"])
    stopping=False

    def signal_stop(*_):
        nonlocal stopping
        stopping=True

    def guard():
        if stopping or (capsule/"STOP.json").exists() or time.time()>=allocation["deadline"]-900:
            raise TimeoutError("Matched parent-anchor run reached its finite save boundary")

    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    @jax.jit
    def finite_parameters(parameters):
        return jnp.stack([jnp.isfinite(value).all() for value in jax.tree.leaves(parameters)]).all()

    complete={}
    old_ledger=source/"CONTROL_NEUTRAL/UPDATES.jsonl"
    old_terminal=read(source/"CONTROL_NEUTRAL/TERMINAL.json")
    pin(old_ledger,old_terminal["optimizer_ledger_sha256"])
    reference_noise=[json.loads(row)["noise"] for row in old_ledger.read_text().splitlines()]
    require(len(reference_noise)==STEPS,"Original paired noise ledger incomplete")
    for arm,weight in zip(ARMS,(0.,coefficient),strict=True):
        guard()
        run=capsule/arm
        run.mkdir(mode=0o700)
        parameters=jax.tree.map(lambda value:jnp.array(value,copy=True),extend_target_io(active))
        anchor=SharedParentAnchor(active["shared"])
        learner=LongLearner(graphs=graphs,parameters=parameters,frozen=frozen,datasets=datasets,tokenizer=tokenizer,
                banks=banks,auxiliary_slots=slots,auxiliary=datasets["target"].aux,
                cursors=dict.fromkeys(BATCH,0),rates=rates,rng=jax.random.PRNGKey(MODEL_NOISE_SEED),
                batch_by_domain=BATCH,gradient_join=partial(reduced_fm,batch=BATCH,aux_weight=AUX),
                target_context=context,context_treatment=False)
        with closing(learner):
            expected=read(pin(original["qualification_input"]["path"],original["qualification_input"]["sha256"]))
            require(input_evidence(learner,0)==expected,"New pair changed original inputs or CUDA noise")
            learner.apply=anchor.wrap(learner.apply,weight)
            witness=exact_noise_witness(learner)
            write(run/"RUN_CONTRACT.json",dict(scope=SCOPE,arm=arm,coefficient=weight,steps=STEPS,
                  neutral_previous_command=True,initial_R4_penalty=0,seed=MODEL_NOISE_SEED,end_step=end_step,
                  batch=BATCH,parts=PARTS,micro=MICRO,auxiliary_weight=AUX,save_steps=SAVE_STEPS,
                  data_bank_sha256=original["bank"]["sha256"],binding_sha256=sha(capsule/"BINDING.json")))
            saved=None
            recorded=0
            status="INTERRUPTED"
            try:
                with (run/"UPDATES.jsonl").open("x",buffering=1) as stream:
                    for step in range(end_step):
                        witness.clear()
                        row=learner.update(end_step,MICRO,4,guard)
                        require(row["step"]==step+1 and row["lr"]==rates[step]
                                and {k:list(v) for k,v in witness.items()}==reference_noise[step]
                                and dict(Counter(groups[banks["target"][step]]))==PARTS
                                and np.isfinite(row["global_gradient_norm"])
                                and bool(finite_parameters(learner.parameters)),
                                "Actual fixed step, mixture or CUDA stream changed")
                        row.update(noise=dict(witness),parent_anchor=dict(anchor.last),go2=PARTS)
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                        recorded=learner.step
                        if learner.step in SAVE_STEPS:
                            saved=save(learner,run,inventory,binding["checkpoint_bytes"])
                status="FORMAL_ENDPOINT_COMPLETE"
            except TimeoutError:
                status="SAVED_AT_FINITE_BOUNDARY"
            finally:
                if learner.step>0 and (saved is None or saved["step"]!=learner.step):
                    saved=save(learner,run,inventory,binding["checkpoint_bytes"])
                terminal=dict(scope=SCOPE,arm=arm,status=status,formal_updates=learner.step,
                              updates_not_in_ledger=learner.step-recorded,coefficient=weight,checkpoint=saved,
                              ledger_sha256=sha(run/"UPDATES.jsonl"))
                write(run/"TERMINAL.json",terminal)
            complete[arm]=dict(terminal_sha256=sha(run/"TERMINAL.json"),formal_updates=learner.step)
            if status!="FORMAL_ENDPOINT_COMPLETE":
                break
    total=sum(value["formal_updates"] for value in complete.values())
    write(capsule/"TERMINAL.json",dict(scope=SCOPE,status="PAIR_COMPLETE" if total==2*end_step else "PAIR_PARTIAL",
          formal_updates=total,end_step=end_step,engineering_updates=0,arms=complete,calibration_sha256=binding["calibration"]["sha256"]))


def main(capsule):
    import jax
    from v4_main250_hopper import allocation
    from v62_source_admission import award

    binding=read(capsule/"BINDING.json")
    require(binding["scope"]==SCOPE and binding["formal_updates"]==2*binding["end_step"],
            "Requires the fixed matched formal pair")
    write_slots = 2*sum(step<=binding["end_step"] for step in SAVE_STEPS)+1
    require(shutil.disk_usage(capsule).free>write_slots*binding["checkpoint_bytes"],"Checkpoint write reserve unavailable")
    with closing(OwnerLock(capsule/"OWNER.lock")):
        assigned=allocation()
        require(jax.default_backend()=="gpu" and "H200" in jax.local_devices()[0].device_kind,"Expected one H200")
        assigned["project_award"]=award(binding,capsule,assigned)
        write(capsule/"START.json",assigned)
        jax.config.update("jax_compilation_cache_dir",str(capsule/"jit-cache"))
        train(capsule,binding,assigned)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule",type=Path)
    main(**vars(parser.parse_args()))
