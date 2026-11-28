"""Calibrate one R4 shared-parameter anchor on fixed TRAIN and qualify its restore."""
import argparse
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v7_parent_anchor import SharedParentAnchor
from v7_loss_replay import BINDING, TERMINAL, SEAL, START, ARM
from v7_formal_worker import exact_noise_witness, save
from v7_full_model_qualification import load_inputs, MODEL_NOISE_SEED
from v7_full_model_paths import extend_target_io

SCOPE = "V7_B6_SHARED_L2SP_QUALIFICATION_V1"
GRADIENT_FRACTION = .1
ENGINEERING_UPDATES = 2


def run(capsule, output, allocation):
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v4_long_engine import LongLearner
    from v4_long_probe import snapshot
    from v6_dual_course import reduced_fm
    from v61_source_course import corrected_parent
    from v65_recovery_course import AUX, BATCH, MICRO
    from v65_recovery_worker import partitions

    binding = read(pin(capsule / "BINDING.json", BINDING))
    original = read(pin(capsule / "TERMINAL.json", TERMINAL))
    require(original["formal_updates"] == 1000, "Original V7-A pair must be terminal")
    config, datasets, tokenizer, banks, slots, rates, context = load_inputs(binding)
    paths = config["parent_paths"]
    graphs, parent, backbone, _ = corrected_parent(Path(paths["custody"]), Path(paths["short"]), Path(paths["r4"]))
    active, frozen, _, inventory = partitions(parent, backbone)
    seal = read(pin(capsule / ARM / "STEP0250_SEALED.json", SEAL))
    require(inventory["total_frozen_sha256"] == seal["frozen_weight_sha256"], "R4 frozen identity changed")

    def guard():
        if (output / "STOP.json").exists() or time.time() >= allocation["deadline"] - 240:
            raise TimeoutError("Qualification reached its existing STOP or save boundary")

    def factory():
        parameters = jax.tree.map(lambda value: jnp.array(value, copy=True), extend_target_io(active))
        return LongLearner(graphs=graphs, parameters=parameters, frozen=frozen,
            datasets=datasets, tokenizer=tokenizer, banks=banks, auxiliary_slots=slots,
            auxiliary=datasets["target"].aux, cursors=dict.fromkeys(BATCH, 0), rates=rates,
            rng=jax.random.PRNGKey(MODEL_NOISE_SEED), batch_by_domain=BATCH,
            gradient_join=partial(reduced_fm, batch=BATCH, aux_weight=AUX),
            target_context=context, context_treatment=False)

    anchor = SharedParentAnchor(active["shared"])
    with closing(factory()) as learner:
        initial_loss, initial_gradient = anchor.value_and_gradient(learner.parameters["shared"])
        require(float(initial_loss) == 0 and float(optax.global_norm(initial_gradient)) == 0,
                "R4 must start with zero parent penalty")
        with ocp.PyTreeCheckpointer() as reader:
            saved = reader.restore((capsule / ARM / "STEP0250").resolve(), item=snapshot(learner))
        for name, values in saved["parameters"].items():
            learner.parameters[name].replace_by_pure_dict(values)
        learner.optimizer, learner.rng, learner.step = saved["optimizer"], saved["rng"], START
        witness = exact_noise_witness(learner)
        joint, _, metrics = learner.components(START, learner.prepared(START, START+1), MICRO, 4, guard)
        ledger = read(pin(capsule/ARM/"TERMINAL.json", original["arms"][ARM]["terminal_sha256"]))
        expected = json.loads(pin(capsule/ARM/"UPDATES.jsonl",ledger["optimizer_ledger_sha256"]).read_bytes().splitlines()[START])
        require({key:list(value) for key,value in witness.items()} == expected["noise"],
                "Calibration must use the archived actual TRAIN noise")
        require(all(metrics[key]["visit_sha256"] == expected["components"][key]["visit_sha256"] for key in BATCH),
                "Calibration sample identity changed")
        loss, regularizer = anchor.value_and_gradient(learner.parameters["shared"])
        original_norm, penalty_norm = float(optax.global_norm(joint["shared"])), float(optax.global_norm(regularizer))
        require(original_norm > 0 and penalty_norm > 0 and np.isfinite([original_norm,penalty_norm,float(loss)]).all(),
                "Calibrate only at the predeclared finite nonzero TRAIN drift point")
        coefficient = GRADIENT_FRACTION * original_norm / penalty_norm
        original_apply = learner.apply
        expected_update = original_apply(learner.parameters,learner.optimizer,joint,float(rates[START]))
        zero_update = anchor.wrap(original_apply,0.)(learner.parameters,learner.optimizer,joint,float(rates[START]))
        require(all(np.array_equal(np.asarray(a),np.asarray(b))
                    for a,b in zip(jax.tree.leaves(expected_update),jax.tree.leaves(zero_update),strict=True)),
                "Zero coefficient changed the original optimizer update")
        write(output/"CALIBRATION.json",dict(scope=SCOPE,train_step=START+1,source_seal_sha256=SEAL,
              gradient_fraction=GRADIENT_FRACTION,shared_objective_gradient_l2=original_norm,
              raw_parent_penalty_gradient_l2=penalty_norm,raw_parent_penalty=float(loss),coefficient=coefficient,
              zero_coefficient_original_update_equal=True,initial_parent_penalty_zero=True,
              optimization_scope="INHERITED_TRAINABLE_SHARED_ACTION_EXPERT",persistent_updates=0,
              sample_noise=expected["noise"],sample_visits={key:metrics[key]["visit_sha256"] for key in BATCH}))
        del saved, joint, regularizer, expected_update, zero_update
    jax.clear_caches()

    with closing(factory()) as learner:
        require(float(anchor.value_and_gradient(learner.parameters["shared"])[0]) == 0,
                "Engineering qualification must restart at the exact R4")
        learner.apply = anchor.wrap(learner.apply, coefficient)
        updates = []
        for _ in range(ENGINEERING_UPDATES):
            row = learner.update(ENGINEERING_UPDATES,MICRO,4,guard)
            require(np.isfinite(row["global_gradient_norm"]), "Nonfinite qualified update")
            row["parent_anchor"] = dict(anchor.last)
            updates.append(row)
            write(output/f"UPDATE{learner.step:04d}.json",row)
        require(updates[0]["parent_anchor"]["weighted_gradient_l2"] == 0
                and updates[1]["parent_anchor"]["weighted_gradient_l2"] > 0,
                "The real parameter anchor did not enter the post-initial update")
        require(frozen_fingerprint(learner.frozen) == inventory["total_frozen_sha256"], "Frozen parameters changed")
        checkpoint = save(learner,output,inventory,read(output/"BINDING.json")["checkpoint_bytes"])
        reference = snapshot(learner)
        with ocp.PyTreeCheckpointer() as reader:
            recovered = reader.restore((output/"STEP0002").resolve(),item=reference)
        require(all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                    zip(jax.tree.leaves(reference),jax.tree.leaves(recovered),strict=True)),
                "Saved parameters, optimizer, random state or counters failed restore")
        del reference,recovered
    jax.clear_caches()
    # The existing endpoint reader performs its independent CPU restore in a separate process.
    write(output/"GPU-TERMINAL.json",dict(scope=SCOPE,status="GPU_QUALIFIED_PENDING_INDEPENDENT_RESTORE",
          coefficient=coefficient,calibration_sha256=sha(output/"CALIBRATION.json"),engineering_updates=ENGINEERING_UPDATES,
          formal_updates=0,checkpoint_sha256=sha(output/"STEP0002_SEALED.json"),
          frozen_weight_sha256=inventory["total_frozen_sha256"],cold_in_process_restore_equal=True))

def main(capsule,output):
    import jax
    from v4_main250_hopper import allocation
    from v62_source_admission import award

    binding=read(output/"BINDING.json")
    require(binding["scope"]==SCOPE and binding["engineering_updates"]==ENGINEERING_UPDATES,
            "Use the sealed two-update qualification")
    with closing(OwnerLock(output/"OWNER.lock")):
        current=allocation()
        require(jax.default_backend()=="gpu" and "H200" in jax.local_devices()[0].device_kind, "Expected one H200")
        current["project_award"]=award(binding,output,current)
        write(output/"START.json",current)
        jax.config.update("jax_compilation_cache_dir",str(output/"jit-cache"))
        run(capsule,output,current)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule",type=Path)
    parser.add_argument("output",type=Path)
    main(**vars(parser.parse_args()))
