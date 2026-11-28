"""Qualify exact B6 step-101 continuation with one counted engineering update."""
import argparse
from contextlib import closing
from functools import partial
from pathlib import Path
import time

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import OwnerLock, write


def run(capsule):
    import jax
    import jax.numpy as jnp
    from v4_long_engine import LongLearner
    from v6_dual_course import reduced_fm
    from v61_source_course import corrected_parent
    from v65_recovery_course import AUX, BATCH, MICRO
    from v65_recovery_worker import partitions
    from v7_formal_worker import exact_noise_witness, save
    from v7_full_model_qualification import load_inputs, MODEL_NOISE_SEED
    from v7_full_model_paths import extend_target_io
    from v7_parent_anchor import SharedParentAnchor
    from v7_parent_anchor_resume import ARMS, next_input, restore_learner
    from v4_main250_hopper import allocation
    from v62_source_admission import award

    binding = read(capsule / "BINDING.json")
    assigned = allocation()
    require(jax.default_backend() == "gpu" and "H200" in jax.local_devices()[0].device_kind,
            "Use the actual H200 training backend")
    assigned["project_award"] = award(binding, capsule, assigned)
    write(capsule / "START.json", assigned)
    source = Path(binding["source_pair"])
    original = read(pin(source / "BINDING.json", binding["source_pair_binding_sha256"]))
    config, datasets, tokenizer, banks, slots, rates, context = load_inputs(original)
    p = config["parent_paths"]
    graphs, parent, backbone, _ = corrected_parent(Path(p["custody"]), Path(p["short"]), Path(p["r4"]))
    active, frozen, _, inventory = partitions(parent, backbone)
    coefficient = read(pin(binding["calibration"]["path"], binding["calibration"]["sha256"]))["coefficient"]
    jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))

    def guard():
        if (capsule / "STOP.json").exists() or time.time() >= assigned["deadline"] - 900:
            raise TimeoutError("Resume qualification save boundary")

    reports = []
    for arm in ARMS:
        parameters = jax.tree.map(lambda value: jnp.array(value, copy=True), extend_target_io(active))
        learner = LongLearner(graphs=graphs, parameters=parameters, frozen=frozen,
            datasets=datasets, tokenizer=tokenizer, banks=banks, auxiliary_slots=slots,
            auxiliary=datasets["target"].aux, cursors=dict.fromkeys(BATCH, 0), rates=rates,
            rng=jax.random.PRNGKey(MODEL_NOISE_SEED), batch_by_domain=BATCH,
            gradient_join=partial(reduced_fm, batch=BATCH, aux_weight=AUX),
            target_context=context, context_treatment=False)
        with closing(learner):
            report = restore_learner(learner, binding, arm)
            report["input"] = next_input(learner, binding, report["step"])
            require(report["next_step"] == 101 and report["next_lr"] == rates[100] != rates[0],
                    "Qualification reset the schedule")
            reports.append(report)
            if arm == "SHARED_L2SP":
                anchor = SharedParentAnchor(active["shared"])
                learner.apply = anchor.wrap(learner.apply, coefficient)
                witness = exact_noise_witness(learner)
                row = learner.update(101, MICRO, 4, guard)
                require(row["step"] == 101 and row["lr"] == rates[100]
                        and anchor.last["raw_loss"] > 0 and anchor.last["weighted_gradient_l2"] > 0,
                        "Actual resumed optimizer or parent anchor failed")
                expected = reports[-1]["input"]
                require(all(list(witness[d]) == [expected[d]["noise"], expected[d]["tau"]] for d in BATCH)
                        and list(witness["auxiliary"]) == [expected["auxiliary"]], "Actual gradient draw differs")
                write(capsule / "UPDATE0101.json", dict(row, parent_anchor=anchor.last, noise=dict(witness)))
                checkpoint = save(learner, capsule, inventory, binding["checkpoint_bytes"])
    require(reports[0]["input"] == reports[1]["input"], "Matched restored next inputs differ")
    write(capsule / "TERMINAL.json", dict(scope="V7_B6_RESUME_QUALIFICATION_V1", status="GPU_QUALIFIED",
          source_step=100, actual_engineering_updates=1, formal_updates=0, arms=reports,
          checkpoint=checkpoint, no_duplicate_steps=True, no_warmup_reset=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule", type=Path)
    args = cli.parse_args()
    with closing(OwnerLock(args.capsule / "OWNER.lock")):
        run(**vars(args))
