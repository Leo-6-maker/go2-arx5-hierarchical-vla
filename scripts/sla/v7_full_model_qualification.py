"""Run the fixed V7-A full R4 qualification on one allocated Hopper H200."""

import argparse
from contextlib import closing
from functools import partial
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v61_source_course import R4, corrected_parent
from v61_source_worker import parent_gate
from v62_source_admission import award, write_capacity
from v65_recovery_course import AUX, BATCH, MICRO, courses, validate
from v65_recovery_worker import partitions, recover
from v7_full_model_paths import PROJECTION_KEY, auxiliary_h40, conditioned_policy, deployment_h40, extend_target_io
from v7_matched_bank import MODEL_NOISE_SEED, STEPS, rates


PLAN = "67d15fb8fad688a2bbd82e225111bb49ec34559b"
BANK = "ec5e7436b839e8452b61fc137f4fe6f659af2aabb522165c7604413b12d20024"
ARRAYS = "a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff"
CONTEXT = "7573a2a63170b2a5bf47d51314e08f0c24724ec5a508bed6c61e9fbb4b85fda2"
ARMS = ("CONTROL_NEUTRAL", "TREATMENT_PREVIOUS_EXECUTED_GRIPPER_COMMAND")
ENGINEERING_UPDATES = 4
OUTPUT_ATOL, OUTPUT_RTOL = 1.e-6, 2.e-5
GRADIENT_REL_L2 = 2.e-5


def file_ref(ref):
    return pin(Path(ref["path"]), ref["sha256"])


def bytes_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def load_inputs(binding):
    config = read(file_ref(binding["data"]))
    datasets, tokenizer, identity = courses(config)
    manifest = read(file_ref(binding["manifest"]))
    require(binding["manifest"]["sha256"] == BANK and binding["bank"]["sha256"] == ARRAYS
            and binding["context"]["sha256"] == CONTEXT and manifest["reader_order"] == identity
            and manifest["bank_sha256"] == ARRAYS and manifest["previous_command_sha256"] == CONTEXT
            and manifest["candidate_model_noise_seed"] == MODEL_NOISE_SEED,
            "The approved real reader, sample ordering or noise seed changed")
    with np.load(file_ref(binding["bank"]), allow_pickle=False) as arrays:
        banks = {d: arrays[d] for d in BATCH}
        slots, schedule = arrays["auxiliary_slots"], arrays["rates"]
    with np.load(file_ref(binding["context"]), allow_pickle=False) as arrays:
        context = dict(arrays)
    planned = rates()
    validate(datasets, banks, slots, schedule, end=STEPS, rate_schedule=lambda step: planned[step-1])
    require(context["command_valid"].all() and np.array_equal(context["query_tick"]-25,
            context["consumed_start_tick"]), "The real previous-command validity or interval changed")
    return config, datasets, tokenizer, banks, slots, schedule, context


def aux_noise(learner, batch, row):
    import jax
    import jax.numpy as jnp

    key = jax.random.fold_in(learner.rng, 37)
    slots = learner.auxiliary_slots[row]
    keys = jnp.stack([jax.random.fold_in(key, batch.cursor + int(slot)) for slot in slots])
    return jax.vmap(lambda k: jax.random.normal(k, (40, 32)))(keys)


def input_evidence(learner, row):
    import jax

    result = {}
    for domain in BATCH:
        batch = learner.prepare(row, domain)
        noise, tau = learner.draws[domain](learner.rng, batch.cursor)
        result[domain] = dict(indices=bytes_hash(batch.indices), cursor=batch.cursor,
                              host=[bytes_hash(x) for x in jax.tree.leaves(batch.host)],
                              noise=bytes_hash(noise), tau=bytes_hash(tau),
                              support=bytes_hash(batch.support) if batch.support is not None else None)
        if domain == "target":
            result["auxiliary"] = bytes_hash(aux_noise(learner, batch, row))
    return result


def agreement(a, b):
    left, right = np.asarray(a), np.asarray(b)
    require(left.shape == right.shape and np.isfinite(right).all(), "Nonfinite or incompatible path output")
    return dict(equal=np.array_equal(left, right), close=np.allclose(left, right, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL),
                maximum_absolute=float(np.max(np.abs(left-right))), sha256=bytes_hash(right))


def paths(learner, graph, original, treatment):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6
    from openpi.models import pi0
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    batch = learner.prepare(0, "target")
    slots = learner.auxiliary_slots[0]
    index = next(int(pos) for pos in slots if treatment[int(pos), 0] != 0)
    aux_index = int(np.where(slots == index)[0][0])
    observation, actions = batch_v2.device_batch(jax.tree.map(lambda x: x[index:index+1], batch.host))
    support = jax.device_put(batch.support[index:index+1])
    actions = jnp.where(support[..., None], actions, 0.)
    noise, tau = learner.draws["target"](learner.rng, batch.cursor)
    flow = noise[index:index+1], tau[index:index+1]
    aux = aux_noise(learner, batch, 0)[aux_index:aux_index+1]
    deploy_rng = jax.random.fold_in(learner.rng, 197)
    deploy_noise = jax.random.normal(deploy_rng, (1, 40, 32))
    condition = jnp.asarray(treatment[index:index+1])
    neutral = jnp.zeros_like(condition)
    shared, target, frozen = learner.parameters["shared"], learner.parameters["target"], learner.frozen
    fields, _ = extracted_fields(batch_v2, pi0)
    old_aux = full_h40_v6.make_sampler(graph, fields, precision="highest")
    new_aux = auxiliary_h40(graph, fields)

    @jax.jit
    def baseline(shared, private, frozen, observation, actions, noise, tau, deploy_noise):
        with jax.default_matmul_precision("highest"):
            model = nnx.merge(graph, shared, private, frozen)
            return (batch_v2.error32(model, observation, actions, noise, tau),
                    model.sample_actions(deploy_rng, observation, num_steps=10, noise=deploy_noise))

    @jax.jit
    def candidate(shared, target, frozen, observation, actions, noise, tau, condition, deploy_noise):
        with jax.default_matmul_precision("highest"):
            model = conditioned_policy(graph, shared, target, frozen, condition)
            return (batch_v2.error32(model, observation, actions, noise, tau),
                    deployment_h40(graph, shared, target, frozen, observation, deploy_noise, condition, deploy_rng))

    def actual(candidate_target, context):
        return (*candidate(shared, candidate_target, frozen, observation, actions, *flow, context, deploy_noise),
                new_aux(dict(shared=shared, private=candidate_target), frozen, observation, aux, context))

    def standard():
        return (*baseline(shared, original, frozen, observation, actions, *flow, deploy_noise),
                old_aux(nnx.State.merge(shared, original), frozen, observation, aux))

    reference, repeated = standard(), standard()
    control, zero_treatment = actual(target, neutral), actual(target, condition)
    injected = jax.tree.map(lambda x: x, target)
    injected.replace_by_pure_dict({**target.to_pure_dict(),
                                  PROJECTION_KEY: jnp.full((1, 1024), .125, dtype=jnp.float32)})
    sensitivity = actual(injected, condition)
    report = {}
    for name, a, b, c, d, e in zip(("fm_error32", "deployment_h40", "aux_h40"),
                                  reference, repeated, control, zero_treatment, sensitivity, strict=True):
        comparison = dict(replay=agreement(a,b), control=agreement(a,c), treatment=agreement(a,d),
                          injection=agreement(d,e))
        require(comparison["replay"]["equal"] and comparison["control"]["close"]
                and comparison["treatment"]["close"] and not comparison["injection"]["close"],
                f"Full-model route failed zero projection or nonzero injection: {name}")
        report[name] = comparison
    return dict(paths=report, input_index=int(batch.indices[index]), fm_noise=bytes_hash(flow[0]),
                fm_tau=bytes_hash(flow[1]), deploy_noise=bytes_hash(deploy_noise), aux_noise=bytes_hash(aux))


def projection(tree):
    return tree["target"][PROJECTION_KEY].value


def relative_common_gradient(left, right):
    import jax
    import optax

    def base(tree):
        _, target = tree["target"].split(lambda path, _: path[0] == PROJECTION_KEY, ...)
        return dict(tree,target=target)

    a, b = base(left), base(right)
    difference = jax.tree.map(lambda x, y: x-y,a,b)
    norm = float(optax.global_norm(a))
    require(np.isfinite(norm) and norm > 0, "Invalid common gradient")
    return float(optax.global_norm(difference))/norm


def append_update(output, arm, learner, guard, joint, slot):
    import jax
    import optax

    guard()
    clip_input = float(optax.global_norm(joint))
    require(np.isfinite(clip_input) and clip_input > 0, "Nonfinite joint gradient")
    ident = output / f"{arm}-ENGINEERING-{slot}.json"
    write(ident, dict(arm=arm, intended_update=slot, optimizer_applied=False))
    previous = projection(learner.parameters)
    params, state, norm = learner.apply(learner.parameters, learner.optimizer, joint, float(learner.rates[learner.step]))
    jax.block_until_ready((params,state))
    leaf, delta = projection(params), float(np.linalg.norm(np.asarray(projection(params)-previous)))
    require(np.isfinite(np.asarray(leaf)).all() and (delta > 0 if arm == ARMS[1] else delta == 0),
            "Projection did not follow the original Adam path")
    learner.parameters, learner.optimizer = params, state
    learner.step += 1
    counts = {name:int(v[0].count) for name,v in state.items()}
    require(set(counts) == {"shared", *BATCH} and set(counts.values()) == {learner.step}, "Adam cursor changed")
    receipt = dict(arm=arm, actual_engineering_update=slot, optimizer_applied=True, adam=counts,
                   clip_factor=min(1.,1./norm), gradient_norm=norm, projection_delta=delta, projection_sha256=bytes_hash(leaf))
    write(output / f"{arm}-ENGINEERING-{slot}-ACTUAL.json", receipt)
    return receipt


def qualify(capsule, binding, site):
    import jax
    import optax
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint
    from v3_main_place_learn import endpoint_files
    from v4_long_probe import snapshot
    from v4_long_engine import LongLearner
    from v6_dual_course import reduced_fm

    config,datasets,tokenizer,banks,slots,schedule,context = load_inputs(binding)
    p = config["parent_paths"]
    graphs, parent, backbone, lineage = corrected_parent(Path(p["custody"]), Path(p["short"]), Path(p["r4"]))
    active, frozen, lora, inventory = partitions(parent,backbone)
    write(capsule/"PARAMETERS.json",dict(inventory, ancestry=lineage, added_projection=[1,1024]))
    original = active["target"]
    rng = jax.random.PRNGKey(MODEL_NOISE_SEED)
    graph = graphs["target"]

    def guard():
        if (capsule/"STOP.json").exists() or time.time() >= site["deadline"]-600:
            raise TimeoutError("Original STOP or save reserve reached")

    def make(treatment):
        return LongLearner(graphs=graphs, parameters=extend_target_io(active), frozen=frozen, datasets=datasets,
                           tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=datasets["target"].aux,
                           cursors=dict.fromkeys(BATCH,0), rates=schedule, rng=rng, batch_by_domain=BATCH,
                           gradient_join=partial(reduced_fm,batch=BATCH,aux_weight=AUX),
                           target_context=context, context_treatment=treatment)

    with closing(make(False)) as control, closing(make(True)) as treatment:
        first = [input_evidence(learner,0) for learner in (control,treatment)]
        require(first[0] == first[1], "Actual CUDA input, support or noise differs between arms")
        write(capsule/"STEP0-INPUTS.json",first[0])
        path = paths(treatment,graph,original,treatment.prepare(0,"target").condition)
        write(capsule/"STEP0-PATHS.json",path)
        measured = [learner.components(0,learner.prepared(0,1),MICRO,4,guard) for learner in (control,treatment)]
        control_gradient, treatment_gradient = (m[0] for m in measured)
        c,t = (float(optax.global_norm(projection(x))) for x in (control_gradient,treatment_gradient))
        common = relative_common_gradient(control_gradient,treatment_gradient)
        require(c == 0 and np.isfinite(t) and t > 0 and common <= GRADIENT_REL_L2,
                "Actual initial projection or common gradient failed")
        write(capsule/"STEP0-GRADIENT.json",dict(control=c,treatment=t,relative_common_l2=common,
                                               component_metrics=[m[2] for m in measured]))
        updates = [append_update(capsule,arm,learner,guard,joint,1) for arm,learner,joint in
                   zip(ARMS,(control,treatment),(control_gradient,treatment_gradient),strict=True)]
        del measured,control_gradient,treatment_gradient
        cold_reports = []
        for arm,hot in zip(ARMS,(control,treatment),strict=True):
            guard()
            require(shutil.disk_usage(capsule).free > 2*site["write_capacity"]["checkpoint_bytes"],
                    "Qualification write reserve changed")
            checkpoint = capsule/f"{arm}-STEP0001"
            with ocp.PyTreeCheckpointer() as writer:
                writer.save(checkpoint.resolve(),snapshot(hot))
            export = capsule/f"{arm}-PROJECTION.npy"
            with export.open("xb") as stream:
                np.save(stream,np.asarray(projection(hot.parameters)),allow_pickle=False)
            hot_inputs = input_evidence(hot,1)
            hot_joint,_,hot_metrics = hot.components(1,hot.prepared(1,2),MICRO,4,guard)
            with closing(make(arm == ARMS[1])) as cold:
                # The schema is independently created, but the original restore
                # helper requires the expected complete state to audit every leaf.
                cold.parameters, cold.optimizer, cold.step = hot.parameters, hot.optimizer, 1
                recover(cold,checkpoint)
                require(hot_inputs == input_evidence(cold,1)
                        and np.array_equal(np.load(export,allow_pickle=False),projection(cold.parameters)),
                        "Cold planner inputs or exported deployment projection changed")
                joint,_,metrics = cold.components(1,cold.prepared(1,2),MICRO,4,guard)
                require(all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(hot_joint),
                            jax.tree.leaves(joint),strict=True))
                        and all(metrics[k]["loss"] == hot_metrics[k]["loss"] for k in metrics),
                        "Cold replay did not reproduce the next gradient and losses")
                updates.append(append_update(capsule,arm,cold,guard,joint,2))
            cold_reports.append(dict(arm=arm, files=endpoint_files(checkpoint), projection_sha256=sha(export),
                                     inputs=hot_inputs, exact_next_gradient=True))
        require(frozen_fingerprint(frozen) == inventory["total_frozen_sha256"]
                and frozen_fingerprint(lora) == inventory["parent_lora_sha256"], "Frozen R4 values changed")
        write(capsule/"TERMINAL.json",dict(status="GPU_QUALIFIED",plan=PLAN, r4=R4, calls=path,
                    cold_restore=cold_reports, updates=updates, actual_engineering_updates=len(updates),formal_updates=0))


def main(capsule):
    import jax
    from v4_main250_hopper import allocation

    binding = read(capsule/"BINDING.json")
    require(binding["plan"] == PLAN and binding["r4_receipt"] == R4
            and binding["engineering_updates"] == ENGINEERING_UPDATES and not binding["formal_updates"],
            "Wrong registered V7 qualification")
    parent_gate(binding["parent"])
    config = read(file_ref(binding["data"]))
    with closing(OwnerLock(capsule/"OWNER.lock")):
        site = allocation()
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind,"Qualification needs the allocated H200")
        site["project_award"] = award(binding,capsule,site)
        site["write_capacity"] = write_capacity(capsule,Path(config["parent_paths"]["r4"]),"probe",0,0,
                            binding["budget_seconds"],quota_root=Path(binding["quota_root"]))
        write(capsule/"START.json",site)
        jax.config.update("jax_compilation_cache_dir",str(capsule/"jit-cache"))
        try:
            qualify(capsule,binding,site)
        except Exception as error:
            known = len(list(capsule.glob("*-ENGINEERING-*-ACTUAL.json")))
            written = len(list(capsule.glob("*-ENGINEERING-?.json")))
            write(capsule/"FAILED.json",dict(type=type(error).__name__,reason=str(error),
                  optimizer_updates_confirmed=known,optimizer_updates_uncertain=written-known,formal_updates=0))
            raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule",type=Path)
    main(cli.parse_args().capsule)
