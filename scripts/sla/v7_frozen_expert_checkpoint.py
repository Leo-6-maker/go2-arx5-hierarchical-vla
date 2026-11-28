"""Complete checkpoints with an honest frozen-shared Adam clock of zero."""

from pathlib import Path
import shutil

import numpy as np

from v7_frozen_expert_optimizer import ACTIVE, GROUPS, SCOPE


def validate_optimizer_clock(optimizer, step):
    import jax

    if type(step) is not int or step < 0 or set(optimizer) != GROUPS:
        raise ValueError("FROZEN_EXPERT_INVALID_STEP_OR_OPTIMIZER_SCOPE")
    actual = {key:int(state[0].count) for key,state in optimizer.items()}
    expected = {key:step if key in ACTIVE else 0 for key in GROUPS}
    if actual != expected:
        raise ValueError("FROZEN_EXPERT_WRONG_ADAM_CLOCK")
    frozen = optimizer["shared"][0]
    if any(np.any(np.asarray(value)) for value in jax.tree.leaves((frozen.mu,frozen.nu))):
        raise ValueError("FROZEN_EXPERT_NONZERO_FROZEN_ADAM_MOMENTS")
    return actual


def validate_parameters(learner, inventory, expected_shared):
    from go2arx5_vla.sla_adapter.model_v1 import frozen_fingerprint

    if frozen_fingerprint(learner.frozen) != inventory["total_frozen_sha256"]:
        raise ValueError("FROZEN_EXPERT_ORIGINAL_FROZEN_VALUES_CHANGED")
    actual = frozen_fingerprint(learner.parameters["shared"])
    if actual != expected_shared:
        raise ValueError("FROZEN_EXPERT_R4_SHARED_VALUES_CHANGED")
    return actual


def save(learner, run, inventory, bound, expected_shared):
    import orbax.checkpoint as ocp
    from m38_gpu_preflight_v1 import sha
    from scripts.pi0292.durable import write
    from v3_main_place_learn import endpoint_files
    from v4_long_probe import snapshot
    from v7_formal_worker import allocated_bytes
    from v7_full_model_qualification import bytes_hash, projection

    # Check the new clock before writing: the old saver writes first and then
    # requires shared_count == step, which is false for a real frozen Expert.
    counts = validate_optimizer_clock(learner.optimizer, learner.step)
    shared = validate_parameters(learner, inventory, expected_shared)
    if shutil.disk_usage(run).free <= bound or np.asarray(projection(learner.parameters)).any():
        raise ValueError("FROZEN_EXPERT_SAVE_RESERVE_OR_NEUTRAL_PROJECTION_CHANGED")
    endpoint = run / f"STEP{learner.step:04d}"
    if endpoint.exists():
        raise FileExistsError("Preserve the previous full endpoint")
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(endpoint.resolve(),snapshot(learner))
    size = allocated_bytes(endpoint)
    if not 0 < size <= bound:
        raise ValueError("FROZEN_EXPERT_CHECKPOINT_SIZE_EXCEEDED")
    seal = dict(scope=SCOPE,step=learner.step,files=endpoint_files(endpoint),allocated_bytes=size,adam_counts=counts,
        frozen_weight_sha256=inventory["total_frozen_sha256"],lora_sha256=inventory["parent_lora_sha256"],
        frozen_shared_expert_sha256=shared,projection_sha256=bytes_hash(projection(learner.parameters)),
        bank_cursor=learner.step,active_optimizer_groups=sorted(ACTIVE))
    write(endpoint.with_name(endpoint.name+"_SEALED.json"),seal)
    return seal


def restore(learner, endpoint, seal, inventory, expected_shared):
    import jax
    import orbax.checkpoint as ocp
    from v3_main_place_learn import endpoint_files
    from v4_long_probe import snapshot

    endpoint = Path(endpoint)
    if (seal["scope"] != SCOPE or seal["frozen_shared_expert_sha256"] != expected_shared
            or seal["files"] != endpoint_files(endpoint) or seal["bank_cursor"] != seal["step"]
            or seal["adam_counts"] != {key:seal["step"] if key in ACTIVE else 0 for key in GROUPS}):
        raise ValueError("FROZEN_EXPERT_WRONG_OR_INCOMPLETE_RESUME_SOURCE")
    if learner.step != 0 or learner.queued or learner.cursors != dict.fromkeys(learner.batch_by_domain,0):
        raise ValueError("FROZEN_EXPERT_RESTORE_REQUIRES_UNCONSUMED_LEARNER")
    with ocp.PyTreeCheckpointer() as reader:
        saved = reader.restore(endpoint.resolve(),item=snapshot(learner))
    if int(saved["step"]) != seal["step"] or set(saved["parameters"]) != GROUPS or set(saved["optimizer"]) != GROUPS:
        raise ValueError("FROZEN_EXPERT_RESTORED_SCHEMA_OR_STEP_CHANGED")
    for key,value in saved["parameters"].items():
        learner.parameters[key].replace_by_pure_dict(value)
    learner.optimizer,learner.rng,learner.step = saved["optimizer"],saved["rng"],int(saved["step"])
    counts = validate_optimizer_clock(learner.optimizer,learner.step)
    validate_parameters(learner,inventory,expected_shared)
    actual = snapshot(learner)
    if (jax.tree.structure(actual) != jax.tree.structure(saved) or not all(
            np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(saved),strict=True))):
        raise ValueError("FROZEN_EXPERT_FULL_RESTORE_CHANGED_VALUES")
    return dict(step=learner.step,adam_counts=counts,full_snapshot_equal=True,
        next_lr=float(learner.rates[learner.step]),next_domain_cursors={key:learner.step*size for key,size in learner.batch_by_domain.items()})
