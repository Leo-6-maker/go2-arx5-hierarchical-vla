"""Persist and restore the R5 LoRA/AdamW state against immutable run inputs."""

import hashlib
import json
from pathlib import Path

from flax.traverse_util import flatten_dict
import jax
import numpy as np
import orbax.checkpoint as ocp


COUNTERS = {"[1][0].count", "[1][2].count"}


def binding_digest(binding):
    return hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def payload_digest(state):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError("invalid checkpoint tensor: " + jax.tree_util.keystr(path))
        identity = (jax.tree_util.keystr(path), str(array.dtype), array.shape)
        digest.update(json.dumps(identity, separators=(",", ":")).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def validate_state(state, adapter_layout, cap):
    step = np.asarray(state["completed_steps"])
    if step.shape != () or step.dtype != np.int32 or not 0 <= int(step) <= cap:
        raise ValueError("optimizer step outside the finite run")
    completed = int(step)

    def layout(arrays):
        return {
            "/".join(key): {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
            for key, leaf in flatten_dict(arrays).items()
        }

    if layout(state["adapters"]) != adapter_layout:
        raise ValueError("LoRA checkpoint does not match the ten-leaf layout")
    counters = {
        jax.tree_util.keystr(key): np.asarray(leaf)
        for key, leaf in jax.tree_util.tree_flatten_with_path(state["adamw"])[0]
        if jax.tree_util.keystr(key) in COUNTERS
    }
    if counters.keys() != COUNTERS or any(
        value.shape != () or value.dtype != np.int32 or int(value) != completed
        for value in counters.values()
    ):
        raise ValueError("AdamW and schedule counters must equal the completed step")
    adam = state["adamw"][1][0]
    if layout(adam.mu.to_pure_dict()) != adapter_layout or layout(adam.nu.to_pure_dict()) != adapter_layout:
        raise ValueError("AdamW moments do not match the live Flax LoRA state")
    return completed


def save(root, *, completed, adapters, optimizer, binding, adapter_layout):
    if type(completed) is not int or not 0 <= completed <= binding["max_steps"] <= 1000:
        raise ValueError("invalid finite checkpoint step")
    state = dict(completed_steps=np.int32(completed), adapters=adapters, adamw=optimizer)
    state = jax.tree.map(np.asarray, state)
    step = validate_state(state, adapter_layout, binding["max_steps"])
    destination = Path(root).resolve() / f"step-{step:05d}"
    metadata = dict(binding_sha256=binding_digest(binding), state_sha256=payload_digest(state))
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(destination, args=ocp.args.PyTreeSave(state), custom_metadata=metadata)
    return destination


def restore(root, *, completed, template, binding, adapter_layout):
    if type(completed) is not int or not 0 <= completed <= binding["max_steps"] <= 1000:
        raise ValueError("invalid finite restore step")
    source = Path(root).resolve() / f"step-{completed:05d}"
    if not source.is_dir() or not ocp.utils.is_checkpoint_finalized(source):
        raise FileNotFoundError("no committed checkpoint at the requested step")
    metadata = ocp.metadata.get_step_metadata(source)
    if not metadata.commit_timestamp_nsecs:
        raise FileNotFoundError("no committed checkpoint at the requested step")
    pinned = metadata.custom_metadata
    if not pinned or pinned.get("binding_sha256") != binding_digest(binding):
        raise ValueError("checkpoint run identity changed")
    with ocp.PyTreeCheckpointer() as reader:
        restored = reader.restore(source, args=ocp.args.PyTreeRestore(item=template))
    if (validate_state(restored, adapter_layout, binding["max_steps"]) != completed
            or payload_digest(restored) != pinned["state_sha256"]):
        raise ValueError("checkpoint step or array bytes changed")
    return restored
