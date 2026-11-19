"""Strict upstream JAX model loading with an explicit target PEFT allowlist."""
import gc
import hashlib
import math
import json
import resource
from pathlib import Path
import numpy as np

from .config_v1 import SEED, trainable


def configuration(*, lora=True, action_horizon=40):
    from openpi.models.pi0_config import Pi0Config
    if action_horizon not in (40, 50):
        raise ValueError('EXPLICIT_TARGET_H40_OR_NATIVE_SOURCE_H50_REQUIRED')
    return Pi0Config(pi05=True, dtype='bfloat16', action_dim=32, action_horizon=action_horizon,
                     paligemma_variant='gemma_2b_lora' if lora else 'gemma_2b',
                     action_expert_variant='gemma_300m_lora' if lora else 'gemma_300m',
                     max_token_len=200)


def flat(tree):
    from flax.traverse_util import flatten_dict
    return flatten_dict(tree)


def frozen_fingerprint(state):
    """Stream audit copies without caching a full CPU mirror on each JAX array.

    In JAX0.5.3 np.asarray(GPU_array) retains the transferred NumPy value on
    the original array. Hash only temporary slices, so the training parent
    does not retain ~6.7GB of additional host audit mirrors for its lifetime.
    This changes neither parameter storage on GPU nor the fingerprint bytes.
    """
    h = hashlib.sha256()
    for path, value in sorted(flat(state.to_pure_dict()).items()):
        h.update(('/'.join(path)+':'+str(value.shape)+':'+str(value.dtype)+'\n').encode())
        _hash_parameter(h, value)
    return h.hexdigest()


def _hash_parameter(digest, value, *, max_bytes=8 * 1024**2):
    if value.size == 0:
        return
    if value.ndim == 0:
        digest.update(np.asarray(value).tobytes())
        return
    item_bytes = np.dtype(value.dtype).itemsize
    trailing_bytes = value.size * item_bytes
    for axis, length in enumerate(value.shape):
        trailing_bytes //= length
        if trailing_bytes <= max_bytes:
            break
    count = max(1, max_bytes // trailing_bytes)
    # Slice before materializing. Flattening a transposed/tiled JAX parameter
    # can allocate a second full GPU buffer even when only a small slice is read.
    # Retain C-order bytes so every existing frozen fingerprint stays valid.
    for prefix in np.ndindex(value.shape[:axis]):
        for start in range(0, length, count):
            chunk = value[prefix + (slice(start, start + count),)]
            digest.update(np.asarray(chunk).tobytes())
            del chunk


def placement_dtype_name(path):
    """Same final precision as the original loader, before materialization."""
    parts = tuple(path)
    if parts and parts[0] == 'params':
        parts = parts[1:]
    if parts and parts[-1] == 'value':
        parts = parts[:-1]
    return 'float32' if trainable(parts) else 'bfloat16'


def _restore_for_placement(params_path, *, dtype_for_path=placement_dtype_name):
    """Pinned upstream restore structure with bounded, final-dtype host reads.

    Keep every trainable interface in FP32 and frozen leaves in BF16, as the
    original loader does after restoring. Avoid a full FP32 host mirror and
    bound concurrent checkpoint reads to1GB. No checkpoint bytes are changed.
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict, unflatten_dict
    handler = ocp.PyTreeCheckpointHandler(restore_concurrent_gb=1)
    with ocp.Checkpointer(handler) as checkpointer:
        metadata = checkpointer.metadata(Path(params_path).resolve())
        item = {'params': metadata['params']}
        def argument(path, _):
            dtype = jnp.float32 if dtype_for_path(tuple(k.key for k in path)) == 'float32' else jnp.bfloat16
            return ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=dtype)
        restored = checkpointer.restore(Path(params_path).resolve(), ocp.args.PyTreeRestore(
            item=item, restore_args=jax.tree_util.tree_map_with_path(argument,item)))['params']
    values = flatten_dict(restored)
    if all(p[-1] == 'value' for p in values):
        values = {p[:-1]: v for p,v in values.items()}
    return unflatten_dict(values)


def load_parent(params_path, *, host_cast=False, action_horizon=40):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    from openpi.models.model import restore_params

    def phase(name):
        print(json.dumps(dict(event='SLA_PARENT_LOAD_PHASE', phase=name,
                              host_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)

    phase('ABSTRACT_MODEL')
    config = configuration(action_horizon=action_horizon)
    abstract = nnx.eval_shape(config.create, jax.random.PRNGKey(SEED))
    graph, reference = nnx.split(abstract)
    expected = flat(reference.to_pure_dict())
    phase('ORBAX_HOST_RESTORE_BEGIN')
    loaded = flat(_restore_for_placement(params_path) if host_cast else
                  restore_params(Path(params_path), restore_type=np.ndarray))
    phase('ORBAX_HOST_RESTORE_COMPLETE')
    if host_cast:
        gc.collect()
    missing = sorted(set(expected)-set(loaded))
    extra = sorted(set(loaded)-set(expected))
    if any('lora' not in '/'.join(p) for p in missing):
        raise ValueError('PRETRAINED_DENSE_PARAMETER_MISSING:'+str(missing))
    if any(p[0] != 'align_projector' for p in extra):
        raise ValueError('UNEXPECTED_PRETRAINED_PARAMETER:'+str(extra))
    parameters, inventory = {}, []
    for path, shape in sorted(expected.items()):
        selected = trainable(path)
        if path in loaded:
            value = loaded.pop(path)
            if tuple(value.shape) != tuple(shape.shape):
                raise ValueError('PRETRAINED_PARAMETER_SHAPE_MISMATCH:'+str(path))
            # Preserve all pretrained interfaces. Only frozen dense weights are
            # cast to BF16, as specified; trainable interfaces retain FP32.
            dtype = jnp.float32 if selected else jnp.bfloat16
            if host_cast:
                # Bound transfer staging: send the final dtype instead of a
                # full FP32 device input followed by a BF16 conversion kernel.
                # S16 verifies the original full frozen fingerprint and all87
                # initial model/Adam/RNG arrays before its first update.
                host = np.asarray(value, dtype=dtype)
                del value
                value = jax.device_put(host)
                jax.block_until_ready(value)
                del host
            else:
                value = jnp.asarray(value, dtype=dtype)
        else:
            if not selected:
                raise ValueError('NEW_PARAMETER_OUTSIDE_POSITIVE_ALLOWLIST')
            # The pinned upstream initializes BOTH factors normally. Explicitly
            # zero B to make each new branch initially function preserving.
            key_id = int.from_bytes(hashlib.sha256('/'.join(path).encode()).digest()[:4], 'little')
            key = jax.random.fold_in(jax.random.PRNGKey(SEED), key_id)
            value = (jnp.zeros(shape.shape, jnp.float32) if path[-1].endswith('lora_b') else
                     jax.random.normal(key, shape.shape, dtype=jnp.float32)*.01)
        # Finish each transfer before dropping its host source, bounding
        # pending host staging copies without offloading training activations.
        jax.block_until_ready(value)
        parameters[path] = value
        inventory.append(dict(path='/'.join(path), shape=list(value.shape), numel=math.prod(value.shape),
                              dtype=str(value.dtype), trainable=selected, devices=sorted(str(d) for d in value.devices())))
    del loaded
    gc.collect()
    phase('GPU_PARAMETERS_COMPLETE')
    reference.replace_by_pure_dict(unflatten_dict(parameters))
    model = nnx.merge(graph, reference)
    graph, active, frozen = nnx.split(model, nnx.All(nnx.Param, lambda path, _: trainable(path)), ...)
    phase('STREAMED_FROZEN_FINGERPRINT_BEGIN')
    fingerprint = frozen_fingerprint(frozen)
    phase('STREAMED_FROZEN_FINGERPRINT_COMPLETE')
    report = dict(missing_keys=['/'.join(p) for p in missing],
                  unexpected_keys=['/'.join(p) for p in extra],
                  removed_training_only_keys=['/'.join(p) for p in extra],
                  new_parameters_only_lora=True, lora_B_zero_initialization=True,
                  pretrained_interfaces_randomly_reset=False, parameters=inventory,
                  total_numel=sum(v['numel'] for v in inventory),
                  trainable_numel=sum(v['numel'] for v in inventory if v['trainable']),
                  frozen_fingerprint=fingerprint)
    return graph, active, frozen, report


def dense_model(active, frozen):
    """Share the loaded dense buffers for the pre/post-LoRA function check."""
    from flax.traverse_util import unflatten_dict
    values = flat(frozen.to_pure_dict()) | flat(active.to_pure_dict())
    values = {p: v for p, v in values.items() if not any('lora' in str(part) for part in p)}
    return configuration(lora=False).load(unflatten_dict(values), remove_extra_params=False)


class Tokenizer:
    """Pinned original tokenizer behavior with token-segment-only dropout."""
    def __init__(self, path):
        from openpi.models.tokenizer import PaligemmaTokenizer
        import sentencepiece
        self.delegate = object.__new__(PaligemmaTokenizer)
        self.delegate._max_len = 200
        self.delegate._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))

    def __call__(self, prompt, normalized_state32, *, drop_state=False):
        # Pinned ModelTransformFactory tokenizes BEFORE PadStatesAndActions.
        # Padding25 is a tensor container, not 25 extra physical state tokens.
        state = np.asarray(normalized_state32)
        if state.shape != (32,):
            raise ValueError('PADDED_32_STATE_CONTAINER_REQUIRED')
        return self.delegate.tokenize(prompt, None if drop_state else state[:7],
                                      use_discrete_state_format=True)


def observation(item, tokenizer, *, drop_state=False):
    import jax
    from openpi.models.model import Observation
    tokens, mask = tokenizer(item['prompt'], item['state'], drop_state=drop_state)
    # The actual normalized state and codec anchor remain intact when the
    # discrete token segment is omitted. Dropout never changes image masks.
    values = dict(image={k: v[None] for k, v in item['image'].items()},
                  image_mask={k: np.array([v]) for k, v in item['image_mask'].items()},
                  image_padding_mask={k: v[None] for k, v in item['image_padding_mask'].items()},
                  state=item['state'][None], tokenized_prompt=tokens[None], tokenized_prompt_mask=mask[None])
    return jax.device_put(Observation.from_dict(values)), jax.device_put(item['actions'][None])
