"""Strict upstream JAX model loading with an explicit target PEFT allowlist."""
import gc
import hashlib
import math
from pathlib import Path
import numpy as np

from .config_v1 import SEED, trainable


def configuration(*, lora=True):
    from openpi.models.pi0_config import Pi0Config
    return Pi0Config(pi05=True, dtype='bfloat16', action_dim=32, action_horizon=40,
                     paligemma_variant='gemma_2b_lora' if lora else 'gemma_2b',
                     action_expert_variant='gemma_300m_lora' if lora else 'gemma_300m',
                     max_token_len=200)


def flat(tree):
    from flax.traverse_util import flatten_dict
    return flatten_dict(tree)


def frozen_fingerprint(state):
    """Call at qualification boundaries, outside the update critical path."""
    h = hashlib.sha256()
    for path, value in sorted(flat(state.to_pure_dict()).items()):
        a = np.asarray(value)
        h.update(('/'.join(path)+':'+str(a.shape)+':'+str(a.dtype)+'\n').encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_parent(params_path):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    from openpi.models.model import restore_params

    config = configuration()
    abstract = nnx.eval_shape(config.create, jax.random.PRNGKey(SEED))
    graph, reference = nnx.split(abstract)
    expected = flat(reference.to_pure_dict())
    loaded = flat(restore_params(Path(params_path), restore_type=np.ndarray))
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
            value = jnp.asarray(value, dtype=jnp.float32 if selected else jnp.bfloat16)
        else:
            if not selected:
                raise ValueError('NEW_PARAMETER_OUTSIDE_POSITIVE_ALLOWLIST')
            # The pinned upstream initializes BOTH factors normally. Explicitly
            # zero B to make each new branch initially function preserving.
            key_id = int.from_bytes(hashlib.sha256('/'.join(path).encode()).digest()[:4], 'little')
            key = jax.random.fold_in(jax.random.PRNGKey(SEED), key_id)
            value = (jnp.zeros(shape.shape, jnp.float32) if path[-1].endswith('lora_b') else
                     jax.random.normal(key, shape.shape, dtype=jnp.float32)*.01)
        parameters[path] = value
        inventory.append(dict(path='/'.join(path), shape=list(value.shape), numel=math.prod(value.shape),
                              dtype=str(value.dtype), trainable=selected, devices=sorted(str(d) for d in value.devices())))
    del loaded
    gc.collect()
    reference.replace_by_pure_dict(unflatten_dict(parameters))
    model = nnx.merge(graph, reference)
    graph, active, frozen = nnx.split(model, nnx.All(nnx.Param, lambda path, _: trainable(path)), ...)
    report = dict(missing_keys=['/'.join(p) for p in missing],
                  unexpected_keys=['/'.join(p) for p in extra],
                  removed_training_only_keys=['/'.join(p) for p in extra],
                  new_parameters_only_lora=True, lora_B_zero_initialization=True,
                  pretrained_interfaces_randomly_reset=False, parameters=inventory,
                  total_numel=sum(v['numel'] for v in inventory),
                  trainable_numel=sum(v['numel'] for v in inventory if v['trainable']),
                  frozen_fingerprint=frozen_fingerprint(frozen))
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
        return self.delegate.tokenize(prompt, None if drop_state else normalized_state32,
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
