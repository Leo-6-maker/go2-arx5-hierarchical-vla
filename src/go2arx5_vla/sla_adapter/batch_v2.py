"""Bounded CPU input prefetch and partition-independent sample randomness."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

from .config_v1 import SEED
from .dataset_v1 import CAMERAS

EFFECTIVE_BATCH = 32
RNG_VERSION = 'SLA_FIXED_EFFECTIVE32_DRAW_BANK_ABSOLUTE_VISIT_V2_1'


def accumulation_steps(microbatch):
    if type(microbatch) is not int or microbatch < 1 or EFFECTIVE_BATCH % microbatch:
        raise ValueError('ONE_DEVICE_MICROBATCH_MUST_DIVIDE_EFFECTIVE32')
    return EFFECTIVE_BATCH // microbatch


def drop_state(cursor):
    if type(cursor) is not int or not 0 <= cursor < 2**32:
        raise ValueError('FINITE_ABSOLUTE_VISIT_CURSOR_REQUIRED')
    return bool(np.random.Generator(np.random.PCG64(SEED + 1000000 + cursor)).random() < .25)


def draws(rng, cursors):
    """Match V1's fixed32 train_rng and its per-sample shape(1,40,32)."""
    import jax
    import jax.numpy as jnp
    keys = jax.vmap(lambda cursor: jax.random.fold_in(rng, cursor))(
        jnp.asarray(cursors, dtype=jnp.uint32))
    parts = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
    noise = jax.vmap(lambda key: jax.random.normal(key, (1, 40, 32)))(parts[:, 1])[:, 0]
    flow_time = jax.vmap(lambda key: jax.random.beta(key, 1.5, 1, (1,)))(parts[:, 2])[:, 0]
    return noise, flow_time * .999 + .001


def draw_bank(rng, cursors):
    """One fixed-shape GPU input bank, produced outside the model gradient.

    The CPU diagnosis found a1-ULP beta/time difference under differently
    shaped compilations. All V2 microbatch variants now slice the same32-row
    producer graph. This does not assert bit identity to V1's scalar graph.
    """
    if cursors.shape != (EFFECTIVE_BATCH,):
        raise ValueError('FIXED_EFFECTIVE32_RANDOM_INPUT_BANK_REQUIRED')
    return draws(rng, cursors)


def host_batch(dataset, tokenizer, visit_map, start, microbatch, *, on_read=None):
    accumulation_steps(microbatch)
    started = time.perf_counter()
    rows, tokens, masks, drops = [], [], [], []
    indices = [int(value) for value in visit_map[start:start + microbatch]]
    if len(indices) != microbatch:
        raise ValueError('NO_PARTIAL_EFFECTIVE_BATCH_OR_VISIT_OVERFLOW')
    for offset, index in enumerate(indices):
        if on_read:
            on_read('host_window_read_attempts')
        item = dataset[index]
        if on_read:
            on_read('host_window_reads')
        drop = drop_state(start + offset)
        token, mask = tokenizer(item['prompt'], item['state'], drop_state=drop)
        rows.append(item); tokens.append(token); masks.append(mask); drops.append(drop)
    observation = dict(image={name: np.stack([item['image'][name] for item in rows]) for name in CAMERAS},
                       image_mask={name: np.asarray([item['image_mask'][name] for item in rows], dtype=bool)
                                   for name in CAMERAS},
                       image_padding_mask={name: np.stack([item['image_padding_mask'][name] for item in rows])
                                           for name in CAMERAS},
                       state=np.stack([item['state'] for item in rows]),
                       tokenized_prompt=np.stack(tokens), tokenized_prompt_mask=np.stack(masks))
    actions = np.stack([item['actions'] for item in rows])
    if observation['state'].shape != (microbatch, 32) or actions.shape != (microbatch, 40, 32):
        raise ValueError('ORIGINAL_PADDED_STATE32_AND_ACTION_H40_32_REQUIRED')
    if any(observation['image'][name].shape != (microbatch, 224, 224, 3) for name in CAMERAS):
        raise ValueError('ORIGINAL_RGB224_BATCH_REQUIRED')
    return dict(observation=observation, actions=actions, indices=indices,
                cursors=np.arange(start, start + microbatch, dtype=np.uint32),
                state_token_dropped=drops, prepare_wall_s=time.perf_counter() - started)


def prepared_batches(dataset, tokenizer, visit_map, cursor, microbatch, *, prefetch=True, on_read=None):
    count = accumulation_steps(microbatch)
    if cursor % EFFECTIVE_BATCH or cursor + EFFECTIVE_BATCH > len(visit_map):
        raise ValueError('COMPLETE_EFFECTIVE_BATCH_BOUNDARY_REQUIRED')
    starts = [cursor + index * microbatch for index in range(count)]
    if not prefetch:
        for start in starts:
            began = time.perf_counter()
            batch = host_batch(dataset, tokenizer, visit_map, start, microbatch, on_read=on_read)
            yield dict(batch, host_wait_s=time.perf_counter() - began)
        return
    # Only the next microbatch is prepared. Joining at this update boundary
    # prevents the mutable source-reader cache racing with endpoint probes.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(host_batch, dataset, tokenizer, visit_map, starts[0], microbatch, on_read=on_read)
        for index in range(count):
            began = time.perf_counter()
            batch = future.result()
            wait = time.perf_counter() - began
            if index + 1 < count:
                future = pool.submit(host_batch, dataset, tokenizer, visit_map, starts[index + 1], microbatch, on_read=on_read)
            yield dict(batch, host_wait_s=wait)


def device_batch(batch):
    import jax
    from openpi.models.model import Observation
    return (jax.device_put(Observation.from_dict(batch['observation'])),
            jax.device_put(batch['actions']))


def error32(model, observation, actions, noise, flow_time):
    """Pinned upstream FM network with explicit independent sample draws.

    Adapted from Spatial-Forcing405ca887/openpi/models/pi0.py, Apache-2.0.
    The entire microbatch traverses the network together; only random draws
    are vmapped. No VLM feature cache or CPU activation offload is introduced.
    """
    import jax.numpy as jnp
    from openpi.models import model as upstream
    from openpi.models.pi0 import make_attn_mask
    observation = upstream.preprocess_observation(None, observation, train=False)
    expanded = flow_time[..., None, None]
    x_t, target = expanded * noise + (1 - expanded) * actions, noise - actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(observation, x_t, flow_time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention = make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(mask, axis=1) - 1
    (_, output), _ = model.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions,
                                       adarms_cond=[None, condition])
    predicted = model.action_out_proj(output[:, -model.action_horizon:])
    return jnp.square(predicted - target)
