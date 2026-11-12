"""Finite TRAIN-only dense/zero-LoRA numerical boundary diagnostic."""
import json
from pathlib import Path
import sys
import time
import numpy as np

from go2arx5_vla.sla_adapter.runtime_v1 import Engine, probe_indices
from go2arx5_vla.sla_adapter.model_v1 import dense_model, flat, observation
from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error
from scripts.sla.train_v1 import exact_trees


def run(spec):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask
    engine = Engine(spec, Path(spec['registered_root'])/'work')
    index = probe_indices(engine.dataset)[0]
    item = engine.dataset[index]
    obs, _ = observation(item, engine.tokenizer)
    key = jax.random.PRNGKey(2026111200)
    noise = jax.random.normal(key, (1, 40, 32))
    base = dense_model(engine.active, engine.frozen)
    graph, params = nnx.split(base)
    parent = flat(engine.frozen.to_pure_dict()) | flat(engine.active.to_pure_dict())
    dense = flat(params.to_pure_dict())
    equal = all(np.array_equal(np.asarray(value), np.asarray(parent[path])) for path, value in dense.items())
    b_values = {('/'.join(p)): bool(np.all(np.asarray(v) == 0)) for p, v in flat(engine.active.to_pure_dict()).items() if p[-1].endswith('lora_b')}
    if not equal or not b_values or not all(b_values.values()):
        raise ValueError('DENSE_INHERITANCE_OR_ZERO_LORA_BRANCH_INVALID')
    dense_infer = jax.jit(lambda p, x, n: nnx.merge(graph, p).sample_actions(key, x, noise=n, num_steps=10))
    outputs = {}
    for name, infer, active, frozen in [('dense', dense_infer, params, None), ('lora', engine.infer, engine.active, engine.frozen)]:
        for repeat in range(2):
            value = infer(active, obs, noise) if frozen is None else infer(active, frozen, obs, noise)
            outputs[f'{name}{repeat}'] = np.asarray(value)
    first = exact_trees(outputs['dense0'], outputs['lora0'])
    state, *_ = engine.dataset.original.raw(index)
    identity = engine.dataset.original.identity(index)
    codec = engine.dataset.codecs[identity['episode_id']]
    for name in ('dense0', 'lora0'):
        a = engine.dataset.statistics['actions'].unnormalize(outputs[name][0, :, :7])
        outputs[name+'_physical_H40'] = codec.decode(state, pad32(a))
    np.savez(engine.output/'INITIAL_FUNCTION_OUTPUTS.private.npz', **outputs)
    dense_infer.clear_cache()

    def prefill(model, x):
        prefix, mask, ar = model.embed_prefix(x)
        attention = make_attn_mask(mask, ar)
        positions = jnp.cumsum(mask, axis=1)-1
        out, cache = model.PaliGemma.llm([prefix, None], mask=attention, positions=positions)
        return prefix, mask, out, cache

    dense_prefix = jax.jit(lambda p, x: prefill(nnx.merge(graph, p), x))
    lora_prefix = jax.jit(lambda a, f, x: prefill(nnx.merge(engine.graph, a, f), x))
    dp, lp = dense_prefix(params, obs), lora_prefix(engine.active, engine.frozen, obs)
    jax.block_until_ready(lp)
    prefix_comparison = [exact_trees(a, b) for a, b in zip(dp, lp)]
    captured = {}
    for name, values in [('dense', dp), ('lora', lp)]:
        leaves, _ = jax.tree.flatten(values)
        for i, value in enumerate(leaves):
            captured[name+f'_{i:02d}'] = np.asarray(value).astype(np.float32)
    np.savez_compressed(engine.output/'PREFILL_BOUNDARIES.private.npz', **captured)
    result = dict(schema='SLA_ZERO_LORA_INITIAL_FUNCTION_DIAGNOSTIC_V1', index=index,
                  shared_dense_parameters_bit_exact=equal, zero_B_leaves=b_values,
                  dense_repeat=exact_trees(outputs['dense0'], outputs['dense1']),
                  lora_repeat=exact_trees(outputs['lora0'], outputs['lora1']), dense_vs_lora=first,
                  active7_max_normalized_abs=float(np.abs(outputs['dense0'][..., :7]-outputs['lora0'][..., :7]).max()),
                  padding25_max_normalized_abs=float(np.abs(outputs['dense0'][..., 7:]-outputs['lora0'][..., 7:]).max()),
                  physical_difference=roundtrip_error(outputs['dense0_physical_H40'], outputs['lora0_physical_H40']),
                  independent_prefill_comparison=dict(zip(('prefix_input', 'mask', 'VLM_output', 'KV_cache'), prefix_comparison)),
                  causal_internal_kernel_identified=False, numerical_tolerance_changed=False,
                  actual_parent_loads=1, actual_free_generations=4, diagnostic_prefill_calls=2,
                  actual_optimizer_updates=0, optimization_visits=0, physical_requests=0,
                  frozen_unchanged=engine.verify_frozen(), GPU_memory=jax.devices()[0].memory_stats(),
                  qualified_for_training=False, epoch=time.time())
    return result


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    result = run(spec)
    (Path(spec['registered_root'])/'RESULT.private.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
