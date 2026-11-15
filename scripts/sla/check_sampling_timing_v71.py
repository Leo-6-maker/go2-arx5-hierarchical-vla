"""Check the timing boundary with the installed Observation/NNX types on CPU.

Uses synthetic RGB and a tiny arithmetic function, never SLA weights or model
generation. An optional archived helper reproduces the original lower error.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load_timer(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SamplingTimer


def run(helper, tokenizer_path, old_helper=None):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import model_v1
    from go2arx5_vla.sla_adapter.dataset_v1 import images224
    from openpi_client.image_tools import resize_with_pad

    assert jax.__version__ == '0.5.3'
    assert all(device.platform == 'cpu' for device in jax.devices())
    started = time.perf_counter()
    tokenizer = model_v1.Tokenizer(tokenizer_path)
    item = images224([np.zeros((256, 256, 3), np.uint8),
                      np.full((256, 256, 3), 127, np.uint8)], resize_with_pad)
    item.update(state=np.linspace(-.2, .2, 32, dtype=np.float32),
                actions=np.zeros((40, 32), np.float32), prompt='pick up the object')

    class Arithmetic(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.asarray(1., jnp.float32))
            self.constant = nnx.Variable(jnp.asarray(3., jnp.float32))

    graph, active, frozen = nnx.split(Arithmetic(), nnx.Param, ...)

    def arithmetic(active, frozen, observation, noise):
        model = nnx.merge(graph, active, frozen)
        scalar = observation.state.sum() + observation.images['base_0_rgb'][0, 0, 0, 0]
        scalar += observation.tokenized_prompt[0, 0].astype(jnp.float32)
        return noise + scalar + model.weight.value + model.constant.value

    function = jax.jit(arithmetic)
    original_observation, original_put = model_v1.observation, jax.device_put
    obs, _ = model_v1.observation(item, tokenizer)
    assert obs.state.shape == (1, 32) and obs.tokenized_prompt.shape == (1, 200)
    assert all(x.shape == (1, 224, 224, 3) for x in obs.images.values())

    def reject_invalid_observation():
        try:
            dataclasses.replace(obs, images={k: jnp.zeros(v.shape, jnp.int32)
                                            for k, v in obs.images.items()})
        except Exception as error:
            assert type(error).__name__ == 'TypeCheckError', type(error).__name__
        else:
            raise AssertionError('ORIGINAL_OBSERVATION_TYPE_CHECK_MUST_REMAIN_ENABLED')

    reject_invalid_observation()
    old_error = None
    noise = jax.random.normal(jax.random.PRNGKey(2026111200), (1, 40, 32))
    if old_helper is not None:
        previous = load_timer(old_helper, 'archived_sampling_timer')(function,
            jax_module=jax, model_module=model_v1)
        try:
            with previous.prediction():
                observed, _ = model_v1.observation(item, tokenizer)
                previous(active, frozen, observed, noise)
        except Exception as error:
            assert type(error).__name__ == 'TypeCheckError' and 'ArgInfo' in str(error)
            old_error = type(error).__name__
        else:
            raise AssertionError('ARCHIVED_TYPED_OBSERVATION_LOWER_FAILURE_NOT_REPRODUCED')
        assert previous.submitted == previous.completed == previous.compilations == 0
        assert model_v1.observation is original_observation and jax.device_put is original_put

    timer = load_timer(helper, 'repaired_sampling_timer')(function,
        jax_module=jax, model_module=model_v1)
    records = []
    for weight in (1., 2.):
        active['weight'].value = jnp.asarray(weight, jnp.float32)
        baseline = np.asarray(jax.block_until_ready(function(active, frozen, obs, noise)))
        with timer.prediction() as timing:
            observed, _ = model_v1.observation(item, tokenizer)
            actual = timer(active, frozen, observed, noise)
        assert np.array_equal(baseline, actual)
        assert timing['input_H2D_calls'] == 2 and timing['output_D2H_bytes'] == 5120
        assert model_v1.observation is original_observation and jax.device_put is original_put
        records.append(dict(timing))

    # A different optional field changes the pytree; never silently reuse an
    # executable against reordered leaves or recompile a new model signature.
    changed = dataclasses.replace(obs, token_ar_mask=jnp.zeros((1, 200), jnp.int32))
    try:
        with timer.prediction():
            model_v1.observation(item, tokenizer)
            timer(active, frozen, changed, noise)
    except ValueError as error:
        assert str(error) == 'COMPILED_SAMPLING_INPUT_STRUCTURE_CHANGED'
    else:
        raise AssertionError('CHANGED_OBSERVATION_STRUCTURE_WAS_NOT_REJECTED')
    assert timer.submitted == timer.completed == 2 and timer.compilations == 1
    assert model_v1.observation is original_observation and jax.device_put is original_put
    reject_invalid_observation()
    return dict(schema='SLA_SAMPLING_TYPED_OBSERVATION_CPU_INTERFACE_CHECK_V71',
        passed=True, helper_sha256=hashlib.sha256(Path(helper).read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        archived_error=old_error, actual_Observation_type=type(obs).__module__+'.'+type(obs).__name__,
        actual_NNX_parameter_trees=True, synthetic_RGB=True, bit_exact_arithmetic_pairs=2,
        toy_device_executions=4, timer_executions=2, timer_compilations=1,
        changed_structure_rejected=True, original_type_checks_preserved=True,
        wrappers_restored=True, records=records, wall_s=time.perf_counter()-started,
        SLA_parent_loads=0, SLA_generations=0, optimizer_updates=0, training_visits=0,
        physical_requests=0, physics_integrations=0, GPU_SLA_qualification=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--old-helper', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.helper, args.tokenizer, args.old_helper)), flush=True)
