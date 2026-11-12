"""Actual CPU data/RNG/reduction seams only; never construct an SLA model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.batch_v2 import RNG_VERSION, draws, prepared_batches
from go2arx5_vla.sla_adapter.config_v1 import SEED, visits
from go2arx5_vla.sla_adapter.dataset_v2 import data
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer


def flattened_arrays(value, prefix=''):
    for name in sorted(value):
        if isinstance(value[name], dict):
            yield from flattened_arrays(value[name], prefix + name + '/')
        else:
            yield prefix + name, np.asarray(value[name])


def concatenate(rows):
    result = {}
    for key in rows[0]:
        if isinstance(rows[0][key], dict):
            result[key] = concatenate([row[key] for row in rows])
        else:
            result[key] = np.concatenate([row[key] for row in rows], axis=0)
    return result


def run(spec):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise PermissionError('EXPLICIT_CPU_ONLY_BATCH_INPUT_AUDIT')
    import jax
    import jax.numpy as jnp
    from openpi.training.utils import train_rng
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise PermissionError('NO_GPU_BATCH_INPUT_AUDIT')
    dataset = data(spec)
    tokenizer = Tokenizer(spec['tokenizer']['path'])
    visit_map = visits(dataset.original.manifest)
    rng = jax.random.PRNGKey(SEED)

    @jax.jit
    def original_draw(cursor):
        key = train_rng(rng, cursor // 32, cursor % 32, 32)
        _, noise_key, time_key = jax.random.split(key, 3)
        return (jax.random.normal(noise_key, (1, 40, 32))[0],
                (jax.random.beta(time_key, 1.5, 1, (1,)) * .999 + .001)[0])

    reference_noise, reference_time = [], []
    for cursor in range(32):
        noise, flow_time = original_draw(cursor)
        reference_noise.append(np.asarray(noise)); reference_time.append(np.asarray(flow_time))
    reference_noise, reference_time = np.stack(reference_noise), np.asarray(reference_time)

    @jax.jit
    def reduction(theta, actions, noise, flow_time):
        expanded = flow_time[..., None, None]
        x_t = expanded * noise + (1 - expanded) * actions
        return jnp.mean(jnp.mean(jnp.square(theta[0] * x_t + theta[1] - (noise - actions)), axis=-1))

    synthetic_gradient = jax.jit(jax.value_and_grad(reduction))
    batched_draw = jax.jit(draws)
    theta = jnp.array([.125, -.05], dtype=jnp.float32)
    reference_inputs = None
    reference_loss = reference_gradient = None
    checks, reads = [], dict(host_window_read_attempts=0, host_window_reads=0)
    def count(key):
        reads[key] += 1
    synthetic_evaluations = 0
    for micro in (1, 8, 16):
        if time.time() >= spec['deadline_epoch']:
            raise TimeoutError('FINITE_CPU_BATCH_SEAM_AUDIT')
        rows = list(prepared_batches(dataset, tokenizer, visit_map, 0, micro,
                                     prefetch=True, on_read=count))
        inputs = concatenate([dict(observation=row['observation'], actions=row['actions'],
                                   cursors=row['cursors'], drops=np.asarray(row['state_token_dropped'])) for row in rows])
        if reference_inputs is None:
            reference_inputs = inputs
        else:
            left, right = dict(flattened_arrays(reference_inputs)), dict(flattened_arrays(inputs))
            if left.keys() != right.keys() or any(not np.array_equal(left[key], right[key]) for key in left):
                raise ValueError('MICROBATCH_CHANGED_TOKEN_PIXEL_ACTION_ORDER_OR_DROPOUT')
        noises, times = [], []
        loss_sum, gradient_sum = 0., np.zeros(2, np.float64)
        for row in rows:
            noise, flow_time = batched_draw(rng, row['cursors'])
            noises.append(np.asarray(noise)); times.append(np.asarray(flow_time))
            value, gradient = synthetic_gradient(theta, jnp.asarray(row['actions']), noise, flow_time)
            loss_sum += float(value) / len(rows)
            gradient_sum += np.asarray(gradient, np.float64) / len(rows)
            synthetic_evaluations += 1
        noise, flow_time = np.concatenate(noises), np.concatenate(times)
        if not np.array_equal(noise, reference_noise) or not np.array_equal(flow_time, reference_time):
            raise ValueError('MICROBATCH_RANDOM_DRAWS_DIFFER_FROM_ORIGINAL_SAMPLE_STREAM')
        if reference_loss is None:
            reference_loss, reference_gradient = loss_sum, gradient_sum
        delta = gradient_sum - reference_gradient
        denom = np.linalg.norm(reference_gradient)
        cosine = float(np.dot(gradient_sum, reference_gradient) / (np.linalg.norm(gradient_sum) * denom))
        relative = float(np.linalg.norm(delta) / denom)
        if not np.isclose(loss_sum, reference_loss, rtol=1e-3, atol=1e-5) or cosine < .999 or relative > .02:
            raise ValueError('CPU_SYNTHETIC_EFFECTIVE32_REDUCTION_CHANGED')
        checks.append(dict(microbatch=micro, accumulation_steps=len(rows),
                           model_inputs_bit_exact=True, random_noise_bit_exact=True, random_time_bit_exact=True,
                           synthetic_loss=loss_sum, synthetic_gradient_cosine=cosine,
                           synthetic_gradient_relative_L2=relative))
    fingerprints = {name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in flattened_arrays(reference_inputs)}
    return dict(schema='SLA_BATCH_INPUT_RNG_CPU_SEAMS_ACTUAL_V2', passes=True,
                plan_revision=spec['scaleup_plan_revision'], action_representation_audit=spec['action_representation_audit'],
                random_stream=RNG_VERSION, microbatches=[1, 8, 16], effective_batch=32,
                input_fingerprints=fingerprints, checks=checks, reader_counts=reads,
                unique_TRAIN_windows_read=len(set(visit_map[:32].tolist())),
                synthetic_reduction_gradient_evaluations=synthetic_evaluations,
                SLA_model_constructions=0, SLA_forwards=0, SLA_backwards=0,
                optimizer_updates=0, optimization_visits=0, ordinary_RGB_policy_generations=0,
                teacher_calls=0, physical_requests=0, native_integrations=0, DEV_TEST_reads=0,
                devices=[str(device) for device in jax.devices()], hardware_qualification_pass=False,
                limitation='CPU_INPUT_RNG_AND_TWO_PARAMETER_REDUCTION_ONLY_NOT_SLA_MODEL_GRADIENTS_OR_GPU_THROUGHPUT')


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root']) / 'RESULT.private.json', run(spec))
