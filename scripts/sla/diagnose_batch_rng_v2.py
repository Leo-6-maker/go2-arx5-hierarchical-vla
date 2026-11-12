"""Bounded CPU diagnosis of the observed random-stream comparison failure."""
import json
import os
from pathlib import Path
import sys
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.batch_v2 import draws
from go2arx5_vla.sla_adapter.config_v1 import SEED


def run(spec):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise PermissionError('CPU_ONLY_RANDOM_STREAM_DIAGNOSIS')
    import jax
    import jax.numpy as jnp
    from openpi.training.utils import train_rng
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise PermissionError('CPU_ONLY_DIAGNOSIS')
    rng = jax.random.PRNGKey(SEED)

    def single(cursor):
        key = train_rng(rng, cursor // 32, cursor % 32, 32)
        _, noise_key, time_key = jax.random.split(key, 3)
        return key, jax.random.normal(noise_key, (1, 40, 32))[0], (jax.random.beta(time_key, 1.5, 1, (1,)) * .999 + .001)[0]

    compiled = jax.jit(single)
    reference = [compiled(cursor) for cursor in range(32)]
    reference = [np.stack([np.asarray(row[index]) for row in reference]) for index in range(3)]
    mapped = jax.jit(lambda cursors: jax.lax.map(single, cursors))
    vectorized = jax.jit(lambda cursors: (jax.vmap(lambda c: jax.random.fold_in(rng, c))(cursors), *draws(rng, cursors)))
    results = []
    for name, function in (('CURRENT_VMAP', vectorized), ('POINTWISE_LAX_MAP', mapped)):
        for micro in (1, 8, 16):
            pieces = [function(jnp.arange(start, start + micro, dtype=jnp.uint32)) for start in range(0, 32, micro)]
            actual = [np.concatenate([np.asarray(row[index]) for row in pieces]) for index in range(3)]
            result = dict(variant=name, microbatch=micro)
            for index, label in enumerate(('keys', 'noise', 'time')):
                a, b = reference[index], actual[index]
                result[label] = dict(bit_exact=bool(np.array_equal(a, b)), differing_elements=int(np.count_nonzero(a != b)),
                                     max_abs=float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
            results.append(result)
    return dict(schema='SLA_BATCH_RNG_CPU_FAILURE_DIAGNOSIS_V2', jax_version=jax.__version__,
                results=results, dataset_reads=0, SLA_model_constructions=0, SLA_forwards=0,
                SLA_backwards=0, optimizer_updates=0, optimization_visits=0,
                physical_requests=0, native_integrations=0, prior_failure_relabelled=False,
                hardware_qualification=False)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root']) / 'RESULT.private.json', run(spec))
