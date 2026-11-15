"""Device-completed timing around an unchanged ordinary-RGB provider.

Only this isolated, serial diagnostic process temporarily wraps observation
transfers. No warmup model execution is performed. The original predictor
still prepares RGB/state/noise and decodes the returned float32 container.

Timing boundaries follow https://docs.jax.dev/en/latest/benchmarking.html and
https://docs.jax.dev/en/latest/async_dispatch.html .
"""
from contextlib import contextmanager
import hashlib
import threading
import time

import numpy as np


class SamplingTimer:
    def __init__(self, function, *, jax_module, model_module):
        self.function = function
        self.jax = jax_module
        self.model_module = model_module
        self.compiled = None
        self.input_structure = None
        self.submitted = self.completed = self.compilations = 0
        self.current = None

    @contextmanager
    def prediction(self):
        """Reuse the original observation function, synchronizing its puts."""
        jax, module = self.jax, self.model_module
        original_observation = module.observation
        original_put = jax.device_put
        owner = threading.get_ident()
        self.current = dict(observation_calls=0, input_H2D_calls=0, input_H2D_bytes=0,
            input_H2D_s=0., observation_host_s=0., lowering_s=0., compilation_s=0.)

        def put(value, *args, **kwargs):
            assert threading.get_ident() == owner, 'SERIAL_DIAGNOSTIC_TRANSFER_ONLY'
            size = sum(int(getattr(x, 'nbytes', 0)) for x in jax.tree_util.tree_leaves(value))
            started = time.perf_counter()
            result = original_put(value, *args, **kwargs)
            jax.block_until_ready(result)
            self.current['input_H2D_s'] += time.perf_counter() - started
            self.current['input_H2D_calls'] += 1
            self.current['input_H2D_bytes'] += size
            return result

        def observation(*args, **kwargs):
            assert threading.get_ident() == owner
            assert jax.device_put is original_put
            started, prior = time.perf_counter(), self.current['input_H2D_s']
            jax.device_put = put
            try:
                return original_observation(*args, **kwargs)
            finally:
                jax.device_put = original_put
                transfer = self.current['input_H2D_s'] - prior
                self.current['observation_host_s'] += time.perf_counter() - started - transfer
                self.current['observation_calls'] += 1

        module.observation = observation
        try:
            yield self.current
        finally:
            module.observation = original_observation
            jax.device_put = original_put

    def __call__(self, active, frozen, observation, noise):
        jax = self.jax
        assert self.current is not None and self.current['observation_calls'] == 1
        started = time.perf_counter()
        jax.block_until_ready((observation, noise))
        self.current['input_and_rng_ready_wait_s'] = time.perf_counter() - started
        started = time.perf_counter()
        host_noise = np.asarray(jax.device_get(noise))
        self.current.update(noise_identity_D2H_s=time.perf_counter()-started,
            noise_identity_D2H_bytes=host_noise.nbytes,
            initial_noise_sha256=hashlib.sha256(host_noise.tobytes()).hexdigest())
        leaves, structure = jax.tree_util.tree_flatten((active, frozen, observation, noise))
        if self.compiled is None:
            # JAX0.5.3 lower() rebuilds its public argument tree with ArgInfo.
            # Keep typed Observation/NNX nodes inside tracing, where their
            # original constructors receive array tracers rather than ArgInfo.
            def invoke_flat(*values):
                return self.function(*jax.tree_util.tree_unflatten(structure, values))

            started = time.perf_counter()
            lowered = jax.jit(invoke_flat).lower(*leaves)
            self.current['lowering_s'] = time.perf_counter() - started
            started = time.perf_counter()
            self.compiled = lowered.compile()
            self.current['compilation_s'] = time.perf_counter() - started
            self.compilations += 1
            self.input_structure = structure
        elif structure != self.input_structure:
            raise ValueError('COMPILED_SAMPLING_INPUT_STRUCTURE_CHANGED')
        started = time.perf_counter()
        self.submitted += 1
        result = self.compiled(*leaves)
        jax.block_until_ready(result)
        self.completed += 1
        self.current['inference_device_completed_s'] = time.perf_counter() - started
        started = time.perf_counter()
        host = np.asarray(jax.device_get(result))
        self.current.update(output_D2H_s=time.perf_counter()-started, output_D2H_bytes=host.nbytes)
        return host
