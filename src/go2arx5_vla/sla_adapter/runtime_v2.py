"""Single-device true microbatches for W1 and the separate R1 logit recipe.

Reuses V1's qualified parent/PEFT/Adam/checkpoint/inference implementation.
No V1 module, active capsule or existing scientific endpoint is modified.
This implementation still requires real hardware qualification before use.
"""
import json
import time
import numpy as np

from .runtime_v1 import Engine as OriginalEngine, read_pin
from .batch_v2 import RNG_VERSION, accumulation_steps, prepared_batches, device_batch, draw_bank, error32


class BatchedEngine(OriginalEngine):
    def __init__(self, spec, output):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        self.microbatch = spec['microbatch']
        self.accumulation_steps = accumulation_steps(self.microbatch)
        super().__init__(spec, output)
        if spec.get('action_representation_audit'):
            from .dataset_v2 import data
            from .gripper_v2 import VERSION
            self.dataset = data(spec)
            audit = read_pin(spec['action_representation_audit'])
            self.bindings.update(codec_version=VERSION,
                                 action_representation_audit=spec['action_representation_audit'],
                                 derived_statistics_sha256=audit['statistics_sha256'])
        self.bindings.update(recipe='SLA_FM32_TRUE_MICROBATCH_V2',
                             microbatch=self.microbatch, accumulation_steps=self.accumulation_steps,
                             effective_batch=32, random_stream=RNG_VERSION,
                             sample_cursor='ABSOLUTE_COMMITTED_OPTIMIZATION_VISIT',
                             prefetch='ONE_NEXT_CPU_INPUT_MICROBATCH_NO_FEATURE_CACHE')
        graph = self.graph

        def objective(active, frozen, observation, actions, noise, flow_time):
            error = error32(nnx.merge(graph, active, frozen), observation, actions, noise, flow_time)
            value = jnp.mean(jnp.mean(error, axis=-1))
            return value, (jnp.mean(error[..., :7]), jnp.mean(error[..., 7:]))

        self.grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
        self.draw_bank = jax.jit(draw_bank)
        divisor = self.accumulation_steps
        self.accumulate = jax.jit(lambda acc, grad: jax.tree.map(lambda a, g: a + g / divisor, acc, grad),
                                 donate_argnums=(0,))

    def update(self):
        import jax
        import jax.numpy as jnp
        start = time.perf_counter()
        accumulator = jax.tree.map(jnp.zeros_like, self.active)
        logs = []
        self.counts['rng_bank_generation_attempts'] = self.counts.get('rng_bank_generation_attempts', 0) + 1
        noise, flow_time = self.draw_bank(self.rng, np.arange(self.cursor, self.cursor + 32, dtype=np.uint32))
        jax.block_until_ready(noise)
        self.counts['rng_banks_generated'] = self.counts.get('rng_banks_generated', 0) + 1
        def read_count(key):
            self.counts[key] = self.counts.get(key, 0) + 1
        batches = prepared_batches(self.dataset, self.tokenizer, self.visit_map, self.cursor,
                                   self.microbatch, prefetch=self.spec.get('prefetch', True), on_read=read_count)
        try:
            for host in batches:
                if time.time() >= self.spec['deadline_epoch']:
                    raise TimeoutError('FINITE_BATCH_V2_DEADLINE_BEFORE_OPTIMIZER_UPDATE')
                observation, actions = device_batch(host)
                self.counts['forward_backward_attempts'] += 1
                self.counts['attempted_training_visits'] += self.microbatch
                offset = int(host['cursors'][0]) - self.cursor
                (value, (active7, padding25)), gradient = self.grad(
                    self.active, self.frozen, observation, actions,
                    noise[offset:offset + self.microbatch], flow_time[offset:offset + self.microbatch])
                jax.block_until_ready(gradient)
                self.counts['completed_forward_backward'] += 1
                self.counts['completed_training_visits'] += self.microbatch
                if not np.isfinite(float(value)):
                    raise ValueError('NONFINITE_BATCH_V2_FM')
                accumulator = self.accumulate(accumulator, gradient)
                logs.append(dict(loss=float(value), active7=float(active7), padding25=float(padding25),
                                 indices=host['indices'], first_absolute_cursor=int(host['cursors'][0]),
                                 samples=self.microbatch, state_token_dropped=host['state_token_dropped'],
                                 host_prepare_wall_s=host['prepare_wall_s'], host_wait_s=host['host_wait_s']))
        finally:
            batches.close()
        self.counts['optimizer_update_attempts'] = self.counts.get('optimizer_update_attempts', 0) + 1
        self.active, self.opt_state, norm = self.apply(self.active, self.opt_state, accumulator)
        jax.block_until_ready(self.active)
        if not np.isfinite(float(norm)):
            raise ValueError('NONFINITE_BATCH_V2_GRADIENT')
        self.counts['optimizer_updates'] += 1
        self.step += 1
        self.cursor += 32
        wall = time.perf_counter() - start
        result = dict(step=self.step, cursor=self.cursor, microbatch=self.microbatch,
                      accumulation_steps=self.accumulation_steps, effective_batch=32,
                      used_lr=self.learning_rate_values[self.step - 1], grad_norm_before_clip=float(norm),
                      mean_FM32_loss=float(np.mean([row['loss'] for row in logs])), micro=logs,
                      wall_s=wall, samples_per_s=32 / wall, updates_per_s=1 / wall,
                      host_wait_s=sum(row['host_wait_s'] for row in logs),
                      GPU_memory=jax.devices()[0].memory_stats())
        with (self.output / 'updates.private.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        return result
