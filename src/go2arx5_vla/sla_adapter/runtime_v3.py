"""Finite single-GPU qualification with device metrics and persistent prefetch.

Large diagnostic gradient exports and checkpoints remain explicit boundaries.
No per-microbatch scalar or gradient is copied to the host in normal updates.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import time
import numpy as np

from .runtime_v2 import BatchedEngine
from .batch_v2 import prepared_batches, device_batch


def training_matmul_precision(spec):
    precision = spec.get('training_matmul_precision', 'default')
    if precision not in ('default', 'highest'):
        raise ValueError('UNSUPPORTED_SLA_TRAINING_MATMUL_PRECISION')
    return precision


class DeviceResidentEngine(BatchedEngine):
    def __init__(self, spec, output):
        super().__init__(spec, output)
        import jax
        import jax.numpy as jnp
        import optax
        self.loader = ThreadPoolExecutor(max_workers=1)
        self.bindings.update(recipe='SLA_EFFECTIVE32_DEVICE_REDUCTION_V3',
                             prefetch='PERSISTENT_ONE_CURRENT_ONE_NEXT_CPU_MICROBATCH')
        self.training_precision = training_matmul_precision(spec)
        if self.training_precision != 'default':
            self.bindings['training_matmul_precision'] = self.training_precision
        divisor = self.accumulation_steps
        self.add_metrics = jax.jit(lambda acc, values: acc + jnp.stack(values) / divisor,
                                  donate_argnums=(0,))
        self.summarize = jax.jit(lambda metrics, gradient:
                                 jnp.concatenate([metrics, jnp.reshape(optax.global_norm(gradient), (1,))]))

    def close(self):
        self.loader.shutdown(wait=True, cancel_futures=True)

    def update(self):
        import jax
        import jax.numpy as jnp
        start = time.perf_counter()
        accumulator = jax.tree.map(jnp.zeros_like, self.active)
        metrics = jnp.zeros((3,), dtype=jnp.float32)
        logs = []
        self.counts['rng_bank_generation_attempts'] = self.counts.get('rng_bank_generation_attempts', 0) + 1
        noise, flow_time = self.draw_bank(self.rng, np.arange(self.cursor, self.cursor + 32, dtype=np.uint32))
        def count(key):
            self.counts[key] = self.counts.get(key, 0) + 1
        batches = prepared_batches(self.dataset, self.tokenizer, self.visit_map, self.cursor,
                                   self.microbatch, prefetch=self.spec.get('prefetch', True),
                                   on_read=count, executor=self.loader)
        try:
            for host in batches:
                if time.time() >= self.spec['deadline_epoch']:
                    raise TimeoutError('FINITE_V3_WINDOW_BEFORE_COMMITTED_UPDATE')
                observation, actions = device_batch(host)
                self.counts['forward_backward_attempts'] += 1
                self.counts['attempted_training_visits'] += self.microbatch
                offset = int(host['cursors'][0]) - self.cursor
                context = (jax.default_matmul_precision('highest')
                           if self.training_precision == 'highest' else nullcontext())
                with context:
                    (loss, (active7, padding25)), gradient = self.grad(self.active, self.frozen, observation, actions,
                        noise[offset:offset+self.microbatch], flow_time[offset:offset+self.microbatch])
                accumulator = self.accumulate(accumulator, gradient)
                metrics = self.add_metrics(metrics, (loss, active7, padding25))
                logs.append(dict(indices=host['indices'], first_absolute_cursor=int(host['cursors'][0]),
                                 samples=self.microbatch, state_token_dropped=host['state_token_dropped'],
                                 host_prepare_wall_s=host['prepare_wall_s'], host_wait_s=host['host_wait_s']))
        finally:
            batches.close()
        # One small terminal read, before any optimizer mutation. Norm covers
        # every active gradient leaf; a nonfinite result leaves weights/cursor.
        summary = np.asarray(self.summarize(metrics, accumulator))
        self.counts['completed_forward_backward'] += self.accumulation_steps
        self.counts['completed_training_visits'] += 32
        count('rng_banks_generated')
        count('update_metric_D2H_calls')
        self.counts['update_metric_D2H_bytes'] = self.counts.get('update_metric_D2H_bytes', 0) + summary.nbytes
        if not np.isfinite(summary).all():
            raise ValueError('NONFINITE_V3_EFFECTIVE32_LOSS_OR_GRADIENT_NO_UPDATE')
        audit = None
        if self.step in self.spec.get('gradient_capture_steps', []):
            from .gradient_v2 import save
            audit = save(self.output / f'gradient-step{self.step:03d}', accumulator,
                         step=self.step, cursor=self.cursor, bindings=self.bindings,
                         stochastic_inputs=dict(noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
                                                time_sha256=hashlib.sha256(np.asarray(flow_time).tobytes()).hexdigest()),
                         indices=[index for row in logs for index in row['indices']])
            count('gradient_audit_snapshots')
        count('optimizer_update_attempts')
        self.active, self.opt_state, _ = self.apply(self.active, self.opt_state, accumulator)
        jax.block_until_ready((self.active, self.opt_state))
        self.counts['optimizer_updates'] += 1
        self.step += 1
        self.cursor += 32
        wall = time.perf_counter() - start
        result = dict(step=self.step, cursor=self.cursor, microbatch=self.microbatch,
                      accumulation_steps=self.accumulation_steps, effective_batch=32,
                      used_lr=self.learning_rate_values[self.step-1], grad_norm_before_clip=float(summary[3]),
                      mean_FM32_loss=float(summary[0]), active7=float(summary[1]), padding25=float(summary[2]),
                      micro=logs, gradient_audit=audit, wall_s=wall, samples_per_s=32/wall,
                      updates_per_s=1/wall, host_wait_s=sum(r['host_wait_s'] for r in logs),
                      GPU_memory=jax.devices()[0].memory_stats())
        with (self.output/'updates.private.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        return result
