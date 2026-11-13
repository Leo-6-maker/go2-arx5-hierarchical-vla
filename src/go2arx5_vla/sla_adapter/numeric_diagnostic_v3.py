"""Fixed-input microbatch numerics diagnosis with no optimizer mutation.

Uses the actual device engine's gradient, parent, data reader and RNG bank.
Diagnostic gradient/state exports are explicit boundaries, not hot-path D2H.
Neither a passing comparison nor this operation admits candidate training.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

from .batch_v2 import host_batch, device_batch
from .checkpoint_v1 import compare, sha
from .gradient_v2 import save as save_gradient, compare as compare_gradient

VERSION = 'FIXED32_MICROBATCH_NUMERICS_NO_UPDATE_V3'
POINTS = (1, 8, 16)
PRECISIONS = ('default', 'highest')


def diagnose(engine, root, *, predecessor=None):
    import jax
    import jax.numpy as jnp
    root = Path(root)
    if engine.step or engine.cursor or engine.counts['optimizer_updates']:
        raise ValueError('NUMERIC_DIAGNOSIS_REQUIRES_UNMODIFIED_PARENT_STEP0')
    initial = engine.checkpoint('numeric-initial')
    def forbidden_update(*args, **kwargs):
        raise RuntimeError('NUMERIC_DIAGNOSIS_OPTIMIZER_FORBIDDEN')
    engine.apply = forbidden_update
    def on_read(key):
        engine.counts[key] = engine.counts.get(key, 0) + 1
    host = host_batch(engine.dataset, engine.tokenizer, engine.visit_map, 0, 32, on_read=on_read)
    observation, actions = device_batch(host)
    noise, flow_time = engine.draw_bank(engine.rng, np.arange(32, dtype=np.uint32))
    random_inputs = dict(noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
                         time_sha256=hashlib.sha256(np.asarray(flow_time).tobytes()).hexdigest())
    if predecessor is not None:
        if predecessor['stochastic_inputs'] != random_inputs or predecessor['indices'] != host['indices']:
            raise ValueError('PRIOR_FAILED_ALLOCATION_DATA_OR_RANDOM_INPUTS_DIFFER')
    result = dict(schema=VERSION, optimizer_updates=0, optimization_visits=0,
                  physical_requests=0, ordinary_RGB_generations=0,
                  diagnostic_forward_backward_visits=0, diagnostic_forward_backward_calls=0,
                  original_tolerances_unchanged=True, candidate_training_admitted=False,
                  input_indices=host['indices'], stochastic_inputs=random_inputs,
                  fixed_input_samples=32, state_token_dropped=host['state_token_dropped'],
                  initial_checkpoint=initial, phases=[], comparisons=[])
    for precision in PRECISIONS:
        context = nullcontext() if precision == 'default' else jax.default_matmul_precision('highest')
        with context:
            for micro in POINTS:
                if time.time() >= engine.spec['deadline_epoch']:
                    raise TimeoutError('FINITE_NUMERIC_DIAGNOSIS_NO_OPTIMIZER_UPDATE')
                phase = root / (precision + '-micro' + str(micro))
                phase.mkdir(exist_ok=False)
                divisor = 32 // micro
                accumulate = jax.jit(lambda a, g: jax.tree.map(lambda x, y: x + y / divisor, a, g),
                                     donate_argnums=(0,))
                add = jax.jit(lambda a, values: a + jnp.stack(values) / divisor, donate_argnums=(0,))
                gradient = jax.tree.map(jnp.zeros_like, engine.active)
                metrics = jnp.zeros((3,), dtype=jnp.float32)
                began = time.perf_counter()
                hlo = None
                for offset in range(0, 32, micro):
                    if time.time() >= engine.spec['deadline_epoch']:
                        raise TimeoutError('FINITE_NUMERIC_DIAGNOSIS_DURING_FORWARD_BACKWARD')
                    obs = jax.tree.map(lambda x: x[offset:offset + micro], observation)
                    args = (engine.active, engine.frozen, obs, actions[offset:offset + micro],
                            noise[offset:offset + micro], flow_time[offset:offset + micro])
                    if offset == 0:
                        hlo = str(engine.grad.lower(*args).compiler_ir(dialect='stablehlo'))
                        (phase / 'GRADIENT.stablehlo.private.txt').write_text(hlo)
                    engine.counts['forward_backward_attempts'] += 1
                    engine.counts['attempted_diagnostic_visits'] = engine.counts.get('attempted_diagnostic_visits', 0) + micro
                    (loss, (active7, padding25)), grad = engine.grad(*args)
                    gradient = accumulate(gradient, grad)
                    metrics = add(metrics, (loss, active7, padding25))
                    jax.block_until_ready((metrics, gradient))
                    engine.counts['completed_forward_backward'] += 1
                    result['diagnostic_forward_backward_calls'] += 1
                    result['diagnostic_forward_backward_visits'] += micro
                    engine.counts['diagnostic_forward_backward_visits'] = result['diagnostic_forward_backward_visits']
                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise ValueError('NONFINITE_DIAGNOSTIC_LOSS_NO_UPDATE')
                binding = dict(engine.bindings, microbatch=micro, accumulation_steps=divisor,
                               diagnostic_matmul_precision=precision)
                audit = save_gradient(phase / 'gradient', gradient, step=0, cursor=0,
                                      bindings=binding, stochastic_inputs=random_inputs, indices=host['indices'])
                dots = [line.strip() for line in hlo.splitlines() if 'stablehlo.dot_general' in line]
                row = dict(precision=precision, microbatch=micro, effective_batch=32,
                           loss=float(values[0]), active7=float(values[1]), padding25=float(values[2]),
                           gradient=audit, wall_s=time.perf_counter() - began,
                           stablehlo_sha256=sha(phase / 'GRADIENT.stablehlo.private.txt'),
                           dot_general_count=len(dots),
                           operand_precision_counts={name: sum(bool(re.search(r'\b' + name + r'\b', line)) for line in dots)
                                                     for name in ('DEFAULT', 'HIGH', 'HIGHEST')},
                           dot_input_types=sorted(set(re.findall(r'tensor<[^>]+>', '\n'.join(dots)))))
                (phase / 'RESULT.private.json').write_text(json.dumps(row, indent=2))
                result['phases'].append(row)
                (root / 'PROGRESS.private.json').write_text(json.dumps(result, indent=2))
        rows = result['phases'][-3:]
        for row in rows[1:]:
            check = compare_gradient(rows[0]['gradient']['path'], row['gradient']['path'],
                                     left_loss=rows[0]['loss'], right_loss=row['loss'])
            file = root / (precision + '-micro1-vs' + str(row['microbatch']) + '.private.json')
            file.write_text(json.dumps(check, indent=2))
            result['comparisons'].append(dict(precision=precision, right_microbatch=row['microbatch'],
                **{k: v for k, v in check.items() if k != 'records'}, sha256=sha(file)))
    final = engine.checkpoint('numeric-final')
    state = compare(initial['path'], final['path'])
    if not state['bit_exact'] or engine.step or engine.cursor or engine.counts['optimizer_updates']:
        raise ValueError('NUMERIC_DIAGNOSIS_MUTATED_TRAINING_STATE')
    result.update(final_checkpoint=final, unchanged_full_state=state,
                  frozen_fingerprint_unchanged=engine.verify_frozen(), actual_engine_counts=engine.counts)
    assert result['diagnostic_forward_backward_visits'] == 192
    return result
