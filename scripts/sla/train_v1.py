"""One registered finite SLA qualification/candidate process; no scheduler."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from go2arx5_vla.sla_adapter.runtime_v1 import Engine, read_pin, probe_indices
from go2arx5_vla.sla_adapter.model_v1 import dense_model, observation
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha

ACTIVE_ENGINE = None


def exact_trees(left, right):
    import jax
    a, ta = jax.tree.flatten(left)
    b, tb = jax.tree.flatten(right)
    if ta != tb or len(a) != len(b):
        raise ValueError('QUALIFICATION_TREE_STRUCTURE_MISMATCH')
    if not all(np.isfinite(np.asarray(v)).all() for v in a+b):
        raise ValueError('NONFINITE_QUALIFICATION_OUTPUT')
    errors = [float(np.max(np.abs(np.asarray(x).astype(np.float64)-np.asarray(y).astype(np.float64)))) for x, y in zip(a, b)]
    return dict(bit_exact=all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)),
                max_abs=max(errors, default=0.), leaves=len(a))


def probes(engine, name, *, indices=None):
    root = engine.output/name
    root.mkdir()
    records = []
    for i, index in enumerate(probe_indices(engine.dataset) if indices is None else indices):
        for noise in range(2):
            normalized, physical = engine.generate(index, noise)
            p = root/f'{i:02d}-{noise}.npz'
            np.savez(p, normalized=normalized, physical_H40=physical)
            records.append(dict(index=index, noise=noise, file=p.name,
                                normalized_sha256=__import__('hashlib').sha256(normalized.tobytes()).hexdigest(),
                                physical_sha256=__import__('hashlib').sha256(physical.tobytes()).hexdigest()))
    (root/'MANIFEST.private.json').write_text(json.dumps(records, indent=2))
    return records


def qualification_seams(engine):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from go2arx5_vla.sla_adapter.model_v1 import flat
    rule = read_pin(engine.spec['initial_function_rule'])
    base = dense_model(engine.active, engine.frozen)
    original = flat(nnx.state(base).to_pure_dict())
    parent = flat(engine.active.to_pure_dict()) | flat(engine.frozen.to_pure_dict())
    if not all(bool(jax.jit(jnp.array_equal)(v, parent[p])) for p, v in original.items()):
        raise ValueError('INITIAL_DENSE_PARENT_NOT_IDENTICAL')
    if not all(np.all(np.asarray(v) == 0) for p, v in flat(engine.active.to_pure_dict()).items() if p[-1].endswith('lora_b')):
        raise ValueError('INITIAL_LORA_B_NOT_ZERO')
    graph = nnx.graphdef(base)
    baseline_fn = jax.jit(lambda p, x, n: nnx.merge(graph, p).sample_actions(jax.random.PRNGKey(0), x, noise=n, num_steps=10))
    root = engine.output/'initial-function';root.mkdir()
    comparisons = []
    for ordinal, index in enumerate(probe_indices(engine.dataset)):
        obs, _ = observation(engine.dataset[index], engine.tokenizer)
        for noise_id in range(2):
            noise = jax.random.normal(jax.random.PRNGKey(2026111200+noise_id), (1, 40, 32))
            engine.counts['diagnostic_generation_attempts'] = engine.counts.get('diagnostic_generation_attempts', 0)+1
            baseline = baseline_fn(nnx.state(base), obs, noise)
            jax.block_until_ready(baseline)
            engine.counts['diagnostic_free_generations'] = engine.counts.get('diagnostic_free_generations', 0)+1
            engine.counts['diagnostic_generation_attempts'] += 1
            adapted = engine.infer(engine.active, engine.frozen, obs, noise)
            jax.block_until_ready(adapted)
            engine.counts['diagnostic_free_generations'] += 1
            check = exact_trees(baseline, adapted)
            np.savez(root/f'{ordinal:02d}-{noise_id}.npz', dense=np.asarray(baseline), lora=np.asarray(adapted))
            if check['max_abs'] > rule['cross_graph_normalized_max_abs']:
                raise ValueError('INITIAL_LORA_REGISTERED_NUMERIC_BOUND_FAILED:'+str(check))
            comparisons.append(dict(index=index, noise=noise_id, **check))
    engine.counts['diagnostic_generation_attempts'] += 1
    repeated = engine.infer(engine.active, engine.frozen, obs, noise)
    if not exact_trees(adapted, repeated)['bit_exact']:
        raise ValueError('SAME_GRAPH_INITIAL_INFERENCE_NOT_BIT_EXACT')
    engine.counts['diagnostic_free_generations'] += 1
    baseline_fn.clear_cache()
    del original, parent, baseline, adapted, base, baseline_fn
    index = probe_indices(engine.dataset)[0]
    obs, action = observation(engine.dataset[index], engine.tokenizer)
    key = jax.random.PRNGKey(2026111200)
    engine.counts['diagnostic_backward_attempts'] = 1
    native_loss, gradient = engine.grad(engine.active, engine.frozen, key, obs, action)
    jax.block_until_ready(gradient)
    engine.counts['diagnostic_backward_calls'] = 1
    diagnostic_loss, dimensional = engine.diagnostics(engine.active, engine.frozen, key, obs, action)
    jax.block_until_ready(diagnostic_loss)
    engine.counts['diagnostic_forward_calls'] = 1
    engine.counts['diagnostic_TRAIN_visits'] = 2
    delta = abs(float(native_loss)-float(diagnostic_loss))
    grad_norm = float(optax.global_norm(gradient))
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError('NATIVE_UPSTREAM_SLA_GRADIENT_NOT_FINITE_NONZERO')
    if delta > rule['diagnostic_forward_decomposition_max_loss_abs']:
        raise ValueError('DIMENSIONAL_DIAGNOSTIC_DIFFERS_FROM_NATIVE_FM')
    return dict(initial_function_comparisons=comparisons,
                numeric_rule_sha256=engine.spec['initial_function_rule']['sha256'],
                same_graph_repeat_bit_exact=True, shared_dense_parameters_bit_exact=True,
                all_B_exact_zero=True, training_gradient_implementation='UPSTREAM_COMPUTE_LOSS_TRAIN_FALSE_MEAN32',
                native_gradient_norm=grad_norm, dimensional_forward_loss_difference=delta,
                active7=float(dimensional[0]), padding25=float(dimensional[1]))


def run(spec):
    import jax
    operation = spec['operation']
    if operation not in ('QUALIFY_CONTINUOUS', 'QUALIFY_SPLIT', 'QUALIFY_RESUME', 'RECONSTRUCT', 'CANDIDATE'):
        raise ValueError('FINITE_SLA_OPERATION_REQUIRED')
    if operation == 'CANDIDATE':
        q = read_pin(spec['training_admission'])
        if (q.get('schema') != 'SLA_CUDA_TRAIN_RESTORE_RGB_ADMISSION_V1' or not q.get('passes')
                or q.get('noncandidate_updates') != 8 or q.get('target_audit_sha256') != spec['target_audit']['sha256']
                or q.get('parent_manifest_sha256') != spec['parent_manifest']['sha256']):
            raise ValueError('ACTUAL_SLA_TRAIN_RESTORE_RGB_QUALIFICATION_REQUIRED')
    global ACTIVE_ENGINE
    engine = ACTIVE_ENGINE = Engine.__new__(Engine)
    Engine.__init__(engine, spec, Path(spec['registered_root'])/'work')
    result = dict(operation=operation, noncandidate=operation != 'CANDIDATE', requested_updates=0,
                  requested_physical_requests=0, started_epoch=time.time())
    if spec.get('resume'):
        engine.resume(spec['resume'])
    if operation == 'RECONSTRUCT':
        result['RGB_probes'] = probes(engine, 'ordinary-RGB-reconstruction')
    else:
        stop = 500 if operation == 'CANDIDATE' else 2 if operation == 'QUALIFY_SPLIT' else 4
        result['requested_updates'] = stop-engine.step
        if operation == 'QUALIFY_CONTINUOUS':
            result['seams'] = qualification_seams(engine)
            result['initial_RGB_probes'] = probes(engine, 'step0-RGB')
        elif operation == 'CANDIDATE' and engine.step == 0:
            result['initial_RGB_probes'] = probes(engine, 'step0-RGB')
        if engine.step == 0:
            engine.checkpoint('step000')
        while engine.step < stop:
            if time.time() >= spec['deadline_epoch']:
                raise TimeoutError('FINITE_SLA_JOB_DEADLINE')
            step = engine.update()
            if engine.step <= 4 or engine.step % 50 == 0:
                print(json.dumps(dict(event='ACTUAL_EFFECTIVE_BATCH_UPDATE', step=engine.step,
                                      cursor=engine.cursor, loss=step['micro'][-1]['loss'],
                                      grad_norm=step['grad_norm_before_clip'], wall_s=step['wall_s'])), flush=True)
            save_steps = (250, 500) if operation == 'CANDIDATE' else (2, 4)
            if engine.step in save_steps:
                result['last_checkpoint'] = engine.checkpoint(f'step{engine.step:03d}')
                if operation == 'CANDIDATE' and engine.step == 250:
                    result['midpoint_RGB_probes'] = probes(engine, 'step250-RGB')
        if operation in ('QUALIFY_CONTINUOUS', 'CANDIDATE'):
            result['endpoint_RGB_probes'] = probes(engine, f'step{engine.step}-RGB')
    result.update(status='COMPLETED_FINITE_SLA_OPERATION', step=engine.step, cursor=engine.cursor,
                  actual=engine.counts, frozen_fingerprint_unchanged=engine.verify_frozen(),
                  residency=engine.residency(), load_wall_s=engine.load_wall_s, completed_epoch=time.time())
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(engine.active)):
        raise ValueError('NONFINITE_SLA_TRAINABLE_PARAMETERS')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    a = parser.parse_args()
    spec = json.loads(a.spec.read_bytes())
    try:
        result = run(spec)
    except Exception as error:
        failure = dict(status='TECHNICAL_FAILURE_NOT_POLICY_RESULT', error=repr(error),
                       operation=spec['operation'], actual=getattr(ACTIVE_ENGINE, 'counts', {}),
                       step=getattr(ACTIVE_ENGINE, 'step', 0), cursor=getattr(ACTIVE_ENGINE, 'cursor', 0),
                       physical_requests=0, epoch=time.time())
        (Path(spec['registered_root'])/'FAILED.private.json').write_text(json.dumps(failure, indent=2))
        raise
    with (Path(spec['registered_root'])/'RESULT.private.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
