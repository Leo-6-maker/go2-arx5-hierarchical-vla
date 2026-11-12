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
    errors = [float(np.max(np.abs(np.asarray(x).astype(np.float64)-np.asarray(y).astype(np.float64)))) for x, y in zip(a, b)]
    return dict(bit_exact=all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)),
                max_abs=max(errors, default=0.), leaves=len(a))


def probes(engine, name):
    root = engine.output/name
    root.mkdir()
    records = []
    for i, index in enumerate(probe_indices(engine.dataset)):
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
    index = probe_indices(engine.dataset)[0]
    item = engine.dataset[index]
    obs, action = observation(item, engine.tokenizer)
    key = jax.random.PRNGKey(2026111200)
    # Share dense buffers; this does not allocate another pretrained model.
    base = dense_model(engine.active, engine.frozen)
    noise = jax.random.normal(key, (1, 40, 32))
    baseline_fn = jax.jit(lambda p, x, n: nnx.merge(nnx.graphdef(base), p).sample_actions(key, x, noise=n, num_steps=10))
    baseline = baseline_fn(nnx.state(base), obs, noise)
    adapted = engine.infer(engine.active, engine.frozen, obs, noise)
    jax.block_until_ready(adapted)
    np.savez(engine.output/'INITIAL_FUNCTION.private.npz', dense=np.asarray(baseline), lora=np.asarray(adapted))
    engine.counts['diagnostic_free_generations'] = 2
    function = exact_trees(baseline, adapted)
    if not function['bit_exact']:
        raise ValueError('INITIAL_LORA_FUNCTION_NOT_BIT_EXACT:'+str(function))
    baseline_fn.clear_cache()
    del baseline, adapted, base, baseline_fn
    (with_aux, gradient) = engine.grad(engine.active, engine.frozen, key, obs, action)
    original_loss, original_gradient = engine.upstream_grad(engine.active, engine.frozen, key, obs, action)
    jax.block_until_ready(original_gradient)
    equivalence = exact_trees((with_aux[0], gradient), (original_loss, original_gradient))
    if not equivalence['bit_exact']:
        raise ValueError('FM_DIAGNOSTIC_WRAPPER_DIFFERS_FROM_UPSTREAM:'+str(equivalence))
    engine.upstream_grad.clear_cache()
    # These are explicit non-update diagnostic calls, distinct from the 8
    # full effective-batch qualification updates and their 256 visits.
    return dict(initial_function=function, original_FM_loss_and_gradient=equivalence,
                extra_generation_calls=2, extra_forward_backward_calls=2, extra_TRAIN_window_visits=4)


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
