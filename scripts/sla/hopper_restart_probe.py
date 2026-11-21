"""Exercise the new restart format and trace actual H200 candidate input shapes on CPU."""
from pathlib import Path
import tempfile

import numpy as np

from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_engine import HopperEngine
from scripts.sla.hopper_parent_custody import digest


def run(restored, cfg, root):
    import jax
    from go2arx5_vla.sla_adapter import batch_v2
    from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch, target_auxiliary_context
    from scripts.sla.hopper_inputs import BATCH, PARENT

    state = restored[2]
    bound = checkpoint.binding(restored, role='CPU_ROUNDTRIP', source=cfg['source_revision'],
                               fm_micro=1, aux_micro=1)
    template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
    with tempfile.TemporaryDirectory(prefix='restart-proof-') as tmp:
        pin = checkpoint.save(Path(tmp)/'phase-zero', state, phase=0, bound=bound, predecessor=PARENT)
        target, clocks = checkpoint.resume(pin, template, phase=0, bound=bound, predecessor=PARENT)
        source_leaves = jax.tree.leaves(state)
        restored_leaves = jax.tree.leaves(target)
        exact = sum(np.array_equal(np.asarray(left).reshape(-1).view(np.uint8),
                                   np.asarray(right).reshape(-1).view(np.uint8))
                    for left, right in zip(source_leaves, restored_leaves, strict=True))
        if exact != 140:
            raise ValueError('New complete-state format changed training arrays')
        other = dict(bound, role='MAIN')
        try:
            checkpoint.resume(pin, template, phase=0, bound=other, predecessor=PARENT)
        except ValueError as err:
            if 'role' not in str(err):
                raise
        else:
            raise ValueError('CPU checkpoint accepted as a formal endpoint')
        del target, source_leaves, restored_leaves
    result = dict(seal=pin['sha256'], exact_arrays=exact, clocks=clocks, cross_role_rejected=True)
    from scripts.sla.hopper_parent_custody import write_once
    write_once(root/'RESTART_PROOF.private.json', result)

    engine = HopperEngine(restored, phase=0, segment_stop=1, fm_micro=1, aux_micro=1)
    traced = {}
    # eval_shape traces the real graph including backward and ten-step rematerialization.
    # It determines shapes but does not dispatch model inference or a training kernel.
    for domain, size in BATCH.items():
        indices, cursor = engine.indices(0, domain)
        host = prepare_host_batch(engine.datasets[domain], engine.tokenizer, domain,
                                  range(cursor, cursor + 16), indices[:16])
        draws = engine.draw[domain](engine.rng, cursor)
        active = dict(shared=engine.parameters['shared'], private=engine.parameters[domain])
        for width in (1, 4, 8, 16):
            observation, actions = batch_v2.device_batch(jax.tree.map(lambda a: a[:width], host))
            args = (observation, actions, draws[0][:width], draws[1][:width])
            loss, grad = jax.eval_shape(engine.grad[domain], active, engine.frozen, *args)
            if loss.shape != () or jax.tree.structure(grad) != jax.tree.structure(active):
                raise ValueError('Invalid FM abstract gradient tree')
            traced[f'{domain}_FM_mb{width}'] = [size, len(jax.tree.leaves(grad))]
        if domain == 'target':
            ctx = [target_auxiliary_context(engine.datasets[domain], engine.rng, cursor+i, int(indices[i]))
                   for i in range(4)]
            for width in (1, 2, 4):
                observation, _ = batch_v2.device_batch(jax.tree.map(lambda a: a[:width], host))
                values = [np.concatenate(v) for v in zip(*ctx[:width], strict=True)]
                loss, grad = jax.eval_shape(engine.aux, active, engine.frozen, observation, *values)
                if loss.shape != () or jax.tree.structure(grad) != jax.tree.structure(active):
                    raise ValueError('Invalid H40 abstract gradient tree')
                traced[f'target_aux_mb{width}'] = [4, len(jax.tree.leaves(grad))]
    return dict(restart=result, restart_receipt_sha256=digest(root/'RESTART_PROOF.private.json'),
                abstract_kernel_shapes=traced, graph_execution=False)
