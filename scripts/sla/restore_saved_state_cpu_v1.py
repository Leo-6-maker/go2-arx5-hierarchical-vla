"""Read back all saved S1_500 state on CPU without a dense parent or update.

This is endpoint file/state integrity evidence. The separate real CUDA4vs2+2
qualification remains the proof of GPU training continuation.
"""
import json
import os
from pathlib import Path
import sys
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import restore, save, compare, sha
from go2arx5_vla.sla_adapter.config_v1 import trainable, learning_rates
from go2arx5_vla.sla_adapter.model_v1 import configuration


def run(spec):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.training.optimizer import AdamW
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or any(d.platform != 'cpu' for d in jax.devices())):
        raise ValueError('ENDPOINT_STATE_READBACK_IS_CPU_ONLY')
    checkpoint = Path(spec['resume']['path'])
    if sha(checkpoint/'COMPLETE.json') != spec['resume']['sha256']:
        raise ValueError('FIXED500_SAVED_COMPLETE_CHANGED')
    manifest = json.loads((checkpoint/'COMPLETE.json').read_bytes())
    if (manifest['step'] != 500 or manifest['cursor'] != 16000 or len(manifest['arrays']) != 87
            or manifest['bindings']['parent_manifest'] != spec['parent_manifest']
            or manifest['bindings']['target_audit'] != spec['target_audit']['sha256']
            or manifest['bindings']['recipe'] != 'S1_FM32_MICRO1_ACCUM32_DROPOUT_TOKEN_ONLY_V1'):
        raise ValueError('COMPLETE_S1_FIXED500_IDENTITY_REQUIRED')
    source = Path(spec['source_root'])
    if any(sha(source/name) != value for name,value in manifest['bindings']['adapter_source'].items()):
        raise ValueError('ENDPOINT_READBACK_CHANGED_TRAINED_ADAPTER_CODE')
    template = nnx.eval_shape(configuration().create, jax.random.PRNGKey(0))
    _, active, _ = nnx.split(template, nnx.All(nnx.Param, lambda path,_:trainable(path)), ...)
    rates = jnp.asarray(learning_rates(), dtype=jnp.float32)
    tx = AdamW(weight_decay=1e-5,clip_gradient_norm=1.).create(lambda count:rates[jnp.minimum(count,499)])
    optimizer = jax.eval_shape(tx.init,active)
    rng = jax.ShapeDtypeStruct((2,),jnp.uint32)
    state,step,cursor = restore(checkpoint,(active,optimizer,rng),bindings=manifest['bindings'])
    leaves = jax.tree.leaves(state)
    if len(leaves) != 87 or any(d.platform != 'cpu' for a in leaves for d in a.devices()):
        raise ValueError('ALL87_STATE_ARRAYS_MUST_BE_ACTUALLY_RESTORED_ON_CPU')
    target = Path(spec['registered_root'])/'restored-state'
    saved = save(target,state,step=step,cursor=cursor,bindings=manifest['bindings'])
    result = compare(checkpoint,target)
    if not result['bit_exact'] or result['arrays'] != 87:
        raise ValueError('FULL_ENDPOINT_STATE_RESERIALIZATION_DIFFERS')
    return dict(schema='SLA_FIXED500_FULL_STATE_CPU_READBACK_ACTUAL_V1',passes=True,
        original_checkpoint=spec['resume'],restored_checkpoint=saved,comparison=result,
        restored_parameter_arrays=28,restored_optimizer_and_RNG_arrays=59,
        actual_CPU_state_arrays=87,abstract_graph_traces=1,dense_parent_loads=0,
        inference_calls=0,forward_backward_calls=0,optimizer_updates=0,
        training_optimization_visits=0,physical_requests=0,CUDA_initialized=False,
        claim='CPU_COMPLETE_STATE_INTEGRITY_ONLY; CUDA_TRAIN_RESTORE_QUALIFICATION_REUSED_SEPARATELY')


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root'])/'RESULT.private.json',run(spec))
