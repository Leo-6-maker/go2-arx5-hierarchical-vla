"""CPU abstract tracing only: no model weights, CUDA, update or qualification."""
import json
from pathlib import Path
import sys
from scripts.pi0292.durable import write


def run(spec):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from openpi.models.model import Observation
    from openpi.models.projectors import AlignProjector
    from go2arx5_vla.sla_adapter.model_v1 import configuration
    from go2arx5_vla.sla_adapter.config_v1 import trainable
    from go2arx5_vla.sla_adapter.spatial_runtime_v1 import spatial_objective
    if any(d.platform!='cpu' for d in jax.devices()):
        raise ValueError('ONLY_CPU_ABSTRACT_TRACE_ALLOWED')
    model=nnx.eval_shape(configuration().create,jax.random.PRNGKey(20261112))
    graph,active,frozen=nnx.split(model,nnx.All(nnx.Param,lambda path,_:trainable(path)),...)
    active=jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape,jnp.float32),active)
    frozen=jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape,jnp.bfloat16),frozen)
    cfg=SimpleNamespace(vggt_dim=1024,vla_adapt_layers_align=False,vggt_adapt_layers_align=False,
                         iSF_zscore=False,use_vlm_norm=True)
    projector=nnx.eval_shape(lambda:AlignProjector(2048,cfg,nnx.Rngs(20261112)))
    pgraph,projector=nnx.split(projector)
    def shape(size,dtype=jnp.float32):return jax.ShapeDtypeStruct(size,dtype)
    cameras=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']
    obs=Observation(images={k:shape((1,224,224,3))for k in cameras},
        image_masks={k:shape((1,),jnp.bool_)for k in cameras},
        image_padding_mask={k:shape((1,224,224),jnp.bool_)for k in cameras},
        state=shape((1,32)),tokenized_prompt=shape((1,200),jnp.int32),
        tokenized_prompt_mask=shape((1,200),jnp.bool_))
    loss,gradients=jax.eval_shape(jax.value_and_grad(spatial_objective(graph,pgraph),argnums=(0,2)),
        active,frozen,projector,shape((2,),jnp.uint32),obs,shape((1,40,32)),
        shape((1,768,2048),jnp.bfloat16),shape((1,768),jnp.bool_))
    if loss.shape!=() or [len(jax.tree.leaves(g))for g in gradients]!=[28,6]:
        raise ValueError('EXPECTED_COMMON_AND_PROJECTOR_GRADIENT_SHAPES_REQUIRED')
    return dict(schema='SLA_SPATIAL_ABSTRACT_TRACE_SOFTWARE_ONLY_V1',passes=True,
        common_gradient_leaves=28,projector_gradient_leaves=6,alignment_tokens=768,
        teacher_valid_tokens=512,missing_wrist_masked_tokens=256,alignment_width=2048,
        model_weights_loaded=0,actual_CUDA_gradients=0,optimizer_updates=0,training_visits=0,
        teacher_loads=0,physical_requests=0,training_admission=False)


if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root'])/'RESULT.private.json',run(spec))
