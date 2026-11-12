"""Matched S2: unchanged native FM plus frozen spatial-gradient supervision.

The two backwards are sequential and GPU-resident. Their gradient sum is the
same objective by linearity; activation lifetimes do not overlap. One global
clip/AdamW covers common parameters and the inherited training-only projector.
"""
import json
from pathlib import Path
import time
import numpy as np
from .runtime_v1 import Engine, read_pin
from .model_v1 import observation, flat
from .config_v1 import SEED, learning_rates
from .checkpoint_v1 import save, restore, sha
from .spatial_assets_v1 import load_projector, FeatureBank


def common_optimizer(state):
    """Read common Adam moments from the paired tree, retaining all counters."""
    if hasattr(state,'mu') and hasattr(state,'nu'):
        return state._replace(mu=state.mu[0],nu=state.nu[0])
    if type(state) is tuple:
        return tuple(common_optimizer(v) for v in state)
    return state


def vlm_lora(path):
    return path[:2]==('PaliGemma','llm') and any('lora' in p for p in path) and not any(p.endswith('_1') for p in path)


def spatial_objective(graph,pgraph):
    import flax.nnx as nnx
    from openpi.models.pi0_align import Pi0Align
    def objective(active,frozen,projector,rng,obs,actions,targets,mask):
        model=nnx.merge(graph,active,frozen)
        model.align_projector=nnx.merge(pgraph,projector)
        model.vla_layers_align=12;model.vla_adapt_layers_align=False
        model.align_loss_coeff=.5
        _,_,alignment=Pi0Align.compute_align_losses(model,rng,obs,actions,targets,mask,train=False)
        return alignment
    return objective


class SpatialEngine(Engine):
    def __init__(self,spec,output):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax
        from openpi.models.pi0_align import Pi0Align
        super().__init__(spec,output)
        recipe=read_pin(spec['spatial_recipe'])
        self.alpha=0. if spec['operation']=='QUALIFY_SPATIAL_ALPHA0' else .5
        if (recipe['alpha']!=.5 or recipe['vla_layers_align']!=12 or recipe['vggt_layers_align']!=-1
            or recipe['vla_adapt_layers_align'] or recipe['vggt_adapt_layers_align']
            or not recipe['use_vlm_norm'] or recipe['use_camera_params'] or recipe['teacher_gradient']):
            raise ValueError('EXACT_FIXED_SPATIAL_RECIPE_REQUIRED')
        self.bank=FeatureBank(spec['teacher_cache'],source_dataset_sha256=self.dataset.original.manifest_sha,
                              target_audit_sha256=spec['target_audit']['sha256'])
        self.projector_graph,self.projector,manifest=load_projector(spec['projector'],spec['parent_manifest'],recipe)
        self.counts.update(projector_asset_loads=1,teacher_model_loads=0,spatial_cache_reads=0,
                           spatial_backward_attempts=0,spatial_backwards_completed=0,
                           diagnostic_reference_FM_backward_calls=0,diagnostic_reference_optimizer_updates=0,
                           diagnostic_reference_training_visits=0,alpha0_gradient_pairs_compared=0,
                           alpha0_update_pairs_compared=0)
        self.bindings.update(recipe='S2_NATIVE_FM32_PLUS_FIXED_SPATIAL_ALPHA_HALF_V1',
            spatial_recipe=spec['spatial_recipe'],teacher_cache=spec['teacher_cache'],projector_initialization=spec['projector'],
            alpha=self.alpha,gradient_execution='SEQUENTIAL_GPU_BACKWARDS_LINEARITY_SUM_GLOBAL_CLIP_ADAMW')
        (self.output/'SPATIAL_PARAMETERS.private.json').write_text(json.dumps(dict(
            projector_arrays=manifest['arrays'],inherited_projector_numel=sum(r['numel']for r in manifest['arrays']),
            common_parent_initialization_unchanged=True,teacher_weights_loaded=False),indent=2))
        self.reference_apply=self.apply
        self.reference_active=self.reference_opt=None
        if self.alpha==0:
            self.reference_active=jax.tree.map(lambda a:jnp.array(a,copy=True),self.active)
            self.reference_opt=self.opt_state
        self.opt_state=self.tx.init((self.active,self.projector))
        graph,pgraph=self.graph,self.projector_graph
        self.spatial_grad=jax.jit(jax.value_and_grad(spatial_objective(graph,pgraph),argnums=(0,2)))
        self.combine=jax.jit(lambda common,space:jax.tree.map(lambda f,s:f+self.alpha*s,common,space))
        def apply(active,projector,optimizer,gradients):
            updates,optimizer=self.tx.update(gradients,optimizer,(active,projector))
            active,projector=optax.apply_updates((active,projector),updates)
            return active,projector,optimizer,optax.global_norm(gradients)
        self.apply_spatial=jax.jit(apply,donate_argnums=(0,1,2))
        self.exact=jax.jit(lambda a,b:jax.tree_util.tree_reduce(jnp.logical_and,
            jax.tree.map(jnp.array_equal,a,b),initializer=jnp.array(True)))
        self.projector_gradient_scale=jax.jit(lambda p:jax.tree.map(lambda g:g*self.alpha,p))

    def checkpoint(self,name):
        return save(self.output/name,(self.active,self.opt_state,self.rng,self.projector),
                    step=self.step,cursor=self.cursor,bindings=self.bindings)

    def resume(self,pin):
        if sha(Path(pin['path'])/'COMPLETE.json')!=pin['sha256']:
            raise ValueError('S2_COMPLETE_STATE_PIN_CHANGED')
        state,self.step,self.cursor=restore(pin['path'],(self.active,self.opt_state,self.rng,self.projector),bindings=self.bindings)
        self.active,self.opt_state,self.rng,self.projector=state

    def residency(self):
        import jax
        value=super().residency()
        leaves=jax.tree.leaves(self.projector)
        if not all(all(d.platform=='gpu'for d in a.devices())for a in leaves):
            raise ValueError('S2_PROJECTOR_NOT_GPU_RESIDENT')
        value['training_state_leaves']+=len(leaves)
        value.update(projector_GPU_resident=True,teacher_weights_loaded=False,
                     spatial_targets_are_frozen_input_cache=True)
        return value

    def update(self):
        import jax
        import jax.numpy as jnp
        import optax
        from openpi.training.utils import train_rng
        started=time.time();logs=[]
        accumulator=jax.tree.map(jnp.zeros_like,self.active)
        projector_accumulator=jax.tree.map(jnp.zeros_like,self.projector)
        reference_accumulator=jax.tree.map(jnp.zeros_like,self.reference_active) if self.alpha==0 else None
        for micro in range(32):
            cursor=self.cursor+micro;index=int(self.visit_map[cursor])
            drop=bool(np.random.Generator(np.random.PCG64(SEED+1000000+cursor)).random()<.25)
            item=self.dataset[index]
            obs,actions=observation(item,self.tokenizer,drop_state=drop)
            targets,mask=self.bank.for_index(index,item);self.counts['spatial_cache_reads']+=1
            key=train_rng(self.rng,self.step,micro,32)
            self.counts['forward_backward_attempts']+=1;self.counts['attempted_training_visits']+=1
            fm,common_gradient=self.grad(self.active,self.frozen,key,obs,actions)
            jax.block_until_ready(common_gradient)
            self.counts['completed_forward_backward']+=1;self.counts['completed_training_visits']+=1
            if not np.isfinite(float(fm)):
                raise ValueError('FINITE_S2_ORIGINAL_FM_REQUIRED')
            if self.alpha==0:
                reference_loss,reference_gradient=self.grad(self.reference_active,self.frozen,key,obs,actions)
                jax.block_until_ready(reference_gradient)
                self.counts['diagnostic_reference_FM_backward_calls']+=1
                self.counts['diagnostic_reference_training_visits']+=1
                if not bool(self.exact(common_gradient,reference_gradient)) or not np.array_equal(np.asarray(fm),np.asarray(reference_loss)):
                    raise ValueError('ALPHA0_COMMON_GRADIENT_NOT_BIT_EXACT_TO_INDEPENDENT_S1_STATE')
                self.counts['alpha0_gradient_pairs_compared']+=1
                reference_accumulator=self.accumulate(reference_accumulator,reference_gradient)
                del reference_gradient
            self.counts['spatial_backward_attempts']+=1
            align,(spatial_common,spatial_projector)=self.spatial_grad(
                self.active,self.frozen,self.projector,key,obs,actions,targets,mask)
            jax.block_until_ready((spatial_common,spatial_projector))
            self.counts['spatial_backwards_completed']+=1
            if not np.isfinite(float(align)) or not np.isfinite(float(optax.global_norm((spatial_common,spatial_projector)))):
                raise ValueError('FINITE_FROZEN_SPATIAL_GRADIENT_REQUIRED')
            if micro==0 and self.step<2:
                norms=jax.tree.map(lambda g:jnp.linalg.norm(g),spatial_common)
                actual={('/'.join(p)):float(v)for p,v in flat(norms.to_pure_dict()).items()}
                required=[v for p,v in flat(norms.to_pure_dict()).items()
                          if vlm_lora(p) and (p[-1].endswith('lora_b') or self.step>0)]
                if len(required)!=(5 if self.step==0 else 10) or not all(float(v)>0 for v in required):
                    raise ValueError('SPATIAL_LOSS_MUST_REACH_DEPLOYED_VLM_LORA_NOT_ONLY_PROJECTOR')
                logs.append(dict(first_micro_spatial_common_gradient_norms=actual,
                                 first_micro_projector_gradient_norm=float(optax.global_norm(spatial_projector))))
            # At alpha0 this is the exact original gradient object, including
            # signed zeros, while the independent spatial backward is retained.
            combined=common_gradient if self.alpha==0 else self.combine(common_gradient,spatial_common)
            projector_gradient=self.projector_gradient_scale(spatial_projector)
            accumulator=self.accumulate(accumulator,combined)
            projector_accumulator=self.accumulate(projector_accumulator,projector_gradient)
            active=padding=None
            if micro==0 and (self.step<2 or (self.step+1)%50==0):
                diagnostic,(a,p)=self.diagnostics(self.active,self.frozen,key,obs,actions)
                jax.block_until_ready(diagnostic)
                if abs(float(diagnostic)-float(fm))>read_pin(self.spec['initial_function_rule'])['diagnostic_forward_decomposition_max_loss_abs']:
                    raise ValueError('S2_NATIVE_FM_DIMENSION_DIAGNOSTIC_DIFFERS')
                active,padding=float(a),float(p)
                self.counts['diagnostic_forward_calls']=self.counts.get('diagnostic_forward_calls',0)+1
                self.counts['diagnostic_TRAIN_visits']=self.counts.get('diagnostic_TRAIN_visits',0)+1
            logs.append(dict(loss=float(fm),spatial_loss=float(align),joint_loss=float(fm)+self.alpha*float(align),
                             active7=active,padding25=padding,drop_state=drop))
        self.active,self.projector,self.opt_state,norm=self.apply_spatial(
            self.active,self.projector,self.opt_state,(accumulator,projector_accumulator))
        jax.block_until_ready((self.active,self.projector))
        if self.alpha==0:
            self.reference_active,self.reference_opt,_=self.reference_apply(
                self.reference_active,self.reference_opt,reference_accumulator)
            self.counts['diagnostic_reference_optimizer_updates']+=1
            if not bool(self.exact((self.active,common_optimizer(self.opt_state),self.rng),
                                   (self.reference_active,self.reference_opt,self.rng))):
                raise ValueError('ALPHA0_COMMON_PARAMETER_ADAM_RNG_UPDATE_NOT_BIT_EXACT')
            self.counts['alpha0_update_pairs_compared']+=1
        self.step+=1;self.cursor+=32;self.counts['optimizer_updates']+=1
        row=dict(step=self.step,cursor=self.cursor,alpha=self.alpha,used_lr=learning_rates()[self.step-1],
                 next_lr=learning_rates()[self.step] if self.step<500 else None,
                 grad_norm_before_clip=float(norm),micro=logs,wall_s=time.time()-started,
                 common_reference_bit_exact=self.alpha==0,GPU_memory=jax.devices()[0].memory_stats())
        with(self.output/'updates.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        return row
