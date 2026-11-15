"""Matched dense FM/spatial seam, with frozen targets staged to GPU once."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .runtime_v4 import DensePilotEngine
from .runtime_v1 import read_pin
from .spatial_runtime_v1 import spatial_objective, common_optimizer, vlm_lora
from .spatial_assets_v1 import load_projector, FeatureBank
from .batch_v2 import prepared_batches, device_batch
from .checkpoint_v1 import save, restore, sha
from .model_v1 import flat

RECIPE = 'SLA_NAMED_DENSE_PICK_HOLD_SF32_3K_ALPHA_HALF_V5'
OPERATION = 'DENSE_SPATIAL_PICK_HOLD_3K_V5'

class DenseFeatureBank(FeatureBank):
    def __init__(self, pin, dataset, audit):
        self.root = Path(pin['path'])
        self.manifest = read_pin(dict(path=str(self.root/'MANIFEST.private.json'),sha256=pin['sha256']))
        m = self.manifest
        if (m['schema'] != 'SLA_FROZEN_OMNIVGGT_NAMED_DENSE_PICK_HOLD_CACHE_V4_1'
                or not m['passes'] or m['source_dataset_sha256'] != dataset.manifest_sha
                or m['target_audit'] != audit or len(m['rows']) != 122
                or len(dataset) != 122 or m['teacher_gradient'] or m['privileged_inputs_used']
                or m['source_camera_count'] != 2 or m['teacher_last_layer'] != -1):
            raise ValueError('EXACT_FROZEN_122_WINDOW_CACHE_FOR_CURRENT_DENSE_TARGET')
        self.rows, self.reads = m['rows'], 0
        import jax
        features, masks = [], []
        for i,row in enumerate(self.rows):
            identity = dataset.identity(i)
            if row['episode_id'] != identity['episode_id'] or row['anchor'] != identity['anchor']:
                raise ValueError('SPATIAL_CACHE_ANCHOR_IDENTITY_CHANGED')
            # Reuse exact pixel/mask/feature validation, on CPU during staging.
            # for_index returns device arrays; one-time reads are explicit and
            # update() never transfers these feature tensors back to the host.
            feature, mask = super().for_index(i,dataset[i])
            features.append(feature[0]);masks.append(mask[0])
        self.features = jax.numpy.stack(features)
        self.masks = jax.numpy.stack(masks)
        jax.block_until_ready((self.features,self.masks))
        self.feature_bytes = self.features.nbytes+self.masks.nbytes
        self.gather = jax.jit(lambda features,masks,indices:(features[indices],masks[indices]))

    def batch(self,indices):
        import jax.numpy as jnp
        return self.gather(self.features,self.masks,jnp.asarray(indices,dtype=jnp.int32))


class DenseSpatialEngine(DensePilotEngine):
    def __init__(self,spec,output):
        super().__init__(spec,output)
        import jax
        import jax.numpy as jnp
        import optax
        self.alpha = spec['spatial_alpha']
        recipe = read_pin(spec['spatial_recipe'])
        if (self.alpha not in (0.,.5) or recipe['alpha'] != .5 or recipe['vla_layers_align'] != 12
                or recipe['vggt_layers_align'] != -1 or recipe['vla_adapt_layers_align']
                or recipe['vggt_adapt_layers_align'] or not recipe['use_vlm_norm']
                or recipe['use_camera_params'] or recipe['teacher_gradient']):
            raise ValueError('FIXED_INHERITED_SPATIAL_OBJECTIVE_REQUIRED')
        self.bank = DenseFeatureBank(spec['teacher_cache'],self.dataset,spec['target_audit'])
        self.projector_graph,self.projector,_ = load_projector(spec['projector'],spec['parent_manifest'],recipe)
        self.reference_apply = self.apply
        if self.alpha == 0:
            self.reference_active = jax.tree.map(lambda a:jnp.array(a,copy=True),self.active)
            self.reference_opt = jax.tree.map(lambda a:jnp.array(a,copy=True),self.opt_state)
        self.opt_state = self.tx.init((self.active,self.projector))
        self.spatial_grad = jax.jit(jax.value_and_grad(spatial_objective(self.graph,self.projector_graph),argnums=(0,2)))
        self.combine = jax.jit(lambda fm,spatial:jax.tree.map(lambda f,s:f+self.alpha*s,fm,spatial))
        self.scale = jax.jit(lambda values:jax.tree.map(lambda a:a*self.alpha,values))
        def apply(active,projector,state,grad):
            updates,state = self.tx.update(grad,state,(active,projector))
            active,projector = optax.apply_updates((active,projector),updates)
            return active,projector,state
        self.apply_spatial = jax.jit(apply,donate_argnums=(0,1,2))
        self.exact = jax.jit(lambda a,b:jax.tree_util.tree_reduce(jnp.logical_and,
            jax.tree.map(jnp.array_equal,a,b),initializer=jnp.array(True)))
        self.spatial_norm = jax.jit(lambda gradients:jnp.stack([jnp.linalg.norm(a)
            for p,a in flat(gradients.to_pure_dict()).items() if vlm_lora(p) and p[-1].endswith('lora_b')]))
        self.norm = jax.jit(optax.global_norm)
        self.bindings.update(recipe=RECIPE if spec['operation'] == OPERATION else 'SLA_NAMED_DENSE_SPATIAL_SEAM_V5',alpha=self.alpha,
            spatial_recipe=spec['spatial_recipe'],teacher_cache=spec['teacher_cache'],projector_initialization=spec['projector'])
        self.counts.update(teacher_model_loads=0,projector_asset_loads=1,spatial_cache_staging_reads=122,
            spatial_cache_staging_H2D_bytes=self.bank.feature_bytes,spatial_cache_training_visits=0,
            spatial_backwards_completed=0,reference_FM_visits=0,alpha0_gradient_pairs_compared=0,
            alpha0_update_pairs_compared=0,reference_optimizer_updates=0)
        self.counts.update(input_H2D_batches=0,input_H2D_bytes=0,index_H2D_bytes=0,
            rng_cursor_H2D_bytes=0,small_metric_D2H_calls=0,small_metric_D2H_bytes=0)

    def checkpoint(self,name):
        return save(self.output/name,(self.active,self.opt_state,self.rng,self.projector),
            step=self.step,cursor=self.cursor,bindings=self.bindings)

    def resume(self,pin):
        if sha(Path(pin['path'])/'COMPLETE.json') != pin['sha256']:
            raise ValueError('SPATIAL_COMPLETE_STATE_PIN_CHANGED')
        state,self.step,self.cursor = restore(pin['path'],
            (self.active,self.opt_state,self.rng,self.projector),bindings=self.bindings)
        self.active,self.opt_state,self.rng,self.projector = state
        if self.alpha == 0:
            import jax
            self.reference_active = jax.tree.map(lambda a:jax.numpy.array(a,copy=True),self.active)
            self.reference_opt = jax.tree.map(lambda a:jax.numpy.array(a,copy=True),common_optimizer(self.opt_state))

    def residency(self):
        import jax
        value = super().residency()
        if not all(all(d.platform == 'gpu' for d in a.devices()) for a in
                   jax.tree.leaves((self.projector,self.bank.features,self.bank.masks))):
            raise ValueError('SPATIAL_PROJECTOR_AND_FEATURE_BANK_MUST_REMAIN_ON_GPU')
        return dict(value,projector_GPU_resident=True,frozen_target_bank_GPU_resident=True,
                    frozen_target_bank_bytes=self.bank.feature_bytes,teacher_weights_loaded=False)

    def update(self):
        import jax
        import jax.numpy as jnp
        began = time.time()
        accumulator = jax.tree.map(jnp.zeros_like,self.active)
        projection = jax.tree.map(jnp.zeros_like,self.projector)
        reference = jax.tree.map(jnp.zeros_like,self.reference_active) if self.alpha == 0 else None
        metrics = jnp.zeros((3,),dtype=jnp.float32)
        noise,flow = self.draw_bank(self.rng,np.arange(self.cursor,self.cursor+32,dtype=np.uint32))
        self.counts['rng_cursor_H2D_bytes'] += 32*4
        logs = []
        batches = prepared_batches(self.dataset,self.tokenizer,self.visit_map,self.cursor,self.microbatch,
            prefetch=True,executor=self.loader,on_read=lambda k:None)
        try:
            for host in batches:
                if time.time() >= self.spec['deadline_epoch']:raise TimeoutError('FINITE_SPATIAL_SEAM_BEFORE_UPDATE')
                obs,actions = device_batch(host)
                self.counts['input_H2D_batches'] += 1
                self.counts['input_H2D_bytes'] += sum(a.nbytes for a in
                    jax.tree.leaves((host['observation'],host['actions'])))
                self.counts['index_H2D_bytes'] += 4*len(host['indices'])
                offset = int(host['cursors'][0])-self.cursor
                targets,mask = self.bank.batch(host['indices'])
                self.counts['attempted_training_visits'] += self.microbatch
                self.counts['forward_backward_attempts'] += 1
                with jax.default_matmul_precision(self.training_precision):
                    (fm,parts),gradient = self.grad(self.active,self.frozen,obs,actions,
                        noise[offset:offset+self.microbatch],flow[offset:offset+self.microbatch])
                    if self.alpha == 0:
                        (rfm,rparts),rg = self.grad(self.reference_active,self.frozen,obs,actions,
                            noise[offset:offset+self.microbatch],flow[offset:offset+self.microbatch])
                        if not bool(self.exact((gradient,fm,parts),(rg,rfm,rparts))):
                            raise ValueError('ALPHA0_DENSE_COMMON_FM_GRADIENT_NOT_BIT_EXACT')
                        reference = self.accumulate(reference,rg)
                        self.counts['reference_FM_visits'] += self.microbatch
                        self.counts['alpha0_gradient_pairs_compared'] += 1
                    align,(common,projector) = self.spatial_grad(self.active,self.frozen,self.projector,
                        jax.random.fold_in(self.rng,self.cursor+offset),obs,actions,targets,mask)
                if offset == 0:
                    norms = np.asarray(self.spatial_norm(common))
                    self.counts['small_metric_D2H_calls'] += 1
                    self.counts['small_metric_D2H_bytes'] += norms.nbytes
                    if norms.shape != (5,) or not np.isfinite(norms).all() or not (norms>0).all():
                        raise ValueError('SPATIAL_GRADIENT_MUST_REACH_ALL_FIVE_DEPLOYED_VLM_LORA_B_LEAVES')
                    logs.append(dict(VLM_LoRA_B_spatial_gradient_norms=norms.tolist()))
                accumulator = self.accumulate(accumulator,gradient if self.alpha == 0 else self.combine(gradient,common))
                projection = self.accumulate(projection,self.scale(projector))
                metrics = self.add_metrics(metrics,(fm,align,fm+self.alpha*align))
                self.counts['completed_forward_backward'] += 1
                self.counts['completed_training_visits'] += self.microbatch
                self.counts['spatial_backwards_completed'] += 1
                self.counts['spatial_cache_training_visits'] += self.microbatch
        finally:batches.close()
        summary = np.asarray(self.summarize(metrics,(accumulator,projection)))
        self.counts['small_metric_D2H_calls'] += 1
        self.counts['small_metric_D2H_bytes'] += summary.nbytes
        if not np.isfinite(summary).all():raise ValueError('NONFINITE_DENSE_SPATIAL_LOSS_OR_GRADIENT_NO_UPDATE')
        self.active,self.projector,self.opt_state = self.apply_spatial(self.active,self.projector,
            self.opt_state,(accumulator,projection))
        jax.block_until_ready((self.active,self.projector,self.opt_state))
        if self.alpha == 0:
            self.reference_active,self.reference_opt,_ = self.reference_apply(self.reference_active,self.reference_opt,reference)
            if not bool(self.exact((self.active,common_optimizer(self.opt_state)),(self.reference_active,self.reference_opt))):
                raise ValueError('ALPHA0_DENSE_COMMON_ADAM_UPDATE_NOT_BIT_EXACT')
            self.counts['reference_optimizer_updates'] += 1
            self.counts['alpha0_update_pairs_compared'] += 1
        self.step += 1;self.cursor += 32;self.counts['optimizer_updates'] += 1
        row=dict(step=self.step,cursor=self.cursor,alpha=self.alpha,mean_FM32_loss=float(summary[0]),
            spatial_loss=float(summary[1]),joint_loss=float(summary[2]),grad_norm_before_clip=float(summary[3]),
            used_lr=self.learning_rate_values[self.step-1],wall_s=time.time()-began,diagnostic=logs,
            GPU_memory=jax.devices()[0].memory_stats(),feature_D2H_bytes=0)
        with(self.output/'updates.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        return row
