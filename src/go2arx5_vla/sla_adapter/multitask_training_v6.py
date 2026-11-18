"""Matched official-base G7 training with four differentiable H40 samples/update."""
import hashlib
from pathlib import Path

import numpy as np

from . import batch_v2, dense_g7_v6, full_h40_v6
from .checkpoint_v1 import sha
from .config_v1 import SEED
from .mfree_training_v1 import learning_rates
from .multitask_contract_v6 import RECIPE
from .multitask_data_v6 import MultitaskDataset, visit_bank
from .multitask_initialization_v6 import sealed_source, transplant
from .target_training_v6 import FullTrajectoryEngine, PLAN

class MultitaskEngine(FullTrajectoryEngine):
    auxiliary_metric = 'weighted_G7_XYZ_and_speed_loss'
    recipe = RECIPE
    source_arm = 'M-SRC'

    def population_binding(self):
        return {'compiled_population_sha256': self.dataset.manifest['compiled_population_sha256']}

    def dataset_for_spec(self, spec):
        from openpi_client.image_tools import resize_with_pad

        pin = spec['multitask_dataset']
        audit_file = Path(pin['path']).parent/'ACTUAL.private.json'
        if (spec['plan_commit'] != PLAN or spec['operation'] != RECIPE or spec['arm'] not in ('M-BASE','M-SRC')
                or spec['microbatch'] != 4 or spec['training_matmul_precision'] != 'highest'
                or spec['planned_candidate_updates'] != 1000 or spec['new_optimizer_updates_budget'] != 1000
                or spec['new_training_visits_budget'] != 32000 or spec['new_physical_requests_budget']
                or spec['physics'] or spec['inference_route'] != 'cached_actual'
                or spec['aux_samples_per_update'] != 4 or spec['aux_sampling_steps'] != 10
                or bool(spec.get('S1000_transfer_result')) != (spec['arm'] == 'M-SRC')
                or any(spec.get(k) for k in ('B1_parent_package','action_representation_audit','resume','continuation'))
                or spec['target_audit'] != dict(path=str(audit_file), sha256=sha(audit_file))
                or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']):
            raise ValueError('EXACT_MATCHED_V6_G7_1000_AND_NO_SHIFT7_PARENT_REQUIRED')
        return MultitaskDataset(pin, resize_with_pad=resize_with_pad, cache_bytes=spec['frame_cache_bytes'])

    def visits_for_dataset(self, dataset):
        values, self.visit_bank_binding = visit_bank(dataset)
        return values

    def learning_rate_values_for_spec(self, spec):
        values = learning_rates()
        digest = hashlib.sha256(''.join(rate.hex()+'\n' for rate in values).encode()).hexdigest()
        if digest != spec['learning_rates_sha256']:
            raise ValueError('MATCHED_ORIGINAL_1000_UPDATE_LR_BANK_REQUIRED')
        return values

    def __init__(self, spec, output):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi.models import pi0
        from scripts.sla.close_goal_g1_v1 import make_sampler
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

        self.transfer_rows, self.transfer_receipt = [], None
        if spec['arm'] == self.source_arm:
            self.transfer_rows, self.transfer_receipt = sealed_source(spec['S1000_transfer_result'], spec['parent_manifest'])
        super().__init__(spec, output)
        if self.inventory['trainable_numel'] != 52153376 or len(jax.tree.leaves(self.active)) != 28:
            raise ValueError('EXACT_ORIGINAL_SHARED_TARGET_PARAMETER_RANGE_REQUIRED')
        if self.transfer_rows:
            self.active, transplant_receipt = transplant(self.active, self.transfer_rows)
            self.transfer_receipt.update(transplant_receipt)
            self.opt_state = self.tx.init(self.active)
        self.bindings.update(recipe=self.recipe, goal_plan_sha=spec['plan_commit'], codec_version=dense_g7_v6.VERSION,
            arm=spec['arm'], dataset_manifest_sha256=self.dataset.manifest_sha,
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],
            visit_bank=self.visit_bank_binding, source_initialization=self.transfer_receipt,
            full_XYZ_lambda=.1, full_gripper_speed_lambda=.1,
            grip_speed_margin_m_s=dense_g7_v6.GRIP_MARGIN, grip_speed_limit_m_s=dense_g7_v6.GRIP_LIMIT,
            aux_samples_per_update=4, full_sampling_steps=10, aux_gradient_rematerialization=True,
            inference_route='cached_actual', state_token_dropout=.25, fresh_Adam=True,
            parent_host_cast_v7=spec.get('parent_host_cast_v7',False),
            BASE_head=False, teacher_memory_input=False, object_truth_input=False, phase_input=False)
        self.bindings.update(self.population_binding())
        graph = self.graph
        fields, _ = extracted_fields(batch_v2, pi0)

        def fm(active, frozen, obs, actions, noise, tau):
            with jax.default_matmul_precision('highest'):
                t = tau[:,None,None]
                field = fields['train_field'](nnx.merge(graph,active,frozen), obs, t*noise+(1-t)*actions, tau)
                error = (field-(noise-actions))**2
                return jnp.mean(error), (jnp.mean(error[...,:7]), jnp.mean(error[...,7:]))

        self.grad = jax.jit(jax.value_and_grad(fm, has_aux=True))
        self.sample = full_h40_v6.make_sampler(graph, fields, rematerialize=True, precision=None)
        native_sampler = make_sampler(graph, fields, cached=True, precision=None)
        self.infer = lambda active, frozen, obs, noise: native_sampler(active, frozen, obs, noise)[0]
        quantiles = self.dataset.statistics['actions']
        low, high = jnp.asarray(quantiles.q01,jnp.float32), jnp.asarray(quantiles.q99,jnp.float32)

        def objective(active, frozen, obs, noise, actual_previous_width):
            generated = self.sample(active, frozen, obs, noise)
            return dense_g7_v6.full_objective(generated, low, high, actual_previous_width, xp=jnp, sigmoid=jax.nn.sigmoid)

        self.aux_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
        self.aux_noise = jax.jit(lambda cursor: jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(SEED+7000000),cursor), (1,40,32)))
        self.add_aux = jax.jit(lambda total, grad: jax.tree.map(lambda a,b: a+b/4,total,grad), donate_argnums=(0,))
        self.add_fm_metrics = self.add_metrics
        self.gradient_norm = jax.jit(__import__('optax').global_norm)

    def _aux_four(self, obs, first_cursor, accumulator, *, diagnose=False):
        import jax
        import jax.numpy as jnp
        from scripts.pi0292.durable import write

        records = []
        key = 'diagnostic' if diagnose else 'training'
        for i in range(4):
            cursor = first_cursor+i
            index = int(self.visit_map[cursor])
            context = self.dataset.reference(index)
            width = dense_g7_v6.previous_command_context(context)
            item = jax.tree.map(lambda array:array[i:i+1], obs)
            noise = self.aux_noise(np.uint32(cursor))
            for counter in (key+'_full_generation_attempts', key+'_full_trajectory_backward_attempts'):
                self.counts[counter] = self.counts.get(counter, 0)+1
            write(self.output/'AUX_PROGRESS.private.json', dict(phase='BEFORE_G7_AUX', step=self.step,
                cursor=cursor, diagnostic=diagnose, actual=self.counts))
            (loss, detail), gradient = self.aux_gradient(self.active, self.frozen, item, noise, jnp.array([width],jnp.float32))
            norm = self.gradient_norm(gradient)
            accumulator = self.add_aux(accumulator, gradient)
            jax.block_until_ready((loss, detail, norm, accumulator))
            metrics = [loss, detail['XYZ_dyn'], detail['gripper_speed'], detail['gripper']['max_speed'], norm]
            summary = np.asarray(metrics)
            if not np.isfinite(summary).all():
                raise ValueError('FINITE_REAL_G7_FULL_GENERATED_OBJECTIVE_REQUIRED')
            for counter in (key+'_full_generations', key+'_completed_full_trajectory_backwards'):
                self.counts[counter] = self.counts.get(counter, 0)+1
            row = dict(absolute_cursor=cursor, index=index, loss=float(summary[0]), XYZ_loss=float(summary[1]),
                gripper_speed_loss=float(summary[2]), maximum_gripper_speed_m_s=float(summary[3]),
                aux_gradient_norm=float(summary[4]), actual_previous_width_m=width)
            if diagnose:
                self.counts['diagnostic_sampler_comparison_attempts'] = self.counts.get('diagnostic_sampler_comparison_attempts',0)+2
                deployed = self.infer(self.active,self.frozen,item,noise)
                static = self.sample(self.active,self.frozen,item,noise)
                jax.block_until_ready((deployed,static))
                deployed, static = np.asarray(deployed), np.asarray(static)
                self.counts['diagnostic_sampler_comparison_generations'] = self.counts.get('diagnostic_sampler_comparison_generations',0)+2
                delta = float(np.max(abs(deployed-static)))
                if not np.isfinite(delta) or delta > 1e-5:
                    raise ValueError('M_G7_DIFFERENTIABLE_AND_DEPLOYED_SAMPLER_MISMATCH')
                row['sampler_exact_equal'] = bool(np.array_equal(deployed,static))
                row['sampler_max_abs_difference'] = delta
            records.append(row)
            write(self.output/'AUX_PROGRESS.private.json', dict(phase='COMPLETE_G7_AUX',step=self.step,
                cursor=cursor,diagnostic=diagnose,actual=self.counts,row=row))
        return accumulator, records

    def qualify_gradient(self):
        before = (self.active, self.opt_state, self.rng)
        host = batch_v2.host_batch(self.dataset, self.tokenizer, self.visit_map, 0, 4)
        observation, actions = batch_v2.device_batch(host)
        noise, tau = self.draw_bank(self.rng, np.arange(32,dtype=np.uint32))
        self.counts['diagnostic_FM_backward_attempts'] = 1
        (fm, _), gradient = self.grad(self.active,self.frozen,observation,actions,noise[:4],tau[:4])
        total, records = self._aux_four(observation, 0, gradient, diagnose=True)
        norm = float(self.gradient_norm(total))
        if not np.isfinite([norm,float(fm)]).all() or norm <= 0 or self.step or self.cursor:
            raise ValueError('REAL_FINITE_M_G7_GRADIENT_WITH_ZERO_UPDATES_REQUIRED')
        if any(a is not b for a,b in zip(before,(self.active,self.opt_state,self.rng),strict=True)):
            raise ValueError('G7_DIAGNOSTIC_CHANGED_TRAINING_STATE')
        self.counts['diagnostic_FM_backward_completed'] = 1
        return dict(gradient_norm=norm, FM=float(fm), aux=records, complete_full_generations=12,
            training_updates=0, optimization_visits=0, qualification_scope='MODEL_GRADIENT_AND_SAMPLER_EQUIVALENCE')
