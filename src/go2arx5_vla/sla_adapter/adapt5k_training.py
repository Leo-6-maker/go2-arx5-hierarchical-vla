"""A separate ADAPT5K engine with inherited SCALE parameters, Adam and RNG."""
import time

import numpy as np

from . import adapt5k_contract as contract
from .scale20k_contract import DOMAINS, make_adamw
from .scale20k_training import Engine

OPTIMIZER = dict(beta1=.9,beta2=.95,eps=1e-8,weight_decay=1e-5,clip_norm=1.)


class AdaptEngine(Engine):
    def __init__(self, graphs, parameters, frozen, datasets, banks, tokenizer, microbatch, *,
                 restored_adam, restored_rng, parent_step, phase_step, stage_start, stage_stop):
        import jax

        groups = {'shared', *DOMAINS}
        if (set(parameters) != groups or set(restored_adam) != groups
                or any(set(values) != set(DOMAINS) for values in (graphs, datasets, banks, microbatch))
                or (stage_start,stage_stop) not in ((0,2000),(0,5000),(2000,5000))
                or not stage_start <= phase_step < stage_stop
                or any(type(size) is not int or size <= 0 or contract.BATCH[d] % size
                       for d,size in microbatch.items())
                or any(bank.shape != ((stage_stop-stage_start)*contract.BATCH[d],)
                       or bank.dtype.kind != 'i' or bank.min() < 0 or bank.max() >= len(datasets[d])
                       for d,bank in banks.items())):
            raise ValueError('ADAPT5K_REQUIRES_RESTORED_ADAM_AND_EXACT_BOUND_PHASE_BANKS')
        contract.progress(parent_step,phase_step,stage_start)
        self.parameters,self.frozen,self.states,self.rng = parameters,frozen,restored_adam,restored_rng
        self.datasets,self.banks,self.tokenizer = datasets,banks,tokenizer
        self.batch_size,self.microbatch = contract.BATCH,microbatch
        self.parent_step,self.step,self.stage_start,self.stage_stop = parent_step,phase_step,stage_start,stage_stop
        self.rates = contract.phase_rates()
        self._setup_gradient_functions(graphs,self.batch_size)
        _,self.apply = make_adamw(**OPTIMIZER)
        self.combine = jax.jit(lambda fm,aux:contract.joint_gradient(fm,aux,tree_map=jax.tree.map))

    def domain_gradient(self, domain, guard):
        import jax
        import jax.numpy as jnp
        import optax

        clocks = contract.progress(self.parent_step,self.step,self.stage_start)
        local, absolute = (clocks[key][domain] for key in ('stage_local_cursors','lineage_cursors'))
        indices = self.banks[domain][local:local+self.batch_size[domain]]
        active = dict(shared=self.parameters['shared'],private=self.parameters[domain])
        fm,metrics = self._fm_gradient(domain,indices,absolute,active,guard)
        profile = self.update_breakdown
        if profile is None:
            self._record_gradient(fm,metrics,indices)
        else:
            profile.run(f'{domain}/FM_gradient_norms',lambda:self._record_gradient(fm,metrics,indices))
        aux = None
        if domain == 'target':
            auxiliary = jax.tree.map(jnp.zeros_like,active)
            slots = contract.auxiliary_slots(self.step)
            aux = self._add_target_auxiliary(indices,absolute,active,auxiliary,metrics,slots,guard)
            def auxiliary_norms():
                return float(optax.global_norm(aux['shared'])),float(optax.global_norm(aux['private']))

            norms = profile.run('target/H40_gradient_norms',auxiliary_norms) if profile else auxiliary_norms()
            metrics['auxiliary_shared_gradient_norm'],metrics['auxiliary_private_gradient_norm'] = norms
            if not all(np.isfinite(metrics[key]) for key in ('auxiliary','auxiliary_shared_gradient_norm',
                                                            'auxiliary_private_gradient_norm')):
                raise ValueError('ADAPT5K_AUXILIARY_REQUIRES_FINITE_TARGET_GRADIENTS')
            metrics['auxiliary_slots'] = slots
        return fm,aux,metrics

    def measure_gradient(self, guard):
        """Measure the complete real pass. The caller needs a separate effect to apply Adam."""
        if self.step >= self.stage_stop:
            raise ValueError('ADAPT5K_REQUIRES_COLD_HANDOFF_AT_THE_STAGE_BOUNDARY')
        fm,aux,metrics = {},None,{}
        for d in DOMAINS:
            fm[d],term,metrics[d] = self.domain_gradient(d,guard)
            if d == 'target':
                aux = term
        profile = self.update_breakdown
        gradient = profile.run('ADAPT/domain_gradient_merge',lambda:self.combine(fm,aux),wait=True) if profile else self.combine(fm,aux)
        return gradient,metrics

    def update(self, guard):
        import jax

        before = time.monotonic()
        gradient,metrics = self.measure_gradient(guard)
        guard()
        lr = float(self.rates[self.step])
        profile = self.update_breakdown
        if profile is None:
            parameters,states,norm = self.apply(self.parameters,self.states,gradient,lr)
            jax.block_until_ready((parameters,states))
        else:
            parameters,states,norm = profile.run('ADAPT/global_clip_AdamW',
                lambda:self.apply(self.parameters,self.states,gradient,lr),wait=True)
        self.parameters,self.states = parameters,states
        self.step += 1
        row = dict(**contract.progress(self.parent_step,self.step,self.stage_start),learning_rate=lr,
                   grad_norm=norm,domains=metrics,wall_s=time.monotonic()-before,
                   actual_GPU_memory_stats=jax.devices()[0].memory_stats())
        if profile is not None:
            row['synchronized_intervals_s'] = profile.seconds.copy()
            row['unattributed_wall_s'] = row['wall_s']-sum(profile.seconds.values())
        return row
