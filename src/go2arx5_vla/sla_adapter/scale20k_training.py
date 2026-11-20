"""SCALE20K training core over three independently admitted native readers."""
import hashlib
import time

import numpy as np

from . import batch_v2, dense_g7_v6, full_h40_v6
from .dataset_v1 import CAMERAS
from .grasp_supervision_d5 import width_losses
from .scale20k_contract import BATCH, DOMAINS, DOMAIN_WEIGHT, HORIZONS, cursors, learning_rates, make_adamw


STATE_CHANNELS = dict(zip(DOMAINS,(7,14,8),strict=True))
ACTION_CHANNELS = dict(zip(DOMAINS,(7,14,7),strict=True))


def host_batch(dataset, tokenizer, domain, positions, indices, *, training=True):
    """Tokenize actual state dimensions and keep raw domain action semantics."""
    rows,token_ids,token_masks = [],[],[]
    for cursor,index in zip(positions,indices,strict=True):
        item = dataset[int(index)]
        state = item['state']
        if (state.shape != (32,) or item['actions'].shape != (HORIZONS[domain],32)
                or np.any(state[STATE_CHANNELS[domain]:]) or np.any(item['actions'][...,ACTION_CHANNELS[domain]:])
                or not np.isfinite(state).all() or not np.isfinite(item['actions']).all()):
            raise ValueError('SCALE20K_TYPED_NATIVE_DOMAIN_STATE_AND_ACTION_REQUIRED:'+domain)
        dropped = training and batch_v2.drop_state(int(cursor)+DOMAINS.index(domain)*1000000)
        state_tokens = None if dropped else state[:STATE_CHANNELS[domain]]
        tokens,mask = tokenizer.delegate.tokenize(item['prompt'],state_tokens,use_discrete_state_format=True)
        rows.append(item)
        token_ids.append(tokens)
        token_masks.append(mask)
    observations = dict(state=np.stack([row['state'] for row in rows]),tokenized_prompt=np.stack(token_ids),
                        tokenized_prompt_mask=np.stack(token_masks))
    for key in ('image','image_mask','image_padding_mask'):
        if any(set(row[key]) != set(CAMERAS) for row in rows):
            raise ValueError('SCALE20K_PINNED_ACTUAL_CAMERA_SLOTS_REQUIRED')
        observations[key] = {camera:np.stack([row[key][camera] for row in rows]) for camera in CAMERAS}
    return batch_v2.device_batch(dict(observation=observations,actions=np.stack([row['actions'] for row in rows])))


def frozen_draws(domain, size):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def generate(rng,cursor):
        domain_rng = jax.random.fold_in(rng,DOMAINS.index(domain))
        positions = jnp.asarray(cursor,dtype=jnp.uint32)+jnp.arange(size,dtype=jnp.uint32)
        keys = jax.vmap(lambda offset:jax.random.fold_in(domain_rng,offset))(positions)
        components = jax.vmap(lambda key:jax.random.split(key,2))(keys)
        noise = jax.vmap(lambda key:jax.random.normal(key,(HORIZONS[domain],32),dtype=jnp.float32))(components[:,0])
        tau = jax.vmap(lambda key:jax.random.beta(key,1.5,1,dtype=jnp.float32))(components[:,1])
        return noise,tau*.999+.001

    return generate


def domain_loss(graph):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    def loss(active, frozen, observation, actions, noise, tau):
        with jax.default_matmul_precision('highest'):
            model = nnx.merge(graph,active['shared'],active['private'],frozen)
            error = batch_v2.error32(model,observation,actions,noise,tau)
            return jnp.mean(error,dtype=jnp.float32)

    return jax.jit(jax.value_and_grad(loss))


def target_auxiliary(graph, quantiles, *, event_weighted=False):
    import jax
    import jax.numpy as jnp
    from flax.nnx import State
    from openpi.models import pi0
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    fields,_ = extracted_fields(batch_v2,pi0)
    sampler = full_h40_v6.make_sampler(graph,fields,rematerialize=True,precision='highest')
    lo,hi = (jnp.asarray(getattr(quantiles,key),jnp.float32) for key in ('q01','q99'))

    def loss(active,frozen,observation,noise,previous_width,label_width,weights=None):
        joined = State.merge(active['shared'],active['private'])
        prediction = sampler(joined,frozen,observation,noise)
        original,_ = dense_g7_v6.full_objective(prediction,lo,hi,previous_width,xp=jnp,sigmoid=jax.nn.sigmoid)
        latent = .5*(prediction[...,6]+1)*(hi[6]-lo[6]+1e-6)+lo[6]
        physical_width = .002+.040*jax.nn.sigmoid(latent)
        if event_weighted:
            from .clock500 import event_weighted_width_loss

            width = event_weighted_width_loss(physical_width,label_width,weights,xp=jnp)
        else:
            width,_ = width_losses(physical_width,label_width,previous_width,xp=jnp)
        return original+width

    return jax.jit(jax.value_and_grad(loss))


class Engine:
    profile_timing = False

    def __init__(self, arm, graphs, parameters, frozen, datasets, banks, tokenizer, microbatch, optimizer, *, rng):
        self.arm,self.parameters,self.frozen = arm,parameters,frozen
        self.datasets,self.banks,self.tokenizer = datasets,banks,tokenizer
        self.batch_size,self.rng,self.step = BATCH[arm],rng,0
        if (set(datasets) != set(DOMAINS) or set(banks) != set(DOMAINS) or set(microbatch) != set(DOMAINS)
                or any(type(size) is not int or size<=0 or (self.batch_size[d] and self.batch_size[d]%size)
                       for d,size in microbatch.items())
                or any(bank.shape != (20000*BATCH['SCALE-MIX20K'][d],) or bank.min() < 0 or bank.max() >= len(datasets[d])
                       for d,bank in banks.items())):
            raise ValueError('SCALE20K_PER_DOMAIN_FULL_BATCH_VISIT_BANK_REQUIRED')
        self.microbatch = microbatch
        self.rates = learning_rates()
        self._setup_gradient_functions(graphs,BATCH['SCALE-MIX20K'])
        initialize,self.apply = make_adamw(**optimizer)
        self.states = initialize(parameters)

    def _setup_gradient_functions(self, graphs, draw_sizes):
        import jax

        self.grad = {d:domain_loss(graph) for d,graph in graphs.items()}
        self.draw = {d:frozen_draws(d,draw_sizes[d]) for d in DOMAINS}
        self.aux = self._target_auxiliary(graphs['target'],self.datasets['target'].statistics['actions'])
        self.add = jax.jit(lambda total,part,weight:jax.tree.map(lambda a,b:a+b*weight,total,part),donate_argnums=(0,))
        self.attempted_aux = self.completed_aux = 0
        self.gradient_windows = dict.fromkeys(DOMAINS,0)

    def _target_auxiliary(self, graph, quantiles):
        return target_auxiliary(graph,quantiles)

    def _fm_gradient(self, domain, indices, cursor, active, guard):
        import jax
        import jax.numpy as jnp

        size,micro = self.batch_size[domain],self.microbatch[domain]
        draws = self.draw[domain](self.rng,cursor)
        dataset = self.datasets[domain]
        accumulator = jax.tree.map(jnp.zeros_like,active)
        totals = dict(FM32=0.,auxiliary=0.,backward_calls=0,training_full_generations=0,data_wait_s=0.)
        if self.profile_timing:
            totals.update(FM_backward_s=0.,auxiliary_input_s=0.,auxiliary_backward_s=0.)
        for offset in range(0,size,micro):
            guard()
            end = offset+micro
            started = time.monotonic()
            obs,actions = host_batch(dataset,self.tokenizer,domain,range(cursor+offset,cursor+end),indices[offset:end])
            if self.profile_timing:
                jax.block_until_ready((obs,actions,draws))
            input_ready = time.monotonic()
            totals['data_wait_s'] += input_ready-started
            value,grad = self.grad[domain](active,self.frozen,obs,actions,draws[0][offset:end],draws[1][offset:end])
            accumulator = self.add(accumulator,grad,micro/size)
            totals['FM32'] += float(value)*micro/size
            if self.profile_timing:
                jax.block_until_ready(accumulator)
                totals['FM_backward_s'] += time.monotonic()-input_ready
            totals['backward_calls'] += 1
            self.gradient_windows[domain] += micro
        return accumulator,totals

    def _add_target_auxiliary(self, indices, cursor, active, accumulator, totals, slots, guard):
        import jax
        import jax.numpy as jnp

        dataset = self.datasets['target']
        for slot in slots:
            guard()
            if self.profile_timing:
                started = time.monotonic()
            index = int(indices[slot]); absolute = cursor+slot
            obs,_ = host_batch(dataset,self.tokenizer,'target',[absolute],[index])
            episode,anchor = dataset.rows[index]
            label = dataset.labels[episode['id']]['nominal_policy_H40'][anchor,:,9]
            width = dense_g7_v6.previous_command_context(dataset.reference(index))
            noise = jax.random.normal(jax.random.fold_in(jax.random.fold_in(self.rng,37),absolute),(1,40,32))
            extra = self._training_only_auxiliary_inputs(dataset,index,width)
            width = jnp.asarray([width],jnp.float32)
            label_width = jnp.asarray((.002+.040*label)[None,:],jnp.float32)
            if self.profile_timing:
                jax.block_until_ready((obs,noise,width,label_width,extra))
                ready = time.monotonic()
                totals['auxiliary_input_s'] += ready-started
                started = ready
            self.attempted_aux += 1
            value,grad = self.aux(active,self.frozen,obs,noise,width,label_width,*extra)
            accumulator = self.add(accumulator,grad,.25)
            totals['auxiliary'] += float(value)*.25
            if self.profile_timing:
                jax.block_until_ready(accumulator)
                totals['auxiliary_backward_s'] += time.monotonic()-started
            self.completed_aux += 1
            totals['training_full_generations'] += 1
        return accumulator

    def _training_only_auxiliary_inputs(self, dataset, index, previous):
        return ()

    def domain_gradient(self, domain, guard):
        cursor = cursors(self.arm,self.step)[domain]
        indices = self.banks[domain][cursor:cursor+self.batch_size[domain]]
        active = dict(shared=self.parameters['shared'],private=self.parameters[domain])
        accumulator,totals = self._fm_gradient(domain,indices,cursor,active,guard)
        if domain=='target':
            accumulator = self._add_target_auxiliary(indices,cursor,active,accumulator,totals,full_h40_v6.aux_slots(self.step),guard)
        self._record_gradient(accumulator, totals, indices)
        return accumulator,totals

    @staticmethod
    def _record_gradient(accumulator, totals, indices):
        import optax

        totals['shared_gradient_norm'] = float(optax.global_norm(accumulator['shared']))
        totals['private_gradient_norm'] = float(optax.global_norm(accumulator['private']))
        if not all(np.isfinite(x) for x in totals.values()) or min(totals['shared_gradient_norm'],totals['private_gradient_norm']) <= 0.:
            raise ValueError('SCALE20K_EACH_DOMAIN_REQUIRES_FINITE_EFFECTIVE_SHARED_AND_PRIVATE_GRADIENT')
        totals['visit_slice_sha256'] = hashlib.sha256(indices.tobytes()).hexdigest()

    def update(self, guard):
        import jax

        before = time.monotonic()
        cursors(self.arm,self.step+1)
        gradient,metrics = {},{}
        for domain in DOMAINS:
            if self.batch_size[domain]:
                mean,metrics[domain] = self.domain_gradient(domain,guard)
                weight = DOMAIN_WEIGHT[domain]
                if domain == 'target':
                    gradient['shared'] = mean['shared']
                else:
                    gradient['shared'] = self.add(gradient['shared'],mean['shared'],weight)
                gradient[domain] = jax.tree.map(lambda leaf:leaf*weight,mean['private'])
                del mean
        guard()
        lr = float(self.rates[self.step])
        if self.profile_timing:
            jax.block_until_ready(gradient)
            adam_started = time.monotonic()
        parameters,states,norm = self.apply(self.parameters,self.states,gradient,lr)
        jax.block_until_ready((parameters,states))
        if self.profile_timing:
            adam_seconds = time.monotonic()-adam_started
        self.parameters,self.states = parameters,states
        self.step += 1
        row = dict(step=self.step,cursors=cursors(self.arm,self.step),learning_rate=lr,grad_norm=float(norm),
                   domains=metrics,actual_GPU_memory_stats=jax.devices()[0].memory_stats(),wall_s=time.monotonic()-before)
        if self.profile_timing:
            row['optimizer_s'] = adam_seconds
        return row
