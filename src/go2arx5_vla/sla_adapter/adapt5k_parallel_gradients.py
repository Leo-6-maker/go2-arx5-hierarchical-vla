"""Replicate the ADAPT gradient kernels on two local devices, with one reduction."""
from contextlib import closing
import hashlib
import time

import numpy as np

from . import adapt5k_contract as contract
from .scale20k_training import prepare_host_batch, prefetch_host_batches, target_auxiliary_context


class ParallelGradients:
    """Accumulate weighted local gradients until the joint update boundary."""

    def __init__(self, devices, fm, auxiliary):
        from functools import partial

        import jax
        import jax.numpy as jnp
        import optax

        self.devices = tuple(devices)
        if (len(self.devices)!=2 or len(set(self.devices))!=2
                or any(device not in jax.local_devices() for device in self.devices)
                or set(fm)!=set(contract.DOMAINS)):
            raise ValueError('ADAPT_DATA_PARALLEL_REQUIRES_TWO_DISTINCT_LOCAL_DEVICES_AND_ALL_DOMAINS')
        self.zeros = jax.pmap(lambda tree:jax.tree.map(jnp.zeros_like,tree),devices=self.devices)
        self.partition_norms = jax.pmap(lambda tree:(optax.global_norm(tree['shared']),
                                                    optax.global_norm(tree['private'])),devices=self.devices)

        def accumulate(backward, total, scale, *args):
            loss,grad = backward(*args)
            updated = jax.tree.map(lambda a,b:a+b*scale,total,grad)
            return loss,updated

        def bind(backward, arity):
            return jax.pmap(partial(accumulate,backward),devices=self.devices,
                            in_axes=(0,None)+(0,)*arity,donate_argnums=(0,))

        self.fm = {domain:bind(backward,6) for domain,backward in fm.items()}
        self.auxiliary = bind(auxiliary,6)

        def combine(fm_terms,auxiliary_term):
            local = contract.joint_gradient(fm_terms,auxiliary_term,tree_map=jax.tree.map)
            return jax.tree.map(lambda x:jax.lax.psum(x,'data'),local)

        self.reduce_joint = jax.pmap(combine,axis_name='data',devices=self.devices)

    def replicate(self, value):
        import jax

        return jax.device_put_replicated(value,self.devices)

    def shard(self, left, right):
        import jax

        return jax.device_put_sharded([left,right],self.devices)

    def to_optimizer_device(self, gradient):
        """Use the summed gradient on device zero for one optimizer invocation."""
        import jax

        def select(value):
            shard = next(part for part in value.addressable_shards if part.device==self.devices[0])
            return shard.data[0]

        return jax.tree.map(select,gradient)


class ParallelAdaptUpdate:
    """Execute one ADAPT update with the same visits and bounded prefetch on two replicas."""

    def __init__(self, engine, devices):
        if (engine.microbatch!=dict.fromkeys(contract.DOMAINS,2) or not engine.input_executor
                or not engine.defer_fm_loss):
            raise ValueError('ADAPT_TWO_DEVICES_REQUIRE_MATCHED_MB2_AND_BOUNDED_DEFERRED_PREFETCH')
        self.engine = engine
        self.kernels = ParallelGradients(devices,engine.grad,engine.aux)
        self.frozen = self.kernels.replicate(engine.frozen)

    @staticmethod
    def _observation(packed):
        from openpi.models.model import Observation

        return Observation.from_dict(packed)

    def _norms(self, gradient):
        result = np.asarray(self.kernels.partition_norms(gradient))
        if not np.isfinite(result).all() or (result<=0.).any():
            raise ValueError('ADAPT_TWO_DEVICES_REQUIRE_FINITE_EFFECTIVE_LOCAL_GRADIENTS')
        return result.tolist()

    def _fm(self, domain, cursor, indices, active, guard):
        import jax

        engine,kernels = self.engine,self.kernels
        dataset = engine.datasets[domain]
        size = len(indices)
        half = size//2
        draws = engine.draw[domain](engine.rng,cursor)
        accumulator = kernels.zeros(active)

        def prepare(offset):
            return tuple(prepare_host_batch(dataset,engine.tokenizer,domain,range(cursor+k,cursor+k+2),indices[k:k+2])
                         for k in (offset,offset+half))

        waited = 0.
        values = []
        with closing(prefetch_host_batches(engine.input_executor,range(0,half,2),prepare)) as batches:
            for offset in range(0,half,2):
                guard()
                before = time.monotonic()
                left,right = next(batches)
                observations = kernels.shard(self._observation(left['observation']),self._observation(right['observation']))
                action = kernels.shard(left['actions'],right['actions'])
                noise,flow = (kernels.shard(x[offset:offset+2],x[offset+half:offset+half+2]) for x in draws)
                waited += time.monotonic()-before
                loss,accumulator = kernels.fm[domain](accumulator,2/size,active,self.frozen,observations,action,noise,flow)
                values.append(loss)
                if len(values)>2:
                    jax.block_until_ready(values[-3])
                engine.gradient_windows[domain] += 4
        fm_loss = sum(float(value) for pair in jax.device_get(values) for value in pair)*(2/size)
        if not np.isfinite(fm_loss):
            raise ValueError('ADAPT_TWO_DEVICES_FINITE_FM_LOSS_REQUIRED')
        metrics = dict(FM32=fm_loss,auxiliary=0.,backward_calls=size//2,parallel_FM_launches=size//4,
            window_count_each_device=[half,half],training_full_generations=0,data_wait_s=waited,
            visit_slice_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),local_partition_norms=self._norms(accumulator))
        return accumulator,metrics

    def _auxiliary(self, cursor, indices, active, metrics, guard):
        import jax

        engine,kernels = self.engine,self.kernels
        dataset = engine.datasets['target']
        slots = contract.auxiliary_slots(engine.step)
        total = kernels.zeros(active)
        values = []
        for local in range(2):
            guard()
            pair = []
            for slot in (slots[local],slots[local+2]):
                absolute,index = cursor+slot,int(indices[slot])
                batch = prepare_host_batch(dataset,engine.tokenizer,'target',[absolute],[index])
                pair.append((self._observation(batch['observation']),*target_auxiliary_context(dataset,engine.rng,absolute,index)))
            inputs = kernels.shard(*pair)
            engine.attempted_aux += 2
            loss,total = kernels.auxiliary(total,.25,active,self.frozen,*inputs)
            jax.block_until_ready(loss)
            engine.completed_aux += 2
            values.append(loss)
        auxiliary_loss = sum(float(value) for pair in jax.device_get(values) for value in pair)*.25
        if not np.isfinite(auxiliary_loss):
            raise ValueError('ADAPT_TWO_DEVICES_FINITE_AUXILIARY_LOSS_REQUIRED')
        metrics.update(auxiliary=auxiliary_loss,training_full_generations=4,auxiliary_slots=slots,
            auxiliaries_each_device=[2,2],local_auxiliary_partition_norms=self._norms(total))
        return total

    def update(self, guard):
        import jax

        engine,kernels = self.engine,self.kernels
        if engine.step>=engine.stage_stop:
            raise ValueError('ADAPT5K_REQUIRES_COLD_HANDOFF_AT_THE_STAGE_BOUNDARY')
        beginning = time.monotonic()
        progress = contract.progress(engine.parent_step,engine.step,engine.stage_start)
        shared = kernels.replicate(engine.parameters['shared'])
        terms,metrics = {},{}
        for domain in contract.DOMAINS:
            offset = progress['stage_local_cursors'][domain]
            cursor = progress['lineage_cursors'][domain]
            indices = engine.banks[domain][offset:offset+engine.batch_size[domain]]
            active = dict(shared=shared,private=kernels.replicate(engine.parameters[domain]))
            terms[domain],metrics[domain] = self._fm(domain,cursor,indices,active,guard)
            if domain=='target':
                auxiliary = self._auxiliary(cursor,indices,active,metrics[domain],guard)
        guard()
        gradient = kernels.to_optimizer_device(kernels.reduce_joint(terms,auxiliary))
        lr = float(engine.rates[engine.step])
        parameters,states,norm = engine.apply(engine.parameters,engine.states,gradient,lr)
        jax.block_until_ready((parameters,states))
        engine.parameters,engine.states = parameters,states
        engine.step += 1
        return dict(**contract.progress(engine.parent_step,engine.step,engine.stage_start),learning_rate=lr,
            grad_norm=norm,domains=metrics,wall_s=time.monotonic()-beginning,devices=2,
            actual_GPU_memory_stats=kernels.devices[0].memory_stats(),
            per_device_memory_stats=[device.memory_stats() for device in kernels.devices])
