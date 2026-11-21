"""One H200 update. The serial reference and batched candidates share model inputs."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract, batch_v2, hopper_serial_reference as serial
from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, make_adamw
from go2arx5_vla.sla_adapter.scale20k_training import compile_fm_accumulation, prepare_host_batch, target_auxiliary_context
from scripts.sla.hopper_inputs import BATCH, STEPS


class UpdateInputs:
    """Prepare two full logical batches in order, serializing each mutable reader."""
    def __init__(self, engine):
        self.engine=engine
        self.pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='sla-input')
        self.locks={d:Lock() for d in DOMAINS}
        self.queue={}

    def _prepare(self, phase, domain):
        indices,absolute=self.engine.indices(phase,domain)
        with self.locks[domain]:
            return prepare_host_batch(self.engine.datasets[domain],self.engine.tokenizer,domain,
                                      range(absolute,absolute+len(indices)),indices)

    def select(self, phase, stop):
        for step in range(phase,min(stop,phase+2)):
            if step not in self.queue:
                self.queue[step]={d:self.pool.submit(self._prepare,step,d) for d in DOMAINS}
        return self.queue.pop(phase)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.queue.clear()


class HopperEngine(serial.Engine):
    def __init__(self, restored, *, phase, segment_stop, fm_micro, aux_micro, target_origin=0):
        import jax

        graphs,self.frozen,state,self.banks,self.datasets,self.tokenizer,self.proof=restored
        if (not 0<=target_origin<=phase<segment_stop<=STEPS or fm_micro not in (1,4,8,16)
                or aux_micro not in (1,2,4)):
            raise ValueError('Invalid H200 stage or microbatch')
        self.parameters,self.states,self.rng=state['parameters'],state['optimizer'],state['rng']
        self.step,self.segment_stop,self.target_origin=phase,segment_stop,target_origin
        self.batch_size=BATCH
        self.microbatch=dict.fromkeys(DOMAINS,fm_micro)
        self.aux_micro=aux_micro
        self._setup_gradient_functions(graphs,BATCH)
        _,self.apply=make_adamw(**OPTIMIZER)
        self.combine=jax.jit(lambda fm,aux:adapt5k_contract.joint_gradient(fm,aux,tree_map=jax.tree.map))
        self.prefetch=None
        self.fused={}
        self.aux_add=None

    def enable_pipeline(self):
        self.prefetch=UpdateInputs(self)
        self.fused={d:compile_fm_accumulation(self.grad[d]) for d in DOMAINS}
        self.aux_add=compile_fm_accumulation(self.aux)

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()

    def indices(self, phase, domain):
        n=BATCH[domain]
        absolute=self.proof['ancestor']['lineage_cursors'][domain]+phase*n
        begin=(phase-(self.target_origin if domain=='target' else 0))*n
        ix=self.banks[domain][begin:begin+n]
        if len(ix)!=n:
            raise ValueError('Visit bank exhausted')
        return ix,absolute

    def _batched_fm(self, domain, host, indices, cursor, active, guard, audit):
        import jax
        import jax.numpy as jnp

        width=self.microbatch[domain]
        size=len(indices)
        noise,tau=self.draw[domain](self.rng,cursor)
        accumulated=jax.tree.map(jnp.zeros_like,active)
        values=[]
        for start in range(0,size,width):
            guard()
            stop=start+width
            obs,actions=batch_v2.device_batch(jax.tree.map(lambda a:a[start:stop],host))
            args=(obs,actions,noise[start:stop],tau[start:stop])
            if audit:
                audit.batch(domain,args)
            loss,accumulated=self.fused[domain](accumulated,width/size,active,self.frozen,*args)
            self.gradient_windows[domain]+=width
            values.append(loss)
            if len(values)>2:
                jax.block_until_ready(values[-3])
        mean=sum(float(x)*width/size for x in jax.device_get(values))
        return accumulated,dict(FM32=mean,auxiliary=0.,training_full_generations=0,backward_calls=len(values))

    def _batched_aux(self, host, slots, indices, cursor, active, guard, audit):
        import jax
        import jax.numpy as jnp

        total=jax.tree.map(jnp.zeros_like,active)
        values=[]
        for start in range(0,len(slots),self.aux_micro):
            guard()
            group=list(slots[start:start+self.aux_micro])
            obs,_=batch_v2.device_batch(jax.tree.map(lambda a:a[group],host))
            contexts=[target_auxiliary_context(self.datasets['target'],self.rng,cursor+s,int(indices[s]))
                      for s in group]
            args=[jnp.concatenate(column,axis=0) for column in zip(*contexts,strict=True)]
            if audit:
                audit.batch('aux',(obs,*args))
            self.attempted_aux+=len(group)
            value,total=self.aux_add(total,len(group)/4,active,self.frozen,obs,*args)
            jax.block_until_ready(value)
            self.completed_aux+=len(group)
            values.append(float(value)*len(group)/4)
        return total,sum(values)

    @staticmethod
    def observed_call(kernel, audit, name):
        def apply(active,frozen,*args):
            audit.batch(name,args)
            return kernel(active,frozen,*args)
        return apply

    def update(self, guard, *, audit=None):
        import jax
        import jax.numpy as jnp

        guard()
        if self.step>=self.segment_stop:
            raise ValueError('End of current capsule')
        begin=time.monotonic()
        start=self.step
        queued=self.prefetch.select(start,self.segment_stop) if self.prefetch else None
        fm,metrics={},{}
        # Continue the parent's 16-step rotation in the ADAPT phase clock.
        slots=adapt5k_contract.auxiliary_slots((self.proof['ancestor']['phase_step']+start)%16)
        for domain in DOMAINS:
            indices,cursor=self.indices(start,domain)
            active=dict(shared=self.parameters['shared'],private=self.parameters[domain])
            fm_started=time.monotonic()
            if queued is None:
                original=self.grad[domain]
                if audit:
                    self.grad[domain]=self.observed_call(original,audit,domain)
                try:
                    fm[domain],stats=self._fm_gradient(domain,indices,cursor,active,guard)
                finally:
                    self.grad[domain]=original
            else:
                before=time.monotonic()
                host=queued[domain].result()
                waited=time.monotonic()-before
                fm[domain],stats=self._batched_fm(domain,host,indices,cursor,active,guard,audit)
                stats['data_wait_s']=waited
            self._record_gradient(fm[domain],stats,indices)
            stats['fm_path_wall_s']=time.monotonic()-fm_started
            if audit:
                audit.component(domain,stats['FM32'],fm[domain])
            if domain=='target':
                aux_started=time.monotonic()
                if queued is None:
                    original=self.aux
                    if audit:
                        self.aux=self.observed_call(original,audit,'aux')
                    try:
                        aux=self._add_target_auxiliary(indices,cursor,active,jax.tree.map(jnp.zeros_like,active),stats,slots,guard)
                    finally:
                        self.aux=original
                else:
                    aux,stats['auxiliary']=self._batched_aux(host,slots,indices,cursor,active,guard,audit)
                    stats['training_full_generations']=len(slots)
                if not np.isfinite(stats['auxiliary']):
                    raise ValueError('Nonfinite target auxiliary')
                if audit:
                    audit.component('aux',stats['auxiliary'],aux)
                stats['auxiliary_path_wall_s']=time.monotonic()-aux_started
                stats['auxiliary_slots']=slots
            metrics[domain]=stats
        guard()
        joint=self.combine(fm,aux)
        if audit:
            audit.component('total',None,joint)
            if not audit.finish():
                raise ArithmeticError('Candidate first-step numerical or input gate failed before Adam')
        rate=float(self.banks['rates'][start])
        adam_started=time.monotonic()
        params,states,norm=self.apply(self.parameters,self.states,joint,rate)
        submitted=time.monotonic()
        jax.block_until_ready((params,states))
        synchronized=time.monotonic()
        self.parameters,self.states=params,states
        self.step+=1
        return dict(phase_step=self.step,ancestor_updates=self.proof['ancestor']['lineage_total_updates'],
                    phase_cursors={d:self.step*BATCH[d] for d in DOMAINS},learning_rate=rate,grad_norm=norm,
                    domains=metrics,wall_s=synchronized-begin,full_H40=4,
                    adam_call_s=submitted-adam_started,final_sync_s=synchronized-submitted,
                    memory=jax.local_devices()[0].memory_stats())
